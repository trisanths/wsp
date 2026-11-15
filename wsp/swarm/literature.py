"""Exa literature priors for the swarm (spec 9.3; docs/research/exa.md sections 1, 3 and 5).

ExaLiterature(config, mode) implements the Literature interface in
wsp/swarm/interfaces.py. Exa is a read-only source of priors: it never
touches the eval or compute budget, and it never sees held-out data. Every
Exa call in part 3 goes through this module, so the cache, the cost log and
the fairness rules live in one place.

Modes:
- "off": no calls. priors() is an empty pack with sha "", search() returns [].
- "replay": cache only, never the Exa network. A miss returns [] and records
  a "replay_miss" call. Graded and demo runs use this mode.
- "live": read through the cache. On a miss, call Exa and write the cache.
  Without EXA_API_KEY, or without the exa-py package, it degrades to replay
  and records why. A 401 or 402 from Exa does the same for the rest of the run.

Cache:
- Key: sha256 of the canonical JSON {endpoint, sdk_version, params}, where
  params are the exact keyword arguments sent to the SDK method, with None
  values and volatile fields (user_identifier, timeout, extra headers) removed.
- Tier one: the git-tracked folder exa_cache/<key[:2]>/<key>.json. It is the
  source of truth for reproducible runs and ships with the repo.
- Tier two (optional): a modal.Dict named "exa-cache", when config
  exa_modal_cache is true, so parallel containers share hits. modal is
  imported lazily and every Dict error is logged and turns tier two off.
- A cache value is {"key", "endpoint", "sdk_version", "params", "response",
  "cost_dollars", "fetched_at"}, where response is the SDK SearchResponse as
  plain JSON (dataclasses.asdict).

Priors pack: priors() reads the frozen pack file and reports its sha256
computed on the bytes with CRLF read as LF (text_sha256), so a Windows
checkout with core.autocrlf and a Linux checkout give the same sha.

Calls log: every search appends one row to self.calls with the keys
endpoint, query, cache_hit, cost_dollars, mode, status and role (plus key,
tier, note and cost_estimate when they apply). status is one of hit, live,
replay_miss, error, capped, degraded, priors_missing, priors_error.
drain_calls() returns the rows and clears the list, so the swarm loop can log
them as event rows. cost_dollars is what Exa reported (0.0 for a cache hit,
None when Exa reported nothing); cost_estimate is a list-price estimate.

Config keys read (config.get(key, default)):
    priors_path             "data/priors.json"  the frozen pack from scripts/build_priors.py
    exa_cache_dir           "exa_cache"         tier one folder
                            (relative paths in both are relative to the repo root)
    exa_modal_cache         False               also use the modal.Dict tier
    exa_modal_dict          "exa-cache"         the modal.Dict name
    exa_search_type         "lite"              "lite" or "auto" for search()
    exa_end_published_date  DEFAULT_END_DATE    pinned so cache keys stay stable; None sends no date
    exa_exclude_domains     []                  domains excluded on top of DATASET_MIRRORS
    exa_quote_chars         400                 longest quote kept per citation
    exa_max_query_chars     400                 longest query sent (a leak bound)
    exa_max_live_calls      100                 live Exa calls allowed per instance
    exa_retries             2                   retries on 429, 5xx and network errors
    exa_backoff_s           2.0                 first retry wait, doubled each retry
    exa_timeout_s           30.0                per-request timeout given to the SDK
    exa_sdk_version         None                cache key version; None uses the installed exa-py

The SDK names used here were checked with inspect against exa-py 2.28.0:
Exa(api_key=None, base_url, user_agent, timeout) reads EXA_API_KEY;
Exa.search(query, *, contents, num_results, exclude_domains,
end_published_date, type, category, system_prompt, output_schema, ...)
returns a SearchResponse dataclass (results, resolved_search_type, auto_date,
context, output, statuses, cost_dollars, search_time, warnings, notices);
output is DeepSearchOutput(content, grounding); errors are ValueError
("Request failed with status code N: ...") or ExaTimeoutError.
"""

import copy
import dataclasses
import hashlib
import importlib.metadata
import json
import logging
import os
import re
import threading
import time
from urllib.parse import urlparse

try:
    from wsp.swarm.interfaces import Literature
except ImportError:  # the core interfaces module is not built yet
    Literature = object

logger = logging.getLogger("wsp.swarm.literature")

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

MODES = ["off", "replay", "live"]
SEARCH_TYPES = ["lite", "auto"]

# Results published after this date are ignored, so a cached query keeps the
# same key on every day. Change it only when rebuilding the priors pack.
DEFAULT_END_DATE = "2026-11-14T00:00:00Z"

# Dataset mirrors and benchmark repos (exa.md 5.3 rule 4). Benchmark items
# must never reach a planner. A plain host also matches its subdomains here.
DATASET_MIRRORS = [
    "huggingface.co/datasets",
    "kaggle.com",
    "*.kaggle.com",
    "paperswithcode.com/dataset",
    "tensorflow.org/datasets",
    "github.com/openai/grade-school-math",
    "github.com/TIGER-AI-Lab/MMLU-Pro",
    "github.com/google-deepmind/bbeh",
]

# Fields that do not change the result set, left out of the cache key.
VOLATILE_PARAMS = ["user_identifier", "timeout", "extra_headers", "extra_query", "stream", "betas"]

# List prices in dollars from exa.md section 4 (checked 2026-11-14). UNVERIFIED:
# whether highlights are billed on top of the search price, the extra cost of
# output_schema, and find_similar pricing (assumed to be the auto rate).
SEARCH_PRICES = {"lite": 0.004, "auto": 0.007, "fast": 0.007, "deep-lite": 0.012, "deep": 0.012}
PRICE_PER_RESULT_ABOVE_10 = 0.001
PRICE_PER_PAGE_PER_CONTENT = 0.001
CONTENT_TYPES = ["text", "highlights", "summary"]


# Small helpers.

def installed_sdk_version():
    """The installed exa-py version, or "not-installed"."""
    try:
        return importlib.metadata.version("exa-py")
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def resolve_path(path):
    """Relative paths are relative to the repo root."""
    if os.path.isabs(path):
        return path
    return os.path.join(REPO_ROOT, path)


def now_iso():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def canonical_json(value):
    """Sorted keys, no whitespace: the same value always gives the same text."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, default=str)


def clean_params(params):
    """A copy of params without None values and without volatile fields."""
    out = {}
    for name in params:
        if name in VOLATILE_PARAMS:
            continue
        if params[name] is None:
            continue
        out[name] = copy.deepcopy(params[name])
    return out


def cache_key(endpoint, params, sdk_version):
    """sha256 hex of canonical JSON {endpoint, sdk_version, params}."""
    body = {"endpoint": endpoint, "sdk_version": sdk_version, "params": clean_params(params)}
    return hashlib.sha256(canonical_json(body).encode("utf-8")).hexdigest()


def cache_file(cache_dir, key):
    return os.path.join(cache_dir, key[:2], key + ".json")


def text_sha256(raw):
    """sha256 hex of a text file's bytes, with CRLF line endings read as LF.

    git on Windows (core.autocrlf) can check a text file out with CRLF
    endings, and a Linux checkout gets LF. Hashing the LF form gives the same
    sha on every checkout. JSON writers escape a carriage return inside a
    string as \\r, so the pack and catalog files never hold a raw CR that
    this would change.
    """
    return hashlib.sha256(raw.replace(b"\r\n", b"\n")).hexdigest()


def clean_text(text, max_chars):
    """One line of plain text, at most max_chars long."""
    if text is None:
        return ""
    words = str(text).split()
    line = " ".join(words)
    if len(line) > max_chars:
        line = line[:max_chars].rstrip() + "..."
    return line


def printable(value, max_chars):
    """clean_text for values whose str() might fail (used only when logging an error)."""
    try:
        return clean_text(value, max_chars)
    except Exception:
        return "<unprintable>"


def normalize_url(url):
    return str(url).strip().rstrip("/")


def host_and_path(url):
    """(host without www., path) of a URL, both lower case."""
    try:
        parsed = urlparse(str(url).strip())
    except ValueError:
        return "", ""
    host = (parsed.hostname or "").lower()
    if host.startswith("www."):
        host = host[len("www."):]
    return host, (parsed.path or "").lower()


def url_matches(url, pattern):
    """True when url falls under a domain pattern: host, *.host, or host/path-prefix."""
    host, path = host_and_path(url)
    if host == "":
        return False
    pattern = pattern.strip().lower()
    if pattern.startswith("*."):
        return host.endswith(pattern[1:])
    pattern_path = ""
    if "/" in pattern:
        slash = pattern.index("/")
        pattern_path = pattern[slash:]
        pattern = pattern[:slash]
    if pattern.startswith("www."):
        pattern = pattern[len("www."):]
    if host != pattern and not host.endswith("." + pattern):
        return False
    if pattern_path == "":
        return True
    return path == pattern_path or path.startswith(pattern_path.rstrip("/") + "/")


def is_excluded(url, patterns):
    for pattern in patterns:
        if url_matches(url, pattern):
            return True
    return False


def status_code_of(error):
    """The HTTP status in an exa-py error message ("... status code 402: ..."), or None."""
    match = re.search(r"status code (\d+)", str(error))
    if match is None:
        return None
    return int(match.group(1))


def short_error(error):
    return type(error).__name__ + ": " + clean_text(str(error), 200)


def response_to_dict(response):
    """The SDK response as plain JSON data (dicts, lists, strings, numbers)."""
    if dataclasses.is_dataclass(response):
        data = dataclasses.asdict(response)
    elif isinstance(response, dict):
        data = response
    else:
        data = {"results": []}
    return json.loads(json.dumps(data, default=str))


def reported_cost(data):
    """costDollars.total from a response dict, or None when Exa reported nothing."""
    cost = data.get("cost_dollars")
    if isinstance(cost, dict):
        cost = cost.get("total")
    if isinstance(cost, (int, float)):
        return float(cost)
    return None


def estimate_cost(endpoint, params):
    """List-price estimate in dollars for one call (exa.md section 4). An estimate, never a bill."""
    search_type = params.get("type") or "auto"
    if endpoint != "search":
        search_type = "auto"
    base = SEARCH_PRICES.get(search_type, SEARCH_PRICES["auto"])
    results = params.get("num_results") or 10
    extra = 0.0
    if results > 10:
        extra = (results - 10) * PRICE_PER_RESULT_ABOVE_10
    contents = params.get("contents") or {}
    content_count = 0
    for name in CONTENT_TYPES:
        if contents.get(name):
            content_count += 1
    return base + extra + results * content_count * PRICE_PER_PAGE_PER_CONTENT


def quote_of(result, max_chars):
    """The first highlight of a result, else its summary, else its text, as one short line."""
    highlights = result.get("highlights")
    if isinstance(highlights, list):
        for item in highlights:
            if isinstance(item, str) and item.strip() != "":
                return clean_text(item, max_chars)
    summary = result.get("summary")
    if isinstance(summary, str) and summary.strip() != "":
        return clean_text(summary, max_chars)
    return clean_text(result.get("text"), max_chars)


def cites_from_response(data, k, exclude_domains, quote_chars=400):
    """Up to k citations {"url", "title", "quote"} from a response dict, mirrors dropped."""
    cites = []
    results = data.get("results") or []
    for result in results:
        if len(cites) >= k:
            break
        if not isinstance(result, dict):
            continue
        url = result.get("url") or ""
        if url == "" or is_excluded(url, exclude_domains):
            continue
        cites.append({"url": url, "title": clean_text(result.get("title"), 300),
                      "quote": quote_of(result, quote_chars)})
    return cites


def claim_index(field):
    """The claim number in a grounding field such as "claims[2].claim" or "claims.2", or None.

    A field of just "claims" returns -1: its citations back every claim.
    """
    if field == "claims":
        return -1
    match = re.match(r"^claims(?:\[|\.)(\d+)", field)
    if match is None:
        return None
    return int(match.group(1))


def grounded_cites(grounding, index, quotes, exclude_domains):
    """Citations that Exa's grounding attaches to claim number index."""
    cites = []
    seen = set()
    for row in grounding:
        if not isinstance(row, dict):
            continue
        where = claim_index(str(row.get("field", "")))
        if where is None or (where != index and where != -1):
            continue
        for citation in row.get("citations") or []:
            if not isinstance(citation, dict):
                continue
            url = citation.get("url") or ""
            if url == "" or url in seen or is_excluded(url, exclude_domains):
                continue
            seen.add(url)
            cites.append({"url": url, "title": clean_text(citation.get("title"), 300),
                          "quote": quotes.get(url, "")})
    return cites


def claims_from_response(data, topic, exclude_domains, quote_chars=400):
    """Claims from a deep-lite response with an output_schema of claims.

    Each claim is {"topic", "claim", "layer_hint", "model", "strength",
    "cites", "grounded"}. cites are the sources Exa's grounding names for
    that claim (grounded True). When Exa names none, cites are the top 3
    results of the search and grounded is False. Returns [] when the output
    is missing or does not parse.
    """
    output = data.get("output")
    if not isinstance(output, dict):
        return []
    content = output.get("content")
    if isinstance(content, str):
        try:
            content = json.loads(content)
        except ValueError:
            return []
    if not isinstance(content, dict):
        return []
    raw_claims = content.get("claims")
    if not isinstance(raw_claims, list):
        return []
    quotes = {}
    for result in data.get("results") or []:
        if isinstance(result, dict) and result.get("url"):
            quotes[result["url"]] = quote_of(result, quote_chars)
    fallback = cites_from_response(data, 3, exclude_domains, quote_chars)
    grounding = output.get("grounding") or []
    claims = []
    for index in range(len(raw_claims)):
        raw = raw_claims[index]
        if not isinstance(raw, dict) or not raw.get("claim"):
            continue
        cites = grounded_cites(grounding, index, quotes, exclude_domains)
        grounded = len(cites) > 0
        if not grounded:
            cites = copy.deepcopy(fallback)
        strength = str(raw.get("strength", "")).lower()
        if strength not in ("weak", "medium", "strong"):
            strength = "weak"
        claims.append({"topic": topic, "claim": clean_text(raw.get("claim"), 600),
                       "layer_hint": clean_text(raw.get("layer_hint"), 200),
                       "model": clean_text(raw.get("model"), 200), "strength": strength,
                       "cites": cites, "grounded": grounded})
    return claims


def urls_in_response(data):
    """Every URL a response dict names: results and grounding citations."""
    urls = []
    for result in data.get("results") or []:
        if isinstance(result, dict) and result.get("url"):
            urls.append(result["url"])
    output = data.get("output")
    if isinstance(output, dict):
        for row in output.get("grounding") or []:
            if not isinstance(row, dict):
                continue
            for citation in row.get("citations") or []:
                if isinstance(citation, dict) and citation.get("url"):
                    urls.append(citation["url"])
    return urls


def empty_pack():
    return {"sha": "", "components": {}, "topics": []}


def exclude_list(extra=None):
    """DATASET_MIRRORS plus extra domains, without repeats, sorted (so cache keys stay stable)."""
    names = []
    for pattern in DATASET_MIRRORS + list(extra or []):
        if pattern not in names:
            names.append(pattern)
    return sorted(names)


def make_exa_client(timeout_s):
    """An exa_py.Exa client. It reads EXA_API_KEY itself, so this module never holds the key."""
    from exa_py import Exa

    return Exa(timeout=timeout_s)


def open_modal_dict(name):
    """The modal.Dict cache tier. Modal auth comes from MODAL_TOKEN_ID and MODAL_TOKEN_SECRET."""
    import modal

    return modal.Dict.from_name(name, create_if_missing=True)


class ExaLiterature(Literature):
    """The Literature interface backed by Exa search and a two-tier cache. See the module docstring."""

    def __init__(self, config, mode):
        if config is None:
            config = {}
        if mode not in MODES:
            raise ValueError("unknown literature mode " + repr(mode) + "; choose off, replay or live")
        search_type = config.get("exa_search_type", "lite")
        if search_type not in SEARCH_TYPES:
            raise ValueError("exa_search_type must be lite or auto, not " + repr(search_type))
        self.config = config
        self.requested_mode = mode
        self.mode = mode
        self.degraded_reason = None
        self.search_type = search_type
        self.priors_path = resolve_path(config.get("priors_path", "data/priors.json"))
        self.cache_dir = resolve_path(config.get("exa_cache_dir", "exa_cache"))
        self.end_date = config.get("exa_end_published_date", DEFAULT_END_DATE)
        self.exclude_domains = exclude_list(config.get("exa_exclude_domains", []))
        self.quote_chars = int(config.get("exa_quote_chars", 400))
        self.max_query_chars = int(config.get("exa_max_query_chars", 400))
        self.max_live_calls = int(config.get("exa_max_live_calls", 100))
        self.retries = int(config.get("exa_retries", 2))
        self.backoff_s = float(config.get("exa_backoff_s", 2.0))
        self.timeout_s = float(config.get("exa_timeout_s", 30.0))
        self.sdk_version = config.get("exa_sdk_version") or installed_sdk_version()
        self.lock = threading.Lock()
        self.calls = []
        self.live_calls = 0
        self.client = None
        self.modal_dict = None
        self.pack = None
        self.urls = set()
        self.url_index_built = False
        if mode == "live":
            self.start_live()
        if self.mode != "off" and config.get("exa_modal_cache", False):
            self.start_modal_tier(config.get("exa_modal_dict", "exa-cache"))

    # Setup.

    def start_live(self):
        if not os.environ.get("EXA_API_KEY"):
            self.degrade("live mode needs EXA_API_KEY in the environment; it is not set")
            return
        try:
            self.client = make_exa_client(self.timeout_s)
        except ImportError:
            self.degrade("live mode needs the exa-py package, which is not installed")
        except Exception as error:
            self.degrade("could not build the Exa client (" + short_error(error) + ")")

    def degrade(self, reason):
        """Switch to replay for the rest of this instance's life and record why."""
        self.mode = "replay"
        self.client = None
        self.degraded_reason = reason
        logger.warning("Exa literature degraded to replay: %s", reason)
        self.record("init", "", False, 0.0, "degraded", note=reason)

    def start_modal_tier(self, name):
        try:
            self.modal_dict = open_modal_dict(name)
        except Exception as error:
            logger.warning("Exa modal cache tier is off: %s", short_error(error))
            self.modal_dict = None

    def drop_modal_tier(self, error):
        logger.warning("Exa modal cache tier failed and is now off: %s", short_error(error))
        self.modal_dict = None

    # The calls log.

    def record(self, endpoint, query, cache_hit, cost, status, role="", key="", tier="", note="",
               cost_estimate=None):
        row = {"endpoint": endpoint, "query": query, "cache_hit": cache_hit, "cost_dollars": cost,
               "mode": self.mode, "status": status, "role": role, "key": key}
        if tier != "":
            row["tier"] = tier
        if note != "":
            row["note"] = note
        if cost_estimate is not None:
            row["cost_estimate"] = cost_estimate
        with self.lock:
            self.calls.append(row)

    def drain_calls(self):
        """Return the call rows recorded since the last drain, and clear them."""
        with self.lock:
            rows = self.calls
            self.calls = []
        return rows

    # The cache.

    def is_cached(self, key):
        """True when tier one holds key. Reads only the local folder."""
        return os.path.exists(cache_file(self.cache_dir, key))

    def read_file_entry(self, key):
        path = cache_file(self.cache_dir, key)
        if not os.path.exists(path):
            return None
        try:
            with open(path, encoding="utf-8") as handle:
                entry = json.load(handle)
        except (OSError, ValueError) as error:
            logger.warning("unreadable Exa cache file %s: %s", path, short_error(error))
            return None
        if not isinstance(entry, dict) or not isinstance(entry.get("response"), dict):
            logger.warning("Exa cache file %s has no response; treating it as a miss", path)
            return None
        return entry

    def read_modal_entry(self, key):
        if self.modal_dict is None:
            return None
        try:
            text = self.modal_dict.get(key)
        except Exception as error:
            self.drop_modal_tier(error)
            return None
        if text is None:
            return None
        try:
            entry = json.loads(text)
        except (TypeError, ValueError):
            return None
        if not isinstance(entry, dict) or not isinstance(entry.get("response"), dict):
            return None
        return entry

    def read_cache(self, key):
        """(entry, tier) for key, or (None, ""). Tier one first, then tier two."""
        entry = self.read_file_entry(key)
        if entry is not None:
            return entry, "file"
        entry = self.read_modal_entry(key)
        if entry is not None:
            if self.mode == "live":
                self.write_file_entry(entry)
            return entry, "modal"
        return None, ""

    def write_file_entry(self, entry):
        path = cache_file(self.cache_dir, entry["key"])
        folder = os.path.dirname(path)
        temp_path = path + "." + str(os.getpid()) + "." + str(threading.get_ident()) + ".tmp"
        try:
            os.makedirs(folder, exist_ok=True)
            with open(temp_path, "w", encoding="utf-8") as handle:
                handle.write(json.dumps(entry, sort_keys=True, indent=1) + "\n")
            os.replace(temp_path, path)
        except OSError as error:
            logger.warning("could not write Exa cache file %s: %s", path, short_error(error))

    def write_cache(self, entry):
        self.write_file_entry(entry)
        if self.modal_dict is not None:
            try:
                self.modal_dict.put(entry["key"], json.dumps(entry, sort_keys=True))
            except Exception as error:
                self.drop_modal_tier(error)

    # Fetching.

    def fetch(self, endpoint, params, role=""):
        """The response dict for one SDK call, from the cache or (live mode) from Exa, or None.

        endpoint is the exa_py method name ("search" or "find_similar") and
        params are its keyword arguments. Never raises.
        """
        if self.mode == "off":
            return None
        query = str(params.get("query", params.get("url", "")))
        key = cache_key(endpoint, params, self.sdk_version)
        entry, tier = self.read_cache(key)
        if entry is not None:
            self.add_urls(urls_in_response(entry["response"]))
            self.record(endpoint, query, True, 0.0, "hit", role=role, key=key, tier=tier)
            return entry["response"]
        if self.mode != "live":
            self.record(endpoint, query, False, 0.0, "replay_miss", role=role, key=key)
            return None
        return self.fetch_live(endpoint, params, key, query, role)

    def fetch_live(self, endpoint, params, key, query, role):
        estimate = estimate_cost(endpoint, params)
        with self.lock:
            allowed = self.live_calls < self.max_live_calls
            if allowed:
                self.live_calls += 1
        if not allowed:
            self.record(endpoint, query, False, 0.0, "capped", role=role, key=key,
                        note="exa_max_live_calls (" + str(self.max_live_calls) + ") reached")
            return None
        response, error = self.call_with_retries(endpoint, params)
        if response is None:
            self.record(endpoint, query, False, None, "error", role=role, key=key, note=error,
                        cost_estimate=estimate)
            return None
        try:
            data = response_to_dict(response)
        except (TypeError, ValueError) as parse_error:
            self.record(endpoint, query, False, None, "error", role=role, key=key,
                        note="unreadable response: " + short_error(parse_error), cost_estimate=estimate)
            return None
        cost = reported_cost(data)
        self.write_cache({"key": key, "endpoint": endpoint, "sdk_version": self.sdk_version,
                          "params": clean_params(params), "response": data, "cost_dollars": cost,
                          "fetched_at": now_iso()})
        self.add_urls(urls_in_response(data))
        self.record(endpoint, query, False, cost, "live", role=role, key=key, cost_estimate=estimate)
        return data

    def call_with_retries(self, endpoint, params):
        """(response, None) on success, (None, error text) on failure. Never raises.

        Retries 429, 5xx and network errors with doubling waits. A 401 or 402
        turns live mode off for the rest of the run. Other errors (400, bad
        options) are not retried.
        """
        client = self.client
        if client is None:
            return None, "no Exa client"
        method = getattr(client, endpoint, None)
        if method is None:
            return None, "the Exa client has no method " + endpoint
        attempts = self.retries + 1
        last_error = "no attempt made"
        for attempt in range(attempts):
            try:
                return method(**params), None
            except Exception as error:
                last_error = short_error(error)
                code = status_code_of(error)
                if code == 401 or code == 402:
                    self.degrade("Exa returned " + str(code) + "; no more live calls this run")
                    return None, last_error
                retryable = False
                if code is not None and (code == 429 or code >= 500):
                    retryable = True
                if code is None and not isinstance(error, ValueError):
                    retryable = True
                if not retryable or attempt == attempts - 1:
                    return None, last_error
                logger.warning("Exa %s failed (%s); retry %d", endpoint, last_error, attempt + 1)
                if self.backoff_s > 0:
                    time.sleep(self.backoff_s * (2 ** attempt))
        return None, last_error

    # The Literature interface.

    def search_params(self, query, k):
        """The exa_py search keyword arguments for one planner query, or None for an empty query."""
        words = str(query or "").split()
        text = " ".join(words)[:self.max_query_chars].strip()
        if text == "":
            return None
        count = 3
        if k is not None:
            count = int(k)
        if count < 1:
            count = 1
        if count > 10:
            count = 10
        params = {"query": text, "type": self.search_type, "category": "publication",
                  "num_results": count, "exclude_domains": list(self.exclude_domains),
                  "contents": {"highlights": True}}
        if self.end_date:
            params["end_published_date"] = self.end_date
        return params

    def search(self, role, query, k=3):
        """At most k citations {"url", "title", "quote"} for one hypothesis. Never raises; [] on failure.

        Quotes are untrusted page text: prompts must show them as quoted
        evidence, never as instructions. The query is cut to
        exa_max_query_chars and must never contain held-out text.
        """
        if self.mode == "off":
            return []
        try:
            params = self.search_params(query, k)
            if params is None:
                return []
            data = self.fetch("search", params, role)
            if data is None:
                return []
            return cites_from_response(data, params["num_results"], self.exclude_domains, self.quote_chars)
        except Exception as error:
            logger.warning("Exa search failed: %s", short_error(error))
            self.record("search", printable(query, 200), False, None, "error", role=role,
                        note=short_error(error))
            return []

    def priors(self):
        """The frozen priors pack {"sha", "components", "topics"}. Never calls the network.

        sha is text_sha256 of the pack file, so a CRLF checkout of the same
        pack gives the same sha as the LF file build_priors writes.
        """
        if self.mode == "off":
            return empty_pack()
        if self.pack is None:
            self.pack = self.load_pack()
        return copy.deepcopy(self.pack)

    def load_pack(self):
        path = self.priors_path
        if not os.path.exists(path):
            logger.warning("no priors pack at %s; using an empty pack", path)
            self.record("priors", "", False, 0.0, "priors_missing", note=path)
            return empty_pack()
        try:
            with open(path, "rb") as handle:
                raw = handle.read()
            data = json.loads(raw.decode("utf-8"))
        except (OSError, ValueError) as error:
            logger.warning("unreadable priors pack %s: %s", path, short_error(error))
            self.record("priors", "", False, 0.0, "priors_error", note=short_error(error))
            return empty_pack()
        components = data.get("components")
        topics = data.get("topics")
        if not isinstance(components, dict) or not isinstance(topics, list):
            self.record("priors", "", False, 0.0, "priors_error", note="pack needs components and topics")
            return empty_pack()
        pack_version = data.get("sdk_version")
        if pack_version is not None and pack_version != self.sdk_version:
            logger.warning("priors pack was built with exa-py %s but cache keys use %s; set exa_sdk_version "
                           "to match, or replay searches will miss", pack_version, self.sdk_version)
        pack = {"sha": text_sha256(raw), "components": components, "topics": topics}
        self.add_urls(urls_in_pack(pack))
        return pack

    def known_url(self, url):
        """True when url is in the priors pack or in the cache (tier one, plus tier-two hits seen)."""
        if self.mode == "off" or not url:
            return False
        self.build_url_index()
        with self.lock:
            return normalize_url(url) in self.urls

    def add_urls(self, urls):
        """Remember URLs for known_url. Dataset mirrors never count as known."""
        with self.lock:
            for url in urls:
                if is_excluded(url, self.exclude_domains):
                    continue
                self.urls.add(normalize_url(url))

    def build_url_index(self):
        """Read every tier-one cache file and the priors pack once, to collect their URLs."""
        if self.url_index_built:
            return
        self.url_index_built = True
        self.priors()
        if not os.path.isdir(self.cache_dir):
            return
        for folder in sorted(os.listdir(self.cache_dir)):
            folder_path = os.path.join(self.cache_dir, folder)
            if not os.path.isdir(folder_path):
                continue
            for name in sorted(os.listdir(folder_path)):
                if not name.endswith(".json"):
                    continue
                entry = self.read_file_entry(name[:-len(".json")])
                if entry is not None:
                    self.add_urls(urls_in_response(entry["response"]))


def urls_in_pack(pack):
    urls = []
    for part_id in pack["components"]:
        for cite in pack["components"][part_id] or []:
            if isinstance(cite, dict) and cite.get("url"):
                urls.append(cite["url"])
    for claim in pack["topics"]:
        if not isinstance(claim, dict):
            continue
        for cite in claim.get("cites") or []:
            if isinstance(cite, dict) and cite.get("url"):
                urls.append(cite["url"])
    return urls
