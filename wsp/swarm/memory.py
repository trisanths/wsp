"""past.dev memory for the swarm (spec 9.2; docs/research/pastdev.md sections 2, 4, 6 and 7).

PastMemory(config) implements the Memory interface of wsp/swarm/interfaces.py
with plain `requests` calls to the past.dev Memory API.

Run scoping (pastdev.md 6.1), the part that keeps benchmark runs fair:
- Each run gets one audience. Its slug comes from the run id (audience_slug):
  lowercase letters, digits and dashes, at most 40 characters, ending in a
  short hash of the full run id so two run ids never share a slug. The
  mapping is kept on the object (self.audience) and in the state file.
- Each role gets one identity, "<run>:<role>", for example "s0-swarm-1794:planner-1".
  The audience lists every identity of the run.
- start_run creates the audience with PUT before any ingest, recomputes it,
  and waits briefly until the guard identity reaches it. Then it recalls as
  one planner with a generic query and raises MemoryLeakError if anything
  comes back: a benchmark run must start cold.
- A run id is used once. Before any request, start_run raises MemoryLeakError
  when the state file shows this run id already sent data or ended: an
  uncleaned run left memory behind, and past.dev never accepts a deleted data
  point id or a used batch key again. The old state file is left untouched.
- If the audience already existed (the PUT answered 200) and the guard
  identity is not confirmed in it in time, the guard recall cannot see what
  the audience holds, so memory degrades rather than start unproven. A new
  audience (201) starts empty, because a slug created again never reaches the
  memory of a deleted audience, so a membership lag there only logs
  memory_membership_lag.
- Every item sent names the run audience. post_batch refuses (MemoryScopeError)
  any item without it, or with the project-wide audience, before any request.

Records are buffered by record(). Each one gets the id "<run>:<kind>:<n>"
(n counts per kind from 1), a short label, the real UTC time of the call,
the run audience, and metadata {"run_id", "kind", "round", "meta"}.
flush() sends them as one /ingest/batch
with an idempotency key, then polls that ingestion (and any earlier one still
processing) until it completes or timeout_s passes. A failed ingestion is
retried once. recall() drops superseded results, "mention" memories and any
result that rests on a data point from another run.

Failures fail soft. A 429 with no body waits (Retry-After, else a pause that
doubles) and retries. A 5xx or a connection error retries with the same body,
so with the same idempotency key. A 401, 402, 403, a reached cap, or retries
running out set enabled to False and log "memory_degraded"; after that
record and recall are no-ops that return empty results and the run carries
on with local history only. end_run(cleanup=True) still tries to delete what
was sent. A 409 on a batch means the run id was reused after a cleanup (a
deleted data point id or a used batch key) and also degrades. Any other
refused batch is dropped alone; a second refusal in a row degrades.

Events (memory_degraded, memory_leak, memory_batch_rejected, ...) go to the
"wsp.swarm.memory" logger, to self.events, and to self.on_event(kind, text,
extra) when the caller sets that hook (for example to env.log_event).

The API key is read from the environment and only ever goes into the
Authorization header. It is never logged, printed or written.

Config keys read (all optional, read with config.get):
    past_base_url                API base URL, default "https://api.past.dev/api/v1"
    past_api_key_env             name of the env var holding the project key, default "PAST_API_KEY"
    memory_level                 recall level used when recall() gets an unknown level, default "low"
    memory_poll_s                seconds between ingestion and membership polls, default 1.5
    memory_http_timeout_s        seconds allowed per HTTP request, default 20
    memory_max_retries           retries per request after the first try, default 3
    memory_backoff_s             first pause before a retry, doubled each time, default 1.0
    memory_membership_timeout_s  how long start_run waits for the audience to reach the guard identity, default 15
    memory_cold_query            the start-of-run guard query, default DEFAULT_COLD_QUERY
    memory_max_chars             longest text sent per record (longer text is cut), default 1500
    memory_verify_cleanup        end_run(cleanup=True) recalls once after deleting and reports whether it
                                 came back empty, default False (it costs one credit)
    memory_state_path            where the state file goes. None (default): <outputs_dir>/runs/<run_id>/memory.json.
                                 "" or False: no state file (and no local check for a reused run id).
                                 start_run reads it first to refuse a run id that was used before.
    outputs_dir                  used for the default state path, default "outputs"
"""

import datetime
import hashlib
import json
import logging
import os
import threading
import time
import urllib.parse

import requests

try:
    from wsp.swarm.interfaces import Memory as MemoryBase
except ImportError:  # the core interfaces module is not there yet
    class MemoryBase(object):
        enabled = False

LOG = logging.getLogger("wsp.swarm.memory")

DEFAULT_BASE_URL = "https://api.past.dev/api/v1"
DEFAULT_COLD_QUERY = "graft score search split collapse lesson fit heal program frontier"

KINDS = ["step", "probe", "collapse", "verdict", "capmap", "plan"]
LABELS = {
    "step": "Step",
    "probe": "Probe",
    "collapse": "Collapse",
    "verdict": "Judge verdict",
    "capmap": "Capability map",
    "plan": "Plan",
}
LEVELS = ["low", "medium", "high", "extra-high"]
DROPPED_ARTIFACT_KINDS = ["mention"]

SLUG_MAX = 40
SLUG_SUFFIX = "-lessons"
SLUG_HASH_CHARS = 6
RUN_KEY_MAX = 100
BATCH_MAX = 1000
DELETE_CHUNK = 1000
RECALL_LIMIT_MAX = 200

# A 429 with one of these codes is a spending cap, not the per-minute limit: stop, do not retry.
CAP_CODES = ["project-cap-reached", "spend-cap-reached"]
# Statuses that mean the key, plan or credits are wrong: retrying cannot help.
STOP_STATUSES = [401, 402, 403]

ID_CHARS = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-"


class MemoryLeakError(RuntimeError):
    """The start-of-run recall found memory, so this run would not start cold."""


class MemoryScopeError(ValueError):
    """An item was about to be sent without this run's audience."""


# Small helpers.

def utc_now_iso():
    """The current UTC time, for example 2026-11-14T23:40:00.123Z."""
    now = datetime.datetime.now(datetime.timezone.utc)
    return now.strftime("%Y-%m-%dT%H:%M:%S.") + "%03dZ" % (now.microsecond // 1000)


def short_hash(text, length):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:length]


def slug_part(text):
    """Lowercase ASCII letters and digits of text; every other run of characters becomes one dash."""
    out = ""
    for ch in text.lower():
        if ("a" <= ch <= "z") or ("0" <= ch <= "9"):
            out += ch
        elif not out.endswith("-"):
            out += "-"
    return out.strip("-")


def audience_slug(run_id):
    """The run's audience slug: a readable prefix of the run id, a 6-character hash, then "-lessons".

    past.dev slugs are 2 to 40 lowercase letters, digits and dashes with a
    letter or digit at each end. The hash covers the full run id, so run ids
    that clean up to the same prefix still get different slugs, and the same
    run id always gets the same slug.
    """
    run_id = str(run_id)
    digest = short_hash(run_id, SLUG_HASH_CHARS)
    room = SLUG_MAX - len(SLUG_SUFFIX) - len(digest) - 1
    base = slug_part(run_id)[:room].strip("-")
    if base == "":
        base = "run"
    return base + "-" + digest + SLUG_SUFFIX


def run_key_for(run_id):
    """The run id as it appears inside identity and data point ids.

    Letters, digits, dot, underscore and dash are kept; anything else (":" and
    "/" included) becomes a dash. A changed or very long run id gets a hash of
    the original, so two run ids never share a key.
    """
    run_id = str(run_id)
    clean = ""
    for ch in run_id:
        if ch in ID_CHARS:
            clean += ch
        else:
            clean += "-"
    if clean == run_id and len(clean) <= RUN_KEY_MAX and clean != "":
        return clean
    return clean[:RUN_KEY_MAX - 9] + "-" + short_hash(run_id, 8)


def clean_meta(meta):
    """A JSON-safe copy of meta (anything odd becomes a string)."""
    if meta is None:
        return {}
    try:
        return json.loads(json.dumps(meta, default=str))
    except (TypeError, ValueError):
        return {"repr": str(meta)}


def clean_text(text, max_chars):
    text = str(text or "").replace("\x00", " ").strip()
    if len(text) > max_chars:
        text = text[:max_chars]
    return text


def read_json(response):
    """The response body as JSON, or None when there is no JSON body."""
    try:
        return response.json()
    except ValueError:
        return None


def error_code(data):
    if isinstance(data, dict):
        return data.get("code")
    return None


def error_text(status, data):
    """A short description of an error answer: status, code and debugMessage (never the request)."""
    text = "status " + str(status)
    if isinstance(data, dict):
        if data.get("code"):
            text += " " + str(data.get("code"))
        if data.get("debugMessage"):
            text += ": " + str(data.get("debugMessage"))[:200]
    return text


def retry_after_s(response, fallback):
    """Seconds from a Retry-After header, else fallback."""
    headers = getattr(response, "headers", None) or {}
    value = headers.get("Retry-After")
    if value is None:
        return fallback
    try:
        return max(0.0, float(value))
    except (TypeError, ValueError):
        return fallback


def source_ids_of(doc):
    """The sourceId of every evidence group of one recall result, in order."""
    out = []
    sources = doc.get("sources") or []
    for source in sources:
        if isinstance(source, dict) and source.get("sourceId"):
            out.append(str(source.get("sourceId")))
    return out


def quote_path(text):
    """Percent-encode one path segment (":" and "/" included)."""
    return urllib.parse.quote(str(text), safe="")


def check_scoped(items, audience):
    """Raise MemoryScopeError unless every item names this run's audience.

    This is the guard of pastdev.md 6.1 rule 4: a data point sent without an
    audience, or with "project", would be readable by every identity in the
    project, including every other benchmark run.
    """
    if not audience or audience == "project":
        raise MemoryScopeError("no run audience: call start_run before sending")
    for item in items:
        item_audience = item.get("audience")
        if not item_audience or item_audience == "project":
            raise MemoryScopeError("refusing to send item " + str(item.get("id")) + " without a run audience")
        if item_audience != audience:
            raise MemoryScopeError("item " + str(item.get("id")) + " names audience " + str(item_audience) +
                                   ", not this run's " + str(audience))


class PastMemory(MemoryBase):
    """Run-scoped lessons in past.dev. See the module docstring for scoping and failure rules."""

    enabled = False

    def __init__(self, config=None):
        if config is None:
            config = {}
        self.config = config
        self.base_url = str(config.get("past_base_url", DEFAULT_BASE_URL)).rstrip("/")
        self.key_env = config.get("past_api_key_env", "PAST_API_KEY")
        self.default_level = config.get("memory_level", "low")
        self.poll_s = float(config.get("memory_poll_s", 1.5))
        self.http_timeout_s = float(config.get("memory_http_timeout_s", 20))
        self.max_retries = int(config.get("memory_max_retries", 3))
        self.backoff_s = float(config.get("memory_backoff_s", 1.0))
        self.membership_timeout_s = float(config.get("memory_membership_timeout_s", 15))
        self.cold_query = config.get("memory_cold_query", DEFAULT_COLD_QUERY)
        self.max_chars = int(config.get("memory_max_chars", 1500))
        self.verify_cleanup = bool(config.get("memory_verify_cleanup", False))
        self.state_path = config.get("memory_state_path", None)
        self.outputs_dir = config.get("outputs_dir", "outputs")
        # Swappable for tests: the HTTP module, the sleep function and the clock.
        self.http = requests
        self.sleep = time.sleep
        self.clock = time.time
        # Optional hook fn(kind, text, extra), for example the env's event logger.
        self.on_event = None
        self.lock = threading.Lock()
        self.reset_state()

    def reset_state(self):
        self.enabled = False
        self.run_id = None
        self.run_key = None
        self.audience = None
        self.identities = {}
        self.guard_identity = None
        self.membership_ready = False
        self.audience_replaced = False
        # Set when start_run refused a reused run id: nothing remote or local is touched after that.
        self.reuse_refused = False
        self.buffer = []
        self.counters = {}
        self.batch_count = 0
        self.refused_in_a_row = 0
        self.inflight = []
        self.sent_ids = []
        self.ingestion_ids = []
        self.last_pending = None
        self.degraded_reason = None
        self.events = []
        self.ended = False
        self.stats = {"batches": 0, "items_sent": 0, "items_dropped": 0, "recalls": 0, "polls": 0,
                      "retries": 0, "failed_ingestions": 0, "foreign_dropped": 0, "deleted": 0}

    # Plumbing: events, degrade, HTTP.

    def count(self, name, amount=1):
        with self.lock:
            self.stats[name] = self.stats.get(name, 0) + amount

    def emit(self, kind, text, extra=None, warn=True):
        """Record one memory event: logger, self.events and the on_event hook."""
        if extra is None:
            extra = {}
        self.events.append({"kind": kind, "text": text, "extra": extra, "at": utc_now_iso()})
        if warn:
            LOG.warning("%s: %s", kind, text)
        else:
            LOG.info("%s: %s", kind, text)
        if self.on_event is not None:
            try:
                self.on_event(kind, text, extra)
            except Exception as error:
                LOG.warning("memory on_event hook failed: %s", type(error).__name__)

    def degrade(self, reason):
        """Turn memory off for the rest of the run and log memory_degraded once."""
        first = self.degraded_reason is None
        if first:
            self.degraded_reason = reason
        self.enabled = False
        with self.lock:
            dropped = len(self.buffer)
            self.buffer = []
        if dropped > 0:
            self.count("items_dropped", dropped)
        if first:
            self.emit("memory_degraded", reason, {"run_id": self.run_id, "dropped": dropped})

    def api_key(self):
        return os.environ.get(self.key_env, "")

    def headers(self):
        return {"Authorization": "Bearer " + self.api_key(), "Content-Type": "application/json"}

    def call(self, method, path, body=None):
        """Send one request. Returns (status, data), or (None, None) when it failed for good.

        A 429 with no body pauses (Retry-After, else the backoff, doubled each
        time) and retries. A 5xx or a connection error retries the same body,
        so the same idempotency key. A 401, 402, 403 or a reached cap degrades
        memory and returns the answer; so does running out of retries. Any
        other answer, 4xx included, goes back to the caller.
        """
        url = self.base_url + path
        wait = self.backoff_s
        last_problem = "no answer"
        for attempt in range(self.max_retries + 1):
            if attempt > 0:
                self.count("retries")
            try:
                response = self.http.request(method, url, headers=self.headers(), json=body,
                                             timeout=self.http_timeout_s)
            except requests.RequestException as error:
                last_problem = type(error).__name__
                LOG.warning("past.dev %s %s: %s (try %d)", method, path, last_problem, attempt + 1)
                if attempt < self.max_retries:
                    self.sleep(wait)
                    wait = wait * 2
                continue
            status = response.status_code
            data = read_json(response)
            if status == 429 and error_code(data) in CAP_CODES:
                self.degrade("past.dev " + method + " " + path + " refused: " + error_text(status, data))
                return status, data
            if status == 429 or status >= 500:
                last_problem = error_text(status, data)
                LOG.warning("past.dev %s %s: %s (try %d)", method, path, last_problem, attempt + 1)
                if attempt < self.max_retries:
                    pause = wait
                    if status == 429:
                        pause = retry_after_s(response, wait)
                    self.sleep(pause)
                    wait = wait * 2
                continue
            if status in STOP_STATUSES:
                self.degrade("past.dev " + method + " " + path + " refused: " + error_text(status, data))
                return status, data
            return status, data
        self.degrade("past.dev " + method + " " + path + " failed after " + str(self.max_retries + 1) +
                     " tries (last: " + last_problem + ")")
        return None, None

    # start_run.

    def start_run(self, run_id, roles):
        """Create the run's audience and identities, then prove a cold start.

        Raises MemoryLeakError if the state file shows the run id was used
        before, or if the guard recall returns anything. Any other problem
        degrades memory to off and returns None.
        """
        try:
            self.open_run(run_id, roles)
        except MemoryLeakError:
            self.save_state()
            raise
        except Exception as error:
            self.degrade("start_run failed: " + type(error).__name__ + ": " + str(error)[:200])
        self.save_state()
        return None

    def open_run(self, run_id, roles):
        self.reset_state()
        self.run_id = str(run_id)
        self.run_key = run_key_for(self.run_id)
        self.audience = audience_slug(self.run_id)
        for role in roles:
            self.identities[str(role)] = self.run_key + ":" + str(role)
        self.refuse_reused_run_id()
        self.guard_identity = self.pick_guard_identity(roles)
        if self.guard_identity is None:
            self.degrade("start_run got no roles, so no identity can read the run audience")
            return
        if self.api_key() == "":
            self.degrade("no " + str(self.key_env) + " in the environment")
            return
        if not self.create_audience():
            return
        self.membership_ready = self.wait_for_membership(self.guard_identity)
        if self.degraded_reason is not None:
            return
        if self.audience_replaced and not self.membership_ready:
            # The audience existed before this run, and the guard recall would see only
            # project-visible memory, not what the audience holds.
            self.degrade("cold start not proven: audience " + self.audience + " already existed and " +
                         self.guard_identity + " was not confirmed in it within " +
                         str(self.membership_timeout_s) + " s, so the start-of-run recall cannot see it")
            return
        self.assert_cold_start()
        if self.degraded_reason is not None:
            return
        self.enabled = True
        self.emit("memory_started", "run " + self.run_id + " uses audience " + self.audience,
                  {"audience": self.audience, "identities": dict(self.identities),
                   "membership_ready": self.membership_ready}, warn=False)

    def refuse_reused_run_id(self):
        """Raise MemoryLeakError when the state file shows this run id already sent data or ended.

        A run that was not cleaned up left memory behind, and past.dev never
        accepts a deleted data point id or a used batch key again, so a reused
        run id cannot give a fair cold-start run. The check is local and makes
        no request. After a refusal the old state file is kept as it was and
        end_run touches nothing.
        """
        path = self.state_file()
        if path is None or not os.path.exists(path):
            return
        try:
            with open(path, encoding="utf-8") as handle:
                old = json.load(handle)
        except (OSError, ValueError) as error:
            self.emit("memory_state_unreadable", "could not read " + path + " (" + type(error).__name__ +
                      "); the cold-start recall and the batch checks still guard against a reused run id")
            return
        if not isinstance(old, dict) or old.get("run_id") != self.run_id:
            return
        sent = old.get("sent_ids") or []
        ended = bool(old.get("ended"))
        if len(sent) == 0 and not ended:
            return
        self.reuse_refused = True
        message = ("start_run: run id " + self.run_id + " was used before (" + path + " shows " + str(len(sent)) +
                   " data points sent, ended " + str(ended) + "). A benchmark run must start cold, and past.dev "
                   "never accepts a deleted data point id or a used batch key again: use a fresh run id.")
        self.emit("memory_run_id_reused", message, {"state_file": path, "sent": len(sent), "ended": ended})
        raise MemoryLeakError(message)

    def pick_guard_identity(self, roles):
        """The identity used for the cold-start recall: the first planner, else the first role."""
        roles = list(roles)
        for role in roles:
            if str(role).startswith("planner"):
                return self.identities[str(role)]
        if len(roles) > 0:
            return self.identities[str(roles[0])]
        return None

    def create_audience(self):
        """PUT the run's fixed audience, then recompute it. Returns True on success."""
        identity_list = []
        for role in sorted(self.identities):
            identity_list.append(self.identities[role])
        body = {"kind": "fixed", "name": "wsp run " + self.run_id[:180], "identities": identity_list}
        status, data = self.call("PUT", "/audiences/" + quote_path(self.audience), body)
        if self.degraded_reason is not None:
            return False
        if status not in (200, 201):
            self.degrade("creating audience " + self.audience + " failed: " + error_text(status, data))
            return False
        if status == 200:
            self.audience_replaced = True
            self.emit("memory_audience_replaced", "audience " + self.audience + " already existed (run id reused?); "
                      "the cold-start recall decides whether it holds memory", {"audience": self.audience})
        status, data = self.call("POST", "/audiences/" + quote_path(self.audience) + "/recompute")
        if self.degraded_reason is not None:
            return False
        if status != 200:
            self.emit("memory_recompute_failed", "recompute of " + self.audience + " answered " +
                      error_text(status, data) + "; waiting for membership instead")
        return True

    def wait_for_membership(self, identity):
        """Poll GET /identities/<identity> until it reaches the run audience. Returns True when it does."""
        deadline = self.clock() + self.membership_timeout_s
        while True:
            status, data = self.call("GET", "/identities/" + quote_path(identity))
            if self.degraded_reason is not None:
                return False
            if status == 200 and isinstance(data, dict):
                reached = data.get("audiences") or []
                if self.audience in reached:
                    return True
            if self.clock() >= deadline:
                self.emit("memory_membership_lag", identity + " did not reach " + self.audience + " within " +
                          str(self.membership_timeout_s) + " s; continuing")
                return False
            self.sleep(self.poll_s)

    def assert_cold_start(self):
        """Recall as the guard identity before anything is sent. Any result means a leak."""
        docs = self.recall_raw(self.guard_identity, self.cold_query, "low", 20)
        if self.degraded_reason is not None:
            return
        if docs is None:
            self.degrade("the cold-start recall could not run, so a cold start is not proven")
            return
        if len(docs) == 0:
            return
        found = []
        for doc in docs:
            if isinstance(doc, dict):
                for source_id in source_ids_of(doc):
                    if source_id not in found:
                        found.append(source_id)
        self.enabled = False
        message = ("start_run: recall as " + self.guard_identity + " returned " + str(len(docs)) +
                   " results before this run wrote anything (sources: " + ", ".join(found[:10]) + "). "
                   "A benchmark run must start cold: use a fresh run id, or delete the leaked data points "
                   "(project-visible points reach every identity).")
        self.emit("memory_leak", message, {"sources": found[:50]})
        raise MemoryLeakError(message)

    # record and flush.

    def record(self, kind, text, meta=None, round_index=None):
        """Queue one lesson for the next flush. Never blocks on the network, never raises."""
        try:
            self.queue_item(kind, text, meta, round_index)
        except Exception as error:
            LOG.warning("memory record failed: %s", type(error).__name__)
        return None

    def queue_item(self, kind, text, meta, round_index):
        if self.audience is None:
            self.emit("memory_scope_refused", "record(" + str(kind) + ") before start_run: no run audience, "
                      "item dropped")
            return
        if not self.enabled:
            return
        if kind not in KINDS:
            self.emit("memory_bad_kind", "unknown record kind " + str(kind) + "; item dropped")
            return
        text = clean_text(text, self.max_chars)
        if text == "":
            return
        metadata = {"run_id": self.run_id, "kind": kind, "round": round_index, "meta": clean_meta(meta)}
        with self.lock:
            number = self.counters.get(kind, 0) + 1
            self.counters[kind] = number
            label = LABELS[kind] + " " + str(number)
            if round_index is not None:
                label += " (round " + str(round_index) + ")"
            item = {
                "id": self.run_key + ":" + kind + ":" + str(number),
                "content": text,
                "label": label,
                "timestamp": utc_now_iso(),
                "audience": self.audience,
                "metadata": metadata,
            }
            self.buffer.append(item)

    def flush(self, timeout_s=30):
        """Send the queued records as one batch and wait up to timeout_s for them to complete.

        Returns {"ok", "lag", "pending"}. ok is False when a send was refused,
        an ingestion failed or memory is degraded. lag is True when the wait
        timed out with ingestions still processing; pending counts their
        records. Never raises.
        """
        try:
            return self.send_and_wait(timeout_s)
        except Exception as error:
            self.degrade("flush failed: " + type(error).__name__ + ": " + str(error)[:200])
            return {"ok": False, "lag": False, "pending": 0}

    def send_and_wait(self, timeout_s):
        if not self.enabled:
            return {"ok": False, "lag": False, "pending": 0}
        with self.lock:
            items = self.buffer
            self.buffer = []
        ok = True
        start = 0
        while start < len(items) and self.enabled:
            if not self.send_batch(items[start:start + BATCH_MAX]):
                ok = False
            start += BATCH_MAX
        if not self.enabled:
            self.save_state()
            return {"ok": False, "lag": False, "pending": 0}
        waited = self.wait_inflight(timeout_s)
        self.save_state()
        return {"ok": ok and waited["ok"], "lag": waited["lag"], "pending": waited["pending"]}

    def post_batch(self, items, key):
        """POST one batch after checking that every item names the run audience."""
        check_scoped(items, self.audience)
        return self.call("POST", "/ingest/batch", {"idempotencyKey": key, "items": items})

    def send_batch(self, items):
        """Send one batch. Returns True when past.dev accepted it.

        A 409 degrades memory: on a batch it means a data point id was deleted
        before (data-point-deleted) or the batch key was used with another body
        (idempotency-conflict), and either way this run id was used before. Any
        other refusal drops this batch only, unless the previous batch was
        refused too, which degrades.
        """
        self.batch_count += 1
        key = self.run_key + ":batch:" + str(self.batch_count)
        status, data = self.post_batch(items, key)
        if self.degraded_reason is not None:
            return False
        if status == 409:
            self.count("items_dropped", len(items))
            self.degrade("run id reused after cleanup: past.dev refused batch " + str(self.batch_count) + " with " +
                         error_text(status, data) + " (a deleted data point id or a used batch key is never "
                         "accepted again); use a fresh run id")
            return False
        if status not in (200, 202) or not isinstance(data, dict):
            self.count("items_dropped", len(items))
            self.refused_in_a_row += 1
            self.emit("memory_batch_rejected", "batch " + str(self.batch_count) + " of " + str(len(items)) +
                      " records refused: " + error_text(status, data), {"key": key})
            if self.refused_in_a_row >= 2:
                self.degrade("past.dev refused " + str(self.refused_in_a_row) + " batches in a row (last: " +
                             error_text(status, data) + ")")
            return False
        self.refused_in_a_row = 0
        for item in items:
            self.sent_ids.append(item["id"])
        self.count("batches")
        self.count("items_sent", len(items))
        ingestion_id = data.get("ingestionId")
        if ingestion_id:
            self.ingestion_ids.append(ingestion_id)
            if data.get("status") != "completed":
                self.inflight.append({"id": ingestion_id, "count": len(items), "retried": False,
                                      "sent_at": self.clock()})
        return True

    def wait_inflight(self, timeout_s):
        """Poll every ingestion still processing until all complete or timeout_s passes."""
        deadline = self.clock() + float(timeout_s)
        failed = 0
        while True:
            still = []
            for entry in self.inflight:
                state = self.poll_ingestion(entry)
                if state == "processing":
                    still.append(entry)
                elif state == "failed":
                    failed += entry["count"]
            self.inflight = still
            if len(still) == 0 or not self.enabled:
                break
            if self.clock() >= deadline:
                break
            self.sleep(self.poll_s)
        if not self.enabled:
            return {"ok": False, "lag": False, "pending": 0}
        pending = 0
        for entry in self.inflight:
            pending += entry["count"]
        return {"ok": failed == 0, "lag": len(self.inflight) > 0, "pending": pending}

    def poll_ingestion(self, entry):
        """One status read. Returns "completed", "processing", "failed" or "gone".

        A failed ingestion is retried once (free); a second failure, or a
        content-rejected failure, which a retry cannot fix, gives "failed".
        """
        status, data = self.call("GET", "/ingest/" + quote_path(entry["id"]))
        self.count("polls")
        if status is None:
            return "processing"
        if status == 404:
            self.emit("memory_ingestion_unknown", "ingestion " + entry["id"] + " is unknown to past.dev")
            return "gone"
        if status != 200 or not isinstance(data, dict):
            return "processing"
        state = data.get("status")
        if state == "completed":
            return "completed"
        if state != "failed":
            return "processing"
        failure = data.get("failure") or {}
        code = failure.get("code") if isinstance(failure, dict) else None
        if entry["retried"] or code == "content-rejected":
            self.count("failed_ingestions")
            self.emit("memory_ingestion_failed", "ingestion " + entry["id"] + " failed (" + str(code) + "), " +
                      str(entry["count"]) + " records lost", {"ingestion_id": entry["id"], "code": code})
            return "failed"
        entry["retried"] = True
        self.emit("memory_ingestion_retry", "ingestion " + entry["id"] + " failed (" + str(code) + "); retrying once",
                  {"ingestion_id": entry["id"], "code": code}, warn=False)
        retry_status, retry_data = self.call("POST", "/ingest/" + quote_path(entry["id"]) + "/retry")
        if retry_status in (200, 202, 409):
            return "processing"
        self.count("failed_ingestions")
        return "failed"

    # recall.

    def recall(self, role, query, level="low", limit=20):
        """Lessons for one role from this run only, best first.

        Returns a list of {"text", "source_id", "occurred_at", "kind"} (plus
        "artifact" and "rank"). kind is the record kind ("step", "collapse",
        ...) read from the source id, or past.dev's memory kind when the id is
        not one of ours. Returns [] on any failure, when memory is off, for an
        unknown role, and before anything was sent (no call is made then).
        """
        try:
            return self.recall_lessons(role, query, level, limit)
        except Exception as error:
            LOG.warning("memory recall failed: %s", type(error).__name__)
            return []

    def recall_lessons(self, role, query, level, limit):
        if not self.enabled:
            return []
        identity = self.identities.get(str(role))
        if identity is None:
            self.emit("memory_unknown_role", "recall by role " + str(role) + ", which start_run did not name")
            return []
        if len(self.sent_ids) == 0:
            return []
        if level not in LEVELS:
            level = self.default_level
        limit = max(1, min(RECALL_LIMIT_MAX, int(limit)))
        docs = self.recall_raw(identity, str(query), level, limit)
        if docs is None:
            return []
        return self.lessons_from(docs)

    def recall_raw(self, identity, query, level="low", limit=20):
        """POST /recall as identity. Returns the raw result list, or None on failure."""
        body = {"query": query, "identity": identity, "level": level, "limit": limit}
        status, data = self.call("POST", "/recall", body)
        self.count("recalls")
        if status is None:
            return None
        if status != 200 or not isinstance(data, dict):
            LOG.warning("past.dev recall answered %s", error_text(status, data))
            return None
        # past.dev sends "pending" only while some ingestion of the project is still processing.
        self.last_pending = data.get("pending")
        results = data.get("results") or []
        return list(results)

    def lessons_from(self, docs):
        """Map raw recall results to the contract shape, dropping superseded, mention and foreign ones."""
        out = []
        for doc in docs:
            if not isinstance(doc, dict):
                continue
            artifact = doc.get("artifact") or {}
            if artifact.get("supersededAt"):
                continue
            artifact_kind = str(artifact.get("kind") or "unknown")
            if artifact_kind in DROPPED_ARTIFACT_KINDS:
                continue
            text = str(doc.get("content") or "").strip()
            if text == "":
                continue
            source_ids = source_ids_of(doc)
            if self.has_foreign_source(source_ids):
                self.count("foreign_dropped")
                self.emit("memory_foreign_result", "dropped a recall result resting on another run's data: " +
                          ", ".join(source_ids[:5]))
                continue
            source_id = ""
            if len(source_ids) > 0:
                source_id = source_ids[0]
            occurred_at = doc.get("occurredAt") or artifact.get("occurredAt")
            out.append({"text": text, "source_id": source_id, "occurred_at": occurred_at,
                        "kind": self.kind_of(source_id, artifact_kind), "artifact": artifact_kind,
                        "rank": doc.get("rank")})
        return out

    def has_foreign_source(self, source_ids):
        prefix = str(self.run_key) + ":"
        for source_id in source_ids:
            if not source_id.startswith(prefix):
                return True
        return False

    def kind_of(self, source_id, artifact_kind):
        prefix = str(self.run_key) + ":"
        if source_id.startswith(prefix):
            kind = source_id[len(prefix):].split(":")[0]
            if kind in KINDS:
                return kind
        return artifact_kind

    # end_run and cleanup.

    def end_run(self, cleanup=False):
        """Close the run. Never raises.

        cleanup=False sends any records still queued (without waiting).
        cleanup=True drops the queue, deletes every data point this run sent
        (POST /data-points/delete, 1,000 ids per call), then deletes the run
        audience and identities. Returns {"deleted", "audience_deleted",
        "verified_empty"}; verified_empty is None unless memory_verify_cleanup
        is set. After start_run refused a reused run id it does nothing, so
        the earlier run's audience and state file stay as they were.
        """
        result = {"deleted": 0, "audience_deleted": False, "verified_empty": None}
        try:
            self.close_run(cleanup, result)
        except Exception as error:
            LOG.warning("memory end_run failed: %s", type(error).__name__)
        self.enabled = False
        self.ended = True
        self.save_state()
        return result

    def close_run(self, cleanup, result):
        if self.run_id is None or self.reuse_refused:
            return
        if not cleanup:
            if self.enabled and len(self.buffer) > 0:
                self.send_and_wait(0)
            return
        with self.lock:
            dropped = len(self.buffer)
            self.buffer = []
        if dropped > 0:
            self.count("items_dropped", dropped)
        if self.api_key() == "" or self.audience is None:
            return
        result["deleted"] = self.delete_points(list(self.sent_ids))
        if self.verify_cleanup and self.guard_identity is not None:
            docs = self.recall_raw(self.guard_identity, self.cold_query, "low", 20)
            if docs is not None:
                result["verified_empty"] = len(docs) == 0
        result["audience_deleted"] = self.delete_scope()
        self.emit("memory_cleanup", "deleted " + str(result["deleted"]) + " of " + str(len(self.sent_ids)) +
                  " data points; audience deleted: " + str(result["audience_deleted"]), dict(result), warn=False)

    def delete_points(self, ids):
        """Delete ids in chunks of 1,000. A chunk refused with 404 falls back to one-by-one deletes."""
        deleted = 0
        start = 0
        while start < len(ids):
            chunk = ids[start:start + DELETE_CHUNK]
            status, data = self.call("POST", "/data-points/delete", {"ids": chunk})
            if status in (200, 202):
                deleted += len(chunk)
            elif status == 404:
                deleted += self.delete_one_by_one(chunk)
            else:
                self.emit("memory_cleanup_failed", "deleting " + str(len(chunk)) + " data points answered " +
                          error_text(status, data))
            start += DELETE_CHUNK
        self.count("deleted", deleted)
        return deleted

    def delete_one_by_one(self, ids):
        """DELETE /data-points/<id> for each id; ids past.dev does not know are skipped."""
        deleted = 0
        for source_id in ids:
            status, data = self.call("DELETE", "/data-points/" + quote_path(source_id))
            if status in (200, 202, 204):
                deleted += 1
        return deleted

    def delete_scope(self):
        """Delete the run audience and its identities. Returns True when the audience went."""
        status, data = self.call("DELETE", "/audiences/" + quote_path(self.audience))
        for role in sorted(self.identities):
            self.call("DELETE", "/identities/" + quote_path(self.identities[role]))
        return status in (200, 204)

    # State file.

    def state_file(self):
        if self.state_path is None:
            if self.run_id is None:
                return None
            return os.path.join(str(self.outputs_dir), "runs", self.run_id, "memory.json")
        if not self.state_path:
            return None
        return str(self.state_path)

    def save_state(self):
        """Write the run-id-to-audience mapping, sent ids and events to the state file. Never raises.

        After a refused reused run id it writes nothing, so the earlier run's file keeps its sent ids.
        """
        if self.reuse_refused:
            return
        path = self.state_file()
        if path is None:
            return
        state = {
            "run_id": self.run_id,
            "run_key": self.run_key,
            "audience": self.audience,
            "identities": dict(self.identities),
            "enabled": self.enabled,
            "degraded_reason": self.degraded_reason,
            "ended": self.ended,
            "sent_ids": list(self.sent_ids),
            "ingestion_ids": list(self.ingestion_ids),
            "inflight": list(self.inflight),
            "stats": dict(self.stats),
            "events": list(self.events[-50:]),
            "saved_at": utc_now_iso(),
        }
        try:
            folder = os.path.dirname(path)
            if folder:
                os.makedirs(folder, exist_ok=True)
            with open(path + ".tmp", "w", encoding="utf-8") as handle:
                json.dump(state, handle, indent=1, default=str)
            os.replace(path + ".tmp", path)
        except OSError as error:
            LOG.warning("memory state file not written: %s", type(error).__name__)
