"""Tests for the Exa literature priors (wsp/swarm/literature.py, scripts/build_priors.py).

CPU only, no network. The exa_py client is replaced by FakeExa, the modal
Dict tier by FakeDict, and requests.get/post raise if anything reaches them.
"""

import hashlib
import importlib.util
import json
import os

import exa_py
import exa_py.api as exa_api
import pytest
import requests

from wsp.swarm import literature
from wsp.swarm.literature import ExaLiterature, cache_file, cache_key

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PAPER_URLS = [
    "https://arxiv.org/abs/2106.07682",
    "https://arxiv.org/abs/2403.17887",
    "https://arxiv.org/abs/2407.09298",
    "https://arxiv.org/abs/2312.15166",
    "https://arxiv.org/abs/2403.13187",
]


# Fakes.

class FakeExaWorld:
    """What the fake Exa service returns, and a record of what it was asked."""

    def __init__(self):
        self.clients = 0
        self.searches = []
        self.errors = []
        self.urls = list(PAPER_URLS)
        self.cost = 0.007


def fake_response(world, query, search_type):
    results = []
    for index in range(len(world.urls)):
        results.append(exa_api.Result(url=world.urls[index], id=str(index), title="Paper " + str(index),
                                      highlights=["Quote " + str(index) + " about " + query]))
    output = None
    if search_type == "deep-lite":
        claims = [{"claim": "Middle layers stitch best for " + query[:20], "layer_hint": "0.4 to 0.6 of depth",
                   "model": "Llama-2-7B", "strength": "medium"}]
        grounding = [exa_api.DeepSearchOutputGrounding(
            field="claims[0].claim",
            citations=[exa_api.DeepSearchOutputGroundingCitation(url=world.urls[0], title="Paper 0")],
            confidence="high")]
        output = exa_api.DeepSearchOutput(content={"claims": claims}, grounding=grounding)
    return exa_api.SearchResponse(results=results, resolved_search_type=search_type, auto_date=None,
                                  output=output, cost_dollars=exa_api.CostDollars(total=world.cost))


def make_fake_exa_class(world):
    class FakeExa:
        def __init__(self, api_key=None, base_url="https://api.exa.ai", user_agent=None, timeout=600.0):
            world.clients += 1

        def search(self, query, **kwargs):
            call = dict(kwargs)
            call["query"] = query
            world.searches.append(call)
            if len(world.errors) > 0:
                raise world.errors.pop(0)
            return fake_response(world, query, kwargs.get("type"))

    return FakeExa


class FakeDict:
    """Stands in for modal.Dict: get and put on a plain dict."""

    def __init__(self, store, fail=False):
        self.store = store
        self.fail = fail

    def get(self, key, default=None):
        if self.fail:
            raise RuntimeError("modal is unreachable")
        if key in self.store:
            return self.store[key]
        return default

    def put(self, key, value, skip_if_exists=False):
        if self.fail:
            raise RuntimeError("modal is unreachable")
        self.store[key] = value
        return True


def no_network(*args, **kwargs):
    raise AssertionError("a test reached the network")


@pytest.fixture
def world(monkeypatch):
    world = FakeExaWorld()
    monkeypatch.setattr(exa_py, "Exa", make_fake_exa_class(world))
    monkeypatch.setattr(requests, "post", no_network)
    monkeypatch.setattr(requests, "get", no_network)
    monkeypatch.setenv("EXA_API_KEY", "placeholder-not-a-key")
    return world


def lit_config(tmp_path, extra=None):
    config = {"exa_cache_dir": str(tmp_path / "exa_cache"), "priors_path": str(tmp_path / "priors.json"),
              "exa_backoff_s": 0.0}
    if extra is not None:
        for name in extra:
            config[name] = extra[name]
    return config


def write_cached_response(config, lit, query, k, world):
    """Put a canned response for lit.search(query, k) into the tier-one cache by hand."""
    params = lit.search_params(query, k)
    key = cache_key("search", params, lit.sdk_version)
    path = cache_file(config["exa_cache_dir"], key)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    entry = {"key": key, "endpoint": "search", "params": params,
             "response": literature.response_to_dict(fake_response(world, query, "lite"))}
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(entry, handle)
    return key


def load_build_priors():
    path = os.path.join(REPO_ROOT, "scripts", "build_priors.py")
    spec = importlib.util.spec_from_file_location("build_priors_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_catalog(path):
    records = [
        {"id": "mask_a", "type": "deletion_mask", "base": "unsloth/Llama-3.2-1B-Instruct",
         "trained_on": "MMLU-Pro biology SFT"},
        {"id": "mask_b", "type": "deletion_mask", "base": "unsloth/Llama-3.2-1B-Instruct",
         "trained_on": "BBEH disambiguation_qa SFT"},
        {"id": "qmath_l14_c0", "type": "mlp_cluster", "model": "Qwen/Qwen2.5-Math-1.5B", "layer": 14,
         "trained_on": "GSM8K train activations"},
        {"id": "qmath_l15_c1", "type": "mlp_cluster", "model": "Qwen/Qwen2.5-Math-1.5B", "layer": 15,
         "trained_on": "GSM8K train activations"},
        {"id": "host_record", "type": "host", "model": "unsloth/Llama-3.2-1B-Instruct"},
    ]
    with open(path, "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")
    return path


# Cache key.

def test_cache_key_is_stable_and_ignores_volatile_fields():
    params = {"query": "model stitching", "type": "lite", "category": "publication", "num_results": 3,
              "contents": {"highlights": True}}
    reordered = {"contents": {"highlights": True}, "num_results": 3, "category": "publication",
                 "type": "lite", "query": "model stitching", "user_identifier": "planner-2", "timeout": 5,
                 "objective": None}
    key = cache_key("search", params, "2.28.0")
    assert key == cache_key("search", reordered, "2.28.0")
    # Pinned: a change here would orphan every committed cache file.
    assert key == "089bfd9748dc1dfb4ca45b720efa9b7b568df269eb64702fb61a24e8221ecaec"
    expected_text = json.dumps({"endpoint": "search", "sdk_version": "2.28.0", "params": params},
                               sort_keys=True, separators=(",", ":"))
    assert key == hashlib.sha256(expected_text.encode("utf-8")).hexdigest()
    assert key != cache_key("search", params, "2.29.0")
    assert key != cache_key("find_similar", params, "2.28.0")
    dated = dict(params)
    dated["end_published_date"] = "2026-11-14T00:00:00Z"
    assert key != cache_key("search", dated, "2.28.0")


def test_cache_file_layout(world, tmp_path):
    config = lit_config(tmp_path)
    lit = ExaLiterature(config, "live")
    lit.search("planner-1", "model stitching", k=3)
    rows = lit.drain_calls()
    key = rows[0]["key"]
    path = os.path.join(config["exa_cache_dir"], key[:2], key + ".json")
    assert os.path.exists(path)
    with open(path, encoding="utf-8") as handle:
        entry = json.load(handle)
    assert entry["key"] == key
    assert entry["endpoint"] == "search"
    assert entry["sdk_version"] == lit.sdk_version
    assert entry["cost_dollars"] == pytest.approx(0.007)
    assert entry["response"]["results"][0]["url"] == PAPER_URLS[0]
    assert "placeholder-not-a-key" not in json.dumps(entry)


# Search parameters.

def test_search_params_follow_the_exa_notes(tmp_path):
    lit = ExaLiterature(lit_config(tmp_path), "replay")
    params = lit.search_params("  model   stitching\nacross families ", 3)
    assert params["query"] == "model stitching across families"
    assert params["type"] == "lite"
    assert params["category"] == "publication"
    assert params["num_results"] == 3
    assert params["contents"] == {"highlights": True}
    assert params["end_published_date"] == literature.DEFAULT_END_DATE
    assert "huggingface.co/datasets" in params["exclude_domains"]
    assert "kaggle.com" in params["exclude_domains"]
    assert lit.search_params("x", 0)["num_results"] == 1
    assert lit.search_params("x", 50)["num_results"] == 10
    assert len(lit.search_params("word " * 500, 3)["query"]) <= 400
    assert lit.search_params("   ", 3) is None
    auto = ExaLiterature(lit_config(tmp_path, {"exa_search_type": "auto"}), "replay")
    assert auto.search_params("x", 3)["type"] == "auto"
    with pytest.raises(ValueError):
        ExaLiterature(lit_config(tmp_path, {"exa_search_type": "neural"}), "replay")
    with pytest.raises(ValueError):
        ExaLiterature(lit_config(tmp_path), "sometimes")


# Modes.

def test_off_mode_makes_no_calls(world, tmp_path):
    config = lit_config(tmp_path)
    with open(config["priors_path"], "w", encoding="utf-8") as handle:
        json.dump({"components": {"p": [{"url": PAPER_URLS[0], "title": "t", "quote": "q"}]}, "topics": []},
                  handle)
    lit = ExaLiterature(config, "off")
    assert lit.mode == "off"
    assert lit.priors() == {"sha": "", "components": {}, "topics": []}
    assert lit.search("planner-1", "model stitching") == []
    assert lit.known_url(PAPER_URLS[0]) is False
    assert lit.drain_calls() == []
    assert world.clients == 0
    assert world.searches == []


def test_replay_never_touches_the_network(world, tmp_path):
    config = lit_config(tmp_path)
    lit = ExaLiterature(config, "replay")
    assert lit.search("planner-1", "model stitching", k=3) == []
    rows = lit.drain_calls()
    assert len(rows) == 1
    assert rows[0]["status"] == "replay_miss"
    assert rows[0]["cache_hit"] is False
    assert rows[0]["cost_dollars"] == 0.0
    assert rows[0]["mode"] == "replay"

    write_cached_response(config, lit, "model stitching", 3, world)
    cites = lit.search("planner-1", "model stitching", k=3)
    assert len(cites) == 3
    assert cites[0] == {"url": PAPER_URLS[0], "title": "Paper 0", "quote": "Quote 0 about model stitching"}
    rows = lit.drain_calls()
    assert rows[0]["status"] == "hit"
    assert rows[0]["tier"] == "file"
    assert rows[0]["cost_dollars"] == 0.0
    assert world.clients == 0
    assert world.searches == []


def test_live_miss_then_hit(world, tmp_path):
    config = lit_config(tmp_path)
    lit = ExaLiterature(config, "live")
    assert lit.mode == "live"
    first = lit.search("planner-1", "layer placement in transformers", k=2)
    assert len(first) == 2
    assert len(world.searches) == 1
    sent = world.searches[0]
    assert sent["type"] == "lite"
    assert sent["category"] == "publication"
    assert sent["num_results"] == 2
    assert sent["contents"] == {"highlights": True}
    assert "huggingface.co/datasets" in sent["exclude_domains"]
    rows = lit.drain_calls()
    assert rows[0]["status"] == "live"
    assert rows[0]["cache_hit"] is False
    assert rows[0]["cost_dollars"] == pytest.approx(0.007)
    assert rows[0]["cost_estimate"] == pytest.approx(0.004 + 2 * 0.001)

    second = lit.search("planner-2", "layer placement in transformers", k=2)
    assert second == first
    assert len(world.searches) == 1
    rows = lit.drain_calls()
    assert rows[0]["status"] == "hit"
    assert rows[0]["cache_hit"] is True
    assert rows[0]["cost_dollars"] == 0.0
    assert rows[0]["role"] == "planner-2"

    replay = ExaLiterature(config, "replay")
    assert replay.search("planner-1", "layer placement in transformers", k=2) == first
    assert replay.known_url(PAPER_URLS[1]) is True
    assert len(world.searches) == 1


def test_live_without_a_key_degrades_to_replay(world, monkeypatch, tmp_path):
    monkeypatch.delenv("EXA_API_KEY", raising=False)
    lit = ExaLiterature(lit_config(tmp_path), "live")
    assert lit.requested_mode == "live"
    assert lit.mode == "replay"
    assert "EXA_API_KEY" in lit.degraded_reason
    assert lit.search("planner-1", "model stitching") == []
    rows = lit.drain_calls()
    assert rows[0]["status"] == "degraded"
    assert "EXA_API_KEY" in rows[0]["note"]
    assert rows[1]["status"] == "replay_miss"
    assert world.clients == 0


def test_live_without_the_sdk_degrades_to_replay(world, monkeypatch, tmp_path):
    def missing_sdk(timeout_s):
        raise ImportError("No module named 'exa_py'")

    monkeypatch.setattr(literature, "make_exa_client", missing_sdk)
    lit = ExaLiterature(lit_config(tmp_path), "live")
    assert lit.mode == "replay"
    assert "exa-py" in lit.degraded_reason


# Priors pack.

def test_priors_pack_and_its_sha(world, tmp_path):
    config = lit_config(tmp_path)
    pack = {"version": 1, "sdk_version": literature.installed_sdk_version(),
            "components": {"qmath_l14_c0": [{"url": PAPER_URLS[0], "title": "Revisiting stitching",
                                             "quote": "stitching connectivity"}]},
            "topics": [{"topic": "layer_placement", "claim": "Middle layers are interchangeable.",
                        "layer_hint": "middle", "model": "Llama-2", "strength": "medium",
                        "cites": [{"url": PAPER_URLS[2], "title": "Painters", "quote": "q"}], "grounded": True}]}
    raw = json.dumps(pack, sort_keys=True, indent=2).encode("utf-8")
    with open(config["priors_path"], "wb") as handle:
        handle.write(raw)
    for mode in ("replay", "live"):
        lit = ExaLiterature(config, mode)
        priors = lit.priors()
        assert sorted(priors.keys()) == ["components", "sha", "topics"]
        assert priors["sha"] == hashlib.sha256(raw).hexdigest()
        assert priors["components"] == pack["components"]
        assert priors["topics"] == pack["topics"]
        priors["components"].clear()
        assert lit.priors()["components"] == pack["components"]
        assert lit.known_url(PAPER_URLS[0]) is True
        assert lit.known_url(PAPER_URLS[2] + "/") is True
        assert lit.known_url("https://example.com/made-up") is False
    assert world.searches == []


def test_priors_sha_is_the_same_for_lf_and_crlf_checkouts(tmp_path):
    pack = {"version": 1, "sdk_version": literature.installed_sdk_version(),
            "components": {"mask_a": [{"url": PAPER_URLS[0], "title": "Stitching", "quote": "line one\r\nline two"}]},
            "topics": []}
    lf_bytes = (json.dumps(pack, sort_keys=True, indent=2) + "\n").encode("utf-8")
    assert b"\r" not in lf_bytes
    crlf_bytes = lf_bytes.replace(b"\n", b"\r\n")
    assert crlf_bytes != lf_bytes
    shas = []
    for name, raw in (("lf.json", lf_bytes), ("crlf.json", crlf_bytes)):
        config = lit_config(tmp_path)
        config["priors_path"] = str(tmp_path / name)
        with open(config["priors_path"], "wb") as handle:
            handle.write(raw)
        priors = ExaLiterature(config, "replay").priors()
        assert priors["components"] == pack["components"]
        shas.append(priors["sha"])
    assert shas[0] == shas[1] == hashlib.sha256(lf_bytes).hexdigest()
    assert literature.text_sha256(crlf_bytes) == literature.text_sha256(lf_bytes)


def test_missing_or_broken_priors_pack_is_empty(tmp_path):
    config = lit_config(tmp_path)
    lit = ExaLiterature(config, "replay")
    assert lit.priors() == {"sha": "", "components": {}, "topics": []}
    assert lit.drain_calls()[0]["status"] == "priors_missing"
    with open(config["priors_path"], "w", encoding="utf-8") as handle:
        handle.write("{not json")
    broken = ExaLiterature(config, "replay")
    assert broken.priors()["sha"] == ""
    assert broken.drain_calls()[0]["status"] == "priors_error"


# Dataset mirrors.

def test_dataset_mirrors_are_excluded(world, tmp_path):
    world.urls = ["https://huggingface.co/datasets/openai/gsm8k",
                  "https://www.kaggle.com/datasets/someone/gsm8k",
                  "https://github.com/openai/grade-school-math/blob/master/README.md",
                  "https://blocked.example.org/page",
                  PAPER_URLS[0], PAPER_URLS[1], PAPER_URLS[2]]
    lit = ExaLiterature(lit_config(tmp_path, {"exa_exclude_domains": ["blocked.example.org"]}), "live")
    cites = lit.search("planner-1", "gsm8k arithmetic layers", k=3)
    urls = []
    for cite in cites:
        urls.append(cite["url"])
    assert urls == [PAPER_URLS[0], PAPER_URLS[1], PAPER_URLS[2]]
    sent = world.searches[0]["exclude_domains"]
    for pattern in literature.DATASET_MIRRORS:
        assert pattern in sent
    assert "blocked.example.org" in sent
    assert lit.known_url(PAPER_URLS[0]) is True
    assert lit.known_url("https://huggingface.co/datasets/openai/gsm8k") is False


def test_domain_patterns():
    mirrors = literature.DATASET_MIRRORS
    assert literature.is_excluded("https://huggingface.co/datasets/openai/gsm8k", mirrors)
    assert literature.is_excluded("https://datasets.kaggle.com/x", mirrors)
    assert literature.is_excluded("https://www.kaggle.com/c/x", mirrors)
    assert not literature.is_excluded("https://huggingface.co/papers/2106.07682", mirrors)
    assert not literature.is_excluded("https://huggingface.co/datasetsfoo", mirrors)
    assert not literature.is_excluded("https://arxiv.org/abs/2106.07682", mirrors)
    assert not literature.is_excluded("not a url", mirrors)


# Failures.

def test_server_errors_are_retried_then_fail_soft(world, tmp_path):
    lit = ExaLiterature(lit_config(tmp_path, {"exa_retries": 2}), "live")
    world.errors = [ValueError("Request failed with status code 503: busy"),
                    ValueError("Request failed with status code 429: slow down")]
    assert len(lit.search("planner-1", "query one")) == 3
    assert len(world.searches) == 3

    world.errors = [ValueError("Request failed with status code 500: oops")] * 3
    assert lit.search("planner-1", "query two") == []
    assert len(world.searches) == 6
    lit.drain_calls()

    world.errors = [ValueError("Request failed with status code 400: bad filter")]
    assert lit.search("planner-1", "query three") == []
    assert len(world.searches) == 7
    row = lit.drain_calls()[0]
    assert row["status"] == "error"
    assert "400" in row["note"]
    assert row["cost_dollars"] is None

    world.errors = [ConnectionError("connection reset")]
    assert len(lit.search("planner-1", "query four")) == 3
    assert len(world.searches) == 9
    assert lit.mode == "live"


def test_out_of_credits_turns_live_off(world, tmp_path):
    lit = ExaLiterature(lit_config(tmp_path), "live")
    world.errors = [ValueError("Request failed with status code 402: no credits")]
    assert lit.search("planner-1", "query one") == []
    assert lit.mode == "replay"
    assert "402" in lit.degraded_reason
    assert lit.search("planner-1", "query two") == []
    assert len(world.searches) == 1
    statuses = []
    for row in lit.drain_calls():
        statuses.append(row["status"])
    assert statuses == ["degraded", "error", "replay_miss"]


def test_live_call_cap(world, tmp_path):
    lit = ExaLiterature(lit_config(tmp_path, {"exa_max_live_calls": 1}), "live")
    assert len(lit.search("planner-1", "query one")) == 3
    assert lit.search("planner-1", "query two") == []
    assert len(world.searches) == 1
    assert lit.drain_calls()[1]["status"] == "capped"


def test_corrupt_cache_file_is_a_miss(world, tmp_path):
    config = lit_config(tmp_path)
    lit = ExaLiterature(config, "live")
    key = cache_key("search", lit.search_params("query one", 3), lit.sdk_version)
    path = cache_file(config["exa_cache_dir"], key)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("{half a file")
    assert ExaLiterature(config, "replay").search("planner-1", "query one") == []
    assert len(lit.search("planner-1", "query one")) == 3
    assert len(ExaLiterature(config, "replay").search("planner-1", "query one")) == 3


def test_search_never_raises(world, tmp_path):
    class Unprintable:
        def __str__(self):
            raise RuntimeError("cannot print")

    lit = ExaLiterature(lit_config(tmp_path), "live")
    assert lit.search("planner-1", Unprintable()) == []
    assert lit.drain_calls()[0]["status"] == "error"


# The modal.Dict tier.

def test_modal_tier_shares_hits(world, monkeypatch, tmp_path):
    store = {}

    def fake_open(name):
        assert name == "exa-cache"
        return FakeDict(store)

    monkeypatch.setattr(literature, "open_modal_dict", fake_open)
    writer = ExaLiterature(lit_config(tmp_path / "a", {"exa_modal_cache": True}), "live")
    first = writer.search("planner-1", "expert reuse", k=3)
    assert len(store) == 1

    reader = ExaLiterature(lit_config(tmp_path / "b", {"exa_modal_cache": True}), "replay")
    assert reader.search("planner-1", "expert reuse", k=3) == first
    assert reader.drain_calls()[0]["tier"] == "modal"

    backfill_config = lit_config(tmp_path / "c", {"exa_modal_cache": True})
    backfill = ExaLiterature(backfill_config, "live")
    assert backfill.search("planner-1", "expert reuse", k=3) == first
    assert len(world.searches) == 1
    assert ExaLiterature(lit_config(tmp_path / "c"), "replay").search("planner-1", "expert reuse", k=3) == first


def test_modal_tier_failures_are_soft(world, monkeypatch, tmp_path):
    def broken_open(name):
        raise RuntimeError("no modal credentials")

    monkeypatch.setattr(literature, "open_modal_dict", broken_open)
    lit = ExaLiterature(lit_config(tmp_path, {"exa_modal_cache": True}), "live")
    assert lit.modal_dict is None
    assert len(lit.search("planner-1", "query one")) == 3

    def failing_open(name):
        return FakeDict({}, fail=True)

    monkeypatch.setattr(literature, "open_modal_dict", failing_open)
    lit = ExaLiterature(lit_config(tmp_path / "other", {"exa_modal_cache": True}), "live")
    assert len(lit.search("planner-1", "query two")) == 3
    assert lit.modal_dict is None


# The calls log.

def test_drain_calls_returns_and_clears(world, tmp_path):
    lit = ExaLiterature(lit_config(tmp_path), "live")
    lit.search("planner-1", "query one")
    lit.search("planner-2", "query one")
    rows = lit.drain_calls()
    assert len(rows) == 2
    for row in rows:
        for name in ("endpoint", "query", "cache_hit", "cost_dollars", "mode"):
            assert name in row
        assert row["endpoint"] == "search"
        assert row["query"] == "query one"
    assert rows[0]["cache_hit"] is False
    assert rows[1]["cache_hit"] is True
    assert lit.drain_calls() == []


# Claims from deep-lite output.

def test_claims_from_response():
    data = {"results": [{"url": PAPER_URLS[0], "title": "Revisiting", "highlights": ["stitching works"]},
                        {"url": "https://huggingface.co/datasets/x", "title": "mirror", "highlights": ["q"]},
                        {"url": PAPER_URLS[1], "title": "Deeper layers", "highlights": ["prune half"]}],
            "output": {"content": {"claims": [
                {"claim": "Stitching works at all depths.", "layer_hint": "all", "model": "ResNet",
                 "strength": "strong"},
                {"claim": "No grounding for this one.", "layer_hint": "", "model": "", "strength": "huge"},
                {"layer_hint": "no claim text"}]},
                "grounding": [{"field": "claims[0].claim", "confidence": "high",
                               "citations": [{"url": PAPER_URLS[0], "title": "Revisiting"},
                                             {"url": "https://huggingface.co/datasets/x", "title": "m"}]}]}}
    claims = literature.claims_from_response(data, "model_stitching", literature.DATASET_MIRRORS)
    assert len(claims) == 2
    assert claims[0]["grounded"] is True
    assert claims[0]["cites"] == [{"url": PAPER_URLS[0], "title": "Revisiting", "quote": "stitching works"}]
    assert claims[0]["strength"] == "strong"
    assert claims[0]["topic"] == "model_stitching"
    assert claims[1]["grounded"] is False
    assert claims[1]["strength"] == "weak"
    fallback_urls = []
    for cite in claims[1]["cites"]:
        fallback_urls.append(cite["url"])
    assert fallback_urls == [PAPER_URLS[0], PAPER_URLS[1]]

    as_text = dict(data)
    as_text["output"] = {"content": json.dumps(data["output"]["content"]), "grounding": []}
    assert len(literature.claims_from_response(as_text, "t", literature.DATASET_MIRRORS)) == 2
    assert literature.claims_from_response({"results": []}, "t", []) == []
    assert literature.claims_from_response({"output": {"content": "not json"}}, "t", []) == []


def test_cost_estimate_uses_list_prices():
    lite = {"type": "lite", "num_results": 3, "contents": {"highlights": True}}
    assert literature.estimate_cost("search", lite) == pytest.approx(0.007)
    auto = {"type": "auto", "num_results": 5, "contents": {"highlights": {"max_characters": 1200}}}
    assert literature.estimate_cost("search", auto) == pytest.approx(0.012)
    deep = {"type": "deep-lite", "num_results": 12, "contents": {}}
    assert literature.estimate_cost("search", deep) == pytest.approx(0.012 + 2 * 0.001)


# scripts/build_priors.py

def test_build_priors_groups_parts_by_model_kind_band_and_skill():
    build = load_build_priors()
    known = {"Qwen/Qwen2.5-Math-1.5B": 28}
    assert build.layer_band({"model": "Qwen/Qwen2.5-Math-1.5B", "layer": 6}, known) == "early"
    assert build.layer_band({"model": "Qwen/Qwen2.5-Math-1.5B", "layer": 14}, known) == "middle"
    assert build.layer_band({"model": "Qwen/Qwen2.5-Math-1.5B", "layer": 20}, known) == "late"
    assert build.layer_band({"model": "Qwen/Qwen2.5-Math-1.5B", "layers": [10, 12]}, known) == "middle"
    assert build.layer_band({"model": "Qwen/Qwen2.5-Math-1.5B"}, known) == "all"
    assert build.layer_band({"model": "Unknown/Model", "layer": 3}, known) == "unknown"
    assert build.layer_band({"model": "Unknown/Model", "layer": 3, "num_layers": 4}, known) == "late"
    assert build.skill_of({"trained_on": "STUB: GSM8K train activations"}) == "GSM8K"
    assert build.skill_of({"trained_on": "MMLU-Pro biology SFT"}) == "MMLU-Pro biology"
    assert build.skill_of({"trained_on": "BBEH disambiguation_qa SFT"}) == "BBEH disambiguation qa"
    pilot = ("top 256 MLP neurons by mean |activation| (down_proj input) on 64 GSM8K heal-split texts "
             "(GSM8K train minus the 200-item search split)")
    assert build.skill_of({"trained_on": pilot}) == "GSM8K"
    assert build.skill_of({}) == ""
    records = [{"id": "b", "type": "mlp_cluster", "model": "Qwen/Qwen2.5-Math-1.5B", "layer": 14,
                "trained_on": "GSM8K"},
               {"id": "a", "type": "mlp_cluster", "model": "Qwen/Qwen2.5-Math-1.5B", "layer": 15,
                "trained_on": "GSM8K"},
               {"id": "c", "type": "mlp_cluster", "model": "Qwen/Qwen2.5-Math-1.5B", "layer": 25,
                "trained_on": "GSM8K"},
               {"id": "h", "type": "host"}]
    groups, skipped = build.group_parts(records, known)
    assert len(groups) == 2
    assert groups[0]["part_ids"] == ["a", "b"]
    assert groups[0]["band"] == "middle"
    assert skipped == ["h"]
    assert build.component_query(groups[0]) == "MLP neurons in the middle layers of Qwen2.5-Math-1.5B and their role in GSM8K"


def test_build_priors_dry_run_needs_no_key_and_no_network(world, monkeypatch, tmp_path, capsys):
    monkeypatch.delenv("EXA_API_KEY", raising=False)
    build = load_build_priors()
    catalog = write_catalog(str(tmp_path / "catalog.jsonl"))
    cache_dir = tmp_path / "exa_cache"
    out = tmp_path / "priors.json"
    code = build.main([catalog, "--dry-run", "--cache-dir", str(cache_dir), "--out", str(out)])
    assert code == 0
    text = capsys.readouterr().out
    # mask_a, mask_b, and one group for the two middle-layer Qwen math clusters; the host record is skipped.
    assert "Planned calls: 11 (3 component searches, 8 topical deep-lite searches)" in text
    assert "List-price estimate, cold cache: $0.172" in text
    assert "host_record" in text
    assert world.clients == 0
    assert world.searches == []
    assert not out.exists()
    assert not cache_dir.exists()


def test_build_priors_live_then_replay_gives_the_same_sha(world, tmp_path, capsys):
    build = load_build_priors()
    catalog = write_catalog(str(tmp_path / "catalog.jsonl"))
    cache_dir = tmp_path / "exa_cache"
    out = tmp_path / "priors.json"
    common = ["--cache-dir", str(cache_dir), "--out", str(out), "--calls-log", str(tmp_path / "calls.jsonl")]
    assert build.main([catalog] + common) == 0
    assert len(world.searches) == 11
    deep = 0
    for call in world.searches:
        if call["type"] == "deep-lite":
            deep += 1
            assert call["output_schema"]["properties"]["claims"]["type"] == "array"
            assert call["category"] == "publication"
    assert deep == 8
    with open(out, "rb") as handle:
        live_bytes = handle.read()
    pack = json.loads(live_bytes.decode("utf-8"))
    assert sorted(pack["components"].keys()) == ["mask_a", "mask_b", "qmath_l14_c0", "qmath_l15_c1"]
    assert pack["components"]["qmath_l14_c0"] == pack["components"]["qmath_l15_c1"]
    assert len(pack["components"]["mask_a"]) == 5
    assert len(pack["topics"]) == 8
    assert pack["topics"][0]["grounded"] is True
    assert pack["misses"] == []
    files = 0
    for folder in os.listdir(cache_dir):
        files += len(os.listdir(os.path.join(cache_dir, folder)))
    assert files == 11

    assert build.main([catalog, "--mode", "replay"] + common) == 0
    assert len(world.searches) == 11
    with open(out, "rb") as handle:
        assert handle.read() == live_bytes
    lit = ExaLiterature({"exa_cache_dir": str(cache_dir), "priors_path": str(out)}, "replay")
    sha = hashlib.sha256(live_bytes).hexdigest()
    assert lit.priors()["sha"] == sha
    assert "priors_sha " + sha in capsys.readouterr().out


def test_build_priors_gives_the_same_pack_for_lf_and_crlf_catalogs(world, monkeypatch, tmp_path, capsys):
    # Two checkouts of one repo: the catalog is LF in one and CRLF in the other.
    build = load_build_priors()
    lf_root = tmp_path / "lf_checkout"
    crlf_root = tmp_path / "crlf_checkout"
    lf_root.mkdir()
    crlf_root.mkdir()
    # write_catalog uses text mode, so the file may come out with either ending.
    first = write_catalog(str(tmp_path / "catalog.jsonl"))
    with open(first, "rb") as handle:
        lf_bytes = handle.read().replace(b"\r\n", b"\n")
    assert b"\r" not in lf_bytes
    lf_catalog = str(lf_root / "catalog.jsonl")
    crlf_catalog = str(crlf_root / "catalog.jsonl")
    with open(lf_catalog, "wb") as handle:
        handle.write(lf_bytes)
    with open(crlf_catalog, "wb") as handle:
        handle.write(lf_bytes.replace(b"\n", b"\r\n"))
    assert (crlf_root / "catalog.jsonl").read_bytes() != lf_bytes
    assert build.file_sha256(crlf_catalog) == build.file_sha256(lf_catalog)
    cache_dir = tmp_path / "exa_cache"
    common = ["--cache-dir", str(cache_dir), "--calls-log", str(tmp_path / "calls.jsonl")]
    lf_out = lf_root / "priors.json"
    crlf_out = crlf_root / "priors.json"
    monkeypatch.setattr(build, "REPO_ROOT", str(lf_root))
    assert build.main([lf_catalog, "--out", str(lf_out)] + common) == 0
    monkeypatch.setattr(build, "REPO_ROOT", str(crlf_root))
    assert build.main([crlf_catalog, "--mode", "replay", "--out", str(crlf_out)] + common) == 0
    assert len(world.searches) == 11
    assert crlf_out.read_bytes() == lf_out.read_bytes()
    printed = capsys.readouterr().out
    sha = ExaLiterature({"exa_cache_dir": str(cache_dir), "priors_path": str(lf_out)}, "replay").priors()["sha"]
    assert printed.count("priors_sha " + sha) == 2


def test_build_priors_refuses_a_live_build_over_the_cost_cap(world, tmp_path):
    build = load_build_priors()
    catalog = write_catalog(str(tmp_path / "catalog.jsonl"))
    code = build.main([catalog, "--cache-dir", str(tmp_path / "exa_cache"), "--out", str(tmp_path / "p.json"),
                       "--max-cost-dollars", "0.01"])
    assert code == 1
    assert world.searches == []
    assert not (tmp_path / "p.json").exists()


def test_build_priors_without_a_key_writes_nothing_from_an_empty_cache(world, monkeypatch, tmp_path):
    monkeypatch.delenv("EXA_API_KEY", raising=False)
    build = load_build_priors()
    catalog = write_catalog(str(tmp_path / "catalog.jsonl"))
    code = build.main([catalog, "--cache-dir", str(tmp_path / "exa_cache"), "--out", str(tmp_path / "p.json"),
                       "--calls-log", str(tmp_path / "calls.jsonl")])
    assert code == 1
    assert world.clients == 0
    assert not (tmp_path / "p.json").exists()
