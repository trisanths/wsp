"""Tests for the benchmark research agent. No network: the search provider and the LLM are faked."""

import json
import socket
import threading
import time

import pytest
from fastapi.testclient import TestClient

from studio.server import bench_research
from studio.server import specs as specs_mod
from studio.server.app import app

REAL_URL = "https://example.org/legal-leaderboard"
REAL_TEXT = ("Table 3. MMLU-Pro law, accuracy\n\n| Model | Size | MMLU-Pro law |\n| --- | --- | --- |\n"
             "| Qwen2.5-1.5B | 1.5B | 31.0 |\n| Phi-3-mini | 3.8B | 45.2 |\n")
PLAN = {"plan": "I will look for legal question answering benchmarks and what small models score.",
        "queries": ["legal question answering benchmark", "LegalBench small model scores"]}
SYNTH = {
    "reasoning": "Legal questions are knowledge heavy, so MMLU style exams fit, and instruction following guards.",
    "benchmarks": [
        {"name": "MMLU professional law", "measures": "law exam questions", "why": "closest runnable exam",
         "runnable_id": "mmlupro:law", "kind": "target",
         "reference_scores": [
             {"model": "Qwen2.5-1.5B", "params_b": 1.5, "score": 31.0, "source_url": REAL_URL,
              "metric": "accuracy", "label": "MMLU-Pro law", "quote": "| Qwen2.5-1.5B | 1.5B | 31.0 |"},
             {"model": "Invented-2B", "params_b": 2, "score": 0.55, "source_url": "https://invented.example/x"}],
         "suggested_target": 0.35},
        {"name": "LegalBench", "measures": "legal reasoning tasks", "why": "the standard legal suite",
         "runnable_id": "legalbench", "kind": "target", "reference_scores": [], "suggested_target": None},
        {"name": "IFEval", "measures": "instruction following", "why": "keeps answers well formed",
         "runnable_id": "if_guard", "kind": "guard", "reference_scores": [], "suggested_target": None},
    ],
}


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    """Any test that tries to open a network connection fails. The providers and the LLM are faked in each test."""
    def refuse(self, *args, **kwargs):
        raise AssertionError("a research test tried to reach the network")

    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)

    def no_llm(messages, schema, max_tokens=2500):
        raise AssertionError("a research test called the real LLM")

    monkeypatch.setattr(bench_research, "ask_llm", no_llm)
    monkeypatch.setattr(bench_research, "search_exa", refuse)
    monkeypatch.setattr(bench_research, "search_exa_scores", refuse)
    monkeypatch.setattr(bench_research, "search_openai_web", refuse)


def fake_search(provider="exa"):
    def search(query, run):
        return {"summary": "", "results": [
            {"title": "Legal leaderboard", "url": REAL_URL, "text": REAL_TEXT},
            {"title": "Paper " + query[:8], "url": "https://example.org/p/" + str(abs(hash(query)) % 1000),
             "text": "more"}]}
    return search


SCORE_URL = "https://example.org/law-results"
SCORES = {"scores": [
    {"benchmark": "MMLU professional law", "model": "SmolLM2-1.7B", "params_b": 1.7, "score": 28.4, "source_url": SCORE_URL,
     "metric": "accuracy", "label": "MMLU law category",
     "quote": "SmolLM2-1.7B reaches 28.4 on the law category"},
    {"benchmark": "MMLU professional law", "model": "Made-Up-1B", "params_b": 1.0, "score": 0.61, "source_url": SCORE_URL,
     "metric": "accuracy", "label": "MMLU law category", "quote": "Made-Up-1B reaches 61.0 on the law category"},
]}


def fake_score_search(name, size_b):
    return [{"title": "Law results", "url": SCORE_URL,
             "text": "MMLU law category. Table 2. SmolLM2-1.7B  reaches 28.4 on the law category. Other rows follow."}]


def fake_llm(messages, schema, max_tokens=2500):
    text = json.dumps(schema)
    if "queries" in text:
        return json.dumps(PLAN)
    if "scores" in text:
        return json.dumps(SCORES)
    return json.dumps(SYNTH)


def fail_draft(*args, **kwargs):
    """The research tests never need an LLM-drafted spec, so the draft falls back to rules with no network."""
    raise RuntimeError("no LLM draft in tests")


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("WSP_STUDIO_OUT", str(tmp_path))
    monkeypatch.setenv("WSP_STUDIO_BACKEND", "stub")
    monkeypatch.setenv("WSP_STUDIO_LLM", "openai_compat")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key-not-used")
    monkeypatch.delenv("EXA_API_KEY", raising=False)
    monkeypatch.setattr(specs_mod, "llm_draft", fail_draft)
    monkeypatch.setattr(bench_research, "choose_provider", lambda: "exa")
    monkeypatch.setattr(bench_research, "search_exa", fake_search())
    monkeypatch.setattr(bench_research, "search_exa_scores", fake_score_search)
    monkeypatch.setattr(bench_research, "ask_llm", fake_llm)
    bench_research._runs.clear()
    return TestClient(app)


def wait_research(client, spec_id, timeout=20.0):
    end = time.time() + timeout
    while time.time() < end:
        body = client.get("/api/spec/" + spec_id + "/research").json()
        if body["status"] != "running":
            return body
        time.sleep(0.05)
    raise AssertionError("research did not finish")


def draft(client, prompt="A small model for answering legal questions, under 2B parameters"):
    response = client.post("/api/spec", data={"prompt": prompt})
    assert response.status_code == 200, response.text
    return response.json()


def test_run_proposes_and_validates(client):
    spec = draft(client)
    assert spec["benchmark_research"]["status"] in ("running", "done")
    research = wait_research(client, spec["id"])
    assert research["status"] == "done", research
    assert research["provider"] == "exa"
    kinds = {step["kind"] for step in research["steps"]}
    assert {"think", "search", "read", "pick"} <= kinds
    by_name = {p["name"]: p for p in research["proposals"]}
    assert by_name["MMLU professional law"]["runnable_id"] == "mmlupro:law"
    assert by_name["LegalBench"]["runnable_id"] is None
    assert by_name["LegalBench"]["in_spec"] is False
    scores = {s["model"]: s for s in by_name["MMLU professional law"]["reference_scores"]}
    assert scores["Qwen2.5-1.5B"]["sourced"] is True
    assert scores["Qwen2.5-1.5B"]["score"] == 0.31
    assert scores["Invented-2B"]["sourced"] is False
    assert len(research["sources"]) >= 2

    stored = client.get("/api/spec/" + spec["id"]).json()
    ids = {item["id"]: item for item in stored["benchmarks"]}
    assert "mmlupro:law" in ids and ids["mmlupro:law"]["research"] is True
    assert ids["mmlupro:law"]["target"] == 0.35
    assert len(ids["mmlupro:law"]["reference_scores"]) == 4  # two from the first pass, two from the score pass
    assert "legalbench" not in ids
    assert ids["if_guard"]["kind"] == "guard"
    assert stored["limits"] == spec["limits"]
    assert stored["title"] == spec["title"]


def test_reference_scores_survive_a_put(client):
    spec = draft(client)
    wait_research(client, spec["id"])
    stored = client.get("/api/spec/" + spec["id"]).json()
    response = client.put("/api/spec/" + spec["id"], json=stored)
    assert response.status_code == 200
    kept = {item["id"]: item for item in response.json()["benchmarks"]}
    assert len(kept["mmlupro:law"]["reference_scores"]) == 4
    assert response.json()["benchmark_research"]["status"] == "done"


def test_percent_scores_and_clean_proposals():
    assert bench_research.clean_score(31) == 0.31
    assert bench_research.clean_score(0.4) == 0.4
    assert bench_research.clean_score(250) is None
    raw = [{"name": "A", "runnable_id": "gsm8k"}, {"name": "A again", "runnable_id": "gsm8k"},
           {"name": "a", "runnable_id": None}, {"name": "B", "runnable_id": "nope"}]
    out = bench_research.clean_proposals(raw, ["gsm8k"], [])
    assert [p["name"] for p in out] == ["A", "B"]
    assert out[1]["runnable_id"] is None


def test_no_llm_is_skipped(client, monkeypatch):
    monkeypatch.setenv("WSP_STUDIO_LLM", "none")
    spec = draft(client)
    assert spec["benchmark_research"]["status"] == "skipped"
    assert spec["benchmark_research"]["error"]
    body = client.get("/api/spec/" + spec["id"] + "/research").json()
    assert body["status"] == "skipped"
    assert len(spec["benchmarks"]) >= 2


def test_second_start_is_409(client, monkeypatch):
    gate = threading.Event()

    def slow_llm(messages, schema, max_tokens=2500):
        gate.wait(10)
        return fake_llm(messages, schema, max_tokens)

    monkeypatch.setattr(bench_research, "ask_llm", slow_llm)
    spec = draft(client)
    response = client.post("/api/spec/" + spec["id"] + "/research")
    assert response.status_code == 409
    gate.set()
    assert wait_research(client, spec["id"])["status"] == "done"
    response = client.post("/api/spec/" + spec["id"] + "/research")
    assert response.status_code == 200
    assert response.json() == {"status": "running"}
    assert wait_research(client, spec["id"])["status"] == "done"


def test_get_research_404(client):
    assert client.get("/api/spec/sp_nope/research").status_code == 404
    assert client.post("/api/spec/sp_nope/research").status_code == 404


def test_search_failure_is_a_step_and_none_provider(client, monkeypatch):
    def broken(query, run):
        raise RuntimeError("network down")

    monkeypatch.setattr(bench_research, "search_exa", broken)
    spec = draft(client)
    research = wait_research(client, spec["id"])
    assert research["status"] == "done"
    assert any("failed" in step["text"] for step in research["steps"])

    monkeypatch.setattr(bench_research, "choose_provider", lambda: "none")
    client.post("/api/spec/" + spec["id"] + "/research")
    research = wait_research(client, spec["id"])
    assert research["provider"] == "none"
    assert any("No web search available" in step["text"] for step in research["steps"])


def test_timeout_records_error(client, monkeypatch):
    monkeypatch.setattr(bench_research, "RUN_TIMEOUT_S", 0.3)
    release = threading.Event()

    def stuck(messages, schema, max_tokens=2500):
        release.wait(5)
        raise RuntimeError("stopped after the test")

    monkeypatch.setattr(bench_research, "ask_llm", stuck)
    spec = draft(client)
    research = wait_research(client, spec["id"])
    release.set()
    assert research["status"] == "error"
    assert "longer than" in research["error"]


def test_second_pass_scores_need_a_quote_from_the_page(client):
    spec = draft(client)
    research = wait_research(client, spec["id"])
    assert research["status"] == "done"
    law = [p for p in research["proposals"] if p["runnable_id"] == "mmlupro:law"][0]
    by_model = {entry["model"]: entry for entry in law["reference_scores"]}
    # The quote is on the page (whitespace differs), so the score is sourced and keeps its quote.
    assert by_model["SmolLM2-1.7B"]["sourced"] is True
    assert by_model["SmolLM2-1.7B"]["score"] == 0.284
    assert "28.4" in by_model["SmolLM2-1.7B"]["quote"]
    # The page never says this, so the score is only an estimate.
    assert by_model["Made-Up-1B"]["sourced"] is False
    assert "quote" not in by_model["Made-Up-1B"]
    urls = [source["url"] for source in research["sources"]]
    assert SCORE_URL in urls
    texts = [step["text"] for step in research["steps"]]
    assert any(text.startswith("Searched for published scores on") for text in texts)


# Rules for tying a score to its model, benchmark and column.

TABLE_URL = "https://example.org/coder-results"
TABLE_TEXT = (
    "## Results\n\nTable 4. Pass at 1 on code benchmarks\n\n"
    "| Model | Size | HumanEval | MBPP |\n| --- | --- | --- | --- |\n"
    "| Coder-1.5B | 1.5B | 43.9 | 69.2 |\n"
    "| Coder-1.5B + CoT | 1.5B | 51.3 | 70.1 |\n"
    "| Big-70B | 70B | 88.4 | 91.0 |\n")
PAGES = {bench_research.url_key(TABLE_URL): {"raw": TABLE_TEXT, "title": "Coder results", "url": TABLE_URL}}


def code_proposal(name="HumanEval", runnable_id="humaneval"):
    return {"name": name, "runnable_id": runnable_id, "kind": "target", "reference_scores": []}


def make_entry(model, score, quote, label, params=1.5, limit=1.5, request="a coder", name="HumanEval", rid="humaneval",
               url=TABLE_URL):
    item = {"model": model, "params_b": params, "score": score, "quote": quote, "label": label, "source_url": url,
            "metric": "pass@1"}
    dropped = {}
    entry = bench_research.score_entry(item, code_proposal(name, rid), PAGES, limit, request, dropped)
    return entry, dropped


def test_row_and_column_match_is_sourced():
    entry, dropped = make_entry("Coder-1.5B", 43.9, "| Coder-1.5B | 1.5B | 43.9 |", "HumanEval")
    assert entry["sourced"] is True
    assert entry["metric"] == "pass@1"
    assert entry["source_kind"] == "other"
    assert dropped == {}


def test_number_from_the_neighbor_column_is_dropped():
    # 69.2 is the MBPP column of the same row. The label says HumanEval, so the column header does not match.
    entry, dropped = make_entry("Coder-1.5B", 69.2, "| Coder-1.5B | 1.5B | 43.9 | 69.2 |", "HumanEval")
    assert entry is None
    assert dropped == {"benchmark_not_near_number": 1}
    # The same number with the right label is fine.
    entry, _dropped = make_entry("Coder-1.5B", 69.2, "| Coder-1.5B | 1.5B | 43.9 | 69.2 |", "MBPP", name="MBPP", rid=None)
    assert entry["sourced"] is True


def test_number_from_another_models_row_is_dropped():
    entry, dropped = make_entry("Coder-1.5B", 88.4, "| Big-70B | 70B | 88.4 |", "HumanEval")
    assert entry is None
    assert dropped == {"model_not_before_number": 1}


def test_method_and_fine_tune_variants_are_dropped():
    entry, dropped = make_entry("Coder-1.5B + CoT", 51.3, "| Coder-1.5B + CoT | 1.5B | 51.3 |", "HumanEval")
    assert entry is None and dropped == {"method_variant": 1}
    entry, dropped = make_entry("Coder-1.5B", 51.3, "| Coder-1.5B + CoT | 1.5B | 51.3 |", "HumanEval")
    assert entry is None and dropped == {"variant": 1}


def test_benchmark_must_be_named_near_the_number():
    # The LLM's label does not count. The page words around the number must name the benchmark.
    entry, dropped = make_entry("Coder-1.5B", 43.9, "| Coder-1.5B | 1.5B | 43.9 |", "MBPP", name="MBPP", rid=None)
    assert entry is None
    assert dropped == {"benchmark_not_near_number": 1}
    entry, dropped = make_entry("Coder-1.5B", 43.9, "| Coder-1.5B | 1.5B | 43.9 |", "")
    assert entry["sourced"] is True and "label" not in entry
    entry, dropped = make_entry("Coder-1.5B", 43.9, "| Coder-1.5B | 1.5B | 43.9 |", "Results")
    assert entry["label"] == "Results"


def test_table_text_above_counts_only_when_the_header_names_no_benchmark():
    url = "https://example.org/open"
    text = "Results on GSM8K, exact match.\n\n| Model | Accuracy |\n| --- | --- |\n| Tiny-1B | 55.5 |\n"
    pages = {bench_research.url_key(url): {"raw": text, "title": "t", "url": url}}
    item = {"model": "Tiny-1B", "params_b": 1, "score": 55.5, "quote": "| Tiny-1B | 55.5 |", "source_url": url}
    assert bench_research.score_entry(item, code_proposal("GSM8K", "gsm8k"), pages, 1.5, "", {})["sourced"] is True
    # A header that names another benchmark means the words above the table do not name this column's benchmark.
    text = "Results on GSM8K.\n\n| Model | MBPP |\n| --- | --- |\n| Tiny-1B | 55.5 |\n"
    pages = {bench_research.url_key(url): {"raw": text, "title": "t", "url": url}}
    dropped = {}
    assert bench_research.score_entry(item, code_proposal("GSM8K", "gsm8k"), pages, 1.5, "", dropped) is None
    assert dropped == {"benchmark_not_near_number": 1}
    # A table with no header row cannot say which column a number is in.
    text = "Results on GSM8K.\n\n| Tiny-1B | 1B | 55.5 | 60.1 |\n| Other-1B | 1B | 20.5 | 30.1 |\n"
    pages = {bench_research.url_key(url): {"raw": text, "title": "t", "url": url}}
    dropped = {}
    item["quote"] = "| Tiny-1B | 1B | 55.5 |"
    assert bench_research.score_entry(item, code_proposal("GSM8K", "gsm8k"), pages, 1.5, "", dropped) is None
    assert dropped == {"benchmark_not_near_number": 1}


def test_a_quote_may_skip_a_cell_but_not_more_than_three():
    raw = "| Model | Size | Pass1 |\n| --- | --- | --- |\n| Coder-1.5B | 1.5B | 0.44 |\n"
    assert bench_research.locate_quote(raw, "Coder-1.5B 0.44") is not None
    assert bench_research.locate_quote(raw, "Coder-1.5B 0.44").group(0) == "Coder-1.5B | 1.5B | 0.44"
    far = "Coder-1.5B one two three four 0.44"
    assert bench_research.locate_quote(far, "Coder-1.5B 0.44") is None
    # A number is matched whole: 0.44 is not inside 10.449.
    assert bench_research.locate_quote("Coder-1.5B 10.449", "Coder-1.5B 0.44") is None


def test_model_card_prose_names_the_model_by_its_page():
    url = "https://huggingface.co/org/NPC-Coder-1.5B"
    text = "Benchmark: HumanEval (instruct, pass@1, greedy): 65.9%."
    pages = {bench_research.url_key(url): {"raw": text, "title": "org/NPC-Coder-1.5B", "url": url}}
    item = {"model": "NPC Coder 1.5B", "params_b": 1.5, "score": 65.9, "quote": text, "source_url": url}
    assert bench_research.score_entry(item, code_proposal(), pages, 1.5, "", {})["sourced"] is True
    # The same sentence on a page about another model does not name this one.
    other = "https://huggingface.co/org/Other-Coder-3B"
    pages = {bench_research.url_key(other): {"raw": text, "title": "org/Other-Coder-3B", "url": other}}
    item["source_url"] = other
    dropped = {}
    assert bench_research.score_entry(item, code_proposal(), pages, 1.5, "", dropped) is None
    assert dropped == {"model_not_before_number": 1}


def test_mmlu_pro_needs_the_subject_in_the_label():
    text = "| Dataset | Tiny-1B |\n| --- | --- |\n| MMLU-Pro | 14.7 |\n| MMLU-Pro biology | 30.2 |\n"
    url = "https://example.org/pro"
    pages = {bench_research.url_key(url): {"raw": text, "title": "Tiny", "url": url}}
    proposal = code_proposal("MMLU Pro biology", "mmlupro:biology")
    item = {"model": "Tiny-1B", "params_b": 1, "score": 14.7, "quote": "| MMLU-Pro | 14.7 |", "label": "MMLU-Pro",
            "source_url": url}
    dropped = {}
    assert bench_research.score_entry(item, proposal, pages, 1.5, "", dropped) is None
    assert dropped == {"benchmark_not_near_number": 1}
    item.update({"score": 30.2, "quote": "| MMLU-Pro biology | 30.2 |", "label": "MMLU-Pro biology"})
    entry = bench_research.score_entry(item, proposal, pages, 1.5, "", {})
    assert entry["sourced"] is True


def test_a_model_card_table_with_models_as_columns_is_read_by_column():
    text = "| Benchmark | Tiny-1B | Other-3B |\n| --- | --- | --- |\n| GSM8K | 70.9 | 47.9 |\n"
    url = "https://example.org/card"
    pages = {bench_research.url_key(url): {"raw": text, "title": "card", "url": url}}
    proposal = code_proposal("GSM8K", "gsm8k")
    item = {"model": "Tiny-1B", "params_b": 1, "score": 70.9, "quote": "| GSM8K | 70.9 | 47.9 |", "label": "GSM8K",
            "source_url": url}
    assert bench_research.score_entry(item, proposal, pages, 1.5, "", {})["sourced"] is True
    # 47.9 sits under Other-3B, not Tiny-1B.
    item.update({"score": 47.9})
    dropped = {}
    assert bench_research.score_entry(item, proposal, pages, 1.5, "", dropped) is None
    assert dropped == {"model_not_before_number": 1}


def test_flat_text_rejects_a_number_after_another_score():
    text = "GSM8K results. Tiny-1B 31.5 and 36.9 on the test set."
    url = "https://example.org/flat"
    pages = {bench_research.url_key(url): {"raw": text, "title": "flat", "url": url}}
    proposal = code_proposal("GSM8K", "gsm8k")
    item = {"model": "Tiny-1B", "params_b": 1, "score": 36.9, "quote": "Tiny-1B 31.5 and 36.9 on the test set",
            "label": "GSM8K", "source_url": url}
    dropped = {}
    assert bench_research.score_entry(item, proposal, pages, 1.5, "", dropped) is None
    assert dropped == {"other_number_between": 1}
    item.update({"score": 31.5, "quote": "Tiny-1B 31.5 and 36.9"})
    assert bench_research.score_entry(item, proposal, pages, 1.5, "", {})["sourced"] is True


def test_size_rule_drops_far_larger_models_unless_the_request_names_them():
    quote = "| Big-70B | 70B | 88.4 |"
    entry, dropped = make_entry("Big-70B", 88.4, quote, "HumanEval", params=70)
    assert entry is None and dropped == {"too_large": 1}
    entry, dropped = make_entry("Big-70B", 88.4, quote, "HumanEval", params=70, request="shrink Big-70B to 3B")
    assert entry["sourced"] is True
    # The size is read from the name when the agent gives none. A closed model has no size and is dropped.
    entry, dropped = make_entry("Big-70B", 88.4, quote, "HumanEval", params=None)
    assert entry is None and dropped == {"too_large": 1}
    entry, dropped = make_entry("GPT-4o", 88.4, quote, "HumanEval", params=None)
    assert entry is None and dropped == {"closed_or_huge_model": 1}
    assert bench_research.guess_params("legal-slm-500m") == 0.5
    assert bench_research.guess_params("Qwen3-1.7B") == 1.7
    assert bench_research.guess_params("Qwen3") is None


def test_no_quote_or_unread_page_stays_an_estimate():
    entry, _ = make_entry("Coder-1.5B", 43.9, "", "HumanEval")
    assert entry["sourced"] is False
    entry, _ = make_entry("Coder-1.5B", 43.9, "| Coder-1.5B | 1.5B | 43.9 |", "HumanEval", url="https://other.example/x")
    assert entry["sourced"] is False
    entry, _ = make_entry("Coder-1.5B", 43.9, "| Coder-1.5B | 1.5B | 99.9 |", "HumanEval")
    assert entry["sourced"] is False


def test_same_model_is_kept_once_and_the_better_source_wins():
    proposal = code_proposal()
    first = {"model": "Qwen2.5-Coder-1.5B", "score": 0.5, "sourced": True, "source_url": "https://sota2.com/x",
             "source_kind": "aggregator"}
    second = {"model": "Qwen2.5Coder 1.5B", "score": 0.44, "sourced": True, "source_url": "https://arxiv.org/abs/1",
              "source_kind": "paper"}
    third = {"model": "qwen2.5-coder-1.5b", "score": 0.9, "sourced": True, "source_url": "https://sota2.com/y",
             "source_kind": "aggregator"}
    assert bench_research.add_score(proposal, first) is True
    assert bench_research.add_score(proposal, second) is False
    assert bench_research.add_score(proposal, third) is False
    assert len(proposal["reference_scores"]) == 1
    assert proposal["reference_scores"][0]["score"] == 0.44
    # A bracketed note does not make a new model. A different size does.
    note = {"model": "Qwen2.5-Coder-1.5B (classifier)", "score": 0.4, "sourced": False, "source_url": None}
    other = {"model": "Qwen2.5-Coder-3B", "score": 0.5, "sourced": False, "source_url": None}
    assert bench_research.add_score(proposal, note) is False
    assert bench_research.add_score(proposal, other) is True


def test_source_kinds_and_order():
    assert bench_research.source_kind("https://arxiv.org/abs/2409.12186") == "paper"
    assert bench_research.source_kind("https://huggingface.co/Qwen/Qwen2-0.5B") == "model card"
    assert bench_research.source_kind("https://huggingface.co/spaces/open-llm-leaderboard/x") == "leaderboard"
    assert bench_research.source_kind("https://github.com/org/repo") == "repo"
    assert bench_research.source_kind("https://www.sota2.com/research/sota/x") == "aggregator"
    assert bench_research.source_kind("https://example.org/page") == "other"
    results = [{"url": "https://sota2.com/a"}, {"url": "https://example.org/b"}, {"url": "https://arxiv.org/abs/1"}]
    ordered = [item["url"] for item in bench_research.by_source(results)]
    assert ordered == ["https://arxiv.org/abs/1", "https://example.org/b", "https://sota2.com/a"]


def test_metric_confidence_and_comparable_flags():
    proposal = code_proposal()
    proposal["reference_scores"] = [
        {"model": "A", "score": 0.4, "sourced": True, "source_url": "https://arxiv.org/a", "source_kind": "paper",
         "metric": "BLEU"},
        {"model": "B", "score": 0.5, "sourced": True, "source_url": "https://arxiv.org/b", "source_kind": "paper",
         "metric": "BLEU"},
        {"model": "C", "score": 0.7, "sourced": True, "source_url": "https://arxiv.org/c", "source_kind": "paper",
         "metric": "chrF"},
        {"model": "D", "score": 0.3, "sourced": False, "source_url": None, "source_kind": None, "metric": None}]
    bench_research.finish_scores(proposal, 1.5)
    by_model = {entry["model"]: entry for entry in proposal["reference_scores"]}
    assert proposal["metric"] == "BLEU"
    assert by_model["C"]["comparable"] is False
    assert "comparable" not in by_model["A"]
    assert proposal["confidence"] == "high"
    assert "3 sourced scores" in proposal["confidence_note"]
    assert proposal["reference_scores"][-1]["model"] == "D"
    empty = code_proposal()
    bench_research.finish_scores(empty, 1.5)
    assert empty["confidence"] == "low"
    guard = code_proposal("IFEval", "if_guard")
    guard["kind"] = "guard"
    bench_research.finish_scores(guard, 1.5)
    assert guard["confidence"] == "medium"


def test_queries_always_ask_for_small_models():
    spec = {"title": "Legal helper", "limits": {"max_total_params_b": 1.5}}
    out = bench_research.clean_queries(["LegalBench leaderboard", "LexGLUE results"], spec)
    assert len(out) == 3
    assert "1.5B" in out[-1]
    full = bench_research.clean_queries(["a", "b", "c", "d"], spec)
    assert len(full) == bench_research.MAX_QUERIES
    assert "1.5B" in full[-1]
    kept = bench_research.clean_queries(["legal benchmarks", "small model legal results"], spec)
    assert kept == ["legal benchmarks", "small model legal results"]


def test_prompts_name_standard_benchmarks_and_ask_for_evidence_fields():
    spec = {"title": "Coder", "prompt": "a coder", "limits": {"max_total_params_b": 1.5}}
    plan = bench_research.plan_messages(spec)[0]["content"]
    assert "MBPP" in plan and "FLORES" in plan and "IFEval" in plan
    synth = bench_research.synth_messages(spec, [], [("gsm8k", "math")])[0]["content"]
    for word in ("metric", "label", "quote", "LegalBench"):
        assert word in synth
    score = bench_research.score_messages(spec, ["GSM8K"], {"GSM8K": []})[0]["content"]
    assert "label" in score and "metric" in score


def test_table_info_reads_headers_with_a_merged_header_row():
    raw = ("Text before.\n\n| Model | HumanEval | | MBPP | |\n| | HE | HE+ | MBPP | MBPP+ |\n| --- | --- | --- | --- | --- |\n"
           "| Coder-3B | 52.4 | 42.7 | 72.2 | 61.4 |\n")
    pos = raw.index("42.7")
    info = bench_research.table_info(raw, pos)
    assert "HumanEval" in info["headers"] and "HE+" in info["headers"]
    assert info["row"].startswith("| Coder-3B")
    assert bench_research.table_info("plain prose 42.7 here", 12) is None


def test_checks_reach_the_run_record(client, monkeypatch):
    bad = json.loads(json.dumps(SYNTH))
    bad["benchmarks"][0]["reference_scores"].append(
        {"model": "Phi-3-mini", "params_b": 3.8, "score": 45.2, "source_url": REAL_URL, "metric": "accuracy",
         "label": "MMLU-Pro law", "quote": "| Phi-3-mini | 3.8B | 45.2 |"})

    def llm(messages, schema, max_tokens=2500):
        if "queries" in json.dumps(schema) or "scores" in json.dumps(schema):
            return fake_llm(messages, schema, max_tokens)
        return json.dumps(bad)

    monkeypatch.setattr(bench_research, "ask_llm", llm)
    spec = draft(client)
    research = wait_research(client, spec["id"])
    law = [p for p in research["proposals"] if p["runnable_id"] == "mmlupro:law"][0]
    assert "Phi-3-mini" not in [entry["model"] for entry in law["reference_scores"]]
    assert research["score_checks"]["dropped"]["too_large"] == 1
    assert law["confidence"] in ("low", "medium", "high")
    assert law["confidence_note"]
    assert law["metric"] == "accuracy"
    texts = [step["text"] for step in research["steps"]]
    assert any("dropped 1" in text for text in texts)
