"""Tests for the benchmark research agent. No network: the search provider and the LLM are faked."""

import json
import threading
import time

import pytest
from fastapi.testclient import TestClient

from studio.server import bench_research
from studio.server.app import app

REAL_URL = "https://example.org/legal-leaderboard"
PLAN = {"plan": "I will look for legal question answering benchmarks and what small models score.",
        "queries": ["legal question answering benchmark", "LegalBench small model scores"]}
SYNTH = {
    "reasoning": "Legal questions are knowledge heavy, so MMLU style exams fit, and instruction following guards.",
    "benchmarks": [
        {"name": "MMLU professional law", "measures": "law exam questions", "why": "closest runnable exam",
         "runnable_id": "mmlupro:law", "kind": "target",
         "reference_scores": [
             {"model": "Qwen2.5-1.5B", "params_b": 1.5, "score": 31.0, "source_url": REAL_URL},
             {"model": "Invented-2B", "params_b": 2, "score": 0.55, "source_url": "https://invented.example/x"}],
         "suggested_target": 0.35},
        {"name": "LegalBench", "measures": "legal reasoning tasks", "why": "the standard legal suite",
         "runnable_id": "legalbench", "kind": "target", "reference_scores": [], "suggested_target": None},
        {"name": "IFEval", "measures": "instruction following", "why": "keeps answers well formed",
         "runnable_id": "if_guard", "kind": "guard", "reference_scores": [], "suggested_target": None},
    ],
}


def fake_search(provider="exa"):
    def search(query, run):
        return {"summary": "", "results": [
            {"title": "Legal leaderboard", "url": REAL_URL, "text": "Qwen2.5-1.5B scores 31 on law."},
            {"title": "Paper " + query[:8], "url": "https://example.org/p/" + str(abs(hash(query)) % 1000),
             "text": "more"}]}
    return search


SCORE_URL = "https://example.org/law-results"
SCORES = {"scores": [
    {"benchmark": "MMLU professional law", "model": "SmolLM2-1.7B", "params_b": 1.7, "score": 28.4, "source_url": SCORE_URL,
     "quote": "SmolLM2-1.7B reaches 28.4 on the law category"},
    {"benchmark": "MMLU professional law", "model": "Made-Up-1B", "params_b": 1.0, "score": 0.61, "source_url": SCORE_URL,
     "quote": "Made-Up-1B reaches 61.0 on the law category"},
]}


def fake_score_search(name, size_b):
    return [{"title": "Law results", "url": SCORE_URL,
             "text": "Table 2. SmolLM2-1.7B  reaches 28.4 on the law category. Other rows follow."}]


def fake_llm(messages, schema, max_tokens=2500):
    text = json.dumps(schema)
    if "queries" in text:
        return json.dumps(PLAN)
    if "scores" in text:
        return json.dumps(SCORES)
    return json.dumps(SYNTH)


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("WSP_STUDIO_OUT", str(tmp_path))
    monkeypatch.setenv("WSP_STUDIO_BACKEND", "stub")
    monkeypatch.setenv("WSP_STUDIO_LLM", "openai_compat")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key-not-used")
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
        return fake_llm(messages, schema, max_tokens)

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
