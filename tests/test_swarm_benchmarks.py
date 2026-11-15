"""Tests for wsp/swarm/benchmarks.py: Trisanth's studio research picks a run's benchmarks.

No network: the search provider and the LLM are faked the way tests/test_studio_research.py fakes them.
"""

import copy
import json
import os
import sys

import pytest

from studio.server import bench_research
from studio.server import specs as specs_mod
from wsp.swarm import benchmarks
from wsp.swarm.config import REPO_ROOT, make_config

SCRIPTS = os.path.join(REPO_ROOT, "scripts")
if SCRIPTS not in sys.path:
    sys.path.insert(0, SCRIPTS)

import run_baseline  # noqa: E402
import run_swarm  # noqa: E402

REAL_URL = "https://example.org/med-leaderboard"
PLAN = {"plan": "I will look for medical exam benchmarks and small model scores.",
        "queries": ["medical exam question answering benchmark", "MedQA small model scores"]}
SYNTH = {
    "reasoning": "Medical exam questions are knowledge heavy; MMLU-Pro health is the closest runnable exam.",
    "benchmarks": [
        {"name": "MMLU-Pro health", "measures": "health exam questions", "why": "closest runnable exam",
         "runnable_id": "mmlupro:health", "kind": "target",
         "reference_scores": [{"model": "Qwen2.5-1.5B", "params_b": 1.5, "score": 30.0, "source_url": REAL_URL}],
         "suggested_target": 0.35},
        {"name": "MedQA", "measures": "USMLE style questions", "why": "the standard medical suite",
         "runnable_id": "medqa", "kind": "target", "reference_scores": [], "suggested_target": None},
        {"name": "IFEval", "measures": "instruction following", "why": "keeps answers well formed",
         "runnable_id": "if_guard", "kind": "guard", "reference_scores": [], "suggested_target": None},
    ],
}


def fake_search(query, run):
    return {"summary": "", "results": [
        {"title": "Medical leaderboard", "url": REAL_URL, "text": "Qwen2.5-1.5B scores 30 on health."}]}


def fake_llm(messages, schema, max_tokens=2500):
    text = json.dumps(schema)
    if "queries" in text:
        return json.dumps(PLAN)
    if "scores" in text:
        return json.dumps({"scores": []})
    return json.dumps(SYNTH)


@pytest.fixture(autouse=True)
def restore_swarm_defaults():
    """The studio writes into the swarm's DEFAULTS (settings.py: openai_model; specs.py: measured bases).

    That is by design in the studio's own process, and scripts/research_benchmarks.py runs in its own
    process too. In this test process the change would leak into later tests, so put DEFAULTS back.
    """
    from wsp.swarm import config as config_mod

    saved = copy.deepcopy(config_mod.DEFAULTS)
    yield
    config_mod.DEFAULTS.clear()
    config_mod.DEFAULTS.update(saved)


@pytest.fixture
def research(tmp_path, monkeypatch):
    monkeypatch.setenv("WSP_STUDIO_OUT", str(tmp_path / "studio"))
    monkeypatch.setenv("WSP_STUDIO_BACKEND", "stub")
    monkeypatch.setenv("WSP_STUDIO_LLM", "openai_compat")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key-not-used")
    monkeypatch.setattr(bench_research, "choose_provider", lambda: "exa")
    monkeypatch.setattr(bench_research, "search_exa", fake_search)
    monkeypatch.setattr(bench_research, "search_exa_scores", lambda name, size_b: [])
    monkeypatch.setattr(bench_research, "ask_llm", fake_llm)
    return tmp_path


def test_research_keeps_runnable_picks_as_targets_and_guards(research):
    record = benchmarks.research_setup("a small model that answers medical exam questions", "medicine")
    setup = record["task_setup"]
    # The runnable pick joins the studio's rule default; MedQA is not runnable here, so it stays reference only.
    assert "mmlupro:health" in setup["tasks"] and "medqa" not in setup["tasks"]
    assert setup["tasks"]["mmlupro:health"] == 1.0
    assert "if_guard" in setup["guards"] and setup["heal_data"] == ["medicine"]
    names = {}
    for proposal in record["proposals"]:
        names[proposal["name"]] = proposal["runnable_id"]
    assert names["MedQA"] is None and names["MMLU-Pro health"] == "mmlupro:health"
    assert record["provider"] == "exa" and record["task_type"] == "researched"
    kinds = []
    for step in record["steps"]:
        kinds.append(step["kind"])
    assert "search" in kinds and "pick" in kinds


def test_research_needs_an_llm(research, monkeypatch):
    monkeypatch.setenv("WSP_STUDIO_LLM", "none")
    with pytest.raises(RuntimeError) as error:
        benchmarks.research_setup("a model for law questions")
    assert "needs an LLM" in str(error.value)


def test_task_setup_follows_run_overrides():
    spec = {"benchmarks": [{"id": "mmlu", "kind": "target"}, {"id": "mmlupro:law", "kind": "target"},
                           {"id": "if_guard", "kind": "guard"}]}
    setup = benchmarks.task_setup(spec, "general")
    overrides = specs_mod.run_overrides(dict(spec, budget={"evals": 10, "compute_min": 30}, task_type="t",
                                             host="h"), "stub", "none")
    assert setup["tasks"] == overrides["task_types"]["t"]["tasks"]
    assert setup["guards"] == overrides["task_types"]["t"]["guards"]
    assert setup["heal_data"] == ["general"]


def write_record(tmp_path):
    record = {"request": "law", "task_type": "researched",
              "task_setup": {"tasks": {"mmlupro:law": 1.0}, "guards": ["if_guard"], "heal_data": ["general"]}}
    path = tmp_path / "law.json"
    path.write_text(json.dumps(record), encoding="utf-8")
    return str(path)


def test_both_runners_read_a_researched_file(tmp_path):
    path = write_record(tmp_path)
    swarm_config = run_swarm.build_config(run_swarm.parse_args(["--benchmarks", path]))
    base_config = run_baseline.build_config(run_baseline.parse_args(["--method", "evolve", "--benchmarks", path]))
    for config in [swarm_config, base_config]:
        assert config["task_type"] == "researched"
        setup = config["task_types"]["researched"]
        assert setup == {"tasks": {"mmlupro:law": 1.0}, "guards": ["if_guard"], "heal_data": ["general"]}


def test_a_stub_swarm_runs_on_a_researched_setup(tmp_path):
    from wsp.swarm.swarm import run_swarm as run

    path = write_record(tmp_path)
    overrides = run_swarm.researched_overrides(path)
    overrides.update({"outputs_dir": str(tmp_path), "run_id": "test-researched", "eval_budget": 8,
                      "heldout_top": 0})
    summary = run(make_config(overrides))
    assert summary["eval_used"] == 8.0
    assert summary["best_search"] is not None
    assert "mmlupro:law" in summary["best_search"]["scores"]
