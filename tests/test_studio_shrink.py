"""Tests for the shrink request type of the studio. CPU only, stub backend, no network, no model downloads."""

import time

import pytest
from fastapi.testclient import TestClient

from studio.server import compare
from studio.server import modal_backend
from studio.server import shrink_models
from studio.server import shrink_spec
from studio.server import sizing
from studio.server import specs as specs_mod
from studio.server.app import app

CODER = "Qwen/Qwen2.5-Coder-7B-Instruct"
MOE = "Qwen/Qwen3-30B-A3B-Instruct-2507"
SHRINK_PROMPTS = [
    "a coding-only model: drop everything else from Qwen2.5-Coder-7B and get it under 3B",
    "keep only the math experts of a 30B MoE",
    "make a smaller version of Llama 3.1 8B that only does math",
]
BUILD_PROMPTS = [
    "I want a math tutor for grade-school word problems on my laptop",
    "a model that answers medical questions, CPU only",
]


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("WSP_STUDIO_OUT", str(tmp_path))
    monkeypatch.setenv("WSP_STUDIO_BACKEND", "stub")
    monkeypatch.setenv("WSP_STUDIO_LLM", "none")
    monkeypatch.setenv("WSP_STUDIO_PACE_S", "0")
    return TestClient(app)


def make_spec(client, prompt):
    response = client.post("/api/spec", data={"prompt": prompt})
    assert response.status_code == 200, response.text
    return response.json()


def wait_job(client, job_id, timeout=30.0):
    end = time.time() + timeout
    while time.time() < end:
        job = client.get("/api/jobs/" + job_id).json()
        if job["status"] in ("done", "error"):
            return job
        time.sleep(0.02)
    raise AssertionError("job timed out")


# Draft detection.

@pytest.mark.parametrize("prompt", SHRINK_PROMPTS)
def test_shrink_prompts_are_detected(prompt):
    assert shrink_spec.looks_like_shrink(prompt)


@pytest.mark.parametrize("prompt", BUILD_PROMPTS)
def test_build_prompts_are_not_shrinks(prompt):
    assert not shrink_spec.looks_like_shrink(prompt)


def test_draft_spec_request_types(client):
    for prompt in SHRINK_PROMPTS:
        spec = make_spec(client, prompt)
        assert spec["request_type"] == "shrink"
        assert spec["shrink"]["source_model"] == spec["host"]
    for prompt in BUILD_PROMPTS:
        spec = make_spec(client, prompt)
        assert spec["request_type"] == "build"
        assert "shrink" not in spec


def test_llm_answer_decides_the_type(monkeypatch):
    draft = {"request_type": "shrink"}
    assert specs_mod.request_type_for("hello", draft) == "shrink"
    assert specs_mod.request_type_for("shrink it", {"request_type": "build"}) == "build"
    assert specs_mod.request_type_for("shrink it", {}) == "shrink"


# Source model choice.

def test_named_model_is_the_source(client):
    spec = make_spec(client, SHRINK_PROMPTS[0])
    block = spec["shrink"]
    assert block["source_model"] == CODER
    assert block["keep_skill"] == "code"
    assert block["method"] == "dense" and block["family"] == "dense"
    assert block["target_params_b"] < 3.0
    assert block["rule"] == "skill" and block["heal_steps"] == 300
    assert spec["limits"]["max_total_params_b"] == 3.0
    assert [item["id"] for item in spec["benchmarks"]] == ["humaneval"]
    assert 1 <= len(block["candidates"]) <= 5
    assert block["source_model"] not in [item["model"] for item in block["candidates"]]


def test_moe_request_picks_a_moe_with_the_skill(client):
    spec = make_spec(client, SHRINK_PROMPTS[1])
    block = spec["shrink"]
    assert block["family"] == "moe" and block["method"] == "experts"
    assert block["source_model"].startswith("Qwen/Qwen3-30B-A3B")
    assert block["keep_skill"] == "math"
    assert block["keep_experts"] == 32 and block["rule"] == "saliency"


def test_source_without_a_name_is_the_smallest_with_the_skill(client):
    spec = make_spec(client, "a coding-only model under 3B")
    assert spec["shrink"]["source_model"] == CODER


def test_every_skill_is_a_healdata_pool():
    from wsp import healdata

    # Every pool is a valid skill. A pool healdata adds later has no studio label yet and
    # falls back to "<pool> text", so the studio keeps working when the pools grow.
    assert shrink_spec.SKILL_POOLS == sorted(healdata.SOURCES)
    for name in ("code", "math", "medicine", "general", "prose"):
        assert name in healdata.SOURCES
    assert shrink_spec.label_of("some_new_pool") == "some_new_pool text"


def test_lists_match_wsp_slim():
    from wsp import slim

    assert list(slim.RULES) == shrink_models.EXPERT_RULES
    assert set(shrink_models.MOE_FAMILIES) == set(slim.FAMILIES)


# Estimates.

def test_moe_plan_estimate():
    plan = {"source_model": MOE, "method": "experts", "keep_experts": 32, "heal_steps": 300}
    est = sizing.estimate_shrink(plan)
    assert est["estimated"] is True
    assert est["total_params_b"] == pytest.approx(8.78, abs=0.02)
    assert est["run_params_b"] == pytest.approx(3.03, abs=0.02)
    assert est["experts_kept"] == 32 and est["experts_before"] == 128
    assert est["source_params_b"] == 30.53
    assert est["heal_min"] == pytest.approx(300 * 0.3 * 3.03 / 60, abs=0.05)


def test_fewer_experts_means_fewer_parameters():
    totals = []
    for keep in [64, 32, 16, 4]:
        plan = {"source_model": MOE, "method": "experts", "keep_experts": keep, "heal_steps": 100}
        totals.append(sizing.estimate_shrink(plan)["total_params_b"])
    assert totals == sorted(totals, reverse=True)
    below_top_k = sizing.estimate_shrink({"source_model": MOE, "method": "experts", "keep_experts": 4,
                                          "heal_steps": 100})
    assert below_top_k["top_k_after"] == 4


def test_dense_plan_estimate():
    plan = {"source_model": CODER, "method": "dense", "target_params_b": 3.1, "heal_steps": 300}
    est = sizing.estimate_shrink(plan)
    assert est["total_params_b"] == 3.1
    assert est["layers_dropped"] == 7 and est["layers_before"] == 28
    assert est["keep_width"] == pytest.approx(0.41, abs=0.02)
    assert est["run_params_b"] == pytest.approx(3.1 - 0.545, abs=0.01)


def test_limit_problems_carry_numbers():
    plan = {"source_model": MOE, "method": "experts", "keep_experts": 32, "heal_steps": 300}
    est = sizing.estimate_shrink(plan)
    limits = {"max_total_params_b": 5.0, "max_run_params_b": 2.0, "max_heal_min": 1}
    problems = sizing.shrink_limit_problems(limits, est)
    assert len(problems) == 3
    assert "8.778B" in problems[0] and "5B" in problems[0]
    assert sizing.shrink_limit_problems({"max_total_params_b": 9.0, "max_run_params_b": 3.1,
                                         "max_heal_min": 20}, est) == []


# The routes.

def test_shrink_session_plan_and_edit(client):
    spec = make_spec(client, SHRINK_PROMPTS[1])
    response = client.post("/api/shrink", json={"spec_id": spec["id"]})
    assert response.status_code == 200, response.text
    body = response.json()
    plan = body["plan"]
    assert body["shrink_id"].startswith("sh_")
    assert plan["estimate"]["estimated"] is True
    assert plan["summary"].startswith("Keep 32 of 128 experts per layer by saliency on math text.")
    assert "(estimated)" in plan["summary"]
    assert plan["ok"] is True
    edited = client.post("/api/shrink/" + body["shrink_id"] + "/plan", json={"keep_experts": 16}).json()
    assert edited["keep_experts"] == 16
    assert edited["estimate"]["total_params_b"] < plan["estimate"]["total_params_b"]


def test_over_limit_run_is_refused_with_numbers(client):
    spec = make_spec(client, SHRINK_PROMPTS[1])
    sid = client.post("/api/shrink", json={"spec_id": spec["id"]}).json()["shrink_id"]
    plan = client.post("/api/shrink/" + sid + "/plan", json={"keep_experts": 64}).json()
    assert plan["ok"] is False and len(plan["problems"]) >= 1
    response = client.post("/api/shrink/" + sid + "/run", json={})
    assert response.status_code == 400
    assert "Over the total size limit" in response.json()["error"]
    assert "B" in response.json()["error"]


def test_run_on_the_stub_returns_a_comparison(client):
    spec = make_spec(client, SHRINK_PROMPTS[0])
    sid = client.post("/api/shrink", json={"spec_id": spec["id"]}).json()["shrink_id"]
    response = client.post("/api/shrink/" + sid + "/run", json={})
    assert response.status_code == 200, response.text
    job = wait_job(client, response.json()["job_id"])
    assert job["status"] == "done", job
    assert job["action"] == "shrink" and job["stub"] is True
    result = job["result"]
    assert result["stub"] is True
    assert result["params_b"] < 3.0 and result["active_params_b"] < result["params_b"]
    assert result["heal"] == {"steps": 300, "kind": "distillation", "teacher": True}
    assert result["removed"]["kind"] == "dense" and result["removed"]["layers_after"] == 21
    assert len(result["log"]) >= 2
    scores = result["scores"]["humaneval"]
    assert result["control"]["scores"]["humaneval"] < scores < result["source_scores"]["humaneval"]
    comparison = result["comparison"]
    assert comparison["verdict"] in ("success", "failed", "unproven")
    ids = [row["id"] for row in comparison["rows"]]
    assert ids[:2] == ["source", "random"] and ids[-1] == "best_model"
    by_id = {row["id"]: row for row in comparison["rows"]}
    assert by_id["random"]["beaten"] is True and by_id["source"]["beaten"] is None
    assert by_id["source"]["delta"] < 0
    assert comparison["headline"]


def test_stub_scores_fall_as_more_is_removed(client):
    spec = make_spec(client, SHRINK_PROMPTS[1])
    limits = {"max_total_params_b": 31, "max_run_params_b": 31, "max_heal_min": 60}
    assert client.put("/api/spec/" + spec["id"], json=dict(spec, limits=limits)).status_code == 200
    sid = client.post("/api/shrink", json={"spec_id": spec["id"]}).json()["shrink_id"]
    skill = []
    random = []
    for keep in [96, 64, 32, 16]:
        client.post("/api/shrink/" + sid + "/plan", json={"keep_experts": keep})
        job = wait_job(client, client.post("/api/shrink/" + sid + "/run", json={}).json()["job_id"])
        assert job["status"] == "done", job
        skill.append(job["result"]["scores"]["gsm8k"])
        random.append(job["result"]["control"]["scores"]["gsm8k"])
    assert skill == sorted(skill, reverse=True)
    assert random == sorted(random, reverse=True)
    assert all(low < high for low, high in zip(random, skill))


def test_shrink_spec_is_not_a_build(client):
    spec = make_spec(client, SHRINK_PROMPTS[0])
    assert client.post("/api/runs", json={"spec_id": spec["id"]}).status_code == 400
    assert client.post("/api/builder", json={"spec_id": spec["id"]}).status_code == 400
    build = make_spec(client, BUILD_PROMPTS[0])
    assert client.post("/api/shrink", json={"spec_id": build["id"]}).status_code == 400


# Spec edits.

def test_put_validates_the_shrink_block(client):
    spec = make_spec(client, SHRINK_PROMPTS[1])
    bad = [{"source_model": "nobody/NoSuchModel"}, {"keep_experts": 0}, {"keep_experts": 129},
           {"keep_skill": "cooking"}, {"rule": "freq2"}, {"heal_steps": -1}, {"method": "dense"}]
    for fields in bad:
        edited = dict(spec)
        edited["shrink"] = dict(spec["shrink"], **fields)
        response = client.put("/api/spec/" + spec["id"], json=edited)
        assert response.status_code == 400, fields
        assert response.json()["error"]
    dense = make_spec(client, SHRINK_PROMPTS[0])
    for fields in [{"target_params_b": 9.0}, {"target_params_b": 0.1}, {"rule": "saliency"}, {"method": "experts"}]:
        edited = dict(dense)
        edited["shrink"] = dict(dense["shrink"], **fields)
        assert client.put("/api/spec/" + dense["id"], json=edited).status_code == 400, fields


def test_put_accepts_good_edits(client):
    spec = make_spec(client, SHRINK_PROMPTS[1])
    edited = dict(spec)
    edited["shrink"] = dict(spec["shrink"], keep_experts=24, rule="freq", keep_skill="code", heal_steps=100)
    response = client.put("/api/spec/" + spec["id"], json=edited)
    assert response.status_code == 200, response.text
    block = response.json()["shrink"]
    assert block["keep_experts"] == 24 and block["rule"] == "freq" and block["heal_steps"] == 100
    again = client.get("/api/spec/" + spec["id"]).json()
    assert again["shrink"]["keep_skill"] == "code"
    assert again["request_type"] == "shrink"


def test_changing_the_source_resets_the_plan(client):
    spec = make_spec(client, SHRINK_PROMPTS[0])
    edited = dict(spec)
    edited["shrink"] = {"source_model": MOE}
    response = client.put("/api/spec/" + spec["id"], json=edited)
    assert response.status_code == 200, response.text
    block = response.json()["shrink"]
    assert block["method"] == "experts" and block["keep_experts"] == 32 and block["rule"] == "saliency"
    assert response.json()["host"] == MOE


def test_build_specs_are_unchanged(client):
    spec = make_spec(client, BUILD_PROMPTS[0])
    assert spec["request_type"] == "build"
    assert spec["limits"] == {"max_total_params_b": 1.5, "max_run_params_b": 1.5, "max_heal_min": 20}
    assert spec["host"] == "unsloth/Llama-3.2-1B-Instruct"
    assert [item["id"] for item in spec["benchmarks"]] == ["gsm8k", "if_guard"]
    response = client.put("/api/spec/" + spec["id"], json=dict(spec, shrink={"keep_experts": 3}))
    assert response.status_code == 200
    assert "shrink" not in response.json()


# The comparison.

def shrink_spec_for(benchmarks):
    return {"benchmarks": benchmarks, "shrink": {"source_model": "Qwen/Qwen2.5-14B-Instruct"}}


def test_comparison_verdicts():
    spec = shrink_spec_for([{"id": "gsm8k", "kind": "target", "base": 0.3}])
    models = {"m/near-1.5B": {"params_b": 1.5, "scores": {"gsm8k_search_raw": {"value": 0.50, "split": "search"}}}}
    good = compare.build_shrink(spec, {"gsm8k": 0.60}, 1.5, {"gsm8k": 0.80}, {"gsm8k": 0.20}, None, True, models)
    assert good["verdict"] == "success" and good["rows"][0]["id"] == "source"
    low = compare.build_shrink(spec, {"gsm8k": 0.40}, 1.5, {"gsm8k": 0.80}, {"gsm8k": 0.20}, None, True, models)
    assert low["verdict"] == "failed" and low["not_beaten"] == ["best_model"]
    random = compare.build_shrink(spec, {"gsm8k": 0.21}, 1.5, {"gsm8k": 0.80}, {"gsm8k": 0.20}, None, True, {})
    assert random["verdict"] == "failed" and random["not_beaten"] == ["random"]
    no_control = compare.build_shrink(spec, {"gsm8k": 0.60}, 1.5, None, None, None, True, models)
    assert no_control["verdict"] == "unproven" and "random" in no_control["unmeasured"]
    nothing_near = compare.build_shrink(spec, {"gsm8k": 0.60}, 1.5, None, {"gsm8k": 0.2}, None, True, {})
    assert nothing_near["verdict"] == "success"
    assert nothing_near["rows"][-1]["required"] is False
    unscored = {"m/near-1.5B": {"params_b": 1.5, "scores": {}}}
    assert compare.build_shrink(spec, {"gsm8k": 0.6}, 1.5, None, {"gsm8k": 0.2}, None, True,
                                unscored)["verdict"] == "unproven"


def test_the_source_does_not_decide_the_verdict():
    spec = shrink_spec_for([{"id": "gsm8k", "kind": "target", "base": 0.3}])
    result = compare.build_shrink(spec, {"gsm8k": 0.30}, 1.5, {"gsm8k": 0.95}, {"gsm8k": 0.10}, None, False, {})
    assert result["verdict"] == "success"
    assert result["rows"][0]["beaten"] is None and result["rows"][0]["delta"] < 0


# The Modal path with a fake remote function.

def fake_remote_result():
    return {"ok": True, "gpu": "fake", "seconds": {"load": 1.0, "heal": 2.0},
            "params": {"total_before": 1_544_000_000, "run_before": 1_300_000_000, "total_after": 1_190_000_000,
                       "run_after": 1_000_000_000},
            "removed": {"kind": "dense", "layers_before": 28, "layers_after": 26, "layers_dropped": [3, 9],
                        "keep_mlp": 0.9, "keep_heads": 0.9},
            "scores": {"gsm8k": 0.5}, "control_scores": {"gsm8k": 0.2}, "source_scores": {"gsm8k": 0.7},
            "general_scores": None, "heal": {"steps": 100, "kind": "distillation", "teacher": True},
            "log": ["fake"], "items": {"gsm8k": 10}}


def test_modal_path_with_a_fake_function(client, monkeypatch):
    monkeypatch.setenv("WSP_STUDIO_BACKEND", "modal")
    calls = []

    class Fake:
        def __init__(self, name):
            self.name = name

        def remote(self, plan, split, tasks, fidelity, options):
            calls.append((self.name, plan, split, tasks, fidelity, options))
            return fake_remote_result()

    monkeypatch.setattr(modal_backend, "lookup", lambda name: Fake(name))
    spec = make_spec(client, "shrink Qwen/Qwen2.5-1.5B-Instruct to about 1.2B and keep math")
    assert spec["shrink"]["source_model"] == "Qwen/Qwen2.5-1.5B-Instruct"
    sid = client.post("/api/shrink", json={"spec_id": spec["id"]}).json()["shrink_id"]
    client.post("/api/shrink/" + sid + "/plan", json={"target_params_b": 1.2, "heal_steps": 100})
    job = wait_job(client, client.post("/api/shrink/" + sid + "/run", json={"fidelity": "check"}).json()["job_id"])
    assert job["status"] == "done", job
    assert job["stub"] is False
    result = job["result"]
    assert result["stub"] is False and result["params_b"] == 1.19
    assert result["comparison"]["verdict"] in ("success", "failed", "unproven")
    name, plan, split, tasks, fidelity, options = calls[0]
    assert name == "shrink_remote" and tasks == ["gsm8k"] and fidelity == "check"
    assert plan["target_params_b"] == 1.2 and plan["heal_steps"] == 100 and plan["keep_skill"] == "math"
    assert options["control"] is True and options["score_source"] is True


def test_big_sources_use_the_h100_twin(monkeypatch):
    names = []

    class Fake:
        def remote(self, *args):
            return fake_remote_result()

    monkeypatch.setattr(modal_backend, "lookup", lambda name: names.append(name) or Fake())
    modal_backend.run_shrink({"source_params_b": 30.5}, "search", ["gsm8k"], "low", {})
    modal_backend.run_shrink({"source_params_b": 1.5}, "search", ["gsm8k"], "low", {})
    assert names == ["shrink_remote_h100", "shrink_remote"]


def test_remote_failure_becomes_a_job_error(client, monkeypatch):
    monkeypatch.setenv("WSP_STUDIO_BACKEND", "modal")

    class Fake:
        def remote(self, *args):
            return {"ok": False, "error": "RuntimeError: boom", "trace": "trace"}

    monkeypatch.setattr(modal_backend, "lookup", lambda name: Fake())
    spec = make_spec(client, SHRINK_PROMPTS[0])
    sid = client.post("/api/shrink", json={"spec_id": spec["id"]}).json()["shrink_id"]
    job = wait_job(client, client.post("/api/shrink/" + sid + "/run", json={}).json()["job_id"])
    assert job["status"] == "error" and "boom" in job["error"]
