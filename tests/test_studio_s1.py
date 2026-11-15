"""Tests for the S1 server fixes: default limits, empty runs, finished agents, verdict wording, builder budget,
leaderboard order, shrink wording, stub consistency, benchmark labels and heal data pools. CPU only, stub backend."""

import pytest
from fastapi.testclient import TestClient

from studio.server import bench_research
from studio.server import builder
from studio.server import compare
from studio.server import programs
from studio.server import shrink_models
from studio.server import specs as specs_mod
from studio.server.app import app
from studio.server.runstate import RunState
from tests.test_studio_server import good_program, make_spec, snapshot, start_run, wait_done
from tests.test_studio_shrink import SHRINK_PROMPTS, wait_job
from tests.test_studio_shrink import make_spec as make_prompt_spec
from wsp.swarm import program_schema as ps


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("WSP_STUDIO_OUT", str(tmp_path))
    monkeypatch.setenv("WSP_STUDIO_BACKEND", "stub")
    monkeypatch.setenv("WSP_STUDIO_LLM", "none")
    monkeypatch.setenv("WSP_STUDIO_PACE_S", "0")
    return TestClient(app)


def draft(client, prompt):
    response = client.post("/api/spec", data={"prompt": prompt})
    assert response.status_code == 200, response.text
    return response.json()


# 1. Default limits.

def test_default_limits_are_host_size_times_one_and_a_half(client):
    spec = draft(client, "a math tutor for word problems")
    host = spec["size"]["host_params_b"]
    assert spec["limits"]["max_total_params_b"] == round(host * 1.5, 1)
    assert spec["limits"]["max_run_params_b"] == spec["limits"]["max_total_params_b"]


def test_a_named_size_is_the_total_and_the_run_limit(client):
    spec = draft(client, "a math tutor under 2B")
    assert spec["limits"]["max_total_params_b"] == 2.0
    assert spec["limits"]["max_run_params_b"] == 2.0


def test_a_build_limit_is_never_below_host_plus_point_two(client):
    spec = draft(client, "a math tutor under 1.3B")
    host = spec["size"]["host_params_b"]
    assert spec["limits"]["max_total_params_b"] >= round(host + 0.2, 2)
    assert spec["limits"]["max_run_params_b"] >= round(host + 0.2, 2)


def test_a_default_drafted_spec_lets_the_stub_swarm_score(client):
    spec = draft(client, "a math tutor for word problems")
    spec["budget"]["evals"] = 6
    spec = client.put("/api/spec/" + spec["id"], json=spec).json()
    final = wait_done(client, start_run(client, spec))
    assert final["status"] == "done"
    assert len(final["leaderboard"]) >= 1
    assert final["best"] is not None


# 2. Empty reason.

def test_a_run_with_nothing_scored_says_why(client):
    spec = make_spec(client, evals=6)
    spec["limits"] = {"max_total_params_b": 1.24, "max_run_params_b": 1.24, "max_heal_min": 20}
    spec = client.put("/api/spec/" + spec["id"], json=spec).json()
    final = wait_done(client, start_run(client, spec))
    assert final["best"] is None
    assert final["refused"] > 0
    assert "refused for going over your limits" in final["empty_reason"]
    assert "Raise the size limits" in final["empty_reason"]


def test_empty_reason_text_for_refused_proposals():
    state = fake_state()
    state.refused = 195
    state.set_run_status("done")
    reason = state.snapshot()["empty_reason"]
    assert reason == ("Every proposal was refused for going over your limits (195 refused). "
                      "Raise the size limits on the spec.")


def test_empty_reason_is_none_while_running_and_when_there_is_a_best(client):
    state = fake_state()
    assert state.snapshot()["empty_reason"] is None
    final = wait_done(client, start_run(client, make_spec(client, evals=4)))
    assert final["best"] is not None and final["empty_reason"] is None


def fake_state():
    spec = {"benchmarks": [{"id": "gsm8k", "kind": "target"}], "limits": None}
    config = {"eval_budget": 5, "compute_budget_s": 100, "host": "h"}
    return RunState("r1", spec, config, "stub", 1.24, 2, ["gsm8k"])


# 3. Agents and outcome after a run.

def test_agents_are_done_and_outcome_is_finished(client):
    final = wait_done(client, start_run(client, make_spec(client, evals=4)))
    assert final["outcome"] == "finished"
    for agent in final["agents"]:
        assert agent["status"] == "done"
        assert agent["now"] == "Finished"


def test_outcome_is_null_while_running():
    assert fake_state().snapshot()["outcome"] is None


def test_a_stopped_run_marks_agents_stopped():
    state = fake_state()
    state.stopped_early = True
    state.set_run_status("done")
    snap = state.snapshot()
    assert snap["outcome"] == "stopped"
    assert {agent["now"] for agent in snap["agents"]} == {"Stopped"}


def test_an_error_run_has_outcome_error():
    state = fake_state()
    state.set_run_status("error", "boom")
    snap = state.snapshot()
    assert snap["outcome"] == "error"
    assert snap["empty_reason"] is not None


# 4. Verdict wording.

def compare_spec():
    return {"benchmarks": [{"id": "gsm8k", "kind": "target", "base": 0.3, "lora": 0.3}],
            "limits": {"max_total_params_b": 1.9}, "size": {"host_params_b": 1.24}}


def test_a_result_within_the_margin_is_worded_that_way():
    out = compare.build(compare_spec(), {"gsm8k": 0.406}, 1.5, {"gsm8k": 0.406}, {"gsm8k": 0.395}, "heldout", False,
                        {})
    rows = {row["id"]: row for row in out["rows"]}
    assert rows["random"]["within_margin"] is True
    assert rows["random"]["beaten"] is False
    assert out["verdict"] == "failed"
    assert out["headline"] == ("Failed attempt. It is within 2 points of Random components (same size) "
                               "(40.6 against 39.5), so the parts are not shown to matter.")


def test_a_result_below_a_reference_is_not_within_margin():
    out = compare.build(compare_spec(), {"gsm8k": 0.3}, 1.5, {"gsm8k": 0.3}, {"gsm8k": 0.4}, "heldout", False, {})
    rows = {row["id"]: row for row in out["rows"]}
    assert rows["random"]["within_margin"] is False
    assert "is below Random components" in out["headline"]
    assert "within" not in out["headline"]


def test_a_clear_win_is_not_within_margin():
    out = compare.build(compare_spec(), {"gsm8k": 0.6}, 1.5, {"gsm8k": 0.6}, {"gsm8k": 0.4}, "heldout", False, {})
    rows = {row["id"]: row for row in out["rows"]}
    assert rows["random"]["within_margin"] is False and rows["random"]["beaten"] is True


def test_shrink_comparison_words_the_margin():
    spec = {"benchmarks": [{"id": "gsm8k", "kind": "target"}], "shrink": {"source_model": "x"}}
    out = compare.build_shrink(spec, {"gsm8k": 0.405}, 2.0, {"gsm8k": 0.6}, {"gsm8k": 0.395}, None, True, {})
    rows = {row["id"]: row for row in out["rows"]}
    assert rows["random"]["within_margin"] is True
    assert out["verdict"] == "failed"
    assert "within 2 points of" in out["headline"]
    assert "does not beat" not in out["headline"]


# 5. Builder budget.

def test_builder_budget_comes_from_the_spec(client):
    spec = make_spec(client, evals=9)
    spec["budget"]["compute_min"] = 70
    spec = client.put("/api/spec/" + spec["id"], json=spec).json()
    created = client.post("/api/builder", json={"spec_id": spec["id"]}).json()
    assert created["budget"]["evals_total"] == 9
    assert created["budget"]["compute_total_s"] == specs_mod.compute_budget_s(9, 70)


# 6. Leaderboard order.

def test_leaderboard_sorts_by_target_score_with_collapsed_last():
    state = fake_state()
    rows = {"a": ("ok", 0.4, 0.9), "b": ("ok", 0.6, 0.1), "c": ("collapsed", 0.9, 0.99), "d": ("ok", 0.5, 0.5)}
    for key in rows:
        status, target, combined = rows[key]
        state.programs[key] = {"row": {"status": status, "scores": {"gsm8k": target}, "combined": combined,
                                        "describe": key, "program": None}, "agent": "planner-1", "donors": []}
    state.set_run_status("done")
    snap = state.snapshot()
    assert snap["sort"] == "target score"
    assert [entry["program_hash"] for entry in snap["leaderboard"]] == ["b", "d", "a", "c"]


# 7. Shrink plan sentence.

def test_shrink_plan_sentence_uses_plain_phrases():
    for rule in shrink_models.EXPERT_RULES + shrink_models.DENSE_RULES:
        assert shrink_models.rule_phrase(rule) in shrink_models.RULE_PHRASES.values()
    assert shrink_models.rule_phrase("specific") == "by how specific they are to the skill"
    assert shrink_models.rule_phrase("random") == "at random (control)"
    assert shrink_models.rule_phrase("general") == "by use on general text (control)"
    assert shrink_models.rule_phrase("skill_cost") == "by their cost on the skill text"


def test_shrink_plan_text_has_no_by_specific_on(client):
    spec = make_prompt_spec(client, SHRINK_PROMPTS[0])
    sid = client.post("/api/shrink", json={"spec_id": spec["id"]}).json()["shrink_id"]
    plan = client.post("/api/shrink/" + sid + "/plan", json={"rule": "specific"}).json()
    assert " by specific on " not in plan["summary"]
    assert "by how specific they are to the skill" in plan["summary"]


# 8. Stub consistency.

def test_stub_shrink_original_equals_the_spec_base(client):
    spec = make_prompt_spec(client, SHRINK_PROMPTS[0])
    bases = {}
    for item in spec["benchmarks"]:
        if item.get("base") is not None:
            bases[item["id"]] = item["base"]
    assert len(bases) > 0
    sid = client.post("/api/shrink", json={"spec_id": spec["id"]}).json()["shrink_id"]
    job = wait_job(client, client.post("/api/shrink/" + sid + "/run", json={}).json()["job_id"])
    original = job["result"]["source_scores"]
    for task in bases:
        assert original[task] == pytest.approx(bases[task], abs=1e-4)


def test_candidate_why_names_the_kept_skill_after_a_skill_edit(client):
    spec = make_prompt_spec(client, "make a smaller version of Llama 3.1 8B that only does math")
    spec["shrink"]["keep_skill"] = "medicine"
    edited = client.put("/api/spec/" + spec["id"], json=spec).json()
    for item in edited["shrink"]["candidates"]:
        facts = shrink_models.all_facts()[item["model"]]
        if "medicine" in facts["skills"]:
            assert "tagged for medicine" in item["why"]
            assert "a general model" not in item["why"]


def test_why_text_names_the_skill_when_tagged():
    facts = {"skills": ["math", "general"], "kind": "dense", "params_b": 3.0}
    assert shrink_models.why_text(facts, "math", 1.0).startswith("tagged for math")
    assert "a general model" not in shrink_models.why_text(facts, "math", 1.0)


# 9. Labels.

def test_research_proposals_carry_a_label():
    raw = [{"name": "Instruction following", "runnable_id": "if_guard", "kind": "guard"},
           {"name": "Law exam", "runnable_id": "mmlupro:law", "kind": "target"},
           {"name": "Some web set", "runnable_id": None, "kind": "target"}]
    out = bench_research.clean_proposals(raw, ["if_guard", "mmlupro:law", "mmlu"], [])
    assert [item["label"] for item in out] == ["IFEval", "MMLU-Pro law", None]
    assert specs_mod.benchmark_name("mmlu") == "MMLU"


# 10. Heal data pools.

@pytest.mark.parametrize("task_type,pool", [("math", "math"), ("knowledge", "general"), ("code", "code"),
                                            ("medicine", "medicine"), ("other", "general")])
def test_heal_pool_for_task_type(task_type, pool):
    assert specs_mod.heal_pool_for(task_type) == pool
    assert specs_mod.heal_data_for(task_type)[0] == pool


def test_run_overrides_heal_on_pools_never_on_benchmarks(client):
    for prompt in ["a math tutor", "a model that knows more facts, history and law"]:
        spec = draft(client, prompt)
        setup = specs_mod.run_overrides(spec, "stub", "none")["task_types"][spec["task_type"]]
        scored = [item["id"] for item in spec["benchmarks"]]
        for name in setup["heal_data"]:
            assert name not in scored
            assert "train" not in name and "aux" not in name
        assert setup["heal_data"][0] == specs_mod.heal_pool_for(spec["task_type"])


def test_extra_task_types_use_pools():
    assert specs_mod.EXTRA_TASK_TYPES["knowledge"]["heal_data"] == ["general"]


def test_builder_default_heal_is_a_pool(client):
    spec = make_spec(client)
    created = client.post("/api/builder", json={"spec_id": spec["id"]}).json()
    session = builder.get_session(created["builder_id"])
    assert ps.heal_data(builder.with_default_heal(session, ps.make_program("h"))) == "math"


def test_a_benchmark_heal_name_is_refused_in_plain_words(client):
    spec = make_spec(client)
    created = client.post("/api/builder", json={"spec_id": spec["id"]}).json()
    program = good_program(client)
    program["heal"]["data"] = "gsm8k_train"
    response = client.post("/api/builder/" + created["builder_id"] + "/size", json={"program": program})
    bad = client.post("/api/program/validate", json={"program": program}).json()
    assert "train set of the benchmark" in bad["errors"][0]
    assert response.status_code in (200, 400)


def test_pool_heal_names_are_accepted(client):
    for name in ["math", "general", "code", "medicine", "prose", "mix"]:
        program = good_program(client)
        program["heal"]["data"] = name
        assert client.post("/api/program/validate", json={"program": program}).json() == {"errors": []}


def test_heal_data_problem_helper():
    assert programs.heal_data_problem(ps.make_program("h", [], ps.make_heal(10, data="math"))) is None
    problem = programs.heal_data_problem(ps.make_program("h", [], ps.make_heal(10, data="mmlu_aux")))
    assert "cais/mmlu" in problem
    assert programs.heal_data_problem(ps.make_program("h")) is None


def test_the_stub_run_still_heals_on_a_pool(client):
    final = wait_done(client, start_run(client, make_spec(client, evals=4)))
    assert final["best"] is not None
