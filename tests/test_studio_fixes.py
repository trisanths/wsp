"""Tests for the studio server fixes: stop, snapshots, finite numbers, sizing, shrink limits, best score,
Modal heal, shrink spec drift, research on shrink specs, and the deploy check. CPU only, no network."""

import asyncio
import json
import math
import threading
import time
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from studio.server import bench_research
from studio.server import modal_backend
from studio.server import runs as runs_mod
from studio.server import shrink as shrink_mod
from studio.server import sizing
from studio.server.app import app, snapshot_stream
from studio.server.runstate import RunState
from studio.server.tracking import StudioSwarm
from tests.test_studio_server import make_spec, snapshot, start_run, wait_done, wait_for
from wsp.swarm import program_schema as ps


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("WSP_STUDIO_OUT", str(tmp_path))
    monkeypatch.setenv("WSP_STUDIO_BACKEND", "stub")
    monkeypatch.setenv("WSP_STUDIO_LLM", "none")
    monkeypatch.setenv("WSP_STUDIO_PACE_S", "0")
    return TestClient(app)


# 1. Stop.

def test_stop_skips_the_held_out_pass():
    swarm = object.__new__(StudioSwarm)
    swarm.controller = SimpleNamespace(gate=SimpleNamespace(stopped=True))
    assert swarm.final_heldout() == ([], [])


def test_stop_skips_the_seed_program_eval():
    swarm = object.__new__(StudioSwarm)
    swarm.controller = SimpleNamespace(gate=SimpleNamespace(stopped=True), seed_program={"host": "x"})
    swarm.env = None
    swarm.score_seed_program()


def test_a_stopped_run_says_so(client, monkeypatch):
    monkeypatch.setenv("WSP_STUDIO_PACE_S", "0.05")
    run_id = start_run(client, make_spec(client, evals=40))
    wait_for(lambda: len(snapshot(client, run_id)["curve"]) >= 2)
    client.post("/api/runs/" + run_id + "/stop")
    final = wait_done(client, run_id)
    assert final["stopped_early"] is True
    assert any("held-out check was skipped" in item["text"] for item in final["activity"])


def test_a_finished_run_is_not_stopped_early(client):
    run_id = start_run(client, make_spec(client, evals=4))
    assert wait_done(client, run_id)["stopped_early"] is False


# 2. Snapshots off the event loop.

def test_stream_survives_a_failing_snapshot():
    class Broken:
        version = 1
        status = "done"

        def snapshot(self):
            raise RuntimeError("no scores")

    async def first_frames():
        frames = []
        async for frame in snapshot_stream(SimpleNamespace(state=Broken())):
            frames.append(frame)
        return frames

    frames = asyncio.run(first_frames())
    assert frames[0].startswith("event: error")
    assert "no scores" in frames[0]


def test_snapshot_never_asks_the_env_for_base_scores():
    calls = []

    class Env:
        def base_scores(self, split):
            calls.append(split)
            return {"gsm8k": 0.3}

    state = SimpleNamespace(env=Env())
    state.load_base_scores = RunState.load_base_scores.__get__(state)
    state.lock = threading.RLock()
    state.touch = lambda: None
    state.add_activity = lambda *args: None
    state.load_base_scores()
    assert calls == ["search"] and state.base_scores == {"gsm8k": 0.3}


def test_snapshot_reads_the_cached_scores(client):
    run_id = start_run(client, make_spec(client, evals=2))
    run = runs_mod.get_run(run_id)
    calls = []
    original = run.state.env.base_scores
    run.state.env.base_scores = lambda split: calls.append(split) or original(split)
    wait_done(client, run_id)
    del calls[:]
    for _ in range(3):
        snapshot(client, run_id)
    assert calls == []


# 3. Finite numbers.

@pytest.mark.parametrize("body", [{"size": {"max_params_b": "nan"}}, {"limits": {"max_run_params_b": "nan"}},
                                  {"limits": {"max_total_params_b": "inf"}}])
def test_nan_in_a_spec_edit_does_not_poison_the_spec(client, body):
    spec = make_spec(client)
    response = client.put("/api/spec/" + spec["id"], json=body)
    assert response.status_code == 200, response.text
    assert client.get("/api/spec/" + spec["id"]).status_code == 200


def test_nan_target_is_ignored(client):
    spec = make_spec(client)
    first = spec["benchmarks"][0]
    spec["benchmarks"][0] = dict(first, target="nan", floor="inf")
    response = client.put("/api/spec/" + spec["id"], json=spec)
    assert response.status_code == 200, response.text
    json.dumps(response.json(), allow_nan=False)


def test_research_scores_must_be_finite():
    assert bench_research.clean_score("nan") is None
    assert bench_research.clean_score("inf") is None
    assert bench_research.clean_score("31") == 0.31


# 4. Unsized parts.

def test_a_part_with_no_size_is_refused(monkeypatch):
    monkeypatch.setattr(sizing, "part_size_b", lambda part_id, ctx: None)
    ctx = SimpleNamespace(host_b=1.24, host_hidden=2048)
    program = ps.make_program("host", [ps.make_graft("mystery.part", 6, glue_rank=64)], None)
    est = sizing.estimate(program, ctx)
    problems = sizing.limit_problems({"max_total_params_b": 1.5, "max_run_params_b": 1.5, "max_heal_min": 20}, est)
    assert len(problems) == 1
    assert "mystery.part" in problems[0]


# 5. Measured shrink size.

def shrink_spec_for(client):
    response = client.post("/api/spec", data={"prompt": "keep only the math experts of a 30B MoE"})
    return response.json()


def test_a_measured_shrink_over_the_limit_fails(client):
    spec = shrink_spec_for(client)
    limit = spec["limits"]["max_total_params_b"]
    plan = spec["shrink"]
    scores = {"gsm8k": 0.8}
    result = shrink_mod.finish(spec, plan, scores, limit + 1.0, limit, {"kind": "moe"}, {"gsm8k": 0.9},
                               {"gsm8k": 0.5}, None, {}, [], {}, False)
    assert result["comparison"]["verdict"] == "failed"
    assert "Over the total size limit" in result["comparison"]["headline"]
    assert len(result["limit_problems"]) >= 1


def test_a_measured_shrink_inside_the_limit_is_not_failed(client):
    spec = shrink_spec_for(client)
    limit = spec["limits"]["max_total_params_b"]
    result = shrink_mod.finish(spec, spec["shrink"], {"gsm8k": 0.8}, limit * 0.5, 0.01, {"kind": "moe"},
                               {"gsm8k": 0.9}, {"gsm8k": 0.5}, None, {}, [], {}, False)
    assert "limit_problems" not in result


def test_measured_run_size_is_checked():
    problems = sizing.measured_problems({"max_total_params_b": 5.0, "max_run_params_b": 1.0}, 2.0, 1.5)
    assert len(problems) == 1 and "per-token" in problems[0]


# 6. Collapsed programs.

def scored_row(name, status, gsm, guard):
    program = {"host": "h", "name": name}
    return {"program_hash": name, "status": status, "scores": {"gsm8k": gsm, "if_guard": guard},
            "combined": gsm, "program": program, "describe": name, "agent": "planner-1", "round": 0}


def test_a_collapsed_program_does_not_set_the_best(client):
    run_id = start_run(client, make_spec(client, evals=2))
    wait_done(client, run_id)
    state = runs_mod.get_run(run_id).state
    with state.lock:
        state.curve = []
        state.best_target = None
        state.target_score = lambda scores: scores.get("gsm8k")
        state.programs = {}
    state.record_eval(scored_row("c1", "collapsed", 0.62, 0.1))
    state.record_eval(scored_row("v1", "ok", 0.55, 0.9))
    state.record_eval(scored_row("v2", "ok", 0.55, 0.9))
    assert state.best_target == 0.55
    assert [point["best"] for point in state.curve] == [None, 0.55, 0.55]
    assert [point["score"] for point in state.curve] == [0.62, 0.55, 0.55]
    flags = [item["data"]["is_best"] for item in state.activity if item["kind"] == "eval"][-3:]
    assert flags == [False, True, False]
    assert state.agents["planner-1"]["best"] == 0.55


# 7. Modal heal.

def test_heal_sends_the_shortened_program(monkeypatch):
    sent = []

    class Fake:
        def remote(self, program, split, tasks, token, fraction, options):
            sent.append(ps.heal_steps(program))
            return {"ok": True, "scores": {"gsm8k": 0.3}, "seconds": {}, "graft_report": [{"id": "x"}]}

    monkeypatch.setattr(modal_backend, "lookup", lambda name: Fake())
    backend = object.__new__(modal_backend.ModalCompileEval)
    backend.lock = threading.RLock()
    backend.score_reports = {}
    backend.config = {"fidelity_cost": {}, "max_heal_min": None}
    backend.options_for = lambda program: {}
    backend.function = lambda base, program: Fake()
    monkeypatch.setattr(modal_backend, "fidelity_cost", lambda config, fidelity: 0.34)
    full = ps.make_program("h", [], ps.make_heal(300))
    short = ps.with_heal_steps(full, 50)
    handle = {"program": ps.normalize(full)}
    healed = backend.heal(handle, short)
    backend.evaluate(healed["handle"], "search", ["gsm8k"], "t", "low")
    assert sent == [50]
    assert backend.report_for(full) == [{"id": "x"}]


# 11. Shrink session and edited specs.

def test_shrink_session_follows_spec_edits(client):
    spec = shrink_spec_for(client)
    sid = client.post("/api/shrink", json={"spec_id": spec["id"]}).json()["shrink_id"]
    plan = client.post("/api/shrink/" + sid + "/plan", json={"keep_experts": 64}).json()
    assert plan["ok"] is False
    limits = dict(spec["limits"], max_total_params_b=40.0, max_run_params_b=40.0)
    response = client.put("/api/spec/" + spec["id"], json={"limits": limits})
    assert response.status_code == 200, response.text
    again = client.post("/api/shrink/" + sid + "/plan", json={}).json()
    assert again["limits"]["max_total_params_b"] == response.json()["limits"]["max_total_params_b"]
    assert again["limits"]["max_total_params_b"] > spec["limits"]["max_total_params_b"]


# 12. Research on a shrink spec.

def test_research_does_not_add_benchmarks_to_a_shrink_spec(client):
    spec = shrink_spec_for(client)
    before = [item["id"] for item in spec["benchmarks"]]
    proposals = [{"runnable_id": "if_guard", "kind": "guard", "reference_scores": [], "suggested_target": None,
                  "in_spec": False},
                 {"runnable_id": "mmlupro:health", "kind": "target", "reference_scores": [],
                  "suggested_target": 0.3, "in_spec": False}]
    bench_research.apply_to_spec(spec, proposals)
    assert [item["id"] for item in spec["benchmarks"]] == before


# 9 and 10. Heal limit and deploy check.

def test_heal_steps_of_every_arm_count_against_the_limit(client):
    spec = shrink_spec_for(client)
    estimate = sizing.estimate_shrink(spec["shrink"])
    limits = {"max_total_params_b": 100.0, "max_run_params_b": 100.0, "max_heal_min": estimate["heal_min"] * 1.5}
    assert sizing.shrink_check(spec["shrink"], limits, 1)[1] == []
    assert len(sizing.shrink_check(spec["shrink"], limits, 3)[1]) == 1


def test_run_overrides_carry_the_heal_limit(client):
    from studio.server import specs as specs_mod
    spec = make_spec(client)
    overrides = specs_mod.run_overrides(spec, "stub", "none")
    assert overrides["max_heal_min"] == spec["limits"]["max_heal_min"]


def test_deploy_check_warns_on_a_different_hash(monkeypatch):
    class Fake:
        def remote(self):
            return "0000"

    monkeypatch.setattr(modal_backend, "lookup", lambda name: Fake())
    monkeypatch.setattr(modal_backend, "_deploy_check", {"done": False, "warning": None})
    assert "older than the local code" in modal_backend.deployed_warning()


def test_deploy_check_is_quiet_on_a_match(monkeypatch):
    from studio import code_version

    class Fake:
        def remote(self):
            return code_version.local_hash(modal_backend.REPO_ROOT)

    monkeypatch.setattr(modal_backend, "lookup", lambda name: Fake())
    monkeypatch.setattr(modal_backend, "_deploy_check", {"done": False, "warning": None})
    assert modal_backend.deployed_warning() is None
