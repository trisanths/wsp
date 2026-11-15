"""Tests for the wsp studio server. CPU only, stub backend, no network."""

import json
import os
import time

import pytest
from fastapi.testclient import TestClient

from studio.server import specs as specs_mod
from studio.server.app import app
from wsp.swarm import program_schema as ps

HOST = "unsloth/Llama-3.2-1B-Instruct"


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("WSP_STUDIO_OUT", str(tmp_path))
    monkeypatch.setenv("WSP_STUDIO_BACKEND", "stub")
    monkeypatch.setenv("WSP_STUDIO_LLM", "none")
    monkeypatch.setenv("WSP_STUDIO_PACE_S", "0")
    return TestClient(app)


def wait_for(check, timeout=60.0, step=0.05):
    """Call check() until it returns something true. Fails the test on timeout."""
    end = time.time() + timeout
    while time.time() < end:
        value = check()
        if value:
            return value
        time.sleep(step)
    raise AssertionError("timed out")


def make_spec(client, evals=6):
    csv_text = b"question,answer\n" + b"1+1,2\n" * 6
    response = client.post("/api/spec", data={"prompt": "I want a math tutor for word problems on my laptop"},
                           files=[("files", ("homework.csv", csv_text, "text/csv"))])
    assert response.status_code == 200
    spec = response.json()
    spec["budget"]["evals"] = evals
    response = client.put("/api/spec/" + spec["id"], json=spec)
    assert response.status_code == 200
    return response.json()


def start_run(client, spec, program=None):
    response = client.post("/api/runs", json={"spec_id": spec["id"], "program": program})
    assert response.status_code == 200, response.text
    return response.json()["run_id"]


def snapshot(client, run_id):
    return client.get("/api/runs/" + run_id).json()


def wait_done(client, run_id, timeout=60.0):
    return wait_for(lambda: snapshot(client, run_id)["status"] in ("done", "error") and snapshot(client, run_id),
                    timeout)


def math_part(client):
    palette = client.get("/api/catalog").json()
    for part in palette["parts"]:
        if part["kind"] == "block" and "Math" in part["label"]:
            return part
    raise AssertionError("no math block in the catalog")


def tap_program():
    program = ps.make_program(HOST)
    program["taps"] = [{"donor": "Qwen/Qwen2.5-7B-Instruct", "donor_layer": 20, "at": 10, "attn_dim": 256,
                        "heads": 8, "tap_id": None}]
    program["tap_train"] = {"data": "mmlu_aux", "steps": 100, "batch_size": 8, "seq_len": 256, "branch_lr": 0.0003,
                            "gate_lr": 0.03}
    return ps.normalize(program)


def good_program(client, heal=True):
    part = math_part(client)
    heal_dict = None
    if heal:
        heal_dict = ps.make_heal(300)
    return ps.make_program(HOST, [ps.make_graft(part["id"], part["default_at"], glue_rank=64)], heal_dict)


def test_health(client):
    body = client.get("/api/health").json()
    assert body == {"ok": True, "product": "Sapien", "backend": "stub", "llm": "none", "llm_model": None,
                    "stub": True}


def test_spec_from_prompt_and_upload(client):
    response = client.post("/api/spec", data={"prompt": "I want a math tutor that solves word problems on my laptop"},
                           files=[("files", ("homework.csv", b"q,a\n" + b"1,2\n" * 9, "text/csv"))])
    assert response.status_code == 200
    spec = response.json()
    assert spec["stub"] is True
    assert spec["drafted_by"] == "rules"
    assert spec["task_type"] == "math"
    assert spec["host"] == HOST
    assert spec["title"] == "Laptop math tutor"
    assert spec["size"]["max_params_b"] == 1.5
    assert spec["limits"] == {"max_total_params_b": 1.5, "max_run_params_b": 1.5, "max_heal_min": 20}
    assert spec["comparators"] == {"distill": {}}
    assert spec["budget"]["evals"] == 30
    assert spec["budget"]["compute_min"] == 45
    assert spec["files"] == [{"name": "homework.csv", "rows": 9, "score_rows": 3, "heal_rows": 6}]
    by_id = {}
    for item in spec["benchmarks"]:
        by_id[item["id"]] = item
    assert by_id["gsm8k"]["kind"] == "target"
    assert by_id["gsm8k"]["lora"] == 0.467
    assert by_id["gsm8k"]["target"] == 0.5
    assert by_id["gsm8k"]["base"] == 0.25  # the stub world's own base
    assert by_id["if_guard"]["kind"] == "guard"
    assert "floor" in by_id["if_guard"]
    assert len(spec["available_benchmarks"]) > 5
    assert len(spec["ingredients"]) > 0
    again = client.get("/api/spec/" + spec["id"]).json()
    assert again == spec
    assert client.get("/api/spec/sp_missing").status_code == 404


def test_spec_put_clamps_and_checks(client):
    spec = make_spec(client)
    spec["budget"]["evals"] = 5000
    spec["size"]["max_params_b"] = 99
    spec["title"] = "My tutor"
    response = client.put("/api/spec/" + spec["id"], json=spec)
    edited = response.json()
    assert edited["budget"]["evals"] == 120
    assert edited["size"]["max_params_b"] == 32.0
    assert edited["limits"]["max_total_params_b"] == 32.0
    assert edited["limits"]["max_run_params_b"] == 1.5
    assert edited["title"] == "My tutor"
    edited["benchmarks"] = [{"id": "mmlupro:biology", "kind": "target"}]
    response = client.put("/api/spec/" + spec["id"], json=edited)
    assert response.status_code == 200
    assert response.json()["benchmarks"][0]["target"] is not None
    edited["benchmarks"] = [{"id": "if_guard", "kind": "guard"}]
    response = client.put("/api/spec/" + spec["id"], json=edited)
    assert response.status_code == 400
    assert "error" in response.json()


def test_count_rows():
    assert specs_mod.count_rows("a.csv", b"h1,h2\n1,2\n3,4\n") == 2
    assert specs_mod.count_rows("a.jsonl", b'{"a":1}\n\n{"a":2}\n') == 2
    assert specs_mod.count_rows("a.json", b'[1,2,3]') == 3
    assert specs_mod.count_rows("a.json", b'{"data":[1,2]}') == 2
    assert specs_mod.count_rows("a.txt", b"x\n\ny\nz\n") == 3
    assert specs_mod.split_rows(1) == (1, 0)
    assert specs_mod.split_rows(9) == (3, 6)
    assert specs_mod.size_from_prompt("fits on a phone", 1.24) == 1.3
    assert specs_mod.size_from_prompt("under 2B please", 1.24) == 2.0


def test_run_finishes_with_results(client):
    spec = make_spec(client)
    run_id = start_run(client, spec)
    snap = wait_done(client, run_id)
    assert snap["status"] == "done"
    assert snap["stub"] is True
    assert snap["backend"] == "stub"
    assert snap["spec"]["id"] == spec["id"]
    assert snap["budget"]["evals_total"] == 6
    assert snap["budget"]["evals_used"] > 0
    assert len(snap["leaderboard"]) > 0
    assert snap["leaderboard"][0]["rank"] == 1
    assert len(snap["curve"]) > 0
    assert snap["curve"][0]["eval"] == 1
    assert snap["best"]["program_hash"] == snap["leaderboard"][0]["program_hash"]
    assert snap["sources"][0]["role"] == "host"
    assert "gsm8k" in snap["base_scores"]
    ids = []
    for agent in snap["agents"]:
        ids.append(agent["id"])
    assert ids == ["analyst", "planner-1", "planner-2", "planner-3", "judge"]
    attempts = 0
    for agent in snap["agents"]:
        attempts += agent["attempts"]
    assert attempts > 0
    assert len(snap["activity"]) > 0
    detail = client.get("/api/runs/" + run_id + "/agents/planner-1").json()
    assert detail["agent"]["id"] == "planner-1"
    assert len(detail["tools"]) > 0
    assert len(detail["history"]) > 0
    assert detail["current"]["text"].startswith("host ")
    assert client.get("/api/runs/" + run_id + "/agents/nobody").status_code == 404
    assert client.get("/api/runs/r_missing").status_code == 404


def test_pause_and_resume(client, monkeypatch):
    monkeypatch.setenv("WSP_STUDIO_PACE_S", "0.05")
    spec = make_spec(client, evals=10)
    run_id = start_run(client, spec)
    wait_for(lambda: snapshot(client, run_id)["budget"]["evals_used"] > 0)
    assert client.post("/api/runs/" + run_id + "/pause").json() == {"ok": True}
    assert snapshot(client, run_id)["status"] == "paused"
    time.sleep(0.4)
    first = snapshot(client, run_id)
    time.sleep(0.5)
    second = snapshot(client, run_id)
    assert second["status"] == "paused"
    assert len(second["activity"]) == len(first["activity"])
    assert second["agents"][0]["status"] == "paused"
    assert client.post("/api/runs/" + run_id + "/resume").json() == {"ok": True}
    final = wait_done(client, run_id)
    assert final["status"] == "done"
    assert final["budget"]["evals_used"] == 10


def test_stop_keeps_best(client, monkeypatch):
    monkeypatch.setenv("WSP_STUDIO_PACE_S", "0.05")
    spec = make_spec(client, evals=40)
    run_id = start_run(client, spec)
    wait_for(lambda: len(snapshot(client, run_id)["curve"]) >= 2)
    assert client.post("/api/runs/" + run_id + "/stop").json() == {"ok": True}
    final = wait_done(client, run_id)
    assert final["status"] == "done"
    assert final["budget"]["evals_used"] < 40
    assert final["best"] is not None
    assert len(final["leaderboard"]) > 0


def test_stop_while_paused(client, monkeypatch):
    monkeypatch.setenv("WSP_STUDIO_PACE_S", "0.05")
    spec = make_spec(client, evals=40)
    run_id = start_run(client, spec)
    wait_for(lambda: len(snapshot(client, run_id)["curve"]) >= 1)
    client.post("/api/runs/" + run_id + "/pause")
    client.post("/api/runs/" + run_id + "/stop")
    assert wait_done(client, run_id)["status"] == "done"


def test_nudge_is_applied(client, monkeypatch):
    monkeypatch.setenv("WSP_STUDIO_PACE_S", "0.05")
    spec = make_spec(client, evals=10)
    run_id = start_run(client, spec)
    response = client.post("/api/runs/" + run_id + "/agents/planner-1/nudge", json={"text": "try deeper layers"})
    assert response.json() == {"ok": True}
    detail = client.get("/api/runs/" + run_id + "/agents/planner-1").json()
    assert detail["nudges"][0]["text"] == "try deeper layers"
    wait_done(client, run_id)
    detail = client.get("/api/runs/" + run_id + "/agents/planner-1").json()
    assert detail["nudges"][0]["applied"] is True
    assert detail["agent"]["nudges_pending"] == 0
    kinds = []
    for item in snapshot(client, run_id)["activity"]:
        kinds.append(item["kind"])
    assert "nudge" in kinds
    assert "nudge_applied" in kinds
    assert client.post("/api/runs/" + run_id + "/agents/planner-1/nudge", json={"text": " "}).status_code == 400
    assert client.post("/api/runs/" + run_id + "/agents/nobody/nudge", json={"text": "x"}).status_code == 404


def test_stream_sends_snapshots(client):
    spec = make_spec(client)
    run_id = start_run(client, spec)
    wait_done(client, run_id)
    events = []
    with client.stream("GET", "/api/runs/" + run_id + "/stream") as response:
        assert response.headers["content-type"].startswith("text/event-stream")
        name = None
        for line in response.iter_lines():
            if line.startswith("event: "):
                name = line[len("event: "):]
            if line.startswith("data: "):
                events.append((name, json.loads(line[len("data: "):])))
    assert len(events) >= 1
    assert events[0][0] == "snapshot"
    assert events[0][1]["run_id"] == run_id
    assert events[-1][1]["status"] == "done"


def test_catalog(client):
    palette = client.get("/api/catalog").json()
    assert palette["stub"] is True
    assert palette["host"]["layers"] == 16
    stub_parts = [part for part in palette["parts"] if part["source"] == "stub"]
    assert len(stub_parts) == 40
    part = palette["parts"][0]
    for key in ["id", "group", "kind", "label", "donor", "layers", "params_m", "default_at", "summary", "status",
                "evidence", "source", "usable"]:
        assert key in part
    assert part["group"] == "add"
    assert len(palette["masks"]) > 0
    assert palette["masks"][0]["group"] == "remove"
    assert [item["id"] for item in palette["train"]] == ["lora", "heal"]


def test_program_text_round_trip(client):
    program = good_program(client)
    text = client.post("/api/program/text", json={"program": program}).json()["text"]
    assert text.startswith("host " + HOST)
    assert "heal steps 300" in text
    parsed = client.post("/api/program/parse", json={"text": text}).json()["program"]
    assert parsed == program
    assert client.post("/api/program/validate", json={"program": program}).json() == {"errors": []}
    bad = ps.make_program(HOST, [ps.make_graft("no_such_part", 3)], None)
    assert len(client.post("/api/program/validate", json={"program": bad}).json()["errors"]) > 0


def test_program_parse_error_has_line(client):
    response = client.post("/api/program/parse", json={"text": "host " + HOST + "\nbogus 1\n"})
    assert response.status_code == 400
    body = response.json()
    assert body["line"] == 2
    assert "bogus" in body["error"]


def test_builder_actions_and_suggestions(client):
    spec = make_spec(client)
    created = client.post("/api/builder", json={"spec_id": spec["id"]}).json()
    builder_id = created["builder_id"]
    assert created["budget"]["evals_total"] == 12
    program = good_program(client)

    def run_job(action, body_program):
        response = client.post("/api/builder/" + builder_id + "/action", json={"program": body_program,
                                                                               "action": action})
        assert response.status_code == 200, response.text
        job_id = response.json()["job_id"]
        return wait_for(lambda: client.get("/api/jobs/" + job_id).json()["status"] in ("done", "error")
                        and client.get("/api/jobs/" + job_id).json())

    compiled = run_job("compile", program)
    assert compiled["status"] == "done"
    assert compiled["stub"] is True
    assert len(compiled["result"]["fit"]) == 1
    assert compiled["result"]["params_b"] > 1.24
    assert compiled["result"]["scores"] is None
    assert compiled["budget"]["evals_used"] == 0
    assert compiled["budget"]["compute_used_s"] > 0

    scored = run_job("score", program)
    assert scored["status"] == "done"
    assert scored["result"]["status"] == "ok"
    assert "gsm8k" in scored["result"]["scores"]
    assert scored["result"]["combined"] is not None
    assert "gsm8k" in scored["result"]["control"]["scores"]
    assert scored["budget"]["evals_used"] == 2

    healed = run_job("heal", good_program(client, heal=False))
    assert healed["result"]["program"]["heal"]["steps"] == 300
    trained = run_job("train", program)
    assert trained["result"]["program"]["lora"]["rank"] == 16

    bare = good_program(client, heal=False)
    response = client.post("/api/builder/" + builder_id + "/suggest", json={"program": bare})
    suggestions = response.json()["suggestions"]
    titles = []
    for item in suggestions:
        titles.append(item["title"])
        assert set(["title", "detail", "tag", "program"]) <= set(item.keys())
    assert "Add a heal pass" in titles
    assert "Use rank-64 glue" not in titles
    empty = client.post("/api/builder/" + builder_id + "/suggest", json={"program": ps.make_program(HOST)}).json()
    assert empty["suggestions"][0]["title"].startswith("Add ")
    assert empty["suggestions"][0]["tag"].startswith("fit ")
    assert len(empty["suggestions"][0]["program"]["grafts"]) == 1

    bad = client.post("/api/builder/" + builder_id + "/action", json={"program": program, "action": "explode"})
    assert bad.status_code == 400
    invalid = ps.make_program(HOST, [ps.make_graft("no_such_part", 3)], None)
    bad = client.post("/api/builder/" + builder_id + "/action", json={"program": invalid, "action": "score"})
    assert bad.status_code == 400
    assert client.post("/api/builder/b_missing/action", json={"program": program, "action": "score"}).status_code == 404


def test_builder_handoff_runs_the_program_first(client):
    spec = make_spec(client)
    builder_id = client.post("/api/builder", json={"spec_id": spec["id"]}).json()["builder_id"]
    program = good_program(client)
    response = client.post("/api/builder/" + builder_id + "/handoff", json={"program": program})
    assert response.status_code == 200
    run_id = response.json()["run_id"]
    final = wait_done(client, run_id)
    assert final["status"] == "done"
    agents = []
    for entry in final["leaderboard"]:
        agents.append(entry["agent"])
    assert "you" in agents


def test_run_rejects_bad_requests(client):
    assert client.post("/api/runs", json={"spec_id": "sp_missing"}).status_code == 404
    spec = make_spec(client)
    invalid = ps.make_program(HOST, [ps.make_graft("no_such_part", 3)], None)
    response = client.post("/api/runs", json={"spec_id": spec["id"], "program": invalid})
    assert response.status_code == 400
    assert "error" in response.json()
    assert client.get("/api/nothing").status_code == 404


# Modal backend tests. A fake function object stands in for every Modal function, so CI never calls Modal.

MODAL_PART = "mlp_cluster_qwen25math15b_l16_gsm8k"


def fake_entries(program):
    entries = []
    for graft in program["grafts"]:
        entries.append({"id": graft["id"], "host_layer": graft["at"], "fit": "ridge", "fit_r2": 0.8, "fit_pairs": 100,
                        "part_params": 300000, "glue_params": 100000, "control": graft.get("control")})
    return entries


class FakeModalFunction:
    """Records the arguments of every call and returns a canned result."""

    def __init__(self, name, calls):
        self.name = name
        self.calls = calls

    def remote(self, *args):
        self.calls.append((self.name, args))
        if self.name.startswith("probe_remote"):
            return {"ok": True, "error": None, "graft_report": fake_entries(args[0]), "seconds": {"compile": 20.0},
                    "gpu": "NVIDIA L40S"}
        if self.name.startswith("score_remote"):
            program, split, tasks = args[0], args[1], args[2]
            scores = {}
            for task in tasks:
                scores[task] = 0.5
            return {"ok": True, "error": None, "graft_report": fake_entries(program), "heal_loss": [1.0, 0.5],
                    "scores": scores, "seconds": {"compile": 30.0, "heal": 15.0, "eval": 10.0},
                    "gpu": "NVIDIA L40S"}
        if self.name == "base_remote":
            tasks = args[2]
            scores = {}
            for task in tasks:
                scores[task] = 0.21
            return {"ok": True, "error": None, "scores": scores, "seconds": {"eval": 9.0}, "gpu": "NVIDIA L40S"}
        raise AssertionError("unexpected modal function " + self.name)


@pytest.fixture
def modal_client(tmp_path, monkeypatch):
    from studio.server import modal_backend

    calls = []
    monkeypatch.setenv("WSP_STUDIO_OUT", str(tmp_path))
    monkeypatch.setenv("WSP_STUDIO_BACKEND", "modal")
    monkeypatch.setenv("WSP_STUDIO_LLM", "none")
    monkeypatch.setenv("WSP_STUDIO_PACE_S", "0")
    monkeypatch.setattr(modal_backend, "lookup", lambda name: FakeModalFunction(name, calls))
    test_client = TestClient(app)
    test_client.modal_calls = calls
    return test_client


def modal_program(steps=100):
    graft = ps.make_graft(MODAL_PART, 6, glue_rank=64)
    return ps.make_program(HOST, [graft], ps.make_heal(steps))


def run_builder_job(client, builder_id, program, action):
    response = client.post("/api/builder/" + builder_id + "/action", json={"program": program, "action": action})
    assert response.status_code == 200, response.text
    job_id = response.json()["job_id"]
    return wait_for(lambda: client.get("/api/jobs/" + job_id).json()["status"] in ("done", "error")
                    and client.get("/api/jobs/" + job_id).json())


def test_modal_catalog_shape(modal_client):
    palette = modal_client.get("/api/catalog").json()
    assert palette["stub"] is False
    assert "tap_train_default" not in palette
    for part in palette["parts"]:
        assert part["kind"] in ("block", "attn_heads", "mlp_cluster", "expert", "inserted_layer")
        assert part["status"] in ("candidate", "validated", "rejected")
    by_id = {}
    for part in palette["parts"]:
        by_id[part["id"]] = part
    assert by_id[MODAL_PART]["usable"] is True
    assert by_id[MODAL_PART]["source"] == "library"
    assert len(palette["masks"]) > 0
    assert os.path.exists(os.path.join(os.environ["WSP_STUDIO_OUT"], "catalog_modal.jsonl"))


def test_modal_builder_scores_a_graft_program(modal_client):
    spec = make_spec(modal_client)
    builder_id = modal_client.post("/api/builder", json={"spec_id": spec["id"]}).json()["builder_id"]
    program = modal_program()

    compiled = run_builder_job(modal_client, builder_id, program, "compile")
    assert compiled["status"] == "done", compiled
    assert compiled["stub"] is False
    key = MODAL_PART + "@6"
    assert compiled["result"]["fit"] == {key: 0.8}
    assert compiled["budget"]["compute_used_s"] >= 20
    assert modal_client.modal_calls[0][0] == "probe_remote"

    scored = run_builder_job(modal_client, builder_id, program, "score")
    assert scored["status"] == "done", scored
    assert scored["result"]["status"] == "ok"
    assert scored["result"]["scores"]["gsm8k"] == 0.5
    assert scored["result"]["compute_s"] == 55.0
    comparison = scored["result"]["comparison"]
    assert comparison["stub"] is False
    assert comparison["verdict"] in ("failed", "unproven")
    names = []
    for name, args in modal_client.modal_calls:
        names.append(name)
    assert names[:2] == ["probe_remote", "score_remote"]

    trained = run_builder_job(modal_client, builder_id, program, "train")
    assert trained["status"] == "done", trained
    sent = modal_client.modal_calls[-1][1][0]
    assert len(sent["grafts"]) == 1
    assert sent["lora"]["rank"] == 16


def test_modal_builder_refuses_taps(modal_client):
    spec = make_spec(modal_client)
    builder_id = modal_client.post("/api/builder", json={"spec_id": spec["id"]}).json()["builder_id"]
    program = tap_program()
    response = modal_client.post("/api/builder/" + builder_id + "/action", json={"program": program,
                                                                                 "action": "score"})
    assert response.status_code == 400
    assert "second model" in response.json()["error"]
    assert len(modal_client.modal_calls) == 0


def test_modal_big_donor_uses_the_h100_function(modal_client):
    from studio.server import modal_backend

    assert modal_backend.model_size_b("Qwen/Qwen2.5-14B-Instruct") == 14.0
    assert modal_backend.model_size_b("unsloth/gemma-2-9b-it") == 9.0
    assert modal_backend.model_size_b("microsoft/Phi-3.5-mini-instruct") == 0.0


def test_modal_base_scores_are_cached(tmp_path, monkeypatch):
    from studio.server import catalog as catalog_mod
    from studio.server import modal_backend

    monkeypatch.setenv("WSP_STUDIO_OUT", str(tmp_path))
    calls = []
    monkeypatch.setattr(modal_backend, "lookup", lambda name: FakeModalFunction(name, calls))
    config, catalog = catalog_mod.load_catalog("modal")
    backend = modal_backend.ModalCompileEval(config, catalog)

    configured = backend.base_scores("search", ["gsm8k"])
    assert configured == {"gsm8k": config["base_scores"]["search"]["gsm8k"]}
    assert len(calls) == 0

    first = backend.base_scores("search", ["humaneval", "gsm8k"])
    assert first["humaneval"] == 0.21
    assert len(calls) == 1
    assert calls[0][1][2] == ["humaneval"]
    second = backend.base_scores("search", ["humaneval"])
    assert second == {"humaneval": 0.21}
    assert len(calls) == 1

    path = os.path.join(str(tmp_path), "base_scores_modal.json")
    with open(path, encoding="utf-8") as handle:
        saved = json.load(handle)
    assert saved[config["host"]]["search"]["humaneval"] == 0.21
    fresh = modal_backend.ModalCompileEval(config, catalog)
    assert fresh.base_scores("search", ["humaneval"]) == {"humaneval": 0.21}
    assert len(calls) == 1
