"""Tests for the wsp studio server. CPU only, stub backend, no network."""

import json
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


def good_program(client, heal=True):
    part = math_part(client)
    heal_dict = None
    if heal:
        heal_dict = ps.make_heal(300)
    return ps.make_program(HOST, [ps.make_graft(part["id"], part["default_at"], glue_rank=64)], heal_dict)


def test_health(client):
    body = client.get("/api/health").json()
    assert body == {"ok": True, "backend": "stub", "llm": "none", "stub": True}


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
    assert edited["size"]["max_params_b"] == 8.0
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
    assert len(palette["parts"]) == 40
    part = palette["parts"][0]
    for key in ["id", "group", "kind", "label", "donor", "layers", "params_m", "default_at", "summary"]:
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
