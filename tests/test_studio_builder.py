"""Tests for the builder fixes: part id checks in program text, budget labels, and the Score cost. CPU only."""

import pytest
from fastapi.testclient import TestClient

from studio.server.app import app
from tests.test_studio_server import HOST, good_program, make_spec, wait_for


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("WSP_STUDIO_OUT", str(tmp_path))
    monkeypatch.setenv("WSP_STUDIO_BACKEND", "stub")
    monkeypatch.setenv("WSP_STUDIO_LLM", "none")
    monkeypatch.setenv("WSP_STUDIO_PACE_S", "0")
    return TestClient(app)


def test_parse_refuses_an_unknown_part_id_with_a_line_number(client):
    text = "host " + HOST + "\n\ngraft at at 5\n"
    response = client.post("/api/program/parse", json={"text": text})
    assert response.status_code == 400
    body = response.json()
    assert body["line"] == 3
    assert "unknown part id at" in body["error"]


def test_parse_accepts_a_catalog_part(client):
    program = good_program(client)
    text = client.post("/api/program/text", json={"program": program}).json()["text"]
    response = client.post("/api/program/parse", json={"text": text})
    assert response.status_code == 200, response.text
    assert response.json()["program"]["grafts"][0]["id"] == program["grafts"][0]["id"]


def run_score(client, builder_id, program):
    response = client.post("/api/builder/" + builder_id + "/action", json={"program": program, "action": "score"})
    assert response.status_code == 200, response.text
    job_id = response.json()["job_id"]
    return wait_for(lambda: client.get("/api/jobs/" + job_id).json()["status"] in ("done", "error")
                    and client.get("/api/jobs/" + job_id).json())


def test_score_cost_matches_the_label_and_the_budget_names_the_stub(client):
    spec = make_spec(client, evals=10)
    created = client.post("/api/builder", json={"spec_id": spec["id"]}).json()
    builder_id = created["builder_id"]
    budget = created["budget"]
    assert budget["stub"] is True
    assert budget["compute_raised"] is True
    assert budget["spec_compute_min"] == spec["budget"]["compute_min"]
    program = good_program(client)
    size = client.post("/api/builder/" + builder_id + "/size", json={"program": program}).json()
    assert size["score_evals"] == 2
    job = run_score(client, builder_id, program)
    assert job["status"] == "done"
    assert job["budget"]["evals_used"] == size["score_evals"]
    # each added part in the estimate carries its glue, and the added total is their sum
    est = size["estimate"]
    assert abs(sum(p["params_b"] for p in est["parts"]) - est["added_params_b"]) < 1e-3
