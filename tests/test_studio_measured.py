"""Measured results: the route shape and the numbers. CPU only, no network."""

import os

import pytest
from fastapi.testclient import TestClient

from studio.server import measured
from studio.server.app import app

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("WSP_STUDIO_OUT", str(tmp_path))
    monkeypatch.setenv("WSP_STUDIO_BACKEND", "stub")
    monkeypatch.setenv("WSP_STUDIO_LLM", "none")
    return TestClient(app)


def test_route_returns_the_list(client):
    response = client.get("/api/measured")
    assert response.status_code == 200
    results = response.json()
    assert len(results) == 1
    result = results[0]
    for key in ["id", "title", "kind", "model", "date", "scorer", "runs_per_cell", "heal", "source", "caveats", "rows"]:
        assert key in result
    assert result["kind"] == "skill_removal"
    assert result["scorer"] == "wsp.skills_eval"
    assert result["runs_per_cell"] == 1
    assert result["heal"] == "500 steps on two rows"
    assert result["source"] == "compiler session, results/README.md run 19, 2026-11-14"
    assert "stub" not in result


def test_rows_hold_the_recalled_numbers(client):
    rows = {r["label"]: r for r in client.get("/api/measured").json()[0]["rows"]}
    assert rows["Full model"]["scores"] == {"humaneval": 91.5, "mbpp": 84.4, "gsm8k": 91.5, "mmlu": 77.8}
    assert rows["Full model"]["total_params_b"] == 30.5
    keep64 = rows["Keep 64 by code saliency"]
    assert keep64["scores"]["humaneval"] == 87.8 and keep64["scores"]["mmlu"] == 44.9
    assert keep64["total_params_b"] == 16.0
    assert rows["Keep 32 by code saliency"]["scores"] == {"humaneval": 57.9, "mbpp": 55.3, "gsm8k": 2.5, "mmlu": 26.1}
    assert rows["Keep 16 by code saliency"]["scores"]["humaneval"] == 3.7
    assert rows["Keep 32 by routing frequency"]["scores"]["mbpp"] == 11.7


def test_heal_rows_follow_results_readme(client):
    rows = {r["label"]: r for r in client.get("/api/measured").json()[0]["rows"]}
    healed = rows["Keep 32 by code saliency, 500-step heal"]
    assert healed["heal_steps"] == 500 and not healed["is_control"]
    assert healed["scores"]["humaneval"] == 73.8 and healed["scores"]["mbpp"] == 61.9
    random_healed = rows["Keep 32 random, 500-step heal (seed 1)"]
    assert random_healed["is_control"] and random_healed["heal_steps"] == 500
    assert random_healed["scores"]["humaneval"] == 11.6 and random_healed["scores"]["mbpp"] == 23.3
    assert all(r["heal_steps"] == 0 for r in rows.values() if "heal" not in r["label"])


def test_controls_are_marked(client):
    rows = client.get("/api/measured").json()[0]["rows"]
    controls = sorted(r["rule"] for r in rows if r["is_control"])
    assert controls == ["random", "random", "selective"]
    for row in rows:
        if row["is_control"] and row["heal_steps"] == 0:
            assert row["scores"]["humaneval"] == 0.0


def test_caveats_are_plain_sentences(client):
    caveats = client.get("/api/measured").json()[0]["caveats"]
    assert len(caveats) >= 4
    text = " ".join(caveats)
    assert "one run" in text and "3.0B" in text and "44.9" in text
    assert "No heal has run yet" not in text and "500-step" in text
    for char in [chr(8212), chr(8211)]:
        assert char not in text


def test_results_are_a_copy():
    first = measured.all_results()
    first[0]["rows"].pop()
    assert len(measured.all_results()[0]["rows"]) == 9


def test_mock_data_matches_the_server():
    with open(os.path.join(REPO, "studio", "web", "mock-extra.js")) as handle:
        text = handle.read()
    for result in measured.all_results():
        assert result["title"] in text
        for row in result["rows"]:
            assert row["label"] in text
            assert "humaneval: " + str(row["scores"]["humaneval"]) in text
