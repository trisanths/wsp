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
    assert result["heal"] == "none"
    assert result["source"] == "compiler session, team memory, 2026-11-14"
    assert "stub" not in result


def test_rows_hold_the_recalled_numbers(client):
    rows = {r["label"]: r for r in client.get("/api/measured").json()[0]["rows"]}
    assert rows["Full model"]["scores"] == {"humaneval": 91.5, "mbpp": 84.4, "gsm8k": 91.5, "mmlu": 77.8}
    assert rows["Full model"]["total_params_b"] == 30.5
    keep64 = rows["Keep 64 by code saliency"]
    assert keep64["scores"]["humaneval"] == 87.8 and keep64["scores"]["mmlu"] == 44.9
    assert keep64["total_params_b"] == 16.0
    assert rows["Keep 32 by code saliency"]["scores"] == {"humaneval": 57.9, "mbpp": 55.3}
    assert rows["Keep 16 by code saliency"]["scores"] == {"humaneval": 3.7, "mbpp": 7.8}
    assert rows["Keep 32 by routing frequency"]["scores"] == {"humaneval": 7.9, "mbpp": 11.7}


def test_controls_are_marked(client):
    rows = client.get("/api/measured").json()[0]["rows"]
    controls = sorted(r["rule"] for r in rows if r["is_control"])
    assert controls == ["random", "selective"]
    for row in rows:
        if row["is_control"]:
            assert row["scores"]["humaneval"] == 0.0


def test_caveats_are_plain_sentences(client):
    caveats = client.get("/api/measured").json()[0]["caveats"]
    assert len(caveats) >= 4
    text = " ".join(caveats)
    assert "One run per cell" in text and "3.0B" in text and "44.9" in text
    for char in [chr(8212), chr(8211)]:
        assert char not in text


def test_results_are_a_copy():
    first = measured.all_results()
    first[0]["rows"].pop()
    assert len(measured.all_results()[0]["rows"]) == 7


def test_mock_data_matches_the_server():
    with open(os.path.join(REPO, "studio", "web", "mock-extra.js")) as handle:
        text = handle.read()
    for result in measured.all_results():
        assert result["title"] in text
        for row in result["rows"]:
            assert row["label"] in text
            assert "humaneval: " + str(row["scores"]["humaneval"]) in text
