"""Tests for the mining routes. CPU only, stub backend, no network."""

import json

import pytest
from fastapi.testclient import TestClient

from studio.server import library
from studio.server import mining
from studio.server.app import app


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("WSP_STUDIO_OUT", str(tmp_path))
    monkeypatch.setenv("WSP_STUDIO_BACKEND", "stub")
    monkeypatch.setenv("WSP_STUDIO_LLM", "none")
    monkeypatch.delenv("WSP_STUDIO_LIBRARY", raising=False)
    monkeypatch.delenv("WSP_STUDIO_ATLAS", raising=False)
    library.get(reload=True)
    mining._lookup_cache.clear()

    def no_network(model):
        raise OSError("offline in tests")

    monkeypatch.setattr(mining, "fetch_hf", no_network)
    return TestClient(app)


def first_model(client):
    models = client.get("/api/mine").json()["models"]
    assert len(models) > 0
    return models[0]


def test_mine_lists_models(client):
    body = client.get("/api/mine").json()
    assert body["requests"] == []
    top = body["models"][0]
    assert top["status"] == "mined"
    assert top["kind"] in ("dense", "moe")
    assert top["parts"]["total"] == sum(top["parts"]["by_kind"].values())
    assert isinstance(top["skills"], dict)


def test_lookup_falls_back_to_library(client):
    model = first_model(client)["model"]
    body = client.get("/api/mine/lookup", params={"model": model}).json()
    assert body["in_library"] is True
    assert body["mined_parts"] > 0


def test_lookup_unknown_offline(client):
    body = client.get("/api/mine/lookup", params={"model": "nobody/nothing-9b"}).json()
    assert body["found"] is False
    assert body["error"]


def test_lookup_bad_id(client):
    body = client.get("/api/mine/lookup", params={"model": "bad id"}).json()
    assert body["found"] is False


def test_lookup_uses_hub(client, monkeypatch):
    facts = {"params_b": 3.5, "architecture": "LlamaForCausalLM", "layers": 28, "experts": None,
             "license": "mit", "gated": False, "downloads": 10, "likes": 2}
    monkeypatch.setattr(mining, "fetch_hf", lambda model: facts)
    body = client.get("/api/mine/lookup", params={"model": "acme/tiny-3b"}).json()
    assert body["found"] is True
    assert body["params_b"] == 3.5
    assert body["license"] == "mit"
    assert body["in_library"] is False


def test_create_and_get_request(client, tmp_path):
    response = client.post("/api/mine", json={"model": "acme/tiny-3b", "skills": ["math", "math"],
                                              "effort": "smoke", "note": "try it"})
    assert response.status_code == 200
    item = response.json()
    assert item["status"] == "queued"
    assert item["skills"] == ["math"]
    assert [s["id"] for s in item["stages"]] == ["fetch", "read", "scan", "ablate", "gate", "publish"]
    assert all(s["status"] == "waiting" for s in item["stages"])
    assert item["estimate"]["gpu_min"] > 0 and item["estimate"]["is_estimate"] is True
    again = client.get("/api/mine/req/" + item["id"]).json()
    assert again == item
    assert client.get("/api/mine").json()["requests"][0]["id"] == item["id"]
    saved = json.load(open(tmp_path / "mine_requests.json"))
    assert saved["requests"][0]["id"] == item["id"]


def test_estimate_scales(client):
    small = client.post("/api/mine", json={"model": "acme/a-1b", "effort": "smoke"}).json()
    big = client.post("/api/mine", json={"model": "acme/b-70b", "effort": "full"}).json()
    assert big["estimate"]["gpu_min"] > small["estimate"]["gpu_min"]
    assert big["estimate"]["cost_usd"] > small["estimate"]["cost_usd"]


def test_duplicate_and_bad_input(client):
    assert client.post("/api/mine", json={"model": "acme/x-1b"}).status_code == 200
    dup = client.post("/api/mine", json={"model": "acme/x-1b"})
    assert dup.status_code == 400 and "queue" in dup.json()["error"]
    for body in [{"model": "nope"}, {"model": "a/b", "effort": "huge"}, {"model": "a/b", "skills": "math"},
                 {"model": "a/b", "skills": [""]}, {"model": "a/b", "note": "x" * 600}, []]:
        assert client.post("/api/mine", json=body).status_code == 400
    assert client.post("/api/mine", content=b"not json").status_code == 400


def test_unknown_request_404(client):
    assert client.get("/api/mine/req/mine_nope").status_code == 404


def test_model_detail(client):
    model = first_model(client)["model"]
    response = client.get("/api/models/" + model)
    assert response.status_code == 200
    body = response.json()
    assert body["model"] == model
    assert len(body["parts"]) > 0
    assert all(p["donor"] == model for p in body["parts"])
    assert len(body["layer_map"]) > 0
    assert isinstance(body["techniques"], list) and len(body["techniques"]) > 0
    assert isinstance(body["findings"], list) and isinstance(body["recipes"], list)


def test_model_detail_404(client):
    assert client.get("/api/models/nobody/nothing").status_code == 404


def test_part_detail(client):
    parts = client.get("/api/catalog").json()["parts"]
    with_raw = None
    for part in parts:
        if part["source"] == "library" and part["kind"] in ("mlp_cluster", "attn_heads"):
            with_raw = part
            break
    assert with_raw is not None
    response = client.get("/api/parts/" + with_raw["id"])
    assert response.status_code == 200
    body = response.json()
    assert body["id"] == with_raw["id"]
    assert set(["signature", "z", "z_heldout", "random_mean", "random_std", "random_n"]) <= set(body["raw"])
    assert isinstance(body["technique"], list)
    assert "recipes" in body and "selection" in body and "units_file" in body


def test_part_detail_with_slashes_and_404(client):
    parts = client.get("/api/catalog").json()["parts"]
    slashed = [p for p in parts if "/" in p["id"]]
    if slashed:
        assert client.get("/api/parts/" + slashed[0]["id"]).status_code == 200
    assert client.get("/api/parts/no_such_part").status_code == 404


def test_raw_block_z():
    raw = {"signature": {"skill": "s", "effect": {"s|search": 0.5, "s|heldout": 0.4, "s__control|heldout": 0.0},
                         "random_drops": [0.0, 0.1, -0.1, 0.0]}}
    block = mining.raw_block(raw, "s")
    assert block["random_n"] == 4
    assert block["z"] > 3
    assert block["signature"]["s"]["control_heldout"] == 0.0
