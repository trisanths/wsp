"""The link from a studio spec to the compiler front door. CPU only, no network."""

import json
import time

import pytest
from fastapi.testclient import TestClient

from studio.server import frontdoor
from studio.server import specs as specs_mod
from studio.server.app import app
from wsp import auto

CODE_BENCH = [{"id": "humaneval", "kind": "target"}]

QWEN15 = {"model_type": "qwen2", "layers": 28, "hidden": 1536, "heads": 12, "kv_heads": 2, "head_dim": 128,
          "mlp_size": 8960, "vocab": 151936, "tied_embeddings": True, "experts": None, "top_k": None,
          "expert_size": None}
QWEN7 = dict(QWEN15, hidden=3584, heads=28, kv_heads=4, mlp_size=18944, vocab=152064, tied_embeddings=False)


def static_lineage():
    """Two Qwen2.5 models with the facts the plan needs, and no network."""
    models = {"Qwen/Qwen2.5-1.5B-Instruct": {"params": 1.544e9, "config": QWEN15, "tokenizer": "qwen"},
              "Qwen/Qwen2.5-Coder-7B-Instruct": {"params": 7.616e9, "config": QWEN7, "tokenizer": "qwen"}}
    kinds = {("Qwen/Qwen2.5-1.5B-Instruct", "Qwen/Qwen2.5-Coder-7B-Instruct"): "same_family_other_size"}
    return auto.StaticLineage(models, kinds)


def build_spec(**over):
    spec = {
        "id": "sp_fd0001", "title": "Python helper for laptops", "request_type": "build", "task_type": "knowledge",
        "host": "Qwen/Qwen2.5-1.5B-Instruct", "size": {"max_params_b": 2.47},
        "limits": {"max_total_params_b": 2.47, "max_run_params_b": 2.4, "max_heal_min": 60},
        "benchmarks": [
            {"id": "humaneval", "kind": "target"}, {"id": "mmlu", "kind": "target"},
            {"id": "if_guard", "kind": "guard"}, {"id": "mystery_bench", "kind": "target"},
        ],
        "ingredients": [{"label": "Code block L8-10", "donor": "Qwen/Qwen2.5-Coder-7B-Instruct", "kind": "block"}],
    }
    spec.update(over)
    return spec


def shrink_spec():
    return {
        "id": "sp_fd0002", "title": "A coding-only model", "request_type": "shrink", "task_type": "knowledge",
        "host": "Qwen/Qwen3-30B-A3B", "size": {"max_params_b": 17.0},
        "limits": {"max_total_params_b": 17.0, "max_run_params_b": 3.1, "max_heal_min": None},
        "benchmarks": [{"id": "humaneval", "kind": "target"}],
        "shrink": {"source_model": "Qwen/Qwen3-30B-A3B", "keep_skill": "code", "method": "experts"},
    }


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("WSP_STUDIO_OUT", str(tmp_path))
    monkeypatch.setenv("WSP_STUDIO_BACKEND", "stub")
    monkeypatch.setenv("WSP_STUDIO_LLM", "none")
    frontdoor._plans.clear()
    return TestClient(app)


def add_spec(monkeypatch, spec):
    monkeypatch.setitem(specs_mod._specs, spec["id"], spec)
    return spec["id"]


def no_hub(message):
    def fail():
        raise OSError(message)
    return fail


# Request shape.

def test_build_request_shape():
    request = frontdoor.request_from_spec(build_spec())
    assert request["name"] == "python_helper_for_laptops"
    assert request["base"] == "Qwen/Qwen2.5-1.5B-Instruct"
    assert request["remove"] == []
    assert request["keep"] == ["general"]
    assert request["seed"] == 0
    skills = {a["skill"]: a for a in request["add"]}
    assert skills["code"]["sources"] == ["Qwen/Qwen2.5-Coder-7B-Instruct"]
    assert skills["knowledge"]["sources"] == "auto"
    assert skills["code"]["components"] is None
    assert request["limits"] == {"max_total_params_b": 2.47, "max_active_params_b": 2.4, "max_heal_minutes": 60}
    assert request["reported"] == ["openai/openai_humaneval", "cais/mmlu", "google/IFEval"]
    assert frontdoor.check_request(request) == []


def test_shrink_request_shape():
    request = frontdoor.request_from_spec(shrink_spec())
    assert request["base"] == "Qwen/Qwen3-30B-A3B"
    assert request["add"] == []
    assert request["remove"] == [{"skill": "*"}]
    assert request["keep"] == ["code"]
    assert request["limits"] == {"max_total_params_b": 17.0, "max_active_params_b": 3.1, "max_heal_minutes": None}
    assert request["reported"] == ["openai/openai_humaneval"]
    assert frontdoor.check_request(request) == []


def test_request_is_the_format_of_the_front_door():
    clean = auto.validate_request(frontdoor.request_from_spec(build_spec()))
    assert clean["add"][0]["skill"] == "code"


def test_unmapped_benchmarks_are_listed_not_guessed():
    request, unmapped = frontdoor.request_and_unmapped(build_spec())
    assert unmapped == ["mystery_bench"]
    assert "mystery_bench" not in json.dumps(request)


def test_dataset_ids():
    assert frontdoor.dataset_for("gsm8k") == "openai/gsm8k"
    assert frontdoor.dataset_for("mmlupro:health") == "TIGER-Lab/MMLU-Pro"
    assert frontdoor.dataset_for("bbeh_mini") == "BBEH/bbeh"
    assert frontdoor.dataset_for("if_guard") == "google/IFEval"
    assert frontdoor.dataset_for("nope") is None


def test_validation_errors_are_plain_messages():
    spec = build_spec(benchmarks=[{"id": "if_guard", "kind": "guard"}], task_type="nothing")
    view = frontdoor.request_view(spec)
    assert view["request"]["add"] == []
    assert len(view["errors"]) == 1
    assert "add or remove" in view["errors"][0]


def test_request_route(client, monkeypatch):
    spec_id = add_spec(monkeypatch, build_spec())
    body = client.get("/api/spec/" + spec_id + "/request").json()
    assert set(body) == {"request", "unmapped", "errors"}
    assert body["unmapped"] == ["mystery_bench"]
    assert body["errors"] == []
    assert client.get("/api/spec/sp_missing/request").status_code == 404


def test_request_route_surfaces_validation_errors(client, monkeypatch):
    spec_id = add_spec(monkeypatch, build_spec(benchmarks=[], task_type="nothing"))
    body = client.get("/api/spec/" + spec_id + "/request").json()
    assert body["errors"] and "add or remove" in body["errors"][0]
    response = client.post("/api/spec/" + spec_id + "/plan")
    assert response.status_code == 400
    assert "error" in response.json()


# Plans.

def test_plan_route_success(client, monkeypatch):
    monkeypatch.setattr(frontdoor, "live_lineage", static_lineage)
    spec_id = add_spec(monkeypatch, build_spec(benchmarks=CODE_BENCH))
    response = client.post("/api/spec/" + spec_id + "/plan")
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["verified"] is False
    assert body["offline"] is False
    assert "draft" in body["note"]
    assert body["seconds"] >= 0
    assert body["plan"]["feasible"] is True
    assert body["plan"]["sources"][0]["chosen"] == "synth"
    assert body["plan"]["stages"][0]["method"] == "synth"


def test_plan_is_cached_per_spec_version(client, monkeypatch):
    monkeypatch.setattr(frontdoor, "live_lineage", static_lineage)
    calls = []
    real = frontdoor.run_plan

    def counting(request, lineage):
        calls.append(1)
        return real(request, lineage)

    monkeypatch.setattr(frontdoor, "run_plan", counting)
    spec = build_spec(benchmarks=CODE_BENCH)
    spec_id = add_spec(monkeypatch, spec)
    client.post("/api/spec/" + spec_id + "/plan")
    client.post("/api/spec/" + spec_id + "/plan")
    assert len(calls) == 1
    spec["limits"] = dict(spec["limits"], max_total_params_b=2.6)
    client.post("/api/spec/" + spec_id + "/plan")
    assert len(calls) == 2


def test_plan_route_with_an_exception(client, monkeypatch):
    monkeypatch.setattr(frontdoor, "live_lineage", static_lineage)

    def boom(request, lineage):
        raise ValueError("lineage broke")

    monkeypatch.setattr(frontdoor, "run_plan", boom)
    spec_id = add_spec(monkeypatch, build_spec(benchmarks=CODE_BENCH))
    response = client.post("/api/spec/" + spec_id + "/plan")
    assert response.status_code < 500
    assert "lineage broke" in response.json()["error"]


def test_plan_route_timeout(client, monkeypatch):
    def slow(request):
        time.sleep(1.0)

    monkeypatch.setattr(frontdoor, "plan_with_fallback", slow)
    monkeypatch.setattr(frontdoor, "PLAN_TIMEOUT_S", 0.1)
    spec_id = add_spec(monkeypatch, build_spec(benchmarks=CODE_BENCH))
    response = client.post("/api/spec/" + spec_id + "/plan")
    assert response.status_code == 408
    assert "longer than" in response.json()["error"]


def test_hub_failure_falls_back_to_offline_facts(client, monkeypatch):
    monkeypatch.setattr(frontdoor, "live_lineage", no_hub("no hub client"))
    spec_id = add_spec(monkeypatch, build_spec(benchmarks=CODE_BENCH))
    response = client.post("/api/spec/" + spec_id + "/plan")
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["offline"] is True
    assert "no hub client" in body["offline_reason"]
    assert body["plan"]["sources"][0]["chosen"] == "synth"


def test_offline_facts_miss_is_an_error_not_a_crash(client, monkeypatch):
    monkeypatch.setattr(frontdoor, "live_lineage", no_hub("network down"))
    spec = build_spec(host="someone/unknown-model", benchmarks=CODE_BENCH)
    response = client.post("/api/spec/" + add_spec(monkeypatch, spec) + "/plan")
    assert response.status_code == 422
    assert "someone/unknown-model" in response.json()["error"]


def test_shrink_plan_offline(client, monkeypatch):
    monkeypatch.setattr(frontdoor, "live_lineage", no_hub("network down"))
    response = client.post("/api/spec/" + add_spec(monkeypatch, shrink_spec()) + "/plan")
    assert response.status_code == 200, response.text
    plan = response.json()["plan"]
    assert plan["stages"][0]["method"] == "slim"
    assert plan["remove"] == [{"skill": "*"}]
