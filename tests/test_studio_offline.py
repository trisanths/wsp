"""The offline backend: real library parts and masks, scored by a simulator."""

import copy

import pytest
from wsp.swarm import program_schema as ps

from studio.server import catalog as catalog_mod
from studio.server import offline_backend

HOST = "unsloth/Llama-3.2-1B-Instruct"
MATCHING = "mlp_cluster_qwen25math15b_l16_gsm8k"


@pytest.fixture
def offline(tmp_path, monkeypatch):
    monkeypatch.setenv("WSP_STUDIO_OUT", str(tmp_path))
    monkeypatch.setenv("WSP_STUDIO_BACKEND", "stub")
    config, catalog = catalog_mod.load_catalog("stub")
    return config, catalog


def backend_for(config, catalog, seed):
    config = copy.deepcopy(config)
    config["stub_seed"] = seed
    return offline_backend.OfflineCompileEval(config, catalog)


def score(backend, program):
    compiled = backend.compile(program)
    assert compiled["ok"], compiled
    healed = backend.heal(compiled["handle"], program)
    return backend.evaluate(healed["handle"], "search", ["gsm8k"], "t", "full")["scores"]["gsm8k"]


def test_placeable_count_is_the_same_as_on_modal(tmp_path, monkeypatch):
    monkeypatch.setenv("WSP_STUDIO_OUT", str(tmp_path))
    stub = catalog_mod.build_palette("stub")
    modal = catalog_mod.build_palette("modal")
    assert stub["counts"]["placeable"] == modal["counts"]["placeable"]
    assert stub["counts"]["placeable"] > 1500
    assert not [part for part in stub["parts"] if part["source"] == "stub"]
    assert len(stub["masks"]) == len(modal["masks"])


def test_real_lock_reasons_stay_and_the_stub_lock_is_gone(tmp_path, monkeypatch):
    monkeypatch.setenv("WSP_STUDIO_OUT", str(tmp_path))
    palette = catalog_mod.build_palette("stub")
    locked = [part for part in palette["parts"] if not part["usable"]]
    assert locked
    reasons = " ".join(part["usable_reason"] for part in locked)
    assert "stub backend" not in reasons.lower()
    assert "tap" in reasons


def test_no_palette_text_calls_a_real_part_made_up(tmp_path, monkeypatch):
    monkeypatch.setenv("WSP_STUDIO_OUT", str(tmp_path))
    palette = catalog_mod.build_palette("stub")
    for part in palette["parts"] + palette["masks"]:
        text = " ".join(str(part.get(key) or "") for key in ["label", "summary", "evidence", "usable_reason"]).lower()
        for word in ["made-up", "made up", "stub part", "mock", "fake"]:
            assert word not in text, (part["id"], word)


def test_a_session_compiles_and_scores_a_real_part_and_a_real_mask(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from studio.server.app import app
    from tests.test_studio_server import make_spec, wait_for

    monkeypatch.setenv("WSP_STUDIO_OUT", str(tmp_path))
    monkeypatch.setenv("WSP_STUDIO_BACKEND", "stub")
    monkeypatch.setenv("WSP_STUDIO_LLM", "none")
    monkeypatch.setenv("WSP_STUDIO_PACE_S", "0")
    client = TestClient(app)
    spec = make_spec(client)
    builder_id = client.post("/api/builder", json={"spec_id": spec["id"]}).json()["builder_id"]
    palette = client.get("/api/catalog").json()
    part = [item for item in palette["parts"] if item["id"] == MATCHING][0]
    assert part["usable"]
    mask = palette["masks"][0]["id"]
    program = ps.make_program(HOST, [ps.make_graft(part["id"], part["default_at"], glue_rank=64)],
                              ps.make_heal(100, data="math"))
    program["masks"] = {"op": "use", "id": mask}
    for action in ["compile", "score"]:
        response = client.post("/api/builder/" + builder_id + "/action", json={"program": program, "action": action})
        assert response.status_code == 200, response.text
        job_id = response.json()["job_id"]
        job = wait_for(lambda: client.get("/api/jobs/" + job_id).json()["status"] in ("done", "error")
                       and client.get("/api/jobs/" + job_id).json())
        assert job["status"] == "done", job
        assert job["stub"] is True


def test_a_matching_part_beats_its_random_control_on_average(offline):
    config, catalog = offline
    part = catalog.records[MATCHING]
    at = 8
    real, control = [], []
    for seed in range(6):
        backend = backend_for(config, catalog, seed)
        heal = ps.make_heal(300, data="math")
        real.append(score(backend, ps.make_program(HOST, [ps.make_graft(MATCHING, at, glue_rank=64)], heal)))
        control.append(score(backend, ps.make_program(
            HOST, [ps.make_graft(MATCHING, at, glue_rank=64, control="random", control_seed=seed)], heal)))
    assert part["model"]
    assert sum(real) / len(real) > sum(control) / len(control)


def test_effects_follow_skills_and_status():
    base = {"id": "p", "model": "Qwen/Qwen2.5-Math-1.5B", "kind": "block", "home": 0.5, "edge": None,
            "status": "candidate"}
    matching = offline_backend.part_content(dict(base, skills=["math", "gsm8k"]), "gsm8k", 0)
    unrelated = offline_backend.part_content(dict(base, skills=["law"], model="x/Law-7B"), "gsm8k", 0)
    rejected = offline_backend.part_content(dict(base, skills=["math", "gsm8k"], status="rejected"), "gsm8k", 0)
    assert matching > 0.015
    assert abs(unrelated) < 0.02
    assert rejected < matching
    assert matching == offline_backend.part_content(dict(base, skills=["math", "gsm8k"]), "gsm8k", 0)


def test_every_catalog_part_and_mask_compiles(offline):
    config, catalog = offline
    backend = backend_for(config, catalog, 0)
    ids = [record["id"] for record in catalog.records.values() if record.get("type") != "deletion_mask"]
    assert len(ids) > 1500
    for part_id in ids[::40]:
        program = ps.make_program(HOST, [ps.make_graft(part_id, 5, glue_rank=64)], ps.make_heal(50, data="math"))
        result = backend.compile(program)
        assert result["ok"], (part_id, result)
        entry = result["graft_report"][0]
        assert 0.0 <= entry["fit_r2"] <= 0.99 and entry["part_params"] > 0


def test_studio_task_ids_match_skills():
    match = offline_backend.skill_match
    assert match("mmlupro:math", ["math"], "x") > 0
    assert match("mmlupro:law", ["law"], "x") > 0
    assert match("mmlupro:law", ["math"], "x") == 0 or match("mmlupro:law", ["math"], "x") < match("mmlupro:law", ["law"], "x")
    assert match("bbeh_mini", ["reasoning"], "x") > 0
    assert match("bbeh_mini", ["law"], "x") == 0


def test_every_usable_part_has_a_size(tmp_path, monkeypatch):
    monkeypatch.setenv("WSP_STUDIO_OUT", str(tmp_path))
    palette = catalog_mod.build_palette("stub")
    for part in palette["parts"]:
        if part["usable"]:
            assert part["params_m"] is not None
    unsized = [part for part in palette["parts"] if part["usable_reason"] == catalog_mod.UNSIZED_REASON]
    assert unsized


def test_offline_headline_says_estimate():
    from studio.server import compare

    assert compare.estimate_headline("Failed attempt. It is below Base.").startswith("Offline estimate")
    assert "Failed attempt" not in compare.estimate_headline("Failed attempt. It is below Base.")


def test_donor_profiles_fill_library_gaps(tmp_path, monkeypatch):
    from studio.server import model_profiles

    monkeypatch.setenv("WSP_STUDIO_OUT", str(tmp_path))
    extra = model_profiles.extra()
    assert "LiquidAI/LFM2-1.2B" in extra and extra["LiquidAI/LFM2-1.2B"]["layers"] == 16
    library_models = {"LiquidAI/LFM2-1.2B": {"params_b": 9.0, "layers": 3}}
    assert model_profiles.merged(library_models)["LiquidAI/LFM2-1.2B"]["params_b"] == 9.0
    palette = catalog_mod.build_palette("stub")
    assert palette["counts"]["placeable"] >= 2000
    sized = [item for item in palette["parts"] if item["donor"] == "LiquidAI/LFM2-1.2B"]
    assert sized and all(item["usable"] for item in sized if item["status"] != "rejected" and not item.get("usable_reason"))
