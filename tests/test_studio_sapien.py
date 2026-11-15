"""Tests for the Sapien studio: library palette, limits, no taps, and the honest comparison set. CPU only."""

import json

import pytest
from fastapi.testclient import TestClient

from studio.server import compare
from studio.server import library
from studio.server import runs
from studio.server import sizing
from studio.server.app import app
from tests.test_studio_server import HOST, good_program, make_spec, math_part, start_run, tap_program, wait_done, wait_for
from wsp.swarm import program_schema as ps


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("WSP_STUDIO_OUT", str(tmp_path))
    monkeypatch.setenv("WSP_STUDIO_BACKEND", "stub")
    monkeypatch.setenv("WSP_STUDIO_LLM", "none")
    monkeypatch.setenv("WSP_STUDIO_PACE_S", "0")
    return TestClient(app)


def write_small_library(root, atlas_root):
    """A tiny library and atlas, so tests do not depend on the real one."""
    comp = root / "components" / "demo"
    comp.mkdir(parents=True)
    rows = []

    def add(entry_id, kind, status, evidence, extra=None):
        data = {"id": entry_id, "kind": kind, "title": entry_id + " title", "status": status,
                "model": "Qwen/Qwen2.5-Math-1.5B", "layers": [8, 11], "evidence": evidence, "skills": ["math"]}
        data.update(extra or {})
        (comp / (entry_id + ".json")).write_text(json.dumps(data))
        rows.append({"id": entry_id, "kind": kind, "path": "components/demo/" + entry_id + ".json",
                     "status": status, "title": data["title"]})

    good = [{"claim": "graft vs random block", "metric": "GSM8K accuracy", "value": 0.41, "control_value": 0.30,
             "control": "random block", "source": "results/x.json"}]
    add("blk_good", "block", "validated", good)
    add("blk_bad", "block", "rejected", [], {"status_reason": "random block scored the same"})
    add("heads_new", "attn_heads", "candidate", [], {"signature": {"n_units": 2}})
    add("exp_one", "expert", "candidate", [], {"params_b": 0.05})
    models = root / "models"
    models.mkdir()
    big = {"id": "Big__model", "kind": "model", "hf_id": "Big/model", "params_b": 6.0, "active_params_b": 6.0,
           "layers": 30, "hidden": 4096, "vocab": 100000, "tokenizer_family": "qwen2",
           "scores": {"gsm8k_search_harness": {"value": 0.7, "split": "GSM8K search split",
                                               "source": "results/b.json"}}}
    (models / "Big__model.json").write_text(json.dumps(big))
    math = dict(big, id="Qwen__Qwen2.5-Math-1.5B", hf_id="Qwen/Qwen2.5-Math-1.5B", params_b=1.5,
                active_params_b=1.5, layers=28, hidden=1536, scores={})
    (models / "Qwen__Qwen2.5-Math-1.5B.json").write_text(json.dumps(math))
    rows.append({"id": "Qwen__Qwen2.5-Math-1.5B", "kind": "model", "path": "models/Qwen__Qwen2.5-Math-1.5B.json",
                 "status": "validated"})
    rows.append({"id": "Big__model", "kind": "model", "path": "models/Big__model.json", "status": "validated"})
    (root / "index.jsonl").write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    atlas_root.mkdir(parents=True)
    entry = {"model": "Qwen/Qwen2.5-7B-Instruct",
             "layer_maps": {"math": {"blocks": [[12, 19]], "block_mass": [0.64], "heldout": {"check_mass": 0.5}}},
             "expert_maps": {"math": {"12": [3, 17]}}}
    (atlas_root / "Qwen__Qwen2.5-7B-Instruct.json").write_text(json.dumps(entry))


@pytest.fixture
def small_library(tmp_path, monkeypatch):
    root = tmp_path / "lib"
    atlas = tmp_path / "atlas" / "entries"
    write_small_library(root, atlas)
    monkeypatch.setenv("WSP_STUDIO_LIBRARY", str(root))
    monkeypatch.setenv("WSP_STUDIO_ATLAS", str(atlas))
    library.get(reload=True)
    yield
    monkeypatch.delenv("WSP_STUDIO_LIBRARY")
    monkeypatch.delenv("WSP_STUDIO_ATLAS")
    library.get(reload=True)


# Library and palette.

def test_palette_lists_library_and_atlas_parts_with_status(client, small_library):
    palette = client.get("/api/catalog").json()
    by_id = {}
    for part in palette["parts"]:
        by_id[part["id"]] = part
    good = by_id["blk_good"]
    assert good["kind"] == "block"
    assert good["status"] == "validated"
    assert "0.41 against 0.3 for random block" in good["evidence"]
    assert good["usable"] is True
    assert by_id["blk_bad"]["status"] == "rejected"
    assert "random block scored the same" in by_id["blk_bad"]["evidence"]
    assert by_id["heads_new"]["kind"] == "attn_heads"
    assert by_id["exp_one"]["kind"] == "expert"
    inserted = by_id["block/Qwen/Qwen2.5-7B-Instruct/12-19"]
    assert inserted["kind"] == "inserted_layer"
    assert "0.64" in inserted["evidence"]
    expert = by_id["expert/Qwen/Qwen2.5-7B-Instruct/12"]
    assert expert["kind"] == "expert"
    kinds = {}
    for item in palette["kinds"]:
        kinds[item["kind"]] = item["count"]
    assert kinds["inserted_layer"] == 1
    assert palette["library"]["atlas"] is True
    assert palette["library"]["counts"] == {"candidate": 4, "validated": 1, "rejected": 1}


def test_palette_has_no_atlas_parts_when_the_folder_is_missing(client, tmp_path, monkeypatch):
    monkeypatch.setenv("WSP_STUDIO_ATLAS", str(tmp_path / "nothing_here"))
    library.get(reload=True)
    palette = client.get("/api/catalog").json()
    assert palette["library"]["atlas"] is False
    for part in palette["parts"]:
        assert part["kind"] != "inserted_layer"
    monkeypatch.delenv("WSP_STUDIO_ATLAS")
    library.get(reload=True)


def test_real_library_entries_reach_the_palette(client):
    palette = client.get("/api/catalog").json()
    statuses = set()
    for part in palette["parts"]:
        if part["source"] == "library":
            statuses.add(part["status"])
            assert part["evidence"] != ""
    assert "rejected" in statuses
    assert "candidate" in statuses
    assert palette["library"]["counts"]["rejected"] > 0


def test_tap_read_points_are_shown_but_locked(client):
    palette = client.get("/api/catalog").json()
    found = [part for part in palette["parts"] if part["id"].startswith("tapread_")]
    assert len(found) > 0
    for part in found:
        assert part["usable"] is False
        assert "second model" in part["usable_reason"]
        assert part["evidence"] != ""


def test_palette_has_no_taps(client):
    palette = client.get("/api/catalog").json()
    for part in palette["parts"]:
        assert part["kind"] != "tap"
    assert "tap_train_default" not in palette
    spec = make_spec(client)
    for item in spec["ingredients"]:
        assert item["kind"] != "tap"


# Taps are refused everywhere.

def test_a_program_with_taps_is_refused(client):
    spec = make_spec(client)
    builder_id = client.post("/api/builder", json={"spec_id": spec["id"]}).json()["builder_id"]
    response = client.post("/api/builder/" + builder_id + "/action", json={"program": tap_program(),
                                                                           "action": "score"})
    assert response.status_code == 400
    assert "second model" in response.json()["error"]
    response = client.post("/api/builder/" + builder_id + "/size", json={"program": tap_program()})
    assert response.json()["ok"] is False
    response = client.post("/api/runs", json={"spec_id": spec["id"], "program": tap_program()})
    assert response.status_code == 400
    assert "second model" in response.json()["error"]


def test_agents_cannot_score_a_tap_program(client):
    spec = make_spec(client)
    run_id = start_run(client, spec)
    wait_done(client, run_id)
    run = runs.get_run(run_id)
    assert run.controller.admit("planner-1", tap_program()) is False
    proposal = {"program": tap_program(), "hypothesis": "x", "citations": [], "probe_first": False, "parent": None}
    assert run.swarm.screen("planner-1", proposal, set()) is None
    assert run.state.refused >= 1


# Limits.

def test_spec_limits_come_from_the_request(client):
    prompt = "a tutor under 3B parameters that runs 1.2B per token and heals for 30 minutes"
    spec = client.post("/api/spec", data={"prompt": prompt}).json()
    assert spec["limits"] == {"max_total_params_b": 3.0, "max_run_params_b": 1.24, "max_heal_min": 30}
    assert spec["size"]["max_params_b"] == 3.0
    other = client.post("/api/spec", data={"prompt": "a math model, heal one hour"}).json()
    assert other["limits"]["max_total_params_b"] == 1.9


def test_spec_limits_can_be_edited(client):
    spec = make_spec(client)
    spec["limits"] = {"max_total_params_b": 2.5, "max_run_params_b": 9, "max_heal_min": 0}
    edited = client.put("/api/spec/" + spec["id"], json=spec).json()
    assert edited["limits"] == {"max_total_params_b": 2.5, "max_run_params_b": 2.5, "max_heal_min": 1}
    assert edited["size"]["max_params_b"] == 2.5
    edited["comparators"] = {"distill": {"gsm8k": 0.4, "bad": "x"}}
    again = client.put("/api/spec/" + spec["id"], json=edited).json()
    assert again["comparators"] == {"distill": {"gsm8k": 0.4}}


def small_limit_spec(client, total, run=None, heal=None):
    spec = make_spec(client)
    spec["limits"] = {"max_total_params_b": total, "max_run_params_b": run or total, "max_heal_min": heal or 20}
    return client.put("/api/spec/" + spec["id"], json=spec).json()


def test_builder_refuses_a_build_over_the_total_limit(client):
    spec = small_limit_spec(client, 1.25)
    builder_id = client.post("/api/builder", json={"spec_id": spec["id"]}).json()["builder_id"]
    program = good_program(client)
    report = client.post("/api/builder/" + builder_id + "/size", json={"program": program}).json()
    assert report["ok"] is False
    assert report["estimate"]["total_params_b"] > 1.25
    assert "Over the total size limit" in report["problems"][0]
    response = client.post("/api/builder/" + builder_id + "/action", json={"program": program, "action": "score"})
    assert response.status_code == 400
    message = response.json()["error"]
    assert "Over the total size limit" in message
    assert "The limit is 1.25B" in message
    response = client.post("/api/builder/" + builder_id + "/handoff", json={"program": program})
    assert response.status_code == 400
    assert "Over the total size limit" in response.json()["error"]
    response = client.post("/api/runs", json={"spec_id": spec["id"], "program": program})
    assert response.status_code == 400


def test_builder_allows_a_grown_base_when_the_limits_allow(client):
    spec = small_limit_spec(client, 4.0)
    builder_id = client.post("/api/builder", json={"spec_id": spec["id"]}).json()["builder_id"]
    program = good_program(client)
    report = client.post("/api/builder/" + builder_id + "/size", json={"program": program}).json()
    assert report["ok"] is True
    assert report["estimate"]["total_params_b"] > report["estimate"]["host_params_b"]
    response = client.post("/api/builder/" + builder_id + "/action", json={"program": program, "action": "compile"})
    assert response.status_code == 200


def test_per_token_and_heal_limits(client):
    spec = small_limit_spec(client, 4.0, run=1.25, heal=20)
    builder_id = client.post("/api/builder", json={"spec_id": spec["id"]}).json()["builder_id"]
    report = client.post("/api/builder/" + builder_id + "/size", json={"program": good_program(client)}).json()
    assert report["ok"] is False
    assert "per-token limit" in report["problems"][0]
    spec = small_limit_spec(client, 4.0, run=4.0, heal=1)
    builder_id = client.post("/api/builder", json={"spec_id": spec["id"]}).json()["builder_id"]
    part = math_part(client)
    long_heal = ps.make_program(HOST, [ps.make_graft(part["id"], part["default_at"], glue_rank=64)],
                                ps.make_heal(3000, data="math"))
    report = client.post("/api/builder/" + builder_id + "/size", json={"program": long_heal}).json()
    assert report["ok"] is False
    assert "heal time limit" in report["problems"][0]
    assert report["estimate"]["heal_min"] > 1


def test_run_never_scores_a_program_over_the_limit(client):
    # The run seed comes from the clock. At 1.3B some seeds propose only programs that fit, so the
    # limit sits just above the host (1.24B) and nearly every graft is refused.
    spec = small_limit_spec(client, 1.25)
    spec["budget"]["evals"] = 6
    spec = client.put("/api/spec/" + spec["id"], json=spec).json()
    run_id = start_run(client, spec)
    snap = wait_done(client, run_id)
    assert snap["status"] == "done"
    assert snap["limits"]["max_total_params_b"] == 1.25
    assert snap["refused"] > 0
    refused = [item for item in snap["activity"] if item["kind"] == "refused"]
    assert len(refused) > 0
    assert "Over the total size limit" in refused[0]["text"]
    for entry in snap["leaderboard"]:
        assert entry["params_b"] <= 1.25 + 1e-6


def test_estimate_matches_the_stub_compile(client):
    spec = small_limit_spec(client, 4.0)
    builder_id = client.post("/api/builder", json={"spec_id": spec["id"]}).json()["builder_id"]
    program = good_program(client)
    report = client.post("/api/builder/" + builder_id + "/size", json={"program": program}).json()
    job_id = client.post("/api/builder/" + builder_id + "/action", json={"program": program,
                                                                         "action": "compile"}).json()["job_id"]
    job = wait_for(lambda: client.get("/api/jobs/" + job_id).json()["status"] == "done"
                   and client.get("/api/jobs/" + job_id).json())
    assert abs(job["result"]["params_b"] - report["estimate"]["total_params_b"]) < 0.01
    assert job["result"]["size"]["total_params_b"] == report["estimate"]["total_params_b"]


# The comparison set.

def make_spec_dict(total=1.5, lora=0.5, base=0.25):
    return {"benchmarks": [{"id": "gsm8k", "kind": "target", "base": base, "lora": lora, "target": 0.6},
                           {"id": "if_guard", "kind": "guard", "floor": 0.4, "base": 0.5}],
            "size": {"host_params_b": 1.24}, "limits": {"max_total_params_b": total}, "comparators": {"distill": {}}}


def row_ids(comparison):
    return [row["id"] for row in comparison["rows"]]


def test_comparison_fails_a_result_that_does_not_beat_lora():
    spec = make_spec_dict()
    result = compare.build(spec, {"gsm8k": 0.40}, 1.3, control={"gsm8k": 0.2})
    assert result["verdict"] == "failed"
    assert "lora" in result["not_beaten"]
    assert result["headline"].startswith("Failed attempt.")
    assert row_ids(result) == ["base", "lora", "distill", "random"]
    assert "distill" in result["unmeasured"]


def test_comparison_needs_every_reference_measured_to_succeed():
    spec = make_spec_dict()
    spec["comparators"] = {"distill": {"gsm8k": 0.45}}
    result = compare.build(spec, {"gsm8k": 0.60}, 1.3, control={"gsm8k": 0.3})
    assert result["verdict"] == "beats"
    assert result["not_beaten"] == []
    unproven = compare.build(spec, {"gsm8k": 0.60}, 1.3, control=None)
    assert unproven["verdict"] == "unproven"
    assert unproven["unmeasured"] == ["random"]
    close = compare.build(spec, {"gsm8k": 0.51}, 1.3, control={"gsm8k": 0.3})
    assert close["verdict"] == "failed"
    assert "lora" in close["not_beaten"]


def test_comparison_shows_the_best_model_only_when_the_limit_is_far_above_the_base(small_library, monkeypatch):
    monkeypatch.setitem(compare.SEARCH_KEYS, "gsm8k", ["gsm8k_search_harness", "gsm8k_search_raw"])
    spec = make_spec_dict(total=1.5)
    assert "best_model" not in row_ids(compare.build(spec, {"gsm8k": 0.6}, 1.3))
    spec = make_spec_dict(total=8.0)
    result = compare.build(spec, {"gsm8k": 0.6}, 7.0)
    assert row_ids(result)[-1] == "best_model"
    row = result["rows"][-1]
    assert row["value"] == 0.7
    assert "Big/model" in row["note"]
    assert row["beaten"] is False
    assert "best_model" in result["not_beaten"]
    assert result["verdict"] == "failed"
    tight = make_spec_dict(total=4.0)
    row = compare.build(tight, {"gsm8k": 0.6}, 3.0)["rows"][-1]
    assert row["measured"] is False


def test_stub_comparison_is_labeled_stub(client):
    spec = make_spec(client)
    builder_id = client.post("/api/builder", json={"spec_id": spec["id"]}).json()["builder_id"]
    program = good_program(client)
    job_id = client.post("/api/builder/" + builder_id + "/action", json={"program": program,
                                                                         "action": "score"}).json()["job_id"]
    job = wait_for(lambda: client.get("/api/jobs/" + job_id).json()["status"] in ("done", "error")
                   and client.get("/api/jobs/" + job_id).json())
    comparison = job["result"]["comparison"]
    assert comparison["stub"] is True
    by_id = {}
    for row in comparison["rows"]:
        by_id[row["id"]] = row
    assert by_id["base"]["stub"] is True
    assert by_id["distill"]["stub"] is True
    assert by_id["distill"]["note"] == "offline estimate, not a measurement"
    assert by_id["lora"]["stub"] is False
    assert by_id["random"]["measured"] is True
    assert comparison["verdict"] in ("failed", "beats")
    if comparison["verdict"] == "failed":
        assert comparison["headline"].startswith("Offline estimate")


def test_run_snapshot_carries_the_comparison(client):
    spec = make_spec(client)
    snap = wait_done(client, start_run(client, spec))
    assert snap["comparison"] is not None
    assert snap["comparison"]["stub"] is True
    assert snap["comparison"]["verdict"] in ("failed", "unproven", "beats")
    for entry in snap["leaderboard"]:
        assert entry["comparison"]["verdict"] in ("failed", "unproven", "beats", "none")
    assert snap["comparison"]["rows"][0]["id"] == "base"


def test_a_result_below_the_base_is_a_failed_attempt():
    spec = make_spec_dict()
    result = compare.build(spec, {"gsm8k": 0.1}, 1.3, control={"gsm8k": 0.05})
    assert result["verdict"] == "failed"
    assert "base" in result["not_beaten"]


def test_limit_messages():
    est = {"total_params_b": 2.0, "run_params_b": 2.0, "heal_min": 30.0, "host_params_b": 1.24,
           "added_params_b": 0.76}
    problems = sizing.limit_problems({"max_total_params_b": 1.5, "max_run_params_b": 1.5, "max_heal_min": 20}, est)
    assert len(problems) == 3
    assert sizing.limit_problems({"max_total_params_b": 2.0, "max_run_params_b": 2.0, "max_heal_min": 30}, est) == []
    assert sizing.limit_problems(None, est) == []


def test_library_reads_once_and_reloads_on_request(tmp_path, monkeypatch):
    root = tmp_path / "lib"
    write_small_library(root, tmp_path / "atlas")
    monkeypatch.setenv("WSP_STUDIO_LIBRARY", str(root))
    monkeypatch.setenv("WSP_STUDIO_ATLAS", str(tmp_path / "missing"))
    first = library.get(reload=True)
    assert library.get() is first
    (root / "index.jsonl").write_text("")
    assert library.get() is first
    assert len(library.get(reload=True)["parts"]) == 0
    monkeypatch.delenv("WSP_STUDIO_LIBRARY")
    monkeypatch.delenv("WSP_STUDIO_ATLAS")
    library.get(reload=True)


# Regression tests from the verification pass.

def test_best_model_is_the_highest_score_over_every_score_key(small_library, monkeypatch):
    monkeypatch.setitem(compare.SEARCH_KEYS, "gsm8k", ["gsm8k_search_harness", "gsm8k_search_raw"])
    models = library.get()["models"]
    models["Small/harness"] = {"hf_id": "Small/harness", "params_b": 1.5, "hidden": 1024, "layers": 16,
                               "scores": {"gsm8k_search_harness": {"value": 0.45, "split": "s"}}}
    models["Mid/raw"] = {"hf_id": "Mid/raw", "params_b": 7.0, "hidden": 4096, "layers": 28,
                         "scores": {"gsm8k_search_raw": {"value": 0.765, "split": "s"}}}
    found = compare.best_model_for("gsm8k", 8.0, models)
    assert found[0] == 0.765
    assert found[1] == "Mid/raw"
    assert compare.best_model_for("gsm8k", 2.0, models)[1] == "Small/harness"


def comparison_state(limit_total):
    from studio.server.runstate import RunState
    spec = make_spec_dict(total=limit_total)
    config = {"host": HOST, "eval_budget": 6, "compute_budget_s": 100}
    return RunState("r1", spec, config, "stub", 1.24, 1, ["gsm8k"])


def test_a_measured_size_over_the_limit_is_a_failed_attempt_in_a_run():
    state = comparison_state(1.3)
    entry = {"program_hash": "h", "scores": {"gsm8k": 0.9}, "params_b": 1.4, "status": "ok"}
    state.controls["h"] = {"gsm8k": 0.1}
    state.heldout["h"] = {"gsm8k": 0.9}
    over = state.comparison_locked(entry)
    assert over["verdict"] == "failed"
    assert "Over the total size limit" in over["headline"]
    fits = state.comparison_locked(dict(entry, params_b=1.3))
    assert fits["verdict"] != "failed" or "Over the total size limit" not in fits["headline"]


def test_a_limit_check_that_cannot_estimate_refuses_the_program(client, monkeypatch):
    spec = make_spec(client)
    run_id = start_run(client, spec)
    wait_done(client, run_id)
    run = runs.get_run(run_id)
    program = good_program(client)

    def broken(program, ctx):
        raise KeyError("no profile")

    monkeypatch.setattr(sizing, "estimate", broken)
    assert run.controller.admit("planner-1", program) is False
    assert run.controller.admit("planner-1", tap_program()) is False


def test_heal_and_train_check_the_default_heal_against_the_heal_limit(client):
    spec = small_limit_spec(client, 4.0, run=4.0, heal=1)
    builder_id = client.post("/api/builder", json={"spec_id": spec["id"]}).json()["builder_id"]
    part = math_part(client)
    no_heal = ps.make_program(HOST, [ps.make_graft(part["id"], part["default_at"], glue_rank=64)])
    path = "/api/builder/" + builder_id + "/action"
    assert client.post(path, json={"program": no_heal, "action": "score"}).status_code == 200
    for action in ("heal", "train"):
        response = client.post(path, json={"program": no_heal, "action": action})
        assert response.status_code == 400, action
        assert "heal time limit" in response.json()["error"]


def test_carve_splice_and_unknown_donors_cannot_pass_a_limit():
    est = {"total_params_b": 1.24, "run_params_b": 1.24, "heal_min": 1.0, "host_params_b": 1.24,
           "added_params_b": 0.0, "unsized_steps": ["a splice step"]}
    problems = sizing.limit_problems({"max_total_params_b": 2.0, "max_run_params_b": 2.0, "max_heal_min": 20}, est)
    assert len(problems) == 1
    assert "a splice step" in problems[0]


def test_inserted_layers_and_experts_count_toward_the_size(client):
    spec = make_spec(client)
    builder_id = client.post("/api/builder", json={"spec_id": spec["id"]}).json()["builder_id"]
    from studio.server import builder as builder_mod
    session = builder_mod.get_session(builder_id)
    base = ps.make_program(HOST)
    plain = sizing.estimate(base, session.size_ctx)
    with_insert = dict(base)
    with_insert["inserts"] = [{"donor": "Qwen/Qwen2.5-7B-Instruct", "layers": [10, 13], "at": 6, "stitch": "linear",
                               "stitch_rank": None, "stitch_hidden": None, "part_lora_rank": None,
                               "gate_init": 0.0, "control": None, "control_seed": 0}]
    inserted = sizing.estimate(with_insert, session.size_ctx)
    assert inserted["total_params_b"] > plain["total_params_b"] + 0.3
    assert inserted["unsized_steps"] == []
    with_experts = dict(base)
    with_experts["experts"] = {"donors": ["Qwen/Qwen2.5-Math-1.5B"], "top_k": 1, "stitch": "linear"}
    moe = sizing.estimate(with_experts, session.size_ctx)
    assert moe["total_params_b"] > plain["total_params_b"] + 0.3
    assert moe["run_params_b"] == plain["run_params_b"]
    unknown = dict(base)
    unknown["inserts"] = [dict(with_insert["inserts"][0], donor="No/such-model")]
    assert sizing.estimate(unknown, session.size_ctx)["unsized_steps"] != []


def test_atlas_blocks_are_listed_once_and_say_the_mass_is_optimistic(tmp_path, monkeypatch):
    atlas = tmp_path / "atlas"
    atlas.mkdir()
    layer_map = {"blocks": [[12, 19], [3, 4]], "block_mass": [0.64, 0.2], "heldout": {"check_mass": 0.5}}
    entry = {"model": "Qwen/Qwen2.5-7B-Instruct", "layer_maps": {"math": layer_map, "code": layer_map}}
    (atlas / "one.json").write_text(json.dumps(entry))
    found = library.atlas_block_entries(str(atlas))
    ids = [item["id"] for item in found]
    assert len(ids) == len(set(ids)) == 2
    first = found[0]
    assert first["skills"] == ["code", "math"]
    assert "optimistic" in first["evidence"]
    assert "also chosen by the math map" in first["evidence"]
    assert "held 0.5 of the mass on the other half" in found[1]["evidence"]


def test_a_per_token_size_is_not_read_as_the_total_limit():
    from studio.server import specs as specs_mod
    assert specs_mod.size_from_prompt("up to 2B per token, laptop", 1.24) != 2.0
    assert specs_mod.run_limit_from_prompt("up to 2B per token", 4.0, 1.24) == 2.0
    assert specs_mod.size_from_prompt("a model under 3B", 1.24) == 3.0
    assert specs_mod.run_limit_from_prompt("prune it to 3B", 4.0, 1.24) == 4.0


def test_programs_with_taps_have_no_readable_tap_name():
    from studio.server import programs
    assert "tap" not in programs.readable(tap_program())


def test_gsm8k_library_scores_are_not_used_until_rescored(small_library):
    models = library.get()["models"]
    assert compare.SEARCH_KEYS["gsm8k"] == []
    assert compare.best_model_for("gsm8k", 8.0, models) is None
    spec = make_spec_dict(total=8.0)
    row = compare.build(spec, {"gsm8k": 0.6}, 7.0)["rows"][-1]
    assert row["measured"] is False
    assert "awaiting re-scoring" in row["note"]
