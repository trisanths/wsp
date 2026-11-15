"""The synthesis pipeline in the studio: component mapping, result shapes, stub verdicts, refusals, the compare
rule and the heal warning. CPU only. A fake Modal function stands in for every GPU call, so nothing uses the network."""

import json
import os

import pytest
from fastapi.testclient import TestClient

from studio.server import compare
from studio.server import modal_backend
from studio.server import settings
from studio.server import synth_report
from studio.server.app import app
from tests.test_studio_modal import modal_studio
from tests.test_studio_server import (FakeModalFunction, MODAL_PART, fake_synth_result, make_spec, modal_program,
                                      run_builder_job, start_run, wait_done)
from wsp.swarm import program_schema as ps

HOST = "unsloth/Llama-3.2-1B-Instruct"


@pytest.fixture
def stub_client(tmp_path, monkeypatch):
    monkeypatch.setenv("WSP_STUDIO_OUT", str(tmp_path))
    monkeypatch.setenv("WSP_STUDIO_BACKEND", "stub")
    monkeypatch.setenv("WSP_STUDIO_LLM", "none")
    monkeypatch.setenv("WSP_STUDIO_PACE_S", "0")
    monkeypatch.delenv("WSP_STUDIO_PIPELINE", raising=False)
    return TestClient(app)


@pytest.fixture
def modal_client(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setenv("WSP_STUDIO_OUT", str(tmp_path))
    monkeypatch.setenv("WSP_STUDIO_BACKEND", "modal")
    monkeypatch.setenv("WSP_STUDIO_LLM", "none")
    monkeypatch.setenv("WSP_STUDIO_PACE_S", "0")
    monkeypatch.delenv("WSP_STUDIO_PIPELINE", raising=False)
    monkeypatch.setattr(modal_backend, "lookup", lambda name: FakeModalFunction(name, calls))
    client = TestClient(app)
    client.modal_calls = calls
    return client


def new_builder(client, roomy=False):
    spec = make_spec(client)
    if roomy:
        limits = dict(spec["limits"], max_total_params_b=6.0, max_run_params_b=6.0)
        response = client.put("/api/spec/" + spec["id"], json={"limits": limits})
        assert response.status_code == 200, response.text
    return client.post("/api/builder", json={"spec_id": spec["id"]}).json()["builder_id"]


# Component mapping, with no network.

RECORDS = {
    "mlp1": {"id": "mlp1", "type": "mlp_cluster", "model": "Qwen/Qwen2.5-Math-1.5B", "layer": 16,
             "units": "parts_extracted/x.pt", "indices": [3, 4, 5]},
    "heads1": {"id": "heads1", "type": "attn_heads", "model": "Qwen/Qwen2.5-Math-1.5B", "layer": 12,
               "units": "parts_extracted/y.pt", "indices": [1, 2]},
    "block1": {"id": "block1", "type": "block", "model": "Qwen/Qwen2.5-Math-1.5B", "layers": [8, 17]},
    "mask1": {"id": "mask1", "type": "deletion_mask", "base": HOST},
}


def test_components_map_each_graft_and_insert():
    program = {"host": HOST, "seed": 0, "grafts": [
        {"id": "mlp1", "at": 9, "control": None, "control_seed": 0},
        {"id": "heads1", "at": None, "control": "random", "control_seed": 4},
        {"id": "block1", "at": 7, "control": None, "control_seed": 0}],
        "inserts": [{"donor": "Qwen/Qwen2.5-Coder-1.5B", "layers": [14, 23], "at": 6, "control": None,
                     "control_seed": 0}]}
    assert modal_studio.refusals(program, RECORDS) == []
    comps = modal_studio.synth_components(program, RECORDS, ["text"], "math")
    assert [c["kind"] for c in comps] == ["mlp_cluster", "attn_heads", "block", "block"]
    assert comps[0]["indices"] == [3, 4, 5] and comps[0]["layer"] == 16 and comps[0]["host_layer"] == 9
    assert comps[1]["indices"] == [1, 2] and comps[1]["control"] == "random"
    assert comps[1]["control_seed"] == 4 and comps[1]["host_layer"] is None
    assert comps[2]["layers"] == [8, 17]
    assert comps[3]["layers"] == [14, 23] and comps[3]["host_layer"] == 5
    assert comps[0]["skill"] == "gsm8k" and comps[0]["texts"] == ["text"]
    from wsp import synth

    for comp in comps:
        clean = synth.normalize_component({k: v for k, v in comp.items() if k != "label_id"})
        assert clean["kind"] == comp["kind"]


def test_duplicate_parts_get_distinct_ids():
    program = {"host": HOST, "grafts": [{"id": "mlp1", "at": 5}, {"id": "mlp1", "at": 9}], "inserts": []}
    comps = modal_studio.synth_components(program, RECORDS, [], "general")
    assert [c["id"] for c in comps] == ["mlp1", "mlp1#2"]
    assert comps[1]["label_id"] == "mlp1"


def test_unsupported_kinds_are_refused_with_plain_messages():
    program = {"host": HOST, "grafts": [{"id": "mask1", "at": 3}], "masks": {"op": "use", "id": "mask1"},
               "task_vectors": [{"id": "tv"}], "experts": {"donors": []}, "carve": {"keep": 0.5}}
    messages = modal_studio.refusals(program, RECORDS)
    text = " ".join(messages)
    assert "deletion mask" in text and "task vector" in text and "routed experts" in text and "carve" in text
    assert "deletion_mask" in text
    assert "block, attn_heads, mlp_cluster, expert" in text
    empty = modal_studio.refusals({"host": HOST, "grafts": []}, RECORDS)
    assert "at least one part" in empty[0]
    missing = modal_studio.refusals({"host": HOST, "grafts": [{"id": "nope", "at": 1}]}, RECORDS)
    assert "nope" in missing[0]


def test_heal_steps_split_and_the_heal_limit():
    program = {"heal": {"steps": 300}}
    assert modal_studio.split_heal_steps(program, {}, 2, True)[:2] == (150, 150)
    assert modal_studio.split_heal_steps({"heal": None}, {}, 1, False)[:2] == (150, 150)
    first, second, note = modal_studio.split_heal_steps(program, {"max_heal_min": 5}, 2, True)
    assert first + second == 150 and "heal limit" in note
    first, second, note = modal_studio.split_heal_steps({"heal": {"steps": 40}}, {"max_heal_min": 60}, 1, False)
    assert (first, second, note) == (20, 20, None)


def test_budget_carries_the_size_limits_and_lora_rank():
    program = {"heal": {"steps": 100}, "seed": 3, "lora": {"rank": 8}}
    options = {"max_total_params_b": 1.9, "max_run_params_b": 1.7}
    budget, note = modal_studio.synth_budget(program, options, 1, False)
    assert budget["max_params"] == 1900000000 and budget["max_params_per_token"] == 1700000000
    assert budget["stage1_steps"] == 50 and budget["stage2_steps"] == 50
    assert budget["seed"] == 3 and budget["lora_rank"] == 8
    assert note is None


def test_pool_comes_from_the_heal_data_or_the_spec():
    assert modal_studio.pool_of({"heal": {"data": "code"}}, {"pool": "math"}) == "code"
    assert modal_studio.pool_of({"heal": None}, {"pool": "math"}) == "math"
    assert modal_studio.pool_of({"heal": {"data": "mix:medicine,code"}}, {}) == "medicine"
    assert modal_studio.pool_of({"heal": {"data": "gsm8k_train"}}, {"pool": "gsm8k_train"}) == "general"


# A fake report in the shape wsp.synth writes.

FAKE_REPORT = {
    "seconds": 300.0, "heal_seconds": 120.0,
    "translate_in": {"a": {"seconds": 20.0, "rel_err": 0.3}}, "translate_out": {"a": {"seconds": 10.0}},
    "stage1": {"seconds": 50.0, "steps": 100, "losses": [1.0, 0.8]},
    "stage2": {"seconds": 70.0, "steps": 100, "losses": [0.8, 0.7]},
    "scores": {"base": {"gsm8k": 0.5}, "stage1": {"gsm8k": 0.52}, "stage2": {"gsm8k": 0.4},
               "after_rollback": {"gsm8k": 0.52}, "final": {"gsm8k": 0.52}},
    "guard": {"events": [{"stage": 2, "failed": {"gsm8k": -0.1}, "action": "rolled back to stage 1"}]},
    "gates": {"general": {"a": 0.1}, "domain:a": {"a": 0.8}},
    "place": [{"id": "a", "kind": "mlp_cluster", "donor": "d", "host_layer": 9, "part_params": 1000}],
    "components": [{"id": "a", "kind": "mlp_cluster", "skill": "gsm8k", "host_layer": 9, "closed": False,
                    "contribution": 0.031, "text_gain": 0.02, "content_gain": 0.03, "heldout_loss_on": 1.0,
                    "heldout_loss_off": 1.02, "scores_without": {"gsm8k": 0.49}, "verdict": "contributes"}],
}


def test_stage_summaries_show_rollback_and_scores():
    rows = modal_studio.stage_summaries(FAKE_REPORT, 400.0)
    assert [row["name"] for row in rows] == ["place", "translate in", "translate out", "heal 1", "heal 2", "verify"]
    by_name = {row["name"]: row for row in rows}
    assert by_name["heal 1"]["practice_before"] == {"gsm8k": 0.5}
    assert by_name["heal 1"]["practice_after"] == {"gsm8k": 0.52}
    assert by_name["heal 2"]["practice_after"] == {"gsm8k": 0.4} and by_name["heal 2"]["rolled_back"] is True
    assert by_name["heal 1"]["rolled_back"] is False
    assert by_name["verify"]["practice_after"] == {"gsm8k": 0.52}
    assert by_name["verify"]["seconds"] == 150.0
    json.dumps(rows)


def test_component_rows_and_graft_entries():
    comps = [{"id": "a", "label_id": "partA", "kind": "mlp_cluster", "donor": "d", "skill": "gsm8k"}]
    rows = modal_studio.component_rows(FAKE_REPORT, comps, {"a": {"transfer": 0.4, "donor_kl_base": 1.0}})
    assert rows[0]["verdict"] == "contributes" and rows[0]["part_id"] == "partA"
    assert rows[0]["gate_domain"] == 0.8 and rows[0]["gate_general"] == 0.1 and rows[0]["fit_error"] == 0.3
    assert rows[0]["transfer"] == 0.4 and rows[0]["scores_without"] == {"gsm8k": 0.49}
    entries = modal_studio.graft_entries(FAKE_REPORT, comps, {"a": 77})
    assert entries[0]["id"] == "partA" and entries[0]["fit_r2"] == 0.7 and entries[0]["glue_params"] == 77
    ridge_only = dict(FAKE_REPORT, translate_in={"a": {"seconds": 1.0, "ridge_r2": 0.64, "rel_err": None}})
    rows = modal_studio.component_rows(ridge_only, comps, {})
    assert rows[0]["fit_error"] is None and rows[0]["fit_r2"] == 0.64
    assert modal_studio.graft_entries(ridge_only, comps, {})[0]["fit_r2"] == 0.64


# The whole synth_body with a fake compiler: no network, no GPU.

def patch_body(monkeypatch, tmp_path, report=None):
    from wsp import synth

    seen = {}

    class FakeModel:
        wsp_grafts = []

    def fake_synthesize(host, comps, texts, budget, practice=None, guard=None, out_dir=None, **options):
        seen["host"] = host
        seen["comps"] = comps
        seen["texts"] = texts
        seen["budget"] = budget
        seen["practice"] = practice
        seen["guard"] = guard
        os.makedirs(out_dir, exist_ok=True)
        return FakeModel(), None, dict(FAKE_REPORT if report is None else report)

    class FakeTokenizer:
        @classmethod
        def from_pretrained(cls, name):
            return cls()

    import sys
    import types

    transformers = types.ModuleType("transformers")
    transformers.AutoTokenizer = FakeTokenizer
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    monkeypatch.setattr(synth, "synthesize", fake_synthesize)
    monkeypatch.setattr(synth, "param_counts", lambda model: {"total": 1300000000, "per_token": 1100000000,
                                                              "parts": 1000, "glue": 77, "gates": 3, "lora": 5})
    monkeypatch.setattr(modal_studio, "synth_texts", lambda tok, pool, program: (
        {"align": ["a"], "general": ["g"]}, ["skill text"], {"skill_pool": pool}))
    monkeypatch.setattr(modal_studio, "synth_practice", lambda tasks, fraction: {
        "gsm8k": {"items": [1, 2, 3], "limit": 30}})
    monkeypatch.setattr(modal_studio, "run_probes", lambda *args: (
        {"base_kl": 0.01, "agree": 0.9, "tokens": 10, "text": "general pool, held out"},
        {"mlp1": {"transfer": 0.5, "donor_kl_base": 1.0, "donor_kl_written": 0.5}}, []))
    monkeypatch.setattr(modal_studio, "score_all", lambda path, split, tasks, fraction, note: (
        {task: 0.6 for task in tasks}, {task: 50 for task in tasks}))
    monkeypatch.setattr(modal_studio, "free_gpu", lambda: None)
    monkeypatch.setattr(modal_studio, "fresh_root", lambda prefix: str(tmp_path))
    monkeypatch.setattr(modal_studio.volume, "commit", lambda: None, raising=False)
    return seen


def synth_program():
    return {"host": HOST, "seed": 0, "masks": None, "task_vectors": [], "heal": {"steps": 100, "data": "math"},
            "grafts": [{"id": "mlp1", "at": 9, "control": None, "control_seed": 0}]}


def test_synth_body_returns_the_contract_shape(monkeypatch, tmp_path):
    seen = patch_body(monkeypatch, tmp_path)
    options = {"catalog_records": [RECORDS["mlp1"]], "pool": "math", "pipeline": "synth", "guards": ["gsm8k"],
               "max_total_params_b": 1.9, "max_run_params_b": 1.7, "token": "t1"}
    result = modal_studio.synth_body(synth_program(), "search", ["gsm8k"], 0.16, options)
    assert result["ok"] is True and result["pipeline"] == "synth"
    assert result["scores"] == {"gsm8k": 0.6}
    assert result["components"][0]["verdict"] == "contributes" and result["components"][0]["transfer"] == 0.5
    assert [row["name"] for row in result["stages"]][-1] == "verify"
    assert result["drift"]["base_kl"] == 0.01
    assert result["sizes"] == {"total": 1300000000, "per_token": 1100000000, "parts": 1000, "glue": 77,
                               "gates": 3, "lora": 5}
    assert result["seconds"]["heal"] == 120.0 and result["seconds"]["compile"] == 180.0
    assert result["seconds"]["eval"] >= 0 and "total" in result["seconds"]
    assert result["graft_report"][0]["id"] == "mlp1" and result["heal_loss"] == [1.0, 0.8, 0.8, 0.7]
    assert seen["host"] == HOST and seen["comps"][0]["id"] == "mlp1"
    assert seen["budget"]["max_params"] == 1900000000 and seen["budget"]["control_run"] is True
    assert seen["budget"]["stage1_steps"] == 50
    assert seen["guard"] == ["gsm8k"]
    json.dumps(result)


def test_a_random_control_program_skips_the_inside_control(monkeypatch, tmp_path):
    seen = patch_body(monkeypatch, tmp_path)
    program = synth_program()
    program["grafts"][0]["control"] = "random"
    options = {"catalog_records": [RECORDS["mlp1"]], "pool": "math", "pipeline": "synth"}
    result = modal_studio.synth_body(program, "search", ["gsm8k"], 1.0, options)
    assert result["ok"] is True and result["control_run"] is False
    assert seen["budget"]["control_run"] is False


def test_synth_body_refuses_an_unsupported_kind_before_any_work(monkeypatch, tmp_path):
    seen = patch_body(monkeypatch, tmp_path)
    program = synth_program()
    program["grafts"] = [{"id": "mask1", "at": 4, "control": None, "control_seed": 0}]
    result = modal_studio.synth_body(program, "search", ["gsm8k"], 1.0,
                                     {"catalog_records": [RECORDS["mask1"]], "pipeline": "synth"})
    assert result["ok"] is False and result["refused"] is True
    assert "deletion_mask" in result["error"]
    assert "comps" not in seen


def test_synth_body_reports_a_budget_refusal(monkeypatch, tmp_path):
    from wsp import synth

    patch_body(monkeypatch, tmp_path)

    def over(*args, **kwargs):
        raise synth.BudgetError("params total 3 is over the limit 2")

    monkeypatch.setattr(synth, "synthesize", over)
    result = modal_studio.synth_body(synth_program(), "search", ["gsm8k"], 1.0,
                                     {"catalog_records": [RECORDS["mlp1"]], "pipeline": "synth"})
    assert result["ok"] is False and result["refused"] is True and "over the limit" in result["error"]


def test_classic_option_runs_the_old_path(monkeypatch, tmp_path):
    called = []

    def fake_score_body(program, split, tasks, token, fidelity, options):
        called.append(token)
        return {"ok": True, "scores": {"gsm8k": 0.3}}

    monkeypatch.setattr(modal_studio, "score_body", fake_score_body)
    result = modal_studio.synth_body(synth_program(), "search", ["gsm8k"], 1.0, {"pipeline": "classic", "token": "q"})
    assert result["pipeline"] == "classic" and called == ["q"]


def fake_healdata_practice(calls):
    def practice(skill, count, reported=None, seed=0):
        calls.append((skill, count, list(reported)))
        if skill == "code":
            raise ValueError("every practice source of code is in the reported set")
        return [{"prompt": "p", "answer": "1", "kind": "number", "choices": None, "source": "ChilleD/SVAMP"}]
    return practice


def test_practice_comes_from_benchmarks_that_are_not_scored(monkeypatch):
    from wsp import healdata

    calls = []
    monkeypatch.setattr(healdata, "practice", fake_healdata_practice(calls))
    out = modal_studio.synth_practice(["gsm8k", "mbpp", "humaneval"], 0.16)
    assert list(out) == ["gsm8k"]
    assert [call[0] for call in calls] == ["math", "code"]
    for call in calls:
        assert "gsm8k" in call[2] and "mbpp" in call[2] and "humaneval" in call[2]
    assert call[1] == 30
    scorer = out["gsm8k"]
    assert modal_studio.practice_count(scorer) == 1
    for source in scorer.sources:
        assert healdata.why_reported(source, calls[0][2]) is None
    assert callable(scorer)


def test_a_task_without_a_practice_source_is_reported_not_hidden():
    notes = modal_studio.practice_gaps(["gsm8k", "mbpp"], {"gsm8k": object()}, ["if_guard", "gsm8k"])
    text = " ".join(notes)
    assert "mbpp" in text and "no practice score or guard ran" in text
    assert "if_guard" in text and "not enforced at compile time" in text
    assert "The guard gsm8k" not in text


def test_a_guard_without_practice_falls_back_to_every_practice_skill():
    practice = {"gsm8k": object(), "mmlu": object()}
    assert modal_studio.synth_guards(["if_guard"], practice) is None
    assert modal_studio.synth_guards([], practice) is None
    assert modal_studio.synth_guards(None, practice) is None
    assert modal_studio.synth_guards(["if_guard", "mmlu"], practice) == ["mmlu"]


def test_synth_body_guards_the_practice_skills_when_the_spec_guard_has_none(monkeypatch, tmp_path):
    seen = patch_body(monkeypatch, tmp_path)
    options = {"catalog_records": [RECORDS["mlp1"]], "pool": "math", "pipeline": "synth", "guards": ["if_guard"],
               "max_total_params_b": 1.9, "max_run_params_b": 1.7, "token": "t2"}
    result = modal_studio.synth_body(synth_program(), "search", ["gsm8k"], 0.16, options)
    assert seen["guard"] is None
    assert any("if_guard" in note and "not enforced" in note for note in result["notes"])


# The modal backend and the builder, with a fake Modal function.

def test_builder_score_carries_components_stages_drift(modal_client):
    builder_id = new_builder(modal_client)
    job = run_builder_job(modal_client, builder_id, modal_program(), "score")
    assert job["status"] == "done", job
    result = job["result"]
    assert result["pipeline"] == "synth" and result["heal_note"] is None
    assert result["components"][0]["id"] == MODAL_PART
    assert result["components"][0]["verdict"] == "contributes" and result["components"][0]["stub"] is False
    assert result["components"][0]["label"] and result["components"][0]["kind"] == "mlp_cluster"
    assert result["components"][0]["detail"]["contribution"] == 0.04
    assert [row["name"] for row in result["stages"]][:2] == ["place", "translate in"]
    assert result["drift"]["base_kl"] == 0.01
    assert result["sizes"]["total"] == 1300000000
    assert result["params_b"] == 1.3 and result["run_params_b"] == 1.1
    names = [name for name, args in modal_client.modal_calls]
    assert "synth_remote" in names and "score_remote" not in names
    sent_options = [args for name, args in modal_client.modal_calls if name == "synth_remote"][0][4]
    assert sent_options["pipeline"] == "synth" and sent_options["pool"]
    assert sent_options["max_total_params_b"] is not None
    assert sent_options["catalog_records"][0]["id"] == MODAL_PART


def test_heal_and_train_use_the_synth_path_too(modal_client):
    builder_id = new_builder(modal_client)
    program = ps.make_program(HOST, [ps.make_graft(MODAL_PART, 6, glue_rank=64)])
    for action in ["heal", "train"]:
        job = run_builder_job(modal_client, builder_id, program, action)
        assert job["status"] == "done", job
        assert job["result"]["pipeline"] == "synth" and job["result"]["components"] is not None
        assert job["result"]["heal_note"] is None
    sent = []
    for name, args in modal_client.modal_calls:
        if name == "synth_remote" and args[0]["grafts"][0]["control"] is None:
            sent.append(args[0])
    # Heal and Train run the same program on a build with parts: the default heal and no host LoRA.
    assert len(sent) >= 1
    assert sent[0]["heal"]["steps"] == 300
    for program_sent in sent:
        assert program_sent.get("lora") is None


def test_the_heal_warning_only_goes_with_classic_results(modal_client, monkeypatch):
    builder_id = new_builder(modal_client)
    synth_job = run_builder_job(modal_client, builder_id, modal_program(), "score")
    assert synth_job["result"]["heal_note"] is None
    monkeypatch.setenv("WSP_STUDIO_PIPELINE", "classic")
    classic_job = run_builder_job(modal_client, builder_id, modal_program(300), "score")
    result = classic_job["result"]
    assert result["pipeline"] == "classic" and result["components"] is None and result["stages"] is None
    assert result["heal_note"] == settings.HEAL_NOTE
    assert "score_remote" in [name for name, args in modal_client.modal_calls]
    unhealed = ps.make_program(HOST, [ps.make_graft(MODAL_PART, 6, glue_rank=64)])
    assert run_builder_job(modal_client, builder_id, unhealed, "score")["result"]["heal_note"] is None


def test_a_program_without_parts_stays_classic(modal_client):
    builder_id = new_builder(modal_client)
    bare = ps.make_program(HOST, [], ps.make_heal(100, data="math"))
    assert synth_report.pipeline_for(bare) == "classic"
    job = run_builder_job(modal_client, builder_id, bare, "score")
    assert job["result"]["pipeline"] == "classic" and job["result"]["components"] is None
    assert [name for name, args in modal_client.modal_calls][-1] == "score_remote"


def test_the_builder_refuses_a_mask_with_parts_in_plain_words(modal_client):
    builder_id = new_builder(modal_client)
    palette = modal_client.get("/api/catalog").json()
    mask_id = palette["masks"][0]["id"]
    program = modal_program()
    program["masks"] = {"op": "use", "id": mask_id}
    response = modal_client.post("/api/builder/" + builder_id + "/action", json={"program": program, "action": "score"})
    assert response.status_code == 400
    assert "deletion mask" in response.json()["error"]
    assert "synth_remote" not in [name for name, args in modal_client.modal_calls]


def test_a_failed_synth_call_gives_an_error_status(modal_client, monkeypatch):
    class Failing(FakeModalFunction):
        def remote(self, *args):
            if self.name.startswith("synth_remote"):
                return {"ok": False, "error": "ValueError: boom", "seconds": {"total": 3.0}}
            return FakeModalFunction.remote(self, *args)

    monkeypatch.setattr(modal_backend, "lookup", lambda name: Failing(name, modal_client.modal_calls))
    builder_id = new_builder(modal_client)
    job = run_builder_job(modal_client, builder_id, modal_program(), "score")
    assert job["status"] == "done"
    assert job["result"]["status"] != "ok" and job["result"]["scores"] is None
    assert job["result"]["components"] is None


def test_big_donors_use_the_h100_synth_function(modal_client):
    from studio.server import catalog as catalog_mod

    config, catalog = catalog_mod.load_catalog("modal")
    backend = modal_backend.ModalCompileEval(config, catalog)
    program = ps.make_program(HOST, [], None)
    program["inserts"] = [{"donor": "Qwen/Qwen2.5-14B-Instruct", "layers": [8, 20], "at": 6, "stitch": None,
                           "stitch_rank": None, "stitch_hidden": None, "part_lora_rank": None, "gate_init": 0.0,
                           "control": None, "control_seed": 0}]
    assert modal_backend.largest_donor_b(program, catalog) == 14.0
    assert backend.function("synth_remote", program).name == "synth_remote_h100"


# Stub verdicts.

def test_stub_builder_score_has_labeled_component_verdicts(stub_client):
    builder_id = new_builder(stub_client, roomy=True)
    palette = stub_client.get("/api/catalog").json()
    parts = [part for part in palette["parts"] if part["usable"]]
    seen = set()
    for part in parts:
        program = ps.make_program(HOST, [ps.make_graft(part["id"], part["default_at"], glue_rank=64)],
                                  ps.make_heal(100, data="math"))
        job = run_builder_job(stub_client, builder_id, program, "score")
        assert job["status"] == "done", job
        result = job["result"]
        assert result["pipeline"] == "synth" and result["heal_note"] is None
        component = result["components"][0]
        assert component["stub"] is True and component["id"] == part["id"]
        assert component["verdict"] in synth_report.VERDICTS and component["label"]
        assert all(row["stub"] is True for row in result["stages"])
        assert result["drift"]["text"] == "stub, not measured"
        seen.add(component["verdict"])
        again = run_builder_job(stub_client, builder_id, program, "score")["result"]
        assert again["components"] == result["components"]
    assert seen == set(synth_report.VERDICTS)


def test_stub_verdicts_follow_the_part_and_the_control():
    assert synth_report.stub_verdict("part_a", None) == synth_report.stub_verdict("part_a", None)
    assert synth_report.stub_verdict("part_a", "random") == "no better than random"
    verdicts = {synth_report.stub_verdict("part_" + str(index), None) for index in range(30)}
    assert verdicts == set(synth_report.VERDICTS)


def test_stub_run_leaderboard_entries_carry_components(stub_client):
    spec = make_spec(stub_client, evals=6)
    run_id = start_run(stub_client, spec)
    final = wait_done(stub_client, run_id)
    with_parts = [entry for entry in final["leaderboard"] if entry["pipeline"] == "synth"]
    assert len(with_parts) > 0
    for entry in with_parts:
        assert len(entry["components"]) == len(entry["program"]["grafts"]) + len(entry["program"].get("inserts") or [])
        assert all(item["stub"] is True for item in entry["components"])
        assert entry["comparison"]["no_component_contributes"] in (True, False)
    for entry in final["leaderboard"]:
        if entry["pipeline"] == "classic":
            assert entry["components"] is None


# The compare rule.

SPEC = {"benchmarks": [{"id": "gsm8k", "kind": "target", "base": 0.3, "lora": 0.35}], "limits": {},
        "host": HOST, "size": {"host_params_b": 1.24}}


def comparison_with(components, scores=None):
    return compare.build(SPEC, scores or {"gsm8k": 0.9}, 1.3, None, {"gsm8k": 0.3}, "search", False, models={},
                         components=components)


def test_a_result_whose_components_all_fail_is_not_a_success():
    plain = comparison_with(None)
    assert plain["verdict"] == "beats" or plain["verdict"] == "unproven" or plain["verdict"] == "failed"
    good = comparison_with([{"id": "a", "label": "Math block", "verdict": "contributes", "evidence": "practice"}])
    assert good["verdict"] == plain["verdict"] and good["no_component_contributes"] is False
    for verdicts in [["no better than random"], ["harmful"], ["harmful", "no better than random"]]:
        components = [{"id": "c" + str(i), "label": "Part " + str(i), "verdict": v, "evidence": "practice"} for i, v in enumerate(verdicts)]
        result = comparison_with(components)
        assert result["verdict"] == "failed"
        assert result["no_component_contributes"] is True
        assert "component" in result["headline"] and "Failed attempt" in result["headline"]
        assert verdicts[0] in result["headline"]


def test_one_contributing_component_keeps_the_verdict():
    components = [{"id": "a", "label": "A", "verdict": "harmful", "evidence": "practice"}, {"id": "b", "label": "B", "verdict": "contributes", "evidence": "practice"}]
    assert comparison_with(components)["no_component_contributes"] is False
    assert comparison_with(components)["verdict"] == comparison_with(None)["verdict"]


def test_a_failed_comparison_keeps_its_reason_and_adds_the_components():
    components = [{"id": "a", "label": "A", "verdict": "harmful", "evidence": "practice"}]
    low = comparison_with(components, {"gsm8k": 0.2})
    assert low["verdict"] == "failed"
    assert "is below" in low["headline"] and "harmful" in low["headline"]


def test_no_score_stays_none():
    result = compare.build(SPEC, None, 1.3, None, None, "search", False, models={},
                           components=[{"id": "a", "label": "A", "verdict": "harmful", "evidence": "practice"}])
    assert result["verdict"] == "none"


def test_builder_comparison_fails_when_no_stub_component_contributes(stub_client, monkeypatch):
    monkeypatch.setattr(synth_report, "stub_verdict", lambda part_id, control: "harmful")
    builder_id = new_builder(stub_client, roomy=True)
    palette = stub_client.get("/api/catalog").json()
    part = [item for item in palette["parts"] if item["usable"]][0]
    program = ps.make_program(HOST, [ps.make_graft(part["id"], part["default_at"], glue_rank=64)],
                              ps.make_heal(100, data="math"))
    job = run_builder_job(stub_client, builder_id, program, "score")
    comparison = job["result"]["comparison"]
    assert comparison["verdict"] == "failed" and comparison["no_component_contributes"] is True
    assert "harmful" in comparison["headline"]


def test_health_heal_note_is_still_a_classic_warning():
    assert settings.heal_note_for("synth", True) is None
    assert settings.heal_note_for("classic", False) is None


def test_text_only_verdicts_do_not_force_a_failure():
    plain = comparison_with(None)
    text_only = [{"id": "a", "label": "A", "verdict": "no better than random", "evidence": "text"},
                 {"id": "b", "label": "B", "verdict": "harmful", "evidence": "text"}]
    result = comparison_with(text_only)
    assert result["no_component_contributes"] is False and result["verdict"] == plain["verdict"]
    no_field = [{"id": "a", "label": "A", "verdict": "harmful"}]
    assert comparison_with(no_field)["no_component_contributes"] is False
    mixed = text_only + [{"id": "c", "label": "C", "verdict": "harmful", "evidence": "practice"}]
    assert comparison_with(mixed)["no_component_contributes"] is True
    assert "A is" not in comparison_with(mixed, {"gsm8k": 0.2})["headline"]


def test_component_rows_mark_the_evidence():
    report = {"components": [{"host_layer": 3, "verdict": "contributes", "contribution": 0.05},
                             {"host_layer": 4, "verdict": "contributes", "contribution": None, "text_gain": 0.1}]}
    comps = [{"id": "a", "label_id": "a", "kind": "block", "donor": "d", "skill": "gsm8k"},
             {"id": "b", "label_id": "b", "kind": "block", "donor": "d", "skill": "humaneval"}]
    rows = modal_studio.component_rows(report, comps, {})
    assert [row["evidence"] for row in rows] == ["practice", "text"]
    public = synth_report.public_components(rows, type("C", (), {"records": {}})(), False)
    assert [row["evidence"] for row in public] == ["practice", "text"]


def test_practice_scorer_grades_generated_text(monkeypatch):
    from wsp import skills_eval

    items = [{"prompt": "1+1?", "answer": "2", "kind": "number", "choices": None, "source": "a/b"},
             {"prompt": "Pick", "answer": "B", "kind": "choice", "choices": ["A. x", "B. y"], "source": "a/c"}]
    scorer = modal_studio.PracticeScorer(items)
    monkeypatch.setattr(skills_eval, "render_prompt", lambda tok, text: text)
    answers = {"number": ["The answer is 2."], "choice": ["A"]}

    def fake_generate(model, tok, prompts, max_new_tokens=512, batch_size=32, stop_strings=None):
        return answers["number"] if max_new_tokens == 512 else answers["choice"]

    monkeypatch.setattr(skills_eval, "generate_texts", fake_generate)
    assert scorer(None, None) == 0.5
    assert scorer.sources == ["a/b", "a/c"]
