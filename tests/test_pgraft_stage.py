"""The pgraft method of the compiler front door: choice, plan fields, and a tiny end-to-end compile. CPU only, no network."""

import json
import os

import pytest
import torch

pytest.importorskip("transformers")

from tests.test_auto import LLAMA, Q7, REPO, cfg  # noqa: E402
from tests.test_xfam import (DONOR_WORDS, HOST_WORDS, items, make_tok, pool_for, texts, tiny_donor,  # noqa: E402
                             tiny_host)
from wsp import auto, pgraft_stage, xfam  # noqa: E402
from wsp.auto import LocalBackend, StaticLineage, plan  # noqa: E402

LLAMA1B = cfg(16, 2048, 32, 8, 8192, 128256, tied=True, head_dim=64, model_type="llama")
HOST, DONOR = "unsloth/Llama-3.2-1B-Instruct", "Qwen/Qwen2.5-7B-Instruct"


def lineage(kinds=None, extra=None):
    models = {
        HOST: {"params": 1235814400, "config": LLAMA1B, "tokenizer": "llama"},
        DONOR: {"params": 7.616e9, "config": Q7, "tokenizer": "qwen"},
        "meta/Llama-8B": {"params": 8.03e9, "config": LLAMA, "tokenizer": "llama"},
        "q/MoE": {"params": 30.5e9, "config": cfg(48, 2048, 32, 4, 6144, 151936, experts=128, top_k=8, expert_size=768,
                                                 head_dim=128), "tokenizer": "qwen3"},
    }
    models.update(extra or {})
    return StaticLineage(models, kinds or {(HOST, "meta/Llama-8B"): "same_family_other_size"})


def request(skill="knowledge", source=DONOR, method=None, limits=None, base=HOST):
    add = {"skill": skill, "sources": [source]}
    if method:
        add["method"] = method
    return {"name": "t", "base": base, "add": [add], "reported": ["cais/mmlu"], "seed": 1,
            "limits": limits or {"max_total_params_b": 5.5, "max_active_params_b": 5.5, "max_heal_minutes": 150}}


def make_plan(req, lin=None):
    return plan(req, lineage=lin or lineage(), atlas_entries=[], library_models=[])


# ---------------------------------------------------------------- choice


def test_foreign_knowledge_source_gets_pgraft():
    p = make_plan(request())
    assert p["feasible"], p["refusals"]
    assert [s["method"] for s in p["stages"]] == ["pgraft"]
    assert p["sources"][0]["chosen"] == "pgraft"
    # the existing method stays listed behind it
    assert [m["method"] for m in p["sources"][0]["methods"]][:2] == ["pgraft", "synth"]


@pytest.mark.parametrize("skill", ["math", "code", "reasoning"])
def test_other_skills_keep_the_old_method(skill):
    p = make_plan(request(skill=skill, limits={"max_total_params_b": 20, "max_active_params_b": 20}))
    assert [s["method"] for s in p["stages"]] == ["synth"]


def test_same_lineage_sources_keep_the_old_methods():
    p = make_plan(request(source="meta/Llama-8B", limits={"max_total_params_b": 20, "max_active_params_b": 20}))
    assert "pgraft" not in [m["method"] for m in p["sources"][0]["methods"]]
    assert p["stages"][0]["method"] == "synth"


def test_a_moe_donor_is_not_a_pgraft_source():
    p = make_plan(request(source="q/MoE", limits={"max_total_params_b": 40, "max_active_params_b": 40}))
    assert "pgraft" not in [m["method"] for m in p["sources"][0]["methods"]]


def test_the_request_can_name_the_method_and_a_wrong_one_is_refused():
    assert make_plan(request(method="pgraft"))["stages"][0]["method"] == "pgraft"
    p = make_plan(request(skill="math", method="pgraft", limits={"max_total_params_b": 20, "max_active_params_b": 20}))
    assert not p["feasible"] and "not valid" in " ".join(p["refusals"])


def test_a_limit_that_does_not_fit_falls_back_and_names_why():
    p = make_plan(request(limits={"max_total_params_b": 5.0, "max_active_params_b": 5.0}))
    chosen = p["sources"][0]["chosen"]
    opt = [m for m in p["sources"][0]["methods"] if m["method"] == "pgraft"][0]
    assert not opt["fits"] and "max_total_params_b" in opt["why_not"]
    assert chosen != "pgraft"


def test_two_different_seam_picks_fall_back_to_the_common_method():
    req = request()
    req["add"].append({"skill": "knowledge", "sources": ["meta/Llama-8B"]})
    req["limits"] = {"max_total_params_b": 40, "max_active_params_b": 40}
    p = make_plan(req)
    assert p["feasible"] and [s["method"] for s in p["stages"]] == ["synth"]


def test_pgraft_takes_one_source():
    req = request()
    req["add"].append({"skill": "knowledge", "sources": ["q/Other7B"]})
    req["limits"] = {"max_total_params_b": 40, "max_active_params_b": 40}
    lin = lineage(extra={"q/Other7B": {"params": 7.6e9, "config": Q7, "tokenizer": "qwen"}})
    p = make_plan(req, lin)
    assert not p["feasible"] and "one donor slice" in " ".join(p["refusals"])


# ---------------------------------------------------------------- plan fields


def test_plan_fields_state_layers_sizes_data_steps_and_controls():
    p = make_plan(request())
    st = p["stages"][0]
    assert st["host_layer"] == 7 and st["donor_span"] == [4, 20] and st["donor_slice_layers"] == 17
    assert st["enc"]["depth"] == 2 and st["enc"]["params"] == 13770625 + 7 and st["enc"]["reads"] == [1, 2, 3, 4, 5, 6, 7] and st["dec"]["rank"] == 256
    assert st["dec"]["params"] == 256 * (3584 + 2048)
    assert st["lora_rank"] == 16 and st["lora_params"] == 11272192
    assert st["settings"]["steps"] == 4000 and st["settings"]["align_steps"] == 600
    assert st["data"]["pool"] == "xfam_mc_v1" and "held-out" not in st["data"]["train_items"]
    assert st["data"]["scored_on"] == ["mmlu_dev", "mmlu_held", "arc_c", "obqa"]
    names = [c["name"] for c in st["controls"]]
    assert names == ["identity", "scrambled", "lora_match", "lora_big", "fullft"]
    assert [c["name"] for c in st["controls"] if "compiler" in c["run_by"]] == ["identity", "scrambled"]


def test_params_per_token_match_the_measured_row_within_a_tenth_of_a_percent():
    st = make_plan(request())["stages"][0]
    measured = 5213011330
    assert abs(st["params_run_after"] - measured) / measured < 0.001
    assert st["params_total_after"] > st["params_run_after"]    # the stored total counts the LoRA


def test_plan_text_prints_the_stage_and_warns_with_the_measured_numbers():
    p = make_plan(request())
    text = auto.format_plan(p)
    for needle in ("Host layer 7", "layers 4..20 of 28", "Enc depth 2", "Dec rank 256", "params per token after the write-in: 5.213B",
                   "control identity", "control fullft", "xfam_mc_v1", "4000 task"):
        assert needle in text, needle
    joined = " ".join(p["warnings"])
    assert "59.6" in joined and "44.6" in joined and "50.3" in joined and "3B host" in joined


def test_a_different_pair_gets_a_warning():
    lin = lineage(extra={"x/host": {"params": 1.5e9, "config": LLAMA1B, "tokenizer": "x"}})
    p = make_plan(request(base="x/host"), lin)
    assert any("not the measured pair" in w for w in p["warnings"])


def test_the_heal_limit_warning():
    p = make_plan(request(limits={"max_total_params_b": 5.5, "max_active_params_b": 5.5, "max_heal_minutes": 20}))
    assert any("max_heal_minutes is 20" in w for w in p["warnings"])


def test_the_demo_request_plans_offline_and_the_saved_plan_names_pgraft():
    req = auto.load_request(os.path.join(REPO, "demo", "requests", "xfam_knowledge.json"))
    assert req["seed"] == 1 and req["add"][0]["sources"] == [DONOR]
    p = make_plan(req)
    assert p["feasible"] and p["stages"][0]["method"] == "pgraft"
    saved = os.path.join(REPO, "demo", "plans", "xfam_knowledge.txt")
    if os.path.exists(saved):
        assert "pgraft" in open(saved).read()


# ---------------------------------------------------------------- tiny end to end


def tiny_backend(dtype):
    host_tok, donor_tok = make_tok(HOST_WORDS), make_tok(DONOR_WORDS)
    backend = LocalBackend({"tiny/host": (tiny_host(dtype), host_tok), "tiny/donor": (tiny_donor(dtype), donor_tok)})
    its = items(40)
    evals = {"mmlu_dev": its[:12], "mmlu_held": its[12:28], "arc_c": its[28:34], "obqa": its[34:]}
    pool = pool_for(40)
    backend.pgraft_data_fn = lambda stage, donor, dtok, host: {"items": pool["items"], "soft": pool["soft"], "evals": evals,
                                                               "source": "tiny test pool"}
    t = texts()
    backend.pgraft_texts_fn = lambda stage, request: {"align": t[:24], "align_eval": t[24:32], "general": t[32:] * 3}
    return backend


def tiny_plan(backend):
    models = {}
    for name in ("tiny/host", "tiny/donor"):
        model, _ = backend.models[name]
        models[name] = auto.static_row(model, tokenizer_id=name)
    req = {"name": "tiny_xfam", "base": "tiny/host", "add": [{"skill": "knowledge", "sources": ["tiny/donor"]}],
           "reported": ["cais/mmlu"], "seed": 1}
    return req, plan(req, lineage=StaticLineage(models, {}), atlas_entries=[], library_models=[])


SMALL = {"steps": 6, "align_steps": 4, "geo_steps": 2, "batch": 4, "dec_rank": 8, "lora_rank": 2, "seq_len": 24,
         "donor_seq_len": 32}


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_tiny_request_compiles_saves_reloads_and_scores(tmp_path, dtype):
    backend = tiny_backend(dtype)
    req, p = tiny_plan(backend)
    assert p["feasible"], p["refusals"]
    assert [s["method"] for s in p["stages"]] == ["pgraft"]
    result = auto.make(req, str(tmp_path), backend, plan_=p, settings={"pgraft_spec": SMALL, "pgraft_controls": ["identity"]})
    report = result.report
    row = report["pgraft"]["graft"]
    # the artifact sits under a versioned path and loads back into the network that was scored
    assert os.path.basename(row["artifact"]) == "pgraft_v1" and os.path.exists(os.path.join(row["artifact"], "glue.safetensors"))
    assert row["reload_gap_mmlu_held"] == 0.0
    assert set(row["scores"]) == {"mmlu_dev", "mmlu_held", "arc_c", "obqa"}
    assert set(report["pgraft"]["reference"]["scores"]) == set(row["scores"])
    assert "identity" in report["pgraft"]["controls"] and report["pgraft"]["controls"]["identity"]["slice_params"] == 0
    assert row["params_per_token"] == row["params_base"] + row["slice_params"] + row["trainable_glue"]
    assert report["verdict"]["verdict"] in ("met", "partly met", "not met")
    assert report["reported_scores"]["eval_items"]["mmlu_held"] == 16
    on_disk = json.load(open(os.path.join(str(tmp_path), "sapien_report.json")))
    assert on_disk["pgraft"]["graft"]["scores"] == row["scores"]
    # the loader rebuilds the same network from the artifact alone
    model, meta = xfam.load_artifact(row["artifact"], base=backend.load("tiny/host")[0], donor=backend.load("tiny/donor")[0],
                                     device="cpu")
    tok = make_tok(HOST_WORDS)
    assert xfam.score_letters(model, tok, items(40)[12:28]) == row["scores"]["mmlu_held"]
    assert meta["host"] == "tiny/host" and meta["donor"] == "tiny/donor"
    assert result.model is not None and len(result.model.wsp_grafts) == 1


def test_a_pgraft_plan_that_is_not_feasible_does_not_run(tmp_path):
    backend = tiny_backend(torch.float32)
    req, p = tiny_plan(backend)
    p["feasible"] = False
    p["refusals"] = ["test"]
    with pytest.raises(auto.PlanError):
        auto.make(req, str(tmp_path), backend, plan_=p)
