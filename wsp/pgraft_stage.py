"""The pgraft method of the compiler front door (wsp/auto.py): planning pieces and the make-time runner.

A parallel graft writes a frozen slice of a donor from another family or tokenizer into a small host as one
network (wsp/pgraft.py). The recipe, its numbers and its controls are in docs/write_in_xfam.md; the training,
saving and loading code is wsp/xfam.py. This module decides when the method applies, sizes it from configs
without reading weights, describes the stage the plan prints, and runs it for auto.make().
"""

import copy
import gc
import json
import os
import time

from wsp import xfam

# Skill pools whose items are four-option questions (knowledge style). The recipe trains on such items.
POOLS = ("knowledge", "medicine", "law")
# The multilingual skill is planned as a write-in too. Its data, controls and runs live in modal_wi_lang.py
# (docs/write_in_language.md); the compiler's make path does not run it.
LANGUAGE_POOL = "multilingual"
CONTROLS_RUN =("identity", "scrambled")   # default for settings["pgraft_controls"]; the others need a separate driver

MEASURED = {
    "host": "unsloth/Llama-3.2-1B-Instruct",
    "donor": "Qwen/Qwen2.5-7B-Instruct",
    "date": "2026-11-15",
    "source": "results/wi_xfam/table.md",
    "mmlu_held": {"base": 44.6, "graft": 59.6, "graft_sd": 0.8, "identity": 50.3, "scrambled": 50.3, "lora_match": 49.7,
                  "lora_big": 50.0, "fullft": 51.1, "donor": 70.8},
    "arc_c": {"base": 55.3, "graft": 85.3, "fullft": 69.5, "donor": 88.8},
    "pool_items": 44155,
    "pool_donor_acc": 85.9,
    "decode_tps": "27 to 36 (hook-based path, batch 1) against 103 for the base",
    "gpu_hours": 0.65,
    "gpu": "H200",
}

def eligible(skill, rel, base_cfg, src_cfg):
    """(True, reason) when pgraft applies to this source, else (False, why not)."""
    from wsp import auto

    pool = auto.SKILL_POOLS.get(skill)
    if pool not in POOLS and pool != LANGUAGE_POOL:
        return False, "skill %r is not a knowledge or multiple-choice skill (pools %s)" % (skill, ", ".join(POOLS))
    foreign = rel["kind"] == "foreign" or not rel["flags"].get("tokenizer_same", True)
    if not foreign:
        return False, "the source shares the base's family and tokenizer; the same-lineage methods apply"
    if rel["flags"].get("quantized"):
        return False, "the source is quantized"
    if base_cfg.get("experts") or src_cfg.get("experts"):
        return False, "the write-in is built for dense hosts and donors"
    for cfg in (base_cfg, src_cfg):
        if not cfg.get("layers") or not cfg.get("hidden"):
            return False, "the config lacks layers or hidden size"
    if src_cfg["layers"] < 7:
        return False, "the donor has under 7 layers, too few to cut a slice"
    kind = "multilingual" if pool == LANGUAGE_POOL else "knowledge"
    reason = ("other tokenizer or family and a %s skill: a frozen slice of the donor runs as a parallel branch "
              "of the host (measured on one pair, see the warning)" % kind)
    return True, reason


def sizes(base_cfg, src_cfg, s):
    """Spec and parameter counts of the write-in from configs alone."""
    from wsp import auto

    spec = xfam.spec_for(base_cfg, src_cfg)
    n_layers = spec["last"] - spec["first"] + 1
    slice_params = n_layers * auto.layer_params(src_cfg)
    glue = xfam.glue_params(spec)
    lora = auto.lora_params(base_cfg, spec["lora_rank"], base_cfg["layers"])
    return {"spec": spec, "slice_layers": n_layers, "slice_params": slice_params, "glue_params": glue,
            "lora_params": lora, "add_total": slice_params + glue + lora, "add_run": slice_params + glue}


LANGUAGE_DATA = {
    "pool": "lang_v1", "dir": "/vol/wi_lang/v1",
    "train_items": "MMMLU questions outside the held-out, dev and English xfam indexes (ar, ja, zh, ko, hi), XNLI train "
                   "recast as three-option items (ar, zh, hi), and the English xfam pool",
    "soft_labels": "the donor's answer-letter distribution on every item, scored once and cached",
    "removed": "every prompt equal to a Belebele, MMMLU held-out, dev or English MMLU prompt, and repeats",
    "size": "78457 items in the cached pool (results/wi_lang/prep_report.json)",
    "text": "alignment texts from Wikipedia (en, ar, ja, zh, ko, hi), general texts from English Wikipedia, plain text",
    "scored_on": ["Belebele test in 10 languages", "MMMLU held-out 1000 per language", "mmlu_held"],
}


def auto_pool(skill):
    from wsp import auto
    return auto.SKILL_POOLS.get(skill)


def stage_info(x, base_cfg, s, controls_run):
    """The pgraft stage fields of the plan for one source item x (from plan())."""
    sz = sizes(base_cfg, x["facts"]["config"], s)
    spec = sz["spec"]
    d_host, d_donor = spec["d_host"], spec["d_donor"]
    enc_params = sz["glue_params"] - spec["dec_rank"] * (d_donor + d_host) - d_host - 1
    presets = {"identity": "identity_l", "scrambled": "scrambled_l", "lora_match": "lora_match_l", "lora_big": "lora_big",
               "fullft": "fullft"}
    controls = []
    for name, what in xfam.CONTROLS:
        if name in controls_run:
            run_by = "the compiler (settings pgraft_controls)"
        else:
            run_by = "modal_wi_xfam.py preset %s (rows in results/wi_xfam, seeds 1 to 3)" % presets[name]
        controls.append({"name": name, "what": what, "same": "data, steps, optimizer, and last-step selection",
                         "run_by": run_by})
    language = auto_pool(x["skill"]) == LANGUAGE_POOL
    if language:
        for c in controls:
            c["run_by"] = "modal_wi_lang.py (rows in results/wi_lang, docs/write_in_language.md)"
    return {
        "source": x["source"], "skill": x["skill"], "host_layer": spec["host_layer"],
        "donor_span": [spec["first"], spec["last"]], "donor_layers": x["facts"]["config"]["layers"],
        "donor_slice_layers": sz["slice_layers"],
        "layer_rule": "host layer = %s; donor span = %s" % (xfam.HOST_LAYER_RULE, xfam.DONOR_SPAN_RULE),
        "enc": {"depth": spec["enc_depth"], "d_host": d_host, "d_donor": d_donor, "params": int(enc_params)},
        "dec": {"rank": spec["dec_rank"], "params": int(spec["dec_rank"] * (d_donor + d_host))},
        "gate_params": d_host + 1, "lora_rank": spec["lora_rank"], "lora_params": int(sz["lora_params"]),
        "slice_params": int(sz["slice_params"]), "glue_params": int(sz["glue_params"]),
        "params_stored_added": int(sz["add_total"]), "params_per_token_added": int(sz["add_run"]),
        "data": LANGUAGE_DATA if language else {
            "pool": "xfam_mc_%s" % xfam.DATA_VERSION, "dir": s.get("pgraft_data_dir") or xfam.DATA_DIR,
            "train_items": "MMLU auxiliary_train, CommonsenseQA, OpenBookQA, QASC, LogiQA train (benchmark train splits)",
            "soft_labels": "the donor's answer-letter distribution on every item, scored once and cached",
            "removed": "every prompt equal to a prompt of an eval set",
            "size": "%d items in the cached pool (donor accuracy %.1f percent)" % (MEASURED["pool_items"], MEASURED["pool_donor_acc"]),
            "text": "alignment texts (healdata prose, knowledge, medicine) and general texts (healdata general, prose) of round 22, plain text",
            "scored_on": list(xfam.EVAL_NAMES),
        },
        "spec": spec,
        "controls": controls,
        "estimate": {"gpu": MEASURED["gpu"], "gpu_hours": MEASURED["gpu_hours"],
                     "basis": "0.62 to 0.66 GPU hours per seed on one H200 in the wi_xfam rows (2026-11-15), more per control"},
        "settings": {k: spec[k] for k in ("steps", "align_steps", "batch", "lora_rank", "enc_depth", "dec_rank", "kd_w",
                                          "preserve", "align_reg")},
    }


# ---------------------------------------------------------------- make


def pgraft_verdict(final, controls, base_held):
    """met / partly met / not met for the write-in, by held-out MMLU: above the base and above every control that ran."""
    graft = final["mmlu_held"] * 100
    reasons = []
    gain = graft - base_held * 100
    best_control = None
    for name, scores in controls.items():
        value = scores["mmlu_held"] * 100
        if best_control is None or value > best_control[1]:
            best_control = (name, value)
    reasons.append("MMLU held-out %.1f against the base %.1f (%+.1f)" % (graft, base_held * 100, gain))
    if best_control is not None:
        reasons.append("best control %s %.1f (%+.1f)" % (best_control[0], best_control[1], graft - best_control[1]))
    one_seed = "one seed: a claim needs three seeds"
    if gain < 1.0 or (best_control is not None and graft <= best_control[1]):
        return "not met", reasons + [one_seed]
    if gain < 3.0 or (best_control is not None and graft - best_control[1] < 3.0):
        return "partly met", reasons + [one_seed]
    return "met", reasons + [one_seed]


def run_make(req, plan_, backend, out_dir, started, atlas_entries=None, log=print):
    """Build, save, reload, and score the write-in of a one-stage pgraft plan. Returns (report dict, model, tokenizer)."""
    import torch

    from wsp import auto, pgraft

    s = plan_["settings"]
    stage = plan_["stages"][0]
    if auto_pool(stage["skill"]) == LANGUAGE_POOL:
        raise ValueError("the language write-in is run by modal_wi_lang.py, not by the compiler (docs/write_in_language.md)")
    seed = req["seed"]
    spec = dict(stage["spec"])
    spec.update(s.get("pgraft_spec", {}))
    host_id, donor_id = req["base"], stage["source"]
    os.makedirs(out_dir, exist_ok=True)

    base, tok = backend.load(host_id)
    donor, donor_tok = backend.load(donor_id)
    device = next(base.parameters()).device
    data = backend.pgraft_data(stage, donor, donor_tok, host_id)
    texts = backend.pgraft_texts(stage, req)
    evals = data["evals"]
    if s.get("pgraft_eval_limit"):
        evals = {k: v[:s["pgraft_eval_limit"]] for k, v in evals.items()}
    pool = {"items": data["items"], "soft": data["soft"]}
    if s.get("pgraft_pool_limit"):
        pool = {k: v[:s["pgraft_pool_limit"]] for k, v in pool.items()}
    log("pgraft: pool %d items, evals %s, data %s" % (len(pool["items"]), {k: len(v) for k, v in evals.items()}, data.get("source")))

    t0 = time.time()
    base_scores = xfam.evaluate(base, tok, evals)
    base_decode = xfam.decode_tps(base, tok) if device.type == "cuda" else None
    log("pgraft: base scores %s" % json.dumps(base_scores))
    ref = copy.deepcopy(base)
    for p in ref.parameters():
        p.requires_grad_(False)
    ref.eval()
    sizes_before = {"total": sum(p.numel() for p in base.parameters()),
                    "per_token": sum(p.numel() for p in base.parameters()), "kind": "plain"}
    built = xfam.train_network(base, tok, donor, donor_tok, pool, texts, spec, seed, arm="graft", ref=ref, log=log)
    row = built.row
    row["scores_in_memory"] = xfam.evaluate(base, tok, evals)
    with pgraft.grafts_off(base):
        row["scores_graft_off"] = xfam.evaluate(base, tok, evals, names=("mmlu_dev", "mmlu_held"))
    log("pgraft: in memory %s" % json.dumps(row["scores_in_memory"]))
    meta = {"host": host_id, "donor": donor_id, "seed": seed, "request": req["name"], "plan_method": "pgraft",
            "data": data.get("source"), "trained_on": stage["data"]["train_items"]}
    path = xfam.save_artifact(out_dir, base, built, spec, meta)
    backend.checkpoint()
    row["artifact"] = path
    row["artifact_bytes"] = row_bytes(path)
    del base, ref
    built.grafts = []
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    fresh, _ = backend.load(host_id)
    model, loaded = xfam.load_artifact(path, base=fresh, donor=donor, device=str(device))
    row["scores"] = xfam.evaluate(model, tok, evals)
    row["decode_tps"] = xfam.decode_tps(model, tok) if device.type == "cuda" else None
    log("pgraft: reloaded %s" % json.dumps(row["scores"]))
    row["reload_gap_mmlu_held"] = round(row["scores"]["mmlu_held"] - row["scores_in_memory"]["mmlu_held"], 5)
    row["seconds"] = round(time.time() - t0, 1)
    row["gpu_hours"] = round(row["seconds"] / 3600.0, 3)
    report_pg = {"reference": {"scores": base_scores, "decode_tps": base_decode, "params": sizes_before["total"]},
                 "graft": row, "controls": {}, "data_source": data.get("source"), "artifact_meta": loaded}
    partial = _write_report(out_dir, req, plan_, report_pg, started, sizes_before, evals, stage)
    backend.checkpoint()
    log("pgraft: graft done, wrote the report; running controls %s" % list(s.get("pgraft_controls", CONTROLS_RUN)))

    control_scores = {}
    for name in s.get("pgraft_controls", CONTROLS_RUN):
        c0 = time.time()
        cbase, ctok = backend.load(host_id)
        cref = copy.deepcopy(cbase)
        for p in cref.parameters():
            p.requires_grad_(False)
        cref.eval()
        cb = xfam.train_network(cbase, ctok, donor, donor_tok, pool, texts, spec, seed, arm=name, ref=cref, log=log)
        crow = cb.row
        crow["scores"] = xfam.evaluate(cbase, ctok, evals)
        crow["gpu_hours"] = round((time.time() - c0) / 3600.0, 3)
        report_pg["controls"][name] = crow
        control_scores[name] = crow["scores"]
        log("pgraft: control %s %s" % (name, json.dumps(crow["scores"])))
        del cbase, cref, cb
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        _write_report(out_dir, req, plan_, report_pg, started, sizes_before, evals, stage)
        backend.checkpoint()

    final = _write_report(out_dir, req, plan_, report_pg, started, sizes_before, evals, stage, control_scores=control_scores)
    return final, model, tok


def row_bytes(path):
    return sum(os.path.getsize(os.path.join(path, f)) for f in os.listdir(path))


def _write_report(out_dir, req, plan_, pg, started, sizes_before, evals, stage, control_scores=None):
    from wsp import auto

    row = pg["graft"]
    controls = control_scores if control_scores is not None else {k: v["scores"] for k, v in pg["controls"].items()}
    word, reasons = pgraft_verdict(row["scores"], controls, pg["reference"]["scores"]["mmlu_held"])
    label = "%s:%s" % (stage["skill"], stage["source"])
    sizes = {"total": int(row["params_per_token"] + row["trainable"] - row["trainable_glue"]),
             "per_token": int(row["params_per_token"]), "kind": "pgraft"}
    limits = req["limits"]
    problems = auto.check_limits(sizes["total"], sizes["per_token"], limits)
    seconds = time.time() - started
    minutes = limits["max_heal_minutes"]
    if minutes is not None and seconds > minutes * 60.0:
        problems.append("compile took %.1f minutes, above max_heal_minutes %.1f" % (seconds / 60.0, minutes))
    whole = word if not problems else "not met"
    final = {
        "name": req["name"], "request": req, "plan": plan_,
        "stages": [{"method": "pgraft", "status": "ok", "saved": row["artifact"], "seconds": row["seconds"]}],
        "pgraft": pg,
        "probe": {"note": "text probes are not run for pgraft; the write-in is scored on held-out four-option items"},
        "sizes": {"base": sizes_before, "final": sizes, "limits": limits, "limit_problems": problems},
        "seconds": round(seconds, 2),
        "verdicts": {label: {"verdict": word, "reasons": reasons}},
        "verdict": {"verdict": whole, "reasons": list(problems) + reasons},
        "scored_on_reported": True,
        "reported_scores": {"result": row["scores"], "base": pg["reference"]["scores"],
                            "controls": controls, "eval_items": {k: len(v) for k, v in evals.items()},
                            "note": "held-out test items; no arm trained on them"},
    }
    with open(os.path.join(out_dir, "sapien_report.json"), "w") as handle:
        json.dump(auto._json_safe(final), handle, indent=1)
    return final
