"""Does a trained stitch make a donor block graft work? One seed, same-tokenizer pair.

Host Qwen/Qwen2.5-0.5B-Instruct, donor Qwen/Qwen2.5-Math-1.5B (same tokenizer).
The stitch study found that training the stitch on the donor's own next-token
loss cuts the stitch penalty from 0.65 to 0.09 nats for this pair (host layer 7
into donor layer 8). This puts that stitch into a real graft and scores it.

Arms (seed 0):
  host           the untrained host
  block_ridge    donor layers 8..19 at host layer 7, ridge-fit w_in, rank-64 w_out
  block_stitch   the same block, w_in trained as a linear stitch (donor_lm, 1000 steps)
  random_stitch  control: a random 12-layer donor block, identical stitch and heal
  lora           control: a host LoRA matched to block_stitch's trainable glue params
Every trained arm heals 300 steps on GSM8K train (lr 3e-4, batch 8, seq 384).
Scored with eval() on the GSM8K search split: interim numbers, not results.

Each arm runs in its own container and its own WSP_ROOT, so parallel arms
never append to the same files on the volume.
"""
import json
import os

from wsp_eval import harness, splits

HOST = "Qwen/Qwen2.5-0.5B-Instruct"
DONOR = "Qwen/Qwen2.5-Math-1.5B"
BLOCK_ID = "qmath_block_8_19"
BLOCK = {"id": BLOCK_ID, "type": "block", "model": DONOR, "layers": [8, 19]}
AT = 7
GLUE_RANK = 64
STITCH = {"kind": "linear", "objective": "donor_lm", "steps": 1000, "lr": 1e-3, "batch_size": 8,
          "seq_len": 256, "seed": 0}
ALIGN_TEXTS = 1200
HEAL = {"steps": 300, "data": "gsm8k_train", "lr": 3e-4, "batch_size": 8, "seq_len": 384}
OUT_ROOT = "/vol/compiled/stitch_graft"
ARMS = ("host", "block_ridge", "block_stitch", "random_stitch", "lora")


def arm_root(arm):
    return os.path.join("/vol/eval_stitch", arm)


def program_for(arm, seed, budget=None):
    from wsp import Program

    program = Program(host=HOST, seed=seed)
    if arm == "block_ridge":
        program.graft(BLOCK_ID, at=AT, glue_rank=GLUE_RANK)
    elif arm == "block_stitch":
        program.graft(BLOCK_ID, at=AT, glue_rank=GLUE_RANK, stitch=dict(STITCH))
    elif arm == "random_stitch":
        program.graft(BLOCK_ID, at=AT, glue_rank=GLUE_RANK, stitch=dict(STITCH), control="random",
                      control_seed=seed + 100)
    elif arm == "lora":
        program.lora(budget_params=budget)
    else:
        raise ValueError("no program for arm " + arm)
    program.heal(**HEAL)
    return program


def compile_arm(arm, seed, budget=None):
    from wsp import Catalog

    out_dir = os.path.join(OUT_ROOT, arm + "_s" + str(seed))
    if os.path.exists(os.path.join(out_dir, "compile_report.json")):
        return out_dir
    heal_texts = splits.heal_texts("gsm8k")
    program = program_for(arm, seed, budget)
    program.compile(out_dir, Catalog([BLOCK], root="/"), align_texts=heal_texts[:ALIGN_TEXTS],
                    heal_texts=heal_texts, device="cuda", cache_dir="/vol/tensor_cache")
    return out_dir


def stitch_budget(seed):
    """Trainable glue params (stitch + w_out) of the block_stitch arm, for the matched LoRA."""
    path = os.path.join(OUT_ROOT, "block_stitch_s" + str(seed), "compile_report.json")
    with open(path) as f:
        return json.load(f)["grafts"][0]["glue_params"]


def compile_with_retry(arm, seed, budget=None, attempts=4):
    """compile_arm, retried when the Hugging Face range fetch drops the connection.

    Fetched tensors are cached in /vol/tensor_cache, so a retry resumes where it stopped.
    """
    import time

    import requests

    for attempt in range(attempts):
        try:
            return compile_arm(arm, seed, budget)
        except requests.exceptions.ConnectionError as error:
            if attempt == attempts - 1:
                raise
            print("[stitch_graft]", arm, "fetch failed, retrying:", repr(error)[:200], flush=True)
            time.sleep(30 * (attempt + 1))


def run_arm(arm, seed=0, budget=None):
    """Compile (if needed) and score one arm. Call with WSP_ROOT=arm_root(arm).

    The lora arm takes budget: the block_stitch arm's row["budget"].
    """
    if arm == "host":
        model_path = HOST
    elif arm == "lora":
        if budget is None:
            budget = stitch_budget(seed)
        model_path = compile_with_retry(arm, seed, budget=budget)
    else:
        model_path = compile_with_retry(arm, seed)
    result = harness.evaluate(model_path, "search", ["gsm8k"], namespace="stitch_graft", backend="hf",
                              note="stitch graft " + arm + " seed " + str(seed))
    row = {"arm": arm, "seed": seed, "model": model_path, "label": result["label"],
           "gsm8k": result["tasks"]["gsm8k"]["score"], "n": result["tasks"]["gsm8k"]["n"],
           "fit_r2": result.get("fit_r2")}
    report_path = os.path.join(str(model_path), "compile_report.json")
    if os.path.exists(report_path):
        with open(report_path) as f:
            report = json.load(f)
        grafts = report.get("grafts", [])
        if len(grafts) > 0:
            stitch = grafts[0].get("stitch")
            row["layers"] = grafts[0].get("layers")
            row["glue_params"] = grafts[0].get("glue_params")
            if stitch is not None:
                row["stitch_before"] = stitch.get("before")
                row["stitch_after"] = stitch.get("after")
        row["lora_params"] = report.get("lora_params")
        if arm == "block_stitch":
            # Match the LoRA to the graft's trainable params (stitch + w_out), as round 3
            # did for blocks. The frozen donor block itself is about 600M params.
            row["budget"] = grafts[0]["glue_params"]
    return row
