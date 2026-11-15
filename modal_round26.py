"""Round 26: rotation test for far-lineage specialists (Experiment 4 of docs/research/2026-11-15-transfer-sweep.md).

Question. Qwen2.5-Math-1.5B-Instruct and Qwen2.5-Coder-1.5B-Instruct may keep
the base's function in a rotated residual coordinate system. Their weight
cosine to Qwen2.5-1.5B-Instruct is 0.2 to 0.5, and run 16c gave chance for
their MLPs and layers under the base's embeddings. If one orthogonal map R
(Procrustes on the token embeddings) restores the 16c cells, the specialists
can be mixed with no training.

CPU CHECK RESULT (2026-11-15, docs/rotate.md): the CPU check already says
STOP. Relative Procrustes residual of the token embeddings against
Qwen2.5-1.5B-Instruct, rotated / unrotated / token-shuffled:
  Math   0.918 / 1.078 / 1.042   (above the 0.8 stop threshold)
  Coder  0.728 / 0.920 / 1.083   (between 0.3 and 0.8: not a go, not a stop)
A residual near 1 means the rotated embeddings explain almost none of the
base's. This script is provided so the plan's cells exist, but do not spend
GPU time on the full set. At most run --smoke (one cell) to confirm the
CPU call. The embedding residual is also recomputed inside every cell and
written to its row (field "residual").

Method. Fold every RMSNorm weight into the readers (wsp/rotate.py), for the
base and the specialist, so every norm has weight 1. Rotate the specialist by
R. Then copy parts into the folded base. No training, no benchmark data.

Cells (names are printed on every line):
  ref_base, ref_math, ref_coder          references, each model alone
  mlp_<spec>_<R>                         all MLPs (gate, up, down) of the specialist, rotated
                                         by R, in the folded base. Base attention, embeddings, head.
  layer_<spec>_<R>                       all decoder layers of the specialist, rotated by R, on the
                                         folded base's embeddings, final norm, and head. Uses the
                                         specialist's own rope_theta (Math 10000), as run 16c did.
  <spec> is math or coder. <R> is one of:
    fit       Procrustes R between the specialist's and the base's token embeddings
              (rows 0 to 151642, the real tokens)
    none      R = identity (same folded pipeline, unrotated; the 16c control. Run 16c itself
              used unfolded weights under the base's norms)
    random    a random orthogonal matrix (seed 0)
    shuffled  Procrustes R fit after permuting the specialist's token rows (seed 0)

Go rule (plan): rotated residual below 0.3 and mlp_math_fit scores at least
15 points above chance on GSM8K (or 10 points above the 25 chance rate on
MMLU) while mlp_math_random stays at chance. Stop rule: residual above 0.8.

Scores: wsp.skills_eval on GSM8K 300, MBPP 200 (seeded 200 of 257), MMLU 1000.
One seed; nothing is random except the control matrices, whose seed is fixed.
That is a limit: a gain of a few points is not a result. Only a jump from chance
counts.

The first cell to run is mlp_math_fit. --smoke runs only that cell, with
GSM8K 30, MBPP 20, MMLU 100.

Rows are written one per cell to /vol/results/round26/<cell>.json and, as each
cell finishes, to results/round26_rows.jsonl on the local machine.

    modal run --detach modal_round26.py --smoke
    modal run --detach modal_round26.py --shard 0 --nshards 1
"""

import json
import os
import random
import time

import modal

app = modal.App("wsp-round26")
volume = modal.Volume.from_name("wsp-models", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch==2.7.1", "transformers==4.57.1", "accelerate", "huggingface_hub", "requests",
        "safetensors", "zstandard", "numpy", "datasets", "sentencepiece", "protobuf",
    )
    .env({"HF_HOME": "/vol/hf", "HF_HUB_DISABLE_XET": "1"})
    .add_local_python_source("wsp")
)

BASE = "Qwen/Qwen2.5-1.5B-Instruct"
SPECIALIST = {"math": "Qwen/Qwen2.5-Math-1.5B-Instruct", "coder": "Qwen/Qwen2.5-Coder-1.5B-Instruct"}
REAL_TOKENS = 151643
SKILLS = ["gsm8k", "mbpp", "mmlu"]
COUNTS = {"gsm8k": 300, "mbpp": 200, "mmlu": 1000}
SMOKE_COUNTS = {"gsm8k": 30, "mbpp": 20, "mmlu": 100}
R_KINDS = ["fit", "none", "random", "shuffled"]
SMOKE_CELL = "mlp_math_fit"


def cell_names():
    out = ["ref_base", "ref_math", "ref_coder"]
    for kind in ["mlp", "layer"]:
        for spec in ["math", "coder"]:
            for r in R_KINDS:
                out.append(f"{kind}_{spec}_{r}")
    # The go-rule cell and its control first, so a stop shows early.
    first = [SMOKE_CELL, "mlp_math_random", "mlp_math_none"]
    return first + [name for name in out if name not in first]


def skills_eval():
    from wsp import skills_eval as module

    return module


def eval_items(ev, counts):
    items = {
        "gsm8k": ev.gsm8k_items(count=counts["gsm8k"], seed=0),
        "mbpp": ev.mbpp_items(),
        "mmlu": ev.mmlu_items(count=counts["mmlu"]),
    }
    order = list(range(len(items["mbpp"])))
    random.Random(0).shuffle(order)
    picked = []
    for index in sorted(order[:counts["mbpp"]]):
        picked.append(items["mbpp"][index])
    items["mbpp"] = picked
    return items


def load(name):
    import torch
    import transformers

    return transformers.AutoModelForCausalLM.from_pretrained(name, dtype=torch.float32).eval()


def pick_rotation(kind, spec_model, base_model):
    """Return (R, residual report) for one R kind. R is float64 (d x d)."""
    import torch

    from wsp.rotate import procrustes, random_orthogonal, residual_report, shuffled_procrustes

    E_m = spec_model.model.embed_tokens.weight.detach()[:REAL_TOKENS]
    E_i = base_model.model.embed_tokens.weight.detach()[:REAL_TOKENS]
    fit = procrustes(E_m, E_i)
    report = residual_report(E_m, E_i, fit)
    d = E_m.shape[1]
    if kind == "fit":
        R = fit
    elif kind == "none":
        R = torch.eye(d, dtype=torch.float64)
    elif kind == "random":
        R = random_orthogonal(d, seed=0)
    elif kind == "shuffled":
        R, _ = shuffled_procrustes(E_m, E_i, seed=0)
    else:
        raise ValueError(kind)
    return R, report


def build(cell):
    """Return (model on CPU in bfloat16, extra row fields) for one cell."""
    import torch

    from wsp.rotate import fold_norms, rotate_residual

    if cell.startswith("ref_"):
        name = {"base": BASE, "math": SPECIALIST["math"], "coder": SPECIALIST["coder"]}[cell[4:]]
        return load(name).to(torch.bfloat16), {}
    kind, spec, r_kind = cell.split("_")
    base = load(BASE)
    specialist = load(SPECIALIST[spec])
    R, report = pick_rotation(r_kind, specialist, base)
    fold_norms(base)
    fold_norms(specialist)
    rotate_residual(specialist, R)
    extra = {"residual": report, "r_kind": r_kind, "spec": spec, "kind": kind}
    if kind == "mlp":
        for index, layer in enumerate(base.model.layers):
            donor = specialist.model.layers[index].mlp
            for name in ["gate_proj", "up_proj", "down_proj"]:
                getattr(layer.mlp, name).weight.data = getattr(donor, name).weight.data.clone()
        model = base
    else:
        model = specialist
        model.model.embed_tokens.weight.data = base.model.embed_tokens.weight.data.clone()
        model.lm_head.weight.data = base.lm_head.weight.data.clone()
        model.model.norm.weight.data = base.model.norm.weight.data.clone()
        extra["rope_theta"] = float(model.config.rope_theta)
    return model.to(torch.bfloat16), extra


@app.function(image=image, volumes={"/vol": volume}, gpu="H100", memory=65536, timeout=3 * 3600)
def run_cell(cell, smoke=False):
    import torch
    import transformers

    start = time.time()
    ev = skills_eval()
    counts = SMOKE_COUNTS if smoke else COUNTS
    row = {"cell": cell, "base": BASE, "eval": "wsp.skills_eval", "counts": counts, "seed": 0, "smoke": smoke}
    tokenizer = transformers.AutoTokenizer.from_pretrained(BASE)
    model, extra = build(cell)
    row.update(extra)
    print(f"[{cell}] built, residual {extra.get('residual')}", flush=True)
    model = model.to("cuda").eval()
    items = eval_items(ev, counts)
    first = items["gsm8k"][0]
    row["gsm8k_prompt_sample"] = ev.render_prompt(tokenizer, ev.user_text("gsm8k", first))
    result = ev.evaluate_suite(model, tokenizer, skills=SKILLS, items=items)
    row["scores"] = dict(result["scores"])
    row["macro"] = result["suite"]
    row["seconds"] = round(time.time() - start, 1)
    print(f"[{cell}] scores {json.dumps(row['scores'])}", flush=True)
    os.makedirs("/vol/results/round26", exist_ok=True)
    full = dict(row)
    full["skills"] = result["skills"]
    with open(f"/vol/results/round26/{cell}{'_smoke' if smoke else ''}.json", "w") as f:
        json.dump(full, f, default=str)
    volume.commit()
    return row


@app.local_entrypoint()
def main(shard: int = 0, nshards: int = 1, smoke: bool = False, only: str = ""):
    if smoke:
        todo = [SMOKE_CELL]
    else:
        todo = cell_names()
        if only:
            todo = [name for name in todo if name in only.split(",")]
    mine = [name for index, name in enumerate(todo) if index % nshards == shard]
    print(f"shard {shard}/{nshards}: {mine}", flush=True)
    os.makedirs("results", exist_ok=True)
    args = [(name, smoke) for name in mine]
    for row in run_cell.starmap(args, return_exceptions=True, order_outputs=False):
        if isinstance(row, Exception):
            print("cell failed:", repr(row)[:600], flush=True)
            continue
        print(f"[{row['cell']}] row {json.dumps(row['scores'])}", flush=True)
        with open("results/round26_rows.jsonl", "a") as f:
            f.write(json.dumps(row, default=str) + "\n")
