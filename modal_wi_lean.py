"""Write-in track lean-graft: cheaper donor spans and native references at equal per-token cost.

Design and results: docs/write_in_lean.md. Reuses the cell of modal_wi_xfam.py (same pool, same prompts, same
losses, same selection rule) and changes only what a job spec names: the host model, the donor span, a sparse
set of kept donor layers, and the output directory.

    modal run modal_wi_lean.py --jobs-file jobs/smoke.json
    modal run --detach modal_wi_lean.py --jobs-file jobs/screen.json
    modal run --detach modal_wi_lean.py --bench true

A job is {"name": preset of modal_wi_xfam, "seed": int, "host": model id, "overrides": {...}, "keep": [donor layers]}.
Rows go to /vol/wi_lean/rows on the volume as each job ends.
"""

import json
import os
import statistics
import time

import modal

import modal_wi_xfam as X

app = modal.App("wsp-wi-lean")
volume = modal.Volume.from_name("wsp-models", create_if_missing=True)
image = X.image.add_local_file("modal_wi_xfam.py", "/root/modal_wi_xfam.py")

LEAN_DIR = "/vol/wi_lean"
ROWS = f"{LEAN_DIR}/rows"
LLAMA_1B = "unsloth/Llama-3.2-1B-Instruct"
NATIVE = {
    "llama1b": LLAMA_1B,
    "llama3b": "unsloth/Llama-3.2-3B-Instruct",
    "qwen1.5b": "Qwen/Qwen2.5-1.5B-Instruct",
    "qwen3b": "Qwen/Qwen2.5-3B-Instruct",
}


def run_job(spec):
    import modal_wi_xfam as xf
    from wsp import pgraft

    xf.HOST = spec.get("host", LLAMA_1B)
    xf.ROWS = ROWS
    xf.volume = volume
    overrides = dict(spec.get("overrides") or {})
    overrides.setdefault("decode", False)
    original = pgraft.build
    keep = spec.get("keep")
    if keep:
        pgraft.build = lambda *a, **k: original(*a, keep=keep, **k)
    try:
        row = xf._cell(spec["name"], int(spec["seed"]), bool(spec.get("smoke")), overrides)
    finally:
        pgraft.build = original
    row["host"] = xf.HOST
    row["keep"] = keep
    return row


@app.function(image=image, volumes={LEAN_DIR.rsplit("/", 1)[0]: volume}, gpu="H200", memory=131072, timeout=6 * 3600, max_containers=3)
def job_h200(spec):
    return run_job(spec)


def _timed_generate(model, tok, batch, prompt_tokens, new, repeats):
    import torch

    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    text = ("The history of the Roman Empire begins with the founding of the city and its early kings. " * 12)
    ids = tok(text, return_tensors="pt").input_ids[:, :prompt_tokens]
    enc = {"input_ids": ids.repeat(batch, 1).to("cuda"), "attention_mask": torch.ones(batch, ids.shape[1], dtype=torch.long, device="cuda")}
    times = []
    with torch.no_grad():
        model.generate(**enc, max_new_tokens=8, min_new_tokens=8, do_sample=False, pad_token_id=tok.pad_token_id)
        for _ in range(repeats):
            torch.cuda.synchronize()
            t = time.time()
            model.generate(**enc, max_new_tokens=new, min_new_tokens=new, do_sample=False, pad_token_id=tok.pad_token_id)
            torch.cuda.synchronize()
            times.append(time.time() - t)
    sec = statistics.median(times)
    return {"batch": batch, "seconds": round(sec, 3), "tok_s_per_seq": round(new / sec, 2), "tok_s_total": round(batch * new / sec, 2)}


@app.function(image=image, volumes={LEAN_DIR.rsplit("/", 1)[0]: volume}, gpu="H200", memory=131072, timeout=3 * 3600)
def bench(only=None):
    """Decode speed at batch 1 and 8 on one GPU type: native models, the donor, and grafts of several spans."""
    import gc

    import torch
    import transformers

    import modal_round22pg as r22
    from wsp import pgraft
    from wsp.graft import disable_cache

    try:
        volume.reload()
    except Exception as e:
        print("reload skipped", repr(e)[:80])
    prompt_tokens, new, repeats = 128, 128, 3
    data = r22.build_data()
    rows = []

    def measure(label, model, tok, extra=None):
        model.eval()
        entry = {"label": label, "gpu": torch.cuda.get_device_name(0), "prompt_tokens": prompt_tokens, "new_tokens": new,
                 "params": sum(p.numel() for p in model.parameters())}
        entry.update(extra or {})
        entry["bs1"] = _timed_generate(model, tok, 1, prompt_tokens, new, repeats)
        entry["bs8"] = _timed_generate(model, tok, 8, prompt_tokens, new, repeats)
        print("BENCH", json.dumps(entry))
        rows.append(entry)

    def load(name):
        tok = transformers.AutoTokenizer.from_pretrained(name)
        model = transformers.AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16).to("cuda").eval()
        return model, tok

    models = dict(NATIVE)
    models["qwen7b_donor"] = X.DONOR
    for label, name in models.items():
        if only and label not in only:
            continue
        model, tok = load(name)
        measure(label, model, tok)
        del model
        gc.collect()
        torch.cuda.empty_cache()

    donor = donor_tok = None
    graft_configs = {
        "graft_4_20": (4, 20, None), "graft_4_7": (4, 7, None), "graft_4_11": (4, 11, None), "graft_4_15": (4, 15, None),
        "graft_sparse_4_20": (4, 20, list(range(4, 21, 2))),
    }
    for label, (first, last, keep) in graft_configs.items():
        if only and label not in only:
            continue
        if donor is None:
            donor, donor_tok = load(X.DONOR)
            for p in donor.parameters():
                p.requires_grad_(False)
        base, tok = load(LLAMA_1B)
        graft = pgraft.build(base, donor, 7, first, last, enc_depth=2, dec_rank=256, d_donor=X.D_DONOR, keep=keep)
        pgraft.attach(base, [graft])
        pgraft.set_sink(graft, donor, donor_tok, first, data["align"][:64])
        # untrained glue: the speed does not depend on the weights
        slice_params = sum(p.numel() for p in graft.part.parameters())
        per_token = sum(p.numel() for p in base.parameters()) + slice_params
        measure(label, base, tok, {"slice_params": slice_params, "params_per_token": per_token})
        if label == "graft_4_20":
            disable_cache(base)
            entry = {"label": label + "_nocache", "gpu": torch.cuda.get_device_name(0), "prompt_tokens": prompt_tokens,
                     "new_tokens": 32, "params_per_token": per_token}
            base.generation_config.use_cache = False
            entry["bs1"] = _timed_generate(base, tok, 1, prompt_tokens, 32, 2)
            print("BENCH", json.dumps(entry))
            rows.append(entry)
        del base, graft
        gc.collect()
        torch.cuda.empty_cache()
    os.makedirs(LEAN_DIR, exist_ok=True)
    with open(f"{LEAN_DIR}/bench{'_' + '_'.join(only) if only else ''}.json", "w") as f:
        json.dump(rows, f)
    volume.commit()
    return rows


@app.local_entrypoint()
def main(jobs_file: str = "", bench_run: bool = False, only: str = ""):
    if bench_run:
        rows = bench.remote([x for x in only.split(",") if x] or None)
        os.makedirs("results/wi_lean", exist_ok=True)
        with open("results/wi_lean/bench.json", "w") as f:
            json.dump(rows, f, indent=1)
        return
    if not jobs_file:
        raise SystemExit("0 jobs: pass --jobs-file path.json")
    with open(jobs_file) as f:
        specs = json.load(f)
    if not specs:
        raise SystemExit("0 jobs in " + jobs_file)
    handles = [job_h200.spawn(s) for s in specs]
    out = []
    for h in handles:
        try:
            out.append(h.get())
        except Exception as e:
            print("job failed:", repr(e)[:1000])
    print("finished", len(out), "of", len(specs))
