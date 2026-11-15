"""Write-in track moe-slice-2: a frozen slice of Qwen3-30B-A3B-Instruct-2507 inside Qwen3-1.7B.

Same recipe as modal_wi_xfam.py (docs/write_in_xfam.md), with a sparse donor. The donor is loaded straight from
its safetensors shards (wsp/moe_fast.py) and its expert layers run as batched matmuls. Design and results:
docs/write_in_moeslice2.md. Rows go to /vol/wi_moe2/v1/rows after each cell.

    modal run modal_wi_moeslice2.py --what prep
    modal run --detach modal_wi_moeslice2.py --what donor
    modal run --detach modal_wi_moeslice2.py --what group --cells graft:1 --smoke true
    modal run --detach modal_wi_moeslice2.py --what small --cells reference:0,fullft:1
    modal run --detach modal_wi_moeslice2.py --what native --cells qwen3_4b,qwen3_8b
"""

import glob
import json
import os
import random
import time

import modal

app = modal.App("wsp-wi-moeslice2")
volume = modal.Volume.from_name("wsp-models", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch==2.7.1", "transformers==4.57.1", "accelerate", "huggingface_hub", "requests",
        "safetensors", "zstandard", "numpy", "datasets", "sentencepiece", "protobuf",
    )
    .env({"HF_HOME": "/vol/hf", "HF_HUB_DISABLE_XET": "1", "HF_HUB_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1"})
    .add_local_python_source("wsp")
    .add_local_file("modal_graft.py", "/root/modal_graft.py")
    .add_local_file("modal_round5.py", "/root/modal_round5.py")
    .add_local_file("modal_round6.py", "/root/modal_round6.py")
    .add_local_file("modal_round7.py", "/root/modal_round7.py")
    .add_local_file("modal_round22pg.py", "/root/modal_round22pg.py")
    .add_local_file("modal_round23pg.py", "/root/modal_round23pg.py")
    .add_local_file("modal_wi_xfam.py", "/root/modal_wi_xfam.py")
)

VERSION = "v1"
DIR = f"/vol/wi_moe2/{VERSION}"
ROWS = f"{DIR}/rows"
HOST = "Qwen/Qwen3-1.7B"
DONOR = "Qwen/Qwen3-30B-A3B-Instruct-2507"
NATIVES = {"qwen3_4b": "Qwen/Qwen3-4B", "qwen3_8b": "Qwen/Qwen3-8B"}
D_HOST = D_DONOR = 2048
TOPK, NEXPERTS = 8, 128

BASE_CFG = {
    "arm": "graft", "seed": 0, "first": 16, "last": 32, "host_layer": 13,
    "steps": 4000, "batch": 16, "align_steps": 600, "kd_w": 1.0, "preserve": 1.0, "align_reg": 1.0,
    "enc_depth": 2, "dec_rank": 256, "lora_rank": 16, "lora_lr": 2e-4,
    "glue_lr": 3e-4, "gate_lr": 1e-4, "enc_lr": 1e-4, "ft_lr": 3e-6, "match_extra": True,
    "decode": True, "tag": "", "ckpt": False,
}
PRESETS = {
    "graft": {},
    "graft_ce": {"kd_w": 0.0},
    "identity": {"arm": "identity"},
    "scrambled": {"arm": "scrambled"},
    "lora_match": {"arm": "lora_match", "match_lora": 16, "lora_rank": 0},
    "lora_big": {"arm": "lora_big", "lora_rank": 128, "lora_lr": 5e-5},
    "fullft": {"arm": "fullft", "lora_rank": 0},
    "reference": {"arm": "reference", "lora_rank": 0},
}
DONOR_ARMS = ("graft", "scrambled")


def reload():
    try:
        volume.reload()
    except Exception as err:
        print("reload skipped:", repr(err)[:100])


def donor_snapshot():
    found = sorted(glob.glob("/vol/hf/hub/models--Qwen--Qwen3-30B-A3B-Instruct-2507/snapshots/*"))
    if not found:
        raise SystemExit("donor snapshot missing on the volume")
    return found[-1]


def model_snapshot(model_id):
    found = sorted(glob.glob("/vol/hf/hub/models--" + model_id.replace("/", "--") + "/snapshots/*"))
    return found[-1] if found else model_id


def letters_of(tok):
    return [tok(" " + c, add_special_tokens=False).input_ids[-1] for c in "ABCD"]


def build_evals():
    """MMLU dev 1000 and held-out 2000 (round 7 split), ARC-Challenge test, OpenBookQA test, as [prompt, gold] lists."""
    from datasets import load_dataset

    from modal_round6 import mmlu_prompt

    test = load_dataset("cais/mmlu", "all", split="test")
    order = list(range(len(test)))
    random.Random(0).shuffle(order)
    out = {}
    for name, span in (("mmlu_dev", order[:1000]), ("mmlu_held", order[1000:3000])):
        out[name] = [[mmlu_prompt(test[i]["question"], test[i]["choices"]), int(test[i]["answer"])] for i in span]
    for key, path, cfg, split, stem in (("arc_c", "allenai/ai2_arc", "ARC-Challenge", "test", "question"),
                                        ("obqa", "allenai/openbookqa", "main", "test", "question_stem")):
        items = []
        for r in load_dataset(path, cfg, split=split):
            labels, texts = list(r["choices"]["label"]), list(r["choices"]["text"])
            if len(texts) != 4 or r["answerKey"] not in labels:
                continue
            items.append([mmlu_prompt(r[stem], texts), labels.index(r["answerKey"])])
        out[key] = items
    return out


@app.function(image=image, volumes={"/vol": volume}, memory=32768, timeout=3 * 3600)
def prep():
    """Eval sets and the 44k pool (the xfam pool code, with exact eval prompts removed). No GPU."""
    import modal_wi_xfam as xf

    reload()
    t0 = time.time()
    os.makedirs(DIR, exist_ok=True)
    evals = build_evals()
    print("evals", {k: len(v) for k, v in evals.items()}, round(time.time() - t0))
    items, n_aux = xf.build_pool(evals)
    print("pool", len(items), "aux", n_aux, round(time.time() - t0))
    with open(f"{DIR}/evals.json", "w") as f:
        json.dump(evals, f)
    with open(f"{DIR}/pool.json", "w") as f:
        json.dump({"items": items, "n_aux": n_aux}, f)
    volume.commit()
    return {"evals": {k: len(v) for k, v in evals.items()}, "pool": len(items), "n_aux": n_aux,
            "seconds": round(time.time() - t0)}


def letter_probs(model, tok, items, batch=48):
    import torch

    ids = letters_of(tok)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    order = sorted(range(len(items)), key=lambda i: len(items[i][0]))
    out = [None] * len(items)
    for s in range(0, len(order), batch):
        idx = order[s:s + batch]
        enc = tok([items[i][0] for i in idx], return_tensors="pt", padding=True).to(model.device)
        with torch.no_grad():
            logits = model(**enc, use_cache=False, logits_to_keep=1).logits[:, -1, :].float()[:, ids]
        for i, p in zip(idx, torch.softmax(logits, -1).cpu().tolist()):
            out[i] = [round(x, 5) for x in p]
        if (s // batch) % 100 == 0:
            print("letter_probs", s, len(order), flush=True)
    tok.padding_side = "right"
    return out


def accuracy(probs, items):
    return sum(int(max(range(4), key=lambda j: p[j]) == g) for p, (_, g) in zip(probs, items)) / len(items)


def load_donor(log=print, layers=None):
    """Copy the needed shards to local disk, then load them onto the GPU. Returns (model, tok, timing)."""

    import torch
    import transformers

    from wsp import moe_fast

    snap = donor_snapshot()
    t0 = time.time()
    shards = moe_fast.shards_needed(snap, layers)
    local = "/tmp/donor"
    nbytes = moe_fast.copy_shards(snap, local, shards)
    t_copy = time.time() - t0
    log("copied", len(shards), "shards", round(nbytes / 1e9, 1), "GB in", round(t_copy), "s")
    tok = transformers.AutoTokenizer.from_pretrained(local)
    model, stats = moe_fast.load_qwen3_moe(local, "cuda", layers=layers, log=lambda *_: None)
    timing = {"copy_seconds": round(t_copy, 1), "load_seconds": stats["seconds"], "gb": round(stats["bytes"] / 1e9, 1),
              "shards": len(shards), "total_seconds": round(time.time() - t0, 1), "layers": None if layers is None else len(layers)}
    log("donor loaded", timing)
    return model, tok, timing


@app.function(image=image, volumes={"/vol": volume}, gpu="H200", cpu=8, memory=131072, timeout=6 * 3600)
def donor_job(slice_only_probe=True):
    """Load the donor, time the loads, score it on the eval sets, and cache its soft letter targets for the pool."""
    import torch

    from modal_round6 import score_mmlu
    from wsp import moe_fast

    reload()
    t0 = time.time()
    os.system("df -h /tmp | tail -1")
    out = {}
    if slice_only_probe:
        m, _, tm = load_donor(layers=range(16, 33))
        out["slice_only_load"] = tm
        del m
        torch.cuda.empty_cache()
    model, tok, tm = load_donor()
    out["full_load"] = tm
    with open(f"{DIR}/evals.json") as f:
        evals = json.load(f)
    with open(f"{DIR}/pool.json") as f:
        pool = json.load(f)
    run, stored = moe_fast.active_params(model.model.layers, TOPK, NEXPERTS)
    emb = model.model.embed_tokens.weight.numel() + model.lm_head.weight.numel() + model.model.norm.weight.numel()
    out["donor_params_total"] = stored + emb
    out["donor_params_per_token"] = run + emb
    t1 = time.time()
    out["acc"] = {}
    for k, v in evals.items():
        out["acc"][k] = accuracy(letter_probs(model, tok, v), v)
    print("donor accuracy", out["acc"], round(time.time() - t1))
    out["acc_score_mmlu_held"] = score_mmlu(model, tok, [tuple(x) for x in evals["mmlu_held"]], batch_size=16)
    t1 = time.time()
    soft = letter_probs(model, tok, pool["items"])
    out["pool_acc"] = accuracy(soft, pool["items"])
    out["soft_seconds"] = round(time.time() - t1)
    print("pool soft done", out["pool_acc"], out["soft_seconds"])
    with open(f"{DIR}/soft.json", "w") as f:
        json.dump(soft, f)
    out["gpu_hours"] = round((time.time() - t0) / 3600, 3)
    os.makedirs(ROWS, exist_ok=True)
    with open(f"{ROWS}/donor.json", "w") as f:
        json.dump(out, f)
    volume.commit()
    return out


def score(model, tok, items, bs=16):
    from modal_round6 import score_mmlu

    return score_mmlu(model, tok, [tuple(x) for x in items], batch_size=bs)


def decode_tps(model, tok, new=96):
    import torch

    try:
        tok.padding_side = "left"
        enc = tok("The history of the Roman Empire begins", return_tensors="pt").to(model.device)
        model.eval()
        with torch.no_grad():
            model.generate(**enc, max_new_tokens=8, do_sample=False)
            torch.cuda.synchronize()
            t = time.time()
            out = model.generate(**enc, max_new_tokens=new, do_sample=False, min_new_tokens=new)
            torch.cuda.synchronize()
        tok.padding_side = "right"
        return round((out.shape[1] - enc["input_ids"].shape[1]) / (time.time() - t), 2)
    except Exception as e:
        return f"error: {repr(e)[:160]}"


def finish(row, t0, smoke):
    row["gpu_hours"] = round((time.time() - t0) / 3600, 3)
    if not smoke:
        os.makedirs(ROWS, exist_ok=True)
        with open(f"{ROWS}/{row['name']}_{row['seed']}{row['config'].get('tag', '')}.json", "w") as f:
            json.dump(row, f)
        volume.commit()
    print(row["cell"], "ROW", json.dumps(row))
    return row


def row_budget(c, config):
    from wsp import pgraft
    from wsp.lora import lora_param_count

    total = (pgraft.count(pgraft.Enc(D_HOST, D_DONOR, depth=c["enc_depth"]).parameters())
             + pgraft.count(pgraft.Dec(D_DONOR, D_HOST, rank=c["dec_rank"]).parameters()) + D_HOST + 1)
    if c.get("match_lora", 0):
        total += lora_param_count(config, c["match_lora"])
    return total


def run_cell(name, seed, smoke=False, overrides=None, donor=None, donor_tok=None):
    import copy
    import gc

    import torch
    import transformers

    import modal_round22pg as r22
    from wsp import moe_fast, pgraft
    from wsp.lora import apply_lora, default_alpha, rank_for_budget

    cname = f"{name}:{seed}" + (":smoke" if smoke else "")
    t0 = time.time()
    c = {**BASE_CFG, **PRESETS[name], **(overrides or {})}
    c["seed"] = seed
    arm = c["arm"]
    torch.manual_seed(seed)
    reload()
    with open(f"{DIR}/pool.json") as f:
        pool = json.load(f)
    with open(f"{DIR}/evals.json") as f:
        evals = json.load(f)
    items = pool["items"]
    soft = None
    if os.path.exists(f"{DIR}/soft.json"):
        with open(f"{DIR}/soft.json") as f:
            soft = json.load(f)
    if soft is None:
        raise SystemExit("run the donor job first: soft.json is missing")
    if smoke:
        c["steps"], c["align_steps"] = 3, 3
        items, soft = items[:64], soft[:64]
        evals = {k: v[:32] for k, v in evals.items()}
    data = r22.build_data()
    if smoke:
        data = {k: v[:16] for k, v in data.items()}
    cfg = dict(pgraft.DEFAULTS, **r22.CFG)
    row = {"cell": cname, "name": name, "seed": seed, "config": c, "pool": len(items), "version": VERSION,
           "trained_on": "44k MC pool (MMLU auxiliary_train, CSQA, OBQA, QASC, LogiQA train; eval prompts removed) "
                         "plus general text for the drift KL and stage-1 text; eval items never trained"}
    device = torch.device("cuda")
    tok = transformers.AutoTokenizer.from_pretrained(model_snapshot(HOST))
    letters = letters_of(tok)
    dtype = torch.float32 if arm == "fullft" else torch.bfloat16
    base = transformers.AutoModelForCausalLM.from_pretrained(HOST, dtype=dtype).to("cuda").eval()
    for p in base.parameters():
        p.requires_grad_(arm == "fullft")
    ref = transformers.AutoModelForCausalLM.from_pretrained(HOST, dtype=torch.bfloat16).to("cuda").eval()
    for p in ref.parameters():
        p.requires_grad_(False)
    row["params_base"] = sum(p.numel() for p in base.parameters())

    if arm == "reference":
        row["scores"] = {k: score(base, tok, v) for k, v in evals.items()}
        row["decode_tps"] = decode_tps(base, tok)
        row["params_per_token"] = row["params_total"] = row["params_base"]
        print(cname, "SCORES base", json.dumps(row["scores"]), flush=True)
        return finish(row, t0, smoke)

    graft, grafts, lora_params = None, [], []
    needs_donor = arm in DONOR_ARMS
    if needs_donor:
        assert donor is not None
        moe_fast.install_fast_alignment()
        moe_fast.set_checkpoint(donor, c["ckpt"])

    joint_lora = c["lora_rank"] if arm in ("graft", "identity", "scrambled") else 0
    if joint_lora:
        lora_params = apply_lora(base, joint_lora, default_alpha(joint_lora))
        for p in lora_params:
            p.requires_grad_(True)

    if arm in ("graft", "identity", "scrambled"):
        control = "identity" if arm == "identity" else None
        if control is None:
            part = moe_fast.block_part(donor, c["first"], c["last"])
            graft = pgraft.build(base, donor, c["host_layer"], c["first"], c["last"], enc_depth=c["enc_depth"],
                                 dec_rank=c["dec_rank"], d_donor=D_DONOR, part=part)
        else:
            graft = pgraft.build(base, None, c["host_layer"], c["first"], c["last"], control="identity",
                                 enc_depth=c["enc_depth"], dec_rank=c["dec_rank"], d_donor=D_DONOR)
        grafts = [graft]
        pgraft.attach(base, grafts)
        if needs_donor:
            pgraft.set_sink(graft, donor, donor_tok, c["first"], data["align"][:256])
        glue = pgraft.trainable(grafts, enc=True)
        row["trainable_glue"] = pgraft.count(glue)
        if needs_donor:
            ta = time.time()
            pgraft.align(base, tok, donor, donor_tok, graft, data["align"], dict(cfg, align_steps=c["align_steps"]),
                         seed=seed, log_every=50)
            row["align_seconds"] = round(time.time() - ta, 1)
            row["alignment"] = pgraft.alignment_report(base, tok, donor, donor_tok, graft, data["align_eval"], cfg)
            print(cname, "alignment", row["alignment"], "align seconds", row["align_seconds"], flush=True)
        row["calib_rms"] = pgraft.calibrate_scale(base, tok, graft, data["align"], cfg)
        if arm == "scrambled":
            graft.part = copy.deepcopy(graft.part)
            row["scramble"] = moe_fast.scramble_part(graft.part, seed=seed + 100)
            print(cname, "scrambled", row["scramble"], flush=True)
        if control is None:
            run, stored = moe_fast.active_params(graft.part, TOPK, NEXPERTS)
            row["slice_params_per_token"], row["slice_params_stored"] = run, stored
        trainable_total = pgraft.count(glue) + pgraft.count(lora_params)
    elif arm in ("lora_match", "lora_big"):
        rank = rank_for_budget(base, row_budget(c, base.config)) if arm == "lora_match" else c["lora_rank"]
        lora_params = apply_lora(base, rank, default_alpha(rank))
        for p in lora_params:
            p.requires_grad_(True)
        row["lora_rank"] = rank
        trainable_total = pgraft.count(lora_params)
    else:
        trainable_total = sum(p.numel() for p in base.parameters())
    row["trainable"] = trainable_total

    if arm in ("graft", "identity", "scrambled"):
        gate = list(graft.token_gate.parameters())
        dec = list(graft.dec.parameters())
        enc = list(graft.enc.parameters())
        groups = [{"params": dec + (enc if arm == "identity" else []), "lr": c["glue_lr"]},
                  {"params": gate, "lr": c["gate_lr"]}]
        if arm != "identity":
            groups.append({"params": enc, "lr": c["enc_lr"]})
        if lora_params:
            groups.append({"params": lora_params, "lr": c["lora_lr"]})
    elif arm == "fullft":
        groups = [{"params": list(base.parameters()), "lr": c["ft_lr"]}]
    else:
        groups = [{"params": lora_params, "lr": c["lora_lr"]}]

    steps = c["steps"]
    if arm != "graft" and c["match_extra"]:
        steps += c["align_steps"]
    row["train_steps"] = steps
    rng = random.Random(seed + 5)
    order = list(range(len(items)))
    rng.shuffle(order)
    ptr = [0]

    def next_idx():
        if ptr[0] + c["batch"] > len(order):
            rng.shuffle(order)
            ptr[0] = 0
        idx = order[ptr[0]:ptr[0] + c["batch"]]
        ptr[0] += c["batch"]
        return idx

    gen_feed = pgraft._batches(data["general"], 4, seed + 41)
    align_feed = pgraft._batches(data["align"], cfg["align_batch"], seed + 11) if needs_donor and arm == "graft" else None
    tokens = [0]

    def step_fn(step):
        idx = next_idx()
        batch = [tuple(items[i]) for i in idx]
        sp = torch.tensor([soft[i] for i in idx])
        loss, info = pgraft.mc_soft_loss(base, tok, batch, letters, device, soft=sp, kd_weight=c["kd_w"])
        tokens[0] += info.pop("tokens")
        if c["preserve"] > 0:
            enc = pgraft._tok_batch(tok, next(gen_feed), cfg["seq_len"], device)
            _, logits = pgraft.seq_lm_loss(base, enc, return_logits=True)
            drift = pgraft.drift_kl(ref, enc, logits)
            loss = loss + c["preserve"] * drift
            info["drift"] = float(drift.detach())
        if arm == "graft" and c["align_reg"] > 0 and step % 2 == 0:
            rl, _ = pgraft.functional_loss(base, tok, donor, donor_tok, graft, next(align_feed), cfg)
            loss = loss + c["align_reg"] * rl
        if graft is not None:
            gm = pgraft.mean_gate(grafts)
            if gm is not None:
                info["gate"] = float(gm.detach())
        return loss, info

    tt = time.time()
    log = pgraft.train(base, groups, steps, step_fn, seed=seed + 1, log_every=50 if not smoke else 1)
    row["train_step_seconds"] = round((time.time() - tt) / max(1, steps), 3)
    row["tokens_trained"] = tokens[0]
    row["gate_curve"] = [[r["step"], round(r["gate"], 4)] for r in log if "gate" in r and r["step"] % 100 == 0]
    row["loss_first50"] = sum(r["loss"] for r in log[:50]) / max(1, len(log[:50]))
    row["loss_last50"] = sum(r["loss"] for r in log[-50:]) / max(1, len(log[-50:]))
    print(cname, "train done", row["loss_first50"], "->", row["loss_last50"], "step s", row["train_step_seconds"], flush=True)
    row["train_seconds"] = round(time.time() - t0, 1)

    if arm == "fullft":
        base = base.to(torch.bfloat16)
    base.eval()
    row["scores"] = {k: score(base, tok, v) for k, v in evals.items()}
    print(cname, "SCORES", json.dumps(row["scores"]), flush=True)
    if grafts and arm != "identity":
        with pgraft.grafts_off(base):
            row["scores_graft_off"] = {k: score(base, tok, evals[k]) for k in ("mmlu_dev", "mmlu_held")}
    per_token = row["params_base"]
    if grafts and arm != "identity":
        per_token += row["slice_params_per_token"] + row["trainable_glue"]
    elif grafts:
        per_token += row["trainable_glue"]
    row["params_per_token"] = per_token
    row["params_total"] = per_token - row.get("slice_params_per_token", 0) + row.get("slice_params_stored", 0)
    if c["decode"] and not smoke:
        row["decode_tps"] = decode_tps(base, tok)
    del base, ref, graft, grafts
    gc.collect()
    torch.cuda.empty_cache()
    return finish(row, t0, smoke)


@app.function(image=image, volumes={"/vol": volume}, gpu="H100", cpu=8, memory=131072, timeout=5 * 3600)
def small_cell(name, seed, smoke=False, overrides=None):
    return run_cell(name, seed, smoke, overrides)


@app.function(image=image, volumes={"/vol": volume}, gpu="H200", cpu=8, memory=131072, timeout=8 * 3600)
def donor_cells(cells, smoke=False, overrides=None):
    """Load the donor once, then run each (name, seed) in order."""
    t0 = time.time()
    donor, dtok, timing = load_donor()
    print("DONOR TIMING", timing, flush=True)
    rows = []
    for name, seed in cells:
        try:
            row = run_cell(name, seed, smoke, overrides, donor=donor, donor_tok=dtok)
            row["donor_load"] = timing
            rows.append(row)
        except Exception as e:
            import traceback

            traceback.print_exc()
            rows.append({"cell": f"{name}:{seed}", "error": repr(e)[:500]})
    print("CONTAINER HOURS", round((time.time() - t0) / 3600, 3))
    return rows


@app.function(image=image, volumes={"/vol": volume}, gpu="H100", cpu=8, memory=131072, timeout=3 * 3600)
def native_cell(key):
    import torch
    import transformers

    t0 = time.time()
    reload()
    with open(f"{DIR}/evals.json") as f:
        evals = json.load(f)
    model_id = NATIVES[key]
    tok = transformers.AutoTokenizer.from_pretrained(model_snapshot(model_id))
    model = transformers.AutoModelForCausalLM.from_pretrained(model_snapshot(model_id), dtype=torch.bfloat16).to("cuda").eval()
    n = sum(p.numel() for p in model.parameters())
    row = {"cell": f"native:{key}", "name": f"native_{key}", "seed": 0, "config": {"tag": ""}, "model": model_id,
           "params_total": n, "params_per_token": n, "trainable": 0}
    row["scores"] = {k: score(model, tok, v) for k, v in evals.items()}
    row["decode_tps"] = decode_tps(model, tok)
    print(row["cell"], "SCORES", json.dumps(row["scores"]), flush=True)
    return finish(row, t0, False)


@app.local_entrypoint()
def main(what: str = "", cells: str = "", smoke: bool = False, overrides: str = ""):
    ov = json.loads(overrides) if overrides else None
    pairs = []
    for item in [x for x in cells.split(",") if x]:
        if ":" in item:
            name, seed = item.split(":")
            if name not in PRESETS:
                raise SystemExit(f"unknown preset {name}")
            pairs.append((name, int(seed)))
    if what == "prep":
        print(json.dumps(prep.remote()))
    elif what == "donor":
        print(json.dumps(donor_job.remote()))
    elif what == "group":
        print(json.dumps(donor_cells.remote(pairs, smoke, ov), indent=1)[:6000])
    elif what == "small":
        handles = [small_cell.spawn(n, s, smoke, ov) for n, s in pairs]
        for h in handles:
            try:
                r = h.get()
                print("done", r["cell"], json.dumps(r.get("scores")))
            except Exception as e:
                print("cell failed:", repr(e)[:800])
    elif what == "native":
        handles = [native_cell.spawn(k) for k in cells.split(",") if k]
        for h in handles:
            print(json.dumps(h.get().get("scores")))
    else:
        raise SystemExit("--what prep|donor|group|small|native")
