"""Write-in track xfamily-knowledge: Qwen2.5-7B-Instruct knowledge into Llama-3.2-1B-Instruct through a parallel graft.

Design and results: docs/write_in_xfam.md. Everything trains on the same item pool (MMLU auxiliary_train,
CommonsenseQA, OpenBookQA, QASC, LogiQA train) with the answer-letter loss plus, for the pool, the donor's cached
soft letter distribution. Scored on MMLU dev 1000 and held-out 2000 (test items no arm trains on), ARC-Challenge
test, OpenBookQA test.

    modal run modal_wi_xfam.py --prep true            # build the pool, donor soft labels, eval sets, donor scores
    modal run --detach modal_wi_xfam.py --cells graft_main:0 --smoke true
    modal run --detach modal_wi_xfam.py --cells graft_main:0,graft_main:1
"""

import json
import os
import random
import time

import modal

app = modal.App("wsp-wi-xfam")
volume = modal.Volume.from_name("wsp-models", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch==2.7.1", "transformers==4.57.1", "accelerate", "huggingface_hub", "requests",
        "safetensors", "zstandard", "numpy", "datasets", "sentencepiece", "protobuf",
    )
    .env({"HF_HOME": "/vol/hf"})
    .add_local_python_source("wsp")
    .add_local_file("modal_graft.py", "/root/modal_graft.py")
    .add_local_file("modal_round5.py", "/root/modal_round5.py")
    .add_local_file("modal_round6.py", "/root/modal_round6.py")
    .add_local_file("modal_round7.py", "/root/modal_round7.py")
    .add_local_file("modal_round22pg.py", "/root/modal_round22pg.py")
    .add_local_file("modal_round23pg.py", "/root/modal_round23pg.py")
)

VERSION = "v1"
DIR = f"/vol/wi_xfam/{VERSION}"
ROWS = f"{DIR}/rows"
HOST = "unsloth/Llama-3.2-1B-Instruct"
DONOR = "Qwen/Qwen2.5-7B-Instruct"
D_HOST, D_DONOR = 2048, 3584
HOST_LAYER = 7
AUX_COUNT = 45000

BASE_CFG = {
    "arm": "graft", "seed": 0, "first": 4, "last": 20, "host_layer": 7,
    "steps": 2500, "batch": 16, "align_steps": 600, "kd_w": 1.0, "preserve": 1.0, "align_reg": 1.0,
    "enc_depth": 2, "dec_rank": 256, "lora_rank": 0, "lora_lr": 2e-4,
    "glue_lr": 3e-4, "gate_lr": 1e-4, "enc_lr": 1e-4, "ft_lr": 1e-5, "match_extra": True,
    "two_points": None, "decode": True, "tag": "",
}
# A preset is a diff over BASE_CFG. Dev-slice pilots use short step counts.
PRESETS = {
    "graft_main": {},
    "graft_ce": {"kd_w": 0.0},
    "graft_wide": {"first": 2, "last": 24},
    "graft_lora": {"lora_rank": 16},
    "graft_wide_lora": {"first": 2, "last": 24, "lora_rank": 16},
    "graft_noreg": {"align_reg": 0.0},
    "graft_two": {"two_points": {"host_layer": 12, "first": 20, "last": 28}},
    "identity": {"arm": "identity"},
    "scrambled": {"arm": "scrambled"},
    "lora_match": {"arm": "lora_match"},
    "lora_big": {"arm": "lora_big", "lora_rank": 128, "lora_lr": 1e-4},
    "fullft": {"arm": "fullft"},
    "fullft_lr2": {"arm": "fullft", "ft_lr": 3e-6},
    "fullft_lr3": {"arm": "fullft", "ft_lr": 3e-5},
    "lora_big_lr2": {"arm": "lora_big", "lora_rank": 128, "lora_lr": 3e-4},
    "reference": {"arm": "reference"},
}


def letters_of(tok):
    return [tok(" " + c, add_special_tokens=False).input_ids[-1] for c in "ABCD"]


def build_eval_sets():
    """MMLU dev 1000, held-out 2000 (test), ARC-Challenge test, OpenBookQA test, as (prompt, gold) lists."""
    from datasets import load_dataset

    from modal_round6 import mmlu_prompt
    from modal_round7 import mmlu_split

    import transformers
    tok = transformers.AutoTokenizer.from_pretrained(HOST)
    _, dev = mmlu_split(tok, heldout=False)
    _, held = mmlu_split(tok, heldout=True)
    out = {"mmlu_dev": [list(x) for x in dev], "mmlu_held": [list(x) for x in held]}
    arc = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test")
    items = []
    for r in arc:
        labels, texts = list(r["choices"]["label"]), list(r["choices"]["text"])
        if len(texts) != 4 or r["answerKey"] not in labels:
            continue
        items.append([mmlu_prompt(r["question"], texts), labels.index(r["answerKey"])])
    out["arc_c"] = items
    obqa = load_dataset("allenai/openbookqa", "main", split="test")
    items = []
    for r in obqa:
        labels, texts = list(r["choices"]["label"]), list(r["choices"]["text"])
        if len(texts) != 4 or r["answerKey"] not in labels:
            continue
        items.append([mmlu_prompt(r["question_stem"], texts), labels.index(r["answerKey"])])
    out["obqa"] = items
    return out


def build_pool(evals):
    """Training pool: MMLU auxiliary_train items plus the round 23 MC pools; minus any prompt in an eval set."""
    from datasets import load_dataset

    import modal_round23pg as r23
    from modal_round6 import mmlu_prompt

    banned = {p for k in evals for p, _ in evals[k]}
    rng = random.Random(7)
    aux = load_dataset("cais/mmlu", "auxiliary_train", split="train")
    order = list(range(len(aux)))
    rng.shuffle(order)
    items, seen = [], set()
    for index in order:
        row = aux[index]
        if "train" in row and isinstance(row["train"], dict):
            row = row["train"]
        if len(row["choices"]) != 4:
            continue
        prompt = mmlu_prompt(row["question"], row["choices"])
        if len(prompt) > 1400 or prompt in banned or prompt in seen:
            continue
        seen.add(prompt)
        items.append([prompt, int(row["answer"])])
        if len(items) >= AUX_COUNT:
            break
    n_aux = len(items)
    mc = r23.mc_items()
    for prompt, gold in mc["train"] + mc["eval"]:
        if prompt in banned or prompt in seen:
            continue
        seen.add(prompt)
        items.append([prompt, int(gold)])
    rng.shuffle(items)
    return items, n_aux


def letter_probs(model, tok, items, batch=32):
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
            logits = model(**enc).logits[:, -1, :].float()[:, ids]
        probs = torch.softmax(logits, -1).cpu().tolist()
        for i, p in zip(idx, probs):
            out[i] = [round(x, 5) for x in p]
    return out


@app.function(image=image, volumes={"/vol": volume}, gpu="H100", memory=131072, timeout=3 * 3600)
def prep():
    import torch
    import transformers

    t0 = time.time()
    os.makedirs(DIR, exist_ok=True)
    evals = build_eval_sets()
    print("eval sets", {k: len(v) for k, v in evals.items()})
    items, n_aux = build_pool(evals)
    print("pool", len(items), "aux", n_aux)
    dtok = transformers.AutoTokenizer.from_pretrained(DONOR)
    donor = transformers.AutoModelForCausalLM.from_pretrained(DONOR, dtype=torch.bfloat16).to("cuda").eval()
    soft = letter_probs(donor, dtok, items)
    donor_acc = {}
    for k, v in evals.items():
        p = letter_probs(donor, dtok, v)
        donor_acc[k] = sum(int(max(range(4), key=lambda j: row[j]) == g) for row, (_, g) in zip(p, v)) / len(v)
    train_acc = sum(int(max(range(4), key=lambda j: row[j]) == g) for row, (_, g) in zip(soft, items)) / len(items)
    print("donor eval accuracy", donor_acc, "on the train pool", round(train_acc, 4))
    with open(f"{DIR}/evals.json", "w") as f:
        json.dump(evals, f)
    with open(f"{DIR}/pool.json", "w") as f:
        json.dump({"items": items, "soft": soft, "n_aux": n_aux, "donor_train_acc": train_acc}, f)
    with open(f"{DIR}/donor_acc.json", "w") as f:
        json.dump(donor_acc, f)
    volume.commit()
    return {"pool": len(items), "donor_acc": donor_acc, "donor_train_acc": train_acc,
            "seconds": round(time.time() - t0)}


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
        return round((out.shape[1] - enc["input_ids"].shape[1]) / (time.time() - t), 2)
    except Exception as e:  # recorded, not fatal
        return f"error: {repr(e)[:120]}"


@app.function(image=image, volumes={"/vol": volume}, gpu="H100", memory=131072, timeout=5 * 3600)
def cell(name, seed, smoke=False, overrides=None):
    import gc

    import torch
    import transformers

    import modal_round22pg as r22
    from wsp import pgraft, scramble
    from wsp.lora import apply_lora, default_alpha, rank_for_budget

    cname = f"{name}:{seed}" + (":smoke" if smoke else "")
    t0 = time.time()
    c = dict(BASE_CFG, **PRESETS[name], **(overrides or {}))
    c["seed"] = seed
    arm = c["arm"]
    torch.manual_seed(seed)
    try:
        volume.reload()
    except Exception as e:
        print(cname, "reload skipped", repr(e)[:80])
    with open(f"{DIR}/pool.json") as f:
        pool = json.load(f)
    with open(f"{DIR}/evals.json") as f:
        evals = json.load(f)
    items, soft = pool["items"], pool["soft"]
    if smoke:
        c["steps"], c["align_steps"] = 3, 3
        items, soft = items[:64], soft[:64]
        evals = {k: v[:32] for k, v in evals.items()}
    data = r22.build_data()
    if smoke:
        data = {k: v[:16] for k, v in data.items()}
    cfg = dict(pgraft.DEFAULTS, **r22.CFG)
    row = {"cell": cname, "name": name, "seed": seed, "config": c, "pool": len(items), "version": VERSION}
    device = torch.device("cuda")
    tok = transformers.AutoTokenizer.from_pretrained(HOST)
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
        print(cname, "SCORES base", json.dumps(row["scores"]))
        return finish(row, t0, smoke, cname, volume)

    donor = donor_tok = None
    graft = None
    grafts = []
    lora_params = []
    needs_donor = arm in ("graft", "scrambled")
    if needs_donor:
        donor_tok = transformers.AutoTokenizer.from_pretrained(DONOR)
        donor = transformers.AutoModelForCausalLM.from_pretrained(DONOR, dtype=torch.bfloat16).to("cuda").eval()
        for p in donor.parameters():
            p.requires_grad_(False)

    joint_lora = c["lora_rank"] if arm in ("graft", "identity", "scrambled") else 0
    if joint_lora:
        lora_params = apply_lora(base, joint_lora, default_alpha(joint_lora))
        for p in lora_params:
            p.requires_grad_(True)

    def make(host_layer, first, last, control):
        g = pgraft.build(base, donor if control is None else None, host_layer, first, last, control=control,
                         enc_depth=c["enc_depth"], dec_rank=c["dec_rank"], d_donor=D_DONOR)
        return g

    if arm in ("graft", "identity", "scrambled"):
        control = "identity" if arm == "identity" else None
        grafts = [make(c["host_layer"], c["first"], c["last"], control)]
        if c["two_points"]:
            tp = c["two_points"]
            grafts.append(make(tp["host_layer"], tp["first"], tp["last"], control))
        pgraft.attach(base, grafts)
        graft = grafts[0]
        if needs_donor:
            for g in grafts:
                pgraft.set_sink(g, donor, donor_tok, g.donor_span[0], data["align"][:256])
        glue = pgraft.trainable(grafts, enc=True)
        row["trainable_glue"] = pgraft.count(glue)
        row["slice_params"] = sum(sum(p.numel() for p in g.part.parameters()) for g in grafts if g.control is None)
        if needs_donor:
            for g in grafts:
                one = [g]
                pgraft.align(base, tok, donor, donor_tok, g, data["align"], dict(cfg, align_steps=c["align_steps"]),
                             seed=seed, log_every=100)
                row.setdefault("alignment", []).append(
                    pgraft.alignment_report(base, tok, donor, donor_tok, g, data["align_eval"], cfg))
                print(cname, "alignment", row["alignment"][-1])
        for g in grafts:
            row.setdefault("calib_rms", []).append(pgraft.calibrate_scale(base, tok, g, data["align"], cfg))
        if arm == "scrambled":
            import copy
            for g in grafts:
                g.part = copy.deepcopy(g.part)
                rep = scramble.scramble_module(g.part, seed=seed + 100, mode="spectrum")
                row["scramble_cos"] = rep.get("mean_abs_cosine")
            del donor
            donor = None
            gc.collect()
            torch.cuda.empty_cache()
        trainable_total = pgraft.count(glue) + pgraft.count(lora_params)
    elif arm in ("lora_match", "lora_big"):
        if arm == "lora_match":
            target = row_budget(c)
            rank = rank_for_budget(base, target)
        else:
            rank = c["lora_rank"]
        lora_params = apply_lora(base, rank, default_alpha(rank))
        for p in lora_params:
            p.requires_grad_(True)
        row["lora_rank"] = rank
        trainable_total = pgraft.count(lora_params)
    else:  # fullft
        trainable_total = sum(p.numel() for p in base.parameters())
    row["trainable"] = trainable_total

    # optimiser groups
    if arm in ("graft", "identity", "scrambled"):
        gate = [p for g in grafts for p in g.token_gate.parameters()]
        dec = [p for g in grafts for p in g.dec.parameters()]
        enc = [p for g in grafts for p in g.enc.parameters()]
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
        steps += c["align_steps"]  # controls spend graft's stage-1 steps on the task, as in round 23
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
    align_feed = pgraft._batches(data["align"], cfg["align_batch"], seed + 11) if arm == "graft" else None
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
            for g in grafts:
                rl, _ = pgraft.functional_loss(base, tok, donor, donor_tok, g, next(align_feed), cfg)
                loss = loss + c["align_reg"] * rl / len(grafts)
        if graft is not None:
            gm = pgraft.mean_gate(grafts)
            if gm is not None:
                info["gate"] = float(gm.detach())
        return loss, info

    log = pgraft.train(base, groups, steps, step_fn, seed=seed + 1, log_every=100)
    row["tokens_trained"] = tokens[0]
    gates = [[r["step"], round(r["gate"], 4)] for r in log if "gate" in r and r["step"] % 100 == 0]
    row["gate_curve"] = gates
    row["loss_first50"] = sum(r["loss"] for r in log[:50]) / max(1, len(log[:50]))
    row["loss_last50"] = sum(r["loss"] for r in log[-50:]) / max(1, len(log[-50:]))
    print(cname, "train done", "loss", round(row["loss_first50"], 4), "->", round(row["loss_last50"], 4),
          "gate", gates[-1] if gates else None)
    row["train_seconds"] = round(time.time() - t0, 1)

    if arm == "fullft":
        base = base.to(torch.bfloat16)
    base.eval()
    row["scores"] = {k: score(base, tok, v) for k, v in evals.items()}
    print(cname, "SCORES", json.dumps(row["scores"]))
    if grafts and arm != "identity":
        with pgraft.grafts_off(base):
            row["scores_graft_off"] = {k: score(base, tok, evals[k]) for k in ("mmlu_dev", "mmlu_held")}
        print(cname, "SCORES graft off", json.dumps(row["scores_graft_off"]))
    # params run per token
    per_token = row["params_base"]
    if grafts and arm != "identity":
        per_token += row["slice_params"] + row["trainable_glue"]
    elif grafts:
        per_token += row["trainable_glue"]
    row["params_per_token"] = per_token
    row["params_total"] = per_token
    if c["decode"] and not smoke:
        row["decode_tps"] = decode_tps(base, tok)
    return finish(row, t0, smoke, cname, volume)


def row_budget(c):
    """Trainable count of the real graft with this config, the matched-LoRA budget."""
    from wsp import pgraft
    return (pgraft.count(pgraft.Enc(D_HOST, D_DONOR, depth=c["enc_depth"]).parameters())
            + pgraft.count(pgraft.Dec(D_DONOR, D_HOST, rank=c["dec_rank"]).parameters()) + D_HOST + 1
            + (0 if not c["lora_rank"] else 0))


def finish(row, t0, smoke, cname, volume):
    row["gpu_hours"] = round((time.time() - t0) / 3600, 3)
    row["seconds"] = round(time.time() - t0, 1)
    if not smoke:
        os.makedirs(ROWS, exist_ok=True)
        with open(f"{ROWS}/{row['name']}_{row['seed']}{row['config'].get('tag', '')}.json", "w") as f:
            json.dump(row, f)
        volume.commit()
    print(cname, "ROW", json.dumps(row))
    return row


@app.local_entrypoint()
def main(cells: str = "", smoke: bool = False, prep_data: bool = False, overrides: str = ""):
    if prep_data:
        print(json.dumps(prep.remote()))
        return
    calls = []
    for item in [x for x in cells.split(",") if x]:
        name, seed = item.split(":")
        if name not in PRESETS:
            raise SystemExit(f"unknown preset {name}")
        calls.append((name, int(seed), smoke, json.loads(overrides) if overrides else None))
    if not calls:
        raise SystemExit("0 cells: pass --cells name:seed,...")
    rows = []
    for row in cell.starmap(calls, return_exceptions=True):
        if isinstance(row, Exception):
            print("cell failed:", repr(row)[:1000])
            continue
        rows.append(row)
    os.makedirs("results", exist_ok=True)
    tag = "smoke" if smoke else cells.replace(":", "").replace(",", "_")[:60]
    with open(f"results/wi_xfam_{tag}.json", "w") as f:
        json.dump(rows, f, indent=2)
    print("wrote", len(rows), "rows")
