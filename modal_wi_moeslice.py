"""Write-in track moe-slice: graft a slice of Qwen3-30B-A3B (sparse MoE) into Qwen3-1.7B.

Base Qwen/Qwen3-1.7B (28 layers, d 2048). Donor Qwen/Qwen3-30B-A3B-Instruct-2507
(48 layers, d 2048, 128 experts, top 8). The graft reads base layer HOST_LAYER, runs
donor layers FIRST..LAST (frozen, routed experts only) on Enc's states, and writes the
donor's contribution back through Dec and a token gate (wsp.pgraft.ParallelGraft).
Both models share a tokenizer family, so stage-1 alignment pairs positions exactly.

Arms (same data, steps, selection rule): base, graft, identity, scrambled, lora_match,
lora_big (rank 128), fullft; reference cells: native donor, Qwen3-4B, Qwen3-8B.
Heal signal: answer-letter CE on a big 4-option pool (MMLU auxiliary_train, which holds
RACE/ARC/OBQA/SciQ items, plus the round 23 pools) mixed with soft letter targets from
the donor, plus KL to the frozen base on general text. MMLU test is never trained on.

    modal run --detach modal_wi_moeslice.py --smoke
    modal run --detach modal_wi_moeslice.py --only graft:0
"""

import json
import os
import random
import time

import modal

app = modal.App("wsp-wi-moeslice")
volume = modal.Volume.from_name("wsp-models", create_if_missing=True)

def make_image(offline):
    env = {"HF_HOME": "/vol/hf", "HF_HUB_DISABLE_XET": "1"}
    if offline:
        env.update({"HF_HUB_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1"})
    return (
        modal.Image.debian_slim(python_version="3.12")
        .pip_install(
            "torch==2.7.1", "transformers==4.57.1", "accelerate", "huggingface_hub", "requests",
            "safetensors", "zstandard", "numpy", "datasets", "sentencepiece", "protobuf",
        )
        .env(env)
        .add_local_python_source("wsp")
        .add_local_file("modal_graft.py", "/root/modal_graft.py")
        .add_local_file("modal_round5.py", "/root/modal_round5.py")
        .add_local_file("modal_round6.py", "/root/modal_round6.py")
        .add_local_file("modal_round7.py", "/root/modal_round7.py")
        .add_local_file("modal_round22pg.py", "/root/modal_round22pg.py")
        .add_local_file("modal_round23pg.py", "/root/modal_round23pg.py")
    )


image = make_image(False)
image_off = make_image(True)

HOST = "Qwen/Qwen3-1.7B"
DONOR = "Qwen/Qwen3-30B-A3B-Instruct-2507"
NATIVES = {"qwen3_4b": "Qwen/Qwen3-4B", "qwen3_8b": "Qwen/Qwen3-8B", "donor": DONOR}
D_MODEL = 2048
HOST_LAYER = 13
FIRST, LAST = 16, 32
TOPK, NEXPERTS = 8, 128
SEEDS = [0, 1]
ARMS = ["graft", "identity", "scrambled", "lora_match", "lora_big", "fullft"]
ALIGN_STEPS = 600
TASK_STEPS = 1500
MC_BATCH = 16
N_POOL = 30000
LR = {"glue": 3e-4, "gate": 1e-4, "enc2": 1e-4, "lora": 2e-4, "full": 1e-5}
PRESERVE = 1.0
ALIGN_REG = 1.0
SOFT_W = 0.5
OUT = "/vol/wi_moe"
GPU = "H200"


def reload():
    try:
        volume.reload()
    except Exception as err:
        print("reload skipped:", repr(err)[:120])


def save_row(name, row):
    os.makedirs(OUT + "/rows", exist_ok=True)
    tmp = f"{OUT}/rows/{name}.{os.getpid()}.tmp"
    with open(tmp, "w") as f:
        json.dump(row, f)
    os.replace(tmp, f"{OUT}/rows/{name}.json")
    volume.commit()


def letter_of(answer):
    return "ABCD".index(answer.strip())


def pool(tok=None, smoke=False):
    """{"train": [[prompt, gold], ...]} cached on the volume: MMLU auxiliary_train items (RACE, ARC, OBQA, SciQ
    style, 4 options). Training items only, never MMLU or ARC test."""
    path = OUT + "/pool.json"
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    from datasets import load_dataset

    from modal_round6 import mmlu_prompt

    t0 = time.time()
    aux = load_dataset("cais/mmlu", "auxiliary_train", split="train")
    print("aux rows", len(aux), round(time.time() - t0), flush=True)
    rows = aux.shuffle(seed=0).select(range(min(len(aux), 3 * N_POOL)))
    items = []
    for row in rows:
        if "train" in row and isinstance(row["train"], dict):
            row = row["train"]
        if len(row["choices"]) != 4:
            continue
        prompt = mmlu_prompt(row["question"], row["choices"])
        if len(prompt) > 1200:
            continue
        items.append([prompt, int(row["answer"])])
        if len(items) >= N_POOL:
            break
    print("pool built", len(items), round(time.time() - t0), flush=True)
    out = {"train": items}
    os.makedirs(OUT, exist_ok=True)
    with open(path + ".tmp", "w") as f:
        json.dump(out, f)
    os.replace(path + ".tmp", path)
    volume.commit()
    return out


def arc_items():
    from datasets import load_dataset

    from modal_round6 import mmlu_prompt

    rows = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test")
    items = []
    for row in rows:
        labels = list(row["choices"]["label"])
        texts = list(row["choices"]["text"])
        if len(texts) != 4 or row["answerKey"] not in labels:
            continue
        items.append((mmlu_prompt(row["question"], texts), labels.index(row["answerKey"])))
    return items


def mmlu_eval_items():
    """Round 6/7 split without the slow training-pair pass: dev = first 1000 of the seed-0 order, held-out = next 2000."""
    from datasets import load_dataset

    from modal_round6 import mmlu_prompt

    test = load_dataset("cais/mmlu", "all", split="test")
    order = list(range(len(test)))
    random.Random(0).shuffle(order)
    out = {}
    for name, span in (("dev", order[:1000]), ("held", order[1000:3000])):
        items = []
        for index in span:
            row = test[index]
            items.append((mmlu_prompt(row["question"], row["choices"]), int(row["answer"])))
        out[name] = items
    return out["dev"], out["held"]


def eval_items(tok, smoke):
    dev, held = mmlu_eval_items()
    arc = arc_items()
    if smoke:
        dev, held, arc = dev[:32], held[:32], arc[:32]
    return dev, held, arc


def humaneval(model, tok, smoke, batch=16, max_new=400):
    """Chat-template greedy pass@1 on HumanEval. Returns (passes, total)."""
    import torch

    from wsp import skills_eval as se

    items = se.humaneval_items()
    if smoke:
        items = items[:4]
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    programs, raw = [], []
    for start in range(0, len(items), batch):
        chunk = items[start:start + batch]
        texts = []
        for it in chunk:
            msg = [{"role": "user", "content": "Complete the following Python function. Reply with the full "
                    "function in one python code block.\n\n```python\n" + it["prompt"] + "```"}]
            try:
                texts.append(tok.apply_chat_template(msg, tokenize=False, add_generation_prompt=True,
                                                     enable_thinking=False))
            except TypeError:
                texts.append(tok.apply_chat_template(msg, tokenize=False, add_generation_prompt=True))
        enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=max_new, do_sample=False, pad_token_id=tok.pad_token_id)
        for it, seq in zip(chunk, out):
            text = tok.decode(seq[enc["input_ids"].shape[1]:], skip_special_tokens=True)
            raw.append(text)
            programs.append(se.assemble_humaneval(it["prompt"], text, it["test"], it["entry_point"]))
    results = se.run_programs(programs, timeout=10.0, workers=8)
    passed = sum(1 for r in results if r["passed"])
    print("humaneval sample output:", repr(raw[0][:300]))
    tok.padding_side = "right"
    return passed, len(items)


def decode_toks_per_s(model, tok, n=96):
    import torch

    tok.padding_side = "left"
    enc = tok("The history of the Roman Empire begins", return_tensors="pt").to(model.device)
    with torch.no_grad():
        model.generate(**enc, max_new_tokens=8, do_sample=False, pad_token_id=tok.pad_token_id or tok.eos_token_id)
        torch.cuda.synchronize()
        t0 = time.time()
        model.generate(**enc, max_new_tokens=n, min_new_tokens=n, do_sample=False,
                       pad_token_id=tok.pad_token_id or tok.eos_token_id)
        torch.cuda.synchronize()
    tok.padding_side = "right"
    return round(n / (time.time() - t0), 2)


def score_all(model, tok, dev, held, arc, smoke, row, he=True):
    from modal_round6 import score_mmlu

    row["mmlu_dev"] = score_mmlu(model, tok, dev, batch_size=16)
    row["mmlu_heldout"] = score_mmlu(model, tok, held, batch_size=16)
    row["arc_c"] = score_mmlu(model, tok, arc, batch_size=16)
    print("SCORE", row.get("cell"), {k: row[k] for k in ("mmlu_dev", "mmlu_heldout", "arc_c")}, flush=True)
    if he:
        try:
            row["humaneval"] = humaneval(model, tok, smoke)
            print("SCORE", row.get("cell"), "humaneval", row["humaneval"], flush=True)
        except Exception as err:
            row["humaneval_error"] = repr(err)[:300]
    try:
        row["tok_s_b1"] = decode_toks_per_s(model, tok)
    except Exception as err:
        row["tok_s_error"] = repr(err)[:300]


def active_params(part):
    """Params run per token in a donor slice: dense tensors fully, routed experts at TOPK/NEXPERTS."""
    total = 0.0
    stored = 0
    for name, p in part.named_parameters():
        stored += p.numel()
        total += p.numel() * (TOPK / NEXPERTS if ".experts." in name else 1.0)
    return int(total), stored


@app.function(image=image_off, volumes={"/vol": volume}, memory=16384, timeout=3600)
def build_pool():
    reload()
    return len(pool()["train"])


@app.function(image=image, volumes={"/vol": volume}, memory=16384, timeout=3 * 3600)
def prefetch(model_id, xet=False):
    """Download a model to the volume (CPU only), so GPU cells do not pay for it."""
    if xet:
        os.environ["HF_HUB_DISABLE_XET"] = "0"
    from huggingface_hub import snapshot_download

    reload()
    t0 = time.time()
    if model_id == "arc":
        from datasets import load_dataset

        load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test")
        volume.commit()
        print("PREFETCHED arc")
        return model_id
    snapshot_download(model_id, ignore_patterns=["*.bin", "*.pt", "*.gguf", "*.onnx", "*.h5", "original/*", "*.msgpack"])
    volume.commit()
    print("PREFETCHED", model_id, round(time.time() - t0))
    return model_id


@app.function(image=image_off, volumes={"/vol": volume}, gpu=GPU, cpu=8, memory=131072, timeout=6 * 3600)
def native_cell(key, smoke=False, soft=False):
    """Score a native model (qwen3_4b, qwen3_8b, donor). soft=True also writes donor letter logits for the pool."""
    import torch
    import transformers

    reload()
    start = time.time()
    model_id = NATIVES[key] if key in NATIVES else key
    tok = transformers.AutoTokenizer.from_pretrained(model_id)
    model = transformers.AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16).to("cuda").eval()
    row = {"cell": f"native:{key}", "model": model_id, "smoke": smoke,
           "total_params": sum(p.numel() for p in model.parameters())}
    if soft:
        from modal_round6 import letter_ids

        items = pool(transformers.AutoTokenizer.from_pretrained(HOST), smoke)["train"]
        if smoke:
            items = items[:64]
        letters = letter_ids(tok)
        order = sorted(range(len(items)), key=lambda i: len(items[i][0]))
        out = [None] * len(items)
        tok.padding_side = "left"
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        for s in range(0, len(order), 32):
            idx = order[s:s + 32]
            enc = tok([items[i][0] for i in idx], return_tensors="pt", padding=True).to("cuda")
            with torch.no_grad():
                lg = model(**enc, use_cache=False).logits[:, -1, :].float()[:, letters]
            for i, v in zip(idx, lg.tolist()):
                out[i] = [round(x, 3) for x in v]
            if (s // 32) % 100 == 0:
                print("soft", s, len(order), round(time.time() - start), flush=True)
        tok.padding_side = "right"
        if not smoke:
            with open(OUT + "/soft.json.tmp", "w") as f:
                json.dump(out, f)
            os.replace(OUT + "/soft.json.tmp", OUT + "/soft.json")
            volume.commit()
        row["soft_acc"] = sum(1 for o, it in zip(out, items) if max(range(4), key=lambda j: o[j]) == it[1]) / len(items)
    htok = transformers.AutoTokenizer.from_pretrained(HOST)
    dev, held, arc = eval_items(htok, smoke)
    score_all(model, tok, dev, held, arc, smoke, row)
    row["gpu_hours"] = round((time.time() - start) / 3600, 3)
    if not smoke:
        save_row(f"native_{key}", row)
    print(json.dumps(row))
    return row


@app.function(image=image_off, volumes={"/vol": volume}, gpu=GPU, cpu=8, memory=131072, timeout=8 * 3600)
def cell(arm, seed, smoke=False, steps=None, hp=None):
    import gc

    import torch
    import transformers

    import modal_round22pg as r22
    from modal_round6 import letter_ids
    from wsp import pgraft
    from wsp.lora import apply_lora, default_alpha, rank_for_budget
    from wsp.scramble import scramble_module

    reload()
    start = time.time()
    print("cell start", arm, seed, flush=True)
    hp = dict(hp or {})
    first, last = hp.get("first", FIRST), hp.get("last", LAST)
    host_layer = hp.get("host_layer", HOST_LAYER)
    name = f"{arm}_s{seed}" + (("_" + hp["tag"]) if hp.get("tag") else "")
    row = {"cell": name, "arm": arm, "seed": seed, "first": first, "last": last, "host_layer": host_layer,
           "hp": hp, "smoke": smoke}
    torch.manual_seed(seed)
    device = torch.device("cuda")
    tok = transformers.AutoTokenizer.from_pretrained(HOST)
    dtype = torch.float32 if arm == "fullft" else torch.bfloat16
    base = transformers.AutoModelForCausalLM.from_pretrained(HOST, dtype=dtype).to(device).eval()
    for p in base.parameters():
        p.requires_grad_(arm == "fullft")
    ref = transformers.AutoModelForCausalLM.from_pretrained(HOST, dtype=torch.bfloat16).to(device).eval()
    for p in ref.parameters():
        p.requires_grad_(False)
    base_total = sum(p.numel() for p in base.parameters())
    letters = letter_ids(tok)
    items = pool(tok, smoke)["train"]
    print("pool ready", len(items), round(time.time() - start), flush=True)
    soft = None
    if os.path.exists(OUT + "/soft.json") and not smoke:
        with open(OUT + "/soft.json") as f:
            soft = json.load(f)
    row["soft"] = soft is not None
    if smoke:
        items = items[:64]
    align_steps = hp.get("align_steps", ALIGN_STEPS)
    task_steps = steps or hp.get("task_steps", TASK_STEPS)
    if smoke:
        align_steps, task_steps = hp.get("align_steps", 3), steps or 3
    data = r22.build_data()
    print("data ready", round(time.time() - start), flush=True)
    if smoke:
        data = {k: v[:16] for k, v in data.items()}
    cfg = dict(pgraft.DEFAULTS, **{"align_steps": align_steps, "geo_steps": min(100, align_steps // 2 + 1)})
    lr = dict(LR, **{k: v for k, v in hp.items() if k in LR})

    graft = None
    lora_params = None
    donor = donor_tok = None
    active, stored_extra = base_total, 0
    if arm in ("graft", "scrambled"):
        donor_tok = transformers.AutoTokenizer.from_pretrained(DONOR)
        donor = transformers.AutoModelForCausalLM.from_pretrained(DONOR, dtype=torch.bfloat16,
                                                                  device_map="cuda").eval()
        print("donor loaded", round(time.time() - start), flush=True)
        for p in donor.parameters():
            p.requires_grad_(False)
    if arm in ("graft", "scrambled", "identity"):
        control = "identity" if arm == "identity" else None
        depth, rank = hp.get("enc_depth", 2), hp.get("dec_rank", 256)
        with torch.device(device):
            graft = pgraft.build(base, donor, host_layer, first, last, control=control, enc_depth=depth,
                                 dec_rank=rank, d_donor=D_MODEL)
        print("graft built", round(time.time() - start), flush=True)
        pgraft.attach(base, [graft])
        if control is None:
            pgraft.set_sink(graft, donor, donor_tok, first, data["align"][:256])
        if arm == "scrambled":
            rep = scramble_module(graft.part, seed=seed + 77)
            row["scramble"] = {k: rep[k] for k in ("mode", "tensors_changed", "rel_fro_error") if k in rep}
        row["trainable"] = pgraft.count(pgraft.trainable([graft], enc=True))
        if control is None:
            a, s = active_params(graft.part)
            active, stored_extra = base_total + a, s
        row["calib_rms"] = pgraft.calibrate_scale(base, tok, graft, data["align"], cfg)
        if arm == "graft":
            t_align = time.time()
            pgraft.align(base, tok, donor, donor_tok, graft, data["align"], cfg, seed=seed, log_every=10)
            row["align_seconds"] = round(time.time() - t_align, 1)
            print("ALIGN_TIME", name, align_steps, row["align_seconds"], flush=True)
            row["alignment"] = pgraft.alignment_report(base, tok, donor, donor_tok, graft, data["align_eval"], cfg)
            print("alignment", name, row["alignment"], flush=True)
            save_row(name + "_stage1", row)
    budget = (pgraft.count(pgraft.Enc(D_MODEL, D_MODEL, depth=hp.get("enc_depth", 2)).parameters())
              + pgraft.count(pgraft.Dec(D_MODEL, D_MODEL, rank=hp.get("dec_rank", 256)).parameters()) + D_MODEL + 1)
    if arm in ("lora_match", "lora_big"):
        rank = rank_for_budget(base, budget) if arm == "lora_match" else 128
        lora_params = apply_lora(base, rank, default_alpha(rank))
        for p in lora_params:
            p.requires_grad_(True)
        row["lora_rank"] = rank
        row["trainable"] = pgraft.count(lora_params)
    if arm == "fullft":
        row["trainable"] = base_total
    row["budget_glue"] = budget

    def groups():
        if arm == "fullft":
            return [{"params": [p for p in base.parameters()], "lr": lr["full"]}]
        if lora_params is not None:
            return [{"params": lora_params, "lr": lr["lora"]}]
        gate = list(graft.token_gate.parameters())
        dec = list(graft.dec.parameters())
        enc = list(graft.enc.parameters())
        if arm == "graft":
            return [{"params": dec, "lr": lr["glue"]}, {"params": gate, "lr": lr["gate"]},
                    {"params": enc, "lr": lr["enc2"]}]
        return [{"params": enc + dec, "lr": lr["glue"]}, {"params": gate, "lr": lr["gate"]}]

    align_feed = pgraft._batches(data["align"], cfg["align_batch"], seed + 11)
    gen_feed = pgraft._batches(data["general"], 4, seed + 41)
    order = list(range(len(items)))
    random.Random(seed + 5).shuffle(order)
    cursor = [0]

    def next_batch():
        idx = [order[(cursor[0] + i) % len(order)] for i in range(MC_BATCH)]
        cursor[0] += MC_BATCH
        return idx

    import torch.nn.functional as F

    def mc_step(step):
        idx = next_batch()
        batch = [(items[i][0], items[i][1]) for i in idx]
        loss, info = pgraft.mc_loss(base, tok, batch, letters, device)
        if soft is not None and SOFT_W > 0:
            tok.padding_side = "left"
            enc = tok([b[0] for b in batch], return_tensors="pt", padding=True).to(device)
            with pgraft.autocast(device):
                lg = base(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"],
                          use_cache=False).logits[:, -1, :].float()[:, letters]
            tok.padding_side = "right"
            target = F.softmax(torch.tensor([soft[i] for i in idx], device=device), dim=-1)
            kd = F.kl_div(F.log_softmax(lg, dim=-1), target, reduction="batchmean")
            loss = (1 - SOFT_W) * loss + SOFT_W * kd
            info["kd"] = float(kd.detach())
        if graft is not None:
            g = pgraft.mean_gate([graft])
            if g is not None:
                info["gate"] = float(g.detach())
        enc = pgraft._tok_batch(tok, next(gen_feed), cfg["seq_len"], device)
        _, logits = pgraft.seq_lm_loss(base, enc, return_logits=True)
        drift = pgraft.drift_kl(ref, enc, logits)
        loss = loss + PRESERVE * drift
        info["drift"] = float(drift.detach())
        if arm == "graft" and step % 2 == 0:
            reg, _ = pgraft.functional_loss(base, tok, donor, donor_tok, graft, next(align_feed), cfg)
            loss = loss + ALIGN_REG * reg
        return loss, info

    extra = 0 if arm == "graft" else align_steps
    t_train = time.time()
    log = pgraft.train(base, groups(), task_steps + extra, mc_step, seed=seed + 1, log_every=10 if smoke else 50)
    row["train_step_seconds"] = round((time.time() - t_train) / max(1, task_steps + extra), 2)
    print("TRAIN_TIME", name, task_steps + extra, row["train_step_seconds"], flush=True)
    gates = [[r["step"], round(r["gate"], 4)] for r in log if "gate" in r and r["step"] % 100 == 0]
    row["gate_curve"] = gates
    row["loss_first50"] = sum(r["loss"] for r in log[:50]) / len(log[:50])
    row["loss_last50"] = sum(r["loss"] for r in log[-50:]) / len(log[-50:])
    row["train_seconds"] = round(time.time() - start, 1)
    row["train_tokens_est"] = (task_steps + extra) * MC_BATCH * 160
    print("TRAINED", name, row["loss_first50"], row["loss_last50"], flush=True)
    if not smoke:
        save_row(name + "_trained", row)

    donor = None
    gc.collect()
    torch.cuda.empty_cache()
    if arm == "fullft":
        base.to(torch.bfloat16)
    base.eval()
    dev, held, arc = eval_items(tok, smoke)
    row["total_params"] = base_total + stored_extra + (row["trainable"] if arm in ("graft", "scrambled", "identity") else 0)
    row["run_params_per_token"] = active + (row["trainable"] if graft is not None else 0)
    if arm in ("lora_match", "lora_big"):
        row["total_params"] = base_total + row["trainable"]
        row["run_params_per_token"] = row["total_params"]
    if arm in ("fullft",):
        row["total_params"] = row["run_params_per_token"] = base_total
    score_all(base, tok, dev, held, arc, smoke, row)
    if graft is not None:
        with pgraft.grafts_off(base):
            from modal_round6 import score_mmlu

            row["mmlu_heldout_graft_off"] = score_mmlu(base, tok, held, batch_size=16)
    row["gpu_hours"] = round((time.time() - start) / 3600, 3)
    if not smoke:
        save_row(name, row)
    print(json.dumps(row))
    return row


@app.function(image=image_off, volumes={"/vol": volume}, gpu=GPU, cpu=8, memory=65536, timeout=3600)
def base_cell(smoke=False):
    """Untrained base, scored the same way."""
    import torch
    import transformers

    reload()
    start = time.time()
    tok = transformers.AutoTokenizer.from_pretrained(HOST)
    model = transformers.AutoModelForCausalLM.from_pretrained(HOST, dtype=torch.bfloat16).to("cuda").eval()
    row = {"cell": "base", "arm": "base", "total_params": sum(p.numel() for p in model.parameters()), "trainable": 0}
    row["run_params_per_token"] = row["total_params"]
    dev, held, arc = eval_items(tok, smoke)
    score_all(model, tok, dev, held, arc, smoke, row)
    row["gpu_hours"] = round((time.time() - start) / 3600, 3)
    if not smoke:
        save_row("base", row)
    print(json.dumps(row))
    return row


@app.local_entrypoint()
def main(only: str = "", smoke: bool = False, steps: int = 0, hp: str = ""):
    """only: comma list of arm:seed, native:key, native:donor:soft, base. hp: json of overrides."""
    hpd = json.loads(hp) if hp else None
    jobs = []
    for item in (only.split(",") if only else []):
        parts = item.split(":")
        if parts[0] == "pool":
            jobs.append(build_pool.spawn())
        elif parts[0] == "prefetch":
            jobs.append(prefetch.spawn(NATIVES.get(parts[1], parts[1]), len(parts) > 2))
        elif parts[0] == "base":
            jobs.append(base_cell.spawn(smoke))
        elif parts[0] == "native":
            jobs.append(native_cell.spawn(parts[1], smoke, len(parts) > 2 and parts[2] == "soft"))
        else:
            jobs.append(cell.spawn(parts[0], int(parts[1]), smoke, steps or None, hpd))
    if not jobs:
        raise SystemExit("name cells with --only")
    for job in jobs:
        try:
            print(json.dumps(job.get()))
        except Exception as err:
            print("cell failed:", repr(err)[:800])
