"""Write-in track math-rl: write Qwen2.5-Math-7B-Instruct's math skill into Qwen2.5-1.5B-Instruct through a parallel graft.

Base Qwen/Qwen2.5-1.5B-Instruct (28 layers, d 1536), donor Qwen/Qwen2.5-Math-7B-Instruct (28 layers, d 3584).
The graft is wsp.pgraft.ParallelGraft: Enc into a frozen donor slice, Dec back, per-token gate, plus a small LoRA on
the base trained jointly. Every arm gets the same data, steps and selection rule (the final state, no early pick):

  stage D  donor_data (vLLM): sample the donor on GSM8K train and MATH train (decontaminated against MATH-500),
           keep answers with the right final answer, at most 2 per problem. 200 problems of each set stay out as dev.
  stage 1  (graft arms) functional alignment of Enc on the donor's own next-token KL, on the donor solutions' text.
  stage 2  SFT on the donor solutions (response tokens only), steps SFT_STEPS.
  stage 3  verified self-training, RFT_ROUNDS rounds: sample the current model on train problems, keep answers
           with the right final answer, train RFT_STEPS steps on them mixed with replayed donor solutions.
           This is expert iteration with an exact-answer reward, not GRPO. Same budget for every arm.

Arms: base, donor (reference, no training), graft, identity (same glue, donor skipped), scrambled (same graft with
the donor slice scrambled after alignment, wsp/scramble.py), lora_match (rank sized to the graft's total trainable),
lora_big (rank 128), fullft (all base weights). Score: GSM8K test 500 (seeded), MATH-500, MMLU 1000 (guard), dev
slice from train, params, trainable, tokens, GPU hours, decode tok/s at batch 1.

    modal run modal_wi_mathrl.py --stage data
    modal run modal_wi_mathrl.py --stage smoke --only graft:0
    modal run --detach modal_wi_mathrl.py --stage run --only graft:0,graft:1
"""

import json
import math
import os
import random
import time

import modal

app = modal.App("wsp-wi-mathrl")
volume = modal.Volume.from_name("wsp-models", create_if_missing=True)

VLLM_VERSION = "0.32.0"
vllm_image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.12")
    .env({"CUDA_HOME": "/usr/local/cuda"})
    .pip_install(
        f"vllm=={VLLM_VERSION}", "accelerate", "huggingface_hub", "requests",
        "safetensors", "zstandard", "numpy", "datasets", "sentencepiece", "protobuf",
    )
    .env({"HF_HOME": "/vol/hf", "HF_HUB_DISABLE_XET": "1"})
    .add_local_python_source("wsp")
)
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch==2.7.1", "transformers==4.57.1", "accelerate", "huggingface_hub", "requests",
        "safetensors", "zstandard", "numpy", "datasets", "sentencepiece", "protobuf",
    )
    .env({"HF_HOME": "/vol/hf"})
    .add_local_python_source("wsp")
)

BASE = "Qwen/Qwen2.5-1.5B-Instruct"
DONOR = "Qwen/Qwen2.5-Math-7B-Instruct"
D_HOST, D_DONOR = 1536, 3584
HOST_LAYER, FIRST, LAST = 14, 4, 20
FORMAT = "v1"
ROOT = "/vol/wi_mathrl/" + FORMAT
ROWS = ROOT + "/rows"
MATH_SUBJECTS = ["algebra", "counting_and_probability", "geometry", "intermediate_algebra", "number_theory",
                 "prealgebra", "precalculus"]
DEV_EACH = 200

ARMS = ["graft", "identity", "scrambled", "lora_match", "lora_big", "fullft"]
SEEDS = [0, 1]
ALIGN_STEPS = 400
SFT_STEPS = 1000
RFT_ROUNDS = 2
RFT_STEPS = 150
RFT_PROMPTS = {"gsm8k": 500, "math500": 300}
RFT_SAMPLE_TOKENS = 640
MICRO = 4
ACCUM = 4
MAX_LEN = 768
GRAFT_LORA_RANK = 8
BIG_RANK = 128
LR = {"glue": 3e-4, "gate": 1e-4, "enc2": 1e-4, "lora": 2e-4, "lora_big": 1e-4, "fullft": 1e-5}
SMOKE = {"align": 3, "sft": 4, "rft_steps": 3, "rft_prompts": {"gsm8k": 8, "math500": 8}, "eval": 16}


# ------------------------------------------------------------------ data (vLLM)


def load_train_problems():
    """GSM8K train and MATH train as grading items, MATH-500 problems removed. Returns {skill: [item]}."""
    from datasets import load_dataset

    from wsp import skills_eval as se

    out = {"gsm8k": [], "math500": []}
    rows = load_dataset("openai/gsm8k", "main", split="train")
    for index in range(len(rows)):
        row = rows[index]
        out["gsm8k"].append({"id": f"gsm8k_train:{index}", "question": row["question"],
                             "gold": se.final_number(row["answer"])})
    banned = {item["problem"].strip() for item in se.math500_items()}
    for subject in MATH_SUBJECTS:
        rows = load_dataset("EleutherAI/hendrycks_math", subject, split="train")
        for index in range(len(rows)):
            row = rows[index]
            if row["problem"].strip() in banned:
                continue
            gold = se.last_boxed(row["solution"])
            if gold is None:
                continue
            out["math500"].append({"id": f"math_train:{subject}:{index}", "problem": row["problem"], "gold": gold})
    for skill in out:
        random.Random(0).shuffle(out[skill])
    return out


@app.function(image=vllm_image, volumes={"/vol": volume}, gpu="H100", memory=131072, cpu=16, timeout=4 * 3600)
def donor_data(n=4, smoke=False):
    """Sample the donor on the train problems; keep verified answers. Writes dev.json, pool.json, sft.jsonl."""
    import transformers
    from vllm import LLM, SamplingParams

    from wsp import skills_eval as se

    start = time.time()
    problems = load_train_problems()
    dev = {s: problems[s][:DEV_EACH] for s in problems}
    pool = {s: problems[s][DEV_EACH:] for s in problems}
    if smoke:
        pool = {s: pool[s][:40] for s in pool}
    tok = transformers.AutoTokenizer.from_pretrained(DONOR)
    llm = LLM(model=DONOR, dtype="bfloat16", seed=0, gpu_memory_utilization=0.90, max_model_len=2048)
    rows, stats = [], {}
    for skill in ("gsm8k", "math500"):
        items = pool[skill]
        texts = [se.user_text(skill, it) for it in items]
        prompts = [se.render_prompt(tok, t) for t in texts]
        outs = llm.generate(prompts, SamplingParams(n=n, temperature=0.7, top_p=0.95,
                                                   max_tokens=1024, seed=0))
        solved = kept = total = 0
        for it, text, out in zip(items, texts, outs):
            good = []
            for comp in out.outputs:
                total += 1
                if comp.finish_reason != "stop":
                    continue
                rec = se.grade_outputs(skill, [it], [comp.text])[0]
                if rec["correct"] and comp.text not in good:
                    good.append(comp.text)
            if good:
                solved += 1
            for g in good[:2]:
                rows.append({"skill": skill, "id": it["id"], "user": text, "response": g.strip()})
                kept += 1
        stats[skill] = {"problems": len(items), "solved": solved, "kept": kept, "samples": total}
        print("donor_data", skill, stats[skill])
    os.makedirs(ROOT, exist_ok=True)
    random.Random(0).shuffle(rows)
    with open(ROOT + "/sft.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    with open(ROOT + "/dev.json", "w") as f:
        json.dump(dev, f)
    with open(ROOT + "/pool.json", "w") as f:
        json.dump(pool, f)
    stats["seconds"] = round(time.time() - start, 1)
    with open(ROOT + "/data_stats.json", "w") as f:
        json.dump(stats, f)
    volume.commit()
    print("donor_data done", json.dumps(stats))
    return stats


# ------------------------------------------------------------------ training helpers


def sample_texts(model, tok, prompts, max_new_tokens, batch_size=64, temperature=0.8, top_p=0.95):
    """Sampled generation, left padding, batches sorted by prompt length. Returns texts in prompt order."""
    import torch

    from wsp import skills_eval as se

    se.prepare_tokenizer(tok)
    device = se.model_device(model)
    stop_ids = se.stop_token_ids(model, tok)
    order = sorted(range(len(prompts)), key=lambda i: -len(prompts[i]))
    texts = [None] * len(prompts)
    for s in range(0, len(order), batch_size):
        chunk = order[s:s + batch_size]
        enc = tok([prompts[i] for i in chunk], return_tensors="pt", padding=True, add_special_tokens=False)
        ids = enc["input_ids"].to(device)
        mask = enc["attention_mask"].to(device)
        with torch.no_grad():
            out = model.generate(input_ids=ids, attention_mask=mask, max_new_tokens=max_new_tokens, do_sample=True,
                                 temperature=temperature, top_p=top_p, top_k=None, pad_token_id=tok.pad_token_id,
                                 eos_token_id=stop_ids)
        finished = []
        for r, i in enumerate(chunk):
            new = out[r, ids.shape[1]:]
            hit = any(int(t) in stop_ids for t in new)
            texts[i] = tok.decode(new, skip_special_tokens=True)
            finished.append(hit)
        for r, i in enumerate(chunk):
            if not finished[r]:
                texts[i] = None
    return texts


def encode_pairs(tok, pairs, max_len=MAX_LEN):
    """pairs: [(user_text, response)]. Right-padded ids, mask, response-only labels."""
    import torch

    from wsp import skills_eval as se

    seqs = []
    for user, resp in pairs:
        p = tok(se.render_prompt(tok, user), add_special_tokens=False).input_ids
        r = tok(resp + "<|im_end|>", add_special_tokens=False).input_ids
        if len(p) + len(r) > max_len:
            continue
        seqs.append((p, r))
    if not seqs:
        return None
    width = max(len(p) + len(r) for p, r in seqs)
    pad = tok.pad_token_id if tok.pad_token_id is not None else 0
    ids = torch.full((len(seqs), width), pad, dtype=torch.long)
    mask = torch.zeros((len(seqs), width), dtype=torch.long)
    labels = torch.full((len(seqs), width), -100, dtype=torch.long)
    for i, (p, r) in enumerate(seqs):
        n = len(p) + len(r)
        ids[i, :n] = torch.tensor(p + r)
        mask[i, :n] = 1
        labels[i, len(p):n] = torch.tensor(r)
    return ids, mask, labels


def sft_phase(model, tok, groups, pairs, steps, seed, name, warmup=30, log_every=50, device="cuda"):
    """Cross-entropy on response tokens. Effective batch MICRO * ACCUM. Returns (log, tokens_seen)."""
    import torch
    import torch.nn.functional as F

    from wsp.stitch import autocast

    groups = [g for g in groups if len(g["params"]) > 0]
    params = [p for g in groups for p in g["params"]]
    opt = torch.optim.AdamW(groups, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warmup) * 0.5 * (1 + math.cos(math.pi * s / steps)))
    rng = random.Random(seed)
    order = list(range(len(pairs)))
    rng.shuffle(order)
    cursor = 0
    log, tokens = [], 0
    model.train()
    for step in range(steps):
        opt.zero_grad()
        total = 0.0
        for _ in range(ACCUM):
            chunk = []
            while len(chunk) < MICRO:
                if cursor >= len(order):
                    rng.shuffle(order)
                    cursor = 0
                chunk.append(pairs[order[cursor]])
                cursor += 1
            enc = encode_pairs(tok, chunk)
            if enc is None:
                continue
            ids, mask, labels = [x.to(device) for x in enc]
            with autocast(torch.device(device)):
                logits = model(input_ids=ids, attention_mask=mask, use_cache=False).logits
            loss = F.cross_entropy(logits[:, :-1].float().reshape(-1, logits.shape[-1]),
                                   labels[:, 1:].reshape(-1), ignore_index=-100)
            (loss / ACCUM).backward()
            total += float(loss.detach()) / ACCUM
            tokens += int(mask.sum())
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        sched.step()
        log.append({"step": step, "loss": total})
        if not math.isfinite(total) or (step > 40 and total > 4.0):
            raise RuntimeError(f"{name}: loss jumped to {total} at step {step}")
        if step % log_every == 0:
            print(f"[{name}] step {step} loss {total:.4f}", flush=True)
    model.eval()
    return log, tokens


def gate_mean(graft):
    from wsp import pgraft

    g = pgraft.mean_gate([graft])
    return None if g is None else float(g.detach())


def evaluate(model, tok, items, smoke=False):
    """GSM8K test 500, MATH-500, MMLU 1000, dev slices. Returns a dict of scores."""
    from wsp import skills_eval as se

    lim = SMOKE["eval"] if smoke else None
    out = {}
    for skill in ("gsm8k", "math500"):
        r = se.evaluate_skill(model, tok, skill, items=items[skill], batch_size=50, limit=lim)
        out[skill] = round(r["score"], 4)
        out[skill + "_n"] = r["n"]
        print(f"EVAL {items['name']} {skill} {r['score']:.4f} ({r['correct']}/{r['n']}) {r['seconds']}s", flush=True)
        out[skill + "_sample"] = r["records"][0]["output"][:300]
    for skill in ("gsm8k", "math500"):
        r = se.evaluate_skill(model, tok, skill, items=items["dev_" + skill], batch_size=50, limit=lim)
        out["dev_" + skill] = round(r["score"], 4)
    r = se.score_choices(model, tok, items["mmlu"][:lim] if lim else items["mmlu"], batch_size=16)
    out["mmlu"] = round(sum(1 for x in r if x["correct"]) / len(r), 4)
    print(f"EVAL {items['name']} mmlu {out['mmlu']:.4f} gsm {out['gsm8k']:.4f} math {out['math500']:.4f}", flush=True)
    return out


def eval_items(name):
    from wsp import skills_eval as se

    with open(ROOT + "/dev.json") as f:
        dev = json.load(f)
    items = {"name": name, "gsm8k": se.gsm8k_items(500), "math500": se.math500_items(), "mmlu": se.mmlu_items(1000),
             "dev_gsm8k": dev["gsm8k"][:150], "dev_math500": dev["math500"][:150]}
    return items


def save_row(name, row):
    os.makedirs(ROWS, exist_ok=True)
    with open(f"{ROWS}/{name}.json", "w") as f:
        json.dump(row, f)
    try:
        volume.commit()
    except Exception as e:
        print("commit failed", repr(e)[:200])


@app.function(image=image, volumes={"/vol": volume}, gpu="H100", memory=131072, timeout=8 * 3600)
def cell(arm, seed, smoke=False, tag="", overrides=None):
    import gc

    import torch
    import transformers

    from wsp import pgraft
    from wsp import skills_eval as se
    from wsp.lora import apply_lora, default_alpha, rank_for_budget
    from wsp.scramble import scramble_module

    start = time.time()
    name = f"{arm}_s{seed}{tag}" + ("_smoke" if smoke else "")
    ov = overrides or {}
    align_steps = SMOKE["align"] if smoke else ov.get("align_steps", ALIGN_STEPS)
    sft_steps = SMOKE["sft"] if smoke else ov.get("sft_steps", SFT_STEPS)
    rft_rounds = ov.get("rft_rounds", RFT_ROUNDS)
    rft_steps = SMOKE["rft_steps"] if smoke else RFT_STEPS
    rft_prompts = SMOKE["rft_prompts"] if smoke else RFT_PROMPTS
    lr = dict(LR, **ov.get("lr", {}))
    torch.manual_seed(seed)
    volume.reload()
    row = {"cell": name, "arm": arm, "seed": seed, "format": FORMAT, "smoke": smoke, "overrides": ov,
           "base": BASE, "donor": DONOR, "span": [FIRST, LAST], "host_layer": HOST_LAYER}
    items = eval_items(name)
    tok = transformers.AutoTokenizer.from_pretrained(BASE)
    dtok = transformers.AutoTokenizer.from_pretrained(DONOR)
    model_id = DONOR if arm == "donor" else BASE
    base = transformers.AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16).to("cuda").eval()
    for p in base.parameters():
        p.requires_grad_(False)
    if arm in ("base", "donor"):
        row["scores"] = evaluate(base, tok if arm == "base" else dtok, items, smoke)
        total, run = se.model_params(base)
        row.update(params_total=total, params_run=run, trainable=0, train_tokens=0)
        row["tok_s"] = round(se.decode_speed(base, tok if arm == "base" else dtok), 1)
        row["gpu_hours"] = round((time.time() - start) / 3600, 3)
        save_row(name, row)
        print("ROW", json.dumps(row))
        return row

    pairs = []
    with open(ROOT + "/sft.jsonl") as f:
        for line in f:
            r = json.loads(line)
            pairs.append((r["user"], r["response"]))
    with open(ROOT + "/pool.json") as f:
        pool = json.load(f)
    if smoke:
        pairs = pairs[:64]
    row["sft_pairs"] = len(pairs)
    graft = None
    lora = []
    glue_budget = (pgraft.count(pgraft.Enc(D_HOST, D_DONOR, depth=2).parameters())
                   + pgraft.count(pgraft.Dec(D_DONOR, D_HOST, rank=256).parameters()) + D_HOST + 1)
    cfg = dict(pgraft.DEFAULTS, align_steps=align_steps)
    logs = {}
    tokens = 0

    if arm in ("graft", "identity", "scrambled"):
        control = "identity" if arm == "identity" else None
        donor = None
        if control is None:
            donor = transformers.AutoModelForCausalLM.from_pretrained(DONOR, dtype=torch.bfloat16).to("cuda").eval()
            for p in donor.parameters():
                p.requires_grad_(False)
        graft = pgraft.build(base, donor, HOST_LAYER, FIRST, LAST, control=control, enc_depth=2, dec_rank=256,
                             d_donor=D_DONOR)
        pgraft.attach(base, [graft])
        texts = [dtok.apply_chat_template([{"role": "user", "content": u}, {"role": "assistant", "content": r}],
                                          tokenize=False) for u, r in pairs]
        a_train, a_eval = texts[:-24], texts[-24:]
        if donor is not None:
            pgraft.set_sink(graft, donor, dtok, FIRST, a_train[:256])
            pgraft.align(base, tok, donor, dtok, graft, a_train, cfg, seed=seed, log_every=100)
            row["alignment"] = pgraft.alignment_report(base, tok, donor, dtok, graft, a_eval, cfg)
            print(f"[{name}] alignment", row["alignment"], flush=True)
            if arm == "graft" and not smoke and row["alignment"]["survival"] < 0.3:
                row["status"] = "halted: alignment survival under 0.3"
                save_row(name, row)
                raise RuntimeError(row["status"])
            del donor
            gc.collect()
            torch.cuda.empty_cache()
        if arm == "scrambled":
            row["scramble"] = {k: v for k, v in scramble_module(graft.part, seed + 100).items()
                               if k in ("mode", "tensors_changed", "mean_abs_cosine", "rel_fro_error")}
        row["calib_rms"] = pgraft.calibrate_scale(base, tok, graft, a_train, cfg)
        lora = apply_lora(base, GRAFT_LORA_RANK, default_alpha(GRAFT_LORA_RANK))
        for p in lora:
            p.requires_grad_(True)
        for p in graft.enc.parameters():
            p.requires_grad_(True)
        glue_dec = list(graft.dec.parameters())
        glue_gate = list(graft.token_gate.parameters())
        glue_enc = list(graft.enc.parameters())
        for p in glue_dec + glue_gate:
            p.requires_grad_(True)
        if arm == "identity":
            groups = [{"params": glue_enc + glue_dec, "lr": lr["glue"]}, {"params": glue_gate, "lr": lr["gate"]},
                      {"params": lora, "lr": lr["lora"]}]
        else:
            groups = [{"params": glue_dec, "lr": lr["glue"]}, {"params": glue_gate, "lr": lr["gate"]},
                      {"params": glue_enc, "lr": lr["enc2"]}, {"params": lora, "lr": lr["lora"]}]
        row["glue_params"] = pgraft.count(glue_enc + glue_dec + glue_gate)
        row["lora_rank"] = GRAFT_LORA_RANK
    else:
        target = glue_budget + 0
        if arm == "lora_match":
            # Match the graft's total trainable: glue plus the graft arms' LoRA.
            probe_rank = GRAFT_LORA_RANK
            from wsp.lora import params_per_rank

            target = glue_budget + probe_rank * params_per_rank(base)
            rank = rank_for_budget(base, target)
            lr_name = "lora"
        elif arm == "lora_big":
            rank, lr_name = BIG_RANK, "lora_big"
        if arm in ("lora_match", "lora_big"):
            lora = apply_lora(base, rank, default_alpha(rank))
            for p in lora:
                p.requires_grad_(True)
            row["lora_rank"] = rank
            groups = [{"params": lora, "lr": lr[lr_name]}]
        elif arm == "fullft":
            params = [p for p in base.parameters()]
            for p in params:
                p.requires_grad_(True)
            groups = [{"params": params, "lr": lr["fullft"]}]
        else:
            raise ValueError(arm)

    trainable_params = [p for g in groups for p in g["params"]]
    row["trainable"] = sum(p.numel() for p in {id(p): p for p in trainable_params}.values())
    print(f"[{name}] trainable {row['trainable']}", flush=True)
    save_row(name, dict(row, status="stage1 done"))

    # ---- stage 2: SFT on the donor's verified solutions
    log, t = sft_phase(base, tok, groups, pairs, sft_steps, seed, name + "/sft")
    tokens += t
    row["sft_first50"] = sum(r["loss"] for r in log[:50]) / len(log[:50])
    row["sft_last50"] = sum(r["loss"] for r in log[-50:]) / len(log[-50:])
    if graft is not None:
        row["gate_after_sft"] = gate_mean(graft)
        print(f"[{name}] mean gate {row['gate_after_sft']}", flush=True)
    save_row(name, dict(row, status="sft done"))
    if not smoke:
        row["after_sft"] = evaluate_light(base, tok, items)
        save_row(name, dict(row, status="sft scored"))

    # ---- stage 3: verified self-training rounds
    rft = []
    for rnd in range(rft_rounds):
        rng = random.Random(seed * 100 + rnd)
        picks = []
        for skill in ("gsm8k", "math500"):
            chosen = rng.sample(pool[skill], min(rft_prompts[skill], len(pool[skill])))
            for it in chosen:
                picks.append((skill, it))
        prompts, meta = [], []
        for skill, it in picks:
            u = se.user_text(skill, it)
            for _ in range(2):
                prompts.append(se.render_prompt(tok, u))
                meta.append((skill, it, u))
        base.eval()
        torch.manual_seed(seed * 100 + rnd)
        outs = sample_texts(base, tok, prompts, RFT_SAMPLE_TOKENS)
        good, seen, tried = [], set(), 0
        for (skill, it, u), text in zip(meta, outs):
            if text is None:
                continue
            tried += 1
            rec = se.grade_outputs(skill, [it], [text])[0]
            if rec["correct"] and (it["id"], text) not in seen:
                seen.add((it["id"], text))
                good.append((u, text.strip()))
        info = {"round": rnd, "sampled": len(meta), "finished": tried, "correct": len(good)}
        print(f"[{name}] rft {info}", flush=True)
        mix = good + random.Random(rnd).sample(pairs, min(len(good), len(pairs)))
        if len(mix) >= 8:
            log, t = sft_phase(base, tok, [dict(g, lr=g["lr"] * 0.5) for g in groups], mix, rft_steps,
                               seed + 7 + rnd, f"{name}/rft{rnd}")
            tokens += t
            info["last50"] = sum(r["loss"] for r in log[-50:]) / len(log[-50:])
        rft.append(info)
        row["rft"] = rft
        save_row(name, dict(row, status=f"rft {rnd} done"))
    row["rft"] = rft

    # ---- score
    base.eval()
    row["scores"] = evaluate(base, tok, items, smoke)
    if graft is not None:
        row["gate_final"] = gate_mean(graft)
        with pgraft.grafts_off(base):
            off = dict(items, name=name + "/graft_off")
            row["scores_graft_off"] = evaluate_light(base, tok, off, smoke)
    seen = {}
    for p in base.parameters():
        seen[id(p)] = p
    row["params_total"] = sum(p.numel() for p in seen.values())
    row["params_run"] = se.model_params(base)[1]
    row["train_tokens"] = tokens
    row["tok_s"] = round(se.decode_speed(base, tok), 1)
    row["gpu_hours"] = round((time.time() - start) / 3600, 3)
    row["status"] = "done"
    save_row(name, row)
    print("ROW", json.dumps(row))
    return row


def evaluate_light(model, tok, items, smoke=False):
    """GSM8K test and the dev slices only, for mid-run checks."""
    from wsp import skills_eval as se

    lim = SMOKE["eval"] if smoke else None
    out = {}
    r = se.evaluate_skill(model, tok, "gsm8k", items=items["dev_gsm8k"], batch_size=50, limit=lim)
    out["dev_gsm8k"] = round(r["score"], 4)
    print(f"EVAL {items['name']} light {out}", flush=True)
    return out


@app.local_entrypoint()
def main(stage: str = "run", only: str = "", smoke: bool = False, tag: str = "", overrides: str = ""):
    ov = json.loads(overrides) if overrides else None
    if stage == "data":
        print(donor_data.remote(4, smoke))
        return
    cells = [(a, s) for a in ["base", "donor"] + ARMS for s in (SEEDS if a not in ("base", "donor") else [0])]
    if only:
        cells = [(x.split(":")[0], int(x.split(":")[1])) for x in only.split(",")]
    if not cells:
        raise SystemExit("zero cells")
    calls = [(a, s, smoke or stage == "smoke", tag, ov) for a, s in cells]
    for row in cell.starmap(calls, return_exceptions=True):
        if isinstance(row, Exception):
            print("cell failed:", repr(row)[:800])
        else:
            print("ROW", json.dumps(row))
