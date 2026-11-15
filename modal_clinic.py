"""Clinic demo: self-training of a slimmed medical MoE on multiple-choice questions (copy of modal_rft.py).

Start models: the keep-64 clinic slim (rebuilt as clinic/clinic_64), a keep-48 clinic slim built here from the round 28
clinic saliency stats (build_k48), and the full Qwen3-30B-A3B-Instruct-2507 as reference and teacher.
Train items: MedMCQA train, MedQA-USMLE train, PubMedQA labeled (non-test) and artificial train, with the
reasoning prompt of wsp.clinic and a letter reward. Every item that shares a 13-gram with any reported test set
(MedMCQA val 1000, MedQA test, PubMedQA test 500, MMLU medical six, MMLU general 1000) is dropped.
Scores come from test items only: letter log-probability ("lp") and chain of thought ("cot") on each set.
Rounds pick the model by a held-out slice of the train items, never by a test score.

Everything lives under /vol/clinic/c1/ on the volume "wsp-models" of the active profile.

    modal run modal_clinic.py --stage verify
    modal run modal_clinic.py --stage data
    modal run modal_clinic.py --stage build48 [--keep 64]
    modal run modal_clinic.py --stage score --run k64          # one cheap cell first
    modal run --detach modal_clinic.py --stage drive --run k64 --rounds 2
"""

import json
import math
import os
import random
import time

import modal

from wsp import rft

app = modal.App("wsp-clinic")
volume = modal.Volume.from_name("wsp-models", create_if_missing=True)

VLLM_VERSION = "0.32.0"

# The vLLM image of modal_serve.py (ran 2026-11-15).
vllm_image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.12")
    .env({"CUDA_HOME": "/usr/local/cuda"})
    .pip_install(
        f"vllm=={VLLM_VERSION}", "accelerate", "huggingface_hub", "requests",
        "safetensors", "zstandard", "numpy", "datasets", "sentencepiece", "protobuf",
    )
    .env({"HF_HOME": "/vol/hf", "HF_HUB_DISABLE_XET": "1"})
    .add_local_python_source("wsp")
    .add_local_file("modal_serve.py", "/root/modal_serve.py")
)
# The training image of modal_round19.py (ran 2026-11-14).
train_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch==2.7.1", "transformers==4.57.1", "accelerate", "huggingface_hub", "requests",
        "safetensors", "zstandard", "numpy", "datasets", "sentencepiece", "protobuf",
    )
    .env({"HF_HOME": "/vol/hf", "HF_HUB_DISABLE_XET": "1"})
    .add_local_python_source("wsp")
)
cpu_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch==2.7.1", "transformers==4.57.1", "accelerate", "huggingface_hub", "requests",
                 "safetensors", "zstandard", "numpy", "datasets==4.0.0", "pyarrow", "sentencepiece", "protobuf")
    .env({"HF_HOME": "/vol/hf", "HF_HUB_DISABLE_XET": "1"})
    .add_local_python_source("wsp")
)

BASE = "Qwen/Qwen3-30B-A3B-Instruct-2507"
SLIM_ROOT = "/vol/slim_models"
ROOT = "/vol/clinic/c1"
DATA = ROOT + "/data"
STARTS = {"k64": "clinic/clinic_64", "k48": "clinic/clinic_48", "full": BASE}

SAMPLE_TOKENS = 400
COT_TOKENS = 400
CHUNK = 500
GPU_USD_PER_HOUR = 4.5


def log(tag, message):
    print(f"[{tag}] {message}", flush=True)


def safe_reload():
    try:
        volume.reload()
    except Exception as error:
        print("volume reload skipped:", repr(error)[:200], flush=True)


def commit():
    try:
        volume.commit()
    except Exception as error:
        print("volume commit failed:", repr(error)[:200], flush=True)


def write_json(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=1, default=str)
    os.replace(tmp, path)


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    os.replace(tmp, path)


def model_dir(ref):
    """An absolute directory, a directory under the slim root, or else a Hub id."""
    if ref.startswith("/"):
        return ref
    if os.path.isdir(os.path.join(SLIM_ROOT, ref)):
        return os.path.join(SLIM_ROOT, ref)
    return ref


# ---------------------------------------------------------------- data


MEDMCQA_TRAIN_CAP = 6000
MEDQA_TRAIN_CAP = 10000
PUBMED_ARTIFICIAL_CAP = 1500
SCORE_SETS = ["medmcqa", "medqa", "pubmedqa", "mmlu_med6", "mmlu_general_ex_med6"]


def load_test_sets():
    """{name: items} for every reported clinic test set (wsp.skills_eval)."""
    from wsp import skills_eval

    return {name: skills_eval.load_items(name) for name in SCORE_SETS}


@app.function(image=cpu_image, volumes={"/vol": volume}, cpu=8, memory=32768, timeout=2 * 3600)
def verify_data():
    """CPU: load every test set and every train source at full size, print counts and two raw items of each."""
    from datasets import load_dataset

    out = {}
    sets = load_test_sets()
    for name, items in sets.items():
        out[name] = len(items)
        print(f"test {name} n={len(items)} gold counts "
              f"{[sum(1 for i in items if i['gold'] == g) for g in range(4)]}", flush=True)
        for item in items[:1] + items[-1:]:
            print(f"test {name} {json.dumps(item)[:500]}", flush=True)
    for spec in (("openlifescienceai/medmcqa", None, "train"), ("GBaker/MedQA-USMLE-4-options", None, "train"),
                 ("qiaojin/PubMedQA", "pqa_labeled", "train"), ("qiaojin/PubMedQA", "pqa_artificial", "train")):
        ds = load_dataset(spec[0], spec[1], split=spec[2]) if spec[1] else load_dataset(spec[0], split=spec[2])
        out[spec[0] + ":" + str(spec[1])] = len(ds)
        print(f"train {spec} n={len(ds)} columns {ds.column_names} row0 {json.dumps(ds[0], default=str)[:600]}", flush=True)
    return out


@app.function(image=cpu_image, volumes={"/vol": volume}, cpu=16, memory=65536, timeout=3 * 3600)
def build_data():
    """Build the clinic train items, decontaminate against every test set, split 5 percent held out. Writes DATA."""
    from datasets import load_dataset

    from wsp import clinic, skills_eval

    safe_reload()
    started = time.time()
    sets = load_test_sets()
    index = clinic.test_index(sets)
    stats = {"format": rft.RFT_FORMAT, "test_sets": {k: len(v) for k, v in sets.items()}, "test_index_grams": len(index)}
    log("data", f"test index {len(index)} 13-grams from {stats['test_sets']}")
    sources = {}
    items = []
    for i, row in enumerate(load_dataset("openlifescienceai/medmcqa", split="train")):
        item = clinic.medmcqa_train_item(row, i)
        if item:
            items.append(item)
    sources["medmcqa_train"] = items
    items = [clinic.medqa_train_item(row, i) for i, row in enumerate(load_dataset("GBaker/MedQA-USMLE-4-options", split="train"))]
    sources["medqa_train"] = items
    test_pubids = {str(k) for k in skills_eval.pubmedqa_test_ids()}
    items = []
    for row in load_dataset("qiaojin/PubMedQA", "pqa_labeled", split="train"):
        if str(row["pubid"]) in test_pubids:
            continue
        item = clinic.pubmedqa_train_item(row, "pubmedqa_labeled")
        if item:
            items.append(item)
    sources["pubmedqa_labeled"] = items
    items = []
    for row in load_dataset("qiaojin/PubMedQA", "pqa_artificial", split="train"):
        if str(row["pubid"]) in test_pubids:
            continue
        item = clinic.pubmedqa_train_item(row, "pubmedqa_artificial")
        if item:
            items.append(item)
        if len(items) >= 40000:
            break
    sources["pubmedqa_artificial"] = items
    for name, rows in sources.items():
        log("data", f"{name}: {len(rows)} candidate items")
    caps = {"medmcqa_train": MEDMCQA_TRAIN_CAP, "medqa_train": MEDQA_TRAIN_CAP, "pubmedqa_labeled": 0,
            "pubmedqa_artificial": PUBMED_ARTIFICIAL_CAP}
    final, source_stats = clinic.build_train(sources, index, caps, seed=0)
    stats["sources"] = source_stats
    for name, s in source_stats.items():
        log("data", f"{name}: {s}")
    train, held = rft.split_items(final, 0.05, salt="clinic-v1")
    stats["train"] = len(train)
    stats["heldout"] = len(held)
    stats["gold_letter_counts_train"] = {l: sum(1 for i in train if i["gold"] == l) for l in "ABCD"}
    stats["seconds"] = round(time.time() - started)
    write_jsonl(DATA + "/items_train.jsonl", train)
    write_jsonl(DATA + "/items_heldout.jsonl", held)
    write_json(DATA + "/test_index.json", sorted(index))
    write_json(DATA + "/stats.json", stats)
    commit()
    return stats


V28 = "v28a"
STATS_DIR = "/vol/slim/round28"


@app.function(image=train_image, volumes={"/vol": volume}, cpu=16, memory=245760, timeout=3 * 3600,
              single_use_containers=True)
def build_k48(keep=48):
    """Keep `keep` experts per layer of the full model, ranked by the round 28 clinic saliency stats (healdata
    medicine pool plus MedMCQA train; cached on the volume), no heal. Saves to /vol/slim_models/clinic/clinic_<keep>."""
    import glob

    import torch
    import transformers

    from wsp import slim

    safe_reload()
    out = f"{SLIM_ROOT}/clinic/clinic_{keep}"
    if os.path.exists(os.path.join(out, "wsp_slim.json")):
        return {"exists": out}
    files = sorted(glob.glob(f"{STATS_DIR}/{V28}_clinic_512_384_*.pt"))
    if len(files) != 1:
        raise SystemExit(f"expected one clinic stats file, found {files}")
    stats = slim.load_stats(files[0])
    tokenizer = transformers.AutoTokenizer.from_pretrained(BASE)
    model = transformers.AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.bfloat16, device_map="cpu").eval()
    student, report = slim.slim_moe(model, None, None, None, rule="saliency", keep=keep, mass=None, seed=0,
                                    skill_stats=stats, share_experts=True)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    manifest = slim.save_slim(student, out + ".tmp", tokenizer)
    os.replace(out + ".tmp", out)
    commit()
    info = {"stats": files[0], "keep": keep, "params_total_after": report["params_total_after"],
            "params_run_after": report["params_run_after"], "plain_hf": manifest["plain_hf"],
            "disk_bytes": slim.directory_bytes(out)}
    write_json(out + "_build.json", info)
    commit()
    log("build_k48", json.dumps(info))
    return info


# ---------------------------------------------------------------- sampling and grading


@app.function(image=vllm_image, volumes={"/vol": volume}, gpu="H200", memory=131072, cpu=16, timeout=6 * 3600,
              single_use_containers=True)
def sample_shard(model_ref, out_dir, shard, nshards, k=8, temperature=0.8, limit=0, items_name="items_train.jsonl"):
    """Sample k answers for every prompt of one shard and grade them. Writes out_dir/s<shard>_c<i>.jsonl per chunk."""
    import transformers
    from vllm import LLM, SamplingParams

    from wsp import skills_eval

    safe_reload()
    tag = f"sample s{shard}/{nshards} {os.path.basename(out_dir)}"
    items = read_jsonl(os.path.join(DATA, items_name))[shard::nshards]
    if limit:
        items = items[:limit]
    chunks = [items[i:i + CHUNK] for i in range(0, len(items), CHUNK)]
    todo = [i for i in range(len(chunks)) if not os.path.exists(os.path.join(out_dir, f"s{shard}_c{i}.jsonl"))]
    if not todo:
        log(tag, "all chunks exist")
        return {"shard": shard, "items": len(items), "skipped": True}
    path = model_dir(model_ref)
    tokenizer = transformers.AutoTokenizer.from_pretrained(BASE)
    started = time.time()
    llm = LLM(model=path, tokenizer=BASE, dtype="bfloat16", seed=shard, gpu_memory_utilization=0.90,
              max_model_len=4096, enable_prefix_caching=True, max_num_seqs=256)
    log(tag, f"loaded {path} in {time.time() - started:.0f} s")
    params = SamplingParams(n=k, temperature=temperature, top_p=0.95, max_tokens=SAMPLE_TOKENS, seed=None)
    task = None
    total = {"samples": 0, "passed": 0}
    for i in todo:
        chunk = chunks[i]
        prompts = [skills_eval.render_prompt(tokenizer, it["user"]) for it in chunk]
        t0 = time.time()
        outs = llm.generate(prompts, params, use_tqdm=False)
        t1 = time.time()
        flat_items, flat_texts, cut = [], [], []
        for item, out in zip(chunk, outs):
            for o in out.outputs:
                flat_items.append(item)
                flat_texts.append(o.text)
                cut.append(o.finish_reason == "length")
        rewards = rft.rewards_for(flat_items, flat_texts, workers=16)
        rows, pos = [], 0
        for item, out in zip(chunk, outs):
            samples = []
            for o in out.outputs:
                reward = 0.0 if cut[pos] else rewards[pos]
                samples.append({"text": o.text, "reward": reward})
                pos += 1
            rows.append({"id": item["id"], "samples": samples})
        passed = sum(1 for r in rows for s in r["samples"] if s["reward"] >= 1.0)
        total["samples"] += len(flat_texts)
        total["passed"] += passed
        write_jsonl(os.path.join(out_dir, f"s{shard}_c{i}.jsonl"), rows)
        commit()
        log(tag, f"chunk {i + 1}/{len(chunks)}: {len(chunk)} prompts, {passed}/{len(flat_texts)} samples pass "
                 f"({passed / max(1, len(flat_texts)):.3f}); generate {t1 - t0:.0f} s, grade {time.time() - t1:.0f} s")
    return {"shard": shard, "items": len(items), **total, "seconds": round(time.time() - started)}


def read_samples(sample_dir):
    """{item id: [samples]} from every chunk file in a directory."""
    out = {}
    for name in sorted(os.listdir(sample_dir)):
        if name.endswith(".jsonl"):
            for row in read_jsonl(os.path.join(sample_dir, name)):
                out[row["id"]] = row["samples"]
    return out


# ---------------------------------------------------------------- training


@app.function(image=train_image, volumes={"/vol": volume}, gpu="H200", memory=262144, timeout=10 * 3600,
              single_use_containers=True)
def train_round(base_dir, sft_path, out_dir, epochs=2, lr=1e-4, router_lr=1e-5, rank=64, max_tokens=8192,
                max_seqs=32, accum=4, max_len=1536, max_steps=0, seed=0, train_routers=True, warmup=10,
                grad_ckpt=False):
    """LoRA SFT of a slim MoE on verified answers, merged and saved to out_dir. Loss on answer tokens only."""
    import torch
    import transformers

    from wsp import slim
    from wsp.lora import apply_lora, merge_lora

    safe_reload()
    tag = "train " + os.path.basename(out_dir)
    if os.path.exists(os.path.join(out_dir, "wsp_slim.json")):
        log(tag, "model exists, skipping")
        return json.load(open(os.path.join(out_dir, "train_log.json")))
    started = time.time()
    base = model_dir(base_dir)
    tokenizer = transformers.AutoTokenizer.from_pretrained(base)
    rows = read_jsonl(sft_path)
    examples = []
    skipped = 0
    for row in rows:
        enc = rft.encode_sft(tokenizer, row["user"], row["assistant"], max_len)
        if enc is None:
            skipped += 1
        else:
            examples.append((enc[0], enc[1], float(row.get("weight", 1.0))))
    log(tag, f"{len(examples)} examples ({skipped} longer than {max_len} tokens skipped), "
             f"{sum(len(e[0]) for e in examples)} tokens")
    torch.manual_seed(seed)
    model = slim.load_slim(base, dtype=torch.bfloat16, device="cuda")
    model.config.use_cache = False
    for p in model.parameters():
        p.requires_grad_(False)
    lora_params = apply_lora(model, rank, targets=slim.ATTENTION_TARGETS)
    router_params, restore = [], None
    if train_routers:
        router_params, restore = slim.prepare_routers(model, slim.family_of(model))
    groups = [{"params": lora_params, "lr": lr}]
    if router_params:
        groups.append({"params": router_params, "lr": router_lr})
    optimizer = torch.optim.AdamW(groups, weight_decay=0.0)
    base_lrs = [g["lr"] for g in groups]
    if grad_ckpt:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train()
    batches = []
    for epoch in range(epochs):
        batches.extend(rft.make_batches(examples, max_tokens, max_seqs, seed + epoch))
    steps = math.ceil(len(batches) / accum)
    if max_steps:
        steps = min(steps, max_steps)
    log(tag, f"{len(batches)} microbatches, {steps} optimizer steps, lora params {sum(p.numel() for p in lora_params):,},"
             f" router params {sum(p.numel() for p in router_params):,}")
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    losses = []
    for step in range(steps):
        group = batches[step * accum:(step + 1) * accum]
        total_tokens = sum(sum(1 for x in examples[i][1] if x != -100) for b in group for i in b)
        scale = min(1.0, (step + 1) / warmup) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * step / max(1, steps))))
        for g, b in zip(optimizer.param_groups, base_lrs):
            g["lr"] = b * scale
        step_loss = 0.0
        for b in group:
            longest = max(len(examples[i][0]) for i in b)
            ids = torch.full((len(b), longest), pad, dtype=torch.long)
            lab = torch.full((len(b), longest), -100, dtype=torch.long)
            att = torch.zeros((len(b), longest), dtype=torch.long)
            for r, i in enumerate(b):
                n = len(examples[i][0])
                ids[r, :n] = torch.tensor(examples[i][0])
                lab[r, :n] = torch.tensor(examples[i][1])
                att[r, :n] = 1
            ids, lab, att = ids.cuda(), lab.cuda(), att.cuda()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                hidden = model.model(input_ids=ids, attention_mask=att, use_cache=False).last_hidden_state
                mask = lab[:, 1:] != -100
                logits = model.lm_head(hidden[:, :-1][mask])
            token_loss = torch.nn.functional.cross_entropy(logits.float(), lab[:, 1:][mask], reduction="none")
            weights = torch.tensor([examples[i][2] for i in b], dtype=torch.float32, device=ids.device)
            weights = weights[:, None].expand(-1, longest - 1)[mask]
            loss = (token_loss * weights).sum() / total_tokens
            loss.backward()
            step_loss += float(loss.detach())
        torch.nn.utils.clip_grad_norm_(lora_params + router_params, 1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        losses.append(step_loss)
        if step % 10 == 0 or step == steps - 1:
            log(tag, f"step {step + 1}/{steps} loss {step_loss:.4f} lr x{scale:.3f} {time.time() - started:.0f} s")
    model.eval()
    if restore is not None:
        restore()
    merge_lora(model)
    tmp = out_dir + ".tmp"
    import shutil

    os.makedirs(tmp, exist_ok=True)
    model.save_pretrained(tmp, max_shard_size="5GB")
    tokenizer.save_pretrained(tmp)
    shutil.copy(os.path.join(base, slim.MANIFEST), os.path.join(tmp, slim.MANIFEST))
    report = {"examples": len(examples), "skipped_long": skipped, "steps": steps, "epochs": epochs, "lr": lr,
              "router_lr": router_lr, "rank": rank, "accum": accum, "max_tokens": max_tokens,
              "train_routers": train_routers, "loss_first": sum(losses[:5]) / max(1, len(losses[:5])),
              "loss_last": sum(losses[-5:]) / max(1, len(losses[-5:])), "seconds": round(time.time() - started)}
    write_json(os.path.join(tmp, "train_log.json"), report)
    os.replace(tmp, out_dir)
    commit()
    log(tag, f"saved {out_dir}: {report}")
    return report


# ---------------------------------------------------------------- scoring


def letter_scores(llm, tokenizer, items):
    """Letter-argmax scoring under vLLM (the rule of wsp.skills_eval.score_choices), restricted to the item's own
    number of choices (3 for PubMedQA)."""
    from vllm import SamplingParams

    from wsp import skills_eval

    ids = skills_eval.letter_ids(tokenizer)
    params = SamplingParams(temperature=0.0, max_tokens=1, logprobs=20)
    outs = llm.generate([it["prompt"] for it in items], params, use_tqdm=False)
    correct = []
    for item, out in zip(items, outs):
        lp = out.outputs[0].logprobs[0]
        n = len(item.get("choices") or ids[:4])
        scores = [lp[t].logprob if t in lp else -1e9 for t in ids[:n]]
        pred = max(range(n), key=lambda j: scores[j])
        correct.append(pred == item["gold"])
    return correct


def cot_scores(llm, tokenizer, items, tokens=COT_TOKENS):
    """Chain-of-thought scoring: the clinic reasoning prompt, greedy, the last 'Answer: X' letter. Returns
    (correct list, outputs, cut-off count)."""
    from vllm import SamplingParams

    from wsp import clinic, skills_eval

    prompts = [skills_eval.render_prompt(tokenizer, clinic.reason_user(it["question"], it["choices"])) for it in items]
    outs = llm.generate(prompts, SamplingParams(temperature=0.0, max_tokens=tokens), use_tqdm=False)
    texts = [o.outputs[0].text for o in outs]
    cut = sum(1 for o in outs if o.outputs[0].finish_reason == "length")
    correct = [rft.letter_of(t) == "ABCD"[it["gold"]] for it, t in zip(items, texts)]
    return correct, texts, cut


@app.function(image=vllm_image, volumes={"/vol": volume}, gpu="H200", memory=131072, cpu=16, timeout=4 * 3600,
              single_use_containers=True)
def score_model(model_ref, name, out_path, speed=False, suites="lp,cot,heldout", limit=0, gpu_label=""):
    """Scores under vLLM on one GPU, plus optionally decode speed. Writes one row to out_path.

    suites: lp (letter log-probability on every clinic set), cot (reasoning prompt, greedy, parsed letter on every
    clinic set), heldout (the held-out train slice, cot reward)."""
    import gc

    import torch
    import transformers
    import vllm
    from vllm import LLM, SamplingParams

    from wsp import skills_eval

    safe_reload()
    started = time.time()
    path = model_dir(model_ref)
    tokenizer = transformers.AutoTokenizer.from_pretrained(BASE)
    row = {"cell": name, "model": model_ref, "vllm": vllm.__version__, "gpu": torch.cuda.get_device_name(0),
           "suites": suites, "lp": {}, "cot": {}, "n": {}, "correct": {"lp": {}, "cot": {}}, "scores": {},
           "limit": limit, "error": None}
    wanted = suites.split(",")
    try:
        import modal_serve

        t0 = time.time()
        llm = LLM(model=path, tokenizer=BASE, dtype="bfloat16", seed=0, gpu_memory_utilization=0.90,
                  max_model_len=6144, enable_prefix_caching=False, max_num_seqs=128)
        row["load_seconds"] = round(time.time() - t0, 1)
        log(name, f"loaded {path} in {row['load_seconds']} s")
        if "lp" in wanted or "cot" in wanted:
            sets = load_test_sets()
            for set_name, items in sets.items():
                if limit:
                    items = items[:limit]
                row["n"][set_name] = len(items)
                if "lp" in wanted:
                    correct = [bool(c) for c in letter_scores(llm, tokenizer, items)]
                    row["lp"][set_name] = sum(correct) / len(correct)
                    row["correct"]["lp"][set_name] = correct
                    log(name, f"lp {set_name} {sum(correct)}/{len(correct)} = {row['lp'][set_name]:.4f}")
                if "cot" in wanted:
                    correct, texts, cut = cot_scores(llm, tokenizer, items)
                    row["cot"][set_name] = sum(correct) / len(correct)
                    row["correct"]["cot"][set_name] = [bool(c) for c in correct]
                    row.setdefault("cot_cut", {})[set_name] = cut
                    row.setdefault("raw", {})[set_name] = [{"id": it["id"], "gold": "ABCD"[it["gold"]], "output": t[:500]}
                                                           for it, t in zip(items[:3], texts[:3])]
                    mean_len = sum(len(t) for t in texts) / len(texts)
                    row.setdefault("cot_chars", {})[set_name] = round(mean_len)
                    log(name, f"cot {set_name} {sum(correct)}/{len(correct)} = {row['cot'][set_name]:.4f} "
                              f"cut {cut} mean chars {mean_len:.0f}")
        if "heldout" in wanted:
            items = read_jsonl(os.path.join(DATA, "items_heldout.jsonl"))
            if limit:
                items = items[:limit]
            prompts = [skills_eval.render_prompt(tokenizer, it["user"]) for it in items]
            outs = llm.generate(prompts, SamplingParams(temperature=0.0, max_tokens=COT_TOKENS), use_tqdm=False)
            rewards = rft.rewards_for(items, [o.outputs[0].text for o in outs], workers=8)
            row["scores"]["heldout"] = sum(rewards) / len(rewards)
            row["scores"]["heldout_n"] = len(rewards)
            by_source = {}
            for it, w in zip(items, rewards):
                by_source.setdefault(it["source"], []).append(w)
            row["heldout_by_source"] = {s: [sum(v) / len(v), len(v)] for s, v in by_source.items()}
            log(name, f"heldout {sum(rewards):.0f}/{len(rewards)} = {row['scores']['heldout']:.4f} {row['heldout_by_source']}")
        if speed:
            prompts = modal_serve.build_prompts(tokenizer, modal_serve.PROMPTS)
            prompt_ids = modal_serve.encode_prompts(tokenizer, prompts)

            def generate(id_lists, max_tokens):
                p = SamplingParams(temperature=0.0, max_tokens=max_tokens, ignore_eos=True)
                llm.generate([{"prompt_token_ids": ids} for ids in id_lists], p, use_tqdm=False)

            generate(prompt_ids[:modal_serve.WARMUP_PROMPTS], 16)
            row["decode_tps"] = {}
            row["prefill_tps"] = modal_serve.measure_prefill(generate, prompt_ids)
            for batch in modal_serve.BATCHES:
                tps, first = modal_serve.measure_decode(generate, prompt_ids, batch, modal_serve.GEN_LEN)
                row["decode_tps"][str(batch)] = tps
                if batch == 1:
                    row["ttft_ms_median"], row["ttft_ms_p90"] = modal_serve.ttft_stats(first)
                log(name, f"decode batch {batch}: {tps} tok/s")
            row["gpu_after_load_gib"] = modal_serve.gpu_used_gib()
            if os.path.isdir(path):
                row["weights_gib"] = round(modal_serve.weights_gib(path), 2)
        del llm
        gc.collect()
    except Exception as error:
        import traceback

        traceback.print_exc()
        row["error"] = repr(error)[:500]
        log(name, "failed: " + row["error"])
    row["seconds"] = round(time.time() - started)
    write_json(out_path, row)
    commit()
    return row


# ---------------------------------------------------------------- the driver


def sft_for_round(items, student, teacher, index, max_rows=0, seed=0):
    rows, stats = rft.build_sft(items, student, teacher)
    rates = {i: sum(s["reward"] for s in v) / max(1, len(v)) for i, v in student.items()}
    stats["rows_before_cap"] = len(rows)
    rows = rft.cap_rows(rows, rates, max_rows, seed)
    clean = []
    dropped = 0
    ok_text = rft.drop_sample_overlap([{"text": r["assistant"], "row": r} for r in rows], index)
    clean = [s["row"] for s in ok_text[0]]
    dropped = ok_text[1]
    stats["dropped_overlap"] = dropped
    stats["rows"] = len(clean)
    return clean, stats


@app.function(image=cpu_image, volumes={"/vol": volume}, cpu=8, memory=32768, timeout=24 * 3600)
def drive(run="k64", rounds=3, shards=4, k=8, teacher_k=4, epochs=2, train_routers=True, smoke=False, first_round=1,
          teacher_from="", max_rows=9000):
    """Run baseline scoring, teacher sampling, and rounds of sample, train, score. Resumes from the volume."""
    safe_reload()
    start = STARTS[run]
    base = os.path.join(SLIM_ROOT, start)
    run_dir = f"{ROOT}/{run}" + ("_smoke" if smoke else "")
    limit = 64 if smoke else 0
    sh = 1 if smoke else shards
    items = read_jsonl(DATA + "/items_train.jsonl")
    if smoke:
        items = items[:64]
    index = set(json.load(open(DATA + "/test_index.json")))
    summary_path = run_dir + "/summary.json"
    summary = json.load(open(summary_path)) if os.path.exists(summary_path) else {"rounds": {}, "format": rft.RFT_FORMAT}
    suites = "lp,cot,heldout"

    def save():
        write_json(summary_path, summary)
        commit()

    # Row 0: the start model, scored first. The full model is the reference.
    jobs = []
    if not smoke and not os.path.exists(f"{run_dir}/score_start.json"):
        jobs.append(score_model.spawn(start, f"{run}_start", f"{run_dir}/score_start.json", False, suites, limit))
    if not os.path.exists(f"{run_dir}/score_full.json") and not smoke and not teacher_from:
        jobs.append(score_model.spawn(BASE, f"{run}_full30b", f"{run_dir}/score_full.json", False, suites, limit))
    teacher_dir = teacher_from or f"{run_dir}/teacher_samples"
    todo = [s for s in range(sh)]
    if teacher_from:
        t_jobs = []
    elif smoke:
        t_jobs = [sample_shard.spawn(BASE, teacher_dir, 0, 1, teacher_k, 0.8, 64)]
    else:
        t_jobs = [sample_shard.spawn(BASE, teacher_dir, s, sh, teacher_k, 0.8, limit) for s in todo]
    for job in jobs + t_jobs:
        job.get()
    safe_reload()
    teacher = read_samples(teacher_dir)
    summary["teacher"] = rft.pass_rate_stats({i: [s["reward"] for s in v] for i, v in teacher.items()})
    save()
    log("drive", f"teacher pass stats {summary['teacher']}")

    current = start
    for r in range(1, rounds + 1):
        rd = f"{run_dir}/r{r}"
        key = f"r{r}"
        entry = summary["rounds"].setdefault(key, {})
        if r < first_round:
            current = f"{rd}/model" if os.path.exists(f"{rd}/model/wsp_slim.json") else current
            continue
        # sample from the current model
        s_jobs = [sample_shard.spawn(current, rd + "/samples", s, sh, k, 0.8, limit) for s in range(sh)]
        results = [j.get() for j in s_jobs]
        safe_reload()
        student = read_samples(rd + "/samples")
        entry["student"] = rft.pass_rate_stats({i: [s["reward"] for s in v] for i, v in student.items()})
        rows, stats = sft_for_round([i for i in items if i["id"] in student], student, teacher, index, max_rows, r)
        write_jsonl(rd + "/sft.jsonl", rows)
        entry["sft"] = stats
        entry["sampled_from"] = current
        save()
        log("drive", f"round {r} sampled from {current}: {entry['student']} sft {stats}")
        if not rows:
            entry["error"] = "no verified rows"
            save()
            break
        rep = train_round.remote(base, rd + "/sft.jsonl", rd + "/model", epochs, 1e-4, 1e-5, 64, 8192, 32, 4, 1536,
                                 3 if smoke else 0, r, train_routers)
        entry["train"] = rep
        save()
        sc = score_model.remote(rd + "/model", f"{run}_r{r}", f"{rd}/score.json", False, suites, limit)
        entry["score"] = sc
        save()
        log("drive", f"round {r} scores {sc.get('scores')} {sc.get('error')}")
        current = rd + "/model"

    # Pick the round by the held-out train slice only.
    best = "start"
    start_path = f"{run_dir}/score_start.json"
    if not os.path.exists(start_path):
        summary["selected"] = None
        save()
        return summary
    best_value = json.load(open(start_path))["scores"].get("heldout", 0.0)
    for key, entry in summary["rounds"].items():
        sc = entry.get("score") or {}
        value = (sc.get("scores") or {}).get("heldout")
        if value is not None and value > best_value:
            best, best_value = key, value
    summary["selected"] = {"round": best, "heldout": best_value}
    save()
    log("drive", f"selected {best} by held-out train slice ({best_value:.4f})")
    return summary


# ---------------------------------------------------------------- GRPO-style iterations


@app.function(image=cpu_image, volumes={"/vol": volume}, cpu=8, memory=32768, timeout=24 * 3600)
def drive_pg(run="k64", from_model="", iterations=4, prompts_per_iter=1500, shards=3, k=8, lr=2e-5, max_prompts=600,
             tag="pg", smoke=False):
    """GRPO-style loop from a given model. Each iteration: sample k per prompt from the current policy, give each
    answer the group advantage (r - mean) / std, take a policy-gradient pass (LoRA r64 on attention, routers
    frozen, no KL, one epoch), merge, and score the held-out train slice. Not strictly on-policy after the
    first optimizer step of an iteration (no ratio clipping), which the small learning rate tolerates.
    Resumes from the volume; every iteration's rows are written there."""
    safe_reload()
    run_dir = f"{ROOT}/{run}_{tag}" + ("_smoke" if smoke else "")
    items = read_jsonl(DATA + "/items_train.jsonl")
    by_id = {i["id"]: i for i in items}
    rng = random.Random(0)
    order = list(range(len(items)))
    rng.shuffle(order)
    summary_path = run_dir + "/summary.json"
    summary = json.load(open(summary_path)) if os.path.exists(summary_path) else {"iterations": {}}
    current = from_model
    for it in range(1, iterations + 1):
        idir = f"{run_dir}/it{it}"
        entry = summary["iterations"].setdefault(f"it{it}", {})
        if os.path.exists(f"{idir}/model/config.json") and entry.get("score"):
            current = f"{idir}/model"
            continue
        chosen = [items[j] for j in order[(it - 1) * prompts_per_iter:it * prompts_per_iter]]
        if smoke:
            chosen = chosen[:64]
        write_jsonl(f"{idir}/items.jsonl", chosen)
        commit()
        sh = 1 if smoke else shards
        jobs = [sample_shard.spawn(current, idir + "/samples", s, sh, k, 1.0, 0, f"{idir}/items.jsonl") for s in range(sh)]
        for j in jobs:
            j.get()
        safe_reload()
        samples = read_samples(idir + "/samples")
        entry["policy"] = rft.pass_rate_stats({i: [s["reward"] for s in v] for i, v in samples.items()})
        rows = rft.pg_rows([by_id[i] for i in samples], samples, max_prompts, it)
        index = set(json.load(open(DATA + "/test_index.json")))
        clean = [r for r in rows if not rft.shares_ngram(r["assistant"], index)]
        entry["rows"] = len(clean)
        entry["rows_dropped_overlap"] = len(rows) - len(clean)
        write_jsonl(idir + "/pg.jsonl", clean)
        save = lambda: (write_json(summary_path, summary), commit())
        save()
        log("drive_pg", f"it{it}: policy {entry['policy']}, {len(clean)} rows")
        if not clean:
            entry["error"] = "no mixed groups"
            save()
            break
        rep = train_round.remote(current, idir + "/pg.jsonl", idir + "/model", 1, lr, 0.0, 64, 8192, 32, 4, 1536,
                                 3 if smoke else 0, it, False)
        entry["train"] = rep
        sc = score_model.remote(idir + "/model", f"{run}_{tag}_it{it}", f"{idir}/score.json", False, "heldout", 64 if smoke else 0)
        entry["score"] = sc
        save()
        log("drive_pg", f"it{it}: held-out {sc.get('scores')}")
        current = idir + "/model"
    best, best_value = None, -1.0
    for key, entry in summary["iterations"].items():
        value = ((entry.get("score") or {}).get("scores") or {}).get("heldout")
        if value is not None and value > best_value:
            best, best_value = key, value
    summary["selected"] = {"iteration": best, "heldout": best_value}
    write_json(summary_path, summary)
    commit()
    return summary


# ---------------------------------------------------------------- entrypoint


def estimate(run, rounds, shards, k, teacher_k, smoke=False):
    """Print the stage list and a GPU-hour estimate. Raise on zero cells."""
    stages = [("score_start", 1, 0.4), ("score_full30b", 1, 0.5), ("teacher_sample", shards, 0.6)]
    for r in range(1, rounds + 1):
        stages += [(f"r{r}_sample", shards, 0.6), (f"r{r}_train", 1, 2.0), (f"r{r}_score", 1, 0.4)]
    hours = sum(n * h for _, n, h in stages)
    print(f"run {run}: {len(stages)} stage groups, about {hours:.1f} GPU hours, about ${hours * GPU_USD_PER_HOUR:.0f}")
    for name, n, h in stages:
        print(f"  {name}: {n} container(s), about {h} h each")
    if not stages or run not in STARTS:
        raise SystemExit(f"zero cells or unknown run {run!r}; runs are {sorted(STARTS)}")
    return hours


# ---------------------------------------------------------------- side-by-side demo answers

DEMO_PROMPTS = [
    ("triage", "A nurse line call: a 3-year-old has had a fever of 39.4 C for two days, is drinking less than usual, "
               "and has had two wet diapers today. The child is alert but fussy. How urgent is this, what questions "
               "should be asked next, and what are the signs that mean the child must go to the emergency room now?"),
    ("drug interaction", "A 72-year-old patient takes warfarin for atrial fibrillation and the clinic wants to start "
                         "her on trimethoprim-sulfamethoxazole for a urinary infection. What is the interaction, what "
                         "should be monitored, and is there a safer choice?"),
    ("lab value meaning", "A 45-year-old woman has a TSH of 8.2 mIU/L with a normal free T4, and she reports fatigue "
                          "and weight gain. What does this result mean, and what is the usual next step?"),
    ("patient-friendly explanation", "Explain to a patient with newly diagnosed type 2 diabetes, in plain language at "
                                     "a sixth-grade reading level, what an A1C of 8.1 percent means and what they can "
                                     "do over the next three months."),
    ("differential", "A 34-year-old woman has had three weeks of fatigue, joint pain in both hands, a facial rash "
                     "that gets worse in sunlight, and mouth sores. List a differential diagnosis in order of "
                     "likelihood and the first tests to order."),
    ("dosing check", "Check this order: amoxicillin 500 mg by mouth every 8 hours for a 14 kg, 4-year-old child with "
                     "acute otitis media. Is the dose in a normal range, and what is the usual dose by weight?"),
    ("discharge instructions", "Write discharge instructions for an adult patient leaving the clinic after a "
                               "diagnosis of uncomplicated acute bronchitis. Include home care, medicines, and when "
                               "to return."),
    ("coding question", "A visit note says: established patient, 25 minutes of total time, one stable chronic illness "
                        "(hypertension) reviewed, medication refill given. Which E/M office visit level fits, and what "
                        "in the note supports it? Do not discuss insurance payment."),
]


@app.function(image=vllm_image, volumes={"/vol": volume}, gpu="H200", memory=131072, cpu=16, timeout=2 * 3600,
              single_use_containers=True)
def demo_outputs(model_ref, name, out_path, max_tokens=700):
    """Greedy answers of one model to the eight demo questions. Writes out_path (JSON)."""
    import transformers
    from vllm import LLM, SamplingParams

    from wsp import skills_eval

    safe_reload()
    tokenizer = transformers.AutoTokenizer.from_pretrained(BASE)
    llm = LLM(model=model_dir(model_ref), tokenizer=BASE, dtype="bfloat16", seed=0, gpu_memory_utilization=0.90,
              max_model_len=4096, max_num_seqs=16)
    prompts = [skills_eval.render_prompt(tokenizer, text) for _, text in DEMO_PROMPTS]
    outs = llm.generate(prompts, SamplingParams(temperature=0.0, max_tokens=max_tokens), use_tqdm=False)
    rows = [{"kind": kind, "question": text, "answer": o.outputs[0].text.strip(),
             "tokens": len(o.outputs[0].token_ids)} for (kind, text), o in zip(DEMO_PROMPTS, outs)]
    write_json(out_path, {"cell": name, "model": model_ref, "rows": rows})
    commit()
    log(name, f"demo answers written to {out_path}")
    return rows


@app.local_entrypoint()
def main(stage: str = "estimate", run: str = "k64", rounds: int = 3, shards: int = 4, k: int = 8, teacher_k: int = 4,
         smoke: bool = False, epochs: int = 2, first_round: int = 1, teacher_from: str = "",
         from_model: str = "", iterations: int = 4, tag: str = "pg", max_rows: int = 9000, keep: int = 48):
    if stage == "estimate":
        estimate(run, rounds, shards, k, teacher_k)
    elif stage == "verify":
        print(json.dumps(verify_data.remote(), indent=1))
    elif stage == "data":
        print(json.dumps(build_data.remote(), indent=1))
    elif stage == "build48":
        print(json.dumps(build_k48.remote(keep), indent=1))
    elif stage == "drive":
        estimate(run, rounds, shards, k, teacher_k)
        print(json.dumps(drive.remote(run, rounds, shards, k, teacher_k, epochs, True, smoke, first_round, teacher_from, max_rows), indent=1)[:4000])
    elif stage == "pg":
        if not from_model:
            raise SystemExit("--from-model is required for stage pg")
        print(json.dumps(drive_pg.remote(run, from_model, iterations, 1500, shards, k, 2e-5, 600, tag, smoke), indent=1)[:4000])
    elif stage == "score_model":
        row = score_model.remote(from_model, f"{run}_{tag}", f"{ROOT}/{run}_{tag}_final/score.json", False,
                                 "lp,cot,heldout", 0)
        print(json.dumps({k_: row.get(k_) for k_ in ("cell", "lp", "cot", "n", "scores", "cot_cut", "cot_chars", "heldout_by_source", "error")}, indent=1))
    elif stage == "score":
        row = score_model.remote(STARTS[run], f"{run}_start", f"{ROOT}/{run}/score_start.json", False,
                                 "lp,cot,heldout", 64 if smoke else 0)
        print(json.dumps({k_: row.get(k_) for k_ in ("cell", "lp", "cot", "n", "scores", "cot_cut", "cot_chars", "heldout_by_source", "error")}, indent=1))
    elif stage == "demo":
        if not from_model:
            raise SystemExit("--from-model is required for stage demo")
        rows = demo_outputs.remote(from_model, tag, f"{ROOT}/demo_{tag}.json")
        print(json.dumps(rows, indent=1)[:2000])
    elif stage == "speed":
        row = score_model.remote(STARTS[run] if run in STARTS else run, f"{run}_speed", f"{ROOT}/speed_{run.replace('/', '_')}.json",
                                 True, "heldout", 0)
        print(json.dumps(row, indent=1)[:3000])
    else:
        raise SystemExit(f"unknown stage {stage!r}")
