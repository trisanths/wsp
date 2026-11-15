"""Round 16: skill fusion. Write several specialists' blocks into one host.

Goal: one written model that matches or beats each specialist on its own
skill at the same time, with only stitches, part LoRAs, gates, and a short
heal trained.

Base (host): Qwen/Qwen2.5-1.5B-Instruct (28 layers, hidden 1536).
Specialists:
  math     Qwen/Qwen2.5-Math-1.5B-Instruct (same shape and tokenizer family)
  coder    Qwen/Qwen2.5-Coder-1.5B-Instruct (same shape and tokenizer family)
  medical  the first of MEDICAL_CANDIDATES that the Hugging Face API shows
           public, ungated, with safetensors and a fast tokenizer
           (m42-health/Llama3-Med42-8B, cross-family and cross-size); the
           fallback unsloth/Meta-Llama-3.1-8B-Instruct is not a medical
           model, so a row with it is not a medical result.

Arms (2 seeds where trained):
  base, math, coder, medical, qwen3b   each model alone (qwen3b is the size bar)
  merge_ties    TIES merge of base, math, and coder: task vectors against
                Qwen/Qwen2.5-1.5B, keep the top 20% by magnitude, sign
                consensus, disjoint mean, scale 1.0. Tensors whose shapes
                differ are skipped (the base tensor stays) and recorded.
  fuse2         base + math block (layers 6..23) + coder block (6..23),
                each at host layer 5 (its output is the residual stream
                that enters layer 6), with a trained MLP stitch (donor_lm
                objective when the vocabulary equals the base's, else
                downstream), the sink, part LoRA rank 8, and token gates
                (wsp.gate). Heal 800 steps on the mixed set.
  fuse3         fuse2 plus the medical block (about 30% to 75% depth) with
                the downstream stitch objective (cross-tokenizer).
  fuse2_nogate  fuse2 without gates (ablation).
  lora          base + LoRA rank 64 (lr 5e-5, the fair LoRA bar of run 4)
                on the same mixed data and steps.

Mixed heal set, equal parts (1600 texts each, 6400 in all, one pass at 800
steps of batch 8): GSM8K train solutions in chat format ending "The answer
is N."; code (MBPP full train in chat format plus CodeAlpaca-20k); MedMCQA
train items with explanations, in the round 6 multiple-choice format
followed by the answer letter and the explanation; general instructions
(alpaca-cleaned) in chat format. Each stitch trains on 1500 texts of its
own domain plus 500 general texts, held apart from the heal texts.

Every arm is scored with wsp/skills_eval.py on gsm8k, math500, humaneval,
mbpp, medmcqa, and mmlu. Rows also hold params total, params run per
token, decode tok/s at batch 1, and compile seconds. Full per-item
records with outputs go to /vol/results/round16/<arm>_s<seed>.json.

    python scripts/modal_fanout.py modal_round16.py round16 --profiles a,b,c
"""

import json
import os
import random
import re
import time

import modal

app = modal.App("wsp-round16")
volume = modal.Volume.from_name("wsp-models", create_if_missing=True)

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
PRETRAINED = "Qwen/Qwen2.5-1.5B"
MATH = "Qwen/Qwen2.5-Math-1.5B-Instruct"
CODER = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
QWEN3B = "Qwen/Qwen2.5-3B-Instruct"
# On 2026-11-14 the Hugging Face API showed the first two public, ungated,
# with safetensors and tokenizer.json. OpenBioLLM-8B has only .bin weights
# and Bio-Medical-Llama-3-8B is gated, so they come last.
MEDICAL_CANDIDATES = ["m42-health/Llama3-Med42-8B", "TsinghuaC3I/Llama-3.1-8B-UltraMedical",
                      "aaditya/Llama3-OpenBioLLM-8B", "ContactDoctor/Bio-Medical-Llama-3-8B"]
MEDICAL_FALLBACK = "unsloth/Meta-Llama-3.1-8B-Instruct"

QWEN_BLOCK = [6, 23]
MEDICAL_DEPTH = [0.30, 0.75]
HEAL = {"steps": 800, "lr": 3e-4, "batch_size": 8, "seq_len": 512}
LORA = {"rank": 64, "lr": 5e-5}
GATE_LR = 1e-2
STITCH = {"kind": "mlp", "steps": 800, "lr": 3e-4, "batch_size": 8, "seq_len": 256}
PART_LORA_RANK = 8
TIES_KEEP = 0.2
TIES_SCALE = 1.0
PER_POOL = 1600
ALIGN_DOMAIN = 1500
ALIGN_GENERAL = 500

PLAIN = ["base", "math", "coder", "medical", "qwen3b"]
TRAINED = ["fuse2", "fuse3", "fuse2_nogate", "fuse2_hostlora", "fuse3_hostlora", "lora"]
ARMS = PLAIN + ["merge_ties"] + TRAINED
SEEDS = [0, 1]


def cells():
    out = []
    for arm in ARMS:
        seeds = [0]
        if arm in TRAINED:
            seeds = SEEDS
        for seed in seeds:
            out.append((arm, seed))
    return out


# ---------------------------------------------------------------- donors


def pick_medical():
    """Return (repo_id, notes): the first medical candidate that is public, ungated, and loadable."""
    from huggingface_hub import HfApi

    api = HfApi()
    notes = []
    for repo in MEDICAL_CANDIDATES:
        info = None
        for attempt in range(3):
            try:
                info = api.model_info(repo)
                break
            except Exception as error:
                name = type(error).__name__
                if name in ("RepositoryNotFoundError", "GatedRepoError"):
                    break
                # A network error must not make one cell pick another donor
                # than the others, so it stops the cell after three tries.
                if attempt == 2:
                    raise
                time.sleep(5)
        if info is None:
            notes.append(f"{repo}: not found or gated")
            continue
        if info.gated not in (False, None) or info.private or info.disabled:
            notes.append(f"{repo}: gated, private, or disabled")
            continue
        files = []
        for sibling in info.siblings or []:
            files.append(sibling.rfilename)
        has_weights = "model.safetensors" in files or "model.safetensors.index.json" in files
        if not has_weights:
            notes.append(f"{repo}: no safetensors")
            continue
        if "tokenizer.json" not in files:
            notes.append(f"{repo}: no fast tokenizer (tokenizer.json)")
            continue
        notes.append(f"{repo}: ok")
        return repo, notes
    notes.append(f"fallback {MEDICAL_FALLBACK}")
    return MEDICAL_FALLBACK, notes


def same_vocab(model_a, model_b):
    import transformers

    a = transformers.AutoTokenizer.from_pretrained(model_a)
    b = transformers.AutoTokenizer.from_pretrained(model_b)
    return a.get_vocab() == b.get_vocab()


def layer_count(model_id):
    import transformers

    config = transformers.AutoConfig.from_pretrained(model_id)
    if hasattr(config, "get_text_config"):
        config = config.get_text_config()
    return config.num_hidden_layers


def medical_block(model_id, host_layers):
    """Return (first, last, host_layer) for the medical block at about 30% to 75% depth."""
    layers = layer_count(model_id)
    first = int(round(MEDICAL_DEPTH[0] * layers))
    last = int(round(MEDICAL_DEPTH[1] * layers)) - 1
    # The host layer whose output sits at the same relative depth as the
    # input of the donor's first layer.
    at = int(round(first / layers * host_layers)) - 1
    if at < 0:
        at = 0
    return first, last, at


# ---------------------------------------------------------------- data


def chat_text(tokenizer, user, assistant):
    messages = [{"role": "user", "content": user}, {"role": "assistant", "content": assistant}]
    return tokenizer.apply_chat_template(messages, tokenize=False)


def gsm8k_solution(answer):
    """GSM8K train answer text without calculator marks, ending 'The answer is N.'."""
    text = re.sub(r"<<[^>]*>>", "", answer)
    parts = text.split("####")
    body = parts[0].strip()
    final = ""
    if len(parts) > 1:
        final = parts[1].strip().replace(",", "")
    return body + "\nThe answer is " + final + "."


def math_pool(tokenizer, count):
    from datasets import load_dataset

    from wsp.skills_eval import GSM8K_INSTRUCTION

    rows = load_dataset("openai/gsm8k", "main", split="train")
    order = list(range(len(rows)))
    random.Random(0).shuffle(order)
    texts = []
    for index in order[:count]:
        row = rows[index]
        texts.append(chat_text(tokenizer, row["question"] + "\n" + GSM8K_INSTRUCTION, gsm8k_solution(row["answer"])))
    return texts


def code_pool(tokenizer, count):
    """MBPP full train split in chat format (disjoint from the sanitized test), then CodeAlpaca-20k."""
    from datasets import load_dataset

    from wsp.skills_eval import CODE_INSTRUCTION

    texts = []
    mbpp = load_dataset("google-research-datasets/mbpp", "full", split="train")
    for index in range(len(mbpp)):
        row = mbpp[index]
        user = (row["text"] + "\nYour code should pass these tests:\n\n" + "\n".join(row["test_list"]) + "\n\n"
                + CODE_INSTRUCTION)
        texts.append(chat_text(tokenizer, user, "```python\n" + row["code"].strip() + "\n```"))
    alpaca = load_dataset("sahil2801/CodeAlpaca-20k", split="train")
    order = list(range(len(alpaca)))
    random.Random(1).shuffle(order)
    for index in order:
        if len(texts) >= count:
            break
        row = alpaca[index]
        user = row["instruction"]
        if row.get("input"):
            user = user + "\n\n" + row["input"]
        texts.append(chat_text(tokenizer, user, row["output"]))
    random.Random(2).shuffle(texts)
    return texts[:count]


def medical_pool(count):
    """MedMCQA train items with an explanation, in the round 6 format plus letter and explanation."""
    from datasets import load_dataset

    from wsp.skills_eval import LETTERS, mcq_prompt

    rows = load_dataset("openlifescienceai/medmcqa", split="train")
    order = list(range(len(rows)))
    random.Random(3).shuffle(order)
    texts = []
    for index in order:
        if len(texts) >= count:
            break
        row = rows[index]
        explanation = (row.get("exp") or "").strip()
        if explanation == "" or row["cop"] not in (0, 1, 2, 3):
            continue
        choices = [row["opa"], row["opb"], row["opc"], row["opd"]]
        texts.append(mcq_prompt(row["question"], choices) + " " + LETTERS[int(row["cop"])] + "\nExplanation: "
                     + explanation)
    return texts


def general_pool(tokenizer, count):
    from datasets import load_dataset

    rows = load_dataset("yahma/alpaca-cleaned", split="train")
    order = list(range(len(rows)))
    random.Random(4).shuffle(order)
    texts = []
    for index in order[:count]:
        row = rows[index]
        user = row["instruction"]
        if row.get("input"):
            user = user + "\n\n" + row["input"]
        texts.append(chat_text(tokenizer, user, row["output"]))
    return texts


def build_data(tokenizer):
    """Return (heal_texts, align) where align maps "math", "code", "medical" to stitch texts.

    The data seeds are fixed, so every arm and seed trains on the same
    texts. Heal texts and stitch texts do not overlap.
    """
    total = PER_POOL + ALIGN_DOMAIN
    pools = {"math": math_pool(tokenizer, total), "code": code_pool(tokenizer, total),
             "medical": medical_pool(total), "general": general_pool(tokenizer, PER_POOL + 3 * ALIGN_GENERAL)}
    heal = []
    for name in ["math", "code", "medical", "general"]:
        for text in pools[name][:PER_POOL]:
            heal.append(text)
    random.Random(5).shuffle(heal)
    align = {}
    spare = pools["general"][PER_POOL:]
    for number, name in enumerate(["math", "code", "medical"]):
        texts = list(pools[name][PER_POOL:PER_POOL + ALIGN_DOMAIN])
        for text in spare[number * ALIGN_GENERAL:(number + 1) * ALIGN_GENERAL]:
            texts.append(text)
        random.Random(6 + number).shuffle(texts)
        align[name] = texts
    sizes = {}
    for name in pools:
        sizes[name] = len(pools[name])
    return heal, align, sizes


# ---------------------------------------------------------------- TIES


def trim_top(tensor, keep):
    """Keep the top keep fraction of entries by magnitude; zero the rest."""
    import torch

    count = tensor.numel()
    k = int(count * keep)
    if k <= 0:
        return torch.zeros_like(tensor)
    if k >= count:
        return tensor
    flat = tensor.abs().reshape(-1)
    threshold = torch.kthvalue(flat, count - k + 1).values
    return tensor * (tensor.abs() >= threshold).to(tensor.dtype)


def ties_merge(pretrained, finetuned, keep=TIES_KEEP, scale=TIES_SCALE, start=None):
    """TIES merge (Yadav et al. 2023) of fine-tuned state dicts against a pretrained one.

    pretrained: name -> tensor. finetuned: a list of name -> tensor dicts.
    start: name -> tensor the result starts from (default pretrained);
    a name is copied from it unchanged when any model lacks the name or
    has another shape. Returns (merged, skipped names). Math runs in float32.
    """
    import torch

    if start is None:
        start = pretrained
    merged = {}
    skipped = []
    for name in start:
        ok = name in pretrained and pretrained[name].shape == start[name].shape
        for model in finetuned:
            if name not in model or model[name].shape != start[name].shape:
                ok = False
        if not ok:
            merged[name] = start[name]
            skipped.append(name)
            continue
        base = pretrained[name].float()
        trimmed = []
        for model in finetuned:
            trimmed.append(trim_top(model[name].float() - base, keep))
        stacked = torch.stack(trimmed)
        sign = torch.sign(stacked.sum(dim=0))
        agree = (torch.sign(stacked) == sign) & (stacked != 0)
        total = (stacked * agree.to(stacked.dtype)).sum(dim=0)
        count = agree.sum(dim=0).clamp(min=1).to(stacked.dtype)
        merged[name] = (base + scale * total / count).to(start[name].dtype)
    return merged, skipped


def build_ties(out_dir):
    """Merge base, math, and coder with TIES and save to out_dir. Return the merge report."""
    import torch
    import transformers

    report_path = os.path.join(out_dir, "merge_report.json")
    if os.path.exists(report_path):
        with open(report_path) as f:
            return json.load(f)
    start = time.time()
    loaded = {}
    for name in [PRETRAINED, BASE, MATH, CODER]:
        model = transformers.AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16)
        loaded[name] = model
    states = {}
    for name in loaded:
        params = {}
        for key, param in loaded[name].named_parameters():
            params[key] = param.detach().to("cuda")
        states[name] = params
    merged, skipped = ties_merge(states[PRETRAINED], [states[BASE], states[MATH], states[CODER]], start=states[BASE])
    target = loaded[BASE]
    with torch.no_grad():
        for key, param in target.named_parameters():
            param.copy_(merged[key].to(param.device))
    os.makedirs(out_dir, exist_ok=True)
    target.save_pretrained(out_dir)
    transformers.AutoTokenizer.from_pretrained(BASE).save_pretrained(out_dir)
    report = {"pretrained": PRETRAINED, "models": [BASE, MATH, CODER], "keep": TIES_KEEP, "scale": TIES_SCALE,
              "skipped": skipped, "tensors": len(merged), "seconds": round(time.time() - start, 1)}
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    return report


# ---------------------------------------------------------------- fusion


def fusion_program(arm, seed, medical_id, host_layers):
    """Return (program, catalog records, per-graft domains, stitch objectives) for a fuse arm."""
    from wsp import Program

    records = [
        {"id": "math_6_23", "type": "block", "model": MATH, "layers": list(QWEN_BLOCK)},
        {"id": "coder_6_23", "type": "block", "model": CODER, "layers": list(QWEN_BLOCK)},
    ]
    at = QWEN_BLOCK[0] - 1
    program = Program(host=BASE, seed=seed)
    domains = []
    objectives = {}
    for record, domain in [(records[0], "math"), (records[1], "code")]:
        objective = "downstream"
        if same_vocab(BASE, record["model"]):
            objective = "donor_lm"
        objectives[record["id"]] = objective
        stitch = dict(STITCH)
        stitch["objective"] = objective
        stitch["seed"] = seed
        program.graft(record["id"], at=at, stitch=stitch, part_lora_rank=PART_LORA_RANK)
        domains.append(domain)
    if arm.endswith("_hostlora"):
        # The base learns to read the written-in skills (run 14 lesson).
        program.lora(rank=16)
    if arm in ("fuse3", "fuse3_hostlora"):
        first, last, med_at = medical_block(medical_id, host_layers)
        record = {"id": f"medical_{first}_{last}", "type": "block", "model": medical_id, "layers": [first, last]}
        records.append(record)
        objective = "downstream"
        if same_vocab(BASE, medical_id):
            objective = "donor_lm"
        objectives[record["id"]] = objective
        stitch = dict(STITCH)
        stitch["objective"] = objective
        stitch["seed"] = seed
        program.graft(record["id"], at=med_at, stitch=stitch, part_lora_rank=PART_LORA_RANK)
        domains.append("medical")
    program.heal(steps=HEAL["steps"], data="round16_mixed", lr=HEAL["lr"], batch_size=HEAL["batch_size"],
                 seq_len=HEAL["seq_len"])
    return program, records, domains, objectives


def slim_grafts(metas):
    out = []
    for meta in metas:
        row = dict(meta)
        row.pop("indices", None)
        if isinstance(row.get("stitch"), dict):
            stitch = dict(row["stitch"])
            stitch.pop("losses", None)
            row["stitch"] = stitch
        out.append(row)
    return out


# ---------------------------------------------------------------- cells


def score_model(model, tokenizer, items, row, batch_size, gated=False):
    """Run the skill suite and the cost numbers on a model. Fill row and return the full result."""
    from wsp.gate import gate_means
    from wsp.skills_eval import SKILLS, decode_speed, evaluate_skill, model_params

    total, run = model_params(model)
    row["params_total_b"] = total / 1e9
    row["params_run_b"] = run / 1e9
    row["decode_tok_s"] = round(decode_speed(model, tokenizer), 1)
    scores = {}
    full = {}
    gates = {}
    for skill in SKILLS:
        if gated:
            gate_means(model, reset=True)
        result = evaluate_skill(model, tokenizer, skill, items=items[skill], batch_size=batch_size)
        scores[skill] = result["score"]
        full[skill] = result
        print(f"{row['arm']} s{row['seed']} {skill}: {result['score']:.4f} in {result['seconds']} s")
        if gated:
            gates[skill] = gate_means(model)
    row["scores"] = scores
    total_score = 0.0
    for skill in scores:
        total_score += scores[skill]
    row["suite"] = total_score / len(scores)
    if gated:
        row["gate_means"] = gates
    records = {}
    for skill in full:
        slim = []
        for record in full[skill]["records"]:
            entry = dict(record)
            entry.pop("output", None)
            entry.pop("stderr", None)
            slim.append(entry)
        records[skill] = slim
    row["records"] = records
    return full


@app.function(image=image, volumes={"/vol": volume}, gpu="H200", memory=196608, timeout=8 * 3600)
def run_cell(arm, seed):
    import torch
    import transformers

    from wsp import Catalog, Program, load
    from wsp.fuse import compile_fusion
    from wsp.fuse import load as load_fused
    from wsp.skills_eval import load_items, SKILLS

    start = time.time()
    torch.manual_seed(seed)
    row = {"arm": arm, "seed": seed}
    medical_id, notes = pick_medical()
    row["medical"] = medical_id
    row["medical_notes"] = notes
    items = {}
    for skill in SKILLS:
        items[skill] = load_items(skill)

    gated = False
    batch_size = 32
    if arm in PLAIN:
        names = {"base": BASE, "math": MATH, "coder": CODER, "medical": medical_id, "qwen3b": QWEN3B}
        name = names[arm]
        row["model"] = name
        model = transformers.AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16).to("cuda").eval()
        tokenizer = transformers.AutoTokenizer.from_pretrained(name)
        row["compile_seconds"] = 0.0
        if arm != "medical":
            batch_size = 64
    elif arm == "merge_ties":
        out_dir = "/vol/compiled/round16_merge_ties"
        report = build_ties(out_dir)
        row["compile_seconds"] = report["seconds"]
        row["merge"] = report
        volume.commit()
        model, tokenizer = load(out_dir, device="cuda")
        batch_size = 64
    elif arm == "lora":
        tokenizer = transformers.AutoTokenizer.from_pretrained(BASE)
        heal_texts, align, sizes = build_data(tokenizer)
        row["data"] = sizes
        program = Program(host=BASE, seed=seed).lora(rank=LORA["rank"])
        program.heal(steps=HEAL["steps"], data="round16_mixed", lr=LORA["lr"], batch_size=HEAL["batch_size"],
                     seq_len=HEAL["seq_len"])
        out_dir = f"/vol/compiled/round16_lora_s{seed}"
        compile_start = time.time()
        program.compile(out_dir, Catalog([], root="/vol"), heal_texts=heal_texts, device="cuda")
        row["compile_seconds"] = round(time.time() - compile_start, 1)
        report = json.load(open(os.path.join(out_dir, "compile_report.json")))
        heal = report.get("heal") or {}
        losses = heal.get("losses") or []
        row["trainable_params"] = heal.get("trainable_params")
        row["heal_loss_last20"] = sum(losses[-20:]) / max(1, len(losses[-20:]))
        volume.commit()
        model, tokenizer = load(out_dir, device="cuda")
        batch_size = 64
    else:
        tokenizer = transformers.AutoTokenizer.from_pretrained(BASE)
        heal_texts, align, sizes = build_data(tokenizer)
        row["data"] = sizes
        host_layers = layer_count(BASE)
        program, records, domains, objectives = fusion_program(arm, seed, medical_id, host_layers)
        row["objectives"] = objectives
        per_graft = []
        for domain in domains:
            per_graft.append(align[domain])
        out_dir = f"/vol/compiled/round16_{arm}_s{seed}"
        gated = arm != "fuse2_nogate"
        compile_start = time.time()
        report = compile_fusion(program, Catalog(records, root="/vol"), out_dir, per_graft, heal_texts, gates=gated,
                                gate_lr=GATE_LR, device="cuda", cache_dir="/vol/tensor_cache")
        row["compile_seconds"] = round(time.time() - compile_start, 1)
        row["grafts"] = slim_grafts(report["grafts"])
        heal = report.get("heal") or {}
        losses = heal.get("losses") or []
        row["trainable_params"] = heal.get("trainable_params")
        row["gate_params"] = heal.get("gate_params")
        row["heal_loss_last20"] = sum(losses[-20:]) / max(1, len(losses[-20:]))
        row["gates"] = report.get("gates")
        volume.commit()
        model, tokenizer = load_fused(out_dir, device="cuda")

    full = score_model(model, tokenizer, items, row, batch_size, gated=gated)
    row["seconds"] = round(time.time() - start, 1)
    os.makedirs("/vol/results/round16", exist_ok=True)
    with open(f"/vol/results/round16/{arm}_s{seed}.json", "w") as f:
        json.dump({"row": row, "full": full}, f, indent=2, default=str)
    volume.commit()
    return row


@app.local_entrypoint()
def main(shard: int = 0, nshards: int = 1):
    mine = []
    for index, cell in enumerate(cells()):
        if index % nshards == shard:
            mine.append(cell)
    print(f"shard {shard}/{nshards}: {mine}")
    rows = []
    for row in run_cell.starmap(mine, return_exceptions=True):
        if isinstance(row, Exception):
            print("cell failed:", repr(row)[:600])
            continue
        rows.append(row)
        summary = {}
        for key in ["arm", "seed", "suite", "scores", "params_run_b", "decode_tok_s", "compile_seconds", "seconds"]:
            summary[key] = row.get(key)
        print(json.dumps(summary))
    os.makedirs("results", exist_ok=True)
    with open(f"results/round16_shard{shard}.json", "w") as f:
        json.dump(rows, f, indent=2, default=str)
