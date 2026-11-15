"""Round 18: same-family expert transplant (wsp/transplant.py).

Base Qwen/Qwen3-1.7B (hidden 2048, 28 layers). Source Qwen/Qwen3-30B-A3B
(hidden 2048, 48 layers, 128 experts, top 8). Same width, same tokenizer.
Next to the MLP of every base layer sits a bank of K experts taken from the
source layer at the same relative depth, read through a stitch pair and a
router that starts from the source router rows. The model stays near base
size and runs alone. The source runs only during the build (selection,
stitch fit) and as the teacher of the distillation heal.

Arms (2 seeds for the trained arms; the seed changes the heal order, the
LoRA init, and the random control draw):
  base                  Qwen3-1.7B alone
  lora                  LoRA r16, LM loss only, same data and steps
  kd_lora               LoRA r16, KL to the source logits plus LM loss, same
                        data and steps. The real competitor: the same loss
                        as transplant_k4, with no bank.
  transplant_k4         K=4 per layer (math, code, knowledge take turns),
                        KL + LM heal, LoRA r16 on the base
  transplant_k4_random  the same build with K=4 random experts per layer
  transplant_k8         K=8
  transplant_k4_lm      K=4, LM loss only (no distillation)
  source                Qwen3-30B-A3B, one cell, for reference

Trainable in the transplant arms: stitches, routers, gates, LoRA r16 on the
base. The stitches start from a ridge fit on paired states of the same
text (STITCH_RANK keeps the fit as a rank r product, so the parameter
counts stay near the target; rows hold total and active counts).

Heal: 1500 steps, batch 8, 384 tokens, one pass over 12000 texts in equal
parts math (GSM8K train, MATH train), code (MBPP train, Magicoder Python),
knowledge (MMLU auxiliary_train, raw multiple-choice text), and general
chat (UltraChat). Expert selection uses 96 texts per skill from the same
train pools, not the scoring sets. The learning rates are first guesses
and untuned: stitches 1e-4, routers 5e-4, gates 5e-3, LoRA 2e-4.

Scores (wsp.skills_eval, every arm with the Qwen3 chat template and
thinking off): GSM8K 500 seeded test items, MATH-500, HumanEval, MBPP
sanitized test (257), MMLU held-out (2000 test items from positions 1000 to
3000 of the seed-0 order; log-prob of the answer letter after the raw
multiple-choice prompt, as skills_eval scores MMLU). Rows hold every score,
total and active params, trainable params, heal seconds, routing shares per
skill and per expert label (prompt tokens of 32 items per skill), the
fitted stitch quality, and the gate values.

Go and stop rule: transplant_k4 must beat kd_lora and transplant_k4_random
by more than seed noise on at least two skills. Seed noise is the largest of
the seed spread of each arm in the comparison and one binomial standard
error (wsp.transplant.go_stop). Otherwise stop.

    python scripts/modal_fanout.py modal_round18.py round18 --profiles a,b,c
    modal run modal_round18.py --verdict 'results/round18_shard*.json'   (reads the rows, runs no cell)

Writes results/round18_shard<i>.json, full per-item records to
/vol/results/round18/<arm>_s<seed>.json, saved transplants to
/vol/written/round18_<arm>_s<seed>, and caches the selection and the
stitch fit in /vol/transplant.
"""

import glob
import json
import os
import random
import time

import modal

app = modal.App("wsp-round18")
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

BASE = "Qwen/Qwen3-1.7B"
SOURCE = "Qwen/Qwen3-30B-A3B"
PAIR = "qwen3"

SKILLS = ["gsm8k", "math500", "humaneval", "mbpp", "mmlu"]
COUNTS = {"gsm8k": 500, "math500": 500, "humaneval": 164, "mbpp": 257, "mmlu": 2000}
TRAINED = ["lora", "kd_lora", "transplant_k4", "transplant_k4_random", "transplant_k8", "transplant_k4_lm"]
SEEDS = [0, 1]
K_OF = {"transplant_k4": 4, "transplant_k4_random": 4, "transplant_k8": 8, "transplant_k4_lm": 4}
DISTILL = ["kd_lora", "transplant_k4", "transplant_k4_random", "transplant_k8"]

STEPS = 1500
BATCH = 8
SEQ_LEN = 384
LORA_RANK = 16
STITCH_RANK = 256
RIDGE = 1e-3
TOP_K = 2
LR = {"stitch": 1e-4, "router": 5e-4, "gate": 5e-3, "lora": 2e-4}
SELECT_TEXTS = 96
FIT_TEXTS = 256
POOL = 3000
ROUTING_ITEMS = 32
MIN_FREQ = 0.02

GSM8K_INSTRUCTION = "Solve step by step and end with: The answer is N."
MATH_INSTRUCTION = "Solve step by step and put the final answer in \\boxed{}."
CODE_INSTRUCTION = "Write the whole function, with its imports, in a single ```python code block."
LETTERS = ["A", "B", "C", "D"]
MATH_CONFIGS = ["algebra", "counting_and_probability", "geometry", "intermediate_algebra", "number_theory",
                "prealgebra", "precalculus"]


def cells():
    out = [("base", 0), ("source", 0)]
    for arm in TRAINED:
        for seed in SEEDS:
            out.append((arm, seed))
    return out


# ---------------------------------------------------------------- tokenizer and items


def thinking_off(tokenizer):
    """Make every apply_chat_template call use enable_thinking=False (Qwen3 template)."""
    original = tokenizer.apply_chat_template

    def patched(*args, **kwargs):
        kwargs.setdefault("enable_thinking", False)
        return original(*args, **kwargs)

    tokenizer.apply_chat_template = patched
    return tokenizer


def mcq_text(question, choices):
    lines = ["Question: " + question.strip()]
    for index, choice in enumerate(choices):
        lines.append(LETTERS[index] + ". " + str(choice).strip())
    return "\n".join(lines) + "\nAnswer:"


def mmlu_heldout_items():
    """2000 MMLU test items from positions 1000 to 3000 of the seed-0 order (the round 7 held-out set)."""
    from datasets import load_dataset

    test = load_dataset("cais/mmlu", "all", split="test")
    order = list(range(len(test)))
    random.Random(0).shuffle(order)
    items = []
    for index in order[1000:3000]:
        row = test[index]
        items.append({"id": f"mmlu_held:{index}", "prompt": mcq_text(row["question"], row["choices"]),
                      "gold": int(row["answer"]), "subject": row["subject"]})
    return items


def eval_items(ev):
    return {"gsm8k": ev.gsm8k_items(count=COUNTS["gsm8k"], seed=0), "math500": ev.math500_items(),
            "humaneval": ev.humaneval_items(), "mbpp": ev.mbpp_items(), "mmlu": mmlu_heldout_items()}


def prompt_ids(tokenizer, ev, skill, item):
    if skill == "mmlu":
        return tokenizer(item["prompt"], return_tensors="pt").input_ids
    text = ev.render_prompt(tokenizer, ev.user_text(skill, item))
    return tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids


def routing_by_skill(model, tokenizer, ev, items):
    """Routing share per expert label per skill, on the prompt tokens of ROUTING_ITEMS items."""
    import torch

    from wsp.transplant import routing_shares, track_routing

    device = next(model.parameters()).device
    out = {}
    for skill in SKILLS:
        track_routing(model)
        for item in items[skill][:ROUTING_ITEMS]:
            with torch.no_grad():
                model(prompt_ids(tokenizer, ev, skill, item).to(device))
        shares = routing_shares(model)
        track_routing(model, on=False)
        by_label = {}
        for label in shares["by_label"]:
            by_label[label] = round(shares["by_label"][label], 4)
        out[skill] = by_label
    return out


# ---------------------------------------------------------------- heal data


def chat_pair(tokenizer, user, assistant):
    prompt = tokenizer.apply_chat_template([{"role": "user", "content": user}], tokenize=False,
                                           add_generation_prompt=True)
    return prompt + assistant + "<|im_end|>\n"


def gsm8k_solution(answer):
    import re

    text = re.sub(r"<<[^>]*>>", "", answer)
    parts = text.split("####")
    final = ""
    if len(parts) > 1:
        final = parts[1].strip().replace(",", "")
    return parts[0].strip() + "\nThe answer is " + final + "."


def math_pool(tokenizer, count):
    from datasets import load_dataset

    texts = []
    rows = load_dataset("openai/gsm8k", "main", split="train")
    order = list(range(len(rows)))
    random.Random(0).shuffle(order)
    half = count // 2
    for index in order[:half]:
        row = rows[index]
        texts.append(chat_pair(tokenizer, row["question"] + "\n" + GSM8K_INSTRUCTION, gsm8k_solution(row["answer"])))
    hard = []
    for name in MATH_CONFIGS:
        try:
            part = load_dataset("EleutherAI/hendrycks_math", name, split="train")
        except Exception as error:
            print("MATH config unavailable:", name, repr(error)[:120])
            continue
        for row in part:
            hard.append((row["problem"], row["solution"]))
    random.Random(1).shuffle(hard)
    for problem, solution in hard[:count - len(texts)]:
        texts.append(chat_pair(tokenizer, problem + "\n" + MATH_INSTRUCTION, solution))
    for index in order[half:]:
        if len(texts) >= count:
            break
        row = rows[index]
        texts.append(chat_pair(tokenizer, row["question"] + "\n" + GSM8K_INSTRUCTION, gsm8k_solution(row["answer"])))
    texts = texts[:count]
    random.Random(3).shuffle(texts)
    return texts


def code_pool(tokenizer, count):
    from datasets import load_dataset

    texts = []
    mbpp = load_dataset("google-research-datasets/mbpp", "full", split="train")
    for row in mbpp:
        user = (row["text"] + "\nYour code should pass these tests:\n\n" + "\n".join(row["test_list"]) + "\n\n"
                + CODE_INSTRUCTION)
        texts.append(chat_pair(tokenizer, user, "```python\n" + row["code"].strip() + "\n```"))
    magic = load_dataset("ise-uiuc/Magicoder-OSS-Instruct-75K", split="train")
    order = list(range(len(magic)))
    random.Random(0).shuffle(order)
    for index in order:
        if len(texts) >= count:
            break
        row = magic[index]
        if row["lang"] != "python":
            continue
        texts.append(chat_pair(tokenizer, row["problem"], row["solution"]))
    random.Random(2).shuffle(texts)
    return texts[:count]


def knowledge_pool(count):
    from datasets import load_dataset

    rows = load_dataset("cais/mmlu", "all", split="auxiliary_train")
    order = list(range(len(rows)))
    random.Random(0).shuffle(order)
    texts = []
    for index in order[:count]:
        row = rows[index]
        texts.append(mcq_text(row["question"], row["choices"]) + " " + LETTERS[int(row["answer"])])
    return texts


def general_pool(tokenizer, count):
    from datasets import load_dataset

    stream = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft", streaming=True)
    rows = []
    for row in stream:
        messages = row["messages"]
        if len(messages) >= 2 and messages[0]["role"] == "user" and messages[1]["role"] == "assistant":
            rows.append((messages[0]["content"], messages[1]["content"]))
        if len(rows) >= 2 * count:
            break
    random.Random(0).shuffle(rows)
    texts = []
    for user, assistant in rows[:count]:
        texts.append(chat_pair(tokenizer, user, assistant))
    return texts


def pools(tokenizer):
    return {"math": math_pool(tokenizer, POOL), "code": code_pool(tokenizer, POOL),
            "knowledge": knowledge_pool(POOL), "general": general_pool(tokenizer, POOL)}


def heal_texts(all_pools):
    texts = []
    for name in all_pools:
        texts += all_pools[name]
    random.Random(0).shuffle(texts)
    return texts


def skill_texts(all_pools):
    """Selection texts: the first SELECT_TEXTS of each skill pool."""
    out = {}
    for name in ["math", "code", "knowledge"]:
        out[name] = all_pools[name][:SELECT_TEXTS]
    return out


# ---------------------------------------------------------------- cells


def load_model(name):
    import torch
    import transformers

    return transformers.AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16)


def trainable_count(model):
    total = 0
    for param in model.parameters():
        if param.requires_grad:
            total += param.numel()
    return total


def selection_for(source, tokenizer, all_pools, layer_map, arm, seed):
    """The expert plan of a transplant arm. The real selection is cached per K."""
    from wsp.transplant import select_experts

    k = K_OF[arm]
    if arm == "transplant_k4_random":
        return select_experts(source, tokenizer, {}, [], layer_map, k, control="random", seed=seed)
    path = f"/vol/transplant/selection_{PAIR}_k{k}.json"
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    plan = select_experts(source, tokenizer, skill_texts(all_pools), all_pools["general"][:SELECT_TEXTS], layer_map, k,
                         min_freq=MIN_FREQ)
    os.makedirs("/vol/transplant", exist_ok=True)
    with open(path, "w") as f:
        json.dump(plan, f)
    volume.commit()
    return plan


def stitches_for(base, source, tokenizer, all_pools, layer_map):
    """The ridge stitch fit. It depends on the layer map and the text only, so it is cached."""
    import torch

    from wsp.transplant import fit_stitches

    path = f"/vol/transplant/fit_{PAIR}_r{STITCH_RANK}.pt"
    if os.path.exists(path):
        return torch.load(path)
    texts = heal_texts(all_pools)[:FIT_TEXTS]
    fits = fit_stitches(base, source, tokenizer, texts, layer_map, rank=STITCH_RANK, ridge=RIDGE, max_length=256,
                        batch_size=4)
    os.makedirs("/vol/transplant", exist_ok=True)
    torch.save(fits, path)
    volume.commit()
    return fits


def param_groups(model):
    from wsp.transplant import set_trainable

    groups = set_trainable(model)
    out = []
    for name in ["stitch", "router", "gate", "lora"]:
        if len(groups[name]) > 0:
            out.append({"params": groups[name], "lr": LR[name]})
    return out


@app.function(image=image, volumes={"/vol": volume}, gpu="H200", memory=262144, timeout=10 * 3600)
def run_cell(arm, seed):
    import torch
    import transformers

    from wsp import skills_eval as ev
    from wsp.lora import apply_lora
    from wsp.transplant import (banks, build_transplant, count_params, heal, make_layer_map, parts_from_model,
                                save_transplant, source_active_params, source_moe_layers)

    start = time.time()
    torch.manual_seed(seed)
    tokenizer = thinking_off(transformers.AutoTokenizer.from_pretrained(BASE))
    row = {"arm": arm, "seed": seed, "base": BASE, "source": SOURCE, "k": K_OF.get(arm)}
    kd = arm in DISTILL
    model = None
    batch_size = 32

    if arm == "base":
        model = load_model(BASE).to("cuda").eval()
    elif arm == "source":
        model = load_model(SOURCE).to("cuda").eval()
        batch_size = 64
    elif arm in ("lora", "kd_lora"):
        model = load_model(BASE).to("cuda")
        for param in model.parameters():
            param.requires_grad_(False)
        lora = apply_lora(model, LORA_RANK)
        all_pools = pools(tokenizer)
        teacher = None
        if kd:
            teacher = load_model(SOURCE).to("cuda").eval()
        heal_start = time.time()
        row["heal"] = heal(model, tokenizer, heal_texts(all_pools), [{"params": lora, "lr": LR["lora"]}], STEPS,
                           BATCH, SEQ_LEN, seed, teacher=teacher, kd_weight=1.0 if kd else 0.0,
                           autocast=torch.bfloat16)
        row["heal_seconds"] = round(time.time() - heal_start, 1)
        del teacher
        torch.cuda.empty_cache()
    else:
        all_pools = pools(tokenizer)
        source = load_model(SOURCE).to("cuda").eval()
        base = load_model(BASE).to("cuda")
        layer_map = make_layer_map(len(base.model.layers), len(source.model.layers),
                                   source_moe_layers(source.config))
        plan = selection_for(source, tokenizer, all_pools, layer_map, arm, seed)
        fits = stitches_for(base, source, tokenizer, all_pools, layer_map)
        parts = parts_from_model(source, plan)
        model = build_transplant(base, plan, parts, stitches=fits, lora_rank=LORA_RANK, top_k=TOP_K,
                                 stitch_rank=STITCH_RANK, base_id=BASE, source_id=SOURCE, pair=PAIR)
        teacher = source if kd else None
        if not kd:
            del source
            torch.cuda.empty_cache()
        row["plan"] = {"layer_map": plan["layer_map"], "experts": plan["experts"], "labels": plan["labels"],
                       "control": plan.get("control")}
        row["fit_r2_in"] = round(sum(fits[i]["r2_in"] for i in fits) / len(fits), 4)
        row["fit_r2_out"] = round(sum(fits[i]["r2_out"] for i in fits) / len(fits), 4)
        heal_start = time.time()
        row["heal"] = heal(model, tokenizer, heal_texts(all_pools), param_groups(model), STEPS, BATCH, SEQ_LEN, seed,
                           teacher=teacher, kd_weight=1.0 if kd else 0.0, autocast=torch.bfloat16)
        row["heal_seconds"] = round(time.time() - heal_start, 1)
        del teacher
        if kd:
            del source
        torch.cuda.empty_cache()
        gates = []
        for bank in banks(model):
            gates.append(round(float(bank.gate.detach()), 4))
        row["gates"] = gates
        out_dir = f"/vol/written/round18_{arm}_s{seed}"
        save_transplant(model, out_dir, tokenizer)
        volume.commit()
        row["saved"] = out_dir

    if arm == "source":
        total, active = source_active_params(model)
        row["params_total"] = total
        row["params_active"] = active
    elif arm.startswith("transplant"):
        counts = count_params(model)
        row["params_total"] = counts["total"]
        row["params_active"] = counts["active"]
        row["total_ratio"] = round(counts["total_ratio"], 4)
        row["active_ratio"] = round(counts["active_ratio"], 4)
        row["params_stitch"] = counts["stitch"]
        row["params_router"] = counts["router"]
    else:
        seen = set()
        total = 0
        for param in model.parameters():
            if id(param) not in seen:
                seen.add(id(param))
                total += param.numel()
        row["params_total"] = total
        row["params_active"] = ev.model_params(model)[1]
    row["params_trainable"] = trainable_count(model)

    items = eval_items(ev)
    row["gsm8k_prompt_sample"] = ev.render_prompt(tokenizer, ev.user_text("gsm8k", items["gsm8k"][0]))
    if arm.startswith("transplant"):
        row["routing"] = routing_by_skill(model, tokenizer, ev, items)
    model.eval()
    result = ev.evaluate_suite(model, tokenizer, skills=SKILLS, items=items, batch_size=batch_size)
    row["scores"] = dict(result["scores"])
    row["macro"] = result["suite"]
    row["seconds"] = round(time.time() - start, 1)

    os.makedirs("/vol/results/round18", exist_ok=True)
    full = dict(row)
    full["skills"] = ev.strip_outputs(result)["skills"]
    with open(f"/vol/results/round18/{arm}_s{seed}.json", "w") as f:
        json.dump(full, f, default=str)
    volume.commit()
    return row


def print_verdict(path):
    from wsp.transplant import go_stop

    rows = []
    for name in sorted(glob.glob(path)):
        with open(name) as f:
            rows += json.load(f)
    if len(rows) == 0:
        raise SystemExit(f"no rows in {path}")
    result = go_stop(rows, SKILLS, COUNTS)
    print(json.dumps(result, indent=2))


@app.local_entrypoint()
def main(shard: int = 0, nshards: int = 1, verdict: str = ""):
    if verdict:
        print_verdict(verdict)
        return
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
        print(json.dumps({k: row.get(k) for k in ["arm", "seed", "scores", "macro", "params_total", "params_active",
                                                  "params_trainable", "heal_seconds"]}))
    os.makedirs("results", exist_ok=True)
    with open(f"results/round18_shard{shard}.json", "w") as f:
        json.dump(rows, f, indent=2)
