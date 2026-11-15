"""Round 7: harden the MMLU tap result and test it where decode cost matters.

Host unsloth/Llama-3.2-1B-Instruct throughout. Taps as in round 6 (branch
lr 3e-4, gate lr 3e-2, 500 steps, batch 8, seq_len 256).

A. held-out MMLU: 2000 MMLU test items disjoint from round 6's 1000 dev
   items, scored once. Arms: host, donor (Qwen2.5-7B-Instruct), lora_r16,
   tap 7B L20 into host L10. 3 seeds for trained arms.
B. layer map on the dev items, 1 seed: donor layers 8, 12, 16, 20, 24 by
   host layers 6, 10, 14.
C. donor scaling on the dev items, 2 seeds: Qwen2.5-1.5B/3B/14B-Instruct,
   tap from about 70% of donor depth into host L10; donor-alone scores.
D. TriviaQA (rc.nocontext), generative, the efficiency case: train on 8000
   train pairs, score 1000 validation questions by alias match within the
   first line of a 16-token greedy answer. Arms: host, donor alone,
   lora_r16, tap 7B L20 into host L10 (3 seeds). Records decode tok/s and
   donor prefill seconds: the tapped model decodes at host speed.
E. staged training on MMLU dev, 3 seeds: train the tap 500 steps, freeze
   it, then train a rank-16 LoRA 500 steps.

Dev-item numbers are development checks; part A is the held-out result.

    MODAL_PROFILE=trisanth modal run modal_round7.py
"""

import json
import os
import random
import re
import string
import time

import modal

app = modal.App("wsp-round7")
volume = modal.Volume.from_name("wsp-models", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch==2.7.1", "transformers==4.57.1", "accelerate", "huggingface_hub", "requests",
        "safetensors", "zstandard", "numpy", "datasets",
    )
    .env({"HF_HOME": "/vol/hf"})
    .add_local_python_source("wsp")
    .add_local_file("modal_graft.py", "/root/modal_graft.py")
    .add_local_file("modal_round5.py", "/root/modal_round5.py")
    .add_local_file("modal_round6.py", "/root/modal_round6.py")
)

HOST = "unsloth/Llama-3.2-1B-Instruct"
DONOR7 = "Qwen/Qwen2.5-7B-Instruct"
SCALE_DONORS = {
    "q1.5b": ("Qwen/Qwen2.5-1.5B-Instruct", 20),
    "q3b": ("Qwen/Qwen2.5-3B-Instruct", 25),
    "q7b": (DONOR7, 20),
    "q14b": ("Qwen/Qwen2.5-14B-Instruct", 34),
}
SEEDS = [0, 1, 2]


def mmlu_split(tokenizer, heldout=False):
    """Dev items are round 6's 1000; held-out items are 2000 others from the same seeded order."""
    from datasets import load_dataset

    from modal_round6 import mmlu_data, mmlu_prompt

    pairs, dev_items = mmlu_data(tokenizer=tokenizer)
    if not heldout:
        return pairs, dev_items
    test = load_dataset("cais/mmlu", "all", split="test")
    order = list(range(len(test)))
    random.Random(0).shuffle(order)
    items = []
    for index in order[1000:3000]:
        row = test[index]
        items.append((mmlu_prompt(row["question"], row["choices"]), int(row["answer"])))
    return pairs, items


def normalize(text):
    text = text.lower()
    text = "".join(ch for ch in text if ch not in set(string.punctuation))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def trivia_data(seed=0, train_count=8000, eval_count=1000):
    from datasets import load_dataset

    train = load_dataset("mandarjoshi/trivia_qa", "rc.nocontext", split="train")
    valid = load_dataset("mandarjoshi/trivia_qa", "rc.nocontext", split="validation")
    order = list(range(len(train)))
    random.Random(seed).shuffle(order)
    pairs = []
    for index in order[:train_count]:
        row = train[index]
        pairs.append(("Question: " + row["question"].strip() + "\nAnswer:", " " + row["answer"]["value"].strip()))
    order = list(range(len(valid)))
    random.Random(seed + 1).shuffle(order)
    items = []
    for index in order[:eval_count]:
        row = valid[index]
        aliases = []
        for alias in row["answer"]["normalized_aliases"]:
            aliases.append(normalize(alias))
        items.append(("Question: " + row["question"].strip() + "\nAnswer:", aliases))
    return pairs, items


def score_trivia(model, tokenizer, items, donors=None, batch_size=32):
    import torch

    from wsp.tap import clear_tap_context, prepare

    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    correct = 0
    samples = []
    for start in range(0, len(items), batch_size):
        batch = items[start:start + batch_size]
        prompts = []
        for prompt, aliases in batch:
            prompts.append(prompt)
        if donors is not None:
            encoded = prepare(model, tokenizer, donors, prompts)
        else:
            encoded = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
        with torch.no_grad():
            out = model.generate(**encoded, max_new_tokens=16, do_sample=False, pad_token_id=tokenizer.pad_token_id)
        if donors is not None:
            clear_tap_context(model)
        for row, (prompt, aliases) in enumerate(batch):
            text = tokenizer.decode(out[row, encoded["input_ids"].shape[1]:], skip_special_tokens=True)
            answer = normalize(text.strip().split("\n")[0])
            hit = False
            for alias in aliases:
                if alias and (alias == answer or (" " + alias + " ") in (" " + answer + " ")):
                    hit = True
            if hit:
                correct += 1
            if len(samples) < 3:
                samples.append(text[:120])
    return correct / len(items), samples


def decode_speed(model, tokenizer, donors=None, tokens=64):
    """Greedy decode tok/s for one prompt, plus donor prefill seconds when tapped."""
    import torch

    from wsp.tap import clear_tap_context, prepare

    prompt = "Question: Which planet is known as the Red Planet?\nAnswer:"
    torch.cuda.synchronize()
    start = time.time()
    if donors is not None:
        inputs = prepare(model, tokenizer, donors, [prompt])
    else:
        inputs = tokenizer([prompt], return_tensors="pt").to(model.device)
    torch.cuda.synchronize()
    prefill = time.time() - start
    start = time.time()
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=tokens, min_new_tokens=tokens, do_sample=False,
                             pad_token_id=tokenizer.eos_token_id)
    torch.cuda.synchronize()
    seconds = time.time() - start
    if donors is not None:
        clear_tap_context(model)
    return round((out.shape[1] - inputs["input_ids"].shape[1]) / seconds, 1), round(prefill, 4)


def plain(name):
    import torch
    import transformers

    model = transformers.AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16).to("cuda").eval()
    tokenizer = transformers.AutoTokenizer.from_pretrained(name)
    return model, tokenizer


def train_tap(host, tokenizer, donors, pairs, lora_params, seed, train_taps=True, steps=500):
    from modal_round6 import param_groups
    from wsp.tap import tap_train

    groups = param_groups(host, lora_params, 1e-4, train_taps)
    return tap_train(host, tokenizer, donors, pairs, groups, steps=steps, lr=3e-4, batch_size=8,
                     seq_len=256, seed=seed)


@app.function(image=image, volumes={"/vol": volume}, gpu="H100", memory=131072, timeout=7200)
def run_cell(part, arm, seed, cfg):
    import transformers

    from modal_round6 import build, score_mmlu

    start = time.time()
    row = {"part": part, "arm": arm, "seed": seed, "cfg": cfg}
    host_tok = transformers.AutoTokenizer.from_pretrained(HOST)

    if part in ("heldout", "layers", "scaling", "staged"):
        pairs, items = mmlu_split(host_tok, heldout=(part == "heldout"))
        if arm in ("host", "donor"):
            name = HOST if arm == "host" else cfg["donor"]
            model, tokenizer = plain(name)
            row["accuracy"] = score_mmlu(model, tokenizer, items)
            return row
        if arm == "lora_r16":
            host, tokenizer, donors, lora_params = build(HOST, DONOR7, 20, 10, 16, seed)
            losses = train_tap(host, tokenizer, donors, pairs, lora_params, seed, train_taps=False)
        elif arm == "staged":
            host, tokenizer, donors, lora_params = build(HOST, DONOR7, 20, 10, 16, seed)
            losses = train_tap(host, tokenizer, donors, pairs, [], seed, train_taps=True)
            for branch in host.wsp_taps:
                for param in branch.parameters():
                    param.requires_grad_(False)
            row["acc_after_tap"] = score_mmlu(host, tokenizer, items, donors=donors)
            losses = losses + train_tap(host, tokenizer, donors, pairs, lora_params, seed + 50, train_taps=False)
        else:
            host, tokenizer, donors, lora_params = build(HOST, cfg["donor"], cfg["donor_layer"], cfg["host_layer"],
                                                         None, seed)
            losses = train_tap(host, tokenizer, donors, pairs, [], seed)
        row["accuracy"] = score_mmlu(host, tokenizer, items, donors=donors)
    else:
        pairs, items = trivia_data()
        if arm in ("host", "donor"):
            name = HOST if arm == "host" else DONOR7
            model, tokenizer = plain(name)
            row["accuracy"], row["samples"] = score_trivia(model, tokenizer, items)
            row["decode_tok_s"], row["prefill_s"] = decode_speed(model, tokenizer)
            return row
        if arm == "lora_r16":
            host, tokenizer, donors, lora_params = build(HOST, DONOR7, 20, 10, 16, seed)
            losses = train_tap(host, tokenizer, donors, pairs, lora_params, seed, train_taps=False)
        else:
            host, tokenizer, donors, lora_params = build(HOST, DONOR7, 20, 10, None, seed)
            losses = train_tap(host, tokenizer, donors, pairs, [], seed)
        row["accuracy"], row["samples"] = score_trivia(host, tokenizer, items, donors=donors)
        tapped = None if arm == "lora_r16" else donors
        row["decode_tok_s"], row["prefill_s"] = decode_speed(host, tokenizer, donors=tapped if tapped else donors)
    gates = []
    for branch in host.wsp_taps:
        gates.append(float(branch.gate.detach()))
    row["gates"] = gates
    row["loss_last10"] = sum(losses[-10:]) / 10
    row["seconds"] = round(time.time() - start, 1)
    return row


@app.local_entrypoint()
def main():
    cells = []
    tap7 = {"donor": DONOR7, "donor_layer": 20, "host_layer": 10}
    cells.append(("heldout", "host", 0, {}))
    cells.append(("heldout", "donor", 0, {"donor": DONOR7}))
    for seed in SEEDS:
        cells.append(("heldout", "lora_r16", seed, {}))
        cells.append(("heldout", "tap", seed, tap7))
    for donor_layer in [8, 12, 16, 20, 24]:
        for host_layer in [6, 10, 14]:
            cells.append(("layers", "tap", 0, {"donor": DONOR7, "donor_layer": donor_layer, "host_layer": host_layer}))
    for key in SCALE_DONORS:
        name, layer = SCALE_DONORS[key]
        cells.append(("scaling", "donor", 0, {"donor": name, "key": key}))
        for seed in [0, 1]:
            cells.append(("scaling", "tap", seed, {"donor": name, "donor_layer": layer, "host_layer": 10, "key": key}))
    for arm in ["host", "donor", "lora_r16", "tap"]:
        seeds = [0] if arm in ("host", "donor") else SEEDS
        for seed in seeds:
            cells.append(("trivia", arm, seed, tap7))
    for seed in SEEDS:
        cells.append(("staged", "staged", seed, {}))
    print(f"{len(cells)} cells")
    rows = []
    for row in run_cell.starmap(cells, return_exceptions=True):
        if isinstance(row, Exception):
            print("cell failed:", repr(row)[:500])
            continue
        rows.append(row)
        print(json.dumps({key: row.get(key) for key in ["part", "arm", "seed", "cfg", "accuracy", "decode_tok_s",
                                                        "prefill_s", "acc_after_tap"]}))
    os.makedirs("results", exist_ok=True)
    with open("results/round7_rows.json", "w") as f:
        json.dump(rows, f, indent=2)
