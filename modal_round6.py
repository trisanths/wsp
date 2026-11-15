"""Round 6: where can a tap beat fine-tuning?

Round 5 showed prefill taps carry no multi-step reasoning on GSM8K: the
donor has not solved the problem during prefill, and real messages score
the same as mismatched ones. Taps should help where the answer already
sits in the donor's prefill states: knowledge. Fine-tuning a 1B host
cannot add knowledge the training data lacks.

Part A, MMLU (knowledge). Host unsloth/Llama-3.2-1B-Instruct, donor
Qwen/Qwen2.5-7B-Instruct. Train on 8000 MMLU auxiliary_train pairs
(prompt = question and choices, completion = " X"), 500 steps, then score
1000 seeded MMLU test items by the log-probability of " A" to " D" after
"Answer:". Arms (3 seeds unless noted):
  host            no training (1 run)
  donor           the donor alone, same scoring (1 run)
  lora_r16        host LoRA rank 16 (lr 1e-4), same loop and data
  tap_l20_h10     donor layer 20 into host layer 10
  tap_l27_h12     donor layer 27 (last) into host layer 12
  tap_l27_derange the same tap trained and scored on mismatched messages
  tap_l27_lora    tap_l27_h12 plus the rank-16 LoRA, trained together

Part B, GSM8K (does a tap add anything on top of fine-tuning?). Host
Llama-1B, LoRA rank 64 (lr 5e-5), 300 steps, completion-only loss, 3
seeds; scored on the 200-item search split with the raw prompt.
  lora64_null     LoRA plus a tap whose gate is frozen at 0 (same loop, no donor effect)
  lora64_lfm      LoRA plus a tap from LiquidAI/LFM2-350M-Math layer 8 at host layer 8
  lora64_qmath    LoRA plus a tap from Qwen2.5-Math-1.5B-Instruct layer 14 at host layer 10

Every LoRA-only arm runs through tap_train with a frozen null tap, so the
loop, data, and loss are identical across arms. Search-split numbers are
interim checks, not results. About 25 cells, about 6 H100 hours in total
across containers.

    MODAL_PROFILE=trisanth modal run modal_round6.py
"""

import json
import os
import random
import time

import modal

app = modal.App("wsp-round6")
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
)

HOST = "unsloth/Llama-3.2-1B-Instruct"
MMLU_DONOR = "Qwen/Qwen2.5-7B-Instruct"
LFM = "LiquidAI/LFM2-350M-Math"
QMATH = "Qwen/Qwen2.5-Math-1.5B-Instruct"
SEEDS = [0, 1, 2]
BRANCH_LR = 3e-4
GATE_LR = 3e-2
LETTERS = ["A", "B", "C", "D"]


def mmlu_prompt(question, choices):
    lines = ["Question: " + question.strip()]
    for index, choice in enumerate(choices):
        lines.append(LETTERS[index] + ". " + str(choice).strip())
    return "\n".join(lines) + "\nAnswer:"


def mmlu_data(seed=0, train_count=8000, eval_count=1000, tokenizer=None, max_prompt_tokens=240):
    from datasets import load_dataset

    test = load_dataset("cais/mmlu", "all", split="test")
    order = list(range(len(test)))
    random.Random(seed).shuffle(order)
    eval_items = []
    for index in order[:eval_count]:
        row = test[index]
        eval_items.append((mmlu_prompt(row["question"], row["choices"]), int(row["answer"])))

    aux = load_dataset("cais/mmlu", "auxiliary_train", split="train")
    order = list(range(len(aux)))
    random.Random(seed + 1).shuffle(order)
    pairs = []
    for index in order:
        row = aux[index]
        if "train" in row and isinstance(row["train"], dict):
            row = row["train"]
        if len(row["choices"]) != 4:
            continue
        prompt = mmlu_prompt(row["question"], row["choices"])
        if tokenizer is not None and len(tokenizer(prompt).input_ids) > max_prompt_tokens:
            continue
        pairs.append((mmlu_prompt(row["question"], row["choices"]), " " + LETTERS[int(row["answer"])]))
        if len(pairs) >= train_count:
            break
    return pairs, eval_items


def letter_ids(tokenizer):
    ids = []
    for letter in LETTERS:
        encoded = tokenizer(" " + letter, add_special_tokens=False).input_ids
        ids.append(encoded[-1])
    return ids


def score_mmlu(model, tokenizer, items, donors=None, derange=False, batch_size=32):
    """Accuracy by the argmax over the log-probs of ' A' to ' D' after the prompt."""
    import torch

    from wsp.tap import clear_tap_context, prepare

    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    choice_ids = letter_ids(tokenizer)
    correct = 0
    for start in range(0, len(items), batch_size):
        batch = items[start:start + batch_size]
        prompts = []
        for prompt, answer in batch:
            prompts.append(prompt)
        if donors is not None:
            encoded = prepare(model, tokenizer, donors, prompts, derange=derange)
        else:
            encoded = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
        with torch.no_grad():
            logits = model(**encoded).logits[:, -1, :].float()
        if donors is not None:
            clear_tap_context(model)
        for row, (prompt, answer) in enumerate(batch):
            scores = logits[row, choice_ids]
            if int(scores.argmax()) == answer:
                correct += 1
    return correct / len(items)


def param_groups(host, lora_params, lora_lr, train_taps=True):
    groups = []
    if train_taps:
        gates = []
        others = []
        for branch in host.wsp_taps:
            for name, param in branch.named_parameters():
                if name == "gate":
                    gates.append(param)
                else:
                    others.append(param)
        groups.append({"params": others, "lr": BRANCH_LR})
        groups.append({"params": gates, "lr": GATE_LR})
    if len(lora_params) > 0:
        groups.append({"params": lora_params, "lr": lora_lr})
    return groups


def build(host_name, donor_name, donor_layer, host_layer, lora_rank, seed):
    """Load host and donor, attach one tap, optionally a host LoRA. Return (host, tok, donors, lora_params)."""
    import torch
    import transformers

    from wsp.lora import apply_lora
    from wsp.tap import TapBranch, attach_taps

    torch.manual_seed(seed)
    host = transformers.AutoModelForCausalLM.from_pretrained(host_name, dtype=torch.bfloat16).to("cuda").eval()
    tokenizer = transformers.AutoTokenizer.from_pretrained(host_name)
    donor = transformers.AutoModelForCausalLM.from_pretrained(donor_name, dtype=torch.bfloat16).to("cuda").eval()
    donor_tokenizer = transformers.AutoTokenizer.from_pretrained(donor_name)
    donors = {"donor": (donor, donor_tokenizer)}
    lora_params = []
    if lora_rank is not None:
        lora_params = apply_lora(host, lora_rank)
        for param in lora_params:
            param.data = param.data.to("cuda")
    branch = TapBranch(host.config.hidden_size, donor.config.hidden_size, attn_dim=256, heads=8,
                       donor_id="donor", donor_layer=donor_layer)
    attach_taps(host, [(host_layer, branch, 0)])
    host.wsp_taps.to("cuda")
    return host, tokenizer, donors, lora_params


MMLU_ARMS = {
    "tap_l20_h10": (20, 10, None, True, False),
    "tap_l27_h12": (27, 12, None, True, False),
    "tap_l27_derange": (27, 12, None, True, True),
    "tap_l27_lora": (27, 12, 16, True, False),
    "lora_r16": (27, 12, 16, False, False),
}

GSM_ARMS = {
    "lora64_null": (QMATH, 14, 10, False),
    "lora64_lfm": (LFM, 8, 8, True),
    "lora64_qmath": (QMATH, 14, 10, True),
}


@app.function(image=image, volumes={"/vol": volume}, gpu="H100", memory=131072, timeout=7200)
def run_cell(part, arm, seed):
    import torch
    import transformers

    from wsp.tap import tap_param_count, tap_train

    start = time.time()
    if part == "mmlu":
        pairs, items = mmlu_data(tokenizer=transformers.AutoTokenizer.from_pretrained(HOST))
        if arm == "host":
            model = transformers.AutoModelForCausalLM.from_pretrained(HOST, dtype=torch.bfloat16).to("cuda").eval()
            tokenizer = transformers.AutoTokenizer.from_pretrained(HOST)
            return {"part": part, "arm": arm, "seed": seed, "accuracy": score_mmlu(model, tokenizer, items)}
        if arm == "donor":
            model = transformers.AutoModelForCausalLM.from_pretrained(MMLU_DONOR, dtype=torch.bfloat16).to("cuda").eval()
            tokenizer = transformers.AutoTokenizer.from_pretrained(MMLU_DONOR)
            return {"part": part, "arm": arm, "seed": seed, "accuracy": score_mmlu(model, tokenizer, items)}
        donor_layer, host_layer, lora_rank, train_taps, derange = MMLU_ARMS[arm]
        host, tokenizer, donors, lora_params = build(HOST, MMLU_DONOR, donor_layer, host_layer, lora_rank, seed)
        groups = param_groups(host, lora_params, 1e-4, train_taps)
        losses = tap_train(host, tokenizer, donors, pairs, groups, steps=500, lr=BRANCH_LR, batch_size=8,
                           seq_len=256, seed=seed, derange=derange)
        accuracy = score_mmlu(host, tokenizer, items, donors=donors, derange=derange)
    else:
        from modal_round5 import score_taps
        from wsp.data import gsm8k_splits

        search, heal_items = gsm8k_splits(seed=0)
        pairs = []
        for item in heal_items:
            pairs.append(("Question: " + item["question"] + "\nAnswer:", " " + item["answer"]))
        donor_name, donor_layer, host_layer, train_taps = GSM_ARMS[arm]
        host, tokenizer, donors, lora_params = build(HOST, donor_name, donor_layer, host_layer, 64, seed)
        groups = param_groups(host, lora_params, 5e-5, train_taps)
        losses = tap_train(host, tokenizer, donors, pairs, groups, steps=300, lr=BRANCH_LR, batch_size=8,
                           seq_len=384, seed=seed)
        accuracy, samples = score_taps(host, tokenizer, donors, search)
        derange = False
    trainable = 0
    for group in groups:
        for param in group["params"]:
            trainable += param.numel()
    gates = []
    for branch in host.wsp_taps:
        gates.append(float(branch.gate.detach()))
    return {"part": part, "arm": arm, "seed": seed, "accuracy": accuracy, "trainable_params": trainable,
            "tap_params": tap_param_count(host), "gates": gates, "seconds": round(time.time() - start, 1),
            "loss_first10": sum(losses[:10]) / 10, "loss_last10": sum(losses[-10:]) / 10}


@app.local_entrypoint()
def main(parts: str = "mmlu,gsm8k"):
    cells = []
    if "mmlu" in parts:
        cells.append(("mmlu", "host", 0))
        cells.append(("mmlu", "donor", 0))
        for arm in MMLU_ARMS:
            for seed in SEEDS:
                cells.append(("mmlu", arm, seed))
    if "gsm8k" in parts:
        for arm in GSM_ARMS:
            for seed in SEEDS:
                cells.append(("gsm8k", arm, seed))
    print(f"{len(cells)} cells")
    rows = []
    for row in run_cell.starmap(cells, return_exceptions=True):
        if isinstance(row, Exception):
            print("cell failed:", repr(row)[:500])
            continue
        rows.append(row)
        print(json.dumps(row))
    os.makedirs("results", exist_ok=True)
    with open(f"results/round6_{parts.replace(',', '_')}_rows.json", "w") as f:
        json.dump(rows, f, indent=2)
