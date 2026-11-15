"""Bake: turn the MMLU tap into a plain 1B model that runs with no donor.

Host unsloth/Llama-3.2-1B-Instruct, donor Qwen/Qwen2.5-7B-Instruct (apache-2.0).
The teacher is round 7's tap: donor layer 20 read at host layer 10, trained
500 steps on 8000 MMLU auxiliary_train pairs (branch lr 3e-4, gate lr 3e-2).

Every trained arm uses the same 24000 MMLU auxiliary_train prompts (never MMLU
test items), a rank-16 host LoRA, lr 1e-4, batch 16, 1500 steps (one pass).

Arms, 3 seeds each:
  bake_tap     distill the tapped teacher's A-D distribution into the bare host
               (wsp.bake, answer mode). The teacher is scored too.
  bake_donor   control: distill Qwen-7B's own A-D distribution into the bare
               host, no tap. Shows whether the tap adds anything over plain
               distillation from the donor.
  lora_label   control: the same LoRA trained on the gold letters (fine-tuning).
  host         the untrained host (1 run)

Eval: round 6's 1000 dev items and round 7's 2000 held-out items (disjoint),
scored by the argmax over " A" to " D". Nothing is tuned on held-out; each
model is scored on it once.

    modal run modal_bake.py
"""

import json
import os
import random
import time

import modal

app = modal.App("wsp-bake")
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
    .add_local_file("modal_round7.py", "/root/modal_round7.py")
)

HOST = "unsloth/Llama-3.2-1B-Instruct"
DONOR = "Qwen/Qwen2.5-7B-Instruct"
DONOR_LAYER = 20
HOST_LAYER = 10
SEEDS = [0, 1, 2]
BAKE_PROMPTS = 24000
STEPS = 1500
BATCH = 16
LR = 1e-4
RANK = 16


def bake_pairs(tokenizer):
    """24000 labeled MMLU auxiliary_train pairs; the first 8000 are the tap's training pairs."""
    from modal_round6 import mmlu_data

    pairs, _ = mmlu_data(train_count=BAKE_PROMPTS, tokenizer=tokenizer)
    return pairs


def eval_items(tokenizer):
    from modal_round7 import mmlu_split

    _, dev = mmlu_split(tokenizer, heldout=False)
    _, heldout = mmlu_split(tokenizer, heldout=True)
    return dev, heldout


def score_both(model, tokenizer, dev, heldout, donors=None):
    from modal_round6 import score_mmlu

    return {"dev": score_mmlu(model, tokenizer, dev, donors=donors),
            "heldout": score_mmlu(model, tokenizer, heldout, donors=donors)}


def train_teacher(tokenizer_pairs, seed):
    from modal_round6 import build
    from modal_round7 import train_tap

    host, tokenizer, donors, _ = build(HOST, DONOR, DONOR_LAYER, HOST_LAYER, None, seed)
    losses = train_tap(host, tokenizer, donors, tokenizer_pairs[:8000], [], seed)
    return host, tokenizer, donors, losses


def donor_letter_logits(donor, donor_tokenizer, prompts, letter_ids):
    """Qwen-7B's own logits over its " A" to " D" tokens at the last prompt position."""
    import torch

    donor_tokenizer.padding_side = "left"
    if donor_tokenizer.pad_token is None:
        donor_tokenizer.pad_token = donor_tokenizer.eos_token
    encoded = donor_tokenizer(prompts, return_tensors="pt", padding=True).to(donor.device)
    with torch.no_grad():
        logits = donor(**encoded).logits[:, -1, :].float()
    return logits[:, letter_ids]


def bake_from_donor(donor, donor_tokenizer, prompts, seed):
    """Control: distill the donor's A-D distribution into a bare host LoRA (no tap)."""
    import torch
    import transformers

    from modal_round6 import letter_ids
    from wsp.bake import distill_kl, student_logits
    from wsp.lora import apply_lora

    torch.manual_seed(seed)
    student = transformers.AutoModelForCausalLM.from_pretrained(HOST, dtype=torch.bfloat16).to("cuda")
    tokenizer = transformers.AutoTokenizer.from_pretrained(HOST)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    for param in student.parameters():
        param.requires_grad_(False)
    lora_params = apply_lora(student, RANK)
    for param in lora_params:
        param.data = param.data.to("cuda")
        param.requires_grad_(True)
    optimizer = torch.optim.AdamW(lora_params, lr=LR, weight_decay=0.0)
    host_letters = letter_ids(tokenizer)
    donor_letters = letter_ids(donor_tokenizer)

    order = list(range(len(prompts)))
    random.Random(seed).shuffle(order)
    losses = []
    student.train()
    for step in range(STEPS):
        batch = []
        for offset in range(BATCH):
            batch.append(prompts[order[(step * BATCH + offset) % len(order)]])
        target = donor_letter_logits(donor, donor_tokenizer, batch, donor_letters)
        encoded = tokenizer(batch, return_tensors="pt", padding=True).to("cuda")
        optimizer.zero_grad()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = student_logits(student, encoded["input_ids"], encoded["attention_mask"])
        logits = logits[:, -1, :].float()[:, host_letters]
        loss = distill_kl(logits, target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(lora_params, 1.0)
        optimizer.step()
        losses.append(float(loss.detach()))
    student.eval()
    return student, tokenizer, losses


@app.function(image=image, volumes={"/vol": volume}, gpu="H100", memory=131072, timeout=4 * 3600)
def run_cell(arm, seed):
    import torch
    import transformers

    start = time.time()
    row = {"arm": arm, "seed": seed}
    host_tok = transformers.AutoTokenizer.from_pretrained(HOST)
    dev, heldout = eval_items(host_tok)
    pairs = bake_pairs(host_tok)
    prompts = []
    for prompt, completion in pairs:
        prompts.append(prompt)
    row["prompts"] = len(prompts)

    if arm == "host":
        from modal_round7 import plain
        model, tokenizer = plain(HOST)
        row["scores"] = score_both(model, tokenizer, dev, heldout)

    elif arm == "bake_tap":
        from modal_round6 import letter_ids
        from wsp.bake import bake, mean_kl, save_baked

        teacher, tokenizer, donors, tap_losses = train_teacher(pairs, seed)
        row["teacher"] = score_both(teacher, tokenizer, dev, heldout, donors=donors)
        row["tap_loss_last10"] = sum(tap_losses[-10:]) / 10
        result = bake(teacher, tokenizer, donors, prompts, rank=RANK, steps=STEPS, lr=LR, batch_size=BATCH,
                      mode="answer", option_ids=letter_ids(tokenizer), seed=seed, log_every=100)
        student = result["student"]
        row["bake_info"] = result["info"]
        row["kl_last10"] = sum(result["losses"][-10:]) / 10
        row["dev_kl"] = mean_kl(teacher, student, tokenizer, donors, [item[0] for item in dev[:200]],
                                option_ids=letter_ids(tokenizer))
        del teacher
        torch.cuda.empty_cache()
        row["scores"] = score_both(student, tokenizer, dev, heldout)
        out = "/vol/baked/mmlu_q7b_l20_h10_s" + str(seed)
        save_baked(student, tokenizer, out, info=result["info"])
        volume.commit()
        row["saved"] = out

    elif arm == "bake_donor":
        donor = transformers.AutoModelForCausalLM.from_pretrained(DONOR, dtype=torch.bfloat16).to("cuda").eval()
        donor_tokenizer = transformers.AutoTokenizer.from_pretrained(DONOR)
        student, tokenizer, losses = bake_from_donor(donor, donor_tokenizer, prompts, seed)
        row["kl_last10"] = sum(losses[-10:]) / 10
        del donor
        torch.cuda.empty_cache()
        row["scores"] = score_both(student, tokenizer, dev, heldout)

    elif arm == "lora_label":
        from modal_round6 import build, param_groups
        from wsp.tap import tap_train

        host, tokenizer, donors, lora_params = build(HOST, DONOR, DONOR_LAYER, HOST_LAYER, RANK, seed)
        groups = param_groups(host, lora_params, LR, train_taps=False)
        losses = tap_train(host, tokenizer, donors, pairs, groups, steps=STEPS, lr=LR, batch_size=BATCH,
                           seq_len=256, seed=seed)
        row["loss_last10"] = sum(losses[-10:]) / 10
        gates = []
        for branch in host.wsp_taps:
            gates.append(float(branch.gate.detach()))
        row["gates"] = gates
        row["scores"] = score_both(host, tokenizer, dev, heldout, donors=donors)

    row["seconds"] = round(time.time() - start, 1)
    return row


@app.local_entrypoint()
def main(seeds: str = "0 1 2"):
    cells = [("host", 0)]
    for value in seeds.split():
        for arm in ("bake_tap", "bake_donor", "lora_label"):
            cells.append((arm, int(value)))
    print(len(cells), "cells")
    rows = []
    for row in run_cell.starmap(cells, return_exceptions=True):
        if isinstance(row, Exception):
            print("cell failed:", repr(row)[:600])
            continue
        rows.append(row)
        print(json.dumps({"arm": row["arm"], "seed": row["seed"], "scores": row.get("scores"),
                          "teacher": row.get("teacher"), "seconds": row.get("seconds")}))
    os.makedirs("results", exist_ok=True)
    with open("results/bake_rows.json", "w") as f:
        json.dump(rows, f, indent=2)
