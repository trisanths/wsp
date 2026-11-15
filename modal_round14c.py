"""Round 14c: heal toward the donor's own predictions.

Run 14 showed the stitch into a written-in donor block is lossless, but the
base cannot read the block's output when only the seam glue trains. Here the
heal teaches the base to read it: the loss is the KL divergence from the
full donor's next-token distribution (same Qwen2.5 tokenizer, so logits line
up exactly) plus the usual LM loss, on unlabeled text. Trainable: the graft
glue and a LoRA r16 on the base. Short: 1500 steps.

Control: the same distillation heal with NO block (base + LoRA), so a gain
from distillation alone is not credited to the written-in block. Also a
random same-length block.

Base Qwen/Qwen2.5-0.5B-Instruct; donor and teacher Qwen/Qwen2.5-14B-Instruct;
block = donor layers 14..34 at host layer 7 with the round 14 stitch.
Scored on MMLU dev 1000, MMLU held-out 2000, MedMCQA 1000. 2 seeds.

    python scripts/modal_fanout.py modal_round14c.py round14c --profiles a,b,c
"""

import json
import os
import random
import time

import modal

app = modal.App("wsp-round14c")
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
    .add_local_file("modal_round10.py", "/root/modal_round10.py")
    .add_local_file("modal_round14.py", "/root/modal_round14.py")
    .add_local_file("modal_round15.py", "/root/modal_round15.py")
)

HOST = "Qwen/Qwen2.5-0.5B-Instruct"
TEACHER = "Qwen/Qwen2.5-14B-Instruct"
ARMS = ["kd_lora_only", "kd_block14", "kd_block14_random"]
SEEDS = [0, 1]
STEPS = 1500
TEMPERATURE = 1.0
KD_WEIGHT = 1.0


def cells():
    out = []
    for arm in ARMS:
        for seed in SEEDS:
            out.append((arm, seed))
    return out


def texts_for_heal(tokenizer):
    from modal_round15 import heal_texts

    return heal_texts(tokenizer)


def trainable_params(model):
    params = []
    for name, param in model.named_parameters():
        keep = False
        if "lora_" in name:
            keep = True
        if "wsp_grafts" in name:
            for key in ("w_in", "w_out", "gate", "rms_scale"):
                if key in name:
                    keep = True
            if ".part." in name:
                keep = False
        param.requires_grad_(keep)
        if keep:
            params.append(param)
    return params


def kd_heal(student, teacher, tokenizer, texts, params, seed):
    import torch

    torch.manual_seed(seed)
    order = list(range(len(texts)))
    random.Random(seed).shuffle(order)
    optimizer = torch.optim.AdamW(params, lr=2e-4, weight_decay=0.0)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    student.train()
    losses = []
    position = 0
    for step in range(STEPS):
        batch = []
        while len(batch) < 8:
            batch.append(texts[order[position % len(order)]])
            position += 1
        enc = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=384).to("cuda")
        with torch.no_grad():
            teacher_logits = teacher(**enc).logits.float()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            student_logits = student(**enc).logits
        student_logits = student_logits.float()
        vocab = min(teacher_logits.shape[-1], student_logits.shape[-1])
        t = torch.log_softmax(teacher_logits[..., :vocab] / TEMPERATURE, dim=-1)
        s = torch.log_softmax(student_logits[..., :vocab] / TEMPERATURE, dim=-1)
        mask = enc["attention_mask"].float()
        kl = (t.exp() * (t - s)).sum(-1)
        kd = (kl * mask).sum() / mask.sum()
        labels = enc["input_ids"][:, 1:]
        lm = torch.nn.functional.cross_entropy(student_logits[:, :-1].reshape(-1, student_logits.shape[-1]),
                                               labels.reshape(-1), reduction="none")
        lm_mask = mask[:, 1:].reshape(-1)
        lm = (lm * lm_mask).sum() / lm_mask.sum()
        loss = KD_WEIGHT * kd + lm
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        optimizer.step()
        losses.append([float(kd), float(lm)])
        if step % 100 == 0:
            print(f"step {step} kd {float(kd):.4f} lm {float(lm):.4f}")
    student.eval()
    return losses


@app.function(image=image, volumes={"/vol": volume}, gpu="H200", memory=262144, timeout=5 * 3600)
def run_cell(arm, seed):
    import torch
    import transformers

    from modal_round14 import RECORDS, STITCH
    from modal_round6 import score_mmlu
    from modal_round7 import mmlu_split
    from modal_round10 import items as r10_items
    from wsp import Catalog, Program, load

    start = time.time()
    tokenizer = transformers.AutoTokenizer.from_pretrained(HOST)
    pairs, dev = mmlu_split(tokenizer, heldout=False)
    _, held = mmlu_split(tokenizer, heldout=True)
    med = []
    for item in r10_items():
        if item["set"] == "medmcqa":
            med.append((item["prompt"], item["gold"]))
    texts = texts_for_heal(tokenizer)

    program = Program(host=HOST, seed=seed)
    program.lora(rank=16)
    if arm != "kd_lora_only":
        control = "random" if arm.endswith("random") else None
        program.graft("q14b_14_34", glue_rank=64, stitch=dict(STITCH), control=control, control_seed=seed + 100)
    program.heal(steps=1, data="gsm8k_train", lr=1e-6, batch_size=1, seq_len=64)
    out_dir = f"/vol/compiled/round14c_{arm}_s{seed}"
    if arm == "kd_lora_only":
        # The compiler merges the LoRA of a LoRA-only program into the host, so
        # nothing would be left to train. Put a fresh LoRA on the base here.
        from wsp.lora import apply_lora, default_alpha

        torch.manual_seed(seed)
        student = transformers.AutoModelForCausalLM.from_pretrained(HOST, dtype=torch.bfloat16).to("cuda")
        tok = tokenizer
        apply_lora(student, 16, default_alpha(16))
        student.eval()
    else:
        program.compile(out_dir, Catalog(RECORDS, root="/vol"), align_texts=texts[:2000], heal_texts=texts[:16],
                        device="cuda", cache_dir="/vol/tensor_cache")
        student, tok = load(out_dir, device="cuda")
    teacher = transformers.AutoModelForCausalLM.from_pretrained(TEACHER, dtype=torch.bfloat16).to("cuda").eval()
    params = trainable_params(student)
    losses = kd_heal(student, teacher, tok, texts, params, seed)
    del teacher
    torch.cuda.empty_cache()
    total = 0
    for param in student.parameters():
        total += param.numel()
    row = {"arm": arm, "seed": seed, "params_b": round(total / 1e9, 3),
           "trainable": sum(p.numel() for p in params),
           "dev": score_mmlu(student, tok, dev), "heldout": score_mmlu(student, tok, held),
           "medmcqa": score_mmlu(student, tok, med),
           "kd_first50": sum(x[0] for x in losses[:50]) / 50, "kd_last50": sum(x[0] for x in losses[-50:]) / 50,
           "seconds": round(time.time() - start, 1)}
    return row


@app.local_entrypoint()
def main(shard: int = 0, nshards: int = 1, only: str = ""):
    todo = cells()
    if only:
        # "arm:seed,arm:seed" reruns chosen cells.
        todo = []
        for item in only.split(","):
            arm, seed = item.split(":")
            todo.append((arm, int(seed)))
    mine = []
    for index, cell in enumerate(todo):
        if index % nshards == shard:
            mine.append(cell)
    print(f"shard {shard}/{nshards}: {mine}")
    rows = []
    for row in run_cell.starmap(mine, return_exceptions=True):
        if isinstance(row, Exception):
            print("cell failed:", repr(row)[:600])
            continue
        rows.append(row)
        print(json.dumps(row))
    os.makedirs("results", exist_ok=True)
    with open(f"results/round14c_shard{shard}.json", "w") as f:
        json.dump(rows, f, indent=2)
