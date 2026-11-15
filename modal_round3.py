"""Round 3: do whole donor blocks beat fine-tuning at the same trainable budget?

Host unsloth/Llama-3.2-1B-Instruct (weak at math). Donor
Qwen/Qwen2.5-Math-1.5B (strong at math). GSM8K, 300 heal steps, the same
data and loop for every arm, 3 seeds, 200-item search split.

Arms:
  lora_matched   host LoRA, trainable params matched to the rank-64 block glue
  lora_r64       host LoRA rank 64 on all linear layers (a standard strong LoRA)
  block_a_r64    donor layers 8..19 grafted, rank-64 glue
  block_b_r64    donor layers 4..23 grafted, rank-64 glue
  block_rand     random same-length donor range (control for block_a), rank-64 glue
  block_a_full   donor layers 8..19, full-rank glue

Trainable params are what training costs; donor params are frozen. Every
report row carries trainable params, total params, and decode tokens/s.

Search-split numbers are interim checks, not results.

    MODAL_PROFILE=trisanth modal run modal_round3.py
"""

import json
import os
import time

import modal

app = modal.App("wsp-round3")
volume = modal.Volume.from_name("wsp-models", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch==2.7.1", "transformers==5.23.0", "accelerate", "huggingface_hub", "requests",
        "safetensors", "zstandard", "numpy", "datasets",
    )
    .env({"HF_HOME": "/vol/hf"})
    .add_local_python_source("wsp")
    .add_local_file("modal_graft.py", "/root/modal_graft.py")
)

HOST = "unsloth/Llama-3.2-1B-Instruct"
DONOR = "Qwen/Qwen2.5-Math-1.5B"
SEEDS = [0, 1, 2]
HEAL_STEPS = 300
RECORDS = [
    {"id": "qmath_block_8_19", "type": "block", "model": DONOR, "layers": [8, 19]},
    {"id": "qmath_block_4_23", "type": "block", "model": DONOR, "layers": [4, 23]},
]


def build_program(arm, seed):
    from wsp import Program

    program = Program(host=HOST, seed=seed)
    if arm == "lora_matched":
        # rank-64 glue for a 2048 to 1536 block: 64 * (2048 + 1536) * 2
        program.lora(budget_params=64 * (2048 + 1536) * 2)
    elif arm == "lora_r64":
        program.lora(rank=64)
    elif arm == "block_a_r64":
        program.graft("qmath_block_8_19", glue_rank=64)
    elif arm == "block_b_r64":
        program.graft("qmath_block_4_23", glue_rank=64)
    elif arm == "block_rand":
        program.graft("qmath_block_8_19", glue_rank=64, control="random", control_seed=seed + 100)
    elif arm == "block_a_full":
        program.graft("qmath_block_8_19")
    else:
        raise ValueError(arm)
    program.heal(steps=HEAL_STEPS, data="gsm8k_train", lr=3e-4, batch_size=8, seq_len=384)
    return program


def decode_speed(model, tokenizer, tokens=128):
    import torch

    ids = tokenizer("Question: What is 12 times 7?\nAnswer:", return_tensors="pt").input_ids.to(model.device)
    torch.cuda.synchronize()
    start = time.time()
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=tokens, min_new_tokens=tokens, do_sample=False,
                             pad_token_id=tokenizer.eos_token_id)
    torch.cuda.synchronize()
    return round((out.shape[1] - ids.shape[1]) / (time.time() - start), 1)


@app.function(image=image, volumes={"/vol": volume}, gpu="H100", memory=65536, timeout=5400)
def run_arm(arm, seed):
    import torch

    from modal_graft import score
    from wsp import Catalog, load
    from wsp.data import format_gsm8k, gsm8k_splits

    search, heal_items = gsm8k_splits(seed=0)
    heal_texts = []
    for item in heal_items:
        heal_texts.append(format_gsm8k(item))
    catalog = Catalog(RECORDS, root="/vol")
    out_dir = f"/vol/compiled/round3_{arm}_s{seed}"
    torch.manual_seed(seed)
    start = time.time()
    program = build_program(arm, seed)
    program.compile(out_dir, catalog, align_texts=heal_texts[:200], heal_texts=heal_texts, device="cuda",
                    cache_dir="/vol/tensor_cache")
    compile_seconds = round(time.time() - start, 1)
    report = json.load(open(f"{out_dir}/compile_report.json"))
    model, tokenizer = load(out_dir, device="cuda")
    accuracy, samples = score(model, tokenizer, search)
    trainable = report["heal"]["trainable_params"]
    total = 0
    for param in model.parameters():
        total += param.numel()
    losses = report["heal"]["losses"] if report.get("heal") else []
    row = {
        "arm": arm, "seed": seed, "accuracy": accuracy, "trainable_params": trainable, "total_params": total,
        "decode_tok_s": decode_speed(model, tokenizer), "compile_seconds": compile_seconds,
        "heal_loss_last10": sum(losses[-10:]) / max(1, len(losses[-10:])),
        "grafts": report.get("grafts", []), "lora": report.get("lora"), "samples": samples[:2],
    }
    for meta in row["grafts"]:
        meta.pop("indices", None)
    volume.commit()
    return row


@app.local_entrypoint()
def main(arms: str = "lora_matched,lora_r64,block_a_r64,block_b_r64,block_rand,block_a_full"):
    cells = []
    for arm in arms.split(","):
        for seed in SEEDS:
            cells.append((arm, seed))
    rows = []
    for row in run_arm.starmap(cells, return_exceptions=True):
        if isinstance(row, Exception):
            print("cell failed:", repr(row)[:400])
            continue
        rows.append(row)
        print(json.dumps({key: row[key] for key in ["arm", "seed", "accuracy", "trainable_params", "decode_tok_s"]}))
    os.makedirs("results", exist_ok=True)
    with open("results/round3_rows.json", "w") as f:
        json.dump(rows, f, indent=2)
