"""Round 14: written models. Donor layers grafted INSIDE the host, one network.

Unlike a tap (donor runs beside the host), a block graft puts the donor's
own layers into the host's forward pass: h = h + W_out(block(W_in(h)) - x).
One model, one tokenizer, one forward pass. Rounds 1 to 4 failed because
the stitch was broken (attention sink, untrained map); the stitch study
showed a stitch trained on the donor's own LM loss keeps the donor intact
(penalty 0.03 to 0.09 nats). So: graft Qwen2.5-7B or 14B "knowledge"
layers into Qwen2.5-0.5B-Instruct (same tokenizer) with a trained MLP
stitch, the donor sink, and optionally a small LoRA on the grafted layers.

Task: MMLU (round 6 dev 1000, round 7 held-out 2000), log-prob of A to D.
Stitch: 800 steps on the donor's LM loss over 2000 MMLU aux texts.
Heal: 500 steps on 8000 MMLU aux texts (prompt + answer), glue (and part
LoRA) only. 2 seeds. Params run per token are reported for cost.

Arms:
  host            Qwen-0.5B alone
  lora            Qwen-0.5B LoRA r16, same heal data and steps
  g7              Qwen-7B layers 8..20 grafted at host layer 7
  g7_plora        g7 plus a rank-8 LoRA on the grafted layers
  g7_random       control: a random same-length Qwen-7B block, same options
  g14             Qwen-14B layers 14..34 grafted at host layer 7
  g14_plora       g14 plus a rank-8 LoRA on the grafted layers

    python scripts/modal_fanout.py modal_round14.py round14 --profiles a,b,c
"""

import json
import os
import time

import modal

app = modal.App("wsp-round14")
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
)

HOST = "Qwen/Qwen2.5-0.5B-Instruct"
RECORDS = [
    {"id": "q7b_8_20", "type": "block", "model": "Qwen/Qwen2.5-7B-Instruct", "layers": [8, 20]},
    {"id": "q14b_14_34", "type": "block", "model": "Qwen/Qwen2.5-14B-Instruct", "layers": [14, 34]},
]
STITCH = {"kind": "mlp", "objective": "donor_lm", "steps": 800, "lr": 3e-4, "batch_size": 8, "seq_len": 256}
ARMS = {
    "host": None,
    "lora": None,
    "g7": ("q7b_8_20", None, None),
    "g7_plora": ("q7b_8_20", 8, None),
    "g7_random": ("q7b_8_20", None, "random"),
    "g14": ("q14b_14_34", None, None),
    "g14_plora": ("q14b_14_34", 8, None),
}
SEEDS = [0, 1]


def cells():
    out = []
    for arm in ARMS:
        seeds = [0] if arm == "host" else SEEDS
        for seed in seeds:
            out.append((arm, seed))
    return out


def params_run(model):
    total = 0
    for param in model.parameters():
        total += param.numel()
    return total


@app.function(image=image, volumes={"/vol": volume}, gpu="H200", memory=196608, timeout=4 * 3600)
def run_cell(arm, seed):
    import torch
    import transformers

    from modal_round6 import mmlu_data, score_mmlu
    from modal_round7 import mmlu_split
    from wsp import Catalog, Program, load

    start = time.time()
    tokenizer = transformers.AutoTokenizer.from_pretrained(HOST)
    pairs, dev = mmlu_split(tokenizer, heldout=False)
    _, held = mmlu_split(tokenizer, heldout=True)
    texts = []
    for prompt, completion in pairs:
        texts.append(prompt + completion)
    row = {"arm": arm, "seed": seed}

    if arm == "host":
        model = transformers.AutoModelForCausalLM.from_pretrained(HOST, dtype=torch.bfloat16).to("cuda").eval()
        row["dev"] = score_mmlu(model, tokenizer, dev)
        row["heldout"] = score_mmlu(model, tokenizer, held)
        row["params_run_b"] = params_run(model) / 1e9
        return row

    torch.manual_seed(seed)
    program = Program(host=HOST, seed=seed)
    if arm == "lora":
        program.lora(rank=16)
    else:
        part_id, part_lora, control = ARMS[arm]
        program.graft(part_id, glue_rank=64, stitch=dict(STITCH), part_lora_rank=part_lora, control=control,
                      control_seed=seed + 100)
    program.heal(steps=500, data="gsm8k_train", lr=3e-4, batch_size=8, seq_len=256)
    out_dir = f"/vol/compiled/round14_{arm}_s{seed}"
    program.compile(out_dir, Catalog(RECORDS, root="/vol"), align_texts=texts[:2000], heal_texts=texts,
                    device="cuda", cache_dir="/vol/tensor_cache")
    report = json.load(open(f"{out_dir}/compile_report.json"))
    model, tok = load(out_dir, device="cuda")
    row["dev"] = score_mmlu(model, tok, dev)
    row["heldout"] = score_mmlu(model, tok, held)
    row["params_run_b"] = params_run(model) / 1e9
    grafts = report.get("grafts", [])
    for meta in grafts:
        meta.pop("indices", None)
        if isinstance(meta.get("stitch"), dict):
            meta["stitch"].pop("losses", None)
    row["grafts"] = grafts
    heal = report.get("heal") or {}
    row["trainable_params"] = heal.get("trainable_params")
    losses = heal.get("losses") or []
    row["heal_loss_last20"] = sum(losses[-20:]) / max(1, len(losses[-20:]))
    row["seconds"] = round(time.time() - start, 1)
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
        print(json.dumps({k: row.get(k) for k in ["arm", "seed", "dev", "heldout", "params_run_b", "seconds"]}))
    os.makedirs("results", exist_ok=True)
    with open(f"results/round14_shard{shard}.json", "w") as f:
        json.dump(rows, f, indent=2)
