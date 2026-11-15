"""Round 8: compose knowledge across donor families (MMLU, round 6 dev items).

Questions:
1. Do taps work for donors of many families, not only Qwen?
2. Do two taps from different families beat either tap alone?
3. Can a smaller host (Qwen2.5-0.5B-Instruct) carry a 7B to 14B donor?

Setup as round 6: 8000 MMLU auxiliary_train pairs, 500 steps, batch 8,
seq_len 256, branch lr 3e-4, gate lr 3e-2; score 1000 MMLU test items by
the log-probability of " A" to " D". 2 seeds for trained cells. Donors are
ungated mirrors where the original repo is gated. Each tap reads the donor
at about 70% of its depth.

Fan out across Modal workspaces:
    python scripts/modal_fanout.py modal_round8.py round8 --profiles a,b,c
"""

import json
import os
import time

import modal

app = modal.App("wsp-round8")
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
)

LLAMA1B = ("unsloth/Llama-3.2-1B-Instruct", 10)
QWEN05 = ("Qwen/Qwen2.5-0.5B-Instruct", 15)
DONORS = {
    "qwen7b": ("Qwen/Qwen2.5-7B-Instruct", 20),
    "qwen14b": ("Qwen/Qwen2.5-14B-Instruct", 34),
    "llama8b": ("unsloth/Meta-Llama-3.1-8B-Instruct", 22),
    "gemma9b": ("unsloth/gemma-2-9b-it", 29),
    "mistral7b": ("unsloth/mistral-7b-instruct-v0.3", 22),
    "phi35": ("microsoft/Phi-3.5-mini-instruct", 22),
    "lfm12b": ("LiquidAI/LFM2.5-1.2B-Instruct", 11),
}
SEEDS = [0, 1]


def cells():
    out = []
    for key in DONORS:
        out.append(("donor_alone", key, 0))
    out.append(("host_alone", "llama1b", 0))
    out.append(("host_alone", "qwen05", 0))
    for key in DONORS:
        for seed in SEEDS:
            out.append(("tap_llama1b", key, seed))
    for pair in ["qwen7b+gemma9b", "qwen7b+llama8b", "gemma9b+llama8b", "qwen14b+gemma9b"]:
        for seed in SEEDS:
            out.append(("two_taps_llama1b", pair, seed))
    for key in ["qwen7b", "qwen14b", "gemma9b"]:
        for seed in SEEDS:
            out.append(("tap_qwen05", key, seed))
    return out


def load(name):
    import torch
    import transformers

    kwargs = {"dtype": torch.bfloat16}
    if "gemma" in name:
        kwargs["attn_implementation"] = "eager"
    model = transformers.AutoModelForCausalLM.from_pretrained(name, **kwargs).to("cuda").eval()
    tokenizer = transformers.AutoTokenizer.from_pretrained(name)
    return model, tokenizer


@app.function(image=image, volumes={"/vol": volume}, gpu="H100", memory=131072, timeout=7200)
def run_cell(kind, key, seed):
    import torch

    from modal_round6 import HOST, mmlu_data, param_groups, score_mmlu
    from wsp.tap import TapBranch, attach_taps, tap_train

    start = time.time()
    row = {"kind": kind, "key": key, "seed": seed}
    if kind == "donor_alone":
        model, tokenizer = load(DONORS[key][0])
        pairs, items = mmlu_data(tokenizer=None)
        row["accuracy"] = score_mmlu(model, tokenizer, items)
        return row
    host_name, host_layer = LLAMA1B
    if kind == "tap_qwen05" or key == "qwen05":
        host_name, host_layer = QWEN05
    host, tokenizer = load(host_name)
    pairs, items = mmlu_data(tokenizer=tokenizer)
    if kind == "host_alone":
        row["accuracy"] = score_mmlu(host, tokenizer, items)
        return row

    torch.manual_seed(seed)
    keys = key.split("+")
    donors = {}
    placements = []
    for index, donor_key in enumerate(keys):
        donor_name, donor_layer = DONORS[donor_key]
        donor, donor_tokenizer = load(donor_name)
        donors[donor_key] = (donor, donor_tokenizer)
        layer = host_layer if index == 0 else min(host_layer + 2, host.config.num_hidden_layers - 2)
        branch = TapBranch(host.config.hidden_size, donor.config.hidden_size, attn_dim=256, heads=8,
                           donor_id=donor_key, donor_layer=donor_layer)
        placements.append((layer, branch, index))
    attach_taps(host, placements)
    host.wsp_taps.to("cuda")
    losses = tap_train(host, tokenizer, donors, pairs, param_groups(host, [], 1e-4, True), steps=500, lr=3e-4,
                       batch_size=8, seq_len=256, seed=seed)
    row["accuracy"] = score_mmlu(host, tokenizer, items, donors=donors)
    gates = []
    for branch in host.wsp_taps:
        gates.append(float(branch.gate.detach()))
    row["gates"] = gates
    row["loss_last10"] = sum(losses[-10:]) / 10
    row["seconds"] = round(time.time() - start, 1)
    row["host"] = host_name
    return row


@app.local_entrypoint()
def main(shard: int = 0, nshards: int = 1):
    mine = []
    for index, cell in enumerate(cells()):
        if index % nshards == shard:
            mine.append(cell)
    print(f"shard {shard}/{nshards}: {len(mine)} cells")
    rows = []
    for row in run_cell.starmap(mine, return_exceptions=True):
        if isinstance(row, Exception):
            print("cell failed:", repr(row)[:500])
            continue
        rows.append(row)
        print(json.dumps({k: row.get(k) for k in ["kind", "key", "seed", "accuracy", "gates"]}))
    os.makedirs("results", exist_ok=True)
    with open(f"results/round8_shard{shard}.json", "w") as f:
        json.dump(rows, f, indent=2)
