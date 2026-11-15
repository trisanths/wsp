"""Round 9: can composition beat the best source model?

A single tap can only pass on what its donor knows, so a weak host plus a
tap stays below the donor (round 7). Here the HOST is the strongest model
and the taps bring knowledge from other families. The bar is the host
alone: a composed model that beats it is better than any of its parts.

MMLU, round 6 dev items (1000 test items, log-probability of " A" to " D");
training on 8000 MMLU auxiliary_train pairs, 500 steps, batch 8, seq_len
256, branch lr 3e-4, gate lr 3e-2; 2 seeds. Taps read donors at about 70%
depth and write into the host at about 40% depth (round 7's layer map).

Fan out:
    python scripts/modal_fanout.py modal_round9.py round9 --profiles a,b,c
"""

import json
import os
import time

import modal

app = modal.App("wsp-round9")
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

# name, donor read layer (about 70% depth), host write layer (about 40% depth)
MODELS = {
    "q14b": ("Qwen/Qwen2.5-14B-Instruct", 34, 19),
    "q7b": ("Qwen/Qwen2.5-7B-Instruct", 20, 11),
    "gemma9b": ("unsloth/gemma-2-9b-it", 29, 17),
    "llama8b": ("unsloth/Meta-Llama-3.1-8B-Instruct", 22, 13),
}
SEEDS = [0, 1]


def cells():
    out = []
    for key in MODELS:
        out.append(("alone", key, "", 0))
    plans = [
        ("q14b", "gemma9b"),
        ("q14b", "llama8b"),
        ("q14b", "gemma9b+llama8b"),
        ("q14b", "q7b"),
        ("gemma9b", "q14b"),
        ("gemma9b", "q7b+llama8b"),
        ("q7b", "gemma9b"),
        ("llama8b", "gemma9b+q7b"),
    ]
    for host, donors in plans:
        for seed in SEEDS:
            out.append(("tap", host, donors, seed))
    for seed in SEEDS:
        out.append(("derange", "q14b", "gemma9b", seed))
        out.append(("lora", "q14b", "", seed))
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


@app.function(image=image, volumes={"/vol": volume}, gpu="H200", memory=196608, timeout=10800)
def run_cell(kind, host_key, donor_keys, seed):
    import torch

    from modal_round6 import mmlu_data, param_groups, score_mmlu
    from wsp.lora import apply_lora
    from wsp.tap import TapBranch, attach_taps, tap_train

    start = time.time()
    row = {"kind": kind, "host": host_key, "donors": donor_keys, "seed": seed}
    host_name, host_read, host_write = MODELS[host_key]
    host, tokenizer = load(host_name)
    pairs, items = mmlu_data(tokenizer=tokenizer)
    if kind == "alone":
        row["accuracy"] = score_mmlu(host, tokenizer, items)
        return row

    torch.manual_seed(seed)
    donors = {}
    placements = []
    lora_params = []
    keys = []
    if donor_keys:
        keys = donor_keys.split("+")
    for index, key in enumerate(keys):
        donor_name, donor_read, _ = MODELS[key]
        donor, donor_tokenizer = load(donor_name)
        donors[key] = (donor, donor_tokenizer)
        write = host_write + 2 * index
        branch = TapBranch(host.config.hidden_size, donor.config.hidden_size, attn_dim=256, heads=8,
                           donor_id=key, donor_layer=donor_read)
        placements.append((write, branch, index))
    if kind == "lora":
        # A LoRA needs a donor-free loop; reuse tap_train with a frozen null tap from a small donor.
        donor_name, donor_read, _ = MODELS["q7b"]
        donor, donor_tokenizer = load(donor_name)
        donors["q7b"] = (donor, donor_tokenizer)
        placements.append((host_write, TapBranch(host.config.hidden_size, donor.config.hidden_size, attn_dim=256,
                                                 heads=8, donor_id="q7b", donor_layer=donor_read), 0))
        lora_params = apply_lora(host, 16)
    attach_taps(host, placements)
    host.wsp_taps.to("cuda")
    derange = kind == "derange"
    groups = param_groups(host, lora_params, 1e-4, kind != "lora")
    losses = tap_train(host, tokenizer, donors, pairs, groups, steps=500, lr=3e-4, batch_size=8, seq_len=256,
                       seed=seed, derange=derange)
    row["accuracy"] = score_mmlu(host, tokenizer, items, donors=donors, derange=derange)
    gates = []
    for branch in host.wsp_taps:
        gates.append(float(branch.gate.detach()))
    row["gates"] = gates
    row["loss_last10"] = sum(losses[-10:]) / 10
    row["seconds"] = round(time.time() - start, 1)
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
        print(json.dumps({k: row.get(k) for k in ["kind", "host", "donors", "seed", "accuracy", "gates"]}))
    os.makedirs("results", exist_ok=True)
    with open(f"results/round9_shard{shard}.json", "w") as f:
        json.dump(rows, f, indent=2)
