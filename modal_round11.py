"""Round 11: frontier MoE donors tapped into small hosts on MMLU, from a donor cache.

modal_extract_frontier.py runs each donor once and writes its states at
two read layers to /vol/tap_cache/<donor>/. Here every cell trains and
scores a tap on one H100 from that cache. No donor model is loaded: the
tap reads its memories with wsp.tap.CachedDonor.

Hosts and write layers:
  llama1b  unsloth/Llama-3.2-1B-Instruct   host layers 6 and 10
  qwen05   Qwen/Qwen2.5-0.5B-Instruct      host layers 9 and 15

Per donor: each read layer by each host and write layer, 2 seeds, plus a
derange control (memories rolled by one within each batch, in training
and scoring) for one config per host, 2 seeds, plus each host alone.
Training as round 6: 500 steps, branch lr 3e-4, gate lr 3e-2, batch 8,
seq_len 256, on the 8000 MMLU auxiliary_train pairs. Scoring with
modal_round6.score_mmlu on the 1000 dev items (round 6) and the 2000
held-out items (round 7).

Both hosts train on the same 8000 pairs, filtered with the Llama-3.2-1B
tokenizer, because those are the prompts in the cache. (Round 8 filtered
the Qwen host's pairs with its own tokenizer.)

Run in the workspace whose volume holds the cache:
    MODAL_PROFILE=trisanth modal run modal_round11.py --donor glm53flash
Writes results/round11_<donor>.json.
"""

import json
import os
import time

import modal

app = modal.App("wsp-round11")
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
    .add_local_file("modal_extract_frontier.py", "/root/modal_extract_frontier.py")
)

PAIR_TOKENIZER = "unsloth/Llama-3.2-1B-Instruct"
HOSTS = {
    "llama1b": ("unsloth/Llama-3.2-1B-Instruct", [6, 10]),
    "qwen05": ("Qwen/Qwen2.5-0.5B-Instruct", [9, 15]),
}
READ_LAYERS = {
    "glm53flash": [31, 36],
    "glm53": [54, 62],
    "dsv4pro": [42, 49],
    "kimik3": [65, 74],
}
SEEDS = [0, 1]
STEPS = 500


def cache_path(donor):
    return "/vol/tap_cache/" + donor


def cells(donor):
    """Return the (kind, host, host_layer, donor_layer, seed) cells for one donor."""
    out = []
    for host in HOSTS:
        out.append(("host_alone", host, None, None, 0))
    for host in HOSTS:
        host_layers = HOSTS[host][1]
        for donor_layer in READ_LAYERS[donor]:
            for host_layer in host_layers:
                for seed in SEEDS:
                    out.append(("tap", host, host_layer, donor_layer, seed))
        # Derange control: the deeper write layer and the shallower (about
        # 70% depth) read layer, the round 8 placement.
        for seed in SEEDS:
            out.append(("derange", host, host_layers[-1], READ_LAYERS[donor][0], seed))
    return out


@app.function(image=image, volumes={"/vol": volume}, gpu="H100", memory=131072, timeout=7200)
def run_cell(donor, kind, host_key, host_layer, donor_layer, seed):
    import torch
    import transformers

    from modal_extract_frontier import tap_prompt_sets
    from modal_round6 import param_groups, score_mmlu
    from wsp.tap import CachedDonor, TapBranch, attach_taps, tap_param_count, tap_train

    start = time.time()
    host_name = HOSTS[host_key][0]
    row = {"donor": donor, "kind": kind, "host": host_name, "host_key": host_key, "host_layer": host_layer,
           "donor_layer": donor_layer, "seed": seed}
    pairs, dev_items, heldout_items = tap_prompt_sets(transformers.AutoTokenizer.from_pretrained(PAIR_TOKENIZER))
    host = transformers.AutoModelForCausalLM.from_pretrained(host_name, dtype=torch.bfloat16).to("cuda").eval()
    tokenizer = transformers.AutoTokenizer.from_pretrained(host_name)
    if kind == "host_alone":
        row["dev_accuracy"] = score_mmlu(host, tokenizer, dev_items)
        row["heldout_accuracy"] = score_mmlu(host, tokenizer, heldout_items)
        row["seconds"] = round(time.time() - start, 1)
        return row

    # Keep every shard of the one read layer in memory: training batches
    # draw prompts from all shards.
    load_start = time.time()
    cached = CachedDonor(cache_path(donor), donor_layer, max_shards=None)
    donors = {donor: cached}
    row["donor_id"] = cached.donor_id
    row["donor_layers_total"] = cached.layers_total
    row["d_donor"] = cached.hidden_size

    torch.manual_seed(seed)
    branch = TapBranch(host.config.hidden_size, cached.hidden_size, attn_dim=256, heads=8, donor_id=donor,
                       donor_layer=donor_layer)
    attach_taps(host, [(host_layer, branch, 0)])
    host.wsp_taps.to("cuda")
    derange = kind == "derange"
    train_start = time.time()
    losses = tap_train(host, tokenizer, donors, pairs, param_groups(host, [], 1e-4, True), steps=STEPS, lr=3e-4,
                       batch_size=8, seq_len=256, seed=seed, derange=derange)
    row["train_seconds"] = round(time.time() - train_start, 1)
    row["dev_accuracy"] = score_mmlu(host, tokenizer, dev_items, donors=donors, derange=derange)
    row["heldout_accuracy"] = score_mmlu(host, tokenizer, heldout_items, donors=donors, derange=derange)
    gates = []
    for tap in host.wsp_taps:
        gates.append(float(tap.gate.detach()))
    row["gates"] = gates
    row["tap_params"] = tap_param_count(host)
    row["loss_first10"] = sum(losses[:10]) / 10
    row["loss_last10"] = sum(losses[-10:]) / 10
    row["cache_shard_reads"] = cached.shard_reads
    row["cache_seconds"] = round(host.wsp_tap_stats["donor_seconds"], 1)
    row["cache_open_seconds"] = round(train_start - load_start, 1)
    row["seconds"] = round(time.time() - start, 1)
    return row


@app.local_entrypoint()
def main(donor: str, shard: int = 0, nshards: int = 1):
    if donor not in READ_LAYERS:
        raise SystemExit(f"--donor must be one of {sorted(READ_LAYERS)}")
    mine = []
    for index, cell in enumerate(cells(donor)):
        if index % nshards == shard:
            mine.append((donor,) + cell)
    print(f"{donor}: shard {shard}/{nshards}, {len(mine)} cells")
    rows = []
    for row in run_cell.starmap(mine, return_exceptions=True):
        if isinstance(row, Exception):
            print("cell failed:", repr(row)[:500])
            continue
        rows.append(row)
        print(json.dumps({k: row.get(k) for k in ["kind", "host_key", "host_layer", "donor_layer", "seed",
                                                  "dev_accuracy", "heldout_accuracy", "gates"]}))
    os.makedirs("results", exist_ok=True)
    name = "round11_" + donor
    if nshards > 1:
        name += f"_shard{shard}"
    with open(os.path.join("results", name + ".json"), "w") as f:
        json.dump(rows, f, indent=2)
