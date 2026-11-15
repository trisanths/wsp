"""Round 2: does a selected donor part beat a random part of the same size?

Host unsloth/Llama-3.2-1B-Instruct, donor Qwen/Qwen2.5-Math-1.5B, GSM8K.
Grid (each cell runs 3 seeds, scored on the 200-item search split):
  donor layer   10, 18, 24 (of 28)
  selection     activation, contrast, attribution, random (the control)
  glue rank     full, 64
Extraction runs once per (layer, selection) and saves units to the volume.
Compiles and evals fan out across containers.

Search-split numbers are interim checks, not results.

    MODAL_PROFILE=trisanth modal run modal_sweep.py
"""

import json
import os
import time

import modal

app = modal.App("wsp-sweep")
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
LAYERS = [10, 18, 24]
METHODS = ["activation", "contrast", "attribution"]
NEURONS = 512
SEEDS = [0, 1, 2]
RANKS = [None, 64]
HEAL_STEPS = 300


def generic_texts(count):
    from datasets import load_dataset

    data = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="train")
    texts = []
    for row in data:
        text = row["text"].strip()
        if len(text) > 200 and not text.startswith("="):
            texts.append(text)
        if len(texts) >= count:
            break
    return texts


@app.function(image=image, volumes={"/vol": volume}, gpu="H100", memory=65536, timeout=3600)
def extract_all():
    """Extract units for every (layer, method). Return the catalog records."""
    import torch
    import transformers

    from wsp.data import format_gsm8k, gsm8k_splits
    from wsp.extract import attribution_neurons, contrast_neurons, top_activation_neurons

    search, heal_items = gsm8k_splits(seed=0)
    task = []
    for item in heal_items[:300]:
        task.append(format_gsm8k(item))
    generic = generic_texts(300)

    donor = transformers.AutoModelForCausalLM.from_pretrained(DONOR, dtype=torch.bfloat16).to("cuda").eval()
    tokenizer = transformers.AutoTokenizer.from_pretrained(DONOR)
    os.makedirs("/vol/units", exist_ok=True)
    records = []
    for layer in LAYERS:
        mlp = donor.model.layers[layer].mlp
        for method in METHODS:
            start = time.time()
            if method == "activation":
                indices = top_activation_neurons(donor, tokenizer, task, mlp, NEURONS)
            elif method == "contrast":
                indices = contrast_neurons(donor, tokenizer, task, generic, mlp, NEURONS)
            else:
                donor.float()
                indices = attribution_neurons(donor, tokenizer, task[:150], mlp, NEURONS)
                donor.to(torch.bfloat16)
            component_id = f"qmath_l{layer}_{method}{NEURONS}"
            path = f"/vol/units/{component_id}.pt"
            torch.save({"kind": "mlp_cluster", "model": DONOR, "layer": layer, "indices": indices}, path)
            records.append({"id": component_id, "type": "mlp_cluster", "model": DONOR, "layer": layer,
                            "units": path, "trained_on": f"{method} on GSM8K train text",
                            "extract_seconds": round(time.time() - start, 1)})
    volume.commit()
    return records


@app.function(image=image, volumes={"/vol": volume}, gpu="H100", memory=65536, timeout=3600)
def run_cell(record, control, rank, seed):
    import torch

    from modal_graft import score
    from wsp import Catalog, Program, load
    from wsp.data import format_gsm8k, gsm8k_splits

    volume.reload()
    search, heal_items = gsm8k_splits(seed=0)
    heal_texts = []
    for item in heal_items:
        heal_texts.append(format_gsm8k(item))
    catalog = Catalog([record], root="/vol")

    label = "random" if control else record["id"].split("_")[-1]
    rank_label = "full" if rank is None else f"r{rank}"
    name = f"sweep2_l{record['layer']}_{label}_{rank_label}_s{seed}"
    if control:
        name = f"sweep2_l{record['layer']}_random_{rank_label}_s{seed}"
    out_dir = f"/vol/compiled/{name}"

    torch.manual_seed(seed)
    start = time.time()
    program = Program(host=HOST, seed=seed).graft(record["id"], control="random" if control else None,
                                                  control_seed=seed + 100, glue_rank=rank)
    program.heal(steps=HEAL_STEPS, data="gsm8k_train", lr=3e-4, batch_size=8, seq_len=384)
    program.compile(out_dir, catalog, align_texts=heal_texts[:200], heal_texts=heal_texts, device="cuda",
                    cache_dir="/vol/tensor_cache")
    compile_seconds = round(time.time() - start, 1)
    report = json.load(open(f"{out_dir}/compile_report.json"))
    model, tokenizer = load(out_dir, device="cuda")
    accuracy, samples = score(model, tokenizer, search)
    meta = report["grafts"][0]
    losses = report["heal"]["losses"]
    return {
        "name": name, "layer": record["layer"], "selection": "random" if control else label,
        "rank": rank_label, "seed": seed, "accuracy": accuracy,
        "fit_r2": meta.get("fit_r2"), "fit_r2_dim": meta.get("fit_r2_dim"),
        "host_layer": meta["host_layer"], "part_params": meta["part_params"], "glue_params": meta["glue_params"],
        "heal_loss_last10": sum(losses[-10:]) / 10, "compile_seconds": compile_seconds,
    }


@app.local_entrypoint()
def main():
    records = extract_all.remote()
    cells = []
    for record in records:
        for rank in RANKS:
            for seed in SEEDS:
                cells.append((record, False, rank, seed))
    for layer in LAYERS:
        record = None
        for candidate in records:
            if candidate["layer"] == layer and candidate["id"].endswith("activation512"):
                record = candidate
        for rank in RANKS:
            for seed in SEEDS:
                cells.append((record, True, rank, seed))
    print(f"{len(cells)} cells")
    rows = []
    for row in run_cell.starmap(cells, return_exceptions=True):
        if isinstance(row, Exception):
            print("cell failed:", repr(row)[:300])
            continue
        rows.append(row)
        print(json.dumps(row))
    os.makedirs("results", exist_ok=True)
    with open("results/sweep2_rows.json", "w") as f:
        json.dump({"records": records, "rows": rows}, f, indent=2)
