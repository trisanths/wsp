"""First cross-family graft run on Modal.

Host: unsloth/Llama-3.2-1B-Instruct. Donor: Qwen/Qwen2.5-Math-1.5B.
Part: the top 512 MLP neurons of donor layer 18 by mean |activation| on
GSM8K heal text. Three models on the 200-item GSM8K search split:
  host      the host, no graft, no heal
  control   a random 512-neuron cluster from the same layer + glue + heal
  graft     the top-activation cluster + glue + heal
These are search-split numbers: an interim check, not results.

Run from a committed snapshot (see scripts/snapshot_run.sh):
    MODAL_PROFILE=trisanth modal run modal_graft.py
"""

import json
import re
import time

import modal

app = modal.App("wsp-graft")
volume = modal.Volume.from_name("wsp-models", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch==2.7.1", "transformers==5.23.0", "accelerate", "huggingface_hub", "requests",
        "safetensors", "zstandard", "numpy", "datasets",
    )
    .env({"HF_HOME": "/vol/hf"})
    .add_local_python_source("wsp")
)

HOST = "unsloth/Llama-3.2-1B-Instruct"
DONOR = "Qwen/Qwen2.5-Math-1.5B"
DONOR_LAYER = 18
NEURONS = 512


def final_number(text):
    """Return the number after '####', else the last number in the text."""
    match = re.search(r"####\s*(-?[\d,]*\.?\d+)", text)
    if match is None:
        numbers = re.findall(r"-?[\d,]*\.?\d+", text)
        if len(numbers) == 0:
            return None
        value = numbers[-1]
    else:
        value = match.group(1)
    value = value.replace(",", "").rstrip(".")
    try:
        return float(value)
    except ValueError:
        return None


def score(model, tokenizer, items, batch_size=32, max_new_tokens=256):
    import torch

    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    correct = 0
    samples = []
    for start in range(0, len(items), batch_size):
        batch = items[start:start + batch_size]
        prompts = []
        for item in batch:
            prompts.append("Question: " + item["question"] + "\nAnswer:")
        encoded = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
        with torch.no_grad():
            out = model.generate(**encoded, max_new_tokens=max_new_tokens, do_sample=False,
                                 pad_token_id=tokenizer.pad_token_id)
        for i, item in enumerate(batch):
            text = tokenizer.decode(out[i, encoded.input_ids.shape[1]:], skip_special_tokens=True)
            text = text.split("Question:")[0]
            predicted = final_number(text)
            gold = final_number(item["answer"])
            if predicted is not None and gold is not None and abs(predicted - gold) < 1e-6:
                correct += 1
            if len(samples) < 3:
                samples.append(text[:300])
    return correct / len(items), samples


@app.function(image=image, volumes={"/vol": volume}, gpu="H100", memory=65536, timeout=3600)
def run(heal_steps=300, seed=0):
    import torch
    import transformers

    from wsp import Catalog, Program, load
    from wsp.data import format_gsm8k, gsm8k_splits
    from wsp.extract import top_activation_neurons

    torch.manual_seed(seed)
    results = {"host": HOST, "donor": DONOR, "donor_layer": DONOR_LAYER, "neurons": NEURONS,
               "heal_steps": heal_steps, "seed": seed, "split": "search (200 from GSM8K train)"}
    search, heal_items = gsm8k_splits(seed=0)
    heal_texts = []
    for item in heal_items:
        heal_texts.append(format_gsm8k(item))
    align_texts = heal_texts[:200]

    start = time.time()
    donor = transformers.AutoModelForCausalLM.from_pretrained(DONOR, dtype=torch.bfloat16).to("cuda").eval()
    donor_tok = transformers.AutoTokenizer.from_pretrained(DONOR)
    indices = top_activation_neurons(donor, donor_tok, heal_texts[:300], donor.model.layers[DONOR_LAYER].mlp, NEURONS)
    del donor
    torch.cuda.empty_cache()
    units_path = f"/vol/units/qmath_l{DONOR_LAYER}_top{NEURONS}.pt"
    import os
    os.makedirs("/vol/units", exist_ok=True)
    torch.save({"kind": "mlp_cluster", "model": DONOR, "layer": DONOR_LAYER, "indices": indices}, units_path)
    results["extract_seconds"] = round(time.time() - start, 1)

    record = {"id": "qmath_l18_top512", "type": "mlp_cluster", "model": DONOR, "layer": DONOR_LAYER,
              "units": units_path, "trained_on": "top mean |activation| on GSM8K train text"}
    catalog = Catalog([record], root="/vol")

    host_model = transformers.AutoModelForCausalLM.from_pretrained(HOST, dtype=torch.bfloat16).to("cuda").eval()
    host_tok = transformers.AutoTokenizer.from_pretrained(HOST)
    accuracy, samples = score(host_model, host_tok, search)
    results["host_accuracy"] = accuracy
    results["host_samples"] = samples
    del host_model
    torch.cuda.empty_cache()

    for label, control in [("control", "random"), ("graft", None)]:
        start = time.time()
        program = Program(host=HOST, seed=seed).graft("qmath_l18_top512", control=control, control_seed=seed + 1)
        program.heal(steps=heal_steps, data="gsm8k_train", lr=3e-4, batch_size=8, seq_len=384)
        out_dir = f"/vol/compiled/graft_run1_{label}_s{seed}"
        program.compile(out_dir, catalog, align_texts=align_texts, heal_texts=heal_texts, device="cuda",
                        cache_dir="/vol/tensor_cache")
        compile_seconds = round(time.time() - start, 1)
        with open(f"{out_dir}/compile_report.json") as f:
            report = json.load(f)
        model, tokenizer = load(out_dir, device="cuda")
        accuracy, samples = score(model, tokenizer, search)
        graft_meta = report["grafts"][0]
        losses = report["heal"]["losses"]
        results[label] = {
            "accuracy": accuracy,
            "fit_r2": graft_meta["fit_r2"],
            "fit_pairs": graft_meta["fit_pairs"],
            "host_layer": graft_meta["host_layer"],
            "part_params": graft_meta["part_params"],
            "glue_params": graft_meta["glue_params"],
            "heal_loss_first10": sum(losses[:10]) / 10,
            "heal_loss_last10": sum(losses[-10:]) / 10,
            "compile_seconds": compile_seconds,
            "samples": samples,
        }
        del model
        torch.cuda.empty_cache()
        volume.commit()

    with open(f"/vol/results/graft_run1_s{seed}.json", "w") if os.path.isdir("/vol/results") else open(
            f"/vol/graft_run1_s{seed}.json", "w") as f:
        json.dump(results, f, indent=2)
    volume.commit()
    return results


@app.local_entrypoint()
def main(heal_steps: int = 300, seed: int = 0):
    results = run.remote(heal_steps, seed)
    print(json.dumps(results, indent=2))
    with open(f"outputs/graft_run1_s{seed}.json", "w") as f:
        json.dump(results, f, indent=2)
