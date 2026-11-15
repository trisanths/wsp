"""Round 15: write a smaller model from a bigger model's parts.

A written model is one standard Hugging Face model at the requested size,
assembled from the most useful layers of a stronger model. No donor runs
beside it at inference and it is not larger than requested.

Carve: score every decoder layer of the source by how much it changes the
residual stream on calibration text (block influence = 1 - cosine between
the layer's input and output, averaged over tokens; ShortGPT, 2024), keep
the highest-influence layers (always the first and the last) to hit a
parameter budget, re-index them, heal with a LoRA on a general plus task
text mix, merge the LoRA, and save a plain model.

Comparisons at matched size: the official smaller model of the same family
(trained from scratch by its lab). Also a uniform-drop carve (every n-th
layer) and the carve without healing.

Benchmarks (no single domain): MMLU dev 1000 and held-out 2000 (round 6
and 7 items), MedMCQA 1000 (round 10 items), all by log-prob of A to D;
GSM8K search split 200 with the raw prompt (modal_graft.score).

Cells:
  official   Qwen2.5-7B-Instruct, Qwen2.5-14B-Instruct (the matched-size bars)
  source     Qwen2.5-14B-Instruct, Qwen2.5-32B-Instruct
  carve      14B -> keep 22 of 48 layers (about 7.6B, Qwen-7B size), 32B -> keep 27 of 64 (about 14.7B, Qwen-14B size)
             selection: influence or uniform; heal: none or LoRA r64 1000 steps

    python scripts/modal_fanout.py modal_round15.py round15 --profiles a,b,c
"""

import json
import os
import random
import time

import modal

app = modal.App("wsp-round15")
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
)

SOURCES = {
    "q14b": ("Qwen/Qwen2.5-14B-Instruct", 22),
    "q32b": ("Qwen/Qwen2.5-32B-Instruct", 27),
}
OFFICIAL = ["Qwen/Qwen2.5-7B-Instruct", "Qwen/Qwen2.5-14B-Instruct", "Qwen/Qwen2.5-32B-Instruct"]


def cells():
    out = []
    for name in OFFICIAL:
        out.append(("official", name, "", "", 0))
    for key in SOURCES:
        for select in ["influence", "uniform"]:
            for heal in ["none", "lora"]:
                out.append(("carve", key, select, heal, 0))
    out.append(("carve", "q14b", "influence", "lora", 1))
    out.append(("carve", "q32b", "influence", "lora", 1))
    return out


def calibration_texts(count=256):
    from datasets import load_dataset

    data = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="train")
    texts = []
    for row in data:
        text = row["text"].strip()
        if len(text) > 400 and not text.startswith("="):
            texts.append(text[:2000])
        if len(texts) >= count:
            break
    return texts


def heal_texts(tokenizer):
    """General text plus multi-domain task text, so the carve is not tuned to one benchmark."""
    from datasets import load_dataset

    from modal_round6 import mmlu_data
    from wsp.data import format_gsm8k, gsm8k_splits

    texts = calibration_texts(3000)[256:]
    pairs, items = mmlu_data(tokenizer=tokenizer, train_count=3000)
    for prompt, completion in pairs:
        texts.append(prompt + completion)
    search, heal_items = gsm8k_splits(seed=0)
    for item in heal_items[:2000]:
        texts.append(format_gsm8k(item))
    random.Random(0).shuffle(texts)
    return texts


def layer_influence(model, tokenizer, texts):
    """Mean (1 - cosine(input, output)) per decoder layer over calibration tokens."""
    import torch

    from wsp.parts import find_decoder_layers

    _, layers = find_decoder_layers(model, 0)
    sums = [0.0] * len(layers)
    counts = [0] * len(layers)
    store = {}
    handles = []
    for index, layer in enumerate(layers):
        def pre(module, args, kwargs, index=index):
            store[index] = (args[0] if len(args) > 0 else kwargs["hidden_states"]).detach()

        def post(module, args, output, index=index):
            out = output[0] if isinstance(output, tuple) else output
            cos = torch.nn.functional.cosine_similarity(store[index].float(), out.detach().float(), dim=-1)
            sums[index] += float((1 - cos).sum())
            counts[index] += cos.numel()

        handles.append(layer.register_forward_pre_hook(pre, with_kwargs=True))
        handles.append(layer.register_forward_hook(post))
    try:
        for text in texts:
            ids = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).input_ids.to(model.device)
            with torch.no_grad():
                model(ids)
    finally:
        for handle in handles:
            handle.remove()
    influence = []
    for index in range(len(layers)):
        influence.append(sums[index] / max(1, counts[index]))
    return influence


def choose_layers(influence, keep, select):
    total = len(influence)
    if select == "uniform":
        chosen = set()
        for step in range(keep):
            chosen.add(round(step * (total - 1) / (keep - 1)))
        return sorted(chosen)
    middle = list(range(1, total - 1))
    middle.sort(key=lambda index: influence[index], reverse=True)
    chosen = [0, total - 1] + middle[:keep - 2]
    return sorted(chosen)


def carve(model, keep_layers):
    """Keep only keep_layers, re-indexed, in place. The result is a plain model of the same class."""
    import torch

    from wsp.parts import find_decoder_layers

    name, layers = find_decoder_layers(model, 0)
    kept = []
    for new_index, old_index in enumerate(keep_layers):
        layer = layers[old_index]
        if hasattr(layer, "self_attn") and hasattr(layer.self_attn, "layer_idx"):
            layer.self_attn.layer_idx = new_index
        kept.append(layer)
    parent = model.get_submodule(name.rsplit(".", 1)[0]) if "." in name else model
    setattr(parent, name.rsplit(".", 1)[-1], torch.nn.ModuleList(kept))
    config = model.config
    old_types = getattr(config, "layer_types", None)
    config.num_hidden_layers = len(keep_layers)
    if old_types is not None:
        new_types = []
        for old_index in keep_layers:
            new_types.append(old_types[old_index])
        config.layer_types = new_types
    return model


def count_params(model):
    total = 0
    for param in model.parameters():
        total += param.numel()
    return total


def evaluate(model, tokenizer):
    from modal_graft import score
    from modal_round6 import score_mmlu
    from modal_round7 import mmlu_split
    from wsp.data import gsm8k_splits

    _, dev = mmlu_split(tokenizer, heldout=False)
    _, held = mmlu_split(tokenizer, heldout=True)
    med = []
    from modal_round10 import items as r10_items

    for item in r10_items():
        if item["set"] == "medmcqa":
            med.append((item["prompt"], item["gold"]))
    search, _ = gsm8k_splits(seed=0)
    out = {
        "mmlu_dev": score_mmlu(model, tokenizer, dev),
        "mmlu_heldout": score_mmlu(model, tokenizer, held),
        "medmcqa": score_mmlu(model, tokenizer, med),
    }
    out["gsm8k_raw"], _ = score(model, tokenizer, search, batch_size=32, max_new_tokens=256)
    return out


@app.function(image=image, volumes={"/vol": volume}, gpu="H200", memory=262144, timeout=6 * 3600)
def run_cell(kind, key, select, heal, seed):
    import torch
    import transformers

    start = time.time()
    name = key if kind == "official" else SOURCES[key][0]
    model = transformers.AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16).to("cuda").eval()
    tokenizer = transformers.AutoTokenizer.from_pretrained(name)
    row = {"kind": kind, "model": name, "select": select, "heal": heal, "seed": seed}
    if kind == "carve":
        keep = SOURCES[key][1]
        influence = layer_influence(model, tokenizer, calibration_texts(256))
        keep_layers = choose_layers(influence, keep, select)
        row["influence"] = [round(x, 5) for x in influence]
        row["keep_layers"] = keep_layers
        carve(model, keep_layers)
        if heal == "lora":
            from wsp.heal import heal as run_heal
            from wsp.lora import apply_lora, merge_lora

            torch.manual_seed(seed)
            params = apply_lora(model, 64)
            losses = run_heal(model, tokenizer, heal_texts(tokenizer), params, steps=1000, lr=1e-4, batch_size=8,
                              seq_len=512, seed=seed)
            merge_lora(model)
            model.eval()
            row["heal_loss_first20"] = sum(losses[:20]) / 20
            row["heal_loss_last20"] = sum(losses[-20:]) / 20
        out_dir = f"/vol/written/round15_{key}_{select}_{heal}_s{seed}"
        model.save_pretrained(out_dir)
        tokenizer.save_pretrained(out_dir)
        volume.commit()
        row["saved"] = out_dir
    row["params_b"] = round(count_params(model) / 1e9, 3)
    row["layers"] = model.config.num_hidden_layers
    row["scores"] = evaluate(model, tokenizer)
    row["seconds"] = round(time.time() - start, 1)
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
        print(json.dumps({k: row.get(k) for k in ["kind", "model", "select", "heal", "params_b", "layers", "scores"]}))
    os.makedirs("results", exist_ok=True)
    with open(f"results/round15_shard{shard}.json", "w") as f:
        json.dump(rows, f, indent=2)
