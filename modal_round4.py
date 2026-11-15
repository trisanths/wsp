"""Round 4: is the stitch the bottleneck?

Two questions:
1. Stitch penalty. Replace the donor's own input to layer l with the
   ridge-mapped host state W_in(h_k), run the rest of the donor, and
   measure the donor's next-token loss on GSM8K text against its native
   loss. Only possible when host and donor share a tokenizer, so the
   positions line up one to one. Host Qwen2.5-0.5B-Instruct, donor
   Qwen2.5-Math-1.5B.
2. Same-tokenizer grafts. Host Qwen2.5-0.5B-Instruct with donor blocks
   from Qwen2.5-Math-1.5B, against LoRA baselines and a random block.
   Also a lower-lr LoRA on the Llama host from round 3, to check that the
   LoRA bar there was fair.

Search-split numbers are interim checks, not results.

    MODAL_PROFILE=trisanth modal run modal_round4.py
"""

import json
import os
import time

import modal

app = modal.App("wsp-round4")
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
    .add_local_file("modal_round3.py", "/root/modal_round3.py")
)

QWEN_HOST = "Qwen/Qwen2.5-0.5B-Instruct"
LLAMA_HOST = "unsloth/Llama-3.2-1B-Instruct"
DONOR = "Qwen/Qwen2.5-Math-1.5B"
SEEDS = [0, 1, 2]
RECORDS = [
    {"id": "qmath_block_8_19", "type": "block", "model": DONOR, "layers": [8, 19]},
    {"id": "qmath_block_4_23", "type": "block", "model": DONOR, "layers": [4, 23]},
]
QWEN_GLUE_R64 = 64 * (896 + 1536) * 2


@app.function(image=image, volumes={"/vol": volume}, gpu="H100", memory=65536, timeout=3600)
def stitch_penalty():
    """Donor loss with its layer-l input replaced by W_in(host layer k), for several (k, l)."""
    import torch
    import transformers

    from wsp.align import collect_pairs, layer_input_capture, ridge_fit
    from wsp.data import format_gsm8k, gsm8k_splits
    from wsp.parts import find_decoder_layers

    search, heal_items = gsm8k_splits(seed=0)
    texts = []
    for item in heal_items[:400]:
        texts.append(format_gsm8k(item))
    fit_texts = texts[:250]
    eval_texts = texts[250:400]

    host = transformers.AutoModelForCausalLM.from_pretrained(QWEN_HOST, dtype=torch.float32).to("cuda").eval()
    donor = transformers.AutoModelForCausalLM.from_pretrained(DONOR, dtype=torch.float32).to("cuda").eval()
    tokenizer = transformers.AutoTokenizer.from_pretrained(DONOR)
    _, host_layers = find_decoder_layers(host, 0)
    _, donor_layers = find_decoder_layers(donor, 0)

    def donor_loss(replace=None):
        total = 0.0
        count = 0
        for text in eval_texts:
            ids = tokenizer(text, return_tensors="pt").input_ids.to("cuda")
            handles = []
            if replace is not None:
                k, l, w = replace
                store = {}

                def host_hook(module, args, output):
                    store["h"] = output[0] if isinstance(output, tuple) else output

                handles.append(host_layers[k].register_forward_hook(host_hook))
                with torch.no_grad():
                    host(ids)

                def donor_pre(module, args, kwargs):
                    mapped = store["h"].float() @ w.T.to("cuda")
                    if len(args) > 0:
                        return (mapped,) + tuple(args[1:]), kwargs
                    kwargs["hidden_states"] = mapped
                    return args, kwargs

                handles.append(donor_layers[l].register_forward_pre_hook(donor_pre, with_kwargs=True))
            with torch.no_grad():
                loss = donor(ids, labels=ids).loss.item()
            for handle in handles:
                handle.remove()
            total += loss * (ids.shape[1] - 1)
            count += ids.shape[1] - 1
        return total / count

    rows = {"native_loss": donor_loss(), "pairs": []}
    for l in [4, 8, 14, 20]:
        k = round(l / len(donor_layers) * len(host_layers))
        capture = layer_input_capture(donor_layers[l])
        x, y = collect_pairs(host, tokenizer, k, donor, tokenizer, capture, fit_texts)
        w, r2, r2_dim = ridge_fit(x, y)
        loss = donor_loss((k, l, w))
        rows["pairs"].append({"host_layer": k, "donor_layer": l, "r2": r2, "r2_dim": r2_dim,
                              "stitched_loss": loss, "penalty": loss - rows["native_loss"]})
        print(json.dumps(rows["pairs"][-1]))
    return rows


def build_program(host, arm, seed):
    from wsp import Program

    program = Program(host=host, seed=seed)
    lr = 3e-4
    if arm == "host":
        return None
    if arm == "lora_r1":
        program.lora(rank=1)
    elif arm == "lora_r16_lr1e4":
        program.lora(rank=16)
        lr = 1e-4
    elif arm == "lora_r64_lr5e5":
        program.lora(rank=64)
        lr = 5e-5
    elif arm == "block_a_r64":
        program.graft("qmath_block_8_19", glue_rank=64)
    elif arm == "block_b_r64":
        program.graft("qmath_block_4_23", glue_rank=64)
    elif arm == "block_rand":
        program.graft("qmath_block_8_19", glue_rank=64, control="random", control_seed=seed + 100)
    else:
        raise ValueError(arm)
    program.heal(steps=300, data="gsm8k_train", lr=lr, batch_size=8, seq_len=384)
    return program


@app.function(image=image, volumes={"/vol": volume}, gpu="H100", memory=65536, timeout=5400)
def run_arm(host, arm, seed):
    import torch
    import transformers

    from modal_graft import score
    from modal_round3 import decode_speed
    from wsp import Catalog, load
    from wsp.data import format_gsm8k, gsm8k_splits

    search, heal_items = gsm8k_splits(seed=0)
    heal_texts = []
    for item in heal_items:
        heal_texts.append(format_gsm8k(item))
    tag = "qwen05" if host == QWEN_HOST else "llama1"
    program = build_program(host, arm, seed)
    start = time.time()
    report = {}
    if program is None:
        model = transformers.AutoModelForCausalLM.from_pretrained(host, dtype=torch.bfloat16).to("cuda").eval()
        tokenizer = transformers.AutoTokenizer.from_pretrained(host)
    else:
        torch.manual_seed(seed)
        out_dir = f"/vol/compiled/round4_{tag}_{arm}_s{seed}"
        program.compile(out_dir, Catalog(RECORDS, root="/vol"), align_texts=heal_texts[:200],
                        heal_texts=heal_texts, device="cuda", cache_dir="/vol/tensor_cache")
        report = json.load(open(f"{out_dir}/compile_report.json"))
        model, tokenizer = load(out_dir, device="cuda")
    compile_seconds = round(time.time() - start, 1)
    accuracy, samples = score(model, tokenizer, search)
    grafts = report.get("grafts", [])
    for meta in grafts:
        meta.pop("indices", None)
    heal = report.get("heal") or {}
    losses = heal.get("losses", [])
    return {"host": tag, "arm": arm, "seed": seed, "accuracy": accuracy,
            "trainable_params": heal.get("trainable_params", 0), "decode_tok_s": decode_speed(model, tokenizer),
            "compile_seconds": compile_seconds, "heal_loss_last10": sum(losses[-10:]) / max(1, len(losses[-10:])),
            "grafts": grafts, "lora": report.get("lora"), "samples": samples[:2]}


@app.local_entrypoint()
def main():
    penalty_call = stitch_penalty.spawn()
    cells = []
    for arm in ["host", "lora_r1", "lora_r16_lr1e4", "lora_r64_lr5e5", "block_a_r64", "block_b_r64", "block_rand"]:
        seeds = [0] if arm == "host" else SEEDS
        for seed in seeds:
            cells.append((QWEN_HOST, arm, seed))
    for arm in ["lora_r16_lr1e4", "lora_r64_lr5e5"]:
        for seed in SEEDS:
            cells.append((LLAMA_HOST, arm, seed))
    rows = []
    for row in run_arm.starmap(cells, return_exceptions=True):
        if isinstance(row, Exception):
            print("cell failed:", repr(row)[:400])
            continue
        rows.append(row)
        print(json.dumps({key: row[key] for key in ["host", "arm", "seed", "accuracy", "trainable_params", "decode_tok_s"]}))
    penalty = penalty_call.get()
    print(json.dumps(penalty, indent=2))
    os.makedirs("results", exist_ok=True)
    with open("results/round4_rows.json", "w") as f:
        json.dump({"stitch_penalty": penalty, "rows": rows}, f, indent=2)
