"""Round 5: taps. A frozen host reads a frozen donor's own states.

Rounds 1 to 4 ran donor layers on a mapped host stream, and the stitch
broke the donor. Here both models stay native (wsp/tap.py). The donor
reads the prompt with its own tokenizer and runs up to one layer. A small
gated attention branch at one host layer reads those states. Only the
branches train, with a completion-only loss on GSM8K heal pairs.

Host unsloth/Llama-3.2-1B-Instruct, donor Qwen/Qwen2.5-Math-1.5B-Instruct.
Arms (3 seeds for every trained arm):
  host_only           the host, no training
  tap_l14_h10         donor layer 14 into host layer 10
  tap_l27_h12         donor last layer (27) into host layer 12
  tap_two             both taps
  tap_derange         tap_l27_h12, each example reads another example's
                      donor memory, in training and in scoring
  tap_shuffled_donor  tap_l27_h12 with a donor whose weights are permuted
                      inside each tensor (seeded)
  lora_r64            Program LoRA rank 64, lr 5e-5 (the round 4 bar)
  donor_alone         the donor itself, same split and prompt

Search-split numbers are interim checks, not results.

    MODAL_PROFILE=trisanth modal run modal_round5.py
"""

import json
import os
import time

import modal

app = modal.App("wsp-round5")
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
DONOR = "Qwen/Qwen2.5-Math-1.5B-Instruct"
DONOR_ID = "qmath_instruct"
SEEDS = [0, 1, 2]
STEPS = 300
BRANCH_LR = 3e-4
GATE_LR = 3e-2
LFM = "LiquidAI/LFM2-350M-Math"
LFM_ID = "lfm2_350m_math"
BATCH_SIZE = 8
SEQ_LEN = 384
ATTN_DIM = 256
HEADS = 8

# Each tap arm is a list of (host_layer, donor_layer, tap_index).
TAP_ARMS = {
    "tap_l14_h10": [(10, 14, 0)],
    "tap_l27_h12": [(12, 27, 0)],
    "tap_two": [(10, 14, 0), (12, 27, 1)],
    "tap_derange": [(12, 27, 0)],
    "tap_shuffled_donor": [(12, 27, 0)],
    "tap_lfm_l15_h12": [(12, 15, 0)],
    "tap_lfm_l8_h8": [(8, 8, 0)],
}


def donor_for(arm):
    if arm.startswith("tap_lfm"):
        return LFM, LFM_ID
    return DONOR, DONOR_ID


def param_groups(host):
    """Branch weights at BRANCH_LR; gates at GATE_LR so a zero-init gate can open in 300 steps."""
    gate_params = []
    other_params = []
    for branch in host.wsp_taps:
        for name, param in branch.named_parameters():
            if name == "gate":
                gate_params.append(param)
            else:
                other_params.append(param)
    return [{"params": other_params, "lr": BRANCH_LR}, {"params": gate_params, "lr": GATE_LR}]


def gsm8k_pairs(items):
    pairs = []
    for item in items:
        pairs.append(("Question: " + item["question"] + "\nAnswer:", " " + item["answer"]))
    return pairs


def shuffle_weights(model, seed):
    """Permute the entries of every parameter tensor, in place, with a seeded generator."""
    import torch

    generator = torch.Generator()
    generator.manual_seed(seed)
    with torch.no_grad():
        for param in model.parameters():
            flat = param.detach().flatten()
            order = torch.randperm(flat.numel(), generator=generator).to(flat.device)
            param.copy_(flat[order].view_as(param))


def decode_speed(model, tokenizer, inputs=None, tokens=128):
    """Greedy decode tok/s for one prompt. inputs come from wsp.tap.prepare for a tapped host."""
    import torch

    if inputs is None:
        ids = tokenizer("Question: What is 12 times 7?\nAnswer:", return_tensors="pt").input_ids.to(model.device)
        inputs = {"input_ids": ids}
    prompt_len = inputs["input_ids"].shape[1]
    torch.cuda.synchronize()
    start = time.time()
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=tokens, min_new_tokens=tokens, do_sample=False,
                             pad_token_id=tokenizer.eos_token_id)
    torch.cuda.synchronize()
    return round((out.shape[1] - prompt_len) / (time.time() - start), 1)


def score_taps(model, tokenizer, donors, items, derange=False, batch_size=32, max_new_tokens=256):
    """modal_graft.score, with wsp.tap.prepare before each generate."""
    import torch

    from modal_graft import final_number
    from wsp.tap import clear_tap_context, prepare

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
        encoded = prepare(model, tokenizer, donors, prompts, derange=derange)
        with torch.no_grad():
            out = model.generate(**encoded, max_new_tokens=max_new_tokens, do_sample=False,
                                 pad_token_id=tokenizer.pad_token_id)
        clear_tap_context(model)
        for i, item in enumerate(batch):
            text = tokenizer.decode(out[i, encoded["input_ids"].shape[1]:], skip_special_tokens=True)
            text = text.split("Question:")[0]
            predicted = final_number(text)
            gold = final_number(item["answer"])
            if predicted is not None and gold is not None and abs(predicted - gold) < 1e-6:
                correct += 1
            if len(samples) < 3:
                samples.append(text[:300])
    return correct / len(items), samples


def run_plain(arm, seed, search):
    import torch
    import transformers

    from modal_graft import score

    name = HOST
    if arm == "donor_alone":
        name = DONOR
    elif arm == "lfm_alone":
        name = LFM
    model = transformers.AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16).to("cuda").eval()
    tokenizer = transformers.AutoTokenizer.from_pretrained(name)
    accuracy, samples = score(model, tokenizer, search)
    return {"arm": arm, "seed": seed, "accuracy": accuracy, "trainable_params": 0,
            "decode_tok_s": decode_speed(model, tokenizer), "donor_prefill_seconds": 0.0,
            "losses": [], "samples": samples[:2]}


def run_lora(arm, seed, search, heal_texts):
    import torch

    from wsp import Catalog, Program, load

    torch.manual_seed(seed)
    program = Program(host=HOST, seed=seed).lora(rank=64)
    program.heal(steps=STEPS, data="gsm8k_train", lr=5e-5, batch_size=BATCH_SIZE, seq_len=SEQ_LEN)
    out_dir = f"/vol/compiled/round5_{arm}_s{seed}"
    program.compile(out_dir, Catalog([], root="/vol"), align_texts=heal_texts[:200], heal_texts=heal_texts,
                    device="cuda", cache_dir="/vol/tensor_cache")
    volume.commit()
    report = json.load(open(f"{out_dir}/compile_report.json"))
    model, tokenizer = load(out_dir, device="cuda")

    from modal_graft import score

    accuracy, samples = score(model, tokenizer, search)
    heal = report.get("heal") or {}
    return {"arm": arm, "seed": seed, "accuracy": accuracy, "trainable_params": heal.get("trainable_params", 0),
            "decode_tok_s": decode_speed(model, tokenizer), "donor_prefill_seconds": 0.0,
            "losses": heal.get("losses", []), "lora": report.get("lora"), "samples": samples[:2]}


def run_tap(arm, seed, search, pairs):
    import torch
    import transformers

    from wsp.tap import (TapBranch, attach_taps, clear_tap_context, donor_flops_note, prepare, save_taps,
                         tap_param_count, tap_parameters, tap_train)

    host = transformers.AutoModelForCausalLM.from_pretrained(HOST, dtype=torch.bfloat16).to("cuda").eval()
    host_tokenizer = transformers.AutoTokenizer.from_pretrained(HOST)
    donor_name, donor_id = donor_for(arm)
    donor = transformers.AutoModelForCausalLM.from_pretrained(donor_name, dtype=torch.bfloat16).to("cuda").eval()
    donor_tokenizer = transformers.AutoTokenizer.from_pretrained(donor_name)
    if arm == "tap_shuffled_donor":
        shuffle_weights(donor, seed + 1000)
    donors = {donor_id: (donor, donor_tokenizer)}
    derange = arm == "tap_derange"

    torch.manual_seed(seed)
    placements = []
    for host_layer, donor_layer, tap_index in TAP_ARMS[arm]:
        branch = TapBranch(host.config.hidden_size, donor.config.hidden_size, attn_dim=ATTN_DIM, heads=HEADS,
                           donor_id=donor_id, donor_layer=donor_layer)
        placements.append((host_layer, branch, tap_index))
    attach_taps(host, placements)

    start = time.time()
    losses = tap_train(host, host_tokenizer, donors, pairs, param_groups(host), steps=STEPS, lr=BRANCH_LR,
                       batch_size=BATCH_SIZE, seq_len=SEQ_LEN, seed=seed, derange=derange)
    train_seconds = round(time.time() - start, 1)
    out_dir = f"/vol/taps/round5_{arm}_s{seed}"
    save_taps(host, out_dir, meta={"donors": {donor_id: donor_name}, "host": HOST, "arm": arm, "seed": seed,
                                   "derange": derange, "shuffled_donor": arm == "tap_shuffled_donor"})
    volume.commit()

    host.wsp_tap_stats["donor_seconds"] = 0.0
    host.wsp_tap_stats["donor_runs"] = 0
    host.wsp_tap_stats["donor_tokens"] = 0
    accuracy, samples = score_taps(host, host_tokenizer, donors, search, derange=derange)
    score_runs = host.wsp_tap_stats["donor_runs"]
    score_donor_seconds = host.wsp_tap_stats["donor_seconds"]
    notes = donor_flops_note(host, donors)

    # Decode speed for one prompt, without derange (it needs 2 prompts and
    # does not change the cost). The donor prefill for it is timed apart.
    host.wsp_tap_stats["donor_seconds"] = 0.0
    inputs = prepare(host, host_tokenizer, donors, ["Question: What is 12 times 7?\nAnswer:"])
    one_prompt_donor_seconds = host.wsp_tap_stats["donor_seconds"]
    tok_s = decode_speed(host, host_tokenizer, inputs)
    clear_tap_context(host)

    gates = []
    for branch in host.wsp_taps:
        gates.append(float(branch.gate.detach()))
    return {"arm": arm, "seed": seed, "accuracy": accuracy, "trainable_params": tap_param_count(host),
            "decode_tok_s": tok_s, "donor_prefill_seconds": round(one_prompt_donor_seconds, 4),
            "score_donor_seconds_per_batch": round(score_donor_seconds / max(1, score_runs), 4),
            "train_seconds": train_seconds, "gates": gates, "donor_note": notes,
            "losses": losses, "samples": samples[:2]}


@app.function(image=image, volumes={"/vol": volume}, gpu="H100", memory=65536, timeout=5400)
def run_cell(arm, seed):
    from wsp.data import format_gsm8k, gsm8k_splits

    search, heal_items = gsm8k_splits(seed=0)
    if arm in ["host_only", "donor_alone", "lfm_alone"]:
        return run_plain(arm, seed, search)
    if arm == "lora_r64":
        heal_texts = []
        for item in heal_items:
            heal_texts.append(format_gsm8k(item))
        return run_lora(arm, seed, search, heal_texts)
    return run_tap(arm, seed, search, gsm8k_pairs(heal_items))


@app.local_entrypoint()
def main():
    cells = []
    for arm in ["host_only", "donor_alone", "lfm_alone"]:
        # Greedy scoring with no training gives the same number for every seed.
        cells.append((arm, 0))
    for arm in ["tap_l14_h10", "tap_l27_h12", "tap_two", "tap_derange", "tap_shuffled_donor",
                "tap_lfm_l15_h12", "tap_lfm_l8_h8", "lora_r64"]:
        for seed in SEEDS:
            cells.append((arm, seed))
    rows = []
    for row in run_cell.starmap(cells, return_exceptions=True):
        if isinstance(row, Exception):
            print("cell failed:", repr(row)[:400])
            continue
        rows.append(row)
        print(json.dumps({key: row.get(key) for key in ["arm", "seed", "accuracy", "trainable_params",
                                                       "decode_tok_s", "donor_prefill_seconds"]}))
    os.makedirs("results", exist_ok=True)
    with open("results/round5_rows.json", "w") as f:
        json.dump({"host": HOST, "donor": DONOR, "steps": STEPS, "branch_lr": BRANCH_LR, "gate_lr": GATE_LR, "rows": rows}, f,
                  indent=2)
