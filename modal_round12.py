"""Round 12: a medical specialist that beats every source model and fine-tuning.

Host Qwen2.5-14B-Instruct (the best single model on MedMCQA in round 10,
63.5). Donors chosen for medical strength relative to their size: Llama-3.1-8B
(60.1 on MedMCQA), Phi-4 (57.1), Gemma-2-9B (56.4), and Qwen2.5-32B.

Training data is in-domain: 24000 MedMCQA train questions (4 options),
prompt format as round 6, completion " X". 1500 steps, batch 8, seq_len 256.
Eval items:
  medmcqa_dev      the 1000 MedMCQA validation items from round 10 (same ids,
                   so per-item results join round 10's per-model scores)
  medmcqa_heldout  1500 other MedMCQA validation items, scored once
  medqa            1273 MedQA-USMLE 4-option test items (transfer)
Arms (2 seeds unless noted):
  host                Qwen-14B alone (1 run)
  lora                Qwen-14B LoRA r16, the fine-tuning baseline, same data and steps
  tap_llama           + Llama-8B tap (read L22, write L19)
  tap_llama_phi4      + Llama-8B and Phi-4 taps
  tap_three           + Llama-8B, Phi-4, Gemma-9B taps
  tap_qwen32          + Qwen-32B tap (read L44, write L19)
  tap_llama_late      + Llama-8B final-layer tap written late (read L31, write L40)
  derange_llama_phi4  tap_llama_phi4 on mismatched messages (control)
  staged              tap_llama_phi4, then freeze taps and train LoRA r16
  small_q05           Qwen-0.5B host + Qwen-14B tap (efficiency)
  small_llama1b       Llama-1B host + Qwen-14B tap (efficiency)
Per-item records are saved for paired statistics and ensembles.

    python scripts/modal_fanout.py modal_round12.py round12 --profiles a,b,c
"""

import json
import os
import random
import time

import modal

app = modal.App("wsp-round12")
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

MODELS = {
    "q14b": ("Qwen/Qwen2.5-14B-Instruct", 34, 19),
    "q32b": ("Qwen/Qwen2.5-32B-Instruct", 44, 25),
    "llama8b": ("unsloth/Meta-Llama-3.1-8B-Instruct", 22, 13),
    "phi4": ("microsoft/phi-4", 28, 16),
    "gemma9b": ("unsloth/gemma-2-9b-it", 29, 17),
    "q05": ("Qwen/Qwen2.5-0.5B-Instruct", 17, 9),
    "llama1b": ("unsloth/Llama-3.2-1B-Instruct", 11, 6),
}

# arm: (host, [(donor, read_layer or None, write_layer or None)], lora_rank, staged, derange)
ARMS = {
    "host": ("q14b", [], None, False, False),
    "lora": ("q14b", [], 16, False, False),
    "tap_llama": ("q14b", [("llama8b", None, None)], None, False, False),
    "tap_llama_phi4": ("q14b", [("llama8b", None, None), ("phi4", None, 21)], None, False, False),
    "tap_three": ("q14b", [("llama8b", None, None), ("phi4", None, 21), ("gemma9b", None, 23)], None, False, False),
    "tap_qwen32": ("q14b", [("q32b", None, None)], None, False, False),
    "tap_llama_late": ("q14b", [("llama8b", 31, 40)], None, False, False),
    "derange_llama_phi4": ("q14b", [("llama8b", None, None), ("phi4", None, 21)], None, False, True),
    "staged": ("q14b", [("llama8b", None, None), ("phi4", None, 21)], 16, True, False),
    "small_q05": ("q05", [("q14b", None, None)], None, False, False),
    "small_llama1b": ("llama1b", [("q14b", None, None)], None, False, False),
}
SEEDS = [0, 1]
STEPS = 1500


def cells():
    out = []
    for arm in ARMS:
        seeds = [0] if arm == "host" else SEEDS
        for seed in seeds:
            out.append((arm, seed))
    return out


def med_prompt(question, options):
    from modal_round6 import mmlu_prompt

    return mmlu_prompt(question, options)


def data(tokenizer):
    """Train pairs, and eval sets as lists of dicts with id, prompt, gold."""
    from datasets import load_dataset

    train = load_dataset("openlifescienceai/medmcqa", split="train")
    order = list(range(len(train)))
    random.Random(3).shuffle(order)
    pairs = []
    for index in order:
        row = train[index]
        if row["cop"] is None or int(row["cop"]) not in (0, 1, 2, 3):
            continue
        prompt = med_prompt(row["question"], [row["opa"], row["opb"], row["opc"], row["opd"]])
        if len(tokenizer(prompt).input_ids) > 240:
            continue
        pairs.append((prompt, " " + "ABCD"[int(row["cop"])]))
        if len(pairs) >= 24000:
            break

    valid = load_dataset("openlifescienceai/medmcqa", split="validation")
    order = list(range(len(valid)))
    random.Random(7).shuffle(order)
    sets = {"medmcqa_dev": [], "medmcqa_heldout": [], "medqa": []}
    for position, index in enumerate(order[:2500]):
        row = valid[index]
        item = {"id": f"medmcqa:{index}", "subject": row.get("subject_name") or "unknown",
                "prompt": med_prompt(row["question"], [row["opa"], row["opb"], row["opc"], row["opd"]]),
                "gold": int(row["cop"])}
        if position < 1000:
            sets["medmcqa_dev"].append(item)
        else:
            sets["medmcqa_heldout"].append(item)

    medqa = load_dataset("GBaker/MedQA-USMLE-4-options", split="test")
    for index in range(len(medqa)):
        row = medqa[index]
        letters = ["A", "B", "C", "D"]
        options = []
        for letter in letters:
            options.append(row["options"][letter])
        sets["medqa"].append({"id": f"medqa:{index}", "subject": row.get("meta_info") or "usmle",
                              "prompt": med_prompt(row["question"], options),
                              "gold": letters.index(row["answer_idx"])})
    return pairs, sets


def load(name):
    import torch
    import transformers

    kwargs = {"dtype": torch.bfloat16}
    if "gemma" in name:
        kwargs["attn_implementation"] = "eager"
    model = transformers.AutoModelForCausalLM.from_pretrained(name, **kwargs).to("cuda").eval()
    tokenizer = transformers.AutoTokenizer.from_pretrained(name)
    return model, tokenizer


def score(model, tokenizer, items, donors=None, derange=False, batch_size=16):
    """Per-item records with option probabilities."""
    import torch

    from modal_round6 import letter_ids
    from wsp.tap import clear_tap_context, prepare

    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    choice_ids = letter_ids(tokenizer)
    records = []
    for start in range(0, len(items), batch_size):
        batch = items[start:start + batch_size]
        prompts = []
        for item in batch:
            prompts.append(item["prompt"])
        if donors is not None:
            encoded = prepare(model, tokenizer, donors, prompts, derange=derange)
        else:
            encoded = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
            encoded = {"input_ids": encoded["input_ids"], "attention_mask": encoded["attention_mask"]}
        with torch.no_grad():
            logits = model(**encoded).logits[:, -1, :].float()
        if donors is not None:
            clear_tap_context(model)
        probs = torch.softmax(logits[:, choice_ids], dim=-1).cpu()
        for row, item in enumerate(batch):
            pred = int(probs[row].argmax())
            records.append({"id": item["id"], "subject": item["subject"], "gold": item["gold"], "pred": pred,
                            "probs": [round(x, 5) for x in probs[row].tolist()], "correct": pred == item["gold"]})
    return records


@app.function(image=image, volumes={"/vol": volume}, gpu="H200", memory=262144, timeout=6 * 3600)
def run_cell(arm, seed):
    import torch

    from modal_round6 import param_groups
    from wsp.lora import apply_lora
    from wsp.tap import TapBranch, attach_taps, tap_train

    start = time.time()
    host_key, taps, lora_rank, staged, derange = ARMS[arm]
    host_name, _, host_write = MODELS[host_key]
    host, tokenizer = load(host_name)
    pairs, sets = data(tokenizer)
    row = {"arm": arm, "seed": seed, "host": host_key}
    torch.manual_seed(seed)

    donors = {}
    placements = []
    for index, (donor_key, read, write) in enumerate(taps):
        donor_name, donor_read, _ = MODELS[donor_key]
        donor, donor_tokenizer = load(donor_name)
        donors[donor_key] = (donor, donor_tokenizer)
        if read is None:
            read = donor_read
        if write is None:
            write = host_write
        branch = TapBranch(host.config.hidden_size, donor.config.hidden_size, attn_dim=256, heads=8,
                           donor_id=donor_key, donor_layer=read)
        placements.append((write, branch, index))
    if len(placements) == 0 and lora_rank is not None:
        # tap_train needs a donor; use a frozen null tap from a small donor so the loop is identical.
        donor, donor_tokenizer = load(MODELS["llama1b"][0])
        donors["null"] = (donor, donor_tokenizer)
        placements.append((host_write, TapBranch(host.config.hidden_size, donor.config.hidden_size, attn_dim=256,
                                                 heads=8, donor_id="null", donor_layer=4), 0))

    losses = []
    if len(placements) > 0:
        attach_taps(host, placements)
        host.wsp_taps.to("cuda")
    if arm != "host":
        train_taps = arm not in ("lora",)
        lora_params = []
        if lora_rank is not None and not staged:
            lora_params = apply_lora(host, lora_rank)
        groups = param_groups(host, lora_params, 1e-4, train_taps)
        losses = tap_train(host, tokenizer, donors, pairs, groups, steps=STEPS, lr=3e-4, batch_size=8,
                           seq_len=256, seed=seed, derange=derange)
        if staged:
            for branch in host.wsp_taps:
                for param in branch.parameters():
                    param.requires_grad_(False)
            lora_params = apply_lora(host, lora_rank)
            groups = param_groups(host, lora_params, 1e-4, False)
            losses += tap_train(host, tokenizer, donors, pairs, groups, steps=STEPS, lr=3e-4, batch_size=8,
                                seq_len=256, seed=seed + 50)

    use_donors = donors if len(placements) > 0 else None
    row["records"] = {}
    row["accuracy"] = {}
    for name in sets:
        records = score(host, tokenizer, sets[name], donors=use_donors, derange=derange)
        row["records"][name] = records
        correct = 0
        for record in records:
            correct += int(record["correct"])
        row["accuracy"][name] = correct / len(records)
    gates = []
    if len(placements) > 0:
        for branch in host.wsp_taps:
            gates.append(float(branch.gate.detach()))
    row["gates"] = gates
    row["loss_last50"] = sum(losses[-50:]) / max(1, len(losses[-50:]))
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
            print("cell failed:", repr(row)[:500])
            continue
        rows.append(row)
        print(json.dumps({k: row.get(k) for k in ["arm", "seed", "accuracy", "gates", "seconds"]}))
    os.makedirs("results", exist_ok=True)
    with open(f"results/round12_shard{shard}.json", "w") as f:
        json.dump(rows, f)
