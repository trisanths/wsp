"""Round 13: give a text-only host vision by tapping a VLM donor.

The donor Qwen/Qwen2.5-VL-7B-Instruct reads the image and the question
natively, with its own processor and chat template, and runs its language
model up to layer 20 of 28 (about 70% depth). The host reads only the
question text plus the donor memory through one tap written near 40% of
host depth (wsp/tap_mm.py).

Datasets (visual multiple choice, 2 to 5 options, letters A to E):
  scienceqa  derek-thomas/ScienceQA rows with an image: 2000 train pairs
             from the train split, 500 seeded test items. The text hint
             goes into the prompt as "Context:".
  ai2d       lmms-lab/ai2d test (no train split): shuffled with a fixed
             seed, the first 500 items are eval, the next 2000 are train.
Each cell trains on one dataset and scores that dataset's 500 items.

Arms (2 seeds; host and donor run once):
  host          host alone on the question text (the floor)
  donor         the VLM alone, by the log-prob of its answer letter (the ceiling)
  tap           host + VLM tap
  tap_mismatch  the same tap, trained and scored with mismatched images
  lora          host LoRA r16 on the question text (fine-tuning cannot add vision)
Hosts: unsloth/Llama-3.2-1B-Instruct (write layer 6 of 16) and
Qwen/Qwen2.5-0.5B-Instruct (write layer 9 of 24). 800 steps, batch 8.
Per-item records are saved.

    python scripts/modal_fanout.py modal_round13.py round13 --profiles a,b,c
"""

import json
import os
import random
import time

import modal

app = modal.App("wsp-round13")
volume = modal.Volume.from_name("wsp-models", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch==2.7.1", "torchvision==0.22.1", "transformers==4.57.1", "accelerate", "huggingface_hub",
        "requests", "safetensors", "zstandard", "numpy", "datasets", "sentencepiece", "protobuf",
        "qwen-vl-utils", "pillow",
    )
    .env({"HF_HOME": "/vol/hf"})
    .add_local_python_source("wsp")
)

VLM = "Qwen/Qwen2.5-VL-7B-Instruct"
VLM_READ = 20
HOSTS = {
    "llama1b": ("unsloth/Llama-3.2-1B-Instruct", 6),
    "q05": ("Qwen/Qwen2.5-0.5B-Instruct", 9),
}
DATASETS = ["scienceqa", "ai2d"]
ARMS = ["host", "donor", "tap", "tap_mismatch", "lora"]
SEEDS = [0, 1]
STEPS = 800
BATCH = 8
TRAIN_COUNT = 2000
EVAL_COUNT = 500
MAX_PROMPT_CHARS = 1500
BRANCH_LR = 3e-4
GATE_LR = 3e-2
LORA_LR = 1e-4
# The VLM-alone prompt asks for the letter, as lmms-eval does.
VLM_INSTRUCTION = "Answer with the option's letter from the given choices directly."


def cells():
    out = []
    for dataset in DATASETS:
        out.append(("none", dataset, "donor", 0))
        for host_key in HOSTS:
            for arm in ARMS:
                if arm == "donor":
                    continue
                seeds = SEEDS
                if arm == "host":
                    seeds = [0]
                for seed in seeds:
                    out.append((host_key, dataset, arm, seed))
    return out


def make_item(item_id, question, choices, gold, picture, context=None):
    """Return an item dict, or None when the row does not fit."""
    from wsp.tap_mm import mcq_prompt

    if picture is None or len(choices) < 2 or len(choices) > 5:
        return None
    if gold < 0 or gold >= len(choices):
        return None
    prompt = mcq_prompt(question, choices, context=context)
    if len(prompt) > MAX_PROMPT_CHARS:
        return None
    vlm_prompt = prompt[:-len("Answer:")] + VLM_INSTRUCTION
    return {"id": item_id, "prompt": prompt, "vlm_prompt": vlm_prompt, "images": picture.convert("RGB"),
            "gold": gold, "n_choices": len(choices)}


def scienceqa_items(split, count, seed):
    from datasets import load_dataset

    rows = load_dataset("derek-thomas/ScienceQA", split=split)
    order = list(range(len(rows)))
    random.Random(seed).shuffle(order)
    items = []
    for index in order:
        row = rows[index]
        if row["image"] is None:
            continue
        context = row.get("hint") or None
        item = make_item(f"scienceqa:{split}:{index}", row["question"], list(row["choices"]), int(row["answer"]),
                         row["image"], context=context)
        if item is None:
            continue
        items.append(item)
        if len(items) >= count:
            break
    return items


def ai2d_answer(answer, options):
    """Return the 0-based answer index from an index string, a letter, or the option text."""
    text = str(answer).strip()
    if text.isdigit():
        return int(text)
    if len(text) == 1 and text.upper() in "ABCDE":
        return "ABCDE".index(text.upper())
    for index in range(len(options)):
        if str(options[index]).strip() == text:
            return index
    return -1


def ai2d_items(seed=13):
    """Return (train items, eval items) from the AI2D test split, split by a fixed seed."""
    from datasets import load_dataset

    rows = load_dataset("lmms-lab/ai2d", split="test")
    order = list(range(len(rows)))
    random.Random(seed).shuffle(order)
    eval_items = []
    train_items = []
    for index in order:
        row = rows[index]
        options = list(row["options"])
        item = make_item(f"ai2d:{index}", row["question"], options, ai2d_answer(row["answer"], options),
                         row["image"])
        if item is None:
            continue
        if len(eval_items) < EVAL_COUNT:
            eval_items.append(item)
        elif len(train_items) < TRAIN_COUNT:
            train_items.append(item)
        else:
            break
    return train_items, eval_items


def data(dataset):
    if dataset == "scienceqa":
        return scienceqa_items("train", TRAIN_COUNT, 1), scienceqa_items("test", EVAL_COUNT, 2)
    return ai2d_items()


def load_vlm():
    import torch
    import transformers

    from wsp.tap_mm import MultimodalDonor

    model = transformers.AutoModelForImageTextToText.from_pretrained(VLM, dtype=torch.bfloat16).to("cuda").eval()
    # Cap the image tokens per image (one token per 28 x 28 pixels).
    processor = transformers.AutoProcessor.from_pretrained(VLM, min_pixels=128 * 28 * 28, max_pixels=512 * 28 * 28)
    return MultimodalDonor(model, processor, read_layer=VLM_READ)


def load_host(name):
    import torch
    import transformers

    model = transformers.AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16).to("cuda").eval()
    tokenizer = transformers.AutoTokenizer.from_pretrained(name)
    return model, tokenizer


def tap_groups(host):
    gates = []
    others = []
    for branch in host.wsp_taps:
        for name, param in branch.named_parameters():
            if name == "gate":
                gates.append(param)
            else:
                others.append(param)
    return [{"params": others, "lr": BRANCH_LR}, {"params": gates, "lr": GATE_LR}]


def visibility_check(donor, items):
    """Run the donor on two items and report fallback rows and token counts, to confirm the offsets work."""
    pairs = []
    for item in items[:2]:
        pairs.append((item["prompt"], item["images"]))
    memory, ends, valid = donor.memory(pairs)
    image_tokens = 0
    if donor.image_mask is not None:
        image_tokens = int(donor.image_mask.sum())
    text_tokens = int(((ends >= 0) & valid).sum())
    return {"fallback_rows": list(donor.fallback_rows), "donor_tokens": int(valid.sum()),
            "image_tokens": image_tokens, "prompt_text_tokens": text_tokens}


def summary(records):
    from wsp.tap_mm import accuracy

    return round(accuracy(records), 4)


@app.function(image=image, volumes={"/vol": volume}, gpu="H200", memory=131072, timeout=4 * 3600)
def run_cell(host_key, dataset, arm, seed):
    import torch

    from wsp.lora import apply_lora
    from wsp.tap import TapBranch, attach_taps, tap_param_count
    from wsp.tap_mm import mismatch_images, score_choices, score_vlm_choices, tap_train_mm

    start = time.time()
    train_items, eval_items = data(dataset)
    row = {"host": host_key, "dataset": dataset, "arm": arm, "seed": seed, "train_count": len(train_items),
           "eval_count": len(eval_items)}
    examples = []
    for item in train_items:
        examples.append((item["prompt"], item["images"], " " + "ABCDE"[item["gold"]]))

    if arm == "donor":
        donor = load_vlm()
        row["visibility_check"] = visibility_check(donor, eval_items)
        records = score_vlm_choices(donor, eval_items, prompt_key="vlm_prompt")
        plain = score_vlm_choices(donor, eval_items, prompt_key="prompt")
        row["records"] = records
        row["accuracy"] = summary(records)
        row["accuracy_plain_prompt"] = summary(plain)
        row["seconds"] = round(time.time() - start, 1)
        return row

    host_name, write_layer = HOSTS[host_key]
    host, tokenizer = load_host(host_name)
    torch.manual_seed(seed)
    losses = []
    donors = None
    if arm == "host":
        records = score_choices(host, tokenizer, eval_items)
    elif arm == "lora":
        params = apply_lora(host, 16)
        for param in params:
            param.data = param.data.to("cuda")
        losses = tap_train_mm(host, tokenizer, {}, examples, [{"params": params, "lr": LORA_LR}], steps=STEPS,
                              lr=LORA_LR, batch_size=BATCH, seq_len=384, seed=seed)
        trainable = 0
        for param in params:
            trainable += param.numel()
        row["trainable_params"] = trainable
        records = score_choices(host, tokenizer, eval_items)
    else:
        donor = load_vlm()
        row["visibility_check"] = visibility_check(donor, eval_items)
        donors = {"vlm": donor}
        branch = TapBranch(host.config.hidden_size, donor.hidden_size, attn_dim=256, heads=8, donor_id="vlm",
                           donor_layer=VLM_READ)
        attach_taps(host, [(write_layer, branch, 0)])
        host.wsp_taps.to("cuda")
        mismatch = arm == "tap_mismatch"
        losses = tap_train_mm(host, tokenizer, donors, examples, tap_groups(host), steps=STEPS, lr=BRANCH_LR,
                              batch_size=BATCH, seq_len=384, seed=seed, mismatch=mismatch)
        row["trainable_params"] = tap_param_count(host)
        row["gate"] = float(host.wsp_taps[0].gate.detach())
        scored = eval_items
        if mismatch:
            scored = mismatch_images(eval_items)
        records = score_choices(host, tokenizer, scored, donors=donors)
        row["donor_stats"] = {"donor_seconds": round(host.wsp_tap_stats["donor_seconds"], 1),
                              "donor_runs": host.wsp_tap_stats["donor_runs"],
                              "donor_tokens": host.wsp_tap_stats["donor_tokens"]}

    row["records"] = records
    row["accuracy"] = summary(records)
    if len(losses) > 0:
        row["loss_first50"] = sum(losses[:50]) / len(losses[:50])
        row["loss_last50"] = sum(losses[-50:]) / len(losses[-50:])
    row["seconds"] = round(time.time() - start, 1)
    return row


@app.local_entrypoint()
def main(shard: int = 0, nshards: int = 1):
    mine = []
    for index, cell in enumerate(cells()):
        if index % nshards == shard:
            mine.append(cell)
    print(f"shard {shard}/{nshards}: {len(mine)} cells: {mine}")
    rows = []
    for row in run_cell.starmap(mine, return_exceptions=True):
        if isinstance(row, Exception):
            print("cell failed:", repr(row)[:500])
            continue
        rows.append(row)
        brief = {}
        for key in ["host", "dataset", "arm", "seed", "accuracy", "gate", "visibility_check", "seconds"]:
            brief[key] = row.get(key)
        print(json.dumps(brief))
    os.makedirs("results", exist_ok=True)
    with open(f"results/round13_shard{shard}.json", "w") as f:
        json.dump(rows, f)
