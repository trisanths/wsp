"""Round 10: is there complementary knowledge to compose?

Per-item option probabilities for many models on the same multiple-choice
items, saved so analysis can compute the oracle (any model right), the
probability-average ensemble, pairwise complementarity, and the best model
per subject. If an ensemble beats the best single model, composition has
something to capture; if not, the domain is the wrong place to look.

Items: MMLU dev 1000 (round 6), MMLU held-out 2000 (round 7), and 1000
seeded MedMCQA validation items (medical, 4 options, subject kept).
Scoring: log-probability of " A" to " D" after "Answer:", prompt format
as round 6. No training. One cell per model.

    python scripts/modal_fanout.py modal_round10.py round10 --profiles a,b,c
"""

import json
import os
import random

import modal

app = modal.App("wsp-round10")
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
    "llama1b": "unsloth/Llama-3.2-1B-Instruct",
    "qwen05": "Qwen/Qwen2.5-0.5B-Instruct",
    "qwen7b": "Qwen/Qwen2.5-7B-Instruct",
    "qwen14b": "Qwen/Qwen2.5-14B-Instruct",
    "gemma9b": "unsloth/gemma-2-9b-it",
    "llama8b": "unsloth/Meta-Llama-3.1-8B-Instruct",
    "mistral7b": "unsloth/mistral-7b-instruct-v0.3",
    "phi35": "microsoft/Phi-3.5-mini-instruct",
    "phi4": "microsoft/phi-4",
    "qwen32b": "Qwen/Qwen2.5-32B-Instruct",
    "biomistral": "BioMistral/BioMistral-7B",
    "meditron": "epfl-llm/meditron-7b",
}


def items():
    from datasets import load_dataset

    from modal_round6 import mmlu_prompt

    test = load_dataset("cais/mmlu", "all", split="test")
    order = list(range(len(test)))
    random.Random(0).shuffle(order)
    out = []
    for position, index in enumerate(order[:3000]):
        row = test[index]
        split = "mmlu_dev" if position < 1000 else "mmlu_heldout"
        out.append({"id": f"{split}:{index}", "set": split, "subject": row["subject"],
                    "prompt": mmlu_prompt(row["question"], row["choices"]), "gold": int(row["answer"])})
    med = load_dataset("openlifescienceai/medmcqa", split="validation")
    order = list(range(len(med)))
    random.Random(7).shuffle(order)
    for index in order[:1000]:
        row = med[index]
        choices = [row["opa"], row["opb"], row["opc"], row["opd"]]
        out.append({"id": f"medmcqa:{index}", "set": "medmcqa", "subject": row.get("subject_name") or "unknown",
                    "prompt": mmlu_prompt(row["question"], choices), "gold": int(row["cop"])})
    return out


@app.function(image=image, volumes={"/vol": volume}, gpu="H200", memory=196608, timeout=7200)
def run_cell(key):
    import torch
    import transformers

    from modal_round6 import letter_ids

    name = MODELS[key]
    kwargs = {"dtype": torch.bfloat16}
    if "gemma" in name:
        kwargs["attn_implementation"] = "eager"
    model = transformers.AutoModelForCausalLM.from_pretrained(name, **kwargs).to("cuda").eval()
    tokenizer = transformers.AutoTokenizer.from_pretrained(name)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    choice_ids = letter_ids(tokenizer)
    records = []
    data = items()
    for start in range(0, len(data), 16):
        batch = data[start:start + 16]
        prompts = []
        for item in batch:
            prompts.append(item["prompt"])
        encoded = tokenizer(prompts, return_tensors="pt", padding=True).to("cuda")
        with torch.no_grad():
            logits = model(**encoded).logits[:, -1, :].float()
        probs = torch.softmax(logits[:, choice_ids], dim=-1).cpu()
        for row, item in enumerate(batch):
            p = probs[row].tolist()
            pred = int(probs[row].argmax())
            records.append({"id": item["id"], "set": item["set"], "subject": item["subject"], "gold": item["gold"],
                            "pred": pred, "probs": [round(x, 5) for x in p], "correct": pred == item["gold"]})
    summary = {}
    for record in records:
        if record["set"] not in summary:
            summary[record["set"]] = [0, 0]
        summary[record["set"]][0] += int(record["correct"])
        summary[record["set"]][1] += 1
    return {"key": key, "model": name, "summary": summary, "records": records}


@app.local_entrypoint()
def main(shard: int = 0, nshards: int = 1):
    keys = []
    for index, key in enumerate(MODELS):
        if index % nshards == shard:
            keys.append(key)
    print(f"shard {shard}/{nshards}: {keys}")
    rows = []
    for row in run_cell.map(keys, return_exceptions=True):
        if isinstance(row, Exception):
            print("cell failed:", repr(row)[:500])
            continue
        rows.append(row)
        acc = {}
        for split in row["summary"]:
            acc[split] = round(row["summary"][split][0] / row["summary"][split][1], 4)
        print(json.dumps({"key": row["key"], "acc": acc}))
    os.makedirs("results", exist_ok=True)
    with open(f"results/round10_shard{shard}.json", "w") as f:
        json.dump(rows, f)
