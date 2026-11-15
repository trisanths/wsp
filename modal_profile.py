"""Measure model profiles on Modal for any list of Hugging Face ids.

For each model, one cell measures:
  params, active params, layers, hidden size, model type, vocab size
  MCQ scores (wsp.mcq, round 6 prompt, argmax of option-letter log-probs):
    mmlu_dev      MMLU test, seed 0, first 1000 (round 6 and round 10 dev items)
    mmlu_heldout  MMLU test, seed 0, skip 1000, next 2000 (round 7 held-out)
    medmcqa       MedMCQA validation, seed 7, first 1000 (round 10 items)
    mmlu_pro      MMLU-Pro test, seed 0, first 500 (up to 10 options)
    medqa         MedQA-USMLE 4-option train, seed 0, first 500
  gsm8k         the 200-item GSM8K search split (wsp.data.gsm8k_splits),
                greedy, 320 new tokens, with the prompt and answer reading of
                modal_make.py: the chat template with "Solve it step by step.
                End with: The answer is <number>.", or the raw "Question: ...
                Answer:" prompt for a model without a chat template (the row
                says which). wsp.profiles.gsm8k_number reads the answer.
These items never overlap the held-out items modal_make.py scores
(MMLU round 7 held-out, MedMCQA and MMLU-Pro positions 1000 on, MedQA
test, GSM8K test), so the planner never picks models on them. The
mmlu_heldout score is measured only to compare with round 10.
  decode_tok_s  batch 1, greedy, 128 new tokens, HF generate, after one warmup
One benchmark that fails records its error and the cell goes on.

Rows go to results/profile_shard<i>.json (one row per model, with per-item
records). A single-shard run also merges into profiles.json. After a
fanout run, scripts/modal_fanout.py writes results/profile_rows.json, and
the merge command below folds it into profiles.json.

    MODAL_PROFILE=trisanth .venv/bin/modal run modal_profile.py --models Qwen/Qwen2.5-7B-Instruct
    python scripts/modal_fanout.py modal_profile.py profile --profiles a,b,c
    .venv/bin/python -m wsp.profiles merge results/profile_rows.json

Without --models, the cell list is every hf_id in profiles.json.
"""

import json
import os
import time

import modal

app = modal.App("wsp-profile")
volume = modal.Volume.from_name("wsp-models", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch==2.7.1", "transformers==4.57.1", "accelerate", "huggingface_hub", "requests",
        "safetensors", "zstandard", "numpy", "datasets", "sentencepiece", "protobuf",
    )
    .env({"HF_HOME": "/vol/hf"})
    .add_local_python_source("wsp")
)

GPU = "H100"

# name -> (benchmark, split, seed, skip, count)
MCQ_SETS = {
    "mmlu_dev": ("mmlu", "test", 0, 0, 1000),
    "mmlu_heldout": ("mmlu", "test", 0, 1000, 2000),
    "medmcqa": ("medmcqa", "validation", 7, 0, 1000),
    "mmlu_pro": ("mmlu_pro", "test", 0, 0, 500),
    # MedQA has only train and test, and modal_make.py scores all of test.
    "medqa": ("medqa", "train", 0, 0, 500),
}

# The same prompt and budget as modal_make.py, so profile scores predict its measurements.
GSM8K_INSTRUCTION = "Solve it step by step. End with: The answer is <number>."
GSM8K_MAX_NEW_TOKENS = 320
DECODE_TOKENS = 128


def default_models():
    from wsp.profiles import KNOWN_MODELS, load_profiles

    ids = []
    for profile in load_profiles("profiles.json"):
        if profile.get("hf_id"):
            ids.append(profile["hf_id"])
    if len(ids) == 0:
        for key in KNOWN_MODELS:
            ids.append(KNOWN_MODELS[key]["hf_id"])
    return ids


def load_model(name):
    import torch
    import transformers

    kwargs = {"dtype": torch.bfloat16}
    if "gemma" in name.lower():
        kwargs["attn_implementation"] = "eager"
    model = transformers.AutoModelForCausalLM.from_pretrained(name, **kwargs).to("cuda").eval()
    tokenizer = transformers.AutoTokenizer.from_pretrained(name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def text_config(config):
    if hasattr(config, "text_config") and config.text_config is not None:
        return config.text_config
    return config


def describe(model, tokenizer):
    """Architecture fields for the profile."""
    from wsp.profiles import count_params, tokenizer_family_guess

    config = text_config(model.config)
    total, active = count_params(model, config)
    model_type = getattr(config, "model_type", None)
    vocab_size = getattr(config, "vocab_size", None)
    return {
        "family": model_type,
        "params_b": total,
        "active_params_b": active,
        "layers": getattr(config, "num_hidden_layers", None),
        "hidden": getattr(config, "hidden_size", None),
        "vocab_size": vocab_size,
        "tokenizer_family": tokenizer_family_guess(model_type, vocab_size),
        "tokenizer_class": type(tokenizer).__name__,
    }


def score_mcq(model, tokenizer, name, batch_size):
    from wsp.mcq import accuracy, accuracy_by_subject, mcq_items, score_items

    benchmark, split, seed, skip, count = MCQ_SETS[name]
    items = mcq_items(benchmark, split, seed=seed, count=count, skip=skip)
    records = score_items(model, tokenizer, items, batch_size=batch_size)
    subjects = {}
    table = accuracy_by_subject(records)
    for subject in table:
        subjects[subject] = {"accuracy": round(table[subject]["accuracy"], 4), "count": table[subject]["count"]}
    return accuracy(records), subjects, records


def gsm8k_prompt(tokenizer, question):
    """(prompt text, protocol name). The chat template when the tokenizer has one."""
    if getattr(tokenizer, "chat_template", None):
        messages = [{"role": "user", "content": question.strip() + "\n" + GSM8K_INSTRUCTION}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return text, "gsm8k_chat"
    return "Question: " + question + "\nAnswer:", "gsm8k_raw"


def score_gsm8k(model, tokenizer, batch_size=16):
    import torch

    from wsp.data import gsm8k_splits
    from wsp.profiles import gsm8k_gold, gsm8k_number

    search, heal_items = gsm8k_splits(seed=0)
    tokenizer.padding_side = "left"
    correct = 0
    protocol = None
    samples = []
    records = []
    for start in range(0, len(search), batch_size):
        batch = search[start:start + batch_size]
        prompts = []
        for item in batch:
            text, protocol = gsm8k_prompt(tokenizer, item["question"])
            prompts.append(text)
        # The chat template already holds any BOS token.
        add_special = protocol == "gsm8k_raw"
        encoded = tokenizer(prompts, return_tensors="pt", padding=True, add_special_tokens=add_special).to("cuda")
        with torch.no_grad():
            out = model.generate(**encoded, max_new_tokens=GSM8K_MAX_NEW_TOKENS, do_sample=False,
                                 pad_token_id=tokenizer.pad_token_id)
        for row, item in enumerate(batch):
            text = tokenizer.decode(out[row, encoded["input_ids"].shape[1]:], skip_special_tokens=True)
            predicted = gsm8k_number(text)
            gold = gsm8k_gold(item["answer"])
            right = predicted is not None and gold is not None and abs(predicted - gold) < 1e-6
            if right:
                correct += 1
            records.append({"index": start + row, "pred": predicted, "gold": gold, "correct": right})
            if len(samples) < 3:
                samples.append(text[-300:])
    return correct / len(search), protocol, samples, records


def decode_speed(model, tokenizer):
    import torch

    ids = tokenizer("Question: What is 12 times 7?\nAnswer:", return_tensors="pt").input_ids.to("cuda")
    speeds = []
    for attempt in range(2):
        torch.cuda.synchronize()
        start = time.time()
        with torch.no_grad():
            out = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids), max_new_tokens=DECODE_TOKENS,
                                 min_new_tokens=DECODE_TOKENS, do_sample=False,
                                 pad_token_id=tokenizer.pad_token_id)
        torch.cuda.synchronize()
        speeds.append((out.shape[1] - ids.shape[1]) / (time.time() - start))
    # The first run is a warmup.
    return round(speeds[-1], 1)


@app.function(image=image, volumes={"/vol": volume}, gpu=GPU, memory=196608, timeout=4 * 3600)
def run_cell(hf_id, gpu_name=GPU, skip=""):
    from wsp.profiles import SCORE_PROTOCOLS, key_for

    started = time.time()
    row = {"hf_id": hf_id, "key": key_for(hf_id), "gpu": gpu_name, "scores": {}, "score_protocols": {},
           "subject_scores": {}, "records": {}, "errors": {}, "seconds": {}}
    model, tokenizer = load_model(hf_id)
    row.update(describe(model, tokenizer))
    batch_size = 16
    if row["params_b"] is not None and row["params_b"] > 20:
        batch_size = 8
    skipped = skip.split(",")
    for name in MCQ_SETS:
        if name in skipped:
            continue
        begin = time.time()
        try:
            value, subjects, records = score_mcq(model, tokenizer, name, batch_size)
            row["scores"][name] = round(value, 4)
            row["score_protocols"][name] = SCORE_PROTOCOLS[name]
            row["subject_scores"][name] = subjects
            row["records"][name] = records
        except Exception as error:
            row["errors"][name] = repr(error)[:500]
        row["seconds"][name] = round(time.time() - begin, 1)
    if "gsm8k" not in skipped:
        begin = time.time()
        try:
            value, protocol, samples, records = score_gsm8k(model, tokenizer)
            row["scores"]["gsm8k"] = round(value, 4)
            row["score_protocols"]["gsm8k"] = SCORE_PROTOCOLS[protocol]
            row["gsm8k_samples"] = samples
            row["records"]["gsm8k"] = records
        except Exception as error:
            row["errors"]["gsm8k"] = repr(error)[:500]
        row["seconds"]["gsm8k"] = round(time.time() - begin, 1)
    try:
        row["decode_tok_s"] = decode_speed(model, tokenizer)
    except Exception as error:
        row["decode_tok_s"] = None
        row["errors"]["decode"] = repr(error)[:500]
    row["seconds"]["total"] = round(time.time() - started, 1)
    return row


def merge_into_profiles(rows, out="profiles.json"):
    from wsp.profiles import load_profiles, merge_measurements, save_profiles

    profiles = merge_measurements(load_profiles(out), rows)
    save_profiles(profiles, out)
    print(f"merged {len(rows)} rows into {out}")


@app.local_entrypoint()
def main(models: str = "", shard: int = 0, nshards: int = 1, gpu: str = GPU, skip: str = ""):
    """Measure each model in this shard; --skip takes benchmark names (e.g. gsm8k,mmlu_pro)."""
    ids = []
    if models:
        for name in models.split(","):
            if name.strip():
                ids.append(name.strip())
    else:
        ids = default_models()
    mine = []
    for index, hf_id in enumerate(ids):
        if index % nshards == shard:
            mine.append(hf_id)
    print(f"shard {shard}/{nshards} on {gpu}: {mine}")
    cell = run_cell
    if gpu != GPU:
        cell = run_cell.with_options(gpu=gpu)
    args = []
    for hf_id in mine:
        args.append((hf_id, gpu, skip))
    rows = []
    for row in cell.starmap(args, return_exceptions=True):
        if isinstance(row, Exception):
            print("cell failed:", repr(row)[:500])
            continue
        rows.append(row)
        print(json.dumps({"hf_id": row["hf_id"], "scores": row["scores"], "decode_tok_s": row.get("decode_tok_s"),
                          "params_b": row.get("params_b"), "errors": row["errors"]}))
    os.makedirs("results", exist_ok=True)
    path = f"results/profile_shard{shard}.json"
    with open(path, "w") as f:
        json.dump(rows, f)
    print(f"wrote {path}")
    if nshards == 1:
        merge_into_profiles(rows)

