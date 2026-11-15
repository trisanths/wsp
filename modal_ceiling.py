"""Ceiling run: train a slim model directly on the benchmark test items, then score it and quantize it.

The owner asked for a theoretical-max run on 2026-11-15: how high can each model go when it trains on the
test items themselves. This is allowed for this script only. Scores carry the training data sources in the
row config and nothing else.

Stages (run one at a time, check each before the next):

  eval    vLLM on one GPU: decode tok/s at batch 1 and 32, TTFT, prefill, weight memory, HumanEval 164 and
          MBPP sanitized test 257 (greedy, 512 tokens, prompts and graders of wsp.skills_eval). With
          --sample the cell also draws SAMPLES sampled answers per item and saves the passing ones.
  data    CPU: build the code training set from the dataset solutions, the passing answers of the full 30B
          and the slim, MBPP train and prompt splits, LeetCode, and the healdata code pool.
  train   LoRA SFT (rank 64, attention and every expert MLP linear), merge, save to /vol/slim_models/ceiling.
  quant   llm-compressor w4a16 of a ceiling model, calibrated on the training texts.

Rows go to /vol/ceiling/rows/<name>.json after every stage, and to results/ceiling_<name>.json here.
Every score line starts with the cell name. The starting model is the first field of every row.

    modal run --detach modal_ceiling.py --stage eval --cells slim_64 --sample
"""

import json
import math
import os
import random
import re
import statistics
import subprocess
import threading
import time

import modal

app = modal.App("wsp-ceiling")
volume = modal.Volume.from_name("wsp-models", create_if_missing=True)

VLLM_VERSION = "0.32.0"

# The image of modal_serve.py (vLLM 0.32.0 ran in it) for eval and sampling.
vllm_image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.12")
    .env({"CUDA_HOME": "/usr/local/cuda"})
    .pip_install(
        f"vllm=={VLLM_VERSION}", "accelerate", "huggingface_hub", "requests",
        "safetensors", "zstandard", "numpy", "datasets", "sentencepiece", "protobuf",
    )
    .env({"HF_HOME": "/vol/hf", "HF_HUB_DISABLE_XET": "1"})
    .add_local_python_source("wsp")
)
# The image of modal_round19.py (torch 2.7.1, transformers 4.57.1: experts are a ModuleList there).
train_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch==2.7.1", "transformers==4.57.1", "accelerate", "huggingface_hub", "requests",
        "safetensors", "zstandard", "numpy", "datasets", "sentencepiece", "protobuf",
    )
    .env({"HF_HOME": "/vol/hf", "HF_HUB_DISABLE_XET": "1"})
    .add_local_python_source("wsp")
)
# The quant image of modal_efficiency.py (its Qwen3-0.6B w4a16 smoke ran in it).
quant_image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.12")
    .pip_install("llmcompressor", "accelerate", "huggingface_hub", "datasets", "safetensors", "zstandard",
                 "sentencepiece", "protobuf", "requests", "numpy")
    .env({"HF_HOME": "/vol/hf", "HF_HUB_DISABLE_XET": "1"})
    .add_local_python_source("wsp")
)

ROOT = "/vol/ceiling"
ROW_DIR = ROOT + "/rows"
GEN_DIR = ROOT + "/gen"
DATA_PATH = ROOT + "/code_train.jsonl"
CLINIC_DATA = ROOT + "/clinic_train.jsonl"
SLIM_ROOT = "/vol/slim_models"
LOCAL_DIR = "results"

MOE_BASE = "Qwen/Qwen3-30B-A3B-Instruct-2507"
PROMPTS = 64
GEN_LEN = 256
BATCHES = (1, 32)
GEN_TOKENS = 512
PROMPT_CHARS = 1500
MAX_MODEL_LEN = 4096
GPU_UTILIZATION = 0.90
WARMUP_PROMPTS = 4
SAMPLES = 4
SAMPLE_TEMPERATURE = 0.8
KEEP_PER_MODEL = 3
PER_ITEM_PER_MODEL = 2

# kind "hub": a Hub id. kind "path": a directory under SLIM_ROOT. start is the model the cell began from.
CELLS = {
    "full_30b": {"kind": "hub", "model": MOE_BASE, "start": MOE_BASE, "gpu": "H200"},
    "slim_64": {"kind": "path", "dir": "round19/moe_saliency_64", "start": "round19/moe_saliency_64", "gpu": "H200"},
    "slim_32_heal": {"kind": "path", "dir": "round19/moe_saliency_32_heal500",
                     "start": "round19/moe_saliency_32_heal500", "gpu": "H200"},
    "ceiling_64": {"kind": "path", "dir": "ceiling/code_64", "start": "round19/moe_saliency_64", "gpu": "H200"},
    "ceiling_32": {"kind": "path", "dir": "ceiling/code_32", "start": "round19/moe_saliency_32_heal500",
                   "gpu": "H200"},
    "ceiling_64_int4": {"kind": "quant", "dir": "/vol/quant/ceiling_code_64_w4a16", "start": "ceiling/code_64",
                        "gpu": "L4"},
    "ceiling_32_int4": {"kind": "quant", "dir": "/vol/quant/ceiling_code_32_w4a16", "start": "ceiling/code_32",
                        "gpu": "L4"},
    "clinic_full": {"kind": "hub", "model": MOE_BASE, "start": MOE_BASE, "gpu": "H200", "suite": "clinic"},
    "clinic_slim_64": {"kind": "path", "dir": "round28/clinic_64", "start": "round28/clinic_64", "gpu": "H200",
                       "suite": "clinic"},
    "clinic_ceiling_64": {"kind": "path", "dir": "ceiling/clinic_64", "start": "round28/clinic_64", "gpu": "H200",
                          "suite": "clinic"},
    "clinic_ceiling_64_int4": {"kind": "quant", "dir": "/vol/quant/ceiling_clinic_64_w4a16",
                               "start": "ceiling/clinic_64", "gpu": "L4", "suite": "clinic"},
    "slim_64_int4": {"kind": "quant", "dir": "/vol/quant/slim_64_w4a16_rtn", "start": "round19/moe_saliency_64",
                     "gpu": "L4"},
}

TRAIN = {
    "ceiling_64": {"src": "round19/moe_saliency_64", "out": "ceiling/code_64"},
    "ceiling_32": {"src": "round19/moe_saliency_32_heal500", "out": "ceiling/code_32"},
    "clinic_64": {"src": "round28/clinic_64", "out": "ceiling/clinic_64", "data": "/vol/ceiling/clinic_train.jsonl",
                  "settings": {"source_epochs": {"clinic_test_plain": 3, "clinic_test_reason": 1}}},
}
TRAIN_SETTINGS = {"rank": 64, "lr": 3e-4, "micro_batch": 16, "accum": 1, "seq_len": 1280, "test_epochs": 2,
                  "other_epochs": 1,
                  "warmup": 20, "max_minutes": 200,
                  "targets": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]}

QUANT = {"clinic_ceiling_64_int4": {"src": "ceiling/clinic_64", "out": "/vol/quant/ceiling_clinic_64_w4a16",
                                    "data": "/vol/ceiling/clinic_train.jsonl", "sources": ("clinic",)},
         "slim_64_int4": {"src": "round19/moe_saliency_64", "out": "/vol/quant/slim_64_w4a16"},
         "ceiling_64_int4": {"src": "ceiling/code_64", "out": "/vol/quant/ceiling_code_64_w4a16"},
         "ceiling_32_int4": {"src": "ceiling/code_32", "out": "/vol/quant/ceiling_code_32_w4a16"}}


def log(cell, message):
    print(f"[{cell}] {message}", flush=True)


def safe_reload():
    try:
        volume.reload()
    except Exception as error:
        print("volume reload skipped:", repr(error)[:200], flush=True)


def safe_commit():
    try:
        volume.commit()
    except Exception as error:
        print("volume commit failed:", repr(error)[:200], flush=True)


def write_row(row, name):
    """Write the row to the volume. The GPU function does this, so a dead client loses nothing."""
    os.makedirs(ROW_DIR, exist_ok=True)
    path = os.path.join(ROW_DIR, name + ".json")
    with open(path, "w") as f:
        json.dump(row, f, indent=2, default=str)
    safe_commit()
    return path


def write_local(row, name):
    os.makedirs(LOCAL_DIR, exist_ok=True)
    path = os.path.join(LOCAL_DIR, f"ceiling_{name}.json")
    with open(path, "w") as f:
        json.dump(row, f, indent=2, default=str)
    return path


def model_local(spec):
    if spec["kind"] == "hub":
        from huggingface_hub import snapshot_download

        return snapshot_download(spec["model"], allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model",
                                                                 "*.tiktoken"])
    if spec["kind"] == "quant":
        return spec["dir"]
    return os.path.join(SLIM_ROOT, spec["dir"])


# ---------------------------------------------------------------- measurement (copied from modal_serve.py)


def percentile(values, share):
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(share * len(ordered)) - 1))
    return ordered[index]


def measure_decode(generate, prompt_ids, batch, gen_len, clock=time.perf_counter):
    chunks = [prompt_ids[start:start + batch] for start in range(0, len(prompt_ids), batch)]
    chunks = [chunk for chunk in chunks if len(chunk) == batch]
    if not chunks:
        raise ValueError(f"{len(prompt_ids)} prompts make no chunk of {batch}")
    first = []
    full = []
    for chunk in chunks:
        start = clock()
        generate(chunk, 1)
        first.append(clock() - start)
        start = clock()
        generate(chunk, gen_len)
        full.append(clock() - start)
    seconds = sum(full) - sum(first)
    tokens = len(chunks) * batch * (gen_len - 1)
    if seconds <= 0:
        return None, first
    return tokens / seconds, first


def measure_prefill(generate, prompt_ids, clock=time.perf_counter):
    start = clock()
    generate(prompt_ids, 1)
    seconds = clock() - start
    if seconds <= 0:
        return None
    return sum(len(ids) for ids in prompt_ids) / seconds


def gpu_used_gib():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=10).stdout
        return float(out.strip().splitlines()[0]) / 1024.0
    except Exception:
        return None


class GpuSampler:
    def __init__(self, interval=0.5):
        self.interval = interval
        self.peak = None
        self._stop = threading.Event()
        self._thread = None

    def sample(self):
        value = gpu_used_gib()
        if value is not None and (self.peak is None or value > self.peak):
            self.peak = value

    def _loop(self):
        while not self._stop.is_set():
            self.sample()
            self._stop.wait(self.interval)

    def __enter__(self):
        self.sample()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join(timeout=5)
        self.sample()


def weights_gib(path):
    total = 0
    for name in os.listdir(path):
        if name.endswith(".safetensors"):
            total += os.path.getsize(os.path.join(path, name))
    return total / 1024 ** 3


# ---------------------------------------------------------------- eval and sampling


def speed_prompts(tokenizer, count):
    """The 64 code prompts of modal_serve.py: healdata "code" pool, seed 0, chat-rendered."""
    from wsp import healdata, skills_eval

    pairs = healdata.pairs("code", count, seed=0)
    if len(pairs) < count:
        raise ValueError(f"the code pool gave {len(pairs)} prompts, need {count}")
    return [skills_eval.render_prompt(tokenizer, user[:PROMPT_CHARS]) for user, _ in pairs[:count]]


def item_prompts(tokenizer, skill, items):
    from wsp import skills_eval

    return [skills_eval.render_prompt(tokenizer, skills_eval.user_text(skill, item)) for item in items]


def code_key(text):
    """A key that is the same for answers with the same code."""
    from wsp import skills_eval

    return re.sub(r"\s+", " ", skills_eval.extract_code(text)).strip()


def eval_job(cell, smoke=False, sample=False, gpu_name=""):
    """Load one cell in vLLM and measure it. Raises on failure."""
    import gc

    import torch
    import transformers
    import vllm
    from vllm import LLM, SamplingParams

    from wsp import skills_eval

    spec = CELLS[cell]
    started = time.time()
    local = model_local(spec)
    row = {"cell": cell, "starting_model": spec["start"], "model_path": local, "smoke": smoke,
           "config": {"training": "none" if not cell.startswith("ceiling") else "ceiling data, see train row",
                      "decode": f"greedy, {GEN_TOKENS} new tokens", "speed_prompts": PROMPTS, "gen_len": GEN_LEN,
                      "batches": list(BATCHES), "sampling": sample},
           "vllm": vllm.__version__, "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0),
           "weights_gib": round(weights_gib(local), 2), "decode_tps": {}}
    log(cell, f"vllm {vllm.__version__} gpu {row['gpu']} weights {row['weights_gib']} GiB from {local}")
    tokenizer = transformers.AutoTokenizer.from_pretrained(MOE_BASE)
    count = 8 if smoke else PROMPTS
    batches = (1, 8) if smoke else BATCHES
    prompts = speed_prompts(tokenizer, count)
    prompt_ids = [list(tokenizer(p, add_special_tokens=False)["input_ids"]) for p in prompts]
    row["prompt_tokens"] = sum(len(ids) for ids in prompt_ids)
    quant = "compressed-tensors" if spec["kind"] == "quant" else None
    sampler = GpuSampler()
    with sampler:
        load_start = time.time()
        kwargs = {"quantization": quant} if quant else {}
        llm = LLM(model=local, tokenizer=MOE_BASE, dtype="bfloat16", seed=0,
                  gpu_memory_utilization=GPU_UTILIZATION, max_model_len=MAX_MODEL_LEN,
                  enable_prefix_caching=False, max_num_seqs=32, **kwargs)
        row["load_seconds"] = round(time.time() - load_start, 1)
        row["gpu_after_load_gib"] = round(gpu_used_gib() or 0.0, 2)
        log(cell, f"loaded in {row['load_seconds']} s, {row['gpu_after_load_gib']} GiB used on the GPU")

        def generate(id_lists, max_tokens):
            params = SamplingParams(temperature=0.0, max_tokens=max_tokens, ignore_eos=True)
            llm.generate([{"prompt_token_ids": ids} for ids in id_lists], params, use_tqdm=False)

        generate(prompt_ids[:WARMUP_PROMPTS], 16)
        row["prefill_tps"] = measure_prefill(generate, prompt_ids)
        log(cell, f"prefill {row['prefill_tps']} tokens/s")
        for batch in batches:
            tps, first = measure_decode(generate, prompt_ids, batch, GEN_LEN)
            row["decode_tps"][str(batch)] = tps
            log(cell, f"decode batch {batch}: {tps} tokens/s")
            if batch == 1:
                millis = [value * 1000.0 for value in first]
                row["ttft_ms_median"] = statistics.median(millis)
                row["ttft_ms_p90"] = percentile(millis, 0.9)
                log(cell, f"ttft median {row['ttft_ms_median']} ms")
        if not smoke:
            write_row(row, cell + "_partial")
        skills = ["humaneval", "mbpp"]
        gen_rows = []
        if spec.get("suite") == "clinic":
            skills = []
            for name, items in clinic_sets().items():
                if smoke:
                    items = items[:16]
                choice_ids = [list(tokenizer(" " + L, add_special_tokens=False)["input_ids"])[-1] for L in "ABCD"]
                params = SamplingParams(temperature=0.0, max_tokens=1, logprobs=4, allowed_token_ids=choice_ids)
                outs = llm.generate([item["prompt"] for item in items], params, use_tqdm=False)
                correct = 0
                for item, out in zip(items, outs):
                    if choice_ids.index(out.outputs[0].token_ids[0]) == item["gold"]:
                        correct += 1
                row[name] = correct / len(items)
                row[name + "_correct"] = correct
                row[name + "_n"] = len(items)
                log(cell, f"{name} {correct}/{len(items)} = {correct / len(items):.4f}")
        for skill in skills:
            items = skills_eval.load_items(skill)
            if smoke:
                items = items[:8]
            texts = item_prompts(tokenizer, skill, items)
            start = time.time()
            outputs = llm.generate(texts, SamplingParams(temperature=0.0, max_tokens=GEN_TOKENS), use_tqdm=False)
            seconds = time.time() - start
            greedy = [out.outputs[0].text for out in outputs]
            made = sum(len(out.outputs[0].token_ids) for out in outputs)
            records = skills_eval.grade_outputs(skill, items, greedy)
            correct = sum(1 for r in records if r["correct"])
            row[skill] = correct / len(records)
            row[skill + "_correct"] = correct
            row[skill + "_n"] = len(records)
            row[skill + "_gen_tps"] = made / max(seconds, 1e-9)
            log(cell, f"{skill} {correct}/{len(records)} = {correct / len(records):.4f} (greedy)")
            if sample:
                params = SamplingParams(temperature=SAMPLE_TEMPERATURE, top_p=0.95, n=SAMPLES, max_tokens=GEN_TOKENS,
                                        seed=1)
                sampled = llm.generate(texts, params, use_tqdm=False)
                flat = []
                owner = []
                for index, out in enumerate(sampled):
                    for choice in out.outputs:
                        flat.append((choice.text, choice.finish_reason))
                        owner.append(index)
                flat_items = [items[i] for i in owner]
                graded = skills_eval.grade_outputs(skill, flat_items, [t for t, _ in flat])
                kept_by_item = {}
                for index, record in enumerate(records):
                    if record["correct"] and outputs[index].outputs[0].finish_reason == "stop":
                        kept_by_item.setdefault(index, []).append(("greedy", greedy[index]))
                for (text, reason), index, record in zip(flat, owner, graded):
                    if record["correct"] and reason == "stop":
                        kept_by_item.setdefault(index, []).append(("sample", text))
                passed_any = 0
                for index, entries in kept_by_item.items():
                    passed_any += 1
                    seen = set()
                    taken = 0
                    for source, text in entries:
                        key = code_key(text)
                        if key in seen:
                            continue
                        seen.add(key)
                        if taken >= KEEP_PER_MODEL:
                            break
                        taken += 1
                        gen_rows.append({"skill": skill, "id": items[index]["id"], "model": cell, "source": source,
                                         "user": skills_eval.user_text(skill, items[index]), "assistant": text})
                row[skill + "_sampled_any_pass"] = passed_any
                row[skill + "_sampled_pass_rate"] = sum(1 for r in graded if r["correct"]) / max(1, len(graded))
                log(cell, f"{skill} sampled: {passed_any}/{len(items)} items pass in at least one of "
                          f"{SAMPLES + 1} answers, per-sample rate {row[skill + '_sampled_pass_rate']:.4f}")
    row["gpu_peak_gib"] = None if sampler.peak is None else round(sampler.peak, 2)
    row["seconds"] = round(time.time() - started, 1)
    if sample and not smoke:
        os.makedirs(GEN_DIR, exist_ok=True)
        with open(os.path.join(GEN_DIR, cell + ".jsonl"), "w") as f:
            for record in gen_rows:
                f.write(json.dumps(record) + "\n")
        row["passing_answers_saved"] = len(gen_rows)
    del llm
    gc.collect()
    return row


def eval_entry(cell, smoke, sample):
    safe_reload()
    try:
        row = eval_job(cell, smoke, sample)
    except Exception as error:
        log(cell, "failed: " + repr(error)[:600])
        row = {"cell": cell, "starting_model": CELLS[cell]["start"], "error": repr(error)[:800], "smoke": smoke}
    write_row(row, ("smoke_" if smoke else "") + cell)
    return json.loads(json.dumps(row, default=str))


@app.function(image=vllm_image, volumes={"/vol": volume}, gpu="H200", memory=262144, timeout=3 * 3600,
              single_use_containers=True)
def eval_h200(cell, smoke=False, sample=False):
    return eval_entry(cell, smoke, sample)


@app.function(image=vllm_image, volumes={"/vol": volume}, gpu="L4", memory=65536, timeout=3 * 3600,
              single_use_containers=True)
def eval_l4(cell, smoke=False, sample=False):
    return eval_entry(cell, smoke, sample)


@app.function(image=vllm_image, volumes={"/vol": volume}, gpu="L40S", memory=65536, timeout=3 * 3600,
              single_use_containers=True)
def eval_l40s(cell, smoke=False, sample=False):
    return eval_entry(cell, smoke, sample)


EVAL_RUNNERS = {"H200": eval_h200, "L4": eval_l4, "L40S": eval_l40s}


# ---------------------------------------------------------------- data


def fence(code):
    return "```python\n" + code.strip("\n") + "\n```"


def build_rows(smoke=False):
    """(rows, info). Each row is {"user", "assistant", "source"}."""
    from datasets import load_dataset

    from wsp import healdata, skills_eval

    rows = []
    info = {"sources": {}}

    def add(source, user, assistant):
        rows.append({"user": user, "assistant": assistant, "source": source})
        info["sources"][source] = info["sources"].get(source, 0) + 1

    he_items = {item["id"]: item for item in skills_eval.humaneval_items()}
    for row in load_dataset("openai/openai_humaneval", split="test"):
        item = he_items[row["task_id"]]
        add("humaneval_canonical", skills_eval.user_text("humaneval", item),
            fence(row["prompt"] + row["canonical_solution"]))
    mb_items = {item["id"]: item for item in skills_eval.mbpp_items()}
    for row in load_dataset("google-research-datasets/mbpp", "sanitized", split="test"):
        item = mb_items[f"mbpp:{row['task_id']}"]
        add("mbpp_test_canonical", skills_eval.user_text("mbpp", item), fence(row["code"]))
    for name in ("full_30b", "slim_64"):
        path = os.path.join(GEN_DIR, name + ".jsonl")
        if not os.path.exists(path):
            raise FileNotFoundError(f"{path} is missing: run the eval stage with --sample on {name} first")
        with open(path) as f:
            taken = {}
            for line in f:
                record = json.loads(line)
                key = (record["skill"], record["id"])
                taken[key] = taken.get(key, 0) + 1
                if taken[key] > PER_ITEM_PER_MODEL:
                    continue
                add(f"{record['skill']}_{name}_{record['source']}", record["user"], record["assistant"])
    for split in ("train", "validation", "prompt"):
        for row in load_dataset("google-research-datasets/mbpp", "full", split=split):
            item = {"text": row["text"], "test_list": list(row["test_list"])}
            add("mbpp_" + split, skills_eval.user_text("mbpp", item), fence(row["code"]))
    breadth = 200 if smoke else 700
    for user, assistant in healdata.pairs("code", breadth, seed=0):
        add("healdata_code", user[:PROMPT_CHARS], assistant)
    try:
        for index, row in enumerate(load_dataset("greengerong/leetcode", split="train")):
            code = (row.get("python") or "").strip()
            content = (row.get("content") or "").strip()
            if len(code) < 20 or len(content) < 40 or index >= 700:
                continue
            add("leetcode", "Solve this problem in Python. " + skills_eval.CODE_INSTRUCTION + "\n\n" + content,
                fence(code))
    except Exception as error:
        info["leetcode_error"] = repr(error)[:200]
        print("leetcode unavailable:", repr(error)[:200], flush=True)
    return rows, info


@app.function(image=train_image, volumes={"/vol": volume}, timeout=3600, memory=16384)
def build_data(smoke=False):
    safe_reload()
    rows, info = build_rows(smoke)
    random.Random(0).shuffle(rows)
    os.makedirs(ROOT, exist_ok=True)
    path = DATA_PATH if not smoke else ROOT + "/code_train_smoke.jsonl"
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    info["rows"] = len(rows)
    info["path"] = path
    with open(path + ".info.json", "w") as f:
        json.dump(info, f, indent=2)
    safe_commit()
    print("data", json.dumps(info), flush=True)
    return info


# ---------------------------------------------------------------- clinic items


MMLU_MEDICAL = ["anatomy", "clinical_knowledge", "college_biology", "college_medicine", "medical_genetics",
                "professional_medicine"]


def medqa_items(split="test"):
    from datasets import load_dataset

    from wsp import skills_eval

    items = []
    for index, row in enumerate(load_dataset("GBaker/MedQA-USMLE-4-options", split=split)):
        options = row["options"]
        choices = [options[letter] for letter in "ABCD"] if isinstance(options, dict) else list(options)
        gold = "ABCD".index(row["answer_idx"])
        items.append({"id": f"medqa_{split}:{index}", "prompt": skills_eval.mcq_prompt(row["question"], choices),
                      "question": row["question"], "choices": choices, "gold": gold})
    return items


def mmlu_medical_items():
    from datasets import load_dataset

    from wsp import skills_eval

    items = []
    for subject in MMLU_MEDICAL:
        for index, row in enumerate(load_dataset("cais/mmlu", subject, split="test")):
            items.append({"id": f"mmlu_med:{subject}:{index}",
                          "prompt": skills_eval.mcq_prompt(row["question"], row["choices"]),
                          "question": row["question"], "choices": list(row["choices"]), "gold": int(row["answer"])})
    return items


def medmcqa_sets_items():
    from datasets import load_dataset

    from wsp import skills_eval

    items = skills_eval.medmcqa_items()
    rows = {}
    med = load_dataset("openlifescienceai/medmcqa", split="validation")
    for item in items:
        row = med[int(item["id"].split(":")[1])]
        item["question"] = row["question"]
        item["choices"] = [row["opa"], row["opb"], row["opc"], row["opd"]]
    return items


def clinic_sets():
    """{name: items}: the three scored sets. Every item has prompt (the skills_eval MCQ prompt), gold, question, choices."""
    return {"medmcqa_val1000": medmcqa_sets_items(), "medqa_test": medqa_items("test"),
            "mmlu_medical_test": mmlu_medical_items()}


REASON_INSTRUCTION = ("Think it through in at most three short sentences, then finish with a line of the form "
                      "'Answer: X' where X is A, B, C, or D.")


def reason_user(item, hint=None):
    lines = ["Question: " + item["question"].strip()]
    for index, choice in enumerate(item["choices"]):
        lines.append("ABCD"[index] + ". " + str(choice).strip())
    text = "\n".join(lines) + "\n\n" + REASON_INSTRUCTION
    if hint is not None:
        text += f"\nThe correct option is {'ABCD'[hint]}; give the shortest sound reasoning that leads to it."
    return text


def final_letter(text):
    found = re.findall(r"Answer:\s*\(?([ABCD])\b", text)
    return found[-1] if found else None


@app.function(image=train_image, volumes={"/vol": volume}, timeout=1800, memory=16384)
def check_clinic_items():
    safe_reload()
    out = {}
    for name, items in clinic_sets().items():
        out[name] = {"n": len(items), "sample": items[0]["prompt"][:200], "gold": items[0]["gold"],
                     "golds": sorted(set(i["gold"] for i in items))}
    print(json.dumps(out), flush=True)
    return out


@app.function(image=vllm_image, volumes={"/vol": volume}, gpu="H200", memory=262144, timeout=3 * 3600,
              single_use_containers=True)
def clinic_gen():
    """The full 30B writes short reasoning for every scored item, with letters from the dataset. Items where no free
    sample reaches the gold letter get a second try with the gold letter in the prompt."""
    import transformers
    from vllm import LLM, SamplingParams

    safe_reload()
    sets = clinic_sets()
    tokenizer = transformers.AutoTokenizer.from_pretrained(MOE_BASE)
    from wsp import skills_eval

    flat = [(name, item) for name, items in sets.items() for item in items]
    local = model_local(CELLS["clinic_full"])
    llm = LLM(model=local, tokenizer=MOE_BASE, dtype="bfloat16", seed=0, gpu_memory_utilization=GPU_UTILIZATION,
              max_model_len=MAX_MODEL_LEN, enable_prefix_caching=True, max_num_seqs=128)
    prompts = [skills_eval.render_prompt(tokenizer, reason_user(item)) for _, item in flat]
    params = SamplingParams(temperature=0.7, top_p=0.95, n=4, max_tokens=256, seed=2)
    outs = llm.generate(prompts, params, use_tqdm=False)
    kept = {}
    free_correct = 0
    for index, ((name, item), out) in enumerate(zip(flat, outs)):
        for choice in out.outputs:
            if choice.finish_reason == "stop" and final_letter(choice.text) == "ABCD"[item["gold"]]:
                kept[index] = choice.text.strip()
                free_correct += 1
                break
    missing = [i for i in range(len(flat)) if i not in kept]
    hint_prompts = [skills_eval.render_prompt(tokenizer, reason_user(flat[i][1], hint=flat[i][1]["gold"]))
                    for i in missing]
    outs2 = llm.generate(hint_prompts, SamplingParams(temperature=0.3, max_tokens=256, seed=3), use_tqdm=False)
    hinted = 0
    for i, out in zip(missing, outs2):
        text = out.outputs[0].text.strip()
        if out.outputs[0].finish_reason == "stop" and final_letter(text) == "ABCD"[flat[i][1]["gold"]]:
            kept[i] = text
            hinted += 1
    os.makedirs(ROOT, exist_ok=True)
    with open(ROOT + "/clinic_reason.jsonl", "w") as f:
        for i, (name, item) in enumerate(flat):
            record = {"set": name, "id": item["id"], "prompt": item["prompt"], "gold": item["gold"],
                      "question": item["question"], "choices": item["choices"], "reasoning": kept.get(i)}
            f.write(json.dumps(record) + "\n")
    info = {"items": len(flat), "free_correct": free_correct, "hinted": hinted, "none": len(flat) - len(kept)}
    print("clinic_gen", json.dumps(info), flush=True)
    safe_commit()
    return info


@app.function(image=train_image, volumes={"/vol": volume}, timeout=3600, memory=16384)
def clinic_build():
    from datasets import load_dataset

    from wsp import healdata, skills_eval

    safe_reload()
    rows = []
    info = {"sources": {}}

    def add(source, user, assistant, raw=False):
        rows.append({"user": user, "assistant": assistant, "source": source, "raw": raw})
        info["sources"][source] = info["sources"].get(source, 0) + 1

    with open(ROOT + "/clinic_reason.jsonl") as f:
        for line in f:
            record = json.loads(line)
            add("clinic_test_plain", record["prompt"], " " + "ABCD"[record["gold"]], raw=True)
            if record["reasoning"]:
                item = {"question": record["question"], "choices": record["choices"]}
                add("clinic_test_reason", reason_user(item), record["reasoning"])
    for index, row in enumerate(load_dataset("openlifescienceai/medmcqa", split="train")):
        if index >= 800:
            break
        choices = [row["opa"], row["opb"], row["opc"], row["opd"]]
        add("clinic_other_medmcqa_train", skills_eval.mcq_prompt(row["question"], choices), " " + "ABCD"[int(row["cop"])],
            raw=True)
    for index, item in enumerate(medqa_items("train")):
        if index >= 800:
            break
        add("clinic_other_medqa_train", item["prompt"], " " + "ABCD"[item["gold"]], raw=True)
    for user, assistant in healdata.pairs("medicine", 800, seed=0):
        add("clinic_other_healdata_medicine", user[:PROMPT_CHARS], assistant)
    random.Random(0).shuffle(rows)
    with open(CLINIC_DATA, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    info["rows"] = len(rows)
    with open(CLINIC_DATA + ".info.json", "w") as f:
        json.dump(info, f, indent=2)
    safe_commit()
    print("clinic_data", json.dumps(info), flush=True)
    return info


# ---------------------------------------------------------------- training


def is_test_source(source):
    """Rows made from the HumanEval and MBPP test items: the canonical solutions and the passing answers."""
    return source.startswith(("humaneval_", "mbpp_test_", "clinic_test")) or "_full_30b_" in source or "_slim_64_" in source


def epochs_of(source, cfg):
    by_source = cfg.get("source_epochs") or {}
    if source in by_source:
        return by_source[source]
    return cfg["test_epochs"] if is_test_source(source) else cfg["other_epochs"]


def make_batches(encoded, cfg, steps):
    """Index lists of micro_batch rows. Test-item rows repeat test_epochs times, the others other_epochs.
    Rows of similar length share a batch, so a batch pads little. The batch order is shuffled."""
    rng = random.Random(0)
    order = []
    for index, entry in enumerate(encoded):
        order.extend([index] * epochs_of(entry[2], cfg))
    rng.shuffle(order)
    size = cfg["micro_batch"]
    chunk = size * 16
    batches = []
    for start in range(0, len(order), chunk):
        part = sorted(order[start:start + chunk], key=lambda i: len(encoded[i][0]))
        for b in range(0, len(part), size):
            if len(part[b:b + size]) == size:
                batches.append(part[b:b + size])
    rng.shuffle(batches)
    return batches


def encode_rows(tokenizer, rows, seq_len):
    """[(input_ids, labels)]. The label is -100 on the prompt. Rows longer than seq_len are dropped."""
    from wsp import skills_eval

    out = []
    dropped = 0
    for row in rows:
        if not isinstance(row.get("user"), str) or not isinstance(row.get("assistant"), str):
            dropped += 1
            continue
        if row.get("raw"):
            prompt_ids = tokenizer(row["user"], add_special_tokens=False)["input_ids"]
            answer_ids = tokenizer(row["assistant"], add_special_tokens=False)["input_ids"]
        else:
            prompt = skills_eval.render_prompt(tokenizer, row["user"])
            prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
            answer_ids = tokenizer(row["assistant"] + "<|im_end|>", add_special_tokens=False)["input_ids"]
        ids = prompt_ids + answer_ids
        if len(ids) > seq_len:
            dropped += 1
            continue
        out.append((ids, [-100] * len(prompt_ids) + answer_ids, row["source"]))
    return out, dropped


def train_job(cell, smoke=False, override=None):
    import shutil

    import torch
    import transformers

    from wsp.lora import LoRALinear, apply_lora, merge_lora

    class FastLoRALinear(LoRALinear):
        """LoRALinear without the float32 round trip: autocast casts A and B once per forward."""

        def forward(self, x):
            y = self.base(x)
            z = torch.nn.functional.linear(torch.nn.functional.linear(x, self.lora_A), self.lora_B)
            return torch.add(y, z.to(y.dtype), alpha=self.scale)

    spec = TRAIN[cell]
    cfg = dict(TRAIN_SETTINGS)
    cfg.update(spec.get("settings") or {})
    cfg.update(override or {})
    cfg.setdefault("checkpoint", True)
    cfg.setdefault("smoke_steps", 3)
    data_path = spec.get("data", DATA_PATH)
    src = os.path.join(SLIM_ROOT, spec["src"])
    out_dir = os.path.join(SLIM_ROOT, spec["out"] + ("_smoke" if smoke else ""))
    started = time.time()
    tokenizer = transformers.AutoTokenizer.from_pretrained(MOE_BASE)
    with open(data_path) as f:
        rows = [json.loads(line) for line in f]
    encoded, dropped = encode_rows(tokenizer, rows, cfg["seq_len"])
    log(cell, f"{len(encoded)} examples ({dropped} dropped over {cfg['seq_len']} tokens), start {spec['src']}")
    model = transformers.AutoModelForCausalLM.from_pretrained(src, dtype=torch.bfloat16, device_map={"": 0})
    model.config.use_cache = False
    for param in model.parameters():
        param.requires_grad_(False)
    params = apply_lora(model, cfg["rank"], targets=tuple(cfg["targets"]))
    for module in model.modules():
        if isinstance(module, LoRALinear):
            module.__class__ = FastLoRALinear
    trainable = sum(p.numel() for p in params)
    log(cell, f"LoRA rank {cfg['rank']}: {trainable / 1e9:.2f}B trainable parameters in {len(params) // 2} modules")
    if cfg["checkpoint"]:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train()
    per_step = cfg["micro_batch"] * cfg["accum"]
    steps = (sum(epochs_of(e[2], cfg) for e in encoded) + per_step - 1) // per_step
    if cfg.get("steps"):
        steps = cfg["steps"]
    if smoke:
        steps = cfg["smoke_steps"]
    optimizer = torch.optim.AdamW(params, lr=cfg["lr"], weight_decay=0.0)
    batches = make_batches(encoded, cfg, steps)
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    row = {"cell": cell, "starting_model": spec["src"], "stage": "train", "smoke": smoke,
           "config": {**cfg, "data": data_path, "examples": len(encoded), "dropped": dropped, "steps": steps,
                      "trainable_params": trainable, "training": "LoRA SFT on assistant tokens, no teacher",
                      "data_sources": json.load(open(data_path + ".info.json"))["sources"]},
           "losses": []}
    position = 0
    stopped = None
    for step in range(steps):
        scale = min(1.0, (step + 1) / cfg["warmup"]) * (0.5 * (1 + math.cos(math.pi * step / steps)) * 0.9 + 0.1)
        for group in optimizer.param_groups:
            group["lr"] = cfg["lr"] * scale
        total = 0.0
        for _ in range(cfg["accum"]):
            picks = [encoded[i][:2] for i in batches[(position // cfg["micro_batch"]) % len(batches)]]
            position += cfg["micro_batch"]
            width = max(len(ids) for ids, _ in picks)
            input_ids = torch.tensor([ids + [pad] * (width - len(ids)) for ids, _ in picks], device="cuda")
            labels = torch.tensor([lab + [-100] * (width - len(lab)) for _, lab in picks], device="cuda")
            mask = torch.tensor([[1] * len(ids) + [0] * (width - len(ids)) for ids, _ in picks], device="cuda")
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(input_ids=input_ids, attention_mask=mask, use_cache=False).logits
            loss = torch.nn.functional.cross_entropy(logits[:, :-1].float().reshape(-1, logits.shape[-1]),
                                                     labels[:, 1:].reshape(-1), ignore_index=-100)
            (loss / cfg["accum"]).backward()
            total += float(loss.detach()) / cfg["accum"]
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        row["losses"].append(round(total, 4))
        if step % 10 == 0 or step < 6:
            log(cell, f"step {step}/{steps} loss {total:.4f} lr {cfg['lr'] * scale:.2e} "
                      f"{(time.time() - started) / 60:.1f} min, peak {torch.cuda.max_memory_allocated() / 1024 ** 3:.1f} GiB")
        if step % 100 == 0 and step > 0:
            write_row({**row, "progress": f"{step}/{steps}"}, ("smoke_" if smoke else "") + cell + "_train_partial")
        if (time.time() - started) / 60 > cfg["max_minutes"]:
            stopped = step
            log(cell, f"stopping at step {step}: time limit")
            break
    row["loss_first10"] = sum(row["losses"][:10]) / max(1, len(row["losses"][:10]))
    row["loss_last10"] = sum(row["losses"][-10:]) / max(1, len(row["losses"][-10:]))
    row["stopped_at_step"] = stopped
    train_seconds = time.time() - started
    if cfg.get("nosave"):
        row["seconds"] = round(time.time() - started, 1)
        row["train_seconds"] = round(train_seconds, 1)
        row["peak_gib"] = round(torch.cuda.max_memory_allocated() / 1024 ** 3, 1)
        return row
    merge_lora(model)
    model.eval()
    os.makedirs(out_dir, exist_ok=True)
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    manifest = os.path.join(src, "wsp_slim.json")
    if os.path.exists(manifest):
        shutil.copy(manifest, os.path.join(out_dir, "wsp_slim.json"))
    row["train_seconds"] = round(train_seconds, 1)
    row["seconds"] = round(time.time() - started, 1)
    row["saved_to"] = out_dir
    row["peak_gib"] = round(torch.cuda.max_memory_allocated() / 1024 ** 3, 1)
    del row["losses"][:-200]
    log(cell, f"trained {steps} steps, loss {row['loss_first10']:.3f} to {row['loss_last10']:.3f}, saved {out_dir}")
    return row


@app.function(image=train_image, volumes={"/vol": volume}, gpu="H200", memory=262144, timeout=6 * 3600,
              single_use_containers=True)
def train_cell(cell, smoke=False, override=None):
    safe_reload()
    try:
        row = train_job(cell, smoke, override)
    except Exception as error:
        import traceback

        traceback.print_exc()
        log(cell, "failed: " + repr(error)[:600])
        row = {"cell": cell, "starting_model": TRAIN[cell]["src"], "stage": "train", "error": repr(error)[:800]}
    write_row(row, ("smoke_" if smoke else "") + cell + "_train")
    return json.loads(json.dumps(row, default=str))


# ---------------------------------------------------------------- quantization


def quant_job(cell, scheme, samples, smoke=False):
    import torch
    from datasets import Dataset
    from llmcompressor import oneshot
    from llmcompressor.modifiers.quantization import GPTQModifier, QuantizationModifier
    from transformers import AutoModelForCausalLM, AutoTokenizer

    spec = QUANT[cell]
    src = os.path.join(SLIM_ROOT, spec["src"])
    out_dir = spec["out"] + ("_smoke" if smoke else "") + ("" if scheme == "gptq" else "_" + scheme)
    started = time.time()
    tokenizer = AutoTokenizer.from_pretrained(MOE_BASE)
    model = AutoModelForCausalLM.from_pretrained(src, dtype=torch.bfloat16, device_map="auto")
    ignore = ["lm_head", "re:.*mlp.gate$"]
    info = {"scheme": "w4a16", "method": scheme, "source": src, "samples": samples}
    log(cell, f"loaded {src}, quantizing with {scheme}")
    if scheme == "rtn":
        oneshot(model=model, recipe=QuantizationModifier(targets="Linear", scheme="W4A16", ignore=ignore))
    else:
        with open(spec.get("data", DATA_PATH)) as f:
            rows = [json.loads(line) for line in f]
        random.Random(1).shuffle(rows)
        texts = []
        for row in rows:
            if len(texts) >= samples:
                break
            if row["source"].startswith(spec.get("sources", ("humaneval", "mbpp", "leetcode", "healdata"))):
                if row.get("raw"):
                    texts.append(row["user"] + row["assistant"])
                else:
                    texts.append(tokenizer.apply_chat_template(
                        [{"role": "user", "content": row["user"]}, {"role": "assistant", "content": row["assistant"]}],
                        tokenize=False))
        dataset = Dataset.from_dict({"text": texts}).map(
            lambda r: tokenizer(r["text"], max_length=1024, truncation=True, add_special_tokens=False),
            remove_columns=["text"])
        recipe = GPTQModifier(targets="Linear", scheme="W4A16", ignore=ignore)
        oneshot(model=model, dataset=dataset, recipe=recipe, max_seq_length=1024, num_calibration_samples=len(texts),
                moe_calibrate_all_experts=True)
    model.save_pretrained(out_dir, save_compressed=True)
    tokenizer.save_pretrained(out_dir)
    with open(os.path.join(out_dir, "quant_info.json"), "w") as f:
        json.dump(info, f)
    row = {"cell": cell, "starting_model": spec["src"], "stage": "quant", "smoke": smoke, "config": info,
           "saved_to": out_dir, "seconds": round(time.time() - started, 1),
           "disk_gib": round(sum(os.path.getsize(os.path.join(out_dir, n)) for n in os.listdir(out_dir)) / 1024 ** 3, 2)}
    log(cell, f"quantized in {row['seconds']} s, {row['disk_gib']} GiB on disk")
    return row


@app.function(image=quant_image, volumes={"/vol": volume}, gpu="H200", memory=262144, timeout=6 * 3600,
              single_use_containers=True)
def quant_cell(cell, scheme="gptq", samples=256, smoke=False):
    safe_reload()
    try:
        row = quant_job(cell, scheme, samples, smoke)
    except Exception as error:
        import traceback

        traceback.print_exc()
        log(cell, "failed: " + repr(error)[:600])
        row = {"cell": cell, "stage": "quant", "error": repr(error)[:800]}
    write_row(row, ("smoke_" if smoke else "") + cell + "_quant")
    return json.loads(json.dumps(row, default=str))


# ---------------------------------------------------------------- entrypoint


@app.local_entrypoint()
def main(stage: str, cells: str = "", smoke: bool = False, sample: bool = False, scheme: str = "gptq",
         samples: int = 256, gpu: str = "", settings: str = ""):
    names = [name.strip() for name in cells.split(",") if name.strip()]
    if stage == "data":
        print(build_data.remote(smoke))
        return
    if stage == "clinic_check":
        print(check_clinic_items.remote())
        return
    if stage == "clinic_gen":
        print(clinic_gen.remote())
        return
    if stage == "clinic_build":
        print(clinic_build.remote())
        return
    table = {"eval": CELLS, "train": TRAIN, "quant": QUANT}.get(stage)
    if table is None:
        raise SystemExit(f"unknown stage {stage!r}; use eval, data, train, or quant")
    if len(names) == 0:
        raise SystemExit(f"stage {stage} needs --cells; the cells are {', '.join(table)}")
    for name in names:
        if name not in table:
            raise SystemExit(f"unknown cell {name!r} for stage {stage}; the cells are {', '.join(table)}")
    calls = []
    for name in names:
        if stage == "eval":
            runner = EVAL_RUNNERS[gpu or CELLS[name]["gpu"]]
            calls.append((name, runner.spawn(name, smoke, sample)))
        elif stage == "train":
            calls.append((name, train_cell.spawn(name, smoke, json.loads(settings) if settings else None)))
        else:
            calls.append((name, quant_cell.spawn(name, scheme, samples, smoke)))
    for name, call in calls:
        row = call.get()
        print(f"[{name}] " + json.dumps({k: v for k, v in row.items() if k not in ("losses", "config")}, default=str))
        print("wrote", write_local(row, ("smoke_" if smoke else "") + name + ("" if stage == "eval" else "_" + stage)))
