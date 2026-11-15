"""Clinic efficiency sweep: medical accuracy, speed and memory of one model under many vLLM configs.

A copy of modal_efficiency.py for the clinic demo. Per config, in a fresh vLLM process on one GPU: chain-of-thought
and letter-probability accuracy on every clinic test set (MedMCQA val 1000, MedQA test, PubMedQA test 500, MMLU
medical six, MMLU general 1000), decode tokens per second at batch 1, 8 and 32 on medical prompts, prefill tokens per
second, TTFT, weights, KV cache, peak GPU memory, and clinic_b1_tps (24 real clinic requests one at a time).

    modal run --detach modal_efficiency_clinic.py --model vol:clinic/c1/k48/r1/model --name k48r1 \
        --configs base,fp8,marlin+eagle3b5 --gpu L40S
    modal run modal_efficiency_clinic.py --name k48r1 --collect
    modal run --detach modal_efficiency_clinic.py --quantize w4a16 --quant-name k48r1_w4a16 \
        --model vol:clinic/c1/k48/r1/model --calib-path clinic/c1/k48/r1/sft.jsonl

Rows go to /vol/results/eff_clinic/<name>/ on the volume wsp-models and to results/eff_clinic_<name>.json.
"""

import argparse
import json
import math
import os
import random
import re
import statistics
import subprocess
import sys
import threading
import time

import modal

app = modal.App("wsp-efficiency-clinic")
volume = modal.Volume.from_name("wsp-models", create_if_missing=True)

VLLM_VERSION = "0.32.0"

# The image of modal_serve.py, which ran vLLM 0.32.0 on an H200 (outputs/serve_full.log).
image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.12")
    .env({"CUDA_HOME": "/usr/local/cuda"})
    .pip_install(
        f"vllm=={VLLM_VERSION}", "accelerate", "huggingface_hub", "requests",
        "safetensors", "zstandard", "numpy", "datasets", "sentencepiece", "protobuf",
    )
    .env({"HF_HOME": "/vol/hf", "HF_HUB_DISABLE_XET": "1"})
    .add_local_python_source("wsp")
)
cpu_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("huggingface_hub", "safetensors", "torch", "numpy", "requests")
    .env({"HF_HOME": "/vol/hf", "HF_HUB_DISABLE_XET": "1"})
)
# llm-compressor brings its own torch and transformers, so it gets its own image and is smoked on its own.
quant_image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.12")
    .pip_install("llmcompressor", "accelerate", "huggingface_hub", "datasets", "safetensors", "zstandard",
                 "sentencepiece", "protobuf", "requests", "numpy")
    .env({"HF_HOME": "/vol/hf", "HF_HUB_DISABLE_XET": "1"})
    .add_local_python_source("wsp")
)

RESULT_DIR = "/vol/results/eff_clinic"
LOCAL_DIR = "results"
BASE_MODEL = "Qwen/Qwen3-30B-A3B-Instruct-2507"

PROMPTS = 64
GEN_LEN = 256
BATCHES = (1, 8, 32)
EVAL_TOKENS = 512
PROMPT_CHARS = 1500
MAX_MODEL_LEN = 4096
GPU_UTILIZATION = 0.90
WARMUP_PROMPTS = 4
SMOKE_PROMPTS = 8
SMOKE_BATCHES = (1, 8)
GPUS = ("H200", "H100", "L40S", "A10G", "L4")
SCORE_SETS = ["medmcqa", "medqa", "pubmedqa", "mmlu_med6", "mmlu_general_ex_med6"]
COT_TOKENS = 400
CLINIC_B1_ITEMS = 24

# ---------------------------------------------------------------- configs

EAGLE3 = "RedHatAI/Qwen3-30B-A3B-Instruct-2507-speculator.eagle3"
EAGLE3_ALT = "lmsys/SGLang-EAGLE3-Qwen3-30B-A3B-Instruct-2507-SpecForge-Nex"


def spec(**kw):
    return {"speculative_config": kw}


# Each part adds vLLM LLM() arguments. hf_overrides edits the model config without touching the saved files.
PARTS = {
    "base": {},
    "fp8": {"quantization": "fp8"},
    "kvfp8": {"kv_cache_dtype": "fp8"},
    "topk6": {"hf_overrides": {"num_experts_per_tok": 6}},
    "topk4": {"hf_overrides": {"num_experts_per_tok": 4}},
    "ngram": spec(method="ngram", num_speculative_tokens=4, prompt_lookup_max=4, prompt_lookup_min=2),
    "ngram8": spec(method="ngram", num_speculative_tokens=8, prompt_lookup_max=5, prompt_lookup_min=2),
    "draft06": spec(method="draft_model", model="Qwen/Qwen3-0.6B", num_speculative_tokens=4),
    "draft17": spec(method="draft_model", model="Qwen/Qwen3-1.7B", num_speculative_tokens=4),
    "eagle3": spec(method="eagle3", model=EAGLE3, num_speculative_tokens=3),
    "eagle3b": spec(method="eagle3", model=EAGLE3_ALT, num_speculative_tokens=3),
    "eagle3b2": spec(method="eagle3", model=EAGLE3_ALT, num_speculative_tokens=2),
    "eagle3b5": spec(method="eagle3", model=EAGLE3_ALT, num_speculative_tokens=5),
    "eagle3b7": spec(method="eagle3", model=EAGLE3_ALT, num_speculative_tokens=7),
    "marlin": {"moe_backend": "marlin"},
    "eager": {"enforce_eager": True},
    "nochunk": {"enable_chunked_prefill": False},
    "seqs32": {"max_num_seqs": 32},
    "seqs256": {"max_num_seqs": 256},
}
DEFAULT_LLM_KWARGS = {"max_num_seqs": 64}


def parse_config(name):
    """LLM() keyword arguments of a config name such as "fp8+kvfp8+ngram". Later parts win. ValueError if unknown."""
    kwargs = dict(DEFAULT_LLM_KWARGS)
    parts = [part for part in name.split("+") if part]
    if not parts:
        raise ValueError("empty config name")
    for part in parts:
        if part not in PARTS:
            raise ValueError(f"unknown config part {part!r} in {name!r}; the parts are {', '.join(PARTS)}")
        for key, value in PARTS[part].items():
            if key in kwargs and isinstance(kwargs[key], dict) and isinstance(value, dict):
                kwargs[key] = {**kwargs[key], **value}
            else:
                kwargs[key] = value
    return kwargs


def select_configs(configs):
    """List of config names from a comma string. ValueError on an unknown part or an empty list."""
    names = [name.strip() for name in configs.split(",") if name.strip()]
    if not names:
        raise ValueError("no configs selected (zero cells)")
    for name in names:
        parse_config(name)
    seen = []
    for name in names:
        if name not in seen:
            seen.append(name)
    return seen


def split_shards(names, shards):
    """Round-robin split into at most `shards` non-empty lists."""
    shards = max(1, min(shards, len(names)))
    return [names[i::shards] for i in range(shards)]


def model_ref(model):
    """(kind, value) from "hub:<id>" or "vol:<path>". ValueError otherwise."""
    kind, _, value = model.partition(":")
    if kind not in ("hub", "vol") or not value:
        raise ValueError(f"--model must be hub:<id> or vol:<path>, got {model!r}")
    return kind, value


def row_key(config, gpu, smoke):
    return ("smoke_" if smoke else "") + config + "__" + gpu


# ---------------------------------------------------------------- decontamination


def ngrams(text, n=13):
    """Set of n-grams of lowercase alphanumeric words."""
    words = re.findall(r"[a-z0-9_]+", text.lower())
    return {" ".join(words[i:i + n]) for i in range(len(words) - n + 1)}


def drop_13gram(texts, test_texts, n=13):
    """(kept texts, dropped count): drop a text that shares an n-gram with any test text."""
    banned = set()
    for text in test_texts:
        banned |= ngrams(text, n)
    kept = []
    dropped = 0
    for text in texts:
        if ngrams(text, n) & banned:
            dropped += 1
        else:
            kept.append(text)
    return kept, dropped


# ---------------------------------------------------------------- measurement helpers (copied from modal_serve.py)


def percentile(values, share):
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(share * len(ordered)) - 1))
    return ordered[index]


def measure_decode(generate, prompt_ids, batch, gen_len, clock=time.perf_counter):
    """(decode tokens per second, [1-token seconds per chunk]). Decode time = 256-token run minus 1-token run."""
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


def ttft_stats(seconds):
    if not seconds:
        return None, None
    millis = [value * 1000.0 for value in seconds]
    return statistics.median(millis), percentile(millis, 0.9)


def gpu_used_gib():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=10).stdout
        return float(out.strip().splitlines()[0]) / 1024.0
    except Exception:
        return None


class GpuSampler:
    """Poll GPU memory in a thread and keep the peak (the engine runs in another process)."""

    def __init__(self, read=gpu_used_gib, interval=0.5):
        self.read = read
        self.interval = interval
        self.peak = None
        self._stop = threading.Event()
        self._thread = None

    def sample(self):
        value = self.read()
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


def dir_gib(path, suffix=".safetensors"):
    total = 0
    for name in os.listdir(path):
        if name.endswith(suffix):
            total += os.path.getsize(os.path.join(path, name))
    return total / 1024 ** 3


# ---------------------------------------------------------------- vLLM log parsing and spec metrics


LOG_PATTERNS = {
    "model_loading_gib": re.compile(r"Model loading took ([\d.]+) GiB"),
    "kv_cache_gib": re.compile(r"Available KV cache memory: ([\d.]+) GiB"),
    "kv_cache_tokens": re.compile(r"GPU KV cache size: ([\d,]+) tokens"),
    "kv_concurrency": re.compile(r"Maximum concurrency for [\d,]+ tokens per request: ([\d.]+)x"),
    "engine_init_seconds": re.compile(r"init engine \(profile, create kv cache, warmup model\) took ([\d.]+) s"),
}


def parse_log(lines):
    """Values from the vLLM engine log. "model_loading_gib" is the first match (the target model), and
    "model_loading_all_gib" lists every match (a draft model loads a second time)."""
    out = {}
    loads = []
    for line in lines:
        for key, pattern in LOG_PATTERNS.items():
            match = pattern.search(line)
            if not match:
                continue
            value = float(match.group(1).replace(",", ""))
            if key == "model_loading_gib":
                loads.append(value)
            elif key not in out:
                out[key] = value
    if loads:
        out["model_loading_gib"] = loads[0]
        out["model_loading_all_gib"] = loads
    return out


SPEC_NAMES = {"vllm:spec_decode_num_drafts": "drafts", "vllm:spec_decode_num_draft_tokens": "draft_tokens",
              "vllm:spec_decode_num_accepted_tokens": "accepted"}


def spec_counts(llm):
    """Cumulative speculative counters {"drafts", "draft_tokens", "accepted"} (zeros when not speculative)."""
    counts = {"drafts": 0, "draft_tokens": 0, "accepted": 0}
    try:
        for metric in llm.get_metrics():
            key = SPEC_NAMES.get(getattr(metric, "name", None))
            if key is not None and hasattr(metric, "value"):
                counts[key] += metric.value
    except Exception as error:
        counts["error"] = repr(error)[:120]
    return counts


def spec_delta(before, after):
    """Acceptance over one phase: rate = accepted/draft_tokens, mean_accept_len = 1 + accepted/drafts."""
    drafts = after["drafts"] - before["drafts"]
    draft_tokens = after["draft_tokens"] - before["draft_tokens"]
    accepted = after["accepted"] - before["accepted"]
    if drafts <= 0 or draft_tokens <= 0:
        return None
    return {"drafts": drafts, "draft_tokens": draft_tokens, "accepted": accepted,
            "rate": accepted / draft_tokens, "mean_accept_len": 1 + accepted / drafts}


# ---------------------------------------------------------------- the worker (one vLLM process per config)


def build_prompts(tokenizer, count):
    """Rendered clinic reasoning prompts (MedQA test vignettes; used for speed only, never scored here)."""
    from wsp import clinic, skills_eval

    items = skills_eval.medqa_items("test")
    if len(items) < count:
        raise ValueError(f"MedQA gave {len(items)} prompts, need {count}")
    return [skills_eval.render_prompt(tokenizer, clinic.reason_user(it["question"], it["choices"]))
            for it in items[:count]]


def clinic_scores(tokenizer, llm, limit=0):
    """{set: {cot, lp, n, cut, mean_tokens}} plus raw samples. cot = reasoning prompt, greedy, parsed letter;
    lp = letter log-probability on the plain prompt."""
    from vllm import SamplingParams

    from wsp import clinic, rft, skills_eval

    ids = skills_eval.letter_ids(tokenizer)
    out, raw = {}, {}
    for name in SCORE_SETS:
        items = skills_eval.load_items(name)
        if limit:
            items = items[:limit]
        prompts = [skills_eval.render_prompt(tokenizer, clinic.reason_user(it["question"], it["choices"])) for it in items]
        start = time.time()
        outs = llm.generate(prompts, SamplingParams(temperature=0.0, max_tokens=COT_TOKENS), use_tqdm=False)
        seconds = time.time() - start
        texts = [o.outputs[0].text for o in outs]
        made = sum(len(o.outputs[0].token_ids) for o in outs)
        cot = [rft.letter_of(t) == "ABCD"[it["gold"]] for it, t in zip(items, texts)]
        lp_outs = llm.generate([it["prompt"] for it in items], SamplingParams(temperature=0.0, max_tokens=1, logprobs=20),
                               use_tqdm=False)
        lp = []
        for it, o in zip(items, lp_outs):
            logs = o.outputs[0].logprobs[0]
            n = len(it["choices"])
            scores = [logs[t].logprob if t in logs else -1e9 for t in ids[:n]]
            lp.append(max(range(n), key=lambda j: scores[j]) == it["gold"])
        out[name] = {"cot": sum(cot) / len(cot), "lp": sum(lp) / len(lp), "n": len(items),
                     "cot_correct": sum(cot), "lp_correct": sum(lp),
                     "cut": sum(1 for o in outs if o.outputs[0].finish_reason == "length"),
                     "mean_tokens": made / len(items), "gen_tps": made / max(seconds, 1e-9), "seconds": round(seconds, 1)}
        raw[name] = [{"id": it["id"], "gold": "ABCD"[it["gold"]], "output": t[:700]} for it, t in zip(items[:3], texts[:3])]
        print(f"[{CELL}] {name} cot {sum(cot)}/{len(cot)} = {out[name]['cot']:.4f} lp {sum(lp)}/{len(lp)} = "
              f"{out[name]['lp']:.4f} cut {out[name]['cut']} tokens {out[name]['mean_tokens']:.0f}", flush=True)
    return out, raw


CELL = "cell"


def worker_main(job_path):
    """Load one config in vLLM, measure it, write the result JSON next to the job file."""
    import torch
    import transformers
    import vllm
    from vllm import LLM, SamplingParams

    from wsp import skills_eval

    with open(job_path) as f:
        job = json.load(f)
    cell = job["cell"]
    global CELL
    CELL = cell
    smoke = job["smoke"]
    prompts_n, batches = (SMOKE_PROMPTS, SMOKE_BATCHES) if smoke else (PROMPTS, BATCHES)
    result = {"vllm": vllm.__version__, "torch": torch.__version__, "transformers": transformers.__version__,
              "gpu": torch.cuda.get_device_name(0), "decode_tps": {}, "accept": {}}
    tokenizer = transformers.AutoTokenizer.from_pretrained(job["tokenizer"])
    prompt_ids = [list(tokenizer(p, add_special_tokens=False)["input_ids"]) for p in build_prompts(tokenizer, prompts_n)]
    result["prompt_tokens"] = sum(len(ids) for ids in prompt_ids)
    print(f"[{cell}] vllm {vllm.__version__} gpu {result['gpu']} kwargs {json.dumps(job['llm_kwargs'])}", flush=True)

    start = time.time()
    llm = LLM(model=job["model"], tokenizer=job["tokenizer"], dtype="bfloat16", seed=0,
              gpu_memory_utilization=job["util"], max_model_len=MAX_MODEL_LEN, enable_prefix_caching=False,
              disable_log_stats=False, **job["llm_kwargs"])
    result["load_seconds"] = round(time.time() - start, 1)
    result["gpu_after_load_gib"] = gpu_used_gib()
    print(f"[{cell}] loaded in {result['load_seconds']} s", flush=True)

    def generate(id_lists, max_tokens):
        params = SamplingParams(temperature=0.0, max_tokens=max_tokens, ignore_eos=True)
        llm.generate([{"prompt_token_ids": ids} for ids in id_lists], params, use_tqdm=False)

    generate(prompt_ids[:WARMUP_PROMPTS], 16)
    generate(prompt_ids, 1)  # the first full-size prefill compiles MoE kernels; time the second
    result["prefill_tps"] = measure_prefill(generate, prompt_ids)
    print(f"[{cell}] prefill {result['prefill_tps']} tokens/s", flush=True)
    for batch in batches:
        before = spec_counts(llm)
        tps, first = measure_decode(generate, prompt_ids, batch, GEN_LEN)
        result["decode_tps"][str(batch)] = tps
        result["accept"][f"decode_b{batch}"] = spec_delta(before, spec_counts(llm))
        print(f"[{cell}] decode batch {batch}: {tps} tokens/s accept {result['accept'][f'decode_b{batch}']}", flush=True)
        if batch == 1:
            result["ttft_ms_median"], result["ttft_ms_p90"] = ttft_stats(first)
            print(f"[{cell}] ttft median {result['ttft_ms_median']} ms", flush=True)

    if not smoke:
        before = spec_counts(llm)
        result["clinic"], result["samples"] = clinic_scores(tokenizer, llm)
        result["accept"]["clinic"] = spec_delta(before, spec_counts(llm))
        # Real batch-1 speed: clinic reasoning requests one at a time, natural stop.
        from wsp import clinic as clinic_mod

        items = skills_eval.load_items("medqa")[:CLINIC_B1_ITEMS]
        c_prompts = [skills_eval.render_prompt(tokenizer, clinic_mod.reason_user(it["question"], it["choices"]))
                     for it in items]
        params = SamplingParams(temperature=0.0, max_tokens=COT_TOKENS)
        llm.generate(c_prompts[:2], params, use_tqdm=False)
        before = spec_counts(llm)
        tokens = 0
        begin = time.perf_counter()
        for prompt in c_prompts:
            out = llm.generate([prompt], params, use_tqdm=False)
            tokens += len(out[0].outputs[0].token_ids)
        seconds = time.perf_counter() - begin
        result["clinic_b1_tps"] = tokens / seconds
        result["clinic_b1_ttft_ms"] = 1000.0 * seconds / len(c_prompts)
        result["accept"]["clinic_b1"] = spec_delta(before, spec_counts(llm))
        print(f"[{cell}] clinic_b1 {result['clinic_b1_tps']:.1f} tok/s accept {result['accept']['clinic_b1']}", flush=True)
    with open(job["result_path"], "w") as f:
        json.dump(result, f, default=str)
    try:
        del llm
    except Exception:
        pass


# ---------------------------------------------------------------- the GPU side


ROW_KEYS = ["cell", "model", "config", "gpu", "smoke", "util", "vllm", "torch", "prompts", "gen_len", "weights_disk_gib",
            "model_loading_gib", "model_loading_all_gib", "kv_cache_gib", "kv_cache_tokens", "kv_concurrency",
            "gpu_after_load_gib", "gpu_peak_gib", "load_seconds", "engine_init_seconds", "prefill_tps",
            "ttft_ms_median", "ttft_ms_p90", "decode_tps", "clinic_b1_tps", "clinic_b1_ttft_ms", "clinic", "accept",
            "samples", "llm_kwargs", "seconds", "error", "date"]


def safe_reload():
    try:
        volume.reload()
    except Exception as error:
        print("volume reload skipped:", repr(error)[:200], flush=True)


def resolve_model(kind, value, log):
    """Local path or Hub id for vLLM, and a local directory for sizing the weights."""
    if kind == "vol":
        path = os.path.join("/vol", value)
        return path, path
    from huggingface_hub import snapshot_download

    log(f"downloading {value}")
    local = snapshot_download(value, allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "*.tiktoken"])
    return local, local


def run_config(name, config, model, tokenizer, gpu, smoke, util, now=time.strftime):
    """Run one config in a subprocess. Returns the row (error row on failure)."""
    import tempfile

    kind, value = model_ref(model)
    cell = f"{name}:{config}"
    log = lambda text: print(f"[{cell}] {text}", flush=True)
    row = {key: None for key in ROW_KEYS}
    row.update({"cell": cell, "model": model, "config": config, "gpu": gpu, "smoke": smoke, "util": util,
                "gen_len": GEN_LEN, "prompts": SMOKE_PROMPTS if smoke else PROMPTS, "date": now("%Y-%m-%d"),
                "llm_kwargs": parse_config(config)})
    started = time.time()
    try:
        local, size_dir = resolve_model(kind, value, log)
        row["weights_disk_gib"] = round(dir_gib(size_dir), 2)
        tmp = tempfile.mkdtemp()
        job = {"cell": cell, "smoke": smoke, "model": local, "tokenizer": tokenizer, "util": util,
               "llm_kwargs": row["llm_kwargs"], "result_path": os.path.join(tmp, "result.json"),
               "raw_path": os.path.join(tmp, "raw")}
        job_path = os.path.join(tmp, "job.json")
        with open(job_path, "w") as f:
            json.dump(job, f)
        lines = []
        sampler = GpuSampler()
        with sampler:
            process = subprocess.Popen(
                [sys.executable, "-c", f"import modal_efficiency as m; m.worker_main({job_path!r})"],
                cwd="/root", stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
            for line in process.stdout:
                lines.append(line)
                if line.startswith(f"[{cell}]") or "ERROR" in line or "Error" in line:
                    print(line.rstrip()[:400], flush=True)
            process.wait()
        row["gpu_peak_gib"] = None if sampler.peak is None else round(sampler.peak, 2)
        row.update({k: v for k, v in parse_log(lines).items() if k in ROW_KEYS})
        if process.returncode != 0 or not os.path.exists(job["result_path"]):
            tail = [line.rstrip() for line in lines if line.strip()][-25:]
            raise RuntimeError(f"worker exit {process.returncode}: " + " | ".join(tail)[-1500:])
        with open(job["result_path"]) as f:
            result = json.load(f)
        for key in ("vllm", "torch", "prefill_tps", "ttft_ms_median", "ttft_ms_p90", "decode_tps", "clinic_b1_tps",
                    "clinic_b1_ttft_ms", "clinic", "accept", "samples", "load_seconds", "gpu_after_load_gib"):
            if key in result:
                row[key] = result[key]
        row["gpu"] = result.get("gpu", gpu)
    except Exception as error:
        log("failed: " + repr(error)[:600])
        row["error"] = repr(error)[:1800]
    row["seconds"] = round(time.time() - started, 1)
    return row


def save_row(name, row):
    directory = os.path.join(RESULT_DIR, name)
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, row_key(row["config"], row["gpu_label"], row["smoke"]) + ".json")
    with open(path, "w") as f:
        json.dump(row, f, indent=2, default=str)
    try:
        volume.commit()
    except Exception as error:
        print("volume commit failed:", repr(error)[:200], flush=True)


def run_configs(name, configs, model, tokenizer, gpu, smoke, util):
    safe_reload()
    rows = []
    for config in configs:
        row = run_config(name, config, model, tokenizer, gpu, smoke, util)
        row["gpu_label"] = gpu
        save_row(name, row)
        rows.append(row)
        if row["error"]:
            print(f"[{row['cell']}] FAILED {row['error'][:300]}", flush=True)
        else:
            print(f"[{row['cell']}] DONE " + json.dumps({k: row[k] for k in
                  ["decode_tps", "prefill_tps", "clinic_b1_tps", "model_loading_gib", "kv_cache_tokens"]}), flush=True)
    return rows


@app.function(image=image, volumes={"/vol": volume}, gpu="H200", memory=131072, timeout=6 * 3600,
              single_use_containers=True)
def run_h200(name, configs, model, tokenizer, smoke, util):
    return run_configs(name, configs, model, tokenizer, "H200", smoke, util)


@app.function(image=image, volumes={"/vol": volume}, gpu="H100", memory=131072, timeout=6 * 3600,
              single_use_containers=True)
def run_h100(name, configs, model, tokenizer, smoke, util):
    return run_configs(name, configs, model, tokenizer, "H100", smoke, util)


@app.function(image=image, volumes={"/vol": volume}, gpu="L40S", memory=65536, timeout=6 * 3600,
              single_use_containers=True)
def run_l40s(name, configs, model, tokenizer, smoke, util):
    return run_configs(name, configs, model, tokenizer, "L40S", smoke, util)


@app.function(image=image, volumes={"/vol": volume}, gpu="A10G", memory=65536, timeout=6 * 3600,
              single_use_containers=True)
def run_a10g(name, configs, model, tokenizer, smoke, util):
    return run_configs(name, configs, model, tokenizer, "A10G", smoke, util)


@app.function(image=image, volumes={"/vol": volume}, gpu="L4", memory=65536, timeout=6 * 3600,
              single_use_containers=True)
def run_l4(name, configs, model, tokenizer, smoke, util):
    return run_configs(name, configs, model, tokenizer, "L4", smoke, util)


RUNNERS = {"H200": run_h200, "H100": run_h100, "L40S": run_l40s, "A10G": run_a10g, "L4": run_l4}


# ---------------------------------------------------------------- volume setup: slim rebuild, quantize, collect


QUANT_SCHEMES = {"w4a16": "W4A16", "fp8": "FP8_DYNAMIC"}
CALIB_SAMPLES = 512
CALIB_LENGTH = 2048


def calibration_texts(tokenizer, count, calib_path, log=print):
    """Clinic calibration text: chat-rendered (question, answer) rows of a round's sft.jsonl on the volume. Those rows
    are train-split questions with verified model answers, already 13-gram filtered against every clinic test set."""
    rows = []
    with open(os.path.join("/vol", calib_path)) as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    random.Random(3).shuffle(rows)
    if len(rows) < count:
        raise ValueError(f"only {len(rows)} calibration rows in {calib_path}, need {count}")
    texts = [tokenizer.apply_chat_template([{"role": "user", "content": r["user"]},
                                            {"role": "assistant", "content": r["assistant"]}], tokenize=False)
             for r in rows[:count]]
    log(f"calibration: {len(rows)} rows in {calib_path}, used {count}")
    return texts, {"rows": len(rows), "used": count, "path": calib_path}


def quantize_model(source, out_dir, scheme, samples=CALIB_SAMPLES, calib_path=""):
    """Write a compressed-tensors checkpoint of the model at `source` (path or Hub id). w4a16 is GPTQ calibrated on
    code TRAIN text, fp8 is data-free FP8_DYNAMIC. The router (mlp.gate) and lm_head stay in bf16."""
    import shutil

    import torch
    from llmcompressor import oneshot
    from llmcompressor.modifiers.quantization import GPTQModifier, QuantizationModifier
    from transformers import AutoModelForCausalLM, AutoTokenizer

    ignore = ["lm_head", "re:.*mlp.gate$"]
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForCausalLM.from_pretrained(source, dtype=torch.bfloat16, device_map="auto")
    info = {"scheme": scheme, "source": source}
    if scheme == "fp8":
        recipe = QuantizationModifier(targets="Linear", scheme="FP8_DYNAMIC", ignore=ignore)
        oneshot(model=model, recipe=recipe)
    else:
        from datasets import Dataset

        texts, stats = calibration_texts(tokenizer, samples, calib_path)
        info.update(stats)
        dataset = Dataset.from_dict({"text": texts})
        dataset = dataset.map(lambda row: tokenizer(row["text"], max_length=CALIB_LENGTH, truncation=True,
                                                    add_special_tokens=False), remove_columns=["text"])
        recipe = GPTQModifier(targets="Linear", scheme="W4A16", ignore=ignore)
        oneshot(model=model, dataset=dataset, recipe=recipe, max_seq_length=CALIB_LENGTH,
                num_calibration_samples=len(texts), moe_calibrate_all_experts=True)
    model.save_pretrained(out_dir, save_compressed=True)
    tokenizer.save_pretrained(out_dir)
    with open(os.path.join(out_dir, "quant_info.json"), "w") as f:
        json.dump(info, f)
    return info


def quantize_job(source, name, scheme, smoke=False, calib_path=""):
    safe_reload()
    out_dir = os.path.join("/vol/quant", name)
    if os.path.exists(os.path.join(out_dir, "quant_info.json")):
        print("already quantized", out_dir)
        return out_dir
    if source.startswith("hub:"):
        local = source[4:]
    else:
        local = os.path.join("/vol", source[4:])
    start = time.time()
    info = quantize_model(local, out_dir, scheme, samples=16 if smoke else CALIB_SAMPLES, calib_path=calib_path)
    volume.commit()
    print(f"quantized {name} {scheme} in {time.time() - start:.0f} s", info, "disk GiB", round(dir_gib(out_dir), 2), flush=True)
    return out_dir


@app.function(image=quant_image, volumes={"/vol": volume}, gpu="H200", memory=262144, timeout=6 * 3600,
              single_use_containers=True)
def quantize_h200(source, name, scheme, smoke=False, calib_path=""):
    return quantize_job(source, name, scheme, smoke, calib_path)


@app.function(image=quant_image, volumes={"/vol": volume}, gpu="L4", memory=65536, timeout=3600,
              single_use_containers=True)
def quantize_l4(source, name, scheme, smoke=False, calib_path=""):
    return quantize_job(source, name, scheme, smoke, calib_path)


@app.function(image=image, timeout=600)
def shell(command):
    """Run a shell command in the vLLM image (to read the installed vLLM source)."""
    out = subprocess.run(command, shell=True, capture_output=True, text=True)
    return out.stdout + out.stderr


@app.function(image=cpu_image, volumes={"/vol": volume}, timeout=1800)
def collect_rows(name):
    safe_reload()
    rows = []
    directory = os.path.join(RESULT_DIR, name)
    if os.path.isdir(directory):
        for file in sorted(os.listdir(directory)):
            if file.endswith(".json"):
                with open(os.path.join(directory, file)) as f:
                    rows.append(json.load(f))
    return rows


def merge_rows(path, rows):
    """Merge rows into a local JSON list by (cell, gpu, smoke); new rows replace old ones. Returns the list."""
    old = []
    if os.path.exists(path):
        with open(path) as f:
            old = json.load(f)
    table = {(r["cell"], r.get("gpu_label") or r.get("gpu"), r.get("smoke")): r for r in old}
    for row in rows:
        table[(row["cell"], row.get("gpu_label") or row.get("gpu"), row.get("smoke"))] = row
    merged = list(table.values())
    with open(path, "w") as f:
        json.dump(merged, f, indent=2, default=str)
    return merged


def summary_line(row):
    clinic = row.get("clinic") or {}
    cot = {k: round(v["cot"], 4) for k, v in clinic.items()}
    return (f"[{row['cell']}] gpu {row.get('gpu_label') or row.get('gpu')} cot {cot} decode {row.get('decode_tps')} "
            f"clinic_b1 {row.get('clinic_b1_tps')} weights {row.get('model_loading_gib')} GiB kv {row.get('kv_cache_tokens')} tok"
            + (f" ERROR {row['error'][:200]}" if row.get("error") else ""))


@app.local_entrypoint()
def main(model: str = "hub:" + BASE_MODEL, name: str = "full", configs: str = "base", gpu: str = "H200",
         tokenizer: str = BASE_MODEL, smoke: bool = False, shards: int = 1, estimate_only: bool = False,
         collect: bool = False, calib_path: str = "", util: float = GPU_UTILIZATION,
         quantize: str = "", quant_name: str = "", shell_command: str = ""):
    if shell_command:
        print(shell.remote(shell_command))
        return
    if collect:
        rows = collect_rows.remote(name)
        merged = merge_rows(os.path.join(LOCAL_DIR, f"eff_clinic_{name}.json"), rows)
        print(f"collected {len(rows)} rows, {len(merged)} in results/eff_clinic_{name}.json")
        for row in rows:
            print(summary_line(row))
        return
    if quantize:
        if quantize not in QUANT_SCHEMES or not quant_name:
            raise SystemExit("--quantize is w4a16 or fp8 and needs --quant-name")
        model_ref(model)
        runner = quantize_l4 if gpu == "L4" else quantize_h200
        print(runner.remote(model, quant_name, quantize, smoke, calib_path))
        return
    if gpu not in RUNNERS:
        raise SystemExit(f"--gpu must be one of {', '.join(RUNNERS)}")
    model_ref(model)
    names = select_configs(configs)
    groups = split_shards(names, shards)
    print(f"model {model}  name {name}  gpu {gpu}  smoke {smoke}  util {util}")
    print(f"{len(names)} cells in {len(groups)} containers: {groups}")
    rough_minutes = len(names) * (4 if smoke else 11)
    print(f"rough GPU time {rough_minutes / 60:.1f} h")
    if estimate_only:
        return
    calls = [RUNNERS[gpu].spawn(name, group, model, tokenizer, smoke, util) for group in groups]
    for call in calls:
        rows = call.get()
        merge_rows(os.path.join(LOCAL_DIR, f"eff_clinic_{name}.json"), rows)
        for row in rows:
            print(summary_line(row))
