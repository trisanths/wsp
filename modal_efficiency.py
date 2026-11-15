"""Efficiency sweep: accuracy, speed, and memory of one model under many vLLM configs.

One model, a list of configs. For every config it measures, in a fresh vLLM process on one GPU:

  HumanEval 164 and MBPP 257 (sanitized test) pass@1, greedy, 512 new tokens, with the prompts and grader of
  wsp.skills_eval; decode tokens per second at batch 1, 8, 32 (256 tokens each, time of a 256-token run minus
  time of a 1-token run); prefill tokens per second; TTFT median and p90; weight bytes on disk; vLLM
  "Model loading took" GiB; KV cache GiB, tokens and concurrency at the fixed gpu_memory_utilization; peak
  nvidia-smi memory; for speculative configs, the draft acceptance rate on the batch-1 decode and on HumanEval;
  and he_b1_tps, the tokens per second of 24 real HumanEval requests run one at a time (natural stop), which
  is the honest batch-1 number for speculative decoding.

A config is a "+" join of parts from PARTS, for example "fp8+kvfp8+ngram". "base" is no change.

    modal run --detach modal_efficiency.py --model hub:Qwen/Qwen3-30B-A3B-Instruct-2507 --name full \
        --configs base,fp8,kvfp8 --gpu H200 --shards 3
    modal run modal_efficiency.py --model vol:slim_models/round19/moe_saliency_64 --name slim64 --smoke
    modal run modal_efficiency.py --name full --collect        (copy rows from the volume to results/)

--model is hub:<id> or vol:<path under /vol>. --tokenizer is the Hub id of the tokenizer (default the 30B
base). Rows go to /vol/results/efficiency/<name>/<config>__<gpu>.json on volume wsp-efficiency (written by the
GPU function after each config, so a dead client loses nothing) and are merged into
results/efficiency_<name>.json by the entrypoint or by --collect. Run --estimate-only first.

Setup steps, one per workspace volume:
    --build-slim   rebuild the keep-64 Qwen3-30B-A3B slim on the volume from the Hub weights and the manifest
                   results/efficiency_slim64_manifest.json (kept experts per layer of round 19)
    --quantize w4a16|fp8   write a compressed-tensors checkpoint of --model to the volume (llm-compressor)
"""

import argparse
import json
import math
import os
import re
import statistics
import subprocess
import sys
import threading
import time

import modal

app = modal.App("wsp-efficiency")
volume = modal.Volume.from_name("wsp-efficiency", create_if_missing=True)

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

RESULT_DIR = "/vol/results/efficiency"
LOCAL_DIR = "results"
BASE_MODEL = "Qwen/Qwen3-30B-A3B-Instruct-2507"
MANIFEST = "results/efficiency_slim64_manifest.json"
SLIM_DIR = "/vol/slim_models/round19/moe_saliency_64"

PROMPTS = 64
GEN_LEN = 256
BATCHES = (1, 8, 32)
EVAL_TOKENS = 512
PROMPT_CHARS = 1500
MAX_MODEL_LEN = 4096
GPU_UTILIZATION = 0.90
WARMUP_PROMPTS = 4
HE_B1_ITEMS = 24
SMOKE_PROMPTS = 8
SMOKE_BATCHES = (1, 8)
GPUS = ("H200", "H100", "L40S", "A10G", "L4")
REPORTED = ["humaneval", "mbpp"]

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


def test_text_pool():
    """Every text of the HumanEval and MBPP test items (prompt, tests, entry point, statement)."""
    from wsp import skills_eval

    texts = []
    for item in skills_eval.humaneval_items():
        texts.append(item["prompt"] + "\n" + item["test"])
    for item in skills_eval.mbpp_items():
        texts.append(item["text"] + "\n" + "\n".join(item["test_list"]))
    return texts


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
    from wsp import healdata, skills_eval

    pairs = healdata.pairs("code", count, seed=0)
    if len(pairs) < count:
        raise ValueError(f"the code pool gave {len(pairs)} prompts, need {count}")
    return [skills_eval.render_prompt(tokenizer, user[:PROMPT_CHARS]) for user, _ in pairs[:count]]


def grade_set(skill, tokenizer, llm, max_tokens=EVAL_TOKENS):
    """Run one benchmark through the engine. Returns (summary, raw records)."""
    from vllm import SamplingParams

    from wsp import skills_eval

    items = skills_eval.humaneval_items() if skill == "humaneval" else skills_eval.mbpp_items()
    prompts = [skills_eval.render_prompt(tokenizer, skills_eval.user_text(skill, item)) for item in items]
    params = SamplingParams(temperature=0.0, max_tokens=max_tokens)
    start = time.time()
    outputs = llm.generate(prompts, params, use_tqdm=False)
    seconds = time.time() - start
    texts = [out.outputs[0].text for out in outputs]
    made = sum(len(out.outputs[0].token_ids) for out in outputs)
    records = skills_eval.grade_outputs(skill, items, texts)
    correct = sum(1 for record in records if record["correct"])
    n = len(records)
    return ({"score": correct / n, "correct": correct, "n": n, "seconds": round(seconds, 1),
             "gen_tps": made / max(seconds, 1e-9), "mean_tokens": made / n}, records, items, prompts)


def worker_main(job_path):
    """Load one config in vLLM, measure it, write the result JSON next to the job file."""
    import torch
    import transformers
    import vllm
    from vllm import LLM, SamplingParams

    from wsp import skills_eval  # noqa: F401

    with open(job_path) as f:
        job = json.load(f)
    cell = job["cell"]
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
        samples = {}
        for skill in ("humaneval", "mbpp"):
            before = spec_counts(llm)
            summary, records, items, prompts = grade_set(skill, tokenizer, llm)
            result[skill] = summary
            result["accept"][skill] = spec_delta(before, spec_counts(llm))
            samples[skill] = [{"id": r["id"], "correct": r["correct"], "output": r["output"][:700]} for r in records[:3]]
            raw_path = job["raw_path"] + f".{skill}.json"
            with open(raw_path, "w") as f:
                json.dump([{"id": r["id"], "correct": r["correct"], "pred": r["pred"], "output": r["output"]}
                           for r in records], f)
            print(f"[{cell}] {skill} {summary['correct']}/{summary['n']} = {summary['score']:.4f} "
                  f"gen {summary['gen_tps']:.0f} tok/s accept {result['accept'][skill]}", flush=True)
        result["samples"] = samples
        # Real batch-1 speed: HumanEval requests one at a time, natural stop.
        items = skills_eval.humaneval_items()[:HE_B1_ITEMS]
        he_prompts = [skills_eval.render_prompt(tokenizer, skills_eval.user_text("humaneval", item)) for item in items]
        params = SamplingParams(temperature=0.0, max_tokens=EVAL_TOKENS)
        llm.generate(he_prompts[:2], params, use_tqdm=False)
        before = spec_counts(llm)
        tokens = 0
        begin = time.perf_counter()
        for prompt in he_prompts:
            out = llm.generate([prompt], params, use_tqdm=False)
            tokens += len(out[0].outputs[0].token_ids)
        seconds = time.perf_counter() - begin
        result["he_b1_tps"] = tokens / seconds
        result["accept"]["he_b1"] = spec_delta(before, spec_counts(llm))
        print(f"[{cell}] he_b1 {result['he_b1_tps']:.1f} tok/s accept {result['accept']['he_b1']}", flush=True)
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
            "ttft_ms_median", "ttft_ms_p90", "decode_tps", "he_b1_tps", "humaneval", "mbpp", "accept",
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
        for key in ("vllm", "torch", "prefill_tps", "ttft_ms_median", "ttft_ms_p90", "decode_tps", "he_b1_tps",
                    "humaneval", "mbpp", "accept", "samples", "load_seconds", "gpu_after_load_gib"):
            if key in result:
                row[key] = result[key]
        row["gpu"] = result.get("gpu", gpu)
        raw_dir = os.path.join(RESULT_DIR, name, "raw")
        os.makedirs(raw_dir, exist_ok=True)
        for skill in ("humaneval", "mbpp"):
            source = job["raw_path"] + f".{skill}.json"
            if os.path.exists(source):
                with open(source) as f:
                    data = f.read()
                with open(os.path.join(raw_dir, f"{row_key(config, gpu, smoke)}.{skill}.json"), "w") as f:
                    f.write(data)
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
                  ["decode_tps", "prefill_tps", "he_b1_tps", "model_loading_gib", "kv_cache_tokens"]}), flush=True)
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


def slim_tensor_plan(name, kept):
    """("drop"|"keep", new name) for one Hub tensor name given kept = {layer: [expert ids]} (sorted).

    Expert e of a layer becomes its position in the kept list. The router "mlp.gate.weight" is cut to kept rows
    by the caller (kind "gate"). Everything else is kept as it is.
    """
    match = re.match(r"model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.(.+)$", name)
    if match:
        layer, expert, rest = int(match.group(1)), int(match.group(2)), match.group(3)
        ids = kept[layer]
        if expert not in ids:
            return "drop", None
        return "keep", f"model.layers.{layer}.mlp.experts.{ids.index(expert)}.{rest}"
    match = re.match(r"model\.layers\.(\d+)\.mlp\.gate\.weight$", name)
    if match:
        return "gate", name
    return "keep", name


def build_slim_files(source_dir, out_dir, manifest, shard_bytes=4 * 1024 ** 3):
    """Write the slim safetensors, index and config from a Hub snapshot and a wsp_slim.json manifest."""
    import shutil

    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file

    kept = {int(layer): sorted(ids) for layer, ids in manifest["kept"].items()}
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(source_dir, "model.safetensors.index.json")) as f:
        index = json.load(f)
    by_file = {}
    for tensor, file in index["weight_map"].items():
        by_file.setdefault(file, []).append(tensor)
    weight_map = {}
    buffer = {}
    buffer_bytes = 0
    shard_no = 0
    pending = []

    def flush():
        nonlocal buffer, buffer_bytes, shard_no
        if not buffer:
            return
        shard_no += 1
        file = f"model-{shard_no:05d}.safetensors"
        save_file(buffer, os.path.join(out_dir, file), metadata={"format": "pt"})
        for tensor in buffer:
            weight_map[tensor] = file
        buffer = {}
        buffer_bytes = 0

    total = 0
    for file in sorted(by_file):
        with safe_open(os.path.join(source_dir, file), framework="pt") as handle:
            for tensor in sorted(by_file[file]):
                action, new_name = slim_tensor_plan(tensor, kept)
                if action == "drop":
                    continue
                value = handle.get_tensor(tensor)
                if action == "gate":
                    layer = int(re.match(r"model\.layers\.(\d+)\.", tensor).group(1))
                    value = value[torch.tensor(kept[int(layer)])].contiguous()
                buffer[new_name] = value
                size = value.numel() * value.element_size()
                buffer_bytes += size
                total += size
                if buffer_bytes >= shard_bytes:
                    flush()
    flush()
    with open(os.path.join(out_dir, "model.safetensors.index.json"), "w") as f:
        json.dump({"metadata": {"total_size": total}, "weight_map": weight_map}, f)
    with open(os.path.join(source_dir, "config.json")) as f:
        config = json.load(f)
    config["num_experts"] = len(next(iter(kept.values())))
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)
    for name in os.listdir(source_dir):
        if name.endswith((".txt", ".jinja")) or name in ("tokenizer.json", "tokenizer_config.json", "vocab.json",
                                                         "merges.txt", "generation_config.json"):
            shutil.copy(os.path.join(source_dir, name), os.path.join(out_dir, name))
    with open(os.path.join(out_dir, "wsp_slim.json"), "w") as f:
        json.dump(manifest, f)
    return total


@app.function(image=cpu_image, volumes={"/vol": volume}, timeout=4 * 3600, memory=65536, cpu=8)
def build_slim(manifest, out_dir=SLIM_DIR):
    from huggingface_hub import snapshot_download

    safe_reload()
    if os.path.exists(os.path.join(out_dir, "model.safetensors.index.json")):
        print("slim already built", out_dir)
        return out_dir
    source = snapshot_download(BASE_MODEL, allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "*.jinja"])
    total = build_slim_files(source, out_dir, manifest)
    volume.commit()
    print("built", out_dir, round(total / 1024 ** 3, 2), "GiB")
    return out_dir


QUANT_SCHEMES = {"w4a16": "W4A16", "fp8": "FP8_DYNAMIC"}
CALIB_SAMPLES = 512
CALIB_LENGTH = 2048


def calibration_texts(tokenizer, count, log=print):
    """Code TRAIN text for calibration: chat-rendered pairs of the healdata "code" pool (default reported set, so no
    HumanEval or MBPP derived source), minus any text that shares a 13-gram with a HumanEval or MBPP test item."""
    from wsp import healdata

    pairs = healdata.pairs("code", count * 2, seed=3)
    texts = [tokenizer.apply_chat_template([{"role": "user", "content": u[:PROMPT_CHARS]},
                                            {"role": "assistant", "content": a}], tokenize=False)
             for u, a in pairs]
    kept, dropped = drop_13gram(texts, test_text_pool())
    log(f"calibration: {len(pairs)} pairs, {dropped} dropped by the 13-gram test check, {len(kept)} left")
    if len(kept) < count:
        raise ValueError(f"only {len(kept)} calibration texts after decontamination, need {count}")
    return kept[:count], {"pairs": len(pairs), "dropped_13gram": dropped, "used": count}


def quantize_model(source, out_dir, scheme, samples=CALIB_SAMPLES):
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

        texts, stats = calibration_texts(tokenizer, samples)
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


def quantize_job(source, name, scheme, smoke=False):
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
    info = quantize_model(local, out_dir, scheme, samples=16 if smoke else CALIB_SAMPLES)
    volume.commit()
    print(f"quantized {name} {scheme} in {time.time() - start:.0f} s", info, "disk GiB", round(dir_gib(out_dir), 2), flush=True)
    return out_dir


@app.function(image=quant_image, volumes={"/vol": volume}, gpu="H200", memory=262144, timeout=6 * 3600,
              single_use_containers=True)
def quantize_h200(source, name, scheme, smoke=False):
    return quantize_job(source, name, scheme, smoke)


@app.function(image=quant_image, volumes={"/vol": volume}, gpu="L4", memory=65536, timeout=3600,
              single_use_containers=True)
def quantize_l4(source, name, scheme, smoke=False):
    return quantize_job(source, name, scheme, smoke)


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
            if file.endswith(".json") and not file.endswith((".humaneval.json", ".mbpp.json")):
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
    he = (row.get("humaneval") or {}).get("score")
    mb = (row.get("mbpp") or {}).get("score")
    return (f"[{row['cell']}] gpu {row.get('gpu_label') or row.get('gpu')} HE {he} MBPP {mb} decode {row.get('decode_tps')} "
            f"he_b1 {row.get('he_b1_tps')} weights {row.get('model_loading_gib')} GiB kv {row.get('kv_cache_tokens')} tok"
            + (f" ERROR {row['error'][:200]}" if row.get("error") else ""))


@app.local_entrypoint()
def main(model: str = "hub:" + BASE_MODEL, name: str = "full", configs: str = "base", gpu: str = "H200",
         tokenizer: str = BASE_MODEL, smoke: bool = False, shards: int = 1, estimate_only: bool = False,
         collect: bool = False, build_slim_model: bool = False, util: float = GPU_UTILIZATION,
         quantize: str = "", quant_name: str = "", shell_command: str = ""):
    if shell_command:
        print(shell.remote(shell_command))
        return
    if collect:
        rows = collect_rows.remote(name)
        merged = merge_rows(os.path.join(LOCAL_DIR, f"efficiency_{name}.json"), rows)
        print(f"collected {len(rows)} rows, {len(merged)} in results/efficiency_{name}.json")
        for row in rows:
            print(summary_line(row))
        return
    if build_slim_model:
        with open(MANIFEST) as f:
            manifest = json.load(f)
        print(build_slim.remote(manifest))
        return
    if quantize:
        if quantize not in QUANT_SCHEMES or not quant_name:
            raise SystemExit("--quantize is w4a16 or fp8 and needs --quant-name")
        model_ref(model)
        runner = quantize_l4 if gpu == "L4" else quantize_h200
        print(runner.remote(model, quant_name, quantize, smoke))
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
        merge_rows(os.path.join(LOCAL_DIR, f"efficiency_{name}.json"), rows)
        for row in rows:
            print(summary_line(row))
