"""Serve a model with vLLM and measure speed and HumanEval together.

Why: skill removal keeps code quality at half the experts (results/README.md, Run 19), but the Hugging
Face MoE loop decodes at about 10 tokens per second, so no speed claim can come from it. This script
runs each model in vLLM on one H200 and reports, for one fixed set of 64 code prompts (the wsp.healdata
"code" pool, rendered with the chat template, greedy, 256 new tokens):

  decode_tps_b1, decode_tps_b32   decode tokens per second at batch 1 and batch 32
  prefill_tps                     prompt tokens per second, all prompts in one call, 1 new token
  ttft_ms_median, ttft_ms_p90     time to first token at batch 1 (wall clock of a 1-token request)
  gpu_peak_gib, gpu_after_load_gib  GPU memory used on the device, polled with nvidia-smi
  weights_gib                     size of the weight files read

A speed number always comes with an accuracy check: HumanEval (164 items) through the same vLLM engine,
with the prompt and answer extraction of wsp.skills_eval (user_text, render_prompt, grade_outputs; nothing
is copied), greedy, 512 new tokens, as in Run 19.

Decode speed is (tokens made by a 256-token run) over (its time minus the time of a 1-token run of the
same prompts), so prefill does not count in it. Every sequence runs exactly 256 tokens (ignore_eos).

vLLM keeps a KV-cache pool of gpu_memory_utilization (0.90) of the device, so gpu_peak_gib is mostly that
pool for every cell. Compare weights_gib and gpu_after_load_gib for the size of the model.

Cells (CELLS):
  full_30b       Qwen/Qwen3-30B-A3B-Instruct-2507 from the Hub
  slim_64        64 experts per layer by code saliency, no heal, read from the volume
  slim_32_heal   32 experts per layer by code saliency, 500-step heal, read from the volume
  coder_7b       Qwen/Qwen2.5-Coder-7B-Instruct (dense reference)
  coder_3b       Qwen/Qwen2.5-Coder-3B-Instruct (dense reference)

The slim cells read plain Hugging Face directories (wsp/slim.py save_slim, plain_hf true) under
SLIM_ROOT on the volume "wsp-models". Each Modal workspace has its own volume, so a directory exists only
in the workspace that saved it. The script checks the path first, on a CPU container, and writes a row
with a clear error for a cell whose path is missing instead of starting a GPU.

Run 19 (modal_round19.py) did not keep the slimmed models: disk_size saves each one to /tmp, measures it,
and deletes it. Only the ranking statistics persist (/vol/slim/round19). The slim cells therefore need a
saved directory from another script before they can run. See docs/serve.md.

Smoke (one cell, no HumanEval, 8 prompts, batch 1 and 8):

    modal run --detach modal_serve.py --smoke

Everything:

    modal run --detach modal_serve.py --profile-volume trisanth
    modal run --detach modal_serve.py --cells full_30b,coder_7b --profile-volume ztrisanth

--profile-volume is a label for the workspace you run in (trisanth, trisanthsrini, or ztrisanth). Modal
picks the workspace from the active profile; the label goes into the rows and the missing-path message.

Rows go to /vol/results/serve/<cell>.json on the volume (written by the GPU function, so a dead client
loses nothing) and to results/serve_<cell>.json here, after each cell.
"""

import json
import math
import os
import statistics
import subprocess
import threading
import time

import modal

app = modal.App("wsp-serve")
volume = modal.Volume.from_name("wsp-models", create_if_missing=True)

VLLM_VERSION = "0.32.0"

# The pip list of modal_round20.py (a script that ran), without its torch and transformers pins:
# vllm 0.32.0 pins torch==2.13.0 and transformers>=5.16.1,<5.20.0 and brings both.
image = (
    # vLLM compiles some kernels at start and needs nvcc and a CUDA home; a plain
    # debian image has neither ("Could not find nvcc", first smoke 2026-11-15).
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.12")
    .env({"CUDA_HOME": "/usr/local/cuda"})
    .pip_install(
        f"vllm=={VLLM_VERSION}", "accelerate", "huggingface_hub", "requests",
        "safetensors", "zstandard", "numpy", "datasets", "sentencepiece", "protobuf",
    )
    .env({"HF_HOME": "/vol/hf", "HF_HUB_DISABLE_XET": "1"})
    .add_local_python_source("wsp")
)
cpu_image = modal.Image.debian_slim(python_version="3.12")

RESULT_DIR = "/vol/results/serve"
SLIM_ROOT = "/vol/slim_models"
LOCAL_DIR = "results"

MOE_BASE = "Qwen/Qwen3-30B-A3B-Instruct-2507"
PROMPTS = 64
GEN_LEN = 256
BATCHES = (1, 32)
HUMANEVAL_TOKENS = 512
PROMPT_CHARS = 1500
MAX_MODEL_LEN = 4096
GPU_UTILIZATION = 0.90
WARMUP_PROMPTS = 4
SMOKE_CELL = "coder_3b"
SMOKE_PROMPTS = 8
SMOKE_BATCHES = (1, 8)

# kind "hub": model is a Hub id. kind "path": model is a directory under SLIM_ROOT, tokenizer is the Hub id of the base.
CELLS = {
    "full_30b": {"kind": "hub", "model": MOE_BASE},
    "slim_64": {"kind": "path", "dir": "round19/moe_saliency_64", "tokenizer": MOE_BASE},
    "slim_32_heal": {"kind": "path", "dir": "round19/moe_saliency_32_heal500", "tokenizer": MOE_BASE},
    "coder_7b": {"kind": "hub", "model": "Qwen/Qwen2.5-Coder-7B-Instruct"},
    "coder_3b": {"kind": "hub", "model": "Qwen/Qwen2.5-Coder-3B-Instruct"},
}

ROW_KEYS = ["cell", "model", "model_path", "profile_volume", "smoke", "vllm", "torch", "gpu", "prompts", "gen_len",
            "batches", "prompt_tokens", "load_seconds", "weights_gib", "gpu_after_load_gib", "gpu_peak_gib",
            "prefill_tps", "ttft_ms_median", "ttft_ms_p90", "decode_tps", "humaneval", "humaneval_correct",
            "humaneval_n", "humaneval_seconds", "humaneval_gen_tps", "seconds", "error", "skipped"]


def log(cell, message):
    print(f"[{cell}] {message}", flush=True)


def safe_reload():
    """Reload the volume to see files other containers wrote. Open files (HF logs) block it; then skip it."""
    try:
        volume.reload()
    except Exception as error:
        print("volume reload skipped:", repr(error)[:200], flush=True)


# ---------------------------------------------------------------- cells and paths


def select_cells(cells="", smoke=False):
    """Cell names to run. Smoke is always the one dense 3B cell. An unknown name raises ValueError."""
    if smoke:
        return [SMOKE_CELL]
    if cells in ("", "all"):
        return list(CELLS)
    names = [name.strip() for name in cells.split(",") if name.strip()]
    for name in names:
        if name not in CELLS:
            raise ValueError(f"unknown cell {name!r}; the cells are {', '.join(CELLS)}")
    return names


def run_settings(smoke):
    """(prompt count, batch sizes, score HumanEval)."""
    if smoke:
        return SMOKE_PROMPTS, SMOKE_BATCHES, False
    return PROMPTS, BATCHES, True


def row_name(cell, smoke):
    return f"smoke_{cell}" if smoke else cell


def model_path(spec, slim_root=SLIM_ROOT):
    """The path or Hub id vLLM loads."""
    if spec["kind"] == "hub":
        return spec["model"]
    return os.path.join(slim_root, spec["dir"])


def tokenizer_id(spec):
    return spec.get("tokenizer", spec.get("model"))


def check_path(cell, spec, slim_root=SLIM_ROOT, profile_volume="", listdir=os.listdir, isdir=os.path.isdir):
    """(ok, message). A Hub cell is always ok. A path cell needs a directory with config.json and safetensors."""
    if spec["kind"] == "hub":
        return True, ""
    path = model_path(spec, slim_root)
    where = f"volume wsp-models of workspace {profile_volume!r}" if profile_volume else "volume wsp-models"
    if not isdir(path):
        near = []
        parent = os.path.dirname(path)
        if isdir(parent):
            near = sorted(listdir(parent))
        found = f" The directory {parent} holds: {near}." if near else ""
        return False, (f"cell {cell}: {path} does not exist on the {where}. Each workspace has its own volume, "
                       f"so run in the workspace that saved the model, or pass --slim-root.{found} "
                       f"Round 19 did not keep its slimmed models; save one with wsp.slim.save_slim first.")
    names = listdir(path)
    if "config.json" not in names:
        return False, f"cell {cell}: {path} on the {where} has no config.json (files: {sorted(names)[:8]})"
    if not any(name.endswith(".safetensors") for name in names):
        return False, f"cell {cell}: {path} on the {where} has no .safetensors file (files: {sorted(names)[:8]})"
    return True, ""


# ---------------------------------------------------------------- prompts


def build_prompts(tokenizer, count=PROMPTS, pairs=None):
    """Prompt strings for the speed test: the first count users of the healdata "code" pool, chat-rendered.

    The pool order is fixed (seed 0) and does not depend on count, so a smaller count is a prefix of the
    full set. Each prompt is the user message with the generation prompt, from wsp.skills_eval.render_prompt.
    """
    from wsp import skills_eval

    if pairs is None:
        from wsp import healdata

        pairs = healdata.pairs("code", count, seed=0)
    if len(pairs) < count:
        raise ValueError(f"the code pool gave {len(pairs)} prompts, need {count}")
    prompts = []
    for user, _assistant in pairs[:count]:
        prompts.append(skills_eval.render_prompt(tokenizer, user[:PROMPT_CHARS]))
    return prompts


def encode_prompts(tokenizer, prompts):
    """Token id lists. Chat prompts already hold any special tokens, so none are added."""
    return [list(tokenizer(prompt, add_special_tokens=False)["input_ids"]) for prompt in prompts]


def humaneval_prompts(tokenizer, items=None):
    """(items, prompt strings) with the prompt format of wsp.skills_eval.evaluate_skill."""
    from wsp import skills_eval

    if items is None:
        items = skills_eval.humaneval_items()
    return items, [skills_eval.render_prompt(tokenizer, skills_eval.user_text("humaneval", item)) for item in items]


def grade_humaneval(items, outputs):
    """{"score", "correct", "n"} with wsp.skills_eval.grade_outputs, the grader of Run 19."""
    from wsp import skills_eval

    records = skills_eval.grade_outputs("humaneval", items, outputs)
    correct = sum(1 for record in records if record["correct"])
    n = len(records)
    return {"score": correct / n if n else 0.0, "correct": correct, "n": n}


# ---------------------------------------------------------------- measurement


def percentile(values, share):
    """Nearest-rank percentile of a list (share in 0..1)."""
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(share * len(ordered)) - 1))
    return ordered[index]


def measure_decode(generate, prompt_ids, batch, gen_len, clock=time.perf_counter):
    """Decode tokens per second at one batch size, and the 1-token times of each chunk.

    generate(list of token id lists, max_tokens) runs the engine to the end. For each chunk of `batch`
    prompts, run 1 new token and then gen_len new tokens. Decode time is the second time minus the first,
    which removes prefill. Returns (tokens per second, [1-token seconds per chunk]).
    """
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
    """Prompt tokens per second: all prompts in one call with 1 new token."""
    start = clock()
    generate(prompt_ids, 1)
    seconds = clock() - start
    if seconds <= 0:
        return None
    return sum(len(ids) for ids in prompt_ids) / seconds


def ttft_stats(seconds):
    """(median ms, p90 ms) of 1-token request times."""
    if not seconds:
        return None, None
    millis = [value * 1000.0 for value in seconds]
    return statistics.median(millis), percentile(millis, 0.9)


def gpu_used_gib():
    """Memory used on GPU 0 in GiB by nvidia-smi, or None."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=10).stdout
        return float(out.strip().splitlines()[0]) / 1024.0
    except Exception:
        return None


class GpuSampler:
    """Poll GPU memory in a thread and keep the peak. vLLM runs its engine in another process, so
    torch.cuda.max_memory_allocated of this process would read near zero."""

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


def weights_gib(path):
    """Size of the .safetensors files of a local directory in GiB."""
    total = 0
    for name in os.listdir(path):
        if name.endswith(".safetensors"):
            total += os.path.getsize(os.path.join(path, name))
    return total / 1024 ** 3


# ---------------------------------------------------------------- rows


def new_row(cell, spec, smoke, profile_volume="", slim_root=SLIM_ROOT):
    """A row with every key in ROW_KEYS (None where unmeasured)."""
    prompts, batches, _ = run_settings(smoke)
    row = {key: None for key in ROW_KEYS}
    row.update({"cell": row_name(cell, smoke), "model": spec.get("model") or spec.get("dir"),
                "model_path": model_path(spec, slim_root), "profile_volume": profile_volume, "smoke": smoke,
                "prompts": prompts, "gen_len": GEN_LEN, "batches": list(batches), "decode_tps": {},
                "error": None, "skipped": False})
    return row


def error_row(cell, spec, smoke, message, profile_volume="", slim_root=SLIM_ROOT, skipped=False):
    row = new_row(cell, spec, smoke, profile_volume, slim_root)
    row["error"] = message
    row["skipped"] = skipped
    return row


def write_row(row, directory):
    """Write one row to <directory>/serve_<cell>.json (or <cell>.json on the volume)."""
    os.makedirs(directory, exist_ok=True)
    name = row["cell"]
    prefix = "serve_" if directory == LOCAL_DIR else ""
    path = os.path.join(directory, f"{prefix}{name}.json")
    with open(path, "w") as f:
        json.dump(row, f, indent=2, default=str)
    return path


# ---------------------------------------------------------------- the GPU function


def resolve_local(spec, slim_root, cell):
    """Local directory of the weights, downloading a Hub model into HF_HOME on the volume."""
    if spec["kind"] == "path":
        return model_path(spec, slim_root)
    from huggingface_hub import snapshot_download

    log(cell, f"downloading {spec['model']} (weights and tokenizer files only)")
    return snapshot_download(spec["model"], allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "*.tiktoken"])


def serve_cell(cell, smoke=False, profile_volume="", slim_root=SLIM_ROOT):
    """Load one cell in vLLM, measure it, return the row. Raises on failure (the caller records it)."""
    import gc

    import torch
    import transformers
    import vllm
    from vllm import LLM, SamplingParams

    from wsp import healdata, skills_eval  # noqa: F401  (fail here, not after the model loads)

    spec = CELLS[cell]
    prompts_n, batches, do_humaneval = run_settings(smoke)
    row = new_row(cell, spec, smoke, profile_volume, slim_root)
    started = time.time()
    row["vllm"] = vllm.__version__
    row["torch"] = torch.__version__
    row["gpu"] = torch.cuda.get_device_name(0)
    log(row["cell"], f"vllm {vllm.__version__} torch {torch.__version__} transformers {transformers.__version__} "
                     f"gpu {row['gpu']}")

    local = resolve_local(spec, slim_root, row["cell"])
    row["weights_gib"] = round(weights_gib(local), 2)
    tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_id(spec))
    prompts = build_prompts(tokenizer, prompts_n)
    prompt_ids = encode_prompts(tokenizer, prompts)
    row["prompt_tokens"] = sum(len(ids) for ids in prompt_ids)
    log(row["cell"], f"{len(prompts)} prompts, {row['prompt_tokens']} prompt tokens")

    sampler = GpuSampler()
    with sampler:
        load_start = time.time()
        llm = LLM(model=local, tokenizer=tokenizer_id(spec), dtype="bfloat16", seed=0,
                  gpu_memory_utilization=GPU_UTILIZATION, max_model_len=MAX_MODEL_LEN,
                  enable_prefix_caching=False, max_num_seqs=max(batches + (32,)))
        row["load_seconds"] = round(time.time() - load_start, 1)
        row["gpu_after_load_gib"] = round(gpu_used_gib() or 0.0, 2)
        log(row["cell"], f"loaded in {row['load_seconds']} s, {row['gpu_after_load_gib']} GiB used on the GPU")

        def generate(id_lists, max_tokens):
            params = SamplingParams(temperature=0.0, max_tokens=max_tokens, ignore_eos=True)
            llm.generate([{"prompt_token_ids": ids} for ids in id_lists], params, use_tqdm=False)

        generate(prompt_ids[:WARMUP_PROMPTS], 16)

        row["prefill_tps"] = measure_prefill(generate, prompt_ids)
        log(row["cell"], f"prefill {row['prefill_tps']} tokens/s")
        for batch in batches:
            tps, first = measure_decode(generate, prompt_ids, batch, GEN_LEN)
            row["decode_tps"][str(batch)] = tps
            log(row["cell"], f"decode batch {batch}: {tps} tokens/s")
            if batch == 1:
                row["ttft_ms_median"], row["ttft_ms_p90"] = ttft_stats(first)
                log(row["cell"], f"ttft median {row['ttft_ms_median']} ms, p90 {row['ttft_ms_p90']} ms")

        if do_humaneval:
            items, he_prompts = humaneval_prompts(tokenizer)
            params = SamplingParams(temperature=0.0, max_tokens=HUMANEVAL_TOKENS)
            start = time.time()
            outputs = llm.generate(he_prompts, params, use_tqdm=False)
            row["humaneval_seconds"] = round(time.time() - start, 1)
            made = sum(len(out.outputs[0].token_ids) for out in outputs)
            row["humaneval_gen_tps"] = made / max(row["humaneval_seconds"], 1e-9)
            texts = [out.outputs[0].text for out in outputs]
            graded = grade_humaneval(items, texts)
            row["humaneval"] = graded["score"]
            row["humaneval_correct"] = graded["correct"]
            row["humaneval_n"] = graded["n"]
            log(row["cell"], f"HumanEval {graded['correct']}/{graded['n']} = {graded['score']:.3f}")
    row["gpu_peak_gib"] = None if sampler.peak is None else round(sampler.peak, 2)
    log(row["cell"], f"peak GPU memory {row['gpu_peak_gib']} GiB")
    row["seconds"] = round(time.time() - started, 1)
    del llm
    gc.collect()
    return row


@app.function(image=cpu_image, volumes={"/vol": volume}, timeout=600)
def check_paths(cells, slim_root, profile_volume):
    """{cell: message} for every path cell whose directory is missing or unreadable."""
    safe_reload()
    problems = {}
    for cell in cells:
        ok, message = check_path(cell, CELLS[cell], slim_root, profile_volume)
        if not ok:
            problems[cell] = message
    return problems


@app.function(image=image, volumes={"/vol": volume}, gpu="H200", memory=262144, timeout=3 * 3600, single_use_containers=True)
def run_cell(cell, smoke=False, profile_volume="", slim_root=SLIM_ROOT):
    safe_reload()
    spec = CELLS[cell]
    try:
        row = serve_cell(cell, smoke, profile_volume, slim_root)
    except Exception as error:
        log(row_name(cell, smoke), "failed: " + repr(error)[:400])
        row = error_row(cell, spec, smoke, repr(error)[:500], profile_volume, slim_root)
    write_row(row, RESULT_DIR)
    try:
        volume.commit()
    except Exception as error:
        print("volume commit failed:", repr(error)[:200], flush=True)
    return row


@app.local_entrypoint()
def main(cells: str = "all", smoke: bool = False, profile_volume: str = "", slim_root: str = SLIM_ROOT):
    names = select_cells(cells, smoke)
    print(f"cells: {names}" + (" (smoke)" if smoke else ""))
    problems = check_paths.remote(names, slim_root, profile_volume)
    for cell in names:
        if cell in problems:
            print(problems[cell])
            row = error_row(cell, CELLS[cell], smoke, problems[cell], profile_volume, slim_root, skipped=True)
            print("wrote", write_row(row, LOCAL_DIR))
    for cell in names:
        if cell in problems:
            continue
        row = run_cell.remote(cell, smoke, profile_volume, slim_root)
        print("wrote", write_row(row, LOCAL_DIR))
        if row.get("error"):
            print(f"[{row['cell']}] failed: {row['error']}")
            continue
        print(f"[{row['cell']}] " + json.dumps({key: row[key] for key in
              ["decode_tps", "prefill_tps", "ttft_ms_median", "gpu_peak_gib", "humaneval"]}))
