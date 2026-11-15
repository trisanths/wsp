"""Moonshot: remove experts from Qwen3-235B-A22B-Instruct-2507 until it fits one GPU, and score what is left.

Request: "Give me Qwen3-235B's coding ability in a model that fits on ONE GPU."

Run 19 (results/README.md) kept 64 of 128 experts per layer of Qwen3-30B-A3B by code saliency, with no heal,
and lost 3.7 HumanEval points. This script runs the same idea on the 235B model, whose config (checked on the
Hub, 2026-11-15) is: 94 layers, every layer MoE, 128 routed experts, top-k 8, hidden 4096, expert width 1536,
64 query heads, 4 KV heads, head dim 128, vocab 151936, untied head, 235,093,634,560 stored parameters
(470,187,269,120 bytes of bf16, 118 safetensors shards, per-expert tensor names). The model is the non-thinking
2507 instruct release.

Stages (each writes rows to the volume, and the local entrypoint copies them to results/):

  sizes      CPU, no Modal. Exact stored and per-token parameters for each keep N, from the config.
  download   CPU container. The 118 shards into HF_HOME on the volume (470 GB), resumable.
  stats      GPU container, H200:4, torch 2.7.1 + transformers 4.57.1 (the round 19 image). Loads the whole model
             with device_map="auto" and runs wsp.slim.collect_moe_stats on 256 code sequences of 256 tokens, in
             chunks, saving partial sums after every chunk. --smoke loads only the first 4 layers on H200:2 and
             runs 8 sequences, to test the loader, the hooks and the split across two devices.
  build      CPU container. Safetensors surgery: for each shard, drop the experts that are not kept and renumber
             the rest, slice the router rows, write plain Hugging Face directories keep_64, keep_32, keep_16.
             No model is instantiated, so no GPU is needed and nothing depends on the transformers version.
  score      GPU containers with vLLM 0.32 (the modal_serve.py image): full 235B on H200:8, keep 64 on H200:4,
             keep 32 on H200:2, keep 16 on H200:1. HumanEval 164 and MBPP sanitized test 257 (generation, the
             wsp.skills_eval prompts and graders, greedy, 512 new tokens), MMLU 1000 (the first-token choice among
             A to D, as wsp.skills_eval.score_choices), and decode speed at batch 1 and 32.

Ranking data (never a scored item): 192 chat texts from the wsp.healdata "code" pool (Magicoder, CodeAlpaca,
python_code_instructions, self-oss-instruct, APPS introductory train, code_contests), and 64 chat texts from
MBPP full train and prompt splits (task ids 1 to 10 and 601 to 974; the scored sanitized test set uses ids from
11 to 510). Texts that contain a HumanEval or MBPP test string are dropped (wsp.slim.drop_overlap).

Cost: stats and eval rows record GPU seconds times GPU count times RATE_PER_GPU_HOUR (4.5 dollars), so the sum
over rows is the spend on this track. Stop at about 150 dollars.

    modal run modal_moonshot.py --stage sizes
    modal run --detach modal_moonshot.py --stage download
    modal run modal_moonshot.py --stage stats --smoke
    modal run --detach modal_moonshot.py --stage stats
    modal run --detach modal_moonshot.py --stage build
    modal run --detach modal_moonshot.py --stage score --cells keep_16,keep_32,keep_64,full_235b

Add --estimate-only to any stage to print what it would do and exit. A stage with nothing to do fails.
"""

import json
import math
import os
import re
import time

import modal

CODE_VERSION = "v1"
MODEL = "Qwen/Qwen3-235B-A22B-Instruct-2507"
REFERENCE_30B = "Qwen/Qwen3-30B-A3B-Instruct-2507"
KEEPS = (64, 48, 32, 24, 16)
BUILD_KEEPS = (64, 32, 16)
RATE_PER_GPU_HOUR = 4.5
BUDGET_DOLLARS = 150.0

ROOT = "/vol/moonshot"
STATS_DIR = f"{ROOT}/stats"
MODEL_DIR = f"{ROOT}/models/{CODE_VERSION}"
ROW_DIR = f"{ROOT}/rows"
LOCAL_DIR = "results"

RANK_TEXTS = 256
RANK_LENGTH = 256
RANK_FROM_POOL = 192
RANK_FROM_MBPP = 64
STATS_CHUNK = 32
SMOKE_LAYERS = 4
SMOKE_TEXTS = 8

HUMANEVAL_TOKENS = 512
MBPP_TOKENS = 512
MMLU_COUNT = 1000
MAX_MODEL_LEN = 4096
GPU_UTILIZATION = 0.90
SPEED_PROMPTS = 32
GEN_LEN = 256
SPEED_BATCHES = (1, 32)

# cell: (kind, directory name or Hub id, tensor parallel size)
CELLS = {
    "keep_16": ("slim", "keep_16", 1),
    "keep_24": ("slim", "keep_24", 1),
    "keep_32": ("slim", "keep_32", 2),
    "keep_48": ("slim", "keep_48", 2),
    "keep_48_tp4": ("slim", "keep_48", 4),
    "keep_64": ("slim", "keep_64", 4),
    "full_235b": ("hub", MODEL, 8),
}

app = modal.App("wsp-moonshot")
volume = modal.Volume.from_name("wsp-models", create_if_missing=True)

# The round 19 image: transformers 4.57.1 holds Qwen3-MoE experts as a ModuleList, which wsp.slim needs.
stats_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch==2.7.1", "transformers==4.57.1", "accelerate", "huggingface_hub", "requests",
        "safetensors", "zstandard", "numpy", "datasets", "sentencepiece", "protobuf",
    )
    .env({"HF_HOME": "/vol/hf", "HF_HUB_DISABLE_XET": "1"})
    .add_local_python_source("wsp")
)

# The modal_serve.py image (it ran: full_30b on 2026-11-15), plus this script's helper module.
VLLM_VERSION = "0.32.0"
vllm_image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.12")
    .env({"CUDA_HOME": "/usr/local/cuda"})
    .pip_install(
        f"vllm=={VLLM_VERSION}", "accelerate", "huggingface_hub", "requests",
        "safetensors", "zstandard", "numpy", "datasets", "sentencepiece", "protobuf",
    )
    .env({"HF_HOME": "/vol/hf", "HF_HUB_DISABLE_XET": "1"})
    .add_local_python_source("wsp")
    .add_local_python_source("modal_serve")
)


def log(tag, message):
    print(f"[{tag}] {message}", flush=True)


def safe_reload():
    """Reload the volume to see files other containers wrote. Open files (HF logs) block it; then skip it."""
    try:
        volume.reload()
    except Exception as error:
        print("volume reload skipped:", repr(error)[:200], flush=True)


def safe_commit():
    try:
        volume.commit()
    except Exception as error:
        print("volume commit failed:", repr(error)[:200], flush=True)


# ---------------------------------------------------------------- sizes (CPU, from the config)


def moe_sizes(config, keep):
    """Exact parameter counts of the model with `keep` experts per layer, from a config dict.

    Returns stored (every tensor once), per_token (what one token runs: attention, the router, top-k experts,
    the head; the input embedding is a lookup and does not count, as in wsp.slim.count_params), and the bf16
    bytes of the stored weights.
    """
    hidden = config["hidden_size"]
    width = config["moe_intermediate_size"]
    layers = config["num_hidden_layers"]
    heads = config["num_attention_heads"]
    kv_heads = config["num_key_value_heads"]
    head_dim = config["head_dim"]
    vocab = config["vocab_size"]
    total_experts = config["num_experts"]
    top_k = config["num_experts_per_tok"]
    if keep < 1 or keep > total_experts:
        raise ValueError(f"cannot keep {keep} of {total_experts} experts")
    expert = 3 * hidden * width
    attention = hidden * heads * head_dim * 2 + hidden * kv_heads * head_dim * 2 + 2 * head_dim
    norms = 2 * hidden
    router = hidden * keep
    embed = vocab * hidden
    head = 0 if config.get("tie_word_embeddings") else vocab * hidden
    stored = embed + head + hidden + layers * (attention + norms + router + keep * expert)
    per_token = head + hidden + layers * (attention + norms + router + min(top_k, keep) * expert)
    if config.get("tie_word_embeddings"):
        per_token += embed
    return {"keep": keep, "stored": stored, "per_token": per_token, "bf16_bytes": 2 * stored,
            "expert_params": expert, "non_expert_params": stored - layers * keep * expert}


def size_table(config, keeps=KEEPS + (128,)):
    rows = [moe_sizes(config, keep) for keep in sorted(set(keeps), reverse=True)]
    for row in rows:
        row["bf16_gb"] = row["bf16_bytes"] / 1e9
        row["fp8_gb"] = row["stored"] / 1e9
        row["fits_one_h200_bf16"] = row["bf16_bytes"] < 141e9 * 0.80
        row["fits_one_80gb_fp8"] = row["stored"] < 80e9 * 0.80
    return rows


def format_size_table(rows):
    lines = ["keep  stored B   per-token B  bf16 GB  fp8 GB  one H200 bf16 (weights < 80% of 141 GB)  one 80 GB fp8"]
    for row in rows:
        lines.append(f"{row['keep']:>4}  {row['stored'] / 1e9:>8.2f}  {row['per_token'] / 1e9:>11.2f}  "
                     f"{row['bf16_gb']:>7.1f}  {row['fp8_gb']:>6.1f}  {str(row['fits_one_h200_bf16']):>38}  "
                     f"{str(row['fits_one_80gb_fp8']):>12}")
    return "\n".join(lines)


# ---------------------------------------------------------------- checkpoint surgery (no model object)

EXPERT_RE = re.compile(r"^model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.weight$")
ROUTER_RE = re.compile(r"^model\.layers\.(\d+)\.mlp\.gate\.weight$")


def keep_maps(kept):
    """{layer: {old expert id: new id}} from {layer: sorted ids}. New ids follow the sorted order."""
    out = {}
    for layer, ids in kept.items():
        ids = sorted(int(i) for i in ids)
        if len(set(ids)) != len(ids):
            raise ValueError(f"layer {layer}: repeated expert ids")
        out[int(layer)] = {old: new for new, old in enumerate(ids)}
    return out


def surgery_name(name, maps):
    """What one tensor becomes under one keep map: ("copy", name), ("drop", None),
    ("rename", new name) or ("router", name, sorted kept ids)."""
    found = EXPERT_RE.match(name)
    if found:
        layer, expert, part = int(found.group(1)), int(found.group(2)), found.group(3)
        if layer not in maps:
            raise ValueError(f"no keep entry for layer {layer}")
        if expert not in maps[layer]:
            return ("drop", None)
        return ("rename", f"model.layers.{layer}.mlp.experts.{maps[layer][expert]}.{part}.weight")
    found = ROUTER_RE.match(name)
    if found:
        layer = int(found.group(1))
        if layer not in maps:
            raise ValueError(f"no keep entry for layer {layer}")
        return ("router", name, sorted(maps[layer]))
    return ("copy", name)


def slim_shard(path, maps_by_keep):
    """Read one safetensors shard and return {keep: {tensor name: tensor}} for each keep map set.

    Only tensors that at least one keep set uses are read.
    """
    from safetensors import safe_open

    out = {keep: {} for keep in maps_by_keep}
    with safe_open(path, framework="pt", device="cpu") as handle:
        for name in handle.keys():
            plans = {keep: surgery_name(name, maps) for keep, maps in maps_by_keep.items()}
            if all(plan[0] == "drop" for plan in plans.values()):
                continue
            tensor = handle.get_tensor(name)
            for keep, plan in plans.items():
                if plan[0] == "drop":
                    continue
                if plan[0] == "copy":
                    out[keep][name] = tensor
                elif plan[0] == "rename":
                    out[keep][plan[1]] = tensor
                else:
                    import torch

                    index = torch.tensor(plan[2], dtype=torch.long)
                    out[keep][name] = tensor.index_select(0, index).contiguous()
    return out


def slim_config(config, keep):
    """The config dict of the model with `keep` experts per layer (top-k becomes min(top-k, keep))."""
    new = dict(config)
    new["num_experts"] = keep
    new["num_experts_per_tok"] = min(config["num_experts_per_tok"], keep)
    return new


def shard_done(directory, shard):
    return os.path.exists(os.path.join(directory, shard))


def slim_checkpoint(src, dst_root, kept_by_keep, workers=4, commit=None, copy_files=None):
    """Write one plain Hugging Face directory per keep under dst_root (keep_<N>), from the checkpoint at src.

    kept_by_keep: {keep: {layer: sorted ids}}. Shards that already exist in the destination are skipped, so a
    rerun resumes. Output shards are written under a temporary name and renamed. Returns
    {keep: {"dir", "tensors", "params", "bytes"}}.
    """
    from concurrent.futures import ThreadPoolExecutor

    from safetensors.torch import save_file

    with open(os.path.join(src, "model.safetensors.index.json")) as f:
        index = json.load(f)
    with open(os.path.join(src, "config.json")) as f:
        config = json.load(f)
    shards = sorted(set(index["weight_map"].values()))
    maps_by_keep = {keep: keep_maps(kept) for keep, kept in kept_by_keep.items()}
    dirs = {keep: os.path.join(dst_root, f"keep_{keep}") for keep in kept_by_keep}
    for directory in dirs.values():
        os.makedirs(directory, exist_ok=True)

    def work(shard):
        todo = {keep: maps for keep, maps in maps_by_keep.items() if not shard_done(dirs[keep], shard)}
        names = {keep: {} for keep in maps_by_keep}
        if todo:
            tensors = slim_shard(os.path.join(src, shard), todo)
            for keep, state in tensors.items():
                if state:
                    temp = os.path.join(dirs[keep], shard + ".tmp")
                    save_file(state, temp, metadata={"format": "pt"})
                    os.replace(temp, os.path.join(dirs[keep], shard))
        # Read the written shard headers back, so the index comes from the files and a resumed run agrees.
        from safetensors import safe_open

        info = {}
        for keep in maps_by_keep:
            path = os.path.join(dirs[keep], shard)
            if os.path.exists(path):
                with safe_open(path, framework="pt", device="cpu") as handle:
                    for name in handle.keys():
                        sl = handle.get_slice(name)
                        count = 1
                        for dim in sl.get_shape():
                            count *= dim
                        names[keep][name] = count
        info[shard] = names
        return info

    results = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(work, shard) for shard in shards]
        for position, future in enumerate(futures):
            results.update(future.result())
            if commit is not None and (position + 1) % 12 == 0:
                commit()
    summary = {}
    for keep in kept_by_keep:
        weight_map = {}
        params = 0
        for shard in shards:
            for name, count in results[shard][keep].items():
                weight_map[name] = shard
                params += count
        total_bytes = 2 * params
        with open(os.path.join(dirs[keep], "model.safetensors.index.json"), "w") as f:
            json.dump({"metadata": {"total_size": total_bytes}, "weight_map": weight_map}, f)
        with open(os.path.join(dirs[keep], "config.json"), "w") as f:
            json.dump(slim_config(config, keep), f, indent=2)
        for name in (copy_files or []):
            source = os.path.join(src, name)
            if os.path.exists(source):
                with open(source, "rb") as fin, open(os.path.join(dirs[keep], name), "wb") as fout:
                    fout.write(fin.read())
        disk = 0
        for name in os.listdir(dirs[keep]):
            disk += os.path.getsize(os.path.join(dirs[keep], name))
        summary[keep] = {"dir": dirs[keep], "tensors": len(weight_map), "params": params, "bytes": disk}
    return summary


# ---------------------------------------------------------------- ranking data


def mbpp_train_texts(tokenizer, ev, count, seed=0):
    """Chat texts from MBPP full train and prompt splits, in the format of the scored prompts.

    The scored set is the sanitized test split (ids from 11 to 510). Ids in it are dropped here as well.
    """
    import random

    from datasets import load_dataset

    tested = set()
    for item in ev.mbpp_items():
        tested.add(int(item["id"].split(":")[1]))
    rows = []
    for split in ("train", "prompt"):
        data = load_dataset("google-research-datasets/mbpp", "full", split=split)
        for index in range(len(data)):
            row = data[index]
            if int(row["task_id"]) in tested:
                continue
            rows.append(row)
    random.Random(seed).shuffle(rows)
    texts = []
    for row in rows:
        user = (row["text"] + "\nYour code should pass these tests:\n\n" + "\n".join(row["test_list"])
                + "\n\n" + ev.CODE_INSTRUCTION)
        assistant = "```python\n" + row["code"].replace("\r\n", "\n").strip() + "\n```"
        messages = [{"role": "user", "content": user}, {"role": "assistant", "content": assistant}]
        texts.append(tokenizer.apply_chat_template(messages, tokenize=False))
    return texts


def overlap_keys(ev):
    """Strings from the scored items. A ranking text that holds one is dropped (as in modal_round19.py)."""
    keys = []
    for item in ev.humaneval_items():
        found = re.search(r'"""\s*(.+)', item["prompt"])
        if found:
            keys.append(found.group(1).strip())
        keys.append(item["prompt"].strip().splitlines()[-1] if item["prompt"].strip() else "")
    for item in ev.mbpp_items():
        keys.append(item["text"])
        for test in item["test_list"]:
            keys.append(test)
    return keys


def text_hash(texts):
    import hashlib

    digest = hashlib.sha256()
    for text in texts:
        digest.update(text.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()[:12]


def ranking_texts(tokenizer, count=RANK_TEXTS):
    """(texts, info): RANK_FROM_POOL healdata code texts and RANK_FROM_MBPP MBPP train texts, scaled to count."""
    from wsp import healdata, skills_eval as ev, slim

    pool_count = round(count * RANK_FROM_POOL / RANK_TEXTS)
    mbpp_count = count - pool_count
    ids = [source[0] for source in healdata.SOURCES["code"]]
    healdata.check_sources(ids)
    keys = overlap_keys(ev)
    pool = healdata.pool("code", pool_count * 3, tokenizer=tokenizer, seed=0)
    clean, dropped_pool = slim.drop_overlap(pool, keys)
    mbpp = mbpp_train_texts(tokenizer, ev, mbpp_count * 3)
    clean_mbpp, dropped_mbpp = slim.drop_overlap(mbpp, keys)
    if len(clean) < pool_count or len(clean_mbpp) < mbpp_count:
        raise ValueError(f"short ranking data: pool {len(clean)}/{pool_count}, mbpp {len(clean_mbpp)}/{mbpp_count}")
    texts = clean[:pool_count] + clean_mbpp[:mbpp_count]
    info = {"pool_sources": ids, "pool_texts": pool_count, "mbpp_texts": mbpp_count,
            "dropped_overlap_pool": dropped_pool, "dropped_overlap_mbpp": dropped_mbpp,
            "hash": text_hash(texts), "length": RANK_LENGTH}
    return texts, info


# ---------------------------------------------------------------- statistics


def add_stats(a, b):
    """Sum two collect_moe_stats results (same model, different texts)."""
    if a is None:
        return b
    if a["num_experts"] != b["num_experts"] or set(a["layers"]) != set(b["layers"]):
        raise ValueError("stats do not match")
    layers = {}
    for layer in a["layers"]:
        x, y = a["layers"][layer], b["layers"][layer]
        layers[layer] = {"count": x["count"] + y["count"], "saliency": x["saliency"] + y["saliency"],
                         "tokens": x["tokens"] + y["tokens"]}
    return {"layers": layers, "top_k": a["top_k"], "num_experts": a["num_experts"]}


def stats_path(key, smoke=False):
    return f"{STATS_DIR}/{CODE_VERSION}_{'smoke_' if smoke else ''}{key}.pt"


def keep_sets(stats, keeps=BUILD_KEEPS):
    """{keep: {layer: sorted ids}} by saliency, and the share of skill routing and of saliency each takes."""
    from wsp import slim

    order = slim.rank_experts("saliency", stats)
    chosen = {}
    shares = {}
    for keep in keeps:
        chosen[keep] = slim.choose_keep(order, stats, "saliency", keep=keep)
        route = slim.kept_share(stats, chosen[keep])
        salient = {}
        for layer, ids in chosen[keep].items():
            values = stats["layers"][layer]["saliency"].tolist()
            whole = sum(values)
            salient[layer] = sum(values[i] for i in ids) / whole if whole > 0 else 0.0
        shares[keep] = {"routing_mean": sum(route.values()) / len(route),
                        "routing_min": min(route.values()),
                        "saliency_mean": sum(salient.values()) / len(salient),
                        "saliency_min": min(salient.values())}
    # Nested: every smaller keep is a subset of the larger one.
    return chosen, shares


# ---------------------------------------------------------------- rows


def write_row(row, directory, name=None):
    os.makedirs(directory, exist_ok=True)
    name = name or row["name"]
    path = os.path.join(directory, f"{name}.json")
    with open(path, "w") as f:
        json.dump(row, f, indent=2, default=str)
    return path


def dollars(seconds, gpus):
    return round(seconds * gpus * RATE_PER_GPU_HOUR / 3600.0, 2)


def spend(rows):
    return round(sum(row.get("dollars") or 0.0 for row in rows), 2)


# ---------------------------------------------------------------- stage: download


@app.function(image=stats_image, volumes={"/vol": volume}, cpu=8, memory=32768, timeout=6 * 3600)
def download(model: str = MODEL):
    from concurrent.futures import ThreadPoolExecutor

    from huggingface_hub import HfApi, hf_hub_download

    safe_reload()
    start = time.time()
    files = [s.rfilename for s in HfApi().model_info(model).siblings]
    wanted = [f for f in files if f.endswith((".safetensors", ".json", ".txt")) and f != "config_1m.json"]
    log("download", f"{len(wanted)} files of {model}")
    done = 0

    def get(name):
        return hf_hub_download(model, name)

    with ThreadPoolExecutor(max_workers=12) as pool:
        for position, path in enumerate(pool.map(get, wanted)):
            done += 1
            if done % 10 == 0:
                log("download", f"{done}/{len(wanted)} files, {round(time.time() - start)} s")
                safe_commit()
    safe_commit()
    snap = os.path.dirname(path)
    row = {"name": "download", "model": model, "files": done, "snapshot": snap, "seconds": round(time.time() - start, 1),
           "dollars": 0.0}
    write_row(row, ROW_DIR)
    safe_commit()
    return row


def snapshot_dir(model=MODEL):
    from huggingface_hub import snapshot_download

    return snapshot_download(model, allow_patterns=["*.json", "*.txt", "*.safetensors"], local_files_only=True)


# ---------------------------------------------------------------- stage: stats


def run_stats(smoke=False, layers=0, texts_count=RANK_TEXTS, gpus=4):
    import torch
    import transformers

    from wsp import slim

    safe_reload()
    start = time.time()
    tag = "stats_smoke" if smoke else "stats"
    path = snapshot_dir()
    tokenizer = transformers.AutoTokenizer.from_pretrained(path)
    texts, info = ranking_texts(tokenizer, texts_count)
    key = f"{texts_count}x{RANK_LENGTH}_{info['hash']}"
    final = stats_path(key + (f"_L{layers}" if layers else ""), smoke)
    log(tag, f"{len(texts)} ranking texts, hash {info['hash']}, dropped {info['dropped_overlap_pool']} pool and "
             f"{info['dropped_overlap_mbpp']} mbpp texts")
    log(tag, "first ranking text: " + texts[0][:200].replace("\n", " | "))
    if os.path.exists(final):
        log(tag, f"{final} exists; nothing to do")
        return {"name": tag, "cached": True, "path": final}
    config = transformers.AutoConfig.from_pretrained(path)
    if layers:
        config.num_hidden_layers = layers
    memory = {i: "125GiB" for i in range(torch.cuda.device_count())}
    log(tag, f"{torch.cuda.device_count()} GPUs, transformers {transformers.__version__}, layers {config.num_hidden_layers}")
    load_start = time.time()
    model = transformers.AutoModelForCausalLM.from_pretrained(
        path, config=config, dtype=torch.bfloat16, device_map="auto", max_memory=memory).eval()
    load_seconds = round(time.time() - load_start, 1)
    spread = {}
    for name, device in getattr(model, "hf_device_map", {}).items():
        spread[str(device)] = spread.get(str(device), 0) + 1
    log(tag, f"loaded in {load_seconds} s; modules per device {spread}")
    total, run = slim.count_params(model)
    log(tag, f"params stored {total}, per token {run}")
    stats = None
    chunk_seconds = []
    for first in range(0, len(texts), STATS_CHUNK):
        part_path = final.replace(".pt", f"_part{first // STATS_CHUNK}.pt")
        if os.path.exists(part_path):
            part = slim.load_stats(part_path)
        else:
            began = time.time()
            part = slim.collect_moe_stats(model, tokenizer, texts[first:first + STATS_CHUNK], RANK_LENGTH)
            os.makedirs(STATS_DIR, exist_ok=True)
            slim.save_stats(part, part_path)
            safe_commit()
            chunk_seconds.append(round(time.time() - began, 1))
            tokens = part["layers"][min(part["layers"])]["tokens"]
            log(tag, f"chunk {first // STATS_CHUNK}: {tokens} tokens in {chunk_seconds[-1]} s")
        stats = add_stats(stats, part)
    slim.save_stats(stats, final)
    safe_commit()
    tokens = [entry["tokens"] for entry in stats["layers"].values()]
    unused = {layer: int((entry["count"] == 0).sum()) for layer, entry in stats["layers"].items()}
    chosen, shares = keep_sets(stats, tuple(k for k in BUILD_KEEPS if k <= stats["num_experts"]))
    seconds = time.time() - start
    row = {"name": tag, "cell": tag, "model": MODEL, "layers": len(stats["layers"]), "texts": len(texts),
           "tokens_per_layer": [min(tokens), max(tokens)], "never_routed_experts_total": sum(unused.values()),
           "stats_path": final, "load_seconds": load_seconds, "chunk_seconds": chunk_seconds,
           "params_stored": total, "params_per_token": run, "shares": shares, "data": info, "gpus": gpus,
           "seconds": round(seconds, 1), "dollars": dollars(seconds, gpus), "code_version": CODE_VERSION}
    write_row(row, ROW_DIR)
    safe_commit()
    log(tag, "row " + json.dumps({k: row[k] for k in ("layers", "texts", "tokens_per_layer", "shares", "dollars")}))
    return row


@app.function(image=stats_image, volumes={"/vol": volume}, gpu="H200:4", memory=262144, timeout=6 * 3600)
def stats_full():
    return run_stats(smoke=False, gpus=4)


@app.function(image=stats_image, volumes={"/vol": volume}, gpu="H200:2", memory=131072, timeout=3600)
def stats_smoke():
    return run_stats(smoke=True, layers=SMOKE_LAYERS, texts_count=SMOKE_TEXTS, gpus=2)


# ---------------------------------------------------------------- stage: build


@app.function(image=stats_image, volumes={"/vol": volume}, cpu=8, memory=65536, timeout=6 * 3600)
def build(keeps: str = ""):
    from wsp import slim

    safe_reload()
    start = time.time()
    wanted = tuple(int(k) for k in keeps.split(",")) if keeps else BUILD_KEEPS
    candidates = sorted(p for p in os.listdir(STATS_DIR)
                        if p.startswith(f"{CODE_VERSION}_{RANK_TEXTS}x") and "_part" not in p and "_L" not in p)
    if not candidates:
        raise RuntimeError(f"no full stats file for code version {CODE_VERSION} in {STATS_DIR}")
    stats = slim.load_stats(os.path.join(STATS_DIR, candidates[-1]))
    chosen, shares = keep_sets(stats, wanted)
    src = snapshot_dir()
    log("build", f"stats {candidates[-1]}, keeps {wanted}, source {src}")
    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(os.path.join(MODEL_DIR, "kept_" + "_".join(str(k) for k in sorted(chosen)) + ".json"), "w") as f:
        json.dump({str(k): {str(l): ids for l, ids in v.items()} for k, v in chosen.items()}, f)
    extra = ["generation_config.json", "tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt"]
    summary = slim_checkpoint(src, MODEL_DIR, chosen, workers=4, commit=safe_commit, copy_files=extra)
    safe_commit()
    config = json.load(open(os.path.join(src, "config.json")))
    rows = []
    for keep, entry in summary.items():
        expected = moe_sizes(config, keep)
        row = {"name": f"build_keep_{keep}", "keep": keep, "dir": entry["dir"], "params": entry["params"],
               "expected_params": expected["stored"], "match": entry["params"] == expected["stored"],
               "disk_bytes": entry["bytes"], "shares": shares[keep], "stats_file": candidates[-1],
               "code_version": CODE_VERSION, "seconds": round(time.time() - start, 1), "dollars": 0.0}
        write_row(row, ROW_DIR)
        rows.append(row)
        log("build", f"keep {keep}: {entry['params']} params ({'matches' if row['match'] else 'DIFFERS FROM'} "
                     f"{expected['stored']}), {entry['bytes']} bytes on disk")
    safe_commit()
    if not all(row["match"] for row in rows):
        raise RuntimeError("a built checkpoint has a parameter count that differs from the config formula")
    return rows


# ---------------------------------------------------------------- stage: eval


def parse_load_gib(text):
    """vLLM's own 'Model loading took X GiB' line (the weight memory per GPU), or None."""
    found = re.findall(r"Model loading took ([\d.]+) ?GiB", text)
    return float(found[-1]) if found else None


def mmlu_prompts(items):
    return [item["prompt"] for item in items]


def eval_cell(cell, smoke=False, gpus=1):
    import gc
    import tempfile

    import torch
    import transformers
    import vllm
    from vllm import LLM, SamplingParams

    import modal_serve as serve
    from wsp import skills_eval as ev

    safe_reload()
    started = time.time()
    kind, target, tp = CELLS[cell]
    local = snapshot_dir(target) if kind == "hub" else os.path.join(MODEL_DIR, target)
    if not os.path.exists(os.path.join(local, "config.json")):
        raise RuntimeError(f"{local} has no config.json")
    tokenizer = transformers.AutoTokenizer.from_pretrained(local if kind == "slim" else local)
    name = f"smoke_{cell}" if smoke else cell
    row = {"name": "eval_" + name, "cell": name, "kind": kind, "tp": tp, "vllm": vllm.__version__,
           "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0), "model_dir": local,
           "code_version": CODE_VERSION}
    row["weights_bytes_on_disk"] = sum(os.path.getsize(os.path.join(local, n)) for n in os.listdir(local)
                                       if n.endswith(".safetensors"))
    row["config"] = {k: json.load(open(os.path.join(local, "config.json")))[k]
                     for k in ("num_experts", "num_experts_per_tok", "num_hidden_layers")}
    log(name, f"{row['weights_bytes_on_disk'] / 1e9:.1f} GB of weights on disk, experts {row['config']}")

    capture = tempfile.TemporaryFile(mode="w+")
    saved = (os.dup(1), os.dup(2))
    load_start = time.time()
    try:
        os.dup2(capture.fileno(), 1)
        os.dup2(capture.fileno(), 2)
        llm = LLM(model=local, tokenizer=local, dtype="bfloat16", seed=0, tensor_parallel_size=tp,
                  gpu_memory_utilization=GPU_UTILIZATION, max_model_len=MAX_MODEL_LEN, enable_prefix_caching=False,
                  max_num_seqs=64)
    finally:
        os.dup2(saved[0], 1)
        os.dup2(saved[1], 2)
        capture.flush()
        capture.seek(0)
        text = capture.read()
        print(text[-6000:], flush=True)
    row["load_seconds"] = round(time.time() - load_start, 1)
    row["vllm_weight_gib_per_gpu"] = parse_load_gib(text)
    log(name, f"loaded in {row['load_seconds']} s, vLLM weight memory {row['vllm_weight_gib_per_gpu']} GiB per GPU")
    out_dir = f"{ROW_DIR}"

    def flush(stage):
        row["stage"] = stage
        row["seconds"] = round(time.time() - started, 1)
        row["dollars"] = dollars(row["seconds"], gpus)
        write_row(row, out_dir)
        safe_commit()

    flush("loaded")
    samples = {}

    def code_stage(skill, items, tokens):
        prompts = [ev.render_prompt(tokenizer, ev.user_text(skill, item)) for item in items]
        began = time.time()
        outputs = llm.generate(prompts, SamplingParams(temperature=0.0, max_tokens=tokens), use_tqdm=False)
        texts = [o.outputs[0].text for o in outputs]
        made = sum(len(o.outputs[0].token_ids) for o in outputs)
        records = ev.grade_outputs(skill, items, texts)
        correct = sum(1 for r in records if r["correct"])
        row[skill] = correct / len(records)
        row[skill + "_correct"] = correct
        row[skill + "_n"] = len(records)
        row[skill + "_gen_tps"] = round(made / max(time.time() - began, 1e-9), 1)
        row[skill + "_hit_token_cap"] = sum(1 for o in outputs if o.outputs[0].finish_reason == "length")
        samples[skill] = [{"id": r["id"], "correct": bool(r["correct"]), "pred": r["pred"], "output": r["output"]}
                          for r in records[:6]]
        log(name, f"{skill} {correct}/{len(records)} = {row[skill]:.3f} ({row[skill + '_hit_token_cap']} hit the cap)")
        os.makedirs(out_dir, exist_ok=True)
        with open(f"{out_dir}/{row['name']}_{skill}_records.json", "w") as f:
            json.dump(records, f)
        flush(skill)

    he_items = ev.humaneval_items()
    if smoke:
        he_items = he_items[:8]
    code_stage("humaneval", he_items, HUMANEVAL_TOKENS)
    if not smoke:
        code_stage("mbpp", ev.mbpp_items(), MBPP_TOKENS)

        items = ev.mmlu_items(MMLU_COUNT)
        choices = ev.letter_ids(tokenizer)
        params = SamplingParams(temperature=0.0, max_tokens=1, allowed_token_ids=choices)
        outputs = llm.generate(mmlu_prompts(items), params, use_tqdm=False)
        correct = 0
        for item, out in zip(items, outputs):
            pred = choices.index(out.outputs[0].token_ids[0])
            correct += int(pred == item["gold"])
        row["mmlu"] = correct / len(items)
        row["mmlu_correct"] = correct
        row["mmlu_n"] = len(items)
        log(name, f"mmlu {correct}/{len(items)} = {row['mmlu']:.3f}")
        flush("mmlu")

    # Speed: the same 64-prompt code pool and method as modal_serve.py, on the first SPEED_PROMPTS prompts.
    from wsp import healdata

    pairs = healdata.pairs("code", 64, seed=0)
    prompts = serve.build_prompts(tokenizer, SPEED_PROMPTS if not smoke else 8, pairs=pairs)
    ids = serve.encode_prompts(tokenizer, prompts)

    def generate(id_lists, max_tokens):
        p = SamplingParams(temperature=0.0, max_tokens=max_tokens, ignore_eos=True)
        llm.generate([{"prompt_token_ids": x} for x in id_lists], p, use_tqdm=False)

    generate(ids[:4], 16)
    row["decode_tps"] = {}
    for batch in (SPEED_BATCHES if not smoke else (1, 8)):
        tps, first = serve.measure_decode(generate, ids, batch, GEN_LEN)
        row["decode_tps"][str(batch)] = tps
        if batch == 1:
            row["ttft_ms_median"], row["ttft_ms_p90"] = serve.ttft_stats(first)
        log(name, f"decode batch {batch}: {tps} tokens/s on {tp} GPU(s)")
    flush("done")
    row["samples"] = samples
    write_row(row, out_dir)
    del llm
    gc.collect()
    safe_commit()
    return row


@app.function(image=vllm_image, volumes={"/vol": volume}, gpu="H200:1", memory=262144, timeout=3 * 3600,
              single_use_containers=True)
def eval_one(cell, smoke=False):
    return eval_cell(cell, smoke, gpus=1)


@app.function(image=vllm_image, volumes={"/vol": volume}, gpu="H200:2", memory=262144, timeout=3 * 3600,
              single_use_containers=True)
def eval_two(cell, smoke=False):
    return eval_cell(cell, smoke, gpus=2)


@app.function(image=vllm_image, volumes={"/vol": volume}, gpu="H200:4", memory=262144, timeout=3 * 3600,
              single_use_containers=True)
def eval_four(cell, smoke=False):
    return eval_cell(cell, smoke, gpus=4)


@app.function(image=vllm_image, volumes={"/vol": volume}, gpu="H200:8", memory=262144, timeout=4 * 3600,
              single_use_containers=True)
def eval_eight(cell, smoke=False):
    return eval_cell(cell, smoke, gpus=8)


EVAL_BY_TP = {1: eval_one, 2: eval_two, 4: eval_four, 8: eval_eight}


def select_cells(cells):
    """Cell names. Raises ValueError on an unknown name or an empty selection."""
    names = [n.strip() for n in cells.split(",") if n.strip()]
    if not names:
        raise ValueError("no cells selected")
    for name in names:
        if name not in CELLS:
            raise ValueError(f"unknown cell {name!r}; the cells are {', '.join(CELLS)}")
    return names


# ---------------------------------------------------------------- entrypoint


def fetch_config():
    import urllib.request

    url = f"https://huggingface.co/{MODEL}/raw/main/config.json"
    return json.load(urllib.request.urlopen(url, timeout=30))


@app.local_entrypoint()
def main(stage: str = "sizes", cells: str = "keep_16", smoke: bool = False, estimate_only: bool = False,
         keeps: str = ""):
    os.makedirs(LOCAL_DIR, exist_ok=True)
    if stage == "sizes":
        config = fetch_config()
        rows = size_table(config)
        print(format_size_table(rows))
        write_row({"name": "sizes", "config": config, "table": rows}, LOCAL_DIR, "moonshot_sizes")
        return
    if stage == "download":
        print(f"download {MODEL}: 118 shards, 470 GB, no GPU")
        if estimate_only:
            return
        print(download.remote())
        return
    if stage == "stats":
        gpus = 2 if smoke else 4
        print(f"stats: {'smoke, ' + str(SMOKE_LAYERS) + ' layers, ' + str(SMOKE_TEXTS) + ' texts' if smoke else 'full, 94 layers, ' + str(RANK_TEXTS) + ' texts'} on H200:{gpus}")
        if estimate_only:
            return
        row = (stats_smoke if smoke else stats_full).remote()
        write_row(row, LOCAL_DIR, "moonshot_" + row["name"])
        print(json.dumps(row, default=str)[:2000])
        return
    if stage == "build":
        wanted = keeps or ",".join(str(k) for k in BUILD_KEEPS)
        print(f"build keeps {wanted}, no GPU")
        if estimate_only:
            return
        for row in build.remote(keeps):
            write_row(row, LOCAL_DIR, "moonshot_" + row["name"])
            print(json.dumps({k: row[k] for k in ("keep", "params", "match", "disk_bytes")}))
        return
    if stage == "score":
        names = select_cells(cells)
        print("eval cells: " + ", ".join(f"{n} (H200:{CELLS[n][2]})" for n in names) + (" smoke" if smoke else ""))
        if estimate_only:
            return
        handles = [(n, EVAL_BY_TP[CELLS[n][2]].spawn(n, smoke)) for n in names]
        for n, handle in handles:
            row = handle.get()
            write_row(row, LOCAL_DIR, "moonshot_" + row["name"])
            print(f"[{row['cell']}] " + json.dumps({k: row.get(k) for k in
                  ("humaneval", "mbpp", "mmlu", "decode_tps", "vllm_weight_gib_per_gpu", "dollars")}, default=str))
        return
    raise SystemExit(f"unknown stage {stage!r}; stages are sizes, download, stats, build, score")
