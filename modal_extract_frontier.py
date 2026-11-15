"""Run a frontier MoE donor once over the MMLU tap prompts and cache its tap memories.

A tap needs only the donor's hidden states at one read layer (about 70%
of donor depth), for a fixed prompt set. So we run each donor once on an
8-GPU node, load only layers 0..max(read layers), and write the states
to the Modal volume. modal_round11.py then trains and scores taps on one
H100 from that cache, with no donor model loaded.

Donors (read layers at about 70% and 80% of depth):
  glm53flash  zai-org/GLM-5.3-Flash         glm5_next, 45 layers, hidden 4096, fp8      layers 31, 36
  glm53       zai-org/GLM-5.3               glm_moe_dsa, 78 layers, hidden 6144, fp8    layers 54, 62
  dsv4pro     deepseek-ai/DeepSeek-V4-Pro   deepseek_v4, 61 layers, hidden 7168, fp8    layers 42, 49
  kimik3      moonshotai/Kimi-K3            kimi_k3 (remote code), 93 layers, 7168, mxfp4  layers 65, 74

Prompts: the 8000 MMLU auxiliary_train prompts of round 6 (filtered with
the Llama-3.2-1B tokenizer), round 6's 1000 dev items, and round 7's 2000
held-out items. The cache goes to /vol/tap_cache/<donor>/ (smoke runs:
/vol/tap_cache/<donor>_smoke/, first 64 prompts).

Steps per donor:
1. download (CPU container): read model.safetensors.index.json, fetch only
   the shards that hold layers 0..max read layer plus embeddings and the
   other non-layer weights the run needs, the config, the tokenizer, and
   any remote code. The local index is rewritten to name only those shards.
2. extract (8 GPUs): load the config, cut num_hidden_layers (of the text
   config for multimodal models) to max read layer + 1, and load with
   device_map="auto", keeping the checkpoint's fp8 or mxfp4 weights. Run
   the language model only, on batches of one exact token length (no
   pads), and stop after the deepest read layer. Write each prompt's
   states with wsp.tap_cache.CacheWriter. Shards are committed as they
   are written, so a timed-out run resumes where it stopped.

    MODAL_PROFILE=trisanth modal run modal_extract_frontier.py --donor glm53flash --smoke
    MODAL_PROFILE=trisanth modal run modal_extract_frontier.py --donor glm53flash
"""

import json
import os
import time

import modal

app = modal.App("wsp-extract-frontier")
volume = modal.Volume.from_name("wsp-models", create_if_missing=True)

# torch with CUDA 12.8 wheels. transformers 5.23 needs torch >= 2.6; 2.15 is
# the version our transformers 5.23 checkouts run with.
TORCH_VERSION = "2.15.0"
TORCH_INDEX = "https://download.pytorch.org/whl/cu128"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install(f"torch=={TORCH_VERSION}", index_url=TORCH_INDEX)
    .pip_install(
        "transformers==5.23.0", "accelerate>=1.15.0", "safetensors", "datasets", "huggingface_hub[hf_transfer]",
        "hf_transfer", "sentencepiece", "protobuf", "tiktoken", "triton", "kernels>=0.18.2,<0.19.0",
        # Kimi-K3 ships compressed-tensors mxfp4 weights, and its remote code
        # imports fla (flash-linear-attention) and einops. GLM-5.3-Flash uses
        # fla for its linear attention when it is installed.
        "compressed-tensors>=0.15.0", "flash-linear-attention", "einops", "numpy",
    )
    .env({"HF_HOME": "/vol/hf", "HF_HUB_ENABLE_HF_TRANSFER": "1"})
    .add_local_python_source("wsp")
    .add_local_file("modal_graft.py", "/root/modal_graft.py")
    .add_local_file("modal_round5.py", "/root/modal_round5.py")
    .add_local_file("modal_round6.py", "/root/modal_round6.py")
    .add_local_file("modal_round7.py", "/root/modal_round7.py")
)

HOST = "unsloth/Llama-3.2-1B-Instruct"
DONORS = {
    "glm53flash": {"repo": "zai-org/GLM-5.3-Flash", "read_layers": [31, 36], "layers": 45, "hidden": 4096,
                   "gpu": "H200"},
    "glm53": {"repo": "zai-org/GLM-5.3", "read_layers": [54, 62], "layers": 78, "hidden": 6144, "gpu": "H200"},
    "dsv4pro": {"repo": "deepseek-ai/DeepSeek-V4-Pro", "read_layers": [42, 49], "layers": 61, "hidden": 7168,
                "gpu": "H200"},
    "kimik3": {"repo": "moonshotai/Kimi-K3", "read_layers": [65, 74], "layers": 93, "hidden": 7168, "gpu": "B200"},
}
# Per-GPU weight budget for device_map="auto". The rest of each GPU is for activations.
GPU_BUDGET = {"H200": "124GiB", "B200": "164GiB"}
SMOKE_PROMPTS = 64
SMALL_FILES = ["*.json", "*.py", "*.model", "*.tiktoken", "*.txt", "*.jinja", "tokenizer*"]


def model_dir(key):
    return "/vol/frontier/" + key


def cache_dir(key, smoke):
    if smoke:
        return "/vol/tap_cache/" + key + "_smoke"
    return "/vol/tap_cache/" + key


def tap_prompt_sets(tokenizer):
    """Return (pairs, dev_items, heldout_items) exactly as rounds 6 and 7 build them.

    tokenizer filters the train prompts (round 6 used the Llama-3.2-1B one).
    """
    from modal_round6 import mmlu_data
    from modal_round7 import mmlu_split

    pairs, dev_items = mmlu_data(tokenizer=tokenizer)
    _, heldout_items = mmlu_split(tokenizer, heldout=True)
    return pairs, dev_items, heldout_items


def tap_prompts(tokenizer):
    """Return the unique prompt texts the cache must hold: train, then dev, then held-out."""
    pairs, dev_items, heldout_items = tap_prompt_sets(tokenizer)
    prompts = []
    seen = set()
    for prompt, completion in pairs:
        if prompt not in seen:
            seen.add(prompt)
            prompts.append(prompt)
    for items in [dev_items, heldout_items]:
        for prompt, answer in items:
            if prompt not in seen:
                seen.add(prompt)
                prompts.append(prompt)
    return prompts


@app.function(image=image, volumes={"/vol": volume}, cpu=16, memory=65536, timeout=6 * 3600)
def download(key):
    """Fetch the shards for layers 0..max read layer, plus config, tokenizer, and remote code."""
    from huggingface_hub import hf_hub_download, snapshot_download

    from wsp.donor_extract import choose_shards, filter_weight_map

    spec = DONORS[key]
    local = model_dir(key)
    max_layer = max(spec["read_layers"])
    start = time.time()
    os.makedirs(local, exist_ok=True)
    full_path = os.path.join(local, "model.safetensors.index.full.json")
    if not os.path.exists(full_path):
        path = hf_hub_download(spec["repo"], "model.safetensors.index.json", local_dir=local)
        os.replace(path, full_path)
    with open(full_path) as f:
        index = json.load(f)
    files = choose_shards(index["weight_map"], max_layer)
    all_files = set(index["weight_map"].values())
    print(f"{key}: {len(files)} of {len(all_files)} shards hold layers 0..{max_layer} and the other needed weights")
    snapshot_download(spec["repo"], local_dir=local, allow_patterns=files + SMALL_FILES, max_workers=16)
    # The small files include the full index again; write ours over it.
    kept = filter_weight_map(index["weight_map"], files, max_layer)
    with open(os.path.join(local, "model.safetensors.index.json"), "w") as f:
        json.dump({"metadata": index.get("metadata", {}), "weight_map": kept}, f)
    size = 0
    for name in files:
        size += os.path.getsize(os.path.join(local, name))
    volume.commit()
    return {"donor": key, "shards": len(files), "shards_total": len(all_files), "gb": round(size / 1e9, 1),
            "keys": len(kept), "seconds": round(time.time() - start, 1)}


def model_class(config):
    """Return the class to load: the native architecture class, or the auto class for remote code."""
    import transformers

    auto_map = getattr(config, "auto_map", None) or {}
    architectures = getattr(config, "architectures", None) or []
    if len(architectures) > 0 and hasattr(transformers, architectures[0]) and "AutoModelForCausalLM" not in auto_map:
        return getattr(transformers, architectures[0])
    return transformers.AutoModelForCausalLM


def readout_name(decoder):
    if getattr(decoder, "use_attn_residuals", False):
        return "output attention residual (Kimi)"
    if getattr(decoder, "hc_head", None) is not None:
        return "hc_head stream collapse"
    return "layer output"


def gpu_memory():
    import torch

    rows = []
    for index in range(torch.cuda.device_count()):
        rows.append({"gpu": index, "allocated_gb": round(torch.cuda.memory_allocated(index) / 1e9, 1),
                     "peak_gb": round(torch.cuda.max_memory_allocated(index) / 1e9, 1)})
    return rows


def load_donor(key, gpu):
    """Load layers 0..max read layer of a donor, quantized as in its checkpoint. Return (model, tokenizer, info)."""
    import importlib.util

    import torch
    import transformers

    from wsp.donor_extract import find_decoder, patch_remote_mask_calls, text_config_of, truncate_text_config

    spec = DONORS[key]
    local = model_dir(key)
    keep = max(spec["read_layers"]) + 1
    config = transformers.AutoConfig.from_pretrained(local, trust_remote_code=True)
    old = truncate_text_config(config, keep)
    if old != spec["layers"]:
        print(f"warning: {key} config has {old} layers, expected {spec['layers']}")
    vision = getattr(config, "vision_config", None)
    if isinstance(vision, dict) and vision.get("_attn_implementation") == "flash_attention_2":
        # We never run the vision tower; this keeps loading from asking for flash-attn.
        vision["_attn_implementation"] = "eager"

    budget = {}
    for index in range(torch.cuda.device_count()):
        budget[index] = GPU_BUDGET[gpu]
    start = time.time()
    model = model_class(config).from_pretrained(local, config=config, device_map="auto", dtype=torch.bfloat16,
                                                trust_remote_code=True, max_memory=budget)
    model.eval()
    load_seconds = time.time() - start
    quantizer = getattr(model, "hf_quantizer", None)
    quant = None
    if quantizer is not None:
        quant_config = quantizer.quantization_config
        quant = str(getattr(quant_config, "quant_method", type(quant_config).__name__))
        if getattr(quant_config, "dequantize", False):
            raise RuntimeError(f"{key} was dequantized to bf16 on load (missing kernels or an unsupported GPU); "
                               "it would not fit, so we stop here")

    decoder, layers = find_decoder(model, keep - 1)
    if len(layers) != keep:
        raise RuntimeError(f"{key} built {len(layers)} decoder layers, expected {keep}")
    patched = patch_remote_mask_calls(model)
    text = text_config_of(decoder.config) if hasattr(decoder, "config") else None
    if text is not None and getattr(text, "_attn_implementation", None) == "flash_attention_2":
        if importlib.util.find_spec("flash_attn") is None:
            # Kimi's remote code forces flash_attention_2; its MLA attention
            # looks the backend up at run time, and SDPA handles its shapes.
            text._attn_implementation = "sdpa"
    tokenizer = transformers.AutoTokenizer.from_pretrained(local, trust_remote_code=True)
    info = {"model_type": getattr(config, "model_type", None), "model_class": type(model).__name__,
            "decoder_class": type(decoder).__name__, "layers_total": old, "layers_loaded": len(layers),
            "quantization": quant, "readout": readout_name(decoder), "patched_remote_modules": patched,
            "attn_implementation": getattr(text, "_attn_implementation", None) if text is not None else None,
            "tokenizer_class": type(tokenizer).__name__, "tokenizer_fast": bool(getattr(tokenizer, "is_fast", False)),
            "load_seconds": round(load_seconds, 1), "memory_after_load": gpu_memory()}
    print(json.dumps(info))
    return model, tokenizer, info


def extract_body(key, smoke, gpu):
    import torch
    import transformers

    from wsp.donor_extract import count_layer_params, extract_prompts, find_decoder
    from wsp.tap_cache import CacheWriter

    spec = DONORS[key]
    start = time.time()
    host_tokenizer = transformers.AutoTokenizer.from_pretrained(HOST)
    prompts = tap_prompts(host_tokenizer)
    if smoke:
        prompts = prompts[:SMOKE_PROMPTS]
    print(f"{key}: {len(prompts)} prompts")

    model, tokenizer, info = load_donor(key, gpu)
    decoder, layers = find_decoder(model, max(spec["read_layers"]))
    hidden = int(decoder.config.hidden_size)
    if hidden != spec["hidden"]:
        print(f"warning: {key} hidden size {hidden}, expected {spec['hidden']}")
    meta = {"donor_id": spec["repo"], "donor_key": key, "tokenizer_id": spec["repo"],
            "layers_total": info["layers_total"], "model_type": info["model_type"], "readout": info["readout"],
            "quantization": info["quantization"], "transformers": transformers.__version__,
            "torch": torch.__version__, "gpu": gpu + ":" + str(torch.cuda.device_count()),
            "prompt_set": "mmlu round6 train 8000 (Llama-3.2-1B filter) + round6 dev 1000 + round7 heldout 2000",
            "params_ran_by_layer": count_layer_params(layers, spec["read_layers"])}
    out_dir = cache_dir(key, smoke)
    writer = CacheWriter(out_dir, spec["read_layers"], hidden, meta=meta, on_shard=volume.commit)
    for index in range(torch.cuda.device_count()):
        torch.cuda.reset_peak_memory_stats(index)
    stats = extract_prompts(model, tokenizer, prompts, spec["read_layers"], writer, max_batch=64, max_tokens=16384)
    final = writer.close()
    report = {"donor": key, "smoke": smoke, "cache": out_dir, "prompts": len(prompts),
              "cached_prompts": final["num_prompts"], "cached_tokens": final["num_tokens"],
              "shards": len(final["shards"]), "seconds": round(time.time() - start, 1),
              "peak_memory": gpu_memory()}
    report.update(stats)
    report.update(info)
    with open(os.path.join(out_dir, "extract_log.json"), "w") as f:
        json.dump(report, f, indent=2)
    volume.commit()
    print(json.dumps(report))
    return report


@app.function(image=image, volumes={"/vol": volume}, gpu="H200:8", cpu=32, memory=262144, timeout=6 * 3600)
def extract_h200(key, smoke=False):
    return extract_body(key, smoke, "H200")


@app.function(image=image, volumes={"/vol": volume}, gpu="B200:8", cpu=32, memory=262144, timeout=6 * 3600)
def extract_b200(key, smoke=False):
    return extract_body(key, smoke, "B200")


@app.local_entrypoint()
def main(donor: str, smoke: bool = False, skip_download: bool = False):
    if donor not in DONORS:
        raise SystemExit(f"--donor must be one of {sorted(DONORS)}")
    if not skip_download:
        print(json.dumps(download.remote(donor)))
    if DONORS[donor]["gpu"] == "B200":
        report = extract_b200.remote(donor, smoke)
    else:
        report = extract_h200.remote(donor, smoke)
    os.makedirs("results", exist_ok=True)
    name = "extract_" + donor
    if smoke:
        name += "_smoke"
    with open(os.path.join("results", name + ".json"), "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps({k: report.get(k) for k in ["donor", "cached_prompts", "cached_tokens", "tokens_per_second",
                                                  "extract_seconds", "seconds", "peak_memory"]}))
