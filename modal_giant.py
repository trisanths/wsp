"""Profile one giant open MoE model on a single H200 by streaming its layers.

No giant fits one GPU. wsp/giant.py reads one decoder layer at a time by byte
range, decodes the stored FP8, FP4 or mxfp4 tensors on the GPU, runs the
cached hidden states of the calibration texts through the layer, records the
router choices per skill, and frees the layer. The result has the format of
wsp.atlas.expert_skill_map and goes to /vol/atlas/entries/<org>__<name>.json
(merged into an existing entry) and results/giant_<name>.json.

Before it starts, the local entrypoint prints the bytes to download and an
hour estimate from the shard headers. The rates behind the hours are
assumptions, set with --mbps and --layer-seconds.

    MODAL_PROFILE=trisanth .venv/bin/modal run modal_giant.py --only zai-org/GLM-5.3-Flash --max-layers 2 --texts 4 --skills math
    MODAL_PROFILE=trisanth .venv/bin/modal run modal_giant.py --only deepseek-ai/DeepSeek-V4-Flash
    MODAL_PROFILE=trisanth .venv/bin/modal run modal_giant.py --shard 0 --nshards 3

Shards split the list of models, not the layers: layer k needs the hidden
states of layer k - 1, so one model runs in one container.

Images. The default image pins transformers 5.23.0 (DeepSeek-V4, GLM-5.3,
Qwen3.5 and Qwen3.8 are native there). Kimi K3 ships remote code that imports
names removed in transformers 5.x, so it runs on a second image with
transformers 4.57.1, einops and fla-core, and with the sdpa attention backend
instead of the flash_attention_2 that its model code forces. The Kimi image
has not been run.

Stored quantization is decoded in wsp.giant: FP8 with 128x128 block scales
(GLM, V4 dense weights), FP4 with one e8m0 scale per 32 inputs (V4 experts),
compressed-tensors mxfp4 (Kimi experts). Nothing is dequantized on disk.
"""

import json
import os

import modal

app = modal.App("wsp-giant")
volume = modal.Volume.from_name("wsp-models", create_if_missing=True)


def build_image(transformers_version, extra=()):
    return (
        modal.Image.debian_slim(python_version="3.12")
        .pip_install(
            "torch==2.7.1", f"transformers=={transformers_version}", "accelerate", "huggingface_hub", "requests",
            "safetensors", "zstandard", "numpy", "datasets", "sentencepiece", "protobuf", "tiktoken", *extra,
        )
        .env({"HF_HOME": "/vol/hf"})
        .add_local_python_source("wsp")
    )


image_tf5 = build_image("5.23.0")
image_kimi = build_image("4.57.1", extra=("einops", "fla-core"))

ZOO_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "atlas", "zoo.json")
VOLUME_ENTRIES = "/vol/atlas/entries"
STATE_ROOT = "/vol/giant_state"
DEFAULT_SKILLS = ["knowledge", "medicine", "law", "math", "code", "science", "reasoning", "multilingual",
                  "multilingual_math", "decision"]


def estimate(zoo_model, mbps, layer_seconds, max_layers=None):
    """Bytes to read and hours, from the shard headers of the repo. Returns a dict."""
    from wsp import giant
    from transformers import AutoConfig

    trust = zoo_model["needs"]["trust_remote_code"]
    config = AutoConfig.from_pretrained(zoo_model["id"], trust_remote_code=trust)
    source = giant.HubSource(zoo_model["id"])
    layout = giant.layout_for(config, source)
    text = config.get_text_config() if hasattr(config, "get_text_config") else config
    count = text.num_hidden_layers
    layers = list(range(count))
    if max_layers is not None:
        layers = layers[:max_layers]
    sizes = giant.stream_bytes(source, layout, count, layers=layers)
    download, compute, total = giant.estimate_hours(sizes["total"], len(layers), mbps, layer_seconds)
    return {"model": zoo_model["id"], "layers": len(layers), "layers_total": count, "layout": layout.family,
            "download_gb": round(sizes["total"] / 1e9, 1), "embedding_gb": round(sizes["embedding"] / 1e9, 2),
            "max_layer_gb": round(max(sizes["per_layer"].values()) / 1e9, 2), "mbps_assumed": mbps,
            "layer_seconds_assumed": layer_seconds, "download_hours": round(download, 2),
            "compute_hours": round(compute, 2), "total_hours": round(total, 2),
            "cost_usd_at_4_5_per_hour": round(total * 4.5, 1)}


def run_profile(zoo_model, options):
    import torch
    from transformers import AutoConfig, AutoTokenizer

    from wsp import atlas, giant

    model_id = zoo_model["id"]
    trust = zoo_model["needs"]["trust_remote_code"]
    config = AutoConfig.from_pretrained(model_id, trust_remote_code=trust)
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=trust)
    source = giant.HubSource(model_id)
    layout = giant.layout_for(config, source)
    skills = options.get("skills") or DEFAULT_SKILLS
    count = int(options.get("texts", 32))
    general = atlas.general_texts(count)
    skill_texts = {}
    for name in skills:
        try:
            skill_texts[name] = atlas.SKILLS[name]["texts"](count)
        except Exception as error:
            print(f"{model_id} {name}: texts failed: {error!r}"[:300])
    state_dir = os.path.join(STATE_ROOT, zoo_model["key"], f"t{count}_" + "-".join(sorted(skill_texts)))
    result = giant.profile_streaming(
        source, config, tokenizer, skill_texts, general, layout=layout, device="cuda", dtype=torch.bfloat16,
        max_layers=options.get("max_layers"), state_dir=state_dir, checkpoint_every=4, prefetch=True,
        trust_remote_code=trust, attn_implementation=options.get("attn_implementation"),
        log=lambda text: print(f"{model_id} {text}", flush=True))
    volume.commit()
    entry = merge_entry(zoo_model, result["maps"], result["info"])
    return {"entry": entry, "info": result["info"]}


def merge_entry(zoo_model, maps, info):
    from wsp import atlas

    path = os.path.join(VOLUME_ENTRIES, atlas.entry_name(zoo_model["id"]))
    base = {}
    if os.path.exists(path):
        base = atlas.read_entry(path)
    entry = atlas.make_entry(zoo_model["id"], base.get("skills", {}), None, maps, zoo_model.get("params_b"),
                             zoo_model.get("layers"), zoo_model.get("hidden"), zoo_model.get("family"),
                             base.get("skill_details", {}))
    for key in ("layer_maps",):
        if base.get(key):
            entry[key] = base[key]
    previous = base.get("expert_maps") or {}
    merged = {}
    for skill in previous:
        merged[skill] = previous[skill]
    for skill in maps:
        merged[skill] = atlas.top_experts(maps[skill])
    entry["expert_maps"] = merged
    entry["zoo"] = {"key": zoo_model["key"], "kind": zoo_model["kind"], "maker": zoo_model["maker"],
                    "listed_skills": zoo_model["skills"]}
    entry["streamed"] = {"layers_done": info["layers_done"], "layers_total": info["layers_total"],
                         "texts": info["texts"], "bytes_read": info["bytes_read"]}
    if info["layers_done"] < info["layers_total"]:
        entry["partial"] = True
    atlas.write_entry(entry, VOLUME_ENTRIES)
    volume.commit()
    return entry


@app.function(image=image_tf5, volumes={"/vol": volume}, gpu="H200", memory=262144, timeout=24 * 3600)
def profile(zoo_model, options):
    """Streaming expert profile of one giant, on the transformers 5.23.0 image."""
    return run_profile(zoo_model, options)


@app.function(image=image_kimi, volumes={"/vol": volume}, gpu="H200", memory=262144, timeout=24 * 3600)
def profile_kimi(zoo_model, options):
    """Same, on the transformers 4.57.1 image for Kimi remote code. Not run yet."""
    options = dict(options)
    options.setdefault("attn_implementation", "sdpa")
    return run_profile(zoo_model, options)


def giant_models(zoo, only):
    wanted = None
    if only:
        wanted = set(only.split(","))
    out = []
    for zoo_model in zoo["models"]:
        stream = zoo_model.get("stream")
        if not zoo_model["moe"] or not stream or not stream.get("supported"):
            continue
        if wanted is not None and zoo_model["id"] not in wanted and zoo_model["key"] not in wanted:
            continue
        out.append(zoo_model)
    return out


def shard_cells(models, shard, nshards):
    if nshards < 1 or shard < 0 or shard >= nshards:
        raise ValueError(f"shard {shard} is not in 0..{nshards - 1}")
    cells = []
    for index, zoo_model in enumerate(models):
        if index % nshards == shard:
            cells.append(zoo_model)
    return cells


@app.local_entrypoint()
def main(only: str = "", skills: str = "", texts: int = 32, max_layers: int = 0, shard: int = 0, nshards: int = 1,
         mbps: float = 300.0, layer_seconds: float = 30.0, estimate_only: bool = False):
    from wsp import atlas

    with open(ZOO_PATH) as f:
        zoo = json.load(f)
    models = giant_models(zoo, only)
    cells = shard_cells(models, shard, nshards)
    if not cells:
        print("no giant models selected; zoo entries need a `stream` field")
        return
    limit = max_layers if max_layers > 0 else None
    options = {"texts": texts, "max_layers": limit}
    if skills:
        options["skills"] = skills.split(",")
    for zoo_model in cells:
        rows = estimate(zoo_model, mbps, layer_seconds, limit)
        print(json.dumps(rows))
    if estimate_only:
        return
    os.makedirs("results", exist_ok=True)
    calls = []
    for zoo_model in cells:
        function = profile_kimi if zoo_model["stream"]["layout"] == "kimi" else profile
        calls.append((zoo_model, function.spawn(zoo_model, options)))
    for zoo_model, call in calls:
        name = zoo_model["key"].split("__", 1)[-1]
        try:
            result = call.get()
        except Exception as error:
            print(f"{zoo_model['id']} failed: {error!r}"[:500])
            with open(f"results/giant_{name}_failed.json", "w") as f:
                json.dump({"model": zoo_model["id"], "error": repr(error)[:2000]}, f, indent=1)
            continue
        atlas.write_entry(result["entry"], os.path.join("atlas", "entries"))
        with open(f"results/giant_{name}.json", "w") as f:
            json.dump({"model": zoo_model["id"], "info": result["info"], "entry": result["entry"]}, f)
        print(json.dumps({"model": zoo_model["id"], "layers_done": result["info"]["layers_done"],
                          "seconds": result["info"]["seconds"], "gb_read": round(result["info"]["bytes_read"] / 1e9, 1)}))
