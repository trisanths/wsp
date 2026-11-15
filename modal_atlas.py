"""Skill atlas: screen every zoo model on every skill and map where each skill lives.

One container per model (H200). Each container loads the model from
atlas/zoo.json, runs the skill screens in wsp/atlas.py, the per-layer
skill maps, and for MoE models the expert routing maps. It writes
/vol/atlas/entries/<org>__<name>.json on the wsp-models volume and returns
the entry. The local entrypoint writes results/atlas_shard<i>.json (a
list of entries) and atlas/entries/<org>__<name>.json.

The image pins transformers 4.57.1. Models whose zoo entry needs
transformers 5.x, gated models, and giants (multi_gpu) are skipped with a
reason, written to results/atlas_skipped_shard<i>.json. Cells that fail
go to results/atlas_failed_shard<i>.json. --giants runs the
giants on H200:8 with expert maps only (no screens, no layer maps).

    MODAL_PROFILE=trisanth .venv/bin/modal run modal_atlas.py --only Qwen/Qwen2.5-0.5B-Instruct --scale 0.1
    python scripts/modal_fanout.py modal_atlas.py atlas --profiles a,b,c
"""

import json
import os

import modal

app = modal.App("wsp-atlas")
volume = modal.Volume.from_name("wsp-models", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch==2.7.1", "transformers==4.57.1", "accelerate", "huggingface_hub", "requests", "safetensors",
        "zstandard", "numpy", "datasets", "sentencepiece", "protobuf", "tiktoken",
    )
    .env({"HF_HOME": "/vol/hf"})
    .add_local_python_source("wsp")
)

ZOO_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "atlas", "zoo.json")
VOLUME_ENTRIES = "/vol/atlas/entries"


def load_model(zoo_model, device_map=None):
    import torch
    import transformers

    from wsp import atlas

    name = zoo_model["id"]
    kwargs = {"dtype": torch.bfloat16, "trust_remote_code": zoo_model["needs"]["trust_remote_code"]}
    if zoo_model["family"] in ("gemma2", "gemma3", "gemma3_text"):
        # Gemma soft-capping and sliding windows are exact only in eager attention.
        kwargs["attn_implementation"] = "eager"
    if device_map is not None:
        kwargs["device_map"] = device_map
    model = transformers.AutoModelForCausalLM.from_pretrained(name, **kwargs)
    if device_map is None:
        model = model.to("cuda")
    model.eval()
    tokenizer = transformers.AutoTokenizer.from_pretrained(name, trust_remote_code=zoo_model["needs"]["trust_remote_code"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # Chat models whose tokenizer has no chat template name one in the zoo.
    atlas.ensure_chat_template(tokenizer, zoo_model.get("chat_format"))
    return model, tokenizer


def generation_budget(zoo_model, options):
    if options.get("gen_tokens"):
        return int(options["gen_tokens"])
    if zoo_model["kind"] == "reasoning":
        return 2048
    return 768


def batch_for(zoo_model, options):
    if options.get("batch_size"):
        return int(options["batch_size"])
    params = zoo_model.get("params_b") or 8.0
    if params < 10:
        return 32
    return 16


def save_to_volume(entry):
    from wsp import atlas

    path = atlas.write_entry(entry, VOLUME_ENTRIES)
    volume.commit()
    return path


def finish(entry, zoo_model):
    entry["zoo"] = {"key": zoo_model["key"], "kind": zoo_model["kind"], "maker": zoo_model["maker"],
                    "listed_skills": zoo_model["skills"]}
    save_to_volume(entry)
    return entry


def saver(zoo_model):
    """on_progress callback: save the entry so far, so a timeout keeps finished skills."""
    def save(entry):
        entry["partial"] = True
        finish(entry, zoo_model)
    return save


@app.function(image=image, volumes={"/vol": volume}, gpu="H200", memory=196608, timeout=6 * 3600)
def measure(zoo_model, options):
    from wsp import atlas

    model, tokenizer = load_model(zoo_model)
    entry = atlas.measure_model(
        model, tokenizer, zoo_model["id"],
        skills=options.get("skills") or None,
        scale=float(options.get("scale", 1.0)),
        map_texts=int(options.get("map_texts", 32)),
        max_new_tokens=generation_budget(zoo_model, options),
        batch_size=batch_for(zoo_model, options),
        screens=bool(options.get("screens", True)),
        layer_maps=bool(options.get("layer_maps", True)),
        on_progress=saver(zoo_model),
    )
    return finish(entry, zoo_model)


@app.function(image=image, volumes={"/vol": volume}, gpu="H200:8", timeout=12 * 3600)
def measure_giant(zoo_model, options):
    """Expert maps only for models that need several GPUs."""
    from wsp import atlas

    model, tokenizer = load_model(zoo_model, device_map="auto")
    entry = atlas.measure_model(
        model, tokenizer, zoo_model["id"],
        skills=options.get("skills") or None,
        map_texts=int(options.get("map_texts", 32)),
        screens=False, layer_maps=False, expert_maps=True,
        on_progress=saver(zoo_model),
    )
    return finish(entry, zoo_model)


def shard_cells(models, shard, nshards):
    """The models of one shard: every nshards-th model, starting at shard."""
    if nshards < 1 or shard < 0 or shard >= nshards:
        raise ValueError(f"shard {shard} is not in 0..{nshards - 1}")
    cells = []
    for index, zoo_model in enumerate(models):
        if index % nshards == shard:
            cells.append(zoo_model)
    return cells


def select(zoo, giants, gated, only):
    """(normal models, giant models, skipped rows) in zoo order."""
    from wsp import atlas

    wanted = None
    if only:
        wanted = set(only.split(","))
    normal = []
    big = []
    skipped = []
    for zoo_model in zoo["models"]:
        if wanted is not None and zoo_model["id"] not in wanted and zoo_model["key"] not in wanted:
            continue
        ok, reason = atlas.runnable(zoo_model, zoo["image_transformers"], giants=giants, gated=gated)
        if not ok:
            skipped.append({"model": zoo_model["id"], "reason": reason})
            continue
        if zoo_model["needs"]["multi_gpu"]:
            big.append(zoo_model)
        else:
            normal.append(zoo_model)
    return normal, big, skipped


@app.local_entrypoint()
def main(shard: int = 0, nshards: int = 1, giants: bool = False, gated: bool = False, only: str = "",
         skills: str = "", scale: float = 1.0, map_texts: int = 32, gen_tokens: int = 0, batch_size: int = 0,
         no_screens: bool = False, no_layer_maps: bool = False):
    from wsp import atlas

    with open(ZOO_PATH) as f:
        zoo = json.load(f)
    normal, big, skipped = select(zoo, giants, gated, only)
    cells = shard_cells(normal + big, shard, nshards)

    options = {"scale": scale, "map_texts": map_texts, "gen_tokens": gen_tokens, "batch_size": batch_size,
               "screens": not no_screens, "layer_maps": not no_layer_maps}
    if skills:
        options["skills"] = skills.split(",")

    names = []
    for zoo_model in cells:
        names.append(zoo_model["id"])
    print(f"shard {shard}/{nshards}: {len(cells)} models: {names}")
    if shard == 0:
        for row in skipped:
            print(f"skip {row['model']}: {row['reason']}")

    normal_cells = []
    giant_cells = []
    for zoo_model in cells:
        if zoo_model["needs"]["multi_gpu"]:
            giant_cells.append(zoo_model)
        else:
            normal_cells.append(zoo_model)

    rows = []
    failed = []
    calls = []
    if normal_cells:
        calls.append((measure, normal_cells))
    if giant_cells:
        calls.append((measure_giant, giant_cells))
    for function, group in calls:
        option_list = []
        for _ in group:
            option_list.append(options)
        # map keeps input order, so each result lines up with its model.
        results = function.map(group, option_list, return_exceptions=True)
        for zoo_model, result in zip(group, results):
            if isinstance(result, Exception):
                print(f"cell failed: {zoo_model['id']}: {result!r}"[:500])
                failed.append({"model": zoo_model["id"], "error": repr(result)[:2000],
                               "note": "a partial entry may be in /vol/atlas/entries"})
                continue
            rows.append(result)
            atlas.write_entry(result, os.path.join("atlas", "entries"))
            print(json.dumps({"model": result["model"], "skills": result["skills"]}))

    os.makedirs("results", exist_ok=True)
    with open(f"results/atlas_shard{shard}.json", "w") as f:
        json.dump(rows, f)
    with open(f"results/atlas_skipped_shard{shard}.json", "w") as f:
        json.dump(skipped, f, indent=1)
    with open(f"results/atlas_failed_shard{shard}.json", "w") as f:
        json.dump(failed, f, indent=1)
