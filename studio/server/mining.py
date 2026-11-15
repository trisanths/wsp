"""Mining routes: the queue of mining requests, the models the library has mined, and part detail.

CPU only. The studio does not start mining. A request is written to
mine_requests.json in the studio output dir, and the mining team reads that
file as a queue. Model and part facts come from the library (library.py) and
the palette (catalog.py).
"""

import json
import math
import os
import re
import threading
import time
import uuid

from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool

from studio.server import catalog as catalog_mod
from studio.server import library
from studio.server import settings

REQUESTS_FILE = "mine_requests.json"
EFFORTS = ["smoke", "medium", "full"]
MODEL_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,95}/[A-Za-z0-9][A-Za-z0-9._-]{0,95}$")
SKILL_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9 ._+-]{0,63}$")
MAX_SKILLS = 20
MAX_NOTE = 500
LOOKUP_TIMEOUT_S = 6.0
LOOKUP_TTL_S = 600.0

STAGES = [
    ("fetch", "Fetch weights", "modal"),
    ("read", "Read the weights", "openai"),
    ("scan", "Scan activations", "modal"),
    ("ablate", "Switch parts off", "modal"),
    ("gate", "Compare with random parts", "modal"),
    ("publish", "Publish and share", "pastdev"),
]

TECHNIQUE_ORDER = ["weight-read", "activation-scan", "attribution", "saliency", "skill-map", "weight-delta",
                   "deletion-mask", "ablation", "gate"]

_lock = threading.Lock()
_lookup_cache = {}
_index_cache = {}
_tech_cache = {}


def error_body(message, status):
    return JSONResponse({"error": message}, status_code=status)


# Requests.

def requests_path():
    return os.path.join(settings.out_dir(), REQUESTS_FILE)


def read_requests():
    try:
        with open(requests_path(), encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, ValueError):
        return []
    if isinstance(data, dict):
        data = data.get("requests")
    if not isinstance(data, list):
        return []
    return [item for item in data if isinstance(item, dict)]


def write_requests(items):
    path = requests_path()
    temp = path + "." + str(os.getpid()) + ".tmp"
    with open(temp, "w", encoding="utf-8") as handle:
        json.dump({"requests": items}, handle, indent=1)
    os.replace(temp, path)


def guess_params_b(model):
    """Parameters in billions for a model id: the library profile, a cached lookup, the name, else 7."""
    profile = library.get()["models"].get(model)
    if profile and profile.get("params_b"):
        return float(profile["params_b"])
    cached = _lookup_cache.get(model)
    if cached and cached[1].get("params_b"):
        return float(cached[1]["params_b"])
    match = re.search(r"(\d+(?:\.\d+)?)\s*[bB](?![a-zA-Z])", model.split("/")[-1])
    if match:
        return float(match.group(1))
    return 7.0


def estimate(params_b, effort):
    """A rough GPU time and cost. It is an estimate, not a quote."""
    mult = {"smoke": 0.6, "medium": 2.5, "full": 6.0}[effort]
    fixed = {"smoke": 4.0, "medium": 12.0, "full": 30.0}[effort]
    gpu_min = int(round(params_b * mult + fixed))
    big = params_b > 10
    rate = 4.0 if big else 2.5
    return {"gpu_min": gpu_min, "cost_usd": round(gpu_min / 60.0 * rate, 2), "gpu": "H100" if big else "L40S",
            "is_estimate": True, "note": "Rough estimate from the parameter count and effort. Real time varies."}


def clean_request_body(body):
    """(model, skills, effort, note) or raises ValueError with a plain message."""
    if not isinstance(body, dict):
        raise ValueError("The request body must be a JSON object.")
    model = body.get("model")
    if not isinstance(model, str) or not MODEL_ID.match(model.strip()):
        raise ValueError("The model must be a Hugging Face id like owner/name.")
    model = model.strip()
    skills = body.get("skills", [])
    if skills is None:
        skills = []
    if not isinstance(skills, list):
        raise ValueError("Skills must be a list of names.")
    if len(skills) > MAX_SKILLS:
        raise ValueError("Ask for at most " + str(MAX_SKILLS) + " skills at a time.")
    clean = []
    for skill in skills:
        if not isinstance(skill, str) or not SKILL_ID.match(skill.strip()):
            raise ValueError("A skill name is empty or has characters the miners cannot use.")
        if skill.strip() not in clean:
            clean.append(skill.strip())
    effort = body.get("effort", "medium")
    if effort not in EFFORTS:
        raise ValueError("Effort must be smoke, medium or full.")
    note = body.get("note") or ""
    if not isinstance(note, str):
        raise ValueError("The note must be text.")
    if len(note) > MAX_NOTE:
        raise ValueError("The note is too long. Keep it under " + str(MAX_NOTE) + " characters.")
    return model, clean, effort, note.strip()


def create_request(body):
    """Validate, queue and persist one request. Raises ValueError for bad input."""
    model, skills, effort, note = clean_request_body(body)
    with _lock:
        items = read_requests()
        for item in items:
            if item.get("model") == model and item.get("status") in ("queued", "running"):
                raise ValueError("This model is already in the queue (" + str(item.get("id")) + ").")
        stages = [{"id": sid, "name": name, "status": "waiting", "sponsor": sponsor} for sid, name, sponsor in STAGES]
        item = {"id": "mine_" + uuid.uuid4().hex[:8], "model": model, "skills": skills, "effort": effort,
                "note": note, "status": "queued", "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "stages": stages, "estimate": estimate(guess_params_b(model), effort)}
        items.append(item)
        write_requests(items)
    return item


def with_facts(item):
    """A copy of the request with model facts (params_b, layers, experts) from the library or the lookup cache. It adds nothing the library does not hold."""
    out = dict(item)
    model = out.get("model")
    profile = library_profile(model) or {}
    cached = (_lookup_cache.get(model) or (0, {}))[1]
    for key in ("params_b", "layers", "experts"):
        if out.get(key) is None:
            value = profile.get(key)
            if value is None:
                value = cached.get(key)
            if value is not None:
                out[key] = value
    return out


def get_request(request_id):
    for item in read_requests():
        if item.get("id") == request_id:
            return item
    return None


# Hugging Face lookup.

def fetch_hf(model):
    """Raw facts from the Hub. Raises on any failure. Tests replace this function."""
    from huggingface_hub import HfApi
    info = HfApi().model_info(model, timeout=LOOKUP_TIMEOUT_S, token=False,
                              expand=["config", "safetensors", "gated", "downloads", "likes", "cardData", "tags"])
    config = getattr(info, "config", None) or {}
    safetensors = getattr(info, "safetensors", None)
    total = None
    if safetensors is not None:
        total = getattr(safetensors, "total", None)
        if total is None and isinstance(safetensors, dict):
            total = safetensors.get("total")
    card = getattr(info, "card_data", None)
    license_name = None
    if card is not None:
        license_name = getattr(card, "license", None)
        if license_name is None and isinstance(card, dict):
            license_name = card.get("license")
    if license_name is None:
        for tag in getattr(info, "tags", None) or []:
            if str(tag).startswith("license:"):
                license_name = str(tag)[len("license:"):]
    experts = None
    for key in ["num_local_experts", "num_experts", "n_routed_experts", "moe_num_experts"]:
        if config.get(key):
            experts = config[key]
            break
    architectures = config.get("architectures") or []
    layers = config.get("num_hidden_layers") or config.get("n_layer") or config.get("num_layers")
    return {"params_b": round(total / 1e9, 3) if total else None,
            "architecture": architectures[0] if architectures else config.get("model_type"),
            "layers": layers, "experts": experts, "license": license_name,
            "gated": bool(getattr(info, "gated", False)), "downloads": getattr(info, "downloads", None),
            "likes": getattr(info, "likes", None)}


def library_profile(model):
    return library.get()["models"].get(model)


def lookup(model):
    """The lookup body for one model id."""
    base = {"model": model, "found": False, "params_b": None, "architecture": None, "layers": None,
            "experts": None, "license": None, "gated": None, "downloads": None, "likes": None,
            "in_library": False, "mined_parts": 0, "error": None}
    mined = 0
    for part in library.get()["parts"] + library.get()["masks"]:
        if part.get("model") == model:
            mined = mined + 1
    profile = library_profile(model)
    base["in_library"] = profile is not None or mined > 0
    base["mined_parts"] = mined
    if not MODEL_ID.match(model or ""):
        base["error"] = "The model must be a Hugging Face id like owner/name."
        return base
    now = time.time()
    cached = _lookup_cache.get(model)
    if cached and now - cached[0] < LOOKUP_TTL_S:
        facts = cached[1]
    else:
        facts = None
        try:
            facts = fetch_hf(model)
            _lookup_cache[model] = (now, facts)
        except Exception as error:
            name = error.__class__.__name__
            if name in ("RepositoryNotFoundError", "RevisionNotFoundError") and profile is None:
                base["error"] = "Hugging Face has no model with this id."
                return base
            if profile is None:
                base["error"] = "Could not reach Hugging Face (" + name + ")."
                return base
            base["error"] = "Could not reach Hugging Face, so this shows library data only."
    if facts is not None:
        base.update({key: facts.get(key) for key in ["params_b", "architecture", "layers", "experts", "license",
                                                      "gated", "downloads", "likes"]})
        base["found"] = True
    if profile is not None:
        base["found"] = True
        for key, value in [("params_b", profile.get("params_b")), ("layers", profile.get("layers")),
                           ("experts", profile.get("experts")), ("license", profile.get("license"))]:
            if base.get(key) is None:
                base[key] = value
        if base.get("architecture") is None:
            base["architecture"] = profile.get("family")
    return base


# Library views.

def path_index():
    """{id or catalog_id: absolute path of the library file}, rebuilt when the library reloads."""
    data = library.get()
    if _index_cache.get("data") is not data:
        root = library.library_dir()
        out = {}
        for row in library.read_index(os.path.join(root, "index.jsonl")):
            if row.get("path") and row.get("id"):
                path = os.path.join(root, *str(row["path"]).split("/"))
                out[row["id"]] = path
                if row.get("catalog_id"):
                    out[row["catalog_id"]] = path
        _index_cache["data"] = data
        _index_cache["map"] = out
        _tech_cache.clear()
    return _index_cache["map"]


def raw_of(part_id):
    """The library file for a part as a dict, or None."""
    path = path_index().get(part_id)
    if path is None:
        return None
    data = library.read_json(path)
    if isinstance(data, dict):
        return data
    return None


def method_techniques(raw, entry):
    """Technique ids (from techniques.js) that found a part, read from its method, source and fields."""
    found = set()
    signature = (raw or {}).get("signature") or {}
    method = str(signature.get("method") or "").lower()
    source = str((raw or {}).get("source") or entry.get("source") or "").lower()
    if "ablation" in method:
        found.add("ablation")
    if "activation" in method:
        found.add("activation-scan")
    if "attribution" in method or "attribution" in source:
        found.add("attribution")
    if "router pick rate" in method or "saliency" in method:
        found.add("saliency")
    if "skip-layer" in method or "layer map" in method or "layer_map" in source or entry.get("kind") == "inserted_layer":
        found.add("skill-map")
    if "weight" in method and ("diff" in method or "delta" in method):
        found.add("weight-delta")
    if "deletion tuning" in method:
        found.add("deletion-mask")
    if signature.get("weight_rationale") or ((raw or {}).get("skill") or {}).get("source") == "agent":
        found.add("weight-read")
    if signature.get("gate") is not None or signature.get("random_drops"):
        found.add("gate")
    if entry.get("source") == "atlas" and entry.get("kind") == "expert":
        found.add("saliency")
    return [item for item in TECHNIQUE_ORDER if item in found]


def techniques_of(entry):
    return method_techniques(raw_of(entry["id"]) or raw_of(entry["catalog_id"]), entry)


def model_parts(model):
    data = library.get()
    return [item for item in data["parts"] + data["masks"] if item.get("model") == model]


def model_facts(model, parts):
    """(params_b, layers, experts) from the library profile, with layers from the parts as a last resort."""
    profile = library_profile(model) or {}
    layers = profile.get("layers")
    if not layers:
        for item in library.get()["model_list"]:
            if item.get("hf_id") == model and item.get("layers"):
                layers = item["layers"]
    if not layers:
        top = 0
        for part in parts:
            pair = part.get("layers")
            if pair:
                top = max(top, pair[-1] + 1)
        layers = top or None
    return profile.get("params_b"), layers, profile.get("experts")


def skill_counts(parts):
    skills = {}
    for part in parts:
        for skill in part.get("skills") or []:
            skills[skill] = skills.get(skill, 0) + 1
    return dict(sorted(skills.items(), key=lambda pair: (-pair[1], pair[0])))


def model_summary(model, parts):
    params_b, layers, experts = model_facts(model, parts)
    by_kind = {}
    for part in parts:
        by_kind[part["kind"]] = by_kind.get(part["kind"], 0) + 1
    return {"model": model, "short": catalog_mod.donor_short(model), "params_b": params_b, "layers": layers,
            "experts": experts, "kind": "moe" if experts else "dense",
            "parts": {"total": len(parts), "by_kind": by_kind}, "skills": skill_counts(parts), "status": "mined"}


def all_models():
    groups = {}
    for part in library.get()["parts"] + library.get()["masks"]:
        if part.get("model"):
            groups.setdefault(part["model"], []).append(part)
    out = [model_summary(model, parts) for model, parts in groups.items()]
    out.sort(key=lambda item: (-item["parts"]["total"], item["model"]))
    return out


def model_detail(model):
    parts = model_parts(model)
    profile = library_profile(model)
    if len(parts) == 0 and profile is None:
        return None
    summary = model_summary(model, parts)
    palette = catalog_mod.build_palette(settings.backend())["parts"]
    palette_parts = [item for item in palette if item.get("donor") == model]
    layer_map = {}
    for item in palette_parts:
        pair = item.get("layers")
        if not pair:
            continue
        key = (pair[0], item["kind"])
        row = layer_map.setdefault(key, {"layer": pair[0], "kind": item["kind"], "count": 0, "to": pair[-1]})
        row["count"] += 1
        row["to"] = max(row["to"], pair[-1])
    found = set()
    for part in parts:
        for tech in techniques_of(part):
            found.add(tech)
    data = library.get()
    findings = [item for item in data["findings"] if model in (item.get("about") or {}).get("models", [])]
    recipes = []
    for recipe in data["recipes"]:
        if model in (recipe.get("models") or []) or recipe.get("host") == model:
            recipes.append({"id": recipe["id"], "title": recipe["title"], "status": recipe["status"],
                            "host": recipe.get("host"), "builds": recipe.get("builds"),
                            "skills": recipe.get("skills"), "evidence": recipe.get("evidence")})
    summary.update({"parts": palette_parts, "layer_map": sorted(layer_map.values(), key=lambda r: (r["layer"], r["kind"])),
                    "techniques": [item for item in TECHNIQUE_ORDER if item in found], "findings": findings,
                    "recipes": recipes, "part_counts": model_summary(model, parts)["parts"]})
    return summary


# Part detail.

def number(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        return None
    return value


def raw_block(raw, skill):
    """The raw numbers for a part: per-skill drops and the comparison with random parts."""
    signature = (raw or {}).get("signature") or {}
    effect = signature.get("effect") or {}
    per_skill = {}
    for key, value in effect.items():
        name, _, split = str(key).partition("|")
        control = name.endswith("__control")
        name = name[:-len("__control")] if control else name
        slot = per_skill.setdefault(name, {"search": None, "heldout": None, "control_search": None,
                                           "control_heldout": None})
        slot[("control_" if control else "") + split] = number(value)
    drops = [item for item in (signature.get("random_drops") or []) if number(item) is not None]
    mean = std = z = z_heldout = None
    if len(drops) > 0:
        mean = sum(drops) / len(drops)
        if len(drops) > 1:
            std = math.sqrt(sum((item - mean) ** 2 for item in drops) / (len(drops) - 1))
    main = per_skill.get(skill) or {}
    if std:
        if main.get("search") is not None:
            z = round((main["search"] - mean) / std, 3)
        if main.get("heldout") is not None:
            z_heldout = round((main["heldout"] - mean) / std, 3)
    return {"signature": per_skill or None, "z": z, "z_heldout": z_heldout,
            "random_mean": None if mean is None else round(mean, 4), "random_std": None if std is None else round(std, 4),
            "random_n": len(drops) if drops else None}


def main_skill(raw, item):
    signature = (raw or {}).get("signature") or {}
    for value in [signature.get("skill"), ((raw or {}).get("skill") or {}).get("name")]:
        if isinstance(value, str) and value:
            return value
    skills = item.get("skills") or []
    return skills[0] if skills else None


def part_detail(part_id):
    palette = catalog_mod.build_palette(settings.backend())
    item = None
    for candidate in palette["parts"] + palette["masks"]:
        if candidate["id"] == part_id:
            item = dict(candidate)
            break
    data = library.get()
    entry = None
    for candidate in data["parts"] + data["masks"]:
        if candidate["id"] == part_id or candidate["catalog_id"] == part_id:
            entry = candidate
            break
    if item is None and entry is None:
        return None
    if item is None:
        item = {"id": entry["catalog_id"], "kind": entry["kind"], "label": entry["title"], "donor": entry.get("model"),
                "layers": entry.get("layers"), "status": entry["status"], "evidence": entry["evidence"],
                "source": entry["source"], "skills": list(entry["skills"]), "usable": False,
                "findings": list(entry["findings"]), "stitches": list(entry["stitches"])}
    entry = entry or {"id": part_id, "catalog_id": part_id, "kind": item.get("kind"), "skills": item.get("skills") or []}
    raw = raw_of(entry["id"]) or raw_of(entry["catalog_id"])
    signature = (raw or {}).get("signature") or {}
    skill = main_skill(raw, item)
    skill_info = (raw or {}).get("skill")
    selection = None
    if raw is not None and (skill or raw.get("map_source") or raw.get("selection_data")):
        selection = {"skill": skill, "source": raw.get("map_source") or raw.get("source"),
                     "datasets": raw.get("selection_data") or signature.get("data")}
    recipes = []
    for recipe in data["recipes"]:
        ids = list(recipe.get("components") or [])
        for graft in (recipe.get("starting_program") or {}).get("grafts") or []:
            ids.append(graft.get("id"))
        if part_id in ids or entry["catalog_id"] in ids:
            recipes.append({"id": recipe["id"], "title": recipe["title"]})
    detail = dict(item)
    detail.update({
        "raw": raw_block(raw, skill),
        "num_units": signature.get("num_units", signature.get("n_units", entry.get("units"))),
        "tier": (raw or {}).get("tier"),
        "skill": skill_info if skill_info is not None else skill,
        "selection": selection,
        "units_file": entry.get("units_file"),
        "technique": method_techniques(raw, entry),
        "evidence": (raw or {}).get("evidence") if raw is not None else None,
        "evidence_line": item.get("evidence"),
        "findings": list(entry.get("findings") or item.get("findings") or []),
        "stitches": list(entry.get("stitches") or item.get("stitches") or []),
        "recipes": recipes,
    })
    return detail


def register(app):
    """Add the mining routes. Call before the catch-all /api route."""

    @app.get("/api/mine")
    def get_mine():
        return {"requests": [with_facts(i) for i in reversed(read_requests())], "models": all_models()}

    @app.get("/api/mine/lookup")
    def get_lookup(model: str = ""):
        return lookup(model.strip())

    @app.post("/api/mine")
    async def post_mine(request: Request):
        try:
            body = await request.json()
        except ValueError:
            return error_body("The request body must be JSON.", 400)
        try:
            return await run_in_threadpool(create_request, body)
        except ValueError as error:
            return error_body(str(error), 400)

    @app.get("/api/mine/req/{request_id}")
    def get_mine_request(request_id: str):
        item = get_request(request_id)
        if item is None:
            return error_body("unknown mining request", 404)
        return with_facts(item)

    @app.get("/api/models/{model:path}")
    def get_model(model: str):
        detail = model_detail(model)
        if detail is None:
            return error_body("no mined model with this id", 404)
        return detail

    @app.get("/api/parts/{part_id:path}")
    def get_part(part_id: str):
        detail = part_detail(part_id)
        if detail is None:
            return error_body("unknown part", 404)
        return detail
