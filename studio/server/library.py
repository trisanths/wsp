"""The Sapien library as the studio sees it: component entries for the palette and model profiles.

The library is read once, when the server starts (or on the first request in
a test). It holds library/index.jsonl, the component files it lists, and the
model profiles in library/models. When atlas/entries/*.json exists, the
atlas layer maps add donor blocks and the expert maps add experts.

Palette kinds: block, attn_heads, mlp_cluster, expert, inserted_layer. Every
palette entry has a status (candidate, validated or rejected) and one
evidence line, so a person sees what was measured before using a part.
"""

import glob
import json
import os
import threading

from studio.server import settings

PALETTE_KINDS = ["block", "attn_heads", "mlp_cluster", "expert"]
STATUSES = ["candidate", "validated", "rejected"]
KIND_NAMES = {"block": "block", "attn_heads": "attention heads", "mlp_cluster": "neuron cluster",
              "expert": "expert (routed)", "inserted_layer": "inserted layers"}
MAX_EVIDENCE_CHARS = 260

_lock = threading.Lock()
_cache = {}


def library_dir():
    """The library folder. WSP_STUDIO_LIBRARY overrides it, so tests can use a small one."""
    path = os.environ.get("WSP_STUDIO_LIBRARY", "").strip()
    if path == "":
        path = os.path.join(settings.REPO_ROOT, "library")
    return path


def atlas_dir():
    """The atlas entries folder, or None when it does not exist yet."""
    path = os.environ.get("WSP_STUDIO_ATLAS", "").strip()
    if path == "":
        path = os.path.join(settings.REPO_ROOT, "atlas", "entries")
    if os.path.isdir(path):
        return path
    return None


def read_json(path):
    try:
        with open(path, encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, ValueError):
        return None


def read_index(path):
    rows = []
    try:
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line == "":
                    continue
                try:
                    rows.append(json.loads(line))
                except ValueError:
                    continue
    except OSError:
        return []
    return rows


def short_number(value):
    """0.4033 becomes 0.403, 12 stays 12."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, int):
        return str(value)
    return ("%.3f" % value).rstrip("0").rstrip(".")


def evidence_line(entry):
    """One line that says what was measured, or why the entry has its status."""
    items = entry.get("evidence") or []
    if len(items) > 0:
        first = items[0]
        text = str(first.get("claim") or "measured")
        metric = first.get("metric")
        if metric:
            text = text + " (" + str(metric) + ")"
        if first.get("value") is not None:
            text = text + ": " + short_number(first["value"])
            if first.get("control_value") is not None:
                text = text + " against " + short_number(first["control_value"])
                if first.get("control"):
                    text = text + " for " + str(first["control"])
        source = first.get("source")
        if source:
            text = text + ", " + str(source)
    else:
        reason = entry.get("status_reason")
        if reason:
            text = "no measurement yet: " + str(reason)
        else:
            text = "no evidence recorded"
    if len(text) > MAX_EVIDENCE_CHARS:
        text = text[:MAX_EVIDENCE_CHARS - 3].rstrip() + "..."
    return text


def layers_of(entry):
    """The donor layer range as [first, last], or None."""
    layers = entry.get("layers")
    if layers is None:
        layer = entry.get("layer")
        if layer is None:
            return None
        layers = layer if isinstance(layer, list) else [layer]
    if not isinstance(layers, list) or len(layers) == 0:
        return None
    return [layers[0], layers[-1]]


def units_of(entry):
    signature = entry.get("signature") or {}
    value = signature.get("n_units")
    if value is None:
        value = entry.get("num_units")
    return value


TAP_ONLY_REASON = ("The library measured this entry through a tap, which runs a second model at inference. Sapien does "
                   "not build taps. Mine and score it as an in-stream part before a program uses it.")


def is_tap_only(merged):
    """True for entries that only exist as tap read points or tap evidence."""
    title = str(merged.get("title") or "").lower()
    return str(merged.get("id", "")).startswith("tapread_") or "tap read point" in title or "read by taps" in title


def component_entry(row, data):
    """One palette-ready dict from an index row and its component file."""
    merged = dict(row)
    merged.update(data or {})
    status = merged.get("status")
    if status not in STATUSES:
        status = "candidate"
    return {
        "id": merged["id"],
        "catalog_id": merged.get("catalog_id") or merged["id"],
        "kind": merged["kind"],
        "title": merged.get("title") or merged["id"],
        "status": status,
        "status_reason": merged.get("status_reason"),
        "evidence": evidence_line(merged),
        "model": merged.get("model"),
        "layers": layers_of(merged),
        "units": units_of(merged),
        "params_b": merged.get("params_b"),
        "skills": list(merged.get("skills") or []),
        "added_by": merged.get("added_by"),
        "source": "library",
        "tap_only": is_tap_only(merged),
    }


def model_info(data):
    """The fields the size estimate reads from a model profile, or None."""
    hf_id = data.get("hf_id")
    if not hf_id or not data.get("params_b"):
        return None
    return {"hf_id": hf_id, "params_b": data["params_b"],
            "active_params_b": data.get("active_params_b") or data["params_b"], "layers": data.get("layers"),
            "hidden": data.get("hidden"), "vocab_size": data.get("vocab"),
            "tokenizer_family": data.get("tokenizer_family"), "scores": data.get("scores") or {},
            "status": data.get("status"), "title": data.get("title")}


def merge_atlas(out, by_id, entry):
    """Add an atlas entry, or add its skill to the entry with the same id (one block can serve several skills)."""
    found = by_id.get(entry["id"])
    if found is None:
        by_id[entry["id"]] = entry
        out.append(entry)
        return
    for skill in entry["skills"]:
        if skill not in found["skills"]:
            found["skills"].append(skill)
            found["evidence"] = found["evidence"] + "; also chosen by the " + skill + " map"


def atlas_block_entries(path):
    """Inserted-layer entries from the atlas layer maps and expert entries from its expert maps."""
    out = []
    by_id = {}
    for file_path in sorted(glob.glob(os.path.join(path, "*.json"))):
        data = read_json(file_path)
        if not isinstance(data, dict) or not data.get("model"):
            continue
        model = data["model"]
        name = os.path.basename(file_path)[:-len(".json")]
        for skill, layer_map in sorted((data.get("layer_maps") or {}).items()):
            blocks = layer_map.get("blocks") or []
            mass = layer_map.get("block_mass") or []
            held = (layer_map.get("heldout") or {}).get("check_mass")
            for index, block in enumerate(blocks):
                if not isinstance(block, list) or len(block) != 2:
                    continue
                share = mass[index] if index < len(mass) else None
                text = "atlas " + skill + " layer map, " + model
                if share is not None:
                    text = text + ": the block holds " + short_number(share) + " of the positive layer-drop mass"
                    text = text + ", measured on the texts that chose it, so optimistic"
                if held is not None:
                    text = text + ". Blocks chosen on half the texts held " + short_number(held) + " of the mass on the other half"
                part_id = "block/" + model + "/" + str(block[0]) + "-" + str(block[1])
                merge_atlas(out, by_id, {
                    "id": part_id, "catalog_id": part_id,
                    "kind": "inserted_layer", "title": model + " layers " + str(block[0]) + " to " + str(block[1]) +
                    " for " + skill, "status": "candidate", "status_reason": None, "evidence": text,
                    "model": model, "layers": [block[0], block[1]], "units": None, "params_b": None,
                    "skills": [skill], "added_by": "atlas", "source": "atlas", "atlas_entry": name})
        experts = data.get("expert_maps") or {}
        for skill, expert_map in sorted(experts.items()) if isinstance(experts, dict) else []:
            per_layer = expert_map
            if not isinstance(per_layer, dict):
                continue
            for layer, ids in sorted(per_layer.items(), key=lambda pair: str(pair[0])):
                if not ids:
                    continue
                part_id = "expert/" + model + "/" + str(layer)
                merge_atlas(out, by_id, {
                    "id": part_id, "catalog_id": part_id,
                    "kind": "expert", "title": model + " layer " + str(layer) + " experts for " + skill,
                    "status": "candidate", "status_reason": None,
                    "evidence": "atlas " + skill + " expert map: " + str(len(ids)) + " experts picked more often on " +
                    skill + " text than on general text", "model": model, "layers": [int(layer), int(layer)],
                    "units": len(ids), "params_b": None, "skills": [skill], "added_by": "atlas",
                    "source": "atlas", "atlas_entry": name})
    return out


def load():
    """Read the library from disk. Returns {"parts": [...], "masks": [...], "models": {...}, "path": ...}."""
    root = library_dir()
    rows = read_index(os.path.join(root, "index.jsonl"))
    parts = []
    masks = []
    models = {}
    for row in rows:
        kind = row.get("kind")
        if kind not in PALETTE_KINDS and kind not in ("deletion_mask", "model"):
            continue
        data = read_json(os.path.join(root, row.get("path", ""))) if row.get("path") else None
        if kind == "model":
            info = model_info(data or {})
            if info is not None:
                info["status"] = row.get("status")
                models[info["hf_id"]] = info
            continue
        entry = component_entry(row, data)
        if kind == "deletion_mask":
            masks.append(entry)
        else:
            parts.append(entry)
    atlas = atlas_dir()
    if atlas is not None:
        parts.extend(atlas_block_entries(atlas))
    return {"parts": parts, "masks": masks, "models": models, "path": root, "atlas": atlas}


def get(reload=False):
    """The library, read once and kept. reload=True reads it again."""
    with _lock:
        key = (library_dir(), atlas_dir())
        if reload or _cache.get("key") != key:
            _cache["key"] = key
            _cache["data"] = load()
        return _cache["data"]


def by_catalog_id():
    """Library entries keyed by the id the compiler catalog uses."""
    out = {}
    data = get()
    for entry in data["parts"] + data["masks"]:
        out[entry["catalog_id"]] = entry
    return out


def status_counts():
    counts = {"candidate": 0, "validated": 0, "rejected": 0}
    for entry in get()["parts"]:
        counts[entry["status"]] += 1
    return counts
