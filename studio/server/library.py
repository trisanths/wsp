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
import logging
import os
import re
import threading

from studio.server import settings

log = logging.getLogger("uvicorn.error")

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
        "layer_list": layer_list(merged),
        "units_file": merged.get("units_file"),
        "units_remote": merged.get("units_remote"),
        "findings": [],
        "stitches": [],
    }


def layer_list(entry):
    """Every donor layer the entry names, as a list of ints (empty when none)."""
    layers = entry.get("layers")
    if layers is None:
        layers = entry.get("layer")
    if layers is None:
        return []
    if not isinstance(layers, list):
        layers = [layers]
    out = []
    for layer in layers:
        if isinstance(layer, int) and not isinstance(layer, bool):
            out.append(layer)
    return out


def model_info(data):
    """The fields the size estimate reads from a model profile, or None."""
    hf_id = data.get("hf_id")
    if not hf_id or not data.get("params_b"):
        return None
    return {"hf_id": hf_id, "params_b": data["params_b"],
            "active_params_b": data.get("active_params_b") or data["params_b"], "layers": data.get("layers"),
            "hidden": data.get("hidden"), "vocab_size": data.get("vocab"),
            "tokenizer_family": data.get("tokenizer_family"), "scores": data.get("scores") or {},
            "status": data.get("status"), "title": data.get("title"), "family": data.get("family"),
            "experts": data.get("experts"), "experts_per_token": data.get("experts_per_token"),
            "skills": list(data.get("skills") or []), "license": data.get("license"),
            "needs": data.get("needs") or {}}


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


FINDINGS_FILE = "findings/findings.jsonl"
STITCHES_FILE = "stitches/stitches.jsonl"
INDEX_KINDS = ["model", "deletion_mask", "mlp_cluster", "attn_heads", "expert", "block", "recipe", "finding", "stitch"]
UNBUILDABLE_PROGRAM_KEYS = ["taps", "tap_train", "bake", "carve", "base_source"]
LAYER_WORDS = re.compile(r"\blayers?\s+(\d+)(?:\s*(?:\.\.|to|-)\s*(\d+))?", re.IGNORECASE)


def read_index_checked(path, failed):
    """The index rows. A line that is not a JSON object goes in `failed` with its line number."""
    rows = []
    try:
        with open(path, encoding="utf-8") as handle:
            for number, line in enumerate(handle, 1):
                line = line.strip()
                if line == "":
                    continue
                try:
                    row = json.loads(line)
                except ValueError as error:
                    failed.append({"id": "line " + str(number), "kind": None, "reason": "not JSON: " + str(error)})
                    continue
                if not isinstance(row, dict):
                    failed.append({"id": "line " + str(number), "kind": None, "reason": "not an object"})
                    continue
                rows.append(row)
    except OSError as error:
        failed.append({"id": "index.jsonl", "kind": None, "reason": "cannot read the index: " + str(error)})
    return rows


def read_jsonl_by_id(path):
    """{id: record} for a jsonl file, or {} when it cannot be read. Bad lines are skipped."""
    out = {}
    for row in read_index(path):
        if isinstance(row, dict) and row.get("id"):
            out[row["id"]] = row
    return out


def clean_status(value):
    if value in STATUSES:
        return value
    return "candidate"


def model_entry(row, data):
    """One model for the models list. A model with no hf_id or params_b is kept and flagged."""
    merged = dict(row)
    merged.update(data or {})
    missing = []
    for field in ["hf_id", "params_b", "layers", "hidden"]:
        if not merged.get(field):
            missing.append(field)
    hf_id = merged.get("hf_id")
    if not hf_id:
        hf_id = (merged.get("models") or [None])[0]
    scores = merged.get("scores") or {}
    return {
        "id": merged["id"], "hf_id": hf_id, "title": merged.get("title") or merged["id"],
        "status": clean_status(merged.get("status")), "status_reason": merged.get("status_reason"),
        "family": merged.get("family"), "params_b": merged.get("params_b"),
        "active_params_b": merged.get("active_params_b"), "layers": merged.get("layers"),
        "hidden": merged.get("hidden"), "vocab": merged.get("vocab"), "license": merged.get("license"),
        "experts": merged.get("experts"), "skills": list(merged.get("skills") or []),
        "added_by": merged.get("added_by"), "n_scores": len(scores),
        "incomplete": "hf_id" in missing or "params_b" in missing,
        "missing": missing, "findings": [],
    }


def finding_entry(row):
    """A finding for the findings list: the claim, the numbers and what it is about."""
    return {
        "id": row["id"], "title": row.get("title") or row["id"], "claim": row.get("claim") or "",
        "setup": row.get("setup"), "numbers": row.get("numbers") or {},
        "numbers_source": row.get("numbers_source"), "status": clean_status(row.get("status")),
        "status_reason": row.get("status_reason"), "source": row.get("source"),
        "skills": list(row.get("skills") or []), "evidence": evidence_line(row),
        "about": {"parts": [], "models": list(row.get("models") or [])},
    }


def penalty_text(entry):
    """The stitch fit numbers as one short line: the GSM8K penalty before and after, when they exist."""
    penalty = entry.get("penalty") or {}
    before = ((penalty.get("before") or {}).get("gsm8k") or {}).get("penalty")
    after = ((penalty.get("after") or {}).get("gsm8k") or {}).get("penalty")
    if before is None or after is None:
        return None
    return "GSM8K penalty " + short_number(before) + " before, " + short_number(after) + " after"


def stitch_entry(row):
    """A stitch: which host layers read which donor layer, how it was fit, and the fit numbers."""
    fit = row.get("ridge_fit") or {}
    return {
        "id": row["id"], "title": row.get("title") or row["id"], "status": clean_status(row.get("status")),
        "status_reason": row.get("status_reason"), "host": row.get("host"), "donor": row.get("donor"),
        "host_layers": row.get("host_layers"), "donor_layer": row.get("donor_layer"),
        "stitch_kind": row.get("stitch_kind"), "method": row.get("method"), "objective": row.get("objective"),
        "steps": row.get("steps"), "sink": row.get("sink"), "params": row.get("params"),
        "penalty": row.get("penalty"), "ridge_r2": fit.get("r2"), "fit": penalty_text(row),
        "evidence": evidence_line(row), "source": row.get("source"), "skills": list(row.get("skills") or []),
    }


def builds_text(program, merged):
    """One line that says what a recipe builds, from its program."""
    if not isinstance(program, dict):
        return merged.get("title") or ""
    bits = []
    host = str(program.get("host") or (program.get("bake") or {}).get("host") or "")
    if host != "":
        bits.append("host " + host.split("/")[-1])
    if program.get("carve"):
        bits.append("carved to " + str(program["carve"].get("keep")) + " layers")
    for graft in program.get("grafts") or []:
        bits.append("graft " + str(graft.get("id")) + " at layer " + str(graft.get("at")))
    for tap in program.get("taps") or []:
        bits.append("tap " + str(tap.get("donor")).split("/")[-1] + " layer " + str(tap.get("donor_layer")))
    if program.get("lora"):
        bits.append("LoRA rank " + str((program["lora"] or {}).get("rank")))
    if program.get("bake"):
        bits.append("bake a tapped teacher into the host")
    heal = program.get("heal")
    if isinstance(heal, dict):
        bits.append("heal " + str(heal.get("steps")) + " steps")
    return ", ".join(bits)


def recipe_entry(row, data):
    """A recipe: a program that was run, with its measured outcome and a starting program when one can be made."""
    merged = dict(row)
    merged.update(data or {})
    program = merged.get("program")
    starting = None
    why_not = None
    if not isinstance(program, dict):
        why_not = "The library file has no program."
    else:
        blocked = []
        for key in UNBUILDABLE_PROGRAM_KEYS:
            if program.get(key):
                blocked.append(key)
        if len(blocked) > 0:
            why_not = "The builder cannot hold " + ", ".join(blocked) + " yet."
        elif not program.get("host"):
            why_not = "The program has no host."
        else:
            starting = program
    return {
        "id": merged["id"], "title": merged.get("title") or merged["id"], "status": clean_status(merged.get("status")),
        "status_reason": merged.get("status_reason"), "host": merged.get("host"),
        "builds": builds_text(program, merged), "components": list(merged.get("components") or []),
        "models": list(merged.get("models") or []), "skills": list(merged.get("skills") or []),
        "evidence": evidence_line(merged), "source": merged.get("source"), "text": merged.get("text"),
        "syntax_gap": merged.get("syntax_gap"), "starting_program": starting,
        "no_start_reason": None if why_not is None else why_not.strip(),
    }


def layer_mentions(text):
    """Layer numbers and ranges a text names, as a set of (first, last)."""
    found = set()
    for match in LAYER_WORDS.finditer(text):
        first = int(match.group(1))
        last = first
        if match.group(2) is not None:
            last = int(match.group(2))
        found.add((first, last))
    return found


KIND_WORDS = {"mlp_cluster": ["cluster", "neuron", "mlp"], "attn_heads": ["head"], "expert": ["expert"]}


def names_layers(part, pair, mentions, lowered):
    """True when the finding text names the part's layer (or layer range) and, for a one-layer part, its kind."""
    if (pair[0], pair[1]) not in mentions:
        return False
    if part["kind"] == "block":
        return True
    for word in KIND_WORDS.get(part["kind"], []):
        if word in lowered:
            return True
    return False


def link_findings(findings, parts, masks, models):
    """Attach each finding to the parts and models it names.

    A finding names a part when the part id appears in its text, or when the
    part's model is one of the finding's models and the text names the part's
    layer (or layer range).
    """
    by_model = {}
    for part in parts:
        if part["kind"] != "inserted_layer":
            by_model.setdefault(part.get("model"), []).append(part)
    for finding in findings:
        text = finding["title"] + " " + finding["claim"] + " " + str(finding.get("setup") or "")
        mentions = layer_mentions(text)
        lowered = text.lower()
        found = {}
        for model in finding["about"]["models"]:
            for part in by_model.get(model, []):
                pair = part.get("layers")
                if pair is None:
                    continue
                if part["id"] in text or part["catalog_id"] in text or names_layers(part, pair, mentions, lowered):
                    found[part["id"]] = part
        for part in masks:
            if part["id"] in text:
                found[part["id"]] = part
        for part_id, part in sorted(found.items()):
            finding["about"]["parts"].append(part_id)
            part["findings"].append({"id": finding["id"], "title": finding["title"]})
        for model in models:
            if model["hf_id"] in finding["about"]["models"]:
                model["findings"].append({"id": finding["id"], "title": finding["title"]})


def link_stitches(stitches, parts):
    """Attach a stitch to the parts of its donor model that sit at its donor layer."""
    for stitch in stitches:
        layer = stitch["donor_layer"]
        for part in parts:
            pair = part.get("layers")
            if part["kind"] == "inserted_layer" or pair is None or part.get("model") != stitch["donor"]:
                continue
            if isinstance(layer, int) and pair[0] <= layer <= pair[1]:
                part["stitches"].append({"id": stitch["id"], "title": stitch["title"]})


def load_row(row, root, side, out):
    """Read one index row into `out`. Raises when the row cannot be read; the caller records that."""
    kind = row.get("kind")
    row_id = row.get("id")
    if not row_id or kind not in INDEX_KINDS:
        raise ValueError("the row has no id, or a kind the library does not know: " + str(kind))
    if kind == "finding":
        record = side[FINDINGS_FILE].get(row_id)
        if record is None:
            raise ValueError("no line with this id in " + FINDINGS_FILE)
        out["findings"].append(finding_entry(record))
        return
    if kind == "stitch":
        record = side[STITCHES_FILE].get(row_id)
        if record is None:
            raise ValueError("no line with this id in " + STITCHES_FILE)
        out["stitches"].append(stitch_entry(record))
        return
    path = row.get("path") or ""
    if path == "":
        raise ValueError("the row has no path")
    data = read_json(os.path.join(root, path))
    if not isinstance(data, dict):
        raise ValueError("cannot read " + path)
    if kind == "model":
        out["model_list"].append(model_entry(row, data))
        info = model_info(data)
        if info is not None:
            info["status"] = row.get("status")
            out["models"][info["hf_id"]] = info
    elif kind == "recipe":
        out["recipes"].append(recipe_entry(row, data))
    elif kind == "deletion_mask":
        out["masks"].append(component_entry(row, data))
    else:
        out["parts"].append(component_entry(row, data))


def load():
    """Read the library from disk.

    Returns parts, masks, models (by hf_id, complete profiles only), model_list (all models),
    recipes, findings, stitches, counts (rows loaded, by kind), failed (rows that did not load,
    each with its reason), rows (index rows read), path and atlas. One bad row never stops the rest.
    """
    root = library_dir()
    failed = []
    rows = read_index_checked(os.path.join(root, "index.jsonl"), failed)
    side = {FINDINGS_FILE: read_jsonl_by_id(os.path.join(root, FINDINGS_FILE)),
            STITCHES_FILE: read_jsonl_by_id(os.path.join(root, STITCHES_FILE))}
    out = {"parts": [], "masks": [], "models": {}, "model_list": [], "recipes": [], "findings": [], "stitches": []}
    counts = {}
    seen = set()
    for row in rows:
        try:
            if row.get("id") in seen:
                raise ValueError("duplicate id")
            load_row(row, root, side, out)
            seen.add(row.get("id"))
            counts[row["kind"]] = counts.get(row["kind"], 0) + 1
        except Exception as error:
            failed.append({"id": str(row.get("id")), "kind": row.get("kind"), "reason": str(error) or error.__class__.__name__})
    atlas = atlas_dir()
    atlas_layers = 0
    if atlas is not None:
        try:
            for entry in atlas_block_entries(atlas):
                out["parts"].append(entry)
                if entry["kind"] == "inserted_layer":
                    atlas_layers = atlas_layers + 1
        except Exception as error:
            failed.append({"id": "atlas/entries", "kind": "inserted_layer", "reason": str(error)})
    counts["inserted_layer"] = atlas_layers
    for part in out["parts"] + out["masks"]:
        part["findings"] = []
        part["stitches"] = []
    link_findings(out["findings"], out["parts"], out["masks"], out["model_list"])
    link_stitches(out["stitches"], out["parts"])
    out.update({"counts": counts, "failed": failed, "rows": len(rows), "path": root, "atlas": atlas})
    return out


def counts_line(data):
    """The one line the server logs at start: counts by kind and every row that failed."""
    pieces = []
    for kind in INDEX_KINDS + ["inserted_layer"]:
        pieces.append(kind + " " + str(data["counts"].get(kind, 0)))
    text = "library: " + str(data["rows"]) + " index rows, loaded " + ", ".join(pieces)
    text = text + ", failed " + str(len(data["failed"]))
    if len(data["failed"]) > 0:
        details = []
        for item in data["failed"][:20]:
            details.append(str(item["id"]) + " (" + str(item["reason"]) + ")")
        text = text + ": " + "; ".join(details)
    return text


def get(reload=False):
    """The library, read once and kept. reload=True reads it again and logs the counts."""
    with _lock:
        key = (library_dir(), atlas_dir())
        if reload or _cache.get("key") != key:
            _cache["key"] = key
            _cache["data"] = load()
            log.info(counts_line(_cache["data"]))
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
