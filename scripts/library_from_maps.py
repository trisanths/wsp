"""Turn the mined skill maps into library entries.

person_a/mine/maps.py writes one JSON file per model to atlas/pool_entries/. Each file says
where every skill lives in that model: which decoder layers (blocks) and which MoE experts.
This script reads those files and writes library entries:

  block   one entry per model, skill, and block in layer_maps[skill].blocks
  expert  one entry per model, skill, and MoE layer in expert_maps[skill], with a .pt index file
          {kind, model, layer, indices} next to the entry

Every entry is a candidate. The numbers come from the donor alone (layer skips and router
counts), so they say where a skill lives. They do not say that the part helps a host.

The script uses the helper functions of scripts/library.py to validate and write entries, and it
writes index.jsonl once at the end. It never changes or removes an entry that exists. Run it
again after new map files arrive.

  .venv/bin/python scripts/library_from_maps.py write --entries-dir ../wsp-mine/atlas/pool_entries
  .venv/bin/python scripts/library_from_maps.py find --kind block --skill law --max-params 8
"""

import argparse
import glob
import importlib.util
import json
import os
import re
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_ENTRIES_DIR = os.path.join(REPO, "atlas", "pool_entries")
DEFAULT_LIBRARY = os.path.join(REPO, "library")
ADDED_BY = "agent: Trisanth-libmaps"
MAPS_SCRIPT = "person_a/mine/maps.py"

MIN_SPLIT_HALF_R = 0.5
MIN_CHECK_MASS = 0.2
MIN_EXPERTS = 2

SEARCH_SPLIT = "search half of the skill text (the half that chose the part)"
HELDOUT_SPLIT = "held-out half of the skill text (checks the choice)"
DONOR_ONLY = "measured inside the donor only, not a gain on a host"

REASONS = [
    "split_half_r missing",
    "split_half_r below minimum",
    "check_mass below minimum",
    "fewer experts than minimum",
    "id already in library",
]


def load_library_module():
    path = os.path.join(REPO, "scripts", "library.py")
    spec = importlib.util.spec_from_file_location("library_script", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


lib = load_library_module()


def slug(text):
    return re.sub(r"[^A-Za-z0-9_.\-]+", "_", str(text)).strip("_").lower()


def round_or_none(value, places=5):
    if value is None:
        return None
    return round(float(value), places)


def mean_or_none(values):
    values = [v for v in values if v is not None]
    if not values:
        return None
    return round(sum(values) / len(values), 5)


def item(claim, metric, value, control_value, control, split, source, note=None):
    """One evidence item with every field the rules need. null where a field does not apply."""
    out = {"claim": claim, "metric": metric, "value": value, "control_value": control_value,
           "split": split, "seeds": None, "source": source, "added_by": ADDED_BY}
    if control is not None:
        out["control"] = control
    if note is not None:
        out["note"] = note
    return out


def model_facts(data):
    """The per-model facts a picker filters on, as top-level entry fields."""
    return {"params_b": data.get("params_b"), "n_layers": data.get("layers"), "hidden": data.get("hidden"),
            "family": data.get("family"), "architecture": data.get("architecture"),
            "license": data.get("license"), "revision": data.get("revision")}


def skill_text(data, skill):
    selection = data.get("selection_data") or {}
    return selection.get(skill)


def unlocked(entry):
    # generated false: scripts/library.py build never rewrites these entries
    entry["generated"] = False
    return entry


def block_entry(data, skill, index, source, tag):
    lm = data["layer_maps"][skill]
    first, last = lm["blocks"][index]
    scores = lm.get("scores") or []
    influence = lm.get("influence") or []
    held = lm.get("heldout") or {}
    check_list = held.get("check_mass") or []
    check = check_list[index] if index < len(check_list) else None
    mass_list = lm.get("block_mass") or []
    mass = mass_list[index] if index < len(mass_list) else None
    r = held.get("split_half_r")
    window = scores[first:last + 1]
    summed = round_or_none(sum(window), 5) if len(window) == last - first + 1 else None
    model = data["model"]
    short = lib.short_model(model)
    layers = list(range(first, last + 1))
    where = "layer " + str(first) if first == last else "layers " + str(first) + " to " + str(last)
    text = skill_text(data, skill)
    signature = {
        "method": "skip-layer map: loss increase on skill text when a decoder layer is skipped, minus the same "
                  "on general_text (nats); the block is the shortest window that holds 60% of the positive "
                  "score on the search half",
        "skill": skill,
        "data": text,
        "first_layer": first, "last_layer": last, "n_layers_in_block": last - first + 1,
        "block_rank_in_skill": index + 1,
        "block_mass": mass, "check_mass": check, "split_half_r": r, "summed_layer_score": summed,
        "layer_scores": [round_or_none(v) for v in window],
        "layer_influence": [round_or_none(v) for v in influence[first:last + 1]],
        "scope": DONOR_ONLY,
    }
    items = []
    claim = ("skipping " + where + " of " + short + " raises the loss on '" + skill +
             "' text more than on general text; " + DONOR_ONLY)
    items.append(item(claim, "summed layer score over the block, nats (skill loss increase minus general loss "
                             "increase)", summed, None,
                      "general_text loss increase, subtracted layer by layer", SEARCH_SPLIT, source))
    items.append(item("share of the positive layer score that falls inside the block (" + where + ")",
                      "block_mass, fraction of the model's positive layer score", mass, None, None,
                      SEARCH_SPLIT, source))
    items.append(item("the same share, recomputed on the other half of the skill text for the same layers",
                      "check_mass, fraction of the positive layer score", check, None, None,
                      HELDOUT_SPLIT, source))
    items.append(item("the per-layer score map from one half of the text agrees with the map from the other half",
                      "split_half_r, Pearson r over layers (whole map, same value for every block of the skill)",
                      r, None, None, "search half against held-out half", source))
    entry_id = "map_block_" + tag + "_" + slug(skill) + "_l" + str(first) + "_" + str(last)
    title = (short + " " + where + ", " + skill + " block (donor skill map)")
    fields = {"model": model, "layer": first if first == last else None, "layers": layers,
              "units_file": None, "units_remote": None, "signature": signature,
              "selection_data": text, "map_source": MAPS_SCRIPT}
    fields.update(model_facts(data))
    return unlocked(lib.entry(entry_id, "block", title, "candidate", ADDED_BY, source, [skill], [model], items, fields))


def expert_entry(data, skill, layer, ids, rows, source, tag, units_rel):
    model = data["model"]
    short = lib.short_model(model)
    text = skill_text(data, skill)
    signature = {
        "method": "router pick rate on skill text over general_text, per expert; an expert is listed when its "
                  "log ratio is positive on both halves; the top experts by search log ratio are kept",
        "skill": skill, "data": text, "n_units": len(ids), "experts": ids,
        "details": rows, "scope": DONOR_ONLY,
    }
    items = []
    if rows:
        skill_freq = mean_or_none([r.get("skill_freq") for r in rows])
        general_freq = mean_or_none([r.get("general_freq") for r in rows])
        search_ratio = mean_or_none([r.get("log_ratio") for r in rows])
        held_ratio = mean_or_none([r.get("heldout_log_ratio") for r in rows])
        about = str(len(ids)) + " experts at layer " + str(layer) + " of " + short
        items.append(item("router pick rate of " + about + " on '" + skill + "' text against general text; " +
                          DONOR_ONLY, "mean pick rate over the listed experts, as recorded by maps.py",
                          skill_freq, general_freq, "the same experts on general_text", SEARCH_SPLIT, source))
        items.append(item("log of the ratio of skill pick rate to general pick rate for " + about,
                          "mean log ratio over the listed experts, nats", search_ratio, None, None,
                          SEARCH_SPLIT, source))
        items.append(item("the same log ratio on the other half of the text; every listed expert is positive on "
                          "both halves", "mean held-out log ratio over the listed experts, nats", held_ratio,
                          None, None, HELDOUT_SPLIT, source))
    entry_id = "map_expert_" + tag + "_" + slug(skill) + "_l" + str(layer)
    title = short + " layer " + str(layer) + ", " + str(len(ids)) + " experts for " + skill + " (donor skill map)"
    fields = {"model": model, "layer": layer, "layers": [layer], "units_file": units_rel, "units_remote": None,
              "signature": signature, "selection_data": text, "map_source": MAPS_SCRIPT}
    fields.update(model_facts(data))
    return unlocked(lib.entry(entry_id, "expert", title, "candidate", ADDED_BY, source, [skill], [model], items, fields))


def plan_model(data, source, args, counts):
    """Return the list of ("block" or "expert", build function arguments) for one map file."""
    plan = []
    model = data["model"]
    tag = slug(lib.model_key(model))
    layer_maps = data.get("layer_maps") or {}
    for skill in sorted(layer_maps):
        lm = layer_maps[skill]
        held = lm.get("heldout") or {}
        r = held.get("split_half_r")
        check_list = held.get("check_mass") or []
        for index in range(len(lm.get("blocks") or [])):
            if r is None:
                counts["split_half_r missing"] += 1
                continue
            if r < args.min_split_half_r:
                counts["split_half_r below minimum"] += 1
                continue
            check = check_list[index] if index < len(check_list) else None
            if check is None or check < args.min_check_mass:
                counts["check_mass below minimum"] += 1
                continue
            plan.append(("block", (data, skill, index, source, tag)))
    expert_maps = data.get("expert_maps") or {}
    details = data.get("expert_details") or {}
    for skill in sorted(expert_maps):
        layers = expert_maps[skill] or {}
        for layer_key in sorted(layers, key=int):
            ids = list(layers[layer_key])
            if len(ids) < args.min_experts:
                counts["fewer experts than minimum"] += 1
                continue
            rows = (details.get(skill) or {}).get(layer_key) or []
            plan.append(("expert", (data, skill, int(layer_key), ids, rows, source, tag)))
    return plan


def write_pt(path, kind, model, layer, indices):
    import torch
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({"kind": kind, "model": model, "layer": layer, "indices": list(indices)}, path)


def write_entries(args):
    library = args.library
    files = sorted(glob.glob(os.path.join(args.entries_dir, "*.json")))
    existing = {}
    for data, rel in lib.load_entries(library):
        if isinstance(data, dict) and data.get("id") is not None:
            existing[data["id"]] = rel
    report = {}
    total_written = {"block": 0, "expert": 0}
    total_skipped = {reason: 0 for reason in REASONS}
    written_ids = []
    for path in files:
        base = os.path.basename(path)
        try:
            with open(path) as f:
                data = json.load(f)
        except ValueError as error:
            print("skip file " + base + ": bad JSON: " + str(error))
            continue
        if not isinstance(data, dict) or "model" not in data:
            print("skip file " + base + ": no 'model' field")
            continue
        source = "atlas/pool_entries/" + base + " (" + MAPS_SCRIPT + ")"
        counts = {reason: 0 for reason in REASONS}
        written = {"block": 0, "expert": 0}
        for kind, call in plan_model(data, source, args, counts):
            if kind == "block":
                entry = block_entry(*call)
            else:
                data_, skill, layer, ids, rows, src, tag = call
                key = lib.model_key(data_["model"])
                units_rel = "components/" + key + "/map_expert_" + tag + "_" + slug(skill) + "_l" + str(layer) + ".pt"
                entry = expert_entry(data_, skill, layer, ids, rows, src, tag, units_rel)
            if entry["id"] in existing:
                counts["id already in library"] += 1
                continue
            rel = lib.entry_path(entry)
            if kind == "expert" and not args.dry_run:
                write_pt(os.path.join(library, entry["units_file"]), "expert", entry["model"], entry["layer"],
                         entry["signature"]["experts"])
            errors = []
            if kind == "block" or not args.dry_run:
                lib.validate_entry(entry, rel, library, errors)
            if errors:
                raise SystemExit("not valid: " + entry["id"] + "\n  " + "\n  ".join(errors))
            if not args.dry_run:
                lib.write_text(os.path.join(library, rel), lib.json_text(entry))
            existing[entry["id"]] = rel
            written[kind] += 1
            written_ids.append(entry["id"])
        report[data["model"]] = (written, counts)
        for kind in written:
            total_written[kind] += written[kind]
        for reason in counts:
            total_skipped[reason] += counts[reason]
    if not args.dry_run and written_ids:
        lib.write_index(library, quiet=True)
    print("model | block written | expert written | skipped")
    for model in sorted(report):
        written, counts = report[model]
        skipped = ", ".join(r + " " + str(n) for r, n in counts.items() if n)
        print(model + " | " + str(written["block"]) + " | " + str(written["expert"]) + " | " + (skipped or "none"))
    prefix = "dry run, would write " if args.dry_run else "wrote "
    print(prefix + str(total_written["block"]) + " block and " + str(total_written["expert"]) +
          " expert entries from " + str(len(files)) + " map files")
    for reason in REASONS:
        print("skipped, " + reason + ": " + str(total_skipped[reason]))
    return 0


def find_entries(args):
    """List map entries by kind, skill, model text, and size. Reads the entry files."""
    rows = lib.query(args.library, kind=args.kind, skill=args.skill, model=args.model, status=args.status)
    out = []
    for row in rows:
        data = lib.read_json(os.path.join(args.library, row["path"]))
        if not str(data.get("id", "")).startswith("map_"):
            continue
        params = data.get("params_b")
        if args.max_params is not None and (params is None or params > args.max_params):
            continue
        if args.min_params is not None and (params is None or params < args.min_params):
            continue
        if args.max_hidden is not None and (data.get("hidden") is None or data.get("hidden") > args.max_hidden):
            continue
        if args.family is not None and args.family.lower() not in str(data.get("family")).lower():
            continue
        if args.license is not None and args.license.lower() not in str(data.get("license")).lower():
            continue
        out.append(data)
    for data in out:
        sig = data["signature"]
        print(data["id"] + "  params_b=" + str(data.get("params_b")) + "  hidden=" + str(data.get("hidden")) +
              "  layers=" + str(data["layers"]) + "  r=" + str(sig.get("split_half_r")) +
              "  check=" + str(sig.get("check_mass")))
    print(str(len(out)) + " entries")
    return 0


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] not in ("write", "find", "-h", "--help"):
        argv.insert(0, "write")
    parser = argparse.ArgumentParser(description="Skill-map entries for the library.")
    sub = parser.add_subparsers(dest="command", required=True)
    w = sub.add_parser("write", help="write library entries from map files (default)")
    w.add_argument("--entries-dir", default=DEFAULT_ENTRIES_DIR)
    w.add_argument("--library", default=DEFAULT_LIBRARY)
    w.add_argument("--min-split-half-r", type=float, default=MIN_SPLIT_HALF_R)
    w.add_argument("--min-check-mass", type=float, default=MIN_CHECK_MASS)
    w.add_argument("--min-experts", type=int, default=MIN_EXPERTS)
    w.add_argument("--dry-run", action="store_true", help="count only, write nothing")
    f = sub.add_parser("find", help="list map entries by skill, kind, and model size")
    f.add_argument("--library", default=DEFAULT_LIBRARY)
    f.add_argument("--kind", choices=["block", "expert"])
    f.add_argument("--skill")
    f.add_argument("--model")
    f.add_argument("--status", default=None)
    f.add_argument("--max-params", type=float, help="params_b at or under this")
    f.add_argument("--min-params", type=float)
    f.add_argument("--max-hidden", type=int)
    f.add_argument("--family")
    f.add_argument("--license")
    args = parser.parse_args(argv)
    if args.command == "write":
        return write_entries(args)
    return find_entries(args)


if __name__ == "__main__":
    sys.exit(main())
