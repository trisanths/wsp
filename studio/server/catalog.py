"""The builder palette and the catalog the studio runs against.

For the stub backend the catalog is the stub catalog plus the real deletion
masks, so the palette has something to remove. The stub ignores masks when it
scores, which is fine because every stub number is labeled STUB.
"""

import json
import os
import threading

from wsp.swarm import config as config_mod
from wsp.swarm import factory
from wsp.swarm import space
from wsp.swarm import stub

from studio.server import library
from studio.server import settings
from studio.server import sizing

HOST_PARAMS_B = {"unsloth/Llama-3.2-1B-Instruct": 1.24}
DEFAULT_HOST_PARAMS_B = 1.24

KIND_LABELS = {"block": "block", "mlp_cluster": "neuron cluster", "attn_heads": "attention heads",
               "expert": "expert", "inserted_layer": "inserted layers"}

_merge_lock = threading.Lock()


def host_params_b(host):
    """Parameter count of the host in billions (1.24 when unknown)."""
    if host in HOST_PARAMS_B:
        return HOST_PARAMS_B[host]
    return DEFAULT_HOST_PARAMS_B


def real_catalog_path():
    return os.path.join(config_mod.REPO_ROOT, "catalog.jsonl")


def read_records(path):
    records = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line != "":
                records.append(json.loads(line))
    return records


def stub_catalog_path(stub_seed=0):
    """Write the stub parts plus the real masks to one file and return its path."""
    path = os.path.join(settings.out_dir(), "catalog_stub.jsonl")
    with _merge_lock:
        records = stub.StubWorld(stub_seed).records()
        real = real_catalog_path()
        if os.path.exists(real):
            for record in read_records(real):
                if record.get("type") != "deletion_mask":
                    continue
                small = dict(record)
                signature = record.get("signature") or {}
                small["signature"] = {"num_deleted": signature.get("num_deleted")}
                records.append(small)
        temp_path = path + "." + str(os.getpid()) + ".tmp"
        with open(temp_path, "w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, sort_keys=True) + "\n")
        os.replace(temp_path, path)
    return path


UNIT_KINDS = ["mlp_cluster", "attn_heads", "expert"]
BLOCK_KINDS = ["block", "inserted_layer"]
_placement_cache = {}


def units_path_for(entry):
    """The units path a compiler record uses: relative to the repo root when the file is inside it."""
    absolute = os.path.join(library.library_dir(), *str(entry["units_file"]).split("/"))
    root = os.path.abspath(settings.REPO_ROOT)
    if os.path.abspath(absolute).startswith(root + os.sep):
        return os.path.relpath(absolute, root).replace(os.sep, "/")
    return absolute


def unit_record(entry):
    """(record, reason) for a part with an index file: mlp_cluster, attn_heads or expert."""
    layers = entry.get("layer_list") or []
    if len(layers) == 0:
        return None, "No donor layer in the library file."
    if len(layers) > 1:
        return None, "The library file lists " + str(len(layers)) + " donor layers; the compiler needs one."
    units_file = entry.get("units_file")
    if not units_file:
        remote = entry.get("units_remote")
        if remote:
            return None, "No unit indices in the library file. The index file is only at " + str(remote) + "."
        return None, "No unit indices in the library file."
    absolute = os.path.join(library.library_dir(), *str(units_file).split("/"))
    if not os.path.isfile(absolute):
        return None, "The index file " + str(units_file) + " is named in the library but is not in the folder."
    record = {"id": entry["catalog_id"], "type": entry["kind"], "model": entry["model"], "layer": layers[0],
              "units": units_path_for(entry), "trained_on": ", ".join(entry.get("skills") or []) or None,
              "source": "library entry " + entry["id"], "library_entry": entry["id"]}
    return record, None


def block_record(entry):
    """(record, reason) for a block or an inserted layer: it needs the donor model and one run of layers."""
    layers = entry.get("layer_list") or entry.get("layers") or []
    if len(layers) == 0:
        return None, "No layer range in the library file."
    first = min(layers)
    last = max(layers)
    if entry["kind"] == "block" and len(layers) > 2 and last - first + 1 != len(set(layers)):
        return None, "The layers are not one run, so the compiler cannot take them as a block."
    record = {"id": entry["catalog_id"], "type": "block", "model": entry["model"], "layers": [first, last],
              "source": "library entry " + entry["id"], "library_entry": entry["id"]}
    return record, None


def placement(entry):
    """(record, reason) for one library entry: a compiler record, or why it cannot be placed."""
    if entry.get("tap_only"):
        return None, library.TAP_ONLY_REASON
    if not entry.get("model"):
        return None, "No donor model in the library file."
    if entry["kind"] in UNIT_KINDS:
        if entry.get("source") == "atlas":
            return None, "The atlas entry lists expert numbers but has no index file."
        return unit_record(entry)
    if entry["kind"] in BLOCK_KINDS:
        return block_record(entry)
    return None, "The compiler does not take this kind of part."


def library_placements():
    """{catalog_id: (record, reason)} for every library and atlas part, built once per library load."""
    data = library.get()
    if _placement_cache.get("key") is not data:
        found = {}
        for entry in data["parts"]:
            if entry["catalog_id"] not in found:
                found[entry["catalog_id"]] = placement(entry)
        _placement_cache["key"] = data
        _placement_cache["map"] = found
    return _placement_cache["map"]


def library_records(skip_ids):
    """Compiler records built from library entries, leaving out ids the real catalog already has."""
    records = []
    for part_id, found in sorted(library_placements().items()):
        if found[0] is not None and part_id not in skip_ids:
            records.append(found[0])
    return records


def modal_catalog_path():
    """Write the catalog the Modal backend runs against and return its path.

    It is the repo's catalog.jsonl plus a record for every library part whose
    file holds what the compiler needs (donor model, layer, and an index file
    or a layer range). Taps need no record. The containers get the records a
    program uses with each call, so this copy is for the server and for
    anyone who wants to see what it used.
    """
    path = os.path.join(settings.out_dir(), "catalog_modal.jsonl")
    with _merge_lock:
        records = read_records(real_catalog_path())
        records.extend(library_records(set(record["id"] for record in records)))
        temp_path = path + "." + str(os.getpid()) + ".tmp"
        with open(temp_path, "w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, sort_keys=True) + "\n")
        os.replace(temp_path, path)
    return path


def catalog_path_for(backend):
    """The catalog_path config value for a backend (None lets the factory choose)."""
    if backend == "stub":
        return stub_catalog_path()
    if backend == "modal":
        return modal_catalog_path()
    return None


def load_catalog(backend):
    """Return (config, catalog) for the backend, used for the palette and for validation."""
    config = config_mod.make_config({"backend": backend, "catalog_path": catalog_path_for(backend),
                                     "outputs_dir": settings.out_dir()})
    catalog, path = factory.build_catalog(config)
    return config, catalog


def donor_topic(model):
    """A short topic word for a donor model."""
    text = str(model)
    if "Math" in text:
        return "Math"
    if "Coder" in text or "Code" in text:
        return "Code"
    if "OLMoE" in text:
        return "MoE"
    return "General"


def donor_short(model):
    """Qwen/Qwen2.5-Math-1.5B becomes Qwen2.5-Math."""
    name = str(model).split("/")[-1]
    pieces = name.split("-")
    if len(pieces) > 1 and pieces[-1].lower().endswith("b") and pieces[-1][:-1].replace(".", "").isdigit():
        pieces = pieces[:-1]
    return "-".join(pieces)


def layers_text(record):
    if record.get("layers") is not None:
        layers = record["layers"]
        if len(layers) == 2:
            return "layers " + str(layers[0]) + "-" + str(layers[1])
        return "layers " + ", ".join(str(item) for item in layers)
    if record.get("layer") is not None:
        layer = record["layer"]
        if isinstance(layer, list):
            if len(layer) == 2:
                return "layers " + str(layer[0]) + "-" + str(layer[1])
            return "layers " + ", ".join(str(item) for item in layer)
        return "layer " + str(layer)
    return ""


def layers_pair(record):
    """The donor layer range as [first, last], or None."""
    layers = record.get("layers")
    if layers is None:
        layers = record.get("layer")
    if layers is None:
        return None
    if isinstance(layers, list):
        if len(layers) == 0:
            return None
        return [layers[0], layers[-1]]
    return [layers, layers]


def part_label(record):
    """A short label, for example: Math block L8-19. The donor goes in the summary."""
    topic = donor_topic(record.get("model"))
    kind = KIND_LABELS.get(record.get("type"), str(record.get("type")))
    label = topic + " " + kind
    pair = layers_pair(record)
    if pair is not None:
        if pair[0] == pair[1]:
            label = label + " L" + str(pair[0])
        else:
            label = label + " L" + str(pair[0]) + "-" + str(pair[1])
    return label


TOPIC_ORDER = ["Math", "Code"]


def part_sort_key(item):
    """Math parts first, then code, then the rest; inside a topic, by donor layer."""
    topic = item["label"].split(" ")[0]
    if topic in TOPIC_ORDER:
        rank = TOPIC_ORDER.index(topic)
    else:
        rank = len(TOPIC_ORDER)
    first = 0
    if item["layers"] is not None:
        first = item["layers"][0]
    return (rank, str(item["donor"]), first, item["id"])


def mask_label(mask_id):
    """mask_mmlupro_computer_science_llama1b becomes Computer science."""
    name = str(mask_id)
    if name.startswith("mask_"):
        name = name[len("mask_"):]
    if name.endswith("_llama1b"):
        name = name[:-len("_llama1b")]
    family = ""
    if name.startswith("mmlupro_"):
        name = name[len("mmlupro_"):]
    elif name.startswith("bbeh_"):
        name = name[len("bbeh_"):]
        family = "BBEH "
    name = name.replace("_", " ")
    if name == "":
        return str(mask_id)
    return family + name[0].upper() + name[1:]


def part_params_m(record, world):
    """Parameter count of a part in millions when it is known, else None."""
    if world is None:
        return None
    part = world.parts.get(record["id"])
    if part is None:
        return None
    return round(world.part_params(part) / 1e6, 1)


def depth_from_profile(record, ctx, n_host):
    """The default host depth from the donor's layer count in the model profile, or the 6 of 16 default."""
    info = ctx.models.get(record.get("model"))
    layer = space.part_layer(record)
    if info is not None and info.get("layers") and layer is not None:
        depth = int(round(layer / float(info["layers"]) * n_host))
        return max(0, min(n_host - 1, depth))
    return int(round(6 / 16.0 * n_host))


def catalog_part(record, config, catalog, n_host, world, ctx, entries):
    """One palette entry for a compiler catalog record, with the library status when the library has the part."""
    entry = entries.get(record["id"])
    params_m = part_params_m(record, world)
    if params_m is None:
        found = sizing.part_size_b(record["id"], ctx)
        if found is not None:
            params_m = round(found[0] * 1000.0, 1)
    if world is not None:
        status = "candidate"
        evidence = "stub part with made-up effects, not a measurement"
        source = "stub"
    elif entry is not None:
        status = entry["status"]
        evidence = entry["evidence"]
        source = "library"
    else:
        status = "candidate"
        evidence = "in the compiler catalog, no library entry yet"
        source = "catalog"
    if record.get("library_entry"):
        evidence = evidence + ". Compiler record built from the library file."
    default_at = space.default_at(record, config, n_host)
    if default_at is None:
        default_at = depth_from_profile(record, ctx, n_host)
    kind = record.get("type")
    if entry is not None and entry["kind"] == "inserted_layer":
        kind = "inserted_layer"
    return {
        "id": record["id"],
        "group": "add",
        "kind": kind,
        "label": part_label(record),
        "donor": record.get("model"),
        "layers": layers_pair(record),
        "params_m": params_m,
        "default_at": default_at,
        "summary": donor_short(record.get("model")) + " " + layers_text(record),
        "status": status,
        "evidence": evidence,
        "source": source,
        "usable": not (entry is not None and entry.get("tap_only")),
        "usable_reason": library.TAP_ONLY_REASON if entry is not None and entry.get("tap_only") else None,
        "findings": list(entry["findings"]) if entry is not None else [],
        "stitches": list(entry["stitches"]) if entry is not None else [],
        "skills": list(entry["skills"]) if entry is not None else [],
        "from_library": bool(record.get("library_entry")),
    }


NOT_IN_CATALOG = "Not in the compiler catalog yet. Mine it into a catalog record before a program can use it."
STUB_LOCK = "The stub backend has made-up parts only. Use the Modal backend to place library parts."


def lock_reason(entry, backend):
    """Why a library part is not in the compiler catalog: the specific thing that is missing."""
    record, reason = library_placements().get(entry["catalog_id"], (None, None))
    if reason is not None:
        return reason
    if record is not None and backend == "stub":
        return STUB_LOCK
    return NOT_IN_CATALOG


def library_part(entry, config, n_host, ctx, backend="modal"):
    """One palette entry for a library or atlas component that the compiler catalog does not have."""
    found = sizing.part_size_b(entry["catalog_id"], ctx)
    params_m = None
    if found is not None:
        params_m = round(found[0] * 1000.0, 1)
    elif entry.get("params_b"):
        params_m = round(entry["params_b"] * 1000.0, 1)
    label = KIND_LABELS.get(entry["kind"], entry["kind"])
    pair = entry["layers"]
    text = donor_topic(entry["model"]) + " " + label
    if pair is not None:
        if pair[0] == pair[1]:
            text = text + " L" + str(pair[0])
        else:
            text = text + " L" + str(pair[0]) + "-" + str(pair[1])
    summary = donor_short(entry["model"])
    if pair is not None:
        summary = summary + " " + layers_text({"layers": [pair[0]] if pair[0] == pair[1] else pair})
    return {
        "id": entry["catalog_id"],
        "group": "add",
        "kind": entry["kind"],
        "label": text,
        "donor": entry["model"],
        "layers": pair,
        "params_m": params_m,
        "default_at": int(round(6 / 16.0 * n_host)),
        "summary": summary,
        "status": entry["status"],
        "evidence": entry["evidence"],
        "source": entry["source"],
        "usable": False,
        "usable_reason": lock_reason(entry, backend),
        "findings": list(entry["findings"]),
        "stitches": list(entry["stitches"]),
        "skills": list(entry["skills"]),
        "from_library": False,
    }


def palette_counts(parts, masks):
    """Counts for the page: total, placeable, and each status, by kind (masks count as their own kind)."""
    kinds = {}
    status = {"candidate": 0, "validated": 0, "rejected": 0}
    placeable = 0
    for item in parts + [dict(mask, kind="deletion_mask") for mask in masks]:
        row = kinds.setdefault(item["kind"], {"total": 0, "placeable": 0, "candidate": 0, "validated": 0, "rejected": 0})
        row["total"] += 1
        row[item["status"]] += 1
        status[item["status"]] += 1
        if item["usable"]:
            row["placeable"] += 1
            placeable += 1
    return {"total": len(parts) + len(masks), "placeable": placeable, "kinds": kinds, "status": status}


def build_palette(backend):
    """Return the catalog response for the builder.

    Parts come from the compiler catalog (the stub world on the stub backend)
    and from the library and the atlas. Every part has a status and an
    evidence line. A part is usable when the compiler catalog has a record
    for it.
    """
    config, catalog = load_catalog(backend)
    n_host = space.host_layers(config, catalog)
    world = None
    if backend == "stub":
        world = stub.StubWorld(config["stub_seed"])
    ctx = sizing.SizeContext(config, catalog, host_params_b(config["host"]))
    entries = library.by_catalog_id()
    parts = []
    seen = []
    for record in space.graftable_parts(catalog):
        parts.append(catalog_part(record, config, catalog, n_host, world, ctx, entries))
        seen.append(record["id"])
    parts.sort(key=part_sort_key)
    extra = []
    for entry in library.get()["parts"]:
        if entry["catalog_id"] in seen:
            continue
        extra.append(library_part(entry, config, n_host, ctx, backend))
    extra.sort(key=lambda item: (str(item["kind"]), str(item["donor"]), item["layers"] or [0], item["id"]))
    parts.extend(extra)
    masks = []
    for part_id in sorted(catalog.records):
        record = catalog.records[part_id]
        if record.get("type") != "deletion_mask":
            continue
        entry = entries.get(part_id)
        status = "candidate"
        evidence = "no library entry yet"
        if entry is not None:
            status = entry["status"]
            evidence = entry["evidence"]
        masks.append({"id": part_id, "group": "remove", "label": mask_label(part_id),
                      "summary": "deletion mask: " + str(record.get("trained_on") or "task mask"),
                      "status": status, "evidence": evidence, "usable": True})
    train = [
        {"id": "lora", "group": "train", "label": "LoRA adapter", "summary": "rank 1 to 64"},
        {"id": "heal", "group": "train", "label": "Heal", "summary": "trains glue only"},
    ]
    data = library.get()
    kinds = []
    for kind in ["block", "attn_heads", "mlp_cluster", "expert", "inserted_layer"]:
        count = 0
        for part in parts:
            if part["kind"] == kind:
                count += 1
        kinds.append({"kind": kind, "name": library.KIND_NAMES[kind], "count": count})
    for mask in masks:
        mask["findings"] = []
        entry = entries.get(mask["id"])
        if entry is not None:
            mask["findings"] = list(entry["findings"])
    return {
        "counts": palette_counts(parts, masks),
        "product": "Sapien",
        "stub": backend == "stub",
        "host": {"model": config["host"], "layers": n_host, "params_b": host_params_b(config["host"])},
        "parts": parts,
        "masks": masks,
        "train": train,
        "kinds": kinds,
        "library": {"path": data["path"], "atlas": data["atlas"] is not None, "counts": library.status_counts(),
                    "loaded": data["counts"], "failed": len(data["failed"])},
    }


def recipe_list(backend):
    """Recipes for the page. A recipe with a starting program also says what the builder would refuse about it."""
    out = []
    config = None
    catalog = None
    for recipe in library.get()["recipes"]:
        item = dict(recipe)
        item["start_problems"] = []
        program = recipe["starting_program"]
        if program is not None:
            if config is None:
                config, catalog = load_catalog(backend)
            problems = []
            for graft in program.get("grafts") or []:
                if graft["id"] not in catalog.records:
                    problems.append("Part " + str(graft["id"]) + " is not in the compiler catalog.")
            if program.get("host") != config["host"]:
                problems.append("The recipe host is " + str(program.get("host")) + ", the builder host is " +
                                str(config["host"]) + ".")
            item["start_problems"] = problems
        out.append(item)
    return out


def finding_list():
    return list(library.get()["findings"])


def stitch_list():
    return list(library.get()["stitches"])


def model_list():
    return list(library.get()["model_list"])
