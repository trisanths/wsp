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

from studio.server import settings

HOST_PARAMS_B = {"unsloth/Llama-3.2-1B-Instruct": 1.24}
DEFAULT_HOST_PARAMS_B = 1.24

KIND_LABELS = {"block": "block", "mlp_cluster": "MLP cluster", "attn_heads": "attention heads",
               "expert": "expert"}

_merge_lock = threading.Lock()

HOST_HIDDEN_SIZE = 2048
HOST_TAP_LAYER = 10
HOST_MMLU_LORA = 0.503
HOST_MMLU_BASE = 0.475

TAP_TRAIN_DEFAULT = {"data": "mmlu_aux", "steps": 500, "batch_size": 8, "seq_len": 256, "branch_lr": 0.0003,
                     "gate_lr": 0.03}

# Donors measured in results/README.md runs 6 to 8 with Llama-3.2-1B-Instruct as the host, tap into host
# layer 10, MMLU dev items scored by log-probability. Fields: donor, donor layer, donor hidden size, label,
# tap score. Run 6 gives Qwen2.5-7B, run 7 the 1.5B and 3B donors, run 8 the rest.
TAP_DONORS = [
    ("unsloth/gemma-2-9b-it", 29, 3584, "Gemma-2-9B", 0.729),
    ("Qwen/Qwen2.5-14B-Instruct", 34, 5120, "Qwen2.5-14B", 0.705),
    ("Qwen/Qwen2.5-7B-Instruct", 20, 3584, "Qwen2.5-7B", 0.679),
    ("microsoft/Phi-3.5-mini-instruct", 22, 3072, "Phi-3.5-mini", 0.663),
    ("unsloth/Meta-Llama-3.1-8B-Instruct", 22, 4096, "Llama-3.1-8B", 0.536),
    ("Qwen/Qwen2.5-3B-Instruct", 25, 2048, "Qwen2.5-3B", 0.579),
    ("Qwen/Qwen2.5-1.5B-Instruct", 20, 1536, "Qwen2.5-1.5B", 0.582),
]

# Total parameters of each donor in billions (public model cards). A tap model
# keeps its donor in memory and runs it on the prompt, so these count toward size.
DONOR_PARAMS_B = {
    "unsloth/gemma-2-9b-it": 9.2,
    "Qwen/Qwen2.5-14B-Instruct": 14.8,
    "Qwen/Qwen2.5-7B-Instruct": 7.6,
    "microsoft/Phi-3.5-mini-instruct": 3.8,
    "unsloth/Meta-Llama-3.1-8B-Instruct": 8.0,
    "Qwen/Qwen2.5-3B-Instruct": 3.1,
    "Qwen/Qwen2.5-1.5B-Instruct": 1.5,
}


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


def modal_catalog_path():
    """Write the catalog the Modal backend runs against and return its path.

    It is the repo's catalog.jsonl as it is. Taps need no record. The
    containers get the records a program uses with each call, so this copy
    is for the server and for anyone who wants to see what it used.
    """
    path = os.path.join(settings.out_dir(), "catalog_modal.jsonl")
    with _merge_lock:
        records = read_records(real_catalog_path())
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


def tap_params_m(donor_size, attn_dim=256):
    """Trainable parameters of one tap in millions, with the host hidden size of the Llama 1B host."""
    total = HOST_HIDDEN_SIZE + donor_size
    total += attn_dim * HOST_HIDDEN_SIZE * 2
    total += attn_dim * donor_size * 2
    total += 1
    return round(total / 1e6, 1)


def tap_parts():
    """Tap palette entries for the donors measured in runs 6 to 8."""
    parts = []
    for donor, layer, hidden, name, score in TAP_DONORS:
        parts.append({
            "id": "tap:" + donor + "@" + str(layer),
            "group": "add",
            "kind": "tap",
            "label": "Knowledge tap, " + name,
            "donor": donor,
            "donor_layer": layer,
            "layers": [layer, layer],
            "params_m": tap_params_m(hidden),
            "donor_params_b": DONOR_PARAMS_B.get(donor),
            "default_at": HOST_TAP_LAYER,
            "summary": "reads donor layer " + str(layer),
            "measured": {"task": "mmlu", "score": score, "lora": HOST_MMLU_LORA, "host": HOST_MMLU_BASE},
        })
    return parts


def build_palette(backend):
    """Return the catalog response for the builder."""
    config, catalog = load_catalog(backend)
    n_host = space.host_layers(config, catalog)
    world = None
    if backend == "stub":
        world = stub.StubWorld(config["stub_seed"])
    parts = []
    for record in space.graftable_parts(catalog):
        pair = layers_pair(record)
        parts.append({
            "id": record["id"],
            "group": "add",
            "kind": record.get("type"),
            "label": part_label(record),
            "donor": record.get("model"),
            "layers": pair,
            "params_m": part_params_m(record, world),
            "default_at": space.default_at(record, config, n_host),
            "summary": donor_short(record.get("model")) + " " + layers_text(record),
        })
    parts.sort(key=part_sort_key)
    if backend == "modal":
        parts = tap_parts() + parts
    masks = []
    for part_id in sorted(catalog.records):
        record = catalog.records[part_id]
        if record.get("type") != "deletion_mask":
            continue
        masks.append({"id": part_id, "group": "remove", "label": mask_label(part_id),
                      "summary": "deletion mask: " + str(record.get("trained_on") or "task mask")})
    train = [
        {"id": "lora", "group": "train", "label": "LoRA adapter", "summary": "rank 1 to 64"},
        {"id": "heal", "group": "train", "label": "Heal", "summary": "trains glue only"},
    ]
    return {
        "stub": backend == "stub",
        "host": {"model": config["host"], "layers": n_host, "params_b": host_params_b(config["host"])},
        "parts": parts,
        "masks": masks,
        "train": train,
        "tap_train_default": dict(TAP_TRAIN_DEFAULT),
    }
