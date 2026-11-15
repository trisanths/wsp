# Weight-difference maps for fine-tune families: where a fine-tune differs from its host model.
# For each pair (member, reference) the job compares every weight tensor that both checkpoints
# have with the same shape, and reports the relative difference ||W_member - W_ref|| / ||W_ref||
# and the cosine, per tensor, per decoder layer and module kind, and for the tensors outside the
# layers. The reference is the family host, or the ancestor a member was really tuned from.
# CPU containers only. The app is ephemeral: use "modal run", never "modal deploy".
#
#   .venv/bin/modal run mine/weight_delta.py --family lfm2
#   .venv/bin/modal run mine/weight_delta.py --all --jobs 4
#   .venv/bin/modal run mine/weight_delta.py --member LiquidAI/LFM2-1.2B-Tool --reference LiquidAI/LFM2-1.2B
#
# Families: atlas/lineage/families.json.
# Output: atlas/lineage/weight_delta/<member key>__vs__<reference key>.json, one file per pair.
import datetime
import json
import math
import os
import re
import struct
import sys
import time

import modal

APP_NAME = "swarmy-delta"
CACHE_DIR = "/models/hf_cache"
# Weights and the json files that say how the shards are laid out. Nothing else is needed.
PATTERNS = ["*.safetensors", "*.json"]

# Tensors are read in slices of at most this many elements. One slice of each model is cast to
# float64 (134 MB each) and the difference is a third array, so the working memory stays under
# 1 GB whatever the tensor size. The largest tensor in the families is the gemma-3-27b
# embedding (1.41 billion elements, 2.8 GB as stored in bf16); it is never held whole.
CHUNK_ELEMS = 1 << 24

# Container size for one pair. 8 cores: the dot products run on all of them. 8 GB of memory:
# the slice size is fixed, so the need does not grow with the model. The LFM2-1.2B-Tool pair
# peaked at 1.1 GB (process maximum, torch included). 8 GB leaves room for the page cache of
# the volume reads. The job records its peak memory in each output file.
MEASURE_CPU = 8
MEASURE_MEMORY_MB = 8192

# Checkpoints of one family store the decoder under different wrappers ("model.layers.3.",
# "layers.3.", "language_model.model.layers.3.", "model.language_model.layers.3."). These leading
# name parts are removed before two names are compared.
WRAPPERS = ["model", "language_model"]
ATTN_PARTS = ["self_attn", "attn", "attention"]
MLP_PARTS = ["mlp", "feed_forward", "ffn"]
LAYER_PATTERN = re.compile(r"^layers\.(\d+)\.(.+)$")
MIN_LAYER_MATCH = 0.9

app = modal.App(APP_NAME)
volume = modal.Volume.from_name("swarmy-models", create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch", index_url="https://download.pytorch.org/whl/cpu")
    .pip_install("huggingface_hub[hf_xet]", "numpy")
    .env({"HF_HOME": CACHE_DIR})
)


class PairError(Exception):
    # A pair that cannot be measured (too few matching layer tensors, no layers, bad checkpoint).
    pass


# ---------- names ----------

def normalise(name):
    # Returns (layer, key, kind). layer is the decoder layer index, or None for a tensor outside
    # the layers. key is the name two checkpoints are matched by. kind is the module kind.
    parts = name.split(".")
    while len(parts) > 1 and parts[0] in WRAPPERS:
        parts = parts[1:]
    stripped = ".".join(parts)
    # Only a name that starts with "layers.<N>." after the wrappers are removed is a decoder
    # layer. A vision tower also has "encoder.layers.<N>." names, and those must stay outside.
    found = LAYER_PATTERN.match(stripped)
    if found:
        return int(found.group(1)), stripped, layer_kind(found.group(2))
    return None, stripped, outside_kind(stripped)


def layer_kind(suffix):
    parts = suffix.split(".")
    # Norm weights first: "self_attn.q_norm.weight" is a norm weight inside the attention module.
    for part in parts[:-1]:
        if "norm" in part:
            return "norm"
    if parts[0] in ATTN_PARTS:
        return "attn"
    if parts[0] in MLP_PARTS:
        return "mlp"
    # Any other mixer keeps its own module name as the kind (LFM2 has "conv"), so that a new
    # module type shows up in the output under its name.
    return parts[0]


def outside_kind(stripped):
    first = stripped.split(".")[0]
    if first == "lm_head":
        return "lm_head"
    if "norm" in first:
        return "norm"
    if "embed" in first:
        return "embed"
    # Vision towers, projectors and anything else that is not part of the text decoder.
    return "other"


def index_names(tensors):
    # tensors: {name: {"shape": [...], "dtype": "BF16", ...}}. Returns {key: entry}.
    index = {}
    for name in sorted(tensors):
        layer, key, kind = normalise(name)
        if key in index:
            raise PairError("two tensors normalise to the same name '" + key + "': "
                            + index[key]["name"] + " and " + name)
        index[key] = {"name": name, "layer": layer, "kind": kind,
                      "shape": list(tensors[name]["shape"]), "dtype": tensors[name]["dtype"]}
    return index


def match_names(member_index, reference_index):
    # Sorts every tensor into exactly one list, so nothing is dropped without a record.
    matched = []
    unmatched_member = []
    unmatched_reference = []
    shape_mismatch = []
    for key in member_index:
        m = member_index[key]
        if key not in reference_index:
            unmatched_member.append({"name": m["name"], "normalised": key, "shape": m["shape"]})
            continue
        r = reference_index[key]
        if m["shape"] != r["shape"]:
            shape_mismatch.append({"name": m["name"], "reference_name": r["name"], "normalised": key,
                                   "layer": m["layer"], "kind": m["kind"],
                                   "member_shape": m["shape"], "reference_shape": r["shape"]})
            continue
        matched.append({"key": key, "name": m["name"], "reference_name": r["name"],
                        "layer": m["layer"], "kind": m["kind"], "shape": m["shape"]})
    for key in reference_index:
        if key not in member_index:
            r = reference_index[key]
            unmatched_reference.append({"name": r["name"], "normalised": key, "shape": r["shape"]})
    return matched, unmatched_member, unmatched_reference, shape_mismatch


def check_layer_match(member_index, matched, unmatched_member, shape_mismatch):
    # The 90% rule: when too few of the member's layer tensors have a partner, the two
    # checkpoints are not the same architecture (or the name rules do not fit), and a map
    # built from the few matches would mislead.
    total = 0
    for key in member_index:
        if member_index[key]["layer"] is not None:
            total = total + 1
    if total == 0:
        raise PairError("the member has no tensor named like a decoder layer ('layers.<N>.'); "
                        + str(len(member_index)) + " tensors in total")
    good = 0
    for row in matched:
        if row["layer"] is not None:
            good = good + 1
    fraction = good / total
    if fraction < MIN_LAYER_MATCH:
        examples = []
        for row in unmatched_member:
            if normalise(row["name"])[0] is not None and len(examples) < 3:
                examples.append(row["name"] + " (not in the reference)")
        for row in shape_mismatch:
            if row["layer"] is not None and len(examples) < 6:
                examples.append(row["name"] + " " + str(row["member_shape"]) + " against "
                                + str(row["reference_shape"]))
        raise PairError("only " + str(good) + " of the member's " + str(total)
                        + " layer tensors match the reference in name and shape ("
                        + str(round(100 * fraction, 1)) + "%, the minimum is "
                        + str(int(100 * MIN_LAYER_MATCH)) + "%). Examples: " + "; ".join(examples))
    return good, total, fraction


# ---------- math ----------

def pair_sums(member, reference):
    # The four sums that every number in the output is built from:
    # [sum (m - r)^2, sum m^2, sum r^2, sum m * r].
    # Both tensors are cast to float64 before the subtraction. A subtraction in bf16 rounds its
    # result to 8 significant bits: exact when the two weights are within a factor of two of
    # each other, wrong in the last bit when a small weight moved a lot or changed sign. So the
    # stored dtype is never used for arithmetic. float64 is wider than the float32 the measure
    # needs: it holds every bf16, float16 and float32 value, and the sums do not lose small terms.
    import torch

    m = member.to(torch.float64).reshape(-1)
    r = reference.to(torch.float64).reshape(-1)
    d = m - r
    return [float(torch.dot(d, d)), float(torch.dot(m, m)), float(torch.dot(r, r)), float(torch.dot(m, r))]


def add_sums(total, part):
    for i in range(4):
        total[i] = total[i] + part[i]


def rel_and_cosine(sums):
    # rel = ||m - r|| / ||r||, cosine = <m, r> / (||m|| ||r||). None when a norm is zero.
    diff_sq, member_sq, reference_sq, dot = sums
    rel = None
    cosine = None
    if reference_sq > 0:
        rel = math.sqrt(diff_sq) / math.sqrt(reference_sq)
        if member_sq > 0:
            # One square root of the product: equal tensors then give exactly 1.
            cosine = dot / math.sqrt(member_sq * reference_sq)
            cosine = max(-1.0, min(1.0, cosine))
    return rel, cosine


def group_stats(rows):
    # One group = all tensors of one module kind in one layer (or outside the layers). The group
    # value comes from the summed squares, so a large tensor counts by its size.
    total = [0.0, 0.0, 0.0, 0.0]
    params = 0
    for row in rows:
        add_sums(total, row["sums"])
        params = params + row["params"]
    rel, cosine = rel_and_cosine(total)
    return {"rel": rel, "cosine": cosine, "tensors": len(rows), "params": params}


def aggregate(rows, layers):
    # rows: one per matched tensor, with layer, kind, sums, params.
    by_layer = {}
    outside = {}
    by_kind = {}
    for row in rows:
        if row["layer"] is None:
            outside.setdefault(row["kind"], []).append(row)
        else:
            by_layer.setdefault(row["layer"], {}).setdefault(row["kind"], []).append(row)
            by_kind.setdefault(row["kind"], []).append(row)
    per_layer = []
    for layer in range(layers):
        entry = {"layer": layer}
        kinds = by_layer.get(layer, {})
        for kind in sorted(kinds):
            entry[kind] = group_stats(kinds[kind])
        per_layer.append(entry)
    # The three standard kinds are always present (None when the checkpoint stores no such
    # tensor, for example a tied lm_head), so a reader can tell "absent" from "forgotten".
    outside_layers = {"embed": None, "lm_head": None, "other": None}
    for kind in sorted(outside):
        outside_layers[kind] = group_stats(outside[kind])
    summary = {"all": group_stats(rows), "layers": {}}
    for kind in sorted(by_kind):
        summary["layers"][kind] = group_stats(by_kind[kind])
    return per_layer, outside_layers, summary


# ---------- checkpoints ----------

# Stored dtype name -> (torch dtype name, bytes per element).
DTYPES = {"F64": ("float64", 8), "F32": ("float32", 4), "F16": ("float16", 2), "BF16": ("bfloat16", 2),
          "I64": ("int64", 8), "I32": ("int32", 4), "I16": ("int16", 2), "I8": ("int8", 1),
          "U8": ("uint8", 1), "BOOL": ("bool", 1)}


def read_header(path):
    # A safetensors file starts with an 8-byte length and a JSON table of names, dtypes, shapes
    # and byte ranges. Returns the table and the file position where the tensor data starts.
    with open(path, "rb") as f:
        size = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(size))
    header.pop("__metadata__", None)
    return header, 8 + size


def weight_files(folder):
    # A sharded checkpoint lists its shards in model.safetensors.index.json. Without that file,
    # every safetensors file at the top of the folder is taken.
    index_path = os.path.join(folder, "model.safetensors.index.json")
    names = []
    if os.path.exists(index_path):
        with open(index_path) as f:
            names = sorted(set(json.load(f)["weight_map"].values()))
    else:
        for name in sorted(os.listdir(folder)):
            if name.endswith(".safetensors"):
                names.append(name)
    if len(names) == 0:
        raise PairError("no safetensors files in " + folder)
    return [os.path.join(folder, name) for name in names]


class Checkpoint:
    # The tensor table of one model folder, and slice reads from its shards. No tensor is
    # loaded when the object is built.
    #
    # Tensors are read with plain file reads at the byte ranges of the safetensors header, not
    # through safetensors.safe_open. safe_open maps each shard into memory and the pages stay
    # mapped: on the first real pair (two 2.34 GB checkpoints) the process peaked at 5.59 GB,
    # so a 65 GB pair would map far more than the container has. With plain reads the process
    # holds only the slices in use. The tests compare this reader with safetensors' own loader.
    def __init__(self, folder):
        if sys.byteorder != "little":
            raise PairError("safetensors data is little-endian; this reader needs a little-endian machine")
        self.folder = folder
        self.tensors = {}
        self.files = {}
        self.bytes = 0
        for path in weight_files(folder):
            self.bytes = self.bytes + os.path.getsize(path)
            header, data_start = read_header(path)
            for name in header:
                if name in self.tensors:
                    raise PairError("tensor " + name + " is stored in two files: "
                                    + os.path.basename(self.tensors[name]["file"]) + " and "
                                    + os.path.basename(path))
                dtype = header[name]["dtype"]
                if dtype not in DTYPES:
                    raise PairError("tensor " + name + " has dtype " + dtype + ", which this job cannot read")
                self.tensors[name] = {"file": path, "shape": list(header[name]["shape"]), "dtype": dtype,
                                      "offset": data_start + header[name]["data_offsets"][0]}

    def read(self, name, first=None, last=None):
        # Rows first..last-1 of one tensor (the whole tensor when first is None), in its stored dtype.
        import torch

        info = self.tensors[name]
        shape = info["shape"]
        torch_name, item_bytes = DTYPES[info["dtype"]]
        if first is None or len(shape) == 0:
            out_shape = shape
            skip = 0
        else:
            out_shape = [last - first] + shape[1:]
            skip = first * count_elements(shape[1:]) * item_bytes
        size = count_elements(out_shape) * item_bytes
        if size == 0:
            return torch.zeros(out_shape, dtype=getattr(torch, torch_name))
        if info["file"] not in self.files:
            self.files[info["file"]] = open(info["file"], "rb")
        f = self.files[info["file"]]
        f.seek(info["offset"] + skip)
        buffer = bytearray(size)
        got = f.readinto(buffer)
        if got != size:
            raise PairError("short read of " + name + " in " + os.path.basename(info["file"]) + ": "
                            + str(got) + " of " + str(size) + " bytes")
        return torch.frombuffer(buffer, dtype=getattr(torch, torch_name)).reshape(out_shape)

    def close(self):
        for path in self.files:
            self.files[path].close()
        self.files = {}

    def dtype_counts(self):
        counts = {}
        for name in self.tensors:
            dtype = self.tensors[name]["dtype"]
            counts[dtype] = counts.get(dtype, 0) + 1
        return counts


def count_elements(shape):
    total = 1
    for size in shape:
        total = total * size
    return total


def tensor_sums(member, member_name, reference, reference_name, rows=None):
    # The four sums for one tensor, read slice by slice along the first dimension.
    # rows limits the comparison to the first rows (used for embeddings of different sizes).
    shape = member.tensors[member_name]["shape"]
    total = [0.0, 0.0, 0.0, 0.0]
    if len(shape) == 0:
        add_sums(total, pair_sums(member.read(member_name), reference.read(reference_name)))
        return total
    if rows is None:
        rows = shape[0]
    row_elems = count_elements(shape[1:])
    step = max(1, CHUNK_ELEMS // max(1, row_elems))
    for first in range(0, rows, step):
        last = min(rows, first + step)
        add_sums(total, pair_sums(member.read(member_name, first, last),
                                  reference.read(reference_name, first, last)))
    return total


def compare_folders(member_folder, reference_folder, log=None):
    # Everything the job measures for one pair, from two folders of safetensors files.
    member = Checkpoint(member_folder)
    reference = Checkpoint(reference_folder)
    member_index = index_names(member.tensors)
    reference_index = index_names(reference.tensors)
    matched, unmatched_member, unmatched_reference, shape_mismatch = match_names(member_index, reference_index)
    good, total, fraction = check_layer_match(member_index, matched, unmatched_member, shape_mismatch)

    layers = 0
    for key in member_index:
        if member_index[key]["layer"] is not None:
            layers = max(layers, member_index[key]["layer"] + 1)

    # Layers in order, the tensors outside the layers first.
    matched.sort(key=lambda row: (-1 if row["layer"] is None else row["layer"], row["key"]))
    rows = []
    table = []
    done = 0
    for row in matched:
        sums = tensor_sums(member, row["name"], reference, row["reference_name"])
        rel, cosine = rel_and_cosine(sums)
        params = count_elements(row["shape"])
        rows.append({"layer": row["layer"], "kind": row["kind"], "sums": sums, "params": params})
        line = {"name": row["name"], "shape": row["shape"], "layer": row["layer"], "kind": row["kind"],
                "rel": rel, "cosine": cosine,
                "diff_norm": math.sqrt(sums[0]), "reference_norm": math.sqrt(sums[2])}
        if row["reference_name"] != row["name"]:
            line["reference_name"] = row["reference_name"]
        table.append(line)
        done = done + 1
        if log is not None and (done % 100 == 0 or done == len(matched)):
            log("compared " + str(done) + " of " + str(len(matched)) + " tensors")

    # An embedding that only gained or lost rows (new tokens) still has the old rows in common.
    # Measure those rows and keep the result with the mismatch record. It is not part of any
    # group value, because the group values cover tensors of equal shape only.
    for row in shape_mismatch:
        ms = row["member_shape"]
        rs = row["reference_shape"]
        if len(ms) >= 2 and len(ms) == len(rs) and ms[1:] == rs[1:]:
            shared = min(ms[0], rs[0])
            sums = tensor_sums(member, row["name"], reference, row["reference_name"], rows=shared)
            rel, cosine = rel_and_cosine(sums)
            row["shared_rows"] = {"rows": shared, "rel": rel, "cosine": cosine}
    member.close()
    reference.close()

    per_layer, outside_layers, summary = aggregate(rows, layers)
    kinds = []
    for kind in sorted(summary["layers"]):
        kinds.append(kind)
    return {
        "layers": layers,
        "kinds": kinds,
        "per_layer": per_layer,
        "outside_layers": outside_layers,
        "summary": summary,
        "tensors": table,
        "unmatched_member": unmatched_member,
        "unmatched_reference": unmatched_reference,
        "shape_mismatch": shape_mismatch,
        "counts": {"member_tensors": len(member_index), "reference_tensors": len(reference_index),
                   "matched": len(matched), "unmatched_member": len(unmatched_member),
                   "unmatched_reference": len(unmatched_reference), "shape_mismatch": len(shape_mismatch),
                   "member_layer_tensors": total, "member_layer_tensors_matched": good,
                   "layer_match_fraction": fraction},
        "dtype": "float64",
        "source_dtypes": {"member": member.dtype_counts(), "reference": reference.dtype_counts()},
        "weights_gb": {"member": round(member.bytes / 1e9, 2), "reference": round(reference.bytes / 1e9, 2)},
    }


# ---------- Modal functions ----------

@app.function(image=image, volumes={"/models": volume}, secrets=[modal.Secret.from_name("hf-token")],
              cpu=4, memory=8192, timeout=4 * 3600)
def fetch(hf_id, revision):
    # One container per model, so that two pairs with the same host never write the same files
    # into the volume at the same time.
    from huggingface_hub import snapshot_download

    start = time.time()
    path = snapshot_download(hf_id, revision=revision, allow_patterns=PATTERNS,
                             token=os.environ.get("HF_TOKEN"), max_workers=16)
    volume.commit()
    total = 0
    for name in os.listdir(path):
        if name.endswith(".safetensors"):
            total = total + os.path.getsize(os.path.join(path, name))
    return {"model": hf_id, "gb": round(total / 1e9, 2), "seconds": round(time.time() - start, 1)}


@app.function(image=image, volumes={"/models": volume}, cpu=MEASURE_CPU, memory=MEASURE_MEMORY_MB,
              timeout=4 * 3600)
def measure(member, member_revision, reference, reference_revision):
    import resource

    import torch
    from huggingface_hub import snapshot_download

    start = time.time()
    # A reused container still shows the volume as it was when the container started.
    volume.reload()
    # os.cpu_count() reports the cores of the machine, not the cores this container may use.
    torch.set_num_threads(MEASURE_CPU)

    def log(text):
        print("[" + member + " vs " + reference + "] " + text, flush=True)

    try:
        # local_files_only: this function reads the cache and never writes to the volume.
        member_folder = snapshot_download(member, revision=member_revision, allow_patterns=PATTERNS,
                                          local_files_only=True)
        reference_folder = snapshot_download(reference, revision=reference_revision, allow_patterns=PATTERNS,
                                             local_files_only=True)
        result = compare_folders(member_folder, reference_folder, log=log)
    except PairError as e:
        return {"error": str(e)}
    result["compute_seconds"] = round(time.time() - start, 1)
    # ru_maxrss is in KB on Linux.
    result["peak_memory_gb"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6, 2)
    result["container"] = {"cpu": MEASURE_CPU, "memory_mb": MEASURE_MEMORY_MB}
    return result


# ---------- local side ----------

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
FAMILIES = os.path.join(REPO, "atlas", "lineage", "families.json")
OUT_DIR = os.path.join(REPO, "atlas", "lineage", "weight_delta")


def model_key(hf_id):
    return hf_id.replace("/", "__")


def pair_path(member, reference, out_dir=None):
    if out_dir is None:
        out_dir = OUT_DIR
    return os.path.join(out_dir, model_key(member) + "__vs__" + model_key(reference) + ".json")


def load_families(path=None):
    if path is None:
        path = FAMILIES
    with open(path) as f:
        return json.load(f)["families"]


def family_pairs(family):
    # Every member against the host, and against its ancestor when it has one.
    pairs = []
    for row in family["members"]:
        pairs.append({"member": row["id"], "reference": family["host"], "relation": "host",
                      "family": family["name"], "host": family["host"]})
        if row.get("ancestor"):
            pairs.append({"member": row["id"], "reference": row["ancestor"], "relation": "ancestor",
                          "family": family["name"], "host": family["host"]})
    return pairs


def select_pairs(families, family="", everything=False, member="", reference=""):
    listed = []
    for fam in families:
        listed = listed + family_pairs(fam)
    if member or reference:
        if not (member and reference):
            raise SystemExit("--member and --reference go together")
        for pair in listed:
            if pair["member"] == member and pair["reference"] == reference:
                return [pair]
        # A pair that families.json does not list is measured as member against host, no family.
        print("note:", member, "against", reference, "is not in families.json; relation set to 'host'")
        return [{"member": member, "reference": reference, "relation": "host", "family": None, "host": reference}]
    if everything:
        return listed
    if family:
        names = [fam["name"] for fam in families]
        if family not in names:
            raise SystemExit("unknown family '" + family + "'. Families: " + ", ".join(names))
        return [pair for pair in listed if pair["family"] == family]
    raise SystemExit("give --family <name>, --all, or --member <id> --reference <id>")


def write_json_atomic(path, data):
    # Write to a temp file and rename, so a stopped run never leaves half a file that a later
    # run would skip as done.
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=1)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def show(value, digits):
    if value is None:
        return "-"
    return ("%." + str(digits) + "f") % value


def format_table(report, top=5):
    # For each module kind except norm: the layers with the largest relative difference.
    lines = [report["member"] + " against " + report["reference"] + " (" + report["relation"] + "): "
             + str(report["layers"]) + " layers, " + str(report["counts"]["matched"]) + " tensors compared, "
             + str(report["counts"]["unmatched_member"]) + " only in member, "
             + str(report["counts"]["unmatched_reference"]) + " only in reference, "
             + str(report["counts"]["shape_mismatch"]) + " shape mismatches"]
    kinds = []
    for kind in ["attn", "mlp"] + report["kinds"]:
        if kind in report["kinds"] and kind not in kinds and kind != "norm":
            kinds.append(kind)
    for kind in kinds:
        rows = []
        for entry in report["per_layer"]:
            if kind in entry and entry[kind]["rel"] is not None:
                rows.append((entry[kind]["rel"], entry["layer"], entry[kind]["cosine"]))
        rows.sort(key=lambda row: -row[0])
        lines.append("  largest " + kind + " rel (" + str(len(rows)) + " layers have " + kind + "):")
        for rel, layer, cosine in rows[:top]:
            lines.append("    layer " + str(layer).rjust(3) + "   rel " + show(rel, 6) + "   cosine " + show(cosine, 8))
    outside = []
    for kind in sorted(report["outside_layers"]):
        group = report["outside_layers"][kind]
        if group is not None:
            outside.append(kind + " rel " + show(group["rel"], 6) + " cosine " + show(group["cosine"], 8))
    lines.append("  outside the layers: " + ("; ".join(outside) if outside else "no matched tensor"))
    return "\n".join(lines)


def run_capped(function, argument_lists, jobs):
    # Runs one container per argument list, at most `jobs` at a time. Returns results in order;
    # a failed call gives its exception in place of the result.
    from concurrent.futures import ThreadPoolExecutor

    def one(arguments):
        try:
            return function.remote(*arguments)
        except Exception as e:
            return e

    with ThreadPoolExecutor(max_workers=max(1, jobs)) as pool:
        return list(pool.map(one, argument_lists))


# The type annotations are here only because Modal builds the command-line options from them.
@app.local_entrypoint()
def main(family: str = "", all: bool = False, member: str = "", reference: str = "",
         jobs: int = 4, force: bool = False):
    # Local-only helper (the containers do not have it, so import it here, not at the top).
    sys.path.insert(0, HERE)
    import common
    from concurrent.futures import ThreadPoolExecutor, as_completed

    wall_start = time.time()
    pairs = select_pairs(load_families(), family=family, everything=all, member=member, reference=reference)
    todo = []
    for pair in pairs:
        path = pair_path(pair["member"], pair["reference"])
        if os.path.exists(path) and not force:
            print("exists:", path)
        else:
            todo.append(pair)
    if len(todo) == 0:
        print("nothing to do")
        return

    # Pin each model once, so that every pair of this run sees the same weights of a shared host.
    revisions = {}
    for pair in todo:
        for hf_id in [pair["member"], pair["reference"]]:
            if hf_id not in revisions:
                revisions[hf_id] = common.pin_revision(hf_id)
                print("pinned", hf_id, revisions[hf_id][:10], flush=True)

    # Step 1: bring every model into the volume cache, one container per model.
    models = sorted(revisions)
    failed_models = {}
    fetched = run_capped(fetch, [(hf_id, revisions[hf_id]) for hf_id in models], jobs)
    for hf_id, result in zip(models, fetched):
        if isinstance(result, Exception):
            failed_models[hf_id] = repr(result)[:300]
            print("FETCH FAILED:", hf_id, failed_models[hf_id], flush=True)
        else:
            print("in cache:", result, flush=True)

    # Step 2: one container per pair, at most `jobs` at a time. Each file is written when its
    # pair finishes, so a stopped run keeps the finished pairs.
    failures = []
    runnable = []
    written = 0
    for pair in todo:
        broken = [hf_id for hf_id in [pair["member"], pair["reference"]] if hf_id in failed_models]
        if broken:
            failures.append((pair, "model not downloaded: " + ", ".join(broken)))
        else:
            runnable.append(pair)

    def one(pair):
        return measure.remote(pair["member"], revisions[pair["member"]],
                              pair["reference"], revisions[pair["reference"]])

    with ThreadPoolExecutor(max_workers=max(1, jobs)) as pool:
        futures = {}
        for pair in runnable:
            futures[pool.submit(one, pair)] = pair
        for future in as_completed(futures):
            pair = futures[future]
            try:
                result = future.result()
            except Exception as e:
                failures.append((pair, repr(e)[:500]))
                continue
            if "error" in result:
                failures.append((pair, result["error"]))
                continue
            report = {
                "member": pair["member"], "reference": pair["reference"], "relation": pair["relation"],
                "family": pair["family"], "host": pair["host"],
                "member_revision": revisions[pair["member"]],
                "reference_revision": revisions[pair["reference"]],
            }
            report.update(result)
            report["measured_at"] = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            path = pair_path(pair["member"], pair["reference"])
            write_json_atomic(path, report)
            written = written + 1
            print(format_table(report))
            print("  wrote", path, "| compute", result["compute_seconds"], "s | peak memory",
                  result["peak_memory_gb"], "GB", flush=True)

    for pair, message in failures:
        print("FAILED:", pair["member"], "against", pair["reference"], "-", message)
    print("done:", written, "pairs written,", len(failures), "failed, wall",
          round(time.time() - wall_start, 1), "s")
    if failures:
        raise SystemExit(1)
