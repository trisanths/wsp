"""Read the atlas, profiles.json, and the catalogs for the composer's tools.

Nothing here opens a weight file. Catalog records name their units files
by path only, and the tools never return the path's contents. sanitize()
also replaces any long list of numbers with a short summary, so no tool
result can carry raw weight values, mined unit lists, or loss curves.

The atlas lives in atlas/entries/*.json (one entry or a list of entries
per file). Its schema is still being written by another workflow, so this
module reads the fields it knows and passes the rest through sanitize().
"""

import glob
import json
import os

PART_TYPES = ["block", "expert", "mlp_cluster", "attn_heads", "deletion_mask", "task_vector"]
# Keys that hold unit lists, samples, curves, or file lists, not facts the composer needs.
DROP_KEYS = ["indices", "samples", "controls", "losses", "units_tensor", "state_dict", "weights", "tensor"]
MAX_NUMBERS = 32
MAX_STRING = 1500

# Person A's dissect (person_a/dissect/prepare_data.py) mines and scores
# its parts on 300 GSM8K test items. GSM8K test is a held-out set here.
PERSON_A_HELDOUT_NOTE = ("mined and scored on GSM8K test items (person_a dissect), which overlap the "
                         "gsm8k_test held-out set; its signature scores are hidden")


def is_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def number_summary(values):
    low = values[0]
    high = values[0]
    total = 0.0
    for value in values:
        if value < low:
            low = value
        if value > high:
            high = value
        total += value
    return {"summary": f"list of {len(values)} numbers", "min": round(low, 4), "max": round(high, 4),
            "mean": round(total / len(values), 4)}


def sanitize(value, depth=0):
    """Return a JSON-safe copy with no tensors, no long numeric lists, and no very long strings."""
    if hasattr(value, "shape") and hasattr(value, "dtype"):
        return "tensor omitted"
    if isinstance(value, dict):
        if depth > 8:
            return "nested too deep, omitted"
        out = {}
        for key in value:
            if key in DROP_KEYS:
                continue
            out[str(key)] = sanitize(value[key], depth + 1)
        return out
    if isinstance(value, (list, tuple)):
        all_numbers = len(value) > 0
        for item in value:
            if not is_number(item):
                all_numbers = False
                break
        if all_numbers and len(value) > MAX_NUMBERS:
            return number_summary(list(value))
        out = []
        for item in value:
            out.append(sanitize(item, depth + 1))
        return out
    if isinstance(value, str) and len(value) > MAX_STRING:
        return value[:MAX_STRING] + " ... (cut)"
    if isinstance(value, float):
        return round(value, 5)
    if value is None or isinstance(value, (bool, int, str)):
        return value
    return str(value)


def read_jsonl(path):
    records = []
    if not os.path.exists(path):
        return records
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line == "":
                continue
            records.append(json.loads(line))
    return records


class Sources:
    """Everything the composer may read, loaded from the repo root."""

    def __init__(self, root):
        self.root = root
        self.profiles = self.load_profiles()
        self.atlas_entries = self.load_atlas()
        self.catalog_records = self.load_catalog()
        self.models = self.model_table()

    def load_profiles(self):
        path = os.path.join(self.root, "profiles.json")
        if not os.path.exists(path):
            return []
        with open(path) as f:
            data = json.load(f)
        if isinstance(data, dict):
            data = data.get("profiles", [])
        return data

    def load_atlas(self):
        entries = []
        paths = sorted(glob.glob(os.path.join(self.root, "atlas", "entries", "*.json")))
        for path in paths:
            try:
                with open(path) as f:
                    data = json.load(f)
            except (OSError, ValueError):
                continue
            if isinstance(data, list):
                for item in data:
                    if isinstance(item, dict):
                        item = dict(item)
                        item["atlas_file"] = os.path.relpath(path, self.root)
                        entries.append(item)
            elif isinstance(data, dict):
                data = dict(data)
                data["atlas_file"] = os.path.relpath(path, self.root)
                entries.append(data)
        return entries

    def load_catalog(self):
        """Return the catalog records with units paths relative to the repo root."""
        records = []
        seen = set()
        for record in read_jsonl(os.path.join(self.root, "catalog.jsonl")):
            if record["id"] not in seen:
                seen.add(record["id"])
                # A copied Person A record still carries its dissect scores.
                if "gsm8k_heldout" in json.dumps(record.get("signature")):
                    record = dict(record)
                    record["heldout_overlap"] = PERSON_A_HELDOUT_NOTE
                records.append(record)
        for record in read_jsonl(os.path.join(self.root, "person_a", "catalog.jsonl")):
            if record["id"] in seen:
                continue
            record = dict(record)
            units = record.get("units")
            if isinstance(units, str) and not units.startswith("/") and not units.startswith("person_a/"):
                record["units"] = "person_a/" + units
            record["heldout_overlap"] = PERSON_A_HELDOUT_NOTE
            seen.add(record["id"])
            records.append(record)
        # Parts listed in the atlas (for example expert maps with ids).
        for entry in self.atlas_entries:
            parts = entry.get("parts")
            if not isinstance(parts, list):
                continue
            for part in parts:
                if not isinstance(part, dict) or "id" not in part or part.get("type") not in PART_TYPES:
                    continue
                if part["id"] in seen:
                    continue
                seen.add(part["id"])
                records.append(dict(part))
        return records

    def records_by_id(self):
        out = {}
        for record in self.catalog_records:
            out[record["id"]] = record
        return out

    def model_table(self):
        """{hf_id: size info} from profiles, then the atlas (the atlas wins on a clash)."""
        models = {}
        for profile in self.profiles:
            hf_id = profile.get("hf_id")
            if not hf_id or profile.get("params_b") is None:
                continue
            models[hf_id] = {
                "params_b": profile.get("params_b"),
                "active_params_b": profile.get("active_params_b") or profile.get("params_b"),
                "layers": profile.get("layers"),
                "hidden": profile.get("hidden"),
                "tokenizer_family": profile.get("tokenizer_family"),
            }
        for entry in self.atlas_entries:
            hf_id = entry.get("hf_id") or entry.get("model")
            params = entry.get("params_b")
            if params is None:
                params = entry.get("total_params_b")
            if not isinstance(hf_id, str) or params is None:
                continue
            info = models.get(hf_id, {})
            info = dict(info)
            info["params_b"] = params
            info["active_params_b"] = entry.get("active_params_b") or info.get("active_params_b") or params
            for key, names in [("layers", ["layers", "num_layers", "num_hidden_layers"]),
                               ("hidden", ["hidden", "hidden_size"]),
                               ("tokenizer_family", ["tokenizer_family"]),
                               ("vocab_size", ["vocab_size"])]:
                for name in names:
                    if entry.get(name) is not None:
                        info[key] = entry[name]
                        break
            models[hf_id] = info
        return models
