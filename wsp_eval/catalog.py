"""Read catalog.jsonl and fill the `signature` field of its records.

Person A owns the catalog; we only ever change `signature`. Writes are atomic
and locked so concurrent writers do not clobber each other. Component paths
follow wsp.catalog: hf://<owner>/<repo>/<subdir>, absolute, or relative to the
catalog file's directory.
"""
import fcntl
import json
import os

from wsp_eval import config


def load(path=None):
    if path is None:
        path = config.catalog_path()
    records = []
    if not os.path.exists(path):
        return records
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def get(component_id, path=None):
    for record in load(path):
        if record["id"] == component_id:
            return record
    raise KeyError("no catalog record with id " + repr(component_id))


def local_path(record, path=None):
    """Local file or directory for a component, downloading hf:// paths if needed."""
    if path is None:
        path = config.catalog_path()
    component = record["path"]
    if component.startswith("hf://"):
        from wsp.catalog import download_hf_dir
        return download_hf_dir(component)
    if os.path.isabs(component):
        return component
    return os.path.join(os.path.dirname(os.path.abspath(path)), component)


def update_signature(component_id, signature_update, path=None):
    """Merge `signature_update` into the record's signature. Other fields are untouched."""
    if path is None:
        path = config.catalog_path()
    lock_path = path + ".lock"
    with open(lock_path, "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        lines = []
        with open(path) as f:
            for line in f:
                if line.strip():
                    lines.append(line.rstrip("\n"))
        found = False
        new_lines = []
        for line in lines:
            record = json.loads(line)
            if record["id"] == component_id:
                found = True
                signature = record.get("signature")
                if signature is None:
                    signature = {}
                for key, value in signature_update.items():
                    signature[key] = value
                record["signature"] = signature
                new_lines.append(json.dumps(record))
            else:
                new_lines.append(line)
        if not found:
            raise KeyError("no catalog record with id " + repr(component_id))
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            f.write("\n".join(new_lines) + "\n")
        os.replace(tmp, path)
        fcntl.flock(lock, fcntl.LOCK_UN)
