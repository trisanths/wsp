"""Resolve model paths and compute model hashes.

The hash covers config, tokenizer and weight files. It is cached per path by
(file name, size, mtime) so we do not re-read gigabytes of weights per call.
"""
import fcntl
import hashlib
import json
import os

from wsp_eval import config

HASHED_SUFFIXES = (".safetensors", ".bin", ".json", ".model", ".txt", ".jinja", ".pt")


def resolve_model_path(model_path):
    """Return a local directory for a model path or a Hugging Face repo id."""
    if str(model_path).startswith("fake:"):
        return str(model_path)
    if os.path.isdir(model_path):
        return os.path.abspath(model_path)
    from huggingface_hub import snapshot_download
    return snapshot_download(model_path)


def _files_to_hash(model_dir):
    names = []
    for name in sorted(os.listdir(model_dir)):
        if name.startswith("wsp_"):
            continue  # our own metadata files do not change the model
        if name.endswith(HASHED_SUFFIXES):
            names.append(name)
    return names


def _fingerprint(model_dir):
    parts = []
    for name in _files_to_hash(model_dir):
        stat = os.stat(os.path.join(model_dir, name))
        parts.append([name, stat.st_size, stat.st_mtime_ns])
    return parts


def model_hash(model_path):
    if str(model_path).startswith("fake:"):
        return hashlib.sha256(model_path.encode("utf-8")).hexdigest()

    model_dir = resolve_model_path(model_path)
    fingerprint = _fingerprint(model_dir)
    store_path = config.work_path("model_hashes.json")
    lock_path = store_path + ".lock"

    with open(lock_path, "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        store = {}
        if os.path.exists(store_path):
            with open(store_path) as f:
                store = json.load(f)
        entry = store.get(model_dir)
        if entry is not None and entry["fingerprint"] == fingerprint:
            return entry["hash"]

    digest = hashlib.sha256()
    for name in _files_to_hash(model_dir):
        digest.update(name.encode("utf-8"))
        with open(os.path.join(model_dir, name), "rb") as f:
            while True:
                block = f.read(16 * 1024 * 1024)
                if not block:
                    break
                digest.update(block)
    value = digest.hexdigest()

    with open(lock_path, "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        store = {}
        if os.path.exists(store_path):
            with open(store_path) as f:
                store = json.load(f)
        store[model_dir] = {"fingerprint": fingerprint, "hash": value}
        tmp = store_path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(store, f, indent=1)
        os.replace(tmp, store_path)
    return value
