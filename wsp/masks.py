"""Deletion masks: load, save, and combine them.

A deletion mask maps a parameter name to a boolean tensor of the same
shape. True means "delete this weight" (set it to zero).

On disk we use the Deletion Tuning format (agentic-learning-ai-lab/
deleting-weights, MIT license). A mask directory holds config.json and
model.safetensors. For each masked weight, model.safetensors stores
"<name>.shape" and "<name>.packed". The packed bytes are zstd-compressed
np.packbits of the KEEP bits. Keep is the opposite of delete.
"""

import json
import os
import re

import numpy as np
import torch
import zstandard
from safetensors.torch import load_file, save_file

LAYER_PATTERN = re.compile(r"\.layers\.(\d+)\.")


def layer_of(name):
    """Return the layer index in a parameter name, or None."""
    match = LAYER_PATTERN.search(name)
    if match is None:
        return None
    return int(match.group(1))


def module_of(name):
    """Return 'mlp', 'attn', or 'other' for a parameter name."""
    if ".mlp." in name:
        return "mlp"
    if ".self_attn." in name:
        return "attn"
    return "other"


def load_mask_dir(mask_dir):
    """Read a mask directory. Return a dict of name -> bool delete tensor."""
    with open(os.path.join(mask_dir, "config.json")) as f:
        config = json.load(f)
    if config.get("format") != "packbits_zstd":
        raise ValueError(f"unsupported mask format in {mask_dir}: {config.get('format')!r}")

    tensors = load_file(os.path.join(mask_dir, "model.safetensors"), device="cpu")
    decompressor = zstandard.ZstdDecompressor()
    mask = {}
    for key in tensors:
        if not key.endswith(".shape"):
            continue
        name = key[: -len(".shape")]
        shape = tuple(tensors[key].tolist())
        numel = int(np.prod(shape))
        raw = decompressor.decompress(tensors[name + ".packed"].numpy().tobytes())
        keep_bits = np.unpackbits(np.frombuffer(raw, dtype=np.uint8))[:numel]
        keep = torch.from_numpy(keep_bits.astype(bool)).reshape(shape)
        mask[name] = ~keep
    return mask


def save_mask_dir(mask, mask_dir):
    """Write a mask in the packbits_zstd format that load_mask_dir reads."""
    os.makedirs(mask_dir, exist_ok=True)
    compressor = zstandard.ZstdCompressor()
    tensors = {}
    total = 0
    deleted = 0
    for name in sorted(mask):
        delete = mask[name]
        keep_bits = (~delete).numpy().astype(np.uint8).reshape(-1)
        packed = compressor.compress(np.packbits(keep_bits).tobytes())
        tensors[name + ".shape"] = torch.tensor(list(delete.shape), dtype=torch.int64)
        tensors[name + ".packed"] = torch.frombuffer(bytearray(packed), dtype=torch.uint8)
        total += delete.numel()
        deleted += int(delete.sum())
    save_file(tensors, os.path.join(mask_dir, "model.safetensors"))
    config = {"format": "packbits_zstd", "total_params": total, "zero_params": deleted}
    with open(os.path.join(mask_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)


def union(a, b):
    """Delete a weight if either mask deletes it."""
    result = {}
    for name in sorted(set(a) | set(b)):
        if name in a and name in b:
            result[name] = a[name] | b[name]
        elif name in a:
            result[name] = a[name].clone()
        else:
            result[name] = b[name].clone()
    return result


def intersect(a, b):
    """Delete a weight only if both masks delete it."""
    result = {}
    for name in sorted(set(a) & set(b)):
        result[name] = a[name] & b[name]
    return result


def difference(a, b):
    """Delete the weights that a deletes and b does not."""
    result = {}
    for name in sorted(a):
        if name in b:
            result[name] = a[name] & ~b[name]
        else:
            result[name] = a[name].clone()
    return result


def restrict(mask, layers=None, modules=None):
    """Keep deletions only in the given layers and modules.

    layers: a list of layer indices, or None for all layers.
    modules: a list such as ["mlp"], or None for all modules.
    Entries outside the selection are dropped, so those weights are kept.
    """
    result = {}
    for name in sorted(mask):
        if layers is not None and layer_of(name) not in layers:
            continue
        if modules is not None and module_of(name) not in modules:
            continue
        result[name] = mask[name].clone()
    return result


def count_deleted(mask):
    """Return the total number of deleted weights."""
    total = 0
    for name in mask:
        total += int(mask[name].sum())
    return total


def deleted_per_layer(mask):
    """Return a dict of layer index -> deleted count."""
    counts = {}
    for name in sorted(mask):
        layer = layer_of(name)
        counts[layer] = counts.get(layer, 0) + int(mask[name].sum())
    return counts
