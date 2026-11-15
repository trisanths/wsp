"""Fetch selected tensors from safetensors files with byte-range reads.

A safetensors file starts with 8 bytes (little-endian uint64) that give the
header length N. Then come N bytes of JSON, then the tensor data. We read
the header first, then read only the bytes of the tensors we need. This
way we never download a whole model.
"""

import json
import os
import struct

import torch
from safetensors.torch import load_file, save_file

DTYPES = {
    "F32": torch.float32,
    "F16": torch.float16,
    "BF16": torch.bfloat16,
    "I64": torch.int64,
    "I32": torch.int32,
    "U8": torch.uint8,
    "BOOL": torch.bool,
}

# BF16 and F16 bytes are read as a 16-bit integer type first, then viewed.
RAW_DTYPES = {
    "BF16": torch.int16,
}


def read_header(read_range):
    """Return (header_dict, data_start) for a safetensors file."""
    length_bytes = read_range(0, 8)
    header_length = struct.unpack("<Q", length_bytes)[0]
    header_bytes = read_range(8, header_length)
    header = json.loads(header_bytes.decode("utf-8"))
    if "__metadata__" in header:
        del header["__metadata__"]
    return header, 8 + header_length


def _tensor_from_bytes(data, dtype_name, shape):
    """Build a tensor from raw bytes."""
    if dtype_name not in DTYPES:
        raise ValueError(f"unsupported dtype {dtype_name}")
    if len(data) == 0:
        return torch.zeros(shape, dtype=DTYPES[dtype_name])
    buffer = bytearray(data)
    if dtype_name in RAW_DTYPES:
        raw = torch.frombuffer(buffer, dtype=RAW_DTYPES[dtype_name])
        tensor = raw.view(DTYPES[dtype_name])
    else:
        tensor = torch.frombuffer(buffer, dtype=DTYPES[dtype_name])
    return tensor.reshape(shape).clone()


def read_tensors(read_range, names):
    """Read only the named tensors. Return a dict of name -> tensor."""
    header, data_start = read_header(read_range)
    for name in names:
        if name not in header:
            raise KeyError(f"tensor not found in safetensors header: {name}")
    result = {}
    for name in names:
        entry = header[name]
        begin, end = entry["data_offsets"]
        data = read_range(data_start + begin, end - begin)
        result[name] = _tensor_from_bytes(data, entry["dtype"], entry["shape"])
    return result


def local_range_reader(path):
    """Return a read_range function over a local file."""

    def read_range(offset, length):
        with open(path, "rb") as f:
            f.seek(offset)
            return f.read(length)

    return read_range


def http_range_reader(url, token=None):
    """Return a read_range function that uses HTTP Range requests."""
    import requests

    def read_range(offset, length):
        if length == 0:
            return b""
        end = offset + length - 1
        headers = {"Range": f"bytes={offset}-{end}"}
        if token:
            headers["Authorization"] = f"Bearer {token}"
        response = requests.get(url, headers=headers, allow_redirects=True, timeout=120)
        if response.status_code not in (200, 206):
            raise RuntimeError(f"range request failed with status {response.status_code}: {url}")
        return response.content

    return read_range


def _cache_path(cache_dir, repo_id, name):
    folder = os.path.join(cache_dir, repo_id.replace("/", "__"))
    return os.path.join(folder, name + ".safetensors")


def _load_cached(cache_dir, repo_id, name):
    """Return the cached tensor, or None."""
    path = _cache_path(cache_dir, repo_id, name)
    if not os.path.exists(path):
        return None
    return load_file(path, device="cpu")["tensor"]


def _save_cached(cache_dir, repo_id, name, tensor):
    path = _cache_path(cache_dir, repo_id, name)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    save_file({"tensor": tensor.contiguous()}, path)


def _shard_map(repo_id, names, revision):
    """Return a dict of shard file name -> list of tensor names."""
    from huggingface_hub import hf_hub_download
    from huggingface_hub.utils import EntryNotFoundError

    try:
        index_path = hf_hub_download(repo_id, "model.safetensors.index.json", revision=revision)
    except EntryNotFoundError:
        return {"model.safetensors": list(names)}

    with open(index_path) as f:
        weight_map = json.load(f)["weight_map"]
    shards = {}
    for name in names:
        if name not in weight_map:
            raise KeyError(f"tensor not found in index: {name}")
        shard = weight_map[name]
        if shard not in shards:
            shards[shard] = []
        shards[shard].append(name)
    return shards


def fetch_tensors(repo_id, names, revision="main", cache_dir=None):
    """Fetch the named tensors from a Hugging Face repo. Return name -> tensor."""
    result = {}
    missing = []
    for name in names:
        tensor = None
        if cache_dir is not None:
            tensor = _load_cached(cache_dir, repo_id, name)
        if tensor is None:
            missing.append(name)
        else:
            result[name] = tensor
    if len(missing) == 0:
        return result

    from huggingface_hub import get_token, hf_hub_url

    # get_token reads HF_TOKEN and the token file that "hf auth login" saves,
    # the same way hf_hub_download does.
    token = get_token()
    shards = _shard_map(repo_id, missing, revision)
    for shard, shard_names in shards.items():
        url = hf_hub_url(repo_id, shard, revision=revision)
        read_range = http_range_reader(url, token=token)
        fetched = read_tensors(read_range, shard_names)
        for name, tensor in fetched.items():
            result[name] = tensor
            if cache_dir is not None:
                _save_cached(cache_dir, repo_id, name, tensor)
    return result


def fetch_config(repo_id, revision="main"):
    """Return the transformers config of a Hugging Face repo."""
    from transformers import AutoConfig

    return AutoConfig.from_pretrained(repo_id, revision=revision, trust_remote_code=True)
