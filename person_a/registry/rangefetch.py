# Fetch single tensors from a Hugging Face model without downloading the whole model.
#
# A .safetensors file starts with 8 bytes (header length), then a JSON header that lists
# every tensor's dtype, shape and byte range. We read just the header with an HTTP range
# request, then fetch only the byte ranges of the tensors we need.
#
#   .venv/bin/python registry/rangefetch.py maps      -> build tensor maps for every donor
#   .venv/bin/python registry/rangefetch.py test      -> fetch one small tensor and check it
import csv
import json
import os
import struct
import sys

import requests

HERE = os.path.dirname(os.path.abspath(__file__))
MAPS_DIR = os.path.join(HERE, "tensor_maps")


def hf_token():
    path = os.path.expanduser("~/.cache/huggingface/token")
    if os.path.exists(path):
        with open(path) as f:
            return f.read().strip()
    return os.environ.get("HF_TOKEN")


def file_url(repo_id, filename):
    return "https://huggingface.co/" + repo_id + "/resolve/main/" + filename


def get_range(url, start, end):
    # end is inclusive, as in the HTTP Range header
    headers = {"Range": "bytes=" + str(start) + "-" + str(end)}
    token = hf_token()
    if token:
        headers["Authorization"] = "Bearer " + token
    response = requests.get(url, headers=headers, timeout=120, allow_redirects=True)
    if response.status_code != 206:
        raise RuntimeError("range request failed: HTTP " + str(response.status_code) + " for " + url)
    return response.content


def read_header(repo_id, filename):
    url = file_url(repo_id, filename)
    first8 = get_range(url, 0, 7)
    header_len = struct.unpack("<Q", first8)[0]
    header_bytes = get_range(url, 8, 8 + header_len - 1)
    header = json.loads(header_bytes)
    data_start = 8 + header_len  # tensor offsets are relative to this
    return header, data_start


def list_shards(repo_id):
    # Prefer the official index: some repos (e.g. Mistral) also ship a second copy of the
    # weights in another format (consolidated.safetensors), which must not be counted twice.
    index_url = file_url(repo_id, "model.safetensors.index.json")
    response = requests.get(index_url, timeout=60, allow_redirects=True)
    if response.status_code == 200:
        shards = []
        weight_map = response.json()["weight_map"]
        for name in weight_map:
            if weight_map[name] not in shards:
                shards.append(weight_map[name])
        shards.sort()
        return shards

    response = requests.get("https://huggingface.co/api/models/" + repo_id, timeout=60)
    response.raise_for_status()
    shards = []
    for item in response.json().get("siblings", []):
        name = item["rfilename"]
        if name.endswith(".safetensors"):
            shards.append(name)
    shards.sort()
    return shards


def build_tensor_map(repo_id):
    # tensor name -> {file, dtype, shape, start, end}  (absolute byte positions, end exclusive)
    tensor_map = {}
    for shard in list_shards(repo_id):
        header, data_start = read_header(repo_id, shard)
        for name in header:
            if name == "__metadata__":
                continue
            entry = header[name]
            begin = entry["data_offsets"][0]
            finish = entry["data_offsets"][1]
            tensor_map[name] = {
                "file": shard,
                "dtype": entry["dtype"],
                "shape": entry["shape"],
                "start": data_start + begin,
                "end": data_start + finish,
            }
    return tensor_map


DTYPES = {"BF16": "bfloat16", "F16": "float16", "F32": "float32", "I64": "int64", "I32": "int32", "U8": "uint8"}


def fetch_tensor(repo_id, tensor_map, name):
    import torch

    entry = tensor_map[name]
    raw = get_range(file_url(repo_id, entry["file"]), entry["start"], entry["end"] - 1)
    dtype = getattr(torch, DTYPES[entry["dtype"]])
    tensor = torch.frombuffer(bytearray(raw), dtype=dtype)
    return tensor.reshape(entry["shape"])


def load_donors():
    rows = []
    with open(os.path.join(HERE, "donors.csv")) as f:
        for row in csv.DictReader(f):
            rows.append(row)
    return rows


def build_all_maps():
    os.makedirs(MAPS_DIR, exist_ok=True)
    for row in load_donors():
        out_path = os.path.join(MAPS_DIR, row["short_name"] + ".json")
        if os.path.exists(out_path):
            print("have", row["short_name"])
            continue
        try:
            tensor_map = build_tensor_map(row["hf_id"])
        except Exception as e:
            print("FAILED", row["short_name"], repr(e)[:200])
            continue
        total = 0
        for name in tensor_map:
            total = total + tensor_map[name]["end"] - tensor_map[name]["start"]
        with open(out_path, "w") as f:
            json.dump(tensor_map, f)
        print(row["short_name"], "|", len(tensor_map), "tensors |", round(total / 1e9, 1), "GB total")


def test_fetch():
    # 1) On a tiny test model (a few MB): every range-fetched tensor must equal a full read.
    import torch
    from huggingface_hub import hf_hub_download
    from safetensors import safe_open

    tiny = "hf-internal-testing/tiny-random-LlamaForCausalLM"
    tiny_map = build_tensor_map(tiny)
    mismatches = 0
    for name in tiny_map:
        fetched = fetch_tensor(tiny, tiny_map, name)
        path = hf_hub_download(tiny, tiny_map[name]["file"])
        with safe_open(path, framework="pt") as f:
            reference = f.get_tensor(name)
        if not torch.equal(fetched, reference):
            mismatches = mismatches + 1
    print("tiny model:", len(tiny_map), "tensors checked,", mismatches, "mismatches")

    # 2) On a real donor: fetch one MLP row block and check size and sanity.
    repo_id = "Qwen/Qwen2.5-Math-1.5B"
    with open(os.path.join(MAPS_DIR, "qwenmath1.5b.json")) as f:
        tensor_map = json.load(f)
    name = "model.layers.3.post_attention_layernorm.weight"
    tensor = fetch_tensor(repo_id, tensor_map, name)
    print(name, tuple(tensor.shape), tensor.dtype, "mean", round(float(tensor.float().mean()), 4),
          "all finite:", bool(torch.isfinite(tensor.float()).all()))


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        test_fetch()
    else:
        build_all_maps()
