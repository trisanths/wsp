# Loaded inside the weight sandbox as `from wsp_helpers import *`.
# Gives the agent read access to the donor's weights and tokenizer, and nothing else:
# no model class, no transformers, no network.
import glob
import json
import os

import torch
from safetensors import safe_open
from tokenizers import Tokenizer

MODEL_DIR = os.environ["WSP_MODEL_DIR"]

with open(os.path.join(MODEL_DIR, "config.json")) as _f:
    CONFIG = json.load(_f)

_files = sorted(glob.glob(os.path.join(MODEL_DIR, "*.safetensors")))
_where = {}
for _path in _files:
    with safe_open(_path, framework="pt") as _f:
        for _name in _f.keys():
            _where[_name] = _path

tok = Tokenizer.from_file(os.path.join(MODEL_DIR, "tokenizer.json"))


def names():
    # All weight tensor names, sorted.
    return sorted(_where.keys())


def shape(name):
    with safe_open(_where[name], framework="pt") as f:
        return tuple(f.get_slice(name).get_shape())


def W(name):
    # One weight tensor as float32 (read lazily from disk).
    with safe_open(_where[name], framework="pt") as f:
        return f.get_tensor(name).float()


def decode(ids):
    # Token ids -> list of token strings.
    out = []
    for i in ids:
        out.append(tok.id_to_token(int(i)))
    return out


def encode(text):
    # Text -> token ids (no special tokens).
    return tok.encode(text, add_special_tokens=False).ids
