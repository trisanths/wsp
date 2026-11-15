# CPU test of the engine's skill_maps(modules=...) on a toy decoder, and of maps.build_entry on its
# result. It runs the real skill_maps, _score, _skill_sets, _tokenized, _batches and _stat_hook code
# of engine_app.EngineBase; only padding and the row scorer are replaced, because they need a GPU.
#   .venv/bin/python mine/tests/test_skill_maps_engine.py
# Nothing here talks to Modal: the engine class is used as a plain Python class, and the test
# blocks every network connection while it imports the engine file and while it runs.
import importlib.util
import json
import math
import os
import socket
import sys

import torch

if "pytest" in sys.modules and importlib.util.find_spec("modal") is None:
    # The repo's own pytest run may collect this file in an environment without modal (it is not a
    # base dependency there). Run as a script, the missing import fails loudly.
    import pytest
    pytest.skip("modal is not installed", allow_module_level=True)

REAL_NETWORK = (socket.socket.connect, socket.getaddrinfo)


def blocked(*args, **kwargs):
    raise RuntimeError("network is blocked in this test")


def block_network():
    socket.socket.connect = blocked
    socket.getaddrinfo = blocked


def restore_network():
    # put the real functions back, so a test runner that loads this file keeps its network
    socket.socket.connect, socket.getaddrinfo = REAL_NETWORK


HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
import adapter  # noqa: E402
import maps  # noqa: E402
import toy_model  # noqa: E402

block_network()
try:
    import engine_app  # noqa: E402
finally:
    restore_network()


class ToyTokenizer:
    pad_token_id = 0

    def __call__(self, text, add_special_tokens=True):
        ids = [2 + (ord(ch) % (toy_model.VOCAB - 2)) for ch in text]
        if add_special_tokens:
            ids = [1] + ids
        return {"input_ids": ids}


class CpuEngine(engine_app.EngineBase):
    def __init__(self):
        self.model, self.config = toy_model.build()
        self.adapter = adapter
        self.info, self.layers = adapter.describe(self.model, self.config)
        self.vocab = toy_model.VOCAB
        self.load_seconds = 0.0
        self.tok = ToyTokenizer()
        self.tok_cache = {}
        self.base_cache = {}
        self.builtin = {}
        self.row_calls = 0

    def _pad(self, rows, batch):
        longest = max(len(rows[i]["ids"]) for i in batch)
        ids = torch.zeros((len(batch), longest), dtype=torch.long)
        mask = torch.zeros((len(batch), longest), dtype=torch.long)
        for j in range(len(batch)):
            r = rows[batch[j]]["ids"]
            ids[j, :len(r)] = torch.tensor(r)
            mask[j, :len(r)] = 1
        return ids, mask

    def _score_rows(self, rows):
        # mean log-prob of the continuation tokens, one row at a time
        self.row_calls += 1
        total = 0.0
        count = 0
        with torch.no_grad():
            for r in rows:
                logits = self.model(input_ids=torch.tensor([r["ids"]])).logits[0]
                lp = torch.log_softmax(logits.float(), dim=-1)
                for t in range(r["start"], len(r["ids"])):
                    total += float(lp[t - 1, r["ids"][t]])
                    count += 1
        mean = total / count
        return {"lp": round(mean, 5), "score": round(math.exp(mean), 5), "n_tokens": count}


def items(words):
    out = []
    for i in range(len(words)):
        out.append({"context": words[i] + " is", "continuation": " " + words[(i + 1) % len(words)],
                    "digits_only": False})
    return out


def payload():
    law = items(["tort", "contract", "appeal", "statute", "verdict", "clause"])
    general = items(["river", "garden", "window", "morning", "letter", "bridge"])
    return {"skills": {
        "law": {"hash": "law1", "control": None, "items": {"search": law[:4], "heldout": law[4:]}},
        maps.REFERENCE: {"hash": "gen1", "control": None, "items": {"search": general[:4], "heldout": general[4:]}},
    }}


def reference_lp(engine, key, layer, skip):
    # the same score with the layer's module left out by the toy model itself, no hooks
    sets = engine._skill_sets(payload())
    rows = engine._tokenized(sets[key][0], sets[key][1])
    engine.layers[layer].skip = skip
    try:
        return engine._score_rows(rows)["lp"]
    finally:
        engine.layers[layer].skip = None


def test_engine_and_entry():
    block_network()
    try:
        check_engine_and_entry()
    finally:
        restore_network()


def check_engine_and_entry():
    engine = CpuEngine()
    n_layers = engine.info["n_layers"]
    names = ["law|search", "law|heldout", maps.REFERENCE + "|search", maps.REFERENCE + "|heldout"]

    plain = engine.skill_maps(payload())
    assert sorted(plain.keys()) == ["base", "expert_freq", "info", "layer_lp", "seconds"]
    # base (4 sets) plus one scoring pass per layer
    assert engine.row_calls == 4 + 4 * n_layers, engine.row_calls
    assert sorted(plain["layer_lp"].keys()) == sorted(names)

    engine.row_calls = 0
    full = engine.skill_maps(payload(), modules=True)
    assert sorted(full.keys()) == ["attn_lp", "base", "expert_freq", "info", "layer_lp", "mlp_lp", "module_parts",
                                   "seconds"]
    # the base is cached; one pass per layer, plus one per attention module (5) and feed-forward module (6)
    assert engine.row_calls == 4 * n_layers + 4 * 11, engine.row_calls
    # the fields a plain caller reads do not change
    assert full["layer_lp"] == plain["layer_lp"] and full["base"] == plain["base"]
    assert full["expert_freq"] == plain["expert_freq"]
    # routing counts only for the fused-expert layers (1 and 3), not for the ModuleList layer (4)
    assert sorted(full["expert_freq"]["law|search"].keys()) == ["1", "3"]
    assert full["module_parts"] == toy_model.EXPECTED_KINDS

    keys = {"law|search": ("law", "search"), maps.REFERENCE + "|heldout": (maps.REFERENCE, "heldout")}
    for name in keys:
        for l in range(n_layers):
            assert abs(full["layer_lp"][name][l] - reference_lp(engine, keys[name], l, "all")) < 2e-5, (name, l)
            for module in ["attn", "mlp"]:
                got = full[module + "_lp"][name][l]
                if not toy_model.EXPECTED_KINDS[l][module]:
                    assert got is None, (name, l, module)
                else:
                    want = reference_lp(engine, keys[name], l, module)
                    assert abs(got - want) < 2e-5, (name, l, module, got, want)
    for module in ["attn", "mlp"]:
        assert sorted(full[module + "_lp"].keys()) == sorted(names)
        for name in names:
            assert len(full[module + "_lp"][name]) == n_layers

    # the engine result goes through maps.build_entry and to JSON as it is
    sources = {"law": "toy law text", maps.REFERENCE: "toy general text"}
    row = {"params_b": 0.001, "family": "toy", "license": "mit"}
    entry = json.loads(json.dumps(maps.build_entry("org/toy", full, sources, row)))
    mm = entry["module_maps"]["law"]
    assert len(mm["attn"]) == n_layers and len(mm["mlp"]) == n_layers and len(mm["module_share"]) == n_layers
    assert mm["attn"][2] is None and all(v is not None for v in mm["mlp"])
    assert [i for i in range(n_layers) if mm["attn"][i] is None] == [2]
    base = full["base"]
    for l in [0, 3, 4]:
        for module in ["attn", "mlp"]:
            own = base["law|search"]["lp"] - reference_lp(engine, ("law", "search"), l, module)
            ref = base[maps.REFERENCE + "|search"]["lp"] - reference_lp(engine, (maps.REFERENCE, "search"), l, module)
            assert abs(mm[module][l] - (own - ref)) < 1e-4, (l, module, mm[module][l], own - ref)
    assert entry["module_parts"] == toy_model.EXPECTED_KINDS
    assert maps.REFERENCE not in entry["module_maps"]

    plain_entry = maps.build_entry("org/toy", plain, sources, row)
    assert "module_maps" not in plain_entry and "module_parts" not in plain_entry
    assert plain_entry["layer_maps"]["law"]["scores"] == entry["layer_maps"]["law"]["scores"]


if __name__ == "__main__":
    test_engine_and_entry()
    print("ok")
