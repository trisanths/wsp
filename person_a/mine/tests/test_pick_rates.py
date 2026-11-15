# Checks the router pick rates on a toy MoE model, on the CPU: adapter.expert_count_layers,
# adapter.pick_rates, and the engine methods expert_pick_rates and builtin_info. The engine methods
# run as plain functions on a local object (no Modal call, no GPU); the routing hook is the engine's own.
#   .venv/bin/python mine/tests/test_pick_rates.py
import hashlib
import json
import os
import sys
import types

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import adapter  # noqa: E402
import engine_app  # noqa: E402
import evidence  # noqa: E402

D = 4
N_EXP = 4


class FusedExperts(torch.nn.Module):
    # transformers-5 style: 3D weights, called as experts(hidden_states, top_k_index, top_k_weights)
    def __init__(self):
        super().__init__()
        self.gate_up_proj = torch.nn.Parameter(torch.zeros(N_EXP, D, D))
        self.down_proj = torch.nn.Parameter(torch.zeros(N_EXP, D, D))

    def forward(self, hidden_states, top_k_index, top_k_weights):
        return torch.zeros_like(hidden_states)


class FusedMoE(torch.nn.Module):
    # The toy router reads the token id from the first hidden value: token t picks experts t % 4 and
    # (t + 1) % 4. That makes every pick count below checkable by hand.
    def __init__(self):
        super().__init__()
        self.experts = FusedExperts()

    def forward(self, x):
        flat = x.reshape(-1, D)
        first = flat[:, 0].long() % N_EXP
        idx = torch.stack([first, (first + 1) % N_EXP], dim=1)
        weights = torch.full(idx.shape, 0.5)
        return x + self.experts(flat, idx, weights).reshape(x.shape)


class ListMoE(torch.nn.Module):
    # older style: one module per expert, which the adapter marks "module_list"
    def __init__(self):
        super().__init__()
        self.experts = torch.nn.ModuleList([torch.nn.Linear(D, D) for _ in range(3)])

    def forward(self, x):
        return x


class DenseMLP(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.down_proj = torch.nn.Linear(8, D)

    def forward(self, x):
        return x


class Layer(torch.nn.Module):
    def __init__(self, mlp):
        super().__init__()
        self.mlp = mlp

    def forward(self, x):
        return self.mlp(x)


class ToyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.ModuleList([Layer(FusedMoE()), Layer(ListMoE()), Layer(DenseMLP())])

    def get_decoder(self):
        return self

    def forward(self, input_ids=None, attention_mask=None, use_cache=False):
        x = torch.zeros(input_ids.shape + (D,))
        x[..., 0] = input_ids.float()
        for layer in self.layers:
            x = layer(x)
        return types.SimpleNamespace(logits=x)


class ToyTokenizer:
    pad_token_id = 0  # a padded position would pick experts 0 and 1 if the mask did not hide it

    def __call__(self, text, add_special_tokens=True):
        return {"input_ids": [int(w) for w in text.split()]}


class LocalEngine(engine_app.EngineBase):
    # The engine's own helpers, except _pad: the engine moves each batch to the GPU there.
    def _pad(self, rows, batch):
        longest = max(len(rows[i]["ids"]) for i in batch)
        ids = torch.full((len(batch), longest), self.tok.pad_token_id, dtype=torch.long)
        mask = torch.zeros((len(batch), longest), dtype=torch.long)
        for j in range(len(batch)):
            r = rows[batch[j]]["ids"]
            ids[j, :len(r)] = torch.tensor(r)
            mask[j, :len(r)] = 1
        return ids, mask


def raw(name):
    # the plain function under Modal's method wrapper
    return engine_app.EngineBase.__dict__[name]._get_raw_f()


def item(context, continuation):
    return {"context": context, "continuation": continuation, "digits_only": False}


def make_engine():
    model = ToyModel()
    engine = LocalEngine.__new__(LocalEngine)
    engine.adapter = adapter
    engine.model = model
    engine.info, engine.layers = adapter.describe(model, types.SimpleNamespace(hidden_size=D))
    engine.vocab = 10
    engine.tok = ToyTokenizer()
    engine.tok_cache = {}
    engine.builtin = {"general_text": {"name": "general_text", "description": "toy general text",
                                       "items": {"search": [item("2 2 1", "1 1")], "heldout": [item("3 3", "3 3")]}}}
    return engine


def test_layers():
    engine = make_engine()
    info = engine.info
    assert info["layers"][0]["experts"] == {"path": "mlp.experts", "n_experts": N_EXP}, info
    assert info["layers"][1]["experts"]["style"] == "module_list", info
    assert "experts" not in info["layers"][2], info
    counted, unsupported = adapter.expert_count_layers(info)
    assert counted == [0] and list(unsupported) == ["1"] and "module_list" in unsupported["1"], (counted, unsupported)
    counted, unsupported = adapter.expert_count_layers(info, [1, 2, 7])
    assert counted == [], counted
    assert "module_list" in unsupported["1"] and "no experts" in unsupported["2"], unsupported
    assert "range" in unsupported["7"], unsupported
    counted, unsupported = adapter.expert_count_layers(info, [0])
    assert counted == [0] and unsupported == {}, (counted, unsupported)


def test_pick_rates_math():
    # 6 picks of expert 1 and 2 of expert 3 over 4 tokens; a layer with no token gives None
    sums = {5: {"sum": torch.tensor([0.0, 6.0, 0.0, 2.0]), "n": 4.0}, 6: {"sum": torch.zeros(4), "n": 0.0}}
    assert adapter.pick_rates(sums) == {"5": {"rates": [0.0, 1.5, 0.0, 0.5], "tokens": 4}, "6": None}


def test_engine_method():
    engine = make_engine()
    payload = {"skills": {"sk": {"hash": "h1", "control": None,
                                 "items": {"search": [item("1 1 1", "1 1"), item("2 2", "2")],
                                           "heldout": [item("3 3 3", "3")]}},
                          "general_text": {"builtin": True}}}
    res = raw("expert_pick_rates")(engine, payload, ["sk", "general_text"], [0, 1], 64)
    assert res["counted_layers"] == [0] and "module_list" in res["unsupported"]["1"] and res["missing"] == [], res

    # sk search: tokens 1 1 1 1 1 and 2 2 2 in one padded batch. The first position of each row is not
    # counted, so 4 tokens "1" (experts 1 and 2) and 2 tokens "2" (experts 2 and 3) remain: 6 tokens,
    # picks 0, 4, 6, 2. The padding (token 0) would add picks of experts 0 and 1 if it were counted.
    got = res["rates"]["sk|search"]
    assert got["n_items"] == 2 and got["layers"]["0"]["tokens"] == 6, got
    assert got["layers"]["0"]["rates"] == [0.0, 0.666667, 1.0, 0.333333], got
    assert abs(sum(got["layers"]["0"]["rates"]) - 2.0) < 1e-5  # top-2 router: the rates add up to 2
    # sk held-out: 3 3 3 3, 3 counted tokens, each picks experts 3 and 0
    assert res["rates"]["sk|heldout"]["layers"]["0"] == {"rates": [1.0, 0.0, 0.0, 1.0], "tokens": 3}, res
    # general search: 2 2 1 1 1, counted 2 1 1 1: expert 1 three times, expert 2 four times, expert 3 once
    assert res["rates"]["general_text|search"]["layers"]["0"] == {"rates": [0.0, 0.75, 1.0, 0.25], "tokens": 4}, res
    assert res["rates"]["general_text|heldout"]["layers"]["0"] == {"rates": [1.0, 0.0, 0.0, 1.0], "tokens": 3}, res
    # the hooks are gone after the call
    assert len(engine.layers[0].mlp.experts._forward_pre_hooks) == 0

    # the row fields: ln(0.666667 / 0.75) = -0.1178, ln(1 / 1) = 0, ln(0.333333 / 0.25) = 0.2877
    routing, note = evidence.routing_fields([1, 2, 3], 0, "sk", res)
    assert note is None, note
    assert routing["experts"]["1"]["search"] == {"skill_rate": 0.666667, "general_rate": 0.75, "log_ratio": -0.1178}
    assert routing["experts"]["2"]["search"]["log_ratio"] == 0.0 and routing["experts"]["3"]["search"]["log_ratio"] == 0.2877
    assert routing["experts"]["1"]["heldout"] == {"skill_rate": 0.0, "general_rate": 0.0, "log_ratio": None}
    assert routing["experts"]["3"]["heldout"] == {"skill_rate": 1.0, "general_rate": 1.0, "log_ratio": 0.0}
    assert routing["tokens"]["sk|search"] == 6 and routing["n_items"]["sk|search"] == 2, routing

    # n_items limits the items of each half: only "1 1 1 1 1" is read
    res = raw("expert_pick_rates")(engine, payload, ["sk"], [0], 1)
    assert res["rates"]["sk|search"] == {"n_items": 1, "layers": {"0": {"rates": [0.0, 1.0, 1.0, 0.0], "tokens": 4}}}, res

    # a module_list layer: nothing is counted, and the row says why
    res = raw("expert_pick_rates")(engine, payload, ["sk", "general_text"], [1], 64)
    assert res["rates"] == {} and res["counted_layers"] == [], res
    routing, note = evidence.routing_fields([0], 1, "sk", res)
    assert routing is None and "module_list" in note, note

    # a skill that the payload does not hold is reported, not guessed
    res = raw("expert_pick_rates")(engine, payload, ["nope"], [0], 64)
    assert res["missing"] == ["nope|search", "nope|heldout"] and res["rates"] == {}, res


def test_builtin_info():
    engine = make_engine()
    out = raw("builtin_info")(engine)
    items = engine.builtin["general_text"]["items"]
    digest = hashlib.sha256(json.dumps(items, sort_keys=True).encode()).hexdigest()[:16]
    assert out == {"general_text": {"description": "toy general text", "dataset": None, "n_search": 1,
                                    "n_heldout": 1, "items_sha": digest}}, out
    # what the client makes of it
    sources = evidence.builtin_sources(out, {})
    assert sources["general_text"] == {"source": "builtin", "description": "toy general text", "datasets": []}


test_layers()
test_pick_rates_math()
test_engine_method()
test_builtin_info()
print("ok")
