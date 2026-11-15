# CPU test for adapter.top_units, the per-layer selection behind the engine methods attribution_units
# and activation_units.
#   .venv/bin/python mine/tests/test_top_units.py
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import adapter  # noqa: E402


def close(a, b, tol=1e-5):
    return len(a) == len(b) and all(abs(x - y) <= tol for x, y in zip(a, b))


def test_selective_by_hand():
    mine = torch.tensor([0.5, 0.1, -0.2, 0.5, 0.3])
    ref = torch.tensor([0.1, -0.4, 0.0, 0.1, 0.3])
    # score = mine - max(ref, 0) = [0.4, 0.1, -0.2, 0.4, 0.0]; units 0 and 3 tie, the lower index goes first
    out = adapter.top_units(mine, ref, 3, "selective")
    assert out["width"] == 5 and out["floor"] == 0.0
    assert out["units"] == [0, 3, 1], out["units"]
    assert close(out["skill"], [0.5, 0.5, 0.1])
    assert close(out["general"], [0.1, 0.1, -0.4])
    assert close(out["score"], [0.4, 0.4, 0.1])
    # a negative reference value gives no bonus: unit 1 scores 0.1, not 0.5
    full = adapter.top_units(mine, ref, 99, "selective")
    assert full["units"] == [0, 3, 1, 4, 2], full["units"]
    assert close(full["score"], [0.4, 0.4, 0.1, 0.0, -0.2])


def test_ratio_by_hand():
    mine = torch.tensor([0.2, 0.2, 0.0])
    ref = torch.tensor([0.1, 0.4, 0.0])
    eps = 0.05 * (0.5 / 3) + 1e-6
    want = [(0.2 + eps) / (0.1 + eps), 1.0, (0.2 + eps) / (0.4 + eps)]
    out = adapter.top_units(mine, ref, 3, "ratio")
    assert out["floor"] == 1.0
    assert out["units"] == [0, 2, 1], out["units"]
    assert close(out["score"], want, 1e-4), (out["score"], want)


def test_edges():
    mine = torch.tensor([1.0, float("nan"), 3.0, float("inf"), 2.0])
    ref = torch.zeros(5)
    out = adapter.top_units(mine, ref, 5, "selective")
    assert out["units"] == [2, 4, 0], out["units"]          # the two non-finite units are left out
    assert out["width"] == 5
    assert adapter.top_units(mine, None, 2, "selective")["units"] == [2, 4]   # no reference: plain order
    assert adapter.top_units(torch.zeros(4), torch.zeros(4), 0, "selective")["units"] == []
    assert adapter.top_units(torch.zeros(4), torch.zeros(4), 9, "selective")["units"] == [0, 1, 2, 3]
    for bad in [lambda: adapter.top_units(torch.zeros(4), torch.zeros(3), 2, "selective"),
                lambda: adapter.top_units(torch.zeros(4), torch.zeros(4), 2, "other")]:
        try:
            bad()
        except adapter.AdapterError:
            continue
        raise AssertionError("expected AdapterError")
    # the indices are plain ints and the values plain floats, so the result goes through JSON
    out = adapter.top_units(torch.rand(8), torch.rand(8), 4, "selective")
    assert all(type(u) is int for u in out["units"]) and all(type(v) is float for v in out["score"])


class ToyAttn(torch.nn.Module):
    def __init__(self, d):
        super().__init__()
        self.o_proj = torch.nn.Linear(d, d, bias=False)


class ToyMlp(torch.nn.Module):
    def __init__(self, d, width):
        super().__init__()
        self.up_proj = torch.nn.Linear(d, width)
        self.down_proj = torch.nn.Linear(width, d, bias=False)

    def forward(self, x):
        return self.down_proj(torch.tanh(self.up_proj(x)))


class ToyLayer(torch.nn.Module):
    def __init__(self, d, width):
        super().__init__()
        self.self_attn = ToyAttn(d)
        self.mlp = ToyMlp(d, width)

    def forward(self, x):
        return x + self.mlp(x)


class ToyModel(torch.nn.Module):
    def __init__(self, d, width, n_layers):
        super().__init__()
        self.layers = torch.nn.ModuleList([ToyLayer(d, width) for _ in range(n_layers)])
        self.readout = torch.nn.Linear(d, 1, bias=False)

    def get_decoder(self):
        return self

    def loss(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.readout(x).sum()


class ToyConfig:
    hidden_size = 8
    num_attention_heads = 2


def toy_effects(model, info, layers, layer, x):
    # the engine's attribution estimate: a zero probe added to the MLP down-projection input,
    # effect = -(gradient * activation), summed over positions
    mod = adapter.get_module(layers[layer], info["layers"][layer]["mlp"]["path"])
    rec = {}

    def pre(module, args):
        probe = torch.zeros_like(args[0], requires_grad=True)
        rec["probe"] = probe
        rec["x"] = args[0].detach()
        return (args[0] + probe,)
    handle = mod.register_forward_pre_hook(pre)
    try:
        grad = torch.autograd.grad(model.loss(x), [rec["probe"]])[0]
    finally:
        handle.remove()
    prod = -(grad * rec["x"])
    return prod.reshape(-1, prod.shape[-1]).sum(dim=0)


def test_against_real_ablation():
    # In the last layer the loss is linear in the down-projection input, so the attribution estimate
    # equals the real change of the loss when a unit is zeroed. top_units must then return the units
    # whose real removal (adapter.install, the hook the gate uses) raises the "skill" loss most,
    # after the rise on the reference input is taken off.
    torch.manual_seed(0)
    width = 24
    model = ToyModel(ToyConfig.hidden_size, width, 2)
    for p in model.parameters():
        p.requires_grad_(False)
    info, layers = adapter.describe(model, ToyConfig())
    assert info["layers"][1]["mlp"]["width"] == width
    skill_x = torch.randn(3, 5, ToyConfig.hidden_size)
    ref_x = torch.randn(3, 5, ToyConfig.hidden_size)
    mine = toy_effects(model, info, layers, 1, skill_x)
    ref = toy_effects(model, info, layers, 1, ref_x)

    def real_rise(x, unit):
        handles = []
        base = float(model.loss(x))
        try:
            adapter.install(info, layers, {"kind": "mlp_cluster", "layer": 1, "indices": [unit]}, handles)
            return float(model.loss(x)) - base
        finally:
            adapter.remove_all(handles)
    real_skill = [real_rise(skill_x, u) for u in range(width)]
    real_ref = [real_rise(ref_x, u) for u in range(width)]
    assert close(mine.tolist(), real_skill, 1e-4), "attribution must match the real ablation on a linear tail"
    out = adapter.top_units(mine, ref, 6, "selective")
    real_score = [real_skill[u] - max(real_ref[u], 0.0) for u in range(width)]
    want = sorted(range(width), key=lambda u: (-real_score[u], u))[:6]
    assert out["units"] == want, (out["units"], want)
    assert close(out["skill"], [real_skill[u] for u in want], 1e-4)
    assert close(out["general"], [real_ref[u] for u in want], 1e-4)
    assert adapter.check_part(info, {"kind": "mlp_cluster", "layer": 1, "indices": out["units"]}) is None


if __name__ == "__main__":
    test_selective_by_hand()
    test_ratio_by_hand()
    test_edges()
    test_against_real_ablation()
    print("ok")
