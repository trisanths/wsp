# CPU test for the engine's per-layer unit rankings (attribution_units, activation_units), on a toy
# language model with dense layers and one MoE layer. Nothing here calls Modal: the test builds the plain
# EngineBase object by hand and runs its helper methods locally.
#   .venv/bin/python mine/tests/test_engine_units.py
#
# The engine code moves tensors with .cuda(). This test sends those calls to the CPU.
# Each ranking is checked against an independent computation:
#   - attribution: the derivative of the mean answer NLL with respect to a scale on each unit
#     (scale 1 -> 0 removes the unit), taken item by item with no padding and no batching
#   - activation: mean |activation| per unit over the same positions, item by item
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import adapter  # noqa: E402
import engine_app  # noqa: E402

VOCAB = 32
D = 8
N_EXPERTS = 6
TOP_K = 2


class ToyAttn(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.v_proj = torch.nn.Linear(D, D)
        self.o_proj = torch.nn.Linear(D, D, bias=False)

    def forward(self, x):
        # causal mean over earlier positions: a later (padding) position never changes an earlier one
        steps = torch.arange(1, x.shape[1] + 1, dtype=x.dtype).reshape(1, -1, 1)
        return self.o_proj(torch.cumsum(self.v_proj(x), dim=1) / steps)


class ToyMlp(torch.nn.Module):
    def __init__(self, width):
        super().__init__()
        self.up_proj = torch.nn.Linear(D, width)
        self.down_proj = torch.nn.Linear(width, D, bias=False)

    def forward(self, x):
        return self.down_proj(torch.tanh(self.up_proj(x)))


class ToyExperts(torch.nn.Module):
    # fused experts, called as experts(hidden_states, top_k_index, top_k_weights)
    def __init__(self):
        super().__init__()
        self.gate_up_proj = torch.nn.Parameter(torch.randn(N_EXPERTS, D, 12) * 0.3)
        self.down_proj = torch.nn.Parameter(torch.randn(N_EXPERTS, 12, D) * 0.3)

    def forward(self, hidden_states, top_k_index, top_k_weights):
        out = torch.zeros_like(hidden_states)
        for j in range(top_k_index.shape[1]):
            e = top_k_index[:, j]
            up = torch.tanh(torch.einsum("nd,ndh->nh", hidden_states, self.gate_up_proj[e]))
            out = out + top_k_weights[:, j:j + 1] * torch.einsum("nh,nhd->nd", up, self.down_proj[e])
        return out


class ToyMoe(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.router = torch.nn.Linear(D, N_EXPERTS)
        self.experts = ToyExperts()

    def forward(self, x):
        flat = x.reshape(-1, D)
        weights, index = torch.topk(torch.softmax(self.router(flat), dim=-1), TOP_K, dim=-1)
        return self.experts(flat, index, weights).reshape(x.shape)


class ToyListMoe(torch.nn.Module):
    # older MoE code: a ModuleList of expert MLPs. Each expert runs on the tokens routed to it, and the
    # block scales its output by the routing weight.
    def __init__(self):
        super().__init__()
        self.router = torch.nn.Linear(D, N_EXPERTS)
        self.experts = torch.nn.ModuleList([ToyMlp(10) for _ in range(N_EXPERTS)])

    def forward(self, x):
        flat = x.reshape(-1, D)
        weights, index = torch.topk(torch.softmax(self.router(flat), dim=-1), TOP_K, dim=-1)
        out = torch.zeros_like(flat)
        for i in range(N_EXPERTS):
            rows, slot = torch.where(index == i)
            if rows.numel() == 0:
                continue                              # an expert with no token is not called
            out = out.index_add(0, rows, self.experts[i](flat[rows]) * weights[rows, slot].unsqueeze(-1))
        return out.reshape(x.shape)


class ToyLayer(torch.nn.Module):
    def __init__(self, mlp):
        super().__init__()
        self.self_attn = ToyAttn()
        self.mlp = mlp

    def forward(self, x):
        x = x + self.self_attn(x)
        return x + self.mlp(x)


class Output:
    pass


class ToyLM(torch.nn.Module):
    def __init__(self, moe):
        super().__init__()
        self.embed = torch.nn.Embedding(VOCAB, D)
        self.layers = torch.nn.ModuleList([ToyLayer(ToyMlp(20)), ToyLayer(ToyMlp(16)), ToyLayer(moe)])
        self.unembed = torch.nn.Linear(D, VOCAB, bias=False)
        self.config = ToyConfig()

    def get_decoder(self):
        return self

    def forward(self, input_ids=None, attention_mask=None, use_cache=False):
        x = self.embed(input_ids)
        for layer in self.layers:
            x = layer(x)
        out = Output()
        out.logits = self.unembed(x)
        return out


class ToyConfig:
    hidden_size = D
    num_attention_heads = 4


class ToyTok:
    pad_token_id = 0

    def __call__(self, text, add_special_tokens=True):
        ids = [3 + (ord(ch) * 7) % (VOCAB - 3) for ch in text]
        if add_special_tokens:
            ids = [1] + ids
        return {"input_ids": ids}


def items(seed, n):
    words = ["court", "ruling", "appeal", "statute", "the", "of", "river", "a", "tort", "city", "judge", "old"]
    g = torch.Generator().manual_seed(seed)
    out = []
    for _ in range(n):
        pick = torch.randint(0, len(words), (6,), generator=g).tolist()
        n_ctx = 2 + int(torch.randint(0, 3, (1,), generator=g))
        out.append({"context": " ".join(words[i] for i in pick[:n_ctx]), "continuation": " " + " ".join(words[i] for i in pick[n_ctx:]),
                    "digits_only": False})
    return out


def make_engine(list_experts=False):
    torch.manual_seed(0)
    model = ToyLM(ToyListMoe() if list_experts else ToyMoe())
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    e = object.__new__(engine_app.EngineBase)
    e.__dict__.update({"adapter": adapter, "model": model, "tok": ToyTok(), "vocab": VOCAB, "model_key": "toy",
                       "is_digit": torch.zeros(VOCAB, dtype=torch.bool), "builtin": {}, "tok_cache": {}, "base_cache": {}})
    e.info, e.layers = adapter.describe(model, model.config)
    return e


PAYLOAD = {"skills": {
    "law": {"hash": "toy-law", "items": {"search": items(1, 9), "heldout": items(2, 3)}, "control": None},
    "general_text": {"hash": "toy-general", "items": {"search": items(3, 9), "heldout": items(4, 3)}, "control": None}}}


def module_for(e, kind, layer):
    key = {"mlp_cluster": "mlp", "attn_heads": "attn", "expert": "experts"}[kind]
    return adapter.get_module(e.layers[layer], e.info["layers"][layer][key]["path"])


def n_units(e, kind, layer):
    entry = e.info["layers"][layer]
    return {"mlp_cluster": lambda: entry["mlp"]["width"], "attn_heads": lambda: entry["attn"]["n_heads"],
            "expert": lambda: entry["experts"]["n_experts"]}[kind]()


def scored(e, skill, n_items):
    # (ids, first scored position) per item, as the engine tokenises them
    out = []
    for it in PAYLOAD["skills"][skill]["items"]["search"][:n_items]:
        ctx = e.tok(it["context"], add_special_tokens=True)["input_ids"]
        cont = e.tok(it["continuation"], add_special_tokens=False)["input_ids"]
        out.append((ctx + cont, len(ctx)))
    return out


def reference_attribution(e, kind, layer, skill, n_items):
    # -d(mean answer NLL)/d(scale of each unit), item by item
    n = n_units(e, kind, layer)
    scale = torch.ones(n, requires_grad=True)
    mod = module_for(e, kind, layer)
    if kind == "expert":
        def pre(module, args):
            return (args[0], args[1], args[2] * scale[args[1]])
    elif kind == "attn_heads":
        head_dim = e.info["layers"][layer]["attn"]["head_dim"]

        def pre(module, args):
            return (args[0] * scale.repeat_interleave(head_dim),)
    else:
        def pre(module, args):
            return (args[0] * scale,)
    handle = mod.register_forward_pre_hook(pre)
    total = 0.0
    count = 0
    try:
        with torch.enable_grad():
            for ids, start in scored(e, skill, n_items):
                logits = e.model(input_ids=torch.tensor([ids])).logits[0]
                lp = torch.log_softmax(logits, dim=-1)
                for t in range(start, len(ids)):
                    total = total - lp[t - 1, ids[t]]
                    count += 1
            grad = torch.autograd.grad(total / count, [scale])[0]
    finally:
        handle.remove()
    return -grad, count


def reference_activation(e, kind, layer, skill, n_items):
    n = n_units(e, kind, layer)
    total = torch.zeros(n)
    positions = [0]
    mod = module_for(e, kind, layer)

    def pre(module, args):
        if kind == "expert":
            idx = args[1].reshape(-1, TOP_K)[1:]          # the first position is not counted
            total.add_(torch.bincount(idx.reshape(-1), minlength=n).float())
        elif kind == "attn_heads":
            x = args[0][0, 1:]
            total.add_(x.reshape(x.shape[0], n, -1).norm(dim=-1).sum(dim=0))
        else:
            total.add_(args[0][0, 1:].abs().sum(dim=0))
        positions[0] += (args[1].reshape(-1, TOP_K).shape[0] if kind == "expert" else args[0].shape[1]) - 1
    handle = mod.register_forward_pre_hook(pre)
    try:
        with torch.no_grad():
            for ids, start in scored(e, skill, n_items):
                e.model(input_ids=torch.tensor([ids]))
    finally:
        handle.remove()
    return total / positions[0], positions[0]


def no_hooks_left(e):
    for mod in e.model.modules():
        assert len(mod._forward_pre_hooks) == 0 and len(mod._forward_hooks) == 0, "a hook was left on " + type(mod).__name__


def test_attribution(e):
    n_items = 7
    for kind, layers in [("mlp_cluster", [0, 1]), ("attn_heads", [0, 1, 2]), ("expert", [2])]:
        assert e._unit_layers(kind, []) == layers, (kind, e._unit_layers(kind, []))
        # a small token budget: several batches of different sizes, with padding
        got, n_tokens = e._attribution_effects(PAYLOAD, kind, "law", layers, n_items, 40)
        assert sorted(got.keys()) == layers
        for layer in layers:
            want, count = reference_attribution(e, kind, layer, "law", n_items)
            assert n_tokens == count, (n_tokens, count)
            assert got[layer].shape == want.shape
            assert torch.allclose(got[layer], want, atol=2e-5, rtol=1e-3), (kind, layer, got[layer], want)
            assert float(want.abs().max()) > 1e-4                  # the check is not on zeros
        # one big batch gives the same numbers: the result does not depend on the batching
        whole, _ = e._attribution_effects(PAYLOAD, kind, "law", layers, n_items, 100000)
        for layer in layers:
            assert torch.allclose(whole[layer], got[layer], atol=2e-5, rtol=1e-3)
    no_hooks_left(e)


def test_activation(e):
    n_items = 9
    for kind, layers in [("mlp_cluster", [0, 1]), ("attn_heads", [0, 1, 2]), ("expert", [2])]:
        got, n_tokens = e._activation_means(PAYLOAD, kind, "law", layers, n_items)
        for layer in layers:
            want, count = reference_activation(e, kind, layer, "law", n_items)
            assert n_tokens == count, (n_tokens, count)
            assert torch.allclose(got[layer], want, atol=1e-5, rtol=1e-4), (kind, layer, got[layer], want)
        if kind == "expert":
            assert abs(float(got[2].sum()) - TOP_K) < 1e-4         # picks per token add up to top_k
    no_hooks_left(e)


def test_ranking(e):
    res = e._unit_ranking(PAYLOAD, "attribution", "mlp_cluster", "law", [], 5, 7, 40)
    assert res["method"] == "attribution" and res["how"] == "selective" and res["reference"] == "general_text"
    assert sorted(res["layers"].keys()) == ["0", "1"] and res["top_n"] == 5 and res["n_items"] == 7
    for layer in [0, 1]:
        mine, _ = reference_attribution(e, "mlp_cluster", layer, "law", 7)
        ref, _ = reference_attribution(e, "mlp_cluster", layer, "general_text", 7)
        want = adapter.top_units(mine, ref, 5, "selective")
        row = res["layers"][str(layer)]
        assert row["units"] == want["units"], (row["units"], want["units"])
        assert row["width"] == n_units(e, "mlp_cluster", layer) and len(row["units"]) == 5
        assert all(abs(a - b) < 1e-4 for a, b in zip(row["skill"], want["skill"]))
        assert all(abs(a - b) < 1e-4 for a, b in zip(row["general"], want["general"]))
        assert adapter.check_part(e.info, {"kind": "mlp_cluster", "layer": layer, "indices": row["units"]}) is None
    # only the asked layers
    one = e._unit_ranking(PAYLOAD, "attribution", "attn_heads", "law", [1], 2, 7, 40)
    assert list(one["layers"].keys()) == ["1"] and len(one["layers"]["1"]["units"]) == 2
    # the general_text pass is kept per (method, kind, layers) and used again
    assert len(e.unit_ref_cache) == 2
    e._unit_ranking(PAYLOAD, "attribution", "attn_heads", "law", [1], 4, 7, 40)
    assert len(e.unit_ref_cache) == 2
    # activation ranking: ratios, floor 1; expert pick rates
    act = e._unit_ranking(PAYLOAD, "activation", "expert", "law", [], 3, 9, None)
    assert act["method"] == "activation" and act["how"] == "ratio" and list(act["layers"].keys()) == ["2"]
    mine, _ = reference_activation(e, "expert", 2, "law", 9)
    ref, _ = reference_activation(e, "expert", 2, "general_text", 9)
    want = adapter.top_units(mine, ref, 3, "ratio")
    assert act["layers"]["2"]["units"] == want["units"] and act["layers"]["2"]["floor"] == 1.0
    # a skill that is not in the payload is an error, not a silent empty result
    for bad in ["medicine"]:
        try:
            e._unit_ranking(PAYLOAD, "attribution", "mlp_cluster", bad, [], 5, 7, 40)
            raise AssertionError("expected ValueError")
        except ValueError:
            pass
    no_hooks_left(e)


def test_module_list_experts():
    # Experts kept as a ModuleList: attribution ranks them with a probe on each expert's output. The check
    # is the derivative of the mean answer NLL with respect to a scale on that output, item by item.
    e = make_engine(list_experts=True)
    entry = e.info["layers"][2]["experts"]
    assert entry == {"path": "mlp.experts", "n_experts": N_EXPERTS, "style": "module_list"}, entry
    assert e._unit_layers("expert", []) == [2] and e._unit_layers("expert", [], "attribution") == [2]
    n_items = 7
    experts = adapter.get_module(e.layers[2], entry["path"])
    scale = torch.ones(N_EXPERTS, requires_grad=True)
    handles = []
    for i in range(N_EXPERTS):
        handles.append(experts[i].register_forward_hook(lambda module, args, output, i=i: output * scale[i]))
    total = 0.0
    count = 0
    try:
        with torch.enable_grad():
            for ids, start in scored(e, "law", n_items):
                lp = torch.log_softmax(e.model(input_ids=torch.tensor([ids])).logits[0], dim=-1)
                for t in range(start, len(ids)):
                    total = total - lp[t - 1, ids[t]]
                    count += 1
            want = -torch.autograd.grad(total / count, [scale])[0]
    finally:
        adapter.remove_all(handles)
    for budget in [40, 100000]:                                    # several padded batches, then one batch
        got, n_tokens = e._attribution_effects(PAYLOAD, "expert", "law", [2], n_items, budget)
        assert n_tokens == count and got[2].shape == want.shape
        assert torch.allclose(got[2], want, atol=2e-5, rtol=1e-3), (got[2], want)
    assert float(want.abs().max()) > 1e-4 and int((want != 0).sum()) >= 3      # the check is not on zeros
    res = e._unit_ranking(PAYLOAD, "attribution", "expert", "law", [], 3, n_items, 40)
    assert list(res["layers"].keys()) == ["2"] and res["layers"]["2"]["width"] == N_EXPERTS
    part = {"kind": "expert", "layer": 2, "indices": res["layers"]["2"]["units"]}
    assert adapter.check_part(e.info, part) is None
    # the activation ranking needs the router's picks, which a ModuleList does not show: no layers
    assert e._unit_layers("expert", [], "activation") == []
    assert e._unit_ranking(PAYLOAD, "activation", "expert", "law", [], 3, 9, None)["layers"] == {}
    # the dense kinds of the same model are ranked as before
    assert e._unit_layers("mlp_cluster", [], "activation") == [0, 1]
    no_hooks_left(e)


class OtherDevice(torch.Tensor):
    # Stands for a tensor on another GPU. An operation that mixes it with a plain tensor raises, as
    # PyTorch does for two devices. Operations on it alone work. .to(device) gives a plain tensor.
    moves = []

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if func is torch.Tensor.to:
            cls.moves.append(args[1] if len(args) > 1 else (kwargs or {}).get("device"))
            return torch.tensor(args[0].tolist())
        for a in args:
            if type(a) is torch.Tensor:
                raise RuntimeError("Expected all tensors to be on the same device, but found at least two devices")
        return super().__torch_function__(func, types, args, kwargs)


def test_mask_follows_the_layer(e):
    # Multi-GPU tiers: the padding mask is made on cuda:0 and the layers are on other GPUs. The mask in
    # this test is on "another device" than every activation, so a statistic that uses it where it is
    # raises. The activation path must move it to the activation's device first.
    tagged = torch.ones(2, 3, dtype=torch.long).as_subclass(OtherDevice)
    tagged[:, 0] = 0
    assert int(tagged.sum()) == 4                                   # operations on the mask alone work
    try:
        torch.ones(2, 3) * tagged
        raise AssertionError("the device tag must refuse a mixed operation")
    except RuntimeError as err:
        assert "same device" in str(err)
    real_pad = e._pad

    def pad_elsewhere(rows, batch):
        ids, mask = real_pad(rows, batch)
        return ids, mask.as_subclass(OtherDevice)
    e._pad = pad_elsewhere
    try:
        for kind, layers in [("mlp_cluster", [0, 1]), ("attn_heads", [0, 1, 2]), ("expert", [2])]:
            del OtherDevice.moves[:]
            got, n_tokens = e._activation_means(PAYLOAD, kind, "law", layers, 9)
            assert len(OtherDevice.moves) >= len(layers) and all(d == torch.device("cpu") for d in OtherDevice.moves)
            for layer in layers:
                want, count = reference_activation(e, kind, layer, "law", 9)
                assert n_tokens == count
                assert torch.allclose(got[layer], want, atol=1e-5, rtol=1e-4), (kind, layer)
    finally:
        del e._pad
    no_hooks_left(e)


def plain_method(name):
    # the function under @modal.method(), to call it locally with no Modal app
    wrapped = engine_app.EngineBase.__dict__[name]
    if hasattr(wrapped, "raw_f"):
        return wrapped.raw_f
    return wrapped._get_raw_f()


def test_quantized_refusal(e):
    raw = plain_method("attribution_units")
    res = raw(e, PAYLOAD, "attn_heads", "law", [0], 2, 7, 40)       # bf16-style model: runs
    assert res["method"] == "attribution" and list(res["layers"].keys()) == ["0"]
    e.model.config.quantization_config = {"quant_method": "fp8"}
    try:
        raw(e, PAYLOAD, "attn_heads", "law", [0], 2, 7, 40)
        raise AssertionError("a quantized model must be refused")
    except RuntimeError as err:
        assert "needs bf16 weights" in str(err)                      # mine/sweep.py falls back on this text
    finally:
        del e.model.config.quantization_config
    assert plain_method("activation_units")(e, PAYLOAD, "attn_heads", "law", [0], 2, 9)["method"] == "activation"


if __name__ == "__main__":
    # send the engine's .cuda() moves to the CPU
    torch.Tensor.cuda = lambda self, *args, **kwargs: self
    real_tensor = torch.tensor

    def cpu_tensor(*args, **kwargs):
        if kwargs.get("device") == "cuda":
            kwargs["device"] = "cpu"
        return real_tensor(*args, **kwargs)
    torch.tensor = cpu_tensor

    engine = make_engine()
    assert [sorted(k for k in entry if k != "layer") for entry in engine.info["layers"]] == [
        ["attn", "mlp"], ["attn", "mlp"], ["attn", "experts"]]
    test_attribution(engine)
    test_activation(engine)
    test_ranking(engine)
    test_mask_follows_the_layer(engine)
    test_module_list_experts()
    test_quantized_refusal(engine)
    print("ok")
