# CPU test of adapter.full_module_parts, install_all and module_ablation_maps on a toy decoder.
#   .venv/bin/python mine/tests/test_module_parts.py
# For every layer and module it checks that the parts the adapter returns, once installed, give
# the same model output as a forward pass that leaves that one module out, and nothing else.
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
import adapter  # noqa: E402
import toy_model  # noqa: E402


def capture(module, store, name):
    def hook(mod, args, kwargs, output):
        store[name] = output.detach().clone()
    return module.register_forward_hook(hook, with_kwargs=True)


def test_parts_and_install():
    model, config = toy_model.build()
    info, layers = adapter.describe(model, config)
    assert info["n_layers"] == 6 and info["d_model"] == toy_model.D
    assert info["layers"][0]["attn"]["n_heads"] == toy_model.HEADS
    assert info["layers"][4]["experts"]["style"] == "module_list"
    ids = torch.randint(0, toy_model.VOCAB, (3, 7), generator=torch.Generator().manual_seed(1))
    decoder = model.get_decoder()
    base = decoder(ids)
    checked = 0
    for l in range(info["n_layers"]):
        for module in ["attn", "mlp"]:
            parts = adapter.full_module_parts(info, l, module)
            assert [p["kind"] for p in parts] == toy_model.EXPECTED_KINDS[l][module], (l, module, parts)
            if not parts:
                continue
            for p in parts:
                assert adapter.check_part(info, p) is None, (l, module, adapter.check_part(info, p))
            # reference: the same forward pass with that one module left out, no hooks
            layers[l].skip = module
            reference = decoder(ids)
            layers[l].skip = None
            assert not torch.allclose(reference, base, atol=1e-4), ("the module does nothing in the toy", l, module)
            seen = {}
            watchers = []
            if layers[l].attn_module() is not None:
                watchers.append(capture(layers[l].attn_module(), seen, "attn"))
            watchers.append(capture(layers[l].ff_module(), seen, "mlp"))
            handles = []
            try:
                adapter.install_all(info, layers, parts, handles)
                assert len(handles) > 0
                hooked = decoder(ids)
            finally:
                adapter.remove_all(handles)
                adapter.remove_all(watchers)
            # the module's own output is exactly zero, the other module of the layer still works
            assert float(seen[module].abs().max()) == 0.0, (l, module)
            other = "mlp" if module == "attn" else "attn"
            if other in seen:
                assert float(seen[other].abs().max()) > 1e-3, (l, module)
            # and the whole model matches the reference, so no other layer or module was touched
            assert torch.allclose(hooked, reference, atol=1e-6), (l, module, float((hooked - reference).abs().max()))
            # the hooks are gone afterwards
            assert torch.equal(decoder(ids), base), (l, module)
            checked += 1
    assert checked == 11  # 5 layers with attention + 6 with a feed-forward module


def test_all_indices():
    model, config = toy_model.build()
    info, layers = adapter.describe(model, config)
    attn = adapter.full_module_parts(info, 0, "attn")
    assert attn == [{"kind": "attn_heads", "layer": 0, "indices": [0, 1]}]
    mlp = adapter.full_module_parts(info, 0, "mlp")
    assert mlp == [{"kind": "mlp_cluster", "layer": 0, "indices": list(range(toy_model.WIDTH))}]
    moe = adapter.full_module_parts(info, 1, "mlp")
    assert moe == [{"kind": "expert", "layer": 1, "indices": list(range(toy_model.N_EXPERTS))}]
    both = adapter.full_module_parts(info, 3, "mlp")
    assert [len(p["indices"]) for p in both] == [toy_model.WIDTH, toy_model.N_EXPERTS]
    listed = adapter.full_module_parts(info, 4, "mlp")
    assert listed == [{"kind": "expert", "layer": 4, "indices": list(range(toy_model.N_EXPERTS))}]
    assert adapter.full_module_parts(info, 2, "attn") == []
    try:
        adapter.full_module_parts(info, 0, "conv")
        raise AssertionError("an unknown module name must raise")
    except adapter.AdapterError:
        pass


def test_half_of_the_parts_is_not_enough():
    # layer 3 has a shared expert and routed experts: either part alone leaves a feed-forward output
    model, config = toy_model.build()
    info, layers = adapter.describe(model, config)
    ids = torch.randint(0, toy_model.VOCAB, (2, 5), generator=torch.Generator().manual_seed(2))
    for part in adapter.full_module_parts(info, 3, "mlp"):
        seen = {}
        watcher = capture(layers[3].ff_module(), seen, "mlp")
        handles = []
        try:
            adapter.install_all(info, layers, [part], handles)
            model.get_decoder()(ids)
        finally:
            adapter.remove_all(handles)
            watcher.remove()
        assert float(seen["mlp"].abs().max()) > 1e-3, part["kind"]


def test_projection_bias_stays():
    # The parts zero the input of the out-projection. A bias on it is a constant and stays.
    torch.manual_seed(3)
    layer = toy_model.Layer(toy_model.Attention(bias=True), "mlp", toy_model.DenseMLP())
    model = toy_model.Model([layer])
    model.eval()
    info, layers = adapter.describe(model, toy_model.Config())
    ids = torch.randint(0, toy_model.VOCAB, (2, 5), generator=torch.Generator().manual_seed(4))
    seen = {}
    watcher = capture(layers[0].attn_module(), seen, "attn")
    handles = []
    try:
        adapter.install_all(info, layers, adapter.full_module_parts(info, 0, "attn"), handles)
        with torch.no_grad():
            model.get_decoder()(ids)
    finally:
        adapter.remove_all(handles)
        watcher.remove()
    bias = layers[0].self_attn.o_proj.bias.detach()
    assert torch.allclose(seen["attn"], bias.expand_as(seen["attn"]), atol=1e-7)


def test_install_all_refuses_a_bad_part():
    model, config = toy_model.build()
    info, layers = adapter.describe(model, config)
    good = adapter.full_module_parts(info, 0, "attn")[0]
    bad = {"kind": "attn_heads", "layer": 0, "indices": [toy_model.HEADS]}
    handles = []
    try:
        adapter.install_all(info, layers, [good, bad], handles)
        raise AssertionError("an out-of-range head must raise")
    except adapter.AdapterError as e:
        assert "out of range" in str(e)
    finally:
        # the good part was installed before the bad one failed: the caller's finally removes it
        assert len(handles) == 1
        adapter.remove_all(handles)
    assert handles == []


def mean_logprob(model, ids):
    # mean log-prob of each next token, like the engine's teacher-forced score
    logits = model(input_ids=ids).logits
    lp = torch.log_softmax(logits[:, :-1].float(), dim=-1)
    return float(lp.gather(2, ids[:, 1:, None]).mean())


def test_module_ablation_maps():
    model, config = toy_model.build()
    info, layers = adapter.describe(model, config)
    gen = torch.Generator().manual_seed(5)
    texts = {"a|search": torch.randint(0, toy_model.VOCAB, (4, 9), generator=gen),
             "b|search": torch.randint(0, toy_model.VOCAB, (4, 9), generator=gen)}
    calls = []

    def score_all():
        calls.append(1)
        return {name: mean_logprob(model, texts[name]) for name in texts}
    base = score_all()
    calls.clear()
    out = adapter.module_ablation_maps(info, layers, list(texts.keys()), score_all)
    assert len(calls) == 11  # one pass per module that exists: 5 attention + 6 feed-forward
    assert out["parts"] == toy_model.EXPECTED_KINDS
    for module in ["attn", "mlp"]:
        assert sorted(out[module].keys()) == ["a|search", "b|search"]
        for name in texts:
            assert len(out[module][name]) == info["n_layers"]
            for l in range(info["n_layers"]):
                got = out[module][name][l]
                if not toy_model.EXPECTED_KINDS[l][module]:
                    assert got is None, (module, name, l)
                    continue
                layers[l].skip = module
                want = mean_logprob(model, texts[name])
                layers[l].skip = None
                assert abs(got - want) < 1e-5, (module, name, l, got, want)
                assert abs(got - base[name]) > 1e-6, (module, name, l)
    # no hook is left behind
    assert abs(mean_logprob(model, texts["a|search"]) - base["a|search"]) < 1e-9

    def broken():
        raise RuntimeError("scoring failed")
    try:
        adapter.module_ablation_maps(info, layers, list(texts.keys()), broken)
        raise AssertionError("the scoring error must reach the caller")
    except RuntimeError:
        pass
    assert abs(mean_logprob(model, texts["a|search"]) - base["a|search"]) < 1e-9


if __name__ == "__main__":
    with torch.no_grad():
        test_parts_and_install()
        test_all_indices()
        test_half_of_the_parts_is_not_enough()
        test_projection_bias_stays()
        test_install_all_refuses_a_bad_part()
        test_module_ablation_maps()
    print("ok")
