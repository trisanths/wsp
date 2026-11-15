import pytest
import torch
from torch import nn

transformers = pytest.importorskip("transformers")

from wsp.graft import Graft, attach_grafts, freeze_all_but_glue, glue_parameters, replace_hidden
from wsp.parts import build_mlp_cluster, find_layer_modules, mlp_tensor_names


def tiny_llama():
    torch.manual_seed(0)
    config = transformers.LlamaConfig(
        hidden_size=64, intermediate_size=128, num_hidden_layers=4,
        num_attention_heads=4, num_key_value_heads=2, vocab_size=300,
    )
    return config, transformers.LlamaForCausalLM(config).eval()


def tiny_qwen2():
    torch.manual_seed(1)
    config = transformers.Qwen2Config(
        hidden_size=48, intermediate_size=96, num_hidden_layers=3,
        num_attention_heads=4, num_key_value_heads=2, vocab_size=300,
    )
    return config, transformers.Qwen2ForCausalLM(config).eval()


def donor_tensors(model):
    out = {}
    for name, value in model.state_dict().items():
        out[name] = value.detach().clone()
    return out


def test_cluster_equals_partial_mlp():
    config, donor = tiny_qwen2()
    layer = 1
    cluster = build_mlp_cluster(config, layer, torch.arange(96), donor_tensors(donor))
    x = torch.randn(2, 5, 48)
    block = donor.model.layers[layer]
    expected = block.mlp(block.post_attention_layernorm(x))
    assert torch.allclose(cluster(x), expected, atol=1e-5)


def test_cluster_subset_is_partial_sum():
    config, donor = tiny_qwen2()
    layer = 2
    tensors = donor_tensors(donor)
    perm = torch.randperm(96)
    first = perm[:30]
    second = perm[30:]
    full = build_mlp_cluster(config, layer, torch.arange(96), tensors)
    part_a = build_mlp_cluster(config, layer, first, tensors)
    part_b = build_mlp_cluster(config, layer, second, tensors)
    x = torch.randn(2, 5, 48)
    normed = full.pre_norm(x)
    total = part_a.mlp(normed) + part_b.mlp(normed)
    assert torch.allclose(total, full.mlp(normed), atol=1e-5)


def test_tensor_names_exist_in_checkpoint():
    config, donor = tiny_qwen2()
    names = mlp_tensor_names(config, 0)
    state = donor.state_dict()
    assert len(names) == 4
    for name in names:
        assert name in state


def make_graft(donor_config, donor, d_host, layer=1, count=16):
    tensors = donor_tensors(donor)
    indices = torch.randperm(96)[:count]
    part = build_mlp_cluster(donor_config, layer, indices, tensors)
    return Graft(d_host, donor_config.hidden_size, part)


def test_fresh_graft_keeps_host_logits():
    host_config, host = tiny_llama()
    donor_config, donor = tiny_qwen2()
    ids = torch.randint(0, 300, (2, 7))
    with torch.no_grad():
        before = host(ids).logits
    graft = make_graft(donor_config, donor, 64)
    handles = attach_grafts(host, [(1, graft)])
    with torch.no_grad():
        after = host(ids).logits
    assert torch.equal(before, after)
    # One layer hook per graft, plus the host pre-hook that stores the padding mask.
    assert len(handles) == 2
    assert len(host.wsp_grafts) == 1


def test_gradients_reach_glue_only():
    host_config, host = tiny_llama()
    donor_config, donor = tiny_qwen2()
    graft = make_graft(donor_config, donor, 64)
    nn.init.normal_(graft.w_out.weight, std=0.02)
    attach_grafts(host, [(1, graft)])
    freeze_all_but_glue(host)
    ids = torch.randint(0, 300, (2, 7))
    host(ids).logits.sum().backward()
    assert len(glue_parameters(host)) == 2
    assert graft.w_in.weight.grad.abs().sum() > 0
    assert graft.w_out.weight.grad.abs().sum() > 0
    for param in host.parameters():
        is_glue = param is graft.w_in.weight or param is graft.w_out.weight
        if not is_glue:
            assert param.grad is None


def test_tuple_and_tensor_layer_outputs():
    def double(h):
        return h * 2

    x = torch.ones(2, 3)
    out = replace_hidden(x, double)
    assert torch.equal(out, x * 2)

    out = replace_hidden((x, "cache", None), double)
    assert isinstance(out, tuple)
    assert len(out) == 3
    assert torch.equal(out[0], x * 2)
    assert out[1] == "cache"

    class TupleLayer(nn.Module):
        def forward(self, h):
            return (h, "extra")

    class Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([TupleLayer()])

        def forward(self, h):
            return self.layers[0](h)

    model = Tiny()
    graft = nn.Identity()
    attach_grafts(model, [(0, graft)])
    result = model(x)
    assert isinstance(result, tuple)
    assert torch.equal(result[0], x)


def test_gemma_style_norms():
    if not hasattr(transformers, "Gemma2Config"):
        pytest.skip("Gemma2Config missing")
    torch.manual_seed(2)
    config = transformers.Gemma2Config(
        hidden_size=48, intermediate_size=96, num_hidden_layers=2,
        num_attention_heads=3, num_key_value_heads=1, head_dim=16, vocab_size=300,
    )
    donor = transformers.Gemma2ForCausalLM(config).eval()
    found = find_layer_modules(config, 1)
    assert found["pre_norm_prefix"] == "model.layers.1.pre_feedforward_layernorm"
    assert found["post_norm_prefix"] == "model.layers.1.post_feedforward_layernorm"
    assert found["mlp_prefix"] == "model.layers.1.mlp"
    cluster = build_mlp_cluster(config, 1, torch.arange(96), donor_tensors(donor))
    block = donor.model.layers[1]
    x = torch.randn(2, 5, 48)
    expected = block.post_feedforward_layernorm(block.mlp(block.pre_feedforward_layernorm(x)))
    assert torch.allclose(cluster(x), expected, atol=1e-5)
