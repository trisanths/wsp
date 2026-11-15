"""LFM2 and LFM3 donors: hybrid models with short conv layers and attention layers.

LFM2 names its parts differently from Llama: feed_forward (w1, w3, w2),
ffn_norm before the MLP, operator_norm before attention or conv, and
out_proj in attention. Its MLP width is not the config's intermediate_size
when block_auto_adjust_ff_dim is on. LFM3 adds sandwich norms and a gate
per attention head. Tiny random configs, CPU only.
"""

import os

import pytest
import torch

transformers = pytest.importorskip("transformers")

from wsp.fetch import local_range_reader, read_tensors
from wsp.parts import (
    attn_tensor_names,
    block_tensor_names,
    build_attn_heads,
    build_block_part,
    build_mlp_cluster,
    find_layer_modules,
    mlp_tensor_names,
    mlp_width,
)
from tests.test_block import capture_block, tiny_donor
from tests.test_review_parts import layer_attention_io

ATTENTION_LAYER = 1
CONV_LAYER = 2


def saved_donor(tmp_path):
    """Save the tiny LFM2 donor and return (config, donor, read) where read(names) reads tensors from disk."""
    config, donor = tiny_donor("lfm2")
    donor.save_pretrained(str(tmp_path))
    reader = local_range_reader(os.path.join(str(tmp_path), "model.safetensors"))

    def read(names):
        return read_tensors(reader, names)

    return config, donor, read


def test_lfm2_layer_modules_use_lfm2_names():
    config, _ = tiny_donor("lfm2")
    found = find_layer_modules(config, ATTENTION_LAYER)
    assert found["mlp_prefix"] == "model.layers.1.feed_forward"
    assert found["pre_norm_prefix"] == "model.layers.1.ffn_norm"
    assert found["input_norm_prefix"] == "model.layers.1.operator_norm"
    assert found["post_norm"] is None
    conv = find_layer_modules(config, CONV_LAYER)
    assert conv["attn"] is None


def test_lfm2_mlp_width_comes_from_the_weights():
    # intermediate_size 96 becomes int(2 * 96 / 3) = 64, rounded up to a
    # multiple of 16.
    config, donor = tiny_donor("lfm2")
    assert config.intermediate_size == 96
    assert mlp_width(config, ATTENTION_LAYER) == 64
    assert donor.model.layers[ATTENTION_LAYER].feed_forward.w1.weight.shape[0] == 64


@pytest.mark.parametrize("layer", [ATTENTION_LAYER, CONV_LAYER])
def test_lfm2_mlp_cluster_matches_donor_neurons(tmp_path, layer):
    config, donor, read = saved_donor(tmp_path)
    names = mlp_tensor_names(config, layer)
    assert f"model.layers.{layer}.feed_forward.w1.weight" in names
    assert f"model.layers.{layer}.ffn_norm.weight" in names
    indices = torch.tensor([0, 5, 17, 40, 63])
    part = build_mlp_cluster(config, layer, indices, read(names))

    block = donor.model.layers[layer]
    mlp = block.feed_forward
    x = torch.randn(2, 6, config.hidden_size)
    with torch.no_grad():
        normed = block.ffn_norm(x)
        gate = torch.nn.functional.silu(normed @ mlp.w1.weight[indices].T)
        up = normed @ mlp.w3.weight[indices].T
        want = (gate * up) @ mlp.w2.weight[:, indices].T
        assert torch.allclose(part(x), want, atol=1e-5)

        # All neurons together give the whole donor MLP.
        whole = build_mlp_cluster(config, layer, torch.arange(64), read(names))
        assert torch.allclose(whole(x), mlp(block.ffn_norm(x)), atol=1e-5)


def test_lfm2_attn_heads_from_an_attention_layer(tmp_path):
    config, donor, read = saved_donor(tmp_path)
    names = attn_tensor_names(config, ATTENTION_LAYER)
    assert "model.layers.1.self_attn.out_proj.weight" in names
    assert "model.layers.1.operator_norm.weight" in names
    part = build_attn_heads(config, ATTENTION_LAYER, torch.arange(4), read(names))
    ids = torch.randint(3, 300, (2, 7))
    x, attn = layer_attention_io(donor, donor.model.layers[ATTENTION_LAYER], ids)
    with torch.no_grad():
        assert torch.allclose(part(x), attn, atol=1e-5)


def test_lfm2_attn_heads_on_a_conv_layer_raises():
    config, donor = tiny_donor("lfm2")
    with pytest.raises(ValueError, match="no self_attn"):
        attn_tensor_names(config, CONV_LAYER)
    with pytest.raises(ValueError, match="no self_attn"):
        build_attn_heads(config, CONV_LAYER, [0], donor.state_dict())


def test_lfm2_uncached_block_over_a_conv_layer(tmp_path):
    config, donor, read = saved_donor(tmp_path)
    names = block_tensor_names(config, CONV_LAYER, CONV_LAYER)
    assert "model.layers.2.conv.conv.weight" in names
    part = build_block_part(config, CONV_LAYER, CONV_LAYER, read(names))
    assert part.recurrent == [True]

    ids = torch.randint(3, 300, (2, 7))
    x, y = capture_block(donor, CONV_LAYER, CONV_LAYER, ids)
    with torch.no_grad():
        assert torch.allclose(part(x), y - x, atol=1e-5)


def test_lfm2_uncached_conv_block_with_left_padding():
    # Pads are zeroed before the conv, so a padded row matches the same
    # row run alone without pads.
    config, donor = tiny_donor("lfm2")
    part = build_block_part(config, CONV_LAYER, CONV_LAYER, donor.state_dict())
    torch.manual_seed(4)
    x = torch.randn(1, 5, config.hidden_size)
    padded = torch.cat([torch.randn(1, 3, config.hidden_size), x], dim=1)
    mask = torch.tensor([[0, 0, 0, 1, 1, 1, 1, 1]])
    with torch.no_grad():
        alone = part(x)
        with_pads = part(padded, attention_mask=mask)
    assert torch.allclose(with_pads[:, 3:], alone, atol=1e-5)


def test_lfm3_attn_heads_keep_the_head_gates():
    # LFM3 attention gates each head with one row of g_proj and adds a
    # post_operator_norm. A part with all heads equals the donor's
    # attention followed by that norm.
    config, donor = tiny_donor("lfm3")
    layer = donor.model.layers[1]
    part = build_attn_heads(config, 1, torch.arange(4), donor.state_dict())
    assert part.attn.g_proj.weight.shape[0] == 4
    ids = torch.randint(3, 300, (2, 7))
    x, attn = layer_attention_io(donor, layer, ids)
    with torch.no_grad():
        assert torch.allclose(part(x), layer.post_operator_norm(attn), atol=1e-5)

    # Two heads take their own g_proj rows.
    two = build_attn_heads(config, 1, [3, 1], donor.state_dict())
    assert torch.equal(two.attn.g_proj.weight, layer.self_attn.g_proj.weight[[3, 1]])


def test_lfm3_mlp_cluster_keeps_the_sandwich_norm():
    config, donor = tiny_donor("lfm3")
    layer = donor.model.layers[CONV_LAYER]
    width = mlp_width(config, CONV_LAYER)
    part = build_mlp_cluster(config, CONV_LAYER, torch.arange(width), donor.state_dict())
    assert part.post_norm is not None
    x = torch.randn(2, 5, config.hidden_size)
    with torch.no_grad():
        want = layer.post_ffn_norm(layer.feed_forward(layer.ffn_norm(x)))
        assert torch.allclose(part(x), want, atol=1e-5)
