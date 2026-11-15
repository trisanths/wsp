"""Regression tests for the parts.py review fixes.

Every test uses a tiny random config on the CPU and needs no network.
"""

import copy

import pytest
import torch

transformers = pytest.importorskip("transformers")
safetensors_torch = pytest.importorskip("safetensors.torch")

from wsp.parts import (
    attn_tensor_names,
    build_attn_heads,
    build_mlp_cluster,
    causal_mask,
    mlp_tensor_names,
)


def state_of(model):
    out = {}
    for name, value in model.state_dict().items():
        out[name] = value.detach().clone()
    return out


def eager(config):
    config._attn_implementation = "eager"
    return config


def layer_attention_io(model, layer_module, input_ids, attention_mask=None):
    """Run the whole donor and return (layer input, self_attn output) of one layer.

    This uses the model's own mask and rotary, so it is the true reference.
    """
    captured = {}

    def before(module, args, kwargs):
        if args:
            captured["x"] = args[0]
        else:
            captured["x"] = kwargs["hidden_states"]

    def after(module, args, kwargs, output):
        captured["attn"] = output[0]

    first = layer_module.register_forward_pre_hook(before, with_kwargs=True)
    second = layer_module.self_attn.register_forward_hook(after, with_kwargs=True)
    try:
        with torch.no_grad():
            model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
    finally:
        first.remove()
        second.remove()
    return captured["x"], captured["attn"]


def tiny_phi3():
    torch.manual_seed(0)
    config = eager(transformers.Phi3Config(
        hidden_size=32, intermediate_size=48, num_hidden_layers=2, num_attention_heads=4,
        num_key_value_heads=2, vocab_size=300, max_position_embeddings=64, pad_token_id=0,
    ))
    return config, transformers.Phi3ForCausalLM(config).eval()


def test_phi3_mlp_cluster_all_units_equal_donor_mlp():
    config, donor = tiny_phi3()
    layer = 1
    part = build_mlp_cluster(config, layer, torch.arange(48), state_of(donor))
    block = donor.model.layers[layer]
    x = torch.randn(2, 5, 32)
    with torch.no_grad():
        expected = block.mlp(block.post_attention_layernorm(x))
        assert torch.allclose(part(x), expected, atol=1e-5)


def test_phi3_mlp_cluster_subset_keeps_gate_and_up_paired():
    config, donor = tiny_phi3()
    layer = 0
    indices = torch.tensor([7, 1, 30])
    part = build_mlp_cluster(config, layer, indices, state_of(donor))
    block = donor.model.layers[layer]
    # The donor MLP with every other neuron's down_proj column zeroed gives
    # the output of just the chosen neurons.
    masked = copy.deepcopy(block.mlp)
    keep = torch.zeros(48, dtype=torch.bool)
    keep[indices] = True
    with torch.no_grad():
        masked.down_proj.weight[:, ~keep] = 0.0
        x = torch.randn(2, 5, 32)
        expected = masked(block.post_attention_layernorm(x))
        assert torch.allclose(part(x), expected, atol=1e-5)


def test_phi3_attn_heads_all_heads_equal_donor_attention():
    config, donor = tiny_phi3()
    layer = 1
    part = build_attn_heads(config, layer, torch.arange(4), state_of(donor))
    input_ids = torch.randint(1, 300, (2, 7))
    x, attn = layer_attention_io(donor, donor.model.layers[layer], input_ids)
    with torch.no_grad():
        assert torch.allclose(part(x), attn, atol=1e-5)


def test_phi3_attn_head_subsets_add_up():
    config, donor = tiny_phi3()
    tensors = state_of(donor)
    first = build_attn_heads(config, 0, torch.tensor([3, 0]), tensors)
    second = build_attn_heads(config, 0, torch.tensor([1, 2]), tensors)
    full = build_attn_heads(config, 0, torch.arange(4), tensors)
    x = torch.randn(2, 6, 32)
    with torch.no_grad():
        assert torch.allclose(first(x) + second(x), full(x), atol=1e-5)


def tiny_gemma3_text():
    torch.manual_seed(1)
    config = eager(transformers.Gemma3TextConfig(
        hidden_size=32, intermediate_size=64, num_hidden_layers=6, num_attention_heads=4,
        num_key_value_heads=2, head_dim=8, vocab_size=300, max_position_embeddings=64,
        sliding_window=4, rope_scaling={"rope_type": "linear", "factor": 8.0},
    ))
    return config, transformers.Gemma3ForCausalLM(config).eval()


@pytest.mark.parametrize("layer", [0, 5])
def test_gemma3_attn_part_equals_donor_attention(layer):
    config, donor = tiny_gemma3_text()
    assert config.layer_types[0] == "sliding_attention"
    assert config.layer_types[5] == "full_attention"
    part = build_attn_heads(config, layer, torch.arange(4), state_of(donor))
    block = donor.model.layers[layer]
    input_ids = torch.randint(1, 300, (2, 10))
    x, attn = layer_attention_io(donor, block, input_ids)
    with torch.no_grad():
        expected = block.post_attention_layernorm(attn)
        assert torch.allclose(part(x), expected, atol=1e-5)


def tiny_gemma3_multimodal():
    torch.manual_seed(2)
    text = dict(
        hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=4,
        num_key_value_heads=2, head_dim=8, vocab_size=300, sliding_window=4,
    )
    vision = dict(
        hidden_size=32, intermediate_size=64, num_hidden_layers=1, num_attention_heads=4,
        image_size=28, patch_size=14,
    )
    return transformers.Gemma3Config(text_config=text, vision_config=vision, mm_tokens_per_image=4)


def tiny_mistral3():
    torch.manual_seed(2)
    text = dict(
        model_type="mistral", hidden_size=32, intermediate_size=64, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, head_dim=8, vocab_size=300,
    )
    vision = dict(
        model_type="pixtral", hidden_size=32, intermediate_size=64, num_hidden_layers=1,
        num_attention_heads=4, image_size=28, patch_size=14, head_dim=8,
    )
    return transformers.Mistral3Config(text_config=text, vision_config=vision)


def saved_checkpoint(model, folder):
    """Save the model the way the hub does and return its tensors by name."""
    model.save_pretrained(folder)
    return safetensors_torch.load_file(str(folder / "model.safetensors"))


@pytest.mark.parametrize("make", [tiny_gemma3_multimodal, tiny_mistral3])
def test_multimodal_tensor_names_exist_in_saved_checkpoint(make, tmp_path):
    config = make()
    try:
        donor = transformers.AutoModelForCausalLM.from_config(config)
    except ValueError:
        donor = transformers.AutoModelForImageTextToText.from_config(config)
    donor.eval()
    saved = saved_checkpoint(donor, tmp_path)
    names = mlp_tensor_names(config, 1) + attn_tensor_names(config, 1)
    for name in names:
        assert name in saved, name
        assert name.startswith("language_model.model.layers.1.")

    # The checkpoint tensors build a part that matches the donor layer.
    layer = donor.model.language_model.layers[1]
    part = build_mlp_cluster(config, 1, torch.arange(64), saved)
    x = torch.randn(2, 5, 32)
    with torch.no_grad():
        if hasattr(layer, "pre_feedforward_layernorm"):
            expected = layer.post_feedforward_layernorm(layer.mlp(layer.pre_feedforward_layernorm(x)))
        else:
            expected = layer.mlp(layer.post_attention_layernorm(x))
        assert torch.allclose(part(x), expected, atol=1e-5)

    # Tensors keyed by module names (the rebuild path) still work.
    build_mlp_cluster(config, 1, torch.arange(64), state_of(donor))
    build_attn_heads(config, 1, torch.arange(4), saved)


def tiny_mixtral():
    torch.manual_seed(3)
    config = eager(transformers.MixtralConfig(
        hidden_size=32, intermediate_size=48, num_hidden_layers=2, num_attention_heads=4,
        num_key_value_heads=2, num_local_experts=4, num_experts_per_tok=2, vocab_size=300,
        max_position_embeddings=64,
    ))
    return config, transformers.MixtralForCausalLM(config).eval()


def test_mixtral_attn_heads_builds_and_equals_donor():
    config, donor = tiny_mixtral()
    names = attn_tensor_names(config, 0)
    assert "model.layers.0.self_attn.q_proj.weight" in names
    part = build_attn_heads(config, 0, torch.arange(4), state_of(donor))
    input_ids = torch.randint(1, 300, (2, 6))
    x, attn = layer_attention_io(donor, donor.model.layers[0], input_ids)
    with torch.no_grad():
        assert torch.allclose(part(x), attn, atol=1e-5)


def test_mixtral_mlp_cluster_raises_value_error():
    config, donor = tiny_mixtral()
    with pytest.raises(ValueError, match="expert"):
        mlp_tensor_names(config, 0)
    with pytest.raises(ValueError, match="expert"):
        build_mlp_cluster(config, 0, torch.tensor([0, 1]), state_of(donor))


def test_moe_mlp_cluster_raises_before_fetch():
    torch.manual_seed(4)
    config = transformers.Qwen3MoeConfig(
        hidden_size=32, intermediate_size=96, moe_intermediate_size=16, num_experts=4,
        num_experts_per_tok=2, num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        vocab_size=300, max_position_embeddings=64,
    )
    with pytest.raises(ValueError, match="sparse MoE"):
        mlp_tensor_names(config, 1)
    with pytest.raises(ValueError, match="sparse MoE"):
        build_mlp_cluster(config, 1, torch.tensor([0, 5, 7]), {})


def test_mistral_sliding_window_applied():
    torch.manual_seed(5)
    config = eager(transformers.MistralConfig(
        hidden_size=32, intermediate_size=48, num_hidden_layers=2, num_attention_heads=4,
        num_key_value_heads=2, vocab_size=300, max_position_embeddings=64, sliding_window=4,
    ))
    donor = transformers.MistralForCausalLM(config).eval()
    part = build_attn_heads(config, 1, torch.arange(4), state_of(donor))
    input_ids = torch.randint(1, 300, (2, 10))
    x, attn = layer_attention_io(donor, donor.model.layers[1], input_ids)
    with torch.no_grad():
        out = part(x)
        assert torch.allclose(out, attn, atol=1e-5)
        # A plain causal mask gives a different answer past the window, so
        # the match above needs the window.
        block = donor.model.layers[1]
        position_ids = torch.arange(10).view(1, 10).expand(2, 10)
        cos_sin = donor.model.rotary_emb(x, position_ids)
        plain = block.self_attn(
            hidden_states=block.input_layernorm(x),
            position_embeddings=cos_sin,
            attention_mask=causal_mask(2, 10, x.dtype, x.device),
        )[0]
        assert not torch.allclose(plain, attn, atol=1e-4)


def tiny_llama():
    torch.manual_seed(6)
    config = eager(transformers.LlamaConfig(
        hidden_size=64, intermediate_size=128, num_hidden_layers=2, num_attention_heads=4,
        num_key_value_heads=2, vocab_size=300, max_position_embeddings=64,
    ))
    return config, transformers.LlamaForCausalLM(config).eval()


def test_attn_part_left_padding_matches_unpadded():
    config, donor = tiny_llama()
    part = build_attn_heads(config, 1, torch.tensor([0, 3]), state_of(donor))
    torch.manual_seed(7)
    short = torch.randn(1, 4, 64)
    long = torch.randn(1, 7, 64)
    pads = torch.randn(1, 3, 64)
    batch = torch.cat([torch.cat([pads, short], dim=1), long], dim=0)
    mask = torch.ones(2, 7, dtype=torch.long)
    mask[0, :3] = 0
    with torch.no_grad():
        out = part(batch, attention_mask=mask)
        assert torch.allclose(out[0, 3:], part(short)[0], atol=1e-5)
        assert torch.allclose(out[1], part(long)[0], atol=1e-5)
        # Without the mask the padded row is wrong, so the mask matters.
        unmasked = part(batch)
        assert not torch.allclose(unmasked[0, 3:], part(short)[0], atol=1e-4)
        # An all-ones mask changes nothing.
        assert torch.allclose(part(long, attention_mask=torch.ones(1, 7)), part(long), atol=1e-6)


def test_attn_part_rejects_cached_decode_step():
    config, donor = tiny_llama()
    part = build_attn_heads(config, 0, torch.arange(4), state_of(donor))
    one_token = torch.randn(1, 1, 64)
    # Without cache_len the part expects the whole sequence.
    with pytest.raises(ValueError, match="needs cache_len"):
        part(one_token, attention_mask=torch.ones(1, 5, dtype=torch.long))
