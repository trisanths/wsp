"""Qwen3.5 and Gemma 4 donors, which need Transformers 5.

Qwen3.5 mixes gated delta net layers (linear_attn) with gated attention
layers. Its q_proj holds a query and a gate for each head. Gemma 4 has a
rotary per layer type, per-layer head sizes, and optional per-layer inputs
and shared KV states. Tiny random configs, CPU only.
"""

import pytest
import torch

transformers = pytest.importorskip("transformers")

from wsp.graft import Graft, attach_grafts
from wsp.parts import attn_tensor_names, block_tensor_names, build_attn_heads, build_block_part
from tests.test_block import capture_block
from tests.test_part_cache import SETTINGS, padded_prompts, tiny_host
from tests.test_review_parts import layer_attention_io


def tiny_qwen3_5():
    torch.manual_seed(0)
    config = transformers.Qwen3_5TextConfig(
        hidden_size=32, intermediate_size=64, num_hidden_layers=3, num_attention_heads=4,
        num_key_value_heads=2, head_dim=8, vocab_size=300, max_position_embeddings=64,
        layer_types=["linear_attention", "full_attention", "linear_attention"],
        linear_num_value_heads=4, linear_num_key_heads=2, linear_key_head_dim=8, linear_value_head_dim=8,
    )
    config._attn_implementation = "eager"
    return config, transformers.Qwen3_5ForCausalLM(config).eval()


def gemma4_config(**changes):
    settings = dict(
        hidden_size=32, intermediate_size=64, num_hidden_layers=4, num_attention_heads=4,
        num_key_value_heads=2, head_dim=8, vocab_size=300, vocab_size_per_layer_input=300,
        hidden_size_per_layer_input=0, max_position_embeddings=64, sliding_window=3,
        layer_types=["sliding_attention", "full_attention", "sliding_attention", "full_attention"],
        per_layer_config={"01": {"head_dim": 16}, "03": {"head_dim": 16}},
    )
    settings.update(changes)
    config = transformers.Gemma4TextConfig(**settings)
    config._attn_implementation = "eager"
    return config


def test_qwen3_5_attn_heads_keep_each_head_gate():
    config, donor = tiny_qwen3_5()
    state = donor.state_dict()
    ids = torch.randint(3, 300, (2, 7))
    x, attn = layer_attention_io(donor, donor.model.layers[1], ids)
    with torch.no_grad():
        whole = build_attn_heads(config, 1, torch.arange(4), state)
        assert torch.allclose(whole(x), attn, atol=1e-5)
        # Each head brings its own gate rows, so two disjoint head sets
        # add up to the whole attention.
        first = build_attn_heads(config, 1, [0, 3], state)
        second = build_attn_heads(config, 1, [2, 1], state)
        assert first.attn.q_proj.weight.shape[0] == 2 * 2 * 8
        assert torch.allclose(first(x) + second(x), attn, atol=1e-5)


def test_qwen3_5_attn_heads_on_a_linear_attention_layer_raises():
    config, _ = tiny_qwen3_5()
    with pytest.raises(ValueError, match="no self_attn"):
        attn_tensor_names(config, 0)


def test_qwen3_5_block_over_linear_and_full_layers():
    config, donor = tiny_qwen3_5()
    part = build_block_part(config, 0, 2, donor.state_dict())
    assert part.recurrent == [True, False, True]
    ids = torch.randint(3, 300, (2, 8))
    mask = torch.ones(2, 8, dtype=torch.long)
    mask[0, :3] = 0
    position_ids = (mask.cumsum(-1) - 1).clamp(min=0)
    x, y = capture_block(donor, 0, 2, ids, attention_mask=mask, position_ids=position_ids)
    with torch.no_grad():
        out = part(x, attention_mask=mask)
    real = mask.bool()
    assert torch.allclose(out[real], (y - x)[real], atol=1e-5)


def test_qwen3_5_block_cached_generate_equals_uncached():
    # The gated delta net layers keep a conv state and a recurrent state
    # in the part cache.
    config, donor = tiny_qwen3_5()
    host = tiny_host()
    part = build_block_part(config, 0, 2, donor.state_dict()).float()
    torch.manual_seed(5)
    graft = Graft(32, config.hidden_size, part)
    torch.nn.init.normal_(graft.w_out.weight, std=1.0)
    attach_grafts(host, [(1, graft)])
    ids, mask = padded_prompts()
    with torch.no_grad():
        cached = host.generate(input_ids=ids, attention_mask=mask, use_cache=True, **SETTINGS)
        uncached = host.generate(input_ids=ids, attention_mask=mask, use_cache=False, **SETTINGS)
        chunked = host.generate(input_ids=ids, attention_mask=mask, prefill_chunk_size=2, **SETTINGS)
    assert torch.equal(cached, uncached)
    assert torch.equal(chunked, uncached)


def test_gemma4_block_with_per_layer_head_sizes():
    torch.manual_seed(0)
    config = gemma4_config()
    donor = transformers.Gemma4ForCausalLM(config).eval()
    part = build_block_part(config, 0, 3, donor.state_dict())
    ids = torch.randint(3, 300, (2, 7))
    x, y = capture_block(donor, 0, 3, ids)
    with torch.no_grad():
        assert torch.allclose(part(x), y - x, atol=1e-5)


def test_gemma4_features_a_part_cannot_run_raise_early():
    with pytest.raises(NotImplementedError, match="per-layer inputs"):
        block_tensor_names(gemma4_config(hidden_size_per_layer_input=8), 0, 1)
    # Layers 4 and 5 reuse the KV states that layers 2 and 3 store.
    shared = gemma4_config(
        num_hidden_layers=6, num_kv_shared_layers=2, per_layer_config={},
        layer_types=["sliding_attention", "full_attention"] * 3,
    )
    assert shared.kv_sharing_modes == [None, None, "store", "store", "reuse", "reuse"]
    for first, last in [(2, 2), (3, 5), (5, 5)]:
        with pytest.raises(NotImplementedError, match="shares KV states"):
            block_tensor_names(shared, first, last)
    # The layers before them still work.
    assert len(block_tensor_names(shared, 0, 1)) > 0
    with pytest.raises(NotImplementedError, match="shares KV states"):
        attn_tensor_names(gemma4_config(), 0)
