"""Atlas checks on tiny random-weight models of families that need newer transformers.

A family is skipped when its config class is missing from the installed
transformers. LFM2, LFM2-MoE, Qwen3.5 and Gemma 4 are the families the zoo
marks as needing transformers 5.x. Under 4.57.1 the ones that exist there
(LFM2) still run, and the rest skip.
"""

import copy
import os
import sys

import pytest
import torch
import transformers

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from test_atlas import GENERAL_WORDS, SKILL_WORDS, texts, tokenizer  # noqa: E402

from wsp import atlas  # noqa: E402

TWO_LAYER_TYPES = ["linear_attention", "linear_attention", "linear_attention", "full_attention"]
CONV_TYPES = ["conv", "full_attention", "conv", "full_attention"]
BASE = dict(vocab_size=40, hidden_size=32, num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=64)


def lfm2(config_class):
    return config_class(intermediate_size=64, num_hidden_layers=4, layer_types=CONV_TYPES, conv_L_cache=3, **BASE)


def lfm2_moe(config_class):
    return config_class(intermediate_size=64, moe_intermediate_size=24, num_hidden_layers=4, num_dense_layers=1,
                        num_experts=6, num_experts_per_tok=2, layer_types=CONV_TYPES, conv_L_cache=3, **BASE)


def qwen3_5(config_class):
    return config_class(intermediate_size=64, num_hidden_layers=4, head_dim=8, layer_types=TWO_LAYER_TYPES,
                        linear_num_key_heads=2, linear_num_value_heads=4, linear_key_head_dim=8,
                        linear_value_head_dim=8, **BASE)


def qwen3_5_moe(config_class):
    return config_class(moe_intermediate_size=24, shared_expert_intermediate_size=24, num_hidden_layers=4,
                        head_dim=8, num_experts=6, num_experts_per_tok=2, layer_types=TWO_LAYER_TYPES,
                        linear_num_key_heads=2, linear_num_value_heads=4, linear_key_head_dim=8,
                        linear_value_head_dim=8, **BASE)


def gemma4(config_class):
    # All layers use full attention and there is no per-layer input, so that
    # deleting a layer leaves the others unchanged.
    return config_class(intermediate_size=64, num_hidden_layers=4, head_dim=8, global_head_dim=8,
                        hidden_size_per_layer_input=0, num_kv_shared_layers=0,
                        layer_types=["full_attention"] * 4, num_global_key_value_heads=2, **BASE)


def gemma4_moe(config_class):
    return config_class(intermediate_size=64, moe_intermediate_size=24, num_hidden_layers=4, head_dim=8,
                        global_head_dim=8, hidden_size_per_layer_input=0, num_kv_shared_layers=0,
                        layer_types=["full_attention"] * 4, num_global_key_value_heads=2, enable_moe_block=True,
                        num_experts=6, top_k_experts=2, **BASE)


# name: (config class name, config builder, is MoE)
FAMILIES = {
    "lfm2": ("Lfm2Config", lfm2, False),
    "lfm2_moe": ("Lfm2MoeConfig", lfm2_moe, True),
    "qwen3_5": ("Qwen3_5TextConfig", qwen3_5, False),
    "qwen3_5_moe": ("Qwen3_5MoeTextConfig", qwen3_5_moe, True),
    "gemma4": ("Gemma4TextConfig", gemma4, False),
    "gemma4_moe": ("Gemma4TextConfig", gemma4_moe, True),
}


def build(name):
    class_name, make, _ = FAMILIES[name]
    config_class = getattr(transformers, class_name, None)
    if config_class is None:
        pytest.skip(f"{class_name} is not in transformers {transformers.__version__}")
    torch.manual_seed(0)
    return transformers.AutoModelForCausalLM.from_config(make(config_class)).eval()


def drop_layer(model, layer):
    """A copy of the model with one decoder layer deleted, config updated to match."""
    removed = copy.deepcopy(model)
    _, layers = atlas.find_decoder_layers(removed, 0)
    count = len(layers)
    del layers[layer]
    configs = {}
    for config in (removed.config, removed.config.get_text_config()):
        configs[id(config)] = config
    for config in configs.values():
        config.num_hidden_layers = count - 1
        if getattr(config, "layer_types", None):
            config.layer_types = config.layer_types[:layer] + config.layer_types[layer + 1:]
    return removed


def sample_texts():
    return texts(SKILL_WORDS + GENERAL_WORDS, 4, 12, seed=1)


@pytest.mark.parametrize("name", list(FAMILIES))
def test_skip_equals_layer_removed(name):
    model = build(name)
    tok = tokenizer()
    sample = sample_texts()
    base = atlas.text_loss(model, tok, sample)
    _, layers = atlas.find_decoder_layers(model, 0)
    changed = 0
    for layer in range(len(layers)):
        skipped = atlas.text_loss(model, tok, sample, skip_layer=layer)
        direct = atlas.text_loss(drop_layer(model, layer), tok, sample)
        assert abs(skipped - direct) < 1e-5
        if abs(skipped - base) > 1e-6:
            changed += 1
    assert changed > 0


@pytest.mark.parametrize("name", list(FAMILIES))
def test_layer_skill_map_runs(name):
    model = build(name)
    tok = tokenizer()
    skill = texts(SKILL_WORDS, 4, 10, seed=2)
    general = texts(GENERAL_WORDS, 4, 10, seed=3)
    result = atlas.layer_skill_map(model, tok, skill, general)
    assert len(result["scores"]) == 4
    assert len(result["influence"]) == 4
    assert result["blocks"]


@pytest.mark.parametrize("name", [key for key in FAMILIES if FAMILIES[key][2]])
def test_expert_counts_sum_to_top_k(name):
    model = build(name)
    tok = tokenizer()
    sample = sample_texts()
    routers = atlas.find_routers(model)
    assert len(routers) > 0
    for _, _, _, top_k, num_experts in routers:
        assert top_k == 2
        assert num_experts == 6
    counts = atlas.routing_counts(model, tok, sample)
    assert sorted(counts) == [router[0] for router in routers]
    expected_tokens = sum(len(ids) for ids in atlas.encode_texts(tok, sample))
    for layer, (per_expert, tokens) in counts.items():
        assert tokens == expected_tokens
        assert abs(float(per_expert.sum()) - tokens * 2) < 1e-6
        assert float(per_expert.max()) <= tokens

    skill = texts(SKILL_WORDS, 4, 10, seed=2)
    general = texts(GENERAL_WORDS, 4, 10, seed=3)
    result = atlas.expert_skill_map(model, tok, skill, general)
    assert result["top_k"] == 2
    assert result["num_experts"] == 6
    for layer_map in result["layers"].values():
        assert abs(sum(layer_map["skill_freq"]) - 2) < 1e-3
        assert abs(sum(layer_map["general_freq"]) - 2) < 1e-3


def test_gemma4_skip_with_shared_kv_and_per_layer_input():
    config_class = getattr(transformers, "Gemma4TextConfig", None)
    if config_class is None:
        pytest.skip("Gemma4TextConfig is not in this transformers")
    torch.manual_seed(0)
    config = config_class(intermediate_size=64, num_hidden_layers=6, head_dim=8, global_head_dim=8,
                          hidden_size_per_layer_input=8, vocab_size_per_layer_input=40, num_kv_shared_layers=2,
                          layer_types=["sliding_attention", "sliding_attention", "full_attention", "sliding_attention",
                                       "sliding_attention", "full_attention"], sliding_window=8,
                          num_global_key_value_heads=2, **BASE)
    model = transformers.AutoModelForCausalLM.from_config(config).eval()
    tok = tokenizer()
    sample = sample_texts()
    base = atlas.text_loss(model, tok, sample)
    for layer in range(6):
        value = atlas.text_loss(model, tok, sample, skip_layer=layer)
        assert value == value
    assert atlas.text_loss(model, tok, sample, skip_layer=0) != base
