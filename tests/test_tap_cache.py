"""Tests for the donor memory cache (wsp.tap_cache), CachedDonor, and the extraction helpers."""

import sys
import tempfile
import types

import pytest
import torch

transformers = pytest.importorskip("transformers")

from tests.test_tap import PROMPTS, donor_tokenizer, host_tokenizer, make_branch, tiny_donor, tiny_host
from wsp.donor_extract import (choose_shards, extract_prompts, filter_weight_map, find_decoder, layer_number,
                               layer_state, length_batches, patch_remote_mask_calls, run_read_layers, token_ends,
                               truncate_text_config)
from wsp.tap import (CachedDonor, attach_taps, donor_flops_note, prepare, run_donor, run_donor_layers, tap_cost,
                     tap_parameters, tap_train)
from wsp.tap_cache import CacheWriter, prompt_hash, read_index, write_cache

PAIRS = [("the cat sat on a", " mat"), ("a dog ran to the", " big red hat"), ("what is the", " cat"),
         ("a big red", " hat")]


def pair_prompts():
    prompts = []
    for prompt, completion in PAIRS:
        prompts.append(prompt)
    return prompts


def cache_from_live(path, prompts, layers, dtype=torch.float32, shard_size=512):
    """Run the tiny donor on prompts and write its memories to a cache at path."""
    memories, ends, valid = run_donor_layers(tiny_donor(), donor_tokenizer(), prompts, layers)
    return write_cache(path, prompts, memories, ends, valid, {"donor_id": "tiny-qwen", "layers_total": 4},
                       shard_size=shard_size, dtype=dtype)


def same_memory(cached, live):
    """Check that two (memory, ends, valid) triples agree on every valid token."""
    memory_a, ends_a, valid_a = cached
    memory_b, ends_b, valid_b = live
    width = min(valid_a.shape[1], valid_b.shape[1])
    assert int(valid_a.sum()) == int(valid_b.sum())
    assert not valid_a[:, width:].any() and not valid_b[:, width:].any()
    assert torch.equal(valid_a[:, :width], valid_b[:, :width])
    assert torch.equal(ends_a[:, :width], ends_b[:, :width])
    mask = valid_a[:, :width]
    assert torch.allclose(memory_a[:, :width][mask], memory_b[:, :width][mask], atol=1e-6)


def test_cache_round_trip_matches_run_donor():
    with tempfile.TemporaryDirectory() as path:
        meta = cache_from_live(path, PROMPTS, [1, 2])
        assert meta["layers"] == [1, 2]
        assert meta["hidden_size"] == 24
        assert meta["dtype"] == "float32"
        assert meta["num_prompts"] == 3
        assert meta["donor_id"] == "tiny-qwen"
        cached = CachedDonor(path, 2)
        assert cached.hidden_size == 24
        # Ask in another order than the cache was written in.
        prompts = [PROMPTS[2], PROMPTS[0], PROMPTS[1]]
        got = cached.memory(prompts)
        assert got[0].dtype == torch.float32
        same_memory(got, run_donor(tiny_donor(), donor_tokenizer(), prompts, 2))
        # run_donor and run_donor_layers read a CachedDonor too.
        same_memory(run_donor(cached, None, prompts, 2), got)


def live_and_cached_hosts(path, prompts, gate=0.5):
    cache_from_live(path, prompts, [2])
    live = tiny_host()
    cached = tiny_host()
    attach_taps(live, [(1, make_branch(gate=gate), 0)])
    attach_taps(cached, [(1, make_branch(gate=gate), 0)])
    return live, cached, {"d": (tiny_donor(), donor_tokenizer())}, {"d": CachedDonor(path, 2)}


def test_prepare_with_cached_donor_gives_live_logits():
    with tempfile.TemporaryDirectory() as path:
        live, cached, live_pool, cached_pool = live_and_cached_hosts(path, PROMPTS)
        tok = host_tokenizer()
        tok.padding_side = "left"
        live_inputs = prepare(live, tok, live_pool, PROMPTS)
        cached_inputs = prepare(cached, tok, cached_pool, PROMPTS)
        assert torch.equal(live_inputs["input_ids"], cached_inputs["input_ids"])
        with torch.no_grad():
            live_logits = live(**live_inputs).logits
            cached_logits = cached(**cached_inputs).logits
        assert torch.allclose(live_logits, cached_logits, atol=1e-5)
        # The tap changes the logits, so the check above is not trivial.
        plain = tiny_host()
        with torch.no_grad():
            plain_logits = plain(**live_inputs).logits
        assert not torch.allclose(plain_logits, live_logits, atol=1e-3)
        stats = cached.wsp_tap_stats
        assert stats["donor_runs"] == 1
        assert stats["donors"]["d"]["cached"] is True
        assert stats["donors"]["d"]["layers_total"] == 4


def test_tap_train_with_cached_donor_matches_live():
    with tempfile.TemporaryDirectory() as path:
        live, cached, live_pool, cached_pool = live_and_cached_hosts(path, pair_prompts(), gate=0.0)
        tok = host_tokenizer()
        live_losses = tap_train(live, tok, live_pool, PAIRS, tap_parameters(live), steps=12, lr=1e-2, batch_size=3,
                                seed=0, log_every=0)
        cached_losses = tap_train(cached, tok, cached_pool, PAIRS, tap_parameters(cached), steps=12, lr=1e-2,
                                  batch_size=3, seed=0, log_every=0)
        for a, b in zip(live_losses, cached_losses):
            assert abs(a - b) < 1e-4
        for a, b in zip(tap_parameters(live), tap_parameters(cached)):
            assert torch.allclose(a, b, atol=1e-4)
        assert float(cached.wsp_taps[0].gate.detach()) != 0.0
        # The derange control also trains from the cache.
        tap_train(cached, tok, cached_pool, PAIRS, tap_parameters(cached), steps=2, lr=1e-2, batch_size=3, seed=0,
                  derange=True, log_every=0)


def test_cached_derange_rolls_memory_by_one():
    with tempfile.TemporaryDirectory() as path:
        live, cached, live_pool, cached_pool = live_and_cached_hosts(path, PROMPTS)
        tok = host_tokenizer()
        tok.padding_side = "left"
        prepare(cached, tok, cached_pool, PROMPTS)
        plain_memory, plain_valid, plain_vis = cached.wsp_tap_state["memories"][0]
        prepare(cached, tok, cached_pool, PROMPTS, derange=True)
        memory, valid, vis = cached.wsp_tap_state["memories"][0]
        assert torch.equal(memory, torch.roll(plain_memory, 1, 0))
        assert torch.equal(valid, torch.roll(plain_valid, 1, 0))
        assert not torch.equal(memory, plain_memory)
        # Live and cached derange agree.
        live_inputs = prepare(live, tok, live_pool, PROMPTS, derange=True)
        with torch.no_grad():
            live_logits = live(**live_inputs).logits
            cached_logits = cached(**live_inputs).logits
        assert torch.allclose(live_logits, cached_logits, atol=1e-5)
        with pytest.raises(ValueError):
            prepare(cached, tok, cached_pool, PROMPTS[:1], derange=True)


def test_missing_prompt_raises_key_error_naming_it():
    with tempfile.TemporaryDirectory() as path:
        cache_from_live(path, PROMPTS, [2])
        cached = CachedDonor(path, 2)
        with pytest.raises(KeyError) as error:
            cached.memory([PROMPTS[0], "a red dog sat"])
        assert "a red dog sat" in str(error.value)
        host = tiny_host()
        attach_taps(host, [(1, make_branch(), 0)])
        tok = host_tokenizer()
        with pytest.raises(KeyError):
            prepare(host, tok, {"d": cached}, ["the hat is red"])


def test_cached_donor_checks_layers():
    with tempfile.TemporaryDirectory() as path:
        cache_from_live(path, PROMPTS, [1, 2])
        with pytest.raises(ValueError):
            CachedDonor(path, 3)
        host = tiny_host()
        attach_taps(host, [(1, make_branch(donor_layer=2), 0)])
        tok = host_tokenizer()
        with pytest.raises(ValueError):
            prepare(host, tok, {"d": CachedDonor(path, 1)}, PROMPTS)
        # With no fixed layer, any cached layer works, and two taps share one lookup.
        both = tiny_host()
        attach_taps(both, [(0, make_branch(donor_layer=1), 0), (2, make_branch(donor_layer=2, seed=6), 1)])
        prepare(both, tok, {"d": CachedDonor(path)}, PROMPTS)
        assert both.wsp_tap_stats["donor_runs"] == 1
        notes = donor_flops_note(both, {"d": CachedDonor(path)})
        assert notes[0]["layers_ran"] == 3
        assert notes[0]["flops_per_token"] is None
        cost = tap_cost(both, {"d": CachedDonor(path)})
        assert cost["donors"][0]["prefill_flops_per_prompt"] is None


def test_float16_cache_is_close_to_live():
    with tempfile.TemporaryDirectory() as path:
        meta = cache_from_live(path, PROMPTS, [2], dtype=torch.float16)
        assert meta["dtype"] == "float16"
        memory, ends, valid = CachedDonor(path, 2).memory(PROMPTS)
        assert memory.dtype == torch.float32
        live = run_donor(tiny_donor(), donor_tokenizer(), PROMPTS, 2)
        width = memory.shape[1]
        assert torch.equal(valid, live[2][:, :width])
        assert torch.allclose(memory[valid], live[0][:, :width][valid], atol=1e-2, rtol=1e-2)


def test_shards_load_lazily_with_a_small_lru():
    with tempfile.TemporaryDirectory() as path:
        meta = cache_from_live(path, PROMPTS, [1, 2], shard_size=2)
        assert meta["shards"] == ["shard_00000", "shard_00001"]
        index = read_index(path)
        assert index[prompt_hash(PROMPTS[2])] == [1, 0]
        cached = CachedDonor(path, 2, max_shards=1)
        assert cached.shard_reads == 0
        got = cached.memory(PROMPTS)
        assert cached.shard_reads == 2
        assert len(cached.layer_cache) == 1
        same_memory(got, run_donor(tiny_donor(), donor_tokenizer(), PROMPTS, 2))
        # Only the asked layer is read from a shard.
        assert list(cached.layer_cache.keys())[0][1] == 2
        everything = CachedDonor(path, 2, max_shards=None)
        everything.memory(PROMPTS)
        everything.memory(PROMPTS)
        assert everything.shard_reads == 2


def test_writer_resumes_and_skips_written_prompts():
    memories, ends, valid = run_donor_layers(tiny_donor(), donor_tokenizer(), PROMPTS, [2])
    with tempfile.TemporaryDirectory() as path:
        first = CacheWriter(path, [2], 24, meta={"donor_id": "tiny"}, shard_size=2, dtype=torch.float32)
        for row in range(2):
            n = int(valid[row].sum())
            first.add(PROMPTS[row], {2: memories[2][row, :n]}, ends[row, :n], valid[row, :n])
        # Two prompts fill one shard; the writer stops here without close.
        assert first.shards == [0]
        second = CacheWriter(path, [2], 24, meta={"donor_id": "tiny"}, shard_size=2, dtype=torch.float32)
        assert second.has(PROMPTS[0]) and not second.has(PROMPTS[2])
        added = []
        for row in range(3):
            n = int(valid[row].sum())
            added.append(second.add(PROMPTS[row], {2: memories[2][row, :n]}, ends[row, :n], valid[row, :n]))
        assert added == [False, False, True]
        meta = second.close()
        assert meta["num_prompts"] == 3
        assert meta["shards"] == ["shard_00000", "shard_00001"]
        same_memory(CachedDonor(path, 2).memory(PROMPTS), (memories[2], ends, valid))
        with pytest.raises(ValueError):
            CacheWriter(path, [1], 24)
        with pytest.raises(ValueError):
            second.add("new prompt", {2: torch.zeros(3, 7)}, torch.zeros(3), torch.ones(3))


def test_extract_prompts_writes_what_run_donor_gives():
    donor = tiny_donor()
    tok = donor_tokenizer()
    prompts = PROMPTS + ["a cat", "a dog", "the cat sat on a mat"]
    with tempfile.TemporaryDirectory() as path:
        writer = CacheWriter(path, [1, 2], 24, meta={"donor_id": "tiny"}, shard_size=4, dtype=torch.float32)
        stats = extract_prompts(donor, tok, prompts, [1, 2], writer, max_batch=2, log=lambda text: None)
        meta = writer.close()
        assert meta["num_prompts"] == 5
        assert stats["prompts_run"] == 5
        cached = CachedDonor(path)
        memories, ends, valid = cached.lookup(PROMPTS, [1, 2])
        live = run_donor_layers(donor, tok, PROMPTS, [1, 2])
        for layer in [1, 2]:
            width = memories[layer].shape[1]
            mask = valid
            assert torch.equal(valid, live[2][:, :width])
            assert torch.equal(ends, live[1][:, :width])
            assert torch.allclose(memories[layer][mask], live[0][layer][:, :width][mask], atol=1e-5)
        # A second run finds every prompt in the cache and runs nothing.
        again = CacheWriter(path, [1, 2], 24, meta={"donor_id": "tiny"}, dtype=torch.float32)
        stats = extract_prompts(donor, tok, prompts, [1, 2], again, log=lambda text: None)
        assert stats["prompts_run"] == 0


def test_run_read_layers_stops_after_the_deepest_layer():
    donor = tiny_donor()
    calls = [0, 0, 0, 0]
    handles = []
    for index, layer in enumerate(donor.model.layers):

        def count(module, inputs, output, index=index):
            calls[index] += 1

        handles.append(layer.register_forward_hook(count))
    states = run_read_layers(donor.model, donor.model.layers, torch.tensor([[3, 4, 5]]), [0, 1])
    for handle in handles:
        handle.remove()
    assert calls == [1, 1, 0, 0]
    assert sorted(states) == [0, 1]
    assert states[1].shape == (1, 3, 24)


def test_length_batches_group_one_length():
    batches = length_batches([5, 3, 5, 3, 5, 9], max_batch=2, max_tokens=100)
    assert batches == [[1, 3], [0, 2], [4], [5]]
    assert length_batches([50, 50, 50], max_batch=8, max_tokens=100) == [[0, 1], [2]]
    assert length_batches([500], max_batch=8, max_tokens=100) == [[0]]


def test_choose_shards_keeps_needed_files_only():
    weight_map = {
        "model.embed_tokens.weight": "a.safetensors",
        "model.layers.0.mlp.weight": "a.safetensors",
        "model.layers.1.mlp.weight": "b.safetensors",
        "model.layers.2.mlp.weight": "b.safetensors",
        "model.layers.3.mlp.weight": "c.safetensors",
        "model.norm.weight": "d.safetensors",
        "lm_head.weight": "d.safetensors",
        "mtp.0.mlp.weight": "e.safetensors",
        "model.hc_head.hc_fn": "f.safetensors",
        "model.visual.layers.5.weight": "g.safetensors",
    }
    assert layer_number("model.language_model.layers.12.mlp.weight") == 12
    assert layer_number("language_model.model.layers.4.input_layernorm.weight") == 4
    assert layer_number("model.visual.layers.5.weight") is None
    assert layer_number("model.embed_tokens.weight") is None
    files = choose_shards(weight_map, 1)
    assert files == ["a.safetensors", "b.safetensors", "f.safetensors", "g.safetensors"]
    kept = filter_weight_map(weight_map, files, 1)
    assert "model.layers.2.mlp.weight" not in kept
    assert "model.layers.1.mlp.weight" in kept
    assert "lm_head.weight" not in kept
    assert kept["model.hc_head.hc_fn"] == "f.safetensors"


def test_truncate_text_config_cuts_per_layer_lists():
    config = transformers.Qwen2Config(num_hidden_layers=6)
    config.layer_types = ["full_attention"] * 6
    assert truncate_text_config(config, 4) == 6
    assert config.num_hidden_layers == 4
    assert config.layer_types == ["full_attention"] * 4
    nested = types.SimpleNamespace(text_config={"num_hidden_layers": 5, "mlp_layer_types": ["dense"] * 5,
                                                "full_attn_layers": [4, 8], "architectures": ["X"]})
    assert truncate_text_config(nested, 2) == 5
    assert nested.text_config["num_hidden_layers"] == 2
    assert nested.text_config["mlp_layer_types"] == ["dense", "dense"]
    assert nested.text_config["full_attn_layers"] == [4, 8]
    with pytest.raises(ValueError):
        truncate_text_config(nested, 9)


class SlowTokenizer:
    """A word tokenizer with no offsets, like a tiktoken tokenizer. It adds a BOS token."""

    is_fast = False
    all_special_ids = [0]

    def __init__(self):
        self.words = ["<bos>", "the", " cat", " sat", "é", "!"]

    def __call__(self, text):
        ids = [0]
        rest = text
        while rest:
            for index in range(len(self.words) - 1, 0, -1):
                if rest.startswith(self.words[index]):
                    ids.append(index)
                    rest = rest[len(self.words[index]):]
                    break
            else:
                raise ValueError(rest)
        return {"input_ids": ids}

    def decode(self, ids, skip_special_tokens=False, clean_up_tokenization_spaces=False):
        text = ""
        for token in ids:
            text += self.words[token]
        return text


def test_token_ends_without_offsets_uses_prefix_decodes():
    ids, ends, valid = token_ends(SlowTokenizer(), "the caté!")
    assert ids == [0, 1, 2, 4, 5]
    assert ends == [-1, 3, 7, 8, 9]
    assert valid == [False, True, True, True, True]
    # A fast tokenizer gives its own offsets, as run_donor uses.
    ids, ends, valid = token_ends(donor_tokenizer(), "a dog")
    assert ends == [1, 3, 4, 5]
    assert valid == [True, True, True, True]


def test_layer_state_collapses_streams_and_attention_residuals():
    plain = torch.randn(2, 3, 4)
    decoder = torch.nn.Module()
    assert layer_state(decoder, (plain, None)) is plain
    streams = torch.randn(2, 3, 5, 4)
    assert torch.allclose(layer_state(decoder, streams), streams.mean(dim=2))

    class Head(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0]))

        def forward(self, x):
            return (self.weight[None, None, :, None] * x).sum(dim=2)

    decoder.hc_head = Head()
    assert torch.allclose(layer_state(decoder, streams), streams[:, :, 0])

    class AttnRes(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.use_attn_residuals = True
            self.output_attn_res_proj = torch.nn.Linear(4, 1, bias=False)

        def _apply_output_attn_res(self, hidden_states, block_residual):
            return hidden_states + block_residual.sum(dim=1).view(hidden_states.shape)

    prefix_sum = torch.randn(2, 3, 4)
    block_residual = torch.randn(6, 2, 4)
    got = layer_state(AttnRes(), (prefix_sum, block_residual))
    assert torch.allclose(got, prefix_sum + block_residual.sum(dim=1).view(2, 3, 4))


def test_patch_remote_mask_calls_maps_old_arguments():
    calls = []
    remote = types.ModuleType("transformers_modules.fake_kimi")

    def old_style(**kwargs):
        calls.append(kwargs)

    remote.create_causal_mask = old_style
    sys.modules[remote.__name__] = remote
    try:
        Layer = type("Layer", (torch.nn.Module,), {"__module__": remote.__name__})
        model = torch.nn.Sequential(Layer())
        assert patch_remote_mask_calls(model) == [remote.__name__]
        assert patch_remote_mask_calls(model) == []
        from transformers import masking_utils

        seen = {}
        real = masking_utils.create_causal_mask

        def fake(*args, **kwargs):
            seen.update(kwargs)

        masking_utils.create_causal_mask = fake
        try:
            remote.create_causal_mask(config=None, input_embeds="x", attention_mask=None, cache_position=3,
                                      past_key_values=None)
        finally:
            masking_utils.create_causal_mask = real
        assert seen == {"config": None, "inputs_embeds": "x", "attention_mask": None, "past_key_values": None}
    finally:
        del sys.modules[remote.__name__]


def tiny_frontier(name):
    """Build a tiny random model of a frontier donor type, or skip when this transformers lacks it."""
    if name == "deepseek_v4":
        if not hasattr(transformers, "DeepseekV4Config"):
            pytest.skip("needs transformers 5 with deepseek_v4")
        config = transformers.DeepseekV4Config(
            vocab_size=64, hidden_size=32, num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=1,
            head_dim=16, q_lora_rank=16, moe_intermediate_size=16, n_routed_experts=4, num_experts_per_tok=2,
            hc_mult=2, max_position_embeddings=128)
        return transformers.DeepseekV4ForCausalLM(config).eval()
    if name == "glm_moe_dsa":
        if not hasattr(transformers, "GlmMoeDsaConfig"):
            pytest.skip("needs transformers 5 with glm_moe_dsa")
        config = transformers.GlmMoeDsaConfig(
            vocab_size=64, hidden_size=32, intermediate_size=48, moe_intermediate_size=16, num_hidden_layers=4,
            num_attention_heads=4, num_key_value_heads=4, n_routed_experts=4, num_experts_per_tok=2,
            q_lora_rank=16, kv_lora_rank=16, qk_rope_head_dim=8, qk_nope_head_dim=8, v_head_dim=8,
            index_n_heads=2, index_head_dim=16, index_topk=8, first_k_dense_replace=1, max_position_embeddings=128)
        return transformers.GlmMoeDsaForCausalLM(config).eval()
    if name == "glm5_next":
        if not hasattr(transformers, "Glm5NextConfig"):
            pytest.skip("needs transformers 5 with glm5_next")
        text = dict(vocab_size=64, pad_token_id=0, hidden_size=32, intermediate_size=48, moe_intermediate_size=16,
                    num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=4, n_routed_experts=4,
                    num_experts_per_tok=2, q_lora_rank=16, kv_lora_rank=16, qk_rope_head_dim=0, qk_nope_head_dim=16,
                    v_head_dim=8, index_n_heads=2, index_head_dim=16, index_topk=16, index_kpool=4,
                    linear_head_dim=8, linear_num_heads=4, hc_mult=2, max_position_embeddings=128)
        vision = dict(depth=1, hidden_size=16, num_heads=2, out_hidden_size=32, intermediate_size=16)
        config = transformers.Glm5NextConfig(text_config=text, vision_config=vision)
        return transformers.Glm5NextForConditionalGeneration(config).eval()
    raise ValueError(name)


@pytest.mark.parametrize("name", ["deepseek_v4", "glm_moe_dsa", "glm5_next"])
def test_frontier_type_truncation_keeps_early_states(name):
    """On a tiny random model: a truncated copy gives the same read-layer states, and the
    collapsed last layer, after the final norm, equals the model's own last hidden state."""
    torch.manual_seed(0)
    model = tiny_frontier(name)
    ids = torch.tensor([[3, 9, 12, 5, 7, 30]])
    decoder, layers = find_decoder(model, 3)
    states = run_read_layers(decoder, layers, ids, [1, 3])
    with torch.no_grad():
        full = decoder(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False).last_hidden_state
    assert torch.allclose(decoder.norm(states[3]), full, atol=1e-4)

    config = model.config.__class__.from_dict(model.config.to_dict())
    truncate_text_config(config, 2)
    short = model.__class__(config).eval()
    missing, unexpected = short.load_state_dict(model.state_dict(), strict=False)
    assert len(missing) == 0
    for key in unexpected:
        assert layer_number(key) is not None and layer_number(key) >= 2
    short_decoder, short_layers = find_decoder(short, 1)
    short_states = run_read_layers(short_decoder, short_layers, ids, [1])
    assert len(short_layers) == 2
    assert torch.allclose(short_states[1], states[1], atol=1e-5)
