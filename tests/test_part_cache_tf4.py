"""attn_heads and block parts keep their own KV cache in step with the host.

A grafted host must give the same output with use_cache=True as with
use_cache=False, also for a left-padded batch and for donors with sliding
windows.
"""

import transformers as _transformers
import pytest as _pytest

pytestmark = _pytest.mark.skipif(
    int(_transformers.__version__.split(".")[0]) >= 5,
    reason="the Transformers 4 version of this test file; the 5.x version is the unsuffixed file",
)

import pytest
import torch

transformers = pytest.importorskip("transformers")

from wsp.graft import Graft, attach_grafts
from wsp.parts import build_attn_heads, build_block_part

NEW_TOKENS = 12
SETTINGS = dict(max_new_tokens=NEW_TOKENS, do_sample=False, pad_token_id=0, eos_token_id=None)


def tiny_donor(kind):
    torch.manual_seed(3)
    if kind == "llama":
        config = transformers.LlamaConfig(
            hidden_size=64, intermediate_size=128, num_hidden_layers=4, num_attention_heads=4,
            num_key_value_heads=2, vocab_size=300, max_position_embeddings=64,
        )
        config._attn_implementation = "eager"
        return config, transformers.LlamaForCausalLM(config).eval()
    if kind == "qwen2_sliding":
        # Layers 1 and up use a sliding window of 3 tokens.
        config = transformers.Qwen2Config(
            hidden_size=48, intermediate_size=96, num_hidden_layers=4, num_attention_heads=4,
            num_key_value_heads=2, vocab_size=300, max_position_embeddings=64,
            use_sliding_window=True, sliding_window=3, max_window_layers=1,
        )
        config._attn_implementation = "eager"
        return config, transformers.Qwen2ForCausalLM(config).eval()
    config = transformers.Gemma3TextConfig(
        hidden_size=48, intermediate_size=96, num_hidden_layers=4, num_attention_heads=4,
        num_key_value_heads=2, head_dim=12, vocab_size=300, max_position_embeddings=64,
        sliding_window=3, rope_local_base_freq=100.0,
        layer_types=["full_attention", "sliding_attention", "full_attention", "sliding_attention"],
    )
    config._attn_implementation = "eager"
    return config, transformers.Gemma3ForCausalLM(config).eval()


def build_part(kind, part_kind):
    """Return (part, d_donor). The parts include sliding layers where the donor has them."""
    config, donor = tiny_donor(kind)
    tensors = {}
    for name, value in donor.state_dict().items():
        tensors[name] = value.detach().clone()
    if part_kind == "attn_heads":
        # Layer 1 is a sliding layer in the qwen2 and gemma3 donors.
        part = build_attn_heads(config, 1, [0, 3], tensors)
    else:
        # Layers 1 and 2 hold a sliding and a full layer in gemma3, and
        # two sliding layers in qwen2.
        part = build_block_part(config, 1, 2, tensors)
    return part.float(), config.hidden_size


def tiny_host():
    torch.manual_seed(0)
    config = transformers.LlamaConfig(
        hidden_size=32, intermediate_size=64, num_hidden_layers=3, num_attention_heads=4,
        num_key_value_heads=2, vocab_size=300, max_position_embeddings=64,
    )
    config._attn_implementation = "eager"
    return transformers.LlamaForCausalLM(config).eval()


def grafted_host(kind, part_kind):
    host = tiny_host()
    part, d_donor = build_part(kind, part_kind)
    torch.manual_seed(5)
    graft = Graft(32, d_donor, part)
    torch.nn.init.normal_(graft.w_out.weight, std=1.0)
    attach_grafts(host, [(1, graft)])
    return host, graft


def padded_prompts():
    """Return three left-padded prompts of lengths 7, 4 and 2."""
    torch.manual_seed(11)
    lengths = [7, 4, 2]
    width = 7
    ids = torch.zeros(3, width, dtype=torch.long)
    mask = torch.zeros(3, width, dtype=torch.long)
    for row in range(3):
        n = lengths[row]
        ids[row, width - n:] = torch.randint(3, 300, (n,))
        mask[row, width - n:] = 1
    return ids, mask


CASES = []
for donor_kind in ["llama", "qwen2_sliding", "gemma3"]:
    for kind_of_part in ["attn_heads", "block"]:
        CASES.append((donor_kind, kind_of_part))


@pytest.mark.parametrize("kind,part_kind", CASES)
def test_cached_generate_equals_uncached(kind, part_kind):
    host, _ = grafted_host(kind, part_kind)
    ids, mask = padded_prompts()
    with torch.no_grad():
        cached = host.generate(input_ids=ids, attention_mask=mask, use_cache=True, **SETTINGS)
        uncached = host.generate(input_ids=ids, attention_mask=mask, use_cache=False, **SETTINGS)
    assert cached.shape == (3, 7 + NEW_TOKENS)
    assert torch.equal(cached, uncached)


@pytest.mark.parametrize("kind,part_kind", CASES)
def test_cached_step_logits_match_full_forward(kind, part_kind):
    host, _ = grafted_host(kind, part_kind)
    ids, mask = padded_prompts()
    with torch.no_grad():
        out = host.generate(
            input_ids=ids, attention_mask=mask, use_cache=True,
            output_logits=True, return_dict_in_generate=True, **SETTINGS,
        )
        sequences = out.sequences
        full_mask = torch.cat([mask, torch.ones(3, NEW_TOKENS, dtype=torch.long)], dim=1)
        full = host(input_ids=sequences, attention_mask=full_mask, use_cache=False).logits
        plain = tiny_host()(input_ids=sequences, attention_mask=full_mask, use_cache=False).logits
    # The graft must change the output, or the test checks nothing.
    assert not torch.allclose(full, plain, atol=1e-3)
    assert len(out.logits) == NEW_TOKENS
    width = ids.shape[1]
    for step in range(NEW_TOKENS):
        expected = full[:, width - 1 + step]
        assert torch.allclose(out.logits[step], expected, atol=1e-4), f"step {step}"


@pytest.mark.parametrize("kind,part_kind", CASES)
def test_generate_twice_resets_part_cache(kind, part_kind):
    host, graft = grafted_host(kind, part_kind)
    ids, mask = padded_prompts()
    with torch.no_grad():
        first = host.generate(input_ids=ids, attention_mask=mask, use_cache=True, **SETTINGS)
        # The part cache now holds the prompt and all but the last new token.
        assert graft.part.cache is not None
        second = host.generate(input_ids=ids, attention_mask=mask, use_cache=True, **SETTINGS)
    assert torch.equal(first, second)


@pytest.mark.parametrize("part_kind", ["attn_heads", "block"])
def test_out_of_sync_cache_raises(part_kind):
    host, graft = grafted_host("llama", part_kind)
    ids, mask = padded_prompts()
    with torch.no_grad():
        first = host(input_ids=ids, attention_mask=mask, use_cache=True)
        # A second prefill of another length resets the part cache.
        host(input_ids=ids[:, 2:], attention_mask=mask[:, 2:], use_cache=True)
        step_mask = torch.cat([mask, torch.ones(3, 1, dtype=torch.long)], dim=1)
        with pytest.raises(ValueError, match="out of sync"):
            host(
                input_ids=torch.tensor([[5], [6], [7]]), attention_mask=step_mask,
                past_key_values=first.past_key_values, use_cache=True,
            )


def test_part_without_cache_drops_its_cache():
    part, d_donor = build_part("llama", "attn_heads")
    x = torch.randn(1, 4, d_donor)
    with torch.no_grad():
        part(x, attention_mask=torch.ones(1, 4, dtype=torch.long), cache_len=0)
        assert part.cache is not None
        part(x)
    assert part.cache is None
    with pytest.raises(ValueError, match="out of sync"):
        part(x[:, :1], attention_mask=torch.ones(1, 5, dtype=torch.long), cache_len=4)


def test_manual_cached_loop_matches_full_forward():
    # A caller that passes use_cache=True on the prefill and then the
    # returned cache gets the same logits as one full forward.
    host, _ = grafted_host("gemma3", "block")
    ids, mask = padded_prompts()
    next_ids = torch.tensor([[8], [9], [10]])
    step_mask = torch.cat([mask, torch.ones(3, 1, dtype=torch.long)], dim=1)
    with torch.no_grad():
        first = host(input_ids=ids, attention_mask=mask, use_cache=True)
        step = host(
            input_ids=next_ids, attention_mask=step_mask,
            past_key_values=first.past_key_values, use_cache=True,
        ).logits[:, -1]
        full = host(
            input_ids=torch.cat([ids, next_ids], dim=1), attention_mask=step_mask, use_cache=False,
        ).logits[:, -1]
    assert torch.allclose(step, full, atol=1e-4)


@pytest.mark.parametrize("kind,part_kind", [("llama", "attn_heads"), ("gemma3", "block")])
def test_beam_search_cached_equals_uncached(kind, part_kind):
    # Beam search reorders the host cache between steps. The part caches
    # must get the same reorder.
    host, _ = grafted_host(kind, part_kind)
    ids, mask = padded_prompts()
    settings = dict(SETTINGS)
    settings["num_beams"] = 2
    with torch.no_grad():
        cached = host.generate(input_ids=ids, attention_mask=mask, use_cache=True, **settings)
        uncached = host.generate(input_ids=ids, attention_mask=mask, use_cache=False, **settings)
    assert torch.equal(cached, uncached)


def test_decode_with_another_cache_of_same_length_raises():
    # Two prefills of the same length: the part caches follow the second,
    # so a step on the first host cache must raise, not mix the two.
    host, _ = grafted_host("llama", "attn_heads")
    a = torch.randint(3, 300, (1, 6))
    b = torch.randint(3, 300, (1, 6))
    with torch.no_grad():
        cache_a = host(a, use_cache=True).past_key_values
        host(b, use_cache=True)
        with pytest.raises(ValueError, match="out of sync"):
            host(torch.tensor([[4]]), past_key_values=cache_a, use_cache=True)


def test_static_cache_raises_and_hybrid_default_is_cleared():
    host = tiny_host()
    host.generation_config.cache_implementation = "hybrid"
    part, d_donor = build_part("llama", "attn_heads")
    attach_grafts(host, [(1, Graft(32, d_donor, part))])
    # A static or hybrid cache gives a 4D mask that hides padding.
    assert host.generation_config.cache_implementation is None
    ids, mask = padded_prompts()
    with torch.no_grad():
        with pytest.raises(ValueError, match="2D attention_mask"):
            host.generate(input_ids=ids, attention_mask=mask, cache_implementation="static", **SETTINGS)


@pytest.mark.parametrize("part_kind", ["attn_heads", "block"])
def test_chunked_prefill_equals_uncached(part_kind):
    # generate with prefill_chunk_size calls host.forward directly, so the
    # host pre-hook does not run for the prefill chunks. The decoder hook
    # must set the graft context for them.
    host, _ = grafted_host("gemma3", part_kind)
    ids, mask = padded_prompts()
    with torch.no_grad():
        # A plain call first leaves a context with no mask and no cache.
        host(input_ids=ids)
        chunked = host.generate(input_ids=ids, attention_mask=mask, prefill_chunk_size=2, **SETTINGS)
        uncached = host.generate(input_ids=ids, attention_mask=mask, use_cache=False, **SETTINGS)
    assert torch.equal(chunked, uncached)
