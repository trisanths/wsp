"""Auto-prepare through generate: chunked prefill, calls that follow a generate, beams, and copies."""

import pytest
import torch

transformers = pytest.importorskip("transformers")

from wsp.bake import bare_host_copy
from wsp.tap import attach_taps, clear_tap_context, detach_taps, prepare, set_tap_sources
from tests.test_tap import PROMPTS, donors, host_tokenizer, make_branch, tiny_host


def auto_host():
    host = tiny_host()
    tok = host_tokenizer()
    tok.padding_side = "left"
    attach_taps(host, [(1, make_branch(gate=1.0), 0), (2, make_branch(donor_layer=3, seed=7, gate=-0.8), 1)])
    with torch.no_grad():
        for branch in host.wsp_taps:
            branch.o.weight.mul_(10.0)
    set_tap_sources(host, tok, donors())
    return host, tok


def encode(tok, prompts):
    tok.padding_side = "left"
    encoded = tok(prompts, padding=True, return_tensors="pt")
    return {"input_ids": encoded["input_ids"], "attention_mask": encoded["attention_mask"]}


SETTINGS = {"max_new_tokens": 5, "do_sample": False, "output_scores": True, "return_dict_in_generate": True}


def worst_gap(a, b):
    worst = 0.0
    for x, y in zip(a.scores, b.scores):
        gap = float((x - y).abs().max())
        if gap > worst:
            worst = gap
    return worst


def test_chunked_prefill_equals_one_prefill():
    host, tok = auto_host()
    inputs = encode(tok, PROMPTS)
    with torch.no_grad():
        one = host.generate(**inputs, pad_token_id=tok.pad_token_id, **SETTINGS)
        chunked = host.generate(**inputs, pad_token_id=tok.pad_token_id, prefill_chunk_size=2, **SETTINGS)
        for branch in host.wsp_taps:
            branch.gate.zero_()
        plain_one = host.generate(**inputs, pad_token_id=tok.pad_token_id, **SETTINGS)
        plain_chunked = host.generate(**inputs, pad_token_id=tok.pad_token_id, prefill_chunk_size=2, **SETTINGS)
    # The taps matter here.
    assert worst_gap(one, plain_one) > 0.05
    # Chunking moves the plain host a little. The taps add no more than that.
    floor = worst_gap(plain_one, plain_chunked)
    assert torch.equal(one.sequences, chunked.sequences)
    assert worst_gap(one, chunked) < max(10 * floor, 1e-4)


def test_auto_generate_equals_manual_prepare():
    host, tok = auto_host()
    inputs = encode(tok, PROMPTS)
    with torch.no_grad():
        auto = host.generate(**inputs, pad_token_id=tok.pad_token_id, **SETTINGS)
        manual_inputs = prepare(host, tok, host.wsp_tap_donors, PROMPTS)
        manual = host.generate(**manual_inputs, pad_token_id=tok.pad_token_id, **SETTINGS)
    clear_tap_context(host)
    assert torch.equal(manual_inputs["input_ids"], inputs["input_ids"])
    assert torch.equal(auto.sequences, manual.sequences)
    assert worst_gap(auto, manual) == 0.0


@pytest.mark.parametrize("use_cache", [True, False])
def test_a_call_does_not_depend_on_an_earlier_generate(use_cache):
    host, tok = auto_host()
    full = encode(tok, [PROMPTS[0] + " is big"])
    with torch.no_grad():
        fresh = host(**full, use_cache=use_cache).logits
        host.generate(**encode(tok, [PROMPTS[0]]), pad_token_id=tok.pad_token_id, max_new_tokens=3,
                      use_cache=use_cache, do_sample=False)
        after = host(**full, use_cache=use_cache).logits
    assert torch.equal(fresh, after)


def test_beams_run_the_donors_once_per_prompt():
    host, tok = auto_host()
    with torch.no_grad():
        host.generate(**encode(tok, PROMPTS), pad_token_id=tok.pad_token_id, max_new_tokens=3, num_beams=3,
                      num_return_sequences=2, do_sample=False, use_cache=False)
    # One donor run per generate call, on the 3 prompts, not on 9 expanded rows per step.
    assert host.wsp_tap_stats["donor_runs"] == 1


def test_manual_context_wins_inside_generate():
    host, tok = auto_host()
    with torch.no_grad():
        auto = host.generate(**encode(tok, PROMPTS), pad_token_id=tok.pad_token_id, **SETTINGS)
        inputs = prepare(host, tok, host.wsp_tap_donors, PROMPTS, derange=True)
        cached = host.generate(**inputs, pad_token_id=tok.pad_token_id, **SETTINGS)
        uncached = host.generate(**inputs, pad_token_id=tok.pad_token_id, use_cache=False, **SETTINGS)
    clear_tap_context(host)
    assert worst_gap(auto, cached) > 1e-3
    assert worst_gap(cached, uncached) < 1e-5


def test_detach_and_bare_copy_give_plain_generate():
    host, tok = auto_host()
    inputs = encode(tok, PROMPTS)
    reference = tiny_host()
    with torch.no_grad():
        expected = reference.generate(**inputs, pad_token_id=tok.pad_token_id, **SETTINGS)
        copied = bare_host_copy(host).generate(**inputs, pad_token_id=tok.pad_token_id, **SETTINGS)
        detach_taps(host)
        assert "generate" not in host.__dict__
        detached = host.generate(**inputs, pad_token_id=tok.pad_token_id, **SETTINGS)
    assert torch.equal(copied.sequences, expected.sequences)
    assert worst_gap(copied, expected) < 1e-5
    assert worst_gap(detached, expected) < 1e-5
