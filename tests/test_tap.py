"""Tests for wsp.tap: a frozen host reads a frozen donor's states through a gated branch."""

import tempfile

import pytest
import torch
from tokenizers import Regex, Tokenizer, models, pre_tokenizers, processors

transformers = pytest.importorskip("transformers")

from wsp.tap import (TapBranch, attach_taps, build_visibility, clear_tap_context, donor_flops_note,
                     host_token_ends, load_taps, prepare, run_donor, save_taps, tap_param_count,
                     tap_parameters, tap_train)

WORDS = ["the", "cat", "sat", "on", "a", "mat", "dog", "ran", "to", "big", "red", "hat", "is", "what"]
PROMPTS = ["the cat sat on a mat", "a dog ran", "what is the big red hat"]


def host_tokenizer():
    """Word-level, with a BOS token, like Llama."""
    vocab = {"<unk>": 0, "</s>": 1, "<pad>": 2, "<s>": 3}
    for word in WORDS:
        vocab[word] = len(vocab)
    base = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    base.pre_tokenizer = pre_tokenizers.Whitespace()
    base.post_processor = processors.TemplateProcessing(single="<s> $A", special_tokens=[("<s>", 3)])
    return transformers.PreTrainedTokenizerFast(
        tokenizer_object=base, unk_token="<unk>", eos_token="</s>", pad_token="<pad>", bos_token="<s>",
    )


def donor_tokenizer():
    """Character-level, with no BOS token, like Qwen. Spaces are dropped."""
    vocab = {"<unk>": 0, "<|end|>": 1}
    for letter in "abcdefghijklmnopqrstuvwxyz":
        vocab[letter] = len(vocab)
    base = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    base.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Whitespace(),
        pre_tokenizers.Split(Regex("."), behavior="isolated"),
    ])
    return transformers.PreTrainedTokenizerFast(
        tokenizer_object=base, unk_token="<unk>", eos_token="<|end|>", pad_token="<|end|>",
    )


def tiny_host(seed=0):
    torch.manual_seed(seed)
    config = transformers.LlamaConfig(
        hidden_size=32, intermediate_size=64, num_hidden_layers=3, num_attention_heads=4,
        num_key_value_heads=2, vocab_size=64, max_position_embeddings=128,
    )
    return transformers.LlamaForCausalLM(config).eval()


def tiny_donor(seed=1):
    torch.manual_seed(seed)
    config = transformers.Qwen2Config(
        hidden_size=24, intermediate_size=48, num_hidden_layers=4, num_attention_heads=4,
        num_key_value_heads=2, vocab_size=40, max_position_embeddings=128,
    )
    return transformers.Qwen2ForCausalLM(config).eval()


def donors():
    return {"d": (tiny_donor(), donor_tokenizer())}


def make_branch(donor_layer=2, seed=5, gate=None):
    torch.manual_seed(seed)
    branch = TapBranch(32, 24, attn_dim=16, heads=4, donor_id="d", donor_layer=donor_layer)
    if gate is not None:
        with torch.no_grad():
            branch.gate.fill_(gate)
    return branch


def test_fresh_tap_keeps_host_logits_exactly():
    host = tiny_host()
    tok = host_tokenizer()
    tok.padding_side = "left"
    encoded = tok(PROMPTS, padding=True, return_tensors="pt")
    with torch.no_grad():
        expected = host(input_ids=encoded["input_ids"], attention_mask=encoded["attention_mask"]).logits
    attach_taps(host, [(1, make_branch(), 0)])
    inputs = prepare(host, tok, donors(), PROMPTS)
    assert torch.equal(inputs["input_ids"], encoded["input_ids"])
    with torch.no_grad():
        got = host(**inputs).logits
    assert torch.equal(got, expected)


def test_visibility_matches_hand_offsets():
    host_tok = host_tokenizer()
    host_tok.padding_side = "left"
    prompts = ["the cat", "a dog ran"]
    _, _, host_ends = host_token_ends(host_tok, prompts)
    # Host, left padded: "the cat" -> pad <s> the cat; "a dog ran" -> <s> a dog ran.
    assert host_ends.tolist() == [[-1, -1, 3, 7], [-1, 1, 5, 9]]

    memory, memory_ends, valid = run_donor(tiny_donor(), donor_tokenizer(), prompts, 1)
    # Donor, right padded, one token per letter.
    assert memory_ends.tolist() == [[1, 2, 3, 5, 6, 7, -1], [1, 3, 4, 5, 7, 8, 9]]
    assert valid.tolist() == [[True] * 6 + [False], [True] * 7]

    vis = build_visibility(host_ends, memory_ends, valid)
    assert vis.shape == (2, 4, 7)
    expected_first = [
        [False] * 7,                                       # pad
        [False] * 7,                                       # <s>
        [True, True, True, False, False, False, False],    # "the" (end 3): t h e
        [True, True, True, True, True, True, False],       # "cat" (end 7): all letters, not the pad
    ]
    expected_second = [
        [False] * 7,                                       # <s>
        [True, False, False, False, False, False, False],  # "a" (end 1)
        [True, True, True, True, False, False, False],     # "dog" (end 5)
        [True] * 7,                                        # "ran" (end 9)
    ]
    assert vis[0].tolist() == expected_first
    assert vis[1].tolist() == expected_second


def test_rows_with_no_visible_memory_give_zero_and_no_nan():
    branch = make_branch(gate=0.7)
    h = torch.randn(2, 3, 32, requires_grad=True)
    memory = torch.randn(2, 5, 24)
    mask = torch.ones(2, 3, 5, dtype=torch.bool)
    mask[0, 0] = False
    mask[1, :] = False
    mask[0, 2, 3:] = False
    out = branch(h, memory, mask)
    assert not torch.isnan(out).any()
    assert torch.equal(out[0, 0], h[0, 0])
    assert torch.equal(out[1], h[1])
    assert not torch.equal(out[0, 1], h[0, 1])
    out.sum().backward()
    for param in branch.parameters():
        assert param.grad is not None
        assert not torch.isnan(param.grad).any()
    assert not torch.isnan(h.grad).any()

    # Memory tokens hidden from a row do not change that row.
    other = memory.clone()
    other[0, 3:] = torch.randn(2, 24)
    out2 = branch(h, other, mask)
    assert torch.allclose(out2[0, 2], out[0, 2])


def test_donor_stops_after_its_layer():
    donor = tiny_donor()
    tok = donor_tokenizer()
    calls = [0, 0, 0, 0]
    handles = []
    for index, layer in enumerate(donor.model.layers):

        def count(module, inputs, output, index=index):
            calls[index] += 1

        handles.append(layer.register_forward_hook(count))
    norm_calls = [0]
    handles.append(donor.model.norm.register_forward_hook(lambda m, i, o: norm_calls.__setitem__(0, norm_calls[0] + 1)))
    memory, ends, valid = run_donor(donor, tok, PROMPTS, 1)
    assert calls == [1, 1, 0, 0]
    assert norm_calls[0] == 0
    for handle in handles:
        handle.remove()

    # The memory equals that layer's output in a full run.
    tok.padding_side = "right"
    encoded = tok(PROMPTS, padding=True, return_tensors="pt")
    with torch.no_grad():
        full = donor(**encoded, output_hidden_states=True)
    assert torch.allclose(memory, full.hidden_states[2], atol=1e-6)
    assert memory.dtype == torch.float32


def test_shared_donor_runs_once_for_two_taps():
    host = tiny_host()
    tok = host_tokenizer()
    tok.padding_side = "left"
    attach_taps(host, [(0, make_branch(donor_layer=1), 0), (2, make_branch(donor_layer=3, seed=6), 1)])
    pool = donors()
    prepare(host, tok, pool, PROMPTS)
    assert host.wsp_tap_stats["donor_runs"] == 1
    notes = donor_flops_note(host, pool)
    assert notes[0]["layers_ran"] == 4
    assert notes[0]["layers_total"] == 4
    memories = host.wsp_tap_state["memories"]
    assert not torch.equal(memories[0][0], memories[1][0])


def test_derange_rolls_memory_by_one():
    host = tiny_host()
    tok = host_tokenizer()
    tok.padding_side = "left"
    attach_taps(host, [(1, make_branch(), 0)])
    pool = donors()
    prepare(host, tok, pool, PROMPTS)
    plain_memory, plain_valid, plain_vis = host.wsp_tap_state["memories"][0]
    prepare(host, tok, pool, PROMPTS, derange=True)
    memory, valid, vis = host.wsp_tap_state["memories"][0]
    assert torch.equal(memory, torch.roll(plain_memory, 1, 0))
    assert torch.equal(valid, torch.roll(plain_valid, 1, 0))
    assert not torch.equal(memory, plain_memory)
    with pytest.raises(ValueError):
        prepare(host, tok, pool, PROMPTS[:1], derange=True)


def test_train_lowers_loss_and_touches_only_branches():
    host = tiny_host()
    tok = host_tokenizer()
    pool = donors()
    donor_model = pool["d"][0]
    attach_taps(host, [(1, make_branch(), 0)])
    pairs = [("the cat sat on a", " mat"), ("a dog ran to the", " big red hat"), ("what is the", " cat")]
    host_before = {}
    for name, value in host.state_dict().items():
        if not name.startswith("wsp_taps"):
            host_before[name] = value.clone()
    donor_before = {}
    for name, value in donor_model.state_dict().items():
        donor_before[name] = value.clone()
    branch_before = []
    for param in tap_parameters(host):
        branch_before.append(param.detach().clone())

    losses = tap_train(host, tok, pool, pairs, tap_parameters(host), steps=30, lr=1e-2, batch_size=3, seed=0,
                       log_every=0)
    assert len(losses) == 30
    assert sum(losses[-5:]) / 5 < sum(losses[:5]) / 5 - 0.1

    for name, value in host.state_dict().items():
        if not name.startswith("wsp_taps"):
            assert torch.equal(value, host_before[name]), name
    for name, value in donor_model.state_dict().items():
        assert torch.equal(value, donor_before[name]), name
    for param in host.parameters():
        is_branch = False
        for tap_param in tap_parameters(host):
            if param is tap_param:
                is_branch = True
        if not is_branch:
            assert param.grad is None
    changed = 0
    for before, param in zip(branch_before, tap_parameters(host)):
        if not torch.equal(before, param.detach()):
            changed += 1
    assert changed == len(branch_before)
    assert host.wsp_tap_state["memories"] is None


def test_train_labels_only_completion():
    from wsp.tap import encode_pairs

    tok = host_tokenizer()
    input_ids, attention_mask, labels, ends = encode_pairs(tok, [("the cat", " sat on"), ("a", " dog")], 16)
    # <s> the cat sat on </s>  and  <s> a dog </s> <pad> <pad>
    assert attention_mask.tolist() == [[1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 0, 0]]
    sat = tok.convert_tokens_to_ids("sat")
    on = tok.convert_tokens_to_ids("on")
    dog = tok.convert_tokens_to_ids("dog")
    assert labels.tolist() == [[-100, -100, -100, sat, on, 1], [-100, -100, dog, 1, -100, -100]]
    assert ends.tolist() == [[-1, 3, 7, 11, 14, 14], [-1, 1, 5, 5, -1, -1]]


def test_cached_generate_equals_uncached_with_left_padding():
    host = tiny_host()
    tok = host_tokenizer()
    tok.padding_side = "left"
    attach_taps(host, [(1, make_branch(gate=1.0), 0), (2, make_branch(donor_layer=3, seed=7, gate=-0.8), 1)])
    pool = donors()
    inputs = prepare(host, tok, pool, PROMPTS)
    settings = {"max_new_tokens": 6, "do_sample": False, "pad_token_id": tok.pad_token_id,
                "output_scores": True, "return_dict_in_generate": True}
    with torch.no_grad():
        cached = host.generate(**inputs, use_cache=True, **settings)
        uncached = host.generate(**inputs, use_cache=False, **settings)
    assert torch.equal(cached.sequences, uncached.sequences)
    for a, b in zip(cached.scores, uncached.scores):
        assert torch.allclose(a, b, atol=1e-5)

    # The branch matters, so the test checks something.
    with torch.no_grad():
        for branch in host.wsp_taps:
            branch.gate.zero_()
        plain = host.generate(**inputs, use_cache=True, **settings)
    differs = False
    for a, b in zip(cached.scores, plain.scores):
        if not torch.allclose(a, b, atol=1e-4):
            differs = True
    assert differs


def test_generated_tokens_see_all_memory():
    # A decode step is past the prompt, so it reads every valid memory token,
    # the same as a full uncached forward over prompt + new token.
    host = tiny_host()
    tok = host_tokenizer()
    tok.padding_side = "left"
    attach_taps(host, [(1, make_branch(gate=1.0), 0)])
    inputs = prepare(host, tok, donors(), PROMPTS)
    extra = torch.full((3, 1), tok.convert_tokens_to_ids("hat"))
    ids = torch.cat([inputs["input_ids"], extra], dim=1)
    mask = torch.cat([inputs["attention_mask"], torch.ones(3, 1, dtype=torch.long)], dim=1)
    with torch.no_grad():
        full = host(input_ids=ids, attention_mask=mask).logits[:, -1]
        prefill = host(**inputs, use_cache=True)
        step = host(input_ids=extra, attention_mask=mask, past_key_values=prefill.past_key_values,
                    use_cache=True).logits[:, -1]
    assert torch.allclose(full, step, atol=1e-5)


def test_forward_without_context_raises():
    host = tiny_host()
    tok = host_tokenizer()
    attach_taps(host, [(1, make_branch(), 0)])
    prepare(host, tok, donors(), PROMPTS)
    clear_tap_context(host)
    with pytest.raises(RuntimeError):
        host(input_ids=torch.tensor([[3, 4, 5]]))


def test_save_then_load_gives_same_logits():
    tok = host_tokenizer()
    tok.padding_side = "left"
    pool = donors()
    host = tiny_host()
    attach_taps(host, [(1, make_branch(gate=0.9), 0), (2, make_branch(donor_layer=3, seed=8, gate=0.4), 1)])
    inputs = prepare(host, tok, pool, PROMPTS)
    with torch.no_grad():
        expected = host(**inputs).logits
    with tempfile.TemporaryDirectory() as path:
        save_taps(host, path, meta={"d": "tiny qwen2"})
        fresh = tiny_host()
        info = load_taps(fresh, path)
    assert info["meta"] == {"d": "tiny qwen2"}
    assert info["taps"][1]["donor_layer"] == 3
    assert tap_param_count(fresh) == tap_param_count(host)
    inputs = prepare(fresh, tok, pool, PROMPTS)
    with torch.no_grad():
        got = fresh(**inputs).logits
    assert torch.equal(got, expected)


def test_param_count():
    host = tiny_host()
    attach_taps(host, [(1, make_branch(), 0)])
    # Two norms, q, k, v, o, and the gate.
    expected = 32 + 24 + 32 * 16 + 24 * 16 * 2 + 16 * 32 + 1
    assert tap_param_count(host) == expected


def test_beam_search_cached_equals_uncached():
    host = tiny_host()
    tok = host_tokenizer()
    tok.padding_side = "left"
    attach_taps(host, [(1, make_branch(gate=1.0), 0)])
    inputs = prepare(host, tok, donors(), PROMPTS)
    settings = {"max_new_tokens": 5, "do_sample": False, "num_beams": 3, "num_return_sequences": 2,
                "pad_token_id": tok.pad_token_id}
    with torch.no_grad():
        cached = host.generate(**inputs, use_cache=True, **settings)
        uncached = host.generate(**inputs, use_cache=False, **settings)
    assert cached.shape[0] == 6
    assert torch.equal(cached, uncached)


def completion_logits_match(derange):
    """Return the max logit gap between the training path and the scoring path on completion tokens."""
    from wsp.tap import train_batch

    host = tiny_host()
    tok = host_tokenizer()
    tok.padding_side = "left"
    attach_taps(host, [(1, make_branch(gate=1.0), 0)])
    pool = donors()
    # Short prompts read long foreign prompts under derange.
    pairs = [("a", " dog ran to the mat"), ("the cat sat on a mat", " big red hat"), ("what is", " the cat")]
    inputs = train_batch(host, tok, pool, pairs, 32, derange=derange)
    with torch.no_grad():
        train = host(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"]).logits
    labels = inputs["labels"]

    prompts = []
    for prompt, completion in pairs:
        prompts.append(prompt)
    encoded = prepare(host, tok, pool, prompts, derange=derange)
    prompt_len = encoded["input_ids"].shape[1]
    longest = 0
    for row in range(len(pairs)):
        count = int((labels[row] != -100).sum())
        if count > longest:
            longest = count
    ids = torch.full((len(pairs), prompt_len + longest), tok.pad_token_id)
    mask = torch.zeros_like(ids)
    ids[:, :prompt_len] = encoded["input_ids"]
    mask[:, :prompt_len] = encoded["attention_mask"]
    for row in range(len(pairs)):
        completion = labels[row][labels[row] != -100]
        ids[row, prompt_len:prompt_len + len(completion)] = completion
        mask[row, prompt_len:prompt_len + len(completion)] = 1
    with torch.no_grad():
        score = host(input_ids=ids, attention_mask=mask).logits

    worst = 0.0
    for row in range(len(pairs)):
        positions = (labels[row] != -100).nonzero().flatten().tolist()
        for k, position in enumerate(positions):
            # The logit at position - 1 predicts the label at position.
            gap = float((train[row, position - 1] - score[row, prompt_len - 1 + k]).abs().max())
            if gap > worst:
                worst = gap
    return worst


def test_training_logits_equal_scoring_logits():
    assert completion_logits_match(derange=False) < 1e-5


def test_training_logits_equal_scoring_logits_under_derange():
    # Completion tokens read all of the other prompt's memory, the same as
    # decode steps at scoring time.
    assert completion_logits_match(derange=True) < 1e-5
