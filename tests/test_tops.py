import copy
import json
import random

import pytest
import torch

transformers = pytest.importorskip("transformers")

from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast

from tests.test_align import WORDS, tiny_tokenizer
from wsp.heal import heal
from wsp.lora import rank_for_budget
from wsp.tops import (ForcedView, SkillTops, check_vocab, config_counts, fit_kd_lora, fit_stitch, sizes_from_configs)

VOCAB = 300


def tiny_qwen2(seed, hidden=48, layers=4, vocab=VOCAB, tie=False, heads=4, kv=2, intermediate=96, theta=1000000.0):
    torch.manual_seed(seed)
    config = transformers.Qwen2Config(
        hidden_size=hidden, intermediate_size=intermediate, num_hidden_layers=layers, num_attention_heads=heads,
        num_key_value_heads=kv, vocab_size=vocab, max_position_embeddings=64, tie_word_embeddings=tie,
        rope_theta=theta,
    )
    return transformers.Qwen2ForCausalLM(config).eval()


def same_embeddings(donor, base):
    with torch.no_grad():
        donor.model.embed_tokens.weight.copy_(base.model.embed_tokens.weight)
    return donor


def cycle_texts(n, seed, words=None):
    words = words or WORDS
    rng = random.Random(seed)
    texts = []
    for _ in range(n):
        start = rng.randrange(len(words))
        step = rng.choice([1, 5])
        out = []
        for i in range(12):
            out.append(words[(start + step * i) % len(words)])
        texts.append(" ".join(out))
    return texts


def planted(words, n, seed, length=10):
    rng = random.Random(seed)
    texts = []
    for _ in range(n):
        out = []
        for _ in range(length):
            out.append(rng.choice(words))
        texts.append(" ".join(out))
    return texts


def encode(texts, left=True):
    tok = tiny_tokenizer()
    tok.padding_side = "left" if left else "right"
    enc = tok(texts, return_tensors="pt", padding=True)
    return enc["input_ids"], enc["attention_mask"]


def make_net(dtype=None, cuts=(1, 2), widths=(48, 32), tied=False):
    """A tiny base and two tops with different cuts and one with another width."""
    base = tiny_qwen2(0)
    net = SkillTops(base, tiny_tokenizer())
    donors = []
    for index, (cut, width) in enumerate(zip(cuts, widths)):
        heads = 4
        donor = tiny_qwen2(10 + index, hidden=width, layers=4 if index == 0 else 3, heads=heads, kv=2,
                           intermediate=2 * width, theta=10000.0 if index else 1000000.0)
        donors.append(donor)
        net.add_top("t" + str(index), donor, cut, stitch_kind="mlp", donor_tokenizer=tiny_tokenizer())
    if dtype is not None:
        net = net.to(dtype)
    return net, base, donors


def with_sinks(net):
    """Give every top a fixed random sink, so the sink path runs without a fit."""
    torch.manual_seed(7)
    for name in net.tops:
        top = net.tops[name]
        with torch.no_grad():
            top.sink.copy_(torch.randn_like(top.sink))
            for param in top.stitch.parameters():
                param.add_(0.05 * torch.randn_like(param))
        top.has_sink = True
    return net


def reference_generate(net, ids, mask, new, top=None, eos=(), pad=0):
    """Uncached greedy decoding: a full forward on the growing sequence every step.

    The route is chosen once from the prompt, as generate does.
    """
    ids = ids.clone()
    mask = mask.clone()
    finished = torch.zeros(ids.shape[0], dtype=torch.bool)
    names = net.choose(ids, mask, top)
    out = []
    for _ in range(new):
        if top is None:
            logits = routed_forward(net, ids, mask, names)
        else:
            logits = net(ids, attention_mask=mask, top=top).logits
        token = logits[:, -1].argmax(dim=-1)
        token = torch.where(finished, torch.full_like(token, pad), token)
        out.append(token)
        if len(eos) > 0:
            finished = finished | torch.isin(token, torch.tensor(list(eos)))
        ids = torch.cat([ids, token[:, None]], dim=1)
        mask = torch.cat([mask, torch.ones(ids.shape[0], 1, dtype=mask.dtype)], dim=1)
        if bool(finished.all()):
            break
    return torch.stack(out, dim=1)


def routed_forward(net, ids, mask, names):
    """Per-row forced forwards with the names the router chose once from the prompt."""
    rows = []
    for row in range(ids.shape[0]):
        rows.append(net(ids[row:row + 1], attention_mask=mask[row:row + 1], top=names[row]).logits)
    return torch.cat(rows, dim=0)


# ---------------------------------------------------------------- exactness


def test_cut_zero_identity_stitch_own_head_equals_the_donor():
    base = tiny_qwen2(0)
    donor = same_embeddings(tiny_qwen2(1), base)
    net = SkillTops(base, tiny_tokenizer())
    top = net.add_top("d", donor, 0, 0, stitch_kind="linear", donor_tokenizer=tiny_tokenizer())
    assert torch.equal(top.stitch.weight, torch.eye(48))
    ids, mask = encode(cycle_texts(3, 0))
    with torch.no_grad():
        want = donor(input_ids=ids, attention_mask=mask).logits
        got = net(ids, attention_mask=mask, top="d").logits
    assert (want - got).abs().max().item() < 1e-5
    assert top.spec["head_masked"] == 0


def test_cut_zero_with_the_mlp_stitch_is_also_the_identity():
    base = tiny_qwen2(0)
    donor = same_embeddings(tiny_qwen2(1), base)
    net = SkillTops(base, tiny_tokenizer())
    net.add_top("d", donor, 0, 0, stitch_kind="mlp", donor_tokenizer=tiny_tokenizer())
    ids, mask = encode(cycle_texts(2, 1))
    with torch.no_grad():
        want = donor(input_ids=ids, attention_mask=mask).logits
        got = net(ids, attention_mask=mask, top="d").logits
    assert (want - got).abs().max().item() < 1e-5


def test_the_base_top_equals_the_base_exactly():
    net, base, _ = make_net()
    ids, mask = encode(cycle_texts(3, 2))
    with torch.no_grad():
        want = base(input_ids=ids, attention_mask=mask).logits
        got = net(ids, attention_mask=mask, top="base").logits
    assert torch.equal(want, got)


def test_a_top_at_the_depth_of_an_identical_donor_equals_the_base():
    # Donor equal to the base, identity stitch, cut in the middle: the path is the base itself.
    base = tiny_qwen2(0)
    donor = copy.deepcopy(base)
    net = SkillTops(base, tiny_tokenizer())
    net.add_top("same", donor, 2, 2, stitch_kind="linear", donor_tokenizer=tiny_tokenizer())
    ids, mask = encode(cycle_texts(3, 3))
    with torch.no_grad():
        want = base(input_ids=ids, attention_mask=mask).logits
        got = net(ids, attention_mask=mask, top="same").logits
    assert (want - got).abs().max().item() < 1e-5


# ---------------------------------------------------------------- stitch fit


@pytest.fixture(scope="module")
def trained_donor():
    donor = tiny_qwen2(0)
    heal(donor, tiny_tokenizer(), cycle_texts(64, seed=0), list(donor.parameters()), steps=60, lr=1e-2,
         batch_size=16, seq_len=32, log_every=0)
    return donor.eval()


def perturbed(model, noise, seed):
    host = copy.deepcopy(model)
    torch.manual_seed(seed)
    with torch.no_grad():
        for param in host.parameters():
            if param.dim() == 2:
                param.add_(noise * param.std() * torch.randn_like(param))
    return host.eval()


def test_a_trained_stitch_lowers_the_donor_penalty(trained_donor):
    donor = trained_donor
    base = perturbed(donor, 1.0, 5)
    net = SkillTops(base, tiny_tokenizer())
    top = net.add_top("d", donor, 2, 2, stitch_kind="linear", donor_tokenizer=tiny_tokenizer())
    texts = cycle_texts(80, 1)
    held = cycle_texts(24, 2)
    report = fit_stitch(top, texts, steps=120, lr=3e-3, batch_size=8, seq_len=24, eval_texts=held,
                        donor_model=donor, max_penalty=5.0)
    assert report["after"]["penalty"] < report["before"]["penalty"]
    assert report["before"]["penalty"] > 0.02
    assert report["loss_last25"] < report["loss_first"]
    assert "penalty_all" in report["after"]
    assert top.has_sink
    assert report["accepted"] is True
    assert report["ridge"]["pairs"] > 20
    assert report["trainable_params"] == 48 * 48


def test_a_penalty_above_the_threshold_flags_the_top_and_it_is_refused(trained_donor):
    donor = trained_donor
    base = perturbed(donor, 0.4, 5)
    net = SkillTops(base, tiny_tokenizer())
    top = net.add_top("d", donor, 2, 2, stitch_kind="linear", donor_tokenizer=tiny_tokenizer())
    report = net.fit_stitch("d", cycle_texts(40, 1), steps=2, lr=1e-4, batch_size=4, seq_len=24,
                            eval_texts=cycle_texts(10, 2), donor_model=donor, max_penalty=-1.0)
    assert report["accepted"] is False
    assert "above" in report["flag"]
    ids, mask = encode(cycle_texts(2, 4))
    with pytest.raises(ValueError):
        net(ids, attention_mask=mask, top="d")
    assert "d" not in net.router_classes()
    net.allow_flagged = True
    assert net(ids, attention_mask=mask, top="d").logits.shape[-1] == VOCAB
    assert "d" in net.router_classes()


def test_top_lora_trains_with_the_stitch_and_is_counted(trained_donor):
    donor = trained_donor
    base = perturbed(donor, 0.4, 5)
    net = SkillTops(base, tiny_tokenizer())
    top = net.add_top("d", donor, 2, 2, stitch_kind="linear", donor_tokenizer=tiny_tokenizer())
    before = net.count_params()["tops"]["d"]["stored"]
    report = net.fit_stitch("d", cycle_texts(40, 1), steps=20, lr=3e-3, batch_size=8, seq_len=24,
                            eval_texts=cycle_texts(10, 2), donor_model=donor, lora_rank=2, max_penalty=9.0)
    assert top.lora["rank"] == 2
    after = net.count_params()["tops"]["d"]
    assert after["lora"] > 0
    assert after["stored"] == before + after["lora"] + after["sink"]
    assert report["trainable_params"] == 48 * 48 + after["lora"]


# ---------------------------------------------------------------- router


def planted_router_net():
    base = tiny_qwen2(0)
    net = SkillTops(base, tiny_tokenizer())
    net.add_top("a", tiny_qwen2(11), 1, donor_tokenizer=tiny_tokenizer())
    net.add_top("b", tiny_qwen2(12), 2, donor_tokenizer=tiny_tokenizer())
    return net


A_WORDS = ["the", "cat", "sat", "on"]
B_WORDS = ["dog", "ran", "to", "big"]
G_WORDS = ["a", "mat", "red", "hat"]


def test_router_sends_planted_domains_to_their_tops_and_unsure_text_to_the_base():
    net = planted_router_net()
    report = net.fit_router({"base": planted(G_WORDS, 60, 0), "a": planted(A_WORDS, 60, 1),
                             "b": planted(B_WORDS, 60, 2)}, lam=1e-3, scale=8.0)
    assert report["classes"] == ["base", "a", "b"]
    assert report["cut"] == 1
    assert net.route_texts(planted(A_WORDS, 12, 10)) == ["a"] * 12
    assert net.route_texts(planted(B_WORDS, 12, 11)) == ["b"] * 12
    assert net.route_texts(planted(G_WORDS, 12, 12)) == ["base"] * 12
    # Unsure text mixes the domains. The base margin pulls it, and everything near the line, to the base.
    mixed = planted(A_WORDS + B_WORDS + G_WORDS, 20, 13)
    plain = net.route_shares(mixed)
    net.base_margin = 3.0
    margin = net.route_shares(mixed)
    assert margin["base"] >= plain["base"]
    assert margin["base"] > 0.5
    net.base_margin = 100.0
    assert net.route_shares(planted(A_WORDS, 12, 10))["base"] == 1.0


def test_tune_base_margin_picks_a_margin_that_keeps_general_text_on_the_base():
    net = planted_router_net()
    net.fit_router({"base": planted(G_WORDS, 60, 0), "a": planted(A_WORDS, 60, 1), "b": planted(B_WORDS, 60, 2)},
                   lam=1e-3)
    result = net.tune_base_margin({"base": planted(G_WORDS, 20, 20), "a": planted(A_WORDS, 20, 21),
                                   "b": planted(B_WORDS, 20, 22)})
    assert result["chosen"] in [row["margin"] for row in result["table"]]
    assert net.base_margin == result["chosen"]
    assert result["table"][0]["recall"]["a"] > 0.9


def test_route_needs_a_fit():
    net = planted_router_net()
    ids, mask = encode(cycle_texts(2, 0))
    with pytest.raises(ValueError):
        net(ids, attention_mask=mask)


def routed_net(dtype=None):
    net = planted_router_net()
    net.fit_router({"base": planted(G_WORDS, 60, 0), "a": planted(A_WORDS, 60, 1), "b": planted(B_WORDS, 60, 2)},
                   lam=1e-3)
    if dtype is not None:
        net = net.to(dtype)
    return net


def test_mixed_batches_equal_per_row_runs():
    net = with_sinks(routed_net())
    texts = [planted(A_WORDS, 1, 3)[0], planted(B_WORDS, 1, 4)[0], planted(G_WORDS, 1, 5)[0],
             planted(A_WORDS, 1, 6)[0], planted(B_WORDS, 1, 7)[0]]
    ids, mask = encode(texts)
    with torch.no_grad():
        got = net(ids, attention_mask=mask)
        assert got.choice == ["a", "b", "base", "a", "b"]
        for row in range(ids.shape[0]):
            want = net(ids[row:row + 1], attention_mask=mask[row:row + 1], top=got.choice[row]).logits
            assert (want[0] - got.logits[row]).abs().max().item() < 1e-5


def test_the_route_tally_counts_every_routed_row():
    net = routed_net()
    texts = planted(A_WORDS, 3, 1) + planted(B_WORDS, 2, 2) + planted(G_WORDS, 4, 3)
    ids, mask = encode(texts)
    net.tally = {}
    net(ids, attention_mask=mask)
    assert net.tally == {"a": 3, "b": 2, "base": 4}


def test_padded_mixed_batches_equal_unpadded_rows():
    net = with_sinks(routed_net(torch.float64))
    texts = [" ".join(planted(A_WORDS, 1, 3, length=6)[0].split()), planted(B_WORDS, 1, 4, length=11)[0],
             planted(G_WORDS, 1, 5, length=8)[0]]
    ids, mask = encode(texts)
    assert int(mask.sum()) < mask.numel()
    with torch.no_grad():
        got = net(ids, attention_mask=mask)
        for row in range(ids.shape[0]):
            n = int(mask[row].sum())
            alone = net(ids[row:row + 1, -n:], top=got.choice[row]).logits[0]
            assert (alone - got.logits[row, -n:]).abs().max().item() < 1e-9


# ---------------------------------------------------------------- generation


@pytest.mark.parametrize("top", ["t0", "t1", "base"])
def test_cached_generation_equals_uncached_batched_left_padded_two_calls(top):
    net, _, _ = make_net(torch.float64)
    with_sinks(net)
    for call in range(2):
        texts = [planted(WORDS, 1, 30 + call)[0], planted(WORDS, 1, 40 + call, length=5)[0],
                 planted(WORDS, 1, 50 + call, length=7)[0]]
        ids, mask = encode(texts)
        assert int(mask.sum()) < mask.numel()
        made = net.generate(ids, attention_mask=mask, max_new_tokens=9, top=top, pad_token_id=0)
        want = reference_generate(net, ids, mask, 9, top=top)
        assert torch.equal(made[:, ids.shape[1]:], want)


def test_routed_cached_generation_equals_uncached_with_mixed_tops():
    net = with_sinks(routed_net(torch.float64))
    for call in range(2):
        texts = [planted(A_WORDS, 1, 60 + call, length=7)[0], planted(B_WORDS, 1, 70 + call, length=5)[0],
                 planted(G_WORDS, 1, 80 + call, length=9)[0], planted(A_WORDS, 1, 90 + call, length=6)[0]]
        ids, mask = encode(texts)
        made = net.generate(ids, attention_mask=mask, max_new_tokens=7, pad_token_id=0)
        choice = list(net.last_choice)
        assert set(choice) == {"a", "b", "base"}
        want = reference_generate(net, ids, mask, 7)
        assert torch.equal(made[:, ids.shape[1]:], want)
        assert torch.equal(made[:, :ids.shape[1]], ids)


def test_generation_stops_a_row_at_eos_and_pads_it():
    net, _, _ = make_net(torch.float64)
    ids, mask = encode([planted(WORDS, 1, 3)[0], planted(WORDS, 1, 4, length=6)[0]])
    free = net.generate(ids, attention_mask=mask, max_new_tokens=8, top="t0", pad_token_id=0)
    eos = int(free[0, ids.shape[1] + 2])
    made = net.generate(ids, attention_mask=mask, max_new_tokens=8, top="t0", pad_token_id=0, eos_token_id=[eos])
    new = made[0, ids.shape[1]:]
    stop = int((new == eos).nonzero()[0])
    assert stop <= 2
    assert bool((new[stop + 1:] == 0).all())
    want = reference_generate(net, ids, mask, 8, top="t0", eos=[eos], pad=0)
    assert torch.equal(made[:, ids.shape[1]:ids.shape[1] + want.shape[1]], want)


def test_generate_accepts_the_eval_harness_arguments_and_the_forced_view():
    net, _, _ = make_net()
    view = ForcedView(net, "t0")
    ids, mask = encode(cycle_texts(2, 0))
    out = view.generate(input_ids=ids, attention_mask=mask, max_new_tokens=4, do_sample=False, temperature=None,
                        top_p=None, top_k=None, repetition_penalty=1.0, pad_token_id=2, eos_token_id=[1],
                        stopping_criteria=None)
    assert out.shape[0] == 2 and out.shape[1] > ids.shape[1]
    assert next(view.parameters()) is not None
    with pytest.raises(ValueError):
        net.generate(ids, attention_mask=mask, do_sample=True)
    with pytest.raises(TypeError):
        net.generate(ids, attention_mask=mask, repetition_penalty=1.3)


def test_stopping_criteria_end_a_group():
    net, _, _ = make_net()
    ids, mask = encode(cycle_texts(2, 0))

    class StopAtThree:
        def __call__(self, input_ids, scores, **kwargs):
            return torch.full((input_ids.shape[0],), input_ids.shape[1] >= ids.shape[1] + 3, dtype=torch.bool)

    out = net.generate(ids, attention_mask=mask, max_new_tokens=9, top="t1", pad_token_id=0,
                       stopping_criteria=StopAtThree())
    assert out.shape[1] == ids.shape[1] + 3


# ---------------------------------------------------------------- shapes, widths, heads


def test_tops_with_other_cuts_and_widths_run_and_return_the_base_width():
    net, base, donors = make_net()
    ids, mask = encode(cycle_texts(3, 0))
    assert net.tops["t0"].cut_host == 1 and net.tops["t1"].cut_host == 2
    assert net.tops["t1"].spec["d_donor"] == 32
    assert net.tops["t1"].cut_donor == 2
    for name in ("t0", "t1", "base"):
        logits = net(ids, attention_mask=mask, top=name).logits
        assert logits.shape == (3, ids.shape[1], VOCAB)
    keep = net(ids, attention_mask=mask, top="t1", logits_to_keep=1).logits
    assert keep.shape == (3, 1, VOCAB)
    with torch.no_grad():
        full = net(ids, attention_mask=mask, top="t1").logits
    assert torch.allclose(full[:, -1], keep[:, 0], atol=1e-6)


def test_donor_rotary_base_is_kept():
    net, _, donors = make_net()
    rotary = net.tops["t1"]._rotary[0]
    assert torch.equal(rotary.inv_freq, donors[1].model.rotary_emb.inv_freq)
    assert float(donors[1].config.rope_theta) == 10000.0


def test_a_narrower_head_is_padded_and_a_wider_head_is_cut():
    base = tiny_qwen2(0)
    net = SkillTops(base, tiny_tokenizer())
    narrow = net.add_top("narrow", tiny_qwen2(1, vocab=280), 1, donor_tokenizer=tiny_tokenizer())
    wide = net.add_top("wide", tiny_qwen2(2, vocab=320), 1, donor_tokenizer=tiny_tokenizer())
    assert narrow.spec["head_rows_padded"] == 20 and narrow.spec["head_masked"] == 20
    assert wide.spec["head_rows_dropped"] == 20 and wide.spec["head_masked"] == 0
    ids, mask = encode(cycle_texts(2, 0))
    with torch.no_grad():
        logits = net(ids, attention_mask=mask, top="narrow").logits
        assert logits.shape[-1] == VOCAB
        assert bool((logits[..., 280:] == torch.finfo(logits.dtype).min).all())
        assert bool((logits[..., :280] > torch.finfo(logits.dtype).min).all())
        assert net(ids, attention_mask=mask, top="wide").logits.shape[-1] == VOCAB
    # The stored head of the narrow top holds the padding rows, and the count says so.
    assert net.count_params()["tops"]["narrow"]["head"] == VOCAB * 48


def test_vocab_check_refuses_moved_ids_and_a_head_that_misses_shared_ids():
    def reordered():
        vocab = {"<unk>": 0, "</s>": 1, "<pad>": 2}
        for word in reversed(WORDS):
            vocab[word] = len(vocab)
        base = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
        base.pre_tokenizer = pre_tokenizers.Whitespace()
        return PreTrainedTokenizerFast(tokenizer_object=base, unk_token="<unk>", eos_token="</s>", pad_token="<pad>")

    with pytest.raises(ValueError, match="different ids"):
        check_vocab(tiny_tokenizer(), reordered(), 300, 300)
    with pytest.raises(ValueError, match="rows"):
        check_vocab(tiny_tokenizer(), tiny_tokenizer(), 10, 300)
    report = check_vocab(tiny_tokenizer(), tiny_tokenizer(), 300, 300)
    assert report["base_only"] == 0 and report["masked_ids"] == []

    def fewer():
        vocab = {"<unk>": 0, "</s>": 1, "<pad>": 2}
        for word in WORDS[:-2]:
            vocab[word] = len(vocab)
        base = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
        base.pre_tokenizer = pre_tokenizers.Whitespace()
        return PreTrainedTokenizerFast(tokenizer_object=base, unk_token="<unk>", eos_token="</s>", pad_token="<pad>")

    report = check_vocab(tiny_tokenizer(), fewer(), 300, 300)
    assert report["base_only"] == 2 and len(report["masked_ids"]) == 2
    base = tiny_qwen2(0)
    net = SkillTops(base, tiny_tokenizer())
    top = net.add_top("f", tiny_qwen2(1), 1, donor_tokenizer=fewer())
    ids, mask = encode(cycle_texts(2, 0))
    with torch.no_grad():
        logits = net(ids, attention_mask=mask, top="f").logits
    for index in report["masked_ids"]:
        assert bool((logits[..., index] == torch.finfo(logits.dtype).min).all())
    assert top.spec["head_masked"] == 2


def test_top_names_and_options_are_checked():
    net, _, _ = make_net()
    with pytest.raises(ValueError):
        net.add_top("base", tiny_qwen2(3), 1, donor_tokenizer=tiny_tokenizer())
    with pytest.raises(ValueError):
        net.add_top("t0", tiny_qwen2(3), 1, donor_tokenizer=tiny_tokenizer())
    with pytest.raises(NotImplementedError):
        net.add_top("x", tiny_qwen2(3), 1, own_head=False, donor_tokenizer=tiny_tokenizer())
    with pytest.raises(ValueError):
        net.add_top("x", tiny_qwen2(3), 1, control="random", donor_tokenizer=tiny_tokenizer())
    with pytest.raises(ValueError):
        net(*encode(cycle_texts(1, 0)), top="nope")


# ---------------------------------------------------------------- controls, save, load


def saved_donor(tmp_path, name, seed, hidden, tie=False, layers=4):
    donor = tiny_qwen2(seed, hidden=hidden, layers=layers, tie=tie, intermediate=2 * hidden)
    path = str(tmp_path / name)
    donor.save_pretrained(path)
    tiny_tokenizer().save_pretrained(path)
    return path


def test_a_scrambled_control_differs_and_is_rescrambled_on_load(tmp_path):
    path = saved_donor(tmp_path, "donor", 21, 48)
    base = tiny_qwen2(0)
    net = SkillTops(base, tiny_tokenizer(), base_id="tiny-base")
    real = net.add_top("real", path, 2)
    scr = net.add_top("scr", path, 2, control="scrambled", control_seed=3)
    scr_other = net.add_top("scr2", path, 2, control="scrambled", control_seed=4)
    ids, mask = encode(cycle_texts(3, 0))
    with torch.no_grad():
        a = net(ids, attention_mask=mask, top="real").logits
        b = net(ids, attention_mask=mask, top="scr").logits
        c = net(ids, attention_mask=mask, top="scr2").logits
    assert (a - b).abs().max().item() > 1e-3
    assert (b - c).abs().max().item() > 1e-3
    assert scr.spec["control"] == "scrambled" and scr.spec["control_seed"] == 3
    # Same shapes and the same norms, different weights.
    wr = real.layers[0].mlp.up_proj.weight
    ws = scr.layers[0].mlp.up_proj.weight
    assert wr.shape == ws.shape and abs(float(wr.norm()) - float(ws.norm())) < 1e-3 * float(wr.norm())
    assert not torch.allclose(wr, ws)
    net.save(str(tmp_path / "net"))
    loaded = SkillTops.load(str(tmp_path / "net"), base_model=base, tokenizer=tiny_tokenizer())
    with torch.no_grad():
        for name, want in (("real", a), ("scr", b), ("scr2", c)):
            got = loaded(ids, attention_mask=mask, top=name).logits
            assert (got - want).abs().max().item() < 1e-6


def test_save_and_load_give_equal_logits_with_tied_head_sink_lora_and_router(tmp_path):
    p0 = saved_donor(tmp_path, "d0", 31, 48, tie=True)
    p1 = saved_donor(tmp_path, "d1", 32, 32, layers=3)
    base = tiny_qwen2(0)
    net = SkillTops(base, tiny_tokenizer(), base_id="tiny-base")
    a = net.add_top("a", p0, 1)
    b = net.add_top("b", p1, 2)
    assert a.spec["donor_tied_head"] is True
    donor_model = transformers.AutoModelForCausalLM.from_pretrained(p0)
    net.fit_stitch("a", cycle_texts(30, 1), steps=6, lr=1e-3, batch_size=4, seq_len=24,
                   eval_texts=cycle_texts(8, 2), donor_model=donor_model.eval(), lora_rank=2, max_penalty=99.0)
    with_sinks(net)
    net.fit_router({"base": planted(G_WORDS, 40, 0), "a": planted(A_WORDS, 40, 1), "b": planted(B_WORDS, 40, 2)},
                   lam=1e-3, base_margin=0.5)
    texts = [planted(A_WORDS, 1, 3)[0], planted(B_WORDS, 1, 4)[0], planted(G_WORDS, 1, 5)[0]]
    ids, mask = encode(texts)
    net.save(str(tmp_path / "net"))
    loaded = SkillTops.load(str(tmp_path / "net"), base_model=base, tokenizer=tiny_tokenizer())
    with torch.no_grad():
        want = net(ids, attention_mask=mask)
        got = loaded(ids, attention_mask=mask)
        assert want.choice == got.choice
        assert (want.logits - got.logits).abs().max().item() < 1e-6
        for name in ("a", "b", "base"):
            assert (net(ids, attention_mask=mask, top=name).logits
                    - loaded(ids, attention_mask=mask, top=name).logits).abs().max().item() < 1e-6
    assert loaded.tops["a"].lora["rank"] == 2
    assert loaded.base_margin == 0.5
    meta = json.load(open(tmp_path / "net" / "tops.json"))
    assert meta["tops"][0]["fit_report"]["steps"] == 6


def test_a_top_from_a_loaded_model_cannot_be_saved(tmp_path):
    net, _, _ = make_net()
    with pytest.raises(ValueError):
        net.save(str(tmp_path / "x"))


# ---------------------------------------------------------------- counts


def test_param_counts_match_direct_counts(tmp_path):
    net, base, donors = make_net()
    with_sinks(net)
    net.fit_router({"base": planted(G_WORDS, 20, 0), "t0": planted(A_WORDS, 20, 1), "t1": planted(B_WORDS, 20, 2)},
                   lam=1e-3)
    counts = net.count_params()
    distinct = {}
    for param in net.parameters():
        distinct[id(param)] = param.numel()
    sinks = 48 + 32
    router = net.router_w.numel() + net.router_b.numel()
    assert counts["stored"] == sum(distinct.values()) + sinks + router
    base_total = sum(p.numel() for p in base.parameters())
    assert counts["base_stored"] == base_total
    assert counts["run"]["base"] == base_total - VOCAB * 48
    layer_total = sum(p.numel() for p in base.model.layers[0].parameters())
    for index, name in enumerate(("t0", "t1")):
        entry = counts["tops"][name]
        direct = sum(p.numel() for p in net.tops[name].parameters()) + net.tops[name].spec["d_donor"]
        assert entry["stored"] == direct
        assert entry["trunk"] == layer_total * net.tops[name].cut_host
        assert counts["run"][name] == entry["trunk"] + entry["stored"]
    donor0 = donors[0]
    top0 = counts["tops"]["t0"]
    assert top0["layers"] == sum(p.numel() for p in donor0.model.layers[1:].parameters())
    assert top0["head"] == donor0.lm_head.weight.numel()
    assert top0["norm"] == 48


def test_a_tied_base_counts_its_head_once_and_runs_it(tmp_path):
    base = tiny_qwen2(0, tie=True)
    net = SkillTops(base, tiny_tokenizer())
    counts = net.count_params()
    total = sum(p.numel() for p in base.parameters())
    assert counts["base_tied_head"] is True
    assert counts["base_stored"] == total
    assert counts["run"]["base"] == total


def test_sizes_from_configs_match_a_built_network():
    net, base, donors = make_net()
    with_sinks(net)
    net.fit_router({"base": planted(G_WORDS, 20, 0), "t0": planted(A_WORDS, 20, 1), "t1": planted(B_WORDS, 20, 2)},
                   lam=1e-3)
    counts = net.count_params()
    sizes = sizes_from_configs(base.config, [
        {"name": "t0", "donor_config": donors[0].config, "cut_host": 1, "cut_donor": 1},
        {"name": "t1", "donor_config": donors[1].config, "cut_host": 2, "cut_donor": 2}], router_classes=3)
    assert sizes["stored"] == counts["stored"]
    assert sizes["run"] == counts["run"]
    assert sizes["base_total"] == counts["base_stored"]
    assert config_counts(base.config)["total"] == sum(p.numel() for p in base.parameters())


def test_sizes_from_configs_count_a_tied_donor_head_in_full():
    base = tiny_qwen2(0)
    donor = tiny_qwen2(1, tie=True)
    sizes = sizes_from_configs(base.config, [{"name": "x", "donor_config": donor.config, "cut_host": 2}])
    assert sizes["tops"]["x"]["head"] == VOCAB * 48
    assert sizes["tops"]["x"]["cut_donor"] == 2


# ---------------------------------------------------------------- reference arm


def test_kd_lora_trains_a_lora_toward_the_donor(trained_donor):
    donor = trained_donor
    base = perturbed(donor, 0.4, 9)
    texts = cycle_texts(40, 1)
    ids, mask = encode(cycle_texts(4, 3))
    with torch.no_grad():
        target = donor(input_ids=ids, attention_mask=mask).logits
        before = (base(input_ids=ids, attention_mask=mask).logits - target).abs().mean().item()
    rank = rank_for_budget(base, 48 * 48)
    report = fit_kd_lora(base, donor, tiny_tokenizer(), texts, rank, steps=60, lr=3e-3, batch_size=8, seq_len=24)
    with torch.no_grad():
        after = (base(input_ids=ids, attention_mask=mask).logits - target).abs().mean().item()
    assert report["loss_last25"] < report["loss_first"]
    assert after < before
    assert report["params"] > 0


# ---------------------------------------------------------------- the round script


def test_round24_cells_and_sizes():
    pytest.importorskip("modal")
    import modal_round24

    cells = modal_round24.cells()
    names = [modal_round24.cell_name(c) for c in cells]
    assert len(names) == len(set(names))
    smoke = modal_round24.smoke_cell()
    assert smoke["pair"] == "T1" and smoke["kind"] == "forced" and smoke["top"] == "math"
    assert smoke["cut"] == 14 and smoke["skills"] == ["gsm8k"] and smoke["steps"] == 200
    kinds = {c["kind"] for c in cells}
    assert {"base", "donor", "kd_lora", "forced", "routed", "scrambled", "lora", "template"} <= kinds
    default = [c for c in cells if c["kind"] in ("forced", "routed", "scrambled", "lora") and c["cut"] == 14]
    assert {c["seed"] for c in default} == {0, 1}
    other = [c for c in cells if c["kind"] == "forced" and c["cut"] != 14]
    assert {c["seed"] for c in other} == {0}
    assert modal_round24.cuts_for(28) == [4, 8, 14, 20, 24]
    ranges = modal_round24.RANGES
    spans = sorted((start, start + count) for start, count in ranges.values())
    for left, right in zip(spans, spans[1:]):
        assert left[1] <= right[0]


def synthetic_rows(donor_gap, mmlu_drop, stored, scrambled):
    import modal_round24 as r

    rows = []
    base_scores = {"gsm8k": 0.6, "math500": 0.4, "humaneval": 0.5, "mbpp": 0.5, "mmlu": 0.6}
    donor_scores = {"math": {"gsm8k": 0.8, "math500": 0.6}, "code": {"humaneval": 0.7, "mbpp": 0.65}}
    rows.append({"pair": "T1", "kind": "base", "top": None, "cut": None, "seed": 0, "scores": base_scores,
                 "params_run": 1_000_000_000})
    for top in ("math", "code"):
        rows.append({"pair": "T1", "kind": "donor", "top": top, "cut": None, "seed": 0,
                     "scores": dict(donor_scores[top], mmlu=0.4), "params_run": 1_000_000_000})
        for cut in r.cuts_for(28):
            gap = donor_gap if cut <= 20 else 0.10
            scores = {k: v - gap for k, v in donor_scores[top].items()}
            rows.append({"pair": "T1", "kind": "forced", "top": top, "cut": cut, "seed": 0, "scores": scores})
        scr = {k: base_scores[k] - 0.05 if scrambled else base_scores[k] + 0.2 for k in donor_scores[top]}
        rows.append({"pair": "T1", "kind": "scrambled", "top": top, "cut": 14, "seed": 0, "scores": scr})
    routed = dict(base_scores)
    for top in ("math", "code"):
        for k, v in donor_scores[top].items():
            routed[k] = v - donor_gap
    routed["mmlu"] = 0.6 - mmlu_drop
    rows.append({"pair": "T1", "kind": "routed", "top": "all", "cut": 14, "seed": 0, "scores": routed,
                 "stored_over_base": stored, "params_run_per_path": {"base": 1_000_000_000, "math": 1_040_000_000,
                                                                      "code": 1_040_000_000}})
    return rows


def test_round24_summary_applies_the_go_rule(capsys):
    pytest.importorskip("modal")
    import modal_round24 as r

    good = r.summarize(synthetic_rows(0.02, 0.005, 1.5, True))["T1"]
    assert good["go"] is True
    assert good["routed"][14]["go"] is True
    assert good["deepest_cut_within_3"]["math"] == 20
    assert good["deepest_cut_within_3"]["code"] == 20
    for bad_rows in (synthetic_rows(0.05, 0.005, 1.5, True), synthetic_rows(0.02, 0.02, 1.5, True),
                     synthetic_rows(0.02, 0.005, 1.7, True), synthetic_rows(0.02, 0.005, 1.5, False)):
        assert r.summarize(bad_rows)["T1"]["go"] is False
    wide = synthetic_rows(0.02, 0.005, 1.5, True)
    wide[-1]["params_run_per_path"]["math"] = 1_100_000_000
    assert r.summarize(wide)["T1"]["go"] is False


def test_round24_plan_counts_every_pair_and_select_filters():
    pytest.importorskip("modal")
    import modal_round24 as r

    plan = r.plan()
    assert set(plan) == {"T1", "T2", "T3"}
    assert plan["T3"]["gpu_hours"] > plan["T2"]["gpu_hours"]
    one = r.select(r.cells(), pairs="T1", cuts="1/2", only="forced", seeds="0")
    assert {(c["top"], c["cut"], c["seed"]) for c in one} == {("math", 14, 0), ("code", 14, 0)}
    assert all(c["kind"] != "forced" or c["cut"] == 14 for c in r.select(r.cells(), cuts="14", only="forced"))
    refs = r.select(r.cells(), cuts="14", only="base,donor")
    assert {c["kind"] for c in refs} == {"base", "donor"}


def test_logits_cast_the_normed_state_to_the_head_dtype():
    # Round 24's smoke died here: a float32 norm output met a bf16 head on the GPU.
    import torch.nn as nn

    class Holder:
        pass

    head = nn.Linear(8, 5, bias=False).to(torch.bfloat16)
    top = Holder()
    top.head = head
    top.has_head_mask = False
    from wsp.tops import TopModule

    out = TopModule.logits(top, torch.randn(2, 3, 8, dtype=torch.float32))
    assert out.dtype == torch.bfloat16
    assert out.shape == (2, 3, 5)
