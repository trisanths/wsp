import json

import pytest
import torch

transformers = pytest.importorskip("transformers")

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
from transformers import PreTrainedTokenizerFast

from tests.test_align import WORDS, tiny_tokenizer
from tests.test_tops import cycle_texts, reference_generate, tiny_qwen2, with_sinks
from wsp import tops_vocab
from wsp.tops import SkillTops

BASE_EXTRA = ["catnap", "hatbox"]
DONOR_EXTRA = ["nap", "box", "zebra"]
BASE_ROWS = 300
DONOR_ROWS = 128


def base_tokenizer(eos="</s>"):
    """Byte-level style words: every word carries the G-with-dot space marker."""
    vocab = {"<unk>": 0, eos: 1, "<pad>": 2}
    for word in WORDS + BASE_EXTRA:
        vocab["Ġ" + word] = len(vocab)
    model = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    model.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
    return PreTrainedTokenizerFast(tokenizer_object=model, unk_token="<unk>", eos_token=eos, pad_token="<pad>")


def donor_tokenizer(eos="</s>"):
    """SentencePiece style BPE: the lower-eighth-block marker, trained on a small word list."""
    model = Tokenizer(models.BPE(unk_token="<unk>"))
    model.pre_tokenizer = pre_tokenizers.Metaspace(replacement="▁", prepend_scheme="always")
    model.decoder = decoders.Metaspace(replacement="▁", prepend_scheme="always")
    trainer = trainers.BpeTrainer(vocab_size=110, min_frequency=1, special_tokens=["<unk>", eos, "<pad>"])
    corpus = []
    for _ in range(30):
        corpus.append(" ".join(WORDS + DONOR_EXTRA + ["cat nap", "hat box"]))
    model.train_from_iterator(corpus, trainer)
    return PreTrainedTokenizerFast(tokenizer_object=model, unk_token="<unk>", eos_token=eos, pad_token="<pad>")


def cross_net(fallback="bias", dtype=None, eos=("</s>", "</s>"), layers=3, cut=2, seed=11, sharp=1.0):
    base = tiny_qwen2(0, vocab=BASE_ROWS)
    net = SkillTops(base, base_tokenizer(eos[0]))
    donor = tiny_qwen2(seed, hidden=32, layers=layers, vocab=DONOR_ROWS, intermediate=64, theta=10000.0)
    with torch.no_grad():
        donor.lm_head.weight.mul_(sharp)
    top = net.add_top("x", donor, cut, stitch_kind="mlp", donor_tokenizer=donor_tokenizer(eos[1]),
                      cross_tokenizer=True, bridge_options={"fallback": fallback})
    if dtype is not None:
        net = net.to(dtype)
        donor = donor.to(dtype)
    return net, base, donor, top


def sample_texts(n, seed, extra=True):
    import random

    rng = random.Random(seed)
    words = WORDS + (BASE_EXTRA if extra else [])
    out = []
    for _ in range(n):
        length = rng.randint(4, 9)
        out.append(" ".join(rng.choice(words) for _ in range(length)))
    return out


def encode_batch(net, texts):
    return net.encode(texts)


# ---------------------------------------------------------------- token bytes


def test_the_three_space_markers_give_the_same_bytes():
    assert tops_vocab.token_bytes("Ġthe", "bytelevel") == b" the"
    assert tops_vocab.token_bytes("▁the", "sentencepiece") == b" the"
    assert tops_vocab.token_bytes(" the", "plain") == b" the"
    assert tops_vocab.token_bytes("the", "plain") == b"the"
    assert tops_vocab.token_bytes("<0x41>", "sentencepiece") == b"A"
    # A byte-level newline is the "C-dot" character, a multi-byte letter decodes through the table.
    assert tops_vocab.token_bytes("Ċ", "bytelevel") == b"\n"
    assert tops_vocab.token_bytes("Ã©", "bytelevel") == "é".encode("utf-8")
    assert tops_vocab.vocab_kind(["Ġa", "Ġb", "Ġc", "d"]) == "bytelevel"
    assert tops_vocab.vocab_kind(["▁a", "▁b", "▁c", "d"]) == "sentencepiece"
    assert tops_vocab.vocab_kind(["a", "b"]) == "plain"


# ---------------------------------------------------------------- the bridge


def test_exact_bridge_maps_the_planted_shared_tokens():
    net, _, _, top = cross_net()
    bridge = top.bridge
    base_vocab = net.tokenizer.get_vocab()
    donor_vocab = top.donor_tokenizer.get_vocab()
    for word in WORDS:
        b = base_vocab["Ġ" + word]
        d = donor_vocab["▁" + word]
        assert int(bridge.kind[b]) == 1
        assert int(bridge.gather[b]) == d
    # The tokenizers use different ids for the same words, so the map is not the identity.
    assert not bridge.identity
    moved = 0
    for word in WORDS:
        if base_vocab["Ġ" + word] != donor_vocab["▁" + word]:
            moved += 1
    assert moved > 0
    assert bridge.report["base_kind"] == "bytelevel" and bridge.report["donor_kind"] == "sentencepiece"
    # The special tokens match by their strings.
    for name in ("<unk>", "</s>", "<pad>"):
        assert int(bridge.gather[base_vocab[name]]) == donor_vocab[name]


def test_logits_at_shared_tokens_equal_the_donor_head_at_the_matched_ids():
    net, _, donor, top = cross_net()
    ids, mask = encode_batch(net, sample_texts(3, 0, extra=False))
    with torch.no_grad():
        logits = net(ids, attention_mask=mask, top="x").logits
        normed = top.run(net.run_trunk(ids, mask, tops_positions(mask), top.cut_host, None, 0), mask,
                         tops_positions(mask), None, 0, None)
        donor_logits = top.head(normed)
    assert logits.shape[-1] == BASE_ROWS
    shared_base = top.bridge.shared_base
    shared_donor = top.bridge.shared_donor
    assert shared_base.numel() == 15
    assert torch.equal(logits[..., shared_base], donor_logits[..., shared_donor])


def tops_positions(mask):
    from wsp.tops import positions_of

    return positions_of(mask)


def test_fallback_tokens_take_the_first_donor_token_and_the_rest_are_masked():
    net, _, _, top = cross_net()
    bridge = top.bridge
    base_vocab = net.tokenizer.get_vocab()
    donor_vocab = top.donor_tokenizer.get_vocab()
    catnap = base_vocab["Ġcatnap"]
    hatbox = base_vocab["Ġhatbox"]
    assert int(bridge.kind[catnap]) == 2 and int(bridge.gather[catnap]) == donor_vocab["▁cat"]
    assert int(bridge.kind[hatbox]) == 2 and int(bridge.gather[hatbox]) == donor_vocab["▁hat"]
    # The bias starts at minus the log of 1 plus the fallback tokens that share the donor token.
    assert abs(float(bridge.bias[catnap]) + 0.6931472) < 1e-6
    # Ids of the head that no base token uses are masked to the dtype minimum.
    ids, mask = encode_batch(net, sample_texts(2, 1))
    with torch.no_grad():
        logits = net(ids, attention_mask=mask, top="x").logits
    low = torch.finfo(logits.dtype).min
    assert bool((logits[..., len(net.tokenizer.get_vocab()):] == low).all())
    assert bool((logits[..., catnap] > low).all())
    # A base token with no way into the donor is masked: partial bytes with no donor byte token.
    assert bridge.report["masked"] == BASE_ROWS - 17


def test_chat_end_tokens_map_through_the_special_pairs():
    net, _, _, top = cross_net(eos=("<|eot_id|>", "<|im_end|>"))
    base_vocab = net.tokenizer.get_vocab()
    donor_vocab = top.donor_tokenizer.get_vocab()
    b = base_vocab["<|eot_id|>"]
    assert int(top.bridge.kind[b]) == 1
    assert int(top.bridge.gather[b]) == donor_vocab["<|im_end|>"]
    assert ("<|eot_id|>", "<|im_end|>") in [tuple(x) for x in top.bridge.report["mapped_special"]]


def test_a_same_tokenizer_donor_gives_the_identity_bridge_and_the_existing_logits():
    for dtype in (torch.float32, torch.bfloat16):
        base = tiny_qwen2(0)
        donor = tiny_qwen2(1, hidden=32, layers=3, intermediate=64)
        old = SkillTops(base, tiny_tokenizer())
        old_top = old.add_top("d", donor, 2, donor_tokenizer=tiny_tokenizer())
        new = SkillTops(base, tiny_tokenizer())
        top = new.add_top("d", donor, 2, donor_tokenizer=tiny_tokenizer(), cross_tokenizer=True)
        top.stitch.load_state_dict(old_top.stitch.state_dict())
        assert top.bridge.identity
        assert torch.equal(top.bridge.gather, torch.arange(300))
        assert bool((top.bridge.kind == 1).all())
        old = old.to(dtype)
        new = new.to(dtype)
        tok = tiny_tokenizer()
        tok.padding_side = "left"
        enc = tok(cycle_texts(3, 2), return_tensors="pt", padding=True)
        with torch.no_grad():
            want = old(enc["input_ids"], attention_mask=enc["attention_mask"], top="d").logits
            got = new(enc["input_ids"], attention_mask=enc["attention_mask"], top="d").logits
        tolerance = 1e-5 if dtype == torch.float32 else 0.05
        assert (want.float() - got.float()).abs().max().item() < tolerance


def test_an_identity_bridge_masks_the_base_tokens_the_donor_lacks_like_the_existing_path():
    fewer = tiny_tokenizer()
    vocab = {"<unk>": 0, "</s>": 1, "<pad>": 2}
    for word in WORDS[:-2]:
        vocab[word] = len(vocab)
    model = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    model.pre_tokenizer = pre_tokenizers.Whitespace()
    fewer = PreTrainedTokenizerFast(tokenizer_object=model, unk_token="<unk>", eos_token="</s>", pad_token="<pad>")
    bridge = tops_vocab.build_bridge(tiny_tokenizer(), fewer, 300, 300, d_donor=32)
    assert bridge.identity and bridge.report["masked"] == 2


def test_coverage_numbers_on_the_planted_vocabularies():
    net, _, _, top = cross_net()
    report = tops_vocab.coverage_report(top.bridge, net.tokenizer, top.donor_tokenizer,
                                        ["the cat sat catnap"])
    assert report["exact"] == 15 and report["fallback"] == 2
    assert abs(report["exact_share"] - 15 / 17) < 1e-9
    assert abs(report["fallback_share"] - 2 / 17) < 1e-9
    text = report["text"]
    assert text["base_tokens"] == 4 and abs(text["base_exact"] - 0.75) < 1e-9
    assert abs(text["base_fallback"] - 0.25) < 1e-9
    assert text["donor_tokens"] == 5 and abs(text["donor_shared"] - 0.8) < 1e-9
    donor = tiny_qwen2(11, hidden=32, layers=3, vocab=DONOR_ROWS, intermediate=64, theta=10000.0)
    mass = tops_vocab.shared_mass_of_model(donor, top.donor_tokenizer, top.bridge, ["the cat sat catnap"])
    assert 0.0 < mass < 1.0
    # With every donor id shared the mass is one.
    top.bridge.shared_donor = torch.arange(DONOR_ROWS)
    assert abs(tops_vocab.shared_mass_of_model(donor, top.donor_tokenizer, top.bridge, ["the cat"]) - 1.0) < 1e-4


# ---------------------------------------------------------------- forward and generation


@pytest.mark.parametrize("fallback", ["bias", "linear"])
def test_cached_generation_equals_uncached_batched_left_padded_two_calls(fallback):
    net, _, _, _ = cross_net(fallback, torch.float64)
    with_sinks(net)
    if fallback == "linear":
        with torch.no_grad():
            torch.manual_seed(3)
            net.tops["x"].bridge.map.weight.normal_(0.0, 0.05)
    for call in range(2):
        texts = [sample_texts(1, 30 + call)[0], "the cat", sample_texts(1, 50 + call)[0]]
        ids, mask = encode_batch(net, texts)
        assert int(mask.sum()) < mask.numel()
        made = net.generate(ids, attention_mask=mask, max_new_tokens=8, top="x", pad_token_id=0)
        want = reference_generate(net, ids, mask, 8, top="x")
        assert torch.equal(made[:, ids.shape[1]:], want)
        # The new tokens are base tokens: valid ids, never a masked one.
        assert int(made[:, ids.shape[1]:].max()) < len(net.tokenizer.get_vocab())


def test_bf16_forward_and_generation_run_with_every_module_in_bf16():
    for fallback in ("bias", "linear"):
        net, _, donor, top = cross_net(fallback, torch.bfloat16)
        with_sinks(net)
        ids, mask = encode_batch(net, sample_texts(3, 5))
        with torch.no_grad():
            logits = net(ids, attention_mask=mask, top="x").logits
            first = net.generate(ids, attention_mask=mask, max_new_tokens=5, top="x", pad_token_id=0)
            second = net.generate(ids, attention_mask=mask, max_new_tokens=5, top="x", pad_token_id=0)
        assert logits.dtype == torch.bfloat16 and logits.shape[-1] == BASE_ROWS
        assert bool(torch.isfinite(logits.float().clamp(min=-1e30)).all())
        assert torch.equal(first, second)
        assert int(first[:, ids.shape[1]:].max()) < len(net.tokenizer.get_vocab())
        stats = net.tops["x"].bridge
        assert stats.bias.dtype == torch.bfloat16


def test_routing_and_a_mixed_batch_keep_working_with_a_cross_tokenizer_top():
    net, base, _, _ = cross_net()
    with_sinks(net)
    net.fit_router({"base": sample_texts(30, 1, extra=False), "x": sample_texts(30, 2)}, lam=1e-3)
    ids, mask = encode_batch(net, sample_texts(4, 8))
    with torch.no_grad():
        out = net(ids, attention_mask=mask)
    assert out.logits.shape == (4, ids.shape[1], BASE_ROWS)


# ---------------------------------------------------------------- alignment and the fit objective


def test_aligned_positions_pair_tokens_that_end_on_the_same_answer_character():
    base = base_tokenizer()
    donor = donor_tokenizer()
    prompt_b = "the cat"
    prompt_d = "dog ran to"
    answer = " sat on a catnap"
    b_ids, b_mask, b_off = tops_vocab.encode_offsets(base, [prompt_b + answer], None, "left")
    d_ids, d_mask, d_off = tops_vocab.encode_offsets(donor, [prompt_d + answer], None, "right")
    br, bi, dr, di = tops_vocab.aligned_positions(b_mask, b_off, [len(prompt_b)], d_mask, d_off, [len(prompt_d)])
    pairs = list(zip(bi, di))
    base_tokens = tops_vocab.real_tokens(b_mask[0], b_off[0])
    donor_tokens = tops_vocab.real_tokens(d_mask[0], d_off[0])
    ends = []
    for i, j in pairs:
        end_b = [t for t in base_tokens if t[0] == i][0][2] - len(prompt_b)
        end_d = [t for t in donor_tokens if t[0] == j][0][2] - len(prompt_d)
        assert end_b == end_d
        ends.append(end_b)
    # 0 is the end of the prompt (the position that predicts the first answer token), then sat, on, a.
    assert ends[:4] == [0, 4, 7, 9]
    # The base token "catnap" ends where the donor's "nap" ends; the donor's "cat" has no base partner.
    assert max(ends) == len(answer) and 13 not in ends
    weight = tops_vocab.answer_weights(b_mask, b_off, [len(prompt_b)], False)
    assert int(weight.sum()) == 4


def fit_inputs(n=24):
    prompts = sample_texts(n, 3, extra=False)
    answers = []
    for index in range(n):
        answers.append(" " + " ".join(sample_texts(1, 100 + index)[0].split()[:5]))
    return prompts, answers


def test_the_cross_fit_lowers_the_shared_kl_and_the_answer_loss():
    net, _, donor, top = cross_net(sharp=40.0)
    prompts = sample_texts(30, 3, extra=False)
    answers = tops_vocab.donor_answers(donor, top.donor_tokenizer, prompts, max_new_tokens=6, batch_size=10)
    report = net.fit_stitch("x", prompts[:24], steps=80, lr=3e-3, batch_size=6, seq_len=40, seed=0,
                            eval_texts=prompts[24:], donor_model=donor, max_penalty=99.0, answers=answers[:24],
                            eval_answers=answers[24:], donor_prompts=prompts[:24])
    assert report["cross_tokenizer"] is True
    before = report["before"]
    after = report["after"]
    assert after["kl_shared"] < before["kl_shared"]
    assert after["top_loss"] < before["top_loss"]
    assert after["penalty"] < before["penalty"]
    assert report["steps_run"] == 80 and not report["diverged"]
    # The bridge's bias is in the trainable set.
    assert report["trainable_params"] >= BASE_ROWS
    # The same call on a same-tokenizer top rejects the cross-only arguments.
    plain = SkillTops(tiny_qwen2(0), tiny_tokenizer())
    plain.add_top("d", tiny_qwen2(1, hidden=32, layers=3, intermediate=64), 2, donor_tokenizer=tiny_tokenizer())
    with pytest.raises(TypeError):
        plain.fit_stitch("d", cycle_texts(20, 0), steps=1, answers=["a"])


def test_the_linear_fallback_is_fit_by_ridge_and_trains():
    net, _, donor, top = cross_net("linear")
    prompts, answers = fit_inputs(30)
    report = net.fit_stitch("x", prompts[:24], steps=10, lr=1e-3, batch_size=6, seq_len=40,
                            eval_texts=prompts[24:], donor_model=donor, max_penalty=99.0, answers=answers[:24],
                            eval_answers=answers[24:])
    assert report["linear_init"]["pairs"] > 20
    assert float(top.bridge.map.weight.abs().sum()) > 0
    assert report["trainable_params"] >= BASE_ROWS + 32 * 48


def test_the_donor_writes_answers_and_the_fit_runs_on_them():
    net, _, donor, top = cross_net()
    prompts = sample_texts(14, 4, extra=False)
    answers = tops_vocab.donor_answers(donor, top.donor_tokenizer, prompts, max_new_tokens=6, batch_size=4)
    assert len(answers) == 14
    assert len([a for a in answers if len(a.strip()) > 0]) >= 10
    report = net.fit_stitch("x", prompts, steps=3, lr=1e-3, batch_size=4, seq_len=40, donor_model=donor,
                            max_penalty=99.0, answer_tokens=6)
    assert report["train_texts"] >= 8 and "penalty" in report


def test_the_cross_penalty_of_the_donor_through_a_perfect_path_is_near_zero_in_bf16_too():
    # Base and donor are the same weights with the same words, so the stitched top equals the donor.
    net, _, donor, top = cross_net(dtype=torch.bfloat16)
    prompts, answers = fit_inputs(6)
    items = tops_vocab.make_items(prompts, None, answers)
    found = tops_vocab.cross_penalty(net, top, items, donor, 40)
    assert found["tokens"] > 0 and found["base_tokens"] > 0
    assert found["penalty"] == found["penalty"]
    assert found["chars"] > 0


# ---------------------------------------------------------------- save and load


@pytest.mark.parametrize("fallback", ["bias", "linear"])
def test_save_and_load_round_trip_the_bridge(tmp_path, fallback):
    donor = tiny_qwen2(11, hidden=32, layers=3, vocab=DONOR_ROWS, intermediate=64, theta=10000.0)
    path = str(tmp_path / "donor")
    donor.save_pretrained(path)
    donor_tokenizer().save_pretrained(path)
    base = tiny_qwen2(0, vocab=BASE_ROWS)
    net = SkillTops(base, base_tokenizer(), base_id="tiny-base")
    top = net.add_top("x", path, 2, cross_tokenizer=True, bridge_options={"fallback": fallback})
    with_sinks(net)
    torch.manual_seed(5)
    with torch.no_grad():
        top.bridge.bias.add_(0.3 * torch.randn_like(top.bridge.bias))
        if fallback == "linear":
            top.bridge.map.weight.normal_(0.0, 0.05)
            top.bridge.shift.fill_(0.7)
            top.bridge.scale.fill_(1.3)
    ids, mask = encode_batch(net, sample_texts(3, 9))
    net.save(str(tmp_path / "net"))
    meta = json.load(open(tmp_path / "net" / "tops.json"))
    assert meta["tops"][0]["cross_tokenizer"] is True
    assert meta["tops"][0]["bridge_options"]["fallback"] == fallback
    loaded = SkillTops.load(str(tmp_path / "net"), base_model=base, tokenizer=base_tokenizer())
    assert loaded.tops["x"].bridge is not None and loaded.tops["x"].bridge.fallback == fallback
    with torch.no_grad():
        want = net(ids, attention_mask=mask, top="x").logits
        got = loaded(ids, attention_mask=mask, top="x").logits
    assert (want - got).abs().max().item() < 1e-6
    counts = loaded.count_params()["tops"]["x"]
    assert counts["head"] == DONOR_ROWS * 32
    assert counts["bridge"] >= BASE_ROWS
    assert counts["stored"] == counts["stitch"] + counts["layers"] + counts["lora"] + counts["norm"] \
        + counts["head"] + counts["sink"] + counts["bridge"]


def test_sizes_from_configs_use_the_donor_rows_for_a_cross_tokenizer_head():
    from wsp.tops import sizes_from_configs

    base = tiny_qwen2(0, vocab=BASE_ROWS).config
    donor = tiny_qwen2(1, hidden=32, layers=3, vocab=DONOR_ROWS, intermediate=64).config
    same = sizes_from_configs(base, [{"name": "x", "donor_config": donor, "cut_host": 2}])
    cross = sizes_from_configs(base, [{"name": "x", "donor_config": donor, "cut_host": 2,
                                       "cross_tokenizer": True}])
    assert same["tops"]["x"]["head"] == BASE_ROWS * 32
    assert cross["tops"]["x"]["head"] == DONOR_ROWS * 32 + BASE_ROWS


# ---------------------------------------------------------------- fresh process


def test_the_module_imports_in_a_fresh_process():
    import subprocess
    import sys

    code = "import wsp.tops_vocab as v; assert v.vocab_kind(['a']) == 'plain'; print('ok')"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=".")
    assert out.returncode == 0 and "ok" in out.stdout, out.stderr[-500:]


# ---------------------------------------------------------------- the round script, pair T4


def test_round24_pair_t4_is_optional_and_selected_only_by_name():
    pytest.importorskip("modal")
    import modal_round24 as r

    assert "T4" not in r.PAIRS and r.OPTIONAL_PAIRS["T4"]["cross_tokenizer"] is True
    assert {c["pair"] for c in r.cells()} == {"T1", "T2", "T3"}
    extra = r.optional_cells()
    names = [r.cell_name(c) for c in extra]
    assert len(names) == len(set(names))
    assert {(c["kind"], c["cut"], c["seed"]) for c in extra if c["kind"] == "forced"} == {
        ("forced", 14, 0), ("forced", 14, 1), ("forced", 21, 0)}
    assert {c["kind"] for c in extra} == {"base", "donor", "forced", "scrambled"}
    assert all(c["skills"] == ["gsm8k"] for c in extra)
    assert {(c["cut"], c["seed"]) for c in extra if c["kind"] == "scrambled"} == {(14, 0), (14, 1)}
    chosen = r.select(r.cells() + extra, pairs="T4", only="forced", cuts="1/2", seeds="0")
    assert [r.cell_name(c) for c in chosen] == ["T4_forced_math_c14_s0"]
    assert r.select(r.cells(), pairs="T4") == []
    smoke = r.smoke_cell("T4")
    assert smoke["pair"] == "T4" and smoke["cut"] == 14 and smoke["steps"] == 200 and smoke["skills"] == ["gsm8k"]
    assert r.smoke_cell()["pair"] == "T1"
    plan = r.plan(extra)
    assert set(plan) == {"T4"}


def test_round24_t4_summary_prints_and_stays_out_of_the_go_rule(capsys):
    pytest.importorskip("modal")
    import modal_round24 as r

    rows = [{"pair": "T4", "kind": "base", "top": None, "cut": None, "seed": 0, "scores": {"gsm8k": 0.7}},
            {"pair": "T4", "kind": "donor", "top": "math", "cut": None, "seed": 0, "scores": {"gsm8k": 0.8}},
            {"pair": "T4", "kind": "forced", "top": "math", "cut": 14, "seed": 0, "scores": {"gsm8k": 0.6}},
            {"pair": "T4", "kind": "forced", "top": "math", "cut": 14, "seed": 1, "scores": {"gsm8k": 0.7}},
            {"pair": "T4", "kind": "scrambled", "top": "math", "cut": 14, "seed": 0, "scores": {"gsm8k": 0.05}}]
    out = r.summarize_t4(rows)
    assert abs(out[("forced", 14)]["gsm8k"] - 0.65) < 1e-9 and out[("forced", 14)]["seeds"] == 2
    assert r.summarize(rows) == {}
    assert "exploratory" in capsys.readouterr().out


def test_a_bos_written_in_the_text_is_dropped_and_offsets_stay_in_text_coordinates():
    vocab = {"<unk>": 0, "</s>": 1, "<pad>": 2, "<s>": 3}
    for word in WORDS:
        vocab["Ġ" + word] = len(vocab)
    model = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    model.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
    tok = PreTrainedTokenizerFast(tokenizer_object=model, unk_token="<unk>", eos_token="</s>", pad_token="<pad>",
                                  bos_token="<s>")
    assert tops_vocab.strip_bos(tok, "<s>the cat") == "the cat"
    assert tops_vocab.strip_bos(tok, "the cat") == "the cat"
    ids, mask, offsets = tops_vocab.encode_offsets(tok, ["<s>the cat", "the cat"], None, "left")
    assert ids[0].tolist() == ids[1].tolist()
    assert offsets[0].tolist() == [[3, 6], [6, 10]]
    assert offsets[1].tolist() == [[0, 3], [3, 7]]
