import copy

import pytest
import torch
import transformers

from wsp import probe


class TinyTokenizer:
    """Whitespace tokenizer over a fixed small vocabulary, enough for the probes."""

    pad_token = "<pad>"
    eos_token = "<pad>"
    pad_token_id = 0
    padding_side = "right"
    chat_template = None

    def __call__(self, texts, return_tensors="pt", padding=True, truncation=True, max_length=64,
                 add_special_tokens=True):
        rows = []
        for text in texts:
            ids = [1 + (sum(ord(c) for c in word) % 30) for word in text.split()][:max_length]
            rows.append(ids)
        width = max(len(row) for row in rows)
        input_ids = torch.zeros(len(rows), width, dtype=torch.long)
        mask = torch.zeros(len(rows), width, dtype=torch.long)
        for index, row in enumerate(rows):
            input_ids[index, :len(row)] = torch.tensor(row)
            mask[index, :len(row)] = 1
        return {"input_ids": input_ids, "attention_mask": mask}


def tiny(seed):
    torch.manual_seed(seed)
    config = transformers.Qwen2Config(vocab_size=32, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=64)
    return transformers.AutoModelForCausalLM.from_config(config).eval()


TEXTS = ["a b c d e f g", "the quick brown fox jumps", "one two three four five six seven eight"]


def blend(base, donor, weight):
    """A model whose weights sit between the base and the donor."""
    out = copy.deepcopy(base)
    donor_state = donor.state_dict()
    state = out.state_dict()
    for name in state:
        state[name] = (1.0 - weight) * state[name] + weight * donor_state[name]
    out.load_state_dict(state)
    return out.eval()


def test_kl_is_zero_for_the_same_model_and_positive_otherwise():
    tokenizer = TinyTokenizer()
    base = tiny(0)
    donor = tiny(1)
    same = probe.logit_kl(base, base, tokenizer, TEXTS)
    assert same["kl"] == pytest.approx(0.0, abs=1e-6)
    assert same["agree"] == pytest.approx(1.0)
    other = probe.logit_kl(donor, base, tokenizer, TEXTS)
    assert other["kl"] > 0.0
    # The first token of each sequence is left out.
    assert other["tokens"] == sum(len(text.split()) - 1 for text in TEXTS)


def test_transfer_is_zero_at_the_base_and_one_at_the_donor():
    tokenizer = TinyTokenizer()
    base = tiny(0)
    donor = tiny(1)
    at_base = probe.transfer(base, base, donor, tokenizer, TEXTS)
    assert at_base["transfer"] == pytest.approx(0.0, abs=1e-6)
    at_donor = probe.transfer(donor, base, donor, tokenizer, TEXTS)
    assert at_donor["transfer"] == pytest.approx(1.0, abs=1e-6)
    part = probe.transfer(blend(base, donor, 0.9), base, donor, tokenizer, TEXTS)
    assert 0.0 < part["transfer"] <= 1.0


def test_gap_closed_edges():
    assert probe.gap_closed(2.0, 1.0, 0.0) == pytest.approx(0.5)
    assert probe.gap_closed(2.0, 3.0, 0.0) == pytest.approx(-0.5)
    assert probe.gap_closed(1.0, 0.5, 1.0) is None


def test_drift_and_report_verdicts():
    tokenizer = TinyTokenizer()
    base = tiny(0)
    donor = tiny(1)
    assert probe.drift(base, base, tokenizer, TEXTS)["base_kl"] == pytest.approx(0.0, abs=1e-6)
    written = blend(base, donor, 0.9)
    random_written = blend(base, tiny(7), 0.9)
    out = probe.report(written, base, tokenizer, {"math": {"donor": donor, "texts": TEXTS}}, TEXTS,
                       random_written=random_written)
    row = out["skills"]["math"]
    assert row["content_gain"] == pytest.approx(row["transfer"] - row["random_transfer"])
    assert out["verdict"]["math"] == "contributes"
    assert out["drift"]["base_kl"] > 0.0
    same = probe.report(base, base, tokenizer, {"math": {"donor": donor, "texts": TEXTS}}, TEXTS)
    assert same["verdict"]["math"] == "no transfer"


def test_verdict_rules():
    assert probe.verdict({"transfer": None}) == "no gap"
    assert probe.verdict({"transfer": -0.2}) == "harmful"
    assert probe.verdict({"transfer": 0.3, "content_gain": 0.0}) == "no better than random"
    assert probe.verdict({"transfer": 0.3, "content_gain": 0.2}) == "contributes"
    assert probe.verdict({"transfer": 0.3}) == "contributes"


def test_text_nll_and_transfer_by_text():
    tokenizer = TinyTokenizer()
    base = tiny(0)
    donor = tiny(1)
    loss = probe.text_nll(base, tokenizer, TEXTS)
    assert loss["nll"] > 0.0
    assert loss["tokens"] == sum(len(text.split()) - 1 for text in TEXTS)
    same = probe.transfer_by_text(base, base, tokenizer, TEXTS)
    assert same["transfer"] == pytest.approx(0.0, abs=1e-6)
    floor = probe.text_nll(donor, tokenizer, TEXTS)["nll"]
    full = probe.transfer_by_text(donor, base, tokenizer, TEXTS, donor_self_nll=floor)
    if full["transfer"] is not None:
        assert full["transfer"] == pytest.approx(1.0, abs=1e-5)


def test_content_z_needs_two_controls_and_scales_by_their_spread():
    assert probe.content_z(0.5, [0.1]) is None
    assert probe.content_z(None, [0.1, 0.2]) is None
    stats = probe.content_z(0.5, [0.1, 0.2, 0.3])
    assert stats["mean"] == pytest.approx(0.2)
    assert stats["std"] == pytest.approx(0.1)
    assert stats["z"] == pytest.approx(3.0)
    assert stats["controls"] == 3
    assert probe.content_z(0.5, [0.2, 0.2])["z"] is None


def test_verdict_uses_the_z_score_when_present():
    noisy = {"transfer": 0.3, "content_gain": 0.2, "content_z": {"z": 0.8, "mean": 0.1, "std": 0.25, "controls": 3}}
    assert probe.verdict(noisy) == "no better than random"
    clear = {"transfer": 0.3, "content_z": {"z": 3.5, "mean": 0.05, "std": 0.07, "controls": 3}}
    assert probe.verdict(clear) == "contributes"
