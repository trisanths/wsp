import random

import pytest
import torch

from wsp import wi


def test_kd_zero_when_equal():
    t = torch.randn(2, 5, 11)
    mask = torch.ones(2, 5)
    assert float(wi.kd_logits_loss(t.clone(), t, mask)) < 1e-6


def test_kd_matches_direct_and_grads_bf16():
    s = torch.randn(2, 6, 13, requires_grad=True)
    t = torch.randn(2, 6, 15)
    mask = torch.tensor([[1, 1, 1, 0, 0, 0], [1, 1, 1, 1, 1, 1]])
    loss = wi.kd_logits_loss(s, t, mask, chunk=3)
    keep = mask.bool()
    direct = wi._kl_rows(t[..., :13][keep], s[keep]).mean()
    assert torch.allclose(loss, direct, atol=1e-5)
    loss.backward()
    assert s.grad.abs().sum() > 0
    sb = torch.randn(2, 6, 13).bfloat16().requires_grad_(True)
    lb = wi.kd_logits_loss(sb, t.bfloat16(), mask, chunk=4)
    lb.backward()
    assert torch.isfinite(lb) and sb.grad.dtype == torch.bfloat16


def test_soft_letter_loss():
    a = torch.randn(3, 4)
    assert float(wi.soft_letter_loss(a, a)) < 1e-6
    assert float(wi.soft_letter_loss(a, torch.randn(3, 4))) > 0


def test_trivia_em():
    assert wi.trivia_em(" The Eiffel Tower.\nQuestion: x", ["eiffel tower"])
    assert not wi.trivia_em("Paris", ["London"])
    assert not wi.trivia_em("", [""])


def test_items():
    row = {"question": "q", "choices": {"label": ["A", "B", "C", "D"], "text": list("wxyz")}, "answerKey": "C"}
    p, g, n = wi.arc_item(row)
    assert g == 2 and n == 4 and p.endswith("Answer:")
    row["choices"]["label"] = ["1", "2", "3", "4"]
    row["answerKey"] = "2"
    assert wi.arc_item(row)[1] == 1
    row["answerKey"] = "9"
    assert wi.arc_item(row) is None
    s = {"question": "q", "distractor1": "a", "distractor2": "b", "distractor3": "c", "correct_answer": "ok"}
    p, g, n = wi.sciq_item(s, random.Random(0))
    assert p.split("\n")[1 + g][3:] == "ok"


def test_letter_logits_tiny():
    pytest.importorskip("transformers")
    from tests.test_align import tiny_tokenizer
    from tests.test_pgraft import tiny_host

    model = tiny_host()
    tok = tiny_tokenizer()
    ids = [5, 6, 7, 8, 9]
    out = wi.letter_logits(model, tok, ["a b c", "b c a b"], ids, batch_size=1)
    out2 = wi.letter_logits(model, tok, ["a b c", "b c a b"], ids, batch_size=2)
    assert out.shape == (2, 5) and torch.allclose(out, out2, atol=1e-3)
