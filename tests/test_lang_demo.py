"""wsp.lang_demo: splits, recasts, filters, long-prompt scoring, demo outputs. CPU only, no network."""

import csv
import random

import pytest
import torch

transformers = pytest.importorskip("transformers")

from tests.test_xfam import HOST_WORDS, items, make_tok, tiny_host  # noqa: E402
from wsp import lang_demo as ld  # noqa: E402
from wsp import xfam  # noqa: E402


def test_split_indexes_are_disjoint_and_complete():
    s = ld.split_indexes()
    sets = {k: set(v) for k, v in s.items()}
    assert len(s["held"]) == ld.HELD_N and len(s["dev"]) == ld.DEV_N
    assert len(sets["en_dev"]) == 1000 and len(sets["en_held"]) == 2000
    names = list(sets)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            assert not (sets[a] & sets[b]), (a, b)
    assert set().union(*sets.values()) == set(range(ld.N_MMLU_TEST))


def test_split_is_stable_and_matches_xfam_order():
    assert ld.split_indexes() == ld.split_indexes()
    order = list(range(ld.N_MMLU_TEST))
    random.Random(0).shuffle(order)
    assert ld.split_indexes()["en_held"] == order[1000:3000]


def test_mmmlu_read_and_align(tmp_path):
    path = tmp_path / "m.csv"
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["", "Question", "A", "B", "C", "D", "Answer", "Subject"])
        w.writerow(["0", "q0", "1", "2", "3", "4", "B", "s"])
        w.writerow(["1", "q1\nline", "a", "b", "c", "d", "D", "s"])
        w.writerow(["2", "q2", "a", "b", "c", "d", "", "s"])
    rows = ld.read_mmmlu(str(path))
    assert rows[2] is None and rows[1]["answer"] == 3 and "\n" in rows[1]["question"]
    english = [{"subject": "s", "answer": 1}, {"subject": "s", "answer": 0}, {"subject": "s", "answer": 0}]
    assert ld.check_alignment(rows, english) == 0.5
    got = ld.mmmlu_items(rows, [0, 1, 2], english, require_same_answer=True)
    assert len(got) == 1 and got[0][1] == 1 and got[0][0].endswith("Answer:")


def test_xnli_recast_has_three_options_and_balanced_letters():
    rows = [{"premise": "p%d" % i, "hypothesis": "h%d" % i, "label": i % 3} for i in range(600)]
    out = ld.xnli_items(rows, 300, seed=1)
    assert len(out) == 300
    counts = [sum(1 for _, g in out if g == k) for k in range(3)]
    assert all(c > 60 for c in counts)
    by_premise = {"p%d" % i: i % 3 for i in range(600)}
    for prompt, gold in out:
        assert "D." not in prompt and "C." in prompt
        lines = prompt.split("\n")
        label = by_premise[lines[0].split("Premise: ", 1)[1]]
        assert lines[3 + gold].endswith(ld.XNLI_OPTIONS[label])


def test_restrict_soft():
    out = ld.restrict_soft([0.4, 0.2, 0.1, 0.3], 3)
    assert out[3] == 0.0 and abs(sum(out) - 1) < 1e-3
    assert ld.restrict_soft([0, 0, 0, 1.0], 3)[3] == 0.0


def test_filter_items_counts():
    tok = make_tok(HOST_WORDS)
    base = items(8)
    its = base + [base[0]] + [["Question " + "cat " * 40 + "\nAnswer", 1]]
    kept, c = ld.filter_items(its, {base[1][0]}, tok, 30)
    assert c["banned"] == 1 and c["duplicate"] == 1 and c["too_long"] == 1 and c["kept"] == len(kept) == 7


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_score_long_matches_xfam_and_runs_in_dtype(dtype):
    tok = make_tok(HOST_WORDS)
    model = tiny_host(dtype)
    its = items(20)
    r = ld.score_long(model, tok, its, budget=64, return_preds=True)
    assert r["n"] == 20 and r["truncated"] == 0 and len(r["preds"]) == 20
    if dtype == torch.float32:
        a = xfam.score_letters(model, tok, its, bs=4)
        assert abs(r["acc"] - a) < 1e-9
        r2 = ld.score_long(model, tok, its, budget=100000)
        assert abs(r2["acc"] - r["acc"]) < 1e-9


def test_score_long_cuts_from_the_left_and_counts():
    tok = make_tok(HOST_WORDS)
    model = tiny_host()
    long_item = ["the cat " * 30 + "Question cat\nA cat\nB dog\nC hat\nD red\nAnswer", 2]
    r = ld.score_long(model, tok, [long_item] + items(3), max_tokens=40)
    assert r["truncated"] == 1 and r["max_len"] > 40 and r["n"] == 4


def test_demo_outputs_greedy_two_calls_bf16():
    tok = make_tok(HOST_WORDS)
    model = tiny_host(torch.bfloat16)
    prompts = {"a": items(1)[0][0], "b": items(2)[1][0]}
    first = ld.demo_outputs(model, tok, prompts, new_tokens=5)
    second = ld.demo_outputs(model, tok, prompts, new_tokens=5)
    assert first == second and [r["prompt_id"] for r in first] == ["a", "b"]
    assert all(r["letter"] in "ABCD" for r in first)


def test_demo_prompts_cover_six_languages_and_are_valid():
    assert set(ld.DEMO_PROMPTS) == {"en", "ar", "ja", "zh", "ko", "hi"}
    for p in ld.DEMO_PROMPTS.values():
        assert p.endswith("Answer:") and p.count("\nC. ") == 1


def test_wiki_chunks():
    texts = ["short", "word " * 400, "ab " * 300]
    out = ld.wiki_chunks(texts, "ja", 5, seed=0)
    assert len(out) == 2 and all(len(x) <= ld.WIKI_CHARS["ja"] for x in out)
