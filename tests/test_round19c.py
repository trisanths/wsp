import importlib.util
import os
import sys

import pytest
import torch

from tests.test_slim import GENERAL, SKILL, logits, sample_ids, tiny_moe, tokenizer
from wsp import slim

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_round():
    if "modal_round19c" in sys.modules:
        return sys.modules["modal_round19c"]
    spec = importlib.util.spec_from_file_location("modal_round19c", os.path.join(ROOT, "modal_round19c.py"))
    module = importlib.util.module_from_spec(spec)
    sys.modules["modal_round19c"] = module
    spec.loader.exec_module(module)
    return module


def count(r, model=None, skill=None, rule="any", heal=None):
    total = 0
    for name, cell in r.CELLS.items():
        if model is not None and cell["model"] != model:
            continue
        if skill is not None and cell["skills"] != (skill,):
            continue
        if rule != "any" and cell["rule"] != rule:
            continue
        if heal is not None and (cell["heal"] > 0) != heal:
            continue
        total += 1
    return total


def test_cell_list_counts():
    r = load_round()
    assert len(r.CELLS) == 68
    assert count(r, "qwen") == 57
    assert count(r, "olmoe") == 11
    for skill in ["code", "math", "medicine"]:
        assert count(r, "qwen", skill, "saliency") == 13
        assert count(r, "qwen", skill, "saliency", heal=True) == 5
        assert count(r, "qwen", skill, "random") == 2
    assert count(r, "qwen", None, "random") == 6 + 6
    assert count(r, "olmoe", "code", "saliency") == 4
    assert count(r, "olmoe", "math", "saliency") == 3
    assert count(r, "olmoe", None, "random") == 3
    assert count(r, "olmoe", heal=True) == 0
    multi = [c for c in r.CELLS.values() if len(c["skills"]) == 2]
    assert len(multi) == 5
    for cell in multi:
        assert cell["skills"] == ("code", "math")
    for n in [64, 48, 40, 32, 24]:
        assert f"qwen_code_sal{n}" in r.CELLS
        assert f"qwen_medicine_sal{n}_heal500" in r.CELLS
    for mass in [90, 80, 70]:
        assert r.CELLS[f"qwen_math_mass{mass}"]["mass"] == mass / 100
        assert r.CELLS[f"qwen_math_mass{mass}"]["keep"] is None
    assert r.CELLS["olmoe_noop_all"]["keep"] == r.MODELS["olmoe"]["experts"]
    assert r.CELLS["qwen_full"]["rule"] is None
    for name, cell in r.CELLS.items():
        assert name.startswith(cell["model"] + "_")
        assert (cell["keep"] is None) != (cell["mass"] is None) or cell["rule"] is None


def test_smoke_selects_exactly_one_cell_and_ignores_other_flags():
    r = load_round()
    assert r.select_cells(smoke=True) == ["olmoe_code_sal32"]
    assert r.select_cells(smoke=True, models="qwen", skills="medicine", only="qwen_full", shard=3, nshards=5) == \
        ["olmoe_code_sal32"]
    cell = r.CELLS["olmoe_code_sal32"]
    assert (cell["model"], cell["skills"], cell["rule"], cell["keep"], cell["heal"]) == \
        ("olmoe", ("code",), "saliency", 32, 0)


def test_filters_and_shards():
    r = load_round()
    everything = r.select_cells()
    assert everything == list(r.CELLS)
    pieces = []
    for shard in range(3):
        pieces += r.select_cells(shard=shard, nshards=3)
    assert sorted(pieces) == sorted(everything)
    assert len(set(pieces)) == len(pieces)
    assert len(r.select_cells(models="olmoe")) == 11
    medicine = r.select_cells(models="qwen", skills="medicine")
    assert "qwen_full" in medicine and "qwen_random48_s0" in medicine
    assert "qwen_medicine_sal64" in medicine and "qwen_code_sal64" not in medicine
    assert not any("+" in name for name in medicine)
    both = r.select_cells(models="qwen", skills="code,math")
    assert "qwen_code+math_sal48" in both
    assert r.select_cells(only="qwen_full,olmoe_full") == ["qwen_full", "olmoe_full"]
    with pytest.raises(ValueError):
        r.select_cells(only="not_a_cell")
    with pytest.raises(ValueError):
        r.select_cells(skills="law")
    with pytest.raises(ValueError):
        r.select_cells(shard=2, nshards=2)


def test_chunks_cover_every_cell_once():
    r = load_round()
    chunks = r.make_chunks(r.select_cells())
    seen = []
    for key, names in chunks:
        assert 0 < len(names) <= r.CHUNK[key]
        for name in names:
            assert r.CELLS[name]["model"] == key
        seen += names
    assert sorted(seen) == sorted(r.CELLS)


def two_skill_stats(model, tok):
    code = slim.collect_moe_stats(model, tok, SKILL, 12)
    other = slim.collect_moe_stats(model, tok, GENERAL, 12)
    return {"code": code, "math": other}


def test_union_ranking_keeps_the_top_experts_of_the_summed_normalized_saliency():
    r = load_round()
    model = tiny_moe(experts=12, layers=2)
    tok = tokenizer()
    stats = two_skill_stats(model, tok)
    cell = r.spec("qwen", ["code", "math"], "saliency", keep=6)
    student, report, extra = r.slim_for_spec(model, cell, stats)
    combined = slim.combine_stats([stats["code"], stats["math"]])
    for layer in extra["kept"]:
        total = []
        for e in range(12):
            value = 0.0
            for skill in ["code", "math"]:
                sal = stats[skill]["layers"][layer]["saliency"]
                value += float(sal[e] / sal.sum())
            total.append(value)
        best = sorted(range(12), key=lambda e: (-total[e], e))
        threshold = sorted(total, reverse=True)[5]
        assert all(total[e] >= threshold - 1e-12 for e in extra["kept"][layer])
        assert len(extra["kept"][layer]) == 6
        assert abs(float(combined["layers"][layer]["saliency"].sum()) - 2.0) < 1e-9
    assert set(extra["kept_share"]) == {"code", "math"}
    assert extra["params_total_after"] == slim.count_params(student)[0]
    # The model it came from is untouched.
    assert len(slim.moe_layers_of(model)[0][1].experts) == 12


def test_mass_mode_bookkeeping():
    r = load_round()
    model = tiny_moe(experts=16, layers=3)
    tok = tokenizer()
    stats = {"code": slim.collect_moe_stats(model, tok, SKILL, 12)}
    low = r.spec("qwen", ["code"], "saliency", mass=0.5)
    high = r.spec("qwen", ["code"], "saliency", mass=0.9)
    results = {}
    for label, cell in [("low", low), ("high", high)]:
        student, report, extra = r.slim_for_spec(model, cell, stats)
        results[label] = extra
        counts = extra["kept_counts"]
        assert len(counts) == 3
        for layer, n in enumerate(counts):
            assert n == len(extra["kept"][layer]) >= 1
            saliency = stats["code"]["layers"][layer]["saliency"].tolist()
            taken = sum(saliency[e] for e in extra["kept"][layer])
            assert taken >= cell["mass"] * sum(saliency) - 1e-9
        assert extra["uniform"] == (len(set(counts)) == 1)
        total, run = slim.count_params(student)
        assert extra["params_total_after"] == total and extra["params_run_after"] == run
        assert total < slim.count_params(model)[0]
    assert sum(results["high"]["kept_counts"]) >= sum(results["low"]["kept_counts"])
    assert results["high"]["params_total_after"] >= results["low"]["params_total_after"]
    assert len(set(results["low"]["kept_counts"] + results["high"]["kept_counts"])) > 1


def test_olmoe_keep_all_is_an_exact_no_op():
    r = load_round()
    model = tiny_moe("olmoe", experts=8, top_k=2, layers=2, norm=False)
    tok = tokenizer()
    stats = {"code": slim.collect_moe_stats(model, tok, SKILL, 12)}
    cell = r.spec("olmoe", ["code"], "saliency", keep=8)
    student, report, extra = r.slim_for_spec(model, cell, stats)
    assert extra["kept_counts"] == [8, 8]
    ids = sample_ids()
    assert torch.equal(logits(model, ids), logits(student, ids))
    check = r.noop_check(student, model, tok, SKILL[:2])
    assert check["max_abs_logit_diff"] == 0.0 and check["argmax_equal"]
    cell32 = r.spec("olmoe", ["code"], "saliency", keep=4)
    smaller, _, _ = r.slim_for_spec(model, cell32, stats)
    assert (logits(model, ids) - logits(smaller, ids)).abs().max().item() > 0


def test_random_cells_need_no_stats_and_are_seeded():
    r = load_round()
    model = tiny_moe(experts=16, layers=2)
    kept = {}
    for seed in [0, 0, 1]:
        cell = r.spec("qwen", [], "random", keep=6, seed=seed)
        _, _, extra = r.slim_for_spec(model, cell, {})
        kept.setdefault(seed, []).append(extra["kept"])
        assert "kept_share" not in extra
    assert kept[0][0] == kept[0][1]
    assert kept[0][0] != kept[1][0]


def fake_pairs(count):
    pairs = []
    for index in range(count):
        pairs.append((f"question number {index} about topic {index * 7} " + "word " * 40, f"answer {index} " * 20))
    return pairs


def test_ranking_and_heal_texts_are_disjoint_and_do_not_depend_on_heal_length(monkeypatch):
    r = load_round()
    pairs = fake_pairs(r.POOL_PAIRS)
    monkeypatch.setattr(r, "skill_pairs", lambda skill: pairs)
    keys = [pairs[3][0]]
    rank, heal, info = r.skill_texts("code", None, keys)
    assert len(rank) == r.RANK_TEXTS and len(heal) == r.MAX_HEAL_TEXTS
    assert set(rank).isdisjoint(heal)
    assert info["dropped_overlap"] >= 1
    assert all(keys[0].lower() not in text.lower() for text in rank + heal)
    short = r.heal_texts_for(["code"], {"code": heal}, 100)
    longer = r.heal_texts_for(["code"], {"code": heal}, 500)
    assert len(short) == 800 and len(longer) == 4000
    assert set(short) <= set(longer)
    both = r.heal_texts_for(["code", "math"], {"code": heal[:2000], "math": heal[2000:]}, 100)
    assert len(both) == 800
    assert 300 < len([t for t in both if t in set(heal[:2000])]) < 500
    with pytest.raises(ValueError):
        r.heal_texts_for(["code"], {"code": heal}, 501)
    assert r.text_hash(rank) == r.text_hash(list(rank))
    assert r.text_hash(rank) != r.text_hash(rank[::-1])


def fake_row(code, math, medicine, general=0.5):
    return {"skill_scores": {"code": code, "math": math, "medicine": medicine, "general": general}, "error": None}


def test_decide_applies_the_go_rule():
    r = load_round()
    rows = {"olmoe_full": fake_row(0.5, 0.4, 0.3)}
    assert r.decide(rows)["olmoe"]["code"]["pass"] is None
    rows["olmoe_code_sal32"] = fake_row(0.47, 0.1, 0.1)
    for seed in range(3):
        rows[f"olmoe_random32_s{seed}"] = fake_row(0.0, 0.0, 0.0)
    verdict = r.decide(rows)
    assert verdict["olmoe"]["code"]["pass"] is True
    rows["olmoe_code_sal32"] = fake_row(0.40, 0.1, 0.1)
    assert r.decide(rows)["olmoe"]["code"]["pass"] is False
    rows["olmoe_code_sal32"] = fake_row(0.47, 0.39, 0.1)
    assert r.decide(rows)["olmoe"]["code"]["pass"] is False
    rows["olmoe_code_sal32"] = fake_row(0.47, 0.1, 0.1)
    rows["olmoe_random32_s1"] = fake_row(0.45, 0.0, 0.0)
    assert r.decide(rows)["olmoe"]["code"]["pass"] is False
    assert "olmoe" in r.decide(rows) and "qwen" in r.decide(rows)


def test_docstring_states_the_smoke_rule_and_the_go_rule():
    r = load_round()
    doc = r.__doc__
    assert "--smoke" in doc and "wait for its row" in doc
    assert "GPU hours" in doc and "15 minutes" in doc
    assert "Go rule" in doc and "within 0.05" in doc and "0.03" in doc
    assert "—" not in doc and "–" not in doc
