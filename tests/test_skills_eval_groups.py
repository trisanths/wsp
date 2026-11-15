import random

import pytest

from tests.test_skills_eval import tiny_lm
from wsp import skills_eval


def fake_mmlu_rows():
    """A small MMLU-shaped table: every subject the groups name, plus two others."""
    subjects = []
    for group in skills_eval.MMLU_GROUPS.values():
        subjects += group
    subjects += ["astronomy", "world_religions"]
    rows = []
    for subject in subjects:
        for k in range(6):
            rows.append({"subject": subject, "question": f"{subject} question {k}",
                         "choices": ["a", "b", "c", "d"], "answer": k % 4})
    return rows


def test_mmlu_subject_items_filter_by_subject_and_keep_test_ids():
    rows = fake_mmlu_rows()
    for group, subjects in skills_eval.MMLU_GROUPS.items():
        items = skills_eval.mmlu_subject_items(subjects, rows=rows)
        assert len(items) == 6 * len(subjects)
        assert {item["subject"] for item in items} == set(subjects)
        assert all(item["id"].startswith("mmlu_test:") for item in items)
        assert all(item["prompt"].endswith("\nAnswer:") for item in items)
    with pytest.raises(ValueError):
        skills_eval.mmlu_subject_items(["not_a_subject"], rows=rows)


def test_mmlu_general_items_never_share_an_item_with_the_domain_set():
    rows = fake_mmlu_rows()
    for group, subjects in skills_eval.MMLU_GROUPS.items():
        domain = skills_eval.mmlu_subject_items(subjects, rows=rows)
        general = skills_eval.mmlu_general_items(subjects, count=1000, rows=rows)
        domain_rows = {item["id"].split(":")[1] for item in domain}
        general_rows = {item["id"].split(":")[1] for item in general}
        assert domain_rows.isdisjoint(general_rows)
        assert not ({item["subject"] for item in general} & set(subjects))
        assert len(general) == len(rows) - len(domain)
        assert len({item["id"] for item in general}) == len(general)


def test_mmlu_general_items_without_exclusion_follow_the_seed_0_order():
    rows = fake_mmlu_rows()
    general = skills_eval.mmlu_general_items((), count=10, rows=rows)
    order = list(range(len(rows)))
    random.Random(0).shuffle(order)
    assert [item["id"] for item in general] == [f"mmlu_dev:{i}" for i in order[:10]]


def test_mmlu_general_items_respect_count():
    rows = fake_mmlu_rows()
    assert len(skills_eval.mmlu_general_items(skills_eval.MMLU_GROUPS["law"], count=7, rows=rows)) == 7


def test_load_items_routes_group_skills(monkeypatch):
    monkeypatch.setattr(skills_eval, "_mmlu_test_rows", fake_mmlu_rows)
    assert len(skills_eval.load_items("mmlu_law")) == 18
    assert len(skills_eval.load_items("mmlu_medical")) == 30
    assert len(skills_eval.load_items("mmlu_accounting")) == 12
    assert len(skills_eval.load_items("mmlu_general_ex_law")) == len(fake_mmlu_rows()) - 18
    assert skills_eval.EXTRA_SKILLS == ["mmlu_medical", "mmlu_general_ex_medical", "mmlu_accounting",
                                        "mmlu_general_ex_accounting", "mmlu_law", "mmlu_general_ex_law"]


def test_extra_skills_score_on_a_tiny_model():
    model, tokenizer = tiny_lm()
    items = {"mmlu_law": [{"id": "x", "prompt": "the cat sat on", "gold": 1}]}
    result = skills_eval.evaluate_suite(model, tokenizer, skills=["mmlu_law"], items=items, log=False)
    assert result["skills"]["mmlu_law"]["n"] == 1
    assert skills_eval.SKILLS == ["gsm8k", "math500", "humaneval", "mbpp", "medmcqa", "mmlu"]
