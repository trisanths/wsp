import pytest

from wsp import benchtrain


def fake_rows(dataset_id, config, split):
    assert split == "train"
    if dataset_id == "openlifescienceai/medmcqa":
        rows = [{"question": f"q{i}", "opa": "a", "opb": "b", "opc": "c", "opd": "d", "cop": i % 4,
                 "exp": "because " * 10 if i % 5 else None} for i in range(50)]
        return rows
    if dataset_id == "openai/gsm8k":
        return [{"question": f"What is {i}+1?", "answer": f"{i}+1=<<{i}+1={i + 1}>>{i + 1}\n#### {i + 1}"}
                for i in range(30)]
    if dataset_id == "EleutherAI/hendrycks_math":
        return [{"problem": f"{config} problem {i}", "solution": f"so \\boxed{{{i}}}"} for i in range(10)]
    if dataset_id == "coastalcph/lex_glue":
        assert config == "case_hold"
        return [{"context": f"court says <HOLDING> {i}", "endings": ["e0", "e1", "e2", "e3", "e4"], "label": i % 5}
                for i in range(40)]
    raise AssertionError(dataset_id)


@pytest.fixture(autouse=True)
def patch_rows(monkeypatch):
    monkeypatch.setattr(benchtrain, "_rows", fake_rows)


def test_medmcqa_pairs_skip_rows_without_an_explanation():
    out = benchtrain.pairs("medicine", 100)
    assert len(out) == 40
    user, assistant = out[0]
    assert user.startswith("Question: q")
    assert assistant.startswith("Answer: ")


def test_medmcqa_pair_rejects_bad_rows():
    assert benchtrain.medmcqa_pair({"question": "q", "opa": "a", "opb": "b", "opc": "c", "opd": "d", "cop": 9,
                                    "exp": "x" * 80}) is None
    assert benchtrain.medmcqa_pair({"question": "q", "opa": "a", "opb": None, "opc": "c", "opd": "d", "cop": 1,
                                    "exp": "x" * 80}) is None


def test_gsm8k_answer_loses_calculator_tags_and_ends_with_the_final_number():
    text = benchtrain.clean_gsm8k_answer("2+3=<<2+3=5>>5\n#### 5")
    assert "<<" not in text
    assert text.endswith("The answer is 5.")
    assert benchtrain.clean_gsm8k_answer("no final marker") is None


def test_math_pairs_alternate_over_gsm8k_and_math():
    out = benchtrain.pairs("math", 6)
    assert len(out) == 6
    assert "Solve step by step and end with" in out[0][0]
    assert "\\boxed{}" in out[1][0]


def test_math_train_reads_every_subject():
    problems = [user for user, _ in benchtrain.math_train()]
    assert len(problems) == 70
    for subject in benchtrain.MATH_SUBJECTS:
        assert any(subject in p for p in problems)


def test_casehold_pairs_hold_the_gold_ending():
    out = benchtrain.pairs("law", 10)
    assert len(out) == 10
    user, assistant = out[0]
    assert "Citing text:" in user
    assert assistant.startswith("Answer: ")
    assert benchtrain.casehold_pair({"context": "c", "endings": ["a"], "label": 0}) is None


def test_order_is_seeded_and_a_prefix():
    a = benchtrain.pairs("law", 5, seed=3)
    b = benchtrain.pairs("law", 12, seed=3)
    assert b[:5] == a
    assert benchtrain.pairs("law", 5, seed=4) != a


def test_unknown_skill_is_an_error():
    with pytest.raises(ValueError):
        benchtrain.pairs("cooking", 3)
