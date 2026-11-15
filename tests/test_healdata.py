import pytest

from wsp import healdata


def test_benchmarks_are_banned_in_any_spelling():
    for name in ["openai/gsm8k", "OpenAI/GSM8K", "cais/mmlu", "google-research-datasets/mbpp",
                 "openlifescienceai/medmcqa", "HuggingFaceH4/MATH-500", "meta-math/MetaMathQA"]:
        assert healdata.is_banned(name)
    with pytest.raises(ValueError):
        healdata.check_sources(["yahma/alpaca-cleaned", "openai/gsm8k"])


def test_no_pool_reads_a_banned_dataset():
    for name in healdata.SOURCES:
        ids = []
        for source in healdata.SOURCES[name]:
            ids.append(source[0])
        assert healdata.check_sources(ids) == ids


def test_row_pairs():
    assert healdata.row_pair({"question": "q", "answer": "a"}, "qa") == ("q", "a")
    assert healdata.row_pair({"instruction": "i", "input": "x", "output": "o"}, "alpaca") == ("i\n\nx", "o")
    assert healdata.row_pair({"instruction": "i", "input": "", "output": ""}, "alpaca") is None
    assert healdata.row_pair({"problem": "p", "solution": "s", "lang": "rust"}, "magicoder") is None
    assert healdata.row_pair({"problem": "p", "solution": "s", "lang": "python"}, "magicoder") == ("p", "s")
    assert healdata.row_pair({"text": " = Heading = "}, "text") is None
    row = {"messages": [{"content": "u"}, {"content": "a"}]}
    assert healdata.row_pair(row, "ultrachat") == ("u", "a")


def test_render_without_a_template():
    assert healdata.render(None, "u", "a") == "u\n\na"
    assert healdata.render(None, "plain", None) == "plain"


# ---------------------------------------------------------------- per-request reported sets, new renderers, practice

import os
import time

FAKE = {}


def fake_load(dataset_id, config, split, stream=False, seed=0):
    return FAKE[(dataset_id, config, split)]


def rows_for_pool(monkeypatch, name, text):
    """Make every source of one pool serve rows that render as (text + id, answer)."""
    FAKE.clear()
    for source in healdata.SOURCES[name] + healdata.EXTRA_SOURCES.get(name, []):
        FAKE[(source[0], source[1], source[2])] = [{"question": text + source[0] + str(i), "answer": "a"}
                                                   for i in range(50)]
    monkeypatch.setattr(healdata, "_load", fake_load)
    for source in healdata.SOURCES[name] + healdata.EXTRA_SOURCES.get(name, []):
        monkeypatch.setattr(healdata, "row_pair", lambda row, kind, features=None: (row["question"], row["answer"]))


def test_default_reported_set_is_banned_and_per_request_sets_differ():
    assert healdata.is_banned("openai/gsm8k")
    assert not healdata.is_banned("openai/gsm8k", reported=["cais/mmlu"])
    assert healdata.is_banned("ChilleD/SVAMP", reported=["chilled/svamp"])
    assert healdata.is_banned("tau/commonsense_qa", reported=["CommonsenseQA"]) is False
    assert healdata.is_banned("coastalcph/lex_glue", config="case_hold")
    assert not healdata.is_banned("coastalcph/lex_glue", config="ecthr_a")
    with pytest.raises(ValueError):
        healdata.check_sources(["ChilleD/SVAMP"], reported=["ChilleD/SVAMP"])
    assert healdata.check_sources(["openai/gsm8k"], reported=["x/y"]) == ["openai/gsm8k"]


def test_derived_from_is_excluded_when_the_parent_is_reported():
    openmath = "nvidia/OpenMathInstruct-2"
    assert healdata.is_banned(openmath)
    assert healdata.is_banned(openmath, reported=["openai/gsm8k"])
    assert healdata.is_banned(openmath, reported=["HuggingFaceH4/MATH-500"])
    assert not healdata.is_banned(openmath, reported=["ChilleD/SVAMP"])
    assert healdata.DERIVED_FROM["nvidia/openmathinstruct-2"] == ["gsm8k", "math"]


def test_pairs_skip_reported_sources_and_use_extras(monkeypatch):
    rows_for_pool(monkeypatch, "math", "q")
    default_ids = set(p[0].split("/")[0] for p in healdata.pairs("math", 60, seed=1))
    assert not any("OpenMath" in p[0] for p in healdata.pairs("math", 60, seed=1))
    other = healdata.pairs("math", 200, seed=1, reported=["ChilleD/SVAMP"])
    text = " ".join(p[0] for p in other)
    assert "SVAMP" not in text and "OpenMathInstruct" in text and default_ids
    assert "orca" in text


def test_emptied_pool_raises(monkeypatch):
    rows_for_pool(monkeypatch, "prose", "t")
    with pytest.raises(ValueError):
        healdata.pairs("prose", 5, reported=["Salesforce/wikitext"])
    with pytest.raises(ValueError):
        healdata.pool("nope", 5)


def test_old_call_signatures_still_work(monkeypatch):
    rows_for_pool(monkeypatch, "general", "g")
    assert len(healdata.pairs("general", 5)) == 5
    assert len(healdata.pairs("general", 5, 3)) == 5
    assert len(healdata.pairs("general", 5, seed=3, skip=2)) == 5
    assert len(healdata.pool("general", 4, None, 0, 1)) == 4
    assert len(healdata.pool("general", 4, tokenizer=None, seed=0, skip=0)) == 4
    assert len(healdata.mix(["general"], 3, None, 0, 0)) == 3
    assert healdata.is_banned("cais/mmlu") and healdata.check_sources(["a/b"]) == ["a/b"]
    assert healdata.row_pair({"question": "q", "answer": "a"}, "qa") == ("q", "a")


NEW_ROWS = [
    ({"Body": "Ann has 3 apples.", "Question": "How many after 2 more?", "Equation": "( 3 + 2 )", "Answer": "5"},
     "svamp", "Ann has 3 apples. How many after 2 more?", "Equation: ( 3 + 2 )\nThe answer is 5."),
    ({"body": "Seven apples.", "question": "How many?", "formula": "7=7", "answer": "7 (apples)"},
     "asdiv", "Seven apples. How many?", "Equation: 7=7\nThe answer is 7 (apples)."),
    ({"question": "2+2?", "options": ["A)3", "B)4"], "rationale": "two plus two", "correct": "B"},
     "aqua", "2+2?\nA. 3\nB. 4", "two plus two\nThe answer is B. 4"),
    ({"problem": "p", "generated_solution": "s"}, "openmath", "p", "s"),
    ({"instruction": "i", "response": "r"}, "selfoss", "i", "r"),
    ({"question": "write", "solutions": '["print(1)"]', "starter_code": ""}, "apps", "write",
     "```python\nprint(1)\n```"),
    ({"description": "d", "solutions": {"language": [2, 3], "solution": ["cpp", "py"]}}, "contests", "d",
     "```python\npy\n```"),
    ({"question": "q?", "context": {"contexts": ["c1", "c2"]}, "long_answer": "because", "final_decision": "yes"},
     "pubmedqa", "Question: q?\n\nContext:\nc1\n\nc2", "because\nAnswer: yes"),
    ({"question": "q", "answer": "a"}, "medquad", "q", "a"),
    ({"Question": "q", "Complex_CoT": "think", "Response": "r"}, "o1med", "q", "think\n\nr"),
    ({"question": "who?", "answer": {"value": "Bob", "aliases": ["Bob"]}}, "trivia", "who?", "Bob"),
    ({"question": "who?", "answer": ["Al", "Alan"]}, "nqopen", "who?", "Al"),
    ({"context": "ctx", "question": "q", "answers": {"text": ["ans"]}}, "squad", "Context: ctx\n\nQuestion: q", "ans"),
    ({"question": "q", "answer": "sentence", "label": 1}, "wikiqa", "q", "sentence"),
    ({"question": "q", "choices": {"label": ["A", "B"], "text": ["x", "y"]}, "answerKey": "B"}, "mc",
     "q\nA. x\nB. y", "B. y"),
    ({"question_stem": "stem", "choices": {"label": ["A", "B"], "text": ["x", "y"]}, "answerKey": "A"}, "mc",
     "stem\nA. x\nB. y", "A. x"),
    ({"question": "q", "answer": True, "facts": "f"}, "strategyqa", "q", "f\nThe answer is Yes."),
    ({"text": '{"answer": 1, "text": "t", "question": "q", "options": ["x", "y"]}'}, "logiqa",
     "t\n\nq\nA. x\nB. y", "B. y"),
    ({"inputs": "i", "targets": "t"}, "aya", "i", "t"),
]


@pytest.mark.parametrize("row,kind,user,assistant", NEW_ROWS)
def test_new_renderers(row, kind, user, assistant):
    assert healdata.row_pair(row, kind) == (user, assistant)


def test_text_renderers_and_skips():
    long = "x" * 250
    assert healdata.row_pair({"abstract": long}, "abstract") == (long, None)
    assert healdata.row_pair({"abstract": "short"}, "abstract") is None
    assert healdata.row_pair({"text": ["a" * 150, "b" * 150]}, "legal_text") == ("a" * 150 + "\n\n" + "b" * 150, None)
    assert healdata.row_pair({"document": long}, "document") == (long, None)
    assert healdata.row_pair({"question": "q", "answer": "s", "label": 0}, "wikiqa") is None
    assert healdata.row_pair({"description": "d", "solutions": {"language": [2], "solution": ["c"]}}, "contests") is None
    assert healdata.row_pair({"question": "q", "answer": []}, "nqopen") is None


def test_classify_uses_label_names():
    class Label:
        names = ["Taxes", "Notices"]
    user, assistant = healdata.row_pair({"text": "A provision.", "label": 1}, "classify", {"label": Label()})
    assert assistant == "Notices" and "A provision." in user
    assert healdata.row_pair({"text": "t", "label": 7}, "classify")[1] == "label 7"


def test_translation_renders_both_directions():
    even = healdata.row_pair({"translation": {"de": "Hallo", "en": "Hi!!"}}, "translation")
    odd = healdata.row_pair({"translation": {"de": "Hallo", "en": "Hi!"}}, "translation")
    assert even == ("Translate from German to English:\n\nHallo", "Hi!!")
    assert odd == ("Translate from English to German:\n\nHi!", "Hallo")


def test_every_source_kind_has_a_renderer():
    kinds = set()
    for sources in list(healdata.SOURCES.values()) + list(healdata.EXTRA_SOURCES.values()):
        for source in sources:
            kinds.add(source[3])
    for kind in kinds:
        try:
            healdata.row_pair({}, kind)
        except ValueError as error:
            assert "unknown row kind" not in str(error)
        except Exception:
            pass


def test_pools_and_rows_table_cover_every_source():
    for name in ["math", "code", "medicine", "law", "knowledge", "reasoning", "multilingual", "general", "prose"]:
        assert name in healdata.SOURCES
    for sources in healdata.SOURCES.values():
        for dataset_id, config, split, kind in sources:
            assert (dataset_id, config, split) in healdata.ROWS


def test_practice_never_reads_a_training_split():
    trained = set()
    for sources in list(healdata.SOURCES.values()) + list(healdata.EXTRA_SOURCES.values()):
        for dataset_id, config, split, kind in sources:
            trained.add((dataset_id, config, split))
    for skill, sources in healdata.PRACTICE.items():
        for dataset_id, config, split, kind in sources:
            assert (dataset_id, config, split) not in trained, skill + " practice reads a training split"


PRACTICE_ROWS = {
    ("ChilleD/SVAMP", None, "test"): [{"Body": "b", "Question": "q", "Equation": "e", "Answer": str(i)}
                                      for i in range(20)],
    ("EleutherAI/asdiv", "asdiv", "validation"): [{"body": "b", "question": "q", "formula": "f", "answer": str(i) + " (x)"}
                                                  for i in range(20)],
    ("deepmind/aqua_rat", "raw", "validation"): [{"question": "q", "options": ["A)1", "B)2"], "rationale": "r",
                                                  "correct": "A"} for i in range(20)],
}


def test_practice_items_and_exclusion(monkeypatch):
    FAKE.clear()
    FAKE.update(PRACTICE_ROWS)
    monkeypatch.setattr(healdata, "_load", fake_load)
    items = healdata.practice("math", 9, seed=0)
    assert len(items) == 9
    assert set(item["kind"] for item in items) == {"number", "choice"}
    for item in items:
        assert set(item) >= {"prompt", "answer", "kind", "choices", "source"}
    assert items == healdata.practice("math", 9, seed=0)
    only = healdata.practice("math", 6, reported=["ChilleD/SVAMP", "EleutherAI/asdiv"])
    assert set(item["source"] for item in only) == {"deepmind/aqua_rat"}
    with pytest.raises(ValueError):
        healdata.practice("math", 3, reported=["ChilleD/SVAMP", "EleutherAI/asdiv", "deepmind/aqua_rat"])
    with pytest.raises(ValueError):
        healdata.practice("law", 3)


def test_practice_items_for_other_kinds():
    trivia = healdata.practice_item({"question": "q", "answer": {"value": "Paris", "aliases": ["paris"]}}, "trivia", "t")
    assert trivia["kind"] == "text" and "Paris" in trivia["aliases"]
    medical = healdata.practice_item({"question": "q", "context": {"contexts": ["c"]}, "final_decision": "maybe"},
                                     "pubmedqa", "p")
    assert medical["choices"] == ["yes", "no", "maybe"] and medical["answer"] == "maybe"
    assert healdata.practice_item({"question": "q", "context": {"contexts": ["c"]}, "final_decision": "?"},
                                  "pubmedqa", "p") is None
    code_row = {"question": "add", "starter_code": "", "input_output": '{"inputs": ["1 2\\n"], "outputs": ["3\\n"]}'}
    code = healdata.practice_item(code_row, "apps", "a")
    assert code["kind"] == "code" and code["tests"] == {"inputs": ["1 2\n"], "outputs": ["3\n"]}
    assert healdata.practice_item(dict(code_row, starter_code="def f():"), "apps", "a") is None


def test_check_answer_number_choice_text():
    number = {"kind": "number", "answer": "145"}
    assert healdata.check_answer(number, "so 290 / 2. The answer is 145.")
    assert healdata.check_answer(number, "#### 145")
    assert not healdata.check_answer(number, "The answer is 146")
    choice = {"kind": "choice", "answer": "C", "choices": ["A. x", "B. y", "C. z"]}
    assert healdata.check_answer(choice, "Reasoning...\nThe answer is C.")
    assert healdata.check_answer(choice, "Answer: (C)")
    assert not healdata.check_answer(choice, "The answer is B")
    assert not healdata.check_answer(choice, "no idea")
    medical = {"kind": "choice", "answer": "maybe", "choices": ["yes", "no", "maybe"]}
    assert healdata.check_answer(medical, "It is unclear, so maybe.")
    assert not healdata.check_answer(medical, "No.")
    text = {"kind": "text", "answer": "Sinclair Lewis", "aliases": ["Harry Sinclair Lewis", "Sinclair Lewis"]}
    assert healdata.check_answer(text, "It was sinclair lewis!")
    assert not healdata.check_answer(text, "Hemingway")


def test_check_answer_code_runs_stdin_tests():
    item = {"kind": "code", "answer": "", "tests": {"inputs": ["1 2\n"], "outputs": ["3\n"]}}
    good = "```python\na, b = map(int, input().split())\nprint(a + b)\n```"
    bad = "```python\nprint(0)\n```"
    assert healdata.check_answer(item, good)
    assert not healdata.check_answer(item, bad)


def test_describe_reports_usable_and_excluded():
    default = healdata.describe()
    assert set(default) >= {"math", "code", "medicine", "law", "knowledge", "reasoning", "multilingual"}
    names = [entry["id"] for entry in default["math"]["usable"]]
    assert "nvidia/OpenMathInstruct-2" not in names
    excluded = default["math"]["excluded"]
    assert any(e["id"] == "nvidia/OpenMathInstruct-2" and "gsm8k" in e["reason"] for e in excluded)
    custom = healdata.describe(reported=["ChilleD/SVAMP"])
    names = [entry["id"] for entry in custom["math"]["usable"]]
    assert "nvidia/OpenMathInstruct-2" in names and "ChilleD/SVAMP" not in names
    assert any(e["id"] == "ChilleD/SVAMP" for e in custom["math"]["excluded"])
    assert default["math"]["usable"][0]["rows"] == 200035


@pytest.mark.skipif(os.environ.get("HEALDATA_NETWORK") != "1", reason="set HEALDATA_NETWORK=1 to read the network")
def test_network_three_rows_from_every_source():
    seen = set()
    failed = []
    entries = []
    for sources in list(healdata.SOURCES.values()) + list(healdata.EXTRA_SOURCES.values()):
        for dataset_id, config, split, kind in sources:
            entries.append((dataset_id, config, split, kind, "train"))
    for skill, sources in healdata.PRACTICE.items():
        for dataset_id, config, split, kind in sources:
            entries.append((dataset_id, config, split, kind, "practice"))
    for dataset_id, config, split, kind, role in entries:
        if (dataset_id, config, split, role) in seen:
            continue
        seen.add((dataset_id, config, split, role))
        time.sleep(1.5)
        try:
            rows = healdata.sample_rows(dataset_id, config, split, 3)
            assert len(rows) == 3
            rendered = 0
            for row in rows:
                if role == "train":
                    if healdata.row_pair(row, kind) is not None:
                        rendered += 1
                elif healdata.practice_item(row, kind, dataset_id) is not None:
                    rendered += 1
            print("NETWORK ok", dataset_id, config, split, kind, role, rendered, "of 3 rendered")
        except Exception as error:
            failed.append((dataset_id, config, split, str(error)[:120]))
    assert failed == []


def test_a_malformed_row_is_skipped_not_an_error():
    # LogiQA2.0 holds rows without "options"; the first real run died on one.
    assert healdata.row_pair({"text": '{"text": "t", "question": "q", "answer": 0}'}, "logiqa") is None
    assert healdata.row_pair({"question": "q"}, "mc") is None
    assert healdata.row_pair({}, "qa") is None


def _fake_sources(monkeypatch, sizes):
    names = list(sizes)
    monkeypatch.setitem(healdata.SOURCES, "fakepool", [(name, None, "train", "qa") for name in names])

    def fake_load(dataset_id, config, split, stream=False, seed=0):
        return [{"question": dataset_id + " q" + str(i), "answer": "a" + str(i)} for i in range(sizes[dataset_id])]

    monkeypatch.setattr(healdata, "_load", fake_load)
    monkeypatch.setattr(healdata, "_features", lambda rows: None)


def test_a_small_source_does_not_leave_the_pool_short(monkeypatch):
    # Round 22 died on this: SVAMP has 700 rows, and an even share per source left the math pool at 4168 of 5096.
    _fake_sources(monkeypatch, {"big/one": 5000, "tiny/two": 50, "big/three": 5000})
    got = healdata.pairs("fakepool", 3000)
    assert len(got) == 3000
    assert len(set(got)) == 3000
    assert sum(1 for user, _ in got if user.startswith("tiny/two")) == 50


def test_pairs_order_is_prefix_stable_so_skips_are_disjoint(monkeypatch):
    _fake_sources(monkeypatch, {"big/one": 900, "tiny/two": 40, "big/three": 900})
    short = healdata.pairs("fakepool", 200)
    long = healdata.pairs("fakepool", 700)
    assert long[:200] == short
    first = healdata.pairs("fakepool", 300, skip=0)
    second = healdata.pairs("fakepool", 300, skip=300)
    assert set(first).isdisjoint(set(second))
    assert healdata.pairs("fakepool", 100, skip=250) == long[250:350]


def test_pairs_is_short_only_when_every_source_is_spent(monkeypatch):
    _fake_sources(monkeypatch, {"a/x": 30, "b/y": 20})
    assert len(healdata.pairs("fakepool", 100)) == 50
