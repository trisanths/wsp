import json

import pytest

from wsp import rft


def test_ngram_overlap_and_decontaminate():
    test_text = "def has_close_elements(numbers, threshold): check if in given list of numbers are any two numbers closer"
    index = rft.ngram_index([test_text])
    items = [
        {"id": "a", "user": "Write has_close_elements(numbers, threshold) check if in given list of numbers are any two numbers closer."},
        {"id": "b", "user": "Reverse a string and return it in upper case with no spaces."},
    ]
    clean, dropped = rft.decontaminate(items, index)
    assert dropped == 1
    assert [i["id"] for i in clean] == ["b"]


def test_short_text_never_matches():
    index = rft.ngram_index(["one two three"])
    assert not rft.shares_ngram("one two three", index)


def test_asserts_program_passes_and_fails():
    item = {"id": "x", "kind": "asserts", "user": "u", "tests": ["assert add(1, 2) == 3", "assert add(0, 0) == 0"]}
    good = "```python\ndef add(a, b):\n    return a + b\n```"
    bad = "```python\ndef add(a, b):\n    return a - b\n```"
    assert rft.code_rewards([item, item], [good, bad]) == [1.0, 0.0]


def test_check_program():
    item = {"id": "y", "kind": "check", "header": "from typing import *", "entry_point": "Solution().twice",
            "test": "def check(candidate):\n    assert candidate(2) == 4\n"}
    good = "```python\nclass Solution:\n    def twice(self, x: int) -> int:\n        return 2 * x\n```"
    bad = "```python\nclass Solution:\n    def twice(self, x: int) -> int:\n        return x\n```"
    assert rft.code_rewards([item, item], [good, bad]) == [1.0, 0.0]


def test_stdin_program_variants():
    item = {"id": "z", "kind": "stdin", "user": "u", "cases": [["3\n1 2 3\n", "6\n"], ["2\n5 5\n", "10"]]}
    plain = "```python\nn = int(input())\nprint(sum(map(int, input().split())))\n```"
    buffered = "```python\nimport sys\ndata = sys.stdin.buffer.read().split()\nprint(sum(map(int, data[1:])))\n```"
    wrong = "```python\nprint(0)\n```"
    crash = "```python\nraise ValueError('x')\n```"
    out = rft.code_rewards([item] * 4, [plain, buffered, wrong, crash])
    assert out == [1.0, 1.0, 0.0, 0.0]


def test_stdin_case_cap():
    cases = [[str(i), str(i)] for i in range(50)]
    prog = rft.stdin_program("print(input())", cases)
    assert prog.count("('") <= rft.MAX_STDIN_CASES * 2 + 2


def test_exit_before_tests_does_not_pass():
    item = {"id": "x", "kind": "asserts", "user": "u", "tests": ["assert f() == 1"]}
    cheat = "```python\nimport sys\ndef f():\n    return 2\nsys.exit(0)\n```"
    assert rft.code_rewards([item], [cheat]) == [0.0]


def test_math_and_mcq_rewards():
    items = [{"id": "m", "gold": "42"}]
    assert rft.get_task("math").rewards(items, ["so \\boxed{42}"]) == [1.0]
    assert rft.get_task("math").rewards(items, ["so \\boxed{41}"]) == [0.0]
    assert rft.get_task("math").rewards(items, ["no box"]) == [0.0]
    q = [{"id": "q", "gold": "B"}]
    assert rft.get_task("mcq").rewards(q, ["Reasoning. The answer is B."]) == [1.0]
    assert rft.get_task("mcq").rewards(q, ["Answer: C"]) == [0.0]
    assert rft.get_task("mcq").rewards(q, ["B"]) == [1.0]
    with pytest.raises(ValueError):
        rft.get_task("nope")


def test_split_is_stable_and_near_fraction():
    items = [{"id": f"p{i}"} for i in range(4000)]
    train, held = rft.split_items(items, 0.05)
    assert len(train) + len(held) == 4000
    assert 150 < len(held) < 250
    again, held2 = rft.split_items(list(reversed(items)), 0.05)
    assert {i["id"] for i in held} == {i["id"] for i in held2}


def test_select_samples_distinct_passing_only():
    samples = [{"text": "a  b", "reward": 1.0}, {"text": "a b", "reward": 1.0}, {"text": "c", "reward": 0.0},
               {"text": "d", "reward": 1.0}, {"text": "e" * 500, "reward": 1.0}]
    picked = rft.select_samples(samples, 2, seed=1)
    assert len(picked) == 2
    texts = {rft.normalized(s["text"]) for s in picked}
    assert len(texts) == 2 and "c" not in texts and "e" * 500 not in texts


def test_select_is_deterministic():
    samples = [{"text": str(i), "reward": 1.0} for i in range(10)]
    assert rft.select_samples(samples, 2, 5) == rft.select_samples(samples, 2, 5)


def test_build_sft_uses_teacher_only_with_budget():
    items = [{"id": "a", "user": "ua"}, {"id": "b", "user": "ub"}, {"id": "c", "user": "uc"}]
    student = {"a": [{"text": "s1", "reward": 1.0}, {"text": "s2", "reward": 1.0}], "b": [{"text": "x", "reward": 0.0}]}
    teacher = {"a": [{"text": "t1", "reward": 1.0}, {"text": "t2", "reward": 1.0}],
               "b": [{"text": "t3", "reward": 1.0}, {"text": "t4", "reward": 1.0}, {"text": "t5", "reward": 1.0}]}
    rows, stats = rft.build_sft(items, student, teacher, student_keep=2, teacher_keep=1)
    by = {}
    for r in rows:
        by.setdefault(r["id"], []).append(r["origin"])
    assert sorted(by["a"]) == ["student", "student", "teacher"]
    assert sorted(by["b"]) == ["teacher", "teacher"]
    assert "c" not in by
    assert stats["solved_by_either"] == 2 and stats["student_rows"] == 2 and stats["teacher_rows"] == 3


def test_drop_sample_overlap():
    index = rft.ngram_index(["alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi"])
    samples = [{"text": "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu"}, {"text": "fine"}]
    clean, dropped = rft.drop_sample_overlap(samples, index)
    assert dropped == 1 and len(clean) == 1


def test_item_builders():
    mbpp = rft.mbpp_train_item({"task_id": 601, "text": "Write f.", "test_list": ["assert f()==1"], "test_setup_code": "",
                                "code": "def f(): return 1"})
    assert mbpp["id"] == "mbpp:601" and "Your code should pass these tests" in mbpp["user"]
    row = {"id": "oss_1", "source": "oss", "question": "Add.",
           "test_cases": json.dumps(["assert add(1,2)==3"] * 3 + ["not a test"]),
           "inferences": str([{"completion": "def add(a,b): return a+b", "pass_rate": 1.0},
                              {"completion": "bad", "pass_rate": 0.0}])}
    ace = rft.ace_item(row)
    assert ace["id"] == "ace:oss_1" and len(ace["tests"]) == 3 and ace["mean_pass_rate"] == 0.5
    row["inferences"] = str([{"completion": "bad", "pass_rate": 0.5}])
    assert rft.ace_item(row) is None
    lc = rft.leetcode_item({"task_id": "two-sum", "prompt": "import math\nclass Solution:\n    pass", "starter_code": "class Solution:\n    def f(self):\n        ",
                            "problem_description": "Do it.", "test": "def check(c):\n    pass", "entry_point": "Solution().f", "completion": "x"})
    assert lc["header"].strip() == "import math" and lc["kind"] == "check"
    apps = rft.apps_item({"problem_id": 1, "question": "q", "difficulty": "introductory", "solutions": json.dumps(["print(1)"]),
                          "input_output": json.dumps({"inputs": ["1\n"], "outputs": ["1\n"]})})
    assert apps["kind"] == "stdin" and apps["reference"] == "print(1)"
    assert rft.apps_item({"problem_id": 2, "question": "q", "difficulty": "x", "solutions": "[]",
                          "input_output": json.dumps({"fn_name": "f", "inputs": [[1]], "outputs": [[1]]})}) is None


def test_reference_solutions_pass_their_own_tests():
    mbpp = rft.mbpp_train_item({"task_id": 1, "text": "t", "test_list": ["assert f(2)==4"], "test_setup_code": "",
                                "code": "def f(x):\n    return x*2"})
    apps = rft.apps_item({"problem_id": 1, "question": "q", "difficulty": "d", "solutions": json.dumps(["print(int(input())*2)"]),
                          "input_output": json.dumps({"inputs": ["2\n"], "outputs": ["4\n"]})})
    outs = ["```python\n" + mbpp["reference"] + "\n```", "```python\n" + apps["reference"] + "\n```"]
    assert rft.code_rewards([mbpp, apps], outs) == [1.0, 1.0]


class FakeTokenizer:
    chat_template = None
    bos_token = ""
    eos_token = "<eos>"

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [ord(c) % 200 for c in text]}


def test_encode_sft_masks_prompt_and_skips_long():
    tok = FakeTokenizer()
    ids, labels = rft.encode_sft(tok, "hi", "yo", max_len=200)
    assert len(ids) == len(labels)
    n_prompt = sum(1 for x in labels if x == -100)
    assert n_prompt > 0 and labels[n_prompt:] == ids[n_prompt:]
    assert rft.encode_sft(tok, "hi", "y" * 500, max_len=100) is None


def test_make_batches_cover_all_once_and_respect_budget():
    examples = [(list(range(n)), list(range(n))) for n in [10, 200, 30, 400, 50, 60, 70, 80] * 20]
    batches = rft.make_batches(examples, max_tokens=800, max_seqs=4, seed=1)
    seen = sorted(i for b in batches for i in b)
    assert seen == list(range(len(examples)))
    for b in batches:
        assert len(b) <= 4
        assert max(len(examples[i][0]) for i in b) * len(b) <= 800 or len(b) == 1


def test_pass_rate_stats():
    s = rft.pass_rate_stats({"a": [1.0, 0.0], "b": [0.0, 0.0], "c": [1.0, 1.0]})
    assert s["prompts"] == 3 and abs(s["mean_pass"] - 0.5) < 1e-9
    assert abs(s["solved_any"] - 2 / 3) < 1e-9 and abs(s["solved_all"] - 1 / 3) < 1e-9
    assert rft.pass_rate_stats({})["prompts"] == 0


def test_apps_item_survives_huge_integers():
    big = "9" * 9000
    row = {"problem_id": 3, "question": "q", "difficulty": "d", "solutions": "[" + big + "]",
           "input_output": "{\"inputs\": [\"1\"], \"outputs\": [" + big + "]}"}
    assert rft.apps_item(row) is None


def test_number_runs_do_not_count_as_overlap():
    index = rft.ngram_index(["assert f([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]) == 0 0 0 0 0 0 0 0 0 0 0 0 0 0"])
    assert not rft.shares_ngram("x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13] and 0 0 0 0 0 0 0 0 0 0 0 0 0", index)
    text = "def has_close_elements numbers threshold check if in given list of numbers are any two closer"
    assert rft.shares_ngram(text, rft.ngram_index([text]))
