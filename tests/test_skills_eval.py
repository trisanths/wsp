import time

import torch

from wsp import skills_eval
from wsp.skills_eval import (
    answers_match,
    assemble_humaneval,
    assemble_mbpp,
    extract_code,
    gsm8k_prediction,
    last_boxed,
    normalize_answer,
    run_program,
    run_programs,
    same_number,
    to_number,
)


def test_gsm8k_prediction_rules_in_order():
    assert gsm8k_prediction("So 3 + 4 = 7. The answer is 7.") == 7.0
    assert gsm8k_prediction("The answer is $1,234.") == 1234.0
    assert gsm8k_prediction("first 5, then The answer is 6, finally the answer is 9") == 9.0
    assert gsm8k_prediction("The answer is \\boxed{42}") == 42.0
    assert gsm8k_prediction("work 3 4\n#### 12") == 12.0
    assert gsm8k_prediction("so we get \\boxed{-8} in the end 99 apples") == -8.0
    assert gsm8k_prediction("I think it is 5 or 17") == 17.0
    assert gsm8k_prediction("no numbers here") is None


def test_final_number_matches_modal_graft():
    assert skills_eval.final_number("Natalia sold 48/2 = 24 clips.\n#### 72") == 72.0
    assert skills_eval.final_number("total 3,500.") == 3500.0
    assert same_number(72.0, 72.0000001)
    assert not same_number(None, 72.0)
    assert not same_number(71.0, 72.0)


def test_last_boxed_nests_braces():
    assert last_boxed("a \\boxed{1} then \\boxed{\\frac{1}{2}} end") == "\\frac{1}{2}"
    assert last_boxed("\\fbox{7}") == "7"
    assert last_boxed("\\boxed 5$.") == "5"
    assert last_boxed("no box") is None
    assert last_boxed("\\boxed{unclosed") is None


def test_normalize_and_match_math_answers():
    assert normalize_answer("\\dfrac{3}{4}") == "\\frac{3}{4}"
    assert normalize_answer("x = 5") == "5"
    assert normalize_answer("90^\\circ") == "90"
    assert normalize_answer("\\text{(B)}") == "(B)"
    assert normalize_answer("\\frac12") == "\\frac{1}{2}"
    assert to_number("\\frac{3}{4}") == 0.75
    assert to_number("-\\frac{1}{2}") == -0.5
    assert to_number("3/4") == 0.75
    assert to_number("1,000") == 1000.0
    assert to_number("\\sqrt{2}") is None
    assert answers_match("0.75", "\\frac{3}{4}")
    assert answers_match("\\dfrac{3}{4}", "\\frac34")
    assert answers_match("10\\%", "10")
    assert answers_match("\\left( 3, \\frac{\\pi}{2} \\right)", "(3,\\frac{\\pi}{2})")
    assert not answers_match("\\frac{2}{3}", "\\frac{3}{4}")
    assert not answers_match(None, "1")


def test_extract_code_picks_function_block():
    text = "Here:\n```\npip install x\n```\nand\n```python\ndef f(x):\n    return x\n```\nDone."
    assert extract_code(text) == "def f(x):\n    return x\n"
    assert extract_code("```python\ndef g():\n    return 1") == "def g():\n    return 1"
    assert extract_code("def h():\n    return 2") == "def h():\n    return 2"


PROMPT = 'from typing import List\n\n\ndef add(xs: List[int]) -> int:\n    """Sum xs."""\n'
TEST = "def check(candidate):\n    assert candidate([1, 2]) == 3\n"


def test_assemble_humaneval_full_function_keeps_prompt_imports():
    output = "```python\ndef add(xs: List[int]) -> int:\n    return sum(xs)\n```"
    program = assemble_humaneval(PROMPT, output, TEST, "add")
    assert program.startswith("from typing import List")
    assert program.count("def add(") == 1
    assert program.rstrip().endswith("check(add)")
    assert run_program(program)["passed"]


def test_assemble_humaneval_body_only_goes_after_prompt():
    program = assemble_humaneval(PROMPT, "return sum(xs)", TEST, "add")
    assert "    return sum(xs)" in program
    assert run_program(program)["passed"]
    wrong = assemble_humaneval(PROMPT, "    return 0\n", TEST, "add")
    assert not run_program(wrong)["passed"]


def test_assemble_mbpp():
    output = "```python\ndef square(n):\n    return n * n\n```"
    program = assemble_mbpp(output, ["assert square(3) == 9", "assert math.isclose(square(0.5), 0.25)"],
                            ["import math"])
    assert program.startswith("import math\n")
    assert run_program(program)["passed"]


def test_sandbox_pass_fail_and_timeout():
    good = run_program("x = 1\nassert x == 1\n")
    assert good["passed"] and good["status"] == "passed"
    bad = run_program("assert 1 == 2, 'nope'\n")
    assert not bad["passed"] and bad["status"] == "failed"
    assert "nope" in bad["stderr"]
    start = time.time()
    slow = run_program("while True:\n    pass\n", timeout=1.0)
    assert slow["status"] == "timeout"
    assert time.time() - start < 8


def test_sandbox_blocks_network_and_processes():
    program = (
        "import socket, subprocess, os\n"
        "for fn in [lambda: socket.socket(), lambda: socket.create_connection(('example.com', 80)),\n"
        "           lambda: subprocess.run(['ls']), lambda: os.system('ls')]:\n"
        "    try:\n"
        "        fn()\n"
        "    except OSError:\n"
        "        continue\n"
        "    raise SystemExit(3)\n"
    )
    assert run_program(program)["passed"]


def test_sandbox_env_has_no_proxy_and_runs_in_temp_dir(monkeypatch):
    monkeypatch.setenv("HTTPS_PROXY", "http://proxy.invalid:1")
    monkeypatch.setenv("http_proxy", "http://proxy.invalid:1")
    program = (
        "import os\n"
        "for name in ['HTTPS_PROXY', 'http_proxy', 'HTTP_PROXY', 'https_proxy', 'ALL_PROXY']:\n"
        "    assert name not in os.environ, name\n"
        "assert os.path.exists('program.py')\n"
    )
    assert run_program(program)["passed"]


def test_run_programs_keeps_order():
    results = run_programs(["pass\n", "raise SystemExit(1)\n", "pass\n"], workers=3)
    statuses = []
    for result in results:
        statuses.append(result["status"])
    assert statuses == ["passed", "failed", "passed"]


class FakeTokenizer:
    """Tiny stand-in for the grading path: no chat template, BOS empty."""

    chat_template = None
    bos_token = None


def test_render_prompt_without_template():
    text = skills_eval.render_prompt(FakeTokenizer(), "2+2?")
    assert text == "Question: 2+2?\nAnswer:"


def test_grade_outputs_for_math_and_code():
    items = [{"id": "a", "gold": 7.0}, {"id": "b", "gold": 3.0}]
    records = skills_eval.grade_outputs("gsm8k", items, ["The answer is 7.", "The answer is 4."])
    assert [records[0]["correct"], records[1]["correct"]] == [True, False]
    items = [{"id": "m", "gold": "\\frac{1}{2}"}]
    records = skills_eval.grade_outputs("math500", items, ["so \\boxed{0.5}"])
    assert records[0]["correct"]
    items = [{"id": "h", "prompt": PROMPT, "test": TEST, "entry_point": "add"}]
    records = skills_eval.grade_outputs("humaneval", items, ["```python\ndef add(xs):\n    return sum(xs)\n```"])
    assert records[0]["correct"] and records[0]["pred"] == "passed"


def tiny_lm():
    from transformers import LlamaConfig, LlamaForCausalLM

    from tests.test_align import tiny_tokenizer

    torch.manual_seed(0)
    model = LlamaForCausalLM(LlamaConfig(
        hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        vocab_size=len(tiny_tokenizer()), max_position_embeddings=128,
    )).eval()
    # No EOS token id inside the vocabulary range would stop it early by chance.
    model.generation_config.eos_token_id = None
    return model, tiny_tokenizer()


def test_generate_texts_batched_matches_one_by_one():
    model, tokenizer = tiny_lm()
    prompts = ["the cat sat", "a dog", "the big red hat on"]
    batched = skills_eval.generate_texts(model, tokenizer, prompts, max_new_tokens=5, batch_size=3)
    single = []
    for prompt in prompts:
        single.append(skills_eval.generate_texts(model, tokenizer, [prompt], max_new_tokens=5, batch_size=1)[0])
    assert len(batched) == 3
    # Left padding with a mask keeps greedy outputs the same in float32.
    assert batched == single


def test_stop_on_strings_marks_rows():
    from tests.test_align import tiny_tokenizer

    tokenizer = tiny_tokenizer()
    rows = tokenizer(["the cat sat on", "the cat dog ran"], return_tensors="pt").input_ids
    criterion = skills_eval.StopOnStrings(tokenizer, ["sat"], prompt_length=1)
    assert criterion(rows, None).tolist() == [True, False]
    # Text inside the prompt does not count.
    criterion = skills_eval.StopOnStrings(tokenizer, ["the"], prompt_length=1)
    assert criterion(rows, None).tolist() == [False, False]


def test_stop_strings_cut_the_output():
    model, tokenizer = tiny_lm()
    full = skills_eval.generate_texts(model, tokenizer, ["the cat"], max_new_tokens=8, batch_size=1)[0]
    words = full.split()
    assert len(words) >= 1
    stop = words[0]
    cut = skills_eval.generate_texts(model, tokenizer, ["the cat"], max_new_tokens=8, batch_size=1,
                                     stop_strings=[stop])[0]
    assert stop not in cut
    assert full.startswith(cut)
    assert len(cut) < len(full)


def test_score_choices_and_suite_on_tiny_model():
    model, tokenizer = tiny_lm()
    items = {
        "medmcqa": [{"id": "q1", "prompt": "the cat sat on", "gold": 0}, {"id": "q2", "prompt": "a dog", "gold": 1}],
        "mmlu": [{"id": "q3", "prompt": "the hat", "gold": 2}],
    }
    result = skills_eval.evaluate_suite(model, tokenizer, skills=["medmcqa", "mmlu"], items=items, log=False)
    assert set(result["scores"]) == {"medmcqa", "mmlu"}
    assert result["skills"]["medmcqa"]["n"] == 2
    for record in result["skills"]["medmcqa"]["records"]:
        assert abs(sum(record["probs"]) - 1.0) < 1e-3
    expected = (result["scores"]["medmcqa"] + result["scores"]["mmlu"]) / 2
    assert abs(result["suite"] - expected) < 1e-9
    slim = skills_eval.strip_outputs(result, keep_output=False)
    assert slim["skills"]["mmlu"]["n"] == 1


def test_model_params_counts_embedding_as_lookup():
    model, _ = tiny_lm()
    total, run = skills_eval.model_params(model)
    embed = model.get_input_embeddings().weight.numel()
    assert run == total - embed


# Regression tests from the round 16 review.


def test_gsm8k_answer_in_inline_math_box_beats_an_earlier_answer_is():
    text = "First the answer is 5 apples. So the answer is \\(\\boxed{42}\\)."
    assert gsm8k_prediction(text) == 42.0
    assert gsm8k_prediction("The final answer is $\\boxed{18}$.") == 18.0
    assert gsm8k_prediction("The answer is \\$5.") == 5.0


def test_math_answers_with_dollars_units_and_set_membership():
    assert answers_match("18.90", "\\$18.90")
    assert answers_match("32", "\\$32")
    assert answers_match("32348", "\\$32,\\!348")
    assert answers_match("5.4", "5.4 \\text{ cents}")
    assert answers_match("864", "864 \\mbox{ inches}^2")
    assert answers_match("\\frac{270}{7}", "\\frac{270}7\\text{ degrees}")
    assert answers_match("[-2,7]", "x \\in [-2,7]")
    assert answers_match("(2,\\infty)", "(2,\\infty)")
    assert not answers_match("5.5", "5.4 \\text{ cents}")
    # Text answers keep their text.
    assert answers_match("\\text{Evelyn}", "Evelyn")
    assert not answers_match("\\text{Navin}", "\\text{Evelyn}")


def test_main_block_does_not_run_in_the_test_program():
    output = ("```python\ndef add(xs):\n    return sum(xs)\n\n"
              "if __name__ == \"__main__\":\n    xs = input()\n    assert add([1]) == 5\n```")
    program = assemble_humaneval(PROMPT, output, TEST, "add")
    assert "__main__" not in program
    assert run_program(program)["passed"]
    output = "```python\ndef square(n):\n    return n * n\nif __name__ == '__main__':\n    print(input())\n```"
    assert run_program(assemble_mbpp(output, ["assert square(3) == 9"]))["passed"]


def test_extract_code_prefers_the_block_with_the_entry_point():
    text = "```python\ndef helper():\n    return 0\n```\nthen\n```python\ndef add(xs):\n    return sum(xs)\n```"
    assert skills_eval.extract_code(text, "add") == "def add(xs):\n    return sum(xs)\n"
    assert run_program(assemble_humaneval(PROMPT, text, TEST, "add"))["passed"]


def test_early_exit_with_status_zero_does_not_pass():
    # A wrong function plus sys.exit(0) used to pass every test.
    output = "```python\nimport sys\ndef add(xs):\n    return 0\nsys.exit(0)\n```"
    result = run_program(assemble_humaneval(PROMPT, output, TEST, "add"))
    assert not result["passed"] and result["status"] == "failed"
    assert not run_program("import os\nos._exit(0)\n")["passed"]
    assert not run_program("def check(c):\n    raise SystemExit(0)\ncheck(1)\n")["passed"]


def test_future_import_still_runs_after_the_guard():
    program = "from __future__ import annotations\ndef f(x: int) -> int:\n    return x\nassert f(2) == 2\n"
    assert run_program(program)["passed"]


def test_output_flood_is_not_kept_in_memory_and_times_out_on_time():
    start = time.time()
    result = run_program("while True:\n    print('x' * 10000)\n", timeout=1.0)
    assert result["status"] == "timeout"
    assert time.time() - start < 6
    flood = run_program("import sys\nwhile True:\n    sys.stderr.write('y' * 10000)\n", timeout=2.0)
    assert not flood["passed"] and len(flood["stderr"]) <= 600


def test_sandbox_blocks_posix_spawn():
    program = (
        "import os\n"
        "try:\n"
        "    os.posix_spawn('/bin/sleep', ['sleep', '30'], {})\n"
        "except OSError:\n"
        "    pass\n"
        "else:\n"
        "    raise SystemExit(3)\n"
    )
    assert run_program(program)["passed"]
