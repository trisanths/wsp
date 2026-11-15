import importlib.util
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_demo():
    if "modal_demo" in sys.modules:
        return sys.modules["modal_demo"]
    spec = importlib.util.spec_from_file_location("modal_demo", os.path.join(ROOT, "modal_demo.py"))
    module = importlib.util.module_from_spec(spec)
    sys.modules["modal_demo"] = module
    spec.loader.exec_module(module)
    return module


def fenced(lang, code):
    return f"Here you go.\n```{lang}\n{code}\n```\n"


MERGE_GOOD = '''import pytest

def merge_intervals(intervals):
    out = []
    for s, e in sorted(intervals):
        if out and s <= out[-1][1]:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return out

def test_basic():
    assert merge_intervals([[1, 2], [2, 3]]) == [[1, 3]]
'''


def test_merge_intervals_grades():
    demo = load_demo()
    assert demo.grade_merge_intervals(fenced("python", MERGE_GOOD))["passed"] is True
    no_tests = MERGE_GOOD.split("def test_basic")[0]
    assert demo.grade_merge_intervals(fenced("python", no_tests))["passed"] is False
    wrong = MERGE_GOOD.replace("s <= out[-1][1]", "s < out[-1][1]")
    assert demo.grade_merge_intervals(fenced("python", wrong))["passed"] is False


def test_moving_average_grades():
    demo = load_demo()
    good = "def moving_average(values, window):\n    return [sum(values[i:i+window]) / window for i in range(len(values) - window + 1)]\n"
    assert demo.grade_moving_average(fenced("python", good))["passed"] is True
    assert demo.grade_moving_average(fenced("python", demo.AVERAGE_CODE))["passed"] is False


def test_sql_grades():
    demo = load_demo()
    good = ("SELECT c.name, SUM(o.total) AS total_spend FROM customers c JOIN orders o ON o.customer_id = c.id "
            "WHERE c.country = 'US' AND strftime('%Y', o.created_at) = '2024' GROUP BY c.id "
            "ORDER BY total_spend DESC LIMIT 3;")
    assert demo.grade_sql(fenced("sql", good))["passed"] is True
    wrong = good.replace("'2024'", "'2025'")
    assert demo.grade_sql(fenced("sql", wrong))["passed"] is False
    assert demo.grade_sql(fenced("sql", "SELEKT 1"))["passed"] is False


def test_cart_grades():
    demo = load_demo()
    assert demo.grade_cart(fenced("python", demo.CART_CODE))["passed"] is True
    broken = demo.CART_CODE.replace('"HALF": ', "").replace('t * 0.5', 't * 0.4')
    assert demo.grade_cart(fenced("python", broken))["passed"] is False


def test_bash_grades():
    demo = load_demo()
    good = "awk '{print $1}' access.log | sort | uniq -c | sort -rn | head -5"
    assert demo.grade_bash(fenced("bash", good))["passed"] is True
    assert demo.grade_bash(fenced("bash", "awk '{print $1}' access.log | sort | uniq -c | head -5"))["passed"] is False
    assert demo.grade_bash("no code at all")["passed"] is False


def test_unclosed_fence_counts_and_side_by_side(tmp_path):
    demo = load_demo()
    assert demo.code_blocks("text\n```python\nx = 1\n")[0][1].strip() == "x = 1"
    rows = [{"id": p["id"], "kind": p["kind"], "title": p["title"], "prompt": p["text"], "output": "```\nok\n```",
             "new_tokens": 3, "finish_reason": "stop", "passed": None, "grade_detail": "d"} for p in demo.PROMPTS]
    results = {name: {"cell": name, "label": "x", "weights_gib_on_disk": 1.0, "rows": rows} for name in demo.CELLS}
    paths = demo.write_side_by_side(results, str(tmp_path))
    assert all(os.path.exists(p) for p in paths)
    assert demo.write_side_by_side({"full_30b": results["full_30b"]}, str(tmp_path)) is None
    assert len(demo.select_prompts(True)) == 1 and len(demo.select_prompts(False)) == 8
