"""The studio GSM8K scorer: items, answer extraction and routing in score_all."""

import importlib.util
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)


def load_module():
    path = os.path.join(REPO, "studio", "modal_studio.py")
    spec = importlib.util.spec_from_file_location("modal_studio_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


modal_studio = load_module()


class FakeTask:
    name = "gsm8k"

    def split_items(self, split, seed):
        rows = []
        for index in range(10):
            rows.append({"id": split + "-" + str(index), "question": "q" + str(index), "answer": float(index)})
        return rows


def fake_get_task(name):
    return FakeTask()


def test_items_use_the_harness_split_and_prefix(monkeypatch):
    from wsp_eval import tasks as tasks_module

    monkeypatch.setattr(tasks_module, "get_task", fake_get_task)
    search = modal_studio.gsm8k_items("search", 1.0)
    held = modal_studio.gsm8k_items("heldout", 1.0)
    assert len(search) == 10 and search[0]["id"] == "search-0" and search[3]["gold"] == 3.0
    assert held[0]["id"] == "heldout-0"
    assert modal_studio.gsm8k_items("search", 0.34)[-1]["id"] == "search-3"
    assert len(modal_studio.gsm8k_items("search", 0.01)) == 1


def test_grading_follows_the_team_extraction_rules():
    items = [{"gold": 18.0}, {"gold": 1250.0}, {"gold": 7.0}, {"gold": 3.0}, {"gold": 5.0}]
    outputs = [
        "She has 16 - 3 - 4 = 9 eggs. 9 * 2 = 18. The answer is 18.",
        "The total is $1,250. The answer is $1,250.",
        "Work: 3 + 4 = 7\n#### 7",
        "so we get \\boxed{3}",
        "The answer is 6.",
    ]
    assert modal_studio.grade_gsm8k(items, outputs) == 4


def test_missing_number_is_wrong():
    assert modal_studio.grade_gsm8k([{"gold": 2.0}], ["no digits here"]) == 0


def test_score_all_routes_gsm8k_and_mmlu_away_from_the_harness(monkeypatch):
    calls = []

    def fake_gsm8k(path, split, fraction):
        calls.append("gsm8k")
        return 0.7, 200

    def fake_mmlu(path, split, fraction):
        calls.append("mmlu")
        return 0.5, 1000

    def fake_harness(path, split, tasks, fraction, note):
        calls.append(("harness", list(tasks)))
        return {"mbpp": 0.4}, {"mbpp": 10}

    monkeypatch.setattr(modal_studio, "score_gsm8k", fake_gsm8k)
    monkeypatch.setattr(modal_studio, "score_mmlu_logprob", fake_mmlu)
    monkeypatch.setattr(modal_studio, "run_harness", fake_harness)
    monkeypatch.setattr(modal_studio, "free_gpu", lambda: None)
    scores, counts = modal_studio.score_all("m", "search", ["gsm8k", "mmlu", "mbpp"], 1.0, "n")
    assert scores == {"gsm8k": 0.7, "mmlu": 0.5, "mbpp": 0.4}
    assert counts["gsm8k"] == 200
    assert ("harness", ["mbpp"]) in calls and "gsm8k" in calls


def test_gsm8k_alone_never_calls_the_harness(monkeypatch):
    monkeypatch.setattr(modal_studio, "score_gsm8k", lambda path, split, fraction: (0.6, 5))

    def boom(*args):
        raise AssertionError("harness called")

    monkeypatch.setattr(modal_studio, "run_harness", boom)
    scores, counts = modal_studio.score_all("m", "search", ["gsm8k"], 1.0, "n")
    assert scores == {"gsm8k": 0.6} and counts == {"gsm8k": 5}


def test_scoring_note_names_the_gsm8k_protocol():
    assert "gsm8k" in modal_studio.SCORING_NOTE and "512" in modal_studio.SCORING_NOTE
