"""No round script that trains or fits may read a reported benchmark through load_dataset.

Scoring goes through wsp.skills_eval, which is exempt and is not listed here.
Every new round script that trains must be added to SCRIPTS below.
"""

import re
from pathlib import Path

import pytest

from wsp import healdata

# Every new round script that trains or fits anything must be listed here.
SCRIPTS = ["modal_round16c.py", "modal_round16d.py", "modal_round18.py", "modal_round19.py", "modal_round19c.py", "modal_round20.py",
           "modal_round21.py", "modal_round22.py"]

# modal_round21.py selects a mix (a search is selection, and the rule covers selection). Its default
# objective is "probe", which reads only wsp.healdata pools. Its old "practice" objective reads benchmark
# train splits, and every one of those load_dataset calls lives in the single function below. The scan
# exempts that function and nothing else. Other tests in this file check that the probe path never reaches it.
EXEMPT_FUNCTIONS = {"modal_round21.py": ["practice_items_for"]}
PROBE_PATH = ["pool_choice", "dedupe", "load_pool_splits", "probe_texts_for", "fill_caches", "build_probe_caches",
              "setup_probe_selection", "arm_row", "probe_arms", "probe_go", "final_items_for"]

ROOT = Path(__file__).resolve().parent.parent


def load_dataset_calls(text):
    """Return the text of every load_dataset( call, from the name to its closing parenthesis."""
    calls = []
    for match in re.finditer(r"load_dataset\(", text):
        depth = 0
        end = match.end() - 1
        for index in range(end, len(text)):
            if text[index] == "(":
                depth += 1
            elif text[index] == ")":
                depth -= 1
                if depth == 0:
                    calls.append(text[match.start():index + 1])
                    break
    return calls


def function_body(text, name):
    """Return the source of one top-level function, from its def line to the next top-level statement."""
    match = re.search(r"^def " + re.escape(name) + r"\(", text, re.M)
    assert match is not None, name + " not found"
    line_end = text.index("\n", match.end())
    rest = text[line_end:]
    stop = re.search(r"^\S", rest, re.M)
    end = len(text) if stop is None else line_end + stop.start()
    return text[match.start():end]


def without_exempt(script, text):
    for name in EXEMPT_FUNCTIONS.get(script, []):
        text = text.replace(function_body(text, name), "")
    return text


def quoted_ids(call):
    return [item.lower() for item in re.findall(r"[\"']([^\"'\n]+)[\"']", call)]


@pytest.mark.parametrize("script", SCRIPTS)
def test_script_has_no_banned_load_dataset(script):
    text = without_exempt(script, (ROOT / script).read_text())
    bad = []
    for call in load_dataset_calls(text):
        for name in quoted_ids(call):
            if healdata.is_banned(name):
                bad.append(name)
    assert bad == [], script + " loads a banned dataset: " + ", ".join(bad)


@pytest.mark.parametrize("script", SCRIPTS)
def test_script_checks_its_sources(script):
    text = (ROOT / script).read_text()
    assert "healdata.check_sources(" in text
    assert "no benchmark data trains anything" in text.lower()


def test_detector_finds_a_banned_call():
    sample = 'rows = load_dataset(\n    "openai/gsm8k", "main", split="train")\n'
    names = []
    for call in load_dataset_calls(sample):
        names += quoted_ids(call)
    assert any(healdata.is_banned(name) for name in names)


def test_round21_probe_is_the_default_objective():
    text = (ROOT / "modal_round21.py").read_text()
    assert 'objective: str = "probe"' in text
    assert 'objective="probe"' in function_body(text, "run_family")


def test_round21_probe_path_loads_no_dataset():
    text = (ROOT / "modal_round21.py").read_text()
    for name in PROBE_PATH:
        body = function_body(text, name)
        assert "load_dataset(" not in body, name + " loads a dataset"
        assert "practice_items_for(" not in body, name + " reaches the practice path"
    run_body = function_body(text, "run_family")
    assert "load_dataset(" not in run_body
    assert "practice_items_for(" not in run_body


def test_round21_benchmark_loads_are_in_one_function():
    text = (ROOT / "modal_round21.py").read_text()
    exempt = function_body(text, "practice_items_for")
    assert len(load_dataset_calls(exempt)) > 0
    rest = text.replace(exempt, "")
    assert load_dataset_calls(rest) == []
    inside = function_body(text, "setup_practice_selection")
    assert inside.count("practice_items_for(") == 1
    assert rest.count("practice_items_for(") == 1
