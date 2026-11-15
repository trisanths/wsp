"""No round script that trains or fits may read a reported benchmark through load_dataset.

Scoring goes through wsp.skills_eval, which is exempt and is not listed here.
Every new round script that trains must be added to SCRIPTS below.
"""

import re
from pathlib import Path

import pytest

from wsp import healdata

# Every new round script that trains or fits anything must be listed here.
SCRIPTS = ["modal_round16c.py", "modal_round16d.py", "modal_round18.py", "modal_round19.py"]

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


def quoted_ids(call):
    return [item.lower() for item in re.findall(r"[\"']([^\"'\n]+)[\"']", call)]


@pytest.mark.parametrize("script", SCRIPTS)
def test_script_has_no_banned_load_dataset(script):
    text = (ROOT / script).read_text()
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
