"""Skills, their practice and held-out sets, answer checks, and the objective.

This module is pure Python, so the CPU tests can import it. The data
loaders and model scoring that need datasets and torch are in
composer/evalsets.py.

Practice sets come only from TRAIN (or validation) splits and are small,
100 to 200 items per skill. Held-out sets are test splits, scored once
after the composer finishes, with fixed settings.
"""

import re

LAW_SUBJECTS = ["international_law", "jurisprudence", "professional_law"]

# skill -> practice sets, held-out sets, and the atlas score names that
# describe it. Each set name is defined in composer/evalsets.py.
SKILLS = {
    "math": {
        "practice": ["gsm8k_practice", "math_practice"],
        "heldout": ["gsm8k_test", "math500"],
        "atlas_scores": ["gsm8k", "gsm8k_raw", "math500", "math"],
        "heal_data": ["gsm8k_train", "math_train"],
    },
    "code": {
        "practice": ["mbpp_practice"],
        "heldout": ["humaneval", "mbpp_test"],
        "atlas_scores": ["humaneval", "mbpp", "code"],
        "heal_data": ["mbpp_train"],
    },
    "medicine": {
        "practice": ["medmcqa_practice"],
        "heldout": ["medmcqa_r10"],
        "atlas_scores": ["medmcqa", "medqa", "medicine"],
        "heal_data": ["medmcqa_train"],
    },
    "knowledge": {
        "practice": ["mmlu_practice"],
        "heldout": ["mmlu_dev"],
        "atlas_scores": ["mmlu", "mmlu_dev", "mmlu_heldout", "knowledge"],
        "heal_data": ["mmlu_aux"],
    },
    "law": {
        "practice": ["law_practice"],
        "heldout": ["mmlu_law"],
        "atlas_scores": ["mmlu_law", "law"],
        "heal_data": ["mmlu_aux"],
        "subjects": LAW_SUBJECTS,
    },
}

# What each set is, in words, for the system prompt and the results file.
SET_NOTES = {
    "gsm8k_practice": "100 GSM8K train items (the fixed search split), raw 'Question: ... Answer:' prompt, greedy 256 tokens",
    "math_practice": "100 MATH train items, chat prompt, answer in \\boxed{}, greedy 768 tokens",
    "mbpp_practice": "150 MBPP train items, chat prompt, run against the item's asserts",
    "medmcqa_practice": "200 MedMCQA train items, log-prob of A to D",
    "mmlu_practice": "200 MMLU validation items, log-prob of A to D",
    "law_practice": "up to 200 MMLU validation and dev items from the law subjects, log-prob of A to D",
    "gsm8k_test": "GSM8K test (1319), raw prompt, greedy 256 tokens",
    "math500": "MATH-500 test (500), chat prompt, answer in \\boxed{}, greedy 768 tokens",
    "humaneval": "HumanEval (164), chat prompt, run against the tests",
    "mbpp_test": "MBPP test (500), chat prompt, run against the asserts",
    "medmcqa_r10": "the 1000 round 10 MedMCQA validation items, log-prob of A to D",
    "mmlu_dev": "the 1000 round 6 MMLU test items, log-prob of A to D",
    "mmlu_law": "MMLU test items from international_law, jurisprudence, professional_law, log-prob of A to D",
}

# Heal data names the compiler knows (wsp.pairs and wsp.data), plus the
# ones modal_composer.py registers from composer/evalsets.py.
BUILTIN_HEAL_DATA = ["gsm8k_train", "mmlu_aux", "trivia_train"]
COMPOSER_HEAL_DATA = ["math_train", "mbpp_train", "medmcqa_train", "composer_mix"]


def skill_names():
    return sorted(SKILLS)


def practice_sets(skills):
    """Return the practice set names for a list of skills, in order."""
    names = []
    for skill in skills:
        for name in SKILLS[skill]["practice"]:
            if name not in names:
                names.append(name)
    return names


def heldout_sets(skills):
    names = []
    for skill in skills:
        for name in SKILLS[skill]["heldout"]:
            if name not in names:
                names.append(name)
    return names


def skill_scores(set_scores, skills, kind):
    """Average the set scores of each skill. kind is "practice" or "heldout".

    A skill with no scored set gets None.
    """
    out = {}
    for skill in skills:
        values = []
        for name in SKILLS[skill][kind]:
            value = set_scores.get(name)
            if value is not None:
                values.append(value)
        if len(values) == 0:
            out[skill] = None
        else:
            out[skill] = sum(values) / len(values)
    return out


def objective(scores, weights):
    """The weighted mean of skill scores. A missing skill counts as 0."""
    total = 0.0
    weight_sum = 0.0
    for skill in weights:
        weight = float(weights[skill])
        value = scores.get(skill)
        if value is None:
            value = 0.0
        total += weight * value
        weight_sum += weight
    if weight_sum == 0:
        return 0.0
    return total / weight_sum


# Answer checks.

def final_number(text):
    """The number after '####', else the last number in the text."""
    if text is None:
        return None
    match = re.search(r"####\s*(-?[\d,]*\.?\d+)", text)
    if match is None:
        numbers = re.findall(r"-?[\d,]*\.?\d+", text)
        if len(numbers) == 0:
            return None
        value = numbers[-1]
    else:
        value = match.group(1)
    value = value.replace(",", "").rstrip(".")
    try:
        return float(value)
    except ValueError:
        return None


def gsm8k_correct(output, gold_answer):
    # Cut the output where the model starts a new question.
    output = output.split("Question:")[0]
    predicted = final_number(output)
    gold = final_number(gold_answer)
    if predicted is None or gold is None:
        return False
    return abs(predicted - gold) < 1e-6


def last_boxed(text):
    """Return the contents of the last \\boxed{...} in text, or None."""
    if text is None:
        return None
    start = text.rfind("\\boxed")
    if start < 0:
        start = text.rfind("\\fbox")
        if start < 0:
            return None
    open_at = text.find("{", start)
    if open_at < 0:
        return None
    depth = 0
    for index in range(open_at, len(text)):
        char = text[index]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[open_at + 1:index]
    return None


def normalize_math(answer):
    """A light normal form for MATH answers, so 1/2 and \\frac{1}{2} style forms match more often."""
    if answer is None:
        return None
    text = answer.strip()
    replacements = [("\\left", ""), ("\\right", ""), ("\\!", ""), ("\\,", ""), ("\\;", ""), ("\\dfrac", "\\frac"),
                    ("\\tfrac", "\\frac"), ("^\\circ", ""), ("^{\\circ}", ""), ("\\%", ""), ("%", ""), ("$", ""),
                    ("\\text{", "{"), ("\\mbox{", "{"), (" ", "")]
    for old, new in replacements:
        text = text.replace(old, new)
    text = text.rstrip(".")
    # \frac12 -> \frac{1}{2}
    text = re.sub(r"\\frac(\d)(\d)", r"\\frac{\1}{\2}", text)
    # A plain integer or decimal: drop a trailing .0
    if re.fullmatch(r"-?\d+\.0+", text):
        text = text.split(".")[0]
    if text.startswith("{") and text.endswith("}") and text.count("{") == 1:
        text = text[1:-1]
    return text


def math_correct(output, gold_answer):
    predicted = normalize_math(last_boxed(output))
    gold = normalize_math(gold_answer)
    if predicted is None or gold is None:
        return False
    if predicted == gold:
        return True
    try:
        return abs(float(predicted) - float(gold)) < 1e-6
    except ValueError:
        return False


def extract_code(output, prompt_code=None, entry_point=None):
    """Take the first python code block, else the whole output.

    For HumanEval, when the block lacks the function header, the prompt
    code goes first.
    """
    match = re.search(r"```(?:python|py)?\n(.*?)```", output, re.DOTALL)
    if match is not None:
        code = match.group(1)
    else:
        code = output
    if entry_point is not None and prompt_code is not None and ("def " + entry_point) not in code:
        code = prompt_code + code
    return code
