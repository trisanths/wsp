"""Skills, their heal and probe text, held-out sets, answer checks, and the objectives.

This module is pure Python, so the CPU tests can import it. The data
loaders and model scoring that need datasets and torch are in
composer/evalsets.py.

Rules (owner, 2026-11-14):
- Nothing that trains or fits may use a reported benchmark in any split.
  Heal text comes from the wsp.healdata pools only.
- Programs are not selected on benchmark data. The practice feedback is the
  benchmark-free probe of wsp/probe.py on held-out pool text.
- Benchmarks (the held-out sets below) score the final model once, in the
  final report.
"""

import re

from wsp import healdata

LAW_SUBJECTS = ["international_law", "jurisprudence", "professional_law"]

# Ranges of one seeded healdata pool (see composer/evalsets.py). They never overlap:
# heal text is pairs HEAL_START to HEAL_START + HEAL_COUNT, probe text is the pairs after it.
HEAL_START = 2000
HEAL_COUNT = 3000
PROBE_START = HEAL_START + HEAL_COUNT
PROBE_COUNT = 128

DEFAULT_DRIFT_TOLERANCE = 0.05
# Objective points lost per nat of drift above the tolerance.
DRIFT_PENALTY_PER_NAT = 10.0

# skill -> held-out sets, the probe pool, the heal data, and words that tag a mined part for the skill.
# Each set name is defined in composer/evalsets.py. Knowledge and law have no approved pool of their
# own, so they probe on the general pool.
SKILLS = {
    "math": {
        "heldout": ["gsm8k_test", "math500"],
        "pool": "math",
        "heal_data": ["math"],
        "tags": ["math", "gsm", "arith"],
    },
    "code": {
        "heldout": ["humaneval", "mbpp_test"],
        "pool": "code",
        "heal_data": ["code"],
        "tags": ["code", "coding", "mbpp", "humaneval", "program"],
    },
    "medicine": {
        "heldout": ["medmcqa_r10"],
        "pool": "medicine",
        "heal_data": ["medicine"],
        "tags": ["medic", "med", "bio", "clinic"],
    },
    "knowledge": {
        "heldout": ["mmlu_dev"],
        "pool": "general",
        "heal_data": ["general"],
        "tags": ["knowledge", "mmlu", "general"],
    },
    "law": {
        "heldout": ["mmlu_law"],
        "pool": "general",
        "heal_data": ["general"],
        "tags": ["law", "legal"],
        "subjects": LAW_SUBJECTS,
    },
}

# What each held-out set is, in words, for the system prompt and the results file.
SET_NOTES = {
    "gsm8k_test": "GSM8K test (1319), raw prompt, greedy 256 tokens",
    "math500": "MATH-500 test (500), chat prompt, answer in \\boxed{}, greedy 768 tokens",
    "humaneval": "HumanEval (164), chat prompt, run against the tests",
    "mbpp_test": "MBPP test (500), chat prompt, run against the asserts",
    "medmcqa_r10": "the 1000 round 10 MedMCQA validation items, log-prob of A to D",
    "mmlu_dev": "the 1000 round 6 MMLU test items, log-prob of A to D",
    "mmlu_law": "MMLU test items from international_law, jurisprudence, professional_law, log-prob of A to D",
}

# Heal data names a program may use. Each is a wsp.healdata pool, or an even mix of them.
HEAL_POOLS = ["math", "code", "medicine", "general"]
MIX_NAME = "composer_mix"
HEAL_DATA = HEAL_POOLS + [MIX_NAME]


def heal_pools_for(name):
    """The healdata pool names behind a heal data name. Raise ValueError for an unknown name."""
    if name == MIX_NAME:
        return list(HEAL_POOLS)
    if name in HEAL_POOLS:
        return [name]
    raise ValueError(f"heal data {name!r} is unknown; the names are " + ", ".join(HEAL_DATA))


def heal_source_ids(name):
    """Every dataset id that training on a heal data name reads."""
    ids = []
    for pool in heal_pools_for(name):
        for source in healdata.SOURCES[pool]:
            if source[0] not in ids:
                ids.append(source[0])
    return ids


def check_heal_sources(name):
    """Raise ValueError when a heal data name reads a reported benchmark. Return the dataset ids."""
    return healdata.check_sources(heal_source_ids(name))


def heldout_sets(skills):
    names = []
    for skill in skills:
        for name in SKILLS[skill]["heldout"]:
            if name not in names:
                names.append(name)
    return names


def skill_scores(set_scores, skills, kind="heldout"):
    """Average the held-out set scores of each skill. A skill with no scored set gets None."""
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


def probe_summary(probe, weights, tolerance=DEFAULT_DRIFT_TOLERANCE):
    """Turn a probe report into the practice feedback and the one objective number.

    probe: {"skills": {skill: {"transfer": float or None, ...}}, "drift": {"base_kl": float}}.
    weights: {skill: weight} for the skills the objective covers. A skill with no donor
    or no gap has transfer 0. The objective is the weighted mean transfer minus
    DRIFT_PENALTY_PER_NAT times the drift above the tolerance.
    """
    probe = probe or {}
    rows = probe.get("skills") or {}
    transfers = {}
    for skill in weights:
        row = rows.get(skill) or {}
        value = row.get("transfer")
        if value is None:
            value = 0.0
        transfers[skill] = float(value)
    drift = (probe.get("drift") or {}).get("base_kl")
    if drift is None:
        drift = 0.0
    excess = max(0.0, float(drift) - float(tolerance))
    penalty = DRIFT_PENALTY_PER_NAT * excess
    mean = objective(transfers, weights)
    return {"transfer": transfers, "mean_transfer": mean, "drift": float(drift), "drift_tolerance": tolerance,
            "penalty": penalty, "objective": mean - penalty}


def skill_names():
    return sorted(SKILLS)


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
