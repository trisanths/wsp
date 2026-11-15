"""The skill atlas: which model is best at which skill, and where the skill lives.

For each model in atlas/zoo.json the atlas records three things:

- skills: a score per skill from a short screen (SKILLS below).
- layer_maps: per decoder layer, how much more the layer matters for a
  skill's text than for general text, and the contiguous layer blocks
  that hold most of that skill-specific effect.
- expert_maps: for MoE models, per layer, the experts that the router
  picks more often on skill text than on general text.

The compiler and planner read the atlas with best_models_for (the
strongest donors for a skill) and best_block_for (the best layer block for
a skill under constraints such as hidden size or block length).

Person A's miner (person_a/mine) finds fine-grained parts (neuron
clusters, heads, experts) by ablation with an agent. The atlas works one
level up, at model and layer level, and catalog_rows writes its blocks
and experts in the same catalog row format, so the compiler can graft
them by id.

Entry format (one JSON file per model, atlas/entries/<org>__<name>.json):

    {"model": "Qwen/Qwen2.5-7B-Instruct",
     "skills": {"knowledge": 0.715, ...},
     "skill_details": {"knowledge": {"score", "count", "by_subject"}, ...},
     "layer_maps": {"math": {"scores": [...], "influence": [...],
                             "blocks": [[a, b], ...], "block_mass": [...],
                             "heldout": {"blocks", "select_mass", "check_mass",
                                         "split_half_r"}}},
     "expert_maps": {"math": {"12": [3, 17, 40], ...}} or null,
     "params_b": 7.62, "layers": 28, "hidden": 3584,
     "family": "qwen2", "measured_at": "2026-11-14T12:00:00Z"}

Layer blocks are [first, last] with both ends included, the same
convention as block records in the catalog. Expert map layer keys are
strings, because JSON object keys are strings.
"""

import concurrent.futures
import datetime
import json
import math
import os
import random
import re
import subprocess
import sys
import tempfile

import torch

from wsp import mcq
from wsp.parts import find_decoder_layers

# ---------------------------------------------------------------------------
# Datasets. Every loader goes through mcq.load_rows, so tests can replace the
# datasets module in one place.


def load_rows(path, config, split):
    return mcq.load_rows(path, config, split)


def shuffled(rows, seed):
    """Row indexes of rows in an order shuffled with random.Random(seed)."""
    order = list(range(len(rows)))
    random.Random(seed).shuffle(order)
    return order


def mcq_item(item_id, subject, question, choices, gold):
    """One multiple-choice item in the wsp.mcq format."""
    return {
        "id": item_id,
        "subject": subject,
        "question": question,
        "choices": choices,
        "prompt": mcq.mcq_prompt(question, choices),
        "gold": gold,
    }


def mcq_text(item):
    """Item prompt plus the gold answer, as text for a layer map."""
    letter = mcq.LETTERS[item["gold"]]
    return item["prompt"] + " " + letter + ". " + str(item["choices"][item["gold"]])


def general_texts(count, seed=0, min_chars=300):
    """WikiText-103 validation paragraphs of at least min_chars characters."""
    rows = load_rows("Salesforce/wikitext", "wikitext-103-raw-v1", "validation")
    texts = []
    for index in shuffled(rows, seed):
        if len(texts) >= count:
            break
        text = rows[index]["text"].strip()
        if len(text) < min_chars:
            continue
        if text.startswith("="):
            continue
        texts.append(text)
    return texts


# Knowledge, medicine: MMLU and MedMCQA through wsp.mcq.

def knowledge_items(count, skip=0):
    # seed 0, skip 0, count 1000 is round 6's MMLU dev sample.
    return mcq.mcq_items("mmlu", "test", seed=0, count=count, skip=skip)


def knowledge_texts(count):
    # Map texts come from MMLU validation, so they never overlap the test screen.
    items = mcq.mcq_items("mmlu", "validation", seed=0, count=count)
    return texts_of(items, mcq_text)


def medicine_items(count, skip=0):
    # seed 7 is round 10's MedMCQA validation order. mcq_items drops rows with
    # no gold answer, which round 10 kept, so a few items can differ.
    return mcq.mcq_items("medmcqa", "validation", seed=7, count=count, skip=skip)


def medicine_texts(count):
    items = mcq.mcq_items("medmcqa", "train", seed=0, count=count)
    return texts_of(items, mcq_text)


# Law: MMLU law subjects, LegalBench tasks with a fixed label set, and
# CaseHOLD (LexGLUE), a 5-way holding multiple choice.

LEGALBENCH_TASKS = {
    "hearsay": ["Yes", "No"],
    "personal_jurisdiction": ["Yes", "No"],
    "proa": ["Yes", "No"],
    "ucc_v_common_law": ["UCC", "Common Law"],
    "abercrombie": ["generic", "descriptive", "suggestive", "arbitrary", "fanciful"],
}

LEGALBENCH_QUESTIONS = {
    "hearsay": "Is the following evidence hearsay?",
    "personal_jurisdiction": "Does the court have personal jurisdiction over the defendant?",
    "proa": "Does this statute create a private right of action?",
    "ucc_v_common_law": "Is this contract governed by the UCC or the common law?",
    "abercrombie": "Under the Abercrombie spectrum, how distinctive is this mark?",
}


def legalbench_items(task, split, limit=None):
    rows = load_rows("nguha/legalbench", task, split)
    labels = LEGALBENCH_TASKS[task]
    lowered = []
    for label in labels:
        lowered.append(label.lower())
    items = []
    for index in range(len(rows)):
        if limit is not None and len(items) >= limit:
            break
        row = rows[index]
        answer = str(row["answer"]).strip().lower()
        if answer not in lowered:
            continue
        if "text" in row and row["text"] is not None:
            body = row["text"]
        else:
            body = row["contract"]
        question = LEGALBENCH_QUESTIONS[task] + "\n" + str(body).strip()
        items.append(mcq_item(f"legalbench/{task}/{split}/{index}", "legalbench_" + task, question,
                              list(labels), lowered.index(answer)))
    return items


def casehold_items(split, count, seed=0):
    rows = load_rows("coastalcph/lex_glue", "case_hold", split)
    items = []
    for index in shuffled(rows, seed):
        if len(items) >= count:
            break
        row = rows[index]
        question = "Which holding completes the citing text?\n" + row["context"].strip()
        items.append(mcq_item(f"case_hold/{split}/{index}", "case_hold", question, list(row["endings"]),
                              int(row["label"])))
    return items


def law_items(count, skip=0):
    # Split count over the three sources: 30% MMLU law, 45% LegalBench, 25% CaseHOLD.
    mmlu_count = max(1, round(count * 0.30))
    bench_count = max(1, round(count * 0.45))
    case_count = max(1, count - mmlu_count - bench_count)
    items = mcq.mcq_items("mmlu", "test", seed=0, count=mmlu_count, subjects=mcq.DOMAINS["law"], skip=skip)
    per_task = max(1, bench_count // len(LEGALBENCH_TASKS))
    for task in LEGALBENCH_TASKS:
        items += legalbench_items(task, "test", limit=per_task)
    items += casehold_items("test", case_count)
    return items


def law_texts(count):
    half = max(1, count // 2)
    items = mcq.mcq_items("mmlu", "validation", seed=0, count=half, subjects=mcq.DOMAINS["law"])
    items += casehold_items("train", count - half)
    return texts_of(items, mcq_text)


# Science: ARC-Challenge.

def arc_items(split, count, seed=0):
    rows = load_rows("allenai/ai2_arc", "ARC-Challenge", split)
    items = []
    for index in shuffled(rows, seed):
        if len(items) >= count:
            break
        row = rows[index]
        labels = list(row["choices"]["label"])
        if row["answerKey"] not in labels:
            continue
        items.append(mcq_item(f"arc_challenge/{split}/{index}", "arc_challenge", row["question"],
                              list(row["choices"]["text"]), labels.index(row["answerKey"])))
    return items


def science_items(count, skip=0):
    return arc_items("test", count)


def science_texts(count):
    return texts_of(arc_items("train", count), mcq_text)


# Multilingual: Global-MMLU in several languages.

GLOBAL_MMLU_LANGS = ["de", "fr", "es", "zh", "ja", "hi", "sw", "ar"]


def global_mmlu_items(split, count, seed=0):
    per_lang = max(1, count // len(GLOBAL_MMLU_LANGS))
    items = []
    for lang in GLOBAL_MMLU_LANGS:
        rows = load_rows("CohereLabs/Global-MMLU", lang, split)
        taken = 0
        for index in shuffled(rows, seed):
            if taken >= per_lang:
                break
            row = rows[index]
            choices = [row["option_a"], row["option_b"], row["option_c"], row["option_d"]]
            answer = str(row["answer"]).strip()
            if answer not in mcq.LETTERS[:4]:
                continue
            items.append(mcq_item(f"global_mmlu/{lang}/{split}/{index}", "global_mmlu_" + lang, row["question"],
                                  choices, mcq.LETTERS.index(answer)))
            taken += 1
    return items


def multilingual_items(count, skip=0):
    return global_mmlu_items("test", count)


def multilingual_texts(count):
    return texts_of(global_mmlu_items("dev", count), mcq_text)


# Decision: fastino/fast-decisions, single-label classifications as
# multiple choice (the open benchmark the d1 model cards report).

FAST_DECISION_TASKS = ["support_intent", "support_topic", "document_type", "review_sentiment", "agent_handoff",
                       "email_triage", "ticket_route", "product_feedback", "banking_intent", "clinic_request",
                       "travel_request", "news_topic", "paper_field", "sports_recap", "restaurant_review",
                       "benefits_request", "screen_tags"]


def decision_all(seed=0):
    items = []
    for task in FAST_DECISION_TASKS:
        rows = load_rows("fastino/fast-decisions", task, "train")
        for index in range(len(rows)):
            row = rows[index]
            output = row["output"]
            if isinstance(output, str):
                output = json.loads(output)
            for number, decision in enumerate(output.get("classifications", [])):
                if decision.get("multi_label"):
                    continue
                labels = list(decision["labels"])
                truth = list(decision["true_label"])
                if len(truth) != 1 or truth[0] not in labels:
                    continue
                if len(labels) < 2 or len(labels) > len(mcq.LETTERS):
                    continue
                question = row["input"].strip() + "\nDecide: " + str(decision["task"])
                items.append(mcq_item(f"fast_decisions/{task}/{index}/{number}", "decision_" + task, question,
                                      labels, labels.index(truth[0])))
    order = list(range(len(items)))
    random.Random(seed).shuffle(order)
    out = []
    for index in order:
        out.append(items[index])
    return out


def decision_items(count, skip=0):
    return decision_all()[skip:skip + count]


def decision_row(item):
    """The dataset row an item comes from: "fast_decisions/<task>/<row>"."""
    parts = item["id"].split("/")
    return "/".join(parts[:3])


def decision_texts(count):
    # One row can hold several decisions (email_triage has four), so the map
    # texts skip every row that any item of the full screen comes from.
    items = decision_all()
    screen_rows = set()
    for item in items[:SKILLS["decision"]["count"]]:
        screen_rows.add(decision_row(item))
    chosen = []
    for item in reversed(items):
        if len(chosen) >= count:
            break
        if decision_row(item) in screen_rows:
            continue
        chosen.append(item)
    return texts_of(chosen, mcq_text)


# Math: GSM8K (generated, step by step) and a MATH-500 subset.

GSM8K_INSTRUCTION = "Solve the problem step by step. Write the final answer as a number after '####'."
MATH_INSTRUCTION = "Solve the problem step by step, and put your final answer within \\boxed{}."


def gsm8k_items(count, seed=0):
    rows = load_rows("openai/gsm8k", "main", "test")
    items = []
    for index in shuffled(rows, seed):
        if len(items) >= count:
            break
        row = rows[index]
        answer = row["answer"].split("####")[-1].strip()
        items.append({"id": f"gsm8k/test/{index}", "subject": "gsm8k", "kind": "number",
                      "prompt": row["question"].strip() + "\n\n" + GSM8K_INSTRUCTION, "answer": answer})
    return items


def math500_items(count, seed=0, skip=0):
    rows = load_rows("HuggingFaceH4/MATH-500", None, "test")
    items = []
    order = shuffled(rows, seed)
    for index in order[skip:]:
        if len(items) >= count:
            break
        row = rows[index]
        items.append({"id": f"math500/test/{index}", "subject": "math500", "kind": "boxed",
                      "prompt": row["problem"].strip() + "\n\n" + MATH_INSTRUCTION, "answer": row["answer"],
                      "solution": row["solution"]})
    return items


def math_items(count, skip=0):
    # 60% GSM8K, 40% MATH-500 (300 and 200 at the full count of 500).
    gsm_count = max(1, round(count * 0.6))
    return gsm8k_items(gsm_count) + math500_items(count - gsm_count)


def math_texts(count):
    rows = load_rows("openai/gsm8k", "main", "train")
    texts = []
    half = max(1, count // 2)
    for index in shuffled(rows, 0)[:half]:
        texts.append(rows[index]["question"].strip() + "\n" + rows[index]["answer"].strip())
    # The full screen uses the first 200 rows of the shuffled order.
    for item in math500_items(count - half, skip=200):
        texts.append(item["prompt"] + "\n" + item["solution"])
    return texts


# Code: MBPP sanitized, run against its asserts in a subprocess.

def mbpp_items(count, split="test"):
    rows = load_rows("google-research-datasets/mbpp", "sanitized", split)
    items = []
    for index in range(len(rows)):
        if len(items) >= count:
            break
        row = rows[index]
        tests = list(row["test_list"])
        prompt = (row["prompt"].strip() + "\nYour code should pass this test:\n" + tests[0]
                  + "\nWrite the Python code in a ```python code block.")
        items.append({"id": f"mbpp/{split}/{row['task_id']}", "subject": "mbpp", "kind": "code",
                      "prompt": prompt, "tests": tests, "imports": list(row["test_imports"]),
                      "reference": row["code"]})
    return items


def code_items(count, skip=0):
    return mbpp_items(count)


def code_texts(count):
    items = mbpp_items(count, split="train")
    if len(items) < count:
        items += mbpp_items(count - len(items), split="validation")
    texts = []
    for item in items:
        texts.append(item["prompt"] + "\n```python\n" + item["reference"] + "\n```")
    return texts


# Reasoning: BIG-Bench Hard, a few tasks with short exact answers.

BBH_TASKS = ["boolean_expressions", "date_understanding", "logical_deduction_three_objects",
             "tracking_shuffled_objects_three_objects", "web_of_lies", "navigate", "causal_judgement",
             "sports_understanding"]
BBH_INSTRUCTION = "Think step by step, then finish with 'So the answer is <answer>.'"


def bbh_items(count, skip=0):
    per_task = max(1, count // len(BBH_TASKS))
    items = []
    for task in BBH_TASKS:
        rows = load_rows("lukaemon/bbh", task, "test")
        order = shuffled(rows, 0)
        for index in order[skip:skip + per_task]:
            row = rows[index]
            items.append({"id": f"bbh/{task}/{index}", "subject": "bbh_" + task, "kind": "bbh",
                          "prompt": row["input"].strip() + "\n\n" + BBH_INSTRUCTION, "answer": row["target"]})
    return items


def reasoning_items(count, skip=0):
    return bbh_items(count, skip)


def reasoning_texts(count):
    texts = []
    # The full screen uses the first 25 rows of each task in the shuffled order.
    for item in bbh_items(count, skip=25):
        texts.append(item["prompt"] + "\nSo the answer is " + item["answer"] + ".")
    return texts


# Multilingual math: MGSM.

MGSM_LANGS = ["de", "es", "fr", "ja", "ru", "zh", "sw", "th", "bn", "te"]


def mgsm_items(count, skip=0):
    per_lang = max(1, count // len(MGSM_LANGS))
    items = []
    for lang in MGSM_LANGS:
        rows = load_rows("juletxara/mgsm", lang, "test")
        order = shuffled(rows, 0)
        for index in order[skip:skip + per_lang]:
            row = rows[index]
            items.append({"id": f"mgsm/{lang}/{index}", "subject": "mgsm_" + lang, "kind": "number",
                          "prompt": row["question"].strip() + "\n\n" + GSM8K_INSTRUCTION,
                          "answer": str(row["answer_number"])})
    return items


def mgsm_texts(count):
    texts = []
    # The full screen uses the first 25 rows of each language in the shuffled order.
    for item in mgsm_items(count, skip=25):
        texts.append(item["prompt"] + "\n#### " + item["answer"])
    return texts


def texts_of(items, to_text):
    texts = []
    for item in items:
        texts.append(to_text(item))
    return texts


# The registry. kind "mcq" scores by letter log-probability (wsp.mcq);
# kind "generate" scores greedy generations with the chat template.
SKILLS = {
    "knowledge": {"kind": "mcq", "count": 1000, "items": knowledge_items, "texts": knowledge_texts,
                  "about": "MMLU test, round 6 dev sample (seed 0, first 1000)"},
    "medicine": {"kind": "mcq", "count": 1000, "items": medicine_items, "texts": medicine_texts,
                 "about": "MedMCQA validation, round 10 order (seed 7, first 1000)"},
    "law": {"kind": "mcq", "count": 1000, "items": law_items, "texts": law_texts,
            "about": "MMLU professional_law, international_law, jurisprudence; LegalBench hearsay, "
                     "personal_jurisdiction, proa, ucc_v_common_law, abercrombie; CaseHOLD test"},
    "math": {"kind": "generate", "count": 500, "items": math_items, "texts": math_texts,
             "about": "GSM8K test 300 (#### answer) and MATH-500 200 (boxed answer), chat template, greedy"},
    "code": {"kind": "generate", "count": 200, "items": code_items, "texts": code_texts,
             "about": "MBPP sanitized test 200, code run against the asserts in a subprocess"},
    "science": {"kind": "mcq", "count": 500, "items": science_items, "texts": science_texts,
                "about": "ARC-Challenge test 500"},
    "reasoning": {"kind": "generate", "count": 200, "items": reasoning_items, "texts": reasoning_texts,
                  "about": "BIG-Bench Hard, 8 tasks with short answers, 25 each, chat template, greedy"},
    "multilingual": {"kind": "mcq", "count": 480, "items": multilingual_items, "texts": multilingual_texts,
                     "about": "Global-MMLU test in de, fr, es, zh, ja, hi, sw, ar (60 each)"},
    "multilingual_math": {"kind": "generate", "count": 250, "items": mgsm_items, "texts": mgsm_texts,
                          "about": "MGSM test in 10 languages (25 each), #### answer"},
    "decision": {"kind": "mcq", "count": 500, "items": decision_items, "texts": decision_texts,
                 "about": "fastino/fast-decisions single-label decisions as multiple choice"},
}

SKILL_ORDER = ["knowledge", "medicine", "law", "math", "code", "science", "reasoning", "multilingual",
               "multilingual_math", "decision"]

# ---------------------------------------------------------------------------
# Answer checks for generated text.


def strip_thinking(text):
    """The answer part of a response: the text after </think>.

    A response that opens <think> and never closes it ran out of tokens
    while thinking, so it has no answer and this returns "". Without this,
    the answer checks would read numbers from the unfinished thinking.
    """
    if "</think>" in text:
        return text.split("</think>")[-1]
    if "<think>" in text:
        return ""
    return text


def last_number(text):
    found = re.findall(r"-?\d[\d,]*(?:\.\d+)?", text)
    if len(found) == 0:
        return None
    return found[-1].replace(",", "")


def boxed(text):
    """The content of the last \\boxed{...} in text, with nested braces, or None."""
    start = text.rfind("\\boxed")
    if start < 0:
        return None
    brace = text.find("{", start)
    if brace < 0:
        return None
    depth = 0
    for position in range(brace, len(text)):
        if text[position] == "{":
            depth += 1
        elif text[position] == "}":
            depth -= 1
            if depth == 0:
                return text[brace + 1:position]
    return None


HASH_LINE = re.compile(r"####([^\n]*)")
HASH_NUMBER = re.compile(r"[\s*]*\$?\s*(-?\d[\d,]*(?:\.\d+)?)(.*)")


def hash_answer(text):
    """The number on the first "####" line that holds an answer, or None.

    "First" is fair to models without a chat template, which often go on
    to write a new problem and its answer. Markdown headers are skipped:
    "#### Step 1: ..." has no leading number, and "#### 1. Add ..." is a
    numbered header (a number, then ".", ")" or ":", then words).
    """
    for line in HASH_LINE.findall(text):
        found = HASH_NUMBER.match(line)
        if found is None:
            continue
        rest = found.group(2).strip()
        if len(rest) > 0 and rest[0] in ".):" and re.search(r"[A-Za-z]", rest):
            continue
        return found.group(1).replace(",", "")
    return None


def number_answer(text):
    """The answer of a #### style response, else a boxed answer, else the last number."""
    text = strip_thinking(text)
    value = hash_answer(text)
    if value is not None:
        return value
    inside = boxed(text)
    if inside is not None:
        value = last_number(inside)
        if value is not None:
            return value
    return last_number(text)


def same_number(a, b):
    if a is None or b is None:
        return False
    try:
        return abs(float(str(a).replace(",", "")) - float(str(b).replace(",", ""))) < 1e-4
    except ValueError:
        return False


def normalize_math(text):
    """A light normalizer for MATH answers: spacing, \\left/\\right, dfrac, units of degrees, dollars."""
    if text is None:
        return None
    out = text.strip()
    for old, new in [("\\left", ""), ("\\right", ""), ("\\!", ""), ("\\,", ""), ("\\;", ""), ("dfrac", "frac"),
                     ("tfrac", "frac"), ("^\\circ", ""), ("^{\\circ}", ""), ("\\%", ""), ("%", ""), ("\\$", ""),
                     ("$", ""), (" ", "")]:
        out = out.replace(old, new)
    out = re.sub(r"\\text\{(.*?)\}", r"\1", out)
    out = out.rstrip(".")
    return out


def math_correct(response, answer):
    got = boxed(strip_thinking(response))
    if got is None:
        return False
    if normalize_math(got) == normalize_math(answer):
        return True
    return same_number(normalize_math(got), normalize_math(answer))


def bbh_correct(response, target):
    text = strip_thinking(response)
    lowered = text.lower()
    marker = "the answer is"
    if marker in lowered:
        text = text[lowered.rfind(marker) + len(marker):]
    # Drop markdown and punctuation before the answer: "is: **Yes**." reads as "Yes".
    text = text.strip().lstrip(":*_`'\" \t\n").strip()
    target = target.strip()
    choice = re.fullmatch(r"\(([A-Z])\)", target)
    if choice is not None:
        found = re.search(r"\(([A-Z])\)", text)
        if found is not None:
            return found.group(1) == choice.group(1)
        found = re.match(r"([A-Z])\b", text)
        if found is not None:
            return found.group(1) == choice.group(1)
        return False
    # Whole-word match, so a target "No" does not match "Not".
    if not text.lower().startswith(target.lower()):
        return False
    after = text[len(target):len(target) + 1]
    return after == "" or not after.isalnum()


def extract_code(response):
    """The first ```python block of a response, else the first ``` block, else the whole text."""
    text = strip_thinking(response)
    found = re.search(r"```python\n(.*?)```", text, re.S)
    if found is None:
        found = re.search(r"```\w*\n(.*?)```", text, re.S)
    if found is None:
        return text
    return found.group(1)


# Runs before the model's code. It caps memory, file size and core dumps,
# and turns off the os and shutil calls that delete, move or kill things.
# It is a guard against accidents, not a security boundary: run the screen
# only in a throwaway container.
CODE_GUARD = """
import builtins as _builtins
import os as _os
import shutil as _shutil
try:
    import resource as _resource
    for _kind, _value in [("RLIMIT_AS", 4 << 30), ("RLIMIT_FSIZE", 16 << 20), ("RLIMIT_CORE", 0)]:
        try:
            _resource.setrlimit(getattr(_resource, _kind), (_value, _value))
        except (ValueError, OSError, AttributeError):
            pass
except ImportError:
    pass


def _blocked(*args, **kwargs):
    raise PermissionError("blocked in the code screen")


for _name in ["system", "kill", "killpg", "remove", "unlink", "rmdir", "removedirs", "rename", "renames",
              "replace", "truncate", "chmod", "chown", "fork", "forkpty", "putenv", "unsetenv"]:
    if hasattr(_os, _name):
        setattr(_os, _name, _blocked)
for _name in ["rmtree", "move", "chown"]:
    setattr(_shutil, _name, _blocked)
"""


def run_python(program, timeout=10, guard=True):
    """True when program runs to its end and exits with code 0 within timeout seconds.

    The program runs isolated (python -I) in a temp dir, with an empty
    stdin, no captured output, and a small environment (no tokens). It
    runs in its own process group, and the whole group is killed when the
    run ends, so a child process cannot outlive the timeout. A marker file
    written after the last line proves the program ran to its end, so an
    early exit(0) does not count as a pass.
    """
    token = os.urandom(8).hex()
    with tempfile.TemporaryDirectory() as folder:
        path = os.path.join(folder, "program.py")
        marker = os.path.join(folder, "done.txt")
        text = program
        if guard:
            text = CODE_GUARD + "\n" + program
        text = text + "\n\nwith open(" + repr(marker) + ", 'w') as _done:\n    _done.write(" + repr(token) + ")\n"
        with open(path, "w") as f:
            f.write(text)
        env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "HOME": folder, "TMPDIR": folder,
               "PYTHONHASHSEED": "0", "PYTHONDONTWRITEBYTECODE": "1"}
        process = subprocess.Popen([sys.executable, "-I", path], cwd=folder, env=env, stdin=subprocess.DEVNULL,
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
        timed_out = False
        try:
            process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
        # The group id is the child's pid (start_new_session). Kill the group
        # even after a normal exit, to stop any children it left behind.
        try:
            os.killpg(process.pid, 9)
        except (ProcessLookupError, PermissionError):
            pass
        process.wait()
        if timed_out or process.returncode != 0:
            return False
        if not os.path.exists(marker):
            return False
        with open(marker) as f:
            return f.read() == token


def code_program(item, code):
    lines = []
    for line in item.get("imports", []):
        lines.append(line)
    lines.append(code)
    for test in item["tests"]:
        lines.append(test)
    return "\n".join(lines) + "\n"


def check_code(items, responses, timeout=10, workers=8):
    programs = []
    for item, response in zip(items, responses):
        programs.append(code_program(item, extract_code(response)))
    results = []
    with concurrent.futures.ThreadPoolExecutor(workers) as pool:
        for passed in pool.map(lambda program: run_python(program, timeout), programs):
            results.append(passed)
    return results


def check_generated(items, responses):
    """Per-item True/False for generation items (number, boxed, bbh, code)."""
    code_items_list = []
    code_responses = []
    results = []
    for item, response in zip(items, responses):
        if item["kind"] == "code":
            code_items_list.append(item)
            code_responses.append(response)
            results.append(None)
        elif item["kind"] == "number":
            results.append(same_number(number_answer(response), item["answer"]))
        elif item["kind"] == "boxed":
            results.append(math_correct(response, item["answer"]))
        elif item["kind"] == "bbh":
            results.append(bbh_correct(response, item["answer"]))
        else:
            raise ValueError(f"unknown item kind {item['kind']!r}")
    if len(code_items_list) > 0:
        passed = check_code(code_items_list, code_responses)
        position = 0
        for index in range(len(results)):
            if results[index] is None:
                results[index] = passed[position]
                position += 1
    return results

# ---------------------------------------------------------------------------
# Model helpers.


def model_device(model):
    return next(model.parameters()).device


# Chat formats for chat models whose tokenizer has no chat template. The zoo
# names one in "chat_format", and ensure_chat_template installs it.
CHAT_FORMATS = {
    # Mistral and Mixtral instruct, as Equall/Saul-7B-Instruct-v1 writes it.
    "mistral_inst": "{{ bos_token }}{% for message in messages %}{% if message['role'] == 'user' %}"
                    "{{ '[INST] ' + message['content'] + ' [/INST]' }}{% else %}{{ message['content'] + eos_token }}"
                    "{% endif %}{% endfor %}",
    # Llama-2-chat, as the AdaptLLM/law-chat card writes it (without the long system prompt).
    "llama2_chat": "{{ bos_token }}{% for message in messages %}{% if message['role'] == 'user' %}"
                   "{{ '[INST] ' + message['content'] + ' [/INST]' }}{% else %}{{ ' ' + message['content'] + ' ' + eos_token }}"
                   "{% endif %}{% endfor %}",
    # Llama 3 instruct.
    "llama3": "{{ bos_token }}{% for message in messages %}"
              "{{ '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n' + message['content'] + '<|eot_id|>' }}"
              "{% endfor %}{% if add_generation_prompt %}{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}{% endif %}",
}

# End-of-turn tokens. generate stops on any of them that the vocabulary has,
# besides the eos ids, because some chat models list only <|end_of_text|>
# as eos and would write past the end of their turn.
END_OF_TURN = ["<|eot_id|>", "<|im_end|>", "<|end|>", "<end_of_turn>", "<|EOT|>", "<|endoftext|>",
               "<|end_of_text|>", "</s>", "<|eom_id|>"]

# Lines that start a new problem or turn in a raw (no template) continuation.
CONTINUATION_MARKERS = ["\nQuestion:", "\nProblem:", "\nQ:", "\nUser:", "\nHuman:", "\n### Instruction",
                        "\n### Question", "[INST]", "<|im_start|>", "<|user|>"]


def ensure_chat_template(tokenizer, chat_format=None):
    """Install CHAT_FORMATS[chat_format] when the tokenizer has no chat template. Returns True when it has one."""
    if getattr(tokenizer, "chat_template", None) is not None:
        return True
    if chat_format is None:
        return False
    tokenizer.chat_template = CHAT_FORMATS[chat_format]
    return True


def stop_token_ids(model, tokenizer):
    """Sorted ids that end a generated turn: the eos ids plus any END_OF_TURN token in the vocabulary."""
    ids = set()
    config = getattr(model, "generation_config", None)
    eos = None
    if config is not None:
        eos = config.eos_token_id
    if isinstance(eos, int):
        ids.add(eos)
    elif eos is not None:
        for value in eos:
            ids.add(int(value))
    if tokenizer.eos_token_id is not None:
        ids.add(tokenizer.eos_token_id)
    vocab = tokenizer.get_vocab()
    for token in END_OF_TURN:
        if token in vocab:
            ids.add(vocab[token])
    return sorted(ids)


def cut_continuation(text):
    """Cut a raw continuation where it starts a new problem or turn."""
    end = len(text)
    for marker in CONTINUATION_MARKERS:
        position = text.find(marker)
        if position >= 0 and position < end:
            end = position
    return text[:end]


def clean_decoded(text, tokenizer):
    """Remove special tokens from decoded text, but keep <think> and </think>."""
    for token in tokenizer.all_special_tokens:
        if "think" in token:
            continue
        text = text.replace(token, "")
    return text


def chat_text(tokenizer, user_text, thinking=None):
    """(text, used_template): the user turn in the tokenizer's chat template when it has one."""
    if getattr(tokenizer, "chat_template", None) is None:
        return user_text, False
    messages = [{"role": "user", "content": user_text}]
    if thinking is None:
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                             enable_thinking=thinking)
    return text, True


def generate(model, tokenizer, prompts, max_new_tokens=512, batch_size=16, thinking=None):
    """Greedy completions for prompts, in prompt order. Uses the chat template when there is one.

    Prompts are sorted by length so each left-padded batch wastes little.
    Generation stops at the eos ids and at end-of-turn tokens
    (stop_token_ids). When the template opens a <think> section in the
    prompt, the output gets that <think> back, so strip_thinking can tell
    an unfinished thought from an answer. Without a template, the output is
    cut where the model starts a new problem (cut_continuation).
    """
    texts = []
    used_template = False
    for prompt in prompts:
        text, used_template = chat_text(tokenizer, prompt, thinking)
        texts.append(text)
    order = list(range(len(texts)))
    order.sort(key=lambda index: -len(texts[index]))
    stops = stop_token_ids(model, tokenizer)

    old_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    device = model_device(model)
    outputs = [None] * len(texts)
    try:
        for start in range(0, len(order), batch_size):
            chunk = order[start:start + batch_size]
            batch = []
            for index in chunk:
                batch.append(texts[index])
            encoded = tokenizer(batch, return_tensors="pt", padding=True,
                                add_special_tokens=not used_template).to(device)
            with torch.no_grad():
                out = model.generate(input_ids=encoded["input_ids"], attention_mask=encoded["attention_mask"],
                                     max_new_tokens=max_new_tokens, do_sample=False, temperature=None,
                                     top_p=None, top_k=None, eos_token_id=stops,
                                     pad_token_id=tokenizer.pad_token_id)
            new_tokens = out[:, encoded["input_ids"].shape[1]:]
            decoded = tokenizer.batch_decode(new_tokens, skip_special_tokens=False)
            for index, text in zip(chunk, decoded):
                text = clean_decoded(text, tokenizer)
                if not used_template:
                    text = cut_continuation(text)
                elif texts[index].rstrip().endswith("<think>") and "<think>" not in text:
                    text = "<think>\n" + text
                outputs[index] = text
    finally:
        tokenizer.padding_side = old_side
    return outputs


def summarize(items, correct):
    """{"score", "count", "by_subject": {subject: accuracy}} from per-item results."""
    right = {}
    total = {}
    all_right = 0
    for item, ok in zip(items, correct):
        subject = item.get("subject", "")
        if subject not in total:
            total[subject] = 0
            right[subject] = 0
        total[subject] += 1
        if ok:
            right[subject] += 1
            all_right += 1
    by_subject = {}
    for subject in sorted(total):
        by_subject[subject] = round(right[subject] / total[subject], 4)
    score = 0.0
    if len(items) > 0:
        score = all_right / len(items)
    return {"score": round(score, 4), "count": len(items), "by_subject": by_subject}


def screen_skill(model, tokenizer, name, scale=1.0, max_new_tokens=512, batch_size=16, thinking=None, items=None):
    """Score one skill. Returns {"score", "count", "by_subject"}.

    scale shrinks the item count (0.1 is a tenth of the full screen). items
    overrides the loader, for tests.
    """
    skill = SKILLS[name]
    if items is None:
        count = max(1, int(round(skill["count"] * scale)))
        items = skill["items"](count)
    if skill["kind"] == "mcq":
        records = mcq.score_items(model, tokenizer, items, batch_size=batch_size)
        correct = []
        for record in records:
            correct.append(record["correct"])
        return summarize(items, correct)
    prompts = []
    for item in items:
        prompts.append(item["prompt"])
    responses = generate(model, tokenizer, prompts, max_new_tokens=max_new_tokens, batch_size=batch_size,
                         thinking=thinking)
    return summarize(items, check_generated(items, responses))

# ---------------------------------------------------------------------------
# Layer maps.


def encode_texts(tokenizer, texts, max_length=256):
    """Token id lists, one per non-empty text, cut to max_length tokens."""
    encoded = []
    for text in texts:
        ids = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)["input_ids"][0]
        ids = ids[:max_length]
        if len(ids) >= 2:
            encoded.append(ids)
    return encoded


def pad_batch(id_lists, pad_id=0):
    """(input_ids, attention_mask), right padded."""
    longest = 0
    for ids in id_lists:
        longest = max(longest, len(ids))
    input_ids = torch.full((len(id_lists), longest), pad_id, dtype=torch.long)
    mask = torch.zeros((len(id_lists), longest), dtype=torch.long)
    for row, ids in enumerate(id_lists):
        input_ids[row, :len(ids)] = ids
        mask[row, :len(ids)] = 1
    return input_ids, mask


def layer_input(args, kwargs):
    if len(args) > 0:
        return args[0]
    return kwargs["hidden_states"]


def skip_hook(module, args, kwargs, output):
    """Forward hook that makes a decoder layer a residual passthrough: its output is its input."""
    hidden = layer_input(args, kwargs)
    if isinstance(output, tuple):
        return (hidden,) + tuple(output[1:])
    return hidden


def pass_stats(model, encoded, batch_size=8, skip_layer=None, influence=False):
    """One pass over encoded texts. Returns {"loss", "tokens", "influence"}.

    loss is the mean next-token loss over all predicted tokens. With
    skip_layer, that decoder layer is a passthrough. With influence, also
    returns each layer's mean (1 - cosine(input, output)) over real tokens
    (ShortGPT block influence).
    """
    _, layers = find_decoder_layers(model, 0)
    device = model_device(model)
    handles = []
    state = {"mask": None}
    sums = []
    counts = []
    for index in range(len(layers)):
        sums.append(0.0)
        counts.append(0)
    store = {}

    if skip_layer is not None:
        handles.append(layers[skip_layer].register_forward_hook(skip_hook, with_kwargs=True))
    if influence:
        for index in range(len(layers)):
            def pre(module, args, kwargs, index=index):
                store[index] = layer_input(args, kwargs).detach()

            def post(module, args, kwargs, output, index=index):
                out = output
                if isinstance(output, tuple):
                    out = output[0]
                cos = torch.nn.functional.cosine_similarity(store[index].float(), out.detach().float(), dim=-1)
                real = state["mask"].to(cos.device).bool()
                sums[index] += float((1 - cos)[real].sum())
                counts[index] += int(real.sum())

            handles.append(layers[index].register_forward_pre_hook(pre, with_kwargs=True))
            handles.append(layers[index].register_forward_hook(post, with_kwargs=True))

    was_training = model.training
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    try:
        for start in range(0, len(encoded), batch_size):
            input_ids, mask = pad_batch(encoded[start:start + batch_size])
            input_ids = input_ids.to(device)
            mask = mask.to(device)
            state["mask"] = mask
            with torch.no_grad():
                logits = model(input_ids=input_ids, attention_mask=mask, use_cache=False).logits.float()
            targets = input_ids[:, 1:]
            keep = mask[:, 1:].bool()
            losses = torch.nn.functional.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]),
                                                       targets.reshape(-1), reduction="none")
            losses = losses.reshape(targets.shape)
            total_loss += float(losses[keep].sum())
            total_tokens += int(keep.sum())
    finally:
        for handle in handles:
            handle.remove()
        if was_training:
            model.train()

    result = {"loss": total_loss / max(1, total_tokens), "tokens": total_tokens, "influence": None, "positions": 0}
    if influence:
        values = []
        for index in range(len(layers)):
            values.append(sums[index] / max(1, counts[index]))
        result["influence"] = values
        result["positions"] = counts[0]
    return result


def merge_stats(first, second):
    """pass_stats over two sets of texts, combined as one pass over both would give it."""
    tokens = first["tokens"] + second["tokens"]
    loss = (first["loss"] * first["tokens"] + second["loss"] * second["tokens"]) / max(1, tokens)
    merged = {"loss": loss, "tokens": tokens, "influence": None,
              "positions": first["positions"] + second["positions"]}
    if first["influence"] is not None and second["influence"] is not None:
        values = []
        for a_value, b_value in zip(first["influence"], second["influence"]):
            total = a_value * first["positions"] + b_value * second["positions"]
            values.append(total / max(1, merged["positions"]))
        merged["influence"] = values
    return merged


def split_halves(encoded):
    """(even positions, odd positions) of a list."""
    first = []
    second = []
    for index in range(len(encoded)):
        if index % 2 == 0:
            first.append(encoded[index])
        else:
            second.append(encoded[index])
    return first, second


def positive_share(scores, blocks):
    """Each block's share of the total positive score mass in scores."""
    total = 0.0
    for value in scores:
        total += max(0.0, value)
    shares = []
    for first, last in blocks:
        if total <= 0:
            shares.append(0.0)
        else:
            shares.append(round(block_mass(scores, first, last) / total, 4))
    return shares


def pearson(a, b):
    """Pearson correlation of two equal-length lists, or None when one is constant."""
    n = len(a)
    if n < 2:
        return None
    mean_a = sum(a) / n
    mean_b = sum(b) / n
    cov = 0.0
    var_a = 0.0
    var_b = 0.0
    for x, y in zip(a, b):
        cov += (x - mean_a) * (y - mean_b)
        var_a += (x - mean_a) ** 2
        var_b += (y - mean_b) ** 2
    if var_a <= 0 or var_b <= 0:
        return None
    return cov / math.sqrt(var_a * var_b)


def text_loss(model, tokenizer, texts, skip_layer=None, max_length=256, batch_size=8):
    """Mean next-token loss on texts; with skip_layer, that layer is a residual passthrough."""
    encoded = encode_texts(tokenizer, texts, max_length)
    return pass_stats(model, encoded, batch_size, skip_layer)["loss"]


def best_window(scores, length):
    """(start, mass) of the window of `length` layers with the largest sum of positive scores."""
    best_start = 0
    best_mass = None
    for start in range(0, len(scores) - length + 1):
        mass = 0.0
        for index in range(start, start + length):
            mass += max(0.0, scores[index])
        if best_mass is None or mass > best_mass:
            best_start = start
            best_mass = mass
    return best_start, best_mass


def suggest_blocks(scores, target_length=None, mass=0.6, count=3):
    """Contiguous layer blocks that hold the skill-specific effect.

    Only positive scores count as mass. Without target_length, the block
    length is the shortest span whose positive mass reaches `mass` of the
    total. The first block is the best window of that length; up to count-1
    more blocks of the same length follow, best first, without overlap.
    Returns (blocks, fractions): blocks are [first, last] with both ends
    included, fractions are each block's share of the total positive mass.
    """
    n = len(scores)
    total = 0.0
    for value in scores:
        total += max(0.0, value)
    if n == 0 or total <= 0:
        return [], []

    if target_length is None:
        length = n
        for size in range(1, n + 1):
            start, window_mass = best_window(scores, size)
            if window_mass >= mass * total - 1e-12:
                length = size
                break
    else:
        length = max(1, min(int(target_length), n))

    blocks = []
    fractions = []
    taken = [False] * n
    while len(blocks) < count:
        best_start = None
        best_mass = 0.0
        for start in range(0, n - length + 1):
            free = True
            window_mass = 0.0
            for index in range(start, start + length):
                if taken[index]:
                    free = False
                    break
                window_mass += max(0.0, scores[index])
            if free and window_mass > best_mass:
                best_start = start
                best_mass = window_mass
        if best_start is None:
            break
        for index in range(best_start, best_start + length):
            taken[index] = True
        blocks.append([best_start, best_start + length - 1])
        fractions.append(round(best_mass / total, 4))
    return blocks, fractions


def layer_skill_map(model, tokenizer, skill_texts, general_texts, max_length=256, batch_size=8,
                    target_length=None, mass=0.6, count=3):
    """Per decoder layer, how much more the layer matters for skill text than for general text.

    Two measures:
    - influence: block influence (1 - cosine of layer input and output) on
      skill text minus on general text.
    - ablation: the loss increase on skill text when that one layer is a
      residual passthrough, minus the same increase on general text.

    scores is the ablation measure; blocks are suggested from it (see
    suggest_blocks). Returns a dict with layers, scores, influence, blocks,
    block_mass, and the raw parts (skill and general base losses, per-layer
    increases, per-layer influence on each set).

    block_mass is measured on the same texts that chose the blocks, so it
    is optimistic. "heldout" checks it: the texts are split into two
    halves (even and odd positions, each side), blocks are chosen on the
    first half only, and their mass is measured on the second half, which
    took no part in the choice. heldout also has the correlation of the two
    halves' scores. It is None with fewer than 2 texts on a side. The split
    costs no extra passes.
    """
    skill = encode_texts(tokenizer, skill_texts, max_length)
    general = encode_texts(tokenizer, general_texts, max_length)
    if len(skill) == 0 or len(general) == 0:
        raise ValueError("layer_skill_map needs skill texts and general texts")
    _, layers = find_decoder_layers(model, 0)

    split = len(skill) >= 2 and len(general) >= 2
    if split:
        skill_sets = split_halves(skill)
        general_sets = split_halves(general)
    else:
        skill_sets = (skill,)
        general_sets = (general,)

    def stats(sets, skip_layer=None, influence=False):
        # One pass per half; the merge equals one pass over all texts.
        halves = []
        for part in sets:
            halves.append(pass_stats(model, part, batch_size, skip_layer=skip_layer, influence=influence))
        merged = halves[0]
        if len(halves) == 2:
            merged = merge_stats(halves[0], halves[1])
        return merged, halves

    skill_base, skill_base_halves = stats(skill_sets, influence=True)
    general_base, general_base_halves = stats(general_sets, influence=True)

    skill_increase = []
    general_increase = []
    scores = []
    influence = []
    half_scores = [[], []]
    for index in range(len(layers)):
        skill_drop, skill_halves = stats(skill_sets, skip_layer=index)
        general_drop, general_halves = stats(general_sets, skip_layer=index)
        skill_increase.append(skill_drop["loss"] - skill_base["loss"])
        general_increase.append(general_drop["loss"] - general_base["loss"])
        scores.append(skill_increase[index] - general_increase[index])
        influence.append(skill_base["influence"][index] - general_base["influence"][index])
        if split:
            for half in range(2):
                skill_up = skill_halves[half]["loss"] - skill_base_halves[half]["loss"]
                general_up = general_halves[half]["loss"] - general_base_halves[half]["loss"]
                half_scores[half].append(skill_up - general_up)

    blocks, fractions = suggest_blocks(scores, target_length, mass, count)
    heldout = None
    if split:
        select_blocks, select_mass = suggest_blocks(half_scores[0], target_length, mass, count)
        heldout = {
            "blocks": select_blocks,
            "select_mass": select_mass,
            "check_mass": positive_share(half_scores[1], select_blocks),
            "split_half_r": pearson(half_scores[0], half_scores[1]),
            "select_scores": half_scores[0],
            "check_scores": half_scores[1],
        }
    return {
        "layers": len(layers),
        "scores": scores,
        "influence": influence,
        "blocks": blocks,
        "block_mass": fractions,
        "heldout": heldout,
        "skill_loss": skill_base["loss"],
        "general_loss": general_base["loss"],
        "skill_increase": skill_increase,
        "general_increase": general_increase,
        "skill_influence": skill_base["influence"],
        "general_influence": general_base["influence"],
    }

# ---------------------------------------------------------------------------
# Expert maps.


def first_attr(names, *objects):
    """First non-None attribute among names, looking at each object in order."""
    for obj in objects:
        if obj is None:
            continue
        for name in names:
            value = getattr(obj, name, None)
            if value is not None:
                return value
    return None


def find_routers(model):
    """[(layer index, moe module, router module, top_k, num_experts)] for every MoE layer.

    An MoE module has an `experts` attribute and a `gate` (or `router`)
    router. It is a child of the decoder layer in Qwen3-MoE, Mixtral, OLMoE,
    LFM2-MoE and Qwen3.5-MoE. In Gemma 4 the decoder layer itself holds
    `experts` and `router`, so the layer counts as the MoE module there.

    Transformers 4.x lists the experts as a ModuleList, so len() gives their
    count. Transformers 5.x fuses them into one module, and top_k and
    num_experts sit on the router, on the experts module or in the config.
    """
    _, layers = find_decoder_layers(model, 0)
    config = model.config
    if hasattr(config, "get_text_config"):
        config = config.get_text_config()
    found = []
    for index in range(len(layers)):
        layer = layers[index]
        holders = []
        for name, child in layer.named_children():
            holders.append(child)
        holders.append(layer)
        for child in holders:
            if child is layer and "experts" not in layer._modules:
                continue
            if not hasattr(child, "experts"):
                continue
            router = getattr(child, "gate", None)
            if router is None:
                router = getattr(child, "router", None)
            if router is None:
                continue
            top_k = first_attr(["top_k"], child, router, child.experts)
            if top_k is None:
                top_k = first_attr(["num_experts_per_tok", "top_k_experts", "moe_top_k"], config)
            num_experts = first_attr(["num_experts"], child, router, child.experts)
            if num_experts is None and hasattr(child.experts, "__len__"):
                num_experts = len(child.experts)
            if num_experts is None:
                num_experts = first_attr(["num_experts", "num_local_experts", "n_routed_experts"], config)
            if top_k is None or num_experts is None:
                # Unknown layout; skip it rather than fail the whole measurement.
                continue
            found.append((index, child, router, int(top_k), int(num_experts)))
            break
    return found


def selected_experts(output, top_k, num_experts):
    """LongTensor (tokens, top_k) of the experts a router output picks.

    A router returns one of: logits (tokens, num_experts), as in
    transformers 4.x Qwen3-MoE; or a tuple. A tuple may hold an integer
    index tensor (tokens, top_k), as in 5.x routers that return
    (logits, scores, indices). When it has one, that tensor is used, since
    it applies any expert bias the router adds before picking. Otherwise
    the first float tensor with num_experts columns is the logits.
    """
    if isinstance(output, (tuple, list)):
        for part in output:
            if torch.is_tensor(part) and not torch.is_floating_point(part) and part.shape[-1] == top_k:
                return part.reshape(-1, top_k)
        for part in output:
            if torch.is_tensor(part) and torch.is_floating_point(part) and part.shape[-1] == num_experts:
                output = part
                break
        else:
            raise NotImplementedError(f"router output has no index tensor or logits of width {num_experts}")
    logits = output.detach().float().reshape(-1, output.shape[-1])
    if logits.shape[-1] != num_experts:
        raise NotImplementedError(f"router output has width {logits.shape[-1]}, expected {num_experts} experts")
    return torch.topk(logits, top_k, dim=-1).indices


def routing_counts(model, tokenizer, texts, max_length=256):
    """{layer: (counts per expert, tokens)}: how often each expert is picked, texts run one at a time."""
    routers = find_routers(model)
    if len(routers) == 0:
        raise ValueError("model has no MoE layers with a gate router")
    device = model_device(model)
    counts = {}
    tokens = {}
    handles = []
    for layer, moe, router, top_k, num_experts in routers:
        counts[layer] = torch.zeros(num_experts, dtype=torch.float64)
        tokens[layer] = 0

        def hook(module, args, output, layer=layer, top_k=top_k, num_experts=num_experts):
            picks = selected_experts(output, top_k, num_experts).reshape(-1).cpu()
            counts[layer] += torch.bincount(picks, minlength=num_experts).double()
            tokens[layer] += picks.numel() // top_k

        handles.append(router.register_forward_hook(hook))
    was_training = model.training
    model.eval()
    try:
        for ids in encode_texts(tokenizer, texts, max_length):
            with torch.no_grad():
                model(input_ids=ids.unsqueeze(0).to(device), use_cache=False)
    finally:
        for handle in handles:
            handle.remove()
        if was_training:
            model.train()
    out = {}
    for layer in counts:
        out[layer] = (counts[layer], tokens[layer])
    return out


def expert_skill_map(model, tokenizer, skill_texts, general_texts, top=8, max_length=256, eps=1e-4, min_freq=0.0):
    """Per MoE layer, which experts the router picks more on skill text than on general text.

    freq is the share of tokens that pick an expert (it sums to top_k over
    experts). log_ratio = log((skill freq + eps) / (general freq + eps)).
    top lists up to `top` experts per layer with a positive log ratio and a
    skill freq of at least min_freq, highest ratio first.

    Returns {"layers": {layer: {"skill_freq", "general_freq", "log_ratio",
    "top"}}, "top_k", "num_experts"}.
    """
    skill = routing_counts(model, tokenizer, skill_texts, max_length)
    general = routing_counts(model, tokenizer, general_texts, max_length)
    routers = find_routers(model)
    result = {"layers": {}, "top_k": routers[0][3], "num_experts": routers[0][4]}
    for layer in skill:
        skill_counts, skill_tokens = skill[layer]
        general_counts, general_tokens = general[layer]
        skill_freq = (skill_counts / max(1, skill_tokens)).tolist()
        general_freq = (general_counts / max(1, general_tokens)).tolist()
        ratios = []
        for s_value, g_value in zip(skill_freq, general_freq):
            ratios.append(math.log((s_value + eps) / (g_value + eps)))
        order = list(range(len(ratios)))
        order.sort(key=lambda expert: -ratios[expert])
        chosen = []
        for expert in order:
            if len(chosen) >= top:
                break
            if ratios[expert] <= 0 or skill_freq[expert] < min_freq:
                continue
            chosen.append(expert)
        result["layers"][layer] = {
            "skill_freq": rounded(skill_freq),
            "general_freq": rounded(general_freq),
            "log_ratio": rounded(ratios),
            "top": chosen,
        }
    return result


def rounded(values, digits=5):
    out = []
    for value in values:
        out.append(round(float(value), digits))
    return out


def top_experts(expert_map):
    """The compact entry form of an expert_skill_map result: {str(layer): [expert ids]}."""
    out = {}
    for layer in expert_map["layers"]:
        out[str(layer)] = list(expert_map["layers"][layer]["top"])
    return out

# ---------------------------------------------------------------------------
# Entries.


def now_utc():
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def make_entry(model, skills, layer_maps=None, expert_maps=None, params_b=None, layers=None, hidden=None,
               family=None, skill_details=None, measured_at=None):
    """An atlas entry. layer_maps values may be full layer_skill_map results; only the stored fields are kept."""
    compact = {}
    if layer_maps is not None:
        for skill in layer_maps:
            full = layer_maps[skill]
            compact[skill] = {"scores": rounded(full["scores"]), "blocks": full["blocks"]}
            if "influence" in full:
                compact[skill]["influence"] = rounded(full["influence"])
            if "block_mass" in full:
                compact[skill]["block_mass"] = full["block_mass"]
            heldout = full.get("heldout")
            if heldout is not None:
                r_value = heldout.get("split_half_r")
                if r_value is not None:
                    r_value = round(float(r_value), 4)
                compact[skill]["heldout"] = {"blocks": heldout["blocks"], "select_mass": heldout["select_mass"],
                                             "check_mass": heldout["check_mass"], "split_half_r": r_value}
    experts = None
    if expert_maps is not None:
        experts = {}
        for skill in expert_maps:
            value = expert_maps[skill]
            if isinstance(value, dict) and "layers" in value:
                value = top_experts(value)
            experts[skill] = value
    if measured_at is None:
        measured_at = now_utc()
    return {
        "model": model,
        "skills": dict(skills),
        "skill_details": skill_details or {},
        "layer_maps": compact,
        "expert_maps": experts,
        "params_b": params_b,
        "layers": layers,
        "hidden": hidden,
        "family": family,
        "measured_at": measured_at,
    }


def entry_name(model):
    return model.replace("/", "__") + ".json"


def write_entry(entry, folder):
    """Write entry to folder/<org>__<name>.json and return the path."""
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, entry_name(entry["model"]))
    with open(path, "w") as f:
        json.dump(entry, f, indent=1)
    return path


def read_entry(path):
    with open(path) as f:
        return json.load(f)


def load_entries(folder):
    """All entries in a folder, sorted by model id."""
    entries = []
    if not os.path.isdir(folder):
        return entries
    for name in sorted(os.listdir(folder)):
        if name.endswith(".json"):
            entries.append(read_entry(os.path.join(folder, name)))
    return entries


def as_entries(entries):
    if isinstance(entries, str):
        return load_entries(entries)
    return list(entries)


def best_models_for(skill, entries, max_params_b=None):
    """[{"model", "score", "params_b"}] for entries that have the skill, best first.

    Ties go to the smaller model. entries is a list or an entries folder.
    """
    rows = []
    for entry in as_entries(entries):
        if skill not in entry.get("skills", {}):
            continue
        params = entry.get("params_b")
        if max_params_b is not None and params is not None and params > max_params_b:
            continue
        rows.append({"model": entry["model"], "score": entry["skills"][skill], "params_b": params})
    rows.sort(key=lambda row: (-row["score"], row["params_b"] if row["params_b"] is not None else 1e9))
    return rows


def as_list(value):
    if value is None:
        return None
    if isinstance(value, (list, tuple, set)):
        return list(value)
    return [value]


def block_mass(scores, first, last):
    mass = 0.0
    for index in range(first, last + 1):
        mass += max(0.0, scores[index])
    return mass


def best_block_for(skill, entries, base_constraints=None):
    """The best donor block for a skill under constraints, or None.

    base_constraints keys (all optional):
        hidden          int or list: donor hidden size must match (no stitch)
        family          str or list: donor model type
        models          list of allowed model ids
        exclude         list of model ids to skip
        max_params_b    largest donor size
        max_length      longest block; a longer block is cut to its best
                        sub-window of this length
        min_length      shortest block
    Candidates rank by the donor's skill score, then by block mass (the
    sum of positive layer-drop scores in the block). Returns {"model",
    "block": [first, last], "length", "skill_score", "mass", "hidden",
    "family"}.
    """
    constraints = base_constraints or {}
    hidden = as_list(constraints.get("hidden"))
    family = as_list(constraints.get("family"))
    allowed = as_list(constraints.get("models"))
    exclude = as_list(constraints.get("exclude")) or []
    max_params = constraints.get("max_params_b")
    max_length = constraints.get("max_length")
    min_length = constraints.get("min_length", 1)

    best = None
    for entry in as_entries(entries):
        layer_map = entry.get("layer_maps", {}).get(skill)
        if layer_map is None:
            continue
        if allowed is not None and entry["model"] not in allowed:
            continue
        if entry["model"] in exclude:
            continue
        if hidden is not None and entry.get("hidden") not in hidden:
            continue
        if family is not None and entry.get("family") not in family:
            continue
        if max_params is not None and entry.get("params_b") is not None and entry["params_b"] > max_params:
            continue
        scores = layer_map["scores"]
        skill_score = entry.get("skills", {}).get(skill, 0.0)
        for first, last in layer_map.get("blocks", []):
            length = last - first + 1
            if max_length is not None and length > max_length:
                start, _ = best_window(scores[first:last + 1], int(max_length))
                first = first + start
                last = first + int(max_length) - 1
                length = int(max_length)
            if length < min_length:
                continue
            candidate = {"model": entry["model"], "block": [first, last], "length": length,
                         "skill_score": skill_score, "mass": round(block_mass(scores, first, last), 6),
                         "hidden": entry.get("hidden"), "family": entry.get("family")}
            if best is None or (candidate["skill_score"], candidate["mass"]) > (best["skill_score"], best["mass"]):
                best = candidate
    return best


def short_name(model):
    return model.split("/")[-1].lower().replace(".", "_")


def catalog_rows(entry, skill, units_dir=None):
    """Catalog rows (Person A's format) for an entry's blocks and, with units_dir, its experts.

    Block rows need no units file: {"id", "type": "block", "model",
    "layers": [first, last], ...}. Expert rows point at a units file
    {kind, model, layer, indices} written under units_dir, as
    person_a/parts/<model>/*.pt.
    """
    rows = []
    layer_map = entry.get("layer_maps", {}).get(skill)
    name = short_name(entry["model"])
    if layer_map is not None:
        masses = layer_map.get("block_mass", [])
        for number, (first, last) in enumerate(layer_map.get("blocks", [])):
            share = None
            if number < len(masses):
                share = masses[number]
            rows.append({
                "id": f"atlas_block_{skill}_{name}_L{first}-{last}",
                "type": "block",
                "label": f"{skill} block",
                "tier": "atlas",
                "trained_on": "atlas layer-drop map (skill minus general loss increase)",
                "model": entry["model"],
                "layers": [first, last],
                "d_model": entry.get("hidden"),
                "signature": {"effect": {skill + "|block_mass": share}},
            })
    experts = entry.get("expert_maps") or {}
    if units_dir is not None and skill in experts:
        os.makedirs(units_dir, exist_ok=True)
        for layer in sorted(experts[skill], key=int):
            ids = experts[skill][layer]
            if len(ids) == 0:
                continue
            part_id = f"atlas_expert_{skill}_{name}_L{layer}"
            path = os.path.join(units_dir, part_id + ".pt")
            torch.save({"kind": "expert", "model": entry["model"], "layer": int(layer), "indices": list(ids)}, path)
            rows.append({
                "id": part_id,
                "type": "expert",
                "label": f"{skill} experts",
                "tier": "atlas",
                "trained_on": "atlas routing map (skill vs general routing frequency)",
                "model": entry["model"],
                "layer": int(layer),
                "units": path,
                "num_units": len(ids),
                "d_model": entry.get("hidden"),
            })
    return rows

# ---------------------------------------------------------------------------
# Whole-model measurement, used by modal_atlas.py.


def describe_model(model):
    config = model.config
    if hasattr(config, "get_text_config"):
        config = config.get_text_config()
    params = 0
    for param in model.parameters():
        params += param.numel()
    _, layers = find_decoder_layers(model, 0)
    return {"params_b": round(params / 1e9, 3), "layers": len(layers), "hidden": getattr(config, "hidden_size", None),
            "family": getattr(config, "model_type", None)}


def measure_model(model, tokenizer, model_id, skills=None, scale=1.0, map_texts=32, max_new_tokens=512,
                  batch_size=16, screens=True, layer_maps=True, expert_maps=None, thinking=None, log=print,
                  on_progress=None):
    """Screen skills and build maps for one loaded model. Returns an atlas entry.

    expert_maps None means "when the model has MoE routers". A skill that
    fails to load or score is logged and left out, so one broken dataset
    does not lose the rest of the entry. on_progress, when given, gets the
    entry so far after each skill, so a run that times out keeps the skills
    it finished.
    """
    if skills is None:
        skills = list(SKILL_ORDER)
    info = describe_model(model)
    has_moe = len(find_routers(model)) > 0
    if expert_maps is None:
        expert_maps = has_moe
    general = []
    if layer_maps or expert_maps:
        general = general_texts(map_texts)

    scores = {}
    details = {}
    maps = {}
    experts = {}
    for name in skills:
        if screens:
            try:
                result = screen_skill(model, tokenizer, name, scale=scale, max_new_tokens=max_new_tokens,
                                      batch_size=batch_size, thinking=thinking)
                scores[name] = result["score"]
                details[name] = result
                log(f"{model_id} {name}: {result['score']} on {result['count']} items")
            except Exception as error:
                log(f"{model_id} {name}: screen failed: {error!r}"[:500])
        texts = None
        if layer_maps or expert_maps:
            try:
                texts = SKILLS[name]["texts"](map_texts)
            except Exception as error:
                log(f"{model_id} {name}: map texts failed: {error!r}"[:500])
        if texts is not None and layer_maps:
            try:
                maps[name] = layer_skill_map(model, tokenizer, texts, general, batch_size=max(1, batch_size // 2))
                log(f"{model_id} {name}: blocks {maps[name]['blocks']}")
            except Exception as error:
                log(f"{model_id} {name}: layer map failed: {error!r}"[:500])
        if texts is not None and expert_maps:
            try:
                experts[name] = expert_skill_map(model, tokenizer, texts, general)
            except Exception as error:
                log(f"{model_id} {name}: expert map failed: {error!r}"[:500])
        if on_progress is not None:
            on_progress(entry_so_far(model_id, scores, maps, experts, expert_maps, info, details))
    return entry_so_far(model_id, scores, maps, experts, expert_maps, info, details)


def entry_so_far(model_id, scores, maps, experts, expert_maps, info, details):
    expert_field = None
    if expert_maps:
        expert_field = experts
    return make_entry(model_id, scores, maps, expert_field, info["params_b"], info["layers"], info["hidden"],
                      info["family"], details)


def load_zoo(path="atlas/zoo.json"):
    with open(path) as f:
        return json.load(f)


def runnable(zoo_model, image_transformers="4.57.1", giants=False, gated=False):
    """(True, "") when this image can run the model, else (False, reason)."""
    needs = zoo_model["needs"]
    if zoo_model["verified"]["status"] == "gated" and not gated:
        return False, "gated on Hugging Face"
    if zoo_model["verified"]["status"] == "unverified":
        return False, "not verified on Hugging Face"
    if needs["transformers"] != image_transformers:
        return False, f"needs transformers {needs['transformers']}; image has {image_transformers}"
    if needs["multi_gpu"] and not giants:
        return False, "giant (multi_gpu); run with --giants"
    return True, ""
