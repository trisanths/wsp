"""Training splits of benchmarks, as (user, assistant) pairs, for ranking and heal texts.

Owner rule of 2026-11-15: a compile, a ranking, and a heal may read the TRAIN
split of a benchmark. A score must come only from TEST items that nothing
trained on or ranked on. Callers drop overlaps with wsp.slim.drop_overlap.

The pairs keep the gold answers, so they are fit for ranking (which only reads
routing statistics) and for a distillation heal where the loss compares the
student with the full model's logits. They are not fit for an LM loss on the
gold text in a foreign style (LESSONS.md).

Sources, one function per source, each a generator in a fixed seeded order:
  medmcqa  openlifescienceai/medmcqa train, rows that carry an explanation
  gsm8k    openai/gsm8k main train
  math     EleutherAI/hendrycks_math, the seven subject configs, train
  casehold coastalcph/lex_glue case_hold train (CaseHOLD, 45000 rows)

TRAIN_SOURCES maps a request skill to the functions it reads. pairs(skill, n)
takes the sources round robin, like wsp.healdata.pairs.
"""

import random
import re

from wsp import healdata

LETTERS = "ABCDE"
MATH_SUBJECTS = ["algebra", "counting_and_probability", "geometry", "intermediate_algebra", "number_theory",
                 "prealgebra", "precalculus"]
MIN_EXPLANATION_CHARS = 40


def _order(count, seed):
    order = list(range(count))
    random.Random(seed).shuffle(order)
    return order


def medmcqa_pair(row):
    """(user, assistant) for one MedMCQA train row, or None when it has no usable explanation."""
    explanation = (row.get("exp") or "").strip()
    if len(explanation) < MIN_EXPLANATION_CHARS:
        return None
    choices = [row.get("opa"), row.get("opb"), row.get("opc"), row.get("opd")]
    if any(choice is None or not str(choice).strip() for choice in choices):
        return None
    gold = row.get("cop")
    if gold is None or int(gold) not in (0, 1, 2, 3):
        return None
    gold = int(gold)
    lines = ["Question: " + str(row["question"]).strip()]
    for index, choice in enumerate(choices):
        lines.append(LETTERS[index] + ". " + str(choice).strip())
    user = "\n".join(lines) + "\nChoose the correct option and explain why."
    assistant = "Answer: " + LETTERS[gold] + ". " + str(choices[gold]).strip() + "\n\n" + explanation
    return user, assistant


def clean_gsm8k_answer(answer):
    """The GSM8K solution without calculator tags, ending in 'The answer is N.'"""
    text = re.sub(r"<<[^>]*>>", "", answer)
    if "####" not in text:
        return None
    body, final = text.rsplit("####", 1)
    final = final.strip()
    if not final:
        return None
    return body.strip() + "\nThe answer is " + final + "."


def gsm8k_pair(row):
    answer = clean_gsm8k_answer(row.get("answer") or "")
    question = (row.get("question") or "").strip()
    if answer is None or not question:
        return None
    return question + "\nSolve step by step and end with: The answer is N.", answer


def math_pair(row):
    problem = (row.get("problem") or "").strip()
    solution = (row.get("solution") or "").strip()
    if not problem or "\\boxed" not in solution:
        return None
    return problem + "\nSolve step by step and put the final answer in \\boxed{}.", solution


def casehold_pair(row):
    """(user, assistant) for one CaseHOLD row: the citing text with a holding blank, five endings, the gold one."""
    endings = row.get("endings")
    context = (row.get("context") or "").strip()
    label = row.get("label")
    if not context or not endings or len(endings) != 5 or label is None or int(label) not in range(5):
        return None
    label = int(label)
    lines = ["Which holding does the citing text rely on? The text ends where the holding is cited.", "",
             "Citing text: " + context, ""]
    for index, ending in enumerate(endings):
        lines.append(LETTERS[index] + ". " + str(ending).strip())
    assistant = "Answer: " + LETTERS[label] + ". " + str(endings[label]).strip()
    return "\n".join(lines), assistant


def _rows(dataset_id, config, split):
    return healdata._load(dataset_id, config, split)


def medmcqa_train(seed=0):
    rows = _rows("openlifescienceai/medmcqa", None, "train")
    for index in _order(len(rows), seed):
        pair = medmcqa_pair(rows[index])
        if pair is not None:
            yield pair


def gsm8k_train(seed=0):
    rows = _rows("openai/gsm8k", "main", "train")
    for index in _order(len(rows), seed):
        pair = gsm8k_pair(rows[index])
        if pair is not None:
            yield pair


def math_train(seed=0):
    streams = []
    for subject in MATH_SUBJECTS:
        rows = _rows("EleutherAI/hendrycks_math", subject, "train")
        streams.append((rows, _order(len(rows), seed)))
    position = 0
    live = True
    while live:
        live = False
        for rows, order in streams:
            if position < len(order):
                live = True
                pair = math_pair(rows[order[position]])
                if pair is not None:
                    yield pair
        position += 1


def casehold_train(seed=0):
    rows = _rows("coastalcph/lex_glue", "case_hold", "train")
    for index in _order(len(rows), seed):
        pair = casehold_pair(rows[index])
        if pair is not None:
            yield pair


TRAIN_SOURCES = {
    "medicine": [medmcqa_train],
    "math": [gsm8k_train, math_train],
    "law": [casehold_train],
}


def pairs(skill, count, seed=0):
    """count (user, assistant) pairs from the benchmark train splits of a skill, round robin over its sources."""
    if skill not in TRAIN_SOURCES:
        raise ValueError(f"no benchmark train source for {skill!r}; the skills are {', '.join(TRAIN_SOURCES)}")
    streams = [source(seed) for source in TRAIN_SOURCES[skill]]
    out = []
    live = list(streams)
    while len(out) < count and live:
        still = []
        for stream in live:
            if len(out) >= count:
                still.append(stream)
                continue
            pair = next(stream, None)
            if pair is None:
                continue
            out.append(pair)
            still.append(stream)
        live = still
    return out
