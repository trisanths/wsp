"""Scoring sets and practice sets for round 20. Never used for training.

Like wsp.skills_eval, this module reads reported benchmarks, so it is exempt
from tests/test_no_benchmark_training.py. Nothing here feeds a heal, a
stitch, a router, a gate, or a distillation. Practice items only drive the
compile-time guard and the component test in wsp.synth, and they share no
item with the scoring items.
"""

import random

COUNTS = {"gsm8k": 300, "mbpp": 200, "mmlu": 1000, "medmcqa": 1000}


def scoring_items(ev, skills, heldout_mmlu=False):
    """The reported items. mmlu: round 6 dev 1000, or with heldout_mmlu the order[1000:3000] items of modal_round7.mmlu_split."""
    from datasets import load_dataset

    out = {}
    for skill in skills:
        if skill == "gsm8k":
            out[skill] = ev.gsm8k_items(COUNTS["gsm8k"])
        elif skill == "mbpp":
            items = ev.mbpp_items()
            random.Random(0).shuffle(items)
            out[skill] = items[:COUNTS["mbpp"]]
        elif skill == "medmcqa":
            out[skill] = ev.medmcqa_items(COUNTS["medmcqa"])
        elif skill == "mmlu" and not heldout_mmlu:
            out[skill] = ev.mmlu_items(COUNTS["mmlu"])
    if heldout_mmlu:
        test = load_dataset("cais/mmlu", "all", split="test")
        order = list(range(len(test)))
        random.Random(0).shuffle(order)
        items = []
        for index in order[1000:3000]:
            row = test[index]
            items.append({"id": f"mmlu_ho:{index}", "prompt": ev.mcq_prompt(row["question"], row["choices"]),
                          "gold": int(row["answer"]), "subject": row["subject"]})
        out["mmlu"] = items
    return out


def practice_sets(ev, skills):
    """Practice items for the guard and the component test. Splits the scoring sets do not use. No gradient."""
    from datasets import load_dataset

    out = {}
    for skill in skills:
        if skill == "gsm8k":
            rows = load_dataset("openai/gsm8k", "main", split="train")
            order = list(range(len(rows)))
            random.Random(5).shuffle(order)
            items = []
            for index in order[:100]:
                row = rows[index]
                items.append({"id": f"gsm8k_tr:{index}", "question": row["question"], "answer": row["answer"],
                              "gold": ev.final_number(row["answer"])})
            out[skill] = items
        elif skill == "mbpp":
            items = []
            for split in ("validation", "train"):
                rows = load_dataset("google-research-datasets/mbpp", "sanitized", split=split)
                for index in range(len(rows)):
                    row = rows[index]
                    items.append({"id": f"mbpp_{split}:{row['task_id']}", "text": row.get("prompt") or row.get("text"),
                                  "test_list": list(row["test_list"]),
                                  "test_imports": list(row.get("test_imports") or [])})
            out[skill] = items[:100]
        elif skill == "mmlu":
            rows = load_dataset("cais/mmlu", "all", split="validation")
            order = list(range(len(rows)))
            random.Random(5).shuffle(order)
            items = []
            for index in order[:150]:
                row = rows[index]
                items.append({"id": f"mmlu_val:{index}", "prompt": ev.mcq_prompt(row["question"], row["choices"]),
                              "gold": int(row["answer"]), "subject": row["subject"]})
            out[skill] = items
        elif skill == "medmcqa":
            rows = load_dataset("openlifescienceai/medmcqa", split="train")
            order = list(range(len(rows)))
            random.Random(5).shuffle(order)
            items = []
            for index in order:
                row = rows[index]
                if int(row["cop"]) not in (0, 1, 2, 3):
                    continue
                choices = [row["opa"], row["opb"], row["opc"], row["opd"]]
                items.append({"id": f"medmcqa_tr:{index}", "prompt": ev.mcq_prompt(row["question"], choices),
                              "gold": int(row["cop"]), "subject": row.get("subject_name") or "unknown"})
                if len(items) >= 150:
                    break
            out[skill] = items
    return out
