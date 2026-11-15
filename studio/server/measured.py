"""Measured results the studio shows next to its estimates.

Every row here was scored by the compiler and written down in team memory.
Nothing here is a stub number. Scores are percent points (0 to 100), as the
scorer printed them. A row with is_control true is a control: it picks the
experts by a rule that should not work, so it shows what chance looks like.
"""

import copy

SOURCE = "compiler session, team memory, 2026-11-14"

CAVEATS = [
    "One run per cell, so a gap of a few points may be noise.",
    "No heal has run yet. These are the raw cuts.",
    "Removing experts keeps the parameters run per token at about 3.0B, so it shrinks storage and memory, not speed.",
    "Other skills fall: MMLU goes from 77.8 to 44.9 at 64 experts.",
    "The 32 expert rows share one size, 8.78B, because the same number of experts per layer gives the same size.",
]


def row(label, experts_kept, rule, total_params_b, scores, is_control=False, is_full=False):
    """One table row."""
    return {"label": label, "experts_kept": experts_kept, "rule": rule, "total_params_b": total_params_b,
            "scores": scores, "is_control": is_control, "is_full": is_full}


RESULTS = [
    {
        "id": "run19_qwen3_30b_a3b_code",
        "title": "Skill removal on Qwen3-30B-A3B, code only",
        "kind": "skill_removal",
        "skill": "code",
        "model": "Qwen/Qwen3-30B-A3B-Instruct-2507",
        "experts_per_layer": 128,
        "params_run_per_token_b": 3.03,
        "date": "2026-11-14",
        "scorer": "wsp.skills_eval",
        "runs_per_cell": 1,
        "heal": "none",
        "source": SOURCE,
        "caveats": CAVEATS,
        "rows": [
            row("Full model", 128, "none", 30.5,
                {"humaneval": 91.5, "mbpp": 84.4, "gsm8k": 91.5, "mmlu": 77.8}, is_full=True),
            row("Keep 64 by code saliency", 64, "saliency", 16.0,
                {"humaneval": 87.8, "mbpp": 79.0, "gsm8k": 80.5, "mmlu": 44.9}),
            row("Keep 32 by code saliency", 32, "saliency", 8.78, {"humaneval": 57.9, "mbpp": 55.3}),
            row("Keep 16 by code saliency", 16, "saliency", 5.15, {"humaneval": 3.7, "mbpp": 7.8}),
            row("Keep 32 by routing frequency", 32, "freq", 8.78, {"humaneval": 7.9, "mbpp": 11.7}),
            row("Keep 32 random (two seeds)", 32, "random", 8.78, {"humaneval": 0.0, "mbpp": 0.0},
                is_control=True),
            row("Keep 32 by skill-to-general ratio", 32, "selective", 8.78, {"humaneval": 0.0, "mbpp": 0.0},
                is_control=True),
        ],
    },
]


def all_results():
    """The measured results, as a fresh copy the caller may change."""
    return copy.deepcopy(RESULTS)
