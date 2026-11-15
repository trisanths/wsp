"""Measured results the studio shows next to its estimates.

Every row here was scored by the compiler and written down in team memory.
Nothing here is a stub number. Scores are percent points (0 to 100), as the
scorer printed them. A row with is_control true is a control: it picks the
experts by a rule that should not work, so it shows what chance looks like.
"""

import copy

SOURCE = "compiler session, results/README.md run 19, 2026-11-14"

CAVEATS = [
    "Each score is from one run, so a gap of a few points may be noise.",
    "Two rows got a 500-step fine-tune after the cut, learning to match the full model on code text. The other rows got no training.",
    "Even fine-tuned, the 32-expert model (73.8 and 61.9) scores below Qwen2.5-Coder-7B (85.4 and 86.0), a coding model of about the same size.",
    "Cutting experts makes the model smaller to store but not faster. Each token still runs through about 3.0B parameters, against 7.6B for Qwen2.5-Coder-7B.",
    "Other skills drop. General knowledge (MMLU) falls from 77.8 to 44.9 with 64 experts kept.",
    "All 32-expert rows are the same size, 8.78B, because size depends only on how many experts are kept.",
]


def row(label, experts_kept, rule, total_params_b, scores, is_control=False, is_full=False, heal_steps=0):
    """One table row. heal_steps is 0 for a raw cut."""
    return {"label": label, "experts_kept": experts_kept, "rule": rule, "total_params_b": total_params_b,
            "scores": scores, "is_control": is_control, "is_full": is_full, "heal_steps": heal_steps}


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
        "heal": "500 steps on two rows",
        "source": SOURCE,
        "caveats": CAVEATS,
        "rows": [
            row("Full model", 128, "none", 30.5,
                {"humaneval": 91.5, "mbpp": 84.4, "gsm8k": 91.5, "mmlu": 77.8}, is_full=True),
            row("Keep 64 by code saliency", 64, "saliency", 16.0,
                {"humaneval": 87.8, "mbpp": 79.0, "gsm8k": 80.5, "mmlu": 44.9}),
            row("Keep 32 by code saliency", 32, "saliency", 8.78,
                {"humaneval": 57.9, "mbpp": 55.3, "gsm8k": 2.5, "mmlu": 26.1}),
            row("Keep 32 by code saliency, 500-step heal", 32, "saliency", 8.78,
                {"humaneval": 73.8, "mbpp": 61.9, "gsm8k": 2.0, "mmlu": 25.6}, heal_steps=500),
            row("Keep 16 by code saliency", 16, "saliency", 5.15,
                {"humaneval": 3.7, "mbpp": 7.8, "gsm8k": 0.5, "mmlu": 23.6}),
            row("Keep 32 by routing frequency", 32, "freq", 8.78,
                {"humaneval": 7.9, "mbpp": 11.7, "gsm8k": 0.0, "mmlu": 24.4}),
            row("Keep 32 random (three seeds)", 32, "random", 8.78, {"humaneval": 0.0, "mbpp": 0.0},
                is_control=True),
            row("Keep 32 random, 500-step heal (seed 1)", 32, "random", 8.78,
                {"humaneval": 11.6, "mbpp": 23.3, "gsm8k": 2.0, "mmlu": 24.1}, is_control=True, heal_steps=500),
            row("Keep 32 by skill-to-general ratio", 32, "selective", 8.78,
                {"humaneval": 0.0, "mbpp": 0.0, "gsm8k": 0.0, "mmlu": 23.9}, is_control=True),
        ],
    },
]


def all_results():
    """The measured results, as a fresh copy the caller may change."""
    return copy.deepcopy(RESULTS)
