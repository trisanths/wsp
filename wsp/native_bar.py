"""The native bar: the best plain model that fits the same per-token budget as a compile.

A write-in is only worth compiling when it beats every native model that
runs no more parameters per token. A requester can always pick a stronger
small base instead. This module holds measured native scores and answers
one question for the planner: which native model sets the bar for a
budget, and does a compile clear it?

All scores are MMLU held-out 2000 in the wi_xfam harness (zero-shot letter
log-prob, the same items as results/wi_xfam/table.md), seed 0, measured
2026-11-15. "tuned" is the same four-option fine-tune the full fine-tune
arm of wi_xfam got, with the learning rate picked on MMLU dev from three
values (1200 steps). Rows: results/wi_lean/rows/.
"""

METRIC = "mmlu_held"

NATIVE = [
    {"model": "unsloth/Llama-3.2-1B-Instruct", "params_per_token": 1235814400, "plain": 44.6, "tuned": 51.1,
     "tuned_note": "3 seeds, 4000 steps, lr 3e-6 (wi_xfam full fine-tune)",
     "source": "results/wi_xfam/table.md"},
    {"model": "Qwen/Qwen2.5-1.5B-Instruct", "params_per_token": 1543714304, "plain": 60.35, "tuned": 61.7,
     "tuned_note": "seed 0, 1200 steps, lr 1e-6 picked on dev",
     "source": "results/wi_lean/rows/reference_0_ref_qwen1.5b.json, fullft_0_pilot_qwen1.5b_1e-6.json"},
    {"model": "Qwen/Qwen2.5-3B-Instruct", "params_per_token": 3085938688, "plain": 65.95, "tuned": 66.6,
     "tuned_note": "seed 0, 1200 steps, lr 3e-6 picked on dev",
     "source": "results/wi_lean/rows/reference_0_ref_qwen3b.json, fullft_0_pilot_qwen3b_3e-6.json"},
    {"model": "unsloth/Llama-3.2-3B-Instruct", "params_per_token": 3212749824, "plain": 57.35, "tuned": 63.4,
     "tuned_note": "seed 0, 1200 steps, lr 1e-6 picked on dev",
     "source": "results/wi_lean/rows/reference_0_ref_llama3b.json, fullft_0_pilot_llama3b_1e-6.json"},
]


def best_native(per_token, natives=None, exclude=()):
    """The native row with the highest score among those running at most per_token parameters, or None.

    The score of a row is its tuned score when it has one, else its plain
    score, since a requester can fine-tune a native base too. Models in
    exclude (for example the host itself) are skipped.
    """
    rows = NATIVE if natives is None else natives
    best = None
    for row in rows:
        if row["model"] in exclude or row["params_per_token"] > per_token:
            continue
        if best is None or score_of(row) > score_of(best):
            best = row
    return best


def score_of(row):
    if row.get("tuned") is not None:
        return row["tuned"]
    return row["plain"]


def check(per_token, expected, natives=None, exclude=()):
    """Compare a compile's expected score with the native bar at its per-token cost.

    Returns {"bar": row or None, "bar_score", "expected", "clears": True/False/None, "text"}.
    clears is None when no native model fits the budget or the compile has
    no expected score.
    """
    bar = best_native(per_token, natives, exclude)
    out = {"metric": METRIC, "per_token_b": round(per_token / 1e9, 3), "bar": bar, "expected": expected,
           "bar_score": None if bar is None else score_of(bar), "clears": None}
    if bar is None:
        out["text"] = "no measured native model runs %.2fB or fewer parameters per token" % (per_token / 1e9)
        return out
    if expected is None:
        out["text"] = ("native bar at %.2fB per token: %s, %s %.1f (%s); this compile has no measured score to compare"
                       % (per_token / 1e9, bar["model"], METRIC, score_of(bar), bar["source"]))
        return out
    out["clears"] = expected > score_of(bar)
    verb = "clears" if out["clears"] else "does not clear"
    out["text"] = ("native bar at %.2fB per token: %s scores %s %.1f (%s, %s). The measured write-in scores %.1f, so it %s "
                   "the bar" % (per_token / 1e9, bar["model"], METRIC, score_of(bar), "tuned" if bar.get("tuned") is not None
                                else "plain", bar["source"], expected, verb))
    if not out["clears"]:
        out["text"] += ("; picking %s as the base gives more for less. A write-in is worth it only on a host that is "
                        "already the best at its size, with a donor that knows what the host family lacks" % bar["model"])
    return out
