"""Print the RFT result table from the score rows, with a paired bootstrap interval against the start row.

    python scripts/rft_report.py results/rft k64 k32

Each model directory holds score_start.json, r1/score.json, r2/score.json and optionally others (any
score.json found under the directory, named by its relative path). A row has "scores" and "correct"
(per-item booleans, the same item order in every row of one suite).
"""

import glob
import json
import os
import random
import sys

SUITES = ["humaneval", "mbpp", "gsm8k", "mmlu"]


def paired_interval(a, b, draws=4000, seed=0):
    """(diff, low, high) of mean(a) - mean(b) over the same items, 95 percent paired bootstrap."""
    n = len(a)
    rng = random.Random(seed)
    diffs = []
    for _ in range(draws):
        picks = [rng.randrange(n) for _ in range(n)]
        diffs.append(sum(int(a[i]) - int(b[i]) for i in picks) / n)
    diffs.sort()
    return sum(int(x) for x in a) / n - sum(int(x) for x in b) / n, diffs[int(0.025 * draws)], diffs[int(0.975 * draws) - 1]


def load_rows(root, model):
    rows = {}
    base = os.path.join(root, model)
    start = os.path.join(base, "score_start.json")
    if os.path.exists(start):
        rows["start"] = json.load(open(start))
    for path in sorted(glob.glob(os.path.join(base, "**", "score.json"), recursive=True)):
        rows[os.path.relpath(os.path.dirname(path), base)] = json.load(open(path))
    return rows


def main(root, models):
    for model in models:
        rows = load_rows(root, model)
        if "start" not in rows:
            print(f"{model}: no start row")
            continue
        print(f"\n{model}: pass@1 greedy, % (paired 95% interval of the change vs start in points)")
        print(f"{'row':<14}" + "".join(f"{s:>22}" for s in SUITES + ["heldout"]))
        for name, row in rows.items():
            cells = []
            for suite in SUITES + ["heldout"]:
                value = row["scores"].get(suite)
                if value is None:
                    cells.append(f"{'-':>22}")
                    continue
                text = f"{100 * value:.1f}"
                if name != "start" and suite in row.get("correct", {}) and suite in rows["start"].get("correct", {}):
                    d, lo, hi = paired_interval(row["correct"][suite], rows["start"]["correct"][suite])
                    text += f" ({100 * d:+.1f} [{100 * lo:+.1f},{100 * hi:+.1f}])"
                elif name != "start":
                    text += f" ({100 * (value - rows['start']['scores'][suite]):+.1f})"
                cells.append(f"{text:>22}")
            print(f"{name:<14}" + "".join(cells))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
