"""Print the clinic score tables from modal_clinic.py score rows.

    python scripts/clinic_report.py full=results/clinic/full_score.json k64=results/clinic/k64_score.json ... [--ref full]

Each argument is label=path of a score row (written by modal_clinic.score_model). The first label is the start row
of the table. Prints two tables, chain-of-thought and letter log-probability accuracy per set, in percent, with a
95 percent paired bootstrap interval of the difference against --ref (default: the first row) for the cot table.
"""

import json
import random
import sys

SETS = ["medmcqa", "medqa", "pubmedqa", "mmlu_med6", "mmlu_general_ex_med6"]
NAMES = {"medmcqa": "MedMCQA val 1000", "medqa": "MedQA test", "pubmedqa": "PubMedQA test", "mmlu_med6": "MMLU medical 6",
         "mmlu_general_ex_med6": "MMLU general 1000"}


def paired_interval(a, b, draws=2000, seed=0):
    """(mean diff, low, high) in points of a minus b over the same items, 95 percent bootstrap."""
    n = len(a)
    rng = random.Random(seed)
    diffs = [int(x) - int(y) for x, y in zip(a, b)]
    mean = sum(diffs) / n
    samples = []
    for _ in range(draws):
        total = 0
        for _ in range(n):
            total += diffs[rng.randrange(n)]
        samples.append(100.0 * total / n)
    samples.sort()
    return 100.0 * mean, samples[int(0.025 * draws)], samples[int(0.975 * draws)]


def load(args):
    rows = []
    for arg in args:
        label, _, path = arg.partition("=")
        with open(path) as f:
            rows.append((label, json.load(f)))
    return rows


def fmt(value):
    return "n/a" if value is None else f"{100.0 * value:.1f}"


def table(rows, kind, ref_label):
    ref = dict(rows)[ref_label]
    lines = ["| model | " + " | ".join(NAMES[s] for s in SETS) + " |", "|---|" + "---|" * len(SETS)]
    for label, row in rows:
        cells = []
        for s in SETS:
            value = (row.get(kind) or {}).get(s)
            cell = fmt(value)
            if kind == "cot" and label != ref_label and value is not None:
                mine = (row.get("correct") or {}).get("cot", {}).get(s)
                theirs = (ref.get("correct") or {}).get("cot", {}).get(s)
                if mine and theirs and len(mine) == len(theirs):
                    d, lo, hi = paired_interval(mine, theirs)
                    cell += f" ({d:+.1f} [{lo:+.1f},{hi:+.1f}])"
            cells.append(cell)
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main(argv):
    ref = None
    args = []
    i = 0
    while i < len(argv):
        if argv[i] == "--ref":
            ref = argv[i + 1]
            i += 2
        else:
            args.append(argv[i])
            i += 1
    rows = load(args)
    ref = ref or rows[0][0]
    print("Chain of thought, greedy (percent; brackets: paired 95 percent interval of the change vs " + ref + ")\n")
    print(table(rows, "cot", ref))
    print("\nLetter log-probability (percent)\n")
    print(table(rows, "lp", ref))


if __name__ == "__main__":
    main(sys.argv[1:])
