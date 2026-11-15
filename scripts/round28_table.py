"""Build the round 28 tables and the side-by-side samples from the per-cell JSON files.

    modal volume get wsp-models results/round28 results/round28_cells   (with MODAL_PROFILE=trisanthsrini)
    python scripts/round28_table.py results/round28_cells --out results/round28_rows.json \
        --tables demo/results_tables.md --samples demo/sample_outputs.json

The cell files hold the scores, parameter counts, disk bytes, kept experts, and the greedy answers to the
fixed prompts. Files whose name starts with smoke_ are skipped.
"""

import argparse
import json
import os

REQUESTS = {
    "clinic": ["medmcqa", "mmlu_medical", "mmlu_general_ex_medical"],
    "accounting": ["gsm8k", "math500", "mmlu_accounting", "mmlu_general_ex_accounting"],
    "law_firm": ["mmlu_law", "mmlu_general_ex_law"],
}
ORDER = ["sal64", "sal48", "sal32_heal500", "random48_s0"]
LABELS = {"sal64": "keep 64 by saliency, no heal", "sal48": "keep 48 by saliency, no heal",
          "sal32_heal500": "keep 32 by saliency, 500-step heal", "random48_s0": "keep 48 random (seed 0), no heal"}
HEADS = {"medmcqa": "MedMCQA val 1000", "mmlu_medical": "MMLU medical test (945)",
         "mmlu_general_ex_medical": "MMLU general 1000 (no medical)", "gsm8k": "GSM8K test 500",
         "math500": "MATH-500", "mmlu_accounting": "MMLU accounting + statistics test (498)",
         "mmlu_general_ex_accounting": "MMLU general 1000 (no acct/stats)", "mmlu_law": "MMLU law test (1763)",
         "mmlu_general_ex_law": "MMLU general 1000 (no law)"}


def load(folder):
    rows = {}
    for name in sorted(os.listdir(folder)):
        if not name.endswith(".json") or name.startswith("smoke_"):
            continue
        with open(os.path.join(folder, name)) as f:
            rows[name[:-5]] = json.load(f)
    return rows


def pct(value):
    return f"{100 * value:.1f}"


def params_b(row):
    return f"{row['params_total_after'] / 1e9:.2f}"


def table(request, rows):
    benches = REQUESTS[request]
    lines = ["| cell | total params (B) | run per token (B) | " + " | ".join(HEADS[b] for b in benches)
             + " | disk (GB) | source |", "|" + "---|" * (len(benches) + 5)]
    names = ["full_30b"] + [f"{request}_{tail}" for tail in ORDER]
    for name in names:
        row = rows.get(name)
        if row is None or row.get("scores") is None:
            lines.append(f"| {name} | missing | | " + " | ".join("" for _ in benches) + " | | |")
            continue
        label = "full model" if name == "full_30b" else LABELS[name[len(request) + 1:]]
        cells = [pct(row["scores"][b]) for b in benches]
        lines.append(f"| {label} | {params_b(row)} | {row['params_run_after'] / 1e9:.2f} | " + " | ".join(cells)
                     + f" | {row['disk_bytes'] / 1e9:.1f} | results/round28_rows.json cell {name} |")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("folder")
    parser.add_argument("--out", default="results/round28_rows.json")
    parser.add_argument("--tables", default="demo/results_tables.md")
    parser.add_argument("--samples", default="demo/sample_outputs.json")
    args = parser.parse_args()
    rows = load(args.folder)
    slim = {}
    for name, row in rows.items():
        slim[name] = {key: row[key] for key in row if key not in ("correct", "kept", "skills_detail", "samples")}
    with open(args.out, "w") as f:
        json.dump(slim, f, indent=2)
    parts = []
    for request in REQUESTS:
        text = rows[f"{request}_sal64"]["request_text"] if f"{request}_sal64" in rows else ""
        parts.append(f"### {request}\n\n\"{text}\"\n\n{table(request, rows)}\n")
    with open(args.tables, "w") as f:
        f.write("\n".join(parts))
    samples = {}
    for name in ["full_30b"] + [f"{r}_{t}" for r in REQUESTS for t in ORDER]:
        if name in rows and rows[name].get("samples"):
            samples[name] = rows[name]["samples"]
    with open(args.samples, "w") as f:
        json.dump(samples, f, indent=2)
    print("\n".join(parts))


if __name__ == "__main__":
    main()
