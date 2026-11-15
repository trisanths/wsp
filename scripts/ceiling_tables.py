"""Print the markdown tables of docs/ceiling.md from the rows in results/ceiling_*.json."""

import json
import os

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")

GROUPS = [
    ("Qwen3-30B-A3B code slim, 64 experts per layer", [
        ("full_30b", "ceiling_full_30b.json", "Full Qwen3-30B-A3B (reference)"),
        ("slim_64", "ceiling_slim_64.json", "Start: slim 64 (round 19, no heal)"),
        ("ceiling_64", "ceiling_row_ceiling_64.json", "Ceiling 64 (bf16)"),
        ("ceiling_64_int4", "ceiling_row_ceiling_64_int4.json", "Ceiling 64, INT4 w4a16 on one L4"),
        ("slim_64_int4", "ceiling_slim_64_int4.json", "Start slim 64, INT4 RTN on one L4 (reference)"),
    ], ("humaneval", "mbpp")),
    ("Keep-32 healed slim (8.8B)", [
        ("slim_32_heal", "ceiling_slim_32_heal.json", "Start: slim 32 healed 500 steps"),
        ("ceiling_32", "ceiling_row_ceiling_32.json", "Ceiling 32 (bf16)"),
        ("ceiling_32_int4", "ceiling_row_ceiling_32_int4.json", "Ceiling 32, INT4 w4a16 on one L4"),
    ], ("humaneval", "mbpp")),
    ("Clinic slim, 64 experts per layer", [
        ("clinic_full", "ceiling_clinic_full.json", "Full Qwen3-30B-A3B (reference)"),
        ("clinic_slim_64", "ceiling_clinic_slim_64.json", "Start: round 28 clinic_64"),
        ("clinic_ceiling_64", "ceiling_row_clinic_ceiling_64.json", "Ceiling clinic (bf16)"),
        ("clinic_ceiling_64_int4", "ceiling_row_clinic_ceiling_64_int4.json", "Ceiling clinic, INT4 w4a16 on one L4"),
    ], ("medmcqa_val1000", "medqa_test", "mmlu_medical_test")),
]


def load(name):
    path = os.path.join(RESULTS, name)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def cell(row, key):
    if row is None or row.get("error"):
        return "n/a"
    if key not in row or row[key] is None:
        return "n/a"
    value = row[key]
    return f"{row[key + '_correct']}/{row[key + '_n']} = {100 * value:.1f}"


def main():
    for title, rows, scores in GROUPS:
        print(f"### {title}\n")
        head = ["Model", "GPU", *scores, "tok/s b1", "tok/s b32", "TTFT ms", "Weights GiB", "GPU used GiB"]
        print("| " + " | ".join(head) + " |")
        print("|" + "---|" * len(head))
        for _cell, file, label in rows:
            row = load(file)
            if row is None or row.get("error"):
                print(f"| {label} | not run | " + " | ".join(["n/a"] * (len(head) - 2)) + " |")
                continue
            tps = row["decode_tps"]
            print("| " + " | ".join([label, row["gpu"].replace("NVIDIA ", ""),
                                     *[cell(row, s) for s in scores],
                                     f"{tps['1']:.0f}", f"{tps['32']:.0f}", f"{row['ttft_ms_median']:.0f}",
                                     f"{row['weights_gib']:.1f}", f"{row['gpu_after_load_gib']:.1f}"]) + " |")
        print()


if __name__ == "__main__":
    main()
