"""Summarise results/wi_clinic/rows/*.json into a markdown table (per seed rows, then mean and sample sd per arm).

    python scripts/wi_clinic_table.py [rows_dir] > results/wi_clinic/table.md
"""
import glob
import json
import statistics as st
import sys

SETS = ["medmcqa", "mmlu_med", "medqa", "mmlu_held", "mmlu_bio"]
HEAD = ["MedMCQA val 1000", "MMLU medical 945", "MedQA test 1273", "MMLU held-out 2000", "MMLU college bio 144"]
ORDER = ["base", "native3b", "graft_both", "graft_med", "graft_qwen", "identity_both", "scrambled_both",
         "lora_match", "lora_big", "fullft"]


def load(root):
    rows = {}
    for path in sorted(glob.glob(f"{root}/*.json")):
        r = json.load(open(path))
        rows.setdefault(r["name"], []).append(r)
    return rows


def main(root):
    rows = load(root)
    print("| arm | seed | " + " | ".join(HEAD) + " | params/token | trainable | train tokens | GPU h | tok/s |")
    print("|---|---|" + "---|" * (len(SETS) + 5))
    for name in ORDER:
        rs = sorted(rows.get(name, []), key=lambda r: r["seed"])
        for r in rs:
            s = r["scores"]
            print(f"| {name} | {r['seed']} | " + " | ".join(f"{100 * s[k]:.1f}" for k in SETS)
                  + f" | {r.get('params_per_token', '')} | {r.get('trainable', '')} | {r.get('tokens_trained', '')}"
                  + f" | {r['gpu_hours']} | {r.get('decode_tps', '')} |")
        if len(rs) > 1:
            cells = []
            for k in SETS:
                v = [100 * r["scores"][k] for r in rs]
                cells.append(f"{st.mean(v):.1f} +-{st.stdev(v):.1f}")
            print(f"| {name} mean, seeds {[r['seed'] for r in rs]} | | " + " | ".join(cells) + " | | | | | |")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results/wi_clinic/rows")
