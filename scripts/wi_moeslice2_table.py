"""Summarise results/wi_moeslice2/rows/*.json into a markdown table (final cells only: no tag).

Usage: python scripts/wi_moeslice2_table.py [rows dir]
"""
import glob
import json
import statistics as st
import sys

SETS = ["mmlu_held", "arc_c", "obqa", "mmlu_dev"]
ORDER = ["reference", "native_qwen3_4b", "native_qwen3_8b", "graft", "identity", "scrambled", "lora_match",
         "lora_big", "fullft"]
root = sys.argv[1] if len(sys.argv) > 1 else "results/wi_moeslice2/rows"
rows = {}
for path in sorted(glob.glob(f"{root}/*.json")):
    r = json.load(open(path))
    if "name" not in r or r["config"].get("tag"):
        continue
    rows.setdefault(r["name"], []).append(r)
print("| arm | seed | MMLU held-out 2000 | ARC-C test | OBQA test | MMLU dev 1000 | params total | params/token "
      "| trainable | train tokens | GPU h | tok/s |")
print("|---|---|---|---|---|---|---|---|---|---|---|---|")
for name in ORDER:
    rs = sorted(rows.get(name, []), key=lambda r: r["seed"])
    for r in rs:
        s = r["scores"]
        print(f"| {name} | {r['seed']} | " + " | ".join(f"{100 * s[k]:.1f}" for k in SETS)
              + f" | {r.get('params_total', '')} | {r.get('params_per_token', '')} | {r.get('trainable', '')}"
              + f" | {r.get('tokens_trained', '')} | {r['gpu_hours']} | {r.get('decode_tps', '')} |")
    if len(rs) > 1:
        cells = []
        for k in SETS:
            v = [100 * r["scores"][k] for r in rs]
            cells.append(f"{st.mean(v):.1f} +-{st.stdev(v):.1f}")
        print(f"| {name} mean, seeds {[r['seed'] for r in rs]} | | " + " | ".join(cells) + " | | | | | | |")
