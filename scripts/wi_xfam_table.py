"""Summarise results/wi_xfam/rows/*_final.json (and the seed 0 4000-step dev runs) into a markdown table."""
import glob
import json
import statistics as st
import sys

SETS = ["mmlu_held", "arc_c", "obqa", "mmlu_dev"]
ORDER = ["reference", "graft_lora", "graft_wide_lora", "identity_l", "scrambled_l", "lora_match_l", "lora_big", "fullft"]
SEED0 = {"graft_lora": "graft_lora_0_s4k", "graft_wide_lora": "graft_wide_lora_0_s4k",
         "lora_match_l": "lora_match_l_0_s4k", "lora_big": "lora_big_0_s4k5e5", "fullft": "fullft_0_s4k"}
root = sys.argv[1] if len(sys.argv) > 1 else "results/wi_xfam/rows"
rows = {}
for path in sorted(glob.glob(f"{root}/*_final.json")):
    r = json.load(open(path))
    rows.setdefault(r["name"], []).append(r)
for name, stem in SEED0.items():
    try:
        rows.setdefault(name, []).append(json.load(open(f"{root}/{stem}.json")))
    except OSError:
        pass
print("| arm | seed | MMLU held-out 2000 | ARC-C test | OBQA test | MMLU dev 1000 | params total | params/token | trainable | train tokens | GPU h | tok/s |")
print("|---|---|---|---|---|---|---|---|---|---|---|---|")
for name in ORDER:
    rs = sorted(rows.get(name, []), key=lambda r: r["seed"])
    for r in rs:
        s = r["scores"]
        print(f"| {name} | {r['seed']} | " + " | ".join(f"{100*s[k]:.1f}" for k in SETS)
              + f" | {r.get('params_total','')} | {r.get('params_per_token','')} | {r.get('trainable','')}"
              + f" | {r.get('tokens_trained','')} | {r['gpu_hours']} | {r.get('decode_tps','')} |")
    fresh = [r for r in rs if r["seed"] > 0 or name == "reference"]
    if len(fresh) > 1:
        cells = []
        for k in SETS:
            v = [100 * r["scores"][k] for r in fresh]
            cells.append(f"{st.mean(v):.1f} +-{st.stdev(v):.1f}")
        print(f"| {name} mean, seeds {[r['seed'] for r in fresh]} | | " + " | ".join(cells) + " | | | | | | |")
