"""Fetch a hillclimb batch's rows from the Modal volume and print a table.

    .venv/bin/python scripts/hill_table.py h3
"""
import glob
import json
import os
import subprocess
import sys


def fetch(batch):
    out = f"results/hill/{batch}"
    os.makedirs(out, exist_ok=True)
    subprocess.run([".venv/bin/modal", "volume", "get", "--force", "wsp-models", f"/hill/rows/{batch}/", out + "/"],
                   check=False, capture_output=True)
    rows = []
    for path in sorted(glob.glob(f"{out}/**/*.json", recursive=True)):
        with open(path) as f:
            rows.append(json.load(f))
    return rows


def pct(x):
    return "" if x is None else f"{100 * x:.1f}"


def main(batch):
    rows = fetch(batch)
    panel_keys = ["medical", "math", "science", "cs", "humanities", "medmcqa"]
    mc_keys = ["general", "science", "math", "medicine"]
    head = ["name", "dev", "dev_off", "held", "mc"] + ["mc:" + k for k in mc_keys] + panel_keys + ["surv", "gate", "law"]
    print(" | ".join(head))
    for r in sorted(rows, key=lambda r: r["config"]["name"]):
        gate = None
        for key in ("mc_gate_curve", "kd_gate_curve", "mc_gate_gate_curve"):
            if r.get(key):
                gate = r[key][-1]
        cells = [r["config"]["name"], pct(r.get("mmlu_dev")), pct(r.get("mmlu_dev_off")), pct(r.get("mmlu_heldout")),
                 pct(r.get("mc_eval"))]
        cells += [pct((r.get("mc_evals") or {}).get(k)) for k in mc_keys]
        cells += [pct((r.get("panel") or {}).get(k)) for k in panel_keys]
        surv = (r.get("alignment_end") or r.get("alignment") or {}).get("survival")
        cells += ["" if surv is None else f"{surv:.2f}", "" if gate is None else f"{gate:.3f}",
                  f"{r['law']['drift_kl']:.3f}" if r.get("law", {}).get("drift_kl") is not None else ""]
        print(" | ".join(cells))


if __name__ == "__main__":
    main(sys.argv[1])
