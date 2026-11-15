"""Pull the write-in rows from the Modal volumes of both workspaces and print a table.

    .venv/bin/python scripts/wi_rows.py [prefix]
"""
import glob
import json
import os
import subprocess
import sys

OUT = "/private/tmp/claude-501/wi_rows"
KEYS = ["mmlu_dev", "mc_eval", "mmlu_heldout", "arc_c_test", "trivia_em", "gsm8k", "ce_last50", "soft_last50",
        "kd_last50", "trainable", "train_gpu_hours", "gpu_hours", "decode_tok_s"]


def pull(track="wi_bigsame"):
    modal = os.path.join(os.path.dirname(sys.executable), "modal")
    for profile in ("ztrisanth", "trisanthsrini"):
        dest = f"{OUT}/{profile}"
        os.makedirs(dest, exist_ok=True)
        env = dict(os.environ, MODAL_PROFILE=profile)
        subprocess.run([modal, "volume", "get", "wsp-models", f"{track}/rows", dest, "--force"], env=env,
                       capture_output=True)


def main():
    prefix = sys.argv[1] if len(sys.argv) > 1 else ""
    pull()
    seen = {}
    for f in sorted(glob.glob(OUT + "/*/**/*.json", recursive=True)):
        r = json.load(open(f))
        if "smoke" in r["name"] or not r["name"].startswith(prefix):
            continue
        seen[r["name"]] = r
    for name, r in seen.items():
        vals = []
        for k in KEYS:
            v = r.get(k)
            vals.append(f"{k}={round(v, 4) if isinstance(v, float) else v}")
        extra = r.get("mmlu_heldout_graft_off")
        print(name, " ".join(vals), "" if extra is None else f"graft_off={round(extra, 4)}")


main()
