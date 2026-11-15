# Re-apply the current light gate to a model's catalog using the stored evidence (no GPU needed).
# Parts that no longer pass move to catalog/<model>.rejected.jsonl with the reasons.
#   .venv/bin/python mine/regate.py --model <short_name>
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402
from client import Miner, light_gate  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    args = ap.parse_args()
    path = common.catalog_path(args.model)
    if not os.path.exists(path):
        print("no catalog for", args.model)
        return
    targets = Miner(args.model).target_skills()
    keep = []
    rejected = []
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            sig = rec.get("signature", {})
            if "random_drops" not in sig or "skill" not in rec:
                keep.append(rec)  # not produced by mine/ (e.g. v1 records): leave alone
                continue
            gate = light_gate(sig["effect"], sig["random_drops"], rec["skill"]["name"], targets)
            if gate["passed"]:
                sig["gate"] = gate
                keep.append(rec)
            else:
                rec["rejected_reasons"] = gate["reasons"]
                rejected.append(rec)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        for rec in keep:
            f.write(json.dumps(rec) + "\n")
    os.replace(tmp, path)
    for rec in rejected:
        common.append_jsonl(os.path.join(common.ROOT, "catalog", args.model + ".rejected.jsonl"), rec)
    print("kept", len(keep), "rejected", len(rejected))
    for rec in rejected:
        print("  rejected", rec["id"], rec["rejected_reasons"])


if __name__ == "__main__":
    main()
