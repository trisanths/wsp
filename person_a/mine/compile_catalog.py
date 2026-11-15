# Merge the per-model catalogs (catalog/<model>.jsonl) into one catalog.jsonl in the team format.
#   .venv/bin/python mine/compile_catalog.py [--out catalog.jsonl]
# Records from other sources already in the output file (e.g. the public deletion masks) are kept;
# records produced by the mining pipeline are replaced by the current per-model files.
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402

REQUIRED = ["id", "type", "trained_on", "signature", "model", "units", "d_model"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(common.ROOT, "catalog.jsonl"))
    args = ap.parse_args()

    mined = []
    cat_dir = os.path.join(common.ROOT, "catalog")
    files = []
    if os.path.isdir(cat_dir):
        files = sorted(os.listdir(cat_dir))
    for name in files:
        if not name.endswith(".jsonl") or name.endswith(".rejected.jsonl"):
            continue
        with open(os.path.join(cat_dir, name)) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                missing = []
                for k in REQUIRED:
                    if k not in rec:
                        missing.append(k)
                if "layer" not in rec and "layers" not in rec:
                    missing.append("layer/layers")
                if missing:
                    print("skipping", rec.get("id"), "missing", missing)
                    continue
                if not os.path.exists(os.path.join(common.ROOT, rec["units"])):
                    print("skipping", rec["id"], "index file not found:", rec["units"])
                    continue
                rec["source"] = "mine"
                mined.append(rec)

    kept = []
    if os.path.exists(args.out):
        with open(args.out) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                if rec.get("source") != "mine":
                    kept.append(rec)

    seen = set()
    out = []
    for rec in kept + mined:
        if rec["id"] in seen:
            continue
        seen.add(rec["id"])
        out.append(rec)
    tmp = args.out + ".tmp"
    with open(tmp, "w") as f:
        for rec in out:
            f.write(json.dumps(rec) + "\n")
    os.replace(tmp, args.out)

    by_model = {}
    for rec in mined:
        by_model[rec["model"]] = by_model.get(rec["model"], 0) + 1
    print("wrote", args.out, "|", len(kept), "kept from other sources +", len(mined), "mined parts")
    for model in sorted(by_model):
        print("  ", model, by_model[model])


if __name__ == "__main__":
    main()
