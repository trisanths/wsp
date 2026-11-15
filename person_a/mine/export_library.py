# Copies every validated part into the team's top-level library/ folder.
#
#   .venv/bin/python mine/export_library.py --repo /path/to/wsp
#
# Reads:  person_a/catalog/<model>.jsonl   (v2 mining runs; *.rejected.jsonl are skipped)
#         person_a/catalog.jsonl           (v1 dissect runs)
# Writes: library/<model>/<part id>.pt     (copy of the index file)
#         library/<model>/parts.jsonl      (records, with "units" pointing at library/...)
#         library/catalog.jsonl            (all records from all models, one line each)
# Safe to re-run: it rebuilds library/ files from the sources each time.
import argparse
import glob
import json
import os
import shutil


def read_jsonl(path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def short_name_for(record, units_path):
    # parts/<short>/<file>.pt -> <short>
    pieces = units_path.replace("\\", "/").split("/")
    if len(pieces) >= 3 and pieces[0] == "parts":
        return pieces[1]
    return record.get("model", "unknown").replace("/", "__")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True, help="root of the team repo (contains person_a/)")
    args = ap.parse_args()
    src = os.path.join(args.repo, "person_a")
    lib = os.path.join(args.repo, "library")

    sources = []
    for path in sorted(glob.glob(os.path.join(src, "catalog", "*.jsonl"))):
        if path.endswith(".rejected.jsonl"):
            continue
        for r in read_jsonl(path):
            r["pipeline"] = "v2-mine"
            sources.append(r)
    v1_path = os.path.join(src, "catalog.jsonl")
    if os.path.exists(v1_path):
        for r in read_jsonl(v1_path):
            r["pipeline"] = "v1-dissect"
            if "tier" not in r:
                r["tier"] = "provisional"
            sources.append(r)

    per_model = {}
    seen_ids = []
    missing = []
    for r in sources:
        if r["id"] in seen_ids:
            continue
        units = r.get("units", "")
        pt_src = os.path.join(src, units)
        if not os.path.exists(pt_src):
            missing.append(r["id"])
            continue
        short = short_name_for(r, units)
        out_dir = os.path.join(lib, short)
        os.makedirs(out_dir, exist_ok=True)
        pt_name = os.path.basename(units)
        shutil.copyfile(pt_src, os.path.join(out_dir, pt_name))
        r["units"] = "library/" + short + "/" + pt_name  # relative to the repo root
        if short not in per_model:
            per_model[short] = []
        per_model[short].append(r)
        seen_ids.append(r["id"])

    all_rows = []
    for short in sorted(per_model):
        with open(os.path.join(lib, short, "parts.jsonl"), "w") as f:
            for r in per_model[short]:
                f.write(json.dumps(r) + "\n")
                all_rows.append(r)
    with open(os.path.join(lib, "catalog.jsonl"), "w") as f:
        for r in all_rows:
            f.write(json.dumps(r) + "\n")

    print("library:", len(all_rows), "parts across", len(per_model), "models")
    for short in sorted(per_model):
        print("  ", short, len(per_model[short]))
    if missing:
        print("skipped (index file missing):", missing)


if __name__ == "__main__":
    main()
