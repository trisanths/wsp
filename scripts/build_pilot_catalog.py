"""Build the PILOT part set on Modal and write data/pilot_catalog.jsonl.

catalog.jsonl has no graft parts yet, and planners need several choices.
This runs Trisanth's stub extractor (wsp/extract.py, top n MLP neurons by
mean |activation| on GSM8K heal-split text, as modal_graft.py does) on
Qwen/Qwen2.5-Math-1.5B for layers 6, 12, 18 and 24 with n in 256 and 512:
8 parts. The index files go to the wsp-swarm-models volume under
units/pilot/; the records go to data/pilot_catalog.jsonl, marked
"pilot": true. These are not the Person A catalog.

Needs the deployed app (python -m modal deploy modal_swarm.py) and the
Modal tokens in the environment (set -a && . ./.env && set +a). The pilot
catalog is baked into the image at deploy time, so redeploy afterwards.

    .venv/Scripts/python scripts/build_pilot_catalog.py
"""

import argparse
import json
import os
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

import modal_swarm  # noqa: E402

OUT_PATH = os.path.join(REPO, "data", "pilot_catalog.jsonl")


def parse_ints(text):
    values = []
    for piece in text.split(","):
        piece = piece.strip()
        if piece != "":
            values.append(int(piece))
    return values


def write_records(records, path):
    """Write one record per line, sorted by layer then size."""
    folder = os.path.dirname(path)
    if folder != "":
        os.makedirs(folder, exist_ok=True)
    ordered = list(records)
    ordered.sort(key=record_order)
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        for record in ordered:
            handle.write(json.dumps(record, sort_keys=True) + "\n")


def record_order(record):
    return (record["layer"], record["id"])


def check_env():
    for name in ["MODAL_TOKEN_ID", "MODAL_TOKEN_SECRET"]:
        if os.environ.get(name, "") == "":
            raise SystemExit(name + " is not set. Run: set -a && . ./.env && set +a")


def build(layers, sizes, out_path=OUT_PATH):
    """Run extract_parts on Modal and write the catalog. Returns (records, seconds dict)."""
    start = time.time()
    result = modal_swarm.deployed("extract_parts").remote(layers, sizes)
    wall = round(time.time() - start, 2)
    write_records(result["records"], out_path)
    seconds = dict(result["seconds"])
    seconds["client_wall"] = wall
    return result["records"], seconds


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--layers", default="6,12,18,24")
    parser.add_argument("--sizes", default="256,512")
    parser.add_argument("--out", default=OUT_PATH)
    args = parser.parse_args()
    check_env()
    records, seconds = build(parse_ints(args.layers), parse_ints(args.sizes), args.out)
    print("PILOT catalog: " + str(len(records)) + " records written to " + args.out)
    for record in records:
        print("  " + record["id"] + "  units " + record["units"])
    print("seconds: " + json.dumps(seconds, sort_keys=True))
    print("Now redeploy so the image carries the new catalog: .venv/Scripts/python -m modal deploy modal_swarm.py")


if __name__ == "__main__":
    main()
