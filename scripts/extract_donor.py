"""Extraction agent: find skill units in a donor, write index files, validate the best ones.

    python scripts/extract_donor.py --model Qwen/Qwen2.5-Math-1.5B --task gsm8k
    python scripts/extract_donor.py --model Qwen/Qwen2.5-Math-1.5B --validate 2   # also ablation-check the top 2

Writes outputs/extracted/<model>/*.pt (index files + random controls) and
records.jsonl for Person A. Every step is logged to interp_log.jsonl as
hypothesis, action, result. Validation evals are charged to namespace "interp".
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from wsp_eval import extraction  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--task", default="gsm8k")
    parser.add_argument("--kinds", nargs="*", default=None, help="mlp_cluster attn_heads expert (default: all)")
    parser.add_argument("--layers", nargs="*", type=int, default=None)
    parser.add_argument("--neurons", type=int, default=64)
    parser.add_argument("--heads", type=int, default=2)
    parser.add_argument("--experts", type=int, default=4)
    parser.add_argument("--top-layers", type=int, default=3)
    parser.add_argument("--texts", type=int, default=256)
    parser.add_argument("--validate", type=int, default=0, help="ablation-check the top N parts (3 evals each)")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    records = extraction.extract(
        args.model, task_name=args.task, kinds=args.kinds, layers=args.layers,
        n_units={"mlp_cluster": args.neurons, "attn_heads": args.heads, "expert": args.experts},
        top_layers=args.top_layers, n_texts=args.texts, out_dir=args.out)

    ordered = sorted(records, key=lambda r: r["signature"]["mean_effect"], reverse=True)
    for record in ordered[:args.validate]:
        result = extraction.validate(record["units"], record["controls"][0], tasks=[args.task])
        record["signature"]["ablation"] = result

    for record in records:
        signature = record["signature"]
        line = "%-45s effect %.2f (random %.2f)" % (record["id"], signature["mean_effect"], signature["random_mean_effect"])
        if "ablation" in signature:
            task_result = signature["ablation"]["tasks"][args.task]
            line += "  ablation drop %.3f vs random %.3f [%s]" % (task_result["drop"], task_result["random_drop"],
                                                                signature["ablation"]["label"])
        print(line)

    if len(records) > 0:
        out_dir = os.path.dirname(records[0]["units"])
        with open(os.path.join(out_dir, "records.jsonl"), "w") as f:
            for record in records:
                f.write(json.dumps(record) + "\n")
        print("records for Person A:", os.path.join(out_dir, "records.jsonl"))


if __name__ == "__main__":
    main()
