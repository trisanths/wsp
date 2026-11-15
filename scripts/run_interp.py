"""Run the interp agent: overlaps, probes, signatures, capability maps.

    python scripts/run_interp.py --ids mask_mmlupro_biology_llama1b mask_mmlupro_chemistry_llama1b --max-evals 40

Probe numbers are search-split numbers. Budget namespace: "interp".
Experiments are logged to interp_log.jsonl; signatures go into catalog.jsonl.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from wsp_eval.interp_agent import InterpAgent  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ids", nargs="*", default=None, help="catalog ids to study (default: all)")
    parser.add_argument("--tasks", nargs="*", default=None, help="tasks to measure (default: the components' own skills)")
    parser.add_argument("--max-evals", type=int, default=80)
    parser.add_argument("--band-size", type=int, default=4)
    parser.add_argument("--catalog", default=None)
    args = parser.parse_args()

    agent = InterpAgent(catalog_path=args.catalog, tasks=args.tasks, max_evals=args.max_evals,
                        band_size=args.band_size, ids=args.ids)
    print(agent.run())


if __name__ == "__main__":
    main()
