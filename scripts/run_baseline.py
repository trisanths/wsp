"""Run one baseline arm at a fixed budget, then its held-out pass, and print the summary.

Examples:
    python scripts/run_baseline.py --method evolve --backend stub --eval-budget 60 --seed 0
    python scripts/run_baseline.py --method random_part --backend stub --eval-budget 60 --seed 0 --cascade
    python scripts/run_baseline.py --method lora --backend stub --eval-budget 60 --compute-budget-s 36000
    python scripts/run_baseline.py --method evolve --backend modal --catalog data/pilot_catalog.jsonl \
        --modal-scorer harness --eval-budget 24 --seed 0

Methods: evolve, random, random_part, greedy, lora, full_ft (wsp/swarm/baselines).
The summary is printed by run_swarm.py's print_summary, so it reads the same
as a swarm run. Stub numbers are plumbing, not results: they are labeled
STUB, and search numbers are labeled SEARCH.
"""

import argparse
import os
import sys

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPTS_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
if SCRIPTS_DIR not in sys.path:
    sys.path.append(SCRIPTS_DIR)

from run_swarm import print_summary, researched_overrides  # noqa: E402
from wsp.swarm.baselines import METHODS, run_baseline  # noqa: E402
from wsp.swarm.config import load_dotenv, make_config  # noqa: E402


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Run one budgeted baseline arm.")
    parser.add_argument("--method", required=True, choices=METHODS)
    parser.add_argument("--backend", default="stub", help="stub | local | modal (lora and full_ft: stub only)")
    parser.add_argument("--eval-budget", type=float, default=60, help="search evals, in full-eval equivalents")
    parser.add_argument("--compute-budget-s", type=float, default=None, help="search compute seconds")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--k", type=int, default=4, help="candidates evaluated per round (as in the swarm)")
    parser.add_argument("--cascade", action="store_true", help="score at low fidelity, promote the top third")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--budget-kind", default="local", help="local | modal")
    parser.add_argument("--outputs-dir", default="outputs")
    parser.add_argument("--catalog", default=None,
                        help="catalog file (default: the stub catalog on stub, else catalog.jsonl); "
                             "Modal runs use data/pilot_catalog.jsonl")
    parser.add_argument("--task-type", default=None,
                        help="an entry of config task_types (default math); Modal interim runs use math_interim")
    parser.add_argument("--modal-scorer", default=None,
                        help="modal backend: interim (gsm8k only, default) | harness (gsm8k + if_guard, no cascade)")
    parser.add_argument("--heldout-top", type=int, default=None,
                        help="programs scored on held-out at the end, each with 2 controls (default 3; 0 skips it)")
    parser.add_argument("--lora", default=None,
                        help="lora_grid with one setting, rank:lr:steps, for example 64:5e-5:300 (a single LoRA run)")
    parser.add_argument("--benchmarks", default=None,
                        help="a researched setup from scripts/research_benchmarks.py (overrides --task-type)")
    return parser.parse_args(argv)


def build_config(args):
    overrides = {
        "method": args.method,
        "backend": args.backend,
        "eval_budget": args.eval_budget,
        "seed": args.seed,
        "k": args.k,
        "cascade": args.cascade,
        "run_id": args.run_id,
        "budget": args.budget_kind,
        "outputs_dir": args.outputs_dir,
    }
    if args.compute_budget_s is not None:
        overrides["compute_budget_s"] = args.compute_budget_s
    if args.catalog is not None:
        overrides["catalog_path"] = args.catalog
    if args.task_type is not None:
        overrides["task_type"] = args.task_type
    if args.modal_scorer is not None:
        overrides["modal_scorer"] = args.modal_scorer
    if args.heldout_top is not None:
        overrides["heldout_top"] = args.heldout_top
    if args.lora is not None:
        pieces = args.lora.split(":")
        if len(pieces) != 3:
            raise SystemExit("--lora takes rank:lr:steps, for example 64:5e-5:300; got " + args.lora)
        overrides["lora_grid"] = {"ranks": [int(pieces[0])], "lrs": [float(pieces[1])], "steps": [int(pieces[2])]}
    if args.benchmarks is not None:
        overrides.update(researched_overrides(args.benchmarks))
    return make_config(overrides)


def main(argv=None):
    args = parse_args(argv)
    names = load_dotenv()
    if len(names) > 0:
        print(".env: " + str(len(names)) + " keys loaded (values hidden)")
    config = build_config(args)
    summary = run_baseline(config)
    print("method: " + summary["method"])
    print_summary(summary)
    return summary


if __name__ == "__main__":
    main()
