"""Run one swarm search, then the held-out pass, and print a short summary.

Examples:
    python scripts/run_swarm.py --backend stub --eval-budget 60 --seed 0
    python scripts/run_swarm.py --backend modal --catalog data/pilot_catalog.jsonl --task-type math_interim \
        --eval-budget 12 --seed 0
    python scripts/run_swarm.py --backend modal --catalog data/pilot_catalog.jsonl --modal-scorer harness \
        --eval-budget 12 --seed 0

Stub numbers are plumbing, not results: they are labeled STUB, and search
numbers are labeled SEARCH.
"""

import argparse
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from wsp.swarm.config import load_dotenv, make_config  # noqa: E402
from wsp.swarm.swarm import run_swarm  # noqa: E402


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Run one budgeted swarm search.")
    parser.add_argument("--backend", default="stub", help="stub | local | modal")
    parser.add_argument("--eval-budget", type=float, default=60, help="search evals, in full-eval equivalents")
    parser.add_argument("--compute-budget-s", type=float, default=None, help="search compute seconds")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--planners", type=int, default=3)
    parser.add_argument("--k", type=int, default=4, help="candidates evaluated per round")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--budget-kind", default="local", help="local | modal")
    parser.add_argument("--memory", default="off", help="off | past")
    parser.add_argument("--literature", default="off", help="off | replay | live")
    parser.add_argument("--llm", default="none", help="none | mock | modal_vllm | openai_compat | anthropic")
    parser.add_argument("--cascade", action="store_true", help="score at low fidelity, promote the top third")
    parser.add_argument("--outputs-dir", default="outputs")
    parser.add_argument("--catalog", default=None,
                        help="catalog file (default: the stub catalog on stub, else catalog.jsonl); "
                             "Modal runs use data/pilot_catalog.jsonl")
    parser.add_argument("--task-type", default=None,
                        help="an entry of config task_types (default math); Modal interim runs use math_interim")
    parser.add_argument("--modal-scorer", default=None,
                        help="modal backend: interim (gsm8k only, default) | harness (gsm8k + if_guard, no cascade)")
    return parser.parse_args(argv)


def build_config(args):
    overrides = {
        "backend": args.backend,
        "eval_budget": args.eval_budget,
        "seed": args.seed,
        "planners": args.planners,
        "k": args.k,
        "run_id": args.run_id,
        "budget": args.budget_kind,
        "memory": args.memory,
        "literature": args.literature,
        "llm": args.llm,
        "cascade": args.cascade,
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
    return make_config(overrides)


def format_score(value):
    if value is None:
        return "none"
    return "%.4f" % value


def print_summary(summary):
    stub_label = ""
    if summary["stub"]:
        stub_label = " [STUB]"
    print("run " + summary["run_id"] + "  backend " + summary["backend"] + stub_label)
    print("evals used: " + str(summary["eval_used"]) + " / " + str(summary["eval_budget"]) + " FEE" + stub_label)
    line = ("compute used: " + "%.0f" % summary["compute_used_s"] + " s / " + "%.0f" % summary["compute_budget_s"] + " s")
    if summary["compute_overshoot_s"] > 0:
        line += " (overshoot " + "%.0f" % summary["compute_overshoot_s"] + " s)"
    print(line + stub_label)
    best = summary["best_search"]
    if best is None:
        print("best search score [SEARCH]" + stub_label + ": none (empty frontier)")
    else:
        print("best search score [SEARCH]" + stub_label + ": " + format_score(best["combined"]) + "  " + best["describe"])
    print("held-out rows" + stub_label + ":")
    for row in summary["heldout"]:
        print("  " + format_score(row["combined"]) + "  " + row["status"] + "  " + row["describe"])
    for row in summary["controls"]:
        print("  " + format_score(row["combined"]) + "  control " + str(row["control"]) + " of " +
              str(row["control_of"]) + "  " + row["describe"])
    print("log: " + summary["log_path"])


def main(argv=None):
    args = parse_args(argv)
    names = load_dotenv()
    if len(names) > 0:
        print(".env: " + str(len(names)) + " keys loaded (values hidden)")
    config = build_config(args)
    summary = run_swarm(config)
    print_summary(summary)
    return summary


if __name__ == "__main__":
    main()
