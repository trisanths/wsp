"""Create part 3's Modal objects and, optionally, fill one eval-token pool.

Creates (if missing) Queue wsp-budget, Dicts wsp-ledger, wsp-cache,
wsp-probe, wsp-exp and Volume wsp-swarm-models in the workspace the Modal
tokens belong to. With --run-id and --n it fills the pool for that run.

    set -a && . ./.env && set +a
    .venv/Scripts/python scripts/modal_setup.py
    .venv/Scripts/python scripts/modal_setup.py --run-id s0-swarm-1 --pool search --n 60
    .venv/Scripts/python scripts/modal_setup.py --run-id s0-swarm-1 --pool search --status

A pool can be filled once per run id and lives 24 h after the fill.
"""

import argparse
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

import modal_swarm  # noqa: E402


def check_env():
    for name in ["MODAL_TOKEN_ID", "MODAL_TOKEN_SECRET"]:
        if os.environ.get(name, "") == "":
            raise SystemExit(name + " is not set. Run: set -a && . ./.env && set +a")


def print_status(run_id, pool):
    remaining = modal_swarm.eval_remaining(run_id, pool)
    spent = modal_swarm.eval_spent(run_id, pool)
    compute = modal_swarm.compute_used(run_id, pool)
    devices = modal_swarm.compute_devices(run_id, pool)
    print("run " + run_id + " pool " + pool + ": " + str(remaining) + " tokens left, " + str(spent) +
          " spent, " + str(round(compute, 1)) + " compute seconds charged on " + str(devices))
    if len(devices) > 1:
        print("WARNING: this run's compute rows mix GPU types " + str(devices) +
              "; its GPU-seconds are not comparable with a single-type run")


def main():
    parser = argparse.ArgumentParser(description="Create part 3's Modal objects and fill a token pool.")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--pool", default="search", help="search | heldout (any name without '|')")
    parser.add_argument("--n", type=int, default=None, help="tokens to put in the pool (0..n-1)")
    parser.add_argument("--method", default="", help="method name written on the fill row")
    parser.add_argument("--status", action="store_true", help="print the pool's counters and exit")
    args = parser.parse_args()
    check_env()

    names = modal_swarm.create_objects()
    print("objects ready: " + ", ".join(names))
    if args.run_id is None:
        return
    if args.status:
        print_status(args.run_id, args.pool)
        return
    if args.n is None:
        raise SystemExit("--n is required to fill a pool (or pass --status)")
    key = modal_swarm.fill_pool(args.run_id, args.pool, args.n, args.method)
    print("filled partition " + key + " with tokens 0.." + str(args.n - 1))
    print_status(args.run_id, args.pool)


if __name__ == "__main__":
    main()
