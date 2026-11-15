# Run mine/maps.py for many models, a few at a time (one subprocess per model).
#   .venv/bin/python mine/maps_many.py --zoo-keys A,B,C --jobs 3        # models of atlas/zoo.json
#   .venv/bin/python mine/maps_many.py --all-zoo --max-params-b 40 --jobs 3
# Models that already have an entry in atlas/pool_entries/ are skipped. Logs: atlas/pool_logs/<key>.log
# Set MODAL_PROFILE to choose the Modal workspace. Stop with Ctrl-C (kill -INT), never kill -9.
import argparse
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import maps  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zoo-keys", default="")
    ap.add_argument("--all-zoo", action="store_true")
    ap.add_argument("--max-params-b", type=float, default=80.0, help="skip bigger models (multi-GPU tiers)")
    ap.add_argument("--min-params-b", type=float, default=0.0)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--jobs", type=int, default=3)
    args = ap.parse_args()
    with open(maps.ZOO) as f:
        zoo = json.load(f)["models"]
    wanted = [k.strip() for k in args.zoo_keys.split(",") if k.strip()]
    todo = []
    for m in zoo:
        if wanted and m["key"] not in wanted:
            continue
        if not wanted and not args.all_zoo:
            continue
        if m.get("params_b") is None:
            if wanted:
                print("no params_b in the zoo, skipped:", m["key"])
            continue
        if not (args.min_params_b <= float(m["params_b"]) <= args.max_params_b):
            continue
        out = os.path.join(maps.OUT_DIR, m["id"].replace("/", "__") + ".json")
        if os.path.exists(out):
            continue
        todo.append(m["key"])
    todo = [k for i, k in enumerate(sorted(todo)) if i % args.nshards == args.shard]
    missing = [k for k in wanted if k not in [m["key"] for m in zoo]]
    if missing:
        print("not in zoo:", ", ".join(missing))
    log_dir = os.path.join(os.path.dirname(maps.OUT_DIR), "pool_logs")
    os.makedirs(log_dir, exist_ok=True)
    print(len(todo), "models to map, jobs", args.jobs, flush=True)
    running = {}
    failed = []
    while todo or running:
        while todo and len(running) < args.jobs:
            key = todo.pop(0)
            log = open(os.path.join(log_dir, key + ".log"), "a")
            cmd = [sys.executable, "-u", os.path.join(HERE, "maps.py"), "--zoo-key", key]
            running[key] = (subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT), log, time.time())
            print("started", key, flush=True)
        time.sleep(5)
        for key in list(running.keys()):
            p, log, t0 = running[key]
            code = p.poll()
            if code is None:
                continue
            log.close()
            del running[key]
            print(key, "exit", code, "in", round(time.time() - t0), "s", flush=True)
            if code != 0:
                failed.append(key)
    if failed:
        print("FAILED:", ", ".join(failed), "(see atlas/pool_logs/<key>.log)")
        sys.exit(1)


if __name__ == "__main__":
    main()
