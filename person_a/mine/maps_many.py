# Run mine/maps.py for many models, a few at a time (one subprocess per model).
#   .venv/bin/python mine/maps_many.py --zoo-keys A,B,C --jobs 3        # models of atlas/zoo.json
#   .venv/bin/python mine/maps_many.py --all-zoo --max-params-b 40 --jobs 3
#   .venv/bin/python mine/maps_many.py --zoo-keys A,B,C --modules       # also the per-module maps
# Models that already have an entry in atlas/pool_entries/ are skipped. With --modules, a model is
# skipped only when its entry already has module_maps. Logs: atlas/pool_logs/<key>.log
# A --modules run that fails leaves atlas/pool_logs/<key>.modules-failed, and later --modules runs
# do not start that model again (a model that fails the same way each time would cost a GPU call
# each time). --retry-failed starts it again; a run that ends well removes the file.
# When maps.py reports an engine without skill_maps(modules=...), no more models are started.
# Set MODAL_PROFILE to choose the Modal workspace. Stop with Ctrl-C (kill -INT), never kill -9.
import argparse
import datetime
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import maps  # noqa: E402


def maps_command(key, modules):
    cmd = [sys.executable, "-u", os.path.join(HERE, "maps.py"), "--zoo-key", key]
    if modules:
        cmd.append("--modules")
    return cmd


def failed_path(log_dir, key):
    return os.path.join(log_dir, key + ".modules-failed")


def drop_failed(todo, log_dir, modules, retry):
    # Returns (models to start, models left out because their last --modules run failed)
    if not modules or retry:
        return list(todo), []
    kept = [key for key in todo if not os.path.exists(failed_path(log_dir, key))]
    return kept, [key for key in todo if key not in kept]


def note_exit(log_dir, key, code, modules):
    # Remembers a --modules run that failed and forgets it after one that ended well. A run that a
    # signal stopped (negative code) and a run that found an old engine say nothing about the model.
    if not modules:
        return
    path = failed_path(log_dir, key)
    if code == 0:
        if os.path.exists(path):
            os.remove(path)
    elif code > 0 and code != maps.OLD_ENGINE_EXIT:
        now = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        with open(path, "w") as f:
            json.dump({"exit": code, "at": now}, f)


def run_all(todo, jobs, modules, log_dir, command=maps_command, poll=5):
    # Runs command(key, modules) for every key, `jobs` at a time. Returns (failed, not_started,
    # old_engine): after a run ends with maps.OLD_ENGINE_EXIT no more models are started, because
    # the deployed engine would refuse every one of them.
    todo = list(todo)
    running = {}
    failed = []
    not_started = []
    old_engine = False
    while todo or running:
        while todo and len(running) < jobs:
            key = todo.pop(0)
            log = open(os.path.join(log_dir, key + ".log"), "a")
            running[key] = (subprocess.Popen(command(key, modules), stdout=log, stderr=subprocess.STDOUT), log,
                            time.time())
            print("started", key, flush=True)
        time.sleep(poll)
        for key in list(running.keys()):
            p, log, t0 = running[key]
            code = p.poll()
            if code is None:
                continue
            log.close()
            del running[key]
            print(key, "exit", code, "in", round(time.time() - t0), "s", flush=True)
            note_exit(log_dir, key, code, modules)
            if code != 0:
                failed.append(key)
            if code == maps.OLD_ENGINE_EXIT:
                old_engine = True
                not_started.extend(todo)
                todo = []
    return failed, not_started, old_engine


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zoo-keys", default="")
    ap.add_argument("--all-zoo", action="store_true")
    ap.add_argument("--max-params-b", type=float, default=80.0, help="skip bigger models (multi-GPU tiers)")
    ap.add_argument("--min-params-b", type=float, default=0.0)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--jobs", type=int, default=3)
    ap.add_argument("--modules", action="store_true", help="passed to maps.py: also map attention and feed-forward")
    ap.add_argument("--retry-failed", action="store_true",
                    help="with --modules: also start the models whose last --modules run failed")
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
        if maps.entry_done(out, args.modules):
            continue
        todo.append(m["key"])
    todo = [k for i, k in enumerate(sorted(todo)) if i % args.nshards == args.shard]
    missing = [k for k in wanted if k not in [m["key"] for m in zoo]]
    if missing:
        print("not in zoo:", ", ".join(missing))
    log_dir = maps.LOG_DIR
    os.makedirs(log_dir, exist_ok=True)
    todo, left_out = drop_failed(todo, log_dir, args.modules, args.retry_failed)
    if left_out:
        print("not started, the last --modules run failed (use --retry-failed):", ", ".join(left_out))
    print(len(todo), "models to map, jobs", args.jobs, flush=True)
    failed, not_started, old_engine = run_all(todo, args.jobs, args.modules, log_dir)
    if failed:
        print("FAILED:", ", ".join(failed), "(see atlas/pool_logs/<key>.log)")
    if old_engine:
        print(maps.OLD_ENGINE_MESSAGE)
        if not_started:
            print("not started:", ", ".join(not_started))
        sys.exit(maps.OLD_ENGINE_EXIT)
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
