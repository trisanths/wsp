"""Split one Modal run's cells across several Modal workspaces, then merge.

The target script's local entrypoint must accept --shard and --nshards,
keep only cells where index % nshards == shard, and write its rows to
results/<name>_shard<shard>.json. Each workspace has its own volume and
Hugging Face cache, so the first run in a workspace downloads models.

Usage:
    python scripts/modal_fanout.py modal_round8.py round8 \\
        --profiles trisanth,trisanthsrini,ztrisanth,trisanthsrinivasan,trisanthcollege,trisanths

Writes results/<name>_rows.json (all shards) and logs to
outputs/fanout_<name>/<profile>.log.
"""

import argparse
import json
import os
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("script")
    parser.add_argument("name")
    parser.add_argument("--profiles", required=True)
    parser.add_argument("--extra", default="", help="extra args passed to the entrypoint, space separated")
    args = parser.parse_args()

    profiles = args.profiles.split(",")
    log_dir = os.path.join("outputs", "fanout_" + args.name)
    os.makedirs(log_dir, exist_ok=True)
    modal_bin = os.path.join(os.path.dirname(sys.executable), "modal")

    processes = []
    for shard, profile in enumerate(profiles):
        command = [modal_bin, "run", args.script, "--shard", str(shard), "--nshards", str(len(profiles))]
        if args.extra:
            command += args.extra.split(" ")
        env = dict(os.environ)
        env["MODAL_PROFILE"] = profile
        log = open(os.path.join(log_dir, profile + ".log"), "w")
        processes.append((profile, subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT), log))
        print(f"shard {shard} on {profile}")

    failed = []
    for profile, process, log in processes:
        code = process.wait()
        log.close()
        print(f"{profile} exited {code}")
        if code != 0:
            failed.append(profile)

    rows = []
    for shard in range(len(profiles)):
        path = os.path.join("results", f"{args.name}_shard{shard}.json")
        if not os.path.exists(path):
            print(f"missing {path}")
            continue
        with open(path) as f:
            rows += json.load(f)
    with open(os.path.join("results", f"{args.name}_rows.json"), "w") as f:
        json.dump(rows, f, indent=2)
    print(f"merged {len(rows)} rows; failed profiles: {failed}")


if __name__ == "__main__":
    main()
