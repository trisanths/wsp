# Run several donor models in parallel (one subprocess each; output goes to logs/<model>/console.log).
#   .venv/bin/python mine/run_many.py --models a,b,c [--smoke] [--budget-usd N] [--max-hours H]
import argparse
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", required=True)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--budget-usd", type=float, default=25.0)
    ap.add_argument("--max-hours", type=float, default=3.0)
    ap.add_argument("--max-screens", type=int, default=60)
    ap.add_argument("--effort", default="high")
    ap.add_argument("--person", default=None)
    args = ap.parse_args()

    donors = common.load_donors()
    models = []
    for m in args.models.split(","):
        m = m.strip()
        if m == "":
            continue
        if m not in donors:
            raise SystemExit("unknown model '" + m + "'")
        models.append(m)

    procs = {}
    for m in models:
        cmd = [sys.executable, os.path.join(HERE, "run.py"), "--model", m,
               "--budget-usd", str(args.budget_usd), "--max-hours", str(args.max_hours),
               "--max-screens", str(args.max_screens), "--effort", args.effort]
        if args.smoke:
            cmd.append("--smoke")
        if args.person:
            cmd = cmd + ["--person", args.person]
        os.makedirs(common.logs_dir(m), exist_ok=True)
        log = open(os.path.join(common.logs_dir(m), "console.log"), "a")
        log.write("\n===== " + time.strftime("%Y-%m-%d %H:%M:%S") + " " + " ".join(cmd) + "\n")
        log.flush()
        procs[m] = (subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT), log)
        print("started", m, "-> logs/" + m + "/console.log")

    failed = []
    for m in procs:
        p, log = procs[m]
        code = p.wait()
        log.close()
        print(m, "finished with exit code", code)
        if code != 0:
            failed.append(m)
    if failed:
        print("FAILED:", ", ".join(failed), "(see logs/<model>/console.log)")
        sys.exit(1)


if __name__ == "__main__":
    main()
