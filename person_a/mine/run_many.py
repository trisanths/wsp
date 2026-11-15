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

# One focus per parallel agent on the same model; they share findings through past.dev memory.
FOCUSES = [
    "attention heads in every layer: read o_proj/v_proj weights through the unembedding; find heads that copy, "
    "carry signs, digits or syntax",
    "MLP neurons in the first half of the layers: read gate/up rows against the embedding to find input features",
    "MLP neurons in the second half of the layers: read down_proj columns through the unembedding to find output "
    "features",
    "MoE experts if the model has them (router weights, expert down projections), otherwise whole-layer blocks; "
    "and invent new skill labels the others have not tried",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", required=True)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--budget-usd", type=float, default=25.0)
    ap.add_argument("--max-hours", type=float, default=3.0)
    ap.add_argument("--max-screens", type=int, default=60)
    ap.add_argument("--effort", default="high")
    ap.add_argument("--person", default=None)
    ap.add_argument("--agents", type=int, default=1, help="parallel agents per model (each gets a different focus)")
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
        n_agents = args.agents
        if args.smoke:
            n_agents = 1
        for i in range(n_agents):
            cmd = [sys.executable, "-u", os.path.join(HERE, "run.py"), "--model", m,
                   "--budget-usd", str(args.budget_usd), "--max-hours", str(args.max_hours),
                   "--max-screens", str(args.max_screens), "--effort", args.effort]
            if args.smoke:
                cmd.append("--smoke")
            if args.person:
                cmd = cmd + ["--person", args.person]
            env = dict(os.environ)
            name = m
            log_name = "console.log"
            if n_agents > 1:
                cmd = cmd + ["--focus", FOCUSES[i % len(FOCUSES)]]
                env["WSP_LANE"] = str(i)
                name = m + "#" + str(i)
                log_name = "console.lane" + str(i) + ".log"
            os.makedirs(common.logs_dir(m), exist_ok=True)
            log = open(os.path.join(common.logs_dir(m), log_name), "a")
            log.write("\n===== " + time.strftime("%Y-%m-%d %H:%M:%S") + " " + " ".join(cmd) + "\n")
            log.flush()
            procs[name] = (subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env), log)
            print("started", name, "-> logs/" + m + "/" + log_name)
            time.sleep(2)  # stagger starts

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
