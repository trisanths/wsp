"""Live smoke test of modal_swarm.py on the first pair (Llama-3.2-1B host, Qwen2.5-Math-1.5B donor, GSM8K).

Steps:
 1. deploy modal_swarm.py with WSP_SWARM_GPU set from --gpu (default L4, pinned)
 2. build the pilot catalog (scripts/build_pilot_catalog.py), then redeploy so the image carries it
 3. fill a 3-token pool for a fresh run id
 4. probe one pilot program (fit only, no token); the container charges the probe
 5. check the probe cache is per run: a hit for this run id, a miss for a second run id,
    then probe the same program under the second run id (a fresh compile charged to it)
 6. spend a token, run_candidate on host + one pilot graft, heal 50 steps, eval_limit 20
 7. spend a token, run the same program with control "random" (a warm container)
 8. score the plain host with eval_limit 20 (no token; the shared cache is read first)
 9. check the ledger: exactly 2 spends, 1 token left, one compute row per charged phase,
    compute equal to the charged seconds, and the GPU on every row
10. spend from 4 threads on a fresh 10-token pool: no duplicates, no losses

Every number printed is SMOKE (a plumbing check), SEARCH (search split
only) and INTERIM (interim scorer, 20 items). None of them is a result.
The summary is saved to outputs/smoke/modal_smoke_<time>.json.

    set -a && . ./.env && set +a
    .venv/Scripts/python scripts/modal_smoke.py
    .venv/Scripts/python scripts/modal_smoke.py --skip-build                  # reuse the deployed pilot catalog
    .venv/Scripts/python scripts/modal_smoke.py --skip-deploy --skip-build    # rerun steps 3 to 10
    .venv/Scripts/python scripts/modal_smoke.py --gpu L4,A10                  # dev only: L4 with an A10 fallback
"""

import argparse
import json
import os
import subprocess
import sys
import threading
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)
SCRIPTS = os.path.join(REPO, "scripts")
if SCRIPTS not in sys.path:
    sys.path.insert(0, SCRIPTS)

import build_pilot_catalog  # noqa: E402
import modal_swarm  # noqa: E402

TAG = "[SMOKE SEARCH INTERIM]"
# List prices from docs/research/modal.md section 4, per second. The functions request 2 cores and 16 GiB.
GPU_PER_S = {"L4": 0.000222, "A10": 0.000306}
CPU_CORE_PER_S = 0.0000131
GIB_PER_S = 0.00000222


def rate_for(gpu_name):
    """Container list price per second for a CUDA device name; the dearer A10 rate when unknown."""
    gpu_rate = GPU_PER_S["A10"]
    if "L4" in str(gpu_name):
        gpu_rate = GPU_PER_S["L4"]
    return gpu_rate + 2 * CPU_CORE_PER_S + 16 * GIB_PER_S


def say(text):
    print(TAG + " " + text, flush=True)


def check_env():
    for name in ["MODAL_TOKEN_ID", "MODAL_TOKEN_SECRET"]:
        if os.environ.get(name, "") == "":
            raise SystemExit(name + " is not set. Run: set -a && . ./.env && set +a")


def deploy(gpu):
    """Run `python -m modal deploy modal_swarm.py` with WSP_SWARM_GPU=gpu. Returns wall seconds."""
    # The Modal CLI prints progress symbols; force UTF-8 so a captured Windows pipe does not crash it.
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    env[modal_swarm.GPU_ENV] = gpu
    start = time.time()
    completed = subprocess.run([sys.executable, "-m", "modal", "deploy", "modal_swarm.py"], cwd=REPO,
                               capture_output=True, text=True, encoding="utf-8", errors="replace", env=env)
    seconds = round(time.time() - start, 2)
    if completed.returncode != 0:
        print(completed.stdout[-3000:])
        print(completed.stderr[-3000:])
        raise SystemExit("modal deploy failed")
    return seconds


def program_for(part_id, heal_steps, control=None):
    return {"host": modal_swarm.HOST, "seed": 0, "masks": None, "task_vectors": [],
            "grafts": [{"id": part_id, "at": None, "control": control, "control_seed": 1}],
            "heal": {"steps": heal_steps, "data": "gsm8k_train", "lr": 0.0003, "batch_size": 8, "seq_len": 384}}


def timed_remote(name, *args, **kwargs):
    """Call a deployed function and return (result, client wall seconds)."""
    start = time.time()
    result = modal_swarm.deployed(name).remote(*args, **kwargs)
    return result, round(time.time() - start, 2)


def settle(result, label):
    """Charge any phase the container could not write (normally none). Returns the keys charged here."""
    keys = modal_swarm.charge_uncharged(result)
    if len(keys) > 0:
        say(label + ": the container left " + str(len(keys)) + " phases uncharged; charged them from here")
    return keys


def charged_seconds(result, phases):
    total = 0.0
    for phase in phases:
        total += result["seconds"][phase]
    return total


def report_fit(graft_report):
    for graft in graft_report:
        say("graft " + str(graft["id"]) + " control=" + str(graft["control"]) + " host_layer=" +
            str(graft["host_layer"]) + " fit=" + str(graft["fit"]) + " fit_r2=" + str(graft["fit_r2"]) +
            " fit_pairs=" + str(graft["fit_pairs"]) + " part_params=" + str(graft["part_params"]) +
            " glue_params=" + str(graft["glue_params"]))


def probe_one(label, run_id, program, summary):
    """Probe a program for run_id (the container charges it) and print it. Returns (result, client wall s)."""
    probe, wall = timed_remote("probe_candidate", program, "pilot", run_id=run_id, pool="search",
                               who="smoke-" + label, method="smoke")
    if not probe["ok"]:
        say(label + " FAILED in phase " + str(probe["phase"]) + ": " + str(probe["error"]))
    settle(probe, label)
    say(label + ": setup " + str(probe["setup_seconds"]) + " s (not charged), compile " +
        str(probe["seconds"]["compile"]) + " s (charged), cache_hit " + str(probe["cache_hit"]) + ", warmed " +
        str(probe["warmed"]) + "; client wall " + str(wall) + "; ran on " + str(probe.get("gpu")))
    report_fit(probe["graft_report"])
    summary[label] = {"result": probe, "client_wall_s": wall}
    return probe, wall


def run_one(label, run_id, program, eval_limit, summary):
    """Spend a token, run the candidate (the container charges each phase), and print it."""
    token = modal_swarm.spend(run_id, "search", "smoke-" + label, "smoke")
    if token is None:
        raise SystemExit("no token for " + label + ": the pool is empty or unreachable")
    result, wall = timed_remote("run_candidate", program, "search", token, "pilot", eval_limit, run_id=run_id,
                                pool="search", who="smoke-" + label, method="smoke")
    if not result["ok"]:
        say(label + " FAILED in phase " + str(result["phase"]) + ": " + str(result["error"]))
    settle(result, label)
    say(label + ": token " + str(token) + ", gsm8k accuracy " + str(result["scores"].get("gsm8k")) + " on " +
        str(result["n_items"]) + " items (eval_backend " + result["eval_backend"] + ")")
    say(label + ": setup " + str(result["setup_seconds"]) + " s (not charged); charged compile " +
        str(result["seconds"]["compile"]) + ", heal " + str(result["seconds"]["heal"]) + ", eval " +
        str(result["seconds"]["eval"]) + "; client wall " + str(wall) + " (includes container start); " +
        str(len(result["charged"])) + " compute rows written by the container")
    if result.get("compile_report") is not None:
        report_fit(result["compile_report"]["grafts"])
    losses = result.get("heal_loss", [])
    if len(losses) >= 10:
        say(label + ": heal loss mean of first 10 steps " + str(round(sum(losses[:10]) / 10, 4)) +
            ", last 10 steps " + str(round(sum(losses[-10:]) / 10, 4)))
    say(label + ": ran on " + str(result.get("gpu")) + ", warmed " + str(result["warmed"]))
    summary[label] = {"result": result, "client_wall_s": wall}
    return result


def thread_test(run_id, n_tokens, n_threads):
    """Spend from n_threads threads until the pool is empty. Returns (tokens, wall seconds)."""
    modal_swarm.fill_pool(run_id, "search", n_tokens, "smoke")
    got = []
    lock = threading.Lock()

    def worker(name):
        while True:
            token = modal_swarm.spend(run_id, "search", name, "smoke")
            if token is None:
                return
            with lock:
                got.append(token)

    threads = []
    for index in range(n_threads):
        threads.append(threading.Thread(target=worker, args=("thread-" + str(index),)))
    start = time.time()
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    return got, round(time.time() - start, 2)


def ratio_text(cold, warm):
    if warm <= 0:
        return "n/a"
    return str(round(cold / warm, 2))


def check_scope(summary, run_id, other_run, program, probe, gpu_walls):
    """The probe cache is per run: a hit here, a miss for other_run, and a fresh probe charged to other_run."""
    same_run = modal_swarm.lookup_probe(program, "pilot", run_id)
    other_before = modal_swarm.lookup_probe(program, "pilot", other_run)
    say("probe cache: this run " + str(same_run is not None) + " (want True), run " + other_run + " " +
        str(other_before is not None) + " (want False)")
    other_probe, wall = probe_one("probe_other_run", other_run, program, summary)
    gpu_walls.append((wall, other_probe.get("gpu")))
    other_compute = modal_swarm.compute_used(other_run, "search")
    summary["checks"]["probe_cached_for_its_run"] = same_run is not None
    summary["checks"]["probe_not_shared_across_runs"] = other_before is None and not other_probe["cache_hit"]
    summary["checks"]["other_run_charged_its_own_probe"] = (other_probe["seconds"]["compile"] > 0 and
                                                           abs(other_compute - other_probe["seconds"]["compile"]) < 0.01)
    say("probe cache: " + other_run + " charged " + str(round(other_compute, 2)) + " s for its own probe")
    say("cold vs warm probe compile (charged): first " + str(probe["seconds"]["compile"]) + " s, second " +
        str(other_probe["seconds"]["compile"]) + " s, ratio " +
        ratio_text(probe["seconds"]["compile"], other_probe["seconds"]["compile"]) + "; setup (not charged) " +
        str(probe["setup_seconds"]) + " s and " + str(other_probe["setup_seconds"]) + " s")
    return other_probe


def score_host(eval_limit, gpu_walls):
    """The host score: from the shared cache on this machine if present, else one host_baseline call."""
    host = modal_swarm.lookup_host("search", eval_limit)
    if host is not None:
        host = modal_swarm.as_cache_hit(host)
        say("host: read from the shared cache on this machine, no GPU container started")
    else:
        host, wall = timed_remote("host_baseline", "search", eval_limit)
        gpu_walls.append((wall, host.get("gpu")))
        say("host: setup " + str(host["setup_seconds"]) + " s (not charged), client wall " + str(wall) +
            ", ran on " + str(host.get("gpu")))
    if not host["ok"]:
        say("host FAILED: " + str(host["error"]))
    say("host: gsm8k accuracy " + str(host["scores"].get("gsm8k")) + " on " + str(host["n_items"]) +
        " items; seconds eval " + str(host["seconds"]["eval"]) + " (cached_seconds " +
        str(host.get("cached_seconds")) + "), cache_hit " + str(host["cache_hit"]))
    return host


def check_ledger(summary, run_id, pinned, probe, graft, control):
    """Spends, tokens left, one compute row per charged phase, compute equal to the charged seconds, GPUs."""
    spent = modal_swarm.eval_spent(run_id, "search")
    left = modal_swarm.eval_remaining(run_id, "search")
    compute = modal_swarm.compute_used(run_id, "search")
    devices = modal_swarm.compute_devices(run_id, "search")
    rows = modal_swarm.ledger_rows(run_id)
    n_compute = 0
    for row in rows:
        if row["axis"] == "compute":
            n_compute += 1
    expected = (charged_seconds(probe, ["compile"]) + charged_seconds(graft, ["compile", "heal", "eval"]) +
                charged_seconds(control, ["compile", "heal", "eval"]))
    say("ledger: " + str(spent) + " spends (want 2), queue " + str(left) + " tokens left (want 1), " +
        str(len(rows)) + " rows (" + str(n_compute) + " compute, want 7), compute charged " +
        str(round(compute, 2)) + " s against " + str(round(expected, 2)) + " s of charged phases; devices " +
        str(devices))
    summary["checks"]["two_spends"] = spent == 2
    summary["checks"]["one_token_left"] = left == 1
    summary["checks"]["one_compute_row_per_phase"] = n_compute == 7
    summary["checks"]["compute_equals_charged_phases"] = abs(compute - expected) < 0.01
    summary["checks"]["gpu_on_every_compute_row"] = len(devices) > 0 and "unknown" not in devices
    if pinned:
        summary["checks"]["one_gpu_type"] = len(devices) == 1
    summary["ledger"] = {"spends": spent, "tokens_left": left, "compute_used_s": compute,
                         "charged_phases_s": expected, "devices": devices, "rows": rows}


def main():
    parser = argparse.ArgumentParser(description="Live smoke test of modal_swarm.py.")
    parser.add_argument("--skip-deploy", action="store_true")
    parser.add_argument("--skip-build", action="store_true", help="reuse data/pilot_catalog.jsonl as deployed")
    parser.add_argument("--gpu", default=os.environ.get(modal_swarm.GPU_ENV, modal_swarm.DEFAULT_GPU),
                        help="WSP_SWARM_GPU for the deploy: one type (graded) or a comma list (dev only)")
    parser.add_argument("--part", default="pilot_qmath15_l18_top512")
    parser.add_argument("--heal-steps", type=int, default=50)
    parser.add_argument("--eval-limit", type=int, default=20)
    args = parser.parse_args()
    check_env()
    pinned = not isinstance(modal_swarm.parse_gpu(args.gpu), list)

    stamp = str(int(time.time()))
    run_id = "smoke-" + stamp
    other_run = run_id + "-b"
    summary = {"labels": ["SMOKE", "SEARCH", "INTERIM"], "run_id": run_id, "other_run_id": other_run,
               "gpu_spec": args.gpu, "gpu_pinned": pinned, "heal_steps": args.heal_steps,
               "eval_limit": args.eval_limit, "part": args.part, "checks": {}}
    gpu_walls = []

    if not args.skip_deploy:
        seconds = deploy(args.gpu)
        say("deploy (WSP_SWARM_GPU=" + args.gpu + "): " + str(seconds) + " s")
        summary["deploy_s"] = seconds
    if not args.skip_build:
        records, seconds = build_pilot_catalog.build(modal_swarm.PILOT_LAYERS, modal_swarm.PILOT_SIZES)
        say("pilot catalog: " + str(len(records)) + " parts; seconds " + json.dumps(seconds, sort_keys=True))
        summary["build_catalog"] = {"n_parts": len(records), "seconds": seconds}
        gpu_walls.append((seconds["client_wall"], "unknown"))
        seconds = deploy(args.gpu)
        say("redeploy with the pilot catalog: " + str(seconds) + " s")
        summary["redeploy_s"] = seconds

    names = modal_swarm.create_objects()
    say("objects ready: " + ", ".join(names))
    key = modal_swarm.fill_pool(run_id, "search", 3, "smoke")
    say("filled partition " + key + " with 3 tokens; eval_remaining " +
        str(modal_swarm.eval_remaining(run_id, "search")))

    program = program_for(args.part, args.heal_steps)
    probe, wall = probe_one("probe", run_id, program, summary)
    gpu_walls.append((wall, probe.get("gpu")))
    other_probe = check_scope(summary, run_id, other_run, program, probe, gpu_walls)

    graft = run_one("graft", run_id, program, args.eval_limit, summary)
    gpu_walls.append((summary["graft"]["client_wall_s"], graft.get("gpu")))
    control = run_one("control_random", run_id, program_for(args.part, args.heal_steps, "random"),
                      args.eval_limit, summary)
    gpu_walls.append((summary["control_random"]["client_wall_s"], control.get("gpu")))
    say("cold vs warm compile (charged): graft " + str(graft["seconds"]["compile"]) + " s, control " +
        str(control["seconds"]["compile"]) + " s, ratio " +
        ratio_text(graft["seconds"]["compile"], control["seconds"]["compile"]) + "; setup (not charged) graft " +
        str(graft["setup_seconds"]) + " s, control " + str(control["setup_seconds"]) + " s")

    host = score_host(args.eval_limit, gpu_walls)
    summary["host"] = {"result": host}
    check_ledger(summary, run_id, pinned, probe, graft, control)

    tokens, wall = thread_test(run_id + "-threads", 10, 4)
    ordered = sorted(tokens)
    no_duplicates = len(set(tokens)) == len(tokens)
    no_losses = ordered == list(range(10))
    thread_spent = modal_swarm.eval_spent(run_id + "-threads", "search")
    say("threads: 4 threads took " + str(len(tokens)) + " tokens from 10 in " + str(wall) + " s; no duplicates " +
        str(no_duplicates) + ", no losses " + str(no_losses) + ", ledger spends " + str(thread_spent))
    summary["checks"]["threads_no_duplicates"] = no_duplicates
    summary["checks"]["threads_no_losses"] = no_losses and thread_spent == 10
    summary["threads"] = {"tokens": tokens, "wall_s": wall, "ledger_spends": thread_spent}

    total_wall = 0.0
    estimate = 0.0
    for wall_s, gpu_name in gpu_walls:
        total_wall += wall_s
        estimate += wall_s * rate_for(gpu_name)
    total_wall = round(total_wall, 2)
    estimate = round(estimate, 3)
    say("GPU call wall clock " + str(total_wall) + " s; cost ESTIMATE $" + str(estimate) +
        " at list price (device GPU rate + 2 cores + 16 GiB), excluding the 60 s idle after each burst")
    summary["gpu_call_wall_s"] = total_wall
    summary["cost_estimate_usd"] = estimate
    summary["checks"]["graft_ok"] = graft["ok"]
    summary["checks"]["control_ok"] = control["ok"]
    summary["checks"]["probe_ok"] = probe["ok"] and other_probe["ok"]
    summary["checks"]["host_ok"] = host["ok"]

    out_dir = os.path.join(REPO, "outputs", "smoke")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "modal_smoke_" + stamp + ".json")
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    say("checks: " + json.dumps(summary["checks"], sort_keys=True))
    print("summary written to " + out_path)
    failed = []
    for name in sorted(summary["checks"]):
        if not summary["checks"][name]:
            failed.append(name)
    if len(failed) > 0:
        raise SystemExit("smoke checks failed: " + ", ".join(failed))


if __name__ == "__main__":
    main()
