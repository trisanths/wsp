"""Run the equal-budget comparison: every condition, every seed, the same two budgets.

Example:
    python scripts/compare.py --name stub-e60 --backend stub --eval-budget 60 --seeds 3

Conditions (spec section 5, rule 10):
    swarm        scripted analyst, planners and judge (no LLM)
    swarm_llm    LLM analyst and planners; skipped with a message when no LLM is configured
    evolve       regularized evolution over grafts (the primary baseline)
    random       random search
    random_part  the random-part search arm
    greedy       greedy add-if-better
    lora         LoRA on the host (compared on the compute axis)
    full_ft      full fine-tuning (compared on the compute axis)

Swarm conditions call wsp.swarm.swarm.run_swarm(config). Baselines call
wsp.swarm.baselines.run_baseline(config), imported only when a baseline is
asked for. Every run gets the same seed list, the same eval budget and the
same compute budget, and goes through the same WspEnv, budget and log.

--llm-config adds LLM, memory and literature keys to the swarm_llm runs
only. It is an allowlist (see llm_key_allowed): a key in config.DEFAULTS or a
shared space key such as glue_ranks is refused, so swarm_llm cannot change
what every arm shares.

Writes outputs/compare/<name>/manifest.json: the conditions, seeds, budgets,
git commit, backend, priors sha (if any) and each run's log path. Its
budget_check compares every run's own manifest with the four budget caps and
the shared settings (fidelities, fit_threshold, k, cascade, glue_ranks,
stub_seed); a mismatch makes the command exit 1. Each run's
own files stay in outputs/runs/<run_id>/. Every best score printed or saved
here is a search-split number, labeled SEARCH, and STUB on the stub backend.
"""

import argparse
import datetime
import importlib
import json
import os
import re
import subprocess
import sys
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from wsp.swarm import factory  # noqa: E402
from wsp.swarm.budget import Ledger  # noqa: E402
from wsp.swarm.config import DEFAULTS, ledger_path, load_dotenv, make_config, public_config  # noqa: E402

CONDITIONS = ["swarm", "swarm_llm", "evolve", "random", "random_part", "greedy", "lora", "full_ft"]
SWARM_CONDITIONS = ["swarm", "swarm_llm"]
NAME_PATTERN = re.compile(r"^[A-Za-z0-9._-]+$")

# --llm-config is an allowlist: it may set only LLM, memory and literature
# keys, named by these prefixes or names, and never a key in DEFAULTS.
LLM_CONFIG_PREFIXES = ["llm_", "openai_", "anthropic_", "modal_llm_", "mock_", "memory_", "past_", "exa_"]
LLM_CONFIG_NAMES = ["priors_path"]
# Keys every arm shares that are not in DEFAULTS (space.validate reads glue_ranks).
SHARED_SPACE_KEYS = ["glue_ranks"]

# Settings besides the four budget caps that every run must share. The
# manifest records them and budget_check compares each run's own manifest.
SHARED_SETTINGS = ["fidelities", "fit_threshold", "k", "cascade", "glue_ranks", "stub_seed"]
# A run manifest may leave these out; missing means the space default.
OPTIONAL_SETTINGS = ["glue_ranks"]


class CompareError(RuntimeError):
    """The comparison cannot run as asked (unknown condition, missing module, name taken)."""


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Run every condition at one equal budget, for seeds 0..S-1.")
    parser.add_argument("--name", default=None, help="compare folder name (default: <backend>-e<budget>-<time>)")
    parser.add_argument("--conditions", default=",".join(CONDITIONS),
                        help="comma list from: " + ", ".join(CONDITIONS))
    parser.add_argument("--seeds", type=int, default=3, help="S: runs seeds 0..S-1 for every condition")
    parser.add_argument("--eval-budget", type=float, default=60, help="search evals per run, in FEE")
    parser.add_argument("--compute-budget-s", type=float, default=None, help="search compute seconds per run")
    parser.add_argument("--heldout-eval-budget", type=float, default=None)
    parser.add_argument("--heldout-compute-budget-s", type=float, default=None)
    parser.add_argument("--backend", default="stub", help="stub | local | modal")
    parser.add_argument("--budget-kind", default="local", help="local | modal")
    parser.add_argument("--llm", default="none", help="LLM for swarm_llm: none | mock | modal_vllm | openai_compat | anthropic")
    parser.add_argument("--memory", default="off", help="memory for swarm_llm: off | past")
    parser.add_argument("--literature", default="off", help="literature for swarm_llm: off | replay | live")
    parser.add_argument("--llm-config", default=None,
                        help="JSON file of LLM, memory and literature keys for swarm_llm runs only "
                             "(names starting " + ", ".join(LLM_CONFIG_PREFIXES) + ", or priors_path)")
    parser.add_argument("--planners", type=int, default=3)
    parser.add_argument("--k", type=int, default=4, help="candidates evaluated per round")
    parser.add_argument("--cascade", action="store_true", help="the fidelity ladder, for every condition")
    parser.add_argument("--outputs-dir", default="outputs")
    return parser.parse_args(argv)


def parse_conditions(text):
    """Split a comma list into known condition names, in the order given, without repeats."""
    names = []
    for piece in text.split(","):
        name = piece.strip()
        if name == "":
            continue
        if name not in CONDITIONS:
            raise CompareError("unknown condition '" + name + "'; choose from " + ", ".join(CONDITIONS))
        if name not in names:
            names.append(name)
    if len(names) == 0:
        raise CompareError("no conditions given")
    return names


def default_name(args):
    return args.backend + "-e" + str(int(args.eval_budget)) + "-" + str(int(time.time() * 1000))


def load_baselines():
    """Import wsp.swarm.baselines lazily. Raises CompareError when it or run_baseline is missing."""
    try:
        module = importlib.import_module("wsp.swarm.baselines")
    except ImportError as error:
        if error.name == "wsp.swarm.baselines":
            raise CompareError("the baseline conditions need wsp.swarm.baselines.run_baseline(config) "
                               "(wsp/swarm/baselines/), which is not built yet")
        raise CompareError("importing wsp.swarm.baselines failed: " + str(error))
    if not hasattr(module, "run_baseline"):
        raise CompareError("wsp.swarm.baselines has no run_baseline(config)")
    return module


def preflight(module, configs):
    """Run the baselines' own config check (when it has one) on every baseline run before any run starts."""
    check = getattr(module, "check_config", None)
    if check is None:
        return
    problems = []
    for config in configs:
        try:
            check(config)
        except (ValueError, NotImplementedError) as error:
            problems.append(config["run_id"] + ": " + str(error))
    if len(problems) > 0:
        raise CompareError("a baseline run cannot start, so nothing ran:\n  " + "\n  ".join(problems))


def llm_key_allowed(key):
    """True when --llm-config may set this key: an LLM, memory or literature key outside DEFAULTS."""
    if key in DEFAULTS or key in SHARED_SPACE_KEYS:
        return False
    if key in LLM_CONFIG_NAMES:
        return True
    for prefix in LLM_CONFIG_PREFIXES:
        if key.startswith(prefix):
            return True
    return False


def read_llm_settings(path):
    """The --llm-config JSON as a dict ({} without one). Every key must pass llm_key_allowed."""
    if path is None:
        return {}
    with open(path, encoding="utf-8") as handle:
        settings = json.load(handle)
    if not isinstance(settings, dict):
        raise CompareError("--llm-config must hold a JSON object")
    for key in settings:
        if key in DEFAULTS or key in SHARED_SPACE_KEYS:
            raise CompareError("--llm-config may not set '" + key + "': every condition shares it")
        if not llm_key_allowed(key):
            raise CompareError("--llm-config may set only LLM, memory and literature keys (names starting " +
                               ", ".join(LLM_CONFIG_PREFIXES) + ", or priors_path); '" + key + "' is not one")
    return settings


def check_llm(args, config):
    """Build the swarm_llm LLM and agents once before any run, so a bad LLM setup fails fast."""
    try:
        factory.build_llm(config)
        factory.build_agents(config)
    except Exception as error:
        raise CompareError("swarm_llm cannot start with --llm " + args.llm + ", so nothing ran: " +
                           type(error).__name__ + ": " + str(error))


def shared_overrides(args):
    """The config values every run shares: backend, both budgets, swarm size, fidelity ladder."""
    overrides = {
        "backend": args.backend,
        "budget": args.budget_kind,
        "eval_budget": args.eval_budget,
        "planners": args.planners,
        "k": args.k,
        "cascade": args.cascade,
        "outputs_dir": args.outputs_dir,
    }
    if args.compute_budget_s is not None:
        overrides["compute_budget_s"] = args.compute_budget_s
    if args.heldout_eval_budget is not None:
        overrides["heldout_eval_budget"] = args.heldout_eval_budget
    if args.heldout_compute_budget_s is not None:
        overrides["heldout_compute_budget_s"] = args.heldout_compute_budget_s
    return overrides


def run_id_for(name, condition, seed):
    return name + "-" + condition + "-s" + str(seed)


def run_config(args, name, condition, seed, llm_settings=None):
    """The config for one run. Only swarm_llm uses an LLM, memory, literature or --llm-config."""
    overrides = shared_overrides(args)
    if condition == "swarm_llm" and llm_settings is not None:
        for key in llm_settings:
            overrides[key] = llm_settings[key]
    overrides["seed"] = seed
    overrides["method"] = condition
    overrides["run_id"] = run_id_for(name, condition, seed)
    overrides["llm"] = "none"
    overrides["memory"] = "off"
    overrides["literature"] = "off"
    if condition == "swarm":
        overrides["agents"] = "scripted"
    elif condition == "swarm_llm":
        overrides["agents"] = "llm"
        overrides["llm"] = args.llm
        overrides["memory"] = args.memory
        overrides["literature"] = args.literature
    return make_config(overrides)


def run_folder(config):
    return os.path.join(config["outputs_dir"], "runs", config["run_id"])


def git_info():
    """The current commit and whether the work tree has changes. None values when git is missing."""
    info = {"git_commit": None, "git_dirty": None}
    try:
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True, timeout=20)
        if head.returncode == 0:
            info["git_commit"] = head.stdout.strip()
        status = subprocess.run(["git", "status", "--porcelain"], cwd=REPO_ROOT, capture_output=True, text=True,
                                timeout=20)
        if status.returncode == 0:
            info["git_dirty"] = status.stdout.strip() != ""
    except (OSError, subprocess.SubprocessError):
        pass
    return info


def read_json(path):
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def write_manifest(path, manifest):
    folder = os.path.dirname(path)
    os.makedirs(folder, exist_ok=True)
    temp = path + ".tmp"
    with open(temp, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
    os.replace(temp, path)
    return path


def check_free(compare_dir, plan, outputs_dir):
    """Refuse to reuse a compare name or a run folder: run logs are append-only."""
    if os.path.exists(os.path.join(compare_dir, "manifest.json")):
        raise CompareError("compare folder " + compare_dir + " already has a manifest; pick another --name")
    for item in plan:
        folder = os.path.join(outputs_dir, "runs", item["run_id"])
        if os.path.exists(folder):
            raise CompareError("run folder " + folder + " already exists; pick another --name")


def make_plan(name, conditions, seeds):
    """Seed-major list of (condition, seed, run_id), so partial results cover every condition."""
    plan = []
    for seed in seeds:
        for condition in conditions:
            plan.append({"condition": condition, "seed": seed, "run_id": run_id_for(name, condition, seed)})
    return plan


def run_one(condition, config, run_baseline):
    """Run one condition with one config. Returns its summary dict (empty when it returned none)."""
    if condition in SWARM_CONDITIONS:
        from wsp.swarm.swarm import run_swarm

        summary = run_swarm(config)
    else:
        summary = run_baseline(config)
    if not isinstance(summary, dict):
        return {}
    return summary


def run_record(condition, seed, config, summary):
    """The manifest entry for one finished run."""
    folder = run_folder(config)
    log_path = summary.get("log_path")
    if log_path is None:
        log_path = os.path.join(folder, "experiments.jsonl")
    run_manifest = read_json(os.path.join(folder, "manifest.json"))
    if run_manifest is None:
        run_manifest = {}
    priors_sha = run_manifest.get("priors_sha") or None
    best = summary.get("best_search")
    if isinstance(best, dict):
        best = dict(best)
        best["label"] = "SEARCH"
    return {
        "condition": condition,
        "seed": seed,
        "run_id": config["run_id"],
        "method": config["method"],
        "status": "done",
        "run_dir": os.path.abspath(folder),
        "log_path": os.path.abspath(log_path),
        "stub": summary.get("stub", run_manifest.get("stub")),
        "eval_used": summary.get("eval_used"),
        "compute_used_s": summary.get("compute_used_s"),
        "compute_overshoot_s": summary.get("compute_overshoot_s"),
        "best_search": best,
        "priors_sha": priors_sha,
        "run_budgets": run_budgets(run_manifest),
        "run_settings": run_settings(run_manifest),
    }


def run_budgets(run_manifest):
    """The four budget caps a run's own manifest says it used (None when unknown)."""
    config = run_manifest.get("config") or {}
    out = {}
    for key in ["eval_budget", "compute_budget_s", "heldout_eval_budget", "heldout_compute_budget_s"]:
        out[key] = config.get(key)
    return out


def shared_settings(config):
    """The SHARED_SETTINGS values of one config, as a new dict."""
    out = {}
    for key in SHARED_SETTINGS:
        out[key] = config.get(key)
    return out


def run_settings(run_manifest):
    """The SHARED_SETTINGS a run's own manifest says it used. A key it left out is left out here."""
    config = run_manifest.get("config") or {}
    out = {}
    for key in SHARED_SETTINGS:
        if key in config:
            out[key] = config[key]
    return out


def canonical(value):
    """A value as sorted JSON text, so dicts, lists and numbers compare the same after a JSON round trip."""
    return json.dumps(value, sort_keys=True)


def budget_check(budgets, records, settings=None):
    """Every run must report the comparison's four caps and shared settings. Returns {"equal", "mismatches"}."""
    if settings is None:
        settings = {}
    mismatches = []
    for record in records:
        for key in budgets:
            got = record["run_budgets"].get(key)
            if got is None:
                mismatches.append(record["run_id"] + ": " + key + " missing from its manifest")
            elif float(got) != float(budgets[key]):
                mismatches.append(record["run_id"] + ": " + key + " " + str(got) + " != " + str(budgets[key]))
        used = record.get("run_settings") or {}
        for key in settings:
            if key in used:
                got = used[key]
            elif key in OPTIONAL_SETTINGS:
                got = None
            else:
                mismatches.append(record["run_id"] + ": " + key + " missing from its manifest")
                continue
            if canonical(got) != canonical(settings[key]):
                mismatches.append(record["run_id"] + ": " + key + " " + canonical(got) + " != " +
                                  canonical(settings[key]))
    return {"equal": len(mismatches) == 0, "mismatches": mismatches}


def ledger_totals(path, run_ids):
    """Search-pool charges per run from the shared ledger: {run_id: {"eval": FEE, "compute_s": s}}."""
    totals = {}
    for run_id in run_ids:
        totals[run_id] = {"eval": 0.0, "compute_s": 0.0}
    if not os.path.exists(path):
        return totals
    for row in Ledger(path).read():
        run_id = row.get("run_id")
        if run_id not in totals or row.get("pool") != "search":
            continue
        if row.get("axis") == "eval":
            totals[run_id]["eval"] += float(row.get("amount") or 0.0)
        elif row.get("axis") == "compute":
            totals[run_id]["compute_s"] += float(row.get("amount") or 0.0)
    for run_id in totals:
        totals[run_id]["eval"] = round(totals[run_id]["eval"], 6)
    return totals


def distinct_priors(records):
    shas = []
    for record in records:
        sha = record.get("priors_sha")
        if sha and sha not in shas:
            shas.append(sha)
    return shas


def labels(stub):
    if stub:
        return "[SEARCH][STUB]"
    return "[SEARCH]"


def format_score(value):
    if value is None:
        return "none"
    return "%.4f" % value


def print_table(records, stub):
    print("condition     seed  picked best " + labels(stub) + "   evals used   compute used s")
    for record in records:
        best = record.get("best_search")
        value = None
        if isinstance(best, dict):
            value = best.get("combined")
        line = record["condition"].ljust(13) + " " + str(record["seed"]).rjust(4) + "  "
        line += format_score(value).rjust(10) + "                   "
        line += str(record.get("eval_used")).rjust(10) + "   "
        compute = record.get("compute_used_s")
        if compute is None:
            line += "none"
        else:
            line += ("%.0f" % compute).rjust(14)
        print(line)
    print("picked best: the program the judge's rule sends to held-out (ties within tie_margin go to fewer "
          "parameters), so it can sit below the best-so-far curve plot_curves.py draws")


def split_conditions(args, conditions):
    """(runnable, skipped): swarm_llm is skipped with a reason when no LLM is configured."""
    runnable = []
    skipped = {}
    for condition in conditions:
        if condition == "swarm_llm" and args.llm == "none":
            skipped[condition] = "no LLM configured (pass --llm modal_vllm, openai_compat, anthropic or mock)"
            print("skipping swarm_llm: " + skipped[condition])
            continue
        runnable.append(condition)
    if len(runnable) == 0:
        raise CompareError("nothing to run: every condition was skipped")
    return runnable, skipped


def prepare_baselines(args, name, plan):
    """Import run_baseline and check every baseline config before any run. None when no baseline is asked."""
    configs = []
    asked = []
    for item in plan:
        if item["condition"] in SWARM_CONDITIONS:
            continue
        configs.append(run_config(args, name, item["condition"], item["seed"]))
        if item["condition"] not in asked:
            asked.append(item["condition"])
    if len(configs) == 0:
        return None
    try:
        module = load_baselines()
    except CompareError as error:
        raise CompareError(str(error) + "; asked for: " + ", ".join(asked) +
                           " (run only the swarm with --conditions swarm)")
    preflight(module, configs)
    return module.run_baseline


def new_manifest(args, name, conditions, runnable, skipped, seeds, sample, llm_settings):
    """The compare manifest before any run: what will run, with which budgets, at which commit."""
    git = git_info()
    return {
        "name": name,
        "created": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "status": "running",
        "conditions": conditions,
        "ran": runnable,
        "skipped": skipped,
        "seeds": seeds,
        "budgets": {
            "eval_budget": float(sample["eval_budget"]),
            "compute_budget_s": float(sample["compute_budget_s"]),
            "heldout_eval_budget": float(sample["heldout_eval_budget"]),
            "heldout_compute_budget_s": float(sample["heldout_compute_budget_s"]),
        },
        "shared_settings": shared_settings(sample),
        "backend": args.backend,
        "budget_kind": args.budget_kind,
        "stub": args.backend == "stub",
        "cascade": args.cascade,
        "planners": args.planners,
        "k": args.k,
        "swarm_llm": {"llm": args.llm, "memory": args.memory, "literature": args.literature,
                      "llm_config": public_config(llm_settings)},
        "git_commit": git["git_commit"],
        "git_dirty": git["git_dirty"],
        "outputs_dir": os.path.abspath(args.outputs_dir),
        "ledger_path": os.path.abspath(ledger_path(sample)),
        "labels": "best scores are search-split numbers (SEARCH); STUB when the backend is the stub",
        "runs": [],
        "priors_sha": [],
    }


def finish_manifest(args, manifest, records, sample):
    """Add the ledger cross-check, the priors shas and the equal-budget check."""
    if args.budget_kind == "local":
        run_ids = []
        for record in records:
            run_ids.append(record["run_id"])
        totals = ledger_totals(ledger_path(sample), run_ids)
        for record in records:
            record["ledger_search"] = totals[record["run_id"]]
    manifest["runs"] = records
    manifest["priors_sha"] = distinct_priors(records)
    manifest["budget_check"] = budget_check(manifest["budgets"], records, manifest["shared_settings"])
    manifest["status"] = "done"


def print_report(manifest, manifest_path, compare_dir):
    print_table(manifest["runs"], manifest["stub"])
    if not manifest["budget_check"]["equal"]:
        print("WARNING: runs did not all use the same budgets and shared settings:")
        for line in manifest["budget_check"]["mismatches"]:
            print("  " + line)
    if len(manifest["priors_sha"]) > 1:
        print("WARNING: runs used different literature packs: " + ", ".join(manifest["priors_sha"]))
    print("manifest: " + manifest_path)
    print("charts: python scripts/plot_curves.py " + compare_dir)


def main(argv=None):
    """Run the comparison and return its manifest. Raises CompareError before any run when it cannot start."""
    args = parse_args(argv)
    names = load_dotenv()
    if len(names) > 0:
        print(".env: " + str(len(names)) + " keys loaded (values hidden)")
    conditions = parse_conditions(args.conditions)
    if args.seeds < 1:
        raise CompareError("--seeds must be at least 1")
    name = args.name
    if name is None:
        name = default_name(args)
    if not NAME_PATTERN.match(name):
        raise CompareError("--name may use only letters, digits, dot, dash and underscore")
    llm_settings = read_llm_settings(args.llm_config)

    runnable, skipped = split_conditions(args, conditions)
    seeds = list(range(args.seeds))
    compare_dir = os.path.join(args.outputs_dir, "compare", name)
    manifest_path = os.path.join(compare_dir, "manifest.json")
    plan = make_plan(name, runnable, seeds)
    check_free(compare_dir, plan, args.outputs_dir)
    run_baseline = prepare_baselines(args, name, plan)
    if "swarm_llm" in runnable:
        check_llm(args, run_config(args, name, "swarm_llm", 0, llm_settings))

    sample = make_config(shared_overrides(args))
    manifest = new_manifest(args, name, conditions, runnable, skipped, seeds, sample, llm_settings)
    write_manifest(manifest_path, manifest)
    print("compare " + name + ": " + str(len(plan)) + " runs (" + ", ".join(runnable) + ") x seeds " + str(seeds))

    records = []
    for item in plan:
        config = run_config(args, name, item["condition"], item["seed"], llm_settings)
        print("run " + config["run_id"])
        try:
            summary = run_one(item["condition"], config, run_baseline)
        except Exception as error:
            manifest["status"] = "failed"
            manifest["failed"] = {"run_id": config["run_id"], "error": type(error).__name__ + ": " + str(error)}
            write_manifest(manifest_path, manifest)
            raise
        records.append(run_record(item["condition"], item["seed"], config, summary))
        manifest["runs"] = records
        write_manifest(manifest_path, manifest)

    finish_manifest(args, manifest, records, sample)
    write_manifest(manifest_path, manifest)
    print_report(manifest, manifest_path, compare_dir)
    return manifest


def cli(argv=None):
    try:
        manifest = main(argv)
    except CompareError as error:
        print("compare: " + str(error), file=sys.stderr)
        return 2
    if not manifest["budget_check"]["equal"]:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(cli())
