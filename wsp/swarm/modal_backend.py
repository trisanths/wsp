"""ModalCompileEval: the env's backend on the deployed wsp-swarm app (modal_swarm.py).

The env (env.py) calls compile, then the fit gate, then heal, then spends the
eval token, then evaluate. This backend maps those calls onto the three
deployed GPU functions in modal_swarm.py:

- compile(program) runs probe_candidate: Trisanth's compile with heal set to
  None, which gives the per-graft report (fit_r2, part and glue params, host
  layer). The env's fit gate therefore sees a real fit before any token is
  spent. The probe is cached per run in the wsp-probe Dict, so a cascade
  promotion or a re-step of the same program reads the cache from this
  machine and starts no container. The cost of this design: a program that
  was not probed first pays one fit-only compile here and a second compile
  inside run_candidate. Every arm pays it the same way.
- heal(handle, program) is deferred. It keeps the heal block it is given
  (the cascade's short heal below full fidelity) and charges nothing.
- evaluate(handle, split, tasks, token, fidelity) runs run_candidate:
  compile, heal and grade in one container, with the token the env already
  spent. It reports compile, heal and eval seconds, and the env charges all
  three, plus the heal losses (the env reads them from evaluate because the
  heal ran here). Fidelity becomes a question count: the first
  round(fraction x split size) items of the split, so the levels are nested
  (check 32, low 68 and full 200 on the search split).
- base_scores(split, tasks): with the interim scorer, host_baseline once per
  split; with the harness, the config base_scores table. Either way the host
  is graded by the same scorer as every candidate, and the reference is
  shared, charged to no arm.

Scores. Config modal_scorer picks the scorer inside run_candidate, the same
for every arm:
- "interim" (default): wsp/swarm/interim_eval.py, gsm8k only, label
  INTERIM. Use task_type "math_interim" (no guard). Every fidelity works.
- "harness": Arjun's wsp_eval.harness.evaluate on the hf backend, gsm8k and
  if_guard, label HARNESS. task_type "math" works. The harness grades whole
  splits only, so the cascade is refused at construction and any fidelity
  below full at evaluate. base_scores come from config base_scores, which the
  same harness measured on Modal (modal_eval.py::base_scores, hf backend).
The backend refuses a task its scorer cannot grade.

Budget. There is no ModalBudget. The search loop runs in one process on this
machine, so a run uses budget "local": that process holds the eval and
compute counters and writes outputs/ledger.jsonl. The containers also write
their own GPU-seconds rows to the wsp-ledger Dict (one write-once row per
phase), which is the audit trail on the Modal side; settle() writes any row a
container could not. A Queue-backed budget matters only for a driver that
runs the whole search inside Modal, and none exists.

Two labels differ between the two ledgers, and neither changes a total:
compile() always charges the Modal pool "search" (the env does not tell a
backend which split a compile serves, and held-out compiles of probed
programs are cache hits), and the container names the probe phase "probe"
where the env names it "compile".

Before a run:
- MODAL_TOKEN_ID and MODAL_TOKEN_SECRET are in the environment (config.load_dotenv reads .env).
- The deployed wsp-swarm app matches modal_swarm.py. The image bakes in the
  catalog files at deploy time, so redeploy after data/pilot_catalog.jsonl changes.
- The run's catalog_path is one of the files the image carries
  (modal_swarm.CATALOG_FILES): data/pilot_catalog.jsonl ("pilot") or catalog.jsonl ("main").
"""

import importlib
import os
import sys
import threading

from wsp.data import SEARCH_SIZE
from wsp.swarm import program_schema as ps
from wsp.swarm.backends import CompileEval
from wsp.swarm.config import REPO_ROOT, all_tasks, fidelity_cost

HELDOUT_SIZE = 1319          # GSM8K test, every item of which modal_swarm.split_items scores for "heldout"
SPLIT_SIZES = {"search": SEARCH_SIZE, "heldout": HELDOUT_SIZE}
PHASES = ["compile", "heal", "eval"]
WHO = "env"                  # the env does not pass the agent to a backend


def load_modal_swarm():
    """Import modal_swarm.py from the repo root (it is a top-level module, not part of the wsp package)."""
    if REPO_ROOT not in sys.path:
        sys.path.insert(0, REPO_ROOT)
    return importlib.import_module("modal_swarm")


def catalog_path_or_default(path):
    """factory.build_catalog loads catalog.jsonl when a non-stub run sets no catalog_path."""
    if path is None:
        return os.path.join(REPO_ROOT, "catalog.jsonl")
    return path


def catalog_name_for(path, catalog_files):
    """The name ("pilot" or "main") of the image's catalog file at this path, or None."""
    target = os.path.normcase(os.path.abspath(path))
    for name in sorted(catalog_files):
        repo_path = catalog_files[name][0]
        candidate = os.path.normcase(os.path.abspath(os.path.join(REPO_ROOT, *repo_path.split("/"))))
        if candidate == target:
            return name
    return None


def other_tasks(tasks, allows):
    """The tasks a scorer cannot grade. allows(task) says whether it can (modal_swarm.scorer_allows)."""
    others = []
    for task in tasks:
        if not allows(task):
            others.append(task)
    return others


def error_text(result, function_name):
    """One line naming the function, the phase that failed and the error."""
    text = function_name + " failed"
    if result.get("phase"):
        text += " in phase " + str(result["phase"])
    return text + ": " + str(result.get("error"))


class ModalCompileEval(CompileEval):
    """The env's backend on the deployed wsp-swarm app. See the module docstring."""

    name = "modal"

    def __init__(self, config, catalog):
        self.config = config
        self.catalog = catalog
        self.ms = load_modal_swarm()

        run_id = config.get("run_id")
        if not run_id:
            raise ValueError("backend 'modal' needs config run_id: every probe, run and Modal compute row is "
                             "scoped to it (factory.build_run fills it)")
        self.run_id = run_id
        self.method = config.get("method") or ""
        self.align_samples = int(config["align_samples"])

        self.scorer = config.get("modal_scorer") or "interim"
        if self.scorer not in self.ms.SCORERS:
            raise ValueError("config modal_scorer must be one of " + ", ".join(self.ms.SCORERS) + "; got " +
                             repr(self.scorer))
        self.label = self.ms.SCORER_LABELS[self.scorer]
        self.allowed_tasks = list(self.ms.SCORER_TASKS[self.scorer])
        scorer = self.scorer
        self.allows = lambda task: self.ms.scorer_allows(scorer, task)
        tasks = all_tasks(config)
        unsupported = other_tasks(tasks, self.allows)
        if len(unsupported) > 0:
            hint = "Use task_type 'math_interim' with the interim scorer, or modal_scorer 'harness'."
            if self.scorer == "harness":
                hint = "The harness scorer grades " + ", ".join(self.allowed_tasks) + "."
            raise ValueError("backend 'modal' with modal_scorer '" + self.scorer + "' grades " +
                             ", ".join(self.allowed_tasks) + " only, but task_type '" + str(config["task_type"]) +
                             "' also asks for " + ", ".join(unsupported) + ". " + hint)
        if self.scorer == "harness":
            if config.get("cascade"):
                raise ValueError("modal_scorer 'harness' grades whole splits only, so the cascade (partial "
                                 "question sets) cannot run with it; turn cascade off or use the interim scorer")

        path = catalog_path_or_default(config.get("catalog_path"))
        self.catalog_name = catalog_name_for(path, self.ms.CATALOG_FILES)
        if self.catalog_name is None:
            carried = []
            for name in sorted(self.ms.CATALOG_FILES):
                carried.append(self.ms.CATALOG_FILES[name][0])
            raise ValueError("backend 'modal' runs only a catalog the Modal image carries (" +
                             ", ".join(carried) + "); got " + str(path))

        self.functions = {}
        self.functions_lock = threading.Lock()
        self.base = {}
        self.base_lock = threading.Lock()

    def function(self, name):
        """A handle to one deployed function, looked up once."""
        with self.functions_lock:
            if name not in self.functions:
                self.functions[name] = self.ms.deployed(name)
            return self.functions[name]

    def settle(self, result):
        """Write the Modal ledger rows a container could not. The env's ledger charges the seconds either way."""
        if not result.get("uncharged"):
            return
        try:
            self.ms.charge_uncharged(result)
        except self.ms.BudgetError as error:
            self.ms.warn("modal_backend: an uncharged phase is still missing from wsp-ledger (" + str(error) + ")")

    def eval_limit_for(self, split, fidelity):
        """The question count for a fidelity: None (every item) at full, else the first round(fraction x size)."""
        fraction = fidelity_cost(self.config, fidelity)
        if fraction >= 1.0:
            return None
        count = int(round(fraction * SPLIT_SIZES[split]))
        if count < 1:
            count = 1
        return count

    def compile(self, program, fit_only=False):
        """The fit-only compile on Modal (probe_candidate), read from the run's probe cache when it is there."""
        program = ps.normalize(program)
        bare = ps.without_heal(program)
        cached = self.ms.lookup_probe(bare, self.catalog_name, self.run_id, self.align_samples)
        if cached is not None:
            result = self.ms.as_cache_hit(cached)
        else:
            try:
                result = self.function("probe_candidate").remote(
                    bare, self.catalog_name, align_samples=self.align_samples, run_id=self.run_id,
                    pool="search", who=WHO, method=self.method)
            except Exception as error:
                return {"ok": False, "error": "probe_candidate failed: " + type(error).__name__ + ": " + str(error),
                        "handle": None, "graft_report": [], "compute_s": {}}
            self.settle(result)
        seconds = result.get("seconds") or {}
        compute = {"compile": float(seconds.get("compile", 0.0))}
        if not result.get("ok"):
            return {"ok": False, "error": error_text(result, "probe_candidate"), "handle": None,
                    "graft_report": [], "compute_s": compute}
        return {"ok": True, "error": None, "handle": {"program": program},
                "graft_report": list(result.get("graft_report") or []), "compute_s": compute}

    def heal(self, handle, program):
        """Deferred: keep the heal block given here; run_candidate heals inside evaluate."""
        return {"ok": True, "error": None, "handle": {"program": ps.normalize(program)}, "heal_loss": [],
                "compute_s": {}}

    def evaluate(self, handle, split, tasks, token, fidelity="full"):
        """Compile, heal and grade on Modal (run_candidate) with the token the env spent."""
        tasks = list(tasks)
        unsupported = other_tasks(tasks, self.allows)
        if len(unsupported) > 0:
            return {"ok": False, "error": "backend 'modal' with modal_scorer '" + self.scorer + "' grades " +
                    ", ".join(self.allowed_tasks) + " only; asked for " + ", ".join(unsupported),
                    "scores": {}, "compute_s": {}}
        if split not in SPLIT_SIZES:
            return {"ok": False, "error": "unknown split " + repr(split), "scores": {}, "compute_s": {}}
        eval_limit = self.eval_limit_for(split, fidelity)
        if self.scorer == "harness" and eval_limit is not None:
            return {"ok": False, "error": "modal_scorer 'harness' grades whole splits only; fidelity " +
                    repr(fidelity) + " asks for " + str(eval_limit) + " items", "scores": {}, "compute_s": {}}
        program = handle["program"]
        cached = self.ms.lookup_result(program, split, self.catalog_name, self.run_id, eval_limit,
                                       self.align_samples, self.scorer, tasks)
        if cached is not None:
            result = self.ms.as_cache_hit(cached)
            result["token"] = token
        else:
            try:
                result = self.function("run_candidate").remote(
                    program, split, token, self.catalog_name, eval_limit=eval_limit,
                    align_samples=self.align_samples, run_id=self.run_id, pool=split, who=WHO, method=self.method,
                    tasks=tasks, scorer=self.scorer)
            except Exception as error:
                return {"ok": False, "error": "run_candidate failed: " + type(error).__name__ + ": " + str(error),
                        "scores": {}, "compute_s": {}}
            self.settle(result)
        seconds = result.get("seconds") or {}
        compute = {}
        for phase in PHASES:
            compute[phase] = float(seconds.get(phase, 0.0))
        out = {"ok": bool(result.get("ok")), "error": None, "scores": dict(result.get("scores") or {}),
               "compute_s": compute, "heal_loss": list(result.get("heal_loss") or []),
               "label": result.get("label", self.label), "eval_backend": result.get("eval_backend"),
               "n_items": result.get("n_items"), "cache_hit": bool(result.get("cache_hit"))}
        if not out["ok"]:
            out["error"] = error_text(result, "run_candidate")
            out["scores"] = {}
        return out

    def base_scores(self, split, tasks):
        """The plain host, graded by this run's scorer.

        Interim: host_baseline once per split. Harness: the config base_scores
        table for the tasks it has (gsm8k and if_guard, measured by the same
        harness with modal_eval.py::base_scores), and host_baseline in harness
        mode for any other task, such as a researched benchmark. Both are
        shared references, cached on Modal and charged to no run.
        """
        if self.scorer == "harness":
            return self.harness_base(split, list(tasks))
        with self.base_lock:
            if split not in self.base:
                result = self.function("host_baseline").remote(split)
                if not result.get("ok"):
                    raise RuntimeError(error_text(result, "host_baseline") + " (split " + split + ")")
                self.base[split] = dict(result["scores"])
            table = self.base[split]
        out = {}
        for task in tasks:
            if task not in table:
                raise RuntimeError("host_baseline has no score for " + task + " on " + split)
            out[task] = table[task]
        return out

    def harness_base(self, split, tasks):
        """Harness host scores: the config table where it has them, else host_baseline in harness mode."""
        table = (self.config.get("base_scores") or {}).get(split) or {}
        out = {}
        missing = []
        for task in tasks:
            if task in table:
                out[task] = float(table[task])
            else:
                missing.append(task)
        if len(missing) == 0:
            return out
        key = split + "|" + "+".join(missing)
        with self.base_lock:
            if key not in self.base:
                result = self.function("host_baseline").remote(split, tasks=missing, scorer="harness")
                if not result.get("ok"):
                    raise RuntimeError(error_text(result, "host_baseline") + " (split " + split + ", harness)")
                self.base[key] = dict(result["scores"])
            measured = self.base[key]
        for task in missing:
            if task not in measured:
                raise RuntimeError("host_baseline has no harness score for " + task + " on " + split)
            out[task] = float(measured[task])
        return out
