"""The studio's Modal backend: ModalCompileEval runs each job on a GPU through the wsp-studio app.

Deploy the app first: MODAL_PROFILE=trisanth modal deploy studio/modal_studio.py

The server looks the functions up with modal.Function.from_name, so it does
not run inside Modal. A fit-only compile is a real GPU probe. A full compile
is deferred: it returns a handle that holds the program, and evaluate() does
compile, heal and eval in one container.
"""

import json
import os
import re
import threading

from wsp.swarm import factory
from wsp.swarm import program_schema as ps
from wsp.swarm import space
from wsp.swarm.backends import CompileEval
from wsp.swarm.config import fidelity_cost

from studio.server import settings
from studio.server import synth_report

APP_NAME = "wsp-studio"
BIG_DONOR_B = 10.0
BASE_FILE_LOCK = threading.Lock()


def lookup(name):
    """The deployed Modal function with this name. Tests replace this function."""
    import modal

    return modal.Function.from_name(APP_NAME, name)


_deploy_check = {"done": False, "warning": None}
_deploy_lock = threading.Lock()
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def deployed_warning():
    """A plain warning when the deployed Modal code differs from the local code, else None.

    The check runs once per server process and the answer is kept.
    """
    from studio import code_version

    with _deploy_lock:
        if _deploy_check["done"]:
            return _deploy_check["warning"]
        warning = None
        try:
            deployed = lookup("code_version").remote()
            if deployed != code_version.local_hash(REPO_ROOT):
                warning = "The deployed Modal code is older than the local code. Run modal deploy studio/modal_studio.py."
        except Exception as error:
            warning = ("The studio could not read the deployed Modal code version (" + type(error).__name__ +
                       "). Run modal deploy studio/modal_studio.py.")
        _deploy_check["done"] = True
        _deploy_check["warning"] = warning
        return warning


def register():
    """Point the factory's "modal" backend at this module. The wsp package is not changed."""
    factory.CHOICES["backend"]["modal"] = ("studio.server.modal_backend", "ModalCompileEval")


def model_size_b(model_id):
    """Parameters in billions read from a model id such as Qwen2.5-14B-Instruct, or 0 when unknown."""
    found = re.findall(r"(?<![\d.])(\d+(?:\.\d+)?)[bB](?![a-zA-Z])", str(model_id))
    if len(found) == 0:
        return 0.0
    return float(found[-1])


def donor_ids(program, catalog):
    """Every donor model of a program: the models of its grafted parts."""
    ids = []
    for graft in ps.grafts(program):
        record = catalog.records.get(ps.graft_id(graft))
        if record is not None and record.get("model") is not None:
            ids.append(record["model"])
    for item in program.get("inserts") or []:
        ids.append(item["donor"])
    return ids


def largest_donor_b(program, catalog):
    largest = 0.0
    for model_id in donor_ids(program, catalog):
        size = model_size_b(model_id)
        if size > largest:
            largest = size
    return largest


def gpu_suffix(program, catalog):
    """"_h100" when the largest donor is above about 10B parameters, else "" (an L40S)."""
    if largest_donor_b(program, catalog) > BIG_DONOR_B:
        return "_h100"
    return ""


def run_shrink(plan, split, tasks, fidelity, options):
    """Run one shrink on a GPU: the shrink_remote function (an H100 twin above about 10B parameters).

    plan holds the fields of the spec's shrink block. Returns the remote result dict. Raises RuntimeError
    with the remote error when the call fails or reports ok false.
    """
    name = "shrink_remote"
    if float(plan.get("source_params_b") or 0.0) > BIG_DONOR_B:
        name = "shrink_remote_h100"
    try:
        result = lookup(name).remote(plan, split, list(tasks), fidelity, options)
    except Exception as error:
        raise RuntimeError("the shrink call failed: " + type(error).__name__ + ": " + str(error))
    if not result.get("ok"):
        raise RuntimeError(str(result.get("error")) + (" " + str(result.get("trace") or "")[-600:]))
    return result


def resolve_indices(record):
    """A copy of a part record with its unit indices read from the local file, when the file is here.

    The container has parts_extracted but not the library, so the server reads the indices and sends them.
    A record whose file is not here goes as it is.
    """
    if record.get("indices") is not None or not isinstance(record.get("units"), str):
        return record
    path = record["units"]
    if not os.path.isabs(path):
        path = os.path.join(REPO_ROOT, path)
    if not os.path.exists(path):
        return record
    try:
        import torch

        indices = torch.load(path)["indices"]
        copy = dict(record)
        copy["indices"] = [int(value) for value in torch.as_tensor(indices).reshape(-1).tolist()]
        return copy
    except Exception:
        return record


def compute_phases(seconds):
    """The phases the env charges: compile (with data and probes), heal and eval. Totals are not charged twice."""
    seconds = dict(seconds)
    if "synthesize" not in seconds:
        return seconds
    out = {"compile": 0.0, "heal": float(seconds.get("heal") or 0.0), "eval": float(seconds.get("eval") or 0.0)}
    for key in ["prepare", "compile", "probe"]:
        out["compile"] += float(seconds.get(key) or 0.0)
    return out


def used_component_ids(program):
    ids = []
    for graft in ps.grafts(program):
        ids.append(ps.graft_id(graft))
    ids.extend(ps.mask_ids(program))
    ids.extend(ps.task_vector_ids(program))
    return ids


class ModalCompileEval(CompileEval):
    """Compile and eval on Modal GPUs. One remote call per probe, score or host base."""

    name = "modal"

    def __init__(self, config, catalog):
        self.config = config
        self.catalog = catalog
        self.score_reports = {}
        self.synth_results = {}
        self.lock = threading.Lock()

    # Remote calls.

    def options_for(self, program):
        """What the container needs besides the program: catalog records and the ridge fit texts."""
        records = []
        pipeline = self.pipeline_of(program)
        for part_id in used_component_ids(program):
            if part_id in self.catalog.records:
                record = self.catalog.records[part_id]
                if pipeline == "synth":
                    record = resolve_indices(record)
                records.append(record)
        setup = self.config["task_types"][self.config["task_type"]]
        data_name = ps.heal_data(program)
        if data_name is None:
            data_name = setup["heal_data"][0]
        limits = self.config.get("studio_limits") or {}
        return {"catalog_records": records, "align_data": data_name,
                "align_samples": self.config["align_samples"], "max_heal_min": self.config.get("max_heal_min"),
                "pipeline": pipeline, "pool": setup["heal_data"][0], "guards": list(setup.get("guards") or []),
                "max_total_params_b": limits.get("max_total_params_b"),
                "max_run_params_b": limits.get("max_run_params_b")}

    def pipeline_of(self, program):
        """"synth" or "classic" for this program (see synth_report.pipeline_for)."""
        return synth_report.pipeline_for(program)

    def function(self, base, program):
        return lookup(base + gpu_suffix(program, self.catalog))

    # Deferred report.

    def deferred_report(self, program):
        """Report entries for a compile that has not run: every fit is skipped."""
        entries = []
        for graft in ps.grafts(program):
            at = space.resolved_at(graft, self.catalog, self.config)
            entries.append({"id": ps.graft_id(graft), "host_layer": at, "fit": "skipped", "fit_r2": None,
                            "fit_pairs": 0, "part_params": None, "glue_params": None,
                            "control": ps.graft_control(graft)})
        return entries

    # CompileEval interface.

    def compile(self, program, fit_only=False):
        program = ps.normalize(program)
        if not fit_only:
            problems = synth_report.refusal_problems(program, self.catalog)
            if len(problems) > 0:
                return {"ok": False, "error": " ".join(problems), "handle": None, "graft_report": [],
                        "compute_s": {}}
            handle = {"program": program}
            return {"ok": True, "error": None, "handle": handle, "graft_report": self.deferred_report(program),
                    "compute_s": {"compile": 0.0}}
        try:
            result = self.function("probe_remote", program).remote(ps.without_heal(program),
                                                                    self.options_for(program))
        except Exception as error:
            return {"ok": False, "error": type(error).__name__ + ": " + str(error), "handle": None,
                    "graft_report": [], "compute_s": {}}
        seconds = result.get("seconds") or {}
        if not result.get("ok"):
            return {"ok": False, "error": result.get("error"), "handle": None, "graft_report": [],
                    "compute_s": dict(seconds)}
        return {"ok": True, "error": None, "handle": {"program": program},
                "graft_report": list(result["graft_report"]), "compute_s": dict(seconds),
                "gpu": result.get("gpu")}

    def heal(self, handle, program):
        """The heal pass runs inside evaluate(), so this reports no compute.

        The new handle holds the program to heal and score, which may have fewer heal steps than the
        original at low fidelity. "origin" keeps the original so report_for still finds its report.
        """
        new_handle = {"program": ps.normalize(program), "origin": handle["program"]}
        return {"ok": True, "error": None, "handle": new_handle, "heal_loss": [], "compute_s": {"heal": 0.0}}

    def evaluate(self, handle, split, tasks, token, fidelity="full"):
        program = handle["program"]
        fraction = fidelity_cost(self.config, fidelity)
        options = self.options_for(program)
        key = space.program_hash(handle.get("origin") or program)
        try:
            if options.get("pipeline") == "synth":
                options["token"] = token
                result = self.function("synth_remote", program).remote(program, split, list(tasks), fraction,
                                                                       options)
            else:
                result = self.function("score_remote", program).remote(program, split, list(tasks), token,
                                                                       fraction, options)
        except Exception as error:
            return {"ok": False, "error": type(error).__name__ + ": " + str(error), "scores": {},
                    "compute_s": {}}
        seconds = compute_phases(result.get("seconds") or {})
        if not result.get("ok"):
            return {"ok": False, "error": result.get("error"), "scores": {}, "compute_s": seconds}
        with self.lock:
            self.score_reports[key] = result
            if options.get("pipeline") == "synth":
                self.synth_results[key] = result
            else:
                self.synth_results.pop(key, None)
        return {"ok": True, "error": None, "scores": result["scores"], "compute_s": seconds,
                "heal_loss": result.get("heal_loss"), "gpu": result.get("gpu")}

    def synth_for(self, program):
        """The synth part of the last score of this program (components, stages, drift, sizes), or None.

        None means the program was scored by the classic pipeline, or has not been scored.
        """
        with self.lock:
            raw = self.synth_results.get(space.program_hash(ps.normalize(program)))
        if raw is None:
            return None
        return synth_report.public(raw, self.catalog)

    def report_for(self, program):
        """The graft report of the last score of this program (with real fit), or None."""
        with self.lock:
            result = self.score_reports.get(space.program_hash(ps.normalize(program)))
        if result is None:
            return None
        return list(result.get("graft_report") or [])

    # Host base scores.

    def base_file(self):
        return os.path.join(settings.out_dir(), "base_scores_modal.json")

    def read_base_file(self):
        path = self.base_file()
        if not os.path.exists(path):
            return {}
        with open(path, encoding="utf-8") as handle:
            return json.load(handle)

    def write_base_file(self, table):
        path = self.base_file()
        temp_path = path + "." + str(os.getpid()) + ".tmp"
        with open(temp_path, "w", encoding="utf-8") as handle:
            json.dump(table, handle, indent=2, sort_keys=True)
        os.replace(temp_path, path)

    def base_scores(self, split, tasks):
        """The host's scores: config base_scores first, then the cache file, then one remote run for the rest."""
        out = {}
        table = self.config.get("base_scores") or {}
        configured = table.get(split) or {}
        missing = []
        for task in tasks:
            if task in configured:
                out[task] = configured[task]
            else:
                missing.append(task)
        if len(missing) == 0:
            return out
        host = self.config["host"]
        with BASE_FILE_LOCK:
            cache = self.read_base_file()
            cached = ((cache.get(host) or {}).get(split)) or {}
            still_missing = []
            for task in missing:
                if task in cached:
                    out[task] = cached[task]
                else:
                    still_missing.append(task)
            if len(still_missing) > 0:
                result = lookup("base_remote").remote(host, split, still_missing)
                if not result.get("ok"):
                    raise RuntimeError("host base scores failed: " + str(result.get("error")))
                if host not in cache:
                    cache[host] = {}
                if split not in cache[host]:
                    cache[host][split] = {}
                for task in still_missing:
                    out[task] = result["scores"][task]
                    cache[host][split][task] = result["scores"][task]
                self.write_base_file(cache)
        return out
