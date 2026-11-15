"""The studio's Modal backend: ModalCompileEval runs each job on a GPU through the wsp-studio app.

Deploy the app first: MODAL_PROFILE=trisanth modal deploy studio/modal_studio.py

The server looks the functions up with modal.Function.from_name, so it does
not run inside Modal. A fit-only compile is a real GPU probe. A full compile
is deferred: it returns a handle that holds the program, and evaluate() does
compile, tap training, heal and eval in one container.
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

APP_NAME = "wsp-studio"
BIG_DONOR_B = 10.0
BASE_FILE_LOCK = threading.Lock()


def lookup(name):
    """The deployed Modal function with this name. Tests replace this function."""
    import modal

    return modal.Function.from_name(APP_NAME, name)


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
    """Every donor model of a program: tap donors and the models of its grafted parts."""
    ids = []
    for tap in program.get("taps") or []:
        ids.append(tap["donor"])
    for graft in ps.grafts(program):
        record = catalog.records.get(ps.graft_id(graft))
        if record is not None and record.get("model") is not None:
            ids.append(record["model"])
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
        self.lock = threading.Lock()

    # Remote calls.

    def options_for(self, program):
        """What the container needs besides the program: catalog records and the ridge fit texts."""
        records = []
        for part_id in used_component_ids(program):
            if part_id in self.catalog.records:
                records.append(self.catalog.records[part_id])
        data_name = ps.heal_data(program)
        if data_name is None:
            data_name = self.config["task_types"][self.config["task_type"]]["heal_data"][0]
        return {"catalog_records": records, "align_data": data_name,
                "align_samples": self.config["align_samples"]}

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
        for tap in program.get("taps") or []:
            entries.append({"id": "tap:" + str(tap["donor"]), "host_layer": tap["at"], "fit": "skipped",
                            "fit_r2": None, "fit_pairs": 0, "part_params": None, "glue_params": None,
                            "donor": tap["donor"], "donor_layer": tap["donor_layer"], "control": None})
        return entries

    # CompileEval interface.

    def compile(self, program, fit_only=False):
        program = ps.normalize(program)
        if not fit_only:
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
        """The heal pass runs inside evaluate(), so this reports no compute."""
        return {"ok": True, "error": None, "handle": handle, "heal_loss": [], "compute_s": {"heal": 0.0}}

    def evaluate(self, handle, split, tasks, token, fidelity="full"):
        program = handle["program"]
        fraction = fidelity_cost(self.config, fidelity)
        try:
            result = self.function("score_remote", program).remote(program, split, list(tasks), token, fraction,
                                                                    self.options_for(program))
        except Exception as error:
            return {"ok": False, "error": type(error).__name__ + ": " + str(error), "scores": {},
                    "compute_s": {}}
        seconds = dict(result.get("seconds") or {})
        if not result.get("ok"):
            return {"ok": False, "error": result.get("error"), "scores": {}, "compute_s": seconds}
        with self.lock:
            self.score_reports[space.program_hash(program)] = result
        return {"ok": True, "error": None, "scores": result["scores"], "compute_s": seconds,
                "heal_loss": result.get("heal_loss"), "gpu": result.get("gpu")}

    def report_for(self, program):
        """The graft and tap report of the last score of this program (with real fit), or None."""
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
