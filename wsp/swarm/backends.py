"""Compile-and-eval backends behind one interface.

The env (env.py) runs every step in this order and charges the compute each
call reports: compile (gives fit), fit gate, heal, spend the eval token,
evaluate. A backend only does the work; it never touches a budget.

- StubCompileEval: the STUB world in stub.py.
- LocalCompileEval: Trisanth's compile on this machine plus Person B's eval.
- ModalCompileEval lives in wsp/swarm/modal_backend.py: the deployed wsp-swarm app (modal_swarm.py).
"""

import importlib
import json
import os
import time

from wsp.swarm import program_schema as ps
from wsp.swarm import space, stub
from wsp.swarm.config import fidelity_cost, guard_tasks


class CompileEval:
    """The backend interface. All methods must be safe to call from several threads.

    Results are plain dicts. A failure is reported with "ok": False and an
    "error" string; a method may also raise, and the env treats a raise in
    compile or heal as compile_error and a raise in evaluate as infra_error.
    "compute_s" maps a phase ("compile", "heal", "eval") to the seconds the
    call used. A backend may defer work: heal() may return the handle with no
    compute, and evaluate() may then report "compile" and "heal" seconds too.
    The env charges every phase it is given.
    """

    name = "base"

    def compile(self, program, fit_only=False):
        """Build the program's model WITHOUT its heal pass, and measure fit.

        fit_only=True is a probe: the env passes the program with heal set to
        None and only needs the fit, so the backend may skip saving weights.
        Returns:
            {"ok": bool, "error": str or None, "handle": anything heal and
             evaluate accept, "graft_report": [per-graft compile report
             entries, in graft order, each with at least "id", "host_layer",
             "fit" ("ridge" or "skipped"), "fit_r2" (float or None),
             "fit_pairs", "part_params", "glue_params", and when known
             "donor", "donor_layer", "control"], "compute_s": {"compile": s}}
        """
        raise NotImplementedError

    def heal(self, handle, program):
        """Run the heal block of `program` on a compiled handle.

        Only called when the program has a heal block. Heal with the steps
        in the program given here, which can differ from the compiled one:
        below full fidelity the env passes the program with its heal steps
        cut to config low_fidelity_heal_steps (the cascade's short heal), and
        a promotion to full compiles again and passes the program's own
        steps. A backend that defers the heal to evaluate() must keep the
        heal block it was given here.
        Returns {"ok", "error", "handle", "heal_loss": [floats], "compute_s": {"heal": s}}.
        """
        raise NotImplementedError

    def evaluate(self, handle, split, tasks, token, fidelity="full"):
        """Score the model on a split ("search" or "heldout") at a fidelity.

        token is the eval token the env already spent for this call; a
        backend passes it through (for example as a cache or log key) and
        never spends one itself. Retries must reuse the same token.
        Returns {"ok", "error", "scores": {task: float in [0, 1]}, "compute_s": {"eval": s}}.
        A backend that deferred its heal to this call also returns "heal_loss": [floats].
        """
        raise NotImplementedError

    def base_scores(self, split, tasks):
        """The host's scores on a split: {task: float}.

        A fixed reference measured once per split and shared by every arm,
        so it is not charged to any arm's budget.
        """
        raise NotImplementedError

    def release(self, handle):
        """Free whatever a handle holds. Optional."""
        return None


class StubCompileEval(CompileEval):
    """The STUB backend: deterministic, instant, with the stub cost model.

    A handle is {"program", "graft_report", "program_hash"}. evaluate()
    scores handle["program"], so the heal steps it sees are the ones heal()
    ran: heal() puts the program it was given (the short heal below full
    fidelity) into the handle. A handle that was never healed scores as if
    the program's own heal ran. The eval noise is keyed by the compiled
    program's hash, so the short and full heals of one program differ by
    the heal alone.
    """

    name = "stub"

    def __init__(self, config, catalog):
        self.config = config
        self.catalog = catalog
        self.world = stub.StubWorld(config["stub_seed"])
        self.n_host = space.host_layers(config, catalog)
        self.guards = guard_tasks(config)

    def default_depth(self, part_id):
        return space.default_at(self.catalog.get(part_id), self.config, self.n_host)

    def compile(self, program, fit_only=False):
        phase_time = stub.compile_cost(program)
        try:
            report = self.world.graft_report(program, self.n_host, self.default_depth)
        except KeyError as error:
            return {"ok": False, "error": "STUB: the stub world has no part " + str(error),
                    "handle": None, "graft_report": [], "compute_s": {"compile": phase_time}, "stub": True}
        handle = {"program": ps.normalize(program), "graft_report": report,
                  "program_hash": space.program_hash(program)}
        return {"ok": True, "error": None, "handle": handle, "graft_report": report,
                "compute_s": {"compile": phase_time}, "stub": True}

    def heal(self, handle, program):
        """Heal with the steps of the program given (the short heal below full fidelity), charging for them."""
        healed = {"program": ps.normalize(program), "graft_report": handle["graft_report"],
                  "program_hash": handle["program_hash"]}
        return {"ok": True, "error": None, "handle": healed,
                "heal_loss": self.world.heal_loss(program),
                "compute_s": {"heal": stub.heal_cost(program)}, "stub": True}

    def evaluate(self, handle, split, tasks, token, fidelity="full"):
        fraction = fidelity_cost(self.config, fidelity)
        scores = self.world.scores(handle["program"], handle["graft_report"], split, tasks, fidelity, fraction,
                                   handle["program_hash"], self.guards)
        return {"ok": True, "error": None, "scores": scores,
                "compute_s": {"eval": stub.eval_cost(fraction)}, "stub": True}

    def base_scores(self, split, tasks):
        return self.world.base_scores(split, tasks)


def load_function(spec):
    """Import "module:function" and return the function."""
    module_name, function_name = spec.split(":", 1)
    module = importlib.import_module(module_name)
    return getattr(module, function_name)


class LocalCompileEval(CompileEval):
    """Trisanth's compile on this machine, and Person B's eval harness.

    compile() builds the program with heal set to None (the compiler runs the
    ridge fit and writes compile_report.json). heal() compiles the program it
    is given, which runs the heal pass inside the compiler, so its time
    includes a second assembly. Its folder is named by that program's hash,
    so the cascade's short heal and the full heal of one program land in
    different folders. evaluate() calls config["local_eval"]
    ("module:function", called as fn(model_path, split, tasks)).
    """

    name = "local"

    def __init__(self, config, catalog):
        self.config = config
        self.catalog = catalog
        self.work_dir = os.path.join(config["outputs_dir"], "local_models")
        self.align_texts = None
        try:
            self.eval_fn = load_function(config["local_eval"])
        except (ImportError, AttributeError, ValueError) as error:
            raise RuntimeError("backend 'local' needs an eval harness at " + config["local_eval"] +
                               " (set config local_eval to 'module:function'): " + str(error))

    def get_align_texts(self, program):
        if self.align_texts is None:
            from wsp.data import task_texts

            data_name = ps.heal_data(program)
            if data_name is None:
                data_name = self.config["task_types"][self.config["task_type"]]["heal_data"][0]
            texts = task_texts(data_name, seed=0)
            self.align_texts = texts[:self.config["align_samples"]]
        return self.align_texts

    def build(self, program, folder):
        start = time.time()
        program_object = ps.to_program(program)
        align = None
        if len(ps.grafts(program)) > 0:
            align = self.get_align_texts(program)
        program_object.compile(folder, self.catalog, align_texts=align)
        seconds = time.time() - start
        with open(os.path.join(folder, "compile_report.json"), encoding="utf-8") as handle:
            report = json.load(handle)
        return report, seconds

    def compile(self, program, fit_only=False):
        bare = ps.without_heal(program)
        folder = os.path.join(self.work_dir, space.program_hash(bare))
        start = time.time()
        try:
            report, seconds = self.build(bare, folder)
        except Exception as error:
            return {"ok": False, "error": type(error).__name__ + ": " + str(error), "handle": None,
                    "graft_report": [], "compute_s": {"compile": time.time() - start}}
        return {"ok": True, "error": None, "handle": folder, "graft_report": report.get("grafts", []),
                "compute_s": {"compile": seconds}}

    def heal(self, handle, program):
        folder = os.path.join(self.work_dir, space.program_hash(program))
        start = time.time()
        try:
            report, seconds = self.build(program, folder)
        except Exception as error:
            return {"ok": False, "error": type(error).__name__ + ": " + str(error), "handle": None,
                    "heal_loss": [], "compute_s": {"heal": time.time() - start}}
        losses = []
        if report.get("heal") is not None:
            losses = report["heal"].get("losses", [])
        return {"ok": True, "error": None, "handle": folder, "heal_loss": losses, "compute_s": {"heal": seconds}}

    def evaluate(self, handle, split, tasks, token, fidelity="full"):
        start = time.time()
        try:
            scores = self.eval_fn(handle, split, tasks)
        except Exception as error:
            return {"ok": False, "error": type(error).__name__ + ": " + str(error), "scores": {},
                    "compute_s": {"eval": time.time() - start}}
        return {"ok": True, "error": None, "scores": scores, "compute_s": {"eval": time.time() - start}}

    def base_scores(self, split, tasks):
        table = self.config.get("base_scores")
        if table is None or split not in table:
            raise RuntimeError("backend 'local' needs config base_scores[" + split + "] (host scores)")
        out = {}
        for task in tasks:
            out[task] = table[split][task]
        return out
