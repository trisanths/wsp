"""The run manager: builds a swarm run from a spec and runs it in a daemon thread."""

import json
import os
import threading
import zlib
import time

from wsp.swarm import config as config_mod
from wsp.swarm import factory
from wsp.swarm import space

from studio.server import catalog as catalog_mod
from studio.server import envs
from studio.server import programs
from studio.server import settings
from studio.server import specs as specs_mod
from studio.server import tracking
from studio.server.runstate import RunState

_lock = threading.Lock()
_runs = {}


class RunError(ValueError):
    """A run could not be started. The message is safe to show."""


def make_run_config(spec, run_id=None, eval_budget=None, compute_budget_s=None, method="swarm"):
    """Build the swarm config for a spec."""
    overrides = specs_mod.run_overrides(spec, settings.backend(), settings.llm())
    if eval_budget is not None:
        overrides["eval_budget"] = eval_budget
    if compute_budget_s is not None:
        overrides["compute_budget_s"] = compute_budget_s
    overrides["method"] = method
    seed_text = str(spec["id"]) + str(time.time())
    overrides["seed"] = zlib.crc32(seed_text.encode("utf-8")) % 1000
    config = config_mod.make_config(overrides)
    if run_id is None:
        run_id = config_mod.make_run_id(method, config["seed"])
    config["run_id"] = run_id
    return config


def target_ids(spec):
    ids = []
    for item in spec["benchmarks"]:
        if item["kind"] == "target":
            ids.append(item["id"])
    return ids


def start_run(spec, program=None):
    """Start a run for a spec. Returns the run id. Raises RunError for a bad program."""
    config = make_run_config(spec)
    parts = envs.build_run(config)
    catalog = parts["catalog"]
    if program is not None:
        try:
            program = programs.normalize(program)
        except programs.ProgramError as error:
            raise RunError("the program is not valid: " + error.message)
        errors = space.validate(program, catalog, config)
        if len(errors) > 0:
            raise RunError("the program is not valid: " + "; ".join(errors))
    analyst, planners, judge = factory.build_agents(config)
    host_params = catalog_mod.host_params_b(config["host"])
    state = RunState(config["run_id"], spec, config, parts["backend"].name, host_params, len(planners),
                     target_ids(spec))
    state.env = parts["env"]
    state.catalog = catalog
    controller = tracking.Controller(state, settings.pace_s(), program)
    analyst, planners, judge = tracking.wrap_agents(analyst, planners, judge, controller)
    swarm = tracking.StudioSwarm(config, parts["env"], analyst, planners, judge, parts["memory"],
                                 parts["literature"], parts["llm"], controller)
    parts["env"].listeners.append(controller.on_row)
    run = Run(state, controller, swarm)
    with _lock:
        _runs[state.run_id] = run
    run.start()
    return state.run_id


def get_run(run_id):
    with _lock:
        return _runs.get(run_id)


class Run:
    def __init__(self, state, controller, swarm):
        self.state = state
        self.controller = controller
        self.swarm = swarm
        self.thread = None

    def start(self):
        self.thread = threading.Thread(target=self.work, name="studio-" + self.state.run_id, daemon=True)
        self.thread.start()

    def work(self):
        state = self.state
        with state.lock:
            if state.status == "starting":
                state.status = "running"
                state.touch()
        try:
            self.swarm.run()
            state.set_run_status("done")
        except Exception as error:
            message = type(error).__name__ + ": " + str(error)
            state.add_activity(None, "error", "bad", "The run stopped with an error: " + message)
            state.set_run_status("error", message)
        self.save()

    def save(self):
        """Write the final snapshot next to the run's log files."""
        try:
            folder = config_mod.run_dir(self.state.config)
            os.makedirs(folder, exist_ok=True)
            with open(os.path.join(folder, "studio_snapshot.json"), "w", encoding="utf-8") as handle:
                json.dump(self.state.snapshot(), handle, indent=2)
        except Exception:
            return

    def pause(self):
        state = self.state
        with state.lock:
            if state.status in ("starting", "running"):
                state.status = "paused"
                state.touch()
        self.controller.gate.pause()

    def resume(self):
        state = self.state
        self.controller.gate.resume()
        with state.lock:
            if state.status == "paused":
                state.status = "running"
                state.touch()

    def stop(self):
        state = self.state
        with state.lock:
            if state.status in ("starting", "running", "paused"):
                state.status = "stopping"
                state.touch()
        self.controller.gate.stop()

    def is_finished(self):
        return self.state.status in ("done", "error")
