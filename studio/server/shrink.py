"""Shrink sessions: remove skills from a source model and compare the result with honest references.

A shrink is the second request type of the studio. A person asks for a smaller model that keeps one
skill, and wsp/slim.py removes the experts (MoE), or the layers, neurons and heads (dense), that the skill
does not use, then heals with distillation toward the original. The result is one network.

A session holds the plan of one spec. The plan can be edited (free, no GPU) and then run as a job. Jobs
are read at GET /api/jobs/{id}, the same route the builder uses.

Backends:
  stub   a made-up result that falls smoothly as more is removed. It is labeled stub.
  modal  the shrink_remote function of the deployed wsp-studio app.
"""

import concurrent.futures
import hashlib
import math
import threading
import time
import uuid

from studio.server import builder as builder_mod
from studio.server import compare
from studio.server import library
from studio.server import modal_backend
from studio.server import settings
from studio.server import shrink_models
from studio.server import shrink_spec
from studio.server import specs as specs_mod
from studio.server import sizing

FIDELITIES = ["check", "low", "full"]
DEFAULT_FIDELITY = "low"
SCORE_SOURCE_BELOW_B = 10.0
PLAN_KEYS = ["source_model", "source_params_b", "source_active_params_b", "family", "keep_skill", "method",
             "target_params_b", "keep_experts", "rule", "heal_steps"]

_lock = threading.Lock()
_sessions = {}
_pool = concurrent.futures.ThreadPoolExecutor(max_workers=2, thread_name_prefix="studio-shrink")


class ShrinkError(ValueError):
    """A shrink request could not be done. The message is safe to show."""


class Session:
    def __init__(self, shrink_id, spec, plan):
        self.id = shrink_id
        self.spec = spec
        self.plan = plan
        self.lock = threading.RLock()


def create_session(spec):
    """Start a shrink session for a shrink spec."""
    if spec.get("request_type") != "shrink" or not isinstance(spec.get("shrink"), dict):
        raise ShrinkError("this spec asks to build a model, not to shrink one")
    session = Session("sh_" + uuid.uuid4().hex[:8], spec, dict(spec["shrink"]))
    with _lock:
        _sessions[session.id] = session
    return session


def get_session(shrink_id):
    with _lock:
        return _sessions.get(shrink_id)


def current_spec(session):
    """The stored spec for the session, so edits made with PUT /api/spec count. Falls back to the session copy."""
    stored = specs_mod.get_spec(session.spec["id"])
    if stored is None:
        return session.spec
    return stored


def target_ids(spec):
    ids = []
    for item in spec["benchmarks"]:
        if item["kind"] == "target":
            ids.append(item["id"])
    return ids


# The plan.

def plan_view(session):
    """The plan with its size estimate, the limits, the problems and a summary. Free: no GPU, no model load."""
    with session.lock:
        plan = dict(session.plan)
    spec = current_spec(session)
    estimate, problems = sizing.shrink_check(plan, spec.get("limits"))
    view = dict(plan)
    view["estimate"] = estimate
    view["limits"] = spec.get("limits")
    view["problems"] = problems
    view["ok"] = len(problems) == 0
    view["summary"] = shrink_spec.plan_summary(plan, estimate)
    view["tasks"] = target_ids(spec)
    view["stub"] = settings.is_stub()
    return view


def update_plan(session, body):
    """Apply edited plan fields and return the re-estimated plan. Raises ShrinkError for a bad field."""
    fields = {}
    for key in PLAN_KEYS:
        if key in body:
            fields[key] = body[key]
    with session.lock:
        try:
            session.plan = shrink_spec.check_block(fields, session.plan)
        except shrink_spec.ShrinkSpecError as error:
            raise ShrinkError(str(error))
    return plan_view(session)


# Running.

def run_options(session, body):
    """The options of a run, from the request body with defaults."""
    fidelity = body.get("fidelity", DEFAULT_FIDELITY)
    if fidelity not in FIDELITIES:
        raise ShrinkError("fidelity must be one of " + ", ".join(FIDELITIES))
    tasks = body.get("tasks")
    if tasks is None:
        tasks = target_ids(current_spec(session))
    if not isinstance(tasks, list) or len(tasks) == 0 or not all(isinstance(task, str) for task in tasks):
        raise ShrinkError("tasks must be a list of benchmark ids")
    control = body.get("control", True)
    general = body.get("general_control", False)
    if not isinstance(control, bool) or not isinstance(general, bool):
        raise ShrinkError("control and general_control must be true or false")
    score_source = body.get("score_source")
    if score_source is None:
        score_source = session.plan["source_params_b"] <= SCORE_SOURCE_BELOW_B
    return {"fidelity": fidelity, "tasks": list(tasks), "control": control, "general_control": general,
            "score_source": bool(score_source)}


def start_run(session, body):
    """Check the plan against the limits, then queue the job. Returns the job id."""
    backend = settings.backend()
    if backend not in ("stub", "modal"):
        raise ShrinkError("the " + backend + " backend does not run shrinks. Use stub or modal.")
    options = run_options(session, body)
    with session.lock:
        plan = dict(session.plan)
    spec = current_spec(session)
    arms = 1
    if options["control"]:
        arms += 1
    if options["general_control"]:
        arms += 1
    estimate, problems = sizing.shrink_check(plan, spec.get("limits"), arms)
    if len(problems) > 0:
        raise ShrinkError(" ".join(problems))
    job = {"job_id": "j_" + uuid.uuid4().hex[:8], "action": "shrink", "status": "queued",
           "stub": backend == "stub", "result": None, "budget": None, "error": None, "lock": threading.Lock()}
    builder_mod.add_job(job)
    _pool.submit(work_job, spec, job, plan, options, backend)
    return job["job_id"]


def work_job(spec, job, plan, options, backend):
    with job["lock"]:
        job["status"] = "running"
    try:
        if backend == "stub":
            result = stub_result(spec, plan, options)
        else:
            result = modal_result(spec, plan, options)
        with job["lock"]:
            job["result"] = result
            job["status"] = "done"
    except Exception as error:
        with job["lock"]:
            job["error"] = type(error).__name__ + ": " + str(error)
            job["status"] = "error"


def finish(spec, plan, scores, params_b, active_b, removed, source_scores, control, general, heal, log, seconds,
           stub, extra=None):
    """The result dict of the contract, with the comparison."""
    models = library.get()["models"]
    comparison = compare.build_shrink(spec, scores, params_b, source_scores, control, general, stub, models)
    problems = []
    if not stub:
        problems = sizing.measured_problems(spec.get("limits"), params_b, active_b)
    if len(problems) > 0:
        comparison["verdict"] = "failed"
        comparison["headline"] = "Failed attempt. " + " ".join(problems)
    result = {"params_b": params_b, "active_params_b": active_b, "removed": removed, "scores": scores,
              "source_scores": source_scores, "control": None, "general": None, "comparison": comparison,
              "stub": stub, "heal": heal, "log": log, "seconds": seconds, "plan": dict(plan)}
    if control is not None:
        result["control"] = {"scores": control}
    if general is not None:
        result["general"] = {"scores": general}
    if len(problems) > 0:
        result["limit_problems"] = problems
    if extra:
        result.update(extra)
    return result


# Stub backend.

def jitter(plan, task, name):
    """A small repeatable offset in -0.005 to 0.005 from the plan and the task."""
    text = repr(sorted((key, str(plan.get(key))) for key in PLAN_KEYS)) + task + name
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return (digest[0] / 255.0 - 0.5) * 0.01


def clip01(value):
    return round(min(max(value, 0.0), 1.0), 4)


def stub_scores(spec, plan, estimate, tasks):
    """(source, skill, random, general) scores per task. They fall smoothly as more is removed."""
    removed = 1.0 - estimate["total_params_b"] / estimate["source_params_b"]
    removed = min(max(removed, 0.0), 1.0)
    heal = 0.02 * (1.0 - math.exp(-plan["heal_steps"] / 300.0))
    bases = {}
    for item in spec["benchmarks"]:
        if item.get("base") is not None:
            bases[item["id"]] = float(item["base"])
    source = {}
    skill = {}
    random_ = {}
    general = {}
    for task in tasks:
        top = min(0.95, bases.get(task, 0.4) + 0.25)
        source[task] = clip01(top + jitter(plan, task, "source"))
        skill[task] = clip01(top * (1.0 - 0.55 * removed ** 2) + heal + jitter(plan, task, "skill"))
        random_[task] = clip01(top * (1.0 - 0.95 * removed ** 0.7) + heal + jitter(plan, task, "random"))
        general[task] = clip01(top * (1.0 - 0.8 * removed) + heal + jitter(plan, task, "general"))
    return source, skill, random_, general


def stub_removed(plan, estimate):
    if plan["method"] == "experts":
        return {"kind": "moe", "experts_before": estimate["experts_before"], "experts_kept": estimate["experts_kept"],
                "experts_removed_per_layer": estimate["experts_before"] - estimate["experts_kept"],
                "layers": estimate["layers"], "top_k_after": estimate["top_k_after"], "kept_share_of_skill": None}
    before = estimate["layers_before"]
    count = estimate["layers_dropped"]
    dropped = []
    if count > 0:
        step = (before - 2) / float(count)
        for index in range(count):
            dropped.append(int(1 + index * step))
    return {"kind": "dense", "layers_before": before, "layers_after": before - count, "layers_dropped": dropped,
            "keep_mlp": estimate["keep_width"], "keep_heads": estimate["keep_width"]}


def stub_result(spec, plan, options):
    """A made-up shrink result with the real result's shape. It does not load any model."""
    pace = settings.pace_s()
    estimate = sizing.estimate_shrink(plan)
    tasks = options["tasks"]
    time.sleep(pace)
    source, skill, random_, general = stub_scores(spec, plan, estimate, tasks)
    time.sleep(pace)
    steps = int(plan["heal_steps"])
    heal = {"steps": steps, "kind": "distillation" if steps > 0 else "none", "teacher": steps > 0}
    log = ["Stub run: no model was loaded and every number is made up.",
           shrink_spec.plan_summary(plan, estimate),
           "Heal: " + str(steps) + " steps of distillation toward the original (stub)."]
    seconds = {"load": 0.0, "rank": 0.0, "slim": 0.0, "heal": round(estimate["heal_min"] * 60.0, 1), "eval": 0.0}
    control = None
    if options["control"]:
        control = random_
        log.append("Random control: the same counts removed at random, same heal (stub).")
    general_scores = None
    if options["general_control"]:
        general_scores = general
    source_scores = None
    if options["score_source"]:
        source_scores = source
    return finish(spec, plan, skill, estimate["total_params_b"], estimate["run_params_b"],
                  stub_removed(plan, estimate), source_scores, control, general_scores, heal, log, seconds, True,
                  {"estimated": True})


# Modal backend.

def modal_plan(plan):
    """The part of the plan the remote function needs."""
    out = {}
    for key in PLAN_KEYS:
        out[key] = plan.get(key)
    return out


def modal_result(spec, plan, options):
    """Run the shrink on a GPU and shape the result. Raises RuntimeError with the remote error."""
    options = dict(options)
    options["max_heal_min"] = (spec.get("limits") or {}).get("max_heal_min")
    raw = modal_backend.run_shrink(modal_plan(plan), "search", options["tasks"], options["fidelity"], options)
    params = raw["params"]
    heal = dict(raw["heal"])
    log = list(raw.get("log") or [])
    warning = modal_backend.deployed_warning()
    if warning is not None:
        log.insert(0, warning)
    return finish(spec, plan, raw["scores"], round(params["total_after"] / 1e9, 4),
                  round(params["run_after"] / 1e9, 4), raw["removed"], raw.get("source_scores"),
                  raw.get("control_scores"), raw.get("general_scores"), heal, log, raw.get("seconds") or {}, False,
                  {"estimated": False, "gpu": raw.get("gpu"), "params_before_b": round(params["total_before"] / 1e9, 4),
                   "items": raw.get("items")})
