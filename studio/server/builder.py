"""Builder sessions: one env per session, jobs in a thread pool, and rule-based suggestions."""

import concurrent.futures
import json
import random
import threading
import time
import uuid

from wsp.swarm import config as config_mod
from wsp.swarm import factory
from wsp.swarm import program_schema as ps
from wsp.swarm import space

from studio.server import catalog as catalog_mod
from studio.server import programs
from studio.server import settings
from studio.server import specs as specs_mod
from studio.server.runstate import number

BUILDER_EVALS = 12
BUILDER_COMPUTE_MIN = 45
DEFAULT_HEAL_STEPS = 300
DEFAULT_LORA_RANK = 16
BEST_DEPTH_FRACTION = 6 / 16.0

_lock = threading.Lock()
_sessions = {}
_jobs = {}
_pool = concurrent.futures.ThreadPoolExecutor(max_workers=4, thread_name_prefix="studio-builder")


class BuilderError(ValueError):
    """A builder request could not be done. The message is safe to show."""


class Session:
    def __init__(self, builder_id, spec, config, parts):
        self.id = builder_id
        self.spec = spec
        self.config = config
        self.parts = parts
        self.env = parts["env"]
        self.catalog = parts["catalog"]
        self.lock = threading.RLock()
        self.target_scores = {}      # program hash -> target score of its last score job
        self.host_params_b = catalog_mod.host_params_b(config["host"])

    def budget_dict(self):
        budget = self.env.budget
        return {"evals_used": number(budget.eval_used()), "evals_total": number(budget.eval_budget),
                "compute_used_s": number(budget.compute_used()), "compute_total_s": number(budget.compute_budget_s)}

    def target_ids(self):
        ids = []
        for item in self.spec["benchmarks"]:
            if item["kind"] == "target":
                ids.append(item["id"])
        return ids

    def target_score(self, scores):
        total = 0.0
        count = 0
        for task in self.target_ids():
            if scores is not None and scores.get(task) is not None:
                total += scores[task]
                count += 1
        if count == 0:
            return None
        return total / count

    def lora_bar(self):
        """The best known LoRA bar over the spec's targets, or None."""
        bar = None
        for item in self.spec["benchmarks"]:
            if item["kind"] == "target" and item.get("lora") is not None:
                if bar is None or item["lora"] > bar:
                    bar = item["lora"]
        return bar


def create_session(spec):
    """Start a builder session for a spec. Returns the session."""
    builder_id = "b_" + uuid.uuid4().hex[:8]
    overrides = specs_mod.run_overrides(spec, settings.backend(), "none")
    overrides["eval_budget"] = BUILDER_EVALS
    overrides["compute_budget_s"] = specs_mod.compute_budget_s(BUILDER_EVALS, BUILDER_COMPUTE_MIN)
    overrides["method"] = "builder"
    config = config_mod.make_config(overrides)
    config["run_id"] = builder_id + "-" + str(int(time.time() * 1000))
    parts = factory.build_run(config)
    session = Session(builder_id, spec, config, parts)
    with _lock:
        _sessions[builder_id] = session
    return session


def get_session(builder_id):
    with _lock:
        return _sessions.get(builder_id)


def get_job(job_id):
    with _lock:
        job = _jobs.get(job_id)
    if job is None:
        return None
    return public_job(job)


def public_job(job):
    with job["lock"]:
        return {"job_id": job["job_id"], "action": job["action"], "status": job["status"],
                "stub": job["stub"], "result": job["result"], "budget": job["budget"], "error": job["error"]}


def submit_action(session, program, action):
    """Queue a builder action. Returns the job id. Raises BuilderError for a bad request."""
    if action not in ("compile", "heal", "train", "score"):
        raise BuilderError("unknown action: " + str(action))
    clean = check_program(session, program)
    job = {"job_id": "j_" + uuid.uuid4().hex[:8], "action": action, "status": "queued",
           "stub": session.env.is_stub(), "result": None, "budget": session.budget_dict(), "error": None,
           "lock": threading.Lock()}
    with _lock:
        _jobs[job["job_id"]] = job
    _pool.submit(work_job, session, job, clean)
    return job["job_id"]


def check_program(session, program):
    try:
        clean = programs.normalize(program)
    except programs.ProgramError as error:
        raise BuilderError("the program is not valid: " + error.message)
    errors = space.validate(clean, session.catalog, session.config)
    if len(errors) > 0:
        raise BuilderError("the program is not valid: " + "; ".join(errors))
    return clean


def work_job(session, job, program):
    with job["lock"]:
        job["status"] = "running"
    try:
        result = run_action(session, job["action"], program)
        with job["lock"]:
            job["result"] = result
            job["status"] = "done"
    except Exception as error:
        with job["lock"]:
            job["error"] = type(error).__name__ + ": " + str(error)
            job["status"] = "error"
    with job["lock"]:
        job["budget"] = session.budget_dict()


# Actions.

def with_default_heal(session, program):
    if ps.heal(program) is not None:
        return program
    return ps.with_heal(program, space.heal_with_steps(DEFAULT_HEAL_STEPS, session.config))


def with_lora(program):
    if program.get("lora") is not None:
        return program
    data = ps.normalize(program)
    data["lora"] = {"rank": DEFAULT_LORA_RANK, "budget_params": None, "targets": None, "alpha": None,
                    "layers": None}
    return ps.normalize(data)


def run_action(session, action, program):
    """Do one action on a program. Returns the result dict of the contract."""
    with session.lock:
        if action == "compile":
            return compile_result(session, program)
        if action == "heal":
            program = with_default_heal(session, program)
        if action == "train":
            program = with_default_heal(session, with_lora(program))
        return score_result(session, program)


def fit_dict(report):
    """Fit per graft from a compile report, keyed by part id and host layer, like <part id>@6."""
    fit = {}
    for entry in report:
        key = str(entry.get("id")) + "@" + str(entry.get("host_layer"))
        if entry.get("control") is not None:
            key = key + "~" + str(entry["control"])
        value = entry.get("fit_r2")
        if value is not None:
            value = round(float(value), 3)
        fit[key] = value
    return fit


def params_b_of(session, report):
    total = 0
    for entry in report:
        total += int(entry.get("part_params") or 0) + int(entry.get("glue_params") or 0)
    return number(session.host_params_b + total / 1e9)


def compile_result(session, program):
    probe = session.env.probe(program, "you", None, "builder compile")
    if probe["status"] not in ("ok", "cache_hit"):
        detail = "; ".join(probe.get("errors") or [])
        raise BuilderError("compile failed: " + probe["status"] + (" " + detail if detail else ""))
    report = probe["graft_report"]
    return {"fit": fit_dict(report), "scores": None, "combined": None, "control": None,
            "params_b": params_b_of(session, report), "compute_s": number(probe["compute_s"]), "status": "ok",
            "program": program}


def row_compute_s(row):
    total = 0.0
    for phase in row.get("compute_s") or {}:
        total += float(row["compute_s"][phase])
    return total


def score_result(session, program):
    env = session.env
    row = env.step(program, "you", "builder score")[4]
    status = row["status"]
    if status == "cache_hit":
        status = "ok"
    report = row.get("graft_report") or []
    result = {"fit": fit_dict(report), "scores": None, "combined": None, "control": None,
              "params_b": params_b_of(session, report), "compute_s": number(row_compute_s(row)),
              "status": status, "program": program}
    if row.get("scores"):
        result["scores"] = dict(row["scores"])
        result["combined"] = number(row["combined"])
        target = session.target_score(row["scores"])
        if target is not None:
            session.target_scores[space.program_hash(program)] = target
    if status in ("ok", "collapsed"):
        result["control"] = control_result(session, program)
    return result


def has_real_grafts(program):
    for graft in ps.grafts(program):
        if ps.graft_control(graft) is None:
            return True
    return False


def control_result(session, program):
    """Score the random-part control of a program. Returns {"scores": ...} or None."""
    if not has_real_grafts(program):
        return None
    rng = random.Random(zlib_seed(program))
    control = space.random_glue_control(program, rng, session.catalog)
    row = session.env.step(control, "you", "random-part control")[4]
    if not row.get("scores"):
        return None
    return {"scores": dict(row["scores"])}


def zlib_seed(program):
    return int(space.program_hash(program)[:8], 16)


# Suggestions.

def suggest(session, program):
    """Return a list of rule-based suggestions for a program."""
    program = check_program(session, program)
    with session.lock:
        suggestions = []
        suggestions.extend(add_part_suggestions(session, program))
        suggestions.extend(move_suggestions(session, program))
        suggestions.extend(heal_suggestions(session, program))
        suggestions.extend(rank_suggestions(session, program))
        suggestions.extend(mask_suggestions(session, program))
        suggestions.extend(handoff_suggestions(session, program))
        suggestions.extend(llm_suggestions(session, program))
    out = []
    for item in suggestions:
        if len(space.validate(item["program"], session.catalog, session.config)) == 0:
            out.append(item)
    return out


def best_depth(session):
    n_host = space.host_layers(session.config, session.catalog)
    return int(round(BEST_DEPTH_FRACTION * n_host))


def add_part_suggestions(session, program):
    """For an empty draft, probe a few likely parts at a few depths and suggest the best fit.

    Probes are free of evals (they use compute only), so the suggestion never
    points at a placement the fit gate would reject.
    """
    grafts = ps.grafts(program)
    if len(grafts) > 0 or session.config["max_grafts"] < 1:
        return []
    ingredients = specs_mod.ingredients_for(session.config, session.catalog, session.spec["benchmarks"])
    if len(ingredients) == 0:
        return []
    gate = session.config["fit_threshold"]
    best = None
    for item in ingredients[:3]:
        layers = [best_depth(session)]
        default = space.resolved_at(ps.make_graft(item["part_id"]), session.catalog, session.config)
        if default is not None and default not in layers:
            layers.append(default)
        for at in layers:
            graft = ps.make_graft(item["part_id"], at, glue_rank=64)
            changed = ps.with_grafts(program, [graft])
            try:
                probe = session.env.probe(changed, "you", None, "suggest: starting part")
            except Exception:
                continue
            fit = graft_fit(probe, 0)
            if fit is None:
                continue
            if best is None or fit > best["fit"]:
                best = {"item": item, "at": at, "fit": fit, "program": changed}
    if best is None or best["fit"] < gate:
        return []
    changed = with_default_heal(session, best["program"])
    label = best["item"]["label"]
    detail = ("Start with " + label + " at host layer " + str(best["at"]) + ", rank-64 glue and a " +
              str(DEFAULT_HEAL_STEPS) + " step heal. A free probe gives it fit " + "%.2f" % best["fit"] +
              ", the best of the likely parts.")
    return [{"title": "Add " + label, "detail": detail, "tag": "fit " + "%.2f" % best["fit"], "program": changed}]


def graft_fit(probe, index):
    report = probe["graft_report"]
    if index >= len(report):
        return None
    return report[index].get("fit_r2")


def candidate_layers(session, graft):
    n_host = space.host_layers(session.config, session.catalog)
    options = [best_depth(session)]
    default = space.resolved_at(graft, session.catalog, session.config)
    if default is not None:
        options.append(default)
    current = ps.graft_at(graft)
    if current is not None:
        options.append(current - 2)
        options.append(current + 2)
    out = []
    for layer in options:
        if layer < 0 or layer > n_host - 1 or layer == current or layer in out:
            continue
        out.append(layer)
    return out[:3]


def move_suggestions(session, program):
    """Probe up to three other layers for the first two grafts and suggest a better fit."""
    out = []
    grafts = ps.grafts(program)
    gate = session.config["fit_threshold"]
    for index in range(min(2, len(grafts))):
        graft = grafts[index]
        if ps.graft_control(graft) is not None:
            continue
        here = session.env.probe(program, "you", None, "suggest: fit here")
        current_fit = graft_fit(here, index)
        best = None
        for layer in candidate_layers(session, graft):
            moved = list(grafts)
            moved[index] = ps.graft_with_at(graft, layer)
            changed = ps.with_grafts(program, moved)
            probe = session.env.probe(changed, "you", None, "suggest: fit at layer " + str(layer))
            if probe["status"] not in ("ok", "cache_hit"):
                continue
            fit = graft_fit(probe, index)
            if fit is None:
                continue
            if best is None or fit > best["fit"]:
                best = {"layer": layer, "fit": fit, "program": changed}
        if best is None:
            continue
        name = space.short_name(ps.graft_id(graft))
        if current_fit is not None and best["fit"] >= gate and best["fit"] - current_fit >= 0.03:
            detail = ("Free fit probe: " + name + " fits at " + "%.2f" % current_fit + " now and at " +
                      "%.2f" % best["fit"] + " at host layer " + str(best["layer"]) + ".")
            out.append({"title": "Move " + name + " to layer " + str(best["layer"]), "detail": detail,
                        "tag": "fit " + "%.2f" % current_fit + " to " + "%.2f" % best["fit"],
                        "program": best["program"]})
            continue
        current_at = space.resolved_at(graft, session.catalog, session.config)
        target = best_depth(session)
        if current_at is not None and abs(current_at - target) > 1 and best["layer"] == target and best["fit"] >= gate:
            detail = ("In the round-2 sweep a graft at host layer " + str(target) + " scored 39.0% on GSM8K, "
                      "against 22 to 26% at layers 10 and 14. A probe puts the fit at layer " + str(target) +
                      " at " + "%.2f" % best["fit"] + ", above the gate of " + str(gate) + ".")
            out.append({"title": "Try " + name + " at layer " + str(target), "detail": detail,
                        "tag": "best depth so far", "program": best["program"]})
    return out


def heal_suggestions(session, program):
    if len(ps.grafts(program)) == 0 or ps.heal(program) is not None:
        return []
    changed = with_default_heal(session, program)
    detail = ("Grafts start with untrained glue, so a program with grafts and no heal usually scores at or below "
              "the host. A " + str(DEFAULT_HEAL_STEPS) + " step heal trains the glue.")
    return [{"title": "Add a heal pass", "detail": detail, "tag": "adds heal", "program": changed}]


def rank_suggestions(session, program):
    grafts = ps.grafts(program)
    changed_grafts = []
    count = 0
    for graft in grafts:
        if ps.graft_control(graft) is None and ps.graft_glue_rank(graft) is None:
            changed_grafts.append(ps.graft_with_glue_rank(graft, 64))
            count += 1
        else:
            changed_grafts.append(graft)
    if count == 0:
        return []
    changed = ps.with_grafts(program, changed_grafts)
    detail = ("In the round-2 sweep rank-64 glue scored 39.0% on GSM8K against 34.3% for full glue, and it has "
              "fewer glue parameters.")
    return [{"title": "Use rank-64 glue", "detail": detail, "tag": "fewer parameters", "program": changed}]


def remove_mask_id(node, mask_id):
    """Return the mask expression without one id, or None when nothing is left."""
    if node is None:
        return None
    op = node.get("op")
    if op == "use":
        if node.get("id") == mask_id:
            return None
        return node
    if op == "restrict":
        inner = remove_mask_id(node["inner"], mask_id)
        if inner is None:
            return None
        changed = dict(node)
        changed["inner"] = inner
        return changed
    left = remove_mask_id(node.get("a"), mask_id)
    right = remove_mask_id(node.get("b"), mask_id)
    if left is None:
        if op == "union":
            return right
        return None
    if right is None:
        return left
    changed = dict(node)
    changed["a"] = left
    changed["b"] = right
    return changed


def mask_task(record):
    try:
        from wsp_eval import tasks as eval_tasks
        return eval_tasks.task_for_training_set(str(record.get("trained_on") or ""))
    except Exception:
        return None


def mask_suggestions(session, program):
    out = []
    spec_ids = []
    for item in session.spec["benchmarks"]:
        spec_ids.append(item["id"])
    for mask_id in ps.mask_ids(program):
        record = session.catalog.records.get(mask_id)
        if record is None:
            continue
        task = mask_task(record)
        if task is None or task in spec_ids:
            continue
        data = ps.normalize(program)
        data["masks"] = remove_mask_id(data.get("masks"), mask_id)
        changed = ps.normalize(data)
        label = catalog_mod.mask_label(mask_id)
        detail = ("The " + label + " mask was trained for " + str(record.get("trained_on")) + ", and " + task +
                  " is not one of this spec's benchmarks. Removing it keeps the host's skills on the benchmarks "
                  "you chose.")
        out.append({"title": "Drop the " + label + " mask", "detail": detail, "tag": "drops a mask",
                    "program": changed})
    return out


def handoff_suggestions(session, program):
    bar = session.lora_bar()
    key = space.program_hash(program)
    score = session.target_scores.get(key)
    if bar is None or score is None or score >= bar:
        return []
    detail = ("Your last score on this program is " + "%.3f" % score + ", below the LoRA bar of " +
              "%.3f" % bar + ". The agents search placements, parts and heal settings and score each one, so "
              "they can look wider than one hand-built program.")
    return [{"title": "Hand off to the agents", "detail": detail, "tag": "run the swarm", "program": program}]


LLM_SCHEMA = {
    "type": "object",
    "properties": {"title": {"type": "string"}, "detail": {"type": "string"}, "program_text": {"type": "string"}},
    "required": ["title", "detail", "program_text"],
}


def llm_suggestions(session, program):
    """One optional suggestion from the configured LLM. Any failure gives no suggestion."""
    name = settings.llm()
    if name == "none":
        return []
    try:
        config = config_mod.make_config({"llm": name, "backend": settings.backend()})
        llm = factory.build_llm(config)
        system = ("You suggest one change to a wsp program. Reply with a JSON object with title, detail (one or "
                  "two plain sentences) and program_text (the full changed program in wsp text). Do not use "
                  "dashes as punctuation.")
        user = "Program:\n" + programs.render_or_empty(program) + "\nGoal: " + str(session.spec["title"])
        reply = llm.complete([{"role": "system", "content": system}, {"role": "user", "content": user}],
                             schema=LLM_SCHEMA, temperature=0.3)
        data = specs_mod.parse_json_reply(reply)
        changed = programs.parse(data["program_text"])
        return [{"title": str(data["title"])[:80], "detail": str(data["detail"])[:400], "tag": "from the LLM",
                 "program": changed}]
    except Exception:
        return []
