"""The link from a studio spec to the compiler's front door (wsp/auto.py).

request_from_spec turns a spec into a request in the format of docs/auto.md.
plan_for_spec runs the front door's plan() in a worker thread. The plan reads
the Hugging Face Hub by range reads: no GPU and no weights. When the Hub is
not reachable the plan falls back to a small table of facts (offline_facts.json).
"""

import json
import logging
import os
import re
import threading
import time

from wsp import auto
from wsp import healdata

log = logging.getLogger("studio.frontdoor")

HERE = os.path.dirname(os.path.abspath(__file__))
FACTS_FILE = os.path.join(HERE, "offline_facts.json")
PLAN_TIMEOUT_S = 90.0
NOTE = "The compiler's plan is new and its lineage check is still being fixed. Treat it as a draft."

# Studio benchmark ids and the Hugging Face dataset each one is scored on.
DATASETS = {
    "gsm8k": "openai/gsm8k",
    "mmlu": "cais/mmlu",
    "humaneval": "openai/openai_humaneval",
    "mbpp": "google-research-datasets/mbpp",
    "if_guard": "google/IFEval",
}
MMLUPRO_DATASET = "TIGER-Lab/MMLU-Pro"
BBEH_DATASET = "BBEH/bbeh"

# The skill each benchmark measures, as a wsp.healdata pool name.
BENCH_SKILLS = {
    "gsm8k": "math", "mmlupro:math": "math", "humaneval": "code", "mbpp": "code", "mmlu": "knowledge",
    "mmlupro:health": "medicine", "mmlupro:biology": "medicine", "mmlupro:law": "law",
    "bbeh_mini": "reasoning", "if_guard": "general",
}
# Words in a donor's label or model id that name a skill.
DONOR_WORDS = {"math": ["math"], "code": ["code", "coder"], "medicine": ["med", "bio"], "law": ["law", "legal"]}

_lock = threading.Lock()
_plans = {}


def slug(title):
    """A request name from a title: letters, digits and underscores."""
    text = re.sub(r"[^a-z0-9]+", "_", str(title or "").lower()).strip("_")
    text = text[:60].strip("_")
    if text == "":
        return "sapien_request"
    return text


def dataset_for(bench_id):
    """The Hugging Face dataset id of a studio benchmark id, or None when it is not known."""
    if bench_id in DATASETS:
        return DATASETS[bench_id]
    if bench_id.startswith("mmlupro:"):
        return MMLUPRO_DATASET
    if bench_id.startswith("bbeh"):
        return BBEH_DATASET
    return None


def reported_for(spec):
    """(dataset ids, unmapped benchmark ids) for the benchmarks of a spec."""
    reported = []
    unmapped = []
    for bench in spec.get("benchmarks") or []:
        bench_id = str(bench.get("id"))
        dataset = dataset_for(bench_id)
        if dataset is None:
            if bench_id not in unmapped:
                unmapped.append(bench_id)
        elif dataset not in reported:
            reported.append(dataset)
    return reported, unmapped


def skills_for(spec):
    """The skills a build spec adds: from its benchmarks first, then its task type."""
    skills = []
    for bench in spec.get("benchmarks") or []:
        if bench.get("kind") != "target":
            continue
        skill = BENCH_SKILLS.get(str(bench.get("id")))
        if skill is not None and skill not in skills:
            skills.append(skill)
    task = spec.get("task_type")
    if len(skills) == 0 and task in healdata.SOURCES:
        skills.append(task)
    return skills


def guard_skills(spec):
    """The skills the guards protect."""
    skills = []
    for bench in spec.get("benchmarks") or []:
        if bench.get("kind") != "guard":
            continue
        skill = BENCH_SKILLS.get(str(bench.get("id")))
        if skill is not None and skill not in skills:
            skills.append(skill)
    return skills


def sources_for(spec, skill, base):
    """The donor models the spec names for a skill, or "auto" when it names none."""
    found = []
    for item in spec.get("ingredients") or []:
        donor = item.get("donor")
        if not isinstance(donor, str) or donor == base or donor in found:
            continue
        text = (str(item.get("label", "")) + " " + donor).lower()
        for word in DONOR_WORDS.get(skill, []):
            if word in text:
                found.append(donor)
                break
    if len(found) == 0:
        return "auto"
    return found


def limits_for(spec):
    """The request limits from the spec limits."""
    limits = spec.get("limits") or {}
    size = spec.get("size") or {}
    total = limits.get("max_total_params_b")
    if total is None:
        total = size.get("max_params_b")
    return {"max_total_params_b": total, "max_active_params_b": limits.get("max_run_params_b"),
            "max_heal_minutes": limits.get("max_heal_min")}


def request_and_unmapped(spec):
    """(request dict, unmapped benchmark ids) for a spec. The request is not validated here."""
    reported, unmapped = reported_for(spec)
    request = {"name": slug(spec.get("title"))}
    if spec.get("request_type") == "shrink" and isinstance(spec.get("shrink"), dict):
        block = spec["shrink"]
        request["base"] = block.get("source_model")
        request["add"] = []
        request["remove"] = [{"skill": "*"}]
        request["keep"] = [block.get("keep_skill")]
    else:
        base = spec.get("host")
        request["base"] = base
        request["add"] = []
        for skill in skills_for(spec):
            request["add"].append({"skill": skill, "sources": sources_for(spec, skill, base), "components": None})
        request["remove"] = []
        request["keep"] = guard_skills(spec)
    request["limits"] = limits_for(spec)
    if len(reported) > 0:
        request["reported"] = reported
    request["seed"] = 0
    return request, unmapped


def request_from_spec(spec):
    """The front door request for a spec, in the format of docs/auto.md."""
    request, unmapped = request_and_unmapped(spec)
    return request


def check_request(request):
    """The front door's own check. Returns a list of plain messages, empty when the request is valid."""
    try:
        auto.validate_request(request)
    except auto.RequestError as error:
        return [str(error)]
    return []


def request_view(spec):
    """The body of GET /api/spec/{id}/request."""
    request, unmapped = request_and_unmapped(spec)
    return {"request": request, "unmapped": unmapped, "errors": check_request(request)}


# Plans.

def offline_lineage():
    """A StaticLineage from offline_facts.json."""
    with open(FACTS_FILE) as handle:
        data = json.load(handle)
    relations = {}
    for row in data.get("relations", []):
        relations[(row[0], row[1])] = row[2]
    return auto.StaticLineage(data["models"], relations)


def live_lineage():
    """The Hub lineage. Raises ImportError when huggingface_hub is not installed."""
    import huggingface_hub  # noqa: F401  (checks that the Hub client exists before the plan starts)
    return auto.LiveLineage()


def run_plan(request, lineage):
    """The front door's plan(). A separate function so tests can replace it."""
    return auto.plan(request, lineage=lineage)


def plan_with_fallback(request):
    """(plan, offline, why). Plans from the Hub, and from the offline table when the Hub fails."""
    try:
        return run_plan(request, live_lineage()), False, None
    except (auto.RequestError, auto.PlanError):
        raise
    except Exception as error:
        log.warning("hub plan failed, trying the offline facts: %r", error)
        hub_error = str(error).splitlines()[0][:200] if str(error) else error.__class__.__name__
    try:
        return run_plan(request, offline_lineage()), True, hub_error
    except KeyError as error:
        raise RuntimeError("the Hub is not reachable (" + hub_error + ") and the offline facts have no row for " +
                           str(error.args[0]))


def run_with_timeout(function, argument, seconds):
    """Run function(argument) in a worker thread. Returns (result, error, timed_out)."""
    box = {}

    def work():
        try:
            box["result"] = function(argument)
        except Exception as error:
            log.exception("front door plan failed")
            box["error"] = error

    thread = threading.Thread(target=work, daemon=True)
    thread.start()
    thread.join(seconds)
    if thread.is_alive():
        return None, None, True
    return box.get("result"), box.get("error"), False


def plan_for_spec(spec, seconds=None):
    """The body of POST /api/spec/{id}/plan, and the status code. The plan is cached per spec version."""
    if seconds is None:
        seconds = PLAN_TIMEOUT_S
    request, unmapped = request_and_unmapped(spec)
    errors = check_request(request)
    if errors:
        return {"error": "The request is not valid: " + errors[0], "errors": errors, "request": request}, 400
    key = json.dumps(request, sort_keys=True)
    with _lock:
        cached = _plans.get(spec.get("id"))
    if cached is not None and cached["key"] == key:
        return cached["body"], 200
    started = time.time()
    outcome, error, timed_out = run_with_timeout(plan_with_fallback, request, seconds)
    if timed_out:
        return {"error": "The compiler plan took longer than " + str(int(seconds)) + " seconds. Try again."}, 408
    if error is not None:
        return {"error": "The compiler could not make a plan: " + str(error)}, 422
    plan, offline, hub_error = outcome
    plan = json.loads(json.dumps(plan, default=str))
    body = {"plan": plan, "seconds": round(time.time() - started, 1), "offline": offline, "verified": False,
            "note": NOTE, "unmapped": unmapped}
    if offline:
        body["offline_reason"] = hub_error
    with _lock:
        _plans[spec.get("id")] = {"key": key, "body": body}
    return body, 200
