"""The join of the composer and the Sapien front door (wsp/auto.py).

The front door knows which compile methods are valid for a source: weave for
true fine-tunes, synth and feedback seams or tops for other sources, transplant
for same-width MoE, slim for removal. The composer chooses among them. Two tools
use this module:

    plan_request  a request dict in, the front door's plan out, as a short summary
    compile_plan  one chosen variant of that request, built by wsp.auto.make on Modal

Nothing here reads or returns weight values. A plan holds sizes, relations, and
reasons. A compile report is reduced to transfer, content_z, drift, nll change,
sizes, and verdicts.
"""

import hashlib
import json

from composer.sources import sanitize

# Settings the agent may set. Each is a knob of wsp.auto.SETTINGS with a range and the methods it changes.
KNOBS = {
    "block_layers": {"type": int, "min": 1, "max": 16, "methods": ["synth", "feedback"],
                     "meaning": "donor layers in the seam block (the block position follows the atlas or the depth rule)"},
    "lora_rank": {"type": int, "min": 1, "max": 256, "methods": ["synth", "feedback", "tops"],
                  "meaning": "rank of the host LoRA that comes with a seam or a top"},
    "transplant_k": {"type": int, "min": 1, "max": 16, "methods": ["transplant"],
                     "meaning": "routed experts per layer (shrinks until the limits hold)"},
    "transplant_rank": {"type": int, "min": 16, "max": 1024, "methods": ["transplant"],
                        "meaning": "stitch rank of a transplanted expert bank"},
    "delta_rank": {"type": int, "min": 1, "max": 512, "methods": ["delta"], "meaning": "rank of the difference"},
    "slim_keep": {"type": float, "min": 0.05, "max": 1.0, "methods": ["slim"],
                  "meaning": "share of experts per layer kept when max_total_params_b sets no size (a total limit "
                             "decides it otherwise)"},
    "slim_steps": {"type": int, "min": 0, "max": 2000, "methods": ["slim"], "meaning": "heal steps after slim"},
    "tops_cut": {"type": float, "min": 0.1, "max": 0.9, "methods": ["tops"],
                 "meaning": "share of the base layers kept as the shared trunk (cut depth); a size estimate only"},
    "stage1_steps": {"type": int, "min": 0, "max": 2000, "methods": ["synth", "feedback", "transplant"],
                     "meaning": "steps of the first seam stage"},
    "stage2_steps": {"type": int, "min": 0, "max": 2000, "methods": ["synth", "feedback", "transplant"],
                     "meaning": "steps of the second seam stage"},
    "weave_population": {"type": int, "min": 2, "max": 64, "methods": ["weave"], "meaning": "weave search population"},
    "weave_generations": {"type": int, "min": 1, "max": 32, "methods": ["weave"], "meaning": "weave search generations"},
    "weave_buckets": {"type": int, "min": 1, "max": 32, "methods": ["weave"], "meaning": "layer buckets of the weave"},
}

# The composer's constraint names against the front door's limit names.
LIMIT_NAMES = {"max_total_params_b": "max_total_params_b", "max_run_params_b": "max_active_params_b",
               "max_heal_minutes": "max_heal_minutes"}

# Fields of the front-door request the agent may not change.
FIXED_FIELDS = ("name", "base", "reported", "seed")

METHOD_RULES = [
    "Method rules (from LESSONS.md and results/README.md). They decide which method you may pick:",
    "- Mixing (weave, delta) is only for true fine-tunes: same shapes, same positional base, layer cosine near 1.",
    "  Qwen2.5-Coder and -Math against Qwen2.5-Instruct (cosine 0.2 to 0.5) gave chance under mixing, MLP experts,",
    "  and layer swaps. plan_request names the relation of each source and lists only the valid methods.",
    "- Seams (synth, feedback) and tops are for every other source: other size, diverged weights, other family.",
    "  A donor block needs a trained stitch. Tops keep the donor's own upper layers and readout, because every",
    "  attempt to write a donor block back into the base's upper layers failed.",
    "- Skill removal uses slim. For a MoE it keeps the top experts per layer by saliency, not by name or size.",
    "- Transplant is for a MoE of the same family and width as a dense base.",
    "- Every claim of transfer needs a scrambled or random control and two seeds. One seed moved GSM8K by 8 points",
    "  and a control moved 8 points on MMLU. content_z in a compile result is the control test; a result with no",
    "  content_z has no control behind it, so say so in the rationale. Run a second seed of any variant you",
    "  would finish with (set seed in the request; the budget counts it).",
    "- Benchmarks score only the final choice. No tool shows a benchmark number. The plan's reported list is fixed.",
    "- Do not chase differences under about 0.03 transfer. Check the base is in the first comparison you make.",
    "- Never write weight values. plan_request and compile_plan return sizes, relations, and probe numbers only.",
    "",
    "Tools for this request: plan_request(request, settings) returns the plan (relations, valid methods with",
    "reasons, sizes against the limits, refusals). compile_plan(request, settings) builds ONE variant of it and",
    "returns the probe report. Pin your choice in the request: add[i].method names the method of one skill, and",
    "add[i].sources lists the models. Knobs you may set in settings: " + ", ".join(sorted(KNOBS)) + ".",
    "A plan refusal costs nothing. A compile costs one unit of the compile budget. Finish with plan_hash from a",
    "compile_plan result (or with a program that you compiled and practiced).",
]


def method_rules_text():
    return "\n".join(METHOD_RULES)


def front_door_request(request):
    """The base front-door request of a composer request, or None."""
    front = request.get("front_door")
    return front if isinstance(front, dict) else None


def check_front_door(path, request):
    """Raise ValueError when the composer request's front_door block is not usable."""
    front = request.get("front_door")
    if front is None:
        return
    if not isinstance(front, dict):
        raise ValueError(f"{path}: front_door is a request object in the wsp/auto.py format")
    from wsp import auto

    try:
        clean = auto.validate_request(front)
    except auto.RequestError as error:
        raise ValueError(f"{path}: front_door: {error}")
    if request.get("base") is not None and request["base"] != clean["base"]:
        raise ValueError(f"{path}: front_door.base {clean['base']!r} is not the request's base {request['base']!r}")
    for entry in clean["add"]:
        if entry["skill"] not in request["skills"]:
            raise ValueError(f"{path}: front_door adds skill {entry['skill']!r}, which is not a skill of the request")


def build_variant(composer_request, args):
    """Build the front-door request and settings of one call. Return (request, settings, error text or None)."""
    front = front_door_request(composer_request)
    if front is None:
        return None, None, "this request has no front_door block, so plan_request and compile_plan are off"
    given = args.get("request")
    if given is None:
        given = {}
    if not isinstance(given, dict):
        return None, None, "request must be an object in the wsp/auto.py format, or null for the base request"
    variant = json.loads(json.dumps(front))
    for key in given:
        if key in FIXED_FIELDS:
            if given[key] != front.get(key):
                return None, None, f"request.{key} is fixed by the user's request and cannot change"
            continue
        variant[key] = given[key]
    from wsp import auto

    constraints = composer_request.get("constraints", {})
    limits = dict(variant.get("limits") or {})
    for name, front_name in LIMIT_NAMES.items():
        cap = constraints.get(name)
        if cap is None:
            continue
        value = limits.get(front_name)
        if value is None:
            limits[front_name] = cap
        elif value > cap:
            return None, None, f"limits.{front_name} {value} is above the request's cap {cap}; limits can only tighten"
    variant["limits"] = limits
    try:
        clean = auto.validate_request(variant)
    except auto.RequestError as error:
        return None, None, f"request: {error}"
    for entry in clean["add"]:
        if entry["skill"] not in composer_request["skills"]:
            return None, None, f"request adds skill {entry['skill']!r}, which is not a skill of the request"
    settings = args.get("settings")
    if settings is None:
        settings = {}
    if not isinstance(settings, dict):
        return None, None, "settings must be an object or null"
    out = {}
    for key in settings:
        if key not in KNOBS:
            return None, None, f"unknown setting {key!r}; the settings are {', '.join(sorted(KNOBS))}"
        knob = KNOBS[key]
        value = settings[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None, None, f"setting {key} must be a number"
        if knob["type"] is int:
            if int(value) != value:
                return None, None, f"setting {key} must be a whole number"
            value = int(value)
        else:
            value = float(value)
        if value < knob["min"] or value > knob["max"]:
            return None, None, f"setting {key} must be between {knob['min']} and {knob['max']}"
        out[key] = value
    return clean, out, None


def variant_hash(request, settings):
    text = json.dumps({"request": request, "settings": settings}, sort_keys=True)
    return hashlib.sha256(text.encode()).hexdigest()[:12]


def summarize_plan(plan, settings=None):
    """The plan as the agent sees it: relations, methods with reasons, sizes against limits, refusals."""
    sources = []
    for row in plan["sources"]:
        methods = []
        for option in row["methods"]:
            methods.append({key: option.get(key) for key in ("method", "fits", "reason", "why_not", "adds_total_b",
                                                             "needs", "runs_in_make")})
        sources.append({"skill": row["skill"], "source": row["source"], "relation": row["relation"],
                        "params_b": row.get("params_b"),
                        "tokenizer_same": row["flags"].get("tokenizer_same"),
                        "weave_reference": row.get("subtract_reference"),
                        "methods": methods, "chosen_by_front_door": row["chosen"]})
    stages = []
    for stage in plan["stages"]:
        item = {"method": stage["method"], "kind": stage["kind"], "heal_minutes": stage.get("heal_minutes"),
                "total_after_b": stage["params_total_after_b"], "run_after_b": stage["params_run_after_b"],
                "limit_problems": stage["limit_problems"]}
        if stage["method"] == "weave":
            item["members"] = [{"skill": m["skill"], "source": m["source"]} for m in stage["members"]]
        elif stage["method"] == "slim":
            item["target"] = {k: v for k, v in stage["target"].items() if k in ("kind", "keep", "experts", "note")}
        else:
            item["components"] = [{k: v for k, v in c.items() if k in ("skill", "source", "method", "adds_params",
                                                                       "n_layers", "k", "rank")}
                                  for c in stage.get("components", [])]
        stages.append(item)
    settings = dict(settings or {})
    effective = plan.get("settings") or {}
    knobs = {}
    for name in sorted(KNOBS):
        knobs[name] = {"value": effective.get(name), "methods": KNOBS[name]["methods"],
                       "range": [KNOBS[name]["min"], KNOBS[name]["max"]], "meaning": KNOBS[name]["meaning"]}
    out = {
        "ok": True,
        "feasible": bool(plan["feasible"]),
        "base": {"id": plan["base"]["id"], "params_b": plan["base"]["params_b"],
                 "active_params_b": plan["base"]["active_params_b"], "moe": plan["base"]["moe"]},
        "limits": plan["limits"],
        "sources": sources,
        "stages": stages,
        "sizes": {"final_total_b": plan["sizes"]["final_total_b"], "final_run_b": plan["sizes"]["final_run_b"]},
        "control_plan": plan["controls"],
        "refusals": list(plan["refusals"]),
        "warnings": list(plan["warnings"]),
        "settings_set": settings,
        "knobs": knobs,
        "how_to_choose": ("Pin a method with request.add[i].method and call compile_plan. A method with "
                          "runs_in_make false (tops) is valid but compile_plan cannot build it."),
    }
    if plan["refusals"]:
        out["refusal_text"] = "No plan runs: " + " | ".join(plan["refusals"])
    elif not plan["stages"]:
        out["refusal_text"] = "No plan runs: no stage was chosen for this request."
    return sanitize(out)


def mean(values):
    values = [v for v in values if v is not None]
    if len(values) == 0:
        return None
    return sum(values) / len(values)


def reduce_report(report):
    """Reduce a sapien_report to what the agent may see. No weights, no curves, no benchmark scores."""
    report = report or {}
    probe = report.get("probe") or {}
    skills = {}
    per_skill = {}
    for key, row in (probe.get("skills") or {}).items():
        if not isinstance(row, dict):
            continue
        item = {name: row.get(name) for name in ("source", "method", "transfer", "content_z", "verdict_word", "note",
                                                 "content_measured_at") if row.get(name) is not None}
        skills[key] = item
        skill = key.split(":")[0]
        per_skill.setdefault(skill, []).append(row.get("transfer"))
    transfer = {}
    for skill in per_skill:
        value = mean(per_skill[skill])
        transfer[skill] = 0.0 if value is None else value
    sizes = report.get("sizes") or {}
    final = sizes.get("final") or {}
    base = sizes.get("base") or {}
    nll = (probe.get("nll") or {}).get("delta") or {}
    stages = [{"method": s.get("method"), "status": s.get("status")} for s in report.get("stages") or []
              if isinstance(s, dict)]
    out = {
        "skills": skills,
        "transfer": transfer,
        "drift": probe.get("drift"),
        "nll_delta": nll,
        "sizes": {"base_total_b": None if not base.get("total") else base["total"] / 1e9,
                  "final_total_b": None if not final.get("total") else final["total"] / 1e9,
                  "final_run_b": None if not final.get("per_token") else final["per_token"] / 1e9,
                  "limit_problems": sizes.get("limit_problems") or []},
        "verdict": report.get("verdict"),
        "verdicts": report.get("verdicts"),
        "stages": stages,
        "seconds": report.get("seconds"),
    }
    return sanitize(out)


def default_plan_fn(request, settings):
    """Plan on this machine. It reads the Hub by range reads and no weights."""
    from wsp import auto

    return auto.plan(request, lineage=auto.LiveLineage(), settings=settings or None)
