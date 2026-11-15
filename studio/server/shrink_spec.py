"""The shrink request type in a spec: detection, the drafted plan, and checks for an edited plan.

A spec has request_type "build" (the default: add components to a host) or
"shrink" (remove the parts a model needs only for skills the person does not
want, as wsp/slim.py does). A shrink spec carries spec["shrink"], the plan.

This module does not import specs.py, so specs.py can call it.
"""

import math
import re

from wsp import healdata

from studio.server import shrink_models as models_mod
from studio.server import sizing

DEFAULT_HEAL_STEPS = 300
MIN_HEAL_STEPS = 0
MAX_HEAL_STEPS = 1000
DEFAULT_DENSE_FRACTION = 0.4
DEFAULT_KEEP_DIVISOR = 4
MIN_LIMIT_B = 0.05
TARGET_MARGIN = 0.97
MIN_TARGET_B = 0.05
LIMIT_STEP_B = 0.1

SKILL_POOLS = sorted(healdata.SOURCES)

# Phrases that mean "remove skills to make a smaller model". A request that matches none is a build.
STRONG_PATTERNS = [
    r"\bshrink\w*", r"\bslim\w*", r"\bsmaller version of\b", r"\bdrop everything\b",
    r"\bremove\b.{0,40}\b(?:skills?|experts?|capabilit\w+)\b", r"\bdistill\w*\s+(?:it\s+)?down\b",
    r"\bkeep\s+(?:only|just)\b", r"\bonly\s+keep\b", r"\bstrip\w*\s+(?:it\s+)?down\b", r"\btrim\w*\s+down\b",
    r"\bprune\b", r"\bonly\s+(?:the\s+)?\w+\s+(?:experts|skills|layers)\b", r"\bdrop\b.{0,30}\bexperts?\b",
]
# "-only" is a shrink word ("a coding-only model") unless it names hardware or a language.
ONLY_SUFFIX = re.compile(r"\b([a-z]+)-only\b")
NOT_SKILL_ONLY = ["cpu", "gpu", "offline", "english", "text", "inference", "read", "api", "web", "mobile", "ios",
                  "android", "browser", "local", "on", "device"]

# A size after these words is a limit on the total.
NOT_PER_TOKEN = r"(?!\s*(?:params\s+|parameters\s+)?(?:per[ -]token|active))"
TOTAL_PATTERN = (r"(?:under|below|less than|at most|max(?:imum)?|up to|smaller than|down to|get (?:it )?to|"
                 r"get (?:it )?under|to about|to around|<=?)\s*~?(\d+(?:\.\d+)?)\s*b\b" + NOT_PER_TOKEN)


class ShrinkSpecError(ValueError):
    """A shrink plan is not valid. The message is plain text for the person."""


def looks_like_shrink(prompt):
    """True when the request asks to remove skills or make a smaller version of a model."""
    text = " ".join(str(prompt or "").lower().split())
    for pattern in STRONG_PATTERNS:
        if re.search(pattern, text):
            return True
    for match in ONLY_SUFFIX.finditer(text):
        if match.group(1) not in NOT_SKILL_ONLY:
            return True
    return False


def stated_total_b(prompt):
    """The size limit in the request ("under 3B"), or None."""
    text = " ".join(str(prompt or "").lower().split())
    match = re.search(TOTAL_PATTERN, text)
    if match is None:
        return None
    return float(match.group(1))


def moe_size_hint_b(prompt):
    """The size in "a 30B MoE", or None."""
    text = " ".join(str(prompt or "").lower().split())
    match = re.search(r"(\d+(?:\.\d+)?)\s*b\s*(?:-\s*)?(?:moe|mixture)", text)
    if match is None:
        return None
    return float(match.group(1))


def wants_moe(prompt):
    text = " ".join(str(prompt or "").lower().split())
    return re.search(r"\bmoe\b|\bmixture of experts\b|\bexperts?\b", text) is not None


def round_up(value, step):
    return round(math.ceil(value / step - 1e-9) * step, 4)


def label_of(skill):
    return models_mod.SKILL_LABELS.get(skill, str(skill) + " text")


def default_rule(method):
    if method == "experts":
        return models_mod.DEFAULT_EXPERT_RULE
    return models_mod.DEFAULT_DENSE_RULE


def experts_for_target(facts, target_b):
    """The most experts per layer that keep the total at or under the target (at least 1)."""
    expert_b, other_b = models_mod.expert_split(facts)
    per_expert = expert_b / facts["experts"]
    if per_expert <= 0:
        return facts["experts"]
    keep = int(math.floor((target_b - other_b) / per_expert + 1e-9))
    return max(1, min(keep, facts["experts"]))


def new_block(facts, skill, target_b=None, rule=None, heal_steps=DEFAULT_HEAL_STEPS):
    """A plan for one source model. target_b is the wanted total size in billions, or None for the default cut."""
    block = {"source_model": facts["hf_id"], "source_params_b": round(facts["params_b"], 4),
             "source_active_params_b": round(models_mod.run_params_b(facts), 4),
             "family": facts["kind"], "keep_skill": skill, "heal_steps": int(heal_steps),
             "experts_total": facts["experts"] if facts["kind"] == "moe" else None}
    if facts["kind"] == "moe":
        block["method"] = "experts"
        if target_b is None:
            keep = max(1, facts["experts"] // DEFAULT_KEEP_DIVISOR)
        else:
            keep = experts_for_target(facts, target_b)
        block["keep_experts"] = keep
        block["target_params_b"] = None
    else:
        block["method"] = "dense"
        block["keep_experts"] = None
        if target_b is None:
            target_b = facts["params_b"] * DEFAULT_DENSE_FRACTION
        block["target_params_b"] = round(max(target_b, MIN_TARGET_B), 3)
    block["rule"] = rule or default_rule(block["method"])
    if block["method"] == "experts":
        estimate = sizing.estimate_shrink(dict(block))
        block["target_params_b"] = estimate["total_params_b"]
    return block


def draft_plan(prompt, hint_model=None, hint_skill=None):
    """The shrink plan for a request, and the limits that follow from it.

    Returns (block, limits). hint_model and hint_skill come from an LLM and are used only when the words of
    the request do not settle the same question. Raises ShrinkSpecError when no model can be a source.
    """
    models = models_mod.all_facts()
    notes = []
    skill = models_mod.skill_from_text(prompt) or hint_skill
    if skill not in healdata.SOURCES:
        skill = "general"
    stated = stated_total_b(prompt)
    named = models_mod.named_model(prompt, models)
    if named is None:
        unsupported = models_mod.named_unsupported(prompt, models)
        if unsupported is not None:
            notes.append(unsupported + " is in the library, but wsp.slim cannot slim its architecture, so the "
                         "studio picked another source.")
    if named is None and hint_model in models and models[hint_model]["supported"]:
        named = hint_model
    target_for_choice = stated
    moe_hint = moe_size_hint_b(prompt)
    want_moe = wants_moe(prompt) or moe_hint is not None
    if named is None:
        named = models_mod.choose_source(models, skill, target_for_choice, want_moe, moe_hint)
        if named is None:
            raise ShrinkSpecError("no source model in the library is larger than " + str(stated) + "B and can be "
                                  "slimmed")
    facts = models[named]
    target = None
    if stated is not None and stated < facts["params_b"]:
        target = stated
        if facts["kind"] == "dense":
            target = stated * TARGET_MARGIN
    block = new_block(facts, skill, target)
    block["notes"] = notes
    block["candidates"] = models_mod.candidates_for(models, skill, stated, want_moe, named, moe_hint)
    estimate = sizing.estimate_shrink(block)
    total_limit = stated
    if total_limit is None:
        total_limit = round_up(estimate["total_params_b"], LIMIT_STEP_B)
    run_limit = round_up(estimate["run_params_b"], LIMIT_STEP_B)
    return block, {"max_total_params_b": total_limit, "max_run_params_b": min(run_limit, total_limit),
                   "max_heal_min": None}


# Checking an edited plan.

def to_int(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or int(value) != value:
        raise ShrinkSpecError(name + " must be a whole number")
    return int(value)


def to_number(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ShrinkSpecError(name + " must be a number")
    return float(value)


def check_block(incoming, old):
    """The checked plan from an edit. incoming holds the fields the person changed; old is the current plan.

    A changed source model gets that model's defaults for every field the edit does not name. Editing the
    target size of an MoE plan recomputes the experts to keep, unless the edit also names keep_experts.
    Raises ShrinkSpecError with a plain message.
    """
    if not isinstance(incoming, dict):
        raise ShrinkSpecError("shrink must be a JSON object")
    models = models_mod.all_facts()
    source = incoming.get("source_model", old["source_model"])
    if not isinstance(source, str) or source not in models:
        raise ShrinkSpecError("the source model must be a model in the library: " + str(source))
    facts = models[source]
    if not facts["supported"]:
        raise ShrinkSpecError(source + " cannot be slimmed: wsp.slim does not support its architecture")
    changed_source = source != old["source_model"]
    skill = incoming.get("keep_skill", old["keep_skill"])
    if skill not in healdata.SOURCES:
        raise ShrinkSpecError("keep_skill must be one of " + ", ".join(SKILL_POOLS))
    steps = to_int(incoming.get("heal_steps", old["heal_steps"]), "heal_steps")
    if steps < MIN_HEAL_STEPS or steps > MAX_HEAL_STEPS:
        raise ShrinkSpecError("heal_steps must be from " + str(MIN_HEAL_STEPS) + " to " + str(MAX_HEAL_STEPS))
    method = "experts" if facts["kind"] == "moe" else "dense"
    wanted = incoming.get("method")
    if wanted is not None and wanted != method:
        if wanted == "experts":
            raise ShrinkSpecError(source + " is a dense model, so it has no experts to remove")
        if wanted == "dense":
            raise ShrinkSpecError(source + " is an MoE model: remove experts, not layers and width")
        raise ShrinkSpecError("method must be experts or dense")
    rules = models_mod.EXPERT_RULES if method == "experts" else models_mod.DENSE_RULES
    rule = incoming.get("rule")
    if rule is None:
        rule = old["rule"] if not changed_source and old["rule"] in rules else default_rule(method)
    if rule not in rules:
        raise ShrinkSpecError("rule must be one of " + ", ".join(rules) + " for " + method + " removal")
    target_in = incoming.get("target_params_b")
    keep_in = incoming.get("keep_experts")
    if changed_source:
        base_target = None
        if incoming.get("target_params_b") is not None:
            base_target = to_number(target_in, "target_params_b")
        elif method == "dense" and old.get("target_params_b") and old["target_params_b"] < facts["params_b"]:
            base_target = old["target_params_b"]
        block = new_block(facts, skill, base_target, rule, steps)
        if keep_in is not None and method == "experts":
            block["keep_experts"] = to_int(keep_in, "keep_experts")
    else:
        block = dict(old)
        block.update({"keep_skill": skill, "heal_steps": steps, "rule": rule, "method": method})
        if method == "experts":
            if keep_in is not None:
                block["keep_experts"] = to_int(keep_in, "keep_experts")
            elif target_in is not None and target_in != old.get("target_params_b"):
                block["keep_experts"] = experts_for_target(facts, to_number(target_in, "target_params_b"))
        else:
            if target_in is not None:
                block["target_params_b"] = to_number(target_in, "target_params_b")
    if method == "experts":
        keep = block["keep_experts"]
        if keep is None or keep < 1 or keep > facts["experts"]:
            raise ShrinkSpecError("keep_experts must be from 1 to " + str(facts["experts"]) + " for " + source)
        block["target_params_b"] = None
        estimate = sizing.estimate_shrink(block)
        block["target_params_b"] = estimate["total_params_b"]
    else:
        target = block["target_params_b"]
        floor = sizing.min_dense_target_b(facts)
        if target is None or target <= floor:
            raise ShrinkSpecError("target_params_b must be above " + sizing.fmt_b(floor) + "B: the embedding and "
                                  "the head alone take that much and slimming does not touch them")
        if target >= facts["params_b"]:
            raise ShrinkSpecError("target_params_b must be below the source size of " +
                                  sizing.fmt_b(facts["params_b"]) + "B")
        block["keep_experts"] = None
    block["source_model"] = source
    block["source_params_b"] = round(facts["params_b"], 4)
    block["source_active_params_b"] = round(models_mod.run_params_b(facts), 4)
    block["family"] = facts["kind"]
    block["experts_total"] = facts["experts"] if method == "experts" else None
    if changed_source or skill != old.get("keep_skill") or "candidates" not in block:
        target_hint = None
        if method == "dense":
            target_hint = block["target_params_b"]
        block["candidates"] = models_mod.candidates_for(models, skill, target_hint, False, source)
        block["notes"] = []
    return block


# Plain-language summary.

def plan_summary(block, estimate):
    """One or two sentences that say what the plan does and its estimated size."""
    rule = models_mod.rule_phrase(block["rule"])
    if block["method"] == "experts":
        text = ("Keep " + str(block["keep_experts"]) + " of " + str(estimate["experts_before"]) +
                " experts per layer " + rule + ".")
    else:
        kept = int(round(estimate["keep_width"] * 100))
        text = ("Remove " + str(estimate["layers_dropped"]) + " of " + str(estimate["layers_before"]) +
                " layers and keep about " + str(kept) + "% of the width " + rule + ".")
    text += (" About " + sizing.fmt_b(round(estimate["total_params_b"], 1)) + "B stored, " +
             sizing.fmt_b(round(estimate["run_params_b"], 1)) + "B run per token (estimated).")
    return text


def draft_summary(block, estimate):
    """The spec summary of a shrink request."""
    return ("Shrink " + block["source_model"].split("/")[-1] + " and keep the " + block["keep_skill"] + " skill. " +
            plan_summary(block, estimate))


# Benchmarks that score each skill. These are scored on the finished model only; nothing trains on them.
SKILL_TARGETS = {"code": ["humaneval"], "math": ["gsm8k"], "medicine": ["mmlupro:health"], "general": ["mmlu"],
                 "prose": ["mmlu"]}


def targets_for(skill):
    return list(SKILL_TARGETS.get(skill, ["mmlu"]))
