"""The comparison set for a finished composer run.

The final program is scored on the held-out sets against these arms, all
with the same heal data, the same heal settings, and the same budget:

- base: the host alone;
- lora_on_base: the host with a LoRA, trained on the program's heal data
  for the sum of the program's training steps;
- distill_on_base: the same LoRA heal with a distillation loss from one
  source model, for each source that has the host's tokenizer;
- random_control: the program itself with control "random" on every
  component, for one or more control seeds;
- best_existing: when the size limit is far above the base, the best
  existing model that fits the limits (atlas entries, then profiles.json);
- source: each source model of the program, as a reference.

This module is pure Python. It builds the arm programs and judges the
scores. composer/comparison.py runs the compiles and the scoring.
"""

import math

from composer import sizes
from composer.skills import SKILLS
from wsp.program import Program

DEFAULT_CONTROL_SEEDS = 2
DEFAULT_FAR_FACTOR = 2.0
DEFAULT_NOISE_FLOOR = 0.01
# Transfer on 128 held-out texts moves by about this much between seeds and text draws.
DEFAULT_PROBE_NOISE_FLOOR = 0.03
# Control seeds are far apart, so the random draws of the arms do not overlap.
CONTROL_SEED_STEP = 1000
# Earlier benchmark scores that name the best existing model at the size limit. Only the final report
# reads them, to pick a rival to score once. The composer never sees them (composer/tools.py hides them).
RIVAL_SCORE_NAMES = {
    "math": ["gsm8k", "gsm8k_raw", "math500", "math"],
    "code": ["humaneval", "mbpp", "code"],
    "medicine": ["medmcqa", "medqa", "medicine"],
    "knowledge": ["mmlu", "mmlu_dev", "mmlu_heldout", "knowledge"],
    "law": ["mmlu_law", "law"],
}
HEAL_KEYS_DROPPED_FOR_LORA = ["distill", "kd_weight", "temperature"]


def source_models(program_dict, records):
    """Every model the program takes components from, host first.

    records: {catalog id: record}. Block ids (block/<org>/<model>/a-b)
    name their model directly. Experts donors, insert donors, and a
    distillation teacher count too.
    """
    models = []

    def add(name):
        if name and name not in models:
            models.append(name)

    add(program_dict.get("host") or program_dict.get("base"))
    base_source = program_dict.get("base_source")
    if isinstance(base_source, dict):
        add(base_source.get("model"))
    splice = program_dict.get("splice")
    if isinstance(splice, dict):
        for key in splice.get("sources", {}):
            add(splice["sources"][key])
    for graft in program_dict.get("grafts", []):
        record = records.get(graft["id"])
        if record is None:
            record = sizes.block_record(graft["id"])
        if record is not None:
            add(record.get("model"))
    experts = program_dict.get("experts")
    if isinstance(experts, dict):
        for donor in experts.get("donors", []):
            add(donor)
    for insert in program_dict.get("inserts", []):
        add(insert.get("donor"))
    for tap in program_dict.get("taps", []):
        add(tap.get("donor"))
    for tv in program_dict.get("task_vectors", []):
        record = records.get(tv["id"])
        if record is not None:
            add(record.get("model"))
    heal = program_dict.get("heal")
    if isinstance(heal, dict) and heal.get("distill"):
        add(heal["distill"])
    return models


def default_heal_data(skills):
    """The heal data for a request with no heal of its own: one skill's data, else the even mix."""
    if skills is not None and len(skills) == 1:
        return SKILLS[list(skills)[0]]["heal_data"][-1]
    return "composer_mix"


def lora_heal_spec(program_dict, constraints, skills=None):
    """The heal spec of the LoRA arm.

    It trains on the program's heal data with the program's heal settings,
    for as many steps as all of the program's training stages together
    (heal, tap_train, and each stitch), so it gets the same training
    budget. A program without a heal falls back to its tap_train data, then
    to the request's skill data. A distillation loss in the program's heal
    is dropped: the distillation arm adds its own.
    """
    program = Program.from_dict(program_dict)
    total = 0
    for what, steps in sizes.training_steps(program):
        if steps is not None:
            total += steps
    heal = program_dict.get("heal")
    if heal is not None:
        spec = dict(heal)
    else:
        spec = {"steps": 300, "data": default_heal_data(skills), "lr": 3e-4, "batch_size": 8, "seq_len": 384}
        if program_dict.get("tap_train") is not None:
            spec["data"] = program_dict["tap_train"]["data"]
    for name in HEAL_KEYS_DROPPED_FOR_LORA:
        spec.pop(name, None)
    if total > 0:
        spec["steps"] = total
    else:
        cap = constraints.get("heal_steps_cap")
        if cap is not None and cap < spec["steps"]:
            spec["steps"] = cap
    return spec


def lora_spec_for(program_dict):
    """The LoRA for the comparison: rank 16, or the program's own LoRA when it has one."""
    lora = program_dict.get("lora")
    if lora is not None and (lora.get("rank") or lora.get("budget_params")):
        return dict(lora)
    return {"rank": 16, "budget_params": None, "targets": None, "alpha": None, "layers": None}


def heal_minutes_budget(program_dict):
    """A budget with only the program's heal time limit, or None. Size limits do not apply to the host."""
    budget = program_dict.get("budget")
    if not isinstance(budget, dict) or budget.get("max_heal_minutes") is None:
        return None
    return {"max_total_params_b": None, "max_active_params_b": None, "max_heal_minutes": budget["max_heal_minutes"]}


def lora_program(program_dict, constraints, skills=None):
    """The host-only LoRA program the final program is compared with."""
    program = {"host": program_dict["host"], "seed": program_dict.get("seed", 0), "masks": None,
               "task_vectors": [], "grafts": [], "heal": lora_heal_spec(program_dict, constraints, skills),
               "lora": lora_spec_for(program_dict)}
    budget = heal_minutes_budget(program_dict)
    if budget is not None:
        program["budget"] = budget
    return program


def distill_program(program_dict, constraints, teacher, skills=None):
    """The LoRA program with a distillation loss from one teacher, at the same heal budget."""
    program = lora_program(program_dict, constraints, skills)
    own = program_dict.get("heal") or {}
    program["heal"]["distill"] = teacher
    if own.get("distill") == teacher:
        program["heal"]["kd_weight"] = own.get("kd_weight", 1.0)
        program["heal"]["temperature"] = own.get("temperature", 1.0)
    else:
        program["heal"]["kd_weight"] = 1.0
        program["heal"]["temperature"] = 1.0
    return program


def control_program(program_dict, control_seed):
    """The program with control random on every component. Raise ValueError when it has none."""
    program = Program.from_dict(program_dict)
    return program.with_control("random", control_seed).to_dict()


def plan_arms(program_dict, request, models, records, sources=None):
    """Plan the compiled arms. Return (arms, skipped).

    arms: list of {"role", "label", "program", "teacher", "control_seed"}.
    skipped: list of {"arm", "reason"} for an arm that does not apply.
    models: {hf_id: size info} from composer.sources.Sources.model_table.
    """
    constraints = request.get("constraints") or {}
    skills = list(request["skills"])
    options = request.get("report") or {}
    arms = []
    skipped = []
    host = program_dict["host"]

    arms.append({"role": "lora_on_base", "label": "base + LoRA", "teacher": None, "control_seed": None,
                 "program": lora_program(program_dict, constraints, skills)})

    for name in source_models(program_dict, records)[1:]:
        match = sizes.same_tokenizer(models, host, name)
        if match is None:
            skipped.append({"arm": f"distill from {name}",
                            "reason": "the tokenizer family of the host or the source is unknown"})
        elif not match:
            skipped.append({"arm": f"distill from {name}", "reason": "the tokenizers differ"})
        else:
            arms.append({"role": "distill_on_base", "label": f"base + distill from {name}", "teacher": name,
                         "control_seed": None, "program": distill_program(program_dict, constraints, name, skills)})
    if len(source_models(program_dict, records)) == 1:
        skipped.append({"arm": "distill", "reason": "the program has no source model besides its host"})

    count = options.get("control_seeds", DEFAULT_CONTROL_SEEDS)
    try:
        for index in range(count):
            seed = CONTROL_SEED_STEP * (index + 1)
            arms.append({"role": "random_control", "label": f"program with random components (seed {seed})",
                         "teacher": None, "control_seed": seed, "program": control_program(program_dict, seed)})
    except ValueError as error:
        skipped.append({"arm": "random control", "reason": str(error)})
    return arms, skipped


# The best existing model at the size limit.

def existing_skill_score(sources, model, skill):
    """A model's earlier score for a skill from the atlas entries, then profiles.json. None when it has none."""
    for entry in sources.atlas_entries:
        name = entry.get("hf_id") or entry.get("model") or entry.get("id")
        if name != model:
            continue
        value = (entry.get("skills") or {}).get(skill)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value), "atlas"
    for profile in sources.profiles:
        if profile.get("hf_id") != model:
            continue
        scores = profile.get("scores") or {}
        for name in [skill] + SKILLS[skill]["tags"]:
            value = scores.get(name)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                return float(value), f"profiles.json:{name}"
    return None


def is_far_above(constraints, host_info, factor):
    cap = constraints.get("max_total_params_b")
    if cap is None or host_info is None or not host_info.get("params_b"):
        return False
    return cap >= factor * host_info["params_b"]


def best_existing_model(request, sources, host):
    """The best existing model that fits the limits, when the size limit is far above the host.

    Return (row or None, reason). The row is {"model", "score", "params_b",
    "active_params_b", "basis"}. The score is the weighted mean of the
    earlier skill scores, a missing skill as 0. Only the report uses these
    scores. The composer never sees them.
    """
    constraints = request.get("constraints") or {}
    options = request.get("report") or {}
    factor = options.get("far_factor", DEFAULT_FAR_FACTOR)
    host_info = sources.models.get(host)
    if not is_far_above(constraints, host_info, factor):
        return None, (f"the size limit is not {factor} times the host ({host_info and host_info.get('params_b')}b), "
                      f"so the best existing model is not a fair rival")
    weights = request["skills"]
    best = None
    for model in sorted(sources.models):
        info = sources.models[model]
        if model == host:
            continue
        total = info.get("params_b")
        if total is None or total > constraints["max_total_params_b"]:
            continue
        cap_run = constraints.get("max_run_params_b")
        active = info.get("active_params_b") or total
        if cap_run is not None and active > cap_run:
            continue
        scores = {}
        basis = {}
        for skill in weights:
            found = existing_skill_score(sources, model, skill)
            if found is not None:
                scores[skill] = found[0]
                basis[skill] = found[1]
        if len(scores) == 0:
            continue
        weight_sum = 0.0
        value = 0.0
        for skill in weights:
            weight_sum += weights[skill]
            value += weights[skill] * scores.get(skill, 0.0)
        value = value / weight_sum if weight_sum > 0 else 0.0
        row = {"model": model, "score": round(value, 4), "params_b": total, "active_params_b": active,
               "basis": basis}
        if best is None or (row["score"], -row["params_b"]) > (best["score"], -best["params_b"]):
            best = row
    if best is None:
        return None, "no model within the limits has an earlier score for a requested skill"
    return best, None


# Noise and verdicts.

def set_se(score, items):
    if not items or score is None:
        return 0.0
    return math.sqrt(max(0.0, score * (1.0 - score)) / items)


def skill_se(row, skill):
    """The item sampling error of a skill's held-out score (the mean of its sets)."""
    sets = row.get("sets") or {}
    parts = []
    for name in SKILLS[skill]["heldout"]:
        item = sets.get(name)
        if isinstance(item, dict) and item.get("score") is not None:
            parts.append(set_se(item["score"], item.get("items")))
    if len(parts) == 0:
        return 0.0
    return math.sqrt(sum(part * part for part in parts)) / len(parts)


def objective_se(row, weights):
    """The item sampling error of a row's objective, from its held-out sets."""
    total = sum(weights.values())
    if total <= 0:
        return 0.0
    squares = 0.0
    for skill in weights:
        share = weights[skill] / total
        error = skill_se(row, skill)
        squares += share * share * error * error
    return math.sqrt(squares)


def judge(name, final_row, arm_rows, weights, noise_floor=DEFAULT_NOISE_FLOOR):
    """Say whether the program beat an arm by more than noise.

    arm_rows: the scored rows of the arm, with objective (one per seed).
    The noise threshold is the largest of the floor, twice the item sampling
    error of the difference, and the spread of the arm's own seeds.
    """
    scored = []
    for row in arm_rows:
        if isinstance(row, dict) and row.get("objective") is not None and not row.get("error"):
            scored.append(row)
    if final_row is None or final_row.get("objective") is None or len(scored) == 0:
        return {"arm": name, "verdict": "not scored", "text": f"{name}: not scored, so there is no verdict."}
    arm_objectives = []
    for row in scored:
        arm_objectives.append(row["objective"])
    arm_value = sum(arm_objectives) / len(arm_objectives)
    spread = max(arm_objectives) - min(arm_objectives)
    se_final = objective_se(final_row, weights)
    se_arm = math.sqrt(sum(objective_se(row, weights) ** 2 for row in scored)) / len(scored)
    sampling = 2.0 * math.sqrt(se_final ** 2 + se_arm ** 2)
    threshold = max(noise_floor, sampling, spread)
    diff = final_row["objective"] - arm_value
    if diff > threshold:
        verdict = "beat"
        phrase = "so the program beat it by more than noise"
    elif diff < -threshold:
        verdict = "lost"
        phrase = "so the program scored below it by more than noise"
    else:
        verdict = "tied"
        phrase = "so the program did not beat it by more than noise"
    text = (f"{name}: the program scored {final_row['objective']:.3f} and the arm {arm_value:.3f} "
            f"({len(scored)} run{'s' if len(scored) != 1 else ''}). The difference {diff:+.3f} against a noise "
            f"threshold of {threshold:.3f}, {phrase}.")
    return {"arm": name, "verdict": verdict, "program": round(final_row["objective"], 4),
            "arm_objective": round(arm_value, 4), "diff": round(diff, 4), "threshold": round(threshold, 4),
            "sampling_threshold": round(sampling, 4), "seed_spread": round(spread, 4), "runs": len(scored),
            "text": text}


def judge_probe(name, final_row, arm_rows, floor=DEFAULT_PROBE_NOISE_FLOOR):
    """Say whether the program beat an arm on the probe objective by more than noise.

    Rows carry probe_objective. The noise threshold is the larger of the floor and the spread of the
    arm's own seeds. The probe has no item sampling error to add, so the floor stands for it.
    """
    values = []
    for row in arm_rows:
        if isinstance(row, dict) and row.get("probe_objective") is not None and not row.get("error"):
            values.append(row["probe_objective"])
    if final_row is None or final_row.get("probe_objective") is None or len(values) == 0:
        return {"arm": name, "basis": "probe", "verdict": "not scored",
                "text": f"{name}: no probe numbers, so there is no probe verdict."}
    arm_value = sum(values) / len(values)
    spread = max(values) - min(values)
    threshold = max(floor, spread)
    diff = final_row["probe_objective"] - arm_value
    if diff > threshold:
        verdict = "beat"
        phrase = "so the program beat it by more than noise"
    elif diff < -threshold:
        verdict = "lost"
        phrase = "so the program scored below it by more than noise"
    else:
        verdict = "tied"
        phrase = "so the program did not beat it by more than noise"
    text = (f"{name} (probe): the program scored {final_row['probe_objective']:.3f} and the arm {arm_value:.3f} "
            f"({len(values)} run{'s' if len(values) != 1 else ''}). The difference {diff:+.3f} against a noise "
            f"threshold of {threshold:.3f}, {phrase}.")
    return {"arm": name, "basis": "probe", "verdict": verdict, "program": round(final_row["probe_objective"], 4),
            "arm_objective": round(arm_value, 4), "diff": round(diff, 4), "threshold": round(threshold, 4),
            "seed_spread": round(spread, 4), "runs": len(values), "text": text}
