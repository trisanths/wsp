"""Build the comparison list and the results file for a finished composer run.

results/composer_<request>.json holds the held-out scores of the final
program, its size, the budget it used, and the same held-out scores for
the base alone, each source model the program used, and a LoRA on the
base trained with the same heal data and steps.
"""

import time

from composer import sizes
from composer.skills import SET_NOTES, SKILLS, heldout_sets, objective
from wsp.program import Program


def source_models(program_dict, records):
    """Every model the program takes components from, host first.

    records: {catalog id: record}. Block ids (block/<org>/<model>/a-b)
    name their model directly.
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
    for tap in program_dict.get("taps", []):
        add(tap.get("donor"))
    for tv in program_dict.get("task_vectors", []):
        record = records.get(tv["id"])
        if record is not None:
            add(record.get("model"))
    return models


def heldout_jobs(final, program_dict, records, lora_dir):
    """Return a list of (role, model_ref) to score on the held-out sets."""
    jobs = [("final", final["model_dir"])]
    models = source_models(program_dict, records)
    if len(models) > 0:
        jobs.append(("base", models[0]))
    for name in models[1:]:
        jobs.append(("source", name))
    if lora_dir:
        jobs.append(("lora_on_base", lora_dir))
    return jobs


def default_heal_data(skills):
    """The heal data for a request with no heal of its own: one skill's data, else the even mix."""
    if skills is not None and len(skills) == 1:
        return SKILLS[list(skills)[0]]["heal_data"][-1]
    return "composer_mix"


def lora_heal_spec(program_dict, constraints, skills=None):
    """The heal spec for the LoRA comparison.

    The LoRA trains on the program's heal data with the program's heal
    settings, for as many steps as all of the program's training stages
    together (heal, tap_train, and each stitch), so it gets the same
    training budget. A program without a heal falls back to its tap_train
    data, then to the request's skill data.
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


def lora_program(program_dict, constraints, skills=None):
    """The host-only LoRA program the final program is compared with."""
    return {"host": program_dict["host"], "seed": program_dict.get("seed", 0), "masks": None,
            "task_vectors": [], "grafts": [], "heal": lora_heal_spec(program_dict, constraints, skills),
            "lora": lora_spec_for(program_dict)}


def heldout_notes(program_dict, records):
    """Notes for the parts the program uses that were mined on held-out items."""
    notes = []
    if program_dict is None:
        return notes
    for component_id in Program.from_dict(program_dict).component_ids():
        record = records.get(component_id)
        if record is not None and record.get("heldout_overlap"):
            notes.append({"id": component_id, "note": record["heldout_overlap"]})
    return notes


def fits(row, constraints):
    """True or False when the row's measured size can be checked against the total cap, else None."""
    cap = (constraints or {}).get("max_total_params_b")
    total = row.get("params_total_b")
    if cap is None or total is None:
        return None
    return total <= cap


def build_results(request, summary, scored, lora_info=None, records=None):
    """scored: list of (role, model_ref, result dict or error text). records: {catalog id: record}."""
    final = summary["final"] or {}
    weights = request["skills"]
    out = {
        "request": request["name"],
        "description": request.get("description"),
        "skills": weights,
        "constraints": request.get("constraints"),
        "date": time.strftime("%Y-%m-%d %H:%M"),
        "program_text": final.get("program_text"),
        "program": final.get("program"),
        "rationale": final.get("rationale"),
        "fallback": final.get("fallback"),
        "no_model": final.get("no_model"),
        "budget": {
            "compiles_used": summary.get("compile_budget_used"),
            "compile_budget": summary.get("compile_budget"),
            "turns": summary.get("turns"),
            "tokens_used": summary.get("tokens_used"),
            "stop_reason": summary.get("stop_reason"),
        },
        "practice": final.get("practice"),
        "heldout_sets": {},
        "final": None,
        "comparisons": {"base": None, "sources": [], "lora_on_base": None},
        "lora_compile": lora_info,
        "archive_dir": summary.get("archive_dir"),
        "heldout_overlap": heldout_notes(final.get("program"), records or {}),
    }
    for name in heldout_sets(list(weights)):
        out["heldout_sets"][name] = SET_NOTES.get(name)
    if final.get("practice"):
        practice = final["practice"]
        out["params"] = {"total_b": practice.get("params_total_b"), "run_b": practice.get("params_run_b"),
                         "decode_tok_s": practice.get("decode_tok_s")}
    for role, model_ref, result in scored:
        if isinstance(result, dict):
            row = dict(result)
            if row.get("skills") is not None:
                row["objective"] = objective(row["skills"], weights)
        else:
            row = {"error": str(result)[:2000]}
        # A comparison over the size cap (a carve's source, a big donor) is not a rival under the request.
        row["fits_constraints"] = fits(row, request.get("constraints"))
        row["role"] = role
        row["model"] = model_ref
        if role == "final":
            out["final"] = row
        elif role == "base":
            out["comparisons"]["base"] = row
        elif role == "source":
            out["comparisons"]["sources"].append(row)
        else:
            out["comparisons"]["lora_on_base"] = row
    return out
