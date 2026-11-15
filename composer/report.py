"""Build the results file for a finished composer run.

results/composer_<request>.json holds, for the final program and for each
comparison arm (composer/arms.py: the base, the base with a LoRA, the base
with a distillation heal from each source with the host's tokenizer, the
program with random components, the best existing model at the size limit
when it is far above the base, and each source model), two kinds of
numbers:

- probe numbers (wsp/probe.py on held-out pool text): transfer per skill,
  drift, and the probe objective. These are the benchmark-free signal.
- benchmark numbers: the held-out sets, scored once at the end.

It also holds a verdict per arm for each kind, and a plain summary of which
arms the program beat by more than noise.
"""

import time

from composer import arms
from composer.arms import (default_heal_data, lora_heal_spec, lora_program, lora_spec_for,  # noqa: F401
                           source_models)
from composer.skills import DEFAULT_DRIFT_TOLERANCE, SET_NOTES, SKILLS, heldout_sets, objective, probe_summary
from wsp.program import Program


def heldout_jobs(final, program_dict, records, lora_dir):
    """Return a list of (role, model_ref) to score on the held-out sets (final, base, sources, LoRA)."""
    jobs = [("final", final["model_dir"])]
    models = source_models(program_dict, records)
    if len(models) > 0:
        jobs.append(("base", models[0]))
    for name in models[1:]:
        jobs.append(("source", name))
    if lora_dir:
        jobs.append(("lora_on_base", lora_dir))
    return jobs


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


def missing_sets(row, weights):
    """The held-out sets of the request's skills that have no score in the row.

    A row that lists no sets cannot be checked by set. Its skills still can: a skill with no score is a failure.
    """
    sets = row.get("sets")
    missing = []
    skills = row.get("skills") or {}
    for skill in weights:
        if skills.get(skill) is None:
            missing += [name for name in SKILLS[skill]["heldout"] if name not in missing]
    if not isinstance(sets, dict) or len(sets) == 0:
        return missing
    for name in heldout_sets(list(weights)):
        item = sets.get(name)
        if not isinstance(item, dict) or item.get("score") is None:
            missing.append(name)
    return missing


def build_results(request, summary, scored, lora_info=None, records=None, plan=None):
    """scored: list of (role, model_ref, result dict or error text) or the same with a fourth item,
    an info dict for an arm ({"label", "teacher", "control_seed"}). records: {catalog id: record}.
    plan: {"skipped": [...], "arm_compiles": {...}, "best_existing": {...}, "heal": {...}}.
    """
    final = summary["final"] or {}
    weights = request["skills"]
    plan = plan or {}
    probes = plan.get("probes") or {}
    tolerance = request.get("drift_tolerance")
    if tolerance is None:
        tolerance = DEFAULT_DRIFT_TOLERANCE
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
        "heldout_items": request.get("heldout_items"),
        "drift_tolerance": tolerance,
        "benchmarks_scored": plan.get("benchmark_models"),
        "final": None,
        "comparisons": {"base": None, "sources": [], "lora_on_base": None, "distill_on_base": [],
                        "random_control": [], "best_existing": None},
        "skipped_arms": plan.get("skipped", []),
        "best_existing_choice": plan.get("best_existing"),
        "same_heal": plan.get("heal"),
        "lora_compile": lora_info,
        "arm_compiles": plan.get("arm_compiles"),
        "verdicts": [],
        "archive_dir": summary.get("archive_dir"),
        "heldout_overlap": heldout_notes(final.get("program"), records or {}),
    }
    for name in heldout_sets(list(weights)):
        out["heldout_sets"][name] = SET_NOTES.get(name)
    if final.get("practice"):
        practice = final["practice"]
        out["params"] = {"total_b": practice.get("params_total_b"), "run_b": practice.get("params_run_b"),
                         "decode_tok_s": practice.get("decode_tok_s")}
    for item in scored:
        role, model_ref, result = item[0], item[1], item[2]
        info = item[3] if len(item) > 3 else {}
        if isinstance(result, dict):
            row = dict(result)
            if row.get("skills") is not None:
                row["objective"] = objective(row["skills"], weights)
                missing = missing_sets(row, weights)
                if len(missing) > 0:
                    # A set that did not score would change what the objective measures, so the row has none.
                    row["objective"] = None
                    row["missing_sets"] = missing
        else:
            row = {"error": str(result)[:2000]}
        # A comparison over the size cap (a carve's source, a big donor) is not a rival under the request.
        row["fits_constraints"] = fits(row, request.get("constraints"))
        probe = probes.get(model_ref)
        if isinstance(probe, dict) and isinstance(probe.get("skills"), dict):
            row["probe"] = probe
            row["probe_summary"] = probe_summary(probe, weights, tolerance)
            row["probe_objective"] = row["probe_summary"]["objective"]
        elif probe is not None:
            row["probe_error"] = str(probe)[:2000]
        row["role"] = role
        row["model"] = model_ref
        for key in info or {}:
            row[key] = info[key]
        if role == "final":
            out["final"] = row
        elif role == "base":
            out["comparisons"]["base"] = row
        elif role == "source":
            out["comparisons"]["sources"].append(row)
        elif role in ("distill_on_base", "random_control"):
            out["comparisons"][role].append(row)
        else:
            out["comparisons"][role] = row
    out["verdicts"] = verdicts(out, weights, request)
    out["content_gain"] = content_gain(out)
    out["summary"] = summary_text(out)
    return out


def arm_rows(out):
    """Every compared row of a results dict as (arm name, [rows])."""
    comparisons = out["comparisons"]
    named = []
    for name, key in [("base", "base"), ("base + LoRA", "lora_on_base"), ("best existing model", "best_existing")]:
        if comparisons.get(key) is not None:
            named.append((name, [comparisons[key]]))
    for row in comparisons["distill_on_base"]:
        named.append((row.get("label") or "base + distill", [row]))
    if len(comparisons["random_control"]) > 0:
        named.append(("random components of the same size", comparisons["random_control"]))
    return named


def content_gain(out):
    """Per skill, the final transfer minus the mean transfer of the random-components arms. None when missing."""
    final = out.get("final") or {}
    final_skills = ((final.get("probe") or {}).get("skills")) or {}
    controls = []
    for row in out["comparisons"]["random_control"]:
        if isinstance(row.get("probe"), dict):
            controls.append(row["probe"]["skills"])
    gains = {}
    for skill in out["skills"]:
        real = (final_skills.get(skill) or {}).get("transfer")
        values = []
        for table in controls:
            value = (table.get(skill) or {}).get("transfer")
            if value is not None:
                values.append(value)
        if real is None or len(values) == 0:
            gains[skill] = None
        else:
            gains[skill] = real - sum(values) / len(values)
    return gains


def summary_text(out):
    """One plain paragraph: which arms the program beat by more than noise, on each kind of number."""
    beat = {"benchmark": [], "probe": []}
    not_beat = {"benchmark": [], "probe": []}
    for item in out["verdicts"]:
        basis = item.get("basis", "benchmark")
        if item["verdict"] == "beat":
            beat[basis].append(item["arm"])
        elif item["verdict"] in ("tied", "lost"):
            not_beat[basis].append(item["arm"])
    lines = []
    for basis in ["benchmark", "probe"]:
        if len(beat[basis]) == 0:
            beaten = "no arm"
        else:
            beaten = ", ".join(beat[basis])
        line = f"On the {basis} numbers the program beat {beaten} by more than noise."
        if len(not_beat[basis]) > 0:
            line += " It did not beat " + ", ".join(not_beat[basis]) + "."
        lines.append(line)
    return " ".join(lines)


def verdicts(out, weights, request):
    """One verdict per arm: did the program beat it by more than noise."""
    final = out.get("final")
    comparisons = out["comparisons"]
    floor = (request.get("report") or {}).get("noise_floor", arms.DEFAULT_NOISE_FLOOR)
    items = []
    probe_floor = (request.get("report") or {}).get("probe_noise_floor", arms.DEFAULT_PROBE_NOISE_FLOOR)
    for name, rows in arm_rows(out):
        item = arms.judge(name, final, rows, weights, floor)
        item["basis"] = "benchmark"
        items.append(item)
    for name, rows in arm_rows(out):
        if any(row.get("probe_objective") is not None for row in rows):
            items.append(arms.judge_probe(name, final, rows, probe_floor))
    for item in out.get("skipped_arms") or []:
        items.append({"arm": item["arm"], "verdict": "skipped", "text": f"{item['arm']}: skipped. {item['reason']}"})
    return items
