"""Run the comparison for a finished composer run.

run_comparison plans the arms (composer/arms.py), compiles them, scores the
final program, the base, the sources, the arms, and the best existing
model on the held-out sets, and builds the results (composer/report.py).

The compile and scoring steps are passed in, so the same code runs on
Modal (modal_composer.py) and in CPU tests with fakes:

    compile_many([(arm_program_dict, payload), ...]) -> [dict or error, ...]
    score_many([(model_ref, payload), ...])          -> [dict or error, ...]
    probe_many([(model_ref, payload), ...])          -> [probe report or error, ...]

score_many scores the benchmarks and is called once, with each distinct
model once. probe_many runs the benchmark-free probe (wsp/probe.py) on
the same models. It is optional.

A compile result is {"ok": bool, "model_dir": str, ...}. A score result is
what modal_composer.score_heldout returns. Either may be an Exception.
"""

from composer import arms as arms_module
from composer import sizes
from composer.report import build_results
from composer.skills import DEFAULT_DRIFT_TOLERANCE, PROBE_COUNT, heldout_sets
from composer import probes as probes_module
from composer.tools import program_hash
from wsp.program import Program


def arm_payload(request, final, arm, records):
    """What the Modal compile of one arm needs."""
    program = Program.from_dict(arm["program"])
    catalog = []
    seen = set()
    for component_id in program.component_ids():
        if component_id in seen:
            continue
        seen.add(component_id)
        # A block id (block/<org>/<model>/<first>-<last>) has no catalog row: build its record from the id,
        # as the practice compile does.
        record = records.get(component_id)
        if record is None:
            record = sizes.block_record(component_id)
        if record is not None:
            catalog.append(record)
    return {"request": request["name"], "program_hash": final["program_hash"],
            "arm": arm["role"] + ("_" + str(arm["control_seed"]) if arm["control_seed"] is not None else "") +
                   ("_" + arm["teacher"].replace("/", "_") if arm["teacher"] else ""),
            "constraints": request["constraints"], "skills": list(request["skills"]), "catalog": catalog,
            "arm_hash": program_hash(arm["program"])}


def error_text(result):
    if isinstance(result, Exception):
        return f"{type(result).__name__}: {result}"
    return str(result)


def run_comparison(request, summary, sources, compile_many, score_many, probe_many=None):
    """Return the results dict for a finished run. summary comes from composer.agent.run_loop."""
    final = summary["final"] or {}
    records = sources.records_by_id()
    plan = {"skipped": [], "arm_compiles": {}, "best_existing": None, "heal": None}
    scored = []
    if final.get("no_model"):
        return build_results(request, summary, scored, None, records, plan)

    program_dict = final["program"]
    planned, skipped = arms_module.plan_arms(program_dict, request, sources.models, records, sources)
    plan["skipped"] = skipped
    lora = lora_arm(planned)
    plan["heal"] = {"data": lora["program"]["heal"]["data"], "steps": lora["program"]["heal"]["steps"],
                    "lr": lora["program"]["heal"]["lr"], "budget": program_dict.get("budget")}

    # Compile the arms in parallel.
    calls = []
    for arm in planned:
        calls.append((arm["program"], arm_payload(request, final, arm, records)))
    compiled = compile_many(calls)
    jobs = [("final", final["model_dir"], {})]
    models = arms_module.source_models(program_dict, records)
    jobs.append(("base", models[0], {}))
    for name in models[1:]:
        jobs.append(("source", name, {}))
    for index in range(len(planned)):
        arm = planned[index]
        result = compiled[index]
        info = {"label": arm["label"], "teacher": arm["teacher"], "control_seed": arm["control_seed"]}
        if isinstance(result, dict):
            plan["arm_compiles"][arm["label"]] = {key: result.get(key) for key in
                                                   ("ok", "model_dir", "seconds", "heal_steps_run",
                                                    "heal_seconds", "error")}
        else:
            plan["arm_compiles"][arm["label"]] = {"ok": False, "error": error_text(result)}
        if isinstance(result, dict) and result.get("ok") and result.get("model_dir"):
            jobs.append((arm["role"], result["model_dir"], info))
        else:
            reason = result.get("error") if isinstance(result, dict) else error_text(result)
            scored.append((arm["role"], None, "compile failed: " + str(reason), info))

    best, reason = arms_module.best_existing_model(request, sources, models[0])
    if best is not None:
        plan["best_existing"] = best
        jobs.append(("best_existing", best["model"], {"label": "best existing model", "atlas_score": best["score"],
                                                        "atlas_basis": best["basis"]}))
    else:
        plan["skipped"].append({"arm": "best existing model", "reason": reason})

    # Score each distinct model once. A role that shares a model reuses its row.
    refs = []
    for role, ref, info in jobs:
        if ref not in refs:
            refs.append(ref)
    plan["benchmark_models"] = list(refs)
    if probe_many is not None:
        plan["probes"] = run_probes(request, program_dict, sources, records, refs, models[0], probe_many)
        plan["probe_plan"] = probe_plan_for(request, program_dict, sources, records)
    score_payload = {"skills": list(request["skills"]), "heldout_sets": heldout_sets(list(request["skills"])),
                     "heldout_items": request.get("heldout_items")}
    results = score_many([(ref, score_payload) for ref in refs])
    by_ref = {}
    for index in range(len(refs)):
        by_ref[refs[index]] = results[index]
    for role, ref, info in jobs:
        result = by_ref[ref]
        if not isinstance(result, dict):
            result = error_text(result)
        scored.append((role, ref, result, info))
    return build_results(request, summary, scored, plan["arm_compiles"].get("base + LoRA"), records, plan)


def probe_plan_for(request, program_dict, sources, records):
    """The per-skill donors and probe paths of the final program."""
    program = Program.from_dict(program_dict)
    full = dict(records)
    for component_id in program.component_ids():
        if component_id not in full:
            record = sizes.block_record(component_id)
            if record is not None:
                full[component_id] = record
    return probes_module.probe_plan(program_dict, list(request["skills"]), full, sources.models)


def run_probes(request, program_dict, sources, records, refs, host, probe_many):
    """Probe every distinct model once against the final program's donors. Return {model ref: probe or error text}.

    The host is the reference, so its probe is zero transfer and zero drift by construction. A model with
    another tokenizer than the host cannot be compared token by token, so it has no probe.
    """
    plan = probe_plan_for(request, program_dict, sources, records)
    skills = list(request["skills"])
    tolerance = request.get("drift_tolerance")
    if tolerance is None:
        tolerance = DEFAULT_DRIFT_TOLERANCE
    payload = {"request": request["name"], "host": host, "skills": skills, "probe_plan": plan,
               "probe_items": request.get("practice_items") or PROBE_COUNT, "drift_tolerance": tolerance}
    out = {}
    calls = []
    call_refs = []
    for ref in refs:
        if ref == host:
            rows = {}
            for skill in skills:
                rows[skill] = {"transfer": 0.0, "note": "the base is the reference"}
            out[ref] = {"skills": rows, "drift": {"base_kl": 0.0}}
        elif ref.startswith("/vol/") or sizes.same_tokenizer(sources.models, host, ref) is True:
            calls.append((ref, payload))
            call_refs.append(ref)
        else:
            out[ref] = "no probe: the tokenizer differs from the host's, so next-token KL against the base is not defined"
    if len(calls) > 0:
        results = probe_many(calls)
        for index in range(len(call_refs)):
            result = results[index]
            if not isinstance(result, dict):
                result = error_text(result)
            out[call_refs[index]] = result
    return out


def lora_arm(planned):
    for arm in planned:
        if arm["role"] == "lora_on_base":
            return arm
    raise ValueError("the plan has no LoRA arm")
