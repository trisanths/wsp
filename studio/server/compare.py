"""The honest comparison set for a result, and the verdict that follows from it.

A built model is compared with five references:

  base         the host alone
  lora         the host with a LoRA on the same data
  distill      the host healed by distillation from the same source, at the same heal budget
  random       the same build with random components of the same size
  best_model   the best existing model that fits the size limit (shown only when the limit is far above the host)

A result is a success only when it beats every reference by CLEAR_MARGIN, and
every reference is measured. A result that does not beat a measured
reference is a failed attempt. A reference with no measurement is listed as
not measured, and the verdict is "unproven" until it is measured.
"""

from studio.server import library

CLEAR_MARGIN = 0.02
FAR_ABOVE_BASE = 2.0
REQUIRED = ["base", "lora", "distill", "random"]

LABELS = {
    "base": "Base",
    "lora": "Base + LoRA",
    "distill": "Base + distillation (same budget)",
    "random": "Random components (same size)",
    "best_model": "Best existing model at this size",
    "source": "Original model (before removal)",
    "general": "General-text control (keeps what general text uses)",
}

# Library score keys per task. They all use the search-split items. The scorers differ a little (harness
# or raw answers, round 7 or round 15 runs), so the best model is the highest score over every key.
# GSM8K has no keys: the library scores were cut at 256 tokens by the old scorer, while the host and the
# builds use the team protocol. Add keys back once the library models are re-scored with skills_eval.
SEARCH_KEYS = {
    "gsm8k": [],
    "mmlu": ["mmlu_dev_round15", "mmlu_dev_round8", "mmlu_dev_round7", "mmlu_dev"],
}


def targets_of(spec):
    ids = []
    for item in spec["benchmarks"]:
        if item["kind"] == "target":
            ids.append(item["id"])
    return ids


def mean_over(targets, values):
    """The mean of values over the targets, or None when any target has no value."""
    if values is None or len(targets) == 0:
        return None
    total = 0.0
    for task in targets:
        if values.get(task) is None:
            return None
        total += float(values[task])
    return total / len(targets)


def spec_values(spec, field):
    """{task: value} of one field (base or lora) over the spec's targets, skipping missing ones."""
    out = {}
    for item in spec["benchmarks"]:
        if item["kind"] == "target" and item.get(field) is not None:
            out[item["id"]] = float(item[field])
    return out


def best_model_for(task, limit_b, models):
    """The best library model within the size limit for a task: (score, model id, score key, split) or None.

    The bar is the highest score over every model and every score key for the task. A model that was scored
    by only one of the scorers must not hide a higher score from another.
    """
    best = None
    for hf_id in sorted(models):
        info = models[hf_id]
        if info["params_b"] > limit_b:
            continue
        scores = info.get("scores") or {}
        for key in SEARCH_KEYS.get(task, []):
            score = scores.get(key)
            if score is None:
                continue
            if best is None or score["value"] > best[0]:
                best = (float(score["value"]), hf_id, key, score.get("split"))
    return best


def best_model_row(spec, limit_b, models):
    """Values and note for the best existing model over all targets. Models differ per task, so the note names each."""
    targets = targets_of(spec)
    per_task = {}
    notes = []
    awaiting = []
    for task in targets:
        found = best_model_for(task, limit_b, models)
        if found is None:
            if len(SEARCH_KEYS.get(task, [])) == 0 and task == "gsm8k":
                awaiting.append(task)
            continue
        per_task[task] = found[0]
        notes.append(task + ": " + found[1] + " (" + library.short_number(found[0]) + ", " + found[2] + ")")
    note = "; ".join(notes)
    if len(notes) == 0:
        note = "no library model with a score on this benchmark fits the limit"
    if len(awaiting) > 0:
        text = "GSM8K library scores are awaiting re-scoring with the current scorer"
        if len(notes) == 0:
            note = text
        else:
            note = note + "; " + text
    return per_task, note


def stub_distill(spec):
    """Fake distillation references for the stub backend, labeled as stub data in the row."""
    out = {}
    for item in spec["benchmarks"]:
        if item["kind"] != "target":
            continue
        if item.get("lora") is not None:
            out[item["id"]] = max(0.0, float(item["lora"]) - 0.02)
        elif item.get("base") is not None:
            out[item["id"]] = float(item["base"]) + 0.05
    return out


def lora_note(spec, stub):
    host = str(spec.get("host") or "the host").split("/")[-1]
    text = "LoRA rank 16 to 64 on " + host + ", from the results log (search split)"
    if stub:
        text += ". The base next to it is stub data."
    return text


def make_row(row_id, per_task, targets, split, note, stub):
    value = mean_over(targets, per_task)
    return {"id": row_id, "label": LABELS[row_id], "split": split, "per_task": dict(per_task or {}),
            "value": value, "measured": value is not None, "note": note, "stub": stub, "result_value": None,
            "delta": None, "beaten": None, "within_margin": None}


def result_on(split, scores, heldout, targets):
    if split == "heldout":
        return mean_over(targets, heldout)
    return mean_over(targets, scores)


def verdict_of(rows, result_value):
    """(verdict, beaten ids, not beaten ids, unmeasured ids) from rows that carry result_value and beaten."""
    beaten = []
    not_beaten = []
    unmeasured = []
    for row in rows:
        if not row["measured"]:
            unmeasured.append(row["id"])
        elif row["beaten"] is True:
            beaten.append(row["id"])
        elif row["beaten"] is False:
            not_beaten.append(row["id"])
        else:
            unmeasured.append(row["id"])
    if result_value is None:
        return "none", beaten, not_beaten, unmeasured
    if len(not_beaten) > 0:
        return "failed", beaten, not_beaten, unmeasured
    if len(unmeasured) > 0:
        return "unproven", beaten, not_beaten, unmeasured
    return "beats", beaten, not_beaten, unmeasured


def points(value):
    return "%.1f" % (value * 100.0)


def mark_margin(row):
    """Set within_margin: true when the result is above the reference but by less than the margin."""
    if row["delta"] is None or row["beaten"] is None:
        row["within_margin"] = None
    else:
        row["within_margin"] = row["delta"] > 0 and not row["beaten"]


def margin_points():
    return "%g" % round(CLEAR_MARGIN * 100.0, 2)


def failed_text(rows_by_id, not_beaten, random_reason):
    """The failed sentence, true to the numbers: within the margin, level, or below, for each reference."""
    parts = []
    for row_id in not_beaten:
        row = rows_by_id[row_id]
        numbers = " (" + points(row["result_value"]) + " against " + points(row["value"]) + ")"
        if row["within_margin"]:
            parts.append("is within " + margin_points() + " points of " + row["label"] + numbers)
        elif row["delta"] is not None and row["delta"] == 0:
            parts.append("only ties " + row["label"] + numbers)
        else:
            parts.append("is below " + row["label"] + numbers)
    text = "Failed attempt. It " + " and ".join(parts)
    within = [row_id for row_id in not_beaten if rows_by_id[row_id]["within_margin"]]
    if "random" in within:
        text += ", so " + random_reason + " not shown to matter"
    elif len(within) > 0:
        text += ", so it is not shown to be better"
    return text + "."


def headline(verdict, rows, beaten, not_beaten, unmeasured):
    by_id = {}
    for row in rows:
        by_id[row["id"]] = row
    if verdict == "none":
        return "No score yet."
    if verdict == "failed":
        return failed_text(by_id, not_beaten, "the parts are")
    if verdict == "unproven":
        names = [by_id[row_id]["label"] for row_id in unmeasured]
        text = "Not proven. "
        if len(beaten) > 0:
            text += "It beats " + ", ".join(by_id[row_id]["label"] for row_id in beaten) + ". "
        return text + "Not measured yet: " + ", ".join(names) + "."
    return "Beats every reference by at least " + points(CLEAR_MARGIN) + " points."


BAD_COMPONENT_VERDICTS = ["no better than random", "harmful"]


def no_component_contributes(components):
    """True when some component rests on a practice score and every such component is no better than random
    or harmful. A verdict from held-out text loss alone says nothing about the task, so it is not counted."""
    found = False
    for item in components or []:
        if item.get("evidence") != "practice":
            continue
        found = True
        if item.get("verdict") not in BAD_COMPONENT_VERDICTS:
            return False
    return found


def components_sentence(components):
    """One plain sentence that names each component and its verdict."""
    parts = []
    components = [item for item in components if item.get("evidence") == "practice"]
    for item in components:
        parts.append(str(item.get("label") or item.get("id")) + " is " + str(item["verdict"]))
    if len(parts) == 1:
        return "Its only component " + parts[0] + "."
    return "No component contributes: " + ", ".join(parts) + "."


def apply_component_rule(verdict, headline_text, components):
    """A result whose components are all no better than random or harmful cannot be a success.

    Returns (verdict, headline). A verdict of none stays none. A failed verdict keeps its headline and gains
    the sentence about the components.
    """
    if verdict == "none" or not no_component_contributes(components):
        return verdict, headline_text
    sentence = components_sentence(components)
    if verdict == "failed":
        return verdict, headline_text + " " + sentence
    return "failed", "Failed attempt. " + sentence + " The score gain is not shown to come from the parts."


def build(spec, scores, params_b=None, heldout=None, control=None, control_split="search", stub=False,
          models=None, components=None):
    """The comparison of a result with the reference set.

    scores: the result's search-split scores {task: value}. heldout: its held-out scores, or None. control: the
    scores of the random-components control, on control_split ("search" or "heldout"), or None. components: the
    per-component verdicts of the synthesis pipeline (each with "verdict" and "label"), or None.
    """
    targets = targets_of(spec)
    if models is None:
        models = library.get()["models"]
    limits = spec.get("limits") or {}
    comparators = spec.get("comparators") or {}
    rows = []
    rows.append(make_row("base", spec_values(spec, "base"), targets, "search",
                         "the host alone, same items and same scorer", stub))
    rows.append(make_row("lora", spec_values(spec, "lora"), targets, "search",
                         lora_note(spec, stub), False))
    distill = dict(comparators.get("distill") or {})
    distill_stub = False
    distill_note = "the host healed by distillation from the same source at the same heal budget"
    if len(distill) == 0 and stub:
        distill = stub_distill(spec)
        distill_stub = True
        distill_note = "stub data, not a measurement"
    elif len(distill) == 0:
        distill_note = "not measured yet. Run a distillation heal with the same budget and add it to the spec."
    rows.append(make_row("distill", distill, targets, "search", distill_note, distill_stub))
    if control is None:
        rows.append(make_row("random", {}, targets, control_split,
                             "not measured yet. The control runs a build with random parts of the same size.", stub))
    else:
        rows.append(make_row("random", control, targets, control_split,
                             "the same build with random components of the same size", stub))
    host_b = (spec.get("size") or {}).get("host_params_b")
    limit_b = limits.get("max_total_params_b")
    far = False
    if host_b and limit_b and limit_b >= FAR_ABOVE_BASE * host_b:
        far = True
        per_task, note = best_model_row(spec, limit_b, models)
        rows.append(make_row("best_model", per_task, targets, "search", note, False))
    result_value = mean_over(targets, scores)
    for row in rows:
        row["result_value"] = result_on(row["split"], scores, heldout, targets)
        if row["measured"] and row["result_value"] is not None:
            row["delta"] = round(row["result_value"] - row["value"], 4)
            row["beaten"] = row["result_value"] > row["value"] + CLEAR_MARGIN
        mark_margin(row)
    verdict, beaten, not_beaten, unmeasured = verdict_of(rows, result_value)
    text = headline(verdict, rows, beaten, not_beaten, unmeasured)
    verdict, text = apply_component_rule(verdict, text, components)
    return {"verdict": verdict, "headline": text,
            "margin": CLEAR_MARGIN, "result": {"label": "This build", "value": result_value,
                                              "per_task": dict(scores or {}), "params_b": params_b,
                                              "heldout": mean_over(targets, heldout)},
            "rows": rows, "beaten": beaten, "not_beaten": not_beaten, "unmeasured": unmeasured,
            "best_model_shown": far, "stub": bool(stub), "no_component_contributes": no_component_contributes(components)}


def verdict_only(spec, scores, heldout=None, control=None, control_split="search", stub=False, models=None,
                 components=None):
    """The verdict string alone, for lists."""
    return build(spec, scores, None, heldout, control, control_split, stub, models, components)["verdict"]


# Shrink comparison. A shrunk model is checked against the same slim with random units removed (the random
# control) and against the best existing model of about its size. The original model is a reference and
# does not decide the verdict: removing skills is expected to cost something. The general-text control is
# optional and also does not decide it.

SHRINK_REQUIRED = ["random", "best_model"]
NEAR_SIZE_LOW = 0.5
NEAR_SIZE_HIGH = 1.05


def models_near_size(models, params_b, exclude):
    """Library models whose size is within NEAR_SIZE_LOW to NEAR_SIZE_HIGH of the result's, except the source."""
    out = {}
    if params_b is None:
        return out
    for hf_id in models:
        size = models[hf_id]["params_b"]
        if hf_id == exclude:
            continue
        if NEAR_SIZE_LOW * params_b <= size <= NEAR_SIZE_HIGH * params_b:
            out[hf_id] = models[hf_id]
    return out


def shrink_best_model_row(spec, params_b, source_id, models, targets, stub):
    """(row, required). required is true when the library has a model near the result's size."""
    near = models_near_size(models, params_b, source_id)
    if len(near) == 0:
        row = make_row("best_model", {}, targets, "search",
                       "no library model is near this size, so there is no existing model to compare with", False)
        return row, False
    limit_b = NEAR_SIZE_HIGH * params_b
    per_task, note = best_model_row(spec, limit_b, near)
    if len(per_task) == 0:
        names = ", ".join(sorted(near)[:3])
        note = "models near this size (" + names + ") have no score on this benchmark yet"
    return make_row("best_model", per_task, targets, "search", note, False), True


def shrink_rows(spec, params_b, source_scores, random_scores, general_scores, models, stub):
    targets = targets_of(spec)
    source_id = (spec.get("shrink") or {}).get("source_model")
    rows = []
    row = make_row("source", source_scores or {}, targets, "search",
                   "the original model on the same items. The shrunk model is expected to be below it.", stub)
    row["required"] = False
    rows.append(row)
    if random_scores is None:
        note = "not measured yet. The control removes the same counts at random and heals the same steps."
    else:
        note = "the same slim with random units removed, same counts, same heal steps"
    row = make_row("random", random_scores or {}, targets, "search", note, stub)
    row["required"] = True
    rows.append(row)
    if general_scores is not None:
        row = make_row("general", general_scores, targets, "search",
                       "the same slim keeping what general text uses instead of the skill", stub)
        row["required"] = False
        rows.append(row)
    row, required = shrink_best_model_row(spec, params_b, source_id, models, targets, stub)
    row["required"] = required
    rows.append(row)
    return rows


def shrink_headline(verdict, rows, beaten, not_beaten, unmeasured, result_value):
    by_id = {}
    for row in rows:
        by_id[row["id"]] = row
    kept = ""
    source = by_id["source"]
    if source["measured"] and result_value is not None and source["value"] > 0:
        kept = " It keeps %.0f%% of the original model's score." % (100.0 * result_value / source["value"])
    if verdict == "none":
        return "No score yet."
    if verdict == "failed":
        return failed_text(by_id, not_beaten, "the choice of units to remove is") + kept
    if verdict == "unproven":
        names = [by_id[row_id]["label"] for row_id in unmeasured]
        text = "Not proven. "
        if len(beaten) > 0:
            text += "It beats " + ", ".join(by_id[row_id]["label"] for row_id in beaten) + ". "
        return text + "Not measured yet: " + ", ".join(names) + "." + kept
    random_row = by_id["random"]
    text = ("Success. It beats the random control by " + points(random_row["delta"]) + " points")
    if "best_model" in beaten:
        text += " and is not below the best existing model of its size"
    return text + "." + kept


def shrink_verdict(rows, result_value):
    """(verdict, beaten ids, not beaten ids, unmeasured ids) over the required rows."""
    beaten = []
    not_beaten = []
    unmeasured = []
    for row in rows:
        if not row.get("required"):
            continue
        if not row["measured"] or row["beaten"] is None:
            unmeasured.append(row["id"])
        elif row["beaten"]:
            beaten.append(row["id"])
        else:
            not_beaten.append(row["id"])
    if result_value is None:
        return "none", beaten, not_beaten, unmeasured
    if len(not_beaten) > 0:
        return "failed", beaten, not_beaten, unmeasured
    if len(unmeasured) > 0:
        return "unproven", beaten, not_beaten, unmeasured
    return "success", beaten, not_beaten, unmeasured


def build_shrink(spec, scores, params_b=None, source_scores=None, random_scores=None, general_scores=None,
                 stub=False, models=None):
    """The comparison of a shrunk model with its reference set. Verdict: success, failed, unproven or none."""
    targets = targets_of(spec)
    if models is None:
        models = library.get()["models"]
    rows = shrink_rows(spec, params_b, source_scores, random_scores, general_scores, models, stub)
    result_value = mean_over(targets, scores)
    for row in rows:
        row["result_value"] = result_value
        if row["measured"] and result_value is not None:
            row["delta"] = round(result_value - row["value"], 4)
            if row["id"] == "random":
                row["beaten"] = result_value > row["value"] + CLEAR_MARGIN
            elif row["id"] == "best_model":
                row["beaten"] = result_value >= row["value"]
            elif row["id"] == "general":
                row["beaten"] = result_value > row["value"] + CLEAR_MARGIN
        mark_margin(row)
    verdict, beaten, not_beaten, unmeasured = shrink_verdict(rows, result_value)
    return {"verdict": verdict, "request_type": "shrink",
            "headline": shrink_headline(verdict, rows, beaten, not_beaten, unmeasured, result_value),
            "margin": CLEAR_MARGIN, "result": {"label": "Shrunk model", "value": result_value,
                                              "per_task": dict(scores or {}), "params_b": params_b,
                                              "heldout": None},
            "rows": rows, "beaten": beaten, "not_beaten": not_beaten, "unmeasured": unmeasured,
            "best_model_shown": True, "stub": bool(stub)}
