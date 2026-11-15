"""Interface 4: eval(model_path, split, tasks) -> scores.

    from wsp_eval import eval
    scores = eval("path/to/model", "search", ["mmlupro:biology", "bbeh"])

Every call:
  * counts against the caller's budget namespace (argument `namespace`, or the
    WSP_BUDGET_NS environment variable, default "default"),
  * reuses cached generations keyed by (model hash, prompt),
  * appends one line to experiments.jsonl.

Held-out calls are allowed once per (model, task set). Search-split results are
labelled as such and must never be reported as results.

Grafted models: the result carries `fit_r2`, the R squared of each graft's
host-to-donor fit from compile_report.json, so it is logged next to the
scores. `program_seed` is the compile/heal seed (claims use 3 seeds).

`ablation` (probe use only): knock out units of the model while generating;
see wsp_eval/units.py.
"""
import datetime
import fcntl
import json
import os

from wsp_eval import backends, budget, cache, config, models, splits, tasks as tasks_module, units

SEARCH_LABEL = "SEARCH SPLIT - for search only, not a result"
HELDOUT_LABEL = "HELD-OUT"


class HeldoutAlreadyRun(Exception):
    pass


def _now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


def _append_jsonl(path, record):
    with open(path, "a") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        f.write(json.dumps(record) + "\n")
        fcntl.flock(f, fcntl.LOCK_UN)


# ---------------------------------------------------------------------------
# Held-out ledger: each (model hash, task) pair is evaluated on held-out once.
# ---------------------------------------------------------------------------

def _ledger_path():
    return config.work_path("heldout_ledger.json")


def _read_ledger():
    if not os.path.exists(_ledger_path()):
        return {}
    with open(_ledger_path()) as f:
        return json.load(f)


def _check_heldout(model_hash, task_names):
    ledger = _read_ledger()
    already = []
    for name in task_names:
        if model_hash + "|" + name in ledger:
            already.append(name)
    if len(already) > 0:
        raise HeldoutAlreadyRun("held-out split already evaluated for this model on " + ", ".join(already) +
                                ". It is run once. Pass allow_heldout_rerun=True only to recover from a crash.")


def _record_heldout(model_hash, task_names, model_path):
    ledger = _read_ledger()
    for name in task_names:
        ledger[model_hash + "|" + name] = {"model": model_path, "time": _now()}
    with open(_ledger_path(), "w") as f:
        json.dump(ledger, f, indent=1)


# ---------------------------------------------------------------------------
# Reference scores (base and parents) and the combined score
# ---------------------------------------------------------------------------

def _reference_path(split):
    return config.work_path("reference_" + split + ".json")


def save_reference(split, base_scores, parent_scores):
    """Store base and single-skill parent scores used to normalize the combined score.

    base_scores:   {task: score}
    parent_scores: {parent label: {task: score}}
    """
    reference = {"split": split, "time": _now(), "base": base_scores, "parents": parent_scores}
    with open(_reference_path(split), "w") as f:
        json.dump(reference, f, indent=2)
    return reference


def load_reference(split):
    if not os.path.exists(_reference_path(split)):
        return None
    with open(_reference_path(split)) as f:
        return json.load(f)


def combined_score(task_scores, reference):
    """Mean over skills of (score - base) / (best single-skill parent - base).

    Each normalized score is clipped to [0, 1] for the mean; the raw value is
    also returned. Skills whose best parent does not beat the base cannot be
    normalized and are left out of the mean (listed in `skipped`).
    """
    if reference is None:
        return None
    per_skill = {}
    skipped = []
    clipped_values = []
    for task, score in task_scores.items():
        if task not in reference["base"]:
            skipped.append(task)
            continue
        base = reference["base"][task]
        best = None
        best_parent = None
        for parent, parent_scores in reference["parents"].items():
            if task in parent_scores:
                if best is None or parent_scores[task] > best:
                    best = parent_scores[task]
                    best_parent = parent
        if best is None or best <= base:
            skipped.append(task)
            continue
        raw = (score - base) / (best - base)
        clipped = min(1.0, max(0.0, raw))
        per_skill[task] = {"normalized": clipped, "raw": raw, "base": base, "best_single": best,
                           "best_parent": best_parent}
        clipped_values.append(clipped)
    if len(clipped_values) == 0:
        mean = None
    else:
        mean = sum(clipped_values) / len(clipped_values)
    return {"score": mean, "per_skill": per_skill, "skipped": skipped}


# ---------------------------------------------------------------------------
# Compile report: fit R squared per graft
# ---------------------------------------------------------------------------

R2_KEYS = ("r2", "r_squared", "fit_r2", "rsquared", "R2")


def _find_r2(entry):
    for key in R2_KEYS:
        if key in entry and isinstance(entry[key], (int, float)):
            return entry[key]
    if isinstance(entry.get("fit"), dict):
        return _find_r2(entry["fit"])
    return None


def read_fit_scores(model_dir):
    """[{graft, r2}] from compile_report.json, or None if there are no grafts.

    Tolerant of the exact field name (r2, r_squared, fit_r2, or fit: {r2}),
    because the compiler's report format is still settling. r2 is None when
    the compiler skipped the fit (donor too large to run).
    """
    import os
    path = os.path.join(str(model_dir), "compile_report.json")
    if not os.path.isfile(path):
        return None
    with open(path) as f:
        report = json.load(f)
    grafts = report.get("grafts")
    if not grafts:
        return None
    scores = []
    index = 0
    for entry in grafts:
        label = entry.get("id") or entry.get("part") or entry.get("name") or ("graft " + str(index))
        row = {"graft": str(label), "r2": _find_r2(entry)}
        for key in ("fit", "fit_r2_dim", "fit_pairs", "part_params", "glue_params", "host_layer", "donor_layer",
                    "control", "glue_rank", "kind"):
            if key in entry:
                row[key] = entry[key]
        scores.append(row)
        index += 1
    return scores


def program_seed(model_dir):
    program = backends.read_program(model_dir)
    if program is None:
        return None
    return program.get("seed")


# ---------------------------------------------------------------------------
# eval
# ---------------------------------------------------------------------------

def _generate_with_cache(backend, model_dir, model_hash_value, prompts, max_new_tokens, ablation=None):
    keys = []
    for prompt in prompts:
        keys.append(cache.make_key(model_hash_value, backend.name, max_new_tokens, prompt))
    found = cache.get_many(keys)

    missing_prompts = []
    missing_keys = []
    seen = {}
    for key, prompt in zip(keys, prompts):
        if key not in found and key not in seen:
            seen[key] = True
            missing_prompts.append(prompt)
            missing_keys.append(key)

    if len(missing_prompts) > 0:
        new_outputs = backend.generate(model_dir, missing_prompts, max_new_tokens, model_hash_value,
                                       ablation=ablation)
        rows = []
        for key, prompt, output in zip(missing_keys, missing_prompts, new_outputs):
            found[key] = output
            rows.append((key, model_hash_value, backend.name, max_new_tokens, prompt, output))
        cache.put_many(rows)

    outputs = []
    for key in keys:
        outputs.append(found[key])
    return outputs, len(missing_prompts)


def evaluate(model_path, split, tasks, namespace=None, allow_heldout_rerun=False, note="",
             return_outputs=False, ablation=None, backend=None):
    if split == "heal":
        raise ValueError("the heal split is training data; never evaluate on it")
    if split not in ("search", "heldout"):
        raise ValueError("split must be 'search' or 'heldout'")
    if isinstance(tasks, str):
        tasks = [tasks]
    if namespace is None:
        namespace = os.environ.get("WSP_BUDGET_NS", "default")

    model_dir = models.resolve_model_path(model_path)
    model_hash_value = models.model_hash(model_dir)
    if ablation is not None:
        # An ablated model is a different model: separate cache entries and held-out ledger.
        model_hash_value = model_hash_value + "-ablate-" + units.ablation_key(ablation)

    if split == "heldout" and not allow_heldout_rerun:
        _check_heldout(model_hash_value, tasks)
    budget.check(namespace, split)

    backend = backends.get_backend(model_dir, needs_hooks=ablation is not None, name=backend)

    # Render every prompt, then generate once per max_new_tokens group so the
    # backend sees large batches.
    task_objects = {}
    task_items = {}
    task_prompts = {}
    groups = {}
    for name in tasks:
        task = tasks_module.get_task(name)
        items = splits.split_items(task, split)
        prompts = []
        for item in items:
            prompts.append(backends.render_item(backend, model_dir, item))
        task_objects[name] = task
        task_items[name] = items
        task_prompts[name] = prompts
        if task.max_new_tokens not in groups:
            groups[task.max_new_tokens] = []
        groups[task.max_new_tokens].append(name)

    task_outputs = {}
    total_items = 0
    total_generated = 0
    for max_new_tokens, names in groups.items():
        all_prompts = []
        for name in names:
            all_prompts.extend(task_prompts[name])
        outputs, generated = _generate_with_cache(backend, model_dir, model_hash_value, all_prompts, max_new_tokens,
                                                  ablation)
        total_generated += generated
        total_items += len(all_prompts)
        position = 0
        for name in names:
            count = len(task_prompts[name])
            task_outputs[name] = outputs[position:position + count]
            position += count

    per_task = {}
    flat_scores = {}
    for name in tasks:
        task = task_objects[name]
        correct = 0.0
        for item, output in zip(task_items[name], task_outputs[name]):
            correct += task.score(output, item)
        n = len(task_items[name])
        per_task[name] = {"score": correct / n, "correct": correct, "n": n}
        flat_scores[name] = correct / n

    usage = budget.charge(namespace, split, total_items, total_generated)
    if split == "heldout":
        _record_heldout(model_hash_value, tasks, str(model_path))

    if split == "search":
        label = SEARCH_LABEL
    else:
        label = HELDOUT_LABEL

    result = {
        "timestamp": _now(),
        "model": str(model_path),
        "model_hash": model_hash_value[:16],
        "split": split,
        "label": label,
        "tasks": per_task,
        "combined": combined_score(flat_scores, load_reference(split)),
        "budget": {"namespace": namespace, "eval_calls": usage["eval_calls"],
                   "heldout_calls": usage["heldout_calls"], "limit": usage["limit"]},
        "backend": backend.name,
        "generations_run": total_generated,
        "fit_r2": read_fit_scores(model_dir),
        "program_seed": program_seed(model_dir),
        "ablation": ablation,
        "note": note,
    }

    _append_jsonl(config.experiments_path(), {
        "timestamp": result["timestamp"], "model": result["model"], "model_hash": result["model_hash"],
        "split": split, "label": label, "scores": flat_scores, "combined": result["combined"],
        "eval_count": usage["eval_calls"], "namespace": namespace, "backend": backend.name, "note": note,
        "fit_r2": result["fit_r2"], "program_seed": result["program_seed"], "ablation": ablation,
    })

    if return_outputs:
        result["outputs"] = task_outputs
    return result


# The brief's name for interface 4. (Shadows the builtin only inside callers
# that import it by this name.)
eval = evaluate


def swarm_eval(model_path, split, tasks):
    """Part 3's eval hook: fn(model_path, split, tasks) -> {task: score}.

    Set config["local_eval"] = "wsp_eval.harness:swarm_eval" in wsp/swarm/config.py.
    Part 3's env owns the comparison budget; calls here are also counted under
    the namespace in WSP_BUDGET_NS (default "swarm") and logged to experiments.jsonl
    with fit_r2. Every model, plain or grafted, generates with the hf backend, so
    candidates and the host base scores are produced the same way.
    """
    namespace = os.environ.get("WSP_BUDGET_NS", "swarm")
    result = evaluate(model_path, split, tasks, namespace=namespace, note="swarm", backend="hf")
    scores = {}
    for task in tasks:
        scores[task] = result["tasks"][task]["score"]
    return scores
