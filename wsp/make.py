"""Make a model for a request: plan candidates, compile and score them, pick the winner.

A request says what the model is for and what it may cost. wsp.planner
turns it into a few candidate programs, a runner (modal_make.py on Modal)
compiles and scores each one, and make() picks the winner and writes a
report card. See docs/make.md.

Request (JSON, in requests/*.json):

    {"name": str,
     "description": str (one line, optional),
     "task": {"kind": "mcq" | "generate",
              "benchmark": "mmlu" | "medmcqa" | "medqa" | "mmlu_pro" | "gsm8k" | ...,
              "subjects": [str] or null},
     "constraints": {"max_decode_params_b": float or null,
                     "max_total_params_b": float or null,
                     "max_train_minutes": float or null,
                     "gpu": "H100" | "H200"},
     "objective": "max_accuracy" | "min_cost_at_target",
     "accuracy_target": float or null}

Profile (one per model, wsp/profiles.py):

    {"key", "hf_id", "family", "params_b", "active_params_b", "layers",
     "hidden", "tokenizer_family", "scores": {"<benchmark>": float, ...},
     "decode_tok_s": float or null, "source": str}

Candidate (wsp/planner.py):

    {"program": Program.to_dict() output,
     "predicted": {"accuracy": float, "decode_params_b": float,
                   "prefill_params_b": float, "train_minutes": float},
     "rationale": str,
     "planner": "rules" | "llm"}

A "bake" key (true or a dict of bake settings) on the candidate or in
its program dict marks a tapped program that is baked into a plain host
after tap training (wsp.bake). make() moves it into the program dict.

planner.plan(request, profiles, k=6) returns a list of Candidate,
feasible ones first, sorted by the objective.

make(request, profiles, run_candidates) returns a report card dict.
run_candidates(list of Candidate) returns a list of measured results:

    {"candidate": Candidate, "accuracy": float or None,
     "decode_tok_s": float or None, "train_seconds": float or None,
     "model_dir": str or None, "error": str or None}

Accuracies are fractions from 0 to 1. A value above 1 (a profile score
written as a percent, such as 48.0) is divided by 100 before use.

Winner rule:
- max_accuracy: the best measured accuracy among candidates that ran
  without an error and meet every constraint. Ties go to fewer decode
  parameters, then fewer total parameters, then less training time.
- min_cost_at_target: among those candidates, the ones at or above
  accuracy_target, the cheapest first: fewest decode parameters, then
  fewest total parameters, then least training time. When no candidate
  reaches the target, the winner is the most accurate one and the card
  says the target was missed.

Total parameters are what must sit in memory to serve the model: the host
plus every distinct tap donor, from the profiles. A baked candidate needs
only its host.

Command line:

    python -m wsp.make requests/med_mcq_1b.json --dry-run
    python -m wsp.make requests/med_mcq_1b.json --results results/make_med_mcq_1b_rows.json

--dry-run prints the plan and compiles nothing. --results builds the
report card from measured results (for example the merged shards of a
modal_make.py fan-out) and writes results/make/<name>.json.
"""

import argparse
import copy
import inspect
import json
import os
import sys

KINDS = ("mcq", "generate")
OBJECTIVES = ("max_accuracy", "min_cost_at_target")
GPUS = ("H100", "H200")
CONSTRAINT_KEYS = ("max_decode_params_b", "max_total_params_b", "max_train_minutes", "gpu")
REQUEST_KEYS = ("name", "description", "task", "constraints", "objective", "accuracy_target")
TASK_KEYS = ("kind", "benchmark", "subjects")


# Requests


def is_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def request_problems(request):
    """Return a list of plain-English problems with a request. An empty list means it is valid."""
    problems = []
    if not isinstance(request, dict):
        return ["a request is a JSON object"]
    for key in request:
        if key not in REQUEST_KEYS:
            problems.append(f"unknown request key {key!r}")
    name = request.get("name")
    if not isinstance(name, str) or name == "":
        problems.append("name must be a non-empty string")
    if "description" in request and not isinstance(request["description"], str):
        problems.append("description must be a string")

    task = request.get("task")
    if not isinstance(task, dict):
        problems.append("task must be an object")
    else:
        for key in task:
            if key not in TASK_KEYS:
                problems.append(f"unknown task key {key!r}")
        if task.get("kind") not in KINDS:
            problems.append(f"task.kind must be one of {', '.join(KINDS)}")
        benchmark = task.get("benchmark")
        if not isinstance(benchmark, str) or benchmark == "":
            problems.append("task.benchmark must be a non-empty string")
        subjects = task.get("subjects")
        if subjects is not None:
            if not isinstance(subjects, list) or len(subjects) == 0:
                problems.append("task.subjects must be null or a non-empty list of strings")
            else:
                for subject in subjects:
                    if not isinstance(subject, str) or subject == "":
                        problems.append("task.subjects must hold only non-empty strings")
                        break

    constraints = request.get("constraints")
    if not isinstance(constraints, dict):
        problems.append("constraints must be an object")
    else:
        for key in constraints:
            if key not in CONSTRAINT_KEYS:
                problems.append(f"unknown constraints key {key!r}")
        for key in ("max_decode_params_b", "max_total_params_b", "max_train_minutes"):
            value = constraints.get(key)
            if value is None:
                continue
            if not is_number(value) or value <= 0:
                problems.append(f"constraints.{key} must be null or a positive number")
        if constraints.get("gpu") not in GPUS:
            problems.append(f"constraints.gpu must be one of {', '.join(GPUS)}")

    objective = request.get("objective")
    if objective not in OBJECTIVES:
        problems.append(f"objective must be one of {', '.join(OBJECTIVES)}")
    target = request.get("accuracy_target")
    if target is not None:
        if not is_number(target) or target <= 0 or target > 1:
            problems.append("accuracy_target must be null or a fraction above 0 and at most 1")
    if objective == "min_cost_at_target" and target is None:
        problems.append("min_cost_at_target needs an accuracy_target")
    return problems


def check_request(request):
    """Raise ValueError when the request does not match the schema."""
    problems = request_problems(request)
    if len(problems) > 0:
        raise ValueError("bad request: " + "; ".join(problems))


def load_request(path):
    with open(path) as f:
        request = json.load(f)
    check_request(request)
    return request


# Profiles


def profile_list(profiles):
    """Accept a list of profiles, a dict of key -> profile, or {"profiles": [...]}. Return a list."""
    if profiles is None:
        return []
    if isinstance(profiles, dict):
        if "profiles" in profiles and isinstance(profiles["profiles"], list):
            return list(profiles["profiles"])
        out = []
        for key in profiles:
            profile = dict(profiles[key])
            if "key" not in profile:
                profile["key"] = key
            out.append(profile)
        return out
    return list(profiles)


def load_profiles(path="profiles.json"):
    """Read profiles from a JSON file. Without the file, ask wsp.profiles for them."""
    if path is not None and os.path.exists(path):
        with open(path) as f:
            return profile_list(json.load(f))
    from wsp import profiles as module

    if hasattr(module, "load_profiles"):
        return profile_list(module.load_profiles())
    if hasattr(module, "PROFILES"):
        return profile_list(module.PROFILES)
    raise FileNotFoundError(f"no profiles file at {path!r}, and wsp.profiles has no load_profiles or PROFILES")


def profiles_by_hf_id(profiles):
    table = {}
    for profile in profile_list(profiles):
        table[profile.get("hf_id")] = profile
    return table


def as_fraction(value):
    """Return an accuracy as a fraction. A value above 1 is a percent."""
    if value is None:
        return None
    value = float(value)
    if value > 1.0:
        return value / 100.0
    return value


def profile_score(profile, benchmark):
    scores = profile.get("scores") or {}
    if benchmark not in scores:
        return None
    return as_fraction(scores[benchmark])


def profile_name(profile):
    if profile is None:
        return "unknown"
    name = profile.get("hf_id") or profile.get("key") or "unknown"
    return name.split("/")[-1]


# Programs


def program_core(program):
    """The program dict without the bake marker or other keys the language does not know."""
    from wsp.lang import PROGRAM_KEYS

    core = {}
    for key in program:
        if key in PROGRAM_KEYS:
            core[key] = program[key]
    return core


def normalize_candidate(candidate):
    """Return a copy whose program dict holds the bake marker.

    The planner can put the bake settings on the candidate ("bake" next
    to "program") or in the program dict. After this, they are in the
    program dict, so the program alone says whether it is baked.
    """
    out = dict(candidate)
    program = dict(candidate.get("program") or {})
    if candidate.get("bake") and not program.get("bake"):
        program["bake"] = copy.deepcopy(candidate["bake"])
    out["program"] = program
    return out


def bake_spec(program):
    """Return the bake settings of a program dict as a dict, or None when it is not baked."""
    value = program.get("bake")
    if not value:
        return None
    if isinstance(value, dict):
        return dict(value)
    return {}


def program_text(program):
    """The wsp text of a candidate program. A bake marker becomes a comment line at the end."""
    from wsp.lang import format_program

    try:
        text = format_program(program_core(program))
    except ValueError as error:
        text = f"# cannot write this program as text: {error}\n" + json.dumps(program_core(program)) + "\n"
    spec = bake_spec(program)
    if spec is not None:
        words = ["# bake"]
        for key in sorted(spec):
            words.append(f"{key} {spec[key]}")
        text += " ".join(words) + "\n"
    return text


def donor_ids(program):
    ids = []
    for tap in program.get("taps") or []:
        if tap["donor"] not in ids:
            ids.append(tap["donor"])
    return ids


def is_bare(program):
    """True when the program is a source model alone: no taps, grafts, LoRA, masks, or task vectors."""
    if len(program.get("taps") or []) > 0:
        return False
    if len(program.get("grafts") or []) > 0:
        return False
    if program.get("lora") is not None:
        return False
    if program.get("masks") is not None:
        return False
    if len(program.get("task_vectors") or []) > 0:
        return False
    return True


def is_host_lora(program):
    """True for a fine-tuning baseline: a LoRA on the host and nothing else."""
    if program.get("lora") is None:
        return False
    if len(program.get("taps") or []) > 0 or len(program.get("grafts") or []) > 0:
        return False
    return bake_spec(program) is None


def describe_program(program):
    """A short plain-English name for a program, such as 'Llama-3.2-1B-Instruct + tap from Qwen2.5-7B-Instruct'."""
    host = program.get("host") or program.get("base") or "unknown"
    words = host.split("/")[-1]
    donors = donor_ids(program)
    if len(donors) == 1:
        words += " + tap from " + donors[0].split("/")[-1]
    elif len(donors) > 1:
        names = []
        for donor in donors:
            names.append(donor.split("/")[-1])
        words += " + taps from " + ", ".join(names)
    if len(program.get("grafts") or []) > 0:
        words += f" + {len(program['grafts'])} graft(s)"
    if program.get("lora") is not None:
        rank = program["lora"].get("rank")
        if rank is None:
            words += " + LoRA"
        else:
            words += f" + LoRA r{rank}"
    if bake_spec(program) is not None:
        words += ", baked into the host"
    if is_bare(program):
        words += " alone"
    return words


def total_params_b(candidate, profiles):
    """Parameters that must sit in memory to serve the candidate, in billions.

    The host plus each distinct tap donor, from the profiles. A baked
    candidate serves the host alone. When a model has no profile, the
    predicted decode and prefill counts give a lower bound.
    """
    program = candidate["program"]
    predicted = candidate.get("predicted") or {}
    table = profiles_by_hf_id(profiles)
    host = program.get("host") or program.get("base")
    decode = predicted.get("decode_params_b")
    if bake_spec(program) is not None:
        if decode is not None:
            return float(decode)
        if host in table:
            return float(table[host]["params_b"])
        return None
    total = 0.0
    known = True
    if host in table:
        total += float(table[host]["params_b"])
    else:
        known = False
    for donor in donor_ids(program):
        if donor in table:
            total += float(table[donor]["params_b"])
        else:
            known = False
    if known:
        if decode is not None and decode > total:
            # The tap branches and LoRA add a little on top of the host.
            total = float(decode)
        return total
    fallback = None
    for key in ("decode_params_b", "prefill_params_b"):
        value = predicted.get(key)
        if value is not None and (fallback is None or value > fallback):
            fallback = float(value)
    return fallback


# Constraints and winner


def check_constraints(request, decode_b, total_b, train_minutes):
    """Return a list of the constraints these numbers break, in plain English."""
    constraints = request.get("constraints") or {}
    broken = []
    limit = constraints.get("max_decode_params_b")
    if limit is not None and decode_b is not None and decode_b > limit:
        broken.append(f"decode params {decode_b:.2f}B over the {limit}B limit")
    limit = constraints.get("max_total_params_b")
    if limit is not None and total_b is not None and total_b > limit:
        broken.append(f"total params {total_b:.2f}B over the {limit}B limit")
    limit = constraints.get("max_train_minutes")
    if limit is not None and train_minutes is not None and train_minutes > limit:
        broken.append(f"training {train_minutes:.1f} min over the {limit} min limit")
    return broken


def program_key(program):
    return json.dumps(program, sort_keys=True)


def match_results(candidates, results):
    """Pair each candidate with its measured result, or None when it did not run.

    Results are matched by their candidate's program, in order, so two
    candidates with the same program each get one result.
    """
    queues = {}
    for result in results or []:
        if not isinstance(result, dict) or not isinstance(result.get("candidate"), dict):
            continue
        key = program_key(normalize_candidate(result["candidate"])["program"])
        if key not in queues:
            queues[key] = []
        queues[key].append(result)
    matched = []
    for candidate in candidates:
        key = program_key(candidate.get("program") or {})
        if key in queues and len(queues[key]) > 0:
            matched.append(queues[key].pop(0))
        else:
            matched.append(None)
    return matched


def candidate_row(index, request, profiles, candidate, result):
    """One report card row: the prediction, the measurement, and whether it is eligible to win."""
    program = candidate["program"]
    predicted = dict(candidate.get("predicted") or {})
    decode_b = predicted.get("decode_params_b")
    prefill_b = predicted.get("prefill_params_b")
    total_b = total_params_b(candidate, profiles)
    row = {"index": index, "description": describe_program(program), "program": program,
           "program_text": program_text(program), "rationale": candidate.get("rationale", ""),
           "planner": candidate.get("planner"), "predicted": predicted, "measured": None,
           "decode_params_b": decode_b, "prefill_params_b": prefill_b, "total_params_b": total_b,
           "status": "not run", "violations": [], "eligible": False}

    if result is None:
        row["violations"] = check_constraints(request, decode_b, total_b, predicted.get("train_minutes"))
        return row
    measured = {}
    for key in ("accuracy", "decode_tok_s", "train_seconds", "model_dir", "error"):
        measured[key] = result.get(key)
    if result.get("decode_params_b") is not None:
        row["decode_params_b"] = result["decode_params_b"]
        decode_b = result["decode_params_b"]
    measured["accuracy"] = as_fraction(measured["accuracy"])
    row["measured"] = measured

    if measured["error"] or measured["accuracy"] is None:
        row["status"] = "failed"
        if not measured["error"]:
            measured["error"] = "no accuracy returned"
        return row

    train_minutes = predicted.get("train_minutes")
    if measured["train_seconds"] is not None:
        train_minutes = measured["train_seconds"] / 60.0
    row["violations"] = check_constraints(request, decode_b, total_b, train_minutes)
    if len(row["violations"]) > 0:
        row["status"] = "over budget"
        return row
    row["status"] = "ok"
    row["eligible"] = True
    if predicted.get("accuracy") is not None:
        row["prediction_error"] = measured["accuracy"] - as_fraction(predicted["accuracy"])
    return row


def big(value):
    """Sort key part: a missing count sorts after every real one."""
    if value is None:
        return float("inf")
    return value


def accuracy_key(row):
    return (-row["measured"]["accuracy"], big(row["decode_params_b"]), big(row["total_params_b"]),
            big(row["measured"]["train_seconds"]))


def cost_key(row):
    return (big(row["decode_params_b"]), big(row["total_params_b"]), big(row["measured"]["train_seconds"]),
            -row["measured"]["accuracy"])


def pick_winner(request, rows):
    """Return (winner row or None, target_met or None) under the request's objective."""
    eligible = []
    for row in rows:
        if row["eligible"]:
            eligible.append(row)
    if len(eligible) == 0:
        return None, None
    target = as_fraction(request.get("accuracy_target"))
    if request["objective"] == "min_cost_at_target":
        reached = []
        for row in eligible:
            if row["measured"]["accuracy"] >= target:
                reached.append(row)
        if len(reached) > 0:
            reached.sort(key=cost_key)
            return reached[0], True
        eligible.sort(key=accuracy_key)
        return eligible[0], False
    eligible.sort(key=accuracy_key)
    winner = eligible[0]
    if target is None:
        return winner, None
    return winner, winner["measured"]["accuracy"] >= target


# Baselines


def measured_for(rows, test):
    """The best measured accuracy among ok rows whose program passes test, or None."""
    best = None
    for row in rows:
        if row["status"] not in ("ok", "over budget"):
            continue
        if not test(row["program"]):
            continue
        if best is None or row["measured"]["accuracy"] > best["measured"]["accuracy"]:
            best = row
    return best


def source_row(profile, benchmark, rows, label):
    hf_id = profile.get("hf_id")

    def alone(program):
        return is_bare(program) and program.get("host") == hf_id and bake_spec(program) is None

    measured = measured_for(rows, alone)
    out = {"label": label, "model": profile_name(profile), "hf_id": hf_id, "params_b": profile.get("params_b"),
           "profile_accuracy": profile_score(profile, benchmark), "measured_accuracy": None}
    if measured is not None:
        out["measured_accuracy"] = measured["measured"]["accuracy"]
    return out


def best_profile(profiles, benchmark, max_params):
    best = None
    best_score = None
    for profile in profile_list(profiles):
        score = profile_score(profile, benchmark)
        if score is None:
            continue
        if max_params is not None and float(profile.get("params_b") or 0) > max_params:
            continue
        if best is None or score > best_score:
            best = profile
            best_score = score
    return best


def baselines(request, profiles, rows, winner):
    """Host alone, the best source model in the total budget and in the decode budget, and host + LoRA."""
    benchmark = request["task"]["benchmark"]
    constraints = request.get("constraints") or {}
    table = profiles_by_hf_id(profiles)
    out = {"host": None, "best_source": None, "best_source_at_decode_budget": None, "host_lora": None}

    host = None
    if winner is not None:
        host = winner["program"].get("host")
    elif len(rows) > 0:
        host = rows[0]["program"].get("host")
    if host is not None:
        profile = table.get(host)
        if profile is None:
            profile = {"hf_id": host}
        out["host"] = source_row(profile, benchmark, rows, "host alone")

        def lora_on_host(program):
            return is_host_lora(program) and program.get("host") == host

        lora_row = measured_for(rows, lora_on_host)
        if lora_row is not None:
            out["host_lora"] = {"label": "host + LoRA", "model": profile_name(profile), "hf_id": host,
                                "program_text": lora_row["program_text"],
                                "measured_accuracy": lora_row["measured"]["accuracy"]}

    best = best_profile(profiles, benchmark, constraints.get("max_total_params_b"))
    if best is not None:
        out["best_source"] = source_row(best, benchmark, rows, "best source model in the total budget")
    best = best_profile(profiles, benchmark, constraints.get("max_decode_params_b"))
    if best is not None:
        out["best_source_at_decode_budget"] = source_row(best, benchmark, rows,
                                                         "best source model in the decode budget")
    return out


def baseline_accuracy(baseline):
    """Measured accuracy when there is one, else the profile accuracy."""
    if baseline is None:
        return None, None
    if baseline.get("measured_accuracy") is not None:
        return baseline["measured_accuracy"], "measured"
    if baseline.get("profile_accuracy") is not None:
        return baseline["profile_accuracy"], "profile"
    return None, None


# Report card


def pct(value):
    if value is None:
        return "n/a"
    return f"{100.0 * value:.1f}%"


def points(value):
    sign = "+"
    if value < 0:
        sign = "-"
    return f"{sign}{abs(100.0 * value):.1f} points"


def compare_note(winner_accuracy, baseline, what):
    accuracy, source = baseline_accuracy(baseline)
    if accuracy is None:
        return None
    where = ""
    if source == "profile":
        where = ", from its profile"
    gap = winner_accuracy - accuracy
    if abs(gap) < 0.0005:
        return f"It ties {what}, {baseline['model']} ({pct(accuracy)}{where})."
    if gap > 0:
        return f"It beats {what}, {baseline['model']} ({pct(accuracy)}{where}), by {points(gap)[1:]}."
    return f"It trails {what}, {baseline['model']} ({pct(accuracy)}{where}), by {points(gap)[1:]}."


def notes_for(request, rows, winner, target_met, base):
    """Plain-English notes on what the winner is and what it beats."""
    notes = []
    benchmark = request["task"]["benchmark"]
    target = as_fraction(request.get("accuracy_target"))
    if winner is None:
        notes.append("No candidate ran without an error inside the constraints, so there is no winner.")
    else:
        accuracy = winner["measured"]["accuracy"]
        decode = winner["decode_params_b"]
        decode_text = "unknown"
        if decode is not None:
            decode_text = f"{decode:.2f}B"
        notes.append(f"The winner is {winner['description']}. It scores {pct(accuracy)} on held-out {benchmark} "
                     f"with {decode_text} parameters at decode.")
        for key, what in (("host", "its host alone"),
                          ("best_source_at_decode_budget", "the best single model in the decode budget"),
                          ("best_source", "the best single model in the total budget")):
            baseline = base.get(key)
            if baseline is None:
                continue
            if key != "host" and base.get("host") is not None and baseline["hf_id"] == base["host"]["hf_id"]:
                continue
            note = compare_note(accuracy, baseline, what)
            if note is not None:
                notes.append(note)
        if base.get("host_lora") is not None:
            gap = accuracy - base["host_lora"]["measured_accuracy"]
            if winner["program"].get("host") == base["host_lora"]["hf_id"] and is_host_lora(winner["program"]):
                notes.append("Fine-tuning the host with a LoRA is the best option here.")
            else:
                notes.append(f"Against a LoRA on the same host ({pct(base['host_lora']['measured_accuracy'])}), "
                             f"the winner is {points(gap)}.")
        if target is not None:
            if target_met:
                notes.append(f"It meets the accuracy target of {pct(target)}.")
            else:
                notes.append(f"No eligible candidate reached the accuracy target of {pct(target)}; "
                             "the winner is the most accurate eligible one.")

    failed = []
    over = []
    errors = []
    for row in rows:
        if row["status"] == "failed":
            failed.append(row)
        elif row["status"] == "over budget":
            over.append(row)
        if row.get("prediction_error") is not None:
            errors.append(abs(row["prediction_error"]))
    if len(failed) > 0:
        names = []
        for row in failed:
            names.append(f"{row['description']} ({str(row['measured']['error'])[:120]})")
        notes.append(f"{len(failed)} candidate(s) failed: " + "; ".join(names) + ".")
    if len(over) > 0:
        names = []
        for row in over:
            names.append(f"{row['description']} ({', '.join(row['violations'])})")
        notes.append(f"{len(over)} candidate(s) broke a constraint and could not win: " + "; ".join(names) + ".")
    if len(errors) > 0:
        mean = sum(errors) / len(errors)
        notes.append(f"Planner accuracy predictions were off by {100.0 * mean:.1f} points on average "
                     f"over {len(errors)} candidate(s).")
    return notes


def winner_entry(winner, target_met):
    if winner is None:
        return None
    measured = winner["measured"]
    return {"index": winner["index"], "description": winner["description"], "program_text": winner["program_text"],
            "program": winner["program"], "rationale": winner["rationale"], "planner": winner["planner"],
            "accuracy": measured["accuracy"], "decode_params_b": winner["decode_params_b"],
            "prefill_params_b": winner["prefill_params_b"], "total_params_b": winner["total_params_b"],
            "decode_tok_s": measured["decode_tok_s"], "train_seconds": measured["train_seconds"],
            "model_dir": measured["model_dir"], "target_met": target_met}


def report_card(request, profiles, candidates, results):
    """Build the report card from the planned candidates and their measured results."""
    normalized = []
    for candidate in candidates:
        normalized.append(normalize_candidate(candidate))
    candidates = normalized
    rows = []
    matched = match_results(candidates, results)
    for index, candidate in enumerate(candidates):
        rows.append(candidate_row(index, request, profiles, candidate, matched[index]))
    winner, target_met = pick_winner(request, rows)
    base = baselines(request, profiles, rows, winner)
    return {"request": request, "objective": request["objective"], "winner": winner_entry(winner, target_met),
            "candidates": rows, "baselines": base, "notes": notes_for(request, rows, winner, target_met, base)}


# Planning


def baseline_candidates(request, candidates, profiles):
    """A source-model-alone candidate for each host in the plan that the plan does not already run alone.

    These cost no training and give measured host numbers on the same
    held-out items as the composed candidates.
    """
    table = profiles_by_hf_id(profiles)
    benchmark = request["task"]["benchmark"]
    have = []
    hosts = []
    for candidate in candidates:
        program = candidate.get("program") or {}
        host = program.get("host") or program.get("base")
        if is_bare(program) and not program.get("bake") and not candidate.get("bake"):
            have.append(host)
        if host is not None and host not in hosts:
            hosts.append(host)
    out = []
    for host in hosts:
        if host in have:
            continue
        profile = table.get(host) or {}
        params = profile.get("params_b")
        program = {"host": host, "seed": 0, "masks": None, "task_vectors": [], "grafts": [], "heal": None}
        predicted = {"accuracy": profile_score(profile, benchmark), "decode_params_b": params,
                     "prefill_params_b": params, "train_minutes": 0.0}
        out.append({"program": program, "predicted": predicted, "planner": "baseline",
                    "rationale": "Baseline: the host alone, scored on the same held-out items."})
    return out



def plan_candidates(request, profiles, k=6, use_llm=False, plan_fn=None):
    """Call the planner. plan_fn defaults to wsp.planner.plan."""
    if plan_fn is None:
        from wsp.planner import plan as plan_fn
    if not use_llm:
        return list(plan_fn(request, profiles, k=k))
    try:
        parameters = inspect.signature(plan_fn).parameters
    except (TypeError, ValueError):
        parameters = {}
    if "use_llm" not in parameters:
        raise ValueError("this planner has no use_llm option")
    return list(plan_fn(request, profiles, k=k, use_llm=True))


def make(request, profiles, run_candidates, k=6, use_llm=False, plan_fn=None, candidates=None,
         add_baselines=False):
    """Plan candidates for the request, run them, and return the report card.

    run_candidates(list of Candidate) returns a list of measured results
    (see the module doc). candidates skips the planner and runs these.
    add_baselines adds each planned host alone (see baseline_candidates).
    """
    check_request(request)
    profiles = profile_list(profiles)
    if candidates is None:
        candidates = plan_candidates(request, profiles, k=k, use_llm=use_llm, plan_fn=plan_fn)
    candidates = list(candidates)
    if add_baselines:
        candidates = candidates + baseline_candidates(request, candidates, profiles)
    normalized = []
    for candidate in candidates:
        normalized.append(normalize_candidate(candidate))
    results = []
    if len(normalized) > 0:
        results = run_candidates(normalized)
    return report_card(request, profiles, normalized, results)


# Text output


def params_text(value):
    if value is None:
        return "?"
    return f"{value:.2f}B"


def format_plan(request, profiles, candidates):
    """The plan as text: each candidate's program, prediction, and rationale."""
    lines = [f"Plan for {request['name']}: {request.get('description', '')}".rstrip(),
             f"objective {request['objective']}, target {pct(as_fraction(request.get('accuracy_target')))}, "
             f"constraints {json.dumps(request.get('constraints'))}", ""]
    if len(candidates) == 0:
        lines.append("The planner found no candidates.")
    for index, candidate in enumerate(candidates):
        candidate = normalize_candidate(candidate)
        predicted = candidate.get("predicted") or {}
        total = total_params_b(candidate, profiles)
        broken = check_constraints(request, predicted.get("decode_params_b"), total, predicted.get("train_minutes"))
        fits = "fits"
        if len(broken) > 0:
            fits = "breaks " + ", ".join(broken)
        lines.append(f"[{index}] {describe_program(candidate['program'])} ({candidate.get('planner', '?')} planner)")
        lines.append(f"    predicted {pct(as_fraction(predicted.get('accuracy')))}, decode "
                     f"{params_text(predicted.get('decode_params_b'))}, prefill "
                     f"{params_text(predicted.get('prefill_params_b'))}, total {params_text(total)}, train "
                     f"{predicted.get('train_minutes', '?')} min; {fits}")
        if candidate.get("rationale"):
            lines.append("    why: " + candidate["rationale"])
        for text_line in program_text(candidate["program"]).splitlines():
            lines.append("    | " + text_line)
        lines.append("")
    return "\n".join(lines)


def format_report(card):
    """The report card as text."""
    request = card["request"]
    lines = [f"Report card for {request['name']} ({card['objective']})", ""]
    winner = card["winner"]
    if winner is None:
        lines.append("No winner.")
    else:
        lines.append(f"Winner: {winner['description']}")
        lines.append(f"  accuracy {pct(winner['accuracy'])}, decode {params_text(winner['decode_params_b'])}, "
                     f"prefill {params_text(winner['prefill_params_b'])}, total {params_text(winner['total_params_b'])}")
        tok_s = winner["decode_tok_s"]
        if tok_s is not None:
            tok_s = f"{tok_s:.0f}"
        train = winner["train_seconds"]
        if train is not None:
            train = f"{train:.0f}"
        lines.append(f"  decode {tok_s} tok/s, trained in {train} s, saved at {winner['model_dir']}")
        for text_line in winner["program_text"].splitlines():
            lines.append("  | " + text_line)
    lines.append("")
    lines.append("Candidates (predicted -> measured):")
    for row in card["candidates"]:
        predicted = as_fraction((row["predicted"] or {}).get("accuracy"))
        measured = None
        if row["measured"] is not None:
            measured = row["measured"]["accuracy"]
        extra = ""
        if len(row["violations"]) > 0:
            extra = " [" + ", ".join(row["violations"]) + "]"
        if row["status"] == "failed":
            extra = " [" + str(row["measured"]["error"])[:120] + "]"
        lines.append(f"  [{row['index']}] {row['description']}: {pct(predicted)} -> {pct(measured)}, "
                     f"decode {params_text(row['decode_params_b'])}, {row['status']}{extra}")
    lines.append("")
    lines.append("Notes:")
    for note in card["notes"]:
        lines.append("  - " + note)
    return "\n".join(lines) + "\n"


# Command line


def read_results(path, name):
    """Read measured results from a JSON list. Rows tagged with another request name are left out."""
    with open(path) as f:
        rows = json.load(f)
    if isinstance(rows, dict) and "results" in rows:
        rows = rows["results"]
    out = []
    for row in rows:
        if row.get("request") not in (None, name):
            continue
        out.append(row)
    return out


def candidates_from_results(results):
    """The candidates of measured results, in plan order when the results carry an index."""
    ordered = []
    for position, result in enumerate(results):
        index = result.get("index")
        if index is None:
            index = position
        ordered.append((index, position, result["candidate"]))
    ordered.sort(key=lambda entry: (entry[0], entry[1]))
    candidates = []
    for entry in ordered:
        candidates.append(entry[2])
    return candidates


def read_plan(path):
    with open(path) as f:
        data = json.load(f)
    if isinstance(data, dict) and "candidates" in data:
        return data["candidates"]
    return data


def write_json(path, data):
    folder = os.path.dirname(path)
    if folder:
        os.makedirs(folder, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def main(argv=None, plan_fn=None):
    parser = argparse.ArgumentParser(prog="python -m wsp.make", description="Plan and report on a model request.")
    parser.add_argument("request", help="a request JSON file, such as requests/med_mcq_1b.json")
    parser.add_argument("--profiles", default="profiles.json", help="profiles JSON (default profiles.json)")
    parser.add_argument("--k", type=int, default=6, help="number of candidates to plan")
    parser.add_argument("--llm", action="store_true", help="use the LLM planner")
    parser.add_argument("--dry-run", action="store_true", help="print the plan and compile nothing")
    parser.add_argument("--save-plan", default=None, help="write the planned candidates to this JSON file")
    parser.add_argument("--plan", default=None, help="read candidates from this JSON file, not the planner")
    parser.add_argument("--results", default=None, help="measured results JSON; builds the report card")
    parser.add_argument("--out", default=None, help="report card path (default results/make/<name>.json)")
    parser.add_argument("--baselines", action="store_true", help="add each planned host alone to the plan")
    parser.add_argument("--json", action="store_true", help="print JSON, not text")
    args = parser.parse_args(argv)

    try:
        request = load_request(args.request)
    except (OSError, ValueError) as error:
        print(f"{args.request}: {error}", file=sys.stderr)
        return 1
    profiles = load_profiles(args.profiles)

    results = None
    if args.results is not None:
        results = read_results(args.results, request["name"])
    if args.plan is not None:
        candidates = read_plan(args.plan)
    elif results is not None and not args.dry_run:
        # The measured results carry their candidates, so the report card
        # does not depend on the planner giving the same plan again.
        candidates = candidates_from_results(results)
    else:
        candidates = plan_candidates(request, profiles, k=args.k, use_llm=args.llm, plan_fn=plan_fn)
        if args.baselines:
            candidates = candidates + baseline_candidates(request, candidates, profiles)
    if args.save_plan is not None:
        write_json(args.save_plan, {"request": request, "candidates": candidates})

    if args.dry_run:
        if args.json:
            print(json.dumps({"request": request, "candidates": candidates}, indent=2))
        else:
            sys.stdout.write(format_plan(request, profiles, candidates) + "\n")
        return 0

    if args.results is None:
        print("Compiling runs on Modal: modal run modal_make.py --request " + args.request +
              ". Use --dry-run to see the plan, or --results to build a report card from measured results.",
              file=sys.stderr)
        return 2

    card = report_card(request, profile_list(profiles), copy.deepcopy(candidates), results)
    out = args.out
    if out is None:
        out = os.path.join("results", "make", request["name"] + ".json")
    write_json(out, card)
    if args.json:
        print(json.dumps(card, indent=2))
    else:
        sys.stdout.write(format_report(card))
        print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
