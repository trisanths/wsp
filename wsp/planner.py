"""Plan wsp programs for a model request, before anything compiles.

A request says what the model must do and what it may cost. The planner
lists programs that compose existing open models (a host, LoRA, taps,
baking), predicts accuracy and cost for each with the measured rules in
wsp/rules.py, drops the ones that break a constraint, and sorts the rest
by the request's objective.

Request (JSON, in requests/*.json):

    {"name": str,
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

Candidate:

    {"program": Program.to_dict() output,
     "predicted": {"accuracy": float, "decode_params_b": float,
                   "prefill_params_b": float, "train_minutes": float},
     "rationale": str,
     "planner": "rules" | "llm"}

A candidate also carries "kind" (host, lora, tap, two_taps, staged, bake,
or llm), "text" (the program in the wsp text syntax), "meets_target",
"checks" (total runtime params and compile memory), and "notes". A bake
candidate has "bake": the program is the tapped teacher, and the bake
step (wsp.bake) then distills it into a plain host LoRA, so no donor runs
at inference.

Accuracies are in percent. predicted["accuracy"] is None only when no
model in the program has a score or a size guess for the benchmark.

    plan(request, profiles, k=6, use_llm=False) -> list of Candidate

The list is feasible candidates only, sorted by the objective: for
max_accuracy, highest predicted accuracy first; for min_cost_at_target,
the candidates predicted to meet accuracy_target first, smallest decode
params first, then the rest by accuracy. The list also has a .notes
attribute (why the LLM planner did not run, missing pairs data, and so
on). plan_details returns the dropped candidates and their reasons too.

The LLM planner (use_llm=True) reads OPENAI_API_KEY and WSP_PLANNER_MODEL
from the environment or a .env file at the repo root. It asks for up to 4
more programs in the wsp text syntax, parses and checks them, and predicts
them with the same rules. Without a key or model, or when the call fails,
the plan has the rule candidates only, and .notes says why. The key never
goes into a candidate, a note, a log, or an exception message.
"""

import json
import os
import re

from wsp import rules
from wsp.program import Program

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# The wsp.pairs data name that trains for each benchmark.
PAIRS_FOR_BENCHMARK = {
    "mmlu": "mmlu_aux",
    "mmlu_pro": "mmlu_aux",
    "medmcqa": "medmcqa_train",
    "medqa": "medmcqa_train",
    "gsm8k": "gsm8k_train",
    "triviaqa": "trivia_train",
    "trivia": "trivia_train",
}

# Training settings that runs 4 to 7 measured.
# GSM8K: LoRA r64 lr 5e-5, 300 steps, seq 384 (run 4, the fine-tuning bar).
REASONING_LORA = {"rank": 64, "steps": 300, "lr": 5e-5, "batch_size": 8, "seq_len": 384}
# MMLU: LoRA r16 lr 1e-4, 500 steps, seq 256 (runs 6 and 7).
MCQ_LORA = {"rank": 16, "steps": 500, "lr": 1e-4, "batch_size": 8, "seq_len": 256}
# Taps: the round 6 settings.
TAP_TRAIN = {"steps": 500, "batch_size": 8, "seq_len": 256, "branch_lr": 3e-4, "gate_lr": 3e-2}
BAKE = {"rank": 16, "steps": 500, "lr": 1e-4, "batch_size": 8}

# How many strong donors per host get a single-tap candidate.
DONORS_PER_HOST = 3

OBJECTIVES = ("max_accuracy", "min_cost_at_target")
KINDS = ("mcq", "generate")

LLM_MAX_PROGRAMS = 4
LLM_TIMEOUT_SECONDS = 90
LANGUAGE_DOC_CHARS = 14000


class PlanList(list):
    """A list of candidates with a .notes list of strings."""

    def __init__(self, items=None, notes=None):
        super().__init__(items or [])
        self.notes = list(notes or [])


# Requests and profiles


def check_request(request):
    """Return a copy of the request with defaults filled in. Raise ValueError on a bad request."""
    if not isinstance(request, dict):
        raise ValueError("a request is a JSON object")
    task = request.get("task")
    if not isinstance(task, dict) or not task.get("benchmark"):
        raise ValueError("a request needs task.benchmark")
    kind = task.get("kind", "mcq")
    if kind not in KINDS:
        raise ValueError(f"task.kind is one of {', '.join(KINDS)}, got {kind!r}")
    constraints = request.get("constraints") or {}
    objective = request.get("objective", "max_accuracy")
    if objective not in OBJECTIVES:
        raise ValueError(f"objective is one of {', '.join(OBJECTIVES)}, got {objective!r}")
    target = request.get("accuracy_target")
    if target is not None:
        target = rules.as_percent(target)
    out = {
        "name": request.get("name", "request"),
        "task": {"kind": kind, "benchmark": task["benchmark"], "subjects": task.get("subjects")},
        "constraints": {
            "max_decode_params_b": constraints.get("max_decode_params_b"),
            "max_total_params_b": constraints.get("max_total_params_b"),
            "max_train_minutes": constraints.get("max_train_minutes"),
            "gpu": constraints.get("gpu") or "H100",
        },
        "objective": objective,
        "accuracy_target": target,
    }
    return out


def profile_list(profiles):
    """Accept a list of profiles or a dict of key -> profile. Return a list."""
    out = []
    if isinstance(profiles, dict):
        for key in profiles:
            profile = dict(profiles[key])
            if not profile.get("key"):
                profile["key"] = key
            out.append(profile)
        return out
    for item in profiles:
        profile = dict(item)
        if not profile.get("key"):
            profile["key"] = profile.get("hf_id")
        out.append(profile)
    return out


def usable(profile):
    """True when a profile has what the cost rules need."""
    for field in ("hf_id", "params_b", "layers", "hidden"):
        if not profile.get(field):
            return False
    return True


def safe_id(text):
    """A tap id the wsp text syntax accepts: letters, digits, and _ - . /, not starting with -."""
    out = ""
    for char in str(text):
        if char.isalnum() or char in "_.-/":
            out += char
        else:
            out += "_"
    if out == "" or out.startswith("-"):
        out = "t" + out
    return out


def available_pairs():
    try:
        from wsp.pairs import pair_names

        return pair_names()
    except Exception:
        return []


def pairs_name(request):
    """Return (pairs data name or None, note or None) for the request's benchmark."""
    benchmark = request["task"]["benchmark"]
    name = PAIRS_FOR_BENCHMARK.get(benchmark)
    note = None
    if name is None:
        if request["task"]["kind"] == "mcq":
            name = "mmlu_aux"
            note = (f"no pairs data is mapped for {benchmark}; using mmlu_aux, which trains the answer "
                    "format but not in-domain knowledge")
        else:
            return None, f"no pairs data is mapped for {benchmark}, so no LoRA or tap training is planned"
    if benchmark == "mmlu_pro" and name == "mmlu_aux":
        note = "mmlu_aux trains 4-option answers, and MMLU-Pro has up to 10 options"
    if name not in available_pairs():
        missing = f"wsp.pairs has no {name} yet; register it (wsp.pairs.register_pairs) before compiling"
        if note is None:
            note = missing
        else:
            note = note + "; " + missing
    return name, note


# Program builders


def lora_settings(request):
    if rules.is_reasoning(request["task"]["benchmark"]):
        return REASONING_LORA
    return MCQ_LORA


def host_program(host):
    return Program(host=host["hf_id"])


def lora_program(host, request, data):
    settings = lora_settings(request)
    program = Program(host=host["hf_id"])
    program.lora(rank=settings["rank"])
    program.heal(steps=settings["steps"], data=data, lr=settings["lr"], batch_size=settings["batch_size"],
                 seq_len=settings["seq_len"])
    return program


def tap_program(host, donors, data, with_lora=None):
    """host + one tap per donor (read at 70% depth, write at 40% depth), trained on data.

    with_lora: None, or LoRA settings for a staged LoRA after the taps.
    """
    program = Program(host=host["hf_id"])
    at = rules.write_layer(host)
    for donor in donors:
        program.tap(donor["hf_id"], rules.read_layer(donor), at, tap_id=safe_id(donor.get("key") or donor["hf_id"]))
    program.tap_train(data=data, steps=TAP_TRAIN["steps"], batch_size=TAP_TRAIN["batch_size"],
                      seq_len=TAP_TRAIN["seq_len"], branch_lr=TAP_TRAIN["branch_lr"], gate_lr=TAP_TRAIN["gate_lr"])
    if with_lora is not None:
        program.lora(rank=with_lora["rank"])
        program.heal(steps=with_lora["steps"], data=data, lr=with_lora["lr"], batch_size=with_lora["batch_size"],
                     seq_len=with_lora["seq_len"])
    return program


# Prediction


def profiles_by_id(profiles):
    out = {}
    for profile in profiles:
        if profile.get("hf_id"):
            out[profile["hf_id"]] = profile
    return out


def predict(program, request, by_id, bake=None):
    """Predict accuracy and cost for a Program.

    Return (predicted, checks, notes, problems). problems lists reasons the
    program cannot be built or costed (an unknown model, a bad layer).
    """
    benchmark = request["task"]["benchmark"]
    kind = request["task"]["kind"]
    gpu = request["constraints"]["gpu"]
    notes = []
    problems = []

    host = by_id.get(program.host)
    if host is None:
        return None, None, notes, ["host " + program.host + " has no profile"]
    if not usable(host):
        return None, None, notes, ["host " + program.host + " lacks hidden or layers in its profile"]
    if len(program.grafts) > 0 or len(program.task_vectors) > 0 or program.masks is not None:
        problems.append("the planner predicts only host, LoRA, and tap programs (no grafts, masks, or task vectors)")

    host_score, host_note = rules.score_of(host, benchmark)
    if host_note != "measured":
        notes.append("host score: " + host_note)

    # Each distinct donor runs once per prompt, up to the deepest layer any tap reads.
    deepest = {}
    donor_order = []
    branch_params = 0
    for tap in program.taps:
        donor = by_id.get(tap.donor)
        if donor is None:
            problems.append("donor " + tap.donor + " has no profile")
            continue
        if not usable(donor):
            problems.append("donor " + tap.donor + " lacks hidden or layers in its profile")
            continue
        if tap.donor_layer >= int(donor["layers"]):
            problems.append(f"tap reads layer {tap.donor_layer} but {tap.donor} has {donor['layers']} layers")
            continue
        if tap.at >= int(host["layers"]):
            problems.append(f"tap writes layer {tap.at} but the host has {host['layers']} layers")
            continue
        if tap.donor_layer < rules.MIN_READ_DEPTH * int(donor["layers"]):
            notes.append(f"{tap.donor} is read at layer {tap.donor_layer}, before 57% depth, where run 7 "
                         "found nothing to read")
        branch_params += rules.tap_branch_params(host, donor, tap.attn_dim)
        if tap.donor not in deepest:
            donor_order.append(tap.donor)
            deepest[tap.donor] = tap.donor_layer
        elif tap.donor_layer > deepest[tap.donor]:
            deepest[tap.donor] = tap.donor_layer

    prefix_b = 0.0
    donor_full_b = 0.0
    for donor_id in donor_order:
        prefix_b += rules.donor_prefix_params_b(by_id[donor_id], deepest[donor_id])
        donor_full_b += float(by_id[donor_id]["params_b"])

    # Accuracy.
    accuracy = host_score
    if len(donor_order) > 0:
        best = None
        for donor_id in donor_order:
            value, note = rules.tap_recovery(host, by_id[donor_id], benchmark, kind)
            notes.append("tap " + by_id[donor_id].get("key", donor_id) + ": " + note)
            if value is not None and (best is None or value > best):
                best = value
        accuracy = best
        if accuracy is not None and len(donor_order) > 1:
            accuracy = accuracy + rules.EXTRA_TAP_DELTA
            notes.append("several donors score near the best single tap, -1.0 (run 8)")
    if program.lora_spec is not None and host_score is not None:
        gain, note = rules.lora_gain(host, benchmark, kind)
        if len(donor_order) == 0:
            accuracy = host_score + gain
            notes.append("LoRA: +" + str(round(gain, 1)) + ", " + note)
        elif rules.is_reasoning(benchmark):
            accuracy = host_score + gain
            notes.append("staged LoRA on a reasoning task: LoRA alone counts, the tap adds nothing (run 6)")
        elif kind == "generate":
            accuracy = accuracy + gain
            notes.append("staged LoRA on free-form answers: +" + str(round(gain, 1)) + " (a prior)")
        else:
            accuracy = accuracy + rules.STAGED_LORA_GAIN
            notes.append("staged tap then LoRA: +1.0 (run 7)")
    if bake is not None and accuracy is not None and host_score is not None:
        accuracy = host_score + rules.BAKE_RETENTION * (accuracy - host_score)
        notes.append("baking keeps half of the tap gain (a prior; no bake run is measured yet)")
    if accuracy is not None:
        accuracy = round(max(0.0, min(100.0, accuracy)), 1)

    # Cost.
    host_decode_b = rules.decode_params_b(host)
    lora_b = 0.0
    if program.lora_spec is not None and len(program.taps) > 0 and program.lora_spec.rank is not None:
        # With taps the LoRA stays a separate module, so it runs at decode.
        lora_b = rules.lora_params(host, program.lora_spec.rank) / 1e9
    if bake is not None:
        decode_b = host_decode_b
        prefill_b = host_decode_b
        runtime_total_b = float(host["params_b"])
    else:
        decode_b = host_decode_b + branch_params / 1e9 + lora_b
        prefill_b = host_decode_b + prefix_b
        runtime_total_b = float(host["params_b"]) + prefix_b + branch_params / 1e9 + lora_b

    minutes = 0.0
    if program.tap_train_spec is not None and len(donor_order) > 0:
        minutes += rules.tap_train_minutes(host, prefix_b, program.tap_train_spec.steps, gpu)
    if program.lora_spec is not None and program.heal_spec is not None:
        if len(donor_order) > 0:
            # The staged LoRA trains with the taps reading the donors.
            minutes += rules.tap_train_minutes(host, prefix_b, program.heal_spec.steps, gpu)
        else:
            minutes += rules.lora_train_minutes(host, program.heal_spec.steps, gpu)
    elif program.heal_spec is not None:
        minutes += rules.heal_train_minutes(host, program.heal_spec.steps, gpu)
    if bake is not None:
        # The teacher (host and donors) and the student both run every bake step.
        minutes += rules.tap_train_minutes(host, prefix_b, bake["steps"], gpu)

    loaded_b = float(host["params_b"]) + donor_full_b
    if bake is not None:
        loaded_b += float(host["params_b"])
    predicted = {"accuracy": accuracy, "decode_params_b": round(decode_b, 4),
                 "prefill_params_b": round(prefill_b, 4), "train_minutes": round(minutes, 2)}
    checks = {"runtime_total_params_b": round(runtime_total_b, 4),
              "compile_memory_gb": round(rules.memory_gb(loaded_b), 1),
              "gpu": gpu, "gpu_memory_gb": rules.GPU_MEMORY_GB.get(gpu, rules.GPU_MEMORY_GB["H100"]),
              "branch_params": branch_params, "donor_prefix_params_b": round(prefix_b, 4)}
    if not rules.fits_gpu(loaded_b, gpu):
        problems.append(f"compile needs about {checks['compile_memory_gb']} GB, more than one {gpu} holds")
    return predicted, checks, notes, problems


def constraint_problems(predicted, checks, request):
    """Reasons a predicted candidate breaks the request's constraints."""
    out = []
    limits = request["constraints"]
    small = 1e-9
    if limits["max_decode_params_b"] is not None:
        if predicted["decode_params_b"] > limits["max_decode_params_b"] + small:
            out.append(f"decode params {predicted['decode_params_b']}B over the {limits['max_decode_params_b']}B limit")
    if limits["max_total_params_b"] is not None:
        if checks["runtime_total_params_b"] > limits["max_total_params_b"] + small:
            out.append(f"runtime params {checks['runtime_total_params_b']}B over the "
                       f"{limits['max_total_params_b']}B total limit")
    if limits["max_train_minutes"] is not None:
        if predicted["train_minutes"] > limits["max_train_minutes"] + small:
            out.append(f"train time {predicted['train_minutes']} min over the {limits['max_train_minutes']} min limit")
    return out


def meets_target(predicted, request):
    target = request["accuracy_target"]
    if target is None:
        return True
    if predicted["accuracy"] is None:
        return False
    return predicted["accuracy"] >= target


def format_text(program):
    try:
        from wsp.lang import format_program

        return format_program(program)
    except Exception:
        return None


def make_candidate(program, request, by_id, kind, rationale, planner="rules", bake=None, extra_notes=None):
    """Build a candidate dict. Its "problems" list is empty when it is feasible."""
    predicted, checks, notes, problems = predict(program, request, by_id, bake)
    if extra_notes is not None:
        notes = list(extra_notes) + notes
    candidate = {"program": program.to_dict(), "predicted": predicted, "rationale": rationale,
                 "planner": planner, "kind": kind, "text": format_text(program), "checks": checks,
                 "notes": notes, "problems": list(problems)}
    if bake is not None:
        candidate["bake"] = dict(bake)
    if predicted is not None:
        candidate["problems"] += constraint_problems(predicted, checks, request)
        candidate["meets_target"] = meets_target(predicted, request)
    else:
        candidate["meets_target"] = False
    return candidate


# Rationales


def fmt(value):
    if value is None:
        return "unknown"
    return str(round(value, 1))


def host_rationale(host, request, score):
    return (f"Rule: the host alone, no training. {host['key']} ({fmt(rules.decode_params_b(host))}B at decode) "
            f"scores {fmt(score)} on {request['task']['benchmark']}. This is the bar every composed program "
            "must beat, and the cheapest option when it already meets the target.")


def lora_rationale(host, request, data):
    benchmark = request["task"]["benchmark"]
    if rules.is_reasoning(benchmark):
        return (f"Rule: LoRA r64 lr 5e-5 is the fine-tuning bar on reasoning tasks (run 4 and run 5: Llama-1B "
                f"47.2 from 25.0 on GSM8K; full fine-tuning at 300 steps is worse). Taps carry no multi-step "
                f"reasoning (run 5), so a host LoRA on {data} is the main plan for {benchmark}.")
    return (f"Rule: a host LoRA on {data} teaches the answer format but cannot add missing knowledge (run 7: "
            f"Llama-1B MMLU 47.7 with LoRA vs 68.3 with a tap). Expect about +3 on hosts up to 1.5B and about 0 "
            f"on hosts of 7B or more (run 9).")


def tap_rationale(host, donor, request, data):
    benchmark = request["task"]["benchmark"]
    if rules.is_reasoning(benchmark):
        return (f"Rule: a tap from {donor['key']} into {host['key']} is listed for comparison only. On GSM8K a "
                "tap scores the same as its mismatched-message control (run 5), so it is not the main plan.")
    if request["task"]["kind"] == "generate":
        return (f"Rule: a tap from {donor['key']} into {host['key']} on free-form answers. Run 7 TriviaQA: the "
                "tap scored 31 vs host 38, so taps are predicted to hurt here.")
    return (f"Rule: a knowledge tap. {host['key']} reads {donor['key']} at layer {rules.read_layer(donor)} of "
            f"{donor['layers']} (about 70% depth, run 7) through a gated branch at host layer "
            f"{rules.write_layer(host)} (about 40% depth), trained 500 steps on {data}. Small hosts recover about "
            "0.9 of the donor-host gap (run 7: Llama-1B 44.7 to 68.3 held out with Qwen-7B; run 8: Qwen-0.5B + "
            "Gemma-9B equals Gemma-9B), and strong hosts gain nothing (run 9). Decode cost stays at the host; "
            "each prompt also runs the donor up to its read layer.")


def two_tap_rationale(host, donors):
    names = []
    for donor in donors:
        names.append(donor["key"])
    return (f"Rule: two taps ({' and '.join(names)}) into {host['key']}. Run 8 found two taps into a weak host "
            "score near the better single tap (-0.9 to +0.8, and worse with an unstable donor), so this costs "
            "more prefill for no predicted gain. It is listed to measure complementarity.")


def staged_rationale(host, donor, request):
    if rules.is_reasoning(request["task"]["benchmark"]):
        return (f"Rule: tap {donor['key']} first, freeze it, then LoRA. On GSM8K the LoRA does the work and the "
                "tap gate stays near 0 (run 6), so this predicts the same as LoRA alone at a higher cost.")
    return (f"Rule: staged training. Tap {donor['key']} into {host['key']} first, freeze the tap, then train a "
            "host LoRA r16 with the tap reading the donor. Run 7: 68.9 vs 67.9 for the tap alone. Joint "
            "training collapses the tap (run 6), so the compiler always stages them.")


def bake_rationale(host, donor, request):
    return (f"Rule: the request allows no donor at runtime ({request['constraints']['max_total_params_b']}B "
            f"total). Compile the {donor['key']} tap as a teacher, then distill it into a plain {host['key']} "
            "LoRA with wsp.bake (mode answer for MCQ), so inference runs the host alone. No bake run is "
            "measured yet; keeping half of the tap gain is a prior.")


# Rule candidates


def strong_donors(host, request, profiles):
    """Donors that beat the host on the benchmark by MIN_DONOR_MARGIN, best first, at most DONORS_PER_HOST."""
    benchmark = request["task"]["benchmark"]
    host_score, _ = rules.score_of(host, benchmark)
    if host_score is None:
        return []
    rows = []
    for donor in profiles:
        if donor["hf_id"] == host["hf_id"] or not usable(donor):
            continue
        score, _ = rules.score_of(donor, benchmark)
        if score is None or score - host_score < rules.MIN_DONOR_MARGIN:
            continue
        rows.append((score, donor))
    rows.sort(key=lambda row: -row[0])
    out = []
    for score, donor in rows[:DONORS_PER_HOST]:
        out.append(donor)
    return out


def rule_candidates(request, profiles, by_id, notes):
    """Every rule candidate, feasible or not."""
    benchmark = request["task"]["benchmark"]
    limit = request["constraints"]["max_decode_params_b"]
    data, data_note = pairs_name(request)
    if data_note is not None:
        notes.append(data_note)
    if request["task"]["subjects"]:
        notes.append("the rules use whole-benchmark scores, so the subject filter does not change predictions; "
                     "the measured run should score the chosen subjects")
    extra = []
    if data_note is not None:
        extra.append(data_note)

    hosts = []
    for profile in profiles:
        if not usable(profile):
            notes.append(f"{profile.get('key') or profile.get('hf_id')}: profile lacks params, layers, or hidden")
            continue
        if limit is not None and rules.decode_params_b(profile) > limit + 1e-9:
            continue
        hosts.append(profile)
    if len(hosts) == 0:
        notes.append("no model fits max_decode_params_b")

    out = []
    for host in hosts:
        score, _ = rules.score_of(host, benchmark)
        out.append(make_candidate(host_program(host), request, by_id, "host", host_rationale(host, request, score)))
        if data is None:
            continue
        out.append(make_candidate(lora_program(host, request, data), request, by_id, "lora",
                                  lora_rationale(host, request, data), extra_notes=extra))
        donors = strong_donors(host, request, profiles)
        tap_cands = []
        for donor in donors:
            candidate = make_candidate(tap_program(host, [donor], data), request, by_id, "tap",
                                       tap_rationale(host, donor, request, data), extra_notes=extra)
            out.append(candidate)
            tap_cands.append((candidate, donor))
        if len(donors) >= 2:
            out.append(make_candidate(tap_program(host, donors[:2], data), request, by_id, "two_taps",
                                      two_tap_rationale(host, donors[:2]), extra_notes=extra))
        if len(donors) >= 1:
            # Stage a LoRA after the tap with the best predicted accuracy.
            best = None
            for candidate, donor in tap_cands:
                if candidate["predicted"] is None or candidate["predicted"]["accuracy"] is None:
                    continue
                if best is None or candidate["predicted"]["accuracy"] > best[0]["predicted"]["accuracy"]:
                    best = (candidate, donor)
            if best is not None:
                donor = best[1]
                staged = tap_program(host, [donor], data, with_lora=lora_settings(request))
                out.append(make_candidate(staged, request, by_id, "staged", staged_rationale(host, donor, request),
                                          extra_notes=extra))
                # Bake when the request forbids running this donor at inference.
                total_limit = request["constraints"]["max_total_params_b"]
                if total_limit is not None and not rules.is_reasoning(benchmark):
                    if best[0]["checks"]["runtime_total_params_b"] > total_limit + 1e-9:
                        bake = dict(BAKE)
                        bake["data"] = data
                        bake["mode"] = "answer" if request["task"]["kind"] == "mcq" else "sequence"
                        out.append(make_candidate(tap_program(host, [donor], data), request, by_id, "bake",
                                                  bake_rationale(host, donor, request), bake=bake,
                                                  extra_notes=extra))
    return out


# Sorting


def accuracy_key(candidate):
    value = candidate["predicted"]["accuracy"]
    if value is None:
        return -1.0
    return value


def sort_candidates(candidates, request):
    """Sort feasible candidates by the objective (see the module docstring)."""
    def by_accuracy(candidate):
        predicted = candidate["predicted"]
        return (-accuracy_key(candidate), predicted["decode_params_b"], predicted["prefill_params_b"],
                predicted["train_minutes"])

    def by_cost(candidate):
        predicted = candidate["predicted"]
        return (predicted["decode_params_b"], predicted["prefill_params_b"], predicted["train_minutes"],
                -accuracy_key(candidate))

    if request["objective"] == "max_accuracy":
        return sorted(candidates, key=by_accuracy)
    meeting = []
    others = []
    for candidate in candidates:
        if candidate["meets_target"]:
            meeting.append(candidate)
        else:
            others.append(candidate)
    return sorted(meeting, key=by_cost) + sorted(others, key=by_accuracy)


def program_signature(candidate):
    signature = json.dumps(candidate["program"], sort_keys=True)
    if candidate.get("bake") is not None:
        signature += json.dumps(candidate["bake"], sort_keys=True)
    return signature


def plan_details(request, profiles, k=6, use_llm=False, env=None, repo_root=None):
    """Plan and return {"candidates", "dropped", "notes", "request"}.

    candidates: the top k feasible candidates, sorted by the objective.
    dropped: infeasible candidates, each with its "problems".
    """
    request = check_request(request)
    profiles = profile_list(profiles)
    by_id = profiles_by_id(profiles)
    notes = []
    candidates = rule_candidates(request, profiles, by_id, notes)
    if use_llm:
        candidates += llm_candidates(request, profiles, by_id, notes, env=env, repo_root=repo_root)

    seen = set()
    feasible = []
    dropped = []
    for candidate in candidates:
        signature = program_signature(candidate)
        if signature in seen:
            continue
        seen.add(signature)
        if len(candidate["problems"]) > 0 or candidate["predicted"] is None:
            dropped.append(candidate)
        else:
            feasible.append(candidate)
    ordered = sort_candidates(feasible, request)
    return {"candidates": ordered[:k], "dropped": dropped, "notes": notes, "request": request}


def plan(request, profiles, k=6, use_llm=False, env=None, repo_root=None):
    """Return up to k feasible candidates, sorted by the objective, as a PlanList with .notes.

    env: a dict to read OPENAI_API_KEY and WSP_PLANNER_MODEL from, or None
    for os.environ and the repo .env file. repo_root: where .env and
    docs/language.md live (default: this repo).
    """
    details = plan_details(request, profiles, k, use_llm, env, repo_root)
    return PlanList(details["candidates"], details["notes"])


# LLM planner


def read_env_file(path):
    """Parse a .env file: KEY=VALUE lines, # comments, optional export and quotes. Return a dict."""
    out = {}
    if not os.path.exists(path):
        return out
    with open(path) as f:
        lines = f.read().splitlines()
    for line in lines:
        line = line.strip()
        if line == "" or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export "):].strip()
        name, value = line.split("=", 1)
        name = name.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        out[name] = value
    return out


def llm_settings(env=None, repo_root=None):
    """Return {"key", "model", "base_url"}. Missing values are None. The environment wins over .env."""
    if repo_root is None:
        repo_root = REPO_ROOT
    if env is None:
        file_values = read_env_file(os.path.join(repo_root, ".env"))
        env = {}
        for name in file_values:
            env[name] = file_values[name]
        for name in ("OPENAI_API_KEY", "WSP_PLANNER_MODEL", "OPENAI_BASE_URL"):
            if os.environ.get(name):
                env[name] = os.environ[name]
    base_url = env.get("OPENAI_BASE_URL") or "https://api.openai.com/v1"
    return {"key": env.get("OPENAI_API_KEY") or None, "model": env.get("WSP_PLANNER_MODEL") or None,
            "base_url": base_url.rstrip("/")}


def http_post(url, headers, body, timeout):
    """POST JSON and return the parsed JSON reply. Tests replace this function."""
    import urllib.request

    data = json.dumps(body).encode("utf-8")
    request = urllib.request.Request(url, data=data, headers=headers, method="POST")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def profiles_table(profiles, benchmark):
    lines = ["key | hf_id | params_b | active_params_b | layers | hidden | family | " + benchmark + " score"]
    for profile in profiles:
        if not usable(profile):
            continue
        score, note = rules.score_of(profile, benchmark)
        score_text = fmt(score)
        if note != "measured":
            score_text += " (" + note + ")"
        lines.append(" | ".join([str(profile.get("key")), str(profile["hf_id"]), str(profile["params_b"]),
                                 str(profile.get("active_params_b")), str(profile["layers"]),
                                 str(profile["hidden"]), str(profile.get("family")), score_text]))
    return "\n".join(lines)


def language_reference(repo_root):
    path = os.path.join(repo_root, "docs", "language.md")
    if not os.path.exists(path):
        return "(docs/language.md is missing)"
    with open(path) as f:
        text = f.read()
    return text[:LANGUAGE_DOC_CHARS]


def llm_messages(request, profiles, repo_root, data):
    system = ("You plan wsp programs. wsp composes existing open models into one model: a host, an optional "
              "LoRA, and taps that read frozen donors. Answer with up to 4 programs in the wsp text syntax, each "
              "in its own ```wsp fenced block, then one short line per program saying why. Use only the host and "
              "donor hf_id values from the table. Use only host, seed, lora, heal, tap, and tap_train statements. "
              "A tap needs a tap_train line; a lora needs a heal line. Layers start at 0 and must be below the "
              "model's layer count.")
    user = "\n\n".join([
        "Request:\n" + json.dumps(request, indent=2),
        "Training pairs name for this benchmark: " + str(data),
        "Models:\n" + profiles_table(profiles, request["task"]["benchmark"]),
        rules.summary_text(),
        "Language reference:\n" + language_reference(repo_root),
        "Propose programs the rule planner may miss: other layer choices, other donors, other training settings.",
    ])
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def extract_programs(text):
    """Return the program texts in fenced blocks (```wsp, or a plain ``` block with a host line)."""
    out = []
    for match in re.finditer(r"```([A-Za-z]*)[ \t]*\n(.*?)```", text, re.DOTALL):
        tag = match.group(1).lower()
        body = match.group(2)
        if tag == "wsp" or (tag == "" and re.search(r"^\s*host\s", body, re.MULTILINE)):
            out.append(body)
    return out[:LLM_MAX_PROGRAMS]


def scrub(text, secret):
    if secret and text:
        return text.replace(secret, "[key]")
    return text


def llm_reply_text(settings, messages):
    url = settings["base_url"] + "/chat/completions"
    headers = {"Content-Type": "application/json", "Authorization": "Bearer " + settings["key"]}
    body = {"model": settings["model"], "messages": messages}
    reply = http_post(url, headers, body, LLM_TIMEOUT_SECONDS)
    return reply["choices"][0]["message"]["content"]


def llm_candidates(request, profiles, by_id, notes, env=None, repo_root=None):
    """Ask the LLM for up to 4 programs, parse, check, and predict them. Return candidates."""
    from wsp.lang import LangError, parse

    if repo_root is None:
        repo_root = REPO_ROOT
    settings = llm_settings(env, repo_root)
    if settings["key"] is None:
        notes.append("llm planner skipped: OPENAI_API_KEY is not set")
        return []
    if settings["model"] is None:
        notes.append("llm planner skipped: WSP_PLANNER_MODEL is not set")
        return []
    data, _ = pairs_name(request)
    try:
        text = llm_reply_text(settings, llm_messages(request, profiles, repo_root, data))
    except Exception as error:
        # Only the error type and HTTP status go into the note, never the
        # message, since a provider can echo part of the key.
        status = getattr(error, "code", None)
        reason = type(error).__name__
        if status is not None:
            reason += " " + str(status)
        notes.append("llm planner failed: " + reason + "; using rule candidates only")
        return []
    if not isinstance(text, str):
        notes.append("llm planner failed: the reply had no text; using rule candidates only")
        return []

    out = []
    blocks = extract_programs(text)
    if len(blocks) == 0:
        notes.append("llm planner returned no ```wsp program blocks")
    for index in range(len(blocks)):
        try:
            program = parse(blocks[index])
        except LangError as error:
            notes.append(scrub(f"llm program {index + 1} does not parse: {error}", settings["key"]))
            continue
        problem = llm_program_problem(program)
        if problem is not None:
            notes.append(f"llm program {index + 1} rejected: {problem}")
            continue
        try:
            Program.from_dict(program.to_dict())
        except Exception as error:
            notes.append(f"llm program {index + 1} rejected: {type(error).__name__}")
            continue
        rationale = ("LLM planner proposal, predicted with the same measured rules as the rule candidates "
                     "(wsp/rules.py). Check its prediction notes before trusting it.")
        candidate = make_candidate(program, request, by_id, "llm", rationale, planner="llm")
        out.append(candidate)
    return out


def llm_program_problem(program):
    """Return why a parsed LLM program cannot run as planned, or None."""
    if len(program.taps) > 0 and program.tap_train_spec is None:
        return "a tap needs a tap_train line"
    if len(program.taps) == 0 and program.tap_train_spec is not None:
        return "a tap_train line needs a tap"
    if program.lora_spec is not None and program.heal_spec is None:
        return "a lora needs a heal line"
    return None
