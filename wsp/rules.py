"""Measured facts about composition, as data and small prediction functions.

The planner (wsp/planner.py) uses these rules to predict what a program
will score and what it will cost before anything compiles. Every number
here comes from results/README.md (runs 1 to 9, baselines, stitch study)
or from round 10 per-model scores (results/round10_rows.json). A number
that no run measured is called a prior in its comment and in the note it
returns.

Scores are accuracies in percent (0 to 100). A profile score of 1.0 or
less is read as a fraction and turned into percent.

Profile (one per model, built by wsp/profiles.py):

    {"key", "hf_id", "family", "params_b", "active_params_b", "layers",
     "hidden", "tokenizer_family", "scores": {"<benchmark>": float, ...},
     "decode_tok_s": float or null, "source": str}

Main functions:

    score_of(profile, benchmark)          -> (score or None, note)
    subject_score(profile, benchmark, subjects) -> (score or None, items)
    tap_recovery(host, donor, benchmark)  -> (accuracy or None, note)
    lora_gain(host, benchmark)            -> (points, note)
    read_layer(profile), write_layer(profile)
    tap_branch_params(host, donor, attn_dim=256)
    donor_prefix_params_b(donor, donor_layer)
    tap_train_minutes, lora_train_minutes, heal_train_minutes
    stacker_gain(benchmark), ensemble_accuracy(scores, benchmark)
    summary_text()                        a plain summary for the LLM planner
"""

import math

# Benchmarks where the answer needs several steps of work after the
# prompt. Run 5 (GSM8K): a tap scores the same as its mismatched-message
# control, so prefill taps carry no reasoning. Run 6: a tap adds nothing
# on top of LoRA on GSM8K.
REASONING_BENCHMARKS = ["gsm8k", "math", "math500", "gsm_hard", "aime", "humaneval", "mbpp", "bbh"]

# When a profile has no score for a benchmark, use the score of a close
# benchmark and say so in the note. The first one found wins. "mmlu"
# falls back to the dev items, then the held-out items, like
# wsp.profiles.get_score.
SCORE_FALLBACK = {
    "mmlu": ["mmlu_dev", "mmlu_heldout"],
    "medqa": ["medmcqa"],
    "mmlu_pro": ["mmlu", "mmlu_dev", "mmlu_heldout"],
    "mmlu_dev": ["mmlu", "mmlu_heldout"],
    "mmlu_heldout": ["mmlu", "mmlu_dev"],
}

# A subject score (profile["subject_scores"][benchmark][subject], from the
# round 10 dev items) is used only when the chosen subjects cover at least
# this many items. Fewer items are too noisy to rank models.
MIN_SUBJECT_ITEMS = 30

# Size curves fit on round 10 scores (10 models, 0.5B to 33B):
# score = a + b * ln(params_b). Only a rough guess for a model with no
# score at all.
SIZE_CURVES = {
    "mmlu": (51.21, 8.49),
    "medmcqa": (40.84, 7.21),
}

# Tap rules.
# Run 7 layer map: the donor is readable near 70% depth (layer 20 of 28
# best) and gives nothing before 57% (layer 16 of 28). Runs 7 and 12 wrote
# into host layers at about 40% depth (Llama-1B layer 6 of 16 was best).
READ_DEPTH = 0.71
MIN_READ_DEPTH = 0.57
WRITE_DEPTH = 0.4

# Recovery fraction: the share of the donor-host gap a tap closes.
# Small hosts (runs 6 to 8, 0.5B and 1.2B hosts, 1.5B to 14B donors): the
# clean cells give 0.79 to 1.0, mean 0.91, so 0.9.
SMALL_HOST_RECOVERY = 0.9
SMALL_HOST_MAX_B = 1.5
# Strong hosts (run 9, 7B to 14B hosts): no gain from equal or weaker
# donors, and Gemma-9B + Qwen-14B closed only 0.07 of its gap. So 0.1.
LARGE_HOST_RECOVERY = 0.1
LARGE_HOST_MIN_B = 7.0
# Donors under 1.5B: only LFM2.5-1.2B was tried and it was weak on the raw
# prompt. Half recovery is a prior, not a measurement.
SMALL_DONOR_MAX_B = 1.5
SMALL_DONOR_FACTOR = 0.5
# A donor must beat the host by this many points before a tap is worth it
# (run 9: equal donors do not help).
MIN_DONOR_MARGIN = 1.0

# Recovery fractions measured on MMLU dev or held-out items for exact
# host and donor pairs, by Hugging Face id. They override the rule.
MEASURED_RECOVERY = [
    # Runs 6, 7, 8: 0.85, 0.90, 0.80.
    ("unsloth/Llama-3.2-1B-Instruct", "Qwen/Qwen2.5-7B-Instruct", 0.85, "runs 6 to 8"),
    # Runs 7 and 8: 0.84 and 0.74.
    ("unsloth/Llama-3.2-1B-Instruct", "Qwen/Qwen2.5-14B-Instruct", 0.79, "runs 7 and 8"),
    ("unsloth/Llama-3.2-1B-Instruct", "unsloth/gemma-2-9b-it", 1.0, "run 8 (72.9 vs donor 72.6)"),
    ("unsloth/Llama-3.2-1B-Instruct", "microsoft/Phi-3.5-mini-instruct", 0.9, "run 8"),
    # Unstable donor at layer 22 (seeds 49.2 and 58.0).
    ("unsloth/Llama-3.2-1B-Instruct", "unsloth/Meta-Llama-3.1-8B-Instruct", 0.32, "run 8, unstable"),
    ("unsloth/Llama-3.2-1B-Instruct", "Qwen/Qwen2.5-1.5B-Instruct", 0.94, "run 7"),
    # Likely a poor read layer.
    ("unsloth/Llama-3.2-1B-Instruct", "Qwen/Qwen2.5-3B-Instruct", 0.56, "run 7, poor read layer"),
    ("Qwen/Qwen2.5-0.5B-Instruct", "Qwen/Qwen2.5-14B-Instruct", 0.88, "run 8 (74.8 vs donor 78.5)"),
    ("Qwen/Qwen2.5-0.5B-Instruct", "unsloth/gemma-2-9b-it", 1.0, "run 8 (72.7 vs donor 72.6)"),
    ("Qwen/Qwen2.5-0.5B-Instruct", "Qwen/Qwen2.5-7B-Instruct", 0.91, "run 8"),
    ("unsloth/gemma-2-9b-it", "Qwen/Qwen2.5-14B-Instruct", 0.07, "run 9"),
]

# Run 7 TriviaQA: tap seeds 25.0, 34.5, 34.3 (mean 31.3) vs host 37.6.
# A tap hurts free-form short answers.
TAP_GENERATE_DELTA = -6.3

# Run 8: two taps into a weak host score near the better single tap. The
# four pairs gave -0.9, +0.8, -2.2, -2.7 against the best single tap.
EXTRA_TAP_DELTA = -1.0

# Run 7: tap 500 steps, freeze, then LoRA r16 500 steps: 68.9 vs 67.9.
STAGED_LORA_GAIN = 1.0

# LoRA gains (points over the host).
# Knowledge MCQ: Llama-1B r16 +3.0 held out (47.7 vs 44.7), +2.8 dev (run 6).
# Qwen-14B r16 -0.4 (run 9). LoRA cannot add missing knowledge.
LORA_MCQ_SMALL_GAIN = 3.0
LORA_MCQ_LARGE_GAIN = 0.0
# Free-form short answers: run 7 TriviaQA r16 41.1 vs host 37.6.
LORA_GENERATE_GAIN = 3.5
# Reasoning (GSM8K raw prompt), measured per host, LoRA r64 lr 5e-5:
# Llama-1B 47.2 vs 25.0 (run 5); Qwen-0.5B 36.3 vs 33.5 (run 4).
MEASURED_LORA_REASONING = {
    "unsloth/Llama-3.2-1B-Instruct": (22.2, "run 5: LoRA r64 47.2 vs host 25.0 on GSM8K"),
    "Qwen/Qwen2.5-0.5B-Instruct": (2.8, "run 4: LoRA r64 36.3 vs host 33.5 on GSM8K"),
}
# For a host with no measurement, the smaller of the two measured gains.
LORA_REASONING_DEFAULT = 2.8

# Baking (distill a tapped host into a host LoRA) has no measured result
# yet. Keeping half of the tap gain is a prior.
BAKE_RETENTION = 0.5

# Output-level combining of several models (round 10 per-item scores):
# probability average +0.5 to +1.3 over the best model, a learned stacker
# +1.2 on MMLU and +1.6 on MedMCQA. Oracle coverage 90 to 94%.
STACKER_GAIN = {"mmlu": 1.2, "medmcqa": 1.6}
STACKER_GAIN_DEFAULT = 1.2
AVERAGE_GAIN = (0.5, 1.3)
ORACLE_COVERAGE = (90.0, 94.0)

# Compile times on one H100 (results/README.md and the round logs).
# Tap: 500 steps, Llama-1B host + Qwen-7B donor read at layer 20, about
# 100 s with loads and scoring. LoRA: 300 steps on Llama-1B, 40 to 53 s.
# Graft heal: 300 steps, 21 to 27 s.
TAP_SECONDS_REF = 100.0
TAP_STEPS_REF = 500
# 1.24B host + 7.6B * 21 / 28 donor prefix.
TAP_RUN_PARAMS_REF = 1.24 + 7.6 * 21.0 / 28.0
LORA_SECONDS_REF = 47.0
LORA_STEPS_REF = 300
LORA_HOST_REF = 1.24
HEAL_SECONDS_REF = 24.0
HEAL_STEPS_REF = 300
# A floor for model loads and scoring.
MIN_TRAIN_SECONDS = 20.0
# No H200 timing was measured; use the H100 numbers.
GPU_SPEED = {"H100": 1.0, "H200": 1.0}
GPU_MEMORY_GB = {"H100": 80.0, "H200": 141.0}
# bf16 weights are 2 bytes per parameter. Leave room for activations.
MEMORY_USE_FRACTION = 0.85
MEMORY_OVERHEAD_GB = 4.0


def as_percent(value):
    """Return a score in percent. A value of 1.0 or less is read as a fraction."""
    if value is None:
        return None
    value = float(value)
    if value <= 1.0:
        return value * 100.0
    return value


def is_reasoning(benchmark):
    return benchmark in REASONING_BENCHMARKS


def decode_params_b(profile):
    """Parameters that run per decoded token: active params for an MoE, else all."""
    active = profile.get("active_params_b")
    if active is None:
        return float(profile["params_b"])
    return float(active)


def score_of(profile, benchmark):
    """Return (score in percent or None, note) for a profile on a benchmark.

    Order: the profile's own score, then a close benchmark's score
    (SCORE_FALLBACK), then a size curve for MCQ benchmarks. None when
    nothing fits. A profile can carry "score_notes" (benchmark -> note),
    which the planner sets when it puts a subject score in place of the
    whole-benchmark score.
    """
    scores = profile.get("scores") or {}
    if scores.get(benchmark) is not None:
        score_notes = profile.get("score_notes") or {}
        if score_notes.get(benchmark):
            return as_percent(scores[benchmark]), score_notes[benchmark]
        return as_percent(scores[benchmark]), "measured"
    others = SCORE_FALLBACK.get(benchmark, [])
    for other in others:
        if scores.get(other) is not None:
            return as_percent(scores[other]), "no " + benchmark + " score, used " + other
    curve = None
    if benchmark in SIZE_CURVES:
        curve = SIZE_CURVES[benchmark]
    else:
        for other in others:
            if curve is None and other in SIZE_CURVES:
                curve = SIZE_CURVES[other]
    if curve is None and not is_reasoning(benchmark):
        curve = SIZE_CURVES["mmlu"]
    if curve is None or not profile.get("params_b"):
        return None, "no score for " + benchmark
    value = curve[0] + curve[1] * math.log(float(profile["params_b"]))
    value = max(25.0, min(90.0, value))
    return value, "no score, rough guess from the round 10 size curve"


def subject_score(profile, benchmark, subjects):
    """Return (score in percent or None, item count) over the chosen subjects.

    The score is the item-weighted mean of the profile's per-subject
    accuracies (profile["subject_scores"][benchmark]). None when the
    subjects cover fewer than MIN_SUBJECT_ITEMS items.
    """
    table = (profile.get("subject_scores") or {}).get(benchmark)
    if not table or not subjects:
        return None, 0
    right = 0.0
    count = 0
    for subject in subjects:
        row = table.get(subject)
        if not isinstance(row, dict) or row.get("accuracy") is None or not row.get("count"):
            continue
        right += as_percent(row["accuracy"]) * int(row["count"])
        count += int(row["count"])
    if count < MIN_SUBJECT_ITEMS:
        return None, count
    return right / count, count


def read_layer(profile):
    """The donor layer a tap reads: about 70% depth (run 7)."""
    layers = int(profile["layers"])
    layer = int(round(READ_DEPTH * layers))
    if layer > layers - 1:
        layer = layers - 1
    if layer < 0:
        layer = 0
    return layer


def write_layer(profile):
    """The host layer a tap writes: about 40% depth (runs 7 and 12)."""
    layers = int(profile["layers"])
    layer = int(round(WRITE_DEPTH * layers))
    if layer > layers - 1:
        layer = layers - 1
    if layer < 0:
        layer = 0
    return layer


def measured_recovery(host, donor):
    for host_id, donor_id, fraction, where in MEASURED_RECOVERY:
        if host.get("hf_id") == host_id and donor.get("hf_id") == donor_id:
            return fraction, where
    return None, None


def recovery_fraction(host, donor):
    """Return (fraction of the donor-host gap a tap closes, note).

    Rule: 0.9 for hosts up to 1.5B, 0.1 for hosts of 7B or more, and a
    straight line in log size between. Donors under 1.5B get half (a
    prior). A measured pair overrides the rule.
    """
    fraction, where = measured_recovery(host, donor)
    if fraction is not None:
        return fraction, "measured recovery " + str(fraction) + " (" + where + ")"
    host_b = decode_params_b(host)
    if host_b <= SMALL_HOST_MAX_B:
        fraction = SMALL_HOST_RECOVERY
    elif host_b >= LARGE_HOST_MIN_B:
        fraction = LARGE_HOST_RECOVERY
    else:
        position = (math.log(host_b) - math.log(SMALL_HOST_MAX_B)) / (math.log(LARGE_HOST_MIN_B) -
                                                                       math.log(SMALL_HOST_MAX_B))
        fraction = SMALL_HOST_RECOVERY + position * (LARGE_HOST_RECOVERY - SMALL_HOST_RECOVERY)
    note = "rule recovery " + str(round(fraction, 2)) + " for a " + str(round(host_b, 2)) + "B host"
    if float(donor["params_b"]) < SMALL_DONOR_MAX_B:
        fraction = fraction * SMALL_DONOR_FACTOR
        note = note + ", halved for a donor under 1.5B (a prior)"
    return fraction, note


def tap_recovery(host, donor, benchmark, kind="mcq"):
    """Predict the accuracy of host + one tap from donor. Return (accuracy or None, note).

    Knowledge MCQ: host + fraction * (donor - host), never above the donor.
    0 gain when the donor is not better than the host by MIN_DONOR_MARGIN
    (run 9). Reasoning benchmarks: no gain (run 5). Free-form short
    answers: TAP_GENERATE_DELTA (run 7 TriviaQA).
    """
    host_score, host_note = score_of(host, benchmark)
    if host_score is None:
        return None, "no host score: " + host_note
    if is_reasoning(benchmark):
        return host_score, "taps carry no multi-step reasoning (run 5: tap equals the mismatched control)"
    if kind == "generate":
        return host_score + TAP_GENERATE_DELTA, "taps hurt free-form answers (run 7 TriviaQA 31.3 vs host 37.6)"
    donor_score, donor_note = score_of(donor, benchmark)
    if donor_score is None:
        return host_score, "no donor score: " + donor_note
    if donor_score - host_score < MIN_DONOR_MARGIN:
        return host_score, "donor not better than host, no gain (run 9)"
    fraction, note = recovery_fraction(host, donor)
    accuracy = host_score + fraction * (donor_score - host_score)
    if accuracy > donor_score:
        accuracy = donor_score
    return accuracy, note


def lora_gain(host, benchmark, kind="mcq"):
    """Return (points a host LoRA adds, note).

    Knowledge MCQ: +3 for hosts up to 1.5B (run 7 Llama-1B), 0 for hosts
    of 7B or more (run 9 Qwen-14B), a line in log size between.
    Reasoning: measured per host (GSM8K raw prompt) or LORA_REASONING_DEFAULT.
    Free-form short answers: +3.5 (run 7 TriviaQA).
    """
    if is_reasoning(benchmark):
        hf_id = host.get("hf_id")
        if hf_id in MEASURED_LORA_REASONING:
            gain, where = MEASURED_LORA_REASONING[hf_id]
            return gain, where
        return LORA_REASONING_DEFAULT, "no LoRA measurement for this host; the smaller measured GSM8K gain (+2.8)"
    if kind == "generate":
        return LORA_GENERATE_GAIN, "run 7 TriviaQA: LoRA r16 41.1 vs host 37.6"
    host_b = decode_params_b(host)
    if host_b <= SMALL_HOST_MAX_B:
        gain = LORA_MCQ_SMALL_GAIN
    elif host_b >= LARGE_HOST_MIN_B:
        gain = LORA_MCQ_LARGE_GAIN
    else:
        position = (math.log(host_b) - math.log(SMALL_HOST_MAX_B)) / (math.log(LARGE_HOST_MIN_B) -
                                                                       math.log(SMALL_HOST_MAX_B))
        gain = LORA_MCQ_SMALL_GAIN + position * (LORA_MCQ_LARGE_GAIN - LORA_MCQ_SMALL_GAIN)
    return gain, "LoRA adds format, not knowledge (run 7: +3.0 on Llama-1B; run 9: -0.4 on Qwen-14B)"


def tap_branch_params(host, donor, attn_dim=256):
    """Trainable parameters in one TapBranch: q and o on the host side, k and v on the donor side, two norms, a gate."""
    d_host = int(host["hidden"])
    d_donor = int(donor["hidden"])
    return 2 * attn_dim * d_host + 2 * attn_dim * d_donor + d_host + d_donor + 1


def lora_params(host, rank):
    """Rough LoRA parameter count: about 21.5 * hidden per rank per layer.

    That factor matches Llama-1B r16 (11.3M in run 4) for q, k, v, o,
    gate, up, down.
    """
    return int(rank * 21.5 * int(host["hidden"]) * int(host["layers"]))


def donor_prefix_params_b(donor, donor_layer):
    """Parameters in donor layers 0..donor_layer, in billions (only those run for a tap)."""
    layers = int(donor["layers"])
    return float(donor["params_b"]) * (donor_layer + 1) / layers


def gpu_speed(gpu):
    return GPU_SPEED.get(gpu, 1.0)


def tap_train_minutes(host, donor_prefix_b, steps=500, gpu="H100"):
    """Minutes to train taps: 100 s for 500 steps at 6.9B run params, scaled by steps and run params."""
    run_params = decode_params_b(host) + donor_prefix_b
    seconds = TAP_SECONDS_REF * (steps / TAP_STEPS_REF) * (run_params / TAP_RUN_PARAMS_REF)
    seconds = max(MIN_TRAIN_SECONDS, seconds) / gpu_speed(gpu)
    return seconds / 60.0


def lora_train_minutes(host, steps=300, gpu="H100"):
    """Minutes to train a host LoRA: 47 s for 300 steps on a 1.24B host, scaled by steps and host size."""
    seconds = LORA_SECONDS_REF * (steps / LORA_STEPS_REF) * (decode_params_b(host) / LORA_HOST_REF)
    seconds = max(MIN_TRAIN_SECONDS, seconds) / gpu_speed(gpu)
    return seconds / 60.0


def heal_train_minutes(host, steps=300, gpu="H100"):
    """Minutes for a graft heal: 24 s for 300 steps on a 1.24B host, scaled by steps and host size."""
    seconds = HEAL_SECONDS_REF * (steps / HEAL_STEPS_REF) * (decode_params_b(host) / LORA_HOST_REF)
    seconds = max(MIN_TRAIN_SECONDS, seconds) / gpu_speed(gpu)
    return seconds / 60.0


def memory_gb(loaded_params_b):
    """GPU memory to hold these bf16 weights, plus a fixed overhead."""
    return 2.0 * loaded_params_b + MEMORY_OVERHEAD_GB


def fits_gpu(loaded_params_b, gpu):
    limit = GPU_MEMORY_GB.get(gpu, GPU_MEMORY_GB["H100"]) * MEMORY_USE_FRACTION
    return memory_gb(loaded_params_b) <= limit


def stacker_gain(benchmark):
    """Points a learned stacker over several models adds to the best one (round 10: +1.2 MMLU, +1.6 MedMCQA)."""
    return STACKER_GAIN.get(benchmark, STACKER_GAIN_DEFAULT)


def ensemble_accuracy(scores, benchmark, method="stacker"):
    """Predict an output-level combination of models from their scores in percent.

    method "stacker": best + stacker_gain. "average": best + the low end
    of the measured probability-average gain (+0.5). A single model gets
    no gain.
    """
    best = None
    for value in scores:
        value = as_percent(value)
        if best is None or value > best:
            best = value
    if best is None:
        return None
    if len(scores) < 2:
        return best
    if method == "average":
        return best + AVERAGE_GAIN[0]
    return best + stacker_gain(benchmark)


def summary_text():
    """The rules in plain sentences, for the LLM planner prompt."""
    lines = [
        "Measured rules (results/README.md, runs 1 to 9 and round 10):",
        "- A tap lets a small host read a frozen donor's states at donor layer about 70% depth "
        "(layer 20 of 28 best; before 57% gives nothing) through a gated cross-attention branch written "
        "at host layer about 40% depth. Only the branch trains, about 100 s for 500 steps.",
        "- Taps transfer knowledge for multiple-choice: Llama-1B 44.7 to 68.3 on held-out MMLU with a "
        "Qwen-7B donor (donor 70.9); Qwen-0.5B + Qwen-14B tap 74.8 (donor 78.5); Llama-1B or Qwen-0.5B + "
        "Gemma-9B tap equal Gemma-9B. Small hosts recover about 0.9 of the donor-host gap.",
        "- Taps do not transfer multi-step reasoning (GSM8K tap equals a mismatched-message control) and "
        "hurt free-form short answers (TriviaQA 31 vs host 38).",
        "- Taps into an already strong host (7B or more) from equal or weaker donors do not help.",
        "- Two taps into a weak host score near the better single tap.",
        "- LoRA r64 lr 5e-5 is the fine-tuning bar on GSM8K (Llama-1B 47.2 from 25.0). LoRA cannot add "
        "missing knowledge (MMLU 47.7 vs tap 68.3). Tap first, freeze, then LoRA adds about 1 point; "
        "training them jointly collapses the tap.",
        "- Output-level combining: probability average +0.5 to +1.3 over the best model, learned stacker "
        "+1.2 (MMLU) and +1.6 (MedMCQA).",
        "- Compile on one H100: LoRA 300 steps 40 to 53 s on a 1B host; tap 500 steps about 100 s.",
        "- Decode cost is the host plus tiny branches; each prompt also runs each donor up to its read layer.",
    ]
    return "\n".join(lines)
