"""Turn eval results into claims, with the required controls and 3 seeds.

Rules (brief update, 2026-11-14):
  * Main control: the same glue around a random cluster of the same size, from
    the same donor layer, with identical heal training. A gain only counts if
    it beats this control.
  * Second control: a LoRA on the host with the same number of trainable
    parameters (reported next to the main control).
  * Claims use 3 seeds.

`compare` pairs runs by seed (program_seed from program.json). The verdict
`beats_main_control` is True only when there are at least 3 paired seeds, the
mean gain over the control is positive, and the graft beats the control on
every seed. Search-split comparisons are labelled as such: they guide search,
they are not results.
"""
import math


def _scores_by_seed(results, task):
    """results: list of eval() results, or a dict {seed: score}."""
    if isinstance(results, dict):
        scores = {}
        for seed, score in results.items():
            scores[int(seed)] = float(score)
        return scores, None
    scores = {}
    split = None
    for result in results:
        seed = result.get("program_seed")
        if seed is None:
            raise ValueError("result for " + result["model"] + " has no program_seed; cannot pair seeds")
        if seed in scores:
            raise ValueError("two results with seed " + str(seed))
        scores[int(seed)] = result["tasks"][task]["score"]
        split = result["split"]
    return scores, split


def _mean_std(values):
    if len(values) == 0:
        return None, None
    mean = sum(values) / len(values)
    if len(values) < 2:
        return mean, None
    variance = 0.0
    for value in values:
        variance += (value - mean) ** 2
    return mean, math.sqrt(variance / (len(values) - 1))


def summarize(results, task):
    scores, split = _scores_by_seed(results, task)
    values = []
    for seed in sorted(scores):
        values.append(scores[seed])
    mean, std = _mean_std(values)
    return {"mean": mean, "std": std, "n_seeds": len(values), "by_seed": scores, "split": split}


def compare(graft_results, main_control_results, task, lora_results=None, host_score=None, min_seeds=3):
    graft, graft_split = _scores_by_seed(graft_results, task)
    control, control_split = _scores_by_seed(main_control_results, task)

    paired = []
    for seed in sorted(graft):
        if seed in control:
            paired.append(seed)

    gains = []
    for seed in paired:
        gains.append(graft[seed] - control[seed])
    mean_gain, std_gain = _mean_std(gains)

    wins = 0
    for gain in gains:
        if gain > 0:
            wins += 1

    reasons = []
    if len(paired) < min_seeds:
        reasons.append("only " + str(len(paired)) + " paired seeds; claims need " + str(min_seeds))
    if mean_gain is None or mean_gain <= 0:
        reasons.append("mean gain over the random-cluster control is not positive")
    if wins < len(paired):
        reasons.append("graft lost to or tied the control on " + str(len(paired) - wins) + " seed(s)")

    split = graft_split or control_split
    if split == "heldout":
        label = "HELD-OUT"
    elif split == "search":
        label = "SEARCH SPLIT - for search only, not a result"
    else:
        label = "split unknown"

    verdict = {
        "task": task,
        "label": label,
        "graft": summarize(graft_results, task),
        "main_control": summarize(main_control_results, task),
        "paired_seeds": paired,
        "gain_by_seed": dict(zip(paired, gains)),
        "mean_gain": mean_gain,
        "std_gain": std_gain,
        "beats_main_control": len(reasons) == 0,
        "why_not": reasons,
    }
    if lora_results is not None:
        lora = summarize(lora_results, task)
        verdict["lora_control"] = lora
        if lora["mean"] is not None and verdict["graft"]["mean"] is not None:
            verdict["gain_over_lora"] = verdict["graft"]["mean"] - lora["mean"]
    if host_score is not None:
        verdict["host"] = host_score
    return verdict


def format_verdict(verdict):
    lines = []
    lines.append("%s [%s]" % (verdict["task"], verdict["label"]))
    graft = verdict["graft"]
    control = verdict["main_control"]
    lines.append("  graft          mean %.3f over %d seeds" % (graft["mean"], graft["n_seeds"]))
    lines.append("  random control mean %.3f over %d seeds" % (control["mean"], control["n_seeds"]))
    if "lora_control" in verdict:
        lines.append("  LoRA control   mean %.3f over %d seeds" % (verdict["lora_control"]["mean"],
                                                               verdict["lora_control"]["n_seeds"]))
    if verdict["mean_gain"] is not None:
        lines.append("  gain over random control: %+.3f" % verdict["mean_gain"])
    if verdict["beats_main_control"]:
        lines.append("  COUNTS: beats the random-cluster control on every seed")
    else:
        lines.append("  DOES NOT COUNT: " + "; ".join(verdict["why_not"]))
    return "\n".join(lines)
