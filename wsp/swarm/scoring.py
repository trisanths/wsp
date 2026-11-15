"""Normalized and combined scores, and the collapse rule (spec 4.2)."""


def normalized(score, base):
    """The fraction of headroom gained: (score - base) / (1 - base).

    Negative when the score is below the host's base.
    """
    if base >= 1.0:
        return 0.0
    return (score - base) / (1.0 - base)


def combined(scores, base_scores, weights, guards=None, guard_drop=0.10, guard_penalty=1.0):
    """The reward.

    The weighted mean of normalized scores over the target tasks, minus a
    penalty for each guard task that falls more than guard_drop (a fraction,
    0.10 is 10 percent) below its base. The penalty is guard_penalty times
    the guard's relative drop.
    """
    total = 0.0
    weight_sum = 0.0
    for task in sorted(weights):
        total += weights[task] * normalized(scores[task], base_scores[task])
        weight_sum += weights[task]
    value = 0.0
    if weight_sum > 0:
        value = total / weight_sum
    if guards is not None:
        for task in guards:
            base = base_scores[task]
            score = scores[task]
            if base > 0 and score < base * (1.0 - guard_drop):
                value -= guard_penalty * (base - score) / base
    return value


def is_collapse(scores, base_scores, fraction=0.5):
    """True when any scored task (target or guard) falls below fraction of its base."""
    for task in sorted(scores):
        if task not in base_scores:
            continue
        if scores[task] < fraction * base_scores[task]:
            return True
    return False


def score_with_config(scores, base_scores, config):
    """Return (combined, collapsed) using the run's task setup."""
    from wsp.swarm.config import guard_tasks, task_weights

    value = combined(scores, base_scores, task_weights(config), guard_tasks(config),
                     config["guard_drop"], config["guard_penalty"])
    collapsed = is_collapse(scores, base_scores, config["collapse_fraction"])
    return value, collapsed
