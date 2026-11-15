"""Normalized and combined scores, the collapse rule (spec 4.2), and the fidelity rule for comparing scores."""


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


def standing(row, config):
    """How a scored row compares with another: (level, combined), where higher is better.

    The fidelity rule (design section 15). The swarm's judge, the cascade
    promotion and every baseline compare scores through this one function.
    Scores compare only at the same fidelity: a low-fidelity score uses a
    third of the questions and a short heal (config low_fidelity_heal_steps,
    50 by default), so it measures a different model from the full score.
    level is the fidelity's FEE cost; the levels are nested, so a higher
    cost is a higher level. Comparing two standings with > therefore puts
    every full-fidelity score above every low-fidelity one, however high the
    low score, and compares combined scores only within one level. A
    low-only program ranks by its low score against other low-only programs
    and never above a program scored at full. With the cascade off every
    score is at full, so standings order exactly as combined scores do.
    """
    from wsp.swarm.config import fidelity_cost

    return (fidelity_cost(config, row["fidelity"]), row["combined"])


def best_standings(rows, config):
    """For each scored program hash in rows (status ok or collapsed), the row with its best standing.

    A program that reached full fidelity ranks on its full row, never on a
    higher low-fidelity score. Returns {program_hash: row}.
    """
    best = {}
    for row in rows:
        if row["status"] not in ("ok", "collapsed"):
            continue
        current = best.get(row["program_hash"])
        if current is None or standing(row, config) > standing(current, config):
            best[row["program_hash"]] = row
    return best
