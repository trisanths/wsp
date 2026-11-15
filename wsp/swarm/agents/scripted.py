"""Scripted agents: no LLM, fixed seeds, plain rules.

- ScriptedAnalyst ranks parts by eval evidence, starting from the signature
  hint, and lists good placements from known fit. Fit picks placements only.
- ScriptedPlanner mixes exploitation (mutate a frontier program) with
  exploration (graft a top part at its best known placement). It asks for a
  probe when a graft sits at a depth whose fit is unknown.
- RuleJudge ranks on search scores only, keeps a top-k frontier (ties within
  tie_margin go to fewer parameters), kills patterns after kill_after
  collapses or fit rejections, and retires parents after retire_after failed
  children.
"""

from wsp.swarm import program_schema as ps
from wsp.swarm import space
from wsp.swarm.agents.base import Analyst, Judge, Planner
from wsp.swarm.env import rank_key
from wsp.swarm.scoring import normalized

EXPLOIT_RATES = [0.7, 0.5, 0.3]  # planner 1 exploits most, planner 3 explores most
HEAL_CHOICES = [300, 300, 300, 100, 600]


def part_credits(history):
    """Per-part credit from scored full-fidelity search rows: combined / number of grafts."""
    credits = {}
    for row in history:
        if row["status"] not in ("ok", "collapsed") or row.get("fidelity") != "full":
            continue
        grafts = ps.grafts(row["program"])
        if len(grafts) == 0:
            continue
        share = row["combined"] / float(len(grafts))
        for graft in grafts:
            if ps.graft_control(graft) is not None:
                continue
            part_id = ps.graft_id(graft)
            if part_id not in credits:
                credits[part_id] = []
            credits[part_id].append(share)
    return credits


def main_task(weights):
    """The target task with the largest weight (ties go to the first name)."""
    best = None
    for task in sorted(weights):
        if best is None or weights[task] > weights[best]:
            best = task
    return best


def signature_prior(record, task, base_scores):
    """The signature's effect hint for a task, as normalized headroom gained."""
    effect = (record.get("signature") or {}).get("effect") or {}
    hint = effect.get(task)
    if hint is None:
        return 0.0
    base = base_scores.get(task, 0.0)
    return normalized(base + hint, base)


class ScriptedAnalyst(Analyst):
    def __init__(self, config=None):
        self.name = "analyst"

    def build_map(self, observation, ctx):
        config = ctx["config"]
        catalog = ctx["env"].catalog
        credits = part_credits(observation["history"])
        by_task = {}
        estimates = {}
        for task in sorted(observation["tasks"]):
            scored = []
            for record in observation["parts"]:
                prior = signature_prior(record, task, observation["base_scores"])
                found = credits.get(record["id"], [])
                total = prior
                for value in found:
                    total += value
                estimate = total / (1.0 + len(found))
                scored.append((-estimate, record["id"]))
                if task not in estimates:
                    estimates[task] = {}
                estimates[task][record["id"]] = estimate
            scored.sort()
            ranking = []
            for pair in scored:
                ranking.append(pair[1])
            by_task[task] = ranking

        placements = {}
        for record in observation["parts"]:
            placements[record["id"]] = self.placements_for(record, observation, config, catalog)

        notes = []
        collapses = 0
        rejections = 0
        for row in observation["history"]:
            if row["status"] == "collapsed":
                collapses += 1
            if row["status"] == "rejected_fit":
                rejections += 1
        notes.append(str(collapses) + " collapses and " + str(rejections) + " fit rejections so far")

        lines = []
        for task in sorted(by_task):
            top = []
            for part_id in by_task[task][:5]:
                count = len(credits.get(part_id, []))
                top.append(space.short_name(part_id) + " (est " + "%.3f" % estimates[task][part_id] +
                           ", " + str(count) + " evals, at " + str(placements[part_id][:2]) + ")")
            lines.append("Top parts for " + task + ": " + "; ".join(top))
        text = " ".join(lines) + ". " + notes[0] + "."
        return {"text": text, "by_task": by_task, "placements": placements, "notes": notes,
                "estimates": estimates}

    def placements_for(self, record, observation, config, catalog):
        """Good `at` values for a part: known fit above the threshold, best first, else the default depth."""
        n_host = observation["host"]["layers"]
        default = space.default_at(record, config, n_host)
        if default is None:
            default = n_host // 2
        layer = space.part_layer(record)
        good = []
        for at in range(n_host):
            slot = space.slot_key(record.get("model"), layer, at)
            fit = observation["known_fits"].get(slot)
            if fit is None or fit < observation["fit_threshold"]:
                continue
            good.append((-fit, abs(at - default), at))
        good.sort()
        out = []
        for item in good:
            out.append(item[2])
        if len(out) == 0:
            out.append(default)
        return out


class ScriptedPlanner(Planner):
    def __init__(self, index, config=None):
        self.index = index
        self.name = "planner-" + str(index + 1)
        self.exploit_rate = EXPLOIT_RATES[index % len(EXPLOIT_RATES)]

    def propose(self, observation, cap_map, lessons, priors, n, ctx):
        rng = ctx["rng"]
        config = ctx["config"]
        catalog = ctx["env"].catalog
        parents = []
        for row in observation["frontier"]:
            if row["program_hash"] not in ctx.get("retired", []):
                parents.append(row)
        proposals = []
        attempts = 0
        while len(proposals) < n and attempts < n * 6:
            attempts += 1
            if len(parents) > 0 and rng.random() < self.exploit_rate:
                proposal = self.exploit(parents, rng, catalog, config)
            else:
                proposal = self.explore(observation, cap_map, parents, rng, catalog, config)
            if proposal is None:
                continue
            if len(space.validate(proposal["program"], catalog, config)) > 0:
                continue
            proposal["probe_first"] = self.needs_probe(proposal["program"], observation, catalog, config)
            proposals.append(proposal)
        return proposals

    def pick_parent(self, parents, rng):
        """Rank-weighted choice: weight 1 / (rank + 1)."""
        weights = []
        for rank in range(len(parents)):
            weights.append(1.0 / (rank + 1))
        return parents[rng.choices(range(len(parents)), weights=weights)[0]], weights

    def exploit(self, parents, rng, catalog, config):
        parent, weights = self.pick_parent(parents, rng)
        child, note = space.mutate_with_note(parent["program"], rng, catalog, config)
        hypothesis = (note + " on " + parent["describe"] + " (search combined " + "%.3f" % parent["combined"] +
                      ") should beat its parent")
        return {"program": child, "hypothesis": hypothesis, "citations": [], "probe_first": False,
                "parent": parent["program_hash"]}

    def explore(self, observation, cap_map, parents, rng, catalog, config):
        task = main_task(observation["tasks"])
        ranking = cap_map["by_task"][task][:6]
        if len(ranking) == 0:
            return None
        rank_weights = []
        for position in range(len(ranking)):
            rank_weights.append(float(len(ranking) - position))
        part_id = ranking[rng.choices(range(len(ranking)), weights=rank_weights)[0]]
        options = cap_map["placements"][part_id]
        at = options[0]
        if len(options) > 1 and rng.random() < 0.3:
            at = rng.choice(options[:3])
        n_host = observation["host"]["layers"]
        if rng.random() < 0.2:
            at = at + rng.choice([-1, 1])
            if at < 0:
                at = 0
            if at > n_host - 1:
                at = n_host - 1
        graft = ps.make_graft(part_id, at)
        name = space.short_name(part_id) + "@" + str(at)
        estimate = cap_map.get("estimates", {}).get(task, {}).get(part_id, 0.0)

        if len(parents) > 0 and rng.random() < 0.5:
            parent = parents[0]
            grafts = ps.grafts(parent["program"])
            if len(grafts) < config["max_grafts"] and (part_id, at) not in space.taken_pairs(grafts):
                grafts.append(graft)
                program = ps.with_grafts(parent["program"], grafts)
                hypothesis = ("adding " + name + " (estimated gain " + "%.3f" % estimate + ") to the best program " +
                              parent["describe"] + " adds " + task + " skill")
                return {"program": program, "hypothesis": hypothesis, "citations": [], "probe_first": False,
                        "parent": parent["program_hash"]}

        steps = rng.choice(HEAL_CHOICES)
        program = ps.make_program(config["host"], [graft], space.heal_with_steps(steps, config))
        hypothesis = ("grafting " + name + " (estimated gain " + "%.3f" % estimate + ") with heal " + str(steps) +
                      " adds " + task + " skill on its own")
        return {"program": program, "hypothesis": hypothesis, "citations": [], "probe_first": False, "parent": None}

    def needs_probe(self, program, observation, catalog, config):
        """True when any graft sits in a slot whose fit has not been measured."""
        for graft in ps.grafts(program):
            slot = space.graft_slot(graft, catalog, config)
            if slot is None or slot not in observation["known_fits"]:
                return True
        return False


class RuleJudge(Judge):
    def __init__(self, config=None):
        self.name = "judge"
        self.collapse_counts = {}
        self.reject_counts = {}
        self.best = {}          # program hash -> best full-fidelity combined on search
        self.children = {}      # parent hash -> {"failed": n, "better": n}

    def update(self, frontier, rows, ctx):
        config = ctx["config"]
        catalog = ctx["env"].catalog
        candidates = list(frontier)
        for row in rows:
            if row.get("type") != "eval" or row.get("split") != "search":
                continue
            status = row["status"]
            if status == "collapsed":
                for pattern in row["extra"].get("patterns", []):
                    self.collapse_counts[pattern] = self.collapse_counts.get(pattern, 0) + 1
            if status == "rejected_fit":
                for pattern in self.rejected_patterns(row, catalog, config):
                    self.reject_counts[pattern] = self.reject_counts.get(pattern, 0) + 1
            if status == "ok" and row["fidelity"] == "full":
                candidates.append(row)
                if row["program_hash"] not in self.best or row["combined"] > self.best[row["program_hash"]]:
                    self.best[row["program_hash"]] = row["combined"]
            self.track_child(row)

        killed = []
        for pattern in sorted(set(list(self.collapse_counts) + list(self.reject_counts))):
            if self.collapse_counts.get(pattern, 0) >= config["kill_after"]:
                killed.append(pattern)
            elif self.reject_counts.get(pattern, 0) >= config["kill_after"]:
                killed.append(pattern)

        retired = []
        for parent in sorted(self.children):
            counts = self.children[parent]
            if counts["failed"] >= config["retire_after"] and counts["better"] == 0:
                retired.append(parent)

        new_frontier = rank_rows(candidates, config["frontier_size"], config["tie_margin"])
        text = verdict_text(new_frontier, killed, retired)
        return {"frontier": new_frontier, "killed": killed, "retired": retired, "text": text}

    def rejected_patterns(self, row, catalog, config):
        rejected = row["extra"].get("rejected", [])
        patterns = []
        for graft in ps.grafts(row["program"]):
            key = space.graft_key(ps.graft_id(graft), space.resolved_at(graft, catalog, config),
                                  ps.graft_control(graft))
            if key in rejected:
                patterns.append(space.graft_pattern(graft, row["program"], catalog, config))
        return patterns

    def track_child(self, row):
        parent = row["extra"].get("parent")
        if parent is None or row["status"] in ("invalid", "budget_exhausted", "cache_hit", "infra_error"):
            return
        if row["status"] == "ok" and row["fidelity"] != "full":
            return
        if parent not in self.children:
            self.children[parent] = {"failed": 0, "better": 0}
        parent_score = self.best.get(parent)
        if row["status"] == "ok" and parent_score is not None and row["combined"] > parent_score:
            self.children[parent]["better"] += 1
        else:
            self.children[parent]["failed"] += 1


def rank_rows(rows, size, tie_margin):
    """Best first, one row per program. Within tie_margin, fewer parameters go first."""
    best = {}
    for row in rows:
        current = best.get(row["program_hash"])
        if current is None or row["combined"] > current["combined"]:
            best[row["program_hash"]] = row
    ranked = list(best.values())
    ranked.sort(key=rank_key)
    changed = True
    while changed:
        changed = False
        for i in range(1, len(ranked)):
            above = ranked[i - 1]
            below = ranked[i]
            close = above["combined"] - below["combined"] <= tie_margin
            if close and params_of(below) < params_of(above):
                ranked[i - 1] = below
                ranked[i] = above
                changed = True
    return ranked[:size]


def params_of(row):
    value = row["extra"].get("params")
    if value is None:
        return 0
    return value


def verdict_text(frontier, killed, retired):
    tops = []
    for row in frontier[:3]:
        tops.append(row["describe"] + " " + "%.3f" % row["combined"])
    text = "Frontier top: " + ("; ".join(tops) if len(tops) > 0 else "empty") + "."
    if len(killed) > 0:
        text += " Killed patterns: " + ", ".join(killed) + "."
    if len(retired) > 0:
        text += " Retired parents: " + ", ".join(retired) + "."
    return text
