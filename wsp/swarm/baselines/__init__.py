"""Baselines at equal budget (design sections 5, 10 and 14; papers.md section 8.5).

run_baseline(config) runs one baseline arm and returns the same summary dict
as run_swarm. config["method"] picks the arm:

    evolve       regularized evolution over grafts, the primary baseline     evolve.py
    random       uniform random programs, deduped by hash, the sanity floor   random_search.py
    random_part  the same evolution with every graft's units made random      random_part.py
    greedy       greedy add-if-better over single grafts, then depth shifts   greedy.py
    lora         LoRA on the host: STUB runner, the real run is a plan         training.py
    lora_grid    host LoRA programs over rank x lr x steps, any backend         lora_grid.py
    full_ft      full fine-tuning of the host: STUB runner, plan               training.py

Fairness. run_baseline builds the env, search budget, held-out budget, log
and backend with factory.build_run, exactly as run_swarm does. Every search
arm hands its programs to env.step_many, so validation, the cache, the fit
gate, token spends, compute charges, the ledger and the log rows are the
same code for every arm, and config["method"] is stamped on every log row
and ledger row. Every search arm uses SearchArm.screen, the swarm's screen:
drop invalid programs and repeats, veto known low-fit slots, and probe a
program with an untested slot (the scripted planner asks for a probe in
exactly that case). The swarm's screen also drops the judge's killed
patterns; baselines have no judge, so that step is the swarm's own.
SearchArm.evaluate is Swarm.evaluate: with config "cascade", a batch is
scored at "low" and the top third (swarm.promotion_picks) goes on to
"full". The final pick uses the swarm judge's ranking
(agents.scripted.rank_rows: best full-fidelity search score, ties within
tie_margin go to fewer parameters); the top heldout_top programs go to
held-out once, each with its two controls.

Every comparison an arm makes follows the fidelity rule the judge follows
(scoring.standing): scores compare only at the same fidelity. Fitness
(evolve) and the order of the singles (greedy) rank a program on its
full-fidelity score when it reached full; a low-only program ranks by its
low score against other low-only programs and never above a full-scored
one. best_scores gives each program's standing for that comparison.

Baselines use no memory, literature or LLM, so run_baseline refuses a
config that switches one on (anti-goal: no arm may get memory the others
lack).

Method settings that are not in config DEFAULTS are read with setting() and
fall back to the defaults in each module: population_size, tournament_size,
crossover_rate (evolve and random_part), greedy_single_share and
greedy_add_share (greedy), train_checkpoints, lora_rank and
lora_match_params (training).
"""

import math
import random

from wsp.swarm import factory
from wsp.swarm import program_schema as ps
from wsp.swarm import space
from wsp.swarm.agents.scripted import rank_rows
from wsp.swarm.config import fidelity_cost, public_config
from wsp.swarm.scoring import best_standings, standing
from wsp.swarm.swarm import promotion_picks, short_row

METHODS = ["evolve", "random", "random_part", "greedy", "lora", "full_ft", "lora_grid"]
TRAINING_METHODS = ["lora", "full_ft"]


def setting(config, name, default):
    """A baseline setting: config[name] when the config sets it, else the module default."""
    if name in config and config[name] is not None:
        return config[name]
    return default


def check_config(config):
    """Raise before anything is built when a config cannot run as a baseline."""
    method = config.get("method")
    if method not in METHODS:
        raise ValueError("unknown baseline method '" + str(method) + "'; choose one of " + ", ".join(METHODS))
    if config["memory"] != "off" or config["literature"] != "off" or config["llm"] != "none":
        raise ValueError("baselines run with memory off, literature off and llm none, so no arm gets "
                         "memory, priors or an LLM the others lack")
    if method in TRAINING_METHODS and config["backend"] != "stub":
        raise NotImplementedError("the " + method + " arm has only a STUB runner; the real run is a plan in "
                                  "wsp/swarm/baselines/training.py")


def make_arm(config, env):
    """The arm object for config["method"]. Modules load only when chosen."""
    method = config["method"]
    if method == "evolve":
        from wsp.swarm.baselines.evolve import EvolveArm
        return EvolveArm(config, env)
    if method == "random":
        from wsp.swarm.baselines.random_search import RandomSearchArm
        return RandomSearchArm(config, env)
    if method == "random_part":
        from wsp.swarm.baselines.random_part import RandomPartArm
        return RandomPartArm(config, env)
    if method == "greedy":
        from wsp.swarm.baselines.greedy import GreedyArm
        return GreedyArm(config, env)
    if method == "lora_grid":
        from wsp.swarm.baselines.lora_grid import LoRAGridArm
        return LoRAGridArm(config, env)
    from wsp.swarm.baselines.training import TrainingArm, make_trainer
    return TrainingArm(config, env, make_trainer(config, method))


def run_baseline(config):
    """Build everything from config, run one baseline arm, and return its summary."""
    check_config(config)
    run = factory.build_run(config)
    arm = make_arm(config, run["env"])
    return arm.run()


def queue_source(queue):
    """A proposal source that hands out the items of a list in order, then None."""

    def next_proposal():
        if len(queue) == 0:
            return None
        return queue.pop(0)

    return next_proposal


def build_summary(config, env, rounds, frontier, heldout, controls):
    """The summary dict, in the same shape as Swarm.summary."""
    budget = env.budget
    best = None
    if len(frontier) > 0:
        top = frontier[0]
        best = {"combined": top["combined"], "describe": top["describe"], "program_hash": top["program_hash"],
                "scores": top["scores"], "label": "SEARCH"}
    heldout_out = []
    for row in heldout:
        heldout_out.append(short_row(row))
    controls_out = []
    for row in controls:
        item = short_row(row)
        item["control"] = row["extra"].get("control")
        item["control_of"] = row["extra"].get("control_of")
        controls_out.append(item)
    return {
        "run_id": config["run_id"],
        "method": config["method"],
        "backend": env.backend.name,
        "stub": env.is_stub(),
        "seed": config["seed"],
        "rounds": rounds,
        "eval_used": budget.eval_used(),
        "eval_budget": budget.eval_budget,
        "compute_used_s": budget.compute_used(),
        "compute_budget_s": budget.compute_budget_s,
        "compute_overshoot_s": budget.overshoot_s(),
        "best_search": best,
        "heldout": heldout_out,
        "controls": controls_out,
        "log_path": env.log.path,
        "config": public_config(config),
    }


class SearchArm:
    """What every search baseline shares: screen, evaluate (with the cascade), stop rules, held-out, summary.

    A subclass implements search(), which loops while keep_going() is True.
    Each round it builds proposals {"program", "hypothesis", "parent", "extra"},
    turns them into candidates with collect() (which calls screen()), and
    scores them with evaluate(). A round that finds no candidate calls
    note_stall(); after config max_stall_rounds stalls in a row, or
    config max_rounds rounds, the arm stops.
    """

    def __init__(self, config, env):
        self.config = config
        self.env = env
        self.catalog = env.catalog
        self.agent = config["method"]
        self.rng = random.Random(config["seed"])
        self.tried = set()       # program hashes scored, vetoed or dropped
        self.rows = []           # every eval row this arm got back, in order
        self.round_index = 0
        self.stalls = 0
        self.stopped = False

    def settings(self):
        """Method settings for the run manifest. Subclasses add their own."""
        return {"method": self.agent, "screen": "swarm screen: fit gate, probe of untested slots"}

    def prepare(self, program):
        """Change a proposed program before it is screened. random_part overrides this."""
        return program

    def search(self):
        raise NotImplementedError

    # Stop rules (the same as the swarm's).

    def first_fidelity(self):
        if self.config["cascade"]:
            return "low"
        return "full"

    def search_over(self):
        if self.env.search_done():
            return True
        cost = fidelity_cost(self.config, self.first_fidelity())
        return not self.env.budget.can_spend(cost)

    def keep_going(self):
        if self.stopped or self.search_over():
            return False
        if self.round_index >= self.config["max_rounds"]:
            self.stop("stopped at max_rounds")
            return False
        return True

    def stop(self, text):
        self.stopped = True
        self.env.log_event("stop", text, self.agent, self.round_index)

    def note_stall(self):
        self.stalls += 1
        self.env.log_event("stall", "no new candidates this round", self.agent, self.round_index)
        self.round_index += 1
        if self.stalls >= self.config["max_stall_rounds"]:
            self.stop("stopped after " + str(self.stalls) + " rounds without candidates")

    def batch_size(self):
        """Candidates per round: config k, or fewer when the eval axis cannot pay for k."""
        cost = fidelity_cost(self.config, self.first_fidelity())
        affordable = int(math.floor(self.env.budget.eval_remaining() / cost + 1e-9))
        return max(0, min(self.config["k"], affordable))

    # Proposals to candidates.

    def collect(self, next_proposal, wanted, max_attempts=None):
        """Call next_proposal() until `wanted` proposals pass screen(), it returns None, or attempts run out."""
        candidates = []
        seen = set()
        attempts = 0
        while len(candidates) < wanted:
            if max_attempts is not None and attempts >= max_attempts:
                break
            attempts += 1
            proposal = next_proposal()
            if proposal is None:
                break
            candidate = self.screen(proposal, seen)
            if candidate is not None:
                candidates.append(candidate)
        return candidates

    def screen(self, proposal, seen):
        """Return a step candidate, or None when the proposal is dropped (free, like the swarm's screen)."""
        program = self.prepare(proposal["program"])
        if len(space.validate(program, self.catalog, self.config)) > 0:
            return None
        program = ps.normalize(program)
        program_hash = space.program_hash(program)
        if program_hash in self.tried or program_hash in seen:
            return None
        if not self.passes_fit_gate(program, program_hash, proposal):
            return None
        seen.add(program_hash)
        extra = {"parent": proposal.get("parent")}
        more = proposal.get("extra") or {}
        for key in sorted(more):
            extra[key] = more[key]
        return {"program": program, "program_hash": program_hash, "agent": self.agent,
                "hypothesis": proposal.get("hypothesis", ""), "citations": [], "round": self.round_index,
                "fidelity": self.first_fidelity(), "extra": extra}

    def passes_fit_gate(self, program, program_hash, proposal):
        """The swarm's fit gate: veto a known low-fit slot; probe an untested slot, then veto on its fit."""
        threshold = self.config["fit_threshold"]
        unknown = False
        for graft in ps.grafts(program):
            slot = space.graft_slot(graft, self.catalog, self.config)
            fit = None
            if slot is not None:
                fit = self.env.fit_of_slot(slot)
            if fit is None:
                unknown = True
            elif fit < threshold:
                self.tried.add(program_hash)
                return False
        if not unknown:
            return True
        result = self.env.probe(program, self.agent, self.round_index, proposal.get("hypothesis", ""))
        if result["status"] not in ("ok", "cache_hit"):
            self.tried.add(program_hash)
            return False
        for key in sorted(result["fit_r2"]):
            value = result["fit_r2"][key]
            if value is not None and value < threshold:
                self.tried.add(program_hash)
                return False
        return True

    # Scoring.

    def evaluate(self, candidates):
        """Step the candidates (one round). With the cascade, promote the top third to full (swarm.promotion_picks)."""
        rows = self.env.step_many(candidates)
        self.remember(rows)
        if self.config["cascade"]:
            promoted = promotion_picks(candidates, rows, self.config)
            if len(promoted) > 0:
                more = self.env.step_many(promoted)
                self.remember(more)
                rows = rows + more
        self.stalls = 0
        self.round_index += 1
        return rows

    def remember(self, rows):
        for row in rows:
            self.rows.append(row)
            if row["program_hash"] is not None:
                self.tried.add(row["program_hash"])

    def best_scores(self, rows):
        """For each scored program hash in rows: {"combined", "fidelity", "status", "standing"}.

        The score is the one the program ranks on under the fidelity rule
        (scoring.standing): its full-fidelity score when it reached full,
        else its low score. Compare two results by "standing", never by
        "combined" alone, so a low-only program never outranks a full one.
        """
        out = {}
        best = best_standings(rows, self.config)
        for program_hash in best:
            row = best[program_hash]
            out[program_hash] = {"combined": row["combined"], "fidelity": row["fidelity"], "status": row["status"],
                                 "standing": standing(row, self.config)}
        return out

    def full_score(self, program_hash, rows):
        """The full-fidelity combined score of an ok row for this hash in rows, or None."""
        for row in rows:
            if row["program_hash"] == program_hash and row["status"] == "ok" and row["fidelity"] == "full":
                return row["combined"]
        return None

    def frontier(self):
        """The swarm judge's ranking over this arm's full-fidelity ok search rows."""
        candidates = []
        for row in self.rows:
            if row["status"] == "ok" and row["fidelity"] == "full" and row["split"] == "search":
                candidates.append(row)
        return rank_rows(candidates, self.config["frontier_size"], self.config["tie_margin"], self.config)

    # The whole run.

    def run(self):
        """Run the search and the held-out pass. Returns the summary dict (the swarm's shape)."""
        env = self.env
        env.write_manifest({"baseline": self.settings(), "k": self.config["k"], "cascade": self.config["cascade"]})
        env.reset(self.config["seed"])
        self.search()
        env.end_search(self.agent + " finished after " + str(self.round_index) + " rounds")
        heldout, controls = self.final_heldout()
        summary = build_summary(self.config, env, self.round_index, self.frontier(), heldout, controls)
        env.log.write_json("summary.json", summary)
        return summary

    def final_heldout(self):
        """Held-out once on the top heldout_top programs, each with its two controls (rule 7)."""
        heldout = []
        controls = []
        if self.env.heldout_budget is None:
            return heldout, controls
        for row in self.frontier()[:self.config["heldout_top"]]:
            heldout.append(self.env.heldout_eval(row["program"], self.agent, "final pick from the search ranking"))
            for control_row in self.env.run_controls(row["program"], self.agent):
                controls.append(control_row)
        return heldout, controls
