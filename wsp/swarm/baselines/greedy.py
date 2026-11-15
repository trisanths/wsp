"""Greedy add-if-better over single grafts, then depth shifts (papers.md 8.5, the model-soups recipe).

1. Singles. Score one single-graft program per graftable part, each at its
   default host depth with full-rank glue and heal 300, in a seeded random
   order and in batches of k, until greedy_single_share (half) of the eval
   budget is used. The remaining singles wait for step 4.
2. Add if better. Rank the scored singles best first. Start from the best
   single that reached full fidelity. Try adding each next single's graft to
   the current best program, one at a time; keep the bigger program only if
   its full-fidelity search score is higher. Programs over the graft limit
   are invalid and skipped for free. This step stops once greedy_add_share
   (three quarters) of the eval budget is used; untried adds wait for step 4.
3. Shifts. Move each graft of the current best by -1, +1, -2 and +2 host
   layers, scored in batches of k. When a batch holds a program that beats
   the best, it becomes the best and the shifts start again from it. Stop
   when every shift of the best has been tried without a gain.
4. If budget is left, score the remaining singles, try the remaining adds
   with no cap, and repeat step 3.

The two shares are choices made here. Without them, at 60 evals the
singles and the add tail use the whole budget and the shifts never run,
although placement is what moved scores most in round 2 (results/README.md).

Every program goes through the swarm's screen and fit gate, and the
cascade applies exactly as in the swarm. A comparison uses full-fidelity
scores only. Greedy is deterministic apart from the seeded order of the
singles; when the budget covers every single, every seed ranks the same
singles and runs the same way. When nothing is left to try, it stops
with budget unspent and logs why.
"""

from wsp.swarm import program_schema as ps
from wsp.swarm import space
from wsp.swarm.baselines import SearchArm, queue_source, setting

SINGLE_HEAL_STEPS = 300
SINGLE_SHARE = 0.5          # singles stop once this fraction of the eval budget is used
ADD_SHARE = 0.75            # adds stop once this fraction is used (until step 4)
SHIFTS = [-1, 1, -2, 2]


class GreedyArm(SearchArm):
    def __init__(self, config, env):
        SearchArm.__init__(self, config, env)
        self.single_share = setting(config, "greedy_single_share", SINGLE_SHARE)
        self.add_share = setting(config, "greedy_add_share", ADD_SHARE)
        self.singles = []           # scored singles: part, graft, program, hash, combined, fidelity, status
        self.added = set()          # part ids already tried as an add (or used as the start)
        self.best_program = None
        self.best_hash = None
        self.best_score = None

    def settings(self):
        out = SearchArm.settings(self)
        out["single_heal_steps"] = SINGLE_HEAL_STEPS
        out["greedy_single_share"] = self.single_share
        out["greedy_add_share"] = self.add_share
        out["shifts"] = list(SHIFTS)
        return out

    def search(self):
        eval_budget = self.env.budget.eval_budget
        queue = self.single_proposals()
        self.score_singles(queue, self.single_share * eval_budget)
        self.climb(self.add_share * eval_budget)
        if self.keep_going():
            self.score_singles(queue, None)
            self.climb(None)
        if not self.stopped and not self.search_over():
            self.stop("greedy has tried every single, add and shift; budget left unspent")

    def eval_used_reaches(self, eval_cap):
        return eval_cap is not None and self.env.budget.eval_used() >= eval_cap

    # Step 1.

    def single_proposals(self):
        parts = list(space.graftable_parts(self.catalog))
        self.rng.shuffle(parts)
        n_host = space.host_layers(self.config, self.catalog)
        heal = space.heal_with_steps(SINGLE_HEAL_STEPS, self.config)
        queue = []
        for record in parts:
            at = space.default_at(record, self.config, n_host)
            graft = ps.make_graft(record["id"], at)
            program = ps.make_program(self.config["host"], [graft], heal)
            hypothesis = "greedy: single " + space.short_name(record["id"]) + "@" + str(at) + " at its default depth"
            queue.append({"program": program, "hypothesis": hypothesis, "parent": None,
                          "extra": {"phase": "single", "part": record["id"]}})
        return queue

    def score_singles(self, queue, eval_cap):
        """Score singles from the queue in batches until it is empty, the budget ends, or eval_cap is reached."""
        while len(queue) > 0 and self.keep_going():
            if self.eval_used_reaches(eval_cap):
                return
            candidates = self.collect(queue_source(queue), self.batch_size())
            if len(candidates) == 0:
                return
            rows = self.evaluate(candidates)
            results = self.best_scores(rows)
            for candidate in candidates:
                result = results.get(candidate["program_hash"])
                if result is None:
                    continue
                self.singles.append({"part": candidate["extra"]["part"], "graft": ps.grafts(candidate["program"])[0],
                                     "program": candidate["program"], "hash": candidate["program_hash"],
                                     "combined": result["combined"], "fidelity": result["fidelity"],
                                     "status": result["status"]})

    def ranked_singles(self):
        """Scored singles, best first (score at the highest fidelity reached; ties by part id)."""
        pairs = []
        for index in range(len(self.singles)):
            entry = self.singles[index]
            pairs.append((-entry["combined"], entry["part"], index))
        pairs.sort()
        ranked = []
        for pair in pairs:
            ranked.append(self.singles[pair[2]])
        return ranked

    # Steps 2 and 3.

    def climb(self, add_cap):
        """Steps 2 and 3: start from the best single, add if better (until add_cap evals), then shift."""
        ranked = self.ranked_singles()
        if self.best_program is None:
            for entry in ranked:
                if entry["status"] == "ok" and entry["fidelity"] == "full":
                    self.added.add(entry["part"])
                    self.accept(entry["program"], entry["hash"], entry["combined"], "best single")
                    break
        if self.best_program is None:
            return
        self.add_phase(ranked, add_cap)
        self.shift_phase()

    def accept(self, program, program_hash, score, why):
        self.best_program = program
        self.best_hash = program_hash
        self.best_score = score
        self.env.log_event("greedy_best", "new best (" + why + "): " + space.describe(program) + " " +
                           "%.3f" % score, self.agent, self.round_index,
                           extra={"program_hash": program_hash, "combined": score})

    def add_phase(self, ranked, add_cap):
        for entry in ranked:
            if not self.keep_going() or self.eval_used_reaches(add_cap):
                return
            if entry["part"] in self.added or entry["status"] != "ok":
                continue
            self.added.add(entry["part"])
            grafts = ps.grafts(self.best_program)
            grafts.append(entry["graft"])
            program = ps.with_grafts(self.best_program, grafts)
            name = space.short_name(entry["part"]) + "@" + str(ps.graft_at(entry["graft"]))
            proposal = {"program": program, "hypothesis": "greedy: add " + name + " to the best " +
                        space.describe(self.best_program), "parent": self.best_hash,
                        "extra": {"phase": "add", "part": entry["part"]}}
            candidates = self.collect(queue_source([proposal]), 1)
            if len(candidates) == 0:
                continue
            rows = self.evaluate(candidates)
            score = self.full_score(candidates[0]["program_hash"], rows)
            if score is not None and score > self.best_score:
                self.accept(candidates[0]["program"], candidates[0]["program_hash"], score, "add " + name)

    def shift_proposals(self):
        n_host = space.host_layers(self.config, self.catalog)
        grafts = ps.grafts(self.best_program)
        proposals = []
        for index in range(len(grafts)):
            graft = grafts[index]
            at = space.resolved_at(graft, self.catalog, self.config)
            if at is None:
                continue
            for shift in SHIFTS:
                new_at = at + shift
                if new_at < 0 or new_at > n_host - 1:
                    continue
                moved = list(grafts)
                moved[index] = ps.graft_with_at(graft, new_at)
                program = ps.with_grafts(self.best_program, moved)
                note = "move " + space.short_name(ps.graft_id(graft)) + " from " + str(at) + " to " + str(new_at)
                proposals.append({"program": program, "hypothesis": "greedy: " + note + " on the best",
                                  "parent": self.best_hash, "extra": {"phase": "shift", "note": note}})
        return proposals

    def shift_phase(self):
        improved = True
        while improved and self.keep_going():
            improved = False
            queue = self.shift_proposals()
            while len(queue) > 0 and self.keep_going():
                candidates = self.collect(queue_source(queue), self.batch_size())
                if len(candidates) == 0:
                    break
                rows = self.evaluate(candidates)
                if self.accept_best(candidates, rows):
                    improved = True
                    break

    def accept_best(self, candidates, rows):
        """Make the batch's best full-fidelity program the best if it beats the best. Return True if it did."""
        winner = None
        winner_score = None
        for candidate in candidates:
            score = self.full_score(candidate["program_hash"], rows)
            if score is None:
                continue
            if winner_score is None or score > winner_score:
                winner = candidate
                winner_score = score
        if winner is None or winner_score <= self.best_score:
            return False
        self.accept(winner["program"], winner["program_hash"], winner_score, winner["extra"]["note"])
        return True
