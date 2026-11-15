"""Regularized evolution over grafts: the primary baseline (papers.md 8.5, recommendation 5).

Regularized evolution (aging evolution, as in NAS-Bench-101), scaled down
for a budget of about 60 to 150 evals:

1. Initial population: 12 single-graft programs, each part at its default
   host depth round(l / L_donor * L_host) with the schema's default
   full-rank glue and heal 300. The parts are the graftable catalog parts in
   a seeded random order. A part whose program is dropped by the screen, or
   does not score, is replaced by the next part.
2. Each round makes up to k children (config "k", the swarm's batch size),
   all from the population as it stands at the start of the round. For each
   child:
   - with probability crossover_rate (0.2) and at least 2 members: two
     tournaments pick two parents, space.crossover mixes their graft lists,
     and the child keeps at most its first 3 grafts;
   - otherwise one tournament picks a parent and space.mutate_with_note
     makes one edit. space.py owns the moves and their odds.
   A tournament samples 3 members at random and returns the fittest.
3. Every child goes through the swarm's screen (SearchArm.screen): invalid
   programs and repeats of a program hash are dropped for free, a child with
   a graft in a slot whose known fit is below the gate is dropped, and a
   child with an untested slot is probed first (compute only, no token) and
   dropped when the probe shows a fit below the gate.
4. The children are scored through env.step_many. With config cascade they
   are scored at "low" and the top third goes on to "full", the swarm's
   rule. Each scored child (ok or collapsed) joins the population once, in
   batch order. Its fitness is its full-fidelity combined score when it
   reached full, else its low-fidelity score. When the population is over
   12, the oldest member dies, whatever its fitness.
5. Repeat until either budget axis is spent.

Fit is used only by the gate, never as fitness (fit ranks placements, not
parts). The rng is random.Random(seed), so a run is fixed by its seed.
"""

from wsp.swarm import program_schema as ps
from wsp.swarm import space
from wsp.swarm.baselines import SearchArm, queue_source, setting

POPULATION_SIZE = 12
TOURNAMENT_SIZE = 3
CROSSOVER_RATE = 0.2
CROSSOVER_MAX_GRAFTS = 3
INITIAL_HEAL_STEPS = 300
CHILD_ATTEMPTS = 60         # proposals tried per round before the round counts as a stall


class Population:
    """The population of regularized evolution: at most `size` members, oldest first.

    A member is a dict with program, program_hash, describe, fitness,
    fidelity and round.
    """

    def __init__(self, size):
        self.size = size
        self.members = []

    def add(self, member):
        """Add a member at the young end. Return the oldest member if it died, else None."""
        self.members.append(member)
        if len(self.members) > self.size:
            return self.members.pop(0)
        return None

    def tournament(self, rng, sample_size):
        """Sample sample_size members without replacement; return the fittest (the first sampled on a tie)."""
        if len(self.members) == 0:
            return None
        count = min(sample_size, len(self.members))
        best = None
        for index in rng.sample(range(len(self.members)), count):
            member = self.members[index]
            if best is None or member["fitness"] > best["fitness"]:
                best = member
        return best

    def best(self):
        best = None
        for member in self.members:
            if best is None or member["fitness"] > best["fitness"]:
                best = member
        return best

    def hashes(self):
        out = []
        for member in self.members:
            out.append(member["program_hash"])
        return out


def keep_first_grafts(program, count):
    """The program with at most its first `count` grafts."""
    grafts = ps.grafts(program)
    if len(grafts) <= count:
        return program
    return ps.with_grafts(program, grafts[:count])


class EvolveArm(SearchArm):
    def __init__(self, config, env):
        SearchArm.__init__(self, config, env)
        self.population = Population(setting(config, "population_size", POPULATION_SIZE))
        self.tournament_size = setting(config, "tournament_size", TOURNAMENT_SIZE)
        self.crossover_rate = setting(config, "crossover_rate", CROSSOVER_RATE)

    def settings(self):
        out = SearchArm.settings(self)
        out["population_size"] = self.population.size
        out["tournament_size"] = self.tournament_size
        out["crossover_rate"] = self.crossover_rate
        out["crossover_max_grafts"] = CROSSOVER_MAX_GRAFTS
        out["initial_heal_steps"] = INITIAL_HEAL_STEPS
        return out

    def search(self):
        self.seed_population()
        self.evolve()

    # Step 1: the initial population.

    def initial_proposals(self):
        """One single-graft program per graftable part, at its default depth with heal 300, in a seeded order."""
        parts = list(space.graftable_parts(self.catalog))
        self.rng.shuffle(parts)
        n_host = space.host_layers(self.config, self.catalog)
        heal = space.heal_with_steps(INITIAL_HEAL_STEPS, self.config)
        proposals = []
        for record in parts:
            at = space.default_at(record, self.config, n_host)
            program = ps.make_program(self.config["host"], [ps.make_graft(record["id"], at)], heal)
            hypothesis = ("initial member: " + space.short_name(record["id"]) + "@" + str(at) +
                          " at its default depth, heal " + str(INITIAL_HEAL_STEPS))
            proposals.append({"program": program, "hypothesis": hypothesis, "parent": None,
                              "extra": {"operator": "init"}})
        return proposals

    def seed_population(self):
        queue = self.initial_proposals()
        while len(self.population.members) < self.population.size and len(queue) > 0 and self.keep_going():
            wanted = min(self.batch_size(), self.population.size - len(self.population.members))
            candidates = self.collect(queue_source(queue), wanted)
            if len(candidates) == 0:
                self.note_stall()
                continue
            rows = self.evaluate(candidates)
            self.add_children(candidates, rows)

    # Steps 2 to 5: breed, screen, score, age.

    def evolve(self):
        while self.keep_going():
            candidates = self.collect(self.next_child, self.batch_size(), CHILD_ATTEMPTS)
            if len(candidates) == 0:
                self.note_stall()
                continue
            rows = self.evaluate(candidates)
            self.add_children(candidates, rows)

    def next_child(self):
        """One child proposal: crossover of two tournament winners, or one mutation of a winner."""
        if len(self.population.members) == 0:
            program = space.random_program(self.rng, self.catalog, self.config)
            return {"program": program, "hypothesis": "evolve: random draft (the population is empty)",
                    "parent": None, "extra": {"operator": "random"}}
        if len(self.population.members) >= 2 and self.rng.random() < self.crossover_rate:
            first = self.population.tournament(self.rng, self.tournament_size)
            second = self.population.tournament(self.rng, self.tournament_size)
            if first["program_hash"] != second["program_hash"]:
                child = space.crossover(first["program"], second["program"], self.rng, self.config)
                child = keep_first_grafts(child, CROSSOVER_MAX_GRAFTS)
                hypothesis = ("evolve: crossover of " + first["describe"] + " (fitness " + "%.3f" % first["fitness"] +
                              ") and " + second["describe"] + " (fitness " + "%.3f" % second["fitness"] + ")")
                return {"program": child, "hypothesis": hypothesis, "parent": first["program_hash"],
                        "extra": {"operator": "crossover", "other_parent": second["program_hash"]}}
        parent = self.population.tournament(self.rng, self.tournament_size)
        child, note = space.mutate_with_note(parent["program"], self.rng, self.catalog, self.config)
        hypothesis = "evolve: " + note + " on " + parent["describe"] + " (fitness " + "%.3f" % parent["fitness"] + ")"
        return {"program": child, "hypothesis": hypothesis, "parent": parent["program_hash"],
                "extra": {"operator": "mutate"}}

    def add_children(self, candidates, rows):
        """Each scored candidate joins the population once, in batch order; the oldest members die."""
        results = self.best_scores(rows)
        removed = []
        for candidate in candidates:
            result = results.get(candidate["program_hash"])
            if result is None:
                continue
            member = {"program": candidate["program"], "program_hash": candidate["program_hash"],
                      "describe": space.describe(candidate["program"]), "fitness": result["combined"],
                      "fidelity": result["fidelity"], "round": candidate["round"]}
            gone = self.population.add(member)
            if gone is not None:
                removed.append(gone["program_hash"])
        self.log_population(removed)

    def log_population(self, removed):
        best = self.population.best()
        text = "population " + str(len(self.population.members)) + " of " + str(self.population.size)
        if best is not None:
            text += ", best " + "%.3f" % best["fitness"] + " (" + best["describe"] + ", " + best["fidelity"] + ")"
        text += ", " + str(len(removed)) + " oldest removed"
        self.env.log_event("population", text, self.agent, self.round_index - 1,
                           extra={"members": self.population.hashes(), "removed": removed})
