"""The random-part search arm (papers.md 8.4 and top recommendation 2).

Why it exists. A graft program can score well for reasons that have nothing
to do with the part it grafts: the glue is a trainable adapter, the heal
trains it on task data, and the host depth decides how much an adapter
helps. Trisanth's round 2 sweep (results/README.md, search split, 3 seeds)
found exactly that for the first pair: no part-selection method beat the
random-units control at any donor layer, host layer 6 gave about 38 to 40
percent against about 22 to 26 percent at host layers 10 and 14 whatever
the part was, and rank-64 glue did as well as full glue. So a search arm
can climb by finding good placements and heal settings alone.

This arm measures that. It is the evolutionary baseline (evolve.py) with
the same population, tournament, mutations, crossover, screen, fit gate,
cascade, budgets and log, except that every program it evaluates has every
graft set to control "random" (program_schema.as_random_control): the same
donor, donor layer, kind, unit count, host depth, glue and heal, with the
part's units replaced by random units. Its fit R squared equals the real
part's, because fit depends only on the slot. Its best score is what
placement, glue and heal reach with the part content removed. The swarm (or
evolve) has shown that parts matter only if it beats this arm at equal
budget, on held-out, by more than one paired standard error.

Details:
- A graft that is already a control keeps its control seed, so a child
  inherits its parent's random draw; a new graft (from an add, a swap or a
  random draft) gets a fresh seed from this arm's own rng. A lineage can
  therefore keep a lucky draw, which is what "best of N random" means.
- The control-seed rng is separate from the search rng, so with the same
  seed this arm starts from the same 12 initial (part, depth) slots as
  evolve, which pairs the two arms' first rounds.
- Glue rank is searched exactly as in evolve: space.mutate's "rank" move
  and space.random_program choose from config glue_ranks, and
  as_random_control keeps each graft's glue rank.
"""

import random

from wsp.swarm import program_schema as ps
from wsp.swarm.baselines.evolve import EvolveArm

CONTROL_SEED_SALT = 211


class RandomPartArm(EvolveArm):
    def __init__(self, config, env):
        EvolveArm.__init__(self, config, env)
        self.control_rng = random.Random(config["seed"] * 1000 + CONTROL_SEED_SALT)

    def settings(self):
        out = EvolveArm.settings(self)
        out["control"] = ps.CONTROL_RANDOM
        return out

    def prepare(self, program):
        """Every graft becomes a random-part control; existing controls keep their seed."""
        grafts = []
        for graft in ps.grafts(program):
            if ps.graft_control(graft) is None:
                graft = ps.as_random_control(graft, self.control_rng.randrange(1, 1000000))
            grafts.append(graft)
        return ps.with_grafts(program, grafts)
