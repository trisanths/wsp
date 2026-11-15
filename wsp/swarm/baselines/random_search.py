"""Random search: the sanity floor (design section 10, papers.md 7.4).

Each round draws programs with space.random_program (one to three random
grafts, each at a random host depth with a random glue rank, and random heal
steps) until it has k new ones, and scores them through env.step_many. A
program hash is never proposed twice. With config cascade the batch is
scored at "low" and the top third goes on to "full", the swarm's rule.

The draws go through the same screen as every other arm's proposals
(SearchArm.screen: drop known low-fit slots for free, probe untested slots
and drop low fits). The fit gate is shared machinery (rule 4), so the only
thing that differs between this arm and the others is how a program is
proposed: here uniformly at random, which makes the arm uniform over the
programs that pass the gate. Without the screen, most random depths are
rejected by the env's gate at step time, each costing a compile, and the
floor would be low on the compute axis for a reason that has nothing to do
with the proposer.
"""

from wsp.swarm import space
from wsp.swarm.baselines import SearchArm

DRAW_ATTEMPTS = 60          # draws per round before the round counts as a stall


class RandomSearchArm(SearchArm):
    def search(self):
        while self.keep_going():
            candidates = self.collect(self.next_draft, self.batch_size(), DRAW_ATTEMPTS)
            if len(candidates) == 0:
                self.note_stall()
                continue
            self.evaluate(candidates)

    def next_draft(self):
        program = space.random_program(self.rng, self.catalog, self.config)
        return {"program": program, "hypothesis": "random draft", "parent": None, "extra": {"operator": "random"}}
