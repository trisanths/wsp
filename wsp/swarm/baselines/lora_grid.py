"""The LoRA arm: regular fine-tuning of the host at the same eval budget (part 2's update: "LoRA ... at equal budget").

Each program is a host LoRA with no grafts, trained by the program's heal
pass on the same heal data the grafts heal on (Trisanth's compiler: a
LoRA spec in the program, wsp/lora.py). It runs through env.step exactly
like a graft program: the same compile, heal and eval path, the same
budgets and the same ledger, so on the Modal backend it trains and grades
on the same GPUs with the same scorer.

LoRA needs its own learning rate (round 3: rank 64 at the glue's 3e-4
scored 28.8% on GSM8K; round 4: 46.7% at 5e-5), so the arm tunes it. The
grid is config lora_grid: every rank x learning rate x heal steps, in a
seeded random order, k per round, until the eval budget or the grid runs
out. With the default grid (4 ranks, 3 learning rates, 2 step counts) a
24-eval budget covers the whole grid once. A tuned LoRA at equal budget is
the fair "versus regular training" bar: it gets as many evals to tune as
the swarm gets to search.

Held-out scores the top programs once with no controls: the random-part
and no-graft controls ask whether a graft's part matters, and a LoRA
program has no graft.
"""

from wsp.swarm import program_schema as ps
from wsp.swarm import space
from wsp.swarm.baselines import SearchArm, queue_source


def grid_programs(config):
    """Every LoRA program in config lora_grid, in a fixed order (steps, then rank, then learning rate)."""
    grid = space.lora_grid(config)
    data = config["task_types"][config["task_type"]]["heal_data"][0]
    programs = []
    for steps in grid["steps"]:
        for rank in grid["ranks"]:
            for lr in grid["lrs"]:
                heal = ps.make_heal(steps=steps, data=data, lr=lr)
                programs.append(ps.make_lora_program(config["host"], rank, heal))
    return programs


class LoRAGridArm(SearchArm):
    def __init__(self, config, env):
        SearchArm.__init__(self, config, env)
        programs = grid_programs(config)
        self.rng.shuffle(programs)
        self.queue = []
        for program in programs:
            lora = ps.lora(program)
            heal = ps.heal(program)
            self.queue.append({"program": program,
                               "hypothesis": ("host LoRA rank " + str(lora["rank"]) + ", lr " + ("%g" % heal["lr"]) +
                                              ", " + str(heal["steps"]) + " steps"),
                               "parent": None, "extra": {"operator": "lora_grid"}})

    def search(self):
        source = queue_source(self.queue)
        while self.keep_going():
            if len(self.queue) == 0:
                self.stop("the LoRA grid is spent")
                return
            candidates = self.collect(source, self.batch_size())
            if len(candidates) == 0:
                self.note_stall()
                continue
            self.evaluate(candidates)

    def final_heldout(self):
        """Held-out once on the top heldout_top programs, with no controls (a LoRA program has no graft)."""
        heldout = []
        if self.env.heldout_budget is None:
            return heldout, []
        for row in self.frontier()[:self.config["heldout_top"]]:
            heldout.append(self.env.heldout_eval(row["program"], self.agent, "final pick from the LoRA grid"))
        return heldout, []
