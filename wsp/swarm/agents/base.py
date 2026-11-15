"""Agent interfaces (spec section 8).

ctx is a dict every agent call receives:
    config      the run config
    rng         this agent's own random.Random (fixed seed per agent)
    llm         an LLM (interfaces.LLM); NullLLM raises if called
    memory      a Memory
    literature  a Literature
    log         the RunLog
    env         the WspEnv (for probes, the catalog and log_event)
    role        this agent's role name, for example "planner-2"
    retired     program hashes the judge retired as parents
    killed      patterns the judge killed
"""


class Analyst:
    def build_map(self, observation, ctx):
        """Summarize the capability map.

        Returns {"text": str,
                 "by_task": {task: [part ids, best first]},
                 "placements": {part id: [good `at` values, best first]},
                 "notes": [str]}.
        Fit may choose placements but must not rank parts.
        """
        raise NotImplementedError


class Planner:
    def propose(self, observation, cap_map, lessons, priors, n, ctx):
        """Propose up to n programs.

        lessons: memory.recall(...) results. priors: literature.priors().
        Returns a list of {"program": dict, "hypothesis": str (one line),
        "citations": [cite], "probe_first": bool, "parent": hash or None}.
        Set probe_first when a graft sits at a depth whose fit is not known yet.
        Programs must pass space.validate; invalid ones are dropped for free.
        """
        raise NotImplementedError


class Judge:
    def update(self, frontier, rows, ctx):
        """Rank on search scores only and update the frontier.

        frontier: the previous frontier (log rows, best first). rows: this
        round's log rows. Returns {"frontier": [rows, best first],
        "killed": [patterns], "retired": [program hashes], "text": str}.
        A pattern is "<part id>|<early|middle|late>|<heal|noheal>|<full|r<rank>>",
        the last field being the graft's glue rank, for example
        "stub_qmath_l14_c0|middle|heal|r64" (space.graft_pattern).
        """
        raise NotImplementedError


def make_ctx(config, rng, llm, memory, literature, log, env, role):
    """Build the ctx dict for one agent."""
    return {"config": config, "rng": rng, "llm": llm, "memory": memory, "literature": literature,
            "log": log, "env": env, "role": role, "retired": [], "killed": []}
