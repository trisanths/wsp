"""The swarm loop (spec section 8).

1. memory.start_run, literature.priors(), env.reset().
2. Each round:
   - rebuild the analyst map in round 0 and every analyst_every rounds;
   - memory.flush();
   - each planner recalls lessons, then the planners propose in parallel;
   - drop invalid programs, duplicates (same program_hash), killed patterns
     and grafts whose known fit is below the gate; probe the proposals marked
     probe_first and drop those the fit gate would reject;
   - env.step_many on up to k candidates, then judge.update; the verdict is
     logged (and reaches memory through the listener).
3. Stop when either search budget axis is spent, then end the search.
4. Score the judge's top heldout_top programs on held-out once, with their
   two controls.

With config cascade=True, candidates are first scored at fidelity "low" and
the top third is promoted to "full" in the same round. The judge ranks
full-fidelity scores only.
"""

import concurrent.futures
import math
import random

from wsp.swarm import program_schema as ps
from wsp.swarm import space
from wsp.swarm.agents.base import make_ctx
from wsp.swarm.config import fidelity_cost, public_config
from wsp.swarm.interfaces import make_memory_listener


class Swarm:
    def __init__(self, config, env, analyst, planners, judge, memory, literature, llm):
        self.config = config
        self.env = env
        self.analyst = analyst
        self.planners = planners
        self.judge = judge
        self.memory = memory
        self.literature = literature
        self.llm = llm
        self.rng = random.Random(config["seed"])
        seed = config["seed"]
        self.analyst_ctx = self.ctx(random.Random(seed * 1000 + 101), "analyst")
        self.judge_ctx = self.ctx(random.Random(seed * 1000 + 102), "judge")
        self.planner_ctxs = []
        for index in range(len(planners)):
            self.planner_ctxs.append(self.ctx(random.Random(seed * 1000 + index + 1), planners[index].name))
        self.frontier = []
        self.killed = []
        self.retired = []
        self.tried = set()
        self.round_index = 0

    def ctx(self, rng, role):
        return make_ctx(self.config, rng, self.llm, self.memory, self.literature, self.env.log, self.env, role)

    def roles(self):
        names = ["analyst"]
        for planner in self.planners:
            names.append(planner.name)
        names.append("judge")
        return names

    def first_fidelity(self):
        if self.config["cascade"]:
            return "low"
        return "full"

    def search_over(self):
        if self.env.search_done():
            return True
        cost = fidelity_cost(self.config, self.first_fidelity())
        return not self.env.budget.can_spend(cost)

    def run(self):
        """Run the whole search and the held-out pass. Returns the summary dict."""
        env = self.env
        self.memory.start_run(self.config["run_id"], self.roles())
        priors = self.literature.priors()
        env.listeners.append(make_memory_listener(self.memory))
        env.write_manifest({"literature": self.literature.mode, "priors_sha": priors.get("sha", ""),
                            "memory": self.config["memory"], "llm": self.config["llm"],
                            "planners": len(self.planners), "k": self.config["k"]})
        observation, info = env.reset(self.config["seed"])
        cap_map = None
        stalls = 0
        while not self.search_over():
            if self.round_index >= self.config["max_rounds"]:
                env.log_event("stop", "stopped at max_rounds", round_index=self.round_index)
                break
            if cap_map is None or self.round_index % self.config["analyst_every"] == 0:
                cap_map = self.analyst.build_map(observation, self.analyst_ctx)
                env.log_event("analyst_map", cap_map["text"], "analyst", self.round_index)
            flushed = self.memory.flush()
            if flushed.get("lag"):
                env.log_event("memory_lag", "memory flush timed out", round_index=self.round_index,
                              extra={"pending": flushed.get("pending")})
            observation["frontier"] = list(self.frontier)
            proposals = self.propose_all(observation, cap_map, priors)
            candidates = self.select(proposals, observation)
            if len(candidates) == 0:
                stalls += 1
                env.log_event("stall", "no new candidates this round", round_index=self.round_index)
                if stalls >= self.config["max_stall_rounds"]:
                    env.log_event("stop", "stopped after " + str(stalls) + " rounds without candidates",
                                  round_index=self.round_index)
                    break
                self.round_index += 1
                observation = env.observation()
                continue
            stalls = 0
            rows = self.evaluate(candidates)
            verdict = self.judge.update(self.frontier, rows, self.judge_ctx)
            self.frontier = verdict["frontier"]
            self.killed = verdict["killed"]
            self.retired = verdict["retired"]
            self.share_judge_state()
            hashes = []
            for row in self.frontier:
                hashes.append(row["program_hash"])
            env.log_event("judge_verdict", verdict["text"], "judge", self.round_index,
                          extra={"frontier": hashes, "killed": list(self.killed), "retired": list(self.retired)})
            self.round_index += 1
            observation = env.observation()

        env.end_search("swarm finished after " + str(self.round_index) + " rounds")
        heldout, controls = self.final_heldout()
        self.memory.end_run(cleanup=self.config["memory_cleanup"])
        summary = self.summary(heldout, controls)
        env.log.write_json("summary.json", summary)
        return summary

    def share_judge_state(self):
        for ctx in self.planner_ctxs:
            ctx["retired"] = list(self.retired)
            ctx["killed"] = list(self.killed)

    def per_planner(self):
        return int(math.ceil(self.config["k"] / float(max(1, len(self.planners))))) + 1

    def propose_all(self, observation, cap_map, priors):
        """Each planner recalls, then proposes, in parallel. Returns lists in planner order."""
        n = self.per_planner()
        task_text = self.config["task_type"]

        def work(index):
            planner = self.planners[index]
            ctx = self.planner_ctxs[index]
            lessons = self.memory.recall(planner.name, "collapses, fit rejections and best grafts for " + task_text)
            return planner.propose(observation, cap_map, lessons, priors, n, ctx)

        results = []
        if len(self.planners) <= 1:
            for index in range(len(self.planners)):
                results.append(work(index))
            return results
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(self.planners)) as pool:
            futures = []
            for index in range(len(self.planners)):
                futures.append(pool.submit(work, index))
            for future in futures:
                results.append(future.result())
        return results

    def interleave(self, proposal_lists):
        """Round-robin over planners so each gets a fair share of the round."""
        out = []
        longest = 0
        for items in proposal_lists:
            if len(items) > longest:
                longest = len(items)
        for position in range(longest):
            for index in range(len(proposal_lists)):
                if position < len(proposal_lists[index]):
                    out.append((self.planners[index].name, proposal_lists[index][position]))
        return out

    def batch_size(self):
        cost = fidelity_cost(self.config, self.first_fidelity())
        affordable = int(math.floor(self.env.budget.eval_remaining() / cost + 1e-9))
        return max(0, min(self.config["k"], affordable))

    def select(self, proposal_lists, observation):
        """Filter proposals into at most k candidates. Tops up with mutations or random drafts."""
        wanted = self.batch_size()
        chosen = []
        seen = set()
        for agent, proposal in self.interleave(proposal_lists):
            if len(chosen) >= wanted:
                break
            candidate = self.screen(agent, proposal, seen)
            if candidate is not None:
                chosen.append(candidate)
        attempts = 0
        while len(chosen) < wanted and attempts < 30:
            attempts += 1
            proposal = self.top_up_proposal()
            candidate = self.screen("swarm-topup", proposal, seen)
            if candidate is not None:
                chosen.append(candidate)
        return chosen

    def top_up_proposal(self):
        catalog = self.env.catalog
        if len(self.frontier) > 0 and self.rng.random() < 0.5:
            parent = self.frontier[self.rng.randrange(len(self.frontier))]
            child, note = space.mutate_with_note(parent["program"], self.rng, catalog, self.config)
            return {"program": child, "hypothesis": "top-up: " + note + " on " + parent["describe"],
                    "citations": [], "probe_first": True, "parent": parent["program_hash"]}
        program = space.random_program(self.rng, catalog, self.config)
        return {"program": program, "hypothesis": "top-up: random draft", "citations": [],
                "probe_first": True, "parent": None}

    def screen(self, agent, proposal, seen):
        """Return a step candidate, or None when the proposal should be dropped (free)."""
        env = self.env
        catalog = env.catalog
        program = proposal.get("program")
        if len(space.validate(program, catalog, self.config)) > 0:
            return None
        program = ps.normalize(program)
        program_hash = space.program_hash(program)
        if program_hash in self.tried or program_hash in seen:
            return None
        for pattern in space.program_patterns(program, catalog, self.config):
            if pattern in self.killed:
                return None
        unknown = False
        for graft in ps.grafts(program):
            slot = space.graft_slot(graft, catalog, self.config)
            fit = None
            if slot is not None:
                fit = env.fit_of_slot(slot)
            if fit is None:
                unknown = True
            elif fit < self.config["fit_threshold"]:
                self.tried.add(program_hash)
                return None
        if unknown and proposal.get("probe_first"):
            result = env.probe(program, agent, self.round_index, proposal.get("hypothesis", ""))
            if result["status"] not in ("ok", "cache_hit"):
                self.tried.add(program_hash)
                return None
            for key in result["fit_r2"]:
                value = result["fit_r2"][key]
                if value is not None and value < self.config["fit_threshold"]:
                    self.tried.add(program_hash)
                    return None
        seen.add(program_hash)
        extra = {"parent": proposal.get("parent")}
        return {"program": program, "agent": agent, "hypothesis": proposal.get("hypothesis", ""),
                "citations": proposal.get("citations") or [], "round": self.round_index,
                "fidelity": self.first_fidelity(), "extra": extra}

    def evaluate(self, candidates):
        """Step the candidates; with the cascade, promote the top third to full fidelity."""
        rows = self.env.step_many(candidates)
        for row in rows:
            if row["program_hash"] is not None:
                self.tried.add(row["program_hash"])
        if not self.config["cascade"]:
            return rows
        scored = []
        for index in range(len(rows)):
            if rows[index]["status"] == "ok":
                scored.append((-rows[index]["combined"], index))
        scored.sort()
        promote_count = int(math.ceil(len(rows) / 3.0))
        promoted = []
        for pair in scored[:promote_count]:
            candidate = dict(candidates[pair[1]])
            candidate["fidelity"] = "full"
            promoted.append(candidate)
        if len(promoted) == 0:
            return rows
        return rows + self.env.step_many(promoted)

    def final_heldout(self):
        """Held-out once on the judge's top programs, each with its two controls."""
        heldout = []
        controls = []
        if self.env.heldout_budget is None:
            return heldout, controls
        for row in self.frontier[:self.config["heldout_top"]]:
            heldout.append(self.env.heldout_eval(row["program"], "judge", "final pick from the search frontier"))
            for control_row in self.env.run_controls(row["program"], "judge"):
                controls.append(control_row)
        return heldout, controls

    def summary(self, heldout, controls):
        budget = self.env.budget
        best = None
        if len(self.frontier) > 0:
            top = self.frontier[0]
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
            "run_id": self.config["run_id"],
            "method": self.config["method"],
            "backend": self.env.backend.name,
            "stub": self.env.is_stub(),
            "seed": self.config["seed"],
            "rounds": self.round_index,
            "eval_used": budget.eval_used(),
            "eval_budget": budget.eval_budget,
            "compute_used_s": budget.compute_used(),
            "compute_budget_s": budget.compute_budget_s,
            "compute_overshoot_s": budget.overshoot_s(),
            "best_search": best,
            "heldout": heldout_out,
            "controls": controls_out,
            "log_path": self.env.log.path,
            "config": public_config(self.config),
        }


def short_row(row):
    return {"program_hash": row["program_hash"], "describe": row["describe"], "combined": row["combined"],
            "scores": row["scores"], "status": row["status"], "split": row["split"],
            "fit_r2": row["fit_r2"], "token": row["token"]}


def run_swarm(config):
    """Build everything from config, run one swarm, and return its summary."""
    from wsp.swarm import factory

    run = factory.build_run(config)
    analyst, planners, judge = factory.build_agents(config)
    swarm = Swarm(config, run["env"], analyst, planners, judge, run["memory"], run["literature"], run["llm"])
    return swarm.run()
