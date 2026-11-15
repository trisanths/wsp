"""Pause and stop control, agent proxies and the swarm subclass used by the studio.

The proxies wrap the swarm's agents so the run state can show what each agent
is doing. They forward every call to the real agent and never change its
result, except that the planner proxy adds pending user nudges to the lessons.
"""

import threading
import time

from wsp.swarm import space
from wsp.swarm.swarm import Swarm


class Gate:
    """Pause, resume and stop flags that worker threads can wait on."""

    def __init__(self):
        self.cond = threading.Condition()
        self.paused = False
        self.stopped = False

    def pause(self):
        with self.cond:
            self.paused = True
            self.cond.notify_all()

    def resume(self):
        with self.cond:
            self.paused = False
            self.cond.notify_all()

    def stop(self):
        with self.cond:
            self.stopped = True
            self.cond.notify_all()

    def wait_if_paused(self):
        """Block while the run is paused. A stop releases the wait."""
        with self.cond:
            while self.paused and not self.stopped:
                self.cond.wait(0.2)

    def sleep(self, seconds):
        """Sleep in short slices. A stop ends the sleep early. Then wait out any pause."""
        end = time.time() + seconds
        while time.time() < end and not self.stopped:
            time.sleep(min(0.05, max(0.0, end - time.time())))
        self.wait_if_paused()


class Controller:
    """What the proxies, the env listener and the swarm subclass share for one run."""

    def __init__(self, state, pace_s, seed_program=None):
        self.state = state
        self.gate = Gate()
        self.pace_s = pace_s
        self.seed_program = seed_program

    def think_pause(self, factor=1.0):
        """A short pause after an agent call, in stub mode only, so the UI can show it."""
        if self.state.stub and self.pace_s > 0:
            self.gate.sleep(self.pace_s * factor)

    def on_row(self, row):
        """Env listener: record the row, wait a moment in stub mode, then settle the agent status."""
        self.state.record_row(row)
        kind = row.get("type")
        pause = 0.0
        if self.state.stub and self.pace_s > 0 and not self.gate.stopped:
            if kind == "eval" and row.get("split") != "heldout":
                pause = self.pace_s
            elif kind == "probe":
                pause = self.pace_s * 0.4
            elif kind == "event" and row.get("kind") in ("analyst_map", "judge_verdict"):
                pause = self.pace_s * 0.5
        if pause > 0:
            self.gate.sleep(pause)
        else:
            self.gate.wait_if_paused()
        self.state.agent_after_row(row)


class AgentProxy:
    """Forwards everything to the real agent. Subclasses add status tracking."""

    def __init__(self, real, controller, agent_id):
        self.real = real
        self.controller = controller
        self.agent_id = agent_id
        self.name = real.name

    def __getattr__(self, name):
        return getattr(self.real, name)


class AnalystProxy(AgentProxy):
    def build_map(self, observation, ctx):
        state = self.controller.state
        state.set_status(self.agent_id, "thinking", "Building the capability map")
        try:
            result = self.real.build_map(observation, ctx)
        finally:
            self.controller.think_pause()
        text = str(result.get("text", ""))
        state.add_thinking(self.agent_id, text)
        state.add_tool(self.agent_id, "build_map(history " + str(len(observation.get("history", []))) + " rows)",
                       text[:120], "ok")
        state.set_status(self.agent_id, "idle", "Waiting for the planners")
        return result


class PlannerProxy(AgentProxy):
    def propose(self, observation, cap_map, lessons, priors, n, ctx):
        state = self.controller.state
        state.set_lessons(self.agent_id, len(lessons))
        state.set_status(self.agent_id, "thinking", "Proposing programs")
        merged = list(lessons)
        for text in state.take_nudges(self.agent_id):
            merged.append({"text": "User nudge: " + text, "source_id": "studio-nudge", "occurred_at": time.time(),
                           "kind": "nudge"})
        try:
            proposals = self.real.propose(observation, cap_map, merged, priors, n, ctx)
        except Exception:
            state.set_status(self.agent_id, "idle", "Proposal failed")
            raise
        for proposal in proposals:
            state.add_thinking(self.agent_id, str(proposal.get("hypothesis", "")))
        if len(proposals) > 0:
            first = proposals[0]
            state.set_current(self.agent_id, first["program"], space.describe(first["program"]))
        self.controller.think_pause()
        state.set_status(self.agent_id, "thinking", "Choosing candidates")
        return proposals


class JudgeProxy(AgentProxy):
    def update(self, frontier, rows, ctx):
        state = self.controller.state
        state.set_status(self.agent_id, "thinking", "Ranking the new results")
        try:
            verdict = self.real.update(frontier, rows, ctx)
        finally:
            self.controller.think_pause()
        text = str(verdict.get("text", ""))
        state.add_thinking(self.agent_id, text)
        state.add_tool(self.agent_id, "update(" + str(len(rows)) + " rows)", text[:200], "ok")
        state.set_status(self.agent_id, "idle", "Waiting for the next round")
        return verdict


class StudioSwarm(Swarm):
    """The swarm, with pause and stop, a seed program from the builder, and status tracking."""

    def __init__(self, config, env, analyst, planners, judge, memory, literature, llm, controller):
        Swarm.__init__(self, config, env, analyst, planners, judge, memory, literature, llm)
        self.controller = controller

    def search_over(self):
        self.controller.gate.wait_if_paused()
        if self.controller.gate.stopped:
            return True
        return Swarm.search_over(self)

    def evaluate(self, candidates):
        self.controller.state.note_candidates(candidates)
        try:
            return Swarm.evaluate(self, candidates)
        finally:
            self.controller.state.clear_candidates()

    def run(self):
        self.score_seed_program()
        return Swarm.run(self)

    def score_seed_program(self):
        """Evaluate the program handed over from the builder before the first round."""
        program = self.controller.seed_program
        if program is None:
            return
        self.controller.state.add_activity("you", "handoff", "info", "Scoring the program from the builder first")
        info = self.env.step(program, "you", "Program handed over from the builder")[4]
        if info["status"] == "ok":
            self.frontier = [info]
        if info["program_hash"] is not None:
            self.tried.add(info["program_hash"])


def wrap_agents(analyst, planners, judge, controller):
    """Return proxies for the swarm's agents."""
    wrapped = []
    for index in range(len(planners)):
        wrapped.append(PlannerProxy(planners[index], controller, "planner-" + str(index + 1)))
    return AnalystProxy(analyst, controller, "analyst"), wrapped, JudgeProxy(judge, controller, "judge")
