"""The state of one swarm run, and the snapshot the web app reads.

Everything here is guarded by one lock per run. Methods that change state add
one to `version`, so the stream knows when to send a new snapshot. Nothing in
this module sleeps or waits while it holds the lock.
"""

import threading
import time

from wsp.swarm import program_schema as ps
from wsp.swarm import space

from studio.server import programs

ACTIVITY_LIMIT = 200
THINKING_LIMIT = 100
TOOLS_LIMIT = 100
HISTORY_LIMIT = 60
LEADERBOARD_LIMIT = 30

PLANNER_BLURBS = ["refines the leading programs", "mixes new parts into the leaders", "explores untried parts"]
FAILED_STATUSES = ["collapsed", "rejected_fit", "compile_error", "invalid", "infra_error"]
SCORED_STATUSES = ["ok", "collapsed"]


def clip_text(text, limit):
    text = str(text)
    if len(text) <= limit:
        return text
    return text[:limit - 3] + "..."


def number(value):
    """Round a float for JSON, and turn whole floats into ints."""
    if value is None:
        return None
    value = round(float(value), 4)
    if value == int(value):
        return int(value)
    return value


def new_agent(agent_id, name, role, blurb):
    return {"id": agent_id, "name": name, "role": role, "blurb": blurb, "status": "idle", "now": "Waiting to start",
            "best": None, "attempts": 0, "collapses": 0, "lessons": 0, "nudges_pending": 0,
            "thinking": [], "tools": [], "history": [], "nudges": [], "current": None}


class RunState:
    def __init__(self, run_id, spec, config, backend_name, host_params_b, planner_count, targets):
        self.lock = threading.RLock()
        self.version = 0
        self.run_id = run_id
        self.spec = spec
        self.config = config
        self.backend_name = backend_name
        self.stub = backend_name == "stub"
        self.host_params_b = host_params_b
        self.targets = list(targets)
        self.env = None
        self.catalog = None
        self.status = "starting"
        self.error = None
        self.started_at = time.time()
        self.ended_at = None
        self.agents = {}
        self.order = []
        self.add_agent(new_agent("analyst", "Analyst", "analyst", "maps what each part can do"))
        for index in range(planner_count):
            blurb = PLANNER_BLURBS[index % len(PLANNER_BLURBS)]
            self.add_agent(new_agent("planner-" + str(index + 1), "Planner " + str(index + 1), "planner", blurb))
        self.add_agent(new_agent("judge", "Judge", "judge", "ranks results and retires weak patterns"))
        self.pending = {}          # agent id -> program hashes sent to eval and not yet scored
        self.programs = {}         # program hash -> {"row": compact best row, "donors": [...]}
        self.heldout = {}          # program hash -> held-out scores
        self.controls = {}         # program hash -> random-part control scores
        self.curve = []
        self.best_target = None
        self.activity = []
        self.eval_rows = 0

    # Small helpers.

    def add_agent(self, agent):
        self.agents[agent["id"]] = agent
        self.order.append(agent["id"])

    def touch(self):
        self.version += 1

    def agent_id_for(self, name):
        """The contract agent id for a row's agent name."""
        if name in self.agents or name == "you":
            return name
        if name is None:
            return "judge"
        return "planner-1"

    def display_name(self, agent_id):
        if agent_id == "you":
            return "You"
        if agent_id in self.agents:
            return self.agents[agent_id]["name"]
        return str(agent_id)

    def target_score(self, scores):
        """The mean score over the target benchmarks, or None when none is present."""
        total = 0.0
        count = 0
        for task in self.targets:
            if scores is not None and scores.get(task) is not None:
                total += scores[task]
                count += 1
        if count == 0:
            return None
        return total / count

    def params_b_of(self, row):
        params = (row.get("extra") or {}).get("params")
        if params is None:
            return number(self.host_params_b)
        return number(self.host_params_b + params / 1e9)

    def donors_of(self, program):
        """The donor model of each graft in a program, skipping random controls."""
        donors = []
        for graft in ps.grafts(program or {}):
            if ps.graft_control(graft) is not None:
                continue
            part_id = ps.graft_id(graft)
            if self.catalog is not None and part_id in self.catalog.records:
                model = self.catalog.records[part_id].get("model")
                if model is not None:
                    donors.append(model)
        return donors

    # Agent state. Each method takes the lock itself.

    def set_status(self, agent_id, status, now=None):
        with self.lock:
            agent = self.agents.get(agent_id)
            if agent is None:
                return
            agent["status"] = status
            if now is not None:
                agent["now"] = clip_text(now, 200)
            self.touch()

    def add_thinking(self, agent_id, text):
        with self.lock:
            agent = self.agents.get(agent_id)
            if agent is None:
                return
            agent["thinking"].append({"ts": time.time(), "text": clip_text(text, 600)})
            del agent["thinking"][:-THINKING_LIMIT]
            self.touch()

    def add_tool(self, agent_id, call, result, status):
        with self.lock:
            agent = self.agents.get(agent_id)
            if agent is None:
                return
            agent["tools"].append({"ts": time.time(), "call": clip_text(call, 200),
                                   "result": clip_text(result, 300), "status": status})
            del agent["tools"][:-TOOLS_LIMIT]
            self.touch()

    def set_current(self, agent_id, program, describe):
        with self.lock:
            agent = self.agents.get(agent_id)
            if agent is None:
                return
            agent["current"] = {"program": program, "text": programs.render_or_empty(program),
                                "describe": describe}
            self.touch()

    def set_lessons(self, agent_id, count):
        with self.lock:
            agent = self.agents.get(agent_id)
            if agent is not None:
                agent["lessons"] = count
                self.touch()

    def add_activity(self, agent_id, kind, status, text):
        with self.lock:
            self.activity.append({"ts": time.time(), "agent": agent_id, "kind": kind, "status": status,
                                  "text": clip_text(text, 400)})
            del self.activity[:-ACTIVITY_LIMIT]
            self.touch()

    def set_run_status(self, status, error=None):
        with self.lock:
            self.status = status
            if error is not None:
                self.error = error
            if status in ("done", "error"):
                self.ended_at = time.time()
            self.touch()

    # Nudges.

    def add_nudge(self, agent_id, text):
        """Queue a nudge. Returns False when the agent is unknown."""
        with self.lock:
            agent = self.agents.get(agent_id)
            if agent is None:
                return False
            agent["nudges"].append({"ts": time.time(), "text": text, "applied": False})
            agent["nudges_pending"] += 1
            self.touch()
        self.add_activity(agent_id, "nudge", "info", "Nudge for " + self.display_name(agent_id) + ": " + text)
        return True

    def take_nudges(self, agent_id):
        """Mark the pending nudges applied and return their texts."""
        texts = []
        with self.lock:
            agent = self.agents.get(agent_id)
            if agent is None:
                return texts
            for item in agent["nudges"]:
                if not item["applied"]:
                    item["applied"] = True
                    texts.append(item["text"])
            agent["nudges_pending"] = 0
            if len(texts) > 0:
                self.touch()
        for text in texts:
            self.add_activity(agent_id, "nudge_applied", "ok",
                              self.display_name(agent_id) + " read the nudge: " + text)
        return texts

    # Candidates sent to eval.

    def note_candidates(self, candidates):
        """Mark each candidate's agent as compiling and remember what it has waiting."""
        with self.lock:
            for candidate in candidates:
                agent_id = self.agent_id_for(candidate.get("agent"))
                program = candidate["program"]
                describe = space.describe(program)
                key = space.program_hash(program)
                if agent_id not in self.pending:
                    self.pending[agent_id] = []
                self.pending[agent_id].append(key)
                agent = self.agents.get(agent_id)
                if agent is not None:
                    agent["status"] = "compiling"
                    agent["now"] = clip_text("Compiling " + describe, 200)
                    agent["current"] = {"program": program, "text": programs.render_or_empty(program),
                                        "describe": describe}
            self.touch()

    def clear_candidates(self):
        with self.lock:
            self.pending = {}
            for agent_id in self.agents:
                agent = self.agents[agent_id]
                if agent["role"] == "planner" and agent["status"] in ("compiling", "scoring"):
                    agent["status"] = "idle"
                    agent["now"] = "Waiting for the other agents"
            self.touch()

    # Rows from the env. Each returns the status text to show while the run waits.

    def record_row(self, row):
        """Record one row from the env listener."""
        kind = row.get("type")
        if kind == "eval":
            if row.get("split") == "heldout":
                self.record_heldout(row)
            else:
                self.record_eval(row)
        elif kind == "probe":
            self.record_probe(row)
        elif kind == "event":
            self.record_event(row)

    def record_probe(self, row):
        agent_id = self.agent_id_for(row.get("agent"))
        fits = []
        for key in sorted(row.get("fit_r2") or {}):
            value = row["fit_r2"][key]
            if value is not None:
                fits.append(round(value, 2))
        names = []
        for graft in ps.grafts(row.get("program") or {}):
            names.append(space.short_name(ps.graft_id(graft)) + "@" + str(ps.graft_at(graft)))
        call = "probe(" + ", ".join(names) + ")"
        if len(fits) > 0:
            result = "fit " + ", ".join(str(item) for item in fits)
        else:
            result = str(row.get("status"))
        bad = row.get("status") not in ("ok", "cache_hit")
        status = "bad" if bad else "ok"
        self.add_tool(agent_id, call, result, status)
        self.set_status(agent_id, "probing", "Probing fit of " + str(row.get("describe")))
        self.add_activity(agent_id, "probe", status, self.display_name(agent_id) + " probed " +
                          str(row.get("describe")) + ": " + result)

    def record_eval(self, row):
        agent_id = self.agent_id_for(row.get("agent"))
        status = row.get("status")
        describe = str(row.get("describe"))
        scores = row.get("scores") or {}
        key = row.get("program_hash")
        with self.lock:
            if agent_id in self.pending and key in self.pending[agent_id]:
                self.pending[agent_id].remove(key)
            if status in ("budget_exhausted", "cache_hit"):
                return
            self.eval_rows += 1
            agent = self.agents.get(agent_id)
            if agent is not None:
                agent["attempts"] += 1
                if status == "collapsed":
                    agent["collapses"] += 1
                history = {"program_hash": key, "describe": describe, "scores": dict(scores),
                           "combined": number(row.get("combined")), "status": status}
                agent["history"].append(history)
                del agent["history"][:-HISTORY_LIMIT]
                agent["status"] = "scoring"
                agent["now"] = clip_text("Scoring " + describe, 200)
            if status in SCORED_STATUSES:
                self.add_scored_locked(row, agent_id, agent)
            self.touch()
        self.add_eval_tools(row, agent_id, describe, scores, status)

    def add_scored_locked(self, row, agent_id, agent):
        """Update the curve, the best score and the program table. The lock is held."""
        key = row["program_hash"]
        target = self.target_score(row.get("scores"))
        if target is not None:
            if self.best_target is None or target > self.best_target:
                self.best_target = target
            self.curve.append({"eval": len(self.curve) + 1, "best": number(self.best_target),
                               "score": number(target)})
            if agent is not None and (agent["best"] is None or target > agent["best"]):
                agent["best"] = number(target)
        current = self.programs.get(key)
        if current is None or row["combined"] > current["row"]["combined"]:
            donors = self.donors_of(row.get("program"))
            self.programs[key] = {"row": row, "donors": donors, "agent": agent_id}

    def add_eval_tools(self, row, agent_id, describe, scores, status):
        """Tool calls and the activity line for an eval row. Takes the lock inside the helpers."""
        name = self.display_name(agent_id)
        compile_status = "ok"
        compile_result = "compiled"
        fits = []
        for key in sorted(row.get("fit_r2") or {}):
            if row["fit_r2"][key] is not None:
                fits.append(round(row["fit_r2"][key], 2))
        if len(fits) > 0:
            compile_result = "fit " + ", ".join(str(item) for item in fits)
        if status == "rejected_fit":
            compile_status = "bad"
            compile_result = compile_result + " is below the fit gate"
        elif status in ("compile_error", "invalid"):
            compile_status = "bad"
            compile_result = str((row.get("extra") or {}).get("error") or status)
        self.add_tool(agent_id, "compile(" + describe + ")", compile_result, compile_status)
        if status in SCORED_STATUSES:
            parts = []
            for task in sorted(scores):
                parts.append(task + " " + "%.3f" % scores[task])
            eval_result = ", ".join(parts)
            eval_status = "ok"
            if status == "collapsed":
                eval_status = "bad"
                eval_result = eval_result + " (collapsed)"
            self.add_tool(agent_id, "eval(search)", eval_result, eval_status)
            line = name + " scored " + describe + ": " + eval_result + ", combined " + "%.3f" % row["combined"]
            self.add_activity(agent_id, "eval", eval_status, line)
        else:
            self.add_activity(agent_id, "eval", "bad", name + " could not score " + describe + " (" + status + ")")

    def record_heldout(self, row):
        key = row.get("program_hash")
        extra = row.get("extra") or {}
        with self.lock:
            if row.get("status") == "control":
                if extra.get("control") == "random" and extra.get("control_of") is not None:
                    self.controls[extra["control_of"]] = dict(row.get("scores") or {})
            elif row.get("status") in SCORED_STATUSES:
                self.heldout[key] = dict(row.get("scores") or {})
            self.touch()
        if row.get("status") in SCORED_STATUSES:
            target = self.target_score(row.get("scores"))
            text = "Held-out check of " + str(row.get("describe")) + ": "
            if target is not None:
                text = text + "target " + "%.3f" % target
            self.add_activity("judge", "heldout", "info", text)

    def record_event(self, row):
        kind = row.get("kind")
        if kind in ("reset", "memory_op", "exa"):
            return
        agent_id = self.agent_id_for(row.get("agent"))
        status = "info"
        if kind in ("llm_error", "exa_error", "memory_lag", "compute_overshoot"):
            status = "bad"
        text = str(row.get("text"))
        if kind == "analyst_map":
            agent_id = "analyst"
        if kind == "judge_verdict":
            agent_id = "judge"
        self.add_activity(agent_id, kind, status, text)

    def agent_after_row(self, row):
        """Set the agent's status after a row has been shown for a while."""
        kind = row.get("type")
        agent_id = self.agent_id_for(row.get("agent"))
        with self.lock:
            agent = self.agents.get(agent_id)
            if agent is None:
                return
            if kind == "probe":
                agent["status"] = "thinking"
                agent["now"] = "Choosing candidates"
            elif kind == "eval" and row.get("split") != "heldout":
                waiting = self.pending.get(agent_id) or []
                if len(waiting) > 0:
                    agent["status"] = "compiling"
                    agent["now"] = "Compiling the next candidate"
                else:
                    agent["status"] = "idle"
                    agent["now"] = "Waiting for the other agents"
            self.touch()

    # Snapshots.

    def budget_dict(self):
        if self.env is None:
            return {"evals_used": 0, "evals_total": self.config["eval_budget"], "compute_used_s": 0,
                    "compute_total_s": self.config["compute_budget_s"]}
        budget = self.env.budget
        return {"evals_used": number(budget.eval_used()), "evals_total": number(budget.eval_budget),
                "compute_used_s": number(budget.compute_used()), "compute_total_s": number(budget.compute_budget_s)}

    def vs_control_of(self, key):
        """Held-out target score minus the random-part control's, or None when either is missing."""
        if key not in self.heldout or key not in self.controls:
            return None
        real = self.target_score(self.heldout[key])
        control = self.target_score(self.controls[key])
        if real is None or control is None:
            return None
        return number(real - control)

    def leaderboard_locked(self):
        entries = []
        for key in self.programs:
            item = self.programs[key]
            row = item["row"]
            entries.append((-row["combined"], key))
        entries.sort()
        out = []
        for position in range(min(len(entries), LEADERBOARD_LIMIT)):
            key = entries[position][1]
            item = self.programs[key]
            row = item["row"]
            out.append({"rank": position + 1, "program_hash": key, "describe": row["describe"],
                        "agent": item["agent"], "scores": dict(row["scores"]),
                        "combined": number(row["combined"]), "status": row["status"],
                        "vs_control": self.vs_control_of(key), "params_b": self.params_b_of(row),
                        "program": row.get("program")})
        return out

    def best_locked(self, leaderboard):
        for entry in leaderboard:
            if entry["status"] == "ok":
                return {"program_hash": entry["program_hash"], "describe": entry["describe"],
                        "scores": entry["scores"], "combined": entry["combined"], "agent": entry["agent"],
                        "program": entry["program"]}
        return None

    def sources_locked(self):
        counts = {}
        total = len(self.programs)
        for key in self.programs:
            for model in self.programs[key]["donors"]:
                counts[model] = counts.get(model, 0) + 1
        out = [{"model": self.config["host"], "role": "host", "uses": total}]
        for model in sorted(counts, key=lambda name: (-counts[name], name)):
            out.append({"model": model, "role": "donor", "uses": counts[model]})
        return out

    def agent_public(self, agent):
        status = agent["status"]
        if self.status == "paused" and status not in ("done",):
            status = "paused"
        if self.status in ("done", "error"):
            status = "done"
        return {"id": agent["id"], "name": agent["name"], "role": agent["role"], "blurb": agent["blurb"],
                "status": status, "now": agent["now"], "best": agent["best"], "attempts": agent["attempts"],
                "collapses": agent["collapses"], "lessons": agent["lessons"],
                "nudges_pending": agent["nudges_pending"]}

    def snapshot(self):
        """The run snapshot of the contract."""
        base_scores = {}
        if self.env is not None:
            base_scores = self.env.base_scores("search")
        budget = self.budget_dict()
        with self.lock:
            end = self.ended_at
            if end is None:
                end = time.time()
            agents = []
            for agent_id in self.order:
                agents.append(self.agent_public(self.agents[agent_id]))
            leaderboard = self.leaderboard_locked()
            snapshot = {
                "run_id": self.run_id,
                "spec": self.spec,
                "status": self.status,
                "stub": self.stub,
                "backend": self.backend_name,
                "started_at": self.started_at,
                "elapsed_s": round(end - self.started_at, 1),
                "budget": budget,
                "base_scores": base_scores,
                "best": self.best_locked(leaderboard),
                "agents": agents,
                "leaderboard": leaderboard,
                "activity": list(self.activity),
                "curve": list(self.curve),
                "sources": self.sources_locked(),
            }
            if self.error is not None:
                snapshot["error"] = self.error
            return snapshot

    def agent_detail(self, agent_id):
        """The agent detail of the contract, or None for an unknown agent."""
        with self.lock:
            agent = self.agents.get(agent_id)
            if agent is None:
                return None
            return {"agent": self.agent_public(agent), "current": agent["current"],
                    "thinking": list(agent["thinking"]), "tools": list(agent["tools"]),
                    "history": list(agent["history"]), "nudges": list(agent["nudges"])}
