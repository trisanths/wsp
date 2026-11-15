"""The state of one swarm run, and the snapshot the web app reads.

Everything here is guarded by one lock per run. Methods that change state add
one to `version`, so the stream knows when to send a new snapshot. Nothing in
this module sleeps or waits while it holds the lock.
"""

import threading
import time

from wsp.swarm import program_schema as ps
from wsp.swarm import space

from studio.server import compare
from studio.server import library
from studio.server import programs
from studio.server import sizing
from studio.server import synth_report

ACTIVITY_LIMIT = 200
THINKING_LIMIT = 100
TOOLS_LIMIT = 100
HISTORY_LIMIT = 60
LEADERBOARD_LIMIT = 30
REFUSED_LOG_LIMIT = 40

PLANNER_BLURBS = ["refines the leading programs", "mixes new parts into the leaders", "explores untried parts"]
FAILED_STATUSES = ["collapsed", "rejected_fit", "compile_error", "invalid", "infra_error"]
SCORED_STATUSES = ["ok", "collapsed"]
FAIL_REASONS = {"rejected_fit": "the part fits the host too poorly", "compile_error": "it did not compile",
                "invalid": "the program is not valid", "infra_error": "the GPU job failed"}


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
        self.size_ctx = None
        self.refused = 0           # proposals the studio dropped for a limit or a tap
        self.status = "starting"
        self.error = None
        self.started_at = time.time()
        self.ended_at = None
        self.base_scores = {}      # filled once by load_base_scores; empty means pending
        self.stopped_early = False  # true when the person pressed Stop before the search finished
        self.agents = {}
        self.order = []
        self.add_agent(new_agent("analyst", "Analyst", "analyst", "maps what each part can do"))
        for index in range(planner_count):
            blurb = PLANNER_BLURBS[index % len(PLANNER_BLURBS)]
            self.add_agent(new_agent("planner-" + str(index + 1), "Planner " + str(index + 1), "planner", blurb))
        self.add_agent(new_agent("judge", "Judge", "judge", "ranks results and retires weak patterns"))
        self.models = library.get()["models"]
        self.pending = {}          # agent id -> program hashes sent to eval and not yet scored
        self.programs = {}         # program hash -> {"row": compact best row, "donors": [...]}
        self.heldout = {}          # program hash -> held-out scores
        self.controls = {}         # program hash -> random-part control scores
        self.curve = []
        self.best_target = None
        self.activity = []
        self.activity_seq = 0
        self.round_seen = None
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

    def readable(self, program, fallback=None):
        """A readable name for a program; fallback (the swarm's describe) when there is no program."""
        if program is None:
            return str(fallback)
        try:
            return programs.readable(program, self.catalog)
        except Exception:
            return str(fallback)

    def guard_report(self, scores):
        """[{id, name, score, floor, ok}] for each guard benchmark in the spec that has a score."""
        out = []
        for item in self.spec.get("benchmarks") or []:
            if item.get("kind") != "guard":
                continue
            value = (scores or {}).get(item["id"])
            if value is None:
                continue
            floor = item.get("floor")
            ok = floor is None or value >= floor
            out.append({"id": item["id"], "name": item.get("name") or item["id"], "score": number(value),
                        "floor": floor, "ok": ok})
        return out

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

    def add_activity(self, agent_id, kind, status, text, data=None, round_index=None):
        """One activity item: a plain sentence in text, and structured fields in data.

        kind groups items on the Activity page: "eval", "probe", "plan", "map",
        "verdict", "heldout", "milestone", "nudge", "error". round is the swarm
        round, or the last round seen when the row has none.
        """
        with self.lock:
            if round_index is not None:
                self.round_seen = round_index
            self.activity_seq += 1
            self.activity.append({"id": self.activity_seq, "ts": time.time(), "agent": agent_id, "kind": kind,
                                  "status": status, "text": clip_text(text, 400), "data": data,
                                  "round": self.round_seen})
            del self.activity[:-ACTIVITY_LIMIT]
            self.touch()

    def note_refused(self, agent_id, describe, problems):
        """Count a proposal dropped for a limit or a tap, and log the first ones."""
        with self.lock:
            self.refused += 1
            count = self.refused
            self.touch()
        if count <= REFUSED_LOG_LIMIT:
            text = "Refused " + describe + ". " + " ".join(problems)
            self.add_activity(agent_id, "refused", "bad", text)
            self.add_tool(agent_id, "check_limits(" + describe + ")", problems[0], "bad")

    def set_run_status(self, status, error=None):
        with self.lock:
            self.status = status
            if error is not None:
                self.error = error
            if status in ("done", "error"):
                self.ended_at = time.time()
                self.finish_agents_locked()
            self.touch()

    def finish_agents_locked(self):
        """When a run ends, no agent is still working: mark every agent done."""
        line = self.end_line()
        for agent_id in self.order:
            agent = self.agents[agent_id]
            agent["status"] = "done"
            agent["now"] = line

    def end_line(self):
        """The now line of every agent after the run ends."""
        if self.stopped_early or self.status == "error":
            return "Stopped"
        return "Finished"

    def outcome_locked(self):
        """"finished", "stopped" or "error" once the run has ended, else None."""
        if self.status == "error":
            return "error"
        if self.status == "done":
            if self.stopped_early:
                return "stopped"
            return "finished"
        return None

    def empty_reason_locked(self, best):
        """Why a run that ended has no best model, in plain words. None while it runs or when there is a best."""
        if self.status not in ("done", "error") or best is not None:
            return None
        if len(self.programs) > 0:
            return ("Every build that was scored failed (it collapsed or did not hold up). "
                    "Try other parts or a longer search.")
        if self.refused > 0:
            return ("Every proposal was refused for going over your limits (" + str(self.refused) +
                    " refused). Raise the size limits on the spec.")
        if self.status == "error":
            return "The run ended with an error before anything was scored."
        if self.stopped_early:
            return "The run was stopped before anything was scored."
        return "The search ended before any build was scored. Raise the eval budget on the spec."

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
        self.add_activity("you", "nudge", "info", text, {"to": agent_id, "to_name": self.display_name(agent_id)})
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
            self.add_activity(agent_id, "nudge_applied", "ok", "Read your nudge and will use it next round",
                              {"nudge": text})
        return texts

    # Candidates sent to eval.

    def note_candidates(self, candidates):
        """Mark each candidate's agent as compiling and remember what it has waiting."""
        with self.lock:
            for candidate in candidates:
                agent_id = self.agent_id_for(candidate.get("agent"))
                program = candidate["program"]
                describe = self.readable(program, space.describe(program))
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
        name = self.readable(row.get("program"), row.get("describe"))
        self.set_status(agent_id, "probing", "Checking fit of " + name)
        fit_values = []
        for key in sorted(row.get("fit_r2") or {}):
            if row["fit_r2"][key] is not None:
                fit_values.append(number(row["fit_r2"][key]))
        if bad:
            text = "Fit check failed for " + name
        else:
            text = "Checked fit of " + name
        self.add_activity(agent_id, "probe", status, text,
                          {"name": name, "fits": fit_values, "program_hash": row.get("program_hash")},
                          row.get("round"))

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
                history = {"program_hash": key, "describe": describe,
                           "name": self.readable(row.get("program"), describe), "scores": dict(scores),
                           "combined": number(row.get("combined")), "status": status}
                agent["history"].append(history)
                del agent["history"][:-HISTORY_LIMIT]
                agent["status"] = "scoring"
                agent["now"] = clip_text("Scoring " + self.readable(row.get("program"), describe), 200)
            is_best = False
            if status in SCORED_STATUSES:
                is_best = self.add_scored_locked(row, agent_id, agent)
            self.touch()
        self.add_eval_tools(row, agent_id, describe, scores, status, is_best)

    def add_scored_locked(self, row, agent_id, agent):
        """Update the curve, the best score and the program table. The lock is held.

        Only a valid (ok) program sets the best score or an agent's best. A collapsed program still
        adds a curve point, with the best score so far. Returns True when this row is a new best.
        """
        key = row["program_hash"]
        target = self.target_score(row.get("scores"))
        is_best = False
        if target is not None:
            if row.get("status") == "ok":
                if self.best_target is None or target > self.best_target:
                    self.best_target = target
                    is_best = True
                if agent is not None and (agent["best"] is None or target > agent["best"]):
                    agent["best"] = number(target)
            self.curve.append({"eval": len(self.curve) + 1, "best": number(self.best_target),
                               "score": number(target)})
        current = self.programs.get(key)
        if current is None or row["combined"] > current["row"]["combined"]:
            donors = self.donors_of(row.get("program"))
            self.programs[key] = {"row": row, "donors": donors, "agent": agent_id,
                                  "synth": self.synth_for(row.get("program"))}
        return is_best

    def synth_for(self, program):
        """The synth part of a scored program (components, stages, drift) from the backend, or None.

        The stub backend gets made-up verdicts, labeled stub. A classic result has none.
        """
        if program is None:
            return None
        backend = None
        if self.env is not None:
            backend = getattr(self.env, "backend", None)
        finder = getattr(backend, "synth_for", None)
        if finder is not None:
            try:
                return finder(program)
            except Exception:
                return None
        if self.stub and self.catalog is not None:
            return synth_report.stub_public(program, self.catalog, self.targets)
        return None

    def add_eval_tools(self, row, agent_id, describe, scores, status, is_best=False):
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
        readable = self.readable(row.get("program"), describe)
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
            target = self.target_score(scores)
            guards = self.guard_report(scores)
            if status == "collapsed":
                failed = []
                for guard in guards:
                    if not guard["ok"]:
                        failed.append(guard["name"])
                text = readable + " collapsed"
                if len(failed) > 0:
                    text = text + ": " + ", ".join(failed) + " fell below the floor"
            elif is_best:
                text = "New best: " + readable
            else:
                text = "Scored " + readable
            data = {"name": readable, "program_hash": row.get("program_hash"), "target": number(target),
                    "best": number(self.best_target), "is_best": is_best, "guards": guards,
                    "scores": dict(scores), "result": status}
            self.add_activity(agent_id, "eval", eval_status, text, data, row.get("round"))
        else:
            reason = FAIL_REASONS.get(status, status)
            self.add_activity(agent_id, "eval", "bad", "Could not score " + readable + ": " + reason,
                              {"name": readable, "program_hash": row.get("program_hash"), "result": status},
                              row.get("round"))

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
            readable = self.readable(row.get("program"), row.get("describe"))
            self.add_activity("judge", "heldout", "info", "Held-out check of " + readable,
                              {"name": readable, "program_hash": key, "target": number(target),
                               "scores": dict(row.get("scores") or {})}, row.get("round"))

    def record_event(self, row):
        kind = row.get("kind")
        if kind in ("reset", "memory_op", "exa"):
            return
        agent_id = self.agent_id_for(row.get("agent"))
        status = "info"
        if kind in ("llm_error", "exa_error", "memory_lag", "compute_overshoot"):
            status = "bad"
        text = str(row.get("text"))
        extra = row.get("extra") or {}
        round_index = row.get("round")
        if kind == "analyst_map":
            # The scripted analyst writes a terse table; an LLM analyst writes prose. Keep the
            # raw text as detail and lead with a plain sentence.
            if self.looks_terse(text):
                self.add_activity("analyst", "map", status, "Updated the map of which parts look most useful",
                                  {"detail": text}, round_index)
            else:
                self.add_activity("analyst", "map", status, text, None, round_index)
            return
        if kind == "judge_verdict":
            self.add_activity("judge", "verdict", status, self.verdict_text(extra), self.verdict_data(extra),
                              round_index)
            return
        if kind == "llm_plan":
            count = extra.get("proposals")
            if count is None:
                count = 0
            plan_text = "Proposed " + str(count) + " program" + ("" if count == 1 else "s")
            if extra.get("repaired"):
                plan_text = plan_text + " (fixed one bad reply)"
            self.add_activity(agent_id, "plan", status, plan_text, {"count": count}, round_index)
            return
        if kind == "search_end":
            self.add_activity("judge", "milestone", "info", "Search finished. Checking the top programs on held-out data.",
                              None, round_index)
            return
        if kind == "stall":
            self.add_activity("judge", "milestone", "info", "No new programs this round", None, round_index)
            return
        if kind == "stop":
            self.add_activity("judge", "milestone", "info", "Search stopped: " + text, None, round_index)
            return
        if status == "bad":
            self.add_activity(agent_id, "error", status, text, None, round_index)
            return
        self.add_activity(agent_id, "system", status, text, None, round_index)

    def looks_terse(self, text):
        """True for machine-style text (estimates, brackets, ids), False for sentences."""
        hits = 0
        for marker in ["est ", "evals,", "at [", ".blk", ".c0", "@"]:
            if marker in text:
                hits += 1
        return hits >= 2

    def verdict_text(self, extra):
        """The judge's verdict as one plain sentence."""
        frontier = extra.get("frontier") or []
        retired = extra.get("retired") or []
        if len(frontier) == 0:
            return "No program is good enough to keep yet"
        with self.lock:
            top = self.programs.get(frontier[0])
        if top is None:
            sentence = "Kept the top " + str(len(frontier)) + " programs"
        else:
            name = self.readable(top["row"].get("program"), top["row"].get("describe"))
            target = self.target_score(top["row"].get("scores"))
            sentence = "Leading: " + name
            if target is not None:
                sentence = sentence + " (" + "%.1f" % (target * 100) + "%)"
        if len(retired) > 0:
            sentence = sentence + ". Retired " + str(len(retired)) + " weak pattern" + ("" if len(retired) == 1 else "s")
        return sentence

    def verdict_data(self, extra):
        """The frontier as [{program_hash, name, target}] for the Activity page."""
        rows = []
        with self.lock:
            for key in (extra.get("frontier") or [])[:5]:
                item = self.programs.get(key)
                if item is None:
                    continue
                rows.append({"program_hash": key,
                             "name": self.readable(item["row"].get("program"), item["row"].get("describe")),
                             "target": number(self.target_score(item["row"].get("scores")))})
        return {"frontier": rows, "retired": len(extra.get("retired") or []), "killed": len(extra.get("killed") or [])}

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

    def sort_key(self, row):
        """Leaderboard order: ok programs first, collapsed last, then the higher mean target score first."""
        group = 1
        if row["status"] == "ok":
            group = 0
        if row["status"] == "collapsed":
            group = 2
        target = self.target_score(row["scores"])
        if target is None:
            target = -1.0
        return (group, -target, -row["combined"])

    def leaderboard_locked(self):
        entries = []
        for key in self.programs:
            item = self.programs[key]
            row = item["row"]
            entries.append((self.sort_key(row), key))
        entries.sort()
        out = []
        for position in range(min(len(entries), LEADERBOARD_LIMIT)):
            key = entries[position][1]
            item = self.programs[key]
            row = item["row"]
            entry = {"rank": position + 1, "program_hash": key, "describe": row["describe"],
                     "name": self.readable(row.get("program"), row["describe"]),
                     "agent": item["agent"], "scores": dict(row["scores"]),
                     "combined": number(row["combined"]), "status": row["status"],
                     "target_score": number(self.target_score(row["scores"])),
                     "vs_control": self.vs_control_of(key), "params_b": self.params_b_of(row),
                     "program": row.get("program")}
            info = item.get("synth")
            entry["pipeline"] = synth_report.pipeline_for(row.get("program") or {})
            entry["components"] = None
            entry["drift"] = None
            if info is not None:
                entry["components"] = info["components"]
                entry["drift"] = info["drift"]
                if info.get("sizes") is not None:
                    entry["params_b"] = number(info["sizes"]["total"] / 1e9)
            entry["comparison"] = self.comparison_locked(entry)
            out.append(entry)
        return out

    def comparison_locked(self, entry):
        """The comparison set for one leaderboard entry. The random reference is the held-out control."""
        key = entry["program_hash"]
        comparison = compare.build(self.spec, entry["scores"], entry["params_b"], self.heldout.get(key),
                                   self.controls.get(key), "heldout", self.stub, self.models,
                                   components=entry.get("components"))
        if entry["status"] != "ok":
            comparison["verdict"] = "failed"
            comparison["headline"] = "Failed attempt. The build ended with status " + str(entry["status"]) + "."
            return comparison
        over = sizing.measured_problems(self.spec.get("limits"), entry["params_b"])
        if len(over) > 0:
            comparison["verdict"] = "failed"
            comparison["headline"] = "Failed attempt. " + " ".join(over)
        return comparison

    def best_locked(self, leaderboard):
        for entry in leaderboard:
            if entry["status"] == "ok":
                return {"program_hash": entry["program_hash"], "describe": entry["describe"], "name": entry["name"],
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
        now = agent["now"]
        if self.status == "paused" and status not in ("done",):
            status = "paused"
        if self.status in ("done", "error"):
            status = "done"
            now = self.end_line()
        return {"id": agent["id"], "name": agent["name"], "role": agent["role"], "blurb": agent["blurb"],
                "status": status, "now": now, "best": agent["best"], "attempts": agent["attempts"],
                "collapses": agent["collapses"], "lessons": agent["lessons"],
                "nudges_pending": agent["nudges_pending"]}

    def load_base_scores(self):
        """Read the host's base scores once, from the run thread. A snapshot never makes this call."""
        if self.env is None:
            return
        try:
            scores = self.env.base_scores("search")
        except Exception as error:
            self.add_activity(None, "error", "bad", "Could not read the host scores: " + str(error))
            return
        with self.lock:
            self.base_scores = dict(scores)
            self.touch()

    def snapshot(self):
        """The run snapshot of the contract."""
        budget = self.budget_dict()
        with self.lock:
            end = self.ended_at
            if end is None:
                end = time.time()
            agents = []
            for agent_id in self.order:
                agents.append(self.agent_public(self.agents[agent_id]))
            leaderboard = self.leaderboard_locked()
            best = self.best_locked(leaderboard)
            comparison = None
            if best is not None:
                for entry in leaderboard:
                    if entry["program_hash"] == best["program_hash"]:
                        comparison = entry["comparison"]
            base_scores = dict(self.base_scores)
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
                "best": best,
                "limits": self.spec.get("limits"),
                "comparison": comparison,
                "refused": self.refused,
                "stopped_early": self.stopped_early,
                "outcome": self.outcome_locked(),
                "empty_reason": self.empty_reason_locked(best),
                "sort": "target score",
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
