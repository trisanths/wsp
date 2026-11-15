"""WspEnv: step(program) = eval(compile(program), "search"), with the fairness rules built in.

Gymnasium-style (design section 14):
    observation, info = env.reset()
    observation, reward, terminated, truncated, info = env.step(program, agent, hypothesis)
    rows = env.step_batch(candidates)          # same as step_many
terminated is always False. truncated turns True when the search budget is
spent (either axis) or the arm calls end_search(). info is the full log row.
The budget is never reset: one episode is one budgeted search.

Order inside every step (spec 5, rule 3), implemented only here:
    1. validate                 invalid programs are free
    2. cache on (hash, split, fidelity)   a hit is free and returns cached scores
    3. compile                  charges compute, gives fit_r2 per graft
    4. fit gate                 any fit_r2 below config fit_threshold: rejected_fit, no token
    5. heal                     charges compute (only when the program has a heal)
    6. spend one eval token     (a fraction of one at lower fidelity)
    7. eval                     charges compute
A compile error charges the compute it used and no token. A failure after
the token spend is infra_error and keeps the token (no refunds).

step_many runs a batch in five phases so the results do not depend on thread
timing: checks (in order), compile and heal (threads), token spends (in
order), evals (threads), rows (in order). Compute is checked once at the
start of a batch and charged after the work, so a batch may overshoot the
compute cap by at most its own jobs; the overshoot is logged.

probe() is the fit-only compile (the program with heal set to None): it
charges compute, never a token, and is open to every arm (rule 4).

Held-out (rule 7): heldout_eval and run_controls work only after the search
is over, use the separate held-out budget, take at most config heldout_top
programs per run, each once, and never feed the observation.
"""

import concurrent.futures
import random
import threading

from wsp.swarm import program_schema as ps
from wsp.swarm import space
from wsp.swarm.config import all_tasks, fidelity_cost, guard_tasks, task_weights
from wsp.swarm.log import new_eval_row, new_event_row, new_probe_row
from wsp.swarm.scoring import score_with_config

SCORED = ["ok", "collapsed"]


class HeldoutRefused(RuntimeError):
    """Raised when a held-out eval breaks rule 7."""


class WspEnv:
    def __init__(self, config, catalog, backend, budget, log, heldout_budget=None):
        self.config = config
        self.catalog = catalog
        self.backend = backend
        self.budget = budget
        self.heldout_budget = heldout_budget
        self.log = log
        self.listeners = []          # fn(row), run after every row is written
        self.listener_errors = []
        self.lock = threading.RLock()
        self.weights = task_weights(config)
        self.guards = guard_tasks(config)
        self.tasks = all_tasks(config)
        self.n_host = space.host_layers(config, catalog)
        self.base = {}
        self.cache = {}              # (hash, split, fidelity) -> scored result
        self.paid = {}               # (hash, split) -> FEE fraction already paid
        self.probe_cache = {}        # probe hash -> probe result
        self.known_fits = {}         # slot key -> fit_r2
        self.history = []            # compact search rows
        self.probes = []             # compact probe rows
        self.heldout_rows = {}       # hash -> held-out row
        self.control_rows = {}       # hash -> control rows
        self.search_closed = False
        self.overshoot_logged = False
        self.did_reset = False
        self.manifest = {}

    # Small helpers.

    def base_scores(self, split):
        if split not in self.base:
            self.base[split] = self.backend.base_scores(split, self.tasks)
        return dict(self.base[split])

    def is_stub(self):
        return self.backend.name == "stub"

    def stamp(self, row, budget):
        row["run_id"] = self.config["run_id"]
        row["method"] = self.config["method"]
        row["backend"] = self.backend.name
        row["stub"] = self.is_stub()
        counters = budget.counters()
        for key in counters:
            row[key] = counters[key]

    def emit(self, row):
        """Write a row, then run every listener. A failing listener never stops the env."""
        self.log.write(row)
        for listener in list(self.listeners):
            try:
                listener(row)
            except Exception as error:
                self.listener_errors.append(type(error).__name__ + ": " + str(error))

    def log_event(self, kind, text, agent=None, round_index=None, extra=None):
        """Write an event row (analyst map, judge verdict, memory op, Exa call, ...)."""
        row = new_event_row(kind, text)
        self.stamp(row, self.budget)
        row["agent"] = agent
        row["round"] = round_index
        if extra is not None:
            row["extra"] = dict(extra)
        self.emit(row)
        return row

    def write_manifest(self, updates):
        """Merge updates into the run manifest and rewrite manifest.json."""
        for key in updates:
            self.manifest[key] = updates[key]
        return self.log.write_json("manifest.json", self.manifest)

    def search_done(self):
        """True once the search budget is spent on either axis, or end_search was called."""
        if self.search_closed:
            return True
        if self.budget.exhausted():
            return True
        cheapest = None
        for name in self.config["fidelities"]:
            cost = fidelity_cost(self.config, name)
            if cheapest is None or cost < cheapest:
                cheapest = cost
        return not self.budget.can_spend(cheapest)

    def done(self):
        return self.search_done()

    def end_search(self, reason=""):
        """Close the search for good. Later steps return budget_exhausted; held-out opens."""
        with self.lock:
            if not self.search_closed:
                self.search_closed = True
                self.log_event("search_end", "search closed: " + reason)

    # Observation.

    def compact(self, row):
        return {
            "program_hash": row["program_hash"],
            "describe": row["describe"],
            "program": row["program"],
            "agent": row["agent"],
            "round": row["round"],
            "status": row["status"],
            "fidelity": row.get("fidelity"),
            "combined": row.get("combined"),
            "scores": dict(row.get("scores") or {}),
            "fit_r2": dict(row.get("fit_r2") or {}),
            "token": row.get("token"),
            "patterns": list(row["extra"].get("patterns", [])),
            "params": row["extra"].get("params"),
            "parent": row["extra"].get("parent"),
        }

    def frontier_rows(self, size=None):
        """The best scored search rows at full fidelity, best first, one per program."""
        if size is None:
            size = self.config["frontier_size"]
        best = {}
        for row in self.history:
            if row["status"] != "ok" or row["fidelity"] != "full":
                continue
            current = best.get(row["program_hash"])
            if current is None or row["combined"] > current["combined"]:
                best[row["program_hash"]] = row
        ranked = list(best.values())
        ranked.sort(key=rank_key)
        return ranked[:size]

    def observation(self):
        parts = []
        for record in space.graftable_parts(self.catalog):
            parts.append(dict(record))
        return {
            "task_type": self.config["task_type"],
            "tasks": dict(self.weights),
            "guards": list(self.guards),
            "base_scores": self.base_scores("search"),
            "host": {"id": self.config["host"], "layers": self.n_host},
            "parts": parts,
            "history": list(self.history),
            "probes": list(self.probes),
            "known_fits": dict(self.known_fits),
            "fit_threshold": self.config["fit_threshold"],
            "fidelities": dict(self.config["fidelities"]),
            "eval_remaining": self.budget.eval_remaining(),
            "compute_remaining_s": self.budget.compute_remaining(),
            "frontier": self.frontier_rows(),
            "search_done": self.search_done(),
        }

    def reset(self, seed=None):
        """Return (observation, info). Logs one reset event; budgets are never refilled."""
        with self.lock:
            if not self.did_reset:
                self.did_reset = True
                self.log_event("reset", "episode start", extra={"seed": seed})
            info = self.budget.counters()
            return self.observation(), info

    # Stepping.

    def step(self, program, agent, hypothesis, citations=None, round_index=None, fidelity="full", extra=None):
        """One step. Returns (observation, reward, terminated, truncated, info).

        reward is the combined score, or None when no eval ran (a cache hit
        returns the cached combined score).
        """
        candidate = {"program": program, "agent": agent, "hypothesis": hypothesis,
                     "citations": citations, "round": round_index, "fidelity": fidelity, "extra": extra}
        row = self.step_many([candidate])[0]
        reward = None
        if row["status"] in ("ok", "collapsed", "cache_hit"):
            reward = row["combined"]
        return self.observation(), reward, False, self.search_done(), row

    def step_batch(self, candidates):
        """Gymnasium-style name for step_many."""
        return self.step_many(candidates)

    def step_many(self, candidates):
        """Run a list of candidates on the search split. Returns their rows in input order.

        Each candidate is a dict: program, agent, hypothesis, and optional
        citations, round, fidelity ("full" by default) and extra (merged into
        the row's extra, for example {"parent": hash}).
        """
        with self.lock:
            jobs = []
            for index in range(len(candidates)):
                jobs.append(make_job(index, candidates[index]))
            rows = self.run_batch(jobs, "search", self.budget, True, True, False)
            self.check_overshoot()
            return rows

    def check_overshoot(self):
        if self.overshoot_logged:
            return
        over = self.budget.overshoot_s()
        if over > 0:
            self.overshoot_logged = True
            self.log_event("compute_overshoot", "compute cap overshot by " + str(round(over, 1)) + " s",
                           extra={"overshoot_s": over})

    def eval_cost(self, program_hash, split, fidelity):
        """FEE for one eval. Fidelity levels are nested, so a promotion pays only the difference."""
        cost = fidelity_cost(self.config, fidelity) - self.paid.get((program_hash, split), 0.0)
        if cost < 0:
            cost = 0.0
        return round(cost, 3)

    def run_batch(self, jobs, split, budget, gate, use_cache, control):
        """The five phases shared by search steps, held-out evals and controls."""
        start_used = budget.compute_used()

        # Phase 1, in order: validate, cache, budget checks.
        can_work = budget.compute_remaining() > 0
        if split == "search" and self.search_closed:
            can_work = False
        planned = 0.0
        first_of_key = {}
        for job in jobs:
            errors = space.validate(job["program_in"], self.catalog, self.config)
            if job["fidelity"] not in self.config["fidelities"]:
                errors.append("unknown fidelity: " + str(job["fidelity"]))
            if len(errors) > 0:
                job["status"] = "invalid"
                job["errors"] = errors
                continue
            job["program"] = ps.normalize(job["program_in"])
            job["hash"] = space.program_hash(job["program"])
            key = (job["hash"], split, job["fidelity"])
            job["key"] = key
            if use_cache and key in self.cache:
                job["status"] = "cache_hit"
                job["cached"] = self.cache[key]
                continue
            if key in first_of_key:
                job["dup_of"] = first_of_key[key]
                continue
            first_of_key[key] = job["index"]
            job["cost"] = self.eval_cost(job["hash"], split, job["fidelity"])
            affordable = budget.eval_remaining() - planned >= job["cost"] - 1e-9
            if not can_work or not affordable:
                job["status"] = "budget_exhausted"
                continue
            planned += job["cost"]
            job["state"] = "compile"

        # Phase 2, in threads: compile, fit gate, heal.
        to_prepare = []
        for job in jobs:
            if job["state"] == "compile":
                to_prepare.append(job)
        self.run_parallel(self.prepare_job, to_prepare, budget, gate)

        # Phase 3, in order: spend tokens.
        for job in jobs:
            if job["state"] == "ready":
                token = budget.spend_eval(job["agent"], job["cost"])
                if token is None:
                    job["status"] = "budget_exhausted"
                    job["state"] = "done"
                    self.release(job)
                else:
                    job["token"] = token
                    job["state"] = "eval"
            job["eval_count"] = budget.eval_used()

        # Phase 4, in threads: eval.
        to_eval = []
        for job in jobs:
            if job["state"] == "eval":
                to_eval.append(job)
        self.run_parallel(self.eval_job, to_eval, budget, split)

        # Phase 5, in order: rows.
        rows = []
        running = start_used
        for job in jobs:
            if job["dup_of"] is not None:
                self.copy_duplicate(job, jobs[job["dup_of"]])
            row = self.finish_row(job, split, budget, control)
            for phase in job["compute"]:
                running += job["compute"][phase]
            row["compute_used_s"] = running
            self.remember(job, row, split)
            self.emit(row)
            rows.append(row)
        return rows

    def run_parallel(self, function, jobs, budget, arg):
        if len(jobs) == 0:
            return
        workers = self.config["workers"]
        if workers <= 1 or len(jobs) == 1:
            for job in jobs:
                function(job, budget, arg)
            return
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(workers, len(jobs))) as pool:
            futures = []
            for job in jobs:
                futures.append(pool.submit(function, job, budget, arg))
            for future in futures:
                future.result()

    def charge(self, job, budget, compute_s, phase_override=None):
        if compute_s is None:
            return
        for phase in sorted(compute_s):
            seconds = float(compute_s[phase])
            name = phase
            if phase_override is not None:
                name = phase_override
            budget.charge_compute(seconds, job["agent"], name)
            job["compute"][name] = job["compute"].get(name, 0.0) + seconds

    def release(self, job):
        if job.get("handle") is not None:
            try:
                self.backend.release(job["handle"])
            except Exception:
                pass
            job["handle"] = None

    def prepare_job(self, job, budget, gate):
        """Compile, apply the fit gate, heal. Runs in a worker thread."""
        program = job["program"]
        try:
            result = self.backend.compile(program, fit_only=False)
        except Exception as error:
            result = {"ok": False, "error": type(error).__name__ + ": " + str(error), "compute_s": {}}
        self.charge(job, budget, result.get("compute_s"))
        job["report"] = list(result.get("graft_report") or [])
        if not result.get("ok"):
            job["status"] = "compile_error"
            job["error"] = result.get("error")
            job["state"] = "done"
            return
        job["handle"] = result.get("handle")
        if gate:
            low = self.fit_rejects(job["report"])
            if len(low) > 0:
                job["status"] = "rejected_fit"
                job["rejected"] = low
                job["state"] = "done"
                self.release(job)
                return
        if ps.heal(program) is not None:
            try:
                healed = self.backend.heal(job["handle"], program)
            except Exception as error:
                healed = {"ok": False, "error": type(error).__name__ + ": " + str(error), "compute_s": {}}
            self.charge(job, budget, healed.get("compute_s"))
            if not healed.get("ok"):
                job["status"] = "compile_error"
                job["error"] = "heal failed: " + str(healed.get("error"))
                job["state"] = "done"
                self.release(job)
                return
            job["handle"] = healed.get("handle")
            job["heal_loss"] = list(healed.get("heal_loss") or [])
        job["state"] = "ready"

    def eval_job(self, job, budget, split):
        """Evaluate one compiled job. Runs in a worker thread."""
        try:
            result = self.backend.evaluate(job["handle"], split, self.tasks, job["token"], job["fidelity"])
        except Exception as error:
            result = {"ok": False, "error": type(error).__name__ + ": " + str(error), "compute_s": {}}
        self.charge(job, budget, result.get("compute_s"))
        self.release(job)
        job["state"] = "done"
        scores = result.get("scores") or {}
        missing = []
        for task in self.tasks:
            if task not in scores:
                missing.append(task)
        if not result.get("ok"):
            job["status"] = "infra_error"
            job["error"] = result.get("error")
        elif len(missing) > 0:
            job["status"] = "infra_error"
            job["error"] = "eval returned no score for " + ", ".join(missing)
        else:
            job["scores"] = dict(scores)
            job["status"] = "scored"

    def fit_rejects(self, report):
        """Keys of grafts whose fit_r2 is below the threshold. A skipped fit (None) is not rejected."""
        low = []
        for entry in report:
            value = entry.get("fit_r2")
            if value is None:
                continue
            if value < self.config["fit_threshold"]:
                low.append(space.graft_key(entry.get("id"), entry.get("host_layer"), entry.get("control")))
        return low

    def copy_duplicate(self, job, first):
        """A repeat inside one batch gets the first copy's result for free."""
        if first["status"] == "scored":
            job["status"] = "cache_hit"
            job["cached"] = {"scores": dict(first["scores"]), "report": list(first["report"]),
                             "heal_loss": list(first["heal_loss"]), "token": first["token"]}
        else:
            job["status"] = first["status"]
            job["rejected"] = list(first["rejected"])
            job["report"] = list(first["report"])
            job["error"] = first["error"]
        job["extra"]["duplicate_of_index"] = first["index"]

    def finish_row(self, job, split, budget, control):
        row = new_eval_row()
        self.stamp(row, budget)
        row["agent"] = job["agent"]
        row["round"] = job["round"]
        row["hypothesis"] = job["hypothesis"]
        row["citations"] = job["citations"]
        row["split"] = split
        row["fidelity"] = job["fidelity"]
        row["token"] = job["token"]
        row["eval_count"] = job["eval_count"]
        row["compute_s"] = dict(job["compute"])
        row["heal_loss"] = list(job["heal_loss"])
        extra = dict(job["extra"])

        if job["program"] is not None:
            row["program"] = job["program"]
            row["program_hash"] = job["hash"]
            row["describe"] = space.describe(job["program"])
            extra["patterns"] = space.program_patterns(job["program"], self.catalog, self.config)
        else:
            row["program"] = job["program_in"]
            row["describe"] = "invalid program"

        report = job["report"]
        scores = job["scores"]
        status = job["status"]
        if status == "cache_hit":
            cached = job["cached"]
            report = cached["report"]
            scores = cached["scores"]
            row["heal_loss"] = list(cached["heal_loss"])
            row["cache_hit"] = True
            extra["cached_token"] = cached["token"]
        fill_fit(row, report)
        extra["params"] = total_params(report)

        if scores is not None:
            value, collapsed = score_with_config(scores, self.base_scores(split), self.config)
            row["scores"] = dict(scores)
            row["combined"] = value
            if status == "scored":
                status = "ok"
                if collapsed:
                    status = "collapsed"
            if control:
                extra["collapsed"] = collapsed
        if control and status in ("ok", "collapsed"):
            status = "control"
        row["status"] = status

        if job["errors"]:
            extra["errors"] = list(job["errors"])
        if job["error"]:
            extra["error"] = job["error"]
        if job["rejected"]:
            extra["rejected"] = list(job["rejected"])
        if job["cost"] is not None:
            extra["eval_cost"] = job["cost"]
        row["extra"] = extra
        return row

    def remember(self, job, row, split):
        """Update caches, known fits and history after a row (search split only)."""
        if split != "search":
            return
        if job["report"] and job["program"] is not None:
            self.learn_fits(job["report"])
            probe_key = space.probe_hash(job["program"])
            if probe_key not in self.probe_cache:
                self.probe_cache[probe_key] = {"fit_r2": dict(row["fit_r2"]), "fit": dict(row["fit"]),
                                               "graft_report": list(job["report"])}
        if row["status"] in SCORED and job["key"] is not None:
            self.cache[job["key"]] = {"scores": dict(row["scores"]), "report": list(job["report"]),
                                      "heal_loss": list(row["heal_loss"]), "token": row["token"]}
            fraction = fidelity_cost(self.config, row["fidelity"])
            paid_key = (job["hash"], split)
            if fraction > self.paid.get(paid_key, 0.0):
                self.paid[paid_key] = fraction
        if row["status"] != "budget_exhausted":
            self.history.append(self.compact(row))

    def learn_fits(self, report):
        for entry in report:
            value = entry.get("fit_r2")
            if value is None or entry.get("donor") is None or entry.get("host_layer") is None:
                continue
            self.known_fits[space.slot_key(entry["donor"], entry.get("donor_layer"), entry["host_layer"])] = value

    def fit_of_slot(self, slot):
        """The known fit_r2 of a slot key, or None if it has not been compiled yet."""
        return self.known_fits.get(slot)

    # Probe.

    def probe(self, program, agent, round_index=None, hypothesis=""):
        """Fit-only compile: the program with heal set to None. Charges compute, never a token.

        Returns {"status", "fit_r2", "fit", "graft_report", "compute_s", "cache_hit", "errors"}.
        status is ok, cache_hit, invalid, compile_error or budget_exhausted.
        Cached by the program's hash without heal.
        """
        with self.lock:
            row = new_probe_row()
            row["agent"] = agent
            row["round"] = round_index
            row["hypothesis"] = hypothesis
            result = {"status": None, "fit_r2": {}, "fit": {}, "graft_report": [], "compute_s": 0.0,
                      "cache_hit": False, "errors": []}
            errors = space.validate(program, self.catalog, self.config)
            if len(errors) > 0:
                result["status"] = "invalid"
                result["errors"] = errors
                row["program"] = program
                row["extra"] = {"errors": errors}
                return self.finish_probe(row, result)
            full = ps.normalize(program)
            bare = ps.without_heal(full)
            key = space.probe_hash(full)
            row["program"] = bare
            row["program_hash"] = key
            row["describe"] = space.describe(bare)
            row["extra"] = {"program_hash_with_heal": space.program_hash(full)}
            if key in self.probe_cache:
                cached = self.probe_cache[key]
                result["status"] = "cache_hit"
                result["cache_hit"] = True
                result["graft_report"] = list(cached["graft_report"])
                fill_fit(row, cached["graft_report"])
                row["cache_hit"] = True
                return self.finish_probe(row, result)
            if self.search_closed or self.budget.compute_remaining() <= 0:
                result["status"] = "budget_exhausted"
                return self.finish_probe(row, result)
            try:
                compiled = self.backend.compile(bare, fit_only=True)
            except Exception as error:
                compiled = {"ok": False, "error": type(error).__name__ + ": " + str(error), "compute_s": {}}
            job = {"agent": agent, "compute": {}}
            total = 0.0
            for phase in sorted(compiled.get("compute_s") or {}):
                total += float(compiled["compute_s"][phase])
            self.charge(job, self.budget, {"probe": total})
            row["compute_s"] = dict(job["compute"])
            result["compute_s"] = total
            if compiled.get("handle") is not None:
                try:
                    self.backend.release(compiled["handle"])
                except Exception:
                    pass
            if not compiled.get("ok"):
                result["status"] = "compile_error"
                row["extra"]["error"] = compiled.get("error")
                return self.finish_probe(row, result)
            report = list(compiled.get("graft_report") or [])
            self.learn_fits(report)
            fill_fit(row, report)
            self.probe_cache[key] = {"fit_r2": dict(row["fit_r2"]), "fit": dict(row["fit"]), "graft_report": report}
            result["status"] = "ok"
            result["graft_report"] = report
            return self.finish_probe(row, result)

    def finish_probe(self, row, result):
        self.stamp(row, self.budget)
        row["status"] = result["status"]
        if row["program_hash"] is not None:
            row["extra"]["patterns"] = space.program_patterns(row["program"], self.catalog, self.config)
        result["fit_r2"] = dict(row["fit_r2"])
        result["fit"] = dict(row["fit"])
        if result["status"] in ("ok", "cache_hit"):
            self.probes.append({"program_hash": row["program_hash"], "describe": row["describe"],
                                "program": row["program"], "agent": row["agent"], "round": row["round"],
                                "status": row["status"], "fit_r2": dict(row["fit_r2"])})
        self.emit(row)
        self.check_overshoot()
        return result

    # Held-out and controls (rule 7).

    def require_heldout_open(self, program_hash, table):
        if self.heldout_budget is None:
            raise HeldoutRefused("this env has no held-out budget")
        if not self.search_done():
            raise HeldoutRefused("held-out runs only after the search budget is spent (or end_search)")
        if program_hash in table:
            raise HeldoutRefused("program " + program_hash + " was already scored once on held-out")
        if len(table) >= self.config["heldout_top"]:
            raise HeldoutRefused("at most " + str(self.config["heldout_top"]) + " programs per run go to held-out")

    def heldout_eval(self, program, agent, hypothesis=""):
        """Score one final program once on the held-out split, from the held-out budget."""
        with self.lock:
            program_hash = space.program_hash(program)
            self.require_heldout_open(program_hash, self.heldout_rows)
            self.search_closed = True
            job = make_job(0, {"program": program, "agent": agent, "hypothesis": hypothesis})
            row = self.run_batch([job], "heldout", self.heldout_budget, False, False, False)[0]
            self.heldout_rows[program_hash] = row
            return row

    def run_controls(self, program, agent):
        """Score a final program's two controls once on held-out: random part, then no graft.

        Rows have status "control" and extra {"control": "random" or "no_graft",
        "control_of": hash, "collapsed": bool}.
        """
        with self.lock:
            program_hash = space.program_hash(program)
            self.require_heldout_open(program_hash, self.control_rows)
            self.search_closed = True
            rng = random.Random(self.config["seed"] * 1000003 + int(program_hash[:8], 16))
            random_part = space.random_glue_control(program, rng, self.catalog)
            no_graft = space.no_graft_control(program)
            jobs = [
                make_job(0, {"program": random_part, "agent": agent, "hypothesis": "random-part control",
                             "extra": {"control": "random", "control_of": program_hash}}),
                make_job(1, {"program": no_graft, "agent": agent, "hypothesis": "no-graft heal control",
                             "extra": {"control": "no_graft", "control_of": program_hash}}),
            ]
            rows = self.run_batch(jobs, "heldout", self.heldout_budget, False, False, True)
            self.control_rows[program_hash] = rows
            return rows


def make_job(index, candidate):
    """The working record for one candidate inside a batch."""
    extra = candidate.get("extra")
    if extra is None:
        extra = {}
    citations = candidate.get("citations")
    if citations is None:
        citations = []
    fidelity = candidate.get("fidelity")
    if fidelity is None:
        fidelity = "full"
    return {
        "index": index,
        "program_in": candidate.get("program"),
        "agent": candidate.get("agent"),
        "hypothesis": candidate.get("hypothesis", ""),
        "citations": list(citations),
        "round": candidate.get("round"),
        "fidelity": fidelity,
        "extra": dict(extra),
        "program": None,
        "hash": None,
        "key": None,
        "state": "done",
        "status": None,
        "errors": [],
        "error": None,
        "cost": None,
        "handle": None,
        "token": None,
        "eval_count": 0.0,
        "report": [],
        "heal_loss": [],
        "rejected": [],
        "scores": None,
        "cached": None,
        "dup_of": None,
        "compute": {},
    }


def fill_fit(row, report):
    """Copy fit_r2, fit and the per-graft report into a row."""
    fit_r2 = {}
    fit = {}
    for entry in report:
        key = space.graft_key(entry.get("id"), entry.get("host_layer"), entry.get("control"))
        fit_r2[key] = entry.get("fit_r2")
        fit[key] = entry.get("fit")
    row["fit_r2"] = fit_r2
    row["fit"] = fit
    row["graft_report"] = list(report)


def total_params(report):
    """Part plus glue parameters over the grafts (None when there is no report)."""
    if len(report) == 0:
        return None
    total = 0
    for entry in report:
        total += int(entry.get("part_params") or 0) + int(entry.get("glue_params") or 0)
    return total


def rank_key(row):
    """Sort key: higher combined first, then the earlier token, then the hash."""
    token = row.get("token")
    if token is None:
        token = 10 ** 9
    return (-row["combined"], token, row["program_hash"])
