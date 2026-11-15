# Local side: connects to the deployed engine for one donor, keeps per-model state,
# applies the light gate, and writes catalog records + .pt index files.
import hashlib
import json
import os
import time

import modal

import common
import evidence

MAX_AGENT_SKILLS = 8
# built-in skills are standard benchmarks; a lower bar is enough to say "not at chance"
# (math scores digit tokens only, which share many digits across items, so its margin is naturally small)
BUILTIN_MIN_MARGIN = 0.25

# Light gate ("screened" tier)
GATE_MIN_TARGET_DROP = 0.15
GATE_N_RANDOM = 5
# "confirmed" tier (optional, math only): relative greedy-accuracy drop on 40 GSM8K problems
CONFIRM_MIN_DROP = 0.20
# items per skill half for the router pick rates of an expert part (the count skill_maps uses)
ROUTING_ITEMS = 64


class Miner:
    def __init__(self, short):
        donors = common.load_donors()
        if short not in donors:
            raise SystemExit("unknown model '" + short + "'. Known: " + ", ".join(sorted(donors.keys())))
        self.short = short
        self.row = donors[short]
        self.hf_id = self.row["hf_id"]
        self.params_b = float(self.row["params_b"])
        self.gpu = common.pick_gpu(self.params_b, short)
        self.state = common.read_json_safe(common.state_path(short), {})
        if not self.state.get("revision"):
            self.state["revision"] = common.pin_revision(self.hf_id)
            self.state["model"] = self.hf_id
            self.save()
        self.revision = self.state["revision"]
        cls = modal.Cls.from_name(common.APP_NAME, self.gpu)
        self.engine = cls(model_key=short, hf_id=self.hf_id, revision=self.revision)
        self.modal_seconds = 0.0

    # ---------- state ----------

    def save(self):
        common.write_json_atomic(common.state_path(self.short), self.state)

    def call(self, method, *args):
        start = time.time()
        result = getattr(self.engine, method).remote(*args)
        self.modal_seconds += time.time() - start
        return result

    # ---------- skills ----------

    def agent_skills(self):
        return self.state.get("agent_skills", {})

    def payload(self, include_skipped=False):
        skills = {}
        for name in common.BUILTIN_TARGETS + common.CONTROL_ONLY:
            if include_skipped or name not in self.state.get("skipped", []):
                skills[name] = {"builtin": True}
        for name in self.agent_skills():
            spec = self.agent_skills()[name]
            with open(os.path.join(common.ROOT, spec["path"])) as f:
                data = json.load(f)
            skills[name] = {"hash": data["hash"], "items": data["items"], "control": data["control"]}
        return {"skills": skills}

    def structure(self):
        if "structure" not in self.state:
            self.state["structure"] = self.call("info_summary")
            self.save()
        return self.state["structure"]

    def base(self):
        if "base" not in self.state:
            res = self.call("base", self.payload(include_skipped=True))
            skipped = []
            for name in common.BUILTIN_TARGETS:
                margin = res["margin_vs_shuffled"].get(name)
                if margin is None or margin < BUILTIN_MIN_MARGIN:
                    skipped.append(name)  # near chance: the model can't do this skill
            self.state["base"] = res
            self.state["skipped"] = skipped
            self.save()
            common.log_event(self.short, "base", {"gpu": self.gpu}, res)
        return self.state["base"]

    def target_skills(self):
        names = []
        for name in common.BUILTIN_TARGETS:
            if name not in self.state.get("skipped", []):
                names.append(name)
        for name in self.agent_skills():
            names.append(name)
        return names

    # ---------- ablation + gate ----------

    def ablate(self, part):
        res = self.call("ablate", self.payload(), part)
        common.log_event(self.short, "ablate", {"part": part_brief(part)}, {"drops": res["drops"], "seconds": res["seconds"]})
        return res

    def screen(self, part, target):
        key = part_key(part) + "|" + target
        done = self.state.get("screens", {})
        if key in done:
            return done[key]
        abl = self.call("ablate", self.payload(), part)
        rnd = self.call("random_drops", self.payload(), part, target, GATE_N_RANDOM, len(done) + 1)
        gate = light_gate(abl["drops"], rnd["drops"], target, self.target_skills())
        result = {"part": part, "target": target, "drops": abl["drops"], "random_drops": rnd["drops"],
                  "gate": gate, "seconds": round(abl["seconds"] + rnd["seconds"], 1)}
        done[key] = result
        self.state["screens"] = done
        self.save()
        common.log_event(self.short, "screen", {"part": part_brief(part), "target": target},
                         {"passed": gate["passed"], "reasons": gate["reasons"], "seconds": result["seconds"]})
        return result

    def confirm(self, part):
        res = self.call("confirm_math", part)
        res["passed"] = res["drop"] is not None and res["drop"] >= CONFIRM_MIN_DROP
        common.log_event(self.short, "confirm_math", {"part": part_brief(part)}, res)
        return res

    # ---------- evidence for catalog rows ----------

    def engine_builtin(self):
        # The built-in skill sets as the deployed engine holds them (its image keeps the file from
        # deploy time). None when the engine cannot be asked, for example a deployment older than
        # builtin_info. Asked once per process: a redeploy between runs must be seen.
        if "_engine_builtin" not in self.__dict__:
            try:
                self._engine_builtin = self.call("builtin_info")
            except Exception as e:
                self._engine_builtin = None
                common.log_event(self.short, "builtin_info_failed", {},
                                 {"error": type(e).__name__ + ": " + str(e)[:200]})
        return self._engine_builtin

    def skill_sources(self):
        # (lookup, builtin_basis) for evidence.selection_data
        described = self.engine_builtin()
        if described:
            builtin = evidence.builtin_sources(described, evidence.local_builders())
            basis = "engine"
        else:
            builtin = evidence.local_builtin()
            basis = "local file"
        return evidence.make_lookup(self.agent_skills(), builtin), basis

    def expert_routing(self, part, target, extra_skills=None):
        # Router pick rates of an expert part on the target skill's text and on general text.
        # Returns (routing, note) and never raises: the row must be written even when the engine is
        # a deployment without expert_pick_rates. extra_skills: payload specs for skills that this
        # state file does not hold (enrich_rows.py passes the frozen test of an agent skill).
        names = [target, evidence.ROUTING_REFERENCE]
        try:
            known = self.payload(include_skipped=True)["skills"]
            known.update(extra_skills or {})
            skills = {}
            for name in names:
                if name in known:
                    skills[name] = known[name]
            res = self.call("expert_pick_rates", {"skills": skills}, names, [int(part["layer"])], ROUTING_ITEMS)
            routing, note = evidence.routing_fields(part["indices"], part["layer"], target, res)
        except Exception as e:
            routing = None
            note = ("no routing: engine call expert_pick_rates failed (a deployment older than this method?): "
                    + type(e).__name__ + ": " + str(e)[:160])
        common.log_event(self.short, "expert_routing", {"part": part_brief(part), "target": target},
                         {"ok": routing is not None, "note": note})
        return routing, note

    # ---------- catalog ----------

    def catalog_rows(self):
        path = common.catalog_path(self.short)
        rows = []
        if os.path.exists(path):
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        rows.append(json.loads(line))
        return rows

    def write_part(self, part, label, target, screen_result, evidence_used, weight_rationale, tier, confirm=None):
        # Engine calls for the evidence fields happen here, before the lock: a GPU call can take
        # minutes on a cold container, and the lock must stay short for the other lanes.
        sources = self.skill_sources()
        routing = None
        if part["kind"] == "expert":
            key = part_key(part)
            known = False
            try:
                for rec in self.catalog_rows():
                    if rec.get("part_key") == key and rec.get("label") == label:
                        known = True  # already catalogued: the locked step returns that row as it is
            except ValueError:
                pass  # read without the lock: another lane may be half-way through a line
            if not known:
                routing = self.expert_routing(part, target)
        # parallel agents on the same model share catalog/<model>.jsonl: hold a file lock while
        # choosing the id and appending, so two agents never write the same id
        import fcntl
        lock_path = common.catalog_path(self.short) + ".lock"
        os.makedirs(os.path.dirname(lock_path), exist_ok=True)
        with open(lock_path, "w") as lock_file:
            fcntl.flock(lock_file, fcntl.LOCK_EX)
            try:
                return self._write_part_locked(part, label, target, screen_result, evidence_used,
                                               weight_rationale, tier, confirm, sources, routing)
            finally:
                fcntl.flock(lock_file, fcntl.LOCK_UN)

    def _write_part_locked(self, part, label, target, screen_result, evidence_used, weight_rationale, tier,
                           confirm=None, sources=None, routing=None):
        # sources: (lookup, builtin_basis) from skill_sources. routing: (routing, note) from
        # expert_routing, or None when it was not asked for.
        import torch

        path = common.catalog_path(self.short)
        existing = self.catalog_rows()
        key = part_key(part)
        for rec in existing:
            if rec.get("part_key") == key and rec.get("label") == label:
                return rec  # already catalogued
        clean = common.clean_name(label)
        short_kind = {"mlp_cluster": "mlp", "attn_heads": "heads", "expert": "expert", "block": "block"}[part["kind"]]
        if part["kind"] == "block":
            where = "L" + str(part["layers"][0]) + "-" + str(part["layers"][-1])
        else:
            where = "L" + str(part["layer"]) + "_k" + str(len(part["indices"]))
        base_id = short_kind + "_" + clean + "_" + self.short + "_" + where
        n = 1
        ids = set()
        for rec in existing:
            ids.add(rec["id"])
        while base_id + "_" + str(n).zfill(3) in ids:
            n += 1
        part_id = base_id + "_" + str(n).zfill(3)

        os.makedirs(common.parts_dir(self.short), exist_ok=True)
        pt_path = os.path.join(common.parts_dir(self.short), part_id + ".pt")
        if part["kind"] == "block":
            index = {"kind": "block", "model": self.hf_id, "layer": int(part["layers"][0]),
                     "layers": torch.tensor(part["layers"], dtype=torch.long),
                     "indices": torch.tensor([], dtype=torch.long)}
        else:
            index = {"kind": part["kind"], "model": self.hf_id, "layer": int(part["layer"]),
                     "indices": torch.tensor(sorted(part["indices"]), dtype=torch.long)}
        torch.save(index, pt_path)

        skill_info = {"source": "builtin"}
        if target in self.agent_skills():
            spec = self.agent_skills()[target]
            skill_info = {"source": "agent", "path": spec["path"], "hash": spec["hash"],
                          "description": spec["description"]}
        # z-scores against the random parts and the largest side effect, next to the raw numbers
        signature = {"effect": screen_result["drops"], "random_drops": screen_result["random_drops"],
                     "gate": screen_result["gate"], "confirm": confirm}
        signature.update(evidence.signature_fields(screen_result["drops"], screen_result["random_drops"], target))
        if sources is None:
            sources = (evidence.make_lookup(self.agent_skills(), evidence.local_builtin()), "local file")
        lookup, basis = sources
        record = {
            "id": part_id,
            "type": part["kind"],
            "label": label,
            "tier": tier,
            "trained_on": "ablation screen on skill '" + target + "'",
            "model": self.hf_id,
            "revision": self.revision,
            "d_model": int(self.structure()["d_model"]),
            "units": os.path.relpath(pt_path, common.ROOT),
            "num_units": len(part["layers"]) if part["kind"] == "block" else len(part["indices"]),
            "skill": dict(skill_info, name=target),
            "signature": signature,
            "evidence_used": evidence_used,
            "weight_rationale": weight_rationale,
            "part_key": key,
            "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        if part["kind"] == "block":
            record["layers"] = list(part["layers"])
        else:
            record["layer"] = int(part["layer"])
        record["selection_data"] = evidence.selection_data(target, screen_result["drops"], lookup,
                                                           evidence.load_healdata(), basis)
        if part["kind"] == "expert":
            if routing is None:
                routing = (None, "no routing: not measured when the row was written")
            record["routing"] = routing[0]
            if routing[1]:
                record["routing_note"] = routing[1]
        common.append_jsonl(path, record)
        return record


def part_key(part):
    if part["kind"] == "block":
        return "block:" + ",".join(str(x) for x in part["layers"])
    return part["kind"] + ":" + str(part["layer"]) + ":" + hashlib.sha1(
        ",".join(str(x) for x in sorted(part["indices"])).encode()).hexdigest()[:12]


def part_brief(part):
    if part["kind"] == "block":
        return {"kind": "block", "layers": part["layers"]}
    return {"kind": part["kind"], "layer": part["layer"], "n": len(part["indices"]),
            "first": sorted(part["indices"])[:8]}


def light_gate(drops, random_drops, target, all_targets):
    # drops keys look like "math|search", "math|heldout", "myskill__control|search"
    reasons = []
    t_search = drops.get(target + "|search")
    t_held = drops.get(target + "|heldout")
    if t_search is None or t_search < GATE_MIN_TARGET_DROP:
        reasons.append("target drop " + str(t_search) + " < " + str(GATE_MIN_TARGET_DROP))
    if t_held is None or t_held < GATE_MIN_TARGET_DROP:
        reasons.append("held-out check failed")  # no number: the held-out half stays unseen
    limit = 0.05
    if t_search is not None:
        limit = max(0.05, t_search / 3.0)
    for key in drops:
        name, split = key.split("|")
        if split != "search" or name == target:
            continue
        if drops[key] is None:
            continue
        if name == target + "__control":
            # the format-matched control must not move in EITHER direction: a big rise means the part
            # just shifts which tokens get predicted, rather than carrying the skill
            if abs(drops[key]) > limit:
                reasons.append("matched control changed by " + str(drops[key]) + " (limit +-" + str(round(limit, 3)) + ")")
        elif not name.endswith("__control"):
            # other skills must not drop much (rises are allowed: relative changes on low-probability
            # skills are noisy upward)
            if drops[key] > limit:
                reasons.append("other skill '" + name + "' dropped " + str(drops[key]) + " > " + str(round(limit, 3)))
    if t_search is not None:
        for r in random_drops:
            if r is not None and r >= t_search:
                reasons.append("a random same-size part dropped the target as much (" + str(r) + ")")
                break
    if len(random_drops) < GATE_N_RANDOM:
        reasons.append("only " + str(len(random_drops)) + " random parts could be drawn")
    return {"passed": len(reasons) == 0, "reasons": reasons, "limit_other": round(limit, 4), "gate_version": 2}


def agent_view(screen_result):
    # what the agent may see: search-split drops, pass/fail, reasons (held-out numbers removed)
    drops = {}
    for key in screen_result["drops"]:
        if key.endswith("|search"):
            drops[key.split("|")[0]] = screen_result["drops"][key]
    return {"passed": screen_result["gate"]["passed"], "reasons": screen_result["gate"]["reasons"],
            "drops": drops, "random_drops_target": screen_result["random_drops"]}
