# Checks mine/enrich_rows.py on rows that mine/sweep.py writes for pool skills (skill.source "pool").
# A sweep row names its pool sets by hash. The backfill must keep what the row stores when the pool
# file is built again, and when another lane has an agent skill with the name of a pool skill. With
# --routing it must send the pool set to the engine, the way the sweep does.
#   .venv/bin/python mine/tests/test_enrich_pool.py
# Everything runs in a temp folder with a stand-in for the Modal class. No engine call.
import copy
import json
import os
import shutil
import sys
import tempfile
import time
import types

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import common  # noqa: E402
import client  # noqa: E402
import enrich_rows  # noqa: E402
import evidence  # noqa: E402
import skills  # noqa: E402

BUILTIN = {"math": {"source": "builtin", "description": "Orca-Math word problems (approved pool)",
                    "datasets": ["microsoft/orca-math-word-problems-200k"], "items_sha": "aaaa"},
           "general_text": {"source": "builtin", "description": "WikiText paragraph continuation (control only)",
                            "datasets": ["Salesforce/wikitext"], "items_sha": "bbbb"}}
ITEMS = {"search": [{"context": "a", "continuation": " b", "digits_only": False}],
         "heldout": [{"context": "c", "continuation": " d", "digits_only": False}]}
POOL_1 = {"law": {"description": "law questions, correct answer text", "dataset": "cais/mmlu professional_law",
                  "hash": "law-hash-1", "items": ITEMS},
          "physics": {"description": "physics questions", "dataset": "cais/mmlu college_physics",
                      "hash": "physics-hash-1", "items": ITEMS}}
# the pool file after it was built again from other data
POOL_2 = {"law": {"description": "law questions from another set", "dataset": "some-org/approved-law train",
                  "hash": "law-hash-2", "items": ITEMS},
          "physics": {"description": "physics from another set", "dataset": "some-org/approved-physics train",
                      "hash": "physics-hash-2", "items": ITEMS}}
OLD_FINAL = (evidence.FINAL + "the target skill is neither a built-in set of the engine nor an agent skill with a"
             " frozen test")


def effect():
    out = {}
    for name, drop in [("law", 0.5), ("general_text", 0.0), ("math", 0.01), ("physics", 0.02)]:
        out[name + "|search"] = drop
        out[name + "|heldout"] = drop
    return out


def sweep_row(ident, kind, hd, pool_path):
    # a row as SweepMiner.write_part writes it: the skill entry of sweep.load_skill_specs, the field
    # "sweep", and the selection_data that evidence.selection_data gives on the day of the write
    rec = {"id": ident, "type": kind, "label": "label of " + ident, "tier": "screened", "model": "org/toy",
           "revision": "rev1", "d_model": 8, "units": "parts/toy/" + ident + ".pt", "num_units": 2,
           "skill": {"source": "pool", "path": os.path.relpath(pool_path, common.ROOT), "hash": "law-hash-1",
                     "dataset": POOL_1["law"]["dataset"], "description": POOL_1["law"]["description"], "name": "law"},
           "signature": {"effect": effect(), "random_drops": [0.1, 0.2, 0.3],
                         "gate": {"passed": True, "reasons": [], "limit_other": 0.1667, "gate_version": 2},
                         "confirm": None},
           "evidence_used": ["attribution", "ablation"], "weight_rationale": "why", "part_key": kind + ":3:" + ident,
           "created": "2026-11-14T22:00:00", "layer": 3,
           "sweep": {"rank_method": "attribution", "guards": ["math", "physics"], "z_search": 3.0}}
    rec["selection_data"] = evidence.selection_data("law", effect(), evidence.make_lookup({}, BUILTIN), hd, "engine")
    return rec


def write(path, rows):
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def by_id(path):
    with open(path) as f:
        return {r["id"]: r for r in (json.loads(line) for line in f if line.strip())}


def use_pool(root, name, data):
    # evidence caches a file by its path, so each version of the pool file gets its own path
    path = os.path.join(root, name)
    with open(path, "w") as f:
        json.dump(data, f)
    evidence.POOL_SKILLS = path
    return path


def key_facts(sd):
    return {k: sd.get(k) for k in ["source", "hash", "description", "dataset", "datasets", "reported_benchmarks"]}


class FakeRemote:
    def __init__(self, engine, name):
        self.engine = engine
        self.name = name

    def remote(self, *args):
        self.engine.calls.append((self.name, args))
        if FakeCls.during is not None:
            during, FakeCls.during = FakeCls.during, None
            during()
        if self.name != "expert_pick_rates":
            raise RuntimeError("the backfill may call expert_pick_rates only, not " + self.name)
        payload, names, layers, n_items = args
        rates = {}
        missing = []
        for name in names:
            for split in ["search", "heldout"]:
                if name not in payload["skills"]:
                    missing.append(name + "|" + split)
                    continue
                picks = [0.25, 0.25, 0.5, 1.0] if name == "general_text" else [0.0, 0.5, 0.25, 1.25]
                rates[name + "|" + split] = {"n_items": 2, "layers": {str(l): {"rates": picks, "tokens": 40, "top_k": 2}
                                                                      for l in layers}}
        return {"rates": rates, "counted_layers": list(layers), "unsupported": {}, "missing": missing}


class FakeEngine:
    def __init__(self):
        self.calls = []

    def __getattr__(self, name):
        return FakeRemote(self, name)


class FakeCls:
    built = []
    during = None  # runs once, inside the next engine call

    @staticmethod
    def from_name(app, gpu):
        def make(**kw):
            FakeCls.built.append(FakeEngine())
            return FakeCls.built[-1]
        return make


def calls():
    return [call for engine in FakeCls.built for call in engine.calls]


def selection(root, path, hd):
    pool_1 = use_pool(root, "pool_1.json", POOL_1)
    row = sweep_row("mlp_law", "mlp_cluster", hd, pool_1)
    written = copy.deepcopy(row["selection_data"])
    # what the write stores: the pool sets by hash, for the target and for the guard
    assert key_facts(written) == {"source": "pool", "hash": "law-hash-1", "description": POOL_1["law"]["description"],
                                  "dataset": POOL_1["law"]["dataset"], "datasets": ["cais/mmlu"],
                                  "reported_benchmarks": ["cais/mmlu"]}, written
    assert written["off_target"] == ["general_text", "math", "physics"], written
    assert written["off_target_data"]["physics"] == {"source": "pool", "description": "physics questions",
                                                     "datasets": ["cais/mmlu"], "dataset": POOL_1["physics"]["dataset"],
                                                     "hash": "physics-hash-1"}, written
    assert written["off_target_data"]["math"] == dict(BUILTIN["math"]), written
    no_data = sweep_row("mlp_law_no_data", "mlp_cluster", hd, pool_1)
    no_data["selection_data"] = None
    no_data["selection_note"] = "selection_data could not be built: OSError"
    write(path, [row, no_data])

    def run(stamp):
        status = enrich_rows.enrich_model("toy", force=True, versions=[], healdata=hd, stamp=stamp)
        assert status == "enriched 2 of 2 rows", status
        return by_id(path)

    # 1. nothing changed since the write: the same data, and the pool file still has the set
    got = run("2026-11-15T00:00:00")
    sd = got["mlp_law"]["selection_data"]
    assert key_facts(sd) == key_facts(written) and sd["off_target_data"] == written["off_target_data"], sd
    assert sd["pool_hash_matches"] is True and sd["builtin_basis"] == "engine" and "hash_unverified" not in sd, sd
    # a row that was written without selection_data: the target comes from the row's skill entry. Which
    # version of the guard's pool set was scored is not recorded, so its hash is marked unverified.
    sd = got["mlp_law_no_data"]["selection_data"]
    assert key_facts(sd) == key_facts(written) and "selection_note" not in got["mlp_law_no_data"], sd
    assert sd["off_target_data"]["physics"] == dict(written["off_target_data"]["physics"], hash_unverified=True), sd

    # 2. the pool file is built again from other data: the row keeps the sets it was selected on
    use_pool(root, "pool_2.json", POOL_2)
    got = run("2026-11-15T01:00:00")
    for ident in ["mlp_law", "mlp_law_no_data"]:
        sd = got[ident]["selection_data"]
        assert key_facts(sd) == key_facts(written), (ident, sd)
        assert sd["pool_hash_matches"] is False and got[ident]["skill"]["hash"] == "law-hash-1", (ident, sd)
    assert got["mlp_law"]["selection_data"]["off_target_data"] == written["off_target_data"]
    # the guard of the row without stored data was named in step 1, with the mark, and stays so
    sd = got["mlp_law_no_data"]["selection_data"]
    assert sd["off_target_data"]["physics"] == dict(written["off_target_data"]["physics"], hash_unverified=True), sd

    # 3. a lane of the model has an agent skill with the name of the pool skill
    use_pool(root, "pool_1.json", POOL_1)
    os.makedirs(os.path.join(root, "logs", "toy"), exist_ok=True)
    with open(os.path.join(root, "logs", "toy", "state.lane1.json"), "w") as f:
        json.dump({"revision": "rev1", "agent_skills": {
            "law": {"path": "mine/skills/toy/law.json", "hash": "agent-law-hash", "description": "agent law"},
            "physics": {"path": "mine/skills/toy/physics.json", "hash": "agent-physics-hash", "description": "x"}}}, f)
    got = run("2026-11-15T02:00:00")
    for ident in ["mlp_law", "mlp_law_no_data"]:
        sd = got[ident]["selection_data"]
        assert key_facts(sd) == key_facts(written), (ident, sd)
        assert sd["off_target_data"]["physics"]["source"] == "pool", (ident, sd)
        assert sd["off_target_data"]["physics"]["hash"] == "physics-hash-1", (ident, sd)
    # a second run gives the same rows
    again = run("2026-11-15T02:00:00")
    assert again == got

    # an agent cannot define a skill with the name of a pool skill
    miner = types.SimpleNamespace(agent_skills=lambda: {}, state={})
    for name in ["law", "Physics"]:
        res = skills.define(miner, None, name, "d", items=[])
        assert "pool skill" in res.get("error", ""), res
    os.remove(os.path.join(root, "logs", "toy", "state.lane1.json"))


def routing(root, path, hd):
    pool_1 = use_pool(root, "pool_1.json", POOL_1)
    retry = sweep_row("expert_retry", "expert", hd, pool_1)        # the engine call failed at the write
    retry["routing"] = None
    retry["routing_note"] = "no routing: engine call expert_pick_rates failed: TimeoutError"
    marked = sweep_row("expert_marked", "expert", hd, pool_1)      # an earlier backfill called it final
    marked["routing"] = None
    marked["routing_note"] = OLD_FINAL
    other_hash = sweep_row("expert_other_hash", "expert", hd, pool_1)
    other_hash["routing"] = None
    other_hash["routing_note"] = "no routing: not measured when the row was written"
    other_hash["skill"]["hash"] = "law-hash-0"
    no_set = sweep_row("expert_no_set", "expert", hd, pool_1)
    no_set["routing"] = None
    no_set["skill"]["name"] = "astronomy"
    no_set["signature"]["effect"]["astronomy|search"] = 0.5
    no_set["signature"]["effect"]["astronomy|heldout"] = 0.5
    final = sweep_row("expert_final", "expert", hd, pool_1)        # a final note for another reason stays
    final["routing"] = None
    final["routing_note"] = evidence.FINAL + "an earlier answer"
    rows = [retry, marked, other_hash, no_set, final]
    write(path, rows)
    os.makedirs(os.path.join(root, "parts", "toy"), exist_ok=True)
    for r in rows:
        torch.save({"kind": "expert", "model": "org/toy", "layer": 3, "indices": torch.tensor([1, 2])},
                   os.path.join(root, r["units"]))
    registry = os.path.join(root, "donors.csv")
    with open(registry, "w") as f:
        f.write("short_name,hf_id,params_b\ntoy,org/toy,1\n")
    real = (common.REGISTRY_CSV, client.modal)
    common.REGISTRY_CSV = registry
    client.modal = types.SimpleNamespace(Cls=FakeCls)
    try:
        plan, settled = enrich_rows.routing_plan("toy", rows, [])
        assert [c[0]["id"] for c in plan] == ["expert_retry", "expert_marked"], [c[0]["id"] for c in plan]
        assert sorted(settled) == ["expert_no_set", "expert_other_hash"], settled
        # The layer list of a state file settles a row only when the state is of the row's revision:
        # module_list experts in layer 3 of another revision say nothing about these rows.
        detail = [{"layer": l} for l in range(3)] + [
            {"layer": 3, "experts": {"path": "mlp.experts", "n_experts": 4, "style": "module_list"}}]
        other_state = {"revision": "rev0", "structure": {"layers_detail": detail}}
        plan, settled = enrich_rows.routing_plan("toy", rows, [other_state])
        assert [c[0]["id"] for c in plan] == ["expert_retry", "expert_marked"], settled
        plan, settled = enrich_rows.routing_plan("toy", rows, [dict(other_state, revision="rev1")])
        assert plan == [] and "module_list" in settled["expert_retry"][1], settled

        # A mining run appends a row while the engine calls of the backfill run. The backfill reads the
        # catalog again under the lock, so that row is in the file afterwards.
        late = sweep_row("mlp_late", "mlp_cluster", hd, pool_1)

        def a_run_appends_a_row():
            with common.catalog_lock("toy"):
                common.append_jsonl(path, late)
        FakeCls.during = a_run_appends_a_row
        status = enrich_rows.enrich_model("toy", routing=True, force=True, versions=[], healdata=hd,
                                          stamp="2026-11-15T03:00:00")
        assert status == "enriched 6 of 6 rows", status
        got = by_id(path)
        assert list(got) == [r["id"] for r in rows] + ["mlp_late"] and got["mlp_late"]["enriched_at"], list(got)
        assert [c[0] for c in calls()] == ["expert_pick_rates", "expert_pick_rates"], calls()
        payload, names, layers, n_items = calls()[0][1]
        # the pool set goes inline, with the hash and the shape that mine/sweep.py sends
        assert names == ["law", "general_text"] and layers == [3] and n_items == client.ROUTING_ITEMS
        assert payload["skills"]["law"] == {"hash": "pool-law-hash-1", "items": ITEMS, "control": None}, payload
        assert payload["skills"]["general_text"] == {"builtin": True}
        for ident in ["expert_retry", "expert_marked"]:
            assert got[ident]["routing"]["skill"] == "law" and got[ident]["routing"]["revision"] == "rev1", got[ident]
            assert "routing_note" not in got[ident], got[ident]
            # ln(0.5 / 0.25) = 0.6931
            assert got[ident]["routing"]["experts"]["1"]["search"] == {"skill_rate": 0.5, "general_rate": 0.25,
                                                                        "log_ratio": 0.6931}, got[ident]["routing"]
        note = got["expert_other_hash"]["routing_note"]
        assert got["expert_other_hash"]["routing"] is None and note.startswith(evidence.FINAL), note
        assert "law-hash-1" in note and "differs from the row's hash law-hash-0" in note, note
        note = got["expert_no_set"]["routing_note"]
        assert note.startswith(evidence.FINAL) and "no set named 'astronomy'" in note, note
        assert got["expert_final"]["routing_note"] == evidence.FINAL + "an earlier answer"

        # a second run asks nothing
        enrich_rows.enrich_model("toy", routing=True, force=True, versions=[], healdata=hd, stamp="2026-11-15T03:00:00")
        assert len(calls()) == 2 and by_id(path) == got
    finally:
        common.REGISTRY_CSV, client.modal = real


def main():
    real = (common.ROOT, evidence.POOL_SKILLS)
    root = tempfile.mkdtemp(prefix="wsp_enrich_pool_")
    common.ROOT = root
    try:
        hd = evidence.load_healdata()
        assert hd is not None and "cais/mmlu" in [str(b).lower() for b in hd.BANNED]
        os.makedirs(os.path.join(root, "catalog"))
        path = common.catalog_path("toy")
        selection(root, path, hd)
        routing(root, path, hd)
    finally:
        common.ROOT, evidence.POOL_SKILLS = real
        shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    main()
    print("ok")
