# Checks mine/enrich_rows.py on a synthetic catalog in a temp folder: the new fields, that old fields
# and the row order survive, that a second run changes nothing but enriched_at, the live-run refusal,
# and that a rejected file is never touched. No engine call.
#   .venv/bin/python mine/tests/test_enrich_rows.py
import copy
import json
import os
import shutil
import sys
import tempfile
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import common  # noqa: E402
import client  # noqa: E402
import enrich_rows  # noqa: E402
import evidence  # noqa: E402

OLD = {"math": {"source": "builtin", "description": "GSM8K word problems; digit tokens of the worked solution",
                "datasets": ["openai/gsm8k"]},
       "code": {"source": "builtin", "description": "MBPP Python tasks; tokens of the reference solution",
                "datasets": ["google-research-datasets/mbpp"]},
       "general_text": {"source": "builtin", "description": "WikiText paragraph continuation (control only)",
                        "datasets": ["Salesforce/wikitext"]}}
NEW = {"math": {"source": "builtin", "description": "Orca-Math word problems (approved pool)",
                "datasets": ["microsoft/orca-math-word-problems-200k"]},
       "code": {"source": "builtin", "description": "CodeAlpaca Python tasks (approved pool)",
                "datasets": ["sahil2801/CodeAlpaca-20k"]},
       "general_text": OLD["general_text"]}
# two versions of the built-in file: "oldrev" from the start, "newrev" from 2026-11-14 20:00 local time
CHANGE = time.mktime(time.strptime("2026-11-14T20:00:00", enrich_rows.TIME_FORMAT))
VERSIONS = [{"rev": "oldrev000000", "epoch": CHANGE - 7200, "time_from": "commit"},
            {"rev": "newrev000000", "epoch": CHANGE, "time_from": "commit"}]


def effect(target):
    out = {"math|search": 0.01, "math|heldout": 0.0, "code|search": -0.02, "code|heldout": 0.03,
           "general_text|search": 0.0, "general_text|heldout": 0.0,
           "other_skill|search": 0.04, "other_skill__control|search": 0.0,
           "other_skill|heldout": -0.06, "other_skill__control|heldout": 0.0}
    out[target + "|search"] = 0.5
    out[target + "|heldout"] = 0.4
    if target not in ["math", "code"]:
        out[target + "__control|search"] = 0.2
        out[target + "__control|heldout"] = 0.2
    return out


def row(ident, kind, target, created, skill):
    return {"id": ident, "type": kind, "label": "label of " + ident, "tier": "screened",
            "trained_on": "ablation screen on skill '" + target + "'", "model": "org/toy", "revision": "rev1",
            "d_model": 8, "units": "parts/toy/" + ident + ".pt", "num_units": 2, "skill": dict(skill, name=target),
            "signature": {"effect": effect(target), "random_drops": [0.1, 0.2, 0.3],
                          "gate": {"passed": True, "reasons": [], "limit_other": 0.1667, "gate_version": 2},
                          "confirm": None},
            "evidence_used": ["weights"], "weight_rationale": "why", "part_key": kind + ":3:abc",
            "created": created, "layer": 3}


def read(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def raw(path):
    with open(path, "rb") as f:
        return f.read()


def without_stamp(rows):
    rows = copy.deepcopy(rows)
    for r in rows:
        r.pop("enriched_at", None)
    return rows


def age_logs(root, seconds):
    folder = os.path.join(root, "logs", "toy")
    for name in os.listdir(folder):
        then = time.time() - seconds
        os.utime(os.path.join(folder, name), (then, then))


def main():
    root = tempfile.mkdtemp(prefix="wsp_enrich_")
    common.ROOT = root
    common.MINE_DIR = os.path.join(root, "mine")  # skills_dir: the frozen tests of the toy model
    enrich_rows._at.update({"oldrev000000": OLD, "newrev000000": NEW})  # the two versions, without git
    hd = evidence.load_healdata()
    assert hd is not None and "openai/gsm8k" in hd.BANNED and "google-research-datasets/mbpp" in hd.BANNED

    os.makedirs(os.path.join(root, "catalog"))
    os.makedirs(os.path.join(root, "parts", "toy"))
    os.makedirs(os.path.join(root, "logs", "toy"))
    os.makedirs(os.path.join(root, "mine", "skills", "toy"))
    for name in ["sk", "other_skill"]:
        with open(os.path.join(root, "mine", "skills", "toy", name + ".json"), "w") as f:
            json.dump({"name": name, "description": "frozen " + name, "hash": "hash-" + name,
                       "items": {"search": [], "heldout": []}, "control": {"search": [], "heldout": []}}, f)
    agent_skill = {"source": "agent", "path": "mine/skills/toy/sk.json", "hash": "hash-sk", "description": "frozen sk"}
    rows = [row("mlp_a", "mlp_cluster", "sk", "2026-11-14T19:00:00", agent_skill),          # before the change
            {"id": "v1_mask", "type": "deletion_mask", "signature": {"note": "not from mine/"}, "extra": [1, 2]},
            row("heads_b", "attn_heads", "math", "2026-11-14T21:00:00", {"source": "builtin"}),  # after it
            row("expert_c", "expert", "sk", "2026-11-14T21:30:00", agent_skill)]
    rows[0]["signature"]["random_drops"] = [0.1]          # one random drop: no z
    rows[3]["custom_field"] = {"kept": True}
    path = common.catalog_path("toy")
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    rejected = os.path.join(root, "catalog", "toy.rejected.jsonl")
    with open(rejected, "w") as f:
        f.write(json.dumps(dict(rows[0], rejected_reasons=["x"])) + "\n")
    rejected_before = raw(rejected)
    torch.save({"kind": "expert", "model": "org/toy", "layer": 3, "indices": torch.tensor([1, 2])},
               os.path.join(root, "parts", "toy", "expert_c.pt"))
    with open(os.path.join(root, "logs", "toy", "state.json"), "w") as f:
        f.write("{}")

    # a log file that just changed: the model counts as live and nothing is written
    before = raw(path)
    status = enrich_rows.enrich_model("toy", versions=VERSIONS, healdata=hd, stamp="2026-11-15T00:00:00")
    assert status.startswith("refused") and "state.json" in status, status
    assert raw(path) == before and enrich_rows.live_files("toy") == ["state.json"]
    assert enrich_rows.enrich_model("toy.rejected", versions=VERSIONS, healdata=hd).startswith("refused")
    assert enrich_rows.enrich_model("no_such_model", versions=VERSIONS, healdata=hd).startswith("skipped")

    # --force runs anyway
    status = enrich_rows.enrich_model("toy", force=True, versions=VERSIONS, healdata=hd, stamp="2026-11-15T00:00:00")
    assert status == "enriched 3 of 4 rows", status
    with open(path, "wb") as f:
        f.write(before)  # start again from the old file for the checks below

    # no log change for an hour: not live
    age_logs(root, 3600)
    assert enrich_rows.live_files("toy") == []
    status = enrich_rows.enrich_model("toy", versions=VERSIONS, healdata=hd, stamp="2026-11-15T00:00:00")
    assert status == "enriched 3 of 4 rows", status
    out = read(path)
    first_bytes = raw(path)

    # order and old fields
    assert [r["id"] for r in out] == ["mlp_a", "v1_mask", "heads_b", "expert_c"]
    assert out[1] == rows[1], "a row that mining did not write stays as it is"
    for old, new in zip(rows, out):
        for key in old:
            if key == "signature":
                for sub in old["signature"]:
                    assert new["signature"][sub] == old["signature"][sub], (old["id"], sub)
            else:
                assert new[key] == old[key], (old["id"], key)
        assert list(new.keys())[:len(old)] == list(old.keys()), "old fields keep their place"
    assert out[3]["custom_field"] == {"kept": True}

    # row 0: one random drop gives no z. Guard: search other_skill 0.04, held-out other_skill -0.06;
    # the target's own control (0.2) does not count.
    sig = out[0]["signature"]
    assert sig["z"] is None and sig["random_n"] == 1 and sig["random_mean"] == 0.1 and sig["random_std"] is None, sig
    assert sig["guard"] == {"search": {"skill": "other_skill", "drop": 0.04, "abs": 0.04},
                            "heldout": {"skill": "other_skill", "drop": -0.06, "abs": 0.06}}, sig
    # rows 2 and 3: (0.5 - 0.2) / 0.1 = 3
    assert out[2]["signature"]["z"] == 3.0 and out[2]["signature"]["random_std"] == 0.1, out[2]["signature"]
    assert out[3]["signature"]["z"] == 3.0 and out[3]["signature"]["z_heldout"] is None

    # row 0 was created before the change: it was gated against the old built-in sets, two benchmarks
    sd = out[0]["selection_data"]
    assert sd["skill"] == "sk" and sd["source"] == "agent" and sd["path"] == "mine/skills/toy/sk.json", sd
    assert sd["description"] == "frozen sk" and sd["control"] == "sk__control", sd
    assert sd["off_target"] == ["math", "code", "general_text", "other_skill"], sd
    assert sd["off_target_data"]["math"]["description"] == OLD["math"]["description"], sd
    assert sd["off_target_data"]["other_skill"] == {"source": "agent", "hash": "hash-other_skill"}, sd
    assert sd["builtin_basis"] == "git history" and sd["builtin_version"]["commit"] == "oldrev000000", sd
    assert sd["reported_benchmarks"] == ["openai/gsm8k", "google-research-datasets/mbpp"] and sd["pool"] is None, sd
    # row 2 was created after it: a built-in target from the new sets, no benchmark
    sd = out[2]["selection_data"]
    assert sd["skill"] == "math" and sd["source"] == "builtin" and sd["description"] == NEW["math"]["description"], sd
    assert sd["datasets"] == NEW["math"]["datasets"] and sd["control"] is None, sd
    assert sd["off_target"] == ["code", "general_text", "other_skill"], sd
    assert sd["builtin_version"] == {"commit": "newrev000000", "time_from": "commit", "since": "2026-11-14T20:00:00"}, sd
    assert sd["reported_benchmarks"] == [] and sd["pool"] == "math" and sd["pool_match"] == "name", sd
    for i in [0, 2, 3]:
        assert out[i]["enriched_at"] == "2026-11-15T00:00:00"
    assert "enriched_at" not in out[1] and "routing" not in out[3], "routing is not touched without --routing"

    # the rejected file and the folder: nothing else changed, no temp file left
    assert raw(rejected) == rejected_before
    assert sorted(os.listdir(os.path.join(root, "catalog"))) == ["toy.jsonl", "toy.jsonl.lock", "toy.rejected.jsonl"]

    # a second run: the same bytes with the same stamp, and the same rows apart from enriched_at with a new one
    enrich_rows.enrich_model("toy", versions=VERSIONS, healdata=hd, stamp="2026-11-15T00:00:00")
    assert raw(path) == first_bytes
    enrich_rows.enrich_model("toy", versions=VERSIONS, healdata=hd, stamp="2026-11-16T09:00:00")
    second = read(path)
    assert second[0]["enriched_at"] == "2026-11-16T09:00:00"
    assert without_stamp(second) == without_stamp(out)
    assert json.dumps(without_stamp(second)) == json.dumps(without_stamp(out)), "key order is the same too"

    # a row that write_part wrote with the engine's own description keeps it, whatever git says
    kept = read(path)
    kept[2]["selection_data"]["builtin_basis"] = "engine"
    kept[2]["selection_data"]["description"] = "what the engine reported"
    kept[2]["selection_data"]["datasets"] = []
    with open(path, "w") as f:
        for r in kept:
            f.write(json.dumps(r) + "\n")
    enrich_rows.enrich_model("toy", versions=VERSIONS, healdata=hd, stamp="2026-11-16T10:00:00")
    sd = read(path)[2]["selection_data"]
    assert sd["builtin_basis"] == "engine" and sd["description"] == "what the engine reported", sd
    assert "builtin_version" not in sd and sd["off_target_data"]["code"]["description"] == NEW["code"]["description"], sd

    # which version a row gets
    assert enrich_rows.pick_version(VERSIONS, CHANGE - 1)["rev"] == "oldrev000000"
    assert enrich_rows.pick_version(VERSIONS, CHANGE)["rev"] == "newrev000000"
    assert enrich_rows.pick_version(VERSIONS, CHANGE - 99999)["rev"] == "oldrev000000"  # older than every commit
    assert enrich_rows.pick_version(VERSIONS, None)["rev"] == "newrev000000"
    assert enrich_rows.pick_version([], CHANGE) is None
    assert enrich_rows.parse_time("not a time") is None

    # --routing, with a stand-in for the engine client: the expert row gets what the engine answered,
    # other rows get nothing, and a later run does not ask again for a row that has routing
    asked = []

    class FakeMiner:
        def __init__(self, short):
            self.modal_seconds = 0.0

        def expert_routing(self, part, target, extra_skills=None):
            asked.append((part, target, sorted(extra_skills or {})))
            return {"layer": part["layer"], "experts": {"1": {}, "2": {}}}, None

    real_miner = client.Miner
    client.Miner = FakeMiner
    try:
        enrich_rows.enrich_model("toy", routing=True, versions=VERSIONS, healdata=hd, stamp="2026-11-16T11:00:00")
        got = read(path)
        assert asked == [({"kind": "expert", "layer": 3, "indices": [1, 2]}, "sk", ["sk"])], asked
        assert got[3]["routing"] == {"layer": 3, "experts": {"1": {}, "2": {}}} and "routing_note" not in got[3]
        assert "routing" not in got[0] and "routing" not in got[2]
        enrich_rows.enrich_model("toy", routing=True, versions=VERSIONS, healdata=hd, stamp="2026-11-16T12:00:00")
        assert len(asked) == 1, asked
    finally:
        client.Miner = real_miner

    # a broken line: stop, and leave the file as it is
    with open(path, "a") as f:
        f.write("{not json\n")
    broken = raw(path)
    try:
        enrich_rows.enrich_model("toy", versions=VERSIONS, healdata=hd)
        raise AssertionError("a broken line must stop the run")
    except SystemExit as e:
        assert "not JSON" in str(e), e
    assert raw(path) == broken

    shutil.rmtree(root)
    git_check()
    print("ok")


def git_check():
    # With the real git history of this checkout: a row older than every commit of the built-in file
    # gets the first version. Skipped when there is no history (a copy outside git, a shallow clone).
    common.MINE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    versions = enrich_rows.builtin_versions()
    if not versions:
        print("git history check: skipped (no history for the built-in file)")
        return
    first = enrich_rows.pick_version(versions, enrich_rows.parse_time("2000-01-01T00:00:00"))
    assert first == versions[0]
    sources = enrich_rows.builtin_at(first["rev"])
    if sources is None:
        print("git history check: skipped (the first version cannot be read)")
        return
    for name in sources:
        assert sources[name]["source"] == "builtin" and sources[name]["description"], sources[name]
    print("git history check: ok,", len(versions), "version(s), first", first["rev"][:7] + ":",
          sources["math"]["description"], sources["math"]["datasets"])


main()
