# Checks mine/enrich_rows.py on a synthetic catalog in a temp folder: the new fields, that old fields
# and the row order survive, that a second run changes nothing but enriched_at, the live-run refusals,
# what a row gets when the version of the built-in file is not known, rows whose frozen test file was
# replaced, the --routing path (with a stand-in for the Modal class, no engine call), and that a
# rejected file is never touched.
#   .venv/bin/python mine/tests/test_enrich_rows.py
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
BENCHMARKS = ["openai/gsm8k", "google-research-datasets/mbpp"]
# Two versions of the built-in file. "oldrev" is there from the start; "newrev" was committed at
# 2026-11-14 20:00 local time. When the engine began to use it is a separate fact:
#   VERSIONS  nobody said, so only the commit time is known
#   KNOWN     the engine used it from the commit time on
#   LATE      the engine used it from 21:15 on
CHANGE = time.mktime(time.strptime("2026-11-14T20:00:00", enrich_rows.TIME_FORMAT))
VERSIONS = [{"rev": "oldrev000000", "epoch": CHANGE - 7200, "time_from": "commit"},
            {"rev": "newrev000000", "epoch": CHANGE, "time_from": "commit"}]
KNOWN = [VERSIONS[0], {"rev": "newrev000000", "epoch": CHANGE, "time_from": "--live-since"}]
LATE = [VERSIONS[0], {"rev": "newrev000000", "epoch": CHANGE + 4500, "time_from": "--live-since"}]
MODULE_LIST = "module_list experts"


def effect(target):
    out = {"math|search": 0.01, "math|heldout": 0.0, "code|search": -0.02, "code|heldout": 0.03,
           "general_text|search": 0.0, "general_text|heldout": 0.0,
           "other_skill|search": 0.04, "other_skill__control|search": 0.0,
           "other_skill|heldout": -0.06, "other_skill__control|heldout": 0.0}
    out[target + "|search"] = 0.5
    out[target + "|heldout"] = 0.4
    if target in ["math", "code"]:
        out["sk|search"] = 0.0  # the agent skill of the run is an off-target skill of a built-in target
        out["sk|heldout"] = 0.0
    else:
        out[target + "__control|search"] = 0.2
        out[target + "__control|heldout"] = 0.2
    return out


def row(ident, kind, target, created, skill, revision="rev1", layer=3):
    return {"id": ident, "type": kind, "label": "label of " + ident, "tier": "screened",
            "trained_on": "ablation screen on skill '" + target + "'", "model": "org/toy", "revision": revision,
            "d_model": 8, "units": "parts/toy/" + ident + ".pt", "num_units": 2, "skill": dict(skill, name=target),
            "signature": {"effect": effect(target), "random_drops": [0.1, 0.2, 0.3],
                          "gate": {"passed": True, "reasons": [], "limit_other": 0.1667, "gate_version": 2},
                          "confirm": None},
            "evidence_used": ["weights"], "weight_rationale": "why", "part_key": kind + ":3:abc",
            "created": created, "layer": layer}


def read(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def write(path, rows):
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def raw(path):
    with open(path, "rb") as f:
        return f.read()


def by_id(path):
    return {r["id"]: r for r in read(path)}


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


def save_part(root, ident, indices, layer=3):
    torch.save({"kind": "expert", "model": "org/toy", "layer": layer, "indices": torch.tensor(indices)},
               os.path.join(root, "parts", "toy", ident + ".pt"))


# ---------- a stand-in for the deployed Modal class (no engine call) ----------

class FakeRemote:
    def __init__(self, engine, name):
        self.engine = engine
        self.name = name

    def remote(self, *args):
        self.engine.calls.append((self.name, args))
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
    def __init__(self, built_with):
        self.built_with = built_with
        self.calls = []

    def __getattr__(self, name):
        return FakeRemote(self, name)


class FakeCls:
    built = []

    @staticmethod
    def from_name(app, gpu):
        def make(**kw):
            FakeCls.built.append(FakeEngine(kw))
            return FakeCls.built[-1]
        return make


def remote_calls():
    return [call for engine in FakeCls.built for call in engine.calls]


def main():
    root = tempfile.mkdtemp(prefix="wsp_enrich_")
    common.ROOT = root
    common.MINE_DIR = os.path.join(root, "mine")  # skills_dir: the frozen tests of the toy model
    enrich_rows._at.update({"oldrev000000": OLD, "newrev000000": NEW})  # the two versions, without git
    hd = evidence.load_healdata()
    assert hd is not None and "openai/gsm8k" in hd.BANNED and "google-research-datasets/mbpp" in hd.BANNED

    os.makedirs(os.path.join(root, "catalog"))
    os.makedirs(os.path.join(root, "parts", "toy"))
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
    write(path, rows)
    rejected = os.path.join(root, "catalog", "toy.rejected.jsonl")
    with open(rejected, "w") as f:
        f.write(json.dumps(dict(rows[0], rejected_reasons=["x"])) + "\n")
    rejected_before = raw(rejected)
    save_part(root, "expert_c", [1, 2])
    before = raw(path)

    # No logs/toy/ folder on this machine: the model can be live on a teammate's machine, so nothing
    # is written without --force
    status = enrich_rows.enrich_model("toy", versions=KNOWN, healdata=hd, stamp="2026-11-15T00:00:00")
    assert status.startswith("refused") and "no logs/toy/ folder" in status, status
    assert raw(path) == before
    status = enrich_rows.enrich_model("toy", force=True, versions=KNOWN, healdata=hd, stamp="2026-11-15T00:00:00")
    assert status == "enriched 3 of 4 rows", status
    with open(path, "wb") as f:
        f.write(before)

    os.makedirs(os.path.join(root, "logs", "toy"))
    state_json = os.path.join(root, "logs", "toy", "state.json")
    with open(state_json, "w") as f:
        json.dump({"revision": "STATE_REV", "model": "org/toy"}, f)

    # a log file that just changed: the model counts as live and nothing is written
    status = enrich_rows.enrich_model("toy", versions=KNOWN, healdata=hd, stamp="2026-11-15T00:00:00")
    assert status.startswith("refused") and "state.json" in status, status
    assert raw(path) == before and enrich_rows.live_files("toy") == ["state.json"]
    assert enrich_rows.enrich_model("toy.rejected", versions=KNOWN, healdata=hd).startswith("refused")
    assert enrich_rows.enrich_model("no_such_model", versions=KNOWN, healdata=hd).startswith("skipped")

    # --force runs anyway
    status = enrich_rows.enrich_model("toy", force=True, versions=KNOWN, healdata=hd, stamp="2026-11-15T00:00:00")
    assert status == "enriched 3 of 4 rows", status
    with open(path, "wb") as f:
        f.write(before)  # start again from the old file for the checks below

    # no log change for an hour: not live
    age_logs(root, 3600)
    assert enrich_rows.live_files("toy") == []
    status = enrich_rows.enrich_model("toy", versions=KNOWN, healdata=hd, stamp="2026-11-15T00:00:00")
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
    assert sig["random_distinct"] == 1 and sig["z_flag"] is None, sig
    assert sig["guard"] == {"search": {"skill": "other_skill", "drop": 0.04, "abs": 0.04},
                            "heldout": {"skill": "other_skill", "drop": -0.06, "abs": 0.06}}, sig
    # rows 2 and 3: (0.5 - 0.2) / 0.1 = 3, from 3 distinct random drops
    assert out[2]["signature"]["z"] == 3.0 and out[2]["signature"]["random_std"] == 0.1, out[2]["signature"]
    assert out[2]["signature"]["random_distinct"] == 3 and out[2]["signature"]["z_flag"] is None
    assert out[3]["signature"]["z"] == 3.0 and out[3]["signature"]["z_heldout"] is None

    # row 0 was created before the change: it was gated against the old built-in sets, two benchmarks
    sd = out[0]["selection_data"]
    assert sd["skill"] == "sk" and sd["source"] == "agent" and sd["path"] == "mine/skills/toy/sk.json", sd
    assert sd["description"] == "frozen sk" and sd["control"] == "sk__control", sd
    assert sd["off_target"] == ["math", "code", "general_text", "other_skill"], sd
    assert sd["off_target_data"]["math"]["description"] == OLD["math"]["description"], sd
    assert sd["off_target_data"]["other_skill"] == {"source": "agent", "hash": "hash-other_skill"}, sd
    assert sd["builtin_basis"] == "git history" and sd["builtin_version"]["commit"] == "oldrev000000", sd
    assert sd["reported_benchmarks"] == BENCHMARKS and sd["pool"] is None, sd
    assert sd["path_matches"] is True and "path_note" not in sd, sd  # sk.json has the hash of the row
    # row 2 was created after the engine took the new file: a built-in target from the new sets, no benchmark
    sd = out[2]["selection_data"]
    assert sd["skill"] == "math" and sd["source"] == "builtin" and sd["description"] == NEW["math"]["description"], sd
    assert sd["datasets"] == NEW["math"]["datasets"] and sd["control"] is None, sd
    assert sd["off_target"] == ["code", "general_text", "other_skill", "sk"], sd
    assert sd["off_target_data"]["sk"] == {"source": "agent", "hash": "hash-sk"}, sd  # one test named sk is known
    assert sd["builtin_basis"] == "git history", sd
    assert sd["builtin_version"] == {"commit": "newrev000000", "time_from": "--live-since",
                                     "since": "2026-11-14T20:00:00"}, sd
    assert sd["reported_benchmarks"] == [] and sd["pool"] == "math" and sd["pool_match"] == "name", sd
    assert "path_matches" not in sd and "builtin_candidates" not in sd and "builtin_note" not in sd, sd
    for i in [0, 2, 3]:
        assert out[i]["enriched_at"] == "2026-11-15T00:00:00"
    assert "enriched_at" not in out[1] and "routing" not in out[3], "routing is not touched without --routing"

    # the rejected file and the folder: nothing else changed, no temp file left
    assert raw(rejected) == rejected_before
    assert sorted(os.listdir(os.path.join(root, "catalog"))) == ["toy.jsonl", "toy.jsonl.lock", "toy.rejected.jsonl"]

    # a second run: the same bytes with the same stamp, and the same rows apart from enriched_at with a new one
    enrich_rows.enrich_model("toy", versions=KNOWN, healdata=hd, stamp="2026-11-15T00:00:00")
    assert raw(path) == first_bytes
    enrich_rows.enrich_model("toy", versions=KNOWN, healdata=hd, stamp="2026-11-16T09:00:00")
    second = read(path)
    assert second[0]["enriched_at"] == "2026-11-16T09:00:00"
    assert without_stamp(second) == without_stamp(out)
    assert json.dumps(without_stamp(second)) == json.dumps(without_stamp(out)), "key order is the same too"

    versions_unknown(path, hd, rows)
    engine_rows(path, hd)
    which_versions()
    replaced_tests(root, path, hd, agent_skill)
    routing(root, path, hd, state_json)

    # a broken line: stop, and leave the file as it is
    with open(path, "a") as f:
        f.write("{not json\n")
    broken = raw(path)
    try:
        enrich_rows.enrich_model("toy", versions=KNOWN, healdata=hd)
        raise AssertionError("a broken line must stop the run")
    except SystemExit as e:
        assert "not JSON" in str(e), e
    assert raw(path) == broken

    shutil.rmtree(root)
    git_check()
    print("ok")


def versions_unknown(path, hd, rows):
    # Only the commit time of the new built-in file is known. Rows 2 and 3 were made after that
    # commit: the engine held the new file, or it still held the old one (the deploy came later).
    # Neither is written as fact, and reported_benchmarks is the union, so a benchmark that the old
    # sets used is not reported as unused.
    status = enrich_rows.enrich_model("toy", versions=VERSIONS, healdata=hd, stamp="2026-11-16T09:30:00")
    assert status.startswith("enriched 3 of 4 rows; 2 with an uncertain built-in file version"), status
    out = read(path)
    sd = out[0]["selection_data"]  # made before the commit of the new file: one version, as before
    assert sd["builtin_basis"] == "git history" and sd["builtin_version"]["commit"] == "oldrev000000", sd
    assert "builtin_candidates" not in sd and sd["reported_benchmarks"] == BENCHMARKS, sd
    sd = out[2]["selection_data"]
    assert sd["builtin_basis"] == "git history, version uncertain" and "builtin_version" not in sd, sd
    assert sd["reported_benchmarks"] == BENCHMARKS, sd
    assert sd["description"] is None, sd
    assert sd["datasets"] == ["openai/gsm8k", "microsoft/orca-math-word-problems-200k"], sd
    assert sd["off_target_data"]["code"] == {"source": "builtin", "description": None, "uncertain": True,
                                             "datasets": ["google-research-datasets/mbpp", "sahil2801/CodeAlpaca-20k"]}, sd
    assert sd["off_target_data"]["general_text"]["description"] == OLD["general_text"]["description"], sd
    assert [c["commit"] for c in sd["builtin_candidates"]] == ["oldrev000000", "newrev000000"], sd
    assert [c["time_from"] for c in sd["builtin_candidates"]] == ["commit", "commit"], sd
    assert sd["builtin_candidates"][0]["skills"]["math"] == {"description": OLD["math"]["description"],
                                                             "datasets": ["openai/gsm8k"]}, sd
    assert sd["builtin_candidates"][1]["skills"]["math"]["description"] == NEW["math"]["description"], sd
    assert "--live-since" in sd["builtin_note"] and sd["pool"] == "math", sd
    sd = out[3]["selection_data"]  # an agent target: its own description is known, the off-target sets are not
    assert sd["builtin_basis"] == "git history, version uncertain" and sd["description"] == "frozen sk", sd
    assert sd["reported_benchmarks"] == BENCHMARKS and sd["off_target_data"]["math"]["uncertain"] is True, sd
    for old, new in zip(rows, out):  # old fields still survive
        for key in old:
            if key != "signature":
                assert new[key] == old[key], (old["id"], key)
    uncertain_bytes = raw(path)
    enrich_rows.enrich_model("toy", versions=VERSIONS, healdata=hd, stamp="2026-11-16T09:30:00")
    assert raw(path) == uncertain_bytes, "a second run gives the same file"

    # The person who deployed says the engine took the new file at 21:15. Row 2 (21:00) was gated
    # against the old sets and row 3 (21:30) against the new ones, and the uncertain marks are gone.
    status = enrich_rows.enrich_model("toy", versions=LATE, healdata=hd, stamp="2026-11-16T09:40:00")
    assert status == "enriched 3 of 4 rows", status
    out = read(path)
    sd = out[2]["selection_data"]
    assert sd["builtin_basis"] == "git history" and sd["builtin_version"]["commit"] == "oldrev000000", sd
    assert sd["description"] == OLD["math"]["description"] and sd["datasets"] == ["openai/gsm8k"], sd
    assert sd["reported_benchmarks"] == BENCHMARKS, sd
    assert "builtin_candidates" not in sd and "builtin_note" not in sd, sd
    assert sd["off_target_data"]["code"] == {"source": "builtin", "description": OLD["code"]["description"],
                                             "datasets": ["google-research-datasets/mbpp"]}, sd
    sd = out[3]["selection_data"]
    assert sd["builtin_version"] == {"commit": "newrev000000", "time_from": "--live-since",
                                     "since": "2026-11-14T21:15:00"}, sd
    assert sd["reported_benchmarks"] == [] and "builtin_candidates" not in sd, sd
    enrich_rows.enrich_model("toy", versions=KNOWN, healdata=hd, stamp="2026-11-16T09:50:00")


def engine_rows(path, hd):
    # a row that write_part wrote with the engine's own description keeps it, whatever git says
    kept = read(path)
    kept[2]["selection_data"]["builtin_basis"] = "engine"
    kept[2]["selection_data"]["description"] = "what the engine reported"
    kept[2]["selection_data"]["datasets"] = []
    kept[2]["selection_data"]["items_sha"] = "sha-from-engine"
    write(path, kept)
    for versions in [KNOWN, VERSIONS]:
        enrich_rows.enrich_model("toy", versions=versions, healdata=hd, stamp="2026-11-16T10:00:00")
        sd = read(path)[2]["selection_data"]
        assert sd["builtin_basis"] == "engine" and sd["description"] == "what the engine reported", sd
        assert sd["items_sha"] == "sha-from-engine" and "builtin_version" not in sd, sd
        assert sd["off_target_data"]["code"]["description"] == NEW["code"]["description"], sd
    enrich_rows.enrich_model("toy", versions=KNOWN, healdata=hd, stamp="2026-11-16T10:10:00")


def which_versions():
    # the versions the engine can have held for a row
    def revs(versions, when):
        return [v["rev"][:6] for v in enrich_rows.candidate_versions(versions, when)]

    assert revs(VERSIONS, CHANGE - 1) == ["oldrev"]                    # before the commit of the new file
    assert revs(VERSIONS, CHANGE - 99999) == ["oldrev"]                # older than every commit
    assert revs(VERSIONS, CHANGE) == ["oldrev", "newrev"]              # committed, deploy time not known
    assert revs(VERSIONS, CHANGE + 99999) == ["oldrev", "newrev"]
    assert revs(VERSIONS, None) == ["oldrev", "newrev"]                # a row without a time
    assert revs(KNOWN, CHANGE - 1) == ["oldrev"] and revs(KNOWN, CHANGE) == ["newrev"]
    assert revs(LATE, CHANGE + 4499) == ["oldrev"] and revs(LATE, CHANGE + 4500) == ["newrev"]
    # three versions: the middle one is known to be live from its time on, the last one has a commit time only
    three = [VERSIONS[0], {"rev": "midrev", "epoch": CHANGE, "time_from": "--live-since"},
             {"rev": "lastrev", "epoch": CHANGE + 100, "time_from": "commit"}]
    assert revs(three, CHANGE + 50) == ["midrev"] and revs(three, CHANGE + 100) == ["midrev", "lastre"]
    assert enrich_rows.candidate_versions([], CHANGE) == []
    assert enrich_rows.parse_time("not a time") is None


def replaced_tests(root, path, hd, agent_skill):
    # Lanes on one model share mine/skills/toy/. Row expert_e was selected on another lane's test
    # named "sk" (hash-sk-lane2), and the file on disk holds hash-sk. A second lane's state file names
    # another test for other_skill.
    rows = read(path)
    lane2 = dict(agent_skill, hash="hash-sk-lane2")
    extra = row("expert_e", "expert", "sk", "2026-11-14T21:40:00", lane2)
    extra["selection_data"] = None
    extra["selection_note"] = "selection_data could not be built: ImportError: x"  # what write_part leaves on an error
    write(path, rows + [extra])
    save_part(root, "expert_e", [0, 3])
    # the layers of revision rev2 as the engine described them: fused experts in layer 3, module_list in layer 5
    detail = [{"layer": l} for l in range(3)] + [
        {"layer": 3, "experts": {"path": "mlp.experts", "n_experts": 4}}, {"layer": 4},
        {"layer": 5, "experts": {"path": "mlp.experts", "n_experts": 4, "style": "module_list"}}]
    with open(os.path.join(root, "logs", "toy", "state.lane2.json"), "w") as f:
        json.dump({"revision": "rev2", "structure": {"d_model": 8, "layers_detail": detail},
                   "agent_skills": {"other_skill": {"path": "mine/skills/toy/other_skill.json",
                                                    "hash": "hash-other-lane2", "description": "d"}}}, f)
    age_logs(root, 3600)
    status = enrich_rows.enrich_model("toy", versions=KNOWN, healdata=hd, stamp="2026-11-16T10:30:00")
    assert status == "enriched 4 of 5 rows", status
    got = by_id(path)
    sd = got["expert_e"]["selection_data"]
    assert sd["hash"] == "hash-sk-lane2" and sd["path"] == "mine/skills/toy/sk.json", sd
    assert sd["path_matches"] is False and sd["path_hash"] == "hash-sk" and "another test" in sd["path_note"], sd
    assert "selection_note" not in got["expert_e"], "the note goes when selection_data is filled"
    assert got["expert_c"]["selection_data"]["path_matches"] is True, got["expert_c"]["selection_data"]
    # off-target agent skills: two tests named sk and two named other_skill are known now, so no hash
    assert got["heads_b"]["selection_data"]["off_target_data"]["sk"] == {
        "source": "agent", "hash": None, "hash_candidates": ["hash-sk", "hash-sk-lane2"]}, got["heads_b"]
    assert got["mlp_a"]["selection_data"]["off_target_data"]["other_skill"] == {
        "source": "agent", "hash": None, "hash_candidates": ["hash-other-lane2", "hash-other_skill"]}, got["mlp_a"]
    # the frozen test file is gone: not known, with the reason
    os.rename(os.path.join(root, "mine", "skills", "toy", "sk.json"), os.path.join(root, "mine", "skills", "toy", "sk.gone"))
    enrich_rows.enrich_model("toy", versions=KNOWN, healdata=hd, stamp="2026-11-16T10:40:00")
    sd = by_id(path)["expert_c"]["selection_data"]
    assert sd["path_matches"] is None and "could not be read" in sd["path_note"] and "path_hash" not in sd, sd
    os.rename(os.path.join(root, "mine", "skills", "toy", "sk.gone"), os.path.join(root, "mine", "skills", "toy", "sk.json"))
    enrich_rows.enrich_model("toy", versions=KNOWN, healdata=hd, stamp="2026-11-16T10:50:00")
    sd = by_id(path)["expert_c"]["selection_data"]
    assert sd["path_matches"] is True and "path_note" not in sd, sd


def routing(root, path, hd, state_json):
    # --routing with the real client.Miner and a stand-in for the Modal class. The state file of this
    # machine names revision STATE_REV; the rows were mined on rev1 and rev2.
    rows = read(path)
    builtin = {"source": "builtin"}
    agent_skill = dict(rows[3]["skill"])
    del agent_skill["name"]
    no_revision = row("expert_f", "expert", "sk", "2026-11-14T21:41:00", agent_skill)
    del no_revision["revision"]
    final = row("expert_j", "expert", "math", "2026-11-14T21:45:00", builtin)
    final["routing"] = None
    final["routing_note"] = evidence.FINAL + "an earlier answer"
    rows += [no_revision,
             row("expert_g", "expert", "math", "2026-11-14T21:42:00", builtin, revision="rev2", layer=5),
             row("expert_h", "expert", "math", "2026-11-14T21:43:00", builtin, revision="rev2"),
             row("expert_i", "expert", "math", "2026-11-14T21:44:00", builtin),   # its part file is missing
             final]
    write(path, rows)
    for ident in ["expert_f", "expert_h", "expert_j"]:
        save_part(root, ident, [1, 2])
    save_part(root, "expert_g", [1], layer=5)
    registry = os.path.join(root, "donors.csv")
    with open(registry, "w") as f:
        f.write("short_name,hf_id,params_b\ntoy,org/toy,1\n")
    pinned = []
    real = (common.REGISTRY_CSV, common.pin_revision, client.modal)
    common.REGISTRY_CSV = registry
    common.pin_revision = lambda hf_id: pinned.append(hf_id) or "HEAD_OF_HF_TODAY"
    client.modal = types.SimpleNamespace(Cls=FakeCls)
    age_logs(root, 3600)
    state_before = raw(state_json)
    logs_before = sorted(os.listdir(os.path.join(root, "logs", "toy")))
    try:
        status = enrich_rows.enrich_model("toy", routing=True, versions=KNOWN, healdata=hd, stamp="2026-11-16T11:00:00")
        assert status == "enriched 9 of 10 rows", status
        got = by_id(path)
        # one engine handle for each revision of the rows, in the order the rows name them; the state
        # file's revision and today's head of the HF repo are not used, and nothing is written to logs/
        assert [e.built_with for e in FakeCls.built] == [
            {"model_key": "toy", "hf_id": "org/toy", "revision": "rev1"},
            {"model_key": "toy", "hf_id": "org/toy", "revision": "rev2"}], [e.built_with for e in FakeCls.built]
        assert pinned == [] and raw(state_json) == state_before
        assert sorted(os.listdir(os.path.join(root, "logs", "toy"))) == logs_before
        assert enrich_rows.live_files("toy") == [], "the routing step does not make the model look live"
        # expert_c on rev1, with the frozen test of its skill; expert_h on rev2, a built-in target
        assert [c[0] for c in remote_calls()] == ["expert_pick_rates", "expert_pick_rates"], remote_calls()
        payload, names, layers, n_items = FakeCls.built[0].calls[0][1]
        assert names == ["sk", "general_text"] and layers == [3] and n_items == client.ROUTING_ITEMS
        assert payload["skills"]["sk"]["hash"] == "hash-sk" and payload["skills"]["general_text"] == {"builtin": True}
        assert FakeCls.built[1].calls[0][1][1] == ["math", "general_text"]
        assert got["expert_c"]["routing"]["revision"] == "rev1" and "routing_note" not in got["expert_c"], got["expert_c"]
        # ln(0.5 / 0.25) = 0.6931
        assert got["expert_c"]["routing"]["experts"]["1"]["search"] == {"skill_rate": 0.5, "general_rate": 0.25,
                                                                        "log_ratio": 0.6931}, got["expert_c"]["routing"]
        assert got["expert_h"]["routing"]["revision"] == "rev2" and got["expert_h"]["routing"]["skill"] == "math"
        # rows that no engine call can serve: null routing, a final note, and no call
        for ident, words in [("expert_e", "differs from the row's hash hash-sk-lane2"), ("expert_f", "no revision"),
                             ("expert_g", MODULE_LIST), ("expert_i", "could not read the part file")]:
            assert got[ident]["routing"] is None, got[ident]
            assert got[ident]["routing_note"].startswith(evidence.FINAL) and words in got[ident]["routing_note"], got[ident]
        # a row with a final note from an earlier run is not asked again
        assert got["expert_j"]["routing"] is None and got["expert_j"]["routing_note"] == evidence.FINAL + "an earlier answer"
        assert "routing" not in got["mlp_a"] and "routing" not in got["heads_b"]

        # a second run: no engine handle is built and no call is made
        enrich_rows.enrich_model("toy", routing=True, versions=KNOWN, healdata=hd, stamp="2026-11-16T11:00:00")
        assert len(FakeCls.built) == 2 and len(remote_calls()) == 2, remote_calls()
        assert without_stamp(read(path)) == without_stamp(list(got.values()))

        # --retry-routing asks again for final notes. The four rows above are settled again without a
        # call; expert_j goes to the engine and gets its routing.
        enrich_rows.enrich_model("toy", routing=True, retry_routing=True, versions=KNOWN, healdata=hd,
                                 stamp="2026-11-16T12:00:00")
        again = by_id(path)
        assert len(FakeCls.built) == 3 and len(remote_calls()) == 3, remote_calls()
        assert FakeCls.built[2].built_with["revision"] == "rev1"
        assert again["expert_j"]["routing"]["revision"] == "rev1" and "routing_note" not in again["expert_j"]
        assert again["expert_e"]["routing"] is None and again["expert_e"]["routing_note"] == got["expert_e"]["routing_note"]
        assert pinned == [] and raw(state_json) == state_before
    finally:
        common.REGISTRY_CSV, common.pin_revision, client.modal = real


def git_check():
    # With the real git history of this checkout: a row older than every commit of the built-in file
    # gets the first version, and a row newer than all of them gets every version unless a deploy
    # time is given. Skipped when there is no history (a copy outside git, a shallow clone).
    common.MINE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    versions = enrich_rows.builtin_versions()
    if not versions:
        print("git history check: skipped (no history for the built-in file)")
        return
    assert enrich_rows.candidate_versions(versions, enrich_rows.parse_time("2000-01-01T00:00:00")) == [versions[0]]
    later = versions[-1]["epoch"] + 1
    assert enrich_rows.candidate_versions(versions, later) == versions
    told = enrich_rows.builtin_versions({versions[-1]["rev"][:10]: versions[-1]["epoch"]})
    assert enrich_rows.candidate_versions(told, later) == [told[-1]] and told[-1]["time_from"] == "--live-since"
    sources = enrich_rows.builtin_at(versions[0]["rev"])
    if sources is None:
        print("git history check: skipped (the first version cannot be read)")
        return
    for name in sources:
        assert sources[name]["source"] == "builtin" and sources[name]["description"], sources[name]
        assert len(sources[name]["items_sha"]) == 16, sources[name]
    print("git history check: ok,", len(versions), "version(s), first", versions[0]["rev"][:7] + ":",
          sources["math"]["description"], sources["math"]["datasets"])


main()
