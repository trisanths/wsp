# Checks what client.Miner.write_part puts in a catalog row: the old fields, the new evidence fields,
# the fallbacks for an engine deployment that has neither builtin_info nor expert_pick_rates, and that
# a row is written when the files behind selection_data cannot be read.
# The engine is a local stand-in (no Modal call); the row logic under test is the real one.
#   .venv/bin/python mine/tests/test_write_part.py
import json
import os
import shutil
import sys
import tempfile

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import common  # noqa: E402
import client  # noqa: E402
import evidence  # noqa: E402

OLD_FIELDS = ["id", "type", "label", "tier", "trained_on", "model", "revision", "d_model", "units", "num_units",
              "skill", "signature", "evidence_used", "weight_rationale", "part_key", "created"]
DROPS = {"math|search": 0.01, "math|heldout": 0.0, "code|search": -0.02, "code|heldout": 0.03,
         "general_text|search": 0.0, "general_text|heldout": 0.0,
         "sk|search": 0.5, "sk__control|search": 0.01, "sk|heldout": 0.4, "sk__control|heldout": 0.0}
SCREEN = {"drops": DROPS, "random_drops": [0.1, 0.2, 0.3], "gate": {"passed": True, "reasons": [], "gate_version": 2}}
PICKS = {"rates": {
    "sk|search": {"n_items": 2, "layers": {"3": {"rates": [0.0, 0.5, 0.25, 1.25], "tokens": 40}}},
    "general_text|search": {"n_items": 2, "layers": {"3": {"rates": [0.25, 0.25, 0.5, 1.0], "tokens": 60}}},
    "sk|heldout": {"n_items": 2, "layers": {"3": {"rates": [0.0, 0.5, 0.25, 1.25], "tokens": 40}}},
    "general_text|heldout": {"n_items": 2, "layers": {"3": {"rates": [0.25, 0.25, 0.5, 1.0], "tokens": 60}}}},
    "counted_layers": [3], "unsupported": {}, "missing": []}
# an engine whose image still holds benchmark built-in sets, while the local file has moved on
ENGINE_BUILTIN = {"math": {"description": "GSM8K word problems; digit tokens of the worked solution", "dataset": None,
                           "items_sha": "sha-old-math"},
                  "code": {"description": "MBPP Python tasks; tokens of the reference solution", "dataset": None,
                           "items_sha": "sha-old-code"},
                  "general_text": {"description": "WikiText paragraph continuation (control only)", "dataset": None,
                                   "items_sha": "sha-general"}}
# the same engine after a deploy with a new built-in file
ENGINE_BUILTIN_NEW = {"math": {"description": "Orca-Math word problems (approved pool)", "dataset": None,
                               "items_sha": "sha-new-math"},
                      "code": {"description": "CodeAlpaca Python tasks (approved pool)", "dataset": None,
                               "items_sha": "sha-new-code"},
                      "general_text": ENGINE_BUILTIN["general_text"]}


class FakeMethod:
    def __init__(self, engine, name):
        self.engine = engine
        self.name = name

    def remote(self, *args):
        self.engine.calls.append((self.name, args))
        if self.name not in self.engine.answers:
            raise RuntimeError("Class has no method " + self.name)  # what an old deployment answers
        return self.engine.answers[self.name]


class FakeEngine:
    def __init__(self, answers):
        self.answers = answers
        self.calls = []

    def __getattr__(self, name):
        return FakeMethod(self, name)


def make_miner(answers):
    m = client.Miner.__new__(client.Miner)
    m.short = "toy"
    m.hf_id = "org/toy"
    m.revision = "rev1"
    m.modal_seconds = 0.0
    m.state = {"structure": {"d_model": 8},
               "agent_skills": {"sk": {"path": "mine/skills/toy/sk.json", "hash": "h1", "description": "toy skill"}}}
    m.engine = FakeEngine(answers)
    return m


def names_called(m):
    return [c[0] for c in m.engine.calls]


def main():
    root = tempfile.mkdtemp(prefix="wsp_write_part_")
    common.ROOT = root  # catalog/, parts/ and logs/ of this test live in a temp folder
    os.makedirs(os.path.join(root, "mine", "skills", "toy"))
    with open(os.path.join(root, "mine", "skills", "toy", "sk.json"), "w") as f:
        json.dump({"hash": "h1", "items": {"search": [], "heldout": []}, "control": {"search": [], "heldout": []}}, f)
    expert = {"kind": "expert", "layer": 3, "indices": [2, 1]}

    # 1. an old deployment: both new engine methods fail, and the row is still written
    m = make_miner({})
    rec = m.write_part(expert, "toy expert", "sk", SCREEN, ["weights"], "why", "screened")
    for key in OLD_FIELDS + ["layer"]:
        assert key in rec, key
    assert rec["type"] == "expert" and rec["num_units"] == 2 and rec["layer"] == 3 and rec["tier"] == "screened"
    assert rec["skill"] == {"source": "agent", "path": "mine/skills/toy/sk.json", "hash": "h1",
                            "description": "toy skill", "name": "sk"}, rec["skill"]
    sig = rec["signature"]
    assert sig["effect"] == DROPS and sig["random_drops"] == [0.1, 0.2, 0.3] and sig["gate"] == SCREEN["gate"]
    assert sig["confirm"] is None
    # (0.5 - 0.2) / 0.1 = 3; the largest side effect on the search half is code at -0.02, held-out code at 0.03
    assert sig["z"] == 3.0 and sig["z_heldout"] is None and sig["random_n"] == 3, sig
    assert sig["guard"]["search"] == {"skill": "code", "drop": -0.02, "abs": 0.02}, sig
    assert sig["guard"]["heldout"] == {"skill": "code", "drop": 0.03, "abs": 0.03}, sig
    assert rec["routing"] is None and "expert_pick_rates failed" in rec["routing_note"], rec
    sd = rec["selection_data"]
    assert sd["skill"] == "sk" and sd["source"] == "agent" and sd["path"] == "mine/skills/toy/sk.json", sd
    assert sd["description"] == "toy skill" and sd["control"] == "sk__control", sd
    assert sd["off_target"] == ["math", "code", "general_text"] and sd["pool"] is None, sd
    assert sd["builtin_basis"] == "local file", sd
    assert sd["path_matches"] is True and "selection_note" not in rec, sd  # sk.json holds the hash of the state
    local = evidence.local_builtin()
    assert sd["off_target_data"]["math"]["description"] == local["math"]["description"], sd
    assert names_called(m) == ["builtin_info", "expert_pick_rates"], names_called(m)
    saved = torch.load(os.path.join(root, rec["units"]))
    assert saved["kind"] == "expert" and saved["indices"].tolist() == [1, 2] and saved["layer"] == 3
    with open(common.catalog_path("toy")) as f:
        lines = [json.loads(line) for line in f if line.strip()]
    assert lines == [rec], "the row on disk is the row returned"

    # the same part and label again: the stored row comes back and the engine is not asked for routing
    again = m.write_part(expert, "toy expert", "sk", SCREEN, ["weights"], "why", "screened")
    assert again == rec and names_called(m) == ["builtin_info", "expert_pick_rates"], names_called(m)

    # 2. a current deployment: the engine names its own built-in sets and counts the router's picks
    m = make_miner({"builtin_info": ENGINE_BUILTIN, "expert_pick_rates": PICKS})
    rec = m.write_part({"kind": "expert", "layer": 3, "indices": [1]}, "second expert", "sk", SCREEN, [], "why", "screened")
    assert "routing_note" not in rec, rec
    assert rec["routing"]["layer"] == 3 and rec["routing"]["skill"] == "sk", rec["routing"]
    assert rec["routing"]["revision"] == "rev1" == rec["revision"], rec["routing"]  # the weights that were counted
    # ln(0.5 / 0.25) = 0.6931
    assert rec["routing"]["experts"] == {"1": {
        "search": {"skill_rate": 0.5, "general_rate": 0.25, "log_ratio": 0.6931},
        "heldout": {"skill_rate": 0.5, "general_rate": 0.25, "log_ratio": 0.6931}}}, rec["routing"]
    sd = rec["selection_data"]
    assert sd["builtin_basis"] == "engine", sd
    assert sd["off_target_data"]["math"]["description"] == ENGINE_BUILTIN["math"]["description"], sd
    assert sd["off_target_data"]["math"]["items_sha"] == "sha-old-math", sd  # names the version the engine held
    hd = evidence.load_healdata()
    assert "openai/gsm8k" in hd.BANNED and "google-research-datasets/mbpp" in hd.BANNED
    assert sd["reported_benchmarks"] == ["openai/gsm8k", "google-research-datasets/mbpp"], sd
    call = [c for c in m.engine.calls if c[0] == "expert_pick_rates"][0][1]
    assert sorted(call[0]["skills"]) == ["general_text", "sk"] and call[0]["skills"]["general_text"] == {"builtin": True}
    assert call[1] == ["sk", "general_text"] and call[2] == [3] and call[3] == client.ROUTING_ITEMS, call[1:]
    assert rec["id"].endswith("_001") and rec["id"] != lines[0]["id"]

    # 3. a part that is not an expert has no routing field, and the engine is not asked for one
    m = make_miner({"builtin_info": ENGINE_BUILTIN, "expert_pick_rates": PICKS})
    rec = m.write_part({"kind": "mlp_cluster", "layer": 2, "indices": [5, 9, 7]}, "toy mlp", "sk", SCREEN, [], "why",
                       "screened")
    assert "routing" not in rec and "routing_note" not in rec and "selection_data" in rec and "z" in rec["signature"]
    assert names_called(m) == ["builtin_info"], names_called(m)
    rec = m.write_part({"kind": "block", "layers": [4, 5]}, "toy block", "sk", SCREEN, [], "why", "screened")
    assert rec["layers"] == [4, 5] and "routing" not in rec and rec["signature"]["z"] == 3.0

    # 4. the engine counted nothing for this layer (module_list experts): null routing with the reason
    unsupported = {"rates": {}, "counted_layers": [], "unsupported": {"3": "module_list experts: no picks"}, "missing": []}
    m = make_miner({"builtin_info": ENGINE_BUILTIN, "expert_pick_rates": unsupported})
    rec = m.write_part({"kind": "expert", "layer": 3, "indices": [0]}, "third expert", "sk", SCREEN, [], "why", "screened")
    assert rec["routing"] is None and "module_list" in rec["routing_note"], rec

    # 5. the state already shows module_list experts in the layer: the engine is not asked at all
    m = make_miner({"builtin_info": ENGINE_BUILTIN, "expert_pick_rates": PICKS})
    m.state["structure"]["layers_detail"] = [{"layer": l} for l in range(3)] + [
        {"layer": 3, "experts": {"path": "mlp.experts", "n_experts": 4, "style": "module_list"}}]
    rec = m.write_part({"kind": "expert", "layer": 3, "indices": [3]}, "fourth expert", "sk", SCREEN, [], "why", "screened")
    assert rec["routing"] is None and rec["routing_note"].startswith(evidence.FINAL), rec
    assert "module_list" in rec["routing_note"] and names_called(m) == ["builtin_info"], (rec, names_called(m))

    # 6. the engine does not hold the target skill: no routing object with null rates, and a note
    no_target = {"rates": {key: PICKS["rates"][key] for key in ["general_text|search", "general_text|heldout"]},
                 "counted_layers": [3], "unsupported": {}, "missing": ["sk|search", "sk|heldout"]}
    m = make_miner({"builtin_info": ENGINE_BUILTIN, "expert_pick_rates": no_target})
    rec = m.write_part({"kind": "expert", "layer": 3, "indices": [0, 3]}, "fifth expert", "sk", SCREEN, [], "why", "screened")
    assert rec["routing"] is None and "the engine does not hold this skill" in rec["routing_note"], rec

    # 7. a deploy with a new built-in file while the run goes on. The answer of builtin_info is kept
    # for BUILTIN_INFO_SECONDS, then asked again, and the row names the new sets.
    m = make_miner({"builtin_info": ENGINE_BUILTIN})
    rec = m.write_part({"kind": "mlp_cluster", "layer": 2, "indices": [1]}, "before deploy", "sk", SCREEN, [], "why", "screened")
    assert rec["selection_data"]["off_target_data"]["math"]["items_sha"] == "sha-old-math"
    m.engine.answers["builtin_info"] = ENGINE_BUILTIN_NEW
    rec = m.write_part({"kind": "mlp_cluster", "layer": 2, "indices": [2]}, "just after", "sk", SCREEN, [], "why", "screened")
    assert names_called(m) == ["builtin_info"], "a fresh answer is not asked again"
    m._engine_builtin_at -= client.BUILTIN_INFO_SECONDS + 1
    rec = m.write_part({"kind": "mlp_cluster", "layer": 2, "indices": [3]}, "later", "sk", SCREEN, [], "why", "screened")
    assert names_called(m) == ["builtin_info", "builtin_info"], names_called(m)
    sd = rec["selection_data"]
    assert sd["builtin_basis"] == "engine" and sd["off_target_data"]["math"]["items_sha"] == "sha-new-math", sd
    assert sd["off_target_data"]["math"]["description"] == ENGINE_BUILTIN_NEW["math"]["description"], sd
    assert sd["reported_benchmarks"] == [], sd

    # 8. another lane replaced the frozen test file of the skill: the row says that the file at
    # path is not the test it was selected on
    with open(os.path.join(root, "mine", "skills", "toy", "sk.json"), "w") as f:
        json.dump({"hash": "h2", "items": {"search": [], "heldout": []}, "control": {"search": [], "heldout": []}}, f)
    rec = m.write_part({"kind": "mlp_cluster", "layer": 2, "indices": [4]}, "other lane", "sk", SCREEN, [], "why", "screened")
    sd = rec["selection_data"]
    assert sd["hash"] == "h1" and sd["path_matches"] is False and sd["path_hash"] == "h2", sd

    # 9. wsp/healdata.py cannot be executed. The part passed the paid gate: its row and its index
    # file are written, with selection_data null and the reason in selection_note.
    real_healdata = evidence.HEALDATA
    evidence.HEALDATA = os.path.join(root, "healdata_broken.py")
    with open(evidence.HEALDATA, "w") as f:
        f.write("from .eval import nothing\n")
    try:
        rec = m.write_part({"kind": "mlp_cluster", "layer": 2, "indices": [5]}, "no healdata", "sk", SCREEN, [], "why",
                           "screened")
    finally:
        evidence.HEALDATA = real_healdata
    assert rec["selection_data"] is None and "ImportError" in rec["selection_note"], rec
    assert rec["signature"]["z"] == 3.0 and os.path.exists(os.path.join(root, rec["units"])), rec
    with open(common.catalog_path("toy")) as f:
        assert json.loads(f.read().strip().splitlines()[-1]) == rec, "the row is on disk"

    # 10. the engine cannot be asked and the local built-in file is broken: the row is written all the same
    real_builtin = common.BUILTIN_SKILLS
    common.BUILTIN_SKILLS = os.path.join(root, "builtin_broken.json")
    with open(common.BUILTIN_SKILLS, "w") as f:
        f.write("{not json")
    try:
        m = make_miner({})
        rec = m.write_part({"kind": "mlp_cluster", "layer": 2, "indices": [6]}, "no builtin file", "sk", SCREEN, [], "why",
                           "screened")
    finally:
        common.BUILTIN_SKILLS = real_builtin
    assert rec["selection_data"] is None and "selection_data could not be built" in rec["selection_note"], rec
    assert os.path.exists(os.path.join(root, rec["units"])), rec

    with open(common.catalog_path("toy")) as f:
        rows = [json.loads(line) for line in f if line.strip()]
    assert len(rows) == 13 and len(set(r["id"] for r in rows)) == 13, [r["id"] for r in rows]
    assert len(os.listdir(os.path.join(root, "parts", "toy"))) == 13, "one index file for each row, none left over"
    shutil.rmtree(root)
    print("ok")


main()
