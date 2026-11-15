"""Tests for scripts/library_from_maps.py. CPU only, no network. Uses a small made-up map file."""

import importlib.util
import json
import os
import shutil

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT = os.path.join(REPO, "scripts", "library_from_maps.py")

spec = importlib.util.spec_from_file_location("library_from_maps", SCRIPT)
maps = importlib.util.module_from_spec(spec)
spec.loader.exec_module(maps)
lib = maps.lib


def fake_map():
    scores = [0.0, 0.1, 0.5, 0.6, 0.1, 0.0, 0.2, 0.3]
    layer_maps = {
        "law": {"scores": scores, "influence": scores, "blocks": [[2, 3], [6, 7]], "block_mass": [0.6, 0.3],
                "heldout": {"blocks": [[2, 3], [6, 7]], "select_mass": [0.6, 0.3], "check_mass": [0.7, 0.1],
                            "split_half_r": 0.9}},
        "noisy": {"scores": scores, "influence": scores, "blocks": [[2, 3]], "block_mass": [0.6],
                  "heldout": {"blocks": [[2, 3]], "select_mass": [0.6], "check_mass": [0.9], "split_half_r": 0.2}},
        "nor": {"scores": scores, "influence": scores, "blocks": [[2, 3]], "block_mass": [0.6],
                "heldout": {"blocks": [[2, 3]], "select_mass": [0.6], "check_mass": [0.9], "split_half_r": None}},
    }
    rows = [{"expert": 4, "skill_freq": 0.2, "general_freq": 0.05, "log_ratio": 1.38, "heldout_log_ratio": 1.1},
            {"expert": 9, "skill_freq": 0.1, "general_freq": 0.05, "log_ratio": 0.69, "heldout_log_ratio": 0.5}]
    return {"model": "Org/Tiny-7B", "skills": {"law": 0.3}, "layer_maps": layer_maps,
            "expert_maps": {"law": {"1": [4, 9], "2": [3]}},
            "expert_details": {"law": {"1": rows, "2": [rows[0]]}},
            "params_b": 7.0, "layers": 8, "hidden": 64, "family": "tiny", "architecture": "TinyForCausalLM",
            "license": "mit", "revision": "abc", "selection_data": {"law": "plain legal text"}}


def setup(tmp_path):
    entries = tmp_path / "entries"
    entries.mkdir()
    with open(entries / "Org__Tiny-7B.json", "w") as f:
        json.dump(fake_map(), f)
    library = str(tmp_path / "library")
    os.makedirs(library)
    shutil.copy(os.path.join(REPO, "library", "index.jsonl"), library)
    return str(entries), library


def run(entries, library, *extra):
    return maps.main(["write", "--entries-dir", entries, "--library", library] + list(extra))


def test_writes_filters_and_validates(tmp_path, capsys):
    entries, library = setup(tmp_path)
    assert run(entries, library) == 0
    out = capsys.readouterr().out
    assert "wrote 1 block and 1 expert entries" in out
    assert "split_half_r missing: 1" in out
    assert "split_half_r below minimum: 1" in out
    assert "check_mass below minimum: 1" in out
    assert "fewer experts than minimum: 1" in out
    assert lib.validate(library) == []
    ids = sorted(row["id"] for row in lib.read_jsonl(os.path.join(library, "index.jsonl")) if row["id"].startswith("map_"))
    assert ids == ["map_block_org__tiny-7b_law_l2_3", "map_expert_org__tiny-7b_law_l1"]
    block = lib.read_json(os.path.join(library, "components/Org__Tiny-7B/map_block_org__tiny-7b_law_l2_3.json"))
    assert block["status"] == "candidate" and block["generated"] is False
    assert block["params_b"] == 7.0 and block["selection_data"] == "plain legal text"
    assert block["layers"] == [2, 3]
    assert block["signature"]["summed_layer_score"] == 1.1


def test_expert_index_file(tmp_path):
    import torch
    entries, library = setup(tmp_path)
    run(entries, library)
    path = os.path.join(library, "components/Org__Tiny-7B/map_expert_org__tiny-7b_law_l1.pt")
    assert torch.load(path) == {"kind": "expert", "model": "Org/Tiny-7B", "layer": 1, "indices": [4, 9]}


def test_second_run_changes_nothing(tmp_path, capsys):
    entries, library = setup(tmp_path)
    run(entries, library)
    before = {}
    for root, _, names in os.walk(library):
        for name in names:
            with open(os.path.join(root, name), "rb") as f:
                before[os.path.join(root, name)] = f.read()
    capsys.readouterr()
    run(entries, library)
    assert "wrote 0 block and 0 expert" in capsys.readouterr().out
    for path, data in before.items():
        with open(path, "rb") as f:
            assert f.read() == data


def test_thresholds_are_options(tmp_path, capsys):
    entries, library = setup(tmp_path)
    run(entries, library, "--min-check-mass", "0.05", "--min-split-half-r", "0.1", "--min-experts", "1")
    out = capsys.readouterr().out
    assert "wrote 3 block and 2 expert entries" in out
    assert lib.validate(library) == []
