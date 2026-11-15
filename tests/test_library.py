"""Tests for scripts/library.py: build, validate, index, add, and query.

The build reads the real repo sources and writes into a temporary library
folder. CPU only, no network.
"""

import importlib.util
import json
import os
import shutil

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT = os.path.join(REPO, "scripts", "library.py")


def load_module():
    spec = importlib.util.spec_from_file_location("library_script", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


lib = load_module()


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    """One built and indexed library, shared by the read-only tests."""
    library = str(tmp_path_factory.mktemp("built") / "library")
    lib.build(REPO, library, quiet=True)
    lib.write_index(library, quiet=True)
    return library


@pytest.fixture
def library(built, tmp_path):
    """A fresh copy of the built library that a test may change."""
    path = str(tmp_path / "library")
    shutil.copytree(built, path)
    return path


def snapshot(folder):
    files = {}
    for root, dirs, names in os.walk(folder):
        for name in names:
            path = os.path.join(root, name)
            with open(path, "rb") as f:
                files[os.path.relpath(path, folder)] = f.read()
    return files


def read(path):
    with open(path) as f:
        return json.load(f)


def write(path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def entries_by_id(library):
    out = {}
    for data, rel in lib.load_entries(library):
        out[data["id"]] = (data, rel)
    return out


def hand_component(model="Qwen/Qwen2.5-Math-1.5B", entry_id="hand_qmath_l5_test", units_file=None):
    return {"id": entry_id, "kind": "mlp_cluster", "title": "A hand-added test cluster", "status": "candidate",
            "added_by": "test person", "source": "tests/test_library.py", "skills": ["math"], "models": [model],
            "evidence": [], "model": model, "layer": 5, "layers": [5], "units_file": units_file,
            "units_remote": None, "signature": {"method": "test"}}


# Build


def test_build_is_idempotent(library):
    before = snapshot(library)
    report = lib.build(REPO, library, quiet=True)
    lib.write_index(library, quiet=True)
    assert report["written"] == []
    assert report["copied"] == []
    assert snapshot(library) == before


def test_build_counts(built):
    counts = {}
    for row in lib.index_rows(built):
        key = row["kind"]
        counts[key] = counts.get(key, 0) + 1
    assert counts["deletion_mask"] == 37
    assert counts["model"] >= 11
    assert counts["finding"] >= 30
    assert counts["stitch"] >= 36
    entries = entries_by_id(built)
    # Person B's phi-4 layer 16 cluster: copied with its 3 random controls, rejected on held-out data.
    phi4, rel = entries["mlp_cluster_phi4_l16_gsm8k"]
    assert phi4["status"] == "rejected"
    assert len(phi4["controls"]) == 3
    assert os.path.exists(os.path.join(built, phi4["units_file"]))
    # Person A's rejected parts stay, marked rejected.
    month, rel = entries["mlp_month-name_answer_support_qwenmath1.5b_L24_k1_001"]
    assert month["status"] == "rejected"
    assert month["added_by"].startswith("Hanav")
    # The knowledge block implied by the run 7 layer map.
    block, rel = entries["block_qwen25_7b_l16_20_knowledge"]
    assert block["kind"] == "block"
    assert block["status"] == "candidate"
    assert block["layers"] == [16, 17, 18, 19, 20]


def test_round2_clusters_have_both_glue_cells(built):
    entries = entries_by_id(built)
    data, rel = entries["qmath_l10_activation512"]
    values = {}
    for item in data["evidence"]:
        if "rank-64 glue" in item["claim"]:
            values["r64"] = (item["value"], item["control_value"])
        if "full-rank glue" in item["claim"]:
            values["full"] = (item["value"], item["control_value"])
    # Run 2 table: layer 10, activation 40.5 vs random 38.7 (rank 64) and 31.2 vs 38.7 (full).
    assert values["r64"] == (0.405, 0.3867)
    assert values["full"] == (0.3117, 0.3867)
    recipe, rel = entries["graft_qmath_l10_activation512_llama1b"]
    assert recipe["evidence"][0]["value"] == 0.405
    assert recipe["evidence"][0]["seeds"] == [0, 1, 2]


def test_finding_values_are_exact_from_the_raw_files(built):
    entries = entries_by_id(built)
    rows = lib.read_json(os.path.join(REPO, "results", "round7_rows.json"))
    tap = []
    lora = []
    for row in rows:
        if row["part"] == "heldout" and row["arm"] == "tap":
            tap.append(row["accuracy"])
        if row["part"] == "heldout" and row["arm"] == "lora_r16":
            lora.append(row["accuracy"])
    item = entries["f_run7_heldout_tap"][0]["evidence"][0]
    assert item["value"] == round(sum(tap) / len(tap), 4)
    assert item["control_value"] == round(sum(lora) / len(lora), 4)
    # The 14B tap mean is over both seeds (73.6 and 67.4), not seed 0 alone.
    scaling = entries["f_run7_donor_scaling"][0]
    assert scaling["evidence"][0]["value"] == 0.705
    assert scaling["numbers"]["q14b"]["tap"] == 70.5
    # Run 8 rows are matched on their "key" column.
    assert entries["f_run8_small_hosts_reach_donors"][0]["evidence"][0]["value"] == 0.7485
    # The audit compares the sink against the same shipped ridge without it.
    sink = entries["f_audit_attention_sink"][0]["evidence"][0]
    assert (sink["value"], sink["control_value"]) == (0.367, 1.2347)
    for data, rel in lib.load_entries(built):
        if data["kind"] == "finding":
            assert data.get("numbers_source")


def test_provenance_by_person(built):
    entries = entries_by_id(built)
    assert entries["mlp_cluster_phi4_l16_gsm8k"][0]["added_by"] == "Person B"
    assert entries["gate_qmath_l10_contrast512"][0]["added_by"] == "Person B"
    assert entries["f_bake_no_gain"][0]["added_by"] == "Person B"
    assert entries["bake_tap_mmlu_llama1b_qwen7b"][0]["added_by"] == "Person B"
    assert entries["f_exit_stitch_no_gain"][0]["added_by"] == "Person B"
    assert entries["mlp_month-name_answer_support_qwenmath1.5b_L24_k1_001"][0]["added_by"] == "Hanav (Person A)"
    assert entries["qmath_block_8_19"][0]["added_by"] == "Trisanth"
    for data, rel in lib.load_entries(built):
        assert "Rex (Person B)" not in json.dumps(data)
        if data["kind"] in lib.COMPONENT_KINDS and data["source"].startswith("data/pilot_catalog"):
            assert data["added_by"] == "Rex (part 3 pilot)"


def test_run14_blocks_carry_their_evidence(built):
    entries = entries_by_id(built)
    block, rel = entries["r14_qwen25_7b_l8_20"]
    assert block["status"] == "rejected"
    assert block["source"].startswith("results/round14_rows.json")
    first = block["evidence"][0]
    assert "random" in first["control"]
    assert first["value"] < first["control_value"]
    assert first["seeds"] == [0, 1]
    big, rel = entries["r14_qwen25_14b_l14_34"]
    assert big["status"] == "rejected"
    assert len(big["evidence"]) >= 1


def test_build_keeps_hand_added_entries(library):
    # A hand-added component file.
    folder = os.path.join(library, "components", "Qwen__Qwen2.5-Math-1.5B")
    hand = hand_component()
    hand["generated"] = False
    hand["updated"] = "2026-11-01"
    write(os.path.join(folder, hand["id"] + ".json"), hand)
    # A hand-added finding line.
    finding = {"id": "f_hand_test", "kind": "finding", "title": "A hand finding", "status": "candidate",
               "added_by": "agent: test", "source": "tests/test_library.py", "updated": "2026-11-01",
               "skills": ["test"], "models": [], "evidence": [], "generated": False, "claim": "c", "setup": "s",
               "numbers": {}}
    with open(os.path.join(library, lib.FINDINGS_FILE), "a") as f:
        f.write(json.dumps(finding) + "\n")
    # Hand evidence and a status change on a generated entry.
    path = os.path.join(library, "components", "Qwen__Qwen2.5-Math-1.5B", "qmath_l10_activation512.json")
    data = read(path)
    data["evidence"].append({"claim": "held-out rerun", "metric": "GSM8K accuracy", "value": 0.5,
                             "control_value": 0.4, "control": "random same-size cluster",
                             "split": "GSM8K test (held-out)", "seeds": [0, 1, 2],
                             "source": "results/hand.json", "added_by": "test person"})
    data["status"] = "validated"
    data["status_reason"] = "hand rerun"
    write(path, data)
    # An entry that a person took over from the build.
    owned_path = os.path.join(library, "components", "microsoft__phi-4", "mlp_cluster_phi4_l17_gsm8k.json")
    owned = read(owned_path)
    owned["generated"] = False
    owned["title"] = "Owned by a person now"
    write(owned_path, owned)

    lib.build(REPO, library, quiet=True)

    assert read(os.path.join(folder, hand["id"] + ".json")) == hand
    ids = []
    for line in lib.read_jsonl(os.path.join(library, lib.FINDINGS_FILE)):
        ids.append(line["id"])
    assert "f_hand_test" in ids
    assert "f_run7_heldout_tap" in ids
    data = read(path)
    claims = []
    for item in data["evidence"]:
        claims.append(item["claim"])
    assert "held-out rerun" in claims
    assert data["status"] == "validated"
    assert data["status_reason"] == "hand rerun"
    assert read(owned_path)["title"] == "Owned by a person now"
    # The model profile links the hand-added component.
    model = read(os.path.join(library, "models", "Qwen__Qwen2.5-Math-1.5B.json"))
    assert hand["id"] in model["components"]


def test_build_never_deletes_files(library):
    extra = os.path.join(library, "components", "microsoft__phi-4", "notes.txt")
    with open(extra, "w") as f:
        f.write("keep me")
    stale = read(os.path.join(library, "recipes", "lora_r1_gsm8k_llama1b.json"))
    stale["id"] = "stale_recipe"
    write(os.path.join(library, "recipes", "stale_recipe.json"), stale)
    report = lib.build(REPO, library, quiet=True)
    assert os.path.exists(extra)
    assert os.path.exists(os.path.join(library, "recipes", "stale_recipe.json"))
    assert "recipes/stale_recipe.json" in report["stale"]


# Validate


def test_validate_passes_on_built_library(built):
    assert lib.validate(built) == []


def test_validate_catches_missing_fields(library):
    path = os.path.join(library, "components", "microsoft__phi-4", "mlp_cluster_phi4_l18_gsm8k.json")
    data = read(path)
    del data["added_by"]
    del data["signature"]
    write(path, data)
    lib.write_index(library, quiet=True)
    errors = lib.validate(library)
    text = "\n".join(errors)
    assert "missing field 'added_by'" in text
    assert "needs 'signature'" in text


def test_validate_catches_duplicate_ids(library):
    folder = os.path.join(library, "components", "microsoft__phi-4")
    data = read(os.path.join(folder, "mlp_cluster_phi4_l18_gsm8k.json"))
    with open(os.path.join(library, lib.STITCHES_FILE)) as f:
        first = json.loads(f.readline())
    first["id"] = "mlp_cluster_phi4_l18_gsm8k"
    with open(os.path.join(library, lib.STITCHES_FILE), "a") as f:
        f.write(json.dumps(first) + "\n")
    assert data["id"] == first["id"]
    errors = lib.validate(library, check_index=False)
    found = False
    for error in errors:
        if error.startswith("duplicate id 'mlp_cluster_phi4_l18_gsm8k'"):
            found = True
    assert found


def test_validate_catches_missing_files(library):
    path = os.path.join(library, "components", "microsoft__phi-4", "mlp_cluster_phi4_l18_gsm8k.json")
    data = read(path)
    os.remove(os.path.join(library, data["units_file"]))
    os.remove(os.path.join(library, data["controls"][0]))
    errors = lib.validate(library, check_index=False)
    text = "\n".join(errors)
    assert "file not found: " + data["units_file"] in text
    assert "file not found: " + data["controls"][0] in text


def test_validate_catches_bad_status_and_kind(library):
    path = os.path.join(library, "recipes", "lora_r1_gsm8k_llama1b.json")
    data = read(path)
    data["status"] = "approved"
    write(path, data)
    other = os.path.join(library, "recipes", "lora_r64_gsm8k_llama1b.json")
    data = read(other)
    data["kind"] = "adapter"
    write(other, data)
    errors = lib.validate(library, check_index=False)
    text = "\n".join(errors)
    assert "status must be one of candidate, validated, rejected, got 'approved'" in text
    assert "kind must be one of" in text


def test_validate_needs_evidence_for_validated(library):
    path = os.path.join(library, "components", "Nanbeige__Nanbeige4.3-33B", "attn_heads_nanbeige4333b_l14_gsm8k.json")
    data = read(path)
    data["status"] = "validated"
    write(path, data)
    errors = lib.validate(library, check_index=False)
    assert any_error(errors, "status 'validated' needs an evidence item")
    # A search-split gain is not enough either.
    data["evidence"] = [{"claim": "c", "metric": "m", "value": 0.4, "control_value": 0.3, "split": "GSM8K search",
                         "seeds": [0], "source": "s", "added_by": "test"}]
    write(path, data)
    errors = lib.validate(library, check_index=False)
    assert any_error(errors, "status 'validated' needs an evidence item")
    data["evidence"][0]["split"] = "GSM8K test (held-out)"
    write(path, data)
    errors = lib.validate(library, check_index=False)
    # Held out now, but the item does not say the control was a random part.
    assert any_error(errors, "status 'validated' needs an evidence item")
    data["evidence"][0]["control"] = "random same-size heads, same layer"
    write(path, data)
    errors = lib.validate(library, check_index=False)
    assert not any_error(errors, "attn_heads_nanbeige4333b_l14_gsm8k")


def test_validate_needs_a_gain_over_the_control(library):
    path = os.path.join(library, "components", "Nanbeige__Nanbeige4.3-33B", "attn_heads_nanbeige4333b_l20_gsm8k.json")
    data = read(path)
    data["status"] = "validated"
    data["evidence"] = [{"claim": "c", "metric": "GSM8K accuracy", "value": 0.30, "control_value": 0.35,
                         "control": "random same-size heads", "split": "GSM8K test (held-out)", "seeds": [0, 1, 2],
                         "source": "s", "added_by": "test"}]
    write(path, data)
    errors = lib.validate(library, check_index=False)
    assert any_error(errors, "where the value beats a random control")
    # A tie is not a gain either.
    data["evidence"][0]["value"] = 0.35
    write(path, data)
    assert any_error(lib.validate(library, check_index=False), "where the value beats a random control")
    # For a loss, lower is better when the item says so.
    data["evidence"][0]["metric"] = "stitch penalty (nats)"
    data["evidence"][0]["value"] = 0.1
    data["evidence"][0]["control_value"] = 0.6
    data["evidence"][0]["lower_is_better"] = True
    write(path, data)
    assert not any_error(lib.validate(library, check_index=False), "attn_heads_nanbeige4333b_l20_gsm8k")
    # No generated component or recipe is validated without such an item.
    for entry, rel in lib.load_entries(library):
        if entry["status"] != "validated" or entry["kind"] not in lib.COMPONENT_KINDS + ["recipe"]:
            continue
        if entry["id"] == "attn_heads_nanbeige4333b_l20_gsm8k":
            continue
        found = False
        for item in entry["evidence"]:
            if lib.beats_control_held_out(item, entry["kind"] != "recipe"):
                found = True
        assert found, entry["id"]


def test_validate_catches_bad_evidence_and_stale_index(library):
    path = os.path.join(library, "recipes", "lora_r1_gsm8k_llama1b.json")
    data = read(path)
    data["evidence"].append({"claim": "no source", "metric": "m", "value": 1})
    data["title"] = "A new title the index does not have"
    write(path, data)
    errors = lib.validate(library)
    assert any_error(errors, "evidence item is missing 'source'")
    assert any_error(errors, "needs 'added_by'")
    assert any_error(errors, "index.jsonl is out of date")


def any_error(errors, text):
    for error in errors:
        if text in error:
            return True
    return False


# Add and query


def test_add_writes_entry_copies_units_and_indexes(library, tmp_path):
    units = tmp_path / "my_part.pt"
    units.write_bytes(b"not a real index file")
    entry = hand_component(units_file=str(units))
    entry["evidence"] = [{"claim": "ablation", "metric": "drop", "value": 0.2, "control_value": 0.01,
                          "split": "held-out items", "seeds": None, "source": "results/x.json"}]
    entry_file = tmp_path / "entry.json"
    entry_file.write_text(json.dumps(entry))
    data = lib.add(library, str(entry_file))
    assert data["generated"] is False
    assert data["units_file"] == "components/Qwen__Qwen2.5-Math-1.5B/my_part.pt"
    assert data["evidence"][0]["added_by"] == "test person"
    assert os.path.exists(os.path.join(library, data["units_file"]))
    assert os.path.exists(os.path.join(library, "components", "Qwen__Qwen2.5-Math-1.5B", entry["id"] + ".json"))
    assert lib.validate(library) == []
    rows = lib.query(library, model="Qwen2.5-Math-1.5B", kind="mlp_cluster")
    ids = []
    for row in rows:
        ids.append(row["id"])
    assert entry["id"] in ids


def test_add_refuses_duplicate_and_invalid(library, tmp_path):
    entry = hand_component(entry_id="qmath_l10_activation512")
    entry_file = tmp_path / "dup.json"
    entry_file.write_text(json.dumps(entry))
    with pytest.raises(SystemExit):
        lib.add(library, str(entry_file))
    bad = hand_component(entry_id="hand_bad_status")
    bad["status"] = "great"
    bad_file = tmp_path / "bad.json"
    bad_file.write_text(json.dumps(bad))
    with pytest.raises(SystemExit):
        lib.add(library, str(bad_file))
    assert not os.path.exists(os.path.join(library, "components", "Qwen__Qwen2.5-Math-1.5B", "hand_bad_status.json"))


def test_add_finding_appends_a_line(library, tmp_path):
    finding = {"id": "f_hand_added", "kind": "finding", "title": "Hand finding", "status": "validated",
               "added_by": "agent: tester", "source": "results/x.json", "skills": ["tap"], "models": [],
               "claim": "a claim", "setup": "a setup", "numbers": {"a": 1},
               "evidence": [{"claim": "a claim", "metric": "m", "value": 0.5, "control_value": None,
                             "split": "MMLU held-out", "seeds": [0], "source": "results/x.json"}]}
    path = tmp_path / "finding.json"
    path.write_text(json.dumps(finding))
    lib.add(library, str(path))
    entries = entries_by_id(library)
    assert entries["f_hand_added"][1] == lib.FINDINGS_FILE
    assert lib.validate(library) == []


def test_add_append_evidence_keeps_old_items(library, tmp_path):
    entries = entries_by_id(library)
    old, rel = entries["mlp_cluster_phi4_l17_gsm8k"]
    count = len(old["evidence"])
    extra = {"id": "mlp_cluster_phi4_l17_gsm8k",
             "evidence": [{"claim": "graft vs random", "metric": "GSM8K accuracy", "value": 0.33, "control_value": 0.31,
                           "control": "random same-size cluster, same layer",
                           "split": "GSM8K test, 1319 items (held-out)", "seeds": [0, 1, 2],
                           "source": "results/new.json", "added_by": "agent: tester"}],
             "status": "validated", "status_reason": "beats random held out"}
    path = tmp_path / "evidence.json"
    path.write_text(json.dumps(extra))
    lib.add(library, str(path), append_evidence=True)
    data = read(os.path.join(library, rel))
    assert len(data["evidence"]) == count + 1
    assert data["evidence"][:count] == old["evidence"]
    assert data["status"] == "validated"
    assert lib.validate(library) == []
    # The next build keeps the appended item and the status.
    lib.build(REPO, library, quiet=True)
    data = read(os.path.join(library, rel))
    assert data["evidence"][-1]["source"] == "results/new.json"
    assert data["status"] == "validated"


def test_query_filters(built):
    rows = lib.query(built, kind="deletion_mask")
    assert len(rows) == 37
    rows = lib.query(built, kind="mlp_cluster", status="rejected", skill="gsm8k")
    assert len(rows) > 0
    for row in rows:
        assert row["kind"] == "mlp_cluster"
        assert row["status"] == "rejected"
        assert "gsm8k" in row["skills"]
    rows = lib.query(built, kind="model", model="qwen2.5-7b-instruct")
    ids = []
    for row in rows:
        ids.append(row["id"])
    assert "Qwen__Qwen2.5-7B-Instruct" in ids
    assert lib.query(built, kind="finding", skill="no-such-skill") == []


def test_query_command_prints_table(built, capsys):
    assert lib.main(["query", "--library", built, "--kind", "block", "--skill", "knowledge"]) == 0
    out = capsys.readouterr().out
    assert "block_qwen25_7b_l16_20_knowledge" in out
    assert "entries" in out


def test_recipe_text_parses_back(built):
    try:
        from wsp.lang import parse, program_from_dict
    except Exception:
        pytest.skip("wsp.lang is not importable")
    checked = 0
    for data, rel in lib.load_entries(built):
        if data["kind"] != "recipe" or not data.get("text"):
            continue
        assert parse(data["text"]).to_dict() == program_from_dict(data["program"]).to_dict()
        assert os.path.exists(os.path.join(built, data["program_file"]))
        checked += 1
    assert checked >= 10


# Missing part files


def mini_repo(tmp_path, with_file):
    """A repo with one parts_extracted record. The part file is present only when with_file is true."""
    repo = tmp_path / "mini_repo"
    folder = repo / "parts_extracted" / "Qwen__Qwen2.5-Math-1.5B"
    folder.mkdir(parents=True, exist_ok=True)
    source = os.path.join(REPO, "parts_extracted", "Qwen__Qwen2.5-Math-1.5B", "records.jsonl")
    with open(source) as f:
        record = json.loads(f.readline())
    record["controls"] = []
    with open(folder / "records.jsonl", "w") as f:
        f.write(json.dumps(record) + "\n")
    units = repo / record["units"]
    if with_file:
        units.write_bytes(b"units-bytes")
    elif units.exists():
        units.unlink()
    return str(repo), record


def mini_build(repo, library, capsys=None):
    report = lib.build(repo, library, quiet=capsys is None)
    out = ""
    if capsys is not None:
        out = capsys.readouterr().out
    return report, out


def test_build_keeps_units_file_when_part_file_is_absent(tmp_path, capsys):
    library = str(tmp_path / "library")
    repo, record = mini_repo(tmp_path, True)
    mini_build(repo, library)
    path = os.path.join(library, "components", lib.model_key(record["model"]), record["id"] + ".json")
    before = read(path)
    assert before["units_file"].endswith(os.path.basename(record["units"]))
    assert "units_missing" not in before
    repo, record = mini_repo(tmp_path, False)
    report, out = mini_build(repo, library, capsys)
    after = read(path)
    assert after["units_file"] == before["units_file"]
    assert after["units_missing"] is True
    assert after["units_expected"] == record["units"]
    assert after["units_remote"] == before["units_remote"]
    assert os.path.exists(os.path.join(library, after["units_file"]))
    assert len(report["units_missing"]) == 1
    assert record["id"] in report["units_missing"][0]
    assert "part files missing" in out
    assert record["id"] in out


def test_build_without_file_or_old_entry_records_the_flag(tmp_path):
    library = str(tmp_path / "library")
    repo, record = mini_repo(tmp_path, False)
    lib.build(repo, library, quiet=True)
    path = os.path.join(library, "components", lib.model_key(record["model"]), record["id"] + ".json")
    data = read(path)
    assert data["units_file"] is None
    assert data["units_missing"] is True


def test_validate_warns_on_units_missing_without_an_error(tmp_path):
    library = str(tmp_path / "library")
    repo, record = mini_repo(tmp_path, True)
    lib.build(repo, library, quiet=True)
    repo, record = mini_repo(tmp_path, False)
    lib.build(repo, library, quiet=True)
    warnings = []
    errors = lib.validate(library, check_index=False, warnings=warnings)
    assert errors == []
    assert len(warnings) == 1
    assert record["id"] in warnings[0]
    assert "part file missing" in warnings[0]
    assert lib.validate(library, check_index=False) == []


def test_later_build_with_the_file_clears_the_flag(tmp_path):
    library = str(tmp_path / "library")
    repo, record = mini_repo(tmp_path, False)
    lib.build(repo, library, quiet=True)
    path = os.path.join(library, "components", lib.model_key(record["model"]), record["id"] + ".json")
    assert read(path)["units_file"] is None
    repo, record = mini_repo(tmp_path, True)
    report = lib.build(repo, library, quiet=True)
    data = read(path)
    assert data["units_file"].endswith(os.path.basename(record["units"]))
    assert "units_missing" not in data
    assert "units_expected" not in data
    assert report["units_missing"] == []
    warnings = []
    assert lib.validate(library, check_index=False, warnings=warnings) == []
    assert warnings == []
