"""Tests for scripts/mine_pilot.py: record writer, id and units naming, index file copies.

CPU only, no network. The extractor output is hand-made in the shape
wsp_eval/extraction.py returns (wsp_eval itself needs fcntl, so it is not
imported here).
"""

import importlib.util
import json
import os

import pytest
import torch

from wsp.catalog import Catalog
from wsp.swarm import prompts, space
from wsp.swarm.agents import scripted

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT = os.path.join(REPO, "scripts", "mine_pilot.py")
MINED_PATH = os.path.join(REPO, "data", "pilot_catalog_mined.jsonl")
PILOT_PATH = os.path.join(REPO, "data", "pilot_catalog.jsonl")

R1 = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
QMI = "Qwen/Qwen2.5-Math-1.5B-Instruct"
OLMOE = "allenai/OLMoE-1B-7B-0125-Instruct"


def load_script():
    spec = importlib.util.spec_from_file_location("mine_pilot_under_test", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


mp = load_script()

PERSON_B_KEYS = ["function", "method", "skill_data", "control_data", "n_units", "n_units_in_layer", "unit_effects",
                 "mean_effect", "random_mean_effect", "layer_rank", "extracted"]


def fake_signature(kind, n_units, rank):
    """A signature with the fields wsp_eval.extraction.extract writes."""
    method = "activation contrast: Cohen's d of |activation|"
    if kind == "expert":
        method = "router contrast: log2 ratio of expert pick rates"
    effects = []
    for index in range(n_units):
        effects.append(round(2.0 - 0.01 * index, 4))
    return {"function": "more active on gsm8k text than on generic text", "method": method,
            "skill_data": "gsm8k heal split (train), 256 texts", "control_data": "WikiText-2 train, 256 texts",
            "n_units": n_units, "n_units_in_layer": 8960, "unit_effects": effects, "mean_effect": 1.5,
            "random_mean_effect": 0.02, "layer_rank": rank, "extracted": "2026-11-14T20:00:00+00:00"}


def fake_raw(model, kind, layer, n_units, rank=1, folder="/tmp/extracted"):
    """One record in the shape wsp_eval.extraction.extract returns."""
    short = model.split("/")[-1].lower().replace(".", "").replace("-", "")
    component_id = kind + "_" + short + "_l" + str(layer) + "_gsm8k"
    controls = []
    for seed in range(3):
        controls.append(folder + "/" + component_id + "_control_seed" + str(seed) + ".pt")
    return {"id": component_id, "type": kind, "model": model, "family": "qwen2", "layer": layer,
            "units": folder + "/" + component_id + ".pt", "trained_on": "gsm8k", "controls": controls,
            "signature": fake_signature(kind, n_units, rank)}


def fake_batch():
    """Hand-made extractor output for a small batch: two donors, all three kinds."""
    records = []
    records.append(mp.build_record(fake_raw(R1, "mlp_cluster", 14, 512, 1), "r1q15"))
    records.append(mp.build_record(fake_raw(R1, "mlp_cluster", 9, 512, 2), "r1q15"))
    records.append(mp.build_record(fake_raw(R1, "attn_heads", 20, 2, 1), "r1q15"))
    records.append(mp.build_record(fake_raw(OLMOE, "expert", 8, 4, 1), "olmoe"))
    records.append(mp.build_record(fake_raw(OLMOE, "expert", 3, 4, 2), "olmoe"))
    return records


# ---------------------------------------------------------------- ids and paths


def test_part_id_names_donor_layer_kind_and_size():
    assert mp.part_id("r1q15", "mlp_cluster", 14, 512) == "pilot_r1q15_l14_mlp512"
    assert mp.part_id("qmi15", "attn_heads", 20, 2) == "pilot_qmi15_l20_heads2"
    assert mp.part_id("olmoe", "expert", 8, 4) == "pilot_olmoe_l8_exp4"
    with pytest.raises(ValueError):
        mp.part_id("r1q15", "block", 3, 1)


def test_units_and_control_paths_live_under_units_pilot():
    assert mp.units_path("pilot_r1q15_l14_mlp512") == "units/pilot/pilot_r1q15_l14_mlp512.pt"
    assert mp.control_path("pilot_r1q15_l14_mlp512", 2) == "units/pilot/pilot_r1q15_l14_mlp512_control_seed2.pt"


def test_ids_never_collide_with_the_existing_pilot_catalog():
    existing = mp.existing_ids(PILOT_PATH)
    made = []
    for short in mp.DONOR_ORDER:
        donor = mp.DONORS[short]
        for kind in donor["kinds"]:
            for layer in range(donor["layers"]):
                pid = mp.part_id(short, kind, layer, mp.UNIT_SIZES[kind])
                assert pid not in existing
                assert pid not in made
                made.append(pid)


# ---------------------------------------------------------------- record writer


def test_build_record_has_the_brief_shape_plus_pilot_fields():
    raw = fake_raw(R1, "mlp_cluster", 14, 512, 1)
    record = mp.build_record(raw, "r1q15")
    expected = ["controls", "id", "layer", "model", "pilot", "signature", "source", "trained_on", "type", "units"]
    keys = sorted(record.keys())
    assert keys == expected
    assert record["id"] == "pilot_r1q15_l14_mlp512"
    assert record["type"] == "mlp_cluster"
    assert record["model"] == R1
    assert record["layer"] == 14
    assert record["units"] == "units/pilot/pilot_r1q15_l14_mlp512.pt"
    assert record["pilot"] is True
    assert record["source"] == "wsp_eval extractor (Cohen d / router log2 ratio); pilot set, not the Person A catalog"
    assert record["controls"] == ["units/pilot/pilot_r1q15_l14_mlp512_control_seed0.pt",
                                  "units/pilot/pilot_r1q15_l14_mlp512_control_seed1.pt",
                                  "units/pilot/pilot_r1q15_l14_mlp512_control_seed2.pt"]


def test_signature_keeps_person_b_fields_and_adds_empty_effect():
    raw = fake_raw(OLMOE, "expert", 8, 4, 3)
    record = mp.build_record(raw, "olmoe")
    signature = record["signature"]
    for key in PERSON_B_KEYS:
        assert signature[key] == raw["signature"][key]
    assert signature["effect"] == {}
    assert signature["fit_r2"] == {}
    assert signature["extractor_id"] == "expert_olmoe1b7b0125instruct_l8_gsm8k"
    # The builder copies; it does not change the extractor's dict.
    assert "effect" not in raw["signature"]


def test_cohens_d_never_reaches_the_swarm_score_hint():
    """signature["effect"] is read as a task-score gain, so extractor statistics must stay out of it."""
    record = mp.build_record(fake_raw(R1, "mlp_cluster", 14, 512, 1), "r1q15")
    assert scripted.signature_prior(record, "gsm8k", {"gsm8k": 0.25}) == 0.0
    assert prompts.signature_text(record) == "none"


def test_trained_on_names_the_method_and_both_text_sets():
    mlp = mp.build_record(fake_raw(R1, "mlp_cluster", 14, 512, 1), "r1q15")
    heads = mp.build_record(fake_raw(R1, "attn_heads", 20, 2, 2), "r1q15")
    expert = mp.build_record(fake_raw(OLMOE, "expert", 8, 4, 1), "olmoe")
    assert mlp["trained_on"].startswith("top 512 MLP neurons by Cohen's d")
    assert heads["trained_on"].startswith("top 2 query heads by Cohen's d")
    assert expert["trained_on"].startswith("top 4 experts by log2 ratio")
    for record in [mlp, heads, expert]:
        assert "gsm8k heal split" in record["trained_on"]
        assert "WikiText-2" in record["trained_on"]
        assert "search split" in record["trained_on"]
    assert "layer rank 2" in heads["trained_on"]


def test_written_catalog_loads_and_resolves_units_on_the_volume(tmp_path):
    path = str(tmp_path / "mined.jsonl")
    records = fake_batch()
    mp.write_records(records, path)
    loaded = mp.read_records(path)
    assert len(loaded) == len(records)
    # Ordered by donor, then kind, then layer.
    ids = []
    for record in loaded:
        ids.append(record["id"])
    assert ids == ["pilot_r1q15_l20_heads2", "pilot_r1q15_l9_mlp512", "pilot_r1q15_l14_mlp512",
                   "pilot_olmoe_l3_exp4", "pilot_olmoe_l8_exp4"]
    # modal_swarm.py loads the pilot catalog with root /vol; assemble.read_units joins root and units.
    catalog = Catalog(loaded, root="/vol")
    record = catalog.get("pilot_r1q15_l14_mlp512")
    resolved = os.path.join(catalog.root, *record["units"].split("/"))
    assert resolved.replace("\\", "/") == "/vol/units/pilot/pilot_r1q15_l14_mlp512.pt"
    assert len(space.graftable_parts(catalog)) == len(records)


def test_a_program_using_a_mined_part_validates(tmp_path):
    catalog = Catalog(fake_batch(), root="/vol")
    program = {"host": "unsloth/Llama-3.2-1B-Instruct", "seed": 0, "masks": None, "task_vectors": [],
               "grafts": [{"id": "pilot_olmoe_l8_exp4", "at": 8, "control": None, "control_seed": 0,
                           "glue_rank": 64}],
               "heal": {"steps": 300, "data": "gsm8k_train", "lr": 0.0003, "batch_size": 8, "seq_len": 384}}
    assert space.validate(program, catalog) == []


def test_merge_replaces_only_the_mined_donor():
    old = fake_batch()
    new = [mp.build_record(fake_raw(OLMOE, "expert", 12, 4, 1), "olmoe")]
    merged = mp.merge_records(old, new, [OLMOE])
    ids = []
    for record in merged:
        ids.append(record["id"])
    assert "pilot_olmoe_l8_exp4" not in ids
    assert "pilot_olmoe_l12_exp4" in ids
    assert "pilot_r1q15_l14_mlp512" in ids
    assert len(merged) == 4


def test_merge_refuses_a_repeated_id():
    old = fake_batch()
    new = [mp.build_record(fake_raw(R1, "mlp_cluster", 14, 512, 1), "r1q15")]
    with pytest.raises(ValueError):
        mp.merge_records(old, new, [OLMOE])


def test_check_ids_finds_repeats_collisions_and_bad_units():
    records = fake_batch()
    assert mp.check_ids(records, ["pilot_qmath15_l6_top256"]) == []
    repeated = records + [records[0]]
    assert len(mp.check_ids(repeated, [])) == 1
    assert len(mp.check_ids(records, ["pilot_olmoe_l8_exp4"])) == 1
    broken = dict(records[0])
    broken["units"] = "units/other.pt"
    assert len(mp.check_ids([broken], [])) == 1


def test_default_probe_parts_pick_one_rank_one_part_per_kind():
    chosen = mp.default_probe_parts(fake_batch())
    assert chosen == ["pilot_r1q15_l20_heads2", "pilot_r1q15_l14_mlp512", "pilot_olmoe_l8_exp4"]


def test_estimate_cost_uses_gpu_cpu_and_memory_rates():
    # One hour on an L4 with 4 cores and 32 GiB: 3600 * (0.000222 + 4 * 0.0000131 + 32 * 0.00000222).
    assert mp.estimate_cost(3600, "L4") == 1.2436
    assert mp.estimate_cost(0, "L4") == 0.0


# ---------------------------------------------------------------- index file copies


def save_person_b_files(folder, raw):
    """Write the part and its controls as wsp_eval.units.save_index_file and random_control do."""
    indices = torch.tensor([3, 7, 11, 19], dtype=torch.long)
    torch.save({"kind": raw["type"], "model": raw["model"], "layer": raw["layer"], "indices": indices,
                "signature": raw["signature"]}, raw["units"])
    for seed in range(3):
        picked = torch.tensor([seed, 20 + seed, 40 + seed, 60 + seed], dtype=torch.long)
        torch.save({"kind": raw["type"], "model": raw["model"], "layer": raw["layer"], "indices": picked,
                    "signature": {"control": "random same-size set from the same layer", "seed": seed},
                    "control_of": raw["units"], "seed": seed}, raw["controls"][seed])


def test_copy_index_files_writes_parts_and_controls_under_units_pilot(tmp_path):
    extracted = tmp_path / "extracted"
    extracted.mkdir()
    vol = tmp_path / "vol"
    raw = fake_raw(OLMOE, "expert", 8, 4, 1, folder=str(extracted).replace("\\", "/"))
    save_person_b_files(str(extracted), raw)
    record = mp.build_record(raw, "olmoe")

    written = mp.copy_index_files(raw, record, str(vol))
    assert len(written) == 4
    part_file = vol / "units" / "pilot" / "pilot_olmoe_l8_exp4.pt"
    with open(raw["units"], "rb") as handle:
        original = handle.read()
    assert part_file.read_bytes() == original

    # The compiler reads units with torch.load's default (weights_only); that must work.
    part = torch.load(str(part_file))
    assert part["kind"] == "expert"
    assert part["layer"] == 8
    assert part["indices"].tolist() == [3, 7, 11, 19]
    for seed in range(3):
        control_file = vol / "units" / "pilot" / ("pilot_olmoe_l8_exp4_control_seed" + str(seed) + ".pt")
        control = torch.load(str(control_file))
        assert control["control_of"] == "units/pilot/pilot_olmoe_l8_exp4.pt"
        assert control["seed"] == seed
        assert len(control["indices"]) == len(part["indices"])


def test_copy_index_files_refuses_a_file_that_does_not_match_its_record(tmp_path):
    extracted = tmp_path / "extracted"
    extracted.mkdir()
    raw = fake_raw(R1, "mlp_cluster", 14, 4, 1, folder=str(extracted).replace("\\", "/"))
    save_person_b_files(str(extracted), raw)
    record = mp.build_record(raw, "r1q15")
    record["layer"] = 15
    with pytest.raises(ValueError):
        mp.copy_index_files(raw, record, str(tmp_path / "vol"))


# ---------------------------------------------------------------- the mined catalog in the repo


def test_mined_catalog_file_is_well_formed():
    if not os.path.exists(MINED_PATH):
        pytest.skip("data/pilot_catalog_mined.jsonl not built yet (scripts/mine_pilot.py mine)")
    records = mp.read_records(MINED_PATH)
    assert len(records) > 0
    assert mp.check_ids(records, mp.existing_ids(PILOT_PATH)) == []
    shorts = {}
    for short in mp.DONOR_ORDER:
        shorts[mp.DONORS[short]["model"]] = short
    per_kind = {}
    for record in records:
        assert record["pilot"] is True
        assert record["source"] == mp.SOURCE
        assert record["model"] in shorts
        short = shorts[record["model"]]
        assert record["type"] in mp.DONORS[short]["kinds"]
        signature = record["signature"]
        assert signature["effect"] == {}
        for key in PERSON_B_KEYS:
            assert key in signature
        assert record["id"] == mp.part_id(short, record["type"], record["layer"], signature["n_units"])
        assert len(record["controls"]) == 3
        for seed in range(3):
            assert record["controls"][seed] == mp.control_path(record["id"], seed)
        key = record["model"] + "|" + record["type"]
        if key not in per_kind:
            per_kind[key] = 0
        per_kind[key] += 1
    for key in per_kind:
        assert per_kind[key] <= mp.TOP_LAYERS
    catalog = Catalog(records, root="/vol")
    assert len(space.graftable_parts(catalog)) == len(records)
