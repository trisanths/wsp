"""Tests for the lora_grid arm: host LoRA programs at the same eval budget. CPU only, stub backend."""

from wsp.catalog import Catalog
from wsp.swarm import program_schema as ps
from wsp.swarm import space, stub
from wsp.swarm.baselines import METHODS, run_baseline
from wsp.swarm.baselines.lora_grid import grid_programs
from wsp.swarm.config import make_config
from wsp.swarm.log import read_rows

HOST = "unsloth/Llama-3.2-1B-Instruct"


def lora_config(tmp_path, overrides=None):
    values = {"outputs_dir": str(tmp_path), "run_id": "test-lora-grid", "method": "lora_grid", "eval_budget": 24}
    if overrides is not None:
        for key in overrides:
            values[key] = overrides[key]
    return make_config(values)


def stub_catalog():
    return Catalog(stub.StubWorld(0).records())


def test_the_grid_is_every_rank_lr_and_steps_and_all_valid(tmp_path):
    config = lora_config(tmp_path)
    programs = grid_programs(config)
    grid = config["lora_grid"]
    assert len(programs) == len(grid["ranks"]) * len(grid["lrs"]) * len(grid["steps"])
    hashes = set()
    catalog = stub_catalog()
    for program in programs:
        assert space.validate(program, catalog, config) == []
        assert ps.grafts(program) == [] and ps.lora(program)["rank"] in grid["ranks"]
        hashes.add(space.program_hash(program))
    assert len(hashes) == len(programs)


def test_validation_keeps_lora_apart_from_grafts_and_the_glue_recipe(tmp_path):
    config = lora_config(tmp_path)
    catalog = stub_catalog()
    part = space.graftable_parts(catalog)[0]["id"]
    heal = ps.make_heal(steps=300, lr=5e-5)
    lora_program = ps.make_lora_program(HOST, 16, heal)
    mixed = dict(lora_program)
    mixed["grafts"] = [ps.make_graft(part, 6)]
    assert any("no grafts" in error for error in space.validate(mixed, catalog, config))
    untrained = dict(lora_program)
    untrained["heal"] = None
    assert any("needs a heal" in error for error in space.validate(untrained, catalog, config))
    odd_rank = ps.make_lora_program(HOST, 7, heal)
    assert any("LoRA rank 7" in error for error in space.validate(odd_rank, catalog, config))
    # The LoRA learning rates are for LoRA programs only: a graft keeps the shared heal recipe.
    graft_at_lora_lr = ps.make_program(HOST, [ps.make_graft(part, 6)], ps.make_heal(steps=300, lr=5e-5))
    assert any("heal lr" in error for error in space.validate(graft_at_lora_lr, catalog, config))


def test_describe_names_the_lora(tmp_path):
    program = ps.make_lora_program(HOST, 64, ps.make_heal(steps=600, lr=5e-5))
    assert space.describe(program) == "host + LoRA r64 lr 5e-05, heal 600 on gsm8k_train"


def test_the_arm_spends_its_budget_on_distinct_lora_programs(tmp_path):
    assert "lora_grid" in METHODS
    summary = run_baseline(lora_config(tmp_path))
    rows = []
    for row in read_rows(summary["log_path"]):
        if row["type"] == "eval" and row["split"] == "search":
            rows.append(row)
    assert len(rows) == 24 and summary["eval_used"] == 24.0
    hashes = set()
    for row in rows:
        assert row["program"]["grafts"] == [] and row["program"]["lora"] is not None
        hashes.add(row["program_hash"])
    assert len(hashes) == 24
    # Held-out scores the top programs with no controls: a LoRA has no graft to control for.
    assert len(summary["heldout"]) == lora_config(tmp_path)["heldout_top"]
    assert summary["controls"] == []


def test_a_smaller_budget_samples_the_grid(tmp_path):
    summary = run_baseline(lora_config(tmp_path, {"eval_budget": 8, "run_id": "test-lora-grid-8"}))
    assert summary["eval_used"] == 8.0
