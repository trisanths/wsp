"""Core tests for part 3 (wsp/swarm). CPU only, stub backend, no network."""

import json
import os
import random
import re
import threading

import pytest

from wsp.catalog import Catalog
from wsp.program import Program
from wsp.swarm import factory, space, stub
from wsp.swarm import program_schema as ps
from wsp.swarm.agents.scripted import ScriptedAnalyst, ScriptedPlanner
from wsp.swarm.backends import StubCompileEval
from wsp.swarm.baselines import run_baseline
from wsp.swarm.budget import Ledger, LocalBudget
from wsp.swarm.config import DEFAULTS, INTEGRATION_DEFAULTS, REPO_ROOT, load_dotenv, make_config
from wsp.swarm.env import HeldoutRefused, WspEnv
from wsp.swarm.interfaces import LLMNotConfigured, NullLLM, make_memory_listener
from wsp.swarm.log import RunLog, read_rows
from wsp.swarm.scoring import score_with_config
from wsp.swarm.swarm import Swarm, run_swarm

HOST = "unsloth/Llama-3.2-1B-Instruct"
MATH_PART = "stub_qmath_l14_c0"     # donor layer 14 of 28, default host depth 8


def run_config(tmp_path, overrides=None):
    values = {"outputs_dir": str(tmp_path), "run_id": "test-run"}
    if overrides is not None:
        for key in overrides:
            values[key] = overrides[key]
    return make_config(values)


def build(tmp_path, overrides=None):
    return factory.build_run(run_config(tmp_path, overrides))


def stub_catalog():
    return Catalog(stub.StubWorld(0).records())


def math_program(at=8, steps=300):
    heal = None
    if steps is not None:
        heal = ps.make_heal(steps=steps)
    return ps.make_program(HOST, [ps.make_graft(MATH_PART, at)], heal)


def stub_combined(backend, program, config, split="search"):
    compiled = backend.compile(program)
    scores = backend.evaluate(compiled["handle"], split, ["gsm8k", "if_guard"], 0)["scores"]
    value, collapsed = score_with_config(scores, backend.base_scores(split, ["gsm8k", "if_guard"]), config)
    return value


def stub_gsm8k(backend, program):
    """The stub's noisy gsm8k search score for a program (no fit gate)."""
    compiled = backend.compile(program)
    return backend.evaluate(compiled["handle"], "search", ["gsm8k", "if_guard"], 0)["scores"]["gsm8k"]


def expected_gsm8k(backend, program):
    """The stub's noise-free gsm8k search score for a program (the calibration truth)."""
    report = backend.world.graft_report(program, 16, backend.default_depth)
    return backend.world.expected_scores(program, report, "search", ["gsm8k", "if_guard"], ["if_guard"])["gsm8k"]


def part_ids(catalog, kinds):
    """Stub part ids of the given kinds, sorted."""
    found = []
    for record in space.graftable_parts(catalog):
        if record["type"] in kinds:
            found.append(record["id"])
    return found


def one_graft(part_id, at, glue_rank=None, steps=300, control=None, control_seed=0):
    graft = ps.make_graft(part_id, at, control, control_seed, glue_rank)
    return ps.make_program(HOST, [graft], ps.make_heal(steps=steps))


# Schema.

def test_schema_round_trip_and_base_alias():
    program = math_program()
    assert ps.normalize(json.loads(json.dumps(program))) == program
    assert Program.from_dict(program).to_dict() == program
    # The locked schema (wsp/program.py) now writes glue_rank in every graft.
    assert sorted(program["grafts"][0]) == ["at", "control", "control_seed", "glue_rank", "id"]
    assert program["grafts"][0]["glue_rank"] is None
    assert program["heal"] == {"steps": 300, "data": "gsm8k_train", "lr": 3e-4, "batch_size": 8, "seq_len": 384}

    old = {"base": HOST, "seed": 0, "masks": None, "task_vectors": []}
    normal = ps.normalize(old)
    assert normal["host"] == HOST
    assert normal["grafts"] == []
    assert normal["heal"] is None
    assert space.program_hash(old) == space.program_hash({"host": HOST})

    control = ps.as_random_control(program["grafts"][0], 7)
    assert control["control"] == "random"
    assert control["control_seed"] == 7
    with pytest.raises(ValueError):
        ps.normalize({"host": HOST, "heal": {"steps": 300, "made_up_field": 1}})
    with pytest.raises(ValueError):
        ps.normalize({"grafts": []})


def test_validation_errors():
    records = stub.StubWorld(0).records()
    records.append({"id": "mask_x", "type": "deletion_mask", "base": HOST, "path": "x"})
    catalog = Catalog(records)
    config = make_config()
    assert space.validate(math_program(), catalog, config) == []

    def errors_for(program):
        return " | ".join(space.validate(program, catalog, config))

    assert "unknown part id" in errors_for(ps.make_program(HOST, [ps.make_graft("nope", 3)]))
    unhashable = {"host": HOST, "grafts": [{"id": [MATH_PART], "at": 8}], "heal": None}
    assert "graft id must be a string" in errors_for(unhashable)
    assert "cannot be grafted" in errors_for(ps.make_program(HOST, [ps.make_graft("mask_x", 3)]))
    assert "outside 0..15" in errors_for(math_program(at=16))
    assert "must be an integer" in errors_for(math_program(at="3"))
    five = []
    for at in range(5):
        five.append(ps.make_graft(MATH_PART, at + 6))
    assert "at most 4 grafts" in errors_for(ps.make_program(HOST, five))
    twice = [ps.make_graft(MATH_PART, 8), ps.make_graft(MATH_PART, 8)]
    assert "appears twice" in errors_for(ps.make_program(HOST, twice))
    assert "heal steps 50" in errors_for(math_program(steps=50))
    bad_data = ps.make_program(HOST, [ps.make_graft(MATH_PART, 8)], ps.make_heal(data="wikitext"))
    assert "heal data wikitext" in errors_for(bad_data)
    assert "not this run's host" in errors_for(ps.make_program("other/model", [ps.make_graft(MATH_PART, 8)]))
    masked = {"host": HOST, "masks": {"op": "use", "id": "missing_mask"}}
    assert "unknown mask id" in errors_for(masked)
    assert "cannot read program" in errors_for({"host": HOST, "masks": {"op": "bogus"}})


def test_glue_rank_survives_every_graft_builder():
    graft = ps.make_graft(MATH_PART, 8, glue_rank=64)
    program = ps.make_program(HOST, [graft], ps.make_heal())
    assert program["grafts"][0]["glue_rank"] == 64
    assert ps.normalize(json.loads(json.dumps(program))) == program
    assert Program.from_dict(program).to_dict() == program
    assert ps.graft_glue_rank(ps.graft_with_at(graft, 6)) == 64
    assert ps.graft_glue_rank(ps.graft_with_id(graft, "stub_qmath_l10_c0")) == 64
    assert ps.graft_glue_rank(ps.as_random_control(graft, 3)) == 64
    assert ps.graft_glue_rank(ps.graft_with_glue_rank(graft, None)) is None
    assert space.program_hash(program) != space.program_hash(math_program())

    control = space.random_glue_control(program, random.Random(0))
    assert control["grafts"][0]["control"] == "random" and control["grafts"][0]["glue_rank"] == 64
    assert space.describe(program) == "host + qmath.l14.c0@8/r64, heal 300 on gsm8k_train"
    assert space.describe(math_program()) == "host + qmath.l14.c0@8, heal 300 on gsm8k_train"
    assert "rand(qmath.l14.c0)@8/r64" in space.describe(control)


def test_glue_rank_validation():
    catalog = stub_catalog()
    config = make_config()

    def errors_for(rank, run_config=config):
        program = ps.make_program(HOST, [ps.make_graft(MATH_PART, 8, glue_rank=rank)], ps.make_heal())
        return " | ".join(space.validate(program, catalog, run_config))

    assert space.glue_ranks(config) == [None, 64]
    assert errors_for(None) == ""
    assert errors_for(64) == ""
    assert "glue_rank 32 not in [None, 64]" in errors_for(32)
    assert "glue_rank must be None or an integer" in errors_for("64")
    assert "glue_rank must be None or an integer" in errors_for(64.0)
    assert "glue_rank must be None or an integer" in errors_for(True)
    low_only = make_config({"glue_ranks": [64]})
    assert "glue_rank None not in [64]" in errors_for(None, low_only)
    assert errors_for(64, low_only) == ""
    wider = make_config({"glue_ranks": [None, 16, 64]})
    assert errors_for(16, wider) == ""


def test_hash_stability():
    program = math_program()
    reordered = {}
    for key in reversed(list(program.keys())):
        reordered[key] = program[key]
    assert space.program_hash(program) == space.program_hash(reordered)
    as_base = dict(program)
    as_base["base"] = as_base.pop("host")
    assert space.program_hash(as_base) == space.program_hash(program)
    assert len(space.program_hash(program)) == 16
    # Changed from a144bb43aff3c2dd when the locked schema added glue_rank (None here) to every graft.
    assert space.program_hash(program) == "f5b9686e11c36953"
    assert space.program_hash(math_program(steps=300)) == space.program_hash(math_program(steps=300))
    assert space.program_hash(math_program(steps=300)) != space.program_hash(math_program(steps=100))
    assert space.probe_hash(math_program(steps=300)) == space.probe_hash(math_program(steps=100))


def test_mutate_and_crossover_keep_programs_valid():
    catalog = stub_catalog()
    config = make_config()
    rng = random.Random(0)
    program = space.random_program(rng, catalog, config)
    other = space.random_program(rng, catalog, config)
    for step in range(200):
        assert space.validate(program, catalog, config) == []
        child = space.mutate(program, rng, catalog, config)
        assert space.validate(child, catalog, config) == []
        assert space.program_hash(child) != space.program_hash(program)
        mixed = space.crossover(child, other, rng, config)
        assert space.validate(mixed, catalog, config) == []
        other = program
        program = child


def test_glue_rank_move_and_rates():
    names = []
    total = 0.0
    for name, rate in space.MOVE_RATES:
        names.append(name)
        total += rate
    assert names == ["add", "drop", "swap", "move", "heal", "rank"]
    assert abs(total - 1.0) < 1e-9
    assert dict(space.MOVE_RATES)["rank"] == 0.10
    rng = random.Random(0)
    rank_picks = 0
    for draw in range(20000):
        if space.pick_move(rng) == "rank":
            rank_picks += 1
    assert 0.09 < rank_picks / 20000.0 < 0.11

    catalog = stub_catalog()
    config = make_config()
    parent = ps.make_program(HOST, [ps.make_graft(MATH_PART, 8), ps.make_graft("stub_qmath_l10_c0", 6, glue_rank=64)],
                             ps.make_heal())
    rank_children = 0
    for attempt in range(300):
        child, note = space.mutate_with_note(parent, rng, catalog, config)
        if not note.startswith("glue rank of "):
            continue
        rank_children += 1
        assert child["heal"] == parent["heal"]
        changed = 0
        for index in range(2):
            old = parent["grafts"][index]
            new = child["grafts"][index]
            assert new["id"] == old["id"] and new["at"] == old["at"]
            if new["glue_rank"] != old["glue_rank"]:
                changed += 1
        assert changed == 1
    assert rank_children > 10

    seen = []
    for draw in range(40):
        for graft in space.random_program(rng, catalog, config)["grafts"]:
            if graft["glue_rank"] not in seen:
                seen.append(graft["glue_rank"])
    assert len(seen) == 2 and None in seen and 64 in seen

    low = one_graft(MATH_PART, 8, glue_rank=64)
    swapped = None
    while swapped is None:
        swapped, note = space.apply_move("swap", low, rng, catalog, config)
    assert swapped["grafts"][0]["id"] != MATH_PART and swapped["grafts"][0]["glue_rank"] == 64
    full_only = make_config({"glue_ranks": [None]})
    assert space.apply_move("rank", math_program(), rng, catalog, full_only) == (None, "")


def test_crossover_keeps_each_grafts_glue_rank():
    config = make_config()
    a = ps.make_program(HOST, [ps.make_graft(MATH_PART, 8, glue_rank=64), ps.make_graft("stub_qmath_l10_c0", 6, glue_rank=64)],
                        ps.make_heal())
    b = ps.make_program(HOST, [ps.make_graft("stub_qcoder_l8_c0", 5), ps.make_graft("stub_llama3b_l14_c0", 8)],
                        ps.make_heal(steps=100))
    source = {}
    for graft in a["grafts"] + b["grafts"]:
        source[(graft["id"], graft["at"])] = graft["glue_rank"]
    rng = random.Random(0)
    mixed_ranks = 0
    for attempt in range(50):
        child = space.crossover(a, b, rng, config)
        ranks = []
        for graft in child["grafts"]:
            assert graft["glue_rank"] == source[(graft["id"], graft["at"])]
            if graft["glue_rank"] not in ranks:
                ranks.append(graft["glue_rank"])
        if len(ranks) == 2:
            mixed_ranks += 1
    assert mixed_ranks > 0


# Budget.

def test_budget_threads_cannot_overspend_either_axis(tmp_path):
    ledger = Ledger(str(tmp_path / "ledger.jsonl"))
    budget = LocalBudget(50, 1000.0, "r", "swarm", "search", ledger)
    tokens = []
    tokens_lock = threading.Lock()

    def spend_many():
        for attempt in range(10):
            token = budget.spend_eval("t")
            if token is not None:
                with tokens_lock:
                    tokens.append(token)

    threads = []
    for index in range(16):
        threads.append(threading.Thread(target=spend_many))
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert sorted(tokens) == list(range(50))
    assert budget.eval_remaining() == 0
    assert budget.exhausted()
    assert budget.spend_eval("late") is None

    compute = LocalBudget(1000, 100.0, "r", "swarm", "search", ledger)

    def burn():
        while not compute.exhausted():
            compute.charge_compute(7.0, "t", "eval")

    threads = []
    for index in range(8):
        threads.append(threading.Thread(target=burn))
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert compute.exhausted()
    assert 100.0 <= compute.compute_used() < 100.0 + 8 * 7.0

    fractional = LocalBudget(1.0, 100.0)
    assert fractional.spend_eval("a", 0.34) == 0
    assert fractional.spend_eval("a", 0.34) == 1
    assert fractional.spend_eval("a", 0.34) is None
    assert abs(fractional.eval_remaining() - 0.32) < 1e-9

    rows = ledger.read()
    eval_rows = 0
    for row in rows:
        if row["axis"] == "eval":
            eval_rows += 1
    assert eval_rows == 50
    assert len(rows) == 50 + len(compute.ledger_rows())


# Env step statuses.

class RaisingEval(StubCompileEval):
    def evaluate(self, handle, split, tasks, token, fidelity="full"):
        raise RuntimeError("GPU went away")


def test_each_step_status(tmp_path):
    run = build(tmp_path)
    env = run["env"]
    budget = run["budget"]

    obs, reward, terminated, truncated, row = env.step(ps.make_program(HOST, [ps.make_graft("nope", 3)]), "t", "bad id")
    assert row["status"] == "invalid" and reward is None and row["token"] is None
    assert budget.compute_used() == 0 and budget.eval_used() == 0

    obs, reward, terminated, truncated, row = env.step(math_program(), "t", "math part at home")
    assert row["status"] == "ok" and row["token"] == 0 and reward == row["combined"]
    assert row["fit_r2"][MATH_PART + "@8"] > 0.3 and row["fit"][MATH_PART + "@8"] == "ridge"
    assert row["backend"] == "stub" and row["stub"] is True
    # Stub compile is 10 s + 15 s per graft: 25 s for one graft, the round-2 median (was 35 s).
    assert row["compute_s"] == {"compile": 25.0, "heal": 150.0, "eval": 120.0}
    assert terminated is False and truncated is False
    used = budget.compute_used()

    obs, again, terminated, truncated, row = env.step(math_program(), "t", "same again")
    assert row["status"] == "cache_hit" and row["cache_hit"] is True and again == reward
    assert budget.eval_used() == 1 and budget.compute_used() == used

    obs, reward, terminated, truncated, row = env.step(math_program(at=0), "t", "far from home")
    assert row["status"] == "rejected_fit" and row["token"] is None and reward is None
    assert budget.eval_used() == 1 and budget.compute_used() == used + 25.0

    weak = []
    for part_id, at in [("stub_qmath_l6_c0", 2), ("stub_qmath_l10_c0", 5), ("stub_qmath_l14_c0", 8),
                        ("stub_qmath_l20_c0", 11)]:
        weak.append(ps.make_graft(part_id, at))
    obs, reward, terminated, truncated, row = env.step(ps.make_program(HOST, weak, ps.make_heal(steps=0)), "t", "4 unhealed")
    assert row["status"] == "collapsed" and row["token"] == 1 and reward is not None

    rows = read_rows(env.log.path)
    statuses = []
    for logged in rows:
        if logged["type"] == "eval":
            statuses.append(logged["status"])
    assert statuses == ["invalid", "ok", "cache_hit", "rejected_fit", "collapsed"]


def test_compile_error_infra_error_and_budget_exhausted(tmp_path):
    config = run_config(tmp_path, {"eval_budget": 2})
    records = stub.StubWorld(0).records()
    records.append({"id": "foreign_l14_c0", "type": "mlp_cluster", "model": "Qwen/Qwen2.5-Math-1.5B", "layer": 14,
                    "units": "x.pt", "trained_on": "x", "signature": {"effect": {}, "fit_r2": {}}})
    catalog = Catalog(records)
    budget = LocalBudget(2, 10000.0)
    log = RunLog(str(tmp_path / "runs" / "manual"))
    env = WspEnv(config, catalog, RaisingEval(config, catalog), budget, log, LocalBudget(9, 10000.0))

    row = env.step(ps.make_program(HOST, [ps.make_graft("foreign_l14_c0", 8)], ps.make_heal()), "t", "x")[4]
    assert row["status"] == "compile_error" and row["token"] is None
    assert budget.eval_used() == 0 and budget.compute_used() == 25.0

    row = env.step(math_program(), "t", "eval fails")[4]
    assert row["status"] == "infra_error" and row["token"] == 0
    assert budget.eval_used() == 1

    env.backend = StubCompileEval(config, catalog)
    assert env.step(math_program(steps=100), "t", "ok")[4]["status"] == "ok"
    row = env.step(math_program(steps=600), "t", "no tokens left")[4]
    assert row["status"] == "budget_exhausted"
    assert env.search_done()


def test_step_many_keeps_input_order_and_dedupes(tmp_path):
    run = build(tmp_path)
    env = run["env"]
    candidates = []
    for steps in [300, 100, 300, 600]:
        candidates.append({"program": math_program(steps=steps), "agent": "p", "hypothesis": str(steps)})
    rows = env.step_many(candidates)
    hypotheses = []
    for row in rows:
        hypotheses.append(row["hypothesis"])
    assert hypotheses == ["300", "100", "300", "600"]
    assert rows[2]["status"] == "cache_hit" and rows[2]["token"] is None
    assert rows[0]["token"] == 0 and rows[1]["token"] == 1 and rows[3]["token"] == 2
    assert run["budget"].eval_used() == 3


def test_probe_charges_compute_and_no_token(tmp_path):
    run = build(tmp_path)
    env = run["env"]
    budget = run["budget"]
    result = env.probe(math_program(), "planner-1")
    assert result["status"] == "ok"
    assert result["fit_r2"][MATH_PART + "@8"] > 0.3
    assert result["compute_s"] == 25.0
    assert budget.eval_used() == 0 and budget.compute_used() == 25.0
    again = env.probe(math_program(steps=600), "planner-2")
    assert again["status"] == "cache_hit" and budget.compute_used() == 25.0
    rows = read_rows(env.log.path)
    probe_rows = []
    for row in rows:
        if row["type"] == "probe":
            probe_rows.append(row)
    assert len(probe_rows) == 2 and probe_rows[0]["program"]["heal"] is None
    assert probe_rows[0]["compute_s"] == {"probe": 25.0} and probe_rows[0]["token"] is None


def test_heldout_refused_before_search_budget_is_spent(tmp_path):
    run = build(tmp_path)
    env = run["env"]
    env.step(math_program(), "t", "x")
    with pytest.raises(HeldoutRefused):
        env.heldout_eval(math_program(), "judge")
    with pytest.raises(HeldoutRefused):
        env.run_controls(math_program(), "judge")
    assert run["heldout_budget"].eval_used() == 0

    env.end_search("test")
    row = env.heldout_eval(math_program(), "judge")
    assert row["split"] == "heldout" and row["status"] == "ok"
    with pytest.raises(HeldoutRefused):
        env.heldout_eval(math_program(), "judge")
    controls = env.run_controls(math_program(), "judge")
    kinds = []
    for control in controls:
        assert control["status"] == "control" and control["split"] == "heldout"
        kinds.append(control["extra"]["control"])
    assert kinds == ["random", "no_graft"]
    assert controls[0]["program"]["grafts"][0]["control"] == "random"
    assert controls[0]["fit_r2"][MATH_PART + "@8~random"] == row["fit_r2"][MATH_PART + "@8"]
    assert controls[1]["program"]["grafts"] == []
    assert run["heldout_budget"].eval_used() == 3
    assert run["budget"].eval_used() == 1
    assert env.step(math_program(steps=100), "t", "after close")[4]["status"] == "budget_exhausted"
    for history_row in env.history:
        assert history_row["status"] != "control"


# Stub.

def test_stub_is_deterministic(tmp_path):
    config = make_config()
    catalog = stub_catalog()
    first = StubCompileEval(config, catalog)
    second = StubCompileEval(config, stub_catalog())
    program = ps.make_program(HOST, [ps.make_graft(MATH_PART, 8), ps.make_graft("stub_qcoder_l20_c0", 12)],
                              ps.make_heal(steps=300))
    assert stub_combined(first, program, config) == stub_combined(second, program, config)
    assert first.compile(program)["graft_report"] == second.compile(program)["graft_report"]
    assert stub.StubWorld(0).records() == stub.StubWorld(0).records()
    path = stub.write_stub_catalog(str(tmp_path / "stub" / "catalog.jsonl"), 0)
    assert Catalog.load(path).records == catalog.records
    for record in catalog.records.values():
        assert record["stub"] is True
        assert sorted(record) == sorted(["id", "type", "model", "layer", "units", "trained_on", "signature", "stub"]) \
            or sorted(record) == sorted(["id", "type", "model", "layers", "units", "trained_on", "signature", "stub"])


def depth_eight(part_id):
    return 8


def test_stub_fit_depends_only_on_the_slot():
    world = stub.StubWorld(0)
    three = []
    for tag in ["c0", "c1", "c2"]:
        program = ps.make_program(HOST, [ps.make_graft("stub_qmath_l14_" + tag, 9)])
        three.append(world.graft_report(program, 16, depth_eight)[0]["fit_r2"])
    assert three[0] == three[1] == three[2]
    control = ps.make_program(HOST, [ps.make_graft("stub_qmath_l14_c0", 9, "random", 5)])
    assert world.graft_report(control, 16, depth_eight)[0]["fit_r2"] == three[0]
    home = world.fit_r2("Qwen/Qwen2.5-Math-1.5B", 14, 8, 16)
    far = world.fit_r2("Qwen/Qwen2.5-Math-1.5B", 14, 2, 16)
    assert home > 0.6 and far < 0.3


def test_stub_has_structure():
    config = make_config()
    catalog = stub_catalog()
    backend = StubCompileEval(config, catalog)
    rng = random.Random(0)
    best = None
    for index in range(30):
        value = stub_combined(backend, space.random_program(rng, catalog, config), config)
        if best is None or value > best:
            best = value
    assert best > 0.0

    healed = stub_combined(backend, math_program(steps=300), config)
    unhealed = stub_combined(backend, math_program(steps=0), config)
    assert healed > unhealed
    assert stub_combined(backend, math_program(steps=None), config) == 0.0
    # The part-beats-control check moved: the calibrated stub gives single-layer parts no content
    # over their random control (round 2), see test_stub_single_layer_parts_match_their_random_control.


# Stub calibration to Trisanth's round-2 sweep (results/README.md, GSM8K search split).

def test_stub_placement_dominates():
    """Round 2: host layer 6 gave 38 to 40%, host layers 10 and 14 gave 22 to 26%, whatever the part."""
    config = make_config()
    catalog = stub_catalog()
    backend = StubCompileEval(config, catalog)
    mlp_parts = part_ids(catalog, ["mlp_cluster"])
    by_depth = {}
    best_at = None
    for at in range(16):
        total = 0.0
        for part_id in mlp_parts:
            total += expected_gsm8k(backend, one_graft(part_id, at, glue_rank=64))
        by_depth[at] = total / len(mlp_parts)
        if best_at is None or by_depth[at] > by_depth[best_at]:
            best_at = at
    assert best_at == 6
    assert abs(by_depth[6] - 0.39) < 0.02
    assert abs(by_depth[10] - 0.25) < 0.02 and abs(by_depth[14] - 0.25) < 0.02
    gap = by_depth[6] - (by_depth[10] + by_depth[14]) / 2.0
    assert 0.11 < gap < 0.17

    # With noise, on cells shaped like round 2's: a donor part at its default host depth 6, 10 or 14
    # (the stub has no qmath layer 24, so depth 14 uses qcoder layer 24), part and random control,
    # full and rank-64 glue, 3 seeds.
    cells = {6: "stub_qmath_l10_c0", 10: "stub_qmath_l18_c0", 14: "stub_qcoder_l24_c0"}
    means = {}
    for at in sorted(cells):
        scores = []
        for seed in range(3):
            for rank in [None, 64]:
                for control in [None, "random"]:
                    graft = ps.make_graft(cells[at], None, control, seed + 100, rank)
                    program = ps.make_program(HOST, [graft], ps.make_heal(steps=300), seed=seed)
                    assert backend.compile(program)["graft_report"][0]["host_layer"] == at
                    scores.append(stub_gsm8k(backend, program))
        means[at] = sum(scores) / len(scores)
    assert means[6] - means[10] > 0.08 and means[6] - means[14] > 0.08


def test_stub_single_layer_parts_match_their_random_control():
    """Round 2: no part selection method beat the random control (pooled per-cell sd 2.3 points)."""
    config = make_config()
    catalog = stub_catalog()
    backend = StubCompileEval(config, catalog)
    total = 0.0
    count = 0
    for part_id in part_ids(catalog, ["mlp_cluster", "attn_heads", "expert"]):
        for rank in [None, 64]:
            part = expected_gsm8k(backend, one_graft(part_id, None, glue_rank=rank))
            control = expected_gsm8k(backend, one_graft(part_id, None, glue_rank=rank, control="random",
                                                        control_seed=7))
            assert abs(part - control) < 0.015
            total += part - control
            count += 1
    assert abs(total / count) < 0.005

    noisy_total = 0.0
    noisy_count = 0
    for part_id in part_ids(catalog, ["mlp_cluster"]):
        for seed in range(3):
            chosen = ps.make_program(HOST, [ps.make_graft(part_id, None, None, 0, 64)], ps.make_heal(), seed=seed)
            control = ps.make_program(HOST, [ps.make_graft(part_id, None, "random", seed + 100, 64)], ps.make_heal(),
                                      seed=seed)
            noisy_total += stub_gsm8k(backend, chosen) - stub_gsm8k(backend, control)
            noisy_count += 1
    assert abs(noisy_total / noisy_count) < 0.015

    # STUB ASSUMPTION until round 3: math-donor blocks carry a small real content effect.
    for part_id in ["stub_qmath_l10-12_blk", "stub_qmath_l16-18_blk"]:
        block = expected_gsm8k(backend, one_graft(part_id, None, glue_rank=64))
        control = expected_gsm8k(backend, one_graft(part_id, None, glue_rank=64, control="random", control_seed=7))
        assert 0.01 < block - control < 0.05


def test_stub_rank64_glue_matches_or_beats_full_glue():
    """Round 2: rank-64 glue was as good as or better than full glue, at 7% of the parameters."""
    config = make_config()
    catalog = stub_catalog()
    backend = StubCompileEval(config, catalog)
    for part_id in part_ids(catalog, space.GRAFT_TYPES):
        for at in range(16):
            full = expected_gsm8k(backend, one_graft(part_id, at))
            low = expected_gsm8k(backend, one_graft(part_id, at, glue_rank=64))
            assert low >= full - 1e-9
    full_six = expected_gsm8k(backend, one_graft("stub_qmath_l10_c0", 6))
    low_six = expected_gsm8k(backend, one_graft("stub_qmath_l10_c0", 6, glue_rank=64))
    assert 0.02 < low_six - full_six < 0.06       # round 2 at host layer 6: 39.0% vs 34.3%
    full_fourteen = expected_gsm8k(backend, one_graft("stub_qcoder_l24_c0", 14))
    low_fourteen = expected_gsm8k(backend, one_graft("stub_qcoder_l24_c0", 14, glue_rank=64))
    assert abs(low_fourteen - full_fourteen) < 0.005

    full_entry = backend.compile(one_graft("stub_qmath_l10_c0", 6))["graft_report"][0]
    low_entry = backend.compile(one_graft("stub_qmath_l10_c0", 6, glue_rank=64))["graft_report"][0]
    assert full_entry["glue_rank"] is None and full_entry["glue_params"] == 6291456   # the round-2 report
    assert low_entry["glue_rank"] == 64 and low_entry["glue_params"] == 458752
    assert full_entry["fit_r2"] == low_entry["fit_r2"]


def test_stub_host_and_heal_only_gain():
    config = make_config()
    catalog = stub_catalog()
    backend = StubCompileEval(config, catalog)
    host = expected_gsm8k(backend, ps.make_program(HOST, [], None))
    heal_only = expected_gsm8k(backend, ps.make_program(HOST, [], ps.make_heal(steps=300)))
    assert host == 0.25                      # run 1: host, no heal, 25.0%
    assert 0.0 < heal_only - host < 0.05     # STUB ASSUMPTION: round 2 had no heal-only arm
    assert stub_gsm8k(backend, ps.make_program(HOST, [], None)) == 0.25
    assert stub.compile_cost(one_graft(MATH_PART, 8)) == 25.0


# Swarm, factory and helpers.

def test_scripted_swarm_end_to_end(tmp_path):
    config = run_config(tmp_path, {"eval_budget": 30})
    summary = run_swarm(config)
    assert summary["eval_used"] <= 30
    assert summary["stub"] is True and summary["best_search"]["label"] == "SEARCH"
    run_dir = os.path.join(str(tmp_path), "runs", "test-run")
    for name in ["experiments.jsonl", "manifest.json", "summary.json"]:
        assert os.path.exists(os.path.join(run_dir, name))
    assert os.path.exists(os.path.join(str(tmp_path), "ledger.jsonl"))
    rows = read_rows(os.path.join(run_dir, "experiments.jsonl"))
    kinds = set()
    eval_rows = 0
    for row in rows:
        assert row["backend"] == "stub" and row["stub"] is True
        if row["type"] == "event":
            kinds.add(row["kind"])
        if row["type"] == "eval" and row["split"] == "search" and row["status"] == "ok":
            eval_rows += 1
            assert len(row["fit_r2"]) == len(row["program"]["grafts"])
    assert eval_rows > 5
    assert "analyst_map" in kinds and "judge_verdict" in kinds and "search_end" in kinds
    assert 1 <= len(summary["heldout"]) <= 3
    assert len(summary["controls"]) == 2 * len(summary["heldout"])

    again = run_swarm(run_config(tmp_path / "second", {"eval_budget": 30}))
    assert again["best_search"]["combined"] == summary["best_search"]["combined"]


def test_factory_names_missing_modules(monkeypatch):
    # wsp/swarm/memory.py has landed (wave A), so point the choice at a module that cannot exist.
    monkeypatch.setitem(factory.CHOICES["memory"], "past", ("wsp.swarm.not_built_yet", "PastMemory"))
    config = make_config({"memory": "past"})
    with pytest.raises(factory.FactoryError) as error:
        factory.build_memory(config)
    assert "wsp/swarm/not_built_yet.py" in str(error.value) and "PastMemory" in str(error.value)
    with pytest.raises(factory.FactoryError) as error:
        factory.load_class("backend", "cloud")
    assert "choose one of" in str(error.value)
    with pytest.raises(LLMNotConfigured):
        NullLLM().complete([{"role": "user", "content": "hi"}])


def test_dotenv_loader_hides_values(tmp_path, monkeypatch):
    path = tmp_path / ".env"
    path.write_text("# comment\nWSP_TEST_A=one\nexport WSP_TEST_B=\"two words\"\nWSP_TEST_C=three\n\n")
    monkeypatch.delenv("WSP_TEST_A", raising=False)
    monkeypatch.delenv("WSP_TEST_B", raising=False)
    monkeypatch.setenv("WSP_TEST_C", "kept")
    names = load_dotenv(str(path))
    assert names == ["WSP_TEST_A", "WSP_TEST_B", "WSP_TEST_C"]
    assert os.environ["WSP_TEST_A"] == "one"
    assert os.environ["WSP_TEST_B"] == "two words"
    assert os.environ["WSP_TEST_C"] == "kept"
    monkeypatch.delenv("WSP_TEST_A", raising=False)
    monkeypatch.delenv("WSP_TEST_B", raising=False)


def test_memory_listener_records_search_rows_only():
    calls = []

    class FakeMemory:
        def record(self, kind, text, meta=None, round_index=None):
            calls.append((kind, text))

    listener = make_memory_listener(FakeMemory())
    # The pattern is in space.graft_pattern's current format, which now ends in the glue rank.
    base = {"type": "eval", "split": "search", "status": "collapsed", "describe": "host + a@3, heal 0",
            "combined": -0.4, "fit_r2": {"a@3": 0.71}, "scores": {"gsm8k": 0.1}, "token": 4, "agent": "planner-1",
            "round": 2, "hypothesis": "h", "extra": {"patterns": ["a|early|noheal|full"]}}
    listener(base)
    held = dict(base)
    held["split"] = "heldout"
    listener(held)
    kinds = []
    for kind, text in calls:
        kinds.append(kind)
    assert kinds == ["step", "collapse"]
    assert "-0.400" in calls[0][1] and "0.71" in calls[0][1]


# Config defaults.

# MockLLM's scripted replies: only a test sets them, and a mock with neither is a setup error.
TEST_HOOK_KEYS = ["mock_fn", "mock_replies"]


def config_keys_read(folder):
    """Every key that code under folder reads as config["key"] or config.get("key"), with the files that read it."""
    pattern = re.compile(r'(?<![A-Za-z_])config(?:\.get\(|\[)"([a-z_0-9]+)"')
    found = {}
    for root, dirs, files in os.walk(folder):
        for name in sorted(files):
            if not name.endswith(".py"):
                continue
            with open(os.path.join(root, name), encoding="utf-8") as handle:
                text = handle.read()
            for key in pattern.findall(text):
                if key not in found:
                    found[key] = []
                found[key].append(name)
    return found


def test_every_config_key_the_swarm_reads_has_a_default():
    read = config_keys_read(os.path.join(REPO_ROOT, "wsp", "swarm"))
    assert "llm_temperature" in read and "exa_cache_dir" in read and "memory_level" in read
    missing = []
    for key in sorted(read):
        if key in DEFAULTS or key in INTEGRATION_DEFAULTS or key in TEST_HOOK_KEYS:
            continue
        missing.append(key + " (read in " + ", ".join(read[key]) + ")")
    assert missing == []


def test_config_tables():
    config = make_config()
    assert config["glue_ranks"] == [None, 64] and space.glue_ranks(config) == [None, 64]
    assert config["low_fidelity_heal_steps"] == 50
    assert config["local_eval"] == "wsp_eval.harness:swarm_eval"      # Person B's eval hook stays
    assert config["agents"] == "auto" and config["memory_cleanup"] is True
    for key in INTEGRATION_DEFAULTS:
        # Apart from DEFAULTS, so compare.py's --llm-config may still set them for the swarm_llm arm.
        assert key not in DEFAULTS
        assert config[key] == INTEGRATION_DEFAULTS[key]
    changed = make_config({"llm_temperature": 0.2, "glue_ranks": [64], "exa_exclude_domains": ["a.org"]})
    assert changed["llm_temperature"] == 0.2 and changed["glue_ranks"] == [64]
    changed["exa_exclude_domains"].append("b.org")
    assert INTEGRATION_DEFAULTS["llm_temperature"] == 0.7 and INTEGRATION_DEFAULTS["exa_exclude_domains"] == []


def test_integration_defaults_match_each_modules_fallback():
    """A config from make_config builds every client with the same settings as a bare dict."""
    from wsp.swarm.agents import llm_agents
    from wsp.swarm.literature import ExaLiterature
    from wsp.swarm.llm import MockLLM
    from wsp.swarm.memory import PastMemory

    config = make_config()
    assert llm_agents.llm_settings(config, 2, 1) == llm_agents.llm_settings({"seed": 0}, 2, 1)
    assert config["llm_batch_wait_s"] == llm_agents.DEFAULT_BATCH_WAIT_S

    hooks = {"mock_replies": ["{}"]}
    ours = MockLLM(make_config(hooks))
    bare = MockLLM(dict(hooks))
    for name in ["tries", "backoff_s", "timeout_s", "workers"]:
        assert getattr(ours, name) == getattr(bare, name), name

    ours = PastMemory(config)
    bare = PastMemory({})
    for name in ["base_url", "key_env", "default_level", "poll_s", "http_timeout_s", "max_retries", "backoff_s",
                 "membership_timeout_s", "cold_query", "max_chars", "verify_cleanup", "state_path"]:
        assert getattr(ours, name) == getattr(bare, name), name

    ours = ExaLiterature(config, "off")
    bare = ExaLiterature({}, "off")
    for name in ["search_type", "priors_path", "cache_dir", "end_date", "exclude_domains", "quote_chars",
                 "max_query_chars", "max_live_calls", "retries", "backoff_s", "timeout_s", "sdk_version"]:
        assert getattr(ours, name) == getattr(bare, name), name


# Glue rank in exploration and in the judge's patterns.

def explore_draws(tmp_path, overrides, seed, draws=40):
    """Glue ranks and programs from the scripted planner's explore move, with a fresh rng per call."""
    run = build(tmp_path, overrides)
    env = run["env"]
    config = run["config"]
    observation, info = env.reset()
    ctx = {"config": config, "env": env}
    cap_map = ScriptedAnalyst(config).build_map(observation, ctx)
    planner = ScriptedPlanner(2, config)
    rng = random.Random(seed)
    ranks = []
    programs = []
    for draw in range(draws):
        proposal = planner.explore(observation, cap_map, [], rng, env.catalog, config)
        assert space.validate(proposal["program"], env.catalog, config) == []
        rank = proposal["program"]["grafts"][0]["glue_rank"]
        assert space.rank_text(rank) + " glue" in proposal["hypothesis"]
        ranks.append(rank)
        programs.append(proposal["program"])
    return ranks, programs


def test_scripted_explore_samples_the_glue_rank(tmp_path):
    ranks, programs = explore_draws(tmp_path / "a", None, 0)
    assert None in ranks and 64 in ranks
    again, programs_again = explore_draws(tmp_path / "b", None, 0)
    assert again == ranks and programs_again == programs          # one seed, one sequence
    other, programs_other = explore_draws(tmp_path / "c", None, 1)
    assert programs_other != programs
    low_only, programs_low = explore_draws(tmp_path / "d", {"glue_ranks": [64]}, 0)
    assert low_only == [64] * 40


def test_graft_pattern_includes_the_glue_rank(tmp_path):
    catalog = stub_catalog()
    config = make_config()
    assert space.program_patterns(one_graft(MATH_PART, 8), catalog, config) == [MATH_PART + "|middle|heal|full"]
    assert space.program_patterns(one_graft(MATH_PART, 8, glue_rank=64), catalog, config) == \
        [MATH_PART + "|middle|heal|r64"]
    assert space.program_patterns(one_graft(MATH_PART, 8, steps=0), catalog, config) == \
        [MATH_PART + "|middle|noheal|full"]

    # Two collapses kill the full-rank patterns; the rank-64 version of the same grafts stays open.
    run = build(tmp_path)
    env = run["env"]
    config = run["config"]
    places = [("stub_qmath_l6_c0", 2), ("stub_qmath_l10_c0", 5), ("stub_qmath_l14_c0", 8), ("stub_qmath_l20_c0", 11)]

    def weak(seed, rank):
        grafts = []
        for part_id, at in places:
            grafts.append(ps.make_graft(part_id, at, glue_rank=rank))
        return ps.make_program(HOST, grafts, ps.make_heal(steps=0), seed=seed)

    rows = env.step_many([{"program": weak(0, None), "agent": "t", "hypothesis": "4 unhealed"},
                          {"program": weak(1, None), "agent": "t", "hypothesis": "4 unhealed again"}])
    assert rows[0]["status"] == "collapsed" and rows[1]["status"] == "collapsed"
    analyst, planners, judge = factory.build_agents(config)
    swarm = Swarm(config, env, analyst, planners, judge, run["memory"], run["literature"], run["llm"])
    verdict = judge.update([], rows, swarm.judge_ctx)
    full_patterns = space.program_patterns(weak(0, None), env.catalog, config)
    assert sorted(verdict["killed"]) == sorted(full_patterns)
    swarm.killed = verdict["killed"]
    assert swarm.screen("t", {"program": weak(2, None)}, set()) is None
    opened = swarm.screen("t", {"program": weak(0, 64)}, set())
    assert opened is not None and opened["program"]["grafts"][0]["glue_rank"] == 64


# Cascade heal: a short heal below full fidelity, the program's own heal on promotion.

def test_low_fidelity_heals_short_and_a_promotion_heals_in_full(tmp_path):
    run = build(tmp_path)
    env = run["env"]
    budget = run["budget"]
    row = env.step(math_program(steps=300), "t", "low first", fidelity="low")[4]
    assert row["status"] == "ok" and row["fidelity"] == "low" and row["heal_steps_used"] == 50
    assert row["program"]["heal"]["steps"] == 300            # the row logs the program as written
    assert row["compute_s"]["compile"] == 25.0 and row["compute_s"]["heal"] == 25.0   # 0.5 s per step per graft
    assert row["compute_s"]["eval"] == pytest.approx(120.0 * 0.34)
    assert budget.eval_used() == pytest.approx(0.34)

    promoted = env.step(math_program(steps=300), "t", "promote", fidelity="full")[4]
    assert promoted["status"] == "ok" and promoted["heal_steps_used"] == 300
    assert promoted["compute_s"] == {"compile": 25.0, "heal": 150.0, "eval": 120.0}
    # The promotion healed again (300 steps, not 50), so it scored a new model and pays the whole level.
    assert promoted["extra"]["eval_cost"] == 1.0
    assert budget.eval_used() == pytest.approx(1.34)
    assert budget.compute_used() == pytest.approx(25.0 + 25.0 + 40.8 + 25.0 + 150.0 + 120.0)

    again = env.step(math_program(steps=300), "t", "low again", fidelity="low")[4]
    assert again["status"] == "cache_hit" and again["heal_steps_used"] == 50 and again["compute_s"] == {}
    pair = env.step_many([{"program": math_program(steps=100), "agent": "t", "hypothesis": "a", "fidelity": "low"},
                          {"program": math_program(steps=100), "agent": "t", "hypothesis": "b", "fidelity": "low"}])
    assert pair[0]["heal_steps_used"] == 50 and pair[1]["status"] == "cache_hit" and pair[1]["heal_steps_used"] == 50

    # A program asking for fewer steps keeps its own; a program with no heal block heals nothing.
    zero = env.step(math_program(steps=0), "t", "heal 0", fidelity="low")[4]
    assert zero["heal_steps_used"] == 0 and zero["compute_s"]["heal"] == 0.0
    bare = env.step(math_program(steps=None), "t", "no heal", fidelity="low")[4]
    assert bare["heal_steps_used"] is None and "heal" not in bare["compute_s"]

    probe = env.probe(math_program(at=7), "t")
    assert probe["status"] == "ok"
    for logged in read_rows(env.log.path):
        if logged["type"] == "eval":
            assert "heal_steps_used" in logged
        if logged["type"] == "probe":
            assert logged["heal_steps_used"] is None


def fee_costs(env, program, fidelities):
    """Step one program at each fidelity in turn; return each row's eval_cost (None for a free row)."""
    costs = []
    for fidelity in fidelities:
        row = env.step(program, "t", "pricing " + fidelity, fidelity=fidelity)[4]
        assert row["status"] in ("ok", "collapsed", "cache_hit"), row["status"]
        if row["token"] is None or row["cache_hit"]:
            costs.append(None)
        else:
            costs.append(row["extra"]["eval_cost"])
    return costs


def test_promotion_pays_the_difference_only_on_the_same_model(tmp_path):
    run = build(tmp_path / "short")
    env = run["env"]
    budget = run["budget"]
    # 300 steps: low heals 50, full heals 300, a new model, so full pays the whole level.
    assert fee_costs(env, math_program(steps=300), ["low", "full", "low"]) == [0.34, 1.0, None]
    # check and low both heal 50, so low pays the difference; full heals 600 and pays 1.0.
    assert fee_costs(env, math_program(steps=600), ["check", "low", "full"]) == [0.16, 0.18, 1.0]
    # heal 0 and no heal block run the same model at every level: nested price.
    assert fee_costs(env, math_program(steps=0), ["low", "full"]) == [0.34, 0.66]
    assert fee_costs(env, math_program(steps=None), ["low", "full"]) == [0.34, 0.66]
    # Full first, then low: the low eval heals 50, a different model, so it pays its own level.
    assert fee_costs(env, math_program(at=7, steps=300), ["full", "low"]) == [1.0, 0.34]
    paid = (0.34 + 1.0) + (0.16 + 0.18 + 1.0) + (0.34 + 0.66) + (0.34 + 0.66) + (1.0 + 0.34)
    assert budget.eval_used() == pytest.approx(paid)

    # A program at or below the short heal keeps the nested price; one above it does not.
    env = build(tmp_path / "hundred", {"low_fidelity_heal_steps": 100})["env"]
    assert fee_costs(env, math_program(steps=100), ["low", "full"]) == [0.34, 0.66]
    assert fee_costs(env, math_program(steps=300), ["low", "full"]) == [0.34, 1.0]

    # With the short heal off, every level heals the program's own steps: nested price.
    env = build(tmp_path / "none", {"low_fidelity_heal_steps": None})["env"]
    assert fee_costs(env, math_program(steps=300), ["check", "low", "full"]) == [0.16, 0.18, 0.66]


def test_low_fidelity_heal_steps_none_and_bad_values(tmp_path):
    run = build(tmp_path / "none", {"low_fidelity_heal_steps": None})
    row = run["env"].step(math_program(steps=300), "t", "old behavior", fidelity="low")[4]
    assert row["heal_steps_used"] == 300 and row["compute_s"]["heal"] == 150.0
    run = build(tmp_path / "hundred", {"low_fidelity_heal_steps": 100})
    row = run["env"].step(math_program(steps=600), "t", "100-step short heal", fidelity="check")[4]
    assert row["fidelity"] == "check" and row["heal_steps_used"] == 100 and row["compute_s"]["heal"] == 50.0
    for value in [-1, "50", 2.5, True]:
        with pytest.raises(ValueError):
            build(tmp_path / ("bad" + str(value)), {"low_fidelity_heal_steps": value})


def test_stub_short_heal_lands_between_no_heal_and_full_heal():
    config = make_config()
    catalog = stub_catalog()
    backend = StubCompileEval(config, catalog)
    tasks = ["gsm8k", "if_guard"]
    cells = [("stub_qmath_l10_c0", 6, 64), (MATH_PART, 8, None), ("stub_qcoder_l24_c0", 14, 64),
             ("stub_qmath_l10-12_blk", None, 64)]
    for part_id, at, rank in cells:
        program = one_graft(part_id, at, glue_rank=rank)
        report = backend.world.graft_report(program, 16, backend.default_depth)
        values = []
        for steps in [0, 50, 300]:
            healed = ps.with_heal_steps(program, steps)
            values.append(backend.world.expected_scores(healed, report, "search", tasks, ["if_guard"]))
        for task in tasks:
            assert values[0][task] < values[1][task] < values[2][task], (part_id, task)

    # The backend scores the heal it ran. Both heals share the compiled program's noise key,
    # so their gap is the heal's effect alone.
    program = one_graft("stub_qmath_l10_c0", 6, glue_rank=64)
    compiled = backend.compile(program)
    short = backend.heal(compiled["handle"], ps.with_heal_steps(program, 50))
    full = backend.heal(compiled["handle"], program)
    assert short["compute_s"] == {"heal": 25.0} and full["compute_s"] == {"heal": 150.0}
    assert compiled["handle"]["program"]["heal"]["steps"] == 300            # heal() never edits the handle it got
    short_scores = backend.evaluate(short["handle"], "search", tasks, 0, "low")["scores"]
    full_scores = backend.evaluate(full["handle"], "search", tasks, 0, "low")["scores"]
    assert short_scores["gsm8k"] < full_scores["gsm8k"]
    report = compiled["graft_report"]
    expected_short = backend.world.expected_scores(ps.with_heal_steps(program, 50), report, "search", tasks, ["if_guard"])
    expected_full = backend.world.expected_scores(program, report, "search", tasks, ["if_guard"])
    gap = full_scores["gsm8k"] - short_scores["gsm8k"]
    assert abs(gap - (expected_full["gsm8k"] - expected_short["gsm8k"])) < 2e-4
    assert backend.evaluate(compiled["handle"], "search", tasks, 0, "low")["scores"] == full_scores


def heal_rows(summary):
    """Scored search eval rows of a run, split by fidelity: {"low": [...], "full": [...]}."""
    out = {"low": [], "full": []}
    for row in read_rows(summary["log_path"]):
        if row["type"] == "eval" and row["split"] == "search" and row["status"] in ("ok", "collapsed"):
            out[row["fidelity"]].append(row)
    return out


def test_every_arm_heals_short_below_full_and_the_cascade_costs_less_compute(tmp_path):
    for method in ["swarm", "evolve", "random"]:
        config = run_config(tmp_path / method, {"eval_budget": 8, "cascade": True, "method": method,
                                                 "run_id": "cascade-" + method})
        if method == "swarm":
            summary = run_swarm(config)
        else:
            summary = run_baseline(config)
        rows = heal_rows(summary)
        assert len(rows["low"]) > 0 and len(rows["full"]) > 0, method
        for fidelity in ["low", "full"]:
            for row in rows[fidelity]:
                steps = ps.heal_steps(row["program"])
                expected = steps
                if fidelity == "low" and steps is not None and steps > 50:
                    expected = 50
                assert row["heal_steps_used"] == expected, method
                if expected is not None:
                    count = max(1, len(row["program"]["grafts"]))
                    assert row["compute_s"]["heal"] == 0.5 * expected * count, method

    # The same scripted swarm with the old rule (full heal at every fidelity) spends more compute per FEE.
    short = run_swarm(run_config(tmp_path / "short", {"eval_budget": 8, "cascade": True}))
    old = run_swarm(run_config(tmp_path / "old", {"eval_budget": 8, "cascade": True, "low_fidelity_heal_steps": None}))
    assert short["compute_used_s"] / short["eval_used"] < old["compute_used_s"] / old["eval_used"]
