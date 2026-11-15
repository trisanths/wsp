"""Core tests for part 3 (wsp/swarm). CPU only, stub backend, no network."""

import json
import os
import random
import threading

import pytest

from wsp.catalog import Catalog
from wsp.program import Program
from wsp.swarm import factory, space, stub
from wsp.swarm import program_schema as ps
from wsp.swarm.backends import StubCompileEval
from wsp.swarm.budget import Ledger, LocalBudget
from wsp.swarm.config import load_dotenv, make_config
from wsp.swarm.env import HeldoutRefused, WspEnv
from wsp.swarm.interfaces import LLMNotConfigured, NullLLM, make_memory_listener
from wsp.swarm.log import RunLog, read_rows
from wsp.swarm.scoring import score_with_config
from wsp.swarm.swarm import run_swarm

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


# Schema.

def test_schema_round_trip_and_base_alias():
    program = math_program()
    assert ps.normalize(json.loads(json.dumps(program))) == program
    assert Program.from_dict(program).to_dict() == program
    assert sorted(program["grafts"][0]) == ["at", "control", "control_seed", "id"]
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
    assert space.program_hash(program) == "a144bb43aff3c2dd"
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
    assert row["compute_s"] == {"compile": 35.0, "heal": 150.0, "eval": 120.0}
    assert terminated is False and truncated is False
    used = budget.compute_used()

    obs, again, terminated, truncated, row = env.step(math_program(), "t", "same again")
    assert row["status"] == "cache_hit" and row["cache_hit"] is True and again == reward
    assert budget.eval_used() == 1 and budget.compute_used() == used

    obs, reward, terminated, truncated, row = env.step(math_program(at=0), "t", "far from home")
    assert row["status"] == "rejected_fit" and row["token"] is None and reward is None
    assert budget.eval_used() == 1 and budget.compute_used() == used + 35.0

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
    assert budget.eval_used() == 0 and budget.compute_used() == 35.0

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
    assert result["compute_s"] == 35.0
    assert budget.eval_used() == 0 and budget.compute_used() == 35.0
    again = env.probe(math_program(steps=600), "planner-2")
    assert again["status"] == "cache_hit" and budget.compute_used() == 35.0
    rows = read_rows(env.log.path)
    probe_rows = []
    for row in rows:
        if row["type"] == "probe":
            probe_rows.append(row)
    assert len(probe_rows) == 2 and probe_rows[0]["program"]["heal"] is None
    assert probe_rows[0]["compute_s"] == {"probe": 35.0} and probe_rows[0]["token"] is None


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
    control = ps.make_program(HOST, [ps.make_graft(MATH_PART, 8, "random", 3)], ps.make_heal(steps=300))
    assert healed > stub_combined(backend, control, config)


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


def test_factory_names_missing_modules():
    config = make_config({"memory": "past"})
    with pytest.raises(factory.FactoryError) as error:
        factory.build_memory(config)
    assert "wsp/swarm/memory.py" in str(error.value) and "PastMemory" in str(error.value)
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
    base = {"type": "eval", "split": "search", "status": "collapsed", "describe": "host + a@3, heal 0",
            "combined": -0.4, "fit_r2": {"a@3": 0.71}, "scores": {"gsm8k": 0.1}, "token": 4, "agent": "planner-1",
            "round": 2, "hypothesis": "h", "extra": {"patterns": ["a|early|noheal"]}}
    listener(base)
    held = dict(base)
    held["split"] = "heldout"
    listener(held)
    kinds = []
    for kind, text in calls:
        kinds.append(kind)
    assert kinds == ["step", "collapse"]
    assert "-0.400" in calls[0][1] and "0.71" in calls[0][1]
