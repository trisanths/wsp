"""Tests for the baselines (wsp/swarm/baselines). CPU only, stub backend, no network."""

import importlib.util
import math
import os
import random

import pytest

from wsp.swarm import factory, space
from wsp.swarm.baselines import METHODS, TRAINING_METHODS, run_baseline
from wsp.swarm.baselines import training
from wsp.swarm.baselines.evolve import EvolveArm, Population
from wsp.swarm.budget import Ledger
from wsp.swarm.config import make_config
from wsp.swarm.log import read_rows
from wsp.swarm.swarm import run_swarm

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEARCH_METHODS = ["evolve", "random", "random_part", "greedy"]
SUMMARY_KEYS = sorted(["run_id", "method", "backend", "stub", "seed", "rounds", "eval_used", "eval_budget",
                       "compute_used_s", "compute_budget_s", "compute_overshoot_s", "best_search", "heldout",
                       "controls", "log_path", "config"])


def baseline_config(tmp_path, method, overrides=None):
    values = {"outputs_dir": str(tmp_path), "run_id": "test-" + method, "method": method, "eval_budget": 12}
    if overrides is not None:
        for key in overrides:
            values[key] = overrides[key]
    return make_config(values)


def search_rows(summary, split="search"):
    rows = []
    for row in read_rows(summary["log_path"]):
        if row["type"] == "eval" and row["split"] == split:
            rows.append(row)
    return rows


def row_compute(row):
    total = 0.0
    for phase in row["compute_s"]:
        total += row["compute_s"][phase]
    return total


def trace(summary):
    """What must repeat exactly for a seed: every eval and probe row's outcome and counters."""
    out = []
    for row in read_rows(summary["log_path"]):
        if row["type"] in ("eval", "probe"):
            out.append((row["type"], row["split"], row["status"], row["program_hash"], row["fidelity"],
                        row["combined"], row["token"], row["eval_count"], row["compute_used_s"]))
    return out


def spent(summary, method):
    """True when the arm stopped because its budget was spent."""
    if method in TRAINING_METHODS:
        left = summary["compute_budget_s"] - summary["compute_used_s"]
        return left < training.STEP_SECONDS[method] + 120.0
    out_of_evals = summary["eval_budget"] - summary["eval_used"] < 1.0 - 1e-9
    out_of_compute = summary["compute_used_s"] >= summary["compute_budget_s"]
    return out_of_evals or out_of_compute


# Every method.

@pytest.mark.parametrize("method", METHODS)
def test_each_method_spends_its_budget_and_logs_its_method(tmp_path, method):
    summary = run_baseline(baseline_config(tmp_path, method))
    assert sorted(summary) == SUMMARY_KEYS
    assert summary["method"] == method and summary["stub"] is True
    assert summary["eval_used"] <= summary["eval_budget"]
    assert spent(summary, method)
    assert summary["best_search"]["label"] == "SEARCH"
    assert 1 <= len(summary["heldout"]) <= 3

    rows = read_rows(summary["log_path"])
    evals = 0
    for row in rows:
        assert row["method"] == method
        assert row["backend"] == "stub" and row["stub"] is True
        if row["type"] == "eval" and row["split"] == "search" and row["token"] is not None:
            evals += 1
    assert evals > 0
    charges = 0
    for charge in Ledger(os.path.join(str(tmp_path), "ledger.jsonl")).read():
        if charge["run_id"] == "test-" + method:
            assert charge["method"] == method
            charges += 1
    assert charges > 0


@pytest.mark.parametrize("method", METHODS)
def test_no_arm_overspends_a_tight_compute_budget(tmp_path, method):
    config = baseline_config(tmp_path, method, {"eval_budget": 60, "compute_budget_s": 3000.0})
    summary = run_baseline(config)
    assert summary["eval_used"] <= 60
    assert spent(summary, method)
    rows = read_rows(summary["log_path"])
    if method in TRAINING_METHODS:
        # The training driver plans its chunks, so it never overshoots.
        assert summary["compute_used_s"] <= 3000.0
        assert summary["compute_overshoot_s"] == 0.0
        return
    # Rule 5: compute is checked before a batch and charged after, so a run
    # may overshoot by at most one batch of k jobs, and the overshoot is logged.
    largest = 0.0
    overshoot_logged = False
    for row in rows:
        if row["type"] in ("eval", "probe") and row["split"] != "heldout":
            largest = max(largest, row_compute(row))
        if row["type"] == "event" and row["kind"] == "compute_overshoot":
            overshoot_logged = True
    assert summary["compute_overshoot_s"] <= config["k"] * largest
    assert overshoot_logged == (summary["compute_overshoot_s"] > 0)


@pytest.mark.parametrize("method", METHODS)
def test_same_seed_gives_the_same_run(tmp_path, method):
    first = run_baseline(baseline_config(tmp_path / "a", method))
    second = run_baseline(baseline_config(tmp_path / "b", method))
    assert trace(first) == trace(second)
    assert first["best_search"] == second["best_search"]
    assert first["heldout"] == second["heldout"] and first["controls"] == second["controls"]


@pytest.mark.parametrize("method", ["evolve", "random", "random_part"])
def test_another_seed_gives_another_run(tmp_path, method):
    first = run_baseline(baseline_config(tmp_path / "a", method))
    other = run_baseline(baseline_config(tmp_path / "b", method, {"seed": 1}))
    assert trace(first) != trace(other)


def test_run_baseline_refuses_bad_configs_before_writing(tmp_path):
    with pytest.raises(ValueError):
        run_baseline(baseline_config(tmp_path, "cmaes"))
    with pytest.raises(ValueError):
        run_baseline(baseline_config(tmp_path, "evolve", {"memory": "past"}))
    with pytest.raises(ValueError):
        run_baseline(baseline_config(tmp_path, "random", {"llm": "mock"}))
    with pytest.raises(NotImplementedError):
        run_baseline(baseline_config(tmp_path, "lora", {"backend": "local"}))
    assert not os.path.exists(os.path.join(str(tmp_path), "runs"))


def test_summary_matches_the_swarm_and_the_cli_prints_it(tmp_path, capsys, monkeypatch):
    swarm = run_swarm(make_config({"outputs_dir": str(tmp_path / "swarm"), "run_id": "test-swarm", "eval_budget": 6}))
    spec = importlib.util.spec_from_file_location("run_baseline_script",
                                                  os.path.join(REPO_ROOT, "scripts", "run_baseline.py"))
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    monkeypatch.setattr(script, "load_dotenv", lambda: [])
    capsys.readouterr()
    summary = script.main(["--method", "random", "--backend", "stub", "--eval-budget", "6", "--seed", "0",
                           "--outputs-dir", str(tmp_path / "cli"), "--run-id", "cli-random"])
    assert sorted(summary) == sorted(swarm)
    assert summary["run_id"] == "cli-random" and summary["eval_used"] == 6.0
    out = capsys.readouterr().out
    assert "method: random" in out and "[SEARCH]" in out and "[STUB]" in out and "held-out rows" in out


# Regularized evolution.

def member(index, fitness):
    return {"program": {}, "program_hash": "h" + str(index), "describe": "m" + str(index), "fitness": fitness,
            "fidelity": "full", "standing": (1.0, fitness), "round": index}


def test_population_keeps_its_size_and_the_oldest_dies():
    population = Population(12)
    removed = []
    for index in range(15):
        gone = population.add(member(index, 1.0 - index * 0.01))
        if gone is not None:
            removed.append(gone["program_hash"])
    assert len(population.members) == 12
    assert removed == ["h0", "h1", "h2"]
    assert population.hashes()[0] == "h3" and population.hashes()[-1] == "h14"
    # Aging, not fitness: h0 was the fittest and still died first.
    assert population.best()["program_hash"] == "h3"


def test_tournament_returns_the_fittest_of_its_sample():
    population = Population(12)
    for index in range(12):
        population.add(member(index, random.Random(index).random()))
    for seed in range(20):
        picks = random.Random(seed).sample(range(12), 3)
        expected = None
        for index in picks:
            if expected is None or population.members[index]["fitness"] > expected["fitness"]:
                expected = population.members[index]
        assert population.tournament(random.Random(seed), 3) is expected
    assert population.tournament(random.Random(0), 12) is population.best()
    small = Population(12)
    small.add(member(0, 0.1))
    small.add(member(1, 0.5))
    assert small.tournament(random.Random(3), 3)["program_hash"] == "h1"
    assert Population(12).tournament(random.Random(0), 3) is None


def test_evolve_seeds_twelve_single_grafts_then_breeds_from_the_population(tmp_path):
    config = baseline_config(tmp_path, "evolve", {"eval_budget": 40, "crossover_rate": 0.5})
    run = factory.build_run(config)
    arm = EvolveArm(config, run["env"])
    summary = arm.run()
    catalog = run["catalog"]
    n_host = space.host_layers(config, catalog)
    scored = []
    init_scored = 0
    crossovers = 0
    for row in search_rows(summary):
        grafts = row["program"]["grafts"]
        operator = row["extra"]["operator"]
        if operator == "init":
            assert len(scored) < 12 and len(grafts) == 1
            record = catalog.records[grafts[0]["id"]]
            assert grafts[0]["at"] == space.default_at(record, config, n_host)
            assert grafts[0]["glue_rank"] is None and row["program"]["heal"]["steps"] == 300
            assert row["extra"]["parent"] is None
        else:
            assert operator in ("mutate", "crossover")
            assert row["extra"]["parent"] in scored
            if operator == "crossover":
                crossovers += 1
                assert len(grafts) <= 3
                assert row["extra"]["other_parent"] in scored
        if row["status"] in ("ok", "collapsed"):
            scored.append(row["program_hash"])
            if operator == "init":
                init_scored += 1
    assert init_scored == 12
    assert crossovers > 0
    # Without the cascade every scored child joins once and the oldest dies,
    # so the population is exactly the 12 most recently scored programs.
    assert arm.population.hashes() == scored[-12:]


def best_first(row):
    return -row["combined"]


def test_cascade_promotes_the_top_third_like_the_swarm(tmp_path):
    summary = run_baseline(baseline_config(tmp_path, "evolve", {"eval_budget": 12, "cascade": True}))
    rounds = {}
    for row in search_rows(summary):
        if row["round"] not in rounds:
            rounds[row["round"]] = {"low": [], "full": []}
        rounds[row["round"]][row["fidelity"]].append(row)
    assert len(rounds) > 1
    paid = 0.0
    for round_index in rounds:
        low = rounds[round_index]["low"]
        full = rounds[round_index]["full"]
        ok_low = []
        for row in low:
            if row["status"] == "ok":
                ok_low.append(row)
        assert len(full) == min(int(math.ceil(len(low) / 3.0)), len(ok_low))
        ok_low.sort(key=best_first)
        promoted = []
        for row in full:
            promoted.append(row["program_hash"])
        top = []
        for row in ok_low[:len(full)]:
            top.append(row["program_hash"])
        assert sorted(promoted) == sorted(top)
        for row in low + full:
            if row["token"] is not None:
                paid += row["extra"]["eval_cost"]
    assert abs(paid - summary["eval_used"]) < 1e-6
    # Promotion pricing (env.py): a promotion pays only the FEE difference when the low level scored the
    # same model, meaning the same heal steps. A program that heals for more than the short heal
    # (config low_fidelity_heal_steps, 50) heals again at full, scores a new model and pays the whole level.
    short = baseline_config(tmp_path, "evolve")["low_fidelity_heal_steps"]
    for row in search_rows(summary):
        if row["fidelity"] == "full" and row["token"] is not None:
            steps = row["heal_steps_used"]
            if steps is None or steps <= short:
                assert row["extra"]["eval_cost"] == round(1.0 - 0.34, 3)
            else:
                assert row["extra"]["eval_cost"] == 1.0


# The random-part arm.

def test_random_part_evaluates_only_random_controls(tmp_path):
    summary = run_baseline(baseline_config(tmp_path, "random_part", {"eval_budget": 20}))
    grafts_seen = 0
    for row in read_rows(summary["log_path"]):
        if row["type"] in ("eval", "probe"):
            for graft in row["program"]["grafts"]:
                assert graft["control"] == "random"
                grafts_seen += 1
    assert grafts_seen > 20

    evolve = run_baseline(baseline_config(tmp_path / "evolve", "evolve", {"eval_budget": 20}))

    def first_slots(rows):
        slots = []
        for row in rows:
            if row["extra"]["operator"] == "init":
                graft = row["program"]["grafts"][0]
                slots.append((graft["id"], graft["at"]))
        return slots

    # Paired start: the same seed grafts the same slots, with random units.
    assert first_slots(search_rows(summary)) == first_slots(search_rows(evolve))


# Greedy.

def test_greedy_scores_singles_then_adds_then_shifts(tmp_path):
    summary = run_baseline(baseline_config(tmp_path, "greedy", {"eval_budget": 40}))
    phases = []
    for row in search_rows(summary):
        phases.append(row["extra"]["phase"])
        if row["extra"]["phase"] == "single":
            assert len(row["program"]["grafts"]) == 1 and row["program"]["heal"]["steps"] == 300
    first_add = phases.index("add")
    assert first_add >= 20 and phases[:first_add] == ["single"] * first_add
    assert "shift" in phases
    bests = []
    for row in read_rows(summary["log_path"]):
        if row["type"] == "event" and row["kind"] == "greedy_best":
            bests.append(row["extra"]["combined"])
    assert len(bests) >= 1
    for index in range(1, len(bests)):
        assert bests[index] > bests[index - 1]
    # The final pick uses the judge's rule, which may prefer a smaller program within tie_margin.
    assert summary["best_search"]["combined"] >= bests[-1] - summary["config"]["tie_margin"] - 1e-9


# Training arms.

def test_training_stubs_charge_train_compute_and_one_token_per_checkpoint(tmp_path):
    steps = {}
    for method in TRAINING_METHODS:
        config = baseline_config(tmp_path / method, method, {"compute_budget_s": 10000.0})
        summary = run_baseline(config)
        rows = search_rows(summary)
        assert len(rows) == training.CHECKPOINTS and summary["eval_used"] == training.CHECKPOINTS
        tokens = []
        for row in rows:
            assert row["fidelity"] == "full" and row["program"] == {} and row["status"] == "ok"
            assert row["compute_s"]["eval"] == 120.0 and row["compute_s"]["train"] > 0
            tokens.append(row["token"])
        assert tokens == [0, 1, 2, 3]
        train_s = 0.0
        for charge in Ledger(os.path.join(str(tmp_path / method), "ledger.jsonl")).read():
            if charge["axis"] == "compute" and charge["phase"] == "train":
                train_s += charge["amount"]
        assert train_s == rows[-1]["extra"]["step"] * training.STEP_SECONDS[method]
        assert 10000.0 - training.STEP_SECONDS[method] < summary["compute_used_s"] <= 10000.0
        heldout = search_rows(summary, "heldout")
        assert len(heldout) == 1 and summary["controls"] == []
        assert heldout[0]["program_hash"] == summary["best_search"]["program_hash"]
        steps[method] = rows[-1]["extra"]["step"]
    # 9520 s of training at 0.5 and 1.5 s per step.
    assert steps["lora"] == 19040 and steps["full_ft"] == 6346


def test_training_stub_scores_depend_on_the_run_seed():
    world = training.stub.StubWorld(0)
    args = [world, "lora", 4000, "search", ["gsm8k", "if_guard"], ["if_guard"], 1.0, 0]
    seed_0 = training.stub_checkpoint_scores(*args, run_seed=0)
    assert training.stub_checkpoint_scores(*args, run_seed=0) == seed_0
    assert training.stub_checkpoint_scores(*args, run_seed=1) != seed_0


def test_lora_parameter_arithmetic_for_the_host():
    assert training.lora_params(1) == 704512
    assert training.full_params() == 1235814400
    assert training.matched_lora_rank(2400000 + 6300000) == 12
    assert training.matched_lora_rank(2400000 + 460000) == 4
    assert training.matched_lora_rank(10) == 1
    assert training.plan_checkpoints(10, 4) == [2, 5, 7, 10]
    config = make_config({"method": "lora", "lora_match_params": 8700000})
    assert training.StubTrainer(config, "lora").describe() == "LoRA r12 on all linear layers"
