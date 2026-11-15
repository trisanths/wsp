"""Tests for wsp/swarm/modal_backend.py, the bridge from WspEnv to the deployed wsp-swarm app.

CPU only, no network: the deployed functions are fakes with a .remote method,
and every Modal Dict is the FakeDict from test_swarm_modal.py.
"""

import os
import threading

import pytest

import modal_swarm
from wsp.catalog import Catalog
from wsp.swarm import modal_backend
from wsp.swarm import program_schema as ps
from wsp.swarm.budget import LocalBudget
from wsp.swarm.config import REPO_ROOT, make_config
from wsp.swarm.env import WspEnv
from wsp.swarm.log import RunLog

PILOT_PATH = os.path.join(REPO_ROOT, "data", "pilot_catalog.jsonl")
PART = "pilot_qmath15_l6_top512"
HOST = modal_swarm.HOST


class FakeDict:
    """A thread-safe stand-in for modal.Dict (the same contract as the one in test_swarm_modal.py)."""

    def __init__(self):
        self.data = {}
        self.lock = threading.Lock()

    def put(self, key, value, skip_if_exists=False):
        with self.lock:
            if skip_if_exists and key in self.data:
                return False
            self.data[key] = value
            return True

    def get(self, key, default=None):
        with self.lock:
            return self.data.get(key, default)

    def items(self):
        with self.lock:
            pairs = list(self.data.items())
        return iter(pairs)


class FakeRemote:
    """A deployed Modal function: .remote(...) records the call and runs a plain function."""

    def __init__(self, function):
        self.function = function
        self.calls = []
        self.lock = threading.Lock()

    def remote(self, *args, **kwargs):
        with self.lock:
            self.calls.append({"args": args, "kwargs": kwargs})
        return self.function(*args, **kwargs)


def graft_entry():
    return {"id": PART, "kind": "mlp_cluster", "control": None, "donor": "Qwen/Qwen2.5-Math-1.5B",
            "donor_layer": 6, "host_layer": 3, "d_host": 2048, "d_donor": 1536, "part_params": 2359296,
            "glue_params": 6291456, "fit": "ridge", "fit_r2": 0.68, "fit_pairs": 9000, "n_units": 512}


def result_fields(kwargs):
    return {"run_id": kwargs.get("run_id"), "pool": kwargs.get("pool"), "who": kwargs.get("who"),
            "method": kwargs.get("method"), "attempt": "attempt1", "charge_prefix": "prefix",
            "charged": [], "uncharged": {}, "cache_hit": False, "gpu": "NVIDIA L4", "setup_seconds": 3.0}


class FakeApp:
    """The three deployed functions, with the per-run caches the real containers keep."""

    def __init__(self):
        self.probe_seconds = 20.0
        self.run_seconds = {"compile": 15.0, "heal": 30.0, "eval": 40.0}
        self.run_uncharged = {}
        self.probe_raises = False
        self.run_raises = False
        self.run_ok = True
        self.task_scores = {"gsm8k": 0.40, "if_guard": 0.42}
        self.host_harness_scores = {"if_guard": 0.41, "mmlupro:law": 0.12}
        self.probe = FakeRemote(self.probe_candidate)
        self.run = FakeRemote(self.run_candidate)
        self.host = FakeRemote(self.host_baseline)

    def probe_candidate(self, program_dict, catalog_name, align_samples=64, run_id=None, pool=None, who="",
                        method=""):
        if self.probe_raises:
            raise ConnectionError("fake Modal outage")
        result = result_fields({"run_id": run_id, "pool": pool, "who": who, "method": method})
        result.update({"ok": True, "error": None, "phase": None, "graft_report": [graft_entry()],
                       "seconds": {"compile": self.probe_seconds}})
        modal_swarm.cache_store(modal_swarm.PROBE_DICT,
                                modal_swarm.probe_key(program_dict, catalog_name, run_id, align_samples), result)
        return result

    def run_candidate(self, program_dict, split, token, catalog_name, eval_limit=None, align_samples=64,
                      run_id=None, pool=None, who="", method="", tasks=None, scorer="interim"):
        if self.run_raises:
            raise ConnectionError("fake Modal outage")
        if tasks is None:
            tasks = ["gsm8k"]
        scores = {}
        for name in tasks:
            scores[name] = self.task_scores[name]
        result = result_fields({"run_id": run_id, "pool": pool, "who": who, "method": method})
        result.update({"ok": self.run_ok, "error": None, "phase": None, "scores": scores,
                       "seconds": dict(self.run_seconds), "heal_loss": [1.2, 1.0], "token": token,
                       "compile_report": {"grafts": [graft_entry()]}, "eval_backend": scorer,
                       "label": modal_swarm.SCORER_LABELS[scorer], "uncharged": dict(self.run_uncharged)})
        if not self.run_ok:
            result["error"] = "RuntimeError: CUDA out of memory"
            result["phase"] = "heal"
            result["scores"] = {}
        return result

    def host_baseline(self, split, eval_limit=None, host=HOST, tasks=None, scorer="interim"):
        if scorer == "harness":
            scores = {}
            for name in tasks:
                scores[name] = self.host_harness_scores[name]
            return {"ok": True, "error": None, "scores": scores, "seconds": {"eval": 90.0}, "split": split,
                    "cache_hit": False, "eval_backend": "harness"}
        return {"ok": True, "error": None, "scores": {"gsm8k": 0.25}, "seconds": {"eval": 50.0},
                "split": split, "cache_hit": False}


@pytest.fixture
def app(monkeypatch):
    """Fake the deployed functions and every Modal Dict. Returns the FakeApp, with .dicts."""
    fake = FakeApp()
    functions = {"probe_candidate": fake.probe, "run_candidate": fake.run, "host_baseline": fake.host}
    dicts = {}
    for name in modal_swarm.DICT_NAMES:
        dicts[name] = FakeDict()

    def fake_deployed(name):
        return functions[name]

    def fake_dict(name):
        return dicts[name]

    monkeypatch.setattr(modal_swarm, "deployed", fake_deployed)
    monkeypatch.setattr(modal_swarm, "get_dict", fake_dict)
    fake.dicts = dicts
    return fake


def bridge_config(tmp_path, overrides=None):
    values = {"outputs_dir": str(tmp_path), "run_id": "modal-test-run", "method": "swarm", "backend": "modal",
              "catalog_path": PILOT_PATH, "task_type": "math_interim"}
    if overrides is not None:
        for key in overrides:
            values[key] = overrides[key]
    return make_config(values)


def pilot_catalog():
    return Catalog.load(PILOT_PATH)


def program(steps=300, at=3, part=PART):
    return ps.make_program(HOST, [ps.make_graft(part, at)], ps.make_heal(steps=steps))


def bridge(tmp_path, overrides=None):
    return modal_backend.ModalCompileEval(bridge_config(tmp_path, overrides), pilot_catalog())


def make_env(tmp_path, backend, eval_budget=3):
    config = backend.config
    budget = LocalBudget(eval_budget, 10000.0, config["run_id"], config["method"], "search")
    heldout = LocalBudget(9, 10000.0, config["run_id"], config["method"], "heldout")
    log = RunLog(str(tmp_path / "runs" / config["run_id"]))
    return WspEnv(config, backend.catalog, backend, budget, log, heldout), budget


# Construction refuses what the Modal image cannot run.


def test_refuses_tasks_the_interim_scorer_cannot_grade(tmp_path, app):
    with pytest.raises(ValueError) as error:
        bridge(tmp_path, {"task_type": "math"})
    assert "if_guard" in str(error.value) and "math_interim" in str(error.value)


def test_refuses_a_catalog_the_image_does_not_carry(tmp_path, app):
    with pytest.raises(ValueError) as error:
        bridge(tmp_path, {"catalog_path": str(tmp_path / "other.jsonl")})
    assert "pilot_catalog.jsonl" in str(error.value)


def test_refuses_a_missing_run_id(tmp_path, app):
    with pytest.raises(ValueError):
        bridge(tmp_path, {"run_id": None})


def test_catalog_names_follow_the_image(tmp_path, app):
    assert bridge(tmp_path).catalog_name == "pilot"
    main = bridge(tmp_path, {"catalog_path": os.path.join(REPO_ROOT, "catalog.jsonl")})
    assert main.catalog_name == "main"


# compile: a fit-only probe, cached per run.


def test_compile_probes_and_returns_the_graft_report(tmp_path, app):
    backend = bridge(tmp_path)
    result = backend.compile(program(), fit_only=False)
    assert result["ok"] is True
    assert result["graft_report"][0]["fit_r2"] == 0.68
    assert result["compute_s"] == {"compile": 20.0}
    assert result["handle"]["program"] == ps.normalize(program())
    call = app.probe.calls[0]
    assert call["args"][1] == "pilot"
    assert call["kwargs"]["run_id"] == "modal-test-run" and call["kwargs"]["pool"] == "search"
    assert call["kwargs"]["align_samples"] == backend.config["align_samples"]


def test_compile_reads_the_run_probe_cache_without_a_container(tmp_path, app):
    backend = bridge(tmp_path)
    backend.compile(program(steps=300), fit_only=True)
    again = backend.compile(program(steps=600), fit_only=False)
    assert len(app.probe.calls) == 1
    assert again["ok"] is True and again["compute_s"] == {"compile": 0.0}
    assert again["graft_report"][0]["fit_r2"] == 0.68


def test_compile_reports_a_failed_probe(tmp_path, app):
    app.probe_raises = True
    result = bridge(tmp_path).compile(program(), fit_only=False)
    assert result["ok"] is False and "fake Modal outage" in result["error"]


# heal: deferred to run_candidate.


def test_heal_is_deferred_and_keeps_the_heal_it_was_given(tmp_path, app):
    backend = bridge(tmp_path)
    handle = backend.compile(program(steps=300))["handle"]
    short = ps.with_heal_steps(program(steps=300), 50)
    healed = backend.heal(handle, short)
    assert healed["ok"] is True and healed["compute_s"] == {}
    assert ps.heal_steps(healed["handle"]["program"]) == 50
    assert len(app.run.calls) == 0


# evaluate: compile, heal and grade in one container.


def test_fidelity_maps_to_nested_question_counts(tmp_path, app):
    backend = bridge(tmp_path)
    assert backend.eval_limit_for("search", "check") == 32
    assert backend.eval_limit_for("search", "low") == 68
    assert backend.eval_limit_for("search", "full") is None
    assert backend.eval_limit_for("heldout", "full") is None


def test_evaluate_runs_the_candidate_with_the_spent_token(tmp_path, app):
    backend = bridge(tmp_path)
    handle = backend.compile(program())["handle"]
    result = backend.evaluate(handle, "search", ["gsm8k"], 7, "low")
    assert result["ok"] is True and result["scores"] == {"gsm8k": 0.40}
    assert result["compute_s"] == {"compile": 15.0, "heal": 30.0, "eval": 40.0}
    assert result["heal_loss"] == [1.2, 1.0] and result["label"] == "INTERIM"
    call = app.run.calls[0]
    assert call["args"][1] == "search" and call["args"][2] == 7 and call["args"][3] == "pilot"
    assert call["kwargs"]["eval_limit"] == 68 and call["kwargs"]["pool"] == "search"


def test_evaluate_reads_the_run_result_cache_first(tmp_path, app):
    backend = bridge(tmp_path)
    handle = backend.compile(program())["handle"]
    first = backend.evaluate(handle, "search", ["gsm8k"], 0, "full")
    key = modal_swarm.result_key(handle["program"], "search", "pilot", "modal-test-run", None,
                                 backend.align_samples)
    stored = dict(first)
    stored["seconds"] = dict(app.run_seconds)
    app.dicts[modal_swarm.CACHE_DICT].put(key, stored)
    again = backend.evaluate(handle, "search", ["gsm8k"], 1, "full")
    assert len(app.run.calls) == 1
    assert again["ok"] is True and again["scores"] == {"gsm8k": 0.40}
    assert again["compute_s"] == {"compile": 0.0, "heal": 0.0, "eval": 0.0}


def test_uncharged_phases_are_settled_in_the_modal_ledger(tmp_path, app):
    app.run_uncharged = {"eval": 40.0}
    backend = bridge(tmp_path)
    handle = backend.compile(program())["handle"]
    backend.evaluate(handle, "search", ["gsm8k"], 0, "full")
    ledger = app.dicts[modal_swarm.LEDGER_DICT].data
    key = modal_swarm.ledger_key("modal-test-run", "search", "compute", "prefix:eval:attempt1")
    assert key in ledger and ledger[key]["amount"] == 40.0 and ledger[key]["phase"] == "eval"


def test_evaluate_refuses_a_task_other_than_gsm8k(tmp_path, app):
    backend = bridge(tmp_path)
    handle = backend.compile(program())["handle"]
    result = backend.evaluate(handle, "search", ["gsm8k", "if_guard"], 0, "full")
    assert result["ok"] is False and "if_guard" in result["error"]
    assert len(app.run.calls) == 0


def test_base_scores_come_from_host_baseline_once_per_split(tmp_path, app):
    backend = bridge(tmp_path)
    assert backend.base_scores("search", ["gsm8k"]) == {"gsm8k": 0.25}
    assert backend.base_scores("search", ["gsm8k"]) == {"gsm8k": 0.25}
    assert len(app.host.calls) == 1
    backend.base_scores("heldout", ["gsm8k"])
    assert len(app.host.calls) == 2


# The bridge inside WspEnv.


def test_env_step_on_the_bridge(tmp_path, app):
    backend = bridge(tmp_path)
    env, budget = make_env(tmp_path, backend)
    row = env.step(program(), "planner-1", "early math cluster helps")[4]
    assert row["status"] == "ok" and row["token"] == 0
    assert row["scores"] == {"gsm8k": 0.40} and row["backend"] == "modal"
    assert row["heal_loss"] == [1.2, 1.0]
    assert list(row["fit_r2"].values()) == [0.68]
    assert row["compute_s"] == {"compile": 35.0, "heal": 30.0, "eval": 40.0}
    assert budget.eval_used() == 1 and budget.compute_used() == 105.0


def test_env_probe_then_step_pays_the_fit_compile_once(tmp_path, app):
    backend = bridge(tmp_path)
    env, budget = make_env(tmp_path, backend)
    env.probe(program(), "planner-1")
    row = env.step(program(), "planner-1", "probed first")[4]
    assert len(app.probe.calls) == 1
    assert row["compute_s"]["compile"] == 15.0
    assert budget.compute_used() == 20.0 + 15.0 + 30.0 + 40.0


def test_env_check_fidelity_sends_the_short_heal_and_32_questions(tmp_path, app):
    backend = bridge(tmp_path, {"cascade": True})
    env, budget = make_env(tmp_path, backend)
    row = env.step(program(steps=300), "planner-1", "cheap check", fidelity="check")[4]
    assert row["status"] == "ok" and row["heal_steps_used"] == 50
    call = app.run.calls[0]
    assert ps.heal_steps(call["args"][0]) == 50 and call["kwargs"]["eval_limit"] == 32
    assert budget.eval_used() == 0.16


def test_env_failures_follow_the_step_rules(tmp_path, app):
    backend = bridge(tmp_path)
    env, budget = make_env(tmp_path, backend)
    app.probe_raises = True
    row = env.step(program(steps=100), "planner-1", "probe outage")[4]
    assert row["status"] == "compile_error" and row["token"] is None
    assert budget.eval_used() == 0
    app.probe_raises = False
    app.run_raises = True
    row = env.step(program(steps=300), "planner-1", "run outage")[4]
    assert row["status"] == "infra_error" and row["token"] == 0
    assert budget.eval_used() == 1
    app.run_raises = False
    app.run_ok = False
    row = env.step(program(steps=600), "planner-1", "heal crash")[4]
    assert row["status"] == "infra_error" and "out of memory" in row["extra"]["error"]
    assert budget.eval_used() == 2


# The harness scorer: gsm8k plus if_guard, whole splits only.


def harness_overrides(extra=None):
    values = {"modal_scorer": "harness", "task_type": "math"}
    if extra is not None:
        for key in extra:
            values[key] = extra[key]
    return values


def test_interim_refuses_the_guard_and_points_at_both_fixes(tmp_path, app):
    with pytest.raises(ValueError) as error:
        bridge(tmp_path, {"task_type": "math"})
    assert "math_interim" in str(error.value) and "harness" in str(error.value)


def test_harness_refuses_the_cascade_and_an_unknown_scorer(tmp_path, app):
    with pytest.raises(ValueError) as error:
        bridge(tmp_path, harness_overrides({"cascade": True}))
    assert "cascade" in str(error.value)
    with pytest.raises(ValueError):
        bridge(tmp_path, {"modal_scorer": "vllm"})


def test_harness_measures_the_bases_the_table_lacks(tmp_path, app):
    # A researched setup: the table has gsm8k, but if_guard (dropped here) and mmlupro:law must be measured.
    partial = {"search": {"gsm8k": 0.23}, "heldout": {"gsm8k": 0.201}}
    researched = {"task_type": "researched", "base_scores": partial,
                  "task_types": {"researched": {"tasks": {"gsm8k": 1.0, "mmlupro:law": 1.0}, "guards": ["if_guard"],
                                                "heal_data": ["gsm8k_train"]}}}
    backend = bridge(tmp_path, harness_overrides(researched))
    scores = backend.base_scores("search", ["gsm8k", "mmlupro:law", "if_guard"])
    assert scores == {"gsm8k": 0.23, "mmlupro:law": 0.12, "if_guard": 0.41}
    call = app.host.calls[0]
    assert call["kwargs"]["scorer"] == "harness" and call["kwargs"]["tasks"] == ["mmlupro:law", "if_guard"]
    backend.base_scores("search", ["gsm8k", "mmlupro:law", "if_guard"])
    assert len(app.host.calls) == 1


def test_harness_mode_sends_tasks_and_scorer_and_whole_splits(tmp_path, app):
    backend = bridge(tmp_path, harness_overrides())
    assert backend.label == "HARNESS"
    handle = backend.compile(program())["handle"]
    result = backend.evaluate(handle, "search", ["gsm8k", "if_guard"], 4, "full")
    assert result["ok"] is True and result["scores"] == {"gsm8k": 0.40, "if_guard": 0.42}
    assert result["label"] == "HARNESS"
    call = app.run.calls[0]
    assert call["kwargs"]["tasks"] == ["gsm8k", "if_guard"] and call["kwargs"]["scorer"] == "harness"
    assert call["kwargs"]["eval_limit"] is None
    refused = backend.evaluate(handle, "search", ["gsm8k", "if_guard"], 5, "check")
    assert refused["ok"] is False and "whole splits" in refused["error"]
    assert len(app.run.calls) == 1


def test_harness_base_scores_come_from_the_config_table(tmp_path, app):
    backend = bridge(tmp_path, harness_overrides())
    table = backend.config["base_scores"]
    for split in ["search", "heldout"]:
        scores = backend.base_scores(split, ["gsm8k", "if_guard"])
        assert scores == {"gsm8k": table[split]["gsm8k"], "if_guard": table[split]["if_guard"]}
    assert len(app.host.calls) == 0


def test_env_step_with_the_harness_counts_the_guard(tmp_path, app):
    backend = bridge(tmp_path, harness_overrides())
    env, budget = make_env(tmp_path, backend)
    kept = env.step(program(steps=300), "planner-1", "guard holds")[4]
    assert kept["status"] == "ok" and kept["scores"] == {"gsm8k": 0.40, "if_guard": 0.42}
    app.task_scores = {"gsm8k": 0.40, "if_guard": 0.30}
    broken = env.step(program(steps=600), "planner-1", "guard breaks")[4]
    assert broken["status"] == "ok"
    # Same GSM8K score, but if_guard fell more than 10% below the host's 0.42 (still above half of it,
    # the collapse line), so the guard penalty lowers the combined score.
    assert broken["combined"] < kept["combined"]
    app.task_scores = {"gsm8k": 0.40, "if_guard": 0.20}
    collapsed = env.step(program(steps=100), "planner-1", "guard collapses")[4]
    assert collapsed["status"] == "collapsed"
    assert budget.eval_used() == 3
