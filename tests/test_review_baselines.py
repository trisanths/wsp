"""Regression tests for modal_baselines.py (review of the baselines stream).

These run on CPU with the fake backend and toy GSM8K rows. Nothing here
talks to Modal: the volume is a stub and the GPU steps are replaced.
"""

import importlib
import json
import os

import pytest

pytest.importorskip("modal")

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class StubVolume:
    def commit(self):
        pass

    def reload(self):
        pass


def toy_gsm8k(name, config, split):
    rows = []
    for i in range(260):
        rows.append({"question": "q" + str(i), "answer": "work\n#### " + str(i % 100)})
    return rows


@pytest.fixture
def mb(tmp_path, monkeypatch):
    monkeypatch.chdir(REPO)
    module = importlib.import_module("modal_baselines")
    from wsp_eval import splits, tasks as tasks_module

    monkeypatch.setattr(tasks_module, "_load_dataset", toy_gsm8k)
    monkeypatch.setattr(splits, "_items_cache", {})
    monkeypatch.setattr(module, "volume", StubVolume())
    monkeypatch.setattr(module, "MODEL_ROOT", str(tmp_path / "baselines"))
    monkeypatch.setattr(module, "CELL_ROOT", str(tmp_path / "cells"))
    monkeypatch.setattr(module, "SHARED_ROOT", str(tmp_path / "shared"))
    monkeypatch.setattr(module, "decode_speed", lambda path, tokens=128: 100.0)
    monkeypatch.setattr(module, "free_gpu", lambda: None)
    # run_cell sets these with os.environ. Set then delete each one here, so
    # monkeypatch records it and restores the old state after the test.
    for name in ["WSP_ROOT", "WSP_BACKEND", "WSP_VLLM_MAX_LEN", "VLLM_ENABLE_V1_MULTIPROCESSING"]:
        monkeypatch.setenv(name, "unset")
        monkeypatch.delenv(name)
    os.makedirs(str(tmp_path / "shared" / "work"))
    return module


def test_cell_env_sets_a_vllm_length_every_model_accepts(mb, tmp_path):
    # The harness default (8192) is above Qwen2.5-Math-1.5B-Instruct's 4096
    # positions, so vLLM would refuse to load that ceiling.
    with open(str(tmp_path / "shared" / "work" / "reference_search.json"), "w") as f:
        json.dump({"base": {}, "parents": {}}, f)
    root = mb.cell_env("ceiling_x", "vllm")
    assert int(os.environ["WSP_VLLM_MAX_LEN"]) <= 4096
    assert int(os.environ["WSP_VLLM_MAX_LEN"]) >= 1024
    assert os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] == "0"
    assert os.environ["WSP_ROOT"] == root
    assert os.path.exists(os.path.join(root, "work", "reference_search.json"))


def test_every_cell_uses_vllm_first(mb):
    # vLLM 0.11 registers Lfm2ForCausalLM, so the Liquid ceilings use the same backend as the rest.
    cells = mb.build_cells(["llama1b_ft", "qwen05_ft", "ceiling"])
    assert len(cells) == 18
    for cell in cells:
        assert cell[6] == "vllm"


def test_score_cell_falls_back_to_hf_when_vllm_fails(mb, monkeypatch):
    from wsp_eval import harness

    calls = []

    def fake_evaluate(model_path, split, tasks, namespace=None, backend=None, note=""):
        calls.append(backend)
        if backend == "vllm":
            raise ValueError("model architecture not supported")
        return {"backend": backend}

    monkeypatch.setattr(harness, "evaluate", fake_evaluate)
    result, used, error = mb.score_cell("some/model", "ceiling", "vllm", "ceiling_x")
    assert calls == ["vllm", "hf"]
    assert used == "hf"
    assert "not supported" in error
    assert result["backend"] == "hf"

    def hf_fails(model_path, split, tasks, namespace=None, backend=None, note=""):
        raise RuntimeError("hf broke")

    monkeypatch.setattr(harness, "evaluate", hf_fails)
    with pytest.raises(RuntimeError):
        mb.score_cell("some/model", "ceiling", "hf", "ceiling_x")


def test_trained_model_is_reused_only_with_the_same_settings(mb, tmp_path):
    path = str(tmp_path / "model")
    os.makedirs(path)
    program = mb.full_ft_program("host/id", "1e-5", 0)
    assert not mb.trained_already(path, program)
    with open(os.path.join(path, "program.json"), "w") as f:
        json.dump(program, f)
    assert not mb.trained_already(path, program)  # no marker yet
    with open(os.path.join(path, "wsp_train.json"), "w") as f:
        json.dump({}, f)
    assert mb.trained_already(path, program)
    changed = dict(program)
    changed["steps"] = program["steps"] + 100
    assert not mb.trained_already(path, changed)


def test_cells_end_to_end_with_the_fake_backend(mb, monkeypatch):
    trained = []

    def fake_train(host, lr, seed, out_dir):
        trained.append(out_dir)
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "config.json"), "w") as f:
            json.dump({"host": host, "seed": seed}, f)
        losses = []
        for step in range(mb.FT_STEPS):
            losses.append(2.0)
        return {"trainable_params": 1, "train_seconds": 0.0, "loss_first": 2.0, "loss_last10": 2.0,
                "losses": losses}

    monkeypatch.setattr(mb, "train_full_ft", fake_train)
    rows = []
    for seed in [0, 1, 2]:
        rows.append(mb.run_cell.local("ft_s" + str(seed), "full_ft", "host/id", "1e-5", seed, "full_ft", "fake"))
    rows.append(mb.run_cell.local("ceiling_x", "ceiling", "fake:donor", None, None, "ceiling", "fake"))
    again = mb.run_cell.local("ft_s0", "full_ft", "host/id", "1e-5", 0, "full_ft", "fake")
    assert len(trained) == 3
    assert again["score"] == rows[0]["score"]

    for row in rows:
        assert row["n"] == 200
        assert row["backend"] == "fake"
    seeds = []
    for row in rows[:3]:
        seeds.append(row["program_seed"])
    assert seeds == [0, 1, 2]

    from wsp_eval import claims
    ft_results = []
    for row in rows[:3]:
        ft_results.append(row["eval_result"])
    assert claims.summarize(ft_results, "gsm8k")["n_seeds"] == 3

    entries = []
    for row in rows:
        entries.append({"cell": row["cell"], "namespace": row["namespace"], "n": row["n"],
                        "generations_run": row["generations_run"], "experiments_line": row["experiments_line"]})
    usage = mb.record_shared.local(entries)
    assert usage["full_ft"]["eval_calls"] == 3
    assert usage["ceiling"]["eval_calls"] == 1

    summary = mb.summarize(rows)
    assert summary[0]["n_runs"] == 3
    assert summary[0]["backends"] == ["fake"]
    assert summary[1]["score_sd"] is None
