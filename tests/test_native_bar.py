"""The native bar: the planner names the best native model at a compile's per-token cost. CPU only, no network."""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from test_pgraft_stage import HOST, make_plan, request  # noqa: E402

from wsp import auto, native_bar  # noqa: E402


def test_best_native_respects_the_budget_and_the_exclusion():
    assert native_bar.best_native(1.3e9)["model"] == HOST
    assert native_bar.best_native(1.3e9, exclude=(HOST,)) is None
    assert native_bar.best_native(2.0e9)["model"] == "Qwen/Qwen2.5-1.5B-Instruct"
    # at 3.2B per token Qwen2.5-3B (66.6 tuned) beats Llama-3.2-3B (63.4 tuned)
    assert native_bar.best_native(3.3e9)["model"] == "Qwen/Qwen2.5-3B-Instruct"


def test_check_reports_clears_and_handles_missing_numbers():
    rows = [{"model": "a", "params_per_token": 1e9, "plain": 50.0, "tuned": None, "source": "x"}]
    assert native_bar.check(2e9, 55.0, natives=rows)["clears"] is True
    low = native_bar.check(2e9, 45.0, natives=rows)
    assert low["clears"] is False and "picking a as the base" in low["text"]
    assert native_bar.check(2e9, None, natives=rows)["clears"] is None
    assert native_bar.check(0.5e9, 55.0, natives=rows)["bar"] is None


def test_the_measured_write_in_does_not_clear_the_native_bar():
    p = make_plan(request())
    assert p["feasible"], p["refusals"]
    bar = p["native_bar"][0]
    assert bar["clears"] is False
    assert bar["bar"]["model"] == "Qwen/Qwen2.5-3B-Instruct"
    assert any("does not clear the native bar" in w for w in p["warnings"])
    assert "Native bar" in auto.format_plan(p)

