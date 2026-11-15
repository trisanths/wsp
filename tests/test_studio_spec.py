"""Spec limit rules: run size follows total size, and the part-room headroom comes from the catalog. CPU only."""

from studio.server import specs


def body_with(total, run=None):
    limits = {"max_total_params_b": total}
    if run is not None:
        limits["max_run_params_b"] = run
    return {"limits": limits}


def make_old():
    return {"id": "spec_x", "title": "t", "summary": "s", "request_type": "build", "host": "unsloth/Llama-3.2-1B-Instruct",
            "size": {"max_params_b": 1.5, "min_b": 0.5, "max_b": 32.0, "host_params_b": 1.24},
            "limits": {"max_total_params_b": 1.5, "max_run_params_b": 1.4, "max_heal_min": 20},
            "budget": {"evals": 30, "compute_min": 45, "min_evals": 5, "max_evals": 120},
            "benchmarks": [], "comparators": {"distill": {}}}


def test_smallest_part_is_small_and_positive(monkeypatch, tmp_path):
    monkeypatch.setenv("WSP_STUDIO_OUT", str(tmp_path))
    monkeypatch.setenv("WSP_STUDIO_BACKEND", "stub")
    specs._smallest_part_cache.clear()
    smallest = specs.smallest_part_b("stub")
    assert smallest is not None and smallest > 0
    assert specs.run_headroom_b() >= smallest - 1e-9


def test_headroom_falls_back_to_constant(monkeypatch):
    monkeypatch.setattr(specs, "smallest_part_b", lambda backend=None: None)
    assert specs.run_headroom_b() == specs.BUILD_ROOM_B


def test_headroom_rounds_up_smallest_part(monkeypatch):
    monkeypatch.setattr(specs, "smallest_part_b", lambda backend=None: 0.091)
    assert specs.run_headroom_b() == 0.1


def test_llm_draft_run_keeps_smallest_part_room(monkeypatch):
    monkeypatch.setattr(specs, "smallest_part_b", lambda backend=None: 0.09)
    spec = {"size": {"max_params_b": 2.0}, "limits": {"max_total_params_b": 2.0, "max_run_params_b": 2.0},
            "benchmarks": []}
    specs.apply_llm_draft(spec, {}, {"max_run_params_b": 1.24}, 1.24)
    assert spec["limits"]["max_run_params_b"] == 1.33


def edited(monkeypatch, tmp_path, body):
    monkeypatch.setenv("WSP_STUDIO_OUT", str(tmp_path))
    monkeypatch.setenv("WSP_STUDIO_BACKEND", "stub")
    return specs.normalize_spec(body, make_old())["limits"]


def test_total_change_moves_run_by_same_amount(monkeypatch, tmp_path):
    limits = edited(monkeypatch, tmp_path, body_with(2.5))
    assert limits["max_total_params_b"] == 2.5
    assert limits["max_run_params_b"] == 2.4


def test_lowering_total_clamps_run(monkeypatch, tmp_path):
    limits = edited(monkeypatch, tmp_path, body_with(1.3))
    assert limits["max_total_params_b"] == 1.3
    assert limits["max_run_params_b"] == 1.24


def test_explicit_run_edit_wins(monkeypatch, tmp_path):
    limits = edited(monkeypatch, tmp_path, body_with(2.5, 1.6))
    assert limits["max_run_params_b"] == 1.6
