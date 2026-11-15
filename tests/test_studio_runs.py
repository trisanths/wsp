from tests.test_studio_s1 import fake_state


def test_run_with_nothing_scored_ends_as_empty():
    state = fake_state()
    state.refused = 195
    state.set_run_status("done")
    snap = state.snapshot()
    assert snap["outcome"] == "empty"
    assert snap["best"] is None
    assert all(agent["now"] == "Ended, nothing scored" for agent in snap["agents"])


def test_stopped_run_with_nothing_scored_stays_stopped():
    state = fake_state()
    state.stopped_early = True
    state.set_run_status("done")
    snap = state.snapshot()
    assert snap["outcome"] == "stopped"
    assert snap["agents"][0]["now"] == "Stopped"


def test_readable_names_tell_glue_and_heal_data_apart():
    from studio.server import programs

    def program(glue, data):
        return {"host": "unsloth/Llama-3.2-1B-Instruct", "seed": 0, "masks": None, "task_vectors": [],
                "grafts": [{"id": "stub_qmath_l10_c0", "at": 6, "control": None, "control_seed": 0, "glue_rank": glue}],
                "heal": {"steps": 300, "data": data, "lr": 3e-4, "batch_size": 8, "seq_len": 384}}

    names = {programs.readable(program(glue, data)) for glue in [None, 64] for data in ["math", "code"]}
    assert len(names) == 4
    assert programs.readable(program(64, "math")).endswith("at L6 (glue r64), heal 300 on math")
