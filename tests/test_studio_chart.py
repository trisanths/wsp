from tests.test_studio_s1 import fake_state


def scored(state, key, score, status="ok", guard=0.5):
    row = {"program_hash": key, "describe": "d-" + key, "scores": {"gsm8k": score, "if_guard": guard},
           "combined": score, "status": status, "program": None, "agent": "planner-1"}
    with state.lock:
        return state.add_scored_locked(row, "planner-1", None)


def test_scored_row_adds_a_curve_point_with_hover_fields():
    state = fake_state()
    assert scored(state, "aa", 0.3) is True
    assert scored(state, "bb", 0.2) is False
    assert scored(state, "cc", 0.9, status="collapsed") is False
    first, second, third = state.snapshot()["curve"]
    assert first["eval"] == 1 and first["best"] == 0.3 and first["score"] == 0.3
    assert first["program_hash"] == "aa" and first["name"] == "d-aa"
    assert first["agent"] == "planner-1" and first["status"] == "ok"
    assert first["scores"] == {"gsm8k": 0.3, "if_guard": 0.5}
    assert first["is_best"] is True
    assert second["is_best"] is False and second["best"] == 0.3
    assert third["status"] == "collapsed" and third["is_best"] is False and third["best"] == 0.3
