# Contract test between the local callers and the engine methods. The callers pass their arguments by
# position (Miner.call(method, *args) -> engine.<method>.remote(*args)). The engine tests call the
# methods by keyword, and the caller tests use a fake engine, so neither notices when the order of two
# arguments differs between the two sides: the engine would rank 24 units on 1024 items with no error.
# This test records the argument tuples that the real callers build and binds each one to the
# signature of the real engine method.
#   .venv/bin/python mine/tests/test_engine_contract.py
# No engine call: the miner's call() is a recorder with synthetic answers.
import inspect
import os
import shutil
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
import client  # noqa: E402
import common  # noqa: E402
import engine_app  # noqa: E402
import maps  # noqa: E402
import sweep  # noqa: E402
import test_sweep  # noqa: E402

# Positional signatures that deployed callers rely on. An old client keeps calling the engine after a
# deploy, so an existing method keeps its argument order, and new arguments go to the end.
SIGNATURES = {
    "info_summary": [],
    "base": ["payload"],
    "ablate": ["payload", "part"],
    "skill_maps": ["payload", "routing_items", "modules", "layers", "plain"],
    "random_drops": ["payload", "part", "target", "n", "seed"],
    "activation_scan": ["payload", "kind", "skills", "layers", "n_items"],
    "attribution_scan": ["payload", "kind", "skills", "layers", "n_items", "token_budget"],
    "builtin_info": [],
    "expert_pick_rates": ["payload", "names", "layers", "n_items"],
    "confirm_math": ["part", "n_items", "max_new_tokens"],
    "attribution_units": ["payload", "kind", "skill", "layers", "top_n", "n_items", "token_budget"],
    "activation_units": ["payload", "kind", "skill", "layers", "top_n", "n_items"],
}
# the tier classes that mine/common.py can pick: each must exist in the engine file
TIERS = ["EngineL4", "EngineL40S", "EngineH100", "EngineH200", "EngineH200x2", "EngineH200x4", "EngineH200x8",
         "EngineB200x8"]


def raw(name):
    method = engine_app.EngineBase.__dict__[name]
    return method.raw_f if hasattr(method, "raw_f") else method._get_raw_f()


def bind(method, args):
    # the arguments as the engine method would receive them, by name
    bound = inspect.signature(raw(method)).bind(None, *args)
    bound.apply_defaults()
    return dict(bound.arguments)


def test_signatures():
    for name in SIGNATURES:
        params = list(inspect.signature(raw(name)).parameters)[1:]
        assert params == SIGNATURES[name], (name, params)
    for short in ["a", "b"]:
        for params_b in [1, 10, 30, 60, 500]:
            assert common.pick_gpu(params_b, short) in TIERS
    for tier in list(common.TIER_OVERRIDE.values()) + TIERS:
        assert hasattr(engine_app, tier), tier


def test_sweep_calls(specs):
    real_root = common.ROOT
    old_lane = os.environ.get("WSP_LANE")
    tmp = tempfile.mkdtemp(prefix="contract-")
    common.ROOT = tmp
    os.environ["WSP_LANE"] = "sweep"
    try:
        short = "toy"
        common.write_json_atomic(common.state_path(short), {
            "revision": "rev-main", "model": "org/Toy", "structure": {"n_layers": 12, "d_model": 64, "layers_detail": []}})
        m, engine = test_sweep.make_miner(short, specs)
        guards = m.guards("law", list(specs.keys()))
        m.base_margins(["law", "general_text"])

        # the rankings: SweepMiner.rank passes (payload, kind, skill, layers, top_n, n_items[, token_budget])
        engine.rank_result = {"layers": {"0": test_sweep.layer_rank(64, dict((u, 0.1) for u in range(20)))}}
        m.rank("law", "mlp_cluster", 8, 24)
        got = bind("attribution_units", [c for c in engine.calls if c[0] == "attribution_units"][-1][1])
        assert got["kind"] == "mlp_cluster" and got["skill"] == "law" and got["layers"] == [], got
        assert got["top_n"] == 8 and got["n_items"] == 24, got
        assert got["token_budget"] == common.scan_token_budget(m.params_b), got
        assert sorted(got["payload"]["skills"]) == ["general_text", "law"]
        engine.quantized = True
        del m.state["sweep_method"]
        m.rank("law", "expert", 4, 16)
        got = bind("activation_units", [c for c in engine.calls if c[0] == "activation_units"][-1][1])
        assert got["kind"] == "expert" and got["skill"] == "law" and got["layers"] == [], got
        assert got["top_n"] == 4 and got["n_items"] == 16, got

        # one screen of an expert part and its row: ablate, random_drops, builtin_info, expert_pick_rates
        c = {"skill": "law", "kind": "expert", "size": 2, "layer": 3, "rank_method": "activation", "rank_score": 0.5,
             "part": {"kind": "expert", "layer": 3, "indices": [1, 2]}}
        engine.drops = {"law": 0.30}
        res = m.screen_candidate(c, guards, 2.0, 24)
        assert res["passed"] and res["catalog_id"], res
        by_method = {}
        for method, args in engine.calls:
            by_method.setdefault(method, []).append(bind(method, args))
        assert sorted(by_method) == ["ablate", "activation_units", "attribution_units", "base", "builtin_info",
                                     "expert_pick_rates", "random_drops"], sorted(by_method)
        assert by_method["ablate"][-1]["part"] == c["part"] and "law" in by_method["ablate"][-1]["payload"]["skills"]
        got = by_method["random_drops"][-1]
        assert got["part"] == c["part"] and got["target"] == "law" and got["n"] == client.GATE_N_RANDOM, got
        assert isinstance(got["seed"], int)
        got = by_method["expert_pick_rates"][-1]
        assert got["names"] == ["law", "general_text"] and got["layers"] == [3], got
        assert got["n_items"] == client.ROUTING_ITEMS, got
        assert sorted(by_method["base"][-1]["payload"]["skills"]) == ["general_text", "law"]
        assert engine.unknown == [], engine.unknown
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
        common.ROOT = real_root
        if old_lane is None:
            del os.environ["WSP_LANE"]
        else:
            os.environ["WSP_LANE"] = old_lane


def test_agent_and_map_calls():
    # The agent tools and maps.py call these with literal arguments. The same calls, bound here:
    payload = {"skills": {}}
    got = bind("activation_scan", (payload, "mlp_cluster", ["math"], [1, 2]))          # agent.py
    assert got["skills"] == ["math"] and got["layers"] == [1, 2] and got["n_items"] == 32, got
    got = bind("attribution_scan", (payload, "attn_heads", ["math"], [], 24, 2048))    # agent.py
    assert got["n_items"] == 24 and got["token_budget"] == 2048, got
    got = bind("confirm_math", ({"kind": "block", "layers": [1]},))                    # client.Miner.confirm
    assert got["n_items"] == 40, got
    got = bind("skill_maps", (payload,))                                               # maps.py, plain call
    assert got["routing_items"] == 64 and got["modules"] is False and got["layers"] is None and got["plain"] is True
    # maps.py --modules passes these by keyword
    params = inspect.signature(raw("skill_maps")).parameters
    for name in ["modules", "layers", "plain"]:
        assert name in params
    assert maps.CALL_TIMEOUT == engine_app.COMMON["timeout"]


if __name__ == "__main__":
    test_signatures()
    test_sweep_calls(sweep.load_skill_specs())
    test_agent_and_map_calls()
    print("ok")
