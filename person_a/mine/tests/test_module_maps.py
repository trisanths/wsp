# CPU test of the maps.py entry fields on a synthetic engine result with hand-checked numbers:
# block_z, block_z_heldout, the disjoint-window z and the window counts, module_maps, module_share,
# the per-module held-out r with its layer count, selection_sources, reported_benchmark_skills, and
# an entry built from a result without module data.
#   .venv/bin/python mine/tests/test_module_maps.py
import copy
import importlib.util
import json
import math
import os
import sys
import tempfile
import types

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import maps  # noqa: E402
import maps_many  # noqa: E402

GENERAL = maps.REFERENCE
BASE = {"law|search": -1.0, "law|heldout": -1.2, GENERAL + "|search": -2.0, GENERAL + "|heldout": -2.0,
        "empty|search": None, "empty|heldout": None}
N = 8
# Loss increase (nats) per layer when the layer is skipped. Score = law increase - general increase.
#   search scores:  [0, 0.4, 0.8, 0.1, 0.05, 0.2, -0.1, 0]
#   heldout scores: [0, 0.1, 0.5, 0, 0, 0, 0, 0]
LAYER = {"law|search": [0.1, 0.5, 0.9, 0.2, 0.15, 0.3, 0.0, 0.1], GENERAL + "|search": [0.1] * N,
         "law|heldout": [0.2, 0.3, 0.7, 0.2, 0.2, 0.2, 0.2, 0.2], GENERAL + "|heldout": [0.2] * N,
         "empty|search": [None] * N, "empty|heldout": [None] * N}
# The same with only the attention output zeroed. Layer 3 has no attention.
#   search scores:  [0.2, 0, 0.4, null, -0.2, 0, 0.3, 0]
#   heldout scores: 2 * search + 0.1, so the Pearson r is 1
ATTN = {"law|search": [0.3, 0.2, 0.5, None, 0.1, 0.1, 0.4, 0.0],
        GENERAL + "|search": [0.1, 0.2, 0.1, None, 0.3, 0.1, 0.1, 0.0],
        "law|heldout": [0.5, 0.1, 0.9, None, -0.3, 0.1, 0.7, 0.1],
        GENERAL + "|heldout": [0.0, 0.0, 0.0, None, 0.0, 0.0, 0.0, 0.0],
        "empty|search": [None] * N, "empty|heldout": [None] * N}
# The same with only the feed-forward output zeroed. Layer 6 has no feed-forward module.
#   search scores:  [0.6, -0.2, 0.1, 0.4, 0.3, -0.1, null, 0]
#   heldout scores: [0.3, 0, 0.1, 0.5, 0.1, -0.1, null, 0]
MLP = {"law|search": [0.7, 0.1, 0.2, 0.6, 0.5, 0.0, None, 0.0],
       GENERAL + "|search": [0.1, 0.3, 0.1, 0.2, 0.2, 0.1, None, 0.0],
       "law|heldout": [0.3, 0.0, 0.1, 0.5, 0.1, -0.1, None, 0.0],
       GENERAL + "|heldout": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, None, 0.0],
       "empty|search": [None] * N, "empty|heldout": [None] * N}
PARTS = [{"attn": ["attn_heads"], "mlp": ["mlp_cluster"]}] * 3 + [{"attn": [], "mlp": ["mlp_cluster"]}] + \
        [{"attn": ["attn_heads"], "mlp": ["expert"]}] * 2 + [{"attn": ["attn_heads"], "mlp": []}] + \
        [{"attn": ["attn_heads"], "mlp": ["mlp_cluster", "expert"]}]
SOURCES = {"law": "some/law-set train: law text", GENERAL: "general text", "empty": "nothing the model can do"}
ROW = {"params_b": 7.0, "family": "toy", "license": "mit"}
OLD_KEYS = ["model", "skills", "skill_metric", "skill_details", "layer_maps", "expert_maps", "expert_details",
            "params_b", "layers", "hidden", "family", "architecture", "license", "structure", "method",
            "selection_data", "benchmark_free", "source", "revision", "gpu", "engine_seconds", "measured_at"]


def lps(increases):
    # what the engine returns: the mean answer log-prob with the part switched off, 5 places
    out = {}
    for key in increases:
        out[key] = [None if BASE[key] is None or v is None else round(BASE[key] - v, 5) for v in increases[key]]
    return out


def engine_result(modules):
    base = {}
    for key in BASE:
        base[key] = {"lp": BASE[key], "score": None if BASE[key] is None else round(math.exp(BASE[key]), 5)}
    res = {"base": base, "layer_lp": lps(LAYER), "expert_freq": None, "seconds": 12.5, "revision": "abc", "gpu": "EngineL4",
           "info": {"n_layers": N, "d_model": 16, "model_class": "Toy", "vocab": 32,
                    "layers_with": {"attn": 7, "mlp": 5, "experts": 3}, "example": {}}}
    if modules:
        res["attn_lp"] = lps(ATTN)
        res["mlp_lp"] = lps(MLP)
        res["module_parts"] = PARTS
    return res


def close(a, b, tol=1e-9):
    return a is not None and abs(a - b) <= tol


def test_window_z():
    search = [0.0, 0.4, 0.8, 0.1, 0.05, 0.2, -0.1, 0.0]
    # block [1,2]: mass 1.2. The six other windows of length 2 hold 0.4, 0.9, 0.15, 0.25, 0.2, 0.0:
    # sum 1.9, sum of squares 1.095, sample variance (1.095 - 1.9^2 / 6) / 5 = 0.098667, sd 0.314113
    z = (1.2 - 1.9 / 6) / math.sqrt((1.095 - 1.9 ** 2 / 6) / 5)
    assert abs(z - 2.812) < 1e-3
    assert close(maps.window_z(search, 1, 2), z, 6e-4)
    # block [4,5]: mass 0.25. Others 0.4, 1.2, 0.9, 0.15, 0.2, 0.0: sum 2.85, sum of squares 2.4725
    z = (0.25 - 2.85 / 6) / math.sqrt((2.4725 - 2.85 ** 2 / 6) / 5)
    assert abs(z - (-0.476)) < 1e-3
    assert close(maps.window_z(search, 4, 5), z, 6e-4)
    # negative scores count as 0: the single layer 6 holds no mass
    assert maps.window_z(search, 6, 6) < 0
    # heldout scores: block [1,2] holds 0.6, the others 0.1, 0.5, 0, 0, 0, 0: mean 0.1, sd 0.2
    held = [0.0, 0.1, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0]
    assert maps.window_z(held, 1, 2) == 2.5
    # block [4,5] holds 0; the others 0.1, 0.6, 0.5, 0, 0, 0: mean 0.2, sample variance 0.38 / 5
    assert close(maps.window_z(held, 4, 5), -0.2 / math.sqrt(0.076), 6e-4)
    assert abs(maps.window_z(held, 4, 5) - (-0.725)) < 1e-3
    # null cases: 2 other windows; exactly 3 other windows gives a number; no other window at all
    assert maps.window_z([0.5, 0.1, 0.0, 0.2], 0, 1) is None
    assert maps.window_z([0.5, 0.1, 0.0, 0.2, 0.3], 0, 1) is not None
    assert maps.window_z([0.5, 0.1, 0.0, 0.2], 0, 3) is None
    # null case: the other windows all hold the same mass (zero deviation)
    assert maps.window_z([1.0, 0.0, 0.0, 0.0, 0.0, 0.0], 0, 0) is None
    assert maps.window_z([1.0, 0.2, 0.2, 0.2, 0.2, 0.2], 0, 0) is None
    assert maps.block_z(search, [[1, 2], [4, 5]]) == [maps.window_z(search, 1, 2), maps.window_z(search, 4, 5)]
    assert maps.block_z(search, []) == []


def test_window_z_disjoint():
    search = [0.0, 0.4, 0.8, 0.1, 0.05, 0.2, -0.1, 0.0]
    # Windows of length 2 by start: 0.4, 1.2, 0.9, 0.15, 0.25, 0.2, 0.0.
    # Block [1,2] has 6 other windows; starts 0 and 2 share a layer with it, so 4 are disjoint:
    # 0.15, 0.25, 0.2, 0.0: mean 0.15, squared deviations 0 + 0.01 + 0.0025 + 0.0225 = 0.035
    assert maps.block_window_counts(search, [[1, 2], [4, 5]]) == ([6, 6], [4, 4])
    z = (1.2 - 0.15) / math.sqrt(0.035 / 3)
    assert abs(z - 9.721) < 1e-3
    assert close(maps.window_z_disjoint(search, 1, 2), z, 6e-4)
    # Block [4,5]: starts 3 and 5 overlap it. Disjoint: 0.4, 1.2, 0.9, 0.0: mean 0.625,
    # squared deviations 0.050625 + 0.330625 + 0.075625 + 0.390625 = 0.8475
    z = (0.25 - 0.625) / math.sqrt(0.8475 / 3)
    assert abs(z - (-0.706)) < 1e-3
    assert close(maps.window_z_disjoint(search, 4, 5), z, 6e-4)
    # the overlapping windows change the value: the plain z of the same blocks is 2.812 and -0.476
    assert maps.window_z_disjoint(search, 1, 2) != maps.window_z(search, 1, 2)
    held = [0.0, 0.1, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0]
    # heldout, block [1,2]: the 4 disjoint windows all hold 0, so there is no deviation
    assert maps.window_z_disjoint(held, 1, 2) is None
    # heldout, block [4,5] holds 0; disjoint 0.1, 0.6, 0.5, 0.0: mean 0.3, squared deviations 0.26
    assert close(maps.window_z_disjoint(held, 4, 5), -0.3 / math.sqrt(0.26 / 3), 6e-4)
    assert abs(maps.window_z_disjoint(held, 4, 5) - (-1.019)) < 1e-3
    # 5 layers, block [1,2]: 3 other windows give block_z a number, but only start 3 is disjoint
    five = [0.5, 0.1, 0.0, 0.2, 0.3]
    assert maps.window_z(five, 1, 2) is not None and maps.window_z_disjoint(five, 1, 2) is None
    assert maps.block_window_counts(five, [[1, 2]]) == ([3], [1])
    # block [0,0] of 5 layers: the 4 other single layers are all disjoint, so the two z are the same
    assert maps.block_window_counts(five, [[0, 0]]) == ([4], [4])
    assert maps.window_z_disjoint(five, 0, 0) == maps.window_z(five, 0, 0) is not None
    assert maps.block_window_counts(five, [[0, 1]]) == ([3], [2]) and maps.window_z_disjoint(five, 0, 1) is None
    assert maps.block_z_disjoint(search, [[1, 2], [4, 5]]) == [9.721, -0.706]
    assert maps.block_z_disjoint(search, []) == [] and maps.block_window_counts(search, []) == ([], [])


def test_module_share_and_r():
    share = maps.module_share([0.2, 0.0, 0.4, None, -0.2, 0.0, 0.3, 0.0, None],
                              [0.6, -0.2, 0.1, 0.4, 0.3, -0.1, None, 0.0, None])
    # 0.2 / 0.8; nothing positive; 0.4 / 0.5; no attention (not a measured 0); attention negative
    # and feed-forward positive (a measured 0); nothing positive; no feed-forward; both zero; both missing
    assert share == [0.25, None, 0.8, None, 0.0, None, None, None, None], share
    # a layer without attention never reads as "no attention share", whatever the other score is,
    # and a score one rounding step above zero does not make a share of 1
    assert maps.module_share([None, None, 0.00001], [0.4, -0.2, -0.3]) == [None, None, None]
    # the floor: four log-probs rounded to 5 places can move a score by 2e-5
    assert maps.SCORE_RESOLUTION == 4 * 0.5e-5
    assert maps.module_share([0.00002, 0.00003, 0.00001, 0.3], [0.0, 0.0, 0.3, 0.00002]) == [None, 1.0, 0.0, 1.0]
    # layers where one list has no number are left out: pairs (1,1), (2,3), (3,2) give r = 0.5
    assert maps.pearson_present([1.0, None, 2.0, 3.0], [1.0, 9.0, 3.0, 2.0]) == (0.5, 3)
    assert maps.pearson_present([None, None], [1.0, 2.0]) == (None, 0)
    assert maps.pearson_present([1.0, 1.0, 1.0], [1.0, 2.0, 3.0]) == (None, 3)
    # a minimum number of layers: 2 layers always give r = 1 or -1, so they give no r here
    assert maps.MIN_R_LAYERS == 4
    assert maps.pearson_present([0.1, None, 0.3, None], [0.2, None, 0.1, None]) == (-1.0, 2)
    assert maps.pearson_present([0.1, None, 0.3, None], [0.2, None, 0.1, None], maps.MIN_R_LAYERS) == (None, 2)
    assert maps.pearson_present([1.0, None, 2.0, 3.0], [1.0, 9.0, 3.0, 2.0], maps.MIN_R_LAYERS) == (None, 3)
    assert maps.pearson_present([1.0, 2.0, 3.0, 5.0], [2.0, 4.0, 6.0, 10.0], maps.MIN_R_LAYERS) == (1.0, 4)


def test_entry_with_modules():
    res = engine_result(True)
    entry = maps.build_entry("org/toy", res, SOURCES, ROW, {"law": {"dataset": "x"}})
    entry = json.loads(json.dumps(entry))  # what a reader of the file gets
    lm = entry["layer_maps"]["law"]
    assert lm["scores"] == [0.0, 0.4, 0.8, 0.1, 0.05, 0.2, -0.1, 0.0], lm["scores"]
    assert lm["influence"] == [0.1, 0.5, 0.9, 0.2, 0.15, 0.3, 0.0, 0.1]
    # positive score 1.55; [1,2] holds 1.2 (77.42%), then [4,5] holds 0.25 (16.13%)
    assert lm["blocks"] == [[1, 2], [4, 5]] and lm["block_mass"] == [0.7742, 0.1613], lm
    assert lm["block_z"] == [2.812, -0.476], lm["block_z"]
    assert lm["block_z_heldout"] == [2.5, -0.725], lm["block_z_heldout"]
    assert lm["heldout"]["check_mass"] == [1.0, 0.0]
    assert abs(lm["heldout"]["split_half_r"] - 0.9225) < 2e-4
    assert lm["block_z_windows"] == [6, 6] and lm["block_z_disjoint_windows"] == [4, 4]
    assert lm["block_z_disjoint"] == [9.721, -0.706], lm["block_z_disjoint"]
    assert lm["block_z_disjoint_heldout"] == [None, -1.019], lm["block_z_disjoint_heldout"]
    assert sorted(lm.keys()) == ["block_mass", "block_z", "block_z_disjoint", "block_z_disjoint_heldout",
                                 "block_z_disjoint_windows", "block_z_heldout", "block_z_windows", "blocks",
                                 "heldout", "influence", "scores"]
    assert sorted(lm["heldout"].keys()) == ["blocks", "check_mass", "select_mass", "split_half_r"]

    assert sorted(entry["module_maps"].keys()) == ["law"]  # not the reference, not a skill with no base score
    mm = entry["module_maps"]["law"]
    assert mm["attn"] == [0.2, 0.0, 0.4, None, -0.2, 0.0, 0.3, 0.0], mm["attn"]
    assert mm["mlp"] == [0.6, -0.2, 0.1, 0.4, 0.3, -0.1, None, 0.0], mm["mlp"]
    # layer 3 has no attention and layer 6 no feed-forward: no share there
    assert mm["module_share"] == [0.25, None, 0.8, None, 0.0, None, None, None], mm["module_share"]
    # attention: heldout = 2 * search + 0.1 on the 7 layers that have attention
    assert mm["heldout"]["split_half_r"]["attn"] == 1.0
    # feed-forward, 7 layers: cov 0.43 - 1.1 * 0.9 / 7, variances 0.67 - 1.21 / 7 and 0.37 - 0.81 / 7
    r = (0.43 - 0.99 / 7) / math.sqrt((0.67 - 1.21 / 7) * (0.37 - 0.81 / 7))
    assert abs(r - 0.8116) < 2e-4
    assert abs(mm["heldout"]["split_half_r"]["mlp"] - r) < 6e-5
    assert mm["heldout"]["n_layers"] == {"attn": 7, "mlp": 7}
    assert sorted(mm["heldout"].keys()) == ["n_layers", "split_half_r"]
    assert sorted(mm.keys()) == ["attn", "heldout", "mlp", "module_share"]
    assert entry["module_parts"] == PARTS
    assert "attention output" in entry["module_method"]
    assert entry["skills"]["empty"] is None and "empty" not in entry["layer_maps"]
    assert entry["selection_sources"] == {"law": {"dataset": "x"}}
    assert entry["reported_benchmark_skills"] is None  # the record does not say

    # a model with attention in 3 layers only (5, 6, 7): too few for an r, and the entry says how many
    res = engine_result(True)
    for key in res["attn_lp"]:
        for l in range(5):
            res["attn_lp"][key][l] = None
    mm = maps.build_entry("org/toy", res, SOURCES, ROW)["module_maps"]["law"]
    assert mm["attn"][:5] == [None] * 5 and mm["attn"][5:] == [0.0, 0.3, 0.0]
    assert mm["heldout"]["split_half_r"]["attn"] is None and mm["heldout"]["n_layers"] == {"attn": 3, "mlp": 7}
    assert mm["heldout"]["split_half_r"]["mlp"] is not None
    # layer 5 has both modules but no positive score, layer 6 has no feed-forward, layer 7 has zeros
    assert mm["module_share"] == [None] * 8


def test_merge_of_module_calls():
    # one call for all layers, and the same layers from three calls that each bring their own base
    whole = maps.module_increases(engine_result(True))
    assert whole["layers"] == list(range(N)) and whole["parts"] == PARTS
    assert abs(whole["attn"]["law|search"][2] - 0.5) < 1e-9 and whole["attn"]["law|search"][3] is None
    assert whole["mlp"]["empty|search"] is None  # no base score
    merged = None
    for shift, layers in [(0.0, [0, 1, 2]), (0.25, [6, 7]), (-0.5, [3, 4, 5])]:
        res = engine_result(True)
        res["layers"] = layers
        for key in res["base"]:
            if res["base"][key]["lp"] is not None:
                res["base"][key]["lp"] = round(res["base"][key]["lp"] + shift, 5)
        for module in ["attn_lp", "mlp_lp"]:
            for key in res[module]:
                # another container: every log-prob of the call moves with its base; the layers
                # that the call was not asked for are null
                res[module][key] = [None if v is None or l not in layers else round(v + shift, 5)
                                    for l, v in enumerate(res[module][key])]
        part = maps.module_increases(res)
        assert part["layers"] == layers
        for l in range(N):
            if l not in layers:
                assert part["mlp"]["law|search"][l] is None
        merged = maps.merge_increases(merged, part)
    assert merged["layers"] == list(range(N))
    for module in ["attn", "mlp"]:
        for key in whole[module]:
            if whole[module][key] is None:
                assert merged[module][key] is None
                continue
            for l in range(N):
                a, b = whole[module][key][l], merged[module][key][l]
                assert (a is None and b is None) or abs(a - b) < 1e-9, (module, key, l, a, b)
    assert maps.module_map(merged, "law") == maps.module_map(whole, "law")
    assert list(maps.module_fields(merged, SOURCES)["module_maps"].keys()) == ["law"]


def test_entry_without_module_data():
    plain = maps.build_entry("org/toy", engine_result(False), SOURCES, ROW)
    assert sorted(plain.keys()) == sorted(OLD_KEYS + ["selection_sources", "reported_benchmark_skills"]), \
        sorted(plain.keys())
    assert plain["selection_sources"] is None and plain["reported_benchmark_skills"] is None
    # the old fields keep their values: the same as in an entry that also has module data
    full = maps.build_entry("org/toy", engine_result(True), SOURCES, ROW)
    for key in OLD_KEYS:
        if key != "measured_at":
            assert plain[key] == full[key], key
    assert plain["selection_data"] == SOURCES and plain["selection_data"]["law"] == "some/law-set train: law text"
    assert plain["layer_maps"]["law"]["block_z"] == [2.812, -0.476]
    assert plain["layers"] == N and plain["hidden"] == 16 and plain["benchmark_free"] is True
    # a result that has the module lists but not this skill gives no module map for it
    res = engine_result(True)
    del res["attn_lp"]["law|heldout"]
    assert maps.module_map(maps.module_increases(res), "law") is None
    assert "law" not in maps.build_entry("org/toy", res, SOURCES, ROW)["module_maps"]
    assert not maps.has_module_data(engine_result(False))


def test_old_entry_is_still_read():
    # an entry as maps.py wrote it before these fields: every reader uses the fields it knows
    old = maps.build_entry("org/toy", engine_result(False), SOURCES, ROW)
    del old["selection_sources"]
    del old["reported_benchmark_skills"]
    for skill in old["layer_maps"]:
        for field in ["block_z", "block_z_heldout", "block_z_windows", "block_z_disjoint_windows",
                      "block_z_disjoint", "block_z_disjoint_heldout"]:
            del old["layer_maps"][skill][field]
    old = json.loads(json.dumps(old))
    assert sorted(old.keys()) == sorted(OLD_KEYS)
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "org__toy.json")
        assert maps.entry_done(path, False) is False and maps.entry_done(path, True) is False
        with open(path, "w") as f:
            json.dump(old, f)
        # done for a plain run, not done for a --modules run
        assert maps.entry_done(path, False) is True and maps.entry_done(path, True) is False
        with open(path, "w") as f:
            json.dump(maps.build_entry("org/toy", engine_result(True), SOURCES, ROW), f)
        assert maps.entry_done(path, False) is True and maps.entry_done(path, True) is True
        with open(path, "w") as f:
            f.write("{ not json")
        assert maps.entry_done(path, True) is False
    # the library builder reads block entries from a new entry the same way as from an old one
    script = os.path.join(maps.REPO, "scripts", "library_from_maps.py")
    if os.path.exists(script):
        spec = importlib.util.spec_from_file_location("library_from_maps_by_path", script)
        reader = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(reader)
        new = json.loads(json.dumps(maps.build_entry("org/toy", engine_result(True), SOURCES, ROW)))
        assert reader.model_facts(new) == reader.model_facts(old)
        assert reader.skill_text(new, "law") == "some/law-set train: law text"
    else:
        print("note: scripts/library_from_maps.py not found, reader check skipped")


def test_selection_sources():
    assert maps.dataset_id("cais/mmlu all test (subjects: a, b)") == "cais/mmlu"
    assert maps.dataset_id("ise-uiuc/Magicoder-OSS-Instruct-75K train (lang=rust)") == "ise-uiuc/magicoder-oss-instruct-75k"
    assert maps.dataset_id("") is None and maps.dataset_id(None) is None
    fake = types.SimpleNamespace(
        SOURCES={"math": [("Org/Math-Set", None, "train", "qa")], "code": [("org/code", "x", "train", "alpaca")],
                 "second": [("org/code", None, "test", "alpaca")]},
        BANNED=["org/bench", "bench"])
    assert maps.source_record("org/math-set train", "sums", fake) == {
        "dataset": "org/math-set train", "description": "sums", "pool": "math", "reported_benchmark": False}
    # first pool that has the dataset; a config or split in the text does not matter
    assert maps.source_record("Org/Code other test", "d", fake)["pool"] == "code"
    rec = maps.source_record("Org/Bench all test (subjects: x)", "questions", fake)
    assert rec["pool"] is None and rec["reported_benchmark"] is True
    rec = maps.source_record("org/unknown train", "d", fake)
    assert rec["pool"] is None and rec["reported_benchmark"] is False
    # healdata not found: not known, which is not the same as "no"
    rec = maps.source_record("org/bench test", "d", None)
    assert rec == {"dataset": "org/bench test", "description": "d", "pool": None, "reported_benchmark": None}
    assert maps.load_healdata(os.path.join(HERE, "no_such_file.py")) is None

    # the real files of this repo
    healdata = maps.load_healdata()
    assert healdata is not None, "wsp/healdata.py not found at " + maps.HEALDATA
    assert "wsp" not in sys.modules  # loaded by path, the wsp package is not imported
    details = maps.load_selection_sources(healdata)
    payload, sources = maps.load_skills()
    assert sorted(details.keys()) == sorted(sources.keys()) == sorted(payload["skills"].keys())
    with open(os.path.join(os.path.dirname(HERE), "data", "pool_skills.json")) as f:
        pool = json.load(f)
    for name in details:
        assert sorted(details[name].keys()) == ["dataset", "description", "pool", "reported_benchmark"], name
        assert details[name]["pool"] is None or details[name]["pool"] in healdata.SOURCES
        assert details[name]["reported_benchmark"] in [True, False]
        assert isinstance(sources[name], str)  # selection_data stays a string per skill
        if name in pool:
            assert sources[name] == pool[name]["dataset"] + ": " + pool[name]["description"]
            assert details[name]["dataset"] == pool[name]["dataset"]
            assert details[name]["reported_benchmark"] == (maps.dataset_id(pool[name]["dataset"]) in healdata.BANNED)
    expected = {"math": ("math", False), "code": ("code", False), GENERAL: ("prose", False),
                "medicine": ("medicine", False), "instruction": ("general", False), "code_rust": ("code", False),
                "physics": (None, True), "law": (None, True), "prose_de": (None, False), "sql": (None, False)}
    for name in expected:
        got = (details[name]["pool"], details[name]["reported_benchmark"])
        assert got == expected[name], (name, got)
    assert details["math"]["dataset"].split()[0] == "microsoft/orca-math-word-problems-200k"

    # one place that names the skills from a reported dataset; benchmark_free stays as it was
    assert maps.reported_skills(None) is None
    assert maps.reported_skills({"a": {"reported_benchmark": True}, "b": {"reported_benchmark": False},
                                 "c": {"reported_benchmark": True}}) == ["a", "c"]
    assert maps.reported_skills({"a": {"reported_benchmark": False}}) == []
    assert maps.reported_skills({"a": {"reported_benchmark": True}, "b": {"reported_benchmark": None}}) is None
    reported = maps.reported_skills(details)
    assert reported == sorted(n for n in details if details[n]["reported_benchmark"])
    assert "physics" in reported and "law" in reported and "math" not in reported and GENERAL not in reported
    for name in reported:
        assert maps.dataset_id(details[name]["dataset"]) in healdata.BANNED
    entry = maps.build_entry("org/toy", engine_result(False), SOURCES, ROW, {
        "law": {"dataset": "cais/mmlu all test", "description": "d", "pool": None, "reported_benchmark": True},
        GENERAL: {"dataset": "w", "description": "d", "pool": "prose", "reported_benchmark": False}})
    assert entry["reported_benchmark_skills"] == ["law"] and entry["benchmark_free"] is True


def test_maps_many_command():
    plain = maps_many.maps_command("Org__Model", False)
    assert plain[-3:] == [os.path.join(os.path.dirname(HERE), "maps.py"), "--zoo-key", "Org__Model"]
    assert "--modules" not in plain
    assert maps_many.maps_command("Org__Model", True) == plain + ["--modules"]


if __name__ == "__main__":
    before = copy.deepcopy(SOURCES)
    test_window_z()
    test_window_z_disjoint()
    test_module_share_and_r()
    test_entry_with_modules()
    test_merge_of_module_calls()
    test_entry_without_module_data()
    test_old_entry_is_still_read()
    test_selection_sources()
    test_maps_many_command()
    assert SOURCES == before
    print("ok")
