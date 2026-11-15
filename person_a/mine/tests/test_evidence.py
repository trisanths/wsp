# Checks the z-score, guard, selection_data and routing math of mine/evidence.py with numbers worked by hand.
#   .venv/bin/python mine/tests/test_evidence.py
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import common  # noqa: E402
import evidence  # noqa: E402


def close(a, b, tol=1e-6):
    return a is not None and abs(a - b) <= tol


def test_z():
    # randoms 0.1, 0.2, 0.3: mean 0.2, sample deviation sqrt((0.01 + 0 + 0.01) / 2) = 0.1; (0.5 - 0.2) / 0.1 = 3
    out = evidence.z_fields({"t|search": 0.5, "t|heldout": 0.4}, [0.1, 0.2, 0.3], "t")
    assert out["z"] == 3.0, out
    assert out["random_n"] == 3 and close(out["random_mean"], 0.2) and close(out["random_std"], 0.1), out
    assert out["z_heldout"] is None and out["random_heldout"] is None, out  # a flat list has no held-out drops

    # a missing value is left out: 0.1 and 0.3 give mean 0.2, deviation sqrt(0.02) = 0.141421; 0.3 / 0.141421 = 2.121
    out = evidence.z_fields({"t|search": 0.5}, [0.1, None, 0.3], "t")
    assert out["random_n"] == 2 and close(out["random_std"], 0.141421) and out["z"] == 2.121, out

    # null cases: one random drop, none, all equal, no target drop
    out = evidence.z_fields({"t|search": 0.5}, [0.1], "t")
    assert out["z"] is None and out["random_n"] == 1 and close(out["random_mean"], 0.1) and out["random_std"] is None, out
    out = evidence.z_fields({"t|search": 0.5}, [], "t")
    assert out["z"] is None and out["random_n"] == 0 and out["random_mean"] is None and out["random_std"] is None, out
    out = evidence.z_fields({"t|search": 0.5}, [0.1, 0.1, 0.1], "t")
    assert out["z"] is None and out["random_std"] == 0.0 and close(out["random_mean"], 0.1), out
    out = evidence.z_fields({"t|search": None}, [0.1, 0.2, 0.3], "t")
    assert out["z"] is None and out["random_n"] == 3, out
    out = evidence.z_fields({}, [0.1, 0.2, 0.3], "t")
    assert out["z"] is None, out

    # both halves: search 0.0, 0.2 (mean 0.1, deviation 0.141421), held-out 0.1, 0.3 (mean 0.2, same deviation)
    # (0.5 - 0.1) / 0.141421 = 2.828 and (0.6 - 0.2) / 0.141421 = 2.828
    effect = {"t|search": 0.5, "t|heldout": 0.6}
    for shape in [{"search": [0.0, 0.2], "heldout": [0.1, 0.3]},
                  [{"search": 0.0, "heldout": 0.1}, {"search": 0.2, "heldout": 0.3}]]:
        out = evidence.z_fields(effect, shape, "t")
        assert out["z"] == 2.828 and out["z_heldout"] == 2.828, out
        assert out["random_heldout"]["n"] == 2 and close(out["random_heldout"]["mean"], 0.2), out
    # held-out z stays null when the held-out half has one random drop
    out = evidence.z_fields(effect, {"search": [0.0, 0.2], "heldout": [0.1]}, "t")
    assert out["z"] == 2.828 and out["z_heldout"] is None and out["random_heldout"]["std"] is None, out

    # a stored row (qwenmath1.5b, fraction_role): target 0.1593, randoms below. Mean 0.00212. Squared
    # deviations sum to 1.548e-6; over 4 gives 3.87e-7; root 6.2209e-4. (0.1593 - 0.00212) / 6.2209e-4 = 252.66
    out = evidence.z_fields({"fraction_role|search": 0.1593}, [0.0023, 0.0031, 0.0016, 0.002, 0.0016], "fraction_role")
    assert close(out["random_mean"], 0.00212) and close(out["random_std"], 0.000622), out
    assert abs(out["z"] - 252.66) < 0.02, out


def test_guard():
    effect = {"t|search": 0.4, "t__control|search": -0.3, "a|search": 0.02, "b|search": -0.07,
              "b__control|search": 0.05, "t|heldout": 0.5, "t__control|heldout": 0.9, "a|heldout": 0.01,
              "b|heldout": None}
    g = evidence.guard(effect, "t")
    # the target (0.4) and its own control (-0.3, 0.9) do not count; a rise counts as much as a drop
    assert g["search"] == {"skill": "b", "drop": -0.07, "abs": 0.07}, g
    assert g["heldout"] == {"skill": "a", "drop": 0.01, "abs": 0.01}, g
    # another skill's control counts
    g = evidence.guard({"t|search": 0.4, "a|search": 0.02, "a__control|search": 0.09}, "t")
    assert g["search"]["skill"] == "a__control" and g["heldout"] is None, g
    # nothing but the target: no guard value
    assert evidence.guard({"t|search": 0.4, "t__control|search": 0.1}, "t") == {"search": None, "heldout": None}
    # equal sizes: the same skill on every run
    assert evidence.guard({"x|search": 0.05, "a|search": -0.05}, "t")["search"]["skill"] == "a"
    fields = evidence.signature_fields(effect, [0.1, 0.2, 0.3], "t")
    assert fields["guard"]["search"]["skill"] == "b" and "z" in fields and "random_n" in fields


# an old build_data.py in small: the builders list names each skill's function and description
OLD_SOURCE = '''
def load(*args, **kwargs):
    pass

def build_math(rng):
    gsm = load("openai/gsm8k", "main")

def build_code(rng):
    data = load("google-research-datasets/mbpp", "full")

def build_general_text(rng):
    data = load("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")

def main():
    builders = [("math", build_math, "GSM8K word problems; digit tokens of the worked solution"),
                ("code", build_code, "MBPP Python tasks; tokens of the reference solution"),
                ("general_text", build_general_text, "WikiText paragraph continuation (control only)")]
'''
OLD_DESCRIBED = {"math": {"description": "GSM8K word problems; digit tokens of the worked solution"},
                 "code": {"description": "MBPP Python tasks; tokens of the reference solution"},
                 "general_text": {"description": "WikiText paragraph continuation (control only)"}}
EFFECT = {"math|search": 0.3, "math|heldout": 0.3, "code|search": 0.01, "code|heldout": 0.0,
          "general_text|search": 0.0, "general_text|heldout": 0.0,
          "fraction_role|search": 0.2, "fraction_role__control|search": 0.0,
          "fraction_role|heldout": 0.2, "fraction_role__control|heldout": 0.0,
          "month_offset|search": 0.0, "month_offset__control|search": 0.0}
AGENT = {"fraction_role": {"path": "mine/skills/toy/fraction_role.json", "hash": "9a67",
                           "description": "numerator or denominator"},
         "month_offset": {"path": "mine/skills/toy/month_offset.json", "hash": "ac66",
                          "description": "month after an offset"}}


def test_selection_data():
    hd = evidence.load_healdata()
    assert hd is not None, "wsp/healdata.py must be readable from the team repo"
    # what the checks below rest on, read from wsp/healdata.py
    assert "openai/gsm8k" in hd.BANNED and "google-research-datasets/mbpp" in hd.BANNED and "cais/mmlu" in hd.BANNED
    assert "math" in hd.SOURCES and "code" in hd.SOURCES

    builders = evidence.builder_datasets(OLD_SOURCE)
    assert builders["math"] == {"description": OLD_DESCRIBED["math"]["description"], "datasets": ["openai/gsm8k"]}
    assert builders["code"]["datasets"] == ["google-research-datasets/mbpp"]
    builtin = evidence.builtin_sources(OLD_DESCRIBED, builders)
    lookup = evidence.make_lookup(AGENT, builtin, pool={})

    # an agent skill: its frozen test file and description; the built-in sets it was gated against were benchmarks
    sd = evidence.selection_data("fraction_role", EFFECT, lookup, hd, "engine")
    assert sd["skill"] == "fraction_role" and sd["source"] == "agent", sd
    assert sd["path"] == "mine/skills/toy/fraction_role.json" and sd["hash"] == "9a67", sd
    assert sd["description"] == "numerator or denominator" and sd["control"] == "fraction_role__control", sd
    assert sd["off_target"] == ["math", "code", "general_text", "month_offset"], sd  # no control sets, not the target
    assert sd["off_target_data"]["math"] == {"source": "builtin", "datasets": ["openai/gsm8k"],
                                             "description": OLD_DESCRIBED["math"]["description"]}, sd
    assert sd["off_target_data"]["month_offset"] == {"source": "agent", "hash": "ac66"}, sd
    assert sd["pool"] is None and sd["pool_match"] is None, sd
    assert sd["reported_benchmarks"] == ["openai/gsm8k", "google-research-datasets/mbpp"], sd
    assert sd["builtin_basis"] == "engine", sd

    # a built-in skill: the dataset description of the built-in file
    sd = evidence.selection_data("math", EFFECT, lookup, hd, "local file")
    assert sd["source"] == "builtin" and sd["description"] == OLD_DESCRIBED["math"]["description"], sd
    assert sd["datasets"] == ["openai/gsm8k"] and "path" not in sd and sd["control"] is None, sd
    assert sd["off_target"] == ["code", "general_text", "fraction_role", "month_offset"], sd
    assert sd["pool"] == "math" and sd["pool_match"] == "name", sd
    assert sd["reported_benchmarks"] == ["openai/gsm8k", "google-research-datasets/mbpp"], sd

    # the file was built by another build_data.py (descriptions differ): no dataset ids, so the
    # benchmark is found by its name in the description
    other = evidence.builtin_sources(OLD_DESCRIBED, {"math": {"description": "something else", "datasets": ["a/b"]}})
    assert other["math"]["datasets"] == [], other
    sd = evidence.selection_data("math", EFFECT, evidence.make_lookup(AGENT, other, pool={}), hd, "engine")
    assert sd["reported_benchmarks"] == ["openai/gsm8k", "google-research-datasets/mbpp"], sd
    # "Orca-Math" is one word: it is not the benchmark named "math"
    banned = [b.lower() for b in hd.BANNED]
    assert evidence.banned_in_text("Orca-Math word problems (approved pool); digit tokens", banned) == []
    assert evidence.banned_in_text("MMLU physics questions", banned) == ["cais/mmlu"]
    assert evidence.banned_match("CAIS/MMLU", banned)
    assert not evidence.banned_match("microsoft/orca-math-word-problems-200k", banned)

    # no healdata file: not checked, which is not the same as nothing found
    sd = evidence.selection_data("math", EFFECT, lookup, None, "local file")
    assert sd["pool"] is None and sd["reported_benchmarks"] is None and "not found" in sd["note"], sd

    # a name that is in no skill file
    sd = evidence.selection_data("math", {"math|search": 0.3, "ghost|search": 0.0}, lookup, hd, "engine")
    assert sd["off_target_data"]["ghost"]["source"] == "unknown", sd


def test_real_files():
    # the built-in and pool files of this checkout, with nothing about them typed in here
    hd = evidence.load_healdata()
    data = evidence.read_json(common.BUILTIN_SKILLS)
    builtin = evidence.local_builtin()
    with open(evidence.BUILD_DATA) as f:
        source = f.read()
    for name in data:
        assert builtin[name]["description"] == data[name]["description"], name
        assert builtin[name]["datasets"], "no dataset id found for built-in " + name
        for d in builtin[name]["datasets"]:
            assert '"' + d + '"' in source, d
    lookup = evidence.make_lookup({}, builtin)
    effect = {}
    for name in data:
        effect[name + "|search"] = 0.0
    sd = evidence.selection_data("math", effect, lookup, hd, "local file")
    assert sd["pool"] == "math" and sd["description"] == data["math"]["description"], sd
    assert isinstance(sd["reported_benchmarks"], list), sd
    pool_ids = {}
    for pool in hd.SOURCES:
        for src in hd.SOURCES[pool]:
            pool_ids[src[0].lower()] = pool
    banned = [b.lower() for b in hd.BANNED]
    pool = evidence.local_pool()
    for name in pool:
        sd = evidence.selection_data(name, {name + "|search": 0.1}, lookup, hd, "local file")
        assert sd["source"] == "pool" and sd["dataset"] == pool[name]["dataset"], sd
        ident = pool[name]["datasets"][0].lower()
        assert sd["reported_benchmarks"] == ([ident] if evidence.banned_match(ident, banned) else []), sd
        if name in hd.SOURCES:
            assert sd["pool"] == name and sd["pool_match"] == "name", sd
        elif ident in pool_ids:
            assert sd["pool"] == pool_ids[ident] and sd["pool_match"] == "dataset", sd
        elif name.split("_")[0] in hd.SOURCES:
            assert sd["pool"] == name.split("_")[0] and sd["pool_match"] == "name prefix", sd
        else:
            assert sd["pool"] is None, sd
    if "physics" in pool:  # an MMLU set: a reported benchmark, and no text pool for it
        sd = evidence.selection_data("physics", {"physics|search": 0.1}, lookup, hd, "local file")
        assert sd["reported_benchmarks"] == ["cais/mmlu"] and sd["pool"] is None, sd
    # an agent skill is matched by its whole name only
    agent = {"code_switching": {"path": "p", "hash": "h", "description": "d"}}
    sd = evidence.selection_data("code_switching", {"code_switching|search": 0.2},
                                 evidence.make_lookup(agent, builtin), hd, "engine")
    assert sd["pool"] is None, sd


def test_routing():
    # layer 3, 4 experts. Search half: the skill text picks expert 1 on 0.5 of its tokens and general text
    # on 0.25: ln(0.5 / 0.25) = ln 2 = 0.6931. Expert 2: ln(0.25 / 0.5) = -0.6931. Held-out: a rate of 0 gives no ratio.
    result = {"rates": {
        "sk|search": {"n_items": 2, "layers": {"3": {"rates": [0.0, 0.5, 0.25, 1.25], "tokens": 40}}},
        "general_text|search": {"n_items": 2, "layers": {"3": {"rates": [0.25, 0.25, 0.5, 1.0], "tokens": 60}}},
        "sk|heldout": {"n_items": 1, "layers": {"3": {"rates": [0.0, 0.3, 0.2, 1.5], "tokens": 20}}},
        "general_text|heldout": {"n_items": 1, "layers": {"3": {"rates": [0.5, 0.0, 0.2, 1.3], "tokens": 30}}}},
        "counted_layers": [3], "unsupported": {}, "missing": []}
    routing, note = evidence.routing_fields([2, 1], 3, "sk", result)
    assert note is None and routing["layer"] == 3 and routing["skill"] == "sk" and routing["general"] == "general_text"
    assert list(routing["experts"].keys()) == ["1", "2"], routing
    assert routing["experts"]["1"]["search"] == {"skill_rate": 0.5, "general_rate": 0.25, "log_ratio": 0.6931}, routing
    assert routing["experts"]["2"]["search"] == {"skill_rate": 0.25, "general_rate": 0.5, "log_ratio": -0.6931}, routing
    assert routing["experts"]["1"]["heldout"] == {"skill_rate": 0.3, "general_rate": 0.0, "log_ratio": None}, routing
    assert routing["experts"]["2"]["heldout"]["log_ratio"] == 0.0, routing
    assert routing["tokens"] == {"sk|search": 40, "general_text|search": 60, "sk|heldout": 20, "general_text|heldout": 30}
    assert routing["n_items"]["sk|search"] == 2 and routing["n_items"]["general_text|heldout"] == 1, routing

    # a layer the engine cannot count: no routing, and the note says why
    routing, note = evidence.routing_fields([0], 5, "sk", {"rates": {}, "unsupported": {"5": "module_list experts: x"}})
    assert routing is None and "module_list" in note, note
    routing, note = evidence.routing_fields([0], 5, "sk", {"rates": {}, "unsupported": {}})
    assert routing is None and "no pick rates" in note, note
    assert evidence.log_ratio(None, 0.2) is None and evidence.log_ratio(0.2, 0.2) == 0.0


test_z()
test_guard()
test_selection_data()
test_real_files()
test_routing()
print("ok")
