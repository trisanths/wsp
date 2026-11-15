# Checks the z-score, guard, selection_data and routing math of mine/evidence.py with numbers worked by hand.
#   .venv/bin/python mine/tests/test_evidence.py
import hashlib
import json
import os
import shutil
import sys
import tempfile

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
    # that row drew one value twice: 5 random drops, 4 distinct values
    assert out["random_n"] == 5 and out["random_distinct"] == 4 and out["z_flag"] is None, out

    # distinct random drops. The engine draws with replacement and rounds, so drops repeat.
    out = evidence.z_fields({"t|search": 0.5}, [0.1, 0.2, 0.3], "t")
    assert out["random_distinct"] == 3 and out["z_flag"] is None, out
    # 0.1, 0.1, 0.3: mean 0.166667; squared deviations 0.004444 + 0.004444 + 0.017778 = 0.026667; over 2
    # gives 0.013333; root 0.11547. (0.5 - 0.166667) / 0.11547 = 2.887. Two distinct values: z keeps its
    # number and gets the flag.
    out = evidence.z_fields({"t|search": 0.5}, [0.1, 0.1, 0.3], "t")
    assert out["random_n"] == 3 and out["random_distinct"] == 2 and close(out["random_std"], 0.11547), out
    assert out["z"] == 2.887 and out["z_flag"] == "fewer than 3 distinct random drops", out
    # a stored row (k2horizon7b, ordinal_word): 5 drops, 3 distinct values, no flag
    out = evidence.z_fields({"ordinal_word|search": 0.3}, [0.0, 0.0, 0.0061, -0.0061, -0.0061], "ordinal_word")
    assert out["random_n"] == 5 and out["random_distinct"] == 3 and out["z_flag"] is None, out
    # no z, no flag
    out = evidence.z_fields({"t|search": 0.5}, [0.1], "t")
    assert out["z"] is None and out["random_distinct"] == 1 and out["z_flag"] is None, out
    out = evidence.z_fields({"t|search": 0.5}, [0.1, 0.1, 0.1], "t")
    assert out["z"] is None and out["random_distinct"] == 1 and out["z_flag"] is None, out
    out = evidence.z_fields({"t|search": 0.5}, [], "t")
    assert out["random_distinct"] == 0 and out["z_flag"] is None, out


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

    # items_sha (the digest the engine's builtin_info reports) stays with each built-in set
    described = {"math": dict(OLD_DESCRIBED["math"], items_sha="aaaa"), "code": dict(OLD_DESCRIBED["code"], items_sha="bbbb")}
    with_sha = evidence.builtin_sources(described, builders)
    assert with_sha["math"]["items_sha"] == "aaaa" and "items_sha" not in builtin["math"], with_sha
    sd = evidence.selection_data("math", EFFECT, evidence.make_lookup(AGENT, with_sha, pool={}), hd, "engine")
    assert sd["items_sha"] == "aaaa" and sd["off_target_data"]["code"]["items_sha"] == "bbbb", sd
    data = {"math": {"description": "d", "items": {"search": [{"context": "1", "continuation": "2"}], "heldout": []}}}
    digest = hashlib.sha256(json.dumps(data["math"]["items"], sort_keys=True).encode()).hexdigest()[:16]
    assert evidence.describe_builtin(data)["math"] == {"description": "d", "dataset": None, "items_sha": digest}

    # an off-target agent skill of which two frozen tests are known: no hash, and the candidates
    two = dict(AGENT, month_offset={"path": "p", "hash": None, "description": "d", "hash_candidates": ["4073", "7062"]})
    sd = evidence.selection_data("fraction_role", EFFECT, evidence.make_lookup(two, builtin, pool={}), hd, "engine")
    assert sd["off_target_data"]["month_offset"] == {"source": "agent", "hash": None,
                                                     "hash_candidates": ["4073", "7062"]}, sd


def test_path_check():
    # is the frozen test file the test the row was selected on?
    folder = tempfile.mkdtemp(prefix="wsp_path_check_")
    path = os.path.join(folder, "product_sign.json")
    with open(path, "w") as f:
        json.dump({"name": "product_sign", "hash": "8245"}, f)
    assert evidence.path_check(path, "8245") == {"path_matches": True}
    out = evidence.path_check(path, "dfed")  # another lane replaced the file
    assert out["path_matches"] is False and out["path_hash"] == "8245" and "8245" in out["path_note"], out
    out = evidence.path_check(os.path.join(folder, "gone.json"), "dfed")
    assert out["path_matches"] is None and "could not be read" in out["path_note"], out
    with open(path, "w") as f:
        f.write("{not json")
    assert evidence.path_check(path, "8245")["path_matches"] is None
    shutil.rmtree(folder)


def test_merge_uncertain():
    # A row whose engine held the old built-in file (GSM8K, MBPP) or the new one (Orca-Math, CodeAlpaca):
    # the deploy time is not known, so neither is written as fact.
    hd = evidence.load_healdata()
    old = evidence.builtin_sources(OLD_DESCRIBED, evidence.builder_datasets(OLD_SOURCE))
    new = {"math": {"source": "builtin", "description": "Orca-Math word problems",
                    "datasets": ["microsoft/orca-math-word-problems-200k"]},
           "code": {"source": "builtin", "description": "CodeAlpaca Python tasks", "datasets": ["sahil2801/CodeAlpaca-20k"]},
           "general_text": old["general_text"]}
    labels = [{"commit": "old", "time_from": "commit", "since": "t0"}, {"commit": "new", "time_from": "commit", "since": "t1"}]
    basis = "git history, version uncertain"

    # an agent target: the two versions differ on the off-target sets math and code
    per = [evidence.selection_data("fraction_role", EFFECT, evidence.make_lookup(AGENT, b, pool={}), hd, basis)
           for b in [old, new]]
    assert per[0]["reported_benchmarks"] == ["openai/gsm8k", "google-research-datasets/mbpp"], per[0]
    assert per[1]["reported_benchmarks"] == [], per[1]
    sd = evidence.merge_uncertain(per, labels)
    assert sd["reported_benchmarks"] == ["openai/gsm8k", "google-research-datasets/mbpp"], sd  # the union
    assert sd["builtin_basis"] == basis and sd["description"] == "numerator or denominator", sd
    assert sd["off_target_data"]["math"] == {"source": "builtin", "description": None, "uncertain": True,
                                             "datasets": ["openai/gsm8k", "microsoft/orca-math-word-problems-200k"]}, sd
    assert sd["off_target_data"]["general_text"] == per[0]["off_target_data"]["general_text"], sd  # same in both
    assert sd["off_target_data"]["month_offset"] == {"source": "agent", "hash": "ac66"}, sd
    assert [c["commit"] for c in sd["builtin_candidates"]] == ["old", "new"], sd
    assert sd["builtin_candidates"][0]["skills"]["math"] == {"description": OLD_DESCRIBED["math"]["description"],
                                                             "datasets": ["openai/gsm8k"]}, sd
    assert sd["builtin_candidates"][1]["skills"]["code"]["datasets"] == ["sahil2801/CodeAlpaca-20k"], sd
    assert sorted(sd["builtin_candidates"][0]["skills"]) == ["code", "math"], sd
    assert "old, new" in sd["builtin_note"] and "math, code" in sd["builtin_note"] and "--live-since" in sd["builtin_note"]
    assert per[1]["off_target_data"]["math"]["description"] == "Orca-Math word problems", "the inputs are not changed"

    # a built-in target: its own description is not known either
    per = [evidence.selection_data("math", EFFECT, evidence.make_lookup(AGENT, b, pool={}), hd, basis) for b in [old, new]]
    sd = evidence.merge_uncertain(per, labels)
    assert sd["source"] == "builtin" and sd["description"] is None, sd
    assert sd["datasets"] == ["openai/gsm8k", "microsoft/orca-math-word-problems-200k"], sd
    assert sd["pool"] == "math" and sd["pool_match"] == "name", sd  # matched by name in both versions
    assert sd["reported_benchmarks"] == ["openai/gsm8k", "google-research-datasets/mbpp"], sd
    assert sd["builtin_candidates"][1]["skills"]["math"]["description"] == "Orca-Math word problems", sd

    # the versions agree on every skill of the row: nothing is blanked, and the note says so
    per = [evidence.selection_data("fraction_role", {"fraction_role|search": 0.2, "general_text|search": 0.0},
                                   evidence.make_lookup(AGENT, b, pool={}), hd, basis) for b in [old, new]]
    sd = evidence.merge_uncertain(per, labels)
    assert sd["reported_benchmarks"] == [] and sd["off_target_data"] == per[0]["off_target_data"], sd
    assert "agree" in sd["builtin_note"] and sd["builtin_candidates"][0]["skills"] == {}, sd

    # one version was not checked against the benchmark list: not checked is not "none found"
    per = [evidence.selection_data("fraction_role", EFFECT, evidence.make_lookup(AGENT, old, pool={}), None, basis),
           evidence.selection_data("fraction_role", EFFECT, evidence.make_lookup(AGENT, new, pool={}), hd, basis)]
    assert evidence.merge_uncertain(per, labels)["reported_benchmarks"] is None


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

    # a layer the engine cannot count: no routing, and the note says why. The reason is final:
    # another engine call gives the same answer.
    routing, note = evidence.routing_fields([0], 5, "sk", {"rates": {}, "unsupported": {"5": "module_list experts: x"}})
    assert routing is None and "module_list" in note and note.startswith(evidence.FINAL), note

    # the engine does not hold the target skill: no routing object with null rates in it
    general_only = {"rates": {
        "general_text|search": {"n_items": 2, "layers": {"3": {"rates": [0.25, 0.25, 0.5, 1.0], "tokens": 60}}},
        "general_text|heldout": {"n_items": 1, "layers": {"3": {"rates": [0.5, 0.0, 0.2, 1.3], "tokens": 30}}}},
        "counted_layers": [3], "unsupported": {}, "missing": ["code_rust|search", "code_rust|heldout"]}
    routing, note = evidence.routing_fields([1], 3, "code_rust", general_only)
    assert routing is None and "code_rust|search (the engine does not hold this skill)" in note, (routing, note)
    assert not note.startswith(evidence.FINAL), note
    # one half of the target is absent: the routing is kept, and the note names the half
    partial = dict(result, rates=dict(result["rates"]))
    del partial["rates"]["sk|heldout"]
    routing, note = evidence.routing_fields([1], 3, "sk", partial)
    assert routing["experts"]["1"]["search"]["log_ratio"] == 0.6931, routing
    assert routing["experts"]["1"]["heldout"] == {"skill_rate": None, "general_rate": 0.0, "log_ratio": None}, routing
    assert note == "routing is partial: no pick rates for sk|heldout (no rates)", note
    assert "sk|heldout" not in routing["tokens"], routing

    # top_k: the rates of a layer must add up to it. 0 + 0.5 + 0.25 + 1.25 = 2 passes; the answer of a
    # hook that counted rows in place of tokens (1 + 3.5 + 4 + 1.5 = 10 for a top-2 router) is refused.
    with_k = json.loads(json.dumps(result))
    for key in with_k["rates"]:
        with_k["rates"][key]["layers"]["3"]["top_k"] = 2
    routing, note = evidence.routing_fields([1], 3, "sk", with_k)
    assert note is None and routing["experts"]["1"]["search"]["skill_rate"] == 0.5, (routing, note)
    with_k["rates"]["sk|search"]["layers"]["3"] = {"rates": [1.0, 3.5, 4.0, 1.5], "tokens": 2, "top_k": 2}
    routing, note = evidence.routing_fields([1], 3, "sk", with_k)
    assert routing is None and note.startswith(evidence.FINAL) and "add up to 10.0" in note and "top-k 2" in note, note

    # what the client knows about a layer without an engine call
    detail = [{"layer": 0, "experts": {"path": "mlp.experts", "n_experts": 4}},
              {"layer": 1, "experts": {"path": "mlp.experts", "n_experts": 3, "style": "module_list"}},
              {"layer": 2, "mlp": {"path": "mlp.down_proj", "width": 8}}]
    assert evidence.client_unsupported(detail, 0) is None
    assert "module_list" in evidence.client_unsupported(detail, 1)
    assert "no experts" in evidence.client_unsupported(detail, 2)
    assert evidence.client_unsupported(None, 1) is None and evidence.client_unsupported([], 1) is None
    routing, note = evidence.routing_fields([0], 5, "sk", {"rates": {}, "unsupported": {}})
    assert routing is None and "no pick rates" in note, note
    assert evidence.log_ratio(None, 0.2) is None and evidence.log_ratio(0.2, 0.2) == 0.0

    # Each counted token picks top-k experts, so the rates of a layer add up to top-k. A sum that is
    # off by more than 1% means the hook did not see one row of indices per token: no routing, final.
    def with_rates(rates, top_k):
        res = json.loads(json.dumps(result))
        res["rates"]["sk|search"]["layers"]["3"] = {"rates": rates, "tokens": 40, "top_k": top_k}
        return evidence.routing_fields([1], 3, "sk", res)
    routing, note = with_rates([0.0, 0.5, 0.25, 1.35], 2)       # adds up to 2.1
    assert routing is None and note.startswith(evidence.FINAL) and "add up to 2.1" in note, note
    routing, note = with_rates([0.0, 0.5, 0.25, 1.26], 2)       # 2.01: within the rounding of the rates
    assert routing is not None and note is None, note


def test_banned_match():
    # the rule of wsp/healdata.py: the id itself, or an id that ends with "/<banned id>"
    assert evidence.banned_match("openai/gsm8k", ["openai/gsm8k"])
    assert evidence.banned_match("SomeOrg/GSM8K ", ["gsm8k"])
    assert evidence.banned_match("gsm8k", ["gsm8k"])
    assert not evidence.banned_match("someorg/gsm8k-extra", ["gsm8k"])
    assert not evidence.banned_match("someorg/not-gsm8k", ["gsm8k"])
    assert evidence.reported_benchmarks([{"source": "pool", "datasets": ["someorg/gsm8k"]}], ["gsm8k"]) == ["someorg/gsm8k"]


test_z()
test_guard()
test_selection_data()
test_path_check()
test_merge_uncertain()
test_real_files()
test_routing()
test_banned_match()
print("ok")
