# CPU tests for mine/sweep.py: candidate building, the z-score helper, order and cut-off, guard skills,
# and the write path (the real light gate and the real Miner.write_part on a temporary folder, with
# synthetic engine results in place of the GPU engine).
#   .venv/bin/python mine/tests/test_sweep.py
import json
import os
import shutil
import sys
import tempfile

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import adapter  # noqa: E402
import common  # noqa: E402
import client  # noqa: E402
import sweep  # noqa: E402


def layer_rank(width, skill_values, general_values=None, top_n=1024, how="selective"):
    # one layer of an engine ranking, made by the function the engine uses
    mine = torch.zeros(width)
    ref = torch.zeros(width)
    for u in skill_values:
        mine[u] = skill_values[u]
    for u in general_values or {}:
        ref[u] = general_values[u]
    return adapter.top_units(mine, ref, top_n, how)


def mlp_ranking():
    layers = {}
    # layer 0: wide, 1500 skill-specific units with falling values -> all three sizes fit
    layers["0"] = layer_rank(4096, dict((u, (1500 - u) * 1e-4) for u in range(1500)))
    # layer 1: the highest values per unit, but only 100 units wide -> a cluster of 64 is more than half
    layers["1"] = layer_rank(100, dict((u, 1.0) for u in range(100)))
    # layer 2: 300 specific units; the first 100 also act on general text, so they rank below the rest
    layers["2"] = layer_rank(2048, dict((u, 0.01) for u in range(300)), dict((u, 0.005) for u in range(100)))
    # layer 3: 70 specific units -> only the size 64 fits
    layers["3"] = layer_rank(2048, dict((u, 0.02) for u in range(500, 570)))
    # layer 4: 64 weak units; layer 5: nothing specific
    layers["4"] = layer_rank(2048, dict((u, 0.001) for u in range(64)))
    layers["5"] = layer_rank(2048, {})
    return {"method": "attribution", "layers": layers}


def test_mlp_candidates():
    rank = mlp_ranking()
    scores = sweep.layer_scores(rank, [64, 256, 1024])
    assert sorted(scores) == [0, 2, 3, 4], scores                    # layers 1 and 5 hold no candidate
    assert abs(scores[0] - 101.2224) < 1e-3, scores[0]               # 1e-4 * sum(1500 - u for u < 1024)
    assert abs(scores[2] - (200 * 0.01 + 56 * 0.005)) < 1e-5, scores[2]
    assert abs(scores[3] - 64 * 0.02) < 1e-5 and abs(scores[4] - 64 * 0.001) < 1e-6
    cands = sweep.unit_candidates("law", "mlp_cluster", rank, [64, 256, 1024], 3)
    got = [(c["layer"], c["size"]) for c in cands]
    assert got == [(0, 64), (0, 256), (0, 1024), (2, 64), (2, 256), (3, 64)], got
    info = {"n_layers": 6, "layers": [{"layer": i, "mlp": {"width": rank["layers"][str(i)]["width"]}} for i in range(6)]}
    by = {}
    for c in cands:
        idx = c["part"]["indices"]
        assert len(idx) == c["size"] and len(set(idx)) == len(idx) and idx == sorted(idx)
        assert 2 * len(idx) <= rank["layers"][str(c["layer"])]["width"]
        assert adapter.check_part(info, c["part"]) is None
        assert c["skill"] == "law" and c["kind"] == "mlp_cluster" and c["rank_method"] == "attribution"
        by[(c["layer"], c["size"])] = c
    assert by[(0, 64)]["part"]["indices"] == list(range(64))
    assert set(by[(0, 64)]["part"]["indices"]) < set(by[(0, 256)]["part"]["indices"]) < set(by[(0, 1024)]["part"]["indices"])
    assert by[(2, 64)]["part"]["indices"] == list(range(100, 164))    # the units that are quiet on general text
    assert by[(2, 256)]["part"]["indices"] == list(range(0, 56)) + list(range(100, 300))
    assert abs(by[(2, 256)]["rank_score"] - 2.28) < 1e-5
    assert abs(by[(2, 256)]["skill_effect"] - 2.56) < 1e-5 and abs(by[(2, 256)]["general_effect"] - 0.28) < 1e-5
    assert by[(3, 64)]["part"]["indices"] == list(range(500, 564))
    # fewer layers and other sizes are options
    one = sweep.unit_candidates("law", "mlp_cluster", rank, [64, 256, 1024], 1)
    assert [(c["layer"], c["size"]) for c in one] == [(0, 64), (0, 256), (0, 1024)]
    small = sweep.unit_candidates("law", "mlp_cluster", rank, [32], 6)
    assert [c["layer"] for c in small] == [1, 0, 3, 2, 4], [c["layer"] for c in small]   # 32 fits the narrow layer
    assert sweep.unit_candidates("law", "mlp_cluster", {"method": "attribution", "layers": {}}, [64], 3) == []


def test_head_and_expert_candidates():
    heads = {"method": "attribution", "layers": {
        "0": layer_rank(16, dict((u, 0.1) for u in range(10))),     # 16 heads: groups of 4 and 8
        "1": layer_rank(12, dict((u, 0.3) for u in range(12))),     # 12 heads: 8 is more than half
        "2": layer_rank(8, dict((u, 0.2) for u in [1, 3, 5])),      # 3 specific heads: no group of 4
        "3": layer_rank(8, dict((u, 0.05) for u in range(8)))}}
    cands = sweep.unit_candidates("math", "attn_heads", heads, [4, 8], 3)
    got = [(c["layer"], c["size"], c["part"]["indices"]) for c in cands]
    assert got == [(1, 4, [0, 1, 2, 3]), (0, 4, [0, 1, 2, 3]), (0, 8, list(range(8))), (3, 4, [0, 1, 2, 3])], got
    experts = {"method": "activation", "layers": {
        # activation ratios: the floor is 1, so only experts picked more often on the skill count
        "4": layer_rank(8, {0: 0.30, 1: 0.05, 2: 0.25, 5: 0.2}, dict((u, 0.1) for u in range(8)), how="ratio")}}
    assert experts["layers"]["4"]["floor"] == 1.0
    cands = sweep.unit_candidates("math", "expert", experts, [2, 4, 8], 3)
    assert [(c["size"], c["part"]["indices"]) for c in cands] == [(2, [0, 2])], cands   # 3 specific experts: 4 and 8 skipped
    assert cands[0]["rank_method"] == "activation"
    # a malformed engine row cannot give a bad part: repeats and out-of-range units are dropped
    bad = {"width": 10, "floor": 0.0, "units": [5, 5, 99, 2, -1], "score": [1, 1, 1, 1, 1], "skill": [1] * 5, "general": [0] * 5}
    assert [u[0] for u in sweep.specific_units(bad)] == [5, 2]
    assert sweep.fitting_sizes([8, 2, 4, 2], 8, 3) == [2] and sweep.fitting_sizes([2, 4], 8, 9) == [2, 4]


def test_block_candidates():
    entry = {"layers": 12, "layer_maps": {"law": {
        "scores": [0.0, 0.1, 0.5, 0.6, 0.4, 0.0, -0.1, 0.3, 0.2, 0.1, 0.0, 0.0],
        "influence": [0.1, 0.2, 0.9, 1.0, 0.8, 0.1, 0.0, 0.5, 0.4, 0.3, 0.1, 0.1],
        "blocks": [[2, 4], [7, 9], [0, 7], [2, 4]]}}}
    cands = sweep.block_candidates("law", entry, 12)
    assert [c["part"] for c in cands] == [{"kind": "block", "layers": [2, 3, 4]}, {"kind": "block", "layers": [7, 8, 9]}]
    assert abs(cands[0]["rank_score"] - 1.5) < 1e-9 and abs(cands[1]["rank_score"] - 0.6) < 1e-9
    assert cands[0]["rank_method"] == "layer_map" and cands[0]["size"] == 3
    for c in cands:
        assert adapter.check_part({"n_layers": 12}, c["part"]) is None
    assert sweep.block_candidates("math", entry, 12) == []      # the map has no such skill
    assert sweep.block_candidates("law", entry, 24) == []       # a map of another model shape
    assert sweep.block_candidates("law", None, 12) == []
    assert sweep.label_for(cands[0]) == "law, layer block 2 to 4 (layer-map-ranked)"
    assert sweep.evidence_for(cands[0]) == ["ablation"]


Z_CASES = [(0.30, [0.01, 0.02, 0.03, 0.02, 0.02]), (0.01, [0.1, 0.2, 0.3]), (0.5, [0.1, None, 0.3]),
           (0.2, [0.0, 0.0, 0.0, 0.0, 0.0]), (0.2, [0.0, 0.0, 0.0, 0.0, 0.0001]), (0.0, [0.0, 0.0]),
           (None, [0.1, 0.2]), (0.3, []), (0.3, None), (0.3, [0.1]), (0.3, [None, 0.1, None]),
           (0.40, [0.0, 0.39, 0.0, 0.0, 0.0])]


def test_z_score():
    # mean 0.02, sample variance (1e-4 + 1e-4) / 4 = 5e-5, std 0.00707107 -> 0.28 / 0.00707107 = 39.598
    assert sweep.z_score(0.30, [0.01, 0.02, 0.03, 0.02, 0.02]) == 39.598
    # mean 0.2, variance (0.01 + 0.01) / 2 = 0.01, std 0.1 -> (0.01 - 0.2) / 0.1 = -1.9
    assert sweep.z_score(0.01, [0.1, 0.2, 0.3]) == -1.9
    # missing random drops are ignored: mean 0.2, variance 0.02, std 0.141421 -> 0.3 / 0.141421 = 2.121
    assert sweep.z_score(0.5, [0.1, None, 0.3]) == 2.121
    # values that are not finite numbers are ignored too: the same two drops as above, shifted
    assert sweep.z_score(0.3, [0.1, float("nan"), 0.2, True, "0.5"]) == 2.121      # 0.15 / 0.0707107
    # equal random drops have no measured deviation: there is no z-score (no floor that makes one up)
    assert sweep.z_score(0.2, [0.0, 0.0, 0.0, 0.0, 0.0]) is None
    assert sweep.z_score(0.0, [0.0, 0.0]) is None
    # one rounding step of spread is a real deviation: mean 2e-5, sum of squares 4 * 4e-10 + 6.4e-9 = 8e-9,
    # variance 2e-9, std 4.47214e-5 -> 0.19998 / 4.47214e-5 = 4471.689
    assert sweep.z_score(0.2, [0.0, 0.0, 0.0, 0.0, 0.0001]) == 4471.689
    for target, randoms in [(None, [0.1, 0.2]), (0.3, []), (0.3, None), (0.3, [0.1]), (0.3, [None, 0.1, None])]:
        assert sweep.z_score(target, randoms) is None, (target, randoms)
    # After the merge with mining/row-evidence a row also holds signature.z from mine/evidence.py.
    # The two numbers must be equal in every case. This part runs when that module is there.
    try:
        import evidence
    except ImportError:
        evidence = None
    if evidence is not None:
        for target, randoms in Z_CASES:
            theirs = evidence.z_fields({"t|search": target}, randoms or [], "t")["z"]
            assert sweep.z_score(target, randoms) == theirs, (target, randoms, theirs)


def test_z_rule():
    assert sweep.NO_SPREAD_GAP == client.GATE_MIN_TARGET_DROP == 0.15
    # measured spread: z decides
    r = sweep.z_rule(0.30, [0.01, 0.02, 0.03, 0.02, 0.02], 2.0)
    assert r == {"passed": True, "z_search": 39.598, "random_spread": "measured", "gap_search": None, "reason": None}
    # mean 0.078, sum of squares 4 * 0.078^2 + 0.312^2 = 0.12168, variance 0.03042, std 0.174413 -> 0.322 / 0.174413
    r = sweep.z_rule(0.40, [0.0, 0.39, 0.0, 0.0, 0.0], 2.0)
    assert not r["passed"] and r["z_search"] == 1.846 and r["reason"] == "z-score 1.846 < 2.0"
    assert sweep.z_rule(0.40, [0.0, 0.39, 0.0, 0.0, 0.0], 1.5)["passed"]
    # no measured spread: no z-score; the gap between the target drop and the random drops decides
    r = sweep.z_rule(0.2, [0.0, 0.0, 0.0, 0.0, 0.0], 2.0)
    assert r == {"passed": True, "z_search": None, "random_spread": "below resolution", "gap_search": 0.2, "reason": None}
    assert sweep.z_rule(0.30, [0.15] * 5, 2.0)["passed"] and sweep.z_rule(0.30, [0.15] * 5, 2.0)["gap_search"] == 0.15
    assert not sweep.z_rule(0.2999, [0.15] * 5, 2.0)["passed"]
    # the light gate accepts this one (0.1501 is above 0.15 and above every random drop); the sweep does not
    r = sweep.z_rule(0.1501, [0.1499] * 5, 2.0)
    assert not r["passed"] and r["z_search"] is None and r["gap_search"] == 0.0002 and "no measured spread" in r["reason"]
    # nothing to compare with: not passed
    for target, randoms in [(None, [0.1, 0.2]), (0.3, []), (0.3, [0.1]), (0.3, [None, 0.1, None])]:
        r = sweep.z_rule(target, randoms, 2.0)
        assert not r["passed"] and r["z_search"] is None and r["random_spread"] is None and r["reason"], (target, randoms)


def cand(skill, kind, score, n, size=64, layer=0):
    # n makes the part (and so the cache key) unique
    return {"skill": skill, "kind": kind, "size": size, "layer": layer, "rank_method": "attribution", "rank_score": score,
            "part": {"kind": "mlp_cluster", "layer": 0, "indices": [n]}}


def test_order_is_size_neutral():
    # One skill, three layers, three sizes. The summed score grows with the size in every layer, and the
    # score per unit falls with it, as it does for nested top-k parts.
    table = {5: {64: 8.0, 256: 24.0, 1024: 60.0}, 4: {64: 4.0, 256: 16.0, 1024: 50.0}, 3: {64: 3.0, 256: 8.0, 1024: 10.0}}
    cands = []
    for layer in table:
        for size in table[layer]:
            cands.append(cand("law", "mlp_cluster", table[layer][size], len(cands), size, layer))
    # class means: 64 -> 5, 256 -> 16, 1024 -> 40
    want = [(5, 64, 1.6), (5, 256, 1.5), (5, 1024, 1.5), (4, 1024, 1.25), (4, 256, 1.0), (4, 64, 0.8),
            (3, 64, 0.6), (3, 256, 0.5), (3, 1024, 0.25)]
    for mode in ["global", "round"]:                     # one (skill, kind) pair: both orders are the same
        ordered = sweep.order_candidates(cands, mode)
        assert [(c["layer"], c["size"], c["norm_score"]) for c in ordered] == want, (mode, ordered)
        assert [c["round"] for c in ordered] == list(range(9))
        assert [c["rank_score"] for c in ordered] == [table[l][s] for l, s, _ in want]     # the sum is kept
    # a run cut after 3 screens holds one part of every size; a sort on the sum holds no part of 64
    work, left = sweep.plan_work(sweep.order_candidates(cands), set(), 3)
    assert sorted(c["size"] for c, _ in work) == [64, 256, 1024] and len(left) == 6
    by_sum = sorted(cands, key=lambda c: -c["rank_score"])[:3]
    assert [c["size"] for c in by_sum] == [1024, 1024, 256]
    # a class whose mean score is not above 0 (blocks can have negative map scores) gets 0, not a division error
    odd = sweep.order_candidates([cand("law", "block", -0.2, 50), cand("math", "block", 0.1, 51)])
    assert [c["norm_score"] for c in odd] == [0.0, 0.0]
    assert sweep.score_class(cand("law", "block", 1.0, 1, 3)) == sweep.score_class(cand("law", "block", 1.0, 2, 7))
    assert sweep.score_class(cand("law", "expert", 1.0, 1, 4)) != sweep.score_class(cand("law", "expert", 1.0, 2, 8))


def test_order_and_cutoff():
    cands = [cand("law", "mlp_cluster", 1.0, 1), cand("law", "mlp_cluster", 5.0, 2), cand("law", "mlp_cluster", 3.0, 3),
             cand("law", "attn_heads", 0.2, 4), cand("law", "attn_heads", 0.5, 5),
             cand("math", "mlp_cluster", 2.0, 6), cand("math", "mlp_cluster", 4.0, 7), cand("math", "block", 9.0, 8)]
    # class means: MLP clusters (1 + 5 + 3 + 2 + 4) / 5 = 3, head groups 0.35, blocks 9
    ordered = sweep.order_candidates(cands, "round")
    got = [(c["round"], c["skill"], c["kind"], c["rank_score"], c["norm_score"]) for c in ordered]
    assert got == [(0, "law", "mlp_cluster", 5.0, 1.66667), (0, "law", "attn_heads", 0.5, 1.42857),
                   (0, "math", "mlp_cluster", 4.0, 1.33333), (0, "math", "block", 9.0, 1.0),
                   (1, "law", "mlp_cluster", 3.0, 1.0), (1, "math", "mlp_cluster", 2.0, 0.666667),
                   (1, "law", "attn_heads", 0.2, 0.571429),
                   (2, "law", "mlp_cluster", 1.0, 0.333333)], got
    assert "round" not in cands[0] and "norm_score" not in cands[0]   # the input is not changed
    # the default: best score first across skills and kinds; equal scores go by skill name
    flat = [c["rank_score"] for c in sweep.order_candidates(cands)]
    assert flat == [5.0, 0.5, 4.0, 3.0, 9.0, 2.0, 0.2, 1.0], flat
    assert sweep.order_candidates(cands) == sweep.order_candidates(cands, "global")
    assert sweep.order_candidates(list(reversed(cands)), "round") == ordered   # the input order does not matter

    # cut-off: 3 new screens; the two cached screens cost nothing and stay in the work
    cached = set([sweep.cand_key(ordered[4]), sweep.cand_key(ordered[7])])
    work, left = sweep.plan_work(ordered, cached, 3)
    new = [c for c, is_cached in work if not is_cached]
    assert new == ordered[0:3]                                       # a cut-off run holds the best candidates
    assert [c for c, is_cached in work if is_cached] == [ordered[4], ordered[7]]
    assert left == [ordered[3], ordered[5], ordered[6]]
    assert [c for c, _ in work] == [ordered[i] for i in [0, 1, 2, 4, 7]]   # the work keeps the order
    # a rerun with the first three done takes the next ones
    done = cached | set(sweep.cand_key(c) for c in new)
    work2, left2 = sweep.plan_work(ordered, done, 2)
    assert [c for c, is_cached in work2 if not is_cached] == [ordered[3], ordered[5]] and left2 == [ordered[6]]
    work0, left0 = sweep.plan_work(ordered, cached, 0)
    assert all(is_cached for _, is_cached in work0) and len(work0) == 2 and len(left0) == 6
    everything, nothing = sweep.plan_work(ordered, set(), 100)
    assert len(everything) == 8 and nothing == []


def test_guards_and_payload(specs):
    names = set(specs.keys())
    for d in sweep.DOMAINS:
        for n in sweep.DOMAINS[d]:
            assert n in names, "domain table names a skill that does not exist: " + n
    assert len(set(sweep.domain_of(g) for g in sweep.GUARDS)) == len(sweep.GUARDS)   # one guard per domain
    usable = list(names)
    assert sweep.guards_for("law", usable) == ["math", "code", "medicine", "prose_zh", "physics"]
    assert sweep.guards_for("clinical_knowledge", usable) == ["math", "code", "law", "prose_zh", "physics"]
    assert sweep.guards_for("math", usable) == ["code", "medicine", "law", "prose_zh", "physics"]
    assert sweep.guards_for("a_new_pool_skill", usable) == ["math", "code", "medicine", "law", "prose_zh"]
    # a guard the model cannot do is replaced by the next skill of the same domain
    assert sweep.guards_for("law", [n for n in usable if n != "code"]) == ["math", "sql", "medicine", "prose_zh", "physics"]
    lacks = [n for n in usable if n not in ["law", "prose_zh", "physics"]]
    assert sweep.guards_for("medicine", lacks) == ["math", "code", "finance", "multilingual", "chemistry"]
    assert sweep.spare_guards(usable, specs) == []
    assert sweep.spare_guards([n for n in usable if n != "physics"], specs) == ["chemistry", "science_papers"]
    # a domain with no usable skill gives no guard: the list is shorter, down to none
    few = sweep.DOMAINS["math"] + sweep.DOMAINS["code"] + sweep.DOMAINS["general"]
    assert sweep.guards_for("law", few) == ["math", "code", "facts"]
    assert sweep.guards_for("math", ["math", "mathematics_theory"]) == [] and sweep.guards_for("math", []) == []
    for target in names:
        if target == sweep.REFERENCE:
            continue
        g = sweep.guards_for(target, usable)
        assert len(g) == sweep.N_GUARDS and len(set(g)) == len(g) and target not in g
        assert all(sweep.domain_of(x) != sweep.domain_of(target) for x in g)
        payload = sweep.build_payload(specs, target, g)
        assert list(payload["skills"].keys()) == [target, sweep.REFERENCE] + g      # 7 sets, not all of them
    payload = sweep.build_payload(specs, "law", sweep.guards_for("law", usable))
    assert payload["skills"]["math"] == {"builtin": True} and payload["skills"]["general_text"] == {"builtin": True}
    law = payload["skills"]["law"]
    assert law["control"] is None and law["hash"].startswith("pool-") and sorted(law["items"].keys()) == ["heldout", "search"]
    assert set(law["items"]["search"][0].keys()) == set(["context", "continuation", "digits_only"])
    assert sweep.parse_skills("law, math,law", specs) == ["law", "math"]
    everything = sweep.parse_skills("all", specs)
    assert sweep.REFERENCE not in everything and "law" in everything and "facts" in everything
    for bad in ["general_text", "no_such_skill", " , "]:
        try:
            sweep.parse_skills(bad, specs)
        except SystemExit:
            continue
        raise AssertionError("expected SystemExit for " + repr(bad))


def test_labels_and_summary():
    c = {"skill": "law", "kind": "mlp_cluster", "size": 256, "layer": 14, "rank_method": "attribution", "rank_score": 0.5}
    assert sweep.label_for(c) == "law, MLP cluster of 256 units, layer 14 (attribution-ranked)"
    assert sweep.evidence_for(c) == ["attribution", "ablation"]
    assert "attribution patching on 24 search items" in sweep.rationale_for(c, 24)
    h = dict(c, kind="attn_heads", size=8, rank_method="activation")
    assert sweep.label_for(h) == "law, group of 8 attention heads, layer 14 (activation-ranked)"
    assert sweep.evidence_for(h) == ["activation", "ablation"] and "quantized" in sweep.rationale_for(h, 24)
    assert sweep.label_for(dict(c, kind="expert", size=4)) == "law, set of 4 experts, layer 14 (attribution-ranked)"
    rows = [{"skill": "law", "kind": "mlp_cluster", "screened": True, "passed": True},
            {"skill": "law", "kind": "mlp_cluster", "screened": True, "passed": False},
            {"skill": "law", "kind": "block", "screened": False, "passed": False},
            {"skill": "math", "kind": "block", "screened": True, "passed": True}]
    s = sweep.summarize(rows)
    assert s["total"] == {"ranked": 4, "screened": 3, "passed": 2}
    assert s["by_kind"]["mlp_cluster"] == {"ranked": 2, "screened": 2, "passed": 1}
    assert s["by_kind"]["block"] == {"ranked": 2, "screened": 1, "passed": 1}
    assert s["by_skill"]["law"] == {"ranked": 3, "screened": 2, "passed": 1}
    assert s["by_skill"]["math"] == {"ranked": 1, "screened": 1, "passed": 1}
    structure = {"layers_detail": [{"layer": 0, "attn": {"n_heads": 8}, "mlp": {"width": 64}},
                                   {"layer": 1, "attn": {"n_heads": 8}, "experts": {"n_experts": 8, "style": "module_list"}}]}
    # experts kept as a ModuleList count: the engine ranks them by attribution
    assert sweep.model_kinds(structure) == ["mlp_cluster", "attn_heads", "expert"]
    assert sweep.model_kinds({"layers_detail": structure["layers_detail"][:1]}) == ["mlp_cluster", "attn_heads"]


def test_add_row_fields():
    row = {"id": "x", "tier": "screened", "skill": {"source": "builtin", "name": "law"}, "signature": {"gate": 1}}
    before = json.dumps(row, sort_keys=True)
    sweep.add_row_fields(row, {})
    assert json.dumps(row, sort_keys=True) == before
    sweep.add_row_fields(row, {"sweep": {"z_search": 3.0}, "tier": "confirmed", "signature": None,
                               "skill": {"source": "pool", "dataset": "d"}})
    assert row["tier"] == "screened" and row["signature"] == {"gate": 1}      # existing fields are never replaced
    assert row["sweep"] == {"z_search": 3.0}
    assert row["skill"] == {"source": "pool", "name": "law", "dataset": "d"}


class FakeEngine:
    # stands in for Miner.call: returns synthetic results in the engine's formats and records the calls
    def __init__(self):
        self.calls = []
        self.drops = {}            # skill -> relative drop, for every part
        self.part_drops = {}       # part key -> {skill: drop}, for one part
        self.random = {}           # target -> drops of the random parts
        self.rank_result = None    # the ranking for every (kind, skill)
        self.rank_by = {}          # (kind, skill) -> ranking
        self.margins = {}          # skill -> margin over shuffled answers
        self.fail_parts = set()    # part keys whose ablation raises
        self.fail_ranks = set()    # (kind, skill) whose ranking raises
        self.quantized = False
        self.clock = None
        self.seconds_per_ablate = 0

    def __call__(self, method, *args):
        self.calls.append((method, args))
        if method == "base":
            scores = {}
            margins = {}
            for name in args[0]["skills"]:
                scores[name + "|search"] = {"score": 0.3}
                margins[name] = self.margins.get(name, 1.0)
            return {"scores": scores, "margin_vs_shuffled": margins, "seconds": 1.0}
        if method == "ablate":
            payload, part = args
            key = client.part_key(part)
            if key in self.fail_parts:
                raise RuntimeError("bad part")
            if self.clock is not None:
                self.clock.now += self.seconds_per_ablate
            table = self.part_drops.get(key, self.drops)
            drops = {}
            for name in payload["skills"]:
                for split in ["search", "heldout"]:
                    drops[name + "|" + split] = table.get(name, 0.002)
            return {"drops": drops, "ablated": {}, "seconds": 1.0}
        if method == "random_drops":
            return {"drops": list(self.random.get(args[2], [0.01, 0.02, 0.03, 0.02, 0.02])), "seconds": 2.0}
        if method in ["attribution_units", "activation_units"]:
            if method == "attribution_units" and self.quantized:
                raise RuntimeError("attribution_units needs bf16 weights, but this model is loaded quantized (FP8); "
                                   "use activation_units instead")
            if (args[1], args[2]) in self.fail_ranks:
                raise RuntimeError("CUDA out of memory")
            rank = self.rank_by.get((args[1], args[2]), self.rank_result)
            return dict(rank, method=method.split("_")[0], seconds=3.0, n_tokens=100)
        raise AssertionError("unexpected engine call " + method)

    def count(self, method):
        return len([c for c in self.calls if c[0] == method])


def make_miner(short, specs):
    # a SweepMiner with no Modal handle and no network: its engine is a FakeEngine
    m = object.__new__(sweep.SweepMiner)
    m.short = short
    m.hf_id = "org/Toy"
    m.revision = "rev-main"
    m.params_b = 1.0
    m.modal_seconds = 0.0
    m.state = common.read_json_safe(common.state_path(short), {})
    m.specs = specs
    m.active_payload = {"skills": {}}
    m.row_extra = {}
    engine = FakeEngine()
    m.call = engine
    return m, engine


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def test_miner_paths(specs):
    # The real Miner.screen, light gate, and write_part, on a temporary person_a folder.
    real_root = common.ROOT
    old_lane = os.environ.get("WSP_LANE")
    tmp = tempfile.mkdtemp(prefix="sweep-test-")
    common.ROOT = tmp
    os.environ["WSP_LANE"] = "sweep"
    try:
        short = "toy"
        structure = {"n_layers": 12, "d_model": 64, "layers_detail": []}
        main_state = {"revision": "rev-main", "model": "org/Toy", "structure": structure,
                      "screens": {"mlp_cluster:0:abc|law": {"gate": {"passed": True}}},
                      "agent_skills": {"x": {"path": "p", "hash": "h", "description": "d"}}, "base": {"scores": {}}}
        main_path = os.path.join(common.logs_dir(short), "state.json")
        common.write_json_atomic(main_path, main_state)
        main_bytes = open(main_path).read()
        old_row = {"id": "mlp_agent_part_toy_L3_k2_001", "type": "mlp_cluster", "label": "Agent part", "tier": "screened",
                   "skill": {"source": "agent", "path": "mine/skills/toy/x.json", "hash": "h", "description": "d", "name": "x"},
                   "signature": {"effect": {}, "random_drops": [], "gate": {"passed": True}, "confirm": None},
                   "part_key": "mlp_cluster:3:0000", "layer": 3}
        common.append_jsonl(common.catalog_path(short), old_row)
        old_line = open(common.catalog_path(short)).read()

        # the lane state: its own file, seeded with the revision and structure only
        lane_path = common.state_path(short)
        assert lane_path.endswith("state.lanesweep.json") and lane_path != main_path
        sweep.seed_lane_state(short)
        assert common.read_json_safe(lane_path, None) == {"revision": "rev-main", "model": "org/Toy", "structure": structure}
        common.write_json_atomic(lane_path, {"revision": "rev-main", "model": "org/Toy", "structure": structure, "mark": 1})
        sweep.seed_lane_state(short)
        assert common.read_json_safe(lane_path, {}).get("mark") == 1      # an existing lane state is kept

        # one sweep per lane
        first = sweep.lock_lane(short)
        try:
            sweep.lock_lane(short)
            raise AssertionError("a second sweep on the lane must be refused")
        except SystemExit:
            pass
        first.close()
        sweep.lock_lane(short).close()

        m, engine = make_miner(short, specs)
        usable = list(specs.keys())
        guards = m.guards("law", usable)
        assert guards == ["math", "code", "medicine", "prose_zh", "physics"]
        assert m.guards("law", ["math"]) == guards                      # kept from the first screen
        append_before = common.append_jsonl

        # 1. a pool-skill MLP cluster that passes the gate and the z rule
        c1 = {"skill": "law", "kind": "mlp_cluster", "size": 4, "layer": 5, "rank_method": "attribution",
              "rank_score": 0.75, "part": {"kind": "mlp_cluster", "layer": 5, "indices": [9, 2, 7, 4]}}
        engine.drops = {"law": 0.30}
        engine.random = {"law": [0.01, 0.02, 0.03, 0.02, 0.02]}
        r1 = m.screen_candidate(c1, guards, 2.0, 24)
        assert r1["passed"] and r1["gate_passed"] and r1["z_search"] == 39.598 and r1["target_drop_search"] == 0.30
        assert r1["random_spread"] == "measured" and r1["gap_search"] is None
        assert common.append_jsonl is append_before                     # the append hook is put back
        ablate_payload = [c for c in engine.calls if c[0] == "ablate"][0][1][0]
        assert list(ablate_payload["skills"].keys()) == ["law", "general_text"] + guards
        rows = read_jsonl(common.catalog_path(short))
        assert len(rows) == 2 and open(common.catalog_path(short)).read().startswith(old_line)   # old row untouched
        row = rows[1]
        assert row["id"] == r1["catalog_id"] and row["id"].endswith("_toy_L5_k4_001") and row["id"].startswith("mlp_law_")
        assert row["tier"] == "screened" and row["type"] == "mlp_cluster" and row["layer"] == 5 and row["num_units"] == 4
        assert row["label"] == "law, MLP cluster of 4 units, layer 5 (attribution-ranked)"
        assert row["evidence_used"] == ["attribution", "ablation"]
        assert row["weight_rationale"].startswith("No weight reading. Ranked by attribution patching")
        assert row["model"] == "org/Toy" and row["revision"] == "rev-main" and row["d_model"] == 64
        assert row["signature"]["effect"]["law|search"] == 0.30 and row["signature"]["gate"]["passed"] is True
        assert row["signature"]["random_drops"] == [0.01, 0.02, 0.03, 0.02, 0.02]
        assert row["skill"]["source"] == "pool" and row["skill"]["name"] == "law"
        assert row["skill"]["dataset"].startswith("cais/mmlu") and row["skill"]["hash"] == specs["law"]["info"]["hash"]
        assert row["sweep"] == {"rank_method": "attribution", "rank_score": 0.75, "rank_items": 24,
                                "reference": "general_text", "guards": guards, "z_search": 39.598, "z_min": 2.0,
                                "random_spread": "measured"}
        index = torch.load(os.path.join(tmp, row["units"]))
        assert index["kind"] == "mlp_cluster" and index["layer"] == 5 and index["indices"].tolist() == [2, 4, 7, 9]

        # 2. the same candidate again: no engine call, no second row
        n_calls = len(engine.calls)
        again = m.screen_candidate(c1, guards, 2.0, 24)
        assert again["passed"] and again["catalog_id"] == row["id"]
        assert len(engine.calls) == n_calls and len(read_jsonl(common.catalog_path(short))) == 2

        # 3. the light gate passes but the random parts spread too widely: z below 2, nothing written
        c2 = dict(c1, part={"kind": "mlp_cluster", "layer": 5, "indices": [11, 12, 13, 14]})
        engine.drops = {"law": 0.40}
        engine.random = {"law": [0.0, 0.39, 0.0, 0.0, 0.0]}
        r2 = m.screen_candidate(c2, guards, 2.0, 24)
        # mean 0.078, sample variance 0.03042, std 0.174413 -> (0.40 - 0.078) / 0.174413 = 1.846
        assert r2["gate_passed"] and not r2["passed"] and r2["z_search"] == 1.846 and r2["catalog_id"] is None
        assert r2["reasons"] == ["z-score 1.846 < 2.0"]
        assert m.screen_candidate(c2, guards, 1.5, 24)["passed"]         # a cached screen follows a new --z-min
        assert len(read_jsonl(common.catalog_path(short))) == 3

        # 4. the light gate fails (a guard skill drops too): not passed, whatever the z-score
        c3 = dict(c1, part={"kind": "mlp_cluster", "layer": 5, "indices": [21, 22, 23, 24]})
        engine.drops = {"law": 0.30, "code": 0.25}
        engine.random = {"law": [0.01, 0.02, 0.03, 0.02, 0.02]}
        r3 = m.screen_candidate(c3, guards, 2.0, 24)
        assert not r3["gate_passed"] and not r3["passed"] and r3["z_search"] == 39.598
        assert any("code" in reason for reason in r3["reasons"])

        # 5. a built-in skill and a block: Miner's own skill entry stays, a block row has "layers"
        c4 = {"skill": "math", "kind": "block", "size": 2, "layers": [3, 4], "rank_method": "layer_map",
              "rank_score": 1.5, "part": {"kind": "block", "layers": [3, 4]}}
        engine.drops = {"math": 0.5}
        engine.random = {"math": [0.05, 0.1, 0.0, 0.07, 0.03]}
        math_guards = m.guards("math", usable)
        r4 = m.screen_candidate(c4, math_guards, 2.0, 24)
        assert r4["passed"]
        row4 = read_jsonl(common.catalog_path(short))[-1]
        assert row4["skill"] == {"source": "builtin", "name": "math"} and row4["layers"] == [3, 4] and "layer" not in row4
        assert row4["evidence_used"] == ["ablation"] and row4["sweep"]["rank_items"] is None
        assert row4["label"] == "math, layer block 3 to 4 (layer-map-ranked)" and row4["id"].endswith("_toy_L3-4_001")

        # the agents' state file is never written; the sweep's screens are in the lane state
        assert open(main_path).read() == main_bytes
        lane = common.read_json_safe(lane_path, {})
        assert len(lane["screens"]) == 4 and "mlp_cluster:0:abc|law" not in lane["screens"]
        assert lane["sweep_guards"]["law"] == guards
        # the lane state holds the catalog id of every row the sweep wrote
        assert lane["sweep_written"] == {sweep.cand_key(c1): row["id"], sweep.cand_key(c2): read_jsonl(
            common.catalog_path(short))[2]["id"], sweep.cand_key(c4): row4["id"]}

        # 5b. random parts with no measured spread: no z-score in the row, the gap decides
        c5 = dict(c1, part={"kind": "mlp_cluster", "layer": 5, "indices": [31, 32, 33, 34]})
        engine.drops = {"law": 0.2}
        engine.random = {"law": [0.0, 0.0, 0.0, 0.0, 0.0]}
        r5 = m.screen_candidate(c5, guards, 2.0, 24)
        assert r5["passed"] and r5["z_search"] is None and r5["random_spread"] == "below resolution" and r5["gap_search"] == 0.2
        row5 = read_jsonl(common.catalog_path(short))[-1]
        assert row5["id"] == r5["catalog_id"] and row5["signature"]["random_drops"] == [0.0] * 5
        assert row5["sweep"]["z_search"] is None and row5["sweep"]["random_spread"] == "below resolution"
        assert row5["sweep"]["gap_search"] == 0.2 and row5["sweep"]["gap_min"] == 0.15
        # the light gate passes (0.1501 is above 0.15 and above every random drop), the gap is too small
        c6 = dict(c1, part={"kind": "mlp_cluster", "layer": 5, "indices": [41, 42, 43, 44]})
        engine.drops = {"law": 0.1501}
        engine.random = {"law": [0.1499] * 5}
        r6 = m.screen_candidate(c6, guards, 2.0, 24)
        assert r6["gate_passed"] and not r6["passed"] and r6["z_search"] is None and r6["catalog_id"] is None
        assert len(r6["reasons"]) == 1 and "no measured spread" in r6["reasons"][0]
        n_rows = len(read_jsonl(common.catalog_path(short)))
        assert n_rows == 5

        # 5c. reruns and rows that left the catalog. Someone deletes the row of c5 from the catalog (an
        # audit): the kept screen still passes, but the row is not written again.
        kept_rows = [r for r in read_jsonl(common.catalog_path(short)) if r["id"] != row5["id"]]
        with open(common.catalog_path(short), "w") as f:
            for r in kept_rows:
                f.write(json.dumps(r) + "\n")
        n_calls = len(engine.calls)
        again5 = m.screen_candidate(c5, guards, 2.0, 24)
        assert again5["passed"] and again5["catalog_id"] == row5["id"] and len(engine.calls) == n_calls
        assert [r["id"] for r in read_jsonl(common.catalog_path(short))] == [r["id"] for r in kept_rows]
        # mine/regate.py moved the row of c1 to the rejected file: the part is not passed and not written,
        # also when the lane state has no record of the first write (a state from before that record)
        rejected_path = os.path.join(tmp, "catalog", short + ".rejected.jsonl")
        common.append_jsonl(rejected_path, dict(row, rejected_reasons=["audit"]))
        kept_rows = [r for r in kept_rows if r["id"] != row["id"]]
        with open(common.catalog_path(short), "w") as f:
            for r in kept_rows:
                f.write(json.dumps(r) + "\n")
        assert sweep.rejected_parts(short) == set([(row["part_key"], "law")])
        for forget in [False, True]:
            if forget:
                del m.state["sweep_written"][sweep.cand_key(c1)]
            again1 = m.screen_candidate(c1, guards, 2.0, 24)
            assert not again1["passed"] and again1["gate_passed"] and again1["catalog_id"] is None
            assert "rejected" in again1["reasons"][-1] and len(engine.calls) == n_calls
            assert [r["id"] for r in read_jsonl(common.catalog_path(short))] == [r["id"] for r in kept_rows]
        # a kept screen is judged by the gate of today: this one says "passed", but a guard skill dropped
        c7 = dict(c1, part={"kind": "mlp_cluster", "layer": 5, "indices": [51, 52, 53, 54]})
        stale = {"part": c7["part"], "target": "law", "random_drops": [0.01, 0.02, 0.03, 0.02, 0.02], "seconds": 1.0,
                 "drops": {"law|search": 0.3, "law|heldout": 0.3, "code|search": 0.25, "code|heldout": 0.0},
                 "gate": {"passed": True, "reasons": [], "gate_version": 1}}
        m.state["screens"][sweep.cand_key(c7)] = stale
        r7 = m.screen_candidate(c7, guards, 2.0, 24)
        assert not r7["gate_passed"] and not r7["passed"] and any("code" in reason for reason in r7["reasons"])
        assert len(engine.calls) == n_calls and len(read_jsonl(common.catalog_path(short))) == len(kept_rows)
        assert m.state["screens"][sweep.cand_key(c7)]["gate"]["gate_version"] == 1     # the kept screen is not changed

        # 6. rankings: kept in the lane folder, redone when the options change
        engine.rank_result = {"layers": {"0": layer_rank(64, dict((u, 0.1) for u in range(20)))}}
        found, cached = m.ranked_candidates("law", "mlp_cluster", [4, 8], 3, 24)
        assert not cached and [(c["layer"], c["size"]) for c in found] == [(0, 4), (0, 8)]
        assert m.state["sweep_method"] == "attribution" and engine.count("attribution_units") == 1
        rank_call = [c for c in engine.calls if c[0] == "attribution_units"][0][1]
        assert list(rank_call[0]["skills"].keys()) == ["law", "general_text"] and rank_call[1:6] == ("mlp_cluster", "law", [], 8, 24)
        again, cached = m.ranked_candidates("law", "mlp_cluster", [4, 8], 3, 24)
        assert cached and again == found and engine.count("attribution_units") == 1
        assert os.path.exists(os.path.join(common.logs_dir(short), "sweep", "rank_law_mlp_cluster.json"))
        other, cached = m.ranked_candidates("law", "mlp_cluster", [4], 3, 24)
        assert not cached and engine.count("attribution_units") == 2 and [c["size"] for c in other] == [4]

        # 7. a quantized model: attribution refuses, the sweep falls back to activation and remembers it
        del m.state["sweep_method"]
        engine.quantized = True
        engine.rank_result = {"layers": {"0": layer_rank(8, {1: 0.4, 6: 0.3}, dict((u, 0.1) for u in range(8)), how="ratio")}}
        found, cached = m.ranked_candidates("law", "expert", [2, 4], 3, 24)
        assert [c["part"]["indices"] for c in found] == [[1, 6]] and found[0]["rank_method"] == "activation"
        assert sweep.evidence_for(found[0]) == ["activation", "ablation"]
        assert m.state["sweep_method"] == "activation" and engine.count("activation_units") == 1
        tried = engine.count("attribution_units")
        m.ranked_candidates("math", "expert", [2, 4], 3, 24)
        assert engine.count("attribution_units") == tried and engine.count("activation_units") == 2
        # any other engine error is not hidden
        del m.state["sweep_method"]

        def broken(method, *args):
            raise RuntimeError("CUDA out of memory")
        m.call = broken
        try:
            m.ranked_candidates("history", "expert", [2, 4], 3, 24)
            raise AssertionError("the error must be raised")
        except RuntimeError as e:
            assert "out of memory" in str(e)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
        common.ROOT = real_root
        if old_lane is None:
            del os.environ["WSP_LANE"]
        else:
            os.environ["WSP_LANE"] = old_lane


class FakeClock:
    # replaces the time module inside sweep.py, so a test can move the clock
    def __init__(self):
        self.now = 1000.0

    def time(self):
        return self.now

    def strftime(self, fmt):
        return "2000-01-01T00:00:00"


def test_run_sweep(specs):
    # The whole run on a toy structure: base check, ranking, blocks from a layer map, order, cut-off,
    # resume, and the limits. Engine results are synthetic; everything else is the real code.
    real_root = common.ROOT
    real_entries = sweep.POOL_ENTRIES
    real_time = sweep.time
    old_lane = os.environ.get("WSP_LANE")
    tmp = tempfile.mkdtemp(prefix="sweep-run-")
    common.ROOT = tmp
    sweep.POOL_ENTRIES = os.path.join(tmp, "pool_entries")
    os.environ["WSP_LANE"] = "sweep"
    try:
        short = "toy"
        layers = [{"layer": i, "attn": {"n_heads": 8, "head_dim": 8}, "mlp": {"width": 64}} for i in range(6)]
        common.write_json_atomic(common.state_path(short), {
            "revision": "rev-main", "model": "org/Toy", "structure": {"n_layers": 6, "d_model": 64, "layers_detail": layers}})
        scores = [0.0] * 6
        common.write_json_atomic(os.path.join(sweep.POOL_ENTRIES, "org__Toy.json"), {"layers": 6, "layer_maps": {
            "law": {"scores": [0.0, 1.0, 2.0, 0.0, 0.0, 0.0], "influence": [0.1, 1.2, 2.2, 0.1, 0.1, 0.1], "blocks": [[1, 2]]},
            "math": {"scores": [0.0, 0.0, 0.0, 0.0, 0.7, 0.0], "influence": scores, "blocks": [[4, 4]]}}})
        m, engine = make_miner(short, specs)
        nothing = {"layers": {"0": layer_rank(8, {})}}
        engine.rank_by = {
            ("mlp_cluster", "law"): {"layers": {"1": layer_rank(64, dict((u, 0.5) for u in range(10))),
                                                "3": layer_rank(64, dict((u, 0.1) for u in range(8))),
                                                "5": layer_rank(64, dict((u, 0.01) for u in range(8)))}},
            ("mlp_cluster", "math"): {"layers": {"2": layer_rank(64, dict((u, 0.3) for u in range(20, 28)))}},
            ("attn_heads", "law"): {"layers": {"0": layer_rank(8, {1: 0.2, 2: 0.2})}},
            ("attn_heads", "math"): nothing}
        engine.margins = {"facts": 0.1}                      # the model cannot do "facts": no target, no guard
        engine.drops = {"law": 0.3}
        law_block = client.part_key({"kind": "block", "layers": [1, 2]})
        math_block = client.part_key({"kind": "block", "layers": [4]})
        engine.part_drops = {law_block: {"law": 0.5, "code": 0.4}, math_block: {"math": 0.4}}
        for k in [4, 8]:                                     # math MLP parts miss the 0.15 target drop
            math_mlp = client.part_key({"kind": "mlp_cluster", "layer": 2, "indices": list(range(20, 20 + k))})
            engine.part_drops[math_mlp] = {"math": 0.1}

        args = sweep.parse_args(["--short", short, "--skills", "law,math,facts", "--mlp-sizes", "4,8", "--head-sizes", "2",
                                 "--top-layers", "2", "--max-screens", "5"])
        s = sweep.run_sweep(m, args)
        # candidates and summed scores: law MLP layer 1 (size 4: 2.0, size 8: 4.0) and layer 3 (0.4, 0.8), math MLP
        # layer 2 (1.2, 2.4), law heads (0.4), law block (3.0), math block (0.7).
        # Class means: MLP size 4 -> 1.2, MLP size 8 -> 2.4, blocks -> 1.85. Scores in the class:
        #   law MLP layer 1: 1.66667 (both sizes) | law block 1.62162 | law heads, math MLP: 1.0 |
        #   math block 0.378378 | law MLP layer 3: 0.333333
        # The default order is the best score first; the limit of 5 screens cuts after the first math MLP part.
        assert s["total"] == {"ranked": 9, "screened": 5, "passed": 3}, s
        assert s["by_kind"] == {"mlp_cluster": {"ranked": 6, "screened": 3, "passed": 2},
                                "attn_heads": {"ranked": 1, "screened": 1, "passed": 1},
                                "block": {"ranked": 2, "screened": 1, "passed": 0}}, s["by_kind"]
        assert s["by_skill"] == {"law": {"ranked": 6, "screened": 4, "passed": 3},
                                 "math": {"ranked": 3, "screened": 1, "passed": 0}}, s["by_skill"]
        assert s["new_screens"] == 5 and s["not_screened"] == 4 and s["stop"] is None and s["rank_failed"] == 0
        # two base calls: the skills and guards, then the other skills of the domain of "facts" (a guard
        # the model cannot do)
        assert engine.count("base") == 2 and engine.count("attribution_units") == 4
        assert sorted([c for c in engine.calls if c[0] == "base"][1][1][0]["skills"]) == ["creative_writing", "instruction"]
        assert engine.count("ablate") == 5 and engine.count("random_drops") == 5
        results = read_jsonl(os.path.join(common.logs_dir(short), "sweep", "results.jsonl"))
        order = [(r["skill"], r["kind"], r["size"], r["norm_score"], r["passed"]) for r in results]
        assert order == [("law", "mlp_cluster", 4, 1.66667, True), ("law", "mlp_cluster", 8, 1.66667, True),
                         ("law", "block", 2, 1.62162, False), ("law", "attn_heads", 2, 1.0, True),
                         ("math", "mlp_cluster", 4, 1.0, False)], order
        assert [r["rank_score"] for r in results] == [2.0, 4.0, 3.0, 0.4, 1.2]       # the summed score is kept
        assert "code" in " ".join(results[2]["reasons"]) and "target drop" in " ".join(results[4]["reasons"])
        rows = read_jsonl(common.catalog_path(short))
        assert [r["type"] for r in rows] == ["mlp_cluster", "mlp_cluster", "attn_heads"]
        assert [r["skill"]["name"] for r in rows] == ["law", "law", "law"]
        assert all(r["tier"] == "screened" and r["sweep"]["z_search"] >= 2.0 for r in rows)
        for r in rows:                                       # no guard from the target's domain, none the model cannot do
            assert "facts" not in r["sweep"]["guards"] and r["skill"]["name"] not in r["sweep"]["guards"]
        assert rows[0]["sweep"]["guards"] == ["math", "code", "medicine", "prose_zh", "physics"]

        # rerun: rankings, base scores, and the 5 screens are kept; 2 new screens, the next in order
        s2 = sweep.run_sweep(m, sweep.parse_args(["--short", short, "--skills", "law,math,facts", "--mlp-sizes", "4,8",
                                                  "--head-sizes", "2", "--top-layers", "2", "--max-screens", "2"]))
        assert s2["total"] == {"ranked": 9, "screened": 7, "passed": 4}, s2
        assert s2["new_screens"] == 2 and s2["not_screened"] == 2
        assert engine.count("base") == 2 and engine.count("attribution_units") == 4 and engine.count("ablate") == 7
        results = read_jsonl(os.path.join(common.logs_dir(short), "sweep", "results.jsonl"))
        assert [(r["skill"], r["kind"], r["size"], r["passed"]) for r in results[5:]] == [
            ("math", "mlp_cluster", 8, False), ("math", "block", 1, True)]
        rows = read_jsonl(common.catalog_path(short))
        assert len(rows) == 4 and rows[3]["type"] == "block" and rows[3]["skill"] == {"source": "builtin", "name": "math"}

        # a rerun after a row left the catalog: the three kept law screens still pass, and no row comes back
        with open(common.catalog_path(short), "w") as f:
            for r in rows[1:]:
                f.write(json.dumps(r) + "\n")
        s2b = sweep.run_sweep(m, sweep.parse_args(["--short", short, "--skills", "law,math,facts", "--mlp-sizes", "4,8",
                                                   "--head-sizes", "2", "--top-layers", "2", "--max-screens", "0"]))
        assert s2b["total"] == {"ranked": 9, "screened": 7, "passed": 4} and s2b["new_screens"] == 0 and engine.count("ablate") == 7
        assert [r["id"] for r in read_jsonl(common.catalog_path(short))] == [r["id"] for r in rows[1:]]
        common.append_jsonl(common.catalog_path(short), rows[0])     # put it back for the checks below

        # --order round: every (skill, kind) pair gets its best candidate before any pair gets a second one
        s2c = sweep.run_sweep(m, sweep.parse_args(["--short", short, "--skills", "law,math", "--mlp-sizes", "4,8",
                                                   "--head-sizes", "2", "--top-layers", "2", "--order", "round", "--rank-only"]))
        assert s2c["total"]["ranked"] == 9 and s2c["new_screens"] == 0
        plan = sweep.order_candidates(
            [c for skill in ["law", "math"] for c in m.ranked_candidates(skill, "mlp_cluster", [4, 8], 2, 24)[0]], "round")
        assert [(c["round"], c["skill"], c["size"]) for c in plan[:2]] == [(0, "law", 4), (0, "math", 4)]

        # --rank-only: no screen, no row
        s3 = sweep.run_sweep(m, sweep.parse_args(["--short", short, "--skills", "law,math", "--mlp-sizes", "4,8",
                                                  "--head-sizes", "2", "--top-layers", "2", "--rank-only"]))
        assert s3["total"]["ranked"] == 9 and s3["new_screens"] == 0 and engine.count("ablate") == 7
        assert len(read_jsonl(common.catalog_path(short))) == 4

        # --kinds limits the work; cached screens need no screen budget
        s4 = sweep.run_sweep(m, sweep.parse_args(["--short", short, "--skills", "law", "--kinds", "attn_heads,block",
                                                  "--head-sizes", "2", "--top-layers", "2", "--max-screens", "0"]))
        assert s4["by_kind"] == {"attn_heads": {"ranked": 1, "screened": 1, "passed": 1},
                                 "block": {"ranked": 1, "screened": 1, "passed": 0}}, s4["by_kind"]
        assert s4["new_screens"] == 0 and engine.count("ablate") == 7

        # --max-hours: the clock passes the limit after two screens
        clock = FakeClock()
        sweep.time = clock
        engine.clock = clock
        engine.seconds_per_ablate = 2000
        engine.rank_by[("mlp_cluster", "history")] = {"layers": {"4": layer_rank(64, dict((u, 0.2) for u in range(30)))}}
        engine.rank_by[("attn_heads", "history")] = nothing
        engine.drops = {"history": 0.3}
        s5 = sweep.run_sweep(m, sweep.parse_args(["--short", short, "--skills", "history", "--mlp-sizes", "2,4,8,16",
                                                  "--head-sizes", "2", "--max-hours", "1"]))
        assert s5["total"] == {"ranked": 4, "screened": 2, "passed": 2} and s5["stop"] == "time limit", s5
        # a failing engine stops the run after three screens in a row
        engine.seconds_per_ablate = 0
        engine.fail_parts = set(client.part_key({"kind": "mlp_cluster", "layer": 4, "indices": list(range(k))})
                                for k in [2, 4, 8, 16])
        engine.rank_by[("mlp_cluster", "physics")] = {"layers": {"4": layer_rank(64, dict((u, 0.2) for u in range(30)))}}
        engine.rank_by[("attn_heads", "physics")] = nothing
        s6 = sweep.run_sweep(m, sweep.parse_args(["--short", short, "--skills", "physics", "--mlp-sizes", "2,4,8,16",
                                                  "--head-sizes", "2"]))
        assert s6["stop"] == "3 screens in a row failed" and s6["total"] == {"ranked": 4, "screened": 0, "passed": 0}, s6

        # one ranking that fails does not end the run: the other pairs are ranked and screened
        engine.fail_parts = set()
        engine.fail_ranks = set([("mlp_cluster", "chemistry")])
        engine.rank_by[("attn_heads", "chemistry")] = nothing
        engine.rank_by[("mlp_cluster", "biology")] = {"layers": {"3": layer_rank(64, dict((u, 0.2) for u in range(40, 50)))}}
        engine.rank_by[("attn_heads", "biology")] = nothing
        engine.drops = {"biology": 0.3}
        ranked_before = engine.count("attribution_units")
        s7 = sweep.run_sweep(m, sweep.parse_args(["--short", short, "--skills", "chemistry,biology", "--mlp-sizes", "4",
                                                  "--head-sizes", "2"]))
        assert s7["rank_failed"] == 1 and s7["total"] == {"ranked": 1, "screened": 1, "passed": 1} and s7["stop"] is None, s7
        assert engine.count("attribution_units") == ranked_before + 4
        assert len([n for n in s7["notes"] if n.startswith("ranking failed, no candidates for chemistry mlp_cluster")]) == 1
        # a rerun tries the failed ranking again (it was not kept) and uses the three kept ones
        s7b = sweep.run_sweep(m, sweep.parse_args(["--short", short, "--skills", "chemistry,biology", "--mlp-sizes", "4",
                                                   "--head-sizes", "2"]))
        assert s7b["rank_failed"] == 1 and engine.count("attribution_units") == ranked_before + 5
        # three failed rankings in a row stop the ranking; what was ranked before is still screened
        engine.fail_ranks = set((kind, skill) for kind in ["mlp_cluster", "attn_heads"]
                                for skill in ["economics_business", "philosophy_ethics"])
        s8 = sweep.run_sweep(m, sweep.parse_args(["--short", short, "--skills", "biology,economics_business,philosophy_ethics",
                                                  "--mlp-sizes", "4", "--head-sizes", "2"]))
        assert s8["rank_failed"] == 3 and s8["total"] == {"ranked": 1, "screened": 1, "passed": 1}, s8
        assert any(n.startswith("3 rankings in a row failed") for n in s8["notes"])
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
        common.ROOT = real_root
        sweep.POOL_ENTRIES = real_entries
        sweep.time = real_time
        if old_lane is None:
            del os.environ["WSP_LANE"]
        else:
            os.environ["WSP_LANE"] = old_lane


def test_guard_limits(specs):
    # A model that can do few guard skills: the skill is skipped below --min-guards, and a short guard
    # set is named in the summary.
    real_root = common.ROOT
    old_lane = os.environ.get("WSP_LANE")
    tmp = tempfile.mkdtemp(prefix="sweep-guards-")
    common.ROOT = tmp
    os.environ["WSP_LANE"] = "sweep"
    try:
        short = "toy"
        layers = [{"layer": i, "attn": {"n_heads": 8, "head_dim": 8}, "mlp": {"width": 64}} for i in range(4)]
        common.write_json_atomic(common.state_path(short), {
            "revision": "rev-main", "model": "org/Toy", "structure": {"n_layers": 4, "d_model": 64, "layers_detail": layers}})
        m, engine = make_miner(short, specs)
        engine.rank_result = {"layers": {"1": layer_rank(64, dict((u, 0.2) for u in range(10)))}}
        engine.drops = {"history": 0.3}
        can = ["history", "general_text", "math", "code", "chemistry"]
        engine.margins = dict((n, 0.1) for n in specs if n not in can)
        base = ["--short", short, "--kinds", "mlp_cluster", "--mlp-sizes", "4"]
        # history (domain society): the guards are math, code, and chemistry in place of physics. The domains
        # medicine, language, and general have no skill the model can do. 3 guards: with --min-guards 4 the
        # skill is skipped before any ranking.
        s = sweep.run_sweep(m, sweep.parse_args(base + ["--skills", "history", "--min-guards", "4"]))
        assert s["total"] == {"ranked": 0, "screened": 0, "passed": 0} and engine.count("attribution_units") == 0
        assert len([n for n in s["notes"] if n.startswith("skipped 'history': only 3 guard skills")]) == 1, s["notes"]
        assert not os.path.exists(common.catalog_path(short))
        # the default minimum is 3: the skills run, and the summary says that the guard set is short
        s = sweep.run_sweep(m, sweep.parse_args(base + ["--skills", "history"]))
        assert s["total"] == {"ranked": 1, "screened": 1, "passed": 1}
        assert "'history' is screened with 3 guard skills, not 5: math, code, chemistry" in s["notes"]
        rows = read_jsonl(common.catalog_path(short))
        assert len(rows) == 1 and rows[0]["sweep"]["guards"] == ["math", "code", "chemistry"]
        assert sorted(k.split("|")[0] for k in rows[0]["signature"]["effect"] if k.endswith("|search")) == [
            "chemistry", "code", "general_text", "history", "math"]
        # no usable guard at all: nothing is ranked, screened, or written
        m2, engine2 = make_miner("toy2", specs)
        m2.state = {"revision": "rev-main", "model": "org/Toy", "structure": m.state["structure"]}
        engine2.rank_result = engine.rank_result
        engine2.drops = {"history": 0.3}
        engine2.margins = dict((n, 0.1) for n in specs if n not in ["history", "general_text"])
        s = sweep.run_sweep(m2, sweep.parse_args(["--short", "toy2", "--kinds", "mlp_cluster", "--mlp-sizes", "4",
                                                  "--skills", "history"]))
        assert s["total"]["ranked"] == 0 and engine2.count("ablate") == 0 and not os.path.exists(common.catalog_path("toy2"))
        assert any("only 0 guard skills can be scored (none)" in n for n in s["notes"]), s["notes"]
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
        common.ROOT = real_root
        if old_lane is None:
            del os.environ["WSP_LANE"]
        else:
            os.environ["WSP_LANE"] = old_lane


def test_seed_from_lanes():
    # run_many --agents N writes state.lane0.json, state.lane1.json, and no state.json. The sweep must
    # take the agents' revision from those files, or it loads a second copy of the model.
    real_root = common.ROOT
    old_lane = os.environ.get("WSP_LANE")
    tmp = tempfile.mkdtemp(prefix="sweep-seed-")
    common.ROOT = tmp
    os.environ["WSP_LANE"] = "sweep"
    try:
        short = "toy"
        folder = common.logs_dir(short)
        lane = common.state_path(short)
        assert sweep.seed_lane_state(short) is None and not os.path.exists(lane)        # no agent run at all
        for i, structure in [(0, {"n_layers": 4}), (1, {"n_layers": 4, "newer": True})]:
            path = os.path.join(folder, "state.lane" + str(i) + ".json")
            common.write_json_atomic(path, {"revision": "rev-agents", "model": "org/Toy", "structure": structure,
                                            "screens": {"k": 1}})
            os.utime(path, (1000 + i, 1000 + i))
        common.write_json_atomic(os.path.join(folder, "state.lane2.json"), {"model": "org/Toy"})    # no revision yet
        open(os.path.join(folder, "state.lane0.json.lock"), "w").close()
        assert sweep.seed_lane_state(short) == "state.lane1.json"                        # the newest lane
        assert common.read_json_safe(lane, None) == {"revision": "rev-agents", "model": "org/Toy",
                                                     "structure": {"n_layers": 4, "newer": True}}
        assert sweep.seed_lane_state(short) is None                                      # an existing lane state is kept
        # lanes that pin different revisions: the sweep stops and names them
        os.remove(lane)
        if os.path.exists(lane + ".bak"):
            os.remove(lane + ".bak")
        common.write_json_atomic(os.path.join(folder, "state.lane2.json"), {"revision": "rev-other", "model": "org/Toy"})
        try:
            sweep.seed_lane_state(short)
            raise AssertionError("two revisions must stop the sweep")
        except SystemExit as e:
            assert "rev-agents" in str(e) and "rev-other" in str(e) and "state.lane2.json" in str(e)
        assert not os.path.exists(lane)
        # state.json, when it has a revision, is the source as before
        common.write_json_atomic(os.path.join(folder, "state.json"), {"revision": "rev-main", "model": "org/Toy"})
        assert sweep.seed_lane_state(short) == "state.json"
        assert common.read_json_safe(lane, None) == {"revision": "rev-main", "model": "org/Toy"}
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
        common.ROOT = real_root
        if old_lane is None:
            del os.environ["WSP_LANE"]
        else:
            os.environ["WSP_LANE"] = old_lane


if __name__ == "__main__":
    all_specs = sweep.load_skill_specs()      # the real built-in and pool skill sets
    test_mlp_candidates()
    test_head_and_expert_candidates()
    test_block_candidates()
    test_z_score()
    test_z_rule()
    test_order_is_size_neutral()
    test_order_and_cutoff()
    test_guards_and_payload(all_specs)
    test_labels_and_summary()
    test_add_row_fields()
    test_miner_paths(all_specs)
    test_run_sweep(all_specs)
    test_guard_limits(all_specs)
    test_seed_from_lanes()
    print("ok")
