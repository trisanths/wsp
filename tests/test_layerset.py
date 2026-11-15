import os
import sys

import numpy as np
import pytest
import torch

transformers = pytest.importorskip("transformers")

from wsp import layerset, probe, weave
from wsp.layerset import LayerScorer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

LAYERS = 12

TEXTS_A = ["alpha beta gamma delta epsilon zeta eta theta", "one two three four five six seven",
           "red green blue yellow orange purple pink brown black", "north south east west up down left right",
           "cat dog bird fish horse cow pig sheep goat", "sun moon star planet comet galaxy nebula quasar"]
TEXTS_B = ["monday tuesday wednesday thursday friday saturday sunday", "apple banana cherry grape lemon mango peach",
           "iron copper silver gold zinc nickel lead tin", "spring summer autumn winter rain snow wind storm",
           "hand foot knee elbow wrist ankle shoulder neck", "river lake ocean sea pond brook creek stream"]
TEXTS_GENERAL = ["the quick brown fox jumps over the lazy dog", "a small step for a man", "to be or not to be"]


class Tok:
    """Whitespace tokenizer into ids 1..30 (the tiny models have vocab 100)."""

    pad_token = "<pad>"
    eos_token = "<pad>"
    pad_token_id = 0
    padding_side = "right"
    chat_template = None

    def __call__(self, texts, return_tensors="pt", padding=True, truncation=True, max_length=64,
                 add_special_tokens=True):
        rows = [[1 + (sum(ord(c) for c in word) % 30) for word in text.split()][:max_length] for text in texts]
        width = max(len(row) for row in rows)
        ids = torch.zeros(len(rows), width, dtype=torch.long)
        mask = torch.zeros(len(rows), width, dtype=torch.long)
        for index, row in enumerate(rows):
            ids[index, :len(row)] = torch.tensor(row)
            mask[index, :len(row)] = 1
        return {"input_ids": ids, "attention_mask": mask}


def tiny(dtype=torch.float32, layers=LAYERS):
    torch.manual_seed(0)
    config = transformers.Qwen2Config(hidden_size=32, intermediate_size=64, num_hidden_layers=layers,
                                      num_attention_heads=4, num_key_value_heads=2, vocab_size=100,
                                      tie_word_embeddings=False)
    return transformers.Qwen2ForCausalLM(config).to(dtype).eval()


def planted(host, layers, seed, scale=0.05, other=0.0):
    """A member: the host plus noise on the listed layers (attention and MLP), and `other` noise elsewhere."""
    out = tiny(host.dtype, host.config.num_hidden_layers)
    out.load_state_dict(host.state_dict())
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for name, param in out.named_parameters():
            info = weave.classify(name)
            if info is None or info[0] not in ("attn", "mlp"):
                continue
            amount = scale if info[1] in layers else other
            if amount == 0.0:
                continue
            noise = torch.randn(param.shape, generator=generator) * amount
            param.copy_((param.float() + noise).to(param.dtype))
    return out


def member_of(name, spec, host):
    return weave.member_from_state(name, name, spec.state_dict(), host, groups=("attn", "mlp"))


def snapshot(model):
    return {k: v.detach().clone() for k, v in model.state_dict().items()}


def same(a, b):
    return all(torch.equal(a[k], b[k]) for k in a)


def setup(specs, dtype=torch.float32, texts=None):
    """host, weaver, joint objective, one scorer per call. specs: {name: planted layers}."""
    host = tiny(dtype)
    tok = Tok()
    members = []
    caches = {}
    for index, (name, layers) in enumerate(specs.items()):
        spec = planted(host, layers, seed=10 + index)
        members.append(member_of(name, spec, host))
        pool = (texts or {}).get(name, TEXTS_A if index % 2 == 0 else TEXTS_B)
        caches[name] = weave.build_teacher_cache(spec, tok, pool, top_k=64, max_length=32, batch_size=4)
    general = weave.build_teacher_cache(host, tok, TEXTS_GENERAL, top_k=64, max_length=32, batch_size=4)
    objective = weave.ProbeObjective(list(specs), caches, general, batch_size=4)
    weaver = layerset.layer_weaver(host, members)
    objective.calibrate(host)
    return host, weaver, objective


A_IDS = list(range(1, 9))
B_IDS = list(range(16, 24))


def token_texts(ids, count, seed):
    """Texts whose words all map to the given token ids."""
    pool = [w for w in ("w%d" % i for i in range(2000)) if 1 + (sum(ord(c) for c in w) % 30) in ids]
    rng = np.random.default_rng(seed)
    return [" ".join(rng.choice(pool, 8)) for _ in range(count)]


def token_host(dtype):
    """A host whose group A tokens live in embedding dims 0 to 15 and group B tokens in dims 16 to 31."""
    host = tiny(dtype)
    with torch.no_grad():
        table = host.get_input_embeddings().weight
        for i in A_IDS:
            table[i, 16:] = 0
        for i in B_IDS:
            table[i, :16] = 0
    return host


def planted_token(host, layers, ids, seed, scale):
    """A member whose difference acts on one token group: rank one writes along that group's embedding direction.

    The write goes into gate, up, and value projections of each listed layer, so attention and MLP both carry
    the skill, and the other group's texts are barely touched.
    """
    out = tiny(host.dtype, host.config.num_hidden_layers)
    out.load_state_dict(host.state_dict())
    generator = torch.Generator().manual_seed(seed)
    table = host.get_input_embeddings().weight.float()
    direction = table[list(ids)]
    direction = (direction / direction.norm(dim=1, keepdim=True)).mean(0)
    direction = direction / direction.norm()
    with torch.no_grad():
        for layer in layers:
            block = out.model.layers[layer]
            for param in (block.mlp.gate_proj.weight, block.mlp.up_proj.weight, block.self_attn.v_proj.weight):
                a = torch.randn(param.shape[0], generator=generator) * scale
                param.copy_((param.float() + a[:, None] * direction[None, :]).to(param.dtype))
    return out


def setup_tokens(dtype, plan):
    """Two members on separate token groups. plan: [(name, layers, ids, scale, texts)]."""
    host = token_host(dtype)
    tok = Tok()
    members = []
    caches = {}
    for index, (name, layers, ids, scale, texts) in enumerate(plan):
        spec = planted_token(host, layers, ids, 1 + index, scale)
        members.append(member_of(name, spec, host))
        caches[name] = weave.build_teacher_cache(spec, tok, texts, top_k=64, max_length=32, batch_size=4)
    general = weave.build_teacher_cache(host, tok, TEXTS_GENERAL, top_k=64, max_length=32, batch_size=4)
    objective = weave.ProbeObjective([p[0] for p in plan], caches, general, batch_size=4)
    weaver = layerset.layer_weaver(host, members)
    objective.calibrate(host)
    return host, weaver, objective


def scorer_for(host, weaver, objective, names):
    return LayerScorer(weaver, host, layerset.restrict_objective(objective, names))


# ---------------------------------------------------------------- planted skill


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_importance_finds_planted_layers_and_minimal_set_returns_them(dtype):
    host, weaver, objective = setup({"a": [2, 9]}, dtype)
    assert weaver.n_buckets == weaver.n_layers == LAYERS
    scorer = scorer_for(host, weaver, objective, ["a"])
    result = layerset.layer_importance(scorer, "a")
    assert sorted(result["ranking"][:2]) == [2, 9]
    by_layer = {row["layer"]: row for row in result["layers"]}
    for layer in range(LAYERS):
        if layer not in (2, 9):
            assert abs(by_layer[layer]["both"]["transfer"]) < 1e-6
            assert abs(by_layer[layer]["attn"]["transfer"]) < 1e-6
    for layer in (2, 9):
        assert by_layer[layer]["both"]["transfer"] > 0.1
        assert "harm" in by_layer[layer]["both"]
        assert by_layer[layer]["attn"]["transfer"] is not None
        assert by_layer[layer]["mlp"]["transfer"] is not None
    assert result["full"]["transfer"] == pytest.approx(1.0, abs=0.02)
    found = layerset.minimal_set(scorer, "a", result["ranking"], target_share=0.9)
    assert found["status"] == "ok"
    assert found["layers"] == [2, 9]
    assert found["share"] >= 0.9
    assert found["within_half"] and not found["stop_rule"]


def test_random_sets_of_the_same_size_keep_less():
    host, weaver, objective = setup({"a": [2, 9]})
    scorer = scorer_for(host, weaver, objective, ["a"])
    ranking = layerset.layer_importance(scorer, "a", kinds=("both",))["ranking"]
    found = layerset.minimal_set(scorer, "a", ranking)
    control = layerset.random_set_control(scorer, "a", found["size"], seeds=range(1, 9), reference=found["layers"])
    assert len(control["runs"]) == 8
    for run in control["runs"]:
        assert len(run["layers"]) == found["size"]
    assert control["share"]["mean"] < 0.5 * found["share"]
    assert control["share"]["mean"] < found["share"] - 0.4
    # a random set that misses both planted layers carries nothing
    empty = [r for r in control["runs"] if r["overlap"] == 0]
    assert len(empty) > 0
    for run in empty:
        assert abs(run["share"]) < 1e-3


def test_norm_cross_and_scrambled_controls():
    host, weaver, objective = setup({"a": [1, 4], "b": [7, 10]})
    scorer_a = scorer_for(host, weaver, objective, ["a"])
    scorer_b = scorer_for(host, weaver, objective, ["b"])
    ranking_a = layerset.layer_importance(scorer_a, "a", kinds=("both",))["ranking"]
    set_a = layerset.minimal_set(scorer_a, "a", ranking_a)["layers"]
    ranking_b = layerset.layer_importance(scorer_b, "b", kinds=("both",))["ranking"]
    set_b = layerset.minimal_set(scorer_b, "b", ranking_b)["layers"]
    assert set_a == [1, 4] and set_b == [7, 10]
    # chosen by norm only: the planted layers have all the difference norm
    by_norm = layerset.norm_set_control(scorer_a, "a", 2, reference=set_a)
    assert by_norm["layers"] == [1, 4] and by_norm["share"] >= 0.9
    # a's difference on b's layers is zero there, so nothing transfers
    cross = layerset.cross_skill_control(scorer_a, "a", set_b)
    assert abs(cross["share"]) < 1e-3
    scrambled = layerset.scrambled_control(scorer_a, "a", set_a, seeds=[1, 2, 3])
    assert scrambled["share"]["mean"] < 0.5
    assert layerset.host_matches(weaver, host)


def test_stop_rule_when_the_skill_is_spread_over_every_layer():
    host = tiny()
    tok = Tok()
    spec = planted(host, [], seed=5, scale=0.0, other=0.03)
    member = member_of("a", spec, host)
    cache = weave.build_teacher_cache(spec, tok, TEXTS_A, top_k=64, max_length=32, batch_size=4)
    general = weave.build_teacher_cache(host, tok, TEXTS_GENERAL, top_k=64, max_length=32, batch_size=4)
    objective = weave.ProbeObjective(["a"], {"a": cache}, general, batch_size=4)
    weaver = layerset.layer_weaver(host, [member])
    objective.calibrate(host)
    scorer = LayerScorer(weaver, host, objective)
    ranking = layerset.layer_importance(scorer, "a", kinds=("both",))["ranking"]
    found = layerset.minimal_set(scorer, "a", ranking, target_share=0.9)
    assert found["size"] > 0.5 * LAYERS
    assert found["size_share"] == found["size"] / LAYERS
    assert found["stop_rule"] == (found["size"] > 0.75 * LAYERS)


# ---------------------------------------------------------------- composition


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_compose_keeps_both_planted_skills(dtype):
    plan = [("a", [1, 4], A_IDS, 0.2, token_texts(A_IDS, 6, 0)), ("b", [7, 10], B_IDS, 0.6, token_texts(B_IDS, 6, 1))]
    host, weaver, objective = setup_tokens(dtype, plan)
    scorer = LayerScorer(weaver, host, objective)
    result = layerset.compose(scorer, {"a": [1, 4], "b": [7, 10]})
    assert result["assignment"] == {"a": [1, 4], "b": [7, 10]}
    assert result["overlap"] == []
    for name in ("a", "b"):
        solo = scorer.full(name)["transfer"][name]
        assert solo > 0.9
        assert result["result"]["transfer"][name] >= 0.75 * solo
    # the skills are separate: a written alone does nothing for b's texts
    assert scorer.full("a")["transfer"]["b"] < 0.3
    assert result["result"]["harm"] < 0.1
    assert layerset.host_matches(weaver, host)


def test_compose_resolves_overlap_by_norm_and_by_probe():
    # layer 5 sits in both sets. b changes it much more than a.
    plan = [("a", [1, 5], A_IDS, 0.2, token_texts(A_IDS, 6, 0)), ("b", [5, 9], B_IDS, 0.6, token_texts(B_IDS, 6, 1))]
    host, weaver, objective = setup_tokens(torch.float32, plan)
    scorer = LayerScorer(weaver, host, objective)
    assert layerset.layer_norms(weaver, 1)["both"][5] > layerset.layer_norms(weaver, 0)["both"][5]
    by_norm = layerset.compose(scorer, {"a": [1, 5], "b": [5, 9]}, resolve="norm")
    assert by_norm["overlap"] == [5]
    assert by_norm["assignment"] == {"a": [1], "b": [5, 9]}
    assert by_norm["decisions"][0]["by_norm"] == "b"
    both = layerset.compose(scorer, {"a": [1, 5], "b": [5, 9]}, resolve="try_both")
    assert both["value"] >= by_norm["value"] - 1e-12
    assert sorted(l for ls in both["assignment"].values() for l in ls) == [1, 5, 9]
    assert both["decisions"][0]["by_probe"] in ("a", "b")
    assert layerset.host_matches(weaver, host)


# ---------------------------------------------------------------- the host stays exact


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_host_is_restored_exactly_after_many_probes(dtype):
    host, weaver, objective = setup({"a": [2, 9], "b": [4, 6]}, dtype)
    before = snapshot(host)
    scorer = LayerScorer(weaver, host, objective)
    assert same(before, snapshot(host))
    rng = np.random.default_rng(0)
    for step in range(40):
        size = int(rng.integers(1, LAYERS + 1))
        assignment = {0: rng.choice(LAYERS, size, replace=False).tolist(),
                      1: rng.choice(LAYERS, size, replace=False).tolist()}
        kinds = ["both", "attn", "mlp"][step % 3]
        scorer.score(assignment, kinds)
        if step % 7 == 0:
            scorer.score(assignment, kinds, weaver=weaver.shuffled(step))
        assert same(before, snapshot(host)), f"drift at step {step}"
    layerset.layer_importance(scorer, "a")
    layerset.compose(scorer, {"a": [2, 9], "b": [4, 9]}, resolve="try_both")
    assert same(before, snapshot(host))
    assert layerset.host_matches(weaver, host)


def test_restricted_objective_matches_the_joint_one():
    host, weaver, objective = setup({"a": [2, 9], "b": [4, 6]})
    joint = LayerScorer(weaver, host, objective)
    only_a = scorer_for(host, weaver, objective, ["a"])
    assignment = {"a": [2], "b": [4]}
    full = joint.score(assignment)
    one = only_a.score(assignment)
    assert list(one["transfer"]) == ["a"]
    assert one["transfer"]["a"] == pytest.approx(full["transfer"]["a"])
    assert one["harm"] == pytest.approx(full["harm"])
    with pytest.raises(ValueError):
        layerset.restrict_objective(objective, ["c"])


def test_memo_skips_repeat_forward_passes():
    host, weaver, objective = setup({"a": [2, 9]})
    scorer = scorer_for(host, weaver, objective, ["a"])
    scorer.score({"a": [2]})
    count = scorer.forward_sets
    scorer.score({0: [2]})
    scorer.score({"a": [2, 2]})
    assert scorer.forward_sets == count


def test_layer_genome_and_bad_inputs():
    genome = layerset.layer_genome(2, 4, {1: [0, 3]}, "attn")
    assert genome.weights[1, 0, 0] == 1.0 and genome.weights[1, 0, 1] == 0.0
    assert genome.weights[0].sum() == 0.0
    with pytest.raises(ValueError):
        layerset.layer_genome(2, 4, {1: [4]})
    with pytest.raises(ValueError):
        layerset.layer_genome(2, 4, {1: [0]}, "norm")
    full = layerset.full_member_genome(2, 4, 1)
    assert full.weights[1].min() == 1.0 and full.weights[0].max() == 0.0
    assert layerset.random_layer_sets(10, 3, [1, 1]) == layerset.random_layer_sets(10, 3, [1, 1])


# ---------------------------------------------------------------- the Modal script (CPU parts only)


def load_round25():
    pytest.importorskip("modal")
    if ROOT not in sys.path:
        sys.path.insert(0, ROOT)
    import modal_round25

    return modal_round25


def test_round25_plan_and_smoke_fit_their_budgets():
    r25 = load_round25()
    full = r25.make_plan("qwen7b", smoke=False)
    smoke = r25.make_plan("qwen7b", smoke=True)
    assert smoke["gpu_hours"] < 1.0
    assert full["gpu_hours"] > smoke["gpu_hours"]
    assert smoke["arms"] == ["host", "member:ocr_nemotron", "minimal:ocr_nemotron", "random:ocr_nemotron"]
    assert len(full["arms"]) == 9
    assert smoke["benchmarks"] == ["mbpp"]


def test_round25_go_rules():
    r25 = load_round25()
    base = dict(size_share=0.4, share_c=0.95, random_share=0.2, member_score=0.60, set_score=0.57)
    assert r25.member_go(**base)["go"]
    for bad in (dict(size_share=0.6), dict(share_c=0.8), dict(random_share=0.6), dict(set_score=0.50),
                dict(share_c=None)):
        args = dict(base)
        args.update(bad)
        assert not r25.member_go(**args)["go"]
    assert "within_5_points_on_gsm8k" in r25.member_go(skill="gsm8k", **base)["checks"]
    assert r25.member_stop(size_share=0.8)["stop"]
    assert not r25.member_stop(size_share=0.5)["stop"]


def test_round25_stage_rows_resume_and_arms(tmp_path, monkeypatch):
    r25 = load_round25()
    monkeypatch.setattr(r25, "OUT_ROOT", str(tmp_path))
    monkeypatch.setattr(r25.volume, "commit", lambda: None, raising=False)
    calls = []
    stages = r25.Stages("cell", set(), False)
    first = stages.get("importance_m", lambda: calls.append(1) or {"value": 1})
    again = stages.get("importance_m", lambda: calls.append(2) or {"value": 2})
    assert calls == [1] and again["value"] == 1 and first["fit_version"] == r25.FIT_VERSION
    only_bench = r25.Stages("cell", {"bench"}, False)
    with pytest.raises(RuntimeError):
        only_bench.get("minimal_m", lambda: {})
    assert only_bench.get("importance_m", lambda: {})["value"] == 1
    assert r25.Stages("cell", {"importance"}, True).get("importance_m", lambda: {"value": 3})["value"] == 3
    assert r25.load_stage("cell", "bench_member:x") is None
    r25.save_stage("cell", "bench_member:x", {"scores": {}})
    assert r25.load_stage("cell", "bench_member:x")["stage"] == "bench_member:x"
    assert (tmp_path / "cell" / "rows.jsonl").read_text().count("\n") == 3
    assert r25.arms_for("qwen7b", smoke=True)[0] == "host"
    assert r25.cell_name("qwen7b", True) == "qwen7b-smoke"
    assert [m["name"] for m in r25.cell_family("qwen7b", True)["members"]] == ["ocr_nemotron"]


def test_round25_whole_cell_on_tiny_cpu_models(tmp_path, monkeypatch):
    """Run run_cell end to end on tiny models, with the Hub, the GPU, and the benchmarks replaced by stand ins."""
    r25 = load_round25()
    r21 = r25.r21
    layers = 6
    host = tiny(torch.bfloat16, layers)
    specs = {"openthinker": planted(host, [1, 4], 21), "ocr_nemotron": planted(host, [2, 5], 22)}
    dirs = {}
    host_dir = tmp_path / "host"
    host.save_pretrained(host_dir)
    host.config.save_pretrained(host_dir)
    dirs["Qwen/Qwen2.5-7B-Instruct"] = str(host_dir)
    ids = {"openthinker": "open-thoughts/OpenThinker-7B", "ocr_nemotron": "nvidia/OpenCodeReasoning-Nemotron-7B"}
    for name, spec in specs.items():
        spec.save_pretrained(tmp_path / name)
        dirs[ids[name]] = str(tmp_path / name)
    reference_dir = tmp_path / "reference"
    planted(host, [3], 23).save_pretrained(reference_dir)
    dirs["Qwen/Qwen2.5-Coder-7B-Instruct"] = str(reference_dir)

    class ChatTok(Tok):
        chat_template = "template"

        def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
            return messages[0]["content"]

    tok = ChatTok()
    pools = {"openthinker": TEXTS_A, "ocr_nemotron": TEXTS_B}
    cache_args = dict(top_k=64, max_length=32, batch_size=4)

    def fake_texts(tokenizer, fam):
        splits = {s: {"general": TEXTS_GENERAL, "members": {m["name"]: pools[m["name"]] for m in fam["members"]}}
                  for s in r21.PROBE_SPLITS}
        return {"splits": splits, "choices": {m["name"]: {"pools": ["code"]} for m in fam["members"]}, "ranges": {}}

    def fake_caches(fam, tokenizer, revisions, data):
        host_cache = {s: {"general": weave.build_teacher_cache(host, tok, TEXTS_GENERAL, **cache_args)}
                      for s in r21.PROBE_SPLITS}
        members = {}
        for m in fam["members"]:
            members[m["name"]] = {}
            for s in r21.PROBE_SPLITS:
                host_cache[s][m["name"]] = weave.build_teacher_cache(host, tok, pools[m["name"]], **cache_args)
                members[m["name"]][s] = weave.build_teacher_cache(specs[m["name"]], tok, pools[m["name"]], **cache_args)
        return {"host": host_cache, "members": members, "vocab": 100}

    real_load = transformers.AutoModelForCausalLM.from_pretrained

    def load(path, *args, **kwargs):
        kwargs.pop("device_map", None)
        return real_load(path, *args, **kwargs)

    monkeypatch.setattr(r25, "OUT_ROOT", str(tmp_path / "out"))
    monkeypatch.setattr(r25, "LAYERS", {"qwen7b": layers})
    monkeypatch.setattr(r25, "safe_reload", lambda cell="": None)
    monkeypatch.setattr(r25, "check_probe_sources", lambda fam, data: None)
    monkeypatch.setattr(r25.volume, "commit", lambda: None, raising=False)
    monkeypatch.setattr(r21, "download", lambda repo, *a, **k: dirs[repo])
    monkeypatch.setattr(r21, "revision_of", lambda repo: "rev")
    monkeypatch.setattr(r21, "probe_texts_for", fake_texts)
    monkeypatch.setattr(r21, "build_probe_caches", fake_caches)
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", lambda *a, **k: tok)
    monkeypatch.setattr(transformers.AutoModelForCausalLM, "from_pretrained", load)
    monkeypatch.setattr(weave, "gate_member", lambda *a, **k: {"passed": True, "cos_mean": 1.0})
    monkeypatch.setattr(r25, "final_items", lambda ev, skills, cell: {
        s: [{"text": "t", "test_list": ["a"], "prompt": "p", "question": "q"}] for s in skills})
    scored = []

    def fake_score_arm(ev, model, tokenizer, items, skills, cell, arm):
        scored.append(arm)
        return {s: {"score": 0.5, "correct": 1, "n": 2} for s in skills}

    monkeypatch.setattr(r25, "score_arm", fake_score_arm)

    row = r25.run_cell("qwen7b", smoke=True, seeds=3)
    cell = "qwen7b-smoke"
    assert scored == ["host", "member:ocr_nemotron", "minimal:ocr_nemotron", "random:ocr_nemotron"]
    out = tmp_path / "out" / cell
    for stage in ("caches", "gates", "importance_ocr_nemotron", "minimal_ocr_nemotron", "controls_ocr_nemotron",
                  "summary", "bench_host", "bench_random_ocr_nemotron"):
        assert (out / (stage + ".json")).exists() or (out / (stage.replace(":", "_") + ".json")).exists(), stage
    minimal = r25.load_stage(cell, "minimal_ocr_nemotron")
    assert minimal["layers"] == [2, 5]
    assert row["members"]["ocr_nemotron"]["size"] == 2
    assert row["members"]["ocr_nemotron"]["go"]["checks"]["size_at_most_half"]
    # a rerun reads every row and scores nothing again
    scored.clear()
    r25.run_cell("qwen7b", smoke=True, seeds=3)
    assert scored == []

    # the whole cell: two members, composition, the reference arm
    scored.clear()
    row = r25.run_cell("qwen7b", smoke=False, seeds=3)
    assert sorted(scored) == sorted(r25.arms_for("qwen7b"))
    compose = r25.load_stage("qwen7b", "compose")
    assert compose["assignment"] == {"openthinker": [1, 4], "ocr_nemotron": [2, 5]}
    assert row["composed_keeps_of_member"]["ocr_nemotron"] == pytest.approx(1.0)
    assert "cross_skill" in r25.load_stage("qwen7b", "controls_openthinker")


def test_round25_scoring_prints_two_raw_generations_with_the_cell_name(capsys):
    r25 = load_round25()

    class Ev:
        @staticmethod
        def render_prompt(tokenizer, text):
            return text

        @staticmethod
        def user_text(skill, item):
            return item["text"]

        @staticmethod
        def generate_texts(model, tokenizer, prompts, max_new_tokens, batch_size):
            assert max_new_tokens == r25.MAX_NEW_LONG["mbpp"] == 2048
            return ["first\nanswer", "second\nanswer", "third"]

        @staticmethod
        def grade_outputs(skill, items, outputs):
            return [{"correct": i != 1} for i in range(len(outputs))]

        @staticmethod
        def evaluate_skill(model, tokenizer, skill, items=None):
            return {"score": 0.25, "correct": 1, "n": 4}

    result = r25.score_skill(Ev, None, Tok(), "mbpp", [{"text": "a"}] * 3, "qwen7b-smoke", "host")
    assert result["correct"] == 2 and result["n"] == 3 and result["reached_cap"] == 0
    r25.score_skill(Ev, None, Tok(), "mmlu", [{}] * 4, "qwen7b-smoke", "host")
    lines = [line for line in capsys.readouterr().out.splitlines() if line.strip()]
    assert len(lines) > 0 and all(line.startswith("[qwen7b-smoke] ") for line in lines)
    assert len([line for line in lines if " raw host mbpp #" in line]) == 2
