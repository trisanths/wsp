import json
import os

import numpy as np
import pytest
import torch

transformers = pytest.importorskip("transformers")

from wsp import weave
from wsp.weave import (Genome, Objective, SepCMAES, Weaver, classify, member_from_state, one_hot_genome,
                       search, task_arithmetic_genome, trim_top, uniform_genome, zero_genome)
from wsp.weave import SepCMAES as _RealCMA


def tiny(seed, dtype=torch.bfloat16, rope_theta=10000.0, tie=False):
    torch.manual_seed(seed)
    config = transformers.Qwen2Config(hidden_size=32, intermediate_size=64, num_hidden_layers=4,
                                      num_attention_heads=4, num_key_value_heads=2, vocab_size=100,
                                      rope_theta=rope_theta, tie_word_embeddings=tie)
    return transformers.Qwen2ForCausalLM(config).to(dtype).eval()


def perturbed(model, seed, scale=0.02):
    """A specialist: the host plus seeded noise, in the host's dtype."""
    out = tiny(0, model.dtype, tie=model.config.tie_word_embeddings)
    out.load_state_dict(model.state_dict())
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for param in out.parameters():
            noise = torch.randn(param.shape, generator=generator) * scale
            param.copy_((param.float() + noise).to(param.dtype))
    return out


def build(host, specialists, **kwargs):
    members = []
    for index, spec in enumerate(specialists):
        members.append(member_from_state("m%d" % index, "skill%d" % index, spec.state_dict(), host,
                                         groups=("attn", "mlp", "embed", "norm")))
    return Weaver(host, members, n_buckets=2, **kwargs)


def snapshot(model):
    return {k: v.detach().clone() for k, v in model.state_dict().items()}


def same(a, b):
    for key in a:
        if not torch.equal(a[key], b[key]):
            return False
    return True


def test_classify():
    assert classify("model.layers.3.self_attn.q_proj.weight") == ("attn", 3)
    assert classify("model.layers.3.mlp.down_proj.weight") == ("mlp", 3)
    assert classify("model.layers.3.input_layernorm.weight") == ("norm", None)
    assert classify("model.embed_tokens.weight") == ("embed", None)
    assert classify("lm_head.weight") == ("embed", None)
    assert classify("model.norm.weight") == ("norm", None)
    assert classify("model.vision_tower.encoder.layers.0.mlp.fc1.weight") is None
    assert classify("layers.5.feed_forward.w1.weight") == ("mlp", 5)
    assert classify("model.language_model.layers.2.self_attn.q_proj.weight") == ("attn", 2)


def test_zero_genome_equals_host_bit_for_bit():
    host = tiny(0)
    before = snapshot(host)
    weaver = build(host, [perturbed(host, 1), perturbed(host, 2)])
    weaver.apply(host, zero_genome(2, 2))
    assert same(before, host.state_dict())
    weaver.apply(host, uniform_genome(2, 2, with_globals=True))
    assert not same(before, host.state_dict())
    weaver.apply(host, zero_genome(2, 2))
    assert same(before, host.state_dict())


def test_restore_is_exact_after_many_applies():
    host = tiny(0)
    before = snapshot(host)
    weaver = build(host, [perturbed(host, 1), perturbed(host, 2), perturbed(host, 3)])
    rng = np.random.default_rng(0)
    for _ in range(12):
        genome = Genome(3, 2, rng.random((3, 2, 2)), rng.random((3, 2)), 0.5 + 0.5 * rng.random(3))
        weaver.apply(host, genome)
    weaver.restore(host)
    assert same(before, host.state_dict())


def test_one_hot_equals_specialist():
    host = tiny(0)
    spec = perturbed(host, 1)
    other = perturbed(host, 2)
    weaver = build(host, [spec, other])
    weaver.apply(host, one_hot_genome(2, 2, 0, 1.0, with_globals=True))
    assert same(snapshot(spec), host.state_dict())
    weaver.restore(host)
    weaver.load_member(host, 1)
    assert same(snapshot(other), host.state_dict())


def test_float32_sum_keeps_what_bf16_steps_lose():
    host = tiny(0)
    name = "model.layers.0.mlp.up_proj.weight"
    one = 1.0
    ulp = 2.0 ** -7
    with torch.no_grad():
        host.get_parameter(name)[0, 0] = one
    specs = []
    for seed in range(3):
        spec = tiny(0)
        spec.load_state_dict(host.state_dict())
        with torch.no_grad():
            spec.get_parameter(name)[0, 0] = one + ulp
        specs.append(spec)
    weaver = build(host, specs)
    genome = Genome(3, 2)
    genome.weights[:] = 0.4
    weaver.apply(host, genome)
    mixed = float(host.get_parameter(name)[0, 0].detach())
    assert mixed == one + ulp
    naive = torch.tensor(one, dtype=torch.bfloat16)
    for _ in range(3):
        naive = naive + torch.tensor(0.4 * ulp, dtype=torch.bfloat16)
    assert float(naive) == one


def test_different_positional_base_touches_only_mlp():
    host = tiny(0)
    other_config = tiny(0, rope_theta=500.0).config
    assert weave.positional_differs(host.config, other_config)
    assert not weave.positional_differs(host.config, host.config)
    spec = perturbed(host, 1)
    member = member_from_state("m", "s", spec.state_dict(), host, mlp_only=True,
                               groups=("attn", "mlp", "embed", "norm"))
    weaver = Weaver(host, [member], n_buckets=2)
    before = snapshot(host)
    genome = Genome(1, 2)
    genome.weights[:] = 1.0
    genome.globals_[:] = 1.0
    weaver.apply(host, genome)
    after = host.state_dict()
    changed_mlp = 0
    for key in before:
        info = classify(key)
        if info is not None and info[0] == "mlp":
            if not torch.equal(before[key], after[key]):
                changed_mlp += 1
        else:
            assert torch.equal(before[key], after[key]), key
    assert changed_mlp > 0


def test_trim_keeps_fraction_and_rescales():
    torch.manual_seed(0)
    d = torch.randn(40, 50)
    out = trim_top(d, 0.25)
    assert int((out != 0).sum()) == 500
    kept = out != 0
    ratio = float(d.norm() / d[kept].norm())
    assert torch.allclose(out[kept], d[kept] * ratio, rtol=1e-5)
    assert abs(float(out.norm()) - float(d.norm())) < 1e-4
    threshold = d.abs().reshape(-1).sort(descending=True).values[499]
    assert float(d[kept].abs().min()) >= float(threshold) - 1e-9
    assert float(d[~kept].abs().max()) <= float(threshold)
    assert torch.equal(trim_top(d, 1.0), d)
    assert int((trim_top(d, 0.25, rescale=False) != 0).sum()) == 500


def test_density_applies_in_apply():
    host = tiny(0, torch.float32)
    spec = perturbed(host, 1)
    weaver = build(host, [spec])
    genome = Genome(1, 2)
    genome.weights[:] = 1.0
    genome.density[:] = 0.5
    weaver.apply(host, genome)
    name = "model.layers.1.mlp.up_proj.weight"
    d = spec.get_parameter(name).detach() - weaver.host[name]
    moved = host.get_parameter(name).detach() - weaver.host[name]
    assert abs(int((moved != 0).sum()) - d.numel() // 2) <= 2
    kept = moved != 0
    ratio = float(d.norm() / d[kept].norm())
    assert 1.0 < ratio < 1.6
    assert torch.allclose(moved[kept], ratio * d[kept], rtol=1e-4, atol=1e-6)


def synthetic_setup():
    host = tiny(0, torch.float32)
    specs = [perturbed(host, 1, 0.05), perturbed(host, 2, 0.05)]
    weaver = build(host, specs)
    target = Genome(2, 2)
    target.weights[0] = [[0.0, 0.8], [0.0, 0.8]]
    target.weights[1] = [[0.0, 0.2], [0.0, 0.2]]
    weaver.apply(host, target)
    goal = {k: v.clone() for k, v in host.state_dict().items()}
    weaver.restore(host)
    return host, weaver, goal, target


def make_score(goal):
    def score(model):
        total = 0.0
        for key, value in model.state_dict().items():
            total += float(((value - goal[key]) ** 2).sum())
        return {"skill": -total}
    return score


def run_planted(seed):
    host, weaver, goal, target = synthetic_setup()
    score = make_score(goal)
    host_scores = score(host)
    members = []
    for index in range(2):
        weaver.load_member(host, index)
        members.append(score(host))
        weaver.restore(host)
    # Normalize into [0, 1]-like numbers: the span between host and the best member.
    objective = Objective.from_reference(["skill"], [], host_scores, members, floor=1e-9)
    result = search(weaver, host, score, score, objective, objective, popsize=14, generations=30, seed=seed,
                    search_density=False, log=lambda *a: None)
    return result, target, host, weaver


def test_search_finds_planted_optimum_and_is_reproducible():
    result, target, host, weaver = run_planted(0)
    best = result["best_genome"]
    assert abs(best.weights[0, :, 1] - 0.8).max() < 0.15
    assert abs(best.weights[1, :, 1] - 0.2).max() < 0.15
    assert result["best"]["objective_a"] > 1.0
    labels = [e["label"] for e in result["log"][:4]]
    assert labels[0] == "zero"
    assert labels[1].startswith("one_hot:") and labels[2].startswith("one_hot:")
    assert labels[3] == "uniform"
    again, _, _, _ = run_planted(0)
    assert again["best"]["genome"] == result["best"]["genome"]
    other, _, _, _ = run_planted(1)
    assert other["log"][5]["genome"] != result["log"][5]["genome"]
    assert len(result["log"]) == 4 + 14 * 30


def test_search_restores_host():
    host, weaver, goal, target = synthetic_setup()
    before = snapshot(host)
    score = make_score(goal)
    objective = Objective(["skill"], [], {"skill": score(host)["skill"]}, {"skill": 0.0}, floor=1e-9)
    search(weaver, host, score, score, objective, objective, popsize=6, generations=2, seed=0,
           log=lambda *a: None)
    assert same(before, host.state_dict())


def test_guard_penalty():
    objective = Objective(["a"], ["g"], {"a": 0.5, "g": 0.6}, {"a": 0.7, "g": 0.6}, tol=0.01, guard_weight=10.0)
    value, _ = objective({"a": 0.7, "g": 0.6})
    assert abs(value - 1.0) < 1e-9
    value, detail = objective({"a": 0.7, "g": 0.55})
    assert abs(detail["guard_excess"]["g"] - 0.04) < 1e-9
    assert abs(value - (1.0 - 0.4)) < 1e-9


def test_random_control_keeps_norms_not_values():
    host = tiny(0, torch.float32)
    weaver = build(host, [perturbed(host, 1), perturbed(host, 2)])
    control = weaver.shuffled(5)
    again = weaver.shuffled(5)
    different = weaver.shuffled(6)
    name = "model.layers.2.mlp.down_proj.weight"
    real = weaver.delta(0, name)
    shuffled = control.delta(0, name)
    assert abs(float(real.norm()) - float(shuffled.norm())) < 1e-6
    assert not torch.equal(real, shuffled)
    assert torch.equal(shuffled, again.delta(0, name))
    assert not torch.equal(shuffled, different.delta(0, name))
    assert torch.equal(torch.sort(real.reshape(-1)).values, torch.sort(shuffled.reshape(-1)).values)
    before = snapshot(host)
    genome = uniform_genome(2, 2)
    control.apply(host, genome)
    assert not same(before, host.state_dict())
    weaver.restore(host)
    assert same(before, host.state_dict())


def test_saved_model_reloads_with_plain_transformers(tmp_path):
    host = tiny(0, torch.float32)
    weaver = build(host, [perturbed(host, 1), perturbed(host, 2)])
    genome = Genome(2, 2)
    genome.weights[0] = 0.7
    genome.weights[1] = 0.3
    weaver.apply(host, genome)
    ids = torch.randint(0, 100, (1, 12))
    with torch.no_grad():
        expected = host(ids).logits.clone()
    weaver.restore(host)
    path = str(tmp_path / "out")
    record = weave.save(weaver, host, genome, path, host_id="tiny", search_summary={"note": "test"})
    assert os.path.exists(os.path.join(path, "weave.json"))
    info = json.load(open(os.path.join(path, "weave.json")))
    assert info["host"] == "tiny" and len(info["members"]) == 2
    assert Genome.from_dict(info["genome"]).weights.shape == (2, 2, 2)
    loaded = transformers.AutoModelForCausalLM.from_pretrained(path, dtype=torch.float32).eval()
    with torch.no_grad():
        got = loaded(ids).logits
    assert torch.allclose(expected, got, atol=1e-5)
    assert sum(p.numel() for p in loaded.parameters()) == sum(p.numel() for p in host.parameters())
    # The saved directory holds no wsp reference and the host is back to itself.
    config = json.load(open(os.path.join(path, "config.json")))
    assert "wsp" not in json.dumps(config)


def test_reference_recipes():
    host = tiny(0)
    weaver = build(host, [perturbed(host, 1), perturbed(host, 2), perturbed(host, 3)])
    uni = uniform_genome(3, 2)
    assert np.allclose(uni.weights, 0.25)
    ta = task_arithmetic_genome(3, 2, 0.5)
    assert np.allclose(ta.weights, 0.5)
    before = snapshot(host)
    weaver.apply_ties(host, keep=0.5, scale=1.0)
    changed = not same(before, host.state_dict())
    assert changed
    after = host.state_dict()
    for key in before:
        info = classify(key)
        if info[0] in ("embed", "norm"):
            assert torch.equal(before[key], after[key])
    weaver.restore(host)
    assert same(before, host.state_dict())


def test_ties_matches_hand_computation():
    host = tiny(0, torch.float32)
    weaver = build(host, [perturbed(host, 1), perturbed(host, 2)])
    name = "model.layers.0.mlp.gate_proj.weight"
    d1 = weaver.delta(0, name)
    d2 = weaver.delta(1, name)
    t1 = trim_top(d1, 1.0, rescale=False)
    t2 = trim_top(d2, 1.0, rescale=False)
    stacked = torch.stack([t1, t2])
    sign = torch.sign(stacked.sum(0))
    agree = (torch.sign(stacked) == sign) & (stacked != 0)
    expected = weaver.host[name] + (stacked * agree).sum(0) / agree.sum(0).clamp(min=1)
    weaver.apply_ties(host, keep=1.0, scale=1.0)
    assert torch.allclose(host.get_parameter(name).detach(), expected, atol=1e-6)


def test_layer_prior_follows_scores():
    scores = {"a": [0, 0, 0, 0, 5, 5, 5, 5], "b": [5, 5, 5, 5, 0, 0, 0, 0]}
    genome = weave.layer_prior_genome(scores, ["a", "b", "other"], 2)
    assert genome.weights[0, 1, 0] > genome.weights[0, 0, 0]
    assert genome.weights[1, 0, 0] > genome.weights[1, 1, 0]
    assert np.allclose(genome.weights[2], genome.weights[2, 0, 0])


def test_genome_vector_roundtrip_and_cma_bounds():
    rng = np.random.default_rng(0)
    genome = Genome(2, 3, rng.random((2, 3, 2)), rng.random((2, 2)), rng.random(2))
    again = Genome.from_vector(genome.to_vector(), 2, 3)
    assert np.array_equal(again.weights, genome.weights)
    assert np.array_equal(again.density, genome.density)
    lower, upper = weave.genome_bounds(2, 3)
    cma = SepCMAES(np.full(len(lower), 0.5), 0.3, lower, upper, 8, 0)
    genes = cma.ask()
    assert genes.min() >= lower.min() - 1e-12 and genes.max() <= upper.max() + 1e-12


def test_gate_flags_far_member():
    host = tiny(0, torch.float32)
    near = perturbed(host, 1, 0.001)
    far = tiny(7, torch.float32)
    near_member = member_from_state("near", "s", near.state_dict(), host)
    far_member = member_from_state("far", "s", far.state_dict(), host)
    assert weave.gate_member(near_member, host)["passed"]
    assert not weave.gate_member(far_member, host)["passed"]


def test_cast_survival_reports_loss():
    host = tiny(0)
    weaver = build(host, [perturbed(host, 1, 0.0005)])
    genome = Genome(1, 2)
    genome.weights[:] = 1.0
    genome.globals_[:] = 1.0
    full = weaver.cast_survival(genome)
    genome.weights[:] = 0.02
    genome.globals_[:] = 0.02
    small = weaver.cast_survival(genome)
    assert full == 1.0
    assert small is not None and small < full


def test_trim_with_ties_keeps_exactly_the_count_and_the_norm():
    torch.manual_seed(0)
    # 60 percent exact zeros, the rest on a coarse grid of one-ulp steps.
    base = torch.randint(1, 4, (50, 40)).float() * torch.where(torch.rand(50, 40) < 0.5, -1.0, 1.0)
    d = base * (torch.rand(50, 40) < 0.4).float()
    assert float((d == 0).float().mean()) > 0.5
    out = trim_top(d, 0.5)
    assert torch.equal(out, d)
    # Dense ties: all magnitudes in {1, 2, 3}.
    dense = torch.randint(1, 4, (50, 40)).float() * torch.where(torch.rand(50, 40) < 0.5, -1.0, 1.0)
    out = trim_top(dense, 0.5, rescale=False)
    assert int((out != 0).sum()) == 1000
    assert float(dense.abs()[out != 0].min()) >= float(dense.abs()[out == 0].max())
    scaled = trim_top(dense, 0.5)
    assert int((scaled != 0).sum()) == 1000
    assert abs(float(scaled.norm()) - float(dense.norm())) < 1e-3
    assert 0.7 < float(scaled.abs().sum() / dense.abs().sum()) < 1.3
    # The cached cutoff gives the same answer.
    keep = 1000
    cutoff = weave.rank_cutoff(dense, keep)
    assert torch.equal(trim_top(dense, 0.5, cutoff=cutoff), scaled)


def test_trim_gaussian_is_not_amplified():
    torch.manual_seed(0)
    d = torch.randn(200, 200)
    out = trim_top(d, 0.5)
    assert abs(float(out.norm() / d.norm()) - 1.0) < 1e-4
    assert float(out.abs().sum() / d.abs().sum()) < 1.35


def test_density_gene_does_not_scale_a_mostly_zero_difference():
    host = tiny(0, torch.float32)
    spec = tiny(0, torch.float32)
    spec.load_state_dict(host.state_dict())
    name = "model.layers.1.mlp.up_proj.weight"
    with torch.no_grad():
        param = spec.get_parameter(name)
        generator = torch.Generator().manual_seed(3)
        pick = torch.rand(param.shape, generator=generator) < 0.3
        param.add_(pick.float() * 2.0 ** -6)
    weaver = build(host, [spec])
    genome = Genome(1, 2)
    genome.weights[:] = 1.0
    genome.density[:] = 0.5
    weaver.apply(host, genome)
    moved = host.get_parameter(name).detach() - weaver.host[name]
    want = spec.get_parameter(name).detach() - weaver.host[name]
    assert torch.allclose(moved, want, atol=1e-7)


def test_cma_tell_never_sees_injected_seeds(monkeypatch):
    seen = {"asked": [], "told": []}

    class Recording(_RealCMA):
        def ask(self):
            out = super().ask()
            seen["asked"].append(out.copy())
            return out

        def tell(self, genes, fitness):
            seen["told"].append(np.array(genes))
            super().tell(genes, fitness)

    monkeypatch.setattr(weave, "SepCMAES", Recording)
    host, weaver, goal, target = synthetic_setup()
    score = make_score(goal)
    objective = Objective(["skill"], [], {"skill": score(host)["skill"]}, {"skill": 0.0}, floor=1e-9)
    search(weaver, host, score, score, objective, objective, popsize=6, generations=2, seed=0,
           search_density=False, log=lambda *a: None)
    assert len(seen["told"]) == 2
    active = np.where(weave.active_mask(2, 2, False, False))[0]
    for generation in range(2):
        assert seen["told"][generation].shape == (6, len(active))
        assert np.allclose(seen["told"][generation], seen["asked"][generation])


def test_cma_move_mean():
    cma = SepCMAES(np.full(4, 0.5), 0.2, np.zeros(4), np.ones(4), 6, 0)
    before = (cma.C.copy(), cma.sigma, cma.ps.copy())
    cma.move_mean(np.array([1.0, 0.0, 0.25, 2.0]))
    assert np.allclose(cma.mean, [1.0, 0.0, 0.25, 1.0])
    assert np.array_equal(cma.C, before[0]) and cma.sigma == before[1] and np.array_equal(cma.ps, before[2])


def test_objective_floor_and_clipping_flag():
    objective = Objective(["a", "b"], [], {"a": 0.5, "b": 0.5}, {"a": 0.51, "b": 0.7})
    assert objective.floor == 0.05
    value, detail = objective({"a": 0.51, "b": 0.6})
    assert abs(detail["gains"]["a"] - 0.2) < 1e-9
    assert detail["denom_clipped"] == {"a": True, "b": False}


def test_tied_embeddings_are_mixed_once():
    host = tiny(0, torch.float32, tie=True)
    assert host.get_parameter("model.embed_tokens.weight") is host.lm_head.weight
    spec = perturbed(host, 1)
    member = member_from_state("m", "s", spec.state_dict(), host, groups=("attn", "mlp", "embed", "norm"))
    assert "lm_head.weight" not in member.tensors
    weaver = Weaver(host, [member], n_buckets=2)
    assert "lm_head.weight" not in weaver.names
    genome = Genome(1, 2)
    genome.weights[:] = 1.0
    genome.globals_[:] = 1.0
    weaver.apply(host, genome)
    embed = host.get_parameter("model.embed_tokens.weight").detach()
    want = spec.get_parameter("model.embed_tokens.weight").detach()
    assert torch.allclose(embed, want, atol=1e-6)
    assert torch.equal(host.lm_head.weight.detach(), embed)
    assert host.lm_head.weight.data_ptr() == host.model.embed_tokens.weight.data_ptr()
    weaver.restore(host)
    assert torch.equal(host.get_parameter("model.embed_tokens.weight").detach(), weaver.host["model.embed_tokens.weight"])


def test_member_with_separate_reference_adds_the_reference_difference():
    host = tiny(0, torch.float32)
    base = perturbed(host, 5, 0.03)
    spec = perturbed(base, 6, 0.02)
    ref = {k: v.detach().clone() for k, v in base.state_dict().items()}
    member = member_from_state("m", "s", spec.state_dict(), host, ref=ref, ref_name="base",
                               groups=("attn", "mlp", "norm"))
    weaver = Weaver(host, [member], n_buckets=2)
    name = "model.layers.2.mlp.down_proj.weight"
    assert weaver.reference_tensor(0, name) is ref[name]
    before = snapshot(host)
    weaver.apply(host, one_hot_genome(1, 2, 0, 1.0))
    got = host.get_parameter(name).detach()
    want = before[name] + (spec.get_parameter(name).detach() - ref[name])
    assert torch.allclose(got, want, atol=1e-6)
    assert not torch.allclose(got, spec.get_parameter(name).detach(), atol=1e-4)
    assert torch.equal(host.get_parameter("model.embed_tokens.weight").detach(), before["model.embed_tokens.weight"])
    weaver.restore(host)
    assert same(before, host.state_dict())


def test_bf16_save_and_reload_keeps_dtype_and_values(tmp_path):
    host = tiny(0, torch.bfloat16)
    weaver = build(host, [perturbed(host, 1), perturbed(host, 2)])
    genome = Genome(2, 2)
    genome.weights[0] = 0.6
    genome.weights[1] = 0.4
    weaver.apply(host, genome)
    mixed = snapshot(host)
    weaver.restore(host)
    path = str(tmp_path / "out16")
    weave.save(weaver, host, genome, path, host_id="tiny")
    from safetensors import safe_open
    count = 0
    for file in os.listdir(path):
        if file.endswith(".safetensors"):
            with safe_open(os.path.join(path, file), framework="pt") as handle:
                for key in handle.keys():
                    tensor = handle.get_tensor(key)
                    assert tensor.dtype == torch.bfloat16, key
                    assert torch.equal(tensor, mixed[key]), key
                    count += 1
    assert count == len(mixed)
    loaded = transformers.AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16)
    assert all(p.dtype == torch.bfloat16 for p in loaded.parameters())
    assert same(mixed, loaded.state_dict())


# ---------------------------------------------------------------- probe objective


class ProbeTokenizer:
    """Whitespace tokenizer into ids 1..30, enough for the tiny models (vocab 100)."""

    pad_token = "<pad>"
    eos_token = "<pad>"
    pad_token_id = 0
    padding_side = "right"
    chat_template = None

    def __call__(self, texts, return_tensors="pt", padding=True, truncation=True, max_length=64,
                 add_special_tokens=True):
        rows = []
        for text in texts:
            rows.append([1 + (sum(ord(c) for c in word) % 30) for word in text.split()][:max_length])
        width = max(len(row) for row in rows)
        input_ids = torch.zeros(len(rows), width, dtype=torch.long)
        mask = torch.zeros(len(rows), width, dtype=torch.long)
        for index, row in enumerate(rows):
            input_ids[index, :len(row)] = torch.tensor(row)
            mask[index, :len(row)] = 1
        return {"input_ids": input_ids, "attention_mask": mask}


PROBE_TEXTS = ["alpha beta gamma delta epsilon zeta eta theta", "one two three four five six seven",
               "red green blue yellow orange purple pink brown black", "north south east west up down left right",
               "cat dog bird fish horse cow pig sheep goat", "sun moon star planet comet galaxy nebula quasar"]


def probe_setup(scale=0.08, k=64):
    host = tiny(0, torch.float32)
    member = perturbed(host, 3, scale=scale)
    tokenizer = ProbeTokenizer()
    cache = weave.build_teacher_cache(member, tokenizer, PROBE_TEXTS, top_k=k, max_length=32, batch_size=4)
    general = weave.build_teacher_cache(host, tokenizer, PROBE_TEXTS[:3], top_k=k, max_length=32, batch_size=4)
    objective = weave.ProbeObjective(["m"], {"m": cache}, general, batch_size=4)
    return host, member, tokenizer, cache, general, objective


def test_probe_host_scores_transfer_zero_and_member_scores_one():
    host, member, tokenizer, cache, general, objective = probe_setup()
    info = objective.calibrate(host)
    assert info["baseline_kl"]["m"] > 0.01
    scores = objective.scores(host)
    assert abs(scores["transfer"]["m"]) < 1e-9
    assert scores["drift"] < 0.01
    own = objective.scores(member)
    assert own["transfer"]["m"] > 0.95
    assert own["transfer"]["m"] <= 1.01
    value, detail = objective(own)
    assert detail["mean_transfer"] > 0.95
    assert abs(value - (detail["mean_transfer"] - detail["penalty"])) < 1e-12


def test_probe_top_k_plus_tail_is_close_to_full_kl():
    from wsp import probe

    host, member, tokenizer, cache, general, objective = probe_setup()
    full = probe.logit_kl(member, host, tokenizer, PROBE_TEXTS, max_length=32, batch_size=4)["kl"]
    approx = weave.kl_to_cache(host, cache, batch_size=4)["kl"]
    assert approx <= full + 1e-3
    assert abs(approx - full) <= 0.05 * full + 1e-3
    small = weave.build_teacher_cache(member, tokenizer, PROBE_TEXTS, top_k=8, max_length=32, batch_size=4)
    coarse = weave.kl_to_cache(host, small, batch_size=4)["kl"]
    assert coarse <= approx + 1e-3
    assert coarse > 0.0


def test_probe_drift_penalty_applies_above_tolerance():
    host, member, tokenizer, cache, general, objective = probe_setup()
    low, detail_low = objective({"transfer": {"m": 0.5}, "drift": 0.03})
    assert abs(low - 0.5) < 1e-12
    assert detail_low["penalty"] == 0.0
    high, detail_high = objective({"transfer": {"m": 0.5}, "drift": 0.15})
    assert abs(high - (0.5 - 10.0 * 0.10)) < 1e-9
    assert abs(detail_high["drift_excess"] - 0.10) < 1e-12
    none_value, none_detail = objective({"transfer": {"m": None}, "drift": 0.0})
    assert none_value == 0.0 and none_detail["mean_transfer"] == 0.0


def test_probe_drift_grows_when_the_model_moves_away_from_host():
    host, member, tokenizer, cache, general, objective = probe_setup(scale=0.3)
    objective.calibrate(host)
    far = objective.scores(member)
    assert far["drift"] > objective.drift_tolerance
    value, detail = objective(far)
    assert detail["penalty"] > 0.0


def test_probe_cache_round_trips(tmp_path):
    host, member, tokenizer, cache, general, objective = probe_setup()
    path = weave.cache_path(str(tmp_path), "member/one", "abc123", cache["texts_hash"])
    weave.save_cache(cache, path)
    again = weave.load_cache(path)
    for key in ("lengths", "ids", "top_ids", "top_logp", "tail"):
        assert again[key].dtype == cache[key].dtype
        assert np.array_equal(again[key], cache[key])
    assert again["top_ids"].dtype == np.int32 and again["top_logp"].dtype == np.float16
    assert again["top_k"] == cache["top_k"] and again["vocab"] == cache["vocab"]
    assert again["texts_hash"] == cache["texts_hash"]
    a = weave.kl_to_cache(host, cache, batch_size=4)["kl"]
    b = weave.kl_to_cache(host, again, batch_size=4)["kl"]
    assert a == b
    assert weave.text_hash(PROBE_TEXTS) != weave.text_hash(PROBE_TEXTS[:-1])
    assert weave.text_hash(PROBE_TEXTS) == weave.text_hash(list(PROBE_TEXTS))


def test_probe_objective_drives_search_and_restores_host():
    host = tiny(0, torch.float32)
    spec = perturbed(host, 3, scale=0.08)
    host_state = host.state_dict()
    with torch.no_grad():
        for name, param in spec.named_parameters():
            if classify(name)[0] not in ("attn", "mlp"):
                param.copy_(host_state[name])
    tokenizer = ProbeTokenizer()
    cache = weave.build_teacher_cache(spec, tokenizer, PROBE_TEXTS, top_k=64, max_length=32, batch_size=4)
    general = weave.build_teacher_cache(host, tokenizer, PROBE_TEXTS[:3], top_k=64, max_length=32, batch_size=4)
    members = [member_from_state("m0", "skill0", spec.state_dict(), host, groups=("attn", "mlp"))]
    weaver = Weaver(host, members, n_buckets=2)
    objective = weave.ProbeObjective(["m0"], {"m0": cache}, general, batch_size=4)
    objective.calibrate(host)
    before = snapshot(host)
    result = search(weaver, host, objective.scores, objective.scores, objective, objective, popsize=4,
                    generations=2, seed=0, log=lambda *a: None)
    assert same(before, snapshot(host))
    zero = [e for e in result["log"] if e["label"] == "zero"][0]
    assert abs(zero["objective_a"]) < 0.02
    assert result["best"]["objective_a"] > 0.3
