"""Tests for wsp.delta on tiny random Qwen2 models (CPU, no downloads)."""

import copy

import pytest
import torch
import transformers

from wsp.delta import (BASE, SkillDeltaNetwork, merge, mine_delta, resolve_rank, summarize_stats)

VOCAB = 100
LINEARS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def tiny_config(theta=10000.0, tie=False):
    return transformers.Qwen2Config(
        vocab_size=VOCAB, hidden_size=32, intermediate_size=64, num_hidden_layers=3, num_attention_heads=4,
        num_key_value_heads=2, max_position_embeddings=128, rope_theta=theta, tie_word_embeddings=tie,
        use_cache=True, pad_token_id=0, bos_token_id=1, eos_token_id=2)


def make_host(seed=0, tie=False):
    torch.manual_seed(seed)
    model = transformers.Qwen2ForCausalLM(tiny_config(tie=tie)).float().eval()
    # Make the norms non-trivial so a norm delta matters.
    for name, param in model.named_parameters():
        if param.ndim == 1 and "norm" in name:
            param.data.add_(0.1 * torch.randn_like(param))
    return model


def make_specialist(host, seed, rank=None, theta=10000.0, scale=0.2, vectors=True):
    """host + a planted delta (rank-limited if rank is set) on every shared linear, plus vector changes."""
    gen = torch.Generator().manual_seed(seed)
    spec = transformers.Qwen2ForCausalLM(tiny_config(theta=theta, tie=host.config.tie_word_embeddings)).float().eval()
    spec.load_state_dict(host.state_dict())
    with torch.no_grad():
        for name, param in spec.named_parameters():
            is_linear = param.ndim == 2 and any(name.endswith(k + ".weight") for k in LINEARS)
            if is_linear:
                out_dim, in_dim = param.shape
                if rank is None:
                    change = torch.randn(out_dim, in_dim, generator=gen)
                    change = change * scale / change.norm() * param.norm()
                else:
                    change = torch.randn(out_dim, rank, generator=gen) @ torch.randn(rank, in_dim, generator=gen)
                    change = change * scale / change.norm() * param.norm()
                param.add_(change)
            elif param.ndim == 1 and vectors:
                param.add_(0.05 * torch.randn(param.shape, generator=gen))
    return spec


def ids_batch(seed=0, rows=3, length=16):
    gen = torch.Generator().manual_seed(seed)
    return torch.randint(3, VOCAB, (rows, length), generator=gen)


def logits_of(model, ids):
    with torch.no_grad():
        return model(input_ids=ids).logits


def test_full_rank_reproduces_specialist_logits_and_rope():
    host = make_host()
    spec = make_specialist(host, 1, theta=1e6)
    ids = ids_batch()
    want = logits_of(spec, ids)
    host_logits = logits_of(host, ids)
    delta = mine_delta(host, spec, "full", name="spec")
    assert delta.attrs["rope_theta"] == 1e6
    assert "rope_theta" in delta.attrs["differences"]
    net = SkillDeltaNetwork(copy.deepcopy(host), [delta])
    net.set_expert("spec")
    assert torch.allclose(logits_of(net.model, ids), want, atol=1e-5, rtol=1e-5)
    assert float((host_logits - want).abs().max()) > 1e-3
    net.set_expert(BASE)
    assert torch.allclose(logits_of(net.model, ids), host_logits, atol=1e-6, rtol=1e-6)


def test_truncated_error_matches_svd_tail():
    host = make_host()
    spec = make_specialist(host, 2)
    delta = mine_delta(host, spec, 8, name="spec")
    host_params = dict(host.named_parameters())
    spec_params = dict(spec.named_parameters())
    checked = 0
    for module in delta.factors:
        D = (spec_params[module + ".weight"] - host_params[module + ".weight"]).detach().double()
        s = torch.linalg.svdvals(D)
        r = delta.stats[module]["rank"]
        tail = float((s[r:] ** 2).sum().sqrt() / (s ** 2).sum().sqrt())
        assert delta.stats[module]["rel_err"] == pytest.approx(tail, abs=1e-5)
        parts = delta.factors[module]
        explicit = float((D - parts["B"].double() @ parts["A"].double()).norm() / D.norm())
        assert explicit == pytest.approx(tail, abs=1e-5)
        assert parts["A"].shape == (r, D.shape[1])
        assert parts["B"].shape == (D.shape[0], r)
        energy = float((s[:r] ** 2).sum() / (s ** 2).sum())
        assert delta.stats[module]["energy_kept"] == pytest.approx(energy, abs=1e-6)
        checked += 1
    assert checked == 3 * len(LINEARS)
    summary = summarize_stats(delta.stats)
    assert summary["all"]["n"] == checked
    assert "rank90_mean" in summary["q"]


def test_planted_rank4_delta_is_recovered_exactly():
    host = make_host()
    spec = make_specialist(host, 3, rank=4)
    delta = mine_delta(host, spec, 4, name="spec")
    for module in delta.stats:
        assert delta.stats[module]["rel_err"] < 1e-5
        assert delta.stats[module]["rank99"] <= 4
    net = SkillDeltaNetwork(copy.deepcopy(host), [delta])
    net.set_expert("spec")
    ids = ids_batch()
    assert torch.allclose(logits_of(net.model, ids), logits_of(spec, ids), atol=1e-4, rtol=1e-4)
    # Rank 3 cannot recover it.
    low = mine_delta(host, spec, 3, name="low")
    assert max(low.stats[m]["rel_err"] for m in low.stats) > 1e-3


def test_truncate_matches_direct_mining():
    host = make_host()
    spec = make_specialist(host, 4)
    big = mine_delta(host, spec, 16, name="spec")
    small = big.truncate(4)
    direct = mine_delta(host, spec, 4, name="spec")
    for module in direct.factors:
        got = small.factors[module]["B"] @ small.factors[module]["A"]
        want = direct.factors[module]["B"] @ direct.factors[module]["A"]
        assert torch.allclose(got, want, atol=1e-5)
        assert small.stats[module]["rel_err"] == pytest.approx(direct.stats[module]["rel_err"], abs=1e-5)


def test_rank_specs():
    sv = torch.tensor([3.0, 2.0, 1.0, 0.1])
    assert resolve_rank(2, "q", (8, 8)) == 2
    assert resolve_rank(100, "q", (8, 6)) == 6
    assert resolve_rank("full", "q", (8, 8)) == "full"
    assert resolve_rank({"q": 3, "default": 0}, "k", (8, 8)) == 0
    assert resolve_rank({"q": 3, "default": 0}, "q", (8, 8)) == 3
    assert resolve_rank(0.9, "q", (8, 8), sv) == 2
    host = make_host()
    spec = make_specialist(host, 5)
    delta = mine_delta(host, spec, {"q": 2, "down": 5}, name="spec")
    assert set(delta.stats[m]["kind"] for m in delta.factors) == {"q", "down"}
    energy = mine_delta(host, spec, 0.9, name="spec")
    for module in energy.stats:
        assert energy.stats[module]["energy_kept"] >= 0.9 - 1e-6


def test_random_control_has_same_norms_and_different_outputs():
    host = make_host()
    spec = make_specialist(host, 6, theta=1e6)
    real = mine_delta(host, spec, 8, name="spec")
    fake = mine_delta(host, spec, 8, name="ctrl", control="random", seed=3)
    fake2 = mine_delta(host, spec, 8, name="ctrl", control="random", seed=3)
    assert len(fake.vectors) == 0
    assert fake.attrs["rope_theta"] is None
    for module in real.factors:
        a = real.factors[module]
        b = fake.factors[module]
        assert a["A"].shape == b["A"].shape and a["B"].shape == b["B"].shape
        norm_real = float((a["B"] @ a["A"]).norm())
        norm_fake = float((b["B"] @ b["A"]).norm())
        assert norm_fake == pytest.approx(norm_real, rel=1e-4)
        assert not torch.allclose(b["A"], a["A"])
        assert torch.equal(b["A"], fake2.factors[module]["A"])
    ids = ids_batch()
    net = SkillDeltaNetwork(copy.deepcopy(host), [real, fake])
    net.set_expert("spec")
    out_real = logits_of(net.model, ids)
    net.set_expert("ctrl")
    out_fake = logits_of(net.model, ids)
    net.set_expert(BASE)
    out_base = logits_of(net.model, ids)
    assert not torch.allclose(out_fake, out_real, atol=1e-3)
    assert not torch.allclose(out_fake, out_base, atol=1e-4)


def test_random_control_from_a_mined_delta():
    from wsp.delta import random_control

    host = make_host()
    spec = make_specialist(host, 6, theta=1e6)
    big = mine_delta(host, spec, 16, name="spec")
    ctrl = random_control(big, rank=4, seed=1)
    cut = big.truncate(4)
    for module in cut.factors:
        real = cut.factors[module]
        fake = ctrl.factors[module]
        assert fake["A"].shape == real["A"].shape
        assert float((fake["B"] @ fake["A"]).norm()) == pytest.approx(float((real["B"] @ real["A"]).norm()), rel=1e-4)
    assert len(ctrl.vectors) == 0 and ctrl.attrs["rope_theta"] is None
    dense = random_control(mine_delta(host, spec, "full", name="spec"), seed=2)
    assert all("D" in dense.factors[m] for m in dense.factors)


def test_base_expert_equals_host():
    host = make_host()
    spec = make_specialist(host, 7, theta=1e6)
    ids = ids_batch()
    want = logits_of(host, ids)
    net = SkillDeltaNetwork(copy.deepcopy(host), [mine_delta(host, spec, 8, name="spec")])
    assert torch.equal(logits_of(net.model, ids), want)
    net.set_expert("spec")
    net.set_expert(BASE)
    assert torch.allclose(logits_of(net.model, ids), want, atol=1e-6)


def domain_texts(count, low, high, length=24, seed=0):
    gen = torch.Generator().manual_seed(seed)
    texts = []
    for _ in range(count):
        texts.append(torch.randint(low, high, (length,), generator=gen).tolist())
    return texts


def router_network():
    host = make_host(seed=11)
    a = mine_delta(host, make_specialist(host, 21), 8, name="a")
    b = mine_delta(host, make_specialist(host, 22), 8, name="b")
    net = SkillDeltaNetwork(copy.deepcopy(host), [a, b])
    texts = {"a": domain_texts(60, 3, 13, seed=1), "b": domain_texts(60, 13, 23, seed=2),
             "base": domain_texts(60, 23, 100, seed=3)}
    return net, texts


def test_router_sends_domains_and_keeps_unsure_on_base():
    net, texts = router_network()
    report = net.fit_router(texts, layer=2, ridge=0.01, base_margin=0.3, holdout=0.2)
    assert report["holdout_accuracy"] > 0.9
    assert net.route(domain_texts(1, 3, 13, seed=50)[0]) == "a"
    assert net.route(domain_texts(1, 13, 23, seed=51)[0]) == "b"
    assert net.route(domain_texts(1, 23, 100, seed=52)[0]) == "base"
    mixed = domain_texts(1, 3, 13, length=12, seed=53)[0] + domain_texts(1, 13, 23, length=12, seed=54)[0]
    net.router.base_margin = 0.0
    loose = net.route(mixed)
    net.router.base_margin = 5.0
    assert net.route(mixed) == BASE
    assert net.route(domain_texts(1, 3, 13, seed=50)[0]) == BASE
    assert loose in ["a", "b", BASE]
    assert net.active == BASE


def test_mixed_expert_batch_equals_per_row_runs():
    net, _ = router_network()
    ids = ids_batch(seed=5, rows=4, length=12)
    experts = ["a", BASE, "b", "a"]
    mixed = net.forward(ids, experts=experts)
    for row in range(4):
        net.set_expert(experts[row])
        single = logits_of(net.model, ids[row:row + 1])[0]
        assert torch.allclose(mixed[row], single, atol=1e-5, rtol=1e-5)
    assert net.active == experts[3]
    net.set_expert(BASE)
    out = net.generate(ids, experts=experts, max_new_tokens=5, do_sample=False)
    for row in range(4):
        net.set_expert(experts[row])
        single = net.model.generate(input_ids=ids[row:row + 1], max_new_tokens=5, do_sample=False)
        assert torch.equal(out[row, :single.shape[1]], single[0])
    net.set_expert(BASE)


def test_generate_routes_once_per_sequence():
    net, texts = router_network()
    net.fit_router(texts, layer=2, ridge=0.01, base_margin=0.0)
    rows = [domain_texts(1, 3, 13, length=10, seed=60)[0], domain_texts(1, 13, 23, length=10, seed=61)[0]]
    ids = torch.tensor(rows)
    out = net.generate(ids, max_new_tokens=4, do_sample=False)
    assert out.shape[0] == 2
    expected = net.route_many(rows)
    assert expected == ["a", "b"]
    for row in range(2):
        net.set_expert(expected[row])
        single = net.model.generate(input_ids=ids[row:row + 1], max_new_tokens=4, do_sample=False)
        assert torch.equal(out[row, :single.shape[1]], single[0])
    net.set_expert(BASE)


def test_kv_cache_generation_equals_uncached():
    net, _ = router_network()
    ids = ids_batch(seed=8, rows=2, length=10)
    for name in ["a", "b", BASE]:
        net.set_expert(name)
        cached = net.model.generate(input_ids=ids, max_new_tokens=8, do_sample=False, use_cache=True)
        plain = net.model.generate(input_ids=ids, max_new_tokens=8, do_sample=False, use_cache=False)
        assert torch.equal(cached, plain)
    net.set_expert(BASE)


def test_save_and_load_give_equal_logits(tmp_path):
    host = make_host(seed=12)
    spec = make_specialist(host, 31, theta=1e6)
    spec2 = make_specialist(host, 32)
    net = SkillDeltaNetwork(copy.deepcopy(host), [mine_delta(host, spec, 8, name="m"),
                                                   mine_delta(host, spec2, "full", name="c")], host_id="tiny")
    texts = {"m": domain_texts(40, 3, 13, seed=1), "c": domain_texts(40, 13, 23, seed=2),
             "base": domain_texts(40, 23, 100, seed=3)}
    net.fit_router(texts, layer=2, ridge=0.1, base_margin=0.2)
    net.save(str(tmp_path / "bank"))
    loaded = SkillDeltaNetwork.load(str(tmp_path / "bank"), host=copy.deepcopy(host))
    ids = ids_batch(seed=9)
    for name in ["m", "c", BASE]:
        net.set_expert(name)
        loaded.set_expert(name)
        assert torch.equal(logits_of(net.model, ids), logits_of(loaded.model, ids))
    probe = domain_texts(5, 3, 13, seed=70)
    assert net.route_many(probe) == loaded.route_many(probe)
    assert loaded.router.base_margin == 0.2
    assert loaded.count_params() == net.count_params()


def test_param_counts_match_direct_counts():
    host = make_host(seed=13)
    spec = make_specialist(host, 41)
    net = SkillDeltaNetwork(copy.deepcopy(host), [mine_delta(host, spec, 4, name="x"),
                                                   mine_delta(host, spec, "full", name="y")])
    net.fit_router({"x": domain_texts(20, 3, 13, seed=1), "y": domain_texts(20, 13, 23, seed=2),
                    "base": domain_texts(20, 23, 100, seed=3)}, layer=2)
    seen = set()
    host_total = 0
    for param in net.model.parameters():
        if id(param) not in seen:
            seen.add(id(param))
            host_total += param.numel()
    direct = {}
    for name in ["x", "y"]:
        total = 0
        factor_total = 0
        delta = net.deltas[name]
        for module in delta.factors:
            for tensor in delta.factors[module].values():
                total += tensor.numel()
                factor_total += tensor.numel()
        for tensor in delta.vectors.values():
            total += tensor.numel()
        direct[name] = (total, factor_total)
    router = net.router.W.numel() + net.router.mean.numel() + net.router.std.numel()
    counts = net.count_params()
    assert counts["host"] == host_total
    assert counts["stored"] == host_total + direct["x"][0] + direct["y"][0] + router
    assert counts["deltas"] == {"x": direct["x"][0], "y": direct["y"][0]}
    embed = net.model.get_input_embeddings().weight.numel()
    assert counts["per_token"] == host_total - embed
    assert net.count_params("x")["per_token"] == host_total - embed + direct["x"][1]
    net.set_expert("y")
    assert net.count_params()["per_token"] == host_total - embed + direct["y"][1]
    # A rank 4 delta is smaller than a dense one.
    assert direct["x"][1] < direct["y"][1]


def test_rope_theta_is_applied_per_expert():
    host = make_host(seed=14)
    spec = transformers.Qwen2ForCausalLM(tiny_config(theta=1e6)).float().eval()
    spec.load_state_dict(host.state_dict())
    delta = mine_delta(host, spec, 4, name="rope")
    assert len(delta.factors) == 0
    assert delta.attrs["rope_theta"] == 1e6
    net = SkillDeltaNetwork(copy.deepcopy(host), [delta])
    ids = ids_batch(seed=15, rows=2, length=40)
    want_host = logits_of(host, ids)
    want_spec = logits_of(spec, ids)
    assert float((want_host - want_spec).abs().max()) > 1e-4
    net.set_expert("rope")
    got = logits_of(net.model, ids)
    assert float((got - want_spec).abs().max()) < 1e-6
    assert float((got - want_host).abs().max()) > 1e-4
    assert net.model.config.rope_theta == 1e6
    net.set_expert(BASE)
    assert torch.allclose(logits_of(net.model, ids), want_host, atol=1e-6)
    assert net.model.config.rope_theta == 10000.0


def test_merge_full_equals_specialist_and_plain_model():
    host = make_host(seed=16)
    spec = make_specialist(host, 51, theta=1e6)
    net = SkillDeltaNetwork(copy.deepcopy(host), [mine_delta(host, spec, "full", name="spec"),
                                                   mine_delta(host, spec, 4, name="low")])
    ids = ids_batch(seed=17)
    merged = merge(net, "spec")
    assert merged.config.rope_theta == 1e6
    assert torch.allclose(logits_of(merged, ids), logits_of(spec, ids), atol=1e-5, rtol=1e-5)
    spec_params = dict(spec.named_parameters())
    for name, param in merged.named_parameters():
        assert torch.allclose(param, spec_params[name], atol=1e-6), name
    for module in merged.modules():
        assert len(module._forward_hooks) == 0
    # The network is untouched, and merging "base" gives the host.
    assert net.active == BASE
    assert torch.equal(logits_of(merge(net, BASE), ids), logits_of(host, ids))
    # A low-rank merge equals the on-the-fly network at the same expert.
    low = merge(net, "low")
    net.set_expert("low")
    assert torch.allclose(logits_of(low, ids), logits_of(net.model, ids), atol=1e-5, rtol=1e-5)


def test_embeddings_optional_and_tied():
    host = make_host(seed=18, tie=True)
    spec = make_specialist(host, 61)
    with torch.no_grad():
        spec.get_input_embeddings().weight.add_(0.3 * torch.randn_like(spec.get_input_embeddings().weight))
    off = mine_delta(host, spec, "full", name="e")
    assert off.attrs["embedding_rel_diff_if_excluded"] > 0.05
    assert not any("embed" in m for m in off.factors)
    on = mine_delta(host, spec, "full", name="e", include_embeddings=True)
    assert "model.embed_tokens" in on.factors
    net = SkillDeltaNetwork(copy.deepcopy(host), [on])
    ids = ids_batch(seed=19)
    net.set_expert("e")
    assert torch.allclose(logits_of(net.model, ids), logits_of(spec, ids), atol=1e-5, rtol=1e-5)
    net.set_expert(BASE)
    assert torch.equal(logits_of(net.model, ids), logits_of(host, ids))


def test_mining_from_saved_directories_matches_models(tmp_path):
    host = make_host(seed=20)
    spec = make_specialist(host, 71, theta=1e6)
    host.save_pretrained(str(tmp_path / "host"))
    spec.save_pretrained(str(tmp_path / "spec"))
    from_dirs = mine_delta(str(tmp_path / "host"), str(tmp_path / "spec"), 8)
    from_models = mine_delta(host, spec, 8)
    assert from_dirs.attrs["rope_theta"] == 1e6
    assert set(from_dirs.factors) == set(from_models.factors)
    for module in from_models.factors:
        a = from_dirs.factors[module]
        b = from_models.factors[module]
        assert torch.allclose(a["B"] @ a["A"], b["B"] @ b["A"], atol=1e-5)
    assert set(from_dirs.vectors) == set(from_models.vectors)
