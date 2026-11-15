import copy

import pytest
import torch

transformers = pytest.importorskip("transformers")

from tests.test_atlas import GENERAL_WORDS, SKILL_WORDS, texts, tokenizer, word_ids
from wsp import transplant
from wsp.lora import lora_modules
from wsp.skills_eval import model_params
from wsp.transplant import (TransplantBank, banks, build_transplant, count_params, fetch_parts, fit_stitches, heal,
                            load_transplant, make_layer_map, parts_from_model, ridge_solve, routing_shares,
                            save_transplant, select_experts, set_trainable, source_active_params, track_routing)

D = 48
EXPERT = 3 * D * 40


def tiny_base(layers=3, seed=0):
    torch.manual_seed(seed)
    return transformers.Qwen3ForCausalLM(transformers.Qwen3Config(
        hidden_size=D, intermediate_size=96, num_hidden_layers=layers, num_attention_heads=4,
        num_key_value_heads=2, vocab_size=40, max_position_embeddings=64, head_dim=12,
    )).eval()


def tiny_source(layers=4, seed=1):
    torch.manual_seed(seed)
    return transformers.Qwen3MoeForCausalLM(transformers.Qwen3MoeConfig(
        hidden_size=D, intermediate_size=96, moe_intermediate_size=40, num_hidden_layers=layers,
        num_attention_heads=4, num_key_value_heads=2, vocab_size=40, max_position_embeddings=64,
        num_experts=6, num_experts_per_tok=2, decoder_sparse_step=1, mlp_only_layers=[], head_dim=12,
        norm_topk_prob=True,
    )).eval()


def plan_for(source, base_layers, k, ids=None):
    layer_map = make_layer_map(base_layers, len(source.model.layers))
    plan = {"layer_map": layer_map, "k": k, "control": None, "seed": 0, "experts": {}, "labels": {}, "skills": ["x"]}
    for layer in sorted(set(layer_map)):
        chosen = list(ids) if ids is not None else list(range(k))
        plan["experts"][str(layer)] = chosen
        plan["labels"][str(layer)] = ["x"] * len(chosen)
    return plan


def make(base_layers=3, k=4, ids=None, lora_rank=None, rank=None, seed=0):
    base = tiny_base(base_layers, seed)
    source = tiny_source()
    plan = plan_for(source, base_layers, k, ids)
    parts = parts_from_model(source, plan)
    reference = copy.deepcopy(base)
    model = build_transplant(base, plan, parts, lora_rank=lora_rank, stitch_rank=rank, base_id="base-local",
                             source_id="source-local", pair="test")
    return model, reference, source, plan


def logits(model, ids):
    with torch.no_grad():
        return model(ids).logits


def randomize(model, seed=3):
    """Move gates and stitches off their start so the bank matters."""
    torch.manual_seed(seed)
    with torch.no_grad():
        for bank in banks(model):
            bank.gate.fill_(0.7)
            for param in list(bank.stitch_in.parameters()) + list(bank.stitch_out.parameters()):
                param.add_(0.02 * torch.randn_like(param))
            bank.router.weight.add_(0.05 * torch.randn_like(bank.router.weight))


def test_fresh_transplant_equals_base():
    model, reference, _, _ = make()
    ids = torch.randint(0, 40, (2, 9))
    assert torch.equal(logits(model, ids), logits(reference, ids))
    assert len(banks(model)) == 3


def test_fresh_transplant_with_ridge_stitches_equals_base():
    base = tiny_base()
    source = tiny_source()
    reference = copy.deepcopy(base)
    plan = plan_for(source, 3, 4)
    tok = tokenizer()
    sample = texts(SKILL_WORDS + GENERAL_WORDS, 8, 12, seed=1)
    fits = fit_stitches(base, source, tok, sample, plan["layer_map"], ridge=1e-2, batch_size=3)
    model = build_transplant(base, plan, parts_from_model(source, plan), stitches=fits)
    ids = torch.randint(0, 40, (2, 9))
    assert torch.equal(logits(model, ids), logits(reference, ids))
    with torch.no_grad():
        for bank in banks(model):
            bank.gate.fill_(1.0)
    assert (logits(model, ids) - logits(reference, ids)).abs().max() > 1e-4


def test_gate_gets_a_gradient_at_zero():
    model, _, _, _ = make()
    ids = torch.randint(0, 40, (2, 9))
    loss = model(ids, labels=ids).loss
    loss.backward()
    for bank in banks(model):
        assert bank.gate.grad is not None
        assert bank.gate.grad.abs().item() > 0


def planted_selection():
    tok = tokenizer()
    source = tiny_source()
    words_a = SKILL_WORDS[:4]
    words_b = SKILL_WORDS[4:]
    torch.manual_seed(0)
    dir_a = torch.randn(D)
    dir_a = dir_a / dir_a.norm()
    dir_b = torch.randn(D)
    dir_b = dir_b - (dir_b @ dir_a) * dir_a
    dir_b = dir_b / dir_b.norm()
    with torch.no_grad():
        embed = source.model.embed_tokens.weight
        for token in word_ids(tok, words_a):
            embed[token] = 4.0 * dir_a + 0.01 * torch.randn(D)
        for token in word_ids(tok, words_b):
            embed[token] = 4.0 * dir_b + 0.01 * torch.randn(D)
        for token in word_ids(tok, GENERAL_WORDS):
            embed[token] = -4.0 * (dir_a + dir_b) + 0.01 * torch.randn(D)
        source.model.layers[0].mlp.gate.weight[3] = 20.0 * dir_a
        source.model.layers[0].mlp.gate.weight[5] = 20.0 * dir_b
    skills = {"math": texts(words_a, 4, 10, seed=6), "code": texts(words_b, 4, 10, seed=7)}
    general = texts(GENERAL_WORDS, 4, 10, seed=8)
    layer_map = make_layer_map(3, 4)
    plan = select_experts(source, tok, skills, general, layer_map, k=2)
    assert layer_map == [0, 2, 3]
    assert plan["experts"]["0"] == [3, 5]
    assert plan["labels"]["0"] == ["math", "code"]
    assert sorted(plan["experts"].keys()) == ["0", "2", "3"]
    for key in plan["experts"]:
        assert len(set(plan["experts"][key])) == 2
    return plan, source, tok, skills, general, layer_map


def test_selection_picks_the_planted_selective_experts():
    plan = planted_selection()[0]
    assert plan["experts"]["0"] == [3, 5]


def test_random_control_differs_from_the_real_pick():
    plan, source, tok, skills, general, layer_map = planted_selection()
    random_plan = select_experts(source, tok, skills, general, layer_map, k=2, control="random", seed=5)
    again = select_experts(source, tok, skills, general, layer_map, k=2, control="random", seed=5)
    other = select_experts(source, tok, skills, general, layer_map, k=2, control="random", seed=6)
    assert random_plan["experts"] == again["experts"]
    assert random_plan["experts"] != other["experts"]
    assert random_plan["experts"]["0"] != plan["experts"]["0"]
    for key in random_plan["experts"]:
        picked = random_plan["experts"][key]
        assert len(picked) == 2 and len(set(picked)) == 2
        assert all(0 <= expert < 6 for expert in picked)
        assert random_plan["labels"][key] == ["random", "random"]
    with pytest.raises(ValueError):
        select_experts(source, tok, skills, general, layer_map, k=7)


def test_k_larger_than_skill_count_takes_turns():
    ids, labels = transplant.pick_round_robin({"a": [4, 1, 2, 0, 3], "b": [4, 2, 1, 0, 3], "c": [0, 1, 2, 3, 4]}, 5)
    assert ids == [4, 2, 0, 1, 3]
    assert labels == ["a", "b", "c", "a", "b"]


def test_router_init_reproduces_source_routing_with_identity_stitch():
    base = tiny_base()
    source = tiny_source()
    plan = plan_for(source, 3, 6, ids=list(range(6)))
    model = build_transplant(base, plan, parts_from_model(source, plan))
    bank = banks(model)[0]
    moe = source.model.layers[0].mlp
    norm = source.model.layers[0].post_attention_layernorm
    torch.manual_seed(2)
    h = torch.randn(20, D) * 3.0
    with torch.no_grad():
        x, idx, weights = bank.route(h)
        probs = torch.softmax(moe.gate(norm(h)), dim=-1)
        vals, want_idx = probs.topk(2, dim=-1)
        want = vals / vals.sum(-1, keepdim=True)
        assert torch.equal(idx, want_idx)
        assert (weights - want).abs().max() < 1e-6
        # With all experts and identity stitches the bank is the source's own sparse MLP.
        direct = moe(norm(h).unsqueeze(0))[0].squeeze(0)
        assert (bank.expert_sum(h.unsqueeze(0)).squeeze(0) - direct).abs().max() < 1e-5


def test_router_on_a_subset_is_the_restricted_source_softmax():
    base = tiny_base()
    source = tiny_source()
    plan = plan_for(source, 3, 3, ids=[1, 3, 4])
    model = build_transplant(base, plan, parts_from_model(source, plan))
    bank = banks(model)[0]
    moe = source.model.layers[0].mlp
    norm = source.model.layers[0].post_attention_layernorm
    h = torch.randn(15, D) * 2.0
    with torch.no_grad():
        _, idx, weights = bank.route(h)
        logits_full = moe.gate(norm(h))[:, [1, 3, 4]]
        probs = torch.softmax(logits_full, dim=-1)
        vals, want_idx = probs.topk(2, dim=-1)
        want = vals / vals.sum(-1, keepdim=True)
    assert torch.equal(idx, want_idx)
    assert (weights - want).abs().max() < 1e-6


def test_only_the_intended_params_train():
    model, _, _, _ = make(lora_rank=4, rank=8)
    trainable = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            trainable[name] = param
    allowed = ("stitch_in", "stitch_out", "router", "gate", "lora_A", "lora_B")
    assert len(trainable) > 0
    for name in trainable:
        assert any(part in name for part in allowed), name
        assert ".experts." not in name
        assert "src_norm" not in name
    kinds = set()
    for name in trainable:
        for part in allowed:
            if part in name:
                kinds.add(part)
    assert kinds == set(allowed)
    groups = set_trainable(model)
    expected = sum(len(group) for group in groups.values())
    assert expected == len(trainable)
    for name, module in lora_modules(model):
        assert ".experts." not in name
        assert name.endswith(("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"))
    for bank in banks(model):
        for param in bank.experts.parameters():
            assert not param.requires_grad
        for param in bank.src_norm.parameters():
            assert not param.requires_grad


def test_heal_changes_only_the_trainable_params():
    model, _, source, _ = make(lora_rank=4)
    randomize(model)
    before = {}
    for name, param in model.named_parameters():
        before[name] = param.detach().clone()
    groups = set_trainable(model)
    train_groups = [{"params": groups["stitch"], "lr": 1e-3}, {"params": groups["router"], "lr": 1e-3},
                    {"params": groups["gate"], "lr": 1e-2}, {"params": groups["lora"], "lr": 1e-3}]
    tok = tokenizer()
    sample = texts(SKILL_WORDS + GENERAL_WORDS, 8, 10, seed=2)
    log = heal(model, tok, sample, train_groups, steps=3, batch_size=2, seq_len=16, seed=0, teacher=source,
               log_every=0)
    assert log["kd_first"] > 0
    changed = set()
    for name, param in model.named_parameters():
        if not torch.equal(before[name], param.detach()):
            changed.add(name)
    assert len(changed) > 0
    for name in changed:
        assert any(part in name for part in ("stitch_in", "stitch_out", "router", "gate", "lora_")), name
    assert any("gate" in name for name in changed)
    assert any("router" in name for name in changed)


def test_save_and_load_give_equal_logits(tmp_path):
    base = tiny_base()
    source = tiny_source()
    source_dir = str(tmp_path / "source")
    source.save_pretrained(source_dir)
    plan = plan_for(source, 3, 4, ids=[0, 2, 3, 5])
    reference = copy.deepcopy(base)
    model = build_transplant(base, plan, parts_from_model(source, plan), lora_rank=4, stitch_rank=16,
                             base_id="base-local", source_id=source_dir, pair="test")
    randomize(model)
    with torch.no_grad():
        for name, param in model.named_parameters():
            if "lora_B" in name:
                param.add_(0.05 * torch.randn_like(param))
    out_dir = str(tmp_path / "out")
    save_transplant(model, out_dir)
    loaded, _ = load_transplant(out_dir, dtype=torch.float32, base_model=copy.deepcopy(reference))
    ids = torch.randint(0, 40, (2, 11))
    assert (logits(model, ids) - logits(loaded, ids)).abs().max() < 1e-6
    assert (logits(model, ids) - logits(reference, ids)).abs().max() > 1e-4
    got = count_params(loaded)
    want = count_params(model)
    assert got == want


def test_parameter_counts():
    model, reference, _, _ = make(k=4)
    _, base_run = model_params(reference)
    base_total, _ = model_params(reference)
    counts = count_params(model)
    fixed = (D * D + D) + D * D + (4 * D + 4) + 1 + D
    assert counts["base"] == base_total
    assert counts["experts"] == 3 * 4 * EXPERT
    assert counts["total"] == base_total + 3 * (4 * EXPERT + fixed)
    assert counts["active"] == base_run + 3 * (2 * EXPERT + fixed)
    assert abs(counts["total_ratio"] - counts["total"] / base_total) < 1e-12
    assert counts["active"] < counts["total"]
    low, _, _, _ = make(k=4, rank=8)
    assert count_params(low)["stitch"] == 3 * (2 * (D * 8 + 8 * D) + D)
    lora, _, _, _ = make(k=4, lora_rank=4)
    assert count_params(lora)["lora"] > 0
    assert count_params(lora)["base"] == base_total


def test_source_active_params():
    source = tiny_source()
    total, active = source_active_params(source)
    plain, _ = model_params(source)
    assert total == plain
    _, run = model_params(source)
    assert active == run - 4 * EXPERT * 4


def test_kv_cache_generation_equals_uncached():
    model, _, _, _ = make(k=4)
    randomize(model)
    ids = torch.randint(0, 40, (2, 7))
    cached = model.generate(ids, max_new_tokens=9, do_sample=False, use_cache=True)
    plain = model.generate(ids, max_new_tokens=9, do_sample=False, use_cache=False)
    assert torch.equal(cached, plain)
    full = logits(model, cached)
    step = None
    with torch.no_grad():
        out = model(cached[:, :-1], use_cache=True)
        step = model(cached[:, -1:], past_key_values=out.past_key_values, use_cache=True).logits[:, -1]
    assert (full[:, -1] - step).abs().max() < 1e-4


def test_shared_source_layer_for_two_base_layers():
    base = tiny_base(layers=6)
    source = tiny_source(layers=2)
    plan = plan_for(source, 6, 3)
    assert plan["layer_map"] == [0, 0, 0, 1, 1, 1]
    model = build_transplant(base, plan, parts_from_model(source, plan))
    reference = tiny_base(layers=6)
    ids = torch.randint(0, 40, (2, 8))
    assert torch.equal(logits(model, ids), logits(reference, ids))
    assert banks(model)[0].router is not banks(model)[1].router
    randomize(model)
    assert logits(model, ids).shape == (2, 8, 40)


def test_routing_shares_sum_to_one():
    model, _, _, plan = make(k=4)
    randomize(model)
    track_routing(model)
    model(torch.randint(0, 40, (2, 10)))
    shares = routing_shares(model)
    assert len(shares["per_layer"]) == 3
    for row in shares["per_layer"]:
        assert abs(sum(row) - 1.0) < 1e-9
    assert abs(sum(shares["by_label"].values()) - 1.0) < 1e-9
    track_routing(model, on=False)


def test_make_layer_map_snaps_to_sparse_layers():
    assert make_layer_map(4, 8) == [0, 2, 5, 7]
    assert make_layer_map(4, 8, [2, 3, 4, 5, 6, 7]) == [2, 2, 5, 7]
    config = transformers.Qwen3MoeConfig(num_hidden_layers=6, mlp_only_layers=[0, 1], decoder_sparse_step=1,
                                         num_experts=4)
    assert transplant.source_moe_layers(config) == [2, 3, 4, 5]


def test_ridge_solve_recovers_a_linear_map():
    torch.manual_seed(0)
    x = torch.randn(400, 6, dtype=torch.float64) * 3.0 + 1.0
    w = torch.randn(5, 6, dtype=torch.float64)
    b = torch.randn(5, dtype=torch.float64)
    y = x @ w.t() + b
    weight, bias = ridge_solve(x.t() @ x, x.t() @ y, x.sum(0), y.sum(0), 400, 1e-9, True)
    assert (weight - w).abs().max() < 1e-5
    assert (bias - b).abs().max() < 1e-5
    y0 = x @ w.t()
    weight0, none = ridge_solve(x.t() @ x, x.t() @ y0, None, None, 400, 1e-9, False)
    assert none is None
    assert (weight0 - w).abs().max() < 1e-5


def test_fit_stitches_shapes_and_rank():
    base = tiny_base()
    source = tiny_source()
    plan = plan_for(source, 3, 4)
    tok = tokenizer()
    sample = texts(SKILL_WORDS + GENERAL_WORDS, 12, 12, seed=4)
    full = fit_stitches(base, source, tok, sample, plan["layer_map"], ridge=1e-3, batch_size=4)
    low = fit_stitches(base, source, tok, sample, plan["layer_map"], rank=4, ridge=1e-3, batch_size=4)
    assert sorted(full) == [0, 1, 2]
    for index in full:
        assert full[index]["in_w"].shape == (D, D)
        assert full[index]["in_b"].shape == (D,)
        assert full[index]["out_w"].shape == (D, D)
        assert low[index]["r2_in"] <= full[index]["r2_in"] + 1e-6
        assert low[index]["r2_out"] <= full[index]["r2_out"] + 1e-6
        assert full[index]["r2_in"] <= 1.0
    model = build_transplant(base, plan, parts_from_model(source, plan), stitches=low, stitch_rank=4)
    bank = banks(model)[0]
    assert torch.linalg.matrix_rank(bank.stitch_in.matrix()) <= 4
    assert (bank.stitch_in.linear is None) and bank.stitch_in.down.weight.shape == (4, D)


def test_fetch_parts_matches_parts_from_model(tmp_path):
    source = tiny_source()
    source_dir = str(tmp_path / "source")
    source.save_pretrained(source_dir)
    plan = plan_for(source, 3, 3, ids=[1, 2, 5])
    direct = parts_from_model(source, plan)
    fetched = fetch_parts(source_dir, plan, dtype=torch.float32)
    assert sorted(direct) == sorted(fetched)
    for key in direct:
        assert torch.equal(direct[key]["router"], fetched[key]["router"])
        assert torch.equal(direct[key]["pre_norm"].weight, fetched[key]["pre_norm"].weight)
        for a, b in zip(direct[key]["experts"], fetched[key]["experts"]):
            assert torch.equal(a.down_proj.weight, b.down_proj.weight)
            assert torch.equal(a.gate_proj.weight, b.gate_proj.weight)


def test_bank_class_exported():
    assert TransplantBank is transplant.TransplantBank


def fake_rows(arm, values, seeds=(0, 1)):
    rows = []
    for seed, shift in zip(seeds, (0.0, 0.004)):
        scores = {}
        for skill in values:
            scores[skill] = values[skill] + shift
        rows.append({"arm": arm, "seed": seed, "scores": scores})
    return rows


def test_go_stop_rule():
    skills = ["gsm8k", "math500", "mbpp"]
    counts = {"gsm8k": 500, "math500": 500, "mbpp": 257}
    rows = []
    rows += fake_rows("transplant_k4", {"gsm8k": 0.60, "math500": 0.40, "mbpp": 0.50})
    rows += fake_rows("kd_lora", {"gsm8k": 0.50, "math500": 0.39, "mbpp": 0.40})
    rows += fake_rows("transplant_k4_random", {"gsm8k": 0.51, "math500": 0.30, "mbpp": 0.41})
    result = transplant.go_stop(rows, skills, counts)
    assert result["skills"]["gsm8k"]["passes"] and result["skills"]["mbpp"]["passes"]
    assert not result["skills"]["math500"]["passes"]
    assert result["verdict"] == "GO"
    rows[3]["scores"]["mbpp"] = 0.495
    rows[2]["scores"]["mbpp"] = 0.495
    assert transplant.go_stop(rows, skills, counts)["verdict"] == "STOP"
    assert transplant.go_stop([], skills, counts)["verdict"] == "STOP"


def test_rank_truncated_stitch_bias_matches_the_truncated_weight():
    """The intercept of stitch_in must be refit for the rank r weight, or the mean state is off."""
    base = tiny_base()
    source = tiny_source()
    plan = plan_for(source, 3, 4)
    tok = tokenizer()
    sample = texts(SKILL_WORDS + GENERAL_WORDS, 12, 12, seed=4)
    low = fit_stitches(base, source, tok, sample, plan["layer_map"], rank=4, ridge=1e-3, batch_size=4)
    probe = transplant.Stitch(D, D, 4, bias=True)
    for index in low:
        entry = low[index]
        probe.set_matrix(entry["in_w"], entry["in_b"])
        weight = probe.matrix()
        # The mean of the fitted output must equal the mean source state: b = my - W_r mx.
        # Rebuild mx and my from a fresh pass and compare.
        seen = {}
        layer = transplant.decoder_layers(base)[index]
        src_layer = transplant.decoder_layers(source)[plan["layer_map"][index]]
        h1 = layer.post_attention_layernorm.register_forward_pre_hook(lambda m, a: seen.__setitem__("b", a[0]))
        h2 = src_layer.post_attention_layernorm.register_forward_pre_hook(lambda m, a: seen.__setitem__("s", a[0]))
        sums_b = 0
        sums_s = 0
        count = 0
        tok.padding_side = "right"
        for start in range(0, len(sample), 4):
            enc = tok(sample[start:start + 4], return_tensors="pt", padding=True, truncation=True, max_length=256)
            mask = enc["attention_mask"].reshape(-1).bool()
            with torch.no_grad():
                base(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"], use_cache=False)
                source(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"], use_cache=False)
            sums_b = sums_b + seen["b"].reshape(-1, D)[mask].double().sum(0)
            sums_s = sums_s + seen["s"].reshape(-1, D)[mask].double().sum(0)
            count += int(mask.sum())
        h1.remove()
        h2.remove()
        mx = (sums_b / count).float()
        my = (sums_s / count).float()
        predicted = weight @ mx + entry["in_b"]
        assert (predicted - my).abs().max() < 1e-4


def test_min_freq_ranks_rarely_used_experts_last():
    log_ratio = [2.0, 1.0, 0.5, 0.1]
    freq = [0.001, 0.05, 0.04, 0.06]
    assert transplant.ranked_experts(log_ratio) == [0, 1, 2, 3]
    assert transplant.ranked_experts(log_ratio, freq, 0.02) == [1, 2, 3, 0]
    assert transplant.ranked_experts(log_ratio, freq, 0.0) == [0, 1, 2, 3]


def test_shared_source_layer_gives_each_bank_its_own_experts():
    base = tiny_base(layers=6)
    source = tiny_source(layers=2)
    plan = plan_for(source, 6, 3)
    model = build_transplant(base, plan, parts_from_model(source, plan))
    found = banks(model)
    assert plan["layer_map"] == [0, 0, 0, 1, 1, 1]
    ids = set()
    for bank in found:
        for param in bank.experts.parameters():
            assert id(param) not in ids
            ids.add(id(param))
        for param in bank.src_norm.parameters():
            assert id(param) not in ids
            ids.add(id(param))
    assert count_params(model)["experts"] == 6 * 3 * EXPERT
