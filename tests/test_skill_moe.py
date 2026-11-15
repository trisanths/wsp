import copy

import pytest
import torch

transformers = pytest.importorskip("transformers")

from wsp.lora import lora_modules
from wsp.skill_moe import (ROUTER_ADAM_EPS, SkillMoE, StitchedExpert, build_skill_moe, count_params,
                           load_balance_loss, load_skill_moe, moe_layers, router_parameters, routing_shares,
                           save_skill_moe, stitch_parameters, stitched, track_routing)


def tiny_qwen2(seed):
    torch.manual_seed(seed)
    config = transformers.Qwen2Config(
        hidden_size=48, intermediate_size=96, num_hidden_layers=3,
        num_attention_heads=4, num_key_value_heads=2, vocab_size=300,
    )
    return transformers.Qwen2ForCausalLM(config).eval()


def tiny_llama(seed):
    torch.manual_seed(seed)
    config = transformers.LlamaConfig(
        hidden_size=32, intermediate_size=64, num_hidden_layers=4,
        num_attention_heads=4, num_key_value_heads=2, vocab_size=300,
    )
    return transformers.LlamaForCausalLM(config).eval()


def param_count(model):
    total = 0
    for param in model.parameters():
        total += param.numel()
    return total


def logits(model, ids):
    with torch.no_grad():
        return model(ids).logits


def moe_of(top_k, load_balance=False):
    base = tiny_qwen2(0)
    reference = copy.deepcopy(base)
    model = build_skill_moe(base, [tiny_qwen2(1), tiny_qwen2(2)], top_k=top_k, load_balance=load_balance)
    return model, reference


@pytest.mark.parametrize("top_k", [1, 2, 3])
def test_fresh_moe_equals_base(top_k):
    model, reference = moe_of(top_k)
    ids = torch.randint(0, 300, (2, 9))
    gap = (logits(model, ids) - logits(reference, ids)).abs().max().item()
    assert gap < 1e-6
    assert len(moe_layers(model)) == 3


def test_only_routers_train():
    model, _ = moe_of(2)
    trainable = []
    for param in model.parameters():
        if param.requires_grad:
            trainable.append(param)
    routers = router_parameters(model)
    assert len(trainable) == len(routers)
    for param in routers:
        assert param.requires_grad


def test_top1_picks_one_expert_per_token():
    model, _ = moe_of(1)
    # Random routers, so tokens spread over experts.
    torch.manual_seed(5)
    with torch.no_grad():
        for moe in moe_layers(model):
            moe.router.weight.normal_(0, 3.0)
            moe.router_bias.zero_()
    moe = moe_layers(model)[1]
    calls = []

    def count_rows(module, args, output):
        calls.append(args[0].shape[0])

    handles = []
    for expert in moe.experts:
        handles.append(expert.register_forward_hook(count_rows))
    track_routing(model)
    ids = torch.randint(0, 300, (2, 10))
    logits(model, ids)
    for handle in handles:
        handle.remove()
    # Each of the 20 tokens ran exactly one expert, and more than one expert was used.
    assert sum(calls) == 20
    assert len(calls) > 1
    counts = moe.route_counts
    assert float(counts.sum()) == 20.0
    shares = routing_shares(model)
    assert abs(sum(shares["mean"]) - 1.0) < 1e-9


def teacher_with_expert_mlps(base, expert):
    """The base with every layer's MLP and pre-MLP norm taken from expert."""
    teacher = copy.deepcopy(base)
    for index in range(len(teacher.model.layers)):
        layer = teacher.model.layers[index]
        source = expert.model.layers[index]
        layer.mlp.load_state_dict(source.mlp.state_dict())
        layer.post_attention_layernorm.load_state_dict(source.post_attention_layernorm.state_dict())
    return teacher.eval()


def shares_on(model, low, high, generator):
    track_routing(model)
    logits(model, torch.randint(low, high, (32, 12), generator=generator))
    shares = routing_shares(model)["mean"]
    track_routing(model, on=False)
    return shares


def test_router_learns_the_better_expert():
    base = tiny_qwen2(0)
    expert_a = tiny_qwen2(1)
    expert_b = tiny_qwen2(2)
    reference = copy.deepcopy(base)
    teacher = teacher_with_expert_mlps(base, expert_a)
    model = build_skill_moe(base, [expert_a, expert_b], top_k=1, load_balance=True)
    # Tokens 0..19 want expert A's MLPs; tokens 20..39 want the base.
    generator = torch.Generator().manual_seed(0)
    before = shares_on(model, 0, 20, generator)
    optimizer = torch.optim.Adam(router_parameters(model), lr=1e-2, eps=ROUTER_ADAM_EPS)
    for step in range(250):
        ids = torch.randint(0, 40, (8, 12), generator=generator)
        with torch.no_grad():
            target = torch.where((ids < 20).unsqueeze(-1), teacher(ids).logits, reference(ids).logits)
        loss = ((model(ids).logits - target) ** 2).mean()
        aux = load_balance_loss(model)
        assert aux is not None
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    low = shares_on(model, 0, 20, generator)
    high = shares_on(model, 20, 40, generator)
    assert before[1] < 0.01
    assert low[1] > 0.5
    assert low[1] > high[1] + 0.3
    assert high[0] > 0.8


def test_cross_family_stitched_expert_starts_as_no_op():
    base = tiny_qwen2(0)
    reference = copy.deepcopy(base)
    donor = tiny_llama(3)
    model = build_skill_moe(base, [tiny_qwen2(1), stitched(donor)], top_k=3)
    ids = torch.randint(0, 300, (2, 8))
    assert (logits(model, ids) - logits(reference, ids)).abs().max().item() < 1e-6
    moe = moe_layers(model)[0]
    expert = moe.experts[2]
    assert isinstance(expert, StitchedExpert)
    # A fresh stitched expert outputs exactly zero.
    h = torch.randn(5, 48)
    x = moe.experts[0].pre_norm(h)
    assert torch.equal(expert(h, x), torch.zeros(5, 48))
    # It trains: the stitches get gradients once the router lets it in.
    with torch.no_grad():
        for layer in moe_layers(model):
            layer.router_bias.zero_()
    out = model(ids).logits
    out.pow(2).mean().backward()
    grads = 0
    for param in stitch_parameters(model):
        assert param.requires_grad
        if param.grad is not None and param.grad.abs().sum() > 0:
            grads += 1
    assert grads > 0
    # Donor layers follow depth: 3 base layers over 4 donor layers.
    assert model.wsp_skill_moe["experts"][1]["layer_map"] == [0, 2, 3]


def test_plain_expert_of_another_shape_is_refused():
    with pytest.raises(ValueError, match="stitched"):
        build_skill_moe(tiny_qwen2(0), [tiny_llama(1)], top_k=1)


def test_save_and_load_round_trip(tmp_path):
    paths = []
    for name, model in [("base", tiny_qwen2(0)), ("coder", tiny_qwen2(1)), ("llama", tiny_llama(3))]:
        path = str(tmp_path / name)
        model.save_pretrained(path)
        paths.append(path)
    base = transformers.AutoModelForCausalLM.from_pretrained(paths[0], dtype=torch.float32)
    coder = transformers.AutoModelForCausalLM.from_pretrained(paths[1], dtype=torch.float32)
    llama = transformers.AutoModelForCausalLM.from_pretrained(paths[2], dtype=torch.float32)
    model = build_skill_moe(base, [coder, stitched(llama, kind="mlp")], top_k=2, lora_rank=4)
    # Give every trained piece nonzero values, so the round trip means something.
    torch.manual_seed(9)
    with torch.no_grad():
        for moe in moe_layers(model):
            moe.router.weight.normal_(0, 1.0)
            moe.router_bias.normal_(0, 1.0)
        for param in stitch_parameters(model):
            param.normal_(0, 0.1)
        for _, module in lora_modules(model):
            module.lora_B.normal_(0, 0.1)
    model.eval()
    ids = torch.randint(0, 300, (2, 8))
    expected = logits(model, ids)
    out_dir = str(tmp_path / "moe")
    save_skill_moe(model, out_dir)
    loaded, tokenizer = load_skill_moe(out_dir, dtype=torch.float32)
    assert tokenizer is None
    assert torch.allclose(logits(loaded, ids), expected, atol=1e-6)
    assert loaded.wsp_skill_moe["lora"]["rank"] == 4
    assert len(lora_modules(loaded)) == 12


def test_active_params_for_top1():
    base = tiny_qwen2(0)
    base_params = param_count(base)
    model = build_skill_moe(base, [tiny_qwen2(1), tiny_qwen2(2)], top_k=1)
    counts = count_params(model)
    assert counts["active"] == base_params + counts["router"]
    assert counts["router"] == 3 * (48 * 3 + 3)
    assert counts["total"] == base_params + counts["router"] + counts["experts"]
    soft = count_params(build_skill_moe(tiny_qwen2(0), [tiny_qwen2(1), tiny_qwen2(2)], top_k=3))
    assert soft["active"] == soft["total"]


def test_skill_moe_needs_two_experts():
    with pytest.raises(ValueError):
        SkillMoE([torch.nn.Identity()], 8, 1)


def test_experts_do_not_tie_at_init():
    # With all-zero expert rows, top-k ties always picked expert 1, and the
    # other expert got no gradient from the LM loss. The noise on the expert
    # rows gives every expert some slots and some gradient from the start.
    torch.manual_seed(0)
    model, _ = moe_of(2)
    track_routing(model)
    ids = torch.randint(0, 300, (8, 16))
    out = model(input_ids=ids, labels=ids)
    shares = routing_shares(model)["mean"]
    track_routing(model, on=False)
    assert abs(shares[0] - 0.5) < 1e-9
    assert shares[1] > 0.1
    assert shares[2] > 0.1
    out.loss.backward()
    for moe in moe_layers(model):
        grad = moe.router.weight.grad
        assert grad[1].abs().sum() > 0
        assert grad[2].abs().sum() > 0
        assert not torch.equal(moe.router.weight[1], moe.router.weight[2])
        assert torch.equal(moe.router.weight[0], torch.zeros(48))


def test_top1_experts_do_not_train_in_lockstep():
    torch.manual_seed(0)
    model, _ = moe_of(1, load_balance=True)
    optimizer = torch.optim.Adam(router_parameters(model), lr=1e-2, eps=ROUTER_ADAM_EPS)
    for step in range(150):
        ids = torch.randint(0, 300, (4, 16))
        loss = model(input_ids=ids, labels=ids).loss + 0.01 * load_balance_loss(model)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    track_routing(model)
    logits(model, torch.randint(0, 300, (8, 16)))
    shares = routing_shares(model)["mean"]
    # Once the base gives up slots, both experts get some of them.
    assert shares[0] < 0.99
    assert shares[1] > 0.01
    assert shares[2] > 0.01


def test_no_noise_gives_all_zero_router():
    base = tiny_qwen2(0)
    model = build_skill_moe(base, [tiny_qwen2(1), tiny_qwen2(2)], top_k=2, router_noise=0.0)
    for moe in moe_layers(model):
        assert torch.equal(moe.router.weight, torch.zeros(3, 48))


def test_lora_heal_trains_only_routers_and_lora():
    base = tiny_qwen2(0)
    model = build_skill_moe(base, [tiny_qwen2(1), tiny_qwen2(2)], top_k=2, lora_rank=4)
    wanted = set()
    for param in router_parameters(model):
        wanted.add(id(param))
    for _, module in lora_modules(model):
        wanted.add(id(module.lora_A))
        wanted.add(id(module.lora_B))
    ids = torch.randint(0, 300, (2, 8))
    model(input_ids=ids, labels=ids).loss.backward()
    for name, param in model.named_parameters():
        if id(param) in wanted:
            assert param.requires_grad, name
        else:
            assert not param.requires_grad, name
            assert param.grad is None, name
    # The LoRA sits on the base attention only, not on any expert MLP.
    for name, _ in lora_modules(model):
        assert ".self_attn." in name


def test_cached_generation_matches_full_forward():
    base = tiny_qwen2(0)
    model = build_skill_moe(base, [tiny_qwen2(1), stitched(tiny_llama(3))], top_k=2)
    torch.manual_seed(1)
    with torch.no_grad():
        for moe in moe_layers(model):
            moe.router.weight.normal_(0, 2.0)
            moe.router_bias.zero_()
        for param in stitch_parameters(model):
            param.normal_(0, 0.1)
    ids = torch.randint(0, 300, (1, 6))
    cached = model.generate(ids, max_new_tokens=8, do_sample=False, use_cache=True)
    plain = model.generate(ids, max_new_tokens=8, do_sample=False, use_cache=False)
    assert torch.equal(cached, plain)
    with torch.no_grad():
        full = model(cached).logits
        first = model(cached[:, :10], use_cache=True)
        step = model(cached[:, 10:11], past_key_values=first.past_key_values, use_cache=True).logits
    assert (full[:, 10] - step[:, 0]).abs().max().item() < 1e-5


# ---------------------------------------------------------------- run 16c: forcing, whole layers, ridge, sequences

from wsp.skill_moe import (LayerMoE, clear_force, clear_sequence_routing, force_route, init_router_ridge,
                           set_sequence_routing)


def tiny_rope(seed, rope_theta):
    torch.manual_seed(seed)
    config = transformers.Qwen2Config(
        hidden_size=48, intermediate_size=96, num_hidden_layers=3,
        num_attention_heads=4, num_key_value_heads=2, vocab_size=300, rope_theta=rope_theta,
    )
    return transformers.Qwen2ForCausalLM(config).eval()


class FakeTokenizer:
    """Texts are space-separated token ids."""

    def __call__(self, text, return_tensors=None, truncation=None, max_length=None, add_special_tokens=None):
        ids = [int(part) for part in text.split()]
        if max_length is not None:
            ids = ids[:max_length]
        return {"input_ids": torch.tensor([ids])}


def planted_texts(generator, per_class=40, length=16):
    ranges = {"base": (100, 200), "math": (0, 100), "coder": (200, 300)}
    texts = {}
    for name in ranges:
        low, high = ranges[name]
        rows = []
        for _ in range(per_class):
            ids = torch.randint(low, high, (length,), generator=generator).tolist()
            rows.append(" ".join(str(i) for i in ids))
        texts[name] = rows
    return texts


def plant_domains(model, strength=2.0):
    """Give the three token ranges (math 0-99, general 100-199, code 200-299) a shared embedding offset."""
    generator = torch.Generator().manual_seed(11)
    weight = model.get_input_embeddings().weight
    with torch.no_grad():
        for index in range(3):
            direction = torch.randn(weight.shape[1], generator=generator)
            weight[index * 100:(index + 1) * 100] += strong(direction, weight, strength)
    return model


def strong(direction, weight, strength):
    return direction / direction.norm() * weight.norm(dim=1).mean() * strength


def swap_io_into(model, source):
    """Copy of model with source's embeddings, final norm, and head."""
    out = copy.deepcopy(model)
    out.set_input_embeddings(copy.deepcopy(source.get_input_embeddings()))
    out.model.norm = copy.deepcopy(source.model.norm)
    out.set_output_embeddings(copy.deepcopy(source.get_output_embeddings()))
    return out.eval()


def layer_moe_of(top_k=1, keep_io=True, **kwargs):
    base = tiny_rope(0, 1000000.0)
    math = tiny_rope(1, 10000.0)
    coder = tiny_rope(2, 1000000.0)
    refs = (copy.deepcopy(base), copy.deepcopy(math), copy.deepcopy(coder))
    model = build_skill_moe(base, [math, coder], top_k=top_k, whole_layer=True, keep_io=keep_io,
                            expert_names=["math", "coder"], expert_ids=["math-id", "coder-id"], **kwargs)
    return model, refs


def test_force_mlp_expert_equals_expert_mlps_and_clear_restores():
    model, reference = moe_of(1)
    base = copy.deepcopy(reference)
    expert = tiny_qwen2(1)
    model = build_skill_moe(base, [expert, tiny_qwen2(2)], top_k=1, expert_names=["math", "coder"])
    teacher = teacher_with_expert_mlps(reference, expert)
    ids = torch.randint(0, 300, (2, 9))
    torch.manual_seed(3)
    with torch.no_grad():
        for moe in moe_layers(model):
            moe.router.weight.normal_(0, 3.0)
            moe.router_bias.zero_()
    routed = logits(model, ids)
    assert force_route(model, "math") == 1
    assert (logits(model, ids) - logits(teacher, ids)).abs().max().item() < 1e-5
    clear_force(model)
    assert torch.equal(logits(model, ids), routed)
    # A layer range forces only those layers.
    force_route(model, 0, layers=range(2, 3))
    forced = [moe.force for moe in moe_layers(model)]
    assert forced == [None, None, 0]
    clear_force(model)
    assert [moe.force for moe in moe_layers(model)] == [None, None, None]
    with pytest.raises(ValueError):
        force_route(model, "nobody")


def test_whole_layer_force_equals_specialist_stack():
    model, (base_ref, math_ref, coder_ref) = layer_moe_of()
    ids = torch.randint(0, 300, (2, 11))
    assert isinstance(moe_layers(model)[0], LayerMoE)
    force_route(model, "math")
    expected = logits(swap_io_into(math_ref, base_ref), ids)
    assert (logits(model, ids) - expected).abs().max().item() < 1e-5
    force_route(model, "coder")
    expected = logits(swap_io_into(coder_ref, base_ref), ids)
    assert (logits(model, ids) - expected).abs().max().item() < 1e-5
    force_route(model, 0)
    assert (logits(model, ids) - logits(base_ref, ids)).abs().max().item() < 1e-6


def test_whole_layer_force_with_own_io_equals_specialist():
    model, (base_ref, math_ref, coder_ref) = layer_moe_of()
    ids = torch.randint(0, 300, (2, 11))
    force_route(model, "math", own_io=True)
    assert (logits(model, ids) - logits(math_ref, ids)).abs().max().item() < 1e-5
    clear_force(model)
    force_route(model, "coder", own_io=True)
    assert (logits(model, ids) - logits(coder_ref, ids)).abs().max().item() < 1e-5
    clear_force(model)
    # Normal routing is back, with the base's own embeddings and head.
    force_route(model, 0)
    assert (logits(model, ids) - logits(base_ref, ids)).abs().max().item() < 1e-6
    # Without keep_io there is nothing to swap in.
    plain, _ = layer_moe_of(keep_io=False)
    with pytest.raises(ValueError, match="keep_io"):
        force_route(plain, "math", own_io=True)


def test_whole_layer_params_and_fresh_equals_base():
    model, (base_ref, _, _) = layer_moe_of()
    ids = torch.randint(0, 300, (2, 9))
    assert (logits(model, ids) - logits(base_ref, ids)).abs().max().item() < 1e-6
    counts = count_params(model)
    layer_params = param_count(base_ref.model.layers[0])
    # Per token: the base model plus the router (one layer per slot, top-1).
    assert counts["active"] == param_count(base_ref) + counts["router"]
    assert counts["experts"] == 2 * 3 * layer_params
    trainable = []
    for param in model.parameters():
        if param.requires_grad:
            trainable.append(param)
    assert len(trainable) == len(router_parameters(model))


@pytest.mark.parametrize("granularity", ["mlp", "layer"])
def test_ridge_init_routes_planted_domains(granularity):
    generator = torch.Generator().manual_seed(0)
    texts = planted_texts(generator)
    base = plant_domains(tiny_qwen2(0))
    experts = [tiny_qwen2(1), tiny_qwen2(2)]
    model = build_skill_moe(base, experts, top_k=1, router_init="uniform", whole_layer=(granularity == "layer"),
                            expert_names=["math", "coder"])
    report = init_router_ridge(model, FakeTokenizer(), texts, lam=0.01, mode="token")
    assert report["layers"] == 3
    for moe in moe_layers(model):
        assert moe.base_margin == 0.0
        assert float(moe.router_bias[0].detach()) < 16.0
    held_out = planted_texts(generator, per_class=1, length=64)

    def shares(name, text_set):
        ids = torch.tensor([[int(part) for part in text_set[name][0].split()]])
        track_routing(model)
        logits(model, ids)
        out = routing_shares(model)["mean"]
        track_routing(model, on=False)
        return out

    math = shares("math", held_out)
    general = shares("base", held_out)
    code = shares("coder", held_out)
    assert math[1] > 0.8
    assert general[0] > 0.8
    assert code[2] > 0.8


def test_ridge_base_margin_moves_the_base_logit():
    generator = torch.Generator().manual_seed(1)
    texts = planted_texts(generator)
    model = build_skill_moe(plant_domains(tiny_qwen2(0)), [tiny_qwen2(1), tiny_qwen2(2)], top_k=1,
                            router_init="uniform", expert_names=["math", "coder"])
    init_router_ridge(model, FakeTokenizer(), texts, base_margin=50.0)
    assert model.wsp_skill_moe["base_margin"] == 50.0
    track_routing(model)
    logits(model, torch.randint(0, 100, (2, 12)))
    assert routing_shares(model)["mean"][0] > 0.99


def test_sequence_routing_applies_one_choice_to_all_layers():
    generator = torch.Generator().manual_seed(2)
    texts = planted_texts(generator, per_class=60)
    model = build_skill_moe(plant_domains(tiny_qwen2(0)), [tiny_qwen2(1), tiny_qwen2(2)], top_k=1,
                            router_init="uniform", whole_layer=True, expert_names=["math", "coder"])
    init_router_ridge(model, FakeTokenizer(), texts, mode="mean")
    set_sequence_routing(model, 1)
    assert model.wsp_skill_moe["seq_layer"] == 1
    # One batch: two math rows then two coder rows. The first layer runs the base.
    ids = torch.cat([torch.randint(0, 100, (2, 16), generator=generator),
                     torch.randint(200, 300, (2, 16), generator=generator)])
    track_routing(model)
    logits(model, ids)
    shares = routing_shares(model)["per_layer"]
    track_routing(model, on=False)
    assert shares[0] == [1.0, 0.0, 0.0]
    for layer in [1, 2]:
        assert shares[layer] == [0.0, 0.5, 0.5]
    # A pure math batch goes to math in every later layer, and the choice is per sequence.
    track_routing(model)
    logits(model, torch.randint(0, 100, (3, 16), generator=generator))
    shares = routing_shares(model)["per_layer"]
    assert shares[1] == [0.0, 1.0, 0.0]
    assert shares[2] == [0.0, 1.0, 0.0]
    # Within one sequence a token of a foreign domain does not change the choice.
    mixed = torch.randint(0, 100, (1, 16), generator=generator)
    mixed[0, 8] = 250
    track_routing(model)
    logits(model, mixed)
    assert routing_shares(model)["per_layer"][2] == [0.0, 1.0, 0.0]
    clear_sequence_routing(model)
    assert model.wsp_skill_moe["seq_layer"] is None
    for moe in moe_layers(model):
        assert moe.seq is None


def test_sequence_routing_trains_the_router_with_a_straight_through_weight():
    model = build_skill_moe(tiny_qwen2(0), [tiny_qwen2(1), tiny_qwen2(2)], top_k=1, router_init="uniform",
                            whole_layer=True)
    set_sequence_routing(model, 1)
    ids = torch.randint(0, 300, (2, 8))
    plain = None
    with torch.no_grad():
        plain = model(ids).logits
    out = model(input_ids=ids, labels=ids)
    out.loss.backward()
    grad = model.model.layers[1].router.weight.grad
    assert grad is not None and grad.abs().sum() > 0
    assert (model(ids).logits - plain).abs().max().item() < 1e-5


def test_whole_layer_cached_generation_matches_uncached():
    for top_k in [1, 2]:
        model, _ = layer_moe_of(top_k=top_k)
        torch.manual_seed(4)
        with torch.no_grad():
            for moe in moe_layers(model):
                moe.router.weight.normal_(0, 2.0)
                moe.router_bias.zero_()
        ids = torch.randint(0, 300, (1, 6))
        cached = model.generate(ids, max_new_tokens=8, do_sample=False, use_cache=True)
        plain = model.generate(ids, max_new_tokens=8, do_sample=False, use_cache=False)
        assert torch.equal(cached, plain)
        with torch.no_grad():
            full = model(cached).logits
            first = model(cached[:, :10], use_cache=True)
            step = model(cached[:, 10:11], past_key_values=first.past_key_values, use_cache=True).logits
        assert (full[:, 10] - step[:, 0]).abs().max().item() < 1e-5
        # The routing really used more than one expert.
        track_routing(model)
        logits(model, torch.randint(0, 300, (2, 12)))
        used = 0
        for share in routing_shares(model)["mean"]:
            if share > 0.05:
                used += 1
        assert used > 1
        track_routing(model, on=False)


def test_whole_layer_save_and_load_keeps_options(tmp_path):
    paths = []
    for name, model in [("base", tiny_rope(0, 1000000.0)), ("math", tiny_rope(1, 10000.0)),
                        ("coder", tiny_rope(2, 1000000.0))]:
        path = str(tmp_path / name)
        model.save_pretrained(path)
        paths.append(path)
    loaded_models = []
    for path in paths:
        loaded_models.append(transformers.AutoModelForCausalLM.from_pretrained(path, dtype=torch.float32).eval())
    reference_math = copy.deepcopy(loaded_models[1])
    model = build_skill_moe(loaded_models[0], loaded_models[1:], top_k=1, whole_layer=True, keep_io=True,
                            base_margin=1.5, seq_layer=1, expert_names=["math", "coder"])
    torch.manual_seed(9)
    with torch.no_grad():
        for moe in moe_layers(model):
            moe.router.weight.normal_(0, 1.0)
            moe.router_bias.normal_(0, 1.0)
    ids = torch.randint(0, 300, (2, 8))
    expected = logits(model, ids)
    out_dir = str(tmp_path / "moe")
    manifest = save_skill_moe(model, out_dir)
    assert manifest["granularity"] == "layer"
    assert manifest["base_margin"] == 1.5
    assert manifest["seq_layer"] == 1
    loaded, _ = load_skill_moe(out_dir, dtype=torch.float32)
    info = loaded.wsp_skill_moe
    assert info["granularity"] == "layer" and info["keep_io"] is True
    assert info["names"] == ["base", "math", "coder"]
    assert info["seq_layer"] == 1
    assert moe_layers(loaded)[0].base_margin == 1.5
    assert moe_layers(loaded)[0].seq is not None
    assert torch.allclose(logits(loaded, ids), expected, atol=1e-6)
    force_route(loaded, "math", own_io=True)
    assert (logits(loaded, ids) - logits(reference_math, ids)).abs().max().item() < 1e-5
