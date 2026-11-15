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
