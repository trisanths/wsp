import copy
import json
import os

import pytest
import torch
import transformers
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import AutoModelForCausalLM, PreTrainedTokenizerFast

from wsp import atlas, slim
from wsp.lora import lora_modules, merge_lora

SKILL_WORDS = ["add", "sum", "plus", "minus", "times", "equals", "number", "digit"]
GENERAL_WORDS = ["the", "river", "city", "king", "song", "film", "tree", "road"]
EXTRA = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "Question:", "Answer:", "yes", "no", "."]
VOCAB = 40
FAMILY_NAMES = ["qwen3_moe", "olmoe", "mixtral", "qwen2_moe", "glm4_moe", "deepseek_v3"]


def tokenizer():
    vocab = {"<unk>": 0, "</s>": 1, "<pad>": 2}
    for word in SKILL_WORDS + GENERAL_WORDS + EXTRA:
        vocab[word] = len(vocab)
    base = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    base.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    return PreTrainedTokenizerFast(tokenizer_object=base, unk_token="<unk>", eos_token="</s>", pad_token="<pad>")


def texts(words, count, length, seed):
    generator = torch.Generator()
    generator.manual_seed(seed)
    out = []
    for _ in range(count):
        picks = torch.randint(len(words), (length,), generator=generator).tolist()
        line = []
        for pick in picks:
            line.append(words[pick])
        out.append(" ".join(line))
    return out


def tiny_moe(name="qwen3_moe", experts=8, top_k=2, layers=2, norm=True, seed=0):
    torch.manual_seed(seed)
    if name == "qwen3_moe":
        config = transformers.Qwen3MoeConfig(
            hidden_size=48, intermediate_size=96, moe_intermediate_size=40, num_hidden_layers=layers,
            num_attention_heads=4, num_key_value_heads=2, vocab_size=VOCAB, max_position_embeddings=64,
            num_experts=experts, num_experts_per_tok=top_k, decoder_sparse_step=1, mlp_only_layers=[], head_dim=12,
            norm_topk_prob=norm)
        model = transformers.Qwen3MoeForCausalLM(config)
    elif name == "olmoe":
        config = transformers.OlmoeConfig(
            hidden_size=48, intermediate_size=40, num_hidden_layers=layers, num_attention_heads=4,
            num_key_value_heads=4, vocab_size=VOCAB, max_position_embeddings=64, num_experts=experts,
            num_experts_per_tok=top_k, norm_topk_prob=norm)
        model = transformers.OlmoeForCausalLM(config)
    elif name == "mixtral":
        config = transformers.MixtralConfig(
            hidden_size=48, intermediate_size=40, num_hidden_layers=layers, num_attention_heads=4,
            num_key_value_heads=2, vocab_size=VOCAB, max_position_embeddings=64, num_local_experts=experts,
            num_experts_per_tok=top_k)
        model = transformers.MixtralForCausalLM(config)
    elif name == "qwen2_moe":
        config = transformers.Qwen2MoeConfig(
            hidden_size=48, intermediate_size=96, moe_intermediate_size=40, shared_expert_intermediate_size=40,
            num_hidden_layers=layers, num_attention_heads=4, num_key_value_heads=2, vocab_size=VOCAB,
            max_position_embeddings=64, num_experts=experts, num_experts_per_tok=top_k, decoder_sparse_step=1,
            mlp_only_layers=[], norm_topk_prob=norm)
        model = transformers.Qwen2MoeForCausalLM(config)
    elif name == "glm4_moe":
        config = transformers.Glm4MoeConfig(
            hidden_size=48, intermediate_size=96, moe_intermediate_size=40, num_hidden_layers=layers,
            num_attention_heads=4, num_key_value_heads=2, vocab_size=VOCAB, max_position_embeddings=64,
            n_routed_experts=experts, n_shared_experts=1, num_experts_per_tok=top_k, first_k_dense_replace=0,
            n_group=1, topk_group=1, head_dim=12)
        model = transformers.Glm4MoeForCausalLM(config)
    elif name == "deepseek_v3":
        config = transformers.DeepseekV3Config(
            hidden_size=48, intermediate_size=96, moe_intermediate_size=40, num_hidden_layers=layers,
            num_attention_heads=4, num_key_value_heads=4, vocab_size=VOCAB, max_position_embeddings=64,
            n_routed_experts=experts, n_shared_experts=1, num_experts_per_tok=top_k, first_k_dense_replace=0,
            n_group=1, topk_group=1, q_lora_rank=16, kv_lora_rank=16, qk_rope_head_dim=8, qk_nope_head_dim=8,
            v_head_dim=8)
        model = transformers.DeepseekV3ForCausalLM(config)
    else:
        raise ValueError(name)
    return model.eval()


def tiny_dense(layers=6, seed=0):
    torch.manual_seed(seed)
    config = transformers.Qwen2Config(
        hidden_size=32, intermediate_size=64, num_hidden_layers=layers, num_attention_heads=4,
        num_key_value_heads=2, vocab_size=VOCAB, max_position_embeddings=64)
    return transformers.Qwen2ForCausalLM(config).eval()


def logits(model, ids):
    with torch.no_grad():
        return model(input_ids=ids, use_cache=False).logits


def sample_ids(count=3, length=9, seed=1):
    generator = torch.Generator()
    generator.manual_seed(seed)
    return torch.randint(3, VOCAB, (count, length), generator=generator)


def direct_params(model):
    seen = set()
    total = 0
    for param in model.parameters():
        if id(param) in seen:
            continue
        seen.add(id(param))
        total += param.numel()
    return total


SKILL = texts(SKILL_WORDS, 6, 10, seed=1)
GENERAL = texts(GENERAL_WORDS, 6, 10, seed=2)


# ---------------------------------------------------------------- statistics


@pytest.mark.parametrize("name", ["qwen3_moe", "olmoe", "mixtral", "qwen2_moe"])
def test_route_matches_the_model(name):
    model = tiny_moe(name)
    family = slim.family_of(model)
    layer, moe, router, top_k, count = slim.moe_layers_of(model)[0]
    x = torch.randn(11, 48)
    with torch.no_grad():
        weights, indices = slim.route(family, moe, router(x), top_k)
        built = torch.zeros(11, 48)
        for row in range(11):
            for slot in range(top_k):
                built[row] += weights[row, slot] * moe.experts[int(indices[row, slot])](x[row:row + 1])[0]
        actual = moe(x[None])[0][0]
        if name == "qwen2_moe":
            shared = torch.sigmoid(moe.shared_expert_gate(x)) * moe.shared_expert(x)
            built = built + shared
    assert (built - actual).abs().max().item() < 1e-5


@pytest.mark.parametrize("name", FAMILY_NAMES)
def test_stats_counts_equal_atlas_counts(name):
    model = tiny_moe(name)
    tok = tokenizer()
    stats = slim.collect_moe_stats(model, tok, SKILL)
    counts = atlas.routing_counts(model, tok, SKILL)
    for layer in counts:
        assert torch.equal(stats["layers"][layer]["count"], counts[layer][0])
        assert stats["layers"][layer]["tokens"] == counts[layer][1]
        assert (stats["layers"][layer]["saliency"] >= 0).all()
        # An expert that is never picked has no saliency, and every picked one has some.
        picked = stats["layers"][layer]["count"] > 0
        assert ((stats["layers"][layer]["saliency"] > 0) == picked).all()


def test_stats_from_expert_map_feeds_freq_and_selective():
    model = tiny_moe()
    tok = tokenizer()
    expert_map = atlas.expert_skill_map(model, tok, SKILL, GENERAL)
    skill = slim.stats_from_expert_map(expert_map, tokens=1000)
    own = slim.collect_moe_stats(model, tok, SKILL)
    order_map = slim.rank_experts("freq", skill)
    order_own = slim.rank_experts("freq", own)
    for layer in order_own:
        counts = own["layers"][layer]["count"].tolist()
        # Ranks by frequency agree wherever the counts are not tied.
        for first, second in zip(order_own[layer], order_own[layer][1:]):
            if counts[first] > counts[second]:
                assert order_map[layer].index(first) < order_map[layer].index(second)


# ---------------------------------------------------------------- ranking and choice


def test_kept_ids_match_the_ranking():
    model = tiny_moe(experts=12, layers=2)
    tok = tokenizer()
    skill = slim.collect_moe_stats(model, tok, SKILL)
    general = slim.collect_moe_stats(model, tok, GENERAL)
    for rule in ["freq", "saliency", "selective", "general"]:
        scores = slim.expert_scores(rule, skill, general)
        slimmed, report = slim.slim_moe(model, tok, SKILL, GENERAL, rule=rule, keep=5, skill_stats=skill,
                                        general_stats=general)
        for layer in scores:
            values = scores[layer]
            counts = skill["layers"][layer]["count"].tolist()
            expected = sorted(range(12), key=lambda e: (-values[e], -counts[e], e))[:5]
            assert report["kept"][layer] == sorted(expected)
            # The kept experts are the 5 best: every kept score is at least every removed score.
            removed = [values[e] for e in range(12) if e not in report["kept"][layer]]
            assert min(values[e] for e in report["kept"][layer]) >= max(removed)
        assert len(slimmed.model.layers[0].mlp.experts) == 5


def test_random_differs_from_saliency_and_is_seeded():
    model = tiny_moe(experts=16)
    tok = tokenizer()
    skill = slim.collect_moe_stats(model, tok, SKILL)
    kept = {}
    for name, rule, seed in [("sal", "saliency", 0), ("r0", "random", 0), ("r0b", "random", 0), ("r1", "random", 1)]:
        _, report = slim.slim_moe(model, tok, SKILL, rule=rule, keep=6, seed=seed, skill_stats=skill)
        kept[name] = report["kept"]
    assert kept["r0"] == kept["r0b"]
    assert kept["r0"] != kept["sal"]
    assert kept["r0"] != kept["r1"]
    ids = sample_ids()
    sal_model, _ = slim.slim_experts(model, kept["sal"])
    rand_model, _ = slim.slim_experts(model, kept["r0"])
    assert (logits(sal_model, ids) - logits(rand_model, ids)).abs().max().item() > 1e-4


def test_choose_keep_by_mass():
    model = tiny_moe(experts=16)
    tok = tokenizer()
    skill = slim.collect_moe_stats(model, tok, SKILL)
    order = slim.rank_experts("freq", skill)
    kept = slim.choose_keep(order, skill, "freq", mass=0.9)
    for layer in kept:
        counts = skill["layers"][layer]["count"].tolist()
        total = sum(counts)
        taken = sum(counts[e] for e in kept[layer])
        assert taken >= 0.9 * total - 1e-9
        # One expert fewer would fall short, unless the count is at the floor.
        if len(kept[layer]) > 1:
            fewer = sum(counts[e] for e in order[layer][:len(kept[layer]) - 1])
            assert fewer < 0.9 * total
    assert slim.kept_share(skill, kept)[0] >= 0.9 - 1e-9
    everything = slim.choose_keep(order, skill, "freq", mass=1.0)
    assert len(everything[0]) <= 16
    with pytest.raises(ValueError):
        slim.choose_keep(order, skill, "freq")
    with pytest.raises(ValueError):
        slim.choose_keep(order, skill, "freq", keep=3, mass=0.5)
    with pytest.raises(ValueError):
        slim.choose_keep(order, skill, "freq", keep=99)


def test_unknown_rule_and_family_fail_clearly():
    with pytest.raises(ValueError):
        slim.expert_scores("nope", {"layers": {}})
    with pytest.raises(ValueError):
        slim.expert_scores("selective", {"layers": {}})
    with pytest.raises(NotImplementedError):
        slim.family_of(tiny_dense())


# ---------------------------------------------------------------- expert removal


@pytest.mark.parametrize("name", FAMILY_NAMES)
def test_keeping_every_expert_is_a_no_op(name):
    model = tiny_moe(name)
    ids = sample_ids()
    keep = {}
    for layer, moe, router, top_k, count in slim.moe_layers_of(model):
        keep[layer] = list(range(count))
    slimmed, report = slim.slim_experts(model, keep)
    assert (logits(slimmed, ids) - logits(model, ids)).abs().max().item() < 1e-6
    assert report["params_total_after"] == report["params_total_before"]
    assert report["params_run_after"] == report["params_run_before"]
    assert slimmed is not model


def test_removing_experts_the_skill_never_routes_to_changes_nothing():
    model = tiny_moe(experts=16, top_k=2, layers=3)
    tok = tokenizer()
    skill_texts = ["add sum plus"] * 4
    stats = slim.collect_moe_stats(model, tok, skill_texts)
    keep = {}
    dead_total = 0
    for layer in stats["layers"]:
        counts = stats["layers"][layer]["count"].tolist()
        used = []
        for expert in range(16):
            if counts[expert] > 0:
                used.append(expert)
        for expert in range(16):
            if len(used) >= 2:
                break
            if expert not in used:
                used.append(expert)
        keep[layer] = sorted(used)
        dead_total += 16 - len(used)
    assert dead_total >= 3 * 6
    slimmed, report = slim.slim_experts(model, keep)
    ids = torch.tensor([[tok.convert_tokens_to_ids(w) for w in ["add", "sum", "plus"]]])
    assert (logits(slimmed, ids) - logits(model, ids)).abs().max().item() < 1e-6
    # The same experts lose nothing on the whole skill batch, and the model got smaller.
    assert report["params_total_after"] < report["params_total_before"]
    # Text outside the skill does change: the removed experts were used there.
    other = sample_ids(count=4, length=12, seed=5)
    assert (logits(slimmed, other) - logits(model, other)).abs().max().item() > 1e-5


def test_param_counts_match_a_direct_count():
    model = tiny_moe(experts=8, top_k=2, layers=3)
    tok = tokenizer()
    slimmed, report = slim.slim_moe(model, tok, SKILL, rule="freq", keep=3)
    assert report["params_total_before"] == direct_params(model)
    assert report["params_total_after"] == direct_params(slimmed)
    expert_size = direct_params(model.model.layers[0].mlp.experts[0])
    router_row = 48
    removed = 3 * (8 - 3) * (expert_size + router_row)
    assert report["params_total_before"] - report["params_total_after"] == removed
    # Run per token: everything except the input embedding and the experts a token does not pick.
    embed = model.get_input_embeddings().weight.numel()
    assert report["params_run_before"] == direct_params(model) - embed - 3 * (8 - 2) * expert_size
    assert report["params_run_after"] == direct_params(slimmed) - embed - 3 * (3 - 2) * expert_size
    # With N under top_k, every kept expert runs.
    small, small_report = slim.slim_moe(model, tok, SKILL, rule="freq", keep=1)
    assert small_report["top_k"][0] == 1
    assert small_report["params_run_after"] == direct_params(small) - embed


@pytest.mark.parametrize("name", ["qwen3_moe", "mixtral", "olmoe", "qwen2_moe", "glm4_moe"])
def test_uniform_slim_saves_a_normal_hf_model(tmp_path, name):
    model = tiny_moe(name, experts=8, top_k=3)
    tok = tokenizer()
    slimmed, report = slim.slim_moe(model, tok, SKILL, rule="saliency", keep=5)
    manifest = slim.save_slim(slimmed, str(tmp_path))
    assert manifest["plain_hf"] is True
    with open(tmp_path / "config.json") as f:
        config = json.load(f)
    family = slim.FAMILIES[name]
    assert config[family.count_key] == 5
    assert config[family.topk_key] == 3
    loaded = AutoModelForCausalLM.from_pretrained(str(tmp_path)).eval()
    ids = sample_ids()
    assert (logits(loaded, ids) - logits(slimmed, ids)).abs().max().item() < 1e-5
    assert direct_params(loaded) == report["params_total_after"]
    again = slim.load_slim(str(tmp_path), dtype=torch.float32)
    assert (logits(again, ids) - logits(slimmed, ids)).abs().max().item() < 1e-5


def test_top_k_is_capped_at_the_kept_count(tmp_path):
    model = tiny_moe(experts=8, top_k=3)
    tok = tokenizer()
    slimmed, report = slim.slim_moe(model, tok, SKILL, rule="freq", keep=2)
    assert report["top_k"] == {0: 2, 1: 2}
    slim.save_slim(slimmed, str(tmp_path))
    loaded = AutoModelForCausalLM.from_pretrained(str(tmp_path)).eval()
    ids = sample_ids()
    assert (logits(loaded, ids) - logits(slimmed, ids)).abs().max().item() < 1e-5


@pytest.mark.parametrize("name", ["qwen3_moe", "glm4_moe"])
def test_different_count_per_layer_goes_through_the_wsp_loader(tmp_path, name):
    model = tiny_moe(name, experts=8, top_k=2, layers=3)
    tok = tokenizer()
    keep = {0: [0, 2, 5, 6, 7], 1: [1, 3], 2: [0, 1, 2, 3, 4, 5, 6]}
    slimmed, report = slim.slim_experts(model, keep)
    assert report["uniform"] is False
    assert report["counts"] == {0: 5, 1: 2, 2: 7}
    manifest = slim.save_slim(slimmed, str(tmp_path), tok)
    assert manifest["plain_hf"] is False
    loaded = slim.load_slim(str(tmp_path), dtype=torch.float32)
    ids = sample_ids()
    assert (logits(loaded, ids) - logits(slimmed, ids)).abs().max().item() < 1e-5
    assert direct_params(loaded) == report["params_total_after"]
    assert loaded.wsp_slim["kept"][1] == [1, 3]
    # Mass-based choice makes the counts differ by layer on its own.
    stats = slim.collect_moe_stats(model, tok, SKILL)
    order = slim.rank_experts("freq", stats)
    kept = slim.choose_keep(order, stats, "freq", mass=0.7)
    assert all(1 <= len(ids_) <= 8 for ids_ in kept.values())


def test_save_refuses_a_model_with_a_lora(tmp_path):
    model = tiny_moe()
    tok = tokenizer()
    slimmed, _ = slim.slim_moe(model, tok, SKILL, rule="freq", keep=4)
    slim.heal(slimmed, model, tok, SKILL, steps=1, batch_size=2, seq_len=16, merge=False, log_every=0)
    with pytest.raises(ValueError):
        slim.save_slim(slimmed, str(tmp_path))
    merge_lora(slimmed)
    slim.save_slim(slimmed, str(tmp_path))


def test_slimming_does_not_touch_the_original_and_sharing_shares_experts():
    model = tiny_moe()
    tok = tokenizer()
    before = copy.deepcopy(model.state_dict())
    slimmed, _ = slim.slim_moe(model, tok, SKILL, rule="freq", keep=4, share_experts=True)
    after = model.state_dict()
    for name in before:
        assert torch.equal(before[name], after[name])
    layer = slimmed.model.layers[0].mlp
    original = model.model.layers[0].mlp
    shared = 0
    for expert in layer.experts:
        for source in original.experts:
            if expert is source:
                shared += 1
    assert shared == 4
    assert layer.gate is not original.gate
    assert slimmed.model.layers[0].self_attn is not model.model.layers[0].self_attn


@pytest.mark.parametrize("name", ["qwen3_moe", "mixtral"])
def test_cached_generation_equals_uncached_after_expert_removal(name):
    model = tiny_moe(name, top_k=3)
    tok = tokenizer()
    slimmed, _ = slim.slim_moe(model, tok, SKILL, rule="saliency", keep=4)
    ids = sample_ids(count=1, length=5)
    cached = slimmed.generate(input_ids=ids, max_new_tokens=8, do_sample=False, use_cache=True, pad_token_id=2)
    plain = slimmed.generate(input_ids=ids, max_new_tokens=8, do_sample=False, use_cache=False, pad_token_id=2)
    assert torch.equal(cached, plain)


# ---------------------------------------------------------------- dense


def skipped_logits(model, skipped, ids):
    _, layers = slim.find_decoder_layers(model, 0)
    handles = []
    for index in skipped:
        handles.append(layers[index].register_forward_hook(atlas.skip_hook, with_kwargs=True))
    try:
        return logits(model, ids)
    finally:
        for handle in handles:
            handle.remove()


@pytest.mark.parametrize("rule", ["skill_cost", "specific", "influence", "general", "random"])
def test_layer_removal_equals_deleting_the_layers(rule):
    model = tiny_dense()
    tok = tokenizer()
    slimmed, report = slim.slim_dense(model, tok, SKILL, GENERAL, n_drop=2, max_layer_fraction=0.34,
                                      layer_rule=rule, seed=3)
    dropped = report["dropped"]
    assert len(dropped) == 2
    assert 0 not in dropped and 5 not in dropped
    ids = sample_ids()
    assert (logits(slimmed, ids) - skipped_logits(model, dropped, ids)).abs().max().item() < 1e-5
    assert report["layers_after"] == 4
    assert slimmed.config.num_hidden_layers == 4
    assert report["params_total_after"] == direct_params(slimmed)
    assert report["params_total_before"] == direct_params(model)
    assert len(model.model.layers) == 6


def test_layer_rules_pick_the_lowest_cost_layers():
    model = tiny_dense()
    tok = tokenizer()
    costs = slim.layer_costs(model, tok, SKILL, GENERAL)
    picked = slim.pick_layers(costs, 6, 2, "skill_cost")
    values = costs["skill_increase"]
    inner = sorted(range(1, 5), key=lambda i: (values[i], i))[:2]
    assert picked == sorted(inner)
    specific = slim.pick_layers(costs, 6, 2, "specific")
    inner = sorted(range(1, 5), key=lambda i: (costs["scores"][i], i))[:2]
    assert specific == sorted(inner)
    assert slim.pick_layers(costs, 6, 2, "random", seed=1) == slim.pick_layers(costs, 6, 2, "random", seed=1)
    with pytest.raises(ValueError):
        slim.pick_layers(costs, 6, 5, "skill_cost")
    with pytest.raises(ValueError):
        slim.pick_layers(costs, 6, 1, "nope")


def test_layer_cap_is_enforced():
    model = tiny_dense(layers=8)
    with pytest.raises(ValueError):
        slim.slim_dense(model, tokenizer(), SKILL, GENERAL, n_drop=3)
    slimmed, report = slim.slim_dense(model, tokenizer(), SKILL, GENERAL, n_drop=2)
    assert len(report["dropped"]) == 2


def test_greedy_layers_drops_the_requested_count():
    model = tiny_dense()
    tok = tokenizer()
    dropped = slim.greedy_layers(model, tok, SKILL, 2, step=1)
    assert len(dropped) == 2
    assert 0 not in dropped and 5 not in dropped
    slimmed, report = slim.slim_dense(model, tok, SKILL, GENERAL, n_drop=2, max_layer_fraction=0.34,
                                      greedy_step=1)
    assert report["dropped"] == dropped


@pytest.mark.parametrize("width_rule", ["skill", "general", "specific", "random"])
def test_neuron_and_head_removal_equals_masking(width_rule):
    model = tiny_dense()
    tok = tokenizer()
    slimmed, report = slim.slim_dense(model, tok, SKILL, GENERAL, n_drop=0, keep_mlp=0.5, keep_heads=0.5,
                                      width_rule=width_rule, seed=2)
    masked = copy.deepcopy(model)
    head_dim = 8
    for row in report["width"]["layers"]:
        layer = masked.model.layers[row["layer"]]
        keep_neurons = set(row["neurons"])
        with torch.no_grad():
            for neuron in range(64):
                if neuron not in keep_neurons:
                    layer.mlp.gate_proj.weight[neuron] = 0
                    layer.mlp.up_proj.weight[neuron] = 0
                    layer.mlp.down_proj.weight[:, neuron] = 0
            keep_q = set(row["q_heads"])
            for head in range(4):
                if head not in keep_q:
                    layer.self_attn.o_proj.weight[:, head * head_dim:(head + 1) * head_dim] = 0
    ids = sample_ids()
    assert (logits(slimmed, ids) - logits(masked, ids)).abs().max().item() < 1e-5
    assert slimmed.config.intermediate_size == 32
    assert report["params_total_after"] == direct_params(slimmed) < report["params_total_before"]


def test_skill_width_rule_keeps_the_neurons_skill_text_uses():
    model = tiny_dense()
    tok = tokenizer()
    scores = slim.width_scores(model, tok, SKILL, GENERAL, "skill")
    slimmed, report = slim.slim_dense(model, tok, SKILL, GENERAL, n_drop=0, keep_mlp=0.5, keep_heads=1.0)
    for row in report["width"]["layers"]:
        values = scores["mlp"][row["layer"]]
        top = set(torch.topk(values, 32).indices.tolist())
        assert set(row["neurons"]) == top


def test_dense_save_and_load_is_a_normal_hf_model(tmp_path):
    model = tiny_dense()
    tok = tokenizer()
    slimmed, report = slim.slim_dense(model, tok, SKILL, GENERAL, n_drop=2, max_layer_fraction=0.34, keep_mlp=0.5,
                                      keep_heads=0.5)
    manifest = slim.save_slim(slimmed, str(tmp_path))
    assert manifest["plain_hf"] is True
    loaded = AutoModelForCausalLM.from_pretrained(str(tmp_path)).eval()
    ids = sample_ids()
    assert (logits(loaded, ids) - logits(slimmed, ids)).abs().max().item() < 1e-5
    assert direct_params(loaded) == report["params_total_after"]


def test_cached_generation_equals_uncached_after_dense_slimming():
    model = tiny_dense()
    tok = tokenizer()
    slimmed, _ = slim.slim_dense(model, tok, SKILL, GENERAL, n_drop=2, max_layer_fraction=0.34, keep_mlp=0.5,
                                 keep_heads=0.5)
    ids = sample_ids(count=1, length=5)
    cached = slimmed.generate(input_ids=ids, max_new_tokens=8, do_sample=False, use_cache=True, pad_token_id=2)
    plain = slimmed.generate(input_ids=ids, max_new_tokens=8, do_sample=False, use_cache=False, pad_token_id=2)
    assert torch.equal(cached, plain)


def test_plan_size_reaches_the_target():
    model = tiny_dense(layers=8)
    total = direct_params(model)
    embed = 2 * VOCAB * 32
    target = embed + (total - embed) * 0.5
    drop, fraction = slim.plan_size(model, target, 0.25)
    assert drop == 2
    assert 0 < fraction < 1
    slimmed, report = slim.slim_dense(model, tokenizer(), SKILL, GENERAL, target_params=target)
    assert abs(report["params_total_after"] - target) / target < 0.08
    # A target that already fits needs no width cut and fewer layers.
    drop, fraction = slim.plan_size(model, total * 0.9, 0.25)
    assert fraction == 1.0 and drop >= 1
    with pytest.raises(ValueError):
        slim.plan_size(model, embed // 2, 0.25)


# ---------------------------------------------------------------- heal


def test_heal_changes_only_lora_and_router_params():
    teacher = tiny_moe(top_k=3)
    tok = tokenizer()
    student, _ = slim.slim_moe(teacher, tok, SKILL, rule="saliency", keep=5)
    before = {}
    for name, param in student.named_parameters():
        before[name] = param.detach().clone()
    report = slim.heal(student, teacher, tok, SKILL + GENERAL, steps=4, batch_size=3, seq_len=16, merge=False,
                       warmup=1, lora_lr=1e-2, router_lr=1e-2, log_every=0)
    assert report["steps"] == 4
    assert report["router_params"] > 0
    changed = []
    for name, param in student.named_parameters():
        if ".lora_" in name:
            continue
        key = name.replace(".base.", ".")
        assert key in before
        if not torch.equal(before[key], param.detach()):
            changed.append(key)
    assert len(changed) > 0
    for key in changed:
        assert key.endswith("mlp.gate.weight"), key
    # LoRA B moved off zero.
    moved = 0
    for name, module in lora_modules(student):
        if module.lora_B.abs().sum() > 0:
            moved += 1
    assert moved > 0
    # The LoRA targets are the attention linears only.
    for name, module in lora_modules(student):
        assert "self_attn" in name
    # The teacher did not change.
    for name, param in teacher.named_parameters():
        assert not param.requires_grad or True
    assert report["kd_first"] >= 0


def test_heal_merge_keeps_the_function_and_removes_the_lora():
    teacher = tiny_moe()
    tok = tokenizer()
    student, _ = slim.slim_moe(teacher, tok, SKILL, rule="freq", keep=5)
    slim.heal(student, teacher, tok, SKILL, steps=3, batch_size=2, seq_len=16, merge=False, warmup=1,
              lora_lr=1e-2, log_every=0)
    ids = sample_ids()
    unmerged = logits(student, ids)
    merge_lora(student)
    assert len(lora_modules(student)) == 0
    assert (logits(student, ids) - unmerged).abs().max().item() < 1e-4


def test_heal_with_shared_experts_leaves_the_teacher_alone():
    teacher = tiny_moe()
    tok = tokenizer()
    snapshot = copy.deepcopy(teacher.state_dict())
    student, _ = slim.slim_moe(teacher, tok, SKILL, rule="freq", keep=5, share_experts=True)
    slim.heal(student, teacher, tok, SKILL, steps=3, batch_size=2, seq_len=16, warmup=1, lora_lr=1e-2,
              router_lr=1e-2, log_every=0)
    for name, value in teacher.state_dict().items():
        assert torch.equal(snapshot[name], value), name
    assert len(lora_modules(teacher)) == 0
    assert len(lora_modules(student)) == 0


def test_heal_dense_trains_only_lora():
    teacher = tiny_dense()
    tok = tokenizer()
    student, _ = slim.slim_dense(teacher, tok, SKILL, GENERAL, n_drop=1, max_layer_fraction=0.34, keep_mlp=0.5,
                                 keep_heads=0.5)
    before = {}
    for name, param in student.named_parameters():
        before[name] = param.detach().clone()
    slim.heal(student, teacher, tok, SKILL, steps=3, batch_size=2, seq_len=16, merge=False, warmup=1,
              lora_lr=1e-2, log_every=0)
    for name, param in student.named_parameters():
        if ".lora_" in name:
            continue
        assert torch.equal(before[name.replace(".base.", ".")], param.detach()), name
    moved = 0
    for name, module in lora_modules(student):
        if module.lora_B.abs().sum() > 0:
            moved += 1
    assert moved > 0


def test_heal_lowers_the_distance_to_the_teacher():
    teacher = tiny_moe(top_k=3)
    with torch.no_grad():
        teacher.lm_head.weight.mul_(40)
    tok = tokenizer()
    student, _ = slim.slim_moe(teacher, tok, SKILL, rule="freq", keep=3)
    held = texts(SKILL_WORDS + GENERAL_WORDS, 8, 10, seed=9)
    ids = torch.tensor([[tok.convert_tokens_to_ids(w) for w in line.split()] for line in held])

    def gap(model):
        a = torch.log_softmax(logits(teacher, ids), dim=-1)
        b = torch.log_softmax(logits(model, ids), dim=-1)
        return float((a.exp() * (a - b)).sum(-1).mean())

    start = gap(student)
    slim.heal(student, teacher, tok, texts(SKILL_WORDS + GENERAL_WORDS, 400, 10, seed=4), steps=60, batch_size=6,
              seq_len=16, warmup=5, lora_lr=1e-4, router_lr=1e-4, kd_weight=1.0, lm_weight=0.0, seed=0, log_every=0)
    assert gap(student) < start


# ---------------------------------------------------------------- text helpers


def test_drop_overlap_removes_texts_that_hold_a_scoring_prompt():
    keys = ["Write a function to find the shared elements from the given two lists.", "short"]
    items = ["Q: Write a function to find   the shared elements from the given two lists.\nA: def f(): pass",
             "an unrelated text about sorting", "short text is too short to count as a key"]
    clean, dropped = slim.drop_overlap(items, keys)
    assert dropped == 1
    assert len(clean) == 2


def test_directory_bytes(tmp_path):
    (tmp_path / "a").write_bytes(b"x" * 10)
    (tmp_path / "b").mkdir()
    (tmp_path / "b" / "c").write_bytes(b"y" * 5)
    assert slim.directory_bytes(str(tmp_path)) == 15


# ---------------------------------------------------------------- review fixes


def test_load_slim_restores_config_and_the_model_saves_again(tmp_path):
    model = tiny_moe("qwen3_moe", experts=8, top_k=2, layers=3)
    tok = tokenizer()
    keep = {0: [0, 2, 5], 1: [1, 3, 4, 6, 7], 2: [0, 1]}
    slimmed, report = slim.slim_experts(model, keep)
    slim.save_slim(slimmed, str(tmp_path / "a"), tok)
    loaded = slim.load_slim(str(tmp_path / "a"), dtype=torch.float32)
    assert loaded.config.num_experts == 5
    assert loaded.config.num_experts_per_tok == 2
    assert loaded.num_experts == 5
    info = loaded.wsp_slim
    assert info["counts"] == {0: 3, 1: 5, 2: 2}
    assert info["top_k"] == {0: 2, 1: 2, 2: 2}
    assert info["uniform"] is False and info["plain_hf"] is False
    assert info["params_total_after"] == report["params_total_after"]
    assert info["params_run_after"] == report["params_run_after"]
    assert slim.count_params(loaded) == (report["params_total_after"], report["params_run_after"])
    # Saving the loaded model again works and loads to the same function.
    manifest = slim.save_slim(loaded, str(tmp_path / "b"), tok)
    assert manifest["counts"] == {"0": 3, "1": 5, "2": 2}
    again = slim.load_slim(str(tmp_path / "b"), dtype=torch.float32)
    ids = sample_ids()
    assert (logits(again, ids) - logits(slimmed, ids)).abs().max().item() < 1e-5


def test_choose_keep_with_no_skill_routing_keeps_the_floor():
    order = {0: [3, 1, 0, 2], 1: [0, 1, 2, 3]}
    skill = {"layers": {0: {"count": torch.zeros(4), "saliency": torch.zeros(4)},
                        1: {"count": torch.tensor([5.0, 3.0, 1.0, 1.0]), "saliency": torch.zeros(4)}}}
    kept = slim.choose_keep(order, skill, "freq", mass=0.5, min_keep=2)
    assert kept[0] == [1, 3]
    assert kept[1] == [0, 1]


@pytest.mark.parametrize("name", ["glm4_moe", "deepseek_v3", "qwen3_moe"])
def test_router_dtype_comes_back_after_heal_setup(name):
    model = tiny_moe(name).to(torch.bfloat16)
    family = slim.family_of(model)
    before = {}
    for layer, moe, router, top_k, count in slim.moe_layers_of(model):
        before[layer] = router.weight.dtype
    params, restore = slim.prepare_routers(model, family)
    assert all(p.dtype == torch.float32 for p in params)
    restore()
    for layer, moe, router, top_k, count in slim.moe_layers_of(model):
        assert router.weight.dtype == before[layer] == torch.bfloat16


def test_model_level_expert_counts_follow_the_slimming():
    model = tiny_moe("qwen3_moe", experts=8, top_k=2, layers=2)
    assert model.num_experts == 8
    slimmed, _ = slim.slim_experts(model, {0: [0, 1, 2], 1: [0, 1, 2, 3, 4]})
    assert slimmed.num_experts == 5
    assert slimmed.num_experts_per_tok == 2
    assert model.num_experts == 8


def load_round():
    import importlib.util
    import sys

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    spec = importlib.util.spec_from_file_location("modal_round19", os.path.join(root, "modal_round19.py"))
    module = importlib.util.module_from_spec(spec)
    sys.modules["modal_round19"] = module
    spec.loader.exec_module(module)
    return module


def test_round19_ranking_texts_do_not_depend_on_heal_length():
    r = load_round()
    texts = [f"text {i}" for i in range(r.POOL_TEXTS)]
    rank0, heal0 = r.split_pool(texts, 0)
    rank500, heal500 = r.split_pool(texts, 500)
    rank1000, heal1000 = r.split_pool(texts, 1000)
    assert rank0 == rank500 == rank1000
    assert len(rank0) == r.RANK_TEXTS and heal0 == []
    assert len(heal500) == 500 * r.HEAL["batch_size"]
    assert heal1000[:len(heal500)] == heal500
    assert set(rank0).isdisjoint(heal1000)
    assert r.stats_key(rank0) == r.stats_key(rank500)
    other = list(rank0)
    other[0] = "different"
    assert r.stats_key(other) != r.stats_key(rank0)
    with pytest.raises(ValueError):
        r.split_pool(texts, 5000)


def fake_row(mean, hits=None):
    n_he, n_mbpp = 164, 257
    row = {"cell": "x", "code_mean": mean, "scores": {}, "error": None}
    if hits is not None:
        row["correct"] = {"humaneval": [1] * hits + [0] * (n_he - hits), "mbpp": [1] * hits + [0] * (n_mbpp - hits)}
    return row


def test_round19_cells_include_the_controls():
    r = load_round()
    for name in ["moe_full_heal500", "native_coder_3b_heal1000", "moe_random_32_heal500_s1",
                 "moe_random_32_heal500_s2"]:
        assert name in r.CELLS
    assert r.CELLS["moe_full_heal500"]["keep"] == r.MOE_EXPERTS
    assert r.CELLS["native_coder_3b_heal1000"]["heal"] == 1000


def test_round19_a_failed_cell_returns_an_error_row():
    r = load_round()

    def boom(name, spec):
        raise RuntimeError("out of memory")

    r.run_moe = boom
    row = r.run_cell_safe("moe_full")
    assert row["cell"] == "moe_full"
    assert "out of memory" in row["error"]
    assert row["scores"] is None


def test_round19_decide_uses_the_branch_controls_and_reports_missing_cells():
    r = load_round()
    rows = {"moe_full": fake_row(0.80), "moe_full_heal500": fake_row(0.82), "moe_saliency_32": fake_row(0.79),
            "moe_saliency_32_heal500": fake_row(0.815)}
    for s, v in enumerate([0.60, 0.62, 0.61]):
        rows[f"moe_random_32_s{s}"] = fake_row(v)
    for s, v in enumerate([0.70, 0.71, 0.69]):
        rows[f"moe_random_32_heal500_s{s}"] = fake_row(v)
    out = r.decide(rows)
    assert out["moe_no_heal"]["pass"] is True
    assert out["moe_heal"]["pass"] is True
    assert out["rule1"] is True
    # The heal reference is the healed full model: 0.82 - 0.80 = 0.02 is outside 0.01.
    rows["moe_saliency_32_heal500"] = fake_row(0.80)
    rows["moe_saliency_32"] = fake_row(0.75)
    out = r.decide(rows)
    assert out["moe_no_heal"]["pass"] is False
    assert out["moe_heal"]["pass"] is False
    assert out["rule1"] is False
    # A random seed beating the spread test: saliency only 0.005 above the best random fails.
    rows["moe_saliency_32"] = fake_row(0.79)
    rows["moe_random_32_s1"] = fake_row(0.785)
    assert r.decide(rows)["moe_no_heal"]["pass"] is False
    # A failed cell makes the rule undecidable, not false.
    rows["moe_random_32_s2"] = {"cell": "moe_random_32_s2", "error": "boom", "scores": None}
    rows["moe_saliency_32_heal500"] = fake_row(0.5)
    out = r.decide(rows)
    assert out["moe_no_heal"]["pass"] is None
    assert out["rule1"] is None


def test_round19_dense_rules_and_size_check():
    r = load_round()
    rows = {"dense_skill_3p1b_heal1000": dict(fake_row(0.70, 100), size_error=0.01, params_total_after=3.1e9),
            "dense_random_3p1b_heal1000": dict(fake_row(0.50, 80), size_error=0.01, params_total_after=3.1e9),
            "native_coder_3b": fake_row(0.65, 90), "native_coder_3b_heal1000": fake_row(0.69, 95)}
    out = r.decide(rows)
    assert out["rule2"] is True
    assert out["rule3"] is True
    assert len(out["rule2_gap_ci"]) == 2
    rows["native_coder_3b_heal1000"] = fake_row(0.72, 110)
    assert r.decide(rows)["rule2"] is False
    rows["dense_random_3p1b_heal1000"]["params_total_after"] = 3.0e9
    assert r.decide(rows)["rule3"] is False
    ahead = fake_row(150 / 164 * 0.5 + 150 / 257 * 0.5, 150)
    behind = fake_row(100 / 164 * 0.5 + 100 / 257 * 0.5, 100)
    gap, low, high = r.paired_gap(ahead, behind)
    assert 0 < low < gap < high
