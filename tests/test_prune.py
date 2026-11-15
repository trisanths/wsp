"""Tests for wsp.prune: domain pruning of tap donors, on tiny random models (CPU, no network)."""

import tempfile

import pytest
import torch
from torch import nn
from tokenizers import Regex, Tokenizer, models, pre_tokenizers

transformers = pytest.importorskip("transformers")

from wsp.parts import find_decoder_layers
from wsp.prune import domain_unit_scores, head_shape, model_params, prune_donor
from wsp.tap import run_donor

DOMAIN = ["the integral of x squared", "solve for x in the equation", "a prime number divides",
          "the derivative of sine", "factor the polynomial"]
GENERIC = ["the cat sat on a mat", "we went to the park today", "rain fell all night",
           "she sang a quiet song", "the train was late again"]
PROMPTS = ["what is the sum of two odd numbers", "a dog ran", "find the area"]
MAX_LAYER = 2


def char_tokenizer():
    """Character-level, no BOS, like the donor tokenizer in test_tap."""
    vocab = {"<unk>": 0, "<|end|>": 1}
    for letter in "abcdefghijklmnopqrstuvwxyz":
        vocab[letter] = len(vocab)
    base = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    base.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Whitespace(),
        pre_tokenizers.Split(Regex("."), behavior="isolated"),
    ])
    return transformers.PreTrainedTokenizerFast(
        tokenizer_object=base, unk_token="<unk>", eos_token="<|end|>", pad_token="<|end|>",
    )


def tiny_llama():
    config = transformers.LlamaConfig(
        hidden_size=32, intermediate_size=64, num_hidden_layers=4, num_attention_heads=8,
        num_key_value_heads=2, vocab_size=40, max_position_embeddings=128,
    )
    return transformers.LlamaForCausalLM(config)


def tiny_qwen2():
    config = transformers.Qwen2Config(
        hidden_size=32, intermediate_size=48, num_hidden_layers=4, num_attention_heads=4,
        num_key_value_heads=2, vocab_size=40, max_position_embeddings=128,
    )
    return transformers.Qwen2ForCausalLM(config)


def tiny_phi3():
    config = transformers.Phi3Config(
        hidden_size=32, intermediate_size=48, num_hidden_layers=4, num_attention_heads=4,
        num_key_value_heads=2, vocab_size=40, max_position_embeddings=128,
        original_max_position_embeddings=128, pad_token_id=1, bos_token_id=1, eos_token_id=1,
    )
    return transformers.Phi3ForCausalLM(config)


def tiny_mistral():
    config = transformers.MistralConfig(
        hidden_size=32, intermediate_size=48, num_hidden_layers=4, num_attention_heads=4,
        num_key_value_heads=2, vocab_size=40, max_position_embeddings=128,
    )
    return transformers.MistralForCausalLM(config)


def tiny_gemma2():
    config = transformers.Gemma2Config(
        hidden_size=32, intermediate_size=48, num_hidden_layers=4, num_attention_heads=4,
        num_key_value_heads=2, head_dim=8, vocab_size=40, max_position_embeddings=128,
        sliding_window=16, query_pre_attn_scalar=8, pad_token_id=1,
    )
    return transformers.Gemma2ForCausalLM(config)


BUILDERS = {"llama": tiny_llama, "qwen2": tiny_qwen2, "phi3": tiny_phi3,
            "mistral": tiny_mistral, "gemma2": tiny_gemma2}
FAMILIES = ["llama", "qwen2", "phi3", "mistral", "gemma2"]


def build(family, seed=0):
    torch.manual_seed(seed)
    return BUILDERS[family]().eval()


def scores_for(model, tok):
    return domain_unit_scores(model, tok, DOMAIN, GENERIC, MAX_LAYER, batch_size=2)


def masked_memory(model, tok, report):
    """The original donor's state at MAX_LAYER with the pruned units zeroed by hooks."""
    _, layers = find_decoder_layers(model, MAX_LAYER)
    handles = []
    for row in report["layers"]:
        layer = layers[row["layer"]]
        neuron_mask = torch.zeros(row["mlp_before"])
        neuron_mask[row["neurons"]] = 1.0
        head_dim = layer.self_attn.head_dim
        head_mask = torch.zeros(row["q_heads_before"], head_dim)
        head_mask[row["q_heads"]] = 1.0
        head_mask = head_mask.reshape(-1)

        def zero_neurons(module, args, mask=neuron_mask):
            return (args[0] * mask,)

        def zero_heads(module, args, mask=head_mask):
            return (args[0] * mask,)

        handles.append(layer.mlp.down_proj.register_forward_pre_hook(zero_neurons))
        handles.append(layer.self_attn.o_proj.register_forward_pre_hook(zero_heads))
    try:
        memory, _, _ = run_donor(model, tok, PROMPTS, MAX_LAYER)
    finally:
        for handle in handles:
            handle.remove()
    return memory


@pytest.mark.parametrize("family", FAMILIES)
def test_full_fraction_keeps_state_at_max_layer_identical(family):
    model = build(family)
    tok = char_tokenizer()
    scores = scores_for(model, tok)
    before, ends, valid = run_donor(model, tok, PROMPTS, MAX_LAYER)
    pruned, report = prune_donor(model, scores, 1.0, 1.0, MAX_LAYER)
    after, after_ends, after_valid = run_donor(pruned, tok, PROMPTS, MAX_LAYER)
    assert torch.equal(before, after)
    assert torch.equal(ends, after_ends)
    assert torch.equal(valid, after_valid)
    _, layers = find_decoder_layers(pruned, 0)
    assert len(layers) == MAX_LAYER + 1
    assert pruned.config.num_hidden_layers == MAX_LAYER + 1
    assert report["flops_per_token_after"] == report["flops_per_token_before"]
    assert report["params_after"] < report["params_before"]


@pytest.mark.parametrize("family", FAMILIES)
def test_half_fraction_equals_original_with_units_zeroed(family):
    # Pruning must compute exactly what the original computes with the dropped
    # neurons and query heads zeroed. That also checks the GQA KV head choice.
    model = build(family)
    tok = char_tokenizer()
    scores = scores_for(model, tok)
    pruned, report = prune_donor(model, scores, 0.5, 0.5, MAX_LAYER)
    expected = masked_memory(model, tok, report)
    memory, _, _ = run_donor(pruned, tok, PROMPTS, MAX_LAYER)
    assert memory.shape == expected.shape
    assert memory.shape[-1] == model.config.hidden_size
    assert torch.allclose(memory, expected, atol=1e-5)


@pytest.mark.parametrize("family", FAMILIES)
def test_half_fraction_model_runs_forward_and_generate(family):
    model = build(family)
    tok = char_tokenizer()
    pruned, _ = prune_donor(model, scores_for(model, tok), 0.5, 0.5, MAX_LAYER)
    ids = torch.tensor([[2, 3, 4, 5, 6]])
    out = pruned(input_ids=ids)
    assert out.logits.shape == (1, 5, model.config.vocab_size)
    generated = pruned.generate(ids, attention_mask=torch.ones_like(ids), max_new_tokens=3, do_sample=False,
                                pad_token_id=1)
    assert generated.shape == (1, 8)


def expected_layer_params(family, d, n_q, n_kv, head_dim, width):
    q = d * n_q * head_dim
    kv = d * n_kv * head_dim
    o = n_q * head_dim * d
    attn = q + 2 * kv + o
    if family == "qwen2":
        attn += n_q * head_dim + 2 * n_kv * head_dim
    mlp = 3 * d * width
    norms = 2 * d
    return attn + mlp + norms


@pytest.mark.parametrize("family", ["llama", "qwen2", "phi3"])
def test_half_fraction_parameter_counts(family):
    model = build(family)
    tok = char_tokenizer()
    pruned, report = prune_donor(model, scores_for(model, tok), 0.5, 0.5, MAX_LAYER)
    config = model.config
    d = config.hidden_size
    n_q = config.num_attention_heads
    n_kv = config.num_key_value_heads
    head_dim = d // n_q
    width = config.intermediate_size
    kv_keep, per_group = head_shape(n_q, n_kv, 0.5)
    assert kv_keep * per_group == n_q // 2

    for row in report["layers"]:
        assert row["params_before"] == expected_layer_params(family, d, n_q, n_kv, head_dim, width)
        assert row["params_after"] == expected_layer_params(family, d, kv_keep * per_group, kv_keep,
                                                            head_dim, width // 2)
        assert row["mlp_after"] == width // 2
        assert row["q_heads_after"] == n_q // 2

    assert report["params_after"] == model_params(pruned)
    assert report["flops_per_token_after"] == 2 * report["run_params_after"]
    assert report["flops_per_token_after"] < report["flops_per_token_before"]
    assert report["flops_per_token_before"] < report["flops_per_token_full_model"]
    # Layers 0..MAX_LAYER keep their norms and the embedding, so the model total
    # equals the original minus what was cut and the dropped layers.
    dropped = 0
    _, layers = find_decoder_layers(model, 0)
    for index in range(MAX_LAYER + 1, len(layers)):
        for param in layers[index].parameters():
            dropped += param.numel()
    cut = report["run_params_before"] - report["run_params_after"]
    assert report["params_after"] == report["params_before"] - dropped - cut

    assert pruned.config.num_attention_heads == kv_keep * per_group
    assert pruned.config.num_key_value_heads == kv_keep
    assert pruned.config.intermediate_size == width // 2
    assert pruned.config.head_dim == head_dim


def test_random_rule_keeps_same_counts_and_is_seeded():
    model = build("llama")
    tok = char_tokenizer()
    scores = scores_for(model, tok)
    _, top = prune_donor(model, scores, 0.5, 0.25, MAX_LAYER)
    _, rand_a = prune_donor(model, None, 0.5, 0.25, MAX_LAYER, keep_rule="random", seed=3)
    _, rand_b = prune_donor(model, None, 0.5, 0.25, MAX_LAYER, keep_rule="random", seed=3)
    _, rand_c = prune_donor(model, None, 0.5, 0.25, MAX_LAYER, keep_rule="random", seed=4)
    assert rand_a["params_after"] == top["params_after"]
    assert rand_a["flops_per_token_after"] == top["flops_per_token_after"]
    differs = False
    for row_top, row_a, row_b, row_c in zip(top["layers"], rand_a["layers"], rand_b["layers"], rand_c["layers"]):
        for key in ["mlp_after", "q_heads_after", "kv_heads_after", "params_after"]:
            assert row_a[key] == row_top[key]
        assert row_a["neurons"] == row_b["neurons"]
        assert row_a["q_heads"] == row_b["q_heads"]
        if row_a["neurons"] != row_c["neurons"] or row_a["neurons"] != row_top["neurons"]:
            differs = True
    assert differs


def test_random_pruned_donor_matches_masked_original():
    model = build("qwen2")
    tok = char_tokenizer()
    pruned, report = prune_donor(model, None, 0.5, 0.5, MAX_LAYER, keep_rule="random", seed=7)
    expected = masked_memory(model, tok, report)
    memory, _, _ = run_donor(pruned, tok, PROMPTS, MAX_LAYER)
    assert torch.allclose(memory, expected, atol=1e-5)


def test_top_rule_keeps_highest_scores():
    model = build("llama")
    width = model.config.intermediate_size
    n_q = model.config.num_attention_heads
    scores = {"mlp": {}, "heads": {}}
    for layer in range(MAX_LAYER + 1):
        scores["mlp"][layer] = torch.arange(width, dtype=torch.float32)
        # heads 0..3 read KV head 0 and heads 4..7 read KV head 1; make head 5 and 6 the best
        head_scores = torch.zeros(n_q)
        head_scores[5] = 3.0
        head_scores[6] = 2.0
        scores["heads"][layer] = head_scores
    _, report = prune_donor(model, scores, 0.25, 0.25, MAX_LAYER)
    for row in report["layers"]:
        assert row["neurons"] == list(range(width - width // 4, width))
        # 2 of 8 heads over 2 KV heads: one head per KV head
        assert row["kv_heads"] == [0, 1]
        assert row["q_heads"][1] == 5


def test_gqa_head_shape_and_grouping():
    assert head_shape(8, 2, 1.0) == (2, 4)
    assert head_shape(8, 2, 0.5) == (2, 2)
    assert head_shape(8, 2, 0.25) == (2, 1)
    assert head_shape(8, 2, 0.75) == (2, 3)
    assert head_shape(8, 2, 0.1) == (1, 1)
    assert head_shape(4, 4, 0.5) == (2, 1)
    # an exact count wins over keeping more KV heads
    assert head_shape(28, 4, 0.5) == (2, 7)

    model = build("llama")
    tok = char_tokenizer()
    scores = scores_for(model, tok)
    for keep in [0.1, 0.25, 0.5, 0.75]:
        pruned, report = prune_donor(model, scores, 1.0, keep, MAX_LAYER)
        group = model.config.num_attention_heads // model.config.num_key_value_heads
        for row in report["layers"]:
            per_kv = {}
            for head in row["q_heads"]:
                kv = head // group
                assert kv in row["kv_heads"]
                if kv not in per_kv:
                    per_kv[kv] = 0
                per_kv[kv] += 1
            assert sorted(per_kv) == row["kv_heads"]
            counts = set(per_kv.values())
            assert len(counts) == 1
            # new query head j reads new KV head j // per_group, which must be its old KV head
            per_group = counts.pop()
            for new_index, head in enumerate(row["q_heads"]):
                assert row["kv_heads"][new_index // per_group] == head // group
        _, layers = find_decoder_layers(pruned, 0)
        attn = layers[0].self_attn
        assert attn.num_key_value_groups == report["layers"][0]["q_heads_after"] // report["layers"][0]["kv_heads_after"]
        expected = masked_memory(model, tok, report)
        memory, _, _ = run_donor(pruned, tok, PROMPTS, MAX_LAYER)
        assert torch.allclose(memory, expected, atol=1e-5)


def test_scores_cover_only_layers_up_to_max_and_ignore_pads():
    model = build("qwen2")
    tok = char_tokenizer()
    _, layers = find_decoder_layers(model, 0)
    calls = []
    handles = []
    for index in range(len(layers)):

        def count(module, inputs, output, index=index):
            calls.append(index)

        handles.append(layers[index].register_forward_hook(count))
    try:
        scores = domain_unit_scores(model, tok, DOMAIN, GENERIC, 1, batch_size=8)
    finally:
        for handle in handles:
            handle.remove()
    assert max(calls) == 1
    assert sorted(scores["mlp"]) == [0, 1]
    assert sorted(scores["heads"]) == [0, 1]
    assert scores["mlp"][0].shape == (model.config.intermediate_size,)
    assert scores["heads"][0].shape == (model.config.num_attention_heads,)
    assert scores["domain_tokens"] > 0 and scores["generic_tokens"] > 0
    # Batches of one have no pads; the scores must match padded batches.
    single = domain_unit_scores(model, tok, DOMAIN, GENERIC, 1, batch_size=1)
    for layer in [0, 1]:
        assert torch.allclose(scores["mlp"][layer], single["mlp"][layer], atol=1e-4)
        assert torch.allclose(scores["heads"][layer], single["heads"][layer], atol=1e-4)
    # Domain against itself scores zero.
    same = domain_unit_scores(model, tok, DOMAIN, DOMAIN, 1)
    assert same["mlp"][0].abs().max() < 1e-3


def test_save_and_reload_pruned_donor():
    model = build("qwen2")
    tok = char_tokenizer()
    pruned, _ = prune_donor(model, scores_for(model, tok), 0.5, 0.5, MAX_LAYER)
    memory, _, _ = run_donor(pruned, tok, PROMPTS, MAX_LAYER)
    with tempfile.TemporaryDirectory() as path:
        pruned.save_pretrained(path)
        loaded = transformers.AutoModelForCausalLM.from_pretrained(path).eval()
    again, _, _ = run_donor(loaded, tok, PROMPTS, MAX_LAYER)
    assert torch.allclose(memory, again, atol=1e-6)


def test_default_copies_and_in_place_prunes_the_donor():
    model = build("llama")
    count = model_params(model)
    prune_donor(model, None, 0.5, 0.5, MAX_LAYER, keep_rule="random")
    assert model_params(model) == count
    pruned, _ = prune_donor(model, None, 0.5, 0.5, MAX_LAYER, keep_rule="random", in_place=True)
    assert pruned is model
    assert model_params(model) < count


class OddMLP(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.fc = nn.Linear(d, d)

    def forward(self, x):
        return self.fc(x)


def test_unsupported_layout_raises_and_leaves_donor_whole():
    model = build("llama")
    _, layers = find_decoder_layers(model, 0)
    layers[1].mlp = OddMLP(model.config.hidden_size)
    count = model_params(model)
    with pytest.raises(ValueError, match="not supported"):
        prune_donor(model, None, 0.5, 0.5, MAX_LAYER, keep_rule="random", in_place=True)
    assert model_params(model) == count


def test_top_rule_needs_scores_and_bad_arguments_raise():
    model = build("llama")
    with pytest.raises(ValueError, match="needs mlp scores"):
        prune_donor(model, None, 0.5, 1.0, MAX_LAYER)
    with pytest.raises(ValueError, match="keep_rule"):
        prune_donor(model, None, 0.5, 0.5, MAX_LAYER, keep_rule="best")
    with pytest.raises(ValueError, match="keep_mlp"):
        prune_donor(model, None, 0.0, 0.5, MAX_LAYER, keep_rule="random")
    pruned, _ = prune_donor(model, None, 1.0, 1.0, MAX_LAYER)
    assert pruned.config.num_hidden_layers == MAX_LAYER + 1


def test_negative_max_layer_raises():
    model = build("llama")
    tok = char_tokenizer()
    with pytest.raises(ValueError, match="max_layer"):
        prune_donor(model, None, 1.0, 1.0, -1)
    with pytest.raises(ValueError, match="max_layer"):
        domain_unit_scores(model, tok, DOMAIN, GENERIC, -1)


def test_quantized_linear_raises_not_supported():
    model = build("llama")
    _, layers = find_decoder_layers(model, 0)
    down = layers[0].mlp.down_proj
    # A packed 4-bit weight: uint8 storage that does not have the (out, in) shape.
    packed = torch.zeros(down.out_features * down.in_features // 2, 1, dtype=torch.uint8)
    down.weight = nn.Parameter(packed, requires_grad=False)
    with pytest.raises(ValueError, match="quantized"):
        prune_donor(model, None, 0.5, 0.5, MAX_LAYER, keep_rule="random")


def tiny_mistral3():
    text = transformers.MistralConfig(
        hidden_size=32, intermediate_size=48, num_hidden_layers=4, num_attention_heads=4,
        num_key_value_heads=2, head_dim=8, vocab_size=40, max_position_embeddings=128,
    )
    vision = transformers.PixtralVisionConfig(
        hidden_size=16, intermediate_size=32, num_hidden_layers=1, num_attention_heads=2,
        image_size=32, patch_size=8,
    )
    config = transformers.Mistral3Config(text_config=text, vision_config=vision, image_token_index=39)
    return transformers.Mistral3ForConditionalGeneration(config).eval()


@pytest.mark.skipif(not hasattr(transformers, "Mistral3ForConditionalGeneration"),
                    reason="needs mistral3 in transformers")
def test_multimodal_donor_updates_text_config_and_reloads():
    torch.manual_seed(0)
    model = tiny_mistral3()
    tok = char_tokenizer()
    scores = domain_unit_scores(model, tok, DOMAIN, GENERIC, MAX_LAYER)
    pruned, _ = prune_donor(model, scores, 0.5, 0.5, MAX_LAYER)
    text = pruned.config.text_config
    assert text.num_hidden_layers == MAX_LAYER + 1
    assert text.num_attention_heads == 2
    assert text.intermediate_size == 24
    # The original donor's config is untouched.
    assert model.config.text_config.num_attention_heads == 4
    memory, _, _ = run_donor(pruned, tok, PROMPTS, MAX_LAYER)
    with tempfile.TemporaryDirectory() as path:
        pruned.save_pretrained(path)
        loaded = transformers.Mistral3ForConditionalGeneration.from_pretrained(path).eval()
    again, _, _ = run_donor(loaded, tok, PROMPTS, MAX_LAYER)
    assert torch.allclose(memory, again, atol=1e-6)
