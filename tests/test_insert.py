"""Insert donor layers into a base: one network that equals the base until the gate opens."""

import pytest
import torch

transformers = pytest.importorskip("transformers")

from wsp.insert import (
    Insertion,
    add_base_lora,
    gate_values,
    heal,
    insert_layers,
    insertion_parameters,
    insertions_of,
    load_inserted,
    param_counts,
    ridge_stitch,
    save_inserted,
)
from tests.test_align import WORDS, tiny_tokenizer

NEW_TOKENS = 8
SETTINGS = dict(max_new_tokens=NEW_TOKENS, do_sample=False, pad_token_id=0, eos_token_id=None)

TEXTS = []
for i in range(16):
    TEXTS.append(" ".join(WORDS[i % 5: i % 5 + 6]))


def tiny_base(layers=4):
    torch.manual_seed(0)
    config = transformers.LlamaConfig(
        hidden_size=64, intermediate_size=128, num_hidden_layers=layers, num_attention_heads=4,
        num_key_value_heads=2, vocab_size=300, max_position_embeddings=64,
    )
    config._attn_implementation = "eager"
    return transformers.LlamaForCausalLM(config).eval()


def tiny_lfm2():
    torch.manual_seed(1)
    config = transformers.Lfm2Config(
        hidden_size=64, intermediate_size=128, num_hidden_layers=4, num_attention_heads=4,
        num_key_value_heads=2, vocab_size=300, max_position_embeddings=64,
        layer_types=["conv", "full_attention", "conv", "full_attention"], block_auto_adjust_ff_dim=False,
    )
    config._attn_implementation = "eager"
    return transformers.Lfm2ForCausalLM(config).eval()


def tiny_qwen():
    torch.manual_seed(3)
    config = transformers.Qwen2Config(
        hidden_size=48, intermediate_size=96, num_hidden_layers=4, num_attention_heads=4,
        num_key_value_heads=2, vocab_size=300, max_position_embeddings=64,
    )
    config._attn_implementation = "eager"
    return transformers.Qwen2ForCausalLM(config).eval()


def tiny_gemma():
    torch.manual_seed(5)
    config = transformers.Gemma3TextConfig(
        hidden_size=40, intermediate_size=80, num_hidden_layers=4, num_attention_heads=4,
        num_key_value_heads=2, head_dim=10, vocab_size=300, max_position_embeddings=64,
        sliding_window=3, rope_local_base_freq=100.0,
        layer_types=["full_attention", "sliding_attention", "full_attention", "sliding_attention"],
    )
    config._attn_implementation = "eager"
    return transformers.Gemma3ForCausalLM(config).eval()


def batch():
    """A left-padded batch of two rows."""
    ids = torch.tensor([[0, 0, 5, 9, 4, 7], [3, 8, 6, 5, 9, 4]])
    mask = torch.tensor([[0, 0, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1]])
    return ids, mask


def logits(model):
    ids, mask = batch()
    with torch.no_grad():
        return model(input_ids=ids, attention_mask=mask).logits.float()


def open_gates(model, value):
    with torch.no_grad():
        for insertion in insertions_of(model):
            insertion.w_out.gate.fill_(value)


def test_fresh_network_equals_base():
    base = tiny_base()
    expected = logits(base)
    insert_layers(base, tiny_qwen(), [1, 2], 2, stitch_kind="mlp")
    assert torch.equal(logits(base), expected)
    assert len(base.wsp_grafts) == 1
    assert isinstance(base.wsp_grafts[0], Insertion)


def test_gate_and_stitches_get_gradients():
    base = tiny_base()
    insertion = insert_layers(base, tiny_qwen(), [1, 2], 2, stitch_kind="linear", part_lora_rank=2)
    ids, mask = batch()

    def loss_grads():
        base.zero_grad()
        found = insertion_parameters(base)
        for name in found:
            for param in found[name]:
                param.requires_grad_(True)
        out = base(input_ids=ids, attention_mask=mask, labels=ids)
        out.loss.backward()
        return found

    # With the gate at 0 only the gate gets a gradient.
    found = loss_grads()
    assert float(insertion.w_out.gate.grad.abs()) > 0
    # With the gate open the stitches and the donor LoRA learn too.
    open_gates(base, 0.5)
    found = loss_grads()
    assert insertion.w_in.weight.grad.abs().sum() > 0
    assert insertion.w_out.linear.weight.grad.abs().sum() > 0
    # B starts at zero, so A gets no gradient yet, but B does.
    lora_grad = 0.0
    for param in found["part_lora"]:
        if param.grad is not None:
            lora_grad += float(param.grad.abs().sum())
    assert lora_grad > 0


def test_open_gate_runs_donor_block():
    base = tiny_base()
    expected = logits(base)
    donor = tiny_qwen()
    insertion = insert_layers(base, donor, [1, 2], 2)
    calls = []
    handle = insertion.part.layers[0].register_forward_hook(lambda module, args, out: calls.append(1))
    open_gates(base, 1.0)
    changed = logits(base)
    handle.remove()
    assert len(calls) > 0
    assert not torch.allclose(changed, expected)


def test_same_width_identity_insert_is_a_plain_stack():
    """With equal widths, identity stitches, and gate 1, the network is base[:at] + donor + base[at:]."""
    base = tiny_base()
    torch.manual_seed(7)
    donor_config = transformers.LlamaConfig(
        hidden_size=64, intermediate_size=96, num_hidden_layers=3, num_attention_heads=4,
        num_key_value_heads=2, vocab_size=300, max_position_embeddings=64,
    )
    donor_config._attn_implementation = "eager"
    donor = transformers.LlamaForCausalLM(donor_config).eval()

    # The plain stack: one Llama with base layers 0, 1, donor layers 1, 2, then base layers 2, 3.
    stack_config = transformers.LlamaConfig(
        hidden_size=64, intermediate_size=128, num_hidden_layers=6, num_attention_heads=4,
        num_key_value_heads=2, vocab_size=300, max_position_embeddings=64,
    )
    stack_config._attn_implementation = "eager"
    stack = transformers.LlamaForCausalLM(stack_config).eval()
    stack.model.embed_tokens = base.model.embed_tokens
    stack.model.norm = base.model.norm
    stack.lm_head = base.lm_head
    layers = [base.model.layers[0], base.model.layers[1], donor.model.layers[1], donor.model.layers[2],
              base.model.layers[2], base.model.layers[3]]
    stack.model.layers = torch.nn.ModuleList(layers)
    ids = torch.tensor([[3, 8, 6, 5, 9, 4]])
    with torch.no_grad():
        expected = stack(input_ids=ids, use_cache=False).logits.float()

    insert_layers(base, donor, [1, 2], 2, stitch_kind="identity")
    open_gates(base, 1.0)
    with torch.no_grad():
        got = base(input_ids=ids).logits.float()
    assert torch.allclose(got, expected, atol=1e-5)


def test_two_insertions_at_two_depths():
    base = tiny_base()
    expected = logits(base)
    first = insert_layers(base, tiny_qwen(), [0, 1], 1, stitch_kind="linear")
    second = insert_layers(base, tiny_gemma(), [1, 2], 3, stitch_kind="mlp")
    assert len(base.wsp_grafts) == 2
    assert first.meta["at"] == 1 and second.meta["at"] == 3
    assert torch.equal(logits(base), expected)

    open_gates(base, 0.5)
    both = logits(base)
    with torch.no_grad():
        first.w_out.gate.fill_(0.0)
    only_second = logits(base)
    assert not torch.allclose(both, expected)
    assert not torch.allclose(only_second, expected)
    assert not torch.allclose(both, only_second)
    assert gate_values(base) == [0.0, 0.5]


@pytest.mark.parametrize("two", [False, True])
def test_cached_generate_equals_uncached(two):
    base = tiny_base()
    insert_layers(base, tiny_qwen(), [1, 2], 2, stitch_kind="linear")
    if two:
        insert_layers(base, tiny_gemma(), [1, 2], 3, stitch_kind="mlp")
    open_gates(base, 0.7)
    ids, mask = batch()
    with torch.no_grad():
        cached = base.generate(input_ids=ids, attention_mask=mask, use_cache=True, **SETTINGS)
        uncached = base.generate(input_ids=ids, attention_mask=mask, use_cache=False, **SETTINGS)
    assert torch.equal(cached, uncached)


def test_save_and_load_round_trip(tmp_path):
    base = tiny_base()
    base_dir = str(tmp_path / "base")
    base.save_pretrained(base_dir)
    qwen = tiny_qwen()
    qwen_dir = str(tmp_path / "qwen")
    qwen.save_pretrained(qwen_dir)
    gemma = tiny_gemma()
    gemma_dir = str(tmp_path / "gemma")
    gemma.save_pretrained(gemma_dir)
    tokenizer = tiny_tokenizer()

    insert_layers(base, qwen_dir, [1, 2], 2, stitch_kind="mlp", part_lora_rank=2, sink=torch.randn(48))
    insert_layers(base, gemma, [0, 1], 3, stitch_kind="lowrank", stitch_rank=8, donor_id=gemma_dir)
    add_base_lora(base, 2, layers=[2, 3])
    lrs = {"stitch": 1e-2, "gate": 5e-2, "part_lora": 1e-2, "base_lora": 1e-2}
    losses = heal(base, tokenizer, TEXTS, lrs, steps=4, batch_size=4, seq_len=16, log_every=0)
    assert len(losses) == 4
    assert gate_values(base)[0] != 0.0
    expected = logits(base)

    out = str(tmp_path / "inserted")
    manifest = save_inserted(base, tokenizer, out, base_dir)
    assert len(manifest["insertions"]) == 2
    assert manifest["base_lora"]["rank"] == 2
    model, loaded_tokenizer = load_inserted(out, dtype=torch.float32)
    assert len(model.wsp_grafts) == 2
    # The reloaded base runs sdpa attention, not eager, and the two differ
    # only on pad positions (rows with no real key), so compare real tokens.
    _, mask = batch()
    real = mask.bool()
    assert torch.allclose(logits(model)[real], expected[real], atol=1e-5)
    assert loaded_tokenizer.get_vocab() == tokenizer.get_vocab()


def test_param_counts():
    base = tiny_base()
    base_total = 0
    for param in base.parameters():
        base_total += param.numel()
    donor = tiny_qwen()
    insert_layers(base, donor, [1, 2], 2, stitch_kind="linear", part_lora_rank=2)
    add_base_lora(base, 2)
    counts = param_counts(base)
    donor_layers = 0
    for layer in donor.model.layers[1:3]:
        for param in layer.parameters():
            donor_layers += param.numel()
    assert counts["base"] == base_total
    assert counts["donor"] == donor_layers
    assert counts["stitch"] == 64 * 48 * 2
    assert counts["gate"] == 1
    assert counts["part_lora"] > 0 and counts["base_lora"] > 0
    pieces = counts["base"] + counts["donor"] + counts["stitch"] + counts["gate"] + counts["part_lora"]
    assert counts["total"] == pieces + counts["base_lora"]
    assert counts["per_token"] == counts["total"] - base.model.embed_tokens.weight.numel()


def test_ridge_stitch_fits_both_ways():
    base = tiny_base()
    donor = tiny_qwen()
    tokenizer = tiny_tokenizer()
    stitch, w_out, report = ridge_stitch(base, tokenizer, 2, donor, tokenizer, 1, TEXTS, kind="linear")
    assert report["pairs"] > 20
    assert tuple(stitch.weight.shape) == (48, 64)
    assert tuple(w_out.shape) == (64, 48)
    insertion = insert_layers(base, donor, [1, 2], 2, stitch_in=stitch, stitch_out=w_out)
    assert torch.allclose(insertion.w_out.linear.weight, w_out)
    assert insertion.meta["stitch"] == {"kind": "linear"}


def test_bad_at_raises():
    base = tiny_base()
    with pytest.raises(ValueError):
        insert_layers(base, tiny_qwen(), [1, 2], 0)
    with pytest.raises(ValueError):
        insert_layers(base, tiny_qwen(), [1, 2], 5)
    with pytest.raises(ValueError):
        insert_layers(base, tiny_qwen(), [1, 2], 2, stitch_kind="identity")


def test_lfm2_base_with_insert():
    """A hybrid conv and attention base: fresh equals base, cached generate equals uncached."""
    base = tiny_lfm2()
    expected = logits(base)
    insert_layers(base, tiny_qwen(), [1, 2], 2)
    assert torch.equal(logits(base), expected)
    add_base_lora(base, 2)
    assert "out_proj" in base._wsp_base_lora["targets"]
    open_gates(base, 0.7)
    ids, mask = batch()
    with torch.no_grad():
        cached = base.generate(input_ids=ids, attention_mask=mask, use_cache=True, **SETTINGS)
        uncached = base.generate(input_ids=ids, attention_mask=mask, use_cache=False, **SETTINGS)
    assert torch.equal(cached, uncached)


def test_lora_only_save_and_load_round_trip(tmp_path):
    """The round 17 lora arm has a base LoRA and no insertions; it must save and load too."""
    base = tiny_base()
    base_dir = str(tmp_path / "base")
    base.save_pretrained(base_dir)
    tokenizer = tiny_tokenizer()
    add_base_lora(base, 2)
    heal(base, tokenizer, TEXTS, {"base_lora": 1e-2}, steps=2, batch_size=4, seq_len=16, log_every=0)
    expected = logits(base)

    out = str(tmp_path / "lora_only")
    manifest = save_inserted(base, tokenizer, out, base_dir)
    assert manifest["insertions"] == []
    model, _ = load_inserted(out, dtype=torch.float32)
    _, mask = batch()
    real = mask.bool()
    assert torch.allclose(logits(model)[real], expected[real], atol=1e-5)


def test_conv_donor_runs_without_cache():
    """LFM2 conv layers as a donor cannot use the part KV cache, so the network turns the cache off."""
    base = tiny_base()
    expected = logits(base)
    insertion = insert_layers(base, tiny_lfm2(), [0, 1], 2)
    assert not insertion.cacheable
    assert base.generation_config.use_cache is False
    assert torch.equal(logits(base), expected)
    open_gates(base, 0.7)
    ids, mask = batch()
    with torch.no_grad():
        default = base.generate(input_ids=ids, attention_mask=mask, **SETTINGS)
        uncached = base.generate(input_ids=ids, attention_mask=mask, use_cache=False, **SETTINGS)
    assert torch.equal(default, uncached)
    with pytest.raises(ValueError):
        with torch.no_grad():
            base.generate(input_ids=ids, attention_mask=mask, use_cache=True, **SETTINGS)


def test_attention_donor_keeps_the_cache():
    base = tiny_base()
    insertion = insert_layers(base, tiny_qwen(), [1, 2], 2)
    assert insertion.cacheable
    assert base.generation_config.use_cache is not False


class TemplateHolder:
    def __init__(self, template):
        self.chat_template = template


def round17():
    pytest.importorskip("modal")
    import modal_round17

    return modal_round17


def test_round17_pins_the_chat_date():
    module = round17()
    template = ('{%- if not date_string is defined %}{%- if strftime_now is defined %}'
                '{%- set date_string = strftime_now("%d %b %Y") %}{%- else %}'
                '{%- set date_string = "26 Jul 2024" %}{%- endif %}{%- endif %}{{ date_string }}')
    holder = TemplateHolder(template)
    assert module.pin_chat_date(holder)
    assert "strftime_now is defined" not in holder.chat_template
    plain = TemplateHolder("{{ messages }}")
    assert not module.pin_chat_date(plain)
    assert plain.chat_template == "{{ messages }}"


def test_round17_fallback_runner_is_safe():
    module = round17()
    assert module.run_tests("assert 1 + 1 == 2\n")
    assert not module.run_tests("assert 1 + 1 == 3\n")
    # An early exit with status 0 skips the asserts, so it must fail.
    assert not module.run_tests("import sys\nsys.exit(0)\nassert False\n")
    import time

    start = time.time()
    assert not module.run_tests("while True:\n    pass\n", timeout=1.0)
    # A grandchild that keeps running must not hold up the runner.
    program = "import subprocess, sys\nsubprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])\n"
    program += "while True:\n    pass\n"
    assert not module.run_tests(program, timeout=1.0)
    assert time.time() - start < 15
