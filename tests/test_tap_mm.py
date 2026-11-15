"""Tests for wsp.tap_mm: a text-only host reads a fake VLM donor through a tap."""

import math
import tempfile

import pytest
import torch
from torch import nn

transformers = pytest.importorskip("transformers")
from tokenizers import Tokenizer, models, pre_tokenizers, processors

from wsp.lora import apply_lora
from wsp.tap import (TapBranch, attach_taps, build_visibility, host_token_ends, load_taps, save_taps,
                     tap_parameters)
from wsp.tap_mm import (MultimodalDonor, accuracy, mcq_prompt, mismatch_images, prepare_mm, score_choices,
                        score_vlm_choices, tap_train_mm)

WORDS = ["what", "is", "it", "the", "color", "a", "dark", "bright", "image", "A", "B", "Question", "Answer",
         "Context", ":", ".", "or"]
IMAGE_TOKENS = 3
PROMPT = "what is the color"


def host_tokenizer():
    """Word-level, with a BOS token, like Llama."""
    vocab = {"<unk>": 0, "</s>": 1, "<pad>": 2, "<s>": 3}
    for word in WORDS:
        vocab[word] = len(vocab)
    base = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    base.pre_tokenizer = pre_tokenizers.Whitespace()
    base.post_processor = processors.TemplateProcessing(single="<s> $A", special_tokens=[("<s>", 3)])
    return transformers.PreTrainedTokenizerFast(
        tokenizer_object=base, unk_token="<unk>", eos_token="</s>", pad_token="<pad>", bos_token="<s>",
    )


def donor_text_tokenizer():
    """Word-level with chat and image special tokens, no BOS, like Qwen."""
    specials = ["<unk>", "<pad>", "<image>", "<|user|>", "<|assistant|>", "<|end|>"]
    vocab = {}
    for token in specials:
        vocab[token] = len(vocab)
    for word in WORDS:
        vocab[word] = len(vocab)
    base = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    base.pre_tokenizer = pre_tokenizers.Whitespace()
    return transformers.PreTrainedTokenizerFast(
        tokenizer_object=base, unk_token="<unk>", eos_token="<|end|>", pad_token="<pad>",
        additional_special_tokens=["<image>", "<|user|>", "<|assistant|>"],
    )


class FakeProcessor:
    """A processor like Qwen2.5-VL's: one <image> placeholder per image expands to IMAGE_TOKENS tokens."""

    image_token = "<image>"

    def __init__(self):
        self.tokenizer = donor_text_tokenizer()

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        text = ""
        for message in messages:
            text += "<|user|>"
            for part in message["content"]:
                if part["type"] == "image":
                    text += "<image>"
                else:
                    text += " " + part["text"]
            text += " <|end|>"
        if add_generation_prompt:
            text += " <|assistant|>"
        return text

    def __call__(self, text=None, images=None, padding=True, return_tensors="pt", padding_side=None):
        expanded = []
        for row in text:
            expanded.append(row.replace("<image>", "<image>" * IMAGE_TOKENS))
        if padding_side is not None:
            self.tokenizer.padding_side = padding_side
        encoded = self.tokenizer(expanded, padding=padding, return_tensors=return_tensors)
        data = {"input_ids": encoded["input_ids"], "attention_mask": encoded["attention_mask"]}
        if images is not None:
            data["pixel_values"] = torch.stack(images)
        return transformers.BatchFeature(data)


class FakeVLM(nn.Module):
    """A tiny Qwen2 language model plus a vision encoder that turns an image into IMAGE_TOKENS states."""

    def __init__(self, image_token_id, seed=1):
        super().__init__()
        torch.manual_seed(seed)
        config = transformers.Qwen2Config(
            hidden_size=24, intermediate_size=48, num_hidden_layers=4, num_attention_heads=4,
            num_key_value_heads=2, vocab_size=40, max_position_embeddings=128,
        )
        self.language_model = transformers.Qwen2ForCausalLM(config)
        self.vision = nn.Linear(3, 24)
        self.vision_position = nn.Parameter(torch.randn(IMAGE_TOKENS, 24) * 0.1)
        self.config = config
        self.config.image_token_id = image_token_id
        self.eval()

    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def forward(self, input_ids, attention_mask=None, pixel_values=None, use_cache=False, **kwargs):
        embeds = self.get_input_embeddings()(input_ids)
        if pixel_values is not None:
            pooled = pixel_values.mean(dim=(2, 3))
            features = self.vision(pooled)[:, None, :] + self.vision_position[None]
            places = input_ids == self.config.image_token_id
            embeds = embeds.clone()
            embeds[places] = features.reshape(-1, 24).to(embeds.dtype)
        return self.language_model(inputs_embeds=embeds, attention_mask=attention_mask, use_cache=use_cache)


def make_donor(read_layer=2, format_text=None):
    processor = FakeProcessor()
    image_id = processor.tokenizer.convert_tokens_to_ids("<image>")
    return MultimodalDonor(FakeVLM(image_id), processor, read_layer=read_layer, format_text=format_text)


def tiny_host(seed=0):
    torch.manual_seed(seed)
    config = transformers.LlamaConfig(
        hidden_size=32, intermediate_size=64, num_hidden_layers=3, num_attention_heads=4,
        num_key_value_heads=2, vocab_size=64, max_position_embeddings=128,
    )
    return transformers.LlamaForCausalLM(config).eval()


def make_branch(donor_layer=2, seed=5, gate=None):
    torch.manual_seed(seed)
    branch = TapBranch(32, 24, attn_dim=16, heads=4, donor_id="vlm", donor_layer=donor_layer)
    if gate is not None:
        with torch.no_grad():
            branch.gate.fill_(gate)
    return branch


def image(brightness, seed=0):
    generator = torch.Generator()
    generator.manual_seed(seed)
    return torch.full((3, 4, 4), float(brightness)) + 0.05 * torch.randn(3, 4, 4, generator=generator)


def test_fresh_tap_keeps_host_logits_exactly():
    host = tiny_host()
    tok = host_tokenizer()
    tok.padding_side = "left"
    prompts = [PROMPT, "what is it", "is it dark or bright"]
    encoded = tok(prompts, padding=True, return_tensors="pt")
    with torch.no_grad():
        expected = host(input_ids=encoded["input_ids"], attention_mask=encoded["attention_mask"]).logits
    attach_taps(host, [(1, make_branch(), 0)])
    items = [(prompts[0], image(0.9)), (prompts[1], image(0.1)), (prompts[2], [image(0.5), image(0.2)])]
    inputs = prepare_mm(host, tok, {"vlm": make_donor()}, items)
    assert torch.equal(inputs["input_ids"], encoded["input_ids"])
    with torch.no_grad():
        got = host(**inputs).logits
    assert torch.equal(got, expected)


def test_donor_sees_the_image():
    donor = make_donor()
    bright, ends, valid = donor.memory([(PROMPT, image(0.9))])
    dark, dark_ends, dark_valid = donor.memory([(PROMPT, image(0.1))])
    assert bright.shape == dark.shape
    assert torch.equal(ends, dark_ends)
    # The user header comes before the image, so its state does not change.
    assert torch.equal(bright[0, 0], dark[0, 0])
    # The image tokens and every token after them change.
    for position in range(1, bright.shape[1]):
        assert not torch.allclose(bright[0, position], dark[0, position])
    # A text-only item runs too, with no image tokens.
    plain, plain_ends, plain_valid = donor.memory([(PROMPT, None)])
    assert plain.shape[1] == bright.shape[1] - IMAGE_TOKENS


def test_donor_stops_after_its_read_layer():
    donor = make_donor()
    calls = [0, 0, 0, 0]
    handles = []
    for index, layer in enumerate(donor.model.language_model.model.layers):

        def count(module, inputs, output, index=index):
            calls[index] += 1

        handles.append(layer.register_forward_hook(count))
    head_calls = [0]
    handles.append(donor.model.language_model.lm_head.register_forward_hook(
        lambda m, i, o: head_calls.__setitem__(0, head_calls[0] + 1)))
    memory, ends, valid = donor.memory([(PROMPT, image(0.9))], layer=1)
    for handle in handles:
        handle.remove()
    assert calls == [1, 1, 0, 0]
    assert head_calls[0] == 0
    assert memory.dtype == torch.float32
    assert donor.hidden_size == 24
    assert donor.layers_total == 4


def test_visibility_marks_image_tokens_always_visible():
    donor = make_donor()
    items = [(PROMPT, image(0.9)), ("what is it", [image(0.1), image(0.5)])]
    memory, ends, valid = donor.memory(items)
    assert donor.fallback_rows == []
    # Row 0: <|user|> <image> x3 what is the color <|end|> <|assistant|> <pad> x2
    # Row 1: <|user|> <image> x6 what is it <|end|> <|assistant|>
    assert ends[0].tolist() == [-1, -1, -1, -1, 4, 7, 11, 17, 17, 17, -1, -1]
    assert valid[0].tolist() == [True] * 10 + [False] * 2
    assert ends[1].tolist() == [-1, -1, -1, -1, -1, -1, -1, 4, 7, 10, 10, 10]
    assert valid[1].tolist() == [True] * 12
    assert donor.image_mask[0].tolist() == [False, True, True, True] + [False] * 8
    assert int(donor.image_mask[1].sum()) == 6

    host_tok = host_tokenizer()
    host_tok.padding_side = "left"
    _, _, host_ends = host_token_ends(host_tok, [PROMPT])
    # <s> what is the color
    assert host_ends.tolist() == [[-1, 4, 7, 11, 17]]
    vis = build_visibility(host_ends, ends[:1], valid[:1])
    image_and_header = [True, True, True, True]
    assert vis[0, 0].tolist() == [False] * 12
    assert vis[0, 1].tolist() == image_and_header + [True, False, False, False, False, False] + [False] * 2
    assert vis[0, 2].tolist() == image_and_header + [True, True, False, False, False, False] + [False] * 2
    # The last prompt token reads the whole donor row, template suffix too, never the pads.
    assert vis[0, 4].tolist() == [True] * 10 + [False] * 2


def test_fallback_makes_every_donor_token_visible():
    # This format rewrites the prompt, so its offsets cannot be matched.
    def shouting(prompt, image_count):
        return "<|user|>" + "<image>" * image_count + " " + prompt.upper() + " <|assistant|>"

    donor = make_donor(format_text=shouting)
    memory, ends, valid = donor.memory([(PROMPT, image(0.9)), ("what is it", image(0.2))])
    assert donor.fallback_rows == [0, 1]
    assert bool((ends == -1).all())
    host_tok = host_tokenizer()
    _, _, host_ends = host_token_ends(host_tok, [PROMPT, "what is it"])
    vis = build_visibility(host_ends, ends, valid)
    for row in range(2):
        for position in range(host_ends.shape[1]):
            if int(host_ends[row, position]) >= 0:
                assert vis[row, position].tolist() == valid[row].tolist()
            else:
                assert not bool(vis[row, position].any())


def test_batch_rows_equal_single_rows():
    donor = make_donor()
    items = [(PROMPT, image(0.9)), ("what is it", [image(0.1), image(0.5)]), ("it", None)]
    batch, batch_ends, batch_valid = donor.memory(items)
    for row in range(len(items)):
        single, single_ends, single_valid = donor.memory([items[row]])
        n = single.shape[1]
        assert torch.allclose(batch[row, :n], single[0], atol=1e-5)
        assert batch_ends[row, :n].tolist() == single_ends[0].tolist()
        assert not bool(batch_valid[row, n:].any())


def toy_items(count, seed):
    generator = torch.Generator()
    generator.manual_seed(seed)
    items = []
    for index in range(count):
        bright = index % 2 == 0
        level = 0.1 + 0.3 * float(torch.rand((), generator=generator))
        if bright:
            level += 0.5
        prompt = mcq_prompt("what is the color", ["bright", "dark"])
        gold = 0 if bright else 1
        items.append({"id": index, "prompt": prompt, "images": image(level, seed=seed * 1000 + index),
                      "gold": gold, "n_choices": 2})
    return items


def test_mcq_prompt_format():
    prompt = mcq_prompt("what is it", ["bright", "dark"], context="the image")
    assert prompt == "Context: the image\nQuestion: what is it\nA. bright\nB. dark\nAnswer:"


def test_training_reads_the_image_and_host_alone_cannot():
    tok = host_tokenizer()
    donors = {"vlm": make_donor()}
    train = toy_items(32, seed=1)
    held_out = toy_items(16, seed=2)
    examples = []
    for item in train:
        examples.append((item["prompt"], item["images"], " " + "AB"[item["gold"]]))

    host = tiny_host()
    alone = score_choices(host, tok, held_out)
    # Every prompt is the same text, so the host alone gives one answer: chance.
    assert accuracy(alone) == 0.5

    attach_taps(host, [(1, make_branch(), 0)])
    gates = []
    others = []
    for branch in host.wsp_taps:
        for name, param in branch.named_parameters():
            if name == "gate":
                gates.append(param)
            else:
                others.append(param)
    groups = [{"params": others, "lr": 1e-2}, {"params": gates, "lr": 5e-2}]
    losses = tap_train_mm(host, tok, donors, examples, groups, steps=80, lr=1e-2, batch_size=8, seed=0,
                          log_every=0)
    assert len(losses) == 80
    assert sum(losses[-10:]) / 10 < sum(losses[:10]) / 10 - 0.2
    assert host.wsp_tap_state["memories"] is None
    tapped = score_choices(host, tok, held_out, donors=donors)
    assert accuracy(tapped) >= 0.9
    # The mismatched-image control drops back toward chance.
    control = score_choices(host, tok, mismatch_images(held_out), donors=donors)
    assert accuracy(control) <= 0.25


def test_text_only_lora_trains_without_taps():
    tok = host_tokenizer()
    host = tiny_host()
    params = apply_lora(host, 2)
    examples = []
    for item in toy_items(16, seed=3):
        examples.append((item["prompt"], item["images"], " " + "AB"[item["gold"]]))
    losses = tap_train_mm(host, tok, {}, examples, params, steps=20, lr=1e-2, batch_size=4, log_every=0)
    assert len(losses) == 20
    # The prompt alone gives near ln 2 nats on the answer at best; it cannot see the image.
    records = score_choices(host, tok, toy_items(8, seed=4))
    assert accuracy(records) == 0.5
    assert math.isfinite(losses[-1])


def test_vlm_alone_scores_choices():
    donor = make_donor()
    items = toy_items(4, seed=5)
    records = score_vlm_choices(donor, items, batch_size=3)
    assert len(records) == 4
    for record in records:
        assert len(record["probs"]) == 2
        assert abs(sum(record["probs"]) - 1.0) < 1e-3


def test_save_then_load_gives_same_logits():
    tok = host_tokenizer()
    tok.padding_side = "left"
    donors = {"vlm": make_donor()}
    items = [(PROMPT, image(0.9)), ("what is it", image(0.2))]
    host = tiny_host()
    attach_taps(host, [(1, make_branch(gate=0.9), 0), (2, make_branch(donor_layer=3, seed=8, gate=0.4), 1)])
    inputs = prepare_mm(host, tok, donors, items)
    with torch.no_grad():
        expected = host(**inputs).logits
    with tempfile.TemporaryDirectory() as path:
        save_taps(host, path, meta={"vlm": "fake vlm"})
        fresh = tiny_host()
        info = load_taps(fresh, path)
    assert info["meta"] == {"vlm": "fake vlm"}
    inputs = prepare_mm(fresh, tok, donors, items)
    with torch.no_grad():
        got = fresh(**inputs).logits
    assert torch.equal(got, expected)
    # One donor run serves both taps.
    assert fresh.wsp_tap_stats["donor_runs"] == 1
    assert fresh.wsp_tap_stats["donors"]["vlm"]["layers_ran"] == 4
