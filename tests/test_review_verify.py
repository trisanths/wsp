"""Checks added when the review fixes were verified together."""

import math

import torch
import transformers
from tokenizers import Tokenizer, models, pre_tokenizers, processors
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

from wsp.assemble import fit_w_in
from wsp.graft import Graft, attach_grafts
from wsp.parts import build_attn_heads

WORDS = ["the", "cat", "sat", "on", "a", "mat", "dog", "ran", "to", "big", "red", "hat"]


def tiny_config():
    return LlamaConfig(
        hidden_size=32, num_hidden_layers=2, vocab_size=50, num_attention_heads=4,
        num_key_value_heads=2, intermediate_size=64, max_position_embeddings=64,
    )


def host_with_attn_graft():
    """Return a tiny host with one attn_heads graft whose w_out is not zero."""
    torch.manual_seed(0)
    host = LlamaForCausalLM(tiny_config()).eval()
    host.config._attn_implementation = "eager"
    donor = LlamaForCausalLM(tiny_config()).eval()
    part = build_attn_heads(donor.config, 1, [0, 3], donor.state_dict())
    graft = Graft(32, 32, part)
    torch.nn.init.normal_(graft.w_out.weight, std=2.0)
    attach_grafts(host, [(0, graft)])
    return host


def left_padded_batch():
    long_row = torch.tensor([[5, 6, 7, 8, 9, 10]])
    short_row = torch.tensor([[11, 12, 13]])
    pad = torch.zeros(1, 3, dtype=torch.long)
    batch = torch.cat([long_row, torch.cat([pad, short_row], dim=1)], dim=0)
    mask = torch.tensor([[1, 1, 1, 1, 1, 1], [0, 0, 0, 1, 1, 1]])
    return long_row, short_row, batch, mask


def test_positional_attention_mask_reaches_attn_graft():
    host = host_with_attn_graft()
    _, short_row, batch, mask = left_padded_batch()
    with torch.no_grad():
        alone = host(input_ids=short_row).logits[0, -1]
        batched = host(batch, mask).logits[1, -1]
    assert torch.allclose(alone, batched, atol=1e-4)


def test_left_padded_generate_matches_single_rows_with_attn_graft():
    host = host_with_attn_graft()
    long_row, short_row, batch, mask = left_padded_batch()
    settings = dict(max_new_tokens=6, do_sample=False, pad_token_id=0, eos_token_id=None)
    with torch.no_grad():
        long_out = host.generate(input_ids=long_row, attention_mask=torch.ones_like(long_row), **settings)
        short_out = host.generate(input_ids=short_row, attention_mask=torch.ones_like(short_row), **settings)
        batch_out = host.generate(input_ids=batch, attention_mask=mask, **settings)
    assert torch.equal(batch_out[0, 6:], long_out[0, 6:])
    assert torch.equal(batch_out[1, 6:], short_out[0, 3:])


def tiny_tokenizer():
    vocab = {"<unk>": 0, "</s>": 1, "<pad>": 2, "<s>": 3}
    for word in WORDS:
        vocab[word] = len(vocab)
    base = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    base.pre_tokenizer = pre_tokenizers.Whitespace()
    base.post_processor = processors.TemplateProcessing(single="<s> $A", special_tokens=[("<s>", 3)])
    return PreTrainedTokenizerFast(
        tokenizer_object=base, unk_token="<unk>", eos_token="</s>", pad_token="<pad>", bos_token="<s>",
    )


def test_fit_w_in_loads_mistral3_donor(tmp_path):
    torch.manual_seed(3)
    text = dict(
        model_type="mistral", hidden_size=32, intermediate_size=64, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, head_dim=8, vocab_size=300,
    )
    vision = dict(
        model_type="pixtral", hidden_size=32, intermediate_size=64, num_hidden_layers=1,
        num_attention_heads=4, image_size=28, patch_size=14, head_dim=8,
    )
    config = transformers.Mistral3Config(text_config=text, vision_config=vision)
    donor = transformers.AutoModelForImageTextToText.from_config(config).eval()
    tokenizer = tiny_tokenizer()
    donor.save_pretrained(tmp_path)
    tokenizer.save_pretrained(tmp_path)

    host = LlamaForCausalLM(tiny_config()).eval()
    texts = ["the cat sat on a mat", "a dog ran to the big red hat", "the red cat ran on the mat"] * 4
    w, r2, r2_dim, pairs = fit_w_in(host, tokenizer, 1, str(tmp_path), 1, texts, "cpu")
    assert w.shape == (32, 32)
    assert pairs > 20
    assert math.isfinite(r2)


def test_composite_gemma3_grafts_rebuild_from_saved_state(tmp_path, monkeypatch):
    """A saved graft from a Gemma 3 composite donor rebuilds to the same output."""
    from tests.test_mixed_compile import local_fetch
    from tests.test_review_compile import tiny_gemma3
    import wsp.assemble
    from wsp.assemble import rebuild_graft
    from wsp.parts import attn_tensor_names, build_mlp_cluster, mlp_tensor_names

    local_fetch(monkeypatch)
    fetch_tensors = wsp.assemble.fetch_tensors
    config, donor = tiny_gemma3(0)
    donor_dir = str(tmp_path / "donor")
    donor.save_pretrained(donor_dir)

    # Layer 0 is a sliding layer, so the attn part uses the local rotary.
    mlp = build_mlp_cluster(config, 1, [2, 5, 11, 40], fetch_tensors(donor_dir, mlp_tensor_names(config, 1)))
    attn = build_attn_heads(config, 0, [1, 2], fetch_tensors(donor_dir, attn_tensor_names(config, 0)))
    metas = [
        {"kind": "mlp_cluster", "donor": donor_dir, "donor_layer": 1, "indices": [2, 5, 11, 40],
         "d_host": 48, "d_donor": 48},
        {"kind": "attn_heads", "donor": donor_dir, "donor_layer": 0, "indices": [1, 2],
         "d_host": 48, "d_donor": 48},
    ]
    built = [Graft(48, 48, mlp), Graft(48, 48, attn)]
    state = {}
    for index, graft in enumerate(built):
        torch.nn.init.normal_(graft.w_out.weight)
        for key, value in graft.state_dict().items():
            state[f"{index}.{key}"] = value

    x = torch.randn(1, 6, 48)
    for index, graft in enumerate(built):
        rebuilt = rebuild_graft(metas[index], state, index)
        own = {}
        for key, value in state.items():
            if key.startswith(f"{index}."):
                own[key[len(f"{index}."):]] = value
        rebuilt.load_state_dict(own, strict=True)
        with torch.no_grad():
            assert torch.allclose(graft(x), rebuilt(x), atol=1e-6)
