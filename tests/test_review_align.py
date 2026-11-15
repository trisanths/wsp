"""Regression tests for review fixes in align.py, graft.py, and fetch.py."""

import torch
from tokenizers import Tokenizer, models, pre_tokenizers, processors
from torch import nn
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

from wsp import fetch
from wsp.align import alignment_tokens, collect_pairs, layer_input_capture, ridge_fit
from wsp.graft import Graft, attach_grafts
from wsp.parts import AttnHeadsPart, build_attn_heads

WORDS = ["the", "cat", "sat", "on", "a", "mat", "dog", "ran", "to", "big", "red", "hat"]
TEXTS = ["the cat sat on a mat", "a dog ran to the big red hat", "the red cat ran"]


def tiny_config():
    return LlamaConfig(
        hidden_size=32, num_hidden_layers=2, vocab_size=300, num_attention_heads=4,
        num_key_value_heads=2, intermediate_size=64, max_position_embeddings=64,
    )


def tiny_model():
    torch.manual_seed(0)
    return LlamaForCausalLM(tiny_config()).eval()


def tiny_tokenizer(with_bos):
    vocab = {"<unk>": 0, "</s>": 1, "<pad>": 2, "<s>": 3}
    for word in WORDS:
        vocab[word] = len(vocab)
    base = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    base.pre_tokenizer = pre_tokenizers.Whitespace()
    if with_bos:
        base.post_processor = processors.TemplateProcessing(
            single="<s> $A", special_tokens=[("<s>", 3)]
        )
    return PreTrainedTokenizerFast(
        tokenizer_object=base, unk_token="<unk>", eos_token="</s>", pad_token="<pad>",
        bos_token="<s>",
    )


class Recorder(nn.Module):
    """Stands in for a graft: records its input and returns it unchanged."""

    def __init__(self):
        super().__init__()
        self.seen = []

    def forward(self, h):
        self.seen.append(h.detach().clone())
        return h


def test_host_rows_match_graft_hook_at_last_layer():
    model = tiny_model()
    tok = tiny_tokenizer(with_bos=True)
    last = model.config.num_hidden_layers - 1
    capture = layer_input_capture(model.model.layers[0])
    x, _ = collect_pairs(model, tok, last, model, tok, capture, TEXTS[:1])

    ids, positions, _ = alignment_tokens(tok, TEXTS[0])
    recorder = Recorder()
    attach_grafts(model, [(last, recorder)])
    with torch.no_grad():
        out = model(ids, output_hidden_states=True)
    seen = recorder.seen[0][0][torch.tensor(positions)]
    assert torch.allclose(x, seen, atol=1e-6)

    # hidden_states[-1] is after the final norm, so it is not what the graft sees.
    normed = out.hidden_states[last + 1][0][torch.tensor(positions)]
    assert not torch.allclose(x, normed, atol=1e-3)


def test_bos_rows_excluded():
    tok = tiny_tokenizer(with_bos=True)
    ids, positions, ends = alignment_tokens(tok, "the cat sat")
    assert ids[0, 0].item() == 3
    assert positions == [1, 2, 3]
    assert ends == [3, 7, 11]


def test_first_token_dropped_without_bos():
    tok = tiny_tokenizer(with_bos=False)
    ids, positions, ends = alignment_tokens(tok, "the cat sat")
    assert ids.shape == (1, 3)
    assert positions == [1, 2]
    assert ends == [7, 11]


def test_collect_pairs_runs_with_bos_and_skips_it():
    model = tiny_model()
    host_tok = tiny_tokenizer(with_bos=True)
    donor_tok = tiny_tokenizer(with_bos=False)
    capture = layer_input_capture(model.model.layers[1])
    x, y = collect_pairs(model, host_tok, 0, model, donor_tok, capture, TEXTS)
    # Each text gives its word count minus one: the host skips BOS and the
    # donor (no BOS) skips its first word, so the first word never pairs.
    expected = 0
    for text in TEXTS:
        expected += len(text.split()) - 1
    assert x.shape == (expected, 32)
    assert y.shape == (expected, 32)


def test_per_dim_r2_lower_than_pooled_when_one_dim_dominates():
    torch.manual_seed(0)
    n = 2000
    x = torch.randn(n, 8)
    y = torch.randn(n, 6)
    # Dim 0 is a large exact function of x; the other dims are noise.
    y[:, 0] = 1000.0 * x[:, 0]
    w, r2, r2_dim = ridge_fit(x, y, lam=1e-6)
    assert w.shape == (6, 8)
    assert r2 > 0.99
    assert r2_dim < 0.3


def test_per_dim_r2_matches_pooled_for_exact_map():
    torch.manual_seed(1)
    x = torch.randn(500, 16)
    a = torch.randn(12, 16)
    w, r2, r2_dim = ridge_fit(x, x @ a.T, lam=1e-6)
    assert r2 > 0.99
    assert r2_dim > 0.99


def test_host_pre_hook_stores_mask_on_grafts():
    model = tiny_model()
    first = Recorder()
    second = Recorder()
    handles = attach_grafts(model, [(0, first), (1, second)])
    assert len(handles) == 3
    ids = torch.randint(4, 15, (2, 5))
    mask = torch.tensor([[0, 0, 1, 1, 1], [1, 1, 1, 1, 1]])
    with torch.no_grad():
        model(input_ids=ids, attention_mask=mask)
    assert torch.equal(first.pad_mask, mask)
    assert torch.equal(second.pad_mask, mask)
    with torch.no_grad():
        model(input_ids=ids)
    assert first.pad_mask is None
    assert second.pad_mask is None


def attn_graft(config, model):
    state = {}
    for name, value in model.state_dict().items():
        state[name] = value.detach().clone()
    part = build_attn_heads(config, 1, [0, 3], state)
    graft = Graft(config.hidden_size, config.hidden_size, part)
    torch.manual_seed(3)
    nn.init.normal_(graft.w_out.weight, std=0.5)
    return graft


def test_use_cache_false_after_attaching_attn_heads():
    host = tiny_model()
    donor_config = tiny_config()
    donor = LlamaForCausalLM(donor_config).eval()
    graft = attn_graft(donor_config, donor)
    assert isinstance(graft.part, AttnHeadsPart)
    assert host.config.use_cache
    attach_grafts(host, [(0, graft)])
    assert host.config.use_cache is False
    assert host.generation_config.use_cache is False


def test_use_cache_kept_for_non_attn_grafts():
    host = tiny_model()
    attach_grafts(host, [(0, Recorder())])
    assert host.config.use_cache


def test_graft_passes_pad_mask_to_attn_part():
    donor_config = tiny_config()
    donor = LlamaForCausalLM(donor_config).eval()
    graft = attn_graft(donor_config, donor)
    calls = []
    original = graft.part.forward

    def spy(x, attention_mask=None):
        calls.append(attention_mask)
        return original(x, attention_mask=attention_mask)

    graft.part.forward = spy
    mask = torch.tensor([[0, 1, 1]])
    graft.pad_mask = mask
    with torch.no_grad():
        graft(torch.randn(1, 3, 32))
    assert calls[0] is mask


def test_fetch_uses_stored_hf_token(monkeypatch):
    import huggingface_hub

    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setattr(huggingface_hub, "get_token", lambda: "stored-token")
    monkeypatch.setattr(fetch, "_shard_map", lambda repo_id, names, revision: {"model.safetensors": list(names)})
    seen = []

    def fake_reader(url, token=None):
        seen.append(token)
        return None

    monkeypatch.setattr(fetch, "http_range_reader", fake_reader)
    monkeypatch.setattr(fetch, "read_tensors", lambda read_range, names: {})
    fetch.fetch_tensors("org/repo", ["a"])
    assert seen == ["stored-token"]


def test_left_padded_row_matches_unpadded_with_attn_graft():
    host = tiny_model()
    donor_config = tiny_config()
    torch.manual_seed(5)
    donor = LlamaForCausalLM(donor_config).eval()
    attach_grafts(host, [(0, attn_graft(donor_config, donor))])
    host.config._attn_implementation = "eager"
    ids = torch.tensor([[5, 6, 7, 8, 9]])
    padded = torch.cat([torch.full((1, 3), 2), ids], dim=1)
    mask = torch.cat([torch.zeros(1, 3, dtype=torch.long), torch.ones(1, 5, dtype=torch.long)], dim=1)
    with torch.no_grad():
        alone = host(input_ids=ids, attention_mask=torch.ones_like(ids)).logits[0, -1]
        batched = host(input_ids=padded, attention_mask=mask).logits[0, -1]
    assert torch.allclose(alone, batched, atol=1e-4)
