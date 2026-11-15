import pytest
import torch
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

from wsp.align import match_offsets, ridge_fit
from wsp.extract import random_neurons, top_activation_neurons
from wsp.heal import heal

WORDS = ["the", "cat", "sat", "on", "a", "mat", "dog", "ran", "to", "big", "red", "hat"]


def tiny_model():
    config = LlamaConfig(
        hidden_size=32, num_hidden_layers=2, vocab_size=300, num_attention_heads=4,
        num_key_value_heads=2, intermediate_size=64, max_position_embeddings=64,
    )
    torch.manual_seed(0)
    return LlamaForCausalLM(config).eval()


def tiny_tokenizer():
    vocab = {"<unk>": 0, "</s>": 1, "<pad>": 2}
    for word in WORDS:
        vocab[word] = len(vocab)
    base = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    base.pre_tokenizer = pre_tokenizers.Whitespace()
    return PreTrainedTokenizerFast(
        tokenizer_object=base, unk_token="<unk>", eos_token="</s>", pad_token="<pad>"
    )


def test_match_offsets():
    assert match_offsets([3, 7, 12], [3, 5, 7, 12]) == [(0, 0), (1, 2), (2, 3)]
    assert match_offsets([], [1, 2]) == []
    assert match_offsets([1, 2], []) == []


def test_ridge_recovers_linear_map():
    torch.manual_seed(0)
    x = torch.randn(500, 16)
    a = torch.randn(12, 16)
    y = x @ a.T
    w, r2, r2_dim = ridge_fit(x, y, lam=1e-5)
    assert w.shape == (12, 16)
    assert torch.allclose(w, a, atol=1e-2)
    assert r2 > 0.99
    assert r2_dim > 0.99


def test_ridge_too_few_rows_raises():
    with pytest.raises(ValueError):
        ridge_fit(torch.randn(10, 4), torch.randn(10, 3))


def test_heal_reduces_loss_and_touches_only_params():
    model = tiny_model()
    tok = tiny_tokenizer()
    for p in model.parameters():
        p.requires_grad_(False)
    head = model.lm_head.weight
    head.requires_grad_(True)
    model.config.tie_word_embeddings = False
    embed_before = model.model.embed_tokens.weight.detach().clone()
    head_before = head.detach().clone()
    texts = [
        "the cat sat on a mat",
        "a dog ran to the hat",
        "the big red hat",
        "a cat ran to a mat",
    ]
    losses = heal(model, tok, texts, [head], steps=30, lr=1e-2, batch_size=4, seq_len=16,
                  log_every=0)
    assert len(losses) == 30
    assert losses[-1] < losses[0]
    assert torch.equal(model.model.embed_tokens.weight, embed_before)
    assert not torch.equal(head, head_before)


def test_top_activation_neurons_picks_large():
    model = tiny_model()
    tok = tiny_tokenizer()
    mlp = model.model.layers[0].mlp
    with torch.no_grad():
        mlp.up_proj.weight[[3, 5]] *= 50.0
    texts = ["the cat sat on a mat", "a dog ran to the hat", "the big red hat"]
    top = top_activation_neurons(model, tok, texts, mlp, 4)
    assert len(top) == 4
    assert 3 in top.tolist()
    assert 5 in top.tolist()


def test_random_neurons_seeded():
    a = random_neurons(64, 10, seed=1)
    b = random_neurons(64, 10, seed=1)
    assert torch.equal(a, b)
    assert len(a) == 10
    assert len(set(a.tolist())) == 10
    assert torch.equal(a, torch.sort(a).values)
