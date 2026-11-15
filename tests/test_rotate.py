import copy

import pytest
import torch
import transformers

from wsp.rotate import (fold_norms, procrustes, random_orthogonal, residual_report, rotate_residual)


def tiny(tie=True, seed=0):
    torch.manual_seed(seed)
    config = transformers.Qwen2Config(
        vocab_size=97, hidden_size=32, intermediate_size=64, num_hidden_layers=3,
        num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=64,
        tie_word_embeddings=tie, torch_dtype="float32")
    model = transformers.Qwen2ForCausalLM(config).float().eval()
    with torch.no_grad():
        for p in model.parameters():
            if p.dim() == 1:
                p.copy_(1 + 0.5 * torch.randn_like(p) if p.shape[0] == 32 else 0.3 * torch.randn_like(p))
    return model


def logits(model):
    ids = torch.randint(0, 97, (2, 11), generator=torch.Generator().manual_seed(1))
    with torch.no_grad():
        return model(ids).logits


@pytest.mark.parametrize("tie", [True, False])
def test_fold_norms_keeps_logits(tie):
    model = tiny(tie)
    before = logits(model)
    fold_norms(model)
    after = logits(model)
    assert (before - after).abs().max() < 1e-5
    assert torch.all(model.model.norm.weight == 1)
    assert model.lm_head.weight.data_ptr() != model.model.embed_tokens.weight.data_ptr()


def test_rotate_residual_keeps_logits():
    model = tiny()
    before = logits(model)
    fold_norms(model)
    rotate_residual(model, random_orthogonal(32, seed=3))
    assert (before - logits(model)).abs().max() < 1e-4


def test_rotate_needs_folded_norms():
    with pytest.raises(ValueError):
        rotate_residual(tiny(), random_orthogonal(32))


def test_rotation_moves_embedding():
    model = tiny()
    fold_norms(model)
    old = model.model.embed_tokens.weight.detach().clone()
    R = random_orthogonal(32, seed=5)
    rotate_residual(model, R)
    assert torch.allclose(model.model.embed_tokens.weight, (old.double() @ R).float(), atol=1e-5)


def test_procrustes_recovers_planted_rotation():
    A = torch.randn(500, 16, dtype=torch.float64)
    R = random_orthogonal(16, seed=2)
    got = procrustes(A, A @ R)
    assert torch.allclose(got, R, atol=1e-9)
    assert torch.allclose(got.T @ got, torch.eye(16, dtype=torch.float64), atol=1e-9)


def test_residual_report_planted():
    gen = torch.Generator().manual_seed(0)
    E_i = torch.randn(2000, 24, generator=gen, dtype=torch.float64)
    R_true = random_orthogonal(24, seed=4)
    E_m = E_i @ R_true.T
    report = residual_report(E_m, E_i, procrustes(E_m, E_i))
    assert report["rotated"] < 1e-9
    assert 1.2 < report["unrotated"] < 1.6
    assert report["shuffled"] > 0.9
    assert report["random"] > 1.2
    noisy = E_m + 0.3 * torch.randn(2000, 24, generator=gen, dtype=torch.float64)
    report = residual_report(noisy, E_i, procrustes(noisy, E_i))
    assert 0.2 < report["rotated"] < 0.4
