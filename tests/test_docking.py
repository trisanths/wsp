import itertools

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import modal_docking as md


def small(seed=0, d_in=12, h=10, d_out=4):
    torch.manual_seed(seed)
    return md.MLP(d_in, h, h, d_out).double()


def test_net2wider_keeps_function_exactly():
    m = small()
    x = torch.randn(7, 12, dtype=torch.float64)
    for k in [2, 3, 4]:
        wide, _ = md.net2wider(m, k, sigma=0.0)
        assert wide.l1.out_features == 10 * k
        assert torch.allclose(wide(x), m(x), atol=1e-10)


def test_noise_breaks_symmetry():
    m = small()
    x = torch.randn(16, 12, dtype=torch.float64)

    def copy_grads(sigma):
        wide, meta = md.net2wider(m, 2, sigma, torch.Generator().manual_seed(1))
        wide(x).pow(2).sum().backward()
        g = wide.l1.weight.grad
        return g[:10], g[10:]

    orig, copy = copy_grads(0.0)
    assert torch.allclose(orig, copy, atol=1e-10)
    orig, copy = copy_grads(0.1)
    assert (orig - copy).abs().max() > 1e-6


def test_hungarian_matches_brute_force():
    rng = np.random.default_rng(0)
    cost = rng.normal(size=(6, 6))
    got = md.hungarian(cost)
    best = min(sum(cost[i, p[i]] for i in range(6)) for p in itertools.permutations(range(6)))
    assert abs(sum(cost[i, got[i]] for i in range(6)) - best) < 1e-9


def test_permute_preserves_function_and_rebasin_recovers_planted_permutation():
    a = small(1, d_in=30, h=24)
    p1 = torch.randperm(24, generator=torch.Generator().manual_seed(3))
    p2 = torch.randperm(24, generator=torch.Generator().manual_seed(4))
    b = md.permute_model(a, p1, p2)
    x = torch.randn(5, 30, dtype=torch.float64)
    assert torch.allclose(a(x), b(x), atol=1e-10)
    assert not torch.allclose(a.l1.weight, b.l1.weight)
    fixed, _ = md.rebasin(a, b)
    for pa, pf in zip(a.parameters(), fixed.parameters()):
        assert torch.allclose(pa, pf, atol=1e-10)


def test_stitch_identity_equals_model():
    m = small()
    stitch = nn.Linear(10, 10).double()
    with torch.no_grad():
        stitch.weight.copy_(torch.eye(10))
        stitch.bias.zero_()
    x = torch.randn(9, 12, dtype=torch.float64)
    for flag in [0, 1]:
        task = torch.full((9,), flag, dtype=torch.long)
        assert torch.allclose(md.stitch_forward(m, m, stitch, x, task), m(x), atol=1e-10)


def test_dock_freezes_originals_and_trains_copies():
    torch.manual_seed(0)
    a = md.MLP(20, 8, 8, 4)
    b = md.MLP(20, 8, 8, 4)
    x = torch.randn(300, 20)
    y = torch.randint(0, 4, (300,))
    mix = md.Mix.__new__(md.Mix)
    mix.x, mix.y = x, y
    mix.src = torch.randint(0, 2, (300,))
    mix.g = torch.Generator().manual_seed(0)
    for mode in ["lin", "reb", "plain"]:
        wide, info = md.dock(a, b, mix, 2, 0.1, mode, 5, torch.Generator().manual_seed(1))
        assert info["frozen_intact"]
        assert info["trainable"] > 0
