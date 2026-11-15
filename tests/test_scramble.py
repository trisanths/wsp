"""wsp.scramble on small tensors and a tiny trained model (CPU)."""

import copy

import pytest
import torch
from torch import nn

from wsp import scramble


def make_module(seed=0):
    torch.manual_seed(seed)
    m = nn.ModuleDict({
        "proj": nn.Linear(24, 40),
        "wide": nn.Linear(60, 16, bias=False),
        "norm": nn.LayerNorm(24),
    })
    m["norm"].weight.data = torch.rand(24) + 0.5
    return m


def cos(a, b):
    return float(a.flatten() @ b.flatten() / (a.norm() * b.norm()))


@pytest.mark.parametrize("mode", ["spectrum", "permute"])
def test_shapes_and_norms_are_kept(mode):
    m = make_module()
    before = {k: v.clone() for k, v in m.state_dict().items()}
    report = scramble.scramble_module(m, 0, mode)
    for k, v in m.state_dict().items():
        assert v.shape == before[k].shape
        assert float(v.norm()) == pytest.approx(float(before[k].norm()), rel=1e-4)
    assert report["tensors_changed"] == 5
    assert report["rel_fro_error"] < 1e-4
    assert report["seconds"] >= 0.0


def test_spectrum_mode_keeps_singular_values():
    m = make_module()
    before = {k: torch.linalg.svdvals(v.double()) for k, v in m.state_dict().items() if v.dim() == 2}
    scramble.scramble_module(m, 3, "spectrum")
    for k, v in m.state_dict().items():
        if v.dim() == 2:
            assert torch.allclose(torch.linalg.svdvals(v.double()), before[k], atol=1e-4, rtol=1e-4)


def test_permute_mode_destroys_the_spectrum_but_keeps_values():
    m = make_module()
    old = m["proj"].weight.detach().clone()
    scramble.scramble_module(m, 0, "permute")
    new = m["proj"].weight.detach()
    assert torch.equal(torch.sort(new.flatten()).values, torch.sort(old.flatten()).values)
    assert not torch.allclose(torch.linalg.svdvals(new), torch.linalg.svdvals(old), atol=1e-3)


@pytest.mark.parametrize("mode", ["spectrum", "permute"])
def test_result_is_far_from_the_original(mode):
    m = make_module()
    old = {k: v.clone() for k, v in m.state_dict().items() if v.dim() == 2}
    report = scramble.scramble_module(m, 1, mode)
    for k, v in m.state_dict().items():
        if v.dim() == 2:
            assert abs(cos(v, old[k])) < 0.15
    assert report["mean_abs_cosine"] < 0.2


def test_seeded_and_reproducible_and_seeds_differ():
    a, b, c = make_module(), make_module(), make_module()
    scramble.scramble_module(a, 7)
    scramble.scramble_module(b, 7)
    scramble.scramble_module(c, 8)
    assert torch.equal(a["proj"].weight, b["proj"].weight)
    assert torch.equal(a["norm"].weight, b["norm"].weight)
    assert not torch.equal(a["proj"].weight, c["proj"].weight)
    assert not torch.equal(a["norm"].weight, c["norm"].weight)


def test_one_dimensional_tensors_keep_their_sorted_values():
    m = make_module()
    old_norm = m["norm"].weight.detach().clone()
    old_bias = m["proj"].bias.detach().clone()
    scramble.scramble_module(m, 2)
    assert torch.equal(torch.sort(m["norm"].weight.detach()).values, torch.sort(old_norm).values)
    assert torch.equal(torch.sort(m["proj"].bias.detach()).values, torch.sort(old_bias).values)
    assert not torch.equal(m["norm"].weight.detach(), old_norm)


def test_embeddings_are_not_touched_unless_asked():
    torch.manual_seed(0)
    m = nn.ModuleDict({"embed": nn.Embedding(20, 8), "lin": nn.Linear(8, 8)})
    old = m["embed"].weight.detach().clone()
    report = scramble.scramble_module(m, 0)
    assert torch.equal(m["embed"].weight.detach(), old)
    assert report["tensors_skipped"] == 1
    scramble.scramble_module(m, 0, include_embeddings=True)
    assert not torch.equal(m["embed"].weight.detach(), old)


def test_large_matrices_use_the_fallback_mode():
    m = make_module()
    report = scramble.scramble_module(m, 0, "spectrum", max_numel=500)
    assert report["modes_used"].get("permute", 0) >= 2
    assert report["modes_used"].get("spectrum", 0) == 0


def test_scramble_tensors_dict_and_dtype():
    tensors = {"layers.0.q.weight": torch.randn(16, 12).bfloat16(), "layers.0.norm.weight": torch.rand(12),
               "embed_tokens.weight": torch.randn(10, 12)}
    keep = tensors["embed_tokens.weight"].clone()
    report = scramble.scramble_tensors(tensors, 0)
    assert tensors["layers.0.q.weight"].dtype == torch.bfloat16
    assert torch.equal(tensors["embed_tokens.weight"], keep)
    assert report["tensors_changed"] == 2


def test_tiny_trained_model_gets_worse_when_a_block_is_scrambled():
    transformers = pytest.importorskip("transformers")
    torch.manual_seed(0)
    config = transformers.LlamaConfig(hidden_size=48, intermediate_size=96, num_hidden_layers=3,
                                      num_attention_heads=4, num_key_value_heads=2, vocab_size=40,
                                      max_position_embeddings=32)
    config._attn_implementation = "eager"
    model = transformers.LlamaForCausalLM(config)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    gen = torch.Generator().manual_seed(0)

    def make_batch(n):
        rows = []
        for _ in range(n):
            cur = int(torch.randint(0, 20, (1,), generator=gen))
            seq = [cur]
            for _ in range(11):
                cur = (cur * 3 + 1) % 20
                seq.append(cur + 5)
            seq[0] += 5
            rows.append(seq)
        return torch.tensor(rows)

    for step in range(250):
        ids = make_batch(16)
        loss = model(input_ids=ids, labels=ids).loss
        opt.zero_grad()
        loss.backward()
        opt.step()
    model.eval()
    held = make_batch(32)

    def lm_loss(m):
        with torch.no_grad():
            return float(m(input_ids=held, labels=held).loss)

    real = lm_loss(model)
    scrambled = copy.deepcopy(model)
    report = scramble.scramble_module(scrambled.model.layers, 0)
    again = copy.deepcopy(model)
    scramble.scramble_module(again.model.layers, 0)
    fake = lm_loss(scrambled)
    assert report["tensors_changed"] > 0
    assert real < 1.0
    assert fake > real + 0.5
    assert fake == pytest.approx(lm_loss(again))
