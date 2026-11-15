"""Cached generation of a parallel graft equals the uncached path (donor branch keeps its own KV cache)."""
import pytest
import torch

transformers = pytest.importorskip("transformers")

from wsp import pgraft
from wsp.graft import disable_cache
from tests.test_pgraft import TEXTS, setup


def _live(graft):
    """Give the fresh graft a nonzero write, so the donor branch changes the output."""
    torch.manual_seed(5)
    for p in list(graft.dec.parameters()) + list(graft.enc.parameters()):
        p.data.add_(0.05 * torch.randn_like(p))
    graft.token_gate.weight.data.normal_(0, 0.5)


def _gen(model, tok, texts, new=6, cached=True):
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    enc = tok(texts, return_tensors="pt", padding=True)
    model.generation_config.use_cache = cached
    model.config.use_cache = cached
    with torch.no_grad():
        out = model.generate(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"], max_new_tokens=new,
                             min_new_tokens=new, do_sample=False, pad_token_id=tok.pad_token_id,
                             output_scores=True, return_dict_in_generate=True)
    return out


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_cached_generate_matches_uncached_two_batches_two_calls(dtype):
    base, donor, tok, graft = setup()
    _live(graft)
    if dtype == torch.bfloat16:
        base.to(dtype)
        for g in pgraft.grafts_of(base):
            g.part.to(dtype)  # as on the GPU: bf16 host and donor slice, float32 glue
            g.w_in.float(), g.w_out.float(), g.token_gate.float()
    batches = [TEXTS[:3], TEXTS[3:8]]  # different lengths per row, so left padding differs
    for texts in batches:
        a = _gen(base, tok, texts, cached=True)
        b = _gen(base, tok, texts, cached=False)
        assert torch.equal(a.sequences, b.sequences)
        tol = 1e-4 if dtype == torch.float32 else 0.25
        for sa, sb in zip(a.scores, b.scores):
            assert torch.allclose(sa.float(), sb.float(), atol=tol)
    # two cached calls in a row on the same batch give the same answer (no stale donor cache)
    first = _gen(base, tok, batches[1], cached=True).sequences
    second = _gen(base, tok, batches[1], cached=True).sequences
    assert torch.equal(first, second)
    assert graft.part.cache is not None, "the donor branch kept a KV cache during cached generation"


def test_sparse_span_runs_every_second_layer_and_caches_like_the_uncached_path():
    base, donor, tok, graft = setup(first=1, last=5)
    assert len(graft.part.layers) == 5
    dense = pgraft.build(base, donor, host_layer=1, first=1, last=5, enc_depth=2)
    sparse = pgraft.build(base, donor, host_layer=1, first=1, last=5, enc_depth=2, keep=[1, 3, 5])
    assert len(sparse.part.layers) == 3
    assert [l.self_attn.layer_idx for l in sparse.part.layers] == [0, 1, 2]
    assert sum(p.numel() for p in sparse.part.parameters()) * 5 == sum(p.numel() for p in dense.part.parameters()) * 3
    # the kept layers are the donor's layers 1, 3, 5 (shared weights, not copies of other layers)
    assert torch.equal(sparse.part.layers[1].mlp.up_proj.weight, donor.model.layers[3].mlp.up_proj.weight)
    # sparse graft: cached generation equals uncached
    base2 = setup()[0]
    g = pgraft.build(base2, donor, host_layer=1, first=1, last=5, enc_depth=2, keep=[1, 3, 5])
    pgraft.attach(base2, [g])
    pgraft.set_sink(g, donor, tok, 1, TEXTS)
    _live(g)
    a = _gen(base2, tok, TEXTS[:4], cached=True)
    b = _gen(base2, tok, TEXTS[:4], cached=False)
    assert torch.equal(a.sequences, b.sequences)
    with pytest.raises(ValueError):
        pgraft.build(base, donor, host_layer=1, first=1, last=5, keep=[0, 2])


def test_sparse_span_forward_runs_in_bf16():
    base = setup()[0]
    donor, tok = tiny_donor_and_tok()
    g = pgraft.build(base, donor, host_layer=1, first=1, last=5, enc_depth=2, keep=[1, 3, 5])
    pgraft.attach(base, [g])
    pgraft.set_sink(g, donor, tok, 1, TEXTS)
    _live(g)
    base.to(torch.bfloat16)
    g.part.to(torch.bfloat16)
    g.w_in.float(), g.w_out.float(), g.token_gate.float()
    out = _gen(base, tok, TEXTS[:3], cached=True)
    assert all(not torch.isnan(s.float()).any() for s in out.scores)


def tiny_donor_and_tok():
    from tests.test_align import tiny_tokenizer
    from tests.test_pgraft import tiny_donor
    return tiny_donor(), tiny_tokenizer()
