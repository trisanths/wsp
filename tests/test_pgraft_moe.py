import pytest
import torch

transformers = pytest.importorskip("transformers")
if not hasattr(transformers, "Qwen3MoeConfig"):
    pytest.skip("no Qwen3 MoE in this transformers", allow_module_level=True)

from wsp import pgraft
from wsp.parts import causal_mask
from wsp.stitch import encode
from tests.test_align import tiny_tokenizer
from tests.test_pgraft import TEXTS, tiny_host


def tiny_moe_donor():
    torch.manual_seed(3)
    config = transformers.Qwen3MoeConfig(
        hidden_size=48, intermediate_size=96, moe_intermediate_size=24, num_hidden_layers=6,
        num_attention_heads=4, num_key_value_heads=2, head_dim=12, num_experts=8,
        num_experts_per_tok=2, decoder_sparse_step=1, mlp_only_layers=[], norm_topk_prob=True,
        vocab_size=300, max_position_embeddings=128)
    config._attn_implementation = "eager"
    model = transformers.Qwen3MoeForCausalLM(config).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def build(dtype):
    base = tiny_host().to(dtype)
    for p in base.parameters():
        p.requires_grad_(False)
    donor = tiny_moe_donor().to(dtype)
    tok = tiny_tokenizer()
    graft = pgraft.build(base, donor, host_layer=1, first=2, last=4, enc_depth=2)
    pgraft.attach(base, [graft])
    pgraft.set_sink(graft, donor, tok, 2, TEXTS)
    return base, donor, tok, graft


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_moe_slice_matches_native_and_trains(dtype):
    base, donor, tok, graft = build(dtype)
    ids, mask, _, _ = encode(tok, TEXTS[:3], 32)
    x = torch.randn(ids.shape[0], ids.shape[1], 48).to(dtype)
    graft.pad_mask = mask
    with torch.no_grad():
        block = graft.contribution(x) + x
        h = x
        pos = torch.arange(ids.shape[1])[None].expand(ids.shape[0], -1)
        cos_sin = donor.model.rotary_emb(h, pos)
        m = causal_mask(ids.shape[0], ids.shape[1], dtype, h.device, None, 0)
        for i in (2, 3, 4):
            h = donor.model.layers[i](h, attention_mask=m, position_ids=pos, position_embeddings=cos_sin)
            h = h[0] if isinstance(h, tuple) else h
    tol = 1e-4 if dtype == torch.float32 else 0.1
    real = mask.bool()[..., None]
    assert ((block.float() - h.float()) * real).abs().max() < tol
    graft.dec.up.weight.data.normal_(0, 0.02)
    out = base(input_ids=ids, attention_mask=mask).logits
    assert torch.isfinite(out.float()).all()
    out.float().sum().backward()
    assert graft.enc.proj.weight.grad is not None
