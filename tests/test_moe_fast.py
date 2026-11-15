import pytest
import torch

transformers = pytest.importorskip("transformers")
if not hasattr(transformers, "Qwen3MoeConfig"):
    pytest.skip("no Qwen3 MoE in this transformers", allow_module_level=True)

from wsp import moe_fast


def tiny_config(layers=4):
    config = transformers.Qwen3MoeConfig(
        hidden_size=48, intermediate_size=96, moe_intermediate_size=24, num_hidden_layers=layers,
        num_attention_heads=4, num_key_value_heads=2, head_dim=12, num_experts=8,
        num_experts_per_tok=2, decoder_sparse_step=1, mlp_only_layers=[], norm_topk_prob=True,
        vocab_size=300, max_position_embeddings=128, tie_word_embeddings=False)
    config._attn_implementation = "eager"
    return config


def tiny_model(dtype):
    torch.manual_seed(3)
    return transformers.Qwen3MoeForCausalLM(tiny_config()).eval().to(dtype)


@pytest.mark.parametrize("dtype,tol", [(torch.float32, 1e-4), (torch.bfloat16, 0.1)])
def test_fast_block_matches_native(dtype, tol):
    model = tiny_model(dtype)
    block = model.model.layers[1].mlp
    fast = moe_fast.stack_native(block)
    x = torch.randn(3, 11, 48).to(dtype)
    with torch.no_grad():
        want, want_logits = block(x)
        got, got_logits = fast(x)
    assert (want.float() - got.float()).abs().max() < tol
    assert torch.equal(want_logits, got_logits)


def test_fast_block_backward_reaches_input():
    model = tiny_model(torch.float32)
    fast = moe_fast.stack_native(model.model.layers[0].mlp)
    x = torch.randn(2, 7, 48, requires_grad=True)
    out, _ = fast(x)
    out.sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert fast.w_gate.grad is None


def test_loader_matches_native_logits(tmp_path):
    model = tiny_model(torch.float32)
    model.save_pretrained(tmp_path, max_shard_size="40KB")
    loaded, stats = moe_fast.load_qwen3_moe(str(tmp_path), "cpu", dtype=torch.float32, log=lambda *_: None)
    assert stats["shards"] > 1
    ids = torch.randint(0, 300, (2, 9))
    with torch.no_grad():
        want = model(input_ids=ids).logits
        got = loaded(input_ids=ids).logits
    assert (want - got).abs().max() < 1e-4


def test_loader_bf16_and_slice_only(tmp_path):
    model = tiny_model(torch.bfloat16)
    model.save_pretrained(tmp_path, max_shard_size="40KB")
    loaded, _ = moe_fast.load_qwen3_moe(str(tmp_path), "cpu", dtype=torch.bfloat16, layers=[1, 2],
                                         log=lambda *_: None)
    assert isinstance(loaded.model.layers[0], moe_fast.Skipped)
    part = moe_fast.block_part(loaded, 1, 2)
    x = torch.randn(2, 6, 48).to(torch.bfloat16)
    ref = moe_fast.block_part(model_with_fast(model), 1, 2)
    with torch.no_grad():
        got = part(x)
        want = ref(x)
    assert (got.float() - want.float()).abs().max() < 0.1
    run, stored = moe_fast.active_params(part, 2, 8)
    assert run < stored


def model_with_fast(model):
    moe_fast.swap_native_blocks(model)
    return model


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_graft_with_shared_fast_part_trains(dtype):
    from wsp import pgraft
    from tests.test_align import tiny_tokenizer
    from tests.test_pgraft import TEXTS, tiny_host
    from wsp.stitch import encode

    donor = model_with_fast(tiny_model(dtype))
    for p in donor.parameters():
        p.requires_grad_(False)
    base = tiny_host().to(dtype)
    for p in base.parameters():
        p.requires_grad_(False)
    part = moe_fast.block_part(donor, 2, 3)
    graft = pgraft.build(base, donor, 1, 2, 3, enc_depth=2, part=part)
    pgraft.attach(base, [graft])
    tok = tiny_tokenizer()
    pgraft.set_sink(graft, donor, tok, 2, TEXTS)
    graft.dec.up.weight.data.normal_(0, 0.02)
    ids, mask, _, _ = encode(tok, TEXTS[:3], 32)
    out = base(input_ids=ids, attention_mask=mask).logits
    out.float().sum().backward()
    assert graft.enc.proj.weight.grad is not None
    assert all(p.grad is None for p in part.parameters())
