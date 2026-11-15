import json
import os

import pytest
import torch
import transformers

import modal_moonshot as mm
from wsp import slim

# config.json of Qwen/Qwen3-235B-A22B-Instruct-2507, read from the Hub on 2026-11-15.
REAL = {
    "hidden_size": 4096, "intermediate_size": 12288, "moe_intermediate_size": 1536, "num_hidden_layers": 94,
    "num_attention_heads": 64, "num_key_value_heads": 4, "head_dim": 128, "vocab_size": 151936,
    "num_experts": 128, "num_experts_per_tok": 8, "tie_word_embeddings": False, "decoder_sparse_step": 1,
    "mlp_only_layers": [], "model_type": "qwen3_moe", "norm_topk_prob": True, "max_position_embeddings": 262144,
    "rope_theta": 5000000, "rms_norm_eps": 1e-06, "hidden_act": "silu", "attention_bias": False,
    "architectures": ["Qwen3MoeForCausalLM"],
}
# model.safetensors.index.json metadata total_size, same date.
REAL_TOTAL_BYTES = 470187269120


def test_sizes_match_the_published_checkpoint():
    row = mm.moe_sizes(REAL, 128)
    assert row["bf16_bytes"] == REAL_TOTAL_BYTES
    assert row["stored"] == 235093634560


def test_sizes_by_keep_and_fit():
    rows = {row["keep"]: row for row in mm.size_table(REAL)}
    assert abs(rows[16]["stored"] - 36.4e9) < 0.2e9
    assert rows[16]["fits_one_h200_bf16"]
    assert rows[16]["fits_one_80gb_fp8"]
    assert not rows[32]["fits_one_h200_bf16"]
    assert not rows[32]["fits_one_80gb_fp8"]
    assert rows[64]["stored"] > rows[48]["stored"] > rows[32]["stored"] > rows[24]["stored"] > rows[16]["stored"]
    assert rows[128]["per_token"] > 21e9
    with pytest.raises(ValueError):
        mm.moe_sizes(REAL, 129)


def tiny(experts=8, top_k=2, layers=3, seed=0):
    torch.manual_seed(seed)
    config = transformers.Qwen3MoeConfig(
        hidden_size=48, intermediate_size=96, moe_intermediate_size=40, num_hidden_layers=layers,
        num_attention_heads=4, num_key_value_heads=2, vocab_size=64, max_position_embeddings=64,
        num_experts=experts, num_experts_per_tok=top_k, decoder_sparse_step=1, mlp_only_layers=[], head_dim=12,
        norm_topk_prob=True)
    return transformers.Qwen3MoeForCausalLM(config).eval()


def test_real_layout_matches_the_adapter_on_meta_weights():
    """One layer of the real 235B config: the tensor names the Hub index lists, and the module layout wsp.slim reads."""
    config = transformers.Qwen3MoeConfig(**{k: v for k, v in REAL.items() if k not in ("architectures", "model_type")})
    config.num_hidden_layers = 1
    with torch.device("meta"):
        model = transformers.Qwen3MoeForCausalLM(config)
    found = slim.moe_layers_of(model)
    assert len(found) == 1
    layer, moe, router, top_k, count = found[0]
    assert (layer, top_k, count) == (0, 8, 128)
    assert slim.family_of(model).name == "qwen3_moe"
    names = list(model.state_dict().keys())
    assert "model.layers.0.mlp.experts.127.down_proj.weight" in names
    assert "model.layers.0.mlp.gate.weight" in names
    expert_names = [n for n in names if mm.EXPERT_RE.match(n)]
    assert len(expert_names) == 128 * 3
    assert sum(1 for n in names if mm.ROUTER_RE.match(n)) == 1
    # Parameter count of this one layer plus embeddings equals the formula.
    one = dict(REAL)
    one["num_hidden_layers"] = 1
    assert sum(p.numel() for p in model.parameters()) == mm.moe_sizes(one, 128)["stored"]


def test_surgery_equals_slim_experts(tmp_path):
    model = tiny()
    source = tmp_path / "src"
    model.save_pretrained(source, max_shard_size="40KB")
    shards = [n for n in os.listdir(source) if n.endswith(".safetensors")]
    assert len(shards) >= 3
    ids = torch.randint(0, 64, (2, 12))
    kept_a = {layer: [1, 2, 5, 7] for layer in range(3)}
    kept_b = {0: [0, 3], 1: [4, 6], 2: [2, 7]}
    out = tmp_path / "out"
    summary = mm.slim_checkpoint(str(source), str(out), {4: kept_a, 2: kept_b}, workers=2)
    config = json.load(open(source / "config.json"))
    for keep, kept in ((4, kept_a), (2, kept_b)):
        loaded = transformers.AutoModelForCausalLM.from_pretrained(out / f"keep_{keep}", dtype=torch.float32).eval()
        reference, report = slim.slim_experts(tiny(), kept)
        reference.eval()
        with torch.no_grad():
            a = loaded(input_ids=ids).logits
            b = reference(input_ids=ids).logits
        assert torch.allclose(a, b, atol=1e-5), float((a - b).abs().max())
        assert summary[keep]["params"] == report["params_total_after"]
        expected = mm.moe_sizes({**config, "head_dim": 12, "moe_intermediate_size": 40, "hidden_size": 48,
                                 "num_attention_heads": 4, "num_key_value_heads": 2, "vocab_size": 64,
                                 "num_hidden_layers": 3, "num_experts": 8, "num_experts_per_tok": 2,
                                 "tie_word_embeddings": False}, keep)
        assert summary[keep]["params"] == expected["stored"]
        saved = json.load(open(out / f"keep_{keep}" / "config.json"))
        assert saved["num_experts"] == keep
        assert saved["num_experts_per_tok"] == min(2, keep)


def test_surgery_resumes_without_rewriting(tmp_path):
    model = tiny()
    source = tmp_path / "src"
    model.save_pretrained(source, max_shard_size="40KB")
    kept = {layer: [0, 1, 2, 3] for layer in range(3)}
    out = tmp_path / "out"
    mm.slim_checkpoint(str(source), str(out), {4: kept}, workers=1)
    first = {n: os.path.getmtime(out / "keep_4" / n) for n in os.listdir(out / "keep_4") if n.endswith(".safetensors")}
    summary = mm.slim_checkpoint(str(source), str(out), {4: kept}, workers=1)
    second = {n: os.path.getmtime(out / "keep_4" / n) for n in os.listdir(out / "keep_4") if n.endswith(".safetensors")}
    assert first == second
    assert summary[4]["params"] > 0


def test_surgery_rejects_a_missing_layer(tmp_path):
    model = tiny()
    source = tmp_path / "src"
    model.save_pretrained(source, max_shard_size="40KB")
    with pytest.raises(ValueError):
        mm.slim_checkpoint(str(source), str(tmp_path / "out"), {2: {0: [0, 1]}}, workers=1)


def test_stats_chunks_add_up_and_keeps_are_nested():
    model = tiny(experts=8, top_k=2, layers=2)
    batch = [torch.randint(0, 64, (16,)) for _ in range(4)]

    class Tok:
        def __call__(self, text, return_tensors=None, truncation=None, max_length=None):
            return {"input_ids": text.unsqueeze(0)}

    whole = slim.collect_moe_stats(model, Tok(), batch, 16)
    a = slim.collect_moe_stats(model, Tok(), batch[:2], 16)
    b = slim.collect_moe_stats(model, Tok(), batch[2:], 16)
    total = mm.add_stats(a, b)
    for layer in whole["layers"]:
        assert torch.allclose(total["layers"][layer]["saliency"], whole["layers"][layer]["saliency"])
        assert torch.equal(total["layers"][layer]["count"], whole["layers"][layer]["count"])
        assert total["layers"][layer]["tokens"] == whole["layers"][layer]["tokens"] == 64
    chosen, shares = mm.keep_sets(total, (6, 4, 2))
    for layer in chosen[2]:
        assert set(chosen[2][layer]) <= set(chosen[4][layer]) <= set(chosen[6][layer])
    assert shares[2]["saliency_mean"] < shares[4]["saliency_mean"] < shares[6]["saliency_mean"] <= 1.0
    with pytest.raises(ValueError):
        mm.add_stats(a, {"layers": {}, "top_k": 2, "num_experts": 3})


def test_cell_selection_fails_loudly():
    assert mm.select_cells("keep_16, keep_32") == ["keep_16", "keep_32"]
    with pytest.raises(ValueError):
        mm.select_cells("")
    with pytest.raises(ValueError):
        mm.select_cells("keep_17")
    assert mm.CELLS["keep_16"][2] == 1 and mm.CELLS["full_235b"][2] == 8


def test_helpers():
    assert mm.parse_load_gib("INFO Model loading took 68.25 GiB and 12 seconds") == 68.25
    assert mm.parse_load_gib("nothing") is None
    assert mm.dollars(3600, 8) == 36.0
    assert mm.spend([{"dollars": 1.5}, {"dollars": None}, {}]) == 1.5
    maps = mm.keep_maps({0: [5, 2, 7]})
    assert maps[0] == {2: 0, 5: 1, 7: 2}
    assert mm.surgery_name("model.layers.0.mlp.experts.3.up_proj.weight", maps) == ("drop", None)
    assert mm.surgery_name("model.layers.0.mlp.experts.5.up_proj.weight", maps) == (
        "rename", "model.layers.0.mlp.experts.1.up_proj.weight")
    assert mm.surgery_name("model.layers.0.mlp.gate.weight", maps)[2] == [2, 5, 7]
    assert mm.surgery_name("model.embed_tokens.weight", maps) == ("copy", "model.embed_tokens.weight")
