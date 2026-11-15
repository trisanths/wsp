import json
import os

import pytest
import torch
from safetensors.torch import load_file
from transformers import LlamaConfig, LlamaForCausalLM, MixtralConfig, MixtralForCausalLM
from transformers import Qwen3MoeConfig, Qwen3MoeForCausalLM

from wsp import Catalog, Program, load
from wsp.extract import random_neurons, top_routed_experts
from wsp.fetch import fetch_config, local_range_reader, read_tensors
from wsp.parts import build_expert_part, expert_tensor_names, find_moe_block
from tests.test_align import tiny_tokenizer
from tests.test_mixed_compile import TEXTS, local_fetch, logits


def tiny_qwen3_moe():
    return Qwen3MoeForCausalLM(Qwen3MoeConfig(
        hidden_size=48, intermediate_size=96, moe_intermediate_size=40, num_hidden_layers=3,
        num_attention_heads=4, num_key_value_heads=2, vocab_size=300, max_position_embeddings=64,
        num_experts=6, num_experts_per_tok=2, decoder_sparse_step=1, mlp_only_layers=[],
        head_dim=12,
    ))


def tiny_mixtral():
    return MixtralForCausalLM(MixtralConfig(
        hidden_size=48, intermediate_size=72, num_hidden_layers=3, num_attention_heads=4,
        num_key_value_heads=2, vocab_size=300, max_position_embeddings=64,
        num_local_experts=5, num_experts_per_tok=2,
    ))


MAKERS = [("qwen3_moe", tiny_qwen3_moe), ("mixtral", tiny_mixtral)]


def save_donor(tmp_path, maker):
    torch.manual_seed(0)
    donor = maker()
    donor_dir = str(tmp_path / "donor")
    donor.save_pretrained(donor_dir)
    tiny_tokenizer().save_pretrained(donor_dir)
    return donor, donor_dir


def part_from_disk(donor_dir, layer, experts):
    config = fetch_config(donor_dir)
    names = expert_tensor_names(config, layer, experts)
    tensors = read_tensors(local_range_reader(os.path.join(donor_dir, "model.safetensors")), names)
    return config, build_expert_part(config, layer, experts, tensors)


def moe_of(donor, layer):
    layer_module = donor.model.layers[layer]
    if hasattr(layer_module, "feed_forward"):
        return layer_module.feed_forward
    return layer_module.mlp


def donor_experts(moe, normed, experts):
    """Return the sum of the given donor experts on normed, each with weight 1.

    Transformers 5 keeps the experts in one fused module, so we route every
    token to each expert in turn with the donor's own experts code.
    """
    flat = normed.reshape(-1, normed.shape[-1])
    total = torch.zeros_like(flat)
    for expert in experts:
        index = torch.full((flat.shape[0], 1), expert)
        weights = torch.ones(flat.shape[0], 1)
        total = total + moe.experts(flat, index, weights)
    return total.reshape(normed.shape)


@pytest.mark.parametrize("name,maker", MAKERS)
def test_single_expert_matches_donor(tmp_path, name, maker):
    donor, donor_dir = save_donor(tmp_path, maker)
    donor.eval()
    config, part = part_from_disk(donor_dir, 1, [2])
    found = find_moe_block(config, 1)
    assert found["num_experts"] in (5, 6)
    x = torch.randn(2, 7, 48)
    layer_module = donor.model.layers[1]
    with torch.no_grad():
        want = donor_experts(moe_of(donor, 1), layer_module.post_attention_layernorm(x), [2])
        got = part(x)
    assert torch.allclose(got, want, atol=1e-6)


@pytest.mark.parametrize("name,maker", MAKERS)
def test_two_experts_sum(tmp_path, name, maker):
    donor, donor_dir = save_donor(tmp_path, maker)
    donor.eval()
    config, part = part_from_disk(donor_dir, 2, [0, 3])
    x = torch.randn(2, 5, 48)
    layer_module = donor.model.layers[2]
    with torch.no_grad():
        normed = layer_module.post_attention_layernorm(x)
        want = donor_experts(moe_of(donor, 2), normed, [0, 3])
        got = part(x)
    assert torch.allclose(got, want, atol=1e-6)


@pytest.mark.parametrize("name,maker", MAKERS)
def test_checkpoint_keeps_per_expert_tensors(tmp_path, name, maker):
    # The checkpoint stores each expert on its own, and we fetch only the
    # chosen experts, not the fused tensors of all of them.
    donor, donor_dir = save_donor(tmp_path, maker)
    names = expert_tensor_names(fetch_config(donor_dir), 1, [2])
    for tensor_name in names:
        assert "experts.gate_up_proj" not in tensor_name
        assert "experts.down_proj" not in tensor_name
        if ".experts." in tensor_name:
            assert ".experts.2." in tensor_name


@pytest.mark.parametrize("name,maker", MAKERS)
def test_part_from_module_names_matches_part_from_disk(tmp_path, name, maker):
    # A state dict keys the fused tensors by module name. That path slices
    # them, and must give the same part as the per-expert checkpoint path.
    donor, donor_dir = save_donor(tmp_path, maker)
    config, from_disk = part_from_disk(donor_dir, 1, [3, 1])
    from_state = build_expert_part(config, 1, [3, 1], donor.state_dict())
    x = torch.randn(2, 4, 48)
    with torch.no_grad():
        assert torch.equal(from_disk(x), from_state(x))


def save_host_and_catalog(tmp_path, donor_dir, indices):
    torch.manual_seed(1)
    host = LlamaForCausalLM(LlamaConfig(
        hidden_size=64, intermediate_size=128, num_hidden_layers=4, num_attention_heads=4,
        num_key_value_heads=2, vocab_size=300, max_position_embeddings=64,
    ))
    host_dir = str(tmp_path / "host")
    host.save_pretrained(host_dir)
    tiny_tokenizer().save_pretrained(host_dir)
    units_dir = tmp_path / "units"
    units_dir.mkdir()
    torch.save({"kind": "expert", "model": donor_dir, "layer": 1, "indices": torch.tensor(indices)}, units_dir / "e.pt")
    record = {"id": "exp", "type": "expert", "model": donor_dir, "layer": 1, "units": "units/e.pt"}
    catalog_path = tmp_path / "catalog.jsonl"
    catalog_path.write_text(json.dumps(record) + "\n")
    return host_dir, Catalog.load(str(catalog_path))


@pytest.mark.parametrize("name,maker", MAKERS)
def test_expert_graft_through_program(tmp_path, monkeypatch, name, maker):
    local_fetch(monkeypatch)
    donor, donor_dir = save_donor(tmp_path, maker)
    host_dir, catalog = save_host_and_catalog(tmp_path, donor_dir, [1, 4])
    out = str(tmp_path / "out")
    Program(host=host_dir).graft("exp").compile(out, catalog, align_texts=TEXTS, device="cpu")

    metas = json.load(open(os.path.join(out, "grafts.json")))
    assert metas[0]["kind"] == "expert"
    assert metas[0]["indices"] == [1, 4]
    report = json.load(open(os.path.join(out, "compile_report.json")))
    assert report["grafts"][0]["fit"] == "ridge"

    model, _ = load(out, dtype=torch.float32)
    host = LlamaForCausalLM.from_pretrained(host_dir, dtype=torch.float32)
    assert torch.equal(logits(model, "the cat sat"), logits(host, "the cat sat"))


def test_random_control_same_count(tmp_path, monkeypatch):
    local_fetch(monkeypatch)
    donor, donor_dir = save_donor(tmp_path, tiny_qwen3_moe)
    host_dir, catalog = save_host_and_catalog(tmp_path, donor_dir, [1, 4])
    out = str(tmp_path / "out")
    Program(host=host_dir).graft("exp", control="random", control_seed=3).compile(out, catalog, device="cpu")
    meta = json.load(open(os.path.join(out, "grafts.json")))[0]
    assert len(meta["indices"]) == 2
    assert len(set(meta["indices"])) == 2
    assert max(meta["indices"]) < 6
    assert meta["indices"] == random_neurons(6, 2, 3).tolist()
    assert meta["control"] == "random"


def test_top_routed_experts_picks_scaled_router_row():
    torch.manual_seed(0)
    donor = tiny_qwen3_moe().eval()
    moe = donor.model.layers[1].mlp
    with torch.no_grad():
        moe.gate.weight[4] *= 50.0
        moe.gate.weight[4] += 0.5
    top = top_routed_experts(donor, tiny_tokenizer(), TEXTS, moe, 2)
    assert top.shape == (2,)
    assert top.tolist() == sorted(top.tolist())
    assert 4 in top.tolist()
    assert top.dtype == torch.long


def tiny_gpt_oss():
    from transformers import GptOssConfig, GptOssForCausalLM

    return GptOssForCausalLM(GptOssConfig(
        hidden_size=32, intermediate_size=32, num_hidden_layers=2, num_attention_heads=2,
        num_key_value_heads=1, head_dim=16, vocab_size=300, num_local_experts=4,
        num_experts_per_tok=2, layer_types=["full_attention", "sliding_attention"],
    ))


def test_fused_checkpoint_experts_are_sliced(tmp_path):
    # GPT-OSS checkpoints store the fused expert tensors themselves, with
    # biases. We fetch them whole and slice out the chosen experts.
    donor, donor_dir = save_donor(tmp_path, tiny_gpt_oss)
    donor.eval()
    config = fetch_config(donor_dir)
    names = expert_tensor_names(config, 1, [3])
    assert "model.layers.1.mlp.experts.gate_up_proj" in names
    assert "model.layers.1.mlp.experts.down_proj_bias" in names
    _, part = part_from_disk(donor_dir, 1, [3])
    assert tuple(part.experts.gate_up_proj.shape[:1]) == (1,)
    x = torch.randn(2, 5, 32)
    layer_module = donor.model.layers[1]
    with torch.no_grad():
        want = donor_experts(layer_module.mlp, layer_module.post_attention_layernorm(x), [3])
        got = part(x)
    assert torch.allclose(got, want, atol=1e-6)


def tiny_lfm2_moe():
    from transformers import Lfm2MoeConfig, Lfm2MoeForCausalLM

    # Layer 0 is dense, layers 1 and 2 are MoE. Layer 1 is an attention
    # layer and layer 2 a conv layer.
    return Lfm2MoeForCausalLM(Lfm2MoeConfig(
        hidden_size=32, intermediate_size=64, moe_intermediate_size=24, num_hidden_layers=3,
        num_attention_heads=4, num_key_value_heads=2, vocab_size=300, num_experts=5,
        num_experts_per_tok=2, num_dense_layers=1, layer_types=["conv", "full_attention", "conv"],
    ))


@pytest.mark.parametrize("layer", [1, 2])
def test_lfm2_moe_experts_match_donor(tmp_path, layer):
    # LFM2 MoE keeps its experts in feed_forward and its pre-MLP norm in
    # ffn_norm. The checkpoint names the expert tensors w1, w2 and w3.
    donor, donor_dir = save_donor(tmp_path, tiny_lfm2_moe)
    donor.eval()
    config = fetch_config(donor_dir)
    found = find_moe_block(config, layer)
    assert found["fused"]
    assert found["num_experts"] == 5
    assert found["pre_norm_prefix"] == f"model.layers.{layer}.ffn_norm"
    names = expert_tensor_names(config, layer, [0, 4])
    assert f"model.layers.{layer}.feed_forward.experts.4.w1.weight" in names
    _, part = part_from_disk(donor_dir, layer, [0, 4])
    x = torch.randn(2, 5, 32)
    layer_module = donor.model.layers[layer]
    with torch.no_grad():
        want = donor_experts(layer_module.feed_forward, layer_module.ffn_norm(x), [0, 4])
        got = part(x)
    assert torch.allclose(got, want, atol=1e-6)


def test_lfm2_moe_dense_layer_has_no_experts():
    config = tiny_lfm2_moe().config
    with pytest.raises(ValueError, match="no MoE block"):
        find_moe_block(config, 0)


@pytest.mark.parametrize("maker", [tiny_mixtral, tiny_qwen3_moe, tiny_lfm2_moe])
def test_block_over_moe_layers_reads_per_expert_checkpoint(tmp_path, maker):
    # The checkpoint stores each expert's tensors on their own, and the
    # Transformers 5 layer holds them fused. The block part asks for the
    # per-expert names and fuses them, as from_pretrained does.
    from wsp.parts import block_tensor_names, build_block_part
    from tests.test_block import capture_block

    donor, donor_dir = save_donor(tmp_path, maker)
    donor = donor.eval()
    donor.config._attn_implementation = "eager"
    config = fetch_config(donor_dir)
    names = block_tensor_names(config, 1, 2)
    for name in names:
        assert not name.endswith("experts.gate_up_proj"), name
    tensors = read_tensors(local_range_reader(os.path.join(donor_dir, "model.safetensors")), names)
    part = build_block_part(config, 1, 2, tensors)
    ids = torch.randint(3, 300, (2, 7))
    x, y = capture_block(donor, 1, 2, ids)
    with torch.no_grad():
        assert torch.allclose(part(x), y - x, atol=1e-5)


def test_moe_block_graft_with_part_lora_compiles_heals_and_reloads(tmp_path, monkeypatch):
    # The block part fuses the per-expert checkpoint tensors, the part LoRA
    # wraps the fused experts inside it, and wsp.load rebuilds both.
    from wsp.lora import FusedExpertsLoRA

    local_fetch(monkeypatch)
    _, donor_dir = save_donor(tmp_path, tiny_lfm2_moe)
    host_dir, _ = save_host_and_catalog(tmp_path, donor_dir, [1, 4])
    record = {"id": "moeblock", "type": "block", "model": donor_dir, "layers": [1, 2]}
    catalog_path = tmp_path / "block_catalog.jsonl"
    catalog_path.write_text(json.dumps(record) + "\n")
    catalog = Catalog.load(str(catalog_path))
    out = str(tmp_path / "out")
    program = Program(host=host_dir).graft("moeblock", part_lora_rank=2).heal(steps=3, lr=1e-2, batch_size=2,
                                                                             seq_len=16)
    program.compile(out, catalog, align_texts=TEXTS, heal_texts=TEXTS, device="cpu")

    loaded, _ = load(out, dtype=torch.bfloat16)
    part = loaded.wsp_grafts[0].part
    assert isinstance(part.layers[0].feed_forward.experts, FusedExpertsLoRA)
    saved = load_file(os.path.join(out, "graft_weights.safetensors"))
    assert "0.part.layers.0.feed_forward.experts.base.gate_up_proj" in saved
    for key, value in loaded.wsp_grafts.state_dict().items():
        assert torch.equal(value, saved[key].to(value.dtype)), key
