import json
import os

import pytest
import torch
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
    if hasattr(layer_module, "block_sparse_moe"):
        return layer_module.block_sparse_moe
    return layer_module.mlp


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
        want = moe_of(donor, 1).experts[2](layer_module.post_attention_layernorm(x))
        got = part(x)
    assert torch.allclose(got, want, atol=1e-6)


@pytest.mark.parametrize("name,maker", MAKERS)
def test_two_experts_sum(tmp_path, name, maker):
    donor, donor_dir = save_donor(tmp_path, maker)
    donor.eval()
    config, part = part_from_disk(donor_dir, 2, [0, 3])
    x = torch.randn(2, 5, 48)
    layer_module = donor.model.layers[2]
    experts = moe_of(donor, 2).experts
    with torch.no_grad():
        normed = layer_module.post_attention_layernorm(x)
        want = experts[0](normed) + experts[3](normed)
        got = part(x)
    assert torch.allclose(got, want, atol=1e-6)


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


def test_fused_experts_raise_not_implemented():
    config = fetch_config_for_fused()
    with pytest.raises(NotImplementedError):
        find_moe_block(config, 0)


def fetch_config_for_fused():
    from transformers import GptOssConfig

    return GptOssConfig(
        hidden_size=32, intermediate_size=32, num_hidden_layers=1, num_attention_heads=2,
        num_key_value_heads=1, head_dim=16, vocab_size=100, num_local_experts=4,
        num_experts_per_tok=2, layer_types=["full_attention"],
    )
