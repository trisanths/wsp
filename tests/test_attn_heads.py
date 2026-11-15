import json
import os

import pytest
import torch

transformers = pytest.importorskip("transformers")

from wsp import Catalog, Program, load
from wsp.parts import attn_tensor_names, build_attn_heads, causal_mask
from tests.test_mixed_compile import local_fetch, logits, TEXTS
from tests.test_align import tiny_tokenizer


def tiny_donor(kind):
    torch.manual_seed(3)
    if kind == "llama":
        config = transformers.LlamaConfig(
            hidden_size=64, intermediate_size=128, num_hidden_layers=3, num_attention_heads=4,
            num_key_value_heads=2, vocab_size=300, max_position_embeddings=64,
        )
        config._attn_implementation = "eager"
        return config, transformers.LlamaForCausalLM(config).eval()
    config = transformers.Qwen2Config(
        hidden_size=48, intermediate_size=96, num_hidden_layers=3, num_attention_heads=4,
        num_key_value_heads=2, vocab_size=300, max_position_embeddings=64,
    )
    config._attn_implementation = "eager"
    return config, transformers.Qwen2ForCausalLM(config).eval()


def donor_tensors(model):
    out = {}
    for name, value in model.state_dict().items():
        out[name] = value.detach().clone()
    return out


def donor_attention_output(model, layer, x):
    """Run the donor layer's own norm and attention on x."""
    block = model.model.layers[layer]
    batch, seq, _ = x.shape
    position_ids = torch.arange(seq).view(1, seq).expand(batch, seq)
    cos_sin = model.model.rotary_emb(x, position_ids)
    mask = causal_mask(batch, seq, x.dtype, x.device)
    normed = block.input_layernorm(x)
    return block.self_attn(hidden_states=normed, position_embeddings=cos_sin, attention_mask=mask)[0]


@pytest.mark.parametrize("kind", ["llama", "qwen2"])
def test_all_heads_equal_donor_attention(kind):
    config, donor = tiny_donor(kind)
    layer = 1
    part = build_attn_heads(config, layer, torch.arange(4), donor_tensors(donor))
    x = torch.randn(2, 7, config.hidden_size)
    expected = donor_attention_output(donor, layer, x)
    assert torch.allclose(part(x), expected, atol=1e-5)


@pytest.mark.parametrize("kind", ["llama", "qwen2"])
def test_two_head_sets_add_up(kind):
    config, donor = tiny_donor(kind)
    layer = 2
    tensors = donor_tensors(donor)
    if "self_attn.o_proj.bias" in " ".join(tensors.keys()):
        pytest.skip("o_proj has a bias, so head outputs do not add")
    first = build_attn_heads(config, layer, torch.tensor([0, 3]), tensors)
    second = build_attn_heads(config, layer, torch.tensor([1, 2]), tensors)
    full = build_attn_heads(config, layer, torch.arange(4), tensors)
    x = torch.randn(2, 6, config.hidden_size)
    assert torch.allclose(first(x) + second(x), full(x), atol=1e-5)


def test_tensor_names_cover_attention_and_norm():
    config, donor = tiny_donor("qwen2")
    names = attn_tensor_names(config, 0)
    assert "model.layers.0.self_attn.q_proj.weight" in names
    assert "model.layers.0.input_layernorm.weight" in names
    assert not any("post_attention_layernorm" in name for name in names)


def save_attn_tiny(tmp_path):
    from transformers import LlamaConfig, LlamaForCausalLM, Qwen2Config, Qwen2ForCausalLM

    torch.manual_seed(0)
    host = LlamaForCausalLM(LlamaConfig(
        hidden_size=64, intermediate_size=128, num_hidden_layers=4, num_attention_heads=4,
        num_key_value_heads=2, vocab_size=300, max_position_embeddings=64,
    ))
    donor = Qwen2ForCausalLM(Qwen2Config(
        hidden_size=48, intermediate_size=96, num_hidden_layers=3, num_attention_heads=4,
        num_key_value_heads=2, vocab_size=300, max_position_embeddings=64,
    ))
    host_dir = str(tmp_path / "host")
    donor_dir = str(tmp_path / "donor")
    for model, path in [(host, host_dir), (donor, donor_dir)]:
        model.save_pretrained(path)
        tiny_tokenizer().save_pretrained(path)
    units_dir = tmp_path / "units"
    units_dir.mkdir()
    heads = torch.tensor([0, 3])
    torch.save({"kind": "attn_heads", "model": donor_dir, "layer": 1, "indices": heads}, units_dir / "h.pt")
    record = {"id": "qheads", "type": "attn_heads", "model": donor_dir, "layer": 1, "units": "units/h.pt"}
    catalog_path = tmp_path / "catalog.jsonl"
    catalog_path.write_text(json.dumps(record) + "\n")
    return host_dir, donor_dir, Catalog.load(str(catalog_path))


def test_fresh_attn_graft_gives_host_logits(tmp_path, monkeypatch):
    local_fetch(monkeypatch)
    host_dir, donor_dir, catalog = save_attn_tiny(tmp_path)
    out = str(tmp_path / "out")
    Program(host=host_dir).graft("qheads").compile(out, catalog, align_texts=TEXTS, device="cpu")

    report = json.load(open(os.path.join(out, "compile_report.json")))
    assert report["grafts"][0]["fit"] == "ridge"
    assert report["grafts"][0]["kind"] == "attn_heads"

    model, _ = load(out, dtype=torch.float32)
    host = transformers.LlamaForCausalLM.from_pretrained(host_dir, dtype=torch.float32)
    assert torch.equal(logits(model, "the cat sat"), logits(host, "the cat sat"))


def test_random_control_picks_same_number_of_heads(tmp_path, monkeypatch):
    local_fetch(monkeypatch)
    host_dir, donor_dir, catalog = save_attn_tiny(tmp_path)
    out = str(tmp_path / "out")
    Program(host=host_dir).graft("qheads", control="random", control_seed=2).compile(out, catalog, device="cpu")
    meta = json.load(open(os.path.join(out, "grafts.json")))[0]
    assert len(meta["indices"]) == 2
    assert meta["control"] == "random"
    assert meta["kind"] == "attn_heads"
    assert max(meta["indices"]) < 4


def test_healed_attn_graft_reloads_with_saved_weights(tmp_path, monkeypatch):
    local_fetch(monkeypatch)
    host_dir, donor_dir, catalog = save_attn_tiny(tmp_path)
    out = str(tmp_path / "out")
    program = Program(host=host_dir).graft("qheads", at=2).heal(steps=3, lr=1e-2, batch_size=2, seq_len=16)
    program.compile(out, catalog, heal_texts=TEXTS, device="cpu")

    model, _ = load(out, dtype=torch.float32)
    host = transformers.LlamaForCausalLM.from_pretrained(host_dir, dtype=torch.float32)
    assert not torch.equal(logits(model, "the cat sat"), logits(host, "the cat sat"))


def test_attn_part_ignores_train_mode():
    import copy

    import torch
    from transformers import Qwen2Config, Qwen2ForCausalLM

    from wsp.parts import attn_tensor_names, build_attn_heads

    config = Qwen2Config(hidden_size=48, intermediate_size=96, num_hidden_layers=2, num_attention_heads=4,
                         num_key_value_heads=2, vocab_size=300, max_position_embeddings=64, attention_dropout=0.5)
    torch.manual_seed(0)
    donor = Qwen2ForCausalLM(config)
    state = donor.state_dict()
    tensors = {}
    for name in attn_tensor_names(config, 1):
        tensors[name] = state[name]
    part = build_attn_heads(copy.deepcopy(config), 1, torch.tensor([0, 2]), tensors).float()
    part.train()
    assert part.training is False
    x = torch.randn(1, 6, 48)
    assert torch.equal(part(x), part(x))
