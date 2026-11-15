"""Regression tests for review entries 0 (load side), 13, 18, and 19."""

import json
import os

import pytest
import torch
import transformers
from transformers import LlamaConfig, LlamaForCausalLM

import wsp.assemble
from wsp import Catalog, Program, load
from wsp.assemble import build_graft, host_hidden_size, host_layer_count
from wsp.program import GraftSpec
from tests.test_align import tiny_tokenizer
from tests.test_attn_heads import save_attn_tiny
from tests.test_expert import save_donor, save_host_and_catalog, tiny_qwen3_moe
from tests.test_mixed_compile import TEXTS, local_fetch, logits, save_tiny


def tiny_llama(seed):
    torch.manual_seed(seed)
    return LlamaForCausalLM(LlamaConfig(
        hidden_size=64, intermediate_size=128, num_hidden_layers=4, num_attention_heads=4,
        num_key_value_heads=2, vocab_size=300, max_position_embeddings=64,
    ))


def skip_fit(monkeypatch):
    """Replace fit_w_in with a stub that records the host layer it was given."""
    calls = []

    def fake_fit(host_model, host_tokenizer, host_layer, donor_id, donor_layer, align_texts, device,
                 norm_prefix=None, layer_input=False, stats=None):
        calls.append(host_layer)
        d_host = host_hidden_size(host_model)
        d_donor = wsp.assemble._text_config(wsp.assemble.fetch_config(donor_id)).hidden_size
        return torch.zeros(d_donor, d_host), 0.5, [0.5], 30

    monkeypatch.setattr(wsp.assemble, "fit_w_in", fake_fit)
    return calls


# Entry 18: negative or out-of-range at.

def test_negative_at_maps_to_last_layer_for_mlp(tmp_path, monkeypatch):
    local_fetch(monkeypatch)
    calls = skip_fit(monkeypatch)
    host_dir, donor_dir, catalog = save_tiny(tmp_path)
    host = LlamaForCausalLM.from_pretrained(host_dir)
    host_layer, graft, meta = build_graft(GraftSpec("qcluster", at=-1), catalog, host, tiny_tokenizer(),
                                          align_texts=TEXTS)
    assert host_layer == 3
    assert meta["host_layer"] == 3
    # The fit must read the same layer the hook goes on.
    assert calls == [3]

    host_layer, graft, meta = build_graft(GraftSpec("qcluster", at=-4), catalog, host, tiny_tokenizer())
    assert host_layer == 0


@pytest.mark.parametrize("at", [4, 9, -5])
def test_out_of_range_at_raises_for_mlp(tmp_path, monkeypatch, at):
    local_fetch(monkeypatch)
    host_dir, donor_dir, catalog = save_tiny(tmp_path)
    host = LlamaForCausalLM.from_pretrained(host_dir)
    with pytest.raises(ValueError, match="out of range"):
        build_graft(GraftSpec("qcluster", at=at), catalog, host, tiny_tokenizer())


def test_negative_and_out_of_range_at_for_attn(tmp_path, monkeypatch):
    local_fetch(monkeypatch)
    calls = skip_fit(monkeypatch)
    host_dir, donor_dir, catalog = save_attn_tiny(tmp_path)
    host = LlamaForCausalLM.from_pretrained(host_dir)
    host_layer, graft, meta = build_graft(GraftSpec("qheads", at=-2), catalog, host, tiny_tokenizer(),
                                          align_texts=TEXTS)
    assert host_layer == 2
    assert meta["host_layer"] == 2
    assert calls == [2]
    with pytest.raises(ValueError, match="out of range"):
        build_graft(GraftSpec("qheads", at=-5), catalog, host, tiny_tokenizer())
    with pytest.raises(ValueError, match="out of range"):
        build_graft(GraftSpec("qheads", at=4), catalog, host, tiny_tokenizer())


def test_negative_and_out_of_range_at_for_expert(tmp_path, monkeypatch):
    local_fetch(monkeypatch)
    calls = skip_fit(monkeypatch)
    donor, donor_dir = save_donor(tmp_path, tiny_qwen3_moe)
    host_dir, catalog = save_host_and_catalog(tmp_path, donor_dir, [1, 4])
    host = LlamaForCausalLM.from_pretrained(host_dir)
    host_layer, graft, meta = build_graft(GraftSpec("exp", at=-1), catalog, host, tiny_tokenizer(),
                                          align_texts=TEXTS)
    assert host_layer == 3
    assert meta["host_layer"] == 3
    assert calls == [3]
    with pytest.raises(ValueError, match="out of range"):
        build_graft(GraftSpec("exp", at=-5), catalog, host, tiny_tokenizer())
    with pytest.raises(ValueError, match="out of range"):
        build_graft(GraftSpec("exp", at=4), catalog, host, tiny_tokenizer())


def test_negative_at_hooks_the_layer_load_uses(tmp_path, monkeypatch):
    local_fetch(monkeypatch)
    host_dir, donor_dir, catalog = save_tiny(tmp_path)
    out = str(tmp_path / "out")
    Program(host=host_dir).graft("qcluster", at=-1).compile(out, catalog, device="cpu")
    meta = json.load(open(os.path.join(out, "grafts.json")))[0]
    assert meta["host_layer"] == 3


# Entry 19: stale files from an earlier compile into the same directory.

def save_task_vector(tmp_path, host_dir, catalog):
    """Add a task vector record whose fine-tuned model differs from the host."""
    finetuned = tiny_llama(7)
    ft_dir = str(tmp_path / "finetuned")
    finetuned.save_pretrained(ft_dir)
    records = list(catalog.records.values())
    records.append({"id": "tv", "type": "task_vector", "base": host_dir, "model": ft_dir})
    path = tmp_path / "catalog2.jsonl"
    lines = []
    for record in records:
        lines.append(json.dumps(record))
    path.write_text("\n".join(lines) + "\n")
    return Catalog.load(str(path))


def test_graft_then_no_graft_loads_plain_host(tmp_path, monkeypatch):
    local_fetch(monkeypatch)
    host_dir, donor_dir, catalog = save_tiny(tmp_path)
    out = str(tmp_path / "out")
    program = Program(host=host_dir).graft("qcluster", at=2).heal(steps=2, lr=1e-2, batch_size=2, seq_len=16)
    program.compile(out, catalog, heal_texts=TEXTS, device="cpu")
    assert os.path.exists(os.path.join(out, "grafts.json"))

    Program(host=host_dir).compile(out, catalog, device="cpu")
    assert not os.path.exists(os.path.join(out, "grafts.json"))
    assert not os.path.exists(os.path.join(out, "graft_weights.safetensors"))

    model, _ = load(out, dtype=torch.float32)
    assert not hasattr(model, "wsp_grafts")
    # The no-graft path saves the host in bf16, so compare with a clean compile.
    clean = str(tmp_path / "clean")
    Program(host=host_dir).compile(clean, catalog, device="cpu")
    expected, _ = load(clean, dtype=torch.float32)
    assert torch.equal(logits(model, "the cat sat"), logits(expected, "the cat sat"))


def test_no_graft_then_graft_loads_graft_model(tmp_path, monkeypatch):
    local_fetch(monkeypatch)
    host_dir, donor_dir, catalog = save_tiny(tmp_path)
    out = str(tmp_path / "out")
    Program(host=host_dir).compile(out, catalog, device="cpu")

    program = Program(host=host_dir).graft("qcluster", at=2).heal(steps=2, lr=1e-2, batch_size=2, seq_len=16)
    program.compile(out, catalog, heal_texts=TEXTS, device="cpu")
    model, _ = load(out, dtype=torch.float32)
    assert len(model.wsp_grafts) == 1
    host = LlamaForCausalLM.from_pretrained(host_dir, dtype=torch.float32)
    assert not torch.equal(logits(model, "the cat sat"), logits(host, "the cat sat"))


def test_changed_host_then_plain_host_drops_stale_host_dir(tmp_path, monkeypatch):
    local_fetch(monkeypatch)
    host_dir, donor_dir, catalog = save_tiny(tmp_path)
    catalog = save_task_vector(tmp_path, host_dir, catalog)
    out = str(tmp_path / "out")
    Program(host=host_dir).tv("tv").graft("qcluster").compile(out, catalog, device="cpu")
    assert os.path.isdir(os.path.join(out, "host"))

    # Same graft, no task vector: the host must be the original host again.
    Program(host=host_dir).graft("qcluster").compile(out, catalog, device="cpu")
    assert not os.path.exists(os.path.join(out, "host"))
    model, _ = load(out, dtype=torch.float32)
    host = LlamaForCausalLM.from_pretrained(host_dir, dtype=torch.float32)
    assert torch.equal(logits(model, "the cat sat"), logits(host, "the cat sat"))


# Entry 13: composite (multimodal) configs keep their text fields in a sub-config.

def tiny_gemma3(seed):
    torch.manual_seed(seed)
    config = transformers.Gemma3Config(
        text_config=dict(
            hidden_size=48, intermediate_size=96, num_hidden_layers=3, num_attention_heads=4,
            num_key_value_heads=2, head_dim=12, vocab_size=300, max_position_embeddings=64,
            sliding_window=16,
        ),
        vision_config=dict(
            hidden_size=16, intermediate_size=32, num_hidden_layers=1, num_attention_heads=2,
            image_size=28, patch_size=14,
        ),
        mm_tokens_per_image=4,
    )
    return config, transformers.Gemma3ForConditionalGeneration(config).eval()


def test_composite_config_has_no_top_level_text_fields():
    config, model = tiny_gemma3(0)
    assert not hasattr(config, "num_hidden_layers")
    assert host_layer_count(model) == 3
    assert host_hidden_size(model) == 48


@pytest.mark.parametrize("control,at", [(None, None), (None, 1), ("random", None)])
def test_composite_donor_and_host_build_mlp_graft(tmp_path, monkeypatch, control, at):
    local_fetch(monkeypatch)
    _, donor = tiny_gemma3(0)
    _, host = tiny_gemma3(1)
    donor_dir = str(tmp_path / "donor")
    donor.save_pretrained(donor_dir)

    units_dir = tmp_path / "units"
    units_dir.mkdir()
    indices = torch.tensor([2, 5, 11, 40])
    torch.save({"kind": "mlp_cluster", "model": donor_dir, "layer": 1, "indices": indices}, units_dir / "c.pt")
    record = {"id": "g3", "type": "mlp_cluster", "model": donor_dir, "layer": 1, "units": "units/c.pt"}
    catalog_path = tmp_path / "catalog.jsonl"
    catalog_path.write_text(json.dumps(record) + "\n")
    catalog = Catalog.load(str(catalog_path))

    spec = GraftSpec("g3", at=at, control=control, control_seed=1)
    host_layer, graft, meta = build_graft(spec, catalog, host, tiny_tokenizer())
    assert host_layer == 1
    assert meta["d_host"] == 48
    assert meta["d_donor"] == 48
    assert len(meta["indices"]) == 4
    assert max(meta["indices"]) < 96

    x = torch.randn(1, 5, 48)
    assert graft(x).shape == (1, 5, 48)


# Entry 0, load side: attn_heads parts keep their own KV cache, so the
# cache stays on.

def test_load_keeps_cache_for_attn_heads(tmp_path, monkeypatch):
    local_fetch(monkeypatch)
    host_dir, donor_dir, catalog = save_attn_tiny(tmp_path)
    out = str(tmp_path / "out")
    Program(host=host_dir).graft("qheads").compile(out, catalog, device="cpu")
    model, _ = load(out, dtype=torch.float32)
    assert model.config.use_cache is not False
    assert model.generation_config.use_cache is not False


def test_load_keeps_cache_for_mlp_graft(tmp_path, monkeypatch):
    local_fetch(monkeypatch)
    host_dir, donor_dir, catalog = save_tiny(tmp_path)
    out = str(tmp_path / "out")
    Program(host=host_dir).graft("qcluster").compile(out, catalog, device="cpu")
    model, _ = load(out, dtype=torch.float32)
    assert model.config.use_cache is not False


# ridge_fit now returns a per-dimension r2 as well.

def test_compile_report_has_fit_r2_dim(tmp_path, monkeypatch):
    local_fetch(monkeypatch)
    host_dir, donor_dir, catalog = save_tiny(tmp_path)
    out = str(tmp_path / "out")
    Program(host=host_dir).graft("qcluster").compile(out, catalog, align_texts=TEXTS, device="cpu")
    report = json.load(open(os.path.join(out, "compile_report.json")))
    graft = report["grafts"][0]
    assert graft["fit"] == "ridge"
    assert "fit_r2_dim" in graft
    assert graft["fit_r2_dim"] is not None


def test_skipped_fit_records_fit_r2_dim_none(tmp_path, monkeypatch):
    local_fetch(monkeypatch)
    host_dir, donor_dir, catalog = save_tiny(tmp_path)
    out = str(tmp_path / "out")
    Program(host=host_dir).graft("qcluster").compile(out, catalog, device="cpu")
    report = json.load(open(os.path.join(out, "compile_report.json")))
    graft = report["grafts"][0]
    assert graft["fit"] == "skipped"
    assert graft["fit_r2_dim"] is None
