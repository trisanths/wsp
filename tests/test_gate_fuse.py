import json
import os

import pytest
import torch

transformers = pytest.importorskip("transformers")

from wsp import Catalog, Program
from wsp import fuse
from wsp.gate import (
    TokenGate,
    attach_gates,
    detach_gates,
    gate_means,
    gate_parameters,
    load_gates,
    save_gates,
)
from wsp.graft import Graft, attach_grafts, disable_cache
from wsp.parts import build_block_part
from tests.test_align import tiny_tokenizer
from tests.test_mixed_compile import TEXTS, local_fetch, logits


def tiny_host():
    torch.manual_seed(0)
    return transformers.LlamaForCausalLM(transformers.LlamaConfig(
        hidden_size=64, intermediate_size=128, num_hidden_layers=4, num_attention_heads=4,
        num_key_value_heads=2, vocab_size=300, max_position_embeddings=64,
    )).eval()


def tiny_qwen(seed):
    torch.manual_seed(seed)
    config = transformers.Qwen2Config(
        hidden_size=48, intermediate_size=96, num_hidden_layers=3, num_attention_heads=4,
        num_key_value_heads=2, vocab_size=300, max_position_embeddings=64,
    )
    return transformers.Qwen2ForCausalLM(config).eval()


def block_graft(donor, first=1, last=2):
    tensors = {}
    for name, value in donor.state_dict().items():
        tensors[name] = value.detach().clone()
    part = build_block_part(donor.config, first, last, tensors)
    return Graft(64, 48, part)


def two_graft_host():
    host = tiny_host()
    a = block_graft(tiny_qwen(1))
    b = block_graft(tiny_qwen(2))
    attach_grafts(host, [(1, a), (2, b)])
    return host, a, b


def test_token_gate_starts_at_sigmoid_bias():
    gate = TokenGate(8)
    h = torch.randn(2, 3, 8)
    assert torch.allclose(gate(h), torch.full((2, 3, 1), 0.5))
    gate = TokenGate(8, init_bias=2.0)
    assert torch.allclose(gate(h), torch.sigmoid(torch.tensor(2.0)).expand(2, 3, 1))
    # The gate reads the direction of h, not its size.
    gate.weight.data.normal_()
    assert torch.allclose(gate(h), gate(10.0 * h), atol=1e-5)


def test_fresh_gated_grafts_keep_host_logits():
    plain = tiny_host()
    host, _, _ = two_graft_host()
    attach_gates(host)
    assert len(host.wsp_gates) == 2
    assert torch.equal(logits(host, "the cat sat"), logits(plain, "the cat sat"))
    means = gate_means(host)
    assert abs(means[0] - 0.5) < 1e-6 and abs(means[1] - 0.5) < 1e-6


def test_gate_scales_the_graft_delta():
    host, a, _ = two_graft_host()
    torch.manual_seed(5)
    with torch.no_grad():
        a.w_out.weight.normal_(std=0.1)
    h = torch.randn(1, 4, 64)
    ungated = a(h) - h
    attach_gates(host)
    with torch.no_grad():
        host.wsp_gates[0].bias.fill_(1.5)
    g = torch.sigmoid(torch.tensor(1.5))
    assert torch.allclose(a(h) - h, g * ungated, atol=1e-5)
    detach_gates(host)
    assert getattr(host, "wsp_gates", None) is None
    assert torch.allclose(a(h) - h, ungated, atol=1e-6)


def test_gates_train_with_the_glue():
    host, _, _ = two_graft_host()
    attach_gates(host)
    from wsp.graft import freeze_all_but_glue, glue_parameters

    freeze_all_but_glue(host)
    gates = gate_parameters(host)
    for param in gates:
        param.requires_grad_(True)
    groups = [{"params": glue_parameters(host), "lr": 1e-2}, {"params": gates, "lr": 1e-1}]
    losses = fuse.heal_groups(host, tiny_tokenizer(), TEXTS, groups, steps=4, batch_size=2, seq_len=16, log_every=0)
    assert len(losses) == 4
    # The first step moves w_out off zero; later steps give the gates a gradient.
    assert float(host.wsp_gates[0].weight.detach().abs().sum()) > 0
    assert float(host.wsp_gates[1].bias.detach()) != 0.0


def test_gates_save_and_reload(tmp_path):
    host, a, b = two_graft_host()
    with torch.no_grad():
        a.w_out.weight.normal_(std=0.1)
        b.w_out.weight.normal_(std=0.1)
    attach_gates(host, which=[1])
    with torch.no_grad():
        host.wsp_gates[0].weight.normal_()
        host.wsp_gates[0].bias.fill_(-0.7)
    expected = logits(host, "the cat sat on a mat")
    save_gates(host, str(tmp_path))
    rows = json.load(open(tmp_path / "gates.json"))
    assert rows[0]["graft"] == 1

    again, a2, b2 = two_graft_host()
    with torch.no_grad():
        a2.w_out.weight.copy_(a.w_out.weight)
        b2.w_out.weight.copy_(b.w_out.weight)
    assert not torch.equal(logits(again, "the cat sat on a mat"), expected)
    gates = load_gates(again, str(tmp_path))
    assert len(gates) == 1 and gates[0].graft_index == 1
    assert torch.equal(logits(again, "the cat sat on a mat"), expected)


def save_fusion_tiny(tmp_path):
    host = tiny_host()
    host_dir = str(tmp_path / "host")
    host.save_pretrained(host_dir)
    tiny_tokenizer().save_pretrained(host_dir)
    records = []
    for number, seed in enumerate([1, 2]):
        donor_dir = str(tmp_path / f"donor{number}")
        tiny_qwen(seed).save_pretrained(donor_dir)
        tiny_tokenizer().save_pretrained(donor_dir)
        records.append({"id": f"block{number}", "type": "block", "model": donor_dir, "layers": [1, 2]})
    return host_dir, Catalog(records, root=str(tmp_path))


def fusion_program(host_dir, heal=True):
    stitch = {"kind": "mlp", "objective": "donor_lm", "steps": 2, "batch_size": 2, "seq_len": 16}
    program = Program(host=host_dir, seed=0)
    program.graft("block0", at=1, stitch=stitch, part_lora_rank=2)
    program.graft("block1", at=2, part_lora_rank=2)
    if heal:
        program.heal(steps=3, lr=1e-2, batch_size=2, seq_len=16)
    return program


def test_fresh_two_graft_fusion_equals_host(tmp_path, monkeypatch):
    local_fetch(monkeypatch)
    host_dir, catalog = save_fusion_tiny(tmp_path)
    model, _, report, metas = fuse.fuse_model(fusion_program(host_dir, heal=False), catalog, [TEXTS, TEXTS[:8]],
                                              None, device="cpu")
    assert len(metas) == 2 and metas[0]["stitch"]["objective"] == "donor_lm"
    assert report["grafts"][1]["align_texts"] == 8
    assert len(model.wsp_gates) == 2
    host = transformers.AutoModelForCausalLM.from_pretrained(host_dir, dtype=torch.bfloat16).eval()
    assert torch.equal(logits(model, "the cat sat"), logits(host, "the cat sat"))


def test_two_graft_fusion_compiles_heals_and_reloads(tmp_path, monkeypatch):
    local_fetch(monkeypatch)
    host_dir, catalog = save_fusion_tiny(tmp_path)
    out = str(tmp_path / "fused")
    program = fusion_program(host_dir)
    model, tokenizer, report, metas = fuse.fuse_model(program, catalog, TEXTS, TEXTS, device="cpu")
    fuse.save_fusion(model, tokenizer, program, metas, report, out)
    for name in ["grafts.json", "graft_weights.safetensors", "gates.json", "gate_weights.safetensors",
                 "program.json", "compile_report.json"]:
        assert os.path.exists(os.path.join(out, name)), name
    assert report["heal"]["gate_params"] == 2 * (64 + 1)
    assert report["heal"]["part_lora_params"] > 0
    assert len(report["heal"]["losses"]) == 3

    expected = logits(model, "the cat sat on a mat")
    host = transformers.AutoModelForCausalLM.from_pretrained(host_dir, dtype=torch.bfloat16).eval()
    assert not torch.equal(expected, logits(host, "the cat sat on a mat"))
    loaded, _ = fuse.load(out)
    assert len(loaded.wsp_gates) == 2
    assert torch.equal(logits(loaded, "the cat sat on a mat"), expected)

    # wsp.load alone gives the same grafts without gates.
    import wsp

    ungated, _ = wsp.load(out)
    assert getattr(ungated, "wsp_gates", None) is None
    assert not torch.equal(logits(ungated, "the cat sat on a mat"), expected)


def test_compile_fusion_writes_dir_and_cached_generate_matches_uncached(tmp_path, monkeypatch):
    local_fetch(monkeypatch)
    host_dir, catalog = save_fusion_tiny(tmp_path)
    out = str(tmp_path / "fused")
    report = fuse.compile_fusion(fusion_program(host_dir), catalog, out, TEXTS, TEXTS, device="cpu", log_every=0)
    assert report["fusion"] is True and report["seconds"] >= 0
    model, tokenizer = fuse.load(out, dtype=torch.float32)
    tokenizer.padding_side = "left"
    encoded = tokenizer(["the cat sat", "a big red dog ran"], return_tensors="pt", padding=True)
    encoded = {"input_ids": encoded["input_ids"], "attention_mask": encoded["attention_mask"]}
    with torch.no_grad():
        cached = model.generate(**encoded, max_new_tokens=4, do_sample=False, pad_token_id=tokenizer.pad_token_id)
    disable_cache(model)
    with torch.no_grad():
        plain = model.generate(**encoded, max_new_tokens=4, do_sample=False, use_cache=False,
                               pad_token_id=tokenizer.pad_token_id)
    assert torch.equal(cached, plain)


def test_nogate_fusion_has_no_gate_files(tmp_path, monkeypatch):
    local_fetch(monkeypatch)
    host_dir, catalog = save_fusion_tiny(tmp_path)
    out = str(tmp_path / "fused")
    report = fuse.compile_fusion(fusion_program(host_dir), catalog, out, TEXTS, TEXTS, gates=False, device="cpu",
                                 log_every=0)
    assert report["gates"] is None and report["heal"]["gate_params"] == 0
    assert not os.path.exists(os.path.join(out, "gates.json"))
    model, _ = fuse.load(out)
    assert getattr(model, "wsp_gates", None) is None


def test_texts_per_graft():
    assert fuse.texts_per_graft(None, 2) == [None, None]
    assert fuse.texts_per_graft(["a", "b"], 2) == [["a", "b"], ["a", "b"]]
    assert fuse.texts_per_graft([["a"], None], 2) == [["a"], None]
    with pytest.raises(ValueError):
        fuse.texts_per_graft([["a"]], 2)


def test_gate_means_count_only_real_tokens():
    gate = TokenGate(8)
    with torch.no_grad():
        gate.weight.normal_()
    h = torch.randn(2, 3, 8)
    real = torch.tensor([[0, 1, 1], [1, 1, 1]])
    values = gate(h, real)
    expected = (values[0, 1:].sum() + values[1].sum()) / 5
    assert abs(gate.mean() - float(expected.detach())) < 1e-6


def test_gate_means_do_not_change_with_left_padding():
    from tests.test_align import tiny_tokenizer

    host, a, b = two_graft_host()
    with torch.no_grad():
        a.w_out.weight.normal_(std=0.1)
        b.w_out.weight.normal_(std=0.1)
    attach_gates(host)
    with torch.no_grad():
        for gate in host.wsp_gates:
            gate.weight.normal_()
    tokenizer = tiny_tokenizer()
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    texts = ["the cat", "a big red dog ran on the mat"]
    sums = [0.0, 0.0]
    count = 0
    for text in texts:
        encoded = tokenizer([text], return_tensors="pt")
        gate_means(host, reset=True)
        with torch.no_grad():
            host(input_ids=encoded["input_ids"], attention_mask=encoded["attention_mask"])
        tokens = int(encoded["attention_mask"].sum())
        means = gate_means(host)
        for index in range(2):
            sums[index] += means[index] * tokens
        count += tokens
    encoded = tokenizer(texts, return_tensors="pt", padding=True)
    assert int((encoded["attention_mask"] == 0).sum()) > 0
    gate_means(host, reset=True)
    with torch.no_grad():
        host(input_ids=encoded["input_ids"], attention_mask=encoded["attention_mask"])
    means = gate_means(host)
    for index in range(2):
        assert abs(means[index] - sums[index] / count) < 1e-4



def test_fusion_with_host_lora_heals_and_reloads(tmp_path, monkeypatch):
    """A host LoRA trains in the fusion heal, and wsp.fuse.load rebuilds it exactly."""
    local_fetch(monkeypatch)
    host_dir, catalog = save_fusion_tiny(tmp_path)
    out = str(tmp_path / "fused_lora")
    program = fusion_program(host_dir)
    program.lora(rank=2)
    model, tokenizer, report, metas = fuse.fuse_model(program, catalog, TEXTS, TEXTS, device="cpu")
    fuse.save_fusion(model, tokenizer, program, metas, report, out)
    assert report["lora"] is not None and report["lora"]["params"] > 0
    assert os.path.exists(os.path.join(out, "lora.json"))
    from wsp.lora import lora_modules

    trained = 0
    for name, module in lora_modules(model):
        trained += int(module.lora_B.abs().sum() > 0)
    assert trained > 0
    expected = logits(model, "the cat sat on a mat")
    loaded, _ = fuse.load(out)
    assert torch.equal(logits(loaded, "the cat sat on a mat"), expected)
