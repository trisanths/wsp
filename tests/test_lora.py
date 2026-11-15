import json
import os

import pytest
import torch
from safetensors.torch import load_file
from torch import nn
from transformers import AutoModelForCausalLM, LlamaConfig, LlamaForCausalLM

import wsp.compiler
from wsp import Program, load
from wsp.compiler import CompileError
from wsp.lora import (
    LoRALinear,
    apply_lora,
    graft_budget,
    load_lora_state,
    lora_modules,
    lora_param_count,
    lora_state_dict,
    merge_lora,
    params_per_rank,
    rank_for_budget,
)
from tests.test_mixed_compile import TEXTS, local_fetch, logits, save_tiny


def tiny_config():
    return LlamaConfig(
        hidden_size=64, intermediate_size=128, num_hidden_layers=4, num_attention_heads=4,
        num_key_value_heads=2, vocab_size=300, max_position_embeddings=64,
    )


def tiny_model():
    torch.manual_seed(0)
    return LlamaForCausalLM(tiny_config()).eval()


def randomize_lora(model):
    torch.manual_seed(1)
    with torch.no_grad():
        for _, module in lora_modules(model):
            module.lora_B.normal_(0, 0.05)


# Per layer, rank 1: q 64+64, k 64+32, v 64+32, o 64+64, gate 64+128, up 64+128, down 128+64.
PER_LAYER = 128 + 96 + 96 + 128 + 192 + 192 + 192


def test_lora_linear_with_zero_b_equals_base():
    torch.manual_seed(0)
    base = nn.Linear(16, 8)
    lora = LoRALinear(base, rank=4)
    x = torch.randn(3, 5, 16)
    assert torch.equal(lora(x), base(x))
    assert lora.lora_A.abs().sum() > 0
    assert lora.lora_A.dtype == torch.float32

    # Only A and B train, and the output dtype follows the base.
    lora(x).sum().backward()
    assert base.weight.grad is None
    assert lora.lora_B.grad is not None
    bf16 = LoRALinear(nn.Linear(16, 8).to(torch.bfloat16), rank=2)
    assert bf16(x.to(torch.bfloat16)).dtype == torch.bfloat16


def test_param_count_matches_built_count_and_rank_for_budget():
    config = tiny_config()
    model = tiny_model()
    assert params_per_rank(model) == 4 * PER_LAYER
    assert lora_param_count(config, 3) == 3 * 4 * PER_LAYER

    assert rank_for_budget(model, 3 * 4 * PER_LAYER) == 3
    assert rank_for_budget(model, 4 * 4 * PER_LAYER - 1) == 3
    assert rank_for_budget(model, 10) == 1

    params = apply_lora(model, 3)
    built = 0
    for param in params:
        built += param.numel()
    assert built == lora_param_count(config, 3)
    # A wrapped model still counts the same, so a budget can be read after apply.
    assert params_per_rank(model) == 4 * PER_LAYER

    targets = ("q_proj", "down_proj")
    other = tiny_model()
    params = apply_lora(other, 5, targets=targets, layers=[1, -1])
    built = 0
    for param in params:
        built += param.numel()
    assert built == lora_param_count(config, 5, targets, layers=[1, -1])
    assert built == 5 * 2 * (128 + 192)


def test_merge_gives_same_logits_and_plain_modules():
    model = tiny_model()
    plain_keys = sorted(model.state_dict().keys())
    apply_lora(model, 4)
    randomize_lora(model)
    ids = torch.tensor([[1, 5, 9, 33, 70]])
    with torch.no_grad():
        before = model(ids).logits
    merge_lora(model)
    with torch.no_grad():
        after = model(ids).logits
    assert torch.allclose(before, after, atol=1e-5)
    assert len(lora_modules(model)) == 0
    assert sorted(model.state_dict().keys()) == plain_keys


def test_lora_state_round_trip():
    model = tiny_model()
    apply_lora(model, 2)
    randomize_lora(model)
    state = lora_state_dict(model)
    other = tiny_model()
    apply_lora(other, 2)
    load_lora_state(other, state)
    ids = torch.tensor([[2, 4, 8]])
    with torch.no_grad():
        assert torch.equal(model(ids).logits, other(ids).logits)
    state["extra.lora_A"] = torch.zeros(1)
    with pytest.raises(KeyError):
        load_lora_state(other, state)


def test_program_lora_round_trip_and_checks():
    program = Program(host="some/model").lora(budget_params=5000, targets=["q_proj"]).heal(steps=2)
    again = Program.from_dict(json.loads(json.dumps(program.to_dict())))
    assert again.to_dict() == program.to_dict()
    assert again.lora_spec.budget_params == 5000
    assert again.lora_spec.targets == ["q_proj"]

    old = Program.from_dict({"base": "some/model", "seed": 0, "masks": None, "task_vectors": []})
    assert old.lora_spec is None
    assert "lora" not in old.to_dict()

    with pytest.raises(ValueError):
        Program(host="some/model").lora()


def test_lora_without_heal_is_an_error(tmp_path):
    with pytest.raises(CompileError, match="heal"):
        Program(host="some/model").lora(rank=2).compile(str(tmp_path / "out"), None, device="cpu")


def capture_model(monkeypatch, name):
    """Wrap wsp.compiler.<name> so the test can see the model it gets."""
    seen = {}
    original = getattr(wsp.compiler, name)

    def wrapped(*args, **kwargs):
        seen["model"] = args[1] if name == "run_heal" else args[0]
        return original(*args, **kwargs)

    monkeypatch.setattr(wsp.compiler, name, wrapped)
    return seen


def test_lora_only_compile_saves_plain_hf_model(tmp_path, monkeypatch):
    host_dir, _, catalog = save_tiny(tmp_path)
    out = str(tmp_path / "out")
    seen = capture_model(monkeypatch, "run_heal")
    program = Program(host=host_dir).lora(rank=4).heal(steps=3, lr=1e-2, batch_size=2, seq_len=16)
    program.compile(out, catalog, heal_texts=TEXTS, device="cpu")

    report = json.load(open(os.path.join(out, "compile_report.json")))
    assert report["lora"]["rank"] == 4
    assert report["lora"]["alpha"] == 8
    assert report["lora"]["params"] == 4 * 4 * PER_LAYER
    assert report["lora"]["merged"] is True
    assert report["heal"]["trainable_params"] == report["lora"]["params"]
    assert len(report["heal"]["losses"]) == 3
    assert not os.path.exists(os.path.join(out, "lora.json"))
    assert not os.path.exists(os.path.join(out, "grafts.json"))

    model = AutoModelForCausalLM.from_pretrained(out, dtype=torch.float32)
    host = LlamaForCausalLM.from_pretrained(host_dir, dtype=torch.float32)
    assert sorted(model.state_dict().keys()) == sorted(host.state_dict().keys())
    assert not torch.equal(logits(model, "the cat sat"), logits(host, "the cat sat"))
    # Only the decoder linears change. Embeddings and norms stay the host's
    # (in bf16, the dtype compile loads the host in).
    host16 = LlamaForCausalLM.from_pretrained(host_dir, dtype=torch.bfloat16)
    model16 = AutoModelForCausalLM.from_pretrained(out, dtype=torch.bfloat16)
    assert torch.equal(model16.model.embed_tokens.weight, host16.model.embed_tokens.weight)
    assert torch.equal(model16.model.layers[0].input_layernorm.weight, host16.model.layers[0].input_layernorm.weight)
    assert not torch.equal(model16.model.layers[0].self_attn.q_proj.weight, host16.model.layers[0].self_attn.q_proj.weight)

    # The saved model is the merged compile model.
    compiled = seen["model"].float()
    assert len(lora_modules(compiled)) == 0
    saved = AutoModelForCausalLM.from_pretrained(out, dtype=torch.bfloat16).float()
    assert torch.equal(logits(saved, "the cat sat"), logits(compiled, "the cat sat"))


def test_graft_plus_lora_saves_and_load_restores(tmp_path, monkeypatch):
    local_fetch(monkeypatch)
    host_dir, _, catalog = save_tiny(tmp_path)
    out = str(tmp_path / "out")
    seen = capture_model(monkeypatch, "save_lora")
    program = (Program(host=host_dir).graft("qcluster", at=2).lora(rank=2)
               .heal(steps=3, lr=1e-2, batch_size=2, seq_len=16))
    program.compile(out, catalog, heal_texts=TEXTS, device="cpu")

    report = json.load(open(os.path.join(out, "compile_report.json")))
    glue = report["grafts"][0]["glue_params"]
    assert report["heal"]["trainable_params"] == glue + report["lora"]["params"]
    assert report["lora"]["merged"] is False
    info = json.load(open(os.path.join(out, "lora.json")))
    assert info["rank"] == 2
    saved = load_file(os.path.join(out, "lora_weights.safetensors"))
    b_sum = 0.0
    for key in saved:
        if key.endswith("lora_B"):
            b_sum += float(saved[key].abs().sum())
    assert b_sum > 0
    assert saved["model.layers.0.self_attn.q_proj.lora_A"].dtype == torch.float32

    model, _ = load(out)
    compiled = seen["model"]
    assert torch.equal(logits(model, "the cat sat"), logits(compiled, "the cat sat"))
    host = LlamaForCausalLM.from_pretrained(host_dir, dtype=torch.bfloat16)
    assert not torch.equal(logits(model, "the cat sat"), logits(host, "the cat sat"))

    # A later compile without the LoRA into the same directory drops lora.json.
    Program(host=host_dir).graft("qcluster", at=2).compile(out, catalog, device="cpu")
    assert not os.path.exists(os.path.join(out, "lora.json"))
    assert not os.path.exists(os.path.join(out, "lora_weights.safetensors"))


def test_budget_from_graft_report_picks_matched_rank(tmp_path, monkeypatch):
    local_fetch(monkeypatch)
    host_dir, _, catalog = save_tiny(tmp_path)
    graft_out = str(tmp_path / "graft")
    Program(host=host_dir).graft("qcluster", at=2).compile(graft_out, catalog, device="cpu")
    graft_report = json.load(open(os.path.join(graft_out, "compile_report.json")))
    budget = graft_budget(graft_report["grafts"])
    assert budget == graft_report["grafts"][0]["glue_params"] + graft_report["grafts"][0]["part_params"]

    lora_out = str(tmp_path / "lora")
    program = Program(host=host_dir).lora(budget_params=budget).heal(steps=1, batch_size=2, seq_len=16)
    program.compile(lora_out, catalog, heal_texts=TEXTS, device="cpu")
    lora = json.load(open(os.path.join(lora_out, "compile_report.json")))["lora"]
    per_rank = 4 * PER_LAYER
    assert lora["params"] <= budget
    assert lora["params"] + per_rank > budget
    assert lora["budget_params"] == budget


def test_fused_projections_are_default_targets():
    # Phi-3 fuses q, k, v into qkv_proj and gate, up into gate_up_proj.
    # A LoRA that skipped them would not be a fair fine-tuning control.
    from transformers import Phi3Config, Phi3ForCausalLM

    config = Phi3Config(hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=4,
                        num_key_value_heads=4, vocab_size=100, pad_token_id=0)
    torch.manual_seed(0)
    model = Phi3ForCausalLM(config).eval()
    params = apply_lora(model, 2)
    names = []
    for name, _ in lora_modules(model):
        names.append(name.split(".")[-1])
    assert sorted(set(names)) == ["down_proj", "gate_up_proj", "o_proj", "qkv_proj"]
    built = 0
    for param in params:
        built += param.numel()
    assert built == lora_param_count(config, 2)


def test_string_target_is_one_name():
    program = Program(host="some/model").lora(rank=2, targets="q_proj")
    assert program.lora_spec.targets == ["q_proj"]
    model = tiny_model()
    apply_lora(model, 2, targets="q_proj")
    names = []
    for name, _ in lora_modules(model):
        names.append(name.split(".")[-1])
    assert names == ["q_proj"] * 4


def test_lora_linear_passes_weight_through():
    base = nn.Linear(8, 4).to(torch.bfloat16)
    lora = LoRALinear(base, 2)
    assert lora.weight is base.weight
    assert lora.bias is base.bias
