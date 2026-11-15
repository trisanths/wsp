"""One language for every primitive: experts, inserts, gated grafts, distill heal, budget, one directory format."""

import json
import os

import pytest
import torch

transformers = pytest.importorskip("transformers")

import wsp
import wsp.compiler
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast
from wsp import Program, load
from wsp.compiler import CompileError, check_program, count_model_params, random_layer_map
from wsp.lang import LangError, format_program, parse
from tests.test_align import WORDS, tiny_tokenizer
from tests.test_gate_fuse import save_fusion_tiny
from tests.test_mixed_compile import TEXTS, local_fetch

HEAL = dict(steps=3, lr=1e-2, batch_size=2, seq_len=16)


def llama(seed, hidden=64, inter=128, layers=4):
    torch.manual_seed(seed)
    return transformers.LlamaForCausalLM(transformers.LlamaConfig(
        hidden_size=hidden, intermediate_size=inter, num_hidden_layers=layers, num_attention_heads=4,
        num_key_value_heads=2, vocab_size=300, max_position_embeddings=64))


def qwen(seed):
    torch.manual_seed(seed)
    return transformers.Qwen2ForCausalLM(transformers.Qwen2Config(
        hidden_size=48, intermediate_size=96, num_hidden_layers=3, num_attention_heads=4,
        num_key_value_heads=2, vocab_size=300, max_position_embeddings=64))


def save(model, path, tokenizer=None):
    model.save_pretrained(path)
    (tokenizer or tiny_tokenizer()).save_pretrained(path)
    return path


@pytest.fixture
def zoo(tmp_path):
    """A host, two same-shape donors, a Qwen donor of another width, and a deeper same-width donor."""
    return {
        "host": save(llama(0), str(tmp_path / "host")),
        "a": save(llama(1), str(tmp_path / "a")),
        "b": save(llama(2), str(tmp_path / "b")),
        "qwen": save(qwen(3), str(tmp_path / "qwen")),
        "deep": save(llama(4, layers=6), str(tmp_path / "deep")),
        "out": str(tmp_path / "out"),
        "tmp": tmp_path,
    }


@pytest.fixture
def keep_model(monkeypatch):
    """Record the model that a compile writes, so a test can compare it with the reloaded one."""
    box = {}
    original = wsp.compiler.finish_outputs

    def wrapper(program, model, report, output_dir):
        box["model"] = model
        try:
            return original(program, model, report, output_dir)
        finally:
            box["report"] = report

    monkeypatch.setattr(wsp.compiler, "finish_outputs", wrapper)
    return box


def logits(model, text="the cat sat on a mat"):
    ids = tiny_tokenizer()(text, return_tensors="pt").input_ids
    with torch.no_grad():
        return model(ids).logits.float()


def read_report(out):
    with open(os.path.join(out, "compile_report.json")) as f:
        return json.load(f)


# Experts


def test_experts_compile_reload_and_report(zoo, keep_model):
    program = Program(host=zoo["host"]).experts([zoo["a"], zoo["b"]], top_k=2, load_balance=True)
    program.lora(rank=2).heal(**HEAL)
    program.compile(zoo["out"], None, heal_texts=TEXTS, device="cpu")
    for name in ["manifest.json", "skill_moe_weights.safetensors", "program.json", "compile_report.json"]:
        assert os.path.exists(os.path.join(zoo["out"], name)), name
    report = read_report(zoo["out"])
    assert report["params_total"] > report["params_active"] > 0
    assert report["heal_seconds"] >= 0
    assert len(report["heal"]["losses"]) == 3
    assert len(report["heal"]["extra_losses"]) == 3
    assert [row["kind"] for row in report["experts"]["donors"]] == ["plain", "plain"]
    assert report["lora"]["rank"] == 2
    model, tokenizer = load(zoo["out"], dtype=torch.bfloat16)
    assert tokenizer is not None
    assert torch.equal(logits(model), logits(keep_model["model"]))
    assert count_model_params(model) == (report["params_total"], report["params_active"])


def test_fresh_experts_equal_the_host(zoo):
    Program(host=zoo["host"]).experts([zoo["a"], zoo["b"]], top_k=1).compile(zoo["out"], None, device="cpu")
    model, _ = load(zoo["out"], dtype=torch.float32)
    host = transformers.AutoModelForCausalLM.from_pretrained(zoo["host"], dtype=torch.float32)
    assert (logits(model) - logits(host)).abs().max() < 1e-4


def test_experts_random_control_keeps_the_shape_and_changes_the_layers(zoo):
    real = Program(host=zoo["host"]).experts([zoo["a"]], top_k=2)
    real.compile(zoo["out"] + "_real", None, device="cpu")
    control = real.with_control("random", seed=5)
    control.compile(zoo["out"] + "_ctl", None, device="cpu")
    real_map = json.load(open(zoo["out"] + "_real/manifest.json"))["experts"][0]["layer_map"]
    ctl_map = json.load(open(zoo["out"] + "_ctl/manifest.json"))["experts"][0]["layer_map"]
    assert real_map == [0, 1, 2, 3]
    assert len(ctl_map) == 4
    for base_layer in range(4):
        assert ctl_map[base_layer] != base_layer
    assert read_report(zoo["out"] + "_real")["params_total"] == read_report(zoo["out"] + "_ctl")["params_total"]
    # The control reloads from its own manifest.
    model, _ = load(zoo["out"] + "_ctl", dtype=torch.float32)
    assert logits(model).shape[-1] == 300


def test_random_layer_map_is_seeded_and_avoids_the_mapped_layer():
    first = random_layer_map(6, [0, 1, 2, 3], 7)
    assert first == random_layer_map(6, [0, 1, 2, 3], 7)
    assert all(first[i] != i for i in range(4))
    assert random_layer_map(1, [0, 0], 3) == [0, 0]


def test_experts_with_a_donor_of_another_width_are_stitched(zoo, keep_model):
    program = Program(host=zoo["host"]).experts([zoo["qwen"]], top_k=2, stitch="mlp", stitch_hidden=16)
    program.heal(**HEAL)
    program.compile(zoo["out"], None, heal_texts=TEXTS, device="cpu")
    row = json.load(open(os.path.join(zoo["out"], "manifest.json")))["experts"][0]
    assert row["kind"] == "stitched" and row["stitch"] == {"kind": "mlp", "hidden": 16}
    assert row["d_donor"] == 48 and len(row["layer_map"]) == 4
    model, _ = load(zoo["out"], dtype=torch.bfloat16)
    assert torch.equal(logits(model), logits(keep_model["model"]))


# Inserts


def test_inserts_compile_reload_and_gate_starts_closed(zoo, keep_model):
    fresh = str(zoo["tmp"] / "fresh")
    Program(host=zoo["host"]).insert(zoo["qwen"], [0, 1], at=2).compile(fresh, None, device="cpu")
    model, _ = load(fresh, dtype=torch.float32)
    host = transformers.AutoModelForCausalLM.from_pretrained(zoo["host"], dtype=torch.float32)
    assert torch.equal(logits(model), logits(host))

    program = Program(host=zoo["host"]).insert(zoo["qwen"], [0, 1], at=2, part_lora_rank=2).lora(rank=2)
    program.heal(**HEAL)
    program.compile(zoo["out"], None, heal_texts=TEXTS, device="cpu")
    for name in ["insert.json", "insert_weights.safetensors", "program.json", "compile_report.json"]:
        assert os.path.exists(os.path.join(zoo["out"], name)), name
    report = read_report(zoo["out"])
    assert report["inserts"][0]["stitch"] == "linear"
    assert report["params_total"] == report["params_active"] + 300 * 64
    assert report["heal"]["gates"][0] != 0.0
    model, _ = load(zoo["out"], dtype=torch.bfloat16)
    assert torch.equal(logits(model), logits(keep_model["model"]))


def test_insert_of_a_same_width_donor_uses_an_identity_stitch(zoo):
    Program(host=zoo["host"]).insert(zoo["deep"], [1, 2], at=2).compile(zoo["out"], None, device="cpu")
    assert read_report(zoo["out"])["inserts"][0]["stitch"] == "identity"


def test_insert_random_control_uses_another_range_of_the_same_length(zoo):
    program = Program(host=zoo["host"]).insert(zoo["deep"], [0, 1], at=2, control="random", control_seed=3)
    program.compile(zoo["out"], None, device="cpu")
    row = read_report(zoo["out"])["inserts"][0]
    assert row["layers"][1] - row["layers"][0] == 1
    assert row["layers"] != [0, 1] and row["layers"][0] > 1


def test_insert_errors_name_the_problem(zoo):
    with pytest.raises(CompileError, match="has 3 layers"):
        Program(host=zoo["host"]).insert(zoo["qwen"], [1, 5], at=2).compile(zoo["out"], None, device="cpu")
    with pytest.raises(CompileError, match="host has 4 layers"):
        Program(host=zoo["host"]).insert(zoo["qwen"], [0, 1], at=9).compile(zoo["out"], None, device="cpu")
    with pytest.raises(CompileError, match="identity stitch"):
        Program(host=zoo["host"]).insert(zoo["qwen"], [0, 1], at=2, stitch="identity").compile(
            zoo["out"], None, device="cpu")


# Gated grafts


def test_gated_grafts_compile_and_wsp_load_brings_the_gates(tmp_path, monkeypatch, keep_model):
    local_fetch(monkeypatch)
    host_dir, catalog = save_fusion_tiny(tmp_path)
    out = str(tmp_path / "gated")
    program = Program(host=host_dir)
    program.graft("block0", at=1, gate=True, gate_bias=-1.0, gate_lr=5e-2, part_lora_rank=2)
    program.graft("block1", at=2, part_lora_rank=2)
    program.heal(**HEAL)
    program.compile(out, catalog, align_texts=TEXTS, heal_texts=TEXTS, device="cpu")
    for name in ["grafts.json", "gates.json", "gate_weights.safetensors", "graft_weights.safetensors"]:
        assert os.path.exists(os.path.join(out, name)), name
    report = read_report(out)
    assert report["heal"]["gate_params"] == 64 + 1
    assert report["heal"]["gate_lr"] == 5e-2
    assert report["gates"][0]["graft"] == 0 and report["gates"][0]["init_bias"] == -1.0
    model, _ = load(out, dtype=torch.bfloat16)
    assert len(model.wsp_gates) == 1
    assert torch.equal(logits(model), logits(keep_model["model"]))
    ungated, _ = load(out, dtype=torch.bfloat16, gates=False)
    assert getattr(ungated, "wsp_gates", None) is None


def test_gate_settings_must_agree(tmp_path):
    program = Program(host="x").graft("a", gate=True, gate_bias=1.0).graft("b", gate=True)
    with pytest.raises(CompileError, match="share gate_bias"):
        check_program(program)


# Distillation heal


def test_distill_heal_runs_with_the_teacher_and_records_kd(zoo):
    teacher = save(llama(9), str(zoo["tmp"] / "teacher"))
    program = Program(host=zoo["host"]).lora(rank=2).heal(distill=teacher, kd_weight=0.5, **HEAL)
    program.compile(zoo["out"], None, heal_texts=TEXTS, device="cpu")
    heal = read_report(zoo["out"])["heal"]
    assert heal["teacher"] == teacher
    assert len(heal["kd_losses"]) == 3 and len(heal["losses"]) == 3
    assert heal["spec"]["kd_weight"] == 0.5
    model, _ = load(zoo["out"], dtype=torch.float32)
    assert logits(model).shape[-1] == 300


def other_tokenizer():
    vocab = {"<unk>": 0, "</s>": 1, "<pad>": 2}
    for word in reversed(WORDS):
        vocab[word] = len(vocab)
    base = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    base.pre_tokenizer = pre_tokenizers.Whitespace()
    return PreTrainedTokenizerFast(tokenizer_object=base, unk_token="<unk>", eos_token="</s>", pad_token="<pad>")


def test_distill_needs_the_same_tokenizer(zoo, monkeypatch):
    teacher = save(llama(9), str(zoo["tmp"] / "teacher"), other_tokenizer())
    calls = []
    monkeypatch.setattr(wsp.compiler, "run_heal", lambda *args, **kwargs: calls.append(1))
    program = Program(host=zoo["host"]).lora(rank=2).heal(distill=teacher, **HEAL)
    with pytest.raises(CompileError, match="same tokenizer"):
        program.compile(zoo["out"], None, heal_texts=TEXTS, device="cpu")
    assert calls == []


def test_distill_heal_function_checks_the_tokenizer_and_trains():
    from wsp.heal import check_same_tokenizer, distill_heal

    with pytest.raises(ValueError, match="same tokenizer"):
        check_same_tokenizer(tiny_tokenizer(), other_tokenizer(), "teacher-x")
    student = llama(1)
    teacher = llama(2)
    before = student.lm_head.weight.detach().clone()
    for param in student.parameters():
        param.requires_grad_(False)
    student.lm_head.weight.requires_grad_(True)
    result = distill_heal(student, teacher, tiny_tokenizer(), TEXTS, [student.lm_head.weight], steps=4, lr=1e-2,
                          batch_size=2, seq_len=12, kd_weight=1.0, teacher_tokenizer=tiny_tokenizer(),
                          log_every=0)
    assert result["steps"] == 4 and len(result["kd"]) == 4
    assert not torch.equal(before, student.lm_head.weight.detach())
    assert all(not p.requires_grad for p in teacher.parameters())


def test_heal_stops_at_max_seconds():
    from wsp.heal import heal

    model = llama(1)
    params = [model.lm_head.weight]
    assert heal(model, tiny_tokenizer(), TEXTS, params, steps=5, batch_size=2, seq_len=12, log_every=0,
                max_seconds=0.0) == []


# Budget


def test_budget_over_total_stops_before_the_heal_and_reports_the_numbers(zoo, monkeypatch):
    calls = []
    monkeypatch.setattr(wsp.compiler, "run_heal", lambda *args, **kwargs: calls.append(1))
    program = Program(host=zoo["host"]).experts([zoo["a"], zoo["b"]]).heal(**HEAL).budget(max_total_params_b=0.0001)
    with pytest.raises(CompileError) as error:
        program.compile(zoo["out"], None, heal_texts=TEXTS, device="cpu")
    message = str(error.value)
    assert "over its budget" in message and "max_total_params_b 0.0001" in message
    assert "params_total" in message and "params_active" in message
    assert calls == []


def test_budget_over_active_params(zoo):
    host_total = sum(p.numel() for p in llama(0).parameters())
    program = Program(host=zoo["host"]).budget(max_active_params_b=(host_total - 300 * 64) / 1e9 - 1e-6)
    with pytest.raises(CompileError, match="max_active_params_b"):
        program.compile(zoo["out"], None, device="cpu")


def test_budget_met_is_recorded(zoo):
    program = Program(host=zoo["host"]).experts([zoo["a"]]).budget(max_total_params_b=1, max_active_params_b=1)
    program.compile(zoo["out"], None, device="cpu")
    budget = read_report(zoo["out"])["budget"]
    assert budget["ok"] is True and budget["params_total"] > 0
    assert budget["spec"]["max_total_params_b"] == 1


def test_heal_minutes_stop_the_heal_at_the_limit(zoo):
    program = Program(host=zoo["host"]).lora(rank=2).heal(**HEAL).budget(max_heal_minutes=1e-9)
    program.compile(zoo["out"], None, heal_texts=TEXTS, device="cpu")
    report = read_report(zoo["out"])
    assert report["heal"]["stopped_early"] is True and report["heal"]["steps_run"] == 0
    assert report["budget"]["ok"] is True and report["budget"]["heal_stopped_at_limit"] is True


def test_heal_minutes_error_keeps_the_saved_model(zoo, monkeypatch):
    monkeypatch.setattr(wsp.compiler, "heal_limit_seconds", lambda program: None)
    program = Program(host=zoo["host"]).lora(rank=2).heal(**HEAL).budget(max_heal_minutes=1e-9)
    with pytest.raises(CompileError, match="heal took"):
        program.compile(zoo["out"], None, heal_texts=TEXTS, device="cpu")
    assert read_report(zoo["out"])["budget"]["ok"] is False
    assert os.path.exists(os.path.join(zoo["out"], "config.json"))


def test_every_directory_reports_params_and_heal_seconds(zoo):
    Program(host=zoo["host"]).lora(rank=2).heal(**HEAL).compile(zoo["out"], None, heal_texts=TEXTS, device="cpu")
    report = read_report(zoo["out"])
    assert report["params_total"] == report["params_active"] + 300 * 64
    assert report["heal_seconds"] == report["heal"]["seconds"]
    # The LoRA is merged, so a plain model comes back.
    model, _ = load(zoo["out"], dtype=torch.float32)
    assert type(model).__name__ == "LlamaForCausalLM"


# Checks and controls


def test_mixing_component_kinds_is_a_compile_error():
    with pytest.raises(CompileError, match="not a mix"):
        check_program(Program(host="x").experts(["a"]).graft("g"))
    with pytest.raises(CompileError, match="not a mix"):
        check_program(Program(host="x").insert("d", [0, 1], 1).graft("g"))
    with pytest.raises(CompileError, match="taps cannot join"):
        check_program(Program(host="x").experts(["a"]).tap("d", 1, 1))
    with pytest.raises(CompileError, match="gates are not available"):
        check_program(Program(host="x").graft("g", gate=True).tap("d", 1, 1))
    with pytest.raises(CompileError, match="distillation heal is not available"):
        check_program(Program(host="x").tap("d", 1, 1).heal(distill="t"))


def test_with_control_marks_every_component_and_leaves_the_original():
    program = Program(host="x").graft("g1").graft("g2").experts(["a", "b"]).insert("d", [1, 2], 2).heal()
    control = program.with_control("random", seed=10)
    assert [g.control for g in control.grafts] == ["random", "random"]
    assert [g.control_seed for g in control.grafts] == [10, 11]
    assert control.experts_spec.control_seed == 12 and control.inserts[0].control_seed == 13
    assert program.experts_spec.control is None and program.grafts[0].control is None
    with pytest.raises(ValueError, match="no random control"):
        Program(host="x").tv("t").with_control()
    with pytest.raises(ValueError, match="taps have no random control"):
        Program(host="x").tap("d", 1, 1).with_control()


def test_program_json_without_new_parts_is_unchanged():
    data = Program(host="x").graft("g", at=3).heal(steps=5).to_dict()
    for key in ["experts", "inserts", "budget"]:
        assert key not in data
    assert "distill" not in data["heal"] and "gate" not in data["grafts"][0]
    assert Program.from_dict(data).to_dict() == data


def test_spec_validation_messages():
    with pytest.raises(ValueError, match="top_k 5"):
        Program(host="x").experts(["a"], top_k=5)
    with pytest.raises(ValueError, match="lowrank"):
        Program(host="x").experts(["a"], stitch="lowrank")
    with pytest.raises(ValueError, match="ends before"):
        Program(host="x").insert("d", [3, 1], 1)
    with pytest.raises(ValueError, match="kd_weight and temperature need distill"):
        Program(host="x").heal(kd_weight=2.0)
    with pytest.raises(ValueError, match="at least one limit|needs max_total"):
        Program(host="x").budget()
    with pytest.raises(ValueError, match="above 0"):
        Program(host="x").budget(max_heal_minutes=0)


# Text syntax


FULL_TEXT = """host org/host
seed 2
lora rank 4
experts org/a,org/b top_k 2 init uniform balance on balance_weight 0.05 router_lr 0.002 control random seed 4
insert org/d layers 3..5 at 2 stitch lowrank stitch_rank 4 part_lora_rank 2 gate_init 0.1 control random seed 2
insert org/d layers 7 at 3
graft g at 1 gate on gate_bias -1.5 gate_lr 0.05
heal steps 5 data gsm8k_train lr 0.0003 batch_size 8 seq_len 384 distill org/t kd_weight 0.5 temperature 2
budget total 2b active 1.5b heal_minutes 30
"""


def test_text_round_trip_of_every_new_statement():
    program = parse(FULL_TEXT)
    assert program.experts_spec.donors == ["org/a", "org/b"] and program.experts_spec.top_k == 2
    assert program.experts_spec.router_init == "uniform" and program.experts_spec.load_balance is True
    assert program.inserts[0].layers == [3, 5] and program.inserts[1].layers == [7, 7]
    assert program.inserts[0].stitch == "lowrank" and program.inserts[0].control == "random"
    assert program.grafts[0].gate is True and program.grafts[0].gate_bias == -1.5
    assert program.heal_spec.distill == "org/t" and program.heal_spec.temperature == 2
    assert program.budget_spec.max_total_params_b == 2 and program.budget_spec.max_heal_minutes == 30
    text = format_program(program)
    assert parse(text).to_dict() == program.to_dict()
    assert format_program(parse(text)) == text


def test_python_program_formats_and_parses_back():
    program = (Program(host="org/host").experts(["org/a"], top_k=1).insert("org/d", [2, 4], 3, stitch="linear")
               .budget(max_total_params_b=1.5).heal(steps=9, distill="org/t"))
    again = parse(format_program(program))
    assert again.to_dict() == program.to_dict()


@pytest.mark.parametrize("text, line, words", [
    ("host x\nexperts\n", 2, "needs donor models"),
    ("host x\nexperts a top_k two\n", 2, "needs an integer"),
    ("host x\nexperts a color red\n", 2, "unknown experts option"),
    ("host x\nexperts a\nexperts b\n", 3, "appears twice"),
    ("host x\nexperts a top_k 9\n", 2, "top_k 9"),
    ("host x\ninsert d at 2\n", 2, "needs layers"),
    ("host x\ninsert d layers 1..3\n", 2, "needs at"),
    ("host x\ninsert d layers 1,3 at 2\n", 2, "one run"),
    ("host x\ninsert d layers 3..1 at 2\n", 2, "ends before"),
    ("host x\ninsert d layers 1..3 at 2 control fake\n", 2, "unknown control"),
    ("host x\ngraft g gate maybe\n", 2, "on or off"),
    ("host x\nheal kd_weight 2\n", 2, "need distill"),
    ("host x\nheal distill\n", 2, "needs a value"),
    ("host x\nbudget\n", 2, "budget needs"),
    ("host x\nbudget total 0\n", 2, "above 0"),
    ("host x\nbudget heal_minutes -3\n", 2, "above 0"),
    ("host x\nbudget speed 3\n", 2, "unknown budget option"),
    ("host x\nbudget total 1b\nbudget total 2b\n", 3, "appears twice"),
])
def test_new_statement_errors_name_the_line(text, line, words):
    with pytest.raises(LangError) as error:
        parse(text)
    assert error.value.line == line
    assert words in str(error.value)


def test_json_keys_for_new_parts_are_checked():
    from wsp.lang import program_from_dict

    with pytest.raises(ValueError, match="experts key"):
        program_from_dict({"host": "x", "experts": {"donors": ["a"], "colour": 1}})
    with pytest.raises(ValueError, match="insert key"):
        program_from_dict({"host": "x", "inserts": [{"donor": "d", "layers": [0, 1], "at": 1, "colour": 1}]})
    program = program_from_dict(parse(FULL_TEXT).to_dict())
    assert program.budget_spec.max_active_params_b == 1.5


def test_needs_catalog():
    assert Program(host="x").experts(["a"]).needs_catalog() is False
    assert Program(host="x").insert("d", [0, 1], 1).needs_catalog() is False
    assert Program(host="x").graft("g").needs_catalog() is True
    assert Program(host="x").experts(["a"]).graft("g").needs_catalog() is True


# Regressions from the verification pass


def test_heal_without_anything_to_train_is_an_error(zoo):
    program = Program(host=zoo["host"]).heal(**HEAL)
    with pytest.raises(CompileError, match="nothing to train"):
        check_program(program)
    with pytest.raises(CompileError, match="nothing to train"):
        program.compile(zoo["out"], None, heal_texts=TEXTS, device="cpu")
    # A distillation heal is a heal too.
    with pytest.raises(CompileError, match="nothing to train"):
        check_program(Program(host=zoo["host"]).heal(distill="t/u", **HEAL))
    check_program(Program(host=zoo["host"]).lora(rank=2).heal(**HEAL))
    check_program(Program(host=zoo["host"]).experts(["a"]).heal(**HEAL))


def test_with_control_needs_a_component():
    with pytest.raises(ValueError, match="no random control"):
        Program(host="x").lora(rank=2).heal().with_control("random")
    with pytest.raises(ValueError, match="no random control"):
        Program(host="x").with_control("random")
    assert Program(host="x").insert("d", [0, 1], at=1).with_control("random").inserts[0].control == "random"


def test_every_arm_heals_on_the_same_texts_for_one_program_seed(zoo, monkeypatch):
    """A LoRA control and an experts program with the same seed and data name must read the same heal items."""
    import wsp.data
    import wsp.heal

    def fake_splits(seed=0):
        return [], [{"question": f"q{seed}", "answer": "a"}]

    taken = []

    def fake_heal(model, tokenizer, texts, params, **kwargs):
        taken.append(list(texts))
        return []

    def fake_groups(model, tokenizer, texts, groups, **kwargs):
        taken.append(list(texts))
        return {"losses": [], "kd": [], "extra": [], "steps": 0, "stopped_early": False}

    monkeypatch.setattr(wsp.data, "gsm8k_splits", fake_splits)
    monkeypatch.setattr(wsp.heal, "heal", fake_heal)
    monkeypatch.setattr(wsp.heal, "heal_groups", fake_groups)
    # gsm8k_train is a benchmark name, so this test of the shared heal items passes the escape hatch.
    for program in (Program(host=zoo["host"], seed=3).lora(rank=2).heal(data="gsm8k_train", **HEAL),
                    Program(host=zoo["host"], seed=3).experts([zoo["a"]]).heal(data="gsm8k_train", **HEAL)):
        program.compile(zoo["out"], None, device="cpu", allow_benchmark_data=True)
    assert taken == [["Question: q3\nAnswer: a"], ["Question: q3\nAnswer: a"]]


def test_score_reports_the_compiler_param_count(zoo, monkeypatch, capsys):
    from wsp import cli, skills_eval

    Program(host=zoo["host"]).experts([zoo["a"], zoo["b"]], top_k=1).compile(zoo["out"], None, device="cpu")
    report = read_report(zoo["out"])
    monkeypatch.setattr(skills_eval, "evaluate_suite",
                        lambda model, tokenizer, skills, batch_size, limit: {"scores": {"gsm8k": 0.0}, "suite": 0.0,
                                                                              "skills": {}})
    code = cli.main(["score", zoo["out"], "--skills", "gsm8k", "--device", "cpu"])
    row = json.loads(capsys.readouterr().out)
    assert code == 0
    assert row["params_run_per_token"] == report["params_active"]
    assert row["params_total"] == report["params_total"]
    assert report["params_active"] < report["params_total"] - 300 * 64


def test_compile_file_reads_pairs_data_with_the_host_tokenizer(zoo, monkeypatch):
    """A graft program with heal data mmlu_aux needs the host tokenizer for the prompt length limit."""
    import wsp.pairs
    from wsp.lang import compile_file

    seen = {}

    def fake_pairs(tokenizer, seed, count, max_prompt_tokens):
        seen["tokenizer"] = tokenizer
        seen["seed"] = seed
        if max_prompt_tokens is not None and tokenizer is None:
            raise ValueError("mmlu_aux limits prompt length, so it needs the host tokenizer")
        return [("Question: q\\nAnswer:", " A")] * 4

    monkeypatch.setitem(wsp.pairs.REGISTRY, "mmlu_aux", fake_pairs)
    monkeypatch.setattr(wsp.compiler, "compile_program", lambda *a, **k: None)
    catalog_path = str(zoo["tmp"] / "catalog.jsonl")
    open(catalog_path, "w").close()
    program = Program(host=zoo["host"], seed=2).graft("g").heal(data="mmlu_aux", **{k: v for k, v in HEAL.items()})
    out = zoo["out"]
    os.makedirs(out)
    with open(os.path.join(out, "compile_report.json"), "w") as f:
        f.write("{}")
    compile_file(program, out, catalog_path, 2, "cpu", allow_benchmark_data=True)
    assert seen["tokenizer"] is not None and seen["seed"] == 2
