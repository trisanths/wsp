"""Review checks for the staged tap compile: freezes, learning rates, early name checks, and host/ loads."""

import json
import os

import pytest
import torch

transformers = pytest.importorskip("transformers")

import wsp.compiler
import wsp.data
import wsp.lang
from wsp import Program, load
from wsp.compiler import CompileError
from wsp.lang import LangError, parse, program_from_dict
from wsp.program import TapTrainSpec
from tests.test_tap import PROMPTS
from tests.test_tap_program import PAIRS, manual_logits, save_models


def tap_state(model):
    state = {}
    for index, branch in enumerate(model.wsp_taps):
        for key, value in branch.state_dict().items():
            state[f"{index}.{key}"] = value.detach().clone()
    return state


def lora_state(model):
    state = {}
    for name, param in model.named_parameters():
        if "lora_" in name:
            state[name] = param.detach().clone()
    return state


def same(a, b):
    if sorted(a) != sorted(b):
        return False
    for key in a:
        if not torch.equal(a[key], b[key]):
            return False
    return True


def small_program(host_dir, donor_dir, branch_lr=3e-3, gate_lr=0.3, lora=True, heal_data="toy"):
    program = Program(host=host_dir, seed=3)
    program.tap(donor_dir, 2, 1, attn_dim=16, heads=4)
    program.tap(donor_dir, 3, 2, attn_dim=8, heads=2)
    program.tap_train(data="toy", steps=3, batch_size=3, seq_len=32, branch_lr=branch_lr, gate_lr=gate_lr)
    if lora:
        program.lora(rank=2)
        program.heal(steps=3, data=heal_data, lr=1e-2, batch_size=3, seq_len=32)
    return program


def record_stages(monkeypatch):
    """Record the tap and LoRA tensors before stage 1, after stage 1, and at save."""
    seen = {}
    original_train = wsp.compiler.train_program_taps
    original_save = wsp.compiler.save_program_taps

    def train_wrapped(program, model, *args):
        seen["taps_start"] = tap_state(model)
        seen["lora_start"] = lora_state(model)
        result = original_train(program, model, *args)
        seen["taps_stage1"] = tap_state(model)
        seen["lora_stage1"] = lora_state(model)
        return result

    def save_wrapped(model, output_dir, report):
        seen["model"] = model
        seen["taps_end"] = tap_state(model)
        seen["lora_end"] = lora_state(model)
        return original_save(model, output_dir, report)

    monkeypatch.setattr(wsp.compiler, "train_program_taps", train_wrapped)
    monkeypatch.setattr(wsp.compiler, "save_program_taps", save_wrapped)
    return seen


def test_stage_two_keeps_every_tap_tensor_and_trains_only_the_lora(tmp_path, monkeypatch):
    host_dir, donor_dir = save_models(tmp_path)
    seen = record_stages(monkeypatch)
    small_program(host_dir, donor_dir).compile(str(tmp_path / "out"), None, device="cpu", tap_pairs=PAIRS)
    assert not same(seen["taps_start"], seen["taps_stage1"])
    assert same(seen["lora_start"], seen["lora_stage1"])
    # Every tap tensor, not only the gates, is the same after stage 2.
    assert same(seen["taps_stage1"], seen["taps_end"])
    assert not same(seen["lora_stage1"], seen["lora_end"])


def test_gates_and_branch_weights_use_their_own_lr(tmp_path, monkeypatch):
    host_dir, donor_dir = save_models(tmp_path)
    seen = record_stages(monkeypatch)
    small_program(host_dir, donor_dir, branch_lr=0.0, gate_lr=0.3, lora=False).compile(
        str(tmp_path / "a"), None, device="cpu", tap_pairs=PAIRS)
    for key in seen["taps_start"]:
        moved = not torch.equal(seen["taps_start"][key], seen["taps_stage1"][key])
        assert moved == key.endswith("gate"), key

    seen = record_stages(monkeypatch)
    small_program(host_dir, donor_dir, branch_lr=3e-3, gate_lr=0.0, lora=False).compile(
        str(tmp_path / "b"), None, device="cpu", tap_pairs=PAIRS)
    for key in seen["taps_stage1"]:
        if key.endswith("gate"):
            assert float(seen["taps_stage1"][key]) == 0.0


def test_bad_pairs_names_fail_before_any_model_loads(tmp_path, monkeypatch):
    def no_load(*args, **kwargs):
        raise AssertionError("a model loaded before the pairs names were checked")

    monkeypatch.setattr(transformers.AutoModelForCausalLM, "from_pretrained", no_load)
    program = small_program("some/host", "some/donor", heal_data="not_a_pairs_name")
    with pytest.raises(CompileError, match="heal data 'not_a_pairs_name'"):
        program.compile(str(tmp_path / "a"), None, device="cpu", tap_pairs=PAIRS)
    program = small_program("some/host", "some/donor")
    program.tap_train_spec.data = "no_such_pairs"
    with pytest.raises(CompileError, match="tap_train data 'no_such_pairs'"):
        program.compile(str(tmp_path / "b"), None, device="cpu")
    # A heal that does not run (no LoRA, no grafts) is not checked.
    program = small_program("some/host", "some/donor", lora=False)
    program.heal(steps=1, data="not_a_pairs_name")
    with pytest.raises(AssertionError, match="a model loaded"):
        program.compile(str(tmp_path / "c"), None, device="cpu", tap_pairs=PAIRS)


def test_tap_train_lr_must_be_a_number():
    for value in ["0.03", None, True, -1.0, float("nan"), float("inf")]:
        with pytest.raises(ValueError, match="gate_lr"):
            TapTrainSpec(gate_lr=value)
        with pytest.raises(ValueError, match="branch_lr"):
            TapTrainSpec(branch_lr=value)
    data = Program(host="h").tap("d", 1, 2).tap_train().to_dict()
    data["tap_train"]["gate_lr"] = "0.03"
    with pytest.raises(ValueError, match="gate_lr"):
        program_from_dict(data)
    with pytest.raises(LangError, match="branch_lr"):
        parse("host h\ntap d layer 1 at 2\ntap_train branch_lr -1e-3\n")
    assert TapTrainSpec(branch_lr=0, gate_lr=1).to_dict()["gate_lr"] == 1


def test_cli_align_texts_for_a_tap_and_graft_program(tmp_path, monkeypatch):
    names = []

    def fake_texts(name, seed=0):
        names.append(name)
        if name != "gsm8k_train":
            raise ValueError(f"unknown heal data: {name}")
        return ["a text"] * 5

    def fake_compile(program, catalog, out_dir, **options):
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "compile_report.json"), "w") as f:
            json.dump({"align": len(options["align_texts"])}, f)

    monkeypatch.setattr(wsp.data, "task_texts", fake_texts)
    monkeypatch.setattr(wsp.compiler, "compile_program", fake_compile)
    catalog = tmp_path / "catalog.jsonl"
    catalog.write_text("")
    program = parse("host h\ngraft g at 1\nlora rank 2\ntap d layer 1 at 2\ntap_train data mmlu_aux\n"
                    "heal data mmlu_aux\n")
    report = wsp.lang.compile_file(program, str(tmp_path / "out"), str(catalog), 3, "cpu")
    assert names == ["gsm8k_train"]
    assert report["align"] == 3


class TaskVectorCatalog:
    def __init__(self, host_dir, tuned_dir):
        self.host_dir = host_dir
        self.tuned_dir = tuned_dir

    def get(self, component_id):
        return {"type": "task_vector", "base": self.host_dir, "model": self.tuned_dir}


def test_task_vector_host_with_taps_loads_from_host_dir(tmp_path, monkeypatch):
    host_dir, donor_dir = save_models(tmp_path)
    tuned_dir = str(tmp_path / "tuned")
    torch.manual_seed(0)
    tuned = transformers.AutoModelForCausalLM.from_pretrained(host_dir)
    with torch.no_grad():
        for param in tuned.parameters():
            param.add_(0.05 * torch.randn_like(param))
    tuned.save_pretrained(tuned_dir)

    seen = record_stages(monkeypatch)
    program = small_program(host_dir, donor_dir).tv("tv1", weight=1.0)
    out = str(tmp_path / "out")
    program.compile(out, TaskVectorCatalog(host_dir, tuned_dir), device="cpu", tap_pairs=PAIRS)
    assert os.path.isdir(os.path.join(out, "host"))
    model, tokenizer = load(out)
    assert same(tap_state(model), seen["taps_end"])
    expected = manual_logits(seen["model"], tokenizer, PROMPTS)
    assert torch.equal(manual_logits(model, tokenizer, PROMPTS), expected)
