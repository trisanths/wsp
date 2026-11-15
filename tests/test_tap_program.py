"""Taps as part of the language: Program, compile, wsp.load, auto-prepare, text syntax, and the eval backend."""

import json
import os
import sys
import types

import pytest
import torch

transformers = pytest.importorskip("transformers")
from tokenizers import Tokenizer, decoders, models, pre_tokenizers

import wsp.compiler
from wsp import Program, load, tap_cost
from wsp.compiler import CompileError
from wsp.lang import format_program, parse, program_from_dict
from wsp.program import TapSpec, TapTrainSpec
from wsp.tap import clear_tap_context, host_text_and_ends, prepare
from tests.test_tap import PROMPTS, donor_tokenizer, host_tokenizer, tiny_donor, tiny_host

PAIRS = [
    ("the cat sat on", " a mat"),
    ("a dog ran to", " the big red hat"),
    ("what is the", " big cat"),
    ("the red dog", " sat on a mat"),
    ("is the cat big", " the cat is big"),
    ("a big hat", " is red"),
]


def save_models(tmp_path):
    """Save a tiny Llama host (word tokenizer) and a tiny Qwen2 donor (letter tokenizer). Return their dirs."""
    host_dir = str(tmp_path / "host")
    donor_dir = str(tmp_path / "donor")
    tiny_host().save_pretrained(host_dir)
    # Like Llama's tokenizer: no token_type_ids, which generate refuses.
    tokenizer = host_tokenizer()
    tokenizer.model_input_names = ["input_ids", "attention_mask"]
    tokenizer.init_kwargs["model_input_names"] = ["input_ids", "attention_mask"]
    tokenizer.save_pretrained(host_dir)
    tiny_donor().save_pretrained(donor_dir)
    donor_tokenizer().save_pretrained(donor_dir)
    return host_dir, donor_dir


def tap_program(host_dir, donor_dir, lora=True):
    program = Program(host=host_dir, seed=3)
    program.tap(donor_dir, 2, 1, attn_dim=16, heads=4, tap_id="mem2")
    program.tap(donor_dir, 2, 2, attn_dim=16, heads=4)
    program.tap(donor_dir, 3, 2, attn_dim=8, heads=2)
    program.tap_train(data="toy", steps=4, batch_size=3, seq_len=32, branch_lr=3e-3, gate_lr=0.3)
    if lora:
        program.lora(rank=2)
        program.heal(steps=3, data="toy", lr=1e-2, batch_size=3, seq_len=32)
    return program


def compile_tapped(tmp_path, monkeypatch, lora=True):
    """Compile a tap program on CPU. Return (out_dir, the in-memory model, what stage 2 saw)."""
    host_dir, donor_dir = save_models(tmp_path)
    seen = {}
    original_save = wsp.compiler.save_program_taps
    original_heal = wsp.compiler.heal_after_taps

    def save_wrapped(model, output_dir, report):
        seen["model"] = model
        return original_save(model, output_dir, report)

    def heal_wrapped(program, model, *args):
        # Record the state at the start of stage 2.
        gates = []
        for branch in model.wsp_taps:
            gates.append(float(branch.gate.detach()))
        b_sum = 0.0
        for name, param in model.named_parameters():
            if name.endswith("lora_B"):
                b_sum += float(param.detach().abs().sum())
        seen["gates_before_heal"] = gates
        seen["lora_b_before_heal"] = b_sum
        return original_heal(program, model, *args)

    monkeypatch.setattr(wsp.compiler, "save_program_taps", save_wrapped)
    monkeypatch.setattr(wsp.compiler, "heal_after_taps", heal_wrapped)
    out = str(tmp_path / "out")
    tap_program(host_dir, donor_dir, lora).compile(out, None, device="cpu", tap_pairs=PAIRS)
    return out, seen["model"], seen


def left_batch(tokenizer, prompts):
    tokenizer.padding_side = "left"
    return tokenizer(prompts, padding=True, return_tensors="pt")


def manual_logits(model, tokenizer, prompts):
    tokenizer.padding_side = "left"
    inputs = prepare(model, tokenizer, model.wsp_tap_donors, prompts)
    with torch.no_grad():
        logits = model(**inputs).logits.float()
    clear_tap_context(model)
    return logits


def auto_logits(model, tokenizer, prompts):
    encoded = left_batch(tokenizer, prompts)
    with torch.no_grad():
        return model(input_ids=encoded["input_ids"], attention_mask=encoded["attention_mask"]).logits.float()


# Program and JSON


def test_program_round_trip_with_taps():
    program = Program(host="h").tap("Qwen/Qwen2.5-7B-Instruct", 20, 10).tap("org/other", 4, 2, attn_dim=64,
                                                                             heads=4, tap_id="t1")
    program.tap_train(data="trivia_train", steps=100)
    data = program.to_dict()
    assert data["taps"][0] == {"donor": "Qwen/Qwen2.5-7B-Instruct", "donor_layer": 20, "at": 10, "attn_dim": 256,
                               "heads": 8, "tap_id": None}
    assert data["tap_train"] == {"data": "trivia_train", "steps": 100, "batch_size": 8, "seq_len": 256,
                                 "branch_lr": 3e-4, "gate_lr": 3e-2}
    again = Program.from_dict(json.loads(json.dumps(data)))
    assert again.to_dict() == data
    assert program_from_dict(data).to_dict() == data


def test_old_program_json_is_unchanged():
    program = Program(host="h", seed=2).graft("g", at=3).lora(rank=4).heal(steps=10)
    data = program.to_dict()
    assert "taps" not in data
    assert "tap_train" not in data
    assert sorted(data) == ["grafts", "heal", "host", "lora", "masks", "seed", "task_vectors"]
    assert Program.from_dict(data).to_dict() == data


def test_tap_spec_checks():
    with pytest.raises(ValueError, match="multiple"):
        TapSpec("d", 1, 1, attn_dim=10, heads=4)
    with pytest.raises(ValueError, match="at"):
        TapSpec("d", 1, -1)
    with pytest.raises(ValueError, match="steps"):
        TapTrainSpec(steps=-1)


def test_taps_need_tap_train_and_back(tmp_path):
    with pytest.raises(CompileError, match="tap_train"):
        Program(host="some/model").tap("d", 1, 1).compile(str(tmp_path / "a"), None, device="cpu")
    with pytest.raises(CompileError, match="at least one tap"):
        Program(host="some/model").tap_train().compile(str(tmp_path / "b"), None, device="cpu")


# Compile and load


def test_staged_compile_then_load_gives_same_logits(tmp_path, monkeypatch):
    out, compiled, seen = compile_tapped(tmp_path, monkeypatch)
    report = json.load(open(os.path.join(out, "compile_report.json")))

    taps = report["taps"]
    assert len(taps["losses"]) == 4
    assert len(taps["gates"]) == 3
    for gate in taps["gates"]:
        assert gate != 0.0
    assert taps["trainable_params"] > 0
    assert taps["seconds"] >= 0
    assert taps["pairs"] == len(PAIRS)
    assert taps["taps"][0]["tap_id"] == "mem2"
    assert taps["taps"][1]["tap_id"] == "tap1"
    # The first two taps read the same donor layer, so they share one memory.
    indices = []
    for row in taps["taps"]:
        indices.append(row["tap_index"])
    assert indices == [0, 0, 1]
    assert taps["cost"]["tap_params"] == taps["trainable_params"]

    # Staged: stage 1 left the LoRA at zero, and stage 2 left the taps alone.
    assert seen["lora_b_before_heal"] == 0.0
    assert seen["gates_before_heal"] == taps["gates"]
    heal = report["heal"]
    assert heal["staged_after_taps"] is True
    assert len(heal["losses"]) == 3
    assert heal["trainable_params"] == report["lora"]["params"]
    assert report["lora"]["merged"] is False

    info = json.load(open(os.path.join(out, "taps.json")))
    assert info["meta"]["donors"] == [str(tmp_path / "donor")]
    assert json.load(open(os.path.join(out, "grafts.json"))) == []
    assert os.path.exists(os.path.join(out, "lora.json"))

    model, tokenizer = load(out)
    expected = manual_logits(compiled, tokenizer, PROMPTS)
    assert torch.equal(manual_logits(model, tokenizer, PROMPTS), expected)
    assert torch.equal(auto_logits(model, tokenizer, PROMPTS), expected)
    assert torch.equal(auto_logits(compiled, tokenizer, PROMPTS), expected)

    # The taps and the LoRA change the output.
    host = transformers.AutoModelForCausalLM.from_pretrained(str(tmp_path / "host"), dtype=torch.bfloat16)
    encoded = left_batch(tokenizer, PROMPTS)
    with torch.no_grad():
        plain = host(**encoded).logits.float()
    assert not torch.allclose(plain, expected, atol=1e-3)


def test_tap_only_program_loads(tmp_path, monkeypatch):
    out, compiled, seen = compile_tapped(tmp_path, monkeypatch, lora=False)
    report = json.load(open(os.path.join(out, "compile_report.json")))
    assert report["heal"] is None
    assert "gates_before_heal" not in seen
    assert not os.path.exists(os.path.join(out, "lora.json"))
    model, tokenizer = load(out)
    assert torch.equal(auto_logits(model, tokenizer, PROMPTS), manual_logits(compiled, tokenizer, PROMPTS))

    # A later compile without taps into the same directory drops the tap files.
    Program(host=str(tmp_path / "host")).compile(out, None, device="cpu")
    assert not os.path.exists(os.path.join(out, "taps.json"))
    assert not os.path.exists(os.path.join(out, "grafts.json"))


# Auto-prepare


@pytest.fixture
def loaded(tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp("tapped")
    host_dir, donor_dir = save_models(tmp_path)
    out = str(tmp_path / "out")
    tap_program(host_dir, donor_dir, lora=False).compile(out, None, device="cpu", tap_pairs=PAIRS)
    model, tokenizer = load(out, dtype=torch.float32)
    return out, model, tokenizer


def test_auto_prepare_equals_manual_on_left_padded_batch(loaded):
    out, model, tokenizer = loaded
    expected = manual_logits(model, tokenizer, PROMPTS)
    got = auto_logits(model, tokenizer, PROMPTS)
    assert torch.equal(got, expected)

    # Right padding works too, row by row.
    tokenizer.padding_side = "right"
    encoded = tokenizer(PROMPTS, padding=True, return_tensors="pt")
    with torch.no_grad():
        right = model(**encoded).logits.float()
    for row in range(len(PROMPTS)):
        n = int(encoded["attention_mask"][row].sum())
        left_start = expected.shape[1] - n
        assert torch.allclose(right[row, :n], expected[row, left_start:], atol=1e-5)

    # A new prefill builds a new context: one prompt alone gives its own row.
    single = auto_logits(model, tokenizer, [PROMPTS[1]])
    n = single.shape[1]
    assert torch.allclose(single[0], expected[1, expected.shape[1] - n:], atol=1e-5)


def test_manual_context_takes_priority(loaded):
    out, model, tokenizer = loaded
    tokenizer.padding_side = "left"
    pool = model.wsp_tap_donors
    # Manual prepare with mismatched memories (derange) must win over auto.
    inputs = prepare(model, tokenizer, pool, PROMPTS, derange=True)
    with torch.no_grad():
        manual = model(**inputs).logits.float()
    clear_tap_context(model)
    auto = auto_logits(model, tokenizer, PROMPTS)
    assert not torch.allclose(manual, auto, atol=1e-4)
    with torch.no_grad():
        again = model(**prepare(model, tokenizer, pool, PROMPTS, derange=True)).logits.float()
    assert torch.equal(again, manual)
    clear_tap_context(model)


def generate(model, tokenizer, prompts, auto, **settings):
    tokenizer.padding_side = "left"
    if auto:
        encoded = tokenizer(prompts, padding=True, return_tensors="pt")
        inputs = {"input_ids": encoded["input_ids"], "attention_mask": encoded["attention_mask"]}
    else:
        inputs = prepare(model, tokenizer, model.wsp_tap_donors, prompts)
    with torch.no_grad():
        out = model.generate(**inputs, do_sample=False, pad_token_id=tokenizer.pad_token_id, **settings)
    if not auto:
        clear_tap_context(model)
    return out


def test_generate_with_cache_equals_without(loaded):
    out, model, tokenizer = loaded
    settings = {"max_new_tokens": 6, "output_scores": True, "return_dict_in_generate": True}
    cached = generate(model, tokenizer, PROMPTS, True, use_cache=True, **settings)
    uncached = generate(model, tokenizer, PROMPTS, True, use_cache=False, **settings)
    manual = generate(model, tokenizer, PROMPTS, False, use_cache=True, **settings)
    assert torch.equal(cached.sequences, uncached.sequences)
    assert torch.equal(cached.sequences, manual.sequences)
    for index in range(len(cached.scores)):
        assert torch.allclose(cached.scores[index], uncached.scores[index], atol=1e-5)
        assert torch.equal(cached.scores[index], manual.scores[index])


def test_beam_search_with_auto_prepare(loaded):
    out, model, tokenizer = loaded
    settings = {"max_new_tokens": 5, "num_beams": 3, "num_return_sequences": 2}
    cached = generate(model, tokenizer, PROMPTS, True, use_cache=True, **settings)
    uncached = generate(model, tokenizer, PROMPTS, True, use_cache=False, **settings)
    manual = generate(model, tokenizer, PROMPTS, False, use_cache=True, **settings)
    assert cached.shape[0] == 6
    assert torch.equal(cached, uncached)
    assert torch.equal(cached, manual)


def test_eval_hf_backend_matches_manual_prepare(loaded, monkeypatch):
    from wsp_eval import backends

    out, model, tokenizer = loaded
    monkeypatch.setattr(backends.HFBackend, "_device", lambda self: "cpu")
    backend = backends.HFBackend(batch_size=8)
    prompts = []
    for prompt in PROMPTS:
        prompts.append(backends.raw_prompt(backend_tokenizer(backend, out), prompt))
    assert backends.is_grafted(out)
    outputs = backend.generate(out, prompts, 5, "tiny")

    # The same model, prompts, and batch, with manual prepare.
    loaded_model = backend.model
    loaded_tok = backend.tokenizer
    order = sorted(range(len(PROMPTS)), key=lambda i: len(prompts[i]))
    batch = []
    rendered = []
    for i in order:
        batch.append(PROMPTS[i])
        rendered.append(prompts[i])
    loaded_tok.padding_side = "left"
    inputs = prepare(loaded_model, loaded_tok, loaded_model.wsp_tap_donors, batch)
    encoded = loaded_tok(rendered, return_tensors="pt", padding=True, add_special_tokens=False)
    assert torch.equal(inputs["input_ids"], encoded["input_ids"])
    with torch.no_grad():
        generated = loaded_model.generate(**inputs, max_new_tokens=5, do_sample=False,
                                          pad_token_id=loaded_tok.pad_token_id)
    clear_tap_context(loaded_model)
    width = inputs["input_ids"].shape[1]
    for row, i in enumerate(order):
        expected = loaded_tok.decode(generated[row, width:], skip_special_tokens=True)
        assert outputs[i] == expected
    backend.unload()


def backend_tokenizer(backend, out):
    backend._load(out)
    return backend.tokenizer


# Rebuilding the text


def fused_tokenizer():
    """A BPE with the merge a + b -> ab, whose decoder joins tokens with no space."""
    vocab = {"<s>": 0, "a": 1, "b": 2, "ab": 3, "c": 4}
    base = Tokenizer(models.BPE(vocab=vocab, merges=[("a", "b")]))
    base.pre_tokenizer = pre_tokenizers.Whitespace()
    base.decoder = decoders.Fuse()
    return transformers.PreTrainedTokenizerFast(tokenizer_object=base, bos_token="<s>", eos_token="<s>")


def test_text_and_ends_for_ids_that_do_not_round_trip():
    tok = fused_tokenizer()
    # a, b, c encode back as ab, c, so the ends come from decoded prefixes.
    text, ends = host_text_and_ends(tok, [0, 1, 2, 4], [True, True, True, True])
    assert text == "abc"
    assert ends == [-1, 1, 2, 3]
    # ab, c round trip, so the ends are the tokenizer's own offsets. A pad is -1.
    text, ends = host_text_and_ends(tok, [0, 0, 3, 4], [False, True, True, True])
    assert text == "abc"
    assert ends == [-1, -1, 2, 3]


def test_text_and_ends_match_prepare_offsets():
    tok = host_tokenizer()
    tok.padding_side = "left"
    encoded = tok(PROMPTS, padding=True, return_offsets_mapping=True, return_special_tokens_mask=True)
    for row in range(len(PROMPTS)):
        real = []
        for value in encoded["attention_mask"][row]:
            real.append(value == 1)
        text, ends = host_text_and_ends(tok, encoded["input_ids"][row], real)
        assert text == PROMPTS[row]
        for index in range(len(ends)):
            if encoded["special_tokens_mask"][row][index] == 1:
                assert ends[index] == -1
            else:
                assert ends[index] == encoded["offset_mapping"][row][index][1]


# Cost


def test_tap_cost_counts_branches_and_donors(loaded):
    out, model, tokenizer = loaded
    cost = tap_cost(model, memory_tokens=10)
    expected_fixed = 0
    expected_per_memory = 0
    for branch in model.wsp_taps:
        expected_fixed += 4 * branch.d_host * branch.attn_dim
        expected_per_memory += 4 * branch.d_donor * branch.attn_dim + 4 * branch.attn_dim
    assert cost["tap_decode_flops_fixed"] == expected_fixed
    assert cost["tap_decode_flops_per_token"] == expected_fixed + 10 * expected_per_memory
    assert len(cost["donors"]) == 1
    donor = cost["donors"][0]
    # Taps read donor layers 2 and 3, so layers 0 to 3 run, out of 4.
    assert donor["layers_ran"] == 4
    assert donor["layers_total"] == 4
    assert donor["prefill_flops_per_prompt"] == 2 * donor["params_ran"] * 10
    assert cost["tap_decode_fraction"] > 0


# Text syntax


def test_lang_round_trip_with_taps():
    program = Program(host="unsloth/Llama-3.2-1B-Instruct")
    program.tap("Qwen/Qwen2.5-7B-Instruct", 20, 10)
    program.tap("LiquidAI/LFM2-350M-Math", 8, 8, attn_dim=128, heads=4, tap_id="lfm")
    program.tap_train(data="mmlu_aux", steps=500, branch_lr=3e-4, gate_lr=3e-2)
    program.lora(rank=16).heal(steps=500, data="mmlu_aux", lr=1e-4)
    text = format_program(program)
    assert "tap Qwen/Qwen2.5-7B-Instruct layer 20 at 10\n" in text
    assert "tap LiquidAI/LFM2-350M-Math layer 8 at 8 dim 128 heads 4 id lfm\n" in text
    assert "tap_train data mmlu_aux steps 500 batch_size 8 seq_len 256 branch_lr 0.0003 gate_lr 0.03\n" in text
    again = parse(text)
    assert again.to_dict() == program.to_dict()
    assert format_program(again) == text

    short = parse("host h\ntap d layer 3 at 2 attn_dim 32 heads 2 tap_id x\ntap_train\n")
    assert short.taps[0].to_dict() == {"donor": "d", "donor_layer": 3, "at": 2, "attn_dim": 32, "heads": 2,
                                       "tap_id": "x"}
    assert short.tap_train_spec.to_dict() == TapTrainSpec().to_dict()


@pytest.mark.parametrize("text, words", [
    ("host h\ntap d at 2\n", "layer"),
    ("host h\ntap d layer 2\n", "at N"),
    ("host h\ntap d layer 2 at 1 dim 10 heads 4\n", "multiple"),
    ("host h\ntap_train\ntap_train\n", "twice"),
    ("host h\ntap d layer 2 at 1 size 3\n", "unknown tap option"),
])
def test_lang_tap_errors(text, words):
    from wsp.lang import LangError

    with pytest.raises(LangError, match=words):
        parse(text)


# Pairs


def fake_datasets(tables):
    module = types.ModuleType("datasets")

    def load_dataset(name, config, split):
        return tables[(name, config, split)]

    module.load_dataset = load_dataset
    return module


def test_pairs_registry(monkeypatch):
    from wsp import pairs

    mmlu_rows = [
        {"question": " What is two? ", "choices": ["1", "2", "3", "4"], "answer": 1},
        {"question": "Three choices", "choices": ["x", "y", "z"], "answer": 0},
        {"train": {"question": "Nested", "choices": ["p", "q", "r", "s"], "answer": 3}},
        {"question": "the " * 50, "choices": ["a", "b", "c", "d"], "answer": 2},
    ]
    trivia_rows = [
        {"question": " Capital of France? ", "answer": {"value": " Paris "}},
        {"question": "Red planet?", "answer": {"value": "Mars"}},
    ]
    tables = {("cais/mmlu", "auxiliary_train", "train"): mmlu_rows,
              ("mandarjoshi/trivia_qa", "rc.nocontext", "train"): trivia_rows}
    monkeypatch.setitem(sys.modules, "datasets", fake_datasets(tables))

    tok = host_tokenizer()
    got = pairs.pairs_for("mmlu_aux", tok, max_prompt_tokens=30)
    assert sorted(got) == sorted([
        ("Question: What is two?\nA. 1\nB. 2\nC. 3\nD. 4\nAnswer:", " B"),
        ("Question: Nested\nA. p\nB. q\nC. r\nD. s\nAnswer:", " D"),
    ])
    assert len(pairs.pairs_for("mmlu_aux", tok, max_prompt_tokens=None)) == 3
    assert len(pairs.pairs_for("mmlu_aux", tok, count=1, max_prompt_tokens=None)) == 1

    got = pairs.pairs_for("trivia_train")
    assert sorted(got) == [("Question: Capital of France?\nAnswer:", " Paris"),
                           ("Question: Red planet?\nAnswer:", " Mars")]

    import wsp.data

    items = [{"question": "Q one", "answer": "step\n#### 4"}]
    monkeypatch.setattr(wsp.data, "gsm8k_splits", lambda seed=0: ([], items))
    assert pairs.pairs_for("gsm8k_train") == [("Question: Q one\nAnswer:", " step\n#### 4")]

    with pytest.raises(ValueError, match="unknown tap data"):
        pairs.pairs_for("nope")
    with pytest.raises(ValueError, match="tokenizer"):
        pairs.pairs_for("mmlu_aux")
