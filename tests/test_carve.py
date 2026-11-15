"""Tests for wsp.carve and written programs, on tiny random models (CPU, no network)."""

import json
import os

import pytest
import torch

transformers = pytest.importorskip("transformers")

from wsp import Program, load
from wsp.carve import (carve_layers, carve_width, choose_layers, count_params, domain_influence, layer_influence,
                       layers_for_budget, params_for_layers, splice)
from wsp.compiler import CompileError
from wsp.lang import LangError, format_program, parse, program_from_dict
from wsp.parts import find_decoder_layers
from tests.test_align import WORDS, tiny_tokenizer

TEXTS = []
for i in range(12):
    TEXTS.append(" ".join(WORDS[i % 6: i % 6 + 6]))
DOMAIN = ["the cat sat on a mat", "a big red hat", "the dog ran to a cat"]
IDS = torch.tensor([[3, 7, 4, 9, 11, 5, 6, 8, 10, 12, 13, 14, 3, 4]])


def tiny(kind, layers=6, seed=0, intermediate=48):
    torch.manual_seed(seed)
    if kind == "llama":
        config = transformers.LlamaConfig(
            hidden_size=32, intermediate_size=intermediate, num_hidden_layers=layers, num_attention_heads=8,
            num_key_value_heads=2, vocab_size=40, max_position_embeddings=128,
        )
        return transformers.LlamaForCausalLM(config).eval()
    if kind == "qwen2":
        # Sliding window on the later layers, so max_window_layers matters.
        config = transformers.Qwen2Config(
            hidden_size=32, intermediate_size=intermediate, num_hidden_layers=layers, num_attention_heads=4,
            num_key_value_heads=2, vocab_size=40, max_position_embeddings=128, use_sliding_window=True,
            sliding_window=6, max_window_layers=3,
        )
        return transformers.Qwen2ForCausalLM(config).eval()
    config = transformers.Gemma2Config(
        hidden_size=32, intermediate_size=intermediate, num_hidden_layers=layers, num_attention_heads=4,
        num_key_value_heads=2, head_dim=8, vocab_size=40, max_position_embeddings=128, sliding_window=4,
        query_pre_attn_scalar=8, pad_token_id=1,
    )
    return transformers.Gemma2ForCausalLM(config).eval()


KINDS = ["llama", "qwen2", "gemma2"]


def logits(model, ids=IDS):
    with torch.no_grad():
        return model(ids).logits.float()


def save_and_reload(model, path):
    model.save_pretrained(str(path))
    return transformers.AutoModelForCausalLM.from_pretrained(str(path)).eval()


def layer_state(layer):
    state = {}
    for name, value in layer.state_dict().items():
        state[name] = value.clone()
    return state


@pytest.mark.parametrize("kind", KINDS)
def test_carve_layers_reloads_with_equal_logits(kind, tmp_path):
    model = tiny(kind)
    old_types = getattr(model.config, "layer_types", None)
    keep = [0, 2, 3, 5]
    carve_layers(model, keep)
    assert model.config.num_hidden_layers == 4
    _, layers = find_decoder_layers(model, 0)
    for index in range(len(layers)):
        assert layers[index].self_attn.layer_idx == index
    if old_types is not None:
        expected = []
        for index in keep:
            expected.append(old_types[index])
        assert model.config.layer_types == expected
    if kind == "qwen2":
        # Layers 0 and 2 were full attention (before max_window_layers 3).
        assert model.config.max_window_layers == 2
    reloaded = save_and_reload(model, tmp_path / "carved")
    assert type(reloaded) is type(model)
    assert reloaded.config.num_hidden_layers == 4
    assert torch.allclose(logits(reloaded), logits(model), atol=1e-6)


@pytest.mark.parametrize("kind", KINDS)
def test_carved_model_counts_params_and_generates_with_cache(kind):
    model = tiny(kind)
    predicted = params_for_layers(model, [0, 4, 5])
    carve_layers(model, [0, 4, 5])
    assert count_params(model) == predicted
    with torch.no_grad():
        out = model(IDS, use_cache=True)
    assert len(out.past_key_values.layers) == 3
    mask = torch.ones_like(IDS)
    cached = model.generate(IDS, attention_mask=mask, max_new_tokens=6, do_sample=False, use_cache=True,
                            pad_token_id=0)
    uncached = model.generate(IDS, attention_mask=mask, max_new_tokens=6, do_sample=False, use_cache=False,
                              pad_token_id=0)
    assert torch.equal(cached, uncached)
    assert cached.shape[1] == IDS.shape[1] + 6


def test_param_count_includes_tied_and_untied_embeddings():
    model = tiny("gemma2")
    # Gemma2 ties the head to the embedding, so it counts once.
    assert model.config.tie_word_embeddings
    total = 0
    for param in model.parameters():
        total += param.numel()
    assert count_params(model) == total
    _, layers = find_decoder_layers(model, 0)
    outside = total
    for layer in layers:
        for param in layer.parameters():
            outside -= param.numel()
    assert outside == 40 * 32 + 32  # the embedding and the final norm
    untied = tiny("llama")
    assert not untied.config.tie_word_embeddings
    assert params_for_layers(untied, []) == 2 * 40 * 32 + 32


@pytest.mark.parametrize("kind", KINDS)
def test_layers_for_budget_respects_the_target(kind):
    model = tiny(kind)
    full = count_params(model)
    assert layers_for_budget(model, full) == 6
    for target in [full * 0.5, full * 0.75, full - 1]:
        keep = layers_for_budget(model, target)
        assert params_for_layers(model, list(range(keep))) <= target
        assert params_for_layers(model, list(range(keep + 1))) > target
    keep = layers_for_budget(model, full * 0.6, keep_mlp=0.5, keep_heads=0.5)
    carve_layers(model, list(range(keep)))
    carve_width(model, 0.5, 0.5)
    assert count_params(model) <= full * 0.6
    with pytest.raises(ValueError, match="does not fit 2 layers"):
        layers_for_budget(tiny(kind), 100)


@pytest.mark.parametrize("kind", KINDS)
def test_carve_width_reloads_as_a_plain_model(kind, tmp_path):
    model = tiny(kind)
    before = logits(model)
    same, report = carve_width(model, 1.0, 1.0)
    assert report["params_after"] == report["params_before"]
    reloaded = save_and_reload(same, tmp_path / "full")
    assert torch.allclose(logits(reloaded), before, atol=1e-6)

    model = tiny(kind)
    full = count_params(model)
    predicted = params_for_layers(model, list(range(6)), 0.5, 0.5)
    carve_width(model, 0.5, 0.5)
    assert count_params(model) == predicted
    assert count_params(model) < full
    assert model.config.intermediate_size == 24
    reloaded = save_and_reload(model, tmp_path / "half")
    assert type(reloaded) is type(model)
    assert count_params(reloaded) == count_params(model)
    assert torch.allclose(logits(reloaded), logits(model), atol=1e-6)


def test_carve_width_keeps_the_highest_scores():
    model = tiny("llama")
    scores = {"mlp": {}, "heads": {}}
    for index in range(6):
        values = torch.zeros(48)
        values[40:] = 1.0
        values[0:4] = 2.0
        scores["mlp"][index] = values
        scores["heads"][index] = torch.arange(8, dtype=torch.float32)
    down = model.model.layers[2].mlp.down_proj.weight.detach().clone()
    _, report = carve_width(model, 0.25, 0.5, scores)
    neurons = report["layers"][2]["neurons"]
    assert neurons == [0, 1, 2, 3, 40, 41, 42, 43, 44, 45, 46, 47]
    assert torch.equal(model.model.layers[2].mlp.down_proj.weight, down[:, neurons])
    # Half of 8 heads over 2 KV heads keeps both KV heads and the best 2 query heads of each.
    assert report["layers"][2]["kv_heads"] == [0, 1]
    assert report["layers"][2]["q_heads"] == [2, 3, 6, 7]


def test_choose_layers_rules():
    scores = [0.9, 0.1, 0.5, 0.2, 0.8, 0.3]
    assert choose_layers(scores, 4, "influence") == [0, 2, 4, 5]
    assert choose_layers(scores, 6, "influence") == [0, 1, 2, 3, 4, 5]
    assert choose_layers(scores, 3, "uniform") == [0, 3, 5]
    assert choose_layers(6, 4, "uniform") == [0, 2, 3, 5]
    domain = {"domain": [0, 0.9, 0.1, 0.0, 0.0, 0], "general": [0, 0.0, 0.2, 0.3, 0.1, 0], "difference": None}
    assert choose_layers(domain, 3, "domain", weight=1.0) == [0, 1, 5]
    assert choose_layers(domain, 3, "domain", weight=0.0) == [0, 3, 5]
    with pytest.raises(ValueError, match="unknown select rule"):
        choose_layers(scores, 3, "best")
    with pytest.raises(ValueError, match="keep is an integer"):
        choose_layers(scores, 1)
    with pytest.raises(ValueError, match="domain_influence"):
        choose_layers(scores, 3, "domain")


def test_layer_influence_and_domain_influence():
    model = tiny("llama")
    tokenizer = tiny_tokenizer()
    scores = layer_influence(model, tokenizer, TEXTS)
    assert len(scores) == 6
    for value in scores:
        assert 0 <= value <= 2
    both = domain_influence(model, tokenizer, DOMAIN, TEXTS)
    for index in range(6):
        assert both["difference"][index] == pytest.approx(both["domain"][index] - both["general"][index])
    # A layer whose output is its input has no influence.
    layer = model.model.layers[3]
    with torch.no_grad():
        layer.self_attn.o_proj.weight.zero_()
        layer.mlp.down_proj.weight.zero_()
    assert layer_influence(model, tokenizer, TEXTS)[3] == pytest.approx(0.0, abs=1e-6)


@pytest.mark.parametrize("kind", ["qwen2", "gemma2"])
def test_splice_keeps_the_planned_layers_in_order(kind, tmp_path):
    a = tiny(kind, seed=0)
    b = tiny(kind, seed=1)
    plan = [("a", 0), ("a", 1), ("b", 2), ("b", 3), ("a", 5), ("a", 5)]
    expected = []
    for key, index in plan:
        source = a if key == "a" else b
        expected.append(layer_state(source.model.layers[index]))
    old_types = {"a": getattr(a.config, "layer_types", None), "b": getattr(b.config, "layer_types", None)}
    embed = a.model.embed_tokens.weight.detach().clone()
    model = splice({"a": a, "b": b}, plan, "a")
    assert model is a
    assert model.config.num_hidden_layers == 6
    for index in range(6):
        state = model.model.layers[index].state_dict()
        for name in expected[index]:
            assert torch.equal(state[name], expected[index][name])
        assert model.model.layers[index].self_attn.layer_idx == index
    assert torch.equal(model.model.embed_tokens.weight, embed)
    # The repeated layer is a copy, not a shared module.
    assert model.model.layers[4] is not model.model.layers[5]
    types = []
    for key, index in plan:
        types.append(old_types[key][index])
    assert model.config.layer_types == types
    reloaded = save_and_reload(model, tmp_path / "spliced")
    assert torch.allclose(logits(reloaded), logits(model), atol=1e-6)


def test_splice_refuses_mismatched_shapes():
    a = tiny("llama")
    wide = tiny("llama", intermediate=64)
    with pytest.raises(ValueError, match="does not match the base"):
        splice({"a": a, "w": wide}, [("a", 0), ("w", 1)], "a")
    other = tiny("qwen2")
    with pytest.raises(ValueError, match="one architecture"):
        splice({"a": a, "q": other}, [("a", 0), ("q", 1)], "a")
    with pytest.raises(ValueError, match="not a source"):
        splice({"a": a}, [("a", 0), ("z", 1)], "a")
    with pytest.raises(ValueError, match="layer 9"):
        splice({"a": a}, [("a", 9)], "a")
    # A failed check leaves the base whole.
    assert a.config.num_hidden_layers == 6
    assert len(a.model.layers) == 6


def save_source(model, path):
    model.save_pretrained(str(path))
    tiny_tokenizer().save_pretrained(str(path))
    return str(path)


def test_compile_carve_program_writes_a_plain_model(tmp_path):
    source = save_source(tiny("llama"), tmp_path / "source")
    full = count_params(tiny("llama"))
    target_b = full * 0.7 / 1e9
    program = (Program(base_source=source)
               .carve(target_params_b=target_b, select="influence", width={"keep_mlp": 0.75, "keep_heads": 0.5})
               .lora(rank=2)
               .heal(steps=3, lr=1e-2, batch_size=2, seq_len=16))
    out = str(tmp_path / "out")
    program.compile(out, None, heal_texts=TEXTS, device="cpu")

    report = json.load(open(os.path.join(out, "compile_report.json")))
    assert report["written"] is True
    assert report["params_before"] == full
    assert report["params_after"] <= full * 0.7
    assert report["layers_before"] == 6
    carve = report["carve"]
    assert len(carve["influence"]) == 6
    assert carve["keep_layers"][0] == 0 and carve["keep_layers"][-1] == 5
    assert report["layers_after"] == len(carve["keep_layers"])
    assert carve["width"]["mlp_after"] == 36
    assert carve["width"]["scores"] == "activations"
    assert report["lora"]["merged"] is True
    assert len(report["heal"]["losses"]) == 3
    assert "load" in report["seconds"] and "total" in report["seconds"]
    for name in ["grafts.json", "taps.json", "lora.json", "host"]:
        assert not os.path.exists(os.path.join(out, name))

    model = transformers.AutoModelForCausalLM.from_pretrained(out)
    assert type(model).__name__ == "LlamaForCausalLM"
    assert count_params(model) == report["params_after"]
    assert model.config.num_hidden_layers == report["layers_after"]
    loaded, _ = load(out, dtype=torch.float32)
    assert torch.allclose(logits(loaded), logits(model.float()), atol=1e-5)
    saved = json.load(open(os.path.join(out, "program.json")))
    assert program_from_dict(saved).to_dict() == program.to_dict()


def test_compile_splice_program_writes_a_plain_model(tmp_path):
    a = save_source(tiny("qwen2", seed=0), tmp_path / "a")
    b = save_source(tiny("qwen2", seed=1), tmp_path / "b")
    b_layer = layer_state(tiny("qwen2", seed=1).model.layers[4])
    program = Program(host=a).splice({"a": a, "b": b}, "a", [["a", 0], ["a", 1], ["b", 4], ["a", 5]])
    program.carve(keep=3, select="uniform")
    program.lora(rank=2).heal(steps=2, lr=1e-2, batch_size=2, seq_len=16)
    out = str(tmp_path / "out")
    program.compile(out, None, heal_texts=TEXTS, device="cpu")

    report = json.load(open(os.path.join(out, "compile_report.json")))
    assert report["splice"]["layers"] == 4
    assert report["carve"]["keep_layers"] == [0, 2, 3]
    assert report["layers_after"] == 3
    model = transformers.AutoModelForCausalLM.from_pretrained(out)
    assert type(model).__name__ == "Qwen2ForCausalLM"
    assert model.config.num_hidden_layers == 3
    # The middle layer is b's layer 4. The LoRA leaves the norms as they were.
    norm = model.model.layers[1].input_layernorm.weight.float()
    assert torch.allclose(norm, b_layer["input_layernorm.weight"].to(torch.bfloat16).float())
    mask = torch.ones_like(IDS)
    model.generate(IDS, attention_mask=mask, max_new_tokens=3, do_sample=False, pad_token_id=0)


def test_compile_splice_refuses_mismatched_sources(tmp_path):
    a = save_source(tiny("llama"), tmp_path / "a")
    wide = save_source(tiny("llama", intermediate=64), tmp_path / "wide")
    program = Program(host=a).splice({"a": a, "w": wide}, "a", [["a", 0], ["w", 1]])
    with pytest.raises(CompileError, match="does not match the base"):
        program.compile(str(tmp_path / "out"), None, device="cpu")


def test_written_program_checks():
    with pytest.raises(CompileError, match="grafts or taps"):
        Program(base_source="m").graft("g").compile("unused", None)
    with pytest.raises(CompileError, match="add .lora"):
        Program(base_source="m").carve(keep=2).heal().compile("unused", None)
    with pytest.raises(ValueError, match="not both"):
        Program(base_source="m").carve(keep=4, target_params_b=1.0)
    with pytest.raises(ValueError, match="needs keep"):
        Program(base_source="m").carve()
    with pytest.raises(ValueError, match="select"):
        Program(base_source="m").carve(keep=4, select="best")
    with pytest.raises(ValueError, match="keep_mlp"):
        Program(base_source="m").carve(width={"keep_mlp": 1.5})
    with pytest.raises(ValueError, match="splice base"):
        Program(host="x").splice({"a": "y"}, "a", [["a", 0]])
    with pytest.raises(ValueError, match="not the program's source"):
        Program(host="x", base_source="y")


def test_old_programs_keep_their_json():
    program = Program(host="h").lora(rank=4).heal()
    data = program.to_dict()
    for name in ["base_source", "carve", "splice"]:
        assert name not in data
    assert not program.is_written()
    carved = Program(base_source="h").carve(keep=4)
    assert carved.host == "h"
    assert carved.to_dict()["carve"] == {"keep": 4, "target_params_b": None, "select": "influence",
                                         "domain_data": None, "width": None}


def round_trip(program):
    text = format_program(program)
    assert parse(text).to_dict() == program.to_dict()
    assert program_from_dict(json.loads(json.dumps(program.to_dict()))).to_dict() == program.to_dict()
    return text


def test_lang_round_trips_written_programs():
    text = round_trip(Program(base_source="Qwen/Qwen2.5-14B-Instruct").carve(target_params_b=7.6)
                      .lora(rank=64).heal(steps=1000, data="mmlu_aux", lr=1e-4))
    assert text.startswith("from Qwen/Qwen2.5-14B-Instruct\n")
    assert "carve target 7.6b select influence" in text
    round_trip(Program(base_source="m").carve(keep=22, select="uniform"))
    round_trip(Program(host="m").carve(keep=22, select="domain", domain_data="gsm8k_train", domain_weight=0.9,
                                       width={"keep_mlp": 0.75, "keep_heads": 0.5}))
    round_trip(Program(base_source="m").carve(width={"keep_mlp": 0.5, "keep_heads": 1.0}))
    round_trip(Program(base_source="m").carve(target_params_b=8))

    sources = {"q": "Qwen/Qwen2.5-7B-Instruct", "m": "Qwen/Qwen2.5-Math-7B-Instruct",
               "c": "Qwen/Qwen2.5-Coder-7B-Instruct"}
    plan = []
    for layer in range(10):
        plan.append(["q", layer])
    for layer in range(10, 19):
        plan.append(["m", layer])
    for layer in range(19, 28):
        plan.append(["c", layer])
    plan.append(["c", 27])
    program = Program(host=sources["q"]).splice(sources, "q", plan).carve(keep=24)
    text = round_trip(program)
    assert "plan q:0..9, m:10..18, c:19..27, c:27" in text
    assert not text.startswith("host")


def test_lang_reads_the_written_statements():
    program = parse("splice base a sources a=org/a, b=org/b plan a:0..1, b:3, a:2\n"
                    "carve keep 3 select domain domain mmlu_aux width heads 0.5 mlp 0.75\n")
    assert program.host == "org/a"
    assert program.splice_spec.plan == [["a", 0], ["a", 1], ["b", 3], ["a", 2]]
    assert program.carve_spec.width == {"keep_mlp": 0.75, "keep_heads": 0.5}
    assert program.carve_spec.domain_data == "mmlu_aux"
    assert parse("from org/a\ncarve target 0.5\n").carve_spec.target_params_b == 0.5


@pytest.mark.parametrize("text, words", [
    ("from a\ncarve keep 3 target 2b", "both keep and target"),
    ("from a\ncarve keep x", "keep needs an integer"),
    ("from a\ncarve keep 3 select best", "select is one of"),
    ("from a\ncarve keep 3 width", "width needs"),
    ("from a\ncarve keep 3 colour red", "unknown carve option"),
    ("from a\ncarve target 0b", "above 0"),
    ("splice base a sources a=x plan a:0..1\nsplice base a sources a=x plan a:0", "appears twice"),
    ("splice base a sources a=x", "splice needs plan"),
    ("splice base a sources a=x plan z:0", "not a source key"),
    ("splice base a sources ax plan a:0", "KEY=MODEL"),
    ("splice base a sources a=x plan a", "KEY:A..B"),
    ("host h\nfrom a", "not the program's source"),
])
def test_lang_errors_for_written_statements(text, words):
    with pytest.raises(LangError, match=words):
        parse(text)


def test_cli_compile_needs_no_catalog_for_a_written_program(tmp_path, monkeypatch, capsys):
    import wsp.compiler
    from wsp.lang import main

    calls = []

    def fake_compile(program, catalog, output_dir, align_texts=None, heal_texts=None, device=None):
        calls.append((program.to_dict(), catalog))
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, "compile_report.json"), "w") as f:
            json.dump({"host": program.host}, f)
        return output_dir

    monkeypatch.setattr(wsp.compiler, "compile_program", fake_compile)
    path = tmp_path / "p.wsp"
    path.write_text("from org/a\ncarve keep 3 select uniform\n")
    assert main(["compile", str(path), str(tmp_path / "out"), "--device", "cpu"]) == 0
    assert calls[0][1] is None
    assert calls[0][0]["carve"]["keep"] == 3
    other = tmp_path / "q.wsp"
    other.write_text("host org/a\nlora rank 2\nheal\n")
    assert main(["compile", str(other), str(tmp_path / "out2")]) == 1
    assert "needs --catalog" in capsys.readouterr().err


@pytest.mark.parametrize("field, value", [("rms_norm_eps", 0.5), ("hidden_act", "gelu"),
                                          ("num_attention_heads", 4)])
def test_splice_refuses_sources_whose_layers_run_with_other_config_values(field, value):
    # Same parameter shapes, but the layer keeps its own value in memory and
    # the saved model rebuilds it with the base value, so the two differ.
    settings = {"hidden_size": 32, "intermediate_size": 48, "num_hidden_layers": 4, "num_attention_heads": 8,
                "num_key_value_heads": 4, "vocab_size": 40}
    base = transformers.LlamaForCausalLM(transformers.LlamaConfig(**settings)).eval()
    settings[field] = value
    other = transformers.LlamaForCausalLM(transformers.LlamaConfig(**settings)).eval()
    before = layer_state(base.model.layers[1])
    with pytest.raises(ValueError, match=field):
        splice({"a": base, "b": other}, [("a", 0), ("b", 1)], "a")
    # The check runs before any change.
    assert len(base.model.layers) == 4
    for name, value in layer_state(base.model.layers[1]).items():
        assert torch.equal(value, before[name])


def test_domain_weight_is_only_for_select_domain():
    with pytest.raises(ValueError, match="only for select domain"):
        Program(base_source="m").carve(keep=4, domain_weight=0.5)
    with pytest.raises(LangError, match="only for select domain"):
        parse("from m\ncarve keep 4 select influence weight 0.5\n")
    # The default weight is fine with any rule.
    assert parse("from m\ncarve keep 4 select uniform weight 0.75\n").carve_spec.select == "uniform"


def test_compile_checks_the_target_before_the_heal(tmp_path, monkeypatch):
    import wsp.carve
    import wsp.compiler

    source = save_source(tiny("llama"), tmp_path / "source")
    full = count_params(tiny("llama"))
    # A wrong layer count stands in for a budget the carve misses.
    monkeypatch.setattr(wsp.carve, "layers_for_budget", lambda *args: 6)
    heals = []
    monkeypatch.setattr(wsp.compiler, "run_heal", lambda *args: heals.append(1))
    program = (Program(base_source=source).carve(target_params_b=full * 0.5 / 1e9, select="uniform")
               .lora(rank=2).heal(steps=2, batch_size=2, seq_len=16))
    with pytest.raises(CompileError, match="over the target"):
        program.compile(str(tmp_path / "out"), None, heal_texts=TEXTS, device="cpu")
    assert heals == []
    assert not os.path.exists(str(tmp_path / "out"))


def qk_norm_model(kind):
    torch.manual_seed(0)
    if kind == "olmo2":
        config = transformers.Olmo2Config(hidden_size=32, intermediate_size=48, num_hidden_layers=3,
                                          num_attention_heads=4, num_key_value_heads=2, vocab_size=40)
        return transformers.Olmo2ForCausalLM(config).eval()
    if kind == "cohere":
        config = transformers.CohereConfig(hidden_size=32, intermediate_size=48, num_hidden_layers=3,
                                           num_attention_heads=4, num_key_value_heads=2, vocab_size=40,
                                           use_qk_norm=True)
        return transformers.CohereForCausalLM(config).eval()
    config = transformers.Qwen3Config(hidden_size=32, intermediate_size=48, num_hidden_layers=3,
                                      num_attention_heads=4, num_key_value_heads=2, head_dim=8, vocab_size=40)
    return transformers.Qwen3ForCausalLM(config).eval()


@pytest.mark.parametrize("kind", ["olmo2", "cohere"])
def test_carve_width_refuses_heads_with_norms_over_all_heads(kind, tmp_path):
    # OLMo 2 normalizes all of q at once, and Cohere keeps one norm row per
    # head. A head cut cannot slice those norms, so it must fail before it
    # changes the model, not in forward.
    model = qk_norm_model(kind)
    before = logits(model)
    with pytest.raises(ValueError, match="head count"):
        carve_width(model, 1.0, 0.5)
    assert torch.equal(logits(model), before)
    # An MLP cut alone still works and reloads.
    carve_width(model, 0.5, 1.0)
    reloaded = save_and_reload(model, tmp_path)
    assert torch.allclose(logits(reloaded), logits(model), atol=1e-5)


def test_carve_width_cuts_heads_with_a_per_head_norm(tmp_path):
    # Qwen3's q_norm and k_norm cover one head, so they stay as they are.
    model = qk_norm_model("qwen3")
    carve_width(model, 1.0, 0.5)
    assert model.config.num_attention_heads == 2
    reloaded = save_and_reload(model, tmp_path)
    assert torch.allclose(logits(reloaded), logits(model), atol=1e-5)
