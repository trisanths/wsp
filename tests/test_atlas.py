import copy
import json
import os

import pytest
import torch
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast
from transformers import Qwen3MoeConfig, Qwen3MoeForCausalLM

from wsp import atlas

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SKILL_WORDS = ["add", "sum", "plus", "minus", "times", "equals", "number", "digit"]
GENERAL_WORDS = ["the", "river", "city", "king", "song", "film", "tree", "road"]
EXTRA = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "Question:", "Answer:", "yes", "no", "."]


def tokenizer():
    vocab = {"<unk>": 0, "</s>": 1, "<pad>": 2}
    for word in SKILL_WORDS + GENERAL_WORDS + EXTRA:
        vocab[word] = len(vocab)
    base = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    base.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    return PreTrainedTokenizerFast(tokenizer_object=base, unk_token="<unk>", eos_token="</s>", pad_token="<pad>")


def word_ids(tok, words):
    ids = []
    for word in words:
        ids.append(tok.convert_tokens_to_ids(word))
    return ids


def texts(words, count, length, seed):
    generator = torch.Generator()
    generator.manual_seed(seed)
    out = []
    for _ in range(count):
        picks = torch.randint(len(words), (length,), generator=generator).tolist()
        line = []
        for pick in picks:
            line.append(words[pick])
        out.append(" ".join(line))
    return out


def tiny_llama(layers=4):
    torch.manual_seed(0)
    return LlamaForCausalLM(LlamaConfig(
        hidden_size=32, intermediate_size=64, num_hidden_layers=layers, num_attention_heads=4,
        num_key_value_heads=2, vocab_size=40, max_position_embeddings=64,
    )).eval()


def tiny_moe():
    torch.manual_seed(0)
    return Qwen3MoeForCausalLM(Qwen3MoeConfig(
        hidden_size=48, intermediate_size=96, moe_intermediate_size=40, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, vocab_size=40, max_position_embeddings=64,
        num_experts=6, num_experts_per_tok=2, decoder_sparse_step=1, mlp_only_layers=[], head_dim=12,
    )).eval()


def test_skip_equals_layer_removed():
    tok = tokenizer()
    model = tiny_llama()
    sample = texts(SKILL_WORDS + GENERAL_WORDS, 5, 12, seed=1)
    for layer in range(4):
        skipped = atlas.text_loss(model, tok, sample, skip_layer=layer)
        removed = copy.deepcopy(model)
        del removed.model.layers[layer]
        removed.config.num_hidden_layers = 3
        direct = atlas.text_loss(removed, tok, sample)
        assert abs(skipped - direct) < 1e-5


def test_padding_does_not_change_loss():
    tok = tokenizer()
    model = tiny_llama()
    sample = texts(SKILL_WORDS, 1, 10, seed=2) + texts(SKILL_WORDS, 1, 4, seed=3)
    together = atlas.text_loss(model, tok, sample, batch_size=2)
    encoded = atlas.encode_texts(tok, sample)
    total = 0.0
    tokens = 0
    for ids in encoded:
        stats = atlas.pass_stats(model, [ids], batch_size=1)
        total += stats["loss"] * stats["tokens"]
        tokens += stats["tokens"]
    assert abs(together - total / tokens) < 1e-5


def test_layer_skill_map_shapes_and_math():
    tok = tokenizer()
    model = tiny_llama()
    # Make layer 2 an exact identity: zero attention output and MLP output.
    with torch.no_grad():
        model.model.layers[2].self_attn.o_proj.weight.zero_()
        model.model.layers[2].mlp.down_proj.weight.zero_()
    skill = texts(SKILL_WORDS, 6, 12, seed=4)
    general = texts(GENERAL_WORDS, 6, 12, seed=5)
    result = atlas.layer_skill_map(model, tok, skill, general, batch_size=4)
    assert result["layers"] == 4
    for key in ["scores", "influence", "skill_increase", "general_increase", "skill_influence", "general_influence"]:
        assert len(result[key]) == 4
    base = atlas.text_loss(model, tok, skill)
    for layer in range(4):
        increase = atlas.text_loss(model, tok, skill, skip_layer=layer) - base
        assert abs(result["skill_increase"][layer] - increase) < 1e-5
        expected = result["skill_increase"][layer] - result["general_increase"][layer]
        assert abs(result["scores"][layer] - expected) < 1e-9
    # The identity layer changes nothing, so it scores zero on both measures.
    assert abs(result["scores"][2]) < 1e-5
    assert abs(result["skill_influence"][2]) < 1e-5
    assert len(result["blocks"]) >= 1
    for first, last in result["blocks"]:
        assert 0 <= first <= last < 4
    for share in result["block_mass"]:
        assert 0 < share <= 1


def test_suggest_blocks():
    scores = [0.0, 0.1, 0.5, 0.4, -0.2, 0.0]
    blocks, shares = atlas.suggest_blocks(scores, mass=0.6, count=1)
    assert blocks == [[2, 3]]
    assert shares == [0.9]
    blocks, _ = atlas.suggest_blocks(scores, target_length=1, count=2)
    assert blocks == [[2, 2], [3, 3]]
    blocks, _ = atlas.suggest_blocks(scores, target_length=3, count=3)
    assert blocks[0] == [1, 3]
    for first, last in blocks:
        assert last - first == 2
    assert atlas.suggest_blocks([-1.0, 0.0], count=2) == ([], [])
    blocks, _ = atlas.suggest_blocks(scores, target_length=99, count=2)
    assert blocks == [[0, 5]]


def test_expert_skill_map_finds_scaled_router_row():
    tok = tokenizer()
    model = tiny_moe()
    skill_ids = word_ids(tok, SKILL_WORDS)
    general_ids = word_ids(tok, GENERAL_WORDS)
    direction = torch.randn(48)
    direction = direction / direction.norm()
    with torch.no_grad():
        # Skill tokens point along +direction, general tokens along -direction.
        for token in skill_ids:
            model.model.embed_tokens.weight[token] = 4.0 * direction + 0.01 * torch.randn(48)
        for token in general_ids:
            model.model.embed_tokens.weight[token] = -4.0 * direction + 0.01 * torch.randn(48)
        # Expert 3's router row is the direction, scaled: high logits for skill tokens only.
        model.model.layers[0].mlp.gate.weight[3] = 20.0 * direction
    skill = texts(SKILL_WORDS, 4, 10, seed=6)
    general = texts(GENERAL_WORDS, 4, 10, seed=7)
    result = atlas.expert_skill_map(model, tok, skill, general, top=2)
    assert result["top_k"] == 2
    assert result["num_experts"] == 6
    assert sorted(result["layers"]) == [0, 1]
    layer0 = result["layers"][0]
    assert layer0["top"][0] == 3
    assert layer0["skill_freq"][3] > 0.95
    assert layer0["general_freq"][3] < 0.05
    total = 0.0
    for value in layer0["skill_freq"]:
        total += value
    assert abs(total - 2.0) < 1e-6
    compact = atlas.top_experts(result)
    assert compact["0"][0] == 3


def test_expert_map_needs_moe():
    with pytest.raises(ValueError):
        atlas.expert_skill_map(tiny_llama(), tokenizer(), ["add sum"], ["the river"])


def sample_entries():
    a = atlas.make_entry("org/a-7b", {"math": 0.6, "law": 0.3},
                         {"math": {"scores": [0.0, 0.2, 0.3, 0.0], "blocks": [[1, 2]], "block_mass": [1.0]}},
                         None, params_b=7.0, layers=4, hidden=64, family="llama")
    b = atlas.make_entry("org/b-1b", {"math": 0.6},
                         {"math": {"scores": [0.1, 0.0, 0.0, 0.5, 0.4, 0.0], "blocks": [[3, 4], [0, 0]],
                                   "block_mass": [0.9, 0.1]}},
                         {"math": {"0": [1, 2], "1": []}}, params_b=1.0, layers=6, hidden=32, family="qwen2")
    c = atlas.make_entry("org/c-14b", {"math": 0.8}, {}, None, params_b=14.0, layers=8, hidden=64, family="llama")
    return [a, b, c]


def test_entry_round_trip(tmp_path):
    entries = sample_entries()
    for entry in entries:
        path = atlas.write_entry(entry, str(tmp_path))
        assert os.path.basename(path) == entry["model"].replace("/", "__") + ".json"
        assert atlas.read_entry(path) == json.loads(json.dumps(entry))
    loaded = atlas.load_entries(str(tmp_path))
    assert len(loaded) == 3
    for key in ["model", "skills", "layer_maps", "expert_maps", "params_b", "layers", "hidden", "measured_at"]:
        assert key in loaded[0]


def test_make_entry_keeps_compact_maps():
    full = {"scores": [0.123456789, 0.2], "influence": [0.1, 0.0], "blocks": [[0, 1]], "block_mass": [1.0],
            "skill_increase": [1.0, 2.0]}
    expert_full = {"layers": {0: {"top": [5], "skill_freq": [], "general_freq": [], "log_ratio": []}},
                   "top_k": 2, "num_experts": 8}
    entry = atlas.make_entry("m", {"math": 0.5}, {"math": full}, {"math": expert_full})
    assert entry["layer_maps"]["math"] == {"scores": [0.12346, 0.2], "blocks": [[0, 1]], "influence": [0.1, 0.0],
                                           "block_mass": [1.0]}
    assert entry["expert_maps"] == {"math": {"0": [5]}}


def test_best_models_for_ordering(tmp_path):
    entries = sample_entries()
    rows = atlas.best_models_for("math", entries)
    names = []
    for row in rows:
        names.append(row["model"])
    # c is best; a and b tie at 0.6 and the smaller b comes first.
    assert names == ["org/c-14b", "org/b-1b", "org/a-7b"]
    assert atlas.best_models_for("law", entries)[0]["model"] == "org/a-7b"
    assert len(atlas.best_models_for("law", entries)) == 1
    assert atlas.best_models_for("math", entries, max_params_b=7.0)[0]["model"] == "org/b-1b"
    for entry in entries:
        atlas.write_entry(entry, str(tmp_path))
    assert atlas.best_models_for("math", str(tmp_path))[0]["model"] == "org/c-14b"


def test_best_block_for_constraints():
    entries = sample_entries()
    best = atlas.best_block_for("math", entries)
    # c has no layer map; a and b tie on skill score; b's block has more mass.
    assert best["model"] == "org/b-1b"
    assert best["block"] == [3, 4]
    best = atlas.best_block_for("math", entries, {"hidden": 64})
    assert best["model"] == "org/a-7b"
    assert best["block"] == [1, 2]
    best = atlas.best_block_for("math", entries, {"family": "qwen2", "max_length": 1})
    assert best["block"] == [3, 3]
    assert best["length"] == 1
    assert atlas.best_block_for("math", entries, {"min_length": 3}) is None
    assert atlas.best_block_for("code", entries) is None


def test_catalog_rows(tmp_path):
    entry = sample_entries()[1]
    rows = atlas.catalog_rows(entry, "math", units_dir=str(tmp_path))
    kinds = []
    for row in rows:
        kinds.append(row["type"])
    assert kinds == ["block", "block", "expert"]
    assert rows[0]["layers"] == [3, 4]
    assert rows[0]["model"] == "org/b-1b"
    units = torch.load(rows[2]["units"])
    assert units == {"kind": "expert", "model": "org/b-1b", "layer": 0, "indices": [1, 2]}


def test_answer_checks():
    assert atlas.number_answer("so 3 + 4 = 7\n#### 7") == "7"
    assert atlas.number_answer("The answer is \\boxed{1,234}.") == "1234"
    assert atlas.number_answer("<think>maybe 5</think> it is 6") == "6"
    assert atlas.same_number("18", "18.0")
    assert not atlas.same_number(None, "1")
    assert atlas.boxed("x \\boxed{\\frac{1}{2}} y") == "\\frac{1}{2}"
    assert atlas.math_correct("so \\boxed{\\dfrac{1}{2}}", "\\frac{1}{2}")
    assert atlas.math_correct("\\boxed{\\left( 3, \\frac{\\pi}{2} \\right)}", "\\left( 3, \\frac{\\pi}{2} \\right)")
    assert not atlas.math_correct("no box here", "3")
    assert atlas.bbh_correct("Step one. So the answer is (B).", "(B)")
    assert not atlas.bbh_correct("So the answer is (C).", "(B)")
    assert atlas.bbh_correct("so the answer is True.", "True")
    assert atlas.bbh_correct("So the answer is valid.", "valid")
    assert atlas.extract_code("Here:\n```python\ndef f():\n    return 1\n```\nDone") == "def f():\n    return 1\n"


def test_code_check_runs_tests():
    item = {"kind": "code", "tests": ["assert f(2) == 4"], "imports": []}
    good = "```python\ndef f(x):\n    return x * 2\n```"
    bad = "```python\ndef f(x):\n    return x + 3\n```"
    loop = "```python\ndef f(x):\n    while True:\n        pass\n```"
    item_loop = {"kind": "code", "tests": ["assert f(2) == 4"], "imports": []}
    assert atlas.check_code([item, item], [good, bad]) == [True, False]
    assert atlas.check_code([item_loop], [loop], timeout=1) == [False]


def test_screen_skill_with_items():
    tok = tokenizer()
    model = tiny_llama()
    items = []
    for index in range(4):
        items.append(atlas.mcq_item(str(index), "toy", "add sum", ["yes", "no"], index % 2))
    result = atlas.screen_skill(model, tok, "knowledge", items=items, batch_size=2)
    assert result["count"] == 4
    assert 0.0 <= result["score"] <= 1.0
    assert list(result["by_subject"]) == ["toy"]
    gen_items = [{"id": "g0", "subject": "toy", "kind": "number", "prompt": "add sum", "answer": "7"}]
    result = atlas.screen_skill(model, tok, "math", items=gen_items, max_new_tokens=3)
    assert result["count"] == 1
    assert result["score"] == 0.0


def test_skill_registry():
    assert sorted(atlas.SKILL_ORDER) == sorted(atlas.SKILLS)
    for name in atlas.SKILLS:
        skill = atlas.SKILLS[name]
        assert skill["kind"] in ("mcq", "generate")
        assert skill["count"] > 0
        assert callable(skill["items"])
        assert callable(skill["texts"])
        assert skill["about"]


def test_zoo_schema():
    zoo = atlas.load_zoo(os.path.join(ROOT, "atlas", "zoo.json"))
    assert zoo["image_transformers"] == "4.57.1"
    seen = set()
    keys = set()
    for model in zoo["models"]:
        for field in ["id", "key", "maker", "family", "params_b", "layers", "hidden", "kind", "moe", "skills",
                      "license", "needs", "expert_level_only", "verified", "notes"]:
            assert field in model, (model["id"], field)
        assert model["id"] not in seen
        seen.add(model["id"])
        assert model["key"] == model["id"].replace("/", "__")
        assert model["key"] not in keys
        keys.add(model["key"])
        assert model["kind"] in ("base", "instruct", "reasoning", "specialist")
        assert model["verified"]["status"] in ("ok", "gated", "unverified")
        assert model["needs"]["transformers"] in ("4.57.1", "5.x")
        assert isinstance(model["needs"]["trust_remote_code"], bool)
        assert isinstance(model["skills"], list) and len(model["skills"]) > 0
        if model["needs"]["multi_gpu"]:
            assert model["needs"]["gpu"] == "H200:8"
            assert model["expert_level_only"]
        if model["moe"]:
            assert model["experts"] and model["experts"] > 1
    for wanted in ["LiquidAI/LFM2-350M", "LiquidAI/LFM2.5-1.2B-Thinking", "LiquidAI/LFM2.5-8B-A1B", "LiquidAI/LFM3-1.7B",
                   "LiquidAI/d1-2B", "Equall/Saul-7B-Instruct-v1", "Qwen/Qwen2.5-Math-7B-Instruct",
                   "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", "Qwen/Qwen2.5-Coder-7B-Instruct",
                   "aaditya/Llama3-OpenBioLLM-8B", "Qwen/Qwen3-30B-A3B", "moonshotai/Kimi-K3",
                   "deepseek-ai/DeepSeek-V4-Pro", "zai-org/GLM-5.3-Flash"]:
        assert wanted in seen, wanted


def test_runnable_rules():
    base = {"needs": {"transformers": "4.57.1", "multi_gpu": False}, "verified": {"status": "ok"}}
    assert atlas.runnable(base) == (True, "")
    newer = {"needs": {"transformers": "5.x", "multi_gpu": False}, "verified": {"status": "ok"}}
    assert atlas.runnable(newer)[0] is False
    giant = {"needs": {"transformers": "4.57.1", "multi_gpu": True}, "verified": {"status": "ok"}}
    assert atlas.runnable(giant)[0] is False
    assert atlas.runnable(giant, giants=True)[0] is True
    gated = {"needs": {"transformers": "4.57.1", "multi_gpu": False}, "verified": {"status": "gated"}}
    assert atlas.runnable(gated)[0] is False
    assert atlas.runnable(gated, gated=True)[0] is True


def fake_rows(path, config, split):
    """Small rows with each dataset's real column names (checked on the Hub 2026-11-14)."""
    rows = []
    for index in range(40):
        if path == "cais/mmlu":
            subjects = ["professional_law", "anatomy", "jurisprudence", "international_law"]
            rows.append({"question": f"q{index}", "choices": ["a", "b", "c", "d"], "answer": index % 4,
                         "subject": subjects[index % 4]})
        elif path == "openlifescienceai/medmcqa":
            rows.append({"question": f"m{index}", "opa": "a", "opb": "b", "opc": "c", "opd": "d", "cop": index % 4,
                         "subject_name": "Anatomy"})
        elif path == "nguha/legalbench":
            label = atlas.LEGALBENCH_TASKS[config][index % len(atlas.LEGALBENCH_TASKS[config])]
            if config == "ucc_v_common_law":
                rows.append({"index": index, "contract": f"c{index}", "answer": label})
            else:
                rows.append({"index": index, "text": f"t{index}", "answer": label})
        elif path == "coastalcph/lex_glue":
            rows.append({"context": f"ctx{index}", "endings": ["h0", "h1", "h2", "h3", "h4"], "label": index % 5})
        elif path == "allenai/ai2_arc":
            rows.append({"id": str(index), "question": f"a{index}",
                         "choices": {"text": ["x", "y", "z"], "label": ["A", "B", "C"]}, "answerKey": "B"})
        elif path == "CohereLabs/Global-MMLU":
            rows.append({"question": f"g{index}", "option_a": "a", "option_b": "b", "option_c": "c", "option_d": "d",
                         "answer": "C", "subject": "anatomy"})
        elif path == "fastino/fast-decisions":
            rows.append({"input": f"ticket {index}", "output": {"classifications": [
                {"task": "urgency", "true_label": ["low"], "labels": ["low", "high"], "multi_label": False},
                {"task": "tags", "true_label": ["a", "b"], "labels": ["a", "b"], "multi_label": True}]}})
        elif path == "openai/gsm8k":
            rows.append({"question": f"How many {index}?", "answer": f"work\n#### {index}"})
        elif path == "HuggingFaceH4/MATH-500":
            rows.append({"problem": f"p{index}", "solution": "s", "answer": "\\frac{1}{2}", "subject": "Algebra"})
        elif path == "google-research-datasets/mbpp":
            rows.append({"task_id": index, "prompt": "Write f.", "code": "def f(x):\n    return x",
                         "test_list": ["assert f(1) == 1"], "test_imports": []})
        elif path == "lukaemon/bbh":
            rows.append({"input": f"b{index}", "target": "(A)"})
        elif path == "juletxara/mgsm":
            rows.append({"question": f"mg{index}", "answer": None, "answer_number": index})
        elif path == "Salesforce/wikitext":
            rows.append({"text": " = Title = "})
            rows.append({"text": "word " * 80})
        else:
            raise KeyError(path)
    return rows


def test_skill_loaders_with_fake_rows(monkeypatch):
    from wsp import mcq

    monkeypatch.setattr(mcq, "load_rows", fake_rows)
    for name in atlas.SKILL_ORDER:
        skill = atlas.SKILLS[name]
        items = skill["items"](20)
        assert len(items) > 0, name
        for item in items:
            assert "id" in item and "prompt" in item and "subject" in item
            if skill["kind"] == "mcq":
                assert 0 <= item["gold"] < len(item["choices"])
            else:
                assert item["kind"] in ("number", "boxed", "code", "bbh")
        texts_out = skill["texts"](8)
        assert len(texts_out) > 0, name
        for text in texts_out:
            assert isinstance(text, str) and len(text) > 0
    assert len(atlas.general_texts(5)) == 5
    decisions = atlas.decision_items(10)
    for item in decisions:
        assert item["choices"] == ["low", "high"]
        assert item["gold"] == 0


# ---------------------------------------------------------------------------
# Regression tests from the break-it review (2026-11-14).


def test_heldout_blocks_come_from_one_half_only():
    tok = tokenizer()
    model = tiny_llama()
    skill = texts(SKILL_WORDS, 6, 12, seed=4)
    general = texts(GENERAL_WORDS, 6, 12, seed=5)
    result = atlas.layer_skill_map(model, tok, skill, general, batch_size=4)
    heldout = result["heldout"]
    assert len(heldout["select_scores"]) == 4
    assert len(heldout["check_scores"]) == 4
    assert len(heldout["check_mass"]) == len(heldout["blocks"])
    for share in heldout["check_mass"]:
        assert 0.0 <= share <= 1.0
    # The select half is the even texts and nothing else.
    even = atlas.layer_skill_map(model, tok, skill[0::2], general[0::2], batch_size=4)
    odd = atlas.layer_skill_map(model, tok, skill[1::2], general[1::2], batch_size=4)
    for layer in range(4):
        assert abs(even["scores"][layer] - heldout["select_scores"][layer]) < 1e-5
        assert abs(odd["scores"][layer] - heldout["check_scores"][layer]) < 1e-5
    assert heldout["blocks"] == atlas.suggest_blocks(even["scores"])[0]
    entry = atlas.make_entry("m", {}, {"math": result})
    assert set(entry["layer_maps"]["math"]["heldout"]) == {"blocks", "select_mass", "check_mass", "split_half_r"}
    # One text a side cannot be split.
    assert atlas.layer_skill_map(model, tok, skill[:1], general[:1])["heldout"] is None


def test_decision_map_texts_skip_screen_rows(monkeypatch):
    from wsp import mcq

    def rows(path, config, split):
        out = []
        for index in range(60):
            out.append({"input": f"{config} {index}", "output": {"classifications": [
                {"task": "urgency", "true_label": ["low"], "labels": ["low", "high"], "multi_label": False},
                {"task": "tone", "true_label": ["calm"], "labels": ["calm", "angry"], "multi_label": False}]}})
        return out

    monkeypatch.setattr(mcq, "load_rows", rows)
    screen_rows = set()
    for item in atlas.decision_items(atlas.SKILLS["decision"]["count"]):
        screen_rows.add(atlas.decision_row(item))
    chosen = atlas.decision_texts(32)
    assert len(chosen) == 32
    for text in chosen:
        task, index = text.split("\n")[0].replace("Question: ", "").split(" ")
        assert f"fast_decisions/{task}/{index}" not in screen_rows


def test_number_answer_is_fair_to_raw_and_markdown_outputs():
    # Markdown headers are not answers; the final #### line is.
    assert atlas.number_answer("#### Step 1: add\n3 + 4 = 7\n#### 7") == "7"
    assert atlas.number_answer("#### 1. Add the apples\n5 + 7 = 12\n#### 12") == "12"
    assert atlas.number_answer("so\n#### **$1,250**") == "1250"
    # A raw continuation goes on to a new problem: the first answer counts.
    assert atlas.number_answer("4 + 1 = 5\n#### 5\n\nQuestion: next one\n#### 9") == "5"
    assert atlas.cut_continuation("5 apples\n#### 5\nQuestion: more\n#### 9") == "5 apples\n#### 5"
    # Thinking that never closed has no answer.
    assert atlas.number_answer("<think>\nmaybe 5, or 6") is None
    assert atlas.strip_thinking("<think>a</think> b") == " b"
    assert not atlas.math_correct("<think>\n\\boxed{3}", "3")


def test_bbh_answers_with_markdown_and_whole_words():
    assert atlas.bbh_correct("So the answer is: **Yes**.", "Yes")
    assert atlas.bbh_correct("So the answer is **(B)**.", "(B)")
    assert atlas.bbh_correct("So the answer is B.", "(B)")
    assert not atlas.bbh_correct("So the answer is Because of x.", "(B)")
    assert not atlas.bbh_correct("So the answer is not clear.", "No")
    assert atlas.bbh_correct("So the answer is no.", "no")
    assert not atlas.bbh_correct("So the answer is invalid.", "valid")


def test_code_screen_needs_the_tests_to_run():
    item = {"kind": "code", "tests": ["assert f(2) == 5"], "imports": []}
    early = "```python\nimport sys\ndef f(x):\n    return 0\nsys.exit(0)\n```"
    hard = "```python\nimport os\ndef f(x):\n    return 0\nos._exit(0)\n```"
    assert atlas.check_code([item, item], [early, hard]) == [False, False]


def test_code_screen_does_not_wait_for_stdin_or_leak_env(monkeypatch):
    import time

    monkeypatch.setenv("HF_TOKEN", "secret")
    start = time.time()
    assert atlas.run_python("x = input()\n", timeout=20) is False
    assert time.time() - start < 10
    assert atlas.run_python("import os\nassert 'HF_TOKEN' not in os.environ\n") is True


def test_code_screen_blocks_deletes(tmp_path):
    keep = tmp_path / "keep.txt"
    keep.write_text("data")
    program = f"import os, shutil\nos.remove({str(keep)!r})\n"
    assert atlas.run_python(program) is False
    program = f"import shutil\nshutil.rmtree({str(tmp_path)!r})\n"
    assert atlas.run_python(program) is False
    assert keep.read_text() == "data"


def test_code_screen_timeout_kills_child_processes(tmp_path):
    import time

    flag = tmp_path / "late.txt"
    child = f"import time; time.sleep(2); open({str(flag)!r}, 'w').write('x')"
    program = ("import subprocess, sys\n"
               f"subprocess.Popen([sys.executable, '-c', {child!r}])\n"
               "while True:\n    pass\n")
    assert atlas.run_python(program, timeout=1, guard=False) is False
    time.sleep(2.5)
    assert not flag.exists()


def test_chat_formats_and_stop_tokens():
    tok = tokenizer()
    assert atlas.ensure_chat_template(tok) is False
    assert atlas.chat_text(tok, "add sum")[1] is False
    assert atlas.ensure_chat_template(tok, "mistral_inst") is True
    text, used = atlas.chat_text(tok, "add sum")
    assert used and "[INST] add sum [/INST]" in text
    model = tiny_llama()
    stops = atlas.stop_token_ids(model, tok)
    assert tok.eos_token_id in stops
    zoo = atlas.load_zoo(os.path.join(ROOT, "atlas", "zoo.json"))
    for zoo_model in zoo["models"]:
        if "chat_format" in zoo_model:
            assert zoo_model["chat_format"] in atlas.CHAT_FORMATS


def test_end_of_turn_token_is_a_stop():
    vocab = {"<unk>": 0, "</s>": 1, "<pad>": 2, "<|eot_id|>": 3, "hi": 4}
    base = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    base.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tok = PreTrainedTokenizerFast(tokenizer_object=base, unk_token="<unk>", eos_token="</s>", pad_token="<pad>")
    model = tiny_llama()
    model.generation_config.eos_token_id = 1
    assert atlas.stop_token_ids(model, tok) == [1, 3]


def test_generate_is_greedy_and_marks_open_thinking():
    tok = tokenizer()
    model = tiny_llama()
    # Sampling settings from a checkpoint's generation_config must not apply.
    model.generation_config.do_sample = True
    model.generation_config.temperature = 0.7
    first = atlas.generate(model, tok, ["add sum", "the river"], max_new_tokens=4)
    second = atlas.generate(model, tok, ["add sum", "the river"], max_new_tokens=4)
    assert first == second
    tok.chat_template = "{% for m in messages %}{{ m['content'] }}{% endfor %} <think>"
    out = atlas.generate(model, tok, ["add sum"], max_new_tokens=3)
    assert out[0].startswith("<think>")
    assert atlas.strip_thinking(out[0]) == ""


def test_find_routers_skips_unknown_layouts():
    class Moe(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.gate = torch.nn.Linear(4, 3)
            self.experts = torch.nn.Linear(4, 4)

    class Layer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.mlp = Moe()

    class Fake(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = torch.nn.ModuleList([Layer()])
            self.config = type("Config", (), {})()

    assert atlas.find_routers(Fake()) == []


def test_measure_model_reports_progress(monkeypatch):
    tok = tokenizer()
    model = tiny_llama()
    calls = []

    def fake_screen(model, tokenizer, name, **kwargs):
        return {"score": 0.5, "count": 1, "by_subject": {}}

    monkeypatch.setattr(atlas, "screen_skill", fake_screen)
    entry = atlas.measure_model(model, tok, "org/tiny", skills=["math", "law"], layer_maps=False,
                                expert_maps=False, on_progress=calls.append, log=lambda text: None)
    assert len(calls) == 2
    assert list(calls[0]["skills"]) == ["math"]
    assert entry["skills"] == {"math": 0.5, "law": 0.5}


def test_modal_shards_cover_each_model_once():
    pytest.importorskip("modal")
    import sys

    sys.path.insert(0, ROOT)
    import modal_atlas

    zoo = atlas.load_zoo(os.path.join(ROOT, "atlas", "zoo.json"))
    normal, big, skipped = modal_atlas.select(zoo, giants=False, gated=False, only="")
    assert len(normal) + len(big) + len(skipped) == len(zoo["models"])
    assert big == []
    for nshards in [1, 3, 6]:
        seen = []
        for shard in range(nshards):
            for zoo_model in modal_atlas.shard_cells(normal + big, shard, nshards):
                seen.append(zoo_model["id"])
        assert sorted(seen) == sorted(model["id"] for model in normal)
    with pytest.raises(ValueError):
        modal_atlas.shard_cells(normal, 3, 3)
    only, _, _ = modal_atlas.select(zoo, False, False, "Qwen/Qwen2.5-0.5B-Instruct,LiquidAI__LFM2-350M")
    assert sorted(model["id"] for model in only) == ["LiquidAI/LFM2-350M", "Qwen/Qwen2.5-0.5B-Instruct"]


def test_modal_tf5_selects_only_models_that_need_5x():
    pytest.importorskip("modal")
    import sys

    sys.path.insert(0, ROOT)
    import modal_atlas

    zoo = atlas.load_zoo(os.path.join(ROOT, "atlas", "zoo.json"))
    normal, big, skipped = modal_atlas.select(zoo, giants=False, gated=False, only="", tf5=True)
    assert big == []
    assert len(normal) > 0
    for zoo_model in normal:
        assert zoo_model["needs"]["transformers"] == "5.x"
        assert not zoo_model["needs"]["multi_gpu"]
    for row in skipped:
        assert "giant" in row["reason"] or "gated" in row["reason"] or "verified" in row["reason"]
    needs_5 = [model for model in zoo["models"] if model["needs"]["transformers"] == "5.x"]
    assert len(normal) + len(skipped) == len(needs_5)
    # The giants stay out even when asked for.
    normal, big, _ = modal_atlas.select(zoo, giants=True, gated=False, only="", tf5=True)
    assert big == []
    # The default run does not take the 5.x models.
    default, _, _ = modal_atlas.select(zoo, giants=False, gated=False, only="")
    for zoo_model in default:
        assert zoo_model["needs"]["transformers"] == "4.57.1"


def test_selected_experts_reads_logits_or_index_tuples():
    logits = torch.tensor([[0.1, 0.9, 0.3, 0.2], [0.8, 0.1, 0.0, 0.7]])
    picks = atlas.selected_experts(logits, 2, 4)
    assert picks.tolist() == [[1, 2], [0, 3]]
    # transformers 5.x routers return (logits, scores, indices).
    indices = torch.tensor([[2, 1], [3, 0]])
    scores = torch.tensor([[0.5, 0.5], [0.5, 0.5]])
    assert atlas.selected_experts((logits, scores, indices), 2, 4).tolist() == [[2, 1], [3, 0]]
    # A tuple of float tensors falls back to the logits.
    assert atlas.selected_experts((logits, scores), 2, 4).tolist() == [[1, 2], [0, 3]]
    with pytest.raises(NotImplementedError):
        atlas.selected_experts(torch.zeros(2, 5), 2, 4)
