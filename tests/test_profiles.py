"""Tests for wsp/profiles.py and the pure parts of modal_profile.py (CPU, no network)."""

import json
import os

import pytest

from wsp import profiles as prof

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def write(path, data):
    with open(path, "w") as f:
        json.dump(data, f)


def record(item_set, subject, correct):
    return {"id": item_set + ":" + subject, "set": item_set, "subject": subject, "gold": 0,
            "pred": 0 if correct else 1, "probs": [0.5, 0.5, 0.0, 0.0], "correct": correct}


def fake_results(folder):
    """A results folder with one small file of each kind the builder reads."""
    qwen_records = [record("mmlu_dev", "anatomy", True), record("mmlu_dev", "law", False),
                    record("mmlu_heldout", "anatomy", True), record("medmcqa", "surgery", True)]
    write(os.path.join(folder, "round10_rows.json"), [
        {"key": "qwen7b", "model": "Qwen/Qwen2.5-7B-Instruct",
         "summary": {"mmlu_dev": [3, 4], "mmlu_heldout": [1, 2], "medmcqa": [1, 1]}, "records": qwen_records},
        # No summary: the scores come from the records.
        {"key": "newmodel", "model": "Acme/New-Model-2B",
         "records": [record("mmlu_dev", "anatomy", True), record("mmlu_dev", "anatomy", False)]},
    ])
    write(os.path.join(folder, "round8_rows.json"), [
        {"kind": "donor_alone", "key": "qwen7b", "seed": 0, "accuracy": 0.1},
        {"kind": "donor_alone", "key": "lfm12b", "seed": 0, "accuracy": 0.262},
        {"kind": "tap_llama1b", "key": "qwen7b", "seed": 0, "accuracy": 0.9},
    ])
    write(os.path.join(folder, "round7_rows.json"), [
        {"part": "heldout", "arm": "host", "seed": 0, "cfg": {}, "accuracy": 0.4465},
        {"part": "heldout", "arm": "tap", "seed": 0, "cfg": {"donor": "Qwen/Qwen2.5-7B-Instruct"}, "accuracy": 0.68},
        {"part": "scaling", "arm": "donor", "seed": 0, "cfg": {"donor": "Qwen/Qwen2.5-3B-Instruct"},
         "accuracy": 0.662},
        {"part": "trivia", "arm": "host", "seed": 0, "cfg": {}, "accuracy": 0.376},
        {"part": "trivia", "arm": "donor", "seed": 0, "cfg": {"donor": "Qwen/Qwen2.5-7B-Instruct"},
         "accuracy": 0.468},
    ])
    write(os.path.join(folder, "baselines_rows.json"), {"summary": [
        {"arm": "ceiling unsloth/Llama-3.2-3B-Instruct", "score_mean": 0.59},
        {"arm": "full_ft unsloth/Llama-3.2-1B-Instruct lr 1e-5", "score_mean": 0.335},
    ], "rows": []})
    write(os.path.join(folder, "round5_rows.json"), {
        "host": "unsloth/Llama-3.2-1B-Instruct", "donor": "Qwen/Qwen2.5-Math-1.5B-Instruct",
        "rows": [{"arm": "host_only", "seed": 0, "accuracy": 0.25},
                 {"arm": "donor_alone", "seed": 0, "accuracy": 0.33},
                 {"arm": "lfm_alone", "seed": 0, "accuracy": 0.01},
                 {"arm": "tap_l14_h10", "seed": 0, "accuracy": 0.9}]})
    write(os.path.join(folder, "round4_rows.json"), {"rows": [
        {"host": "qwen05", "arm": "host", "seed": 0, "accuracy": 0.335},
        {"host": "qwen05", "arm": "lora_r1", "seed": 0, "accuracy": 0.9},
    ]})


def test_build_from_fake_results(tmp_path):
    fake_results(str(tmp_path))
    profiles = prof.build_profiles_from_results(str(tmp_path))
    table = prof.by_key(profiles)

    qwen = table["qwen7b"]
    assert qwen["hf_id"] == "Qwen/Qwen2.5-7B-Instruct"
    assert qwen["layers"] == 28 and qwen["hidden"] == 3584 and qwen["tokenizer_family"] == "qwen2"
    # Round 10 wins over round 8 for mmlu_dev.
    assert qwen["scores"]["mmlu_dev"] == 0.75
    assert "round10_rows.json" in qwen["score_sources"]["mmlu_dev"]
    assert qwen["scores"]["mmlu_heldout"] == 0.5
    # "mmlu" is the dev score: modal_make.py scores the held-out items.
    assert qwen["scores"]["mmlu"] == 0.75
    assert qwen["score_sources"]["mmlu"] == "alias of mmlu_dev"
    assert qwen["scores"]["medmcqa"] == 1.0
    assert qwen["scores"]["triviaqa"] == 0.468
    # The "mmlu" subject table uses the dev records only; held-out is kept apart.
    assert qwen["subject_scores"]["mmlu"]["anatomy"] == {"accuracy": 1.0, "count": 1}
    assert qwen["subject_scores"]["mmlu"]["law"] == {"accuracy": 0.0, "count": 1}
    assert qwen["subject_scores"]["mmlu_heldout"]["anatomy"] == {"accuracy": 1.0, "count": 1}
    assert "law" not in qwen["subject_scores"]["mmlu_heldout"]
    assert qwen["subject_scores"]["medmcqa"]["surgery"]["count"] == 1
    assert "results/round10_rows.json" in qwen["source"]
    assert "results/round7_rows.json" in qwen["source"]

    # Only model-alone rows count, never tap or LoRA rows.
    assert table["lfm25_12b"]["scores"]["mmlu_dev"] == 0.262
    assert table["lfm25_12b"]["scores"]["mmlu"] == 0.262
    assert table["llama1b"]["scores"]["mmlu_heldout"] == 0.4465
    # Without a dev score, "mmlu" falls back to the held-out score.
    assert table["llama1b"]["scores"]["mmlu"] == 0.4465
    assert table["llama1b"]["scores"]["triviaqa"] == 0.376
    assert table["llama1b"]["scores"]["gsm8k"] == 0.25
    assert table["qwen3b"]["scores"]["mmlu_dev"] == 0.662
    assert table["llama3b"]["scores"]["gsm8k"] == 0.59
    assert "raw" in table["llama3b"]["score_sources"]["gsm8k"]
    assert table["qwen_math15b"]["scores"]["gsm8k"] == 0.33
    assert table["lfm2_350m_math"]["scores"]["gsm8k"] == 0.01
    assert table["qwen05"]["scores"]["gsm8k"] == 0.335
    # Raw-prompt gsm8k scores are also kept under gsm8k_raw.
    assert table["qwen05"]["scores"]["gsm8k_raw"] == 0.335
    assert table["llama3b"]["scores"]["gsm8k_raw"] == 0.59

    # A model outside the table gets a profile with unknown config fields.
    new = table["new_model_2b"]
    assert new["hf_id"] == "Acme/New-Model-2B"
    assert new["scores"]["mmlu_dev"] == 0.5
    assert new["params_b"] is None and new["layers"] is None
    # Unknown sizes sort last.
    assert profiles[-1]["key"] == "new_model_2b"


def test_missing_scores_and_files(tmp_path):
    profiles = prof.build_profiles_from_results(str(tmp_path))
    assert len(profiles) == len(prof.KNOWN_MODELS)
    for profile in profiles:
        assert profile["scores"] == {}
        assert prof.get_score(profile, "mmlu") is None
        assert prof.get_score(profile, "gsm8k") is None
        assert profile["params_b"] is not None
        assert profile["active_params_b"] == profile["params_b"]
    assert prof.build_profiles_from_results(str(tmp_path), include_known=False) == []
    # The size order holds.
    sizes = []
    for profile in profiles:
        sizes.append(profile["params_b"])
    assert sizes == sorted(sizes)


def test_get_score_fallbacks():
    profile = {"scores": {"mmlu_dev": 0.4}}
    assert prof.get_score(profile, "mmlu") == 0.4
    profile = {"scores": {"mmlu_dev": 0.4, "mmlu_heldout": 0.45}}
    assert prof.get_score(profile, "mmlu") == 0.4
    profile = {"scores": {"mmlu_heldout": 0.45}}
    assert prof.get_score(profile, "mmlu") == 0.45
    assert prof.get_score({"scores": None}, "mmlu") is None
    assert prof.get_score({}, "medqa") is None
    wrapped = prof.Profile.blank("qwen05")
    wrapped.set_score("medqa", 0.31234567, "test")
    assert wrapped.score("medqa") == 0.3123
    # set_score keeps the first value unless asked to overwrite.
    assert wrapped.set_score("medqa", 0.9, "later") is False
    assert wrapped.score("medqa") == 0.3123
    assert wrapped.set_score("medqa", None, "none", overwrite=True) is False


def test_load_save_round_trip(tmp_path):
    fake_results(str(tmp_path))
    profiles = prof.build_profiles_from_results(str(tmp_path))
    path = str(tmp_path / "profiles.json")
    prof.save_profiles(profiles, path)
    loaded = prof.load_profiles(path)
    assert loaded == profiles
    # Profile objects save the same way.
    wrapped = []
    for profile in profiles:
        wrapped.append(prof.Profile(profile))
    path2 = str(tmp_path / "profiles2.json")
    prof.save_profiles(wrapped, path2)
    assert prof.load_profiles(path2) == profiles


def test_load_fills_missing_keys(tmp_path):
    path = str(tmp_path / "partial.json")
    write(path, [{"key": "x", "hf_id": "Acme/X"}])
    loaded = prof.load_profiles(path)
    assert len(loaded) == 1
    for name in prof.INTERFACE_KEYS:
        assert name in loaded[0]
    assert loaded[0]["scores"] == {}
    assert loaded[0]["decode_tok_s"] is None
    assert prof.load_profiles(str(tmp_path / "missing.json")) == []
    write(path, {"profiles": [{"key": "y", "hf_id": "Acme/Y", "scores": {"mmlu": 0.5}}]})
    assert prof.get_score(prof.load_profiles(path)[0], "mmlu") == 0.5


def test_merge_measurements():
    base = prof.build_profiles_from_results("/nonexistent")
    rows = [
        {"hf_id": "Qwen/Qwen2.5-7B-Instruct", "params_b": 7.616, "active_params_b": 7.616, "layers": 28,
         "hidden": 3584, "family": "qwen2", "tokenizer_family": "qwen2_vocab152064", "gpu": "H100",
         "scores": {"mmlu_heldout": 0.71, "mmlu_pro": 0.45, "gsm8k": 0.88},
         "score_protocols": {"gsm8k": prof.SCORE_PROTOCOLS["gsm8k_chat"]},
         "subject_scores": {"mmlu_pro": {"math": {"accuracy": 0.5, "count": 10}}},
         "decode_tok_s": 41.5, "errors": {}},
        {"hf_id": "Acme/Tiny-MoE", "params_b": 2.0, "active_params_b": 0.5, "layers": 12, "hidden": 1024,
         "family": "acme", "tokenizer_family": "acme_vocab1000", "gpu": "H200",
         "scores": {"medqa": 0.3}, "decode_tok_s": None, "errors": {"gsm8k": "boom"}},
        {"key": "no id"},
    ]
    merged = prof.merge_measurements(base, rows)
    table = prof.by_key(merged)
    qwen = table["qwen7b"]
    assert qwen["scores"]["mmlu"] == 0.71
    assert qwen["score_sources"]["mmlu"] == "alias of mmlu_heldout"
    assert qwen["scores"]["mmlu_pro"] == 0.45
    assert qwen["scores"]["gsm8k"] == 0.88
    assert "chat template" in qwen["score_sources"]["gsm8k"]
    assert "H100" in qwen["score_sources"]["mmlu_pro"]
    assert qwen["decode_tok_s"] == 41.5
    # The hard-coded tokenizer family stays.
    assert qwen["tokenizer_family"] == "qwen2"
    assert qwen["subject_scores"]["mmlu_pro"]["math"]["count"] == 10
    assert qwen["source"].startswith("config: measured by modal_profile.py")
    moe = table["tiny_moe"]
    assert moe["active_params_b"] == 0.5
    assert moe["tokenizer_family"] == "acme_vocab1000"
    assert moe["decode_tok_s"] is None
    assert "gsm8k" not in moe["scores"]
    assert len(merged) == len(prof.KNOWN_MODELS) + 1


def test_gsm8k_number():
    assert prof.gsm8k_number("so 3 + 4 = 7\n#### 7") == 7.0
    assert prof.gsm8k_number("#### 1,234") == 1234.0
    assert prof.gsm8k_number("The answer is $12.50.") == 12.5
    assert prof.gsm8k_number("The answer is: **42**") == 42.0
    assert prof.gsm8k_number("first 3 then 5 then -2") == -2.0
    assert prof.gsm8k_number("no numbers here") is None
    assert prof.gsm8k_number(None) is None
    # The same reading as modal_make.final_number: the first number after
    # the last "the answer is", before "####", and text after a new question
    # is dropped.
    assert prof.gsm8k_number("The answer is \\boxed{18}. Check: 18 * 2 = 36") == 18.0
    assert prof.gsm8k_number("#### 5\nThe answer is 7") == 7.0
    assert prof.gsm8k_number("It is 4.\nQuestion: a new one with 99") == 4.0
    assert prof.gsm8k_gold("Half of 10 is 5, the answer is 5.\n#### 6") == 6.0
    assert prof.gsm8k_gold("no mark 3") is None


def test_gsm8k_number_matches_modal_make():
    pytest.importorskip("modal")
    import sys

    sys.path.insert(0, REPO)
    import modal_make

    texts = ["The answer is \\boxed{18}. Check: 18 * 2 = 36", "#### 1,234", "so 3 + 4 = 7\n#### 7",
             "The answer is: **42**", "It is 4.\nQuestion: next 99", "nothing", "The answer is $12.50."]
    for text in texts:
        assert prof.gsm8k_number(text) == modal_make.final_number(text)


def test_count_params_dense_and_moe():
    torch = pytest.importorskip("torch")

    class Experts(torch.nn.Module):
        def __init__(self):
            super().__init__()
            layers = []
            for index in range(4):
                layers.append(torch.nn.Linear(10, 10, bias=False))
            self.experts = torch.nn.ModuleList(layers)

    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = torch.nn.Linear(10, 100, bias=False)
            self.mlp = Experts()

    class Config:
        num_experts_per_tok = 1
        num_local_experts = 4

    model = Tiny()
    assert prof.count_params(model, scale=1) == (1400, 1400)
    # 1000 dense and 400 expert params, one of four experts active: 1000 + 100.
    assert prof.count_params(model, Config(), scale=1) == (1400, 1100)
    assert prof.count_params(model, Config()) == (0.0, 0.0)


def test_tokenizer_family_guess():
    assert prof.tokenizer_family_guess("llama", 128256) == "llama3"
    assert prof.tokenizer_family_guess("qwen2", 152064) == "qwen2"
    assert prof.tokenizer_family_guess("phi3", 100352) == "phi4"
    assert prof.tokenizer_family_guess("olmo", 50304) == "olmo_vocab50304"
    # Gemma 3 has a different vocabulary from Gemma 2.
    assert prof.tokenizer_family_guess("gemma2", 256000) == "gemma"
    assert prof.tokenizer_family_guess("gemma3_text", 262144) == "gemma3"


def test_merge_keeps_keys_unique():
    rows = [{"hf_id": "a/Foo-1B", "params_b": 1.0, "layers": 2, "hidden": 8},
            {"hf_id": "b/Foo-1B", "params_b": 1.0, "layers": 2, "hidden": 8},
            {"hf_id": "c/Foo-1B", "key": "foo_1b", "params_b": 1.0, "layers": 2, "hidden": 8},
            # A row whose key is a table key but whose id is another model's.
            {"hf_id": "Other/Qwen-Like", "key": "qwen7b", "params_b": 3.0, "layers": 4, "hidden": 16}]
    merged = prof.merge_measurements(prof.build_profiles_from_results("/nonexistent"), rows)
    keys = []
    for profile in merged:
        keys.append(profile["key"])
    assert len(set(keys)) == len(keys)
    assert prof.find_profile(merged, "a/Foo-1B")["key"] == "foo_1b"
    assert prof.find_profile(merged, "b/Foo-1B")["key"] == "foo_1b_b"
    assert prof.find_profile(merged, "c/Foo-1B")["key"] == "foo_1b_c"
    other = prof.find_profile(merged, "Other/Qwen-Like")
    assert other["key"] == "qwen7b_other" and other["params_b"] == 3.0
    # The real Qwen-7B profile keeps its own id and size.
    assert prof.by_key(merged)["qwen7b"]["hf_id"] == "Qwen/Qwen2.5-7B-Instruct"
    assert prof.by_key(merged)["qwen7b"]["params_b"] == 7.616


def test_merge_raw_gsm8k_and_subject_alias():
    rows = [{"hf_id": "Acme/Plain", "params_b": 1.0, "layers": 2, "hidden": 8, "gpu": "H100",
             "scores": {"gsm8k": 0.2, "mmlu_dev": 0.5},
             "score_protocols": {"gsm8k": prof.SCORE_PROTOCOLS["gsm8k_raw"]},
             "subject_scores": {"mmlu_dev": {"law": {"accuracy": 0.5, "count": 2}}}}]
    merged = prof.merge_measurements([], rows)
    plain = merged[0]
    assert plain["scores"]["gsm8k_raw"] == 0.2
    assert plain["scores"]["mmlu"] == 0.5
    assert plain["subject_scores"]["mmlu"]["law"]["count"] == 2


def test_load_keyed_table(tmp_path):
    # wsp.make accepts {key: profile}; load_profiles reads it the same way.
    path = str(tmp_path / "keyed.json")
    write(path, {"x": {"hf_id": "Acme/X", "params_b": 1.0}})
    loaded = prof.load_profiles(path)
    assert len(loaded) == 1
    assert loaded[0]["key"] == "x" and loaded[0]["scores"] == {}


def test_key_for():
    assert prof.key_for("unsloth/gemma-2-9b-it") == "gemma9b"
    assert prof.key_for("Org/Some.Model-7B") == "some_model_7b"
    assert prof.hf_id_for("q14b") == "Qwen/Qwen2.5-14B-Instruct"
    assert prof.hf_id_for("Org/X") == "Org/X"


def test_repo_profiles_file():
    """The committed profiles.json covers the round 10 models and the four extra ones."""
    loaded = prof.load_profiles(os.path.join(REPO, "profiles.json"))
    ids = []
    for profile in loaded:
        ids.append(profile["hf_id"])
    wanted = ["unsloth/Llama-3.2-1B-Instruct", "Qwen/Qwen2.5-0.5B-Instruct", "Qwen/Qwen2.5-7B-Instruct",
              "Qwen/Qwen2.5-14B-Instruct", "Qwen/Qwen2.5-32B-Instruct", "unsloth/gemma-2-9b-it",
              "unsloth/Meta-Llama-3.1-8B-Instruct", "microsoft/Phi-3.5-mini-instruct", "microsoft/phi-4",
              "unsloth/mistral-7b-instruct-v0.3", "BioMistral/BioMistral-7B", "LiquidAI/LFM2-350M-Math",
              "LiquidAI/LFM2.5-1.2B-Instruct", "Qwen/Qwen2.5-Math-1.5B-Instruct", "unsloth/Llama-3.2-3B-Instruct"]
    for hf_id in wanted:
        assert hf_id in ids
    table = prof.by_key(loaded)
    assert table["qwen14b"]["scores"]["mmlu_dev"] == 0.786
    assert table["llama1b"]["scores"]["medmcqa"] == 0.41
    assert table["qwen32b"]["scores"]["mmlu_heldout"] == 0.793
    # The planner's "mmlu" is the dev score, never the held-out items modal_make.py scores.
    for profile in loaded:
        if "mmlu_dev" in profile["scores"]:
            assert profile["score_sources"]["mmlu"] == "alias of mmlu_dev"
            assert profile["scores"]["mmlu"] == profile["scores"]["mmlu_dev"]
    assert table["qwen32b"]["scores"]["mmlu"] == 0.829


def test_modal_profile_sets():
    pytest.importorskip("modal")
    import sys

    sys.path.insert(0, REPO)
    import modal_profile

    assert modal_profile.MCQ_SETS["mmlu_dev"] == ("mmlu", "test", 0, 0, 1000)
    assert modal_profile.MCQ_SETS["mmlu_heldout"] == ("mmlu", "test", 0, 1000, 2000)
    assert modal_profile.MCQ_SETS["medmcqa"] == ("medmcqa", "validation", 7, 0, 1000)
    # MedQA profile items come from train, since modal_make.py scores all of test.
    assert modal_profile.MCQ_SETS["medqa"][1] == "train"

    class ChatTokenizer:
        chat_template = "x"

        def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
            return "<user>" + messages[0]["content"] + "<assistant>"

    class PlainTokenizer:
        chat_template = None

    text, protocol = modal_profile.gsm8k_prompt(ChatTokenizer(), "What is 2+2?")
    assert protocol == "gsm8k_chat" and "step by step" in text and text.startswith("<user>")
    text, protocol = modal_profile.gsm8k_prompt(PlainTokenizer(), "What is 2+2?")
    assert protocol == "gsm8k_raw" and text.startswith("Question: What is 2+2?")

    # The profile measures GSM8K with modal_make.py's prompt and budget.
    import modal_make

    assert modal_profile.gsm8k_prompt(ChatTokenizer(), "Q?")[0] == modal_make.gsm8k_prompt(ChatTokenizer(), "Q?", "chat")
    assert modal_profile.gsm8k_prompt(PlainTokenizer(), "Q?")[0] == modal_make.gsm8k_prompt(PlainTokenizer(), "Q?", "chat")
    assert modal_profile.GSM8K_MAX_NEW_TOKENS == modal_make.GSM8K_MAX_NEW_TOKENS

    # No profile item set overlaps the held-out items modal_make.py scores.
    for name in modal_profile.MCQ_SETS:
        benchmark, split, seed, skip, count = modal_profile.MCQ_SETS[name]
        if name == "mmlu_heldout":
            continue
        held_split, held_seed, held_skip, held_count = modal_make.HELDOUT[benchmark]
        if split != held_split or seed != held_seed:
            continue
        held_end = 10 ** 9
        if held_count is not None:
            held_end = held_skip + held_count
        assert skip + count <= held_skip or skip >= held_end, name
