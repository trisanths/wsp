import importlib.util
import json
import os
import sys

import pytest

from wsp import auto, skills_eval

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_round():
    if "modal_round28" in sys.modules:
        return sys.modules["modal_round28"]
    spec = importlib.util.spec_from_file_location("modal_round28", os.path.join(ROOT, "modal_round28.py"))
    module = importlib.util.module_from_spec(spec)
    sys.modules["modal_round28"] = module
    spec.loader.exec_module(module)
    return module


def test_cell_list_has_fourteen_cells_in_order():
    r = load_round()
    names = list(r.CELLS)
    assert len(names) == 14
    assert names[0] == "full_30b"
    assert names[1] == "full_30b_mcq"
    for request in r.REQUESTS:
        assert f"{request}_sal64" in r.CELLS
        assert f"{request}_sal48" in r.CELLS
        assert f"{request}_sal32_heal500" in r.CELLS
        assert f"{request}_random48_s0" in r.CELLS
    assert r.CELLS["clinic_sal32_heal500"]["heal"] == 500
    assert r.CELLS["clinic_random48_s0"]["rule"] == "random"
    assert r.CELLS["clinic_random48_s0"]["seed"] == 0
    assert [n for n, c in r.CELLS.items() if c["keep_model"]] == ["clinic_sal64", "accounting_sal64",
                                                                 "law_firm_sal64"]


def test_every_benchmark_is_a_known_skill_and_has_a_row_entry():
    r = load_round()
    known = set(skills_eval.SKILLS) | set(skills_eval.EXTRA_SKILLS)
    for request, entry in r.REQUESTS.items():
        for bench in entry["bench"]:
            assert bench in known
            assert bench in r.ALL_BENCH
        for bench in r.DOMAIN[request]:
            assert bench in entry["bench"]
        assert r.GENERAL[request] in entry["bench"]
        assert r.GENERAL[request] == "mmlu_general_ex_" + entry["group"]
    assert set(r.ALL_BENCH) <= known


def test_the_full_model_scores_everything_and_a_cell_scores_its_request():
    r = load_round()
    assert r.benchmarks_of(r.CELLS["full_30b"]) == r.ALL_BENCH
    assert r.benchmarks_of(r.CELLS["full_30b_mcq"]) == [b for b in r.ALL_BENCH if b not in ("gsm8k", "math500")]
    assert r.benchmarks_of(r.CELLS["law_firm_sal64"]) == ["mmlu_law", "mmlu_general_ex_law"]
    assert "gsm8k" in r.benchmarks_of(r.CELLS["accounting_sal48"])


def test_select_cells_fails_loudly():
    r = load_round()
    with pytest.raises(ValueError):
        r.select_cells(only="nope")
    with pytest.raises(ValueError):
        r.select_cells(requests="cooking")
    with pytest.raises(ValueError):
        r.select_cells(only="clinic_sal64", requests="law_firm")
    assert r.select_cells(smoke=True) == r.SMOKE_CELLS
    assert r.select_cells(requests="law_firm") == ["full_30b", "full_30b_mcq", "law_firm_sal64", "law_firm_sal48",
                                                   "law_firm_sal32_heal500", "law_firm_random48_s0"]


def test_chunks_group_cells_by_request():
    r = load_round()
    chunks = r.make_chunks(r.select_cells())
    assert len(chunks) == 4
    assert chunks[0] == ["full_30b", "full_30b_mcq"]
    assert sum(len(c) for c in chunks) == 14
    assert r.make_chunks(r.SMOKE_CELLS) == [r.SMOKE_CELLS]


def test_heal_pool_exactly_covers_the_heal_steps():
    r = load_round()
    assert r.HEAL_POOL + r.HEAL_BENCH == r.HEAL_STEPS * r.HEAL["batch_size"]
    assert r.HEAL["lm_weight"] == 0.0


def test_overlap_keys_cover_every_scored_question():
    r = load_round()
    items = {"gsm8k": [{"question": "What is 2+2 and more words here?"}],
             "math500": [{"problem": "Find x such that x+1=2, please."}],
             "mmlu_law": [{"prompt": "Question: Is a tort a wrong?\nA. yes\nB. no\nAnswer:"}]}
    keys = r.overlap_keys(items)
    assert "What is 2+2 and more words here?" in keys
    assert "Find x such that x+1=2, please." in keys
    assert "Is a tort a wrong?" in keys


def test_request_texts_drop_overlap_and_slice_without_sharing(monkeypatch):
    r = load_round()
    pool = [(f"pool question number {i} with enough words to count", f"pool answer {i}") for i in range(60)]
    train = [(f"train question number {i} with enough words to count", f"train answer {i}") for i in range(60)]
    monkeypatch.setattr(r, "pool_pairs", lambda skill: pool)
    monkeypatch.setattr(r, "bench_pairs", lambda skill: train)

    class Tokenizer:
        chat_template = None

    keys = ["pool question number 3 with enough words to count",
            "train question number 40 with enough words to count"]
    rank, heal, info = r.request_texts("law_firm", Tokenizer(), keys, 10, 10, 20, 20)
    assert len(rank) == 20 and len(heal) == 40
    assert not (set(rank) & set(heal))
    assert not any("question number 3 " in t and "pool" in t for t in rank + heal)
    assert not any("train question number 40 " in t for t in rank + heal)
    assert info["pool"]["dropped_overlap"] == 1 and info["train"]["dropped_overlap"] == 1
    # The same texts rank the same way on a second call.
    rank2, heal2, info2 = r.request_texts("law_firm", Tokenizer(), keys, 10, 10, 20, 20)
    assert rank == rank2 and info["rank_hash"] == info2["rank_hash"]


def test_request_texts_fail_when_a_source_is_short(monkeypatch):
    r = load_round()
    monkeypatch.setattr(r, "pool_pairs", lambda skill: [("q" * 40, "a")] * 5)
    monkeypatch.setattr(r, "bench_pairs", lambda skill: [("q" * 40, "a")] * 5)

    class Tokenizer:
        chat_template = None

    with pytest.raises(ValueError):
        r.request_texts("clinic", Tokenizer(), [], 10, 10, 20, 20)


def test_request_scores_need_every_benchmark():
    r = load_round()
    scores = {"mmlu_law": 0.6, "mmlu_general_ex_law": 0.7}
    out = r.request_scores(scores)
    assert out == {"law_firm": {"domain": 0.6, "general": 0.7}}
    scores.update({"medmcqa": 0.5, "mmlu_medical": 0.7, "mmlu_general_ex_medical": 0.8})
    assert abs(r.request_scores(scores)["clinic"]["domain"] - 0.6) < 1e-9


def test_demo_requests_validate_and_name_the_request_wording():
    r = load_round()
    folder = os.path.join(ROOT, "demo", "requests")
    names = sorted(os.listdir(folder))
    assert names == ["accounting.json", "clinic.json", "law_firm.json"]
    for name in names:
        request = auto.load_request(os.path.join(folder, name))
        assert request["base"] == r.MODEL_ID
        assert request["remove"] == [{"skill": "*"}]
        key = name[:-5]
        assert request["keep"] == [r.REQUESTS[key]["skill"]]
        with open(os.path.join(folder, name)) as f:
            assert json.load(f)["notes"] == r.REQUESTS[key]["text"]
