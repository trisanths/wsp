"""Tests for wsp.mcq: per-item multiple-choice scoring and complementarity analysis."""

import sys
import types

import pytest
import torch
from tokenizers import Regex, Tokenizer, models, pre_tokenizers, processors

transformers = pytest.importorskip("transformers")

from wsp import mcq
from wsp.tap import TapBranch, attach_taps

SUBJECTS = ["anatomy", "virology", "law"]


def fake_mmlu_rows(count=60):
    rows = []
    for index in range(count):
        rows.append({
            "question": f" question {index} ",
            "choices": [f"alpha {index}", f"beta {index}", " gamma ", f"delta {index}"],
            "answer": index % 4,
            "subject": SUBJECTS[index % 3],
        })
    return rows


def fake_datasets(tables):
    """A stand-in datasets module. tables maps (path, config, split) to a list of rows."""
    calls = []

    def load_dataset(path, config=None, split=None):
        calls.append((path, config, split))
        return tables[(path, config, split)]

    module = types.ModuleType("datasets")
    module.load_dataset = load_dataset
    module.calls = calls
    return module


@pytest.fixture
def mmlu_tables(monkeypatch):
    aux = []
    for index in range(10):
        aux.append({"train": {"question": f"aux {index}", "choices": ["w", "x", "y", "z"], "answer": 1,
                              "subject": ""}})
    tables = {("cais/mmlu", "all", "test"): fake_mmlu_rows(), ("cais/mmlu", "auxiliary_train", "train"): aux}
    module = fake_datasets(tables)
    monkeypatch.setitem(sys.modules, "datasets", module)
    return module


# Prompts and sampling

def test_prompt_matches_round6():
    pytest.importorskip("modal")
    import modal_round6

    choices = [" one ", "two", 3, "four"]
    assert mcq.mcq_prompt("  What is it? ", choices) == modal_round6.mmlu_prompt("  What is it? ", choices)
    assert mcq.mcq_prompt("q", ["a", "b"]) == "Question: q\nA. a\nB. b\nAnswer:"


def test_ten_options_reach_letter_j_and_eleven_fail():
    choices = []
    for index in range(10):
        choices.append(str(index))
    prompt = mcq.mcq_prompt("q", choices)
    assert prompt.endswith("\nJ. 9\nAnswer:")
    with pytest.raises(ValueError):
        mcq.mcq_prompt("q", choices + ["10"])


def test_round6_dev_items_are_reproduced(mmlu_tables):
    pytest.importorskip("modal")
    import modal_round6

    _, eval_items = modal_round6.mmlu_data(seed=0, train_count=2, eval_count=25)
    items = mcq.mcq_items("mmlu", "test", seed=0, count=25)
    assert len(items) == 25
    for item, (prompt, answer) in zip(items, eval_items):
        assert item["prompt"] == prompt
        assert item["gold"] == answer


def test_skip_reproduces_round7_heldout_and_stays_disjoint(mmlu_tables):
    import random

    order = list(range(60))
    random.Random(0).shuffle(order)
    heldout = mcq.mcq_items("mmlu", "test", seed=0, count=30, skip=20)
    expected = []
    for index in order[20:50]:
        expected.append(f"mmlu/test/{index}")
    got = []
    for item in heldout:
        got.append(item["id"])
    assert got == expected

    dev = set()
    for item in mcq.mcq_items("mmlu", "test", seed=0, count=20):
        dev.add(item["id"])
    filtered = mcq.mcq_items("mmlu", "test", seed=0, subjects=["anatomy"], skip=20)
    assert len(filtered) > 0
    for item in filtered:
        assert item["subject"] == "anatomy"
        assert item["id"] not in dev


def test_sampling_is_seeded_and_subject_filter_keeps_order(mmlu_tables):
    a = mcq.mcq_items("mmlu", "test", seed=3, count=10)
    b = mcq.mcq_items("mmlu", "test", seed=3, count=10)
    c = mcq.mcq_items("mmlu", "test", seed=4, count=10)
    assert a == b
    assert a != c
    everything = mcq.mcq_items("mmlu", "test", seed=3)
    assert len(everything) == 60
    wanted = []
    for item in everything:
        if item["subject"] in ("virology", "law"):
            wanted.append(item["id"])
    got = []
    for item in mcq.mcq_items("mmlu", "test", seed=3, subjects=["virology", "law"]):
        got.append(item["id"])
    assert got == wanted


def test_other_benchmarks_read_their_rows(monkeypatch):
    pro = [{"question": "p", "options": ["o0", "o1", "o2", "o3", "o4", "o5", "o6", "o7", "o8", "o9"],
            "answer": "J", "answer_index": 9, "category": "physics"}]
    med = [{"question": "m", "opa": "a", "opb": "b", "opc": "c", "opd": "d", "cop": 2, "subject_name": "Anatomy"},
           {"question": "bad", "opa": "a", "opb": "b", "opc": "c", "opd": "d", "cop": -1, "subject_name": ""}]
    usmle = [{"question": "u", "options": {"B": "bee", "A": "ay", "D": "dee", "C": "cee"}, "answer_idx": "C",
              "answer": "cee", "meta_info": "step1"}]
    tables = {
        ("TIGER-Lab/MMLU-Pro", None, "test"): pro,
        ("openlifescienceai/medmcqa", None, "validation"): med,
        ("GBaker/MedQA-USMLE-4-options", None, "test"): usmle,
    }
    module = fake_datasets(tables)
    monkeypatch.setitem(sys.modules, "datasets", module)

    item = mcq.mcq_items("mmlu_pro", "test")[0]
    assert item["subject"] == "physics"
    assert item["gold"] == 9
    assert "\nJ. o9\nAnswer:" in item["prompt"]

    items = mcq.mcq_items("medmcqa", "validation")
    assert len(items) == 1
    assert items[0]["subject"] == "Anatomy"
    assert items[0]["gold"] == 2

    item = mcq.mcq_items("medqa", "test")[0]
    assert item["choices"] == ["ay", "bee", "cee", "dee"]
    assert item["gold"] == 2
    assert item["subject"] == "step1"

    with pytest.raises(ValueError):
        mcq.mcq_items("nope", "test")


def test_as_pairs_matches_round6_completion():
    items = [{"prompt": "Question: q\nA. a\nB. b\nAnswer:", "gold": 1}]
    assert mcq.as_pairs(items) == [("Question: q\nA. a\nB. b\nAnswer:", " B")]


# Scoring

WORDS = ["Question", "Answer", "question", "alpha", "beta", "gamma", "delta", ":", ".", "?"]


def tiny_tokenizer():
    vocab = {"<unk>": 0, "</s>": 1, "<pad>": 2, "<s>": 3}
    for letter in mcq.LETTERS:
        vocab[letter] = len(vocab)
    for word in WORDS:
        vocab[word] = len(vocab)
    base = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    base.pre_tokenizer = pre_tokenizers.Whitespace()
    base.post_processor = processors.TemplateProcessing(single="<s> $A", special_tokens=[("<s>", 3)])
    return transformers.PreTrainedTokenizerFast(
        tokenizer_object=base, unk_token="<unk>", eos_token="</s>", pad_token="<pad>", bos_token="<s>",
    )


def donor_tokenizer():
    vocab = {"<unk>": 0, "<|end|>": 1}
    for letter in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJQ:.0123456789":
        if letter not in vocab:
            vocab[letter] = len(vocab)
    base = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    base.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Whitespace(),
        pre_tokenizers.Split(Regex("."), behavior="isolated"),
    ])
    return transformers.PreTrainedTokenizerFast(
        tokenizer_object=base, unk_token="<unk>", eos_token="<|end|>", pad_token="<|end|>",
    )


def tiny_host(seed=0):
    torch.manual_seed(seed)
    config = transformers.LlamaConfig(
        hidden_size=32, intermediate_size=64, num_hidden_layers=3, num_attention_heads=4,
        num_key_value_heads=2, vocab_size=64, max_position_embeddings=256,
    )
    return transformers.LlamaForCausalLM(config).eval()


def tiny_donor(seed=1):
    torch.manual_seed(seed)
    config = transformers.Qwen2Config(
        hidden_size=24, intermediate_size=48, num_hidden_layers=4, num_attention_heads=4,
        num_key_value_heads=2, vocab_size=64, max_position_embeddings=256,
    )
    return transformers.Qwen2ForCausalLM(config).eval()


class FixedLetterModel:
    """Prefers the letters in rank order at the last position: rank[0] scores highest."""

    def __init__(self, tokenizer, rank):
        self.device = torch.device("cpu")
        self.ids = mcq.letter_ids(tokenizer)
        self.rank = rank
        self.vocab = len(tokenizer)

    def __call__(self, input_ids, attention_mask=None):
        logits = torch.zeros(input_ids.shape[0], input_ids.shape[1], self.vocab)
        for place, letter in enumerate(self.rank):
            logits[:, -1, self.ids[mcq.LETTERS.index(letter)]] = 10.0 - place
        return types.SimpleNamespace(logits=logits)


def item(index, choices, gold, subject="s"):
    return {"id": f"t/{index}", "subject": subject, "choices": choices, "gold": gold,
            "prompt": mcq.mcq_prompt(f"question {index}", choices)}


def test_letter_ids_are_the_letter_tokens():
    tok = tiny_tokenizer()
    ids = mcq.letter_ids(tok)
    assert len(ids) == 10
    assert len(set(ids)) == 10
    assert tok.convert_ids_to_tokens(ids[:4]) == ["A", "B", "C", "D"]


def test_letter_scoring_limits_to_each_items_options():
    tok = tiny_tokenizer()
    model = FixedLetterModel(tok, ["J", "C", "A"])
    ten = []
    for index in range(10):
        ten.append(f"alpha {index}")
    items = [item(0, ["alpha", "beta", "gamma", "delta"], 2), item(1, ten, 9), item(2, ten, 2)]
    records = mcq.score_items(model, tok, items, batch_size=2)
    assert [r["pred"] for r in records] == [2, 9, 9]
    assert [r["correct"] for r in records] == [True, True, False]
    assert len(records[0]["probs"]) == 4
    assert len(records[1]["probs"]) == 10
    assert abs(sum(records[1]["probs"]) - 1.0) < 1e-4
    assert records[0]["probs"][2] > records[0]["probs"][0] > records[0]["probs"][1]
    assert tok.padding_side == "right"


def test_round6_tuples_are_accepted():
    tok = tiny_tokenizer()
    model = FixedLetterModel(tok, ["B"])
    records = mcq.score_items(model, tok, [("Question: q\nA. a\nB. b\nC. c\nD. d\nAnswer:", 1)])
    assert records[0]["correct"]
    assert records[0]["id"] == "0"


def test_real_tiny_model_pred_is_argmax_over_letters():
    tok = tiny_tokenizer()
    model = tiny_host()
    items = [item(0, ["alpha", "beta", "gamma", "delta"], 0), item(1, ["beta", "gamma"], 1)]
    records = mcq.score_items(model, tok, items)
    ids = mcq.letter_ids(tok)
    for record, one in zip(records, items):
        encoded = tok([one["prompt"]], return_tensors="pt")
        with torch.no_grad():
            logits = model(**encoded).logits[0, -1]
        scores = logits[ids[:len(one["choices"])]]
        assert record["pred"] == int(scores.argmax())
        assert record["correct"] == (record["pred"] == one["gold"])


def test_tapped_host_with_fresh_tap_matches_plain_and_clears_context():
    tok = tiny_tokenizer()
    host = tiny_host()
    items = [item(0, ["alpha", "beta", "gamma", "delta"], 0), item(1, ["beta", "gamma", "delta", "alpha"], 3),
             item(2, ["gamma", "delta"], 1)]
    plain = mcq.score_items(host, tok, items)
    torch.manual_seed(5)
    branch = TapBranch(32, 24, attn_dim=16, heads=4, donor_id="d", donor_layer=2)
    attach_taps(host, [(1, branch, 0)])
    donors = {"d": (tiny_donor(), donor_tokenizer())}
    tapped = mcq.score_items(host, tok, items, donors=donors, batch_size=2)
    assert tapped == plain
    assert host.wsp_tap_state["memories"] is None

    with torch.no_grad():
        branch.gate.fill_(0.9)
    moved = mcq.score_items(host, tok, items, donors=donors, derange=True)
    assert len(moved) == 3
    assert host.wsp_tap_state["memories"] is None


# Analysis

def records_from(name_correct, subjects=None):
    out = []
    for index, correct in enumerate(name_correct):
        subject = "s"
        if subjects is not None:
            subject = subjects[index]
        out.append({"id": f"i{index}", "subject": subject, "gold": 0, "pred": 0 if correct else 1,
                    "probs": [0.5, 0.5], "correct": bool(correct)})
    return out


def test_accuracy_by_subject_math():
    records = records_from([1, 0, 1, 1, 0], ["a", "a", "b", "b", "c"])
    table = mcq.accuracy_by_subject(records)
    assert list(table) == ["a", "b", "c"]
    assert table["a"] == {"accuracy": 0.5, "correct": 1, "count": 2}
    assert table["b"]["accuracy"] == 1.0
    assert table["c"]["accuracy"] == 0.0
    assert mcq.accuracy(records) == 0.6


def test_oracle_counts_any_correct_on_shared_items():
    a = records_from([1, 0, 0, 0])
    b = records_from([0, 1, 0, 0])
    c = records_from([0, 0, 0, 1])
    assert mcq.oracle_best_of([a]) == 0.25
    assert mcq.oracle_best_of([a, b]) == 0.5
    assert mcq.oracle_best_of([a, b, c]) == 0.75
    # Items missing from one list drop out of the oracle.
    assert mcq.oracle_best_of([a, b[:2]]) == 1.0


def test_complementarity_splits_items():
    a = records_from([1, 1, 0, 0, 1])
    b = records_from([1, 0, 1, 0, 0])
    split = mcq.complementarity(a, b)
    assert split == {"only_a": ["i1", "i4"], "only_b": ["i2"], "both": ["i0"], "neither": ["i3"]}


def test_best_model_per_subject():
    subjects = ["x", "x", "y", "y"]
    runs = {"m1": records_from([1, 1, 0, 0], subjects), "m2": records_from([1, 0, 1, 1], subjects),
            "m3": records_from([1, 1, 1, 1], subjects)}
    best = mcq.best_model_per_subject(runs)
    # m1 and m3 tie on x; the first listed wins.
    assert best["x"]["model"] == "m1"
    assert best["y"]["model"] == "m2"
    assert best["y"]["scores"] == {"m1": 0.0, "m2": 1.0, "m3": 1.0}
    assert best["x"]["count"] == 2


def test_suggest_ranks_by_rescues_and_by_accuracy():
    subjects = ["med", "med", "med", "med", "med", "med", "law", "law"]
    runs = {
        "host": records_from([1, 1, 1, 0, 0, 0, 1, 1], subjects),
        # strong but overlaps the host: rescues 1
        "twin": records_from([1, 1, 1, 1, 0, 0, 0, 0], subjects),
        # weaker on the domain but rescues 2 items the host misses
        "comp": records_from([0, 0, 0, 0, 1, 1, 0, 0], subjects),
        "none": records_from([0, 0, 0, 0, 0, 0, 1, 1], subjects),
    }
    plan = mcq.suggest_composition(runs, ["med"], host="host")
    assert plan["host"] == "host"
    assert plan["host_accuracy"] == 0.5
    assert plan["count"] == 6
    assert plan["by_accuracy"] == ["twin", "comp", "none"]
    assert plan["by_rescues"] == ["comp", "twin", "none"]
    first = plan["donors"][0]
    assert first["name"] == "comp"
    assert first["rescues"] == 2
    assert first["hurts"] == 3
    assert abs(first["oracle_with_host"] - 5 / 6) < 1e-9
    assert plan["greedy"][0][0] == "comp"
    assert plan["greedy"][1][0] == "twin"
    assert plan["greedy"][1][1] == 1.0
    assert len(plan["greedy"]) == 2

    # With no host given, the best model on the domain becomes the host.
    auto = mcq.suggest_composition(runs, ["med"])
    assert auto["host"] == "twin"
    # On law alone, ties keep the listed order.
    law = mcq.suggest_composition(runs, ["law"])
    assert law["host"] == "host"


def test_suggest_counts_unscored_items_as_misses():
    # The host is right on the even items of 20.
    host = []
    for index in range(20):
        host.append(1 if index % 2 == 0 else 0)
    # "partial" scored only 4 items, all ones the host already gets.
    partial = records_from([1, 1, 1, 1])
    partial_ids = ["i0", "i2", "i4", "i6"]
    for record, item_id in zip(partial, partial_ids):
        record["id"] = item_id
    # "full" scored every item and rescues i1 and i3.
    full = []
    for index in range(20):
        full.append(1 if index % 2 == 0 or index < 4 else 0)
    runs = {"host": records_from(host), "partial": partial, "full": records_from(full)}
    plan = mcq.suggest_composition(runs, None, host="host")
    assert plan["greedy"] == [("full", 12 / 20)]
    rows = {}
    for row in plan["donors"]:
        rows[row["name"]] = row
    assert rows["partial"]["oracle_with_host"] == 0.5
    assert rows["partial"]["shared"] == 4
    assert abs(rows["full"]["oracle_with_host"] - 0.6) < 1e-9


def test_single_subject_string_filters_by_name(mmlu_tables):
    items = mcq.mcq_items("mmlu", "test", seed=0, subjects="anatomy")
    assert len(items) == 20
    for item in items:
        assert item["subject"] == "anatomy"


def test_records_round_trip_jsonl(tmp_path):
    records = records_from([1, 0, 1], ["a", "b", "c"])
    path = tmp_path / "records.jsonl"
    mcq.save_records(records, str(path))
    assert len(path.read_text().splitlines()) == 3
    assert mcq.load_records(str(path)) == records


def test_mean_log_prob_gold():
    records = [{"probs": [0.5, 0.5], "gold": 0}, {"probs": [1.0, 0.0], "gold": 0}]
    assert abs(mcq.mean_log_prob_gold(records) - (-0.6931471805599453 / 2)) < 1e-9
