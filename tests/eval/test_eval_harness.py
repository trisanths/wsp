import json

import pytest

from wsp_eval import budget, cache, harness, splits, tasks as tasks_module


def test_extract_letter():
    assert tasks_module.extract_letter("B", 4) == "B"
    assert tasks_module.extract_letter("B. Mitochondria", 4) == "B"
    assert tasks_module.extract_letter("The answer is (C).", 10) == "C"
    assert tasks_module.extract_letter("Answer: J", 10) == "J"
    assert tasks_module.extract_letter("A cell is ... so (D)", 4) == "D"
    assert tasks_module.extract_letter("J", 4) is None
    assert tasks_module.extract_letter("no idea", 4) is None


def test_gsm8k_and_bbeh_parsing():
    assert tasks_module.extract_number("so 3 + 4 = 7. The answer is: 1,234") == "1234"
    assert tasks_module.extract_number("we get 12 apples, then 18") == "18"
    assert tasks_module.numbers_equal("18.0", "18")
    assert tasks_module.bbeh_extract_answer("blah\nThe final answer is: **(b)**.") == "(b)"
    assert tasks_module.bbeh_fuzzy_match("(b)", "b")
    assert tasks_module.bbeh_fuzzy_match("42.0", "42")
    assert tasks_module.bbeh_fuzzy_match("disproved", "Disproved")
    assert not tasks_module.bbeh_fuzzy_match("proved", "disproved")


def test_task_names_and_training_set_mapping():
    assert tasks_module.get_task("mmlupro:computer_science").category == "computer science"
    assert tasks_module.get_task("bbeh:web_of_lies").subtask == "web of lies"
    assert tasks_module.task_for_training_set("MMLU-Pro biology SFT") == "mmlupro:biology"
    assert tasks_module.task_for_training_set("BBEH boolean expressions") == "bbeh:boolean_expressions"
    assert tasks_module.task_for_training_set("BBEH") == "bbeh"
    with pytest.raises(ValueError):
        tasks_module.get_task("mmlupro:astrology")


def test_splits_are_deterministic_and_disjoint(workspace):
    task = tasks_module.get_task("toy")
    search_one = splits.split_items(task, "search")
    search_two = splits.split_items(task, "search")
    heldout = splits.split_items(task, "heldout")
    ids_search = [item["id"] for item in search_one]
    assert ids_search == [item["id"] for item in search_two]
    assert len(search_one) == 150
    assert len(heldout) == 250
    assert set(ids_search).isdisjoint(item["id"] for item in heldout)


def test_eval_format_cache_budget_and_log(workspace):
    result = harness.eval("fake:base", "search", ["toy", "toy2"], namespace="swarm")
    assert result["split"] == "search"
    assert result["label"] == harness.SEARCH_LABEL
    assert set(result["tasks"].keys()) == {"toy", "toy2"}
    assert 0.0 <= result["tasks"]["toy"]["score"] <= 1.0
    assert result["tasks"]["toy"]["n"] == 150
    assert result["generations_run"] == 300
    assert result["budget"]["eval_calls"] == 1
    assert result["combined"] is None  # no reference yet

    again = harness.eval("fake:base", "search", ["toy", "toy2"], namespace="swarm")
    assert again["tasks"] == result["tasks"]
    assert again["generations_run"] == 0          # served from the cache
    assert again["budget"]["eval_calls"] == 2     # but still counted

    other = harness.eval("fake:other", "search", ["toy"], namespace="lora")
    assert other["budget"]["eval_calls"] == 1     # separate namespace

    lines = open(workspace / "experiments.jsonl").read().strip().split("\n")
    assert len(lines) == 3
    first = json.loads(lines[0])
    for key in ("timestamp", "model", "split", "scores", "eval_count"):
        assert key in first


def test_budget_limit(workspace):
    budget.set_limit("swarm", 2)
    harness.eval("fake:m1", "search", ["toy"], namespace="swarm")
    harness.eval("fake:m2", "search", ["toy"], namespace="swarm")
    with pytest.raises(budget.BudgetExhausted):
        harness.eval("fake:m3", "search", ["toy"], namespace="swarm")
    # Held-out calls are never blocked by the search budget.
    harness.eval("fake:m1", "heldout", ["toy"], namespace="swarm")
    assert budget.usage("swarm")["eval_calls"] == 2
    assert budget.usage("swarm")["heldout_calls"] == 1


def test_heldout_runs_once(workspace):
    result = harness.eval("fake:final", "heldout", ["toy"])
    assert result["label"] == harness.HELDOUT_LABEL
    assert result["tasks"]["toy"]["n"] == 250
    with pytest.raises(harness.HeldoutAlreadyRun):
        harness.eval("fake:final", "heldout", ["toy"])
    # A different task on the same model is still allowed once.
    harness.eval("fake:final", "heldout", ["toy2"])


def test_combined_score(workspace):
    reference = harness.save_reference("search", {"toy": 0.2, "toy2": 0.5},
                                       {"parent_toy": {"toy": 0.6, "toy2": 0.5},
                                        "parent_toy2": {"toy": 0.2, "toy2": 0.7}})
    combined = harness.combined_score({"toy": 0.4, "toy2": 0.8}, reference)
    assert combined["per_skill"]["toy"]["normalized"] == pytest.approx(0.5)
    assert combined["per_skill"]["toy2"]["raw"] == pytest.approx(1.5)
    assert combined["per_skill"]["toy2"]["normalized"] == 1.0
    assert combined["score"] == pytest.approx(0.75)

    # A skill whose best parent does not beat the base is skipped, not divided by zero.
    flat = harness.save_reference("search", {"toy": 0.5}, {"p": {"toy": 0.5}})
    combined = harness.combined_score({"toy": 0.9}, flat)
    assert combined["score"] is None
    assert combined["skipped"] == ["toy"]

    # eval() picks the reference up automatically.
    harness.save_reference("search", {"toy": 0.0}, {"p": {"toy": 1.0}})
    result = harness.eval("fake:base", "search", ["toy"])
    assert result["combined"]["score"] == pytest.approx(result["tasks"]["toy"]["score"])


def test_cache_key_separates_models_and_lengths():
    a = cache.make_key("h1", "hf", 32, "prompt")
    assert a != cache.make_key("h2", "hf", 32, "prompt")
    assert a != cache.make_key("h1", "hf", 64, "prompt")
    assert a != cache.make_key("h1", "vllm", 32, "prompt")


def test_gsm8k_scoring_matches_team_rule():
    task = tasks_module.get_task("gsm8k")
    item = {"answer": 72.0}
    assert task.score(" She sold 48 + 24 = 72 clips.\n#### 72\n\nQuestion: next one 5", item) == 1.0
    assert task.score(" 48/2 = 24, so 48 + 24 = 72\n\nQuestion: 9", item) == 1.0
    assert task.score(" 1,000 apples", {"answer": 1000.0}) == 1.0
    assert task.score(" maybe 71", item) == 0.0
    assert tasks_module.gsm8k_final_number("#### 1,234") == 1234.0


def test_gsm8k_splits_match_wsp_data():
    pytest.importorskip("datasets")
    from wsp.data import gsm8k_splits
    task = tasks_module.get_task("gsm8k")
    try:
        search, heal = gsm8k_splits(seed=0)
    except Exception as error:   # no network and no local copy
        pytest.skip("GSM8K not available: " + str(error))
    ours = splits.split_items(task, "search")
    assert [item["question"] for item in ours] == [row["question"] for row in search]
    assert ours[0]["prompt"] == "Question: " + search[0]["question"] + "\nAnswer:"
    assert len(splits.split_items(task, "heal")) == len(heal)
    from wsp.data import format_gsm8k
    assert splits.heal_texts("gsm8k")[0] == format_gsm8k(heal[0])
    heldout_ids = set(item["id"] for item in splits.split_items(task, "heldout"))
    assert len(heldout_ids) == 1319
