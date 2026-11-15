"""The harness task rule (modal_swarm.scorer_allows). CPU only, no network."""

import modal_swarm


def test_the_harness_grades_mmlu_subjects_and_its_named_tasks():
    for task in ["gsm8k", "if_guard", "humaneval", "mmlu", "mmlu:professional_law", "mmlupro:biology"]:
        assert modal_swarm.scorer_allows("harness", task), task


def test_the_rule_refuses_what_no_scorer_grades():
    assert modal_swarm.scorer_allows("interim", "gsm8k")
    assert not modal_swarm.scorer_allows("interim", "mmlu:professional_law")
    assert not modal_swarm.scorer_allows("harness", "mmlu:")
    assert not modal_swarm.scorer_allows("harness", "legalbench")
