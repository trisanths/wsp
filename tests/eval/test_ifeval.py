import pytest

from wsp_eval import ifeval, splits, tasks as tasks_module


def none_kwargs(**values):
    keys = ["num_highlights", "relation", "num_words", "num_placeholders", "prompt_to_repeat", "num_bullets",
            "section_spliter", "num_sections", "capital_relation", "capital_frequency", "keywords",
            "num_paragraphs", "language", "let_relation", "letter", "let_frequency", "end_phrase",
            "forbidden_words", "keyword", "frequency", "num_sentences", "postscript_marker", "first_word",
            "nth_paragraph"]
    kwargs = {}
    for key in keys:
        kwargs[key] = values.get(key)
    return kwargs


def check(instruction, response, **values):
    return ifeval.check_instruction(instruction, none_kwargs(**values), response, "")


def test_checks():
    assert check("punctuation:no_comma", "no commas here")
    assert not check("punctuation:no_comma", "a, b")
    assert check("length_constraints:number_words", "one two three", relation="less than", num_words=4)
    assert not check("length_constraints:number_words", "one two three", relation="at least", num_words=4)
    assert check("length_constraints:number_sentences", "Hi. How are you? Fine!", relation="at least", num_sentences=3)
    assert check("keywords:forbidden_words", "a cat sat", forbidden_words=["dog"])
    assert not check("keywords:forbidden_words", "a Dog sat", forbidden_words=["dog"])
    assert check("keywords:frequency", "lime lime lime", keyword="lime", relation="at least", frequency=3)
    assert check("detectable_format:number_highlighted_sections", "*one* and **two**", num_highlights=2)
    assert check("startend:quotation", '"quoted"')
    assert check("change_case:english_lowercase", "all lower")
    assert not check("change_case:english_capital", "Not All")
    assert check("detectable_format:title", "<<My Title>>\nbody")
    assert check("detectable_format:number_bullet_lists", "* a\n* b\n- c", num_bullets=3)
    assert check("detectable_content:number_placeholders", "[name] at [address]", num_placeholders=2)
    assert check("length_constraints:number_paragraphs", "p1 *** p2 *** p3", num_paragraphs=3)
    assert check("startend:end_checker", 'Thanks. Any other questions?"', end_phrase="Any other questions?")
    assert check("detectable_content:postscript", "body\nP.S. remember", postscript_marker="P.S.")
    assert check("combination:two_responses", "first answer\n******\nsecond answer")
    assert not check("combination:two_responses", "same\n******\nsame")
    assert check("detectable_format:json_format", '```json\n{"a": 1}\n```')
    assert not check("detectable_format:json_format", "{not json")
    assert check("detectable_format:multiple_sections", "SECTION 1 a\nSECTION 2 b", section_spliter="SECTION",
                 num_sections=2)
    assert check("length_constraints:nth_paragraph_first_word", "Alpha a\n\nBeta b", num_paragraphs=2,
                 nth_paragraph=2, first_word="beta")
    assert check("detectable_format:constrained_response", "My answer is yes.")
    assert check("change_case:capital_word_frequency", "THE CAT sat", capital_relation="at least",
                 capital_frequency=2)
    assert check("keywords:letter_frequency", "aaa", letter="a", let_relation="at least", let_frequency=3)
    assert check("combination:repeat_prompt", "Say hi. Hi!", prompt_to_repeat="say hi.")
    with pytest.raises(ValueError):
        check("language:response_language", "hola", language="es")


def test_if_guard_splits_and_every_instruction_supported():
    pytest.importorskip("datasets")
    task = tasks_module.get_task("if_guard")
    try:
        items = task.load_items()
    except Exception as error:
        pytest.skip("IFEval not available: " + str(error))
    assert len(items) == 510
    search = splits.split_items(task, "search")
    heldout = splits.split_items(task, "heldout")
    assert len(search) == 100 and len(heldout) == 410
    assert set(i["id"] for i in search).isdisjoint(i["id"] for i in heldout)
    # Every remaining instruction has a checker (an empty response must not raise).
    for item in items:
        task.score("", item)
