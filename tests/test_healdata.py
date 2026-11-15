import pytest

from wsp import healdata


def test_benchmarks_are_banned_in_any_spelling():
    for name in ["openai/gsm8k", "OpenAI/GSM8K", "cais/mmlu", "google-research-datasets/mbpp",
                 "openlifescienceai/medmcqa", "HuggingFaceH4/MATH-500", "meta-math/MetaMathQA"]:
        assert healdata.is_banned(name)
    with pytest.raises(ValueError):
        healdata.check_sources(["yahma/alpaca-cleaned", "openai/gsm8k"])


def test_no_pool_reads_a_banned_dataset():
    for name in healdata.SOURCES:
        ids = []
        for source in healdata.SOURCES[name]:
            ids.append(source[0])
        assert healdata.check_sources(ids) == ids


def test_row_pairs():
    assert healdata.row_pair({"question": "q", "answer": "a"}, "qa") == ("q", "a")
    assert healdata.row_pair({"instruction": "i", "input": "x", "output": "o"}, "alpaca") == ("i\n\nx", "o")
    assert healdata.row_pair({"instruction": "i", "input": "", "output": ""}, "alpaca") is None
    assert healdata.row_pair({"problem": "p", "solution": "s", "lang": "rust"}, "magicoder") is None
    assert healdata.row_pair({"problem": "p", "solution": "s", "lang": "python"}, "magicoder") == ("p", "s")
    assert healdata.row_pair({"text": " = Heading = "}, "text") is None
    row = {"messages": [{"content": "u"}, {"content": "a"}]}
    assert healdata.row_pair(row, "ultrachat") == ("u", "a")


def test_render_without_a_template():
    assert healdata.render(None, "u", "a") == "u\n\na"
    assert healdata.render(None, "plain", None) == "plain"
