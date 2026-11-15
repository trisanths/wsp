import json

import pytest

from wsp import data as wdata
from wsp import healdata
from wsp.compiler import CompileError, check_program, pairs_by_name, written_heal_texts
from wsp.lang import format_program, parse
from wsp.program import HealSpec, Program, TapTrainSpec


class FakePairs:
    """Stands in for healdata.pairs. One fetch per pool, with unique rows and no network."""

    def __init__(self):
        self.calls = []

    def __call__(self, name, count, seed=0, skip=0):
        self.calls.append((name, count, seed, skip))
        rows = []
        for index in range(count + skip):
            rows.append((f"{name} question {index}", f"{name} answer {index}"))
        return rows[skip:skip + count]


@pytest.fixture
def fake(monkeypatch):
    fake_pairs = FakePairs()
    monkeypatch.setattr(healdata, "pairs", fake_pairs)
    return fake_pairs


def test_default_data_is_general():
    assert wdata.DEFAULT_DATA == "general"
    assert HealSpec().data == "general"
    assert Program(host="h").heal().heal_spec.data == "general"
    assert TapTrainSpec().data == "general"
    assert parse("host h\nlora rank 2\nheal steps 5\n").heal_spec.data == "general"


def test_pool_names_resolve_through_healdata(fake):
    for name in ["general", "math", "code", "medicine", "prose"]:
        texts = wdata.task_texts(name, seed=3)
        assert len(texts) == wdata.HEAL_TEXT_COUNT
        assert texts[0].startswith(name + " question")
    assert fake.calls[0] == ("general", wdata.HEAL_TEXT_COUNT, 3, 0)


def test_programs_with_no_data_name_compile_with_pool_text(fake):
    program = parse("host h\nlora rank 2\nheal steps 5\n")
    texts = written_heal_texts(program.heal_spec, None, None, None, 0)
    assert texts[0].startswith("general question")


def test_mix_is_an_even_mix_of_four_pools(fake):
    texts = wdata.pool_texts("mix", 40)
    assert len(texts) == 40
    for pool in ["general", "math", "code", "medicine"]:
        assert len([t for t in texts if t.startswith(pool)]) == 10
    chosen = wdata.pool_texts("mix:math,code", 10)
    assert sorted(set(t.split()[0] for t in chosen)) == ["code", "math"]


def test_mix_validates_pool_names():
    assert wdata.pool_names_of("mix:math,code") == ["math", "code"]
    for bad in ["mix:", "mix:math,", "mix:math,gsm8k", "mix:math,math", "mix:nope"]:
        with pytest.raises(ValueError):
            wdata.pool_names_of(bad)
    with pytest.raises(CompileError):
        check_program(Program(host="h").lora(rank=2).heal(data="mix:math,nope"))
    with pytest.raises(ValueError):
        parse("host h\nlora rank 2\nheal data mix:math,nope\n")


def test_benchmark_names_raise_without_the_escape_hatch():
    for name in ["gsm8k_train", "mmlu_aux"]:
        program = Program(host="h").lora(rank=2).heal(data=name)
        with pytest.raises(CompileError) as error:
            check_program(program)
        message = str(error.value)
        assert "benchmark" in message and "general" in message and "mix" in message
        check_program(program, allow_benchmark_data=True)
    assert wdata.benchmark_source("gsm8k_train") == "openai/gsm8k"
    assert wdata.benchmark_source("mmlu_aux") == "cais/mmlu"
    # TriviaQA is not banned.
    check_program(Program(host="h").tap("d", 1, 1).tap_train(data="trivia_train"))
    assert wdata.benchmark_source("trivia_train") is None
    assert wdata.benchmark_source("general") is None


def test_carve_domain_and_tap_train_names_are_checked():
    carve = Program(host="h").carve(keep=2, select="domain", domain_data="gsm8k_train")
    with pytest.raises(CompileError):
        check_program(carve)
    tapped = Program(host="h").tap("d", 1, 1).tap_train(data="mmlu_aux")
    with pytest.raises(CompileError):
        check_program(tapped)
    check_program(tapped, allow_benchmark_data=True)


def test_compile_refuses_before_loading_a_model():
    program = Program(host="no/such-model").lora(rank=2).heal(data="gsm8k_train")
    with pytest.raises(CompileError):
        program.compile("/nonexistent/out", None)


def test_report_records_benchmark_data(tmp_path):
    from wsp.compiler import finish_outputs

    class Model:
        def parameters(self):
            return []

        def named_parameters(self):
            return []

    import wsp.compiler as compiler

    compiler_count = compiler.count_model_params
    compiler.count_model_params = lambda model, exclude_lora=False: (1, 1)
    try:
        for name, expected in [("gsm8k_train", True), ("general", False)]:
            out = tmp_path / name
            out.mkdir()
            program = Program(host="h").lora(rank=2).heal(data=name)
            finish_outputs(program, Model(), {"heal": None}, str(out))
            report = json.loads((out / "compile_report.json").read_text())
            assert report.get("benchmark_data", False) is expected
    finally:
        compiler.count_model_params = compiler_count


def test_text_syntax_round_trip():
    for data in ["general", "mix", "mix:math,code", "prose"]:
        text = f"host h\nlora rank 2\nheal steps 300 data {data} lr 3e-4\n"
        program = parse(text)
        assert program.heal_spec.data == data
        again = parse(format_program(program))
        assert again.to_dict() == program.to_dict()
        assert f"data {data} " in format_program(program)
    carve = parse("from m\ncarve keep 3 select domain domain mix:math,code\n")
    assert carve.carve_spec.domain_data == "mix:math,code"
    assert parse(format_program(carve)).to_dict() == carve.to_dict()
    tap = parse("host h\ntap d layer 1 at 2\ntap_train data mix:math,code\n")
    assert parse(format_program(tap)).to_dict() == tap.to_dict()


def test_align_and_heal_slices_are_disjoint_from_one_fetch(fake):
    align, heal = wdata.pool_split("math", 50, 200, None, seed=1)
    assert len(align) == 50 and len(heal) == 200
    assert set(align).isdisjoint(heal)
    assert len(fake.calls) == 1
    assert fake.calls[0] == ("math", 250, 1, 0)


def test_mix_align_and_heal_slices_are_disjoint(fake):
    align, heal = wdata.pool_split("mix", 20, 80, None, seed=0)
    assert len(align) == 20 and len(heal) == 80
    assert set(align).isdisjoint(heal)
    assert len(fake.calls) == 4
    for call in fake.calls:
        assert call[3] == 0


def test_same_pool_carve_and_heal_use_disjoint_ranges(fake):
    domain, heal = wdata.pool_split("math", 256, wdata.HEAL_TEXT_COUNT, None, 0)
    assert set(domain).isdisjoint(heal)


def test_pairs_by_name_serves_pools_for_taps(fake):
    got = pairs_by_name("mix:math,code", None)
    assert len(got) == wdata.HEAL_TEXT_COUNT
    prompt, completion = got[0]
    assert prompt.endswith("\nAnswer:") and completion.startswith(" ")


def test_compile_file_loads_disjoint_align_and_heal_texts(fake, monkeypatch, tmp_path):
    import wsp.compiler
    import wsp.lang

    seen = {}

    def fake_compile(program, catalog, out_dir, **options):
        seen.update(options)
        return out_dir

    monkeypatch.setattr(wsp.compiler, "compile_program", fake_compile)

    import transformers
    from wsp.catalog import Catalog

    monkeypatch.setattr(Catalog, "load", classmethod(lambda cls, path: object()))

    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", lambda *a, **k: None)
    out = tmp_path / "out"
    out.mkdir()
    (out / "compile_report.json").write_text("{}")
    program = Program(host="h").graft("g").heal(data="mix:math,code")
    wsp.lang.compile_file(program, str(out), "catalog.jsonl", 10, "cpu")
    assert len(seen["align_texts"]) == 10
    assert set(seen["align_texts"]).isdisjoint(seen["heal_texts"])
    assert "allow_benchmark_data" not in seen
    with pytest.raises(CompileError):
        wsp.lang.compile_file(Program(host="h").graft("g").heal(data="gsm8k_train"), str(out), "catalog.jsonl", 10, "cpu")
