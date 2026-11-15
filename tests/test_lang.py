import json
import os
import random
import re
import subprocess
import sys

import pytest

from wsp.lang import LangError, format_program, load_program, main, parse
from wsp.program import GraftSpec, Program, use

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HOST = "unsloth/Llama-3.2-1B-Instruct"


def repo_python():
    path = os.path.join(REPO, ".venv", "bin", "python")
    if os.path.exists(path):
        return path
    return sys.executable


def round_trip(program):
    text = format_program(program)
    again = parse(text)
    assert again.to_dict() == program.to_dict(), text
    # Writing the parsed program gives the same text again.
    assert format_program(again) == text
    return text


def full_program():
    bio = use("mask_mmlupro_biology_llama1b")
    chem = use("mask_mmlupro_chemistry_llama1b")
    law = use("law.v2")
    masks = ((bio | chem) - law).restrict(layers=range(4, 13), modules=["mlp", "self_attn"])
    program = Program(host=HOST, masks=masks, seed=7)
    program.tv("code_tv", weight=0.5, layers=[10, 11, 12, 20], modules=["mlp"])
    program.tv("org/other-tv", weight=-1)
    program.tv("plain_tv")
    program.lora(rank=16, targets=["q_proj", "v_proj"], alpha=32, layers=[0, 1, 2, 5])
    program.graft("qmath_block_8_19", at=5, glue_rank=64,
                  stitch={"kind": "mlp", "objective": "downstream", "steps": 500, "lr": 1e-3, "hidden": 256})
    program.graft("qmath_l18_contrast512", control="random", control_seed=101)
    program.graft("qmath_multi", at=0, stitch={"kind": "multi", "offsets": [-2, 0], "seed": 3})
    program.graft("qmath_flags", sink=False, rms_match=True, part_lora_rank=4)
    program.graft("qmath_nokind", stitch={"objective": "mse", "batch_size": 4, "end_layer": None})
    program.graft("qmath_empty_stitch", stitch={})
    program.heal(steps=300, data="gsm8k_train", lr=3e-4, batch_size=4, seq_len=256)
    return program


def test_round_trip_every_feature():
    text = round_trip(full_program())
    assert "graft qmath_l18_contrast512 control random seed 101" in text
    assert "layers 4..12 modules mlp,self_attn" in text
    assert "sink off rms_match on part_lora_rank 4" in text
    assert "offsets [-2,0]" in text


def test_round_trip_small_programs():
    round_trip(Program(host=HOST))
    round_trip(Program(host=HOST).lora(budget_params=123456).heal())
    round_trip(Program(host=HOST, masks=use("a")))
    round_trip(Program(host=HOST, masks=use("a").restrict()))
    round_trip(Program(host=HOST, masks=use("a").restrict(modules=["mlp"]).restrict(layers=[3])))
    round_trip(Program(host=HOST, masks=(use("a").restrict(layers=[1]) | use("b"))))
    round_trip(Program(host=HOST, masks=use("a") - (use("b") - use("c"))))
    round_trip(Program(host=HOST, masks=(use("a") | use("b")) & (use("c") | use("d"))))
    round_trip(Program(host=HOST, masks=use("a") | (use("b") | use("c"))))
    round_trip(Program(host=HOST).tv("t", layers=[1, 1, 2, 4, 5, 9]))


def test_suggested_example():
    text = """
host unsloth/Llama-3.2-1B-Instruct
seed 0
graft qmath_block_8_19 at 5 rank 64 stitch mlp objective downstream steps 500
graft qmath_l18_contrast512 control random seed 101
mask (bio | chem) - law restrict layers 4..12 modules mlp
tv code_tv weight 0.5 layers 10..20 modules mlp
lora rank 16 targets q_proj,v_proj
heal steps 300 data gsm8k_train lr 3e-4
"""
    data = parse(text).to_dict()
    assert data["host"] == HOST
    assert data["grafts"][0] == {"id": "qmath_block_8_19", "at": 5, "control": None, "control_seed": 0,
                                 "glue_rank": 64,
                                 "stitch": {"kind": "mlp", "objective": "downstream", "steps": 500}}
    assert data["grafts"][1]["control"] == "random"
    assert data["grafts"][1]["control_seed"] == 101
    assert data["masks"] == {
        "op": "restrict",
        "inner": {"op": "difference",
                  "a": {"op": "union", "a": {"op": "use", "id": "bio"}, "b": {"op": "use", "id": "chem"}},
                  "b": {"op": "use", "id": "law"}},
        "layers": list(range(4, 13)),
        "modules": ["mlp"],
    }
    assert data["task_vectors"] == [{"id": "code_tv", "weight": 0.5, "layers": list(range(10, 21)),
                                     "modules": ["mlp"]}]
    assert data["lora"] == {"rank": 16, "budget_params": None, "targets": ["q_proj", "v_proj"], "alpha": None,
                            "layers": None}
    assert data["heal"] == {"steps": 300, "data": "gsm8k_train", "lr": 3e-4, "batch_size": 8, "seq_len": 384}


def test_mask_precedence_matches_python():
    a = use("a")
    b = use("b")
    c = use("c")
    host = "host " + HOST + "\n"
    assert parse(host + "mask a | b - c").to_dict()["masks"] == (a | b - c).to_dict()
    assert parse(host + "mask a & b | c").to_dict()["masks"] == (a & b | c).to_dict()
    assert parse(host + "mask a - b - c").to_dict()["masks"] == (a - b - c).to_dict()
    assert parse(host + "mask a - b & c").to_dict()["masks"] == (a - b & c).to_dict()
    assert "a - (b - c)" in format_program(Program(host=HOST, masks=a - (b - c)))
    assert "mask a - b - c\n" in format_program(Program(host=HOST, masks=a - b - c))


def test_comments_blank_lines_and_aliases():
    text = """# a whole-line comment

base unsloth/Llama-3.2-1B-Instruct   # base is an older name for host
   seed 3
graft g glue_rank 8 control_seed 2 control random   # long names work too
lora budget 1000
"""
    data = parse(text).to_dict()
    assert data["host"] == HOST
    assert data["seed"] == 3
    assert data["grafts"][0]["glue_rank"] == 8
    assert data["grafts"][0]["control_seed"] == 2
    assert data["lora"]["budget_params"] == 1000


def test_ranges_and_lists():
    text = "host h\ntv t layers 1..3,7,9..10 modules mlp, self_attn\nlora rank 4 targets q_proj , v_proj"
    data = parse(text).to_dict()
    assert data["task_vectors"][0]["layers"] == [1, 2, 3, 7, 9, 10]
    assert data["task_vectors"][0]["modules"] == ["mlp", "self_attn"]
    assert data["lora"]["targets"] == ["q_proj", "v_proj"]
    stitch = parse("host h\ngraft g stitch multi offsets [ -3, -1 , 0 ] lr 0.01 flag true").to_dict()
    assert stitch["grafts"][0]["stitch"] == {"kind": "multi", "offsets": [-3, -1, 0], "lr": 0.01, "flag": True}


def test_numbers_keep_their_type():
    data = parse("host h\nlora rank 4 alpha 8\ntv t weight 2\nheal lr 1e-4").to_dict()
    assert isinstance(data["lora"]["alpha"], int)
    assert data["task_vectors"][0]["weight"] == 2
    assert data["heal"]["lr"] == 1e-4


@pytest.mark.parametrize("text, line, words", [
    ("host h\ngraf g", 2, "unknown statement 'graf'"),
    ("seed 0", None, "needs a host line"),
    ("host h\n\nhost g", 3, "host appears twice (first on line 1)"),
    ("host h\nseed zero", 2, "seed needs an integer"),
    ("host h\ngraft g at five", 2, "at needs an integer"),
    ("host h\ngraft g depth 3", 2, "unknown graft option 'depth'"),
    ("host h\ngraft g at 3 at 4", 2, "sets 'at' twice"),
    ("host h\ngraft g rank 3 glue_rank 4", 2, "sets 'rank' twice"),
    ("host h\ngraft g at", 2, "at needs a value"),
    ("host h\ngraft g control other", 2, "unknown control"),
    ("host h\ngraft g sink maybe", 2, "sink is on or off"),
    ("host h\ngraft g part_lora_rank 0", 2, "part_lora_rank"),
    ("host h\ngraft g stitch kind blob", 2, "unknown stitch kind"),
    ("host h\ngraft g stitch mlp steps", 2, "stitch option steps needs a value"),
    ("host h\ngraft g stitch mlp offsets [1,2", 2, "has no closing ]"),
    ("host h\nmask (a | b", 2, "has no closing )"),
    ("host h\nmask a |", 2, "ends too early"),
    ("host h\nmask a b", 2, "unexpected 'b'"),
    ("host h\nmask a -b", 2, "put spaces around -"),
    ("host h\nmask", 2, "mask needs an expression"),
    ("host h\nmask a restrict layers 5..2", 2, "ends before it starts"),
    ("host h\nmask a restrict layers 1,,2", 2, "empty item"),
    ("host h\nmask a\nmask b", 3, "mask appears twice"),
    ("host h\ntv", 2, "tv needs a catalog id"),
    ("host h\ntv bad$id", 2, "a tv id must use only"),
    ("host h\nlora targets q_proj", 2, "needs a rank or a budget_params"),
    ("host h\nlora rank 4\nlora rank 8", 3, "lora appears twice"),
    ("host h\nheal steps 10 lr fast", 2, "lr needs a number"),
    ("host a b", 1, "host takes one model name"),
])
def test_errors_name_the_line(text, line, words):
    with pytest.raises(LangError) as caught:
        parse(text)
    assert caught.value.line == line
    assert words in str(caught.value)
    if line is not None:
        assert str(caught.value).startswith(f"line {line}: ")


def test_format_refuses_what_text_cannot_hold():
    with pytest.raises(ValueError, match="empty layers list"):
        format_program(Program(host=HOST).tv("t", layers=[]))
    with pytest.raises(ValueError, match="cannot be written"):
        format_program(Program(host="has space"))
    with pytest.raises(ValueError, match="different type"):
        format_program(Program(host=HOST).graft("g", stitch={"note": "12"}))


def test_format_accepts_a_dict_and_catches_new_keys(monkeypatch):
    program = full_program()
    assert format_program(program.to_dict()) == format_program(program)
    # A new GraftSpec option that this printer does not know must fail, not vanish.
    old_to_dict = GraftSpec.to_dict

    def to_dict_with_new_key(self):
        data = old_to_dict(self)
        data["new_option"] = 1
        return data

    monkeypatch.setattr(GraftSpec, "to_dict", to_dict_with_new_key)
    with pytest.raises(ValueError, match="new_option"):
        format_program(program)


def random_mask(rng, depth):
    if depth == 0 or rng.random() < 0.3:
        return use(rng.choice(["a", "b", "c", "mask_x.1", "m-2"]))
    choice = rng.randrange(4)
    left = random_mask(rng, depth - 1)
    if choice == 3:
        layers = None
        modules = None
        if rng.random() < 0.6:
            layers = rng.sample(range(16), rng.randint(1, 6))
        if rng.random() < 0.6:
            modules = rng.sample(["mlp", "self_attn", "mlp.down_proj"], rng.randint(1, 2))
        return left.restrict(layers=layers, modules=modules)
    right = random_mask(rng, depth - 1)
    if choice == 0:
        return left | right
    if choice == 1:
        return left & right
    return left - right


def test_random_programs_round_trip():
    rng = random.Random(0)
    for trial in range(200):
        program = Program(host=HOST, seed=rng.randint(0, 9), masks=random_mask(rng, 4))
        for count in range(rng.randint(0, 2)):
            program.tv("tv" + str(count), weight=rng.choice([1.0, 0.25, -2, 1e-5]),
                       layers=rng.choice([None, [0, 1, 2], [3, 7]]))
        for count in range(rng.randint(0, 3)):
            stitch = rng.choice([None, {"kind": "lowrank", "rank": 8}, {"objective": "donor_lm", "lr": 0.002}])
            program.graft("g" + str(count), at=rng.choice([None, 0, 7]), control=rng.choice([None, "random"]),
                          control_seed=rng.choice([0, 5]), glue_rank=rng.choice([None, 32]), stitch=stitch,
                          sink=rng.choice([True, False]), rms_match=rng.choice([True, False]),
                          part_lora_rank=rng.choice([None, 2]))
        if rng.random() < 0.5:
            program.lora(rank=rng.choice([None, 8]), budget_params=rng.choice([1000, 5000]),
                         targets=rng.choice([None, ["q_proj"]]))
        if rng.random() < 0.5:
            program.heal(steps=rng.randint(1, 500), lr=rng.choice([3e-4, 1e-3]))
        round_trip(program)


def test_docs_examples_parse():
    with open(os.path.join(REPO, "docs", "language.md")) as f:
        docs = f.read()
    blocks = re.findall(r"```wsp\n(.*?)```", docs, re.DOTALL)
    assert len(blocks) >= 5
    for block in blocks:
        if re.search(r"^(host|base|from|splice) ", block, re.MULTILINE) is None:
            block = "host " + HOST + "\n" + block
        round_trip(parse(block))


def write(tmp_path, name, text):
    path = tmp_path / name
    path.write_text(text)
    return str(path)


def test_cli_check_prints_json(tmp_path):
    path = write(tmp_path, "p.wsp", format_program(full_program()))
    result = subprocess.run([repo_python(), "-m", "wsp.lang", "check", path], cwd=REPO, capture_output=True,
                            text=True, timeout=120)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == full_program().to_dict()


def test_cli_check_reports_the_line(tmp_path):
    path = write(tmp_path, "bad.wsp", "host h\n# fine\ngraft g at x\n")
    result = subprocess.run([repo_python(), "-m", "wsp.lang", "check", path], cwd=REPO, capture_output=True,
                            text=True, timeout=120)
    assert result.returncode == 1
    assert "line 3: at needs an integer" in result.stderr


def test_cli_format_reads_json(tmp_path, capsys):
    path = write(tmp_path, "p.json", json.dumps(full_program().to_dict()))
    assert main(["format", path]) == 0
    assert capsys.readouterr().out == format_program(full_program())
    assert load_program(path).to_dict() == full_program().to_dict()


def test_cli_compile_calls_the_compiler(tmp_path, monkeypatch, capsys):
    import wsp.compiler

    calls = []

    def fake_compile(program, catalog, output_dir, align_texts=None, heal_texts=None, device=None):
        calls.append((program.to_dict(), output_dir, align_texts, heal_texts, device))
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, "compile_report.json"), "w") as f:
            json.dump({"host": program.host}, f)
        return output_dir

    monkeypatch.setattr(wsp.compiler, "compile_program", fake_compile)
    catalog = write(tmp_path, "catalog.jsonl", json.dumps({"id": "g", "type": "mlp_cluster"}) + "\n")
    path = write(tmp_path, "p.wsp", "host h\ngraft g\nheal steps 5\n")
    out_dir = str(tmp_path / "out")
    assert main(["compile", path, out_dir, "--catalog", catalog, "--align", "0", "--device", "cpu"]) == 0
    assert json.loads(capsys.readouterr().out) == {"host": "h"}
    assert calls[0][1:] == (out_dir, None, None, "cpu")
    assert calls[0][0]["grafts"][0]["id"] == "g"
    with open(os.path.join(out_dir, "program.wsp")) as f:
        assert parse(f.read()).to_dict() == calls[0][0]


# Regression tests from the review.


def test_local_path_host_round_trips():
    # Tests and local runs use a model folder on disk as the host.
    round_trip(Program(host="/tmp/pytest-of-me/test_x0/host"))
    round_trip(Program(host="./models/tiny").graft("g"))
    assert parse("host /models/tiny").host == "/models/tiny"
    with pytest.raises(LangError, match="must use only"):
        parse("host -bad")


def test_format_refuses_what_parse_would_read_differently():
    # restrict is a mask keyword, so it cannot be a mask id in text.
    with pytest.raises(ValueError, match="keyword"):
        format_program(Program(host=HOST, masks=use("restrict")))
    # With no kind, a first stitch key named like a kind would read as the kind.
    with pytest.raises(ValueError, match="also a stitch kind"):
        format_program(Program(host=HOST).graft("g", stitch={"mlp": 1}))
    round_trip(Program(host=HOST).graft("g", stitch={"kind": "lowrank", "mlp": 1}))


def test_huge_numbers_and_deep_masks_are_lang_errors():
    with pytest.raises(LangError, match="line 2: weight '1e400' is too large"):
        parse("host h\ntv t weight 1e400")
    with pytest.raises(LangError, match="line 2: .*nests too deep"):
        parse("host h\nmask " + "(" * 5000 + "a" + ")" * 5000)


def test_json_with_unknown_keys_is_an_error(tmp_path):
    # Program.from_dict drops keys it does not read, so wsp.lang checks them first.
    data = full_program().to_dict()
    data["probes"] = [{"id": "x"}]
    with pytest.raises(ValueError, match="'probes'"):
        format_program(data)
    data = full_program().to_dict()
    data["taps"] = [{"id": "x"}]
    with pytest.raises(ValueError, match="tap key 'id'"):
        format_program(data)
    data = full_program().to_dict()
    data["grafts"][0]["new_option"] = 1
    with pytest.raises(ValueError, match="'new_option'"):
        format_program(data)
    data = full_program().to_dict()
    data["heal"]["epochs"] = 2
    with pytest.raises(ValueError, match="'epochs'"):
        format_program(data)
    data = full_program().to_dict()
    data["masks"]["inner"]["extra"] = 1
    with pytest.raises(ValueError, match="'extra'"):
        format_program(data)
    with pytest.raises(ValueError, match="missing the key 'weight'"):
        format_program({"host": HOST, "task_vectors": [{"id": "t", "layers": None, "modules": None}]})
    # The base alias still loads.
    assert format_program({"base": HOST}) == format_program(Program(host=HOST))

    path = write(tmp_path, "p.json", json.dumps({"host": HOST, "probes": []}))
    assert main(["check", path]) == 1


def test_cli_compile_json_that_text_cannot_hold(tmp_path, monkeypatch, capsys):
    import wsp.compiler

    def fake_compile(program, catalog, output_dir, align_texts=None, heal_texts=None, device=None):
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, "compile_report.json"), "w") as f:
            json.dump({"host": program.host}, f)
        return output_dir

    monkeypatch.setattr(wsp.compiler, "compile_program", fake_compile)
    catalog = write(tmp_path, "catalog.jsonl", json.dumps({"id": "g", "type": "mlp_cluster"}) + "\n")
    out_dir = str(tmp_path / "out")
    os.makedirs(out_dir)
    # A program.wsp from an older compile must not stay next to the new program.json.
    write(tmp_path, "out/program.wsp", "host old\n")
    path = write(tmp_path, "p.json", json.dumps(Program(host=HOST).tv("t", layers=[]).to_dict()))
    assert main(["compile", path, out_dir, "--catalog", catalog, "--align", "0", "--device", "cpu"]) == 0
    captured = capsys.readouterr()
    assert json.loads(captured.out) == {"host": HOST}
    assert "warning: no program.wsp" in captured.err
    assert not os.path.exists(os.path.join(out_dir, "program.wsp"))
