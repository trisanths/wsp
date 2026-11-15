"""The sapien command line."""

import io
import json
import os
import re
import subprocess
import sys

import pytest
import torch

transformers = pytest.importorskip("transformers")

import wsp.skills_eval
from wsp.cli import CliError, estimate_program, main, run_check
from wsp.lang import format_program, parse
from tests.test_unified import HEAL, llama, qwen, save, zoo  # noqa: F401  (zoo is a fixture)

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def write(tmp_path, name, text):
    path = tmp_path / name
    path.write_text(text)
    return str(path)


def read_report(out):
    with open(os.path.join(out, "compile_report.json")) as f:
        return json.load(f)


def test_check_prints_sizes_and_budget(zoo, capsys):
    path = write(zoo["tmp"], "p.sap", f"host {zoo['host']}\nexperts {zoo['a']},{zoo['b']} top_k 2\n"
                                       "budget total 1b active 1b\n")
    assert main(["check", path]) == 0
    out = capsys.readouterr().out
    assert "params_total" in out and "params_active" in out
    assert "budget max_total_params_b 1: ok" in out
    assert out.strip().endswith("ok")


def test_check_estimate_matches_the_compiled_model_for_experts(zoo):
    program = parse(f"host {zoo['host']}\nexperts {zoo['a']},{zoo['b']} top_k 2\nlora rank 2\nheal steps 1\n")
    estimate = estimate_program(program)
    program.compile(zoo["out"], None, heal_texts=["the cat sat"], device="cpu")
    report = read_report(zoo["out"])
    assert estimate["params_total"] == report["params_total"]
    assert estimate["params_active"] == report["params_active"]


def test_check_estimate_matches_the_compiled_model_for_stitched_experts_and_inserts(zoo):
    program = parse(f"host {zoo['host']}\nexperts {zoo['qwen']} top_k 1 stitch linear\n")
    estimate = estimate_program(program)
    program.compile(zoo["out"], None, device="cpu")
    report = read_report(zoo["out"])
    assert estimate["params_total"] == report["params_total"]
    assert estimate["params_active"] == report["params_active"]

    program = parse(f"host {zoo['host']}\ninsert {zoo['qwen']} layers 0..1 at 2\n")
    estimate = estimate_program(program)
    program.compile(zoo["out"] + "2", None, device="cpu")
    report = read_report(zoo["out"] + "2")
    assert estimate["params_total"] == report["params_total"]
    assert estimate["params_active"] == report["params_active"]


def test_check_estimate_of_a_plain_host_and_a_written_program(zoo):
    program = parse(f"host {zoo['host']}\n")
    host = transformers.AutoModelForCausalLM.from_pretrained(zoo["host"])
    total = sum(p.numel() for p in host.parameters())
    estimate = estimate_program(program)
    assert estimate["params_total"] == total
    assert estimate["params_active"] == total - 300 * 64

    written = parse(f"from {zoo['host']}\ncarve keep 2 select uniform\n")
    estimate = estimate_program(written)
    layer = sum(p.numel() for p in host.model.layers[0].parameters())
    assert estimate["params_total"] == total - 2 * layer
    assert estimate["layers_kept"] == 2


def test_check_fails_on_an_estimate_over_the_budget(zoo, capsys):
    path = write(zoo["tmp"], "p.sap", f"host {zoo['host']}\nexperts {zoo['a']}\nbudget total 0.0001b\n")
    assert main(["check", path]) == 1
    out = capsys.readouterr().out
    assert "OVER" in out and "estimated max_total_params_b is over the limit" in out


def test_check_json_output(zoo, capsys):
    path = write(zoo["tmp"], "p.sap", f"host {zoo['host']}\nbudget active 1b\n")
    assert main(["check", path, "--json"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["ok"] is True and data["params_total"] > 0
    assert data["budget"][0]["limit"] == "max_active_params_b" and data["budget"][0]["ok"] is True


def test_check_reports_a_parse_error_with_its_line(zoo, capsys):
    path = write(zoo["tmp"], "bad.sap", f"host {zoo['host']}\ninsert d at 2\n")
    assert main(["check", path]) == 1
    assert "line 2" in capsys.readouterr().err
    assert main(["check", str(zoo["tmp"] / "missing.sap")]) == 1


def test_check_reports_a_bad_mix(zoo, capsys):
    path = write(zoo["tmp"], "mix.sap", f"host {zoo['host']}\nexperts {zoo['a']}\ninsert {zoo['qwen']} layers 0..1 at 2\n")
    assert main(["check", path]) == 1
    assert "not a mix" in capsys.readouterr().out


def test_check_with_an_unreadable_donor_warns_and_does_not_guess(zoo, capsys):
    path = write(zoo["tmp"], "p.sap", f"host {zoo['host']}\nexperts {zoo['tmp']}/nowhere\nbudget total 1b\n")
    code = main(["check", path])
    out = capsys.readouterr().out
    assert code == 0
    assert "params_total  unknown" in out and "leaves out" in out and "not checked" in out


def test_taps_are_refused_without_allow_parallel(zoo, capsys):
    path = write(zoo["tmp"], "tap.sap", f"host {zoo['host']}\ntap {zoo['qwen']} layer 1 at 2\n"
                                         "tap_train steps 1\n")
    assert main(["check", path]) == 1
    out = capsys.readouterr().out
    assert "not Sapien" in out and "--allow-parallel" in out
    assert main(["check", path, "--allow-parallel"]) == 0
    assert "not Sapien" in capsys.readouterr().out
    assert main(["compile", path, "-o", str(zoo["tmp"] / "o"), "--device", "cpu"]) == 1
    assert "not Sapien" in capsys.readouterr().err


def test_compile_builds_the_directory_and_prints_a_summary(zoo, capsys):
    path = write(zoo["tmp"], "p.sap", f"host {zoo['host']}\nexperts {zoo['a']} top_k 2\nbudget total 1b\n")
    assert main(["compile", path, "-o", zoo["out"], "--device", "cpu"]) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["params_total"] > 0 and summary["budget_ok"] is True and summary["heal_seconds"] == 0.0
    model, _ = wsp.load(zoo["out"], dtype=torch.float32)
    assert type(model).__name__ == "LlamaForCausalLM"
    assert os.path.exists(os.path.join(zoo["out"], "program.wsp"))


def test_compile_control_random_writes_the_control(zoo, capsys):
    path = write(zoo["tmp"], "p.sap", f"host {zoo['host']}\nexperts {zoo['a']}\n")
    assert main(["compile", path, "-o", zoo["out"], "--device", "cpu", "--control", "random",
                 "--control-seed", "3"]) == 0
    capsys.readouterr()
    program = json.load(open(os.path.join(zoo["out"], "program.json")))
    assert program["experts"]["control"] == "random" and program["experts"]["control_seed"] == 3
    manifest = json.load(open(os.path.join(zoo["out"], "manifest.json")))
    assert manifest["experts"][0]["layer_map"] != [0, 1, 2, 3]


def test_compile_errors_print_and_exit_1(zoo, capsys):
    path = write(zoo["tmp"], "p.sap", f"host {zoo['host']}\nexperts {zoo['a']}\nbudget total 0.0001b\n")
    assert main(["compile", path, "-o", zoo["out"], "--device", "cpu"]) == 1
    assert "over its budget" in capsys.readouterr().err
    needs = write(zoo["tmp"], "g.sap", f"host {zoo['host']}\ngraft g1\n")
    assert main(["compile", needs, "-o", zoo["out"], "--device", "cpu"]) == 1
    assert "needs --catalog" in capsys.readouterr().err
    taps_and_control = write(zoo["tmp"], "c.sap", f"host {zoo['host']}\nmask m\n")
    assert main(["compile", taps_and_control, "-o", zoo["out"], "--control", "random", "--catalog", "x"]) == 1
    assert "no random control" in capsys.readouterr().err


def test_score_runs_the_suite_and_writes_json(zoo, monkeypatch, capsys):
    from wsp.program import Program

    Program(host=zoo["host"]).compile(zoo["out"], None, device="cpu")
    seen = {}

    def fake_suite(model, tokenizer, skills=None, items=None, batch_size=32, limit=None, **kwargs):
        seen["skills"] = skills
        seen["limit"] = limit
        scores = {}
        for skill in skills:
            scores[skill] = 0.5
        return {"scores": scores, "suite": 0.5, "skills": {skill: {"score": 0.5, "correct": 1, "n": 2,
                                                                    "seconds": 0.1, "records": []}
                                                           for skill in skills}}

    monkeypatch.setattr(wsp.skills_eval, "evaluate_suite", fake_suite)
    result_path = str(zoo["tmp"] / "scores.json")
    assert main(["score", zoo["out"], "--skills", "gsm8k,mmlu", "--limit", "4", "--device", "cpu",
                 "--json", result_path]) == 0
    row = json.loads(capsys.readouterr().out)
    assert seen == {"skills": ["gsm8k", "mmlu"], "limit": 4}
    assert row["scores"] == {"gsm8k": 0.5, "mmlu": 0.5} and row["params_total"] > row["params_run_per_token"]
    assert json.load(open(result_path))["summary"]["suite"] == 0.5
    assert main(["score", zoo["out"], "--skills", "gsm8k,chess"]) == 1
    assert "unknown skill 'chess'" in capsys.readouterr().err
    assert main(["score", str(zoo["tmp"] / "none"), "--skills", "gsm8k"]) == 1


def test_library_query_wraps_the_library_script(capsys):
    assert main(["library", "query", "--kind", "model", "--json"]) == 0
    lines = [line for line in capsys.readouterr().out.splitlines() if line.strip() != ""]
    assert len(lines) > 0
    assert json.loads(lines[0])["kind"] == "model"


def write_atlas_entry(folder, model, skills, params_b):
    os.makedirs(folder, exist_ok=True)
    name = model.replace("/", "__") + ".json"
    with open(os.path.join(folder, name), "w") as f:
        json.dump({"model": model, "skills": skills, "params_b": params_b}, f)


def test_atlas_best_ranks_models_under_a_size_limit(tmp_path, capsys):
    folder = str(tmp_path / "entries")
    write_atlas_entry(folder, "org/small", {"math": 0.4}, 1.5)
    write_atlas_entry(folder, "org/big", {"math": 0.9}, 7.0)
    write_atlas_entry(folder, "org/mid", {"math": 0.6, "code": 0.1}, 1.9)
    write_atlas_entry(folder, "org/other", {"code": 0.7}, 1.0)
    assert main(["atlas", "best", "--skill", "math", "--max-params-b", "2", "--entries", folder]) == 0
    lines = capsys.readouterr().out.strip().splitlines()
    assert [line.split()[0] for line in lines[1:]] == ["org/mid", "org/small"]
    assert main(["atlas", "best", "--skill", "math", "--entries", folder, "--json", "--top", "1"]) == 0
    assert json.loads(capsys.readouterr().out)["model"] == "org/big"


def test_atlas_best_without_entries_says_so(tmp_path, capsys):
    assert main(["atlas", "best", "--skill", "math", "--entries", str(tmp_path / "none")]) == 1
    assert "no atlas entries" in capsys.readouterr().err
    folder = str(tmp_path / "entries")
    write_atlas_entry(folder, "org/small", {"math": 0.4}, 1.5)
    assert main(["atlas", "best", "--skill", "chess", "--entries", folder]) == 1
    assert "no atlas entry has the skill" in capsys.readouterr().err


def test_python_m_wsp_cli_runs(tmp_path):
    python = os.path.join(REPO, ".venv", "bin", "python")
    if not os.path.exists(python):
        python = sys.executable
    result = subprocess.run([python, "-m", "wsp.cli", "--help"], cwd=REPO, capture_output=True, text=True,
                            timeout=120)
    assert result.returncode == 0, result.stderr
    for word in ["check", "compile", "score", "library", "atlas"]:
        assert word in result.stdout


def test_pyproject_has_the_sapien_script():
    text = open(os.path.join(REPO, "pyproject.toml")).read()
    assert re.search(r'\[project\.scripts\]\s+sapien = "wsp\.cli:main"', text)


def test_quickstart_programs_parse_and_pass_the_checks():
    from wsp.compiler import check_program

    with open(os.path.join(REPO, "docs", "quickstart.md")) as f:
        text = f.read()
    blocks = re.findall(r"```sap\n(.*?)```", text, re.DOTALL)
    assert len(blocks) >= 3
    for block in blocks:
        program = parse(block)
        check_program(program)
        assert parse(format_program(program)).to_dict() == program.to_dict()
        assert not program.has_taps()


def test_readme_mentions_what_is_not_proven():
    text = open(os.path.join(REPO, "README.md")).read()
    assert "Not proven" in text and "sapien check" in text
