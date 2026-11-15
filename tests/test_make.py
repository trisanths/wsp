"""Tests for wsp.make: winner choice, constraints, the report card, the dry-run CLI, and the request files."""

import glob
import json
import os

import pytest

from wsp import make
from wsp.program import Program

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

LLAMA1B = "unsloth/Llama-3.2-1B-Instruct"
QWEN05 = "Qwen/Qwen2.5-0.5B-Instruct"
QWEN7B = "Qwen/Qwen2.5-7B-Instruct"
QWEN14B = "Qwen/Qwen2.5-14B-Instruct"


def profile(key, hf_id, params_b, mmlu, medmcqa):
    # Scores as percents, the way round 10 reports them; make reads them as fractions.
    return {"key": key, "hf_id": hf_id, "family": key, "params_b": params_b, "active_params_b": params_b,
            "layers": 28, "hidden": 1024, "tokenizer_family": key, "scores": {"mmlu": mmlu, "medmcqa": medmcqa},
            "decode_tok_s": None, "source": "results/README.md round 10"}


PROFILES = [
    profile("llama1b", LLAMA1B, 1.24, 48.0, 41.0),
    profile("qwen05", QWEN05, 0.49, 48.3, 37.2),
    profile("qwen7b", QWEN7B, 7.6, 71.2, 56.7),
    profile("qwen14b", QWEN14B, 14.8, 78.6, 63.5),
]


def request(objective="max_accuracy", target=None, decode=None, total=None, minutes=None):
    return {"name": "test_request", "description": "a test",
            "task": {"kind": "mcq", "benchmark": "mmlu", "subjects": None},
            "constraints": {"max_decode_params_b": decode, "max_total_params_b": total,
                            "max_train_minutes": minutes, "gpu": "H200"},
            "objective": objective, "accuracy_target": target}


def candidate(program, accuracy, decode, prefill, minutes, rationale="test"):
    return {"program": program, "predicted": {"accuracy": accuracy, "decode_params_b": decode,
                                              "prefill_params_b": prefill, "train_minutes": minutes},
            "rationale": rationale, "planner": "rules"}


def tap_llama_qwen7b():
    program = Program(host=LLAMA1B).tap(QWEN7B, 20, 10).tap_train().to_dict()
    return candidate(program, 0.68, 1.24, 7.6, 2.0, "tap reads Qwen-7B at 70% depth")


def tap_qwen05_qwen14b():
    program = Program(host=QWEN05).tap(QWEN14B, 34, 9).tap_train().to_dict()
    return candidate(program, 0.74, 0.50, 14.8, 2.5)


def lora_llama():
    program = Program(host=LLAMA1B).lora(rank=16).heal(steps=500, data="mmlu_aux", lr=1e-4).to_dict()
    return candidate(program, 0.48, 1.25, 1.25, 1.0)


def bare(hf_id, accuracy, params):
    return candidate(Program(host=hf_id).to_dict(), accuracy, params, params, 0.0)


def baked(base):
    program = dict(base["program"])
    program["bake"] = {"rank": 16, "steps": 500}
    out = dict(base)
    out["program"] = program
    out["predicted"] = dict(base["predicted"])
    out["predicted"]["prefill_params_b"] = base["predicted"]["decode_params_b"]
    return out


def fake_planner(candidates):
    calls = []

    def plan(request, profiles, k=6):
        calls.append((request["name"], k))
        return candidates[:k]

    plan.calls = calls
    return plan


def fake_runner(scores, errors=None, train_seconds=None, skip=None):
    """run_candidates that returns a set accuracy per host + donor description."""
    errors = errors or {}
    train_seconds = train_seconds or {}
    skip = skip or []
    seen = []

    def run(candidates):
        out = []
        for index, item in enumerate(candidates):
            seen.append(item)
            if index in skip:
                continue
            result = {"candidate": item, "accuracy": scores.get(index), "decode_tok_s": 100.0 + index,
                      "train_seconds": train_seconds.get(index, 100.0), "model_dir": f"/vol/make/test/{index}",
                      "error": errors.get(index)}
            if index in errors:
                result["accuracy"] = None
            out.append(result)
        return out

    run.seen = seen
    return run


# Winner choice


def test_max_accuracy_picks_best_measured():
    candidates = [tap_llama_qwen7b(), tap_qwen05_qwen14b(), lora_llama(), bare(QWEN05, 0.48, 0.49)]
    run = fake_runner({0: 0.70, 1: 0.74, 2: 0.47, 3: 0.46})
    card = make.make(request(decode=1.5), PROFILES, run, plan_fn=fake_planner(candidates))
    assert card["winner"]["index"] == 1
    assert card["winner"]["accuracy"] == 0.74
    assert card["winner"]["target_met"] is None
    assert card["winner"]["model_dir"] == "/vol/make/test/1"
    assert len(run.seen) == 4


def test_max_accuracy_tie_goes_to_fewer_decode_params():
    candidates = [tap_llama_qwen7b(), tap_qwen05_qwen14b()]
    card = make.make(request(), PROFILES, fake_runner({0: 0.70, 1: 0.70}), plan_fn=fake_planner(candidates))
    assert card["winner"]["index"] == 1


def test_min_cost_picks_cheapest_reaching_target():
    candidates = [tap_llama_qwen7b(), tap_qwen05_qwen14b()]
    run = fake_runner({0: 0.70, 1: 0.62})
    card = make.make(request("min_cost_at_target", 0.60), PROFILES, run, plan_fn=fake_planner(candidates))
    assert card["winner"]["index"] == 1
    assert card["winner"]["target_met"] is True

    card = make.make(request("min_cost_at_target", 0.65), PROFILES, run, plan_fn=fake_planner(candidates))
    assert card["winner"]["index"] == 0
    assert card["winner"]["target_met"] is True


def test_min_cost_without_any_reaching_target_falls_back_to_best():
    candidates = [tap_llama_qwen7b(), tap_qwen05_qwen14b()]
    card = make.make(request("min_cost_at_target", 0.90), PROFILES, fake_runner({0: 0.70, 1: 0.62}),
                     plan_fn=fake_planner(candidates))
    assert card["winner"]["index"] == 0
    assert card["winner"]["target_met"] is False
    text = " ".join(card["notes"])
    assert "No eligible candidate reached the accuracy target of 90.0%" in text


def test_target_reported_under_max_accuracy():
    candidates = [tap_llama_qwen7b()]
    card = make.make(request("max_accuracy", 0.65), PROFILES, fake_runner({0: 0.70}),
                     plan_fn=fake_planner(candidates))
    assert card["winner"]["target_met"] is True
    assert "It meets the accuracy target of 65.0%." in card["notes"]


# Constraints


def test_decode_limit_excludes_candidates():
    candidates = [tap_llama_qwen7b(), tap_qwen05_qwen14b(), bare(QWEN14B, 0.78, 14.8)]
    card = make.make(request(decode=1.0), PROFILES, fake_runner({0: 0.70, 1: 0.60, 2: 0.79}),
                     plan_fn=fake_planner(candidates))
    assert card["winner"]["index"] == 1
    rows = card["candidates"]
    assert rows[0]["status"] == "over budget"
    assert "decode params 1.24B over the 1.0B limit" in rows[0]["violations"]
    assert rows[2]["status"] == "over budget"
    assert rows[1]["eligible"] is True


def test_total_limit_counts_donors_but_not_for_baked():
    candidates = [tap_llama_qwen7b(), tap_qwen05_qwen14b(), baked(tap_qwen05_qwen14b())]
    card = make.make(request(total=10.0), PROFILES, fake_runner({0: 0.66, 1: 0.74, 2: 0.70}),
                     plan_fn=fake_planner(candidates))
    rows = card["candidates"]
    assert rows[0]["total_params_b"] == pytest.approx(1.24 + 7.6)
    assert rows[1]["total_params_b"] == pytest.approx(0.49 + 14.8)
    assert rows[1]["status"] == "over budget"
    assert rows[2]["total_params_b"] == pytest.approx(0.50)
    assert card["winner"]["index"] == 2
    assert "baked into the host" in card["winner"]["description"]
    assert "# bake rank 16 steps 500" in card["winner"]["program_text"]


def test_measured_train_time_breaks_limit():
    candidates = [tap_llama_qwen7b(), tap_qwen05_qwen14b()]
    run = fake_runner({0: 0.66, 1: 0.74}, train_seconds={1: 900.0})
    card = make.make(request(minutes=5.0), PROFILES, run, plan_fn=fake_planner(candidates))
    assert card["winner"]["index"] == 0
    assert card["candidates"][1]["violations"] == ["training 15.0 min over the 5.0 min limit"]


def test_measured_decode_params_override_prediction():
    candidates = [tap_qwen05_qwen14b()]

    def run(items):
        return [{"candidate": items[0], "accuracy": 0.7, "decode_tok_s": 90.0, "train_seconds": 60.0,
                 "model_dir": "x", "error": None, "decode_params_b": 0.6}]

    card = make.make(request(decode=0.55), PROFILES, run, plan_fn=fake_planner(candidates))
    assert card["winner"] is None
    assert card["candidates"][0]["decode_params_b"] == 0.6


# Failures


def test_failed_and_missing_candidates():
    candidates = [tap_llama_qwen7b(), tap_qwen05_qwen14b(), lora_llama()]
    run = fake_runner({0: 0.66, 1: 0.74, 2: 0.47}, errors={1: "CUDA out of memory"}, skip=[2])
    card = make.make(request(), PROFILES, run, plan_fn=fake_planner(candidates))
    rows = card["candidates"]
    assert rows[1]["status"] == "failed"
    assert rows[1]["measured"]["error"] == "CUDA out of memory"
    assert rows[2]["status"] == "not run"
    assert card["winner"]["index"] == 0
    assert any("1 candidate(s) failed" in note and "CUDA out of memory" in note for note in card["notes"])


def test_all_failed_gives_no_winner():
    candidates = [tap_llama_qwen7b(), tap_qwen05_qwen14b()]
    run = fake_runner({}, errors={0: "boom", 1: "boom"})
    card = make.make(request(), PROFILES, run, plan_fn=fake_planner(candidates))
    assert card["winner"] is None
    assert card["notes"][0].startswith("No candidate ran")
    assert "No winner." in make.format_report(card)


def test_result_without_accuracy_is_a_failure():
    candidates = [tap_llama_qwen7b()]

    def run(items):
        return [{"candidate": items[0], "accuracy": None, "decode_tok_s": None, "train_seconds": None,
                 "model_dir": None, "error": None}]

    card = make.make(request(), PROFILES, run, plan_fn=fake_planner(candidates))
    assert card["candidates"][0]["status"] == "failed"
    assert card["candidates"][0]["measured"]["error"] == "no accuracy returned"


def test_empty_plan():
    run = fake_runner({})
    card = make.make(request(), PROFILES, run, plan_fn=fake_planner([]))
    assert card["winner"] is None
    assert card["candidates"] == []
    assert run.seen == []


# Report card


def test_report_card_fields_and_baselines():
    candidates = [tap_qwen05_qwen14b(), bare(QWEN05, 0.48, 0.49), lora_llama(), tap_llama_qwen7b()]
    run = fake_runner({0: 0.745, 1: 0.452, 2: 0.47, 3: 0.68})
    card = make.make(request(decode=1.5, total=16.0), PROFILES, run, plan_fn=fake_planner(candidates))
    winner = card["winner"]
    assert winner["index"] == 0
    assert winner["program_text"].startswith("host Qwen/Qwen2.5-0.5B-Instruct\n")
    assert "tap Qwen/Qwen2.5-14B-Instruct layer 34 at 9" in winner["program_text"]
    assert winner["decode_params_b"] == 0.50
    assert winner["prefill_params_b"] == 14.8
    assert winner["decode_tok_s"] == 100.0
    assert winner["train_seconds"] == 100.0

    base = card["baselines"]
    # The host baseline is measured, since candidate 1 is Qwen-0.5B alone.
    assert base["host"]["hf_id"] == QWEN05
    assert base["host"]["measured_accuracy"] == 0.452
    assert base["host"]["profile_accuracy"] == pytest.approx(0.483)
    assert base["best_source"]["hf_id"] == QWEN14B
    assert base["best_source"]["measured_accuracy"] is None
    assert base["best_source_at_decode_budget"]["hf_id"] == QWEN05
    # The LoRA candidate is on Llama-1B, not the winner's host.
    assert base["host_lora"] is None

    rows = card["candidates"]
    assert rows[0]["predicted"]["accuracy"] == 0.74
    assert rows[0]["measured"]["accuracy"] == 0.745
    assert rows[0]["prediction_error"] == pytest.approx(0.005)

    notes = " ".join(card["notes"])
    assert "It beats its host alone, Qwen2.5-0.5B-Instruct (45.2%), by 29.3 points." in notes
    assert "It trails the best single model in the total budget, Qwen2.5-14B-Instruct (78.6%, from its profile)" \
        in notes
    text = make.format_report(card)
    assert "Winner: Qwen2.5-0.5B-Instruct + tap from Qwen2.5-14B-Instruct" in text
    assert "74.0% -> 74.5%" in text
    json.dumps(card)


def test_host_lora_baseline():
    candidates = [tap_llama_qwen7b(), lora_llama()]
    card = make.make(request(), PROFILES, fake_runner({0: 0.68, 1: 0.477}), plan_fn=fake_planner(candidates))
    assert card["baselines"]["host_lora"]["measured_accuracy"] == 0.477
    assert "Against a LoRA on the same host (47.7%), the winner is +20.3 points." in card["notes"]


def test_planner_gets_k_and_use_llm_needs_support():
    plan = fake_planner([tap_llama_qwen7b(), tap_qwen05_qwen14b()])
    card = make.make(request(), PROFILES, fake_runner({0: 0.6, 1: 0.6}), k=1, plan_fn=plan)
    assert plan.calls == [("test_request", 1)]
    assert len(card["candidates"]) == 1
    with pytest.raises(ValueError):
        make.make(request(), PROFILES, fake_runner({}), use_llm=True, plan_fn=plan)

    def llm_plan(request, profiles, k=6, use_llm=False):
        assert use_llm is True
        return []

    make.make(request(), PROFILES, fake_runner({}), use_llm=True, plan_fn=llm_plan)


def test_profiles_as_dict_and_bad_request():
    table = {}
    for row in PROFILES:
        table[row["key"]] = row
    card = make.make(request(), table, fake_runner({0: 0.7}), plan_fn=fake_planner([tap_llama_qwen7b()]))
    assert card["winner"]["index"] == 0
    bad = request("min_cost_at_target", None)
    with pytest.raises(ValueError):
        make.make(bad, PROFILES, fake_runner({}), plan_fn=fake_planner([]))


# Command line


def write(path, data):
    with open(path, "w") as f:
        json.dump(data, f)
    return str(path)


def test_dry_run_cli(tmp_path, capsys):
    request_path = write(tmp_path / "req.json", request(decode=1.0))
    profiles_path = write(tmp_path / "profiles.json", PROFILES)
    plan = fake_planner([tap_qwen05_qwen14b(), tap_llama_qwen7b()])
    code = make.main([request_path, "--dry-run", "--profiles", profiles_path, "--save-plan",
                      str(tmp_path / "plan.json")], plan_fn=plan)
    assert code == 0
    out = capsys.readouterr().out
    assert "Plan for test_request" in out
    assert "[0] Qwen2.5-0.5B-Instruct + tap from Qwen2.5-14B-Instruct (rules planner)" in out
    assert "| tap Qwen/Qwen2.5-14B-Instruct layer 34 at 9" in out
    assert "breaks decode params 1.24B over the 1.0B limit" in out
    with open(tmp_path / "plan.json") as f:
        saved = json.load(f)
    assert len(saved["candidates"]) == 2


def test_cli_without_dry_run_or_results(tmp_path):
    request_path = write(tmp_path / "req.json", request())
    profiles_path = write(tmp_path / "profiles.json", PROFILES)
    code = make.main([request_path, "--profiles", profiles_path], plan_fn=fake_planner([]))
    assert code == 2


def test_cli_bad_request(tmp_path):
    request_path = write(tmp_path / "req.json", {"name": "x"})
    assert make.main([request_path, "--dry-run"], plan_fn=fake_planner([])) == 1


def test_cli_results_builds_card(tmp_path, capsys):
    request_path = write(tmp_path / "req.json", request())
    profiles_path = write(tmp_path / "profiles.json", PROFILES)
    first = tap_llama_qwen7b()
    second = tap_qwen05_qwen14b()
    rows = [
        {"request": "test_request", "index": 1, "candidate": second, "accuracy": 0.74, "decode_tok_s": 80.0,
         "train_seconds": 120.0, "model_dir": "b", "error": None},
        {"request": "test_request", "index": 0, "candidate": first, "accuracy": 0.69, "decode_tok_s": 90.0,
         "train_seconds": 110.0, "model_dir": "a", "error": None},
        {"request": "other", "index": 0, "candidate": first, "accuracy": 0.99, "decode_tok_s": 90.0,
         "train_seconds": 110.0, "model_dir": "c", "error": None},
    ]
    results_path = write(tmp_path / "rows.json", rows)
    out_path = str(tmp_path / "card.json")
    code = make.main([request_path, "--profiles", profiles_path, "--results", results_path, "--out", out_path],
                     plan_fn=fake_planner([]))
    assert code == 0
    with open(out_path) as f:
        card = json.load(f)
    assert len(card["candidates"]) == 2
    assert card["candidates"][0]["measured"]["model_dir"] == "a"
    assert card["winner"]["model_dir"] == "b"
    assert "Report card for test_request" in capsys.readouterr().out


# Request files


def request_files():
    return sorted(glob.glob(os.path.join(REPO, "requests", "*.json")))


def test_request_files_validate():
    paths = request_files()
    assert len(paths) >= 8
    benchmarks = set()
    for path in paths:
        with open(path) as f:
            data = json.load(f)
        assert make.request_problems(data) == [], path
        assert data["name"] == os.path.basename(path)[:-len(".json")]
        assert isinstance(data["description"], str) and len(data["description"]) > 10
        assert "\n" not in data["description"]
        benchmarks.add(data["task"]["benchmark"])
    for needed in ("mmlu", "medmcqa", "medqa", "gsm8k"):
        assert needed in benchmarks


def test_request_files_cover_the_brief():
    table = {}
    for path in request_files():
        with open(path) as f:
            data = json.load(f)
        table[data["name"]] = data
    assert table["gsm8k_1b_40"]["objective"] == "min_cost_at_target"
    assert table["gsm8k_1b_40"]["accuracy_target"] == 0.40
    assert table["gsm8k_1b_40"]["task"]["kind"] == "generate"
    assert table["mmlu_05b"]["constraints"]["max_decode_params_b"] == 0.5
    assert table["mmlu_16b_total"]["constraints"]["max_total_params_b"] == 16.0
    assert "professional_law" in table["mmlu_law_1b"]["task"]["subjects"]
    assert "college_physics" in table["mmlu_stem_2b"]["task"]["subjects"]
    assert table["medqa_8b"]["constraints"]["max_decode_params_b"] == 8.0


def test_request_problems_catch_mistakes():
    good = request()
    assert make.request_problems(good) == []
    bad = json.loads(json.dumps(good))
    bad["task"]["kind"] = "chat"
    bad["constraints"]["gpu"] = "A100"
    bad["constraints"]["max_decode_params_b"] = -1
    bad["accuracy_target"] = 40
    bad["extra"] = 1
    problems = make.request_problems(bad)
    assert len(problems) == 5
    bad = json.loads(json.dumps(good))
    bad["task"]["subjects"] = []
    assert make.request_problems(bad) == ["task.subjects must be null or a non-empty list of strings"]


def test_as_fraction():
    assert make.as_fraction(48.3) == pytest.approx(0.483)
    assert make.as_fraction(0.483) == 0.483
    assert make.as_fraction(None) is None


# modal_make helpers (CPU only)


def test_modal_make_helpers():
    pytest.importorskip("modal")
    import modal_make

    assert modal_make.final_number("so 3 + 4 = 7. The answer is 1,250.") == 1250.0
    assert modal_make.final_number("work 12 then #### 18\nQuestion: next 99") == 18.0
    assert modal_make.final_number("it is 5 or 6") == 6.0
    assert modal_make.final_number("no digits") is None
    assert modal_make.gsm8k_gold("steps\n#### 1,000") == 1000.0

    lora_math = Program(host=LLAMA1B).lora(rank=64).heal(data="gsm8k_train", lr=5e-5).to_dict()
    assert modal_make.gsm8k_style(lora_math) == "raw"
    assert modal_make.gsm8k_style(Program(host=LLAMA1B).to_dict()) == "chat"
    tap_math = Program(host=LLAMA1B).tap(QWEN7B, 20, 10).tap_train(data="gsm8k_train").to_dict()
    assert modal_make.gsm8k_style(tap_math) == "raw"

    med = {"task": {"kind": "mcq", "benchmark": "medmcqa", "subjects": None}}
    assert modal_make.eval_spec(med) == ("validation", 7, 1000, 1500)
    assert modal_make.eval_spec(med, 200) == ("validation", 7, 1000, 200)

    plans = [({"name": "a"}, [1, 2, 3]), ({"name": "b"}, [4, 5])]
    shard0 = modal_make.cell_list(plans, 0, 2)
    shard1 = modal_make.cell_list(plans, 1, 2)
    assert len(shard0) + len(shard1) == 5
    assert shard0[0] == ({"name": "a"}, 1, 0)
    assert shard1[-1] == ({"name": "b"}, 4, 0)


def test_bake_marker_on_the_candidate():
    base = tap_qwen05_qwen14b()
    marked = dict(base)
    marked["bake"] = {"rank": 16, "steps": 500, "data": "mmlu_aux"}
    marked["predicted"] = dict(base["predicted"])
    seen = []

    def run(items):
        seen.extend(items)
        # The runner answers with the candidate as the planner gave it.
        return [{"candidate": marked, "accuracy": 0.7, "decode_tok_s": 90.0, "train_seconds": 60.0,
                 "model_dir": "baked", "error": None}]

    card = make.make(request(total=10.0), PROFILES, run, plan_fn=fake_planner([marked]))
    assert seen[0]["program"]["bake"]["data"] == "mmlu_aux"
    assert card["winner"]["model_dir"] == "baked"
    assert card["winner"]["total_params_b"] == pytest.approx(0.50)
    assert "# bake data mmlu_aux rank 16 steps 500" in card["winner"]["program_text"]


def test_predicted_can_be_missing():
    item = tap_llama_qwen7b()
    item["predicted"] = None
    card = make.make(request(), PROFILES, fake_runner({0: 0.6}), plan_fn=fake_planner([item]))
    assert card["winner"]["index"] == 0
    assert "prediction_error" not in card["candidates"][0]
    make.format_report(card)
    make.format_plan(request(), PROFILES, [item])


def test_baseline_candidates_add_each_host_alone():
    candidates = [tap_llama_qwen7b(), tap_qwen05_qwen14b(), bare(QWEN05, 0.48, 0.49)]
    extra = make.baseline_candidates(request(), candidates, PROFILES)
    assert len(extra) == 1
    assert extra[0]["program"]["host"] == LLAMA1B
    assert make.is_bare(extra[0]["program"])
    assert extra[0]["predicted"]["accuracy"] == pytest.approx(0.48)
    assert extra[0]["planner"] == "baseline"
    run = fake_runner({0: 0.68, 1: 0.7, 2: 0.45, 3: 0.44})
    card = make.make(request(), PROFILES, run, plan_fn=fake_planner(candidates[:2]), add_baselines=True)
    assert len(card["candidates"]) == 4
    assert card["baselines"]["host"]["measured_accuracy"] == 0.44
