"""Tests for wsp.planner and wsp.rules. CPU only, no network: the LLM call is mocked."""

import json

import pytest

from wsp import planner, rules
from wsp.lang import parse
from wsp.program import Program


def profile(key, hf_id, params_b, layers, hidden, mmlu=None, medmcqa=None, gsm8k=None, family="x"):
    scores = {}
    if mmlu is not None:
        scores["mmlu"] = mmlu
    if medmcqa is not None:
        scores["medmcqa"] = medmcqa
    if gsm8k is not None:
        scores["gsm8k"] = gsm8k
    return {"key": key, "hf_id": hf_id, "family": family, "params_b": params_b, "active_params_b": params_b,
            "layers": layers, "hidden": hidden, "tokenizer_family": family, "scores": scores,
            "decode_tok_s": None, "source": "test"}


def profiles():
    # Round 10 held-out MMLU and MedMCQA scores. The GSM8K scores are fakes for the tests.
    return [
        profile("llama1b", "unsloth/Llama-3.2-1B-Instruct", 1.24, 16, 2048, 44.5, 41.0, 25.0, "llama"),
        profile("qwen05", "Qwen/Qwen2.5-0.5B-Instruct", 0.49, 24, 896, 45.4, 37.2, 33.5, "qwen"),
        profile("qwen7b", "Qwen/Qwen2.5-7B-Instruct", 7.6, 28, 3584, 70.9, 56.7, 80.0, "qwen"),
        profile("qwen14b", "Qwen/Qwen2.5-14B-Instruct", 14.7, 48, 5120, 77.9, 63.5, 85.0, "qwen"),
        profile("qwen32b", "Qwen/Qwen2.5-32B-Instruct", 32.8, 64, 5120, 79.3, 67.2, 88.0, "qwen"),
        profile("gemma9b", "unsloth/gemma-2-9b-it", 9.2, 42, 3584, 72.2, 56.4, 75.0, "gemma"),
        profile("llama8b", "unsloth/Meta-Llama-3.1-8B-Instruct", 8.0, 32, 4096, 65.6, 60.1, 70.0, "llama"),
        profile("phi35", "microsoft/Phi-3.5-mini-instruct", 3.8, 32, 3072, 69.0, 51.7, 70.0, "phi"),
        profile("biomistral", "BioMistral/BioMistral-7B", 7.2, 32, 4096, 51.6, 41.7, None, "mistral"),
    ]


def request(benchmark="mmlu", kind="mcq", max_decode=None, max_total=None, max_minutes=None, gpu="H100",
            objective="max_accuracy", target=None):
    return {"name": "t", "task": {"kind": kind, "benchmark": benchmark, "subjects": None},
            "constraints": {"max_decode_params_b": max_decode, "max_total_params_b": max_total,
                            "max_train_minutes": max_minutes, "gpu": gpu},
            "objective": objective, "accuracy_target": target}


def by_key():
    out = {}
    for row in profiles():
        out[row["hf_id"]] = row
    return out


def test_hosts_filtered_by_decode_limit():
    details = planner.plan_details(request(max_decode=1.5), profiles(), k=100)
    assert len(details["candidates"]) > 0
    for candidate in details["candidates"]:
        assert candidate["program"]["host"] in ("unsloth/Llama-3.2-1B-Instruct", "Qwen/Qwen2.5-0.5B-Instruct")
        assert candidate["predicted"]["decode_params_b"] <= 1.5


def test_programs_are_valid_and_round_trip_through_text():
    for bench in ("mmlu", "gsm8k", "medmcqa"):
        for candidate in planner.plan(request(benchmark=bench, max_decode=2.0, gpu="H200"), profiles(), k=20):
            program = Program.from_dict(candidate["program"])
            assert program.to_dict() == candidate["program"]
            assert parse(candidate["text"]).to_dict() == candidate["program"]
            assert candidate["planner"] == "rules"
            assert isinstance(candidate["rationale"], str) and len(candidate["rationale"]) > 40
            for field in ("accuracy", "decode_params_b", "prefill_params_b", "train_minutes"):
                assert field in candidate["predicted"]


def test_medical_request_picks_small_host_and_strong_medical_donor():
    result = planner.plan(request(benchmark="medmcqa", max_decode=1.5, gpu="H200"), profiles())
    top = result[0]
    assert top["program"]["host"] in ("unsloth/Llama-3.2-1B-Instruct", "Qwen/Qwen2.5-0.5B-Instruct")
    assert len(top["program"]["taps"]) >= 1
    medical = by_key()
    for tap in top["program"]["taps"]:
        assert medical[tap["donor"]]["scores"]["medmcqa"] >= 60
    # The tap reads about 70% depth and writes about 40% depth.
    donor = medical[top["program"]["taps"][0]["donor"]]
    assert top["program"]["taps"][0]["donor_layer"] == rules.read_layer(donor)
    assert top["predicted"]["accuracy"] > 55
    # medmcqa_train is not in wsp.pairs yet, so the plan says so.
    joined = " ".join(result.notes)
    assert "medmcqa_train" in joined


def test_medical_request_on_h100_drops_donors_that_do_not_fit():
    details = planner.plan_details(request(benchmark="medmcqa", max_decode=1.5, gpu="H100"), profiles(), k=100)
    for candidate in details["candidates"]:
        for tap in candidate["program"].get("taps", []):
            assert tap["donor"] != "Qwen/Qwen2.5-32B-Instruct"
    reasons = []
    for candidate in details["dropped"]:
        reasons += candidate["problems"]
    assert any("H100" in reason for reason in reasons)


def test_gsm8k_request_does_not_lead_with_taps():
    result = planner.plan(request(benchmark="gsm8k", kind="generate", max_decode=1.5), profiles())
    top = result[0]
    assert "taps" not in top["program"]
    assert top["kind"] == "lora"
    assert top["program"]["lora"]["rank"] == 64
    assert top["program"]["heal"]["data"] == "gsm8k_train"
    # Every tap candidate predicts no gain over its host.
    host_scores = {"unsloth/Llama-3.2-1B-Instruct": 25.0, "Qwen/Qwen2.5-0.5B-Instruct": 33.5}
    for candidate in planner.plan(request(benchmark="gsm8k", kind="generate", max_decode=1.5), profiles(), k=50):
        if candidate["kind"] in ("tap", "two_taps"):
            assert candidate["predicted"]["accuracy"] <= host_scores[candidate["program"]["host"]]


def test_min_cost_at_target_picks_smallest_feasible():
    req = request(objective="min_cost_at_target", target=70.0)
    details = planner.plan_details(req, profiles(), k=200)
    candidates = details["candidates"]
    top = candidates[0]
    assert top["meets_target"]
    smallest = None
    for candidate in candidates:
        if candidate["predicted"]["accuracy"] >= 70.0:
            value = candidate["predicted"]["decode_params_b"]
            if smallest is None or value < smallest:
                smallest = value
    assert top["predicted"]["decode_params_b"] == smallest
    # Qwen-0.5B with a strong tap beats 70 at 0.5B decode cost.
    assert top["program"]["host"] == "Qwen/Qwen2.5-0.5B-Instruct"
    # Meeting candidates come before the rest.
    seen_miss = False
    for candidate in candidates:
        if not candidate["meets_target"]:
            seen_miss = True
        else:
            assert not seen_miss


def test_max_accuracy_sorts_by_accuracy():
    result = planner.plan(request(max_decode=1.5), profiles(), k=20)
    values = []
    for candidate in result:
        values.append(candidate["predicted"]["accuracy"])
    assert values == sorted(values, reverse=True)


def test_predicted_costs_add_up():
    table = by_key()
    for candidate in planner.plan(request(max_decode=1.5), profiles(), k=50):
        program = candidate["program"]
        host = table[program["host"]]
        expected_decode = host["active_params_b"]
        expected_prefill = host["active_params_b"]
        deepest = {}
        for tap in program.get("taps", []):
            donor = table[tap["donor"]]
            expected_decode += rules.tap_branch_params(host, donor, tap["attn_dim"]) / 1e9
            if tap["donor"] not in deepest or tap["donor_layer"] > deepest[tap["donor"]]:
                deepest[tap["donor"]] = tap["donor_layer"]
        for donor_id in deepest:
            expected_prefill += rules.donor_prefix_params_b(table[donor_id], deepest[donor_id])
        if program.get("lora") is not None and len(program.get("taps", [])) > 0:
            expected_decode += rules.lora_params(host, program["lora"]["rank"]) / 1e9
        assert candidate["predicted"]["decode_params_b"] == pytest.approx(expected_decode, abs=1e-3)
        assert candidate["predicted"]["prefill_params_b"] == pytest.approx(expected_prefill, abs=1e-3)
        if candidate["kind"] == "host":
            assert candidate["predicted"]["train_minutes"] == 0


def test_branch_params_match_run6():
    table = by_key()
    count = rules.tap_branch_params(table["unsloth/Llama-3.2-1B-Instruct"], table["Qwen/Qwen2.5-7B-Instruct"])
    # Run 6: 2.9M trainable parameters per tap.
    assert 2.8e6 < count < 3.0e6


def test_train_minutes_limit_drops_candidates():
    details = planner.plan_details(request(max_decode=1.5, max_minutes=1.0), profiles(), k=100)
    for candidate in details["candidates"]:
        assert candidate["predicted"]["train_minutes"] <= 1.0
    assert len(details["dropped"]) > 0


def test_bake_when_donors_are_forbidden_at_runtime():
    req = request(max_decode=0.6, max_total=0.49, gpu="H200")
    result = planner.plan(req, profiles(), k=20)
    kinds = []
    for candidate in result:
        kinds.append(candidate["kind"])
        assert candidate["predicted"]["decode_params_b"] <= 0.6
    assert "bake" in kinds
    assert "tap" not in kinds
    for candidate in result:
        if candidate["kind"] == "bake":
            assert candidate["bake"]["mode"] == "answer"
            assert candidate["predicted"]["prefill_params_b"] == pytest.approx(0.49)
            assert len(candidate["program"]["taps"]) == 1


def test_rules_tap_recovery():
    table = by_key()
    llama1b = table["unsloth/Llama-3.2-1B-Instruct"]
    qwen7b = table["Qwen/Qwen2.5-7B-Instruct"]
    qwen14b = table["Qwen/Qwen2.5-14B-Instruct"]
    value, _ = rules.tap_recovery(llama1b, qwen7b, "mmlu")
    assert 65 < value < 70.9
    # A weaker donor gives nothing.
    value, note = rules.tap_recovery(qwen14b, qwen7b, "mmlu")
    assert value == pytest.approx(77.9)
    assert "run 9" in note
    # Reasoning: no gain.
    value, _ = rules.tap_recovery(llama1b, qwen14b, "gsm8k")
    assert value == pytest.approx(25.0)
    # Free-form short answers: the tap hurts.
    trivia_host = profile("h", "h", 1.0, 16, 2048)
    trivia_host["scores"]["triviaqa"] = 37.6
    trivia_donor = profile("d", "d", 7.0, 28, 3584)
    trivia_donor["scores"]["triviaqa"] = 46.8
    value, _ = rules.tap_recovery(trivia_host, trivia_donor, "triviaqa", kind="generate")
    assert value < 37.6
    # An unmeasured small host and big donor: the 0.9 rule.
    host = profile("h", "new/host", 1.0, 20, 2048, mmlu=40.0)
    donor = profile("d", "new/donor", 10.0, 40, 4096, mmlu=80.0)
    value, _ = rules.tap_recovery(host, donor, "mmlu")
    assert value == pytest.approx(76.0)


def test_rules_lora_gain_and_layers():
    table = by_key()
    llama1b = table["unsloth/Llama-3.2-1B-Instruct"]
    gain, _ = rules.lora_gain(llama1b, "gsm8k")
    assert gain == pytest.approx(22.2)
    gain, _ = rules.lora_gain(llama1b, "mmlu")
    assert gain == pytest.approx(3.0)
    gain, _ = rules.lora_gain(table["Qwen/Qwen2.5-14B-Instruct"], "mmlu")
    assert gain == pytest.approx(0.0)
    assert rules.read_layer(table["Qwen/Qwen2.5-7B-Instruct"]) == 20
    assert rules.read_layer(table["Qwen/Qwen2.5-14B-Instruct"]) == 34
    assert rules.write_layer(llama1b) == 6
    assert rules.stacker_gain("medmcqa") == pytest.approx(1.6)
    assert rules.ensemble_accuracy([70.0, 72.0], "mmlu") == pytest.approx(73.2)


def test_score_fallbacks():
    table = by_key()
    score, note = rules.score_of(table["Qwen/Qwen2.5-7B-Instruct"], "medqa")
    assert score == pytest.approx(56.7)
    assert "medmcqa" in note
    score, note = rules.score_of(profile("u", "u", 7.0, 28, 3584), "mmlu")
    assert 60 < score < 75
    assert "guess" in note
    assert rules.as_percent(0.45) == pytest.approx(45.0)


def test_profiles_as_dict_and_fraction_target():
    table = {}
    for row in profiles():
        key = row.pop("key")
        table[key] = row
    req = request(objective="min_cost_at_target", target=0.7)
    result = planner.plan(req, table)
    assert result[0]["meets_target"]
    assert result[0]["predicted"]["accuracy"] >= 70.0


def test_bad_request_raises():
    with pytest.raises(ValueError):
        planner.plan({"task": {}}, profiles())
    with pytest.raises(ValueError):
        planner.plan(request(objective="cheapest"), profiles())


LLM_REPLY = """Here are two ideas.

```wsp
host Qwen/Qwen2.5-0.5B-Instruct
tap unsloth/gemma-2-9b-it layer 28 at 9
tap_train data mmlu_aux steps 300
```

```wsp
host unsloth/Llama-3.2-1B-Instruct
tap Qwen/Qwen2.5-7B-Instruct layer 99 at 6
tap_train data mmlu_aux
```

```wsp
host unsloth/Llama-3.2-1B-Instruct
tap nobody
```
"""


def test_llm_planner_mocked(monkeypatch):
    secret = "sk-test-secret-123"
    calls = []

    def fake_post(url, headers, body, timeout):
        calls.append({"url": url, "headers": headers, "body": body})
        return {"choices": [{"message": {"content": LLM_REPLY}}]}

    monkeypatch.setattr(planner, "http_post", fake_post)
    env = {"OPENAI_API_KEY": secret, "WSP_PLANNER_MODEL": "test-model"}
    details = planner.plan_details(request(max_decode=1.5), profiles(), k=100, use_llm=True, env=env)
    assert len(calls) == 1
    assert calls[0]["body"]["model"] == "test-model"
    assert calls[0]["headers"]["Authorization"] == "Bearer " + secret
    prompt = calls[0]["body"]["messages"][1]["content"]
    assert secret not in prompt
    assert "tap_train" in prompt and "Qwen/Qwen2.5-14B-Instruct" in prompt

    llm = []
    for candidate in details["candidates"] + details["dropped"]:
        if candidate["planner"] == "llm":
            llm.append(candidate)
    # The first program is valid; the second reads a layer the donor lacks; the third does not parse.
    assert len(llm) == 2
    good = None
    for candidate in llm:
        program = Program.from_dict(candidate["program"])
        assert program.to_dict() == candidate["program"]
        if candidate["program"]["host"] == "Qwen/Qwen2.5-0.5B-Instruct":
            good = candidate
    assert good is not None
    assert good in details["candidates"]
    assert good["program"]["taps"][0]["donor_layer"] == 28
    assert good["predicted"]["accuracy"] > 60
    bad = None
    for candidate in details["dropped"]:
        if candidate["planner"] == "llm":
            bad = candidate
    assert bad is not None and "layers" in " ".join(bad["problems"])
    assert any("does not parse" in note for note in details["notes"])
    assert secret not in json.dumps(details)


def test_llm_failure_falls_back_without_leaking_key(monkeypatch):
    secret = "sk-test-secret-456"

    def failing_post(url, headers, body, timeout):
        raise RuntimeError("Incorrect API key provided: " + secret)

    monkeypatch.setattr(planner, "http_post", failing_post)
    env = {"OPENAI_API_KEY": secret, "WSP_PLANNER_MODEL": "test-model"}
    result = planner.plan(request(max_decode=1.5), profiles(), use_llm=True, env=env)
    assert len(result) > 0
    for candidate in result:
        assert candidate["planner"] == "rules"
    joined = " ".join(result.notes)
    assert "llm planner failed: RuntimeError" in joined
    assert secret not in joined
    assert secret not in json.dumps(list(result))


def test_llm_skipped_without_key_or_model(monkeypatch):
    def never(url, headers, body, timeout):
        raise AssertionError("no call expected")

    monkeypatch.setattr(planner, "http_post", never)
    result = planner.plan(request(max_decode=1.5), profiles(), use_llm=True, env={})
    assert any("OPENAI_API_KEY" in note for note in result.notes)
    result = planner.plan(request(max_decode=1.5), profiles(), use_llm=True, env={"OPENAI_API_KEY": "x"})
    assert any("WSP_PLANNER_MODEL" in note for note in result.notes)


def test_env_file_parsing(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("WSP_PLANNER_MODEL", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    (tmp_path / ".env").write_text("# comment\nexport OPENAI_API_KEY=\"abc\"\nWSP_PLANNER_MODEL = m1\n\nJUNK\n")
    settings = planner.llm_settings(repo_root=str(tmp_path))
    assert settings["key"] == "abc"
    assert settings["model"] == "m1"
    assert settings["base_url"] == "https://api.openai.com/v1"
    monkeypatch.setenv("WSP_PLANNER_MODEL", "m2")
    assert planner.llm_settings(repo_root=str(tmp_path))["model"] == "m2"


def test_extract_programs():
    blocks = planner.extract_programs(LLM_REPLY + "\n```\nhost a/b\n```\n```python\nprint(1)\n```")
    assert len(blocks) == 4
    assert blocks[3].strip() == "host a/b"
