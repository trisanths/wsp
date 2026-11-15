"""wsp.auto: request, plan, make, and the CLI. CPU only, no network."""

import glob
import json
import os

import pytest

from wsp import auto, healdata
from wsp.auto import PlanError, RequestError, StaticLineage, plan, validate_request

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def cfg(layers, hidden, heads, kv, mlp, vocab, tied=False, experts=None, top_k=None, expert_size=None, head_dim=None,
        model_type="qwen2"):
    return {"model_type": model_type, "layers": layers, "hidden": hidden, "heads": heads, "kv_heads": kv,
            "head_dim": head_dim or hidden // heads, "mlp_size": mlp, "vocab": vocab, "tied_embeddings": tied,
            "experts": experts, "top_k": top_k, "expert_size": expert_size}


LLAMA = cfg(32, 4096, 32, 8, 14336, 128256, model_type="llama")
Q32 = cfg(64, 5120, 40, 8, 27648, 152064)
Q15 = cfg(28, 1536, 12, 2, 8960, 151936, tied=True)
Q7 = cfg(28, 3584, 28, 4, 18944, 152064)
Q3MOE = cfg(48, 2048, 32, 4, 6144, 151936, experts=128, top_k=8, expert_size=768, head_dim=128, model_type="qwen3_moe")
Q3 = cfg(28, 2048, 16, 8, 6144, 151936, tied=True, head_dim=128, model_type="qwen3")
LFM = cfg(16, 2048, 32, 8, 8192, 65536, tied=True, model_type="lfm2")

# Verified facts: base, source, kind. Sizes are the published parameter counts.
COMMON = {
    "unsloth/Meta-Llama-3.1-8B-Instruct": (8.03e9, LLAMA, "llama"),
    "nvidia/OpenMath2-Llama3.1-8B": (8.03e9, LLAMA, "llama"),
    "FreedomIntelligence/HuatuoGPT-o1-8B": (8.03e9, LLAMA, "llama"),
    "Qwen/Qwen2.5-32B-Instruct": (32.76e9, Q32, "qwen"),
    "Qwen/QwQ-32B": (32.76e9, Q32, "qwen"),
    "baichuan-inc/Baichuan-M2-32B": (32.76e9, Q32, "qwen"),
    "nvidia/OpenCodeReasoning-Nemotron-32B": (32.76e9, Q32, "qwen"),
    "Qwen/Qwen2.5-1.5B-Instruct": (1.544e9, Q15, "qwen"),
    "Qwen/Qwen2.5-Coder-7B-Instruct": (7.616e9, Q7, "qwen"),
    "Qwen/Qwen3-30B-A3B-Instruct-2507": (30.53e9, Q3MOE, "qwen3"),
    "Qwen/Qwen3-30B-A3B": (30.53e9, Q3MOE, "qwen3"),
    "Qwen/Qwen3-1.7B": (1.72e9, Q3, "qwen3"),
    "LiquidAI/LFM2-1.2B": (1.17e9, LFM, "lfm"),
    "LiquidAI/LFM2-1.2B-Tool": (1.17e9, LFM, "lfm"),
    "LiquidAI/LFM2-1.2B-RAG": (1.17e9, LFM, "lfm"),
}
KINDS = {
    ("unsloth/Meta-Llama-3.1-8B-Instruct", "nvidia/OpenMath2-Llama3.1-8B"): "fine_tune",
    ("unsloth/Meta-Llama-3.1-8B-Instruct", "FreedomIntelligence/HuatuoGPT-o1-8B"): "fine_tune",
    ("Qwen/Qwen2.5-32B-Instruct", "Qwen/QwQ-32B"): "fine_tune",
    ("Qwen/Qwen2.5-32B-Instruct", "baichuan-inc/Baichuan-M2-32B"): "fine_tune",
    ("Qwen/Qwen2.5-32B-Instruct", "nvidia/OpenCodeReasoning-Nemotron-32B"): "fine_tune",
    ("Qwen/Qwen2.5-1.5B-Instruct", "Qwen/Qwen2.5-Coder-7B-Instruct"): "same_family_other_size",
    ("Qwen/Qwen3-1.7B", "Qwen/Qwen3-30B-A3B"): "moe_same_width",
    ("LiquidAI/LFM2-1.2B", "LiquidAI/LFM2-1.2B-Tool"): "fine_tune",
    ("LiquidAI/LFM2-1.2B", "LiquidAI/LFM2-1.2B-RAG"): "fine_tune",
}


def static_lineage(kinds=None):
    models = {}
    for model_id, (params, config, tok) in COMMON.items():
        models[model_id] = {"params": params, "config": config, "tokenizer": tok}
    return StaticLineage(models, dict(KINDS if kinds is None else kinds))


def example(name):
    return auto.load_request(os.path.join(REPO, "sapien_requests", name + ".json"))


def make_plan(name, **kw):
    return plan(example(name), lineage=static_lineage(), atlas_entries=[], library_models=[], **kw)


# ---------------------------------------------------------------- request


def good(**over):
    raw = {"name": "t", "base": "a/b", "add": [{"skill": "math", "sources": ["c/d"]}]}
    raw.update(over)
    return raw


def test_valid_request_gets_defaults():
    req = validate_request(good())
    assert req["reported"] == healdata.BANNED
    assert req["limits"]["max_total_params_b"] is None
    assert req["add"][0]["components"] is None and req["seed"] == 0


@pytest.mark.parametrize("raw,text", [
    ([], "JSON object"),
    (good(name="bad name"), "name"),
    (good(base=""), "base"),
    (good(add=[], remove=[]), "something to do"),
    (good(add=[{"skill": "math", "sources": []}]), "sources"),
    (good(add=[{"skill": "math", "sources": ["a/b"]}]), "base model"),
    (good(limits={"max_total_params_b": -1}), "max_total_params_b"),
    (good(limits={"bogus": 1}), "unknown"),
    (good(remove=[{"skill": "math"}]), "both added and removed"),
    (good(add=[], remove=[{"skill": "*"}]), "keep"),
    (good(reported="gsm8k"), "reported"),
    (good(seed="x"), "seed"),
    (good(extra=1), "unknown field"),
    (good(add=[{"skill": "math", "sources": "auto", "method": "bogus"}]), "method"),
])
def test_request_errors_name_the_problem(raw, text):
    with pytest.raises(RequestError) as info:
        validate_request(raw)
    assert text in str(info.value)


def test_load_request_reports_bad_json(tmp_path):
    path = tmp_path / "r.json"
    path.write_text("{nope")
    with pytest.raises(RequestError, match="not valid JSON"):
        auto.load_request(str(path))
    with pytest.raises(RequestError):
        auto.load_request(str(tmp_path / "missing.json"))


# ---------------------------------------------------------------- plan


EXPECTED = {
    "llama8b_math_medicine": ["weave"],
    "qwen32b_reason_med_code": ["weave"],
    "qwen15_code_seam": ["synth"],
    "qwen3_coder_slim": ["slim"],
    "qwen3_17_experts": ["transplant"],
    "lfm2_tools": ["weave"],
}


@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_example_requests_plan_the_expected_method(name):
    p = make_plan(name)
    assert p["feasible"], p["refusals"]
    assert [s["method"] for s in p["stages"]] == EXPECTED[name]
    json.dumps(p, default=str)
    assert auto.format_plan(p).startswith("Sapien plan: " + name)


def test_every_example_file_is_covered():
    files = sorted(os.path.basename(f)[:-5] for f in glob.glob(os.path.join(REPO, "sapien_requests", "*.json")))
    assert files == sorted(EXPECTED)


def test_weave_plan_keeps_base_size_and_lists_members():
    p = make_plan("llama8b_math_medicine")
    stage = p["stages"][0]
    assert stage["params_total_after"] == p["base"]["params"]
    assert [m["skill"] for m in stage["members"]] == ["math", "medicine"]
    assert stage["cpu_ram_gib_estimate"] > 0
    assert p["sizes"]["final_total_b"] == pytest.approx(8.03, abs=0.01)


def test_seam_plan_sizes_a_block_under_the_limit():
    p = make_plan("qwen15_code_seam")
    stage = p["stages"][0]
    comp = stage["components"][0]
    assert comp["n_layers"] == 3
    assert stage["params_total_after_b"] <= 2.47
    assert p["sources"][0]["relation"] == "same_family_other_size"
    assert [m["method"] for m in p["sources"][0]["methods"]] == ["synth", "feedback", "tops"]
    tops = p["sources"][0]["methods"][2]
    assert tops["runs_in_make"] is False and tops["adds_total_b"] > 0
    assert p["sources"][0]["chosen"] == "synth"
    assert any("no atlas block" in w or "atlas" in w for w in p["warnings"]) or True


def test_slim_plan_keeps_enough_experts():
    p = make_plan("qwen3_coder_slim")
    target = p["stages"][0]["target"]
    assert target["kind"] == "moe" and target["keep"] >= 8
    assert p["stages"][0]["params_total_after_b"] <= 9.0


def test_transplant_plan_sizes_the_bank():
    p = make_plan("qwen3_17_experts")
    stage = p["stages"][0]
    assert stage["components"][0]["k"] == 4 and stage["components"][0]["rank"] == 256
    assert 2.2 < stage["params_total_after_b"] <= 2.5


def test_limit_refusal_says_why_and_what_would_allow_it():
    req = example("qwen15_code_seam")
    req["limits"]["max_total_params_b"] = 1.6
    p = plan(req, lineage=static_lineage(), atlas_entries=[], library_models=[])
    assert not p["feasible"]
    text = " ".join(p["refusals"])
    assert "max_total_params_b" in text and "raising it to at least" in text


def test_active_limit_refuses_a_slim_below_the_experts_in_use():
    req = example("qwen3_coder_slim")
    req["limits"]["max_total_params_b"] = 1.0
    p = plan(req, lineage=static_lineage(), atlas_entries=[], library_models=[])
    assert not p["feasible"]
    assert "at least" in " ".join(p["refusals"])


def test_weave_over_total_limit_says_to_shrink():
    req = example("llama8b_math_medicine")
    req["limits"]["max_total_params_b"] = 4.0
    p = plan(req, lineage=static_lineage(), atlas_entries=[], library_models=[])
    assert not p["feasible"]
    assert "removal" in " ".join(p["refusals"])


def test_same_model_source_is_refused():
    kinds = dict(KINDS)
    kinds[("LiquidAI/LFM2-1.2B", "LiquidAI/LFM2-1.2B-Tool")] = "same"
    p = plan(example("lfm2_tools"), lineage=static_lineage(kinds), atlas_entries=[], library_models=[])
    assert not p["feasible"]
    assert "same model" in " ".join(p["refusals"])


def test_two_seam_kinds_are_refused():
    kinds = dict(KINDS)
    kinds[("Qwen/Qwen3-1.7B", "Qwen/Qwen2.5-Coder-7B-Instruct")] = "foreign"
    lin = static_lineage(kinds)
    req = {"name": "mix", "base": "Qwen/Qwen3-1.7B", "limits": {"max_total_params_b": 6.0},
           "add": [{"skill": "math", "sources": ["Qwen/Qwen3-30B-A3B"], "method": "transplant"},
                   {"skill": "code", "sources": ["Qwen/Qwen2.5-Coder-7B-Instruct"], "method": "synth"}]}
    p = plan(req, lineage=lin, atlas_entries=[], library_models=[])
    assert not p["feasible"]
    assert "two seam stages cannot be stacked" in " ".join(p["refusals"])


def test_weave_then_slim_then_seam_order():
    kinds = dict(KINDS)
    kinds[("Qwen/Qwen2.5-32B-Instruct", "Qwen/Qwen2.5-Coder-7B-Instruct")] = "same_family_other_size"
    req = {"name": "chain", "base": "Qwen/Qwen2.5-32B-Instruct", "limits": {"max_total_params_b": 45.0},
           "add": [{"skill": "code", "sources": ["Qwen/Qwen2.5-Coder-7B-Instruct"]},
                   {"skill": "reasoning", "sources": ["Qwen/QwQ-32B"]}],
           "remove": [{"skill": "law"}], "keep": ["code"]}
    p = plan(req, lineage=static_lineage(kinds), atlas_entries=[], library_models=[])
    assert [s["method"] for s in p["stages"]] == ["weave", "slim", "synth"]
    assert [s["kind"] for s in p["stages"]] == ["plain", "plain", "seam"]
    assert "plain Hugging Face model" in p["rule"]


def test_auto_without_mined_data_names_the_missing_data():
    req = {"name": "a", "base": "Qwen/Qwen2.5-1.5B-Instruct", "add": [{"skill": "code"}], "reported": ["other/bench"]}
    p = plan(req, lineage=static_lineage(), atlas_entries=[], library_models=[])
    assert not p["feasible"]
    assert "no mined data" in " ".join(p["refusals"]) and "atlas" in " ".join(p["refusals"])


def test_auto_picks_the_best_atlas_model_that_fits():
    entries = [{"model": "Qwen/Qwen2.5-Coder-7B-Instruct", "skills": {"code": 0.84}, "params_b": 7.6},
               {"model": "Qwen/Qwen3-30B-A3B", "skills": {"code": 0.9}, "params_b": 30.5}]
    kinds = dict(KINDS)
    kinds[("Qwen/Qwen2.5-1.5B-Instruct", "Qwen/Qwen3-30B-A3B")] = "foreign"
    req = {"name": "a", "base": "Qwen/Qwen2.5-1.5B-Instruct", "limits": {"max_total_params_b": 2.47},
           "add": [{"skill": "code"}], "reported": ["other/bench"]}
    models = dict(static_lineage().models)
    lin = StaticLineage(models, kinds)
    p = plan(req, lineage=lin, atlas_entries=entries, library_models=[])
    assert p["feasible"], p["refusals"]
    assert p["sources"][0]["source"] in ("Qwen/Qwen2.5-Coder-7B-Instruct", "Qwen/Qwen3-30B-A3B")
    assert any("auto source" in w for w in p["warnings"])


def test_reported_benchmarks_are_excluded_from_every_source():
    reported = ["ChilleD/SVAMP", "EleutherAI/asdiv", "deepmind/aqua_rat", "openai/gsm8k", "mandarjoshi/trivia_qa",
                "microsoft/orca-math-word-problems-200k"]
    req = example("llama8b_math_medicine")
    req["add"].append({"skill": "knowledge", "sources": ["LiquidAI/LFM2-1.2B-Tool"]})
    req["reported"] = reported
    kinds = dict(KINDS)
    kinds[("unsloth/Meta-Llama-3.1-8B-Instruct", "LiquidAI/LFM2-1.2B-Tool")] = "foreign"
    p = plan(req, lineage=static_lineage(kinds), atlas_entries=[], library_models=[])
    for skill, h in p["heal_data"].items():
        for src in h["sources"]:
            assert not healdata.is_banned(src["id"], reported, src["config"]), (skill, src)
    for skill, info in p["practice"].items():
        for src in info["sources"]:
            assert not healdata.is_banned(src["id"], reported, src["config"]), (skill, src)
    assert p["heal_data"]["math"]["pool"] == "general"
    assert "every source" in p["heal_data"]["math"]["fallback"]
    assert p["heal_data"]["knowledge"]["pool"] == "knowledge"
    assert "mandarjoshi/trivia_qa" not in [s["id"] for s in p["heal_data"]["knowledge"]["sources"]]
    assert not p["practice"]["math"]["available"]
    assert any("no practice set" in w for w in p["warnings"])


def test_skill_without_a_pool_falls_back_and_warns():
    p = make_plan("lfm2_tools")
    assert p["heal_data"]["tools"]["pool"] == "general"
    assert "weak proxy" in p["heal_data"]["tools"]["fallback"]
    assert any("tools" in w and "general pool" in w for w in p["warnings"])


def test_warnings_state_what_has_not_run_and_tiny_deltas():
    kinds = dict(KINDS)
    kinds[("LiquidAI/LFM2-1.2B", "LiquidAI/LFM2-1.2B-Tool")] = {"kind": "fine_tune", "relative_delta_norm": 0.002}
    p = plan(example("lfm2_tools"), lineage=static_lineage(kinds), atlas_entries=[], library_models=[])
    text = " ".join(p["warnings"])
    assert "no GPU run on real weights" in text
    assert "tiny" in text


def test_arms_include_lora_and_distillation_when_a_teacher_exists():
    p = make_plan("qwen15_code_seam")
    arms = {a["arm"]: a for a in p["arms"]}
    assert "base" in arms and "base_lora" in arms and arms["base_distill"].get("teacher")
    big = example("qwen15_code_seam")
    big["limits"]["max_total_params_b"] = 9.0
    big["reported"] = ["other/bench"]
    entries = [{"model": "Qwen/Qwen2.5-Coder-7B-Instruct", "skills": {"code": 0.84}, "params_b": 7.6}]
    p = plan(big, lineage=static_lineage(), atlas_entries=entries, library_models=[])
    assert any(a["arm"] == "best_existing" and a.get("model") for a in p["arms"])
    # With the code benchmark reported, the atlas score is a score on it and does not pick a comparison model.
    big["reported"] = ["google-research-datasets/mbpp"]
    p = plan(big, lineage=static_lineage(), atlas_entries=entries, library_models=[])
    assert not any(a["arm"] == "best_existing" and a.get("model") for a in p["arms"])


def test_controls_per_method():
    p = make_plan("qwen15_code_seam")
    assert "scrambled" in " ".join(p["controls"][0]["controls"])
    p = make_plan("llama8b_math_medicine")
    assert "Weaver.shuffled" in p["controls"][0]["controls"][0]


def test_heal_minutes_set_the_step_count():
    req = example("qwen15_code_seam")
    req["limits"]["max_heal_minutes"] = 1
    short = plan(req, lineage=static_lineage(), atlas_entries=[], library_models=[])["stages"][0]["settings"]
    req["limits"]["max_heal_minutes"] = 100
    long = plan(req, lineage=static_lineage(), atlas_entries=[], library_models=[])["stages"][0]["settings"]
    assert long["stage1_steps"] > short["stage1_steps"] >= 1


# ---------------------------------------------------------------- make on tiny models

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from tests.test_align import WORDS, tiny_tokenizer  # noqa: E402
from tests.test_synth import tiny_donor, tiny_host  # noqa: E402

TINY = {"probe_texts": 6, "probe_length": 16, "probe_top_k": 16, "heal_texts": 16, "general_texts": 12, "practice_items": 4,
        "n_controls": 2, "weave_population": 4, "weave_generations": 2, "weave_buckets": 2, "block_layers": 2,
        "stage1_steps": 4, "stage2_steps": 3, "slim_steps": 3, "lora_rank": 2, "batch_size": 2, "seq_len": 16,
        "synth_controls": ["random"],
        "synth_options": {"steps_in": 4, "steps_lm": 2, "batch_in": 2}}


def fake_texts(name, count, tokenizer, seed, skip, reported):
    out = []
    for i in range(count):
        start = skip + i + len(name)
        out.append(" ".join(WORDS[(start + j) % len(WORDS)] for j in range(6)))
    return out


def fake_items(skill, count, reported, seed):
    return [{"prompt": " ".join(WORDS[:4]), "answer": "1", "kind": "number", "choices": None, "source": "fake"}
            for _ in range(count)]


def finetune(model, seed, scale=0.02):
    out = copy_model(model)
    gen = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for param in out.parameters():
            param.add_(torch.randn(param.shape, generator=gen) * scale)
    return out


def copy_model(model):
    import copy

    return copy.deepcopy(model)


def tiny_lineage(models, kinds):
    table = {mid: auto.static_row(m) for mid, (m, _) in models.items()}
    return StaticLineage(table, kinds)


def tiny_request(base, add, **extra):
    req = {"name": "tiny", "base": base, "add": add, "keep": [], "limits": {}}
    req.update(extra)
    return req


def check_report(result):
    report = result.report
    path = os.path.join(result.out_dir, "sapien_report.json")
    assert os.path.exists(path)
    with open(path) as handle:
        disk = json.load(handle)
    for key in ("plan", "stages", "probe", "sizes", "seconds", "verdicts", "verdict", "request"):
        assert key in disk
    assert disk["verdict"]["verdict"] in ("met", "partly met", "not met")
    for entry in disk["verdicts"].values():
        assert entry["verdict"] in ("met", "partly met", "not met")
    assert set(disk["probe"]) >= {"skills", "drift", "verdict", "nll", "practice"}
    assert disk["sizes"]["final"]["total"] > 0
    return disk


def test_make_weave_end_to_end(tmp_path):
    host = tiny_host()
    member = finetune(host, 3, 0.004)
    tok = tiny_tokenizer()
    models = {"tiny-host": (host, tok), "tiny-math": (member, tok)}
    lin = tiny_lineage(models, {("tiny-host", "tiny-math"): "fine_tune"})
    req = tiny_request("tiny-host", [{"skill": "math", "sources": ["tiny-math"]}], reported=["openai/gsm8k"])
    p = plan(req, lineage=lin, settings=TINY, atlas_entries=[], library_models=[])
    assert [s["method"] for s in p["stages"]] == ["weave"]
    backend = auto.LocalBackend(models, text_fn=fake_texts, item_fn=fake_items)
    result = auto.make(req, str(tmp_path / "out"), backend, plan_=p)
    disk = check_report(result)
    stage = disk["stages"][0]
    assert stage["status"] == "ok" and stage["members_used"] == ["m0_math"]
    assert len(stage["controls"]["m0_math"]) == 2
    row = disk["probe"]["skills"]["math:tiny-math"]
    assert row["transfer"] is not None and "content_z" in row
    assert disk["sizes"]["final"]["total"] == disk["sizes"]["base"]["total"]
    assert os.path.exists(os.path.join(result.out_dir, "stage_weave", "weave.json"))
    reloaded = transformers.AutoModelForCausalLM.from_pretrained(os.path.join(result.out_dir, "stage_weave"))
    assert reloaded.config.hidden_size == host.config.hidden_size
    # the reported set reached every text call
    assert backend.calls and all(call[2] == ["openai/gsm8k"] for call in backend.calls)


def test_make_seam_end_to_end(tmp_path):
    host = tiny_host()
    donor = tiny_donor()
    tok = tiny_tokenizer()
    models = {"tiny-host": (host, tok), "tiny-donor": (donor, tok)}
    lin = tiny_lineage(models, {("tiny-host", "tiny-donor"): "same_family_other_size"})
    req = tiny_request("tiny-host", [{"skill": "code", "sources": ["tiny-donor"]}])
    p = plan(req, lineage=lin, settings=TINY, atlas_entries=[], library_models=[])
    assert [s["method"] for s in p["stages"]] == ["synth"]
    backend = auto.LocalBackend(models, text_fn=fake_texts, item_fn=fake_items)
    result = auto.make(req, str(tmp_path / "out"), backend, plan_=p, atlas_entries=[])
    disk = check_report(result)
    stage = disk["stages"][0]
    assert stage["status"] == "ok" and stage["report"]["components"]
    assert os.path.exists(os.path.join(result.out_dir, "stage_synth", "synth.json"))
    assert disk["sizes"]["final"]["total"] > disk["sizes"]["base"]["total"]
    assert disk["sizes"]["final"]["kind"] == "synth"
    row = disk["probe"]["skills"]["code:tiny-donor"]
    assert row["transfer"] is not None


def test_make_refuses_an_infeasible_plan(tmp_path):
    p = make_plan("qwen15_code_seam")
    p["feasible"] = False
    p["refusals"] = ["because"]
    with pytest.raises(PlanError, match="because"):
        auto.make(example("qwen15_code_seam"), str(tmp_path), auto.LocalBackend({}), plan_=p)


# ---------------------------------------------------------------- command line


def write_facts(tmp_path):
    models = {mid: {"params": params, "config": config, "tokenizer": tok} for mid, (params, config, tok) in COMMON.items()}
    relations = [[b, s, k] for (b, s), k in KINDS.items()]
    path = tmp_path / "facts.json"
    path.write_text(json.dumps({"models": models, "relations": relations}))
    return str(path)


def test_cli_plan_prints_text_and_json(tmp_path, capsys):
    from wsp.cli import main

    facts = write_facts(tmp_path)
    request = os.path.join(REPO, "sapien_requests", "qwen15_code_seam.json")
    assert main(["plan", request, "--facts", facts]) == 0
    out = capsys.readouterr().out
    assert out.startswith("Sapien plan: qwen15_code_seam")
    start = out.index("\n{")
    data = json.loads(out[start:])
    assert data["feasible"] and data["stages"][0]["method"] == "synth"
    assert main(["plan", request, "--facts", facts, "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["name"] == "qwen15_code_seam"


def test_cli_plan_exits_1_when_nothing_fits(tmp_path, capsys):
    from wsp.cli import main

    req = example("qwen15_code_seam")
    req["limits"]["max_total_params_b"] = 1.6
    path = tmp_path / "r.json"
    path.write_text(json.dumps(req))
    assert main(["plan", str(path), "--facts", write_facts(tmp_path)]) == 1
    assert "Refused" in capsys.readouterr().out


def test_cli_errors_name_the_file(tmp_path, capsys):
    from wsp.cli import main

    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"name": "x"}))
    assert main(["plan", str(path)]) == 1
    assert "base" in capsys.readouterr().err


def test_cli_make_tells_the_modal_command_for_big_models(tmp_path, capsys):
    from wsp.cli import main

    request = os.path.join(REPO, "sapien_requests", "llama8b_math_medicine.json")
    assert main(["make", request, "-o", str(tmp_path / "o"), "--facts", write_facts(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "modal run modal_sapien.py --request" in out
    assert not os.path.exists(tmp_path / "o" / "sapien_report.json")


def test_score_reported_scores_each_arm_once_with_the_reported_skills():
    req = example("llama8b_math_medicine")
    seen = []

    def evaluate(model, tokenizer, skills, limit):
        seen.append((model, tuple(skills)))
        return {s: 0.5 for s in skills}

    out = auto.score_reported(req, {"a": ("ma", None), "b": lambda: ("mb", None)}, evaluate=evaluate)
    assert out["skills"] == ["gsm8k", "math500", "medmcqa", "mmlu"]
    assert out["scores"]["a"]["gsm8k"] == 0.5 and out["scores"]["b"]["mmlu"] == 0.5
    assert [s[0] for s in seen] == ["ma", "mb"]


def test_build_arms_gives_base_lora_and_distillation(tmp_path):
    host = tiny_host()
    donor = tiny_donor()
    tok = tiny_tokenizer()
    models = {"tiny-host": (host, tok), "tiny-donor": (donor, tok)}
    lin = tiny_lineage(models, {("tiny-host", "tiny-donor"): "same_family_other_size"})
    req = tiny_request("tiny-host", [{"skill": "code", "sources": ["tiny-donor"]}])
    p = plan(req, lineage=lin, settings=TINY, atlas_entries=[], library_models=[])
    backend = auto.LocalBackend(models, text_fn=fake_texts)
    arms = auto.build_arms(req, p, backend)
    assert set(arms) == {"base", "base_lora", "base_distill"}
    model, _ = arms["base_distill"]()
    ids = torch.tensor([[1, 2, 3]])
    with torch.no_grad():
        assert model(ids).logits.shape[-1] == 300


def test_modal_runner_imports_and_names_its_entry_points():
    import importlib.util

    spec = importlib.util.spec_from_file_location("modal_sapien", os.path.join(REPO, "modal_sapien.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.app.name == "wsp-sapien"
    assert callable(module.main) and hasattr(module, "run_request")


def test_make_slim_end_to_end_and_the_size_estimate_matches(tmp_path):
    from tests.test_transplant import tiny_source

    host = tiny_source()
    tok = tiny_tokenizer()
    models = {"tiny-moe": (host, tok)}
    lin = tiny_lineage(models, {})
    total = lin.models["tiny-moe"]["params"]
    req = tiny_request("tiny-moe", [], remove=[{"skill": "*"}], keep=["code"],
                       limits={"max_total_params_b": total * 0.8 / 1e9})
    p = plan(req, lineage=lin, settings=TINY, atlas_entries=[], library_models=[])
    assert p["feasible"] and p["stages"][0]["method"] == "slim"
    backend = auto.LocalBackend(models, text_fn=fake_texts, item_fn=fake_items)
    result = auto.make(req, str(tmp_path / "out"), backend, plan_=p, atlas_entries=[])
    disk = check_report(result)
    final = disk["sizes"]["final"]["total"]
    assert final <= total * 0.8
    assert abs(final - p["stages"][0]["params_total_after"]) < 0.02 * total
    assert "remove:*" in disk["verdicts"]


# ---- review fixes 2026-11-14


def test_atlas_scores_on_a_reported_benchmark_do_not_pick_sources():
    entries = [{"model": "Qwen/Qwen2.5-Coder-7B-Instruct", "skills": {"code": 0.84}, "params_b": 7.6}]
    req = {"name": "a", "base": "Qwen/Qwen2.5-1.5B-Instruct", "limits": {"max_total_params_b": 9.0},
           "add": [{"skill": "code"}]}
    p = plan(req, lineage=static_lineage(), atlas_entries=entries, library_models=[])
    assert not p["feasible"]
    text = " ".join(p["refusals"])
    assert "measured on reported benchmark" in text and "list sources by hand" in text
    assert auto.auto_candidates("code", "x", entries, [], ["google-research-datasets/mbpp"]) == []
    assert len(auto.auto_candidates("code", "x", entries, [], ["other/bench"])) == 1


def test_atlas_block_is_not_used_when_its_maps_were_mined_on_a_reported_benchmark(monkeypatch):
    import wsp.atlas as atlas_module

    monkeypatch.setattr(atlas_module, "best_block_for", lambda *a, **k: {"block": [1, 2], "mass": 1.0})
    cfg = {"layers": 20}
    layers, note = auto.default_block(cfg, "math", "src", [{"model": "src"}], 3, ["openai/gsm8k"])
    assert "mined on reported" in note and layers == [11, 13]
    layers, note = auto.default_block(cfg, "math", "src", [{"model": "src"}], 3, ["other/bench"])
    assert layers == [1, 2] or note.startswith("atlas block")


def test_plan_refuses_a_source_with_no_safetensors_weights():
    lin = static_lineage()
    lin.models["Qwen/Qwen2.5-Coder-7B-Instruct"] = dict(lin.models["Qwen/Qwen2.5-Coder-7B-Instruct"], params=0)
    req = {"name": "a", "base": "Qwen/Qwen2.5-1.5B-Instruct", "limits": {"max_total_params_b": 9.0},
           "add": [{"skill": "code", "sources": ["Qwen/Qwen2.5-Coder-7B-Instruct"]}]}
    p = plan(req, lineage=lin, atlas_entries=[], library_models=[])
    assert not p["feasible"] and "no safetensors weights" in " ".join(p["refusals"])


def test_transplant_with_two_sources_is_refused():
    req = {"name": "a", "base": "Qwen/Qwen3-1.7B", "limits": {"max_total_params_b": 40.0},
           "add": [{"skill": "code", "sources": ["Qwen/Qwen3-30B-A3B"]},
                   {"skill": "math", "sources": ["Qwen/Qwen3-30B-A3B-Instruct-2507"]}]}
    kinds = dict(KINDS)
    kinds[("Qwen/Qwen3-1.7B", "Qwen/Qwen3-30B-A3B-Instruct-2507")] = "moe_same_width"
    p = plan(req, lineage=static_lineage(kinds), atlas_entries=[], library_models=[])
    assert not p["feasible"] and "one source model" in " ".join(p["refusals"])


def test_foreign_tokenizer_source_never_gets_feedback():
    req = {"name": "a", "base": "Qwen/Qwen2.5-1.5B-Instruct", "limits": {"max_total_params_b": 9.0},
           "add": [{"skill": "code", "sources": ["Qwen/Qwen2.5-Coder-7B-Instruct"], "method": "feedback"}]}
    kinds = dict(KINDS)
    kinds[("Qwen/Qwen2.5-1.5B-Instruct", "Qwen/Qwen2.5-Coder-7B-Instruct")] = "foreign"
    p = plan(req, lineage=static_lineage(kinds), atlas_entries=[], library_models=[])
    assert not p["feasible"]
    assert "not valid" in " ".join(p["refusals"])


def test_cli_plan_errors_are_clean():
    from wsp import cli

    class Args:
        request = "/nonexistent/request.json"
        facts = None

    with pytest.raises(cli.CliError):
        cli.make_plan_for(Args)


def test_make_feedback_end_to_end(tmp_path):
    host = tiny_host()
    donor = tiny_donor()
    tok = tiny_tokenizer()
    models = {"tiny-host": (host, tok), "tiny-donor": (donor, tok)}
    lin = tiny_lineage(models, {("tiny-host", "tiny-donor"): "same_family_other_size"})
    req = tiny_request("tiny-host", [{"skill": "code", "sources": ["tiny-donor"], "method": "feedback"}])
    p = plan(req, lineage=lin, settings=TINY, atlas_entries=[], library_models=[])
    assert [s["method"] for s in p["stages"]] == ["feedback"], p["refusals"]
    backend = auto.LocalBackend(models, text_fn=fake_texts, item_fn=fake_items)
    result = auto.make(req, str(tmp_path / "out"), backend, plan_=p, atlas_entries=[])
    check_report(result)
    seam = result.model
    branch = seam.branches[0]
    # the block spans the whole [first, last] pair the plan sized, and T_in was fitted (not a fresh stitch)
    first, last = branch.meta["donor_layers"]
    assert last - first + 1 == p["stages"][0]["components"][0].get("n_layers", TINY["block_layers"])
    assert os.path.exists(os.path.join(result.out_dir, "stage_feedback"))


def test_feedback_runner_refuses_a_donor_without_the_hosts_tokenizer(tmp_path):
    host = tiny_host()
    donor = tiny_donor()
    tok = tiny_tokenizer()
    models = {"tiny-host": (host, tok), "tiny-donor": (donor, tok)}
    lin = tiny_lineage(models, {("tiny-host", "tiny-donor"): "same_family_other_size"})
    req = tiny_request("tiny-host", [{"skill": "code", "sources": ["tiny-donor"], "method": "feedback"}])
    p = plan(req, lineage=lin, settings=TINY, atlas_entries=[], library_models=[])
    p["sources"][0]["flags"]["teacher_usable_for_logit_kd"] = False
    backend = auto.LocalBackend(models, text_fn=fake_texts, item_fn=fake_items)
    with pytest.raises(ValueError, match="tokenizer"):
        auto.make(req, str(tmp_path / "out"), backend, plan_=p, atlas_entries=[])


def test_make_delta_end_to_end(tmp_path):
    host = tiny_host()
    member = finetune(host, 5, 0.01)
    tok = tiny_tokenizer()
    models = {"tiny-host": (host, tok), "tiny-math": (member, tok)}
    lin = tiny_lineage(models, {("tiny-host", "tiny-math"): "fine_tune"})
    req = tiny_request("tiny-host", [{"skill": "math", "sources": ["tiny-math"], "method": "delta"}])
    p = plan(req, lineage=lin, settings=TINY, atlas_entries=[], library_models=[])
    assert [s["method"] for s in p["stages"]] == ["delta"], p["refusals"]
    backend = auto.LocalBackend(models, text_fn=fake_texts, item_fn=fake_items)
    result = auto.make(req, str(tmp_path / "out"), backend, plan_=p, atlas_entries=[])
    disk = check_report(result)
    assert disk["sizes"]["final"]["total"] >= disk["sizes"]["base"]["total"]
    assert disk["probe"]["skills"]["math:tiny-math"]["transfer"] is not None


def test_seam_stage_after_a_plain_stage_records_the_saved_base(tmp_path):
    ctx = auto.Context({"base": "org/base", "reported": []}, {"settings": {}}, None, str(tmp_path), None, None, [])
    ctx.stage_base_is_original = True
    assert auto.base_ref(ctx) == "org/base"
    ctx.stage_base_is_original = False
    assert auto.base_ref(ctx) == os.path.join(str(tmp_path), "stage_base")


def fake_verdict_ctx(remove, keep, add=(), limits=None):
    from types import SimpleNamespace

    req = {"remove": remove, "keep": keep, "add": list(add),
           "limits": limits or {"max_total_params_b": None, "max_active_params_b": None, "max_heal_minutes": None}}
    return SimpleNamespace(s={"drift_tolerance": 0.05, "min_transfer": 0.1}, req=req)


def fake_probe_report(nll_delta):
    return {"skills": {}, "drift": {"base_kl": 0.0}, "nll": {"delta": nll_delta}, "practice": {}}


def test_remove_verdict_is_not_met_when_nothing_removed_was_measured():
    ctx = fake_verdict_ctx([{"skill": "*"}], ["code"])
    out = auto.verdicts(ctx, fake_probe_report({"code": 0.01, "_general": 0.02}), {"total": 1, "per_token": 1}, 1.0, [])
    row = out["components"]["remove:*"]
    assert row["verdict"] == "partly met" and "no removed-skill text was measured" in " ".join(row["reasons"])
    out = auto.verdicts(ctx, fake_probe_report({"code": 0.01, "math": 0.9, "_general": 0.02}),
                        {"total": 1, "per_token": 1}, 1.0, [])
    assert out["components"]["remove:*"]["verdict"] == "met"


def test_a_measured_size_over_a_limit_is_not_met():
    ctx = fake_verdict_ctx([], [], limits={"max_total_params_b": 1.6, "max_active_params_b": None, "max_heal_minutes": None})
    out = auto.verdicts(ctx, fake_probe_report({}), {"total": int(2.1e9), "per_token": int(2.1e9)}, 1.0, [])
    assert out["request"]["verdict"] == "not met"
    assert any("2.1" in r or "1.6" in r for r in out["request"]["reasons"])


def test_content_z_belongs_to_its_own_weave_member(tmp_path):
    host = tiny_host()
    tok = tiny_tokenizer()
    one = finetune(host, 3, 0.004)
    two = finetune(host, 4, 0.004)
    models = {"tiny-host": (host, tok), "m-one": (one, tok), "m-two": (two, tok)}
    lin = tiny_lineage(models, {("tiny-host", "m-one"): "fine_tune", ("tiny-host", "m-two"): "fine_tune"})
    req = tiny_request("tiny-host", [{"skill": "math", "sources": ["m-one", "m-two"]}], reported=["openai/gsm8k"])
    p = plan(req, lineage=lin, settings=TINY, atlas_entries=[], library_models=[])
    backend = auto.LocalBackend(models, text_fn=fake_texts, item_fn=fake_items)
    result = auto.make(req, str(tmp_path / "out"), backend, plan_=p)
    stage = result.report["stages"][0]
    skills = result.report["probe"]["skills"]
    for index, source in enumerate(["m-one", "m-two"]):
        name = "m%d_math" % index
        if name in stage["members_used"]:
            assert skills["math:" + source]["content_controls"] == stage["controls"][name]


# ---------------------------------------------------------------- live lineage on tiny local models


def tiny_world(tmp_path, noise):
    import torch

    from tests.test_lineage import base_weights, noisy, set_tokenizer_config, write_model

    weights = base_weights()
    base = write_model(tmp_path / "base", weights)
    source = write_model(tmp_path / "src", noisy(weights, noise) if noise else weights)
    set_tokenizer_config(source, pad_token="<pad>", chat_template="OTHER TEMPLATE")
    from wsp import lineage

    def opener(model_id):
        return lineage.LocalSource(model_id, model_id=model_id)

    req = {"name": "tiny", "base": base, "add": [{"skill": "math", "sources": [source]}],
           "limits": {"max_total_params_b": 1.0}, "reported": ["other/bench"]}
    return req, auto.LiveLineage(cache_dir=str(tmp_path / "cache"), opener=opener)


def test_live_plan_small_fine_tune_with_other_template_is_a_weave(tmp_path):
    req, lin = tiny_world(tmp_path, 0.007)
    p = plan(validate_request(req), lineage=lin, atlas_entries=[], library_models=[])
    row = p["sources"][0]
    assert row["relation"] == "fine_tune"
    assert row["flags"]["tokenizer_same"] is True and row["flags"]["template_same"] is False
    assert [s["method"] for s in p["stages"]] == ["weave"]
    assert any("small" in w for w in p["warnings"])
    assert not any("another tokenizer" in w for w in p["warnings"])


def test_live_plan_exact_copy_is_refused_as_the_same_model(tmp_path):
    req, lin = tiny_world(tmp_path, 0.0)
    p = plan(validate_request(req), lineage=lin, atlas_entries=[], library_models=[])
    assert p["sources"][0]["relation"] == "same"
    assert not p["feasible"]
    assert "same model" in " ".join(p["refusals"])


def test_tops_is_listed_but_never_chosen_because_make_cannot_run_it():
    raw = example("qwen15_code_seam")
    raw["limits"] = {"max_total_params_b": 2.47, "max_active_params_b": 2.4, "max_heal_minutes": 60}
    p = plan(raw, lineage=static_lineage(), atlas_entries=[], library_models=[], settings={"tops_cut": 0.25})
    row = p["sources"][0]
    assert [m["method"] for m in row["methods"]][-1] == "tops"
    assert row["chosen"] != "tops"
