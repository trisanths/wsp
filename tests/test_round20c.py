"""Round 20c script checks on CPU: cell list, star design, smoke, controls phase, probe bookkeeping, equal inputs."""

import re
from pathlib import Path

import pytest
import torch

pytest.importorskip("modal")
transformers = pytest.importorskip("transformers")

import modal_round20c as r
from tests.test_align import tiny_tokenizer
from tests.test_synth import SMALL, TEXTS, block_comp, donors_of, sentences, tiny_donor, tiny_host
from wsp import reader, synth

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = (ROOT / "modal_round20c.py").read_text()


def by(cells, pair=None, config=None, arm=None):
    out = []
    for c in cells:
        if pair is not None and c[0] != pair:
            continue
        if config is not None and c[1] != config:
            continue
        if arm is not None and c[2] != arm:
            continue
        out.append(c)
    return out


# ---------------------------------------------------------------- cell list and star design


def test_star_design_has_one_default_and_each_factor_varied_alone():
    default = r.DEFAULT_CFG
    seen_factors = set()
    for name, factor, change in r.STAR:
        cfg = r.config_of(name)
        differing = [k for k in default if cfg[k] != default[k]]
        assert len(differing) == 1, name
        assert differing == list(change)
        seen_factors.add(factor)
    assert seen_factors == set(r.FACTORS)
    assert [r.config_of(n)["steps"] for n in ("steps500", "default", "steps8000")] == [500, 2000, 8000]
    assert sorted(r.config_of(n)["size"] for n in ("size2", "size4", "default", "size13")) == [2, 4, 8, 13]
    assert r.config_of("entry_top")["entry"] == "top" and r.config_of("entry_feedback")["entry"] == "feedback"
    assert r.config_of("rank64")["rank"] == 64 and r.config_of("lora_all")["scope"] == "all"
    assert r.config_of("teacher_lm")["teacher"] == "lm"
    for factor in r.FACTORS:
        names = r.factor_configs(factor)
        assert "default" in names
        for name in names:
            r.config_of(name)


def test_cell_list_seeds_controls_and_uniqueness():
    cells = r.star_cells()
    names = [r.cell_name(*c) for c in cells]
    assert len(names) == len(set(names))
    for pair in r.PAIRS:
        assert by(cells, pair, "base", "base") == [(pair, "base", "base", 0)]
        default_real = by(cells, pair, "default", "real")
        assert sorted(c[3] for c in default_real) == [0, 1]
        for arm in r.CONTROL_ARMS:
            assert sorted(c[3] for c in by(cells, pair, "default", arm)) == [0, 1]
        for name, _, _ in r.STAR:
            star = by(cells, pair, name, "real")
            assert [c[3] for c in star] == [0], name
            assert by(cells, pair, name, "scrambled") == []
    assert len(by(cells, None, None, "ref")) == len(set(m for v in r.REFERENCES.values() for m in v))
    assert len(cells) == 64


def test_pair_ids_and_references():
    assert r.PAIRS["R1"] == {"base": "Qwen/Qwen2.5-1.5B-Instruct", "donor": "Qwen/Qwen2.5-Coder-7B-Instruct"}
    assert r.PAIRS["R2"] == {"base": "Qwen/Qwen2.5-7B-Instruct", "donor": "Qwen/Qwen2.5-Coder-7B-Instruct"}
    assert r.PAIRS["R3"] == {"base": "Qwen/Qwen2.5-7B-Instruct", "donor": "Qwen/Qwen2.5-Coder-32B-Instruct"}
    assert "Qwen/Qwen2.5-Coder-1.5B-Instruct" in r.REFERENCES["R1"]
    assert "Qwen/Qwen2.5-Coder-3B-Instruct" in r.REFERENCES["R1"]
    for pair in r.PAIRS:
        assert r.PAIRS[pair]["base"] in r.REFERENCES[pair]


def test_filters_pairs_factors_seeds_only():
    only_r1 = r.star_cells(["R1"], refs=False)
    assert {c[0] for c in only_r1} == {"R1"}
    a_only = r.star_cells(["R1"], ["a"], refs=False)
    configs = {c[1] for c in a_only if c[2] == "real"}
    assert configs == {"default", "steps500", "steps8000"}
    seed0 = r.star_cells(["R1"], None, [0], refs=False)
    assert {c[3] for c in seed0} == {0}
    assert len(r.select_cells(only_r1, ["real"])) == len([c for c in only_r1 if c[2] == "real"])
    assert {c[2] for c in r.select_cells(only_r1, ["kd_lora"])} == {"kd_lora"}
    assert {c[1] for c in r.select_cells(only_r1, ["size4"])} == {"size4"}


def test_smoke_is_exactly_one_cell_with_200_heal_steps_and_no_controls():
    cells = r.smoke_cells()
    assert cells == [("R1", "default", "real", 0)]
    plan = r.arm_plan("R1", "default", "real", 0, smoke=True)
    assert plan["total_steps"] == 100 + 200
    assert "smoke: bool" in SCRIPT
    assert "mine = todo if smoke" in SCRIPT
    assert r.cell_steps(cells[0]) == r.HEAL["stage1_steps"] + r.DEFAULT_CFG["steps"]


def test_controls_phase_picks_the_best_config_of_each_factor():
    rows = [{"pair": "R1", "config": "default", "arm": "real", "seed": 0, "transfer": 0.3},
            {"pair": "R1", "config": "default", "arm": "real", "seed": 1, "transfer": 0.3},
            {"pair": "R1", "config": "steps8000", "arm": "real", "seed": 0, "transfer": 0.6},
            {"pair": "R1", "config": "steps500", "arm": "real", "seed": 0, "transfer": 0.1},
            {"pair": "R1", "config": "size13", "arm": "real", "seed": 0, "transfer": 0.5},
            {"pair": "R1", "config": "size2", "arm": "real", "seed": 0, "transfer": 0.2},
            {"pair": "R1", "config": "entry_top", "arm": "real", "seed": 0, "transfer": 0.2},
            {"pair": "R1", "config": "rank64", "arm": "real", "seed": 0, "transfer": 0.2},
            {"pair": "R1", "config": "teacher_lm", "arm": "real", "seed": 0, "transfer": 0.9},
            {"pair": "R1", "config": "size13", "arm": "scrambled", "seed": 0}]
    cells = r.best_cells(rows, ["R1"])
    configs = sorted({c[1] for c in cells})
    assert configs == ["size13", "steps8000", "teacher_lm"]
    assert all(c[3] == 0 for c in cells)
    assert ("R1", "size13", "scrambled", 0) not in cells
    assert ("R1", "size13", "random", 0) in cells
    assert ("R1", "steps8000", "kd_lora", 0) in cells
    assert by(cells, "R1", "default") == []


def test_estimate_counts_hours_per_pair():
    out = r.estimate(r.star_cells())
    assert set(out) == {"R1", "R2", "R3", "ref"}
    assert out["R3"]["hours"] > out["R2"]["hours"] > out["R1"]["hours"] > 0


def test_shards_cover_every_cell_once():
    cells = r.star_cells()
    seen = []
    for shard in range(5):
        seen += r.shard_of(cells, shard, 5)
    assert sorted(seen) == sorted(cells)


# ---------------------------------------------------------------- docstring, image, hygiene


def test_docstring_labels_one_seed_and_states_the_go_rule():
    doc = r.__doc__
    assert "ONE SEED" in doc
    assert "Go rule" in doc and "beats kd_lora and the scrambled control" in doc
    assert "more than 3 points" in doc and "within 1" in doc
    assert "no benchmark data trains anything" in doc.lower()
    assert "smallest block" in doc and "shortest heal" in doc


def test_image_is_copied_from_round20_exactly():
    def image_block(text):
        start = text.index("image = (")
        end = text.index("\n)\n", start)
        return text[start:end]

    assert image_block(SCRIPT) == image_block((ROOT / "modal_round20.py").read_text())


def test_no_hard_links_and_reload_only_in_try():
    assert "os.link" not in SCRIPT
    for match in re.finditer(r"volume\.reload\(\)", SCRIPT):
        before = SCRIPT[:match.start()].rstrip().splitlines()[-1].strip()
        assert before == "try:"
    assert "except Exception" in SCRIPT


def test_no_pattern_kills():
    assert "pkill" not in SCRIPT and "killall" not in SCRIPT


def test_every_print_starts_with_a_cell_name_or_is_a_plan_line():
    allowed = ("f\"round20c", "f\"estimate", "label", "name", "f\"{name}", "\"volume reload", "f\"{label}", "cell_name(", "\"verdict ")
    bad = []
    for match in re.finditer(r"print\(", SCRIPT):
        rest = SCRIPT[match.end():match.end() + 40]
        if not rest.startswith(allowed):
            bad.append(rest)
    assert bad == []


def test_script_reads_no_benchmark_for_training():
    assert "load_dataset(" not in SCRIPT
    assert "practice_sets" not in SCRIPT
    ids = r.source_ids.__doc__
    assert "reported benchmark" in ids


# ---------------------------------------------------------------- equal inputs and probe bookkeeping


def test_trained_arms_of_one_config_share_texts_and_steps():
    for pair in r.PAIRS:
        for config in ["default"] + [n for n, _, _ in r.STAR]:
            for seed in (0, 1):
                plans = []
                for arm in ("real", "scrambled", "random", "kd_lora"):
                    plan = r.arm_plan(pair, config, arm, seed)
                    plans.append(plan)
                for plan in plans[1:]:
                    assert plan == plans[0]
                assert plans[0]["total_steps"] == r.HEAL["stage1_steps"] + r.config_of(config)["steps"]
                assert plans[0]["domain_source"] == r.PAIRS[pair]["donor"]


def record_student_batches(monkeypatch, student_ref):
    calls = []
    original = synth.forward_logits

    def spy(model, ids, attn):
        if model is student_ref[0]:
            calls.append(ids.clone())
        return original(model, ids, attn)

    monkeypatch.setattr(synth, "forward_logits", spy)
    return calls


def test_kd_lora_sees_the_same_batches_as_the_real_arm_in_stage_two(monkeypatch):
    tok = tiny_tokenizer()
    domain = sentences(16, 6)
    s1, s2 = 3, 4
    budget = dict(SMALL, stage1_steps=s1, stage2_steps=s2, seed=1, component_tests=False)

    host = tiny_host(seed=0)
    student = [host]
    real_calls = record_student_batches(monkeypatch, student)
    marks = []
    hook = reader.StepHook([0], lambda step: marks.append(len(real_calls)))
    comp = block_comp(host_layer=1, texts=domain)
    with reader.hooked_stage2(synth, hook):
        synth.synthesize((host, tok), [comp], TEXTS, budget, donors=donors_of(tiny_donor()))
    stage2_real = real_calls[marks[0]:]

    host2 = tiny_host(seed=0)
    student[0] = host2
    kd_calls = record_student_batches(monkeypatch, student)
    comps = [{"id": "coder", "donor": "donorA", "skill": "code", "texts": domain}]
    s = synth.settings_of(budget, {})
    enc, _ = synth.prepare_data(tok, comps, TEXTS, s)
    synth.lora_baseline(host2, tok, comps, donors_of(tiny_donor()), enc, s, s1 + s2)
    per_step = len(kd_calls) // (s1 + s2)
    assert per_step >= 2 and len(kd_calls) == per_step * (s1 + s2)
    n = per_step * s2
    assert len(stage2_real) >= n
    for a, b in zip(stage2_real[:n], kd_calls[:n]):
        assert torch.equal(a, b)


def test_probe_checkpoints_of_a_kd_lora_cell_land_on_total_steps():
    host = tiny_host()
    donor = tiny_donor()
    tok = tiny_tokenizer()
    s1, s2 = 3, 4
    ks = reader.checkpoint_steps(s2, (0, 2, 100))
    curve = []
    cache = {}
    held = {"code": sentences(6), "general": sentences(6, 7), "math": sentences(6, 5)}
    view = reader.ContextView(host, synth.lora_off)

    def fire(step):
        curve.append(r.probe_now(host, view, donor, tok, held, cache, step))

    hook = reader.StepHook([s1 + k for k in ks], fire)
    comps = [{"id": "coder", "donor": "donorA", "skill": "code", "texts": sentences(16, 6)}]
    budget = dict(SMALL, stage1_steps=s1, stage2_steps=s2, seed=0)
    s = synth.settings_of(budget, {})
    enc, _ = synth.prepare_data(tok, comps, TEXTS, s)
    with reader.hooked_stage2(synth, hook):
        synth.lora_baseline(host, tok, comps, donors_of(donor), enc, s, s1 + s2)
    assert [p["heal_steps"] for p in curve] == [3, 5]
    assert set(curve[0]["transfer"]) == {"code"}


def test_matched_kd_rank_on_a_tiny_host():
    host = tiny_host()
    cfg = r.config_of("default")
    rank, target = r.kd_rank_for(host, cfg, 48, 1)
    from wsp.lora import params_per_rank

    assert target == reader.glue_params(64, 48) + 16 * params_per_rank(host, layers=[2, 3])
    assert rank * params_per_rank(host) <= target
    big, big_target = r.kd_rank_for(host, r.config_of("rank64"), 48, 1)
    assert big > rank
    everywhere, _ = r.kd_rank_for(host, r.config_of("lora_all"), 48, 1)
    assert everywhere > rank


def test_select_cells_accepts_full_cell_names():
    import modal_round20c as m
    cells = m.star_cells(["R1"], None, None)
    names = ["R1.default.real.s0", "R1.steps8000.real.s0", "R1.base.s0"]
    picked = [m.cell_name(*c) for c in m.select_cells(cells, names)]
    assert sorted(picked) == sorted(names)


def test_arm_plan_accepts_the_base_cell():
    import modal_round20c as m
    plan = m.arm_plan("R1", "base", "base", 0)
    assert plan["config"] == "base"
