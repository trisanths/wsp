"""wsp.reader on tiny random models (CPU)."""

import pytest
import torch

transformers = pytest.importorskip("transformers")

from tests.test_align import tiny_tokenizer
from tests.test_synth import SMALL, TEXTS, block_comp, donors_of, sentences, tiny_donor, tiny_host
from wsp import reader, synth


def test_block_range_keeps_the_relative_depth():
    assert reader.block_range(28, 13) == [8, 20]
    assert reader.block_range(28, 8) == [8, 15]
    assert reader.block_range(28, 2) == [8, 9]
    assert reader.block_range(64, 8) == [18, 25]
    assert reader.block_range(64, 13) == [18, 30]
    assert reader.block_range(10, 9) == [1, 9]
    with pytest.raises(ValueError):
        reader.block_range(4, 5)


def test_entry_layers():
    assert reader.entry_layer("mid", 8, 28, 28) == 8
    assert reader.entry_layer("mid", 18, 64, 28) == 8
    assert reader.entry_layer("top", 8, 28, 28) == 25
    assert reader.entry_layer("feedback", 8, 28, 28) == 7
    assert reader.entry_layer("feedback", 0, 28, 28) == 0
    with pytest.raises(ValueError):
        reader.entry_layer("bottom", 8, 28, 28)
    # the rule is the one wsp.assemble uses
    from wsp.assemble import default_host_layer

    for first, donor_n, host_n in ((8, 28, 28), (8, 28, 24), (18, 64, 28), (0, 28, 28)):
        assert reader.mid_host_layer(first, donor_n, host_n) == default_host_layer(first, donor_n, host_n)


def test_top_entry_leaves_two_host_layers():
    layer = reader.entry_layer("top", 8, 28, 28)
    left = 28 - 1 - layer
    assert left == 2
    assert reader.lora_layer_list("after", layer, 28) == [26, 27]


def test_lora_layer_scopes():
    assert reader.lora_layer_list("after", 8, 12) == [9, 10, 11]
    assert reader.lora_layer_list("all", 8, 4) == [0, 1, 2, 3]
    with pytest.raises(ValueError):
        reader.lora_layer_list("some", 8, 12)


def test_glue_params_match_a_real_branch():
    host = tiny_host()
    donor = tiny_donor()
    tok = tiny_tokenizer()
    s = synth.settings_of(SMALL, {})
    model, branches, _, _ = synth.place(host, tok, [block_comp()], donors_of(donor), s)
    counts = synth.param_counts(model)
    assert counts["glue"] + counts["gates"] == reader.glue_params(64, 48)


def test_matched_rank_and_target():
    host = tiny_host()
    from wsp.lora import params_per_rank

    per_all = params_per_rank(host)
    per_after = params_per_rank(host, layers=[3])
    target = reader.trainable_target(64, 48, per_after, 4)
    assert target == 2 * 64 * 48 + 65 + 4 * per_after
    rank = reader.matched_rank(per_all, target)
    assert rank >= 1
    assert rank * per_all <= target < (rank + 1) * per_all
    assert reader.matched_rank(per_all, 1) == 1


def test_checkpoint_steps():
    assert reader.checkpoint_steps(500) == [0, 250]
    assert reader.checkpoint_steps(2000) == [0, 250, 500, 1000]
    assert reader.checkpoint_steps(8000) == [0, 250, 500, 1000, 2000, 4000]
    assert reader.checkpoint_steps(200, (0, 100, 100, 300)) == [0, 100]
    assert reader.checkpoint_steps(0) == []


def test_step_hook_fires_once_per_step():
    seen = []
    hook = reader.StepHook([0, 2, 2, 5], seen.append)
    losses = []

    def next_loss(step):
        losses.append(step)
        return step

    wrapped = hook.wrap(next_loss)
    for step in range(4):
        assert wrapped(step) == step
    assert seen == [0, 2]
    assert losses == [0, 1, 2, 3]
    assert hook.pending == {5}


def test_forward_counter_restores_the_module_mode():
    module = torch.nn.Linear(2, 2)
    module.train()
    modes = []
    hook = reader.StepHook([1], lambda step: modes.append(module.training))
    counter = reader.ForwardCounter(hook, module)
    assert counter.tick() is False
    assert counter.tick() is True
    assert counter.tick() is False
    assert modes == [False]
    assert module.training is True


def tiny_texts():
    return {"code": sentences(6), "general": sentences(6, 7), "math": sentences(6, 5)}


def test_checkpointed_probe_on_a_tiny_compile_restores_synth():
    host = tiny_host()
    donor = tiny_donor()
    tok = tiny_tokenizer()
    original_run = synth.run_heal
    original_opt = synth.optimize_groups
    cache = {}
    curve = []
    view = reader.ContextView(host, synth.frozen_base)
    held = tiny_texts()
    s1, s2 = 4, 6
    ks = reader.checkpoint_steps(s2, (0, 2, 4, 100))
    assert ks == [0, 2, 4]

    def fire(step):
        curve.append(reader.timed_point(
            lambda: reader.probe_point(host, view, donor, tok, held, ["code"], cache, max_length=16, batch_size=2),
            s1 + step))

    hook = reader.StepHook(ks, fire)
    comp = block_comp(host_layer=1, texts=sentences(12, 6))
    budget = dict(SMALL, stage1_steps=s1, stage2_steps=s2, seed=0, component_tests=False)
    with reader.hooked_stage2(synth, hook):
        model, _, report = synth.synthesize((host, tok), [comp], TEXTS, budget, donors=donors_of(donor))
    assert synth.run_heal is original_run
    assert synth.optimize_groups is original_opt
    assert hook.fired == [0, 2, 4]
    assert [p["heal_steps"] for p in curve] == [4, 6, 8]
    for point in curve:
        assert set(point["transfer"]) == {"code"}
        assert set(point["drift"]) == {"general", "math"}
        assert point["drift"]["general"]["base_kl"] >= 0.0
        assert point["probe_seconds"] >= 0.0
    assert "code" in cache
    # the stage-2 LoRA sits only on the layers after the seam
    names = [name for name, _ in synth.lora_modules(model)]
    assert len(names) > 0
    assert not any(".layers.0." in n or ".layers.1." in n for n in names)
    # a point is the probe of the same model: the base side is constant, the written side is a real number
    assert all(p["transfer"]["code"]["kl_donor_base"] == cache["code"]["kl"] for p in curve)
    assert report["stage2"]["steps"] == s2


def test_lora_on_all_layers_covers_layers_before_the_seam():
    host = tiny_host()
    donor = tiny_donor()
    tok = tiny_tokenizer()
    original = synth.apply_lora
    comp = block_comp(host_layer=2, texts=sentences(12, 6))
    budget = dict(SMALL, stage1_steps=3, stage2_steps=2, seed=0, component_tests=False)
    with reader.lora_on_all_layers(synth):
        model, _, _ = synth.synthesize((host, tok), [comp], TEXTS, budget, donors=donors_of(donor))
    assert synth.apply_lora is original
    names = [name for name, _ in synth.lora_modules(model)]
    assert any(".layers.0." in n for n in names)
    assert any(".layers.3." in n for n in names)


def test_probe_point_of_the_base_is_zero_transfer_and_zero_drift():
    host = tiny_host()
    donor = tiny_donor()
    tok = tiny_tokenizer()
    point = reader.probe_point(host, host, donor, tok, tiny_texts(), ["code"], {}, max_length=16, batch_size=2)
    assert abs(point["transfer"]["code"]["transfer"]) < 1e-9
    assert point["drift"]["general"]["base_kl"] < 1e-9
    assert reader.mean_transfer_of(point) == pytest.approx(0.0, abs=1e-9)


def rows_for(real_transfer, scr_transfer, kd_transfer, real_code, ctrl_code, real_mmlu=0.60, base_mmlu=0.60):
    rows = [{"pair": "R1", "config": "base", "arm": "base", "seed": 0, "transfer": 0.0,
             "scores": {"humaneval": 0.40, "mbpp": 0.50, "mmlu": base_mmlu}}]
    for seed, (r, s, k) in enumerate(zip(real_transfer, scr_transfer, kd_transfer)):
        rows.append({"pair": "R1", "config": "default", "arm": "real", "seed": seed, "transfer": r,
                     "params_added": 100, "cfg": {"size": 8, "steps": 2000},
                     "scores": {"humaneval": real_code, "mbpp": real_code, "mmlu": real_mmlu}})
        rows.append({"pair": "R1", "config": "default", "arm": "scrambled", "seed": seed, "transfer": s,
                     "scores": {"humaneval": ctrl_code, "mbpp": ctrl_code, "mmlu": 0.6}})
        rows.append({"pair": "R1", "config": "default", "arm": "kd_lora", "seed": seed, "transfer": k,
                     "scores": {"humaneval": ctrl_code, "mbpp": ctrl_code, "mmlu": 0.6}})
    return rows


def test_go_check_reads_a_clear_win():
    rows = rows_for([0.50, 0.52], [0.20, 0.22], [0.25, 0.27], 0.60, 0.50)
    out = reader.go_check(rows, "R1")
    assert out["read"] is True
    assert out["spread"] == pytest.approx(0.02)
    assert out["humaneval_over_base"] == pytest.approx(0.20)


def test_go_check_fails_inside_the_seed_spread_or_without_code_gain_or_with_mmlu_loss():
    close = reader.go_check(rows_for([0.30, 0.50], [0.20, 0.22], [0.25, 0.27], 0.60, 0.50), "R1")
    assert close["transfer_ok"] is False and close["read"] is False
    flat = reader.go_check(rows_for([0.50, 0.52], [0.20, 0.22], [0.25, 0.27], 0.51, 0.50), "R1")
    assert flat["code_ok"] is False and flat["read"] is False
    damaged = reader.go_check(rows_for([0.50, 0.52], [0.20, 0.22], [0.25, 0.27], 0.60, 0.50, real_mmlu=0.55), "R1")
    assert damaged["mmlu_ok"] is False and damaged["read"] is False


def test_go_check_reports_missing_rows():
    rows = rows_for([0.5, 0.5], [0.2, 0.2], [0.2, 0.2], 0.6, 0.5)
    rows = [r for r in rows if r["arm"] != "kd_lora"]
    out = reader.go_check(rows, "R1")
    assert out["read"] is None
    assert "kd_lora" in out["missing"]


def test_star_point_uses_default_controls_and_spread_and_is_marked_approximate():
    rows = rows_for([0.50, 0.52], [0.20, 0.22], [0.25, 0.27], 0.60, 0.50)
    rows.append({"pair": "R1", "config": "size4", "arm": "real", "seed": 0, "transfer": 0.45, "params_added": 40,
                 "cfg": {"size": 4, "steps": 2000},
                 "scores": {"humaneval": 0.58, "mbpp": 0.58, "mmlu": 0.60}})
    out = reader.go_check(rows, "R1", "size4")
    assert out["approximate"] is True
    assert out["spread_source"] == "default config seeds"
    assert out["read"] is True
    summary = reader.read_summary(rows, "R1")
    assert summary["smallest_block"]["config"] == "size4"
    assert summary["smallest_block"]["params_added"] == 40


def test_pick_best_takes_the_highest_mean_transfer():
    rows = rows_for([0.5, 0.5], [0.2, 0.2], [0.2, 0.2], 0.6, 0.5)
    rows.append({"pair": "R1", "config": "steps8000", "arm": "real", "seed": 0, "transfer": 0.7})
    rows.append({"pair": "R1", "config": "steps500", "arm": "real", "seed": 0, "transfer": 0.1})
    assert reader.pick_best(rows, "R1", ["steps500", "default", "steps8000"]) == "steps8000"
    assert reader.pick_best(rows, "R1", ["size2"]) is None
