"""wsp.synth on tiny random models (CPU)."""

import copy
import math

import pytest
import torch

transformers = pytest.importorskip("transformers")

from wsp import synth
from wsp.lora import lora_modules
from wsp.synth import (
    BudgetError,
    branches_off,
    encode_items,
    load_synth,
    place,
    preservation_kl,
    stage_trainables,
    synthesize,
    text_loss,
    unembed_map,
)
from tests.test_align import WORDS, tiny_tokenizer

SMALL = dict(steps_in=4, steps_lm=2, stage1_steps=6, stage2_steps=4, batch_size=2, seq_len=16, lora_rank=2,
             batch_in=2)


def tiny_host(layers=4, seed=0):
    torch.manual_seed(seed)
    config = transformers.LlamaConfig(
        hidden_size=64, intermediate_size=128, num_hidden_layers=layers, num_attention_heads=4,
        num_key_value_heads=2, vocab_size=300, max_position_embeddings=64,
    )
    config._attn_implementation = "eager"
    return transformers.LlamaForCausalLM(config).eval()


def tiny_donor(layers=4, seed=3):
    torch.manual_seed(seed)
    config = transformers.Qwen2Config(
        hidden_size=48, intermediate_size=96, num_hidden_layers=layers, num_attention_heads=4,
        num_key_value_heads=2, vocab_size=300, max_position_embeddings=64,
    )
    config._attn_implementation = "eager"
    return transformers.Qwen2ForCausalLM(config).eval()


def sentences(n, width=6):
    out = []
    for i in range(n):
        out.append(" ".join(WORDS[(i + j) % len(WORDS)] for j in range(width)))
    return out


TEXTS = {"align": sentences(40), "general": sentences(24, 7)}


def batch():
    ids = torch.tensor([[0, 0, 5, 9, 4, 7], [3, 8, 6, 5, 9, 4]])
    mask = torch.tensor([[0, 0, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1]])
    return ids, mask


def logits_of(model):
    ids, mask = batch()
    with torch.no_grad():
        return model(input_ids=ids, attention_mask=mask).logits.float()


def block_comp(host_layer=2, skill="s", texts=None, **extra):
    c = {"kind": "block", "donor": "donorA", "layers": [1, 2], "skill": skill, "host_layer": host_layer, "id": "blk"}
    if texts is not None:
        c["texts"] = texts
    c.update(extra)
    return c


def donors_of(model=None):
    return {"donorA": (model if model is not None else tiny_donor(), tiny_tokenizer())}


def settings(**kw):
    s = dict(synth.DEFAULTS)
    s.update(SMALL)
    s.update(kw)
    return s


def staged(host, comps, donors, s):
    """Run place, translate in, translate out init. Return (model, tokenizer, branches, comps, donor_map, data)."""
    tok = tiny_tokenizer()
    model, branches, located, donor_map = place(host, tok, comps, donors, s)
    data, _ = synth.prepare_data(tok, located, TEXTS, s)
    for c, b in zip(located, branches):
        dm, dt = donor_map[c["donor"]]
        synth.translate_in(model, tok, b, c, dm, dt, data["align"], s)
        synth.translate_out(model, tok, b, c, dm, dt, data["align"], s)
    return model, tok, branches, located, donor_map, data


def test_fresh_network_equals_base():
    host = tiny_host()
    expected = logits_of(host)
    comps = [block_comp(), {"kind": "attn_heads", "donor": "donorA", "layer": 1, "indices": [0, 2], "host_layer": 1,
                            "id": "att"},
             {"kind": "mlp_cluster", "donor": "donorA", "layer": 2, "indices": list(range(8)), "host_layer": 3,
              "id": "mlp"}]
    model, branches, _, _ = place(host, tiny_tokenizer(), comps, donors_of(), settings())
    assert len(branches) == 3
    assert torch.equal(logits_of(model), expected)


def test_stage1_trains_only_translations_and_gates():
    host = tiny_host()
    s = settings(stage1_steps=8)
    model, tok, branches, comps, donor_map, data = staged(host, [block_comp()], donors_of(), s)
    before = {k: v.detach().clone() for k, v in model.named_parameters()}
    synth.run_heal(model, tok, comps, branches, donor_map, data, 1, s, 8)
    allowed = set(stage_trainables(model, 1))
    changed = set()
    for k, v in model.named_parameters():
        if not torch.equal(before[k], v):
            changed.add(k)
    assert len(changed) > 0
    assert changed <= allowed
    assert any("w_out" in k for k in changed)
    assert any("token_gate" in k for k in changed)
    assert all(not k.startswith("wsp_grafts.0.part.") for k in allowed)
    assert not any(k.startswith("model.") for k in changed)


def test_stage2_adds_only_lora_after_first_seam():
    host = tiny_host()
    s = settings()
    model, tok, branches, comps, donor_map, data = staged(host, [block_comp(host_layer=1)], donors_of(), s)
    synth.run_heal(model, tok, comps, branches, donor_map, data, 1, s, 3)
    from wsp.lora import apply_lora

    apply_lora(model, 2, layers=[2, 3])
    names = stage_trainables(model, 2)
    lora_names = [n for n in names if "lora_" in n]
    assert len(lora_names) > 0
    for n in lora_names:
        layer = int(n.split("layers.")[1].split(".")[0])
        assert layer > 1
    assert [n for n in stage_trainables(model, 1) if "lora_" in n] == []
    before = {k: v.detach().clone() for k, v in model.named_parameters()}
    synth.run_heal(model, tok, comps, branches, donor_map, data, 2, s, 5)
    changed = {k for k, v in model.named_parameters() if not torch.equal(before[k], v)}
    assert changed <= set(names)
    assert any("lora_B" in k for k in changed)
    for k in changed:
        assert ".base." not in k
        assert not k.startswith("wsp_grafts.0.part.")


def test_preservation_term_zero_at_base():
    host = tiny_host()
    s = settings()
    model, branches, comps, _ = place(host, tiny_tokenizer(), [block_comp()], donors_of(), s)
    enc, _ = encode_items(tiny_tokenizer(), sentences(4), 16)
    ids, attn, lmask = synth.collate(enc, 2, torch.device("cpu"))
    assert float(preservation_kl(model, ids, attn, lmask)) == 0.0
    # With the branch open and T_out nonzero, the term is positive.
    branches[0].ramp = 1.0
    torch.nn.init.normal_(branches[0].w_out.weight, std=0.5)
    assert float(preservation_kl(model, ids, attn, lmask)) > 0.0


def trained_permutation_donor():
    """A 6-layer donor that maps each word to a fixed next word. Layers 0, 1, 4, 5 are identity (zero
    o_proj and down_proj, frozen), so the skill sits only in layers 2 and 3."""
    torch.manual_seed(11)
    config = transformers.LlamaConfig(
        hidden_size=48, intermediate_size=96, num_hidden_layers=6, num_attention_heads=4,
        num_key_value_heads=2, vocab_size=300, max_position_embeddings=64,
    )
    config._attn_implementation = "eager"
    donor = transformers.LlamaForCausalLM(config)
    for i in (0, 1, 4, 5):
        layer = donor.model.layers[i]
        layer.self_attn.o_proj.weight.data.zero_()
        layer.mlp.down_proj.weight.data.zero_()
        for p in layer.parameters():
            p.requires_grad_(False)
    params = [p for p in donor.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=3e-3)
    tok = tiny_tokenizer()
    gen = torch.Generator().manual_seed(0)
    for step in range(250):
        rows = []
        for _ in range(16):
            cur = int(torch.randint(0, 12, (1,), generator=gen))
            seq = [cur]
            for _ in range(6):
                cur = (cur + 5) % 12
                seq.append(cur)
            rows.append([3 + t for t in seq])
        ids = torch.tensor(rows)
        loss = donor(input_ids=ids, labels=ids).loss
        opt.zero_grad()
        loss.backward()
        opt.step()
    donor.eval()
    return donor


def chain_texts(n, seed):
    gen = torch.Generator().manual_seed(seed)
    out = []
    for _ in range(n):
        cur = int(torch.randint(0, 12, (1,), generator=gen))
        seq = [cur]
        for _ in range(6):
            cur = (cur + 5) % 12
            seq.append(cur)
        out.append(" ".join(WORDS[t] for t in seq))
    return out


def test_planted_function_gain_real_over_random():
    donor = trained_permutation_donor()
    host = tiny_host(seed=5)
    # A random host has near-uniform logits at any state; a sharper head lets a branch steer them.
    host.lm_head.weight.data *= 40
    tok = tiny_tokenizer()
    skill_texts = chain_texts(40, 1)
    practice_texts = chain_texts(12, 2)
    enc, _ = encode_items(tok, practice_texts, 16)

    def practice_fn(model, tokenizer):
        return 1.0 / (1.0 + text_loss(model, tokenizer, enc, {"batch_size": 4}))

    comp = {"kind": "block", "donor": "perm", "layers": [2, 3], "skill": "perm", "host_layer": 2, "id": "perm",
            "texts": skill_texts}
    texts = {"align": sentences(40) + skill_texts[:20], "general": sentences(24, 7)}
    s = dict(SMALL)
    s.update(stage1_steps=300, stage2_steps=0, lr_translation=3e-2, lr_gate=1e-2, batch_size=8, seq_len=16,
             steps_in=30, steps_lm=10, batch_in=4)
    model, _, report = synthesize((host, tok), [comp], texts, s, practice={"perm": practice_fn},
                                  donors={"perm": (donor, tok)}, control_run=True)
    row = report["components"][0]
    assert row["content_gain"] is not None
    assert row["content_gain"] > 0.1
    assert row["contribution"] > 0.1
    assert report["scores"]["final"]["perm"] > report["scores"]["base"]["perm"] + 0.1
    assert row["verdict"] == "contributes"
    assert report["random_control"]["scores"]["perm"] < report["scores"]["final"]["perm"]


def test_guard_rolls_back_a_harmful_heal():
    host = tiny_host()
    expected = logits_of(host)
    tok = tiny_tokenizer()
    practice_enc, _ = encode_items(tok, sentences(8, 7), 16)

    def practice_fn(model, tokenizer):
        return math.exp(-text_loss(model, tokenizer, practice_enc, {"batch_size": 4}))

    # Garbage domain text, no preservation term, a huge learning rate: the heal hurts general text.
    garbage = ["hat hat hat hat hat hat hat"] * 12
    comp = block_comp(texts=garbage)
    s = dict(SMALL)
    s.update(stage1_steps=30, stage2_steps=0, pres_weight=0.0, lr_translation=0.5, lr_gate=0.5, kd_weight=0.0,
             lm_weight_kd=1.0, batch_size=4, tolerance=0.0001, out_alpha=20.0)
    model, _, report = synthesize((host, tok), [comp], TEXTS, s, practice={"general": practice_fn},
                                  donors=donors_of())
    events = report["guard"]["events"]
    assert len(events) > 0
    assert any(e["action"].startswith("closed") for e in events)
    assert report["components"][0]["closed"] is True
    assert report["components"][0]["verdict"] == "harmful"
    assert torch.equal(logits_of(model), expected)


def test_library_catalog_rows_compile(tmp_path):
    donor = tiny_donor()
    units_path = tmp_path / "units.pt"
    torch.save({"indices": [1, 4, 9, 16, 25, 30]}, units_path)
    rows = [
        {"id": "mlp_row", "type": "mlp_cluster", "model": "donorA", "layer": 2, "units": str(units_path),
         "num_units": 6, "skill": {"name": "math", "description": "x"}},
        {"id": "block_row", "type": "block", "model": "donorA", "layers": [1, 2], "skill": {"name": "math"}},
    ]
    texts = dict(TEXTS)
    texts["skills"] = {"math": sentences(12, 7)}
    s = settings(stage2_steps=0)
    model, _, report = synthesize((tiny_host(), tiny_tokenizer()), rows, texts, s, donors=donors_of(donor))
    assert [p["kind"] for p in report["place"]] == ["mlp_cluster", "block"]
    assert [c["id"] for c in report["components"]] == ["mlp_row", "block_row"]
    assert report["place"][0]["part_params"] > 0


def test_save_and_load_equal_logits(tmp_path):
    host = tiny_host()
    s = settings()
    comps = [block_comp(),
             {"kind": "attn_heads", "donor": "donorA", "layer": 1, "indices": [0, 2], "host_layer": 1, "id": "att"}]
    model, tok, report = synthesize((host, tiny_tokenizer()), comps, TEXTS, s, donors=donors_of(),
                                    out_dir=str(tmp_path / "out"))
    expected = logits_of(model)
    loaded, tokenizer, saved_report = load_synth(str(tmp_path / "out"))
    assert len(loaded.wsp_grafts) == 2
    assert len(lora_modules(loaded)) == len(lora_modules(model))
    _, mask = batch()
    real = mask.bool()
    assert torch.allclose(logits_of(loaded)[real], expected[real], atol=1e-5)
    assert saved_report["params"]["total"] == report["params"]["total"]
    assert not torch.equal(expected, logits_of(tiny_host()))


@pytest.mark.parametrize("kind", ["block", "attn_heads", "mlp_cluster"])
def test_kv_cache_generation_equals_uncached(kind):
    host = tiny_host()
    if kind == "block":
        comp = block_comp()
    elif kind == "attn_heads":
        comp = {"kind": "attn_heads", "donor": "donorA", "layer": 1, "indices": [0, 2], "host_layer": 1, "id": "a"}
    else:
        comp = {"kind": "mlp_cluster", "donor": "donorA", "layer": 1, "indices": list(range(10)), "host_layer": 1,
                "id": "m"}
    s = settings()
    model, tok, branches, comps, donor_map, data = staged(host, [comp], donors_of(), s)
    with torch.no_grad():
        torch.nn.init.normal_(branches[0].w_out.weight, std=0.3)
    branches[0].ramp = 1.0
    ids, mask = batch()
    kw = dict(max_new_tokens=8, do_sample=False, pad_token_id=0, eos_token_id=None)
    with torch.no_grad():
        cached = model.generate(input_ids=ids, attention_mask=mask, use_cache=True, **kw)
        uncached = model.generate(input_ids=ids, attention_mask=mask, use_cache=False, **kw)
    assert torch.equal(cached, uncached)
    with branches_off(model):
        with torch.no_grad():
            off = model.generate(input_ids=ids, attention_mask=mask, use_cache=True, **kw)
    assert not torch.equal(cached, off)


def test_branch_delta_is_zero_at_first_real_position():
    host = tiny_host()
    s = settings()
    model, tok, branches, comps, donor_map, data = staged(host, [block_comp()], donors_of(), s)
    with torch.no_grad():
        torch.nn.init.normal_(branches[0].w_out.weight, std=0.3)
    branches[0].ramp = 1.0
    ids, mask = batch()
    store = {}
    layer = model.model.layers[2]

    def grab(m, a, o):
        store["x"] = (o[0] if isinstance(o, tuple) else o).detach()

    # Compare the open and closed network at layer 2: the first real position must not change.
    h = layer.register_forward_hook(grab)
    with torch.no_grad():
        model(input_ids=ids, attention_mask=mask)
        on = store["x"].clone()
        with branches_off(model):
            model(input_ids=ids, attention_mask=mask)
            off = store["x"].clone()
    h.remove()
    assert torch.equal(on[0, 2], off[0, 2])
    assert torch.equal(on[1, 0], off[1, 0])
    assert not torch.equal(on[1, 3], off[1, 3])


def test_budget_refusal_reports_numbers():
    host = tiny_host()
    with pytest.raises(BudgetError) as err:
        synthesize((host, tiny_tokenizer()), [block_comp()], TEXTS, settings(max_params=1000),
                   donors=donors_of())
    assert "params total" in str(err.value)
    assert "limit 1000" in str(err.value)
    host = tiny_host()
    with pytest.raises(BudgetError) as err:
        synthesize((host, tiny_tokenizer()), [block_comp()], TEXTS, settings(max_params_per_token=10),
                   donors=donors_of())
    assert "per token" in str(err.value)


def test_unembed_map_reproduces_donor_logits_up_to_least_squares_error():
    torch.manual_seed(0)
    host = tiny_host()
    donor = tiny_donor()
    res = unembed_map(host, donor, lam=1e-8)
    m = res["M"]
    assert m.shape == (64, 48)
    e_h = synth.effective_unembedding(host, torch.device("cpu"), torch.full((300,), 1 / 300))
    e_d = synth.effective_unembedding(donor, torch.device("cpu"), torch.full((300,), 1 / 300))
    err = float(((e_h @ m - e_d) ** 2).sum() / (e_d ** 2).sum())
    assert abs((1 - err) - res["capture"]) < 1e-4
    # No other map does better, in the least-squares sense.
    other = m + 0.01 * torch.randn_like(m)
    assert float(((e_h @ other - e_d) ** 2).sum()) > float(((e_h @ m - e_d) ** 2).sum())
    # A donor whose unembedding lies in the host's column space is reproduced exactly.
    donor2 = tiny_donor(seed=9)
    mix = torch.randn(64, 48)
    with torch.no_grad():
        donor2.get_output_embeddings().weight.copy_(host.get_output_embeddings().weight @ mix)
        donor2.model.norm.weight.fill_(1.0)
        host.model.norm.weight.fill_(1.0)
    exact = unembed_map(host, donor2, lam=1e-9)
    assert exact["capture"] > 0.999
    ids, attn = batch()
    z_d = torch.randn(5, 48)
    u_d = z_d / z_d.pow(2).mean(-1, keepdim=True).sqrt()
    logit_d = u_d @ donor2.get_output_embeddings().weight.T
    logit_h = (u_d @ exact["M"].T) @ host.get_output_embeddings().weight.T
    assert torch.allclose(logit_h - logit_h.mean(-1, keepdim=True), logit_d - logit_d.mean(-1, keepdim=True),
                          atol=1e-2 * float(logit_d.abs().max()))


def test_old_output_init_is_zero_and_unembed_init_is_used_near_top():
    host = tiny_host()
    s = settings(output_init="old")
    model, tok, branches, comps, donor_map, data = staged(host, [block_comp()], donors_of(), s)
    assert float(branches[0].w_out.weight.abs().sum()) == 0.0
    host = tiny_host()
    donor = tiny_donor()
    comp = block_comp(layers=[2, 3])
    s = settings()
    tok = tiny_tokenizer()
    model, branches, located, donor_map = place(host, tok, [comp], {"donorA": (donor, tok)}, s)
    data, _ = synth.prepare_data(tok, located, TEXTS, s)
    out = synth.translate_out(model, tok, branches[0], located[0], donor, tok, data["align"], s)
    assert out["init"] == "unembed_map"
    assert 0.0 < out["logit_capture"] <= 1.0
    out = synth.translate_out(model, tok, branches[0], located[0], donor, tok, data["align"],
                              dict(s, near_top=0))
    assert out["init"].startswith("missing_residual")
    assert float(branches[0].w_out.weight.abs().sum()) > 0.0


def test_missing_residual_fit_without_shared_tokenizer_uses_gold_labels():
    host = tiny_host()
    donor = tiny_donor()
    tok = tiny_tokenizer()
    other_tok = tiny_tokenizer()
    vocab = dict(other_tok.get_vocab())
    s = settings()
    model, branches, located, donor_map = place(host, tok, [block_comp()], {"donorA": (donor, tok)}, s)
    data, _ = synth.prepare_data(tok, located, TEXTS, s)
    enc, _ = encode_items(tok, data["align"], 16, truncate=True)
    synth.translate_in(model, tok, branches[0], located[0], donor, tok, data["align"], s)
    res = synth.missing_residual_fit(model, tok, branches[0], located[0], donor, tok, enc, False, s)
    assert res["fit_rows"] > 20


def test_stage2_failure_rolls_back_to_stage1_and_removes_lora():
    host = tiny_host()
    tok = tiny_tokenizer()

    def practice_fn(model, tokenizer):
        # Passes at the base and after stage 1, fails once a LoRA is on the network.
        return 0.1 if len(lora_modules(model)) > 0 else 0.5

    s = settings(stage1_steps=4, stage2_steps=4)
    model, _, report = synthesize((host, tok), [block_comp()], TEXTS, s, practice={"p": practice_fn},
                                  donors=donors_of())
    events = report["guard"]["events"]
    assert events[0]["stage"] == 2
    assert events[0]["action"] == "rolled back to stage 1"
    assert report["guard"]["final_stage"] == "stage1"
    assert len(lora_modules(model)) == 0
    assert report["params"]["lora"] == 0


def test_lora_baseline_trains_only_lora_and_components_per_step_cycles():
    host = tiny_host()
    tok = tiny_tokenizer()
    donor = tiny_donor()
    base_state = {k: v.clone() for k, v in host.state_dict().items()}
    comps = [{"id": "a", "donor": "donorA", "skill": "s", "texts": sentences(12, 7)},
             {"id": "b", "donor": "donorA", "skill": "s", "texts": sentences(12, 6)}]
    s = synth.settings_of(dict(SMALL, kd=True, components_per_step=1), {})
    data, _ = synth.prepare_data(tok, comps, TEXTS, s)
    out = synth.lora_baseline(host, tok, comps, {"donorA": (donor, tok)}, data, s, 4)
    assert out["steps"] == 4
    assert out["lora_delta_ratio"] > 0.0
    assert len(lora_modules(host)) > 0
    for name, value in host.state_dict().items():
        if "lora_" in name:
            continue
        assert torch.equal(value, base_state[name.replace(".base.", ".")]), name
    counts = synth.param_counts(host)
    assert counts["lora"] > 0 and counts["parts"] == 0


def test_control_block_range_takes_least_overlap_when_no_disjoint_range_exists():
    # 13 of 28 layers: no disjoint range. The old fallback gave (11, 23) at seed 1, which shares 10 layers.
    for seed in (0, 1, 2):
        start, end, overlap = synth.control_block_range(28, 8, 20, seed)
        assert end - start == 12
        assert start != 8
        assert overlap == 5
        assert max(0, min(end, 20) - max(start, 8) + 1) == overlap
    start, end, overlap = synth.control_block_range(28, 8, 17, 0)
    assert overlap == 0


def test_control_gets_the_output_init_of_the_real_block():
    # Real block [0, 1] of 4 layers is not near the top (near_top=1), the control range (2, 3) is.
    host = tiny_host()
    donor = tiny_donor()
    tok = tiny_tokenizer()
    s = settings(near_top=1)
    comp = block_comp(layers=[0, 1], control="random", control_seed=0)
    model, branches, located, donor_map = place(host, tok, [comp], {"donorA": (donor, tok)}, s)
    assert located[0]["layers"][1] == 3
    assert located[0]["real_last"] == 1
    data, _ = synth.prepare_data(tok, located, TEXTS, s)
    synth.translate_in(model, tok, branches[0], located[0], donor, tok, data["align"], s)
    out = synth.translate_out(model, tok, branches[0], located[0], donor, tok, data["align"], s)
    assert out["init"].startswith("missing_residual")


def test_translate_in_reports_the_error_of_the_final_stitch(monkeypatch):
    host = tiny_host()
    s = settings(steps_lm=2)
    tok = tiny_tokenizer()
    model, branches, located, donor_map = place(host, tok, [block_comp()], donors_of(), s)
    data, _ = synth.prepare_data(tok, located, TEXTS, s)
    dm, dt = donor_map["donorA"]
    out = synth.translate_in(model, tok, branches[0], located[0], dm, dt, data["align"], s)
    assert isinstance(out["rel_err_final"], float)
    assert out["rel_err_final"] > 0.0


def test_translate_out_without_a_shared_tokenizer_keeps_zero_init(monkeypatch):
    host = tiny_host()
    donor = tiny_donor()
    tok = tiny_tokenizer()
    s = settings(near_top=0)
    model, branches, located, donor_map = place(host, tok, [block_comp()], {"donorA": (donor, tok)}, s)
    data, _ = synth.prepare_data(tok, located, TEXTS, s)
    monkeypatch.setattr(synth, "same_vocab", lambda a, b: False)
    out = synth.translate_out(model, tok, branches[0], located[0], donor, tok, data["align"], s)
    assert out["init"] == "zero_no_teacher"
    assert float(branches[0].w_out.weight.abs().sum()) == 0.0


def test_empty_guard_never_closes_gates():
    host = tiny_host()
    tok = tiny_tokenizer()
    s = settings(stage1_steps=2, stage2_steps=0)
    model, _, report = synthesize((host, tok), [block_comp()], TEXTS, s, practice={"p": lambda m, t: 0.5},
                                  guard=[], donors=donors_of())
    assert report["guard"]["events"] == []


def test_round20_components_use_the_cell_seed_and_screened_rows(tmp_path):
    pytest.importorskip("modal")
    import modal_round20 as r20

    data = {"domain": {"math": [], "big": [], "coder": []}}
    rows = [{"id": "r", "type": "mlp_cluster", "model": r20.MATH_BASE, "layer": 3, "indices": [1, 2]}]
    for pair, arm in (("p1", "synth"), ("p2", "synth"), ("p3", "synth"), ("p1", "synth_random")):
        comps = r20.components_for(pair, arm, data, 1, rows)
        for c in comps:
            assert c["control_seed"] == 1
            assert (c.get("control") == "random") == (arm == "synth_random")
    catalog = tmp_path / "catalog.jsonl"
    catalog.write_text("")
    assert r20.library_rows(str(catalog)) == []
