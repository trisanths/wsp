import copy

import pytest
import torch

transformers = pytest.importorskip("transformers")

from wsp import pgraft
from wsp.stitch import encode, feed_hooks, layer_outputs, remove_hooks
from tests.test_align import tiny_tokenizer, WORDS

TEXTS = []
for i in range(24):
    words = [WORDS[(i * 7 + j * 3) % len(WORDS)] for j in range(10 + i % 5)]
    TEXTS.append(" ".join(words))


def tiny_host():
    torch.manual_seed(1)
    config = transformers.LlamaConfig(hidden_size=32, intermediate_size=64, num_hidden_layers=4,
                                      num_attention_heads=4, num_key_value_heads=2, vocab_size=300,
                                      max_position_embeddings=128)
    config._attn_implementation = "eager"
    return transformers.LlamaForCausalLM(config).eval()


def tiny_donor():
    torch.manual_seed(2)
    config = transformers.Qwen2Config(hidden_size=48, intermediate_size=96, num_hidden_layers=6,
                                      num_attention_heads=4, num_key_value_heads=2, vocab_size=300,
                                      max_position_embeddings=128)
    config._attn_implementation = "eager"
    model = transformers.Qwen2ForCausalLM(config).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def setup(control=None, first=2, last=4):
    base = tiny_host()
    for p in base.parameters():
        p.requires_grad_(False)
    donor = tiny_donor()
    tok = tiny_tokenizer()
    graft = pgraft.build(base, donor, host_layer=1, first=first, last=last, control=control, enc_depth=2)
    pgraft.attach(base, [graft])
    pgraft.set_sink(graft, donor, tok, first, TEXTS)
    return base, donor, tok, graft


def logits(model, tok, texts):
    ids, mask, _, _ = encode(tok, texts, 32)
    with torch.no_grad():
        return model(input_ids=ids, attention_mask=mask).logits


def test_fresh_graft_is_the_base():
    base, donor, tok, graft = setup()
    plain = tiny_host()
    assert torch.allclose(logits(base, tok, TEXTS[:4]), logits(plain, tok, TEXTS[:4]), atol=1e-6)


def test_identity_control_has_the_same_budget():
    _, _, _, real = setup()
    _, _, _, ident = setup(control="identity")
    real_n = pgraft.count(pgraft.trainable([real]))
    ident_n = pgraft.count(pgraft.trainable([ident]))
    assert real_n == ident_n
    # The identity control holds no donor weights.
    assert sum(p.numel() for p in ident.part.parameters()) == 0


def test_block_matches_the_donor_run_on_the_same_input():
    base, donor, tok, graft = setup(first=2, last=4)
    ids, mask, _, _ = encode(tok, TEXTS[:3], 32)
    torch.manual_seed(0)
    x = torch.randn(ids.shape[0], ids.shape[1], 48)
    graft.pad_mask = mask
    with torch.no_grad():
        block_out = graft.contribution(x) + x
        holder = {"x": x}
        handles = feed_hooks(donor, 2, None, holder)
        try:
            found = layer_outputs(donor, [4], ids, mask)
        finally:
            remove_hooks(handles)
    graft.pad_mask = None
    keep = mask.bool()
    assert torch.allclose(block_out[keep], found[4][keep], atol=1e-4)


def test_alignment_lowers_kl_and_reports_survival():
    base, donor, tok, graft = setup()
    cfg = {"align_steps": 60, "geo_steps": 10, "align_lr": 3e-3, "seq_len": 32, "donor_seq_len": 32}
    before = pgraft.alignment_report(base, tok, donor, tok, graft, TEXTS[16:], cfg)
    pgraft.align(base, tok, donor, tok, graft, TEXTS[:16], cfg, log_every=0)
    after = pgraft.alignment_report(base, tok, donor, tok, graft, TEXTS[16:], cfg)
    assert after["kl"] < before["kl"]
    assert 0.0 <= after["survival"]
    assert after["pairs"] > 0


def test_integration_trains_and_off_restores_the_base():
    for control in (None, "identity"):
        base, donor, tok, graft = setup(control=control)
        pgraft.calibrate_scale(base, tok, graft, TEXTS)
        params = pgraft.trainable([graft], enc=control == "identity")
        cfg = {"int_steps": 12, "int_batch": 4, "seq_len": 32, "drop_layers": 1, "drop_prob": 1.0,
               "domain_frac": 0.5}
        log = pgraft.integrate(base, tok, TEXTS[:12], TEXTS[12:], params, cfg, log_every=0)
        assert len(log) == 12
        assert any("gain" in row for row in log)
        assert graft.dec.up.weight.abs().sum() > 0
        plain = tiny_host()
        with pgraft.grafts_off(base):
            assert torch.allclose(logits(base, tok, TEXTS[:4]), logits(plain, tok, TEXTS[:4]), atol=1e-6)
        util = pgraft.heldout_utility(base, tok, TEXTS[:8], cfg)
        assert set(util) >= {"on", "off", "gain", "gate"}


def test_preservation_and_stage2_enc():
    base, donor, tok, graft = setup()
    pgraft.calibrate_scale(base, tok, graft, TEXTS)
    ref = tiny_host()
    enc_before = graft.enc.proj.weight.detach().clone()
    cfg = {"int_steps": 8, "int_batch": 4, "seq_len": 32, "domain_frac": 0.5, "preserve_weight": 1.0}
    log = pgraft.integrate(base, tok, TEXTS[:12], TEXTS[12:], pgraft.trainable([graft], enc=False), cfg,
                           log_every=0, ref=ref, enc_params=list(graft.enc.parameters()))
    assert any("drift" in row for row in log)
    assert not torch.equal(enc_before, graft.enc.proj.weight)


def test_stage2_regularizer_probe_and_contribution_metrics():
    base, donor, tok, graft = setup()
    pgraft.calibrate_scale(base, tok, graft, TEXTS)
    ref = tiny_host()
    acfg = {"seq_len": 32, "donor_seq_len": 32}
    calls = []

    def reg_fn(step):
        loss, _ = pgraft.functional_loss(base, tok, donor, tok, graft, TEXTS[:4], acfg)
        return 0.5 * loss

    def probe_fn(step):
        calls.append(step)
        return pgraft.alignment_report(base, tok, donor, tok, graft, TEXTS[16:20], acfg)

    cfg = {"int_steps": 6, "int_batch": 4, "seq_len": 32, "preserve_weight": 1.0}
    log = pgraft.integrate(base, tok, TEXTS[:12], TEXTS[12:], pgraft.trainable([graft], enc=False), cfg,
                           log_every=0, ref=ref, enc_params=list(graft.enc.parameters()),
                           reg_fn=reg_fn, probe_fn=probe_fn, probe_every=3)
    assert calls == [0, 3, 5]
    assert all("reg" in row for row in log)
    util = pgraft.heldout_utility(base, tok, TEXTS[:8], cfg, ref=ref)
    assert util["update_ratio"] > 0
    assert util["drift_kl"] >= 0
    with pgraft.grafts_off(base):
        off = pgraft.heldout_utility(base, tok, TEXTS[:8], cfg, ref=ref)
    assert off["drift_kl"] < 1e-6


def test_kd_and_mc_losses_train_the_graft():
    base, donor, tok, graft = setup()
    pgraft.calibrate_scale(base, tok, graft, TEXTS)
    vmap = pgraft.vocab_map(tok, tok)
    assert vmap[0].numel() == len(tok.get_vocab()) - 3
    cfg = {"seq_len": 32, "donor_seq_len": 32}
    before = pgraft.kd_report(base, tok, donor, tok, TEXTS[16:], vmap, cfg)
    params = pgraft.trainable([graft], enc=True)

    def step_fn(step):
        return pgraft.kd_loss(base, tok, donor, tok, TEXTS[:8], vmap, cfg)

    pgraft.train(base, [{"params": params, "lr": 3e-3}], 30, step_fn, log_every=0)
    after = pgraft.kd_report(base, tok, donor, tok, TEXTS[16:], vmap, cfg)
    assert after["kl"] < before["kl"]
    items = [(" ".join(TEXTS[i].split()[:6]), i % 4) for i in range(8)]
    letter_ids = [5, 6, 7, 8]
    loss, info = pgraft.mc_loss(base, tok, items, letter_ids, torch.device("cpu"))
    loss.backward()
    assert graft.dec.up.weight.grad is not None


def test_bundle_of_subnetworks_aligns_and_matches_native():
    base = tiny_host()
    for p in base.parameters():
        p.requires_grad_(False)
    donor = tiny_donor()
    tok = tiny_tokenizer()
    specs = pgraft.expand_specs([{"kind": "mlp_cluster", "layer": 2, "indices": [0, 5, 9]},
                                 {"kind": "attn_heads", "layer": 3, "heads": [1]},
                                 {"kind": "mlp_layer", "layer": 4}], donor.config)
    assert len(specs[2]["indices"]) == 96
    rand = pgraft.expand_specs(specs, donor.config, control="random", seed=1)
    assert rand[0]["indices"] != specs[0]["indices"] and len(rand[0]["indices"]) == 3
    graft = pgraft.build_bundle_graft(base, donor, 1, specs, enc_depth=3)
    pgraft.attach(base, [graft])
    pgraft.set_sink(graft, donor, tok, 2, TEXTS)
    pgraft.bundle_input_rms(graft.part, donor, tok, TEXTS, {"donor_seq_len": 32})
    plain = tiny_host()
    assert torch.allclose(logits(base, tok, TEXTS[:4]), logits(plain, tok, TEXTS[:4]), atol=1e-6)
    cfg = {"seq_len": 32, "donor_seq_len": 32, "align_steps": 40, "align_lr": 3e-3}
    before = pgraft.part_report(base, tok, donor, tok, graft, TEXTS[16:], cfg)
    pgraft.align_parts(base, tok, donor, tok, graft, TEXTS[:16], cfg, log_every=0)
    after = pgraft.part_report(base, tok, donor, tok, graft, TEXTS[16:], cfg)
    assert sum(after.values()) > sum(before.values())
    # The bundle runs inside the base and the branch trains.
    pgraft.calibrate_scale(base, tok, graft, TEXTS)
    vmap = pgraft.vocab_map(tok, tok)
    params = pgraft.trainable([graft], enc=True)
    pgraft.train(base, [{"params": params, "lr": 1e-3}], 5,
                 lambda s: pgraft.kd_loss(base, tok, donor, tok, TEXTS[:4], vmap, cfg), log_every=0)
    assert graft.dec.up.weight.abs().sum() > 0


def test_trunk_bundle_reads_parts_in_native_context():
    base = tiny_host()
    for p in base.parameters():
        p.requires_grad_(False)
    donor = tiny_donor()
    tok = tiny_tokenizer()
    specs = [{"kind": "mlp_cluster", "layer": 3, "indices": [1, 2]}, {"kind": "attn_heads", "layer": 4, "heads": [0]}]
    graft = pgraft.build_trunk_bundle_graft(base, donor, 1, 2, specs)
    pgraft.attach(base, [graft])
    ids, mask, _, _ = encode(tok, TEXTS[:3], 32)
    torch.manual_seed(0)
    x = torch.randn(ids.shape[0], ids.shape[1], 48)
    # The parts see exactly the donor's own states when the donor is fed x at layer 2.
    from wsp.pgraft import native_part_inputs
    holder = {"x": x}
    handles = feed_hooks(donor, 2, None, holder)
    try:
        native = native_part_inputs(donor, specs, ids, mask)
    finally:
        remove_hooks(handles)
    graft.pad_mask = mask
    with torch.no_grad():
        got = graft.contribution(x)
        want = graft.part.parts[0](native[0]).float() + graft.part.parts[1](native[1], attention_mask=mask).float()
    graft.pad_mask = None
    keep = mask.bool()
    assert torch.allclose(got[keep], want[keep], atol=1e-4)
    cfg = {"align_steps": 5, "geo_steps": 2, "seq_len": 32, "donor_seq_len": 32}
    pgraft.set_sink(graft, donor, tok, 2, TEXTS)
    pgraft.align(base, tok, donor, tok, graft, TEXTS[:8], cfg, log_every=0)


def test_mc_soft_loss_matches_ce_without_soft_and_trains_with_it():
    base, donor, tok, graft = setup()
    items = [(" ".join(TEXTS[i].split()[:6]), i % 4) for i in range(8)]
    letter_ids = [5, 6, 7, 8]
    dev = torch.device("cpu")
    plain, _ = pgraft.mc_loss(base, tok, items, letter_ids, dev)
    same, info = pgraft.mc_soft_loss(base, tok, items, letter_ids, dev)
    assert abs(float(plain) - float(same)) < 1e-5 and info["tokens"] > 0
    soft = torch.softmax(torch.randn(8, 4), -1)
    loss, info = pgraft.mc_soft_loss(base, tok, items, letter_ids, dev, soft=soft, kd_weight=1.0)
    assert info["kd"] >= 0 and float(loss) >= float(same) - 1e-6
    loss.backward()
    assert graft.dec.up.weight.grad is not None


def test_mc_soft_loss_runs_in_bf16():
    base = tiny_host().to(torch.bfloat16)
    tok = tiny_tokenizer()
    items = [(" ".join(TEXTS[i].split()[:6]), i % 4) for i in range(4)]
    soft = torch.softmax(torch.randn(4, 4), -1)
    loss, _ = pgraft.mc_soft_loss(base, tok, items, [5, 6, 7, 8], torch.device("cpu"), soft=soft, kd_weight=1.0)
    assert torch.isfinite(loss)

def split_setup(write_layer, dtype=torch.float32):
    base = tiny_host().to(dtype)
    for p in base.parameters():
        p.requires_grad_(False)
    donor = tiny_donor().to(dtype)
    tok = tiny_tokenizer()
    graft = pgraft.build(base, donor, host_layer=1, first=2, last=4, enc_depth=2, write_layer=write_layer)
    pgraft.set_sink(graft, donor, tok, 2, TEXTS)
    return base, tok, graft


def test_split_graft_writes_late_and_off_restores_base():
    base, tok, graft = split_setup(3)
    assert isinstance(graft, pgraft.SplitGraft)
    pgraft.attach_split(base, graft)
    plain = tiny_host()
    assert torch.allclose(logits(base, tok, TEXTS[:4]), logits(plain, tok, TEXTS[:4]), atol=1e-6)
    torch.nn.init.normal_(graft.dec.up.weight, std=0.05)
    on = logits(base, tok, TEXTS[:4])
    assert not torch.allclose(on, logits(plain, tok, TEXTS[:4]), atol=1e-4)
    with pgraft.grafts_off(base):
        off = logits(base, tok, TEXTS[:4])
    assert torch.allclose(off, logits(plain, tok, TEXTS[:4]), atol=1e-6)
    assert graft.pending is None


def test_split_graft_at_host_layer_matches_parallel_and_trains_bf16():
    base, tok, graft = split_setup(1)
    assert type(graft) is pgraft.ParallelGraft
    base2, tok2, graft2 = split_setup(3, dtype=torch.bfloat16)
    pgraft.attach_split(base2, graft2)
    ids, mask, _, _ = encode(tok2, TEXTS[:4], 32)
    out = base2(input_ids=ids, attention_mask=mask).logits
    loss = out.float().logsumexp(-1).mean()
    loss.backward()
    assert graft2.dec.up.weight.grad is not None
    assert torch.isfinite(loss)


def test_split_graft_generate_with_cache_matches_no_cache():
    base, tok, graft = split_setup(3)
    pgraft.attach_split(base, graft)
    torch.nn.init.normal_(graft.dec.up.weight, std=0.05)
    ids, mask, _, _ = encode(tok, TEXTS[:2], 12)
    cached = base.generate(input_ids=ids, attention_mask=mask, max_new_tokens=4, do_sample=False)
    seq = ids
    m = mask
    for _ in range(4):
        nxt = base(input_ids=seq, attention_mask=m, use_cache=False).logits[:, -1].argmax(-1, keepdim=True)
        seq = torch.cat([seq, nxt], dim=1)
        m = torch.cat([m, torch.ones_like(nxt)], dim=1)
    assert torch.equal(cached, seq)


def test_out_scale_calibration_scales_the_update():
    base, tok, graft = split_setup(3)
    pgraft.attach_split(base, graft)
    value = pgraft.calibrate_out_scale(base, tok, graft, TEXTS, 3, frac=0.5)
    assert value > 0 and abs(float(graft.out_scale) - 0.5 * value) < 1e-5
    torch.nn.init.normal_(graft.dec.up.weight, std=0.05)
    ids, mask, _, _ = encode(tok, TEXTS[:3], 32)
    with torch.no_grad():
        a = base(input_ids=ids, attention_mask=mask).logits
        graft.out_scale.fill_(0.0)
        b = base(input_ids=ids, attention_mask=mask).logits
    plain = tiny_host()
    with torch.no_grad():
        c = plain(input_ids=ids, attention_mask=mask).logits
    assert torch.allclose(b, c, atol=1e-6) and not torch.allclose(a, c, atol=1e-4)


def logit_setup(control=None, dtype=torch.float32):
    base = tiny_host().to(dtype)
    for p in base.parameters():
        p.requires_grad_(False)
    donor = tiny_donor().to(dtype)
    tok = tiny_tokenizer()
    graft = pgraft.build(base, donor, host_layer=1, first=2, last=5, control=control, enc_depth=2, logit=True)
    pgraft.attach_logit(base, graft)
    if control != "identity":
        pgraft.set_sink(graft, donor, tok, 2, TEXTS)
    return base, donor, tok, graft


def test_logit_graft_mixes_and_off_restores_base():
    base, donor, tok, graft = logit_setup()
    assert isinstance(graft, pgraft.LogitGraft)
    plain = tiny_host()
    on = logits(base, tok, TEXTS[:4])
    assert not torch.allclose(on, logits(plain, tok, TEXTS[:4]), atol=1e-4)
    with pgraft.grafts_off(base):
        off = logits(base, tok, TEXTS[:4])
    assert torch.allclose(off, logits(plain, tok, TEXTS[:4]), atol=1e-6)
    graft.token_gate.bias.data.fill_(-30.0)
    assert torch.allclose(logits(base, tok, TEXTS[:4]), logits(plain, tok, TEXTS[:4]), atol=1e-4)


def test_logit_graft_trains_enc_and_gate_in_bf16_and_identity_control_runs():
    base, donor, tok, graft = logit_setup(dtype=torch.bfloat16)
    ids, mask, _, _ = encode(tok, TEXTS[:4], 32)
    loss = base(input_ids=ids, attention_mask=mask).logits.float().logsumexp(-1).mean()
    loss.backward()
    assert graft.enc.proj.weight.grad is not None and graft.token_gate.weight.grad is not None
    assert graft.dec.up.weight.grad is None
    base2, _, tok2, ident = logit_setup(control="identity")
    assert torch.isfinite(logits(base2, tok2, TEXTS[:2])).all()


def test_logit_graft_generate_with_cache_matches_no_cache():
    base, donor, tok, graft = logit_setup()
    ids, mask, _, _ = encode(tok, TEXTS[:2], 12)
    cached = base.generate(input_ids=ids, attention_mask=mask, max_new_tokens=4, do_sample=False)
    seq, m = ids, mask
    for _ in range(4):
        nxt = base(input_ids=seq, attention_mask=m, use_cache=False).logits[:, -1].argmax(-1, keepdim=True)
        seq = torch.cat([seq, nxt], dim=1)
        m = torch.cat([m, torch.ones_like(nxt)], dim=1)
    assert torch.equal(cached, seq)


def test_graft_reads_a_mix_of_earlier_host_layers():
    base = tiny_host()
    for p in base.parameters():
        p.requires_grad_(False)
    donor = tiny_donor()
    tok = tiny_tokenizer()
    graft = pgraft.build(base, donor, host_layer=2, first=2, last=4, enc_depth=2, read_layers=[0, 1])
    assert graft.enc.host_layers == [0, 1]
    pgraft.attach(base, [graft])
    pgraft.set_sink(graft, donor, tok, 2, TEXTS)
    plain = tiny_host()
    assert torch.allclose(logits(base, tok, TEXTS[:4]), logits(plain, tok, TEXTS[:4]), atol=1e-6)
    cfg = {"align_steps": 30, "geo_steps": 5, "align_lr": 3e-3, "seq_len": 32, "donor_seq_len": 32}
    before = pgraft.alignment_report(base, tok, donor, tok, graft, TEXTS[16:], cfg)
    pgraft.align(base, tok, donor, tok, graft, TEXTS[:16], cfg, log_every=0)
    after = pgraft.alignment_report(base, tok, donor, tok, graft, TEXTS[16:], cfg)
    assert after["kl"] < before["kl"]
    assert graft.enc.mix.grad is not None
    pgraft.calibrate_scale(base, tok, graft, TEXTS)
    vmap = pgraft.vocab_map(tok, tok)
    pgraft.train(base, [{"params": pgraft.trainable([graft]), "lr": 1e-3}], 4,
                 lambda s: pgraft.kd_loss(base, tok, donor, tok, TEXTS[:4], vmap, cfg), log_every=0)
    assert graft.dec.up.weight.abs().sum() > 0
    # Reading a later layer than the graft writes at is refused.
    other = pgraft.build(tiny_host(), donor, host_layer=1, first=2, last=4, read_layers=[2])
    with pytest.raises(ValueError):
        pgraft.attach(tiny_host(), [other])
    # Reading only the graft's own layer is the plain encoder.
    same = pgraft.build(base, donor, host_layer=1, first=2, last=4, read_layers=[1])
    assert getattr(same.enc, "host_layers", None) is None
