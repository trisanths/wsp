"""Feedback seams: exact recursion, parallel passes, padding, controls, save and load, a planted skill."""

import copy
import random

import pytest
import torch

transformers = pytest.importorskip("transformers")

from wsp.feedback import (
    FeedbackSeam,
    base_logits,
    count_params,
    first_real,
    heal_distill,
    heal_seam,
    kd_kl,
    load_seam,
    make_block_branch,
    make_plain_branch,
    positions_of,
    random_range,
    feedback_stats,
    held_out_cosine,
    save_seam,
    shift_feedback,
    stage2_groups,
    warm_start_t_out,
)

NEW_TOKENS = 6


def tiny_llama(width, layers, seed, vocab=300, heads=4, kv=2, inter=None):
    torch.manual_seed(seed)
    config = transformers.LlamaConfig(
        hidden_size=width, intermediate_size=inter or 2 * width, num_hidden_layers=layers,
        num_attention_heads=heads, num_key_value_heads=kv, vocab_size=vocab, max_position_embeddings=64,
    )
    config._attn_implementation = "eager"
    return transformers.LlamaForCausalLM(config).eval()


def host_model():
    return tiny_llama(32, 4, seed=0)


def donor_model():
    return tiny_llama(48, 6, seed=1, heads=4, kv=2)


def sink_vector():
    torch.manual_seed(9)
    return torch.randn(48) * 3.0


def set_gate(branch, value):
    """Set the effective gate of a branch (legacy or not)."""
    with torch.no_grad():
        if branch.legacy:
            branch.gate_raw.fill_(value)
        else:
            branch.gate_raw.fill_(float(torch.atanh(torch.tensor(value / float(branch.g_max)))))


def randomize(seam, scale=0.3, seed=5, gate=0.3):
    """Give every branch a new T_out and an open gate, so the feedback matters."""
    generator = torch.Generator()
    generator.manual_seed(seed)
    for branch in seam.branches:
        weight = torch.randn(branch.t_out.weight.shape, generator=generator) * scale
        with torch.no_grad():
            branch.t_out.weight.copy_(weight)
        set_gate(branch, gate)
    return seam


def block_seam(host=None, donor=None, sink=True, plain=False, control=None, seed=0):
    host = host if host is not None else host_model()
    donor = donor if donor is not None else donor_model()
    branches = []
    branches.append(make_block_branch(host, donor, [1, 2], 1, sink=sink_vector() if sink else None,
                                      control=control, control_seed=seed))
    if plain:
        branches.append(make_plain_branch(host))
    return FeedbackSeam(host, branches).eval()


def plain_seam():
    host = host_model()
    return FeedbackSeam(host, [make_plain_branch(host)]).eval()


def padded_batch():
    ids = torch.tensor([[0, 0, 5, 9, 4, 7, 11], [3, 8, 6, 5, 9, 4, 2]])
    mask = torch.tensor([[0, 0, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1]])
    return ids, mask


def plain_ids():
    torch.manual_seed(2)
    return torch.randint(3, 300, (2, 7))


def worst(a, b, mask=None):
    """Largest absolute gap. With a mask, only the real positions count (a pad position has no defined output)."""
    gap = (a - b).abs()
    if mask is not None:
        gap = gap * mask.view(mask.shape[0], mask.shape[1], 1).to(gap.dtype)
    return float(gap.max())


# ---------------------------------------------------------------- helpers


def test_shift_feedback_gives_position_zero_nothing_and_skips_pads():
    fb = torch.arange(1.0, 7.0).view(1, 6, 1).repeat(2, 1, 3)
    mask = torch.tensor([[1, 1, 1, 1, 1, 1], [0, 0, 1, 1, 1, 0]])
    out = shift_feedback(fb, mask)
    assert out[0, 0].abs().sum() == 0
    assert torch.equal(out[0, 1:, 0], torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0]))
    # Row 1: positions 0, 1 are pads, position 2 is the first real token (no feedback), 3 and 4 get it, 5 is a pad.
    assert torch.equal(out[1, :, 0], torch.tensor([0.0, 0.0, 0.0, 3.0, 4.0, 0.0]))


def test_first_real_and_positions_count_only_real_tokens():
    mask = torch.tensor([[0, 0, 1, 1, 1], [1, 1, 1, 1, 1], [0, 1, 1, 0, 0]])
    assert first_real(mask).long().tolist() == [[0, 0, 1, 0, 0], [1, 0, 0, 0, 0], [0, 1, 0, 0, 0]]
    assert positions_of(mask).tolist() == [[0, 0, 0, 1, 2], [0, 1, 2, 3, 4], [0, 0, 1, 1, 1]]


def test_random_range_is_seeded_same_length_and_never_the_real_range():
    for seed in range(20):
        first, last = random_range(10, 3, 5, seed)
        assert last - first == 2
        assert first != 3
        assert 0 <= first and last <= 9
    assert random_range(10, 3, 5, 4) == random_range(10, 3, 5, 4)
    with pytest.raises(ValueError):
        random_range(3, 0, 2, 0)


# ---------------------------------------------------------------- fresh equals base


@pytest.mark.parametrize("kind", ["block", "plain", "both"])
def test_fresh_seam_equals_base_forward_and_generate(kind):
    host = host_model()
    donor = donor_model()
    branches = []
    if kind in ("block", "both"):
        branches.append(make_block_branch(host, donor, [1, 2], 1, sink=sink_vector()))
    if kind in ("plain", "both"):
        branches.append(make_plain_branch(host))
    seam = FeedbackSeam(host, branches).eval()
    ids, mask = padded_batch()
    pos = positions_of(mask)
    with torch.no_grad():
        want = host(input_ids=ids, attention_mask=mask, position_ids=pos).logits
        for passes in (1, 2, 3):
            assert worst(seam(ids, mask, passes=passes).logits, want) == 0.0
        assert worst(seam.forward_sequential(ids, mask), want, mask) < 1e-5
        assert worst(seam.forward_cached(ids, mask), want, mask) < 1e-5
        settings = dict(max_new_tokens=NEW_TOKENS, do_sample=False, pad_token_id=0, eos_token_id=None)
        expected = host.generate(input_ids=ids, attention_mask=mask, **settings)
        got = seam.generate(input_ids=ids, attention_mask=mask, **settings)
    assert torch.equal(got, expected)


# ---------------------------------------------------------------- exact recursion


@pytest.mark.parametrize("plain", [False, True])
def test_cached_generate_equals_exact_sequential_forward(plain):
    seam = randomize(block_seam(plain=plain))
    ids, mask = padded_batch()
    with torch.no_grad():
        out, scores = seam.generate(input_ids=ids, attention_mask=mask, max_new_tokens=NEW_TOKENS, do_sample=False,
                                    pad_token_id=0, eos_token_id=None, return_logits=True)
        full_mask = torch.cat([mask, torch.ones(2, NEW_TOKENS, dtype=mask.dtype)], dim=1)
        sequential = seam.forward_sequential(out, full_mask)
        cached = seam.forward_cached(out, full_mask)
        no_feedback = seam(out, full_mask, passes=1).logits
    start = ids.shape[1] - 1
    assert out.shape == (2, ids.shape[1] + NEW_TOKENS)
    assert worst(scores, sequential[:, start:start + NEW_TOKENS]) < 1e-4
    assert worst(cached, sequential, full_mask) < 1e-4
    # The feedback matters here: the recursion differs from a no-feedback run.
    assert worst(sequential, no_feedback, full_mask) > 1e-3
    # The new tokens are the argmax of the logits that chose them.
    assert torch.equal(out[:, ids.shape[1]:], scores.argmax(-1))


def test_generate_accepts_the_scorer_arguments_and_stops_at_eos():
    seam = randomize(block_seam())
    ids, mask = padded_batch()

    class Stop:
        def __call__(self, input_ids, scores, **kwargs):
            return torch.ones(input_ids.shape[0], dtype=torch.bool)

    with torch.no_grad():
        out = seam.generate(input_ids=ids, attention_mask=mask, max_new_tokens=5, do_sample=False, temperature=None,
                            top_p=None, top_k=None, repetition_penalty=1.0, pad_token_id=0, eos_token_id=[1, 2],
                            stopping_criteria=Stop())
    assert out.shape[1] == ids.shape[1] + 1
    with pytest.raises(ValueError):
        seam.generate(input_ids=ids, do_sample=True)


# ---------------------------------------------------------------- passes


@pytest.mark.parametrize("plain", [False, True])
def test_passes_converge_to_the_recursion_and_are_exact_for_short_sequences(plain):
    seam = randomize(block_seam(plain=plain), scale=0.15)
    length = 6
    ids = plain_ids()[:, :length]
    with torch.no_grad():
        exact = seam.forward_sequential(ids)
        gaps = []
        for passes in range(1, length + 1):
            got = seam(ids, passes=passes).logits
            gaps.append(worst(got, exact))
            # After n passes, positions 0 to n - 1 are exact.
            assert worst(got[:, :passes], exact[:, :passes]) < 1e-4
    # The error of a fixed-point pass need not fall at every step, but the exact prefix grows by one position
    # per pass, so the full-length run is exact and the one-pass run is not.
    assert gaps[0] > 1e-3
    assert gaps[-1] < 1e-4
    assert gaps[-1] < gaps[0] * 1e-3


def test_only_the_last_pass_builds_the_graph_by_default():
    seam = randomize(block_seam())
    seam.set_trainable(("t_in", "t_out", "gate"))
    ids = plain_ids()
    out = seam(ids, passes=2).logits
    out.sum().backward()
    # T_in sits before the donor block in the feedback path of the last pass, so it gets a gradient.
    assert any(param.grad is not None for param in seam.branches[0].t_in.parameters())
    seam.zero_grad()
    # With the graph through the first pass too, T_in still gets one: no error and a finite value.
    out = seam(ids, passes=2, grad_through_passes=True).logits
    out.sum().backward()
    assert torch.isfinite(seam.branches[0].t_out.weight.grad).all()


# ---------------------------------------------------------------- positions and padding


def test_position_zero_gets_no_feedback():
    fresh = block_seam()
    seam = randomize(block_seam())
    ids = plain_ids()
    with torch.no_grad():
        a = fresh(ids, passes=2).logits
        b = seam(ids, passes=2).logits
    assert worst(a[:, 0], b[:, 0]) == 0.0
    assert worst(a[:, 1], b[:, 1]) > 1e-4


@pytest.mark.parametrize("side", ["left", "right"])
@pytest.mark.parametrize("plain", [False, True])
def test_padding_is_handled_per_row(side, plain):
    seam = randomize(block_seam(plain=plain), scale=0.5)
    ids = plain_ids()
    short = ids[0, :5].unsqueeze(0)
    pad = 3
    if side == "left":
        padded = torch.cat([torch.zeros(1, pad, dtype=torch.long), short], dim=1)
        mask = torch.cat([torch.zeros(1, pad, dtype=torch.long), torch.ones(1, 5, dtype=torch.long)], dim=1)
        real = slice(pad, pad + 5)
    else:
        padded = torch.cat([short, torch.zeros(1, pad, dtype=torch.long)], dim=1)
        mask = torch.cat([torch.ones(1, 5, dtype=torch.long), torch.zeros(1, pad, dtype=torch.long)], dim=1)
        real = slice(0, 5)
    with torch.no_grad():
        want_exact = seam.forward_sequential(short)
        want_passes = seam(short, passes=3).logits
        assert worst(seam.forward_sequential(padded, mask)[:, real], want_exact) < 1e-4
        assert worst(seam.forward_cached(padded, mask)[:, real], want_exact) < 1e-4
        assert worst(seam(padded, mask, passes=3).logits[:, real], want_passes) < 1e-4


def test_a_batch_with_different_padding_matches_each_row_alone():
    seam = randomize(block_seam(plain=True), scale=0.5)
    ids, mask = padded_batch()
    with torch.no_grad():
        batch = seam.forward_cached(ids, mask)
        for row in range(2):
            real = mask[row].bool()
            alone = seam.forward_cached(ids[row][real].unsqueeze(0))
            assert worst(batch[row][real], alone[0]) < 1e-4


# ---------------------------------------------------------------- the plain branch against a hand-written reference


def test_plain_branch_matches_a_hand_written_reference():
    seam = randomize(plain_seam(), scale=0.4)
    host = seam.host
    branch = seam.branches[0]
    w = branch.t_out.weight.detach().clone()
    gate = float(branch.gate.detach())
    embed_rms = float(branch.embed_rms)
    ids = plain_ids()
    batch, length = ids.shape
    embed = host.get_input_embeddings()

    # Reference: step through time. The input at t is embed(token_t) plus gate * W top_{t-1}, with nothing at t = 0.
    top = []
    logits = []
    with torch.no_grad():
        for t in range(length):
            inputs = embed(ids[:, :t + 1]).clone()
            for u in range(1, t + 1):
                normed = top[u - 1] * torch.rsqrt(top[u - 1].pow(2).mean(-1, keepdim=True) + 1e-6)
                y = normed @ w.T
                inputs[:, u] = inputs[:, u] + gate * embed_rms * y / y.norm(dim=-1, keepdim=True)
            out = host.model(inputs_embeds=inputs, use_cache=False)
            top = [out.last_hidden_state[:, u] for u in range(t + 1)]
            logits.append(host.lm_head(out.last_hidden_state[:, -1]))
        reference = torch.stack(logits, dim=1)
        assert worst(seam.forward_sequential(ids), reference) < 1e-4
        assert worst(seam.forward_cached(ids), reference) < 1e-4
        assert worst(seam(ids, passes=length).logits, reference) < 1e-4


# ---------------------------------------------------------------- gradients


def test_stage_two_gradients_reach_only_t_out_gates_and_lora():
    seam = block_seam(plain=True)
    seam.add_host_lora(4)
    seam.to(torch.float32)
    groups = stage2_groups(seam, train_t_in=False)
    assert [group["name"] for group in groups] == ["t_out", "gate", "lora"]
    ids = plain_ids()
    seam.train()
    loss = seam(ids, passes=2).logits.pow(2).mean()
    loss.backward()
    allowed = {}
    for group in groups:
        for param in group["params"]:
            allowed[id(param)] = group["name"]
    for name, param in seam.named_parameters():
        if id(param) in allowed:
            assert param.requires_grad, name
        else:
            assert not param.requires_grad, name
            assert param.grad is None, name
    # A fresh seam has the gate at 0, so at step 0 only the gate has a gradient. T_out gets none yet.
    for branch in seam.branches:
        assert float(branch.gate_raw.grad.abs()) > 0
        assert branch.t_out.weight.grad is None or float(branch.t_out.weight.grad.abs().sum()) == 0
    # With an open gate, T_out gets a gradient too.
    for branch in seam.branches:
        set_gate(branch, 0.2)
    seam.zero_grad()
    seam(ids, passes=2).logits.pow(2).mean().backward()
    for branch in seam.branches:
        assert branch.t_out.weight.grad is not None
        assert float(branch.t_out.weight.grad.abs().sum()) > 0
    lora_b = []
    for name, param in seam.named_parameters():
        if name.endswith("lora_B"):
            lora_b.append(float(param.grad.abs().sum()))
    assert len(lora_b) > 0 and max(lora_b) > 0
    # One pass has no feedback, so T_out gets no gradient then.
    seam.zero_grad()
    seam(ids, passes=1).logits.pow(2).mean().backward()
    assert seam.branches[0].t_out.weight.grad is None or float(seam.branches[0].t_out.weight.grad.abs().sum()) == 0


def test_base_logits_inside_a_healed_seam_equal_the_frozen_base():
    host = host_model()
    reference = copy.deepcopy(host)
    seam = FeedbackSeam(host, [make_plain_branch(host)]).eval()
    seam.add_host_lora(4)
    for _, module in [(n, m) for n, m in seam.host.named_modules() if hasattr(m, "lora_B")]:
        with torch.no_grad():
            module.lora_B.normal_(0, 0.5)
    randomize(seam)
    ids, mask = padded_batch()
    with torch.no_grad():
        want = reference(input_ids=ids, attention_mask=mask, position_ids=positions_of(mask)).logits
        assert worst(base_logits(seam, ids, mask), want) < 1e-5
        assert worst(seam(ids, mask).logits, want) > 1e-3


# ---------------------------------------------------------------- the random control


def test_random_control_differs_from_real_with_the_same_shapes_and_counts():
    host = host_model()
    donor = donor_model()
    real = FeedbackSeam(host, [make_block_branch(host, donor, [1, 2], 1, sink=sink_vector())])
    other_host = copy.deepcopy(host)
    control = FeedbackSeam(other_host, [make_block_branch(other_host, donor, [1, 2], 1, sink=sink_vector(),
                                                          control="random", control_seed=3)])
    assert real.branches[0].meta["donor_layers"] == [1, 2]
    assert control.branches[0].meta["donor_layers"] != [1, 2]
    assert len(set(control.branches[0].meta["donor_layers"])) == 2
    real_state = real.branches[0].state_dict()
    control_state = control.branches[0].state_dict()
    assert sorted(real_state) == sorted(control_state)
    for key in real_state:
        assert real_state[key].shape == control_state[key].shape
    assert count_params(real) == count_params(control)
    assert not torch.equal(real_state["part.layers.0.mlp.up_proj.weight"],
                           control_state["part.layers.0.mlp.up_proj.weight"])
    randomize(real)
    randomize(control)
    # Same T_in and T_out values, so only the block differs.
    control.branches[0].t_in.load_state_dict(real.branches[0].t_in.state_dict())
    ids = plain_ids()
    with torch.no_grad():
        assert worst(real(ids).logits, control(ids).logits) > 1e-4


# ---------------------------------------------------------------- counts


def test_count_params_counts_the_branch_layers_as_run_every_token():
    host = host_model()
    donor = donor_model()
    base_total = sum(param.numel() for param in host.parameters())
    embed = host.get_input_embeddings().weight.numel()
    seam = FeedbackSeam(host, [make_block_branch(host, donor, [1, 2], 1), make_plain_branch(host)])
    counts = count_params(seam)
    layer = sum(param.numel() for param in donor.model.layers[1].parameters())
    assert counts["component"] == 2 * layer
    assert counts["t_in"] == 32 * 48
    assert counts["t_out"] == 48 * 32 + 32 * 32
    assert counts["gate"] == 2
    assert counts["total"] == base_total + counts["component"] + counts["t_in"] + counts["t_out"] + 2
    assert counts["per_token"] == counts["total"] - embed
    assert counts["host"] == base_total
    # The donor layers are frozen; everything else has requires_grad set on a new seam.
    assert counts["trainable"] == counts["total"] - counts["component"]


# ---------------------------------------------------------------- save and load


def test_save_and_load_give_equal_logits_and_generation(tmp_path):
    donor = donor_model()
    donor_dir = tmp_path / "donor"
    donor.save_pretrained(donor_dir)
    host = host_model()
    pristine = copy.deepcopy(host)
    branch = make_block_branch(host, str(donor_dir), [1, 2], 1, sink=sink_vector(),
                               donor_config=donor.config, donor_id=str(donor_dir))
    seam = FeedbackSeam(host, [branch, make_plain_branch(host)], base_id="unused")
    seam.add_host_lora(4)
    seam.to(torch.float32)
    randomize(seam)
    with torch.no_grad():
        for name, param in seam.named_parameters():
            if name.endswith("lora_B"):
                param.normal_(0, 0.1)
        set_gate(seam.branches[0], 0.4)
    seam.eval()
    out_dir = tmp_path / "saved"
    manifest = save_seam(seam, str(out_dir), base_id="unused")
    assert manifest["format"] == "wsp-feedback-2"
    assert manifest["branches"][0]["donor_layers"] == [1, 2]
    loaded, tokenizer = load_seam(str(out_dir), host=pristine, dtype=torch.float32)
    assert tokenizer is None
    ids, mask = padded_batch()
    with torch.no_grad():
        assert worst(seam(ids, mask).logits, loaded(ids, mask).logits) == 0.0
        assert worst(seam.forward_cached(ids, mask), loaded.forward_cached(ids, mask), mask) == 0.0
        settings = dict(max_new_tokens=NEW_TOKENS, do_sample=False, pad_token_id=0, eos_token_id=None)
        assert torch.equal(seam.generate(input_ids=ids, attention_mask=mask, **settings),
                           loaded.generate(input_ids=ids, attention_mask=mask, **settings))
    assert count_params(seam)["total"] == count_params(loaded)["total"]


def test_save_and_load_keep_a_random_control_range(tmp_path):
    donor = donor_model()
    donor_dir = tmp_path / "donor"
    donor.save_pretrained(donor_dir)
    host = host_model()
    pristine = copy.deepcopy(host)
    branch = make_block_branch(host, str(donor_dir), [1, 2], 1, donor_config=donor.config, donor_id=str(donor_dir),
                               control="random", control_seed=11)
    seam = randomize(FeedbackSeam(host, [branch]).eval())
    save_seam(seam, str(tmp_path / "saved"), base_id="unused")
    loaded, _ = load_seam(str(tmp_path / "saved"), host=pristine, dtype=torch.float32)
    assert loaded.branches[0].meta["donor_layers"] == branch.meta["donor_layers"]
    assert loaded.branches[0].meta["control"] == "random"
    ids = plain_ids()
    with torch.no_grad():
        assert worst(seam(ids).logits, loaded(ids).logits) == 0.0


# ---------------------------------------------------------------- stage 1


def test_fit_t_in_trains_the_stitch_on_the_donor_loss_and_builds_a_working_branch():
    from tests.test_align import WORDS, tiny_tokenizer
    from wsp.feedback import fit_t_in

    tokenizer = tiny_tokenizer()
    host = host_model()
    donor = donor_model()
    texts = []
    for i in range(12):
        texts.append(" ".join(WORDS[i % 5: i % 5 + 8]))
    t_in, sink, report = fit_t_in(host, tokenizer, 1, donor, tokenizer, 1, texts, kind="linear", steps=30, lr=1e-3,
                                  batch_size=4, seq_len=16, seed=0, sink=True)
    assert tuple(t_in.weight.shape) == (48, 32)
    assert sink.shape == (48,)
    assert report["penalty_after"] < report["penalty_before"]
    branch = make_block_branch(host, donor, [1, 2], 1, t_in=t_in, sink=sink)
    seam = randomize(FeedbackSeam(host, [branch]).eval())
    ids = plain_ids()
    with torch.no_grad():
        assert worst(seam.forward_cached(ids), seam.forward_sequential(ids)) < 1e-4


# ---------------------------------------------------------------- the training loop


class Tok:
    padding_side = "right"
    pad_token = "p"
    eos_token = "p"

    def __call__(self, batch, return_tensors=None, padding=None, truncation=None, max_length=None,
                 add_special_tokens=None):
        rows = []
        for text in batch:
            rows.append([int(piece) for piece in text.split()][:max_length])
        longest = max(len(row) for row in rows)
        ids = torch.zeros(len(rows), longest, dtype=torch.long)
        mask = torch.zeros(len(rows), longest, dtype=torch.long)
        for index, row in enumerate(rows):
            ids[index, :len(row)] = torch.tensor(row)
            mask[index, :len(row)] = 1
        return {"input_ids": ids, "attention_mask": mask}


def test_heal_distill_lowers_the_kl_to_a_teacher_and_keeps_the_pass_schedule():
    host = host_model()
    teacher = tiny_llama(32, 4, seed=7)
    seam = FeedbackSeam(host, [make_plain_branch(host)])
    seam.add_host_lora(4)

    torch.manual_seed(4)
    texts = []
    for _ in range(24):
        texts.append(" ".join(str(int(v)) for v in torch.randint(3, 300, (10,))))
    flags = [index % 3 == 0 for index in range(24)]
    ids = torch.tensor([[int(piece) for piece in texts[1].split()]])
    with torch.no_grad():
        before = float(kd_kl(seam(ids).logits, teacher(ids).logits, torch.ones(1, 10)))
    groups = stage2_groups(seam, lr_t_out=2e-3, lr_gate=1e-3, lr_lora=2e-3)
    seam.train()

    def forward(ids, mask, passes):
        return seam(ids, mask, passes=passes).logits

    def anchor(ids, mask):
        return base_logits(seam, ids, mask)

    log = heal_distill(forward, anchor, groups, Tok(), teacher, texts, flags, steps=40, batch_size=6, seq_len=10,
                       seed=1, log_every=0)
    seam.eval()
    with torch.no_grad():
        after = float(kd_kl(seam(ids).logits, teacher(ids).logits, torch.ones(1, 10)))
    assert after < before
    assert len(log["kd"]) == 40
    assert set(log["passes"]) <= {1, 2}
    again = heal_distill(forward, anchor, groups, Tok(), teacher, texts, flags, steps=40, batch_size=6, seq_len=10,
                         seed=1, log_every=0)
    assert again["passes"] == log["passes"]


# ---------------------------------------------------------------- a planted skill

VOCAB = 8


def planted_batch(count, length, generator):
    """Sequences over 8 tokens with x[n] = x[n-2] + x[n-3] (mod 8) from position 3 on.

    Predicting x[n] at position n - 1 needs s = x[n-2] + x[n-3], which is what
    the donor block computes at position n - 2 (from its own two last tokens).
    """
    ids = torch.randint(0, VOCAB, (count, length), generator=generator)
    for n in range(3, length):
        ids[:, n] = (ids[:, n - 2] + ids[:, n - 3]) % VOCAB
    return ids


def lm_loss(logits, ids, first):
    """Next-token loss on targets at positions >= first."""
    pred = logits[:, first - 1:-1].reshape(-1, logits.shape[-1])
    gold = ids[:, first:].reshape(-1)
    return torch.nn.functional.cross_entropy(pred, gold)


def train_planted_donor():
    """A 4-layer donor whose logits at position t are (x[t] + x[t-1]) mod 8, on i.i.d. tokens.

    Only the donor's block computes this sum of two earlier tokens. The skill
    does not depend on any property of the training sequences.
    """
    donor = tiny_llama(32, 4, seed=11, vocab=VOCAB, heads=2, kv=2, inter=128)
    donor.train()
    generator = torch.Generator()
    generator.manual_seed(0)
    optimizer = torch.optim.AdamW(donor.parameters(), lr=4e-3)
    for step in range(700):
        ids = torch.randint(0, VOCAB, (64, 12), generator=generator)
        target = (ids[:, 1:] + ids[:, :-1]) % VOCAB
        logits = donor(input_ids=ids).logits[:, 1:]
        loss = torch.nn.functional.cross_entropy(logits.reshape(-1, VOCAB), target.reshape(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    donor.eval()
    return donor, float(loss.detach())


def planted_host():
    """A 3-layer host that cannot do the task alone. Layer 0 sees one token at a time (its o_proj is zero), so a
    block that skips the donor's first layer cannot recover the previous token from the host state. The head is
    scaled up so the logits can be sharp."""
    host = tiny_llama(24, 3, seed=21, vocab=VOCAB, heads=2, kv=2, inter=96)
    host.lm_head.weight.data.mul_(25)
    host.model.layers[0].self_attn.o_proj.weight.data.zero_()
    return host


def ridge_t_in(host, read_layer, donor, first, generator):
    """T_in as a least-squares map from host layer read_layer output to the donor's layer first input."""
    ids = planted_batch(256, 12, generator)
    with torch.no_grad():
        x = host(input_ids=ids, output_hidden_states=True).hidden_states[read_layer + 1]
        y = donor(input_ids=ids, output_hidden_states=True).hidden_states[first]
    x = x.reshape(-1, x.shape[-1])
    y = y.reshape(-1, y.shape[-1])
    solution = torch.linalg.lstsq(x, y).solution
    t_in = torch.nn.Linear(x.shape[-1], y.shape[-1], bias=False)
    with torch.no_grad():
        t_in.weight.copy_(solution.T)
    return t_in


def heal_planted(donor, control, seed):
    """Heal the host with one block branch on the planted rule, 2 passes. Returns the held-out loss."""
    host = planted_host()
    generator = torch.Generator()
    generator.manual_seed(100 + seed)
    first = 0
    if control == "random":
        first = random_range(4, 0, 2, seed)[0]
    t_in = ridge_t_in(host, 0, donor, first, generator)
    branch = make_block_branch(host, donor, [0, 2], 0, t_in=t_in, control=control, control_seed=seed)
    assert branch.meta["donor_layers"][0] == first
    seam = FeedbackSeam(host, [branch])
    seam.add_host_lora(4, layers=[1, 2])
    groups = stage2_groups(seam, lr_t_out=1e-3, lr_gate=1e-3, lr_lora=3e-3)
    optimizer = torch.optim.AdamW(groups, weight_decay=0.0)
    seam.train()
    for step in range(250):
        ids = planted_batch(32, 12, generator)
        loss = lm_loss(seam(ids, passes=2).logits, ids, 3)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    seam.eval()
    held = torch.Generator()
    held.manual_seed(999)
    ids = planted_batch(256, 12, held)
    with torch.no_grad():
        return float(lm_loss(seam(ids, passes=2).logits, ids, 3))


def test_planted_skill_real_block_beats_the_random_block_after_a_short_heal():
    donor, donor_loss = train_planted_donor()
    # The donor knows its sum, so the skill is real before the seam test starts.
    assert donor_loss < 0.05
    real = []
    controls = []
    for seed in range(2):
        real.append(heal_planted(donor, None, seed))
        controls.append(heal_planted(donor, "random", seed))
    print("planted real", real, "random", controls)
    assert max(real) < min(controls) - 0.2


# ---------------------------------------------------------------- the go rule


def verdict_row(pair, arm, seed, transfer, drift=0.01):
    return {"pair": pair, "arm": arm, "seed": seed,
            "probe": {"skills": {"math": {"transfer": transfer}, "code": {"transfer": transfer + 0.02},
                                 "medicine": {"transfer": None}}, "drift": {"base_kl": drift}}}


def test_go_verdict_needs_a_gap_over_the_seed_spread_and_low_drift():
    from wsp.feedback import go_verdict

    rows = []
    for seed, (block, lora, plain, random_arm) in enumerate([(0.50, 0.30, 0.31, 0.32), (0.54, 0.34, 0.30, 0.30)]):
        rows.append(verdict_row("A", "feedback_block", seed, block))
        rows.append(verdict_row("A", "kd_lora", seed, lora))
        rows.append(verdict_row("A", "feedback_plain", seed, plain))
        rows.append(verdict_row("A", "feedback_block_random", seed, random_arm))
        rows.append(verdict_row("A", "midstack_block", seed, 0.51))
    result = go_verdict(rows)["A"]
    assert result["go"] is True
    assert result["midstack_equals_feedback"] is True
    assert result["plain_equals_block"] is False
    assert abs(result["beats"]["kd_lora"]["gap"] - 0.20) < 1e-9
    # A gap under the seed spread is not a go.
    close = []
    for seed, (block, other) in enumerate([(0.30, 0.20), (0.50, 0.40)]):
        close.append(verdict_row("A", "feedback_block", seed, block))
        for name in ("kd_lora", "feedback_plain", "feedback_block_random"):
            close.append(verdict_row("A", name, seed, other))
    assert go_verdict(close)["A"]["go"] is False
    # High drift is not a go either.
    drifted = [dict(row) for row in rows]
    drifted[0] = verdict_row("A", "feedback_block", 0, 0.50, drift=0.2)
    assert go_verdict(drifted)["A"]["go"] is False
    assert go_verdict(rows)["A"]["content_gain"] is not None


def test_a_midstack_insertion_heals_with_the_same_loop_and_its_base_view_is_the_frozen_base():
    from wsp.feedback import BaseView
    from wsp.insert import add_base_lora, insert_layers, insertion_parameters

    host = host_model()
    reference = copy.deepcopy(host)
    donor = donor_model()
    teacher = tiny_llama(32, 4, seed=7)
    insertion = insert_layers(host, donor, [1, 2], 2, gate_init=0.0)
    add_base_lora(host, 4, layers=[2, 3])
    for param in host.parameters():
        param.requires_grad_(False)
    found = insertion_parameters(host)
    groups = [{"params": list(insertion.w_out.linear.parameters()), "lr": 1e-3},
              {"params": found["gate"], "lr": 5e-3}, {"params": found["base_lora"], "lr": 2e-3}]
    for group in groups:
        for param in group["params"]:
            param.requires_grad_(True)

    def forward(ids, mask, passes):
        return host(input_ids=ids, attention_mask=mask, use_cache=False).logits

    def anchor(ids, mask):
        return base_logits(host, ids, mask)

    torch.manual_seed(4)
    texts = []
    for _ in range(12):
        texts.append(" ".join(str(int(v)) for v in torch.randint(3, 300, (9,))))
    flags = [index % 3 == 0 for index in range(12)]
    log = heal_distill(forward, anchor, groups, Tok(), teacher, texts, flags, steps=15, batch_size=6, seq_len=9,
                       seed=2, log_every=0)
    assert len(log["kd"]) == 15
    assert float(insertion.w_out.gate.detach()) != 0.0
    ids, mask = padded_batch()
    with torch.no_grad():
        want = reference(input_ids=ids, attention_mask=mask, position_ids=positions_of(mask)).logits
        view = BaseView(host)
        assert worst(view(input_ids=ids, attention_mask=mask).logits, want) < 1e-5
        assert worst(host(input_ids=ids, attention_mask=mask).logits, want, mask) > 1e-4


def test_scrambled_control_keeps_shapes_and_survives_save_and_load(tmp_path):
    donor = donor_model()
    donor_dir = tmp_path / "donor"
    donor.save_pretrained(donor_dir)
    host = host_model()
    real_host = copy.deepcopy(host)
    real = make_block_branch(real_host, str(donor_dir), [1, 2], 1, donor_config=donor.config, donor_id=str(donor_dir))
    branch = make_block_branch(host, str(donor_dir), [1, 2], 1, donor_config=donor.config, donor_id=str(donor_dir),
                               control="scrambled", control_seed=5)
    assert branch.meta["donor_layers"] == real.meta["donor_layers"]
    real_state = real.part.state_dict()
    state = branch.part.state_dict()
    changed = 0
    for name in state:
        assert state[name].shape == real_state[name].shape
        if state[name].dim() == 2 and not torch.equal(state[name], real_state[name]):
            changed += 1
            assert torch.allclose(state[name].float().norm(), real_state[name].float().norm(), rtol=1e-3)
    assert changed > 0
    # Loading must scramble again with the saved seed, not fetch the real weights.
    seam = FeedbackSeam(host, [branch]).eval()
    save_seam(seam, str(tmp_path / "seam"), base_id="unused")
    loaded, _ = load_seam(str(tmp_path / "seam"), host=copy.deepcopy(real_host), dtype=torch.float32)
    assert loaded.branches[0].meta["control"] == "scrambled"
    loaded_state = loaded.branches[0].part.state_dict()
    for name in state:
        assert torch.allclose(loaded_state[name].float(), state[name].float(), atol=1e-5)


# ---------------------------------------------------------------- scale control, warm start, optimizer safety


def loud_donor(amp):
    """A donor whose block output is amp times larger, like the outlier dimensions of a real residual stream."""
    donor = donor_model()
    with torch.no_grad():
        for layer in donor.model.layers:
            layer.self_attn.o_proj.weight.mul_(amp)
            layer.mlp.down_proj.weight.mul_(amp)
    return donor


def one_step_change(legacy, amp, lr_t_out=3e-4, lr_gate=1e-3):
    """Largest logit change after ONE optimizer step from a fresh seam, and the written norm over the embedding norm."""
    host = host_model()
    donor = loud_donor(amp)
    teacher = tiny_llama(32, 4, seed=7)
    branch = make_block_branch(host, donor, [1, 2], 1, sink=sink_vector(), legacy=legacy)
    seam = FeedbackSeam(host, [branch]).eval()
    seam.to(torch.float32)
    groups = []
    found = seam.set_trainable(("t_out", "gate"))
    groups.append({"params": found["t_out"], "lr": lr_t_out})
    groups.append({"params": found["gate"], "lr": lr_gate})
    ids = plain_ids()
    with torch.no_grad():
        before = seam(ids, passes=2).logits.clone()
    optimizer = torch.optim.AdamW(groups, weight_decay=0.0)
    loss = kd_kl(seam(ids, passes=2).logits, teacher(ids).logits, torch.ones(ids.shape))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    with torch.no_grad():
        after = seam(ids, passes=2).logits
        mask = torch.ones_like(ids)
        out = seam.host_pass(ids, mask, positions_of(mask), None)
        written = branch.output(seam.reads_of(out.hidden_states)[0], mask)
        ratio = float(written.norm(dim=-1).mean() / seam.get_input_embeddings()(ids).norm(dim=-1).mean())
    return float((after - before).abs().max()), float(before.std()), ratio


def test_one_optimizer_step_from_a_fresh_seam_moves_the_logits_by_a_small_bounded_amount():
    # The old path (legacy=True): the first Adam step gives every T_out entry a move of lr, so the written vector
    # grows with the raw scale of the block output. The new path does not depend on that scale.
    legacy_small, std, legacy_ratio_small = one_step_change(True, 1.0)
    legacy_big, _, legacy_ratio_big = one_step_change(True, 300.0)
    new_small, _, new_ratio_small = one_step_change(False, 1.0)
    new_big, _, new_ratio_big = one_step_change(False, 300.0)
    print("legacy", legacy_small, legacy_big, legacy_ratio_small, legacy_ratio_big)
    print("new", new_small, new_big, new_ratio_small, new_ratio_big)
    # Confirms the cause: with the old init the move scales with the block output.
    assert legacy_big > 30 * legacy_small
    assert legacy_ratio_big > 1.0
    # The fix: bounded and independent of the block scale.
    assert new_big < 0.01 * std
    assert new_small < 0.01 * std
    assert new_ratio_big < 0.01
    assert legacy_big > 20 * new_big


@pytest.mark.parametrize("scale", [1e-3, 1.0, 1e3])
def test_written_vector_norm_is_gate_times_embed_rms_for_any_t_out_scale(scale):
    host = host_model()
    donor = donor_model()
    branch = make_block_branch(host, donor, [1, 2], 1, sink=sink_vector(), g_max=0.5)
    plain = make_plain_branch(host, g_max=0.5)
    seam = FeedbackSeam(host, [branch, plain]).eval()
    seam.to(torch.float32)
    expected_rms = float(torch.sqrt(host.get_input_embeddings().weight.pow(2).sum(-1).mean()))
    for item in seam.branches:
        assert float(item.embed_rms) == pytest.approx(expected_rms, rel=1e-5)
        with torch.no_grad():
            item.t_out.weight.mul_(scale)
        set_gate(item, 0.25)
    ids = plain_ids()
    mask = torch.ones_like(ids)
    with torch.no_grad():
        out = seam.host_pass(ids, mask, positions_of(mask), None)
        for item, read in zip(seam.branches, seam.reads_of(out.hidden_states)):
            written = item.output(read, mask)
            norms = written.norm(dim=-1)
            assert float(norms.max()) == pytest.approx(0.25 * expected_rms, rel=1e-4)
            assert float(norms.min()) == pytest.approx(0.25 * expected_rms, rel=1e-4)
    stats = feedback_stats(seam, Tok(), [" ".join(str(int(v)) for v in ids[0])])
    for found in stats:
        assert found["gate"] == pytest.approx(0.25, rel=1e-4)
        assert found["ratio"] == pytest.approx(0.25, rel=1e-3)


def test_gate_zero_gives_the_base_exactly_in_forward_and_generate_even_with_a_big_t_out():
    host = host_model()
    donor = donor_model()
    seam = FeedbackSeam(host, [make_block_branch(host, donor, [1, 2], 1, sink=sink_vector()),
                               make_plain_branch(host)]).eval()
    for branch in seam.branches:
        with torch.no_grad():
            branch.t_out.weight.mul_(50.0)
        assert float(branch.gate) == 0.0
    ids, mask = padded_batch()
    pos = positions_of(mask)
    with torch.no_grad():
        want = host(input_ids=ids, attention_mask=mask, position_ids=pos).logits
        for passes in (1, 2, 3):
            assert worst(seam(ids, mask, passes=passes).logits, want) == 0.0
        settings = dict(max_new_tokens=NEW_TOKENS, do_sample=False, pad_token_id=0, eos_token_id=None)
        assert torch.equal(seam.generate(input_ids=ids, attention_mask=mask, **settings),
                           host.generate(input_ids=ids, attention_mask=mask, **settings))
    # The gate is bounded: raw values far from 0 stay inside (-g_max, g_max).
    with torch.no_grad():
        seam.branches[0].gate_raw.fill_(50.0)
    assert float(seam.branches[0].gate) <= 0.5 + 1e-6


def test_the_gate_opens_slowly_and_t_out_gets_no_gradient_at_step_zero():
    seam = block_seam()
    seam.set_trainable(("t_out", "gate"))
    seam(plain_ids(), passes=2).logits.pow(2).mean().backward()
    branch = seam.branches[0]
    assert float(branch.gate_raw.grad.abs()) > 0
    assert branch.t_out.weight.grad is None or float(branch.t_out.weight.grad.abs().sum()) == 0.0


class DivergingForward:
    """A student that gets worse every call: the logits drift away from the teacher's."""

    def __init__(self, teacher, vocab):
        self.teacher = teacher
        self.vocab = vocab
        self.calls = 0
        self.param = torch.nn.Parameter(torch.zeros(1))

    def __call__(self, ids, mask, passes):
        self.calls += 1
        with torch.no_grad():
            logits = self.teacher(input_ids=ids, attention_mask=mask).logits
        noise = torch.randn(logits.shape, generator=torch.Generator().manual_seed(self.calls))
        if passes == 1:
            return logits + self.param
        return logits + self.param + noise * (0.5 * self.calls)


def test_the_heal_loop_halts_on_a_planted_diverging_run_and_not_on_a_healthy_one():
    teacher = tiny_llama(32, 4, seed=7)
    torch.manual_seed(4)
    texts = []
    for _ in range(24):
        texts.append(" ".join(str(int(v)) for v in torch.randint(3, 300, (10,))))
    flags = [False] * 24
    student = DivergingForward(teacher, 300)
    groups = [{"params": [student.param], "lr": 1e-3}]

    def anchor(ids, mask):
        return teacher(input_ids=ids, attention_mask=mask).logits

    log = heal_distill(student, anchor, groups, Tok(), teacher, texts, flags, steps=200, batch_size=4, seq_len=10,
                       seed=1, log_every=0, one_pass_prob=0.0, three_pass_prob=0.0, halt_patience=5)
    assert log["halted"]
    assert log["halt_step"] < 60
    assert len(log["kd"]) == log["halt_step"]
    assert "two-pass KD" in log["halt_reason"] or "non-finite" in log["halt_reason"]
    # A healthy loop (the teacher itself as student) never halts.
    healthy = heal_distill(lambda ids, mask, passes: teacher(input_ids=ids, attention_mask=mask).logits
                           + groups[0]["params"][0], anchor, groups, Tok(), teacher, texts, flags, steps=40,
                           batch_size=4, seq_len=10, seed=1, log_every=0, halt_patience=5)
    assert not healthy["halted"]
    assert len(healthy["kd"]) == 40


def test_the_pass_schedule_has_three_pass_steps_after_the_warmup_and_stays_reproducible():
    from wsp.feedback import draw_passes

    first = random.Random(5)
    second = random.Random(5)
    one = [draw_passes(first, step, 0.15, 0.3, 50) for step in range(400)]
    two = [draw_passes(second, step, 0.15, 0.3, 50) for step in range(400)]
    assert one == two
    assert 3 not in one[:50]
    late = one[50:]
    assert late.count(3) > 0.15 * len(late)
    assert late.count(1) > 0
    # The pass count does not change the draws used by the other arms.
    no_three = [draw_passes(random.Random(5), 0, 0.15, 0.0, 50)]
    assert no_three[0] in (1, 2)


def test_three_pass_training_runs_and_gradients_reach_only_the_intended_parameters():
    seam = block_seam(plain=True)
    seam.add_host_lora(4)
    seam.to(torch.float32)
    randomize(seam)
    groups = stage2_groups(seam, train_t_in=True)
    assert [group["name"] for group in groups] == ["t_out", "gate", "t_in", "lora"]
    ids = plain_ids()
    seam.train()
    seam(ids, passes=3).logits.pow(2).mean().backward()
    allowed = {}
    for group in groups:
        for param in group["params"]:
            allowed[id(param)] = group["name"]
    for name, param in seam.named_parameters():
        if id(param) in allowed:
            assert param.requires_grad, name
        else:
            assert param.grad is None, name
    assert float(seam.branches[0].t_in.weight.grad.abs().sum()) > 0
    assert float(seam.branches[0].t_out.weight.grad.abs().sum()) > 0


def test_heal_distill_runs_three_pass_steps_with_a_group_warmup_and_logs_the_passes_gap():
    host = host_model()
    teacher = tiny_llama(32, 4, seed=7)
    seam = FeedbackSeam(host, [make_plain_branch(host)])
    seam.add_host_lora(4)
    torch.manual_seed(4)
    texts = []
    for _ in range(24):
        texts.append(" ".join(str(int(v)) for v in torch.randint(3, 300, (10,))))
    flags = [index % 3 == 0 for index in range(24)]
    log = heal_seam(seam, Tok(), teacher, texts, flags, lrs={"lr_t_out": 1e-3, "lr_gate": 1e-2, "lr_lora": 1e-3},
                    gap_texts=texts[:2], gap_every=10, train_t_in=False, steps=30, batch_size=6, seq_len=10, seed=1,
                    log_every=0, warmup=5, three_pass_prob=0.5, one_pass_prob=0.1)
    assert 3 in log["passes"]
    assert 3 not in log["passes"][:5]
    assert [entry["step"] for entry in log["gap"]] == [10, 20, 30]
    assert set(log["gap"][0]) >= {"kl", "agree", "tokens"}
    assert not log["halted"]
    assert float(seam.branches[0].gate) != 0.0


def test_t_in_regularizer_gives_a_loss_with_a_gradient_on_t_in_only():
    from tests.test_align import WORDS, tiny_tokenizer
    from wsp.feedback import t_in_regularizer

    tokenizer = tiny_tokenizer()
    host = host_model()
    donor = donor_model()
    branch = make_block_branch(host, donor, [1, 2], 1, sink=sink_vector())
    seam = FeedbackSeam(host, [branch])
    seam.set_trainable(("t_in",))
    texts = []
    for i in range(6):
        texts.append(" ".join(WORDS[i % 5: i % 5 + 8]))
    regularizer = t_in_regularizer(seam, tokenizer, donor, texts, weight=0.5, batch_size=2, seq_len=16)
    loss = regularizer(0)
    assert torch.isfinite(loss)
    loss.backward()
    assert float(branch.t_in.weight.grad.abs().sum()) > 0
    assert branch.t_out.weight.grad is None
    assert branch.gate_raw.grad is None


def test_save_and_load_keep_the_gate_the_scale_and_the_t_out_start(tmp_path):
    donor = donor_model()
    donor_dir = tmp_path / "donor"
    donor.save_pretrained(donor_dir)
    host = host_model()
    pristine = copy.deepcopy(host)
    branch = make_block_branch(host, str(donor_dir), [1, 2], 1, donor_config=donor.config, donor_id=str(donor_dir),
                               g_max=0.4, init_seed=3)
    seam = FeedbackSeam(host, [branch, make_plain_branch(host, g_max=0.4)], base_id="unused").eval()
    seam.to(torch.float32)
    set_gate(seam.branches[0], 0.2)
    set_gate(seam.branches[1], 0.1)
    assert branch.meta["t_out_init"] == "random"
    save_seam(seam, str(tmp_path / "saved"), base_id="unused")
    loaded, _ = load_seam(str(tmp_path / "saved"), host=pristine, dtype=torch.float32)
    for old, new in zip(seam.branches, loaded.branches):
        assert float(new.g_max) == pytest.approx(0.4)
        assert float(new.embed_rms) == pytest.approx(float(old.embed_rms))
        assert float(new.gate) == pytest.approx(float(old.gate))
        assert torch.equal(new.gate_raw, old.gate_raw)
        assert not new.legacy
    ids = plain_ids()
    with torch.no_grad():
        assert worst(seam(ids).logits, loaded(ids).logits) == 0.0


def test_a_new_branch_starts_random_and_a_legacy_branch_starts_at_zero():
    host = host_model()
    new = make_plain_branch(host)
    old = make_plain_branch(host, legacy=True)
    assert float(new.gate) == 0.0
    assert float(new.t_out.weight.abs().sum()) > 0
    assert float(new.t_out.weight.std()) == pytest.approx(1.0 / 32 ** 0.5, rel=0.2)
    assert float(old.gate) == 1.0
    assert float(old.t_out.weight.abs().sum()) == 0.0
    again = make_plain_branch(host)
    other = make_plain_branch(host, init_seed=7)
    assert torch.equal(new.t_out.weight, again.t_out.weight)
    assert not torch.equal(new.t_out.weight, other.t_out.weight)
    with pytest.raises(ValueError):
        make_plain_branch(host, gate_init=0.6, g_max=0.5)


def test_warm_start_fits_t_out_toward_the_donor_and_beats_the_random_start_on_held_out_rows():
    donor, donor_loss = train_planted_donor()
    host = planted_host()
    generator = torch.Generator()
    generator.manual_seed(31)
    t_in = ridge_t_in(host, 0, donor, 0, generator)
    branch = make_block_branch(host, donor, [0, 2], 0, t_in=t_in)
    seam = FeedbackSeam(host, [branch]).eval()
    before = branch.t_out.weight.detach().clone()
    ids = planted_batch(48, 12, generator)
    texts = [" ".join(str(int(v)) for v in row) for row in ids]
    reports = warm_start_t_out(seam, Tok(), donor, texts, batch_size=4, seq_len=12, min_rows=20)
    print("warm start", reports)
    report = reports[0]
    assert report["mode"] == "ridge"
    assert report["lag"] in (0, 1)
    best = report["cosine"][f"lag{report['lag']}"]
    assert best > 0
    assert best > report["random_cosine"][f"lag{report['lag']}"] + 0.05
    assert not torch.equal(before, branch.t_out.weight)
    # The fit has the same scale as a random start, and the gate did not move.
    assert float(branch.t_out.weight.pow(2).mean().sqrt()) == pytest.approx(1.0 / 32 ** 0.5, rel=1e-3)
    assert float(branch.gate) == 0.0
    # Too few rows: the random start stays and the report says so.
    seam2 = FeedbackSeam(planted_host(), [make_block_branch(planted_host(), donor, [0, 2], 0, t_in=t_in)]).eval()
    kept = seam2.branches[0].t_out.weight.detach().clone()
    again = warm_start_t_out(seam2, Tok(), donor, texts[:2], batch_size=2, seq_len=12, min_rows=10 ** 6)
    assert again[0]["mode"] == "random"
    assert torch.equal(kept, seam2.branches[0].t_out.weight)
