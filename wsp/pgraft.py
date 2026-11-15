"""Parallel graft: a frozen slice of a foreign model as a removable residual branch.

    h_j (base residual after layer j)
     |--------------------------------------------------------------> + --> base layer j+1
     '--> Enc --> [donor sink at pos 0] --> Donor[k..m] (frozen) --> (out - in) --> Dec --> gate --'

The base path is never changed. With the gate (or Dec) at zero the network
is exactly the base, so each graft is a branch that can be taken off. The
donor runs over the base's token positions on the states Enc makes; it never
sees its own tokenizer at inference. Only one network runs.

What is new here against wsp.synth and the earlier rounds:

  1. Functional alignment across tokenizers (stage 1). Enc trains so the
     donor, fed Enc's states at layer k and run to its own LM head, predicts
     what it predicts natively. Positions pair by character end offsets
     (wsp.align), and the loss is KL(native || grafted) over the donor's own
     vocabulary at the paired positions, so it works for any tokenizer pair.
     A short geometric warm-up (normalized MSE plus cosine against the
     donor's native layer-k state) starts it. The donor's sink state goes in
     at position 0.
  2. A quality gate before any task training: the share of the donor's
     native next-token accuracy that survives through the graft
     (alignment_report). Under about 0.3, the donor is a noise generator.
  3. Dec reads the donor's contribution (block output minus block input,
     which BlockPart returns), so Enc -> Dec cannot route around the donor.
  4. The identity-donor control: control="identity" skips the donor and
     gives Dec the Enc output instead, with the same Enc, Dec, gate,
     budget and data. If it matches the real graft, the donor did nothing.
  5. Integration (stage 2) on the base's LM loss with aids against the
     graft being ignored, all annealed to zero: a counterfactual utility
     hinge on domain text, relu(margin - (L_off - L_on)); stochastic depth
     on the base layers after the graft; and an L1 penalty on gate values,
     so training can pick among several insertion depths.

Heal data rule: callers pass texts from wsp.healdata pools, never benchmark
data. Not built yet: several grafts with separate write subspaces, and a
saved format (the round script scores in process).
"""

import math
import random

import torch
import torch.nn.functional as F
from torch import nn

from wsp.gate import TokenGate
from wsp.graft import Graft, attach_grafts, part_dtype, replace_hidden
from wsp.parts import AttnHeadsPart, BlockPart, build_block_part, find_decoder_layers
from wsp.stitch import (
    autocast,
    device_of,
    donor_sink,
    encode,
    feed_hooks,
    layer_outputs,
    matched_index,
    remove_hooks,
    with_sink,
)

DEFAULTS = {
    # stage 1: alignment
    "align_steps": 600,
    "geo_steps": 100,          # geometric warm-up steps; its weight then decays to 0 by 2 * geo_steps
    "align_lr": 1e-3,
    "align_batch": 4,
    "seq_len": 256,
    "donor_seq_len": 320,      # longer cut, so the donor covers the base's characters
    # stage 2: integration
    "int_steps": 1000,
    "int_lr": 1e-3,
    "gate_lr": 1e-2,
    "int_batch": 8,
    "domain_frac": 0.5,
    "utility_weight": 1.0,
    "utility_margin": 0.05,    # nats per token
    "drop_prob": 0.25,
    "drop_layers": 2,
    "gate_l1": 1e-3,
    "anneal_frac": 0.6,        # every aid falls linearly to 0 by this share of int_steps
    "enc_lr2": 1e-4,           # Enc learning rate in stage 2, when Enc params are passed to integrate
    "preserve_weight": 0.0,    # KL to a frozen reference on general batches (needs ref=)
}


# ------------------------------------------------------------------ modules


def rms(x, eps=1e-6):
    return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)


class ResidualMLP(nn.Module):
    """x + W2 gelu(W1 rms(x)), W2 starts at zero."""

    def __init__(self, d, hidden):
        super().__init__()
        self.up = nn.Linear(d, hidden)
        self.down = nn.Linear(hidden, d)
        nn.init.zeros_(self.down.weight)
        nn.init.zeros_(self.down.bias)

    def forward(self, x):
        return x + self.down(F.gelu(self.up(rms(x))))


class Enc(nn.Module):
    """Host state -> donor layer-k input: a linear map on rms(h), residual MLPs, a learned output scale."""

    def __init__(self, d_host, d_donor, depth=1, hidden=None, out_rms=1.0):
        super().__init__()
        self.proj = nn.Linear(d_host, d_donor)
        if hidden is None:
            hidden = d_donor // 4
        blocks = []
        for _ in range(max(0, depth - 1)):
            blocks.append(ResidualMLP(d_donor, hidden))
        self.blocks = nn.ModuleList(blocks)
        # nn.Linear's default init gives outputs of RMS about 1/sqrt(3) on unit-RMS input.
        self.log_scale = nn.Parameter(torch.tensor(math.log(max(out_rms, 1e-3) * math.sqrt(3.0))))

    def forward(self, h):
        y = self.proj(rms(h.to(torch.float32)))
        for block in self.blocks:
            y = block(y)
        return y * self.log_scale.exp()


class Dec(nn.Module):
    """Donor contribution -> host residual update: low-rank, optional residual MLPs, up starts at zero."""

    def __init__(self, d_donor, d_host, rank=64, depth=1, hidden=None):
        super().__init__()
        self.down = nn.Linear(d_donor, rank, bias=False)
        if hidden is None:
            hidden = rank * 4
        blocks = []
        for _ in range(max(0, depth - 1)):
            blocks.append(ResidualMLP(rank, hidden))
        self.blocks = nn.ModuleList(blocks)
        self.up = nn.Linear(rank, d_host, bias=False)
        nn.init.zeros_(self.up.weight)

    def forward(self, y):
        z = self.down(y)
        for block in self.blocks:
            z = block(z)
        return self.up(z)


class ParallelGraft(Graft):
    """h + gate(h) * Dec(scale * contribution), contribution = Donor[k..m](Enc(h)) - Enc(h).

    control: None (the real graft), "identity" (no donor: Dec reads Enc's
    output, same budget). forced_off returns h untouched. The update is zero
    at each sequence's first real position, so the host's own sink stays.
    Subclass of Graft, so attach_grafts keeps the block's KV cache in step.
    """

    def __init__(self, d_host, d_donor, part, enc, dec, control=None, gate_bias=0.0):
        super().__init__(d_host, d_donor, part)
        if control not in (None, "identity"):
            raise ValueError(f"unknown control {control!r}")
        self.w_in = enc
        self.w_out = dec
        self.control = control
        self.token_gate = TokenGate(d_host, gate_bias)
        self.register_buffer("in_scale", torch.ones((), dtype=torch.float32))
        self.forced_off = False
        self.last_gate = None
        self.last_real = None
        # When track is on, forward adds ||gate * Dec(.)|| / ||h|| over real tokens (not the first).
        self.track = False
        self.ratio_sum = 0.0
        self.ratio_count = 0

    @property
    def enc(self):
        return self.w_in

    @property
    def dec(self):
        return self.w_out

    def donor_input(self, h):
        x = self.map_in(h)
        if self.sink is not None:
            x = self.put_sink(x)
        return x

    def contribution(self, x):
        """What Dec reads: the donor's change to its residual stream, or Enc's output under the identity control."""
        if self.control == "identity":
            return x
        if isinstance(self.part, (PartBundle, TrunkBundle)):
            # The bundle sets each part's input size itself (unit RMS times the part's native RMS).
            if self.host_uses_cache:
                return self.part(x, attention_mask=self.pad_mask, cache_len=self.host_cache_len)
            return self.part(x, attention_mask=self.pad_mask)
        dtype = part_dtype(self.part)
        if self.host_uses_cache:
            y = self.part(x.to(dtype), attention_mask=self.pad_mask, cache_len=self.host_cache_len)
        else:
            y = self.part(x.to(dtype), attention_mask=self.pad_mask)
        return y.to(torch.float32)

    def first_position(self, h):
        batch, seq, _ = h.shape
        start = self.host_cache_len if self.host_uses_cache else 0
        if self.pad_mask is None:
            first = torch.zeros(batch, dtype=torch.long, device=h.device)
            has_real = torch.ones(batch, dtype=torch.bool, device=h.device)
        else:
            real = self.pad_mask.to(h.device) != 0
            first = real.long().argmax(dim=-1)
            has_real = real.any(dim=-1)
        positions = torch.arange(start, start + seq, device=h.device)
        hit = (positions.view(1, seq) == first.view(batch, 1)) & has_real.view(batch, 1)
        return hit.unsqueeze(-1)

    def real_mask(self, h):
        if self.pad_mask is None or self.host_uses_cache:
            return None
        return self.pad_mask[:, -h.shape[1]:]

    def forward(self, h):
        if self.forced_off:
            return h
        x = self.donor_input(h)
        y = self.contribution(x) * self.in_scale
        update = self.dec(y)
        real = self.real_mask(h)
        gate = self.token_gate(h, real)
        self.last_gate = gate
        self.last_real = real
        first = self.first_position(h)
        update = (update * gate).masked_fill(first, 0.0)
        if self.track:
            with torch.no_grad():
                ratio = update.norm(dim=-1) / h.float().norm(dim=-1).clamp(min=1e-6)
                keep = ~first.squeeze(-1)
                if real is not None:
                    keep = keep & (real.to(keep.device) != 0)
                self.ratio_sum += float(ratio[keep].sum())
                self.ratio_count += int(keep.sum())
        return h + update.to(h.dtype)


def grafts_of(model):
    return [g for g in getattr(model, "wsp_grafts", []) if isinstance(g, ParallelGraft)]


class grafts_off:
    """Context: every parallel graft returns h untouched."""

    def __init__(self, model):
        self.grafts = grafts_of(model)

    def __enter__(self):
        self.old = [g.forced_off for g in self.grafts]
        for g in self.grafts:
            g.forced_off = True

    def __exit__(self, *exc):
        for g, old in zip(self.grafts, self.old):
            g.forced_off = old


def hidden_size(model):
    config = model.config
    config = getattr(config, "text_config", None) or config
    return config.hidden_size


def num_layers(model):
    config = model.config
    config = getattr(config, "text_config", None) or config
    return config.num_hidden_layers


def build(base, donor, host_layer, first, last, control=None, enc_depth=1, dec_rank=64, dec_depth=1,
          gate_bias=0.0, out_rms=1.0, d_donor=None):
    """Make one ParallelGraft that runs donor layers first..last, and attach it after base layer host_layer.

    The block is the donor's own decoder layer code, built from the loaded
    donor's tensors (frozen). Under control="identity" no block is built.
    The identity control needs no donor model, only d_donor. Returns the graft.
    """
    d_host = hidden_size(base)
    if d_donor is None:
        d_donor = hidden_size(donor)
    part = None
    if control != "identity":
        tensors = {}
        for name, value in donor.state_dict().items():
            tensors[name] = value.detach()
        part = build_block_part(donor.config, first, last, tensors).to(device_of(donor))
    else:
        part = nn.Identity()
    enc = Enc(d_host, d_donor, depth=enc_depth, out_rms=out_rms)
    dec = Dec(d_donor, d_host, rank=dec_rank, depth=dec_depth)
    graft = ParallelGraft(d_host, d_donor, part, enc, dec, control=control, gate_bias=gate_bias)
    graft.donor_span = (first, last)
    device = device_of(base)
    graft.w_in.to(device)
    graft.w_out.to(device)
    graft.token_gate.to(device)
    graft.in_scale = graft.in_scale.to(device)
    graft.planned_layer = host_layer
    return graft


def attach(base, grafts):
    """Hook a list of built grafts onto the base, each after its planned host layer."""
    placements = []
    for graft in grafts:
        placements.append((graft.planned_layer, graft))
    return attach_grafts(base, placements)


def trainable(grafts, enc=True, dec=True, gate=True):
    params = []
    for g in grafts:
        if enc:
            params += list(g.enc.parameters())
        if dec:
            params += list(g.dec.parameters())
        if gate:
            params += list(g.token_gate.parameters())
    return params


def count(params):
    return sum(p.numel() for p in params)


# ------------------------------------------------------- stage 1: alignment


def _donor_native(donor, ids, mask, layer):
    """Native donor run: (final hidden after the last norm, hidden at the input of layer)."""
    with torch.no_grad():
        with autocast(device_of(donor)):
            out = donor.model(input_ids=ids, attention_mask=mask, output_hidden_states=True, use_cache=False)
    return out.last_hidden_state, out.hidden_states[layer]


def _host_state(base, ids, mask, layer):
    with torch.no_grad():
        with grafts_off(base):
            with autocast(device_of(base)):
                found = layer_outputs(base, [layer], ids, mask)
    return found[layer]


def _grafted_final(donor, layer, x, mask):
    """Run the donor from layer on x over the base's positions; return its final hidden (after the norm).

    The donor gets zero embeddings, so the layers before `layer` see nothing
    useful and carry no gradient, and a pre-hook swaps in x at `layer`.
    """
    holder = {"x": x}
    handles = feed_hooks(donor, layer, None, holder)
    try:
        dtype = next(donor.parameters()).dtype
        zeros = torch.zeros(x.shape[0], x.shape[1], hidden_size(donor), dtype=dtype, device=x.device)
        with autocast(device_of(donor)):
            out = donor.model(inputs_embeds=zeros, attention_mask=mask, use_cache=False)
    finally:
        remove_hooks(handles)
    return out.last_hidden_state


def _pairs(base_tok, donor_tok, texts, seq_len, donor_seq_len, device):
    ids_h, mask_h, pos_h, ends_h = encode(base_tok, texts, seq_len)
    ids_d, mask_d, pos_d, ends_d = encode(donor_tok, texts, donor_seq_len)
    b, hp, dp = matched_index(pos_h, ends_h, pos_d, ends_d)
    return (ids_h.to(device), mask_h.to(device), ids_d.to(device), mask_d.to(device),
            b.to(device), hp.to(device), dp.to(device))


def _kl(native_logits, graft_logits):
    t = torch.log_softmax(native_logits.float(), dim=-1)
    s = torch.log_softmax(graft_logits.float(), dim=-1)
    return (t.exp() * (t - s)).sum(-1)


def _nmse_cos(x, y):
    x = x.float()
    y = y.float()
    nmse = (x - y).pow(2).mean() / (y.pow(2).mean() + 1e-6)
    cos = 1.0 - F.cosine_similarity(x, y, dim=-1).mean()
    return nmse + cos


def set_sink(graft, donor, donor_tok, layer, texts):
    """Store the donor's mean state at position 0 at the input of `layer` (its attention sink)."""
    graft.set_sink(donor_sink(donor, donor_tok, layer, texts).to(device_of(donor)))


def functional_loss(base, base_tok, donor, donor_tok, graft, texts, cfg, geo_weight=0.0, stats=False):
    host_layer = graft.planned_layer
    layer = graft.donor_span[0]
    ids_h, mask_h, ids_d, mask_d, b, hp, dp = _pairs(base_tok, donor_tok, texts, cfg["seq_len"],
                                                    cfg["donor_seq_len"], device_of(base))
    if b.numel() == 0:
        return None, {}
    native_final, native_in = _donor_native(donor, ids_d, mask_d, layer)
    h = _host_state(base, ids_h, mask_h, host_layer)
    x = graft.enc(h)
    # encode pads on the right, so position 0 is every row's first real token.
    x = with_sink(x, graft.sink)
    final = _grafted_final(donor, layer, x, mask_h)
    head = donor.get_output_embeddings()
    with torch.no_grad():
        native_logits = head(native_final[b, dp]).float()
    graft_logits = head(final[b, hp]).float()
    kl = _kl(native_logits, graft_logits).mean()
    loss = kl
    geo = None
    if geo_weight > 0:
        geo = _nmse_cos(x[b, hp], native_in[b, dp])
        loss = loss + geo_weight * geo
    info = {"kl": float(kl.detach())}
    if geo is not None:
        info["geo"] = float(geo.detach())
    if stats:
        with torch.no_grad():
            # Next donor token after each paired donor position, where it is real.
            nxt = dp + 1
            ok = nxt < ids_d.shape[1]
            nxt = nxt.clamp(max=ids_d.shape[1] - 1)
            ok = ok & (mask_d[b, nxt] != 0)
            label = ids_d[b, nxt]
            native_top = native_logits.argmax(-1)
            graft_top = graft_logits.argmax(-1)
            info["pairs"] = int(b.numel())
            info["scored"] = int(ok.sum())
            info["native_correct"] = int(((native_top == label) & ok).sum())
            info["graft_correct"] = int(((graft_top == label) & ok).sum())
            info["agree"] = int((native_top == graft_top).sum())
            info["kl_sum"] = float(_kl(native_logits, graft_logits).sum())
    return loss, info


def _batches(texts, batch, seed):
    order = list(range(len(texts)))
    rng = random.Random(seed)
    rng.shuffle(order)
    position = 0
    while True:
        out = []
        while len(out) < batch:
            if position >= len(order):
                rng.shuffle(order)
                position = 0
            out.append(texts[order[position]])
            position += 1
        yield out


def align(base, base_tok, donor, donor_tok, graft, texts, cfg=None, seed=0, log_every=50):
    """Stage 1: train Enc (only) so the donor still computes what it natively computes. Returns the loss log."""
    cfg = dict(DEFAULTS, **(cfg or {}))
    params = list(graft.enc.parameters())
    opt = torch.optim.AdamW(params, lr=cfg["align_lr"], weight_decay=0.0)
    steps = cfg["align_steps"]
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 20) * 0.5 * (1 + math.cos(math.pi * s / steps)))
    log = []
    feed = _batches(texts, cfg["align_batch"], seed)
    for step in range(steps):
        geo_steps = cfg["geo_steps"]
        if step < geo_steps:
            geo_w = 1.0
        else:
            geo_w = max(0.0, 1.0 - (step - geo_steps) / max(1, geo_steps))
        loss, info = functional_loss(base, base_tok, donor, donor_tok, graft, next(feed), cfg, geo_weight=geo_w)
        if loss is None:
            continue
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        sched.step()
        info["step"] = step
        log.append(info)
        if log_every and step % log_every == 0:
            print(f"align step {step} " + " ".join(f"{k} {v:.4f}" for k, v in info.items() if isinstance(v, float)))
    return log


def alignment_report(base, base_tok, donor, donor_tok, graft, texts, cfg=None, batch=4):
    """Benchmark-free graft quality on held-out text: how much of the donor's native next-token skill survives.

    survival = grafted top-1 accuracy / native top-1 accuracy, on the donor's
    own next tokens at offset-paired positions. agree = top-1 agreement.
    """
    cfg = dict(DEFAULTS, **(cfg or {}))
    tot = {"pairs": 0, "scored": 0, "native_correct": 0, "graft_correct": 0, "agree": 0, "kl_sum": 0.0}
    with torch.no_grad():
        for start in range(0, len(texts), batch):
            _, info = functional_loss(base, base_tok, donor, donor_tok, graft, texts[start:start + batch], cfg,
                                      stats=True)
            for key in tot:
                tot[key] += info.get(key, 0)
    out = {"pairs": tot["pairs"], "kl": tot["kl_sum"] / max(1, tot["pairs"])}
    out["native_acc"] = tot["native_correct"] / max(1, tot["scored"])
    out["graft_acc"] = tot["graft_correct"] / max(1, tot["scored"])
    out["agree"] = tot["agree"] / max(1, tot["pairs"])
    out["survival"] = out["graft_acc"] / max(1e-9, out["native_acc"])
    return out


def calibrate_scale(base, tok, graft, texts, cfg=None, batch=4):
    """Set graft.in_scale so what Dec reads has RMS about 1 on these texts (the real graft and the control alike)."""
    cfg = dict(DEFAULTS, **(cfg or {}))
    total = 0.0
    n = 0
    graft.in_scale.fill_(1.0)
    with torch.no_grad():
        for start in range(0, min(len(texts), 64), batch):
            ids, mask, _, _ = encode(tok, texts[start:start + batch], cfg["seq_len"])
            ids = ids.to(device_of(base))
            mask = mask.to(device_of(base))
            h = _host_state(base, ids, mask, graft.planned_layer)
            graft.pad_mask = mask
            graft.host_uses_cache = False
            x = graft.donor_input(h)
            y = graft.contribution(x)
            keep = mask.bool().clone()
            keep[:, 0] = False
            total += float(y[keep].float().pow(2).mean(-1).sqrt().sum())
            n += int(keep.sum())
            graft.pad_mask = None
    value = total / max(1, n)
    graft.in_scale.fill_(1.0 / max(value, 1e-6))
    return value


# ----------------------------------------------------- stage 2: integration


class depth_dropper:
    """Hooks on base layers that, when active, make each hooked layer return its input (stochastic depth)."""

    def __init__(self, base, layers):
        self.active = False
        self.handles = []
        for index in layers:
            _, all_layers = find_decoder_layers(base, index)
            layer = all_layers[index]

            def hook(module, args, kwargs, output, self=self):
                if not self.active:
                    return output
                if len(args) > 0:
                    h_in = args[0]
                else:
                    h_in = kwargs["hidden_states"]
                return replace_hidden(output, lambda _: h_in)

            self.handles.append(layer.register_forward_hook(hook, with_kwargs=True))

    def remove(self):
        remove_hooks(self.handles)


def seq_lm_loss(model, enc, return_logits=False):
    """Per-sequence mean next-token loss, shape (batch,); with return_logits also the float logits."""
    with autocast(device_of(model)):
        full = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"], use_cache=False).logits
    logits = full[:, :-1].float()
    labels = enc["input_ids"][:, 1:]
    mask = enc["attention_mask"][:, 1:].float()
    ce = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1), reduction="none")
    ce = ce.view(labels.shape) * mask
    loss = ce.sum(1) / mask.sum(1).clamp(min=1.0)
    if return_logits:
        return loss, logits
    return loss


def drift_kl(ref, enc, logits):
    """Mean per-token KL(ref || model) on real tokens: the preservation term against a frozen reference."""
    with torch.no_grad():
        with autocast(device_of(ref)):
            ref_logits = ref(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"],
                             use_cache=False).logits[:, :-1].float()
    mask = enc["attention_mask"][:, 1:].float()
    kl = _kl(ref_logits, logits) * mask
    return kl.sum() / mask.sum().clamp(min=1.0)


def _tok_batch(tok, texts, seq_len, device):
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    enc = tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=seq_len)
    return {"input_ids": enc["input_ids"].to(device), "attention_mask": enc["attention_mask"].to(device)}


def mean_gate(grafts):
    vals = []
    for g in grafts:
        gate = g.last_gate
        if gate is None:
            continue
        real = g.last_real
        if real is None:
            vals.append(gate.mean())
        else:
            keep = real.to(gate.device).float().unsqueeze(-1)
            vals.append((gate * keep).sum() / keep.sum().clamp(min=1.0))
    if not vals:
        return None
    return torch.stack(vals).mean()


def integrate(base, tok, domain_texts, general_texts, params, cfg=None, seed=0, aids=True, log_every=100,
              ref=None, enc_params=None, reg_fn=None, probe_fn=None, probe_every=100):
    """Stage 2: train the given params on the base's LM loss over mixed domain and general text.

    With grafts and aids on, domain batches add the utility hinge (removing
    the graft should hurt) and stochastic depth on the base layers right
    after the first graft; every batch adds gate L1. Each aid falls
    linearly to 0 by anneal_frac of the steps. With no grafts (the LoRA
    arm) this is a plain LM heal on the same data. ref (a frozen copy of
    the base) adds preserve_weight * KL(ref || model) on general batches.
    enc_params, when given, train at enc_lr2. reg_fn(step) returns a loss
    tensor (or None) added to the step, for example the stage-1 alignment
    KL as a regularizer. probe_fn(step) returns a dict logged every
    probe_every steps (and at the end), for example graft fidelity.
    Returns the log.
    """
    cfg = dict(DEFAULTS, **(cfg or {}))
    grafts = grafts_of(base)
    torch.manual_seed(seed)
    rng = random.Random(seed + 7)
    gate_ids = set()
    for g in grafts:
        for p in g.token_gate.parameters():
            gate_ids.add(id(p))
    enc_ids = set(id(p) for p in (enc_params or []))
    groups = [{"params": [p for p in params if id(p) not in gate_ids and id(p) not in enc_ids], "lr": cfg["int_lr"]}]
    if enc_params:
        groups.append({"params": list(enc_params), "lr": cfg["enc_lr2"]})
        params = list(params) + [p for p in enc_params if id(p) not in set(id(q) for q in params)]
    gate_params = [p for p in params if id(p) in gate_ids]
    if gate_params:
        groups.append({"params": gate_params, "lr": cfg["gate_lr"]})
    opt = torch.optim.AdamW(groups, weight_decay=0.0)
    steps = cfg["int_steps"]
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 30) * 0.5 * (1 + math.cos(math.pi * s / steps)))
    dropper = None
    if grafts and aids and cfg["drop_prob"] > 0 and cfg["drop_layers"] > 0:
        first = min(g.planned_layer for g in grafts)
        layers = list(range(first + 1, min(num_layers(base), first + 1 + cfg["drop_layers"])))
        dropper = depth_dropper(base, layers)
    dom = _batches(domain_texts, cfg["int_batch"], seed)
    gen = _batches(general_texts, cfg["int_batch"], seed + 1)
    device = device_of(base)
    base.train()
    log = []
    try:
        for step in range(steps):
            anneal = max(0.0, 1.0 - step / max(1.0, cfg["anneal_frac"] * steps)) if aids else 0.0
            is_domain = rng.random() < cfg["domain_frac"]
            texts = next(dom) if is_domain else next(gen)
            enc = _tok_batch(tok, texts, cfg["seq_len"], device)
            if dropper is not None:
                dropper.active = is_domain and rng.random() < cfg["drop_prob"] * anneal
            lm_on, logits = seq_lm_loss(base, enc, return_logits=True)
            loss = lm_on.mean()
            info = {"step": step, "domain": is_domain, "lm": float(loss.detach())}
            if ref is not None and cfg["preserve_weight"] > 0 and not is_domain:
                drift = drift_kl(ref, enc, logits)
                loss = loss + cfg["preserve_weight"] * drift
                info["drift"] = float(drift.detach())
            del logits
            if reg_fn is not None:
                reg = reg_fn(step)
                if reg is not None:
                    loss = loss + reg
                    info["reg"] = float(reg.detach())
            if grafts and aids and is_domain and cfg["utility_weight"] > 0 and anneal > 0:
                with torch.no_grad():
                    with grafts_off(base):
                        lm_off = seq_lm_loss(base, enc)
                gain = lm_off - lm_on
                hinge = F.relu(cfg["utility_margin"] - gain).mean()
                loss = loss + cfg["utility_weight"] * anneal * hinge
                info["gain"] = float(gain.mean().detach())
            if grafts and cfg["gate_l1"] > 0:
                g = mean_gate(grafts)
                if g is not None:
                    loss = loss + cfg["gate_l1"] * g
                    info["gate"] = float(g.detach())
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()
            if probe_fn is not None and (step % probe_every == 0 or step == steps - 1):
                base.eval()
                info["probe"] = probe_fn(step)
                base.train()
            log.append(info)
            if log_every and step % log_every == 0:
                print("int " + " ".join(f"{k} {v:.4f}" if isinstance(v, float) else f"{k} {v}" for k, v in info.items()))
    finally:
        if dropper is not None:
            dropper.remove()
        base.eval()
    return log


def heldout_utility(base, tok, texts, cfg=None, batch=8, ref=None):
    """Held-out text: mean per-token LM loss with grafts on and off ({on, off, gain}), mean gate, the
    update ratio ||gate * Dec|| / ||h|| per token, and with ref the KL(ref || model) drift."""
    cfg = dict(DEFAULTS, **(cfg or {}))
    on = []
    off = []
    gates = []
    grafts = grafts_of(base)
    drifts = []
    for g in grafts:
        g.track = True
        g.ratio_sum = 0.0
        g.ratio_count = 0
    with torch.no_grad():
        for start in range(0, len(texts), batch):
            enc = _tok_batch(tok, texts[start:start + batch], cfg["seq_len"], device_of(base))
            loss_on, logits = seq_lm_loss(base, enc, return_logits=True)
            on.append(loss_on)
            if ref is not None:
                drifts.append(float(drift_kl(ref, enc, logits)))
            del logits
            g = mean_gate(grafts)
            if g is not None:
                gates.append(float(g))
            with grafts_off(base):
                off.append(seq_lm_loss(base, enc))
    on = float(torch.cat(on).mean())
    off = float(torch.cat(off).mean())
    out = {"on": on, "off": off, "gain": off - on}
    ratio_sum = sum(g.ratio_sum for g in grafts)
    ratio_count = sum(g.ratio_count for g in grafts)
    for g in grafts:
        g.track = False
    if ratio_count:
        out["update_ratio"] = ratio_sum / ratio_count
    if drifts:
        out["drift_kl"] = sum(drifts) / len(drifts)
    if gates:
        out["gate"] = sum(gates) / len(gates)
    return out


# ------------------------------------------- cross-tokenizer distillation


def vocab_map(base_tok, donor_tok):
    """Token ids present in both vocabularies by exact token string: (base_ids, donor_ids), aligned.

    Llama 3 and Qwen2.5 both use GPT-2 style byte-level tokens, so equal
    strings are the same bytes. Special tokens are left out.
    """
    base_vocab = base_tok.get_vocab()
    donor_vocab = donor_tok.get_vocab()
    specials = set(base_tok.all_special_tokens) | set(donor_tok.all_special_tokens)
    shared = sorted(t for t in base_vocab.keys() & donor_vocab.keys() if t not in specials)
    base_ids = torch.tensor([base_vocab[t] for t in shared], dtype=torch.long)
    donor_ids = torch.tensor([donor_vocab[t] for t in shared], dtype=torch.long)
    return base_ids, donor_ids


def _shared_kl(donor_logits, base_logits, vmap, temperature=1.0):
    """KL(donor || base) over the shared tokens, each side renormalized on them. Per row."""
    base_ids, donor_ids = vmap
    d = donor_logits.float()[:, donor_ids.to(donor_logits.device)] / temperature
    b = base_logits.float()[:, base_ids.to(base_logits.device)] / temperature
    return _kl(d, b)


def kd_loss(base, base_tok, donor, donor_tok, texts, vmap, cfg=None, stats=False):
    """Distill the donor into the (grafted) base at offset-paired positions, over the shared vocabulary.

    At a pair (host position i, donor position j) both tokens end at the same
    character, so both models predict the text that starts there. The donor
    runs natively (no gradient); gradients reach whatever in the base trains.
    Returns (loss, info); with stats also the top-1 agreement on shared tokens.
    """
    cfg = dict(DEFAULTS, **(cfg or {}))
    ids_h, mask_h, ids_d, mask_d, b, hp, dp = _pairs(base_tok, donor_tok, texts, cfg["seq_len"],
                                                    cfg["donor_seq_len"], device_of(base))
    if b.numel() == 0:
        return None, {}
    with torch.no_grad():
        with autocast(device_of(donor)):
            native = donor.model(input_ids=ids_d, attention_mask=mask_d, use_cache=False).last_hidden_state
        donor_logits = donor.get_output_embeddings()(native[b, dp]).float()
    with autocast(device_of(base)):
        logits = base(input_ids=ids_h, attention_mask=mask_h, use_cache=False).logits
    base_logits = logits[b, hp].float()
    kl = _shared_kl(donor_logits, base_logits, vmap)
    loss = kl.mean()
    info = {"kd": float(loss.detach())}
    if stats:
        base_ids, donor_ids = vmap
        dt = donor_logits[:, donor_ids.to(donor_logits.device)].argmax(-1)
        bt = base_logits[:, base_ids.to(base_logits.device)].argmax(-1)
        info["pairs"] = int(b.numel())
        info["agree"] = int((dt == bt).sum())
        info["kd_sum"] = float(kl.sum())
    return loss, info


def kd_report(base, base_tok, donor, donor_tok, texts, vmap, cfg=None, batch=4):
    """Held-out distillation gap: mean shared-vocab KL(donor || base) and top-1 agreement with the donor."""
    tot = {"pairs": 0, "agree": 0, "kd_sum": 0.0}
    with torch.no_grad():
        for start in range(0, len(texts), batch):
            _, info = kd_loss(base, base_tok, donor, donor_tok, texts[start:start + batch], vmap, cfg, stats=True)
            for key in tot:
                tot[key] += info.get(key, 0)
    return {"pairs": tot["pairs"], "kl": tot["kd_sum"] / max(1, tot["pairs"]),
            "agree": tot["agree"] / max(1, tot["pairs"])}


# ------------------------------------------------------ generic trainer


def train(base, groups, steps, step_fn, seed=0, warmup=30, log_every=100, probe_fn=None, probe_every=100):
    """Train param groups [{"params", "lr"}] for steps; step_fn(step) returns (loss, info). Returns the log.

    Cosine schedule with linear warm-up, grad clip 1.0, the base in train
    mode during the loop (donor blocks stay in eval mode).
    """
    torch.manual_seed(seed)
    groups = [g for g in groups if len(g["params"]) > 0]
    params = [p for g in groups for p in g["params"]]
    opt = torch.optim.AdamW(groups, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warmup) * 0.5 * (1 + math.cos(math.pi * s / steps)))
    log = []
    base.train()
    try:
        for step in range(steps):
            loss, info = step_fn(step)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()
            info = dict(info)
            info["step"] = step
            info["loss"] = float(loss.detach())
            if probe_fn is not None and (step % probe_every == 0 or step == steps - 1):
                base.eval()
                info["probe"] = probe_fn(step)
                base.train()
            log.append(info)
            if log_every and step % log_every == 0:
                print("train " + " ".join(f"{k} {v:.4f}" if isinstance(v, float) else f"{k} {v}"
                                          for k, v in info.items() if k != "probe"))
    finally:
        base.eval()
    return log


def mc_loss(model, tok, items, letter_ids, device):
    """Answer-letter cross-entropy: items are (prompt, gold index); logits at the last prompt token over the letters."""
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    enc = tok([p for p, _ in items], return_tensors="pt", padding=True).to(device)
    with autocast(device):
        logits = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"], use_cache=False).logits
    letters = logits[:, -1, :].float()[:, letter_ids]
    gold = torch.tensor([g for _, g in items], dtype=torch.long, device=device)
    loss = F.cross_entropy(letters, gold)
    acc = float((letters.argmax(-1) == gold).float().mean())
    tok.padding_side = "right"
    return loss, {"mc": float(loss.detach()), "mc_acc": acc}


def mc_soft_loss(model, tok, items, letter_ids, device, soft=None, kd_weight=0.0):
    """Answer-letter loss with optional distillation: CE to gold plus kd_weight * KL(soft || student) over the letters.

    items are (prompt, gold index); soft is a (batch, letters) tensor of the donor's probabilities over the same letters.
    info also holds the real token count of the batch ("tokens").
    """
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    enc = tok([p for p, _ in items], return_tensors="pt", padding=True).to(device)
    with autocast(device):
        logits = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"], use_cache=False).logits
    letters = logits[:, -1, :].float()[:, letter_ids]
    gold = torch.tensor([g for _, g in items], dtype=torch.long, device=device)
    ce = F.cross_entropy(letters, gold)
    loss = ce
    info = {"mc": float(ce.detach()), "mc_acc": float((letters.argmax(-1) == gold).float().mean()),
            "tokens": int(enc["attention_mask"].sum())}
    if soft is not None and kd_weight > 0:
        soft = soft.to(device).float().clamp(min=1e-8)
        soft = soft / soft.sum(-1, keepdim=True)
        logq = F.log_softmax(letters, dim=-1)
        kl = (soft * (soft.log() - logq)).sum(-1).mean()
        loss = ce + kd_weight * kl
        info["kd"] = float(kl.detach())
    tok.padding_side = "right"
    return loss, info


# ------------------------------------------------- subnetwork parts (bundles)


def _donor_tensors(donor):
    out = {}
    for name, value in donor.state_dict().items():
        out[name] = value.detach()
    return out


def resolve_library(spec, root="."):
    """A {"kind": "library", "id": ...} spec -> the plain part spec its units file describes."""
    import json
    import os

    with open(os.path.join(root, "library", "catalog.jsonl")) as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if row["id"] != spec["id"]:
                continue
            units = torch.load(os.path.join(root, row["units"]))
            out = {"kind": row["type"], "layer": int(row["layer"]), "model": row["model"], "id": row["id"],
                   "skill": (row.get("skill") or {}).get("name")}
            key = "indices" if row["type"] == "mlp_cluster" else "heads"
            values = units.get("indices", units.get("heads"))
            out[key] = [int(v) for v in torch.as_tensor(values).reshape(-1).tolist()]
            return out
    raise KeyError(spec["id"])


def _random_like(spec, config, seed):
    """Same kind, layer and size as spec, with random units of the same donor layer (the random control)."""
    text = getattr(config, "text_config", None) or config
    g = torch.Generator().manual_seed(seed)
    out = dict(spec)
    if spec["kind"] == "mlp_cluster":
        out["indices"] = torch.randperm(text.intermediate_size, generator=g)[:len(spec["indices"])].tolist()
    elif spec["kind"] == "attn_heads":
        out["heads"] = torch.randperm(text.num_attention_heads, generator=g)[:len(spec["heads"])].tolist()
    out["random"] = True
    return out


def expand_specs(specs, config, control=None, seed=0):
    """Turn whole-layer kinds into plain ones and apply the random control. Library specs must be resolved first."""
    text = getattr(config, "text_config", None) or config
    out = []
    for index, spec in enumerate(specs):
        spec = dict(spec)
        if spec["kind"] == "mlp_layer":
            spec = {"kind": "mlp_cluster", "layer": spec["layer"], "indices": list(range(text.intermediate_size))}
        elif spec["kind"] == "attn_layer":
            spec = {"kind": "attn_heads", "layer": spec["layer"], "heads": list(range(text.num_attention_heads))}
        if control == "random":
            spec = _random_like(spec, config, seed + 1000 * index)
        out.append(spec)
    return out


class PartBundle(nn.Module):
    """Several donor subnetworks fed the same translated state; returns the sum of their contributions.

    Each part's input is scaled to the RMS its own donor layer sees natively
    (in_rms, measured by bundle_input_rms) times a trained scalar, because
    parts from different layers expect different input sizes.
    """

    def __init__(self, parts, specs):
        super().__init__()
        self.parts = nn.ModuleList(parts)
        self.specs = list(specs)
        self.register_buffer("in_rms", torch.ones(len(parts)))
        self.gains = nn.Parameter(torch.ones(len(parts)))

    def train(self, mode=True):
        return super().train(False)

    def part_input(self, x, index):
        unit = rms(x.float())
        return (unit * (self.in_rms[index] * self.gains[index])).to(x.dtype)

    def forward(self, x, attention_mask=None, cache_len=None):
        total = None
        dtype = part_dtype(self.parts[0])
        for index, part in enumerate(self.parts):
            xi = self.part_input(x, index).to(dtype)
            if isinstance(part, AttnHeadsPart):
                if cache_len is None:
                    y = part(xi, attention_mask=attention_mask)
                else:
                    y = part(xi, attention_mask=attention_mask, cache_len=cache_len)
            else:
                y = part(xi)
            y = y.to(torch.float32)
            total = y if total is None else total + y
        return total


def build_bundle(donor, specs):
    from wsp.parts import build_attn_heads, build_mlp_cluster

    tensors = _donor_tensors(donor)
    parts = []
    for spec in specs:
        if spec["kind"] == "mlp_cluster":
            part = build_mlp_cluster(donor.config, spec["layer"], torch.tensor(spec["indices"]), tensors)
        elif spec["kind"] == "attn_heads":
            part = build_attn_heads(donor.config, spec["layer"], torch.tensor(spec["heads"]), tensors)
        else:
            raise ValueError(f"unknown part kind {spec['kind']!r}")
        parts.append(part.to(device_of(donor)))
    return PartBundle(parts, specs)


def _input_point(donor, spec):
    """The donor module whose input is this part's native input: the layer (attention) or its pre-MLP norm."""
    _, layers = find_decoder_layers(donor, spec["layer"])
    layer = layers[spec["layer"]]
    if spec["kind"] == "attn_heads":
        return layer
    return layer.post_attention_layernorm


def native_part_inputs(donor, specs, ids, mask):
    """Run the donor natively and return each part's input state, shape (batch, seq_d, d_donor)."""
    store = {}
    handles = []
    for index, spec in enumerate(specs):
        module = _input_point(donor, spec)

        def pre(module, args, kwargs, index=index):
            x = args[0] if len(args) > 0 else kwargs["hidden_states"]
            store[index] = x.detach()

        handles.append(module.register_forward_pre_hook(pre, with_kwargs=True))
    last = max(spec["layer"] for spec in specs)
    try:
        with torch.no_grad():
            with autocast(device_of(donor)):
                layer_outputs(donor, [last], ids, mask)
    finally:
        remove_hooks(handles)
    return [store[i] for i in range(len(specs))]


def bundle_input_rms(bundle, donor, donor_tok, texts, cfg=None, batch=4):
    """Set bundle.in_rms to each part's mean native input RMS over texts."""
    cfg = dict(DEFAULTS, **(cfg or {}))
    sums = torch.zeros(len(bundle.specs))
    count = 0
    for start in range(0, min(len(texts), 32), batch):
        ids, mask, _, _ = encode(donor_tok, texts[start:start + batch], cfg["donor_seq_len"])
        ins = native_part_inputs(donor, bundle.specs, ids.to(device_of(donor)), mask.to(device_of(donor)))
        keep = mask.bool().clone()
        keep[:, 0] = False
        for i, x in enumerate(ins):
            sums[i] += float(x[keep.to(x.device)].float().pow(2).mean(-1).sqrt().mean())
        count += 1
    bundle.in_rms.copy_((sums / max(1, count)).to(bundle.in_rms.device))
    return bundle.in_rms.tolist()


def part_output_loss(base, base_tok, donor, donor_tok, graft, texts, cfg, stats=False):
    """Output matching for a bundle: each part on the translated state vs the same part on its native input.

    Paired positions only; per-part normalized MSE plus cosine, averaged. The
    sink goes in at position 0 as in the block graft.
    """
    bundle = graft.part
    ids_h, mask_h, ids_d, mask_d, b, hp, dp = _pairs(base_tok, donor_tok, texts, cfg["seq_len"],
                                                    cfg["donor_seq_len"], device_of(base))
    if b.numel() == 0:
        return None, {}
    native_in = native_part_inputs(donor, bundle.specs, ids_d, mask_d)
    h = _host_state(base, ids_h, mask_h, graft.planned_layer)
    x = graft.enc(h)
    x = with_sink(x, graft.sink)
    dtype = part_dtype(bundle.parts[0])
    losses = []
    info = {}
    for index, part in enumerate(bundle.parts):
        with torch.no_grad():
            if isinstance(part, AttnHeadsPart):
                target = part(native_in[index].to(dtype), attention_mask=mask_d).float()
            else:
                target = part(native_in[index].to(dtype)).float()
        xi = bundle.part_input(x, index).to(dtype)
        if isinstance(part, AttnHeadsPart):
            got = part(xi, attention_mask=mask_h).float()
        else:
            got = part(xi).float()
        loss = _nmse_cos(got[b, hp], target[b, dp])
        losses.append(loss)
        if stats:
            with torch.no_grad():
                g = got[b, hp]
                t = target[b, dp]
                resid = (g - t).pow(2).sum()
                total = (t - t.mean(0, keepdim=True)).pow(2).sum().clamp(min=1e-9)
                info[f"r2_{index}"] = float(1.0 - resid / total)
    loss = torch.stack(losses).mean()
    info["part_loss"] = float(loss.detach())
    return loss, info


def align_parts(base, base_tok, donor, donor_tok, graft, texts, cfg=None, seed=0, log_every=100):
    """Stage 1 for a bundle: train Enc (and the bundle gains) on part output matching."""
    cfg = dict(DEFAULTS, **(cfg or {}))
    params = list(graft.enc.parameters()) + [graft.part.gains]
    groups = [{"params": params, "lr": cfg["align_lr"]}]
    feed = _batches(texts, cfg["align_batch"], seed)

    def step_fn(step):
        loss, info = part_output_loss(base, base_tok, donor, donor_tok, graft, next(feed), cfg)
        return loss, info

    return train(base, groups, cfg["align_steps"], step_fn, seed=seed, warmup=20, log_every=log_every)


def part_report(base, base_tok, donor, donor_tok, graft, texts, cfg=None, batch=4):
    """Held-out output-match R2 per part (1 = the part computes exactly what it computes natively)."""
    cfg = dict(DEFAULTS, **(cfg or {}))
    sums = {}
    n = 0
    with torch.no_grad():
        for start in range(0, len(texts), batch):
            _, info = part_output_loss(base, base_tok, donor, donor_tok, graft, texts[start:start + batch], cfg,
                                       stats=True)
            for key, value in info.items():
                if key.startswith("r2_"):
                    sums[key] = sums.get(key, 0.0) + value
            n += 1
    return {key: value / max(1, n) for key, value in sums.items()}


def build_bundle_graft(base, donor, host_layer, specs, control=None, enc_depth=3, dec_rank=256, dec_depth=1,
                       gate_bias=0.0, d_donor=None):
    """A ParallelGraft whose part is a PartBundle of donor subnetworks (or nothing, for the identity control)."""
    d_host = hidden_size(base)
    if d_donor is None:
        d_donor = hidden_size(donor)
    if control == "identity":
        part = nn.Identity()
        graft_control = "identity"
    else:
        part = build_bundle(donor, specs)
        graft_control = None
    enc = Enc(d_host, d_donor, depth=enc_depth)
    dec = Dec(d_donor, d_host, rank=dec_rank, depth=dec_depth)
    graft = ParallelGraft(d_host, d_donor, part, enc, dec, control=graft_control, gate_bias=gate_bias)
    graft.donor_span = (min(s["layer"] for s in specs), max(s["layer"] for s in specs))
    graft.planned_layer = host_layer
    device = device_of(base)
    for module in (graft.w_in, graft.w_out, graft.token_gate):
        module.to(device)
    graft.in_scale = graft.in_scale.to(device)
    return graft


# --------------------------------- trunk bundles: subnetworks read in their native context


class TrunkBundle(nn.Module):
    """Donor layers first..last (frozen) build each subnetwork's native input; only the named
    subnetworks' outputs come back (their sum), not the trunk's change to the residual.

    Each part reads the trunk's state at its own input point (the layer input
    for attention heads, the pre-MLP norm input for MLP neurons), captured by
    hooks on the trunk's own donor layers. Parts must sit in first..last.
    So the question is whether these units, in their own context, carry
    what the whole block delivers. Alignment is the block's functional KL
    (Enc feeds layer `first`).
    """

    def __init__(self, trunk, first, parts, specs):
        super().__init__()
        self.trunk = trunk
        self.first = first
        self.parts = nn.ModuleList(parts)
        self.specs = list(specs)
        self._store = {}
        self._handles = []
        for index, spec in enumerate(specs):
            layer = trunk.layers[spec["layer"] - first]
            module = layer if spec["kind"] == "attn_heads" else layer.post_attention_layernorm

            def pre(module, args, kwargs, index=index):
                x = args[0] if len(args) > 0 else kwargs["hidden_states"]
                self._store[index] = x

            self._handles.append(module.register_forward_pre_hook(pre, with_kwargs=True))

    def train(self, mode=True):
        return super().train(False)

    def forward(self, x, attention_mask=None, cache_len=None):
        self._store = {}
        x = x.to(part_dtype(self.trunk))
        if cache_len is None:
            self.trunk(x, attention_mask=attention_mask)
        else:
            self.trunk(x, attention_mask=attention_mask, cache_len=cache_len)
        total = None
        for index, part in enumerate(self.parts):
            xi = self._store[index]
            if isinstance(part, AttnHeadsPart):
                if cache_len is None:
                    y = part(xi, attention_mask=attention_mask)
                else:
                    y = part(xi, attention_mask=attention_mask, cache_len=cache_len)
            else:
                y = part(xi)
            y = y.to(torch.float32)
            total = y if total is None else total + y
        self._store = {}
        return total


def build_trunk_bundle_graft(base, donor, host_layer, first, specs, enc_depth=3, dec_rank=256, dec_depth=1,
                             gate_bias=0.0):
    """ParallelGraft whose part is a TrunkBundle: donor layers first..max(spec layer), then the named parts."""
    from wsp.parts import build_attn_heads, build_mlp_cluster

    specs = [s for s in specs if s["layer"] >= first]
    if not specs:
        raise ValueError(f"no part sits at or after donor layer {first}")
    last = max(s["layer"] for s in specs)
    tensors = _donor_tensors(donor)
    trunk = build_block_part(donor.config, first, last, tensors).to(device_of(donor))
    parts = []
    for spec in specs:
        if spec["kind"] == "mlp_cluster":
            part = build_mlp_cluster(donor.config, spec["layer"], torch.tensor(spec["indices"]), tensors)
        else:
            part = build_attn_heads(donor.config, spec["layer"], torch.tensor(spec["heads"]), tensors)
        parts.append(part.to(device_of(donor)))
    bundle = TrunkBundle(trunk, first, parts, specs)
    d_host = hidden_size(base)
    d_donor = hidden_size(donor)
    enc = Enc(d_host, d_donor, depth=enc_depth)
    dec = Dec(d_donor, d_host, rank=dec_rank, depth=dec_depth)
    graft = ParallelGraft(d_host, d_donor, bundle, enc, dec, gate_bias=gate_bias)
    graft.donor_span = (first, last)
    graft.planned_layer = host_layer
    device = device_of(base)
    for module in (graft.w_in, graft.w_out, graft.token_gate):
        module.to(device)
    graft.in_scale = graft.in_scale.to(device)
    return graft
