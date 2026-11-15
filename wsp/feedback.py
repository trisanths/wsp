"""Feedback seams: a component writes into the NEXT token's input, not into the host mid-stack.

Run 14 and 14b showed that a trained input stitch feeds a donor block almost
losslessly, but the host's few remaining layers cannot read what the block
writes back into the residual stream. A random block did as well as a real
one. The reader is the bottleneck.

A feedback seam gives the output a place where the whole host stack reads
it. The idea comes from "Latent Feedback Is a Compute and Data Amplifier"
(Huang et al., 2026): each token's top-layer state goes through a learned
map W and is ADDED TO THE NEXT TOKEN'S INPUT. At generation this is one
extra vector per step and the same KV cache.

A branch has these parts:

    read_layer k   the host layer whose output the branch reads
    T_in           a stitch from the host width to the donor width
    component      a range of the donor's own decoder layers (wsp.parts
                   BlockPart, with its own KV cache and rotary base),
                   or None
    T_out          a bias-free linear map to the host width, random or
                   ridge-fitted start (never zero)
    gate           g_max * tanh(raw), raw is one trained scalar that starts
                   at 0, so a fresh seam equals the host exactly

For position t the branch computes (u is the unit-RMS version of the
branch output)

    u_t  = normalize(component(T_in(h_t^k)) - T_in(h_t^k))        component given
    u_t  = normalize(top_state_t)                                 component None
    fb_t = gate * embed_rms * y_t / |y_t|,   y_t = T_out(u_t)

and the host input at position t+1 is embed(token_{t+1}) + sum of fb_t over
the branches. Position 0 (the first real token of each row) gets nothing.
embed_rms is the typical L2 norm of a row of the host's input embedding
table. So the norm of the vector a branch writes is gate times the embedding's
own norm, whatever the raw scale of the branch output or of T_out. The first
real run (2026-11-14) used fb = gate * T_out(raw output) with T_out at zero
and the gate at 1. Qwen residual states have huge outlier dimensions, and
the first Adam step moved every T_out entry by the learning rate, so the
vector added to the next input jumped to many times the embedding norm and
destroyed the model. Adam moves each entry by about lr in the first step
whatever the gradient size, so gradient clipping cannot stop that jump. The
normalized form bounds it: one step changes the written vector by at most
g_max * lr * embed_rms in norm through the gate, and T_out only rotates it.
legacy=True keeps the old form for the regression test.

Three ways to run it, which agree where they should:

  generate / forward_cached   token by token, host KV cache and one private
                              KV cache per component. Exact recursion: the
                              feedback for step t+1 comes from step t.
  forward_sequential          the same recursion with no cache (slow, for
                              tests). Equals forward_cached.
  forward(passes=n)           teacher forced, n parallel passes. Pass 1 has
                              no feedback. Pass p+1 feeds the branch outputs
                              of pass p, shifted by one position. After n
                              passes, logits at positions 0..n-1 are exact
                              and later positions approach the recursion.
                              Only the last pass builds the graph by
                              default (grad_through_passes=False), as in the
                              paper: the states of earlier passes are data,
                              and the maps T_in, T_out, the gate, and the
                              host LoRA learn through the last pass.

Training (round 22):
  stage 1  fit_t_in: T_in trains on the component's own fidelity with the
           wsp.stitch donor_lm objective (the donor's LM loss with T_in's
           output at the block input), after a ridge start, with the sink.
  stage 2  heal_distill: T_out, gates, a host LoRA r16, and T_in (small lr,
           with the stage-1 fidelity loss as a regularizer) train on
           KL(donor || seam) on skill pool text plus anchor_weight *
           KL(frozen base || seam) on general pool text. One pass with
           probability 0.15, else two, or three with probability 0.3 after
           the warmup, so the read layer sees fed-back inputs. No LM loss on
           gold answers. The loop halts when the two-pass KD runs away.
           warm_start_t_out fits T_out by ridge before stage 2.
"""

import contextlib
import json
import math
import os
import random
import time
from types import SimpleNamespace

import torch
from torch import nn

from wsp.graft import part_dtype
from wsp.insert import add_base_lora, build_stitch_in, donor_block_tensors, donor_config_of, stitch_config
from wsp.lora import apply_lora, load_lora_state, lora_modules, lora_state_dict
from wsp.parts import _text_config, build_block_part, find_decoder_layers

FORMAT = "wsp-feedback-2"
OUT_MODES = ("delta", "full")


# ---------------------------------------------------------------- small helpers


def shift_feedback(fb, mask):
    """Move the feedback of position t to position t + 1. Position 0 gets zeros.

    A position gets feedback only when it and the position before it are real
    tokens (mask 1). So the first real token of a left-padded row gets none,
    and a pad position never leaks into a real one.
    """
    batch, _, width = fb.shape
    zero = torch.zeros(batch, 1, width, dtype=fb.dtype, device=fb.device)
    moved = torch.cat([zero, fb[:, :-1]], dim=1)
    if mask is None:
        return moved
    real = mask != 0
    valid = torch.cat([torch.zeros_like(real[:, :1]), real[:, 1:] & real[:, :-1]], dim=1)
    return moved * valid.unsqueeze(-1).to(moved.dtype)


def first_real(mask):
    """Bool (batch, seq): True at the first real position of each row."""
    real = mask != 0
    return real & (real.long().cumsum(-1) == 1)


def put_sink(x, sink, mask, past=0):
    """Replace x at each row's first real position with the sink vector.

    mask is the full 2D mask (batch, past + new). x holds the new positions
    only, so a decode step that is not the row's first real token changes
    nothing.
    """
    hit = first_real(mask)[:, past:]
    return torch.where(hit.unsqueeze(-1), sink.to(device=x.device, dtype=x.dtype).view(1, 1, -1), x)


def positions_of(mask):
    """Position ids that count only real tokens, so left padding does not shift a row."""
    return (mask.long().cumsum(-1) - 1).clamp(min=0)


def random_range(n_layers, first, last, seed):
    """A seeded donor layer range of the same length as [first, last], never the same range."""
    length = last - first + 1
    starts = []
    for start in range(0, n_layers - length + 1):
        if start != first:
            starts.append(start)
    if len(starts) == 0:
        raise ValueError(f"the donor has {n_layers} layers, so no other range of {length} layers exists")
    pick = random.Random(seed).choice(starts)
    return [pick, pick + length - 1]


# ---------------------------------------------------------------- branch


def rms_normalize(x, eps=1e-6):
    """Divide each vector by its root mean square over the last dim. A zero vector stays zero."""
    return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)


def embed_rms_of(host, chunk=8192):
    """Typical L2 norm of a row of the host's input embedding table: sqrt(mean of the squared row norms)."""
    weight = host.get_input_embeddings().weight
    total = 0.0
    for first in range(0, weight.shape[0], chunk):
        total += float(weight[first:first + chunk].detach().float().pow(2).sum())
    return math.sqrt(total / weight.shape[0])


def random_t_out(t_out, seed):
    """Small random start for T_out: normal with std 1/sqrt(width), from a private generator (the global RNG is untouched)."""
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    width = t_out.weight.shape[1]
    with torch.no_grad():
        t_out.weight.copy_(torch.randn(t_out.weight.shape, generator=generator) / math.sqrt(width))


class Branch(nn.Module):
    """One feedback branch. part None makes the plain latent feedback of the paper (the capacity control).

    legacy=False (default): the written vector has norm gate * embed_rms (see
    the module docstring). The gate is g_max * tanh(gate_raw) and starts at 0
    unless gate_init says otherwise (gate_init is a gate value, |gate_init| <
    g_max). T_out starts random with std 1/sqrt(width).
    legacy=True: the round 22 first-run form, fb = gate * T_out(raw output),
    T_out at zero, gate at gate_init (default 1). Kept for the regression test.
    """

    def __init__(self, d_host, read_layer=None, t_in=None, part=None, d_comp=None, gate_init=None, sink=None,
                 out_mode="delta", meta=None, g_max=0.5, embed_rms=1.0, legacy=False, init_seed=0):
        super().__init__()
        if out_mode not in OUT_MODES:
            raise ValueError(f"out_mode must be one of {OUT_MODES}, got {out_mode!r}")
        if part is not None and (t_in is None or read_layer is None or d_comp is None):
            raise ValueError("a branch with a component needs read_layer, t_in, and d_comp")
        if g_max <= 0:
            raise ValueError("g_max must be positive")
        self.read_layer = read_layer
        self.t_in = t_in
        self.part = part
        self.out_mode = out_mode
        self.legacy = bool(legacy)
        self.meta = meta if meta is not None else {}
        self.meta["legacy"] = self.legacy
        self.meta["g_max"] = float(g_max)
        self.meta["embed_rms"] = float(embed_rms)
        width = d_host if part is None else d_comp
        self.t_out = nn.Linear(width, d_host, bias=False)
        self.register_buffer("g_max", torch.tensor(float(g_max), dtype=torch.float32))
        self.register_buffer("embed_rms", torch.tensor(float(embed_rms), dtype=torch.float32))
        if self.legacy:
            nn.init.zeros_(self.t_out.weight)
            raw = 1.0 if gate_init is None else float(gate_init)
        else:
            random_t_out(self.t_out, init_seed)
            value = 0.0 if gate_init is None else float(gate_init)
            if abs(value) >= g_max:
                raise ValueError(f"gate_init {value} must be inside (-g_max, g_max) = (-{g_max}, {g_max})")
            raw = math.atanh(value / g_max)
        self.meta["t_out_init"] = "zeros" if self.legacy else "random"
        self.gate_raw = nn.Parameter(torch.tensor(raw, dtype=torch.float32))
        self.register_buffer("sink", None)
        if sink is not None:
            self.sink = sink.detach().to(torch.float32).reshape(-1).clone()

    @property
    def gate(self):
        """The effective gate (a tensor with a graph to gate_raw)."""
        if self.legacy:
            return self.gate_raw
        return self.g_max * torch.tanh(self.gate_raw)

    def pre(self, read, mask, past=0, cache_len=None):
        """What T_out reads: the branch output, RMS-normalized per position (raw in legacy mode)."""
        if self.part is None:
            vec = read.float()
        else:
            x = self.t_in(read.float())
            if self.sink is not None:
                x = put_sink(x, self.sink, mask, past)
            delta = self.part(x.to(part_dtype(self.part)), attention_mask=mask, cache_len=cache_len)
            vec = delta.float()
            if self.out_mode == "full":
                vec = vec + x.float()
        if self.legacy:
            return vec
        return rms_normalize(vec)

    def write(self, vec):
        """The vector added to the next input, from the normalized branch output."""
        y = self.t_out(vec)
        if self.legacy:
            return self.gate * y
        unit = y * torch.rsqrt(y.pow(2).sum(dim=-1, keepdim=True) + 1e-12)
        return self.gate * self.embed_rms * unit

    def output(self, read, mask, past=0, cache_len=None):
        """The feedback vectors for the positions in read, shape (batch, seq, d_host).

        read is the host state at read_layer (the top state for a plain
        branch). mask is the full 2D mask (batch, past + seq). cache_len is
        None for a whole-sequence call, else the number of cached tokens.
        """
        return self.write(self.pre(read, mask, past, cache_len))


# ---------------------------------------------------------------- the seam


class FeedbackSeam(nn.Module):
    """A host causal LM plus feedback branches. See the module docstring for the three run modes."""

    def __init__(self, host, branches=None, base_id=None):
        super().__init__()
        self.host = host
        self.branches = nn.ModuleList(list(branches) if branches is not None else [])
        self.base_id = base_id

    # ---- the pieces a scorer expects from a model
    @property
    def config(self):
        return self.host.config

    @property
    def generation_config(self):
        return self.host.generation_config

    def get_input_embeddings(self):
        return self.host.get_input_embeddings()

    def get_output_embeddings(self):
        return self.host.get_output_embeddings()

    def add_branch(self, branch):
        self.branches.append(branch)
        return branch

    # ---- shared steps
    def prepare(self, input_ids, attention_mask):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        return attention_mask, positions_of(attention_mask)

    def host_pass(self, input_ids, mask, pos, fb_in, cache=None, use_cache=False):
        embed = self.host.get_input_embeddings()(input_ids)
        if fb_in is not None:
            embed = embed + fb_in.to(embed.dtype)
        return self.host(inputs_embeds=embed, attention_mask=mask, position_ids=pos, past_key_values=cache,
                         use_cache=use_cache, output_hidden_states=True)

    def reads_of(self, hidden_states):
        """The host state each branch reads. hidden_states[i + 1] is the output of layer i; the last is normed."""
        reads = []
        for branch in self.branches:
            if branch.part is None:
                reads.append(hidden_states[-1])
            else:
                reads.append(hidden_states[branch.read_layer + 1])
        return reads

    def feedback_of(self, reads, mask, past=0, cache_len=None):
        total = None
        for branch, read in zip(self.branches, reads):
            out = branch.output(read, mask, past, cache_len)
            if total is None:
                total = out
            else:
                total = total + out
        return total

    # ---- run mode 3: teacher forced, parallel passes
    def forward(self, input_ids, attention_mask=None, passes=2, grad_through_passes=False):
        """Teacher-forced logits with n passes. Returns an object with .logits."""
        if passes < 1:
            raise ValueError("passes must be at least 1")
        mask, pos = self.prepare(input_ids, attention_mask)
        reads = None
        out = None
        for index in range(passes):
            last = index == passes - 1
            graph = last or grad_through_passes
            context = contextlib.nullcontext() if graph else torch.no_grad()
            with context:
                fb_in = None
                if reads is not None and len(self.branches) > 0:
                    fb_in = shift_feedback(self.feedback_of(reads, mask), mask)
                out = self.host_pass(input_ids, mask, pos, fb_in)
                if not last and len(self.branches) > 0:
                    reads = self.reads_of(out.hidden_states)
        return SimpleNamespace(logits=out.logits)

    # ---- run mode 2: exact recursion, no cache
    def forward_sequential(self, input_ids, attention_mask=None):
        """Exact recursion with no cache: step t runs the whole prefix again. Slow. Returns logits (batch, seq, vocab)."""
        mask, pos = self.prepare(input_ids, attention_mask)
        batch, length = input_ids.shape
        steps = []
        fb_prev = None
        for t in range(length):
            prefix_mask = mask[:, :t + 1]
            fb_in = None
            if fb_prev is not None:
                zero = torch.zeros(batch, 1, fb_prev.shape[-1], dtype=fb_prev.dtype, device=fb_prev.device)
                fb_in = shift_feedback(torch.cat([fb_prev, zero], dim=1), prefix_mask)
            out = self.host_pass(input_ids[:, :t + 1], prefix_mask, pos[:, :t + 1], fb_in)
            steps.append(out.logits[:, -1])
            if len(self.branches) > 0:
                fb_prev = self.feedback_of(self.reads_of(out.hidden_states), prefix_mask)
        return torch.stack(steps, dim=1)

    # ---- run mode 1: exact recursion with KV caches
    def start_state(self):
        import transformers

        return {"cache": transformers.DynamicCache(), "fb": None, "j": 0}

    def step(self, state, token, mask):
        """Run one column. token (batch, 1), mask (batch, j + 1) is the full mask. Returns logits (batch, vocab)."""
        j = state["j"]
        pos = positions_of(mask)[:, -1:]
        fb_in = None
        if state["fb"] is not None:
            real = mask != 0
            valid = (real[:, -1] & real[:, -2]).view(-1, 1, 1)
            fb_in = state["fb"] * valid.to(state["fb"].dtype)
        out = self.host_pass(token, mask, pos, fb_in, cache=state["cache"], use_cache=True)
        state["cache"] = out.past_key_values
        if len(self.branches) > 0:
            state["fb"] = self.feedback_of(self.reads_of(out.hidden_states), mask, past=j, cache_len=j)
        state["j"] = j + 1
        return out.logits[:, -1]

    def forward_cached(self, input_ids, attention_mask=None):
        """The exact recursion with caches, over given tokens. Returns logits (batch, seq, vocab)."""
        mask, _ = self.prepare(input_ids, attention_mask)
        state = self.start_state()
        steps = []
        for t in range(input_ids.shape[1]):
            steps.append(self.step(state, input_ids[:, t:t + 1], mask[:, :t + 1]))
        return torch.stack(steps, dim=1)

    @torch.no_grad()
    def generate(self, input_ids, attention_mask=None, max_new_tokens=20, min_new_tokens=0, do_sample=False,
                 pad_token_id=None, eos_token_id=None, stopping_criteria=None, return_logits=False, **ignored):
        """Greedy generation with the exact recursion and KV caches.

        The prompt runs token by token too, because the feedback of each
        prompt token depends on the one before it. Extra keyword arguments of
        transformers generate (temperature=None, top_p=None, and so on) are
        accepted and ignored. Returns the ids, or (ids, logits) with
        return_logits, where logits[:, n] chose new token n.
        """
        if do_sample:
            raise ValueError("FeedbackSeam.generate is greedy only")
        mask = attention_mask
        if mask is None:
            mask = torch.ones_like(input_ids)
        batch = input_ids.shape[0]
        eos = []
        if eos_token_id is not None:
            eos = [eos_token_id] if isinstance(eos_token_id, int) else list(eos_token_id)
        pad = pad_token_id
        if pad is None:
            pad = eos[0] if len(eos) > 0 else 0
        eos_ids = torch.tensor(eos, dtype=torch.long, device=input_ids.device)

        state = self.start_state()
        logits = None
        for j in range(input_ids.shape[1]):
            logits = self.step(state, input_ids[:, j:j + 1], mask[:, :j + 1])
        ids = input_ids
        finished = torch.zeros(batch, dtype=torch.bool, device=input_ids.device)
        scores = []
        for n in range(max_new_tokens):
            row = logits.float()
            scores.append(row)
            if len(eos) > 0 and n < min_new_tokens:
                row = row.clone()
                row[:, eos_ids] = float("-inf")
            token = row.argmax(dim=-1)
            token = torch.where(finished, torch.full_like(token, pad), token)
            ids = torch.cat([ids, token.unsqueeze(1)], dim=1)
            mask = torch.cat([mask, torch.ones(batch, 1, dtype=mask.dtype, device=mask.device)], dim=1)
            if len(eos) > 0:
                finished = finished | torch.isin(token, eos_ids)
            if stopping_criteria is not None:
                finished = finished | stopping_criteria(ids, None).to(finished.device).bool()
            if bool(finished.all()) or n == max_new_tokens - 1:
                break
            logits = self.step(state, token.unsqueeze(1), mask)
        if return_logits:
            return ids, torch.stack(scores, dim=1)
        return ids

    # ---- training sets
    def param_groups(self):
        """Parameter lists by kind: t_in, t_out, gate, lora."""
        groups = {"t_in": [], "t_out": [], "gate": [], "lora": []}
        for branch in self.branches:
            if branch.t_in is not None:
                groups["t_in"] += list(branch.t_in.parameters())
            groups["t_out"] += list(branch.t_out.parameters())
            groups["gate"].append(branch.gate_raw)
        for _, module in lora_modules(self.host):
            groups["lora"].append(module.lora_A)
            groups["lora"].append(module.lora_B)
        return groups

    def set_trainable(self, names=("t_out", "gate", "lora")):
        """Freeze everything, then train the named groups. Stage 2 is the default. Returns the groups."""
        for param in self.parameters():
            param.requires_grad_(False)
        groups = self.param_groups()
        for name in names:
            for param in groups[name]:
                param.requires_grad_(True)
        return groups

    def add_host_lora(self, rank, alpha=None, layers=None):
        """Put a LoRA of this rank on the host decoder layers (layers: a list of indices, default all).

        A branch reads the host state at read_layer through a frozen T_in. A
        LoRA on layers at or below read_layer would move that state away from
        what T_in was fitted on, so round 22 puts the LoRA on the layers above
        the read layer only. Returns the LoRA parameters.
        """
        return add_base_lora(self.host, rank, alpha, layers)

    # ---- counts
    def count_params(self):
        return count_params(self)


# ---------------------------------------------------------------- controls and the frozen base


@contextlib.contextmanager
def disabled(model):
    """Turn off everything a heal added: LoRA scales go to 0, and insertion gates go to 0.

    Inside the block, a plain host call gives the frozen base's logits.
    """
    host = model.host if isinstance(model, FeedbackSeam) else model
    modules = []
    for _, module in lora_modules(host):
        modules.append(module)
    scales = []
    for module in modules:
        scales.append(module.scale)
        module.scale = 0.0
    gates = []
    old = []
    for graft in getattr(host, "wsp_grafts", []):
        gates.append(graft.w_out.gate)
        old.append(graft.w_out.gate.detach().clone())
    with torch.no_grad():
        for gate in gates:
            gate.zero_()
    try:
        yield
    finally:
        for module, scale in zip(modules, scales):
            module.scale = scale
        with torch.no_grad():
            for gate, value in zip(gates, old):
                gate.copy_(value)


def base_logits(model, input_ids, attention_mask):
    """Logits of the frozen base under a model that has a LoRA, insertions, or feedback branches."""
    host = model.host if isinstance(model, FeedbackSeam) else model
    with disabled(model), torch.no_grad():
        return host(input_ids=input_ids, attention_mask=attention_mask,
                    position_ids=positions_of(attention_mask), use_cache=False).logits


class BaseView:
    """The frozen base inside a healed model, as a callable for wsp.probe (it has parameters() and a call)."""

    def __init__(self, model):
        self.model = model
        self.host = model.host if isinstance(model, FeedbackSeam) else model

    def parameters(self):
        return self.host.parameters()

    def __call__(self, input_ids=None, attention_mask=None):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        return SimpleNamespace(logits=base_logits(self.model, input_ids, attention_mask))


# ---------------------------------------------------------------- building


def make_plain_branch(host, gate_init=None, g_max=0.5, legacy=False, init_seed=0, embed_rms=None):
    """The paper's latent feedback: T_out applied to the host's top-layer state. The capacity control."""
    d_host = _text_config(host.config).hidden_size
    if embed_rms is None:
        embed_rms = embed_rms_of(host)
    branch = Branch(d_host, gate_init=gate_init, meta={"kind": "plain", "d_host": int(d_host)}, g_max=g_max,
                    embed_rms=embed_rms, legacy=legacy, init_seed=init_seed)
    branch.to(host.get_input_embeddings().weight.device)
    return branch


def make_block_branch(host, donor, donor_layers, read_layer, t_in=None, stitch_kind="linear", stitch_hidden=None,
                      stitch_rank=None, donor_config=None, donor_id=None, sink=None, gate_init=None,
                      out_mode="delta", control=None, control_seed=None, cache_dir=None, g_max=0.5, legacy=False,
                      init_seed=0, embed_rms=None):
    """A branch with a donor block.

    donor: a loaded donor model, a dict of tensors (with donor_config), a
    local folder, or a Hugging Face id (fetched by byte range). donor_layers
    = [first, last], inclusive. With control="random" the block is a seeded
    other range of the same length of the same donor (donor_layers gives the
    length), and control_seed picks it. t_in is a ready stitch (stage 1), or
    a new one of stitch_kind.
    """
    config = donor_config_of(donor, donor_config)
    first, last = int(donor_layers[0]), int(donor_layers[1])
    if control == "random":
        n_layers = _text_config(config).num_hidden_layers
        first, last = random_range(n_layers, first, last, control_seed if control_seed is not None else 0)
    elif control is not None and control != "scrambled":
        raise ValueError(f"unknown control {control!r}; use None, 'random', or 'scrambled'")
    if donor_id is None:
        if isinstance(donor, str):
            donor_id = donor
        else:
            donor_id = getattr(config, "_name_or_path", None) or None
    tensors = donor_block_tensors(donor, config, first, last, cache_dir)
    if control == "scrambled":
        # The same layers with no learned structure: shapes, norms, and spectra kept (wsp.scramble).
        from wsp.scramble import scramble_tensors

        scramble_tensors(tensors, control_seed if control_seed is not None else 0)
    part = build_block_part(config, first, last, tensors)
    d_host = _text_config(host.config).hidden_size
    d_donor = _text_config(config).hidden_size
    if t_in is None:
        t_in = build_stitch_in(stitch_kind, d_host, d_donor, stitch_hidden, stitch_rank)
    meta = {"kind": "block", "read_layer": int(read_layer), "donor_id": donor_id, "donor_layers": [first, last],
            "d_host": int(d_host), "d_donor": int(d_donor), "stitch": stitch_config(t_in),
            "sink": sink is not None, "out_mode": out_mode, "control": control, "control_seed": control_seed}
    if embed_rms is None:
        embed_rms = embed_rms_of(host)
    branch = Branch(d_host, read_layer, t_in, part, d_donor, gate_init, sink, out_mode, meta, g_max=g_max,
                    embed_rms=embed_rms, legacy=legacy, init_seed=init_seed)
    branch.donor_config = config
    branch.to(host.get_input_embeddings().weight.device)
    return branch


def build_seam(host, branches, base_id=None):
    return FeedbackSeam(host, branches, base_id)


# ---------------------------------------------------------------- counts


def count_params(seam):
    """Count parameters. total counts each tensor once. per_token drops the input embedding table.

    The embedding is a lookup, so it does not run per token, unless the
    output head shares it. Every branch layer runs on every token (the
    component runs beside the host layers at each step), so branch layers
    count in per_token in full. Also: by kind (host, lora, component, t_in,
    t_out, gate) and trainable.
    """
    kinds = {"host": 0, "lora": 0, "component": 0, "t_in": 0, "t_out": 0, "gate": 0}
    seen = set()
    total = 0
    trainable = 0
    for name, param in seam.named_parameters():
        if id(param) in seen:
            continue
        seen.add(id(param))
        n = param.numel()
        total += n
        if param.requires_grad:
            trainable += n
        if name.startswith("host."):
            if name.endswith("lora_A") or name.endswith("lora_B"):
                kinds["lora"] += n
            else:
                kinds["host"] += n
        elif ".part." in name:
            kinds["component"] += n
        elif ".t_in." in name:
            kinds["t_in"] += n
        elif ".t_out." in name:
            kinds["t_out"] += n
        else:
            kinds["gate"] += n
    per_token = total
    embed = seam.get_input_embeddings()
    head = seam.get_output_embeddings()
    if embed is not None:
        tied = head is not None and head.weight is embed.weight
        if not tied:
            per_token -= embed.weight.numel()
    out = dict(kinds)
    out["total"] = total
    out["per_token"] = per_token
    out["trainable"] = trainable
    return out


# ---------------------------------------------------------------- stage 1


STAGE1_THRESHOLD = 0.3


def fit_t_in(host, host_tokenizer, read_layer, donor, donor_tokenizer, first, texts, kind="mlp", hidden=None,
             rank=None, ridge_texts=None, steps=1000, lr=3e-4, batch_size=8, seq_len=256, seed=0, eval_texts=None,
             sink=True, log_every=0, threshold=STAGE1_THRESHOLD, extra_rounds=2, extra_steps=None):
    """Stage 1: train T_in on the component's own fidelity.

    T_in maps the output of host layer read_layer to the input of donor
    layer first. It starts from a ridge fit and trains on the donor's LM
    loss (wsp.stitch.train_stitch, objective donor_lm), with the donor's
    sink state at position 0. Host and donor stay frozen. host and donor
    here are loaded models that share a tokenizer.

    Defaults follow the stitch study (results/README.md), which reached a
    0.03 nat penalty with 1000 steps on 2400 texts, lr 1e-3 for linear and
    3e-4 for mlp. The first round 22 run used a linear stitch, 400 steps, and
    256 texts, and ended 1.98 nats above the donor's native loss. When the
    penalty after training is above threshold, up to extra_rounds more
    rounds of extra_steps (default steps // 2) continue from the current
    weights. Returns (t_in, sink, report). sink is a (d_donor,) tensor or
    None. report has penalty_before, penalty_after, penalty (the last one),
    threshold, ok (penalty <= threshold), and rounds.
    """
    from wsp import stitch as stitch_module

    d_host = _text_config(host.config).hidden_size
    d_donor = _text_config(donor.config).hidden_size
    config = {"kind": kind}
    if hidden is not None:
        config["hidden"] = hidden
    if rank is not None:
        config["rank"] = rank
    t_in = stitch_module.build_stitch(config, d_host, d_donor)
    ridge = stitch_module.ridge_init(t_in, host, host_tokenizer, read_layer, donor, donor_tokenizer, first,
                                     ridge_texts if ridge_texts is not None else texts)
    sink_vector = None
    if sink:
        with torch.no_grad():
            sink_vector = stitch_module.resolve_sink(True, donor, donor_tokenizer, first, texts)
    if extra_steps is None:
        extra_steps = max(1, steps // 2)
    use_sink = sink_vector if sink_vector is not None else False
    result = stitch_module.train_stitch(host, host_tokenizer, read_layer, donor, donor_tokenizer, first, t_in, texts,
                                        objective="donor_lm", steps=steps, lr=lr, batch_size=batch_size,
                                        seq_len=seq_len, seed=seed, eval_texts=eval_texts, log_every=log_every,
                                        sink=use_sink)
    penalty_before = result["before"]["penalty"]
    penalty = result["after"]["penalty"]
    seconds = result["seconds"]
    rounds = [{"steps": steps, "penalty": penalty}]
    for index in range(extra_rounds):
        if penalty <= threshold:
            break
        more = stitch_module.train_stitch(host, host_tokenizer, read_layer, donor, donor_tokenizer, first, t_in,
                                          texts, objective="donor_lm", steps=extra_steps, lr=lr,
                                          batch_size=batch_size, seq_len=seq_len, seed=seed + 1 + index,
                                          eval_texts=eval_texts, log_every=log_every, sink=use_sink)
        result = more
        penalty = more["after"]["penalty"]
        seconds += more["seconds"]
        rounds.append({"steps": extra_steps, "penalty": penalty})
    report = {"ridge": ridge, "before": result["before"], "after": result["after"], "seconds": seconds,
              "penalty_before": penalty_before, "penalty_after": penalty, "penalty": penalty,
              "native_loss": result["after"].get("native_loss"), "stitched_loss": result["after"].get("stitched_loss"),
              "threshold": threshold, "ok": bool(penalty <= threshold), "rounds": rounds, "kind": kind,
              "steps": steps, "lr": lr, "train_texts": len(texts)}
    t_in.cpu()
    return t_in, sink_vector, report


# ---------------------------------------------------------------- stage 2


def kd_kl(student_logits, teacher_logits, mask):
    """Mean KL(teacher || student) per masked token, in nats. Logits are cut to the shared vocabulary."""
    vocab = min(student_logits.shape[-1], teacher_logits.shape[-1])
    t = torch.log_softmax(teacher_logits[..., :vocab].float(), dim=-1)
    s = torch.log_softmax(student_logits[..., :vocab].float(), dim=-1)
    kl = (t.exp() * (t - s)).sum(-1)
    return (kl * mask).sum() / mask.sum().clamp(min=1.0)


def draw_passes(chooser, step, one_pass_prob, three_pass_prob, three_after):
    """The pass count of one step. Two draws per step, always, so every arm keeps the same schedule."""
    first = chooser.random()
    second = chooser.random()
    if first < one_pass_prob:
        return 1
    if step >= three_after and second < three_pass_prob:
        return 3
    return 2


def heal_distill(forward, base_forward, groups, tokenizer, teacher, texts, general_flags, steps=1000, batch_size=8,
                 seq_len=384, seed=0, one_pass_prob=0.15, anchor_weight=0.5, autocast_dtype=None, warmup=50,
                 log_every=100, three_pass_prob=0.3, three_pass_after=None, clip_norm=1.0, halt_ratio=3.0,
                 halt_patience=20, halt_floor=0.05, gap_fn=None, gap_every=200, regularizer=None):
    """Stage 2 loop, the same for every trained arm of round 22.

    forward(ids, mask, passes) -> student logits. base_forward(ids, mask) ->
    logits of the frozen base. groups: AdamW parameter groups (dicts with
    params and lr, and optionally warmup: steps of linear lr warmup for that
    group, default the warmup argument). The loss is KL(teacher || student)
    on the rows whose general_flags entry is False (skill text), plus
    anchor_weight * KL(frozen base || student) on the flagged rows (general
    text). There is no LM loss on any text. The batch order depends only on
    seed, and the pass count draws from its own generator, so every arm sees
    the same texts in the same order.

    Passes: 1 with probability one_pass_prob, else 3 with probability
    three_pass_prob once step >= three_pass_after (default warmup), else 2.
    Gradients are clipped per group to clip_norm. regularizer(step) may
    return an extra loss tensor (already weighted).

    Halt: the first step with two or more passes and skill rows sets the
    baseline KD. When the KD of that kind of step is above halt_ratio *
    max(baseline, halt_floor), or not finite, for halt_patience such steps
    in a row, the loop stops and log["halted"] is True. gap_fn() is called
    every gap_every steps and its result goes to log["gap"] with the step.
    Returns {"kd", "anchor", "passes", "gap", "halted", "halt_step",
    "halt_reason", "seconds"}.
    """
    if len(general_flags) != len(texts):
        raise ValueError("general_flags needs one entry per text")
    if three_pass_after is None:
        three_pass_after = warmup
    params = []
    for group in groups:
        params += list(group["params"])
    base_lrs = []
    warmups = []
    for group in groups:
        base_lrs.append(group["lr"])
        warmups.append(group.get("warmup", warmup))
    optimizer = torch.optim.AdamW(groups, weight_decay=0.0)
    order = list(range(len(texts)))
    random.Random(seed).shuffle(order)
    chooser = random.Random(seed * 7919 + 13)
    side = tokenizer.padding_side
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    device = params[0].device
    log = {"kd": [], "anchor": [], "passes": [], "gap": [], "reg": [], "halted": False, "halt_step": None,
           "halt_reason": None}
    start = time.time()
    cursor = 0
    baseline = None
    streak = 0
    for step in range(steps):
        rows = []
        while len(rows) < batch_size:
            if cursor >= len(order):
                cursor = 0
            rows.append(order[cursor])
            cursor += 1
        batch = []
        flags = []
        for row in rows:
            batch.append(texts[row])
            flags.append(bool(general_flags[row]))
        encoded = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=seq_len,
                            add_special_tokens=False)
        ids = encoded["input_ids"].to(device)
        mask = encoded["attention_mask"].to(device)
        flag = torch.tensor(flags, dtype=torch.bool, device=device)
        passes = draw_passes(chooser, step, one_pass_prob, three_pass_prob, three_pass_after)
        weight = mask.float()
        weight[:, 0] = 0.0

        for group, lr, warm in zip(optimizer.param_groups, base_lrs, warmups):
            group["lr"] = lr * min(1.0, (step + 1) / max(1, warm))
        optimizer.zero_grad()
        skill_rows = ~flag
        # The teacher and the frozen base run first. base_forward may switch off a gate in place, which must
        # not happen between the student forward and its backward.
        t_logits = None
        if bool(skill_rows.any()):
            with torch.no_grad():
                teacher_device = next(teacher.parameters()).device
                if autocast_dtype is not None:
                    with torch.autocast(teacher_device.type, dtype=autocast_dtype):
                        t_logits = teacher(input_ids=ids[skill_rows].to(teacher_device),
                                           attention_mask=mask[skill_rows].to(teacher_device)).logits
                else:
                    t_logits = teacher(input_ids=ids[skill_rows].to(teacher_device),
                                       attention_mask=mask[skill_rows].to(teacher_device)).logits
        b_logits = None
        if bool(flag.any()) and anchor_weight > 0:
            with torch.no_grad():
                if autocast_dtype is not None:
                    with torch.autocast(device.type, dtype=autocast_dtype):
                        b_logits = base_forward(ids[flag], mask[flag])
                else:
                    b_logits = base_forward(ids[flag], mask[flag])
        if autocast_dtype is not None:
            with torch.autocast(device.type, dtype=autocast_dtype):
                logits = forward(ids, mask, passes)
        else:
            logits = forward(ids, mask, passes)
        logits = logits.float()
        loss = logits.sum() * 0.0
        kd_value = 0.0
        anchor_value = 0.0
        reg_value = 0.0
        if t_logits is not None:
            kd = kd_kl(logits[skill_rows], t_logits.to(device), weight[skill_rows])
            loss = loss + kd
            kd_value = float(kd.detach())
        if b_logits is not None:
            anchor = kd_kl(logits[flag], b_logits, weight[flag])
            loss = loss + anchor_weight * anchor
            anchor_value = float(anchor.detach())
        if regularizer is not None:
            extra = regularizer(step)
            if extra is not None:
                loss = loss + extra
                reg_value = float(extra.detach())
        del t_logits, b_logits
        loss.backward()
        for group in groups:
            torch.nn.utils.clip_grad_norm_(list(group["params"]), clip_norm)
        optimizer.step()
        log["kd"].append(kd_value)
        log["anchor"].append(anchor_value)
        log["passes"].append(passes)
        log["reg"].append(reg_value)
        if log_every and (step + 1) % log_every == 0:
            print(f"heal step {step + 1}/{steps} kd {kd_value:.4f} anchor {anchor_value:.4f} passes {passes}")
        if gap_fn is not None and gap_every and (step + 1) % gap_every == 0:
            if autocast_dtype is not None:
                with torch.autocast(device.type, dtype=autocast_dtype):
                    found = gap_fn()
            else:
                found = gap_fn()
            log["gap"].append({"step": step + 1, **found})
            if log_every:
                print(f"heal step {step + 1}/{steps} passes_gap kl {found['kl']:.4f} agree {found['agree']:.3f}")
        # The runaway check: two or more passes, skill rows present.
        if passes >= 2 and not bool(skill_rows.sum() == 0):
            if not math.isfinite(kd_value) or not math.isfinite(anchor_value):
                log["halted"] = True
                log["halt_step"] = step + 1
                log["halt_reason"] = "non-finite loss"
                break
            if baseline is None:
                baseline = kd_value
            limit = halt_ratio * max(baseline, halt_floor)
            if kd_value > limit:
                streak += 1
            else:
                streak = 0
            if streak >= halt_patience:
                log["halted"] = True
                log["halt_step"] = step + 1
                log["halt_reason"] = (f"two-pass KD above {halt_ratio}x its first value ({baseline:.3f}) "
                                      f"for {halt_patience} steps in a row")
                break
    tokenizer.padding_side = side
    log["baseline_kd"] = baseline
    log["seconds"] = round(time.time() - start, 1)
    return log


def stage2_groups(seam, lr_t_out=3e-4, lr_gate=1e-3, lr_lora=2e-4, lr_t_in=1e-5, train_t_in=False, warmup_seam=100,
                  warmup_lora=50):
    """Freeze all but T_out, the gates, the host LoRA, and optionally T_in. Return AdamW groups.

    Each group has its own lr and warmup. T_out and the gate warm up over
    warmup_seam steps, the LoRA and T_in over warmup_lora. train_t_in is the
    stage-2 fine-tune of T_in; heal_seam turns it on by default.
    """
    names = ["t_out", "gate", "lora"]
    if train_t_in:
        names.append("t_in")
    found = seam.set_trainable(tuple(names))
    groups = []
    for name, lr, warm in (("t_out", lr_t_out, warmup_seam), ("gate", lr_gate, warmup_seam),
                           ("t_in", lr_t_in, warmup_lora), ("lora", lr_lora, warmup_lora)):
        if name in names and len(found[name]) > 0:
            groups.append({"params": found[name], "lr": lr, "name": name, "warmup": warm})
    return groups


def t_in_regularizer(seam, tokenizer, donor, texts, weight=1.0, batch_size=2, seq_len=256, seed=0):
    """regularizer(step) for heal_distill: the stage-1 fidelity loss (the donor's LM loss through T_in), times weight.

    Keeps T_in a good feed for the donor block while it learns to read
    states that carry feedback. It needs the donor model and tokenizer text
    (same tokenizer as the host). Returns None when no branch has a T_in.
    """
    from wsp import stitch as stitch_module

    branches = []
    for branch in seam.branches:
        if branch.t_in is not None:
            branches.append(branch)
    if len(branches) == 0:
        return None
    sampler = stitch_module.text_sampler(stitch_module.lm_texts(tokenizer, texts, seq_len), batch_size, seed)

    def regularizer(step):
        batch = sampler()
        total = None
        for branch in branches:
            sink = branch.sink
            loss = stitch_module.stitched_lm_loss(seam.host, tokenizer, [branch.read_layer], donor,
                                                  branch.meta["donor_layers"][0], None, branch.t_in, batch, seq_len,
                                                  sink)
            total = loss if total is None else total + loss
        return weight * total

    return regularizer


def heal_seam(seam, tokenizer, teacher, texts, general_flags, lrs=None, gap_texts=None, gap_every=200,
              train_t_in=True, t_in_reg_weight=1.0, **options):
    """Stage 2 for a seam: heal_distill with the seam forward and the frozen-base anchor.

    lrs: keyword arguments of stage2_groups (lr_t_out, lr_gate, lr_lora,
    lr_t_in). train_t_in (default on) also fine-tunes T_in at lr_t_in, with
    the stage-1 fidelity loss times t_in_reg_weight as a regularizer (the
    teacher is the donor, so it must be a model of the donor's family; the
    regularizer is skipped when t_in_reg_weight is 0). gap_texts: held-out
    texts for passes_gap every gap_every steps, logged under "gap".
    options go to heal_distill (steps, batch_size, seq_len, seed,
    one_pass_prob, three_pass_prob, halt_ratio, ...).
    """
    lrs = lrs or {}
    groups = stage2_groups(seam, train_t_in=train_t_in, **lrs)
    seam.train()

    def forward(ids, mask, passes):
        return seam(ids, mask, passes=passes).logits

    def base_forward(ids, mask):
        return base_logits(seam, ids, mask)

    gap_fn = None
    if gap_texts is not None and len(gap_texts) > 0:
        def gap_fn():
            seam.eval()
            try:
                return passes_gap(seam, tokenizer, gap_texts, passes=2, max_length=options.get("seq_len", 128))
            finally:
                seam.train()

    regularizer = None
    if train_t_in and t_in_reg_weight > 0:
        regularizer = t_in_regularizer(seam, tokenizer, teacher, texts, weight=t_in_reg_weight,
                                       seq_len=min(256, options.get("seq_len", 256)), seed=options.get("seed", 0))
    log = heal_distill(forward, base_forward, groups, tokenizer, teacher, texts, general_flags,
                       gap_fn=gap_fn, gap_every=gap_every, regularizer=regularizer, **options)
    seam.eval()
    return log


@torch.no_grad()
def warm_start_t_out(seam, tokenizer, teacher, texts, batch_size=4, seq_len=256, lam=1e-3, min_rows=64,
                     holdout=0.2, init_scale=None):
    """Fit T_out of every branch by ridge, so a fresh seam already points from the base toward the donor.

    Needs a tokenizer the host and the teacher share (same vocabulary
    ids). For each real position t whose feedback reaches position t + 1:

      x_t = the branch's normalized output at t (what T_out reads)
      y_t = E[top1 of the teacher] - E[top1 of the base]   (host input embeddings)

    on rows where the two top-1 tokens differ. y is the direction in host
    embedding space from what the base would say to what the donor says. Two
    lags are tried: lag 0 takes both top-1 tokens from the logits at t, and
    lag 1 from the logits at t + 1 (the prediction the position that
    receives the feedback must make). The lag with the larger held-out mean
    cosine between T_out(x) and y wins. The fit scales to the same size as a
    random start (std 1/sqrt(width)), which costs nothing in the forward
    pass because the written vector is normalized, and keeps one Adam step
    small relative to the weights. The gate stays where it is (0 for a fresh
    seam).

    A branch keeps its random start when it has too few rows (min_rows) or
    its held-out cosine is not above 0. Returns a list of reports, one per
    branch: mode "ridge" or "random", rows, lag, cosines, and the cosine of
    the random start on the same held-out rows.
    """
    if len(seam.branches) == 0:
        return []
    side = tokenizer.padding_side
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    device = next(seam.parameters()).device
    teacher_device = next(teacher.parameters()).device
    embed = seam.get_input_embeddings().weight.detach().float()
    was_training = seam.training
    seam.eval()
    count = (len(texts) + batch_size - 1) // batch_size
    cut = max(1, int(round(count * (1.0 - holdout)))) if count > 1 else 1
    stats = []
    for branch in seam.branches:
        width = branch.t_out.weight.shape[1]
        d_host = branch.t_out.weight.shape[0]
        stats.append({"xtx": [torch.zeros(width, width, dtype=torch.float64, device=device) for _ in range(2)],
                      "xty": [torch.zeros(width, d_host, dtype=torch.float64, device=device) for _ in range(2)],
                      "rows": [0, 0], "val": [[], []]})
    for index in range(count):
        batch = texts[index * batch_size:(index + 1) * batch_size]
        enc = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=seq_len,
                        add_special_tokens=False)
        ids = enc["input_ids"].to(device)
        mask = enc["attention_mask"].to(device)
        if ids.shape[1] < 3:
            continue
        mask_p, pos = seam.prepare(ids, mask)
        out = seam.host_pass(ids, mask_p, pos, None)
        teacher_logits = teacher(input_ids=ids.to(teacher_device), attention_mask=mask.to(teacher_device)).logits
        vocab = min(out.logits.shape[-1], teacher_logits.shape[-1], embed.shape[0])
        base_top = out.logits[..., :vocab].argmax(-1)
        donor_top = teacher_logits[..., :vocab].argmax(-1).to(device)
        real = mask != 0
        reads = seam.reads_of(out.hidden_states)
        is_val = index >= cut
        for branch, read, stat in zip(seam.branches, reads, stats):
            x = branch.pre(read, mask_p).float()
            # valid[t]: the feedback of t reaches a real position t + 1, and t + 1 has a prediction (lag 1).
            valid = real[:, :-1] & real[:, 1:]
            for lag in (0, 1):
                if lag == 0:
                    b1, d1 = base_top[:, :-1], donor_top[:, :-1]
                else:
                    b1, d1 = base_top[:, 1:], donor_top[:, 1:]
                use = valid & (b1 != d1)
                if not bool(use.any()):
                    continue
                rows_x = x[:, :-1][use].double()
                rows_y = (embed[d1[use]] - embed[b1[use]]).double()
                if is_val:
                    stat["val"][lag].append((rows_x, rows_y))
                else:
                    stat["xty"][lag] += rows_x.T @ rows_y
                    stat["rows"][lag] += rows_x.shape[0]
                    stat["xtx"][lag] += rows_x.T @ rows_x

    reports = []
    for branch, stat in zip(seam.branches, stats):
        width = branch.t_out.weight.shape[1]
        report = {"mode": "random", "rows": max(stat["rows"]), "lag": None, "cosine": {}, "random_cosine": {}}
        scale = init_scale if init_scale is not None else 1.0 / math.sqrt(width)
        best = None
        for lag in (0, 1):
            if stat["rows"][lag] < min_rows:
                continue
            xtx = stat["xtx"][lag]
            ridge = lam * float(torch.trace(xtx)) / width
            eye = torch.eye(width, dtype=torch.float64, device=xtx.device)
            weight = torch.linalg.solve(xtx + ridge * eye, stat["xty"][lag]).T.float()
            cos = held_out_cosine(weight, stat["val"][lag])
            rnd = held_out_cosine(branch.t_out.weight.detach().float().to(weight.device), stat["val"][lag])
            report["cosine"][f"lag{lag}"] = cos
            report["random_cosine"][f"lag{lag}"] = rnd
            if cos is not None and cos > 0 and (best is None or cos > best[0]):
                best = (cos, lag, weight)
        if best is not None:
            weight = best[2]
            rms = float(weight.pow(2).mean().sqrt())
            if rms > 0:
                with torch.no_grad():
                    branch.t_out.weight.copy_((weight * (scale / rms)).to(branch.t_out.weight.device))
                report["mode"] = "ridge"
                report["lag"] = best[1]
        reports.append(report)
    tokenizer.padding_side = side
    seam.train(was_training)
    return reports


def held_out_cosine(weight, rows):
    """Mean cosine between weight @ x and y over held-out (x, y) row blocks. None when there are none."""
    total = 0.0
    count = 0
    for x, y in rows:
        pred = x.float() @ weight.T.to(x.device)
        sim = torch.nn.functional.cosine_similarity(pred, y.float(), dim=-1)
        total += float(sim.sum())
        count += sim.numel()
    if count == 0:
        return None
    return total / count


@torch.no_grad()
def feedback_stats(seam, tokenizer, texts, max_length=128, batch_size=4):
    """Per branch: the gate, embed_rms, and the mean norm of the written vector, as a ratio to embed_rms.

    The reads come from a no-feedback pass. For a fresh seam the ratio is 0.
    In general it equals the gate, whatever the raw scale of T_out. The
    token_ratio is the same norm divided by the norm of the actual input
    embedding of the token it is added to.
    """
    side = tokenizer.padding_side
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    device = next(seam.parameters()).device
    totals = [0.0] * len(seam.branches)
    token_totals = [0.0] * len(seam.branches)
    count = 0.0
    was_training = seam.training
    seam.eval()
    for first in range(0, len(texts), batch_size):
        enc = tokenizer(texts[first:first + batch_size], return_tensors="pt", padding=True, truncation=True,
                        max_length=max_length, add_special_tokens=False)
        ids = enc["input_ids"].to(device)
        mask = enc["attention_mask"].to(device)
        mask, pos = seam.prepare(ids, mask)
        out = seam.host_pass(ids, mask, pos, None)
        embed = seam.get_input_embeddings()(ids).float()
        valid = (mask != 0)
        valid = valid[:, 1:] & valid[:, :-1]
        reads = seam.reads_of(out.hidden_states)
        for index, (branch, read) in enumerate(zip(seam.branches, reads)):
            written = branch.output(read, mask).float()
            norms = written[:, :-1].norm(dim=-1)
            totals[index] += float((norms * valid).sum())
            token_totals[index] += float(((norms / embed[:, 1:].norm(dim=-1).clamp(min=1e-6)) * valid).sum())
        count += float(valid.sum())
    tokenizer.padding_side = side
    seam.train(was_training)
    out = []
    for index, branch in enumerate(seam.branches):
        rms = float(branch.embed_rms)
        mean_norm = totals[index] / count if count > 0 else 0.0
        out.append({"gate": float(branch.gate.detach()), "embed_rms": rms, "written_norm": mean_norm,
                    "ratio": mean_norm / rms if rms > 0 else None,
                    "token_ratio": token_totals[index] / count if count > 0 else None})
    return out


@torch.no_grad()
def passes_gap(seam, tokenizer, texts, passes=2, max_length=128, batch_size=4):
    """How far teacher-forced passes are from the exact recursion: mean KL(exact || passes) and top-1 agreement."""
    side = tokenizer.padding_side
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    device = next(seam.parameters()).device
    total = 0.0
    agree = 0.0
    count = 0.0
    for first in range(0, len(texts), batch_size):
        enc = tokenizer(texts[first:first + batch_size], return_tensors="pt", padding=True, truncation=True,
                        max_length=max_length, add_special_tokens=False)
        ids = enc["input_ids"].to(device)
        mask = enc["attention_mask"].to(device)
        exact = seam.forward_cached(ids, mask).float()
        approx = seam(ids, mask, passes=passes).logits.float()
        weight = mask.float()
        weight[:, 0] = 0.0
        t = torch.log_softmax(exact, dim=-1)
        s = torch.log_softmax(approx, dim=-1)
        kl = (t.exp() * (t - s)).sum(-1)
        total += float((kl * weight).sum())
        agree += float(((t.argmax(-1) == s.argmax(-1)).float() * weight).sum())
        count += float(weight.sum())
    tokenizer.padding_side = side
    if count == 0:
        return {"kl": 0.0, "agree": 0.0, "tokens": 0}
    return {"kl": total / count, "agree": agree / count, "tokens": int(count)}


# ---------------------------------------------------------------- save and load


def branch_state(seam):
    """The tensors to save: every branch tensor except the frozen donor layers."""
    state = {}
    for key, value in seam.branches.state_dict().items():
        if ".part." in key:
            continue
        state[key] = value.detach().cpu().contiguous()
    return state


def save_seam(seam, out_dir, tokenizer=None, base_id=None, extra=None):
    """Save a seam to one directory. Donor layers are not saved: load_seam fetches them by byte range.

    Files: feedback.json (manifest), feedback_weights.safetensors (T_in, T_out,
    gates, sinks), lora_weights.safetensors (host LoRA, when there is one),
    donors/<i>/config.json, and the tokenizer.
    """
    from safetensors.torch import save_file

    os.makedirs(out_dir, exist_ok=True)
    metas = []
    for index, branch in enumerate(seam.branches):
        meta = dict(branch.meta)
        if meta["kind"] == "block":
            if meta["donor_id"] is None:
                raise ValueError(f"branch {index} has no donor_id, so load_seam could not fetch its donor")
            folder = os.path.join(out_dir, "donors", str(index))
            os.makedirs(folder, exist_ok=True)
            branch.donor_config.save_pretrained(folder)
        meta["gate"] = float(branch.gate.detach())
        meta["gate_raw"] = float(branch.gate_raw.detach())
        metas.append(meta)
    save_file(branch_state(seam), os.path.join(out_dir, "feedback_weights.safetensors"))
    lora = getattr(seam.host, "_wsp_base_lora", None)
    if lora is not None:
        save_file(lora_state_dict(seam.host), os.path.join(out_dir, "lora_weights.safetensors"))
    manifest = {"format": FORMAT, "base": base_id if base_id is not None else seam.base_id, "branches": metas,
                "base_lora": lora, "params": count_params(seam)}
    if extra is not None:
        manifest["extra"] = extra
    with open(os.path.join(out_dir, "feedback.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    if tokenizer is not None:
        tokenizer.save_pretrained(out_dir)
    return manifest


def load_seam(path, device="cpu", dtype=torch.bfloat16, cache_dir=None, host=None, donors=None):
    """Load a directory written by save_seam. Returns (seam, tokenizer or None).

    host: a loaded base to use in place of the manifest's base id. donors: an
    optional dict of donor_id -> loaded model or tensors, used in place of a fetch.
    """
    import transformers
    from safetensors.torch import load_file

    with open(os.path.join(path, "feedback.json")) as f:
        manifest = json.load(f)
    if manifest.get("format") != FORMAT:
        raise ValueError(f"{path} is not a {FORMAT} directory")
    if host is None:
        host = transformers.AutoModelForCausalLM.from_pretrained(manifest["base"], dtype=dtype)
    lora = manifest.get("base_lora")
    if lora is not None:
        apply_lora(host, lora["rank"], lora["alpha"], tuple(lora["targets"]), lora["layers"])
        load_lora_state(host, load_file(os.path.join(path, "lora_weights.safetensors")))
        host._wsp_base_lora = lora
    branches = []
    for index, meta in enumerate(manifest["branches"]):
        if meta["kind"] == "plain":
            branches.append(make_plain_branch(host, g_max=meta.get("g_max", 0.5), legacy=meta.get("legacy", False),
                                              embed_rms=meta.get("embed_rms")))
            continue
        config = transformers.AutoConfig.from_pretrained(os.path.join(path, "donors", str(index)))
        source = meta["donor_id"]
        if donors is not None and source in donors:
            source = donors[source]
        stitch = meta["stitch"]
        t_in = build_stitch_in(stitch["kind"], meta["d_host"], meta["d_donor"], stitch.get("hidden"),
                               stitch.get("rank"))
        sink = torch.zeros(meta["d_donor"]) if meta.get("sink") else None
        branch = make_block_branch(host, source, meta["donor_layers"], meta["read_layer"], t_in=t_in,
                                   donor_config=config, donor_id=meta["donor_id"], sink=sink,
                                   out_mode=meta.get("out_mode", "delta"), cache_dir=cache_dir,
                                   g_max=meta.get("g_max", 0.5), legacy=meta.get("legacy", False),
                                   embed_rms=meta.get("embed_rms"),
                                   control="scrambled" if meta.get("control") == "scrambled" else None,
                                   control_seed=meta.get("control_seed"))
        branch.meta["control"] = meta.get("control")
        branch.meta["control_seed"] = meta.get("control_seed")
        branches.append(branch)
    seam = FeedbackSeam(host, branches, manifest["base"])
    result = seam.branches.load_state_dict(load_file(os.path.join(path, "feedback_weights.safetensors")),
                                           strict=False)
    if len(result.unexpected_keys) > 0:
        raise KeyError(f"saved branch tensors the seam does not have: {result.unexpected_keys[:5]}")
    for key in result.missing_keys:
        if ".part." not in key:
            raise KeyError(f"branch tensor {key} is missing from the saved weights")
    seam = seam.to(device).eval()
    tokenizer = None
    if os.path.exists(os.path.join(path, "tokenizer_config.json")):
        tokenizer = transformers.AutoTokenizer.from_pretrained(path)
    return seam, tokenizer



# ---------------------------------------------------------------- the round 22 verdict


CONTROLS = ("kd_lora", "feedback_plain", "feedback_block_random")


def mean_transfer(row):
    """Mean probe transfer over the skills of one row (skills with no gap are left out), or None."""
    values = []
    for entry in row["probe"]["skills"].values():
        if entry.get("transfer") is not None:
            values.append(entry["transfer"])
    if len(values) == 0:
        return None
    return sum(values) / len(values)


def arm_numbers(rows, pair, arm):
    """(transfers, drifts) of the trained cells of one arm of one pair, one entry per seed."""
    transfers = []
    drifts = []
    for row in rows:
        if row.get("pair") != pair or row.get("arm") != arm or "probe" not in row:
            continue
        value = mean_transfer(row)
        if value is None:
            continue
        transfers.append(value)
        drifts.append(row["probe"]["drift"]["base_kl"])
    return transfers, drifts


def spread_of(values):
    if len(values) < 2:
        return 0.0
    return max(values) - min(values)


def go_verdict(rows, drift_limit=0.05):
    """Apply the round 22 go rule to the rows. Returns {pair: result}.

    Go for a pair: feedback_block beats kd_lora, feedback_plain, and
    feedback_block_random on mean probe transfer, each by more than the
    seed spread (the larger of the two arms' ranges over seeds), and its
    drift (KL from the frozen base on general text) is under drift_limit
    nats for every seed. With one seed the spread is 0 and the result says so.
    Also gives the content gain (real minus random, per seed) and the two
    contrasts that say what a non-go means.
    """
    pairs = []
    for row in rows:
        if row.get("pair") not in pairs:
            pairs.append(row.get("pair"))
    out = {}
    for pair in pairs:
        block, block_drift = arm_numbers(rows, pair, "feedback_block")
        result = {"seeds": len(block), "feedback_block": block, "drift": block_drift, "beats": {}}
        if len(block) == 0:
            result["go"] = None
            result["note"] = "no feedback_block rows"
            out[pair] = result
            continue
        all_beat = True
        for control in CONTROLS:
            other, _ = arm_numbers(rows, pair, control)
            if len(other) == 0:
                result["beats"][control] = None
                all_beat = False
                continue
            gap = sum(block) / len(block) - sum(other) / len(other)
            spread = max(spread_of(block), spread_of(other))
            result["beats"][control] = {"gap": gap, "spread": spread, "beats": gap > spread}
            if not gap > spread:
                all_beat = False
        drift_ok = True
        for value in block_drift:
            if not value < drift_limit:
                drift_ok = False
        result["drift_ok"] = drift_ok
        result["go"] = bool(all_beat and drift_ok)
        if len(block) < 2:
            result["note"] = "one seed: the spread is 0, so the go call is weak"
        plain, _ = arm_numbers(rows, pair, "feedback_plain")
        if len(plain) > 0:
            gap = sum(block) / len(block) - sum(plain) / len(plain)
            result["plain_equals_block"] = abs(gap) <= max(spread_of(block), spread_of(plain))
        for name in ("midstack_block", "midstack_block_random"):
            other, _ = arm_numbers(rows, pair, name)
            if len(other) > 0 and name == "midstack_block":
                gap = sum(block) / len(block) - sum(other) / len(other)
                result["midstack_equals_feedback"] = abs(gap) <= max(spread_of(block), spread_of(other))
        result["content_gain"] = content_gain(rows, pair, "feedback_block", "feedback_block_random")
        result["content_gain_midstack"] = content_gain(rows, pair, "midstack_block", "midstack_block_random")
        out[pair] = result
    return out


def content_gain(rows, pair, real_arm, random_arm):
    """Per skill, mean over seeds of (real transfer - random transfer). None when an arm has no rows."""
    real = {}
    control = {}
    for row in rows:
        if row.get("pair") != pair or "probe" not in row:
            continue
        target = None
        if row["arm"] == real_arm:
            target = real
        elif row["arm"] == random_arm:
            target = control
        if target is None:
            continue
        for skill, entry in row["probe"]["skills"].items():
            if entry.get("transfer") is not None:
                target.setdefault(skill, []).append(entry["transfer"])
    gains = {}
    for skill in real:
        if skill in control:
            gains[skill] = sum(real[skill]) / len(real[skill]) - sum(control[skill]) / len(control[skill])
    if len(gains) == 0:
        return None
    return gains
