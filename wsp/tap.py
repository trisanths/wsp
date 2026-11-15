"""Taps: a frozen host reads a frozen donor's own states through a small branch.

Grafts run donor layers on a mapped host residual stream, and that breaks
the donor. A tap keeps both models native instead:

- The donor reads the prompt with its own tokenizer and runs only up to one
  layer. Its hidden states there are the donor memory.
- At one host layer, a TapBranch lets each host token attend over the donor
  memory and adds the result to the host hidden state through a tanh gate
  that starts at zero. A fresh branch leaves the host exactly unchanged.
- Only the branches train (and, if the caller wants, a host LoRA).

Visibility: a host token may read a donor memory token only when the donor
token's character end is at or before the host token's own character end in
the prompt. Host tokens past the prompt (the completion and every generated
token) see all valid memory. Host pads and special tokens see nothing.

Usage:

    branch = TapBranch(d_host, d_donor, donor_id="qmath", donor_layer=27)
    attach_taps(host, [(12, branch, 0)])
    encoded = prepare(host, host_tokenizer, {"qmath": (donor, donor_tokenizer)}, prompts)
    out = host.generate(**encoded, max_new_tokens=256)
    clear_tap_context(host)
"""

import contextlib
import json
import math
import os
import time

import torch
from torch import nn

from wsp.graft import decoder_of, forward_args, host_cache_length, replace_hidden
from wsp.parts import find_decoder_layers


class RMSNorm(nn.Module):
    """x / rms(x) * weight, in float32. The weight starts at 1."""

    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.eps = eps

    def forward(self, x):
        x = x.to(torch.float32)
        scale = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x * scale * self.weight


class TapBranch(nn.Module):
    """h + tanh(gate) * o(attention of q(h) over k(memory), v(memory)).

    The gate starts at 0, so a fresh branch returns h exactly. Every
    parameter is float32, and the branch turns autocast off, so it runs in
    float32 even inside a bf16 autocast region. h comes in and goes out in
    the host dtype.

    donor_id and donor_layer name the memory this branch reads. prepare
    uses them to run the right donor. attach_taps sets host_layer and
    tap_index.
    """

    def __init__(self, d_host, d_donor, attn_dim=256, heads=8, donor_id=None, donor_layer=None):
        super().__init__()
        if attn_dim % heads != 0:
            raise ValueError(f"attn_dim {attn_dim} must be a multiple of heads {heads}")
        self.d_host = d_host
        self.d_donor = d_donor
        self.attn_dim = attn_dim
        self.heads = heads
        self.head_dim = attn_dim // heads
        self.donor_id = donor_id
        self.donor_layer = donor_layer
        self.host_layer = None
        self.tap_index = None
        # The host residual stream is not normalized, so q reads a normed h.
        self.host_norm = RMSNorm(d_host)
        self.memory_norm = RMSNorm(d_donor)
        self.q = nn.Linear(d_host, attn_dim, bias=False)
        self.k = nn.Linear(d_donor, attn_dim, bias=False)
        self.v = nn.Linear(d_donor, attn_dim, bias=False)
        self.o = nn.Linear(attn_dim, d_host, bias=False)
        self.gate = nn.Parameter(torch.zeros((), dtype=torch.float32))
        self.float()

    def attend(self, h, memory, mask):
        """Return the branch output before the gate, shape (batch, seq, d_host), float32.

        mask is a boolean (batch, seq, m) tensor: True where the host token
        may read the memory token. A row with no True gives exactly zero.
        """
        x = h.to(torch.float32)
        mem = memory.to(device=x.device, dtype=torch.float32)
        batch = x.shape[0]
        seq = x.shape[1]
        m = mem.shape[1]
        q = self.q(self.host_norm(x)).view(batch, seq, self.heads, self.head_dim).transpose(1, 2)
        normed = self.memory_norm(mem)
        k = self.k(normed).view(batch, m, self.heads, self.head_dim).transpose(1, 2)
        v = self.v(normed).view(batch, m, self.heads, self.head_dim).transpose(1, 2)
        scores = (q @ k.transpose(-1, -2)) / math.sqrt(self.head_dim)
        visible = mask.to(device=x.device, dtype=torch.bool)[:, None, :, :]
        # A large negative number, not -inf: a row with nothing visible then
        # gives a finite softmax, and the multiply below sets it to zero.
        scores = scores.masked_fill(~visible, torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=-1) * visible.to(scores.dtype)
        out = (weights @ v).transpose(1, 2).reshape(batch, seq, self.attn_dim)
        return self.o(out)

    def forward(self, h, memory, mask):
        if h.device.type == "cuda":
            no_autocast = torch.autocast(device_type="cuda", enabled=False)
        else:
            no_autocast = contextlib.nullcontext()
        with no_autocast:
            delta = torch.tanh(self.gate) * self.attend(h, memory, mask)
        return h + delta.to(h.dtype)


class _StopDonor(Exception):
    """Raised by a donor hook once the last needed layer has run."""


def run_donor_layers(donor_model, donor_tokenizer, prompts, layers):
    """Run the donor on prompts up to the deepest layer in layers, and stop there.

    The donor reads the prompts with its own tokenizer, default special
    tokens, and right padding. A hook on each wanted layer stores that
    layer's output. The hook on the deepest one raises an exception that
    we catch here, so later donor layers never run.

    Returns (memories, ends, valid):
    - memories: dict of layer -> (batch, m, d_donor) float32 tensor.
    - ends: (batch, m) long tensor of character end offsets, -1 for
      special tokens and pads.
    - valid: (batch, m) bool tensor, True for real text tokens.
    """
    if donor_tokenizer.pad_token is None:
        donor_tokenizer.pad_token = donor_tokenizer.eos_token
    old_side = donor_tokenizer.padding_side
    donor_tokenizer.padding_side = "right"
    try:
        encoded = donor_tokenizer(list(prompts), padding=True, return_offsets_mapping=True,
                                  return_special_tokens_mask=True, return_tensors="pt")
    finally:
        donor_tokenizer.padding_side = old_side

    valid = (encoded["attention_mask"] == 1) & (encoded["special_tokens_mask"] == 0)
    ends = encoded["offset_mapping"][:, :, 1].clone().long()
    ends[~valid] = -1

    wanted = sorted(set(layers))
    last = wanted[-1]
    _, donor_layers = find_decoder_layers(donor_model, last)
    store = {}
    handles = []
    for layer in wanted:

        def hook(module, inputs, output, layer=layer):
            if isinstance(output, tuple):
                output = output[0]
            store[layer] = output.detach().to(torch.float32)
            if layer == last:
                raise _StopDonor()

        handles.append(donor_layers[layer].register_forward_hook(hook))

    device = next(donor_model.parameters()).device
    try:
        with torch.no_grad():
            donor_model(input_ids=encoded["input_ids"].to(device),
                        attention_mask=encoded["attention_mask"].to(device), use_cache=False)
    except _StopDonor:
        pass
    finally:
        for handle in handles:
            handle.remove()
    if last not in store:
        raise RuntimeError(f"donor layer {last} did not run")
    return store, ends, valid


def run_donor(donor_model, donor_tokenizer, prompts, donor_layer):
    """Return (memory, memory_ends, valid) for one donor layer. See run_donor_layers."""
    memories, ends, valid = run_donor_layers(donor_model, donor_tokenizer, prompts, [donor_layer])
    return memories[donor_layer], ends, valid


def host_token_ends(host_tokenizer, prompts):
    """Tokenize prompts for the host and return (input_ids, attention_mask, ends).

    The host tokenizer's own padding side is used (left for generation).
    ends holds each token's character end offset, and -1 for special
    tokens and pads, so those tokens see no memory.
    """
    if host_tokenizer.pad_token is None:
        host_tokenizer.pad_token = host_tokenizer.eos_token
    encoded = host_tokenizer(list(prompts), padding=True, return_offsets_mapping=True,
                             return_special_tokens_mask=True, return_tensors="pt")
    real = (encoded["attention_mask"] == 1) & (encoded["special_tokens_mask"] == 0)
    ends = encoded["offset_mapping"][:, :, 1].clone().long()
    ends[~real] = -1
    return encoded["input_ids"], encoded["attention_mask"], ends


def build_visibility(host_ends, memory_ends, memory_valid):
    """Return a (batch, host_len, m) bool tensor: True where the host token may read the memory token.

    A host token with end e reads every valid memory token whose end is at
    or before e. A host end of -1 (pad or special token) reads nothing.
    """
    host_ends = host_ends.long()
    memory_ends = memory_ends.to(host_ends.device).long()
    memory_valid = memory_valid.to(host_ends.device).bool()
    allowed = memory_ends[:, None, :] <= host_ends[:, :, None]
    allowed = allowed & memory_valid[:, None, :]
    allowed = allowed & (host_ends[:, :, None] >= 0)
    return allowed


def host_device(host_model):
    return next(host_model.parameters()).device


def attach_taps(host_model, placements):
    """Hook tap branches onto host decoder layers.

    placements: list of (host_layer, TapBranch, tap_index). tap_index picks
    which memory in the tap context the branch reads, so several branches
    can share one memory. The indices must be 0, 1, ..., n-1. The branches
    are stored in host_model.wsp_taps, so they move with .to() and show up
    in .parameters().

    A pre-hook on the host decoder reads each forward call: the host cache
    length before the call and the 2D padding mask. With them, the layer
    hooks know the absolute position of each token, so decode steps count
    as past the prompt, and cached generation equals uncached generation.
    Returns the hook handles.
    """
    if getattr(host_model, "wsp_taps", None) is not None:
        raise ValueError("this host already has taps; call detach_taps first")
    if len(placements) == 0:
        raise ValueError("attach_taps needs at least one placement")

    indices = set()
    for host_layer, branch, tap_index in placements:
        indices.add(tap_index)
    for index in range(len(indices)):
        if index not in indices:
            raise ValueError(f"tap indices must be 0..{len(indices) - 1}, got {sorted(indices)}")
    for tap_index in indices:
        sources = set()
        for host_layer, branch, other_index in placements:
            if other_index == tap_index:
                sources.add((branch.donor_id, branch.donor_layer))
        if len(sources) > 1:
            raise ValueError(f"branches with tap index {tap_index} read different donor memories: {sorted(sources, key=str)}")

    state = {"memories": None, "cache_len": 0, "pad_mask": None}
    handles = []
    branches = []
    for host_layer, branch, tap_index in placements:
        _, layers = find_decoder_layers(host_model, host_layer)
        branch.host_layer = host_layer
        branch.tap_index = tap_index

        def hook(module, inputs, output, branch=branch):
            return replace_hidden(output, lambda h: apply_branch(state, branch, h))

        handles.append(layers[host_layer].register_forward_hook(hook))
        branches.append(branch)

    def store_call(module, args, kwargs):
        found = forward_args(module, args, kwargs, ["attention_mask", "past_key_values", "cache_position"])
        mask = found["attention_mask"]
        if mask is not None and (not torch.is_tensor(mask) or mask.dim() != 2):
            mask = None
        state["pad_mask"] = mask
        state["cache_len"] = host_cache_length(found["past_key_values"], found["cache_position"])

    decoder = decoder_of(host_model, placements[0][0])
    if decoder is None:
        decoder = host_model
    handles.append(decoder.register_forward_pre_hook(store_call, with_kwargs=True))

    device = host_device(host_model)
    for branch in branches:
        branch.to(device=device, dtype=torch.float32)
    host_model.wsp_taps = nn.ModuleList(branches)
    host_model.wsp_tap_state = state
    host_model.wsp_tap_handles = handles
    host_model.wsp_tap_stats = {"donor_seconds": 0.0, "donor_runs": 0, "donor_tokens": 0, "donors": {}}
    return handles


def detach_taps(host_model):
    """Remove the tap hooks and branches from the host."""
    for handle in getattr(host_model, "wsp_tap_handles", []):
        handle.remove()
    for name in ["wsp_taps", "wsp_tap_state", "wsp_tap_handles", "wsp_tap_stats"]:
        if hasattr(host_model, name):
            delattr(host_model, name)


def tap_count(host_model):
    """Return the number of distinct memories the taps read."""
    indices = set()
    for branch in host_model.wsp_taps:
        indices.add(branch.tap_index)
    return len(indices)


def set_tap_context(host_model, memories):
    """Give the taps their donor memory for the next forward calls.

    memories: a list with one entry per tap index. Each entry is
    (memory, valid, visibility):
    - memory: (batch, m, d_donor)
    - valid: (batch, m) bool
    - visibility: (batch, host_prompt_len, m) bool, built against the
      padded host prompt the caller feeds the host (left or right padding,
      it is already in the rows). Host positions at or past
      host_prompt_len see all valid memory.
    """
    if len(memories) != tap_count(host_model):
        raise ValueError(f"the taps read {tap_count(host_model)} memories, got {len(memories)}")
    device = host_device(host_model)
    entries = []
    for memory, valid, visibility in memories:
        if memory.shape[0] != valid.shape[0] or memory.shape[0] != visibility.shape[0]:
            raise ValueError("memory, valid, and visibility must have the same batch size")
        if memory.shape[1] != valid.shape[1] or memory.shape[1] != visibility.shape[2]:
            raise ValueError("memory, valid, and visibility must have the same memory length")
        entries.append((memory.to(device=device, dtype=torch.float32), valid.to(device).bool(),
                        visibility.to(device).bool()))
    host_model.wsp_tap_state["memories"] = entries


def clear_tap_context(host_model):
    """Drop the donor memory. A forward call after this raises an error."""
    host_model.wsp_tap_state["memories"] = None


def apply_branch(state, branch, h):
    """Run one branch on the hidden state h of the current host call."""
    memories = state["memories"]
    if memories is None:
        raise RuntimeError("the taps have no donor memory: call prepare or set_tap_context first")
    memory, valid, visibility = memories[branch.tap_index]
    batch = h.shape[0]
    seq = h.shape[1]
    if batch != memory.shape[0]:
        # generate repeats each prompt for beams and for
        # num_return_sequences with repeat_interleave, so we do the same.
        if batch % memory.shape[0] != 0:
            raise ValueError(f"host batch {batch} does not match the tap memory batch {memory.shape[0]}")
        repeats = batch // memory.shape[0]
        memory = memory.repeat_interleave(repeats, dim=0)
        valid = valid.repeat_interleave(repeats, dim=0)
        visibility = visibility.repeat_interleave(repeats, dim=0)

    m = memory.shape[1]
    mask = valid[:, None, :].expand(batch, seq, m).clone()
    start = state["cache_len"]
    prompt_len = visibility.shape[1]
    count = prompt_len - start
    if count > seq:
        count = seq
    if count > 0:
        mask[:, :count] = visibility[:, start:start + count]
    pad = state["pad_mask"]
    if pad is not None and pad.shape[0] == batch and pad.shape[1] >= seq:
        new_pad = pad[:, pad.shape[1] - seq:].to(device=mask.device).bool()
        mask = mask & new_pad[:, :, None]
    return branch(h, memory, mask)


def donor_requests(host_model):
    """Return a list, by tap index, of (donor_id, donor_layer)."""
    requests = [None] * tap_count(host_model)
    for branch in host_model.wsp_taps:
        requests[branch.tap_index] = (branch.donor_id, branch.donor_layer)
    return requests


def donor_memories(host_model, donors, prompts, derange=False):
    """Run each donor once on prompts and return, by tap index, (memory, ends, valid).

    donors: dict of donor_id -> (donor_model, donor_tokenizer). Taps that
    read the same donor share one run, up to the deepest layer any of them
    needs. derange=True rolls each memory by one along the batch, so
    example i reads the memory of example i - 1 (the Draft-KV control).
    """
    requests = donor_requests(host_model)
    layers_by_donor = {}
    for donor_id, donor_layer in requests:
        if donor_id not in donors:
            raise KeyError(f"no donor named {donor_id!r}; donors given: {sorted(donors)}")
        if donor_id not in layers_by_donor:
            layers_by_donor[donor_id] = []
        layers_by_donor[donor_id].append(donor_layer)

    if derange and len(prompts) < 2:
        raise ValueError("derange needs at least 2 prompts in a batch")

    stats = host_model.wsp_tap_stats
    runs = {}
    for donor_id in layers_by_donor:
        donor_model, donor_tokenizer = donors[donor_id]
        device = next(donor_model.parameters()).device
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        start = time.time()
        memories, ends, valid = run_donor_layers(donor_model, donor_tokenizer, prompts, layers_by_donor[donor_id])
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        stats["donor_seconds"] += time.time() - start
        stats["donor_runs"] += 1
        stats["donor_tokens"] += int(valid.sum())
        _, all_layers = find_decoder_layers(donor_model, 0)
        stats["donors"][donor_id] = {"layers_ran": max(layers_by_donor[donor_id]) + 1,
                                     "layers_total": len(all_layers)}
        runs[donor_id] = (memories, ends, valid)

    out = []
    for donor_id, donor_layer in requests:
        memories, ends, valid = runs[donor_id]
        memory = memories[donor_layer]
        if derange:
            memory = torch.roll(memory, shifts=1, dims=0)
            ends = torch.roll(ends, shifts=1, dims=0)
            valid = torch.roll(valid, shifts=1, dims=0)
        out.append((memory, ends, valid))
    return out


def prepare(host_model, host_tokenizer, donors, prompts, derange=False):
    """Run the taps' donors on prompts, set the tap context, and return the host inputs.

    Returns a dict with input_ids and attention_mask on the host device,
    tokenized with the host tokenizer's own padding side. Feed exactly
    these to the host (for example host_model.generate(**inputs)), because
    the visibility rows are built against this padded encoding.
    """
    input_ids, attention_mask, ends = host_token_ends(host_tokenizer, prompts)
    entries = []
    for memory, memory_ends, memory_valid in donor_memories(host_model, donors, prompts, derange):
        visibility = build_visibility(ends, memory_ends, memory_valid)
        entries.append((memory, memory_valid, visibility))
    set_tap_context(host_model, entries)
    device = host_device(host_model)
    return {"input_ids": input_ids.to(device), "attention_mask": attention_mask.to(device)}


def tap_parameters(host_model):
    """Return every parameter of every tap branch."""
    params = []
    for branch in host_model.wsp_taps:
        for param in branch.parameters():
            params.append(param)
    return params


def flat_params(params):
    """Return a flat list of tensors from a list of tensors or of optimizer param groups."""
    flat = []
    for item in params:
        if isinstance(item, dict):
            for param in item["params"]:
                flat.append(param)
        else:
            flat.append(item)
    return flat


def param_count(params):
    """Return the number of scalars in params (tensors or param groups)."""
    total = 0
    for param in flat_params(params):
        total += param.numel()
    return total


def tap_param_count(host_model):
    return param_count(tap_parameters(host_model))


def encode_pairs(host_tokenizer, pairs, seq_len):
    """Tokenize (prompt, completion) pairs for training, right padded.

    Each text is prompt + completion with the host's default special
    tokens, then the host EOS token. Labels keep only completion tokens
    (character end past the prompt) and the EOS. Returns
    (input_ids, attention_mask, labels, ends), each (batch, seq).
    """
    if host_tokenizer.pad_token is None:
        host_tokenizer.pad_token = host_tokenizer.eos_token
    rows = []
    for prompt, completion in pairs:
        text = prompt + completion
        encoded = host_tokenizer(text, return_offsets_mapping=True, return_special_tokens_mask=True)
        ids = list(encoded["input_ids"])
        special = list(encoded["special_tokens_mask"])
        ends = []
        for start, end in encoded["offset_mapping"]:
            ends.append(int(end))
        labels = []
        for index in range(len(ids)):
            if special[index] == 1:
                ends[index] = -1
                labels.append(-100)
            elif ends[index] > len(prompt):
                labels.append(ids[index])
            else:
                labels.append(-100)
        if host_tokenizer.eos_token_id is not None:
            ids.append(host_tokenizer.eos_token_id)
            ends.append(len(text))
            labels.append(host_tokenizer.eos_token_id)
        rows.append((ids[:seq_len], ends[:seq_len], labels[:seq_len]))

    longest = 0
    for ids, ends, labels in rows:
        if len(ids) > longest:
            longest = len(ids)
    batch = len(rows)
    input_ids = torch.full((batch, longest), host_tokenizer.pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros((batch, longest), dtype=torch.long)
    label_tensor = torch.full((batch, longest), -100, dtype=torch.long)
    end_tensor = torch.full((batch, longest), -1, dtype=torch.long)
    for row, (ids, ends, labels) in enumerate(rows):
        n = len(ids)
        input_ids[row, :n] = torch.tensor(ids, dtype=torch.long)
        attention_mask[row, :n] = 1
        label_tensor[row, :n] = torch.tensor(labels, dtype=torch.long)
        end_tensor[row, :n] = torch.tensor(ends, dtype=torch.long)
    return input_ids, attention_mask, label_tensor, end_tensor


def train_visibility(ends, prompt_chars, memory_ends, memory_valid):
    """Return build_visibility for a training batch, with completion tokens reading all valid memory.

    prompt_chars is a (batch,) long tensor of prompt lengths in characters.
    A token whose end is past its prompt is a completion token. At scoring
    time the same token is a decode step past the host prompt, and it reads
    all valid memory. Without derange, build_visibility already gives that.
    With derange, the memory comes from another prompt, and its ends can
    be past this token's end, so we set those rows here.
    """
    visibility = build_visibility(ends, memory_ends, memory_valid)
    prompt_chars = prompt_chars.to(ends.device).long()
    past_prompt = ends > prompt_chars[:, None]
    every = past_prompt[:, :, None] & memory_valid.to(ends.device).bool()[:, None, :]
    return visibility | every


def train_batch(host_model, host_tokenizer, donors, batch, seq_len, derange=False):
    """Encode (prompt, completion) pairs, run the donors on the prompts, and set the tap context.

    Returns a dict with input_ids, attention_mask, and labels on the host
    device, ready for host_model(**inputs).
    """
    input_ids, attention_mask, labels, ends = encode_pairs(host_tokenizer, batch, seq_len)
    if int((labels != -100).sum()) == 0:
        raise ValueError("no completion tokens fit in seq_len for this batch")
    prompts = []
    prompt_chars = []
    for prompt, completion in batch:
        prompts.append(prompt)
        prompt_chars.append(len(prompt))
    prompt_chars = torch.tensor(prompt_chars, dtype=torch.long)
    entries = []
    for memory, memory_ends, memory_valid in donor_memories(host_model, donors, prompts, derange):
        visibility = train_visibility(ends, prompt_chars, memory_ends, memory_valid)
        entries.append((memory, memory_valid, visibility))
    set_tap_context(host_model, entries)
    device = host_device(host_model)
    return {"input_ids": input_ids.to(device), "attention_mask": attention_mask.to(device),
            "labels": labels.to(device)}


def tap_train(host_model, host_tokenizer, donors, pairs, params, steps=300, lr=1e-4, batch_size=8,
              seq_len=384, seed=0, derange=False, log_every=10):
    """Train only params on (prompt, completion) pairs. Return the per-step losses.

    The loss is next-token cross-entropy on completion tokens only. The
    donors read only the prompt. params is a list of tensors, or a list of
    optimizer param groups (dicts with "params" and, if wanted, their own
    "lr"). Every host and donor parameter is frozen first, then params are
    unfrozen, so a host LoRA trains only if its parameters are in params.
    derange=True gives each example the donor memory of another example in
    the batch. Completion tokens read all valid memory, the same as decode
    steps at scoring time (see train_visibility).
    """
    flat = flat_params(params)
    for param in host_model.parameters():
        param.requires_grad_(False)
    for donor_id in donors:
        donor_model, _ = donors[donor_id]
        donor_model.eval()
        for param in donor_model.parameters():
            param.requires_grad_(False)
    for param in flat:
        param.requires_grad_(True)

    device = host_device(host_model)
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    generator = torch.Generator()
    generator.manual_seed(seed)
    use_autocast = device.type == "cuda"

    host_model.train()
    losses = []
    order = []
    try:
        for step in range(steps):
            batch = []
            while len(batch) < batch_size:
                if len(order) == 0:
                    order = torch.randperm(len(pairs), generator=generator).tolist()
                batch.append(pairs[order.pop()])

            inputs = train_batch(host_model, host_tokenizer, donors, batch, seq_len, derange)
            optimizer.zero_grad()
            if use_autocast:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    out = host_model(**inputs)
            else:
                out = host_model(**inputs)
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(flat, 1.0)
            optimizer.step()

            loss = float(out.loss.detach())
            losses.append(loss)
            if log_every and (step + 1) % log_every == 0:
                print(f"tap step {step + 1}/{steps} loss {loss:.4f}")
    finally:
        clear_tap_context(host_model)
        host_model.eval()
    return losses


def tap_meta(host_model):
    """Return one dict per branch: host layer, tap index, donor id and layer, sizes."""
    rows = []
    for branch in host_model.wsp_taps:
        rows.append({"host_layer": branch.host_layer, "tap_index": branch.tap_index,
                     "donor_id": branch.donor_id, "donor_layer": branch.donor_layer,
                     "d_host": branch.d_host, "d_donor": branch.d_donor,
                     "attn_dim": branch.attn_dim, "heads": branch.heads,
                     "gate": float(branch.gate.detach())})
    return rows


def save_taps(host_model, path, meta=None):
    """Write taps.json and tap_weights.safetensors to the directory path.

    meta is any extra JSON data to keep with the taps (for example the
    model name behind each donor id).
    """
    from safetensors.torch import save_file

    os.makedirs(path, exist_ok=True)
    state = {}
    for index, branch in enumerate(host_model.wsp_taps):
        for name, value in branch.state_dict().items():
            state[f"{index}.{name}"] = value.detach().cpu().contiguous()
    save_file(state, os.path.join(path, "tap_weights.safetensors"))
    with open(os.path.join(path, "taps.json"), "w") as f:
        json.dump({"taps": tap_meta(host_model), "meta": meta}, f, indent=2)


def load_taps(host_model, path):
    """Rebuild the branches saved in path and attach them to host_model.

    The caller loads the donors. Returns the saved info: {"taps": [...],
    "meta": ...}.
    """
    from safetensors.torch import load_file

    with open(os.path.join(path, "taps.json")) as f:
        info = json.load(f)
    state = load_file(os.path.join(path, "tap_weights.safetensors"))
    placements = []
    for index, row in enumerate(info["taps"]):
        branch = TapBranch(row["d_host"], row["d_donor"], attn_dim=row["attn_dim"], heads=row["heads"],
                           donor_id=row["donor_id"], donor_layer=row["donor_layer"])
        own = {}
        prefix = f"{index}."
        for key in state:
            if key.startswith(prefix):
                own[key[len(prefix):]] = state[key]
        branch.load_state_dict(own)
        placements.append((row["host_layer"], branch, row["tap_index"]))
    attach_taps(host_model, placements)
    return info


def donor_flops_note(host_model, donors):
    """Return one dict per donor the taps read: layers run, parameters run, and rough FLOPs.

    flops_per_token is 2 times the parameters in the decoder layers that
    run, a rough count that leaves out attention over the sequence and the
    embedding lookup. It also gives the totals prepare has measured so far.
    """
    needed = {}
    for donor_id, donor_layer in donor_requests(host_model):
        if donor_id not in needed or donor_layer > needed[donor_id]:
            needed[donor_id] = donor_layer
    stats = host_model.wsp_tap_stats
    notes = []
    for donor_id in needed:
        donor_model, _ = donors[donor_id]
        _, layers = find_decoder_layers(donor_model, 0)
        ran = needed[donor_id] + 1
        params_ran = 0
        for index in range(ran):
            for param in layers[index].parameters():
                params_ran += param.numel()
        notes.append({"donor_id": donor_id, "layers_ran": ran, "layers_total": len(layers),
                      "params_ran": params_ran, "flops_per_token": 2 * params_ran,
                      "donor_seconds_so_far": round(stats["donor_seconds"], 4),
                      "donor_runs_so_far": stats["donor_runs"],
                      "donor_tokens_so_far": stats["donor_tokens"]})
    return notes
