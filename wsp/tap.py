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

Auto-prepare: after set_tap_sources(host, host_tokenizer, donors), the
host also works with plain host(input_ids) and host.generate(input_ids,
attention_mask=...). On a prefill call with no manual context, the taps
decode each row's host ids back to text, run the donors on those texts,
and build the visibility against the given ids (see auto_context). A
manual prepare or set_tap_context takes priority until
clear_tap_context. wsp.load sets the sources for a compiled model.

Cached donors: a donors dict entry can be a CachedDonor in place of
(donor_model, donor_tokenizer). It reads memories that a donor computed
once for a fixed prompt set (see wsp.tap_cache and wsp.donor_extract), so
taps on very large donors train and score with no donor model loaded.
"""

import collections
import contextlib
import json
import math
import os
import time
import types

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

    def __init__(self, d_host, d_donor, attn_dim=256, heads=8, donor_id=None, donor_layer=None, tap_id=None):
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
        self.tap_id = tap_id
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


class CachedDonor:
    """A donor whose memories were computed once and stored with wsp.tap_cache.

    Use it in a donors dict in place of (donor_model, donor_tokenizer):

        donors = {"glm": CachedDonor("/vol/tap_cache/glm53flash", 31)}

    It answers lookups by exact prompt text, with the outputs run_donor
    gives: memory (batch, m, d) float32, memory_ends (batch, m) character
    end offsets in the prompt text (-1 for specials and pads), and valid
    (batch, m) bool. Rows are right padded to the longest row. A prompt
    that is not in the cache raises a KeyError that names it.

    layer: the one cached layer this donor gives, or None for any cached
    layer. max_shards: how many shards (per layer) stay loaded, least
    recently used first out; None keeps every shard loaded. A shard is
    read on first use, and only the layers asked for. device: where lookup
    puts its outputs by default (CPU when None).
    """

    def __init__(self, path, layer=None, max_shards=4, device=None):
        from wsp.tap_cache import read_index, read_meta

        self.path = path
        self.meta = read_meta(path)
        self.index = read_index(path)
        self.layers = list(self.meta["layers"])
        self.hidden_size = int(self.meta["hidden_size"])
        self.donor_id = self.meta.get("donor_id")
        self.layers_total = self.meta.get("layers_total")
        if layer is not None and layer not in self.layers:
            raise ValueError(f"layer {layer} is not in the cache at {path}; it holds layers {self.layers}")
        self.layer = layer
        self.max_shards = max_shards
        self.device = device
        # Least recently used caches: shard number -> row info, and (shard, layer) -> states.
        self.row_cache = collections.OrderedDict()
        self.layer_cache = collections.OrderedDict()
        self.shard_reads = 0

    def __repr__(self):
        return f"CachedDonor({self.path!r}, layer={self.layer}, donor_id={self.donor_id!r})"

    def check_layers(self, layers):
        wanted = sorted(set(layers))
        for layer in wanted:
            if self.layer is not None and layer != self.layer:
                raise ValueError(f"this CachedDonor gives layer {self.layer} only; a tap asked for layer {layer}")
            if layer not in self.layers:
                raise ValueError(f"layer {layer} is not in the cache at {self.path}; it holds layers {self.layers}")
        return wanted

    def remember(self, store, key, value):
        store[key] = value
        store.move_to_end(key)
        if self.max_shards is not None:
            while len(store) > self.max_shards:
                store.popitem(last=False)

    def shard_rows(self, number):
        from wsp.tap_cache import read_shard_rows

        if number in self.row_cache:
            self.row_cache.move_to_end(number)
            return self.row_cache[number]
        rows = read_shard_rows(self.path, number)
        self.remember(self.row_cache, number, rows)
        return rows

    def shard_layer(self, number, layer):
        from wsp.tap_cache import read_shard_layer

        key = (number, layer)
        if key in self.layer_cache:
            self.layer_cache.move_to_end(key)
            return self.layer_cache[key]
        states = read_shard_layer(self.path, number, layer)
        self.shard_reads += 1
        self.remember(self.layer_cache, key, states)
        return states

    def locate(self, prompt):
        """Return (shard number, row) of a prompt, or raise a KeyError that names the prompt."""
        from wsp.tap_cache import prompt_hash

        found = self.index.get(prompt_hash(prompt))
        if found is None:
            raise KeyError(f"prompt not in the donor cache at {self.path}: {prompt[:200]!r}")
        return int(found[0]), int(found[1])

    def lookup(self, prompts, layers=None, device=None):
        """Return (memories, ends, valid) like run_donor_layers: memories is a dict of layer -> (batch, m, d)."""
        from wsp.tap_cache import prompt_hash

        if layers is None:
            if self.layer is None:
                raise ValueError("this CachedDonor holds several layers; name the layers to look up")
            layers = [self.layer]
        wanted = self.check_layers(layers)
        if device is None:
            device = self.device
        if device is None:
            device = torch.device("cpu")
        prompts = list(prompts)
        places = []
        for prompt in prompts:
            places.append(self.locate(prompt))

        # Visit each shard once, so a batch never loads a shard twice.
        by_shard = collections.OrderedDict()
        for position in range(len(prompts)):
            number, row = places[position]
            if number not in by_shard:
                by_shard[number] = []
            by_shard[number].append(position)

        pieces = [None] * len(prompts)
        for number in by_shard:
            offsets, ends, valid, shard_prompts = self.shard_rows(number)
            states = {}
            for layer in wanted:
                states[layer] = self.shard_layer(number, layer)
            for position in by_shard[number]:
                row = places[position][1]
                if shard_prompts[row] != prompts[position]:
                    raise KeyError(f"cache index points {prompt_hash(prompts[position])[:12]} at a different "
                                   f"prompt in {self.path}: {prompts[position][:200]!r}")
                start = int(offsets[row])
                stop = int(offsets[row + 1])
                row_states = {}
                for layer in wanted:
                    row_states[layer] = states[layer][start:stop]
                pieces[position] = (row_states, ends[start:stop], valid[start:stop])

        longest = 1
        for row_states, row_ends, row_valid in pieces:
            if row_ends.shape[0] > longest:
                longest = row_ends.shape[0]
        batch = len(prompts)
        out_ends = torch.full((batch, longest), -1, dtype=torch.long)
        out_valid = torch.zeros((batch, longest), dtype=torch.bool)
        memories = {}
        for layer in wanted:
            memories[layer] = torch.zeros((batch, longest, self.hidden_size), dtype=torch.float32)
        for position in range(batch):
            row_states, row_ends, row_valid = pieces[position]
            n = row_ends.shape[0]
            out_ends[position, :n] = row_ends
            out_valid[position, :n] = row_valid
            for layer in wanted:
                memories[layer][position, :n] = row_states[layer].to(torch.float32)
        out_ends[~out_valid] = -1
        for layer in wanted:
            memories[layer] = memories[layer].to(device)
        return memories, out_ends.to(device), out_valid.to(device)

    def memory(self, prompts, layer=None, device=None):
        """Return (memory, memory_ends, valid) for one layer, like run_donor."""
        if layer is None:
            layer = self.layer
        memories, ends, valid = self.lookup(prompts, [layer], device=device)
        return memories[layer], ends, valid


def is_cached_donor(source):
    return isinstance(source, CachedDonor)


def donor_model_of(source):
    """Return the donor model of a donors dict entry, or None for a CachedDonor."""
    if is_cached_donor(source):
        return None
    return source[0]


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

    donor_model can also be a CachedDonor; then donor_tokenizer is not
    used and the memories come from the cache.
    """
    if is_cached_donor(donor_model):
        return donor_model.lookup(prompts, layers)
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

    # auto is True when auto_context built the memories. in_generate is
    # True while tap_generate runs: it built the context from the whole
    # prompt, so the forward calls inside generate keep it.
    state = {"memories": None, "cache_len": 0, "pad_mask": None, "auto": False, "in_generate": False}
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
        found = forward_args(module, args, kwargs, ["input_ids", "attention_mask", "past_key_values",
                                                    "cache_position"])
        mask = found["attention_mask"]
        if mask is not None and (not torch.is_tensor(mask) or mask.dim() != 2):
            mask = None
        state["pad_mask"] = mask
        state["cache_len"] = host_cache_length(found["past_key_values"], found["cache_position"])
        if state["cache_len"] == 0:
            maybe_auto_context(host_model, found["input_ids"], mask)

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


def set_tap_sources(host_model, host_tokenizer, donors):
    """Store the host tokenizer and the donors on the host, so auto-prepare can run.

    donors: dict of donor_id -> (donor_model, donor_tokenizer). They are
    kept in plain attributes, not as submodules, so the donors are not
    part of host_model.parameters() or its state dict, and host_model.to()
    does not move them.
    """
    host_model.wsp_tap_tokenizer = host_tokenizer
    host_model.wsp_tap_donors = dict(donors)
    # generate gets a wrapper that auto-prepares on the whole prompt (see
    # tap_generate). It is bound to host_model, so a deep copy binds it to
    # the copy, and on a model without tap sources it is plain generate.
    if "generate" not in host_model.__dict__:
        host_model.generate = types.MethodType(tap_generate, host_model)


def tap_generate(self, *args, **kwargs):
    """generate, with auto-prepare on the whole prompt before the first forward call.

    The forward hooks alone see only one call at a time. With chunked
    prefill, the first call holds only part of the prompt, and with
    use_cache=False every step looks like a new prefill. So here we build
    the context once from the full input_ids and attention_mask, and the
    forward calls inside this generate keep it. A manual context is kept
    as it is, and inputs_embeds without input_ids fall back to the hooks.
    """
    plain = type(self).generate
    state = getattr(self, "wsp_tap_state", None)
    if state is None or getattr(self, "wsp_tap_donors", None) is None:
        return plain(self, *args, **kwargs)
    if state["memories"] is not None and not state["auto"]:
        return plain(self, *args, **kwargs)
    input_ids = kwargs.get("input_ids")
    if input_ids is None:
        input_ids = kwargs.get("inputs")
    if input_ids is None and len(args) > 0:
        input_ids = args[0]
    if not torch.is_tensor(input_ids) or input_ids.dim() != 2 or input_ids.is_floating_point():
        return plain(self, *args, **kwargs)
    mask = kwargs.get("attention_mask")
    if mask is not None and (not torch.is_tensor(mask) or mask.dim() != 2):
        mask = None
    auto_context(self, input_ids, mask)
    before = state["in_generate"]
    state["in_generate"] = True
    try:
        return plain(self, *args, **kwargs)
    finally:
        state["in_generate"] = before


def detach_taps(host_model):
    """Remove the tap hooks and branches from the host."""
    for handle in getattr(host_model, "wsp_tap_handles", []):
        handle.remove()
    for name in ["wsp_taps", "wsp_tap_state", "wsp_tap_handles", "wsp_tap_stats", "wsp_tap_tokenizer",
                 "wsp_tap_donors"]:
        if hasattr(host_model, name):
            delattr(host_model, name)
    if "generate" in host_model.__dict__:
        del host_model.__dict__["generate"]


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
    state = host_model.wsp_tap_state
    state["memories"] = entries
    state["auto"] = False


def clear_tap_context(host_model):
    """Drop the donor memory.

    Without tap sources, a forward call after this raises an error. With
    them, the next prefill builds the context itself (auto-prepare).
    """
    state = host_model.wsp_tap_state
    state["memories"] = None
    state["auto"] = False


def common_prefix_length(a, b):
    """Return the number of leading characters a and b share."""
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


def host_text_and_ends(host_tokenizer, ids, real):
    """Rebuild the prompt text of one host row and each token's character end in it.

    ids: a list of host token ids. real: a list of bools, False for pads.
    Pads and special tokens get end -1 and add no text. The text is the
    decode of the other tokens.

    When the text encodes back to exactly those ids, the ends are the
    tokenizer's own offsets, the same ones prepare uses. Otherwise each
    token's end is the number of characters that the decode of the ids up
    to it shares with the text. A token then never sees past its own end,
    so the visibility is exact or stricter.
    """
    special = set(host_tokenizer.all_special_ids)
    positions = []
    kept = []
    for index in range(len(ids)):
        if not real[index] or ids[index] in special:
            continue
        positions.append(index)
        kept.append(ids[index])
    ends = [-1] * len(ids)
    text = host_tokenizer.decode(kept, skip_special_tokens=False, clean_up_tokenization_spaces=False)
    if len(kept) == 0:
        return text, ends

    encoded = host_tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    if list(encoded["input_ids"]) == kept:
        for k in range(len(positions)):
            ends[positions[k]] = int(encoded["offset_mapping"][k][1])
        return text, ends

    for k in range(len(positions)):
        prefix = host_tokenizer.decode(kept[:k + 1], skip_special_tokens=False, clean_up_tokenization_spaces=False)
        ends[positions[k]] = common_prefix_length(prefix, text)
    return text, ends


def auto_context(host_model, input_ids, attention_mask=None):
    """Build and set the tap context from host input ids (auto-prepare).

    Each row's non-pad, non-special ids decode to a text with the host
    tokenizer. The donors read those texts, and the visibility rows match
    input_ids position by position. Left and right padding both work.
    """
    tokenizer = host_model.wsp_tap_tokenizer
    donors = host_model.wsp_tap_donors
    ids_rows = input_ids.tolist()
    mask_rows = None
    if attention_mask is not None:
        mask_rows = attention_mask.tolist()
    texts = []
    all_ends = []
    for row in range(len(ids_rows)):
        real = [True] * len(ids_rows[row])
        if mask_rows is not None:
            for index in range(len(real)):
                real[index] = mask_rows[row][index] == 1
        text, ends = host_text_and_ends(tokenizer, ids_rows[row], real)
        texts.append(text)
        all_ends.append(ends)
    host_ends = torch.tensor(all_ends, dtype=torch.long)

    entries = []
    for memory, memory_ends, memory_valid in donor_memories(host_model, donors, texts):
        visibility = build_visibility(host_ends, memory_ends, memory_valid)
        entries.append((memory, memory_valid, visibility))
    set_tap_context(host_model, entries)
    host_model.wsp_tap_state["auto"] = True
    return texts


def maybe_auto_context(host_model, input_ids, attention_mask):
    """Run auto-prepare on a prefill call when the taps have sources and no manual context.

    A manual context (prepare, set_tap_context, a training batch) is kept.
    Inside generate, tap_generate already built the context from the
    whole prompt, so it is kept too. Outside generate, every call with an
    empty cache is a new prompt and builds a new auto context, so a call
    does not depend on the calls before it.
    """
    if getattr(host_model, "wsp_tap_donors", None) is None or input_ids is None:
        return
    state = host_model.wsp_tap_state
    if state["memories"] is not None and not state["auto"]:
        return
    if state["memories"] is not None and state["in_generate"]:
        return
    auto_context(host_model, input_ids, attention_mask)


def apply_branch(state, branch, h):
    """Run one branch on the hidden state h of the current host call."""
    memories = state["memories"]
    if memories is None:
        raise RuntimeError("the taps have no donor memory: call prepare or set_tap_context first, "
                           "or set_tap_sources for auto-prepare")
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

    donors: dict of donor_id -> (donor_model, donor_tokenizer), or
    donor_id -> CachedDonor. Taps that read the same donor share one run,
    up to the deepest layer any of them needs. A CachedDonor runs no
    model: it looks the prompts up and puts the memories on the host
    device. derange=True rolls each memory by one along the batch, so
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
        source = donors[donor_id]
        if is_cached_donor(source):
            start = time.time()
            memories, ends, valid = source.lookup(prompts, layers_by_donor[donor_id], device=host_device(host_model))
            stats["donor_seconds"] += time.time() - start
            stats["donor_runs"] += 1
            stats["donor_tokens"] += int(valid.sum())
            stats["donors"][donor_id] = {"layers_ran": max(layers_by_donor[donor_id]) + 1,
                                         "layers_total": source.layers_total, "cached": True}
            runs[donor_id] = (memories, ends, valid)
            continue
        donor_model, donor_tokenizer = source
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
        donor_model = donor_model_of(donors[donor_id])
        if donor_model is None:
            continue
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
        rows.append({"host_layer": branch.host_layer, "tap_index": branch.tap_index, "tap_id": branch.tap_id,
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
                           donor_id=row["donor_id"], donor_layer=row["donor_layer"], tap_id=row.get("tap_id"))
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
        ran = needed[donor_id] + 1
        source = donors[donor_id]
        if is_cached_donor(source):
            # The cache meta may give the parameter count of layers 0..L
            # (params_ran_by_layer); without it the FLOPs are unknown.
            by_layer = source.meta.get("params_ran_by_layer", {})
            params_ran = by_layer.get(str(needed[donor_id]))
            layers_total = source.layers_total
        else:
            _, layers = find_decoder_layers(source[0], 0)
            layers_total = len(layers)
            params_ran = 0
            for index in range(ran):
                for param in layers[index].parameters():
                    params_ran += param.numel()
        flops = None
        if params_ran is not None:
            flops = 2 * params_ran
        notes.append({"donor_id": donor_id, "layers_ran": ran, "layers_total": layers_total,
                      "params_ran": params_ran, "flops_per_token": flops,
                      "donor_seconds_so_far": round(stats["donor_seconds"], 4),
                      "donor_runs_so_far": stats["donor_runs"],
                      "donor_tokens_so_far": stats["donor_tokens"]})
    return notes


def module_params(module):
    total = 0
    for param in module.parameters():
        total += param.numel()
    return total


def host_flops_per_token(host_model):
    """Return 2 times the host decoder layer and output head parameters, a rough decode cost per token.

    Tap branches are left out. A host LoRA counts, since it sits inside the
    decoder layers.
    """
    _, layers = find_decoder_layers(host_model, 0)
    total = 0
    for layer in layers:
        total += module_params(layer)
    head = host_model.get_output_embeddings()
    if head is not None:
        total += module_params(head)
    return 2 * total


def tap_cost(host_model, donors=None, memory_tokens=128):
    """Estimate what the taps add per decoded token and per prompt, in rough FLOPs.

    memory_tokens is the donor prompt length in donor tokens. Costs:
    - decode: each branch runs q and o on the new token, attends over the
      memory, and computes k and v over the whole memory again on every
      call (the branch keeps no k and v cache). So the per-token cost
      grows with the memory length.
    - prefill: each donor runs once per prompt, up to the deepest layer
      any tap reads: 2 times the parameters of the layers that run, per
      donor token. Attention over the sequence is left out.
    donors: dict of donor_id -> (model, tokenizer), or None to use the
    sources set by set_tap_sources. Without donors, the donor fields are
    left out.
    """
    m = memory_tokens
    branch_rows = []
    decode_fixed = 0
    decode_per_memory = 0
    for branch in host_model.wsp_taps:
        fixed = 2 * branch.d_host * branch.attn_dim * 2
        per_memory = 2 * branch.d_donor * branch.attn_dim * 2 + 2 * branch.attn_dim * 2
        decode_fixed += fixed
        decode_per_memory += per_memory
        branch_rows.append({"tap_id": branch.tap_id, "donor_id": branch.donor_id,
                            "donor_layer": branch.donor_layer, "host_layer": branch.host_layer,
                            "params": module_params(branch),
                            "decode_flops_per_token": fixed + per_memory * m})
    host_flops = host_flops_per_token(host_model)
    tap_decode = decode_fixed + decode_per_memory * m
    cost = {"memory_tokens": m, "host_decode_flops_per_token": host_flops,
            "tap_decode_flops_per_token": tap_decode,
            "tap_decode_flops_fixed": decode_fixed,
            "tap_decode_flops_per_memory_token": decode_per_memory,
            "tap_decode_fraction": tap_decode / host_flops, "tap_params": tap_param_count(host_model),
            "branches": branch_rows, "donors": []}

    if donors is None:
        donors = getattr(host_model, "wsp_tap_donors", None)
    if donors is None:
        return cost
    total_prefill = 0
    for note in donor_flops_note(host_model, donors):
        per_prompt = None
        if note["flops_per_token"] is not None:
            per_prompt = note["flops_per_token"] * m
            total_prefill += per_prompt
        cost["donors"].append({"donor_id": note["donor_id"], "layers_ran": note["layers_ran"],
                               "layers_total": note["layers_total"], "params_ran": note["params_ran"],
                               "prefill_flops_per_donor_token": note["flops_per_token"],
                               "prefill_flops_per_prompt": per_prompt})
    cost["donor_prefill_flops_per_prompt"] = total_prefill
    return cost
