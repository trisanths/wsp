"""Grafts: attach a donor part to a host model through trainable glue.

A graft reads the host hidden state, maps it into the donor's width with
w_in, runs the frozen donor part, maps the result back with w_out, and adds
it to the hidden state. w_out starts at zero, so a fresh graft changes
nothing. Only w_in and w_out train.

Three options change the donor input after w_in:
  sink        a stored donor state replaces the mapped vector at each
              sequence's first real position. That position is the donor's
              attention sink (the BOS, or the first text token when there is
              no BOS). Sink states are huge and do not depend on the input,
              and a mapped vector there breaks attention for every later
              token. The audit cut the stitch penalty 2.1x to 3.4x with it.
  rms match   each other mapped vector is scaled to the donor's typical RMS
              at the input point, times a trained scalar that starts at 1.
  part LoRA   wsp.lora adapters on the linears inside the donor part (the
              receiver), trained with the glue. B starts at zero, so a new
              part LoRA changes nothing.
"""

import inspect
import weakref

import torch
from torch import nn

from wsp.parts import AttnHeadsPart, BlockPart, find_decoder_layers


class LowRankLinear(nn.Module):
    """A bias-free linear map factored as up(down(x)), with rank r."""

    def __init__(self, d_in, d_out, rank):
        super().__init__()
        self.down = nn.Linear(d_in, rank, bias=False)
        self.up = nn.Linear(rank, d_out, bias=False)

    def forward(self, x):
        return self.up(self.down(x))

    def set_from_full(self, w):
        """Set the factors to the truncated SVD of w, shape (d_out, d_in)."""
        rank = self.down.weight.shape[0]
        u, s, vh = torch.linalg.svd(w.float(), full_matrices=False)
        root = s[:rank].sqrt()
        self.up.weight.copy_((u[:, :rank] * root).to(self.up.weight.dtype))
        self.down.weight.copy_((root[:, None] * vh[:rank]).to(self.down.weight.dtype))


class Graft(nn.Module):
    """h + w_out(part(w_in(h))), with float32 glue and a frozen part.

    attach_grafts sets a context for each host forward call before it runs:
    pad_mask is the host's 2D padding mask (batch, past + new) or None,
    host_uses_cache says if the host call uses a KV cache, and
    host_cache_len is the host cache length before the call (0 for a new
    prefill). Only an AttnHeadsPart or a BlockPart uses the context,
    because only attention mixes positions. When the host uses a cache,
    the part keeps its own cache in step with it.
    """

    def __init__(self, d_host, d_donor, part, rank=None, w_in=None):
        super().__init__()
        self.rank = rank
        if rank is None:
            self.w_in = nn.Linear(d_host, d_donor, bias=False)
            self.w_out = nn.Linear(d_donor, d_host, bias=False)
            nn.init.zeros_(self.w_out.weight)
        else:
            self.w_in = LowRankLinear(d_host, d_donor, rank)
            self.w_out = LowRankLinear(d_donor, d_host, rank)
            nn.init.zeros_(self.w_out.up.weight)
        if w_in is not None:
            # Any module can be w_in, for example a trained stitch from
            # wsp.stitch. A module with host_layers (a multi stitch) gets a
            # list of host states, one per layer; attach_grafts captures them.
            self.w_in = w_in
        self.part = part
        self.pad_mask = None
        self.host_uses_cache = False
        self.host_cache_len = 0
        self.host_layer = None
        self.stitch_inputs = {}
        # The sink and RMS options are off until set_sink and set_rms_match
        # turn them on. A None buffer or parameter is left out of the state.
        self.register_buffer("sink", None)
        self.register_buffer("donor_rms", None)
        self.register_parameter("rms_scale", None)

    def set_sink(self, vector):
        """Feed the part this donor state, shape (d_donor,), at each sequence's first real position."""
        self.sink = vector.detach().to(torch.float32).reshape(-1).clone()

    def set_rms_match(self, donor_rms):
        """Scale each mapped vector to donor_rms times a trained scalar that starts at 1."""
        self.donor_rms = torch.as_tensor(donor_rms, dtype=torch.float32).detach().reshape(()).clone()
        self.rms_scale = nn.Parameter(torch.ones((), dtype=torch.float32))

    def set_w_in(self, w):
        """Start w_in from a full (d_donor, d_in) matrix, for example a ridge fit.

        A low-rank w_in takes the best rank-r approximation of w. Any other
        module must have a set_from_full(w) method.
        """
        with torch.no_grad():
            if isinstance(self.w_in, nn.Linear):
                self.w_in.weight.copy_(w.to(self.w_in.weight.dtype))
            else:
                self.w_in.set_from_full(w)

    def map_in(self, h):
        """Apply w_in to the host state h, or to the host states of every layer it reads."""
        stitch_layers = getattr(self.w_in, "host_layers", None)
        if stitch_layers is None:
            return self.w_in(h.to(torch.float32))
        states = []
        for layer in stitch_layers:
            if layer == self.host_layer:
                states.append(h.to(torch.float32))
            else:
                states.append(self.stitch_inputs[layer].to(torch.float32))
        # Drop the captured states, so they do not outlive this call.
        self.stitch_inputs = {}
        return self.w_in(states)

    def match_rms(self, x):
        """Scale each row of x to RMS donor_rms * rms_scale."""
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + 1e-12)
        target = self.donor_rms.to(x.device) * self.rms_scale.to(x.device)
        return x * (target / rms)

    def put_sink(self, x):
        """Replace x at each sequence's first real position with the sink vector.

        The first real position is the first 1 in pad_mask, so left padding
        works, or 0 when there is no mask. A cached call holds only the new
        tokens, which start at host_cache_len, so a decode step or a later
        prefill chunk that does not hold the first position changes nothing.
        """
        batch, seq, _ = x.shape
        start = 0
        if self.host_uses_cache:
            start = self.host_cache_len
        if self.pad_mask is None:
            first = torch.zeros(batch, dtype=torch.long, device=x.device)
            has_real = torch.ones(batch, dtype=torch.bool, device=x.device)
        else:
            real = self.pad_mask.to(x.device) != 0
            # argmax gives the first maximum, so the first real position.
            first = real.long().argmax(dim=-1)
            has_real = real.any(dim=-1)
        positions = torch.arange(start, start + seq, device=x.device)
        hit = (positions.view(1, seq) == first.view(batch, 1)) & has_real.view(batch, 1)
        sink = self.sink.to(device=x.device, dtype=x.dtype).view(1, 1, -1)
        return torch.where(hit.unsqueeze(-1), sink, x)

    def forward(self, h):
        dtype = part_dtype(self.part)
        x = self.map_in(h)
        if self.rms_scale is not None:
            x = self.match_rms(x)
        if self.sink is not None:
            x = self.put_sink(x)
        if isinstance(self.part, (AttnHeadsPart, BlockPart)):
            if self.host_uses_cache:
                x = self.part(x.to(dtype), attention_mask=self.pad_mask, cache_len=self.host_cache_len)
            else:
                x = self.part(x.to(dtype), attention_mask=self.pad_mask)
        else:
            x = self.part(x.to(dtype))
        delta = self.w_out(x.to(torch.float32))
        return h + delta.to(h.dtype)


def part_dtype(part):
    """Return the dtype of the donor weights in a part.

    Part LoRA factors are float32 whatever the donor dtype, so they are
    skipped. A part with no weights runs in float32.
    """
    for name, param in part.named_parameters():
        if name.endswith("lora_A") or name.endswith("lora_B"):
            continue
        return param.dtype
    return torch.float32


def replace_hidden(output, fn):
    """Apply fn to the hidden state in a layer output.

    The output is either a tensor or a tuple whose first item is the hidden
    state. The same structure comes back.
    """
    if isinstance(output, tuple):
        return (fn(output[0]),) + tuple(output[1:])
    return fn(output)


def attach_grafts(host_model, placements):
    """Hook grafts onto host decoder layers.

    placements: list of (host_layer_index, graft). The grafts are stored in
    host_model.wsp_grafts so they move with .to() and show up in
    .parameters(). Returns the list of hook handles.

    A pre-hook on host_model reads each forward call and sets the context
    of every graft: pad_mask (the 2D attention_mask, or None),
    host_uses_cache and host_cache_len. The host uses a cache when the call
    passes past_key_values or use_cache=True. generate does both, so
    attn_heads and block parts keep their own KV caches in step with the
    host cache. To run without any cache, call disable_cache(host_model).
    """
    for layer_index, graft in placements:
        stitch_layers = getattr(getattr(graft, "w_in", None), "host_layers", None)
        if stitch_layers is None:
            continue
        for source in stitch_layers:
            if source > layer_index:
                raise ValueError(f"a graft at host layer {layer_index} cannot read host layer {source}: "
                                 "a multi stitch reads only its own layer and earlier ones")

    handles = []
    grafts = []
    for layer_index, graft in placements:
        _, layers = find_decoder_layers(host_model, layer_index)

        def hook(module, inputs, output, graft=graft):
            return replace_hidden(output, graft)

        handles.append(layers[layer_index].register_forward_hook(hook))
        grafts.append(graft)
        graft.host_layer = layer_index
    # A multi stitch also reads earlier host layers. These hooks come after
    # the graft hooks, so they see each layer's output after its grafts,
    # the same residual stream a graft at that layer would see.
    for layer_index, graft in placements:
        stitch_layers = getattr(getattr(graft, "w_in", None), "host_layers", None)
        if stitch_layers is None:
            continue
        for source in stitch_layers:
            if source == layer_index:
                continue
            _, layers = find_decoder_layers(host_model, source)

            def capture(module, inputs, output, graft=graft, source=source):
                if isinstance(output, tuple):
                    output = output[0]
                graft.stitch_inputs[source] = output

            handles.append(layers[source].register_forward_hook(capture))
    host_model.wsp_grafts = nn.ModuleList(grafts)
    # follow["cache"] is a weak reference to the host cache that the part
    # caches are in step with, or None. Only attn_heads and block parts
    # keep a cache.
    follow = {"cache": None}
    has_part_cache = False
    for graft in grafts:
        if isinstance(getattr(graft, "part", None), (AttnHeadsPart, BlockPart)):
            has_part_cache = True
    # A graft with a sink reads the 2D mask to find each row's first real
    # token, so it needs a 2D mask too.
    needs_2d_mask = has_part_cache or any_sink(grafts)
    generation_config = getattr(host_model, "generation_config", None)
    if needs_2d_mask and generation_config is not None:
        # Some checkpoints (Gemma 3) ask generate for a static hybrid cache.
        # Part caches need the default dynamic cache and a 2D mask.
        if getattr(generation_config, "cache_implementation", None) is not None:
            generation_config.cache_implementation = None

    def store_call_context(module, args, kwargs):
        found = forward_args(module, args, kwargs, ["attention_mask", "past_key_values", "use_cache", "cache_position"])
        mask = found["attention_mask"]
        past = found["past_key_values"]
        uses_cache = past is not None or found["use_cache"] is True
        # Only a 2D (batch, seq) tensor is a padding mask. Static and hybrid
        # caches give a 4D mask or a dict of masks instead.
        if mask is not None and (not torch.is_tensor(mask) or mask.dim() != 2):
            if has_part_cache and uses_cache:
                raise ValueError(
                    "attn_heads and block grafts need a 2D attention_mask when the host uses a "
                    "cache; static and hybrid caches give a 4D mask that hides the padding, so "
                    "use the default dynamic cache (generation_config.cache_implementation = None)"
                )
            # Without the 2D mask, a sink would go on position 0, which is a
            # pad under left padding. We check at call time, because
            # set_sink can run after attach_grafts.
            if any_sink(grafts):
                raise ValueError(
                    "a graft with a sink needs a 2D attention_mask to find each row's first real "
                    "token; a static cache gives a 4D mask that hides the padding, so use the "
                    "default dynamic cache (generation_config.cache_implementation = None)"
                )
            mask = None
        cache_len = host_cache_length(past, found["cache_position"])
        if has_part_cache and uses_cache and cache_len > 0:
            # A decode step must continue the host cache that the part
            # caches follow, not another cache of the same length.
            followed = None
            if follow["cache"] is not None:
                followed = follow["cache"]()
            if past is not followed:
                raise ValueError(
                    "graft part caches are out of sync: this call passes a host cache that the "
                    "part caches do not follow (a copy, or a cache from another prefill); start "
                    "again with a new prefill (an empty cache and use_cache=True)"
                )
        if has_part_cache and uses_cache and cache_len == 0 and past is not None:
            follow_host_cache(follow, past, grafts)
        for graft in grafts:
            graft.pad_mask = mask
            graft.host_uses_cache = uses_cache
            graft.host_cache_len = cache_len

    def follow_returned_cache(module, args, output):
        # A prefill without past_key_values makes its cache inside the
        # host, so we find it in the output.
        if not has_part_cache or not grafts[0].host_uses_cache or grafts[0].host_cache_len != 0:
            return
        cache = getattr(output, "past_key_values", None)
        if cache is not None:
            follow_host_cache(follow, cache, grafts)

    # generate with prefill_chunk_size calls host_model.forward directly,
    # so the host pre-hook does not run for the prefill chunks. The
    # decoder model inside is still called through its hooks, so a
    # second pre-hook there reads the call when the host hook did not.
    in_host_call = {"active": False}

    def enter_host_call(module, args, kwargs):
        in_host_call["active"] = True
        store_call_context(module, args, kwargs)

    def leave_host_call(module, args, output):
        in_host_call["active"] = False
        follow_returned_cache(module, args, output)

    def store_direct_call_context(module, args, kwargs):
        if not in_host_call["active"]:
            store_call_context(module, args, kwargs)

    handles.append(host_model.register_forward_pre_hook(enter_host_call, with_kwargs=True))
    handles.append(host_model.register_forward_hook(leave_host_call, always_call=True))
    if len(placements) > 0:
        decoder = decoder_of(host_model, placements[0][0])
        if decoder is not None:
            handles.append(decoder.register_forward_pre_hook(store_direct_call_context, with_kwargs=True))
    return handles


def any_sink(grafts):
    """Return True when any graft in the list has a sink."""
    for graft in grafts:
        if getattr(graft, "sink", None) is not None:
            return True
    return False


def decoder_of(host_model, layer_index):
    """Return the module that holds the decoder layer list, or None when it is host_model."""
    list_name, _ = find_decoder_layers(host_model, layer_index)
    if "." not in list_name:
        return None
    return host_model.get_submodule(list_name.rsplit(".", 1)[0])


def follow_host_cache(follow, cache, grafts):
    """Keep the part caches in step with this host cache object.

    Beam search calls cache.reorder_cache(beam_idx) on the host cache
    between steps. We wrap that method on this cache object, so the part
    caches get the same reorder.
    """
    follow["cache"] = weakref.ref(cache)
    if getattr(cache, "wsp_reorders_parts", False):
        return
    host_reorder = cache.reorder_cache
    this_cache = weakref.ref(cache)

    def reorder_cache(beam_idx):
        host_reorder(beam_idx)
        # The part caches follow only one host cache at a time.
        if follow["cache"] is None or follow["cache"]() is not this_cache():
            return
        for graft in grafts:
            part_cache = getattr(getattr(graft, "part", None), "cache", None)
            if part_cache is not None:
                part_cache.reorder_cache(beam_idx)

    cache.reorder_cache = reorder_cache
    cache.wsp_reorders_parts = True


def forward_args(module, args, kwargs, names):
    """Return a dict of the named arguments of a forward call, given by keyword or by position.

    A name the call does not give maps to None. The positions differ
    between model classes (Gemma 3 puts pixel_values before
    attention_mask), so we bind the args to the forward signature instead
    of reading fixed indices.
    """
    found = {}
    for name in names:
        found[name] = kwargs.get(name)
    if len(args) == 0:
        return found
    try:
        bound = inspect.signature(module.forward).bind_partial(*args, **kwargs)
    except TypeError:
        return found
    for name in names:
        if name in bound.arguments:
            found[name] = bound.arguments[name]
    return found


def attention_mask_arg(module, args, kwargs):
    """Return the attention_mask of a forward call, given by keyword or by position."""
    return forward_args(module, args, kwargs, ["attention_mask"])["attention_mask"]


def host_cache_length(past_key_values, cache_position):
    """Return the host cache length before a forward call.

    It is 0 when there is no cache or the cache is empty. We ask the cache
    first, because that needs no device sync. cache_position is the
    fallback for a cache object without get_seq_length.
    """
    if past_key_values is None:
        return 0
    if hasattr(past_key_values, "get_seq_length"):
        return int(past_key_values.get_seq_length())
    if cache_position is not None and cache_position.numel() > 0:
        return int(cache_position[0])
    return 0


def disable_cache(model):
    """Set use_cache False on the model config and generation config.

    Grafted models generate with a KV cache by default. Call this to turn
    it off, for example to compare against the uncached path.
    """
    config = getattr(model, "config", None)
    if config is not None:
        config.use_cache = False
    generation_config = getattr(model, "generation_config", None)
    if generation_config is not None:
        generation_config.use_cache = False


def glue_parameters(host_model):
    """Return the w_in and w_out parameters of every graft, and the RMS scale when it is on.

    w_in can be any module (a trained stitch, for example), so all of its
    parameters count as glue and train during heal.
    """
    params = []
    for graft in host_model.wsp_grafts:
        for param in graft.w_in.parameters():
            params.append(param)
        for param in graft.w_out.parameters():
            params.append(param)
        rms_scale = getattr(graft, "rms_scale", None)
        if rms_scale is not None:
            params.append(rms_scale)
    return params


def part_lora_parameters(host_model):
    """Return the LoRA factors inside every graft part (empty when no part has a LoRA)."""
    from wsp.lora import LoRALinear

    params = []
    for graft in host_model.wsp_grafts:
        part = getattr(graft, "part", None)
        if part is None:
            continue
        for module in part.modules():
            if isinstance(module, LoRALinear):
                params.append(module.lora_A)
                params.append(module.lora_B)
    return params


def freeze_all_but_glue(host_model):
    """Freeze everything, then unfreeze the glue and any part LoRA."""
    for param in host_model.parameters():
        param.requires_grad_(False)
    for param in glue_parameters(host_model):
        param.requires_grad_(True)
    for param in part_lora_parameters(host_model):
        param.requires_grad_(True)
