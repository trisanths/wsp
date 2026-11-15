"""Grafts: attach a donor part to a host model through trainable glue.

A graft reads the host hidden state, maps it into the donor's width with
w_in, runs the frozen donor part, maps the result back with w_out, and adds
it to the hidden state. w_out starts at zero, so a fresh graft changes
nothing. Only w_in and w_out train.
"""

import inspect

import torch
from torch import nn

from wsp.parts import AttnHeadsPart, find_decoder_layers


class Graft(nn.Module):
    """h + w_out(part(w_in(h))), with float32 glue and a frozen part.

    pad_mask holds the host's 2D padding mask (batch, seq) for the current
    forward call, or None. attach_grafts sets it before each host forward.
    Only an AttnHeadsPart uses it, because only attention mixes positions.
    """

    def __init__(self, d_host, d_donor, part):
        super().__init__()
        self.w_in = nn.Linear(d_host, d_donor, bias=False)
        self.w_out = nn.Linear(d_donor, d_host, bias=False)
        nn.init.zeros_(self.w_out.weight)
        self.part = part
        self.pad_mask = None

    def forward(self, h):
        part_dtype = next(self.part.parameters()).dtype
        x = self.w_in(h.to(torch.float32))
        if isinstance(self.part, AttnHeadsPart):
            x = self.part(x.to(part_dtype), attention_mask=self.pad_mask)
        else:
            x = self.part(x.to(part_dtype))
        delta = self.w_out(x.to(torch.float32))
        return h + delta.to(h.dtype)


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

    A pre-hook on host_model copies the attention_mask of each forward call
    to every graft's pad_mask. An attn_heads part has no KV cache, so when
    any graft holds one, this function turns use_cache off on the model.
    """
    handles = []
    grafts = []
    for layer_index, graft in placements:
        _, layers = find_decoder_layers(host_model, layer_index)

        def hook(module, inputs, output, graft=graft):
            return replace_hidden(output, graft)

        handles.append(layers[layer_index].register_forward_hook(hook))
        grafts.append(graft)
    host_model.wsp_grafts = nn.ModuleList(grafts)

    def store_pad_mask(module, args, kwargs):
        mask = attention_mask_arg(module, args, kwargs)
        # Only a 2D (batch, seq) mask is a padding mask.
        if mask is not None and mask.dim() != 2:
            mask = None
        for graft in grafts:
            graft.pad_mask = mask

    handles.append(host_model.register_forward_pre_hook(store_pad_mask, with_kwargs=True))

    has_attn = False
    for graft in grafts:
        if isinstance(getattr(graft, "part", None), AttnHeadsPart):
            has_attn = True
    if has_attn:
        disable_cache(host_model)
    return handles


def attention_mask_arg(module, args, kwargs):
    """Return the attention_mask of a forward call, given by keyword or by position.

    The position of attention_mask differs between model classes (Gemma 3
    puts pixel_values before it), so we bind the args to the forward
    signature instead of reading a fixed index.
    """
    if "attention_mask" in kwargs:
        return kwargs["attention_mask"]
    if len(args) == 0:
        return None
    try:
        bound = inspect.signature(module.forward).bind_partial(*args, **kwargs)
    except TypeError:
        return None
    return bound.arguments.get("attention_mask")


def disable_cache(model):
    """Set use_cache False on the model config and generation config."""
    config = getattr(model, "config", None)
    if config is not None:
        config.use_cache = False
    generation_config = getattr(model, "generation_config", None)
    if generation_config is not None:
        generation_config.use_cache = False


def glue_parameters(host_model):
    """Return the w_in and w_out parameters of every graft."""
    params = []
    for graft in host_model.wsp_grafts:
        params.append(graft.w_in.weight)
        params.append(graft.w_out.weight)
    return params


def freeze_all_but_glue(host_model):
    """Freeze everything, then unfreeze the glue."""
    for param in host_model.parameters():
        param.requires_grad_(False)
    for param in glue_parameters(host_model):
        param.requires_grad_(True)
