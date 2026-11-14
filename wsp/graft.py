"""Grafts: attach a donor part to a host model through trainable glue.

A graft reads the host hidden state, maps it into the donor's width with
w_in, runs the frozen donor part, maps the result back with w_out, and adds
it to the hidden state. w_out starts at zero, so a fresh graft changes
nothing. Only w_in and w_out train.
"""

import torch
from torch import nn

from wsp.parts import find_decoder_layers


class Graft(nn.Module):
    """h + w_out(part(w_in(h))), with float32 glue and a frozen part."""

    def __init__(self, d_host, d_donor, part):
        super().__init__()
        self.w_in = nn.Linear(d_host, d_donor, bias=False)
        self.w_out = nn.Linear(d_donor, d_host, bias=False)
        nn.init.zeros_(self.w_out.weight)
        self.part = part

    def forward(self, h):
        part_dtype = next(self.part.parameters()).dtype
        x = self.w_in(h.to(torch.float32))
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
    return handles


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
