"""Token gates for several grafts that write into one host.

When several block grafts write into the host, each graft's change to the
hidden state is multiplied by its own gate:

    h = h + g_i(h) * delta_i        g_i(h) = sigmoid(w_i . rms(h) + b_i)

rms(h) is h divided by its root mean square, so the gate reads the
direction of the host state and not its size. The gate reads the same
host state its graft reads, at the graft's layer, and it is computed per
token. So the model can send math tokens through the math block and code
tokens through the code block.

w_i starts at zero and b_i at init_bias (default 0), so every gate starts
at sigmoid(init_bias), 0.5 by default. A fresh graft has w_out at zero,
so its delta is zero and the gated model equals the host for any gate
start.

The gates live in host_model.wsp_gates, next to host_model.wsp_grafts.
They are not inside the grafts, so the graft state that
wsp.compiler.save_grafts writes and wsp.load reads does not change. Each
gate attaches through two hooks: a pre-hook on the graft keeps the host
state, and a hook on the graft's w_out scales its output by the gate.
wsp/graft.py does not change.

save_gates writes gates.json and gate_weights.safetensors. wsp.load does
not read them: use wsp.fuse.load, which calls wsp.load and then
load_gates.
"""

import json
import os

import torch
from torch import nn


class TokenGate(nn.Module):
    """g(h) = sigmoid(weight . rms(h) + bias), one value per token, in float32."""

    def __init__(self, d_host, init_bias=0.0):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(d_host, dtype=torch.float32))
        self.bias = nn.Parameter(torch.full((), float(init_bias), dtype=torch.float32))
        self.init_bias = float(init_bias)
        # Running gate statistics over every real token of every call
        # since reset_stats, for reports. They stay tensors, so a forward
        # call does not wait for the GPU.
        self.stat_sum = None
        self.stat_count = 0

    def forward(self, h, real=None):
        """Return the gate for each token of h, shape (batch, seq, 1).

        real: None, or a (batch, seq) mask of the real tokens. Only real
        tokens count in the gate statistics, so left padding does not
        change a gate mean. The gate value itself does not depend on it.
        """
        x = h.to(torch.float32)
        x = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + 1e-6)
        weight = self.weight.to(device=x.device, dtype=torch.float32)
        bias = self.bias.to(device=x.device, dtype=torch.float32)
        logit = (x * weight).sum(dim=-1, keepdim=True) + bias
        gate = torch.sigmoid(logit)
        if real is None:
            total = gate.detach().sum()
            count = gate.numel()
        else:
            keep = (real != 0).to(device=gate.device, dtype=torch.float32).unsqueeze(-1)
            total = (gate.detach() * keep).sum()
            count = keep.sum()
        if self.stat_sum is None:
            self.stat_sum = total
        else:
            self.stat_sum = self.stat_sum + total
        self.stat_count = self.stat_count + count
        return gate

    def reset_stats(self):
        self.stat_sum = None
        self.stat_count = 0

    def mean(self):
        """Return the mean gate value since reset_stats, or None before any call."""
        if self.stat_sum is None or float(self.stat_count) == 0:
            return None
        return float(self.stat_sum) / float(self.stat_count)


def attach_gates(host_model, init_bias=0.0, which=None):
    """Put a TokenGate on each graft of host_model (or on the graft indices in which).

    Returns the list of gates. The gates go to host_model.wsp_gates, on the
    device of their graft's w_out. Each gate records the index of its graft
    in gate.graft_index.
    """
    if getattr(host_model, "wsp_grafts", None) is None:
        raise ValueError("attach_gates needs a model with grafts (host_model.wsp_grafts)")
    if getattr(host_model, "wsp_gates", None) is not None:
        raise ValueError("this model already has gates")
    if which is None:
        which = list(range(len(host_model.wsp_grafts)))
    gates = []
    handles = []
    for index in which:
        graft = host_model.wsp_grafts[index]
        d_host = graft_host_width(graft)
        gate = TokenGate(d_host, init_bias)
        gate.graft_index = index
        device = None
        for param in graft.w_out.parameters():
            device = param.device
            break
        if device is not None:
            gate.to(device)
        for handle in hook_gate(graft, gate):
            handles.append(handle)
        gates.append(gate)
    host_model.wsp_gates = nn.ModuleList(gates)
    # Plain attribute, not a module: the handles let detach_gates undo the hooks.
    host_model.__dict__["wsp_gate_handles"] = handles
    return gates


def graft_host_width(graft):
    """Return the host width a graft writes to (the output size of its w_out)."""
    w_out = graft.w_out
    if isinstance(w_out, nn.Linear):
        return w_out.out_features
    return w_out.up.out_features


def hook_gate(graft, gate):
    """Scale the graft's delta by the gate. Return the hook handles.

    The graft calls w_out once per forward, on the part output. The
    pre-hook keeps the host state h that the graft got, the w_out hook
    multiplies the delta by gate(h), and the graft's own forward hook drops
    h again so it does not outlive the call.
    """
    held = {"h": None}

    def keep_host_state(module, args):
        held["h"] = args[0]

    def scale_delta(module, args, output):
        h = held["h"]
        if h is None:
            return output
        g = gate(h, real_tokens(graft, h)).to(output.dtype)
        return output * g

    def drop_host_state(module, args, output):
        held["h"] = None

    handles = []
    handles.append(graft.register_forward_pre_hook(keep_host_state))
    handles.append(graft.w_out.register_forward_hook(scale_delta))
    handles.append(graft.register_forward_hook(drop_host_state))
    return handles


def real_tokens(graft, h):
    """Return the (batch, seq) mask of real tokens in h, from the graft's call context, or None.

    The graft's pad_mask covers the cached and the new positions, so the
    new tokens are its last seq columns.
    """
    mask = getattr(graft, "pad_mask", None)
    if mask is None or not torch.is_tensor(mask) or mask.dim() != 2 or h.dim() != 3:
        return None
    batch, seq = h.shape[0], h.shape[1]
    if mask.shape[0] != batch or mask.shape[1] < seq:
        return None
    return mask[:, mask.shape[1] - seq:]


def detach_gates(host_model):
    """Remove the gate hooks and host_model.wsp_gates. The grafts run ungated again."""
    handles = host_model.__dict__.get("wsp_gate_handles", [])
    for handle in handles:
        handle.remove()
    host_model.__dict__["wsp_gate_handles"] = []
    if getattr(host_model, "wsp_gates", None) is not None:
        del host_model.wsp_gates


def gate_parameters(host_model):
    """Return the weight and bias of every gate (empty when the model has no gates)."""
    params = []
    gates = getattr(host_model, "wsp_gates", None)
    if gates is None:
        return params
    for gate in gates:
        params.append(gate.weight)
        params.append(gate.bias)
    return params


def gate_means(host_model, reset=False):
    """Return the mean gate value since the last reset, one per gate (None for a gate not yet run).

    reset=True starts new statistics after reading them.
    """
    out = []
    gates = getattr(host_model, "wsp_gates", None)
    if gates is None:
        return out
    for gate in gates:
        out.append(gate.mean())
        if reset:
            gate.reset_stats()
    return out


def gate_meta(host_model):
    """Return one dict per gate: its graft index, host width, start bias, and current bias."""
    rows = []
    for gate in host_model.wsp_gates:
        rows.append({"graft": gate.graft_index, "d_host": gate.weight.shape[0], "init_bias": gate.init_bias,
                     "bias": float(gate.bias.detach()), "weight_norm": float(gate.weight.detach().norm())})
    return rows


def save_gates(host_model, output_dir):
    """Write gates.json and gate_weights.safetensors for the model's gates."""
    from safetensors.torch import save_file

    state = {}
    for number, gate in enumerate(host_model.wsp_gates):
        state[f"{number}.weight"] = gate.weight.detach().float().cpu().contiguous()
        state[f"{number}.bias"] = gate.bias.detach().float().cpu().contiguous()
    save_file(state, os.path.join(output_dir, "gate_weights.safetensors"))
    with open(os.path.join(output_dir, "gates.json"), "w") as f:
        json.dump(gate_meta(host_model), f, indent=2)


def load_gates(host_model, path):
    """Attach the gates saved in path to host_model's grafts and load their weights.

    Returns the list of gates, or an empty list when path has no gates.json.
    """
    from safetensors.torch import load_file

    meta_path = os.path.join(path, "gates.json")
    if not os.path.exists(meta_path):
        return []
    with open(meta_path) as f:
        rows = json.load(f)
    which = []
    for row in rows:
        which.append(row["graft"])
    gates = attach_gates(host_model, which=which)
    state = load_file(os.path.join(path, "gate_weights.safetensors"))
    with torch.no_grad():
        for number, gate in enumerate(gates):
            gate.weight.copy_(state[f"{number}.weight"].to(gate.weight.device))
            gate.bias.copy_(state[f"{number}.bias"].to(gate.bias.device))
            gate.init_bias = rows[number].get("init_bias", 0.0)
    return gates
