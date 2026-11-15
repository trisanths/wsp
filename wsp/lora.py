"""LoRA on the host: the regular fine-tuning control.

A graft must beat a LoRA on the host that trains the same number of
parameters, with the same heal loop and data. This module adds that LoRA.

Each target nn.Linear in the host decoder layers becomes a LoRALinear:

    y = base(x) + (alpha / rank) * B(A(x))

A starts from a Kaiming uniform init and B starts at zero, so a fresh LoRA
leaves the host unchanged. The base weight is frozen. A and B are float32,
the same as graft glue.
"""

import math

import torch
from torch import nn

from wsp.parts import find_decoder_layers

# qkv_proj and gate_up_proj are the fused projections of Phi-3 style models.
# Without them, a LoRA on such a host would skip q, k, v, gate, and up.
DEFAULT_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
                   "qkv_proj", "gate_up_proj")


class LoRALinear(nn.Module):
    """A frozen nn.Linear plus a trainable rank-r update B @ A."""

    def __init__(self, base, rank, alpha=None):
        super().__init__()
        if rank < 1:
            raise ValueError(f"LoRA rank must be at least 1, got {rank}")
        if alpha is None:
            alpha = default_alpha(rank)
        self.base = base
        self.rank = rank
        self.alpha = alpha
        self.scale = alpha / rank
        for param in self.base.parameters():
            param.requires_grad_(False)
        device = base.weight.device
        self.lora_A = nn.Parameter(torch.empty(rank, base.in_features, dtype=torch.float32, device=device))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, rank, dtype=torch.float32, device=device))
        # The same init as nn.Linear and PEFT.
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    @property
    def weight(self):
        # Some attention code reads q_proj.weight.dtype, so we pass it through.
        return self.base.weight

    @property
    def bias(self):
        return self.base.bias

    def forward(self, x):
        y = self.base(x)
        z = torch.nn.functional.linear(x.to(torch.float32), self.lora_A)
        z = torch.nn.functional.linear(z, self.lora_B)
        return y + (self.scale * z).to(y.dtype)

    @torch.no_grad()
    def merged_weight(self):
        """Return base.weight + scale * B @ A in the base weight dtype."""
        w = self.base.weight
        delta = self.scale * (self.lora_B.to(w.device) @ self.lora_A.to(w.device))
        return (w.to(torch.float32) + delta).to(w.dtype)


def default_alpha(rank):
    """alpha = 2 * rank, so the scale is 2 for every rank."""
    return 2 * rank


def target_names(targets):
    """Return targets as a tuple of names. One name as a string is allowed."""
    if targets is None:
        return DEFAULT_TARGETS
    if isinstance(targets, str):
        return (targets,)
    return tuple(targets)


def target_linears(model, targets=DEFAULT_TARGETS, layers=None):
    """Return (parent, child_name, module) for each target linear in the decoder layers.

    A target is an nn.Linear or a LoRALinear whose own name is in targets.
    layers: None for every decoder layer, or a list of layer indices.
    """
    targets = target_names(targets)
    _, decoder_layers = find_decoder_layers(model, 0)
    wanted = None
    if layers is not None:
        wanted = set()
        for index in layers:
            if index < 0:
                index += len(decoder_layers)
            if index < 0 or index >= len(decoder_layers):
                raise ValueError(f"LoRA layer {index} is out of range for {len(decoder_layers)} layers")
            wanted.add(index)

    found = []
    for index in range(len(decoder_layers)):
        if wanted is not None and index not in wanted:
            continue
        layer = decoder_layers[index]
        for name, module in layer.named_modules():
            if name == "":
                continue
            parts = name.split(".")
            child_name = parts[-1]
            if child_name not in targets:
                continue
            if not isinstance(module, (nn.Linear, LoRALinear)):
                continue
            parent = layer
            for part in parts[:-1]:
                parent = getattr(parent, part)
            found.append((parent, child_name, module))
    return found


def params_per_rank(model, targets=DEFAULT_TARGETS, layers=None):
    """Return the LoRA parameter count for rank 1: the sum of in + out over the targets."""
    total = 0
    for _, _, module in target_linears(model, targets, layers):
        if isinstance(module, LoRALinear):
            module = module.base
        total += module.in_features + module.out_features
    return total


def apply_lora(model, rank, alpha=None, targets=DEFAULT_TARGETS, layers=None):
    """Replace each target linear in the decoder layers with a LoRALinear.

    Returns the list of trainable LoRA parameters (A and B of each module).
    Linears that already hold a LoRA are left alone.
    """
    found = target_linears(model, targets, layers)
    params = []
    for parent, child_name, module in found:
        if isinstance(module, LoRALinear):
            continue
        wrapped = LoRALinear(module, rank, alpha)
        setattr(parent, child_name, wrapped)
        params.append(wrapped.lora_A)
        params.append(wrapped.lora_B)
    if len(params) == 0:
        raise ValueError(f"no LoRA targets {list(target_names(targets))} found in the decoder layers")
    return params


def lora_param_count(model_config, rank, targets=DEFAULT_TARGETS, layers=None):
    """Return the LoRA parameter count for a model config, without loading weights.

    The model is built on the meta device, so no memory is used for weights.
    """
    from wsp.parts import _build_meta_model

    model = _build_meta_model(model_config)
    return rank * params_per_rank(model, targets, layers)


def rank_for_budget(model, target_params, targets=DEFAULT_TARGETS, layers=None):
    """Return the largest rank whose LoRA parameter count is <= target_params, at least 1."""
    per_rank = params_per_rank(model, targets, layers)
    if per_rank == 0:
        raise ValueError(f"no LoRA targets {list(target_names(targets))} found in the decoder layers")
    rank = int(target_params) // per_rank
    if rank < 1:
        rank = 1
    return rank


def graft_budget(grafts):
    """Return the glue, part, and part LoRA parameter count of a list of graft metas.

    grafts is report["grafts"] from compile_report.json, or the list in
    grafts.json. A LoRA control with this budget matches the graft run.
    """
    total = 0
    for meta in grafts:
        total += meta["glue_params"] + meta["part_params"] + meta.get("part_lora_params", 0)
    return total


# Linear names inside a donor part that get a part LoRA: the host targets
# plus the Phi-2 names (dense, fc1, fc2). Every linear inside an experts
# list also gets one, whatever its name (Mixtral uses w1, w2, w3). Routers
# (gate, shared_expert_gate) do not.
PART_TARGETS = DEFAULT_TARGETS + ("dense", "fc1", "fc2")


def part_linears(part):
    """Return (parent, child_name, module) for each nn.Linear in a donor part that takes a part LoRA."""
    found = []
    for name, module in part.named_modules():
        if name == "" or not isinstance(module, nn.Linear):
            continue
        parts = name.split(".")
        parent = part
        for piece in parts[:-1]:
            parent = getattr(parent, piece)
        if isinstance(parent, LoRALinear):
            # The base of a LoRA that is already there.
            continue
        in_experts = "experts" in parts[:-1]
        if parts[-1] not in PART_TARGETS and not in_experts:
            continue
        found.append((parent, parts[-1], module))
    return found


def apply_part_lora(part, rank, alpha=None):
    """Wrap the linears of a donor part in LoRALinear. Return the new LoRA parameters.

    B starts at zero, so the part gives the same output as before. The
    donor weights stay frozen.
    """
    found = part_linears(part)
    if len(found) == 0:
        raise ValueError(f"no linear layers for a part LoRA found in {type(part).__name__}")
    params = []
    for parent, child_name, module in found:
        wrapped = LoRALinear(module, rank, alpha)
        setattr(parent, child_name, wrapped)
        params.append(wrapped.lora_A)
        params.append(wrapped.lora_B)
    return params


def lora_modules(model):
    """Return (name, module) for every LoRALinear on the host.

    Part LoRAs inside grafts (under wsp_grafts) are left out. They are saved
    with the graft weights, not with the host LoRA.
    """
    found = []
    for name, module in model.named_modules():
        if name == "wsp_grafts" or name.startswith("wsp_grafts."):
            continue
        if isinstance(module, LoRALinear):
            found.append((name, module))
    return found


def lora_state_dict(model):
    """Return the LoRA tensors as {module_name.lora_A: tensor, module_name.lora_B: tensor} on CPU."""
    state = {}
    for name, module in lora_modules(model):
        state[name + ".lora_A"] = module.lora_A.detach().cpu().contiguous()
        state[name + ".lora_B"] = module.lora_B.detach().cpu().contiguous()
    return state


@torch.no_grad()
def load_lora_state(model, state):
    """Copy LoRA tensors from state into the model's LoRALinear modules.

    The keys must match the model's LoRA modules exactly.
    """
    expected = set()
    for name, module in lora_modules(model):
        for suffix in ["lora_A", "lora_B"]:
            key = name + "." + suffix
            expected.add(key)
            if key not in state:
                raise KeyError(f"LoRA state is missing {key}")
            param = getattr(module, suffix)
            value = state[key]
            if tuple(value.shape) != tuple(param.shape):
                raise ValueError(f"{key}: saved shape {tuple(value.shape)} vs model {tuple(param.shape)}")
            param.copy_(value.to(device=param.device, dtype=param.dtype))
    extra = []
    for key in state:
        if key not in expected:
            extra.append(key)
    if len(extra) > 0:
        raise KeyError(f"LoRA state has keys the model does not: {sorted(extra)[:5]}")


@torch.no_grad()
def merge_lora(model):
    """Fold scale * B @ A into each base weight and put the plain nn.Linear back.

    After this the model has no LoRA modules, so save_pretrained writes a
    normal Hugging Face model that vLLM can serve.
    """
    replace = []
    for name, module in lora_modules(model):
        replace.append((name, module))
    for name, module in replace:
        module.base.weight.copy_(module.merged_weight())
        parts = name.split(".")
        parent = model
        for part in parts[:-1]:
            parent = getattr(parent, part)
        setattr(parent, parts[-1], module.base)
    return model
