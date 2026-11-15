"""Skill MoE: one network whose MLPs route over the same layer's MLP from several specialists.

build_skill_moe takes a base model and expert models. In every decoder
layer of the base, the MLP becomes a SkillMoE that holds:

  expert 0     the base MLP, with the base's own pre-MLP norm
  expert 1..n  the same layer's MLP from each expert, with the expert's
               own pre-MLP norm (and post-MLP norm, Gemma style)
  router       Linear(hidden, n_experts), no bias, plus a separate bias
               vector, on the base's post-attention normed hidden state

The decoder layer's own pre-MLP norm (and post-MLP norm) become
nn.Identity, and the SkillMoE applies each expert's own norms to the
residual stream. So each expert MLP sees exactly the input it was trained
on, given the residual stream.

Routing: probs = softmax(router(x) + bias) over experts.
  top_k == n    soft mixture, out = sum_e probs_e * expert_e
  top_k == 1    Switch style, out = probs_top * expert_top (not renormalized,
                so the router gets a gradient from the LM loss)
  1 < top_k < n Mixtral style, the top_k probs renormalized to sum to 1

Init "base": bias [base_logit, 0, ..., 0], the base row of the router
weight zero, and the expert rows small random noise (std router_noise /
sqrt(hidden), so an expert logit has std router_noise times the RMS of
the normed state). The base gets prob about 1 / (1 + (n - 1) *
exp(-base_logit)). With the default base_logit 16 that is 1 - 2.3e-7 for
3 experts. So the MLP output of a fresh model is off by about 2.3e-7
times (expert output - base output), or times the base output under
top_k 1. On the tiny test models the logits match the base to 2e-7 in
float32 for top_k 1, 2, and 3. Init "uniform": bias zero (a plain soft
mixture from the start), with the same noise.

The noise breaks the tie between experts. With all-zero expert rows,
every expert has the same logit for every token, top-k ties always pick
the lowest index, and under top_k 1 the experts get the same gradient
forever, so expert 1 would win every slot that the base gives up and the
other experts would get no signal. router_noise=0 gives the all-zero
init back.

The router gradient at this init is scaled by about exp(-base_logit), the
same factor as the gap. Adam divides by the gradient scale, so the router
still trains, but only with a small Adam eps (ROUTER_ADAM_EPS); the
default eps 1e-8 swamps the tiny gradient. lower_base_logit() is the
other option: it moves the base bias down by hand, for an opening
schedule.

Cross-family experts: stitched(model) wraps a donor of any family. Its
expert in each layer is w_out(post_norm(mlp(norm(w_in(x))))), where x is
the base's normed state, norm and mlp are the donor's own modules from
the mapped donor layer, and w_out starts at zero (wsp.graft style). So a
fresh stitched expert outputs zero.

Trainable: router weights and biases, and the stitches. Expert and base
weights are frozen. lora_rank adds wsp.lora on the base attention.

save_skill_moe writes manifest.json (base id, expert ids, layer maps,
routing options, LoRA options) and skill_moe_weights.safetensors (routers,
stitches, LoRA). load_skill_moe loads the base by id and fetches only the
expert MLP tensors it needs with wsp.fetch.fetch_tensors (or from a local
model directory).
"""

import json
import os

import torch
from torch import nn

from wsp.parts import find_decoder_layers

BASE_LOGIT = 16.0
ROUTER_NOISE = 0.1
ROUTER_ADAM_EPS = 1e-16
ATTN_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj")
MANIFEST = "manifest.json"
WEIGHTS = "skill_moe_weights.safetensors"
FORMAT = "skill_moe/1"


# ---------------------------------------------------------------- experts


class PlainExpert(nn.Module):
    """post_norm(mlp(pre_norm(h))) for an expert of the base's own family, post_norm optional."""

    def __init__(self, pre_norm, mlp, post_norm=None):
        super().__init__()
        self.pre_norm = pre_norm
        self.mlp = mlp
        self.post_norm = post_norm

    def forward(self, h, x_base):
        out = self.mlp(self.pre_norm(h))
        if self.post_norm is not None:
            out = self.post_norm(out)
        return out


class BaseExpert(PlainExpert):
    """The base MLP. Its pre-norm output is the router input, so the MoE passes it in."""

    def forward(self, h, x_base):
        out = self.mlp(x_base)
        if self.post_norm is not None:
            out = self.post_norm(out)
        return out


class StitchedExpert(nn.Module):
    """w_out(post_norm(mlp(norm(w_in(x_base))))), with float32 stitches and a frozen donor MLP.

    w_out starts at zero, so a fresh stitched expert outputs zero.
    """

    def __init__(self, w_in, norm, mlp, w_out, post_norm=None):
        super().__init__()
        self.w_in = w_in
        self.norm = norm
        self.mlp = mlp
        self.post_norm = post_norm
        self.w_out = w_out

    def forward(self, h, x_base):
        dtype = donor_dtype(self.mlp)
        x = self.w_in(x_base.to(torch.float32))
        out = self.mlp(self.norm(x.to(dtype)))
        if self.post_norm is not None:
            out = self.post_norm(out)
        return self.w_out(out.to(torch.float32)).to(h.dtype)


def donor_dtype(module):
    for param in module.parameters():
        return param.dtype
    return torch.float32


def make_stitch(config, d_in, d_out):
    """Return w_in for a stitched expert: a wsp.stitch kind ("linear", "lowrank", "mlp")."""
    from wsp.stitch import build_stitch

    if config is None:
        config = {"kind": "linear"}
    if config.get("kind") == "multi":
        raise ValueError("a stitched expert reads one layer; the multi stitch kind is not supported")
    return build_stitch(config, d_in, d_out)


def make_w_out(d_donor, d_host):
    w_out = nn.Linear(d_donor, d_host, bias=False).float()
    nn.init.zeros_(w_out.weight)
    return w_out


# ---------------------------------------------------------------- the MoE layer


class SkillMoE(nn.Module):
    """Route each token over experts. Expert 0 is the base and owns the router input norm."""

    def __init__(self, experts, hidden_size, top_k, base_logit=BASE_LOGIT, router_init="base",
                 load_balance=False, router_noise=ROUTER_NOISE):
        super().__init__()
        n = len(experts)
        if n < 2:
            raise ValueError("a SkillMoE needs the base and at least one expert")
        if top_k < 1 or top_k > n:
            raise ValueError(f"top_k must be between 1 and {n}, got {top_k}")
        self.experts = nn.ModuleList(experts)
        self.top_k = int(top_k)
        self.load_balance = bool(load_balance)
        self.router = nn.Linear(hidden_size, n, bias=False).float()
        nn.init.zeros_(self.router.weight)
        if router_noise > 0:
            # Noise on the expert rows only, so the experts do not tie (see the module doc).
            with torch.no_grad():
                self.router.weight[1:].normal_(0.0, float(router_noise) / hidden_size ** 0.5)
        bias = torch.zeros(n, dtype=torch.float32)
        if router_init == "base":
            bias[0] = float(base_logit)
        elif router_init != "uniform":
            raise ValueError(f"router_init must be 'base' or 'uniform', got {router_init!r}")
        self.router_bias = nn.Parameter(bias)
        self.aux_loss = None
        self.track = False
        self.register_buffer("route_counts", torch.zeros(n, dtype=torch.float64), persistent=False)
        self.register_buffer("prob_sums", torch.zeros(n, dtype=torch.float64), persistent=False)

    def n_experts(self):
        return len(self.experts)

    def route(self, x):
        """Return (probs, top_idx, top_w) for flat normed states x of shape (tokens, hidden)."""
        with torch.autocast(device_type=x.device.type, enabled=False):
            weight = self.router.weight
            logits = torch.nn.functional.linear(x.to(weight.dtype), weight) + self.router_bias
            probs = torch.softmax(logits, dim=-1)
        n = self.n_experts()
        if self.top_k == n:
            top_idx = None
            top_w = probs
        else:
            top_vals, top_idx = probs.topk(self.top_k, dim=-1)
            if self.top_k == 1:
                top_w = top_vals
            else:
                top_w = top_vals / top_vals.sum(dim=-1, keepdim=True)
        return probs, top_idx, top_w

    def assignments(self, probs, top_idx):
        """Return a (tokens, n) 0/1 float tensor: the experts each token was routed to.

        A soft mixture counts the argmax expert.
        """
        n = self.n_experts()
        if top_idx is None:
            top_idx = probs.argmax(dim=-1, keepdim=True)
        hits = torch.zeros(probs.shape[0], n, dtype=probs.dtype, device=probs.device)
        hits.scatter_(1, top_idx, 1.0)
        return hits

    def balance_loss(self, probs, top_idx):
        """Switch Transformer load-balance loss: n * sum_e f_e * P_e.

        f_e is the share of routing slots that went to expert e (argmax for a
        soft mixture) and P_e the mean router prob. It is 1 when balanced.
        Padding tokens count too.
        """
        n = self.n_experts()
        hits = self.assignments(probs, top_idx)
        share = hits.sum(dim=0) / hits.sum().clamp(min=1.0)
        mean_prob = probs.mean(dim=0)
        return n * (share.detach() * mean_prob).sum()

    def forward(self, h):
        shape = h.shape
        flat_h = h.reshape(-1, shape[-1])
        x_base = self.experts[0].pre_norm(flat_h)
        probs, top_idx, top_w = self.route(x_base)

        if self.load_balance and torch.is_grad_enabled():
            self.aux_loss = self.balance_loss(probs, top_idx)
        else:
            self.aux_loss = None
        if self.track:
            with torch.no_grad():
                hits = self.assignments(probs, top_idx)
                self.route_counts += hits.sum(dim=0).to(torch.float64)
                self.prob_sums += probs.sum(dim=0).to(torch.float64)

        out = None
        for index in range(self.n_experts()):
            if top_idx is None:
                rows = None
                weight = top_w[:, index]
            else:
                hit = top_idx == index
                rows = hit.any(dim=-1).nonzero().squeeze(-1)
                if rows.numel() == 0:
                    continue
                weight = (top_w * hit.to(top_w.dtype)).sum(dim=-1)[rows]
            expert = self.experts[index]
            if rows is None:
                y = expert(flat_h, x_base)
            else:
                y = expert(flat_h[rows], x_base[rows])
            if out is None:
                out = torch.zeros(flat_h.shape[0], y.shape[-1], dtype=torch.float32, device=flat_h.device)
            contribution = y.to(torch.float32) * weight.unsqueeze(-1)
            if rows is None:
                out = out + contribution
            else:
                out = out.index_add(0, rows, contribution)
        return out.to(h.dtype).reshape(shape[:-1] + (out.shape[-1],))


# ---------------------------------------------------------------- building


def layer_norm_names(layer):
    """Return (pre_name, post_name) of a decoder layer's MLP norms. post_name is None without one."""
    if hasattr(layer, "pre_feedforward_layernorm"):
        pre_name = "pre_feedforward_layernorm"
    elif hasattr(layer, "post_attention_layernorm"):
        pre_name = "post_attention_layernorm"
    else:
        raise ValueError(f"{type(layer).__name__} has no pre-MLP norm this version knows "
                         "(post_attention_layernorm or pre_feedforward_layernorm)")
    post_name = None
    if hasattr(layer, "post_feedforward_layernorm"):
        post_name = "post_feedforward_layernorm"
    return pre_name, post_name


def decoder_layers(model):
    _, layers = find_decoder_layers(model, 0)
    return layers


def layer_parts(model, index):
    """Return (pre_norm, mlp, post_norm) of one decoder layer, the model's own modules."""
    layer = decoder_layers(model)[index]
    mlp = getattr(layer, "mlp", None)
    if mlp is None:
        raise ValueError(f"layer {index} of {type(model).__name__} has no mlp module")
    if isinstance(mlp, SkillMoE):
        raise ValueError(f"layer {index} is already a SkillMoE")
    sparse = hasattr(mlp, "experts")
    if hasattr(mlp, "gate") and not hasattr(mlp, "gate_proj"):
        sparse = True
    if sparse:
        raise ValueError(f"layer {index} mlp is a sparse MoE block ({type(mlp).__name__}); "
                         "this first version takes dense MLPs only")
    pre_name, post_name = layer_norm_names(layer)
    post_norm = None
    if post_name is not None:
        post_norm = getattr(layer, post_name)
    return getattr(layer, pre_name), mlp, post_norm


def text_config(model):
    config = model.config
    if hasattr(config, "get_text_config"):
        return config.get_text_config()
    return config


def model_id(model):
    config = getattr(model, "config", None)
    name = getattr(config, "_name_or_path", None) or getattr(model, "name_or_path", None)
    return name or None


def stitched(model, kind="linear", rank=None, hidden=None, layer_map=None, model_id=None):
    """Mark a donor of another family (or width) as a stitched expert for build_skill_moe."""
    stitch = {"kind": kind}
    if rank is not None:
        stitch["rank"] = int(rank)
    if hidden is not None:
        stitch["hidden"] = int(hidden)
    return {"model": model, "stitch": stitch, "layer_map": layer_map, "id": model_id}


def depth_map(n_base, n_donor):
    """Map base layer i to the donor layer at the same relative depth."""
    out = []
    for index in range(n_base):
        if n_base == 1:
            out.append(0)
        else:
            out.append(int(round(index * (n_donor - 1) / (n_base - 1))))
    return out


def check_plain_expert(base, expert, position):
    """Raise unless an expert has the base's architecture, hidden size, and depth."""
    base_config = text_config(base)
    expert_config = text_config(expert)
    problems = []
    if getattr(base_config, "model_type", None) != getattr(expert_config, "model_type", None):
        problems.append(f"model_type {expert_config.model_type!r} vs base {base_config.model_type!r}")
    if base_config.hidden_size != expert_config.hidden_size:
        problems.append(f"hidden_size {expert_config.hidden_size} vs base {base_config.hidden_size}")
    n_base = len(decoder_layers(base))
    n_expert = len(decoder_layers(expert))
    if n_base != n_expert:
        problems.append(f"{n_expert} layers vs base {n_base}")
    if len(problems) > 0:
        raise ValueError(f"expert {position} does not match the base ({'; '.join(problems)}); a plain expert "
                         "needs the same architecture, hidden size, and depth. Wrap it with "
                         "wsp.skill_moe.stitched(model) to use it as a stitched expert.")


def normalize_specs(base_model, expert_models, expert_ids):
    """Return one spec dict per expert: model, id, kind, stitch, layer_map."""
    n_base = len(decoder_layers(base_model))
    specs = []
    for position, entry in enumerate(expert_models):
        if isinstance(entry, dict):
            model = entry["model"]
            stitch = entry.get("stitch") or {"kind": "linear"}
            kind = "stitched"
            layer_map = entry.get("layer_map")
            if layer_map is None:
                layer_map = depth_map(n_base, len(decoder_layers(model)))
            name = entry.get("id")
        else:
            model = entry
            check_plain_expert(base_model, model, position + 1)
            stitch = None
            kind = "plain"
            layer_map = list(range(n_base))
            name = None
        if len(layer_map) != n_base:
            raise ValueError(f"expert {position + 1} layer_map has {len(layer_map)} entries for {n_base} base layers")
        if expert_ids is not None:
            name = expert_ids[position]
        if name is None:
            name = model_id(model)
        specs.append({"model": model, "id": name, "kind": kind, "stitch": stitch, "layer_map": list(layer_map)})
    return specs


def build_skill_moe(base_model, expert_models, top_k, router_init="base", base_logit=BASE_LOGIT,
                    load_balance=False, lora_rank=None, lora_alpha=None, base_id=None, expert_ids=None,
                    router_noise=ROUTER_NOISE):
    """Turn base_model (in place) into a skill MoE over expert_models. Return base_model.

    expert_models: a list of models of the base's family and shape (plain
    experts), or stitched(model, ...) dicts for donors of another family.
    The experts' MLP modules move into the MoE by reference, so the expert
    models can be deleted afterwards.
    """
    specs = normalize_specs(base_model, expert_models, expert_ids)
    d_host = text_config(base_model).hidden_size
    layer_experts = []
    for index in range(len(decoder_layers(base_model))):
        experts = []
        for spec in specs:
            donor_layer = spec["layer_map"][index]
            pre_norm, mlp, post_norm = layer_parts(spec["model"], donor_layer)
            if spec["kind"] == "plain":
                experts.append(PlainExpert(pre_norm, mlp, post_norm))
            else:
                d_donor = text_config(spec["model"]).hidden_size
                w_in = make_stitch(spec["stitch"], d_host, d_donor)
                experts.append(StitchedExpert(w_in, pre_norm, mlp, make_w_out(d_donor, d_host), post_norm))
        layer_experts.append(experts)

    info = {
        "base_id": base_id or model_id(base_model),
        "experts": [],
        "top_k": int(top_k),
        "router_init": router_init,
        "base_logit": float(base_logit),
        "load_balance": bool(load_balance),
        "router_noise": float(router_noise),
        "lora": None,
    }
    for spec in specs:
        row = {"id": spec["id"], "kind": spec["kind"], "stitch": spec["stitch"], "layer_map": spec["layer_map"]}
        if spec["kind"] == "stitched":
            row["d_donor"] = text_config(spec["model"]).hidden_size
        info["experts"].append(row)
    assemble(base_model, layer_experts, info)
    if lora_rank is not None:
        add_lora(base_model, lora_rank, lora_alpha)
    return base_model


def assemble(base_model, layer_experts, info):
    """Put a SkillMoE in every decoder layer, freeze all but routers and stitches, and store info."""
    hidden = text_config(base_model).hidden_size
    layers = decoder_layers(base_model)
    if len(layer_experts) != len(layers):
        raise ValueError(f"{len(layer_experts)} expert lists for {len(layers)} layers")
    for param in base_model.parameters():
        param.requires_grad_(False)
    for index in range(len(layers)):
        layer = layers[index]
        pre_norm, mlp, post_norm = layer_parts(base_model, index)
        experts = [BaseExpert(pre_norm, mlp, post_norm)]
        for expert in layer_experts[index]:
            for param in expert.parameters():
                param.requires_grad_(False)
            experts.append(expert)
        noise = info.get("router_noise", 0.0)
        moe = SkillMoE(experts, hidden, info["top_k"], info["base_logit"], info["router_init"], info["load_balance"],
                       noise)
        for param in mlp.parameters():
            # The experts are there already; this moves the router and its buffers.
            moe.to(param.device)
            break
        pre_name, post_name = layer_norm_names(layer)
        setattr(layer, pre_name, nn.Identity())
        if post_name is not None:
            setattr(layer, post_name, nn.Identity())
        layer.mlp = moe
        for param in router_parameters_of(moe):
            param.requires_grad_(True)
        for param in stitch_parameters_of(moe):
            param.requires_grad_(True)
    base_model.wsp_skill_moe = info
    return base_model


def add_lora(model, rank, alpha=None, targets=ATTN_TARGETS):
    """Put a wsp.lora LoRA on the base attention (default). Return the new LoRA parameters."""
    from wsp.lora import apply_lora, default_alpha

    if alpha is None:
        alpha = default_alpha(rank)
    params = apply_lora(model, rank, alpha, targets=targets)
    info = getattr(model, "wsp_skill_moe", None)
    if info is not None:
        info["lora"] = {"rank": int(rank), "alpha": alpha, "targets": list(targets)}
    return params


# ---------------------------------------------------------------- access


def moe_layers(model):
    """Return the SkillMoE modules of a model, in layer order."""
    found = []
    for module in model.modules():
        if isinstance(module, SkillMoE):
            found.append(module)
    return found


def router_parameters_of(moe):
    return [moe.router.weight, moe.router_bias]


def stitch_parameters_of(moe):
    params = []
    for expert in moe.experts:
        if isinstance(expert, StitchedExpert):
            for param in expert.w_in.parameters():
                params.append(param)
            for param in expert.w_out.parameters():
                params.append(param)
    return params


def router_parameters(model):
    params = []
    for moe in moe_layers(model):
        params += router_parameters_of(moe)
    return params


def stitch_parameters(model):
    params = []
    for moe in moe_layers(model):
        params += stitch_parameters_of(moe)
    return params


def lora_parameters(model):
    from wsp.lora import lora_modules

    params = []
    for _, module in lora_modules(model):
        params.append(module.lora_A)
        params.append(module.lora_B)
    return params


def load_balance_loss(model):
    """Return the mean load-balance loss over layers from the last forward, or None."""
    total = None
    count = 0
    for moe in moe_layers(model):
        if moe.aux_loss is None:
            continue
        if total is None:
            total = moe.aux_loss
        else:
            total = total + moe.aux_loss.to(total.device)
        count += 1
    if total is None:
        return None
    return total / count


@torch.no_grad()
def lower_base_logit(model, amount):
    """Move every router's base bias down by amount (for an opening schedule)."""
    for moe in moe_layers(model):
        moe.router_bias[0] -= amount


def track_routing(model, on=True):
    """Start (or stop) counting routing choices. Starting also clears the counts."""
    for moe in moe_layers(model):
        moe.track = on
        if on:
            moe.route_counts.zero_()
            moe.prob_sums.zero_()


def routing_shares(model):
    """Return {"per_layer": [[share per expert]], "mean": [...], "mean_prob": [...]} from the counts.

    A share is the fraction of routing slots that went to the expert (the
    argmax expert for a soft mixture). mean_prob is the mean router prob.
    """
    per_layer = []
    per_layer_prob = []
    for moe in moe_layers(model):
        counts = moe.route_counts.double().cpu()
        probs = moe.prob_sums.double().cpu()
        slots = float(counts.sum())
        tokens = float(probs.sum())
        row = []
        prob_row = []
        for index in range(moe.n_experts()):
            if slots > 0:
                row.append(float(counts[index]) / slots)
                prob_row.append(float(probs[index]) / tokens)
            else:
                row.append(0.0)
                prob_row.append(0.0)
        per_layer.append(row)
        per_layer_prob.append(prob_row)
    mean = []
    mean_prob = []
    if len(per_layer) > 0:
        for index in range(len(per_layer[0])):
            total = 0.0
            total_prob = 0.0
            for layer_index in range(len(per_layer)):
                total += per_layer[layer_index][index]
                total_prob += per_layer_prob[layer_index][index]
            mean.append(total / len(per_layer))
            mean_prob.append(total_prob / len(per_layer))
    return {"per_layer": per_layer, "mean": mean, "mean_prob": mean_prob}


def count_params(model):
    """Return {"total", "active", "router", "stitch", "experts"} parameter counts.

    total counts every distinct parameter once. active is what one token
    runs: total minus all expert params plus, per layer, the top_k largest
    experts (all experts for a soft mixture). router counts router weights
    and biases, stitch the stitched experts' w_in and w_out, experts every
    expert but the base.
    """
    seen = set()
    total = 0
    for param in model.parameters():
        if id(param) in seen:
            continue
        seen.add(id(param))
        total += param.numel()
    active = total
    router = 0
    stitch = 0
    experts_total = 0
    for moe in moe_layers(model):
        sizes = []
        for index in range(moe.n_experts()):
            size = 0
            for param in moe.experts[index].parameters():
                size += param.numel()
            sizes.append(size)
            if index > 0:
                experts_total += size
        for param in router_parameters_of(moe):
            router += param.numel()
        for param in stitch_parameters_of(moe):
            stitch += param.numel()
        largest = sorted(sizes, reverse=True)
        active -= sum(sizes)
        active += sum(largest[:moe.top_k])
    return {"total": total, "active": active, "router": router, "stitch": stitch, "experts": experts_total}


# ---------------------------------------------------------------- save and load


def save_skill_moe(model, path, tokenizer=None):
    """Write manifest.json and skill_moe_weights.safetensors (routers, stitches, LoRA) to path."""
    from safetensors.torch import save_file

    from wsp.lora import lora_state_dict

    info = getattr(model, "wsp_skill_moe", None)
    if info is None:
        raise ValueError("this model has no skill MoE; build it with build_skill_moe first")
    if info.get("base_id") is None:
        raise ValueError("the base id is unknown; pass base_id to build_skill_moe")
    for row in info["experts"]:
        if row.get("id") is None:
            raise ValueError("an expert id is unknown; pass expert_ids to build_skill_moe")
    os.makedirs(path, exist_ok=True)
    state = {}
    for layer_index, moe in enumerate(moe_layers(model)):
        prefix = f"layers.{layer_index}."
        state[prefix + "router.weight"] = moe.router.weight.detach().cpu().contiguous()
        state[prefix + "router_bias"] = moe.router_bias.detach().cpu().contiguous()
        for expert_index in range(1, moe.n_experts()):
            expert = moe.experts[expert_index]
            if not isinstance(expert, StitchedExpert):
                continue
            for name, value in expert.w_in.state_dict().items():
                state[f"{prefix}experts.{expert_index}.w_in.{name}"] = value.detach().cpu().contiguous()
            for name, value in expert.w_out.state_dict().items():
                state[f"{prefix}experts.{expert_index}.w_out.{name}"] = value.detach().cpu().contiguous()
    for name, value in lora_state_dict(model).items():
        state["lora." + name] = value
    save_file(state, os.path.join(path, WEIGHTS))
    manifest = dict(info)
    manifest["format"] = FORMAT
    manifest["n_layers"] = len(moe_layers(model))
    with open(os.path.join(path, MANIFEST), "w") as f:
        json.dump(manifest, f, indent=2)
    if tokenizer is not None:
        tokenizer.save_pretrained(path)
    return manifest


def local_tensors(directory, names):
    """Read the named tensors from a local model directory with byte-range reads."""
    from wsp.fetch import local_range_reader, read_tensors

    index_path = os.path.join(directory, "model.safetensors.index.json")
    shards = {}
    if os.path.exists(index_path):
        with open(index_path) as f:
            weight_map = json.load(f)["weight_map"]
        for name in names:
            if name not in weight_map:
                raise KeyError(f"tensor not found in index: {name}")
            shard = weight_map[name]
            if shard not in shards:
                shards[shard] = []
            shards[shard].append(name)
    else:
        shards["model.safetensors"] = list(names)
    result = {}
    for shard in shards:
        found = read_tensors(local_range_reader(os.path.join(directory, shard)), shards[shard])
        for name in found:
            result[name] = found[name]
    return result


def fetch_expert_tensors(repo_id, names, cache_dir=None):
    """Fetch tensors from a local model directory, or from the Hub with wsp.fetch.fetch_tensors."""
    if os.path.isdir(repo_id):
        return local_tensors(repo_id, names)
    from wsp.fetch import fetch_tensors

    return fetch_tensors(repo_id, names, cache_dir=cache_dir)


def expert_layer_modules(repo_id, layers, dtype, cache_dir=None):
    """Return (config, {donor_layer: (pre_norm, mlp, post_norm)}) built from fetched MLP tensors only."""
    import transformers

    from wsp.parts import build_mlp_cluster, mlp_tensor_names

    config = transformers.AutoConfig.from_pretrained(repo_id)
    wanted = []
    for layer in layers:
        if layer not in wanted:
            wanted.append(layer)
    names = []
    for layer in wanted:
        names += mlp_tensor_names(config, layer)
    tensors = fetch_expert_tensors(repo_id, names, cache_dir)
    inner = config
    if hasattr(config, "get_text_config"):
        inner = config.get_text_config()
    modules = {}
    for layer in wanted:
        cluster = build_mlp_cluster(config, layer, torch.arange(inner.intermediate_size), tensors)
        cluster = cluster.to(dtype)
        modules[layer] = (cluster.pre_norm, cluster.mlp, cluster.post_norm)
    return config, modules


def load_skill_moe(path, device="cpu", dtype=torch.bfloat16, cache_dir=None, base_model=None):
    """Rebuild a saved skill MoE. Return (model, tokenizer); tokenizer is None when none was saved.

    The base loads by its id (or pass base_model). Each expert fetches only
    the MLP and norm tensors of the layers it uses.
    """
    import transformers
    from safetensors.torch import load_file

    with open(os.path.join(path, MANIFEST)) as f:
        manifest = json.load(f)
    if manifest.get("format") != FORMAT:
        raise ValueError(f"{path} is not a {FORMAT} directory")
    if base_model is None:
        base_model = transformers.AutoModelForCausalLM.from_pretrained(manifest["base_id"], dtype=dtype)
    n_layers = len(decoder_layers(base_model))
    if n_layers != manifest["n_layers"]:
        raise ValueError(f"the base has {n_layers} layers, the manifest {manifest['n_layers']}")
    d_host = text_config(base_model).hidden_size

    layer_experts = []
    for index in range(n_layers):
        layer_experts.append([])
    for row in manifest["experts"]:
        config, modules = expert_layer_modules(row["id"], row["layer_map"], dtype, cache_dir)
        for index in range(n_layers):
            # Two base layers can map to one donor layer. They share its frozen
            # modules, as in build_skill_moe.
            pre_norm, mlp, post_norm = modules[row["layer_map"][index]]
            if row["kind"] == "plain":
                layer_experts[index].append(PlainExpert(pre_norm, mlp, post_norm))
            else:
                w_in = make_stitch(row["stitch"], d_host, row["d_donor"])
                w_out = make_w_out(row["d_donor"], d_host)
                layer_experts[index].append(StitchedExpert(w_in, pre_norm, mlp, w_out, post_norm))

    info = {}
    for key in ["base_id", "experts", "top_k", "router_init", "base_logit", "load_balance"]:
        info[key] = manifest[key]
    # The saved router weights replace the init, so no noise is needed here.
    info["router_noise"] = 0.0
    info["lora"] = None
    assemble(base_model, layer_experts, info)
    info["router_noise"] = manifest.get("router_noise", 0.0)
    lora = manifest.get("lora")
    if lora is not None:
        add_lora(base_model, lora["rank"], lora["alpha"], tuple(lora["targets"]))

    state = load_file(os.path.join(path, WEIGHTS))
    lora_state = {}
    with torch.no_grad():
        for layer_index, moe in enumerate(moe_layers(base_model)):
            prefix = f"layers.{layer_index}."
            moe.router.weight.copy_(state[prefix + "router.weight"])
            moe.router_bias.copy_(state[prefix + "router_bias"])
            for expert_index in range(1, moe.n_experts()):
                expert = moe.experts[expert_index]
                if not isinstance(expert, StitchedExpert):
                    continue
                for part_name in ["w_in", "w_out"]:
                    module = getattr(expert, part_name)
                    part_state = {}
                    for name in module.state_dict():
                        part_state[name] = state[f"{prefix}experts.{expert_index}.{part_name}.{name}"]
                    module.load_state_dict(part_state, strict=True)
    for key in state:
        if key.startswith("lora."):
            lora_state[key[len("lora."):]] = state[key]
    if lora is not None:
        from wsp.lora import load_lora_state

        load_lora_state(base_model, lora_state)

    tokenizer = None
    if os.path.exists(os.path.join(path, "tokenizer_config.json")):
        tokenizer = transformers.AutoTokenizer.from_pretrained(path)
    return base_model.to(device).eval(), tokenizer
