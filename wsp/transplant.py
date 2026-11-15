"""Same-family expert transplant: skill experts from a big MoE written into a small dense model.

The base is a small dense model (Qwen3-1.7B). The source is a big MoE of
the same family, with the same hidden width and the same tokenizer
(Qwen3-30B-A3B). Next to the MLP of every base layer we put a small bank
of K experts taken from the source layer at the same relative depth. The
model stays near base size and runs alone. The source is not run at
inference.

Per base layer i, with source layer j = layer_map[i]:

  h        the residual state at the MLP input (before the base pre-MLP norm)
  base     out = base_mlp(norm(h)), unchanged
  bank     s = stitch_in(h)                 base basis to source basis
           x = source_norm_j(s)             the source's own pre-MLP norm
           p = softmax(router(x))           router rows start as the source router rows of the K chosen experts
           top 2 of K, renormalized         the source's norm_topk_prob rule
           y = sum_e w_e * expert_e(x)      the source's own expert MLPs, frozen
           z = stitch_out(y)                source basis to base basis
  total    out + gate * z                   gate is a scalar per layer, starts at 0

A fresh transplant equals the base exactly because every gate is 0.
stitch_in and stitch_out start from ridge fits on paired states (fit_stitches)
or from the identity. Only the stitches, routers, gates, and an optional
LoRA on the base train (set_trainable).

Selection (select_experts): per skill, the K experts per mapped source
layer with the highest log ratio of routing frequency on skill text over
general text (wsp.atlas.expert_skill_map). The skills take turns, so K=4
over math, code, and knowledge gives 2, 1, and 1 experts per layer.
control="random" draws K random experts of the same source layer.

Stitch rank: a full 2048 by 2048 stitch has 4.2M parameters, 8.4M per
layer for both, which is as large as two experts. stitch_rank r keeps the
ridge solution as a rank r product (truncated SVD) with 2 * 2048 * r
parameters. Counts are in count_params.

save_transplant writes manifest.json and transplant_weights.safetensors.
load_transplant loads the base by id and fetches only the chosen expert
tensors of the source by byte range (wsp.fetch), the way skill_moe does.

The pair is a parameter (PAIRS). The Liquid pair needs transformers 5 and
fused expert support in wsp.parts, and is not tested here.
"""

import copy
import json
import os
import random

import torch
from torch import nn

from wsp.parts import build_expert_part, expert_tensor_names, find_decoder_layers
from wsp.skill_moe import depth_map, fetch_expert_tensors, text_config

FORMAT = "transplant/1"
MANIFEST = "manifest.json"
WEIGHTS = "transplant_weights.safetensors"

PAIRS = {
    "qwen3": {"base": "Qwen/Qwen3-1.7B", "source": "Qwen/Qwen3-30B-A3B"},
    "lfm2": {"base": "LiquidAI/LFM2-1.2B", "source": "LiquidAI/LFM2-8B-A1B"},
}

MLP_NAMES = ("mlp", "feed_forward")
NORM_NAMES = ("post_attention_layernorm", "pre_feedforward_layernorm", "ffn_norm")


# ---------------------------------------------------------------- layer access


def decoder_layers(model):
    _, layers = find_decoder_layers(model, 0)
    return layers


def ffn_slots(layer):
    """Return (mlp attribute name, pre-MLP norm attribute name) of a decoder layer."""
    mlp_name = None
    for name in MLP_NAMES:
        if hasattr(layer, name):
            mlp_name = name
            break
    norm_name = None
    for name in NORM_NAMES:
        if hasattr(layer, name):
            norm_name = name
            break
    if mlp_name is None or norm_name is None:
        raise ValueError(f"{type(layer).__name__} has no MLP and pre-MLP norm this version knows")
    return mlp_name, norm_name


def source_moe_layers(config):
    """Return the indices of the sparse layers of a MoE source config."""
    inner = config
    if hasattr(config, "get_text_config"):
        inner = config.get_text_config()
    count = inner.num_hidden_layers
    dense_first = getattr(inner, "num_dense_layers", None)
    only = getattr(inner, "mlp_only_layers", None)
    step = getattr(inner, "decoder_sparse_step", None)
    found = []
    for layer in range(count):
        if dense_first is not None and layer < dense_first:
            continue
        if only is not None and layer in only:
            continue
        if step is not None and (layer + 1) % step != 0:
            continue
        found.append(layer)
    return found


def make_layer_map(n_base, n_source, moe_layers=None):
    """Base layer i takes source layer at the same relative depth, snapped to a sparse layer."""
    mapped = depth_map(n_base, n_source)
    if moe_layers is None:
        return mapped
    out = []
    for layer in mapped:
        best = moe_layers[0]
        for candidate in moe_layers:
            if abs(candidate - layer) < abs(best - layer):
                best = candidate
        out.append(best)
    return out


# ---------------------------------------------------------------- selection


def ranked_experts(log_ratio, freq=None, min_freq=0.0):
    """Expert ids sorted by log ratio, highest first. Ties keep the lower id first.

    An expert whose routing frequency on the skill text is below min_freq
    goes after all the others. A rarely used expert can have a high log
    ratio from a handful of tokens, and it has little to give.
    """
    order = list(range(len(log_ratio)))
    if freq is None:
        order.sort(key=lambda expert: (-log_ratio[expert], expert))
        return order
    order.sort(key=lambda expert: (freq[expert] < min_freq, -log_ratio[expert], expert))
    return order


def pick_round_robin(rankings, k):
    """Take turns over the skills; each takes its best unused expert. Return (ids, skills)."""
    names = list(rankings.keys())
    used = set()
    pointers = {}
    for name in names:
        pointers[name] = 0
    ids = []
    labels = []
    turn = 0
    while len(ids) < k:
        name = names[turn % len(names)]
        turn += 1
        order = rankings[name]
        while pointers[name] < len(order) and order[pointers[name]] in used:
            pointers[name] += 1
        if pointers[name] >= len(order):
            if all(pointers[other] >= len(rankings[other]) for other in names):
                break
            continue
        expert = order[pointers[name]]
        used.add(expert)
        ids.append(expert)
        labels.append(name)
    return ids, labels


def select_experts(source, tokenizer, skill_texts, general_texts, layer_map, k, control=None, seed=0, max_length=256,
                   min_freq=0.0):
    """Choose K experts per mapped source layer. Return a plan dict (string layer keys).

    skill_texts: {skill name: list of texts}. general_texts: list of texts.
    control "random": K random experts of the same layer, seeded.
    min_freq: experts routed on fewer than this share of the skill tokens rank last.
    """
    from wsp.atlas import expert_skill_map, find_routers

    routers = find_routers(source)
    if len(routers) == 0:
        raise ValueError("the source has no MoE layers")
    num_experts = {}
    for layer, _, _, _, count in routers:
        num_experts[layer] = count
    source_layers = sorted(set(layer_map))
    for layer in source_layers:
        if layer not in num_experts:
            raise ValueError(f"source layer {layer} is not a sparse layer")
        if k > num_experts[layer]:
            raise ValueError(f"K={k} is more than the {num_experts[layer]} experts of source layer {layer}")
    plan = {"layer_map": list(layer_map), "k": int(k), "control": control, "seed": int(seed), "min_freq": min_freq,
            "experts": {}, "labels": {}, "skills": list(skill_texts.keys()), "selectivity": {}}
    if control == "random":
        for layer in source_layers:
            rng = random.Random(seed * 100003 + layer)
            ids = sorted(rng.sample(range(num_experts[layer]), k))
            plan["experts"][str(layer)] = ids
            plan["labels"][str(layer)] = ["random"] * k
        return plan
    if control is not None:
        raise ValueError(f"control must be None or 'random', got {control!r}")
    maps = {}
    for skill in skill_texts:
        maps[skill] = expert_skill_map(source, tokenizer, skill_texts[skill], general_texts, top=1,
                                       max_length=max_length)
    for layer in source_layers:
        rankings = {}
        for skill in maps:
            entry = maps[skill]["layers"][layer]
            rankings[skill] = ranked_experts(entry["log_ratio"], entry["skill_freq"], min_freq)
        ids, labels = pick_round_robin(rankings, k)
        plan["experts"][str(layer)] = ids
        plan["labels"][str(layer)] = labels
        ratios = []
        for expert, skill in zip(ids, labels):
            ratios.append(round(maps[skill]["layers"][layer]["log_ratio"][expert], 4))
        plan["selectivity"][str(layer)] = ratios
    return plan


# ---------------------------------------------------------------- source parts


def parts_from_model(source, plan):
    """Copy the chosen experts, pre-MLP norm, and router rows out of a loaded source model."""
    from wsp.atlas import find_routers

    found = {}
    for layer, moe, router, _, _ in find_routers(source):
        found[layer] = (moe, router)
    layers = decoder_layers(source)
    parts = {}
    for key, ids in plan["experts"].items():
        layer = int(key)
        moe, router = found[layer]
        _, norm_name = ffn_slots(layers[layer])
        experts = []
        for expert in ids:
            experts.append(copy.deepcopy(moe.experts[expert]))
        rows = router.weight.detach()[list(ids)].float().cpu().clone()
        parts[key] = {"pre_norm": copy.deepcopy(getattr(layers[layer], norm_name)), "experts": experts,
                      "router": rows}
    return parts


def fetch_parts(source_id, plan, dtype=torch.bfloat16, cache_dir=None, with_router=True):
    """Fetch the chosen experts of the source by byte range. Same result as parts_from_model."""
    import transformers

    config = transformers.AutoConfig.from_pretrained(source_id)
    parts = {}
    for key, ids in plan["experts"].items():
        layer = int(key)
        names = expert_tensor_names(config, layer, ids)
        wanted = list(names)
        gate_name = names[0].split(".experts.")[0] + ".gate.weight"
        if with_router:
            wanted.append(gate_name)
        tensors = fetch_expert_tensors(source_id, wanted, cache_dir)
        chosen = {}
        for name in names:
            chosen[name] = tensors[name]
        part = build_expert_part(config, layer, ids, chosen).to(dtype)
        rows = None
        if with_router:
            rows = tensors[gate_name][list(ids)].float()
        parts[key] = {"pre_norm": part.pre_norm, "experts": list(part.experts), "router": rows}
    return parts


# ---------------------------------------------------------------- stitches and the bank


class Stitch(nn.Module):
    """A float32 linear map, full or as a rank r product, with an optional bias."""

    def __init__(self, d_in, d_out, rank=None, bias=True):
        super().__init__()
        if rank is not None and rank >= min(d_in, d_out):
            rank = None
        self.rank = rank
        if rank is None:
            self.linear = nn.Linear(d_in, d_out, bias=bias).float()
            self.down = None
            self.up = None
        else:
            self.linear = None
            self.down = nn.Linear(d_in, rank, bias=False).float()
            self.up = nn.Linear(rank, d_out, bias=bias).float()

    def forward(self, x):
        if self.rank is None:
            return self.linear(x)
        return self.up(self.down(x))

    def matrix(self):
        """The effective (d_out, d_in) weight."""
        if self.rank is None:
            return self.linear.weight.detach().float()
        return (self.up.weight.detach().float() @ self.down.weight.detach().float())

    @torch.no_grad()
    def set_matrix(self, weight, bias=None):
        weight = weight.detach().to(torch.float32)
        if self.rank is None:
            self.linear.weight.copy_(weight.to(self.linear.weight.device))
            if self.linear.bias is not None:
                self.linear.bias.zero_()
                if bias is not None:
                    self.linear.bias.copy_(bias.to(self.linear.bias.device, torch.float32))
            return
        u, s, vt = torch.linalg.svd(weight.cpu(), full_matrices=False)
        root = s[:self.rank].sqrt()
        self.up.weight.copy_((u[:, :self.rank] * root).to(self.up.weight.device))
        self.down.weight.copy_((root.unsqueeze(1) * vt[:self.rank]).to(self.down.weight.device))
        if self.up.bias is not None:
            self.up.bias.zero_()
            if bias is not None:
                self.up.bias.copy_(bias.to(self.up.bias.device, torch.float32))

    @torch.no_grad()
    def set_identity(self):
        """The identity, or its first r coordinates as a rank r product (no SVD)."""
        if self.rank is None:
            self.linear.weight.copy_(torch.eye(self.linear.out_features, self.linear.in_features))
            if self.linear.bias is not None:
                self.linear.bias.zero_()
            return
        self.down.weight.copy_(torch.eye(self.rank, self.down.in_features))
        self.up.weight.copy_(torch.eye(self.up.out_features, self.rank))
        if self.up.bias is not None:
            self.up.bias.zero_()


class NormTap(nn.Module):
    """The decoder layer's pre-MLP norm. It also keeps its input for the bank."""

    def __init__(self, norm, stash):
        super().__init__()
        self.norm = norm
        self.stash = stash

    def forward(self, h):
        self.stash["h"] = h
        return self.norm(h)


class TransplantBank(nn.Module):
    """base MLP output plus gate * stitch_out(top-2 of K source experts on the stitched state)."""

    def __init__(self, base_mlp, stash, src_norm, experts, router_rows, d_model, d_source, stitch_rank=None,
                 top_k=2):
        super().__init__()
        n = len(experts)
        if n < 1:
            raise ValueError("a bank needs at least one expert")
        self.base_mlp = base_mlp
        self.stash = stash
        self.src_norm = src_norm
        self.experts = nn.ModuleList(experts)
        self.top_k = min(int(top_k), n)
        self.stitch_in = Stitch(d_model, d_source, stitch_rank, bias=True)
        self.stitch_out = Stitch(d_source, d_model, stitch_rank, bias=False)
        self.stitch_in.set_identity()
        self.stitch_out.set_identity()
        self.router = nn.Linear(d_source, n, bias=True).float()
        with torch.no_grad():
            self.router.bias.zero_()
            if router_rows is not None:
                self.router.weight.copy_(router_rows.to(torch.float32))
        self.gate = nn.Parameter(torch.zeros((), dtype=torch.float32))
        self.track = False
        self.register_buffer("route_counts", torch.zeros(n, dtype=torch.float64), persistent=False)

    def n_experts(self):
        return len(self.experts)

    def expert_dtype(self):
        for param in self.experts[0].parameters():
            return param.dtype
        return torch.float32

    def route(self, h):
        """Return (x, idx, weights) for flat residual states h of shape (tokens, hidden).

        x is the source-normed stitched state the experts read.
        """
        with torch.autocast(device_type=h.device.type, enabled=False):
            s = self.stitch_in(h.to(torch.float32))
            x = self.src_norm(s.to(self.expert_dtype()))
            logits = self.router(x.to(torch.float32))
            probs = torch.softmax(logits, dim=-1)
            vals, idx = probs.topk(self.top_k, dim=-1)
            weights = vals / vals.sum(dim=-1, keepdim=True)
        return x, idx, weights

    def expert_sum(self, h):
        """The weighted sum of the chosen experts, in the source basis, float32, shape of h."""
        shape = h.shape
        flat = h.reshape(-1, shape[-1])
        x, idx, weights = self.route(flat)
        if self.track:
            with torch.no_grad():
                hits = torch.zeros(flat.shape[0], self.n_experts(), dtype=torch.float64, device=flat.device)
                hits.scatter_(1, idx, 1.0)
                self.route_counts += hits.sum(dim=0)
        out = None
        for index in range(self.n_experts()):
            hit = idx == index
            rows = hit.any(dim=-1).nonzero().squeeze(-1)
            if rows.numel() == 0:
                continue
            weight = (weights * hit.to(weights.dtype)).sum(dim=-1)[rows]
            y = self.experts[index](x[rows]).to(torch.float32)
            if out is None:
                out = torch.zeros(flat.shape[0], y.shape[-1], dtype=torch.float32, device=flat.device)
            out = out.index_add(0, rows, y * weight.unsqueeze(-1))
        if out is None:
            out = torch.zeros(flat.shape[0], self.router.in_features, dtype=torch.float32, device=flat.device)
        return out.reshape(shape[:-1] + (out.shape[-1],))

    def forward(self, x):
        out = self.base_mlp(x)
        h = self.stash["h"]
        with torch.autocast(device_type=h.device.type, enabled=False):
            z = self.stitch_out(self.expert_sum(h))
            z = self.gate * z
        return out + z.to(out.dtype)


# ---------------------------------------------------------------- building


def build_transplant(base, plan, parts, stitches=None, lora_rank=None, lora_alpha=None, top_k=2, stitch_rank=None,
                     base_id=None, source_id=None, pair=None):
    """Put a TransplantBank next to the MLP of every base layer (in place). Return base.

    plan: from select_experts (needs layer_map, experts, labels).
    parts: from parts_from_model or fetch_parts.
    stitches: None, or {base layer: {"in_w", "in_b", "out_w"}} from fit_stitches.
    lora_rank: a LoRA on the base attention and MLP, put on before the bank so
    that the source experts get none.
    """
    layers = decoder_layers(base)
    layer_map = plan["layer_map"]
    if len(layer_map) != len(layers):
        raise ValueError(f"the plan maps {len(layer_map)} layers, the base has {len(layers)}")
    for param in base.parameters():
        param.requires_grad_(False)
    lora = None
    if lora_rank is not None:
        from wsp.lora import apply_lora, default_alpha

        if lora_alpha is None:
            lora_alpha = default_alpha(lora_rank)
        apply_lora(base, lora_rank, lora_alpha)
        lora = {"rank": int(lora_rank), "alpha": lora_alpha}
    d_model = text_config(base).hidden_size
    used = set()
    for index in range(len(layers)):
        layer = layers[index]
        mlp_name, norm_name = ffn_slots(layer)
        part = parts[str(layer_map[index])]
        if layer_map[index] in used:
            part = {"pre_norm": copy.deepcopy(part["pre_norm"]), "experts": copy.deepcopy(part["experts"]),
                    "router": part["router"]}
        used.add(layer_map[index])
        d_source = part["router"].shape[1] if part["router"] is not None else d_model
        stash = {}
        tap = NormTap(getattr(layer, norm_name), stash)
        bank = TransplantBank(getattr(layer, mlp_name), stash, part["pre_norm"], part["experts"], part["router"],
                              d_model, d_source, stitch_rank, top_k)
        if stitches is not None and index in stitches:
            entry = stitches[index]
            bank.stitch_in.set_matrix(entry["in_w"], entry["in_b"])
            bank.stitch_out.set_matrix(entry["out_w"])
        device = None
        for param in getattr(layer, mlp_name).parameters():
            device = param.device
            break
        if device is not None:
            bank.to(device)
        setattr(layer, norm_name, tap)
        setattr(layer, mlp_name, bank)
    base.wsp_transplant = {
        "format": FORMAT, "pair": pair, "base_id": base_id, "source_id": source_id,
        "layer_map": list(layer_map), "k": plan["k"], "control": plan.get("control"),
        "seed": plan.get("seed"), "experts": plan["experts"], "labels": plan["labels"],
        "skills": plan.get("skills"), "top_k": int(top_k), "stitch_rank": stitch_rank,
        "d_source": int(d_source), "lora": lora, "n_layers": len(layers),
    }
    set_trainable(base)
    return base


# ---------------------------------------------------------------- access and counts


def banks(model):
    found = []
    for module in model.modules():
        if isinstance(module, TransplantBank):
            found.append(module)
    return found


def stitch_parameters(model):
    params = []
    for bank in banks(model):
        for stitch in (bank.stitch_in, bank.stitch_out):
            for param in stitch.parameters():
                params.append(param)
    return params


def router_parameters(model):
    params = []
    for bank in banks(model):
        params.append(bank.router.weight)
        params.append(bank.router.bias)
    return params


def gate_parameters(model):
    params = []
    for bank in banks(model):
        params.append(bank.gate)
    return params


def lora_parameters(model):
    from wsp.skill_moe import lora_parameters as found

    return found(model)


def set_trainable(model, lora=True):
    """Freeze everything, then unfreeze stitches, routers, gates, and the base LoRA. Return the groups."""
    for param in model.parameters():
        param.requires_grad_(False)
    groups = {"stitch": stitch_parameters(model), "router": router_parameters(model),
              "gate": gate_parameters(model), "lora": []}
    if lora:
        groups["lora"] = lora_parameters(model)
    for name in groups:
        for param in groups[name]:
            param.requires_grad_(True)
    return groups


def count_params(model):
    """Return parameter counts of a transplanted model.

    total: every distinct parameter once. base: the original dense model
    (total minus the banks and the LoRA). active: params one token runs,
    which is the base (with the embedding rule of wsp.skills_eval.model_params:
    a tied embedding counts, an untied input table does not), plus the LoRA,
    plus per layer top_k experts, both stitches, the router, the gate, and the
    source norm. total_ratio and active_ratio divide by the base values.
    """
    from wsp.skills_eval import model_params

    total, _ = model_params(model)
    seen = set()
    bank_total = 0
    bank_active = 0
    experts_total = 0
    stitch = 0
    router = 0
    for bank in banks(model):
        size = 0
        for param in bank.experts[0].parameters():
            size += param.numel()
        fixed = 1
        for module in (bank.stitch_in, bank.stitch_out, bank.router, bank.src_norm):
            for param in module.parameters():
                fixed += param.numel()
        for param in list(bank.stitch_in.parameters()) + list(bank.stitch_out.parameters()):
            stitch += param.numel()
        router += bank.router.weight.numel() + bank.router.bias.numel()
        bank_active += bank.top_k * size + fixed
        for module in (bank.src_norm, bank.experts, bank.stitch_in, bank.stitch_out, bank.router):
            for param in module.parameters():
                if id(param) not in seen:
                    seen.add(id(param))
                    bank_total += param.numel()
        bank_total += 1
        for param in bank.experts.parameters():
            experts_total += param.numel()
    lora = 0
    for param in lora_parameters(model):
        lora += param.numel()
    base_total = total - bank_total - lora
    base_run = base_total
    embed = model.get_input_embeddings()
    head = model.get_output_embeddings()
    if embed is not None:
        tied = head is not None and head.weight is embed.weight
        if not tied:
            base_run -= embed.weight.numel()
    active = base_run + lora + bank_active
    return {"total": total, "active": active, "base": base_total, "base_run": base_run, "bank": bank_total,
            "experts": experts_total, "stitch": stitch, "router": router, "lora": lora,
            "total_ratio": total / max(1, base_total), "active_ratio": active / max(1, base_run)}


def source_active_params(source):
    """Return (total, active) params of a MoE source: top_k of the experts per sparse layer run."""
    from wsp.atlas import find_routers
    from wsp.skills_eval import model_params

    total, run = model_params(source)
    for _, moe, _, top_k, num_experts in find_routers(source):
        size = 0
        for param in moe.experts[0].parameters():
            size += param.numel()
        run -= size * (num_experts - top_k)
    return total, run


def track_routing(model, on=True):
    for bank in banks(model):
        bank.track = on
        if on:
            bank.route_counts.zero_()


def routing_shares(model):
    """Return {"per_layer": [[share per expert]], "by_label": {label: mean share over layers}}."""
    info = getattr(model, "wsp_transplant", None)
    per_layer = []
    by_label = {}
    layer_index = 0
    for bank in banks(model):
        counts = bank.route_counts.double().cpu()
        slots = float(counts.sum())
        row = []
        for expert in range(bank.n_experts()):
            row.append(float(counts[expert]) / slots if slots > 0 else 0.0)
        per_layer.append(row)
        if info is not None:
            labels = info["labels"][str(info["layer_map"][layer_index])]
            for expert in range(bank.n_experts()):
                by_label[labels[expert]] = by_label.get(labels[expert], 0.0) + row[expert]
        layer_index += 1
    if len(per_layer) > 0:
        for label in by_label:
            by_label[label] = by_label[label] / len(per_layer)
    return {"per_layer": per_layer, "by_label": by_label}


# ---------------------------------------------------------------- ridge fit of the stitches


def ridge_solve(xx, xy, sx, sy, n, ridge, intercept):
    """Solve min |Y - X W^T - b|^2 + ridge term from sums. Return (W (d_out, d_in), b or None).

    xx = X^T X, xy = X^T Y, sx and sy are column sums, n the row count. The
    ridge strength is ridge times the mean diagonal of X^T X (centered).
    """
    d_in = xx.shape[0]
    if intercept:
        mx = sx / n
        my = sy / n
        xxc = xx - n * torch.outer(mx, mx)
        xyc = xy - n * torch.outer(mx, my)
    else:
        xxc = xx
        xyc = xy
    lam = ridge * float(torch.trace(xxc)) / d_in
    eye = torch.eye(d_in, dtype=xxc.dtype, device=xxc.device)
    solution = torch.linalg.solve(xxc + lam * eye, xyc)
    weight = solution.t().contiguous()
    bias = None
    if intercept:
        bias = my - weight @ mx
    return weight, bias


def probe_in_bias(stitch):
    """The bias of a stitch, or None."""
    if stitch.rank is None:
        return stitch.linear.bias
    return stitch.up.bias


def explained(weight, bias, xx, xy, sx, sy, yy, n, intercept):
    """R squared of a map on the accumulated data (1 is a perfect fit, 0 is the mean).

    With an intercept the data are centered, which makes the fit independent of
    the bias, so the bias argument is only a placeholder here.
    """
    weight = weight.to(xx.dtype)
    if intercept:
        mx = sx / n
        my = sy / n
        xxc = xx - n * torch.outer(mx, mx)
        xyc = xy - n * torch.outer(mx, my)
        sst = float(yy - n * (my * my).sum())
    else:
        xxc = xx
        xyc = xy
        sst = float(yy)
    sse = sst - 2.0 * float((weight.t() * xyc).sum()) + float(((xxc @ weight.t()) * weight.t()).sum())
    if sst <= 0:
        return 0.0
    return 1.0 - sse / sst


class PairStats:
    """Running sums for the two ridge fits of one base layer."""

    def __init__(self, d_base, d_source, device):
        kw = {"dtype": torch.float64, "device": device}
        self.n = 0
        self.in_xx = torch.zeros(d_base, d_base, **kw)
        self.in_xy = torch.zeros(d_base, d_source, **kw)
        self.in_sx = torch.zeros(d_base, **kw)
        self.in_sy = torch.zeros(d_source, **kw)
        self.in_yy = 0.0
        self.out_xx = torch.zeros(d_source, d_source, **kw)
        self.out_xy = torch.zeros(d_source, d_base, **kw)
        self.out_yy = 0.0

    def add(self, base_h, source_h, base_out, source_out):
        bh = base_h.to(self.in_xx.device, torch.float64)
        sh = source_h.to(self.in_xx.device, torch.float64)
        bo = base_out.to(self.in_xx.device, torch.float64)
        so = source_out.to(self.in_xx.device, torch.float64)
        self.n += bh.shape[0]
        self.in_xx += bh.t() @ bh
        self.in_xy += bh.t() @ sh
        self.in_sx += bh.sum(dim=0)
        self.in_sy += sh.sum(dim=0)
        self.in_yy += float((sh * sh).sum())
        self.out_xx += so.t() @ so
        self.out_xy += so.t() @ bo
        self.out_yy += float((bo * bo).sum())


def first_tensor(output):
    if isinstance(output, (tuple, list)):
        return output[0]
    return output


@torch.no_grad()
def fit_stitches(base, source, tokenizer, texts, layer_map, rank=None, ridge=1e-3, max_length=256, batch_size=4,
                 stats_device=None):
    """Ridge-fit stitch_in and stitch_out per base layer on paired states of the same text.

    stitch_in: base residual state at the MLP input to the source residual
    state at the mapped layer's MLP input, with a bias.
    stitch_out: the source's sparse-MLP output to the base MLP output, no bias.
    Return {base layer: {"in_w", "in_b", "out_w", "r2_in", "r2_out"}}. With a
    rank, r2 is measured after the truncation.
    """
    base_layers = decoder_layers(base)
    source_layers = decoder_layers(source)
    device = next(base.parameters()).device
    source_device = next(source.parameters()).device
    if stats_device is None:
        stats_device = device
    d_base = text_config(base).hidden_size
    d_source = text_config(source).hidden_size
    stats = []
    for _ in range(len(base_layers)):
        stats.append(PairStats(d_base, d_source, stats_device))
    seen = {}
    handles = []

    def keep(key, take_input):
        def hook(module, args, output=None):
            value = args[0] if take_input else first_tensor(output)
            seen[key] = value.detach()
        return hook

    for index, layer in enumerate(base_layers):
        mlp_name, norm_name = ffn_slots(layer)
        handles.append(getattr(layer, norm_name).register_forward_pre_hook(keep(("bh", index), True)))
        handles.append(getattr(layer, mlp_name).register_forward_hook(keep(("bo", index), False)))
    for j in sorted(set(layer_map)):
        mlp_name, norm_name = ffn_slots(source_layers[j])
        handles.append(getattr(source_layers[j], norm_name).register_forward_pre_hook(keep(("sh", j), True)))
        handles.append(getattr(source_layers[j], mlp_name).register_forward_hook(keep(("so", j), False)))
    was_base = base.training
    was_source = source.training
    base.eval()
    source.eval()
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    try:
        for start in range(0, len(texts), batch_size):
            encoded = tokenizer(texts[start:start + batch_size], return_tensors="pt", padding=True,
                                truncation=True, max_length=max_length)
            mask = encoded["attention_mask"].reshape(-1).bool()
            base(input_ids=encoded["input_ids"].to(device), attention_mask=encoded["attention_mask"].to(device),
                 use_cache=False)
            source(input_ids=encoded["input_ids"].to(source_device),
                   attention_mask=encoded["attention_mask"].to(source_device), use_cache=False)
            for index in range(len(base_layers)):
                j = layer_map[index]
                rows = []
                for key in (("bh", index), ("sh", j), ("bo", index), ("so", j)):
                    value = seen[key]
                    rows.append(value.reshape(-1, value.shape[-1])[mask.to(value.device)])
                stats[index].add(rows[0], rows[1], rows[2], rows[3])
            seen.clear()
    finally:
        for handle in handles:
            handle.remove()
        if was_base:
            base.train()
        if was_source:
            source.train()

    out = {}
    for index in range(len(base_layers)):
        st = stats[index]
        in_w, in_b = ridge_solve(st.in_xx, st.in_xy, st.in_sx, st.in_sy, st.n, ridge, True)
        out_w, _ = ridge_solve(st.out_xx, st.out_xy, None, None, st.n, ridge, False)
        probe_in = Stitch(d_base, d_source, rank, bias=True)
        probe_in.set_matrix(in_w.float().cpu(), in_b.float().cpu())
        probe_out = Stitch(d_source, d_base, rank, bias=False)
        probe_out.set_matrix(out_w.float().cpu())
        mean_in = st.in_sx / st.n
        mean_out = st.in_sy / st.n
        in_b = (mean_out - probe_in.matrix().to(mean_in.device, torch.float64) @ mean_in).float().cpu()
        r2_in = explained(probe_in.matrix().to(st.in_xx.device), probe_in_bias(probe_in), st.in_xx, st.in_xy,
                          st.in_sx, st.in_sy, st.in_yy, st.n, True)
        r2_out = explained(probe_out.matrix().to(st.out_xx.device), None, st.out_xx, st.out_xy, None, None,
                           st.out_yy, st.n, False)
        out[index] = {"in_w": in_w.float().cpu(), "in_b": in_b.float().cpu(), "out_w": out_w.float().cpu(),
                      "r2_in": round(r2_in, 4), "r2_out": round(r2_out, 4)}
    return out


# ---------------------------------------------------------------- heal


def distill_loss(student_logits, teacher_logits, mask, temperature=1.0):
    """Mean KL(teacher || student) over the unmasked positions. Logits are float32 (batch, seq, vocab)."""
    vocab = min(student_logits.shape[-1], teacher_logits.shape[-1])
    t = torch.log_softmax(teacher_logits[..., :vocab] / temperature, dim=-1)
    s = torch.log_softmax(student_logits[..., :vocab] / temperature, dim=-1)
    kl = (t.exp() * (t - s)).sum(-1)
    mask = mask.to(kl.dtype)
    return (kl * mask).sum() / mask.sum().clamp(min=1.0)


def lm_loss(logits, input_ids, mask):
    """Next-token cross entropy over the unmasked positions."""
    labels = input_ids[:, 1:]
    flat = torch.nn.functional.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]), labels.reshape(-1),
                                             reduction="none")
    keep = mask[:, 1:].reshape(-1).to(flat.dtype)
    return (flat * keep).sum() / keep.sum().clamp(min=1.0)


def heal(student, tokenizer, texts, groups, steps, batch_size, seq_len, seed, teacher=None, kd_weight=1.0,
         lm_weight=1.0, autocast=None, log_every=100, anchor=None, anchor_weight=0.0, anchor_flags=None):
    """Train the param groups on texts, one pass in a seeded order. Return a loss log.

    groups: a list of {"params": [...], "lr": value, ...} for AdamW. With a
    teacher and kd_weight > 0 the loss is kd_weight * KL + lm_weight * LM.
    The teacher reads the same text as the student (teacher forcing), so the
    donor signal on a pool answer comes through the KL term only. Set
    lm_weight to 0 for a heal with no LM loss.
    anchor: a frozen model (the base). With anchor_weight > 0 the loss gains
    anchor_weight * KL(anchor || student) on the texts whose entry in
    anchor_flags (one bool per text) is True. This is the preservation term.
    autocast: None, or a torch dtype such as torch.bfloat16 (CUDA only).
    """
    use_anchor = anchor is not None and anchor_weight > 0
    if use_anchor and (anchor_flags is None or len(anchor_flags) != len(texts)):
        raise ValueError("anchor_flags needs one entry per text")
    device = next(student.parameters()).device
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    params = []
    for group in groups:
        params += list(group["params"])
    optimizer = torch.optim.AdamW(groups, weight_decay=0.0)
    order = list(range(len(texts)))
    random.Random(seed).shuffle(order)
    log = {"kd": [], "lm": [], "anchor": []}
    student.train()
    cursor = 0
    for step in range(steps):
        batch = []
        rows = []
        while len(batch) < batch_size:
            if cursor >= len(order):
                cursor = 0
            batch.append(texts[order[cursor]])
            rows.append(order[cursor])
            cursor += 1
        encoded = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=seq_len,
                            add_special_tokens=False)
        input_ids = encoded["input_ids"].to(device)
        mask = encoded["attention_mask"].to(device)
        teacher_logits = None
        if teacher is not None and kd_weight > 0:
            with torch.no_grad():
                teacher_device = next(teacher.parameters()).device
                teacher_logits = teacher(input_ids=input_ids.to(teacher_device),
                                         attention_mask=mask.to(teacher_device)).logits.float().to(device)
        if autocast is not None:
            with torch.autocast(device.type, dtype=autocast):
                logits = student(input_ids=input_ids, attention_mask=mask).logits
        else:
            logits = student(input_ids=input_ids, attention_mask=mask).logits
        logits = logits.float()
        lm = lm_loss(logits, input_ids, mask)
        loss = lm_weight * lm
        kd_value = 0.0
        if teacher_logits is not None:
            kd = distill_loss(logits, teacher_logits, mask)
            loss = loss + kd_weight * kd
            kd_value = float(kd.detach())
            del teacher_logits
        anchor_value = 0.0
        if use_anchor:
            flags = torch.tensor([bool(anchor_flags[row]) for row in rows], device=device)
            if bool(flags.any()):
                with torch.no_grad():
                    anchor_device = next(anchor.parameters()).device
                    anchor_logits = anchor(input_ids=input_ids.to(anchor_device),
                                           attention_mask=mask.to(anchor_device)).logits.float().to(device)
                keep = mask * flags.unsqueeze(1).to(mask.dtype)
                pres = distill_loss(logits, anchor_logits, keep)
                loss = loss + anchor_weight * pres
                anchor_value = float(pres.detach())
                del anchor_logits
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        optimizer.step()
        log["kd"].append(kd_value)
        log["lm"].append(float(lm.detach()))
        log["anchor"].append(anchor_value)
        if log_every and (step + 1) % log_every == 0:
            print(f"step {step + 1}/{steps} lm {log['lm'][-1]:.4f} kd {log['kd'][-1]:.4f}")
    student.eval()
    first = max(1, min(20, steps))
    return {"lm_first": sum(log["lm"][:first]) / first, "lm_last": sum(log["lm"][-first:]) / first,
            "kd_first": sum(log["kd"][:first]) / first, "kd_last": sum(log["kd"][-first:]) / first,
            "anchor_last": sum(log["anchor"][-first:]) / first}


@torch.no_grad()
def measure_kl(student, reference, tokenizer, texts, batch_size=4, seq_len=384, autocast=None):
    """Mean per-token KL(reference || student) on texts, and the student's LM loss. Lower KL is closer.

    No benchmark is read. The texts are held-out pool text. Both models read
    the same tokens with teacher forcing. Returns {"kl", "lm", "tokens"}.
    """
    device = next(student.parameters()).device
    was_training = student.training
    student.eval()
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    kl_sum = 0.0
    lm_sum = 0.0
    kl_tokens = 0
    lm_tokens = 0
    for start in range(0, len(texts), batch_size):
        encoded = tokenizer(texts[start:start + batch_size], return_tensors="pt", padding=True, truncation=True,
                            max_length=seq_len, add_special_tokens=False)
        input_ids = encoded["input_ids"].to(device)
        mask = encoded["attention_mask"].to(device)
        ref_device = next(reference.parameters()).device
        ref_logits = reference(input_ids=input_ids.to(ref_device),
                               attention_mask=mask.to(ref_device)).logits.float().to(device)
        if autocast is not None:
            with torch.autocast(device.type, dtype=autocast):
                logits = student(input_ids=input_ids, attention_mask=mask).logits
        else:
            logits = student(input_ids=input_ids, attention_mask=mask).logits
        logits = logits.float()
        count = int(mask.sum())
        kl_sum += float(distill_loss(logits, ref_logits, mask)) * count
        kl_tokens += count
        pairs = int(mask[:, 1:].sum())
        lm_sum += float(lm_loss(logits, input_ids, mask)) * pairs
        lm_tokens += pairs
        del ref_logits, logits
    if was_training:
        student.train()
    return {"kl": kl_sum / max(1, kl_tokens), "lm": lm_sum / max(1, lm_tokens), "tokens": kl_tokens}


def set_gates(model, value):
    """Set every bank gate to value (the fixed-gate no-heal mode). Return the number of banks."""
    found = banks(model)
    with torch.no_grad():
        for bank in found:
            bank.gate.fill_(float(value))
    return len(found)


def stitch_size(d_in, d_out, rank, bias):
    """Parameter count of one Stitch (see the Stitch class for the rank rule)."""
    if rank is None or rank >= min(d_in, d_out):
        return d_in * d_out + (d_out if bias else 0)
    return d_in * rank + rank * d_out + (d_out if bias else 0)


def planned_trainable(model, d_source, k, stitch_rank, lora_rank):
    """Trainable count of a transplant arm built from the plain base model, before any build.

    Stitches, routers (weight and bias), gates, and a base LoRA of lora_rank.
    Equals the count after build_transplant (tested).
    """
    from wsp.lora import params_per_rank

    d_model = text_config(model).hidden_size
    layers = len(decoder_layers(model))
    bank = (stitch_size(d_model, d_source, stitch_rank, True) + stitch_size(d_source, d_model, stitch_rank, False)
            + k * d_source + k + 1)
    return layers * bank + int(lora_rank) * params_per_rank(model)


def matched_lora_rank(model, d_source, k, stitch_rank, lora_rank):
    """The LoRA rank whose parameter count is closest to the trainable count of the transplant arm."""
    from wsp.lora import params_per_rank

    target = planned_trainable(model, d_source, k, stitch_rank, lora_rank)
    per_rank = params_per_rank(model)
    return max(1, int(round(target / per_rank)))


# ---------------------------------------------------------------- save and load


def save_transplant(model, path, tokenizer=None):
    """Write manifest.json and transplant_weights.safetensors (stitches, routers, gates, LoRA) to path."""
    from safetensors.torch import save_file

    from wsp.lora import lora_state_dict

    info = getattr(model, "wsp_transplant", None)
    if info is None:
        raise ValueError("this model has no transplant; build it with build_transplant first")
    if info.get("base_id") is None or info.get("source_id") is None:
        raise ValueError("pass base_id and source_id to build_transplant before saving")
    os.makedirs(path, exist_ok=True)
    state = {}
    for index, bank in enumerate(banks(model)):
        prefix = f"layers.{index}."
        for part_name in ("stitch_in", "stitch_out"):
            for name, value in getattr(bank, part_name).state_dict().items():
                state[f"{prefix}{part_name}.{name}"] = value.detach().cpu().contiguous()
        state[prefix + "router.weight"] = bank.router.weight.detach().cpu().contiguous()
        state[prefix + "router.bias"] = bank.router.bias.detach().cpu().contiguous()
        state[prefix + "gate"] = bank.gate.detach().cpu().reshape(1).contiguous()
    for name, value in lora_state_dict(model).items():
        state["lora." + name] = value
    save_file(state, os.path.join(path, WEIGHTS))
    with open(os.path.join(path, MANIFEST), "w") as f:
        json.dump(info, f, indent=2)
    if tokenizer is not None:
        tokenizer.save_pretrained(path)
    return info


def load_transplant(path, device="cpu", dtype=torch.bfloat16, cache_dir=None, base_model=None):
    """Rebuild a saved transplant. Return (model, tokenizer); tokenizer is None when none was saved.

    The base loads by id (or pass base_model). The source's chosen expert
    tensors come by byte range from the Hub or from a local directory.
    """
    import transformers
    from safetensors.torch import load_file

    with open(os.path.join(path, MANIFEST)) as f:
        info = json.load(f)
    if info.get("format") != FORMAT:
        raise ValueError(f"{path} is not a {FORMAT} directory")
    if base_model is None:
        base_model = transformers.AutoModelForCausalLM.from_pretrained(info["base_id"], dtype=dtype)
    if len(decoder_layers(base_model)) != info["n_layers"]:
        raise ValueError(f"the base has {len(decoder_layers(base_model))} layers, the manifest {info['n_layers']}")
    parts = fetch_parts(info["source_id"], info, dtype=dtype, cache_dir=cache_dir, with_router=False)
    for key in parts:
        parts[key]["router"] = torch.zeros(len(info["experts"][key]), info["d_source"])
    lora = info.get("lora")
    lora_rank = None
    lora_alpha = None
    if lora is not None:
        lora_rank = lora["rank"]
        lora_alpha = lora["alpha"]
    build_transplant(base_model, info, parts, lora_rank=lora_rank, lora_alpha=lora_alpha, top_k=info["top_k"],
                     stitch_rank=info["stitch_rank"], base_id=info["base_id"], source_id=info["source_id"],
                     pair=info.get("pair"))
    base_model.wsp_transplant = info
    state = load_file(os.path.join(path, WEIGHTS))
    lora_state = {}
    with torch.no_grad():
        for index, bank in enumerate(banks(base_model)):
            prefix = f"layers.{index}."
            for part_name in ("stitch_in", "stitch_out"):
                module = getattr(bank, part_name)
                part_state = {}
                for name in module.state_dict():
                    part_state[name] = state[f"{prefix}{part_name}.{name}"]
                module.load_state_dict(part_state, strict=True)
            bank.router.weight.copy_(state[prefix + "router.weight"])
            bank.router.bias.copy_(state[prefix + "router.bias"])
            bank.gate.copy_(state[prefix + "gate"].reshape(()))
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


# ---------------------------------------------------------------- go and stop rule


def go_stop(rows, skills, counts, arm="transplant_k4", challengers=("kd_lora", "transplant_k4_random")):
    """Apply the round 18 rule to merged rows. Return {"verdict", "passed", "skills": {skill: detail}}.

    A skill passes when the arm's mean score beats the mean of every
    challenger by more than the noise. The noise for one comparison is the
    largest of: the spread (max minus min) of the arm's seeds, the spread of
    the challenger's seeds, and one binomial standard error of the arm's mean
    score on that skill's item count. The verdict is "GO" when at least two
    skills pass, else "STOP".
    """
    scores = {}
    for row in rows:
        if row.get("scores") is None:
            continue
        for skill in skills:
            if skill in row["scores"]:
                scores.setdefault(row["arm"], {}).setdefault(skill, []).append(row["scores"][skill])
    detail = {}
    passed = 0
    for skill in skills:
        entry = {"passes": False, "vs": {}}
        if arm in scores and skill in scores[arm]:
            mine = scores[arm][skill]
            mean = sum(mine) / len(mine)
            se = (max(mean * (1.0 - mean), 1e-12) / counts[skill]) ** 0.5
            entry["mean"] = round(mean, 4)
            wins = 0
            for other in challengers:
                if other not in scores or skill not in scores[other]:
                    entry["vs"][other] = None
                    continue
                theirs = scores[other][skill]
                their_mean = sum(theirs) / len(theirs)
                noise = max(max(mine) - min(mine), max(theirs) - min(theirs), se)
                gap = mean - their_mean
                entry["vs"][other] = {"mean": round(their_mean, 4), "gap": round(gap, 4), "noise": round(noise, 4),
                                      "beats": gap > noise}
                if gap > noise:
                    wins += 1
            entry["passes"] = wins == len(challengers)
        if entry["passes"]:
            passed += 1
        detail[skill] = entry
    verdict = "GO" if passed >= 2 else "STOP"
    return {"verdict": verdict, "passed": passed, "skills": detail}
