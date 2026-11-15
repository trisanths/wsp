"""Skill removal ("slim"): delete the parts a model needs only for skills you do not want.

Sapien builds a model by choosing parts. Removing parts is as useful as
adding them. A model that must only write code does not need the experts,
layers, neurons, and heads that carry other skills. The result is one
smaller network that runs alone. The only training is a short heal.

Two mechanisms:

A. Expert removal for MoE models (slim_moe, slim_experts). Rank each
   layer's experts by how much the skill text uses them, keep the top N,
   and delete the rest: a shorter expert list and a smaller router. The
   model's own top-k still picks among the kept experts (capped at N).
   Ranking rules (RULES):
     freq       how often the router picks the expert on skill text
     saliency   sum over skill tokens of gate weight times the norm of the
                expert's output (the REAP idea)
     selective  log ratio of skill frequency to general frequency
     random     seeded control
     general    frequency on general text (control: keeps what general text uses)
   N is one count for all layers, or per layer to cover a mass fraction of
   the skill routing. With one N for every layer, the saved directory is a
   normal Hugging Face model with an updated config. With a different N per
   layer, save_slim writes the same files plus wsp_slim.json, and only
   load_slim rebuilds the model.

B. Dense slimming (slim_dense). Remove whole layers by their cost on skill
   text (wsp.atlas.layer_skill_map), then MLP neurons and attention heads by
   their activation on skill text (wsp.prune). Controls: "random" and
   "general". The result is a normal Hugging Face model with an updated
   config.

C. heal. Distillation from the original model's logits on skill text plus
   LM loss. LoRA (default r16) on the attention (and, for dense models, the
   MLP) trains, and for MoE models the kept routers train too. Everything
   else stays frozen. The LoRA merges into the weights at the end.

Families for expert removal (FAMILIES): qwen3_moe, qwen2_moe, olmoe,
mixtral, minimax (a softmax router that is an nn.Linear), and glm4_moe,
deepseek_v3 (a top-k router module, one routing group). Each family is one
small entry. Transformers 5.x fuses the experts into one module; slimming
that layout is not supported.

When a family does not renormalize the top-k weights (norm_topk_prob false
in OLMoE and Qwen2-MoE), the softmax runs over fewer experts after removal,
so the kept experts' weights change a little even for tokens that never
used a removed expert. Qwen3-MoE and Mixtral renormalize, and for them
removing experts a skill never routes to changes nothing.
"""

import copy
import json
import math
import os
import random
import time

import torch
from torch import nn

from wsp import atlas, carve, prune, skills_eval
from wsp.lora import apply_lora, lora_modules, merge_lora
from wsp.parts import _text_config, find_decoder_layers

RULES = ("freq", "saliency", "selective", "random", "general")
LAYER_RULES = ("skill_cost", "specific", "influence", "general", "random")
WIDTH_RULES = ("skill", "specific", "general", "random")
MANIFEST = "wsp_slim.json"
FORMAT = "wsp-slim-v1"
ATTENTION_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "qkv_proj")


# ---------------------------------------------------------------- families


class Family:
    """How one model family names its expert count and top-k, and how its router works."""

    def __init__(self, name, count_key, topk_key, router):
        self.name = name
        self.count_key = count_key
        self.topk_key = topk_key
        # "softmax": the router is an nn.Linear that returns logits.
        # "topk": the router is a module with a weight and a bias buffer that returns (indices, weights).
        self.router = router


FAMILIES = {}
for _family in [Family("qwen3_moe", "num_experts", "num_experts_per_tok", "softmax"),
                Family("qwen2_moe", "num_experts", "num_experts_per_tok", "softmax"),
                Family("olmoe", "num_experts", "num_experts_per_tok", "softmax"),
                Family("mixtral", "num_local_experts", "num_experts_per_tok", "softmax"),
                Family("minimax", "num_local_experts", "num_experts_per_tok", "softmax"),
                Family("glm4_moe", "n_routed_experts", "num_experts_per_tok", "topk"),
                Family("deepseek_v3", "n_routed_experts", "num_experts_per_tok", "topk")]:
    FAMILIES[_family.name] = _family


def family_of(model):
    """The Family entry of a model, or NotImplementedError."""
    config = _text_config(model.config)
    name = getattr(config, "model_type", None)
    if name not in FAMILIES:
        raise NotImplementedError(f"expert removal does not support model_type {name!r}; "
                                  f"it supports {', '.join(sorted(FAMILIES))}")
    return FAMILIES[name]


def moe_top_k(moe, router, default):
    for holder in (moe, router):
        value = getattr(holder, "top_k", None)
        if value is not None:
            return int(value)
    return default


def moe_layers_of(model):
    """[(layer index, moe module, router module, top_k, expert count)] for every MoE layer.

    Raises NotImplementedError when the experts are not a ModuleList.
    """
    found = atlas.find_routers(model)
    out = []
    for layer, moe, router, top_k, count in found:
        if not isinstance(moe.experts, nn.ModuleList):
            raise NotImplementedError("the experts are fused into one module (transformers 5.x layout); "
                                      "expert removal needs a ModuleList of experts")
        top_k = moe_top_k(moe, router, top_k)
        out.append((layer, moe, router, top_k, len(moe.experts)))
    return out


def route(family, moe, router_output, top_k):
    """(weights, indices), both (tokens, top_k): the gate weights and experts a router output picks."""
    if family.router == "topk":
        indices, weights = router_output
        return weights.float(), indices
    logits = router_output
    if isinstance(logits, (tuple, list)):
        logits = logits[0]
    probs = torch.softmax(logits.float(), dim=-1)
    weights, indices = torch.topk(probs, top_k, dim=-1)
    if getattr(moe, "norm_topk_prob", True):
        weights = weights / weights.sum(dim=-1, keepdim=True)
    return weights, indices


# ---------------------------------------------------------------- statistics


def collect_moe_stats(model, tokenizer, texts, max_length=256):
    """Per MoE layer, per expert: how often the router picks it and its saliency, on texts.

    Returns {"layers": {layer: {"count": tensor, "saliency": tensor, "tokens": int}}, "top_k", "num_experts"}.
    count[e] is the number of tokens that pick expert e (the same number as
    wsp.atlas.routing_counts). saliency[e] is the sum, over the tokens that
    pick e, of the gate weight times the L2 norm of e's output on that
    token. Texts run one at a time, so nothing is padded.
    """
    family = family_of(model)
    routers = moe_layers_of(model)
    if len(routers) == 0:
        raise ValueError("model has no MoE layers")
    device = next(model.parameters()).device
    layers = {}
    state = {}
    handles = []
    for layer, moe, router, top_k, count in routers:
        layers[layer] = {"count": torch.zeros(count, dtype=torch.float64),
                         "saliency": torch.zeros(count, dtype=torch.float64), "tokens": 0}
        state[layer] = {}

        def router_hook(module, args, output, layer=layer):
            state[layer]["router"] = output

        def pre_hook(module, args, kwargs, layer=layer):
            state[layer]["x"] = args[0]

        def post_hook(module, args, kwargs, output, layer=layer, moe=moe, top_k=top_k):
            x = state[layer]["x"].reshape(-1, state[layer]["x"].shape[-1])
            weights, indices = route(family, moe, state[layer]["router"], top_k)
            entry = layers[layer]
            entry["tokens"] += x.shape[0]
            for expert in torch.unique(indices).tolist():
                rows, cols = torch.where(indices == expert)
                out = moe.experts[expert](x[rows])
                entry["count"][expert] += rows.numel()
                gain = weights[rows, cols].float() * out.float().norm(dim=-1)
                entry["saliency"][expert] += float(gain.sum())
            state[layer].clear()

        handles.append(router.register_forward_hook(router_hook))
        handles.append(moe.register_forward_pre_hook(pre_hook, with_kwargs=True))
        handles.append(moe.register_forward_hook(post_hook, with_kwargs=True))
    was_training = model.training
    model.eval()
    try:
        for ids in atlas.encode_texts(tokenizer, texts, max_length):
            with torch.no_grad():
                model(input_ids=ids.unsqueeze(0).to(device), use_cache=False)
    finally:
        for handle in handles:
            handle.remove()
        if was_training:
            model.train()
    return {"layers": layers, "top_k": routers[0][3], "num_experts": routers[0][4]}


def stats_from_expert_map(expert_map, tokens=1):
    """Stats in the collect_moe_stats format from a wsp.atlas.expert_skill_map result.

    The atlas map holds frequencies, not saliency, so only the rules that
    need frequencies (freq, selective, general, random) work on it.
    """
    layers = {}
    for layer in expert_map["layers"]:
        freq = torch.tensor(expert_map["layers"][layer]["skill_freq"], dtype=torch.float64)
        layers[int(layer)] = {"count": freq * tokens, "saliency": None, "tokens": tokens}
    return {"layers": layers, "top_k": expert_map["top_k"], "num_experts": expert_map["num_experts"]}


def save_stats(stats, path):
    torch.save(stats, path)


def load_stats(path):
    return torch.load(path, weights_only=False)


# ---------------------------------------------------------------- ranking


def expert_scores(rule, skill, general=None, seed=0, eps=1e-4):
    """{layer: list of scores, higher means keep} for one ranking rule."""
    if rule not in RULES:
        raise ValueError(f"unknown rule {rule!r}; the rules are {', '.join(RULES)}")
    if rule in ("selective", "general") and general is None:
        raise ValueError(f"rule {rule!r} needs general stats")
    out = {}
    for layer in skill["layers"]:
        entry = skill["layers"][layer]
        count = entry["count"]
        total = max(1, entry["tokens"])
        if rule == "freq":
            values = (count / total).tolist()
        elif rule == "saliency":
            if entry["saliency"] is None:
                raise ValueError("these stats have no saliency")
            values = entry["saliency"].tolist()
        elif rule == "selective":
            other = general["layers"][layer]
            values = []
            for s_value, g_value in zip((count / total).tolist(), (other["count"] / max(1, other["tokens"])).tolist()):
                values.append(math.log((s_value + eps) / (g_value + eps)))
        elif rule == "general":
            other = general["layers"][layer]
            values = (other["count"] / max(1, other["tokens"])).tolist()
        else:
            generator = random.Random(seed * 1000003 + layer)
            values = []
            for _ in range(len(count)):
                values.append(generator.random())
        out[layer] = values
    return out


def rank_experts(rule, skill, general=None, seed=0):
    """{layer: expert ids, best first}. Ties go to the more used expert on skill text, then the lower id."""
    scores = expert_scores(rule, skill, general, seed)
    order = {}
    for layer in scores:
        values = scores[layer]
        counts = skill["layers"][layer]["count"].tolist()
        ids = list(range(len(values)))
        ids.sort(key=lambda expert: (-values[expert], -counts[expert], expert))
        order[layer] = ids
    return order


def choose_keep(order, skill, rule, keep=None, mass=None, min_keep=1):
    """{layer: sorted expert ids to keep}.

    keep: one count for every layer, or {layer: count}. mass: per layer, the
    fewest experts (in ranking order) that take this share of the skill
    routing, at least min_keep. For "saliency" the share is of saliency; for
    every other rule it is of the skill routing counts.
    """
    if (keep is None) == (mass is None):
        raise ValueError("give exactly one of keep and mass")
    kept = {}
    for layer in order:
        total = len(order[layer])
        if keep is not None:
            count = keep[layer] if isinstance(keep, dict) else keep
        else:
            entry = skill["layers"][layer]
            values = entry["count"]
            if rule == "saliency":
                values = entry["saliency"]
            values = values.tolist()
            whole = sum(values)
            count = min_keep
            running = 0.0
            if whole > 0:
                for position, expert in enumerate(order[layer]):
                    running += values[expert]
                    if running >= mass * whole:
                        count = max(min_keep, position + 1)
                        break
                else:
                    count = total
        count = int(count)
        if count < 1 or count > total:
            raise ValueError(f"layer {layer}: cannot keep {count} of {total} experts")
        kept[layer] = sorted(order[layer][:count])
    return kept


# ---------------------------------------------------------------- parameter counts


def count_params(model):
    """(params total, params run per token).

    The total counts each distinct parameter once. Run per token follows
    wsp.skills_eval.model_params (the input embedding is a lookup, so it does
    not count unless the head ties it), minus the experts a token does not
    pick: per MoE layer, (experts - top_k) times one expert's size.
    """
    total, run = skills_eval.model_params(model)
    for layer, moe, router, top_k, count in moe_layers_of(model):
        size = 0
        for param in moe.experts[0].parameters():
            size += param.numel()
        run -= (count - min(top_k, count)) * size
    return total, run


# ---------------------------------------------------------------- expert removal


def structure_copy(model, share_experts=False):
    """A deep copy of the model. With share_experts, the experts stay the same objects, not copies.

    A student that shares the teacher's experts costs only the memory of
    the non-expert weights. Use it only when the experts stay frozen.
    """
    memo = {}
    if share_experts:
        for layer, moe, router, top_k, count in moe_layers_of(model):
            for expert in moe.experts:
                memo[id(expert)] = expert
    return copy.deepcopy(model, memo)


def slice_router(family, moe, router, ids):
    """Replace the router's rows with only the rows of ids. Returns the new router module."""
    if family.router == "softmax":
        new = prune.slice_linear(router, rows=ids)
        moe.gate = new
        return new
    config = getattr(moe, "config", None)
    group = getattr(router, "n_group", 1)
    if group != 1:
        raise NotImplementedError("expert removal supports routers with one routing group (n_group 1)")
    index = torch.tensor(ids, dtype=torch.long, device=router.weight.device)
    weight = nn.Parameter(router.weight.detach().index_select(0, index).clone(),
                          requires_grad=router.weight.requires_grad)
    router.weight = weight
    bias = getattr(router, "e_score_correction_bias", None)
    if bias is not None:
        router.e_score_correction_bias = bias.detach().index_select(0, index).clone()
    router.n_routed_experts = len(ids)
    return router


def slim_experts(model, keep, in_place=False, share_experts=False):
    """Delete every expert not in keep ({layer: expert ids}). Returns (model, report).

    The kept experts keep their order. Each router keeps its rows for the
    kept experts, and top_k becomes min(top_k, kept count). When every layer
    keeps the same count, the config follows (expert count and top-k), so
    save_pretrained writes a normal Hugging Face model.
    """
    family = family_of(model)
    total_before, run_before = count_params(model)
    layers_before = moe_layers_of(model)
    for layer, moe, router, top_k, count in layers_before:
        if layer not in keep:
            raise ValueError(f"keep has no entry for MoE layer {layer}")
        ids = keep[layer]
        if len(ids) == 0 or len(set(ids)) != len(ids) or min(ids) < 0 or max(ids) >= count:
            raise ValueError(f"layer {layer}: bad expert ids {ids[:8]} for {count} experts")
    target = model
    if not in_place:
        target = structure_copy(model, share_experts)
    top_ks = {}
    counts = {}
    kept = {}
    for layer, moe, router, top_k, count in moe_layers_of(target):
        ids = sorted(int(i) for i in keep[layer])
        moe.experts = nn.ModuleList([moe.experts[i] for i in ids])
        slice_router(family, moe, router, ids)
        new_k = min(top_k, len(ids))
        for holder in (moe, moe.gate if family.router == "softmax" else router):
            if hasattr(holder, "num_experts"):
                holder.num_experts = len(ids)
            if hasattr(holder, "top_k"):
                holder.top_k = new_k
        top_ks[layer] = new_k
        counts[layer] = len(ids)
        kept[layer] = ids
    uniform = len(set(counts.values())) == 1
    # The model object keeps its own copies of the counts (used by the load-balancing loss).
    if hasattr(target, "num_experts"):
        target.num_experts = max(counts.values())
    if hasattr(target, "num_experts_per_tok"):
        target.num_experts_per_tok = max(top_ks.values())
    config = _text_config(target.config)
    original_top_k = getattr(config, family.topk_key)
    if uniform:
        setattr(config, family.count_key, counts[next(iter(counts))])
        setattr(config, family.topk_key, min(original_top_k, counts[next(iter(counts))]))
    total_after, run_after = count_params(target)
    report = {"kind": "moe", "uniform": uniform, "family": family.name,
              "params_total_before": total_before, "params_total_after": total_after,
              "params_run_before": run_before, "params_run_after": run_after,
              "kept": kept, "counts": counts, "top_k": top_ks}
    target.wsp_slim = report
    return target, report


def slim_moe(model, tokenizer, skill_texts, general_texts=None, rule="saliency", keep=None, mass=None, seed=0,
             skill_stats=None, general_stats=None, max_length=256, min_keep=1, in_place=False, share_experts=False):
    """Rank experts on skill text and delete the rest. Returns (model, report).

    skill_stats and general_stats may come from collect_moe_stats (or a
    cache); otherwise they are measured here. The report adds the rule, the
    seed, and the ranking to the slim_experts report.
    """
    if skill_stats is None:
        skill_stats = collect_moe_stats(model, tokenizer, skill_texts, max_length)
    if general_stats is None and rule in ("selective", "general"):
        general_stats = collect_moe_stats(model, tokenizer, general_texts, max_length)
    order = rank_experts(rule, skill_stats, general_stats, seed)
    chosen = choose_keep(order, skill_stats, rule, keep=keep, mass=mass, min_keep=min_keep)
    slim, report = slim_experts(model, chosen, in_place=in_place, share_experts=share_experts)
    report["rule"] = rule
    report["seed"] = seed
    report["ranking"] = order
    return slim, report


def kept_share(stats, kept):
    """Per layer, the share of skill routing counts that the kept experts take."""
    out = {}
    for layer in kept:
        counts = stats["layers"][layer]["count"].tolist()
        total = sum(counts)
        taken = 0.0
        for expert in kept[layer]:
            taken += counts[expert]
        out[layer] = taken / total if total > 0 else 0.0
    return out


# ---------------------------------------------------------------- dense: layers


def layer_costs(model, tokenizer, skill_texts, general_texts, max_length=256, batch_size=8):
    """wsp.atlas.layer_skill_map: per layer, the loss increase on skill and general text when it is skipped."""
    return atlas.layer_skill_map(model, tokenizer, skill_texts, general_texts, max_length=max_length,
                                 batch_size=batch_size)


def layer_values(costs, rule):
    """Per layer values where a LOWER value means remove first."""
    if rule == "skill_cost":
        return list(costs["skill_increase"])
    if rule == "specific":
        return list(costs["scores"])
    if rule == "influence":
        return list(costs["skill_influence"])
    if rule == "general":
        return list(costs["general_increase"])
    raise ValueError(f"unknown layer rule {rule!r}; the rules are {', '.join(LAYER_RULES)}")


def pick_layers(costs, total, count, rule="skill_cost", seed=0, protect_ends=True):
    """The sorted indices of the `count` layers to remove.

    skill_cost: lowest loss increase on skill text when the layer is skipped.
    specific: lowest (skill increase minus general increase).
    influence: lowest block influence on skill text.
    general: lowest loss increase on general text (control).
    random: seeded (control). The first and last layer stay in every rule
    when protect_ends is true, so the random control is not a straw man.
    """
    if rule not in LAYER_RULES:
        raise ValueError(f"unknown layer rule {rule!r}; the rules are {', '.join(LAYER_RULES)}")
    candidates = list(range(total))
    if protect_ends and total > 2:
        candidates = candidates[1:-1]
    if count > len(candidates):
        raise ValueError(f"cannot remove {count} of {total} layers")
    if rule == "random":
        generator = random.Random(seed)
        values = {}
        for index in candidates:
            values[index] = generator.random()
    else:
        raw = layer_values(costs, rule)
        values = {}
        for index in candidates:
            values[index] = raw[index]
    candidates.sort(key=lambda index: (values[index], index))
    return sorted(candidates[:count])


def skipped_loss(model, encoded, skipped, batch_size=8):
    """Mean next-token loss on encoded texts with the layers in skipped made passthrough."""
    _, layers = find_decoder_layers(model, 0)
    handles = []
    for index in skipped:
        handles.append(layers[index].register_forward_hook(atlas.skip_hook, with_kwargs=True))
    try:
        return atlas.pass_stats(model, encoded, batch_size)["loss"]
    finally:
        for handle in handles:
            handle.remove()


def greedy_layers(model, tokenizer, skill_texts, count, step=1, max_length=256, batch_size=8, protect_ends=True):
    """Remove layers a few at a time, scoring again after each step.

    Each round skips every remaining layer once on top of the layers already
    skipped, and removes the `step` layers whose skip raises the skill loss
    least. One-shot costs ignore that layers interact; this does not.
    Returns the sorted indices.
    """
    _, layers = find_decoder_layers(model, 0)
    encoded = atlas.encode_texts(tokenizer, skill_texts, max_length)
    removed = []
    while len(removed) < count:
        base = skipped_loss(model, encoded, removed, batch_size)
        costs = []
        for index in range(len(layers)):
            if index in removed:
                continue
            if protect_ends and len(layers) > 2 and index in (0, len(layers) - 1):
                continue
            costs.append((skipped_loss(model, encoded, removed + [index], batch_size) - base, index))
        costs.sort()
        take = min(step, count - len(removed))
        for _, index in costs[:take]:
            removed.append(index)
    return sorted(removed)


def drop_layers(model, drop):
    """Delete the layers in drop, in place, with wsp.carve.carve_layers. Returns the model."""
    _, layers = find_decoder_layers(model, 0)
    keep = []
    for index in range(len(layers)):
        if index not in drop:
            keep.append(index)
    return carve.carve_layers(model, keep)


# ---------------------------------------------------------------- dense: size plan


def plan_size(model, target_params, max_layer_fraction=0.25):
    """(layers to remove, width keep fraction) that reach about target_params.

    Layers go first, up to floor(max_layer_fraction * layers). If the model
    still has more than target_params, MLP neurons and query heads thin by
    one fraction until the count is closest to the target. Raises
    ValueError when even the thinnest model is above the target.
    """
    _, layers = find_decoder_layers(model, 0)
    total = len(layers)
    cap = int(math.floor(max_layer_fraction * total))
    for drop in range(0, cap + 1):
        keep = list(range(total - drop))
        if carve.params_for_layers(model, keep, 1.0, 1.0) <= target_params:
            return drop, 1.0
    keep = list(range(total - cap))
    floor_params = carve.params_for_layers(model, keep, 1e-6, 1e-6)
    if floor_params > target_params:
        raise ValueError(f"even the thinnest model with {len(keep)} layers has {floor_params} parameters, "
                         f"above the target {target_params}")
    low = 1e-6
    high = 1.0
    best = (abs(floor_params - target_params), low)
    for _ in range(40):
        middle = (low + high) / 2
        size = carve.params_for_layers(model, keep, middle, middle)
        gap = abs(size - target_params)
        if gap < best[0]:
            best = (gap, middle)
        if size <= target_params:
            low = middle
        else:
            high = middle
    return cap, best[1]


# ---------------------------------------------------------------- dense: slimming


def width_scores(model, tokenizer, skill_texts, general_texts, rule, max_length=256, batch_size=8):
    """Unit scores in the wsp.prune format for a width rule (None for random)."""
    if rule == "random":
        return None
    if rule == "skill":
        return carve.unit_scores(model, tokenizer, skill_texts, max_tokens=max_length, batch_size=batch_size)
    if rule == "general":
        return carve.unit_scores(model, tokenizer, general_texts, max_tokens=max_length, batch_size=batch_size)
    if rule == "specific":
        _, layers = find_decoder_layers(model, 0)
        return prune.domain_unit_scores(model, tokenizer, skill_texts, general_texts, len(layers) - 1,
                                        max_tokens=max_length, batch_size=batch_size)
    raise ValueError(f"unknown width rule {rule!r}; the rules are {', '.join(WIDTH_RULES)}")


def slim_dense(model, tokenizer, skill_texts, general_texts, target_params=None, n_drop=None, keep_mlp=1.0,
               keep_heads=1.0, max_layer_fraction=0.25, layer_rule="skill_cost", width_rule="skill", seed=0,
               greedy_step=None, costs=None, max_length=256, batch_size=8, in_place=False):
    """Remove layers, then MLP neurons and attention heads, by skill statistics. Returns (model, report).

    Give target_params (see plan_size) or n_drop with keep_mlp and
    keep_heads. layer_rule is one of LAYER_RULES; with greedy_step set (and
    layer_rule "skill_cost") layers are chosen by greedy_layers. width_rule
    is one of WIDTH_RULES. "random" in either rule is the control: same
    counts, random units. The first and last layer always stay.
    """
    _, layers = find_decoder_layers(model, 0)
    total_layers = len(layers)
    cap = int(math.floor(max_layer_fraction * total_layers))
    if target_params is not None:
        n_drop, fraction = plan_size(model, target_params, max_layer_fraction)
        keep_mlp = fraction
        keep_heads = fraction
    if n_drop is None:
        n_drop = 0
    if n_drop > cap:
        raise ValueError(f"n_drop {n_drop} is above the cap of {cap} layers ({max_layer_fraction} of {total_layers})")
    if layer_rule not in LAYER_RULES or width_rule not in WIDTH_RULES:
        raise ValueError(f"bad rules: layer {layer_rule!r}, width {width_rule!r}")
    total_before, run_before = count_params(model)
    target = model
    if not in_place:
        target = copy.deepcopy(model)

    dropped = []
    if n_drop > 0:
        if layer_rule == "skill_cost" and greedy_step is not None:
            dropped = greedy_layers(target, tokenizer, skill_texts, n_drop, step=greedy_step,
                                    max_length=max_length, batch_size=batch_size)
        else:
            if costs is None and layer_rule != "random":
                costs = layer_costs(target, tokenizer, skill_texts, general_texts, max_length, batch_size)
            dropped = pick_layers(costs, total_layers, n_drop, layer_rule, seed)
        drop_layers(target, dropped)

    width_report = None
    if keep_mlp < 1.0 or keep_heads < 1.0:
        scores = width_scores(target, tokenizer, skill_texts, general_texts, width_rule, max_length, batch_size)
        keep_rule = "random" if width_rule == "random" else "top"
        _, kept_layers = find_decoder_layers(target, 0)
        target, width_report = prune.prune_donor(target, scores, keep_mlp, keep_heads, len(kept_layers) - 1,
                                                 keep_rule=keep_rule, seed=seed, in_place=True)
    total_after, run_after = count_params(target)
    report = {"kind": "dense", "uniform": True, "layer_rule": layer_rule, "width_rule": width_rule, "seed": seed,
              "layers_before": total_layers, "layers_after": total_layers - len(dropped), "dropped": dropped,
              "keep_mlp": keep_mlp, "keep_heads": keep_heads, "target_params": target_params,
              "params_total_before": total_before, "params_total_after": total_after,
              "params_run_before": run_before, "params_run_after": run_after, "width": width_report}
    target.wsp_slim = report
    return target, report


# ---------------------------------------------------------------- heal


class Fp32Gate(nn.Module):
    """A router that keeps its weight in float32 so Adam can move it. Logits come out in float32."""

    def __init__(self, linear):
        super().__init__()
        self.weight = nn.Parameter(linear.weight.detach().to(torch.float32).clone())
        self.out_dtype = linear.weight.dtype

    def forward(self, x):
        return torch.nn.functional.linear(x.to(torch.float32), self.weight)

    def to_linear(self):
        weight = self.weight.detach().to(self.out_dtype)
        new = nn.Linear(weight.shape[1], weight.shape[0], bias=False, device=weight.device, dtype=weight.dtype)
        with torch.no_grad():
            new.weight.copy_(weight)
        return new


def prepare_routers(model, family):
    """Make the kept routers trainable. Returns (params, restore function)."""
    params = []
    wrapped = []
    cast = []
    for layer, moe, router, top_k, count in moe_layers_of(model):
        if family.router == "softmax":
            gate = Fp32Gate(router)
            moe.gate = gate
            wrapped.append((moe, gate))
            params.append(gate.weight)
        else:
            cast.append((router, router.weight.dtype))
            router.weight.data = router.weight.data.to(torch.float32)
            router.weight.requires_grad_(True)
            params.append(router.weight)

    def restore():
        for moe, gate in wrapped:
            moe.gate = gate.to_linear()
        for router, dtype in cast:
            router.weight.data = router.weight.data.to(dtype)

    return params, restore


def distill_loss(student_logits, teacher_logits, input_ids, mask, temperature=1.0, kd_weight=1.0, lm_weight=1.0):
    """(total, kd, lm): KL(teacher || student) per token plus next-token cross entropy."""
    vocab = min(teacher_logits.shape[-1], student_logits.shape[-1])
    t = torch.log_softmax(teacher_logits[..., :vocab] / temperature, dim=-1)
    s = torch.log_softmax(student_logits[..., :vocab] / temperature, dim=-1)
    weights = mask.float()
    kl = (t.exp() * (t - s)).sum(-1)
    kd = (kl * weights).sum() / weights.sum()
    labels = input_ids[:, 1:]
    lm = torch.nn.functional.cross_entropy(student_logits[:, :-1].reshape(-1, student_logits.shape[-1]),
                                           labels.reshape(-1), reduction="none")
    lm_mask = weights[:, 1:].reshape(-1)
    lm = (lm * lm_mask).sum() / lm_mask.sum()
    return kd_weight * kd + lm_weight * lm, kd, lm


def heal(student, teacher, tokenizer, texts, steps, batch_size=8, seq_len=384, lora_rank=16, lora_lr=2e-4,
         router_lr=1e-4, kd_weight=1.0, lm_weight=1.0, temperature=1.0, seed=0, warmup=20, merge=True, train_routers=True,
         targets=None, log_every=50):
    """Heal a slimmed model toward the original model's logits. Returns a report dict.

    Loss: KL(teacher || student) plus the LM loss, on texts. Trainable: a
    LoRA on the attention linears (and the MLP linears of a dense model),
    and the kept routers of a MoE model. Everything else is frozen. teacher
    may be None (LM loss only). With merge, the LoRA folds into the weights
    at the end, so the model saves as a normal Hugging Face model.
    """
    if steps < 1:
        raise ValueError("steps must be at least 1")
    started = time.time()
    torch.manual_seed(seed)
    is_moe = len(atlas.find_routers(student)) > 0
    if targets is None:
        targets = ATTENTION_TARGETS if is_moe else None
    for param in student.parameters():
        param.requires_grad_(False)
    lora_params = apply_lora(student, lora_rank, targets=targets) if targets is not None else apply_lora(
        student, lora_rank)
    router_params = []
    restore = None
    if is_moe and train_routers:
        router_params, restore = prepare_routers(student, family_of(student))
    groups = [{"params": lora_params, "lr": lora_lr}]
    if len(router_params) > 0:
        groups.append({"params": router_params, "lr": router_lr})
    optimizer = torch.optim.AdamW(groups, weight_decay=0.0)
    base_lrs = [group["lr"] for group in groups]

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    device = next(student.parameters()).device
    use_amp = device.type == "cuda"
    order = list(range(len(texts)))
    random.Random(seed).shuffle(order)
    student.train()
    if teacher is not None:
        teacher.eval()
    losses = []
    position = 0
    for step in range(steps):
        batch = []
        while len(batch) < batch_size:
            batch.append(texts[order[position % len(order)]])
            position += 1
        encoded = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=seq_len)
        enc = {"input_ids": encoded["input_ids"].to(device), "attention_mask": encoded["attention_mask"].to(device)}
        scale = min(1.0, (step + 1) / max(1, warmup))
        for group, base in zip(optimizer.param_groups, base_lrs):
            group["lr"] = base * scale
        teacher_logits = None
        if teacher is not None:
            with torch.no_grad():
                teacher_logits = teacher(**enc, use_cache=False).logits.float()
        if use_amp:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                student_logits = student(**enc, use_cache=False).logits
        else:
            student_logits = student(**enc, use_cache=False).logits
        student_logits = student_logits.float()
        if teacher_logits is None:
            teacher_logits = student_logits.detach()
        loss, kd, lm = distill_loss(student_logits, teacher_logits, enc["input_ids"], enc["attention_mask"],
                                    temperature, kd_weight, lm_weight)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(lora_params + router_params, 1.0)
        optimizer.step()
        losses.append([float(kd.detach()), float(lm.detach())])
        if log_every and step % log_every == 0:
            print(f"heal step {step} kd {float(kd.detach()):.4f} lm {float(lm.detach()):.4f}")
    student.eval()
    trainable = 0
    for param in lora_params + router_params:
        trainable += param.numel()
    if restore is not None:
        restore()
    if merge:
        merge_lora(student)
    window = min(50, steps)
    report = {"steps": steps, "batch_size": batch_size, "seq_len": seq_len, "lora_rank": lora_rank,
              "trainable_params": trainable, "router_params": sum(p.numel() for p in router_params),
              "kd_first": sum(row[0] for row in losses[:window]) / window,
              "kd_last": sum(row[0] for row in losses[-window:]) / window,
              "lm_first": sum(row[1] for row in losses[:window]) / window,
              "lm_last": sum(row[1] for row in losses[-window:]) / window,
              "seconds": round(time.time() - started, 1)}
    return report


# ---------------------------------------------------------------- text helpers


def drop_overlap(texts, keys):
    """The texts that contain none of the keys (strings from the scoring items), and how many were dropped."""
    clean = []
    dropped = 0
    lowered = []
    for key in keys:
        key = " ".join(str(key).split()).lower()
        if len(key) >= 30:
            lowered.append(key)
    for text in texts:
        flat = " ".join(text.split()).lower()
        hit = False
        for key in lowered:
            if key in flat:
                hit = True
                break
        if hit:
            dropped += 1
        else:
            clean.append(text)
    return clean, dropped


# ---------------------------------------------------------------- save and load


def directory_bytes(path):
    """Total size in bytes of the files under path."""
    total = 0
    for root, _, names in os.walk(path):
        for name in names:
            total += os.path.getsize(os.path.join(root, name))
    return total


def save_slim(model, path, tokenizer=None):
    """Write the model with save_pretrained, plus wsp_slim.json. Returns the manifest.

    When every layer keeps the same expert count (or the model is dense),
    the directory loads with AutoModelForCausalLM and no wsp code. With a
    different count per layer, only load_slim rebuilds it ("plain_hf" is
    false in the manifest).
    """
    info = getattr(model, "wsp_slim", None)
    if info is None:
        raise ValueError("this model was not slimmed; use slim_moe or slim_dense first")
    if len(lora_modules(model)) > 0:
        raise ValueError("the model still holds a LoRA; heal with merge=True or call merge_lora first")
    os.makedirs(path, exist_ok=True)
    model.save_pretrained(path)
    if tokenizer is not None:
        tokenizer.save_pretrained(path)
    manifest = {"format": FORMAT, "kind": info["kind"], "plain_hf": bool(info["uniform"])}
    if info["kind"] == "moe":
        manifest["family"] = info["family"]
        manifest["kept"] = {str(layer): ids for layer, ids in info["kept"].items()}
        manifest["counts"] = {str(layer): n for layer, n in info["counts"].items()}
        manifest["top_k"] = {str(layer): k for layer, k in info["top_k"].items()}
    for key in ["params_total_before", "params_total_after", "params_run_before", "params_run_after"]:
        manifest[key] = info[key]
    with open(os.path.join(path, MANIFEST), "w") as f:
        json.dump(manifest, f, indent=2)
    return manifest


def read_state(path):
    """All tensors of the safetensors files in a model directory."""
    from safetensors.torch import load_file

    state = {}
    for name in sorted(os.listdir(path)):
        if name.endswith(".safetensors"):
            state.update(load_file(os.path.join(path, name)))
    return state


def load_slim(path, dtype=torch.bfloat16, device="cpu"):
    """Rebuild a model saved by save_slim, including one with a different expert count per layer."""
    import transformers

    with open(os.path.join(path, MANIFEST)) as f:
        manifest = json.load(f)
    if manifest.get("format") != FORMAT:
        raise ValueError(f"{path} is not a {FORMAT} directory")
    if manifest["plain_hf"]:
        return transformers.AutoModelForCausalLM.from_pretrained(path, dtype=dtype).to(device).eval()
    config = transformers.AutoConfig.from_pretrained(path)
    family = FAMILIES[manifest["family"]]
    text = _text_config(config)
    # Build with one expert per layer, then grow each layer to its saved count.
    setattr(text, family.count_key, 1)
    setattr(text, family.topk_key, 1)
    model = transformers.AutoModelForCausalLM.from_config(config, dtype=dtype)
    for layer, moe, router, top_k, count in moe_layers_of(model):
        n = int(manifest["counts"][str(layer)])
        k = int(manifest["top_k"][str(layer)])
        moe.experts = nn.ModuleList([copy.deepcopy(moe.experts[0]) for _ in range(n)])
        if family.router == "softmax":
            weight = router.weight
            moe.gate = nn.Linear(weight.shape[1], n, bias=False, device=weight.device, dtype=weight.dtype)
        else:
            router.weight = nn.Parameter(torch.zeros((n, router.weight.shape[1]), dtype=router.weight.dtype))
            router.e_score_correction_bias = torch.zeros(n, dtype=torch.float32)
            router.n_routed_experts = n
        for holder in (moe, moe.gate if family.router == "softmax" else router):
            if hasattr(holder, "num_experts"):
                holder.num_experts = n
            if hasattr(holder, "top_k"):
                holder.top_k = k
    state = read_state(path)
    result = model.load_state_dict(state, strict=False)
    missing = []
    for name in result.missing_keys:
        if name != "lm_head.weight":
            missing.append(name)
    if len(missing) > 0 or len(result.unexpected_keys) > 0:
        raise ValueError(f"state mismatch: missing {missing[:5]}, unexpected {list(result.unexpected_keys)[:5]}")
    model.tie_weights()
    counts = {int(l): int(n) for l, n in manifest["counts"].items()}
    top_ks = {int(l): int(k) for l, k in manifest["top_k"].items()}
    # Restore the real config: the largest count and top-k, as slim_experts leaves it.
    setattr(text, family.count_key, max(counts.values()))
    setattr(text, family.topk_key, max(top_ks.values()))
    if hasattr(model, "num_experts"):
        model.num_experts = max(counts.values())
    if hasattr(model, "num_experts_per_tok"):
        model.num_experts_per_tok = max(top_ks.values())
    total_after, run_after = count_params(model)
    info = {"kind": "moe", "uniform": False, "plain_hf": False, "family": family.name,
            "kept": {int(l): ids for l, ids in manifest["kept"].items()}, "counts": counts, "top_k": top_ks,
            "params_total_after": total_after, "params_run_after": run_after}
    for key in ["params_total_before", "params_run_before"]:
        info[key] = manifest[key]
    model.wsp_slim = info
    return model.to(device).eval()
