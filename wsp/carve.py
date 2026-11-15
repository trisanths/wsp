"""Write one standard model from the parts of existing models.

A written model is a plain Hugging Face model of the source's own class.
No donor runs beside it, and save_pretrained writes a directory that
from_pretrained (and so vLLM) loads with no wsp code.

Tools:

- layer_influence scores each decoder layer by how much it changes the
  residual stream: 1 - cosine(layer input, layer output), averaged over
  tokens (ShortGPT block influence, 2024).
- domain_influence gives that score on domain text and on general text,
  and their difference.
- choose_layers picks the layers to keep: "influence" (the highest
  scores, always the first and last layer), "uniform" (evenly spaced),
  or "domain" (a weighted sum of domain and general influence).
- params_for_layers and layers_for_budget count parameters, so a carve
  meets a size budget. Embeddings (tied or not) and norms count.
- carve_layers keeps a list of layers and re-indexes them and the config.
- carve_width slices MLP neurons and attention heads in every layer with
  wsp.prune, and updates the config.
- splice builds one model from layer segments of several models of the
  same architecture and hidden size.

Example:

    scores = layer_influence(model, tokenizer, texts)
    keep = layers_for_budget(model, 7.6e9)
    carve_layers(model, choose_layers(scores, keep, "influence"))
    model.save_pretrained(out_dir)
"""

import copy
import math

import torch
from torch import nn

from wsp.parts import _text_config, find_decoder_layers
from wsp.prune import (attention_layout, collect_unit_stats, head_counts, head_shape, keep_count, mlp_layout,
                       mlp_width, module_params, parent_and_name, prune_donor)

SELECT_RULES = ("influence", "uniform", "domain")

# Config lists that hold layer indices, not one entry per layer.
INDEX_LISTS = ("mlp_only_layers", "moe_layers")

# Config lists that can have the same length as the layer list by chance.
NOT_PER_LAYER = ("architectures", "eos_token_id", "bos_token_id", "pad_token_id", "bad_words_ids",
                 "suppress_tokens", "begin_suppress_tokens", "forced_decoder_ids")


def count_params(model):
    """All parameters of a model. A tied embedding and head count once."""
    total = 0
    for param in model.parameters():
        total += param.numel()
    return total


def model_device(model):
    return next(model.parameters()).device


# Layer scores


def layer_influence(model, tokenizer, texts, max_length=512):
    """Return the mean (1 - cosine(input, output)) of each decoder layer over the tokens of texts.

    Each text runs alone, so there is no padding. The mean is over all
    tokens of all texts.
    """
    _, layers = find_decoder_layers(model, 0)
    sums = []
    counts = []
    for index in range(len(layers)):
        sums.append(0.0)
        counts.append(0)
    store = {}
    handles = []
    for index in range(len(layers)):
        layer = layers[index]

        def pre(module, args, kwargs, index=index):
            if len(args) > 0:
                store[index] = args[0].detach()
            else:
                store[index] = kwargs["hidden_states"].detach()

        def post(module, args, kwargs, output, index=index):
            out = output
            if isinstance(output, tuple):
                out = output[0]
            cos = torch.nn.functional.cosine_similarity(store[index].float(), out.detach().float(), dim=-1)
            sums[index] += float((1 - cos).sum())
            counts[index] += cos.numel()

        handles.append(layer.register_forward_pre_hook(pre, with_kwargs=True))
        handles.append(layer.register_forward_hook(post, with_kwargs=True))

    was_training = model.training
    model.eval()
    device = model_device(model)
    try:
        for text in texts:
            ids = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length).input_ids
            if ids.shape[1] == 0:
                continue
            with torch.no_grad():
                model(ids.to(device), use_cache=False)
    finally:
        for handle in handles:
            handle.remove()
        if was_training:
            model.train()

    influence = []
    for index in range(len(layers)):
        influence.append(sums[index] / max(1, counts[index]))
    return influence


def domain_influence(model, tokenizer, domain_texts, general_texts, max_length=512):
    """Return {"domain", "general", "difference"}: per-layer influence on each text set, and domain - general."""
    if len(domain_texts) == 0 or len(general_texts) == 0:
        raise ValueError("domain_influence needs domain texts and general texts")
    domain = layer_influence(model, tokenizer, domain_texts, max_length)
    general = layer_influence(model, tokenizer, general_texts, max_length)
    difference = []
    for index in range(len(domain)):
        difference.append(domain[index] - general[index])
    return {"domain": domain, "general": general, "difference": difference}


def score_count(scores):
    """The number of layers that a list of scores or a domain_influence dict covers."""
    if isinstance(scores, dict):
        return len(scores["domain"])
    if isinstance(scores, int):
        return scores
    return len(scores)


def choose_layers(scores, keep, rule="influence", weight=0.75):
    """Return the sorted layer indices to keep.

    scores: a list from layer_influence, a dict from domain_influence (for
        "domain"), or for "uniform" only, the layer count.
    keep: how many layers to keep, 2 or more.
    rule:
      "influence"  keep the first and last layer and the keep - 2 middle
                   layers with the highest scores.
      "uniform"    keep keep layers evenly spaced from the first to the last.
      "domain"     like "influence", but rank by
                   weight * domain + (1 - weight) * general.
    """
    if rule not in SELECT_RULES:
        raise ValueError(f"unknown select rule {rule!r}; the rules are {', '.join(SELECT_RULES)}")
    total = score_count(scores)
    if isinstance(keep, bool) or not isinstance(keep, int) or keep < 2:
        raise ValueError(f"keep is an integer of 2 or more, got {keep!r}")
    if keep >= total:
        return list(range(total))

    if rule == "uniform":
        chosen = []
        for step in range(keep):
            # Round half up. The spacing is at least 1, so no index repeats.
            chosen.append(int(math.floor(step * (total - 1) / (keep - 1) + 0.5)))
        return chosen

    if rule == "domain":
        if not isinstance(scores, dict):
            raise ValueError("the domain rule needs the dict from domain_influence")
        if not (0 <= weight <= 1):
            raise ValueError(f"the domain weight is in [0, 1], got {weight}")
        values = []
        for index in range(total):
            values.append(weight * scores["domain"][index] + (1 - weight) * scores["general"][index])
    else:
        if isinstance(scores, dict):
            values = scores["general"]
        else:
            values = list(scores)

    middle = list(range(1, total - 1))
    middle.sort(key=lambda index: values[index], reverse=True)
    chosen = [0, total - 1]
    for index in middle[:keep - 2]:
        chosen.append(index)
    chosen.sort()
    return chosen


# Parameter budgets


def linear_count(linear, rows, cols):
    """The parameters of a linear sliced to rows x cols."""
    total = rows * cols
    if linear.bias is not None:
        total += rows
    return total


def layer_params_after_width(layer, keep_mlp=1.0, keep_heads=1.0):
    """The parameter count of a decoder layer after carve_width, without changing it."""
    total = module_params(layer)
    if keep_mlp == 1.0 and keep_heads == 1.0:
        return total
    attn, attn_kind = attention_layout(layer)
    mlp, mlp_kind = mlp_layout(layer)
    n_q, n_kv, head_dim = head_counts(attn, attn_kind)
    kv_keep, per_group = head_shape(n_q, n_kv, keep_heads)
    q_rows = kv_keep * per_group * head_dim
    kv_rows = kv_keep * head_dim
    if attn_kind == "split":
        for name, rows in [("q_proj", q_rows), ("k_proj", kv_rows), ("v_proj", kv_rows)]:
            linear = getattr(attn, name)
            total += linear_count(linear, rows, linear.in_features) - module_params(linear)
    else:
        linear = attn.qkv_proj
        total += linear_count(linear, q_rows + 2 * kv_rows, linear.in_features) - module_params(linear)
    total += linear_count(attn.o_proj, attn.o_proj.out_features, q_rows) - module_params(attn.o_proj)

    neurons = keep_count(keep_mlp, mlp_width(mlp, mlp_kind))
    if mlp_kind == "split":
        for name in ["gate_proj", "up_proj"]:
            linear = getattr(mlp, name)
            total += linear_count(linear, neurons, linear.in_features) - module_params(linear)
    else:
        linear = mlp.gate_up_proj
        total += linear_count(linear, 2 * neurons, linear.in_features) - module_params(linear)
    total += linear_count(mlp.down_proj, mlp.down_proj.out_features, neurons) - module_params(mlp.down_proj)
    return total


def params_for_layers(model, keep_layers, keep_mlp=1.0, keep_heads=1.0):
    """The parameter count of the model with only keep_layers (and the width fractions, if given)."""
    _, layers = find_decoder_layers(model, 0)
    total = count_params(model)
    for layer in layers:
        total -= module_params(layer)
    for index in keep_layers:
        total += layer_params_after_width(layers[index], keep_mlp, keep_heads)
    return total


def layers_for_budget(model, target_params, keep_mlp=1.0, keep_heads=1.0):
    """Return the largest layer count whose model has at most target_params parameters.

    The count holds whichever layers are kept: it adds the largest layers
    first. Raises ValueError when even 2 layers do not fit.
    """
    _, layers = find_decoder_layers(model, 0)
    total = count_params(model)
    sizes = []
    for layer in layers:
        total -= module_params(layer)
        sizes.append(layer_params_after_width(layer, keep_mlp, keep_heads))
    sizes.sort(reverse=True)
    outside = total
    keep = 0
    for size in sizes:
        if total + size > target_params:
            break
        total += size
        keep += 1
    if keep < 2 and len(layers) >= 2:
        raise ValueError(f"a budget of {target_params} parameters does not fit 2 layers; the model outside its "
                         f"layers has {outside} parameters")
    return keep


# Re-indexing layers


def config_updates(config, picks):
    """Return {name: new value} for the per-layer fields of config, for the new layer list.

    picks has one (source text config, old layer index) per new layer.
    config is the text config of the model that gets the layers. A
    per-layer list is any config list as long as the old layer count
    (such as layer_types). Index lists (such as mlp_only_layers) and
    Qwen2's max_window_layers are mapped too.
    """
    old_total = config.num_hidden_layers
    updates = {"num_hidden_layers": len(picks)}

    for name in INDEX_LISTS:
        value = getattr(config, name, None)
        if not isinstance(value, (list, tuple)):
            continue
        new_value = []
        for new_index, (source, old_index) in enumerate(picks):
            if old_index in getattr(source, name, []):
                new_value.append(new_index)
        updates[name] = new_value

    for name in list(vars(config)):
        if name.startswith("_") or name in INDEX_LISTS or name in NOT_PER_LAYER:
            continue
        value = getattr(config, name)
        if not isinstance(value, (list, tuple)) or len(value) != old_total:
            continue
        new_value = []
        for source, old_index in picks:
            source_value = getattr(source, name, None)
            if not isinstance(source_value, (list, tuple)) or len(source_value) != source.num_hidden_layers:
                raise ValueError(f"config field {name} is per layer in the base but not in a source")
            new_value.append(source_value[old_index])
        updates[name] = new_value

    if isinstance(getattr(config, "max_window_layers", None), int):
        # The layers before max_window_layers use full attention.
        count = 0
        for source, old_index in picks:
            if old_index < getattr(source, "max_window_layers", 0):
                count += 1
        updates["max_window_layers"] = count
    return updates


def install_layers(model, list_name, new_layers, picks):
    """Put new_layers in the model's decoder list and update layer_idx and the config."""
    config = _text_config(model.config)
    if getattr(config, "hidden_size_per_layer_input", None):
        raise ValueError(f"{type(model).__name__} has per-layer input embeddings; carving it is not supported")
    # Work out the config first, so a bad config leaves the model whole.
    updates = config_updates(config, picks)

    parent, child = parent_and_name(model, list_name)
    setattr(parent, child, nn.ModuleList(new_layers))
    for new_index, layer in enumerate(new_layers):
        # Attention modules read layer_idx to find their KV cache slot.
        for module in layer.modules():
            if isinstance(getattr(module, "layer_idx", None), int):
                module.layer_idx = new_index
    for name in updates:
        setattr(config, name, updates[name])
    # Some decoders keep a flag made from layer_types at init (Qwen2).
    if hasattr(parent, "has_sliding_layers") and isinstance(getattr(config, "layer_types", None), list):
        parent.has_sliding_layers = "sliding_attention" in config.layer_types
    return model


def check_layer_list(indices, total, what):
    if len(indices) == 0:
        raise ValueError(f"{what} is empty")
    seen = set()
    for index in indices:
        if isinstance(index, bool) or not isinstance(index, int):
            raise ValueError(f"{what} holds {index!r}, not a layer index")
        if index < 0 or index >= total:
            raise ValueError(f"{what} holds layer {index}, but the model has {total} layers")
        if index in seen:
            raise ValueError(f"{what} holds layer {index} twice")
        seen.add(index)


def carve_layers(model, keep_layers):
    """Keep only keep_layers, in that order, re-indexed, in place. Return the model.

    The result is a plain model of the same class: layer_idx (for the KV
    cache), num_hidden_layers, layer_types, and the other per-layer config
    fields follow the kept layers, so save_pretrained and from_pretrained
    round trip.
    """
    list_name, layers = find_decoder_layers(model, 0)
    keep_layers = list(keep_layers)
    check_layer_list(keep_layers, len(layers), "keep_layers")
    config = _text_config(model.config)
    new_layers = []
    picks = []
    for old_index in keep_layers:
        new_layers.append(layers[old_index])
        picks.append((config, old_index))
    return install_layers(model, list_name, new_layers, picks)


# Width


def weight_unit_scores(model):
    """Data-free unit scores in the wsp.prune format, for carve_width.

    A neuron scores |up row| * |down column| (with the gate row for a
    split MLP); a query head scores the norm of its o_proj columns.
    """
    _, layers = find_decoder_layers(model, 0)
    scores = {"max_layer": len(layers) - 1, "mlp": {}, "heads": {}}
    for index in range(len(layers)):
        layer = layers[index]
        attn, attn_kind = attention_layout(layer)
        mlp, mlp_kind = mlp_layout(layer)
        width = mlp_width(mlp, mlp_kind)
        down = mlp.down_proj.weight.detach().float().norm(dim=0)
        if mlp_kind == "split":
            up = mlp.up_proj.weight.detach().float().norm(dim=1)
        else:
            up = mlp.gate_up_proj.weight.detach().float()[width:].norm(dim=1)
        scores["mlp"][index] = (up * down).cpu()
        n_q, _, head_dim = head_counts(attn, attn_kind)
        columns = attn.o_proj.weight.detach().float().norm(dim=0)
        scores["heads"][index] = columns.reshape(n_q, head_dim).norm(dim=-1).cpu()
    return scores


def unit_scores(model, tokenizer, texts, max_tokens=512, batch_size=8):
    """Activation unit scores in the wsp.prune format: mean |activation| per neuron, mean norm per head."""
    if len(texts) == 0:
        raise ValueError("unit_scores needs texts")
    _, layers = find_decoder_layers(model, 0)
    max_layer = len(layers) - 1
    mlp_stats, head_stats, tokens = collect_unit_stats(model, tokenizer, texts, max_layer, max_tokens, batch_size)
    scores = {"max_layer": max_layer, "mlp": {}, "heads": {}, "tokens": tokens}
    for index in range(max_layer + 1):
        mean, _ = mlp_stats[index].mean_var()
        scores["mlp"][index] = mean.to(torch.float32)
        mean, _ = head_stats[index].mean_var()
        scores["heads"][index] = mean.to(torch.float32)
    return scores


def carve_width(model, keep_mlp, keep_heads, scores=None):
    """Slice every decoder layer to keep_mlp of its MLP neurons and about keep_heads of its query heads.

    Every layer keeps the same counts, so the config stays a plain config
    (one intermediate_size, one head count). Grouped-query attention stays
    consistent (see wsp.prune.head_shape). scores are unit scores in the
    wsp.prune format (unit_scores or wsp.prune.domain_unit_scores); None
    uses weight_unit_scores. The highest scores stay. In place. Returns
    (model, report), with the wsp.prune report.
    """
    _, layers = find_decoder_layers(model, 0)
    if scores is None and (keep_mlp < 1 or keep_heads < 1):
        scores = weight_unit_scores(model)
    model, report = prune_donor(model, scores, keep_mlp, keep_heads, max_layer=len(layers) - 1, keep_rule="top",
                                in_place=True)
    return model, report


# Splicing


# Config fields that a decoder layer copies into its modules when it is
# built (norm eps, activation, head layout, attention window and scale).
LAYER_FIELDS = ("num_attention_heads", "num_key_value_heads", "head_dim", "hidden_act", "hidden_activation",
                "rms_norm_eps", "layer_norm_eps", "attention_bias", "mlp_bias", "sliding_window",
                "query_pre_attn_scalar", "attn_logit_softcapping")


def layer_shapes(layer):
    shapes = {}
    for name, param in layer.named_parameters():
        shapes[name] = tuple(param.shape)
    return shapes


def check_splice(sources, plan, base):
    """Raise ValueError unless every planned layer fits the base model."""
    if base not in sources:
        raise ValueError(f"the splice base {base!r} is not a source; the sources are {', '.join(sources)}")
    if len(plan) == 0:
        raise ValueError("a splice plan needs at least one layer")
    base_model = sources[base]
    base_config = _text_config(base_model.config)
    _, base_layers = find_decoder_layers(base_model, 0)
    want = layer_shapes(base_layers[0])
    for key in sources:
        model = sources[key]
        if type(model) is not type(base_model):
            raise ValueError(f"splice source {key} is a {type(model).__name__}, the base is a "
                             f"{type(base_model).__name__}; a splice needs one architecture")
        config = _text_config(model.config)
        hidden = config.hidden_size
        if hidden != base_config.hidden_size:
            raise ValueError(f"splice source {key} has hidden size {hidden}, the base has {base_config.hidden_size}")
        # A layer keeps these values from its own config, but the saved
        # model rebuilds every layer from the base config. A difference
        # would make the saved model compute something else.
        for name in LAYER_FIELDS:
            value = getattr(config, name, None)
            base_value = getattr(base_config, name, None)
            if value != base_value:
                raise ValueError(f"splice source {key} has {name} {value!r}, the base has {base_value!r}; "
                                 "the saved model would run its layers with the base value")
    for item in plan:
        key, index = item
        if key not in sources:
            raise ValueError(f"the splice plan names {key!r}, which is not a source")
        _, layers = find_decoder_layers(sources[key], 0)
        if isinstance(index, bool) or not isinstance(index, int) or index < 0 or index >= len(layers):
            raise ValueError(f"the splice plan names layer {index!r} of {key}, which has {len(layers)} layers")
        shapes = layer_shapes(layers[index])
        if shapes != want:
            for name in want:
                if shapes.get(name) != want[name]:
                    raise ValueError(f"layer {index} of {key} does not match the base: {name} is "
                                     f"{shapes.get(name)}, the base has {want[name]}")
            raise ValueError(f"layer {index} of {key} has parameters the base layers do not")


def splice(sources, plan, base):
    """Build one model from layers of several models. Return the base model, changed in place.

    sources: {key: model}, all of one class and hidden size.
    plan: a list of (source key, layer index), in the new model's order.
    base: the key whose embeddings, final norm, head, and config the new
        model keeps.
    A layer used twice is copied. The other sources give up the planned
    layers, so drop them after the splice.
    """
    check_splice(sources, plan, base)
    base_model = sources[base]
    list_name, _ = find_decoder_layers(base_model, 0)
    embed = base_model.get_input_embeddings().weight
    # Two keys can name one model object, so a repeat is found by identity.
    used = set()
    new_layers = []
    picks = []
    for key, index in plan:
        _, layers = find_decoder_layers(sources[key], 0)
        layer = layers[index]
        if id(layer) in used:
            layer = copy.deepcopy(layer)
        used.add(id(layer))
        layer.to(device=embed.device, dtype=embed.dtype)
        new_layers.append(layer)
        picks.append((_text_config(sources[key].config), index))
    return install_layers(base_model, list_name, new_layers, picks)


# Fields that differ between models without changing what a layer computes.
SKIP_DIFFERENCES = ("_name_or_path", "architectures", "torch_dtype", "dtype", "transformers_version",
                    "num_hidden_layers", "layer_types", "max_window_layers")


def config_differences(configs, base):
    """Return {field: {key: value}} for the config fields where a source differs from the base.

    The spliced model runs every layer with the base config (for example
    its rope_theta), so these are worth a look.
    """
    base_values = _text_config(configs[base]).to_dict()
    found = {}
    for key in configs:
        if key == base:
            continue
        values = _text_config(configs[key]).to_dict()
        for name in values:
            if name in SKIP_DIFFERENCES:
                continue
            if values[name] != base_values.get(name):
                if name not in found:
                    found[name] = {base: base_values.get(name)}
                found[name][key] = values[name]
    return found
