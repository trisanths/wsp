"""Domain pruning for tap donors.

A tap reads the donor's hidden state at one layer, max_layer, so only donor
layers 0..max_layer ever run. This module makes those layers cheaper for
one domain:

- domain_unit_scores runs the donor on domain text and on generic text and
  scores each MLP neuron and each attention query head in layers
  0..max_layer. The score is Cohen's d, domain vs generic, the method in
  wsp_eval/README.md: |activation| at the down_proj input for neurons, and
  the head's output norm (its slice of the o_proj input) for heads.
- prune_donor keeps a fraction of the neurons and query heads in each of
  those layers, slices the weights for real, and drops the layers after
  max_layer. The result is an ordinary Transformers model of the donor's
  own class, so wsp.tap.run_donor works on it unchanged.

Every pruned layer keeps the same counts, so the model config stays true:
num_attention_heads, num_key_value_heads, head_dim, intermediate_size, and
num_hidden_layers are updated, and save_pretrained / from_pretrained round
trip.

Grouped-query attention: query head i reads KV head i // groups. To keep
that rule, every kept KV head serves the same number of kept query heads.
For a keep fraction we pick (kept KV heads, query heads per KV head) whose
product is closest to the target count. On a tie we keep more KV heads.
The "top" rule then keeps the KV groups whose best query heads score
highest, and the best query heads inside each kept group. The "random"
rule keeps the same counts with random groups and heads (seeded), as the
control.

Supported layouts: separate q_proj, k_proj, v_proj, o_proj or a fused
qkv_proj (phi3), and separate gate_proj, up_proj, down_proj or a fused
gate_up_proj (phi3). This covers llama, qwen2, gemma2, mistral, and phi3.
Other layouts (MoE MLPs, MLA attention, and so on) raise ValueError.

Usage:

    scores = domain_unit_scores(donor, tok, domain_texts, generic_texts, max_layer=20)
    pruned, report = prune_donor(donor, scores, keep_mlp=0.5, keep_heads=0.5, max_layer=20)
    memory, ends, valid = run_donor(pruned, tok, prompts, 20)
"""

import copy
import math
import numbers

import torch
from torch import nn

from wsp.parts import find_decoder_layers


class _StopDonor(Exception):
    """Raised by a hook after max_layer so later layers never run."""


class RunningStats:
    """Per-unit count, sum, and sum of squares, in float64 on the CPU."""

    def __init__(self):
        self.n = 0
        self.sum = None
        self.sumsq = None

    def add(self, values):
        # values: (tokens, units)
        values = values.detach().to(torch.float64).cpu()
        total = values.sum(dim=0)
        total_sq = (values * values).sum(dim=0)
        if self.sum is None:
            self.sum = torch.zeros_like(total)
            self.sumsq = torch.zeros_like(total_sq)
        self.sum += total
        self.sumsq += total_sq
        self.n += values.shape[0]

    def mean_var(self):
        count = max(self.n, 1)
        mean = self.sum / count
        var = self.sumsq / count - mean * mean
        return mean, var.clamp(min=0)


def cohens_d(domain_stat, generic_stat):
    """Effect size per unit, positive when the unit is larger on domain text."""
    domain_mean, domain_var = domain_stat.mean_var()
    generic_mean, generic_var = generic_stat.mean_var()
    pooled = torch.sqrt((domain_var + generic_var) / 2 + 1e-8)
    return ((domain_mean - generic_mean) / pooled).to(torch.float32)


def is_linear(module):
    """True for a plain nn.Linear. A quantized linear (packed or int weights) raises ValueError."""
    if not isinstance(module, nn.Linear):
        return False
    weight = module.weight
    if not weight.is_floating_point() or tuple(weight.shape) != (module.out_features, module.in_features):
        raise ValueError(f"{type(module).__name__} has a quantized or packed weight "
                         f"({weight.dtype}, shape {tuple(weight.shape)}); quantized layers are not supported, "
                         "so prune a full-precision donor")
    return True


def check_max_layer(model, max_layer):
    """Raise ValueError unless max_layer is a decoder layer index of the model."""
    if isinstance(max_layer, bool) or not isinstance(max_layer, numbers.Integral) or max_layer < 0:
        raise ValueError(f"max_layer must be an int >= 0, got {max_layer!r}")
    find_decoder_layers(model, max_layer)


def attention_layout(layer):
    """Return (attention module, "split" or "fused"), or raise ValueError."""
    attn = getattr(layer, "self_attn", None)
    if attn is None:
        raise ValueError(f"{type(layer).__name__} has no self_attn; this layout is not supported")
    if not hasattr(attn, "head_dim") or not hasattr(attn, "num_key_value_groups"):
        raise ValueError(f"{type(attn).__name__} has no head_dim or num_key_value_groups; "
                         "this attention layout is not supported")
    if not is_linear(getattr(attn, "o_proj", None)):
        raise ValueError(f"{type(attn).__name__} has no o_proj linear; this attention layout is not supported")
    split = True
    for name in ["q_proj", "k_proj", "v_proj"]:
        if not is_linear(getattr(attn, name, None)):
            split = False
    if split:
        return attn, "split"
    if is_linear(getattr(attn, "qkv_proj", None)):
        return attn, "fused"
    raise ValueError(f"{type(attn).__name__} has neither q/k/v_proj nor qkv_proj linears; "
                     "this attention layout is not supported")


def mlp_layout(layer):
    """Return (MLP module, "split" or "fused"), or raise ValueError."""
    mlp = getattr(layer, "mlp", None)
    if mlp is None:
        raise ValueError(f"{type(layer).__name__} has no mlp; this layout is not supported")
    if not is_linear(getattr(mlp, "down_proj", None)):
        raise ValueError(f"{type(mlp).__name__} has no down_proj linear (a MoE MLP?); "
                         "this MLP layout is not supported")
    if is_linear(getattr(mlp, "gate_proj", None)) and is_linear(getattr(mlp, "up_proj", None)):
        return mlp, "split"
    if is_linear(getattr(mlp, "gate_up_proj", None)):
        return mlp, "fused"
    raise ValueError(f"{type(mlp).__name__} has neither gate_proj/up_proj nor gate_up_proj linears; "
                     "this MLP layout is not supported")


def head_counts(attn, layout):
    """Return (query heads, KV heads, head_dim) read from the weight shapes."""
    head_dim = attn.head_dim
    n_q = attn.o_proj.in_features // head_dim
    if layout == "split":
        n_kv = attn.k_proj.out_features // head_dim
    else:
        n_kv = (attn.qkv_proj.out_features - n_q * head_dim) // (2 * head_dim)
    if n_kv < 1 or n_q % n_kv != 0:
        raise ValueError(f"{type(attn).__name__}: {n_q} query heads do not split over {n_kv} KV heads")
    if attn.num_key_value_groups != n_q // n_kv:
        raise ValueError(f"{type(attn).__name__}: num_key_value_groups is {attn.num_key_value_groups}, "
                         f"but the weights give {n_q // n_kv}")
    return n_q, n_kv, head_dim


def mlp_width(mlp, layout):
    return mlp.down_proj.in_features


def model_device(model):
    return next(model.parameters()).device


def collect_unit_stats(model, tokenizer, texts, max_layer, max_tokens=512, batch_size=8):
    """Run texts through layers 0..max_layer and gather |activation| and head norm stats.

    Returns (mlp, heads, tokens): dicts of layer -> RunningStats, and the
    number of real tokens seen. Pads do not count.
    """
    _, layers = find_decoder_layers(model, max_layer)
    mlp_stats = {}
    head_stats = {}
    current = {"mask": None}
    handles = []

    def rows_of(tensor):
        # (batch, seq, units) -> (real tokens, units)
        return tensor[current["mask"]]

    for index in range(max_layer + 1):
        layer = layers[index]
        mlp, _ = mlp_layout(layer)
        attn, layout = attention_layout(layer)
        n_q, _, head_dim = head_counts(attn, layout)
        mlp_stats[index] = RunningStats()
        head_stats[index] = RunningStats()

        def neuron_hook(module, args, index=index):
            mlp_stats[index].add(rows_of(args[0]).abs())

        def head_hook(module, args, index=index, n_q=n_q, head_dim=head_dim):
            rows = rows_of(args[0]).to(torch.float32)
            heads = rows.reshape(rows.shape[0], n_q, head_dim)
            head_stats[index].add(heads.norm(dim=-1))

        handles.append(mlp.down_proj.register_forward_pre_hook(neuron_hook))
        handles.append(attn.o_proj.register_forward_pre_hook(head_hook))

    def stop_hook(module, inputs, output):
        raise _StopDonor()

    handles.append(layers[max_layer].register_forward_hook(stop_hook))

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    old_side = tokenizer.padding_side
    was_training = model.training
    model.eval()
    device = model_device(model)
    tokens = 0
    try:
        tokenizer.padding_side = "right"
        for start in range(0, len(texts), batch_size):
            batch = list(texts[start:start + batch_size])
            encoded = tokenizer(batch, return_tensors="pt", padding=True, truncation=True,
                                max_length=max_tokens)
            mask = encoded["attention_mask"].bool().to(device)
            current["mask"] = mask
            tokens += int(mask.sum())
            try:
                with torch.no_grad():
                    model(input_ids=encoded["input_ids"].to(device), attention_mask=mask.long(),
                          use_cache=False)
            except _StopDonor:
                pass
    finally:
        tokenizer.padding_side = old_side
        for handle in handles:
            handle.remove()
        if was_training:
            model.train()
    return mlp_stats, head_stats, tokens


def domain_unit_scores(donor_model, tokenizer, domain_texts, generic_texts, max_layer,
                       max_tokens=512, batch_size=8):
    """Score every MLP neuron and query head in donor layers 0..max_layer.

    Each score is Cohen's d of the unit's statistic on domain text against
    generic text, so a high score means the domain uses the unit more:
    - mlp: |activation| at the down_proj input, one score per neuron.
    - heads: the norm of the head's slice of the o_proj input, one score
      per query head.
    Layers after max_layer never run.

    Returns a dict {"max_layer", "mlp": {layer: tensor}, "heads": {layer:
    tensor}, "domain_tokens", "generic_tokens"}. torch.save stores it.
    """
    if len(domain_texts) == 0 or len(generic_texts) == 0:
        raise ValueError("domain_unit_scores needs domain texts and generic texts")
    check_max_layer(donor_model, max_layer)
    domain_mlp, domain_heads, domain_tokens = collect_unit_stats(
        donor_model, tokenizer, domain_texts, max_layer, max_tokens, batch_size)
    generic_mlp, generic_heads, generic_tokens = collect_unit_stats(
        donor_model, tokenizer, generic_texts, max_layer, max_tokens, batch_size)
    scores = {"max_layer": max_layer, "mlp": {}, "heads": {},
              "domain_tokens": domain_tokens, "generic_tokens": generic_tokens}
    for index in range(max_layer + 1):
        scores["mlp"][index] = cohens_d(domain_mlp[index], generic_mlp[index])
        scores["heads"][index] = cohens_d(domain_heads[index], generic_heads[index])
    return scores


def keep_count(fraction, total):
    """Round fraction * total half up, and keep at least one unit."""
    count = int(math.floor(fraction * total + 0.5))
    if count < 1:
        count = 1
    if count > total:
        count = total
    return count


def head_shape(n_q, n_kv, fraction):
    """Return (kept KV heads, query heads per kept KV head) for a keep fraction.

    The product is the closest to fraction * n_q. On a tie we keep more KV
    heads, then more query heads.
    """
    group = n_q // n_kv
    target = keep_count(fraction, n_q)
    best = None
    best_gap = None
    for kv_keep in range(n_kv, 0, -1):
        for per_group in range(group, 0, -1):
            gap = abs(kv_keep * per_group - target)
            if best is None or gap < best_gap:
                best = (kv_keep, per_group)
                best_gap = gap
    return best


def sorted_list(indices):
    values = []
    for value in indices:
        values.append(int(value))
    values.sort()
    return values


def layer_scores(scores, kind, layer, size):
    """Return the score tensor for one kind and layer, checked against the unit count."""
    if scores is None or kind not in scores or layer not in scores[kind]:
        raise ValueError(f"keep_rule='top' needs {kind} scores for layer {layer}; "
                         "run domain_unit_scores with max_layer at least this deep")
    values = scores[kind][layer].detach().to(torch.float32).cpu()
    if values.numel() != size:
        raise ValueError(f"{kind} scores for layer {layer} have {values.numel()} entries, the layer has {size}")
    return values


def pick_neurons(width, count, rule, values, generator):
    if count == width:
        return list(range(width))
    if rule == "top":
        return sorted_list(torch.topk(values, count).indices)
    return sorted_list(torch.randperm(width, generator=generator)[:count])


def pick_heads(n_q, n_kv, kv_keep, per_group, rule, values, generator):
    """Return (kept query heads, kept KV heads), both sorted.

    Kept query heads come grouped by their KV head, per_group per group, so
    the new query head j reads the new KV head j // per_group.
    """
    group = n_q // n_kv
    if kv_keep == n_kv and per_group == group:
        return list(range(n_q)), list(range(n_kv))

    if rule == "top":
        grouped = values.reshape(n_kv, group)
        best_in_group = torch.topk(grouped, per_group, dim=1)
        group_totals = best_in_group.values.sum(dim=1)
        kv_heads = sorted_list(torch.topk(group_totals, kv_keep).indices)
        q_heads = []
        for kv in kv_heads:
            for local in sorted_list(best_in_group.indices[kv]):
                q_heads.append(kv * group + local)
        return q_heads, kv_heads

    kv_heads = sorted_list(torch.randperm(n_kv, generator=generator)[:kv_keep])
    q_heads = []
    for kv in kv_heads:
        for local in sorted_list(torch.randperm(group, generator=generator)[:per_group]):
            q_heads.append(kv * group + local)
    return q_heads, kv_heads


def head_rows(heads, head_dim, offset=0):
    """Weight rows (or columns) for a list of heads."""
    rows = []
    for head in heads:
        start = offset + head * head_dim
        for row in range(start, start + head_dim):
            rows.append(row)
    return rows


def slice_linear(linear, rows=None, cols=None):
    """Return a new nn.Linear with only the given output rows and input columns."""
    weight = linear.weight.detach()
    if rows is not None:
        weight = weight.index_select(0, torch.tensor(rows, dtype=torch.long, device=weight.device))
    if cols is not None:
        weight = weight.index_select(1, torch.tensor(cols, dtype=torch.long, device=weight.device))
    has_bias = linear.bias is not None
    new = nn.Linear(weight.shape[1], weight.shape[0], bias=has_bias, device=weight.device, dtype=weight.dtype)
    with torch.no_grad():
        new.weight.copy_(weight)
        if has_bias:
            bias = linear.bias.detach()
            if rows is not None:
                bias = bias.index_select(0, torch.tensor(rows, dtype=torch.long, device=bias.device))
            new.bias.copy_(bias)
    new.weight.requires_grad_(linear.weight.requires_grad)
    if has_bias:
        new.bias.requires_grad_(linear.bias.requires_grad)
    return new


def prune_mlp(mlp, layout, neurons):
    width = mlp.down_proj.in_features
    if layout == "split":
        mlp.gate_proj = slice_linear(mlp.gate_proj, rows=neurons)
        mlp.up_proj = slice_linear(mlp.up_proj, rows=neurons)
    else:
        # gate_up_proj rows are [gate (width), up (width)]; forward splits them with chunk(2)
        rows = list(neurons)
        for neuron in neurons:
            rows.append(width + neuron)
        mlp.gate_up_proj = slice_linear(mlp.gate_up_proj, rows=rows)
    mlp.down_proj = slice_linear(mlp.down_proj, cols=neurons)
    if hasattr(mlp, "intermediate_size"):
        mlp.intermediate_size = len(neurons)


def prune_attention(attn, layout, n_q, n_kv, head_dim, q_heads, kv_heads):
    q_rows = head_rows(q_heads, head_dim)
    kv_rows = head_rows(kv_heads, head_dim)
    if layout == "split":
        attn.q_proj = slice_linear(attn.q_proj, rows=q_rows)
        attn.k_proj = slice_linear(attn.k_proj, rows=kv_rows)
        attn.v_proj = slice_linear(attn.v_proj, rows=kv_rows)
    else:
        # qkv_proj rows are [q (n_q heads), k (n_kv heads), v (n_kv heads)]
        rows = list(q_rows)
        rows.extend(head_rows(kv_heads, head_dim, offset=n_q * head_dim))
        rows.extend(head_rows(kv_heads, head_dim, offset=(n_q + n_kv) * head_dim))
        attn.qkv_proj = slice_linear(attn.qkv_proj, rows=rows)
    attn.o_proj = slice_linear(attn.o_proj, cols=q_rows)
    attn.num_key_value_groups = len(q_heads) // len(kv_heads)
    if hasattr(attn, "num_heads"):
        attn.num_heads = len(q_heads)
    if hasattr(attn, "num_attention_heads"):
        attn.num_attention_heads = len(q_heads)
    if hasattr(attn, "num_key_value_heads"):
        attn.num_key_value_heads = len(kv_heads)


def module_params(module):
    total = 0
    for param in module.parameters():
        total += param.numel()
    return total


def model_params(model):
    total = 0
    for param in model.parameters():
        total += param.numel()
    return total


def parent_and_name(model, dotted):
    if "." not in dotted:
        return model, dotted
    parent_name, child = dotted.rsplit(".", 1)
    return model.get_submodule(parent_name), child


def decoder_configs(model, layers):
    """Return the config objects that describe the decoder layers.

    For a multimodal model (for example mistral3) that is the text config,
    not the top-level one. An attention module that holds its own config
    object (phi3 reads num_attention_heads from it in forward) is included.
    """
    config = model.config
    if hasattr(config, "get_text_config"):
        config = config.get_text_config()
    configs = [config]
    for layer in layers:
        own = getattr(layer.self_attn, "config", None)
        if own is None:
            continue
        seen = False
        for known in configs:
            if own is known:
                seen = True
        if not seen:
            configs.append(own)
    return configs


def update_config(config, n_q, n_kv, head_dim, width, layer_count):
    config.num_attention_heads = n_q
    config.num_key_value_heads = n_kv
    config.head_dim = head_dim
    config.intermediate_size = width
    config.num_hidden_layers = layer_count
    layer_types = getattr(config, "layer_types", None)
    if isinstance(layer_types, list):
        config.layer_types = layer_types[:layer_count]


def prune_donor(donor_model, scores, keep_mlp, keep_heads, max_layer, keep_rule="top", seed=0,
                in_place=False):
    """Prune donor layers 0..max_layer to a domain and drop the layers after max_layer.

    keep_mlp and keep_heads are fractions in (0, 1], applied to every layer:
    keep_mlp of the MLP neurons and about keep_heads of the query heads (see
    head_shape for grouped-query attention). keep_rule "top" keeps the units
    with the highest domain scores; "random" keeps the same counts at random
    (seeded), the control. scores comes from domain_unit_scores; with
    keep_rule "random", or with both fractions at 1.0, it may be None.

    The donor is copied first unless in_place is True. Returns (model,
    report). The report gives parameters before and after, FLOPs per token
    before and after for the layers a tap at max_layer runs (2 times their
    parameters; attention over the sequence and the embedding lookup are
    left out, as in wsp.tap.donor_flops_note), the full model's FLOPs per
    token for comparison, and per-layer counts and kept indices.
    """
    if keep_rule not in ("top", "random"):
        raise ValueError(f"keep_rule must be 'top' or 'random', got {keep_rule!r}")
    for name, fraction in [("keep_mlp", keep_mlp), ("keep_heads", keep_heads)]:
        if not (0 < fraction <= 1):
            raise ValueError(f"{name} must be in (0, 1], got {fraction}")
    check_max_layer(donor_model, max_layer)

    list_name, layers = find_decoder_layers(donor_model, max_layer)
    layer_total = len(layers)

    # Check every layout before touching anything, so an error leaves the donor whole.
    shapes = []
    for index in range(max_layer + 1):
        attn, attn_kind = attention_layout(layers[index])
        mlp, mlp_kind = mlp_layout(layers[index])
        n_q, n_kv, head_dim = head_counts(attn, attn_kind)
        width = mlp_width(mlp, mlp_kind)
        shapes.append((n_q, n_kv, head_dim, width))
    for shape in shapes:
        if shape != shapes[0]:
            raise ValueError(f"donor layers differ in shape ({shapes[0]} vs {shape}); "
                             "prune_donor needs uniform layers")
    n_q, n_kv, head_dim, width = shapes[0]

    params_before = model_params(donor_model)
    run_params_before = 0
    for index in range(max_layer + 1):
        run_params_before += module_params(layers[index])
    full_params = 0
    for layer in layers:
        full_params += module_params(layer)
    head = donor_model.get_output_embeddings()
    if head is not None:
        full_params += module_params(head)

    if in_place:
        model = donor_model
    else:
        model = copy.deepcopy(donor_model)
    _, layers = find_decoder_layers(model, max_layer)

    neuron_count = keep_count(keep_mlp, width)
    kv_keep, per_group = head_shape(n_q, n_kv, keep_heads)
    generator = torch.Generator().manual_seed(seed)
    rows = []
    for index in range(max_layer + 1):
        layer = layers[index]
        attn, attn_kind = attention_layout(layer)
        mlp, mlp_kind = mlp_layout(layer)
        before = module_params(layer)

        neuron_scores = None
        head_scores = None
        if keep_rule == "top":
            if neuron_count < width:
                neuron_scores = layer_scores(scores, "mlp", index, width)
            if kv_keep * per_group < n_q:
                head_scores = layer_scores(scores, "heads", index, n_q)
        neurons = pick_neurons(width, neuron_count, keep_rule, neuron_scores, generator)
        q_heads, kv_heads = pick_heads(n_q, n_kv, kv_keep, per_group, keep_rule, head_scores, generator)

        prune_mlp(mlp, mlp_kind, neurons)
        prune_attention(attn, attn_kind, n_q, n_kv, head_dim, q_heads, kv_heads)
        rows.append({"layer": index, "mlp_before": width, "mlp_after": len(neurons),
                     "q_heads_before": n_q, "q_heads_after": len(q_heads),
                     "kv_heads_before": n_kv, "kv_heads_after": len(kv_heads),
                     "params_before": before, "params_after": module_params(layer),
                     "neurons": neurons, "q_heads": q_heads, "kv_heads": kv_heads})

    # The tap never reads past max_layer, so those layers go.
    parent, child = parent_and_name(model, list_name)
    kept_layers = []
    for index in range(max_layer + 1):
        kept_layers.append(layers[index])
    setattr(parent, child, nn.ModuleList(kept_layers))
    for config in decoder_configs(model, kept_layers):
        update_config(config, kv_keep * per_group, kv_keep, head_dim, neuron_count, max_layer + 1)

    run_params_after = 0
    for row in rows:
        run_params_after += row["params_after"]
    report = {"keep_rule": keep_rule, "keep_mlp": keep_mlp, "keep_heads": keep_heads, "seed": seed,
              "max_layer": max_layer, "layers_before": layer_total, "layers_after": max_layer + 1,
              "params_before": params_before, "params_after": model_params(model),
              "run_params_before": run_params_before, "run_params_after": run_params_after,
              "flops_per_token_before": 2 * run_params_before,
              "flops_per_token_after": 2 * run_params_after,
              "flops_per_token_full_model": 2 * full_params,
              "flops_ratio": run_params_after / run_params_before,
              "layers": rows}
    return model, report
