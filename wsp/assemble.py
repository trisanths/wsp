"""Build grafts from catalog parts.

A catalog record of type mlp_cluster has:
  model  the donor Hugging Face id (any family)
  layer  the donor layer index
  units  a .pt file holding {"kind", "model", "layer", "indices"}

Only the tensors the part needs are fetched from the donor. W_in starts
from a ridge fit on offset-matched hidden states when alignment texts are
given and the donor is small enough to run. Otherwise it starts random and
the heal pass trains it.
"""

import os

import torch

from wsp.extract import random_neurons
from wsp.fetch import fetch_config, fetch_tensors
from wsp.graft import Graft
from wsp.parts import (
    attn_tensor_names,
    build_attn_heads,
    build_expert_part,
    build_mlp_cluster,
    expert_tensor_names,
    find_layer_modules,
    find_moe_block,
    mlp_tensor_names,
    _text_config,
)


def read_units(catalog, component_id):
    record = catalog.get(component_id)
    path = record["units"]
    if not os.path.isabs(path):
        path = os.path.join(catalog.root, *path.split("/"))
    return torch.load(path)


def host_layer_count(host_model):
    return _text_config(host_model.config).num_hidden_layers


def host_hidden_size(host_model):
    return _text_config(host_model.config).hidden_size


def default_host_layer(donor_layer, donor_layers, host_layers):
    k = round(donor_layer / donor_layers * host_layers)
    if k < 0:
        k = 0
    if k > host_layers - 1:
        k = host_layers - 1
    return k


def resolve_host_layer(spec, donor_layer, donor_layers, host_layers):
    """Return the host layer index for a graft spec.

    spec.at None picks a layer at the same relative depth as the donor
    layer. A negative spec.at counts from the last layer, as in Python. An
    index outside the host raises ValueError.
    """
    if spec.at is None:
        return default_host_layer(donor_layer, donor_layers, host_layers)
    host_layer = spec.at
    if host_layer < 0:
        host_layer += host_layers
    if host_layer < 0 or host_layer >= host_layers:
        raise ValueError(f"{spec.component_id}: at={spec.at} is out of range for {host_layers} host layers")
    return host_layer


def fit_w_in(host_model, host_tokenizer, host_layer, donor_id, donor_layer, align_texts, device, norm_prefix=None,
             layer_input=False):
    """Fit W_in on offset-matched hidden states. Return (w, r2, r2_dim, pairs)."""
    import transformers

    from wsp.align import collect_pairs, mlp_input_capture, ridge_fit

    try:
        donor_model = transformers.AutoModelForCausalLM.from_pretrained(donor_id, dtype=torch.bfloat16)
    except ValueError:
        # Some multimodal donors (for example mistral3) have no causal LM
        # class. The image-text class holds the same language model.
        donor_model = transformers.AutoModelForImageTextToText.from_pretrained(donor_id, dtype=torch.bfloat16)
    donor_model.to(device)
    donor_model.eval()
    donor_tokenizer = transformers.AutoTokenizer.from_pretrained(donor_id)
    if norm_prefix is None:
        modules = find_layer_modules(donor_model.config, donor_layer)
        norm_prefix = modules["pre_norm_prefix"]
    norm = donor_model.get_submodule(norm_prefix)
    capture = mlp_input_capture(norm)
    if layer_input:
        from wsp.align import layer_input_capture

        modules = find_layer_modules(donor_model.config, donor_layer)
        capture = layer_input_capture(donor_model.get_submodule(modules["layer_prefix"]))
    x, y = collect_pairs(host_model, host_tokenizer, host_layer, donor_model, donor_tokenizer, capture, align_texts)
    w, r2, r2_dim = ridge_fit(x, y)
    del donor_model
    if device == "cuda":
        torch.cuda.empty_cache()
    return w, r2, r2_dim, x.shape[0]


def build_graft(spec, catalog, host_model, host_tokenizer, align_texts=None, device="cpu", cache_dir=None):
    """Return (host_layer, graft, meta) for one GraftSpec."""
    record = catalog.get(spec.component_id)
    if record["type"] == "attn_heads":
        return build_attn_graft(spec, record, catalog, host_model, host_tokenizer, align_texts, device, cache_dir)
    if record["type"] == "expert":
        return build_expert_graft(spec, record, catalog, host_model, host_tokenizer, align_texts, device, cache_dir)
    if record["type"] != "mlp_cluster":
        raise ValueError(f"{spec.component_id}: graft type {record['type']} is not built yet")

    donor_id = record["model"]
    donor_layer = record["layer"]
    config = fetch_config(donor_id)
    text_config = _text_config(config)
    units = read_units(catalog, spec.component_id)
    indices = units["indices"]
    if spec.control == "random":
        indices = random_neurons(text_config.intermediate_size, len(indices), spec.control_seed)

    tensors = fetch_tensors(donor_id, mlp_tensor_names(config, donor_layer), cache_dir=cache_dir)
    part = build_mlp_cluster(config, donor_layer, indices, tensors)

    host_layers = host_layer_count(host_model)
    host_layer = resolve_host_layer(spec, donor_layer, text_config.num_hidden_layers, host_layers)

    d_host = host_hidden_size(host_model)
    d_donor = text_config.hidden_size
    graft = Graft(d_host, d_donor, part, rank=getattr(spec, "glue_rank", None))

    meta = {
        "id": spec.component_id,
        "kind": "mlp_cluster",
        "control": spec.control,
        "donor": donor_id,
        "donor_layer": donor_layer,
        "indices": indices.tolist(),
        "host_layer": host_layer,
        "d_host": d_host,
        "d_donor": d_donor,
        "part_params": count_params(part),
        "glue_params": count_params(graft.w_in) + count_params(graft.w_out),
        "glue_rank": graft.rank,
        "fit_r2": None,
        "fit_r2_dim": None,
        "fit_pairs": 0,
        "fit": "skipped",
    }

    if align_texts is not None:
        w, r2, r2_dim, pairs = fit_w_in(host_model, host_tokenizer, host_layer, donor_id, donor_layer,
                                        align_texts, device)
        with torch.no_grad():
            graft.set_w_in(w)
        meta["fit_r2"] = r2
        meta["fit_r2_dim"] = r2_dim
        meta["fit_pairs"] = pairs
        meta["fit"] = "ridge"
    return host_layer, graft, meta


def rebuild_graft(meta, graft_state, index):
    """Rebuild one saved graft without fetching donor weights.

    The saved part holds the sliced tensors, so we build the part with a
    config whose intermediate size already equals the cluster size and
    indices 0..n-1, which makes slicing a no-op.
    """
    import copy

    if meta.get("kind") == "attn_heads":
        return rebuild_attn_graft(meta, graft_state, index)
    if meta.get("kind") == "expert":
        return rebuild_expert_graft(meta, graft_state, index)
    config = copy.deepcopy(fetch_config(meta["donor"]))
    n = len(meta["indices"])
    text_config = _text_config(config)
    text_config.intermediate_size = n
    modules = find_layer_modules(config, meta["donor_layer"])
    tensors = {}
    prefix = f"{index}.part."
    for key in graft_state:
        if not key.startswith(prefix):
            continue
        rest = key[len(prefix):]
        if rest.startswith("mlp."):
            tensors[modules["mlp_prefix"] + rest[len("mlp"):]] = graft_state[key]
        elif rest.startswith("pre_norm."):
            tensors[modules["pre_norm_prefix"] + rest[len("pre_norm"):]] = graft_state[key]
        elif rest.startswith("post_norm."):
            tensors[modules["post_norm_prefix"] + rest[len("post_norm"):]] = graft_state[key]
    part = build_mlp_cluster(config, meta["donor_layer"], torch.arange(n), tensors)
    return Graft(meta["d_host"], meta["d_donor"], part, rank=meta.get("glue_rank"))


def count_params(module):
    total = 0
    for param in module.parameters():
        total += param.numel()
    return total


def build_expert_graft(spec, record, catalog, host_model, host_tokenizer, align_texts, device, cache_dir):
    """Build a graft for an expert record. Returns (host_layer, graft, meta).

    Units "indices" are expert ids in the donor layer. Only those experts and
    the layer norm are fetched. The random control picks the same number of
    random expert ids.
    """
    donor_id = record["model"]
    donor_layer = record["layer"]
    config = fetch_config(donor_id)
    units = read_units(catalog, spec.component_id)
    experts = torch.as_tensor(units["indices"], dtype=torch.long)
    found = find_moe_block(config, donor_layer)
    if spec.control == "random":
        experts = random_neurons(found["num_experts"], len(experts), spec.control_seed)

    names = expert_tensor_names(config, donor_layer, experts.tolist())
    tensors = fetch_tensors(donor_id, names, cache_dir=cache_dir)
    part = build_expert_part(config, donor_layer, experts, tensors)

    text_config = _text_config(config)
    host_layers = host_layer_count(host_model)
    host_layer = resolve_host_layer(spec, donor_layer, text_config.num_hidden_layers, host_layers)

    d_host = host_hidden_size(host_model)
    d_donor = text_config.hidden_size
    graft = Graft(d_host, d_donor, part, rank=getattr(spec, "glue_rank", None))

    meta = {
        "id": spec.component_id,
        "kind": "expert",
        "control": spec.control,
        "donor": donor_id,
        "donor_layer": donor_layer,
        "indices": experts.tolist(),
        "host_layer": host_layer,
        "d_host": d_host,
        "d_donor": d_donor,
        "part_params": count_params(part),
        "glue_params": count_params(graft.w_in) + count_params(graft.w_out),
        "glue_rank": graft.rank,
        "fit_r2": None,
        "fit_r2_dim": None,
        "fit_pairs": 0,
        "fit": "skipped",
    }

    if align_texts is not None:
        w, r2, r2_dim, pairs = fit_w_in(host_model, host_tokenizer, host_layer, donor_id, donor_layer,
                                        align_texts, device, norm_prefix=found["pre_norm_prefix"])
        with torch.no_grad():
            graft.set_w_in(w)
        meta["fit_r2"] = r2
        meta["fit_r2_dim"] = r2_dim
        meta["fit_pairs"] = pairs
        meta["fit"] = "ridge"
    return host_layer, graft, meta


def rebuild_expert_graft(meta, graft_state, index):
    """Rebuild one saved expert graft without fetching donor weights.

    Only the config is fetched. The saved part keys experts as 0..n-1, so we
    rebuild with expert ids 0..n-1 and map each saved key to the donor name.
    """
    config = fetch_config(meta["donor"])
    n = len(meta["indices"])
    found = find_moe_block(config, meta["donor_layer"])
    tensors = {}
    prefix = f"{index}.part."
    for key in graft_state:
        if not key.startswith(prefix):
            continue
        rest = key[len(prefix):]
        if rest.startswith("experts."):
            tensors[found["moe_prefix"] + "." + rest] = graft_state[key]
        elif rest.startswith("pre_norm."):
            tensors[found["pre_norm_prefix"] + rest[len("pre_norm"):]] = graft_state[key]
        elif rest.startswith("post_norm."):
            tensors[found["post_norm_prefix"] + rest[len("post_norm"):]] = graft_state[key]
    part = build_expert_part(config, meta["donor_layer"], torch.arange(n), tensors)
    return Graft(meta["d_host"], meta["d_donor"], part, rank=meta.get("glue_rank"))


def random_heads(num_heads, n, seed):
    """Return n distinct random query head ids, sorted, from a seeded generator."""
    generator = torch.Generator()
    generator.manual_seed(seed)
    picks = torch.randperm(num_heads, generator=generator)[:n]
    return torch.sort(picks).values


def build_attn_graft(spec, record, catalog, host_model, host_tokenizer, align_texts, device, cache_dir):
    """Return (host_layer, graft, meta) for an attn_heads record.

    The units file holds query head ids. W_in is fit against the input of
    the donor layer. Grafts of this kind need use_cache=False when the host
    generates, because the part recomputes attention over the full sequence.
    """
    donor_id = record["model"]
    donor_layer = record["layer"]
    config = fetch_config(donor_id)
    text_config = _text_config(config)
    heads = torch.as_tensor(read_units(catalog, spec.component_id)["indices"], dtype=torch.long)
    if spec.control == "random":
        heads = random_heads(text_config.num_attention_heads, len(heads), spec.control_seed)

    tensors = fetch_tensors(donor_id, attn_tensor_names(config, donor_layer), cache_dir=cache_dir)
    part = build_attn_heads(config, donor_layer, heads, tensors)

    host_layers = host_layer_count(host_model)
    host_layer = resolve_host_layer(spec, donor_layer, text_config.num_hidden_layers, host_layers)

    d_host = host_hidden_size(host_model)
    d_donor = text_config.hidden_size
    graft = Graft(d_host, d_donor, part, rank=getattr(spec, "glue_rank", None))

    meta = {
        "id": spec.component_id,
        "kind": "attn_heads",
        "control": spec.control,
        "donor": donor_id,
        "donor_layer": donor_layer,
        "indices": heads.tolist(),
        "host_layer": host_layer,
        "d_host": d_host,
        "d_donor": d_donor,
        "part_params": count_params(part),
        "glue_params": count_params(graft.w_in) + count_params(graft.w_out),
        "glue_rank": graft.rank,
        "fit_r2": None,
        "fit_r2_dim": None,
        "fit_pairs": 0,
        "fit": "skipped",
    }
    if align_texts is not None:
        w, r2, r2_dim, pairs = fit_w_in(host_model, host_tokenizer, host_layer, donor_id, donor_layer,
                                        align_texts, device, layer_input=True)
        with torch.no_grad():
            graft.set_w_in(w)
        meta["fit_r2"] = r2
        meta["fit_r2_dim"] = r2_dim
        meta["fit_pairs"] = pairs
        meta["fit"] = "ridge"
    return host_layer, graft, meta


def rebuild_attn_graft(meta, graft_state, index):
    """Rebuild a saved attn_heads graft without fetching donor weights.

    The saved part holds one KV head per query head, so we build with both
    head counts equal to n, head_dim fixed to the donor's, and heads 0..n-1.
    That makes the slicing a no-op.
    """
    import copy

    from wsp.parts import _head_dim

    config = copy.deepcopy(fetch_config(meta["donor"]))
    n = len(meta["indices"])
    text_config = _text_config(config)
    text_config.head_dim = _head_dim(text_config)
    text_config.num_attention_heads = n
    text_config.num_key_value_heads = n
    modules = find_layer_modules(config, meta["donor_layer"])
    names = {
        "attn": modules["attn_prefix"],
        "input_norm": modules["input_norm_prefix"],
        "post_norm": modules["post_attn_norm_prefix"],
    }
    tensors = {}
    prefix = f"{index}.part."
    for key in graft_state:
        if not key.startswith(prefix):
            continue
        rest = key[len(prefix):]
        owner = rest.split(".")[0]
        if owner in names:
            tensors[names[owner] + rest[len(owner):]] = graft_state[key]
    part = build_attn_heads(config, meta["donor_layer"], torch.arange(n), tensors)
    return Graft(meta["d_host"], meta["d_donor"], part, rank=meta.get("glue_rank"))
