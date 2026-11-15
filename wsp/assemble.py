"""Build grafts from catalog parts.

A catalog record of type mlp_cluster has:
  model  the donor Hugging Face id (any family)
  layer  the donor layer index
  units  a .pt file holding {"kind", "model", "layer", "indices"}

Only the tensors the part needs are fetched from the donor. W_in starts
from a ridge fit on offset-matched hidden states when alignment texts are
given and the donor is small enough to run. Otherwise it starts random and
the heal pass trains it. The same fit measures the donor's mean sink state
and its typical RMS at the input point, for the graft's sink and rms_match
options. A GraftSpec with a stitch dict then trains W_in as a stitch
(wsp.stitch) on the align texts, before heal. A part_lora_rank wraps the
part's linears in a LoRA last, so the fit and the stitch see the plain part.
"""

import os

import torch

from wsp.extract import random_neurons
from wsp.fetch import fetch_config, fetch_tensors
from wsp.graft import Graft
from wsp.parts import (
    attn_tensor_names,
    block_layer_prefixes,
    block_tensor_names,
    build_attn_heads,
    build_block_part,
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


def load_donor(donor_id, device):
    """Load the whole donor in bf16 on device, in eval mode. Return (model, tokenizer)."""
    import transformers

    try:
        donor_model = transformers.AutoModelForCausalLM.from_pretrained(donor_id, dtype=torch.bfloat16)
    except ValueError:
        # Some multimodal donors (for example mistral3) have no causal LM
        # class. The image-text class holds the same language model.
        donor_model = transformers.AutoModelForImageTextToText.from_pretrained(donor_id, dtype=torch.bfloat16)
    donor_model.to(device)
    donor_model.eval()
    donor_tokenizer = transformers.AutoTokenizer.from_pretrained(donor_id)
    return donor_model, donor_tokenizer


def fit_w_in(host_model, host_tokenizer, host_layer, donor_id, donor_layer, align_texts, device, norm_prefix=None,
             layer_input=False, stats=None):
    """Fit W_in on offset-matched hidden states. Return (w, r2, r2_dim, pairs).

    stats: None, or a dict that gets "sink" (the donor's mean state at
    position 0) and "rms" (the donor's mean row RMS), both at the input
    point. See wsp.align.collect_pairs.
    """
    from wsp.align import collect_pairs, mlp_input_capture, ridge_fit

    donor_model, donor_tokenizer = load_donor(donor_id, device)
    if norm_prefix is None:
        modules = find_layer_modules(donor_model.config, donor_layer)
        norm_prefix = modules["pre_norm_prefix"]
    norm = donor_model.get_submodule(norm_prefix)
    capture = mlp_input_capture(norm)
    if layer_input:
        from wsp.align import layer_input_capture

        modules = find_layer_modules(donor_model.config, donor_layer)
        capture = layer_input_capture(donor_model.get_submodule(modules["layer_prefix"]))
    x, y = collect_pairs(host_model, host_tokenizer, host_layer, donor_model, donor_tokenizer, capture, align_texts,
                         stats=stats)
    w, r2, r2_dim = ridge_fit(x, y)
    del donor_model
    if device == "cuda":
        torch.cuda.empty_cache()
    return w, r2, r2_dim, x.shape[0]


STITCH_DEFAULTS = {"objective": "donor_lm", "steps": 500, "lr": 1e-3, "batch_size": 8, "seq_len": 256, "seed": 0}


def stitch_settings(spec, graft, host_layer, d_donor):
    """Return the full stitch settings for a spec: its dict plus defaults.

    The settings hold what rebuild_graft needs to build the module again:
    kind, rank (lowrank), hidden (mlp), and host_layers (multi, absolute).
    """
    settings = {"kind": "linear"}
    for key in STITCH_DEFAULTS:
        settings[key] = STITCH_DEFAULTS[key]
    for key in spec.stitch:
        settings[key] = spec.stitch[key]
    kind = settings["kind"]
    if kind == "lowrank" and settings.get("rank") is None:
        settings["rank"] = graft.rank
        if settings["rank"] is None:
            raise ValueError(f"{spec.component_id}: a lowrank stitch needs a rank or a glue_rank")
    if kind == "mlp" and settings.get("hidden") is None:
        settings["hidden"] = 2 * d_donor
    if kind == "multi":
        if "host_layers" in spec.stitch:
            # host_layers would be replaced by the offsets below without a word.
            raise ValueError(f"{spec.component_id}: a multi stitch in a program takes offsets from the "
                             "graft layer (0 or less), not host_layers")
        offsets = settings.get("offsets")
        if offsets is None:
            offsets = [-2, 0]
            settings["offsets"] = offsets
        host_layers = []
        for offset in offsets:
            if offset > 0:
                raise ValueError(f"{spec.component_id}: multi stitch offset {offset} reads a host layer after "
                                 "the graft; offsets must be 0 or less")
            layer = host_layer + int(offset)
            if layer < 0:
                layer = 0
            if layer not in host_layers:
                host_layers.append(layer)
        host_layers.sort()
        settings["host_layers"] = host_layers
    return settings


def stitch_graft(spec, graft, meta, host_model, host_tokenizer, align_texts, device, ridge_w, norm_prefix=None):
    """Train a stitch for the graft and install it as the graft's w_in.

    The stitch starts from ridge_w, the graft's ridge fit (a multi stitch
    gets its own ridge fit on its concatenated host layers), and trains
    with wsp.stitch.train_stitch on the align texts. With 10 or more texts,
    the last tenth is held out for the before and after metrics.
    norm_prefix is the donor's pre-MLP norm for mlp_cluster and expert
    parts, or None when the part reads the donor layer input. When the
    graft has a sink, the stitch trains with that same sink at position 0.
    Records meta["stitch"] and updates meta["glue_params"].
    """
    from wsp.stitch import build_stitch, init_from_ridge, ridge_init, train_stitch

    if align_texts is None:
        raise ValueError(f"{spec.component_id}: a stitch trains on align_texts, and none were given")
    host_layer = meta["host_layer"]
    donor_layer = meta["donor_layer"]
    settings = stitch_settings(spec, graft, host_layer, meta["d_donor"])
    stitch = build_stitch(settings, meta["d_host"], meta["d_donor"])

    texts = list(align_texts)
    eval_texts = None
    if len(texts) >= 10:
        n_eval = len(texts) // 10
        eval_texts = texts[len(texts) - n_eval:]
        texts = texts[:len(texts) - n_eval]

    donor_model, donor_tokenizer = load_donor(meta["donor"], device)
    if settings["kind"] == "multi":
        fit = ridge_init(stitch, host_model, host_tokenizer, host_layer, donor_model, donor_tokenizer, donor_layer,
                         list(align_texts), donor_norm_prefix=norm_prefix)
        settings["fit_r2"] = fit["r2"]
        settings["fit_r2_dim"] = fit["r2_dim"]
    else:
        init_from_ridge(stitch, ridge_w)
    sink = False
    if getattr(graft, "sink", None) is not None:
        sink = graft.sink
    result = train_stitch(host_model, host_tokenizer, host_layer, donor_model, donor_tokenizer, donor_layer, stitch,
                          texts, objective=settings["objective"], steps=settings["steps"], lr=settings["lr"],
                          batch_size=settings["batch_size"], seq_len=settings["seq_len"], seed=settings["seed"],
                          end_layer=settings.get("end_layer"), eval_texts=eval_texts, donor_norm_prefix=norm_prefix,
                          sink=sink)
    del donor_model
    if device == "cuda":
        torch.cuda.empty_cache()

    graft.w_in = stitch.float().cpu()
    settings["end_layer"] = result.get("end_layer", settings.get("end_layer"))
    settings["donor_norm_prefix"] = norm_prefix
    settings["params"] = count_params(stitch)
    settings["before"] = result["before"]
    settings["after"] = result["after"]
    settings["losses"] = result["losses"]
    settings["seconds"] = result["seconds"]
    settings["train_texts"] = result["train_texts"]
    settings["eval_texts"] = result["eval_texts"]
    settings["sink"] = result["sink"]
    meta["stitch"] = settings
    meta["glue_params"] = glue_param_count(graft)


def finish_graft(spec, graft, meta, host_model, host_tokenizer, align_texts, device, donor_layer,
                 fit_norm_prefix=None, layer_input=False, stitch_norm_prefix=None):
    """Fit w_in, set the sink and RMS options, train the stitch, and add the part LoRA.

    donor_layer is the donor layer the part reads. fit_norm_prefix and
    layer_input pick the input point for fit_w_in, and stitch_norm_prefix
    for the stitch. Records every step in meta.
    """
    from wsp.lora import apply_part_lora, default_alpha

    sink_on = getattr(spec, "sink", True)
    rms_on = getattr(spec, "rms_match", False)
    lora_rank = getattr(spec, "part_lora_rank", None)
    if rms_on and align_texts is None:
        raise ValueError(f"{spec.component_id}: rms_match measures the donor RMS on align_texts, and none were given")

    w = None
    stats = {}
    if align_texts is not None:
        w, r2, r2_dim, pairs = fit_w_in(host_model, host_tokenizer, meta["host_layer"], meta["donor"], donor_layer,
                                        align_texts, device, norm_prefix=fit_norm_prefix, layer_input=layer_input,
                                        stats=stats)
        with torch.no_grad():
            graft.set_w_in(w)
        meta["fit_r2"] = r2
        meta["fit_r2_dim"] = r2_dim
        meta["fit_pairs"] = pairs
        meta["fit"] = "ridge"

    # The sink needs the fit's donor run. Without align texts it stays off.
    meta["sink"] = False
    meta["sink_norm"] = None
    if sink_on and stats.get("sink") is not None:
        graft.set_sink(stats["sink"])
        meta["sink"] = True
        meta["sink_norm"] = float(stats["sink"].norm())

    meta["rms_match"] = False
    meta["donor_rms"] = None
    if rms_on:
        if stats.get("rms") is None:
            raise ValueError(f"{spec.component_id}: rms_match needs the donor RMS, and the fit did not measure it")
        graft.set_rms_match(stats["rms"])
        meta["rms_match"] = True
        meta["donor_rms"] = stats["rms"]

    if getattr(spec, "stitch", None) is not None:
        stitch_graft(spec, graft, meta, host_model, host_tokenizer, align_texts, device, w, stitch_norm_prefix)

    meta["part_lora_rank"] = None
    meta["part_lora_alpha"] = None
    meta["part_lora_params"] = 0
    if lora_rank is not None:
        alpha = default_alpha(lora_rank)
        params = apply_part_lora(graft.part, lora_rank, alpha)
        total = 0
        for param in params:
            total += param.numel()
        meta["part_lora_rank"] = lora_rank
        meta["part_lora_alpha"] = alpha
        meta["part_lora_params"] = total
    meta["glue_params"] = glue_param_count(graft)


def glue_param_count(graft):
    """Return the count of w_in, w_out, and the RMS scale (when on) of a graft."""
    total = count_params(graft.w_in) + count_params(graft.w_out)
    if getattr(graft, "rms_scale", None) is not None:
        total += graft.rms_scale.numel()
    return total


def build_graft(spec, catalog, host_model, host_tokenizer, align_texts=None, device="cpu", cache_dir=None):
    """Return (host_layer, graft, meta) for one GraftSpec."""
    record = catalog.get(spec.component_id)
    if record["type"] == "attn_heads":
        return build_attn_graft(spec, record, catalog, host_model, host_tokenizer, align_texts, device, cache_dir)
    if record["type"] == "expert":
        return build_expert_graft(spec, record, catalog, host_model, host_tokenizer, align_texts, device, cache_dir)
    if record["type"] == "block":
        return build_block_graft(spec, record, catalog, host_model, host_tokenizer, align_texts, device, cache_dir)
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

    stitch_norm_prefix = None
    if getattr(spec, "stitch", None) is not None:
        stitch_norm_prefix = find_layer_modules(config, donor_layer)["pre_norm_prefix"]
    finish_graft(spec, graft, meta, host_model, host_tokenizer, align_texts, device, donor_layer,
                 stitch_norm_prefix=stitch_norm_prefix)
    return host_layer, graft, meta


def rebuild_graft(meta, graft_state, index):
    """Rebuild one saved graft without fetching donor weights.

    A graft with meta["stitch"] gets its stitch module back as w_in. A
    graft with meta["sink"] or meta["rms_match"] gets those options back
    from the saved state, and one with meta["part_lora_rank"] gets its part
    LoRA back. Grafts saved before these options have none of the keys, so
    they load with every option off. The caller loads the saved weights.
    """
    part_state = graft_state
    if meta.get("part_lora_rank") is not None:
        part_state = without_part_lora(graft_state, index)
    graft = rebuild_part_graft(meta, part_state, index)
    if meta.get("stitch") is not None:
        from wsp.stitch import build_stitch

        graft.w_in = build_stitch(meta["stitch"], meta["d_host"], meta["d_donor"])
    if meta.get("sink"):
        graft.set_sink(graft_state[f"{index}.sink"])
    if meta.get("rms_match"):
        graft.set_rms_match(graft_state[f"{index}.donor_rms"])
    if meta.get("part_lora_rank") is not None:
        from wsp.lora import apply_part_lora

        apply_part_lora(graft.part, meta["part_lora_rank"], meta.get("part_lora_alpha"))
    return graft


def without_part_lora(graft_state, index):
    """Return the saved state of graft index as the plain part would name it.

    A LoRALinear saves its donor linear as name.base.weight next to
    name.lora_A and name.lora_B. The part builders want name.weight, so we
    drop the LoRA factors and the ".base" step. Other grafts' keys stay as
    they are. No donor module in the supported families is named "base".
    """
    prefix = f"{index}.part."
    out = {}
    for key in graft_state:
        if not key.startswith(prefix):
            out[key] = graft_state[key]
            continue
        if key.endswith(".lora_A") or key.endswith(".lora_B"):
            continue
        out[key.replace(".base.", ".")] = graft_state[key]
    return out


def rebuild_part_graft(meta, graft_state, index):
    """Rebuild one saved graft with the default glue.

    The saved part holds the sliced tensors, so we build the part with a
    config whose intermediate size already equals the cluster size and
    indices 0..n-1, which makes slicing a no-op.
    """
    import copy

    if meta.get("kind") == "attn_heads":
        return rebuild_attn_graft(meta, graft_state, index)
    if meta.get("kind") == "expert":
        return rebuild_expert_graft(meta, graft_state, index)
    if meta.get("kind") == "block":
        return rebuild_block_graft(meta, graft_state, index)
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

    finish_graft(spec, graft, meta, host_model, host_tokenizer, align_texts, device, donor_layer,
                 fit_norm_prefix=found["pre_norm_prefix"], stitch_norm_prefix=found["pre_norm_prefix"])
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
    the donor layer. The part keeps its own KV cache, which follows the
    host cache during generation.
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
    finish_graft(spec, graft, meta, host_model, host_tokenizer, align_texts, device, donor_layer, layer_input=True)
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


def random_block_range(num_layers, first, last, seed):
    """Return a seeded random [start, end] range with the same length as [first, last].

    The range differs from [first, last] and does not overlap it when the
    donor has room for that. Raises ValueError when the block covers every
    donor layer, since no other range of that length exists.
    """
    length = last - first + 1
    apart = []
    different = []
    for start in range(num_layers - length + 1):
        end = start + length - 1
        if start == first:
            continue
        different.append(start)
        if end < first or start > last:
            apart.append(start)
    choices = apart
    if len(choices) == 0:
        choices = different
    if len(choices) == 0:
        raise ValueError(f"block [{first}, {last}] covers all {num_layers} donor layers; "
                         "no other range of that length exists for the random control")
    generator = torch.Generator()
    generator.manual_seed(seed)
    pick = int(torch.randint(len(choices), (1,), generator=generator))
    start = choices[pick]
    return start, start + length - 1


def build_block_graft(spec, record, catalog, host_model, host_tokenizer, align_texts, device, cache_dir):
    """Return (host_layer, graft, meta) for a block record.

    The record holds "layers": [i, j], the first and last donor layer, both
    included. No units file is needed. The random control runs another range
    of the same length from the same donor, at the same host layer. W_in is
    fit against the input of the first donor layer that runs. The part
    keeps its own KV cache, which follows the host cache during generation.
    """
    donor_id = record["model"]
    first = int(record["layers"][0])
    last = int(record["layers"][1])
    config = fetch_config(donor_id)
    text_config = _text_config(config)
    donor_layers = text_config.num_hidden_layers
    if first < 0 or last < first or last >= donor_layers:
        raise ValueError(f"{spec.component_id}: layers {record['layers']} are not a range in "
                         f"{donor_layers} donor layers")

    host_layers = host_layer_count(host_model)
    host_layer = resolve_host_layer(spec, first, donor_layers, host_layers)

    if spec.control == "random":
        first, last = random_block_range(donor_layers, first, last, spec.control_seed)

    tensors = fetch_tensors(donor_id, block_tensor_names(config, first, last), cache_dir=cache_dir)
    part = build_block_part(config, first, last, tensors)

    d_host = host_hidden_size(host_model)
    d_donor = text_config.hidden_size
    graft = Graft(d_host, d_donor, part, rank=getattr(spec, "glue_rank", None))

    layer_ids = []
    for index in range(first, last + 1):
        layer_ids.append(index)
    meta = {
        "id": spec.component_id,
        "kind": "block",
        "control": spec.control,
        "donor": donor_id,
        "donor_layer": first,
        "layers": [first, last],
        "indices": layer_ids,
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
    finish_graft(spec, graft, meta, host_model, host_tokenizer, align_texts, device, first, layer_input=True)
    return host_layer, graft, meta


def rebuild_block_graft(meta, graft_state, index):
    """Rebuild a saved block graft without fetching donor weights.

    Only the config is fetched. The saved part keys its layers as
    layers.0, layers.1, and so on, so we map each one back to the donor
    module name of layer first + k.
    """
    config = fetch_config(meta["donor"])
    first = int(meta["layers"][0])
    last = int(meta["layers"][1])
    prefixes = block_layer_prefixes(config, first, last)
    tensors = {}
    prefix = f"{index}.part.layers."
    for key in graft_state:
        if not key.startswith(prefix):
            continue
        rest = key[len(prefix):]
        offset, name = rest.split(".", 1)
        tensors[prefixes[int(offset)] + "." + name] = graft_state[key]
    part = build_block_part(config, first, last, tensors)
    return Graft(meta["d_host"], meta["d_donor"], part, rank=meta.get("glue_rank"))
