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
from wsp.parts import build_mlp_cluster, find_layer_modules, mlp_tensor_names


def read_units(catalog, component_id):
    record = catalog.get(component_id)
    path = record["units"]
    if not os.path.isabs(path):
        path = os.path.join(catalog.root, *path.split("/"))
    return torch.load(path)


def host_layer_count(host_model):
    return host_model.config.num_hidden_layers


def default_host_layer(donor_layer, donor_layers, host_layers):
    k = round(donor_layer / donor_layers * host_layers)
    if k < 0:
        k = 0
    if k > host_layers - 1:
        k = host_layers - 1
    return k


def fit_w_in(host_model, host_tokenizer, host_layer, donor_id, donor_layer, align_texts, device):
    """Fit W_in on offset-matched hidden states. Return (w, r2, pairs)."""
    import transformers

    from wsp.align import collect_pairs, mlp_input_capture, ridge_fit

    donor_model = transformers.AutoModelForCausalLM.from_pretrained(donor_id, dtype=torch.bfloat16)
    donor_model.to(device)
    donor_model.eval()
    donor_tokenizer = transformers.AutoTokenizer.from_pretrained(donor_id)
    modules = find_layer_modules(donor_model.config, donor_layer)
    norm = donor_model.get_submodule(modules["pre_norm_prefix"])
    capture = mlp_input_capture(norm)
    x, y = collect_pairs(host_model, host_tokenizer, host_layer, donor_model, donor_tokenizer, capture, align_texts)
    w, r2 = ridge_fit(x, y)
    del donor_model
    if device == "cuda":
        torch.cuda.empty_cache()
    return w, r2, x.shape[0]


def build_graft(spec, catalog, host_model, host_tokenizer, align_texts=None, device="cpu", cache_dir=None):
    """Return (host_layer, graft, meta) for one GraftSpec."""
    record = catalog.get(spec.component_id)
    if record["type"] != "mlp_cluster":
        raise ValueError(f"{spec.component_id}: graft type {record['type']} is not built yet (only mlp_cluster)")

    donor_id = record["model"]
    donor_layer = record["layer"]
    config = fetch_config(donor_id)
    units = read_units(catalog, spec.component_id)
    indices = units["indices"]
    if spec.control == "random":
        indices = random_neurons(config.intermediate_size, len(indices), spec.control_seed)

    tensors = fetch_tensors(donor_id, mlp_tensor_names(config, donor_layer), cache_dir=cache_dir)
    part = build_mlp_cluster(config, donor_layer, indices, tensors)

    host_layers = host_layer_count(host_model)
    host_layer = spec.at
    if host_layer is None:
        host_layer = default_host_layer(donor_layer, config.num_hidden_layers, host_layers)

    d_host = host_model.config.hidden_size
    d_donor = config.hidden_size
    graft = Graft(d_host, d_donor, part)

    meta = {
        "id": spec.component_id,
        "control": spec.control,
        "donor": donor_id,
        "donor_layer": donor_layer,
        "indices": indices.tolist(),
        "host_layer": host_layer,
        "d_host": d_host,
        "d_donor": d_donor,
        "part_params": count_params(part),
        "glue_params": count_params(graft.w_in) + count_params(graft.w_out),
        "fit_r2": None,
        "fit_pairs": 0,
        "fit": "skipped",
    }

    if align_texts is not None:
        w, r2, pairs = fit_w_in(host_model, host_tokenizer, host_layer, donor_id, donor_layer, align_texts, device)
        with torch.no_grad():
            graft.w_in.weight.copy_(w.to(graft.w_in.weight.dtype))
        meta["fit_r2"] = r2
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

    config = copy.deepcopy(fetch_config(meta["donor"]))
    n = len(meta["indices"])
    text_config = config
    if hasattr(config, "get_text_config"):
        text_config = config.get_text_config()
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
    return Graft(meta["d_host"], meta["d_donor"], part)


def count_params(module):
    total = 0
    for param in module.parameters():
        total += param.numel()
    return total
