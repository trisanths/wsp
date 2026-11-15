# Architecture adapter: finds, for any decoder-only (or vision-language) transformers model,
#   - the list of decoder layers
#   - per layer: the attention output projection (+ heads, head_dim), the dense MLP down
#     projection (+ width), and the MoE experts module (+ number of experts) if present
# and installs zero-ablation hooks for parts. Runs inside the Modal engine.
# full_module_parts() names the parts that switch off a whole attention or feed-forward module.
#
# Part formats (plain dicts):
#   {"kind": "mlp_cluster", "layer": L, "indices": [...]}   columns of the MLP down-projection input
#   {"kind": "attn_heads",  "layer": L, "indices": [...]}   heads = slices of the attention out-proj input
#   {"kind": "expert",      "layer": L, "indices": [...]}   routed experts: their routing weight is set to 0
#   {"kind": "block",       "layers": [L, L+1, ...]}        whole layers skipped (output = input)
import torch

ATTN_NAMES = ["self_attn", "attention", "attn", "linear_attn", "self_attention", "mixer"]
ATTN_OUT_NAMES = ["o_proj", "out_proj", "dense", "c_proj", "wo", "o"]
MLP_NAMES = ["mlp", "feed_forward", "ffn", "block_sparse_moe", "moe"]
DOWN_NAMES = ["down_proj", "w2", "fc2", "c_proj", "wo", "dense_4h_to_h"]
HEAD_NORM_NAMES = ["q_norm", "k_norm", "q_layernorm", "k_layernorm", "norm"]


class AdapterError(Exception):
    pass


def find_layers(decoder):
    best = None
    for name, mod in decoder.named_children():
        if isinstance(mod, torch.nn.ModuleList) and len(mod) > 0:
            if best is None or len(mod) > len(best[1]):
                best = (name, mod)
    if best is None:
        raise AdapterError("no ModuleList of decoder layers found in " + type(decoder).__name__)
    return best[1]


def child(module, names):
    for n in names:
        sub = getattr(module, n, None)
        if isinstance(sub, torch.nn.Module):
            return n, sub
    return None, None


def linear_out(module, names, d_model):
    for n in names:
        sub = getattr(module, n, None)
        if isinstance(sub, torch.nn.Linear) and sub.out_features == d_model:
            return n, sub
    return None, None


def head_dim_for(attn, config, out_in):
    text = config.get_text_config() if hasattr(config, "get_text_config") else config
    for n in HEAD_NORM_NAMES:
        sub = getattr(attn, n, None)
        w = getattr(sub, "weight", None) if sub is not None else None
        if isinstance(w, torch.Tensor) and w.dim() == 1 and 1 < w.numel() < out_in and out_in % w.numel() == 0:
            return int(w.numel())
    hd = getattr(text, "head_dim", None)
    if hd and out_in % hd == 0:
        return int(hd)
    heads = getattr(text, "num_attention_heads", None)
    if heads and out_in % heads == 0:
        return out_in // heads
    return None


def is_experts_module(mod):
    # transformers-5 style fused experts: 3D gate_up_proj / down_proj parameters, called as
    # experts(hidden_states, top_k_index, top_k_weights)
    for pname, p in mod.named_parameters(recurse=False):
        if p.dim() == 3 and ("down" in pname or "gate_up" in pname or "w2" in pname):
            return True
    return False


def describe(model, config):
    decoder = model.get_decoder()
    layers = find_layers(decoder)
    text = config.get_text_config() if hasattr(config, "get_text_config") else config
    d_model = int(text.hidden_size)
    info = {"n_layers": len(layers), "d_model": d_model, "layers": []}
    for i, layer in enumerate(layers):
        entry = {"layer": i}
        an, attn = child(layer, ATTN_NAMES)
        if attn is not None:
            on, out = linear_out(attn, ATTN_OUT_NAMES, d_model)
            if out is not None:
                hd = head_dim_for(attn, config, out.in_features)
                if hd is not None:
                    entry["attn"] = {"path": an + "." + on, "n_heads": out.in_features // hd, "head_dim": hd,
                                     "type": type(attn).__name__}
        mn, mlp = child(layer, MLP_NAMES)
        if mlp is not None:
            dn, down = linear_out(mlp, DOWN_NAMES, d_model)
            if down is not None:
                entry["mlp"] = {"path": mn + "." + dn, "width": down.in_features}
            for sub_name, sub in mlp.named_children():
                if is_experts_module(sub):
                    n_exp = None
                    for pname, p in sub.named_parameters(recurse=False):
                        if p.dim() == 3:
                            n_exp = int(p.shape[0])
                            break
                    entry["experts"] = {"path": mn + "." + sub_name, "n_experts": n_exp}
                if sub_name in ["shared_expert", "shared_experts", "shared_mlp"] and "mlp" not in entry:
                    dn2, down2 = linear_out(sub, DOWN_NAMES, d_model)
                    if down2 is not None:
                        entry["mlp"] = {"path": mn + "." + sub_name + "." + dn2, "width": down2.in_features,
                                        "note": "shared expert"}
        if "experts" not in entry:
            # Some models keep the experts outside the MLP module (Gemma 4: layer.experts next to
            # layer.mlp) or under another block name. Search the whole layer for a fused experts module.
            for path, sub in layer.named_modules():
                if path and is_experts_module(sub):
                    n_exp = None
                    for pname, p in sub.named_parameters(recurse=False):
                        if p.dim() == 3:
                            n_exp = int(p.shape[0])
                            break
                    entry["experts"] = {"path": path, "n_experts": n_exp}
                    break
        if "experts" not in entry:
            # Older MoE code (DeepSeek-V3 style, e.g. G9v3): a ModuleList of per-expert MLPs named
            # "experts"; the block multiplies each expert output by its routing weight.
            for path, sub in layer.named_modules():
                if path.split(".")[-1] == "experts" and isinstance(sub, torch.nn.ModuleList) and len(sub) > 1:
                    entry["experts"] = {"path": path, "n_experts": len(sub), "style": "module_list"}
                    break
        info["layers"].append(entry)
    return info, layers


def get_module(layer, path):
    mod = layer
    for piece in path.split("."):
        mod = getattr(mod, piece)
    return mod


def summarize(info):
    counts = {"attn": 0, "mlp": 0, "experts": 0}
    for e in info["layers"]:
        for k in counts:
            if k in e:
                counts[k] = counts[k] + 1
    first = {}
    for e in info["layers"]:
        for k in ["attn", "mlp", "experts"]:
            if k in e and k not in first:
                first[k] = e[k]
    return {"n_layers": info["n_layers"], "d_model": info["d_model"], "layers_with": counts, "example": first}


def check_part(info, part):
    kind = part.get("kind")
    n = info["n_layers"]
    if kind == "block":
        layers = part.get("layers") or []
        if not layers:
            return "block needs layers"
        for j in range(len(layers)):
            if layers[j] < 0 or layers[j] >= n:
                return "layer out of range"
            if j > 0 and layers[j] != layers[j - 1] + 1:
                return "block layers must be contiguous"
        return None
    if kind not in ["mlp_cluster", "attn_heads", "expert"]:
        return "kind must be mlp_cluster, attn_heads, expert or block"
    layer = part.get("layer")
    if layer is None or layer < 0 or layer >= n:
        return "layer missing or out of range"
    entry = info["layers"][layer]
    key = {"mlp_cluster": "mlp", "attn_heads": "attn", "expert": "experts"}[kind]
    if key not in entry:
        return "layer " + str(layer) + " has no " + key + " module this adapter supports"
    limit = {"mlp": "width", "attn": "n_heads", "experts": "n_experts"}[key]
    size = entry[key][limit]
    idx = part.get("indices") or []
    if not idx:
        return "indices must be non-empty"
    for i in idx:
        if i < 0 or i >= size:
            return "index " + str(i) + " out of range 0.." + str(size - 1)
    if len(set(idx)) != len(idx):
        return "duplicate indices"
    return None


def full_module_parts(info, layer, module):
    # The zero-ablation parts that together remove one whole module of one layer:
    #   "attn": every head of the attention out-projection
    #   "mlp":  the feed-forward path, which is every column of the dense MLP (or shared expert) and
    #           every routed expert, whichever of the two the layer has
    # Returns a list, because a layer with a shared expert next to routed experts needs two parts.
    # The list is empty when the layer has no such module that this adapter supports.
    # The parts zero the INPUT of the out / down projection, so a bias on that projection stays:
    # it is a constant that does not depend on the text.
    if module not in ["attn", "mlp"]:
        raise AdapterError("module must be attn or mlp, not " + str(module))
    entry = info["layers"][layer]
    parts = []
    if module == "attn":
        if "attn" in entry:
            parts.append({"kind": "attn_heads", "layer": layer, "indices": list(range(entry["attn"]["n_heads"]))})
        return parts
    if "mlp" in entry:
        parts.append({"kind": "mlp_cluster", "layer": layer, "indices": list(range(entry["mlp"]["width"]))})
    if "experts" in entry and entry["experts"].get("n_experts"):
        parts.append({"kind": "expert", "layer": layer, "indices": list(range(entry["experts"]["n_experts"]))})
    return parts


def install_all(info, layers, parts, handles):
    # Checks and installs several parts at once (same handle rule as install: the caller removes
    # them in `finally`). Raises AdapterError on a part that check_part refuses.
    for part in parts:
        err = check_part(info, part)
        if err:
            raise AdapterError("bad part: " + err)
        install(info, layers, part, handles)


def module_ablation_maps(info, layers, names, score_all):
    # For every layer: switch off its whole attention module, call score_all(), switch it on again,
    # then do the same for its feed-forward module. score_all() returns {name: value} for the model
    # as it is hooked at that moment. Returns
    #   {"attn": {name: [value per layer]}, "mlp": {name: [value per layer]},
    #    "parts": [{"attn": [part kinds], "mlp": [part kinds]} per layer]}
    # with None for a layer that lacks the module (score_all is not called for it).
    out = {"attn": {}, "mlp": {}, "parts": []}
    for module in ["attn", "mlp"]:
        for name in names:
            out[module][name] = []
    for l in range(info["n_layers"]):
        kinds = {}
        for module in ["attn", "mlp"]:
            parts = full_module_parts(info, l, module)
            kinds[module] = [p["kind"] for p in parts]
            scores = None
            if parts:
                handles = []
                try:
                    install_all(info, layers, parts, handles)
                    scores = score_all()
                finally:
                    remove_all(handles)
            for name in names:
                out[module][name].append(None if scores is None else scores[name])
        out["parts"].append(kinds)
    return out


def install(info, layers, part, handles):
    # Appends every hook handle to the caller-held `handles` list as soon as it is registered,
    # so a failure half-way never leaves hooks behind (the caller removes them in `finally`).
    kind = part["kind"]
    if kind == "block":
        for l in part["layers"]:
            def hook(module, args, kwargs, output):
                hs_in = args[0] if len(args) > 0 else kwargs["hidden_states"]
                if isinstance(output, tuple):
                    return (hs_in,) + tuple(output[1:])
                return hs_in
            handles.append(layers[int(l)].register_forward_hook(hook, with_kwargs=True))
        return
    layer = int(part["layer"])
    entry = info["layers"][layer]
    if kind == "expert" and entry["experts"].get("style") == "module_list":
        experts = get_module(layers[layer], entry["experts"]["path"])

        def zero_out(module, args, output):
            # the block scales this output by the routing weight, so zero output = expert removed
            return torch.zeros_like(output)
        for i in part["indices"]:
            handles.append(experts[int(i)].register_forward_hook(zero_out))
        return
    if kind == "expert":
        mod = get_module(layers[layer], entry["experts"]["path"])
        banned = torch.tensor([int(i) for i in part["indices"]], dtype=torch.long)

        def pre(module, args, kwargs):
            # zero the routing weight of the banned experts: their output is removed, the other
            # selected experts keep their weights (no renormalisation)
            args = list(args)
            if len(args) >= 3:
                idx, w = args[1], args[2]
            else:
                idx = kwargs["top_k_index"]
                w = kwargs["top_k_weights"]
            keep = ~torch.isin(idx, banned.to(idx.device))
            w = w * keep.to(w.dtype)
            if len(args) >= 3:
                args[2] = w
            else:
                kwargs["top_k_weights"] = w
            return tuple(args), kwargs
        handles.append(mod.register_forward_pre_hook(pre, with_kwargs=True))
        return
    if kind == "attn_heads":
        mod = get_module(layers[layer], entry["attn"]["path"])
        hd = entry["attn"]["head_dim"]
        cols = []
        for h in part["indices"]:
            for c in range(int(h) * hd, (int(h) + 1) * hd):
                cols.append(c)
    else:
        mod = get_module(layers[layer], entry["mlp"]["path"])
        cols = [int(i) for i in part["indices"]]
    idx_t = torch.tensor(cols, dtype=torch.long)

    def pre_cols(module, args):
        x = args[0].clone()
        x[..., idx_t.to(x.device)] = 0
        return (x,) + tuple(args[1:])
    handles.append(mod.register_forward_pre_hook(pre_cols))


def remove_all(handles):
    while handles:
        h = handles.pop()
        h.remove()
