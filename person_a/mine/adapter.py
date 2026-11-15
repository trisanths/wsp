# Architecture adapter: finds, for any decoder-only (or vision-language) transformers model,
#   - the list of decoder layers
#   - per layer: the attention output projection (+ heads, head_dim), the dense MLP down
#     projection (+ width), and the MoE experts module (+ number of experts) if present
# and installs zero-ablation hooks for parts. Runs inside the Modal engine.
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


def expert_count_layers(info, layers=None):
    # Splits the wanted layers (all layers when `layers` is empty) into those whose router picks a
    # pre-hook can count, and those it cannot, with the reason. Fused experts get the router's top-k
    # indices as a call argument. The "module_list" style gives them to no module this adapter knows,
    # so those layers are reported and never counted.
    counted = []
    unsupported = {}
    wanted = [int(l) for l in (layers or [])]
    for l in wanted:
        if l < 0 or l >= info["n_layers"]:
            unsupported[str(l)] = "layer out of range"
    for e in info["layers"]:
        if wanted and e["layer"] not in wanted:
            continue
        if "experts" not in e:
            if wanted:
                unsupported[str(e["layer"])] = "layer has no experts module this adapter supports"
            continue
        if e["experts"].get("style") == "module_list":
            unsupported[str(e["layer"])] = "module_list experts: the router's picks are not visible to a hook"
            continue
        counted.append(e["layer"])
    return counted, unsupported


def router_picks(idx, mask, n_exp):
    # idx: the router's top-k expert indices as the experts module gets them for one batch: one row
    # per token, (tokens, k), or unflattened, (batch, seq, k); a top-1 router may give (tokens,).
    # mask: 1 for each token to count, in batch order. Returns (picks per expert, tokens counted, k),
    # or None when idx does not hold one row of indices for each token of the mask.
    # The mask is built on the input device. With the layers spread over several GPUs the indices
    # of a layer are on that layer's GPU, so the mask is moved there before it is used.
    flat = mask.reshape(-1).bool().to(idx.device)
    tokens = int(flat.numel())
    if tokens == 0:
        return None
    if idx.dim() >= 2 and idx.numel() == tokens * int(idx.shape[-1]):
        k = int(idx.shape[-1])
    elif idx.dim() == 1 and idx.numel() == tokens:
        k = 1
    else:
        return None
    sel = idx.reshape(tokens, k)[flat]
    counts = torch.bincount(sel.reshape(-1), minlength=n_exp)[:n_exp].float()
    return counts, float(sel.shape[0]), k


def install_pick_counter(info, layers, layer, sums, state, bad, handles):
    # Counts the router's picks of one fused-experts layer. sums[layer] gets {"sum", "n", "top_k"};
    # state["mask"] is the mask of the batch that runs. A layer whose indices cannot be counted gets
    # its reason in bad[layer] and no counts.
    entry = info["layers"][layer]
    mod = get_module(layers[layer], entry["experts"]["path"])
    n_exp = entry["experts"]["n_experts"]

    def pre(module, args, kwargs):
        idx = args[1] if len(args) >= 3 else kwargs.get("top_k_index")
        if not isinstance(idx, torch.Tensor):
            bad[layer] = "the experts module was called without top-k indices"
            return
        got = router_picks(idx, state["mask"], n_exp)
        if got is None:
            bad[layer] = ("router indices of shape " + str(list(idx.shape)) + " do not fit the "
                          + str(int(state["mask"].numel())) + " tokens of the batch")
            return
        counts, n, k = got
        if layer not in sums:
            sums[layer] = {"sum": torch.zeros(n_exp, device=counts.device), "n": 0.0, "top_k": k}
        sums[layer]["sum"] += counts
        sums[layer]["n"] += n
    handles.append(mod.register_forward_pre_hook(pre, with_kwargs=True))


def pick_rates(sums):
    # sums: {layer: {"sum": picks per expert (tensor), "n": tokens counted, "top_k": k or absent}}, as
    # the routing hook fills it. Returns per layer the picks per counted token for each expert, so the
    # rates of one layer add up to the router's top-k. A layer with no counted token gives None.
    out = {}
    for l in sums:
        n = float(sums[l]["n"])
        if n <= 0:
            out[str(l)] = None
            continue
        freq = (sums[l]["sum"].float() / n).cpu().tolist()
        out[str(l)] = {"rates": [round(float(x), 6) for x in freq], "tokens": int(n)}
        if "top_k" in sums[l]:
            out[str(l)]["top_k"] = int(sums[l]["top_k"])
    return out


def drop_uncounted(rates, counted, unsupported, bad):
    # After the passes: a layer whose indices did not fit on one batch or more (bad) leaves every
    # rate table and moves from the counted list to "unsupported" with its reason.
    for key in rates:
        for l in bad:
            rates[key]["layers"].pop(str(l), None)
    for l in bad:
        unsupported[str(l)] = bad[l]
    return [l for l in counted if l not in bad]


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
