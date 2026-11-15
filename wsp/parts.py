"""Parts: pieces of a donor model that can be grafted into a host.

A part runs the donor's own Transformers code (its MLP class and its norm
classes), so any family that Transformers supports works. We find the
classes by building the donor on the meta device from its config alone,
which allocates no memory and downloads no weights.
"""

import copy
import re

import torch
from torch import nn


def _text_config(config):
    """Return the text part of a config (the config itself for plain LMs)."""
    if hasattr(config, "get_text_config"):
        return config.get_text_config()
    return config


def find_decoder_layers(model, layer):
    """Return (name, ModuleList) of the decoder layers that contain `layer`.

    We look for a ModuleList whose name ends with "layers". If a model has
    more than one (for example a vision tower and a language model), the
    language model wins.
    """
    best_name = None
    best_list = None
    best_score = -1
    for name, module in model.named_modules():
        if not isinstance(module, nn.ModuleList):
            continue
        if not name.endswith("layers"):
            continue
        if len(module) <= layer:
            continue
        score = 1
        if "language_model" in name or "text_model" in name:
            score = 3
        elif "vision" in name or "visual" in name or "encoder" in name:
            score = 0
        if score > best_score:
            best_name = name
            best_list = module
            best_score = score
    if best_list is None:
        raise ValueError(f"no decoder layer list with layer {layer} found")
    return best_name, best_list


def _build_meta_model(config):
    """Build the donor on the meta device from its config alone."""
    import transformers

    with torch.device("meta"):
        try:
            return transformers.AutoModelForCausalLM.from_config(config)
        except ValueError:
            # Some multimodal configs (for example mistral3) have no causal
            # LM class. The image-text class holds the same language model.
            return transformers.AutoModelForImageTextToText.from_config(config)


def _name_rules(model):
    """Return the rename rules between checkpoint names and module names.

    Multimodal models such as Gemma 3 and Mistral 3 keep old names in their
    checkpoints (language_model.model.layers.N) while their modules are
    named model.language_model.layers.N. Transformers renames the keys with
    _checkpoint_conversion_mapping when it loads and saves. We do the same.
    Returns (to_module, to_checkpoint), two lists of (pattern, replacement).
    Plain LMs have no mapping, so both lists are empty.
    """
    mapping = getattr(model, "_checkpoint_conversion_mapping", None)
    to_module = []
    to_checkpoint = []
    if not mapping:
        return to_module, to_checkpoint
    for pattern, replacement in mapping.items():
        to_module.append((pattern, replacement))
        old_name = pattern.lstrip("^")
        old_name = re.sub(r"\(.*\)", "", old_name)
        to_checkpoint.append(("^" + re.escape(replacement), old_name))
    return to_module, to_checkpoint


def _rename(name, rules):
    """Apply the first rule that matches name, like Transformers does."""
    for pattern, replacement in rules:
        new_name, count = re.subn(pattern, replacement, name)
        if count > 0:
            return new_name
    return name


def _checkpoint_names(found, names):
    """Turn module names into the tensor names the checkpoint uses."""
    out = []
    for name in names:
        out.append(_rename(name, found["to_checkpoint"]))
    return out


def _module_named(tensors, found):
    """Return tensors keyed by module names.

    Fetched tensors use checkpoint names. Tensors rebuilt from a saved graft
    already use module names, and the rules leave those names unchanged.
    """
    if not found["to_module"]:
        return tensors
    out = {}
    for name, tensor in tensors.items():
        out[_rename(name, found["to_module"])] = tensor
    return out


def find_layer_modules(config, layer):
    """Find the MLP, the attention, and their norms for one donor layer.

    Builds the donor on the meta device. Returns a dict with mlp, mlp_prefix,
    pre_norm, pre_norm_prefix, post_norm, post_norm_prefix and the attention
    keys. The prefixes are module names in the donor model, so get_submodule
    works with them. Use _checkpoint_names to get the checkpoint tensor
    names. mlp is None when the layer has no plain mlp attribute (for example
    Mixtral, whose layers hold block_sparse_moe). post_norm is None when the
    layer has no post-feedforward norm.
    """
    model = _build_meta_model(config)
    to_module, to_checkpoint = _name_rules(model)
    list_name, layers = find_decoder_layers(model, layer)
    layer_module = layers[layer]
    layer_prefix = f"{list_name}.{layer}"

    if hasattr(layer_module, "pre_feedforward_layernorm"):
        pre_name = "pre_feedforward_layernorm"
    else:
        pre_name = "post_attention_layernorm"

    result = {}
    result["to_module"] = to_module
    result["to_checkpoint"] = to_checkpoint
    result["mlp"] = getattr(layer_module, "mlp", None)
    result["mlp_prefix"] = f"{layer_prefix}.mlp"
    result["pre_norm"] = getattr(layer_module, pre_name)
    result["pre_norm_prefix"] = f"{layer_prefix}.{pre_name}"
    if hasattr(layer_module, "post_feedforward_layernorm"):
        result["post_norm"] = layer_module.post_feedforward_layernorm
        result["post_norm_prefix"] = f"{layer_prefix}.post_feedforward_layernorm"
    else:
        result["post_norm"] = None
        result["post_norm_prefix"] = None
    _add_attention_modules(result, model, layer_module, layer_prefix)
    return result


def _add_attention_modules(result, model, layer_module, layer_prefix):
    """Add the attention keys to a find_layer_modules result."""
    result["attn"] = getattr(layer_module, "self_attn", None)
    result["attn_prefix"] = f"{layer_prefix}.self_attn"
    result["input_norm"] = getattr(layer_module, "input_layernorm", None)
    result["input_norm_prefix"] = f"{layer_prefix}.input_layernorm"
    result["layer_prefix"] = layer_prefix
    result["post_attn_norm"] = None
    result["post_attn_norm_prefix"] = None
    if hasattr(layer_module, "pre_feedforward_layernorm"):
        result["post_attn_norm"] = layer_module.post_attention_layernorm
        result["post_attn_norm_prefix"] = f"{layer_prefix}.post_attention_layernorm"
    result["rotary"] = None
    best_score = -1
    for name, module in model.named_modules():
        if not name.endswith("rotary_emb"):
            continue
        score = 1
        if "language_model" in name or "text_model" in name:
            score = 3
        elif "vision" in name or "visual" in name:
            score = 0
        if score > best_score:
            result["rotary"] = module
            best_score = score


def _prefixed_keys(module, prefix):
    """Return the state_dict keys of a module with a prefix added."""
    names = []
    for key in module.state_dict().keys():
        names.append(f"{prefix}.{key}")
    return names


def _require_dense_mlp(found, layer):
    """Raise ValueError when the layer has no dense MLP to take neurons from."""
    mlp = found["mlp"]
    if mlp is None:
        raise ValueError(
            f"layer {layer} has no mlp module (it may be a sparse MoE layer); "
            "use kind 'expert' instead of 'mlp_cluster'"
        )
    if hasattr(mlp, "experts"):
        raise ValueError(
            f"layer {layer} mlp is a sparse MoE block ({type(mlp).__name__}); "
            "use kind 'expert' instead of 'mlp_cluster'"
        )


def mlp_tensor_names(config, layer):
    """List the checkpoint tensor names a cluster needs for one layer."""
    found = find_layer_modules(config, layer)
    _require_dense_mlp(found, layer)
    names = _prefixed_keys(found["mlp"], found["mlp_prefix"])
    names += _prefixed_keys(found["pre_norm"], found["pre_norm_prefix"])
    if found["post_norm"] is not None:
        names += _prefixed_keys(found["post_norm"], found["post_norm_prefix"])
    return _checkpoint_names(found, names)


class MLPCluster(nn.Module):
    """post_norm(mlp(pre_norm(x))), with the post norm optional."""

    def __init__(self, pre_norm, mlp, post_norm=None):
        super().__init__()
        self.pre_norm = pre_norm
        self.mlp = mlp
        self.post_norm = post_norm

    def forward(self, x):
        out = self.mlp(self.pre_norm(x))
        if self.post_norm is not None:
            out = self.post_norm(out)
        return out


def _make_mlp(mlp_cls, config, n):
    """Build the donor MLP class with a smaller intermediate size."""
    small = copy.deepcopy(config)
    small.intermediate_size = n
    try:
        return mlp_cls(small)
    except TypeError:
        return mlp_cls(small, intermediate_size=n)


def _load_norm(meta_norm, prefix, tensors, dtype):
    """Copy a meta norm, give it real memory, and load its weights."""
    norm = copy.deepcopy(meta_norm)
    norm = norm.to_empty(device="cpu")
    norm = norm.to(dtype)
    state = {}
    for key in norm.state_dict().keys():
        state[key] = tensors[f"{prefix}.{key}"].to(dtype)
    norm.load_state_dict(state, strict=True)
    return norm


def build_mlp_cluster(config, layer, indices, tensors):
    """Build a cluster of donor MLP neurons.

    indices: LongTensor of neuron ids into the donor's intermediate size.
    tensors: dict of full checkpoint name -> tensor.
    The result is in eval mode with all parameters frozen.
    """
    found = find_layer_modules(config, layer)
    _require_dense_mlp(found, layer)
    tensors = _module_named(tensors, found)
    text_config = _text_config(config)
    full_size = text_config.intermediate_size
    indices = torch.as_tensor(indices, dtype=torch.long)
    n = len(indices)

    mlp_prefix = found["mlp_prefix"]
    first_name = _prefixed_keys(found["mlp"], mlp_prefix)[0]
    dtype = tensors[first_name].dtype

    mlp = _make_mlp(type(found["mlp"]), text_config, n)
    mlp = mlp.to(dtype)
    state = {}
    for key in found["mlp"].state_dict().keys():
        tensor = tensors[f"{mlp_prefix}.{key}"]
        idx = indices.to(tensor.device)
        if "gate_up" in key:
            # Phi-3 and Phi-4 fuse gate and up into one tensor of 2 * I rows.
            # The gate half comes first, so take each neuron from both halves.
            tensor = tensor.index_select(0, torch.cat([idx, idx + full_size]))
        else:
            for dim in range(tensor.dim()):
                if tensor.shape[dim] == full_size:
                    tensor = tensor.index_select(dim, idx)
        state[key] = tensor.to(dtype).contiguous()
    mlp.load_state_dict(state, strict=True)

    pre_norm = _load_norm(found["pre_norm"], found["pre_norm_prefix"], tensors, dtype)
    post_norm = None
    if found["post_norm"] is not None:
        post_norm = _load_norm(found["post_norm"], found["post_norm_prefix"], tensors, dtype)

    cluster = MLPCluster(pre_norm, mlp, post_norm)
    cluster.eval()
    for param in cluster.parameters():
        param.requires_grad_(False)
    return cluster


def find_moe_block(config, layer):
    """Find the sparse MoE block of one donor layer, with its norms.

    Builds the donor on the meta device. Returns a dict with moe, moe_prefix,
    expert_cls, num_experts, pre_norm, pre_norm_prefix, post_norm and
    post_norm_prefix. Raises ValueError when the layer has no MoE block and
    NotImplementedError when the experts are stored as fused tensors.
    """
    model = _build_meta_model(config)
    to_module, to_checkpoint = _name_rules(model)
    list_name, layers = find_decoder_layers(model, layer)
    layer_module = layers[layer]
    layer_prefix = f"{list_name}.{layer}"

    moe = None
    moe_name = None
    for name, child in layer_module.named_children():
        if hasattr(child, "experts"):
            moe = child
            moe_name = name
    if moe is None:
        raise ValueError(f"layer {layer} has no MoE block (no child with an experts attribute)")
    if not isinstance(moe.experts, nn.ModuleList):
        model_type = getattr(_text_config(config), "model_type", "unknown")
        raise NotImplementedError(
            f"experts of model type {model_type} are fused tensors, not a ModuleList; not supported yet"
        )

    if hasattr(layer_module, "pre_feedforward_layernorm"):
        pre_name = "pre_feedforward_layernorm"
    else:
        pre_name = "post_attention_layernorm"

    result = {}
    result["to_module"] = to_module
    result["to_checkpoint"] = to_checkpoint
    result["moe"] = moe
    result["moe_prefix"] = f"{layer_prefix}.{moe_name}"
    result["expert_cls"] = type(moe.experts[0])
    result["num_experts"] = len(moe.experts)
    result["pre_norm"] = getattr(layer_module, pre_name)
    result["pre_norm_prefix"] = f"{layer_prefix}.{pre_name}"
    if hasattr(layer_module, "post_feedforward_layernorm"):
        result["post_norm"] = layer_module.post_feedforward_layernorm
        result["post_norm_prefix"] = f"{layer_prefix}.post_feedforward_layernorm"
    else:
        result["post_norm"] = None
        result["post_norm_prefix"] = None
    return result


def expert_tensor_names(config, layer, experts):
    """List the checkpoint tensor names needed for the given expert ids."""
    found = find_moe_block(config, layer)
    names = []
    for e in experts:
        names += _prefixed_keys(found["moe"].experts[int(e)], f"{found['moe_prefix']}.experts.{int(e)}")
    names += _prefixed_keys(found["pre_norm"], found["pre_norm_prefix"])
    if found["post_norm"] is not None:
        names += _prefixed_keys(found["post_norm"], found["post_norm_prefix"])
    return _checkpoint_names(found, names)


class ExpertPart(nn.Module):
    """post_norm(sum of experts(pre_norm(x))), with the post norm optional."""

    def __init__(self, pre_norm, experts, post_norm=None):
        super().__init__()
        self.pre_norm = pre_norm
        self.experts = nn.ModuleList(experts)
        self.post_norm = post_norm

    def forward(self, x):
        normed = self.pre_norm(x)
        out = None
        for expert in self.experts:
            piece = expert(normed)
            if out is None:
                out = piece
            else:
                out = out + piece
        if self.post_norm is not None:
            out = self.post_norm(out)
        return out


def _shapes(module):
    shapes = {}
    for key, value in module.state_dict().items():
        shapes[key] = tuple(value.shape)
    return shapes


def _make_expert(expert_cls, text_config, meta_expert):
    """Build one expert module on CPU whose tensor shapes match the donor's."""
    wanted = _shapes(meta_expert)
    tries = [
        lambda: expert_cls(text_config),
        lambda: expert_cls(text_config, intermediate_size=text_config.moe_intermediate_size),
    ]
    for make in tries:
        try:
            module = make()
        except (TypeError, AttributeError):
            continue
        if _shapes(module) == wanted:
            return module
    raise ValueError(f"cannot build {expert_cls.__name__} with the donor's tensor shapes")


def build_expert_part(config, layer, experts, tensors):
    """Build an expert part from a donor layer.

    experts: list or LongTensor of expert ids.
    tensors: dict of full checkpoint name -> tensor.
    The result is in eval mode with all parameters frozen.
    """
    found = find_moe_block(config, layer)
    tensors = _module_named(tensors, found)
    text_config = _text_config(config)
    ids = []
    for e in torch.as_tensor(experts, dtype=torch.long).tolist():
        ids.append(int(e))

    first = f"{found['moe_prefix']}.experts.{ids[0]}"
    first_key = _prefixed_keys(found["moe"].experts[ids[0]], first)[0]
    dtype = tensors[first_key].dtype

    modules = []
    for e in ids:
        meta_expert = found["moe"].experts[e]
        module = _make_expert(found["expert_cls"], text_config, meta_expert)
        module = module.to(dtype)
        state = {}
        for key in meta_expert.state_dict().keys():
            state[key] = tensors[f"{found['moe_prefix']}.experts.{e}.{key}"].to(dtype).contiguous()
        module.load_state_dict(state, strict=True)
        modules.append(module)

    pre_norm = _load_norm(found["pre_norm"], found["pre_norm_prefix"], tensors, dtype)
    post_norm = None
    if found["post_norm"] is not None:
        post_norm = _load_norm(found["post_norm"], found["post_norm_prefix"], tensors, dtype)

    part = ExpertPart(pre_norm, modules, post_norm)
    part.eval()
    for param in part.parameters():
        param.requires_grad_(False)
    return part


def _head_dim(text_config):
    """Return the per-head width of a donor config."""
    head_dim = getattr(text_config, "head_dim", None)
    if head_dim is None:
        head_dim = text_config.hidden_size // text_config.num_attention_heads
    return head_dim


def _kv_head_count(text_config):
    count = getattr(text_config, "num_key_value_heads", None)
    if count is None:
        count = text_config.num_attention_heads
    return count


def attn_tensor_names(config, layer):
    """List the checkpoint tensor names an attention part needs for one layer."""
    found = find_layer_modules(config, layer)
    names = _prefixed_keys(found["attn"], found["attn_prefix"])
    names += _prefixed_keys(found["input_norm"], found["input_norm_prefix"])
    if found["post_attn_norm"] is not None:
        names += _prefixed_keys(found["post_attn_norm"], found["post_attn_norm_prefix"])
    return _checkpoint_names(found, names)


def causal_mask(batch, seq, dtype, device, window=None):
    """Return an additive causal mask of shape (batch, 1, seq, seq).

    Allowed places hold 0. Blocked places hold the most negative number of
    the dtype. If window is set, a token sees only the last `window` tokens.
    """
    blocked = torch.finfo(dtype).min
    rows = torch.arange(seq, device=device).view(seq, 1)
    cols = torch.arange(seq, device=device).view(1, seq)
    allowed = cols <= rows
    if window is not None:
        allowed = allowed & (cols > rows - window)
    mask = torch.zeros(seq, seq, dtype=dtype, device=device)
    mask = mask.masked_fill(~allowed, blocked)
    return mask.view(1, 1, seq, seq).expand(batch, 1, seq, seq)


class AttnHeadsPart(nn.Module):
    """post_norm(attn(input_norm(x))) for a subset of donor query heads.

    x has shape (batch, seq, d_donor) and holds the whole sequence. There is
    no KV cache, so this part needs use_cache=False when the host generates.
    attention_mask is the host's 2D padding mask (batch, seq), with 1 for
    real tokens and 0 for pads, or None. With it, positions count only real
    tokens and no token attends to a pad. The rotary module is kept out of
    the module tree, so a dtype change on the part does not round its
    frequencies.
    """

    def __init__(self, input_norm, attn, rotary, post_norm=None):
        super().__init__()
        self.input_norm = input_norm
        self.attn = attn
        self.post_norm = post_norm
        self._rotary = [rotary]

    def forward(self, x, attention_mask=None):
        batch, seq, _ = x.shape
        rotary = self._rotary[0]
        if rotary.inv_freq.device != x.device:
            rotary.to(x.device)
        if attention_mask is not None:
            if tuple(attention_mask.shape) != (batch, seq):
                raise ValueError(
                    f"attention_mask has shape {tuple(attention_mask.shape)} but the input is "
                    f"({batch}, {seq}); an attn_heads part needs the whole sequence, "
                    "so run the host with use_cache=False"
                )
            attention_mask = attention_mask.to(x.device)
            position_ids = (attention_mask.long().cumsum(-1) - 1).clamp(min=0)
        else:
            position_ids = torch.arange(seq, device=x.device).view(1, seq).expand(batch, seq)
        normed = self.input_norm(x)
        cos_sin = rotary(normed, position_ids)
        # Mistral, Mixtral and Phi-3 attention keep the window only in the
        # config. Qwen and Gemma set the attribute, and None on full layers.
        if hasattr(self.attn, "sliding_window"):
            window = self.attn.sliding_window
        else:
            window = getattr(self.attn.config, "sliding_window", None)
        mask = causal_mask(batch, seq, normed.dtype, x.device, window)
        if attention_mask is not None:
            # causal_mask returns an expanded view, so clone before the fill.
            pad_keys = attention_mask[:, None, None, :] == 0
            mask = mask.clone().masked_fill(pad_keys, torch.finfo(mask.dtype).min)
        out = self.attn(hidden_states=normed, position_embeddings=cos_sin, attention_mask=mask)[0]
        if self.post_norm is not None:
            out = self.post_norm(out)
        return out


def _head_rows(heads, head_dim):
    """Return the row ids of the given heads, in order."""
    rows = []
    for head in heads:
        for i in range(head_dim):
            rows.append(int(head) * head_dim + i)
    return torch.tensor(rows, dtype=torch.long)


def build_attn_heads(config, layer, heads, tensors):
    """Build a part that runs a subset of query heads of one donor layer.

    heads: query head ids. Every selected head gets its own copy of the KV
    head it uses, so the new layer has len(heads) query heads and len(heads)
    KV heads, which is always a valid grouping. The donor's own attention
    class runs the math. The result is in eval mode with frozen parameters.
    """
    found = find_layer_modules(config, layer)
    tensors = _module_named(tensors, found)
    text_config = _text_config(config)
    heads = torch.as_tensor(heads, dtype=torch.long)
    n = len(heads)

    head_dim = _head_dim(text_config)
    num_heads = text_config.num_attention_heads
    group = num_heads // _kv_head_count(text_config)

    kv_heads = []
    for head in heads.tolist():
        kv_heads.append(head // group)
    q_rows = _head_rows(heads.tolist(), head_dim)
    kv_rows = _head_rows(kv_heads, head_dim)

    small = copy.deepcopy(text_config)
    small.head_dim = head_dim
    small.num_attention_heads = n
    small.num_key_value_heads = n
    small._attn_implementation = "eager"

    attn_prefix = found["attn_prefix"]
    first_name = _prefixed_keys(found["attn"], attn_prefix)[0]
    dtype = tensors[first_name].dtype

    attn = type(found["attn"])(small, layer)
    attn = attn.to(dtype)
    state = {}
    for key in found["attn"].state_dict().keys():
        tensor = tensors[f"{attn_prefix}.{key}"]
        owner = key.split(".")[0]
        if owner == "q_proj":
            tensor = tensor.index_select(0, q_rows.to(tensor.device))
        elif owner in ("k_proj", "v_proj"):
            tensor = tensor.index_select(0, kv_rows.to(tensor.device))
        elif owner == "qkv_proj":
            # Phi-3 and Phi-4 fuse the rows as [all q | all k | all v].
            q_size = num_heads * head_dim
            kv_size = _kv_head_count(text_config) * head_dim
            rows = torch.cat([q_rows, kv_rows + q_size, kv_rows + q_size + kv_size])
            tensor = tensor.index_select(0, rows.to(tensor.device))
        elif key == "o_proj.weight":
            tensor = tensor.index_select(1, q_rows.to(tensor.device))
        state[key] = tensor.to(dtype).contiguous()
    attn.load_state_dict(state, strict=True)

    input_norm = _load_norm(found["input_norm"], found["input_norm_prefix"], tensors, dtype)
    post_norm = None
    if found["post_attn_norm"] is not None:
        post_norm = _load_norm(found["post_attn_norm"], found["post_attn_norm_prefix"], tensors, dtype)

    rotary_config = copy.deepcopy(text_config)
    rotary_config.head_dim = head_dim
    # Gemma 3 sliding layers use a local rotary with its own base and no
    # scaling, like Gemma3TextModel.rotary_emb_local.
    local_base = getattr(text_config, "rope_local_base_freq", None)
    if getattr(found["attn"], "is_sliding", False) and local_base is not None:
        rotary_config.rope_theta = local_base
        rotary_config.rope_scaling = {"rope_type": "default"}
    rotary = type(found["rotary"])(config=rotary_config)

    part = AttnHeadsPart(input_norm, attn, rotary, post_norm)
    part.eval()
    for param in part.parameters():
        param.requires_grad_(False)
    return part
