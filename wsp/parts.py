"""Parts: pieces of a donor model that can be grafted into a host.

A part runs the donor's own Transformers code (its MLP class and its norm
classes), so any family that Transformers supports works. We find the
classes by building the donor on the meta device from its config alone,
which allocates no memory and downloads no weights.
"""

import copy

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


def find_layer_modules(config, layer):
    """Find the MLP and its norms for one donor layer.

    Builds the donor on the meta device. Returns a dict with mlp, mlp_prefix,
    pre_norm, pre_norm_prefix, post_norm, post_norm_prefix. The prefixes are
    the dotted names used in the checkpoint. post_norm is None when the
    layer has no post-feedforward norm.
    """
    import transformers

    with torch.device("meta"):
        model = transformers.AutoModelForCausalLM.from_config(config)
    list_name, layers = find_decoder_layers(model, layer)
    layer_module = layers[layer]
    layer_prefix = f"{list_name}.{layer}"

    if hasattr(layer_module, "pre_feedforward_layernorm"):
        pre_name = "pre_feedforward_layernorm"
    else:
        pre_name = "post_attention_layernorm"

    result = {}
    result["mlp"] = layer_module.mlp
    result["mlp_prefix"] = f"{layer_prefix}.mlp"
    result["pre_norm"] = getattr(layer_module, pre_name)
    result["pre_norm_prefix"] = f"{layer_prefix}.{pre_name}"
    if hasattr(layer_module, "post_feedforward_layernorm"):
        result["post_norm"] = layer_module.post_feedforward_layernorm
        result["post_norm_prefix"] = f"{layer_prefix}.post_feedforward_layernorm"
    else:
        result["post_norm"] = None
        result["post_norm_prefix"] = None
    return result


def _prefixed_keys(module, prefix):
    """Return the state_dict keys of a module with a prefix added."""
    names = []
    for key in module.state_dict().keys():
        names.append(f"{prefix}.{key}")
    return names


def mlp_tensor_names(config, layer):
    """List the checkpoint tensor names a cluster needs for one layer."""
    found = find_layer_modules(config, layer)
    names = _prefixed_keys(found["mlp"], found["mlp_prefix"])
    names += _prefixed_keys(found["pre_norm"], found["pre_norm_prefix"])
    if found["post_norm"] is not None:
        names += _prefixed_keys(found["post_norm"], found["post_norm_prefix"])
    return names


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
        for dim in range(tensor.dim()):
            if tensor.shape[dim] == full_size:
                tensor = tensor.index_select(dim, indices.to(tensor.device))
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
