"""Parts: pieces of a donor model that can be grafted into a host.

A part runs the donor's own Transformers code (its MLP class and its norm
classes), so any family that Transformers supports works. We find the
classes by building the donor on the meta device from its config alone,
which allocates no memory and downloads no weights.
"""

import copy
import inspect
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

    # Import the model class BEFORE entering the device context. transformers
    # model modules import torch._dynamo, and that import fails with a
    # circular import under an active torch.device context. In a fresh
    # process this showed as "Could not find Qwen2ForCausalLM" (round 24).
    try:
        import torch._dynamo  # noqa: F401
    except Exception:
        pass
    try:
        transformers.AutoModelForCausalLM._model_mapping[type(config)]
    except Exception:
        pass

    with torch.device("meta"):
        try:
            return transformers.AutoModelForCausalLM.from_config(config)
        except ValueError:
            # Some multimodal configs (for example mistral3) have no causal
            # LM class. The image-text class holds the same language model.
            return transformers.AutoModelForImageTextToText.from_config(config)


def _transformers_5():
    """True on Transformers 5.x. Without Transformers installed, False, so the module still imports."""
    try:
        import transformers
    except ImportError:
        return False
    return int(transformers.__version__.split(".")[0]) >= 5


def _legacy_name_rules(model):
    """Transformers 4 rules: (pattern, replacement) pairs from _checkpoint_conversion_mapping."""
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


def _name_rules(model):
    """Return the rename rules between checkpoint names and module names.

    Multimodal models such as Gemma 3 and Mistral 3 keep old names in their
    checkpoints (language_model.model.layers.N) while their modules are
    named model.language_model.layers.N. Transformers 5 keeps these renames
    in its conversion mapping: loading applies them forward and saving
    applies them in reverse. We use the same rules. Returns (to_module,
    to_checkpoint), two lists of Transformers WeightRenaming objects. Plain
    LMs have no renames, so both lists are empty.
    """
    if not _transformers_5():
        return _legacy_name_rules(model)
    from transformers.conversion_mapping import get_model_conversion_mapping
    from transformers.core_model_loading import PrefixChange, WeightRenaming

    to_module = []
    for transform in get_model_conversion_mapping(model):
        if isinstance(transform, WeightRenaming):
            to_module.append(transform)
    # Saving runs the transforms in reverse order and leaves out the legacy
    # renames. It also drops prefix changes for a model built from a config,
    # as ours are.
    to_checkpoint = []
    saved = get_model_conversion_mapping(model, add_legacy=False)
    for transform in reversed(saved):
        if isinstance(transform, PrefixChange):
            continue
        if isinstance(transform, WeightRenaming):
            to_checkpoint.append(transform.reverse_transform())
    return to_module, to_checkpoint


def _expert_converters(model):
    """Return the Transformers converters that fuse per-expert checkpoint tensors."""
    if not _transformers_5():
        return []
    from transformers.conversion_mapping import get_model_conversion_mapping
    from transformers.core_model_loading import WeightConverter

    converters = []
    for transform in get_model_conversion_mapping(model):
        if isinstance(transform, WeightConverter):
            converters.append(transform)
    return converters


def _rename(name, rules):
    """Apply the rules like Transformers does.

    Transformers 5 rules are objects applied in turn. Transformers 4 rules
    are (pattern, replacement) pairs and the first match wins.
    """
    for rule in rules:
        if isinstance(rule, tuple):
            new_name, count = re.subn(rule[0], rule[1], name)
            if count > 0:
                return new_name
            continue
        name, _ = rule.rename_source_key(name)
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


# Layer attribute names differ by family. LFM2 and LFM3 call the MLP
# feed_forward, the pre-MLP norm ffn_norm, and the pre-attention (or
# pre-conv) norm operator_norm. LFM3 adds the sandwich norms
# post_operator_norm and post_ffn_norm. The first name a layer has wins.
MLP_NAMES = ["mlp", "feed_forward"]
PRE_MLP_NORM_NAMES = ["pre_feedforward_layernorm", "post_attention_layernorm", "ffn_norm"]
POST_MLP_NORM_NAMES = ["post_feedforward_layernorm", "post_ffn_norm"]
INPUT_NORM_NAMES = ["input_layernorm", "operator_norm"]


def _first_child(layer_module, names):
    """Return the first name in names that the layer has, or None.

    An nn.Identity child counts as missing (LFM3 without sandwich norms).
    """
    for name in names:
        child = getattr(layer_module, name, None)
        if child is None or isinstance(child, nn.Identity):
            continue
        return name
    return None


def find_layer_modules(config, layer):
    """Find the MLP, the attention, and their norms for one donor layer.

    Builds the donor on the meta device. Returns a dict with mlp, mlp_prefix,
    pre_norm, pre_norm_prefix, post_norm, post_norm_prefix and the attention
    keys. The prefixes are module names in the donor model, so get_submodule
    works with them. Use _checkpoint_names to get the checkpoint tensor
    names. mlp is None when the layer has no MLP attribute. post_norm is None
    when the layer has no post-feedforward norm.
    """
    model = _build_meta_model(config)
    to_module, to_checkpoint = _name_rules(model)
    list_name, layers = find_decoder_layers(model, layer)
    layer_module = layers[layer]
    layer_prefix = f"{list_name}.{layer}"

    pre_name = _first_child(layer_module, PRE_MLP_NORM_NAMES)
    mlp_name = _first_child(layer_module, MLP_NAMES)
    if mlp_name is None:
        mlp_name = "mlp"

    result = {}
    result["to_module"] = to_module
    result["to_checkpoint"] = to_checkpoint
    result["mlp"] = getattr(layer_module, mlp_name, None)
    result["mlp_prefix"] = f"{layer_prefix}.{mlp_name}"
    result["pre_norm"] = None
    result["pre_norm_prefix"] = None
    if pre_name is not None:
        result["pre_norm"] = getattr(layer_module, pre_name)
        result["pre_norm_prefix"] = f"{layer_prefix}.{pre_name}"
    post_name = _first_child(layer_module, POST_MLP_NORM_NAMES)
    result["post_norm"] = None
    result["post_norm_prefix"] = None
    if post_name is not None:
        result["post_norm"] = getattr(layer_module, post_name)
        result["post_norm_prefix"] = f"{layer_prefix}.{post_name}"
    _add_attention_modules(result, model, layer_module, layer_prefix)
    return result


def _add_attention_modules(result, model, layer_module, layer_prefix):
    """Add the attention keys to a find_layer_modules result.

    attn is None for a layer with no self_attn (an LFM2 conv layer, for
    example).
    """
    result["attn"] = getattr(layer_module, "self_attn", None)
    result["attn_prefix"] = f"{layer_prefix}.self_attn"
    input_name = _first_child(layer_module, INPUT_NORM_NAMES)
    if input_name is None:
        input_name = "input_layernorm"
    result["input_norm"] = getattr(layer_module, input_name, None)
    result["input_norm_prefix"] = f"{layer_prefix}.{input_name}"
    result["layer_prefix"] = layer_prefix
    result["post_attn_norm"] = None
    result["post_attn_norm_prefix"] = None
    # Gemma uses post_attention_layernorm after attention when it also has
    # a pre_feedforward_layernorm. LFM3 calls it post_operator_norm.
    post_name = None
    if hasattr(layer_module, "pre_feedforward_layernorm"):
        post_name = "post_attention_layernorm"
    else:
        post_name = _first_child(layer_module, ["post_operator_norm"])
    if post_name is not None:
        result["post_attn_norm"] = getattr(layer_module, post_name)
        result["post_attn_norm_prefix"] = f"{layer_prefix}.{post_name}"
    result["rotary"] = None
    best_score = -1
    for name, module in model.named_modules():
        # Most models name the rotary rotary_emb. LFM2-MoE calls it pos_emb,
        # so we also match the class name.
        class_name = type(module).__name__
        if not name.endswith("rotary_emb") and not class_name.endswith("RotaryEmbedding"):
            continue
        score = 1
        if "language_model" in name or "text_model" in name:
            score = 3
        elif "vision" in name or "visual" in name or "Vision" in class_name:
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


def set_mlp_width(text_config, n):
    """Set the MLP neuron count of a config to n.

    LFM2 shrinks intermediate_size by 2/3 and rounds it when
    block_auto_adjust_ff_dim is on, so we turn that off to get n exactly.
    """
    text_config.intermediate_size = n
    if getattr(text_config, "block_auto_adjust_ff_dim", False):
        text_config.block_auto_adjust_ff_dim = False


def _mlp_width(mlp, text_config):
    """Return the number of neurons of a (meta) donor MLP.

    This is the config's intermediate_size when some MLP tensor has that
    size. Otherwise (LFM2 with block_auto_adjust_ff_dim) it is the input
    width of the last linear layer, the down projection. With
    block_auto_adjust_ff_dim on we always read the down projection, as the
    config size can equal the hidden size by chance.
    """
    width = text_config.intermediate_size
    if not getattr(text_config, "block_auto_adjust_ff_dim", False):
        for value in mlp.state_dict().values():
            if width in tuple(value.shape):
                return width
    last = None
    for child in mlp.children():
        if isinstance(child, nn.Linear):
            last = child
    if last is None:
        raise ValueError(f"cannot find the neuron count of {type(mlp).__name__}")
    return last.in_features


def mlp_width(config, layer):
    """Return the number of MLP neurons in one donor layer."""
    found = find_layer_modules(config, layer)
    _require_dense_mlp(found, layer)
    return _mlp_width(found["mlp"], _text_config(config))


def _make_mlp(mlp_cls, config, n, layer):
    """Build the donor MLP class of one layer with n neurons.

    Most MLP classes take the config, some (Qwen MoE) an intermediate_size
    too, and Gemma 4 the layer index.
    """
    small = copy.deepcopy(config)
    set_mlp_width(small, n)
    # Gemma 4 doubles the width on layers that reuse KV states. n is
    # already the width we want.
    if getattr(small, "use_double_wide_mlp", False):
        small.use_double_wide_mlp = False
    try:
        return mlp_cls(small)
    except TypeError:
        pass
    try:
        return mlp_cls(small, intermediate_size=n)
    except TypeError:
        pass
    return mlp_cls(small, layer)


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
    full_size = _mlp_width(found["mlp"], text_config)
    indices = torch.as_tensor(indices, dtype=torch.long)
    n = len(indices)

    mlp_prefix = found["mlp_prefix"]
    first_name = _prefixed_keys(found["mlp"], mlp_prefix)[0]
    dtype = tensors[first_name].dtype

    mlp = _make_mlp(type(found["mlp"]), text_config, n, layer)
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
    experts (the meta experts module), fused, num_experts, expert_cls,
    converters, pre_norm, pre_norm_prefix, post_norm and post_norm_prefix.

    In Transformers 5 most MoE classes keep all experts in one module with
    3D tensors whose first dimension is the expert id. Then fused is True,
    expert_cls is None, and converters holds the Transformers converters
    that build those tensors from per-expert checkpoint tensors. Older
    classes keep a ModuleList of expert modules. Then fused is False.
    Raises ValueError when the layer has no MoE block.
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

    pre_name = _first_child(layer_module, PRE_MLP_NORM_NAMES)
    if pre_name is None:
        raise ValueError(f"layer {layer} has no norm before its MoE block")

    result = {}
    result["to_module"] = to_module
    result["to_checkpoint"] = to_checkpoint
    result["moe"] = moe
    result["moe_prefix"] = f"{layer_prefix}.{moe_name}"
    result["experts"] = moe.experts
    if isinstance(moe.experts, nn.ModuleList):
        result["fused"] = False
        result["expert_cls"] = type(moe.experts[0])
        result["num_experts"] = len(moe.experts)
        result["converters"] = []
    else:
        result["fused"] = True
        result["expert_cls"] = None
        names = _fused_param_names(moe.experts)
        result["num_experts"] = getattr(moe.experts, "num_experts", None)
        if result["num_experts"] is None:
            result["num_experts"] = getattr(moe.experts, names[0]).shape[0]
        result["converters"] = _expert_converters(model)
    result["pre_norm"] = getattr(layer_module, pre_name)
    result["pre_norm_prefix"] = f"{layer_prefix}.{pre_name}"
    post_name = _first_child(layer_module, POST_MLP_NORM_NAMES)
    result["post_norm"] = None
    result["post_norm_prefix"] = None
    if post_name is not None:
        result["post_norm"] = getattr(layer_module, post_name)
        result["post_norm_prefix"] = f"{layer_prefix}.{post_name}"
    return result


def _fused_param_names(experts):
    """Return the parameter names of a fused experts module, in order.

    Every parameter has the expert id as its first dimension.
    """
    names = []
    for name, _ in experts.named_parameters(recurse=False):
        names.append(name)
    total = 0
    for _ in experts.parameters():
        total += 1
    if total != len(names):
        raise NotImplementedError(f"{type(experts).__name__} keeps parameters in child modules; not supported")
    if len(names) == 0:
        raise ValueError(f"{type(experts).__name__} has no parameters")
    return names


def _expert_sources(found, param_name, expert):
    """Return (converter, names) for one expert's share of a fused tensor.

    Most checkpoints store each expert's tensors on their own, for example
    mlp.experts.3.gate_proj.weight and mlp.experts.3.up_proj.weight, and
    Transformers fuses them into mlp.experts.gate_up_proj when it loads.
    names are the module-side names of the expert's source tensors, in the
    converter's order. Returns (None, []) when no converter builds the
    tensor, which means the checkpoint stores the fused tensor itself.
    """
    full = f"{found['moe_prefix']}.experts.{param_name}"
    for converter in found["converters"]:
        backward = converter.reverse_transform()
        match = backward.compiled_sources.search(full)
        if match is None:
            continue
        names = []
        for pattern in converter.source_patterns:
            piece = pattern.replace("\\", "").replace("*", str(expert))
            names.append(full.replace(match.group(0), piece, 1))
        return converter, names
    return None, []


def expert_tensor_names(config, layer, experts):
    """List the checkpoint tensor names needed for the given expert ids.

    When a fused experts module has no converter, the checkpoint stores the
    fused tensors, so the names cover all experts of the layer.
    """
    found = find_moe_block(config, layer)
    names = []
    if not found["fused"]:
        for e in experts:
            names += _prefixed_keys(found["moe"].experts[int(e)], f"{found['moe_prefix']}.experts.{int(e)}")
    else:
        for param_name in _fused_param_names(found["experts"]):
            converter, _ = _expert_sources(found, param_name, 0)
            if converter is None:
                names.append(f"{found['moe_prefix']}.experts.{param_name}")
                continue
            for e in experts:
                _, sources = _expert_sources(found, param_name, int(e))
                for name in sources:
                    if name not in names:
                        names.append(name)
    names += _prefixed_keys(found["pre_norm"], found["pre_norm_prefix"])
    if found["post_norm"] is not None:
        names += _prefixed_keys(found["post_norm"], found["post_norm_prefix"])
    return _checkpoint_names(found, names)


class ExpertPart(nn.Module):
    """post_norm(sum of experts(pre_norm(x))), with the post norm optional.

    experts is a list of expert modules, or one fused experts module (a
    Transformers 5 MoE class) that holds only the chosen experts. The fused
    module runs every token through each of its experts with weight 1.
    """

    def __init__(self, pre_norm, experts, post_norm=None):
        super().__init__()
        self.pre_norm = pre_norm
        if isinstance(experts, (list, tuple)):
            self.experts = nn.ModuleList(experts)
            self.fused = False
        else:
            self.experts = experts
            self.fused = True
        self.post_norm = post_norm

    def forward(self, x):
        normed = self.pre_norm(x)
        if self.fused:
            flat = normed.reshape(-1, normed.shape[-1])
            count = self.experts.num_experts
            index = torch.arange(count, device=flat.device).expand(flat.shape[0], count)
            weights = torch.ones(flat.shape[0], count, dtype=flat.dtype, device=flat.device)
            out = self.experts(flat, index, weights)
            out = out.reshape(tuple(normed.shape[:-1]) + (out.shape[-1],))
        else:
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


def _fused_expert_tensor(found, param_name, ids, tensors, text_config):
    """Return one fused expert tensor for the chosen experts only.

    The first dimension of the result has len(ids) entries. tensors uses
    module-side names. It holds either the fused tensor of all experts,
    which we slice, or per-expert tensors, which we fuse with the donor's
    own Transformers converter, the same way from_pretrained does.
    """
    full = f"{found['moe_prefix']}.experts.{param_name}"
    if full in tensors:
        tensor = tensors[full]
        return tensor.index_select(0, torch.tensor(ids, dtype=torch.long, device=tensor.device))
    converter, _ = _expert_sources(found, param_name, ids[0])
    if converter is None:
        raise KeyError(f"tensor not found: {full}")
    collected = {}
    for pattern in converter.source_patterns:
        collected[pattern] = []
    for e in ids:
        _, sources = _expert_sources(found, param_name, e)
        for index in range(len(sources)):
            collected[converter.source_patterns[index]].append(tensors[sources[index]])
    for op in converter.operations:
        collected = op.convert(collected, source_patterns=converter.source_patterns,
                               target_patterns=converter.target_patterns, full_layer_name=full,
                               config=text_config)
    values = list(collected.values())
    if len(values) != 1:
        raise ValueError(f"the converter for {full} gave {len(values)} tensors, expected 1")
    return values[0]


def _build_fused_experts(found, ids, tensors, text_config):
    """Build a copy of the donor's fused experts module that holds only experts ids."""
    meta = found["experts"]
    for name, _ in meta.named_buffers():
        raise NotImplementedError(f"{type(meta).__name__} has buffer {name}; not supported")
    module = copy.deepcopy(meta)
    for name in _fused_param_names(meta):
        tensor = _fused_expert_tensor(found, name, ids, tensors, text_config)
        setattr(module, name, nn.Parameter(tensor.contiguous(), requires_grad=False))
    module.num_experts = len(ids)
    if hasattr(module, "config"):
        # Run the plain loop over experts, as we use eager attention.
        module.config = copy.deepcopy(module.config)
        module.config._experts_implementation = "eager"
    return module


def build_expert_part(config, layer, experts, tensors):
    """Build an expert part from a donor layer.

    experts: list or LongTensor of expert ids.
    tensors: dict of full checkpoint name -> tensor (module names work too).
    The result is in eval mode with all parameters frozen.
    """
    found = find_moe_block(config, layer)
    tensors = _module_named(tensors, found)
    text_config = _text_config(config)
    ids = []
    for e in torch.as_tensor(experts, dtype=torch.long).tolist():
        ids.append(int(e))

    if found["fused"]:
        experts_part = _build_fused_experts(found, ids, tensors, text_config)
        dtype = next(experts_part.parameters()).dtype
    else:
        first = f"{found['moe_prefix']}.experts.{ids[0]}"
        first_key = _prefixed_keys(found["moe"].experts[ids[0]], first)[0]
        dtype = tensors[first_key].dtype
        experts_part = []
        for e in ids:
            meta_expert = found["moe"].experts[e]
            module = _make_expert(found["expert_cls"], text_config, meta_expert)
            module = module.to(dtype)
            state = {}
            for key in meta_expert.state_dict().keys():
                state[key] = tensors[f"{found['moe_prefix']}.experts.{e}.{key}"].to(dtype).contiguous()
            module.load_state_dict(state, strict=True)
            experts_part.append(module)

    pre_norm = _load_norm(found["pre_norm"], found["pre_norm_prefix"], tensors, dtype)
    post_norm = None
    if found["post_norm"] is not None:
        post_norm = _load_norm(found["post_norm"], found["post_norm_prefix"], tensors, dtype)

    part = ExpertPart(pre_norm, experts_part, post_norm)
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
    _require_attention(found, layer)
    names = _prefixed_keys(found["attn"], found["attn_prefix"])
    names += _prefixed_keys(found["input_norm"], found["input_norm_prefix"])
    if found["post_attn_norm"] is not None:
        names += _prefixed_keys(found["post_attn_norm"], found["post_attn_norm_prefix"])
    return _checkpoint_names(found, names)


def _require_attention(found, layer):
    """Raise ValueError when the layer has no self_attn (an LFM2 conv layer, for example)."""
    if found["attn"] is None:
        raise ValueError(f"layer {layer} has no self_attn module (it may be a conv layer); "
                         "pick an attention layer for kind 'attn_heads'")
    # Gemma 4 attention takes the KV states shared between layers and has
    # its own head sizes per layer. Our head slicing does not handle that.
    params = inspect.signature(found["attn"].forward).parameters
    if "shared_kv_states" in params:
        raise NotImplementedError(f"{type(found['attn']).__name__} shares KV states between layers; "
                                  "attn_heads does not support it yet")


def _layer_type(text_config, layer):
    """Return the config's layer type of a donor layer, or None."""
    layer_types = getattr(text_config, "layer_types", None)
    if layer_types is None or layer >= len(layer_types):
        return None
    return layer_types[layer]


def _rotary_to(rotary, device):
    """Move a rotary module to device when its buffers are on another device."""
    for buffer in rotary.buffers():
        if buffer.device != device:
            rotary.to(device)
        return


def _takes_layer_type(rotary):
    """True when the rotary module keeps one set of frequencies per layer type (Gemma 3, LFM3)."""
    return "layer_type" in inspect.signature(rotary.forward).parameters


def causal_mask(batch, seq, dtype, device, window=None, past=0):
    """Return an additive causal mask of shape (batch, 1, seq, past + seq).

    Allowed places hold 0. Blocked places hold the most negative number of
    the dtype. The queries are the last seq positions: query i sits at
    position past + i and the keys are positions 0 to past + seq - 1. If
    window is set, a token sees only the last `window` positions. With
    past 0 this is the plain (seq, seq) causal mask.
    """
    blocked = torch.finfo(dtype).min
    total = past + seq
    rows = torch.arange(past, total, device=device).view(seq, 1)
    cols = torch.arange(total, device=device).view(1, total)
    allowed = cols <= rows
    if window is not None:
        allowed = allowed & (cols > rows - window)
    mask = torch.zeros(seq, total, dtype=dtype, device=device)
    mask = mask.masked_fill(~allowed, blocked)
    return mask.view(1, 1, seq, total).expand(batch, 1, seq, total)


def _start_step(part, x, attention_mask, cache_len, kind):
    """Get a part ready for one forward call. Return (past, attention_mask, position_ids).

    cache_len is None when the host runs without a KV cache. Then the part
    drops its own cache and x holds the whole sequence. cache_len 0 means a
    new prefill with a cache: the part starts a new cache and fills it.
    A larger cache_len is a decode step: the host cache held cache_len
    tokens before this call, x holds only the new tokens, and the part
    cache must hold the same number of tokens.

    part.cache_tokens counts the tokens in the part cache. We count them
    ourselves, because conv cache layers do not track a sequence length.

    attention_mask is the 2D padding mask (batch, past + new) or None. The
    position ids count only real tokens and are returned for the new
    tokens only. past is the number of cached tokens before this call.
    """
    batch, seq, _ = x.shape
    if cache_len is None:
        part.cache = None
        part.cache_tokens = 0
        past = 0
    elif cache_len == 0:
        part.cache = part.new_cache()
        part.cache_tokens = 0
        past = 0
    else:
        own = 0
        if part.cache is not None:
            own = part.cache_tokens
        if own != cache_len:
            raise ValueError(
                f"{kind} part cache is out of sync: it holds {own} tokens but the host cache "
                f"held {cache_len} before this call; start again with a new prefill "
                "(an empty cache and use_cache=True)"
            )
        past = cache_len
    total = past + seq
    if part.cache is not None:
        part.cache_tokens = total

    if attention_mask is not None:
        if tuple(attention_mask.shape) != (batch, total):
            raise ValueError(
                f"attention_mask has shape {tuple(attention_mask.shape)} but {kind} part expects "
                f"({batch}, {total}) for {past} cached and {seq} new tokens; for a cached decode "
                "step the part needs cache_len, the host cache length before the call"
            )
        attention_mask = attention_mask.to(x.device)
        position_ids = (attention_mask.long().cumsum(-1) - 1).clamp(min=0)
        position_ids = position_ids[:, past:]
    else:
        position_ids = torch.arange(past, total, device=x.device).view(1, seq).expand(batch, seq)
    return past, attention_mask, position_ids


def _padded_mask(mask, attention_mask):
    """Block the pad keys of a 2D padding mask in an additive causal mask."""
    if attention_mask is None:
        return mask
    # causal_mask returns an expanded view, so clone before the fill.
    pad_keys = attention_mask[:, None, None, :] == 0
    return mask.clone().masked_fill(pad_keys, torch.finfo(mask.dtype).min)


class AttnHeadsPart(nn.Module):
    """post_norm(attn(input_norm(x))) for a subset of donor query heads.

    x has shape (batch, seq, d_donor). attention_mask is the host's 2D
    padding mask, with 1 for real tokens and 0 for pads, or None. With it,
    positions count only real tokens and no token attends to a pad.

    With cache_len None there is no KV cache: x holds the whole sequence
    and the mask is (batch, seq). With cache_len set, the part keeps its
    own KV cache in self.cache, in step with the host cache. cache_len is
    the host cache length before the call: 0 starts a new cache, and a
    larger value is a decode step where x holds only the new tokens and
    the mask covers the cached and the new tokens.

    The rotary module is kept out of the module tree, so a dtype change on
    the part does not round its frequencies. layer_type picks the rotary
    frequencies when the donor keeps one set per layer type (Gemma 3).
    """

    def __init__(self, input_norm, attn, rotary, post_norm=None, layer_type=None):
        super().__init__()
        self.input_norm = input_norm
        self.attn = attn
        self.post_norm = post_norm
        self._rotary = [rotary]
        self.layer_type = layer_type
        self.cache = None
        self.cache_tokens = 0

    def train(self, mode=True):
        # The donor attention is frozen, so it stays in eval mode. The heal
        # pass calls model.train(), which would otherwise turn on donor
        # attention dropout.
        return super().train(False)

    def new_cache(self):
        import transformers

        return transformers.DynamicCache()

    def forward(self, x, attention_mask=None, cache_len=None):
        batch, seq, _ = x.shape
        rotary = self._rotary[0]
        _rotary_to(rotary, x.device)
        past, attention_mask, position_ids = _start_step(self, x, attention_mask, cache_len, "an attn_heads")
        normed = self.input_norm(x)
        if _takes_layer_type(rotary):
            cos_sin = rotary(normed, position_ids, self.layer_type)
        else:
            cos_sin = rotary(normed, position_ids)
        # Mistral, Mixtral and Phi-3 attention keep the window only in the
        # config. Qwen and Gemma set the attribute, and None on full layers.
        if hasattr(self.attn, "sliding_window"):
            window = self.attn.sliding_window
        else:
            window = getattr(self.attn.config, "sliding_window", None)
        mask = causal_mask(batch, seq, normed.dtype, x.device, window, past)
        mask = _padded_mask(mask, attention_mask)
        kwargs = {"hidden_states": normed, "position_embeddings": cos_sin, "attention_mask": mask}
        if self.cache is not None:
            # The cache has only full layers, so it returns every past key
            # and the mask above applies the window.
            kwargs["past_key_values"] = self.cache
        out = self.attn(**kwargs)[0]
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
    _require_attention(found, layer)
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
            rows = q_rows
            if tensor.shape[0] == 2 * num_heads * head_dim:
                # Qwen3.5 and Qwen3-Next keep a query and a gate per head:
                # 2 * head_dim rows for each head.
                rows = _head_rows(heads.tolist(), 2 * head_dim)
            tensor = tensor.index_select(0, rows.to(tensor.device))
        elif owner in ("k_proj", "v_proj"):
            tensor = tensor.index_select(0, kv_rows.to(tensor.device))
        elif owner == "qkv_proj":
            # Phi-3 and Phi-4 fuse the rows as [all q | all k | all v].
            q_size = num_heads * head_dim
            kv_size = _kv_head_count(text_config) * head_dim
            rows = torch.cat([q_rows, kv_rows + q_size, kv_rows + q_size + kv_size])
            tensor = tensor.index_select(0, rows.to(tensor.device))
        elif key in ("o_proj.weight", "out_proj.weight"):
            # LFM2 calls its output projection out_proj.
            tensor = tensor.index_select(1, q_rows.to(tensor.device))
        elif owner == "g_proj":
            # LFM3 gates each query head with one row of g_proj.
            tensor = tensor.index_select(0, heads.to(tensor.device))
        state[key] = tensor.to(dtype).contiguous()
    attn.load_state_dict(state, strict=True)

    input_norm = _load_norm(found["input_norm"], found["input_norm_prefix"], tensors, dtype)
    post_norm = None
    if found["post_attn_norm"] is not None:
        post_norm = _load_norm(found["post_attn_norm"], found["post_attn_norm_prefix"], tensors, dtype)

    rotary_config = copy.deepcopy(text_config)
    rotary_config.head_dim = head_dim
    rotary = type(found["rotary"])(config=rotary_config)

    part = AttnHeadsPart(input_norm, attn, rotary, post_norm, _layer_type(text_config, layer))
    part.eval()
    for param in part.parameters():
        param.requires_grad_(False)
    return part


def _require_plain_block(text_config, first, last):
    """Raise NotImplementedError for decoder layers that need inputs a block part cannot give.

    Gemma 3n and Gemma 4 E2B and E4B feed each layer its own per-layer
    input, and their last layers reuse the KV states of earlier layers.
    """
    if getattr(text_config, "hidden_size_per_layer_input", 0):
        raise NotImplementedError("the donor layers take per-layer inputs (Gemma 3n, Gemma 4 E2B/E4B); "
                                  "block parts do not support them yet")
    modes = getattr(text_config, "kv_sharing_modes", None)
    if modes is None:
        return
    for index in range(first, last + 1):
        if index < len(modes) and modes[index] is not None:
            raise NotImplementedError(f"donor layer {index} shares KV states with other layers; "
                                      "block parts do not support it yet")


def _find_block(config, first, last):
    """Find donor decoder layers first..last (inclusive) on the meta device.

    Returns a dict with layers (the meta layer modules, in order), prefixes
    (their module names), rotary (the donor's rotary module), to_module and
    to_checkpoint. Raises ValueError for a bad range.
    """
    model = _build_meta_model(config)
    to_module, to_checkpoint = _name_rules(model)
    if first < 0 or last < first:
        raise ValueError(f"bad block range [{first}, {last}]")
    list_name, layers = find_decoder_layers(model, last)
    _require_plain_block(_text_config(config), first, last)

    result = {}
    result["to_module"] = to_module
    result["to_checkpoint"] = to_checkpoint
    result["converters"] = _expert_converters(model)
    result["layers"] = []
    result["prefixes"] = []
    for index in range(first, last + 1):
        result["layers"].append(layers[index])
        result["prefixes"].append(f"{list_name}.{index}")
    # _add_attention_modules finds the rotary module. We only keep that key.
    scratch = {}
    _add_attention_modules(scratch, model, layers[first], result["prefixes"][0])
    result["rotary"] = scratch["rotary"]
    return result


def block_layer_prefixes(config, first, last):
    """Return the module names of donor decoder layers first..last."""
    return _find_block(config, first, last)["prefixes"]


def _layer_fused_experts(layer_module, layer_prefix, converters):
    """Return the fused experts of one decoder layer, or None when it has none.

    The result has the keys that _expert_sources and _fused_expert_tensor
    read: moe_prefix, experts, num_experts and converters.
    """
    for name, child in layer_module.named_children():
        experts = getattr(child, "experts", None)
        if experts is None or isinstance(experts, nn.ModuleList):
            continue
        param_names = _fused_param_names(experts)
        count = getattr(experts, "num_experts", None)
        if count is None:
            count = getattr(experts, param_names[0]).shape[0]
        found = {}
        found["moe_prefix"] = f"{layer_prefix}.{name}"
        found["experts"] = experts
        found["num_experts"] = count
        found["converters"] = converters
        return found
    return None


def block_tensor_names(config, first, last):
    """List the checkpoint tensor names of donor decoder layers first..last.

    A Transformers 5 MoE layer holds its experts fused, but most
    checkpoints store each expert's tensors on their own. For those we list
    the per-expert names, and build_block_part fuses them.
    """
    found = _find_block(config, first, last)
    names = []
    for layer_module, prefix in zip(found["layers"], found["prefixes"]):
        keys = _prefixed_keys(layer_module, prefix)
        moe = _layer_fused_experts(layer_module, prefix, found["converters"])
        if moe is None:
            names += keys
            continue
        fused = {}
        for param_name in _fused_param_names(moe["experts"]):
            converter, _ = _expert_sources(moe, param_name, 0)
            if converter is not None:
                fused[f"{moe['moe_prefix']}.experts.{param_name}"] = param_name
        for key in keys:
            if key not in fused:
                names.append(key)
                continue
            for e in range(moe["num_experts"]):
                _, sources = _expert_sources(moe, fused[key], e)
                for name in sources:
                    if name not in names:
                        names.append(name)
    return _checkpoint_names(found, names)


def _layer_window(layer, text_config):
    """Return the sliding window of a donor decoder layer, or None."""
    attn = getattr(layer, "self_attn", None)
    if attn is None:
        return None
    # Mistral keeps the window only in the config. Qwen and Gemma set the
    # attribute on the attention, and None on full layers.
    if hasattr(attn, "sliding_window"):
        return attn.sliding_window
    return getattr(text_config, "sliding_window", None)


class BlockPart(nn.Module):
    """block(x) - x, where block runs whole donor decoder layers in order.

    A block returns a full residual stream, so the part returns the change
    the block makes. A block that does nothing adds nothing. x has shape
    (batch, seq, d_donor). attention_mask is the host's 2D padding mask,
    with 1 for real tokens and 0 for pads, or None. cache_len works as in
    AttnHeadsPart: None runs without a cache over the whole sequence, and a
    number keeps self.cache in step with the host cache. The rotary module
    is kept out of the module tree, so a dtype change on the part does not
    round its frequencies.

    A layer with no self_attn (an LFM2 conv layer) is recurrent: it takes
    the 2D padding mask of the new tokens instead of a causal mask, and its
    cache layer holds a conv state instead of keys and values. first is the
    donor index of the first layer, which is the cache index it uses.
    """

    def __init__(self, layers, rotary, windows=None, layer_types=None, first=0):
        super().__init__()
        self.layers = nn.ModuleList(layers)
        self._rotary = [rotary]
        if windows is None:
            windows = []
            for _ in layers:
                windows.append(None)
        self.windows = list(windows)
        if layer_types is None:
            layer_types = []
            for _ in layers:
                layer_types.append(None)
        self.layer_types = list(layer_types)
        self.recurrent = []
        for layer in layers:
            self.recurrent.append(getattr(layer, "self_attn", None) is None)
        self.first = first
        self.cache = None
        self.cache_tokens = 0

    def train(self, mode=True):
        # The donor layers are frozen, so they stay in eval mode. The heal
        # pass calls model.train(), which would otherwise turn on donor
        # dropout (Phi-2 has resid_pdrop 0.1, for example).
        return super().train(False)

    def new_cache(self):
        """Return an empty cache with one layer per donor index up to the last block layer."""
        import transformers
        try:
            from transformers.cache_utils import DynamicLayer, LinearAttentionLayer
        except ImportError:
            return transformers.DynamicCache()

        if not any(self.recurrent):
            return transformers.DynamicCache()
        layers = []
        for index in range(self.first + len(self.layers)):
            offset = index - self.first
            if offset >= 0 and self.recurrent[offset]:
                layers.append(LinearAttentionLayer())
            else:
                layers.append(DynamicLayer())
        return transformers.Cache(layers=layers)

    def forward(self, x, attention_mask=None, cache_len=None):
        batch, seq, _ = x.shape
        rotary = self._rotary[0]
        _rotary_to(rotary, x.device)
        past, attention_mask, position_ids = _start_step(self, x, attention_mask, cache_len, "a block")

        # Gemma 3 and LFM3 keep one set of rotary frequencies per layer
        # type. Other donors share one set for every layer.
        per_type = _takes_layer_type(rotary)
        shared_cos_sin = None
        if not per_type:
            shared_cos_sin = rotary(x, position_ids)

        # One mask per window size, shared by the layers that use it. The
        # cache has only full layers, so it returns every past key and the
        # mask applies the window.
        masks = {}
        embeddings = {}
        hidden = x
        for index in range(len(self.layers)):
            layer = self.layers[index]
            if self.recurrent[index]:
                # Like Transformers: the padding mask of the new tokens, or
                # None for a one-token decode step.
                mask = None
                if attention_mask is not None and seq > 1:
                    mask = attention_mask[:, past:]
                cos_sin = shared_cos_sin
            else:
                window = self.windows[index]
                if window not in masks:
                    mask = causal_mask(batch, seq, x.dtype, x.device, window, past)
                    masks[window] = _padded_mask(mask, attention_mask)
                mask = masks[window]
                cos_sin = shared_cos_sin
                if per_type:
                    layer_type = self.layer_types[index]
                    if layer_type not in embeddings:
                        embeddings[layer_type] = rotary(x, position_ids, layer_type)
                    cos_sin = embeddings[layer_type]

            out = layer(hidden, attention_mask=mask, position_ids=position_ids,
                        past_key_values=self.cache, position_embeddings=cos_sin)
            if isinstance(out, tuple):
                out = out[0]
            hidden = out
        return hidden - x


def build_block_part(config, first, last, tensors):
    """Build a part that runs donor decoder layers first..last in order.

    Each layer is the donor's own decoder layer class, built on a copy of
    the text config with eager attention and loaded strictly from tensors
    (checkpoint names or module names). The result is in eval mode with
    frozen parameters.
    """
    found = _find_block(config, first, last)
    # A copy, so the fused expert tensors we add stay out of the caller's dict.
    tensors = dict(_module_named(tensors, found))
    text_config = _text_config(config)
    block_config = copy.deepcopy(text_config)
    block_config._attn_implementation = "eager"

    # Fuse per-expert checkpoint tensors into the 3D tensors the layers hold.
    for meta_layer, prefix in zip(found["layers"], found["prefixes"]):
        moe = _layer_fused_experts(meta_layer, prefix, found["converters"])
        if moe is None:
            continue
        ids = list(range(moe["num_experts"]))
        for param_name in _fused_param_names(moe["experts"]):
            full = f"{moe['moe_prefix']}.experts.{param_name}"
            if full not in tensors:
                tensors[full] = _fused_expert_tensor(moe, param_name, ids, tensors, text_config)

    first_key = _prefixed_keys(found["layers"][0], found["prefixes"][0])[0]
    dtype = tensors[first_key].dtype

    layers = []
    windows = []
    layer_types = []
    for offset in range(len(found["layers"])):
        meta_layer = found["layers"][offset]
        prefix = found["prefixes"][offset]
        layer = type(meta_layer)(block_config, first + offset)
        layer = layer.to(dtype)
        state = {}
        for key in meta_layer.state_dict().keys():
            state[key] = tensors[f"{prefix}.{key}"].to(dtype).contiguous()
        layer.load_state_dict(state, strict=True)
        layers.append(layer)
        windows.append(_layer_window(layer, block_config))
        layer_types.append(_layer_type(block_config, first + offset))

    rotary = type(found["rotary"])(config=copy.deepcopy(text_config))
    part = BlockPart(layers, rotary, windows, layer_types, first)
    part.eval()
    for param in part.parameters():
        param.requires_grad_(False)
    return part


def _use_transformers_4_module():
    """On Transformers 4, serve the 4.x code in wsp/parts_tf4.py under this name."""
    import sys

    if not _transformers_5():
        from wsp import parts_tf4

        sys.modules[__name__] = parts_tf4


_use_transformers_4_module()
