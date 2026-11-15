"""Index files for donor parts, random controls, and ablation hooks.

Index file (agreed format, see docs/superpowers/specs/...mixed-family-compiler-design.md):
    a .pt file holding a dict
        kind      "mlp_cluster" | "attn_heads" | "expert"
        model     donor model id, e.g. "Qwen/Qwen2.5-Math-1.5B"
        layer     donor layer index
        indices   LongTensor of units: MLP neurons, query heads, or expert ids
    plus our extra keys (the compiler ignores them):
        signature   how it was found and what it does (see extraction.py)
        control_of  for a random control: the path of the file it controls for
        seed        for a random control

Ablation spec (for probing a donor): {"kind", "layer", "indices"}. The hooks
zero the units' contribution:
    mlp_cluster  zero those neurons' activations (the input columns of down_proj)
    attn_heads   zero those query heads' outputs (their slice of the o_proj input)
    expert       make the router never choose those experts
"""
import hashlib
import json
import random

KINDS = ("mlp_cluster", "attn_heads", "expert")


def save_index_file(path, kind, model, layer, indices, signature=None, extra=None):
    import torch
    if kind not in KINDS:
        raise ValueError("kind must be one of " + str(KINDS))
    data = {"kind": kind, "model": model, "layer": int(layer),
            "indices": torch.tensor(sorted(int(i) for i in indices), dtype=torch.long)}
    if signature is not None:
        data["signature"] = signature
    if extra is not None:
        for key, value in extra.items():
            data[key] = value
    torch.save(data, path)
    return path


def load_index_file(path):
    import torch
    data = torch.load(path, map_location="cpu", weights_only=False)
    for key in ("kind", "model", "layer", "indices"):
        if key not in data:
            raise ValueError(path + " is missing '" + key + "'")
    return data


def unit_count(model_config, kind):
    """How many units of this kind one layer of the donor has."""
    if kind == "mlp_cluster":
        if model_config.get("moe_intermediate_size") and not model_config.get("intermediate_size"):
            return model_config["moe_intermediate_size"]
        return model_config["intermediate_size"]
    if kind == "attn_heads":
        return model_config["num_attention_heads"]
    if kind == "expert":
        for key in ("num_experts", "num_local_experts", "n_routed_experts"):
            if model_config.get(key):
                return model_config[key]
        raise ValueError("model config has no expert count")
    raise ValueError("unknown kind " + kind)


def random_control(index_path, out_path, model_config, seed):
    """Same kind, model, layer and size; random units from that layer, excluding the chosen ones."""
    data = load_index_file(index_path)
    chosen = set()
    for value in data["indices"].tolist():
        chosen.add(value)
    pool = []
    for unit in range(unit_count(model_config, data["kind"])):
        if unit not in chosen:
            pool.append(unit)
    if len(pool) < len(chosen):
        raise ValueError("layer has too few other units for a same-size random control")
    rng = random.Random(seed)
    picked = rng.sample(pool, len(chosen))
    return save_index_file(out_path, data["kind"], data["model"], data["layer"], picked,
                           signature={"control": "random same-size set from the same layer", "seed": seed},
                           extra={"control_of": index_path, "seed": seed})


def ablation_from_file(path):
    data = load_index_file(path)
    return {"kind": data["kind"], "layer": int(data["layer"]), "indices": data["indices"].tolist()}


def ablation_key(ablation):
    if ablation is None:
        return ""
    text = json.dumps({"kind": ablation["kind"], "layer": int(ablation["layer"]),
                       "indices": sorted(int(i) for i in ablation["indices"])}, sort_keys=True)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Finding modules inside a Hugging Face decoder (Llama, Qwen2, Qwen3, Gemma, Mistral, MoE variants)
# ---------------------------------------------------------------------------

def decoder_layers(model):
    for path in ("model.layers", "model.model.layers", "model.language_model.layers", "transformer.h"):
        node = model
        found = True
        for part in path.split("."):
            if not hasattr(node, part):
                found = False
                break
            node = getattr(node, part)
        if found:
            return node
    raise ValueError("cannot find the decoder layers of " + type(model).__name__)


def mlp_down_proj(layer):
    mlp = layer.mlp
    if hasattr(mlp, "down_proj"):
        return mlp.down_proj
    raise ValueError("layer MLP has no down_proj (MoE layers use kind 'expert')")


def attn_o_proj(layer):
    attention = layer.self_attn
    if hasattr(attention, "o_proj"):
        return attention.o_proj
    raise ValueError("attention has no o_proj")


def moe_router(layer):
    mlp = layer.mlp
    for name in ("gate", "router"):
        if hasattr(mlp, name) and hasattr(mlp, "experts"):
            return getattr(mlp, name)
    raise ValueError("layer is not an MoE layer with a router")


def head_dim(model):
    config = model.config
    if getattr(config, "head_dim", None):
        return config.head_dim
    return config.hidden_size // config.num_attention_heads


class RouterBlock:
    """Handle for block_router_experts. remove() puts the router's earlier forward back."""

    def __init__(self, router, earlier):
        self.router = router
        # The forward set on this router before the block, or None for the class forward.
        self.earlier = earlier

    def remove(self):
        if self.earlier is not None:
            self.router.forward = self.earlier
        elif "forward" in self.router.__dict__:
            del self.router.forward


def block_router_experts(router, indices):
    """Make the router never choose the experts in indices. Returns a handle with remove().

    In Transformers 4 the router (gate) is an nn.Linear that returns logits,
    and the MoE block picks the top experts. In Transformers 5 the router
    module picks them itself and returns (logits, weights, indices). A hook
    on the router output is then too late. Both call
    torch.nn.functional.linear for the logits, so while the router runs we
    set the blocked experts' logits to -inf at that call. The router then
    picks and weights the other experts the same way it always does. A
    router that adds a score bias after a sigmoid (DeepSeek V3 style) can
    still pick a blocked expert whose bias is large.
    """
    import torch
    functional = torch.nn.functional
    earlier = router.__dict__.get("forward")
    original_forward = router.forward

    def blocked_forward(*args, **kwargs):
        original_linear = functional.linear

        def masked_linear(*linear_args, **linear_kwargs):
            logits = original_linear(*linear_args, **linear_kwargs).clone()
            logits[..., indices.to(logits.device)] = float("-inf")
            return logits

        functional.linear = masked_linear
        try:
            return original_forward(*args, **kwargs)
        finally:
            functional.linear = original_linear

    router.forward = blocked_forward
    return RouterBlock(router, earlier)


def add_ablation_hooks(model, ablation):
    """Register hooks that knock out the units. Returns the hook handles (call .remove())."""
    import torch
    layer = decoder_layers(model)[ablation["layer"]]
    indices = torch.tensor(sorted(int(i) for i in ablation["indices"]), dtype=torch.long)
    handles = []

    if ablation["kind"] == "mlp_cluster":
        def zero_neurons(module, args):
            hidden = args[0].clone()
            hidden[..., indices.to(hidden.device)] = 0
            return (hidden,) + tuple(args[1:])
        handles.append(mlp_down_proj(layer).register_forward_pre_hook(zero_neurons))

    elif ablation["kind"] == "attn_heads":
        size = head_dim(model)
        columns = []
        for head in indices.tolist():
            for offset in range(size):
                columns.append(head * size + offset)
        column_tensor = torch.tensor(columns, dtype=torch.long)

        def zero_heads(module, args):
            hidden = args[0].clone()
            hidden[..., column_tensor.to(hidden.device)] = 0
            return (hidden,) + tuple(args[1:])
        handles.append(attn_o_proj(layer).register_forward_pre_hook(zero_heads))

    elif ablation["kind"] == "expert":
        handles.append(block_router_experts(moe_router(layer), indices))

    else:
        raise ValueError("unknown ablation kind " + str(ablation["kind"]))
    return handles
