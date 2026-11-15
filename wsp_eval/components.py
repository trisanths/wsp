"""Load components and build models from them, using the team's `wsp` package.

Masks: we read them with wsp.masks.load_mask_dir (the Deletion Tuning
packbits_zstd directories, local or hf://). For overlap math we turn each
boolean mask into sorted flat indices: {param_name: 1-D int64 tensor}. That
is small (masks delete well under 0.1% of weights) and fast to intersect.
A single .safetensors / .pt file of bool tensors or 1-D flat indices is also
accepted, for hand-made test masks.

Models: `build_model` writes a model with Trisanth's compile(), so a probe's
ablated model is exactly what a program like
    use(id).restrict(layers=keep)
compiles to. Same program -> same directory, so repeated builds are free.
"""
import hashlib
import json
import os
import re

from wsp_eval import catalog, config, models

LAYER_PATTERN = re.compile(r"layers\.(\d+)\.")


def layer_of(param_name):
    match = LAYER_PATTERN.search(param_name)
    if match is None:
        return None
    return int(match.group(1))


def module_of(param_name):
    """'model.layers.14.mlp.gate_proj.weight' -> 'mlp.gate_proj'."""
    name = LAYER_PATTERN.split(param_name)[-1]
    if name.endswith(".weight"):
        name = name[:-len(".weight")]
    return name


def _to_indices(name, tensor, source):
    import torch
    if tensor.dtype == torch.bool:
        return torch.nonzero(tensor.reshape(-1), as_tuple=False).reshape(-1)
    if tensor.dtype in (torch.int64, torch.int32, torch.int16) and tensor.dim() == 1:
        return torch.unique(tensor.to(torch.int64))
    raise ValueError("mask tensor " + name + " in " + source + " must be bool (True = deleted) or 1-D int "
                     "(flat indices); got " + str(tensor.dtype) + " " + str(tuple(tensor.shape)))


def load_mask(path):
    """Return {param_name: sorted 1-D int64 tensor of deleted flat indices}."""
    if os.path.isdir(path):
        from wsp import masks as wsp_masks
        raw = wsp_masks.load_mask_dir(path)
    elif path.endswith(".safetensors"):
        from safetensors.torch import load_file
        raw = load_file(path)
    else:
        import torch
        raw = torch.load(path, map_location="cpu")
    mask = {}
    for name, tensor in raw.items():
        indices = _to_indices(name, tensor, path)
        if indices.numel() > 0:
            mask[name] = indices
    return mask


def mask_layers(mask):
    layers = set()
    for name in mask:
        layer = layer_of(name)
        if layer is not None:
            layers.add(layer)
    return sorted(layers)


def deleted_per_layer(mask, num_layers=None):
    if num_layers is None:
        layers = mask_layers(mask)
        if len(layers) == 0:
            num_layers = 0
        else:
            num_layers = layers[-1] + 1
    counts = [0] * num_layers
    for name, indices in mask.items():
        layer = layer_of(name)
        if layer is not None and layer < num_layers:
            counts[layer] += int(indices.numel())
    return counts


def base_config(base):
    model_dir = models.resolve_model_path(base)
    with open(os.path.join(model_dir, "config.json")) as f:
        return json.load(f)


def num_layers_of_base(base):
    """Number of transformer layers in a base model (None if the config cannot be read)."""
    try:
        return base_config(base)["num_hidden_layers"]
    except Exception:
        return None


_mask_cache = {}


def load_component_mask(record, catalog_path=None):
    """The mask of a deletion_mask record, as flat indices (cached per process)."""
    if record["type"] != "deletion_mask":
        raise ValueError(record["id"] + " is a " + record["type"] + ", not a deletion_mask")
    path = catalog.local_path(record, catalog_path)
    if path not in _mask_cache:
        _mask_cache[path] = load_mask(path)
    return _mask_cache[path]


def component_layers(record, catalog_path=None):
    """Layers a component touches."""
    if record.get("layers"):
        return sorted(record["layers"])
    if record["type"] == "deletion_mask":
        return mask_layers(load_component_mask(record, catalog_path))
    if record["type"] == "task_vector":
        num_layers = num_layers_of_base(record["base"])
        if num_layers is None:
            raise ValueError("cannot read the layer count of " + record["base"])
        return list(range(num_layers))
    raise NotImplementedError("component type " + record["type"] + " is not supported by the probe tool yet")


# ---------------------------------------------------------------------------
# Building models with wsp's compile()
# ---------------------------------------------------------------------------

def program_for(record, layers=None):
    """A wsp Program that applies one component, optionally only at `layers`."""
    from wsp import Program, use
    if record["type"] == "deletion_mask":
        expression = use(record["id"])
        if layers is not None:
            expression = expression.restrict(layers=layers)
        return Program(base=record["base"], masks=expression)
    if record["type"] == "task_vector":
        program = Program(base=record["base"])
        program.tv(record["id"], weight=1.0, layers=layers)
        return program
    raise NotImplementedError("cannot build a program for a " + record["type"])


def build_model(program, catalog_path=None, out_root=None):
    """Compile a wsp Program into work/models/<hash>/ and return the path.

    A program with no masks and no task vectors is just the base model, so we
    return the base itself instead of re-saving it.
    """
    from wsp import Catalog

    if program.masks is None and len(program.task_vectors) == 0:
        return models.resolve_model_path(program.base)

    if catalog_path is None:
        catalog_path = config.catalog_path()
    if out_root is None:
        out_root = config.work_path("models")
    os.makedirs(out_root, exist_ok=True)

    # The hash covers the program and the catalog records it uses.
    wsp_catalog = Catalog.load(catalog_path)
    used = []
    for component_id in sorted(set(program.component_ids())):
        used.append(wsp_catalog.get(component_id))
    text = json.dumps({"program": program.to_dict(), "records": used}, sort_keys=True)
    out_dir = os.path.join(out_root, hashlib.sha256(text.encode("utf-8")).hexdigest()[:20])

    if os.path.exists(os.path.join(out_dir, "compile_report.json")):
        return out_dir
    program.compile(out_dir, wsp_catalog)
    return out_dir
