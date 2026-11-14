"""overlap(a, b, by='layer') -> stats on shared deletions between two masks.

`a` and `b` are catalog ids, catalog records, or in-memory masks.
`by` is 'layer', 'module' (e.g. mlp.gate_proj) or 'param' (full name).
No model is run, so this costs no eval budget.
"""
import torch

from wsp_eval import catalog, components


def _as_mask(x, catalog_path=None):
    if isinstance(x, str):
        return components.load_component_mask(catalog.get(x, catalog_path), catalog_path)
    if isinstance(x, dict) and "type" in x and "path" in x:
        return components.load_component_mask(x, catalog_path)
    return x


def _group_key(name, by):
    if by == "layer":
        return components.layer_of(name)
    if by == "module":
        return components.module_of(name)
    if by == "param":
        return name
    raise ValueError("by must be 'layer', 'module' or 'param'")


def _stats(n_a, n_b, shared):
    union = n_a + n_b - shared
    if union == 0:
        jaccard = None
    else:
        jaccard = shared / union
    return {"a": n_a, "b": n_b, "shared": shared, "union": union, "jaccard": jaccard}


def overlap(a, b, by="layer", catalog_path=None):
    mask_a = _as_mask(a, catalog_path)
    mask_b = _as_mask(b, catalog_path)

    names = set(mask_a.keys())
    names.update(mask_b.keys())

    groups = {}
    total_a = 0
    total_b = 0
    total_shared = 0
    for name in sorted(names):
        empty = torch.zeros(0, dtype=torch.int64)
        indices_a = mask_a.get(name, empty)
        indices_b = mask_b.get(name, empty)
        n_a = int(indices_a.numel())
        n_b = int(indices_b.numel())
        shared = int(torch.isin(indices_a, indices_b).sum())

        key = _group_key(name, by)
        if key not in groups:
            groups[key] = [0, 0, 0]
        groups[key][0] += n_a
        groups[key][1] += n_b
        groups[key][2] += shared
        total_a += n_a
        total_b += n_b
        total_shared += shared

    group_stats = {}
    for key in sorted(groups.keys(), key=lambda k: (k is None, str(k).zfill(6))):
        n_a, n_b, shared = groups[key]
        group_stats[key] = _stats(n_a, n_b, shared)

    return {"by": by, "groups": group_stats, "total": _stats(total_a, total_b, total_shared)}
