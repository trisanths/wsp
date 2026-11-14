import json
import os

import numpy as np
import pytest
import torch
import zstandard
from safetensors.torch import save_file

from wsp import Catalog, Program, use
from wsp import masks as mask_ops
from wsp.compiler import CompileError, add_task_vector, apply_mask, check_same_base
from wsp.program import expr_from_dict

NAMES = [
    "model.layers.0.mlp.gate_proj.weight",
    "model.layers.1.mlp.gate_proj.weight",
    "model.layers.2.mlp.gate_proj.weight",
]


def random_mask(seed, density=0.3):
    generator = torch.Generator().manual_seed(seed)
    mask = {}
    for name in NAMES:
        mask[name] = torch.rand(4, 6, generator=generator) < density
    return mask


def write_authors_format(keep, mask_dir):
    """Write a mask exactly the way the Deletion Tuning repo stores it."""
    os.makedirs(mask_dir, exist_ok=True)
    compressor = zstandard.ZstdCompressor()
    tensors = {}
    for name in keep:
        packed = compressor.compress(np.packbits(keep[name].numpy().astype(np.uint8).reshape(-1)).tobytes())
        tensors[name + ".shape"] = torch.tensor(list(keep[name].shape))
        tensors[name + ".packed"] = torch.frombuffer(bytearray(packed), dtype=torch.uint8)
    save_file(tensors, os.path.join(mask_dir, "model.safetensors"))
    with open(os.path.join(mask_dir, "config.json"), "w") as f:
        json.dump({"format": "packbits_zstd"}, f)


def test_load_reads_keep_bits_as_not_deleted(tmp_path):
    delete = random_mask(0)
    keep = {}
    for name in delete:
        keep[name] = ~delete[name]
    write_authors_format(keep, tmp_path / "m")
    loaded = mask_ops.load_mask_dir(tmp_path / "m")
    for name in NAMES:
        assert torch.equal(loaded[name], delete[name])


def test_save_then_load_round_trip(tmp_path):
    mask = random_mask(1)
    mask_ops.save_mask_dir(mask, tmp_path / "m")
    loaded = mask_ops.load_mask_dir(tmp_path / "m")
    for name in NAMES:
        assert torch.equal(loaded[name], mask[name])


def test_algebra_counts():
    a = random_mask(2)
    b = random_mask(3)
    both = mask_ops.count_deleted(mask_ops.intersect(a, b))
    either = mask_ops.count_deleted(mask_ops.union(a, b))
    only_a = mask_ops.count_deleted(mask_ops.difference(a, b))
    assert either == mask_ops.count_deleted(a) + mask_ops.count_deleted(b) - both
    assert only_a == mask_ops.count_deleted(a) - both


def test_restrict_by_layer_and_module():
    mask = random_mask(4)
    restricted = mask_ops.restrict(mask, layers=[1, 2])
    assert sorted(restricted) == NAMES[1:]
    assert mask_ops.restrict(mask, modules=["attn"]) == {}


def test_program_expression_and_round_trip():
    masks = {"a": random_mask(5), "b": random_mask(6), "c": random_mask(7)}
    expr = ((use("a") | use("b")) - use("c")).restrict(layers=range(0, 2))
    result = expr.evaluate(lambda component_id: masks[component_id])
    expected = mask_ops.restrict(
        mask_ops.difference(mask_ops.union(masks["a"], masks["b"]), masks["c"]), layers=[0, 1]
    )
    assert sorted(result) == sorted(expected)
    for name in expected:
        assert torch.equal(result[name], expected[name])

    program = Program(base="toy", masks=expr, seed=3).tv("tv_x", weight=0.5, layers=[1], modules=["mlp"])
    again = Program.from_dict(json.loads(json.dumps(program.to_dict())))
    assert again.to_dict() == program.to_dict()
    assert again.component_ids() == ["a", "b", "c", "tv_x"]
    assert repr(expr_from_dict(expr.to_dict())) == repr(expr)


def test_apply_mask_zeros_only_deleted_weights():
    params = {}
    for name in NAMES:
        params[name] = torch.nn.Parameter(torch.ones(4, 6))
    mask = random_mask(8)
    deleted = apply_mask(params, mask)
    assert deleted == mask_ops.count_deleted(mask)
    for name in NAMES:
        assert torch.equal(params[name].data == 0, mask[name])


def test_empty_mask_changes_nothing():
    params = {NAMES[0]: torch.nn.Parameter(torch.randn(4, 6))}
    before = params[NAMES[0]].detach().clone()
    apply_mask(params, {})
    assert torch.equal(params[NAMES[0]].data, before)


def test_add_task_vector_on_selected_layers():
    base = {}
    finetuned = {}
    params = {}
    for name in NAMES:
        base[name] = torch.zeros(2, 2)
        finetuned[name] = torch.full((2, 2), 4.0)
        params[name] = torch.nn.Parameter(torch.zeros(2, 2))
    changed = add_task_vector(params, base, finetuned, weight=0.5, layers=[2])
    assert changed == [NAMES[2]]
    assert torch.equal(params[NAMES[2]].data, torch.full((2, 2), 2.0))
    assert torch.equal(params[NAMES[0]].data, torch.zeros(2, 2))


def test_mixed_base_is_refused(tmp_path):
    catalog = Catalog([
        {"id": "m1", "type": "deletion_mask", "base": "base-a", "path": "x"},
        {"id": "m2", "type": "deletion_mask", "base": "base-b", "path": "y"},
    ])
    program = Program(base="base-a", masks=use("m1") | use("m2"))
    with pytest.raises(CompileError):
        check_same_base(program, catalog)


def test_catalog_resolves_relative_paths(tmp_path):
    path = tmp_path / "catalog.jsonl"
    path.write_text(json.dumps({"id": "m", "type": "deletion_mask", "base": "b", "path": "components/m"}) + "\n")
    catalog = Catalog.load(str(path))
    assert catalog.local_path("m") == str(tmp_path / "components" / "m")
