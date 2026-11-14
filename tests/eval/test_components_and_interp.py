import json
import os

import pytest
import torch
from safetensors.torch import load_file

from wsp_eval import catalog, components, heatmaps, interp_agent, overlap_tool as overlap, probe_tool as probe_module


def _weights(model_dir):
    tensors = {}
    for name in os.listdir(model_dir):
        if name.endswith(".safetensors"):
            tensors.update(load_file(os.path.join(model_dir, name)))
    return tensors


def test_load_masks(tiny_model):
    mask_a = components.load_component_mask(catalog.get("mask_a"))
    mask_b = components.load_component_mask(catalog.get("mask_b"))
    assert components.deleted_per_layer(mask_a) == [10, 11, 12, 13]
    assert components.deleted_per_layer(mask_b) == [15, 15, 15, 15]
    assert components.mask_layers(mask_a) == [0, 1, 2, 3]
    assert components.module_of("model.layers.3.mlp.gate_proj.weight") == "mlp.gate_proj"
    assert components.component_layers(catalog.get("tv_c")) == [0, 1, 2, 3]


def test_overlap(tiny_model):
    stats = overlap.overlap("mask_a", "mask_b", by="layer")
    # layer 0: a = 0..9, b = 5..19 -> shared 5, union 20
    assert stats["groups"][0]["shared"] == 5
    assert stats["groups"][0]["union"] == 20
    assert stats["groups"][0]["jaccard"] == pytest.approx(0.25)
    assert stats["groups"][3]["shared"] == 8
    assert stats["total"]["shared"] == 5 + 6 + 7 + 8
    by_module = overlap.overlap("mask_a", "mask_b", by="module")
    assert list(by_module["groups"].keys()) == ["mlp.gate_proj"]


def test_build_model_uses_wsp_compile(tiny_model):
    from wsp import Program
    base = _weights(tiny_model)
    record_a = catalog.get("mask_a")

    only_layer_1 = components.build_model(components.program_for(record_a, [1]))
    edited = _weights(only_layer_1)
    flat = edited["model.layers.1.mlp.gate_proj.weight"].reshape(-1)
    assert torch.all(flat[:11] == 0)
    assert torch.equal(flat[11:], base["model.layers.1.mlp.gate_proj.weight"].reshape(-1)[11:])
    assert torch.equal(edited["model.layers.0.mlp.gate_proj.weight"], base["model.layers.0.mlp.gate_proj.weight"])
    assert os.path.exists(os.path.join(only_layer_1, "program.json"))

    # Same program -> same directory, no rebuild.
    assert components.build_model(components.program_for(record_a, [1])) == only_layer_1

    tv = components.build_model(components.program_for(catalog.get("tv_c"), [2]))
    delta = _weights(tv)["model.layers.2.mlp.down_proj.weight"].float() - base["model.layers.2.mlp.down_proj.weight"].float()
    assert torch.allclose(delta, torch.full_like(delta, 0.25), atol=0.02)
    untouched = _weights(tv)["model.layers.1.mlp.down_proj.weight"]
    assert torch.equal(untouched, base["model.layers.1.mlp.down_proj.weight"])

    # An empty program is the base model itself.
    assert components.build_model(Program(base=str(tiny_model))) == str(tiny_model)


def test_probe(tiny_model):
    result = probe_module.probe("mask_a", [2, 3], tasks=["toy", "toy2"])
    assert result["layers"] == [2, 3]
    assert result["label"].startswith("SEARCH SPLIT")
    for task in ("toy", "toy2"):
        assert result["drop"][task] == pytest.approx(result["full"][task] - result["ablated"][task])
    # Probing all layers compares against the base model.
    everything = probe_module.probe("mask_a", [0, 1, 2, 3], tasks=["toy"])
    from wsp_eval import harness
    base_score = harness.eval(str(tiny_model), "search", ["toy"])["tasks"]["toy"]["score"]
    assert everything["ablated"]["toy"] == base_score


def test_interp_agent_end_to_end(tiny_model):
    agent = interp_agent.InterpAgent(max_evals=40, band_size=2, noise_multiplier=0.0)
    agent.run()
    records = {}
    for record in catalog.load():
        records[record["id"]] = record
    signature = records["mask_a"]["signature"]
    assert signature["deleted_per_layer"] == [10, 11, 12, 13]
    assert "mask_b" in signature["overlap_with"]
    assert signature["overlap_with"]["mask_b"]["shared"] == 26
    assert set(signature["effect"].keys()) == {"toy", "toy2"}
    assert signature["label"].startswith("SEARCH SPLIT")
    assert "effect" in records["tv_c"]["signature"]
    assert signature["num_deleted"] == 46
    # Person A's fields are untouched.
    assert records["mask_a"]["path"] == "components/a"
    log = [json.loads(line) for line in open(interp_agent.config.interp_log_path())]
    for entry in log:
        assert "hypothesis" in entry and "action" in entry and "result" in entry
    assert agent.evals_used <= 40


def test_interp_agent_localizes_with_synthetic_probe(tiny_model, monkeypatch):
    """Planner logic: a component whose effect lives in layer 3 should be localized there."""
    def fake_probe(record, layers, tasks=None, split="search", namespace="interp", full_scores=None,
                   catalog_path=None):
        drop = {}
        for task in tasks:
            if 3 in layers:
                drop[task] = 0.30
            else:
                drop[task] = 0.0
        full = {}
        ablated = {}
        for task in tasks:
            full[task] = 0.6
            ablated[task] = 0.6 - drop[task]
        return {"component": record["id"], "layers": sorted(layers), "split": split,
                "label": "SEARCH SPLIT", "full": full, "ablated": ablated, "drop": drop, "n": 150}

    monkeypatch.setattr(probe_module, "probe", fake_probe)
    agent = interp_agent.InterpAgent(max_evals=100, band_size=2, ids=["mask_a"])
    agent.run()
    signature = catalog.get("mask_a")["signature"]
    assert "3" in signature["layer_effects"]          # refined down to one layer
    maps = signature["capability_map"]
    assert maps[0]["layers"] == [2, 3]
    assert "lives mostly in layers 2 to 3" in maps[0]["text"]


def test_capability_map():
    entry = interp_agent.capability_map("t", [([0, 1], 0.0), ([2, 3], 0.1), ([4, 5], 0.3), ([6, 7], 0.05)])
    assert entry["layers"] == [4, 5]
    assert interp_agent.capability_map("t", [([0, 1], -0.1)]) is None


def test_heatmaps(tiny_model, tmp_path):
    assert os.path.exists(heatmaps.deletion_heatmap(str(tmp_path / "del.png")))
    assert os.path.exists(heatmaps.overlap_heatmap(str(tmp_path / "ovl.png")))
    assert os.path.exists(heatmaps.neuron_heatmap("mask_a", str(tmp_path / "neu.png"), num_bins=8))
