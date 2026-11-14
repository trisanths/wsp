"""Demo visuals: per-layer deletion heatmaps and overlaps.

  deletion_heatmap   masks x layers, number of deleted gate_proj entries
  neuron_heatmap     one mask, layers x neuron bins (rows of gate_proj)
  overlap_heatmap    mask pairs x layers, Jaccard of the deleted sets
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from wsp_eval import catalog, components, overlap_tool as overlap_module  # noqa: E402


def _mask_records(ids=None, catalog_path=None):
    records = []
    for record in catalog.load(catalog_path):
        if record["type"] != "deletion_mask":
            continue
        if ids is not None and record["id"] not in ids:
            continue
        records.append(record)
    return records


def _num_layers(records, masks):
    num_layers = 0
    for record, mask in zip(records, masks):
        n = components.num_layers_of_base(record["base"])
        if n is None:
            layers = components.mask_layers(mask)
            if len(layers) > 0:
                n = layers[-1] + 1
            else:
                n = 0
        num_layers = max(num_layers, n)
    return num_layers


def deletion_heatmap(out_path, ids=None, catalog_path=None):
    records = _mask_records(ids, catalog_path)
    masks = []
    for record in records:
        masks.append(components.load_component_mask(record, catalog_path))
    num_layers = _num_layers(records, masks)

    grid = np.zeros((len(records), num_layers))
    for row in range(len(records)):
        counts = components.deleted_per_layer(masks[row], num_layers)
        for layer in range(num_layers):
            grid[row, layer] = counts[layer]

    fig, ax = plt.subplots(figsize=(max(6, num_layers * 0.45), 1.2 + 0.45 * len(records)))
    image = ax.imshow(grid, aspect="auto", cmap="viridis")
    ax.set_yticks(range(len(records)))
    labels = []
    for record in records:
        labels.append(record["id"])
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xticks(range(num_layers))
    ax.set_xlabel("layer")
    ax.set_title("Deleted gate_proj entries per layer")
    fig.colorbar(image, ax=ax, label="deleted entries")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def neuron_heatmap(component_id, out_path, num_bins=64, catalog_path=None):
    """Where inside each layer's gate_proj the deletions fall (binned by output neuron)."""
    import json
    import os
    from wsp_eval import models

    record = catalog.get(component_id, catalog_path)
    mask = components.load_component_mask(record, catalog_path)
    model_dir = models.resolve_model_path(record["base"])
    with open(os.path.join(model_dir, "config.json")) as f:
        model_config = json.load(f)
    hidden = model_config["hidden_size"]
    intermediate = model_config["intermediate_size"]
    num_layers = model_config["num_hidden_layers"]

    grid = np.zeros((num_layers, num_bins))
    for name, indices in mask.items():
        if "gate_proj" not in name:
            continue
        layer = components.layer_of(name)
        rows = (indices // hidden).numpy()
        bins = (rows * num_bins) // intermediate
        for b in bins:
            grid[layer, b] += 1

    fig, ax = plt.subplots(figsize=(8, max(3, num_layers * 0.25)))
    image = ax.imshow(grid, aspect="auto", cmap="magma", origin="lower")
    ax.set_xlabel("gate_proj output neuron (" + str(num_bins) + " bins of " + str(intermediate // num_bins) + ")")
    ax.set_ylabel("layer")
    ax.set_title(component_id + ": deletions by neuron")
    fig.colorbar(image, ax=ax, label="deleted entries")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def overlap_heatmap(out_path, ids=None, catalog_path=None):
    records = _mask_records(ids, catalog_path)
    masks = []
    for record in records:
        masks.append(components.load_component_mask(record, catalog_path))
    num_layers = _num_layers(records, masks)

    if len(records) * (len(records) - 1) // 2 > 8:
        return overlap_matrix(out_path, records, masks)

    pair_labels = []
    rows = []
    for i in range(len(records)):
        for j in range(i + 1, len(records)):
            if records[i]["base"] != records[j]["base"]:
                continue
            stats = overlap_module.overlap(masks[i], masks[j], by="layer")
            row = np.full(num_layers, np.nan)
            for layer, values in stats["groups"].items():
                if layer is not None and values["jaccard"] is not None and layer < num_layers:
                    row[layer] = values["jaccard"]
            rows.append(row)
            total = stats["total"]["jaccard"]
            if total is None:
                total_text = "n/a"
            else:
                total_text = "%.3f" % total
            pair_labels.append(records[i]["id"] + " vs " + records[j]["id"] + " (J=" + total_text + ")")

    if len(rows) == 0:
        raise ValueError("need at least two masks on the same base to plot overlaps")

    grid = np.vstack(rows)
    fig, ax = plt.subplots(figsize=(max(6, num_layers * 0.45), 1.2 + 0.45 * len(rows)))
    image = ax.imshow(grid, aspect="auto", cmap="Reds", vmin=0)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(pair_labels, fontsize=8)
    ax.set_xticks(range(num_layers))
    ax.set_xlabel("layer")
    ax.set_title("Jaccard overlap of deleted entries per layer")
    fig.colorbar(image, ax=ax, label="Jaccard")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def overlap_matrix(out_path, records, masks):
    """Mask x mask matrix of total Jaccard overlap (used when there are too many pairs for per-layer rows)."""
    count = len(records)
    grid = np.full((count, count), np.nan)
    for i in range(count):
        for j in range(i + 1, count):
            if records[i]["base"] != records[j]["base"]:
                continue
            total = overlap_module.overlap(masks[i], masks[j], by="layer")["total"]["jaccard"]
            if total is not None:
                grid[i, j] = total
                grid[j, i] = total
    labels = []
    for record in records:
        labels.append(record["id"].replace("mask_", "").replace("_llama1b", ""))
    fig, ax = plt.subplots(figsize=(1.5 + 0.5 * count, 1.0 + 0.5 * count))
    image = ax.imshow(grid, cmap="Reds")
    ax.set_xticks(range(count))
    ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=8)
    ax.set_yticks(range(count))
    ax.set_yticklabels(labels, fontsize=8)
    for i in range(count):
        for j in range(count):
            if not np.isnan(grid[i, j]):
                ax.text(j, i, "%.2f" % grid[i, j], ha="center", va="center", fontsize=6)
    ax.set_title("Jaccard overlap of deleted entries (all layers)")
    fig.colorbar(image, ax=ax, label="Jaccard")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path
