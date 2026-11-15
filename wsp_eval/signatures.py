"""Structural signatures for deletion masks: no model runs, so no eval budget.

For every deletion_mask record in the catalog this fills:
  num_deleted         total deleted weights
  deleted_per_layer   deleted weights per host layer
  overlap_with        {other id: {jaccard, shared, jaccard_per_layer}} for every
                      other mask on the same base
  top_overlaps        the 5 masks it shares the most deletions with

Measured effects (score changes) come from the interp agent, which needs GPU
evals; this module is the free part.
"""
import datetime

from wsp_eval import catalog, components, overlap_tool


def structural_signatures(catalog_path=None):
    records = []
    for record in catalog.load(catalog_path):
        if record["type"] == "deletion_mask":
            records.append(record)

    masks = {}
    for record in records:
        masks[record["id"]] = components.load_component_mask(record, catalog_path)
        print("[signatures] loaded", record["id"], flush=True)

    overlaps = {}
    for record in records:
        overlaps[record["id"]] = {}
    for i in range(len(records)):
        for j in range(i + 1, len(records)):
            a = records[i]
            b = records[j]
            if a["base"] != b["base"]:
                continue
            stats = overlap_tool.overlap(masks[a["id"]], masks[b["id"]], by="layer")
            per_layer = {}
            for layer, values in stats["groups"].items():
                per_layer[str(layer)] = values["jaccard"]
            summary = {"jaccard": stats["total"]["jaccard"], "shared": stats["total"]["shared"],
                       "jaccard_per_layer": per_layer}
            overlaps[a["id"]][b["id"]] = summary
            overlaps[b["id"]][a["id"]] = summary

    now = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
    for record in records:
        num_layers = components.num_layers_of_base(record["base"])
        per_layer = components.deleted_per_layer(masks[record["id"]], num_layers)
        ranked = sorted(overlaps[record["id"]].items(), key=lambda pair: pair[1]["jaccard"] or 0, reverse=True)
        top = []
        for other_id, summary in ranked[:5]:
            top.append({"id": other_id, "jaccard": summary["jaccard"]})
        catalog.update_signature(record["id"], {
            "num_deleted": sum(per_layer),
            "deleted_per_layer": per_layer,
            "overlap_with": overlaps[record["id"]],
            "top_overlaps": top,
            "structure_updated": now,
        }, catalog_path)
        print("[signatures] wrote", record["id"], "deleted", sum(per_layer), flush=True)
    return len(records)
