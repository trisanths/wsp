"""Demo visuals: per-layer deletion heatmap, pairwise overlap heatmap, neuron maps.

    python scripts/make_heatmaps.py --ids mask_mmlupro_biology_llama1b mask_mmlupro_chemistry_llama1b
    python scripts/make_heatmaps.py --family mmlupro          # all 14 MMLU-Pro masks

No model is run, so this costs no eval budget. Images go to outputs/heatmaps/.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from wsp_eval import catalog, heatmaps  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ids", nargs="*", default=None)
    parser.add_argument("--family", default=None, help="substring of the id, e.g. mmlupro or bbeh")
    parser.add_argument("--out", default="outputs/heatmaps")
    parser.add_argument("--neuron-maps", action="store_true", help="also draw one neuron map per mask")
    args = parser.parse_args()

    ids = args.ids
    if ids is None:
        ids = []
        for record in catalog.load():
            if record["type"] == "deletion_mask" and (args.family is None or args.family in record["id"]):
                ids.append(record["id"])

    os.makedirs(args.out, exist_ok=True)
    print(heatmaps.deletion_heatmap(os.path.join(args.out, "deletions_per_layer.png"), ids))
    if len(ids) >= 2:
        print(heatmaps.overlap_heatmap(os.path.join(args.out, "overlap_per_layer.png"), ids))
    if args.neuron_maps:
        for component_id in ids:
            print(heatmaps.neuron_heatmap(component_id, os.path.join(args.out, "neurons_" + component_id + ".png")))


if __name__ == "__main__":
    main()
