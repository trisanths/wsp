"""Mine a PILOT batch of real parts from three donors with Person B's extractor, on Modal.

Round 2 found single-layer clusters picked by mean |activation| about equal
to random ones, so this batch uses Person B's contrast extractor
(wsp_eval/extraction.py): units that respond more to GSM8K heal-split text
than to WikiText-2 train text, scored by Cohen's d (MLP neurons, query heads)
or by the log2 ratio of router pick rates (MoE experts). For each donor and
kind it keeps the top 4 layers, and each part comes with Person B's three
random same-size controls from the same layer.

Donors (repo id, license and config checked on the Hugging Face API):
    r1q15  deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B   mit         28 layers x 1536  mlp_cluster, attn_heads
    qmi15  Qwen/Qwen2.5-Math-1.5B-Instruct             apache-2.0  28 layers x 1536  mlp_cluster, attn_heads
    olmoe  allenai/OLMoE-1B-7B-0125-Instruct           apache-2.0  16 layers x 2048  expert (64 experts, top 8)

Where things go:
    index files  wsp-swarm-models volume, units/pilot/<id>.pt and
                 units/pilot/<id>_control_seed<k>.pt. The wsp-swarm app loads
                 the pilot catalog with root /vol, so a record's "units"
                 resolves to /vol/units/pilot/<id>.pt there, like the
                 existing pilot records.
    records      data/pilot_catalog_mined.jsonl, marked "pilot": true. Not the
                 Person A catalog.

Why not modal_eval.py::extract: in workspace rexheng that app would create a
new "wsp-models" volume, and its _extract_body runs the extractor with cwd
/vol/eval, which nothing creates on a fresh volume, so the call fails. It
also fixes the cluster size at the extractor's default (64 neurons). The
thin function here runs the same body (wsp_eval.extraction.extract) and
writes straight to the volume the wsp-swarm app reads.

Usage (Modal tokens from .env; nothing here names a workspace):
    set -a && . ./.env && set +a
    .venv/Scripts/python scripts/mine_pilot.py mine                      # all three donors
    .venv/Scripts/python scripts/mine_pilot.py mine --donors olmoe       # one donor; other donors' records stay
    .venv/Scripts/python scripts/mine_pilot.py probe                     # fit-only compile, one part per kind
    .venv/Scripts/python scripts/mine_pilot.py probe --parts pilot_r1q15_l14_mlp512

Every number this prints is PILOT: extractor statistics on GSM8K heal-split
text, and fit R squared on 64 heal-split alignment texts. None of them is a
search-split or held-out score. Costs are estimates from Modal list prices.
"""

import argparse
import json
import os
import shutil
import sys
import tempfile
import time

import modal

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if os.path.isdir(os.path.join(REPO, "wsp")) and REPO not in sys.path:
    sys.path.insert(0, REPO)

APP_NAME = "wsp-mine-pilot"
VOLUME_NAME = "wsp-swarm-models"
VOL = "/vol"
TENSOR_CACHE = VOL + "/tensor_cache"
UNITS_DIR = "units/pilot"
CONTAINER_ROOT = "/tmp/wsp_mine"   # WSP_ROOT inside the container: Person B's interp log lands here

HOST = "unsloth/Llama-3.2-1B-Instruct"
TASK = "gsm8k"
OUT_PATH = os.path.join(REPO, "data", "pilot_catalog_mined.jsonl")
EXISTING_PILOT_PATH = os.path.join(REPO, "data", "pilot_catalog.jsonl")

SOURCE = "wsp_eval extractor (Cohen d / router log2 ratio); pilot set, not the Person A catalog"

DONORS = {
    "r1q15": {"model": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", "license": "mit",
              "kinds": ["mlp_cluster", "attn_heads"], "gpu": "L4", "layers": 28, "hidden": 1536},
    "qmi15": {"model": "Qwen/Qwen2.5-Math-1.5B-Instruct", "license": "apache-2.0",
              "kinds": ["mlp_cluster", "attn_heads"], "gpu": "L4", "layers": 28, "hidden": 1536},
    "olmoe": {"model": "allenai/OLMoE-1B-7B-0125-Instruct", "license": "apache-2.0",
              "kinds": ["expert"], "gpu": "L4", "layers": 16, "hidden": 2048},
}
DONOR_ORDER = ["r1q15", "qmi15", "olmoe"]

# Units per part. 512 neurons matches round 2 and the hour-8 gate, so the
# parts compare with those runs. 2 heads and 4 experts are Person B's defaults.
UNIT_SIZES = {"mlp_cluster": 512, "attn_heads": 2, "expert": 4}
TOP_LAYERS = 4
N_TEXTS = 256
CONTROL_SEEDS = [0, 1, 2]
ALIGN_SAMPLES = 64     # alignment texts for the W_in ridge fit, as modal_swarm.py's probe_candidate uses
PROBE_GLUE_RANK = 64   # the ridge fit and its R squared do not depend on the glue rank

KIND_TAGS = {"mlp_cluster": "mlp", "attn_heads": "heads", "expert": "exp"}
KIND_METHODS = {
    "mlp_cluster": "MLP neurons by Cohen's d of |activation| (down_proj input)",
    "attn_heads": "query heads by Cohen's d of head output norm (o_proj input)",
    "expert": "experts by log2 ratio of router pick rates",
}

# Modal list prices in dollars per second (modal.com/pricing, read 2026-11-14).
PRICES = {"L4": 0.000222, "A10": 0.000306, "L40S": 0.000542, "cpu_core": 0.0000131, "memory_gib": 0.00000222}
FUNCTION_CPU = 4.0
FUNCTION_MEMORY_MIB = {"L4": 32768, "L40S": 65536}


# ---------------------------------------------------------------- names and records (pure, tested on CPU)


def part_id(short, kind, layer, n_units):
    """The pilot id of one mined part, for example pilot_r1q15_l14_mlp512."""
    if kind not in KIND_TAGS:
        raise ValueError("unknown part kind " + repr(kind))
    return "pilot_" + short + "_l" + str(int(layer)) + "_" + KIND_TAGS[kind] + str(int(n_units))


def units_path(pid):
    """The volume-relative index file of a part: units/pilot/<id>.pt."""
    return UNITS_DIR + "/" + pid + ".pt"


def control_path(pid, seed):
    """The volume-relative index file of one random control: units/pilot/<id>_control_seed<k>.pt."""
    return UNITS_DIR + "/" + pid + "_control_seed" + str(int(seed)) + ".pt"


def trained_on_text(kind, signature):
    """One line saying how the units were picked and on which texts."""
    return ("top " + str(signature["n_units"]) + " " + KIND_METHODS[kind] + ", " + signature["skill_data"] +
            " (GSM8K train minus the 200-item search split) vs " + signature["control_data"] +
            "; layer rank " + str(signature["layer_rank"]) + " for this donor and kind")


def build_signature(raw_signature, extractor_id):
    """Person B's signature fields plus the brief's empty "effect" and "fit_r2".

    "effect" stays empty on purpose: the swarm reads it as a task-score hint,
    and Cohen's d is not a score. Person B's original id is kept as
    "extractor_id".
    """
    signature = {"effect": {}, "fit_r2": {}}
    for key in raw_signature:
        signature[key] = raw_signature[key]
    signature["extractor_id"] = extractor_id
    return signature


def build_record(raw, short):
    """A pilot catalog record from one record that wsp_eval.extraction.extract returned.

    The shape is the brief's ({"id", "type", "model", "layer", "units",
    "trained_on", "signature"}) plus "pilot", "source" and Person B's
    "controls". Paths are volume-relative, so they resolve under /vol in the
    wsp-swarm containers.
    """
    kind = raw["type"]
    raw_signature = raw["signature"]
    pid = part_id(short, kind, raw["layer"], raw_signature["n_units"])
    controls = []
    for seed in range(len(raw["controls"])):
        controls.append(control_path(pid, seed))
    return {
        "id": pid,
        "type": kind,
        "model": raw["model"],
        "layer": int(raw["layer"]),
        "units": units_path(pid),
        "trained_on": trained_on_text(kind, raw_signature),
        "signature": build_signature(raw_signature, raw["id"]),
        "pilot": True,
        "source": SOURCE,
        "controls": controls,
    }


def record_order(record):
    """Sort key: donor order, then kind, then layer."""
    donor_rank = len(DONOR_ORDER)
    for index in range(len(DONOR_ORDER)):
        if DONORS[DONOR_ORDER[index]]["model"] == record["model"]:
            donor_rank = index
    return (donor_rank, record["type"], record["layer"], record["id"])


def merge_records(existing, new, mined_models):
    """Keep existing records of donors not mined this time, then add the new ones. Raises on a repeated id."""
    merged = []
    for record in existing:
        if record["model"] not in mined_models:
            merged.append(record)
    for record in new:
        merged.append(record)
    seen = []
    for record in merged:
        if record["id"] in seen:
            raise ValueError("repeated pilot id " + record["id"])
        seen.append(record["id"])
    merged.sort(key=record_order)
    return merged


def read_records(path):
    """Records from a JSONL file, or an empty list when the file does not exist."""
    records = []
    if not os.path.exists(path):
        return records
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line != "":
                records.append(json.loads(line))
    return records


def write_records(records, path):
    """Write one record per line with sorted keys, in record_order."""
    folder = os.path.dirname(path)
    if folder != "":
        os.makedirs(folder, exist_ok=True)
    ordered = list(records)
    ordered.sort(key=record_order)
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        for record in ordered:
            handle.write(json.dumps(record, sort_keys=True) + "\n")


def check_ids(records, other_ids):
    """Errors for repeated ids, ids already used by another catalog, or a units path that does not match the id."""
    errors = []
    seen = []
    for record in records:
        pid = record["id"]
        if pid in seen:
            errors.append("repeated id " + pid)
        seen.append(pid)
        if pid in other_ids:
            errors.append(pid + " is already used by another catalog")
        if record["units"] != units_path(pid):
            errors.append(pid + ": units " + record["units"] + " should be " + units_path(pid))
    return errors


def estimate_cost(seconds, gpu, cpu=FUNCTION_CPU, memory_mib=None):
    """Dollars for one container running `seconds` with this GPU, CPU and memory request (an estimate)."""
    if memory_mib is None:
        memory_mib = FUNCTION_MEMORY_MIB.get(gpu, 32768)
    per_second = PRICES[gpu] + cpu * PRICES["cpu_core"] + (memory_mib / 1024.0) * PRICES["memory_gib"]
    return round(seconds * per_second, 4)


def default_probe_parts(records):
    """One part per kind: the layer-rank-1 part of the first donor (in donor order) that has the kind."""
    chosen = []
    kinds_done = []
    ordered = list(records)
    ordered.sort(key=record_order)
    for record in ordered:
        if record["type"] in kinds_done:
            continue
        if record["signature"].get("layer_rank") != 1:
            continue
        chosen.append(record["id"])
        kinds_done.append(record["type"])
    return chosen


# ---------------------------------------------------------------- index file copies (container side, CPU-testable)


def copy_index_files(raw, record, root):
    """Copy one part's index file and its controls from the extractor's folder to root/<units paths>.

    The part file is copied byte for byte. Each control gets "control_of"
    rewritten to the part's new volume-relative path. Every written file is
    loaded back with torch.load's default (weights_only), which is how the
    compiler reads units. Returns the written paths.
    """
    import torch

    written = []
    target = os.path.join(root, *record["units"].split("/"))
    os.makedirs(os.path.dirname(target), exist_ok=True)
    shutil.copyfile(raw["units"], target)
    written.append(target)
    for seed in range(len(raw["controls"])):
        data = torch.load(raw["controls"][seed], map_location="cpu", weights_only=False)
        data["control_of"] = record["units"]
        control_target = os.path.join(root, *record["controls"][seed].split("/"))
        torch.save(data, control_target)
        written.append(control_target)
    for path in written:
        data = torch.load(path)
        if data["kind"] != record["type"] or int(data["layer"]) != record["layer"]:
            raise ValueError(path + " does not match record " + record["id"])
    return written


# ---------------------------------------------------------------- image and app


def build_image():
    """modal_swarm.py's image (same pins, so the pip layer is shared) plus wsp_eval."""
    return (
        modal.Image.debian_slim(python_version="3.12")
        .pip_install(
            "torch==2.7.1", "transformers==4.57.1", "accelerate", "huggingface_hub", "requests",
            "safetensors", "zstandard", "numpy", "datasets",
        )
        .env({"HF_HOME": VOL + "/hf", "WSP_ROOT": CONTAINER_ROOT})
        .add_local_python_source("wsp", "wsp_eval")
    )


app = modal.App(APP_NAME)
image = build_image()
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True, version=2)


def gpu_name():
    """The CUDA device this container got, or "none"."""
    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.get_device_name(0)
    except ImportError:
        pass
    return "none"


def free_gpu():
    import gc

    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def read_interp_log():
    """Person B's hypothesis, action, result lines written during this call."""
    path = os.path.join(CONTAINER_ROOT, "interp_log.jsonl")
    lines = []
    if os.path.exists(path):
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line != "":
                    lines.append(json.loads(line))
        os.remove(path)
    return lines


def mine_body(short, unit_sizes, top_layers, n_texts):
    """Run wsp_eval.extraction.extract on one donor and copy its index files to the volume."""
    start = time.time()
    os.makedirs(CONTAINER_ROOT, exist_ok=True)
    donor = DONORS[short]
    model_id = donor["model"]
    seconds = {}

    clock = time.time()
    from wsp_eval import extraction

    seconds["import"] = round(time.time() - clock, 2)
    clock = time.time()
    model, tokenizer, device = extraction.load_donor(model_id)
    seconds["load"] = round(time.time() - clock, 2)
    clock = time.time()
    skill = extraction.skill_texts(TASK, n_texts)
    control = extraction.control_texts(n_texts)
    seconds["texts"] = round(time.time() - clock, 2)

    clock = time.time()
    work_dir = tempfile.mkdtemp(prefix="wsp-mine-")
    raw_records = extraction.extract(model_id, task_name=TASK, kinds=donor["kinds"], n_units=unit_sizes,
                                     top_layers=top_layers, n_texts=n_texts, out_dir=work_dir,
                                     control_seeds=tuple(CONTROL_SEEDS), model=model, tokenizer=tokenizer,
                                     device=device, skill=skill, control=control)
    seconds["extract"] = round(time.time() - clock, 2)
    del model
    free_gpu()

    clock = time.time()
    records = []
    for raw in raw_records:
        record = build_record(raw, short)
        copy_index_files(raw, record, VOL)
        records.append(record)
    volume.commit()
    shutil.rmtree(work_dir, ignore_errors=True)
    seconds["copy"] = round(time.time() - clock, 2)
    seconds["total"] = round(time.time() - start, 2)
    return {"short": short, "model": model_id, "records": records, "seconds": seconds, "gpu": gpu_name(),
            "n_skill_texts": len(skill), "n_control_texts": len(control), "interp_log": read_interp_log()}


@app.function(image=image, gpu="L4", volumes={VOL: volume}, cpu=FUNCTION_CPU, memory=FUNCTION_MEMORY_MIB["L4"],
              timeout=2400, max_containers=1, scaledown_window=60)
def mine_l4(short, unit_sizes, top_layers, n_texts):
    return mine_body(short, unit_sizes, top_layers, n_texts)


@app.function(image=image, gpu="L40S", volumes={VOL: volume}, cpu=FUNCTION_CPU,
              memory=FUNCTION_MEMORY_MIB["L40S"], timeout=2400, max_containers=1, scaledown_window=60)
def mine_l40s(short, unit_sizes, top_layers, n_texts):
    return mine_body(short, unit_sizes, top_layers, n_texts)


MINE_FUNCTIONS = {"L4": mine_l4, "L40S": mine_l40s}


def graft_summary(graft):
    """The per-graft compile report fields worth reporting (no index lists)."""
    keys = ["id", "kind", "donor", "donor_layer", "host_layer", "d_host", "d_donor", "part_params", "glue_params",
            "glue_rank", "fit", "fit_r2", "fit_r2_dim", "fit_pairs"]
    summary = {}
    for key in keys:
        summary[key] = graft.get(key)
    summary["n_units"] = len(graft.get("indices", []))
    return summary


@app.function(image=image, gpu="L4", volumes={VOL: volume}, cpu=FUNCTION_CPU, memory=FUNCTION_MEMORY_MIB["L4"],
              timeout=2400, max_containers=1, scaledown_window=60)
def probe_l4(records, part_ids, glue_rank, align_samples):
    """Fit-only compile (heal None) of the host plus one mined part, for each part id.

    Same inputs as modal_swarm.py's probe_candidate: Trisanth's compile with
    the first align_samples GSM8K heal-split texts for the ridge fit, the
    tensor cache on the volume, and the catalog loaded with root /vol.
    """
    from wsp.catalog import Catalog
    from wsp.data import format_gsm8k, gsm8k_splits
    from wsp.program import Program

    start = time.time()
    catalog = Catalog(records, root=VOL)
    search_items, heal_items = gsm8k_splits(seed=0)
    align = []
    for item in heal_items[:align_samples]:
        align.append(format_gsm8k(item))
    setup_s = round(time.time() - start, 2)

    results = []
    for pid in part_ids:
        program = Program(host=HOST, seed=0).graft(pid, at=None, glue_rank=glue_rank)
        work_dir = tempfile.mkdtemp(prefix="wsp-mine-probe-")
        clock = time.time()
        entry = {"id": pid, "ok": False, "error": None, "graft": None}
        try:
            program.compile(work_dir, catalog, align_texts=align, heal_texts=None, device="cuda",
                            cache_dir=TENSOR_CACHE)
            with open(os.path.join(work_dir, "compile_report.json"), encoding="utf-8") as handle:
                report = json.load(handle)
            entry["graft"] = graft_summary(report["grafts"][0])
            entry["ok"] = True
        except Exception as error:
            entry["error"] = type(error).__name__ + ": " + str(error)[:2000]
        entry["compile_s"] = round(time.time() - clock, 2)
        shutil.rmtree(work_dir, ignore_errors=True)
        free_gpu()
        results.append(entry)
    volume.commit()
    return {"results": results, "gpu": gpu_name(), "setup_s": setup_s, "total_s": round(time.time() - start, 2)}


# ---------------------------------------------------------------- local driver


def check_env():
    for name in ["MODAL_TOKEN_ID", "MODAL_TOKEN_SECRET"]:
        if os.environ.get(name, "") == "":
            raise SystemExit(name + " is not set. Run: set -a && . ./.env && set +a")


def parse_list(text):
    values = []
    for piece in text.split(","):
        piece = piece.strip()
        if piece != "":
            values.append(piece)
    return values


def existing_ids(path):
    ids = []
    for record in read_records(path):
        ids.append(record["id"])
    return ids


def run_mine(shorts, unit_sizes, top_layers, n_texts, out_path):
    """Mine each donor in turn on Modal and write the catalog. Returns (results, records)."""
    results = []
    with modal.enable_output():
        with app.run():
            for short in shorts:
                donor = DONORS[short]
                function = MINE_FUNCTIONS[donor["gpu"]]
                clock = time.time()
                try:
                    result = function.remote(short, unit_sizes, top_layers, n_texts)
                    result["error"] = None
                except Exception as error:
                    result = {"short": short, "model": donor["model"], "records": [], "seconds": {},
                              "gpu": None, "error": type(error).__name__ + ": " + str(error)[:2000]}
                result["gpu_requested"] = donor["gpu"]
                result["client_wall_s"] = round(time.time() - clock, 2)
                result["cost_estimate_usd"] = estimate_cost(result["client_wall_s"], donor["gpu"])
                results.append(result)

    new_records = []
    mined_models = []
    for result in results:
        if result["error"] is None:
            mined_models.append(result["model"])
            for record in result["records"]:
                new_records.append(record)
    records = merge_records(read_records(out_path), new_records, mined_models)
    errors = check_ids(records, existing_ids(EXISTING_PILOT_PATH))
    if len(errors) > 0:
        raise SystemExit("catalog not written: " + "; ".join(errors))
    write_records(records, out_path)
    return results, records


def print_mine(results, records, out_path):
    total_cost = 0.0
    for result in results:
        print("")
        print("== " + result["short"] + "  " + result["model"])
        if result["error"] is not None:
            print("   FAILED: " + result["error"])
        print("   gpu requested " + str(result["gpu_requested"]) + ", got " + str(result["gpu"]) +
              "; client wall " + str(result["client_wall_s"]) + " s; container " +
              json.dumps(result["seconds"], sort_keys=True))
        print("   estimated cost $" + str(result["cost_estimate_usd"]) + " (Modal list prices, client wall)")
        total_cost += result["cost_estimate_usd"]
        for record in result["records"]:
            signature = record["signature"]
            print("   %-28s layer %2d  effect %7.3f  random %7.3f  [PILOT, extractor contrast]" % (
                record["id"], record["layer"], signature["mean_effect"], signature["random_mean_effect"]))
    print("")
    print("estimated total $" + str(round(total_cost, 4)))
    print("PILOT catalog: " + str(len(records)) + " records in " + out_path)


def run_probe(records, part_ids, glue_rank):
    with modal.enable_output():
        with app.run():
            clock = time.time()
            result = probe_l4.remote(records, part_ids, glue_rank, ALIGN_SAMPLES)
            result["client_wall_s"] = round(time.time() - clock, 2)
    result["cost_estimate_usd"] = estimate_cost(result["client_wall_s"], "L4")
    return result


def print_probe(result):
    print("")
    print("probe on " + str(result["gpu"]) + ": client wall " + str(result["client_wall_s"]) + " s, estimated $" +
          str(result["cost_estimate_usd"]))
    for entry in result["results"]:
        if not entry["ok"]:
            print("   %-28s COMPILE FAILED after %s s: %s" % (entry["id"], entry["compile_s"], entry["error"]))
            continue
        graft = entry["graft"]
        print("   %-28s host layer %2d  fit %s  fit_r2 %.4f  fit_r2_dim %s  pairs %s  part %s  glue %s  %s s"
              "  [PILOT, 64 heal-split align texts]" % (
                  entry["id"], graft["host_layer"], graft["fit"], graft["fit_r2"], graft["fit_r2_dim"],
                  graft["fit_pairs"], graft["part_params"], graft["glue_params"], entry["compile_s"]))


def save_summary(path, data):
    if path is None:
        return
    folder = os.path.dirname(path)
    if folder != "":
        os.makedirs(folder, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True)
    print("summary written to " + path)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    mine = sub.add_parser("mine", help="extract parts on Modal and write the catalog")
    mine.add_argument("--donors", default=",".join(DONOR_ORDER))
    mine.add_argument("--neurons", type=int, default=UNIT_SIZES["mlp_cluster"])
    mine.add_argument("--heads", type=int, default=UNIT_SIZES["attn_heads"])
    mine.add_argument("--experts", type=int, default=UNIT_SIZES["expert"])
    mine.add_argument("--top-layers", type=int, default=TOP_LAYERS)
    mine.add_argument("--texts", type=int, default=N_TEXTS)
    mine.add_argument("--out", default=OUT_PATH)
    mine.add_argument("--summary", default=None, help="optional JSON file for the full results")
    probe = sub.add_parser("probe", help="fit-only compile of host plus a mined part on Modal")
    probe.add_argument("--parts", default="", help="comma list of ids (default: one part per kind)")
    probe.add_argument("--catalog", default=OUT_PATH)
    probe.add_argument("--glue-rank", type=int, default=PROBE_GLUE_RANK)
    probe.add_argument("--summary", default=None)
    args = parser.parse_args()
    check_env()

    if args.command == "mine":
        shorts = parse_list(args.donors)
        for short in shorts:
            if short not in DONORS:
                raise SystemExit("unknown donor " + short + "; choose from " + ", ".join(DONOR_ORDER))
        unit_sizes = {"mlp_cluster": args.neurons, "attn_heads": args.heads, "expert": args.experts}
        results, records = run_mine(shorts, unit_sizes, args.top_layers, args.texts, args.out)
        print_mine(results, records, args.out)
        save_summary(args.summary, {"results": results, "records": records})
        return

    records = read_records(args.catalog)
    if len(records) == 0:
        raise SystemExit(args.catalog + " has no records; run the mine command first")
    part_ids = parse_list(args.parts)
    if len(part_ids) == 0:
        part_ids = default_probe_parts(records)
    result = run_probe(records, part_ids, args.glue_rank)
    print_probe(result)
    save_summary(args.summary, result)


if __name__ == "__main__":
    main()
