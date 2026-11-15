"""Hour-8 gate: host, donor, no-graft control, and a compiled graft, scored by eval().

Runs on a GPU (see modal_eval.py::gate). For each seed it compiles, with the
same heal (data, steps, lr, batch, seq_len) for every arm:
  graft     the extracted part, grafted with rank-64 glue
  random    control 1: the same glue around a random part of the same size
            from the same donor layer (Program.graft(..., control="random"))
  lora      control 2: a host LoRA matched to the graft's glue + part params
and scores host, donor, and every compiled model with eval() on the GSM8K
search split, all with the hf backend so every row is generated the same way.
Then claims.compare decides whether the graft beats control 1 on every seed.

All numbers are SEARCH-SPLIT numbers: they gate the pipeline, they are not results.
"""
import json
import os

from wsp_eval import claims, config, extraction, harness, splits, units

HOST = "unsloth/Llama-3.2-1B-Instruct"
DONOR = "Qwen/Qwen2.5-Math-1.5B"


def extract_part(donor_layer, neurons, out_dir):
    """A contrast-selected MLP cluster at one donor layer (GSM8K heal text vs WikiText-2)."""
    records = extraction.extract(DONOR, task_name="gsm8k", kinds=["mlp_cluster"], layers=[donor_layer],
                                 n_units={"mlp_cluster": neurons}, top_layers=1, n_texts=256, out_dir=out_dir)
    return records[0]


def compile_arm(program, catalog, out_dir, align_texts, heal_texts):
    if os.path.exists(os.path.join(out_dir, "compile_report.json")):
        return out_dir
    program.compile(out_dir, catalog, align_texts=align_texts, heal_texts=heal_texts, device="cuda",
                    cache_dir="/vol/tensor_cache")
    return out_dir


def run_gate(seeds=(0, 1, 2), donor_layer=10, neurons=512, glue_rank=64, heal_steps=300,
             out_root="/vol/compiled/gate", namespace="gate"):
    from wsp import Catalog, Program
    from wsp.lora import graft_budget

    os.makedirs(out_root, exist_ok=True)
    part = extract_part(donor_layer, neurons, os.path.join(config.work_dir(), "gate_parts"))
    part_id = "qmath_l" + str(donor_layer) + "_contrast" + str(neurons)
    record = {"id": part_id, "type": "mlp_cluster", "model": DONOR, "layer": donor_layer,
              "units": part["units"], "trained_on": "GSM8K heal-split text vs WikiText-2 (contrast)",
              "signature": part["signature"]}
    catalog = Catalog([record], root="/")

    heal_texts = splits.heal_texts("gsm8k")
    align_texts = heal_texts[:200]

    rows = []

    def score(label, model_path, seed):
        result = harness.evaluate(model_path, "search", ["gsm8k"], namespace=namespace, backend="hf",
                                  note="gate " + label + " seed " + str(seed))
        row = {"arm": label, "seed": seed, "model": model_path, "score": result["tasks"]["gsm8k"]["score"],
               "n": result["tasks"]["gsm8k"]["n"], "fit_r2": result["fit_r2"], "label": result["label"],
               "program_seed": result["program_seed"]}
        rows.append(row)
        print("[gate]", label, "seed", seed, "score", row["score"], "fit", json.dumps(row["fit_r2"])[:200], flush=True)
        return result

    score("host", HOST, None)
    score("donor", DONOR, None)

    results = {"graft": [], "random": [], "lora": []}
    for seed in seeds:
        graft = Program(host=HOST, seed=seed).graft(part_id, glue_rank=glue_rank)
        graft.heal(steps=heal_steps, data="gsm8k_train", lr=3e-4, batch_size=8, seq_len=384)
        graft_dir = compile_arm(graft, catalog, os.path.join(out_root, "graft_s" + str(seed)), align_texts, heal_texts)
        results["graft"].append(score("graft", graft_dir, seed))

        control = Program(host=HOST, seed=seed).graft(part_id, glue_rank=glue_rank, control="random",
                                                       control_seed=seed + 1)
        control.heal(steps=heal_steps, data="gsm8k_train", lr=3e-4, batch_size=8, seq_len=384)
        control_dir = compile_arm(control, catalog, os.path.join(out_root, "random_s" + str(seed)), align_texts,
                                  heal_texts)
        results["random"].append(score("random", control_dir, seed))

        with open(os.path.join(graft_dir, "compile_report.json")) as f:
            budget = graft_budget(json.load(f)["grafts"])
        lora = Program(host=HOST, seed=seed).lora(budget_params=budget)
        lora.heal(steps=heal_steps, data="gsm8k_train", lr=3e-4, batch_size=8, seq_len=384)
        lora_dir = compile_arm(lora, catalog, os.path.join(out_root, "lora_s" + str(seed)), align_texts, heal_texts)
        results["lora"].append(score("lora", lora_dir, seed))

    verdict = claims.compare(results["graft"], results["random"], "gsm8k", lora_results=results["lora"],
                             host_score=rows[0]["score"])
    print(claims.format_verdict(verdict), flush=True)
    summary = {"part": record, "rows": rows, "verdict": verdict,
               "settings": {"donor_layer": donor_layer, "neurons": neurons, "glue_rank": glue_rank,
                            "heal_steps": heal_steps, "heal": "gsm8k_train, lr 3e-4, batch 8, seq 384",
                            "seeds": list(seeds)}}
    with open(os.path.join(config.work_dir(), "gate_summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)
    return summary


def load_records(donor):
    """The extraction records for a donor, from the Modal volume (written by modal_eval.py::extract)."""
    path = os.path.join(config.work_dir(), "extracted", donor.replace("/", "__"), "records.jsonl")
    records = []
    with open(path) as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def pick_part(records, part_id=None):
    """The record named part_id, else the top-ranked mlp_cluster."""
    if part_id:
        for record in records:
            if record["id"] == part_id:
                return record
        raise KeyError("no extracted part " + part_id)
    best = None
    for record in records:
        if record["type"] != "mlp_cluster":
            continue
        if best is None or record["signature"]["layer_rank"] < best["signature"]["layer_rank"]:
            best = record
    if best is None:
        raise KeyError("no mlp_cluster in the extraction records")
    return best


def run_part_gate(donor, part_id=None, seeds=(0, 1, 2), glue_rank=64, heal_steps=300,
                  out_root="/vol/compiled/part_gate", namespace="gate"):
    """Hour-16 control for one donor: an extracted part against a random part of the same size.

    Same heal for both arms; 3 seeds; scored with eval() on the GSM8K search split.
    SEARCH-SPLIT numbers: they gate the pipeline, they are not results.
    """
    from wsp import Catalog, Program

    record = pick_part(load_records(donor), part_id)
    catalog = Catalog([record], root="/")
    heal_texts = splits.heal_texts("gsm8k")
    align_texts = heal_texts[:200]
    out_dir = os.path.join(out_root, donor.replace("/", "__"), record["id"])

    rows = []
    results = {"graft": [], "random": []}
    for seed in seeds:
        for arm in ("graft", "random"):
            program = Program(host=HOST, seed=seed)
            if arm == "graft":
                program.graft(record["id"], glue_rank=glue_rank)
            else:
                program.graft(record["id"], glue_rank=glue_rank, control="random", control_seed=seed + 1)
            program.heal(steps=heal_steps, data="gsm8k_train", lr=3e-4, batch_size=8, seq_len=384)
            arm_dir = compile_arm(program, catalog, os.path.join(out_dir, arm + "_s" + str(seed)), align_texts,
                                  heal_texts)
            result = harness.evaluate(arm_dir, "search", ["gsm8k"], namespace=namespace, backend="hf",
                                      note="part gate " + record["id"] + " " + arm + " seed " + str(seed))
            results[arm].append(result)
            row = {"arm": arm, "seed": seed, "score": result["tasks"]["gsm8k"]["score"],
                   "n": result["tasks"]["gsm8k"]["n"], "fit_r2": result["fit_r2"]}
            rows.append(row)
            print("[part_gate]", record["id"], arm, "seed", seed, "score", row["score"], flush=True)

    verdict = claims.compare(results["graft"], results["random"], "gsm8k")
    print(claims.format_verdict(verdict), flush=True)
    summary = {"donor": donor, "part": record["id"], "layer": record["layer"],
               "n_units": record["signature"]["n_units"], "rows": rows, "verdict": verdict,
               "settings": {"glue_rank": glue_rank, "heal_steps": heal_steps, "seeds": list(seeds),
                            "heal": "gsm8k_train, lr 3e-4, batch 8, seq 384"}}
    name = "part_gate_" + donor.replace("/", "__") + ".json"
    with open(os.path.join(config.work_dir(), name), "w") as f:
        json.dump(summary, f, indent=2, default=str)
    return summary
