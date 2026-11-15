"""End-of-run held-out table: one part, its random-part control and a matched LoRA, 3 seeds.

The part is chosen on the search split (wsp_eval.gate.run_part_gate). Here every
arm is scored once on the held-out split (GSM8K test, and the if_guard guard).
Nothing is tuned on these numbers.

Each (arm, seed) runs in its own Modal container. Parallel containers must not
append to the same files on the volume (the last commit wins), so each one
works under its own root, /vol/eval_final/<donor>/<arm>_s<seed>. merge_ledgers
then copies their held-out entries into the main ledger in one step, so the
run-once rule still covers these models.
"""
import json
import os

from wsp_eval import claims

MAIN_ROOT = "/vol/eval"
FINAL_ROOT = "/vol/eval_final"
GATE_ROOT = "/vol/compiled/part_gate"
ARMS = ("graft", "random", "lora")


def arm_dir(donor, part_id, arm, seed):
    return os.path.join(GATE_ROOT, donor.replace("/", "__"), part_id, arm + "_s" + str(seed))


def arm_root(donor, arm, seed):
    return os.path.join(FINAL_ROOT, donor.replace("/", "__"), arm + "_s" + str(seed))


def main_ledger():
    path = os.path.join(MAIN_ROOT, "work", "heldout_ledger.json")
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


def already_run(model_path):
    """True if the main ledger already holds a held-out entry for this model path."""
    for entry in main_ledger().values():
        if entry["model"] == model_path:
            return True
    return False


def compile_lora(donor, part_id, seed, heal_steps=300):
    """A host LoRA matched to the graft's glue + part params, with the graft's heal."""
    from wsp import Catalog, Program
    from wsp.lora import graft_budget
    from wsp_eval import splits

    out_dir = arm_dir(donor, part_id, "lora", seed)
    if os.path.exists(os.path.join(out_dir, "compile_report.json")):
        return out_dir
    with open(os.path.join(arm_dir(donor, part_id, "graft", seed), "compile_report.json")) as f:
        budget = graft_budget(json.load(f)["grafts"])
    program = Program(host="unsloth/Llama-3.2-1B-Instruct", seed=seed).lora(budget_params=budget)
    program.heal(steps=heal_steps, data="gsm8k_train", lr=3e-4, batch_size=8, seq_len=384)
    heal_texts = splits.heal_texts("gsm8k")
    program.compile(out_dir, Catalog([], root="/"), align_texts=heal_texts[:200], heal_texts=heal_texts,
                    device="cuda", cache_dir="/vol/tensor_cache")
    return out_dir


def run_arm(donor, part_id, arm, seed, tasks=("gsm8k", "if_guard")):
    """Score one compiled arm on the held-out split, once. Call with WSP_ROOT=arm_root(...)."""
    from wsp_eval import harness

    if arm == "lora":
        model_path = compile_lora(donor, part_id, seed)
    else:
        model_path = arm_dir(donor, part_id, arm, seed)
    if already_run(model_path):
        raise harness.HeldoutAlreadyRun(model_path + " already has a held-out entry in the main ledger")
    result = harness.evaluate(model_path, "heldout", list(tasks), namespace="final", backend="hf",
                              note="final held-out " + part_id + " " + arm + " seed " + str(seed))
    row = {"arm": arm, "seed": seed, "model": model_path, "label": result["label"]}
    for task in tasks:
        row[task] = result["tasks"][task]["score"]
        row[task + "_n"] = result["tasks"][task]["n"]
    return {"row": row, "result": result}


def merge_ledgers(donor):
    """Copy every per-arm held-out entry into the main ledger (single writer)."""
    path = os.path.join(MAIN_ROOT, "work", "heldout_ledger.json")
    ledger = main_ledger()
    added = 0
    donor_root = os.path.join(FINAL_ROOT, donor.replace("/", "__"))
    for name in sorted(os.listdir(donor_root)):
        arm_ledger = os.path.join(donor_root, name, "work", "heldout_ledger.json")
        if not os.path.exists(arm_ledger):
            continue
        with open(arm_ledger) as f:
            entries = json.load(f)
        for key, entry in entries.items():
            if key not in ledger:
                ledger[key] = entry
                added += 1
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(ledger, f, indent=1)
    return added


def summarize(donor, part_id, outputs, host_heldout):
    """The final table and the claim verdict on GSM8K held-out."""
    by_arm = {}
    rows = []
    for arm in ARMS:
        by_arm[arm] = []
    for output in outputs:
        by_arm[output["row"]["arm"]].append(output["result"])
        rows.append(output["row"])
    verdict = claims.compare(by_arm["graft"], by_arm["random"], "gsm8k", lora_results=by_arm["lora"],
                             host_score=host_heldout.get("gsm8k"))
    return {"donor": donor, "part": part_id, "split": "heldout", "host": host_heldout, "rows": rows,
            "verdict": verdict, "verdict_text": claims.format_verdict(verdict)}
