# Local side: calls the deployed Modal Dissector, applies the validation gate,
# and writes results, catalog records and part index files.
import json
import os
import sys
import time

import modal

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from experiments import log  # noqa: E402

DONOR = "Qwen/Qwen2.5-Math-1.5B"
DONOR_SHORT = "qwenmath1.5b"
FAMILY = "qwen2.5"
D_MODEL = 1536
NUM_LAYERS = 28
NUM_HEADS = 12
MLP_WIDTH = 8960

CATALOG_PATH = os.path.join(ROOT, "catalog.jsonl")
PARTS_DIR = os.path.join(ROOT, "parts", DONOR_SHORT)
VALIDATIONS_PATH = os.path.join(HERE, "results", "validations.jsonl")

# Validation thresholds (from PLAN.md). Drops are relative: (base - ablated) / base.
TARGET_MIN_DROP = 0.30
CONTROL_MAX_DROP = 0.10
RANDOM_PERCENTILE_MIN = 0.95
CONTROL_TASKS = ["trivia", "wiki", "wiki_digits"]

_dissector = None


def dissector():
    global _dissector
    if _dissector is None:
        cls = modal.Cls.from_name("swarmy-dissect", "Dissector")
        _dissector = cls()
    return _dissector


def part_size(part):
    if part["kind"] == "block":
        return len(part["layers"])
    return len(part["indices"])


def check_part(part):
    # Returns an error string if the part is malformed, else None.
    kind = part.get("kind")
    if kind not in ["mlp_cluster", "attn_heads", "block"]:
        return "kind must be mlp_cluster, attn_heads or block"
    if kind == "block":
        if not part.get("layers"):
            return "block needs a non-empty 'layers' list"
        for l in part["layers"]:
            if l < 0 or l >= NUM_LAYERS:
                return "layer " + str(l) + " out of range"
        # team spec: a block is a contiguous range of whole layers
        for j in range(1, len(part["layers"])):
            if part["layers"][j] != part["layers"][j - 1] + 1:
                return "block layers must be a contiguous increasing range, e.g. [10, 11, 12]"
        return None
    if part.get("layer") is None or not part.get("indices"):
        return kind + " needs 'layer' and a non-empty 'indices' list"
    limit = MLP_WIDTH if kind == "mlp_cluster" else NUM_HEADS
    if part["layer"] < 0 or part["layer"] >= NUM_LAYERS:
        return "layer out of range"
    for i in part["indices"]:
        if i < 0 or i >= limit:
            return "index " + str(i) + " out of range 0.." + str(limit - 1) + " (use a candidate handle, not -1)"
    if len(set(part["indices"])) != len(part["indices"]):
        return "duplicate indices"
    return None


def percentile_above(value, others):
    # fraction of random drops that are strictly smaller than the candidate's drop
    below = 0
    for d in others:
        if d is not None and value is not None and d < value:
            below += 1
    return below / max(len(others), 1)


def apply_gate(evidence):
    # Target metric: GSM8K generation accuracy (greedy, exact match). The teacher-forced log-prob proxy
    # understated reasoning damage badly in the first run (9% proxy drop vs 48% accuracy drop), so it is
    # reported but not gated on. Controls are log-prob metrics.
    checks = {}
    reasons = []
    for method in ["zero", "mean"]:
        drop = evidence["methods"][method]["drop"]
        ok_target = drop["gsm8k_gen"] is not None and drop["gsm8k_gen"] >= TARGET_MIN_DROP
        ok_heldout = drop["gsm8k_gen_heldout"] is not None and drop["gsm8k_gen_heldout"] >= TARGET_MIN_DROP
        ok_controls = True
        for t in CONTROL_TASKS:
            if drop[t] is None or drop[t] >= CONTROL_MAX_DROP:
                ok_controls = False
                reasons.append(method + ": control " + t + " dropped " + str(drop[t]))
        if not ok_target:
            reasons.append(method + ": gen accuracy drop " + str(drop["gsm8k_gen"]) + " < " + str(TARGET_MIN_DROP))
        if not ok_heldout:
            # no number here: reasons are shown to the agent, and the held-out set must stay unseen
            reasons.append(method + ": held-out check failed")
        checks[method + "_target"] = ok_target
        checks[method + "_heldout"] = ok_heldout
        checks[method + "_controls"] = ok_controls

    mean_drop = evidence["methods"]["mean"]["drop"]
    pct_gen = percentile_above(mean_drop["gsm8k_gen"], evidence["random_gen_drops_mean"])
    pct_lp = percentile_above(mean_drop["gsm8k"], evidence["random_gsm8k_drops_mean"])
    checks["beats_random"] = pct_gen >= RANDOM_PERCENTILE_MIN
    if not checks["beats_random"]:
        reasons.append("random percentile (generation) " + str(round(pct_gen, 3)) + " < " + str(RANDOM_PERCENTILE_MIN))

    passed = True
    for k in checks:
        if not checks[k]:
            passed = False
    return {"passed": passed, "checks": checks, "random_percentile": round(pct_gen, 3),
            "random_percentile_logprob": round(pct_lp, 3), "reasons": reasons}


def validate_part(part, label, proposed_by, run_id, n_random=20):
    err = check_part(part)
    if err:
        return {"error": err}
    start = time.time()
    evidence = dissector().validate.remote(part, n_random)
    gate = apply_gate(evidence)
    result = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "run_id": run_id,
        "model": DONOR,
        "part": part,
        "label": label,
        "proposed_by": proposed_by,
        "gate": gate,
        "evidence": evidence,
        "seconds": round(time.time() - start, 1),
    }
    os.makedirs(os.path.dirname(VALIDATIONS_PATH), exist_ok=True)
    with open(VALIDATIONS_PATH, "a") as f:
        f.write(json.dumps(result) + "\n")
    if gate["passed"]:
        record = write_catalog(part, label, evidence, gate, proposed_by, run_id)
        result["catalog_id"] = record["id"]
    log("validate_part", {"part_kind": part["kind"], "size": part_size(part), "label": label, "run_id": run_id},
        {"passed": gate["passed"], "reasons": gate["reasons"],
         "drop_zero": evidence["methods"]["zero"]["drop"], "drop_mean": evidence["methods"]["mean"]["drop"],
         "random_percentile_gen": gate["random_percentile"]})
    return result


def summarize_validation(result):
    # Compact view for the agent (keeps tokens down).
    if "error" in result:
        return result
    ev = result["evidence"]
    # Held-out numbers are never shown to the agent (or to notes.jsonl, which future runs read):
    # only pass/fail per method. Otherwise the held-out set slowly turns into a tuning set.
    drop_zero = {}
    drop_mean = {}
    for k in ev["methods"]["zero"]["drop"]:
        if "heldout" not in k:
            drop_zero[k] = ev["methods"]["zero"]["drop"][k]
            drop_mean[k] = ev["methods"]["mean"]["drop"][k]
    checks = result["gate"]["checks"]
    return {
        "passed": result["gate"]["passed"],
        "reasons": result["gate"]["reasons"],
        "drop_zero": drop_zero,
        "drop_mean": drop_mean,
        "heldout_passed": {"zero": checks.get("zero_heldout"), "mean": checks.get("mean_heldout")},
        "gen_accuracy_base": ev["base"]["gsm8k_gen"]["score"],
        "gen_accuracy_zero": ev["methods"]["zero"]["ablated"]["gsm8k_gen"]["score"],
        "gen_accuracy_mean": ev["methods"]["mean"]["ablated"]["gsm8k_gen"]["score"],
        "random_gen_drops": ev["random_gen_drops_mean"],
        "random_percentile": result["gate"]["random_percentile"],
        "catalog_id": result.get("catalog_id"),
    }


def indices_list(part):
    if part["kind"] == "block":
        return list(part["layers"])
    return list(part["indices"])


def next_part_id(part, label):
    if part["kind"] == "block":
        where = "L" + str(part["layers"][0]) + "-" + str(part["layers"][-1])
    else:
        where = "L" + str(part["layer"]) + "_k" + str(len(part["indices"]))
    short_kind = {"mlp_cluster": "mlp", "attn_heads": "heads", "block": "block"}[part["kind"]]
    clean_label = ""
    for ch in label.lower().replace(" ", "-"):
        if ch.isalnum() or ch in "-_":
            clean_label = clean_label + ch
    if clean_label == "":
        clean_label = "unlabeled"
    base = short_kind + "_" + clean_label + "_" + DONOR_SHORT + "_" + where
    existing = set()
    if os.path.exists(CATALOG_PATH):
        with open(CATALOG_PATH) as f:
            for line in f:
                line = line.strip()
                if line:
                    existing.add(json.loads(line)["id"])
    n = 1
    while base + "_" + str(n).zfill(3) in existing:
        n += 1
    return base + "_" + str(n).zfill(3)


def same_part(record, part):
    if record.get("model") != DONOR or record.get("type") != part["kind"]:
        return False
    if part["kind"] == "block":
        return record.get("layers") == list(part["layers"])
    if record.get("layer") != int(part["layer"]):
        return False
    if record.get("num_units") != len(part["indices"]):
        return False
    stored = record.get("signature", {}).get("indices")
    if stored is None:
        return False
    return sorted(stored) == sorted(part["indices"])


def find_existing(part):
    if not os.path.exists(CATALOG_PATH):
        return None
    with open(CATALOG_PATH) as f:
        for line in f:
            line = line.strip()
            if line and same_part(json.loads(line), part):
                return json.loads(line)
    return None


def write_catalog(part, label, evidence, gate, proposed_by, run_id):
    existing = find_existing(part)
    if existing is not None:
        # already catalogued: keep one record; the new validation stays in results/validations.jsonl
        return existing
    part_id = next_part_id(part, label)
    os.makedirs(PARTS_DIR, exist_ok=True)
    index_path = os.path.join(PARTS_DIR, part_id + ".pt")
    if part["kind"] == "block":
        layer = int(part["layers"][0])
        layers = []
        for x in part["layers"]:
            layers.append(int(x))
    else:
        layer = int(part["layer"])
        layers = [layer]
    # torch.save runs on Modal so this machine doesn't need torch installed
    data = dissector().index_file_bytes.remote(part)
    with open(index_path, "wb") as f:
        f.write(data)

    record = {
        "id": part_id,
        "type": part["kind"],
        "trained_on": "GSM8K ablation (gate: generation-accuracy drop under zero+mean ablation incl. held-out, "
                      "log-prob controls trivia/wiki/wiki_digits, random same-size baseline)",
        "model": DONOR,
        "family": FAMILY,
        "layer": layer,
        "layers": layers,
        "units": os.path.relpath(index_path, ROOT),
        "d_model": D_MODEL,
        "num_units": len(part["indices"]) if part["kind"] != "block" else len(layers),
        "label": label,
        "signature": {
            "effect": {
                "zero": evidence["methods"]["zero"]["drop"],
                "mean": evidence["methods"]["mean"]["drop"],
            },
            "base": evidence["base"],
            "random_percentile": gate["random_percentile"],
            "checks": gate["checks"],
            "proposed_by": proposed_by,
            "run_id": run_id,
            "indices": indices_list(part),
        },
    }
    with open(CATALOG_PATH, "a") as f:
        f.write(json.dumps(record) + "\n")
    return record
