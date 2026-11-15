"""The shared Sapien library: build, validate, index, add, and query.

The library (library/ at the repo root) holds the mined building blocks
that Sapien programs are written from: model profiles, components
(deletion masks, MLP clusters, attention heads, experts, blocks), measured
findings, stitch recipes, and programs that were run. See
library/README.md for the layout and library/CONTRIBUTING.md for the rules.

    python scripts/library.py build      regenerate the generated entries from repo sources
    python scripts/library.py index      write library/index.jsonl
    python scripts/library.py validate   check every entry; exit 1 on an error
    python scripts/library.py add FILE   add one new entry from a JSON file
    python scripts/library.py add FILE --append-evidence
                                         append the evidence items in FILE to entry FILE["id"]
    python scripts/library.py query --kind mlp_cluster --status validated --skill math --model Qwen

Every command takes --library DIR (default: library/ in this repo).
build also takes --repo DIR (default: this repo).

build never deletes a file and never touches an entry whose "generated"
field is not true. On a generated entry it keeps every evidence item that
is not marked "generated": true, and then it keeps the entry's status too.
"""

import argparse
import csv
import datetime
import glob
import json
import os
import re
import shutil
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_LIBRARY = os.path.join(REPO, "library")

COMPONENT_KINDS = ["deletion_mask", "mlp_cluster", "attn_heads", "expert", "block"]
KINDS = ["model"] + COMPONENT_KINDS + ["recipe", "finding", "stitch"]
STATUSES = ["candidate", "validated", "rejected"]

# Fields every entry has.
COMMON_FIELDS = ["id", "kind", "title", "status", "added_by", "source", "updated", "skills", "models",
                 "evidence", "generated"]
# Fields by kind, on top of the common ones.
KIND_FIELDS = {
    "model": ["hf_id", "family", "params_b", "layers", "hidden", "vocab", "tokenizer_family", "license",
              "scores", "structure_notes", "components"],
    "component": ["model", "layer", "layers", "units_file", "units_remote", "signature"],
    "recipe": ["host", "program", "text", "program_file", "syntax_gap"],
    "finding": ["claim", "setup", "numbers"],
    "stitch": ["host", "donor", "host_layers", "donor_layer", "stitch_kind", "objective", "steps", "sink",
               "penalty"],
}
EVIDENCE_FIELDS = ["claim", "metric", "value", "control_value", "split", "seeds", "source"]
SCORE_FIELDS = ["value", "split", "prompt", "source"]
INDEX_FIELDS = ["id", "kind", "path", "title", "skills", "models", "status", "added_by", "updated"]
ID_PATTERN = re.compile(r"^[A-Za-z0-9_.\-]+$")

FINDINGS_FILE = "findings/findings.jsonl"
STITCHES_FILE = "stitches/stitches.jsonl"

# Who did what, from the briefs, the folder READMEs, and the git log.
# Trisanth: the compiler and the runs in results/README.md (rounds, audit, baselines, stitch study).
# Hanav (Person A): person_a/ mining. Person B (part 1b): wsp_eval/, parts_extracted/, and the
# gate, held-out, bake, stitch graft, and exit stitch files in results/ (git author airjan-airlines,
# branch person-b-eval). Rex (part 3, the swarm): the pilot parts in data/pilot_catalog*.jsonl.
TRISANTH = "Trisanth"
HANAV = "Hanav (Person A)"
PERSON_B = "Person B"
REX_PILOT = "Rex (part 3 pilot)"

GSM8K_SEARCH = "GSM8K search split (200 items from GSM8K train)"
GSM8K_HELDOUT = "GSM8K test, 1319 items (held-out)"
MMLU_DEV = "MMLU test, 1000 seeded items (round 6 dev)"
MMLU_HELDOUT = "MMLU test, 2000 seeded items disjoint from dev (round 7 held-out)"
MEDMCQA = "MedMCQA validation, 1000 seeded items (round 10)"
MCQ_PROMPT = "round 6 prompt (Question, lettered options, Answer:), argmax of the option-letter log-probs"
RAW_PROMPT = "raw 'Question: ... Answer:' prompt, greedy generation"

# Vocabulary sizes by tokenizer family, copied from composer/sizes.py.
VOCAB_BY_TOKENIZER = {
    "qwen2": 151936,
    "llama3": 128256,
    "gemma": 256000,
    "phi3": 32064,
    "phi4": 100352,
    "mistral_v3": 32768,
    "mistral_v1": 32000,
    "lfm2": 65536,
}

# Short names the results files use for models (from wsp/profiles.py, plus a few more).
RESULT_KEYS = {
    "llama1b": "unsloth/Llama-3.2-1B-Instruct",
    "llama1": "unsloth/Llama-3.2-1B-Instruct",
    "qwen05": "Qwen/Qwen2.5-0.5B-Instruct",
    "q05": "Qwen/Qwen2.5-0.5B-Instruct",
    "qwen7b": "Qwen/Qwen2.5-7B-Instruct",
    "q7b": "Qwen/Qwen2.5-7B-Instruct",
    "qwen14b": "Qwen/Qwen2.5-14B-Instruct",
    "q14b": "Qwen/Qwen2.5-14B-Instruct",
    "qwen32b": "Qwen/Qwen2.5-32B-Instruct",
    "q32b": "Qwen/Qwen2.5-32B-Instruct",
    "gemma9b": "unsloth/gemma-2-9b-it",
    "llama8b": "unsloth/Meta-Llama-3.1-8B-Instruct",
    "mistral7b": "unsloth/mistral-7b-instruct-v0.3",
    "phi35": "microsoft/Phi-3.5-mini-instruct",
    "phi4": "microsoft/phi-4",
    "biomistral": "BioMistral/BioMistral-7B",
    "lfm12b": "LiquidAI/LFM2.5-1.2B-Instruct",
}

# Tap read layers used in runs 8 and 9 (modal_round8.py DONORS, modal_round9.py MODELS).
ROUND8_READ = {
    "qwen7b": 20,
    "qwen14b": 34,
    "llama8b": 22,
    "gemma9b": 29,
    "mistral7b": 22,
    "phi35": 22,
    "lfm12b": 11,
}


# Small file helpers


def today():
    return datetime.date.today().isoformat()


def read_json(path):
    with open(path) as f:
        return json.load(f)


def read_jsonl(path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def json_text(data):
    return json.dumps(data, indent=2) + "\n"


def write_text(path, text):
    """Write text to path unless the file already holds it. Return True when it wrote."""
    if os.path.exists(path):
        with open(path) as f:
            if f.read() == text:
                return False
    folder = os.path.dirname(path)
    if folder:
        os.makedirs(folder, exist_ok=True)
    with open(path, "w") as f:
        f.write(text)
    return True


def copy_file(src, dest):
    """Copy src to dest unless dest already has the same bytes. Return True when it copied."""
    if os.path.exists(dest):
        with open(src, "rb") as a:
            with open(dest, "rb") as b:
                if a.read() == b.read():
                    return False
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    shutil.copyfile(src, dest)
    return True


def model_key(hf_id):
    """Qwen/Qwen2.5-7B-Instruct -> Qwen__Qwen2.5-7B-Instruct (the folder and file name)."""
    return hf_id.replace("/", "__")


def pct(value):
    """A fraction rounded to 4 places, or None."""
    if value is None:
        return None
    return round(float(value), 4)


def mean(values):
    if len(values) == 0:
        return None
    total = 0.0
    for value in values:
        total += value
    return total / len(values)


def tag(hf_id):
    """Qwen/Qwen2.5-14B-Instruct -> qwen25_14b, for readable ids."""
    text = short_model(hf_id).lower().replace(".", "").replace("-instruct", "")
    return text.replace("-", "_")


def short_model(hf_id):
    if "/" in hf_id:
        return hf_id.split("/", 1)[1]
    return hf_id


# Entry builders


def evidence(claim, metric, value, control_value, split, seeds, source, note=None, control=None):
    """One evidence item. Items made by build carry generated: true."""
    item = {"claim": claim, "metric": metric, "value": value, "control_value": control_value,
            "split": split, "seeds": seeds, "source": source, "generated": True}
    if control is not None:
        item["control"] = control
    if note is not None:
        item["note"] = note
    # Losses and penalties: a lower value is better.
    lowered = str(metric).lower()
    if "penalty" in lowered or "loss" in lowered:
        item["lower_is_better"] = True
    return item


def entry(entry_id, kind, title, status, added_by, source, skills, models, items, fields):
    data = {"id": entry_id, "kind": kind, "title": title, "status": status, "added_by": added_by,
            "source": source, "updated": None, "skills": skills, "models": models, "evidence": items,
            "generated": True}
    for key in fields:
        data[key] = fields[key]
    return data


def component(entry_id, kind, title, status, added_by, source, skills, model, layers, units_file, units_remote,
              signature, items, extra=None):
    layer = None
    if len(layers) == 1:
        layer = layers[0]
    fields = {"model": model, "layer": layer, "layers": layers, "units_file": units_file,
              "units_remote": units_remote, "signature": signature}
    if extra is not None:
        for key in extra:
            fields[key] = extra[key]
    return entry(entry_id, kind, title, status, added_by, source, skills, [model], items, fields)


def entry_path(data):
    """Where an entry lives, relative to the library folder."""
    kind = data["kind"]
    if kind == "model":
        return "models/" + data["id"] + ".json"
    if kind in COMPONENT_KINDS:
        return "components/" + model_key(data["model"]) + "/" + data["id"] + ".json"
    if kind == "recipe":
        return "recipes/" + data["id"] + ".json"
    if kind == "finding":
        return FINDINGS_FILE
    if kind == "stitch":
        return STITCHES_FILE
    raise ValueError("unknown kind: " + str(kind))


# Models


def parse_donors_md(repo):
    """Rows of the Liquid AI table in docs/donors.md, by Hugging Face id."""
    path = os.path.join(repo, "docs", "donors.md")
    out = {}
    if not os.path.exists(path):
        return out
    with open(path) as f:
        for line in f:
            if not line.startswith("| LiquidAI/"):
                continue
            cells = line.strip().strip("|").split("|")
            clean = []
            for cell in cells:
                clean.append(cell.strip())
            names = clean[0].split(",")
            for index in range(len(names)):
                name = names[index].strip()
                if not name.startswith("LiquidAI/"):
                    name = "LiquidAI/" + name
                row = {"type": clean[1], "params": clean[2], "layers": clean[3], "hidden": clean[4],
                       "vocab": clean[5], "transformers": clean[6]}
                out[name] = row
    return out


def int_or_none(text):
    text = str(text).strip()
    match = re.match(r"^(\d+)", text)
    if match is None:
        return None
    return int(match.group(1))


def read_registry(repo):
    """person_a/registry/donors.csv rows with their config.json, by Hugging Face id."""
    path = os.path.join(repo, "person_a", "registry", "donors.csv")
    out = {}
    if not os.path.exists(path):
        return out
    with open(path) as f:
        for row in csv.DictReader(f):
            config_path = os.path.join(repo, "person_a", "registry", "configs", row["short_name"] + ".json")
            config = {}
            if os.path.exists(config_path):
                config = read_json(config_path)
                if "text_config" in config:
                    text = dict(config["text_config"])
                    text["model_type_outer"] = config.get("model_type")
                    config = text
            row["config"] = config
            row["config_path"] = "person_a/registry/configs/" + row["short_name"] + ".json"
            out[row["hf_id"]] = row
    return out


def read_zoo(repo):
    path = os.path.join(repo, "atlas", "zoo.json")
    out = {}
    if not os.path.exists(path):
        return out
    for model in read_json(path)["models"]:
        out[model["id"]] = model
    return out


def read_profiles(repo):
    path = os.path.join(repo, "profiles.json")
    out = {}
    if not os.path.exists(path):
        return out
    for profile in read_json(path):
        out[profile["hf_id"]] = profile
    return out


def add_score(scores, hf_id, name, value, split, prompt, source, n=None, seeds=None):
    if hf_id not in scores:
        scores[hf_id] = {}
    if name in scores[hf_id]:
        return
    row = {"value": pct(value), "split": split, "prompt": prompt, "source": source}
    if n is not None:
        row["n"] = n
    if seeds is not None:
        row["seeds"] = seeds
    scores[hf_id][name] = row


def collect_scores(repo):
    """Measured scores of models with no training, from the results files."""
    scores = {}
    results = os.path.join(repo, "results")

    path = os.path.join(results, "round10_rows.json")
    if os.path.exists(path):
        for model in read_json(path):
            summary = model["summary"]
            names = [("mmlu_dev", MMLU_DEV), ("mmlu_heldout", MMLU_HELDOUT), ("medmcqa", MEDMCQA)]
            for name, split in names:
                if name in summary:
                    right, count = summary[name]
                    add_score(scores, model["model"], name, right / count, split, MCQ_PROMPT,
                              "results/round10_rows.json", n=count)

    path = os.path.join(results, "round15_rows.json")
    if os.path.exists(path):
        for row in read_json(path):
            if row.get("kind") != "official":
                continue
            row_scores = row["scores"]
            splits = {"mmlu_dev": MMLU_DEV, "mmlu_heldout": MMLU_HELDOUT, "medmcqa": MEDMCQA}
            for name in ["mmlu_dev", "mmlu_heldout", "medmcqa"]:
                if name in row_scores:
                    add_score(scores, row["model"], name + "_round15", row_scores[name], splits[name], MCQ_PROMPT,
                              "results/round15_rows.json")
            if "gsm8k_raw" in row_scores:
                add_score(scores, row["model"], "gsm8k_search_raw", row_scores["gsm8k_raw"], GSM8K_SEARCH,
                          RAW_PROMPT, "results/round15_rows.json")

    path = os.path.join(results, "round8_rows.json")
    if os.path.exists(path):
        for row in read_json(path):
            if row["kind"] in ["donor_alone", "host_alone"] and row["key"] in RESULT_KEYS:
                add_score(scores, RESULT_KEYS[row["key"]], "mmlu_dev_round8", row["accuracy"], MMLU_DEV, MCQ_PROMPT,
                          "results/round8_rows.json")

    path = os.path.join(results, "round7_rows.json")
    if os.path.exists(path):
        for row in read_json(path):
            cfg = row.get("cfg") or {}
            if row["part"] == "heldout" and row["arm"] == "host":
                add_score(scores, "unsloth/Llama-3.2-1B-Instruct", "mmlu_heldout_round7", row["accuracy"],
                          MMLU_HELDOUT, MCQ_PROMPT, "results/round7_rows.json")
            if row["part"] == "heldout" and row["arm"] == "donor":
                add_score(scores, cfg["donor"], "mmlu_heldout_round7", row["accuracy"], MMLU_HELDOUT, MCQ_PROMPT,
                          "results/round7_rows.json")
            if row["part"] == "scaling" and row["arm"] == "donor":
                add_score(scores, cfg["donor"], "mmlu_dev_round7", row["accuracy"], MMLU_DEV, MCQ_PROMPT,
                          "results/round7_rows.json")
            if row["part"] == "trivia" and row["arm"] in ["host", "donor"]:
                hf_id = "unsloth/Llama-3.2-1B-Instruct"
                if row["arm"] == "donor":
                    hf_id = cfg["donor"]
                add_score(scores, hf_id, "triviaqa", row["accuracy"],
                          "TriviaQA validation, 1000 questions", "alias match on a 16-token greedy answer",
                          "results/round7_rows.json")

    path = os.path.join(results, "baselines_rows.json")
    if os.path.exists(path):
        for row in read_json(path)["summary"]:
            if row["arm"].startswith("ceiling "):
                hf_id = row["arm"][len("ceiling "):]
                add_score(scores, hf_id, "gsm8k_search_raw", row["score_mean"], GSM8K_SEARCH,
                          RAW_PROMPT + " (wsp_eval harness)", "results/baselines_rows.json")

    path = os.path.join(results, "round5_rows.json")
    if os.path.exists(path):
        data = read_json(path)
        for row in data["rows"]:
            hf_id = None
            if row["arm"] == "host_only":
                hf_id = data["host"]
            if row["arm"] == "donor_alone":
                hf_id = data["donor"]
            if row["arm"] == "lfm_alone":
                hf_id = "LiquidAI/LFM2-350M-Math"
            if hf_id is not None:
                add_score(scores, hf_id, "gsm8k_search_raw", row["accuracy"], GSM8K_SEARCH, RAW_PROMPT,
                          "results/round5_rows.json")

    path = os.path.join(results, "round4_rows.json")
    if os.path.exists(path):
        for row in read_json(path)["rows"]:
            if row["arm"] == "host" and row.get("host") in RESULT_KEYS:
                add_score(scores, RESULT_KEYS[row["host"]], "gsm8k_search_raw", row["accuracy"], GSM8K_SEARCH,
                          RAW_PROMPT, "results/round4_rows.json")

    path = os.path.join(results, "gate_hour8_3seeds.json")
    if os.path.exists(path):
        for row in read_json(path)["rows"]:
            if row["arm"] in ["host", "donor"]:
                add_score(scores, row["model"], "gsm8k_search_harness", row["score"], GSM8K_SEARCH,
                          "wsp_eval harness GSM8K prompt", "results/gate_hour8_3seeds.json", n=row.get("n"))

    path = os.path.join(results, "base_scores_heldout.json")
    if os.path.exists(path):
        heldout = read_json(path)["heldout"]
        host = "unsloth/Llama-3.2-1B-Instruct"
        if "gsm8k" in heldout:
            add_score(scores, host, "gsm8k_heldout_harness", heldout["gsm8k"], GSM8K_HELDOUT,
                      "wsp_eval harness GSM8K prompt", "results/base_scores_heldout.json")
        if "if_guard" in heldout:
            add_score(scores, host, "if_guard_heldout", heldout["if_guard"], "wsp_eval if_guard held-out, 410 items",
                      "wsp_eval harness", "results/base_scores_heldout.json")
    return scores


# Structure notes by model: (note, source). Numbers come from results/README.md.
STRUCTURE_NOTES = {
    "Qwen/Qwen2.5-7B-Instruct": [
        ("knowledge readable near layer 20 of 28 by taps (MMLU dev 67.1 to 68.5 at host layers 6 to 14); "
         "nothing before layer 16 (48 to 49 from layers 8 and 12): run 7 layer map", "results/round7_rows.json"),
        ("tap read at layer 27 into Llama-1B layer 12: 65.6 MMLU dev, 3 seeds (run 6)", "results/round6_mmlu_rows.json"),
    ],
    "Qwen/Qwen2.5-14B-Instruct": [
        ("tap read at layer 34 of 48: Qwen-0.5B host 74.8, Llama-1B host 70.5 MMLU dev (run 8)",
         "results/round8_rows.json"),
        ("as a host (write layer 19) it gains nothing from equal or weaker donors: best tap 77.9 vs 78.5 alone (run 9)",
         "results/round9_rows.json"),
        ("carved to 22 of 48 layers it falls to 33.6 MMLU dev at best after a LoRA heal (run 15)",
         "results/round15_rows.json"),
    ],
    "Qwen/Qwen2.5-32B-Instruct": [
        ("carved to 27 of 64 layers it falls to 40.3 and 41.2 MMLU dev after a LoRA heal (run 15)",
         "results/round15_rows.json"),
    ],
    "unsloth/gemma-2-9b-it": [
        ("tap read at layer 29 of 42 reaches the donor: Llama-1B host 72.9, Qwen-0.5B host 72.7 vs 72.6 alone "
         "(run 8)", "results/round8_rows.json"),
        ("loaded with attn_implementation eager in modal_round8.py", "modal_round8.py"),
    ],
    "microsoft/Phi-3.5-mini-instruct": [
        ("tap read at layer 22 of 32: 66.3 MMLU dev vs 68.5 alone (run 8)", "results/round8_rows.json"),
    ],
    "unsloth/Meta-Llama-3.1-8B-Instruct": [
        ("tap read at layer 22 of 32 is unstable: seeds 49.2 and 58.0 vs 66.6 alone (run 8)",
         "results/round8_rows.json"),
    ],
    "LiquidAI/LFM2.5-1.2B-Instruct": [
        ("tap read at layer 11 of 16 gives 49.9 MMLU dev (host 47.5); alone 26.2 on the raw prompt (run 8)",
         "results/round8_rows.json"),
    ],
    "Qwen/Qwen2.5-1.5B-Instruct": [
        ("tap read at layer 20 of 28: 58.2 MMLU dev vs 58.9 alone (run 7)", "results/round7_rows.json"),
    ],
    "Qwen/Qwen2.5-3B-Instruct": [
        ("tap read at layer 25 of 36: 57.9 MMLU dev vs 66.2 alone, likely a poor read layer (run 7)",
         "results/round7_rows.json"),
    ],
    "unsloth/Llama-3.2-1B-Instruct": [
        ("tap write layer 10 of 16 in runs 6 to 8; for donor layer 20, host layer 6 scored 68.5 and layer 10 "
         "67.7 (run 7 layer map)", "results/round7_rows.json"),
        ("graft placement: host layer 6 gives about 38 to 40% on GSM8K search, layers 10 and 14 about 22 to 26%, "
         "whatever the part (run 2)", "results/sweep2_rows.json"),
        ("the 37 Deletion Tuning masks in catalog.jsonl are built on this model", "catalog.jsonl"),
    ],
    "Qwen/Qwen2.5-0.5B-Instruct": [
        ("tap write layer 15 of 24 in run 8 (Qwen-14B tap 74.8 MMLU dev)", "results/round8_rows.json"),
    ],
    "Qwen/Qwen2.5-Math-1.5B": [
        ("no BOS token: the first text token is the attention sink and carries massive, input-invariant "
         "activations; feed the donor's mean sink state at position 0 (audit)", "results/audit_sink_rows.json"),
        ("range-fetched tensors equal the checkpoint; a block part of layers 8..19 equals the donor layers "
         "(max diff 4e-5) (audit)", "results/README.md"),
        ("a donor_lm stitch into layer 8 leaves 0.03 to 0.09 nats of penalty from same-tokenizer hosts "
         "(stitch study)", "results/stitch_study_rows.json"),
        ("single-layer 512-neuron GSM8K clusters at layers 10, 18, 24 do not beat random (run 2)",
         "results/sweep2_rows.json"),
    ],
    "LiquidAI/LFM2-350M-Math": [
        ("tap read at layer 8 of 16 into Llama-1B layer 8: 32.7 GSM8K search vs host 25.0, the only tap above "
         "noise in run 5; needs its chat template (1.0 alone on the raw prompt)", "results/round5_rows.json"),
    ],
    "microsoft/phi-4": [
        ("GSM8K-contrast MLP cluster at layer 16 grafted at Llama-1B layer 6: +2.8 over random on the search split, "
         "-0.6 on held-out GSM8K (Person B part gate)", "results/final_heldout_microsoft__phi-4.json"),
    ],
    "Nanbeige/Nanbeige4.3-33B": [
        ("GSM8K-contrast MLP cluster at layer 36 grafted at Llama-1B layer 11: -1.8 vs random on the search split",
         "results/part_gate_Nanbeige__Nanbeige4.3-33B.json"),
    ],
}


def model_hf_ids(repo, profiles, registry, zoo, donors_md, scores):
    """Every model the library profiles, in a fixed order."""
    out = []

    def add(hf_id):
        if hf_id and hf_id not in out:
            out.append(hf_id)

    for hf_id in profiles:
        add(hf_id)
    for hf_id in scores:
        add(hf_id)
    for hf_id in registry:
        add(hf_id)
    for hf_id in donors_md:
        add(hf_id)
    for hf_id in zoo:
        if zoo[hf_id].get("moe"):
            add(hf_id)
    # Models named by components, stitches, and recipes.
    for hf_id in ["Qwen/Qwen2.5-Math-1.5B", "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", "Nanbeige/Nanbeige4.3-33B",
                  "microsoft/phi-4", "Qwen/Qwen2.5-1.5B"]:
        add(hf_id)
    return out


def build_models(repo):
    profiles = read_profiles(repo)
    registry = read_registry(repo)
    zoo = read_zoo(repo)
    donors_md = parse_donors_md(repo)
    scores = collect_scores(repo)
    entries = []
    for hf_id in model_hf_ids(repo, profiles, registry, zoo, donors_md, scores):
        profile = profiles.get(hf_id) or {}
        reg = registry.get(hf_id) or {}
        config = reg.get("config") or {}
        zoo_row = zoo.get(hf_id) or {}
        md = donors_md.get(hf_id) or {}
        sources = []
        if profile:
            sources.append("profiles.json")
        if config:
            sources.append(reg["config_path"])
        if zoo_row:
            sources.append("atlas/zoo.json")
        if md:
            sources.append("docs/donors.md")

        family = profile.get("family") or config.get("model_type") or zoo_row.get("family") or md.get("type")
        params_b = profile.get("params_b") or zoo_row.get("params_b")
        if params_b is None and reg.get("params_b"):
            params_b = float(reg["params_b"])
        layers = profile.get("layers") or config.get("num_hidden_layers") or zoo_row.get("layers")
        if layers is None and md:
            layers = int_or_none(md["layers"])
        hidden = profile.get("hidden") or config.get("hidden_size") or zoo_row.get("hidden")
        if hidden is None and md:
            hidden = int_or_none(md["hidden"])
        tokenizer_family = profile.get("tokenizer_family")
        vocab = None
        vocab_source = None
        if config.get("vocab_size"):
            vocab = config["vocab_size"]
            vocab_source = reg["config_path"]
        elif md and int_or_none(md["vocab"]):
            vocab = int_or_none(md["vocab"])
            vocab_source = "docs/donors.md"
        elif tokenizer_family in VOCAB_BY_TOKENIZER:
            vocab = VOCAB_BY_TOKENIZER[tokenizer_family]
            vocab_source = "composer/sizes.py VOCAB by tokenizer family"
        license_note = zoo_row.get("license") or reg.get("license")
        if license_note is None and family in ["lfm2", "lfm2_moe", "lfm3"]:
            license_note = "LFM License: free for commercial use under $10M annual revenue (docs/donors.md)"
        experts = config.get("num_experts") or config.get("n_routed_experts") or config.get("num_local_experts")
        if not experts:
            experts = zoo_row.get("experts")
        top_k = config.get("num_experts_per_tok") or None
        needs = zoo_row.get("needs")
        if needs is None and md:
            needs = {"transformers": md["transformers"]}

        notes = []
        for note, source in STRUCTURE_NOTES.get(hf_id, []):
            notes.append({"note": note, "source": source})
        if experts:
            text = "MoE: " + str(experts) + " experts per layer"
            if top_k:
                text = text + ", top " + str(top_k)
            if reg:
                notes.append({"note": text, "source": reg["config_path"]})
            else:
                notes.append({"note": text, "source": "atlas/zoo.json"})
        if md:
            notes.append({"note": "docs/donors.md: type " + md["type"] + ", layers (conv / attn) " + md["layers"] +
                                  ", needs transformers " + md["transformers"], "source": "docs/donors.md"})
            if md["type"] == "lfm2":
                notes.append({"note": "conv layers: block parts with conv layers need the conv state cache, which "
                                      "the part cache does not handle yet; taps work", "source": "docs/donors.md"})
            if md["type"] == "lfm2_moe":
                notes.append({"note": "32 small experts per layer are good expert parts but need transformers 5.x",
                              "source": "docs/donors.md"})
        if config.get("layer_types"):
            kinds = []
            for layer_type in config["layer_types"]:
                if layer_type not in kinds:
                    kinds.append(layer_type)
            notes.append({"note": "layer types: " + ", ".join(kinds), "source": reg["config_path"]})
        if reg.get("role"):
            notes.append({"note": "mining registry role " + reg["role"] + ", status " + reg["status"],
                          "source": "person_a/registry/donors.csv"})

        model_scores = scores.get(hf_id, {})
        status = "candidate"
        if len(model_scores) > 0:
            status = "validated"
        # The scores are the evidence for a model; the evidence list is for hand-added items.
        items = []
        skills = []
        if zoo_row.get("skills"):
            for skill in zoo_row["skills"]:
                skills.append(skill)
        title = short_model(hf_id)
        if params_b:
            title = title + " (" + str(round(float(params_b), 2)) + "B)"
        fields = {"hf_id": hf_id, "family": family, "params_b": params_b,
                  "active_params_b": profile.get("active_params_b"), "layers": layers, "hidden": hidden,
                  "vocab": vocab, "vocab_source": vocab_source, "tokenizer_family": tokenizer_family,
                  "license": license_note, "experts": experts, "experts_per_token": top_k, "needs": needs,
                  "scores": model_scores, "structure_notes": notes, "components": [], "arch_sources": sources}
        source = ", ".join(sources)
        if source == "":
            source = "results files"
        entries.append(entry(model_key(hf_id), "model", title, status, TRISANTH, source, skills, [hf_id], items,
                             fields))
    return entries


# Components


def mask_skills(mask_id):
    name = mask_id
    if name.startswith("mask_"):
        name = name[len("mask_"):]
    if name.endswith("_llama1b"):
        name = name[:-len("_llama1b")]
    if name.startswith("mmlupro_"):
        return ["mmlu_pro", name[len("mmlupro_"):]]
    if name.startswith("bbeh_"):
        return ["bbeh", name[len("bbeh_"):]]
    return [name]


def build_masks(repo):
    out = []
    path = os.path.join(repo, "catalog.jsonl")
    if not os.path.exists(path):
        return out
    line_number = 0
    for record in read_jsonl(path):
        line_number += 1
        if record.get("type") != "deletion_mask":
            continue
        signature = record.get("signature") or {}
        overlap = {}
        for other in signature.get("overlap_with", {}):
            overlap[other] = round(signature["overlap_with"][other]["jaccard"], 4)
        layers = []
        per_layer = signature.get("deleted_per_layer") or []
        for index in range(len(per_layer)):
            layers.append(index)
        sig = {"method": "Deletion Tuning: a learned KEEP mask over the base weights, trained on " +
                         str(record.get("trained_on")),
               "num_deleted": signature.get("num_deleted"), "deleted_per_layer": per_layer,
               "jaccard_with": overlap, "top_overlaps": signature.get("top_overlaps"),
               "structure_updated": signature.get("structure_updated"),
               "note": "per-layer Jaccard overlaps are in catalog.jsonl; heatmaps in results/heatmaps/",
               "signature_by": PERSON_B + " (structural signatures filled by wsp_eval, git log of catalog.jsonl)"}
        title = "Deletion mask, " + str(record.get("trained_on")) + ", on " + short_model(record["base"])
        out.append(component(record["id"], "deletion_mask", title, "candidate", TRISANTH,
                             "catalog.jsonl line " + str(line_number) + " (" + record.get("source", "") + ")",
                             mask_skills(record["id"]), record["base"], layers, None, record.get("path"), sig, [],
                             {"catalog_id": record["id"], "family": record.get("family"),
                              "trained_on": record.get("trained_on"),
                              "status_reason": "no graft or mask-algebra run in results/ measures it yet"}))
    return out


def build_round2(repo):
    """Trisanth's round 2 clusters (sweep2_rows.json), each against the random control."""
    out = []
    path = os.path.join(repo, "results", "sweep2_rows.json")
    if not os.path.exists(path):
        return out
    data = read_json(path)
    cells = {}
    for row in data["rows"]:
        key = (row["layer"], row["selection"], str(row["rank"]))
        if key not in cells:
            cells[key] = []
        cells[key].append(row)
    for record in data["records"]:
        layer = record["layer"]
        method = record["id"].split("_")[2]
        selection = method
        items = []
        fit = None
        host_layer = None
        part_params = None
        # sweep2_rows.json writes the glue rank as "full" or "r64".
        for rank in ["full", "r64"]:
            graft_rows = cells.get((layer, selection, rank), [])
            random_rows = cells.get((layer, "random", rank), [])
            if len(graft_rows) == 0 or len(random_rows) == 0:
                continue
            graft_scores = []
            seeds = []
            for row in graft_rows:
                graft_scores.append(row["accuracy"])
                seeds.append(row["seed"])
                fit = [row["fit_r2"], row["fit_r2_dim"]]
                host_layer = row["host_layer"]
                part_params = row["part_params"]
            random_scores = []
            for row in random_rows:
                random_scores.append(row["accuracy"])
            glue = "full-rank glue"
            if rank != "full":
                glue = "rank-" + rank[1:] + " glue"
            items.append(evidence("graft at host layer " + str(host_layer) + " with " + glue +
                                  " vs a random 512-neuron cluster from the same layer",
                                  "GSM8K search accuracy, mean over seeds", pct(mean(graft_scores)),
                                  pct(mean(random_scores)), GSM8K_SEARCH, sorted(seeds), "results/sweep2_rows.json",
                                  control="random same-size cluster, same layer, same glue",
                                  note="host unsloth/Llama-3.2-1B-Instruct, 300 heal steps"))
        sig = {"method": selection + " (512 MLP neurons; activation = mean |activation|, contrast = GSM8K text "
                                     "minus WikiText, attribution = estimated loss increase when zeroed)",
               "data": record.get("trained_on"), "n_units": 512, "fit_r2_pooled_and_per_dim": fit,
               "part_params": part_params, "extract_seconds": record.get("extract_seconds")}
        out.append(component(record["id"], "mlp_cluster",
                             "Qwen2.5-Math-1.5B layer " + str(layer) + ", 512 MLP neurons by " + selection,
                             "rejected", TRISANTH, "results/sweep2_rows.json (run 2)", ["math", "gsm8k"],
                             record["model"], [layer], None, record.get("units"), sig, items,
                             {"status_reason": "run 2: no selection method beats the random control at any layer "
                                               "(3 seeds, search split; gaps within noise)"}))
    return out


def build_run1(repo):
    path = os.path.join(repo, "results", "graft_run1_s0.json")
    if not os.path.exists(path):
        return []
    data = read_json(path)
    items = [evidence("graft at host layer " + str(data["graft"]["host_layer"]) + " with full-rank glue vs a random "
                      "512-neuron control", "GSM8K search accuracy", pct(data["graft"]["accuracy"]),
                      pct(data["control"]["accuracy"]), GSM8K_SEARCH, [data["seed"]], "results/graft_run1_s0.json",
                      control="random same-size cluster, same layer",
                      note="host 25.0 untrained; the control's gain comes from the heal format and 6.3M glue params")]
    sig = {"method": "top 512 MLP neurons by mean |activation| on GSM8K train text", "n_units": 512,
           "fit_r2_pooled": data["graft"]["fit_r2"],
           "fit_note": "the 0.987 R squared was inflated by first-token outlier rows (results/README.md run 1)",
           "part_params": data["graft"]["part_params"], "glue_params": data["graft"]["glue_params"]}
    return [component("run1_qmath_l18_top512", "mlp_cluster", "Qwen2.5-Math-1.5B layer 18, top 512 MLP neurons by "
                      "mean |activation| (run 1)", "rejected", TRISANTH, "results/graft_run1_s0.json (run 1)",
                      ["math", "gsm8k"], data["donor"], [data["donor_layer"]], None, None, sig, items,
                      {"status_reason": "run 1: 25.0 vs the random control 33.0 (1 seed, search split)"})]


def gate_items(summary, source, split, note=None):
    """Evidence items from a wsp_eval gate verdict (graft vs random, and vs LoRA when present)."""
    items = []
    graft = summary["graft"]
    control = summary["main_control"]
    seeds = []
    for seed in summary.get("paired_seeds", []):
        seeds.append(seed)
    items.append(evidence("graft vs a random same-size cluster through the same glue and heal",
                          "GSM8K accuracy, mean over seeds", pct(graft["mean"]), pct(control["mean"]), split, seeds,
                          source, control="random same-size cluster, same layer", note=note))
    if summary.get("lora_control"):
        items.append(evidence("graft vs a host LoRA matched to the glue and part params",
                              "GSM8K accuracy, mean over seeds", pct(graft["mean"]),
                              pct(summary["lora_control"]["mean"]), split, seeds, source,
                              control="matched LoRA"))
    return items


def build_gate_part(repo):
    path = os.path.join(repo, "results", "gate_hour8_3seeds.json")
    if not os.path.exists(path):
        return []
    data = read_json(path)
    part = data["part"]
    signature = dict(part.get("signature") or {})
    if "unit_effects" in signature:
        signature["unit_effects"] = signature["unit_effects"][:16]
        signature["unit_effects_note"] = "first 16 of " + str(signature.get("n_units")) + " kept here"
    items = gate_items(data["verdict"], "results/gate_hour8_3seeds.json", GSM8K_SEARCH,
                       note="host unsloth/Llama-3.2-1B-Instruct, rank-64 glue, 300 heal steps")
    return [component("gate_" + part["id"], "mlp_cluster",
                      "Qwen2.5-Math-1.5B layer 10, 512 MLP neurons by Cohen's d contrast (hour-8 gate)", "rejected",
                      PERSON_B, "results/gate_hour8_3seeds.json", ["math", "gsm8k"], part["model"], [part["layer"]], None,
                      part.get("units"), signature, items,
                      {"original_id": part["id"],
                       "status_reason": "hour-8 gate: +0.7 over random, lost or tied on 2 of 3 seeds; LoRA 44.5 "
                                        "beats it (search split)"})]


def person_b_gates(repo):
    """Part gate and held-out verdicts by part id."""
    out = {}
    for path in sorted(glob.glob(os.path.join(repo, "results", "part_gate_*.json"))):
        data = read_json(path)
        rel = "results/" + os.path.basename(path)
        if data["part"] not in out:
            out[data["part"]] = []
        out[data["part"]].append(("search", data["verdict"], rel, data))
    for path in sorted(glob.glob(os.path.join(repo, "results", "final_heldout_*.json"))):
        data = read_json(path)
        rel = "results/" + os.path.basename(path)
        if data["part"] not in out:
            out[data["part"]] = []
        out[data["part"]].append(("heldout", data["verdict"], rel, data))
    return out


def host_layer_of(data):
    for row in data.get("rows", []):
        for fit in row.get("fit_r2") or []:
            if isinstance(fit, dict) and fit.get("host_layer") is not None:
                return fit["host_layer"]
    return None


def build_person_b(repo):
    """Rex's parts in parts_extracted/, with their random controls copied next to them."""
    out = []
    gates = person_b_gates(repo)
    for records_path in sorted(glob.glob(os.path.join(repo, "parts_extracted", "*", "records.jsonl"))):
        for record in read_jsonl(records_path):
            folder = "components/" + model_key(record["model"]) + "/"
            copies = []
            units_file = None
            src = os.path.join(repo, record["units"])
            missing = {}
            if os.path.exists(src):
                units_file = folder + os.path.basename(record["units"])
                copies.append((src, units_file))
            else:
                missing = {"units_missing": True, "units_expected": record["units"]}
            controls = []
            for control in record.get("controls") or []:
                control_src = os.path.join(repo, control)
                if os.path.exists(control_src):
                    rel = folder + os.path.basename(control)
                    copies.append((control_src, rel))
                    controls.append(rel)
            items = []
            status = "candidate"
            reason = "not grafted and scored yet"
            for split_name, verdict, rel, data in gates.get(record["id"], []):
                host_layer = host_layer_of(data)
                note = "host unsloth/Llama-3.2-1B-Instruct"
                if host_layer is not None:
                    note = note + ", host layer " + str(host_layer)
                note = note + ", rank-64 glue, 300 heal steps"
                split = GSM8K_SEARCH
                if split_name == "heldout":
                    split = GSM8K_HELDOUT
                for item in gate_items(verdict, rel, split, note=note):
                    items.append(item)
                if verdict.get("beats_main_control"):
                    if split_name == "heldout":
                        status = "validated"
                        reason = "beats the random control on held-out GSM8K"
                    elif status == "candidate":
                        reason = "beats the random control on the search split only"
                else:
                    status = "rejected"
                    reason = split_name + " gate: " + "; ".join(verdict.get("why_not") or [])
            signature = dict(record.get("signature") or {})
            if "unit_effects" in signature:
                signature["unit_effects"] = signature["unit_effects"][:16]
                signature["unit_effects_note"] = "first 16 of " + str(signature.get("n_units")) + " kept here"
            signature["selection_note"] = ("contrast effect is a selection statistic, not evidence of function "
                                           "(parts_extracted/README.md)")
            kind = record["type"]
            what = "MLP neurons"
            if kind == "attn_heads":
                what = "attention heads"
            title = (short_model(record["model"]) + " layer " + str(record["layer"]) + ", " +
                     str(signature.get("n_units")) + " " + what + " by GSM8K contrast")
            data = component(record["id"], kind, title, status, PERSON_B,
                             "parts_extracted/" + os.path.basename(os.path.dirname(records_path)) + "/records.jsonl",
                             ["math", "gsm8k"], record["model"], [record["layer"]], units_file,
                             "/vol/eval/extracted/" + record["units"].split("/", 1)[1], signature, items,
                             dict({"controls": controls, "catalog_id": record["id"], "family": record.get("family"),
                                   "status_reason": reason}, **missing))
            out.append((data, copies))
    return out


def person_a_items(record, rejected):
    """Ablation evidence from Person A's mining gate."""
    items = []
    signature = record.get("signature") or {}
    effect = signature.get("effect") or {}
    source = record["_source"]
    if "zero" in effect:
        zero = effect["zero"]
        checks = signature.get("checks") or {}
        note = "zero ablation in the donor; controls trivia " + str(zero.get("trivia")) + ", wiki " + \
               str(zero.get("wiki")) + "; random percentile " + str(signature.get("random_percentile"))
        if checks:
            passed = []
            for name in checks:
                if checks[name]:
                    passed.append(name)
            note = note + "; checks passed: " + ", ".join(passed)
        items.append(evidence("zeroing the part drops GSM8K generation accuracy in the donor more than random "
                              "same-size parts", "relative drop in GSM8K generation accuracy",
                              zero.get("gsm8k_gen_heldout"), None, "GSM8K held-out ablation items (person_a mining)",
                              None, source, control="random same-size parts", note=note))
    skill = (record.get("skill") or {}).get("name")
    if skill is not None:
        drops = signature.get("random_drops") or []
        control_value = None
        if len(drops) > 0:
            control_value = round(mean(drops), 4)
        note = "ablation screen in the donor"
        matched = effect.get(skill + "__control|heldout")
        if matched is not None:
            note = note + "; matched control task changed by " + str(matched)
        items.append(evidence("zeroing the part drops the '" + skill + "' skill on held-out items more than random "
                              "same-size parts", "relative drop in skill score", effect.get(skill + "|heldout"),
                              control_value, "person_a held-out skill items (" + skill + ")", None, source,
                              control="mean drop of 5 random same-size parts", note=note))
    return items


def build_person_a(repo):
    """Hanav's parts: person_a/catalog.jsonl (v1), person_a/catalog/*.jsonl (v2), and the rejected lists."""
    out = []
    sources = []
    v1 = os.path.join(repo, "person_a", "catalog.jsonl")
    if os.path.exists(v1):
        for record in read_jsonl(v1):
            record["_source"] = "person_a/catalog.jsonl"
            record["_pipeline"] = "v1-dissect"
            sources.append(record)
    for path in sorted(glob.glob(os.path.join(repo, "person_a", "catalog", "*.jsonl"))):
        for record in read_jsonl(path):
            record["_source"] = "person_a/catalog/" + os.path.basename(path)
            record["_pipeline"] = "v2-mine"
            record["_rejected"] = path.endswith(".rejected.jsonl")
            sources.append(record)
    seen = []
    for record in sources:
        if record["id"] in seen:
            continue
        seen.append(record["id"])
        rejected = record.get("_rejected", False)
        folder = "components/" + model_key(record["model"]) + "/"
        copies = []
        units_file = None
        src = os.path.join(repo, "person_a", record["units"])
        missing = {}
        if os.path.exists(src):
            units_file = folder + os.path.basename(record["units"])
            copies.append((src, units_file))
        else:
            missing = {"units_missing": True, "units_expected": "person_a/" + record["units"]}
        layers = record.get("layers") or [record["layer"]]
        skill = (record.get("skill") or {}).get("name")
        skills = ["math"]
        if skill:
            skills.append(skill)
        else:
            skills.append("gsm8k")
        status = "candidate"
        reason = "ablation evidence in the donor only; not grafted against a random control yet"
        if rejected:
            status = "rejected"
            reason = "; ".join(record.get("rejected_reasons") or ["rejected by the mining gate"])
        signature = {"method": record.get("trained_on"), "label": record.get("label"),
                     "num_units": record.get("num_units"), "d_model": record.get("d_model"),
                     "effect": (record.get("signature") or {}).get("effect"),
                     "gate": (record.get("signature") or {}).get("gate"),
                     "random_drops": (record.get("signature") or {}).get("random_drops"),
                     "random_percentile": (record.get("signature") or {}).get("random_percentile"),
                     "checks": (record.get("signature") or {}).get("checks"),
                     "weight_rationale": record.get("weight_rationale")}
        exported = None
        short = record["units"].split("/")
        if len(short) >= 3:
            candidate_path = "library/" + short[1] + "/" + short[2]
            if os.path.exists(os.path.join(repo, candidate_path)):
                exported = candidate_path
        kind = record["type"]
        title = short_model(record["model"]) + " layer " + str(layers[0]) + ", " + str(record.get("label"))
        data = component(record["id"], kind, title, status, HANAV, record["_source"], skills, record["model"],
                         layers, units_file, "person_a/" + record["units"], signature,
                         person_a_items(record, rejected),
                         dict({"tier": record.get("tier") or "provisional", "pipeline": record["_pipeline"],
                               "skill": record.get("skill"), "catalog_id": record["id"],
                               "exported_copy": exported, "status_reason": reason}, **missing))
        out.append((data, copies))
    return out


def build_pilots(repo):
    out = []
    files = [("data/pilot_catalog.jsonl", "scripts/build_pilot_catalog.py"),
             ("data/pilot_catalog_mined.jsonl", "scripts/mine_pilot.py")]
    for rel, script in files:
        path = os.path.join(repo, rel)
        if not os.path.exists(path):
            continue
        for record in read_jsonl(path):
            signature = dict(record.get("signature") or {})
            signature["data"] = record.get("trained_on")
            if "controls" in record:
                signature["controls_remote"] = record["controls"]
            title = (short_model(record["model"]) + " layer " + str(record["layer"]) + ", pilot " +
                     record["type"].replace("_", " "))
            out.append(component(record["id"], record["type"], title, "candidate", REX_PILOT, rel + " (" + script + ")",
                                 ["math", "gsm8k"], record["model"], [record["layer"]], None, record.get("units"),
                                 signature, [],
                                 {"status_reason": "pilot set for the swarm; units on the Modal volume; not scored",
                                  "catalog_id": record["id"]}))
    return out


def tap_read_block(block_id, hf_id, layer, title, status, items, reason, skills, source):
    layers = []
    for index in range(layer + 1):
        layers.append(index)
    return component(block_id, "block", title, status, TRISANTH, source, skills, hf_id, layers, None, None,
                     {"method": "tap read point: the donor runs layers 0.." + str(layer) + " on the prompt and a "
                                "gated cross-attention branch in the host reads layer " + str(layer),
                      "read_layer": layer},
                     items, {"status_reason": reason, "use": "tap " + hf_id + " layer " + str(layer) + " at <host layer>"})


def build_blocks(repo):
    out = []
    results = os.path.join(repo, "results")

    # Qwen2.5-7B knowledge block from the run 7 layer map.
    path = os.path.join(results, "round7_rows.json")
    if os.path.exists(path):
        rows = read_json(path)
        layer_map = {}
        heldout_tap = []
        heldout_seeds = []
        heldout_host = None
        heldout_lora = []
        for row in rows:
            cfg = row.get("cfg") or {}
            if row["part"] == "layers" and row["arm"] == "tap":
                layer_map[str(cfg["donor_layer"]) + "->" + str(cfg["host_layer"])] = row["accuracy"]
            if row["part"] == "heldout" and row["arm"] == "tap":
                heldout_tap.append(row["accuracy"])
                heldout_seeds.append(row["seed"])
            if row["part"] == "heldout" and row["arm"] == "host":
                heldout_host = row["accuracy"]
            if row["part"] == "heldout" and row["arm"] == "lora_r16":
                heldout_lora.append(row["accuracy"])
        items = []
        best_16_20 = None
        best_8_12 = None
        for key in layer_map:
            donor_layer = int(key.split("->")[0])
            value = layer_map[key]
            if donor_layer >= 16 and donor_layer <= 20:
                if best_16_20 is None or value > best_16_20:
                    best_16_20 = value
            if donor_layer <= 12:
                if best_8_12 is None or value > best_8_12:
                    best_8_12 = value
        if best_16_20 is not None:
            items.append(evidence("a tap reading donor layers 16 to 20 vs a tap reading layers 8 or 12",
                                  "MMLU dev accuracy, best host layer", pct(best_16_20), pct(best_8_12), MMLU_DEV, [0],
                                  "results/round7_rows.json", control="same tap reading an earlier layer",
                                  note="layer map: " + json.dumps(layer_map, sort_keys=True)))
        if len(heldout_tap) > 0:
            items.append(evidence("Llama-1B + tap reading layer 20 at host layer 10 vs a rank-16 LoRA",
                                  "MMLU held-out accuracy, mean over seeds", pct(mean(heldout_tap)),
                                  pct(mean(heldout_lora)), MMLU_HELDOUT, sorted(heldout_seeds),
                                  "results/round7_rows.json", control="host LoRA rank 16, same data",
                                  note="host alone " + str(heldout_host) + "; donor alone 0.7085"))
        path6 = os.path.join(results, "round6_mmlu_rows.json")
        if os.path.exists(path6):
            real = []
            fake = []
            for row in read_json(path6):
                if row["arm"] == "tap_l27_h12":
                    real.append(row["accuracy"])
                if row["arm"] == "tap_l27_derange":
                    fake.append(row["accuracy"])
            if real and fake:
                items.append(evidence("a tap reading layer 27 vs the same tap reading mismatched questions",
                                      "MMLU dev accuracy, mean over seeds", pct(mean(real)), pct(mean(fake)), MMLU_DEV,
                                      None, "results/round6_mmlu_rows.json", control="mismatched-message control"))
        layers = [16, 17, 18, 19, 20]
        out.append(component("block_qwen25_7b_l16_20_knowledge", "block",
                             "Qwen2.5-7B-Instruct layers 16 to 20, the knowledge block read by taps", "candidate",
                             TRISANTH, "results/round7_rows.json (run 7 layer map)", ["knowledge", "mmlu"],
                             "Qwen/Qwen2.5-7B-Instruct", layers, None, None,
                             {"method": "run 7 layer map: tap read layer vs MMLU accuracy", "data": "MMLU dev items",
                              "layer_map": layer_map},
                             items,
                             {"catalog_id": "block/Qwen/Qwen2.5-7B-Instruct/16-20",
                              "status_reason": "taps read it on held-out MMLU, but no in-stream block graft of these "
                                               "layers has been scored against a random same-length block"}))

    # Tap read points from runs 7 and 8.
    path = os.path.join(results, "round8_rows.json")
    if os.path.exists(path):
        rows = read_json(path)
        by_kind = {}
        for row in rows:
            key = row["kind"] + ":" + row["key"]
            if key not in by_kind:
                by_kind[key] = []
            by_kind[key].append(row)
        host_scores = {"tap_llama1b": None, "tap_qwen05": None}
        for row in by_kind.get("host_alone:llama1b", []):
            host_scores["tap_llama1b"] = row["accuracy"]
        for row in by_kind.get("host_alone:qwen05", []):
            host_scores["tap_qwen05"] = row["accuracy"]
        for key in ["qwen14b", "gemma9b", "phi35", "llama8b", "lfm12b"]:
            hf_id = RESULT_KEYS[key]
            layer = ROUND8_READ[key]
            items = []
            alone = None
            for row in by_kind.get("donor_alone:" + key, []):
                alone = row["accuracy"]
            for kind, host_name, host_layer in [("tap_llama1b", "Llama-1B", 10), ("tap_qwen05", "Qwen-0.5B", 15)]:
                scores = []
                seeds = []
                for row in by_kind.get(kind + ":" + key, []):
                    scores.append(row["accuracy"])
                    seeds.append(row["seed"])
                if len(scores) == 0:
                    continue
                items.append(evidence(host_name + " + tap reading layer " + str(layer) + " at host layer " +
                                      str(host_layer) + " vs the host alone", "MMLU dev accuracy, mean over seeds",
                                      pct(mean(scores)), pct(host_scores[kind]), MMLU_DEV, sorted(seeds),
                                      "results/round8_rows.json", control="host alone, no training",
                                      note="donor alone " + str(alone)))
            status = "candidate"
            reason = "read point measured on MMLU dev items only"
            if key == "llama8b":
                reason = "unstable read point: seeds 49.2 and 58.0 (run 8)"
            if key == "lfm12b":
                reason = "weak read point: 49.9 vs host 47.5 (run 8)"
            out.append(tap_read_block("tapread_" + tag(hf_id) + "_l" + str(layer),
                                      hf_id, layer, short_model(hf_id) + " layers 0.." + str(layer) +
                                      ", tap read point for knowledge", status, items, reason,
                                      ["knowledge", "mmlu"], "results/round8_rows.json (run 8)"))

    # GSM8K read points from run 5: taps carry no reasoning.
    path = os.path.join(results, "round5_rows.json")
    if os.path.exists(path):
        rows = read_json(path)["rows"]
        arms = {}
        for row in rows:
            if row["arm"] not in arms:
                arms[row["arm"]] = []
            arms[row["arm"]].append(row)

        def arm_mean(name):
            values = []
            seeds = []
            for row in arms.get(name, []):
                values.append(row["accuracy"])
                seeds.append(row["seed"])
            return pct(mean(values)), sorted(seeds)

        host_value, _ = arm_mean("host_only")
        real, seeds = arm_mean("tap_l27_h12")
        fake, _ = arm_mean("tap_derange")
        items = [evidence("Llama-1B + tap reading layer 27 at host layer 12 vs the same tap on mismatched questions",
                          "GSM8K search accuracy, mean over seeds", real, fake, GSM8K_SEARCH, seeds,
                          "results/round5_rows.json", control="mismatched-message control",
                          note="host alone " + str(host_value))]
        qmath_instruct = "Qwen/Qwen2.5-Math-1.5B-Instruct"
        out.append(tap_read_block("tapread_" + tag(qmath_instruct) + "_l27", qmath_instruct, 27,
                                  "Qwen2.5-Math-1.5B-Instruct layers 0..27, tap read point for GSM8K", "rejected",
                                  items, "run 5: the tap equals its mismatched-message control; prefill taps carry "
                                         "no reasoning", ["math", "gsm8k"], "results/round5_rows.json (run 5)"))
        lfm, lfm_seeds = arm_mean("tap_lfm_l8_h8")
        items = [evidence("Llama-1B + tap reading layer 8 at host layer 8 vs the host alone",
                          "GSM8K search accuracy, mean over seeds", lfm, host_value, GSM8K_SEARCH, lfm_seeds,
                          "results/round5_rows.json", control="host alone, no training",
                          note="no mismatched control for this tap; on top of LoRA r64 it adds nothing (run 6)")]
        out.append(tap_read_block("tapread_lfm2_350m_math_l8", "LiquidAI/LFM2-350M-Math", 8,
                                  "LFM2-350M-Math layers 0..8, tap read point for GSM8K", "candidate", items,
                                  "the only run 5 arm above noise (+7.7 over host), search split only",
                                  ["math", "gsm8k"], "results/round5_rows.json (run 5)"))

    # Qwen2.5-Math-1.5B blocks grafted in-stream (runs 3 and 4, stitch graft).
    block_rows = []
    for name in ["round3_rows.json", "round4_rows.json"]:
        path = os.path.join(results, name)
        if not os.path.exists(path):
            continue
        data = read_json(path)
        rows = data
        if isinstance(data, dict):
            rows = data["rows"]
        for row in rows:
            row["_source"] = "results/" + name
            block_rows.append(row)
    for block_id, layers, arm in [("qmath_block_8_19", [8, 19], "block_a_r64"), ("qmath_block_4_23", [4, 23],
                                                                                   "block_b_r64")]:
        items = []
        for host in ["llama1b", "qwen05"]:
            graft = []
            control = []
            seeds = []
            source = None
            for row in block_rows:
                row_host = row.get("host", "llama1b")
                if row_host != host:
                    continue
                if row["arm"] == arm:
                    graft.append(row["accuracy"])
                    seeds.append(row["seed"])
                    source = row["_source"]
                if row["arm"] == "block_rand":
                    control.append(row["accuracy"])
            if len(graft) == 0:
                continue
            items.append(evidence("block graft with rank-64 glue vs a random same-length block from the same donor",
                                  "GSM8K search accuracy, mean over seeds", pct(mean(graft)), pct(mean(control)),
                                  GSM8K_SEARCH, sorted(seeds), source, control="random same-length block",
                                  note="host " + RESULT_KEYS[host] + ", 300 heal steps, ridge W_in"))
        if block_id == "qmath_block_8_19":
            path = os.path.join(results, "stitch_graft_s0.json")
            if os.path.exists(path):
                arms = {}
                for row in read_json(path):
                    arms[row["arm"]] = row
                if "block_stitch" in arms and "random_stitch" in arms:
                    items.append(evidence("block graft at Qwen-0.5B layer 7 with a trained donor_lm linear stitch vs "
                                          "a random block with the same stitch", "GSM8K search accuracy",
                                          arms["block_stitch"]["gsm8k"], arms["random_stitch"]["gsm8k"], GSM8K_SEARCH,
                                          [0], "results/stitch_graft_s0.json", control="random same-length block",
                                          note="matched LoRA " + str(arms.get("lora", {}).get("gsm8k")) +
                                               ", host " + str(arms.get("host", {}).get("gsm8k"))))
        full = []
        for index in range(layers[0], layers[1] + 1):
            full.append(index)
        out.append(component(block_id, "block", "Qwen2.5-Math-1.5B layers " + str(layers[0]) + " to " +
                             str(layers[1]) + " as an in-stream block", "rejected", TRISANTH,
                             "results/round3_rows.json, results/round4_rows.json (runs 3 and 4)", ["math", "gsm8k"],
                             "Qwen/Qwen2.5-Math-1.5B", full, None, None,
                             {"method": "whole donor layers run with the donor's own layer code",
                              "part_params": 561573888 if block_id == "qmath_block_8_19" else 935956480},
                             items,
                             {"catalog_id": block_id,
                              "status_reason": "runs 3 and 4: the block equals a random block on both hosts; "
                                               "the stitch graft (1 seed) also ties the random block"}))

    # Run 14 blocks: donor layers written into Qwen2.5-0.5B-Instruct (results/round14_rows.json).
    rows = raw_rows(repo, "round14_rows.json")
    host_heldout = raw_mean(repo, "round14_rows.json", field="heldout", arm="host")
    lora_heldout = raw_mean(repo, "round14_rows.json", field="heldout", arm="lora")
    plans = [("r14_qwen25_7b_l8_20", QWEN7B, 8, 20, "g7", "g7_random",
              "loses to a random same-length Qwen-7B block on held-out MMLU (2 seeds)"),
             ("r14_qwen25_14b_l14_34", QWEN14B, 14, 34, "g14", None,
              "about +3 held-out over the base from 6B parameters, inside the spread of the 7B random-block "
              "control (45.4, 49.5); results/README.md run 14: the base cannot read the block when only the "
              "seam glue trains")]
    for block_id, hf_id, first, last, arm, random_arm, reason in plans:
        full = []
        for index in range(first, last + 1):
            full.append(index)
        items = []
        value, seeds = seeds_and_mean(rows_where(rows, arm=arm), key="heldout")
        penalties = []
        for row in rows_where(rows, arm=arm):
            stitch_info = (row.get("grafts") or [{}])[0].get("stitch") or {}
            after = (stitch_info.get("after") or {}).get("penalty")
            if after is not None:
                penalties.append(round(after, 4))
        note = ("host Qwen/Qwen2.5-0.5B-Instruct at layer 7, MLP stitch trained 800 steps on the donor LM loss with "
                "the sink, glue rank 64, heal 500 steps on MMLU aux; stitch penalty after training by seed " +
                json.dumps(penalties) + "; host alone " + str(host_heldout))
        if random_arm is not None:
            random_value, _ = seeds_and_mean(rows_where(rows, arm=random_arm), key="heldout")
            items.append(evidence("block written into the base vs a random same-length block from the same donor",
                                  "MMLU held-out accuracy, mean over seeds", value, random_value, MMLU_HELDOUT, seeds,
                                  "results/round14_rows.json", control="random same-length block, same stitch and heal",
                                  note=note))
        if value is not None:
            items.append(evidence("block written into the base vs a rank-16 LoRA on the base",
                                  "MMLU held-out accuracy, mean over seeds", value, lora_heldout, MMLU_HELDOUT, seeds,
                                  "results/round14_rows.json", control="base LoRA rank 16, same heal data and steps",
                                  note=note))
        plora, plora_seeds = seeds_and_mean(rows_where(rows, arm=arm + "_plora"), key="heldout")
        if plora is not None:
            items.append(evidence("the same block plus a rank-8 LoRA on the grafted layers vs a rank-16 LoRA on the "
                                  "base", "MMLU held-out accuracy, mean over seeds", plora, lora_heldout, MMLU_HELDOUT,
                                  plora_seeds, "results/round14_rows.json", control="base LoRA rank 16"))
        source = "results/round14_rows.json (run 14)"
        if len(rows) == 0:
            source = "modal_round14.py (run 14); rows not found in results/"
        out.append(component(block_id, "block", short_model(hf_id) + " layers " + str(first) + " to " + str(last) +
                             " written into Qwen2.5-0.5B-Instruct (run 14)", "rejected", TRISANTH, source,
                             ["knowledge", "mmlu"], hf_id, full, None, None,
                             {"method": "donor 'knowledge' layers grafted inside the host as a residual branch "
                                        "h + W_out(block(W_in h) - x), with a trained MLP stitch and the donor sink "
                                        "(modal_round14.py)"},
                             items,
                             {"catalog_id": "block/" + hf_id + "/" + str(first) + "-" + str(last),
                              "status_reason": reason}))
    return out


def build_components(repo):
    """Return a list of (entry, copies), copies being (source path, library-relative path)."""
    out = []
    for data in build_masks(repo):
        out.append((data, []))
    for data in build_round2(repo):
        out.append((data, []))
    for data in build_run1(repo):
        out.append((data, []))
    for data in build_gate_part(repo):
        out.append((data, []))
    for pair in build_person_b(repo):
        out.append(pair)
    for pair in build_person_a(repo):
        out.append(pair)
    for data in build_pilots(repo):
        out.append((data, []))
    for data in build_blocks(repo):
        out.append((data, []))
    return out


# Findings


def finding(finding_id, title, claim, setup, numbers, source, status, skills, models, metric, value, control_value,
            split, seeds, status_reason=None, control=None, added_by=TRISANTH, numbers_source="results/README.md"):
    item = evidence(claim, metric, value, control_value, split, seeds, source, control=control)
    fields = {"claim": claim, "setup": setup, "numbers": numbers, "numbers_source": numbers_source}
    if status_reason is not None:
        fields["status_reason"] = status_reason
    return entry(finding_id, "finding", title, status, added_by, source, skills, models, [item], fields)


# Exact values from the raw results files. The "numbers" of a finding are the rounded
# percents that results/README.md prints; the evidence value and control_value come
# from these helpers, so they match the source file exactly.


def raw_json(repo, name):
    path = os.path.join(repo, "results", name)
    if not os.path.exists(path):
        return None
    return read_json(path)


def raw_rows(repo, name):
    """The rows of a results file: the file itself when it is a list, else its 'rows' list."""
    data = raw_json(repo, name)
    if data is None:
        return []
    if isinstance(data, dict):
        return data.get("rows") or []
    return data


def raw_mean(repo, name, field="accuracy", **match):
    """Mean of field over the rows that match, rounded to 4 places, or None."""
    values = []
    for row in rows_where(raw_rows(repo, name), **match):
        if row.get(field) is not None:
            values.append(row[field])
    return pct(mean(values))


def stitch_study_value(repo, cell, side, domain, metric):
    for row in raw_rows(repo, "stitch_study_rows.json"):
        if row.get("cell") == cell:
            block = (row.get(side) or {}).get(domain) or {}
            return pick(block, metric)
    return None


def audit_value(repo, host, variant):
    for row in raw_rows(repo, "audit_sink_rows.json"):
        if row.get("host") == host:
            value = (row["variants"].get(variant) or {}).get("penalty")
            if value is not None:
                return round(value, 4)
    return None


def round9_best_tap(repo, host):
    """(best mean over tap rows on host, its donors), from round9_rows.json."""
    groups = {}
    for row in rows_where(raw_rows(repo, "round9_rows.json"), kind="tap", host=host):
        if row["donors"] not in groups:
            groups[row["donors"]] = []
        groups[row["donors"]].append(row["accuracy"])
    best = None
    best_donors = None
    for donors in groups:
        value = mean(groups[donors])
        if best is None or value > best:
            best = value
            best_donors = donors
    return pct(best), best_donors


def exact_finding_values(repo):
    """{finding id: (value, control_value)} computed from the raw results files."""
    out = {}
    run1 = raw_json(repo, "graft_run1_s0.json")
    if run1 is not None:
        out["f_run1_graft_below_random"] = (pct(run1["graft"]["accuracy"]), pct(run1["control"]["accuracy"]))
        out["f_run1_fit_r2_inflated"] = (pct(run1["graft"]["fit_r2"]), None)
    sweep = "sweep2_rows.json"
    out["f_run2_selection_equals_random"] = (
        raw_mean(repo, sweep, layer=10, selection="activation512", rank="r64"),
        raw_mean(repo, sweep, layer=10, selection="random", rank="r64"))
    out["f_run2_placement_dominates"] = (raw_mean(repo, sweep, layer=10, selection="random", rank="r64"),
                                         raw_mean(repo, sweep, layer=18, selection="random", rank="r64"))
    out["f_run2_rank64_glue"] = (raw_mean(repo, sweep, layer=10, selection="activation512", rank="r64"),
                                 raw_mean(repo, sweep, layer=10, selection="activation512", rank="full"))
    out["f_run3_lora_beats_blocks"] = (raw_mean(repo, "round3_rows.json", arm="block_a_r64"),
                                       raw_mean(repo, "round3_rows.json", arm="block_rand"))
    out["f_run3_lora_r64_underperforms"] = (raw_mean(repo, "round3_rows.json", arm="lora_r64"), None)
    out["f_run3_block_decode_cost"] = (raw_mean(repo, "round3_rows.json", field="decode_tok_s", arm="block_a_r64"),
                                       raw_mean(repo, "round3_rows.json", field="decode_tok_s", arm="lora_matched"))
    round4 = raw_json(repo, "round4_rows.json")
    if round4 is not None:
        for pair in round4["stitch_penalty"]["pairs"]:
            if pair["host_layer"] == 7 and pair["donor_layer"] == 8:
                out["f_run4_linear_stitch_penalty"] = (round(pair["stitched_loss"], 4),
                                                       round(round4["stitch_penalty"]["native_loss"], 4))
    out["f_run4_same_tokenizer_no_rescue"] = (raw_mean(repo, "round4_rows.json", host="qwen05", arm="block_a_r64"),
                                              raw_mean(repo, "round4_rows.json", host="qwen05", arm="block_rand"))
    out["f_run4_lora_bar"] = (raw_mean(repo, "round4_rows.json", host="llama1", arm="lora_r64_lr5e5"),
                              raw_mean(repo, "round5_rows.json", arm="host_only"))
    out["f_audit_attention_sink"] = (audit_value(repo, "Qwen/Qwen2.5-1.5B", "current_lam1e-2_meanvar+sink_mean"),
                                     audit_value(repo, "Qwen/Qwen2.5-1.5B", "current_lam1e-2_meanvar"))
    out["f_audit_ridge_lambda"] = (audit_value(repo, QMATH, "lam1e-6"),
                                   audit_value(repo, QMATH, "current_lam1e-2_meanvar"))
    out["f_run5_taps_no_reasoning"] = (raw_mean(repo, "round5_rows.json", arm="tap_l27_h12"),
                                       raw_mean(repo, "round5_rows.json", arm="tap_derange"))
    out["f_run5_lfm_tap"] = (raw_mean(repo, "round5_rows.json", arm="tap_lfm_l8_h8"),
                             raw_mean(repo, "round5_rows.json", arm="host_only"))
    out["f_run6_tap_beats_finetune_dev"] = (raw_mean(repo, "round6_mmlu_rows.json", arm="tap_l20_h10"),
                                            raw_mean(repo, "round6_mmlu_rows.json", arm="lora_r16"))
    out["f_run6_joint_training_collapses"] = (raw_mean(repo, "round6_mmlu_rows.json", arm="tap_l27_lora"),
                                              raw_mean(repo, "round6_mmlu_rows.json", arm="tap_l27_h12"))
    r7 = "round7_rows.json"
    out["f_run7_heldout_tap"] = (raw_mean(repo, r7, part="heldout", arm="tap"),
                                 raw_mean(repo, r7, part="heldout", arm="lora_r16"))
    layer_map = {}
    scaling_tap = {}
    scaling_alone = {}
    for row in raw_rows(repo, r7):
        cfg = row.get("cfg") or {}
        if row["part"] == "layers" and row["arm"] == "tap":
            layer_map[(cfg["donor_layer"], cfg["host_layer"])] = row["accuracy"]
        if row["part"] == "scaling" and row["arm"] == "tap":
            if cfg["key"] not in scaling_tap:
                scaling_tap[cfg["key"]] = []
            scaling_tap[cfg["key"]].append(row["accuracy"])
        if row["part"] == "scaling" and row["arm"] == "donor":
            scaling_alone[cfg["key"]] = row["accuracy"]
    if (20, 6) in layer_map and (8, 6) in layer_map:
        out["f_run7_layer_map"] = (pct(layer_map[(20, 6)]), pct(layer_map[(8, 6)]))
    if "q14b" in scaling_tap and "q7b" in scaling_alone:
        out["f_run7_donor_scaling"] = (pct(mean(scaling_tap["q14b"])), pct(scaling_alone["q7b"]))
    out["f_run7_staged_lora"] = (raw_mean(repo, r7, part="staged", arm="staged"),
                                 raw_mean(repo, "round6_mmlu_rows.json", arm="tap_l20_h10"))
    out["f_run7_tap_hurts_generation"] = (raw_mean(repo, r7, part="trivia", arm="tap"),
                                          raw_mean(repo, r7, part="trivia", arm="host"))
    r8 = "round8_rows.json"
    out["f_run8_small_hosts_reach_donors"] = (raw_mean(repo, r8, kind="tap_qwen05", key="qwen14b"),
                                              raw_mean(repo, r8, kind="host_alone", key="qwen05"))
    out["f_run8_two_taps_not_additive"] = (raw_mean(repo, r8, kind="two_taps_llama1b", key="qwen14b+gemma9b"),
                                           raw_mean(repo, r8, kind="tap_llama1b", key="gemma9b"))
    out["f_run8_weak_donors"] = (raw_mean(repo, r8, kind="tap_llama1b", key="llama8b"),
                                 raw_mean(repo, r8, kind="host_alone", key="llama1b"))
    full_ft = None
    for row in (raw_json(repo, "baselines_rows.json") or {}).get("summary", []):
        if row["arm"] == "full_ft unsloth/Llama-3.2-1B-Instruct lr 1e-5":
            full_ft = pct(row["score_mean"])
        if row["arm"] == "ceiling Qwen/Qwen2.5-1.5B-Instruct":
            out["f_baselines_raw_prompt"] = (pct(row["score_mean"]), None)
    out["f_baselines_full_ft"] = (full_ft, raw_mean(repo, "round5_rows.json", arm="lora_r64"))
    best, _ = round9_best_tap(repo, "q14b")
    out["f_run9_strong_host"] = (best, raw_mean(repo, "round9_rows.json", kind="alone", host="q14b"))
    out["f_stitch_donor_lm"] = (
        stitch_study_value(repo, "qwen05b_instruct_h7_d8_donor_lm_linear", "after", "gsm8k", "penalty"),
        stitch_study_value(repo, "qwen05b_instruct_h7_d8_ridge", "after", "gsm8k", "penalty"))
    out["f_stitch_mse_no_gain"] = (
        stitch_study_value(repo, "qwen15b_h7_d8_mse_linear", "after", "gsm8k", "penalty"),
        stitch_study_value(repo, "qwen15b_h7_d8_ridge", "after", "gsm8k", "penalty"))
    out["f_stitch_negative_wiki"] = (
        stitch_study_value(repo, "qwen15b_h7_d8_donor_lm_mlp", "after", "wiki", "penalty"),
        stitch_study_value(repo, "qwen15b_h7_d8_ridge", "after", "wiki", "penalty"))
    out["f_stitch_cross_tokenizer"] = (
        stitch_study_value(repo, "llama1b_instruct_h5_d8_downstream_mlp", "after", "gsm8k", "downstream_loss"),
        stitch_study_value(repo, "llama1b_instruct_h5_d8_ridge", "after", "gsm8k", "downstream_loss"))
    official_7b = None
    carve = {}
    for row in raw_rows(repo, "round15_rows.json"):
        if row.get("kind") == "official" and row["model"] == QWEN7B:
            official_7b = row["scores"]["mmlu_heldout"]
        if row.get("kind") == "carve" and row["model"] == QWEN14B and row["heal"] == "lora":
            carve[row["select"]] = row["scores"]["mmlu_heldout"]
    if "uniform" in carve:
        out["f_run15_carving_half_destroys"] = (pct(carve["uniform"]), pct(official_7b))
        if "influence" in carve:
            out["f_run15_influence_vs_uniform"] = (pct(carve["influence"]), pct(carve["uniform"]))
    final = raw_json(repo, "final_heldout_microsoft__phi-4.json")
    if final is not None:
        out["f_part_gate_search_not_heldout"] = (pct(final["verdict"]["graft"]["mean"]),
                                                 pct(final["verdict"]["main_control"]["mean"]))
    gate = raw_json(repo, "gate_hour8_3seeds.json")
    if gate is not None:
        out["f_gate_hour8_contrast"] = (pct(gate["verdict"]["graft"]["mean"]),
                                        pct(gate["verdict"]["main_control"]["mean"]))
    bake = {}
    for row in raw_rows(repo, "bake_rows.json"):
        bake[row["arm"]] = row
    if "bake_tap" in bake and "bake_donor" in bake:
        out["f_bake_no_gain"] = (pct(bake["bake_tap"]["scores"]["heldout"]),
                                 pct(bake["bake_donor"]["scores"]["heldout"]))
    graft = {}
    for row in raw_rows(repo, "stitch_graft_s0.json"):
        graft[row["arm"]] = row
    if "block_stitch" in graft and "random_stitch" in graft:
        out["f_stitch_graft_ties_random"] = (pct(graft["block_stitch"]["gsm8k"]), pct(graft["random_stitch"]["gsm8k"]))
    exit_rows = {}
    for row in raw_rows(repo, "exit_stitch_s0.json"):
        exit_rows[row["arm"]] = row
    if "graft" in exit_rows and "random" in exit_rows:
        out["f_exit_stitch_no_gain"] = (pct(exit_rows["graft"]["gsm8k"]), pct(exit_rows["random"]["gsm8k"]))
    r14 = "round14_rows.json"
    out["f_run14_reader_needed"] = (raw_mean(repo, r14, field="heldout", arm="g7"),
                                    raw_mean(repo, r14, field="heldout", arm="g7_random"))
    return out


def apply_exact_values(findings, values):
    """Put the exact values into each finding's first evidence item. A missing value stays as written."""
    for data in findings:
        pair = values.get(data["id"])
        if pair is None:
            continue
        item = data["evidence"][0]
        if pair[0] is not None:
            item["value"] = pair[0]
        if pair[1] is not None or item.get("control_value") is None:
            item["control_value"] = pair[1]
        item["note"] = "value and control_value are exact from the source file; 'numbers' are as printed in " + \
                       data.get("numbers_source", "results/README.md")


LLAMA1B = "unsloth/Llama-3.2-1B-Instruct"
QWEN05 = "Qwen/Qwen2.5-0.5B-Instruct"
QMATH = "Qwen/Qwen2.5-Math-1.5B"
QWEN7B = "Qwen/Qwen2.5-7B-Instruct"
QWEN14B = "Qwen/Qwen2.5-14B-Instruct"
QWEN32B = "Qwen/Qwen2.5-32B-Instruct"
GEMMA9B = "unsloth/gemma-2-9b-it"


def complementarity_finding(repo):
    """Oracle and probability-average numbers over the round 10 per-item records."""
    path = os.path.join(repo, "results", "round10_rows.json")
    if not os.path.exists(path):
        return []
    models = read_json(path)
    numbers = {}
    dev_accuracy = {}
    for name in ["mmlu_dev", "mmlu_heldout", "medmcqa"]:
        tables = {}
        accuracy = {}
        for model in models:
            table = {}
            right = 0
            for record in model["records"]:
                if record["set"] == name:
                    table[record["id"]] = record
                    if record["correct"]:
                        right += 1
            tables[model["key"]] = table
            accuracy[model["key"]] = right / len(table)
        if name == "mmlu_dev":
            dev_accuracy = accuracy
        keys = []
        for model in models:
            keys.append(model["key"])
        # The top 3 models by MMLU dev accuracy, so held-out numbers are not chosen on held-out items.
        top3 = sorted(keys, key=lambda key: -dev_accuracy[key])[:3]
        ids = []
        for item_id in tables[keys[0]]:
            present = True
            for key in keys:
                if item_id not in tables[key]:
                    present = False
            if present:
                ids.append(item_id)
        best = keys[0]
        for key in keys:
            if accuracy[key] > accuracy[best]:
                best = key
        row = {"items": len(ids), "best_model": RESULT_KEYS.get(best, best), "best": round(accuracy[best], 4),
               "top3_by_dev": []}
        for key in top3:
            row["top3_by_dev"].append(RESULT_KEYS.get(key, key))
        for label, group in [("all_11", keys), ("top3", top3)]:
            oracle = 0
            average = 0
            for item_id in ids:
                any_right = False
                total = None
                for key in group:
                    record = tables[key][item_id]
                    if record["correct"]:
                        any_right = True
                    if total is None:
                        total = list(record["probs"])
                    else:
                        for index in range(len(total)):
                            total[index] += record["probs"][index]
                if any_right:
                    oracle += 1
                if total.index(max(total)) == tables[keys[0]][item_id]["gold"]:
                    average += 1
            row["oracle_" + label] = round(oracle / len(ids), 4)
            row["average_" + label] = round(average / len(ids), 4)
        numbers[name] = row
    held = numbers["mmlu_heldout"]
    claim = ("Models are right on different items: on held-out MMLU some model is right on " +
             str(round(held["oracle_all_11"] * 100, 1)) + "% of items against " + str(round(held["best"] * 100, 1)) +
             "% for the best model, but averaging option probabilities captures little of it (" +
             str(round(held["average_all_11"] * 100, 1)) + "% over all 11, " +
             str(round(held["average_top3"] * 100, 1)) + "% over the top 3 by dev accuracy)")
    model_ids = []
    for model in models:
        model_ids.append(model["model"])
    return [finding("f_round10_complementarity", "Complementarity is large; plain averaging captures little of it",
                    claim, "round 10: 11 models, no training, per-item option probabilities on MMLU dev, MMLU held-out "
                    "and MedMCQA; oracle = any model right; average = argmax of the summed probabilities. Computed by "
                    "scripts/library.py build.", numbers, "results/round10_rows.json", "validated",
                    ["knowledge", "mmlu", "medmcqa", "ensemble"], model_ids, "MMLU held-out oracle accuracy",
                    held["oracle_all_11"], held["best"], MMLU_HELDOUT, None, control="best single model",
                    numbers_source="computed by scripts/library.py build from results/round10_rows.json")]


def build_findings(repo):
    out = []
    add = out.append
    add(finding("f_run1_graft_below_random", "A cross-family MLP graft did not beat its random control",
                "A 512-neuron Qwen2.5-Math-1.5B layer 18 cluster grafted into Llama-1B at layer 10 scored 25.0 on "
                "GSM8K search; a random 512-neuron control with the same glue and heal scored 33.0",
                "run 1: host Llama-3.2-1B-Instruct, full-rank glue (6.3M params), 300 heal steps, 1 seed",
                {"host": 25.0, "graft": 25.0, "random_control": 33.0, "glue_params_m": 6.3, "part_params_m": 2.4},
                "results/graft_run1_s0.json", "candidate", ["math", "gsm8k", "graft"], [LLAMA1B, QMATH],
                "GSM8K search accuracy", 0.25, 0.33, GSM8K_SEARCH, [0], control="random same-size cluster"))
    add(finding("f_run1_fit_r2_inflated", "Run 1's fit R squared of 0.987 was not real",
                "The ridge fit from Llama-1B states to Qwen-Math states explains 98.7% of variance",
                "run 1 pooled fit R squared; a review found the fit read the host state without BOS, so first-token "
                "outlier rows dominated it", {"run1_pooled_r2": 0.987, "run2_pooled_r2": [0.57, 0.68],
                                              "run2_per_dim_r2": [0.45, 0.53]},
                "results/graft_run1_s0.json", "rejected", ["stitch", "graft"], [LLAMA1B, QMATH], "pooled fit R squared",
                0.987, None, GSM8K_SEARCH, [0],
                status_reason="run 2 with the review fixes measured 0.57 to 0.68 pooled (results/sweep2_rows.json)"))
    add(finding("f_run2_selection_equals_random", "Single-layer 512-neuron clusters do not beat random",
                "No selection method (activation, contrast, attribution) beats a random same-size cluster at donor "
                "layers 10, 18, or 24, with full or rank-64 glue",
                "run 2: Llama-1B host, Qwen2.5-Math-1.5B donor, 72 cells, 300 heal steps, 3 seeds",
                {"layer10_rank64": {"activation": 40.5, "contrast": 38.7, "attribution": 38.3, "random": 38.7},
                 "layer10_full": {"activation": 31.2, "contrast": 32.8, "attribution": 34.5, "random": 38.7},
                 "layer18_rank64": {"activation": 25.7, "contrast": 24.7, "attribution": 26.2, "random": 24.8},
                 "layer24_rank64": {"activation": 24.8, "contrast": 24.2, "attribution": 23.5, "random": 23.2}},
                "results/sweep2_rows.json", "candidate", ["math", "gsm8k", "graft"], [LLAMA1B, QMATH],
                "GSM8K search accuracy, best method at layer 10 rank 64", 0.405, 0.387, GSM8K_SEARCH, [0, 1, 2],
                control="random same-size cluster"))
    add(finding("f_run2_placement_dominates", "Graft placement matters more than the part",
                "Grafting at host layer 6 gives about 38 to 40% on GSM8K search whatever the part; host layers 10 and "
                "14 give about 22 to 26%. The glue acts as an adapter at a good depth",
                "run 2, Llama-1B host", {"host_layer_6": [38.3, 40.5], "host_layer_10": [22.3, 26.2],
                                          "host_layer_14": [21.7, 24.8]},
                "results/sweep2_rows.json", "candidate", ["graft", "placement"], [LLAMA1B, QMATH],
                "GSM8K search accuracy, random part at host layer 6, rank 64", 0.387, 0.248, GSM8K_SEARCH, [0, 1, 2],
                control="random part at host layer 10"))
    add(finding("f_run2_rank64_glue", "Rank-64 glue matches or beats full glue at 7% of the params",
                "Rank-64 glue (0.46M params) is as good as or better than full-rank glue (6.3M) in every run 2 cell",
                "run 2, 3 seeds", {"rank64_params_m": 0.46, "full_params_m": 6.3},
                "results/sweep2_rows.json", "candidate", ["graft", "glue"], [LLAMA1B, QMATH],
                "GSM8K search accuracy, activation part at layer 10", 0.405, 0.312, GSM8K_SEARCH, [0, 1, 2],
                control="full-rank glue"))
    add(finding("f_run3_lora_beats_blocks", "A rank-1 LoRA beats every block graft",
                "On Llama-1B a rank-1 LoRA (0.70M params) scores 43.5 on GSM8K search; donor blocks of Qwen-Math "
                "layers 8..19 and 4..23 score 35.2 and 33.5, a random block 36.3",
                "run 3: 300 heal steps, same loop and data, 3 seeds",
                {"lora_r1": 43.5, "block_8_19": 35.2, "block_4_23": 33.5, "random_block": 36.3, "block_full_glue": 16.5},
                "results/round3_rows.json", "candidate", ["math", "gsm8k", "graft", "lora"], [LLAMA1B, QMATH],
                "GSM8K search accuracy", 0.352, 0.363, GSM8K_SEARCH, [0, 1, 2], control="random same-length block"))
    add(finding("f_run3_lora_r64_underperforms", "LoRA r64 at lr 3e-4 looked weak (28.8); the lr was too high",
                "A rank-64 LoRA is a weak fine-tuning bar on GSM8K (28.8)",
                "run 3 used lr 3e-4; run 4 reran rank 64 at lr 5e-5", {"run3_r64_lr3e-4": 28.8, "run4_r64_lr5e-5": 46.7},
                "results/round3_rows.json", "rejected", ["lora", "baseline"], [LLAMA1B], "GSM8K search accuracy",
                0.288, None, GSM8K_SEARCH, [0, 1, 2],
                status_reason="run 4: rank 64 at lr 5e-5 scores 46.7, the fair bar (results/round4_rows.json)"))
    add(finding("f_run3_block_decode_cost", "Block grafts cost decode speed",
                "Block grafts decode at 37 to 48 tok/s against 87 to 110 for LoRA arms, because the KV cache was off "
                "for them", "run 3 decode speed at batch 1", {"block_tok_s": [37, 48], "lora_tok_s": [87, 110]},
                "results/round3_rows.json", "candidate", ["cost", "graft"], [LLAMA1B, QMATH], "decode tokens per second",
                42, 87, GSM8K_SEARCH, [0, 1, 2], control="LoRA rank 1"))
    add(finding("f_run4_linear_stitch_penalty", "A linear per-token stitch breaks the donor",
                "Replacing the donor's input with a ridge map of the host state raises the donor's own loss by 1.4 to "
                "2.2 nats, even with the same tokenizer",
                "run 4: host Qwen2.5-0.5B-Instruct, donor Qwen2.5-Math-1.5B, donor native loss 0.80 on 150 GSM8K texts",
                {"native": 0.80, "3_to_4": 2.85, "7_to_8": 2.16, "12_to_14": 2.99, "17_to_20": 2.78},
                "results/round4_rows.json", "candidate", ["stitch"], [QWEN05, QMATH], "stitched donor loss, 7 to 8",
                2.16, 0.80, "150 GSM8K texts", None, control="donor native loss"))
    add(finding("f_run4_same_tokenizer_no_rescue", "The same tokenizer does not rescue block grafts",
                "On a same-tokenizer host (Qwen-0.5B) donor blocks equal a random block: 36.3 and 36.3 vs 37.8",
                "run 4: 3 seeds, 300 heal steps", {"block_8_19": 36.3, "block_4_23": 36.3, "random_block": 37.8,
                                                   "host": 33.5, "lora_r1": 37.3},
                "results/round4_rows.json", "candidate", ["graft", "stitch"], [QWEN05, QMATH], "GSM8K search accuracy",
                0.363, 0.378, GSM8K_SEARCH, [0, 1, 2], control="random same-length block"))
    add(finding("f_run4_lora_bar", "The fair LoRA bar on Llama-1B GSM8K is 46.7",
                "LoRA rank 64 at lr 5e-5 scores 46.7 on GSM8K search for Llama-1B (47.2 in run 5)",
                "run 4: 3 seeds, 300 steps", {"llama1b_r64_lr5e-5": 46.7, "llama1b_r16_lr1e-4": 44.0, "run5_r64": 47.2},
                "results/round4_rows.json", "candidate", ["lora", "baseline", "gsm8k"], [LLAMA1B],
                "GSM8K search accuracy", 0.467, 0.25, GSM8K_SEARCH, [0, 1, 2], control="host alone"))
    add(finding("f_audit_extraction_exact", "Part extraction is exact",
                "Range-fetched tensors equal the checkpoint (max diff 0.0); a block of layers 8..19 equals the donor's "
                "own layers on real text (max diff 4e-5 at scale 243); a full mlp_cluster equals the donor MLP",
                "audit on real Qwen2.5-Math-1.5B", {"tensor_max_diff": 0.0, "block_max_diff": 4e-5},
                "results/README.md (audit)", "validated", ["extraction"], [QMATH], "max abs difference", 0.0, None,
                "real text, run 4 penalty texts", None))
    add(finding("f_audit_attention_sink", "Feed the donor its attention sink at position 0",
                "Qwen adds no BOS token, so the first text token is the attention sink. Feeding the donor's mean sink "
                "state at position 0 cuts the stitch penalty by 2.1x to 3.4x; the mean equals the per-text oracle",
                "audit: stitch penalty into Qwen2.5-Math-1.5B layer 8, ridge as shipped (lam 1e-2), donor native "
                "loss 0.805",
                {"qwen15b_base_L7": {"as_shipped": 1.235, "lam1e-6": 1.170, "sink": 0.367},
                 "qwen05_instruct_L7": {"as_shipped": 1.357, "lam1e-6": 1.441, "sink": 0.643}},
                "results/audit_sink_rows.json", "validated", ["stitch", "sink"], [QMATH, "Qwen/Qwen2.5-1.5B", QWEN05],
                "stitch penalty (nats), Qwen2.5-1.5B L7 to L8", 0.367, 1.235, "150 GSM8K texts (run 4 eval texts)", None,
                control="the same ridge stitch (as shipped) without the sink"))
    add(finding("f_audit_ridge_lambda", "The shipped ridge regularizer cost 0.05 nats on an exact map",
                "With lam 1e-2 the identity map (Math L7 to Math L8) has 0.052 nats of penalty; relative lam 1e-6 gives "
                "0.000", "audit", {"lam1e-2": 0.052, "lam1e-6": 0.0}, "results/audit_sink_rows.json", "validated",
                ["stitch"], [QMATH], "stitch penalty on the identity map", 0.0, 0.052, "150 GSM8K texts", None,
                control="shipped regularizer"))
    add(finding("f_run5_taps_no_reasoning", "Prefill taps carry no multi-step reasoning",
                "A tap reading Qwen-Math-1.5B-Instruct scores the same as a tap reading mismatched questions on GSM8K "
                "(25.8 vs 26.2); during prefill the donor has not solved the problem",
                "run 5: Llama-1B host, 300 steps, 3 seeds; run 6 confirms on top of LoRA r64",
                {"tap_l27": 25.8, "mismatched": 26.2, "host": 25.0, "lora_r64": 47.2,
                 "run6_lora_plus_tap": [47.0, 47.5], "run6_lora_null_tap": 47.5},
                "results/round5_rows.json", "candidate", ["math", "gsm8k", "tap", "reasoning"],
                [LLAMA1B, "Qwen/Qwen2.5-Math-1.5B-Instruct"], "GSM8K search accuracy", 0.258, 0.262, GSM8K_SEARCH,
                [0, 1, 2], control="mismatched-message control"))
    add(finding("f_run5_lfm_tap", "An LFM2-350M-Math tap at layer 8 is the only GSM8K tap above noise",
                "LFM2-350M-Math layer 8 tapped into Llama-1B layer 8 scores 32.7 vs host 25.0",
                "run 5, 3 seeds; it adds nothing on top of LoRA r64 (run 6)", {"tap": 32.7, "host": 25.0},
                "results/round5_rows.json", "candidate", ["math", "gsm8k", "tap"], [LLAMA1B, "LiquidAI/LFM2-350M-Math"],
                "GSM8K search accuracy", 0.327, 0.25, GSM8K_SEARCH, [0, 1, 2], control="host alone"))
    add(finding("f_run6_tap_beats_finetune_dev", "A knowledge tap beats fine-tuning on MMLU (dev items)",
                "Llama-1B + a Qwen2.5-7B tap (layer 20 into host layer 10) scores 67.9 on MMLU dev; a rank-16 LoRA "
                "gains 2.8 points (50.3); the mismatched-message control falls to 48.6",
                "run 6: 500 steps on 8000 MMLU auxiliary_train pairs, 3 seeds",
                {"host": 47.5, "donor": 71.5, "lora_r16": 50.3, "tap_l20_h10": 67.9, "tap_l27_h12": 65.6,
                 "mismatched": 48.6, "joint_tap_lora": 49.2},
                "results/round6_mmlu_rows.json", "candidate", ["knowledge", "mmlu", "tap"], [LLAMA1B, QWEN7B],
                "MMLU dev accuracy", 0.679, 0.503, MMLU_DEV, [0, 1, 2], control="LoRA rank 16"))
    add(finding("f_run6_joint_training_collapses", "Training a tap and a LoRA together collapses the tap",
                "Joint tap plus LoRA training scores 49.2 (the LoRA takes the easy path and the gate stays near 0); "
                "train the tap first, then the LoRA", "run 6, 3 seeds", {"joint": 49.2, "tap_alone": 65.6},
                "results/round6_mmlu_rows.json", "candidate", ["tap", "lora", "training"], [LLAMA1B, QWEN7B],
                "MMLU dev accuracy", 0.492, 0.656, MMLU_DEV, [0, 1, 2], control="tap alone (layer 27)"))
    add(finding("f_run7_heldout_tap", "Held out: a knowledge tap beats fine-tuning by 20.6 points",
                "On 2000 held-out MMLU items Llama-1B + the Qwen2.5-7B layer 20 tap scores 68.3 (3 seeds) vs LoRA r16 "
                "47.7 and host 44.7; the donor alone scores 70.9, so the tap recovers 90% of the gap",
                "run 7: tap settings as run 6, scored once on held-out items",
                {"host": 44.7, "lora_r16": 47.7, "tap": 68.3, "tap_seeds": [67.7, 69.9, 67.2], "donor": 70.9},
                "results/round7_rows.json", "validated", ["knowledge", "mmlu", "tap"], [LLAMA1B, QWEN7B],
                "MMLU held-out accuracy", 0.6827, 0.477, MMLU_HELDOUT, [0, 1, 2], control="LoRA rank 16"))
    add(finding("f_run7_layer_map", "Knowledge is readable near 70% of donor depth",
                "Qwen2.5-7B is readable at layer 20 of 28 (67.1 to 68.5) and gives nothing before layer 16 "
                "(48 to 49 at layers 8 and 12); layer 24 is worse than 20",
                "run 7 layer map: donor layers 8 to 24 by host layers 6, 10, 14, dev items, 1 seed",
                {"L8": [48.5, 48.3, 49.4], "L12": [47.9, 48.6, 48.9], "L16": [63.8, 59.3, 56.3],
                 "L20": [68.5, 67.7, 67.1], "L24": [53.6, 63.9, 60.9]},
                "results/round7_rows.json", "candidate", ["knowledge", "mmlu", "tap", "layers"], [LLAMA1B, QWEN7B],
                "MMLU dev accuracy, donor layer 20 at host layer 6", 0.685, 0.485, MMLU_DEV, [0],
                control="donor layer 8 at host layer 6"))
    add(finding("f_run7_donor_scaling", "A bigger donor gives a better tap, but the tap stays below its donor",
                "Tap means on MMLU dev rise with donor size (58.2, 57.9, 67.7, 70.5 for Qwen2.5 1.5B, 3B, 7B, 14B). "
                "Llama-1B + the Qwen2.5-14B tap (layer 34) scores 70.5 (seeds 73.6 and 67.4), just below Qwen2.5-7B "
                "alone (71.5). results/README.md run 7 prints 73.6 for the 14B tap, which is the seed 0 score only",
                "run 7 donor scaling, tap into host layer 10, 2 seeds",
                {"q1.5b": {"alone": 58.9, "tap": 58.2}, "q3b": {"alone": 66.2, "tap": 57.9},
                 "q7b": {"alone": 71.5, "tap": 67.7}, "q14b": {"alone": 78.5, "tap": 70.5, "tap_seeds": [73.6, 67.4]}},
                "results/round7_rows.json", "candidate", ["knowledge", "mmlu", "tap", "scaling"],
                [LLAMA1B, QWEN14B, QWEN7B], "MMLU dev accuracy, Llama-1B + Qwen-14B tap", 0.705, 0.715, MMLU_DEV,
                [0, 1], control="Qwen2.5-7B alone",
                numbers_source="results/round7_rows.json (the README run 7 table, with the 14B tap mean taken over "
                               "both seeds)"))
    add(finding("f_run7_staged_lora", "Tap first, freeze, then LoRA adds about 1 point",
                "Staged training scores 67.9, 69.2, 69.7 (mean 68.9) vs tap alone 67.9 on MMLU dev",
                "run 7: tap 500 steps, freeze, then LoRA r16 500 steps; the tap-alone control is run 6 "
                "(results/round6_mmlu_rows.json, tap_l20_h10, 3 seeds)", {"staged": 68.9, "tap_alone": 67.9},
                "results/round7_rows.json", "candidate", ["tap", "lora", "training"], [LLAMA1B, QWEN7B],
                "MMLU dev accuracy", 0.689, 0.679, MMLU_DEV, [0, 1, 2], control="tap alone (run 6)"))
    add(finding("f_run7_tap_hurts_generation", "Taps hurt free-form short answers",
                "On TriviaQA the tap scores 25.0, 34.5, 34.3 (mean 31.3) vs host 37.6 and LoRA r16 41.1; it decodes "
                "at host speed", "run 7: 1000 TriviaQA validation questions, alias match on a 16-token answer",
                {"host": 37.6, "donor": 46.8, "lora_r16": 41.1, "tap": [25.0, 34.5, 34.3]},
                "results/round7_rows.json", "candidate", ["knowledge", "triviaqa", "tap", "generation"],
                [LLAMA1B, QWEN7B], "TriviaQA accuracy", 0.3127, 0.376, "TriviaQA validation, 1000 questions",
                [0, 1, 2], control="host alone"))
    add(finding("f_run8_small_hosts_reach_donors", "Small hosts reach their donors through one tap",
                "Llama-1B and Qwen-0.5B with a Gemma-2-9B tap match Gemma-9B alone (72.9 and 72.7 vs 72.6); "
                "Qwen-0.5B with a Qwen-14B tap scores 74.8, above every 7B to 9B model, at 0.5B decode cost",
                "run 8: MMLU dev, 2 seeds, read about 70% depth",
                {"llama1b_gemma9b": 72.9, "qwen05_gemma9b": 72.7, "gemma9b_alone": 72.6, "qwen05_qwen14b": 74.8,
                 "qwen14b_alone": 78.5},
                "results/round8_rows.json", "candidate", ["knowledge", "mmlu", "tap"], [LLAMA1B, QWEN05, GEMMA9B,
                                                                                        QWEN14B],
                "MMLU dev accuracy, Qwen-0.5B + Qwen-14B tap", 0.748, 0.48, MMLU_DEV, [0, 1], control="host alone"))
    add(finding("f_run8_two_taps_not_additive", "Two taps into a weak host are not additive",
                "Two-tap pairs score near the better single tap: Qwen-14B + Gemma-9B 73.7 vs Gemma-9B 72.9 (+0.8), "
                "Qwen-7B + Gemma-9B 72.0 vs 72.9 (-0.9), Gemma-9B + Llama-8B 70.7 vs 72.9 (-2.2), Qwen-7B + "
                "Llama-8B 64.0 vs Qwen-7B 66.7 (-2.7)",
                "run 8: Llama-1B host, 2 seeds", {"qwen14b+gemma9b": 73.7, "qwen7b+gemma9b": 72.0,
                                                  "gemma9b+llama8b": 70.7, "qwen7b+llama8b": 64.0},
                "results/round8_rows.json", "candidate", ["knowledge", "mmlu", "tap", "composition"], [LLAMA1B],
                "MMLU dev accuracy, Qwen-14B + Gemma-9B taps", 0.737, 0.729, MMLU_DEV, [0, 1],
                control="best single tap (Gemma-9B)"))
    add(finding("f_run8_weak_donors", "Some donors are weak or unstable at their read layer",
                "Llama-3.1-8B at layer 22 (seeds 49.2, 58.0) and LFM2.5-1.2B at layer 11 (49.9) are weak donors in "
                "this setup", "run 8", {"llama8b_tap": [49.2, 58.0], "lfm12b_tap": 49.9, "host": 47.5},
                "results/round8_rows.json", "candidate", ["knowledge", "mmlu", "tap"],
                [LLAMA1B, "unsloth/Meta-Llama-3.1-8B-Instruct", "LiquidAI/LFM2.5-1.2B-Instruct"],
                "MMLU dev accuracy, Llama-8B tap", 0.536, 0.475, MMLU_DEV, [0, 1], control="host alone"))
    add(finding("f_baselines_full_ft", "Full fine-tuning is worse than LoRA r64 at 300 steps",
                "Llama-1B full fine-tuning scores 33.5 (lr 1e-5) and 13.8 (lr 3e-5) on GSM8K search vs 47.2 for LoRA "
                "r64; Qwen-0.5B 41.2 and 21.2", "baselines: 300 steps, batch 8, seq 384, 3 seeds, wsp_eval harness",
                {"llama1b_lr1e-5": 33.5, "llama1b_lr3e-5": 13.8, "qwen05_lr1e-5": 41.2, "qwen05_lr3e-5": 21.2},
                "results/baselines_rows.json", "candidate", ["baseline", "gsm8k", "fine-tuning"], [LLAMA1B, QWEN05],
                "GSM8K search accuracy", 0.335, 0.472, GSM8K_SEARCH, [0, 1, 2], control="LoRA rank 64 (run 5)"))
    add(finding("f_baselines_raw_prompt", "Chat-tuned donors score badly on the raw prompt",
                "Donor ceilings with the raw prompt: Llama-3.2-3B 59.0, LFM2.5-1.2B-Instruct 38.5, Qwen-Math-1.5B-"
                "Instruct 32.0, LFM2.5-1.2B-Thinking 27.5, Qwen2.5-1.5B-Instruct 7.0, LFM2-350M-Math 1.0",
                "baselines, no training", {"llama3b": 59.0, "lfm12b": 38.5, "qmath15_instruct": 32.0,
                                           "lfm12b_thinking": 27.5, "qwen15b": 7.0, "lfm350m_math": 1.0},
                "results/baselines_rows.json", "candidate", ["baseline", "gsm8k", "prompt"], [],
                "GSM8K search accuracy, Qwen2.5-1.5B-Instruct", 0.07, None, GSM8K_SEARCH, None))
    add(finding("f_run9_strong_host", "Taps into a strong host do not help",
                "Qwen2.5-14B alone scores 78.5 on MMLU dev; with taps 75.6 to 77.9, the mismatched control 76.0, and "
                "LoRA r16 78.1. No composition beats its best source",
                "run 9: 2 seeds, read about 70% depth, write about 40%",
                {"q14b_alone": 78.5, "q14b_llama8b": 77.9, "q14b_q7b": 77.2, "q14b_gemma9b": 75.6, "derange": 76.0,
                 "lora": 78.1, "gemma9b_alone": 72.6, "gemma9b_q14b": 73.0},
                "results/round9_rows.json", "candidate", ["knowledge", "mmlu", "tap", "composition"], [QWEN14B, GEMMA9B],
                "MMLU dev accuracy, best tap on Qwen-14B", 0.779, 0.785, MMLU_DEV, [0, 1], control="host alone"))
    add(finding("f_stitch_donor_lm", "A stitch trained on the donor's own LM loss keeps the donor intact",
                "Training the stitch on the donor's next-token loss cuts the penalty from 0.40 to 0.65 nats to 0.03 to "
                "0.09 at the early cut for same-tokenizer hosts",
                "stitch study: donor Qwen2.5-Math-1.5B, 1000 steps, sink on, 150 held-out texts per domain",
                {"qwen15b_7_8": {"ridge": 0.399, "donor_lm_linear": 0.029, "donor_lm_mlp": -0.014},
                 "qwen05_7_8": {"ridge": 0.647, "donor_lm_linear": 0.090, "donor_lm_mlp": 0.078}},
                "results/stitch_study_rows.json", "validated", ["stitch"], [QMATH, "Qwen/Qwen2.5-1.5B", QWEN05],
                "GSM8K penalty (nats), Qwen-0.5B 7 to 8, donor_lm linear", 0.090, 0.647, "150 held-out GSM8K heal texts",
                [0], control="ridge fit"))
    add(finding("f_stitch_mse_no_gain", "MSE regression does nothing beyond ridge",
                "An MSE-trained linear stitch leaves the penalty at the ridge level (0.399 vs 0.399)",
                "stitch study, Qwen2.5-1.5B host, 7 to 8", {"ridge": 0.399, "mse_linear": 0.399},
                "results/stitch_study_rows.json", "validated", ["stitch"], [QMATH, "Qwen/Qwen2.5-1.5B"],
                "GSM8K penalty (nats)", 0.399, 0.399, "150 held-out GSM8K heal texts", [0], control="ridge fit"))
    add(finding("f_stitch_negative_wiki", "A stitched composite predicts held-out text better than the donor",
                "Host layers 0..k plus donor layers l..end beat the donor itself on held-out WikiText by 0.1 to 0.2 "
                "nats (negative penalties)", "stitch study, donor_lm stitches",
                {"qwen15b_7_8_mlp": -0.207, "qwen15b_7_8_linear": -0.164, "qwen05_7_8_mlp": -0.164},
                "results/stitch_study_rows.json", "validated", ["stitch", "composition"], [QMATH, "Qwen/Qwen2.5-1.5B",
                                                                                          QWEN05],
                "WikiText penalty (nats), Qwen2.5-1.5B 7 to 8 donor_lm mlp", -0.207, 0.182,
                "150 held-out WikiText-103 paragraphs", [0], control="ridge fit"))
    add(finding("f_stitch_cross_tokenizer", "Across tokenizers the downstream objective closes a third to a half",
                "With a Llama-1B host, a downstream MLP stitch cuts the downstream loss from 0.404 to 0.250 (5 to 8) "
                "and 0.515 to 0.360 (9 to 14); position mismatch remains",
                "stitch study, cross-family host", {"5_to_8": {"ridge": 0.404, "downstream_mlp": 0.250},
                                                    "9_to_14": {"ridge": 0.515, "downstream_mlp": 0.360}},
                "results/stitch_study_rows.json", "validated", ["stitch", "cross-tokenizer"], [LLAMA1B, QMATH],
                "downstream loss at donor layer l + 4", 0.250, 0.404, "150 held-out GSM8K heal texts", [0],
                control="ridge fit"))
    add(finding("f_run15_carving_half_destroys", "Carving more than half the layers destroys a model",
                "Qwen2.5-14B carved to 22 of 48 layers scores 33.6 MMLU dev at best (official Qwen2.5-7B 71.5); "
                "Qwen2.5-32B carved to 27 of 64 scores 40.3 and 41.2 (official Qwen2.5-14B 78.5). A short heal "
                "recovers little; with no heal they are near chance",
                "run 15: keep N layers (first and last always) by block influence or uniformly; LoRA r64 heal 1000 "
                "steps on WikiText plus MMLU aux plus GSM8K text, merged",
                {"official_q7b": [71.5, 70.9], "q14b_uniform_lora": [33.6, 35.8], "q14b_influence_lora": [25.5, 25.4],
                 "q14b_no_heal": [21.9, 23.3], "official_q14b": [78.5, 77.9], "q32b_influence_lora": [40.3, 41.2],
                 "q32b_uniform_lora": [36.0, 36.2], "official_q32b": [82.9, 79.5]},
                "results/round15_rows.json", "validated", ["carve", "written-model"], [QWEN7B, QWEN14B, QWEN32B],
                "MMLU held-out accuracy, 14B carved to 7.6B (uniform, LoRA)", 0.358, 0.7085, MMLU_HELDOUT, [0],
                control="official Qwen2.5-7B-Instruct at the same size"))
    add(finding("f_run15_influence_vs_uniform", "Ranking layers by block influence is no better than uniform",
                "At matched size, influence-ranked carves score 25.5 (14B) and 40.3 to 41.2 (32B) MMLU dev; uniform "
                "carves score 33.6 and 36.0", "run 15", {"q14b": {"influence": 25.5, "uniform": 33.6},
                                                         "q32b": {"influence": [40.3, 41.2], "uniform": 36.0}},
                "results/round15_rows.json", "validated", ["carve"], [QWEN14B, QWEN32B],
                "MMLU held-out accuracy, 14B influence vs uniform", 0.254, 0.358, MMLU_HELDOUT, [0],
                control="uniform layer selection"))
    add(finding("f_run14_reader_needed", "A donor block written into a 0.5B base does not beat a random block",
                "Qwen2.5-7B layers 8..20 grafted into Qwen2.5-0.5B-Instruct at layer 7 score 45.9 and 47.0 on "
                "held-out MMLU; a random same-length Qwen-7B block scores 45.4 and 49.5, the base 45.8, and a LoRA "
                "r16 45.1 and 45.9. Qwen2.5-14B layers 14..34 add about 3 points (49.4, 48.6) from 6B parameters "
                "of a model that scores 78. The trained stitch is near lossless (+0.008 to +0.024 nats for 7B, "
                "-0.02 for 14B), so the block computes what it does in its own model; the base's later layers "
                "cannot read its output when only the seam glue trains",
                "run 14: one network, residual block graft h + W_out(block(W_in h) - x) at host layer 7, MLP stitch "
                "trained 800 steps on the donor's own LM loss with the donor sink, glue rank 64, heal 500 steps on "
                "MMLU aux text (glue, plus a rank-8 part LoRA where noted), 2 seeds",
                {"base": [48.0, 45.8], "lora_r16_heldout": [45.1, 45.9], "q7b_8_20_heldout": [45.9, 47.0],
                 "q7b_part_lora_heldout": [47.1, 46.7], "q7b_random_heldout": [45.4, 49.5],
                 "q14b_14_34_heldout": [49.4, 48.6], "q14b_part_lora_heldout": [50.0, 47.4],
                 "stitch_penalty_after": {"q7b": 0.008, "q7b_part_lora": 0.024, "q7b_random": 0.182, "q14b": -0.020}},
                "results/round14_rows.json", "validated", ["graft", "block", "written-model", "knowledge", "mmlu"],
                [QWEN05, QWEN7B, QWEN14B], "MMLU held-out accuracy, Qwen-7B block 8..20, mean over seeds", 0.4645,
                0.4748, MMLU_HELDOUT, [0, 1], control="random same-length Qwen-7B block, same stitch and heal"))
    add(finding("f_round10_stacker", "A learned stacker adds 1.2 to 1.6 points over the best model",
                "Output-level combining of several models: probability average +0.5 to +1.3 over the best model, a "
                "learned stacker +1.2 on MMLU and +1.6 on MedMCQA; oracle coverage 90 to 94%",
                "round 10 per-item scores, as recorded in wsp/rules.py (the stacker analysis rows are not in results/)",
                {"stacker_mmlu": 1.2, "stacker_medmcqa": 1.6, "average_gain": [0.5, 1.3], "oracle_coverage": [90, 94]},
                "wsp/rules.py", "candidate", ["ensemble", "mmlu", "medmcqa"], [], "points over the best model", 1.2,
                0.0, "round 10 items", None, control="best single model", numbers_source="wsp/rules.py"))
    add(finding("f_part_gate_search_not_heldout", "A search-split gain over random did not hold out",
                "The Phi-4 layer 16 GSM8K cluster beat its random control by 2.8 points on the search split (3 seeds) "
                "but lost by 0.6 on held-out GSM8K (31.4 vs 32.0); a matched LoRA scored 37.1",
                "Person B part gate and final held-out run: Llama-1B host, host layer 6, rank-64 glue, 300 heal steps",
                {"search": {"graft": 38.5, "random": 35.7}, "heldout": {"graft": 31.4, "random": 32.0, "lora": 37.1,
                                                                       "host": 20.1}},
                "results/final_heldout_microsoft__phi-4.json", "validated", ["math", "gsm8k", "graft", "method"],
                [LLAMA1B, "microsoft/phi-4"], "held-out GSM8K accuracy", 0.3139, 0.3199, GSM8K_HELDOUT, [0, 1, 2],
                control="random same-size cluster", added_by=PERSON_B, numbers_source="wsp_eval/README.md"))
    add(finding("f_gate_hour8_contrast", "A Cohen's d contrast cluster ties its random control",
                "Qwen-Math layer 10, 512 neurons by GSM8K contrast, grafted at Llama-1B layer 6: 41.0 vs random 40.3 "
                "(lost or tied on 2 seeds); a matched LoRA scored 44.5",
                "hour-8 gate, wsp_eval harness, 3 seeds", {"graft": 41.0, "random": 40.3, "lora": 44.5, "host": 23.0,
                                                          "donor": 45.0},
                "results/gate_hour8_3seeds.json", "candidate", ["math", "gsm8k", "graft"], [LLAMA1B, QMATH],
                "GSM8K search accuracy", 0.41, 0.4033, GSM8K_SEARCH, [0, 1, 2], control="random same-size cluster",
                added_by=PERSON_B, numbers_source="wsp_eval/README.md"))
    add(finding("f_bake_no_gain", "Baking the tap into the host keeps almost none of its gain",
                "Distilling the tapped teacher (67.7) into a bare Llama-1B LoRA scores 48.75 held-out, the same as "
                "distilling Qwen-7B directly (48.75) and as a LoRA on gold letters (48.6); host 44.65",
                "bake (modal_bake.py): 20557 MMLU auxiliary_train prompts (24000 requested), rank-16 host LoRA, lr 1e-4, "
                "batch 16, 1500 steps, 1 seed; numbers are [dev, held-out]",
                {"host": [47.5, 44.65], "teacher": 67.7, "bake_tap": [51.2, 48.75], "bake_donor": [50.2, 48.75],
                 "lora_label": [50.3, 48.6]},
                "results/bake_rows.json", "validated", ["knowledge", "mmlu", "bake"], [LLAMA1B, QWEN7B],
                "MMLU held-out accuracy", 0.4875, 0.4875, MMLU_HELDOUT, [0],
                control="distill the donor directly (bake_donor)", added_by=PERSON_B,
                numbers_source="results/bake_rows.json"))
    add(finding("f_stitch_graft_ties_random", "A trained stitch does not make a block graft beat a random block",
                "Qwen-Math layers 8..19 grafted into Qwen-0.5B at layer 7 with a donor_lm stitch: 38.0 on GSM8K search "
                "vs 37.5 for a random block with the same stitch and 36.5 with ridge; a matched LoRA scored 45.5",
                "stitch graft: 1 seed, 300 heal steps", {"host": 34.5, "block_ridge": 36.5, "block_stitch": 38.0,
                                                         "random_stitch": 37.5, "lora": 45.5},
                "results/stitch_graft_s0.json", "candidate", ["graft", "stitch", "block"], [QWEN05, QMATH],
                "GSM8K search accuracy", 0.38, 0.375, GSM8K_SEARCH, [0], control="random same-length block",
                added_by=PERSON_B, numbers_source="results/stitch_graft_s0.json"))
    add(finding("f_exit_stitch_no_gain", "Distilling the donor through the graft glue does not help",
                "Training the stitched Qwen-Math block graft (Qwen-0.5B host, layer 7) for 1000 more steps to match "
                "Qwen2.5-Math-1.5B's next-token distribution scores 38.5 on GSM8K search; the random block with the "
                "same training scores 41.0 and a LoRA of about the same size 37.5",
                "exit stitch (wsp_eval/exit_stitch.py): starts from the stitch graft models, trains the 1.53M glue "
                "(or a fresh LoRA) on full-vocab KL to the donor over GSM8K heal texts, lr 3e-4, batch 8, seq 256, "
                "1 seed. The random control drew donor layers 9..20, a weak control",
                {"graft": 38.5, "random": 41.0, "lora": 37.5},
                "results/exit_stitch_s0.json", "candidate", ["graft", "stitch", "block", "distill"], [QWEN05, QMATH],
                "GSM8K search accuracy", 0.385, 0.41, GSM8K_SEARCH, [0], control="random same-length block",
                added_by=PERSON_B, numbers_source="results/exit_stitch_s0.json"))
    add(finding("f_contrast_is_selection", "Contrast effect sizes select parts; they do not show function",
                "Cohen's d between GSM8K and WikiText activations ranks units, but ablation checks on Qwen2.5-Math did "
                "not pass as specific", "Person B extraction (parts_extracted/README.md, wsp_eval/README.md)", {},
                "parts_extracted/README.md", "candidate", ["extraction", "method"], [QMATH, "microsoft/phi-4",
                                                                                     "Nanbeige/Nanbeige4.3-33B"],
                "ablation specificity", None, None, "GSM8K heal split vs WikiText-2", None, added_by=PERSON_B,
                numbers_source="parts_extracted/README.md"))
    for item in complementarity_finding(repo):
        add(item)
    apply_exact_values(out, exact_finding_values(repo))
    return out


# Stitches


def stitch(stitch_id, title, status, host, donor, host_layers, donor_layer, stitch_kind, objective, steps, sink,
           penalty, items, source, extra):
    fields = {"host": host, "donor": donor, "host_layers": host_layers, "donor_layer": donor_layer,
              "stitch_kind": stitch_kind, "objective": objective, "steps": steps, "sink": sink, "penalty": penalty}
    for key in extra:
        fields[key] = extra[key]
    models = [host]
    if donor != host:
        models.append(donor)
    return entry(stitch_id, "stitch", title, status, TRISANTH, source, ["stitch"], models, items, fields)


def pick(block, key):
    if block is None:
        return None
    value = block.get(key)
    if value is None:
        return None
    return round(value, 4)


def build_stitches(repo):
    out = []
    path = os.path.join(repo, "results", "stitch_study_rows.json")
    if os.path.exists(path):
        data = read_json(path)
        for row in data["rows"]:
            before = row.get("before") or {}
            after = row.get("after") or {}
            penalty = {"before": {}, "after": {}}
            for domain in ["gsm8k", "wiki"]:
                for side, block in [("before", before.get(domain)), ("after", after.get(domain))]:
                    penalty[side][domain] = {"penalty": pick(block, "penalty"),
                                             "penalty_no_sink": pick(block, "penalty_no_sink"),
                                             "downstream_loss": pick(block, "downstream_loss"),
                                             "r2": pick(block, "r2")}
            metric = "penalty"
            if not row.get("same_tokenizer"):
                metric = "downstream_loss"
            value = penalty["after"]["gsm8k"][metric]
            control_value = penalty["before"]["gsm8k"][metric]
            items = [evidence("trained stitch vs the ridge fit it starts from", "GSM8K " + metric + " (nats) after vs "
                              "before training", value, control_value, "150 held-out GSM8K heal texts", [row["seed"]],
                              "results/stitch_study_rows.json", control="ridge fit (before training)",
                              note="WikiText " + metric + " after " + str(penalty["after"]["wiki"][metric]) +
                                   ", before " + str(penalty["before"]["wiki"][metric]))]
            status = "validated"
            reason = None
            if row["method"] == "ridge":
                status = "validated"
                reason = "the ridge baseline, measured with the sink"
            elif value is not None and control_value is not None and value >= control_value - 0.005:
                status = "rejected"
                reason = "training did not lower the " + metric + " below the ridge fit"
            extra = {"method": row["method"], "lr": row.get("lr"), "batch_size": row.get("batch_size"),
                     "seq_len": row.get("seq_len"), "params": row.get("params"), "end_layer": row.get("end_layer"),
                     "same_tokenizer": row.get("same_tokenizer"), "ridge_fit": row.get("ridge_fit"),
                     "train_seconds": row.get("train_seconds"), "sink_mode": row.get("sink_mode"),
                     "data": "1200 GSM8K heal texts + 1200 WikiText-103 paragraphs, interleaved"}
            if reason is not None:
                extra["status_reason"] = reason
            host_layers = []
            for layer in row["host_layers"]:
                host_layers.append(str(layer))
            title = (short_model(row["host"]) + " L" + "+".join(host_layers) + " to " +
                     short_model(row["donor"]) + " L" + str(row["donor_layer"]) + ", " + row["method"])
            out.append(stitch("stitch_" + row["cell"], title, status, row["host"], row["donor"], row["host_layers"],
                              row["donor_layer"], row["kind"], row.get("objective"), row.get("steps"), True, penalty,
                              items, "results/stitch_study_rows.json (stitch study)", extra))

    path = os.path.join(repo, "results", "audit_sink_rows.json")
    if os.path.exists(path):
        for row in read_json(path):
            variants = row["variants"]
            no_sink = {}
            for name in variants:
                if "+sink" not in name and variants[name].get("penalty") is not None:
                    no_sink[name] = variants[name]["penalty"]
            for name in variants:
                variant = variants[name]
                if variant.get("penalty") is None:
                    continue
                sink = "+sink" in name
                base = name.split("+")[0]
                control_value = None
                status = "candidate"
                reason = "ridge stitch without the donor sink"
                if sink and base in no_sink:
                    control_value = round(no_sink[base], 4)
                    status = "validated"
                    reason = "the sink cut the penalty against the same ridge without it"
                if name.startswith("current") and not sink:
                    status = "rejected"
                    reason = "the stitch as shipped before the audit: garbage sink and lam 1e-2"
                items = [evidence("ridge stitch variant " + name, "stitch penalty (nats)", round(variant["penalty"], 4),
                                  control_value, "150 GSM8K texts (run 4 eval texts)", None,
                                  "results/audit_sink_rows.json", control="same ridge without the sink")]
                tag = row["pair"].split(":")[0].strip().replace(" ", "_").lower()
                stitch_id = "audit_" + re.sub(r"[^a-z0-9_.\-]", "_", tag + "_" + name.lower())
                penalty = {"after": {"gsm8k": {"penalty": round(variant["penalty"], 4), "r2": pick(variant, "r2"),
                                               "r2_dim": pick(variant, "r2_dim")}}}
                out.append(stitch(stitch_id, "audit: " + row["pair"] + ", " + name, status, row["host"], QMATH,
                                  [row["host_layer"]], row["donor_layer"], "linear", "ridge", 0, sink, penalty, items,
                                  "results/audit_sink_rows.json (audit)",
                                  {"method": "ridge " + name, "native_loss": round(row["native_loss"], 4),
                                   "status_reason": reason, "pair": row["pair"]}))

    path = os.path.join(repo, "results", "round4_rows.json")
    if os.path.exists(path):
        data = read_json(path)["stitch_penalty"]
        for pair in data["pairs"]:
            items = [evidence("ridge stitch with no sink, run 4", "stitch penalty (nats)", round(pair["penalty"], 4),
                              None, "150 GSM8K texts", None, "results/round4_rows.json")]
            penalty = {"after": {"gsm8k": {"penalty": round(pair["penalty"], 4), "r2": round(pair["r2"], 4),
                                           "r2_dim": round(pair["r2_dim"], 4)}}}
            out.append(stitch("r4_qwen05_h" + str(pair["host_layer"]) + "_d" + str(pair["donor_layer"]) + "_ridge",
                              "run 4: Qwen2.5-0.5B-Instruct L" + str(pair["host_layer"]) + " to Qwen2.5-Math-1.5B L" +
                              str(pair["donor_layer"]) + ", ridge, no sink", "rejected", QWEN05, QMATH,
                              [pair["host_layer"]], pair["donor_layer"], "linear", "ridge", 0, False, penalty, items,
                              "results/round4_rows.json (run 4)",
                              {"method": "ridge", "native_loss": round(data["native_loss"], 4),
                               "status_reason": "fed the donor a garbage attention sink (audit); penalty 1.4 to 2.2 nats"}))
    return out


# Recipes


def make_program(host, seed=0, grafts=None, lora=None, heal=None, taps=None, tap_train=None, base_source=None,
                 carve=None):
    program = {"host": host, "seed": seed, "masks": None, "task_vectors": [], "grafts": grafts or [], "heal": heal}
    if lora is not None:
        program["lora"] = lora
    if taps is not None:
        program["taps"] = taps
    if tap_train is not None:
        program["tap_train"] = tap_train
    if base_source is not None:
        program["base_source"] = {"model": base_source}
    if carve is not None:
        program["carve"] = carve
    return program


def lora_spec(rank):
    return {"rank": rank, "budget_params": None, "targets": None, "alpha": None, "layers": None}


def heal_spec(steps, data, lr, batch_size, seq_len):
    return {"steps": steps, "data": data, "lr": lr, "batch_size": batch_size, "seq_len": seq_len}


def tap_spec(donor, layer, at, tap_id=None):
    return {"donor": donor, "donor_layer": layer, "at": at, "attn_dim": 256, "heads": 8, "tap_id": tap_id}


def graft_spec(graft_id, at, glue_rank=64, control=None, control_seed=0, stitch=None):
    data = {"id": graft_id, "at": at, "control": control, "control_seed": control_seed, "glue_rank": glue_rank}
    if stitch is not None:
        data["stitch"] = stitch
    return data


TAP_TRAIN_MMLU = {"data": "mmlu_aux", "steps": 500, "batch_size": 8, "seq_len": 256, "branch_lr": 0.0003,
                  "gate_lr": 0.03}
TAP_TRAIN_GSM8K = {"data": "gsm8k_train", "steps": 300, "batch_size": 8, "seq_len": 256, "branch_lr": 0.0003,
                   "gate_lr": 0.03}
HEAL_GSM8K = heal_spec(300, "gsm8k_train", 0.0003, 8, 384)


def format_text(program):
    """Return (text, problem). text is None when wsp.lang cannot write the program."""
    try:
        if REPO not in sys.path:
            sys.path.insert(0, REPO)
        from wsp.lang import format_program
        return format_program(program), None
    except Exception as error:
        return None, "wsp.lang.format_program failed: " + str(error)[:200]


def recipe(recipe_id, title, status, host, program, items, source, skills, models, reason=None, syntax_gap=None,
           components=None, added_by=TRISANTH):
    text = None
    problem = syntax_gap
    if program is not None and syntax_gap is None:
        text, problem = format_text(program)
    program_file = None
    if text is not None:
        program_file = "recipes/" + recipe_id + ".wsp"
    fields = {"host": host, "program": program, "text": text, "program_file": program_file, "syntax_gap": problem,
              "components": components or []}
    if reason is not None:
        fields["status_reason"] = reason
    return entry(recipe_id, "recipe", title, status, added_by, source, skills, models, items, fields)


def seeds_and_mean(rows, key="accuracy"):
    values = []
    seeds = []
    for row in rows:
        values.append(row[key])
        seeds.append(row["seed"])
    return pct(mean(values)), sorted(seeds)


def rows_where(rows, **match):
    out = []
    for row in rows:
        keep = True
        for key in match:
            if row.get(key) != match[key]:
                keep = False
        if keep:
            out.append(row)
    return out


def build_recipes(repo):
    out = []
    results = os.path.join(repo, "results")

    # Held-out MMLU tap (runs 6 and 7).
    path = os.path.join(results, "round7_rows.json")
    if os.path.exists(path):
        rows = read_json(path)
        tap_value, tap_seeds = seeds_and_mean(rows_where(rows, part="heldout", arm="tap"))
        lora_value, _ = seeds_and_mean(rows_where(rows, part="heldout", arm="lora_r16"))
        host_value, _ = seeds_and_mean(rows_where(rows, part="heldout", arm="host"))
        program = make_program(LLAMA1B, taps=[tap_spec(QWEN7B, 20, 10)], tap_train=dict(TAP_TRAIN_MMLU))
        items = [evidence("Llama-1B + Qwen2.5-7B tap vs a rank-16 LoRA", "MMLU held-out accuracy, mean over seeds",
                          tap_value, lora_value, MMLU_HELDOUT, tap_seeds, "results/round7_rows.json",
                          control="host LoRA rank 16, same data", note="host alone " + str(host_value) +
                                                                       ", donor alone 0.7085")]
        out.append(recipe("tap_mmlu_llama1b_qwen7b_l20_h10", "Llama-1B reads Qwen2.5-7B layer 20 at layer 10 (MMLU)",
                          "validated", LLAMA1B, program, items, "results/round7_rows.json (run 7)",
                          ["knowledge", "mmlu", "tap"], [LLAMA1B, QWEN7B], components=["block_qwen25_7b_l16_20_knowledge"]))
        staged_value, staged_seeds = seeds_and_mean(rows_where(rows, part="staged", arm="staged"))
        program = make_program(LLAMA1B, taps=[tap_spec(QWEN7B, 20, 10)], tap_train=dict(TAP_TRAIN_MMLU),
                               lora=lora_spec(16), heal=heal_spec(500, "mmlu_aux", 0.0001, 8, 256))
        tap_alone = None
        for row in rows_where(rows, part="layers", arm="tap"):
            if row["cfg"].get("donor_layer") == 20 and row["cfg"].get("host_layer") == 10:
                tap_alone = row["accuracy"]
        items = [evidence("staged tap then LoRA r16 vs the tap alone", "MMLU dev accuracy, mean over seeds",
                          staged_value, pct(tap_alone), MMLU_DEV, staged_seeds, "results/round7_rows.json",
                          control="tap alone (run 7 layer map, donor layer 20 at host layer 10, seed 0)")]
        out.append(recipe("tap_then_lora_mmlu_llama1b_qwen7b", "Tap first, freeze, then a rank-16 LoRA (MMLU)",
                          "candidate", LLAMA1B, program, items, "results/round7_rows.json (run 7)",
                          ["knowledge", "mmlu", "tap", "lora"], [LLAMA1B, QWEN7B],
                          reason="dev items only; +1.0 over the tap alone"))
        lora_rows = rows_where(rows, part="heldout", arm="lora_r16")
        program = make_program(LLAMA1B, lora=lora_spec(16), heal=heal_spec(500, "mmlu_aux", 0.0001, 8, 256))
        items = [evidence("Llama-1B LoRA r16 on MMLU aux vs the host alone", "MMLU held-out accuracy, mean over seeds",
                          lora_value, host_value, MMLU_HELDOUT, seeds_and_mean(lora_rows)[1], "results/round7_rows.json",
                          control="host alone")]
        out.append(recipe("lora_r16_mmlu_llama1b", "Llama-1B rank-16 LoRA on MMLU aux (fine-tuning baseline)",
                          "validated", LLAMA1B, program, items, "results/round7_rows.json (run 7)",
                          ["knowledge", "mmlu", "lora", "baseline"], [LLAMA1B]))

    # Run 8 single and double taps.
    path = os.path.join(results, "round8_rows.json")
    if os.path.exists(path):
        rows = read_json(path)
        host_scores = {}
        for row in rows_where(rows, kind="host_alone"):
            host_scores[row["key"]] = row["accuracy"]
        plans = [("tap_mmlu_qwen05_qwen14b_l34_h15", "tap_qwen05", "qwen14b", QWEN05, 15, "qwen05"),
                 ("tap_mmlu_qwen05_gemma9b_l29_h15", "tap_qwen05", "gemma9b", QWEN05, 15, "qwen05"),
                 ("tap_mmlu_llama1b_gemma9b_l29_h10", "tap_llama1b", "gemma9b", LLAMA1B, 10, "llama1b"),
                 ("tap_mmlu_llama1b_qwen14b_l34_h10", "tap_llama1b", "qwen14b", LLAMA1B, 10, "llama1b")]
        for recipe_id, kind, key, host, at, host_key in plans:
            value, seeds = seeds_and_mean(rows_where(rows, kind=kind, key=key))
            if value is None:
                continue
            donor = RESULT_KEYS[key]
            program = make_program(host, taps=[tap_spec(donor, ROUND8_READ[key], at)], tap_train=dict(TAP_TRAIN_MMLU))
            items = [evidence(short_model(host) + " + " + short_model(donor) + " tap vs the host alone",
                              "MMLU dev accuracy, mean over seeds", value, pct(host_scores.get(host_key)), MMLU_DEV,
                              seeds, "results/round8_rows.json", control="host alone")]
            out.append(recipe(recipe_id, short_model(host) + " reads " + short_model(donor) + " layer " +
                              str(ROUND8_READ[key]) + " at layer " + str(at) + " (MMLU)", "candidate", host, program,
                              items, "results/round8_rows.json (run 8)", ["knowledge", "mmlu", "tap"], [host, donor],
                              reason="dev items only; not scored on held-out items"))
        value, seeds = seeds_and_mean(rows_where(rows, kind="two_taps_llama1b", key="qwen14b+gemma9b"))
        if value is not None:
            program = make_program(LLAMA1B, taps=[tap_spec(QWEN14B, 34, 10), tap_spec(GEMMA9B, 29, 12)],
                                   tap_train=dict(TAP_TRAIN_MMLU))
            gemma_value, _ = seeds_and_mean(rows_where(rows, kind="tap_llama1b", key="gemma9b"))
            items = [evidence("two taps vs the better single tap (Gemma-9B)", "MMLU dev accuracy, mean over seeds",
                              value, gemma_value, MMLU_DEV, seeds, "results/round8_rows.json",
                              control="best single tap",
                              note="second tap at host layer + 2, as modal_round8.py places it")]
            out.append(recipe("two_taps_mmlu_llama1b_qwen14b_gemma9b", "Llama-1B reads Qwen2.5-14B and Gemma-2-9B (MMLU)",
                              "candidate", LLAMA1B, program, items, "results/round8_rows.json (run 8)",
                              ["knowledge", "mmlu", "tap"], [LLAMA1B, QWEN14B, GEMMA9B],
                              reason="near the better single tap; dev items only"))

    # GSM8K: LFM tap, LoRA bars, grafts.
    path = os.path.join(results, "round5_rows.json")
    if os.path.exists(path):
        rows = read_json(path)["rows"]
        value, seeds = seeds_and_mean(rows_where(rows, arm="tap_lfm_l8_h8"))
        host_value, _ = seeds_and_mean(rows_where(rows, arm="host_only"))
        program = make_program(LLAMA1B, taps=[tap_spec("LiquidAI/LFM2-350M-Math", 8, 8)],
                               tap_train=dict(TAP_TRAIN_GSM8K))
        items = [evidence("LFM2-350M-Math tap vs the host alone", "GSM8K search accuracy, mean over seeds", value,
                          host_value, GSM8K_SEARCH, seeds, "results/round5_rows.json", control="host alone")]
        out.append(recipe("tap_gsm8k_llama1b_lfm350m_l8_h8", "Llama-1B reads LFM2-350M-Math layer 8 at layer 8 (GSM8K)",
                          "candidate", LLAMA1B, program, items, "results/round5_rows.json (run 5)",
                          ["math", "gsm8k", "tap"], [LLAMA1B, "LiquidAI/LFM2-350M-Math"],
                          components=["tapread_lfm2_350m_math_l8"],
                          reason="search split only; adds nothing on top of LoRA r64 (run 6)"))
        value, seeds = seeds_and_mean(rows_where(rows, arm="lora_r64"))
        program = make_program(LLAMA1B, lora=lora_spec(64), heal=heal_spec(300, "gsm8k_train", 5e-05, 8, 384))
        items = [evidence("Llama-1B LoRA r64 lr 5e-5 vs the host alone", "GSM8K search accuracy, mean over seeds",
                          value, host_value, GSM8K_SEARCH, seeds, "results/round5_rows.json", control="host alone")]
        out.append(recipe("lora_r64_gsm8k_llama1b", "Llama-1B rank-64 LoRA, lr 5e-5 (the GSM8K fine-tuning bar)",
                          "candidate", LLAMA1B, program, items, "results/round5_rows.json (run 5)",
                          ["math", "gsm8k", "lora", "baseline"], [LLAMA1B], reason="search split only"))
    path = os.path.join(results, "round3_rows.json")
    if os.path.exists(path):
        rows = read_json(path)
        value, seeds = seeds_and_mean(rows_where(rows, arm="lora_matched"))
        block_value, _ = seeds_and_mean(rows_where(rows, arm="block_a_r64"))
        random_value, _ = seeds_and_mean(rows_where(rows, arm="block_rand"))
        program = make_program(LLAMA1B, lora=lora_spec(1), heal=dict(HEAL_GSM8K))
        items = [evidence("rank-1 LoRA vs the best block graft", "GSM8K search accuracy, mean over seeds", value,
                          block_value, GSM8K_SEARCH, seeds, "results/round3_rows.json", control="block graft 8..19")]
        out.append(recipe("lora_r1_gsm8k_llama1b", "Llama-1B rank-1 LoRA (0.70M params) on GSM8K", "candidate", LLAMA1B,
                          program, items, "results/round3_rows.json (run 3)", ["math", "gsm8k", "lora", "baseline"],
                          [LLAMA1B], reason="search split only"))
        program = make_program(LLAMA1B, grafts=[graft_spec("qmath_block_8_19", 5)], heal=dict(HEAL_GSM8K))
        items = [evidence("block graft vs a random same-length block", "GSM8K search accuracy, mean over seeds",
                          block_value, random_value, GSM8K_SEARCH, seeds, "results/round3_rows.json",
                          control="random same-length block")]
        out.append(recipe("graft_block_qmath_8_19_llama1b", "Qwen-Math layers 8..19 grafted into Llama-1B at layer 5",
                          "rejected", LLAMA1B, program, items, "results/round3_rows.json (run 3)",
                          ["math", "gsm8k", "graft", "block"], [LLAMA1B, QMATH], components=["qmath_block_8_19"],
                          reason="ties a random block and loses to a rank-1 LoRA"))
    path = os.path.join(results, "sweep2_rows.json")
    if os.path.exists(path):
        rows = read_json(path)["rows"]
        # The glue rank is written as "r64" in sweep2_rows.json.
        graft_rows = rows_where(rows, layer=10, selection="activation512", rank="r64")
        random_rows = rows_where(rows, layer=10, selection="random", rank="r64")
        value, seeds = seeds_and_mean(graft_rows)
        random_value, _ = seeds_and_mean(random_rows)
        program = make_program(LLAMA1B, grafts=[graft_spec("qmath_l10_activation512", 6)], heal=dict(HEAL_GSM8K))
        items = [evidence("best run 2 cell vs its random control", "GSM8K search accuracy, mean over seeds", value,
                          random_value, GSM8K_SEARCH, seeds, "results/sweep2_rows.json",
                          control="random same-size cluster, same layer, same glue")]
        out.append(recipe("graft_qmath_l10_activation512_llama1b", "Qwen-Math layer 10 cluster grafted at Llama-1B "
                          "layer 6, rank-64 glue (best run 2 cell)", "rejected", LLAMA1B, program, items,
                          "results/sweep2_rows.json (run 2)", ["math", "gsm8k", "graft"], [LLAMA1B, QMATH],
                          components=["qmath_l10_activation512"],
                          reason="+1.8 over random is inside the noise; run 2 reads no method above random"))
    path = os.path.join(results, "final_heldout_microsoft__phi-4.json")
    if os.path.exists(path):
        data = read_json(path)
        verdict = data["verdict"]
        program = make_program(LLAMA1B, grafts=[graft_spec("mlp_cluster_phi4_l16_gsm8k", 6)], heal=dict(HEAL_GSM8K))
        items = gate_items(verdict, "results/final_heldout_microsoft__phi-4.json", GSM8K_HELDOUT)
        out.append(recipe("graft_phi4_l16_llama1b", "Phi-4 layer 16 GSM8K cluster grafted at Llama-1B layer 6",
                          "rejected", LLAMA1B, program, items, "results/final_heldout_microsoft__phi-4.json",
                          ["math", "gsm8k", "graft"], [LLAMA1B, "microsoft/phi-4"],
                          components=["mlp_cluster_phi4_l16_gsm8k"], added_by=PERSON_B,
                          reason="held-out GSM8K: -0.6 vs random; the matched LoRA scores 37.1"))
    path = os.path.join(results, "stitch_graft_s0.json")
    if os.path.exists(path):
        arms = {}
        for row in read_json(path):
            arms[row["arm"]] = row
        stitch_spec = {"kind": "linear", "objective": "donor_lm", "steps": 1000, "lr": 0.001, "batch_size": 8,
                       "seq_len": 256, "seed": 0}
        program = make_program(QWEN05, grafts=[graft_spec("qmath_block_8_19", 7, stitch=stitch_spec)],
                               heal=dict(HEAL_GSM8K))
        items = [evidence("block graft with a trained stitch vs a random block with the same stitch",
                          "GSM8K search accuracy", arms.get("block_stitch", {}).get("gsm8k"),
                          arms.get("random_stitch", {}).get("gsm8k"), GSM8K_SEARCH, [0], "results/stitch_graft_s0.json",
                          control="random same-length block",
                          note="matched LoRA " + str(arms.get("lora", {}).get("gsm8k")))]
        out.append(recipe("graft_block_qmath_8_19_stitch_qwen05", "Qwen-Math layers 8..19 with a donor_lm stitch in "
                          "Qwen-0.5B at layer 7", "rejected", QWEN05, program, items, "results/stitch_graft_s0.json",
                          ["math", "gsm8k", "graft", "block", "stitch"], [QWEN05, QMATH],
                          components=["qmath_block_8_19", "stitch_qwen05b_instruct_h7_d8_donor_lm_linear"], added_by=PERSON_B,
                          reason="ties the random block (1 seed); the matched LoRA scores 45.5"))

    # Run 15 carves.
    path = os.path.join(results, "round15_rows.json")
    if os.path.exists(path):
        rows = read_json(path)
        official = {}
        for row in rows_where(rows, kind="official"):
            official[row["model"]] = row
        matched = {QWEN14B: QWEN7B, QWEN32B: QWEN14B}
        groups = {}
        for row in rows_where(rows, kind="carve"):
            key = (row["model"], row["select"], row["heal"])
            if key not in groups:
                groups[key] = []
            groups[key].append(row)
        for key in sorted(groups):
            model, select, heal = key
            group = groups[key]
            values = []
            seeds = []
            for row in group:
                values.append(row["scores"]["mmlu_heldout"])
                seeds.append(row["seed"])
            first = group[0]
            bar = official.get(matched.get(model))
            control_value = None
            if bar is not None:
                control_value = bar["scores"]["mmlu_heldout"]
            lora = None
            heal_data = None
            if heal == "lora":
                lora = lora_spec(64)
                heal_data = heal_spec(1000, "mmlu_aux", 0.0001, 8, 512)
            program = make_program(model, base_source=model, lora=lora, heal=heal_data,
                                   carve={"keep": first["layers"], "target_params_b": None, "select": select,
                                          "domain_data": None, "width": None})
            gap = None
            if heal == "lora":
                gap = ("the heal used a WikiText + MMLU aux + GSM8K text mix (modal_round15.heal_texts), which has no "
                       "pairs name; the program field is the closest program, with data mmlu_aux")
            note = "params " + str(first["params_b"]) + "B, kept layers " + json.dumps(first.get("keep_layers"))
            items = [evidence("carved model vs the official model of the same size", "MMLU held-out accuracy, mean over "
                              "seeds", pct(mean(values)), pct(control_value), MMLU_HELDOUT, sorted(seeds),
                              "results/round15_rows.json", control="official " + str(matched.get(model)), note=note)]
            recipe_id = "carve_" + tag(model) + "_keep" + str(first["layers"]) + "_" + select + "_" + heal
            text = None
            if gap is not None:
                text = gap
            out.append(recipe(recipe_id, short_model(model) + " carved to " + str(first["layers"]) + " layers, " +
                              select + ", heal " + heal, "rejected", model, program, items,
                              "results/round15_rows.json (run 15)", ["carve", "written-model"], [model],
                              reason="far below the official model of the same size", syntax_gap=text))

    # Bake.
    path = os.path.join(results, "bake_rows.json")
    if os.path.exists(path):
        arms = {}
        for row in read_json(path):
            arms[row["arm"]] = row
        if "bake_tap" in arms and "bake_donor" in arms:
            program = {"bake": {"host": LLAMA1B, "teacher": "tap_mmlu_llama1b_qwen7b_l20_h10",
                                "lora": {"rank": 16}, "lr": 0.0001, "batch_size": 16, "steps": 1500,
                                "data": str(arms["bake_tap"].get("prompts")) + " MMLU auxiliary_train prompts",
                                "mode": "answer"}}
            items = [evidence("bake the tapped teacher vs distil the donor directly", "MMLU held-out accuracy",
                              arms["bake_tap"]["scores"]["heldout"], arms["bake_donor"]["scores"]["heldout"],
                              MMLU_HELDOUT, [0], "results/bake_rows.json", control="bake_donor",
                              note="LoRA on gold letters " + str(arms.get("lora_label", {}).get("scores", {}).get(
                                  "heldout")) + ", host " + str(arms.get("host", {}).get("scores", {}).get("heldout")))]
            out.append(recipe("bake_tap_mmlu_llama1b_qwen7b", "Bake the Qwen-7B tap into a plain Llama-1B", "rejected",
                              LLAMA1B, program, items, "results/bake_rows.json (modal_bake.py)",
                              ["knowledge", "mmlu", "bake"], [LLAMA1B, QWEN7B],
                              reason="equals plain donor distillation and a label LoRA; keeps 4 of 23 points",
                              syntax_gap="the text syntax has no bake statement; see wsp/bake.py and modal_bake.py",
                              added_by=PERSON_B))
    return out


# Merge and write


def strip_updated(data):
    copy = dict(data)
    copy.pop("updated", None)
    return copy


def merge_generated(old, new):
    """Merge a regenerated entry into the stored one. Keeps hand-added evidence, fields, and status."""
    merged = {}
    for key in new:
        merged[key] = new[key]
    for key in old:
        if key not in merged:
            merged[key] = old[key]
    hand = []
    for item in old.get("evidence") or []:
        if item.get("generated") is not True:
            hand.append(item)
    evidence_list = []
    for item in new.get("evidence") or []:
        evidence_list.append(item)
    for item in hand:
        evidence_list.append(item)
    merged["evidence"] = evidence_list
    if new.get("units_missing"):
        # The part file is absent in this checkout. Do not erase what an earlier build found.
        if merged.get("units_file") is None and old.get("units_file"):
            merged["units_file"] = old["units_file"]
    else:
        merged.pop("units_missing", None)
        merged.pop("units_expected", None)
    if len(hand) > 0:
        merged["status"] = old.get("status", merged["status"])
        if "status_reason" in old:
            merged["status_reason"] = old["status_reason"]
    if strip_updated(merged) == strip_updated(old) and old.get("updated"):
        merged["updated"] = old["updated"]
    else:
        merged["updated"] = today()
    return merged


def write_entry_file(library, data, report):
    rel = entry_path(data)
    path = os.path.join(library, rel)
    if os.path.exists(path):
        old = read_json(path)
        if old.get("generated") is not True:
            report["kept_hand"].append(rel)
            return None
        merged = merge_generated(old, data)
    else:
        merged = dict(data)
        merged["updated"] = today()
    if write_text(path, json_text(merged)):
        report["written"].append(rel)
    if merged.get("units_missing"):
        report["units_missing"].append(merged["id"] + " (expected " + str(merged.get("units_expected")) + ")")
    return merged


def write_collection(library, rel, entries, report):
    path = os.path.join(library, rel)
    old_lines = []
    if os.path.exists(path):
        old_lines = read_jsonl(path)
    old_by_id = {}
    for line in old_lines:
        old_by_id[line["id"]] = line
    out = []
    done = []
    for data in entries:
        old = old_by_id.get(data["id"])
        if old is not None and old.get("generated") is not True:
            continue
        if old is None:
            merged = dict(data)
            merged["updated"] = today()
        else:
            merged = merge_generated(old, data)
        out.append(merged)
        done.append(data["id"])
    for line in old_lines:
        if line["id"] in done:
            continue
        if line.get("generated") is True:
            report["stale"].append(rel + "#" + line["id"])
        else:
            report["kept_hand"].append(rel + "#" + line["id"])
        out.append(line)
    text = ""
    for line in out:
        text += json.dumps(line) + "\n"
    if write_text(path, text):
        report["written"].append(rel)


def recipe_program_text(data):
    lines = ["# " + data["id"] + ": " + data["title"]]
    for item in data["evidence"]:
        lines.append("# outcome: " + str(item["metric"]) + " " + str(item["value"]) + " vs " +
                     str(item.get("control") or "control") + " " + str(item["control_value"]) + " (" +
                     str(item["split"]) + ")")
    lines.append("# source: " + data["source"])
    lines.append("# generated by scripts/library.py build")
    return "\n".join(lines) + "\n" + data["text"]


def scan_files(library):
    """Every entry stored as one file: (entry, relative path)."""
    out = []
    patterns = ["models/*.json", "components/*/*.json", "recipes/*.json"]
    for pattern in patterns:
        for path in sorted(glob.glob(os.path.join(library, pattern))):
            rel = os.path.relpath(path, library).replace(os.sep, "/")
            try:
                data = read_json(path)
            except ValueError as error:
                data = {"_error": "bad JSON: " + str(error)}
            out.append((data, rel))
    return out


def scan_collections(library):
    out = []
    for rel in [FINDINGS_FILE, STITCHES_FILE]:
        path = os.path.join(library, rel)
        if not os.path.exists(path):
            continue
        with open(path) as f:
            number = 0
            for line in f:
                number += 1
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except ValueError as error:
                    data = {"_error": "bad JSON on line " + str(number) + ": " + str(error)}
                out.append((data, rel))
    return out


def load_entries(library):
    out = []
    for pair in scan_files(library):
        out.append(pair)
    for pair in scan_collections(library):
        out.append(pair)
    return out


def build(repo=REPO, library=DEFAULT_LIBRARY, quiet=False):
    """Regenerate the generated entries. Return a report dict."""
    report = {"written": [], "kept_hand": [], "stale": [], "copied": [], "units_missing": []}
    os.makedirs(library, exist_ok=True)

    produced = []
    components = build_components(repo)
    for data, copies in components:
        for src, rel in copies:
            if copy_file(src, os.path.join(library, rel)):
                report["copied"].append(rel)
        produced.append(data)
    models = build_models(repo)

    # Link each model to its components, generated and hand-added.
    links = {}
    for data in produced:
        model = data["model"]
        if model not in links:
            links[model] = []
        links[model].append(data["id"])
    for data, rel in scan_files(library):
        if data.get("kind") in COMPONENT_KINDS and data.get("generated") is not True:
            model = data.get("model")
            if model not in links:
                links[model] = []
            if data.get("id") not in links[model]:
                links[model].append(data["id"])
    for data in models:
        data["components"] = links.get(data["hf_id"], [])

    recipes = build_recipes(repo)
    for data in produced + models + recipes:
        merged = write_entry_file(library, data, report)
        if data["kind"] == "recipe" and merged is not None and merged.get("text"):
            if write_text(os.path.join(library, merged["program_file"]), recipe_program_text(merged)):
                report["written"].append(merged["program_file"])

    write_collection(library, FINDINGS_FILE, build_findings(repo), report)
    write_collection(library, STITCHES_FILE, build_stitches(repo), report)

    # Generated files that this build no longer produces stay, but are reported.
    ids = []
    for data in produced + models + recipes:
        ids.append(entry_path(data))
    for data, rel in scan_files(library):
        if data.get("generated") is True and rel not in ids:
            report["stale"].append(rel)
    if not quiet:
        print("build: wrote " + str(len(report["written"])) + " files, copied " + str(len(report["copied"])) +
              " unit files, kept " + str(len(report["kept_hand"])) + " hand-added entries, " +
              str(len(report["stale"])) + " stale generated entries")
        for rel in report["stale"]:
            print("  stale (not deleted): " + rel)
        if report["units_missing"]:
            print("part files missing: " + str(len(report["units_missing"])) +
                  " entries keep their old units_file and are flagged units_missing")
            for text in report["units_missing"]:
                print("  part files missing: " + text)
    return report


# Index


def index_rows(library):
    rows = []
    for data, rel in load_entries(library):
        if "_error" in data:
            continue
        path = rel
        if rel in [FINDINGS_FILE, STITCHES_FILE]:
            path = rel + "#" + str(data.get("id"))
        row = {}
        for field in INDEX_FIELDS:
            row[field] = data.get(field)
        row["path"] = path
        rows.append(row)
    order = {}
    for index in range(len(KINDS)):
        order[KINDS[index]] = index
    rows.sort(key=lambda row: (order.get(row["kind"], 99), str(row["id"])))
    return rows


def write_index(library, quiet=False):
    rows = index_rows(library)
    text = ""
    for row in rows:
        text += json.dumps(row) + "\n"
    write_text(os.path.join(library, "index.jsonl"), text)
    if not quiet:
        print("index: " + str(len(rows)) + " entries")
    return rows


# Validate


def is_heldout(split):
    if not isinstance(split, str):
        return False
    lowered = split.lower()
    return "held-out" in lowered or "heldout" in lowered or "held out" in lowered


def beats_control_held_out(item, needs_random):
    """True when an evidence item shows a held-out gain over its control.

    Higher is better unless the item says "lower_is_better": true (a loss or a penalty).
    For a component the control must be a random part, so "control" must say "random".
    """
    value = item.get("value")
    control_value = item.get("control_value")
    if not isinstance(value, (int, float)) or not isinstance(control_value, (int, float)):
        return False
    if not is_heldout(item.get("split")) or not item.get("seeds"):
        return False
    if item.get("lower_is_better") is True:
        if value >= control_value:
            return False
    elif value <= control_value:
        return False
    if needs_random and "random" not in str(item.get("control") or "").lower():
        return False
    return True


def check_evidence_item(item, where, errors):
    if not isinstance(item, dict):
        errors.append(where + ": an evidence item must be an object")
        return False
    good = True
    for field in EVIDENCE_FIELDS:
        if field not in item:
            errors.append(where + ": evidence item is missing '" + field + "'")
            good = False
    if item.get("generated") is not True and not item.get("added_by"):
        errors.append(where + ": a hand-added evidence item needs 'added_by'")
        good = False
    return good


def validate_entry(data, rel, library, errors, warnings=None):
    """Append every problem with one entry to errors. Soft problems go to warnings when it is given."""
    if "_error" in data:
        errors.append(rel + ": " + data["_error"])
        return
    label = rel + " (" + str(data.get("id")) + ")"
    for field in COMMON_FIELDS:
        if field not in data:
            errors.append(label + ": missing field '" + field + "'")
    entry_id = data.get("id")
    if not isinstance(entry_id, str) or ID_PATTERN.match(entry_id) is None:
        errors.append(label + ": id must use only letters, digits, '_', '.', '-'")
    kind = data.get("kind")
    if kind not in KINDS:
        errors.append(label + ": kind must be one of " + ", ".join(KINDS) + ", got " + repr(kind))
        return
    status = data.get("status")
    if status not in STATUSES:
        errors.append(label + ": status must be one of " + ", ".join(STATUSES) + ", got " + repr(status))
    for field in ["title", "added_by", "source"]:
        if field in data and (not isinstance(data[field], str) or data[field].strip() == ""):
            errors.append(label + ": '" + field + "' must be a non-empty string")
    for field in ["skills", "models", "evidence"]:
        if field in data and not isinstance(data[field], list):
            errors.append(label + ": '" + field + "' must be a list")
    group = kind
    if kind in COMPONENT_KINDS:
        group = "component"
    for field in KIND_FIELDS[group]:
        if field not in data:
            errors.append(label + ": a " + kind + " entry needs '" + field + "'")

    # The file must be where the kind says.
    if isinstance(entry_id, str) and kind in KINDS:
        try:
            expected = entry_path(data)
        except (KeyError, AttributeError, TypeError):
            expected = None
        if expected is not None and expected != rel:
            errors.append(label + ": a " + kind + " entry belongs at " + expected)

    # Files the entry names must exist.
    names = []
    if group == "component":
        if data.get("units_file"):
            names.append(data["units_file"])
        for control in data.get("controls") or []:
            names.append(control)
    if group == "recipe" and data.get("program_file"):
        names.append(data["program_file"])
    for name in names:
        if not isinstance(name, str) or os.path.isabs(name) or name.startswith(".."):
            errors.append(label + ": a file path must be relative to the library folder: " + str(name))
        elif not os.path.exists(os.path.join(library, name)):
            errors.append(label + ": file not found: " + str(name))
    if group == "component" and data.get("units_missing") and warnings is not None:
        warnings.append(label + ": part file missing at build time (expected " + str(data.get("units_expected")) +
                        "); units_file is " + str(data.get("units_file")))
    if group == "component":
        layers = data.get("layers")
        if not isinstance(layers, list):
            errors.append(label + ": 'layers' must be a list of layer numbers")

    items = data.get("evidence")
    if not isinstance(items, list):
        items = []
    good_items = []
    for number in range(len(items)):
        if check_evidence_item(items[number], label + " evidence " + str(number + 1), errors):
            good_items.append(items[number])

    if status == "validated":
        if group in ["component", "recipe"]:
            found = False
            for item in good_items:
                if beats_control_held_out(item, group == "component"):
                    found = True
            if not found:
                what = "its control"
                if group == "component":
                    what = "a random control named in 'control'"
                errors.append(label + ": status 'validated' needs an evidence item with value, control_value, seeds, "
                                      "and a held-out split, where the value beats " + what)
        elif group == "model":
            scores = data.get("scores")
            found = False
            if isinstance(scores, dict):
                for name in scores:
                    row = scores[name]
                    complete = isinstance(row, dict)
                    if complete:
                        for field in SCORE_FIELDS:
                            if row.get(field) is None:
                                complete = False
                    if complete:
                        found = True
                    else:
                        errors.append(label + ": score '" + name + "' needs " + ", ".join(SCORE_FIELDS))
            if not found:
                errors.append(label + ": status 'validated' needs a measured score with its split named")
        else:
            found = False
            for item in good_items:
                if item.get("value") is not None and item.get("split"):
                    found = True
            if not found:
                errors.append(label + ": status 'validated' needs an evidence item with a value and a split")
    if status == "rejected":
        if not data.get("status_reason") and len(good_items) == 0:
            errors.append(label + ": status 'rejected' needs a 'status_reason' or an evidence item")


def validate(library=DEFAULT_LIBRARY, check_index=True, warnings=None):
    """Return a list of error strings. An empty list means the library is valid.

    Pass a list as warnings to collect soft problems, such as entries flagged units_missing."""
    errors = []
    entries = load_entries(library)
    seen = {}
    for data, rel in entries:
        validate_entry(data, rel, library, errors, warnings)
        entry_id = data.get("id")
        if entry_id is None:
            continue
        if entry_id in seen:
            errors.append("duplicate id '" + str(entry_id) + "' in " + seen[entry_id] + " and " + rel)
        else:
            seen[entry_id] = rel
    if check_index:
        path = os.path.join(library, "index.jsonl")
        if not os.path.exists(path):
            errors.append("index.jsonl is missing: run python scripts/library.py index")
        else:
            stored = read_jsonl(path)
            fresh = index_rows(library)
            if stored != fresh:
                errors.append("index.jsonl is out of date: run python scripts/library.py index")
            for row in stored:
                path_part = str(row.get("path")).split("#")[0]
                if not os.path.exists(os.path.join(library, path_part)):
                    errors.append("index.jsonl names a missing file: " + str(row.get("path")))
    return errors


# Add


def resolve_file(library, name, folder, base_dir):
    """Return a library-relative path for name, copying the file into folder when it lives outside."""
    if not os.path.isabs(name) and os.path.exists(os.path.join(library, name)):
        return name
    candidates = [name, os.path.join(base_dir, name), os.path.join(REPO, name)]
    for candidate in candidates:
        if os.path.isfile(candidate):
            rel = folder + os.path.basename(candidate)
            copy_file(candidate, os.path.join(library, rel))
            return rel
    return name


def add(library, path, append_evidence=False):
    """Add the entry in JSON file path, or append its evidence to an existing entry. Return the entry."""
    data = read_json(path)
    base_dir = os.path.dirname(os.path.abspath(path))
    entries = load_entries(library)
    if append_evidence:
        return add_evidence(library, data, entries)
    for old, rel in entries:
        if old.get("id") == data.get("id"):
            raise SystemExit("add: id '" + str(data.get("id")) + "' already exists in " + rel +
                             "; use --append-evidence to add evidence to it")
    data["generated"] = False
    if not data.get("updated"):
        data["updated"] = today()
    if "evidence" not in data:
        data["evidence"] = []
    for item in data["evidence"]:
        if isinstance(item, dict) and "added_by" not in item and data.get("added_by"):
            item["added_by"] = data["added_by"]
    kind = data.get("kind")
    if kind in COMPONENT_KINDS and data.get("model"):
        folder = "components/" + model_key(data["model"]) + "/"
        if data.get("units_file"):
            data["units_file"] = resolve_file(library, data["units_file"], folder, base_dir)
        controls = []
        for control in data.get("controls") or []:
            controls.append(resolve_file(library, control, folder, base_dir))
        if "controls" in data:
            data["controls"] = controls
    if kind == "recipe" and data.get("program_file"):
        data["program_file"] = resolve_file(library, data["program_file"], "recipes/", base_dir)
    errors = []
    try:
        rel = entry_path(data)
    except (ValueError, KeyError) as error:
        raise SystemExit("add: cannot place the entry: " + str(error))
    validate_entry(data, rel, library, errors)
    if errors:
        raise SystemExit("add: the entry is not valid:\n  " + "\n  ".join(errors))
    if rel in [FINDINGS_FILE, STITCHES_FILE]:
        target = os.path.join(library, rel)
        os.makedirs(os.path.dirname(target), exist_ok=True)
        with open(target, "a") as f:
            f.write(json.dumps(data) + "\n")
    else:
        write_text(os.path.join(library, rel), json_text(data))
    write_index(library, quiet=True)
    print("add: wrote " + rel)
    return data


def add_evidence(library, data, entries):
    """Append data["evidence"] to the entry with id data["id"]. Never removes or edits old items."""
    items = data.get("evidence") or []
    if len(items) == 0:
        raise SystemExit("add: --append-evidence needs an 'evidence' list")
    errors = []
    for number in range(len(items)):
        item = items[number]
        if isinstance(item, dict):
            item.pop("generated", None)
        check_evidence_item(item, "evidence " + str(number + 1), errors)
    if errors:
        raise SystemExit("add: the evidence is not valid:\n  " + "\n  ".join(errors))
    for old, rel in entries:
        if old.get("id") != data.get("id"):
            continue
        for item in items:
            old["evidence"].append(item)
        if data.get("status") in STATUSES:
            old["status"] = data["status"]
            if data.get("status_reason"):
                old["status_reason"] = data["status_reason"]
        old["updated"] = today()
        check = []
        validate_entry(old, rel, library, check)
        if check:
            raise SystemExit("add: the entry would not be valid:\n  " + "\n  ".join(check))
        if rel in [FINDINGS_FILE, STITCHES_FILE]:
            lines = read_jsonl(os.path.join(library, rel))
            text = ""
            for line in lines:
                if line.get("id") == old["id"]:
                    line = old
                text += json.dumps(line) + "\n"
            write_text(os.path.join(library, rel), text)
        else:
            write_text(os.path.join(library, rel), json_text(old))
        write_index(library, quiet=True)
        print("add: appended " + str(len(items)) + " evidence items to " + old["id"])
        return old
    raise SystemExit("add: no entry with id '" + str(data.get("id")) + "'")


# Query


def query(library, kind=None, skill=None, model=None, status=None):
    """Index rows that match every given filter. model matches part of a model id, case-insensitive."""
    path = os.path.join(library, "index.jsonl")
    if os.path.exists(path):
        rows = read_jsonl(path)
    else:
        rows = index_rows(library)
    out = []
    for row in rows:
        if kind is not None and row.get("kind") != kind:
            continue
        if status is not None and row.get("status") != status:
            continue
        if skill is not None:
            skills = []
            for name in row.get("skills") or []:
                skills.append(name.lower())
            if skill.lower() not in skills:
                continue
        if model is not None:
            found = False
            for name in row.get("models") or []:
                if model.lower() in str(name).lower():
                    found = True
            if not found:
                continue
        out.append(row)
    return out


def clip(text, width):
    text = str(text)
    if len(text) <= width:
        return text.ljust(width)
    return text[:width - 3] + "..."


def print_table(rows):
    columns = [("id", 44), ("kind", 13), ("status", 10), ("models", 30), ("skills", 24), ("title", 60)]
    header = ""
    for name, width in columns:
        header += clip(name, width) + " "
    print(header.rstrip())
    for row in rows:
        line = ""
        for name, width in columns:
            value = row.get(name)
            if isinstance(value, list):
                shorts = []
                for item in value:
                    shorts.append(short_model(str(item)) if name == "models" else str(item))
                value = ",".join(shorts)
            line += clip(value if value is not None else "", width) + " "
        print(line.rstrip())
    print(str(len(rows)) + " entries")


# Command line


def main(argv=None):
    parser = argparse.ArgumentParser(description="Build, check, and search the shared Sapien library.")
    sub = parser.add_subparsers(dest="command", required=True)

    build_parser = sub.add_parser("build", help="regenerate the generated entries from repo sources")
    build_parser.add_argument("--repo", default=REPO)
    build_parser.add_argument("--library", default=DEFAULT_LIBRARY)

    validate_parser = sub.add_parser("validate", help="check every entry")
    validate_parser.add_argument("--library", default=DEFAULT_LIBRARY)

    index_parser = sub.add_parser("index", help="write index.jsonl")
    index_parser.add_argument("--library", default=DEFAULT_LIBRARY)

    add_parser = sub.add_parser("add", help="add one entry from a JSON file")
    add_parser.add_argument("file")
    add_parser.add_argument("--library", default=DEFAULT_LIBRARY)
    add_parser.add_argument("--append-evidence", action="store_true",
                            help="append the file's evidence items to the entry with its id")

    query_parser = sub.add_parser("query", help="list entries that match the filters")
    query_parser.add_argument("--library", default=DEFAULT_LIBRARY)
    query_parser.add_argument("--kind")
    query_parser.add_argument("--skill")
    query_parser.add_argument("--model")
    query_parser.add_argument("--status")
    query_parser.add_argument("--json", action="store_true", help="print JSON lines instead of a table")

    args = parser.parse_args(argv)
    if args.command == "build":
        build(args.repo, args.library)
        return 0
    if args.command == "index":
        write_index(args.library)
        return 0
    if args.command == "validate":
        warnings = []
        errors = validate(args.library, warnings=warnings)
        for warning in warnings:
            print("warning: " + warning)
        if errors:
            for error in errors:
                print("error: " + error)
            print("validate: " + str(len(errors)) + " errors")
            return 1
        print("validate: ok, " + str(len(load_entries(args.library))) + " entries, " + str(len(warnings)) +
              " warnings")
        return 0
    if args.command == "add":
        add(args.library, args.file, append_evidence=args.append_evidence)
        return 0
    if args.command == "query":
        rows = query(args.library, kind=args.kind, skill=args.skill, model=args.model, status=args.status)
        if args.json:
            for row in rows:
                print(json.dumps(row))
        else:
            print_table(rows)
        return 0
    return 2


if __name__ == "__main__":
    sys.exit(main())
