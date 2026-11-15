# Shared helpers for the v2 mining pipeline. Paths are relative to this file, so the
# scripts work from ~/swarmy and from person_a/ in the team repo alike.
import csv
import json
import os
import re
import time

MINE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(MINE_DIR)
REGISTRY_CSV = os.path.join(ROOT, "registry", "donors.csv")
BUILTIN_SKILLS = os.path.join(MINE_DIR, "data", "builtin_skills.json")

APP_NAME = "swarmy-mine"
VOLUME_NAME = "swarmy-models"

# Custom-code donors: only these may use trust_remote_code (always with a pinned revision)
CUSTOM_CODE = ["k2horizon7b", "k2mova36b", "g9v3_39a5b"]

BUILTIN_TARGETS = ["math", "code", "facts", "multilingual"]
CONTROL_ONLY = ["general_text"]


def catalog_path(short):
    return os.path.join(ROOT, "catalog", short + ".jsonl")


def parts_dir(short):
    return os.path.join(ROOT, "parts", short)


def skills_dir(short):
    return os.path.join(MINE_DIR, "skills", short)


def logs_dir(short):
    return os.path.join(ROOT, "logs", short)


def state_path(short):
    # WSP_LANE is set by run_many --agents N: each parallel agent on a model keeps its own state file.
    lane = os.environ.get("WSP_LANE", "")
    if lane == "":
        return os.path.join(logs_dir(short), "state.json")
    return os.path.join(logs_dir(short), "state.lane" + lane + ".json")


def load_donors():
    donors = {}
    with open(REGISTRY_CSV) as f:
        for row in csv.DictReader(f):
            donors[row["short_name"]] = row
    return donors


def pick_gpu(params_b):
    # bf16 weights take ~2 GB per billion params, plus room for activations / attribution.
    # Returns the engine class for that tier; each class has Modal GPU fallbacks (see engine_app.py):
    #   EngineL4: L4 | EngineL40S: L40S, then A100-40GB | EngineH100: H100, then H200 | EngineH200: H200, then B200
    #   EngineH200x8: 8 x H200, then 8 x B200 (above 80B params)
    if params_b <= 3:
        return "EngineL4"
    if params_b <= 16:
        return "EngineL40S"
    if params_b <= 34:
        return "EngineH100"
    if params_b <= 80:
        return "EngineH200"
    # frontier models: 8 x H200 (or B200), layers spread over the GPUs
    return "EngineH200x8"


def scan_token_budget(params_b):
    # tokens per micro-batch for backward passes (attribution); smaller for big models
    if params_b <= 3:
        return 4096
    if params_b <= 16:
        return 2048
    return 1024


# ---------- atomic, self-healing JSON state ----------

def write_json_atomic(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=1)
        f.flush()
        os.fsync(f.fileno())
    if os.path.exists(path):
        # keep the last good copy for self-healing loads
        try:
            os.replace(path, path + ".bak")
        except OSError:
            pass
    os.replace(tmp, path)


def read_json_safe(path, default):
    for candidate in [path, path + ".bak"]:
        if not os.path.exists(candidate):
            continue
        try:
            with open(candidate) as f:
                return json.load(f)
        except Exception:
            print("warning: could not read", candidate, "- trying backup")
    return default


def append_jsonl(path, record):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")


def log_event(short, what, config, result):
    # Per-model experiment log (never the shared top-level experiments.jsonl: avoids merge conflicts)
    record = {"timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"), "what": what, "config": config, "result": result}
    append_jsonl(os.path.join(logs_dir(short), "experiments.jsonl"), record)
    return record


def clean_name(text, fallback="unnamed"):
    out = ""
    for ch in str(text).lower().replace(" ", "_"):
        if ch.isalnum() or ch in "-_":
            out = out + ch
    out = re.sub(r"_+", "_", out).strip("_")[:48]
    if out == "":
        return fallback
    return out


def pin_revision(hf_id):
    import requests

    r = requests.get("https://huggingface.co/api/models/" + hf_id, timeout=60)
    r.raise_for_status()
    return r.json()["sha"]
