"""Specs: drafting from a prompt (rules, or an LLM when one is set), upload counting and storage."""

import csv
import io
import json
import math
import os
import re
import threading
import uuid

from wsp.swarm import config as config_mod
from wsp.swarm import factory
from wsp.swarm import space

from studio.server import catalog as catalog_mod
from studio.server import library
from studio.server import settings
from studio.server import shrink_models
from studio.server import shrink_spec
from studio.server import sizing

# Fine-tuning bars measured on the Llama-1B host (results/README.md), by task.
# mmlu: LoRA r16 on the round 6 dev items, log-prob scoring, the same scorer the studio uses.
# gsm8k has no bar: the 0.467 in results/README.md (LoRA r64) was measured with the old
# harness scorer, which cut answers at 256 tokens and read the Llama-1B host at 0.23. The
# studio now scores GSM8K with the team's skills_eval protocol (host 0.58), and no LoRA has
# been measured under it. A bar goes back here when one is.
LORA_BARS = {"mmlu": 0.503}

# Task types the studio adds to the swarm config.
EXTRA_TASK_TYPES = {
    "knowledge": {"tasks": {"mmlu": 1.0}, "guards": ["if_guard"], "heal_data": ["general"]},
}

# The heal data pool for each task type. Heal, fit and selection text always comes from a pool,
# never from a benchmark the request is scored on.
HEAL_POOLS = {"math": "math", "knowledge": "general", "code": "code", "medicine": "medicine"}
EXTRA_HEAL_POOLS = ["general", "math", "code", "medicine", "prose", "mix"]


def heal_pool_for(task_type):
    """The pool name for the default heal of a task type: math, general, code or medicine."""
    return HEAL_POOLS.get(task_type, "general")


def heal_data_for(task_type):
    """The heal data names a task type allows: its own pool first, then the other approved pools."""
    first = heal_pool_for(task_type)
    names = [first]
    for name in EXTRA_HEAL_POOLS:
        if name != first:
            names.append(name)
    return names
KNOWLEDGE_WORDS = ["knowledge", "facts", "fact", "trivia", "mmlu", "exam", "quiz", "medical", "medicine", "law",
                   "legal", "history", "science", "biology", "chemistry", "physics", "economics", "know more",
                   "smarter", "knows"]
MATH_WORDS = ["math", "arithmetic", "gsm8k", "word problem", "algebra", "calculate"]

# Measured host scores the swarm config does not have. mmlu: Llama-3.2-1B-Instruct,
# log-prob scoring, round 6 dev items (search) and round 7 held-out items.
# gsm8k: 200 search items, skills_eval protocol (chat template, 512 new tokens), L40S,
# 2026-11-15. It replaces the 0.23 in wsp/swarm/config.py, which came from the old scorer.
MEASURED_BASES = {"search": {"mmlu": 0.475, "gsm8k": 0.58}, "heldout": {"mmlu": 0.447}}
# Tasks whose value in the swarm defaults is stale and must be replaced, not just filled in.
REPLACE_BASES = {"search": ["gsm8k"]}
# Held-out bases from the old scorer that have not been measured again: drop them so a
# backend measures the host on the same scorer as the candidates.
DROP_BASES = {"heldout": ["gsm8k"]}


def add_measured_bases():
    """Put MEASURED_BASES into the in-memory swarm defaults (wsp/swarm/config.py is not edited)."""
    table = config_mod.DEFAULTS["base_scores"]
    for split in MEASURED_BASES:
        if split not in table:
            table[split] = {}
        for task in MEASURED_BASES[split]:
            if task not in table[split] or task in REPLACE_BASES.get(split, []):
                table[split][task] = MEASURED_BASES[split][task]
    for split in DROP_BASES:
        for task in DROP_BASES[split]:
            if split in table and task in table[split]:
                del table[split][task]


add_measured_bases()

MAX_DRAFT_TARGETS = 2

DEFAULT_BUDGET = {"evals": 30, "compute_min": 45, "min_evals": 5, "max_evals": 120}
MIN_COMPUTE_MIN = 5
MAX_COMPUTE_MIN = 600
SIZE_MIN_B = 0.5
SIZE_MAX_B = 32.0
DEFAULT_SIZE_B = 1.5
DEFAULT_ROOM_FACTOR = 1.5
BUILD_ROOM_B = 0.2
DEFAULT_HEAL_MIN = 20
MIN_HEAL_MIN = 1
MAX_HEAL_MIN = 600

BENCHMARK_NAMES = {
    "gsm8k": "GSM8K",
    "if_guard": "IFEval",
    "humaneval": "HumanEval",
    "mmlu": "MMLU",
    "bbeh_mini": "BBEH mini",
}
GUARD_IDS = ["if_guard"]
MATH_TASKS = ["gsm8k", "mmlupro:math"]
EXTRA_TARGET_IDS = ["humaneval", "mmlu", "bbeh_mini"]

LEAD_WORDS = ["i", "we", "want", "need", "would", "like", "to", "build", "make", "create", "me", "my", "a", "an",
              "the", "model", "that", "please", "can", "you", "give", "train", "get", "some", "something", "for",
              "which", "is", "good", "at", "of", "our"]
STOP_WORDS = ["that", "which", "for", "to", "on", "in", "with", "and", "who", "can", "so", "but", "from", "under",
              "using", "by", "at", "of"]

SIZE_WORDS = [("phone", 1.3), ("mobile", 1.3), ("tiny", 1.3), ("laptop", 1.5), ("small", 1.5), ("cpu", 1.5)]

_lock = threading.Lock()
_specs = {}


# Small helpers.

def new_spec_id():
    return "sp_" + uuid.uuid4().hex[:6]


def clamp(value, low, high):
    """Keep value between low and high. A value that is not a finite number becomes low."""
    if not math.isfinite(value):
        return low
    if value < low:
        return low
    if value > high:
        return high
    return value


def round_step(value, step):
    """Round to the nearest multiple of step."""
    return round(round(value / step) * step, 4)


def to_float(value, default):
    """A finite float, or the default for anything else (text, nan, inf)."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(number):
        return default
    return number


def to_int(value, default):
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


# Upload counting.

def count_rows(name, data):
    """Count the rows in an uploaded file (bytes). Unknown file types count as 0."""
    text = data.decode("utf-8", errors="replace")
    lower = name.lower()
    if lower.endswith(".csv"):
        rows = 0
        for row in csv.reader(io.StringIO(text)):
            if len(row) > 0 and "".join(row).strip() != "":
                rows += 1
        if rows > 0:
            rows -= 1
        return rows
    if lower.endswith(".jsonl") or lower.endswith(".txt"):
        rows = 0
        for line in text.splitlines():
            if line.strip() != "":
                rows += 1
        return rows
    if lower.endswith(".json"):
        try:
            value = json.loads(text)
        except ValueError:
            return 0
        if isinstance(value, list):
            return len(value)
        if isinstance(value, dict):
            for key in ["data", "rows", "examples", "items"]:
                if isinstance(value.get(key), list):
                    return len(value[key])
        return 1
    return 0


def split_rows(rows):
    """One third of rows (at least 1) are score rows, the rest are heal rows."""
    if rows <= 0:
        return 0, 0
    score_rows = max(1, rows // 3)
    return score_rows, rows - score_rows


def safe_name(name):
    base = os.path.basename(str(name).replace("\\", "/"))
    base = re.sub(r"[^A-Za-z0-9._-]", "_", base)
    if base == "" or base.startswith("."):
        base = "upload" + base
    return base


def save_uploads(spec_id, uploads):
    """Save (filename, bytes) pairs under uploads/<spec_id>/. Returns the spec's file list."""
    files = []
    folder = os.path.join(settings.out_dir(), "uploads", spec_id)
    for name, data in uploads:
        clean = safe_name(name)
        os.makedirs(folder, exist_ok=True)
        with open(os.path.join(folder, clean), "wb") as handle:
            handle.write(data)
        rows = count_rows(clean, data)
        score_rows, heal_rows = split_rows(rows)
        files.append({"name": clean, "rows": rows, "score_rows": score_rows, "heal_rows": heal_rows})
    return files


# Benchmarks.

def benchmark_name(task_id):
    if task_id in BENCHMARK_NAMES:
        return BENCHMARK_NAMES[task_id]
    if task_id.startswith("mmlupro:"):
        return "MMLU-Pro " + task_id.split(":", 1)[1].replace("_", " ")
    return task_id


def base_for(config, task_id):
    """The host's search-split score for a task, or None when it is not known.

    On the stub backend this is the stub's own base, so the spec's tick marks
    agree with the scores the stub run reports.
    """
    if config.get("backend") == "stub":
        from wsp.swarm import stub
        return float(stub.STUB_BASE["search"].get(task_id, stub.UNKNOWN_TASK_BASE))
    value = config["base_scores"]["search"].get(task_id)
    if value is None:
        return None
    return float(value)


def target_entry(config, task_id, recommended):
    base = base_for(config, task_id)
    lora = LORA_BARS.get(task_id)
    if lora is not None:
        goal = round_step(lora + 0.03, 0.05)
    elif base is not None:
        goal = round_step(base + 0.10, 0.05)
    else:
        goal = 0.5
    return {"id": task_id, "name": benchmark_name(task_id), "kind": "target", "target": clamp(goal, 0.0, 1.0),
            "base": base, "lora": lora, "recommended": recommended}


def guard_entry(config, task_id, recommended):
    base = base_for(config, task_id)
    if base is not None:
        floor = round(base * (1.0 - config["guard_drop"]), 2)
    else:
        floor = 0.3
    return {"id": task_id, "name": benchmark_name(task_id), "kind": "guard", "floor": floor,
            "base": base, "recommended": recommended}


def default_benchmarks(config):
    """The task type's tasks as targets and its guards as guards."""
    setup = config["task_types"][config["task_type"]]
    items = []
    for task_id in sorted(setup["tasks"]):
        items.append(target_entry(config, task_id, True))
    for task_id in setup["guards"]:
        items.append(guard_entry(config, task_id, True))
    return items


def known_benchmark_ids(config):
    """Every benchmark id the menu can offer, in menu order."""
    ids = []
    setup = config["task_types"][config["task_type"]]
    for task_id in sorted(setup["tasks"]):
        ids.append(task_id)
    for task_id in setup["guards"]:
        ids.append(task_id)
    for task_id in EXTRA_TARGET_IDS:
        ids.append(task_id)
    for category in mmlupro_categories():
        ids.append("mmlupro:" + category.replace(" ", "_"))
    return ids


def mmlupro_categories():
    try:
        from wsp_eval import tasks as eval_tasks
        return list(eval_tasks.MMLUPRO_CATEGORIES)
    except Exception:
        return ["biology", "business", "chemistry", "computer science", "economics", "engineering", "health",
                "history", "law", "math", "other", "philosophy", "physics", "psychology"]


def entry_for_id(config, task_id, recommended):
    if task_id in GUARD_IDS or task_id in config["task_types"][config["task_type"]]["guards"]:
        return guard_entry(config, task_id, recommended)
    return target_entry(config, task_id, recommended)


def available_benchmarks(config, chosen):
    """Benchmarks the add menu offers: known ones that are not in the spec yet."""
    taken = []
    for item in chosen:
        taken.append(item["id"])
    out = []
    for task_id in known_benchmark_ids(config):
        if task_id in taken:
            continue
        entry = entry_for_id(config, task_id, False)
        item = {"id": task_id, "name": entry["name"], "kind": entry["kind"], "base": entry["base"]}
        out.append(item)
    return out


# Ingredients.

def ingredients_for(config, catalog, benchmarks):
    """Up to three parts whose signature hints help the main target, one per donor and kind."""
    main = None
    for item in benchmarks:
        if item["kind"] == "target":
            main = item["id"]
            break
    if main is None:
        return []
    scored = []
    for record in space.graftable_parts(catalog):
        effect = (record.get("signature") or {}).get("effect") or {}
        if main in effect:
            block_first = 1
            if record.get("type") == "block":
                block_first = 0
            topic_first = 1
            if main in MATH_TASKS and "Math" in str(record.get("model")):
                topic_first = 0
            scored.append((block_first, topic_first, -float(effect[main]), record["id"]))
    scored.sort()
    out = []
    seen = []
    for pair in scored:
        record = catalog.get(pair[3])
        key = (record.get("model"), record.get("type"))
        if key in seen:
            continue
        seen.append(key)
        out.append({"part_id": record["id"], "label": catalog_mod.part_label(record),
                    "donor": record.get("model"), "kind": record.get("type")})
        if len(out) >= 3:
            break
    if len(out) == 0:
        out = library_ingredients(main)
    return out


# Library skill tags that belong to a benchmark.
TASK_SKILLS = {"gsm8k": ["math", "gsm8k"], "mmlu": ["knowledge", "mmlu"], "humaneval": ["code"],
               "bbeh_mini": ["reasoning"]}


def library_ingredients(task_id):
    """Up to three library parts (not rejected) tagged with the benchmark's skills, one per donor and kind."""
    wanted = TASK_SKILLS.get(task_id, [])
    if task_id.startswith("mmlupro:"):
        wanted = ["knowledge", "mmlu"]
    placements = catalog_mod.library_placements()
    found = []
    for entry in library.get()["parts"]:
        if entry["status"] == "rejected" or entry.get("tap_only"):
            continue
        hit = False
        for skill in entry["skills"]:
            if skill in wanted:
                hit = True
        if not hit:
            continue
        placeable = placements.get(entry["catalog_id"], (None, None))[0] is not None
        # Placeable parts first, then parts tagged with the benchmark itself, then small ones.
        size_rank = 1 if entry["kind"] in ("block", "inserted_layer") else 0
        found.append(((not placeable, task_id not in entry["skills"], size_rank, entry["catalog_id"]), entry))
    found.sort(key=lambda pair: pair[0])
    out = []
    seen = []
    for _rank, entry in found:
        key = (entry["model"], entry["kind"])
        if key in seen:
            continue
        seen.append(key)
        out.append({"part_id": entry["catalog_id"], "label": entry["title"], "donor": entry["model"],
                    "kind": entry["kind"], "status": entry["status"]})
        if len(out) >= 3:
            break
    return out


# Rules drafting.

def clean_prompt(prompt):
    return re.sub(r"\s+", " ", str(prompt or "")).strip()


# A size followed by "per token" or "active" is a per-token limit, not the total.
NOT_PER_TOKEN = r"(?!\s*(?:params\s+|parameters\s+)?(?:per[ -]token|active))"


_smallest_part_cache = {}


def smallest_part_b(backend=None):
    """Billions of parameters in the smallest usable part of the catalog, or None when no part has a size.

    This is the same number the spec page uses for its "no part fits" warning. It is read once per backend.
    """
    if backend is None:
        backend = settings.backend()
    if backend not in _smallest_part_cache:
        found = None
        try:
            for part in catalog_mod.build_palette(backend)["parts"]:
                size = part.get("params_m")
                if part.get("usable") and size is not None and size > 0:
                    if found is None or size / 1000.0 < found:
                        found = size / 1000.0
        except Exception:
            found = None
        _smallest_part_cache[backend] = found
    return _smallest_part_cache[backend]


def run_headroom_b():
    """The per-token room above the host that a draft keeps: the smallest part, rounded up to 0.01B.

    When the catalog gives no part size, BUILD_ROOM_B stands in.
    """
    smallest = smallest_part_b()
    if smallest is None:
        return BUILD_ROOM_B
    return math.ceil(smallest * 100 - 1e-9) / 100.0


def build_floor(host_params):
    """The smallest total limit a build spec gets by default: the host size plus 0.2B."""
    return round(host_params + BUILD_ROOM_B, 2)


def size_from_prompt(prompt, host_params):
    """The size limit in billions of parameters for a build.

    A size in the words ("under 2B", "phone") is used, but never below the host size plus 0.2B.
    With no size in the words the limit is the host size times 1.5, rounded to 0.1B.
    """
    text = clean_prompt(prompt).lower()
    match = re.search(r"(?:under|below|less than|at most|max(?:imum)?|up to|<=?)\s*(\d+(?:\.\d+)?)\s*b\b" + NOT_PER_TOKEN,
                      text)
    if match is not None:
        return clamp_size(max(float(match.group(1)), build_floor(host_params)), host_params)
    for word, value in SIZE_WORDS:
        if re.search(r"\b" + word + r"\b", text):
            return clamp_size(max(value, build_floor(host_params)), host_params)
    return clamp_size(max(round(host_params * DEFAULT_ROOM_FACTOR, 1), build_floor(host_params)), host_params)


RUN_PATTERNS = [
    r"(\d+(?:\.\d+)?)\s*b\s+(?:params\s+|parameters\s+)?(?:per[ -]token|active)",
    r"\b(?:per[ -]token|active|runs?|running)\b\D{0,16}?(\d+(?:\.\d+)?)\s*b\b",
]
HEAL_PATTERNS = [
    r"(?:heal|healing|train|training)\D{0,24}?(\d+(?:\.\d+)?)\s*(minutes?|mins?|hours?|hrs?|h)\b",
    r"(\d+(?:\.\d+)?)\s*(minutes?|mins?|hours?|hrs?|h)\s+(?:of\s+|to\s+)?(?:heal|healing|train|training)",
]


def run_limit_from_prompt(prompt, total_b, host_params):
    """The per-token limit in billions: words like '1B per token', else the total limit."""
    text = clean_prompt(prompt).lower()
    for pattern in RUN_PATTERNS:
        match = re.search(pattern, text)
        if match is not None:
            return clamp_run(float(match.group(1)), total_b, host_params)
    return clamp_run(total_b, total_b, host_params)


def heal_limit_from_prompt(prompt):
    """The heal limit in minutes: words like 'heal for 30 minutes', else the default."""
    text = clean_prompt(prompt).lower()
    for pattern in HEAL_PATTERNS:
        match = re.search(pattern, text)
        if match is not None:
            value = float(match.group(1))
            if match.group(2).startswith("h"):
                value = value * 60.0
            return clamp_heal(value)
    return DEFAULT_HEAL_MIN


def clamp_run(value, total_b, host_params):
    """Keep the per-token limit between the host size and the total limit."""
    return round(clamp(float(value), host_params, max(host_params, total_b)), 2)


def clamp_heal(value):
    return int(round(clamp(float(value), MIN_HEAL_MIN, MAX_HEAL_MIN)))


def clamp_size(value, host_params):
    """Keep the limit between the host size and the largest size the studio offers."""
    low = max(SIZE_MIN_B, host_params)
    return round(clamp(float(value), low, SIZE_MAX_B), 2)


def draft_title(prompt):
    words = re.findall(r"[A-Za-z0-9][A-Za-z0-9'-]*", clean_prompt(prompt))
    while len(words) > 0 and words[0].lower() in LEAD_WORDS:
        words = words[1:]
    picked = []
    for word in words:
        if word.lower() in STOP_WORDS and len(picked) >= 2:
            break
        picked.append(word)
        if len(picked) >= 4:
            break
    if len(picked) == 0:
        return "Untitled model"
    title = " ".join(picked).lower()
    lower_prompt = clean_prompt(prompt).lower()
    for word in ["laptop", "phone"]:
        if word in lower_prompt and word not in title:
            title = word + " " + title
            break
    return title[0].upper() + title[1:]


def draft_summary(prompt, size_b):
    text = clean_prompt(prompt)
    if text == "":
        return "A math model built from a small host and donor parts. Fits under " + str(size_b) + "B parameters."
    text = re.sub(r"^(?:i|we)\s+(?:want|need|would like|am looking for)(?:\s+to\s+(?:build|make|create|get))?\s+", "",
                  text, flags=re.IGNORECASE)
    text = re.sub(r"^(?:please\s+)?(?:build|make|create)\s+(?:me\s+)?", "", text, flags=re.IGNORECASE)
    if text == "":
        text = "A model"
    text = text[0].upper() + text[1:]
    sentence = re.split(r"(?<=[.!?])\s", text)[0]
    if len(sentence) > 200:
        sentence = sentence[:197].rstrip() + "..."
    if sentence[-1] not in ".!?":
        sentence += "."
    return sentence + " Fits under " + str(size_b) + "B parameters."


def clip(text, limit):
    text = clean_prompt(text)
    if len(text) <= limit:
        return text
    return text[:limit - 3].rstrip() + "..."


# LLM drafting.

DRAFT_SCHEMA = {
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "summary": {"type": "string"},
        "max_params_b": {"type": "number"},
        "max_run_params_b": {"type": "number"},
        "max_heal_min": {"type": "number"},
        "benchmarks": {"type": "array", "items": {"type": "string"}},
        "request_type": {"type": "string", "enum": ["build", "shrink"]},
        "source_model": {"type": "string"},
        "keep_skill": {"type": "string"},
    },
    "required": ["title", "summary", "max_params_b", "benchmarks"],
}


def parse_json_reply(text):
    """Read a JSON object from a model reply, also when it sits inside a code fence."""
    text = str(text).strip()
    if text.startswith("```"):
        lines = text.splitlines()
        lines = lines[1:]
        if len(lines) > 0 and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines)
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end < start:
        raise ValueError("no JSON object in the reply")
    value = json.loads(text[start:end + 1])
    if not isinstance(value, dict):
        raise ValueError("the reply is not a JSON object")
    return value


def llm_draft(config, prompt, files, known_ids, host_params):
    """Ask the configured LLM for a draft. Returns a dict, or raises on any failure."""
    llm = factory.build_llm(config)
    file_lines = []
    for item in files:
        file_lines.append(item["name"] + ": " + str(item["rows"]) + " rows")
    system = ("You draft a small model spec from a plain request. Reply with one JSON object with the keys "
              "title (at most 6 words), summary (one or two plain sentences), max_params_b (total parameters, a "
              "number between 0.5 and 32), optional max_run_params_b (parameters run per token, at most the total), "
              "optional max_heal_min (minutes of heal), and benchmarks (a list of ids from the allowed list; the first id is the main target). "
              "Also set request_type: shrink when the request asks to remove skills from a model or to make a smaller "
              "version of one (then give source_model, a model id from the allowed list, when one fits, and keep_skill, "
              "one of " + ", ".join(shrink_spec.SKILL_POOLS) + "), else build. "
              "Write plain text without dashes used as punctuation.")
    user = ("Request: " + clean_prompt(prompt) + "\nUploaded files: " + (", ".join(file_lines) or "none") +
            "\nAllowed benchmark ids: " + ", ".join(known_ids) +
            "\nAllowed source models: " + ", ".join(shrink_models.source_ids()))
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    reply = llm.complete(messages, schema=DRAFT_SCHEMA, temperature=0.3)
    return parse_json_reply(reply)


def apply_llm_draft(spec, config, draft, host_params):
    """Copy the valid parts of an LLM draft into the spec. Anything invalid is ignored."""
    title = draft.get("title")
    if isinstance(title, str) and clean_prompt(title) != "":
        spec["title"] = clip(title, 60)
    summary = draft.get("summary")
    if isinstance(summary, str) and clean_prompt(summary) != "":
        spec["summary"] = clip(summary, 400)
    size = draft.get("max_params_b")
    if isinstance(size, (int, float)) and not isinstance(size, bool):
        spec["size"]["max_params_b"] = clamp_size(max(size, build_floor(host_params)), host_params)
        spec["limits"]["max_total_params_b"] = spec["size"]["max_params_b"]
        spec["limits"]["max_run_params_b"] = clamp_run(spec["limits"]["max_total_params_b"],
                                                       spec["limits"]["max_total_params_b"], host_params)
    run = draft.get("max_run_params_b")
    if isinstance(run, (int, float)) and not isinstance(run, bool):
        spec["limits"]["max_run_params_b"] = clamp_run(run, spec["limits"]["max_total_params_b"], host_params)
        # A run limit at the host size leaves no room for any part, so a draft keeps the build room.
        room = min(spec["limits"]["max_total_params_b"], round(host_params + run_headroom_b(), 2))
        if spec["limits"]["max_run_params_b"] < room:
            spec["limits"]["max_run_params_b"] = round(room, 2)
    heal = draft.get("max_heal_min")
    if isinstance(heal, (int, float)) and not isinstance(heal, bool):
        spec["limits"]["max_heal_min"] = clamp_heal(heal)
    ids = draft.get("benchmarks")
    if isinstance(ids, list):
        known = known_benchmark_ids(config)
        # The task type's own targets come first and stay. Each extra target
        # costs an eval per score (a slow one on real GPUs), so a draft keeps
        # at most MAX_DRAFT_TARGETS targets; the user can add more on the spec page.
        main_targets = list(config["task_types"][config["task_type"]]["tasks"])
        chosen = []
        for task_id in main_targets:
            chosen.append(task_id)
        target_count = len(chosen)
        for task_id in ids:
            if task_id not in known or task_id in chosen:
                continue
            is_guard = task_id in GUARD_IDS or task_id in config["task_types"][config["task_type"]]["guards"]
            if not is_guard:
                if target_count >= MAX_DRAFT_TARGETS:
                    continue
                target_count += 1
            chosen.append(task_id)
        items = []
        for task_id in chosen:
            items.append(entry_for_id(config, task_id, True))
        has_target = False
        for item in items:
            if item["kind"] == "target":
                has_target = True
        if has_target:
            has_guard = False
            for item in items:
                if item["kind"] == "guard":
                    has_guard = True
            if not has_guard:
                for task_id in config["task_types"][config["task_type"]]["guards"]:
                    items.append(guard_entry(config, task_id, True))
            spec["benchmarks"] = items


# Task types.

def add_task_types(config):
    """Add the studio's extra task types to a swarm config dict."""
    for name in EXTRA_TASK_TYPES:
        if name not in config["task_types"]:
            config["task_types"][name] = dict(EXTRA_TASK_TYPES[name])
            config["task_types"][name]["heal_data"] = heal_data_for(name)
    # The swarm defaults still name a benchmark train set for math. The studio lists pools instead.
    for name in config["task_types"]:
        config["task_types"][name] = dict(config["task_types"][name])
        config["task_types"][name]["heal_data"] = heal_data_for(name)


def task_type_for(prompt):
    """math or knowledge, from the words in the prompt. Math wins a tie."""
    text = " " + str(prompt or "").lower() + " "
    math_hits = 0
    for word in MATH_WORDS:
        if word in text:
            math_hits += 1
    knowledge_hits = 0
    for word in KNOWLEDGE_WORDS:
        if " " + word in text:
            knowledge_hits += 1
    if knowledge_hits > math_hits:
        return "knowledge"
    return "math"


# Draft, normalize, store.

def draft_spec(prompt, uploads):
    """Draft a new spec from a prompt and uploads [(filename, bytes)], and store it."""
    backend = settings.backend()
    config, catalog = catalog_mod.load_catalog(backend)
    host_params = catalog_mod.host_params_b(config["host"])
    add_task_types(config)
    config["task_type"] = task_type_for(prompt)
    spec_id = new_spec_id()
    files = save_uploads(spec_id, uploads)
    size_b = size_from_prompt(prompt, host_params)
    limits = {"max_total_params_b": size_b, "max_run_params_b": run_limit_from_prompt(prompt, size_b, host_params),
              "max_heal_min": heal_limit_from_prompt(prompt)}
    benchmarks = default_benchmarks(config)
    spec = {
        "id": spec_id,
        "title": draft_title(prompt),
        "summary": draft_summary(prompt, size_b),
        "prompt": str(prompt or ""),
        "files": files,
        "task_type": config["task_type"],
        "host": config["host"],
        "size": {"max_params_b": size_b, "min_b": SIZE_MIN_B, "max_b": SIZE_MAX_B, "host_params_b": host_params},
        "limits": limits,
        "comparators": {"distill": {}},
        "request_type": "build",
        "budget": dict(DEFAULT_BUDGET),
        "benchmarks": benchmarks,
        "available_benchmarks": [],
        "ingredients": [],
        "drafted_by": "rules",
        "stub": settings.is_stub(),
    }
    llm_name = settings.llm()
    draft = {}
    if llm_name != "none":
        llm_config = config_mod.make_config({"llm": llm_name, "backend": backend})
        try:
            draft = llm_draft(llm_config, prompt, files, known_benchmark_ids(config), host_params)
            apply_llm_draft(spec, config, draft, host_params)
            spec["drafted_by"] = "llm:" + llm_name
        except Exception:
            draft = {}
            spec["drafted_by"] = "rules"
    if request_type_for(prompt, draft) == "shrink":
        make_shrink_spec(spec, config, prompt, draft)
    spec["available_benchmarks"] = available_benchmarks(config, spec["benchmarks"])
    if spec["request_type"] == "shrink":
        spec["ingredients"] = []
    else:
        spec["ingredients"] = ingredients_for(config, catalog, spec["benchmarks"])
    store(spec)
    return spec


def request_type_for(prompt, draft):
    """"shrink" or "build". An LLM that answered with a valid request_type decides; otherwise the words do."""
    asked = draft.get("request_type") if isinstance(draft, dict) else None
    if asked in ("build", "shrink"):
        return asked
    if shrink_spec.looks_like_shrink(prompt):
        return "shrink"
    return "build"


def stated_run_b(prompt):
    """The per-token limit in the request ("1B per token"), or None."""
    text = clean_prompt(prompt).lower()
    for pattern in RUN_PATTERNS:
        match = re.search(pattern, text)
        if match is not None:
            return float(match.group(1))
    return None


def make_shrink_spec(spec, config, prompt, draft):
    """Turn a drafted spec into a shrink spec in place. Raises ValueError when no source model can be found."""
    hint_model = draft.get("source_model") if isinstance(draft, dict) else None
    hint_skill = draft.get("keep_skill") if isinstance(draft, dict) else None
    try:
        block, limits = shrink_spec.draft_plan(prompt, hint_model, hint_skill)
    except shrink_spec.ShrinkSpecError as error:
        raise ValueError(str(error))
    run = stated_run_b(prompt)
    if run is not None:
        limits["max_run_params_b"] = min(run, limits["max_total_params_b"])
    limits["max_heal_min"] = heal_limit_from_prompt(prompt)
    estimate = sizing.estimate_shrink(block)
    skill = block["keep_skill"]
    config["task_type"] = "math" if skill == "math" else "knowledge"
    spec["task_type"] = config["task_type"]
    benchmarks = []
    for task_id in shrink_spec.targets_for(skill):
        benchmarks.append(entry_for_id(config, task_id, True))
    spec["benchmarks"] = benchmarks
    spec["request_type"] = "shrink"
    spec["shrink"] = block
    spec["host"] = block["source_model"]
    spec["limits"] = limits
    spec["size"] = shrink_size(limits, block)
    if spec["drafted_by"] == "rules":
        spec["summary"] = shrink_spec.draft_summary(block, estimate)
    else:
        spec["summary"] = clip(spec["summary"] + " " + shrink_spec.plan_summary(block, estimate), 400)


def shrink_size(limits, block):
    """The spec's size block for a shrink: the limit, and the source size as the largest."""
    return {"max_params_b": limits["max_total_params_b"], "min_b": shrink_spec.MIN_LIMIT_B,
            "max_b": max(SIZE_MAX_B, block["source_params_b"]), "host_params_b": block["source_params_b"]}


def edit_shrink_limits(old_limits, body):
    """Limits of a shrink spec after an edit. They may be below the host size, since a shrink is smaller."""
    limits = dict(old_limits)
    incoming = body.get("limits") if isinstance(body.get("limits"), dict) else {}
    size = body.get("size") if isinstance(body.get("size"), dict) else {}
    total = incoming.get("max_total_params_b")
    if total is None:
        total = size.get("max_params_b")
    if total is not None:
        limits["max_total_params_b"] = round(clamp(to_float(total, limits["max_total_params_b"]),
                                                   shrink_spec.MIN_LIMIT_B, 2000.0), 4)
    if incoming.get("max_run_params_b") is not None:
        limits["max_run_params_b"] = to_float(incoming["max_run_params_b"], limits["max_run_params_b"])
    limits["max_run_params_b"] = round(clamp(limits["max_run_params_b"], shrink_spec.MIN_LIMIT_B,
                                             limits["max_total_params_b"]), 4)
    if incoming.get("max_heal_min") is not None:
        limits["max_heal_min"] = clamp_heal(to_float(incoming["max_heal_min"], limits["max_heal_min"]))
    return limits


def default_limits(size, host_params):
    """Limits for a stored spec that has none: the old size limit as the total, no tighter run or heal limit."""
    total = size.get("max_params_b", DEFAULT_SIZE_B)
    return {"max_total_params_b": total, "max_run_params_b": clamp_run(total, total, host_params),
            "max_heal_min": DEFAULT_HEAL_MIN}


def clean_comparators(incoming, old):
    """The measured reference scores a person adds to a spec: {"distill": {task id: score in 0..1}}."""
    out = {"distill": {}}
    source = old
    if isinstance(incoming, dict):
        source = incoming
    if not isinstance(source, dict):
        return out
    distill = source.get("distill")
    if isinstance(distill, dict):
        for task_id in distill:
            value = distill[task_id]
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                out["distill"][str(task_id)] = clamp(float(value), 0.0, 1.0)
    return out


def clean_benchmark(config, item):
    """Return a valid benchmark entry from an edited one, or None when it has no usable id."""
    if not isinstance(item, dict):
        return None
    task_id = item.get("id")
    if not isinstance(task_id, str) or task_id.strip() == "":
        return None
    task_id = task_id.strip()
    kind = item.get("kind")
    if kind not in ("target", "guard"):
        kind = entry_for_id(config, task_id, False)["kind"]
    base = item.get("base")
    if base is None:
        base = base_for(config, task_id)
    else:
        base = clamp(to_float(base, 0.0), 0.0, 1.0)
    out = {"id": task_id, "name": str(item.get("name") or benchmark_name(task_id)), "kind": kind}
    if kind == "target":
        default = target_entry(config, task_id, False)["target"]
        out["target"] = clamp(to_float(item.get("target"), default), 0.0, 1.0)
    else:
        default = guard_entry(config, task_id, False)["floor"]
        out["floor"] = clamp(to_float(item.get("floor"), default), 0.0, 1.0)
    out["base"] = base
    if kind == "target":
        lora = item.get("lora")
        if lora is None:
            lora = LORA_BARS.get(task_id)
        out["lora"] = lora
    out["recommended"] = bool(item.get("recommended", False))
    if item.get("research"):
        out["research"] = True
    if isinstance(item.get("reference_scores"), list):
        out["reference_scores"] = item["reference_scores"]
    return out


def normalize_spec(body, old):
    """Return a checked copy of an edited spec. Raises ValueError with a plain message."""
    if not isinstance(body, dict):
        raise ValueError("the spec must be a JSON object")
    config, catalog = catalog_mod.load_catalog(settings.backend())
    add_task_types(config)
    if old.get("task_type") in config["task_types"]:
        config["task_type"] = old["task_type"]
    host_params = catalog_mod.host_params_b(config["host"])
    spec = dict(old)
    for key in ["title", "summary", "prompt"]:
        if key in body and isinstance(body[key], str):
            spec[key] = body[key]
    spec["title"] = clip(spec["title"], 80) or "Untitled model"

    size = dict(old["size"])
    limits = dict(old.get("limits") or default_limits(size, host_params))
    previous_total = limits["max_total_params_b"]
    is_shrink = old.get("request_type") == "shrink"
    if is_shrink:
        try:
            spec["shrink"] = shrink_spec.check_block(body.get("shrink") or {}, old["shrink"])
        except shrink_spec.ShrinkSpecError as error:
            raise ValueError(str(error))
        limits = edit_shrink_limits(limits, body)
        spec["host"] = spec["shrink"]["source_model"]
    elif isinstance(body.get("size"), dict):
        value = body["size"].get("max_params_b")
        if value is not None:
            limits["max_total_params_b"] = clamp_size(to_float(value, previous_total), host_params)
    if not is_shrink and isinstance(body.get("limits"), dict):
        incoming = body["limits"]
        # The old size field is also sent by older pages. A changed limit wins over a changed size.
        if incoming.get("max_total_params_b") is not None:
            wanted = clamp_size(to_float(incoming["max_total_params_b"], previous_total), host_params)
            if wanted != previous_total or limits["max_total_params_b"] == previous_total:
                limits["max_total_params_b"] = wanted
        if incoming.get("max_run_params_b") is not None:
            limits["max_run_params_b"] = to_float(incoming["max_run_params_b"], limits["max_run_params_b"])
        if incoming.get("max_heal_min") is not None:
            limits["max_heal_min"] = clamp_heal(to_float(incoming["max_heal_min"], limits["max_heal_min"]))
    if is_shrink:
        size = shrink_size(limits, spec["shrink"])
    else:
        sent = body["limits"].get("max_run_params_b") if isinstance(body.get("limits"), dict) else None
        run_edited = sent is not None
        if limits["max_total_params_b"] != previous_total and not run_edited:
            # Grafted parts add to the total and the run size alike, so the run limit moves with the total.
            limits["max_run_params_b"] = (min(limits["max_run_params_b"], previous_total) +
                                          limits["max_total_params_b"] - previous_total)
        limits["max_run_params_b"] = clamp_run(limits["max_run_params_b"], limits["max_total_params_b"],
                                               host_params)
        size["max_params_b"] = limits["max_total_params_b"]
    spec["size"] = size
    spec["limits"] = limits
    spec["comparators"] = clean_comparators(body.get("comparators"), old.get("comparators"))

    budget = dict(old["budget"])
    if isinstance(body.get("budget"), dict):
        incoming = body["budget"]
        budget["evals"] = clamp(to_int(incoming.get("evals"), budget["evals"]), budget["min_evals"],
                                budget["max_evals"])
        budget["compute_min"] = clamp(to_int(incoming.get("compute_min"), budget["compute_min"]),
                                      MIN_COMPUTE_MIN, MAX_COMPUTE_MIN)
    spec["budget"] = budget

    if "benchmarks" in body:
        items = []
        seen = []
        if not isinstance(body["benchmarks"], list):
            raise ValueError("benchmarks must be a list")
        for raw in body["benchmarks"]:
            item = clean_benchmark(config, raw)
            if item is None or item["id"] in seen:
                continue
            seen.append(item["id"])
            items.append(item)
        has_target = False
        for item in items:
            if item["kind"] == "target":
                has_target = True
        if not has_target:
            raise ValueError("the spec needs at least one target benchmark")
        spec["benchmarks"] = items
    spec["available_benchmarks"] = available_benchmarks(config, spec["benchmarks"])
    if is_shrink:
        spec["ingredients"] = []
    else:
        spec["ingredients"] = ingredients_for(config, catalog, spec["benchmarks"])
    spec["stub"] = settings.is_stub()
    spec["id"] = old["id"]
    return spec


def spec_path(spec_id):
    folder = os.path.join(settings.out_dir(), "specs")
    os.makedirs(folder, exist_ok=True)
    return os.path.join(folder, spec_id + ".json")


def store(spec):
    """Keep the spec in memory and write it to outputs/studio/specs/<id>.json."""
    with _lock:
        _specs[spec["id"]] = spec
    with open(spec_path(spec["id"]), "w", encoding="utf-8") as handle:
        json.dump(spec, handle, indent=2)


def get_spec(spec_id):
    """Return the stored spec, reading it from disk when it is not in memory. None when unknown."""
    with _lock:
        spec = _specs.get(spec_id)
    if spec is not None:
        return spec
    if not re.match(r"^[A-Za-z0-9_]+$", str(spec_id)):
        return None
    path = spec_path(spec_id)
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as handle:
        spec = json.load(handle)
    add_missing_fields(spec)
    with _lock:
        _specs[spec_id] = spec
    return spec


def add_missing_fields(spec):
    """Give a spec saved by an older studio the fields it lacks (limits, comparators)."""
    if "limits" not in spec:
        size = spec.get("size") or {}
        host_params = size.get("host_params_b", catalog_mod.DEFAULT_HOST_PARAMS_B)
        spec["limits"] = default_limits(size, host_params)
    if "comparators" not in spec:
        spec["comparators"] = {"distill": {}}
    if "request_type" not in spec:
        spec["request_type"] = "build"


def update_spec(spec_id, body):
    """Replace the editable parts of a stored spec. Returns the new spec, or None when unknown."""
    old = get_spec(spec_id)
    if old is None:
        return None
    spec = normalize_spec(body, old)
    store(spec)
    return spec


def run_overrides(spec, backend, llm_name):
    """The make_config overrides that turn a spec into a swarm run."""
    targets = {}
    guards = []
    for item in spec["benchmarks"]:
        if item["kind"] == "target":
            targets[item["id"]] = 1.0
        else:
            guards.append(item["id"])
    setup = {"tasks": targets, "guards": guards,
             "heal_data": heal_data_for(spec["task_type"])}
    return {
        "backend": backend,
        "llm": llm_name,
        "agents": "auto",
        "eval_budget": spec["budget"]["evals"],
        "compute_budget_s": compute_budget_s(spec["budget"]["evals"], spec["budget"]["compute_min"]),
        "task_type": spec["task_type"],
        "task_types": {spec["task_type"]: setup},
        "host": spec["host"],
        "outputs_dir": settings.out_dir(),
        "catalog_path": catalog_mod.catalog_path_for(backend),
        "max_heal_min": (spec.get("limits") or {}).get("max_heal_min"),
        "studio_limits": dict(spec.get("limits") or {}),
    }


# The stub cost model charges about 300 to 450 compute seconds per eval, so a
# 45 minute budget would end a stub run after about 9 evals. In stub mode the
# compute axis is at least this many seconds per eval, so the eval axis ends the run.
STUB_SECONDS_PER_EVAL = 450.0


def compute_budget_s(evals, compute_min):
    """The compute cap in seconds for a run or a builder session."""
    seconds = float(compute_min) * 60.0
    if settings.is_stub():
        floor = float(evals) * STUB_SECONDS_PER_EVAL
        if floor > seconds:
            seconds = floor
    return seconds
