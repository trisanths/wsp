"""Model profiles: what each open model scores and what it costs to run.

The planner (wsp/planner.py) reads profiles to choose hosts and donors for
a request. One profile is a plain dict:

    {"key": "qwen7b",                       short name, unique in a profile list
     "hf_id": "Qwen/Qwen2.5-7B-Instruct",
     "family": "qwen2",                     config model_type
     "params_b": 7.616,                     total parameters, billions
     "active_params_b": 7.616,              parameters used per token (MoE: less)
     "layers": 28,                          decoder layers
     "hidden": 3584,                        hidden size
     "tokenizer_family": "qwen2",           same value means same tokenization
     "scores": {"mmlu": 0.709, ...},        accuracy as a fraction from 0 to 1
     "decode_tok_s": null,                  batch-1 greedy decode speed, or null
     "source": "...",                       where the config and scores came from
     "score_sources": {"mmlu": "..."},      where each score came from
     "subject_scores": {"mmlu": {"anatomy": {"accuracy": 0.6, "count": 17}}}}

The first eleven keys are the fixed interface. score_sources and
subject_scores are extra keys; code that does not know them can ignore them.

Score names:
    mmlu_dev      1000 seeded MMLU test items (round 6 dev items)
    mmlu_heldout  2000 other MMLU test items (round 7 held-out items)
    mmlu          the dev score (the held-out score if no dev score)
    medmcqa       1000 seeded MedMCQA validation items (round 10)
    mmlu_pro      500 seeded MMLU-Pro test items (modal_profile.py)
    medqa         500 seeded MedQA-USMLE 4-option train items (modal_profile.py)
    gsm8k         the 200-item GSM8K search split; the source says if the
                  prompt was raw or the chat template
    gsm8k_raw     the same items with the raw "Question: ... Answer:" prompt
    triviaqa      1000 TriviaQA validation questions (round 7)

The planner reads the names that match a request benchmark (mmlu,
medmcqa, medqa, mmlu_pro, gsm8k). Those scores come only from items that
modal_make.py never scores: modal_make.py measures MMLU on the round 7
held-out items, MedMCQA and MMLU-Pro on positions 1000 on, MedQA on its
test split, and GSM8K on its test split. So "mmlu" is the dev score, and
mmlu_heldout is kept only to report. subject_scores["mmlu"] uses the dev
records only, for the same reason.
All MCQ scores take the argmax of the option-letter log-probabilities
after the round 6 prompt (wsp.mcq). A missing score is a missing key, never
0. Use get_score(profile, name) to read one.

Build profiles.json from the results files (no network):

    .venv/bin/python -m wsp.profiles build
    .venv/bin/python -m wsp.profiles merge results/profile_rows.json

The second command folds measured rows from modal_profile.py into the file.
"""

import json
import os
import re
import sys

# Hard-coded architecture facts from each model's Hugging Face config.json.
# params_b is approximate (sum of tensor sizes, tied embeddings counted
# once). modal_profile.py measures the exact numbers on a GPU.
KNOWN_MODELS = {
    "llama1b": {"hf_id": "unsloth/Llama-3.2-1B-Instruct", "family": "llama", "params_b": 1.236,
                "layers": 16, "hidden": 2048, "tokenizer_family": "llama3"},
    "llama3b": {"hf_id": "unsloth/Llama-3.2-3B-Instruct", "family": "llama", "params_b": 3.213,
                "layers": 28, "hidden": 3072, "tokenizer_family": "llama3"},
    "llama8b": {"hf_id": "unsloth/Meta-Llama-3.1-8B-Instruct", "family": "llama", "params_b": 8.030,
                "layers": 32, "hidden": 4096, "tokenizer_family": "llama3"},
    "qwen05": {"hf_id": "Qwen/Qwen2.5-0.5B-Instruct", "family": "qwen2", "params_b": 0.494,
               "layers": 24, "hidden": 896, "tokenizer_family": "qwen2"},
    "qwen15b": {"hf_id": "Qwen/Qwen2.5-1.5B-Instruct", "family": "qwen2", "params_b": 1.544,
                "layers": 28, "hidden": 1536, "tokenizer_family": "qwen2"},
    "qwen3b": {"hf_id": "Qwen/Qwen2.5-3B-Instruct", "family": "qwen2", "params_b": 3.086,
               "layers": 36, "hidden": 2048, "tokenizer_family": "qwen2"},
    "qwen7b": {"hf_id": "Qwen/Qwen2.5-7B-Instruct", "family": "qwen2", "params_b": 7.616,
               "layers": 28, "hidden": 3584, "tokenizer_family": "qwen2"},
    "qwen14b": {"hf_id": "Qwen/Qwen2.5-14B-Instruct", "family": "qwen2", "params_b": 14.770,
                "layers": 48, "hidden": 5120, "tokenizer_family": "qwen2"},
    "qwen32b": {"hf_id": "Qwen/Qwen2.5-32B-Instruct", "family": "qwen2", "params_b": 32.764,
                "layers": 64, "hidden": 5120, "tokenizer_family": "qwen2"},
    "qwen_math15b": {"hf_id": "Qwen/Qwen2.5-Math-1.5B-Instruct", "family": "qwen2", "params_b": 1.544,
                     "layers": 28, "hidden": 1536, "tokenizer_family": "qwen2"},
    "gemma9b": {"hf_id": "unsloth/gemma-2-9b-it", "family": "gemma2", "params_b": 9.242,
                "layers": 42, "hidden": 3584, "tokenizer_family": "gemma"},
    "phi35": {"hf_id": "microsoft/Phi-3.5-mini-instruct", "family": "phi3", "params_b": 3.821,
              "layers": 32, "hidden": 3072, "tokenizer_family": "phi3"},
    "phi4": {"hf_id": "microsoft/phi-4", "family": "phi3", "params_b": 14.660,
             "layers": 40, "hidden": 5120, "tokenizer_family": "phi4"},
    "mistral7b": {"hf_id": "unsloth/mistral-7b-instruct-v0.3", "family": "mistral", "params_b": 7.248,
                  "layers": 32, "hidden": 4096, "tokenizer_family": "mistral_v3"},
    "biomistral": {"hf_id": "BioMistral/BioMistral-7B", "family": "mistral", "params_b": 7.242,
                   "layers": 32, "hidden": 4096, "tokenizer_family": "mistral_v1"},
    "lfm2_350m_math": {"hf_id": "LiquidAI/LFM2-350M-Math", "family": "lfm2", "params_b": 0.354,
                       "layers": 16, "hidden": 1024, "tokenizer_family": "lfm2"},
    "lfm25_12b": {"hf_id": "LiquidAI/LFM2.5-1.2B-Instruct", "family": "lfm2", "params_b": 1.170,
                  "layers": 16, "hidden": 2048, "tokenizer_family": "lfm2"},
    "lfm25_12b_thinking": {"hf_id": "LiquidAI/LFM2.5-1.2B-Thinking", "family": "lfm2", "params_b": 1.170,
                           "layers": 16, "hidden": 2048, "tokenizer_family": "lfm2"},
}

# The fixed profile keys that every consumer can rely on.
INTERFACE_KEYS = ["key", "hf_id", "family", "params_b", "active_params_b", "layers", "hidden",
                  "tokenizer_family", "scores", "decode_tok_s", "source"]

CONFIG_SOURCE = "config: hard-coded table in wsp/profiles.py from Hugging Face config.json (params approximate)"

# Short names that the results files use for models, mapped to Hugging Face ids.
RESULT_KEYS = {
    "llama1b": "unsloth/Llama-3.2-1B-Instruct",
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

# What each score name measures, written into score_sources.
SCORE_PROTOCOLS = {
    "mmlu_dev": "1000 seeded MMLU test items (round 6 dev), letter log-prob",
    "mmlu_heldout": "2000 seeded MMLU test items (round 7 held-out), letter log-prob",
    "medmcqa": "1000 seeded MedMCQA validation items (round 10), letter log-prob",
    "mmlu_pro": "500 seeded MMLU-Pro test items, letter log-prob",
    "medqa": "500 seeded MedQA-USMLE 4-option train items, letter log-prob",
    "gsm8k_raw": "GSM8K search split (200 train items), raw 'Question: ... Answer:' prompt",
    "gsm8k_chat": ("GSM8K search split (200 train items), chat template, 'Solve it step by step. "
                   "End with: The answer is <number>.' (the modal_make.py prompt)"),
    "triviaqa": "1000 TriviaQA validation questions, alias match on a 16-token answer",
}


class Profile:
    """A thin wrapper around one profile dict. to_dict() gives the dict back.

    profile = Profile.blank("qwen7b")        # hard-coded config, no scores
    profile.set_score("mmlu", 0.709, "results/round10_rows.json")
    profile.score("mmlu")                    # 0.709, or None when missing
    """

    def __init__(self, data):
        self.data = data
        if "scores" not in self.data or self.data["scores"] is None:
            self.data["scores"] = {}
        if "score_sources" not in self.data or self.data["score_sources"] is None:
            self.data["score_sources"] = {}

    @classmethod
    def blank(cls, key, hf_id=None):
        """A profile with config fields from KNOWN_MODELS (or None) and no scores."""
        data = {"key": key, "hf_id": hf_id, "family": None, "params_b": None, "active_params_b": None,
                "layers": None, "hidden": None, "tokenizer_family": None, "scores": {},
                "decode_tok_s": None, "source": "", "score_sources": {}, "subject_scores": {}}
        # The table applies only to its own model, not to another id that
        # happens to get the same key.
        if key in KNOWN_MODELS and (hf_id is None or hf_id == KNOWN_MODELS[key]["hf_id"]):
            known = KNOWN_MODELS[key]
            for name in ["hf_id", "family", "params_b", "layers", "hidden", "tokenizer_family"]:
                data[name] = known[name]
            # Every model in the table is dense.
            data["active_params_b"] = known["params_b"]
            data["source"] = CONFIG_SOURCE
        return cls(data)

    @property
    def key(self):
        return self.data["key"]

    def score(self, name):
        return get_score(self.data, name)

    def set_score(self, name, value, source, overwrite=False):
        """Set one score unless it is already set (or overwrite is True)."""
        if value is None:
            return False
        if name in self.data["scores"] and not overwrite:
            return False
        self.data["scores"][name] = round(float(value), 4)
        self.data["score_sources"][name] = source
        return True

    def to_dict(self):
        return self.data


# The order in which "mmlu" falls back to the two MMLU item sets. Dev
# comes first, because modal_make.py scores the held-out items.
MMLU_ORDER = ["mmlu_dev", "mmlu_heldout"]


def get_score(profile, name):
    """One score from a profile dict, or None when it is missing.

    "mmlu" falls back to mmlu_dev, then mmlu_heldout, so older profiles
    without the alias still answer.
    """
    if isinstance(profile, Profile):
        profile = profile.data
    scores = profile.get("scores") or {}
    if name in scores and scores[name] is not None:
        return scores[name]
    if name == "mmlu":
        for other in MMLU_ORDER:
            if other in scores and scores[other] is not None:
                return scores[other]
    return None


def slug_key(hf_id):
    """A profile key for a Hugging Face id that is not in KNOWN_MODELS."""
    name = hf_id.split("/")[-1].lower()
    name = re.sub(r"[^a-z0-9]+", "_", name)
    return name.strip("_")


def key_for(hf_id):
    """The KNOWN_MODELS key for a Hugging Face id, else a slug of the id."""
    for key in KNOWN_MODELS:
        if KNOWN_MODELS[key]["hf_id"] == hf_id:
            return key
    return slug_key(hf_id)


def hf_id_for(name):
    """Resolve a results-file model name (key or Hugging Face id) to a Hugging Face id."""
    if name in RESULT_KEYS:
        return RESULT_KEYS[name]
    if name in KNOWN_MODELS:
        return KNOWN_MODELS[name]["hf_id"]
    return name


def find_profile(profiles, name):
    """The profile dict whose key or hf_id is name, or None."""
    for profile in profiles:
        if profile.get("key") == name or profile.get("hf_id") == name:
            return profile
    return None


def by_key(profiles):
    """{key: profile dict} for a profile list."""
    table = {}
    for profile in profiles:
        table[profile["key"]] = profile
    return table


def load_profiles(path="profiles.json"):
    """A list of profile dicts. A missing file gives an empty list.

    A missing interface key loads as None, and a missing scores,
    score_sources, or subject_scores loads as {}, so callers can index
    every key.
    """
    if not os.path.exists(path):
        return []
    with open(path) as f:
        data = json.load(f)
    if isinstance(data, dict) and "profiles" in data:
        data = data["profiles"]
    elif isinstance(data, dict):
        # A {key: profile} table, as wsp.make also accepts.
        keyed = []
        for key in data:
            item = dict(data[key])
            if not item.get("key"):
                item["key"] = key
            keyed.append(item)
        data = keyed
    profiles = []
    for item in data:
        full = dict(item)
        for name in INTERFACE_KEYS:
            if name not in full:
                full[name] = None
        for name in ["scores", "score_sources", "subject_scores"]:
            if full.get(name) is None:
                full[name] = {}
        profiles.append(full)
    return profiles


def save_profiles(profiles, path="profiles.json"):
    """Write profiles (dicts or Profile objects) as a JSON list."""
    out = []
    for profile in profiles:
        if isinstance(profile, Profile):
            profile = profile.data
        out.append(profile)
    out = sort_profiles(out)
    with open(path, "w") as f:
        json.dump(out, f, indent=1)
        f.write("\n")
    return path


def sort_profiles(profiles):
    """Smallest model first; models without a size go last."""
    def order(profile):
        size = profile.get("params_b")
        if size is None:
            return (1, 0.0, profile.get("key") or "")
        return (0, size, profile.get("key") or "")
    return sorted(profiles, key=order)


# Reading the results files.

def read_json(results_dir, name):
    path = os.path.join(results_dir, name)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


class ProfileTable:
    """Profiles by Hugging Face id while building. New ids get a blank profile."""

    def __init__(self):
        self.profiles = {}

    def get(self, hf_id):
        if hf_id not in self.profiles:
            self.profiles[hf_id] = Profile.blank(key_for(hf_id), hf_id)
        return self.profiles[hf_id]

    def all(self):
        out = []
        for hf_id in self.profiles:
            out.append(self.profiles[hf_id].data)
        return out


def record_accuracy(records):
    right = 0
    for record in records:
        if record["correct"]:
            right += 1
    if len(records) == 0:
        return None
    return right / len(records)


def subject_table(records):
    """{subject: {"accuracy", "count"}} for a list of per-item records."""
    counts = {}
    rights = {}
    for record in records:
        subject = record.get("subject") or "unknown"
        if subject not in counts:
            counts[subject] = 0
            rights[subject] = 0
        counts[subject] += 1
        if record["correct"]:
            rights[subject] += 1
    out = {}
    for subject in sorted(counts):
        out[subject] = {"accuracy": round(rights[subject] / counts[subject], 4), "count": counts[subject]}
    return out


def add_round10(table, results_dir):
    """mmlu_dev, mmlu_heldout, medmcqa, and subject scores from round 10."""
    rows = read_json(results_dir, "round10_rows.json")
    if rows is None:
        return 0
    source = os.path.join("results", "round10_rows.json")
    count = 0
    for row in rows:
        hf_id = row.get("model") or hf_id_for(row["key"])
        profile = table.get(hf_id)
        by_set = {}
        for record in row.get("records", []):
            name = record["set"]
            if name not in by_set:
                by_set[name] = []
            by_set[name].append(record)
        summary = row.get("summary") or {}
        for name in ["mmlu_dev", "mmlu_heldout", "medmcqa"]:
            value = None
            if name in summary and summary[name][1] > 0:
                value = summary[name][0] / summary[name][1]
            elif name in by_set:
                value = record_accuracy(by_set[name])
            if profile.set_score(name, value, source + ": " + SCORE_PROTOCOLS[name], overwrite=True):
                count += 1
        # The planner reads subject_scores["mmlu"], so it holds the dev
        # records only. The held-out records are kept apart, to report.
        for name in ["mmlu_dev", "mmlu_heldout"]:
            if name in by_set:
                profile.data["subject_scores"][name] = subject_table(by_set[name])
        add_subject_alias(profile)
        if "medmcqa" in by_set:
            profile.data["subject_scores"]["medmcqa"] = subject_table(by_set["medmcqa"])
    return count


def add_round8(table, results_dir):
    """Donor-alone and host-alone MMLU dev scores from round 8 (used only where round 10 has none)."""
    rows = read_json(results_dir, "round8_rows.json")
    if rows is None:
        return 0
    source = os.path.join("results", "round8_rows.json") + ": " + SCORE_PROTOCOLS["mmlu_dev"]
    count = 0
    for row in rows:
        if row.get("kind") not in ["donor_alone", "host_alone"]:
            continue
        profile = table.get(hf_id_for(row["key"]))
        if profile.set_score("mmlu_dev", row.get("accuracy"), source):
            count += 1
    return count


def add_round9(table, results_dir):
    """Model-alone MMLU dev scores from round 9 (only where nothing else has one)."""
    rows = read_json(results_dir, "round9_rows.json")
    if rows is None:
        return 0
    source = os.path.join("results", "round9_rows.json") + ": " + SCORE_PROTOCOLS["mmlu_dev"]
    count = 0
    for row in rows:
        if row.get("kind") != "alone":
            continue
        profile = table.get(hf_id_for(row["host"]))
        if profile.set_score("mmlu_dev", row.get("accuracy"), source):
            count += 1
    return count


def add_round7(table, results_dir):
    """Round 7: held-out MMLU for host and donor, donor scaling (dev), TriviaQA."""
    rows = read_json(results_dir, "round7_rows.json")
    if rows is None:
        return 0
    path = os.path.join("results", "round7_rows.json")
    host = "unsloth/Llama-3.2-1B-Instruct"
    count = 0
    for row in rows:
        part = row.get("part")
        arm = row.get("arm")
        cfg = row.get("cfg") or {}
        if arm == "host":
            hf_id = host
        elif arm == "donor":
            hf_id = cfg.get("donor")
        else:
            continue
        if hf_id is None:
            continue
        if part == "heldout":
            name = "mmlu_heldout"
        elif part == "scaling":
            name = "mmlu_dev"
        elif part == "trivia":
            name = "triviaqa"
        else:
            continue
        if table.get(hf_id).set_score(name, row.get("accuracy"), path + ": " + SCORE_PROTOCOLS[name]):
            count += 1
    return count


def add_baselines(table, results_dir):
    """GSM8K raw-prompt donor ceilings from the wsp_eval baselines (vLLM)."""
    data = read_json(results_dir, "baselines_rows.json")
    if data is None:
        return 0
    source = os.path.join("results", "baselines_rows.json") + ": " + SCORE_PROTOCOLS["gsm8k_raw"]
    count = 0
    for row in data.get("summary", []):
        arm = row.get("arm", "")
        if not arm.startswith("ceiling "):
            continue
        hf_id = arm[len("ceiling "):].strip()
        if table.get(hf_id).set_score("gsm8k", row.get("score_mean"), source):
            count += 1
    return count


def add_round5(table, results_dir):
    """GSM8K raw-prompt scores from round 5 (host, Qwen-Math donor, LFM2-350M-Math)."""
    data = read_json(results_dir, "round5_rows.json")
    if data is None:
        return 0
    source = os.path.join("results", "round5_rows.json") + ": " + SCORE_PROTOCOLS["gsm8k_raw"]
    arms = {"host_only": data.get("host"), "donor_alone": data.get("donor"),
            "lfm_alone": "LiquidAI/LFM2-350M-Math"}
    count = 0
    for row in data.get("rows", []):
        hf_id = arms.get(row.get("arm"))
        if hf_id is None:
            continue
        if table.get(hf_id).set_score("gsm8k", row.get("accuracy"), source):
            count += 1
    return count


def add_round4(table, results_dir):
    """GSM8K raw-prompt score of the untrained hosts in round 4."""
    data = read_json(results_dir, "round4_rows.json")
    if data is None:
        return 0
    source = os.path.join("results", "round4_rows.json") + ": " + SCORE_PROTOCOLS["gsm8k_raw"]
    count = 0
    for row in data.get("rows", []):
        if row.get("arm") != "host":
            continue
        if table.get(hf_id_for(row["host"])).set_score("gsm8k", row.get("accuracy"), source):
            count += 1
    return count


def add_mmlu_alias(profile):
    """scores["mmlu"]: the dev score, else the held-out score."""
    for name in MMLU_ORDER:
        value = profile.score(name)
        if value is not None:
            profile.set_score("mmlu", value, "alias of " + name, overwrite=True)
            return


def add_subject_alias(profile):
    """subject_scores["mmlu"]: the dev subject table, else the held-out one."""
    subjects = profile.data.get("subject_scores")
    if subjects is None:
        return
    for name in MMLU_ORDER:
        if name in subjects and subjects[name]:
            subjects["mmlu"] = subjects[name]
            return


def add_gsm8k_raw(profile):
    """scores["gsm8k_raw"]: a copy of a raw-prompt gsm8k score from the results files.

    A modal_profile.py run replaces "gsm8k" with a chat-template score;
    this copy keeps the raw-prompt number next to it.
    """
    sources = profile.data["score_sources"]
    source = sources.get("gsm8k")
    if not isinstance(source, str):
        return
    if source.endswith(SCORE_PROTOCOLS["gsm8k_raw"]):
        profile.set_score("gsm8k_raw", profile.score("gsm8k"), source)


def finish_source(profile):
    """The profile-level source: the config source plus each distinct score file."""
    parts = []
    if profile.data.get("source"):
        parts.append(profile.data["source"].split("; scores: ")[0])
    files = []
    sources = profile.data["score_sources"]
    for name in sorted(sources):
        if not isinstance(sources[name], str):
            continue
        place = sources[name].split(":")[0]
        if place.startswith("alias of"):
            continue
        if place not in files:
            files.append(place)
    if len(files) > 0:
        parts.append("scores: " + ", ".join(files))
    profile.data["source"] = "; ".join(parts)


def build_profiles_from_results(results_dir="results", include_known=True):
    """Profiles from the results files in results_dir. Needs no network.

    Score priority (the first file with a value wins for each score):
      mmlu_dev      round10, round8 (alone rows), round7 scaling, round9 alone
      mmlu_heldout  round10, round7 held-out
      medmcqa       round10
      gsm8k         baselines ceilings (raw prompt), round5, round4
      triviaqa      round7
    Then "mmlu" is set from the dev score and "gsm8k_raw" copies the
    raw-prompt gsm8k score.
    Missing files are skipped. With include_known, every KNOWN_MODELS entry
    gets a profile even when no file scores it. Returns a list of dicts.
    """
    table = ProfileTable()
    if include_known:
        for key in KNOWN_MODELS:
            table.get(KNOWN_MODELS[key]["hf_id"])
    add_round10(table, results_dir)
    add_round8(table, results_dir)
    add_round7(table, results_dir)
    add_round9(table, results_dir)
    add_baselines(table, results_dir)
    add_round5(table, results_dir)
    add_round4(table, results_dir)
    for hf_id in table.profiles:
        profile = table.profiles[hf_id]
        add_mmlu_alias(profile)
        add_gsm8k_raw(profile)
        finish_source(profile)
    return sort_profiles(table.all())


# Measured rows from modal_profile.py.

def tokenizer_family_guess(model_type, vocab_size):
    """A tokenizer family name from the config, for models outside KNOWN_MODELS."""
    if model_type in ["qwen2", "qwen2_moe", "qwen3", "qwen3_moe"]:
        return "qwen2"
    if model_type in ["gemma", "gemma2"]:
        return "gemma"
    if model_type in ["gemma3", "gemma3_text"]:
        # Gemma 3 has a new 262k vocabulary.
        return "gemma3"
    if model_type in ["lfm2", "lfm2_moe"]:
        return "lfm2"
    if model_type == "llama" and vocab_size == 128256:
        return "llama3"
    if model_type == "phi3" and vocab_size == 32064:
        return "phi3"
    if model_type == "phi3" and vocab_size == 100352:
        return "phi4"
    if model_type == "mistral" and vocab_size == 32768:
        return "mistral_v3"
    if model_type == "mistral" and vocab_size == 32000:
        return "mistral_v1"
    return str(model_type) + "_vocab" + str(vocab_size)


def count_params(model, config=None, scale=1e9):
    """(total, active) parameter counts, in billions by default (scale=1 gives raw counts).

    For a mixture of experts (config has num_experts_per_tok and
    num_local_experts, num_experts, or n_routed_experts), active counts only
    k of E experts: total - expert_params * (1 - k / E). Parameters with
    ".experts." in the name are the expert parameters.
    """
    total = 0
    experts = 0
    for name, param in model.named_parameters():
        total += param.numel()
        if ".experts." in name:
            experts += param.numel()
    active = total
    if config is not None:
        k = getattr(config, "num_experts_per_tok", None)
        count = None
        for field in ["num_local_experts", "num_experts", "n_routed_experts"]:
            if getattr(config, field, None):
                count = getattr(config, field)
                break
        if k and count and experts > 0:
            active = total - experts * (1.0 - k / count)
    return round(total / scale, 3), round(active / scale, 3)


# A number as modal_make.py reads it: digits with optional commas and decimals.
NUMBER = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def clean_number(text):
    text = text.replace(",", "").rstrip(".")
    try:
        return float(text)
    except ValueError:
        return None


def gsm8k_number(text):
    """The final answer in a generated GSM8K solution, as a float, or None.

    This reads answers the same way as modal_make.final_number, so profile
    scores and measured scores agree: cut the text at a new "Question:",
    then take the first number after the last "the answer is" (any case),
    else after the last "####", else the last number in the text.
    """
    if text is None:
        return None
    text = text.split("\nQuestion:")[0]
    lowered = text.lower()
    for marker in ["the answer is", "####"]:
        position = lowered.rfind(marker)
        if position != -1:
            found = NUMBER.findall(text[position + len(marker):])
            if len(found) > 0:
                return clean_number(found[0])
    found = NUMBER.findall(text)
    if len(found) == 0:
        return None
    return clean_number(found[-1])


def gsm8k_gold(answer):
    """The number after the last "####" in a GSM8K reference answer, or None."""
    if answer is None or "####" not in answer:
        return None
    found = NUMBER.findall(answer.split("####")[-1])
    if len(found) == 0:
        return None
    return clean_number(found[0])


def unique_key(profiles, key, hf_id):
    """key, or key plus the org name (then a number) when another model already has it."""
    taken = []
    for profile in profiles:
        if profile.get("hf_id") != hf_id:
            taken.append(profile.get("key"))
    if key not in taken:
        return key
    org = ""
    if "/" in hf_id:
        org = slug_key(hf_id.split("/")[0])
    if org:
        candidate = key + "_" + org
        if candidate not in taken:
            return candidate
    number = 2
    while key + "_" + str(number) in taken:
        number += 1
    return key + "_" + str(number)


def merge_measurements(profiles, rows, source="modal_profile.py"):
    """Fold measured rows from modal_profile.py into a profile list.

    A row is {"hf_id", "key", "family", "params_b", "active_params_b",
    "layers", "hidden", "tokenizer_family", "vocab_size", "scores",
    "score_protocols", "subject_scores", "decode_tok_s", "gpu", "errors"}.
    Measured values replace stored ones; a None or missing value keeps the
    stored one. A known tokenizer_family is kept, because the measured one
    is a guess. Returns a new sorted list of dicts.
    """
    out = []
    for profile in profiles:
        if isinstance(profile, Profile):
            profile = profile.data
        out.append(profile)
    for row in rows:
        hf_id = row.get("hf_id")
        if hf_id is None:
            continue
        found = find_profile(out, hf_id)
        if found is None:
            key = unique_key(out, row.get("key") or key_for(hf_id), hf_id)
            found = Profile.blank(key, hf_id).data
            out.append(found)
        profile = Profile(found)
        if "subject_scores" not in found or found["subject_scores"] is None:
            found["subject_scores"] = {}
        gpu = row.get("gpu") or "GPU"
        for name in ["family", "params_b", "active_params_b", "layers", "hidden"]:
            if row.get(name) is not None:
                found[name] = row[name]
        if found.get("tokenizer_family") is None and row.get("tokenizer_family") is not None:
            found["tokenizer_family"] = row["tokenizer_family"]
        if row.get("decode_tok_s") is not None:
            found["decode_tok_s"] = row["decode_tok_s"]
            found["score_sources"]["decode_tok_s"] = source + " (" + gpu + ", batch 1, greedy, 128 tokens, HF generate)"
        scores = row.get("scores") or {}
        protocols = row.get("score_protocols") or {}
        for name in scores:
            protocol = protocols.get(name) or SCORE_PROTOCOLS.get(name) or name
            profile.set_score(name, scores[name], source + " (" + gpu + "): " + protocol, overwrite=True)
            # A model without a chat template was measured on the raw prompt.
            if name == "gsm8k" and protocol == SCORE_PROTOCOLS["gsm8k_raw"]:
                profile.set_score("gsm8k_raw", scores[name], source + " (" + gpu + "): " + protocol, overwrite=True)
        subjects = row.get("subject_scores") or {}
        for name in subjects:
            found["subject_scores"][name] = subjects[name]
        add_mmlu_alias(profile)
        add_subject_alias(profile)
        if row.get("params_b") is not None:
            found["source"] = "config: measured by " + source + " (" + gpu + ")"
        finish_source(profile)
    return sort_profiles(out)


def read_rows(path):
    with open(path) as f:
        rows = json.load(f)
    if isinstance(rows, dict):
        rows = [rows]
    return rows


def main(argv):
    if len(argv) < 1 or argv[0] not in ["build", "merge"]:
        print("usage: python -m wsp.profiles build [results_dir] [out]")
        print("       python -m wsp.profiles merge rows.json [profiles.json]")
        return 1
    if argv[0] == "build":
        results_dir = "results"
        out = "profiles.json"
        if len(argv) > 1:
            results_dir = argv[1]
        if len(argv) > 2:
            out = argv[2]
        profiles = build_profiles_from_results(results_dir)
        save_profiles(profiles, out)
        print(f"wrote {len(profiles)} profiles to {out}")
        return 0
    if len(argv) < 2:
        print("merge needs a rows file")
        return 1
    out = "profiles.json"
    if len(argv) > 2:
        out = argv[2]
    profiles = merge_measurements(load_profiles(out), read_rows(argv[1]))
    save_profiles(profiles, out)
    print(f"merged {argv[1]} into {out} ({len(profiles)} profiles)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
