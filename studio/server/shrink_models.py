"""Facts about the models a shrink request can start from, and the choice of a source model.

The facts come from the library model profiles (library/models/*.json) plus
a short table of what the profiles lack: the number of experts a token
uses, the size of the embedding, and a few models that wsp.slim supports but
the library does not list yet. Every number here feeds an estimate, so each
size the studio shows from it is labeled as an estimate.

wsp.slim is not imported here. The server stays light and does not load
torch. The rule and family lists below are copies, and a test checks that
they match wsp.slim.
"""

import re

from studio.server import library

# Families wsp.slim.FAMILIES can remove experts from, with the usual number of experts a token uses.
MOE_FAMILIES = {"qwen3_moe": 8, "qwen2_moe": 4, "olmoe": 8, "mixtral": 2, "minimax": 8, "glm4_moe": 8,
                "deepseek_v3": 8}

# Dense families wsp.slim's layer, neuron and head removal is built for (plain decoder layers).
DENSE_FAMILIES = ["qwen2", "qwen3", "llama", "mistral", "phi3", "gemma2"]

# Rules for expert removal (wsp.slim.RULES) and for dense slimming. A dense rule picks both the layer
# rule and the width rule: skill is skill_cost layers and skill neurons, and so on.
EXPERT_RULES = ["freq", "saliency", "selective", "random", "general"]
DENSE_RULES = ["skill", "skill_cost", "specific", "general", "random"]
DEFAULT_EXPERT_RULE = "saliency"
DEFAULT_DENSE_RULE = "skill"

# Facts the library profiles lack. active_params_b counts everything one token touches, the input
# embedding included. embed_b is vocabulary times hidden size. tied is true when the head reuses it.
FACTS = {
    "Qwen/Qwen3-30B-A3B": {"active_params_b": 3.34, "top_k": 8, "embed_b": 0.311, "tied": False},
    "Qwen/Qwen3-30B-A3B-Instruct-2507": {"active_params_b": 3.34, "top_k": 8, "embed_b": 0.311, "tied": False},
    "allenai/OLMoE-1B-7B-0924-Instruct": {"active_params_b": 1.3, "top_k": 8, "embed_b": 0.103, "tied": False},
    "mistralai/Mixtral-8x7B-Instruct-v0.1": {"active_params_b": 12.9, "top_k": 2, "embed_b": 0.131,
                                             "tied": False},
}

# Models wsp.slim supports that the library has no profile for.
EXTRA = {
    "Qwen/Qwen2.5-Coder-7B-Instruct": {"params_b": 7.616, "layers": 28, "hidden": 3584, "family": "qwen2",
                                       "skills": ["code", "general"], "license": "apache-2.0",
                                       "embed_b": 0.545, "tied": False},
    "Qwen/Qwen2.5-Coder-3B-Instruct": {"params_b": 3.086, "layers": 36, "hidden": 2048, "family": "qwen2",
                                       "skills": ["code", "general"], "license": "qwen-research",
                                       "embed_b": 0.311, "tied": True},
}

# Vocabulary sizes by family, used when a profile has none.
DEFAULT_VOCAB = {"qwen2": 151936, "qwen3": 151936, "llama": 128256, "mistral": 32768, "gemma2": 256000,
                 "phi3": 32064}
DEFAULT_VOCAB_ANY = 128000
TIED_BELOW_B = 3.5
DEFAULT_EXPERT_SHARE = 0.9
# A source must be at least this many times the target size, or the cut is too small to be worth a shrink.
MIN_SOURCE_RATIO = 1.3


def number(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def build_facts(hf_id, info):
    """One model's facts, or None when the model cannot be a source."""
    params = number(info.get("params_b"))
    if params is None or params <= 0:
        return None
    family = str(info.get("family") or "")
    layers = info.get("layers")
    if not isinstance(layers, int):
        return None
    table = FACTS.get(hf_id) or {}
    experts = info.get("experts")
    is_moe = isinstance(experts, int) and experts > 1 and family in MOE_FAMILIES
    out = {"hf_id": hf_id, "title": info.get("title") or hf_id, "family": family, "params_b": params,
           "layers": layers, "hidden": info.get("hidden"), "skills": list(info.get("skills") or []),
           "license": info.get("license"), "scores": info.get("scores") or {},
           "kind": "moe" if is_moe else "dense", "experts": experts if is_moe else None, "top_k": None,
           "active_params_b": None, "supported": True, "from_library": info.get("from_library", True)}
    if is_moe:
        top_k = info.get("experts_per_token") or table.get("top_k") or MOE_FAMILIES[family]
        out["top_k"] = min(int(top_k), experts)
        out["active_params_b"] = number(table.get("active_params_b")) or number(info.get("active_params_b"))
    elif family in MOE_FAMILIES or (isinstance(experts, int) and experts > 1):
        # An MoE family that wsp.slim cannot slim, or a profile with experts of an unknown family.
        out["supported"] = False
        out["kind"] = "moe"
        out["experts"] = experts
    hidden = info.get("hidden")
    embed = number(table.get("embed_b")) or number(info.get("embed_b"))
    if embed is None and isinstance(hidden, int):
        vocab = info.get("vocab_size") or DEFAULT_VOCAB.get(family) or DEFAULT_VOCAB.get(family[:5])
        embed = hidden * (vocab or DEFAULT_VOCAB_ANY) / 1e9
    out["embed_b"] = embed or 0.0
    tied = table.get("tied")
    if tied is None:
        tied = info.get("tied")
    if tied is None:
        tied = params <= TIED_BELOW_B
    out["tied"] = bool(tied)
    if out["kind"] == "dense" and (out["layers"] < 2 or family not in DENSE_FAMILIES):
        out["supported"] = False
    return out


def all_facts():
    """{hf id: facts} for every model that can be a source: the library's models and the studio's extras."""
    models = {}
    for hf_id, info in library.get()["models"].items():
        found = build_facts(hf_id, info)
        if found is not None:
            models[hf_id] = found
    for hf_id, info in EXTRA.items():
        if hf_id in models:
            continue
        merged = dict(info)
        merged["from_library"] = False
        found = build_facts(hf_id, merged)
        if found is not None:
            models[hf_id] = found
    return models


def facts_for(hf_id):
    return all_facts().get(hf_id)


def source_ids():
    """The ids a shrink can start from, sorted."""
    out = []
    models = all_facts()
    for hf_id in sorted(models):
        if models[hf_id]["supported"]:
            out.append(hf_id)
    return out


def run_params_b(facts):
    """Parameters one token runs in the full source model (the input embedding is a lookup, so it does not count)."""
    if facts["kind"] == "moe":
        active = facts["active_params_b"]
        if active is None:
            return facts["params_b"] * 0.12
        return max(0.0, active - (0.0 if facts["tied"] else facts["embed_b"]))
    if facts["tied"]:
        return facts["params_b"]
    return max(0.0, facts["params_b"] - facts["embed_b"])


def expert_split(facts):
    """(parameters in all experts, parameters outside the experts, billions) for an MoE model."""
    total = facts["params_b"]
    experts = facts["experts"]
    top_k = facts["top_k"]
    active = facts["active_params_b"]
    if active is not None and top_k < experts:
        expert_b = (total - active) / (1.0 - float(top_k) / experts)
    else:
        expert_b = total * DEFAULT_EXPERT_SHARE
    expert_b = min(max(expert_b, 0.0), total)
    return expert_b, total - expert_b


# Choosing a source.

SKILL_WORDS = [
    ("code", ["code", "coding", "coder", "programming", "programmer", "python", "software", "developer"]),
    ("math", ["math", "maths", "mathematics", "arithmetic", "algebra", "calculus", "gsm8k"]),
    ("medicine", ["medical", "medicine", "clinical", "health", "doctor", "biomedical", "healthcare"]),
    ("prose", ["prose", "writing", "story", "stories", "essay", "essays", "creative"]),
    ("general", ["general", "chat", "chatbot", "assistant", "conversation", "instruction"]),
]
SKILL_LABELS = {"code": "coding text", "math": "math text", "medicine": "medical text",
                "general": "general chat text", "prose": "prose"}
# The library tags a model with these words for each pool.
SKILL_TAGS = {"code": ["code"], "math": ["math"], "medicine": ["medicine"], "general": ["general", "knowledge"],
              "prose": ["general"]}


def skill_from_text(text):
    """The healdata pool name the request is about, or None."""
    lower = " " + re.sub(r"[^a-z0-9]+", " ", str(text or "").lower()) + " "
    for name, words in SKILL_WORDS:
        for word in words:
            if " " + word + " " in lower:
                return name
    return None


def normal(text):
    return re.sub(r"[^a-z0-9]", "", str(text).lower())


def core_name(hf_id):
    """The model name without its owner and its trailing instruct tag, in lower case letters and digits."""
    name = normal(hf_id.split("/")[-1])
    for tail in ["instruct2507", "instructv01", "instruct", "chat", "it"]:
        if name.endswith(tail) and len(name) > len(tail) + 3:
            return name[:-len(tail)]
    return name


def named_model(prompt, models):
    """The supported model the prompt names, as a full id or by its name without the owner, or None."""
    lower = normal(prompt)
    best = None
    for hf_id, facts in models.items():
        if not facts["supported"]:
            continue
        full = normal(hf_id.split("/")[-1])
        core = core_name(hf_id)
        if len(core) < 6:
            continue
        hit = 0
        if normal(hf_id) in lower or full in lower:
            hit = 3
        elif core in lower or (core.startswith("meta") and core[4:] in lower):
            hit = 2
        if hit == 0:
            continue
        instruct = 1 if full != core else 0
        key = (hit, len(core), instruct)
        if best is None or key > best[0]:
            best = (key, hf_id)
    if best is None:
        return None
    return best[1]


def named_unsupported(prompt, models):
    """The id of a library model the prompt names that wsp.slim cannot slim, or None."""
    lower = normal(prompt)
    for hf_id in sorted(models):
        core = core_name(hf_id)
        if not models[hf_id]["supported"] and len(core) >= 6 and core in lower:
            return hf_id
    return None


def skill_rank(facts, skill):
    """0 when the model is tagged with the skill, 1 when it is a general model, 2 otherwise."""
    for tag in SKILL_TAGS.get(skill, [skill]):
        if tag in facts["skills"]:
            return 0
    if "general" in facts["skills"] or "knowledge" in facts["skills"]:
        return 1
    return 2


def why_text(facts, skill, target_b):
    pieces = []
    rank = skill_rank(facts, skill)
    if rank == 0:
        pieces.append("tagged for " + skill)
    elif rank == 1:
        pieces.append("a general model")
    if facts["kind"] == "moe":
        pieces.append("MoE with " + str(facts["experts"]) + " experts per layer")
    else:
        pieces.append("dense")
    if target_b is not None:
        pieces.append("%.1fx the target size" % (facts["params_b"] / target_b))
    return ", ".join(pieces)


def candidates_for(models, skill, target_b, want_moe, exclude, moe_hint_b=None, limit=5):
    """Supported models larger than the target, best first: skill match, then MoE when asked, then size."""
    rows = []
    for hf_id, facts in models.items():
        if not facts["supported"] or hf_id == exclude:
            continue
        if target_b is not None and facts["params_b"] <= MIN_SOURCE_RATIO * target_b:
            continue
        wrong_kind = 0
        if want_moe and facts["kind"] != "moe":
            wrong_kind = 1
        distance = 0.0
        if moe_hint_b is not None:
            distance = abs(facts["params_b"] - moe_hint_b)
        base_model = 1
        if core_name(hf_id) != normal(hf_id.split("/")[-1]):
            base_model = 0
        rows.append(((wrong_kind, skill_rank(facts, skill), round(distance, 1), facts["params_b"], base_model, hf_id),
                     facts))
    rows.sort(key=lambda row: row[0])
    out = []
    for key, facts in rows[:limit]:
        out.append({"model": facts["hf_id"], "params_b": facts["params_b"],
                    "why": why_text(facts, skill, target_b)})
    return out


def choose_source(models, skill, target_b, want_moe, moe_hint_b=None):
    """The source id to start from, or None when no supported model is larger than the target."""
    rows = candidates_for(models, skill, target_b, want_moe, None, moe_hint_b, limit=1)
    if len(rows) == 0:
        return None
    return rows[0]["model"]
