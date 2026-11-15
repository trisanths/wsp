"""The agnostic front door of the Sapien compiler: one request in, one network out.

A request names a base model, skills to add (with source models or "auto"),
skills to remove, and limits. This module does three things.

    plan(request, lineage=...)       choose a compile method per source from
                                     how the source relates to the base, order
                                     the stages, size every stage against the
                                     limits, and pick the text and the controls.
                                     No weights are read and no GPU is used.
    make(request, out_dir, backend)  run the plan through a backend, verify with
                                     wsp.probe, and write sapien_report.json.
    score_reported(...)              score the result and the comparison arms on
                                     the benchmarks the request reports, once.

Methods and why each one is chosen are in docs/auto.md. The stage rule: stages
whose output is a plain Hugging Face directory (weave, then slim) run first and
each takes the previous output as its base. At most one seam stage (synth,
feedback, transplant, delta) runs last. Two seam kinds cannot be stacked.
"""

import copy
import json
import math
import os
import re
import time

from wsp import healdata

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ATLAS_ENTRIES = os.path.join(REPO, "atlas", "entries")
ATLAS_POOL_ENTRIES = os.path.join(REPO, "atlas", "pool_entries")
LIBRARY_MODELS = os.path.join(REPO, "library", "models")
CATALOG_FILES = [os.path.join(REPO, "library", "catalog.jsonl"), os.path.join(REPO, "catalog.jsonl")]

PLAIN_STAGES = ("slim", "weave")
SEAM_STAGES = ("pgraft", "synth", "feedback", "transplant", "delta")
STAGE_ORDER = ("weave", "slim", "pgraft", "synth", "feedback", "transplant", "delta")
SEAM_PRIORITY = ("pgraft", "synth", "feedback", "transplant", "delta")

# Skill names the request may use, mapped to a wsp.healdata pool. Other names fall back to "general".
SKILL_POOLS = {
    "math": "math", "maths": "math", "mathematics": "math",
    "code": "code", "coding": "code", "programming": "code",
    "medicine": "medicine", "medical": "medicine", "health": "medicine",
    "law": "law", "legal": "law",
    "knowledge": "knowledge", "facts": "knowledge", "science": "knowledge",
    "reasoning": "reasoning", "reason": "reasoning", "logic": "reasoning",
    "multilingual": "multilingual", "translation": "multilingual",
    "general": "general", "chat": "general", "prose": "prose",
}
# The datasets the atlas scores each skill on. An atlas score or layer map is selection on that benchmark.
ATLAS_SKILL_DATA = {
    "math": ["openai/gsm8k", "huggingfaceh4/math-500"],
    "knowledge": ["cais/mmlu"],
    "medicine": ["openlifescienceai/medmcqa"],
    "law": ["cais/mmlu", "casehold/casehold"],
    "code": ["google-research-datasets/mbpp"],
    "science": ["allenai/ai2_arc"],
    "reasoning": ["lukaemon/bbh"],
    "multilingual": ["cohereforai/global-mmlu"],
    "multilingual_math": ["juletxara/mgsm"],
}
RUN_MEMORY_GIB = 384  # memory of the run_request function in modal_sapien.py
PRACTICE_SKILLS = ("math", "knowledge", "reasoning", "medicine", "code")
REMOVABLE_POOLS = ("math", "code", "medicine", "law", "knowledge", "reasoning", "multilingual")

# What has been run, by method. Printed as a warning in every plan that uses the method.
MODULE_STATE = {
    "weave": "wsp/weave.py has passed CPU tests only; no GPU run on real weights has finished",
    "synth": "wsp/synth.py has passed CPU tests on tiny models only; no GPU run on real weights has finished",
    "feedback": "wsp/feedback.py has passed CPU tests on tiny models only; no GPU run has happened",
    "transplant": "wsp/transplant.py has passed CPU tests on tiny models only; no GPU run has happened",
    "delta": "wsp/delta.py has passed CPU tests on tiny models only; no GPU run has happened",
    "slim": "wsp/slim.py has one studio run (Qwen2.5-1.5B dense) with no trusted quality number",
    "pgraft": ("wsp/pgraft.py has one measured cross-family write-in: Llama-3.2-1B-Instruct host with a frozen "
               "Qwen2.5-7B-Instruct slice (layers 4..20), MMLU held-out 2000 59.6 +-0.8 over 3 seeds against base 44.6, "
               "identity 50.3, scrambled donor 50.3, matched LoRA 49.7, rank-128 LoRA 50.0, full fine-tune 51.1 "
               "(ARC-C test 85.3 against base 55.3), 2026-11-15, results/wi_xfam/table.md. No other host and donor pair "
               "has run. An earlier 1-seed run on a Llama-3.2-3B host only tied LoRA, and the plain 3B base beat the 1B "
               "host with the graft"),
}

SETTINGS = {
    "probe_texts": 96,          # texts per probe split (a, b, c) and per skill
    "probe_length": 384,
    "probe_top_k": 64,
    "probe_skip": 2000,         # probe splits read pool rows from here on; heal reads from 0
    "heal_texts": 400,          # texts per skill pool for heal and alignment
    "general_texts": 200,
    "practice_items": 60,
    "n_controls": 4,
    "drift_tolerance": 0.05,
    "min_transfer": 0.3,
    "weave_buckets": 8,
    "weave_population": 12,
    "weave_generations": 8,
    "block_layers": 4,          # default donor block length when no limit says otherwise
    "lora_rank": 16,
    "stage1_steps": 200,
    "stage2_steps": 200,
    "slim_steps": 100,
    "transplant_k": 4,
    "transplant_rank": 256,
    "delta_rank": 64,
    "batch_size": 4,
    "seq_len": 256,
    "synth_controls": ["random", "scrambled"],
    "gpu_flops": 2.0e14,        # effective bf16 FLOPs per second of one H200 in a heal step (a guess)
    "seam_threshold": 3,
    "slim_keep": 0.5,           # share of experts a MoE slim keeps when max_total_params_b sets no size
    "pgraft_controls": ["identity", "scrambled"],   # controls the compiler trains itself after the write-in (pgraft)
    "pgraft_read_layers": None,   # "dense": Enc reads a mix of host layers 1..host_layer (not measured in this recipe)
    "tops_cut": 0.5,            # share of the base layers kept as the shared trunk of a skill top (size estimate)
}


class RequestError(ValueError):
    """The request file is not valid."""


class PlanError(RuntimeError):
    """The plan cannot be run. The message says why and which limit would change it."""


# ---------------------------------------------------------------- request


def _number(value, label, minimum_exclusive=0):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RequestError("%s must be a number, got %r" % (label, value))
    if value <= minimum_exclusive:
        raise RequestError("%s must be above %s, got %r" % (label, minimum_exclusive, value))
    return float(value)


def _model_id(value, label):
    if not isinstance(value, str) or value.strip() == "":
        raise RequestError("%s must be a model id string, got %r" % (label, value))
    return value.strip()


def _skill_name(value, label):
    if not isinstance(value, str) or value.strip() == "":
        raise RequestError("%s must be a skill name string, got %r" % (label, value))
    return value.strip().lower()


def validate_request(raw):
    """Return the request as a clean dict. Raise RequestError with a message that names the field."""
    if not isinstance(raw, dict):
        raise RequestError("the request must be a JSON object")
    known = {"name", "base", "add", "remove", "keep", "limits", "reported", "seed", "notes"}
    extra = sorted(set(raw) - known)
    if extra:
        raise RequestError("unknown field(s): %s; fields are %s" % (", ".join(extra), ", ".join(sorted(known))))
    name = raw.get("name")
    if not isinstance(name, str) or not re.match(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$", name or ""):
        raise RequestError("name must be a short string of letters, digits, '_', '-', '.' (it names the output files)")
    base = _model_id(raw.get("base"), "base")

    add = []
    raw_add = raw.get("add", [])
    if not isinstance(raw_add, list):
        raise RequestError("add must be a list of {skill, sources, components}")
    for index, entry in enumerate(raw_add):
        label = "add[%d]" % index
        if not isinstance(entry, dict):
            raise RequestError("%s must be an object with a skill" % label)
        bad = sorted(set(entry) - {"skill", "sources", "components", "method"})
        if bad:
            raise RequestError("%s: unknown field(s): %s" % (label, ", ".join(bad)))
        skill = _skill_name(entry.get("skill"), label + ".skill")
        sources = entry.get("sources", "auto")
        if sources != "auto":
            if not isinstance(sources, list) or len(sources) == 0:
                raise RequestError('%s.sources must be "auto" or a non-empty list of model ids' % label)
            sources = [_model_id(s, label + ".sources item") for s in sources]
            if base in sources:
                raise RequestError("%s.sources lists the base model %s; a source must be another model" % (label, base))
        components = entry.get("components")
        if components is not None:
            if not isinstance(components, list) or len(components) == 0:
                raise RequestError("%s.components must be null or a non-empty list of library or atlas ids" % label)
            components = [_model_id(c, label + ".components item") for c in components]
        method = entry.get("method")
        if method is not None and method not in STAGE_ORDER:
            raise RequestError("%s.method must be one of %s" % (label, ", ".join(STAGE_ORDER)))
        add.append({"skill": skill, "sources": sources, "components": components, "method": method})

    remove = []
    raw_remove = raw.get("remove", [])
    if raw_remove is None:
        raw_remove = []
    if not isinstance(raw_remove, list):
        raise RequestError("remove must be a list of {skill}")
    for index, entry in enumerate(raw_remove):
        label = "remove[%d]" % index
        if not isinstance(entry, dict) or "skill" not in entry:
            raise RequestError("%s must be an object with a skill" % label)
        remove.append({"skill": _skill_name(entry["skill"], label + ".skill")})

    keep = raw.get("keep", [])
    if keep is None:
        keep = []
    if not isinstance(keep, list):
        raise RequestError("keep must be a list of skill names")
    keep = [_skill_name(k, "keep item") for k in keep]
    if len(add) == 0 and len(remove) == 0:
        raise RequestError("the request needs something to do: add or remove must not both be empty")
    for entry in remove:
        if entry["skill"] == "*" and len(keep) == 0:
            raise RequestError('remove "*" means every skill except the kept ones, so keep must name at least one skill')
    added = {e["skill"] for e in add}
    removed = {e["skill"] for e in remove}
    both = sorted(added & removed)
    if both:
        raise RequestError("skill(s) %s are both added and removed" % ", ".join(both))

    limits_raw = raw.get("limits", {})
    if limits_raw is None:
        limits_raw = {}
    if not isinstance(limits_raw, dict):
        raise RequestError("limits must be an object")
    bad = sorted(set(limits_raw) - {"max_total_params_b", "max_active_params_b", "max_heal_minutes"})
    if bad:
        raise RequestError("limits: unknown field(s): %s" % ", ".join(bad))
    limits = {"max_total_params_b": None, "max_active_params_b": None, "max_heal_minutes": None}
    for key in limits:
        if limits_raw.get(key) is not None:
            limits[key] = _number(limits_raw[key], "limits." + key)

    reported = raw.get("reported")
    if reported is None:
        reported = list(healdata.BANNED)
    else:
        if not isinstance(reported, list) or not all(isinstance(r, str) and r for r in reported):
            raise RequestError("reported must be a list of benchmark dataset ids")
        reported = list(reported)
    seed = raw.get("seed", 0)
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise RequestError("seed must be an integer")
    return {"name": name, "base": base, "add": add, "remove": remove, "keep": keep, "limits": limits,
            "reported": reported, "seed": seed, "notes": raw.get("notes")}


def load_request(path):
    """Read and validate a request file."""
    try:
        with open(path) as handle:
            raw = json.load(handle)
    except OSError as error:
        raise RequestError("%s: %s" % (path, error.strerror))
    except ValueError as error:
        raise RequestError("%s: not valid JSON (%s)" % (path, error))
    return validate_request(raw)


# ---------------------------------------------------------------- lineage adapters


class LiveLineage:
    """The real lineage probe (wsp.lineage): reads the Hub by range reads."""

    def __init__(self, cache_dir=None, opener=None):
        from wsp import lineage

        self.module = lineage
        self.kwargs = {}
        if cache_dir is not None:
            self.kwargs["cache_dir"] = cache_dir
        if opener is not None:
            self.kwargs["opener"] = opener

    def facts(self, model_id):
        return self.module.facts(model_id, **self.kwargs)

    def relation(self, base_id, source_id):
        return self.module.relation(base_id, source_id, **self.kwargs)

    def methods(self, rel, want="add"):
        return self.module.methods(rel, want)


class StaticLineage:
    """Lineage from a table, with no network. Used for offline plans and tests.

    models: {id: {"params": int, "config": {layers, hidden, heads, kv_heads, head_dim, mlp_size, vocab,
              tied_embeddings, experts, top_k, expert_size, model_type}, "license": str}}
    relations: {(base, source): kind or {"kind": ..., "warnings": [...], "subtract_reference": id,
                "flags": {...}}}. A pair that is not listed is "foreign".
    """

    def __init__(self, models, relations):
        from wsp import lineage

        self.module = lineage
        self.models = models
        self.relations = relations

    def facts(self, model_id):
        if model_id not in self.models:
            raise KeyError("no facts for %s in the static table" % model_id)
        row = self.models[model_id]
        config = dict(row.get("config", {}))
        return {"model_id": model_id, "config": config, "params": row["params"], "license": row.get("license"),
                "tokenizer": {"core_hash": row.get("tokenizer", "t")}, "base_model": row.get("base_model", [])}

    def relation(self, base_id, source_id):
        base = self.facts(base_id)
        source = self.facts(source_id)
        entry = self.relations.get((base_id, source_id), "foreign")
        if isinstance(entry, str):
            entry = {"kind": entry}
        kind = entry["kind"]
        same_tokenizer = base["tokenizer"]["core_hash"] == source["tokenizer"]["core_hash"] and kind != "foreign"
        flags = {"positional_base_differs": kind == "diverged_same_shape" and bool(entry.get("rope_differs")),
                 "vocab_differs": False, "tokenizer_same": same_tokenizer, "template_same": True,
                 "teacher_usable_for_logit_kd": same_tokenizer, "embeddings_mixable": kind in ("fine_tune", "same"),
                 "same_family": kind != "foreign"}
        flags.update(entry.get("flags", {}))
        return {"base": base_id, "source": source_id, "kind": kind,
                "evidence": {"base": base["config"], "source": source["config"], "static": True,
                             "relative_delta_norm": entry.get("relative_delta_norm"),
                             "mean_cosine": entry.get("mean_cosine")},
                "flags": flags, "warnings": list(entry.get("warnings", [])),
                "subtract_reference": entry.get("subtract_reference"),
                "base_is_moe": bool(base["config"].get("experts")),
                "source_is_moe": bool(source["config"].get("experts"))}

    def methods(self, rel, want="add"):
        return self.module.methods(rel, want)


# ---------------------------------------------------------------- sizes


def embed_params(cfg):
    if cfg.get("vocab") and cfg.get("hidden"):
        return cfg["vocab"] * cfg["hidden"]
    return 0


def attn_params(cfg):
    hidden = cfg["hidden"]
    heads = cfg["heads"]
    kv = cfg.get("kv_heads") or heads
    head_dim = cfg.get("head_dim") or hidden // heads
    return hidden * heads * head_dim + 2 * hidden * kv * head_dim + heads * head_dim * hidden


def expert_params(cfg):
    """Parameters of one routed expert (gate, up, down projections)."""
    size = cfg.get("expert_size") or cfg.get("mlp_size")
    return 3 * cfg["hidden"] * size


def layer_params(cfg, run=False):
    """Parameters of one decoder layer, estimated from the config. run=True counts only the experts a token uses."""
    hidden = cfg["hidden"]
    if cfg.get("experts"):
        used = cfg.get("top_k") or 1 if run else cfg["experts"]
        mlp = used * expert_params(cfg) + cfg["experts"] * hidden
    else:
        mlp = 3 * hidden * cfg["mlp_size"]
    return attn_params(cfg) + mlp + 2 * hidden


def model_sizes(facts):
    """Return (total, run per token) from the facts: counted tensors, then the shared rule of skills_eval.model_params."""
    cfg = facts["config"]
    total = int(facts["params"])
    run = total
    if cfg.get("experts") and cfg.get("layers"):
        run -= cfg["layers"] * (cfg["experts"] - (cfg.get("top_k") or 1)) * expert_params(cfg)
    if not cfg.get("tied_embeddings"):
        run -= embed_params(cfg)
    return total, run


def lora_params(cfg, rank, layers):
    """LoRA parameters on q, k, v, o, gate, up, down of `layers` host layers."""
    hidden = cfg["hidden"]
    heads = cfg["heads"]
    kv = cfg.get("kv_heads") or heads
    head_dim = cfg.get("head_dim") or hidden // heads
    per_rank = (hidden + heads * head_dim) + 2 * (hidden + kv * head_dim) + (heads * head_dim + hidden)
    if cfg.get("experts"):
        per_rank += 0
    else:
        per_rank += 3 * (hidden + cfg["mlp_size"])
    return rank * per_rank * max(0, layers)


def billions(value):
    return value / 1e9


# ---------------------------------------------------------------- mined data


def _read_json(path):
    with open(path) as handle:
        return json.load(handle)


def load_atlas(entries_dir=None, pool_dir=None):
    """Atlas entries and pool entries, when the folders exist."""
    from wsp.atlas import load_entries

    rows = []
    for folder in (entries_dir or ATLAS_ENTRIES, pool_dir or ATLAS_POOL_ENTRIES):
        if os.path.isdir(folder):
            rows.extend(load_entries(folder))
    return rows


def load_library_models(folder=None):
    folder = folder or LIBRARY_MODELS
    rows = []
    if os.path.isdir(folder):
        for name in sorted(os.listdir(folder)):
            if name.endswith(".json"):
                try:
                    rows.append(_read_json(os.path.join(folder, name)))
                except ValueError:
                    continue
    return rows


def atlas_leak(skill, reported):
    """The reported benchmark an atlas score or layer map for this skill was measured on, or None."""
    from wsp import healdata as hd

    if reported is None:
        return None
    listed = hd_reported(reported)
    for dataset in ATLAS_SKILL_DATA.get(skill, []):
        if hd.why_reported(dataset, listed) is not None:
            return dataset
    return None


def auto_candidates(skill, base_id, atlas_entries, library_models, reported=None):
    """Models mined for a skill, best first: [{"model", "score", "params_b", "from"}].

    With reported given, atlas scores of a skill measured on a reported benchmark are not used to rank or pick.
    """
    seen = {}
    leaked = atlas_leak(skill, reported) is not None
    for entry in atlas_entries:
        model = entry.get("model")
        if leaked or model == base_id or skill not in entry.get("skills", {}):
            continue
        seen[model] = {"model": model, "score": float(entry["skills"][skill]), "params_b": entry.get("params_b"),
                       "from": "atlas"}
    for row in library_models:
        model = row.get("hf_id")
        if not model or model == base_id or model in seen:
            continue
        if skill in (row.get("skills") or []):
            seen[model] = {"model": model, "score": None, "params_b": row.get("params_b"), "from": "library"}
    rows = list(seen.values())
    rows.sort(key=lambda r: (r["score"] is None, -(r["score"] or 0.0), r["params_b"] or 1e9))
    return rows


def find_component_rows(ids, catalog_files=None, atlas_entries=None):
    """Resolve library and atlas component ids to catalog rows. Returns (rows, missing ids)."""
    wanted = set(ids)
    found = {}
    for path in catalog_files or CATALOG_FILES:
        if not os.path.exists(path):
            continue
        with open(path) as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except ValueError:
                    continue
                if row.get("id") in wanted and row["id"] not in found:
                    found[row["id"]] = row
    if wanted - set(found) and atlas_entries:
        from wsp.atlas import catalog_rows

        for entry in atlas_entries:
            for skill in entry.get("layer_maps", {}):
                for row in catalog_rows(entry, skill):
                    if row["id"] in wanted and row["id"] not in found:
                        found[row["id"]] = row
    rows = [found[i] for i in ids if i in found]
    missing = [i for i in ids if i not in found]
    return rows, missing


def row_params(row, donor_cfg):
    """Parameters of one catalog component, estimated from the donor config."""
    kind = row.get("type") or row.get("kind")
    if kind == "block":
        first, last = row["layers"]
        return (last - first + 1) * layer_params(donor_cfg, run=False)
    units = row.get("num_units") or 1
    hidden = donor_cfg["hidden"]
    if kind == "mlp_cluster":
        return units * 3 * hidden
    if kind == "attn_heads":
        head_dim = donor_cfg.get("head_dim") or hidden // donor_cfg["heads"]
        return units * 4 * hidden * head_dim
    if kind == "expert":
        return units * expert_params(donor_cfg)
    return 0


# ---------------------------------------------------------------- planning


def hd_reported(reported):
    """What to pass to wsp.healdata as reported=. The default set is passed as None so the config-level bans apply."""
    if reported is None or list(reported) == list(healdata.BANNED):
        return None
    return list(reported)


def pool_for(skill, reported):
    """Return (pool name, note). Falls back to "general" with a stated reason."""
    name = SKILL_POOLS.get(skill)
    if name is None:
        return "general", "skill %r has no pool in wsp.healdata; the general pool is a weak proxy" % skill
    try:
        usable, excluded = healdata._usable(name, reported)
    except ValueError as error:
        return "general", str(error)
    if len(usable) == 0:
        return "general", "every source of pool %r is in the reported set; the general pool is used instead" % name
    return name, None


def pool_sources(pool, reported):
    usable, excluded = healdata._usable(pool, reported)
    return ([{"id": s[0], "config": s[1], "split": s[2]} for s in usable],
            [{"id": s[0], "config": s[1], "reason": r} for s, r in excluded])


def practice_info(skill, reported):
    """Practice set facts for a skill under a reported set. {"available": bool, "sources": [...], "reason": str}"""
    name = SKILL_POOLS.get(skill, skill)
    if name not in PRACTICE_SKILLS:
        return {"available": False, "skill": name, "sources": [],
                "reason": "healdata has no practice set for %r (sets exist for %s)" % (skill, ", ".join(PRACTICE_SKILLS))}
    sources = []
    excluded = []
    for dataset_id, config, split, kind in healdata.PRACTICE[name]:
        reason = healdata.why_reported(dataset_id, reported, config)
        if reason is None:
            sources.append({"id": dataset_id, "config": config, "split": split})
        else:
            excluded.append({"id": dataset_id, "reason": reason})
    if len(sources) == 0:
        return {"available": False, "skill": name, "sources": [], "excluded": excluded,
                "reason": "every practice source of %r is in the reported set" % name}
    return {"available": True, "skill": name, "sources": sources, "excluded": excluded, "reason": None}


def split_minutes(minutes, stages):
    """Divide heal minutes between stage methods. A seam stage gets twice the share of a plain stage."""
    if minutes is None:
        return {s: None for s in stages}
    weights = {s: (2.0 if s in SEAM_STAGES else 1.0) for s in stages}
    whole = sum(weights.values())
    return {s: minutes * weights[s] / whole for s in stages}


def seconds_per_step(active_params, batch, seq_len, flops):
    """Seconds of one heal step: forward and backward (6 FLOPs per parameter per token) plus fixed cost."""
    return 0.05 + 6.0 * active_params * batch * seq_len / flops


def steps_for(minutes, active_params, settings, fraction=0.8):
    if minutes is None:
        return None
    per = seconds_per_step(active_params, settings["batch_size"], settings["seq_len"], settings["gpu_flops"])
    return max(2, int(minutes * 60.0 * fraction / per))


def default_block(donor_cfg, skill, source, atlas_entries, n_layers, reported=None):
    """Pick donor layers [first, last] of length n_layers. Returns (layers, note).

    An atlas layer map for a skill mined on a reported benchmark is not used; the depth rule applies instead.
    """
    from wsp.atlas import best_block_for

    total = donor_cfg["layers"]
    n_layers = max(1, min(n_layers, total))
    leaked = atlas_leak(skill, reported)
    if leaked is not None:
        atlas_entries = None
    if atlas_entries:
        try:
            best = best_block_for(skill, atlas_entries, {"models": [source], "max_length": n_layers})
        except (KeyError, ValueError, TypeError):
            best = None
        if best is not None:
            first, last = best["block"]
            if last - first + 1 < n_layers:
                first = max(0, min(first, total - n_layers))
                last = first + n_layers - 1
            return [first, last], "atlas block for %s (mass %.3f)" % (skill, best["mass"])
    first = int(round(0.55 * total))
    first = max(0, min(first, total - n_layers))
    why = "atlas maps for %s are mined on reported %s" % (skill, leaked) if leaked is not None else "no atlas block for %s" % skill
    return [first, first + n_layers - 1], "%s in %s; layers centred near 70 percent depth" % (why, source)


def seam_addition(method, base_cfg, base_total, donor_cfg, n_layers, rank, host_layers_after):
    """Parameters a seam of n_layers donor layers adds: (total, run). Glue is T_in, T_out, and a gate."""
    parts = n_layers * layer_params(donor_cfg, run=False)
    parts_run = n_layers * layer_params(donor_cfg, run=True)
    glue = 2 * base_cfg["hidden"] * donor_cfg["hidden"] + 2 * base_cfg["hidden"]
    lora = lora_params(base_cfg, rank, host_layers_after)
    return parts + glue + lora, parts_run + glue + lora


def transplant_addition(base_cfg, source_cfg, k, rank, top_k=2):
    """Parameters of K routed experts per layer with stitches of the given rank: (total, run)."""
    layers = base_cfg["layers"]
    per_expert = expert_params(source_cfg)
    d_base = base_cfg["hidden"]
    d_source = source_cfg["hidden"]
    if rank is None:
        stitch = 2 * d_base * d_source
    else:
        stitch = rank * (d_base + d_source) * 2
    router = k * d_source
    gate = 1 + d_source
    total = layers * (k * per_expert + stitch + router + gate)
    run = layers * (min(top_k, k) * per_expert + stitch + router + gate)
    return total, run


def tops_addition(base_cfg, base_run, donor_cfg, cut_fraction, rank):
    """Estimate of one skill top (wsp/tops.py): (stored, run). The measured size comes from SkillTops.count_params."""
    cut_host = max(1, min(base_cfg["layers"] - 1, int(round(cut_fraction * base_cfg["layers"]))))
    cut_donor = max(0, min(donor_cfg["layers"] - 1, int(round(cut_host / base_cfg["layers"] * donor_cfg["layers"]))))
    layers = (donor_cfg["layers"] - cut_donor) * layer_params(donor_cfg)
    layers_run = (donor_cfg["layers"] - cut_donor) * layer_params(donor_cfg, run=True)
    stitch = 2 * base_cfg["hidden"] * donor_cfg["hidden"] + base_cfg["hidden"] + donor_cfg["hidden"]
    head = donor_cfg["vocab"] * donor_cfg["hidden"] if donor_cfg.get("vocab") else 0
    lora = lora_params(donor_cfg, rank, donor_cfg["layers"] - cut_donor) if rank else 0
    stored = layers + stitch + head + donor_cfg["hidden"] + lora
    trunk = cut_host * layer_params(base_cfg, run=True)
    path = trunk + layers_run + stitch + head + lora
    base_layers = base_cfg["layers"] * layer_params(base_cfg, run=True)
    return int(stored), int(max(0, path - base_layers))


def delta_addition(base_cfg, rank):
    """Parameters of one rank-r delta over the attention and MLP matrices: (stored, run when active)."""
    hidden = base_cfg["hidden"]
    heads = base_cfg["heads"]
    kv = base_cfg.get("kv_heads") or heads
    head_dim = base_cfg.get("head_dim") or hidden // heads
    per_layer = rank * ((hidden + heads * head_dim) + 2 * (hidden + kv * head_dim) + (heads * head_dim + hidden)
                        + 3 * (hidden + base_cfg["mlp_size"]))
    stored = per_layer * base_cfg["layers"]
    return stored, stored


def slim_target(base_cfg, base_total, base_run, limits, seam_reserve, keep_default=0.5):
    """Size a skill removal. Returns {"kind", "total_after", "run_after", "keep"/"drop", "note"} or {"error": text}."""
    out = {"warnings": []}
    limit = limits["max_total_params_b"]
    target_total = None
    if limit is not None:
        target_total = limit * 1e9 - seam_reserve
    if base_cfg.get("experts"):
        layers = base_cfg["layers"]
        per = expert_params(base_cfg)
        fixed = base_total - layers * base_cfg["experts"] * per
        top_k = base_cfg.get("top_k") or 1
        if target_total is None:
            keep = max(top_k, int(base_cfg["experts"] * keep_default))
            out["warnings"].append("no max_total_params_b: slim keeps %d of %d experts per layer by default"
                                   % (keep, base_cfg["experts"]))
        else:
            keep = int(math.floor((target_total - fixed) / (layers * per)))
            need = fixed + layers * top_k * per
            if keep < top_k:
                out["error"] = ("max_total_params_b %.3f is below what slim can reach: the non-expert weights are "
                                "%.3fB and each token needs %d experts per layer, so at least %.3fB is needed"
                                % (limit, billions(fixed), top_k, billions(need + seam_reserve)))
                return out
            keep = min(keep, base_cfg["experts"])
        out.update({"kind": "moe", "keep": keep, "experts": base_cfg["experts"],
                    "total_after": fixed + layers * keep * per,
                    "run_after": base_run})
        out["note"] = "keep %d of %d experts per layer; the tokens still use %d experts per layer" % (keep, base_cfg["experts"], top_k)
        return out
    # dense: layers dropped (cap 25 percent) and width thinned (rough; slim.plan_size decides at run time)
    layers = base_cfg["layers"]
    per_layer = layer_params(base_cfg)
    fixed = base_total - layers * per_layer
    if target_total is None:
        target_total = base_total * 0.8
        out["warnings"].append("no max_total_params_b: slim aims at 80 percent of the base by default")
    cap_drop = int(math.floor(0.25 * layers))
    if target_total < fixed + 0.5 * (layers - cap_drop) * per_layer:
        reachable = fixed + 0.5 * (layers - cap_drop) * per_layer
        out["error"] = ("max_total_params_b %.3f is below what dense slim reaches: embeddings and head are %.3fB and "
                        "slim drops at most %d layers and thins the rest; about %.3fB is the floor"
                        % (limit if limit is not None else billions(target_total), billions(fixed), cap_drop,
                           billions(reachable + seam_reserve)))
        return out
    total_after = min(base_total, target_total)
    out.update({"kind": "dense", "target_params": int(total_after), "total_after": total_after,
                "run_after": base_run - (base_total - total_after)})
    out["note"] = "dense slim to about %.3fB: drop up to %d layers by skill cost, thin MLP neurons and heads" % (
        billions(total_after), cap_drop)
    return out


def check_limits(total, run, limits):
    """Return a list of problems for a size pair. Empty when the sizes fit."""
    problems = []
    if limits["max_total_params_b"] is not None and total > limits["max_total_params_b"] * 1e9 * (1 + 1e-9):
        problems.append("total %.3fB is above max_total_params_b %.3f" % (billions(total), limits["max_total_params_b"]))
    if limits["max_active_params_b"] is not None and run > limits["max_active_params_b"] * 1e9 * (1 + 1e-9):
        problems.append("params per token %.3fB is above max_active_params_b %.3f"
                        % (billions(run), limits["max_active_params_b"]))
    return problems


def _item_row(skill, source, rows=None, method=None):
    return {"skill": skill, "source": source, "rows": rows, "method_override": method}


def plan(request, lineage=None, settings=None, atlas_entries=None, library_models=None, catalog_files=None):
    """Plan a request. Never reads weights and never touches a GPU.

    lineage: an object with facts(id), relation(base, source), methods(rel, want)
    (LiveLineage by default, StaticLineage for offline use). atlas_entries and
    library_models default to the files in the repo. Returns a plan dict; when
    no method fits, plan["feasible"] is False and plan["refusals"] says why.
    """
    req = validate_request(request)
    s = dict(SETTINGS)
    if settings:
        s.update(settings)
    lin = lineage if lineage is not None else LiveLineage()
    if atlas_entries is None:
        atlas_entries = load_atlas()
    if library_models is None:
        library_models = load_library_models()
    reported = hd_reported(req["reported"])
    limits = req["limits"]
    warnings = []
    refusals = []

    base_facts = lin.facts(req["base"])
    base_cfg = base_facts["config"]
    base_total, base_run = model_sizes(base_facts)
    if base_total == 0:
        refusals.append("base %s has no safetensors weights, so its size is unknown (a repo with only .bin files); "
                        "use a safetensors checkpoint" % req["base"])
    base_info = {"id": req["base"], "params": base_total, "params_run": base_run,
                 "params_b": round(billions(base_total), 4), "active_params_b": round(billions(base_run), 4),
                 "model_type": base_cfg.get("model_type"), "moe": bool(base_cfg.get("experts"))}
    for problem in check_limits(base_total, base_run, {"max_total_params_b": None, "max_active_params_b":
                                                       limits["max_active_params_b"]}):
        if len(req["remove"]) == 0:
            refusals.append("the base alone is over a limit: " + problem + " and no skill removal is requested")

    # 1. sources
    items = []
    for add in req["add"]:
        rows_by_source = {}
        if add["components"] is not None:
            rows, missing = find_component_rows(add["components"], catalog_files, atlas_entries)
            if missing:
                refusals.append("add %s: component id(s) not found in library/catalog.jsonl, catalog.jsonl, or the atlas: %s"
                                % (add["skill"], ", ".join(missing)))
            for row in rows:
                rows_by_source.setdefault(row["model"], []).append(row)
        if add["sources"] == "auto":
            if rows_by_source:
                sources = list(rows_by_source)
            else:
                sources = None
        else:
            sources = list(add["sources"])
            for model in rows_by_source:
                if model not in sources:
                    refusals.append("add %s: component rows come from %s, which is not in sources" % (add["skill"], model))
        if sources is None:
            candidates = auto_candidates(add["skill"], req["base"], atlas_entries, library_models, req["reported"])
            items.append({"skill": add["skill"], "auto": True, "candidates": candidates, "method_override": add["method"]})
            continue
        for source in sources:
            items.append(_item_row(add["skill"], source, rows_by_source.get(source), add["method"]))

    # resolve "auto" items: first candidate with a valid method that fits alone
    resolved = []
    for item in items:
        if not item.get("auto"):
            resolved.append(item)
            continue
        leaked = atlas_leak(item["skill"], req["reported"])
        if len(item["candidates"]) == 0 and leaked is not None:
            refusals.append("add %s with sources \"auto\": atlas scores for skill %s are measured on reported benchmark %s; "
                            "list sources by hand" % (item["skill"], item["skill"], leaked))
            continue
        if len(item["candidates"]) == 0:
            refusals.append(
                "add %s with sources \"auto\": no mined data names a model for this skill. The atlas has no entry with "
                "the skill (atlas/entries and atlas/pool_entries hold %d entries) and library/models has no profile for "
                "it. Run modal_atlas.py or add library profiles, or list sources by hand." % (item["skill"], len(atlas_entries)))
            continue
        tried = []
        picked = None
        for cand in item["candidates"][:8]:
            try:
                rel = lin.relation(req["base"], cand["model"])
                options = _method_options(lin, rel, base_cfg, base_total, base_run, lin.facts(cand["model"]), item["skill"],
                                          limits, s, atlas_entries, None, None)
            except Exception as error:
                tried.append("%s (%s)" % (cand["model"], str(error)[:80]))
                continue
            fitting = [o for o in options if o["fits"]]
            if fitting:
                picked = cand
                break
            tried.append("%s (%s)" % (cand["model"], options[0]["why_not"] if options else "no valid method"))
        if picked is None:
            refusals.append("add %s with sources \"auto\": no mined model fits a relation with a valid method inside the "
                            "limits; tried %s" % (item["skill"], "; ".join(tried)))
            continue
        resolved.append(_item_row(item["skill"], picked["model"], None, item["method_override"]))
        warnings.append("auto source for %s: %s (%s score %s)" % (item["skill"], picked["model"], picked["from"], picked["score"]))
    items = resolved

    # 2. relations and candidate methods per source
    sources_out = []
    for item in items:
        source = item["source"]
        try:
            src_facts = lin.facts(source)
            rel = lin.relation(req["base"], source)
        except Exception as error:
            refusals.append("add %s from %s: the lineage probe failed (%s)" % (item["skill"], source, error))
            continue
        if src_facts["params"] == 0:
            refusals.append("add %s from %s: the source has no safetensors weights, so its size is unknown; use a "
                            "safetensors checkpoint" % (item["skill"], source))
            continue
        options = _method_options(lin, rel, base_cfg, base_total, base_run, src_facts, item["skill"], limits, s,
                                  atlas_entries, item["rows"], item["method_override"])
        sources_out.append({"skill": item["skill"], "source": source, "rows": item["rows"], "rel": rel,
                            "facts": src_facts, "options": options})

    # 3. choose methods
    chosen = _choose_methods(sources_out, refusals)

    # 4. removals
    slim_info = None
    seam_reserve = 0.0
    if len(req["remove"]) > 0:
        seam_items = [x for x in sources_out if chosen.get(id(x)) in SEAM_STAGES]
        seam_reserve = sum(chosen_add(x, chosen[id(x)])[0] for x in seam_items)
        slim_info = slim_target(base_cfg, base_total, base_run, limits, seam_reserve, keep_default=s["slim_keep"])
        if "error" in slim_info:
            refusals.append("remove: " + slim_info["error"])
        else:
            warnings.extend(slim_info["warnings"])
        if len(req["keep"]) == 0:
            warnings.append("remove without keep: the kept skill texts for ranking units come from the general pool")

    # 5. stages
    stages = []
    totals = {"total": base_total, "run": base_run}
    after_plain_total = base_total
    weave_items = [x for x in sources_out if chosen.get(id(x)) == "weave"]
    seam_items = [x for x in sources_out if chosen.get(id(x)) in SEAM_STAGES]
    seam_methods = sorted({chosen[id(x)] for x in seam_items})
    if seam_methods == ["transplant"] and len({x["source"] for x in seam_items}) > 1:
        refusals.append("transplant takes experts from one source model; the seam skills name %s. Use one MoE source or "
                        "split the request in two compiles." % ", ".join(sorted({x["source"] for x in seam_items})))
    if seam_methods == ["pgraft"]:
        if len(seam_items) > 1:
            refusals.append("pgraft writes one donor slice into the host; the seam skills name %d sources (%s). Split the "
                            "request in one compile per source." % (len(seam_items), ", ".join(x["source"] for x in seam_items)))
        if weave_items or (slim_info is not None and "error" not in slim_info):
            refusals.append("pgraft cannot run after a weave or slim stage yet: its recipe was measured on an unmodified "
                            "host. Split the request in two compiles.")
    if len(seam_methods) > 1:
        refusals.append("the sources need different seam methods (%s), and two seam stages cannot be stacked: each one "
                        "replaces the other's branches. Use sources that share a method, or split the request in two "
                        "compiles." % ", ".join(seam_methods))
    plan_stages = []
    if weave_items:
        plan_stages.append("weave")
    if slim_info is not None and "error" not in slim_info:
        plan_stages.append("slim")
    if seam_items and len(seam_methods) == 1:
        plan_stages.append(seam_methods[0])
    minutes = split_minutes(limits["max_heal_minutes"], plan_stages)

    sizes_after = []
    running_total, running_run = base_total, base_run
    for method in plan_stages:
        stage = {"order": len(stages) + 1, "method": method, "kind": "seam" if method in SEAM_STAGES else "plain",
                 "module": MODULE_STATE[method].split(" has ")[0], "heal_minutes": minutes[method]}
        if method == "weave":
            stage["members"] = [{"skill": x["skill"], "source": x["source"], "relation": x["rel"]["kind"],
                                 "reference": x["rel"].get("subtract_reference") or "base",
                                 "mlp_only": bool(x["rel"]["flags"].get("positional_base_differs"))} for x in weave_items]
            stage["settings"] = _weave_settings(s, minutes[method], base_total, len(weave_items))
            stage["cpu_ram_gib_estimate"] = round((2 * len(weave_items) + 1) * base_total * 2 / 2**30, 1)
            if stage["cpu_ram_gib_estimate"] > RUN_MEMORY_GIB:
                warnings.append("weave host memory estimate %.0f GiB is above the %d GiB of the Modal runner; the real "
                                "need is lower than this upper bound but has not been measured"
                                % (stage["cpu_ram_gib_estimate"], RUN_MEMORY_GIB))
            stage["downloads_gib"] = round(sum(x["facts"]["params"] * 2 for x in weave_items) / 2**30, 1)
        elif method == "slim":
            stage["target"] = {k: v for k, v in slim_info.items() if k != "warnings"}
            running_total, running_run = slim_info["total_after"], slim_info["run_after"]
            stage["settings"] = {"steps": steps_for(minutes[method], running_run, s) or s["slim_steps"],
                                 "teacher": "the model before slim", "rule": "saliency" if slim_info["kind"] == "moe" else "skill_cost"}
        elif method == "pgraft":
            from wsp import pgraft_stage

            x = seam_items[0]
            stage.update(pgraft_stage.stage_info(x, base_cfg, s, tuple(s["pgraft_controls"])))
            stage["module"] = "wsp/pgraft.py (wsp/xfam.py)"
            stage["components"] = [{"skill": x["skill"], "source": x["source"], "method": "pgraft",
                                    "adds_params": stage["params_stored_added"]}]
            stage["added_params"] = stage["params_stored_added"]
            running_total += stage["params_stored_added"]
            running_run += stage["params_per_token_added"]
        else:
            add_total, add_run = 0, 0
            comps = []
            if method == "transplant":
                done = set()
                for x in seam_items:
                    if x["source"] in done:
                        continue
                    done.add(x["source"])
                    a_t, a_r, info = chosen_add(x, method, full=True)
                    add_total += a_t
                    add_run += a_r
                    comps.append(info)
                skills = [x["skill"] for x in seam_items]
                stage["source"] = seam_items[0]["source"]
                stage["skills"] = skills
            else:
                for x in seam_items:
                    a_t, a_r, info = chosen_add(x, method, full=True)
                    add_total += a_t
                    add_run += a_r
                    comps.append(info)
            stage["components"] = comps
            stage["added_params"] = add_total
            running_total += add_total
            running_run += add_run
            stage["settings"] = _seam_settings(method, s, minutes[method], running_run)
        stage["params_total_after"] = int(running_total)
        stage["params_run_after"] = int(running_run)
        stage["params_total_after_b"] = round(billions(running_total), 4)
        stage["params_run_after_b"] = round(billions(running_run), 4)
        stage["limit_problems"] = check_limits(running_total, running_run, limits)
        stages.append(stage)
        sizes_after.append({"after": method, "total_b": stage["params_total_after_b"], "run_b": stage["params_run_after_b"]})
    for problem in check_limits(running_total, running_run, limits):
        refusals.append("the final network breaks a limit: " + problem)

    # 6. text, probes, practice, controls, arms
    skills_added = [x["skill"] for x in sources_out]
    heal_data = {}
    probe_texts = {}
    practice = {}
    for skill in list(dict.fromkeys(skills_added + req["keep"] + [r["skill"] for r in req["remove"] if r["skill"] != "*"])):
        pool, note = pool_for(skill, reported)
        use, excluded = pool_sources(pool, reported)
        heal_data[skill] = {"pool": pool, "fallback": note, "count": s["heal_texts"], "seed": req["seed"], "skip": 0,
                            "sources": use, "excluded": excluded}
        probe_texts[skill] = {"pool": pool, "count": s["probe_texts"], "seed": req["seed"], "skip": s["probe_skip"],
                              "splits": ["a", "b", "c"], "max_length": s["probe_length"]}
        if note is not None:
            warnings.append("skill %s: %s" % (skill, note))
        practice[skill] = practice_info(skill, reported)
        if not practice[skill]["available"]:
            warnings.append("skill %s has no practice set (%s); verification uses probe transfer only" % (
                skill, practice[skill]["reason"]))
    general_use, general_excluded = pool_sources("general", reported)
    heal_data["_general"] = {"pool": "general", "fallback": None, "count": s["general_texts"], "seed": req["seed"], "skip": 0,
                             "sources": general_use, "excluded": general_excluded}
    heal_data["_align"] = {"pool": "prose", "fallback": None, "count": 150, "seed": req["seed"] + 3, "skip": 0,
                           "sources": pool_sources("prose", reported)[0], "excluded": []}
    probe_texts["_general"] = {"pool": "general", "count": s["probe_texts"], "seed": req["seed"], "skip": s["probe_skip"],
                               "splits": ["a", "b", "c"], "max_length": s["probe_length"]}

    controls = _controls(plan_stages)
    arms = _arms(req, plan_stages, sources_out, base_total, atlas_entries, limits)
    if "pgraft" in plan_stages:
        arms = [{"arm": "base", "what": "the host alone, scored first in the same run on the same items"},
                {"arm": "donor", "what": "the donor alone, scored once on the same items (donor_acc.json of the cached pool)"}]
    for method in plan_stages:
        warnings.append(MODULE_STATE[method])
    native_bars = []
    for st in stages:
        if st["method"] != "pgraft":
            continue
        from wsp import pgraft_stage

        if req["base"] != pgraft_stage.MEASURED["host"] or st["source"] != pgraft_stage.MEASURED["donor"]:
            warnings.append("pgraft: %s with %s is not the measured pair (%s with %s); expect a different gain, and none "
                            "is possible if the donor has no more knowledge than the host" % (
                                req["base"], st["source"], pgraft_stage.MEASURED["host"], pgraft_stage.MEASURED["donor"]))
        if limits["max_heal_minutes"] is not None and limits["max_heal_minutes"] < st["estimate"]["gpu_hours"] * 60.0:
            warnings.append("pgraft needs about %.0f minutes on one %s (measured, 2026-11-15) and max_heal_minutes is %.0f"
                            % (st["estimate"]["gpu_hours"] * 60.0, st["estimate"]["gpu"], limits["max_heal_minutes"]))
        warnings.append("pgraft trains on benchmark train splits (MMLU auxiliary_train and four-option train sets, every "
                        "eval prompt removed) and is scored on held-out test items nothing trained on; each claim needs "
                        "3 seeds, and a compile is one seed")
        warnings.append("pgraft decodes through a hook-based path (27 to 36 tokens per second at batch 1 in the measured "
                        "run against 103 for the host); the per-token cost is the host plus the donor slice")
        from wsp import native_bar

        measured = req["base"] == pgraft_stage.MEASURED["host"] and st["source"] == pgraft_stage.MEASURED["donor"]
        expected = pgraft_stage.MEASURED["mmlu_held"]["graft"] if measured else None
        bar = native_bar.check(running_run, expected, exclude=(req["base"],))
        native_bars.append(bar)
        if bar["clears"] is False:
            warnings.append("pgraft: the measured write-in (%.1f) does not clear the native bar: %s scores %.1f at %.2fB "
                            "per token (see Native bar)" % (expected, bar["bar"]["model"], bar["bar_score"],
                                                            bar["bar"]["params_per_token"] / 1e9))
    for x in sources_out:
        for text in x["rel"].get("warnings", []):
            warnings.append("%s from %s: %s" % (x["skill"], x["source"], text))
        if x["rel"]["kind"] in ("fine_tune", "fine_tune_of_ancestor", "same") and x["rel"].get("evidence", {}).get(
                "relative_delta_norm") is not None and x["rel"]["evidence"]["relative_delta_norm"] < 0.01:
            warnings.append("%s: the weight difference is tiny (relative L2 %.5f); a mix cannot add much"
                            % (x["source"], x["rel"]["evidence"]["relative_delta_norm"]))
        if not x["rel"]["flags"].get("tokenizer_same", True):
            warnings.append("%s has another tokenizer: transfer cannot be measured by KL, so its verdict is limited" % x["source"])
    warnings = list(dict.fromkeys(warnings))

    source_rows = []
    for x in sources_out:
        rel = x["rel"]
        source_rows.append({
            "skill": x["skill"], "source": x["source"], "relation": rel["kind"],
            "flags": rel["flags"], "evidence": {k: v for k, v in rel.get("evidence", {}).items() if k != "probe"},
            "subtract_reference": rel.get("subtract_reference"),
            "params_b": round(billions(x["facts"]["params"]), 4),
            "methods": [{"method": o["method"], "fits": o["fits"], "reason": o["reason"], "why_not": o["why_not"],
                         "adds_total_b": None if o["add_total"] is None else round(billions(o["add_total"]), 4),
                         "needs": o.get("needs"), "runs_in_make": o.get("runs_in_make", True)} for o in x["options"]],
            "chosen": chosen.get(id(x)),
            "components": x["rows"]})
    return {
        "name": req["name"], "base": base_info, "limits": limits, "reported": req["reported"], "seed": req["seed"],
        "keep": req["keep"], "remove": req["remove"],
        "rule": ("stages whose output is a plain Hugging Face model (weave, then slim) run first and each takes the "
                 "previous output as its base. At most one seam stage (pgraft, synth, feedback, transplant, or delta) runs last "
                 "on top. Two seam kinds are refused because each replaces the other's branches."),
        "sources": source_rows, "stages": stages, "sizes": {"base_b": base_info["params_b"], "after": sizes_after,
                                                            "final_total_b": round(billions(running_total), 4),
                                                            "final_run_b": round(billions(running_run), 4)},
        "heal_data": heal_data, "probe_texts": probe_texts, "practice": practice, "controls": controls, "arms": arms,
        "settings": s, "warnings": warnings, "refusals": refusals, "feasible": len(refusals) == 0 and len(plan_stages) > 0,
        "native_bar": native_bars,
    }


def _weave_settings(s, minutes, base_total, members):
    population = s["weave_population"]
    generations = s["weave_generations"]
    note = "defaults"
    if minutes is not None:
        candidate = 2.0 + 0.5 * billions(base_total) * (members + 1) * 0.2
        evals = max(4, int(minutes * 60.0 * 0.85 / candidate))
        while population * generations > evals and generations > 2:
            generations -= 1
        while population * generations > evals and population > 4:
            population -= 2
        note = "shrunk to fit %.1f minutes at about %.1f s per candidate (a guess)" % (minutes, candidate)
    return {"buckets": s["weave_buckets"], "population": population, "generations": generations,
            "objective": "probe (wsp.weave.ProbeObjective, forward passes only)", "note": note}


def _seam_settings(method, s, minutes, active):
    if method in ("synth", "feedback", "transplant"):
        steps = steps_for(minutes, active, s)
        if steps is None:
            return {"stage1_steps": s["stage1_steps"], "stage2_steps": s["stage2_steps"], "lora_rank": s["lora_rank"],
                    "batch_size": s["batch_size"], "seq_len": s["seq_len"], "note": "defaults"}
        return {"stage1_steps": max(1, steps // 2), "stage2_steps": max(1, steps - steps // 2),
                "lora_rank": s["lora_rank"], "batch_size": s["batch_size"], "seq_len": s["seq_len"],
                "note": "%d steps for %.1f minutes at the estimated step time" % (steps, minutes)}
    return {"rank": s["delta_rank"], "note": "delta stores low-rank differences; no heal steps"}


def _method_options(lin, rel, base_cfg, base_total, base_run, src_facts, skill, limits, s, atlas_entries, rows, override):
    """List the valid methods for one source with size and fit. First fitting one in order is chosen."""
    listed = [m for m in lin.methods(rel, "add") if m["method"] != "whole_model"]
    if not rel["flags"].get("teacher_usable_for_logit_kd", False):
        listed = [m for m in listed if m["method"] != "feedback"]
    options = []
    kind = rel["kind"]
    if kind == "same":
        return [{"method": "none", "fits": False, "reason": "", "why_not": "the source is the same model as the base "
                 "(relative L2 under 1e-6 on every sampled tensor, so the weights are equal); there is nothing to add", "add_total": None,
                 "add_run": None}]
    src_cfg = src_facts["config"]
    for entry in listed:
        method = entry["method"]
        if override is not None and method != override:
            continue
        option = {"method": method, "reason": entry["reason"], "fits": False, "why_not": None, "add_total": None,
                  "add_run": None}
        try:
            _size_option(option, method, base_cfg, base_total, base_run, src_cfg, src_facts, skill, limits, s,
                         atlas_entries, rows)
        except (KeyError, TypeError) as error:
            option["why_not"] = "the config lacks a field the size estimate needs (%s)" % error
        _fix_floor_text(option, base_total, base_run, limits)
        options.append(option)
    from wsp import pgraft_stage

    ok, why = pgraft_stage.eligible(skill, rel, base_cfg, src_cfg)
    if ok and override in (None, "pgraft"):
        option = {"method": "pgraft", "reason": why, "fits": False, "why_not": None, "add_total": None, "add_run": None}
        try:
            _size_option(option, "pgraft", base_cfg, base_total, base_run, src_cfg, src_facts, skill, limits, s,
                         atlas_entries, rows)
        except (KeyError, TypeError, ValueError) as error:
            option["why_not"] = "the config lacks a field the size estimate needs (%s)" % error
        _fix_floor_text(option, base_total, base_run, limits)
        options.insert(0, option)
    if override is not None and len(options) == 0:
        options.append({"method": override, "fits": False, "reason": "", "why_not": "method %s is not valid for a %s source; "
                        "valid: %s" % (override, kind, ", ".join(m["method"] for m in listed) or "none"),
                        "add_total": None, "add_run": None})
    if len(options) == 0:
        options.append({"method": "none", "fits": False, "reason": "", "why_not": "no method is valid for relation %s" % kind,
                        "add_total": None, "add_run": None})
    return options


def _headroom(base_total, base_run, limits):
    """Parameters a seam may add under the limits, as (total, run). None means no limit."""
    total = None
    run = None
    if limits["max_total_params_b"] is not None:
        total = limits["max_total_params_b"] * 1e9 - base_total
    if limits["max_active_params_b"] is not None:
        run = limits["max_active_params_b"] * 1e9 - base_run
    return total, run


def _size_option(option, method, base_cfg, base_total, base_run, src_cfg, src_facts, skill, limits, s, atlas_entries, rows):
    head_total, head_run = _headroom(base_total, base_run, limits)
    if method == "weave":
        option.update(fits=True, add_total=0, add_run=0, needs="download of the source (%.1f GiB at bf16)" % (
            src_facts["params"] * 2 / 2**30))
        if head_total is not None and head_total < 0 and limits["max_total_params_b"] < billions(base_total):
            option.update(fits=False, why_not="weave keeps the base size %.3fB, above max_total_params_b %.3f; add a skill "
                          "removal to shrink it" % (billions(base_total), limits["max_total_params_b"]))
        return
    if method == "pgraft":
        from wsp import pgraft_stage

        sz = pgraft_stage.sizes(base_cfg, src_cfg, s)
        spec = sz["spec"]
        option.update(add_total=sz["add_total"], add_run=sz["add_run"],
                      needs="donor layers %d..%d of %d (%.3fB frozen), Enc and Dec glue %.1fM, LoRA rank %d" % (
                          spec["first"], spec["last"], src_cfg["layers"], billions(sz["slice_params"]),
                          sz["glue_params"] / 1e6, spec["lora_rank"]))
        _fit(option, head_total, head_run, "a donor slice with its glue", sz["add_total"], sz["add_run"])
        return
    if method == "delta":
        stored, run = delta_addition(base_cfg, s["delta_rank"])
        option.update(add_total=stored, add_run=run)
        _fit(option, head_total, head_run, "a rank-%d delta" % s["delta_rank"], stored, run)
        return
    if method == "transplant":
        trial = [(s["transplant_k"], s["transplant_rank"]), (3, s["transplant_rank"]), (2, s["transplant_rank"]),
                 (2, 128), (1, 128), (1, 64)]
        last = None
        for k, rank in trial:
            total, run = transplant_addition(base_cfg, src_cfg, k, rank)
            last = (k, rank, total, run)
            if (head_total is None or total <= head_total) and (head_run is None or run <= head_run):
                option.update(fits=True, add_total=total, add_run=run, k=k, rank=rank,
                              needs="K=%d experts per layer, stitch rank %d" % (k, rank))
                return
        k, rank, total, run = last
        option.update(add_total=total, add_run=run)
        _fit(option, head_total, head_run, "the smallest bank (K=%d, rank %d)" % (k, rank), total, run)
        return
    if method in ("synth", "feedback"):
        if rows:
            total = 0
            run = 0
            for row in rows:
                total += row_params(row, src_cfg)
            glue = 2 * base_cfg["hidden"] * src_cfg["hidden"] * len(rows) + 2 * base_cfg["hidden"] * len(rows)
            lora = lora_params(base_cfg, s["lora_rank"], base_cfg["layers"] // 2)
            option.update(add_total=total + glue + lora, add_run=total + glue + lora, needs="%d catalog component(s)" % len(rows))
            _fit(option, head_total, head_run, "the listed components", total + glue + lora, total + glue + lora)
            return
        host_after = max(1, base_cfg["layers"] // 2)
        one_total, one_run = seam_addition(method, base_cfg, base_total, src_cfg, 1, s["lora_rank"], host_after)
        want = min(s["block_layers"], src_cfg["layers"])
        n_fit = want
        while n_fit >= 1:
            a_t, a_r = seam_addition(method, base_cfg, base_total, src_cfg, n_fit, s["lora_rank"], host_after)
            if (head_total is None or a_t <= head_total) and (head_run is None or a_r <= head_run):
                option.update(fits=True, add_total=a_t, add_run=a_r, n_layers=n_fit,
                              needs="a block of %d donor layers (%d layers wanted)" % (n_fit, want))
                return
            n_fit -= 1
        option.update(add_total=one_total, add_run=one_run)
        _fit(option, head_total, head_run, "even one donor layer with its translation layers", one_total, one_run)
        return
    if method == "tops":
        option["runs_in_make"] = False
        add_total, add_run = tops_addition(base_cfg, base_run, src_cfg, s["tops_cut"], s["lora_rank"])
        option.update(add_total=add_total, add_run=add_run,
                      needs="trunk of %d of %d base layers, then the donor's upper layers and head"
                            % (max(1, int(round(s["tops_cut"] * base_cfg["layers"]))), base_cfg["layers"]))
        _fit(option, head_total, head_run, "a skill top", add_total, add_run)
        return
    option["why_not"] = "no size rule for method %s" % method


def _fit(option, head_total, head_run, what, total, run):
    problems = []
    if head_total is not None and total > head_total:
        problems.append("%s needs %.3fB more parameters but max_total_params_b leaves %.3fB; raising it to at least "
                        "%.3f would allow it" % (what, billions(total), max(0.0, billions(head_total)),
                                                  option.get("_floor_total", 0) or 0))
    if head_run is not None and run > head_run:
        problems.append("%s adds %.3fB run per token but max_active_params_b leaves %.3fB" % (
            what, billions(run), max(0.0, billions(head_run))))
    if problems:
        option["why_not"] = "; ".join(problems)
        option["fits"] = False
    else:
        option["fits"] = True
        option["why_not"] = None


def _fix_floor_text(option, base_total, base_run, limits):
    """Fill in the limit that would allow the option, in the why_not text."""
    if option["why_not"] and option["add_total"] is not None and limits["max_total_params_b"] is not None:
        floor = billions(base_total + option["add_total"])
        option["why_not"] = re.sub(r"raising it to at least [0-9.]+", "raising it to at least %.3f" % floor, option["why_not"])


def chosen_add(item, method, full=False):
    """Parameters the chosen method adds for a source: (total, run) or (total, run, info)."""
    option = None
    for o in item["options"]:
        if o["method"] == method:
            option = o
    total = option["add_total"] or 0
    run = option["add_run"] or 0
    if not full:
        return total, run
    info = {"skill": item["skill"], "source": item["source"], "method": method, "adds_params": int(total)}
    for key in ("n_layers", "k", "rank"):
        if key in option:
            info[key] = option[key]
    if item["rows"]:
        info["rows"] = [r["id"] for r in item["rows"]]
    return total, run, info


def _choose_methods(sources_out, refusals):
    """Pick the first fitting method per source. Plain methods first because the list is in method order."""
    chosen = {}
    for x in sources_out:
        fitting = [o for o in x["options"] if o["fits"] and o.get("runs_in_make", True)]
        if fitting:
            chosen[id(x)] = fitting[0]["method"]
            continue
        reasons = []
        for o in x["options"]:
            if o["fits"] and not o.get("runs_in_make", True):
                reasons.append("%s: valid, but wsp.auto.make has no %s stage; build it as a program (wsp/tops.py)"
                               % (o["method"], o["method"]))
            else:
                reasons.append("%s: %s" % (o["method"], o["why_not"] or "does not fit"))
        refusals.append("add %s from %s (relation %s): no method fits. %s" % (
            x["skill"], x["source"], x["rel"]["kind"], "; ".join(reasons)))
    # sources sharing one seam method: when the picks differ but one method fits every seam source, use it
    seam = [x for x in sources_out if chosen.get(id(x)) in SEAM_STAGES]
    if len({chosen[id(x)] for x in seam}) > 1:
        for method in SEAM_PRIORITY:
            if all(any(o["method"] == method and o["fits"] for o in x["options"]) for x in seam):
                for x in seam:
                    chosen[id(x)] = method
                break
    return chosen


def _controls(stages):
    out = []
    for method in stages:
        if method == "weave":
            out.append({"stage": "weave", "controls": ["random: best genome on differences shuffled inside each tensor "
                                                       "(Weaver.shuffled, several seeds, for content_z)"]})
        elif method == "synth":
            out.append({"stage": "synth", "controls": ["random: another layer range of the same donor",
                                                       "scrambled: the same layers with weights rewritten by wsp.scramble"]})
        elif method == "feedback":
            out.append({"stage": "feedback", "controls": ["random: another layer range", "plain branch: the capacity control"]})
        elif method == "transplant":
            out.append({"stage": "transplant", "controls": ["random: random experts of the same layer"]})
        elif method == "delta":
            out.append({"stage": "delta", "controls": ["random: same shapes and norm, random directions"]})
        elif method == "slim":
            out.append({"stage": "slim", "controls": ["random: same counts, random experts or units"]})
        elif method == "pgraft":
            from wsp import xfam

            out.append({"stage": "pgraft", "controls": ["%s: %s" % (name, what) for name, what in xfam.CONTROLS]})
    return out


def _arms(req, stages, sources_out, base_total, atlas_entries, limits):
    arms = [{"arm": "base", "what": "the base model alone"},
            {"arm": "base_lora", "what": "the base plus a rank-16 LoRA healed on the same texts and steps"}]
    teacher_ok = [x for x in sources_out if x["rel"]["flags"].get("teacher_usable_for_logit_kd")]
    if teacher_ok:
        arms.append({"arm": "base_distill", "what": "the base plus a LoRA distilled from %s on the same texts and steps"
                     % teacher_ok[0]["source"], "teacher": teacher_ok[0]["source"]})
    else:
        arms.append({"arm": "base_distill", "what": "not built: no source shares the base's tokenizer", "skipped": True})
    limit = limits["max_total_params_b"]
    if limit is not None and limit > 2.0 * base_total / 1e9 and req["add"]:
        best = None
        for add in req["add"]:
            rows = auto_candidates(add["skill"], req["base"], atlas_entries, [], req["reported"])
            rows = [r for r in rows if r["params_b"] is None or r["params_b"] <= limit]
            if rows:
                best = rows[0]
                break
        if best is not None:
            arms.append({"arm": "best_existing", "what": "the best mined model within the size limit", "model": best["model"],
                         "params_b": best["params_b"]})
        else:
            arms.append({"arm": "best_existing", "what": "not built: no atlas entry within the limit", "skipped": True})
    return arms


# ---------------------------------------------------------------- plan text


def _format_pgraft(st, out):
    first, last = st["donor_span"]
    out("     write %s into the host as one network. Host layer %d (%s)" % (st["source"], st["host_layer"], st["layer_rule"]))
    out("     donor span layers %d..%d of %d (%d layers, %.3fB parameters, frozen)" % (
        first, last, st["donor_layers"], st["donor_slice_layers"], billions(st["slice_params"])))
    out("     Enc depth %d (%d to %d, %.2fM)  Dec rank %d (%.2fM)  per-token gate (%d)  host LoRA rank %d (%.2fM)" % (
        st["enc"]["depth"], st["enc"]["d_host"], st["enc"]["d_donor"], st["enc"]["params"] / 1e6, st["dec"]["rank"],
        st["dec"]["params"] / 1e6, st["gate_params"], st["lora_rank"], st["lora_params"] / 1e6))
    reads = st["enc"].get("reads") or [st["host_layer"]]
    if len(reads) > 1:
        out("     Enc reads a learned mix of host layers %d..%d (each RMS-normed), not host layer %d alone" % (
            reads[0], reads[-1], st["host_layer"]))
    d = st["data"]
    out("     data pool %s: %s; %s; %s" % (d["pool"], d["train_items"], d["soft_labels"], d["size"]))
    out("     prompts removed: %s. Text: %s" % (d["removed"], d["text"]))
    out("     scored on held-out items: %s" % ", ".join(d["scored_on"]))
    out("     steps: %d alignment, %d task, batch %d; about %.2f GPU hours on one %s (measured)" % (
        st["settings"]["align_steps"], st["settings"]["steps"], st["settings"]["batch"], st["estimate"]["gpu_hours"],
        st["estimate"]["gpu"]))
    out("     params per token after the write-in: %.3fB (host %.3fB + donor slice %.3fB + glue %.3fB); the LoRA merges "
        "into the host. Stored total %.3fB" % (
            st["params_run_after_b"], st["params_run_after_b"] - billions(st["params_per_token_added"]),
            billions(st["slice_params"]), billions(st["glue_params"]), st["params_total_after_b"]))
    for c in st["controls"]:
        out("     control %s: %s. Run by %s" % (c["name"], c["what"], c["run_by"]))


def format_plan(p):
    """Return the plan as readable text."""
    lines = []
    out = lines.append
    base = p["base"]
    out("Sapien plan: %s" % p["name"])
    out("base %s  %.3fB total, %.3fB per token%s" % (base["id"], base["params_b"], base["active_params_b"],
                                                    ", MoE" if base["moe"] else ""))
    lim = p["limits"]
    out("limits  total %s  active %s  heal minutes %s" % (lim["max_total_params_b"], lim["max_active_params_b"],
                                                          lim["max_heal_minutes"]))
    out("")
    out("Stage rule: " + p["rule"])
    out("")
    for x in p["sources"]:
        out("add %s from %s" % (x["skill"], x["source"]))
        out("  relation %s  tokenizer same: %s" % (x["relation"], x["flags"].get("tokenizer_same")))
        for m in x["methods"]:
            mark = "ok " if m["fits"] else "no "
            text = m["reason"] if m["fits"] else (m["why_not"] or "")
            out("  %s%s: %s" % (mark, m["method"], text))
        out("  chosen: %s" % x["chosen"])
    for r in p["remove"]:
        out("remove %s" % r["skill"])
    if p["keep"]:
        out("keep %s" % ", ".join(p["keep"]))
    out("")
    out("Stages")
    for st in p["stages"]:
        out("  %d. %s (%s)  total %.3fB  per token %.3fB" % (st["order"], st["method"], st["kind"],
                                                               st["params_total_after_b"], st["params_run_after_b"]))
        if st.get("settings"):
            out("     settings: %s" % json.dumps(st["settings"]))
        if st["method"] == "pgraft":
            _format_pgraft(st, out)
        for problem in st["limit_problems"]:
            out("     over limit: %s" % problem)
    out("")
    if any(st["method"] == "pgraft" for st in p["stages"]):
        out("Heal text, probe text, and practice sets: not used by pgraft. Its data pool and eval sets are in the stage above.")
    else:
        out("Heal text (reported benchmarks excluded)")
        for skill, h in p["heal_data"].items():
            ids = ", ".join(s["id"] for s in h["sources"]) or "none"
            note = "  fallback: %s" % h["fallback"] if h["fallback"] else ""
            out("  %s: pool %s  [%s]%s" % (skill, h["pool"], ids, note))
        out("Probe text: rows %d and later of each pool, three splits of %d" % (
            p["settings"]["probe_skip"], p["settings"]["probe_texts"]))
        out("Practice sets")
        for skill, info in p["practice"].items():
            if info["available"]:
                out("  %s: %s" % (skill, ", ".join(s["id"] for s in info["sources"])))
            else:
                out("  %s: none (%s)" % (skill, info["reason"]))
    out("Controls")
    for c in p["controls"]:
        out("  %s: %s" % (c["stage"], "; ".join(c["controls"])))
    out("Comparison arms")
    for a in p["arms"]:
        out("  %s: %s" % (a["arm"], a["what"]))
    out("")
    if p.get("native_bar"):
        out("Native bar")
        for bar in p["native_bar"]:
            out("  " + bar["text"])
        out("")
    if p["warnings"]:
        out("Warnings")
        for w in p["warnings"]:
            out("  - " + w)
    if p["refusals"]:
        out("Refused")
        for r in p["refusals"]:
            out("  - " + r)
    out("")
    out("feasible: %s" % p["feasible"])
    return "\n".join(lines)


# ================================================================ running a plan
#
# make() runs a plan through a backend. The backend loads models, builds text,
# and writes directories. LocalBackend holds tiny models in memory (tests).
# modal_sapien.ModalBackend implements the same calls on a GPU.


def static_row(model, tokenizer_id="t"):
    """A StaticLineage table row for a loaded model (config and parameter count)."""
    from wsp import lineage

    seen = set()
    total = 0
    for param in model.parameters():
        if id(param) not in seen:
            seen.add(id(param))
            total += param.numel()
    return {"params": total, "config": lineage.normalize_config(model.config.to_dict()), "tokenizer": tokenizer_id}


class Backend:
    """The calls make() needs. Subclasses fill them in."""

    def load(self, model_id):
        """Return a fresh (model, tokenizer) of a plain Hugging Face model."""
        raise NotImplementedError

    def reference(self, model_id, host):
        """Return the mapped reference tensors of a common ancestor for weave."""
        raise NotImplementedError

    def member(self, name, skill, model_id, host, ref, ref_name, mlp_only, meta):
        """Return a weave.Member for a source checkpoint."""
        raise NotImplementedError

    def teacher_cache(self, model_id, texts, tokenizer, top_k, max_length, batch_size, vocab=None):
        """Run a source model on texts once and return a weave teacher cache. The source is freed after."""
        raise NotImplementedError

    def teacher_caches(self, model_id, texts_list, tokenizer, top_k, max_length, batch_size, vocab=None):
        """One cache per text list. A backend that loads the source overrides this to load it once."""
        return [self.teacher_cache(model_id, texts, tokenizer, top_k, max_length, batch_size, vocab) for texts in texts_list]

    def pool(self, name, count, tokenizer, seed, skip, reported):
        raise NotImplementedError

    def practice_items(self, skill, count, reported, seed):
        raise NotImplementedError

    def generate(self, model, tokenizer, prompts, max_new_tokens):
        raise NotImplementedError

    def save_plain(self, model, tokenizer, path):
        raise NotImplementedError

    def pgraft_data(self, stage, donor, donor_tok, host_id):
        """Four-option items, the donor's soft letter distribution on them, and the eval sets, for a pgraft stage.

        Returns {"items", "soft", "evals", "source"}.
        """
        raise NotImplementedError

    def pgraft_texts(self, stage, request):
        """Alignment and general text pools for a pgraft stage: {"align", "align_eval", "general"}."""
        raise NotImplementedError

    def checkpoint(self):
        """Called after a stage wrote files that must survive a crash (the Modal runner commits its volume)."""
        return None


class LocalBackend(Backend):
    """Tiny models held in memory: models is {id: (model, tokenizer)}.

    text_fn(name, count, tokenizer, seed, skip, reported) and item_fn(skill, count, reported, seed) replace the
    network calls to wsp.healdata. calls records every healdata-style call so a test can check the reported set.
    """

    def __init__(self, models, text_fn=None, item_fn=None):
        self.models = models
        self.text_fn = text_fn
        self.item_fn = item_fn
        self.calls = []

    def load(self, model_id):
        model, tokenizer = self.models[model_id]
        return copy.deepcopy(model), tokenizer

    def reference(self, model_id, host):
        from wsp import weave

        model, _ = self.models[model_id]
        return weave.map_ref(model.state_dict(), weave.host_index(host), ("attn", "mlp", "norm"))

    def member(self, name, skill, model_id, host, ref, ref_name, mlp_only, meta):
        from wsp import weave

        model, _ = self.models[model_id]
        return weave.member_from_state(name, skill, model.state_dict(), host, ref=ref, ref_name=ref_name,
                                       mlp_only=mlp_only, groups=("attn", "mlp", "norm"), meta=meta)

    def teacher_cache(self, model_id, texts, tokenizer, top_k, max_length, batch_size, vocab=None):
        from wsp import weave

        model, _ = self.models[model_id]
        return weave.build_teacher_cache(model, tokenizer, texts, top_k=top_k, max_length=max_length,
                                         batch_size=batch_size, vocab=vocab)

    def pool(self, name, count, tokenizer, seed, skip, reported):
        self.calls.append(("pool", name, reported))
        if self.text_fn is not None:
            return self.text_fn(name, count, tokenizer, seed, skip, reported)
        return healdata.pool(name, count, tokenizer=tokenizer, seed=seed, skip=skip, reported=reported)

    def practice_items(self, skill, count, reported, seed):
        self.calls.append(("practice", skill, reported))
        if self.item_fn is not None:
            return self.item_fn(skill, count, reported, seed)
        return healdata.practice(skill, count, reported=reported, seed=seed)

    def generate(self, model, tokenizer, prompts, max_new_tokens):
        return generate_texts(model, tokenizer, prompts, max_new_tokens)

    def save_plain(self, model, tokenizer, path):
        os.makedirs(path, exist_ok=True)
        model.save_pretrained(path)
        if tokenizer is not None:
            tokenizer.save_pretrained(path)

    pgraft_data_fn = None
    pgraft_texts_fn = None

    def pgraft_data(self, stage, donor, donor_tok, host_id):
        if self.pgraft_data_fn is None:
            raise NotImplementedError("set pgraft_data_fn on the LocalBackend (a tiny pool for tests)")
        return self.pgraft_data_fn(stage, donor, donor_tok, host_id)

    def pgraft_texts(self, stage, request):
        if self.pgraft_texts_fn is None:
            raise NotImplementedError("set pgraft_texts_fn on the LocalBackend (tiny texts for tests)")
        return self.pgraft_texts_fn(stage, request)


class HubBackend(Backend):
    """Models from the Hugging Face Hub, loaded whole. The Modal runner subclasses this.

    Weave members are read from downloaded checkpoint directories. Donors and sources load fully into memory.
    """

    PATTERNS = ["*.safetensors", "*.json", "*.model", "*.jinja", "merges.txt", "tokenizer*"]

    def __init__(self, device=None, dtype=None, cache_dir=None):
        import torch

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype if dtype is not None else (torch.bfloat16 if self.device == "cuda" else torch.float32)
        self.cache_dir = cache_dir
        self.calls = []

    def download(self, model_id):
        if os.path.isdir(model_id):
            return model_id
        from huggingface_hub import snapshot_download

        return snapshot_download(model_id, allow_patterns=self.PATTERNS, cache_dir=self.cache_dir)

    def load(self, model_id):
        import transformers

        directory = self.download(model_id)
        model = transformers.AutoModelForCausalLM.from_pretrained(directory, dtype=self.dtype).to(self.device).eval()
        tokenizer = transformers.AutoTokenizer.from_pretrained(directory)
        return model, tokenizer

    def reference(self, model_id, host):
        from wsp import weave

        return weave.reference_from_directory(self.download(model_id), host, groups=("attn", "mlp", "norm"))

    def member(self, name, skill, model_id, host, ref, ref_name, mlp_only, meta):
        from wsp import weave

        return weave.member_from_directory(name, skill, self.download(model_id), host, ref=ref, ref_name=ref_name,
                                           mlp_only=mlp_only, groups=("attn", "mlp", "norm"), meta=meta)

    def teacher_caches(self, model_id, texts_list, tokenizer, top_k, max_length, batch_size, vocab=None):
        import gc

        import torch
        from wsp import weave

        model, _ = self.load(model_id)
        caches = []
        for texts in texts_list:
            caches.append(weave.build_teacher_cache(model, tokenizer, texts, top_k=top_k, max_length=max_length,
                                                    batch_size=batch_size, vocab=vocab))
        del model
        gc.collect()
        if self.device == "cuda":
            torch.cuda.empty_cache()
        return caches

    def teacher_cache(self, model_id, texts, tokenizer, top_k, max_length, batch_size, vocab=None):
        return self.teacher_caches(model_id, [texts], tokenizer, top_k, max_length, batch_size, vocab)[0]

    def pool(self, name, count, tokenizer, seed, skip, reported):
        self.calls.append(("pool", name, reported))
        return healdata.pool(name, count, tokenizer=tokenizer, seed=seed, skip=skip, reported=reported)

    def practice_items(self, skill, count, reported, seed):
        self.calls.append(("practice", skill, reported))
        return healdata.practice(skill, count, reported=reported, seed=seed)

    def generate(self, model, tokenizer, prompts, max_new_tokens):
        return generate_texts(model, tokenizer, prompts, max_new_tokens)

    def save_plain(self, model, tokenizer, path):
        os.makedirs(path, exist_ok=True)
        model.save_pretrained(path)
        if tokenizer is not None:
            tokenizer.save_pretrained(path)

    def pgraft_data(self, stage, donor, donor_tok, host_id):
        """The cached pool and eval sets of the wi_xfam rows (stage["data"]["dir"]); built from scratch on a miss."""
        from wsp import xfam

        return xfam.load_or_build_data(stage["data"]["dir"], host_id, donor, donor_tok)

    def pgraft_texts(self, stage, request):
        from wsp import xfam

        return xfam.load_or_build_texts(xfam.TEXT_CACHE, hd_reported(request["reported"]))


def generate_texts(model, tokenizer, prompts, max_new_tokens=64, batch_size=8):
    """Greedy answers to prompts, batch by batch. Chat template when the tokenizer has one."""
    import torch

    device = next(model.parameters()).device
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    outputs = []
    model.eval()
    for start in range(0, len(prompts), batch_size):
        chunk = prompts[start:start + batch_size]
        if getattr(tokenizer, "chat_template", None):
            chunk = [tokenizer.apply_chat_template([{"role": "user", "content": p}], tokenize=False,
                                                   add_generation_prompt=True) for p in chunk]
        enc = tokenizer(chunk, return_tensors="pt", padding=True, add_special_tokens=not getattr(
            tokenizer, "chat_template", None)).to(device)
        with torch.no_grad():
            out = model.generate(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"], max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tokenizer.pad_token_id)
        for row in out:
            outputs.append(tokenizer.decode(row[enc["input_ids"].shape[1]:], skip_special_tokens=True))
    tokenizer.padding_side = side
    return outputs


def practice_scorer(backend, items, max_new_tokens=48):
    """Return fn(model, tokenizer) -> share of practice items answered right (wsp.healdata.check_answer)."""
    def score(model, tokenizer):
        answers = backend.generate(probe_view(model), tokenizer, [item["prompt"] for item in items], max_new_tokens)
        right = 0
        for item, text in zip(items, answers):
            try:
                if healdata.check_answer(item, text):
                    right += 1
            except Exception:
                continue
        return right / float(max(1, len(items)))

    return score


class Context:
    """State shared by the stages of one make() call."""

    def __init__(self, req, plan_, backend, out_dir, model, tokenizer, atlas_entries):
        self.req = req
        self.plan = plan_
        self.s = plan_["settings"]
        self.backend = backend
        self.out_dir = out_dir
        self.model = model
        self.tokenizer = tokenizer
        self.reported = hd_reported(req["reported"])
        self.atlas_entries = atlas_entries
        self.texts = {}
        self.caches = {}
        self.baseline = {}
        self.practice = {}
        self.base_practice = {}

    def pool_texts(self, key, split=None):
        """Heal text (split None) or one probe split a, b, c of a plan pool entry."""
        if split is None:
            entry = self.plan["heal_data"][key]
            return self.backend.pool(entry["pool"], entry["count"], self.tokenizer, entry["seed"], entry["skip"], self.reported)
        entry = self.plan["probe_texts"][key]
        n = entry["count"]
        index = "abc".index(split)
        everything = self.backend.pool(entry["pool"], 3 * n, self.tokenizer, entry["seed"], entry["skip"], self.reported)
        return everything[index * n:(index + 1) * n]


def _quiet(*args, **kwargs):
    return None


def base_ref(ctx):
    """The host a saved seam stage rebuilds from: the request's base, or the saved output of the plain stages before."""
    if getattr(ctx, "stage_base_is_original", True):
        return ctx.req["base"]
    return os.path.join(ctx.out_dir, "stage_base")


def run_weave(ctx, stage):
    """Per-layer mix of true fine-tunes at exactly the base size (wsp/weave.py)."""
    from wsp import weave

    s = ctx.s
    model, tok = ctx.model, ctx.tokenizer
    started = time.time()
    members = []
    gates = {}
    refs = {}
    names = []
    for index, spec in enumerate(stage["members"]):
        name = "m%d_%s" % (index, spec["skill"])
        ref = None
        ref_name = "host"
        if spec["reference"] != "base":
            if spec["reference"] not in refs:
                refs[spec["reference"]] = ctx.backend.reference(spec["reference"], model)
            ref = refs[spec["reference"]]
            ref_name = spec["reference"]
        member = ctx.backend.member(name, spec["skill"], spec["source"], model, ref, ref_name, spec["mlp_only"],
                                    {"id": spec["source"]})
        gate = weave.gate_member(member, model)
        gate["mlp_only"] = spec["mlp_only"]
        gates[name] = gate
        if gate["passed"]:
            members.append(member)
            names.append(name)
    report = {"method": "weave", "gate": gates, "members_used": names}
    if len(members) == 0:
        report["status"] = "no_admissible_member"
        report["seconds"] = round(time.time() - started, 2)
        return report
    weaver = weave.Weaver(model, members, n_buckets=stage["settings"]["buckets"])
    sources = {m.name: stage["members"][int(m.name[1:].split("_")[0])] for m in members}
    po = {}
    for split in "abc":
        member_caches = {}
        for m in members:
            member_caches[m.name] = ctx.caches[("donor", sources[m.name]["source"], m.skill, split)]
        po[split] = weave.ProbeObjective([m.name for m in members], member_caches, ctx.caches[("general", split)],
                                         s["drift_tolerance"], 10.0, s["batch_size"])
        po[split].calibrate(model)
    result = weave.search(weaver, model, po["a"].scores, po["b"].scores, po["a"], po["b"],
                          popsize=stage["settings"]["population"], generations=stage["settings"]["generations"],
                          seed=ctx.req["seed"], log=_quiet)
    best = result["best_genome"]
    weaver.apply(model, best)
    real = po["c"].scores(model)
    weaver.restore(model)
    controls = {m.name: [] for m in members}
    for j in range(s["n_controls"]):
        clone = weaver.shuffled(ctx.req["seed"] + 1 + j)
        clone.apply(model, best)
        scores = po["c"].scores(model)
        weaver.restore(model)
        for m in members:
            controls[m.name].append(scores["transfer"][m.name])
    weaver.apply(model, best)
    path = os.path.join(ctx.out_dir, "stage_weave")
    ctx.backend.save_plain(model, tok, path)
    with open(os.path.join(path, "weave.json"), "w") as handle:
        json.dump({"genome": best.to_dict(), "members": weaver.summary(), "n_buckets": weaver.n_buckets}, handle, indent=1,
                  default=str)
    report.update({"status": "ok", "search": result["summary"], "best_genome": best.to_dict(),
                   "transfer_c": real["transfer"], "drift_c": real["drift"], "controls": controls, "saved": path,
                   "seconds": round(time.time() - started, 2)})
    return report


def _kept_removed_texts(ctx):
    req = ctx.req
    kept = []
    for skill in req["keep"]:
        if skill in ctx.plan["heal_data"]:
            kept += ctx.texts["skill"][skill]
    if len(kept) == 0:
        for x in req["add"]:
            kept += ctx.texts["skill"].get(x["skill"], [])
    general = list(ctx.texts["general"])
    for r in req["remove"]:
        if r["skill"] != "*" and r["skill"] in ctx.texts["skill"]:
            general += ctx.texts["skill"][r["skill"]]
    if len(kept) == 0:
        kept = list(ctx.texts["general"])
    return kept, general


def run_slim(ctx, stage):
    """Remove skills: keep experts (MoE) or drop layers and width (dense), then heal toward the unslimmed model."""
    from wsp import slim

    started = time.time()
    model, tok = ctx.model, ctx.tokenizer
    kept, general = _kept_removed_texts(ctx)
    target = stage["target"]
    # slim_moe and slim_dense copy the model, so the input is the unchanged teacher. Heal trains only LoRA and
    # routers, so a MoE student can share the experts with it.
    teacher = model
    if target["kind"] == "moe":
        small, rep = slim.slim_moe(model, tok, kept, general, rule=stage["settings"]["rule"], keep=target["keep"],
                                   seed=ctx.req["seed"], max_length=ctx.s["seq_len"], share_experts=True)
    else:
        small, rep = slim.slim_dense(model, tok, kept, general, target_params=target["target_params"],
                                     seed=ctx.req["seed"], max_length=ctx.s["seq_len"])
    heal_texts = kept + general
    steps = stage["settings"]["steps"]
    heal_report = None
    if steps and steps > 0:
        heal_report = slim.heal(small, teacher, tok, heal_texts, steps, batch_size=ctx.s["batch_size"],
                                seq_len=ctx.s["seq_len"], lora_rank=ctx.s["lora_rank"], seed=ctx.req["seed"],
                                log_every=0)
    teacher = None
    ctx.model = small
    path = os.path.join(ctx.out_dir, "stage_slim")
    manifest = slim.save_slim(small, path, tok)
    rep = {k: v for k, v in rep.items() if k not in ("ranking",)}
    return {"method": "slim", "status": "ok", "report": rep, "heal": heal_report, "manifest": manifest, "saved": path,
            "seconds": round(time.time() - started, 2)}


def _donor_config(model):
    from wsp import lineage

    return lineage.normalize_config(model.config.to_dict())


def run_synth(ctx, stage):
    """Staged synthesis: donor blocks go through translation layers into the host (wsp/synth.py)."""
    from wsp import synth

    started = time.time()
    s = ctx.s
    st = stage["settings"]
    comps = []
    donors = {}
    for item in stage["components"]:
        source = item["source"]
        if source not in donors:
            donors[source] = ctx.backend.load(source)
        if item.get("rows"):
            for row_id in item["rows"]:
                rows, _ = find_component_rows([row_id], None, ctx.atlas_entries)
                row = dict(rows[0])
                row["skill"] = item["skill"]
                comps.append(row)
            continue
        cfg = _donor_config(donors[source][0])
        layers, note = default_block(cfg, item["skill"], source, ctx.atlas_entries, item.get("n_layers", s["block_layers"]),
                                     ctx.req["reported"])
        comps.append({"kind": "block", "donor": source, "layers": layers, "skill": item["skill"],
                      "id": "%s_%s_L%d-%d" % (item["skill"], len(comps), layers[0], layers[1]), "note": note})
    texts = {"align": ctx.texts["align"], "general": ctx.texts["general"], "skills": ctx.texts["skill"]}
    options = {"stage1_steps": st["stage1_steps"], "stage2_steps": st["stage2_steps"], "lora_rank": st["lora_rank"],
               "batch_size": st["batch_size"], "seq_len": st["seq_len"], "seed": ctx.req["seed"],
               "controls": list(s["synth_controls"]), "test_limit": s.get("test_limit")}
    limits = ctx.req["limits"]
    if limits["max_total_params_b"] is not None:
        options["max_params"] = int(limits["max_total_params_b"] * 1e9)
    if limits["max_active_params_b"] is not None:
        options["max_params_per_token"] = int(limits["max_active_params_b"] * 1e9)
    for key, value in s.get("synth_options", {}).items():
        options[key] = value
    practice = dict(ctx.practice) or None
    model, tok, rep = synth.synthesize((ctx.model, ctx.tokenizer), comps, texts, None, practice, None, donors, **options)
    ctx.model = model
    path = os.path.join(ctx.out_dir, "stage_synth")
    base_id = ctx.req["base"] if ctx.stage_base_is_original else os.path.join(ctx.out_dir, "stage_base")
    synth.save_synth(model, tok, path, base_id, rep)
    rep = synth.json_safe(rep)
    return {"method": "synth", "status": "ok", "components": comps, "report": rep, "saved": path,
            "seconds": round(time.time() - started, 2)}


def run_transplant(ctx, stage):
    """Routed experts of a same-family MoE next to each MLP (wsp/transplant.py)."""
    from wsp import transplant

    started = time.time()
    s = ctx.s
    st = stage["settings"]
    source_id = stage["source"]
    source, source_tok = ctx.backend.load(source_id)
    model = ctx.model
    layer_map = transplant.make_layer_map(len(model.model.layers), len(source.model.layers),
                                          transplant.source_moe_layers(source.config))
    skill_texts = {skill: ctx.texts["skill"][skill] for skill in stage["skills"]}
    k = stage["components"][0].get("k", s["transplant_k"])
    rank = stage["components"][0].get("rank", s["transplant_rank"])
    expert_plan = transplant.select_experts(source, ctx.tokenizer, skill_texts, ctx.texts["general"], layer_map, k,
                                            seed=ctx.req["seed"], max_length=s["seq_len"])
    parts = transplant.parts_from_model(source, expert_plan)
    stitches = transplant.fit_stitches(model, source, ctx.tokenizer, ctx.texts["align"], layer_map, rank=rank,
                                       max_length=s["seq_len"], batch_size=s["batch_size"])
    anchor = copy.deepcopy(model)
    model = transplant.build_transplant(model, expert_plan, parts, stitches=stitches, lora_rank=st["lora_rank"],
                                        stitch_rank=rank, base_id=base_ref(ctx), source_id=source_id)
    groups = transplant.set_trainable(model)
    train = [{"params": groups["stitch"], "lr": 1e-3}, {"params": groups["router"], "lr": 1e-3},
             {"params": groups["gate"], "lr": 1e-2}, {"params": groups["lora"], "lr": 2e-4}]
    texts = []
    flags = []
    for skill in stage["skills"]:
        texts += ctx.texts["skill"][skill]
        flags += [False] * len(ctx.texts["skill"][skill])
    texts += ctx.texts["general"]
    flags += [True] * len(ctx.texts["general"])
    steps = st["stage1_steps"] + st["stage2_steps"]
    log = transplant.heal(model, ctx.tokenizer, texts, train, steps, s["batch_size"], s["seq_len"], ctx.req["seed"],
                          teacher=source, anchor=anchor, anchor_weight=0.5, anchor_flags=flags, log_every=0)
    del anchor
    ctx.model = model
    path = os.path.join(ctx.out_dir, "stage_transplant")
    transplant.save_transplant(model, path, ctx.tokenizer)
    return {"method": "transplant", "status": "ok", "k": k, "rank": rank, "heal": {key: (v[:3] if isinstance(v, list) else v)
            for key, v in log.items()}, "saved": path, "seconds": round(time.time() - started, 2)}


def run_feedback(ctx, stage):
    """Block output written into the next token's input (wsp/feedback.py). Needs a donor with the host's tokenizer."""
    from wsp import feedback

    started = time.time()
    s = ctx.s
    st = stage["settings"]
    branches = []
    donors = {}
    for item in stage["components"]:
        source = item["source"]
        for src in ctx.plan["sources"]:
            if src["source"] == source and src["skill"] == item["skill"]:
                if not src["flags"].get("teacher_usable_for_logit_kd", False):
                    raise ValueError("feedback distils from the donor on host-tokenizer text; %s does not share the "
                                     "host's tokenizer and added tokens" % source)
        if source not in donors:
            donors[source] = ctx.backend.load(source)
        cfg = _donor_config(donors[source][0])
        layers, note = default_block(cfg, item["skill"], source, ctx.atlas_entries, item.get("n_layers", s["block_layers"]),
                                     ctx.req["reported"])
        host_layers = len(ctx.model.model.layers)
        read_layer = max(0, min(host_layers - 1, int(round(layers[0] / float(cfg["layers"]) * host_layers))))
        t_in, sink, _ = feedback.fit_t_in(ctx.model, ctx.tokenizer, read_layer, donors[source][0], donors[source][1],
                                          layers[0], ctx.texts["skill"][item["skill"]], steps=st["stage1_steps"],
                                          batch_size=s["batch_size"], seq_len=s["seq_len"], seed=ctx.req["seed"])
        branch = feedback.make_block_branch(ctx.model, donors[source][0], list(layers), read_layer, t_in=t_in, sink=sink,
                                            donor_id=source)
        branches.append(branch)
    ref = base_ref(ctx)
    seam = feedback.build_seam(ctx.model, branches, base_id=ref)
    # Stage 2 trains T_out, the gates, and a host LoRA on the layers above every read layer. Without the LoRA a
    # one-pass step has no trainable parameter in its graph and backward fails.
    above = list(range(max(b.read_layer for b in branches) + 1, len(ctx.model.model.layers)))
    if above:
        seam.add_host_lora(s["lora_rank"], layers=above)
    texts = []
    flags = []
    for item in stage["components"]:
        texts += ctx.texts["skill"][item["skill"]]
        flags += [False] * len(ctx.texts["skill"][item["skill"]])
    texts += ctx.texts["general"]
    flags += [True] * len(ctx.texts["general"])
    teacher = donors[stage["components"][0]["source"]][0]
    log = feedback.heal_seam(seam, ctx.tokenizer, teacher, texts, flags, steps=st["stage1_steps"] + st["stage2_steps"],
                             batch_size=s["batch_size"], seq_len=s["seq_len"], log_every=0)
    ctx.model = seam
    path = os.path.join(ctx.out_dir, "stage_feedback")
    feedback.save_seam(seam, path, tokenizer=ctx.tokenizer, base_id=ref)
    return {"method": "feedback", "status": "ok", "saved": path, "heal": str(type(log)), "seconds": round(time.time() - started, 2)}


def run_delta(ctx, stage):
    """Same-shape specialists as switchable low-rank differences (wsp/delta.py)."""
    from wsp import delta

    started = time.time()
    st = stage["settings"]
    deltas = []
    for item in stage["components"]:
        specialist, _ = ctx.backend.load(item["source"])
        deltas.append(delta.mine_delta(ctx.model, specialist, rank=st["rank"], name=item["skill"]))
    network = delta.SkillDeltaNetwork(ctx.model, deltas=deltas, tokenizer=ctx.tokenizer, host_id=base_ref(ctx))
    by_expert = {"base": ctx.texts["general"]}
    for item in stage["components"]:
        by_expert[item["skill"]] = ctx.texts["skill"][item["skill"]]
    network.fit_router(by_expert)
    ctx.model = network
    path = os.path.join(ctx.out_dir, "stage_delta")
    network.save(path)
    return {"method": "delta", "status": "ok", "saved": path, "seconds": round(time.time() - started, 2)}


RUNNERS = {"weave": run_weave, "slim": run_slim, "synth": run_synth, "transplant": run_transplant,
           "feedback": run_feedback, "delta": run_delta}


# ---------------------------------------------------------------- measurement


def measured_sizes(model):
    """Return {"total", "per_token"} measured on the built network."""
    from wsp import skills_eval

    try:
        from wsp import transplant

        if len(transplant.banks(model)) > 0:
            counts = transplant.count_params(model)
            return {"total": int(counts["total"]), "per_token": int(counts["active"]), "kind": "transplant"}
    except Exception:
        pass
    if len(getattr(model, "wsp_grafts", [])) > 0:
        from wsp import synth

        counts = synth.param_counts(model)
        return {"total": int(counts["total"]), "per_token": int(counts["per_token"]), "kind": "synth"}
    if hasattr(model, "count_params") and not hasattr(model, "config_class"):
        counts = model.count_params()
        return {"total": int(counts.get("total", counts.get("stored"))), "per_token": int(counts.get("per_token", counts.get("active"))),
                "kind": "seam"}
    try:
        from wsp import slim

        total, run = slim.count_params(model)
    except Exception:
        total, run = skills_eval.model_params(model)
    return {"total": int(total), "per_token": int(run), "kind": "plain"}


def probe_view(model):
    """The model as the probes read it: a module with parameters() and .logits output.

    A SkillDeltaNetwork is not a module, so it gets a thin module view that routes each sequence.
    """
    import torch

    if isinstance(model, torch.nn.Module):
        return model
    if not (hasattr(model, "route_many") and hasattr(model, "model")):
        return model

    class DeltaView(torch.nn.Module):
        def __init__(self, network):
            super().__init__()
            self.host = network.model
            self.network_ref = [network]
            self.config = network.model.config

        def forward(self, input_ids=None, attention_mask=None, **kwargs):
            from types import SimpleNamespace

            return SimpleNamespace(logits=self.network_ref[0].forward(input_ids, attention_mask))

        def generate(self, **kwargs):
            return self.network_ref[0].generate(**kwargs)

    return DeltaView(model)


def _kl_row(model, cache, batch):
    from wsp import weave

    return weave.kl_to_cache(probe_view(model), cache, batch)["kl"]


def prepare(ctx):
    """Texts, teacher caches, and baselines measured on the untouched base, before any stage."""
    from wsp import probe, weave

    s = ctx.s
    plan_ = ctx.plan
    tok = ctx.tokenizer
    skills = [k for k in plan_["heal_data"] if not k.startswith("_")]
    ctx.texts["skill"] = {skill: ctx.pool_texts(skill) for skill in skills}
    ctx.texts["general"] = ctx.pool_texts("_general")
    ctx.texts["align"] = ctx.pool_texts("_align") + ctx.texts["general"][:100]
    splits_needed = "abc" if any(st["method"] == "weave" for st in plan_["stages"]) else "c"
    general_probe = {}
    for split in "abc":
        general_probe[split] = ctx.pool_texts("_general", split)
    ctx.texts["probe_general"] = general_probe
    ctx.texts["probe"] = {}
    for skill in skills:
        ctx.texts["probe"][skill] = {split: ctx.pool_texts(skill, split) for split in "abc"}
    for split in splits_needed:
        cache = weave.build_teacher_cache(ctx.model, tok, general_probe[split], top_k=s["probe_top_k"],
                                          max_length=s["probe_length"], batch_size=s["batch_size"])
        ctx.caches[("general", split)] = cache
    for src in plan_["sources"]:
        if not src["flags"].get("tokenizer_same", True):
            continue
        split_texts = [ctx.texts["probe"][src["skill"]][split] for split in splits_needed]
        built = ctx.backend.teacher_caches(src["source"], split_texts, tok, s["probe_top_k"], s["probe_length"],
                                           s["batch_size"], None)
        for split, cache in zip(splits_needed, built):
            ctx.caches[("donor", src["source"], src["skill"], split)] = cache
            if split == "c":
                ctx.baseline[(src["source"], src["skill"])] = _kl_row(ctx.model, cache, s["batch_size"])
    ctx.baseline["nll"] = {}
    for skill in skills:
        ctx.baseline["nll"][skill] = probe.text_nll(probe_view(ctx.model), tok, ctx.texts["probe"][skill]["c"], s["probe_length"],
                                                    s["batch_size"])["nll"]
    ctx.baseline["nll"]["_general"] = probe.text_nll(probe_view(ctx.model), tok, general_probe["c"], s["probe_length"],
                                                     s["batch_size"])["nll"]
    for skill in skills:
        info = plan_["practice"].get(skill)
        if info and info["available"]:
            items = ctx.backend.practice_items(skill, s["practice_items"], ctx.reported, ctx.req["seed"])
            if len(items) > 0:
                ctx.practice[skill] = practice_scorer(ctx.backend, items)
    for skill, fn in ctx.practice.items():
        ctx.base_practice[skill] = fn(ctx.model, tok)


def verify(ctx, stage_reports):
    """The final probe report: transfer per added skill, drift, content_z, practice, sizes, guards."""
    from wsp import probe, weave

    s = ctx.s
    model = ctx.model
    tok = ctx.tokenizer
    out = {"skills": {}, "drift": None, "verdict": {}}
    weave_report = None
    for rep in stage_reports:
        if rep["method"] == "weave" and rep.get("status") == "ok":
            weave_report = rep
    member_names = {}
    for plan_stage in ctx.plan["stages"]:
        if plan_stage["method"] == "weave":
            for index, member in enumerate(plan_stage["members"]):
                member_names.setdefault((member["source"], member["skill"]), []).append("m%d_%s" % (index, member["skill"]))
    for src in ctx.plan["sources"]:
        skill = src["skill"]
        key = (src["source"], skill)
        row = {"source": src["source"], "method": src["chosen"]}
        if key not in ctx.baseline:
            row.update(transfer=None, note="no shared tokenizer with the source: transfer by KL cannot be measured")
            out["skills"][skill + ":" + src["source"]] = row
            out["verdict"][skill + ":" + src["source"]] = "not measured"
            continue
        cache = ctx.caches[("donor", src["source"], skill, "c")]
        kl = _kl_row(model, cache, s["batch_size"])
        row["donor_kl_base"] = ctx.baseline[key]
        row["donor_kl_final"] = kl
        row["transfer"] = probe.gap_closed(ctx.baseline[key], kl, 0.0)
        if weave_report is not None:
            for name in member_names.get(key, []):
                if name in weave_report["controls"] and name in weave_report["members_used"]:
                    real = weave_report["transfer_c"][name]
                    stats = probe.content_z(real, weave_report["controls"][name])
                    if stats is not None:
                        row["content_z"] = stats
                        row["content_controls"] = weave_report["controls"][name]
                        row["content_measured_at"] = "weave stage, split c"
        row["verdict_word"] = probe.verdict(row, min_transfer=0.05, min_gain=0.03, min_z=2.0)
        out["skills"][skill + ":" + src["source"]] = row
        out["verdict"][skill + ":" + src["source"]] = row["verdict_word"]
    drift_kl = _kl_row(model, ctx.caches[("general", "c")], s["batch_size"])
    out["drift"] = {"base_kl": drift_kl, "tolerance": s["drift_tolerance"]}
    nll_after = {}
    for skill, texts in ctx.texts["probe"].items():
        nll_after[skill] = probe.text_nll(probe_view(model), tok, texts["c"], s["probe_length"], s["batch_size"])["nll"]
    nll_after["_general"] = probe.text_nll(probe_view(model), tok, ctx.texts["probe_general"]["c"], s["probe_length"],
                                           s["batch_size"])["nll"]
    out["nll"] = {"base": ctx.baseline["nll"], "final": nll_after,
                  "delta": {k: nll_after[k] - ctx.baseline["nll"][k] for k in nll_after}}
    practice = {}
    for skill, fn in ctx.practice.items():
        practice[skill] = {"base": ctx.base_practice[skill], "final": fn(model, tok)}
    out["practice"] = practice
    return out


def verdicts(ctx, probe_report, sizes, seconds, stage_reports):
    """One verdict per component and one for the request: met, partly met, or not met."""
    s = ctx.s
    comps = {}
    drift_ok = probe_report["drift"]["base_kl"] <= s["drift_tolerance"]
    for label, row in probe_report["skills"].items():
        reasons = []
        transfer = row.get("transfer")
        word = row.get("verdict_word")
        if transfer is None:
            comps[label] = {"verdict": "partly met", "reasons": [row.get("note", "transfer could not be measured")]}
            continue
        if word in ("harmful", "no transfer"):
            comps[label] = {"verdict": "not met", "reasons": ["probe verdict: %s (transfer %.3f)" % (word, transfer)]}
            continue
        if word == "no gap":
            comps[label] = {"verdict": "partly met", "reasons": ["the source is no better than the base on this text"]}
            continue
        if transfer < s["min_transfer"]:
            reasons.append("transfer %.3f is below %.2f" % (transfer, s["min_transfer"]))
        if word == "no better than random":
            reasons.append("not above the random controls")
        if "content_z" not in row and row.get("method") == "weave":
            reasons.append("content_z not measured")
        if not drift_ok:
            reasons.append("drift %.4f is above %.2f" % (probe_report["drift"]["base_kl"], s["drift_tolerance"]))
        comps[label] = {"verdict": "met" if not reasons else "partly met", "reasons": reasons or ["transfer %.3f" % transfer]}
    for rep in stage_reports:
        if rep["method"] == "synth" and rep.get("report"):
            for c in rep["report"].get("components", []):
                label = [k for k in comps if k.startswith(str(c.get("skill")) + ":")]
                if label and c.get("verdict") in ("harmful",):
                    comps[label[0]] = {"verdict": "not met", "reasons": ["synth component verdict: harmful"]}
                elif label and c.get("verdict") not in (None, "contributes", "not tested") and comps[label[0]]["verdict"] == "met":
                    comps[label[0]] = {"verdict": "partly met", "reasons": ["synth component verdict: %s" % c["verdict"]]}
    for r in ctx.req["remove"]:
        delta = probe_report["nll"]["delta"]
        kept = [d for k, d in delta.items() if k in ctx.req["keep"]]
        if r["skill"] == "*":
            added = [x["skill"] for x in ctx.req["add"]]
            gone = [d for k, d in delta.items() if not k.startswith("_") and k not in ctx.req["keep"] and k not in added]
        else:
            gone = [d for k, d in delta.items() if k == r["skill"]]
        reasons = []
        kept_mean = sum(kept) / len(kept) if kept else None
        gone_mean = sum(gone) / len(gone) if gone else None
        if kept_mean is not None and kept_mean > 0.1:
            reasons.append("kept skill loss rose by %.3f nats" % kept_mean)
        if gone_mean is None:
            reasons.append("no removed-skill text was measured (general text loss changed by %.3f nats)"
                           % delta.get("_general", 0.0))
        elif kept_mean is not None and gone_mean < kept_mean + 0.05:
            reasons.append("the removed skills did not lose more than the kept ones")
        comps["remove:" + r["skill"]] = {"verdict": "met" if not reasons else "partly met", "reasons": reasons or ["selective"]}
    limit_problems = check_limits(sizes["total"], sizes["per_token"], ctx.req["limits"])
    minutes = ctx.req["limits"]["max_heal_minutes"]
    if minutes is not None and seconds > minutes * 60.0:
        limit_problems.append("compile took %.1f minutes, above max_heal_minutes %.1f" % (seconds / 60.0, minutes))
    guard_problems = []
    for skill in ctx.req["keep"]:
        info = probe_report["practice"].get(skill)
        if info and info["final"] < info["base"] - 0.03:
            guard_problems.append("kept skill %s fell from %.3f to %.3f on its practice set" % (skill, info["base"], info["final"]))
    values = [c["verdict"] for c in comps.values()]
    reasons = list(limit_problems) + guard_problems
    if len(values) > 0 and all(v == "not met" for v in values):
        whole = "not met"
        reasons.append("no component met its test")
    elif limit_problems or guard_problems or any(v != "met" for v in values):
        whole = "not met" if limit_problems else "partly met"
        for label, c in comps.items():
            if c["verdict"] != "met":
                reasons.append("%s: %s (%s)" % (label, c["verdict"], "; ".join(c["reasons"])))
    else:
        whole = "met"
    if any(rep.get("status") == "no_admissible_member" for rep in stage_reports):
        whole = "not met"
        reasons.append("weave found no member that passed the gate")
    return {"components": comps, "request": {"verdict": whole, "reasons": reasons}}


class MakeResult:
    """What make() returns: the report, plus the built model for scoring."""

    def __init__(self, report, model, tokenizer, out_dir):
        self.report = report
        self.model = model
        self.tokenizer = tokenizer
        self.out_dir = out_dir


def make(request, out_dir, backend, lineage=None, plan_=None, settings=None, atlas_entries=None):
    """Run a request through a backend and write out_dir/sapien_report.json. Returns a MakeResult.

    plan_: a plan already made (else plan() runs with lineage). settings:
    overrides for the plan's settings (for example tiny sizes in a test).
    Raises PlanError when the plan is not feasible.
    """
    started = time.time()
    req = validate_request(request)
    if atlas_entries is None:
        atlas_entries = load_atlas()
    if plan_ is None:
        plan_ = plan(req, lineage=lineage, settings=settings, atlas_entries=atlas_entries)
    elif settings:
        plan_ = copy.deepcopy(plan_)
        plan_["settings"].update(settings)
    if not plan_["feasible"]:
        raise PlanError("the plan cannot run: " + " | ".join(plan_["refusals"] or ["no stage"]))
    os.makedirs(out_dir, exist_ok=True)
    if [st["method"] for st in plan_["stages"]] == ["pgraft"]:
        # The write-in has its own build, save, reload, and score path (wsp/pgraft_stage.py): the text probes of
        # prepare() and verify() read same-tokenizer KL and do not apply to a four-option write-in.
        from wsp import pgraft_stage

        final, model, tokenizer = pgraft_stage.run_make(req, plan_, backend, out_dir, started, atlas_entries)
        return MakeResult(final, model, tokenizer, out_dir)
    model, tokenizer = backend.load(req["base"])
    ctx = Context(req, plan_, backend, out_dir, model, tokenizer, atlas_entries)
    ctx.stage_base_is_original = True
    prepare(ctx)
    sizes_before = measured_sizes(model)
    model = None
    stage_reports = []
    for stage in plan_["stages"]:
        ctx.stage_base_is_original = len(stage_reports) == 0
        if stage["kind"] == "seam" and len(stage_reports) > 0:
            base_dir = os.path.join(out_dir, "stage_base")
            backend.save_plain(ctx.model, ctx.tokenizer, base_dir)
        report = RUNNERS[stage["method"]](ctx, stage)
        stage_reports.append(report)
    report_probe = verify(ctx, stage_reports)
    sizes = measured_sizes(ctx.model)
    seconds = time.time() - started
    verdict_block = verdicts(ctx, report_probe, sizes, seconds, stage_reports)
    final = {
        "name": req["name"], "request": req, "plan": plan_, "stages": stage_reports, "probe": report_probe,
        "sizes": {"base": sizes_before, "final": sizes,
                  "limits": req["limits"], "limit_problems": check_limits(sizes["total"], sizes["per_token"], req["limits"])},
        "seconds": round(seconds, 2), "verdicts": verdict_block["components"], "verdict": verdict_block["request"],
        "scored_on_reported": False,
    }
    with open(os.path.join(out_dir, "sapien_report.json"), "w") as handle:
        json.dump(_json_safe(final), handle, indent=1)
    return MakeResult(final, ctx.model, ctx.tokenizer, out_dir)


def _json_safe(value):
    try:
        import torch
    except ImportError:
        torch = None
    if isinstance(value, dict):
        return {str(k) if not isinstance(k, tuple) else "|".join(str(x) for x in k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if torch is not None and isinstance(value, torch.Tensor):
        return value.tolist()
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    if isinstance(value, (int, float, str, bool)) or value is None:
        return value
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return str(value)


# ---------------------------------------------------------------- comparison arms and the reported scores


REPORTED_SKILLS = {"gsm8k": "gsm8k", "math": "math500", "mbpp": "mbpp", "humaneval": "humaneval",
                   "medmcqa": "medmcqa", "mmlu": "mmlu"}


def reported_skills(reported):
    """Map reported dataset ids to the skill names of wsp.skills_eval. Returns (runnable skills, ids that cannot run)."""
    skills = []
    other = []
    for name in reported:
        family = healdata._family(str(name).lower().strip())
        skill = REPORTED_SKILLS.get(family)
        if skill is None:
            other.append(name)
        elif skill not in skills:
            skills.append(skill)
    return skills, other


def lora_heal(model, tokenizer, texts, steps, rank=16, teacher=None, batch_size=4, seq_len=256, seed=0, lr=2e-4):
    """Heal a LoRA on the model with the LM loss, plus KL to a teacher when one is given. Merges at the end."""
    import random

    import torch
    import torch.nn.functional as F

    from wsp.lora import apply_lora, merge_lora

    params = apply_lora(model, rank)
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    device = next(model.parameters()).device
    order = list(range(len(texts)))
    random.Random(seed).shuffle(order)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    model.train()
    losses = []
    for step in range(steps):
        pick = [texts[order[(step * batch_size + j) % len(order)]] for j in range(batch_size)]
        enc = tokenizer(pick, return_tensors="pt", padding=True, truncation=True, max_length=seq_len).to(device)
        logits = model(**enc).logits[:, :-1].float()
        target = enc["input_ids"][:, 1:]
        mask = enc["attention_mask"][:, 1:].float()
        lm = (F.cross_entropy(logits.transpose(1, 2), target, reduction="none") * mask).sum() / mask.sum().clamp(min=1)
        loss = lm
        if teacher is not None:
            with torch.no_grad():
                ref = teacher(**enc).logits[:, :-1].float()
            kl = (F.kl_div(F.log_softmax(logits, -1), F.log_softmax(ref, -1), log_target=True, reduction="none").sum(-1) * mask).sum() / mask.sum().clamp(min=1)
            loss = lm + kl
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(float(loss))
    model.eval()
    merge_lora(model)
    return losses


def build_arms(request, plan_, backend, result_loader=None, texts=None):
    """Return {arm label: function returning (model, tokenizer)} for the comparison set. Each loads when called.

    texts: heal texts for the LoRA arms (default: the plan's pools through the backend). The steps match the
    seam stage's stage1 + stage2 steps, or the plan default.
    """
    req = validate_request(request)
    hd = hd_reported(req["reported"])
    arms = {}
    steps = plan_["settings"]["stage1_steps"] + plan_["settings"]["stage2_steps"]
    for stage in plan_["stages"]:
        if stage["kind"] == "seam" and "stage1_steps" in stage["settings"]:
            steps = stage["settings"]["stage1_steps"] + stage["settings"]["stage2_steps"]

    def heal_texts(tok):
        if texts is not None:
            return texts
        out = []
        for key, entry in plan_["heal_data"].items():
            if key == "_align":
                continue
            out += backend.pool(entry["pool"], entry["count"], tok, entry["seed"], entry["skip"], hd)
        return out

    def base():
        return backend.load(req["base"])

    def base_lora():
        model, tok = backend.load(req["base"])
        lora_heal(model, tok, heal_texts(tok), steps, plan_["settings"]["lora_rank"], None, plan_["settings"]["batch_size"],
                  plan_["settings"]["seq_len"], req["seed"])
        return model, tok

    arms["base"] = base
    arms["base_lora"] = base_lora
    for arm in plan_["arms"]:
        if arm["arm"] == "base_distill" and arm.get("teacher"):
            def base_distill(teacher_id=arm["teacher"]):
                model, tok = backend.load(req["base"])
                teacher, _ = backend.load(teacher_id)
                lora_heal(model, tok, heal_texts(tok), steps, plan_["settings"]["lora_rank"], teacher,
                          plan_["settings"]["batch_size"], plan_["settings"]["seq_len"], req["seed"])
                return model, tok

            arms["base_distill"] = base_distill
        if arm["arm"] == "best_existing" and arm.get("model"):
            arms["best_existing"] = (lambda m=arm["model"]: backend.load(m))
    if result_loader is not None:
        arms["result"] = result_loader
    return arms


def score_reported(request, arms, evaluate=None, limit=None, batch_size=32, keep_going=False):
    """Score the result and the comparison arms on the reported benchmarks. Call once.

    arms: {label: (model, tokenizer) or a function returning one}. Each arm loads, scores, and is freed before the
    next. evaluate(model, tokenizer, skills, limit) -> {skill: score} defaults to wsp.skills_eval.evaluate_suite.
    With keep_going, an arm that raises gets {"error": text} and the others still run.
    Returns {"skills": [...], "not_runnable": [...], "scores": {label: {skill: score}}}.
    """
    req = validate_request(request)
    skills, other = reported_skills(req["reported"])
    if evaluate is None:
        from wsp import skills_eval

        def evaluate(model, tokenizer, skills, limit):
            return skills_eval.evaluate_suite(model, tokenizer, skills=skills, limit=limit, batch_size=batch_size,
                                              log=False)["scores"]
    scores = {}
    for label, arm in arms.items():
        try:
            pair = arm() if callable(arm) else arm
            model, tokenizer = pair
            scores[label] = dict(evaluate(model, tokenizer, skills, limit))
            del model, pair
        except Exception as error:
            if not keep_going:
                raise
            scores[label] = {"error": repr(error)[:300]}
    return {"skills": skills, "not_runnable": other, "scores": scores}
