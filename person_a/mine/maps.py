# Model-level skill maps for the library: where each skill lives in a model, by layer, by module,
# and by expert. For one model, the GPU engine skips each decoder layer in turn and scores every
# skill set, and for MoE layers it counts how often the router picks each expert on each skill's
# text. No agent, no LLM.
#   .venv/bin/python mine/maps.py --zoo-key Qwen__Qwen2.5-7B-Instruct          # a model of atlas/zoo.json
#   .venv/bin/python mine/maps.py --short lfm2_24b_a2b                         # a row of registry/donors.csv
#   .venv/bin/python mine/maps.py --zoo-key Qwen__Qwen2.5-7B-Instruct --modules
# Output: atlas/pool_entries/<org>__<name>.json, in the atlas entry format (layer_maps, expert_maps)
# with the measured numbers kept, so a planner can pick a block or an expert set for any skill.
#
# Layer score (layer_maps[skill].scores): the loss increase on skill text when one layer is skipped,
# minus the same on general_text, in nats, on the search half.
# Block z-score (layer_maps[skill].block_z, block_z_heldout): the positive score mass of a block
# against the positive mass of every other window of the same length in the same model, in sample
# standard deviations. Lists aligned with blocks. block_z uses the search scores, which also chose
# the block; block_z_heldout uses the held-out scores for the same block.
# How to read it: block_z is not a test against independent random blocks. Most of the other
# windows share layers with the block (block_z_windows and block_z_disjoint_windows give the
# counts). The first block is the window with the most mass on the search scores, so its block_z
# is positive or null by construction; only the held-out values check it. A short block is
# compared with windows that mostly hold zero mass, so the deviation is small and z is large.
# block_z_disjoint and block_z_disjoint_heldout use only the windows that share no layer with
# the block (null with fewer than 3 of them).
#
# --modules also measures each layer with only its attention output zeroed and with only its
# feed-forward output zeroed (dense MLP, routed experts, or both). The entry then has
#   module_maps[skill].attn, .mlp     the layer score definition, per module (null: no such module)
#   module_maps[skill].module_share   per layer, the attention share of the positive attn + mlp
#                                     score; null where the layer lacks one of the two modules
#   module_maps[skill].heldout.split_half_r.attn, .mlp   Pearson r of the search and held-out lists
#   module_maps[skill].heldout.n_layers.attn, .mlp       the layers that r is over (null r below 4)
#   module_parts                      per layer, the part kinds that were zeroed for each module
#   module_engine_seconds, module_calls, module_measured_at
# Cost. A plain run takes one scoring pass for the base and one per layer (every skill set scored
# once per pass), in one engine call. --modules adds one pass per layer for the attention module
# and one for the feed-forward module, where the layer has them: up to three passes per layer.
# How a --modules run works. An engine call that reaches its timeout returns nothing, so the
# module passes do not go into the plain call:
#   1. The layer and expert maps come from the entry that is there, when it is for the same model
#      revision and skill text. Without such an entry the plain call runs first and its entry is
#      written before any module call.
#   2. The module passes go to the engine in calls of a few layers each (at most 1 + 2 * layers
#      passes per call). The layers per call come from the time per pass of the plain run, so the
#      estimate for a call stays within half of the 3600 s timeout; --layers-per-call sets it.
#   3. After each call the merged result is saved in atlas/pool_logs/<org>__<name>.modules-partial.json.
#      A run that stops starts again after the saved layers. The file is removed with the entry write.
# --modules needs an engine deployed with skill_maps(modules=, layers=, plain=). With an older
# engine the run prints that and exits with code 3; a plain run sends the same call as before.
# A plain run never replaces an entry that has module maps, unless --force is set.
#
# Data: skill text comes from mine/data/pool_skills.json and the built-in math, code and
# general_text sets. Mining may use any dataset, and some pool skills come from a benchmark (the
# MMLU domain skills). selection_sources says, per skill, the dataset, the wsp/healdata.py pool that
# draws on the same dataset, and whether the dataset is on the default reported list (BANNED), so
# compile and heal can keep away from data they are scored on. reported_benchmark_skills lists the
# skills where that is true. benchmark_free is written as true for every entry, as before, and
# does not look at the skills: read reported_benchmark_skills.
# Selection uses the "search" half of each set; the "heldout" half only checks it.
import argparse
import datetime
import hashlib
import importlib.util
import json
import math
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common  # noqa: E402

REPO = os.path.dirname(common.ROOT)
OUT_DIR = os.path.join(REPO, "atlas", "pool_entries")
ZOO = os.path.join(REPO, "atlas", "zoo.json")
REFERENCE = "general_text"
N_SEARCH = 48
N_HELDOUT = 24
TOP_EXPERTS = 8
EPS = 1.0e-4
MODULES = ["attn", "mlp"]
MIN_OTHER_WINDOWS = 3
# A module's held-out r needs this many layers that have the module. With 2 layers r is always 1 or -1.
MIN_R_LAYERS = 4
# A score is built from four log-probs that the engine rounds to 5 places (two base scores, two
# ablated scores), so rounding alone can move it by 4 * 0.5e-5. A score at or below this is not
# told apart from 0 in module_share.
SCORE_RESOLUTION = 2.0e-5
# The "timeout" of the single-GPU engine classes in engine_app.py (COMMON), in seconds. A call that
# reaches it returns nothing. The multi-GPU classes have more time; this file plans for the smaller.
CALL_TIMEOUT = 3600
# The share of CALL_TIMEOUT that one planned module call may fill, by the estimate from the plain run.
CALL_BUDGET = 0.5
# Exit code of a --modules run that found an engine without skill_maps(modules=, layers=, plain=).
# maps_many.py stops when it sees it, because every other model would fail the same way.
OLD_ENGINE_EXIT = 3
OLD_ENGINE_MESSAGE = ("the deployed engine has no skill_maps(modules=..., layers=..., plain=...): deploy the "
                      "engine from a branch that has it, then run --modules again")
LOG_DIR = os.path.join(REPO, "atlas", "pool_logs")
HEALDATA = os.path.join(REPO, "wsp", "healdata.py")
# Datasets of the built-in sets, read from mine/build_data.py (builtin_skills.json has no dataset field).
BUILTIN_DATASETS = {
    "math": "microsoft/orca-math-word-problems-200k train",
    "code": "sahil2801/CodeAlpaca-20k train",
    REFERENCE: "Salesforce/wikitext wikitext-2-raw-v1 test",
}


def load_skills():
    # every skill as inline items, so one payload holds all sets and nothing depends on the engine image
    with open(os.path.join(HERE, "data", "pool_skills.json")) as f:
        pool = json.load(f)
    with open(common.BUILTIN_SKILLS) as f:
        builtin = json.load(f)
    skills = {}
    sources = {}
    for name in ["math", "code", REFERENCE]:
        items = builtin[name]["items"]
        used = {"search": items["search"][:N_SEARCH], "heldout": items["heldout"][:N_HELDOUT]}
        # The hash names the items: the engine keeps tokens and base scores by this hash while a
        # container lives, and the built-in file can be built again in that time.
        digest = hashlib.sha256(json.dumps(used, sort_keys=True).encode()).hexdigest()[:12]
        skills[name] = {"hash": "maps-builtin-" + name + "-" + digest, "control": None, "items": used}
        sources[name] = builtin[name]["description"]
    for name in pool:
        skills[name] = {"hash": "maps-" + pool[name]["hash"], "control": None, "items": pool[name]["items"]}
        sources[name] = pool[name]["dataset"] + ": " + pool[name]["description"]
    return {"skills": skills}, sources


def load_healdata(path=HEALDATA):
    # Loaded by path: importing the wsp package would run its __init__ and need the compiler's
    # dependencies. Returns None when the file is not there (person_a/ used outside the team repo).
    if not os.path.exists(path):
        return None
    spec = importlib.util.spec_from_file_location("wsp_healdata_by_path", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def dataset_id(dataset):
    # "cais/mmlu all test (subjects: ...)" -> "cais/mmlu"
    words = (dataset or "").split()
    if not words:
        return None
    return words[0].lower()


def source_record(dataset, description, healdata):
    # pool: the first healdata.SOURCES pool with a source from the same dataset (any config or split).
    # reported_benchmark: the dataset id is in healdata.BANNED, the default reported list.
    # Both are None when healdata could not be loaded, which means "not known", not "no".
    ident = dataset_id(dataset)
    pool = None
    reported = None
    if healdata is not None:
        reported = False
        if ident is not None:
            for name in healdata.SOURCES:
                if any(src[0].lower() == ident for src in healdata.SOURCES[name]):
                    pool = name
                    break
            reported = ident in [b.lower() for b in healdata.BANNED]
    return {"dataset": dataset, "description": description, "pool": pool, "reported_benchmark": reported}


def load_selection_sources(healdata):
    # one record per skill set of load_skills(), general_text included
    with open(os.path.join(HERE, "data", "pool_skills.json")) as f:
        pool = json.load(f)
    with open(common.BUILTIN_SKILLS) as f:
        builtin = json.load(f)
    out = {}
    for name in ["math", "code", REFERENCE]:
        out[name] = source_record(BUILTIN_DATASETS[name], builtin[name]["description"], healdata)
    for name in pool:
        out[name] = source_record(pool[name]["dataset"], pool[name]["description"], healdata)
    return out


def best_window(scores, length):
    best_start = 0
    best_mass = None
    for start in range(0, len(scores) - length + 1):
        mass = sum(max(0.0, v) for v in scores[start:start + length])
        if best_mass is None or mass > best_mass:
            best_start = start
            best_mass = mass
    return best_start, best_mass


def suggest_blocks(scores, mass=0.6, count=3):
    # Same rule as wsp/atlas.py suggest_blocks: the shortest window that holds `mass` of the positive
    # score, then up to count-1 more windows of that length, best first, without overlap.
    n = len(scores)
    total = sum(max(0.0, v) for v in scores)
    if n == 0 or total <= 0:
        return [], []
    length = n
    for size in range(1, n + 1):
        start, window_mass = best_window(scores, size)
        if window_mass >= mass * total - 1e-12:
            length = size
            break
    blocks = []
    fractions = []
    taken = [False] * n
    while len(blocks) < count:
        best_start = None
        best_mass = 0.0
        for start in range(0, n - length + 1):
            if any(taken[start:start + length]):
                continue
            window_mass = sum(max(0.0, v) for v in scores[start:start + length])
            if window_mass > best_mass:
                best_start = start
                best_mass = window_mass
        if best_start is None:
            break
        for i in range(best_start, best_start + length):
            taken[i] = True
        blocks.append([best_start, best_start + length - 1])
        fractions.append(round(best_mass / total, 4))
    return blocks, fractions


def positive_share(scores, blocks):
    total = sum(max(0.0, v) for v in scores)
    out = []
    for first, last in blocks:
        mass = sum(max(0.0, v) for v in scores[first:last + 1])
        out.append(0.0 if total <= 0 else round(mass / total, 4))
    return out


def window_masses(scores, first, last):
    # The positive score mass of the block, of every other window of the same length, and of the
    # other windows that share no layer with the block.
    length = last - first + 1
    own = sum(max(0.0, v) for v in scores[first:last + 1])
    others = []
    disjoint = []
    for start in range(0, len(scores) - length + 1):
        if start == first:
            continue
        mass = sum(max(0.0, v) for v in scores[start:start + length])
        others.append(mass)
        if start > last or start + length - 1 < first:
            disjoint.append(mass)
    return own, others, disjoint


def z_against(own, others):
    # None with fewer than MIN_OTHER_WINDOWS windows, or when they all have the same mass
    if len(others) < MIN_OTHER_WINDOWS:
        return None
    mean = sum(others) / len(others)
    sd = math.sqrt(sum((v - mean) ** 2 for v in others) / (len(others) - 1))
    if sd <= 1.0e-9:
        return None
    return round((own - mean) / sd, 3)


def window_z(scores, first, last):
    # How far the block's positive score mass is above the positive mass of every other window of
    # the same length in the same model, in sample standard deviations of those other windows.
    # The other windows may overlap the block, so they are not independent of it.
    own, others, disjoint = window_masses(scores, first, last)
    return z_against(own, others)


def window_z_disjoint(scores, first, last):
    # The same against only the windows that share no layer with the block
    own, others, disjoint = window_masses(scores, first, last)
    return z_against(own, disjoint)


def block_z(scores, blocks):
    return [window_z(scores, first, last) for first, last in blocks]


def block_z_disjoint(scores, blocks):
    return [window_z_disjoint(scores, first, last) for first, last in blocks]


def block_window_counts(scores, blocks):
    # per block: how many other windows block_z compares with, and how many of them do not overlap it
    n_others = []
    n_disjoint = []
    for first, last in blocks:
        own, others, disjoint = window_masses(scores, first, last)
        n_others.append(len(others))
        n_disjoint.append(len(disjoint))
    return n_others, n_disjoint


def pearson(a, b):
    n = len(a)
    if n < 2:
        return None
    ma = sum(a) / n
    mb = sum(b) / n
    cov = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    va = sum((x - ma) ** 2 for x in a)
    vb = sum((y - mb) ** 2 for y in b)
    if va <= 0 or vb <= 0:
        return None
    return round(cov / math.sqrt(va * vb), 4)


def loss_increase(res, skill, split):
    # nats of answer log-prob lost when each layer is skipped (positive = the layer matters)
    base = res["base"][skill + "|" + split]["lp"]
    out = []
    for lp in res["layer_lp"][skill + "|" + split]:
        out.append(None if base is None or lp is None else base - lp)
    return out


def layer_map(res, skill):
    maps = {}
    for split in ["search", "heldout"]:
        own = loss_increase(res, skill, split)
        ref = loss_increase(res, REFERENCE, split)
        if any(v is None for v in own) or any(v is None for v in ref):
            return None
        maps[split] = {"influence": own, "scores": [o - r for o, r in zip(own, ref)]}
    blocks, mass = suggest_blocks(maps["search"]["scores"])
    n_others, n_disjoint = block_window_counts(maps["search"]["scores"], blocks)
    return {"scores": [round(v, 5) for v in maps["search"]["scores"]],
            "influence": [round(v, 5) for v in maps["search"]["influence"]],
            "blocks": blocks, "block_mass": mass,
            "block_z": block_z(maps["search"]["scores"], blocks),
            "block_z_heldout": block_z(maps["heldout"]["scores"], blocks),
            # so a reader can judge block_z: most of its windows overlap the block
            "block_z_windows": n_others, "block_z_disjoint_windows": n_disjoint,
            "block_z_disjoint": block_z_disjoint(maps["search"]["scores"], blocks),
            "block_z_disjoint_heldout": block_z_disjoint(maps["heldout"]["scores"], blocks),
            "heldout": {"blocks": blocks, "select_mass": mass,
                        "check_mass": positive_share(maps["heldout"]["scores"], blocks),
                        "split_half_r": pearson(maps["search"]["scores"], maps["heldout"]["scores"])}}


def has_module_data(res):
    return res.get("attn_lp") is not None and res.get("mlp_lp") is not None


def module_increases(res):
    # One engine result with module data, as loss increases: for each module and each "skill|split"
    # set, the nats of answer log-prob lost per layer when the module of that layer is zeroed.
    # None for a layer that lacks the module or was not measured in this call; None in place of
    # the list for a set with no base score. "layers" names the layers this call measured.
    # Increases, and not log-probs, are what several calls can share: each call brings its own base.
    n_layers = len(res["module_parts"])
    measured = res.get("layers")
    if measured is None:
        measured = list(range(n_layers))
    out = {"layers": list(measured), "parts": res["module_parts"]}
    for module in MODULES:
        out[module] = {}
        for key in res[module + "_lp"]:
            base = res["base"][key]["lp"]
            if base is None:
                out[module][key] = None
            else:
                out[module][key] = [None if lp is None else base - lp for lp in res[module + "_lp"][key]]
    return out


def merge_increases(done, new):
    # Adds the layers that one more call measured to what earlier calls measured. `done` is None
    # before the first call. Returns the merged record; `new` is not changed.
    if done is None:
        return json.loads(json.dumps(new))
    for module in MODULES:
        for key in new[module]:
            if new[module][key] is None or done[module].get(key) is None:
                done[module][key] = None
                continue
            for l in new["layers"]:
                done[module][key][l] = new[module][key][l]
    for l in new["layers"]:
        if l not in done["layers"]:
            done["layers"].append(l)
    done["layers"].sort()
    return done


def module_scores(inc, skill, split, module):
    # The layer score definition for one module: nats of answer log-prob lost on skill text when
    # the module of each layer is zeroed, minus the same on general_text. None for a layer that
    # lacks the module. Returns None when the skill or the reference has no base score.
    own = inc[module].get(skill + "|" + split)
    ref = inc[module].get(REFERENCE + "|" + split)
    if own is None or ref is None:
        return None
    return [None if o is None or r is None else o - r for o, r in zip(own, ref)]


def module_share(attn, mlp):
    # Per layer: the attention share of the positive attn + mlp score.
    # None when the layer lacks either module: a share needs both scores, and a module that was
    # not measured must not read as a measured 0. The attn and mlp lists keep the one score.
    # None when neither score is above SCORE_RESOLUTION; a score at or below it counts as 0.
    out = []
    for a, m in zip(attn, mlp):
        if a is None or m is None:
            out.append(None)
            continue
        pa = a if a > SCORE_RESOLUTION else 0.0
        pm = m if m > SCORE_RESOLUTION else 0.0
        out.append(None if pa + pm <= 0 else round(pa / (pa + pm), 4))
    return out


def pearson_present(a, b, min_n=2):
    # Pearson r over the layers where both lists have a number, and how many layers that is.
    # r is None with fewer than min_n such layers.
    xs = []
    ys = []
    for x, y in zip(a, b):
        if x is not None and y is not None:
            xs.append(x)
            ys.append(y)
    return (pearson(xs, ys) if len(xs) >= min_n else None), len(xs)


def module_map(inc, skill):
    # inc: the record of module_increases(), or of merge_increases() over several calls
    scores = {}
    for module in MODULES:
        for split in ["search", "heldout"]:
            found = module_scores(inc, skill, split, module)
            if found is None:
                return None
            scores[module + "|" + split] = found
    out = {}
    agreement = {}
    counts = {}
    for module in MODULES:
        out[module] = [None if v is None else round(v, 5) for v in scores[module + "|search"]]
        agreement[module], counts[module] = pearson_present(scores[module + "|search"], scores[module + "|heldout"],
                                                            MIN_R_LAYERS)
    # from the rounded scores, so the share matches the attn and mlp lists that a reader sees
    out["module_share"] = module_share(out["attn"], out["mlp"])
    out["heldout"] = {"split_half_r": agreement, "n_layers": counts}
    return out


MODULE_METHOD = ("module scores: loss increase on skill text when the attention output (attn) or "
                 "the feed-forward output (mlp: dense MLP, routed experts, or both) of one layer "
                 "is zeroed, minus the same for general_text (nats, search half). module_share: "
                 "attention share of the positive attn + mlp score per layer, null where the layer lacks "
                 "one of the two modules. split_half_r: Pearson r of the search and held-out score "
                 "lists, per module, over the n_layers layers that have the module. module_parts: "
                 "the part kinds zeroed per layer.")


def module_fields(inc, names):
    # the entry fields of a --modules run, for the skills in `names`
    module_maps = {}
    for name in names:
        if name == REFERENCE:
            continue
        mm = module_map(inc, name)
        if mm is not None:
            module_maps[name] = mm
    return {"module_maps": module_maps, "module_parts": inc["parts"], "module_method": MODULE_METHOD}


def reported_skills(selection_sources):
    # The skills whose text comes from a dataset on the reported list, in one place. None when
    # that is not known (no selection_sources, or wsp/healdata.py was not found).
    if selection_sources is None:
        return None
    flags = [selection_sources[name].get("reported_benchmark") for name in selection_sources]
    if any(flag is None for flag in flags):
        return None
    return sorted(name for name in selection_sources if selection_sources[name]["reported_benchmark"])


def expert_map(res, skill):
    # per layer: experts the router picks more on skill text than on general text, on both halves
    freq = res.get("expert_freq")
    if not freq or skill + "|search" not in freq:
        return None, None
    top = {}
    details = {}
    for layer in sorted(freq[skill + "|search"], key=int):
        rows = []
        s_search = freq[skill + "|search"][layer]
        g_search = freq[REFERENCE + "|search"][layer]
        s_held = freq[skill + "|heldout"][layer]
        g_held = freq[REFERENCE + "|heldout"][layer]
        for e in range(len(s_search)):
            ratio = math.log((s_search[e] + EPS) / (g_search[e] + EPS))
            held = math.log((s_held[e] + EPS) / (g_held[e] + EPS))
            if ratio > 0 and held > 0:
                rows.append({"expert": e, "skill_freq": s_search[e], "general_freq": g_search[e],
                             "log_ratio": round(ratio, 4), "heldout_log_ratio": round(held, 4)})
        rows.sort(key=lambda r: -r["log_ratio"])
        rows = rows[:TOP_EXPERTS]
        if rows:
            top[layer] = [r["expert"] for r in rows]
            details[layer] = rows
    return top, details


def build_entry(model, res, sources, row, selection_sources=None):
    # selection_sources: the records of load_selection_sources(), or None when the caller has none.
    # The module fields are written only when the engine result has module data (--modules).
    info = res["info"]
    skills = {}
    layer_maps = {}
    expert_maps = {}
    expert_details = {}
    for name in sources:
        skills[name] = res["base"][name + "|search"]["score"]
        if name == REFERENCE:
            continue
        lm = layer_map(res, name)
        if lm is not None:
            layer_maps[name] = lm
        top, details = expert_map(res, name)
        if top:
            expert_maps[name] = top
            expert_details[name] = details
    entry = {
        "model": model, "skills": skills,
        "skill_metric": "geometric-mean probability of the answer tokens on the skill's search items "
                        "(teacher forced). Plain non-benchmark text; not a benchmark score.",
        "skill_details": {}, "layer_maps": layer_maps, "expert_maps": expert_maps or None,
        "expert_details": expert_details or None,
        "params_b": row.get("params_b"), "layers": info["n_layers"], "hidden": info["d_model"],
        "family": row.get("family"), "architecture": info.get("model_class"), "license": row.get("license"),
        "structure": {"layers_with": info["layers_with"], "example": info["example"], "vocab": info.get("vocab")},
        "method": "layer scores: loss increase on skill text when one layer is skipped, minus the same for "
                  "general_text (nats). Blocks: shortest window with 60% of the positive score, chosen on the "
                  "search half; check_mass and split_half_r come from the held-out half. Experts: router pick "
                  "rate on skill text over general text, positive log ratio on both halves.",
        "selection_data": sources, "benchmark_free": True, "source": "person_a/mine/maps.py",
        "revision": res.get("revision"), "gpu": res.get("gpu"), "engine_seconds": res["seconds"],
        "measured_at": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "selection_sources": selection_sources,
        # benchmark_free above is always true and does not look at the skills; this list does
        "reported_benchmark_skills": reported_skills(selection_sources),
    }
    if has_module_data(res):
        entry.update(module_fields(module_increases(res), sources))
    return entry


def read_entry(path):
    # the entry file as a dict, or None when it is not there or is not a JSON object
    try:
        with open(path) as f:
            entry = json.load(f)
    except (OSError, ValueError):
        return None
    return entry if isinstance(entry, dict) else None


def entry_done(path, modules):
    # True when the entry file exists and already has what this run would measure. For --modules
    # the test is the key: an entry with an empty module_maps was measured and must not be paid again.
    if not os.path.exists(path):
        return False
    if not modules:
        return True
    entry = read_entry(path)
    return entry is not None and "module_maps" in entry


def write_entry(path, entry, force=False):
    # Writes the entry file. Returns False and leaves the file as it is when the write would put
    # an entry without module maps over one that has them (a plain run that ends after a --modules
    # run of the same model) and force is not set. The tmp name has the process id, because two
    # runs of one model can write at the same time.
    current = read_entry(path)
    if not force and current is not None and "module_maps" in current and "module_maps" not in entry:
        return False
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + "." + str(os.getpid()) + ".tmp"
    with open(tmp, "w") as f:
        json.dump(entry, f, indent=1)
    os.replace(tmp, path)
    return True


def skill_hashes(payload):
    return dict((name, payload["skills"][name]["hash"]) for name in payload["skills"])


def plain_entry_fits(entry, model, revision, sources, hashes=None):
    # True when a --modules run can keep the layer and expert maps of this entry and add the module
    # maps to it: the same model revision and the same skill text. An entry of another revision or
    # of other skill sets is measured again from the start.
    # hashes: skill_hashes() of the payload of this run. An entry that stores skill_hashes is
    # compared by them, so sets built again with other items and the same descriptions do not fit.
    # An entry from before that field has only the dataset and description strings to compare.
    if entry is None or "layer_maps" not in entry or not entry.get("layers"):
        return False
    if hashes is not None and entry.get("skill_hashes") is not None and entry["skill_hashes"] != hashes:
        return False
    return entry.get("model") == model and entry.get("revision") == revision and entry.get("selection_data") == sources


def seconds_per_pass(entry):
    # Estimate from a plain entry: its engine time over its scoring passes (one for the base and
    # one per layer). The time also holds the router counts, so the estimate is a little high.
    if not entry.get("engine_seconds") or not entry.get("layers"):
        return None
    return entry["engine_seconds"] / (1 + entry["layers"])


def plan_module_calls(n_layers, per_pass, per_call=0):
    # Splits the layers of a --modules run over engine calls. One call for c layers takes at most
    # 1 + 2 * c scoring passes (the base, then attention and feed-forward per layer).
    # per_pass: seconds per pass from seconds_per_pass(), or None when not known.
    # per_call: layers per call set by the user; 0 picks the most layers whose estimate stays
    # within CALL_BUDGET of CALL_TIMEOUT, or 1 layer per call when there is no estimate.
    # Returns None when the estimate says that a call for a single layer does not fit the timeout.
    if per_pass is not None and 3 * per_pass > CALL_TIMEOUT:
        return None
    if per_call <= 0:
        per_call = 1
        if per_pass is not None:
            per_call = max(1, int((CALL_TIMEOUT * CALL_BUDGET / per_pass - 1) / 2))
    per_call = min(per_call, n_layers)
    return [list(range(first, min(n_layers, first + per_call))) for first in range(0, n_layers, per_call)]


class OldEngine(Exception):
    pass


def is_old_engine_error(exc):
    # An engine deployed before skill_maps had modules, layers, and plain refuses the call with a
    # TypeError on the unknown keyword, and no result comes back to check.
    return isinstance(exc, OldEngine) or "unexpected keyword argument" in str(exc)


def partial_path(model):
    # next to the logs, and not in pool_entries, where every *.json file is read as an entry
    return os.path.join(LOG_DIR, model.replace("/", "__") + ".modules-partial.json")


def load_partial(path, stamp):
    # the saved state of an earlier --modules run that stopped, when it is for the same model
    # revision, skill sets, and layer count; else a new state
    saved = read_entry(path)
    if saved is not None and saved.get("stamp") == stamp:
        return saved
    return {"stamp": stamp, "increases": None, "engine_seconds": 0.0, "calls": 0}


def measure_modules(call, chunks, path, stamp):
    # Runs the module calls and merges their results.
    # call(layers) returns the engine result of skill_maps(payload, modules=True, layers=layers,
    # plain=False). The merged state goes to `path` after every call, so a call that fails or
    # reaches its timeout loses only its own layers: the next run starts after the saved ones.
    state = load_partial(path, stamp)
    for chunk in chunks:
        done = [] if state["increases"] is None else state["increases"]["layers"]
        todo = [l for l in chunk if l not in done]
        if not todo:
            continue
        res = call(todo)
        if not has_module_data(res) or res.get("layers") is None:
            raise OldEngine("the engine result has no module data for a layer list")
        if sorted(res["layers"]) != todo:
            raise RuntimeError("asked for layers " + str(todo) + ", the engine measured " + str(res["layers"]))
        state["increases"] = merge_increases(state["increases"], module_increases(res))
        state["engine_seconds"] = round(state["engine_seconds"] + res["seconds"], 1)
        state["calls"] += 1
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = path + "." + str(os.getpid()) + ".tmp"
        with open(tmp, "w") as f:
            json.dump(state, f)
        os.replace(tmp, path)
        print("module call", state["calls"], "| layers", todo[0], "to", todo[-1], "| engine", res["seconds"], "s",
              flush=True)
    return state


BLOCK_Z_NOTE = ("block_z, block_z_windows, block_z_disjoint_windows, and block_z_disjoint were computed by a "
                "--modules run from the stored search scores (rounded to 5 places). The held-out scores of this "
                "entry were not stored, so the held-out values are null.")


def with_block_z(lm):
    # A layer map written before block_z existed, with the block z-scores that need no GPU: they come
    # from the stored search scores and blocks. The held-out lists hold null for every block.
    if "block_z" in lm or "scores" not in lm or "blocks" not in lm:
        return lm
    out = dict(lm)
    n_others, n_disjoint = block_window_counts(lm["scores"], lm["blocks"])
    out["block_z"] = block_z(lm["scores"], lm["blocks"])
    out["block_z_heldout"] = [None] * len(lm["blocks"])
    out["block_z_windows"] = n_others
    out["block_z_disjoint_windows"] = n_disjoint
    out["block_z_disjoint"] = block_z_disjoint(lm["scores"], lm["blocks"])
    out["block_z_disjoint_heldout"] = [None] * len(lm["blocks"])
    out["block_z_note"] = BLOCK_Z_NOTE
    return out


def add_module_fields(entry, state, sources, selection_sources):
    # The plain entry with the module fields added. The fields that the plain run wrote stay as
    # they are; an entry written before selection_sources existed gets it now (it is not measured),
    # and a layer map written before block_z existed gets the block z-scores of its search scores.
    out = dict(entry)
    if isinstance(out.get("layer_maps"), dict):
        out["layer_maps"] = dict((name, with_block_z(out["layer_maps"][name])) for name in out["layer_maps"])
    out.update(module_fields(state["increases"], sources))
    out["module_engine_seconds"] = state["engine_seconds"]
    out["module_calls"] = state["calls"]
    out["module_measured_at"] = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    if out.get("selection_sources") is None:
        out["selection_sources"] = selection_sources
    if out.get("reported_benchmark_skills") is None:
        out["reported_benchmark_skills"] = reported_skills(out["selection_sources"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zoo-key", default=None, help="key of a model in atlas/zoo.json")
    ap.add_argument("--short", default=None, help="short name in registry/donors.csv")
    ap.add_argument("--force", action="store_true", help="measure again when the entry exists")
    ap.add_argument("--modules", action="store_true",
                    help="also map the attention and feed-forward module of every layer (up to 2 more scoring "
                         "passes per layer, in engine calls of their own)")
    ap.add_argument("--layers-per-call", type=int, default=0,
                    help="with --modules: layers per engine call (default: from the time of the plain run)")
    args = ap.parse_args()
    import modal

    if args.short:
        d = common.load_donors()[args.short]
        key = args.short
        row = {"hf_id": d["hf_id"], "params_b": float(d["params_b"]), "family": None, "license": d.get("license")}
    else:
        with open(ZOO) as f:
            zoo = json.load(f)["models"]
        found = [m for m in zoo if m["key"] == args.zoo_key]
        if not found:
            raise SystemExit("unknown zoo key " + str(args.zoo_key))
        key = args.zoo_key
        row = {"hf_id": found[0]["id"], "params_b": float(found[0]["params_b"]), "family": found[0].get("family"),
               "license": found[0].get("license")}
    out_path = os.path.join(OUT_DIR, row["hf_id"].replace("/", "__") + ".json")
    if entry_done(out_path, args.modules) and not args.force:
        print("exists:", out_path)
        return
    payload, sources = load_skills()
    healdata = load_healdata()
    if healdata is None:
        print("warning:", HEALDATA, "not found: pool and reported_benchmark are null in selection_sources")
    details = load_selection_sources(healdata)
    gpu = common.pick_gpu(row["params_b"], key)
    revision = common.pin_revision(row["hf_id"])
    print("maps", row["hf_id"], "| revision", revision[:10], "| GPU", gpu, "|", len(sources), "skill sets", flush=True)
    start = time.time()
    engine = modal.Cls.from_name(common.APP_NAME, gpu)(model_key=key, hf_id=row["hf_id"], revision=revision)
    entry = None
    hashes = skill_hashes(payload)
    if args.modules and not args.force:
        entry = read_entry(out_path)
        if plain_entry_fits(entry, row["hf_id"], revision, sources, hashes):
            print("keeping the layer and expert maps of", out_path, "and adding the module maps", flush=True)
        else:
            if entry is not None:
                print("the entry is for another revision or other skill sets, measuring it again:", out_path)
            entry = None
    if entry is None:
        # same call as before --modules existed, so it also works with an older deployed engine
        res = engine.skill_maps.remote(payload)
        res["revision"] = revision
        res["gpu"] = gpu
        entry = build_entry(row["hf_id"], res, sources, row, details)
        entry["skill_hashes"] = hashes  # the items every map of this entry was measured on
        entry["wall_seconds"] = round(time.time() - start, 1)
        # written now, before any module call, so a module call that fails does not lose it
        if write_entry(out_path, entry, args.force):
            print("wrote", out_path, "| engine", res["seconds"], "s | wall", entry["wall_seconds"], "s")
        else:
            print("kept", out_path, "as it is: it has module maps and this run has none (use --force to replace it)")
            return
        best = []
        for name in entry["layer_maps"]:
            lm = entry["layer_maps"][name]
            if lm["blocks"]:
                best.append(name + " " + str(lm["blocks"][0]) + " r=" + str(lm["heldout"]["split_half_r"]) +
                            " z=" + str(lm["block_z"][0]))
        print("first block per skill:", "; ".join(best))
    if not args.modules:
        return
    per_pass = seconds_per_pass(entry)
    chunks = plan_module_calls(entry["layers"], per_pass, args.layers_per_call)
    if chunks is None:
        raise SystemExit("the plain run took " + str(round(per_pass, 1)) + " s per scoring pass: the 3 passes of a "
                         "module call for one layer do not fit the " + str(CALL_TIMEOUT) + " s call timeout")
    print("module maps:", len(chunks), "engine calls of at most", 1 + 2 * len(chunks[0]), "scoring passes | plain run:",
          "no time known" if per_pass is None else str(round(per_pass, 1)) + " s per pass", flush=True)
    stamp = {"model": row["hf_id"], "revision": revision, "layers": entry["layers"], "skills": hashes}
    partial = partial_path(row["hf_id"])
    if args.force and os.path.exists(partial):
        os.remove(partial)  # --force measures again: the saved calls of an earlier run are not used

    def call(layers):
        return engine.skill_maps.remote(payload, modules=True, layers=layers, plain=False)
    try:
        state = measure_modules(call, chunks, partial, stamp)
    except Exception as exc:
        if is_old_engine_error(exc):
            print(OLD_ENGINE_MESSAGE, "(" + type(exc).__name__ + ": " + str(exc) + ")")
            sys.exit(OLD_ENGINE_EXIT)
        raise
    # the entry may have been written again while the module calls ran: add to the file as it is now
    current = read_entry(out_path)
    if plain_entry_fits(current, row["hf_id"], revision, sources, hashes):
        entry = current
    entry = add_module_fields(entry, state, sources, details)
    write_entry(out_path, entry, True)
    if os.path.exists(partial):
        os.remove(partial)
    print("wrote", out_path, "| module maps:", len(entry["module_maps"]), "skills |", state["calls"], "calls | engine",
          state["engine_seconds"], "s")


if __name__ == "__main__":
    main()
