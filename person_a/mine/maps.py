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
#
# --modules also measures each layer with only its attention output zeroed and with only its
# feed-forward output zeroed (dense MLP, routed experts, or both). The entry then has
#   module_maps[skill].attn, .mlp     the layer score definition, per module (null: no such module)
#   module_maps[skill].module_share   per layer, the attention share of the positive attn + mlp score
#   module_maps[skill].heldout.split_half_r.attn, .mlp   Pearson r of the search and held-out lists
#   module_parts                      per layer, the part kinds that were zeroed for each module
# A plain run takes one scoring pass per layer (every skill set scored once). --modules adds one
# pass per layer for the attention module and one for the feed-forward module, where the layer has
# them: up to three passes per layer. It needs an engine deployed with skill_maps(modules=...).
# An entry that exists without module_maps is measured again when --modules is set.
#
# Data: skill text comes from mine/data/pool_skills.json and the built-in math, code and
# general_text sets. Mining may use any dataset, and some pool skills come from a benchmark (the
# MMLU domain skills). selection_sources says, per skill, the dataset, the wsp/healdata.py pool that
# draws on the same dataset, and whether the dataset is on the default reported list (BANNED), so
# compile and heal can keep away from data they are scored on.
# Selection uses the "search" half of each set; the "heldout" half only checks it.
import argparse
import datetime
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
        skills[name] = {"hash": "maps-builtin-" + name, "control": None,
                        "items": {"search": items["search"][:N_SEARCH], "heldout": items["heldout"][:N_HELDOUT]}}
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


def window_z(scores, first, last):
    # How far the block's positive score mass is above the positive mass of every other window of
    # the same length in the same model, in sample standard deviations of those other windows.
    # The other windows may overlap the block. None with fewer than MIN_OTHER_WINDOWS other
    # windows, or when they all have the same mass.
    length = last - first + 1
    own = sum(max(0.0, v) for v in scores[first:last + 1])
    others = []
    for start in range(0, len(scores) - length + 1):
        if start != first:
            others.append(sum(max(0.0, v) for v in scores[start:start + length]))
    if len(others) < MIN_OTHER_WINDOWS:
        return None
    mean = sum(others) / len(others)
    sd = math.sqrt(sum((v - mean) ** 2 for v in others) / (len(others) - 1))
    if sd <= 1.0e-9:
        return None
    return round((own - mean) / sd, 3)


def block_z(scores, blocks):
    return [window_z(scores, first, last) for first, last in blocks]


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
    return {"scores": [round(v, 5) for v in maps["search"]["scores"]],
            "influence": [round(v, 5) for v in maps["search"]["influence"]],
            "blocks": blocks, "block_mass": mass,
            "block_z": block_z(maps["search"]["scores"], blocks),
            "block_z_heldout": block_z(maps["heldout"]["scores"], blocks),
            "heldout": {"blocks": blocks, "select_mass": mass,
                        "check_mass": positive_share(maps["heldout"]["scores"], blocks),
                        "split_half_r": pearson(maps["search"]["scores"], maps["heldout"]["scores"])}}


def module_scores(res, skill, split, module):
    # The layer score definition for one module: nats of answer log-prob lost on skill text when
    # the module of each layer is zeroed, minus the same on general_text. None for a layer that
    # lacks the module. Returns None when the skill or the reference has no base score.
    lps = res[module + "_lp"]
    base_own = res["base"][skill + "|" + split]["lp"]
    base_ref = res["base"][REFERENCE + "|" + split]["lp"]
    own = lps.get(skill + "|" + split)
    ref = lps.get(REFERENCE + "|" + split)
    if base_own is None or base_ref is None or own is None or ref is None:
        return None
    out = []
    for o, r in zip(own, ref):
        out.append(None if o is None or r is None else (base_own - o) - (base_ref - r))
    return out


def module_share(attn, mlp):
    # Per layer: the attention share of the positive attn + mlp score. A module that the layer
    # lacks counts as 0. None when neither score is positive.
    out = []
    for a, m in zip(attn, mlp):
        pa = 0.0 if a is None else max(0.0, a)
        pm = 0.0 if m is None else max(0.0, m)
        out.append(None if pa + pm <= 0 else round(pa / (pa + pm), 4))
    return out


def pearson_present(a, b):
    # Pearson r over the layers where both lists have a number
    xs = []
    ys = []
    for x, y in zip(a, b):
        if x is not None and y is not None:
            xs.append(x)
            ys.append(y)
    return pearson(xs, ys)


def has_module_data(res):
    return res.get("attn_lp") is not None and res.get("mlp_lp") is not None


def module_map(res, skill):
    if not has_module_data(res):
        return None
    scores = {}
    for module in MODULES:
        for split in ["search", "heldout"]:
            found = module_scores(res, skill, split, module)
            if found is None:
                return None
            scores[module + "|" + split] = found
    out = {}
    agreement = {}
    for module in MODULES:
        out[module] = [None if v is None else round(v, 5) for v in scores[module + "|search"]]
        agreement[module] = pearson_present(scores[module + "|search"], scores[module + "|heldout"])
    # from the rounded scores: the engine rounds log-probs to 5 places, so a difference below that
    # is float noise, and noise must not turn a zero score into a share of 1
    out["module_share"] = module_share(out["attn"], out["mlp"])
    out["heldout"] = {"split_half_r": agreement}
    return out


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
    module_maps = {}
    expert_maps = {}
    expert_details = {}
    for name in sources:
        skills[name] = res["base"][name + "|search"]["score"]
        if name == REFERENCE:
            continue
        lm = layer_map(res, name)
        if lm is not None:
            layer_maps[name] = lm
        mm = module_map(res, name)
        if mm is not None:
            module_maps[name] = mm
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
    }
    if has_module_data(res):
        entry["module_maps"] = module_maps
        entry["module_parts"] = res.get("module_parts")
        entry["module_method"] = ("module scores: loss increase on skill text when the attention output (attn) or "
                                  "the feed-forward output (mlp: dense MLP, routed experts, or both) of one layer "
                                  "is zeroed, minus the same for general_text (nats, search half). module_share: "
                                  "attention share of the positive attn + mlp score per layer. split_half_r: "
                                  "Pearson r of the search and held-out score lists, per module. module_parts: "
                                  "the part kinds zeroed per layer.")
    return entry


def entry_done(path, modules):
    # True when the entry file exists and already has what this run would measure
    if not os.path.exists(path):
        return False
    if not modules:
        return True
    try:
        with open(path) as f:
            return bool(json.load(f).get("module_maps"))
    except (OSError, ValueError):
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zoo-key", default=None, help="key of a model in atlas/zoo.json")
    ap.add_argument("--short", default=None, help="short name in registry/donors.csv")
    ap.add_argument("--force", action="store_true", help="measure again when the entry exists")
    ap.add_argument("--modules", action="store_true",
                    help="also map the attention and feed-forward module of every layer (up to 3 scoring "
                         "passes per layer in place of 1)")
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
    if os.path.exists(out_path) and not args.force:
        print("entry has no module maps, measuring again:", out_path)
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
    if args.modules:
        res = engine.skill_maps.remote(payload, modules=True)
        if not has_module_data(res):
            raise SystemExit("the engine returned no module data; deploy the engine with skill_maps(modules=...)")
    else:
        # same call as before --modules existed, so it also works with an older deployed engine
        res = engine.skill_maps.remote(payload)
    res["revision"] = revision
    res["gpu"] = gpu
    entry = build_entry(row["hf_id"], res, sources, row, details)
    entry["wall_seconds"] = round(time.time() - start, 1)
    os.makedirs(OUT_DIR, exist_ok=True)
    tmp = out_path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(entry, f, indent=1)
    os.replace(tmp, out_path)
    best = []
    for name in entry["layer_maps"]:
        lm = entry["layer_maps"][name]
        if lm["blocks"]:
            best.append(name + " " + str(lm["blocks"][0]) + " r=" + str(lm["heldout"]["split_half_r"]) +
                        " z=" + str(lm["block_z"][0]))
    print("wrote", out_path, "| engine", res["seconds"], "s | wall", entry["wall_seconds"], "s")
    print("first block per skill:", "; ".join(best))
    if "module_maps" in entry:
        print("module maps:", len(entry["module_maps"]), "skills")


if __name__ == "__main__":
    main()
