# Model-level skill maps for the library: where each skill lives in a model, by layer and by expert.
# For one model, the GPU engine skips each decoder layer in turn and scores every skill set, and for
# MoE layers it counts how often the router picks each expert on each skill's text. No agent, no LLM.
#   .venv/bin/python mine/maps.py --zoo-key Qwen__Qwen2.5-7B-Instruct          # a model of atlas/zoo.json
#   .venv/bin/python mine/maps.py --short lfm2_24b_a2b                         # a row of registry/donors.csv
# Output: atlas/pool_entries/<org>__<name>.json, in the atlas entry format (layer_maps, expert_maps)
# with the measured numbers kept, so a planner can pick a block or an expert set for any skill.
#
# Data rule: skill text comes only from mine/data/pool_skills.json and the built-in math, code and
# general_text sets, all built from plain non-benchmark text. No reported benchmark is used.
# Selection uses the "search" half of each set; the "heldout" half only checks it.
import argparse
import datetime
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
            "heldout": {"blocks": blocks, "select_mass": mass,
                        "check_mass": positive_share(maps["heldout"]["scores"], blocks),
                        "split_half_r": pearson(maps["search"]["scores"], maps["heldout"]["scores"])}}


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


def build_entry(model, res, sources, row):
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
    return {
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
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zoo-key", default=None, help="key of a model in atlas/zoo.json")
    ap.add_argument("--short", default=None, help="short name in registry/donors.csv")
    ap.add_argument("--force", action="store_true", help="measure again when the entry exists")
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
    if os.path.exists(out_path) and not args.force:
        print("exists:", out_path)
        return
    payload, sources = load_skills()
    gpu = common.pick_gpu(row["params_b"], key)
    revision = common.pin_revision(row["hf_id"])
    print("maps", row["hf_id"], "| revision", revision[:10], "| GPU", gpu, "|", len(sources), "skill sets", flush=True)
    start = time.time()
    engine = modal.Cls.from_name(common.APP_NAME, gpu)(model_key=key, hf_id=row["hf_id"], revision=revision)
    res = engine.skill_maps.remote(payload)
    res["revision"] = revision
    res["gpu"] = gpu
    entry = build_entry(row["hf_id"], res, sources, row)
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
            best.append(name + " " + str(lm["blocks"][0]) + " r=" + str(lm["heldout"]["split_half_r"]))
    print("wrote", out_path, "| engine", res["seconds"], "s | wall", entry["wall_seconds"], "s")
    print("first block per skill:", "; ".join(best))


if __name__ == "__main__":
    main()
