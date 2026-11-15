# Evidence fields of a catalog row, computed from what the row already stores (no GPU):
#   - z: the target drop as a z-score against the drops of random same-size parts
#   - guard: the largest side effect on any other skill
#   - selection_data: the data the part was selected on, and which of it is a reported benchmark
#   - routing: router pick rates of an expert part, put together from the engine's counts
# client.py calls these when it writes a row. enrich_rows.py calls them to backfill old rows.
import ast
import importlib.util
import json
import math
import os
import re

import common

SPLITS = ["search", "heldout"]
ROUTING_REFERENCE = "general_text"
POOL_SKILLS = os.path.join(common.MINE_DIR, "data", "pool_skills.json")
BUILD_DATA = os.path.join(common.MINE_DIR, "build_data.py")
# The repo root is one level above person_a/. A copy of mine/ outside the team repo has no wsp/ folder.
HEALDATA = os.path.join(os.path.dirname(common.ROOT), "wsp", "healdata.py")

_cache = {}


def is_number(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


# ---------- z-scores against random same-size parts ----------

def random_split(random_drops):
    # The engine's random_drops gives a flat list: target drops on the search half only. Two wider
    # shapes are read as well, so a later engine that also scores the held-out half needs no change
    # here: {"search": [...], "heldout": [...]}, and a list of {"search": x, "heldout": y}.
    # Missing values (None) are left out.
    out = {"search": [], "heldout": []}
    if isinstance(random_drops, dict):
        for split in SPLITS:
            for v in random_drops.get(split) or []:
                if is_number(v):
                    out[split].append(float(v))
        return out
    for v in random_drops or []:
        if isinstance(v, dict):
            for split in SPLITS:
                if is_number(v.get(split)):
                    out[split].append(float(v[split]))
        elif is_number(v):
            out["search"].append(float(v))
    return out


def spread(values):
    # count, mean, and sample standard deviation (n - 1). The deviation needs 2 values.
    n = len(values)
    if n == 0:
        return {"n": 0, "mean": None, "std": None}
    mean = math.fsum(values) / n
    std = None
    if n >= 2:
        if max(values) == min(values):
            std = 0.0  # exact: a rounded mean would leave a tiny false deviation
        else:
            std = math.sqrt(math.fsum((v - mean) ** 2 for v in values) / (n - 1))
    return {"n": n, "mean": mean, "std": std}


def z_score(drop, values):
    # None when there are fewer than 2 random drops, when they are all equal, or when the drop is missing
    s = spread(values)
    if not is_number(drop) or s["n"] < 2 or s["std"] == 0:
        return None
    return (drop - s["mean"]) / s["std"]


def rounded(value, digits):
    if value is None:
        return None
    return round(value, digits)


def z_fields(effect, random_drops, target):
    rnd = random_split(random_drops)
    search = spread(rnd["search"])
    out = {"z": rounded(z_score(effect.get(target + "|search"), rnd["search"]), 3),
           "z_heldout": rounded(z_score(effect.get(target + "|heldout"), rnd["heldout"]), 3),
           "random_n": search["n"], "random_mean": rounded(search["mean"], 6),
           "random_std": rounded(search["std"], 6), "random_heldout": None}
    if rnd["heldout"]:
        held = spread(rnd["heldout"])
        out["random_heldout"] = {"n": held["n"], "mean": rounded(held["mean"], 6), "std": rounded(held["std"], 6)}
    return out


# ---------- guard: the largest side effect ----------

def guard(effect, target):
    # Per half: the skill, other than the target and the target's own control, whose score moved the
    # most in either direction when the part was switched off. Other skills' controls count.
    out = {}
    for split in SPLITS:
        best = None
        for key in sorted(effect):  # sorted, so equal values give the same answer on every run
            if "|" not in key:
                continue
            name, key_split = key.rsplit("|", 1)
            if key_split != split or name == target or name == target + "__control":
                continue
            v = effect[key]
            if not is_number(v):
                continue
            if best is None or abs(v) > best["abs"]:
                best = {"skill": name, "drop": v, "abs": abs(v)}
        out[split] = best
    return out


def signature_fields(effect, random_drops, target):
    out = z_fields(effect, random_drops, target)
    out["guard"] = guard(effect, target)
    return out


# ---------- where the skill data comes from ----------

def read_json(path):
    if path not in _cache:
        with open(path) as f:
            _cache[path] = json.load(f)
    return _cache[path]


def load_healdata(path=None):
    # Loaded by file path: importing the wsp package would run its __init__, which pulls in heavy
    # dependencies. Returns None when the file is not there.
    path = path or HEALDATA
    key = "healdata:" + path
    if key not in _cache:
        module = None
        if os.path.exists(path):
            spec = importlib.util.spec_from_file_location("wsp_healdata_by_path", path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
        _cache[key] = module
    return _cache[key]


def dataset_id(text):
    # pool skills record "<dataset id> <config or split words>": the id is the first word
    words = str(text or "").split()
    if not words:
        return None
    return words[0]


def builder_datasets(source_text):
    # Reads the text of a build_data.py without running it. For each built-in skill named in the
    # builders list, ("<name>", build_<name>, "<description>"), returns the description and the
    # dataset ids that its builder function passes to load().
    tree = ast.parse(source_text)
    loads = {}
    for node in tree.body:
        if not isinstance(node, ast.FunctionDef):
            continue
        ids = []
        for call in ast.walk(node):
            if not isinstance(call, ast.Call) or not isinstance(call.func, ast.Name) or call.func.id != "load":
                continue
            if call.args and isinstance(call.args[0], ast.Constant) and isinstance(call.args[0].value, str):
                if call.args[0].value not in ids:
                    ids.append(call.args[0].value)
        loads[node.name] = ids
    out = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Tuple) or len(node.elts) != 3:
            continue
        name, fn, desc = node.elts
        if (isinstance(name, ast.Constant) and isinstance(name.value, str) and isinstance(fn, ast.Name)
                and fn.id in loads and isinstance(desc, ast.Constant) and isinstance(desc.value, str)):
            out[name.value] = {"description": desc.value, "datasets": loads[fn.id]}
    return out


def local_builders():
    key = "builders:" + BUILD_DATA
    if key not in _cache:
        builders = {}
        if os.path.exists(BUILD_DATA):
            with open(BUILD_DATA) as f:
                builders = builder_datasets(f.read())
        _cache[key] = builders
    return _cache[key]


def builtin_sources(described, builders):
    # described: {name: {"description": ..., "dataset": ... or absent}}, read from a built-in skill
    # file or reported by the engine. The file holds a description and no dataset id, so the ids come
    # from the builder, and only when the builder's description is the one in the file. If they
    # differ, the file was built by another version of build_data.py and the ids are left empty.
    out = {}
    for name in described:
        desc = described[name].get("description")
        datasets = []
        if described[name].get("dataset"):
            datasets = [dataset_id(described[name]["dataset"])]
        elif name in builders and builders[name]["description"] == desc:
            datasets = list(builders[name]["datasets"])
        out[name] = {"source": "builtin", "description": desc, "datasets": datasets}
    return out


def local_builtin():
    data = read_json(common.BUILTIN_SKILLS)
    described = {}
    for name in data:
        described[name] = {"description": data[name].get("description"), "dataset": data[name].get("dataset")}
    return builtin_sources(described, local_builders())


def local_pool():
    out = {}
    if not os.path.exists(POOL_SKILLS):
        return out
    data = read_json(POOL_SKILLS)
    for name in data:
        ids = []
        if dataset_id(data[name].get("dataset")):
            ids = [dataset_id(data[name]["dataset"])]
        out[name] = {"source": "pool", "description": data[name].get("description"),
                     "dataset": data[name].get("dataset"), "hash": data[name].get("hash"), "datasets": ids}
    return out


def make_lookup(agent, builtin, pool=None):
    # Returns lookup(name) -> where a skill's test data comes from. agent: {name: {"path", "hash",
    # "description"}}; builtin: the output of builtin_sources; pool: the output of local_pool, or
    # None to read the pool file only when a name is in neither of the others. An agent skill
    # cannot take a built-in name (skills.define refuses it), and the miner scores agent and built-in
    # skills only, so the order agent, built-in, pool is the order of what a row can hold.
    held = {"pool": pool}

    def lookup(name):
        if name in agent:
            spec = agent[name]
            return {"source": "agent", "path": spec.get("path"), "hash": spec.get("hash"),
                    "description": spec.get("description"), "datasets": []}
        if name in builtin:
            return dict(builtin[name])
        if held["pool"] is None:
            held["pool"] = local_pool()
        if name in held["pool"]:
            return dict(held["pool"][name])
        return {"source": "unknown", "description": None, "datasets": []}
    return lookup


# ---------- reported benchmarks and text pools ----------

def banned_match(dataset, banned):
    # the rule of wsp/healdata.py: the id is a banned id, or ends with "/<banned id>"
    name = str(dataset).lower().strip()
    for b in banned:
        if name == b or name.endswith("/" + b):
            return True
    return False


def banned_in_text(text, banned):
    # For a built-in set whose dataset id is not known: a banned id, or the part after its "/", that
    # stands as a whole word in the description ("GSM8K word problems" gives openai/gsm8k). "Orca-Math"
    # is one word, so it does not match a benchmark named "math". A benchmark that the description
    # names in another spelling is not found.
    words = set(re.findall(r"[a-z0-9][a-z0-9_./-]*", str(text or "").lower()))
    found = []
    seen = set()
    for b in banned:
        short = b.split("/")[-1]
        if (b in words or short in words) and short not in seen:
            seen.add(short)
            found.append(b)
    return found


def reported_benchmarks(infos, banned):
    out = []
    for info in infos:
        hits = []
        for d in info.get("datasets") or []:
            if banned_match(d, banned):
                hits.append(str(d).lower().strip())
        if not info.get("datasets") and info.get("source") in ["builtin", "pool"]:
            hits = banned_in_text(info.get("description"), banned)
        for h in hits:
            if h not in out:
                out.append(h)
    return out


def match_pool(name, info, sources):
    # The text pool of wsp/healdata.py SOURCES for this skill, and how it was matched. In order: the
    # skill name is a pool name; the skill's dataset is a source of a pool; for built-in and pool
    # skills, the name's first word is a pool name ("code_rust", "prose_de"). Agent skill names are
    # free text, so their first word is not used.
    if name in sources:
        return name, "name"
    ids = [str(d).lower() for d in info.get("datasets") or []]
    for pool in sources:
        for source in sources[pool]:
            if str(source[0]).lower() in ids:
                return pool, "dataset"
    first = name.split("_")[0]
    if info.get("source") in ["builtin", "pool"] and first in sources:
        return first, "name prefix"
    return None, None


def off_target_names(effect, target):
    # The skills the light gate checks besides the target: every scored skill that is not the target
    # and not a control set. They are read from the search half, the half the gate limits apply to.
    names = []
    for key in effect:
        if "|" not in key:
            continue
        name, split = key.rsplit("|", 1)
        if split != "search" or name == target or name.endswith("__control") or name in names:
            continue
        names.append(name)
    return names


def selection_data(target, effect, lookup, healdata, builtin_basis):
    # Names the data a part was selected on. builtin_basis says where the built-in descriptions came
    # from: "engine" (the deployed engine reported them when the row was written), "local file"
    # (the engine could not be asked; its copy can be older than the local file), or "git history"
    # (backfill: the version of the built-in file at the row's creation time).
    info = lookup(target)
    out = {"skill": target, "source": info["source"]}
    for key in ["path", "hash", "description", "dataset", "datasets"]:
        if key in info:
            out[key] = info[key]
    out["control"] = target + "__control" if (target + "__control|search") in effect else None
    out["off_target"] = off_target_names(effect, target)
    out["off_target_data"] = {}
    infos = [info]
    for name in out["off_target"]:
        other = lookup(name)
        infos.append(other)
        if other["source"] == "agent":
            # the hash names the frozen test in mine/skills/<model>/; its text stays in that file
            out["off_target_data"][name] = {"source": "agent", "hash": other.get("hash")}
        else:
            out["off_target_data"][name] = {"source": other["source"], "description": other.get("description"),
                                            "datasets": other.get("datasets") or []}
    out["builtin_basis"] = builtin_basis
    if healdata is None:
        # not checked is not the same as none found
        out["pool"] = None
        out["pool_match"] = None
        out["reported_benchmarks"] = None
        out["note"] = "wsp/healdata.py not found: pool and reported_benchmarks were not checked"
        return out
    out["pool"], out["pool_match"] = match_pool(target, info, healdata.SOURCES)
    out["reported_benchmarks"] = reported_benchmarks(infos, [str(b).lower() for b in healdata.BANNED])
    return out


# ---------- routing of expert parts ----------

def log_ratio(skill_rate, general_rate):
    # natural log of skill rate over general rate; None when a rate is 0 or missing (both rates are stored)
    if not is_number(skill_rate) or not is_number(general_rate) or skill_rate <= 0 or general_rate <= 0:
        return None
    return round(math.log(skill_rate / general_rate), 4)


def layer_rates(result, name, split, layer):
    entry = (result.get("rates") or {}).get(name + "|" + split) or {}
    return (entry.get("layers") or {}).get(str(layer))


def routing_fields(indices, layer, target, result, general=ROUTING_REFERENCE):
    # result: what the engine's expert_pick_rates returned for this layer. Returns (routing, note);
    # routing is None when the layer cannot be counted, and the note says why.
    reason = (result.get("unsupported") or {}).get(str(layer))
    if reason:
        return None, "no routing: " + reason
    experts = {}
    tokens = {}
    n_items = {}
    found = False
    for i in sorted(int(x) for x in indices):
        per = {}
        for split in SPLITS:
            rates = {}
            for label, name in [("skill", target), ("general", general)]:
                counted = layer_rates(result, name, split, layer)
                rates[label] = None
                if counted and i < len(counted["rates"]):
                    rates[label] = counted["rates"][i]
                    tokens[name + "|" + split] = counted["tokens"]
                    n_items[name + "|" + split] = result["rates"][name + "|" + split].get("n_items")
                    found = True
            per[split] = {"skill_rate": rates["skill"], "general_rate": rates["general"],
                          "log_ratio": log_ratio(rates["skill"], rates["general"])}
        experts[str(i)] = per
    if not found:
        return None, "no routing: the engine returned no pick rates for layer " + str(layer)
    return {"layer": int(layer), "skill": target, "general": general, "n_items": n_items, "tokens": tokens,
            "experts": experts}, None
