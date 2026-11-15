# Evidence fields of a catalog row, computed from what the row already stores (no GPU):
#   - z: the target drop as a z-score against the drops of random same-size parts
#   - guard: the largest side effect on any other skill
#   - selection_data: the data the part was selected on, and which of it is a reported benchmark
#   - routing: router pick rates of an expert part, put together from the engine's counts
# client.py calls these when it writes a row. enrich_rows.py calls them to backfill old rows.
import ast
import hashlib
import importlib.util
import json
import math
import os
import re

import common

SPLITS = ["search", "heldout"]
ROUTING_REFERENCE = "general_text"
# A z-score over fewer distinct random drops than this gets a flag (see z_fields)
MIN_DISTINCT = 3
# A routing_note that starts with this names a reason that another engine call cannot change.
# enrich_rows.py --routing skips rows that carry one, unless --retry-routing is given.
FINAL = "no routing, final: "
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
    # z is a number of the search half, the half the part was selected on, so it is larger than a
    # fresh sample would give. The engine draws the random parts with replacement and rounds each
    # score to 5 decimals, so the stored drops can repeat: random_distinct counts the distinct values,
    # and z_flag marks a z whose deviation rests on fewer than MIN_DISTINCT of them.
    rnd = random_split(random_drops)
    search = spread(rnd["search"])
    out = {"z": rounded(z_score(effect.get(target + "|search"), rnd["search"]), 3),
           "z_heldout": rounded(z_score(effect.get(target + "|heldout"), rnd["heldout"]), 3),
           "random_n": search["n"], "random_mean": rounded(search["mean"], 6),
           "random_std": rounded(search["std"], 6), "random_heldout": None}
    if rnd["heldout"]:
        held = spread(rnd["heldout"])
        out["random_heldout"] = {"n": held["n"], "mean": rounded(held["mean"], 6), "std": rounded(held["std"], 6)}
    out["random_distinct"] = len(set(rnd["search"]))
    out["z_flag"] = None
    if out["z"] is not None and out["random_distinct"] < MIN_DISTINCT:
        out["z_flag"] = "fewer than " + str(MIN_DISTINCT) + " distinct random drops"
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


def items_sha(items):
    # The same digest as the engine's builtin_info: it names one version of a built-in set, whatever
    # its description says. (The engine image holds no copy of this file, so the line is written twice.)
    return hashlib.sha256(json.dumps(items, sort_keys=True).encode()).hexdigest()[:16]


def describe_builtin(data):
    # data: the content of a built-in skill file. Returns what the engine's builtin_info reports.
    described = {}
    for name in data:
        described[name] = {"description": data[name].get("description"), "dataset": data[name].get("dataset")}
        if isinstance(data[name].get("items"), dict):
            described[name]["items_sha"] = items_sha(data[name]["items"])
    return described


def builtin_sources(described, builders):
    # described: {name: {"description": ..., "dataset": ... or absent, "items_sha": ... or absent}},
    # read from a built-in skill file or reported by the engine. The file holds a description and no
    # dataset id, so the ids come from the builder, and only when the builder's description is the one
    # in the file. If they differ, the file was built by another version of build_data.py and the ids
    # are left empty.
    out = {}
    for name in described:
        desc = described[name].get("description")
        datasets = []
        if described[name].get("dataset"):
            datasets = [dataset_id(described[name]["dataset"])]
        elif name in builders and builders[name]["description"] == desc:
            datasets = list(builders[name]["datasets"])
        out[name] = {"source": "builtin", "description": desc, "datasets": datasets}
        if described[name].get("items_sha"):
            out[name]["items_sha"] = described[name]["items_sha"]
    return out


def local_builtin():
    return builtin_sources(describe_builtin(read_json(common.BUILTIN_SKILLS)), local_builders())


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
    # "description"}}, and "hash_candidates" when more than one frozen test of that name is known
    # (the hash is None then); builtin: the output of builtin_sources; pool: the output of local_pool, or
    # None to read the pool file only when a name is in neither of the others. An agent skill
    # cannot take a built-in name (skills.define refuses it), and the miner scores agent and built-in
    # skills only, so the order agent, built-in, pool is the order of what a row can hold.
    held = {"pool": pool}

    def lookup(name):
        if name in agent:
            spec = agent[name]
            out = {"source": "agent", "path": spec.get("path"), "hash": spec.get("hash"),
                   "description": spec.get("description"), "datasets": []}
            if spec.get("hash_candidates"):
                out["hash_candidates"] = list(spec["hash_candidates"])
            return out
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
    # (the engine could not be asked; its copy can be older than the local file), "git history"
    # (backfill: the one version of the built-in file the engine can have held at the row's creation
    # time), or "git history, version uncertain" (backfill: see merge_uncertain).
    info = lookup(target)
    out = {"skill": target, "source": info["source"]}
    for key in ["path", "hash", "description", "dataset", "datasets", "items_sha"]:
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
            if other.get("hash_candidates"):
                # lanes on one model can each freeze a test of this name: which one the row's lane
                # used is not known, so no hash is given
                out["off_target_data"][name]["hash_candidates"] = other["hash_candidates"]
        else:
            out["off_target_data"][name] = {"source": other["source"], "description": other.get("description"),
                                            "datasets": other.get("datasets") or []}
            if other.get("items_sha"):
                out["off_target_data"][name]["items_sha"] = other["items_sha"]
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


def path_check(path, row_hash):
    # For an agent skill: is the frozen test file at `path` the test the row was selected on? Lanes
    # on one model share mine/skills/<model>/, so a later lane can replace a file of the same name.
    # path_matches is True or False, or None when the file cannot be read (path_note says why).
    try:
        with open(path) as f:
            found = json.load(f).get("hash")
    except (OSError, ValueError, AttributeError, TypeError) as e:
        return {"path_matches": None, "path_note": "the frozen test file could not be read: " + type(e).__name__}
    if found == row_hash:
        return {"path_matches": True}
    return {"path_matches": False, "path_hash": found,
            "path_note": "the file at path holds another test of this name (hash " + str(found)
                         + "), not the test this row was selected on"}


def union(lists):
    out = []
    for values in lists:
        for v in values or []:
            if v not in out:
                out.append(v)
    return out


def merge_uncertain(per, labels):
    # Backfill only. per: the selection_data of one row under each version of the built-in file that
    # the engine can have held when the row was made, oldest first (2 or more). labels: for each
    # version {"commit", "time_from", "since"}. The engine uses the file from its last deploy, and the
    # deploy time is not in git, so no single version is written as fact: a skill on which the versions
    # differ gets description None and the datasets of all of them, reported_benchmarks is the union
    # (a benchmark that any of the versions used is reported), and builtin_candidates lists what each
    # version says.
    out = dict(per[-1])
    out["off_target_data"] = dict(per[-1]["off_target_data"])
    differ = []
    if out["source"] == "builtin":
        keys = ["description", "datasets", "items_sha"]
        if any([p.get(k) for k in keys] != [per[0].get(k) for k in keys] for p in per):
            differ.append(out["skill"])
            out["description"] = None
            out["datasets"] = union(p.get("datasets") for p in per)
            if "items_sha" in out:
                out["items_sha"] = None
    for name in out["off_target_data"]:
        entries = [p["off_target_data"].get(name) for p in per]
        if all(e == entries[0] for e in entries):
            continue
        differ.append(name)
        sources = union([[(e or {}).get("source")] for e in entries])
        out["off_target_data"][name] = {"source": sources[0] if len(sources) == 1 else None, "description": None,
                                        "datasets": union((e or {}).get("datasets") for e in entries),
                                        "uncertain": True}
    for key in ["pool", "pool_match"]:
        if any(p.get(key) != per[0].get(key) for p in per):
            out[key] = None
    if any(p.get("reported_benchmarks") is None for p in per):
        out["reported_benchmarks"] = None
    else:
        out["reported_benchmarks"] = union(p["reported_benchmarks"] for p in per)
    out["builtin_candidates"] = []
    for i in range(len(per)):
        skills = {}
        for name in differ:
            if name == out["skill"]:
                entry = per[i]
            else:
                entry = per[i]["off_target_data"].get(name) or {}
            skills[name] = {"description": entry.get("description"), "datasets": entry.get("datasets") or []}
            if entry.get("items_sha"):
                skills[name]["items_sha"] = entry["items_sha"]
        out["builtin_candidates"].append(dict(labels[i], skills=skills))
    commits = ", ".join(str(label.get("commit")) for label in labels)
    if differ:
        out["builtin_note"] = ("the built-in file the engine held for this row is not known: one of commits " + commits
                               + ". They differ on " + ", ".join(differ) + "; reported_benchmarks is the union over"
                               + " them. Pass --live-since <commit>=<deploy time> to enrich_rows.py to settle it.")
    else:
        out["builtin_note"] = ("the built-in file the engine held for this row is not known: one of commits " + commits
                               + ". They agree on every built-in skill of this row.")
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


def client_unsupported(layers_detail, layer):
    # Why the router picks of this layer cannot be counted, decided from the layer list that the
    # engine's info_summary returned (state["structure"]["layers_detail"]), so no engine call is
    # paid for an answer that is known. None when the layer can be counted or the list is missing.
    if not isinstance(layers_detail, list) or not layers_detail:
        return None
    import adapter

    info = {"n_layers": len(layers_detail), "layers": layers_detail}
    return adapter.expert_count_layers(info, [int(layer)])[1].get(str(int(layer)))


def routing_fields(indices, layer, target, result, general=ROUTING_REFERENCE):
    # result: what the engine's expert_pick_rates returned for this layer. Returns (routing, note).
    # routing is None when the layer cannot be counted or when no pick rate of the target skill came
    # back, and the note says why. A routing that lacks one half gets a note too, so a null rate in
    # it is never read as a measured one.
    reason = (result.get("unsupported") or {}).get(str(layer))
    if reason:
        return None, FINAL + reason
    experts = {}
    tokens = {}
    n_items = {}
    gaps = []
    skill_found = False
    for label, name in [("skill", target), ("general", general)]:
        for split in SPLITS:
            key = name + "|" + split
            counted = layer_rates(result, name, split, layer)
            if not counted or not counted.get("rates"):
                why = "the engine does not hold this skill" if key in (result.get("missing") or []) else "no rates"
                gaps.append(key + " (" + why + ")")
                continue
            top_k = counted.get("top_k")
            total = math.fsum(counted["rates"])
            if is_number(top_k) and abs(total - top_k) > 0.01 * max(top_k, 1):
                # each counted token picks top-k experts, so the rates of a layer add up to top-k;
                # another sum means the hook did not see one row of indices per token
                return None, (FINAL + "the pick rates of layer " + str(layer) + " on " + key + " add up to "
                              + str(round(total, 4)) + ", not to the router's top-k " + str(top_k))
            tokens[key] = counted["tokens"]
            n_items[key] = result["rates"][key].get("n_items")
            if label == "skill":
                skill_found = True
    if not skill_found:
        if not tokens:
            return None, "no routing: the engine returned no pick rates for layer " + str(layer)
        return None, "no routing: no pick rates for the target skill: " + "; ".join(gaps)
    for i in sorted(int(x) for x in indices):
        per = {}
        for split in SPLITS:
            rates = {}
            for label, name in [("skill", target), ("general", general)]:
                rates[label] = None
                if name + "|" + split in tokens:
                    counted = layer_rates(result, name, split, layer)
                    if i < len(counted["rates"]):
                        rates[label] = counted["rates"][i]
            per[split] = {"skill_rate": rates["skill"], "general_rate": rates["general"],
                          "log_ratio": log_ratio(rates["skill"], rates["general"])}
        experts[str(i)] = per
    note = None
    if gaps:
        note = "routing is partial: no pick rates for " + "; ".join(gaps)
    return {"layer": int(layer), "skill": target, "general": general, "n_items": n_items, "tokens": tokens,
            "experts": experts}, note
