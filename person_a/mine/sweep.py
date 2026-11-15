# Sweep miner: finds large parts for many skills on one donor model, with no LLM in the loop.
#   .venv/bin/python mine/sweep.py --short <short_name> --skills law,medicine,math
#   .venv/bin/python mine/sweep.py --short <short_name> --skills all --max-screens 200 --max-hours 4
#   .venv/bin/python mine/sweep.py --short <short_name> --skills law --rank-only      # print the plan, screen nothing
#
# Why. The LLM agents pick candidates by reading weights. A post on auto weight-space programming (Naik,
# Liu, Gangal, Nov 2026) reports that attribution patching predicts a part's measured effect and that
# labels read from weights do not. The compiler reports that single neurons and groups of 2 to 4 heads
# do not beat random parts in a compile; it wants larger units. A ranking by attribution patching orders
# every unit of a model in a few passes, so only the best candidates need a real switch-off test.
#
# Steps:
#   1. Rank. For each skill and each part kind the model has (mlp_cluster, attn_heads, expert), the
#      engine method attribution_units returns the best units of every layer: the estimated rise in the
#      skill's loss when a unit is zeroed, next to the same number on general_text. A unit's score is
#      the skill value minus the general_text value (a negative general_text value counts as 0, the rule
#      the engine uses for attribution scans). The ranking reads the "search" half only.
#      Quantized (FP8) models cannot run the backward pass: attribution_units refuses, and the sweep uses
#      activation_units instead (forward passes only). The rows then say "activation" in evidence_used.
#   2. Candidates. MLP clusters of 64, 256, and 1024 units, head groups of 4 and 8, and expert sets of
#      2, 4, and 8, each in the 3 layers with the highest summed score. A size is skipped when it needs
#      more than half of the layer (the gate needs random parts of the same size) or more units than
#      have a skill-specific score. Layer blocks come from atlas/pool_entries/<org>__<name>.json (written
#      by mine/maps.py) when that file exists and has the skill.
#   3. Intervene. Each candidate goes through Miner.screen, the same path the agents use: one ablation,
#      GATE_N_RANDOM random parts of the same size, and the light gate. A candidate passes when the
#      light gate passes AND its z-score against the random parts is at least Z_MIN on the search half.
#   4. Write. Passing parts go to catalog/<model>.jsonl through Miner.write_part with tier "screened".
#
# Order and limits. Round r holds the r-th best candidate of every (skill, kind) pair; inside a round the
# candidate with the highest score goes first. So a run that stops at --max-screens or --max-hours holds
# the best candidate of every skill and kind before any second-best one. "--order global" sorts by
# score alone. A rerun skips finished work: rankings are kept in logs/<model>/<lane>/ and screens in the
# lane state (Miner.screen caches by part key and target).
#
# Guard skills. The agents score every built-in and agent skill in each ablation. A sweep over many
# skills would then score dozens of sets per screen, so each screen scores only: the target, general_text,
# and N_GUARDS guard skills from other domains (GUARDS below, one per domain; the guard of the target's
# own domain is left out, because a real medicine part also moves a second medicine set). A skill keeps
# the guard set of its first screen, so cached screens stay comparable. The gate thresholds are unchanged.
#
# Running next to an LLM agent run on the same model is safe:
#   - State. The sweep sets WSP_LANE to "sweep" (plus --lane), so its state is
#     logs/<model>/state.lanesweep.json and never the agents' state.json or state.lane<N>.json. It
#     copies only the pinned revision and the structure from the main state, so it loads the same engine
#     container as the agents; it does not copy their screens, which used another guard set. A lock
#     file stops two sweeps from sharing one lane.
#   - Catalog. Rows are appended by Miner.write_part, which holds catalog/<model>.jsonl.lock while it
#     picks the id and appends. The agents hold the same lock, so ids never collide and lines never mix.
#   - Engine. Calls from both runs queue on the same deployed class; score caches are keyed by set hash.
#
# Catalog rows. All existing fields keep their meaning. The sweep adds one field, "sweep" (how the part
# was ranked, its z-score, the guard skills), and for a pool skill it writes skill.source "pool" with the
# dataset, because Miner would call it "builtin". One line per new screen also goes to
# logs/<model>/<lane>/results.jsonl, passed or not.
#
# The engine must be deployed from a branch that has attribution_units and activation_units.
import argparse
import fcntl
import json
import math
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common  # noqa: E402
import client  # noqa: E402

REPO = os.path.dirname(common.ROOT)
POOL_SKILLS = os.path.join(HERE, "data", "pool_skills.json")
POOL_ENTRIES = os.path.join(REPO, "atlas", "pool_entries")
REFERENCE = "general_text"

Z_MIN = 2.0
# The engine rounds relative drops to 4 decimals, so a spread below this is not measured. Without the
# floor, five random parts that all round to the same drop would give an infinite z-score.
STD_FLOOR = 1.0e-4
SIZES = {"mlp_cluster": [64, 256, 1024], "attn_heads": [4, 8], "expert": [2, 4, 8]}
TOP_LAYERS = 3
RANK_ITEMS = 24
KIND_KEYS = {"mlp_cluster": "mlp", "attn_heads": "attn", "expert": "experts"}
KIND_TEXT = {"mlp_cluster": "MLP cluster of %d units", "attn_heads": "group of %d attention heads",
             "expert": "set of %d experts"}
ALL_KINDS = ["mlp_cluster", "attn_heads", "expert", "block"]

# Guard skills: one per domain, in this order. A screen uses the first N_GUARDS that are not in the
# target's domain and that the model can do (base margin over shuffled answers, as for built-in skills).
N_GUARDS = 5
GUARDS = ["math", "code", "medicine", "law", "prose_zh", "physics", "facts"]
DOMAINS = {
    "math": ["math", "mathematics_theory"],
    "code": ["code", "sql", "tool_calling", "computer_science", "code_typescript", "code_java", "code_cpp",
             "code_csharp", "code_rust", "code_php", "code_swift", "code_shell"],
    "medicine": ["medicine", "clinical_dialogue", "clinical_knowledge", "biology"],
    "society": ["law", "finance", "economics_business", "history", "philosophy_ethics", "psychology_social",
                "politics_geography"],
    "language": ["multilingual", "prose_de", "prose_fr", "prose_es", "prose_zh", "prose_ja", "prose_hi",
                 "prose_ar", "prose_ru"],
    "science": ["physics", "chemistry", "science_papers"],
    "general": ["facts", "instruction", "creative_writing"],
}


# ---------------- skills and guards ----------------

def load_skill_specs():
    # name -> {"spec": payload entry, "source": "builtin" or "pool", "id": hash that names the set,
    #          "info": fields for the catalog row}
    # Built-in sets live in the engine image and are sent by name. Pool sets are sent inline (the same
    # way mine/maps.py sends them); a pool set has no control set.
    specs = {}
    for name in common.BUILTIN_TARGETS + common.CONTROL_ONLY:
        specs[name] = {"spec": {"builtin": True}, "source": "builtin", "id": "builtin-" + name, "info": None}
    if os.path.exists(POOL_SKILLS):
        with open(POOL_SKILLS) as f:
            pool = json.load(f)
        for name in pool:
            if name in specs:
                print("warning: pool skill '" + name + "' has the name of a built-in skill; the built-in one is used")
                continue
            digest = "pool-" + pool[name]["hash"]
            specs[name] = {"spec": {"hash": digest, "items": pool[name]["items"], "control": None},
                           "source": "pool", "id": digest,
                           "info": {"source": "pool", "path": os.path.relpath(POOL_SKILLS, common.ROOT),
                                    "hash": pool[name]["hash"], "dataset": pool[name].get("dataset"),
                                    "description": pool[name].get("description")}}
    return specs


def parse_skills(text, specs):
    if text.strip() == "all":
        return [n for n in specs if n != REFERENCE]
    names = []
    for n in text.split(","):
        n = n.strip()
        if n == "" or n in names:
            continue
        if n == REFERENCE or n not in specs:
            raise SystemExit("unknown skill '" + n + "'. Known: " + ", ".join(k for k in specs if k != REFERENCE))
        names.append(n)
    if not names:
        raise SystemExit("--skills needs at least one skill, or 'all'")
    return names


def domain_of(name):
    for d in DOMAINS:
        if name in DOMAINS[d]:
            return d
    return None


def guards_for(target, usable):
    # the first N_GUARDS guard skills from other domains that exist and that the model can do
    out = []
    for g in GUARDS:
        if g == target or g not in usable:
            continue
        if domain_of(target) is not None and domain_of(g) == domain_of(target):
            continue
        out.append(g)
        if len(out) == N_GUARDS:
            break
    return out


def build_payload(specs, target, guards):
    skills = {}
    for name in [target, REFERENCE] + list(guards):
        skills[name] = specs[name]["spec"]
    return {"skills": skills}


# ---------------- z-score ----------------

def z_score(target_drop, random_drops):
    # How many standard deviations the candidate's drop is above the drops of the random same-size parts
    # (sample standard deviation). None when the drop is missing or fewer than 2 random drops exist.
    vals = []
    for r in random_drops or []:
        if r is not None:
            vals.append(float(r))
    if target_drop is None or len(vals) < 2:
        return None
    mean = sum(vals) / len(vals)
    var = sum((v - mean) ** 2 for v in vals) / (len(vals) - 1)
    return round((float(target_drop) - mean) / max(math.sqrt(var), STD_FLOOR), 2)


# ---------------- candidates ----------------

def specific_units(layer_rank):
    # units of one layer with a skill-specific score, best first: [unit, gain, skill value, general value].
    # gain = score above the floor (0 for attribution, 1 for the activation ratio).
    out = []
    seen = set()
    width = int(layer_rank["width"])
    for i in range(len(layer_rank["units"])):
        unit = int(layer_rank["units"][i])
        gain = float(layer_rank["score"][i]) - float(layer_rank.get("floor", 0.0))
        if gain <= 0 or unit < 0 or unit >= width or unit in seen:
            continue
        seen.add(unit)
        out.append([unit, gain, float(layer_rank["skill"][i]), float(layer_rank["general"][i])])
    return out


def fitting_sizes(sizes, width, n_specific):
    # A size fits when the layer has that many skill-specific units and the part is at most half of the
    # layer: the gate compares against random parts of the same size, which need room to differ.
    out = []
    for k in sorted(set(sizes)):
        if k >= 1 and 2 * k <= width and k <= n_specific:
            out.append(k)
    return out


def layer_scores(rank, sizes):
    # per layer: the summed gain of its largest fitting candidate. Layers where no size fits are left out.
    scores = {}
    for key in rank["layers"]:
        units = specific_units(rank["layers"][key])
        fit = fitting_sizes(sizes, int(rank["layers"][key]["width"]), len(units))
        if fit:
            scores[int(key)] = sum(u[1] for u in units[:fit[-1]])
    return scores


def unit_candidates(skill, kind, rank, sizes, top_layers):
    # rank: the result of the engine method attribution_units or activation_units
    scores = layer_scores(rank, sizes)
    chosen = sorted(scores, key=lambda l: (-scores[l], l))[:top_layers]
    out = []
    for layer in chosen:
        entry = rank["layers"][str(layer)]
        units = specific_units(entry)
        for k in fitting_sizes(sizes, int(entry["width"]), len(units)):
            top = units[:k]
            out.append({"skill": skill, "kind": kind, "size": k, "layer": layer,
                        "part": {"kind": kind, "layer": layer, "indices": sorted(u[0] for u in top)},
                        "rank_method": rank["method"], "rank_score": float("%.6g" % sum(u[1] for u in top)),
                        "skill_effect": float("%.6g" % sum(u[2] for u in top)),
                        "general_effect": float("%.6g" % sum(u[3] for u in top))})
    return out


def block_candidates(skill, entry, n_layers):
    # entry: an atlas/pool_entries file from mine/maps.py. Its layer scores are measured: the loss rise on
    # the skill when one layer is skipped, minus the same on general_text (nats). A block's score is the
    # sum over its layers. Blocks longer than half of the model are skipped (no room for random blocks).
    if not entry or int(entry.get("layers") or -1) != n_layers:
        return []
    lm = (entry.get("layer_maps") or {}).get(skill)
    if not lm:
        return []
    out = []
    seen = set()
    for pair in lm.get("blocks") or []:
        first, last = int(pair[0]), int(pair[1])
        length = last - first + 1
        if first < 0 or last >= n_layers or length < 1 or 2 * length > n_layers or (first, last) in seen:
            continue
        seen.add((first, last))
        out.append({"skill": skill, "kind": "block", "size": length, "layers": [first, last],
                    "part": {"kind": "block", "layers": list(range(first, last + 1))},
                    "rank_method": "layer_map", "rank_score": float("%.6g" % sum(lm["scores"][first:last + 1])),
                    "skill_effect": float("%.6g" % sum(lm["influence"][first:last + 1])), "general_effect": None})
    return out


def model_kinds(structure):
    # the unit kinds this model has. Experts kept as a ModuleList cannot be ranked by the engine hooks.
    kinds = []
    for kind in ["mlp_cluster", "attn_heads", "expert"]:
        for e in structure["layers_detail"]:
            sub = e.get(KIND_KEYS[kind])
            if sub and not (kind == "expert" and sub.get("style") == "module_list"):
                kinds.append(kind)
                break
    return kinds


# ---------------- order and cut-off ----------------

def cand_key(c):
    return client.part_key(c["part"]) + "|" + c["skill"]


def order_candidates(cands, mode="round"):
    # "round": round r holds the r-th best candidate of every (skill, kind) pair, and inside a round the
    # highest score goes first. "global": highest score first, whatever the skill and kind.
    # With attribution, unit scores and block scores are both nats of loss, so they can be compared. With
    # the activation fallback the unit scores are ratios, and the order between units and blocks is rough.
    groups = {}
    for c in cands:
        groups.setdefault((c["skill"], c["kind"]), []).append(c)
    out = []
    for g in groups:
        rows = sorted(groups[g], key=lambda c: (-c["rank_score"], c["size"], cand_key(c)))
        for i in range(len(rows)):
            c = dict(rows[i])
            c["round"] = i
            out.append(c)
    if mode == "global":
        out.sort(key=lambda c: (-c["rank_score"], c["skill"], c["kind"], cand_key(c)))
    else:
        out.sort(key=lambda c: (c["round"], -c["rank_score"], c["skill"], c["kind"], cand_key(c)))
    return out


def plan_work(ordered, cached_keys, max_screens):
    # Splits the ordered candidates into the work of this run and the rest. A candidate with a cached
    # screen costs nothing, so it is always kept; new screens stop at max_screens.
    work = []
    left = []
    new = 0
    for c in ordered:
        if cand_key(c) in cached_keys:
            work.append((c, True))
        elif new < max_screens:
            work.append((c, False))
            new += 1
        else:
            left.append(c)
    return work, left


# ---------------- catalog text ----------------

def label_for(c):
    if c["kind"] == "block":
        return c["skill"] + ", layer block " + str(c["layers"][0]) + " to " + str(c["layers"][1]) + " (layer-map-ranked)"
    return (c["skill"] + ", " + KIND_TEXT[c["kind"]] % c["size"] + ", layer " + str(c["layer"])
            + " (" + c["rank_method"] + "-ranked)")


def rationale_for(c, n_items):
    start = "No weight reading. "
    if c["kind"] == "block":
        return (start + "Block from the model's layer map (mine/maps.py): loss rise on '" + c["skill"]
                + "' when one layer is skipped, minus the same on general_text, summed over layers "
                + str(c["layers"][0]) + " to " + str(c["layers"][1]) + " (" + str(c["rank_score"]) + " nats).")
    what = str(c["size"]) + " units of layer " + str(c["layer"])
    if c["rank_method"] == "attribution":
        return (start + "Ranked by attribution patching on " + str(n_items) + " search items of '" + c["skill"]
                + "': the " + what + " with the highest estimated loss rise on the skill minus general_text (sum "
                + str(c["rank_score"]) + " nats).")
    return (start + "Ranked by activation on " + str(n_items) + " search items of '" + c["skill"] + "': the " + what
            + " with the highest ratio of mean activation (experts: router pick rate) on the skill to general_text. "
            "The model is loaded quantized, so attribution patching was not available.")


def evidence_for(c):
    if c["kind"] == "block":
        return ["ablation"]
    return [c["rank_method"], "ablation"]


def add_row_fields(record, extra):
    # Adds the sweep's fields to a catalog row that Miner built. It never replaces a field the row has,
    # with one exception: Miner knows only built-in and agent skills, so for a pool skill the "skill"
    # entry gets the pool's source, dataset, hash, and description (the name stays).
    for k in extra:
        if k != "skill" and k not in record:
            record[k] = extra[k]
    if extra.get("skill"):
        record["skill"] = dict(record.get("skill") or {}, **extra["skill"])
    return record


# ---------------- miner ----------------

def lane_dir(short):
    return os.path.join(common.logs_dir(short), os.environ.get("WSP_LANE") or "sweep")


def seed_lane_state(short):
    # A new sweep lane takes the pinned revision and the structure from the model's main state, so it
    # uses the same engine container as a live agent run. It does not take the screens (another guard set).
    lane = common.state_path(short)
    main = os.path.join(common.logs_dir(short), "state.json")
    if lane == main or os.path.exists(lane) or os.path.exists(lane + ".bak"):
        return
    src = common.read_json_safe(main, {})
    seed = {}
    for k in ["revision", "model", "structure"]:
        if k in src:
            seed[k] = src[k]
    if seed:
        common.write_json_atomic(lane, seed)


def lock_lane(short):
    # one sweep per lane: two processes would overwrite each other's state file
    path = common.state_path(short) + ".lock"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    handle = open(path, "w")
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        raise SystemExit("another sweep already runs on " + short + " in this lane; give this one --lane <name>")
    return handle  # the caller keeps it open; the lock ends with the process


class SweepMiner(client.Miner):
    # Miner with the sweep's skills. screen() and write_part() are the inherited ones, so the gate, the
    # screen cache, and the catalog lock are the same code the agents use.

    def __init__(self, short, specs):
        self.specs = specs
        self.active_payload = {"skills": {}}
        self.row_extra = {}
        client.Miner.__init__(self, short)

    def payload(self, include_skipped=False):
        return self.active_payload

    def target_skills(self):
        return [n for n in self.active_payload["skills"] if n != REFERENCE]

    def base_margins(self, names, chunk=12):
        # chance check, once per skill set: the model must prefer each item's own answer over the answer
        # of another item. Kept in the lane state.
        done = self.state.setdefault("sweep_base", {})
        missing = [n for n in names if done.get(n, {}).get("id") != self.specs[n]["id"]]
        for i in range(0, len(missing), chunk):
            part = missing[i:i + chunk]
            res = self.call("base", {"skills": dict((n, self.specs[n]["spec"]) for n in part)})
            for n in part:
                done[n] = {"id": self.specs[n]["id"], "margin": res["margin_vs_shuffled"].get(n),
                           "score": (res["scores"].get(n + "|search") or {}).get("score")}
            self.save()
        return done

    def guards(self, target, usable):
        kept = self.state.setdefault("sweep_guards", {})
        if target not in kept:
            kept[target] = guards_for(target, usable)
            self.save()
        return kept[target]

    def rank(self, skill, kind, top_n, n_items):
        # attribution first; a quantized model refuses it, and then every ranking of this model uses activation
        payload = build_payload(self.specs, skill, [])
        method = self.state.get("sweep_method")
        res = None
        if method != "activation":
            try:
                res = self.call("attribution_units", payload, kind, skill, [], top_n, n_items,
                                common.scan_token_budget(self.params_b))
                method = "attribution"
            except Exception as e:
                if "needs bf16 weights" not in str(e):
                    raise
                method = "activation"
        if res is None:
            res = self.call("activation_units", payload, kind, skill, [], top_n, n_items)
        if self.state.get("sweep_method") != method:
            self.state["sweep_method"] = method
            self.save()
        common.log_event(self.short, method + "_units", {"kind": kind, "skill": skill, "top_n": top_n},
                         {"seconds": res["seconds"], "layers": len(res["layers"])})
        return res

    def ranked_candidates(self, skill, kind, sizes, top_layers, n_items):
        # candidates of one (skill, kind), from the lane's file when the same ranking was done before
        path = os.path.join(lane_dir(self.short), "rank_" + common.clean_name(skill) + "_" + kind + ".json")
        params = {"revision": self.revision, "skill_id": self.specs[skill]["id"], "sizes": sorted(set(sizes)),
                  "top_layers": top_layers, "n_items": n_items}
        kept = common.read_json_safe(path, {})
        if kept.get("params") == params:
            return kept["candidates"], True
        rank = self.rank(skill, kind, max(sizes), n_items)
        cands = unit_candidates(skill, kind, rank, sizes, top_layers)
        scores = layer_scores(rank, sizes)
        common.write_json_atomic(path, {"params": params, "method": rank["method"], "n_tokens": rank["n_tokens"],
                                        "seconds": rank["seconds"],
                                        "layer_scores": dict((str(l), float("%.6g" % scores[l])) for l in sorted(scores)),
                                        "candidates": cands})
        return cands, False

    def screen_candidate(self, c, guards, z_min, n_items):
        # Miner.screen (ablation, random parts, light gate), then the z-score rule; writes the row on a pass
        skill = c["skill"]
        self.active_payload = build_payload(self.specs, skill, guards)
        res = self.screen(c["part"], skill)
        target_drop = res["drops"].get(skill + "|search")
        z = z_score(target_drop, res["random_drops"])
        passed = bool(res["gate"]["passed"]) and z is not None and z >= z_min
        reasons = list(res["gate"]["reasons"])
        if res["gate"]["passed"] and not passed:
            reasons.append("z-score " + str(z) + " < " + str(z_min))
        out = {"passed": passed, "gate_passed": bool(res["gate"]["passed"]), "z_search": z,
               "target_drop_search": target_drop, "reasons": reasons, "seconds": res["seconds"], "catalog_id": None}
        if passed:
            self.row_extra = {"sweep": {"rank_method": c["rank_method"], "rank_score": c["rank_score"],
                                        "rank_items": None if c["kind"] == "block" else n_items,
                                        "reference": REFERENCE, "guards": list(guards), "z_search": z, "z_min": z_min}}
            if self.specs[skill]["info"]:
                self.row_extra["skill"] = self.specs[skill]["info"]
            try:
                rec = self.write_part(c["part"], label_for(c), skill, res, evidence_for(c),
                                      rationale_for(c, n_items), "screened")
            finally:
                self.row_extra = {}
            out["catalog_id"] = rec["id"]
        return out

    def _write_part_locked(self, *args, **kwargs):
        # Runs under the catalog lock that Miner.write_part holds. Miner builds the row and appends it with
        # common.append_jsonl; for that one call the append also adds the sweep's fields (add_row_fields),
        # so the row is written once and complete. The lock makes this the only writer at that moment.
        # The arguments go to Miner unchanged, whatever they are: another branch adds arguments there.
        extra = self.row_extra
        catalog = common.catalog_path(self.short)
        real_append = common.append_jsonl

        def append(path, record):
            if path == catalog:
                add_row_fields(record, extra)
            real_append(path, record)

        common.append_jsonl = append
        try:
            return client.Miner._write_part_locked(self, *args, **kwargs)
        finally:
            common.append_jsonl = real_append


# ---------------- summary ----------------

def summarize(rows):
    # rows: one dict per candidate: {"skill", "kind", "screened": bool, "passed": bool}
    total = {"ranked": 0, "screened": 0, "passed": 0}
    by_kind = {}
    by_skill = {}
    for r in rows:
        for table, key in [(by_kind, r["kind"]), (by_skill, r["skill"])]:
            if key not in table:
                table[key] = {"ranked": 0, "screened": 0, "passed": 0}
        for counts in [total, by_kind[r["kind"]], by_skill[r["skill"]]]:
            counts["ranked"] += 1
            if r["screened"]:
                counts["screened"] += 1
            if r["passed"]:
                counts["passed"] += 1
    return {"total": total, "by_kind": by_kind, "by_skill": by_skill}


def print_summary(summary, notes):
    t = summary["total"]
    print("")
    print("sweep summary:", t["ranked"], "candidates ranked,", t["screened"], "screened,", t["passed"], "passed")
    for title, table in [("kind", summary["by_kind"]), ("skill", summary["by_skill"])]:
        width = max([len(title)] + [len(k) for k in table])
        print("  " + title.ljust(width) + "  ranked  screened  passed")
        for key in sorted(table):
            c = table[key]
            print("  " + key.ljust(width) + "  " + str(c["ranked"]).rjust(6) + "  " + str(c["screened"]).rjust(8)
                  + "  " + str(c["passed"]).rjust(6))
    for line in notes:
        print(line)


def int_list(text):
    return [int(x) for x in text.split(",") if x.strip()]


def parse_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--short", required=True, help="short name in registry/donors.csv")
    ap.add_argument("--skills", required=True, help="comma-separated skill names, or 'all'")
    ap.add_argument("--max-screens", type=int, default=60, help="new screens in this run (cached ones are free)")
    ap.add_argument("--max-hours", type=float, default=3.0)
    ap.add_argument("--kinds", default=",".join(ALL_KINDS), help="part kinds to mine")
    ap.add_argument("--mlp-sizes", default=",".join(str(k) for k in SIZES["mlp_cluster"]))
    ap.add_argument("--head-sizes", default=",".join(str(k) for k in SIZES["attn_heads"]))
    ap.add_argument("--expert-sizes", default=",".join(str(k) for k in SIZES["expert"]))
    ap.add_argument("--top-layers", type=int, default=TOP_LAYERS, help="layers per skill and kind")
    ap.add_argument("--rank-items", type=int, default=RANK_ITEMS, help="search items per ranking pass")
    ap.add_argument("--z-min", type=float, default=Z_MIN)
    ap.add_argument("--min-margin", type=float, default=client.BUILTIN_MIN_MARGIN,
                    help="skip a skill when the model's margin over shuffled answers is below this (nats)")
    ap.add_argument("--order", default="round", choices=["round", "global"])
    ap.add_argument("--lane", default="", help="suffix for a second sweep on the same model at the same time")
    ap.add_argument("--rank-only", action="store_true", help="rank and print the plan; screen nothing")
    args = ap.parse_args(argv)
    args.kind_list = [k.strip() for k in args.kinds.split(",") if k.strip()]
    for k in args.kind_list:
        if k not in ALL_KINDS:
            raise SystemExit("unknown kind '" + k + "'. Known: " + ", ".join(ALL_KINDS))
    args.sizes = {"mlp_cluster": int_list(args.mlp_sizes), "attn_heads": int_list(args.head_sizes),
                  "expert": int_list(args.expert_sizes)}
    return args


def run_sweep(miner, args):
    # The whole sweep on a ready miner: base check, ranking, order, screens, summary. Returns the summary.
    start = time.time()
    deadline = start + args.max_hours * 3600.0
    specs = miner.specs
    wanted = parse_skills(args.skills, specs)
    structure = miner.structure()
    n_layers = int(structure["n_layers"])
    kinds = [k for k in model_kinds(structure) if k in args.kind_list and args.sizes[k]]
    notes = []

    # 1. which skills the model can do
    margins = miner.base_margins([n for n in specs if n in wanted or n in GUARDS or n == REFERENCE])
    usable = [n for n in margins if margins[n]["margin"] is not None and margins[n]["margin"] >= args.min_margin]
    targets = [n for n in wanted if n in usable]
    skipped = [n for n in wanted if n not in usable]
    if skipped:
        notes.append("skipped, model near chance (margin < " + str(args.min_margin) + " nats): " + ", ".join(skipped))

    # 2. rank and build candidates
    entry = None
    if "block" in args.kind_list:
        entry_path = os.path.join(POOL_ENTRIES, miner.hf_id.replace("/", "__") + ".json")
        if os.path.exists(entry_path):
            entry = common.read_json_safe(entry_path, None)
        else:
            notes.append("no layer blocks: " + os.path.relpath(entry_path, REPO) + " does not exist (run mine/maps.py)")
    cands = []
    out_of_time = False
    try:
        for skill in targets:
            for kind in kinds:
                if time.time() > deadline:
                    out_of_time = True
                    break
                found, kept = miner.ranked_candidates(skill, kind, args.sizes[kind], args.top_layers, args.rank_items)
                cands.extend(found)
                print("ranked", skill, kind, "->", len(found), "candidates", "(kept from an earlier run)" if kept else "",
                      flush=True)
            if out_of_time:
                break
            cands.extend(block_candidates(skill, entry, n_layers))
    except Exception as e:
        if "attribution_units" in str(e) or "activation_units" in str(e):
            print("The deployed engine may not have the unit-ranking methods; deploy mine/engine_app.py from a "
                  "branch that has attribution_units.")
        raise
    if out_of_time:
        notes.append("time limit reached while ranking: not every skill was ranked")

    # 3. order, cut, screen
    ordered = order_candidates(cands, args.order)
    work, left = plan_work(ordered, set(miner.state.get("screens", {}).keys()), args.max_screens)
    rows = {}
    for c in ordered:
        rows[cand_key(c)] = {"skill": c["skill"], "kind": c["kind"], "screened": False, "passed": False}
    if args.rank_only:
        print("plan (" + args.order + " order); nothing is screened or written with --rank-only:")
        for c, cached in work:
            print("  round", c["round"], "|", label_for(c), "| score", c["rank_score"],
                  "| screen kept from an earlier run" if cached else "")
        if left:
            print("  and", len(left), "more after --max-screens", args.max_screens)
        work = []
        left = []
    results_path = os.path.join(lane_dir(miner.short), "results.jsonl")
    passed_lines = []
    new_screens = 0
    errors = 0
    stop = None
    try:
        for c, cached in work:
            if not cached and time.time() > deadline:
                stop = "time limit"
                break
            guards = miner.guards(c["skill"], usable)
            try:
                res = miner.screen_candidate(c, guards, args.z_min, args.rank_items)
            except Exception as e:
                # one bad candidate must not end the run, but a dead engine must
                errors += 1
                print("screen failed:", label_for(c), "|", repr(e)[:300], flush=True)
                if errors >= 3:
                    stop = "3 screens in a row failed"
                    break
                continue
            errors = 0
            row = rows[cand_key(c)]
            row["screened"] = True
            row["passed"] = res["passed"]
            if res["passed"]:
                passed_lines.append("  " + str(res["catalog_id"]) + " | z " + str(res["z_search"]) + " | target drop "
                                    + str(res["target_drop_search"]))
            if not cached:
                new_screens += 1
                common.append_jsonl(results_path, dict(
                    {"timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"), "skill": c["skill"], "kind": c["kind"],
                     "size": c["size"], "where": c.get("layer", c.get("layers")), "part_key": client.part_key(c["part"]),
                     "rank_method": c["rank_method"], "rank_score": c["rank_score"], "round": c["round"],
                     "guards": guards}, **res))
                print("screen", new_screens, "|", label_for(c), "| drop", res["target_drop_search"], "| z", res["z_search"],
                      "| PASS " + str(res["catalog_id"]) if res["passed"] else "| fail: " + "; ".join(res["reasons"])[:200],
                      flush=True)
    except KeyboardInterrupt:
        stop = "interrupted"

    if stop:
        notes.append("stopped early: " + stop)
    if left:
        notes.append(str(len(left)) + " candidates not screened: --max-screens " + str(args.max_screens) + " reached")
    notes.append("rank method: " + str(miner.state.get("sweep_method")) + " | new screens: " + str(new_screens)
                 + " | engine call seconds: " + str(round(miner.modal_seconds)) + " | wall seconds: "
                 + str(round(time.time() - start)))
    if passed_lines:
        notes.append("passed parts (catalog/" + miner.short + ".jsonl):")
        notes.extend(passed_lines)
    summary = summarize(list(rows.values()))
    summary["new_screens"] = new_screens
    summary["not_screened"] = len([r for r in rows.values() if not r["screened"]])
    summary["stop"] = stop
    print_summary(summary, notes)
    return summary


def main():
    args = parse_args()
    # the sweep's own state file, never an agent's (see the header comment)
    os.environ["WSP_LANE"] = "sweep" + common.clean_name(args.lane, "")
    specs = load_skill_specs()
    parse_skills(args.skills, specs)  # a wrong skill name stops here, before any engine call
    seed_lane_state(args.short)
    lane_lock = lock_lane(args.short)
    miner = SweepMiner(args.short, specs)
    print("sweep", args.short, "|", miner.hf_id, "| revision", miner.revision[:10], "| GPU", miner.gpu,
          "| lane", os.environ["WSP_LANE"], flush=True)
    try:
        run_sweep(miner, args)
    finally:
        lane_lock.close()


if __name__ == "__main__":
    main()
