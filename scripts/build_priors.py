"""Build the frozen Exa priors pack before a run (spec 9.3; docs/research/exa.md 5.1).

Examples, from the repo root:
    .venv/Scripts/python scripts/build_priors.py catalog.jsonl --dry-run
    .venv/Scripts/python scripts/build_priors.py data/pilot_catalog.jsonl
    .venv/Scripts/python scripts/build_priors.py data/pilot_catalog.jsonl --mode replay

The pre-pass plans two kinds of search. All of them use category
"publication", exclude dataset mirrors, and pin the latest published date:
- one "auto" search with highlights (5 results) per group of catalog parts
  that share a donor model, a part kind, a layer band (early, middle or late
  by relative donor depth; "all" for parts without a layer) and a skill (the
  record's trained_on, tidied). Every part in a group gets the group's cites.
- TOPICS: 8 "deep-lite" searches with an output_schema of claims
  (claim, layer_hint, model, strength).

It writes the pack (default data/priors.json) and fills the cache (default
exa_cache/). The pack has sorted keys, LF line endings and no timestamps, so
rebuilding it from the cache with --mode replay gives the same sha256. Both
the pack sha and the catalog_sha256 field are hashed with CRLF read as LF,
so a Windows checkout (core.autocrlf) and a Linux checkout agree.

--dry-run prints the planned queries and a list-price cost estimate. It
needs no key and makes no network call. Live mode refuses to start when the
estimate for calls not yet cached is above --max-cost-dollars. Live mode
reads EXA_API_KEY from the environment (load it with
set -a && . ./.env && set +a); without it, the build uses the cache only and
says so, and writes nothing when the cache holds none of the planned calls.
"""

import argparse
import json
import os
import re
import sys
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from wsp.swarm import literature as lit_mod  # noqa: E402

KIND_PHRASES = {
    "mlp_cluster": "MLP neurons",
    "attn_heads": "attention heads",
    "expert": "mixture-of-experts experts",
    "block": "transformer layers",
    "deletion_mask": "sparse weight deletion masks",
    "task_vector": "task vectors",
    "lora": "LoRA adapters",
}
BAND_ORDER = ["early", "middle", "late", "unknown", "all"]
NOISE_WORDS = ["activations", "sft", "statistics", "train", "data", "texts", "text", "prompts",
               "heal-split", "split"]
COMPONENT_RESULTS = 5
TOPIC_RESULTS = 5

TOPICS = [
    ("model_stitching",
     "Model stitching between language models: do linear map fit scores such as R squared or CKA "
     "predict how well the stitched model performs?"),
    ("layer_placement",
     "At what relative depth should a layer or module taken from one transformer be inserted into "
     "another, and how does a depth mismatch affect accuracy?"),
    ("cross_family_transplant",
     "Transplanting neurons, attention heads or features between language model families such as "
     "Qwen and Llama through learned linear maps"),
    ("arithmetic_layers",
     "Which transformer layers and MLP neurons carry arithmetic and grade-school math reasoning "
     "in large language models?"),
    ("code_layers",
     "Which layers and neurons of code language models carry programming skill, and can that "
     "skill be transferred to another model?"),
    ("heal_after_surgery",
     "How many fine-tuning steps does it take to heal a language model after layer pruning, layer "
     "stacking or model surgery, and which parameters should be trained?"),
    ("glue_only_training",
     "How much task skill can small trained adapters or interface layers learn on their own around "
     "frozen or random transformer weights?"),
    ("moe_expert_reuse",
     "Reusing or transplanting individual mixture-of-experts experts into other models, including "
     "dense models"),
]

SYSTEM_PROMPT = ("Prefer peer-reviewed papers and arXiv preprints about transformer language models. "
                 "Give layer locations as fractions of model depth where you can, name the model each "
                 "claim was measured on, and rate how strong the evidence is.")

CLAIMS_SCHEMA = {
    "type": "object",
    "required": ["claims"],
    "properties": {
        "claims": {
            "type": "array",
            "maxItems": 5,
            "items": {
                "type": "object",
                "required": ["claim", "layer_hint", "model", "strength"],
                "properties": {
                    "claim": {"type": "string"},
                    "layer_hint": {"type": "string"},
                    "model": {"type": "string"},
                    "strength": {"type": "string", "enum": ["weak", "medium", "strong"]},
                },
            },
        },
    },
}


# Reading the catalog.

def read_catalog(path):
    records = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line == "":
                continue
            records.append(json.loads(line))
    return records


def known_donor_layers():
    """Layer counts per model from the swarm config defaults (host included), or {} if missing."""
    try:
        from wsp.swarm.config import DEFAULTS
    except ImportError:
        print("note: wsp/swarm/config.py is missing, so layer bands need a layer count in each record",
              file=sys.stderr)
        return {}
    known = dict(DEFAULTS.get("donor_layers", {}))
    if DEFAULTS.get("host") and DEFAULTS.get("host_layers"):
        known[DEFAULTS["host"]] = DEFAULTS["host_layers"]
    return known


def model_of(record):
    """The donor model of a part, or the base model of a host component (mask, LoRA)."""
    if record.get("type") in ("deletion_mask", "lora"):
        return record.get("base") or record.get("model") or ""
    return record.get("model") or record.get("base") or ""


def layer_count(record, known):
    for name in ("donor_layers", "num_layers", "n_layers"):
        value = record.get(name)
        if isinstance(value, int) and value > 0:
            return value
    return known.get(model_of(record))


def layer_band(record, known):
    """early, middle or late by relative depth; "all" with no layer; "unknown" with no layer count."""
    layer = None
    if isinstance(record.get("layer"), (int, float)):
        layer = float(record["layer"])
    elif isinstance(record.get("layers"), (list, tuple)) and len(record["layers"]) > 0:
        layer = (float(record["layers"][0]) + float(record["layers"][-1])) / 2.0
    if layer is None:
        return "all"
    count = layer_count(record, known)
    if not count:
        return "unknown"
    depth = layer / count
    if depth < 1.0 / 3.0:
        return "early"
    if depth < 2.0 / 3.0:
        return "middle"
    return "late"


def skill_of(record):
    """The skill a part is about, as a short phrase taken from its trained_on.

    Rules: drop a "STUB:" prefix and anything in parentheses, keep what
    follows the last " on ", then drop numbers and NOISE_WORDS. Examples:
    "GSM8K train activations" gives "GSM8K"; "MMLU-Pro biology SFT" gives
    "MMLU-Pro biology"; the pilot catalog's "top 256 MLP neurons by mean
    |activation| (down_proj input) on 64 GSM8K heal-split texts (...)" gives
    "GSM8K".
    """
    text = str(record.get("trained_on") or "")
    if text.upper().startswith("STUB:"):
        text = text[len("STUB:"):]
    text = re.sub(r"\([^)]*\)", " ", text)
    if " on " in text:
        text = text[text.rindex(" on ") + len(" on "):]
    text = text.replace("_", " ")
    words = []
    for word in text.split():
        if word.isdigit() or word.lower() in NOISE_WORDS:
            continue
        words.append(word)
    return " ".join(words)


def group_parts(records, known):
    """Group usable records by (model, kind, band, skill). Returns (groups, skipped ids)."""
    groups = {}
    skipped = []
    for record in records:
        part_id = record.get("id")
        kind = record.get("type")
        if not part_id or kind not in KIND_PHRASES:
            skipped.append(str(part_id))
            continue
        model = model_of(record)
        band = layer_band(record, known)
        skill = skill_of(record)
        key = (model, kind, band, skill)
        if key not in groups:
            groups[key] = {"model": model, "kind": kind, "band": band, "skill": skill, "part_ids": []}
        groups[key]["part_ids"].append(part_id)
    ordered = []
    for key in sorted(groups, key=group_sort_key):
        group = groups[key]
        group["part_ids"] = sorted(group["part_ids"])
        ordered.append(group)
    return ordered, skipped


def group_sort_key(key):
    model, kind, band, skill = key
    return (model, kind, BAND_ORDER.index(band), skill)


# Planning the calls.

def component_query(group):
    model_name = group["model"].split("/")[-1] or "an open-weight language model"
    text = KIND_PHRASES[group["kind"]]
    if group["band"] in ("early", "middle", "late"):
        text += " in the " + group["band"] + " layers"
    text += " of " + model_name
    if group["skill"]:
        text += " and their role in " + group["skill"]
    return text


def component_params(group, end_date, exclude):
    highlight_query = KIND_PHRASES[group["kind"]]
    if group["skill"]:
        highlight_query = group["skill"] + " " + highlight_query
    params = {"query": component_query(group), "type": "auto", "category": "publication",
              "num_results": COMPONENT_RESULTS, "exclude_domains": list(exclude),
              "contents": {"highlights": {"query": highlight_query, "max_characters": 1200}}}
    if end_date:
        params["end_published_date"] = end_date
    return params


def topic_params(question, end_date, exclude):
    params = {"query": question, "type": "deep-lite", "category": "publication",
              "num_results": TOPIC_RESULTS, "exclude_domains": list(exclude),
              "contents": {"highlights": True}, "system_prompt": SYSTEM_PROMPT,
              "output_schema": CLAIMS_SCHEMA}
    if end_date:
        params["end_published_date"] = end_date
    return params


def group_label(group):
    return group["model"].split("/")[-1] + "|" + group["kind"] + "|" + group["band"] + "|" + group["skill"]


def plan_calls(groups, end_date, exclude, sdk_version):
    """The ordered call plan: component searches, then topical searches."""
    plan = []
    for group in groups:
        params = component_params(group, end_date, exclude)
        plan.append({"kind": "component", "label": group_label(group), "params": params,
                     "part_ids": list(group["part_ids"]),
                     "key": lit_mod.cache_key("search", params, sdk_version),
                     "estimate": lit_mod.estimate_cost("search", params)})
    for topic_id, question in TOPICS:
        params = topic_params(question, end_date, exclude)
        plan.append({"kind": "topic", "label": topic_id, "params": params, "topic": topic_id,
                     "key": lit_mod.cache_key("search", params, sdk_version),
                     "estimate": lit_mod.estimate_cost("search", params)})
    return plan


def plan_totals(plan, literature):
    """Dollar estimates: (cold total, component part, topic part, uncached count, uncached total)."""
    cold = 0.0
    component = 0.0
    topic = 0.0
    uncached = 0
    uncached_cost = 0.0
    for call in plan:
        cold += call["estimate"]
        if call["kind"] == "component":
            component += call["estimate"]
        else:
            topic += call["estimate"]
        if not literature.is_cached(call["key"]):
            uncached += 1
            uncached_cost += call["estimate"]
    return cold, component, topic, uncached, uncached_cost


# Dry run.

def print_dry_run(plan, literature, catalog_path, records, skipped, end_date):
    components = 0
    for call in plan:
        if call["kind"] == "component":
            components += 1
    print("Exa priors pre-pass plan (dry run: no network call, no key needed)")
    print("catalog: " + catalog_path + " (" + str(len(records)) + " records, " +
          str(len(records) - len(skipped)) + " used, " + str(len(skipped)) + " skipped)")
    print("end_published_date: " + str(end_date) + "   exa-py for cache keys: " + literature.sdk_version)
    print("cache: " + literature.cache_dir)
    print("")
    number = 0
    for call in plan:
        number += 1
        params = call["params"]
        cached = "cached" if literature.is_cached(call["key"]) else "not cached"
        line = ("%3d  %-9s  %-9s  %d results  est $%.3f  %s  %s" %
                (number, call["kind"], params["type"], params["num_results"], call["estimate"], cached,
                 call["label"]))
        print(line)
        print("       query: " + params["query"])
        if call["kind"] == "component":
            print("       parts: " + str(len(call["part_ids"])))
    cold, component_cost, topic_cost, uncached, uncached_cost = plan_totals(plan, literature)
    print("")
    print("Planned calls: " + str(len(plan)) + " (" + str(components) + " component searches, " +
          str(len(plan) - components) + " topical deep-lite searches)")
    print("List-price estimate, cold cache: $%.3f (component $%.3f, topical $%.3f)" %
          (cold, component_cost, topic_cost))
    print("Not cached now: %d calls, estimate $%.3f" % (uncached, uncached_cost))
    print("Estimate rules (exa.md section 4): auto $0.007, deep-lite $0.012 per request, plus $0.001 per "
          "result per content type. UNVERIFIED: whether highlights bill on top of the search price, "
          "and any extra cost of output_schema on deep-lite (counted as $0 here).")
    if skipped:
        print("Skipped records (no id or a kind with no query): " + ", ".join(skipped))


# Building the pack.

def display_path(path):
    full = os.path.abspath(path)
    try:
        relative = os.path.relpath(full, REPO_ROOT)
    except ValueError:
        return full.replace("\\", "/")
    if relative.startswith(".."):
        return full.replace("\\", "/")
    return relative.replace("\\", "/")


def file_sha256(path):
    """sha256 of a text file with CRLF read as LF, so every checkout gives the same value."""
    with open(path, "rb") as handle:
        return lit_mod.text_sha256(handle.read())


def build_pack(plan, literature, catalog_path, end_date):
    """Run every planned call through the literature cache. Returns (pack, found count)."""
    components = {}
    topics = []
    queries = []
    misses = []
    found = 0
    for call in plan:
        data = literature.fetch("search", call["params"], role="prepass")
        row = {"kind": call["kind"], "label": call["label"], "key": call["key"],
               "query": call["params"]["query"], "type": call["params"]["type"], "found": data is not None}
        if data is not None:
            found += 1
        else:
            misses.append(call["label"])
        if call["kind"] == "component":
            row["part_ids"] = list(call["part_ids"])
            cites = []
            if data is not None:
                cites = lit_mod.cites_from_response(data, COMPONENT_RESULTS, literature.exclude_domains,
                                                    literature.quote_chars)
            for part_id in call["part_ids"]:
                components[part_id] = list(cites)
        else:
            row["topic"] = call["topic"]
            if data is not None:
                claims = lit_mod.claims_from_response(data, call["topic"], literature.exclude_domains,
                                                      literature.quote_chars)
                for claim in claims:
                    topics.append(claim)
        queries.append(row)
    pack = {"version": 1, "sdk_version": literature.sdk_version, "end_published_date": end_date,
            "catalog": display_path(catalog_path), "catalog_sha256": file_sha256(catalog_path),
            "exclude_domains": list(literature.exclude_domains), "queries": queries,
            "components": components, "topics": topics, "misses": misses}
    return pack, found


def write_pack(path, pack):
    """Write the pack as sorted, indented JSON bytes with LF endings and return their sha.

    The sha is lit_mod.text_sha256 of the bytes, the same value
    ExaLiterature.priors() reports for this file on any checkout.
    """
    data = (json.dumps(pack, sort_keys=True, indent=2, ensure_ascii=True) + "\n").encode("utf-8")
    folder = os.path.dirname(os.path.abspath(path))
    os.makedirs(folder, exist_ok=True)
    temp_path = path + "." + str(os.getpid()) + ".tmp"
    with open(temp_path, "wb") as handle:
        handle.write(data)
    os.replace(temp_path, path)
    return lit_mod.text_sha256(data)


def write_calls_log(path, rows):
    folder = os.path.dirname(os.path.abspath(path))
    os.makedirs(folder, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def summarize_calls(rows):
    counts = {}
    reported = 0.0
    unreported = 0
    for row in rows:
        status = row.get("status", "")
        counts[status] = counts.get(status, 0) + 1
        if row.get("status") == "live":
            if row.get("cost_dollars") is None:
                unreported += 1
            else:
                reported += row["cost_dollars"]
    return counts, reported, unreported


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Build the frozen Exa priors pack.")
    parser.add_argument("catalog", help="catalog JSONL, for example data/pilot_catalog.jsonl or catalog.jsonl")
    parser.add_argument("--out", default="data/priors.json", help="where to write the pack")
    parser.add_argument("--cache-dir", default="exa_cache", help="tier-one cache folder")
    parser.add_argument("--mode", default="live", choices=["live", "replay"],
                        help="live calls Exa on a cache miss; replay uses the cache only")
    parser.add_argument("--dry-run", action="store_true", help="print the plan and cost estimate only")
    parser.add_argument("--end-date", default=lit_mod.DEFAULT_END_DATE,
                        help="latest published date; keep it fixed so cache keys stay stable")
    parser.add_argument("--max-cost-dollars", type=float, default=3.0,
                        help="refuse a live build whose estimate for uncached calls is above this")
    parser.add_argument("--modal-cache", action="store_true", help="also use the modal.Dict exa-cache tier")
    parser.add_argument("--calls-log", default=None,
                        help="where to write the call rows (default outputs/priors/calls-<time>.jsonl)")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    catalog_path = args.catalog
    if not os.path.isabs(catalog_path) and not os.path.exists(catalog_path):
        catalog_path = os.path.join(REPO_ROOT, catalog_path)
    if not os.path.exists(catalog_path):
        print("catalog not found: " + args.catalog, file=sys.stderr)
        return 1
    records = read_catalog(catalog_path)
    groups, skipped = group_parts(records, known_donor_layers())

    out_path = lit_mod.resolve_path(args.out)
    cache_dir = lit_mod.resolve_path(args.cache_dir)
    mode = args.mode
    use_modal = args.modal_cache
    if args.dry_run:
        mode = "replay"
        use_modal = False
    config = {"exa_cache_dir": cache_dir, "priors_path": out_path,
              "exa_modal_cache": use_modal, "exa_end_published_date": args.end_date,
              "exa_max_live_calls": len(groups) + len(TOPICS)}
    literature = lit_mod.ExaLiterature(config, mode)
    plan = plan_calls(groups, args.end_date, literature.exclude_domains, literature.sdk_version)

    if args.dry_run:
        print_dry_run(plan, literature, display_path(catalog_path), records, skipped, args.end_date)
        return 0

    if args.mode == "live" and literature.mode != "live":
        print("live mode is not available (" + str(literature.degraded_reason) + "); building from the "
              "cache only")
    if literature.mode == "live":
        cold, component_cost, topic_cost, uncached, uncached_cost = plan_totals(plan, literature)
        if uncached_cost > args.max_cost_dollars:
            print("refusing to start: %d uncached calls are estimated at $%.3f, above --max-cost-dollars "
                  "%.2f" % (uncached, uncached_cost, args.max_cost_dollars), file=sys.stderr)
            return 1
        print("live build: %d of %d calls not cached, estimate $%.3f" % (uncached, len(plan), uncached_cost))

    pack, found = build_pack(plan, literature, catalog_path, args.end_date)
    rows = literature.drain_calls()
    calls_log = args.calls_log
    if calls_log is None:
        stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        calls_log = os.path.join(REPO_ROOT, "outputs", "priors", "calls-" + stamp + ".jsonl")
    write_calls_log(calls_log, rows)
    counts, reported, unreported = summarize_calls(rows)
    print("calls by status: " + json.dumps(counts, sort_keys=True))
    print("Exa-reported cost of live calls: $%.4f (%d live calls reported no cost)" % (reported, unreported))
    print("call rows: " + calls_log)
    if found == 0:
        print("no planned call has a cached or live result, so no pack was written", file=sys.stderr)
        return 1
    sha = write_pack(out_path, pack)
    with_cites = 0
    for part_id in pack["components"]:
        if len(pack["components"][part_id]) > 0:
            with_cites += 1
    print("wrote " + display_path(out_path) + ": " + str(with_cites) + " of " + str(len(pack["components"])) +
          " parts have cites, " + str(len(pack["topics"])) + " topical claims, " +
          str(len(pack["misses"])) + " planned calls missed")
    print("priors_sha " + sha)
    return 0


if __name__ == "__main__":
    sys.exit(main())
