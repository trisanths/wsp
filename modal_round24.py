"""Round 24: skill tops. A shared base trunk, and a stitch plus the donor's own upper layers, norm, and head per skill.

Question. Runs 14, 14b, 20, and 22 sent a donor block's output back into the
base's upper layers, and it gave nothing: the base's upper layers cannot read
donor features. The stitch study showed the other direction works. Base
bottom, a trained stitch, and the donor's own remaining layers and head give
a donor LM loss within 0.03 to 0.08 nats of the donor. So keep the donor's
readout. The base keeps layers 0..cut-1 as a shared frozen trunk, each skill
gets a top (stitch, donor layers from the cut, donor final norm, donor head),
and a router picks one top, or the base's own top, per sequence. One network,
one path per sequence. Code: wsp/tops.py, mechanism and limits: docs/tops.md.

No benchmark data trains anything (owner rule, 2026-11-14). Stitch fits,
router fits, and the router margin all read only wsp.healdata pools under the
default reported set, and healdata.check_sources runs on every pool. Three
disjoint ranges of every pool (RANGES): stitch fit, router fit, probe. The
probe range gives the stitch penalty and the router margin. Scoring items come
from wsp.skills_eval and wsp.practice, which are exempt. Nothing here trains
on them.

Pairs (every donor shares the Qwen2.5 token ids; wsp.tops.check_vocab compares
the token to id maps and prints what it masks or drops):
  T1  base Qwen/Qwen2.5-1.5B-Instruct, math top from Qwen/Qwen2.5-Math-1.5B-Instruct,
      code top from Qwen/Qwen2.5-Coder-1.5B-Instruct. Math has rope_theta 10000
      and its own chat template. The top keeps the donor's rotary base. Prompts
      stay in the base's template. The "template" cell scores the forced math
      top with the donor's own template, to price the template.
  T2  base Qwen/Qwen2.5-1.5B-Instruct, code top from Qwen/Qwen2.5-Coder-7B-Instruct
      (another width: the stitch maps 1536 to 3584).
  T3  base Qwen/Qwen2.5-7B-Instruct, code top from Qwen/Qwen2.5-Coder-7B-Instruct
      and math top from Qwen/Qwen2.5-Math-7B-Instruct.
  T4  EXPLORATORY, optional, not part of the go rule and not in cells(). Cross-tokenizer: base
      unsloth/Llama-3.2-3B-Instruct (Llama 3 tokenizer, 128256 ids) with a math top from
      Qwen/Qwen2.5-Math-1.5B-Instruct (Qwen tokenizer, 151665 ids). The donor head scores the
      donor's vocabulary and wsp/tops_vocab.py bridges it to the base's ids (exact byte match,
      first-donor-token fallback with a learned bias). Both models have 28 layers. Cuts 1/2 and
      3/4 (14 and 21). The fit trains on the donor's own greedy answers to pool prompts plus a
      KL on the shared tokens (docs/tops.md, "Cross-tokenizer tops"). Cells: base, donor,
      forced math (GSM8K), scrambled control. Run it only with --pairs T4. One forced seed at
      cut 21. Nothing in T4 enters summarize()'s go rule; summarize_t4 prints its own lines.
Cuts, as base layers kept in the trunk (28 layers): 1/7, 2/7, 1/2, 5/7, 6/7 =
4, 8, 14, 20, 24. The donor cut is the same relative depth (28 layers in every
donor, so the same number).

Cells (cell_name(cell) is the first token of every score line):
  base          the base alone. Greedy decoding, no training, one run.
  donor         each donor alone, with its own chat template, same items.
  kd_lora       reference: the donor distilled into a LoRA on the base (KL on the
                same texts and steps as the stitch fit). Rank is the largest whose
                LoRA params fit in the stitch's params. Seeds 0 and 1.
  forced        the top forced for every sequence: stitch penalty, the skill's
                benchmarks, MMLU. Every cut. Seeds 0 and 1 at the default cut
                (1/2). ONE SEED (seed 0) at the other four cuts.
  routed        default cut, all tops and the router over every benchmark, router
                choice shares per benchmark. Seeds 0 and 1.
  scrambled     default cut, the top's donor layers scrambled (wsp.scramble, spectrum
                mode, control_seed = seed) before the stitch is trained. Seeds 0 and 1.
  lora          default cut, forced top with a rank 16 LoRA on its donor layers,
                trained 800 steps together with the stitch. Seeds 0 and 1.
  template      T1 only: the forced math top at the default cut, scored with the
                donor's own template. Seeds 0 and 1. Uses the forced cell's stitch.
A seed changes the batch order, the LoRA init, and the scramble. The fit texts and
the ridge init are the same for every seed, and decoding is greedy, so a seed spread
is the spread of training order only. The reference cells base and donor have no
training, so one run each.

Benchmarks (wsp.skills_eval prompts and graders): GSM8K 300 and MATH-500 for math,
HumanEval and MBPP for code (200 of the sanitized test items, as in round 20),
MMLU 1000 as the general guard. Routed and base cells score all five.

Rows. Per cell: scores, n, stitch penalty (before and after, per text set), params
stored and run per token (per path for the routed network), added params as a multiple
of the base, stitch fit seconds, decode tokens per second at batch 1, router shares,
the vocabulary check report. Results are written to /vol/round24/results/<cell>.json
after every cell and to results/round24_shard<i>.json as each cell returns. A stitch
penalty above ABORT_PENALTY stops the cell before scoring (stage 1 must work before
stage 2 is paid for). A penalty above MAX_PENALTY flags the top. The script sets
allow_flagged, so a flagged top is still scored and the row says "flagged".

Go rule. At some cut, the routed network scores within 3 points of each donor on its
own skill and within 1 point of the base on MMLU, with stored params at most 1.6 times
the base for T1 and params run per token at most 1.05 times the larger of the two models'
per-path cost (the base's and the donor's, per path), while the scrambled control falls
to the base or below. The stored limit is 1.6 for T3 as well (the same ratio on the 7B
base) and 2.0 for T2: one 7B head holds 0.55B params and four 7B layers 0.93B, so a 1.5B
base with a 7B top cannot get under 1.98 times at any cut (see Sizes). STORED_LIMIT holds
the three numbers. The routed cell at the default cut 1/2 cannot meet the stored limit
(T1 2.17 times, T3 2.02 times, T2 3.49 times), so the round also runs one routed cell at
the deepest cut (6/7, seed 0, "extra") where the limit can be met for T1 (1.56 times) and
T3 (1.41 times). summarize() tests the rule at every cut that has a routed cell, using
the scrambled control from the default cut. The round also reports the deepest cut
(smallest top) at which the forced top stays within 3 points of the donor on its own
skill: the size of the component a skill needs.

Sizes (stored params as a multiple of the base, from the real configs by
wsp.tops.sizes_from_configs, MLP stitch, sinks and router included; run
`python modal_round24.py sizes`). Cuts 4, 8, 14, 20, 24 of 28 layers:
  T1  2.773x 2.530x 2.166x 1.803x 1.560x   (base 1.544B, per-path run 1.556B vs 1.544B)
  T2  5.003x 4.400x 3.494x 2.588x 1.984x   (code path 6.37B to 2.64B vs a 7.07B donor)
  T3  2.629x 2.384x 2.017x 1.650x 1.405x   (base 7.616B, per-path run 7.135B vs 7.071B)
The tied 1.5B head alone is 233M params per top, and every top stores its own copy.

Usage.
    python scripts/modal_fanout.py modal_round24.py round24 --profiles a,b,c
    modal run modal_round24.py --smoke
    modal run modal_round24.py --shard 0 --nshards 3 --only forced,scrambled,lora,kd_lora,base,donor
    modal run modal_round24.py --only routed,template --pairs T1
    modal run modal_round24.py --plan-only
    modal run --detach modal_round24.py --diag             smoke cell's fitted top against its donor on 32 GSM8K prompts
    modal run modal_round24.py --smoke --pairs T4          one cross-tokenizer cell, 200 steps, GSM8K only
    modal run modal_round24.py --pairs T4 --only forced --cuts 1/2 --seeds 0
Run the smoke first and wait for its first score line. A fan-out comes only after it.
"""

import json
import os
import random
import time

import modal

app = modal.App("wsp-round24")
volume = modal.Volume.from_name("wsp-models", create_if_missing=True)


def safe_reload():
    """Reload the volume to see other shards' files. Open files (HF logs, caches) block a reload; then skip it."""
    try:
        volume.reload()
    except Exception as error:
        print("volume reload skipped:", repr(error)[:200])


image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch==2.7.1", "transformers==4.57.1", "accelerate", "huggingface_hub", "requests",
        "safetensors", "zstandard", "numpy", "datasets", "sentencepiece", "protobuf",
    )
    .env({"HF_HOME": "/vol/hf", "HF_HUB_DISABLE_XET": "1"})
    .add_local_python_source("wsp")
)

QWEN_SMALL = "Qwen/Qwen2.5-1.5B-Instruct"
QWEN_BIG = "Qwen/Qwen2.5-7B-Instruct"
MATH_15 = "Qwen/Qwen2.5-Math-1.5B-Instruct"
CODER_15 = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
CODER_7 = "Qwen/Qwen2.5-Coder-7B-Instruct"
MATH_7 = "Qwen/Qwen2.5-Math-7B-Instruct"

PAIRS = {
    "T1": {"base": QWEN_SMALL, "tops": {"math": MATH_15, "code": CODER_15}, "n_layers": 28},
    "T2": {"base": QWEN_SMALL, "tops": {"code": CODER_7}, "n_layers": 28},
    "T3": {"base": QWEN_BIG, "tops": {"code": CODER_7, "math": MATH_7}, "n_layers": 28},
}
# Optional pair, not in cells(): a Llama-family base with a Qwen-family donor (cross tokenizer).
LLAMA_3B = "unsloth/Llama-3.2-3B-Instruct"
OPTIONAL_PAIRS = {
    "T4": {"base": LLAMA_3B, "tops": {"math": MATH_15}, "n_layers": 28, "cross_tokenizer": True,
           "cut_fractions": [(1, 2), (3, 4)]},
}
ALL_PAIRS = dict(PAIRS)
ALL_PAIRS.update(OPTIONAL_PAIRS)
ANSWER_TOKENS = 128
CROSS_FIT = {"math": [("math", 600), ("general", 100), ("reasoning", 100)]}
CROSS_PROBE = {"math": [("math", 50), ("general", 25), ("reasoning", 25)]}
OWN_SKILLS = {"math": ["gsm8k", "math500"], "code": ["humaneval", "mbpp"]}
GUARD = ["mmlu"]
STORED_LIMIT = {"T1": 1.6, "T2": 2.0, "T3": 1.6}
ALL_SKILLS = ["gsm8k", "math500", "humaneval", "mbpp", "mmlu"]
CUT_FRACTIONS = [(1, 7), (2, 7), (1, 2), (5, 7), (6, 7)]
DEFAULT_FRACTION = (1, 2)
SEEDS = [0, 1]
STEPS = 800
LORA_RANK = 16
SEQ_LEN = 256
BATCH = 8
MAX_PENALTY = 0.35
ABORT_PENALTY = 1.5
SCORE_COUNTS = {"gsm8k": 300, "mbpp": 200, "mmlu": 1000}
EVAL_BATCH = {"gsm8k": 32, "math500": 32, "humaneval": 32, "mbpp": 32, "mmlu": 32}

# Disjoint slices of every healdata pool, by role: (skip, most texts taken). Pools read round robin
# in a fixed order, so these slices never overlap.
RANGES = {"stitch": (0, 1200), "router": (1200, 400), "probe": (1600, 150)}
FIT = {"math": [("math", 1000), ("general", 150), ("prose", 150)],
       "code": [("code", 1000), ("general", 150), ("prose", 150)]}
PROBE = {"math": [("math", 100), ("general", 25), ("prose", 25)],
         "code": [("code", 100), ("general", 25), ("prose", 25)]}
ROUTER = {"math": [("math", 300)], "code": [("code", 300)],
          "base": [("general", 150), ("knowledge", 100), ("reasoning", 100), ("medicine", 100)]}
ROUTER_PROBE = {"math": [("math", 100)], "code": [("code", 100)],
                "base": [("general", 40), ("knowledge", 30), ("reasoning", 30), ("medicine", 30)]}

DATA_DIR = "/vol/round24/data"
TOPS_DIR = "/vol/round24/tops"
RESULTS_DIR = "/vol/round24/results"
FETCH_DIR = "/vol/round24/fetch"

GPU_SPEC = {"gpu": "H200", "memory": 131072, "timeout": 6 * 3600}


# ------------------------------------------------------------------ cells


def cuts_for(n_layers):
    """The five trunk sizes as base layer counts."""
    out = []
    for numerator, denominator in CUT_FRACTIONS:
        out.append(int(round(n_layers * numerator / denominator)))
    return out


def default_cut(n_layers):
    return int(round(n_layers * DEFAULT_FRACTION[0] / DEFAULT_FRACTION[1]))


def skills_of(top):
    return OWN_SKILLS[top] + GUARD


def new_cell(pair, kind, top=None, cut=None, seed=0, skills=None, **extra):
    cell = {"pair": pair, "kind": kind, "top": top, "cut": cut, "seed": seed, "skills": skills, "steps": STEPS}
    cell.update(extra)
    return cell


def cells():
    """Every cell of the round, in an order that puts the references and forced cells first."""
    out = []
    for pair, spec in PAIRS.items():
        n = spec["n_layers"]
        default = default_cut(n)
        tops = list(spec["tops"].keys())
        pair_skills = list(ALL_SKILLS)
        out.append(new_cell(pair, "base", skills=pair_skills))
        for top in tops:
            out.append(new_cell(pair, "donor", top=top, skills=skills_of(top)))
        for top in tops:
            for seed in SEEDS:
                out.append(new_cell(pair, "kd_lora", top=top, seed=seed, skills=skills_of(top)))
        for cut in cuts_for(n):
            seeds = SEEDS if cut == default else [0]
            for top in tops:
                for seed in seeds:
                    out.append(new_cell(pair, "forced", top=top, cut=cut, seed=seed, skills=skills_of(top)))
        for top in tops:
            for seed in SEEDS:
                out.append(new_cell(pair, "scrambled", top=top, cut=default, seed=seed, skills=skills_of(top)))
                out.append(new_cell(pair, "lora", top=top, cut=default, seed=seed, skills=skills_of(top),
                                    lora_rank=LORA_RANK))
        for seed in SEEDS:
            out.append(new_cell(pair, "routed", top="all", cut=default, seed=seed, skills=pair_skills))
        deepest = cuts_for(n)[-1]
        if deepest != default:
            # Extra, not in the brief: the default cut cannot meet the stored limit, so the go rule needs one
            # routed network at a cut that can. One seed. It reuses the forced cells' stitches.
            out.append(new_cell(pair, "routed", top="all", cut=deepest, seed=0, skills=pair_skills, extra=True))
        if "math" in tops and pair == "T1":
            for seed in SEEDS:
                out.append(new_cell(pair, "template", top="math", cut=default, seed=seed,
                                    skills=list(OWN_SKILLS["math"])))
    return out


def optional_cells():
    """The T4 cells (exploratory): references, forced math at 1/2 (two seeds) and 3/4 (one seed), scrambled control."""
    out = []
    pair = "T4"
    spec = OPTIONAL_PAIRS[pair]
    n = spec["n_layers"]
    cuts = [int(round(n * a / b)) for a, b in spec["cut_fractions"]]
    default = cuts[0]
    skills = ["gsm8k"]
    out.append(new_cell(pair, "base", skills=list(skills)))
    out.append(new_cell(pair, "donor", top="math", skills=list(skills)))
    for cut in cuts:
        for seed in (SEEDS if cut == default else [0]):
            out.append(new_cell(pair, "forced", top="math", cut=cut, seed=seed, skills=list(skills)))
    for seed in SEEDS:
        out.append(new_cell(pair, "scrambled", top="math", cut=default, seed=seed, skills=list(skills)))
    return out


def smoke_cell(pair="T1"):
    """The one smoke cell: T1, math top forced, cut 1/2, 200 stitch steps, GSM8K only. pair T4 gives the cross cell."""
    cell = new_cell(pair, "forced", top="math", cut=default_cut(28), seed=0, skills=["gsm8k"])
    cell["steps"] = 200
    cell["smoke"] = True
    return cell


def cell_name(cell):
    parts = [cell["pair"], cell["kind"], cell["top"] or "none"]
    if cell["cut"] is not None:
        parts.append("c" + str(cell["cut"]))
    parts.append("s" + str(cell["seed"]))
    if cell.get("smoke"):
        parts.append("smoke")
    return "_".join(parts)


VARIANT_OF = {"forced": "plain", "routed": "plain", "template": "plain", "scrambled": "scrambled", "lora": "lora"}


FIT_VERSION = "kd1"  # wsp.tops fit_stitch objective "kd" (KL to the donor); bump when the fit changes


def artifact_dir(cell, top):
    """Where the fitted top of a cell lives. Forced, routed, and template cells share the plain stitch."""
    variant = VARIANT_OF[cell["kind"]]
    # FIT_VERSION names the stitch objective, so a change of objective never reuses an old fit.
    tag = f"{cell['pair']}_{top}_c{cell['cut']}_{variant}_s{cell['seed']}_{FIT_VERSION}"
    if cell["steps"] != STEPS:
        tag += f"_n{cell['steps']}"
    return os.path.join(TOPS_DIR, tag)


def select(todo, pairs="", cuts="", only="", seeds=""):
    """Filter cells. Reference cells (no cut) pass a cuts filter. cuts: fractions like 1/2 or layer counts."""
    out = []
    wanted_cuts = []
    if cuts:
        for item in cuts.split(","):
            item = item.strip()
            if "/" in item:
                numerator, denominator = item.split("/")
                wanted_cuts.append((int(numerator), int(denominator)))
            else:
                wanted_cuts.append(int(item))
    for cell in todo:
        if pairs and cell["pair"] not in pairs.split(","):
            continue
        if only and cell["kind"] not in only.split(","):
            continue
        if seeds and str(cell["seed"]) not in seeds.split(","):
            continue
        if wanted_cuts and cell["cut"] is not None:
            n = ALL_PAIRS[cell["pair"]]["n_layers"]
            ok = False
            for item in wanted_cuts:
                if isinstance(item, tuple):
                    if int(round(n * item[0] / item[1])) == cell["cut"]:
                        ok = True
                elif item == cell["cut"]:
                    ok = True
            if not ok:
                continue
        out.append(cell)
    return out


# ------------------------------------------------------------------ cost plan (estimates, not measurements)

# Minutes on one H200. Guesses from round 20 timings and the stitch study; the smoke cell replaces them.
EVAL_MIN = {"small": {"gsm8k": 4, "math500": 12, "humaneval": 4, "mbpp": 4, "mmlu": 2},
            "big": {"gsm8k": 8, "math500": 25, "humaneval": 8, "mbpp": 8, "mmlu": 4}}
FIT_MIN = {"small": 10, "big": 30}
LOAD_MIN = {"small": 3, "big": 8}
SCRAMBLE_MIN = {"small": 1, "big": 5}
PEAK_GB = {"small": 22, "big": 70}


def size_class(cell):
    """big when the base or a donor the cell loads has 7B params (it sets fit time, eval time, and memory)."""
    spec = ALL_PAIRS[cell["pair"]]
    if spec["base"] == QWEN_BIG:
        return "big"
    if cell["kind"] == "base":
        return "small"
    if cell["top"] in (None, "all"):
        donors = list(spec["tops"].values())
    else:
        donors = [spec["tops"][cell["top"]]]
    for donor in donors:
        if "7B" in donor:
            return "big"
    return "small"


def cell_minutes(cell):
    """Estimated H200 minutes for a cell, and its peak memory in GB."""
    size = size_class(cell)
    minutes = LOAD_MIN[size]
    memory = PEAK_GB[size]
    if size == "big" and ALL_PAIRS[cell["pair"]]["base"] != QWEN_BIG:
        memory = 50
    for skill in cell["skills"]:
        minutes += EVAL_MIN[size][skill]
    kind = cell["kind"]
    if kind in ("forced", "scrambled", "lora", "kd_lora"):
        minutes += FIT_MIN[size] * cell["steps"] / STEPS
    if kind == "lora":
        minutes += 0.2 * FIT_MIN[size]
    if kind == "scrambled":
        minutes += SCRAMBLE_MIN[size]
    if kind == "routed":
        # Loads or fits each top (fits are skipped when the forced cells ran first), then the router.
        minutes += 2 + 1.5 * len(ALL_PAIRS[cell["pair"]]["tops"])
    if kind == "donor":
        memory = PEAK_GB[size] * 0.6
    return minutes, memory


def plan(todo=None):
    """Per pair: cell counts by kind, estimated H200 hours and the peak memory. Printed and returned."""
    todo = cells() if todo is None else todo
    out = {}
    for cell in todo:
        entry = out.setdefault(cell["pair"], {"cells": 0, "minutes": 0.0, "peak_gb": 0.0, "by_kind": {}})
        minutes, memory = cell_minutes(cell)
        entry["cells"] += 1
        entry["minutes"] += minutes
        entry["peak_gb"] = max(entry["peak_gb"], memory)
        kind = entry["by_kind"].setdefault(cell["kind"], {"cells": 0, "minutes": 0.0})
        kind["cells"] += 1
        kind["minutes"] += minutes
    for pair in out:
        out[pair]["gpu_hours"] = round(out[pair]["minutes"] / 60.0, 1)
        print(f"plan {pair}: {out[pair]['cells']} cells, about {out[pair]['gpu_hours']} H200 hours, "
              f"peak about {out[pair]['peak_gb']:.0f} GB")
        for kind, info in out[pair]["by_kind"].items():
            print(f"plan   {kind}: {info['cells']} cells, {info['minutes'] / 60.0:.1f} h")
    total = sum(out[pair]["minutes"] for pair in out) / 60.0
    print(f"plan total: {len(todo)} cells, about {total:.0f} H200 hours (estimates, not measurements)")
    return out


# ------------------------------------------------------------------ data


def safe_name(text):
    return text.replace("/", "__")


def pool_texts(base_tokenizer, base_id, pool, role, count):
    """Chat-formatted texts of one healdata pool in one range. Never benchmark data. Cached on the volume."""
    from wsp import healdata

    skip, size = RANGES[role]
    if count > size:
        raise ValueError(f"role {role} holds at most {size} texts per pool, asked for {count}")
    path = os.path.join(DATA_DIR, f"{role}_{pool}_{count}_{safe_name(base_id)}.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    healdata.check_sources([source[0] for source in healdata.SOURCES[pool]])
    texts = healdata.pool(pool, count, tokenizer=base_tokenizer, seed=0, skip=skip)
    if len(texts) < count:
        raise ValueError(f"pool {pool} gave {len(texts)} texts for {role}, wanted {count}")
    os.makedirs(DATA_DIR, exist_ok=True)
    tmp = f"{path}.{os.getpid()}.tmp"
    with open(tmp, "w") as f:
        json.dump(texts, f)
    # No hard links on a Modal volume: a plain rename. A file that is already there is kept.
    if os.path.exists(path):
        os.remove(tmp)
    else:
        os.replace(tmp, path)
    return texts


def pool_prompts(base_tokenizer, base_id, pool, role, count, suffix):
    """User prompts of a pool in one range, in the chat format and the raw format the MMLU items use."""
    from wsp import healdata
    from wsp import skills_eval as ev

    skip, size = RANGES[role]
    if count > size:
        raise ValueError(f"role {role} holds at most {size} texts per pool, asked for {count}")
    path = os.path.join(DATA_DIR, f"prompts_{role}_{pool}_{count}_{suffix}_{safe_name(base_id)}.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    healdata.check_sources([source[0] for source in healdata.SOURCES[pool]])
    pairs = healdata.pairs(pool, count, seed=0, skip=skip)
    if len(pairs) < count:
        raise ValueError(f"pool {pool} gave {len(pairs)} prompts for {role}, wanted {count}")
    prompts = []
    for index, (user, _) in enumerate(pairs):
        text = user
        if suffix == "math":
            text = user + "\n" + (ev.GSM8K_INSTRUCTION if index % 2 == 0 else ev.MATH_INSTRUCTION)
        elif suffix == "code":
            text = user + "\n\n" + ev.CODE_INSTRUCTION
        prompts.append(ev.render_prompt(base_tokenizer, text))
        prompts.append("Question: " + user.strip() + "\nAnswer:")
    os.makedirs(DATA_DIR, exist_ok=True)
    tmp = f"{path}.{os.getpid()}.tmp"
    with open(tmp, "w") as f:
        json.dump(prompts, f)
    if os.path.exists(path):
        os.remove(tmp)
    else:
        os.replace(tmp, path)
    return prompts


def interleave(domain, others):
    """Domain texts with the other pools spread through them, so the ridge subset sees every kind."""
    out = []
    taken = [0] * len(others)
    for index, text in enumerate(domain):
        out.append(text)
        for slot in range(len(others)):
            if index % 7 == 3 * slot and taken[slot] < len(others[slot]):
                out.append(others[slot][taken[slot]])
                taken[slot] += 1
    return out


def texts_for(base_tokenizer, base_id, top, table, role):
    """The mixed text list of one top for one role. Returns (mixed, domain_only, others)."""
    parts = []
    for pool, count in table[top]:
        parts.append(pool_texts(base_tokenizer, base_id, pool, role, count))
    return interleave(parts[0], parts[1:]), parts[0], parts[1:]


def router_classes_texts(base_tokenizer, base_id, tops, table, role):
    out = {}
    for name in ["base"] + list(tops):
        texts = []
        for pool, count in table[name]:
            suffix = name if name in ("math", "code") else "general"
            texts += pool_prompts(base_tokenizer, base_id, pool, role, count, suffix)
        out[name] = texts
    return out


# ------------------------------------------------------------------ scoring helpers


def load_items(skills):
    from wsp import skills_eval as ev
    from wsp.practice import scoring_items

    items = {}
    need = [s for s in skills if s in ("gsm8k", "mbpp", "mmlu")]
    if need:
        items.update(scoring_items(ev, need))
    if "math500" in skills:
        items["math500"] = ev.math500_items()
    if "humaneval" in skills:
        items["humaneval"] = ev.humaneval_items()
    return items


def decode_speed(model, tokenizer, new_tokens=128, repeats=2):
    """Greedy decode tokens per second at batch 1. Never stops on EOS, so every run makes new_tokens."""
    import torch

    from wsp import skills_eval as ev

    ev.prepare_tokenizer(tokenizer)
    text = ev.render_prompt(tokenizer, "Explain how a binary search works, step by step.")
    encoded = tokenizer([text], return_tensors="pt", add_special_tokens=False)
    device = next(model.parameters()).device
    ids = encoded["input_ids"].to(device)
    mask = encoded["attention_mask"].to(device)
    never = [len(tokenizer) + 5000]
    best = 0.0
    for run in range(repeats + 1):
        count = 8 if run == 0 else new_tokens
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.time()
        with torch.no_grad():
            out = model.generate(input_ids=ids, attention_mask=mask, max_new_tokens=count, do_sample=False,
                                 temperature=None, top_p=None, top_k=None, repetition_penalty=1.0,
                                 pad_token_id=tokenizer.pad_token_id, eos_token_id=never)
        if device.type == "cuda":
            torch.cuda.synchronize()
        seconds = time.time() - start
        made = out.shape[1] - ids.shape[1]
        if run > 0 and seconds > 0 and made / seconds > best:
            best = made / seconds
    return best


def score_model(name, model, tokenizer, skills, items, tally_owner=None):
    """Score every skill. Print one line per skill that starts with the cell name. Return (scores, counts, shares)."""
    from wsp import skills_eval as ev

    scores = {}
    counts = {}
    shares = {}
    for skill in skills:
        if tally_owner is not None:
            tally_owner.tally = {}
        result = ev.evaluate_skill(model, tokenizer, skill, items=items[skill], batch_size=EVAL_BATCH[skill])
        scores[skill] = result["score"]
        counts[skill] = {"correct": result["correct"], "n": result["n"]}
        line = f"{name} score {skill} {result['score']:.4f} ({result['correct']}/{result['n']}) in {result['seconds']} s"
        if tally_owner is not None:
            total = sum(tally_owner.tally.values())
            shares[skill] = {k: v / max(1, total) for k, v in tally_owner.tally.items()}
            line += " route " + json.dumps({k: round(v, 3) for k, v in shares[skill].items()})
            tally_owner.tally = None
        print(line, flush=True)
    return scores, counts, shares


# ------------------------------------------------------------------ building networks


def load_base(base_id):
    import torch
    import transformers

    tokenizer = transformers.AutoTokenizer.from_pretrained(base_id)
    model = transformers.AutoModelForCausalLM.from_pretrained(base_id, dtype=torch.bfloat16).to("cuda").eval()
    return model, tokenizer


def load_donor(donor_id):
    import torch
    import transformers

    tokenizer = transformers.AutoTokenizer.from_pretrained(donor_id)
    model = transformers.AutoModelForCausalLM.from_pretrained(donor_id, dtype=torch.bfloat16).to("cuda").eval()
    return model, tokenizer


def cross_prompt_pairs(base_tokenizer, donor_tokenizer, base_id, pool, role, count, suffix):
    """Prompts of a pool in the base's chat format and in the donor's chat format. Never benchmark data.

    Each pair gives two prompts: a chat prompt (rendered by each tokenizer's
    template) and the raw "Question: ... Answer:" prompt, the same string on
    both sides. Cached on the volume.
    """
    from wsp import healdata
    from wsp import skills_eval as ev

    skip, size = RANGES[role]
    if count > size:
        raise ValueError(f"role {role} holds at most {size} texts per pool, asked for {count}")
    path = os.path.join(DATA_DIR, f"cross_{role}_{pool}_{count}_{suffix}_{safe_name(base_id)}.json")
    if os.path.exists(path):
        with open(path) as f:
            found = json.load(f)
        return found["base"], found["donor"]
    healdata.check_sources([source[0] for source in healdata.SOURCES[pool]])
    pairs = healdata.pairs(pool, count, seed=0, skip=skip)
    if len(pairs) < count:
        raise ValueError(f"pool {pool} gave {len(pairs)} prompts for {role}, wanted {count}")
    base_prompts = []
    donor_prompts = []
    for index, (user, _) in enumerate(pairs):
        text = user
        if suffix == "math":
            text = user + "\n" + (ev.GSM8K_INSTRUCTION if index % 2 == 0 else ev.MATH_INSTRUCTION)
        base_prompts.append(ev.render_prompt(base_tokenizer, text))
        donor_prompts.append(ev.render_prompt(donor_tokenizer, text))
        raw = "Question: " + user.strip() + "\nAnswer:"
        base_prompts.append(raw)
        donor_prompts.append(raw)
    os.makedirs(DATA_DIR, exist_ok=True)
    tmp = f"{path}.{os.getpid()}.tmp"
    with open(tmp, "w") as f:
        json.dump({"base": base_prompts, "donor": donor_prompts}, f)
    if os.path.exists(path):
        os.remove(tmp)
    else:
        os.replace(tmp, path)
    return base_prompts, donor_prompts


def cross_fit_inputs(base_tokenizer, donor_tokenizer, base_id, top, table, role):
    base_prompts = []
    donor_prompts = []
    for pool, count in table[top]:
        suffix = top if pool == "math" else "general"
        b, d = cross_prompt_pairs(base_tokenizer, donor_tokenizer, base_id, pool, role, count, suffix)
        base_prompts += b
        donor_prompts += d
    return base_prompts, donor_prompts


def fit_one_top(cell, net, top, base_tokenizer, base_id):
    """Add and fit one top, or restore it from its artifact. Return its fit report."""
    import torch

    name = cell_name(cell)
    donor_id = ALL_PAIRS[cell["pair"]]["tops"][top]
    art = artifact_dir(cell, top)
    if os.path.exists(os.path.join(art, "tops.json")):
        net.add_saved_tops(art, names=[top], device="cuda", cache_dir=FETCH_DIR)
        print(f"{name} restored top {top} from {art}", flush=True)
        return net.tops[top].fit_report
    control = "scrambled" if cell["kind"] == "scrambled" else None
    start = time.time()
    cross = bool(ALL_PAIRS[cell["pair"]].get("cross_tokenizer"))
    net.add_top(top, donor_id, cell["cut"], control=control, control_seed=cell["seed"], device="cuda",
                cache_dir=FETCH_DIR, cross_tokenizer=cross)
    fit_texts, _, _ = texts_for(base_tokenizer, base_id, top, FIT, "stitch")
    probe_mixed, probe_domain, probe_others = texts_for(base_tokenizer, base_id, top, PROBE, "probe")
    if cross:
        from wsp import tops_vocab

        cover = tops_vocab.coverage_report(net.tops[top].bridge, base_tokenizer, net.tops[top].donor_tokenizer,
                                           probe_domain[:40] + probe_others[0][:20] + probe_others[1][:20])
        net.tops[top].spec["vocab"] = cover
        print(f"{name} vocab bridge {top}: exact {cover['exact_share']:.4f} fallback {cover['fallback_share']:.4f} "
              f"masked {cover['masked']} text base_exact {cover['text']['base_exact']:.4f} "
              f"donor_shared {cover['text']['donor_shared']:.4f}", flush=True)
    print(f"{name} built top {top} in {time.time() - start:.1f} s: {json.dumps(net.tops[top].spec['vocab'])[:400]}",
          flush=True)
    donor_model, donor_tokenizer = load_donor(donor_id)
    lora_rank = cell.get("lora_rank", 0)
    if cross:
        fit_prompts, fit_donor_prompts = cross_fit_inputs(base_tokenizer, donor_tokenizer, base_id, top, CROSS_FIT,
                                                          "stitch")
        eval_prompts, eval_donor_prompts = cross_fit_inputs(base_tokenizer, donor_tokenizer, base_id, top,
                                                            CROSS_PROBE, "probe")
        report = net.fit_stitch(top, fit_prompts, steps=cell["steps"], batch_size=BATCH, seq_len=SEQ_LEN,
                                seed=cell["seed"], eval_texts=eval_prompts, donor_model=donor_model,
                                max_penalty=MAX_PENALTY, lora_rank=lora_rank, log_every=100, log_name=name,
                                donor_prompts=fit_donor_prompts, eval_donor_prompts=eval_donor_prompts,
                                answer_tokens=ANSWER_TOKENS)
    else:
        report = net.fit_stitch(top, fit_texts, steps=cell["steps"], batch_size=BATCH, seq_len=SEQ_LEN,
                                seed=cell["seed"], eval_texts=probe_mixed, donor_model=donor_model,
                                max_penalty=MAX_PENALTY, lora_rank=lora_rank, log_every=100, log_name=name)
    by_set = {}
    sets = {"domain": probe_domain, "general": probe_others[0], "prose": probe_others[1]}
    for key, texts in sets.items():
        found = net.penalty(top, texts, donor_model, SEQ_LEN)
        found.pop("native_ce")
        by_set[key] = {"native_loss": found["native_loss"], "top_loss": found["top_loss"],
                       "penalty": found["penalty"], "tokens": found["tokens"]}
    report["by_set"] = by_set
    print(f"{name} penalty {top} before {report['before']['penalty']:.4f} after {report['after']['penalty']:.4f} "
          f"domain {by_set['domain']['penalty']:.4f} general {by_set['general']['penalty']:.4f} "
          f"prose {by_set['prose']['penalty']:.4f} accepted {report['accepted']} in {report['seconds']} s",
          flush=True)
    del donor_model
    torch.cuda.empty_cache()
    net.save(art, names=[top])
    volume.commit()
    return report


def row_base(cell):
    return {"cell": cell_name(cell), "pair": cell["pair"], "kind": cell["kind"], "top": cell["top"],
            "cut": cell["cut"], "seed": cell["seed"], "steps": cell["steps"], "skills": cell["skills"],
            "smoke": bool(cell.get("smoke"))}


def finish_net_row(row, net):
    counts = net.count_params()
    row["params"] = counts
    row["params_stored"] = counts["stored"]
    row["added_over_base"] = counts["added_over_base"]
    row["stored_over_base"] = counts["stored_over_base"]
    row["params_run_per_path"] = counts["run"]


def run_cell(cell):
    import gc

    import torch

    from wsp import skills_eval as ev
    from wsp.tops import ForcedView, SkillTops

    start = time.time()
    safe_reload()
    seed = cell["seed"]
    torch.manual_seed(seed)
    random.seed(seed)
    name = cell_name(cell)
    kind = cell["kind"]
    spec = ALL_PAIRS[cell["pair"]]
    base_id = spec["base"]
    row = row_base(cell)
    items = load_items(cell["skills"])
    print(f"{name} start kind {kind} skills {cell['skills']}", flush=True)

    if kind == "donor":
        donor_id = spec["tops"][cell["top"]]
        model, tokenizer = load_donor(donor_id)
        row["donor"] = donor_id
        row["scores"], row["counts"], _ = score_model(name, model, tokenizer, cell["skills"], items)
        row["tokens_per_second"] = decode_speed(model, tokenizer)
        row["params_total"], row["params_run"] = ev.model_params(model)
        return finish(row, start, name)

    base, base_tokenizer = load_base(base_id)
    base_layers = len(base.model.layers)
    if base_layers != spec["n_layers"]:
        raise ValueError(f"{base_id} has {base_layers} layers, the cell plan assumed {spec['n_layers']}")

    if kind == "base":
        row["scores"], row["counts"], _ = score_model(name, base, base_tokenizer, cell["skills"], items)
        row["tokens_per_second"] = decode_speed(base, base_tokenizer)
        row["params_total"], row["params_run"] = ev.model_params(base)
        return finish(row, start, name)

    if kind == "kd_lora":
        from wsp.lora import rank_for_budget
        from wsp.tops import fit_kd_lora, stitch_param_count

        top = cell["top"]
        donor_id = spec["tops"][top]
        donor, _ = load_donor(donor_id)
        budget = stitch_param_count("mlp", base.config.hidden_size, donor.config.hidden_size)
        rank = rank_for_budget(base, budget)
        fit_texts, _, _ = texts_for(base_tokenizer, base_id, top, FIT, "stitch")
        report = fit_kd_lora(base, donor, base_tokenizer, fit_texts, rank, steps=cell["steps"], batch_size=BATCH,
                             seq_len=SEQ_LEN, seed=seed)
        print(f"{name} kd_lora rank {rank} params {report['params']} budget {budget} loss "
              f"{report['loss_first']:.4f} to {report['loss_last25']:.4f} in {report['seconds']} s", flush=True)
        del donor
        gc.collect()
        torch.cuda.empty_cache()
        row["kd_lora"] = report
        row["budget_stitch_params"] = budget
        row["scores"], row["counts"], _ = score_model(name, base, base_tokenizer, cell["skills"], items)
        row["tokens_per_second"] = decode_speed(base, base_tokenizer)
        row["params_total"], row["params_run"] = ev.model_params(base)
        row["trainable_params"] = report["params"]
        return finish(row, start, name)

    # Forced, routed, scrambled, lora, template: all use a SkillTops network.
    net = SkillTops(base, base_tokenizer, base_id=base_id)
    net.allow_flagged = True
    names = list(spec["tops"].keys()) if kind == "routed" else [cell["top"]]
    row["fit"] = {}
    for top in names:
        report = fit_one_top(cell, net, top, base_tokenizer, base_id)
        row["fit"][top] = report
        penalty = report["after"]["penalty"] if report else None
        row.setdefault("penalty", {})[top] = penalty
        row.setdefault("flagged", {})[top] = (not net.tops[top].accepted)
        row.setdefault("vocab", {})[top] = net.tops[top].spec["vocab"]
        if penalty is not None and penalty > ABORT_PENALTY:
            row["skipped"] = f"stitch penalty {penalty:.3f} above {ABORT_PENALTY}; not scored"
            print(f"{name} {row['skipped']}", flush=True)
            finish_net_row(row, net)
            return finish(row, start, name)
    finish_net_row(row, net)

    if kind == "routed":
        texts = router_classes_texts(base_tokenizer, base_id, names, ROUTER, "router")
        report = net.fit_router(texts, lam=0.01, scale=8.0, base_margin=0.0, max_length=384, batch_size=8)
        probe = router_classes_texts(base_tokenizer, base_id, names, ROUTER_PROBE, "probe")
        tuned = net.tune_base_margin(probe)
        row["router"] = {"fit": report, "tune": tuned, "base_margin": net.base_margin}
        print(f"{name} router classes {report['classes']} cut {report['cut']} margin {net.base_margin} "
              f"train accuracy {json.dumps({k: round(v, 3) for k, v in report['train_accuracy'].items()})}",
              flush=True)
        finish_net_row(row, net)
        row["scores"], row["counts"], row["route_shares"] = score_model(name, net, base_tokenizer, cell["skills"],
                                                                        items, tally_owner=net)
        model = net
    else:
        model = ForcedView(net, cell["top"])
        tokenizer = base_tokenizer
        if kind == "template":
            import transformers

            tokenizer = transformers.AutoTokenizer.from_pretrained(spec["tops"][cell["top"]])
            row["template_of"] = spec["tops"][cell["top"]]
        row["scores"], row["counts"], _ = score_model(name, model, tokenizer, cell["skills"], items)
    row["tokens_per_second"] = decode_speed(model, base_tokenizer)
    return finish(row, start, name)


def finish(row, start, name):
    row["seconds"] = round(time.time() - start, 1)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, name + ".json"), "w") as f:
        json.dump(row, f, default=str)
    volume.commit()
    print(f"{name} done in {row['seconds']} s scores {json.dumps(row.get('scores'))}", flush=True)
    return row


@app.function(image=image, volumes={"/vol": volume}, gpu=GPU_SPEC["gpu"], memory=GPU_SPEC["memory"],
              timeout=GPU_SPEC["timeout"])
def cell(spec):
    return run_cell(spec)


@app.function(image=image, volumes={"/vol": volume}, memory=16384, timeout=4 * 3600)
def prefetch(donor_id, first_layer):
    """Fetch a donor's layers first_layer..end, final norm, and head by byte range into the volume cache."""
    import transformers

    from wsp.parts import block_tensor_names
    from wsp.skill_moe import fetch_expert_tensors

    safe_reload()
    config = transformers.AutoConfig.from_pretrained(donor_id)
    n = config.num_hidden_layers
    names = block_tensor_names(config, first_layer, n - 1)
    head = "model.embed_tokens.weight" if getattr(config, "tie_word_embeddings", False) else "lm_head.weight"
    os.makedirs(FETCH_DIR, exist_ok=True)
    fetch_expert_tensors(donor_id, names + ["model.norm.weight", head], FETCH_DIR)
    volume.commit()
    return {"donor": donor_id, "tensors": len(names) + 2}


# ------------------------------------------------------------------ summary


def mean(values):
    return sum(values) / len(values) if values else None


def summarize(rows):
    """Print the go rule per pair from result rows. Returns a dict of the numbers.

    Scores are fractions. Points are scores times 100. Seeds are averaged.
    """
    out = {}
    by = {}
    for row in rows:
        if row.get("scores") is None:
            continue
        by.setdefault((row["pair"], row["kind"], row["top"], row["cut"]), []).append(row)

    def score(pair, kind, top, cut, skill):
        found = by.get((pair, kind, top, cut), [])
        values = [r["scores"][skill] for r in found if skill in r["scores"]]
        return mean(values)

    for pair, spec in PAIRS.items():
        base_row = by.get((pair, "base", None, None), [])
        if not base_row:
            continue
        default = default_cut(spec["n_layers"])
        info = {"pair": pair, "forced": {}, "deepest_cut_within_3": {}}
        for top in spec["tops"]:
            donor_scores = {}
            for skill in OWN_SKILLS[top]:
                donor_scores[skill] = score(pair, "donor", top, None, skill)
            info["forced"][top] = {}
            deepest = None
            for cut in cuts_for(spec["n_layers"]):
                gaps = []
                for skill in OWN_SKILLS[top]:
                    value = score(pair, "forced", top, cut, skill)
                    if value is not None and donor_scores[skill] is not None:
                        gaps.append(100.0 * (donor_scores[skill] - value))
                if gaps:
                    worst = max(gaps)
                    info["forced"][top][cut] = worst
                    print(f"summary {pair} forced {top} cut {cut}: worst gap to donor {worst:.1f} points")
                    if worst <= 3.0 and (deepest is None or cut > deepest):
                        deepest = cut
            info["deepest_cut_within_3"][top] = deepest
            print(f"summary {pair} {top}: deepest cut within 3 points of the donor: {deepest}")
        info["routed"] = {}
        for cut in cuts_for(spec["n_layers"]):
            routed = by.get((pair, "routed", "all", cut), [])
            if not routed:
                continue
            r = routed[0]
            base_mmlu = score(pair, "base", None, None, "mmlu")
            checks = {}
            for top in spec["tops"]:
                for skill in OWN_SKILLS[top]:
                    donor_value = score(pair, "donor", top, None, skill)
                    routed_value = score(pair, "routed", "all", cut, skill)
                    if donor_value is not None and routed_value is not None:
                        checks[skill + "_within_3_of_donor"] = 100.0 * (donor_value - routed_value) <= 3.0
            routed_mmlu = score(pair, "routed", "all", cut, "mmlu")
            checks["mmlu_within_1_of_base"] = (base_mmlu is not None and routed_mmlu is not None
                                               and 100.0 * (base_mmlu - routed_mmlu) <= 1.0)
            checks["stored_within_limit"] = r["stored_over_base"] <= STORED_LIMIT[pair]
            worst_ratio = 0.0
            for path, run in r["params_run_per_path"].items():
                ref = base_row[0]["params_run"]
                if path != "base":
                    donor_rows = by.get((pair, "donor", path, None), [])
                    if donor_rows:
                        ref = max(ref, donor_rows[0]["params_run"])
                worst_ratio = max(worst_ratio, run / ref)
            checks["run_per_token_at_most_1.05x"] = worst_ratio <= 1.05
            scr_ok = True
            for top in spec["tops"]:
                for skill in OWN_SKILLS[top]:
                    scr = score(pair, "scrambled", top, default, skill)
                    base_value = score(pair, "base", None, None, skill)
                    if scr is None or base_value is None or scr > base_value:
                        scr_ok = False
            checks["scrambled_at_or_below_base"] = scr_ok
            go = all(checks.values())
            info["routed"][cut] = {"checks": checks, "stored_over_base": r["stored_over_base"],
                                   "run_ratio": worst_ratio, "go": go}
            print(f"summary {pair} routed at cut {cut}: {json.dumps(checks)} stored {r['stored_over_base']:.3f}x "
                  f"worst run ratio {worst_ratio:.3f} go {go}")
        info["go"] = any(entry["go"] for entry in info["routed"].values())
        out[pair] = info
    return out


def summarize_t4(rows):
    """Print the exploratory T4 numbers (GSM8K, seeds averaged) and return them. No go rule."""
    by = {}
    for row in rows:
        if row.get("pair") != "T4" or row.get("scores") is None:
            continue
        by.setdefault((row["kind"], row["cut"]), []).append(row)

    def value(kind, cut):
        found = by.get((kind, cut), [])
        values = [r["scores"]["gsm8k"] for r in found if "gsm8k" in r["scores"]]
        return mean(values), len(values)

    out = {}
    base, _ = value("base", None)
    donor, _ = value("donor", None)
    out["base"] = base
    out["donor"] = donor
    print(f"summary T4 (exploratory) base {base} donor {donor}")
    for kind in ("forced", "scrambled"):
        for cut in sorted({key[1] for key in by if key[0] == kind}):
            found, seeds = value(kind, cut)
            out[(kind, cut)] = {"gsm8k": found, "seeds": seeds}
            gap = None if found is None or donor is None else round(100.0 * (donor - found), 1)
            above = None if found is None or base is None else round(100.0 * (found - base), 1)
            print(f"summary T4 {kind} cut {cut}: GSM8K {found} over {seeds} seed(s); points below donor {gap}; "
                  f"points above base {above}")
    return out


def sizes_table():
    """Stored and per-token params per pair and cut from the real configs (needs Hub metadata, no weights)."""
    import transformers

    from wsp.tops import sizes_from_configs

    configs = {}

    def config(repo):
        if repo not in configs:
            configs[repo] = transformers.AutoConfig.from_pretrained(repo)
        return configs[repo]

    table = {}
    for pair, spec in PAIRS.items():
        base_cfg = config(spec["base"])
        for cut in cuts_for(spec["n_layers"]):
            tops = []
            for top, donor in spec["tops"].items():
                tops.append({"name": top, "donor_config": config(donor), "cut_host": cut})
            sizes = sizes_from_configs(base_cfg, tops, router_classes=1 + len(tops))
            table[(pair, cut)] = sizes
            run = {k: round(v / 1e9, 3) for k, v in sizes["run"].items()}
            print(f"sizes {pair} cut {cut}: base {sizes['base_total'] / 1e9:.3f}B stored "
                  f"{sizes['stored'] / 1e9:.3f}B ({sizes['stored_over_base']:.3f}x) run per path (B) {run}")
    return table


# ------------------------------------------------------------------ diagnostic (--diag)

DIAG_N = 32
DIAG_UNCACHED = 8
DIAG_UNCACHED_TOKENS = 96
DIAG_BATCH = 32
DIAG_FILE = "round24_diag.json"
DIAG_EDGES = [0, 128, 256, 384, 512, 100000]


class GenerateRecorder:
    """Stands in for a model inside wsp.skills_eval.generate_texts and keeps the prompt width and output ids of every call.

    generate_texts reads parameters() and generation_config and calls generate(). Everything goes through
    to the real model unchanged, so the recorded run is the skills_eval run.
    """

    def __init__(self, model):
        self.model = model
        self.generation_config = getattr(model, "generation_config", None)
        self.config = getattr(model, "config", None)
        self.calls = []

    def parameters(self):
        return self.model.parameters()

    def generate(self, **kwargs):
        out = self.model.generate(**kwargs)
        self.calls.append((int(kwargs["input_ids"].shape[1]), out.detach().cpu()))
        return out


def recorded_generate(model, tokenizer, prompts, max_new_tokens, batch_size):
    """Run wsp.skills_eval.generate_texts as the harness does. Return (texts, answers, stop_ids).

    answers[i] holds the new token ids of prompt i, cut after the first stop id (included).
    """
    from wsp import skills_eval as ev

    recorder = GenerateRecorder(model)
    texts = ev.generate_texts(recorder, tokenizer, prompts, max_new_tokens=max_new_tokens, batch_size=batch_size)
    order = sorted(range(len(prompts)), key=lambda index: -len(prompts[index]))
    stops = ev.stop_token_ids(model, tokenizer)
    answers = [None] * len(prompts)
    for call, (width, out) in enumerate(recorder.calls):
        chunk = order[call * batch_size:(call + 1) * batch_size]
        for row, index in enumerate(chunk):
            new = out[row, width:].tolist()
            for position, token in enumerate(new):
                if token in stops:
                    new = new[:position + 1]
                    break
            answers[index] = new
    return texts, answers, stops


def first_difference(a, b):
    """First index where two token lists differ, the shorter length when one is a prefix, None when equal."""
    for index in range(min(len(a), len(b))):
        if a[index] != b[index]:
            return index
    if len(a) != len(b):
        return min(len(a), len(b))
    return None


def uncached_greedy(forward, prompt_ids, max_new_tokens, stops, device):
    """Greedy decoding with a full forward on the growing sequence at every step. forward(ids) returns logits."""
    import torch

    ids = torch.tensor([prompt_ids], device=device)
    new = []
    for _ in range(max_new_tokens):
        with torch.no_grad():
            logits = forward(ids)
        token = int(logits[0, -1].float().argmax())
        new.append(token)
        if token in stops:
            break
        ids = torch.cat([ids, torch.tensor([[token]], device=device)], dim=1)
    return new


def bucket_of(position, edges=DIAG_EDGES):
    for index in range(len(edges) - 1):
        if edges[index] <= position < edges[index + 1]:
            return index
    return len(edges) - 2


def teacher_forced_against_donor(net, top, donor, prompt_ids, answer_ids, device):
    """The top against the donor on the donor's own answer, one sequence, no padding.

    Returns per-position lists for the answer tokens and for the prompt tokens (the first
    token is the sink and is left out): KL(donor || top), agreement of the two argmax,
    the top's and the donor's NLL of the token that follows, the first answer position at
    which each model's argmax is not the donor's answer token, and the relative error of
    the stitch output against the donor's true state, per position.
    """
    import torch

    from wsp.tops import first_index, positions_of

    ids = torch.tensor([list(prompt_ids) + list(answer_ids)], device=device)
    mask = torch.ones_like(ids)
    cut_donor = net.get_top(top).cut_donor
    with torch.no_grad():
        donor_out = donor(input_ids=ids, attention_mask=mask, output_hidden_states=True)
        donor_logits = donor_out.logits[0].float()
        true_state = donor_out.hidden_states[cut_donor][0].float()
        top_logits = net(ids, attention_mask=mask, top=top).logits[0].float()
        module = net.get_top(top)
        pos = positions_of(mask)
        h = net.run_trunk(ids, mask, pos, module.cut_host, None, 0)
        mapped = module.map_state(h, first_index(mask))[0].float()
    rows = min(donor_logits.shape[-1], top_logits.shape[-1])
    log_d = torch.log_softmax(donor_logits[:, :rows], dim=-1)
    log_t = torch.log_softmax(top_logits[:, :rows], dim=-1)
    kl = (log_d.exp() * (log_d - log_t)).sum(-1)
    agree = donor_logits[:, :rows].argmax(-1) == top_logits[:, :rows].argmax(-1)
    nxt = ids[0, 1:]
    nll_d = -log_d[:-1].gather(1, nxt[:, None])[:, 0]
    nll_t = -log_t[:-1].gather(1, nxt[:, None])[:, 0]
    err = ((mapped - true_state) ** 2).sum(-1) / (true_state ** 2).sum(-1).clamp(min=1e-12)
    start = len(prompt_ids)
    answer_positions = list(range(start - 1, start + len(answer_ids) - 1))
    prompt_positions = list(range(1, start - 1))

    def pick(values, positions):
        return [float(values[p]) for p in positions]

    def first_miss(logits):
        guess = logits[:, :rows].argmax(-1)
        for offset, position in enumerate(answer_positions):
            if int(guess[position]) != int(ids[0, position + 1]):
                return offset
        return None

    return {
        "answer_positions": answer_positions,
        "kl_answer": pick(kl, answer_positions), "kl_prompt": pick(kl, prompt_positions),
        "agree_answer": [bool(agree[p]) for p in answer_positions],
        "nll_top_answer": pick(nll_t, answer_positions), "nll_donor_answer": pick(nll_d, answer_positions),
        "state_err_answer": pick(err[:], [p + 1 for p in answer_positions]),
        "state_err_prompt": pick(err, prompt_positions),
        "first_miss_top": first_miss(top_logits), "first_miss_donor": first_miss(donor_logits),
        "sink_err": float(((mapped[0] - true_state[0]) ** 2).sum() / (true_state[0] ** 2).sum().clamp(min=1e-12)),
    }


def mean_of(values):
    return sum(values) / len(values) if len(values) > 0 else None


def diag_core(name, net, top, donor, tokenizer, items, max_new_tokens=512, batch_size=DIAG_BATCH,
              uncached_prompts=DIAG_UNCACHED, uncached_tokens=DIAG_UNCACHED_TOKENS, log=print):
    """Compare the donor alone with the forced top on the same GSM8K prompts. Return the report dict.

    Both models run through wsp.skills_eval.generate_texts with the same prompts, batch size,
    and token limit. Every printed line starts with the cell name.
    """
    import torch

    from wsp import skills_eval as ev
    from wsp.tops import ForcedView

    view = ForcedView(net, top)
    device = ev.model_device(donor)
    prompts = [ev.render_prompt(tokenizer, ev.user_text("gsm8k", item)) for item in items]
    d_texts, d_answers, d_stops = recorded_generate(donor, tokenizer, prompts, max_new_tokens, batch_size)
    t_texts, t_answers, t_stops = recorded_generate(view, tokenizer, prompts, max_new_tokens, batch_size)
    log(f"{name} diag stop ids donor {d_stops} top {t_stops} same {d_stops == t_stops}")
    d_records = ev.grade_outputs("gsm8k", items, d_texts)
    t_records = ev.grade_outputs("gsm8k", items, t_texts)
    prompt_ids = []
    for prompt in prompts:
        prompt_ids.append(tokenizer(prompt, add_special_tokens=False).input_ids)

    per_prompt = []
    kl_abs = [[] for _ in range(len(DIAG_EDGES) - 1)]
    err_abs = [[] for _ in range(len(DIAG_EDGES) - 1)]
    agree_all = []
    nll_gap = []
    prompt_kl_all = []
    for index in range(len(prompts)):
        tf = teacher_forced_against_donor(net, top, donor, prompt_ids[index], d_answers[index], device)
        start = len(prompt_ids[index])
        for offset, value in enumerate(tf["kl_answer"]):
            kl_abs[bucket_of(start + offset)].append(value)
        for offset, value in enumerate(tf["state_err_answer"]):
            err_abs[bucket_of(start + offset)].append(value)
        agree_all += tf["agree_answer"]
        for a, b in zip(tf["nll_top_answer"], tf["nll_donor_answer"]):
            nll_gap.append(a - b)
        prompt_kl_all += tf["kl_prompt"]
        first = first_difference(d_answers[index], t_answers[index])
        record = {"index": index, "prompt_tokens": start, "donor_tokens": len(d_answers[index]),
                  "top_tokens": len(t_answers[index]), "first_difference": first, "identical": first is None,
                  "donor_correct": bool(d_records[index]["correct"]), "top_correct": bool(t_records[index]["correct"]),
                  "kl_answer_mean": mean_of(tf["kl_answer"]), "kl_prompt_mean": mean_of(tf["kl_prompt"]),
                  "top1_agree_answer": mean_of([1.0 if v else 0.0 for v in tf["agree_answer"]]),
                  "nll_gap_answer": mean_of([a - b for a, b in zip(tf["nll_top_answer"], tf["nll_donor_answer"])]),
                  "teacher_forced_first_miss_top": tf["first_miss_top"],
                  "teacher_forced_first_miss_donor": tf["first_miss_donor"],
                  "state_err_prompt": mean_of(tf["state_err_prompt"]),
                  "state_err_answer": mean_of(tf["state_err_answer"]), "sink_err": tf["sink_err"],
                  "donor_pred": d_records[index]["pred"], "top_pred": t_records[index]["pred"],
                  "gold": d_records[index]["gold"]}
        per_prompt.append(record)
        log(f"{name} diag prompt {index} first_difference {first} donor_tokens {record['donor_tokens']} "
            f"top_tokens {record['top_tokens']} donor_correct {record['donor_correct']} "
            f"top_correct {record['top_correct']} kl_answer {record['kl_answer_mean']:.4f} "
            f"kl_prompt {record['kl_prompt_mean']:.4f} agree {record['top1_agree_answer']:.3f} "
            f"tf_first_miss_top {tf['first_miss_top']} tf_first_miss_donor {tf['first_miss_donor']} "
            f"sink_err {tf['sink_err']:.4f}")

    n = len(prompts)
    identical = sum(1 for r in per_prompt if r["identical"])
    donor_acc = sum(1 for r in per_prompt if r["donor_correct"]) / n
    top_acc = sum(1 for r in per_prompt if r["top_correct"]) / n
    by_position = []
    for bucket in range(len(DIAG_EDGES) - 1):
        by_position.append({"from": DIAG_EDGES[bucket], "to": DIAG_EDGES[bucket + 1],
                            "tokens": len(kl_abs[bucket]), "kl": mean_of(kl_abs[bucket]),
                            "stitch_state_rel_err": mean_of(err_abs[bucket])})
        log(f"{name} diag by_absolute_position {DIAG_EDGES[bucket]} to {DIAG_EDGES[bucket + 1]} "
            f"tokens {len(kl_abs[bucket])} kl_to_donor {by_position[-1]['kl']} "
            f"stitch_state_rel_err {by_position[-1]['stitch_state_rel_err']}")
    firsts = [r["first_difference"] for r in per_prompt if r["first_difference"] is not None]
    summary = {"n": n, "identical_share": identical / n, "donor_accuracy": donor_acc, "top_accuracy": top_acc,
               "median_first_difference": sorted(firsts)[len(firsts) // 2] if firsts else None,
               "kl_answer_mean": mean_of([v for bucket in kl_abs for v in bucket]),
               "kl_prompt_mean": mean_of(prompt_kl_all), "top1_agree_answer": mean_of(
                   [1.0 if v else 0.0 for v in agree_all]),
               "nll_gap_answer": mean_of(nll_gap), "by_absolute_position": by_position,
               "sink_err_mean": mean_of([r["sink_err"] for r in per_prompt])}
    log(f"{name} diag summary identical_share {summary['identical_share']:.3f} donor_accuracy {donor_acc:.4f} "
        f"top_accuracy {top_acc:.4f} median_first_difference {summary['median_first_difference']} "
        f"kl_answer {summary['kl_answer_mean']} kl_prompt {summary['kl_prompt_mean']} "
        f"top1_agree {summary['top1_agree_answer']} nll_gap {summary['nll_gap_answer']} "
        f"sink_err {summary['sink_err_mean']}")

    # Path check on the first prompts: batched cached, batch of one cached, and a full forward every step.
    # The donor runs the same three ways, so its own gaps are the noise floor of bf16 and caching.
    paths = {}
    count = min(uncached_prompts, n)
    short = prompts[:count]
    stops = set(d_stops)
    for label, model, forward, batched in (
            ("donor", donor, lambda ids: donor(input_ids=ids).logits, d_answers),
            ("top", view, lambda ids: net(ids, top=top).logits, t_answers)):
        _, single, _ = recorded_generate(model, tokenizer, short, max_new_tokens, 1)
        rows = []
        for index in range(count):
            full = uncached_greedy(forward, prompt_ids[index], uncached_tokens, stops, device)
            cached_batch = batched[index][:uncached_tokens]
            cached_one = single[index][:uncached_tokens]
            rows.append({"index": index, "batched_vs_one": first_difference(cached_batch, cached_one),
                         "batched_vs_uncached": first_difference(cached_batch, full),
                         "one_vs_uncached": first_difference(cached_one, full)})
            log(f"{name} diag path {label} prompt {index} batched_vs_one {rows[-1]['batched_vs_one']} "
                f"batched_vs_uncached {rows[-1]['batched_vs_uncached']} one_vs_uncached {rows[-1]['one_vs_uncached']}")
        paths[label] = rows
        clean = sum(1 for r in rows if r["batched_vs_uncached"] is None)
        log(f"{name} diag path {label} summary batched_equals_uncached {clean}/{count}")
    return {"cell": name, "n": n, "summary": summary, "per_prompt": per_prompt, "paths": paths,
            "stop_ids": {"donor": d_stops, "top": t_stops}, "texts": {"donor": d_texts[:4], "top": t_texts[:4]}}


def run_diag():
    """Restore the smoke cell's fitted top, then run diag_core on DIAG_N GSM8K test prompts. Needs one GPU."""
    import torch

    from wsp.tops import SkillTops

    safe_reload()
    cell = smoke_cell("T1")
    name = cell_name(cell)
    spec = ALL_PAIRS[cell["pair"]]
    top = cell["top"]
    art = artifact_dir(cell, top)
    print(f"{name} diag artifact {art} exists {os.path.exists(os.path.join(art, 'tops.json'))}", flush=True)
    if not os.path.exists(os.path.join(art, "tops.json")):
        raise FileNotFoundError(f"{name}: no fitted top at {art}; run the smoke cell first")
    base, tokenizer = load_base(spec["base"])
    donor, _ = load_donor(spec["tops"][top])
    net = SkillTops(base, tokenizer, base_id=spec["base"])
    net.allow_flagged = True
    net.add_saved_tops(art, names=[top], device="cuda", cache_dir=FETCH_DIR)
    module = net.tops[top]
    print(f"{name} diag restored top {top} cut_host {module.cut_host} cut_donor {module.cut_donor} "
          f"has_sink {module.has_sink} accepted {module.accepted} penalty "
          f"{(module.fit_report or {}).get('penalty')}", flush=True)
    items = load_items(["gsm8k"])["gsm8k"][:DIAG_N]
    report = diag_core(name, net, top, donor, tokenizer, items, max_new_tokens=512)
    report["artifact"] = art
    report["fit_version"] = FIT_VERSION
    report["torch"] = torch.__version__
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, DIAG_FILE), "w") as f:
        json.dump(report, f, default=str)
    volume.commit()
    return report


@app.function(image=image, volumes={"/vol": volume}, gpu=GPU_SPEC["gpu"], memory=GPU_SPEC["memory"],
              timeout=GPU_SPEC["timeout"])
def diag_remote():
    return run_diag()


# ------------------------------------------------------------------ entry


@app.local_entrypoint()
def main(shard: int = 0, nshards: int = 1, pairs: str = "", cuts: str = "", only: str = "", seeds: str = "",
         smoke: bool = False, plan_only: bool = False, no_prefetch: bool = False, summary: str = "",
         diag: bool = False):
    if diag:
        report = diag_remote.remote()
        os.makedirs("results", exist_ok=True)
        with open("results/round24_diag.json", "w") as f:
            json.dump(report, f, indent=2, default=str)
        print(f"{report['cell']} diag written to results/round24_diag.json")
        return
    if summary:
        with open(summary) as f:
            rows = json.load(f)
        summarize(rows)
        summarize_t4(rows)
        return
    wants_t4 = "T4" in pairs.split(",")
    if smoke:
        todo = [smoke_cell("T4" if wants_t4 else "T1")]
    else:
        pool = cells()
        if wants_t4:
            pool = pool + optional_cells()
        todo = select(pool, pairs, cuts, only, seeds)
    mine = []
    for index, spec in enumerate(todo):
        if index % nshards == shard:
            mine.append(spec)
    print(f"shard {shard}/{nshards}: {len(mine)} cells")
    plan(mine)
    if plan_only:
        return
    if not no_prefetch:
        wanted = {}
        for spec in mine:
            if spec["kind"] in ("base", "donor", "kd_lora"):
                continue
            tops = list(ALL_PAIRS[spec["pair"]]["tops"]) if spec["kind"] == "routed" else [spec["top"]]
            for top in tops:
                donor = ALL_PAIRS[spec["pair"]]["tops"][top]
                first = int(round(spec["cut"] / ALL_PAIRS[spec["pair"]]["n_layers"] * 28))
                wanted[donor] = min(first, wanted.get(donor, first))
        calls = [(donor, first) for donor, first in wanted.items()]
        for found in prefetch.starmap(calls):
            print("prefetched", found, flush=True)
    rows = []
    out_path = "results/round24_smoke.json" if smoke else f"results/round24_shard{shard}.json"
    os.makedirs("results", exist_ok=True)
    for row in cell.map(mine, order_outputs=False, return_exceptions=True):
        if isinstance(row, Exception):
            print("cell failed:", repr(row)[:600])
            continue
        rows.append(row)
        print(f"{row['cell']} row scores {json.dumps(row.get('scores'))} penalty {json.dumps(row.get('penalty'))} "
              f"stored {row.get('params_stored')} tps {row.get('tokens_per_second')}", flush=True)
        with open(out_path, "w") as f:
            json.dump(rows, f, indent=2, default=str)
    summarize(rows)
    summarize_t4(rows)


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "sizes":
        sizes_table()
    elif len(sys.argv) > 1 and sys.argv[1] == "plan":
        plan()
