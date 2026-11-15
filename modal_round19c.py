"""Round 19c: skill removal across skills, sizes, and models. Extends run 19 (modal_round19.py).

Named 19c because another session wrote a different modal_round19b.py (a
follow-up of the code-only cells) while this one was built. Nothing here
reads or changes that file.

Run 19 showed that Qwen3-30B-A3B-Instruct-2507 keeps its code ability when
each layer keeps its top experts by code saliency (N=64 of 128: HumanEval
87.8, MBPP 79.0, no training, 16.0B total params; N=32: 57.9 and 55.3, then
73.8 and 61.9 after a 500-step heal; random, frequency and ratio rankings
collapsed). This round asks whether that is general: other skills, a size
knob, two skills at once, and a second model. wsp/slim.py does the removal and
the heal. This script only chooses cells and measures them. Every result is
one network that runs alone.

Skills and data. Three skills: code, math, medicine. For each, the ranking
text and the heal text come from the wsp.healdata pool of the same name
("code", "math", "medicine") under the default reported set, so no reported
benchmark is a source (healdata.check_sources runs before any fetch). No
benchmark data trains anything here and none ranks anything. Each skill
fetches its pairs once and slices them: ranking texts are [0, 256), heal
texts follow in [256, 256 + 4000), so the ranking never depends on the heal
length and the two ranges never overlap. Texts that hold a line of any
scoring item (HumanEval, MBPP, GSM8K, MATH-500, MedMCQA) are dropped with
wsp.slim.drop_overlap.

Scores (wsp.skills_eval, greedy, chat template for generation, the raw
multiple-choice prompt for MedMCQA and MMLU). Every cell scores all six
benchmarks, so a row shows what was kept and what was removed:
  code      HumanEval (164) and MBPP sanitized test (257)
  math      GSM8K (the first 200 of the seeded 500) and MATH-500 (500)
  medicine  MedMCQA (1000)
  general   MMLU (1000)
Skill scores in a row: code is the mean of HumanEval and MBPP, math is the
mean of GSM8K and MATH-500, medicine is MedMCQA. All are fractions.

Cells (68 in all; build_cells() is the list, and the tests count it).
Qwen/Qwen3-30B-A3B-Instruct-2507 (128 experts per layer, top 8, 48 layers),
57 cells:
  qwen_full                                        1
  per skill (code, math, medicine), 13 cells each, 39 in all:
    saliency N = 64, 48, 40, 32, 24, no heal       5
    saliency N = 64, 48, 40, 32, 24, 500-step heal 5
    saliency by mass 0.9, 0.8, 0.7 (N differs per layer), no heal   3
  random controls, no heal, N = 48 and 32, seeds 0, 1, 2          6
  random controls with the heal, seed 0, N = 48 and 32, per skill  6
  code plus math kept together (union ranking), budget N per layer
    in total: N = 64, 48, 32 no heal; N = 48, 32 with the heal    5
allenai/OLMoE-1B-7B-0924-Instruct (64 experts, top 8, 16 layers;
config checked on the Hub on 2026-11-14: model_type olmoe, num_experts 64,
num_experts_per_tok 8, norm_topk_prob false), 11 cells:
  olmoe_full                                       1
  saliency N = 48, 32, 24 for code and for math, no heal   6
  random N = 32, seeds 0, 1, 2, no heal            3
  olmoe_noop_all: N = 64 (every expert) by code saliency   1
The olmoe adapter exists in wsp/slim.py. OLMoE does not renormalize the
top-k weights, so removal shifts the kept weights a little (docs/slim.md).
The no-op cell keeps every expert, checks max |logit difference| against
the full model on two ranking texts before scoring, and writes it to the row
as noop_check. It must read 0 and the scores must equal olmoe_full.

Multi-skill ranking: wsp.slim.combine_stats divides each skill's saliency by
its own sum per layer and adds them, then the usual saliency ranking keeps
the top N. A skill with large raw numbers does not dominate.

Mass mode: per layer, the fewest experts (in saliency order) that take the
given share of the skill's saliency. Layers then keep different counts, so
the saved model needs wsp.slim.load_slim (plain_hf is false). Rows hold
kept_counts (the count per layer) and the total and run-per-token params.

Rows hold: cell, model, skills, rule, N or mass, seed, every score, params
total and run per token (before and after), decode tokens per second at
batch 1, disk bytes (measured for slimmed models, 2 bytes per parameter for
unslimmed ones and for N = all), heal seconds, kept expert ids and counts
per layer, the share of each kept skill's routing the kept experts take, and
per-item correctness for every scored benchmark. Every printed score line
starts with the cell name.

Heal (as run 19): LoRA r16 on the attention, kept routers trainable, KL to
the original model's logits plus LM loss, batch 8, seq 384, LoRA lr 2e-4,
router lr 2e-5, 20 warmup steps, 500 steps, on the kept skill's heal texts
(for the code plus math family, half from each).

Go rule, per skill, fixed before the run. decide(rows) computes it per model
and skill. The skill is a go on a model when at some N at or below half the
experts (64 for Qwen, 32 for OLMoE) that also has a random control at the
same N (N = 48 or 32 for Qwen, N = 32 for OLMoE):
  1. the kept skill's score is within 0.05 of the full model with no heal,
     or within 0.03 after the 500-step heal;
  2. the random controls at that N (no heal for the no-heal pass, the
     one-seed heal control for the heal pass) score at least 0.20 under the
     full model on the kept skill, and the best of them is under the
     saliency cell;
  3. each of the two dropped skills falls at least 0.10 below the full model
     in the same cell.
Qwen N = 64, 40, and 24 are reported but cannot decide the rule, because
they have no random control at the same N. The point values are one run;
the standard error of code and math skill scores is about 0.02 and of
MedMCQA about 0.015, so read a pass within 0.02 of a threshold as open.
Dropped-skill checks on OLMoE are weak when its full score on that skill is
already low.

Not in this round: the full model with a heal and no removal (run 19 had it
for code). A heal on skill text can lift a score by format alone, so a healed
pass is read against the full model with no heal, as the rule says, and a
reader who wants the format effect needs that control.

Cost estimate in GPU hours (assumptions, not measurements; run 19 did not
record cell times). The 30B model loads in about 15 minutes per container
start. Scoring one cell on an H200 with the HF eager MoE loop (about 9 tok/s
at batch 1, batched generation at 32) is assumed at 20 minutes for all six
benchmarks, saving a slimmed model to disk at 3 minutes, a 500-step heal at
30 minutes, and one ranking pass of 256 texts at 10 minutes per skill, once
because stats are cached on the volume.
  Qwen full: 20 min                                            0.3 h
  Qwen per skill: 5 no-heal at 23 min, 5 heal at 53 min, 3 mass at 23 min,
    one ranking pass 10 min = 7.7 h, times 3 skills            23.0 h
  Qwen random no heal 6 cells at 23 min                        2.3 h
  Qwen random heal 6 cells at 53 min                           5.3 h
  Qwen code plus math: 3 at 23 min, 2 at 53 min                2.9 h
  Qwen container loads: 12 chunks of at most 5 cells, 15 min   3.0 h
  OLMoE: 11 cells at 8 min, three ranking passes of 2 min, loads 1 h   about 2 h
  Total about 39 GPU hours (H200 for Qwen, A100 80GB for OLMoE).
The 30B family dominates. The Math-500 generation (1024 new tokens) is the
likeliest source of a larger scoring time on collapsed models, which run to
the token limit. Use --skills and --only to run less.

Order of work. ALWAYS run --smoke first and wait for its row before any
larger run (LESSONS.md: setup failures cost 20 to 40 minutes in four earlier
rounds, and CPU tests do not catch image, volume, or dataset problems). The
smoke cell is olmoe_code_sal32 (OLMoE, code, N = 32 by saliency, no heal), the
cheapest cell that exercises the image, the volume, the pool fetch, the stats
cache, slimming, the save, and all six benchmarks.

    modal run modal_round19c.py --smoke
    python scripts/modal_fanout.py modal_round19c.py round19c --profiles trisanth,trisanthsrini,ztrisanth
    modal run modal_round19c.py --shard 0 --nshards 3 --models olmoe
    modal run modal_round19c.py --only qwen_full,qwen_code_sal64 --skills code

Flags: --shard and --nshards (cell index modulo, over the cells left after
the filters), --only (comma separated cell names), --models (qwen, olmoe),
--skills (code, math, medicine; a cell stays when every skill it keeps is
listed, and cells that keep no skill, like full and random, always stay),
--smoke (exactly one cell, other flags ignored, row printed, written to
results/round19c_smoke.json).

Writes results/round19c_shard<i>.json after every cell, so a crash loses
one cell (a failed cell leaves a row with an error). scripts/modal_fanout.py
merges the shards into results/round19c_rows.json. Per-cell full records go
to /vol/results/round19c/<cell>.json. Ranking statistics cache in
/vol/slim/round19c, one file per (model, skill, hash of the ranking texts).
Each workspace has its own volume; nothing here is shared across workspaces.
"""

import hashlib
import json
import os
import random
import re
import shutil
import threading
import time

import modal

app = modal.App("wsp-round19c")
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

MODELS = {
    "qwen": {"id": "Qwen/Qwen3-30B-A3B-Instruct-2507", "experts": 128, "top_k": 8, "layers": 48},
    "olmoe": {"id": "allenai/OLMoE-1B-7B-0924-Instruct", "experts": 64, "top_k": 8, "layers": 16},
}
SKILLS = ["code", "math", "medicine"]
# Benchmarks that score each skill. Every cell scores all of them.
BENCH = {"code": ["humaneval", "mbpp"], "math": ["gsm8k", "math500"], "medicine": ["medmcqa"]}
SCORED = ["humaneval", "mbpp", "gsm8k", "math500", "medmcqa", "mmlu"]
GSM8K_COUNT = 200

RANK_TEXTS = 256
RANK_LENGTH = 384
HEAL_STEPS = 500
MAX_HEAL_TEXTS = 4000
POOL_PAIRS = RANK_TEXTS + MAX_HEAL_TEXTS + 600
HEAL = {"batch_size": 8, "seq_len": 384, "lora_rank": 16, "lora_lr": 2e-4, "router_lr": 2e-5, "warmup": 20}

QWEN_SIZES = [64, 48, 40, 32, 24]
QWEN_MASSES = [0.9, 0.8, 0.7]
QWEN_RANDOM_SIZES = [48, 32]
QWEN_MULTI = {"skills": ("code", "math"), "no_heal": [64, 48, 32], "heal": [48, 32]}
OLMOE_SIZES = [48, 32, 24]
OLMOE_SKILLS = ["code", "math"]
OLMOE_RANDOM_SIZE = 32
SEEDS = [0, 1, 2]
SMOKE_CELL = "olmoe_code_sal32"
CHUNK = {"qwen": 5, "olmoe": 12}
GPU = {"qwen": "H200", "olmoe": "A100-80GB"}

CELL_DIR = "/vol/results/round19c"
STATS_DIR = "/vol/slim/round19c"
PAIRS_DIR = "/vol/slim/round19c/pairs"


# ---------------------------------------------------------------- cells


def spec(model, skills, rule, keep=None, mass=None, seed=0, heal=0):
    return {"model": model, "skills": tuple(skills), "rule": rule, "keep": keep, "mass": mass, "seed": seed,
            "heal": heal}


def build_cells():
    """{name: spec}, in run order. A spec has model, skills (kept), rule, keep (N), mass, seed, heal (steps)."""
    cells = {}
    cells["qwen_full"] = spec("qwen", [], None)
    for skill in SKILLS:
        for n in QWEN_SIZES:
            cells[f"qwen_{skill}_sal{n}"] = spec("qwen", [skill], "saliency", keep=n)
        for n in QWEN_SIZES:
            cells[f"qwen_{skill}_sal{n}_heal{HEAL_STEPS}"] = spec("qwen", [skill], "saliency", keep=n,
                                                                  heal=HEAL_STEPS)
        for mass in QWEN_MASSES:
            cells[f"qwen_{skill}_mass{round(mass * 100)}"] = spec("qwen", [skill], "saliency", mass=mass)
    for n in QWEN_RANDOM_SIZES:
        for seed in SEEDS:
            cells[f"qwen_random{n}_s{seed}"] = spec("qwen", [], "random", keep=n, seed=seed)
    for skill in SKILLS:
        for n in QWEN_RANDOM_SIZES:
            cells[f"qwen_{skill}_random{n}_heal{HEAL_STEPS}_s0"] = spec("qwen", [skill], "random", keep=n, seed=0,
                                                                        heal=HEAL_STEPS)
    multi = QWEN_MULTI["skills"]
    label = "+".join(multi)
    for n in QWEN_MULTI["no_heal"]:
        cells[f"qwen_{label}_sal{n}"] = spec("qwen", multi, "saliency", keep=n)
    for n in QWEN_MULTI["heal"]:
        cells[f"qwen_{label}_sal{n}_heal{HEAL_STEPS}"] = spec("qwen", multi, "saliency", keep=n, heal=HEAL_STEPS)

    cells["olmoe_full"] = spec("olmoe", [], None)
    for skill in OLMOE_SKILLS:
        for n in OLMOE_SIZES:
            cells[f"olmoe_{skill}_sal{n}"] = spec("olmoe", [skill], "saliency", keep=n)
    for seed in SEEDS:
        cells[f"olmoe_random{OLMOE_RANDOM_SIZE}_s{seed}"] = spec("olmoe", [], "random", keep=OLMOE_RANDOM_SIZE,
                                                                 seed=seed)
    cells["olmoe_noop_all"] = spec("olmoe", ["code"], "saliency", keep=MODELS["olmoe"]["experts"])
    return cells


CELLS = build_cells()


def select_cells(models=None, skills=None, only=None, smoke=False, shard=0, nshards=1, cells=None):
    """The cell names this launch runs, in run order.

    smoke: exactly SMOKE_CELL, every other argument ignored. Otherwise the
    filters apply (models; skills, where a cell stays when every skill it
    keeps is listed; only), then the shard takes cells whose index modulo
    nshards equals shard.
    """
    if cells is None:
        cells = CELLS
    if smoke:
        return [SMOKE_CELL]
    if nshards < 1 or shard < 0 or shard >= nshards:
        raise ValueError(f"shard {shard} of {nshards} is not valid")
    names = list(cells)
    if only:
        wanted = [item.strip() for item in only.split(",") if item.strip()]
        for name in wanted:
            if name not in cells:
                raise ValueError(f"unknown cell {name!r}")
        names = [name for name in names if name in wanted]
    if models:
        keep_models = [item.strip() for item in models.split(",") if item.strip()]
        for item in keep_models:
            if item not in MODELS:
                raise ValueError(f"unknown model {item!r}; the models are {', '.join(MODELS)}")
        names = [name for name in names if cells[name]["model"] in keep_models]
    if skills:
        keep_skills = [item.strip() for item in skills.split(",") if item.strip()]
        for item in keep_skills:
            if item not in SKILLS:
                raise ValueError(f"unknown skill {item!r}; the skills are {', '.join(SKILLS)}")
        kept = []
        for name in names:
            if all(skill in keep_skills for skill in cells[name]["skills"]):
                kept.append(name)
        names = kept
    mine = []
    for index, name in enumerate(names):
        if index % nshards == shard:
            mine.append(name)
    return mine


def make_chunks(names, cells=None):
    """[(model key, [cell names])]: the cells of one model, CHUNK[model] at a time, so one load serves many cells."""
    if cells is None:
        cells = CELLS
    chunks = []
    for key in MODELS:
        group = [name for name in names if cells[name]["model"] == key]
        for start in range(0, len(group), CHUNK[key]):
            chunks.append((key, group[start:start + CHUNK[key]]))
    return chunks


# ---------------------------------------------------------------- data


def text_hash(texts):
    """Short hash of a list of texts, to tell which ranking texts a cached result used."""
    digest = hashlib.sha256()
    for text in texts:
        digest.update(text.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()[:12]


def overlap_keys(items):
    """Strings from every scoring item. A ranking or heal text that holds one is dropped."""
    keys = []
    for item in items["humaneval"]:
        found = re.search(r'"""\s*(.+)', item["prompt"])
        if found:
            keys.append(found.group(1).strip())
        stripped = item["prompt"].strip()
        keys.append(stripped.splitlines()[-1] if stripped else "")
    for item in items["mbpp"]:
        keys.append(item["text"])
    for item in items["gsm8k"]:
        keys.append(item["question"])
    for item in items["math500"]:
        keys.append(item["problem"])
    for item in items["medmcqa"]:
        question = item["prompt"].split("\nA. ")[0]
        if question.startswith("Question: "):
            question = question[len("Question: "):]
        keys.append(question)
    return keys


_PAIRS = {}


def skill_pairs(skill):
    """(user, assistant) pairs of one healdata pool: one fetch per skill, kept on the volume and in memory."""
    from wsp import healdata

    if skill in _PAIRS:
        return _PAIRS[skill]
    path = f"{PAIRS_DIR}/{skill}.json"
    safe_reload()
    try:
        if os.path.exists(path):
            with open(path) as f:
                _PAIRS[skill] = [tuple(pair) for pair in json.load(f)]
            return _PAIRS[skill]
    except Exception as error:
        print("pairs cache read failed:", repr(error)[:200])
    ids = []
    for source in healdata.SOURCES[skill]:
        ids.append(source[0])
    healdata.check_sources(ids)
    pairs = healdata.pairs(skill, POOL_PAIRS, seed=0, skip=0)
    _PAIRS[skill] = pairs
    try:
        os.makedirs(PAIRS_DIR, exist_ok=True)
        temp = path + ".tmp"
        with open(temp, "w") as f:
            json.dump(pairs, f)
        os.replace(temp, path)
        volume.commit()
    except Exception as error:
        print("pairs cache write failed:", repr(error)[:200])
    return pairs


def skill_texts(skill, tokenizer, keys):
    """(ranking texts, heal texts, info) for one skill. No benchmark data trains anything here.

    The pairs are rendered with the tokenizer's chat template, texts that hold
    a scoring item are dropped, then the first RANK_TEXTS rank and the next
    MAX_HEAL_TEXTS heal.
    """
    from wsp import healdata, slim

    pairs = skill_pairs(skill)
    rendered = []
    for user, assistant in pairs:
        rendered.append(healdata.render(tokenizer, user, assistant))
    texts, dropped = slim.drop_overlap(rendered, keys)
    if len(texts) < RANK_TEXTS + MAX_HEAL_TEXTS:
        raise ValueError(f"only {len(texts)} {skill} texts, need {RANK_TEXTS + MAX_HEAL_TEXTS}")
    rank = texts[:RANK_TEXTS]
    heal = texts[RANK_TEXTS:RANK_TEXTS + MAX_HEAL_TEXTS]
    info = {"pool": skill, "pairs": len(pairs), "dropped_overlap": dropped, "rank_hash": text_hash(rank)}
    return rank, heal, info


def heal_texts_for(skills, heal_by_skill, steps):
    """The heal texts of a cell: steps times batch size texts, split evenly over the kept skills."""
    need = steps * HEAL["batch_size"]
    if need > MAX_HEAL_TEXTS:
        raise ValueError(f"{steps} steps need {need} texts, the pool holds {MAX_HEAL_TEXTS}")
    per = (need + len(skills) - 1) // len(skills)
    out = []
    for skill in skills:
        out += heal_by_skill[skill][:per]
    random.Random(0).shuffle(out)
    return out[:need]


def eval_items(ev):
    return {"humaneval": ev.humaneval_items(), "mbpp": ev.mbpp_items(), "gsm8k": ev.gsm8k_items(GSM8K_COUNT),
            "math500": ev.math500_items(), "medmcqa": ev.medmcqa_items(), "mmlu": ev.mmlu_items()}


# ---------------------------------------------------------------- slimming


def zero_stats(model):
    """Stats with no counts, for random cells, which need only the layer and expert counts."""
    import torch

    from wsp import slim

    layers = {}
    for layer, moe, router, top_k, count in slim.moe_layers_of(model):
        layers[layer] = {"count": torch.zeros(count, dtype=torch.float64),
                         "saliency": torch.zeros(count, dtype=torch.float64), "tokens": 0}
    return {"layers": layers, "top_k": 0, "num_experts": 0}


def slim_for_spec(model, cell, stats_by_skill, share_experts=True):
    """(student, report, extra): remove experts as a cell says. extra holds the row fields about the choice.

    stats_by_skill: {skill: stats} for the kept skills. One skill ranks on
    its stats, several rank on slim.combine_stats of them, and random cells
    need no stats. Rule "saliency" with keep or mass.
    """
    from wsp import slim

    skills = list(cell["skills"])
    if cell["rule"] == "random":
        ranking_stats = zero_stats(model)
    elif len(skills) == 1:
        ranking_stats = stats_by_skill[skills[0]]
    else:
        ranking_stats = slim.combine_stats([stats_by_skill[skill] for skill in skills])
    student, report = slim.slim_moe(model, None, None, None, rule=cell["rule"], keep=cell["keep"],
                                    mass=cell["mass"], seed=cell["seed"], skill_stats=ranking_stats,
                                    share_experts=share_experts)
    counts = []
    for layer in sorted(report["counts"]):
        counts.append(report["counts"][layer])
    extra = {"kept": report["kept"], "kept_counts": counts, "uniform": report["uniform"],
             "params_total_after": report["params_total_after"], "params_run_after": report["params_run_after"]}
    if cell["rule"] != "random":
        share = {}
        for skill in skills:
            share[skill] = slim.kept_share(stats_by_skill[skill], report["kept"])
        extra["kept_share"] = share
    return student, report, extra


def noop_check(student, model, tokenizer, texts):
    """Max absolute logit difference between a student and the model on texts, and whether the argmax agrees."""
    import torch

    device = next(model.parameters()).device
    worst = 0.0
    same = True
    for text in texts:
        ids = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)["input_ids"].to(device)
        with torch.no_grad():
            a = model(input_ids=ids, use_cache=False).logits.float()
            b = student(input_ids=ids, use_cache=False).logits.float()
        worst = max(worst, float((a - b).abs().max()))
        if not bool((a.argmax(-1) == b.argmax(-1)).all()):
            same = False
    return {"max_abs_logit_diff": worst, "argmax_equal": same, "texts": len(texts)}


# ---------------------------------------------------------------- measuring


def skill_scores(scores):
    """{code, math, medicine, general}: the mean score of each skill's benchmarks, and MMLU."""
    out = {}
    for skill in SKILLS:
        total = 0.0
        for bench in BENCH[skill]:
            total += scores[bench]
        out[skill] = total / len(BENCH[skill])
    out["general"] = scores["mmlu"]
    return out


def measure(name, model, tokenizer, ev, items, row):
    """Score every benchmark and the decode speed into row. Prints one line per score, starting with the cell name."""
    result = ev.evaluate_suite(model, tokenizer, skills=SCORED, items=items, batch_size=32, log=False)
    row["scores"] = dict(result["scores"])
    for bench in SCORED:
        found = result["skills"][bench]
        print(f"{name} {bench} {found['score']:.4f} ({found['correct']}/{found['n']}) in {found['seconds']} s",
              flush=True)
    row["skill_scores"] = skill_scores(result["scores"])
    row["correct"] = {}
    for bench in SCORED:
        row["correct"][bench] = [int(bool(record["correct"])) for record in result["skills"][bench]["records"]]
    row["decode_tok_s"] = round(ev.decode_speed(model, tokenizer), 1)
    print(f"{name} decode {row['decode_tok_s']} tok/s", flush=True)
    return result


def save_cell(name, row, result):
    os.makedirs(CELL_DIR, exist_ok=True)
    full = dict(row)
    if result is not None:
        full["skills_detail"] = result["skills"]
    temp = f"{CELL_DIR}/{name}.json.tmp"
    with open(temp, "w") as f:
        json.dump(full, f, default=str)
    os.replace(temp, f"{CELL_DIR}/{name}.json")
    volume.commit()


def cached_stats(slim, key, compute):
    """Ranking statistics from the volume, or computed and stored."""
    path = f"{STATS_DIR}/{key}.pt"
    safe_reload()
    try:
        if os.path.exists(path):
            return slim.load_stats(path)
    except Exception as error:
        print("stats cache read failed:", repr(error)[:200])
    stats = compute()
    try:
        os.makedirs(STATS_DIR, exist_ok=True)
        temp = path + ".tmp"
        slim.save_stats(stats, temp)
        os.replace(temp, path)
        volume.commit()
    except Exception as error:
        print("stats cache write failed:", repr(error)[:200])
    return stats


def disk_size(slim, model, tokenizer, name):
    """Save the slimmed model to local disk, return its size in bytes and whether it is a plain HF model, delete it."""
    path = f"/tmp/round19c_{name}"
    try:
        manifest = slim.save_slim(model, path, tokenizer)
        return slim.directory_bytes(path), manifest["plain_hf"]
    finally:
        shutil.rmtree(path, ignore_errors=True)


def heal_row(report):
    keep = ["steps", "trainable_params", "router_params", "kd_first", "kd_last", "lm_first", "lm_last", "seconds"]
    out = {}
    for key in keep:
        out[key] = report[key]
    return out


# ---------------------------------------------------------------- one cell


class Context:
    """What one container holds across its cells: the model, tokenizer, items, and per-skill texts."""

    def __init__(self, key):
        import torch
        import transformers

        from wsp import skills_eval as ev

        self.key = key
        self.model_id = MODELS[key]["id"]
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(self.model_id)
        load_start = time.time()
        self.model = transformers.AutoModelForCausalLM.from_pretrained(
            self.model_id, dtype=torch.bfloat16, device_map="cuda").eval()
        print(f"{key} model loaded in {time.time() - load_start:.0f} s", flush=True)
        self.ev = ev
        self.items = eval_items(ev)
        self.keys = overlap_keys(self.items)
        self.texts = {}

    def texts_of(self, skill):
        if skill not in self.texts:
            self.texts[skill] = skill_texts(skill, self.tokenizer, self.keys)
        return self.texts[skill]


def run_cell(ctx, name):
    import torch

    from wsp import slim

    cell = CELLS[name]
    start = time.time()
    row = {"cell": name, "model": ctx.model_id, "model_key": ctx.key, "skills": list(cell["skills"]),
           "rule": cell["rule"], "keep": cell["keep"], "mass": cell["mass"], "seed": cell["seed"],
           "heal_steps": cell["heal"]}
    total, run = slim.count_params(ctx.model)
    row["params_total_before"] = total
    row["params_run_before"] = run
    student = ctx.model
    if cell["rule"] is None:
        row["params_total_after"] = total
        row["params_run_after"] = run
        row["disk_bytes"] = 2 * total
        row["disk_measured"] = False
    else:
        stats_by_skill = {}
        row["data"] = {}
        for skill in cell["skills"]:
            rank, heal_pool, info = ctx.texts_of(skill)
            row["data"][skill] = info
            key = f"{ctx.key}_{skill}_{RANK_TEXTS}_{RANK_LENGTH}_{info['rank_hash']}"
            stats_by_skill[skill] = cached_stats(
                slim, key, lambda rank=rank: slim.collect_moe_stats(ctx.model, ctx.tokenizer, rank, RANK_LENGTH))
        student, report, extra = slim_for_spec(ctx.model, cell, stats_by_skill)
        row.update(extra)
        row["experts_per_layer_mean"] = sum(extra["kept_counts"]) / len(extra["kept_counts"])
        all_experts = MODELS[ctx.key]["experts"]
        if cell["keep"] == all_experts:
            first = ctx.texts_of("code")[0][:2]
            row["noop_check"] = noop_check(student, ctx.model, ctx.tokenizer, first)
            print(f"{name} noop {json.dumps(row['noop_check'])}", flush=True)
        if cell["heal"] > 0:
            heal_by_skill = {}
            for skill in cell["skills"]:
                heal_by_skill[skill] = ctx.texts_of(skill)[1]
            texts = heal_texts_for(list(cell["skills"]), heal_by_skill, cell["heal"])
            healed = slim.heal(student, ctx.model, ctx.tokenizer, texts, cell["heal"],
                               batch_size=HEAL["batch_size"], seq_len=HEAL["seq_len"], lora_rank=HEAL["lora_rank"],
                               lora_lr=HEAL["lora_lr"], router_lr=HEAL["router_lr"], warmup=HEAL["warmup"],
                               seed=cell["seed"])
            row["heal"] = heal_row(healed)
            row["heal_seconds"] = healed["seconds"]
        if cell["keep"] == all_experts and cell["heal"] == 0:
            row["disk_bytes"] = 2 * row["params_total_after"]
            row["disk_measured"] = False
        else:
            row["disk_bytes"], row["plain_hf"] = disk_size(slim, student, ctx.tokenizer, name)
            row["disk_measured"] = True
    result = measure(name, student, ctx.tokenizer, ctx.ev, ctx.items, row)
    row["seconds"] = round(time.time() - start, 1)
    save_cell(name, row, result)
    if student is not ctx.model:
        del student
    torch.cuda.empty_cache()
    return row


def run_chunk_rows(key, names):
    """Yield one row per cell. A failed cell is a row with an error, so the rules see the gap."""
    import gc

    import torch

    try:
        ctx = Context(key)
    except Exception as error:
        for name in names:
            yield {"cell": name, "error": "load failed: " + repr(error)[:400], "scores": None}
        return
    for name in names:
        try:
            yield run_cell(ctx, name)
        except Exception as error:
            print(f"{name} FAILED {repr(error)[:300]}", flush=True)
            yield {"cell": name, "error": repr(error)[:500], "scores": None}
        gc.collect()
        torch.cuda.empty_cache()


@app.function(image=image, volumes={"/vol": volume}, gpu=GPU["qwen"], memory=262144, timeout=20 * 3600)
def run_qwen_chunk(names):
    for row in run_chunk_rows("qwen", names):
        yield row


@app.function(image=image, volumes={"/vol": volume}, gpu=GPU["olmoe"], memory=131072, timeout=10 * 3600)
def run_olmoe_chunk(names):
    for row in run_chunk_rows("olmoe", names):
        yield row


RUNNERS = {"qwen": run_qwen_chunk, "olmoe": run_olmoe_chunk}


# ---------------------------------------------------------------- go rule


def usable(row):
    return row is not None and row.get("error") is None and row.get("skill_scores") is not None


def half_of(key):
    return MODELS[key]["experts"] // 2


def decide_skill(rows, key, skill):
    """The go rule for one skill on one model. Returns {"pass": True, False, or None, "passes": [...], "checked": [...]}."""
    full = rows.get(f"{key}_full")
    out = {"pass": None, "passes": [], "checked": []}
    if not usable(full):
        out["missing"] = f"{key}_full"
        return out
    others = [s for s in SKILLS if s != skill]
    base = full["skill_scores"]
    random_sizes = QWEN_RANDOM_SIZES if key == "qwen" else [OLMOE_RANDOM_SIZE]
    sizes = QWEN_SIZES if key == "qwen" else OLMOE_SIZES
    undecided = False
    for n in sizes:
        if n > half_of(key) or n not in random_sizes:
            continue
        for healed in (False, True):
            tail = f"_heal{HEAL_STEPS}" if healed else ""
            name = f"{key}_{skill}_sal{n}{tail}"
            if name not in CELLS:
                continue
            row = rows.get(name)
            if not usable(row):
                undecided = True
                continue
            if healed:
                controls = [rows.get(f"{key}_{skill}_random{n}_heal{HEAL_STEPS}_s0")]
            else:
                controls = [rows.get(f"{key}_random{n}_s{seed}") for seed in SEEDS]
            if not all(usable(c) for c in controls):
                undecided = True
                continue
            tolerance = 0.03 if healed else 0.05
            near = base[skill] - row["skill_scores"][skill] <= tolerance
            best_random = max(c["skill_scores"][skill] for c in controls)
            collapse = base[skill] - best_random >= 0.20 and best_random < row["skill_scores"][skill]
            fall = all(base[o] - row["skill_scores"][o] >= 0.10 for o in others)
            entry = {"cell": name, "n": n, "healed": healed, "near_full": near, "random_collapse": collapse,
                     "others_fall": fall}
            out["checked"].append(entry)
            if near and collapse and fall:
                out["passes"].append(name)
    if len(out["passes"]) > 0:
        out["pass"] = True
    elif undecided:
        out["pass"] = None
    else:
        out["pass"] = False
    return out


def decide(rows):
    """{model: {skill: decide_skill result}} for the skills each model runs."""
    verdict = {}
    for key in MODELS:
        verdict[key] = {}
        for skill in (SKILLS if key == "qwen" else OLMOE_SKILLS):
            verdict[key][skill] = decide_skill(rows, key, skill)
    return verdict


# ---------------------------------------------------------------- entrypoint


def short_row(row):
    """A row without the per-item arrays and the kept id lists, for printing."""
    out = {}
    for key in row:
        if key in ("correct", "kept"):
            continue
        out[key] = row[key]
    return out


def write_rows(path, rows):
    os.makedirs("results", exist_ok=True)
    temp = path + ".tmp"
    with open(temp, "w") as f:
        json.dump(rows, f, indent=2, default=str)
    os.replace(temp, path)


@app.local_entrypoint()
def main(shard: int = 0, nshards: int = 1, only: str = "", models: str = "", skills: str = "", smoke: bool = False):
    names = select_cells(models=models, skills=skills, only=only, smoke=smoke, shard=shard, nshards=nshards)
    if smoke:
        shard = 0
        path = "results/round19c_smoke.json"
    else:
        path = f"results/round19c_shard{shard}.json"
    print(f"shard {shard}/{nshards}: {len(names)} cells: {names}", flush=True)
    chunks = make_chunks(names)
    rows = []
    lock = threading.Lock()
    columns = ["skill_scores", "params_total_after", "params_run_after", "decode_tok_s", "disk_bytes",
               "heal_seconds"]

    def work(key, chunk):
        seen = set()
        try:
            for row in RUNNERS[key].remote_gen(chunk):
                seen.add(row["cell"])
                with lock:
                    rows.append(row)
                    write_rows(path, rows)
                    if row.get("error") is not None:
                        print(f"{row['cell']} FAILED {row['error']}", flush=True)
                    elif smoke:
                        print(row["cell"] + " row " + json.dumps(short_row(row), default=str), flush=True)
                    else:
                        print(row["cell"] + " row " + json.dumps({k: row.get(k) for k in columns}), flush=True)
        except Exception as error:
            print(f"chunk {key} {chunk} failed: {repr(error)[:400]}", flush=True)
        for name in chunk:
            if name not in seen:
                with lock:
                    rows.append({"cell": name, "error": "chunk ended before this cell", "scores": None})
                    write_rows(path, rows)

    threads = []
    for key, chunk in chunks:
        thread = threading.Thread(target=work, args=(key, chunk))
        thread.start()
        threads.append(thread)
    for thread in threads:
        thread.join()
    write_rows(path, rows)
    print(f"wrote {len(rows)} rows to {path}", flush=True)
