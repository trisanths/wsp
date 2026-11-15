"""Round 19b: follow-ups on round 19 (heal the 64-expert model, 48 experts, per-layer keep by saliency mass,
a longer heal at 32). The full-model row is round 19's moe_full, scored with this same code.

The rest of this docstring is round 19's.

Round 19: skill removal. Delete experts, layers, neurons, and heads that code does not need.

The owner's point: removing skills is as useful as adding them. If a model
only has to write code, its performance elsewhere does not matter, so the
parts that carry other skills can go. wsp/slim.py does the removal and the
heal; this script measures it. Each result is ONE network that runs alone,
loads with AutoModelForCausalLM (every layer keeps the same expert count,
and the dense models keep a plain config), and was healed for at most 1000
steps of LoRA r16 plus distillation from the original model's logits.

Skill: code. Ranking text and heal text are chat-format code solutions:
MBPP full train (task ids outside the sanitized test split; any id in that
split is dropped) and Python rows of ise-uiuc/Magicoder-OSS-Instruct-75K.
Texts that contain a HumanEval prompt line or an MBPP sanitized test prompt
are dropped (wsp.slim.drop_overlap). The general text for the selective and
general rules is the first turn of HuggingFaceH4/ultrachat_200k train_sft.

MoE model: Qwen/Qwen3-30B-A3B-Instruct-2507 (checked through the Hugging
Face API on 2026-11-14: model_type qwen3_moe, 48 layers, 128 experts, top-k
8, norm_topk_prob true, 30,532,122,624 parameters; the 2507 instruct
release has one chat mode, no thinking). Cells:
  moe_full, moe_full_heal500     (the second is the control for what any code heal adds)
  moe_saliency_64, moe_saliency_32, moe_saliency_16
  moe_freq_32, moe_selective_32
  moe_random_32_s0, _s1, _s2     (three seeds give the seed noise)
  moe_general_32
  moe_saliency_32_heal500, moe_saliency_16_heal500
  moe_random_32_heal500_s0, _s1, _s2   (three seeds for the healed branch)

Dense model: Qwen/Qwen2.5-Coder-7B-Instruct (28 layers, hidden 3584, MLP
18944, 28 query heads, 4 KV heads, 7,615,616,512 parameters; 1.09B of them
are the untied embedding and head, which slimming does not touch). Cells:
  dense_full
  dense_skill_3p1b_heal1000, dense_skill_1p6b_heal1000
  dense_random_3p1b_heal1000, dense_random_1p6b_heal1000
  dense_skill_3p1b_noheal, dense_skill_1p6b_noheal      (what the heal adds)
  dense_skill_3p1b_greedy_heal1000                      (layers chosen greedily)
  native_coder_3b, native_coder_1p5b                    (the existing-model bar)
  native_coder_3b_heal1000                              (same heal on the native 3B, the fair bar)
Sizes: layers go first, at most 25% of them, by loss increase on skill text
when the layer is skipped (wsp.atlas.layer_skill_map); then MLP neurons and
query heads thin by one fraction to reach the target (wsp.slim.plan_size),
chosen by mean activation on skill text. The random cells remove the same
counts at random. The 1.6B target leaves about 0.5B for 21 layers because
the embedding and head already take 1.09B, so the layers keep about a tenth
of their width. Expect that cell to fail; it marks where the path ends.

Scores (wsp.skills_eval, greedy). HumanEval, MBPP and GSM8K prompts use the
chat template; MMLU uses the raw round-6 multiple-choice prompt with no
template. Every arm of a model family gets the same prompts. HumanEval (164) and MBPP
sanitized test (257) are the kept skill; GSM8K (the first 200 of the
seeded 500) and MMLU (1000 items, log-prob of A to D) are expected to fall
and show what was removed. Rows also hold params total and run per token
before and after, decode tokens per second at batch 1, the size of the
saved directory in bytes (measured for slimmed models, 2 bytes per
parameter for the unslimmed ones), heal seconds, and for MoE cells the kept
expert ids per layer and the share of skill routing they take.

Heal: LoRA r16 on the attention (dense: attention and MLP), kept routers
trainable (MoE), KL to the original model's logits plus LM loss, batch 8,
seq 384, LoRA lr 2e-4, router lr 2e-5 (the router weights are small, so a
larger step reorders the top-k), 20 warmup steps. Steps: 500 (MoE) and 1000
(dense).

Go and stop rules, fixed before the run. code_mean is the mean of the
HumanEval and MBPP scores, a fraction between 0 and 1. decide(rows) computes
them from the merged shard rows.
 1. MoE. Go when either pass holds.
    No-heal pass: code_mean(moe_full) - code_mean(moe_saliency_32) <= 0.02,
    and code_mean(moe_saliency_32) - max(code_mean of moe_random_32_s0, s1, s2)
    is above 0 and above max - min of those three.
    Heal pass: max(code_mean of moe_full, moe_full_heal500) -
    code_mean(moe_saliency_32_heal500) <= 0.01, and code_mean(moe_saliency_32_heal500)
    - max(code_mean of moe_random_32_heal500_s0, s1, s2) is above 0 and above
    max - min of those three. A healed model that scores above the reference
    passes the gap test. The full model with the same heal is the reference
    so that the format and code adaptation of the heal does not count as
    recovery.
    Otherwise stop: expert counts this low are not a free cut for code on
    this model. A lower N that passes is a separate, smaller result.
 2. Dense. Go when code_mean(dense_skill_3p1b_heal1000) is above both
    code_mean(native_coder_3b) and code_mean(native_coder_3b_heal1000).
    Otherwise stop: depth and width removal does not beat a model trained
    at that size.
 3. Size. Every dense cell must land within 3% of its target, and the skill
    and random cells of one size must have the same total parameters.
Measurement error: code_mean averages 164 and 257 items, so its standard
error is near 0.017. The rows hold the per-item results of both code skills
(row["correct"]), and decide() reports a paired bootstrap 95% interval for each
gap. The rules above use point values; read the interval before trusting a
pass near a threshold.
The random controls decide whether the skill ranking mattered. The MMLU and
GSM8K scores are reported to show the removal; they are not part of a rule.

    python scripts/modal_fanout.py modal_round19.py round19 --profiles a,b,c

Writes results/round19b_shard<i>.json (a failed cell leaves a row with an error) and full per-item records to
/vol/results/round19/<cell>.json. Ranking statistics cache in /vol/slim/round19.
"""

import copy
import hashlib
import json
import os
import random
import re
import shutil
import time

import modal

app = modal.App("wsp-round19b")
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

MOE_MODEL = "Qwen/Qwen3-30B-A3B-Instruct-2507"
DENSE_MODEL = "Qwen/Qwen2.5-Coder-7B-Instruct"
NATIVE = {"native_coder_3b": "Qwen/Qwen2.5-Coder-3B-Instruct", "native_coder_1p5b": "Qwen/Qwen2.5-Coder-1.5B-Instruct"}
TARGETS = {"3p1b": 3.1e9, "1p6b": 1.6e9}

MOE_EXPERTS = 128
MOE_TOP_K = 8
MOE_LAYERS = 48

RANK_TEXTS = 256
RANK_LENGTH = 384
MAX_HEAL_TEXTS = 8000
POOL_TEXTS = RANK_TEXTS + MAX_HEAL_TEXTS + 8
SIZE_TOLERANCE = 0.03
GENERAL_TEXTS = 256
HEAL = {"batch_size": 8, "seq_len": 384, "lora_rank": 16, "lora_lr": 2e-4, "router_lr": 2e-5, "warmup": 20}
SCORED = ["humaneval", "mbpp", "gsm8k", "mmlu"]
GSM8K_COUNT = 200
CODE_SKILLS = ["humaneval", "mbpp"]
CELL_DIR = "/vol/results/round19b"
STATS_DIR = "/vol/slim/round19"


def build_cells():
    """{name: spec}. Round 19b: follow-ups on expert removal for code.

    keep is one count for every layer. mass keeps, per layer, the fewest
    experts (by saliency) that hold that share of the layer's code saliency.
    """
    cells = {}
    cells["moe_saliency_64_heal500"] = {"kind": "moe", "rule": "saliency", "keep": 64, "heal": 500}
    cells["moe_saliency_64_heal500_s1"] = {"kind": "moe", "rule": "saliency", "keep": 64, "heal": 500, "seed": 1}
    cells["moe_random_64_s0"] = {"kind": "moe", "rule": "random", "keep": 64, "seed": 0}
    cells["moe_random_64_heal500_s0"] = {"kind": "moe", "rule": "random", "keep": 64, "heal": 500, "seed": 0}
    cells["moe_saliency_48"] = {"kind": "moe", "rule": "saliency", "keep": 48}
    cells["moe_saliency_48_heal500"] = {"kind": "moe", "rule": "saliency", "keep": 48, "heal": 500}
    for share in [80, 90, 95]:
        cells[f"moe_mass{share}"] = {"kind": "moe", "rule": "saliency", "mass": share / 100.0}
        cells[f"moe_mass{share}_heal500"] = {"kind": "moe", "rule": "saliency", "mass": share / 100.0, "heal": 500}
    cells["moe_saliency_32_heal1500"] = {"kind": "moe", "rule": "saliency", "keep": 32, "heal": 1500}
    return cells


CELLS = build_cells()


# ---------------------------------------------------------------- data


def chat_text(tokenizer, user, assistant):
    messages = [{"role": "user", "content": user}, {"role": "assistant", "content": assistant}]
    return tokenizer.apply_chat_template(messages, tokenize=False)


def overlap_keys(ev):
    """Strings from the scoring items. A heal or ranking text that holds one is dropped."""
    keys = []
    for item in ev.humaneval_items():
        # The first docstring line of the prompt, which a copied task would repeat.
        found = re.search(r'"""\s*(.+)', item["prompt"])
        if found:
            keys.append(found.group(1).strip())
        keys.append(item["prompt"].strip().splitlines()[-1] if item["prompt"].strip() else "")
    for item in ev.mbpp_items():
        keys.append(item["text"])
    return keys


def code_pool(tokenizer, ev):
    """Chat-format code texts from wsp.healdata. Returns (texts, info).

    No benchmark data trains anything, and none ranks anything: the texts are Magicoder
    and CodeAlpaca instructions (the healdata "code" pool), never MBPP or
    HumanEval in any split. The pool does not depend on how many texts a
    cell needs, so every cell ranks on the same first RANK_TEXTS texts.
    Heal texts come after them.
    """
    from wsp import healdata, slim

    ids = []
    for source in healdata.SOURCES["code"]:
        ids.append(source[0])
    healdata.check_sources(ids)
    pool = healdata.pool("code", POOL_TEXTS * 2, tokenizer=tokenizer, seed=0)
    keys = overlap_keys(ev)
    texts, dropped = slim.drop_overlap(pool, keys)
    if len(texts) < POOL_TEXTS:
        raise ValueError(f"only {len(texts)} code texts, need {POOL_TEXTS}")
    texts = texts[:POOL_TEXTS]
    info = {"pools": ["code"], "sources": ids, "dropped_overlap": dropped,
            "pool": len(texts), "rank_hash": text_hash(texts[:RANK_TEXTS])}
    return texts, info


def text_hash(texts):
    """Short hash of a list of texts, to tell which ranking texts a cached result used."""
    digest = hashlib.sha256()
    for text in texts:
        digest.update(text.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()[:12]


def split_pool(texts, heal_steps, batch_size=None):
    """(ranking texts, heal texts). The ranking texts are the same for every heal length."""
    if batch_size is None:
        batch_size = HEAL["batch_size"]
    need = heal_steps * batch_size
    if need > MAX_HEAL_TEXTS:
        raise ValueError(f"{heal_steps} steps need {need} texts, the pool holds {MAX_HEAL_TEXTS}")
    return texts[:RANK_TEXTS], texts[RANK_TEXTS:RANK_TEXTS + need]


def stats_key(rank_texts):
    return f"qwen3_30b_code_{RANK_TEXTS}_{RANK_LENGTH}_{text_hash(rank_texts)}"


def general_pool(tokenizer, count):
    from datasets import load_dataset

    stream = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft", streaming=True)
    rows = []
    for row in stream:
        messages = row["messages"]
        if len(messages) >= 2 and messages[0]["role"] == "user" and messages[1]["role"] == "assistant":
            rows.append((messages[0]["content"], messages[1]["content"]))
        if len(rows) >= 3 * count:
            break
    random.Random(0).shuffle(rows)
    texts = []
    for user, assistant in rows[:count]:
        texts.append(chat_text(tokenizer, user, assistant))
    return texts


def eval_items(ev):
    return {"humaneval": ev.humaneval_items(), "mbpp": ev.mbpp_items(), "gsm8k": ev.gsm8k_items(GSM8K_COUNT),
            "mmlu": ev.mmlu_items()}


# ---------------------------------------------------------------- helpers


def load_causal(name, dtype):
    import transformers

    return transformers.AutoModelForCausalLM.from_pretrained(name, dtype=dtype, device_map="cuda").eval()


def cached_stats(slim, key, compute):
    """Ranking statistics from the volume, or computed and stored."""
    path = f"{STATS_DIR}/{key}.pt"
    try:
        safe_reload()
        if os.path.exists(path):
            return slim.load_stats(path)
    except Exception as error:
        print("stats cache read failed:", repr(error)[:200])
    stats = compute()
    try:
        os.makedirs(STATS_DIR, exist_ok=True)
        slim.save_stats(stats, path)
        volume.commit()
    except Exception as error:
        print("stats cache write failed:", repr(error)[:200])
    return stats


def code_mean(scores):
    total = 0.0
    for skill in CODE_SKILLS:
        total += scores[skill]
    return total / len(CODE_SKILLS)


def measure(model, tokenizer, ev, items, row):
    """Score the four skills and the decode speed into row."""
    result = ev.evaluate_suite(model, tokenizer, skills=SCORED, items=items, batch_size=32)
    row["scores"] = dict(result["scores"])
    row["code_mean"] = code_mean(result["scores"])
    row["correct"] = {}
    for skill in CODE_SKILLS:
        row["correct"][skill] = [int(bool(record["correct"])) for record in result["skills"][skill]["records"]]
    row["decode_tok_s"] = round(ev.decode_speed(model, tokenizer), 1)
    return result


def save_cell(name, row, result):
    os.makedirs(CELL_DIR, exist_ok=True)
    full = dict(row)
    if result is not None:
        full["skills"] = result["skills"]
    with open(f"{CELL_DIR}/{name}.json", "w") as f:
        json.dump(full, f, default=str)
    volume.commit()


def disk_size(slim, model, tokenizer, name):
    """Save the slimmed model to local disk, return its size in bytes, and delete it."""
    path = f"/tmp/round19_{name}"
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


# ---------------------------------------------------------------- cells


def run_moe(name, spec):
    import torch
    import transformers

    from wsp import skills_eval as ev
    from wsp import slim

    start = time.time()
    tokenizer = transformers.AutoTokenizer.from_pretrained(MOE_MODEL)
    model = load_causal(MOE_MODEL, torch.bfloat16)
    row = {"cell": name, "kind": "moe", "model": MOE_MODEL, "rule": spec.get("rule"), "keep": spec.get("keep"),
           "seed": spec.get("seed", 0), "heal_steps": spec.get("heal", 0)}
    total, run = slim.count_params(model)
    row["params_total_before"] = total
    row["params_run_before"] = run
    student = model
    if spec["rule"] is not None:
        heal_steps = spec.get("heal", 0)
        texts, info = code_pool(tokenizer, ev)
        rank_texts, heal_texts = split_pool(texts, heal_steps)
        row["data"] = info
        skill_stats = cached_stats(slim, stats_key(rank_texts),
                                   lambda: slim.collect_moe_stats(model, tokenizer, rank_texts, RANK_LENGTH))
        general_stats = None
        if spec["rule"] in ("selective", "general"):
            general_stats = cached_stats(
                slim, f"qwen3_30b_general_{GENERAL_TEXTS}_{RANK_LENGTH}",
                lambda: slim.collect_moe_stats(model, tokenizer, general_pool(tokenizer, GENERAL_TEXTS), RANK_LENGTH))
        # The student shares the frozen experts with the model, which stays the teacher.
        student, report = slim.slim_moe(model, tokenizer, None, None, rule=spec["rule"], keep=spec.get("keep"),
                                        mass=spec.get("mass"), min_keep=MOE_TOP_K,
                                        seed=spec.get("seed", 0), skill_stats=skill_stats,
                                        general_stats=general_stats, share_experts=True)
        row["kept"] = report["kept"]
        row["mass"] = spec.get("mass")
        counts = [len(report["kept"][layer]) for layer in sorted(report["kept"])]
        row["kept_counts"] = counts
        row["kept_mean"] = sum(counts) / len(counts)
        row["kept_share_of_skill_routing"] = slim.kept_share(skill_stats, report["kept"])
        row["params_total_after"] = report["params_total_after"]
        row["params_run_after"] = report["params_run_after"]
        if heal_steps > 0:
            healed = slim.heal(student, model, tokenizer, heal_texts, heal_steps, batch_size=HEAL["batch_size"],
                               seq_len=HEAL["seq_len"], lora_rank=HEAL["lora_rank"], lora_lr=HEAL["lora_lr"],
                               router_lr=HEAL["router_lr"], warmup=HEAL["warmup"], seed=spec.get("seed", 0))
            row["heal"] = heal_row(healed)
            row["heal_seconds"] = healed["seconds"]
        if spec.get("keep") == MOE_EXPERTS:
            # Nothing was removed: skip the 61 GB save.
            row["disk_bytes"] = 2 * report["params_total_after"]
            row["disk_measured"] = False
        else:
            row["disk_bytes"], row["plain_hf"] = disk_size(slim, student, tokenizer, name)
            row["disk_measured"] = True
    else:
        row["params_total_after"] = total
        row["params_run_after"] = run
        row["disk_bytes"] = 2 * total
        row["disk_measured"] = False
    items = eval_items(ev)
    result = measure(student, tokenizer, ev, items, row)
    row["seconds"] = round(time.time() - start, 1)
    save_cell(name, row, result)
    return row


def run_dense(name, spec):
    import torch
    import transformers

    from wsp import skills_eval as ev
    from wsp import slim

    start = time.time()
    tokenizer = transformers.AutoTokenizer.from_pretrained(DENSE_MODEL)
    model = load_causal(DENSE_MODEL, torch.bfloat16)
    row = {"cell": name, "kind": "dense", "model": DENSE_MODEL, "rule": spec.get("rule"), "size": spec.get("size"),
           "heal_steps": spec.get("heal", 0), "greedy": bool(spec.get("greedy"))}
    total, run = slim.count_params(model)
    row["params_total_before"] = total
    row["params_run_before"] = run
    student = model
    if spec["rule"] is not None:
        heal_steps = spec.get("heal", 0)
        texts, info = code_pool(tokenizer, ev)
        rank_texts, heal_texts = split_pool(texts, heal_steps)
        general = general_pool(tokenizer, GENERAL_TEXTS)
        row["data"] = info
        layer_rule = "random" if spec["rule"] == "random" else "skill_cost"
        width_rule = "random" if spec["rule"] == "random" else "skill"
        greedy_step = None
        if spec.get("greedy"):
            greedy_step = 1
        student, report = slim.slim_dense(model, tokenizer, rank_texts, general, target_params=TARGETS[spec["size"]],
                                          layer_rule=layer_rule, width_rule=width_rule, seed=0,
                                          greedy_step=greedy_step, max_length=RANK_LENGTH)
        row["dropped_layers"] = report["dropped"]
        row["keep_width"] = report["keep_mlp"]
        row["target_params"] = TARGETS[spec["size"]]
        row["size_error"] = abs(report["params_total_after"] - TARGETS[spec["size"]]) / TARGETS[spec["size"]]
        row["width_report"] = report["width"]
        row["params_total_after"] = report["params_total_after"]
        row["params_run_after"] = report["params_run_after"]
        if heal_steps > 0:
            healed = slim.heal(student, model, tokenizer, heal_texts, heal_steps, batch_size=HEAL["batch_size"],
                               seq_len=HEAL["seq_len"], lora_rank=HEAL["lora_rank"], lora_lr=HEAL["lora_lr"],
                               warmup=HEAL["warmup"], seed=0)
            row["heal"] = heal_row(healed)
            row["heal_seconds"] = healed["seconds"]
        row["disk_bytes"], row["plain_hf"] = disk_size(slim, student, tokenizer, name)
        row["disk_measured"] = True
    else:
        row["params_total_after"] = total
        row["params_run_after"] = run
        row["disk_bytes"] = 2 * total
        row["disk_measured"] = False
    items = eval_items(ev)
    result = measure(student, tokenizer, ev, items, row)
    row["seconds"] = round(time.time() - start, 1)
    save_cell(name, row, result)
    return row


def run_native(name, spec):
    import torch
    import transformers

    from wsp import skills_eval as ev
    from wsp import slim

    start = time.time()
    tokenizer = transformers.AutoTokenizer.from_pretrained(spec["model"])
    model = load_causal(spec["model"], torch.bfloat16)
    total, run = slim.count_params(model)
    heal_steps = spec.get("heal", 0)
    row = {"cell": name, "kind": "native", "model": spec["model"], "params_total_after": total,
           "params_run_after": run, "params_total_before": total, "params_run_before": run,
           "disk_bytes": 2 * total, "disk_measured": False, "heal_steps": heal_steps}
    student = model
    if heal_steps > 0:
        # Same heal data, steps and seed as the slimmed dense cells; the original is the teacher.
        texts, info = code_pool(tokenizer, ev)
        _, heal_texts = split_pool(texts, heal_steps)
        row["data"] = info
        student = copy.deepcopy(model)
        healed = slim.heal(student, model, tokenizer, heal_texts, heal_steps, batch_size=HEAL["batch_size"],
                           seq_len=HEAL["seq_len"], lora_rank=HEAL["lora_rank"], lora_lr=HEAL["lora_lr"],
                           warmup=HEAL["warmup"], seed=0)
        row["heal"] = heal_row(healed)
        row["heal_seconds"] = healed["seconds"]
    result = measure(student, tokenizer, ev, eval_items(ev), row)
    row["seconds"] = round(time.time() - start, 1)
    save_cell(name, row, result)
    return row


def run_cell_safe(name):
    """Run one cell. A failure becomes a row with an error, so the rules see the gap."""
    spec = CELLS[name]
    try:
        if spec["kind"] == "moe":
            return run_moe(name, spec)
        if spec["kind"] == "dense":
            return run_dense(name, spec)
        return run_native(name, spec)
    except Exception as error:
        return {"cell": name, "error": repr(error)[:500], "scores": None}


@app.function(image=image, volumes={"/vol": volume}, gpu="H200", memory=262144, timeout=10 * 3600)
def run_cell(name):
    return run_cell_safe(name)


# ---------------------------------------------------------------- rules


def paired_gap(a, b, draws=2000, seed=0):
    """(gap, low, high): code_mean of row a minus row b, with a paired bootstrap 95% interval.

    Items are resampled within each skill, the same indices for both rows.
    """
    rng = random.Random(seed)
    gaps = []
    for _ in range(draws):
        total = 0.0
        for skill in CODE_SKILLS:
            left = a["correct"][skill]
            right = b["correct"][skill]
            if len(left) != len(right):
                raise ValueError(f"{skill}: rows hold {len(left)} and {len(right)} items")
            n = len(left)
            picks = [rng.randrange(n) for _ in range(n)]
            total += (sum(left[i] for i in picks) - sum(right[i] for i in picks)) / n
        gaps.append(total / len(CODE_SKILLS))
    gaps.sort()
    return a["code_mean"] - b["code_mean"], gaps[int(0.025 * draws)], gaps[int(0.975 * draws) - 1]


def usable(row):
    return row is not None and row.get("error") is None and row.get("code_mean") is not None


def random_branch(rows, names):
    """(max, spread) of code_mean over the named random cells, or None when one is missing."""
    values = []
    for name in names:
        if not usable(rows.get(name)):
            return None
        values.append(rows[name]["code_mean"])
    return max(values), max(values) - min(values)


def moe_pass(rows, reference, saliency, randoms, tolerance):
    """One pass of rule 1. Returns a dict with the numbers and "pass" (True, False, or None if undecidable)."""
    needed = list(reference) + [saliency]
    for name in needed:
        if not usable(rows.get(name)):
            return {"pass": None, "missing": name}
    branch = random_branch(rows, randoms)
    if branch is None:
        return {"pass": None, "missing": "random cells"}
    best = 0.0
    for name in reference:
        best = max(best, rows[name]["code_mean"])
    value = rows[saliency]["code_mean"]
    gap = best - value
    over_random = value - branch[0]
    out = {"gap_to_full": gap, "over_random_max": over_random, "random_spread": branch[1], "tolerance": tolerance}
    out["pass"] = bool(gap <= tolerance and over_random > 0 and over_random > branch[1])
    ref_row = rows[reference[0]]
    if "correct" in ref_row and "correct" in rows[saliency]:
        out["gap_ci"] = paired_gap(ref_row, rows[saliency])[1:]
    return out


def decide(rows):
    """Apply the go and stop rules to {cell: row}. Returns a dict; "pass" is None when a cell is missing."""
    verdict = {}
    no_heal = moe_pass(rows, ["moe_full"], "moe_saliency_32", [f"moe_random_32_s{s}" for s in range(3)], 0.02)
    healed = moe_pass(rows, ["moe_full", "moe_full_heal500"], "moe_saliency_32_heal500",
                      [f"moe_random_32_heal500_s{s}" for s in range(3)], 0.01)
    verdict["moe_no_heal"] = no_heal
    verdict["moe_heal"] = healed
    if no_heal["pass"] or healed["pass"]:
        verdict["rule1"] = True
    elif no_heal["pass"] is None or healed["pass"] is None:
        verdict["rule1"] = None
    else:
        verdict["rule1"] = False
    slim_row = rows.get("dense_skill_3p1b_heal1000")
    bars = ["native_coder_3b", "native_coder_3b_heal1000"]
    if not usable(slim_row) or not all(usable(rows.get(b)) for b in bars):
        verdict["rule2"] = None
    else:
        verdict["rule2"] = all(slim_row["code_mean"] > rows[b]["code_mean"] for b in bars)
        verdict["rule2_gap_ci"] = [paired_gap(slim_row, rows[b])[1:] for b in bars
                                   if "correct" in rows[b] and "correct" in slim_row]
    sizes = []
    for name, row in rows.items():
        if name.startswith("dense_") and "size_error" in row:
            sizes.append(row["size_error"] <= SIZE_TOLERANCE)
    for size in TARGETS:
        for tail in ["heal1000"]:
            a = rows.get(f"dense_skill_{size}_{tail}")
            b = rows.get(f"dense_random_{size}_{tail}")
            if a is not None and b is not None and "params_total_after" in a and "params_total_after" in b:
                sizes.append(a["params_total_after"] == b["params_total_after"])
    verdict["rule3"] = all(sizes) if len(sizes) > 0 else None
    return verdict


@app.local_entrypoint()
def main(shard: int = 0, nshards: int = 1, only: str = ""):
    names = list(CELLS)
    if only:
        names = only.split(",")
    mine = []
    for index, name in enumerate(names):
        if index % nshards == shard:
            mine.append(name)
    print(f"shard {shard}/{nshards}: {mine}")
    rows = []
    for name, row in zip(mine, run_cell.map(mine, return_exceptions=True)):
        if isinstance(row, Exception):
            print("cell failed:", repr(row)[:600])
            rows.append({"cell": name, "error": repr(row)[:500], "scores": None})
            continue
        rows.append(row)
        if row.get("error") is not None:
            print("cell failed:", name, row["error"])
            continue
        print(json.dumps({k: row.get(k) for k in ["cell", "scores", "code_mean", "kept_mean", "params_total_after",
                                                  "params_run_after", "decode_tok_s", "disk_bytes"]}))
        if row.get("size_error") is not None and row["size_error"] > SIZE_TOLERANCE:
            print(f"size check failed: {name} is {row['size_error']:.3f} from its target")
    os.makedirs("results", exist_ok=True)
    with open(f"results/round19b_shard{shard}.json", "w") as f:
        json.dump(rows, f, indent=2, default=str)
