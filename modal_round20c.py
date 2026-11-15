"""Round 20c: the reader study. What makes a base able to use what a written-in donor block computes.

No benchmark data trains anything. Every text that trains a stitch, a translation, a gate, or a LoRA, or serves
as a distillation or probe text, comes from the wsp.healdata pools (code, general, math, prose) with the default
reported set excluded, checked with healdata.check_sources. Benchmarks only score the finished model, through
wsp.skills_eval.

Why. Run 14c: a frozen donor block plus a distillation heal adds about 4 points over distillation alone. The
translation into the block is nearly lossless, but the base's later layers read the block's output poorly, the
heals were 500 to 1500 steps, and the block was large. This round varies the reader, one factor at a time, at
more than toy size, and asks the owner's question: can a skill from another model be written into a base so the
result is near base size and clearly better than the base, a LoRA, and distillation?

Pairs (ids, shapes, and tokenizers checked on the Hub API, 2026-11-14; the tokenizer files are the same size in
all of them, and the logit tables differ only by unused tail rows):
  R1  base Qwen/Qwen2.5-1.5B-Instruct (28 layers, d 1536)   donor Qwen/Qwen2.5-Coder-7B-Instruct (28 layers, d 3584)
      same tokenizer, other width.
  R2  base Qwen/Qwen2.5-7B-Instruct (28 layers, d 3584)     donor Qwen/Qwen2.5-Coder-7B-Instruct
      same shapes, far weights (cosine about 0.42), so a seam and not a mix.
  R3  base Qwen/Qwen2.5-7B-Instruct                         donor Qwen/Qwen2.5-Coder-32B-Instruct (64 layers, d 5120)
Skill: code. The donor writes the answers to code prompts (cached once, then the donor generator is freed).
The heal teacher is the same donor, kept on the GPU during the heal.

Star design. One default cell per pair, and each factor varied alone around it. The factors:
  a  heal budget       stage 2 steps 500, 2000 (default), 8000
  b  block size        donor layers 2, 4, 8 (default), 13, starting at the same relative depth (8/28 of the donor)
  c  re-entry          mid-stack at the matching depth (default); near the top, two host layers left; the next
                       token's input (the feedback seam of wsp/feedback.py)
  d  reader capacity   LoRA rank 16 (default) or 64 on the host layers after the seam; LoRA on all host layers
                       instead of only after the seam (rank 16)
  e  teacher signal    donor logits on donor-written answers (default), or LM loss on the same text
The default cell runs with seeds 0 and 1. Every other star point runs with ONE SEED (seed 0) and its number is
a diagnostic, not a result: GSM8K moved 8 points between seeds in run 16b, and a control moved 8 points on MMLU
in run 14c. Only the default cell and its controls carry a two-seed spread.

Arms of a cell (the arm is the last part of the cell name):
  base       the base alone, one cell per pair
  real       the donor block, both stages
  scrambled  the same layers with scrambled weights (wsp.scramble): same shapes, norms, and spectra, no learned
             structure. The knowledge-free control. Same texts, steps, seed.
  random     another range of the same length of the same donor (the choice-of-layers control)
  kd_lora    distillation into a LoRA of matched trainable count (T_in, T_out, gate, and the real arm's LoRA),
             same texts, same total steps (stage 1 plus stage 2), same donor KL, same preservation term
  ref        reference rows: the base, the donor, and the best existing model at each compiled size of the
             same family (Qwen2.5-Coder 1.5B, 3B, 7B, 14B), scored with the same suite
Controls run for the default cell (seeds 0 and 1) and for the best cell of each factor (seed 0, run in a second
phase, --phase best, which reads the rows of the first phase). "Best" is the highest mean final probe transfer.
A control is a separate cell, so one cell holds one network.

Heal budget. Stage 1 trains the translations and gates (300 steps, wsp/synth.py), stage 2 adds the LoRA (the
factor a value). Batch 4, 512 tokens, lr_lora 2e-4 (untuned first guess, the same for every arm), preservation
term KL(frozen base || network) on the base's own answers to general, math, and medicine prompts. There is no
guard and no rollback (no practice set), so a cell is exactly what the heal made. A feedback cell trains T_in for
300 steps (wsp.stitch donor_lm) and then T_out, a gate, and a LoRA for the factor a value of steps with
wsp.feedback.heal_distill. Its batch order differs from the synth arms (its own shuffle), its loss covers every
token of the text and not only the answer, and its anchor weight is 0.5. kd_lora of a feedback config uses the same
heal_distill loop.

Probe. Held-out text from the healdata pools at pool positions [5000, 5096), never read by a heal: code (transfer
toward the donor: the share of KL(donor || base) that the written model closes, 0 is the base and 1 the donor),
general and math (drift: KL(base || written) in nats, the harm measure). Probed at step 0 of stage 2 and at stage 2
steps 250, 500, 1000, 2000, 4000, and 8000 where they are below the run's length, and at the end, so one long run
gives the whole curve. The step count in a point is the total heal steps (stage 1 plus stage 2). A point inside a
long run sits on a cosine schedule that has not decayed, so it is NOT the same as the end of a short run. The
factor a cells (500, 2000, 8000) are separate runs for that reason.

Scores. Once at the end through wsp.skills_eval: HumanEval 164, MBPP 257 (all sanitized test items), GSM8K 300
and MMLU 3000 (guards; MMLU has a standard error near 0.9 points at 3000 items, so the one-point rule below is
at the edge of what it resolves). Params total and run per token, params added over the base, trainable count,
heal seconds.

Go rule. A component is "read" when the real block beats kd_lora and the scrambled control on probe transfer by
more than the two-seed spread, and beats both on HumanEval or MBPP by more than 3 points, with MMLU within 1
point of the base. The two-seed spread is the largest spread among the real arm and the two controls of the
default config; a star point with one seed uses the default config's spread and, when it has no controls of its
own, the default config's controls (marked approximate). The round also reports the smallest block and the
shortest heal at which the rule holds, and the params added there (--verdict).

Rows. results/round20c_shard<i>.json is rewritten after every cell. Full rows also go to
/vol/results/round20c/<cell name>.json. Every printed line starts with the cell name.

    modal run modal_round20c.py --smoke                      one cell: R1 default, 200 heal steps, no controls
    modal run modal_round20c.py --estimate-only              print the cell list and the GPU-hour estimate, run nothing
    modal run modal_round20c.py --shard 0 --nshards 6        the star phase
    modal run modal_round20c.py --phase best --shard 0 --nshards 6      controls for the best cell of each factor
    modal run modal_round20c.py --verdict 'results/round20c_shard*.json'
"""

import glob
import json
import os
import random
import time

import modal

app = modal.App("wsp-round20c")
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

DATA_DIR = "/vol/round20c"
RESULT_DIR = "/vol/results/round20c"

QWEN_1_5B = "Qwen/Qwen2.5-1.5B-Instruct"
QWEN_7B = "Qwen/Qwen2.5-7B-Instruct"
CODER_1_5B = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
CODER_3B = "Qwen/Qwen2.5-Coder-3B-Instruct"
CODER_7B = "Qwen/Qwen2.5-Coder-7B-Instruct"
CODER_14B = "Qwen/Qwen2.5-Coder-14B-Instruct"
CODER_32B = "Qwen/Qwen2.5-Coder-32B-Instruct"

PAIRS = {
    "R1": {"base": QWEN_1_5B, "donor": CODER_7B},
    "R2": {"base": QWEN_7B, "donor": CODER_7B},
    "R3": {"base": QWEN_7B, "donor": CODER_32B},
}
REFERENCES = {
    "R1": [QWEN_1_5B, CODER_1_5B, CODER_3B, CODER_7B],
    "R2": [QWEN_7B, CODER_7B, CODER_14B],
    "R3": [QWEN_7B, CODER_7B, CODER_14B, CODER_32B],
}

SEEDS = [0, 1]
CONTROL_ARMS = ["scrambled", "random", "kd_lora"]
SKILLS = ["humaneval", "mbpp", "gsm8k", "mmlu"]
COUNTS = {"gsm8k": 300, "mmlu": 3000}
PROBE_SKILLS = ["code"]
PROBE_POOLS = ["code", "general", "math"]

DEFAULT_CFG = {"entry": "mid", "size": 8, "steps": 2000, "rank": 16, "scope": "after", "teacher": "kd"}
# (config name, factor letter, changes to the default). The default belongs to every factor.
STAR = [
    ("steps500", "a", {"steps": 500}),
    ("steps8000", "a", {"steps": 8000}),
    ("size2", "b", {"size": 2}),
    ("size4", "b", {"size": 4}),
    ("size13", "b", {"size": 13}),
    ("entry_top", "c", {"entry": "top"}),
    ("entry_feedback", "c", {"entry": "feedback"}),
    ("rank64", "d", {"rank": 64}),
    ("lora_all", "d", {"scope": "all"}),
    ("teacher_lm", "e", {"teacher": "lm"}),
]
FACTORS = ["a", "b", "c", "d", "e"]

HEAL = {"stage1_steps": 300, "batch_size": 4, "seq_len": 512, "steps_in": 100, "steps_lm": 60, "batch_in": 4,
        "tolerance": 0.04, "margin": 0.02, "lr_lora": 2e-4, "component_tests": False, "controls": None}
FEEDBACK_LR = {"lr_t_out": 3e-4, "lr_gate": 1e-3, "lr_lora": 2e-4}
ANCHOR_WEIGHT = 0.5
ONE_PASS_PROB = 0.15
STITCH = {"kind": "linear", "steps": 300, "lr": 1e-3, "batch_size": 8, "seq_len": 256}

GEN_TOKENS = 256
GSM8K_INSTRUCTION = "Solve step by step and end with: The answer is N."
CODE_INSTRUCTION = "Write the whole function, with its imports, in a single ```python code block."

# Pool positions. One fetch of POOL_TOTAL pairs per pool, so ranges never overlap.
POOL_TOTAL = 5200
DOMAIN_COUNT = 3000                      # code prompts [0, 3000) get the donor's answers
GENERAL_RANGES = [("general", 500, None), ("math", 150, GSM8K_INSTRUCTION), ("medicine", 150, None)]
ALIGN_PROSE = 150
PROBE_START = 5000
PROBE_COUNT = 96
POOLS = ["code", "general", "math", "medicine", "prose"]

# Unmeasured first guesses, only for --estimate. Seconds per heal step (both teachers and a block run on every
# step), minutes of scoring, minutes of loading, on one H200.
SEC_PER_STEP = {"R1": 1.0, "R2": 2.2, "R3": 4.0}
EVAL_MIN = {"R1": 8.0, "R2": 15.0, "R3": 15.0}
LOAD_MIN = {"R1": 3.0, "R2": 4.0, "R3": 9.0}
PROBE_MIN = 1.5
PEAK_GB = {"R1": 55, "R2": 70, "R3": 110}


# ---------------------------------------------------------------- configs and cells


def config_of(name):
    """The full config dict of a star point or of "default"."""
    cfg = dict(DEFAULT_CFG)
    cfg["name"] = name
    if name == "default":
        return cfg
    for star_name, _, change in STAR:
        if star_name == name:
            cfg.update(change)
            return cfg
    raise ValueError(f"unknown config {name!r}")


def factor_configs(factor):
    """Config names of one factor, the default included, in the order of the factor's values."""
    names = ["default"]
    for name, letter, _ in STAR:
        if letter == factor:
            names.append(name)
    order = {"a": ["steps500", "default", "steps8000"], "b": ["size2", "size4", "default", "size13"],
             "c": ["default", "entry_top", "entry_feedback"], "d": ["default", "rank64", "lora_all"],
             "e": ["default", "teacher_lm"]}
    return order[factor]


def cell_name(pair, config, arm, seed):
    if pair == "ref":
        return "ref." + config.split("/")[-1]
    if arm == "base":
        return f"{pair}.base.s{seed}"
    return f"{pair}.{config}.{arm}.s{seed}"


def star_cells(pairs=None, factors=None, seeds=None, refs=True):
    """The cells of the star phase: (pair, config, arm, seed). Reference cells are ("ref", model id, "ref", 0)."""
    pairs = list(pairs) if pairs else list(PAIRS)
    factors = list(factors) if factors else list(FACTORS)
    out = []
    for pair in pairs:
        out.append((pair, "base", "base", 0))
        configs = ["default"]
        for factor in factors:
            for name in factor_configs(factor):
                if name not in configs:
                    configs.append(name)
        for config in configs:
            run_seeds = SEEDS if config == "default" else [0]
            for seed in run_seeds:
                out.append((pair, config, "real", seed))
        for arm in CONTROL_ARMS:
            for seed in SEEDS:
                out.append((pair, "default", arm, seed))
    if refs:
        seen = []
        for pair in pairs:
            for model_id in REFERENCES[pair]:
                if model_id not in seen:
                    seen.append(model_id)
                    out.append(("ref", model_id, "ref", 0))
    if seeds:
        keep = set(str(s) for s in seeds)
        out = [c for c in out if str(c[3]) in keep or c[2] in ("base", "ref")]
    return out


def best_cells(rows, pairs=None):
    """Second phase: controls (scrambled, random, kd_lora, seed 0) for the best config of each factor.

    The best config of a factor is the one with the highest mean final probe transfer among its real rows. The
    default config already has its controls from the first phase, so it adds nothing. Cells already in rows are
    left out.
    """
    from wsp import reader

    pairs = list(pairs) if pairs else list(PAIRS)
    done = set()
    for row in rows:
        done.add((row.get("pair"), row.get("config"), row.get("arm"), row.get("seed")))
    out = []
    for pair in pairs:
        for factor in FACTORS:
            best = reader.pick_best(rows, pair, factor_configs(factor))
            if best is None or best == "default":
                continue
            for arm in CONTROL_ARMS:
                cell = (pair, best, arm, 0)
                if cell not in done and cell not in out:
                    out.append(cell)
    return out


def smoke_cells():
    """Exactly one cell: R1 default real, seed 0. The caller runs it with 200 heal steps and no controls."""
    return [("R1", "default", "real", 0)]


def select_cells(cells, only=None):
    """Keep the cells whose full name, arm, or config is in `only` (a list of names), or all of them."""
    if not only:
        return list(cells)
    wanted = set(only)
    return [c for c in cells if cell_name(*c) in wanted or c[2] in wanted or c[1] in wanted]


def cell_steps(cell):
    """Heal steps of a cell: stage 1 plus stage 2 (a base or reference cell has none)."""
    pair, config, arm, seed = cell
    if arm in ("base", "ref"):
        return 0
    return HEAL["stage1_steps"] + config_of(config)["steps"]


def cell_hours(cell):
    """Estimated H200 hours of one cell, from the unmeasured constants above."""
    pair, config, arm, seed = cell
    if pair == "ref":
        return 0.25
    minutes = LOAD_MIN[pair] + EVAL_MIN[pair]
    steps = cell_steps(cell)
    hours = steps * SEC_PER_STEP[pair] / 3600.0
    n_probes = 0
    if steps > 0:
        from wsp import reader

        n_probes = len(reader.checkpoint_steps(config_of(config)["steps"])) + 1
    if arm == "base":
        n_probes = 1
    return hours + (minutes + n_probes * PROBE_MIN) / 60.0


def estimate(cells):
    """GPU hours by pair, and the cell count."""
    by_pair = {}
    for cell in cells:
        pair = cell[0]
        entry = by_pair.setdefault(pair, {"cells": 0, "hours": 0.0})
        entry["cells"] += 1
        entry["hours"] += cell_hours(cell)
    return by_pair


def arm_plan(pair, config, arm, seed, smoke=False):
    """What a trained arm reads: its text sources, its steps, its batch. Equal across the arms of one config and seed."""
    cfg = config_of(config)
    steps = (100 if smoke else HEAL["stage1_steps"]) + (200 if smoke else cfg["steps"])
    return {"pair": pair, "config": config, "seed": seed, "domain_source": PAIRS[pair]["donor"],
            "general_source": PAIRS[pair]["base"], "domain_count": DOMAIN_COUNT, "total_steps": steps,
            "batch_size": HEAL["batch_size"], "seq_len": HEAL["seq_len"], "teacher": PAIRS[pair]["donor"],
            "signal": cfg["teacher"], "feedback": cfg["entry"] == "feedback"}


def shard_of(cells, shard, nshards):
    """Deal cells to shards, longest first, so shard loads are close."""
    ordered = sorted(cells, key=lambda c: (-cell_hours(c), cell_name(*c)))
    mine = []
    for index, cell in enumerate(ordered):
        if index % nshards == shard:
            mine.append(cell)
    return mine


# ---------------------------------------------------------------- pool text (wsp.healdata only)

_POOL_CACHE = {}


def source_ids():
    """Dataset ids of every pool this script reads. Raises when one is a reported benchmark."""
    from wsp import healdata

    ids = []
    for name in POOLS:
        for source in healdata.SOURCES[name]:
            ids.append(source[0])
    return healdata.check_sources(ids)


def pool_pairs(name, start, count):
    """Pairs [start, start + count) of one healdata pool. One fetch per pool, so ranges never overlap."""
    from wsp import healdata

    if name not in _POOL_CACHE:
        source_ids()
        _POOL_CACHE[name] = healdata.pairs(name, POOL_TOTAL, seed=0, skip=0)
    if start + count > len(_POOL_CACHE[name]):
        raise ValueError(f"pool {name} has {len(_POOL_CACHE[name])} pairs, need {start + count}")
    return _POOL_CACHE[name][start:start + count]


def prompt_of(pool, user):
    if pool == "math":
        return user + "\n" + GSM8K_INSTRUCTION
    if pool == "code":
        return user + "\n\n" + CODE_INSTRUCTION
    return user


def probe_texts(tokenizer):
    """Held-out probe texts: pool positions [5000, 5096) of code, general, and math, in the chat format."""
    from wsp import healdata

    out = {}
    for name in PROBE_POOLS:
        out[name] = [healdata.render(tokenizer, user, assistant)
                     for user, assistant in pool_pairs(name, PROBE_START, PROBE_COUNT)]
    return out


def answers_of(model, tokenizer, prompts):
    """Greedy answers of one model. Items whose answer hit the token limit are dropped."""
    from wsp import skills_eval as ev

    rendered = [ev.render_prompt(tokenizer, p) for p in prompts]
    outputs = ev.generate_texts(model, tokenizer, rendered, max_new_tokens=GEN_TOKENS, batch_size=64)
    pairs = []
    for prompt, answer in zip(prompts, outputs):
        answer = answer.strip()
        if len(answer) == 0:
            continue
        if len(tokenizer(answer, add_special_tokens=False)["input_ids"]) >= GEN_TOKENS - 2:
            continue
        pairs.append({"prompt": prompt, "answer": answer})
    return pairs


def slug(model_id):
    return model_id.split("/")[-1].lower()


def load_model(name):
    import torch
    import transformers

    return transformers.AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16, device_map={"": 0})


@app.function(image=image, volumes={"/vol": volume}, gpu="H200", memory=262144, timeout=6 * 3600)
def prep(kind, model_id):
    """Cache heal texts. kind "domain": the donor's greedy answers to code prompts. kind "general": the base's
    own answers to general, math, and medicine prompts, plus prose for the alignment fit. The generator is freed
    before any heal runs, so a 32B donor never shares the GPU with the heal for this step."""
    import torch
    import transformers

    from wsp import healdata

    os.makedirs(DATA_DIR, exist_ok=True)
    path = f"{DATA_DIR}/{kind}_{slug(model_id)}.json"
    safe_reload()
    if os.path.exists(path):
        return path
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)
    model = load_model(model_id).eval()
    if kind == "domain":
        prompts = [prompt_of("code", user) for user, _ in pool_pairs("code", 0, DOMAIN_COUNT)]
        data = {"items": answers_of(model, tokenizer, prompts)}
    else:
        prompts = []
        for pool, count, _ in GENERAL_RANGES:
            for user, _ in pool_pairs(pool, 0, count):
                prompts.append(prompt_of(pool, user))
        items = answers_of(model, tokenizer, prompts)
        align = healdata.pool("prose", ALIGN_PROSE, tokenizer=tokenizer, seed=3)
        for item in items[:100]:
            align.append(healdata.render(tokenizer, item["prompt"], item["answer"]))
        data = {"items": items, "align": align}
    del model
    torch.cuda.empty_cache()
    tmp = f"{path}.{os.getpid()}.tmp"
    with open(tmp, "w") as f:
        json.dump(data, f)
    # Modal volumes do not allow hard links. A plain rename; a file that is already there is kept.
    if os.path.exists(path):
        os.remove(tmp)
    else:
        os.replace(tmp, path)
    volume.commit()
    return path


def load_data(pair):
    spec = PAIRS[pair]
    safe_reload()
    with open(f"{DATA_DIR}/domain_{slug(spec['donor'])}.json") as f:
        domain = json.load(f)["items"]
    with open(f"{DATA_DIR}/general_{slug(spec['base'])}.json") as f:
        general = json.load(f)
    return {"domain": domain, "general": general["items"], "align": general["align"]}


def feedback_texts(tokenizer, data):
    """(texts, flags) for wsp.feedback.heal_distill: rendered chat texts, flag True for the general ones."""
    from wsp import healdata

    texts = []
    flags = []
    for item in data["domain"]:
        texts.append(healdata.render(tokenizer, item["prompt"], item["answer"]))
        flags.append(False)
    for item in data["general"]:
        texts.append(healdata.render(tokenizer, item["prompt"], item["answer"]))
        flags.append(True)
    return texts, flags


# ---------------------------------------------------------------- scoring items (scoring only)


def eval_items(ev):
    """HumanEval 164, MBPP 257, GSM8K 300, MMLU 3000. Reported benchmarks, read only to score."""
    return {"humaneval": ev.humaneval_items(), "mbpp": ev.mbpp_items(),
            "gsm8k": ev.gsm8k_items(count=COUNTS["gsm8k"], seed=0), "mmlu": ev.mmlu_items(count=COUNTS["mmlu"])}


def score_model(name, model, tokenizer):
    from wsp import skills_eval as ev

    items = eval_items(ev)
    result = ev.evaluate_suite(model, tokenizer, skills=SKILLS, items=items, batch_size=32, log=False)
    for skill in SKILLS:
        entry = result["skills"][skill]
        print(f"{name} score {skill} {entry['score']:.4f} ({entry['correct']}/{entry['n']})", flush=True)
    return dict(result["scores"]), {s: result["skills"][s]["n"] for s in SKILLS}


# ---------------------------------------------------------------- the cells


def probe_now(model, base_view, donor, tokenizer, held, cache, heal_steps):
    from wsp import reader

    return reader.timed_point(
        lambda: reader.probe_point(model, base_view, donor, tokenizer, held, PROBE_SKILLS, cache, max_length=384,
                                   batch_size=4), heal_steps)


def log_point(name, point):
    transfer = point["transfer"]["code"]["transfer"]
    shown = "none" if transfer is None else f"{transfer:.4f}"
    print(f"{name} probe heal_steps {point['heal_steps']} transfer_code {shown} "
          f"drift_general {point['drift']['general']['base_kl']:.4f} drift_math {point['drift']['math']['base_kl']:.4f}",
          flush=True)


def make_hook(name, model, base_view, donor, tokenizer, held, cache, curve, steps, offset):
    """A StepHook that appends a probe point to curve at each of steps; the point's heal step count is offset + step."""
    from wsp import reader

    def fire(step):
        point = probe_now(model, base_view, donor, tokenizer, held, cache, offset + step)
        curve.append(point)
        log_point(name, point)

    return reader.StepHook(steps, fire)


def wanted_checkpoints(smoke):
    return (0, 100) if smoke else None


def kd_rank_for(host, cfg, d_donor, seam_layer):
    """LoRA rank on all layers whose trainable count matches T_in + T_out + gate + the real arm's LoRA."""
    from wsp import reader
    from wsp.lora import params_per_rank

    n_layers = host.config.num_hidden_layers
    real_layers = reader.lora_layer_list(cfg["scope"], seam_layer, n_layers)
    per_real = params_per_rank(host, layers=real_layers)
    target = reader.trainable_target(host.config.hidden_size, d_donor, per_real, cfg["rank"])
    return reader.matched_rank(params_per_rank(host), target), target


def run_synth_arm(name, pair, cfg, arm, seed, host, tokenizer, donor, data, held, smoke, row):
    """real, scrambled, random, and kd_lora cells that use the wsp.synth pipeline. Returns (model, base_view, curve)."""
    from wsp import reader, synth
    from wsp.assemble import host_layer_count
    spec = PAIRS[pair]
    s1 = 100 if smoke else HEAL["stage1_steps"]
    s2 = 200 if smoke else cfg["steps"]
    n_host = host_layer_count(host)
    n_donor = donor.config.num_hidden_layers
    first, last = reader.block_range(n_donor, cfg["size"])
    at = reader.entry_layer(cfg["entry"], first, n_donor, n_host)
    row["block"] = [first, last]
    row["host_layer"] = at
    budget = dict(HEAL)
    budget.update({"stage1_steps": s1, "stage2_steps": s2, "lora_rank": cfg["rank"], "kd": cfg["teacher"] == "kd",
                   "seed": seed})
    texts = {"align": data["align"], "general": data["general"]}
    cache = {}
    curve = []
    donors = {spec["donor"]: (donor, tokenizer)}
    ks = reader.checkpoint_steps(s2, wanted_checkpoints(smoke) or reader.CHECKPOINTS)
    if arm == "kd_lora":
        rank, target = kd_rank_for(host, cfg, donor.config.hidden_size, at)
        row["kd_lora_rank"] = rank
        row["match_target"] = target
        budget["lora_rank"] = rank
        comps = [{"id": "coder", "donor": spec["donor"], "skill": "code", "texts": data["domain"]}]
        enc, _ = synth.prepare_data(tokenizer, comps, texts, synth.settings_of(budget, {}))
        s = synth.settings_of(budget, {})
        model = host
        base_view = reader.ContextView(model, synth.lora_off)
        hook = make_hook(name, model, base_view, donor, tokenizer, held, cache, curve, [s1 + k for k in ks], 0)
        with reader.hooked_stage2(synth, hook):
            heal = synth.lora_baseline(model, tokenizer, comps, donors, enc, s, s1 + s2)
        row["heal_seconds"] = heal["seconds"]
        row["lora_delta_ratio"] = heal.get("lora_delta_ratio")
        row["trainable"] = synth.param_counts(model)["lora"]
        return model, base_view, curve, cache, s1 + s2
    comp = {"id": "coder_block", "kind": "block", "donor": spec["donor"], "layers": [first, last], "skill": "code",
            "texts": data["domain"], "host_layer": at, "control_seed": seed}
    if arm == "scrambled":
        comp["control"] = "scrambled"
    elif arm == "random":
        comp["control"] = "random"
    model = host
    base_view = reader.ContextView(model, synth.frozen_base)
    hook = make_hook(name, model, base_view, donor, tokenizer, held, cache, curve, ks, s1)
    contexts = [reader.hooked_stage2(synth, hook)]
    if cfg["scope"] == "all":
        contexts.append(reader.lora_on_all_layers(synth))
    for context in contexts:
        context.__enter__()
    try:
        model, tokenizer, report = synth.synthesize((model, tokenizer), [comp], texts, budget, donors=donors)
    finally:
        for context in reversed(contexts):
            context.__exit__(None, None, None)
    keep = {k: report.get(k) for k in ("params", "place", "translate_in", "translate_out", "stage1", "stage2",
                                       "gates", "heal_seconds", "planned_lora")}
    row["report"] = synth.json_safe(keep)
    row["heal_seconds"] = report["heal_seconds"]
    counts = synth.param_counts(model)
    row["trainable"] = counts["glue"] + counts["gates"] + counts["lora"]
    row["params_by_kind"] = {k: counts[k] for k in ("parts", "glue", "gates", "lora")}
    return model, base_view, curve, cache, s1 + s2


def fit_t_in_for(name, pair, cfg, host, donor, tokenizer, read_layer, first, last, control, seed, held, data, smoke):
    """Stage 1 of a feedback cell: T_in on the donor's own LM loss with the sink, cached on the volume.

    For a scrambled control the scrambled layers are swapped into the donor while T_in trains, so T_in is not fit
    to the real block (round 22 did not do this).
    """
    import contextlib
    from types import SimpleNamespace

    import torch

    from wsp import synth
    from wsp.feedback import fit_t_in
    from wsp.insert import donor_block_tensors, donor_config_of
    from wsp.scramble import scramble_tensors
    from wsp.stitch import build_stitch

    steps = 60 if smoke else STITCH["steps"]
    path = f"{DATA_DIR}/t_in_{pair}_{first}_{last}_l{read_layer}_{control}_s{seed}_n{steps}.pt"
    d_host, d_donor = host.config.hidden_size, donor.config.hidden_size
    if os.path.exists(path):
        saved = torch.load(path)
        t_in = build_stitch({"kind": STITCH["kind"]}, d_host, d_donor)
        t_in.load_state_dict(saved["state"])
        return t_in, saved["sink"], saved["report"]
    from wsp import healdata

    stitch = []
    for item in data["domain"][:256]:
        stitch.append(healdata.render(tokenizer, item["prompt"], item["answer"]))
    eval_texts = held["code"][:16] + held["general"][:8] + held["math"][:8]
    context = contextlib.nullcontext()
    if control == "scrambled":
        tensors = donor_block_tensors(donor, donor_config_of(donor), first, last)
        scramble_tensors(tensors, seed)
        context = synth.scrambled_donor_layers(donor, SimpleNamespace(part_tensors=tensors), True)
    with context:
        t_in, sink, report = fit_t_in(host, tokenizer, read_layer, donor, tokenizer, first, stitch,
                                      kind=STITCH["kind"], steps=steps, lr=STITCH["lr"],
                                      batch_size=STITCH["batch_size"], seq_len=STITCH["seq_len"], seed=seed,
                                      eval_texts=eval_texts, sink=True, log_every=100)
    os.makedirs(DATA_DIR, exist_ok=True)
    tmp = f"{path}.{os.getpid()}.tmp"
    torch.save({"state": t_in.state_dict(), "sink": sink, "report": report}, tmp)
    if os.path.exists(path):
        os.remove(tmp)
    else:
        os.replace(tmp, path)
    volume.commit()
    return t_in, sink, report


def run_feedback_arm(name, pair, cfg, arm, seed, host, tokenizer, donor, data, held, smoke, row):
    """The feedback seam: the block output goes into the next token's input (wsp/feedback.py).

    Arms real, scrambled, random use a block branch; kd_lora is distillation into a LoRA of matched count, with the
    same heal_distill loop. Returns (model, base_view, curve, cache, total_steps).
    """
    import torch

    from wsp import feedback as fb
    from wsp import reader
    from wsp.lora import apply_lora

    s1 = 60 if smoke else STITCH["steps"]
    s2 = 200 if smoke else cfg["steps"]
    n_host = host.config.num_hidden_layers
    n_donor = donor.config.num_hidden_layers
    first, last = reader.block_range(n_donor, cfg["size"])
    read_layer = reader.entry_layer("feedback", first, n_donor, n_host)
    lora_layers = reader.lora_layer_list("after", read_layer, n_host)
    row["block"] = [first, last]
    row["read_layer"] = read_layer
    texts, flags = feedback_texts(tokenizer, data)
    cache = {}
    curve = []
    ks = reader.checkpoint_steps(s2, wanted_checkpoints(smoke) or reader.CHECKPOINTS)
    common = dict(steps=s2, batch_size=HEAL["batch_size"], seq_len=HEAL["seq_len"], seed=seed,
                  one_pass_prob=ONE_PASS_PROB, anchor_weight=ANCHOR_WEIGHT, autocast_dtype=torch.bfloat16)

    if arm == "kd_lora":
        rank, target = kd_rank_for(host, cfg, donor.config.hidden_size, read_layer)
        row["kd_lora_rank"] = rank
        row["match_target"] = target
        for param in host.parameters():
            param.requires_grad_(False)
        lora = apply_lora(host, rank)
        groups = [{"params": lora, "lr": FEEDBACK_LR["lr_lora"]}]
        model = host
        base_view = fb.BaseView(model)
        total = s1 + s2
        common["steps"] = total
        common["one_pass_prob"] = 0.0
        hook = make_hook(name, model, base_view, donor, tokenizer, held, cache, curve, [s1 + k for k in ks], 0)
        counter = reader.ForwardCounter(hook, host)

        def forward(ids, mask, passes):
            counter.tick()
            return host(input_ids=ids, attention_mask=mask, use_cache=False).logits

        def anchor(ids, mask):
            return fb.base_logits(host, ids, mask)

        log = fb.heal_distill(forward, anchor, groups, tokenizer, donor, texts, flags, **common)
        host.eval()
        row["heal_seconds"] = log["seconds"]
        row["trainable"] = sum(p.numel() for g in groups for p in g["params"])
        return model, base_view, curve, cache, total

    control = {"scrambled": "scrambled", "random": "random"}.get(arm)
    use_first = first
    use_last = last
    if control == "random":
        use_first, use_last = fb.random_range(n_donor, first, last, seed)
    t_in, sink, stage1 = fit_t_in_for(name, pair, cfg, host, donor, tokenizer, read_layer, use_first, use_last,
                                      control, seed, held, data, smoke)
    row["stage1"] = stage1
    branch = fb.make_block_branch(host, donor, [first, last], read_layer, t_in=t_in, sink=sink, control=control,
                                  control_seed=seed)
    row["donor_layers_used"] = branch.meta["donor_layers"]
    seam = fb.FeedbackSeam(host, [branch], PAIRS[pair]["base"])
    seam.add_host_lora(cfg["rank"], layers=lora_layers)
    groups = fb.stage2_groups(seam, **FEEDBACK_LR)
    seam.train()
    base_view = fb.BaseView(seam)
    hook = make_hook(name, seam, base_view, donor, tokenizer, held, cache, curve, ks, s1)

    counter = reader.ForwardCounter(hook, seam)

    def forward(ids, mask, passes):
        counter.tick()
        return seam(ids, mask, passes=passes).logits

    def anchor(ids, mask):
        return fb.base_logits(seam, ids, mask)

    log = fb.heal_distill(forward, anchor, groups, tokenizer, donor, texts, flags, **common)
    seam.eval()
    counts = fb.count_params(seam)
    row["heal_seconds"] = log["seconds"]
    row["trainable"] = counts["trainable"]
    row["params_by_kind"] = {k: counts[k] for k in ["host", "lora", "component", "t_in", "t_out", "gate"]}
    row["gate"] = [round(float(b.gate.detach()), 4) for b in seam.branches]
    return seam, base_view, curve, cache, s1 + s2


def run_cell(pair, config, arm, seed, smoke=False):
    import gc

    import torch
    import transformers

    from wsp import skills_eval as ev
    from wsp import synth

    start = time.time()
    name = cell_name(pair, config, arm, seed)
    source_ids()
    torch.manual_seed(seed)

    if pair == "ref":
        tokenizer = transformers.AutoTokenizer.from_pretrained(config)
        model = load_model(config).eval()
        total, run = ev.model_params(model)
        row = {"name": name, "pair": pair, "config": config, "arm": arm, "seed": seed, "model": config,
               "params_total": total, "params_per_token": run}
        row["scores"], row["scores_n"] = score_model(name, model, tokenizer)
        row["seconds"] = round(time.time() - start, 1)
        return row

    spec = PAIRS[pair]
    cfg = config_of("default" if arm == "base" else config)
    tokenizer = transformers.AutoTokenizer.from_pretrained(spec["base"])
    host = load_model(spec["base"]).eval()
    base_total, base_run = ev.model_params(host)
    donor = load_model(spec["donor"]).eval()
    data = load_data(pair)
    held = probe_texts(tokenizer)
    row = {"name": name, "pair": pair, "config": config, "arm": arm, "seed": seed, "base": spec["base"],
           "donor": spec["donor"], "cfg": cfg, "smoke": bool(smoke), "plan": arm_plan(pair, config, arm, seed, smoke),
           "pool_ranges": {"domain": [0, DOMAIN_COUNT], "probe": [PROBE_START, PROBE_START + PROBE_COUNT]},
           "heal_budget": dict(HEAL)}

    if arm == "base":
        model = host
        cache = {}
        curve = []
        base_view = host
        total_steps = 0
        row["trainable"] = 0
        row["heal_seconds"] = 0.0
    elif cfg["entry"] == "feedback":
        model, base_view, curve, cache, total_steps = run_feedback_arm(name, pair, cfg, arm, seed, host, tokenizer,
                                                                       donor, data, held, smoke, row)
    else:
        model, base_view, curve, cache, total_steps = run_synth_arm(name, pair, cfg, arm, seed, host, tokenizer,
                                                                    donor, data, held, smoke, row)

    # parameters
    from wsp import feedback as fb

    if isinstance(model, fb.FeedbackSeam):
        counts = fb.count_params(model)
        row["params_total"], row["params_per_token"] = counts["total"], counts["per_token"]
    elif hasattr(model, "wsp_grafts"):
        counts = synth.param_counts(model)
        row["params_total"], row["params_per_token"] = counts["total"], counts["per_token"]
    else:
        row["params_total"], row["params_per_token"] = ev.model_params(model)
    row["params_added"] = row["params_total"] - base_total
    row["base_params_total"] = base_total

    # the final probe, with the donor still loaded
    model.eval()
    final = probe_now(model, base_view, donor, tokenizer, held, cache, total_steps)
    log_point(name, final)
    row["probe"] = final
    row["probe_curve"] = curve + [final]
    row["transfer"] = final["transfer"]["code"]["transfer"]
    row["drift_general"] = final["drift"]["general"]["base_kl"]
    row["drift_math"] = final["drift"]["math"]["base_kl"]
    del donor
    gc.collect()
    torch.cuda.empty_cache()

    # scores, once
    row["scores"], row["scores_n"] = score_model(name, model, tokenizer)
    row["decode_tok_s"] = round(ev.decode_speed(model, tokenizer, new_tokens=64), 1)
    row["seconds"] = round(time.time() - start, 1)
    return row


@app.function(image=image, volumes={"/vol": volume}, gpu="H200", memory=262144, timeout=24 * 3600)
def cell(pair, config, arm, seed, smoke=False):
    row = run_cell(pair, config, arm, seed, smoke)
    os.makedirs(RESULT_DIR, exist_ok=True)
    with open(f"{RESULT_DIR}/{row['name']}.json", "w") as f:
        json.dump(row, f, default=str)
    volume.commit()
    return row


# ---------------------------------------------------------------- entry point


def read_rows(pattern):
    rows = []
    for path in sorted(glob.glob(pattern)):
        with open(path) as f:
            rows += json.load(f)
    return rows


def short_row(row):
    keep = ["name", "scores", "transfer", "drift_general", "drift_math", "params_total", "params_per_token",
            "params_added", "trainable", "heal_seconds"]
    return {k: row.get(k) for k in keep}


def print_estimate(cells):
    by_pair = estimate(cells)
    for pair in sorted(by_pair):
        print(f"estimate {pair}: {by_pair[pair]['cells']} cells, {by_pair[pair]['hours']:.1f} H200 hours")
    total = sum(v["hours"] for v in by_pair.values())
    print(f"estimate total: {len(cells)} cells, {total:.1f} H200 hours (unmeasured first guesses)")


@app.local_entrypoint()
def main(shard: int = 0, nshards: int = 1, pairs: str = "", factors: str = "", only: str = "", seeds: str = "",
         phase: str = "star", rows: str = "results/round20c_shard*.json", smoke: bool = False,
         estimate_only: bool = False, verdict: str = ""):
    from wsp import reader

    if verdict:
        found = read_rows(verdict)
        names = pairs.split(",") if pairs else list(PAIRS)
        for pair in names:
            print("verdict " + pair + " " + json.dumps(reader.read_summary(found, pair), indent=2, default=str))
        return
    pair_list = pairs.split(",") if pairs else None
    if smoke:
        todo = smoke_cells()
    elif phase == "best":
        todo = best_cells(read_rows(rows), pair_list)
    else:
        todo = star_cells(pair_list, factors.split(",") if factors else None, seeds.split(",") if seeds else None)
    todo = select_cells(todo, only.split(",") if only else None)
    if estimate_only:
        for c in todo:
            print(cell_name(*c))
        print_estimate(todo)
        return
    mine = todo if smoke else shard_of(todo, shard, nshards)
    print(f"round20c shard {shard}/{nshards}: {len(mine)} cells", flush=True)
    if not mine:
        raise SystemExit("round20c: no cells match these arguments; check --only names with --estimate-only")
    needed = []
    for pair, config, arm, seed in mine:
        if pair == "ref":
            continue
        for kind, model_id in (("domain", PAIRS[pair]["donor"]), ("general", PAIRS[pair]["base"])):
            if (kind, model_id) not in needed:
                needed.append((kind, model_id))
    for kind, model_id in needed:
        print(f"round20c prep {kind} {model_id}", flush=True)
        prep.remote(kind, model_id)
    args = [(pair, config, arm, seed, smoke) for pair, config, arm, seed in mine]
    out = []
    os.makedirs("results", exist_ok=True)
    out_path = "results/round20c_smoke.json" if smoke else f"results/round20c_shard{shard}.json"
    for index, result in enumerate(cell.starmap(args, return_exceptions=True)):
        label = cell_name(*mine[index])
        if isinstance(result, Exception):
            print(label, "cell failed:", repr(result)[:600], flush=True)
            continue
        out.append(result)
        print(label, json.dumps(short_row(result), default=str), flush=True)
        with open(out_path, "w") as f:
            json.dump(out, f, indent=2, default=str)
