"""Round 18: same-family expert transplant (wsp/transplant.py).

No benchmark data trains anything. Every text that selects experts, fits a
stitch, trains a router or gate, or heals a LoRA comes from the wsp.healdata
pools (math, code, medicine, general), checked with healdata.check_sources.
Benchmarks only score the finished model, through wsp.skills_eval. The
benchmarks are chosen per user request in the product, so selection and heal
see generic skill-domain text only. The pools are one fetch per pool, and the
ranges never overlap:

  selection       pool positions [0, 96)       math, code, and medicine (the
                                                knowledge skill), general as contrast
  stitch fit      [1000, 1064) of each pool    256 texts
  heal            [2000, 5000) of each pool    12000 texts, 3000 per pool
  held-out KL     [5000, 5128) of each pool    the benchmark-free quality signal

Base Qwen/Qwen3-1.7B (hidden 2048, 28 layers). Source Qwen/Qwen3-30B-A3B
(hidden 2048, 48 layers, 128 experts, top 8). Same width, same tokenizer.
Next to the MLP of every base layer sits a bank of K experts taken from the
source layer at the same relative depth, read through a stitch pair and a
router that starts from the source router rows. The model stays near base
size and runs alone. The source runs only during the build (selection,
stitch fit), as the donor in the heal, and as the reference of the KL
measurements.

Heal target (owner rule 3): the donor's own behaviour. The loss is
KL(donor || student) on pool text, where the donor reads the same tokens
(teacher forcing), so a pool answer reaches the student only through the KL
term. There is no LM loss on pool answers in the distillation arms. A
preservation term adds 0.5 * KL(frozen base || student) on the general pool
texts of each batch. The two LM-only arms (lora, transplant_k4_lm) keep an
LM loss, on pool text, and have no donor and no preservation term.

Cells (2 seeds for the trained arms; the seed changes the heal order, the
LoRA init, and the random control draw):
  base                       Qwen3-1.7B alone, scored; also the donor KL of the base
  source                     Qwen3-30B-A3B, one cell, for reference
  transplant_k4_noheal       the gate: no training. Real K=4 experts, ridge stitches,
                             gate opened to a fixed small value (0.05, 0.1, 0.25;
                             headline 0.1). Reports donor KL on held-out pool text.
                             No benchmark scoring. Seed 0 only.
  transplant_k4_random_noheal  the same with K=4 random experts per layer
  lora                       LoRA, LM loss on pool text only; rank matched to kd_lora
  kd_lora                    LoRA, KL to the donor plus the preservation term, same
                             texts and steps as the transplant arms. The rank is
                             chosen so its trainable count is close to the count of
                             transplant_k4 (wsp.transplant.matched_lora_rank).
  transplant_k4              K=4 per layer (math, code, knowledge take turns),
                             KL + preservation heal, LoRA r16 on the base
  transplant_k4_random       the same build with K=4 random experts per layer
  transplant_k8              K=8
  transplant_k4_lm           K=4, LM loss only on pool text (no donor)

Every trained arm reports, after its heal, donor_kl (mean KL from the donor
logits to the student logits on held-out pool text, lower is better) and
base_kl_general (KL from the frozen base to the student on held-out general
pool text). These two numbers need no benchmark. Spend order: run the two
noheal cells first (`--stage gate`), read the donor KL, then pay for heals.

Trainable in the transplant arms: stitches, routers, gates, LoRA r16 on the
base. The stitches start from a ridge fit on paired states of the same
text (STITCH_RANK keeps the fit as a rank r product, so the parameter
counts stay near the target; rows hold total and active counts).

Heal: 1500 steps, batch 8, 384 tokens, one pass over 12000 pool texts. The
learning rates are first guesses and untuned: stitches 1e-4, routers 5e-4,
gates 5e-3, LoRA 2e-4.

Scores (wsp.skills_eval, every arm with the Qwen3 chat template and
thinking off): GSM8K 500 seeded test items, MATH-500, HumanEval, MBPP
sanitized test (257), MMLU held-out (2000 test items from positions 1000 to
3000 of the seed-0 order; log-prob of the answer letter after the raw
multiple-choice prompt, as skills_eval scores MMLU). Rows hold every score,
total and active params, trainable params, heal seconds, routing shares per
skill and per expert label (prompt tokens of 32 items per skill), the
fitted stitch quality, the gate values, the pool names and ranges, and the
two KL numbers.

Go and stop rule: transplant_k4 must beat kd_lora and transplant_k4_random
by more than seed noise on at least two skills. Seed noise is the largest of
the seed spread of each arm in the comparison and one binomial standard
error (wsp.transplant.go_stop). Otherwise stop.

    python scripts/modal_fanout.py modal_round18.py round18 --profiles a,b,c
    modal run modal_round18.py --stage gate    (base and the two noheal cells only)
    modal run modal_round18.py --verdict 'results/round18_shard*.json'   (reads the rows, runs no cell)

Writes results/round18_shard<i>.json, full per-item records to
/vol/results/round18/<arm>_s<seed>.json, saved transplants to
/vol/written/round18_<arm>_s<seed>, and caches the selection and the
stitch fit in /vol/transplant (file names end in _hd, so no cache built from
older text is read).
"""

import glob
import json
import os
import random
import time

import modal

app = modal.App("wsp-round18")
volume = modal.Volume.from_name("wsp-models", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch==2.7.1", "transformers==4.57.1", "accelerate", "huggingface_hub", "requests",
        "safetensors", "zstandard", "numpy", "datasets", "sentencepiece", "protobuf",
    )
    .env({"HF_HOME": "/vol/hf"})
    .add_local_python_source("wsp")
)

BASE = "Qwen/Qwen3-1.7B"
SOURCE = "Qwen/Qwen3-30B-A3B"
PAIR = "qwen3"

SKILLS = ["gsm8k", "math500", "humaneval", "mbpp", "mmlu"]
COUNTS = {"gsm8k": 500, "math500": 500, "humaneval": 164, "mbpp": 257, "mmlu": 2000}
TRAINED = ["lora", "kd_lora", "transplant_k4", "transplant_k4_random", "transplant_k8", "transplant_k4_lm"]
NOHEAL = ["transplant_k4_noheal", "transplant_k4_random_noheal"]
SEEDS = [0, 1]
K_OF = {"transplant_k4": 4, "transplant_k4_random": 4, "transplant_k8": 8, "transplant_k4_lm": 4,
        "transplant_k4_noheal": 4, "transplant_k4_random_noheal": 4}
DISTILL = ["kd_lora", "transplant_k4", "transplant_k4_random", "transplant_k8"]
NOHEAL_GATES = [0.0, 0.05, 0.1, 0.25]
NOHEAL_HEADLINE = 0.1
ANCHOR_WEIGHT = 0.5

STEPS = 1500
BATCH = 8
SEQ_LEN = 384
LORA_RANK = 16
STITCH_RANK = 256
RIDGE = 1e-3
TOP_K = 2
LR = {"stitch": 1e-4, "router": 5e-4, "gate": 5e-3, "lora": 2e-4}
SELECT_TEXTS = 96
FIT_PER_POOL = 64
HEAL_PER_POOL = 3000
KL_PER_POOL = 128
SELECT_START = 0
FIT_START = 1000
HEAL_START = 2000
KL_START = 5000
POOL_TOTAL = 5200
HEAL_POOLS = ["math", "code", "medicine", "general"]
SKILL_POOL = {"math": "math", "code": "code", "knowledge": "medicine"}
CACHE_TAG = "hd"
ROUTING_ITEMS = 32
MIN_FREQ = 0.02



def cells():
    out = [("base", 0)]
    for arm in NOHEAL:
        out.append((arm, 0))
    out.append(("source", 0))
    for arm in TRAINED:
        for seed in SEEDS:
            out.append((arm, seed))
    return out


# ---------------------------------------------------------------- tokenizer and items


def thinking_off(tokenizer):
    """Make every apply_chat_template call use enable_thinking=False (Qwen3 template)."""
    original = tokenizer.apply_chat_template

    def patched(*args, **kwargs):
        kwargs.setdefault("enable_thinking", False)
        return original(*args, **kwargs)

    tokenizer.apply_chat_template = patched
    return tokenizer


def mmlu_heldout_items():
    """2000 MMLU test items from positions 1000 to 3000 of the seed-0 order (the round 7 held-out set).

    Scoring only. wsp.skills_eval loads the items; this script never reads MMLU itself.
    """
    items = []
    for item in mmlu_all(3000)[1000:3000]:
        item = dict(item)
        item["id"] = item["id"].replace("mmlu_dev", "mmlu_held")
        items.append(item)
    return items


def mmlu_all(count):
    from wsp import skills_eval as ev

    return ev.mmlu_items(count=count)


def eval_items(ev):
    return {"gsm8k": ev.gsm8k_items(count=COUNTS["gsm8k"], seed=0), "math500": ev.math500_items(),
            "humaneval": ev.humaneval_items(), "mbpp": ev.mbpp_items(), "mmlu": mmlu_heldout_items()}


def prompt_ids(tokenizer, ev, skill, item):
    if skill == "mmlu":
        return tokenizer(item["prompt"], return_tensors="pt").input_ids
    text = ev.render_prompt(tokenizer, ev.user_text(skill, item))
    return tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids


def routing_by_skill(model, tokenizer, ev, items):
    """Routing share per expert label per skill, on the prompt tokens of ROUTING_ITEMS items."""
    import torch

    from wsp.transplant import routing_shares, track_routing

    device = next(model.parameters()).device
    out = {}
    for skill in SKILLS:
        track_routing(model)
        for item in items[skill][:ROUTING_ITEMS]:
            with torch.no_grad():
                model(prompt_ids(tokenizer, ev, skill, item).to(device))
        shares = routing_shares(model)
        track_routing(model, on=False)
        by_label = {}
        for label in shares["by_label"]:
            by_label[label] = round(shares["by_label"][label], 4)
        out[skill] = by_label
    return out


# ---------------------------------------------------------------- pool text (wsp.healdata only)

_POOL_CACHE = {}


def source_ids():
    """Dataset ids of every pool this script reads. Raises when one is a reported benchmark."""
    from wsp import healdata

    ids = []
    for name in HEAL_POOLS:
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


def pool_texts(tokenizer, name, start, count):
    from wsp import healdata

    out = []
    for user, assistant in pool_pairs(name, start, count):
        out.append(healdata.render(tokenizer, user, assistant))
    return out


def selection_texts(tokenizer):
    """(skill texts, general texts) for expert selection. Pools only."""
    skills = {}
    for skill in SKILL_POOL:
        skills[skill] = pool_texts(tokenizer, SKILL_POOL[skill], SELECT_START, SELECT_TEXTS)
    return skills, pool_texts(tokenizer, "general", SELECT_START, SELECT_TEXTS)


def fit_texts(tokenizer):
    """Texts for the stitch fit: FIT_PER_POOL from each heal pool, a range disjoint from selection and heal."""
    texts = []
    for name in HEAL_POOLS:
        texts += pool_texts(tokenizer, name, FIT_START, FIT_PER_POOL)
    random.Random(0).shuffle(texts)
    return texts


def heal_data(tokenizer):
    """(texts, flags) for the heal. flags[i] is True for a general text (the preservation term reads those)."""
    rows = []
    for name in HEAL_POOLS:
        for text in pool_texts(tokenizer, name, HEAL_START, HEAL_PER_POOL):
            rows.append((text, name == "general"))
    random.Random(0).shuffle(rows)
    return [row[0] for row in rows], [row[1] for row in rows]


def kl_texts(tokenizer):
    """Held-out pool text per pool for the KL measurements. No overlap with selection, fit, or heal."""
    return {name: pool_texts(tokenizer, name, KL_START, KL_PER_POOL) for name in HEAL_POOLS}


def pool_record():
    return {"pools": HEAL_POOLS, "skill_pool": SKILL_POOL, "sources": source_ids(),
            "ranges": {"select": [SELECT_START, SELECT_START + SELECT_TEXTS],
                       "fit": [FIT_START, FIT_START + FIT_PER_POOL],
                       "heal": [HEAL_START, HEAL_START + HEAL_PER_POOL],
                       "kl": [KL_START, KL_START + KL_PER_POOL]}}


def kl_report(model, tokenizer, donor, anchor, held):
    """The two benchmark-free numbers: donor KL on held-out pool text, KL to the frozen base on general text."""
    from wsp.transplant import measure_kl

    by_pool = {}
    total = 0.0
    tokens = 0
    for name in held:
        out = measure_kl(model, donor, tokenizer, held[name], batch_size=4, seq_len=SEQ_LEN)
        by_pool[name] = round(out["kl"], 5)
        total += out["kl"] * out["tokens"]
        tokens += out["tokens"]
    base = measure_kl(model, anchor, tokenizer, held["general"], batch_size=4, seq_len=SEQ_LEN)
    return {"donor_kl": round(total / max(1, tokens), 5), "donor_kl_by_pool": by_pool,
            "base_kl_general": round(base["kl"], 5)}


# ---------------------------------------------------------------- cells


def load_model(name):
    import torch
    import transformers

    return transformers.AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16)


def trainable_count(model):
    total = 0
    for param in model.parameters():
        if param.requires_grad:
            total += param.numel()
    return total


def selection_for(source, tokenizer, layer_map, arm, seed):
    """The expert plan of a transplant arm. The real selection is cached per K."""
    from wsp.transplant import select_experts

    k = K_OF[arm]
    if "random" in arm:
        return select_experts(source, tokenizer, {}, [], layer_map, k, control="random", seed=seed)
    path = f"/vol/transplant/selection_{PAIR}_k{k}_{CACHE_TAG}.json"
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    skills, general = selection_texts(tokenizer)
    plan = select_experts(source, tokenizer, skills, general, layer_map, k, min_freq=MIN_FREQ)
    os.makedirs("/vol/transplant", exist_ok=True)
    with open(path, "w") as f:
        json.dump(plan, f)
    volume.commit()
    return plan


def stitches_for(base, source, tokenizer, layer_map):
    """The ridge stitch fit. It depends on the layer map and the text only, so it is cached."""
    import torch

    from wsp.transplant import fit_stitches

    path = f"/vol/transplant/fit_{PAIR}_r{STITCH_RANK}_{CACHE_TAG}.pt"
    if os.path.exists(path):
        return torch.load(path)
    fits = fit_stitches(base, source, tokenizer, fit_texts(tokenizer), layer_map, rank=STITCH_RANK, ridge=RIDGE,
                        max_length=256, batch_size=4)
    os.makedirs("/vol/transplant", exist_ok=True)
    torch.save(fits, path)
    volume.commit()
    return fits


def param_groups(model):
    from wsp.transplant import set_trainable

    groups = set_trainable(model)
    out = []
    for name in ["stitch", "router", "gate", "lora"]:
        if len(groups[name]) > 0:
            out.append({"params": groups[name], "lr": LR[name]})
    return out


@app.function(image=image, volumes={"/vol": volume}, gpu="H200", memory=262144, timeout=10 * 3600)
def run_cell(arm, seed):
    import torch
    import transformers

    from wsp import skills_eval as ev
    from wsp.lora import apply_lora
    from wsp.transplant import (banks, build_transplant, count_params, heal, make_layer_map, matched_lora_rank,
                                parts_from_model, planned_trainable, save_transplant, set_gates,
                                source_active_params, source_moe_layers)

    start = time.time()
    torch.manual_seed(seed)
    tokenizer = thinking_off(transformers.AutoTokenizer.from_pretrained(BASE))
    row = {"arm": arm, "seed": seed, "base": BASE, "source": SOURCE, "k": K_OF.get(arm)}
    noheal = arm.endswith("_noheal")
    kd = arm in DISTILL
    model = None
    batch_size = 32
    trained = arm in TRAINED
    donor = None
    anchor = None
    d_source = transformers.AutoConfig.from_pretrained(SOURCE).hidden_size
    if arm != "source":
        row.update(pool_record())

    if arm == "base":
        model = load_model(BASE).to("cuda").eval()
        donor = load_model(SOURCE).to("cuda").eval()
        anchor = model
        row["kl"] = kl_report(model, tokenizer, donor, anchor, kl_texts(tokenizer))
    elif arm == "source":
        model = load_model(SOURCE).to("cuda").eval()
        batch_size = 64
    elif arm in ("lora", "kd_lora"):
        model = load_model(BASE).to("cuda")
        for param in model.parameters():
            param.requires_grad_(False)
        rank = matched_lora_rank(model, d_source, K_OF["transplant_k4"], STITCH_RANK, LORA_RANK)
        row["lora_rank"] = rank
        row["match_target"] = planned_trainable(model, d_source, K_OF["transplant_k4"], STITCH_RANK, LORA_RANK)
        lora = apply_lora(model, rank)
        texts, flags = heal_data(tokenizer)
        donor = load_model(SOURCE).to("cuda").eval()
        anchor = load_model(BASE).to("cuda").eval()
        heal_start = time.time()
        row["heal"] = heal(model, tokenizer, texts, [{"params": lora, "lr": LR["lora"]}], STEPS, BATCH, SEQ_LEN, seed,
                           teacher=donor if kd else None, kd_weight=1.0 if kd else 0.0, lm_weight=0.0 if kd else 1.0,
                           autocast=torch.bfloat16, anchor=anchor if kd else None,
                           anchor_weight=ANCHOR_WEIGHT if kd else 0.0, anchor_flags=flags)
        row["heal_seconds"] = round(time.time() - heal_start, 1)
    else:
        donor = load_model(SOURCE).to("cuda").eval()
        base = load_model(BASE).to("cuda")
        anchor = load_model(BASE).to("cuda").eval()
        layer_map = make_layer_map(len(base.model.layers), len(donor.model.layers), source_moe_layers(donor.config))
        plan = selection_for(donor, tokenizer, layer_map, arm, seed)
        fits = stitches_for(base, donor, tokenizer, layer_map)
        parts = parts_from_model(donor, plan)
        model = build_transplant(base, plan, parts, stitches=fits, lora_rank=None if noheal else LORA_RANK,
                                 top_k=TOP_K, stitch_rank=STITCH_RANK, base_id=BASE, source_id=SOURCE, pair=PAIR)
        row["plan"] = {"layer_map": plan["layer_map"], "experts": plan["experts"], "labels": plan["labels"],
                       "control": plan.get("control")}
        row["fit_r2_in"] = round(sum(fits[i]["r2_in"] for i in fits) / len(fits), 4)
        row["fit_r2_out"] = round(sum(fits[i]["r2_out"] for i in fits) / len(fits), 4)
        if noheal:
            held = kl_texts(tokenizer)
            model.eval()
            row["noheal"] = {}
            for gate in NOHEAL_GATES:
                set_gates(model, gate)
                row["noheal"][str(gate)] = kl_report(model, tokenizer, donor, anchor, held)
            row["donor_kl"] = row["noheal"][str(NOHEAL_HEADLINE)]["donor_kl"]
            row["donor_kl_gate0"] = row["noheal"]["0.0"]["donor_kl"]
            row["gate_headline"] = NOHEAL_HEADLINE
            row["seconds"] = round(time.time() - start, 1)
            os.makedirs("/vol/results/round18", exist_ok=True)
            with open(f"/vol/results/round18/{arm}_s{seed}.json", "w") as f:
                json.dump(row, f, default=str)
            volume.commit()
            return row
        texts, flags = heal_data(tokenizer)
        heal_start = time.time()
        row["heal"] = heal(model, tokenizer, texts, param_groups(model), STEPS, BATCH, SEQ_LEN, seed,
                           teacher=donor if kd else None, kd_weight=1.0 if kd else 0.0, lm_weight=0.0 if kd else 1.0,
                           autocast=torch.bfloat16, anchor=anchor if kd else None,
                           anchor_weight=ANCHOR_WEIGHT if kd else 0.0, anchor_flags=flags)
        row["heal_seconds"] = round(time.time() - heal_start, 1)
        gates = []
        for bank in banks(model):
            gates.append(round(float(bank.gate.detach()), 4))
        row["gates"] = gates
        out_dir = f"/vol/written/round18_{arm}_s{seed}"
        save_transplant(model, out_dir, tokenizer)
        volume.commit()
        row["saved"] = out_dir

    if trained:
        model.eval()
        row["kl"] = kl_report(model, tokenizer, donor, anchor, kl_texts(tokenizer))
        row["donor_kl"] = row["kl"]["donor_kl"]
        row["base_kl_general"] = row["kl"]["base_kl_general"]
    if arm == "base":
        row["donor_kl"] = row["kl"]["donor_kl"]
        row["base_kl_general"] = 0.0
    del donor, anchor
    torch.cuda.empty_cache()

    if arm == "source":
        total, active = source_active_params(model)
        row["params_total"] = total
        row["params_active"] = active
    elif arm.startswith("transplant"):
        counts = count_params(model)
        row["params_total"] = counts["total"]
        row["params_active"] = counts["active"]
        row["total_ratio"] = round(counts["total_ratio"], 4)
        row["active_ratio"] = round(counts["active_ratio"], 4)
        row["params_stitch"] = counts["stitch"]
        row["params_router"] = counts["router"]
    else:
        seen = set()
        total = 0
        for param in model.parameters():
            if id(param) not in seen:
                seen.add(id(param))
                total += param.numel()
        row["params_total"] = total
        row["params_active"] = ev.model_params(model)[1]
    row["params_trainable"] = trainable_count(model)

    items = eval_items(ev)
    row["gsm8k_prompt_sample"] = ev.render_prompt(tokenizer, ev.user_text("gsm8k", items["gsm8k"][0]))
    if arm.startswith("transplant"):
        row["routing"] = routing_by_skill(model, tokenizer, ev, items)
    model.eval()
    result = ev.evaluate_suite(model, tokenizer, skills=SKILLS, items=items, batch_size=batch_size)
    row["scores"] = dict(result["scores"])
    row["macro"] = result["suite"]
    row["seconds"] = round(time.time() - start, 1)

    os.makedirs("/vol/results/round18", exist_ok=True)
    full = dict(row)
    full["skills"] = ev.strip_outputs(result)["skills"]
    with open(f"/vol/results/round18/{arm}_s{seed}.json", "w") as f:
        json.dump(full, f, default=str)
    volume.commit()
    return row


def print_verdict(path):
    from wsp.transplant import go_stop

    rows = []
    for name in sorted(glob.glob(path)):
        with open(name) as f:
            rows += json.load(f)
    if len(rows) == 0:
        raise SystemExit(f"no rows in {path}")
    result = go_stop(rows, SKILLS, COUNTS)
    print(json.dumps(result, indent=2))


@app.local_entrypoint()
def main(shard: int = 0, nshards: int = 1, verdict: str = "", stage: str = "all", only: str = ""):
    if verdict:
        print_verdict(verdict)
        return
    mine = []
    chosen = cells()
    if stage == "gate":
        chosen = [cell for cell in chosen if cell[0] == "base" or cell[0] in NOHEAL]
    elif stage != "all":
        raise SystemExit("stage must be all or gate")
    if only:
        # "arm:seed,arm:seed" runs chosen cells.
        wanted = set(only.split(","))
        chosen = [cell for cell in chosen if f"{cell[0]}:{cell[1]}" in wanted]
    for index, cell in enumerate(chosen):
        if index % nshards == shard:
            mine.append(cell)
    print(f"shard {shard}/{nshards}: {mine}")
    rows = []
    for row in run_cell.starmap(mine, return_exceptions=True):
        if isinstance(row, Exception):
            print("cell failed:", repr(row)[:600])
            continue
        rows.append(row)
        print(json.dumps({k: row.get(k) for k in ["arm", "seed", "scores", "macro", "params_total", "params_active",
                                                  "params_trainable", "heal_seconds", "donor_kl", "base_kl_general"]}))
    os.makedirs("results", exist_ok=True)
    with open(f"results/round18_shard{shard}.json", "w") as f:
        json.dump(rows, f, indent=2)
