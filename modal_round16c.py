"""Round 16c: where does a specialist's skill live, and can a router pick it up?

Run 16b put the MLPs of Qwen2.5-Math-1.5B-Instruct and
Qwen2.5-Coder-1.5B-Instruct into Qwen2.5-1.5B-Instruct as routed experts.
It did not beat the base: the router started with a base bias of 16 and
never left the base (92% of routing on GSM8K prompts went to the base MLP).
GSM8K: base 68.7, Math specialist 81.7, top-1 MoE 64.0 and 72.3. The LoRA
heal cost the base 11 GSM8K points, and the TIES baseline scored 0.3.

This round separates three questions with wsp/skill_moe.py:

  1. Does the skill live in the MLPs, or does it need attention too?
     Forced routing, no training. Every token in the forced layers goes to
     one expert.
  2. Can a router find the skill? Router started from a ridge regression on
     domain texts (the DUME idea), no base bias, no heal.
  3. Does a short router-only heal improve on the ridge start?

All arms use the Qwen2.5-1.5B-Instruct tokenizer and chat template, so every
arm sees identical prompts (the rows keep a prompt sample). Only the cell
math_own_template scores Math with its own template (another default system
prompt), to price that choice against run 16b.

Cells (rows record every score, macro, params total and per token, decode
tokens per second, and routing shares per skill):

  base, math, coder          references, each model alone
  math_own_template          Math with its own tokenizer and chat template
  force_math_mlp             Math MLPs everywhere, base attention, base
  force_coder_mlp            embeddings and head
  force_math_layer           whole Math decoder layers everywhere (attention,
  force_coder_layer          MLP, both norms, Math's own rope), base
                             embeddings, final norm, and head
  force_math_all             whole layers plus the specialist's own
  force_coder_all            embeddings, final norm, and head. Must equal
                             the specialist; the row holds the logit gap
                             to the real specialist (plumbing check)
  force_math_mlp_upper       the same as force_math_mlp and
  force_math_layer_upper     force_math_layer, upper half of the layers
                             only (lower half stays base)
  ridge_mlp_top1             ridge-initialized per-token router, top-1, MLP
  ridge_layer_top1           experts, no heal; whole-layer experts, no heal
  ridge_layer_seq            sequence-level routing over whole layers, no heal
  ridge_layer_top1_heal      the same with a 300-step router-only heal
  ridge_layer_seq_heal       (lr 5e-5, no LoRA), 2 seeds
  ties_fixed                 TIES with the true common ancestor Qwen2.5-1.5B,
                             density 0.5, embeddings kept from the base
  ties_fixed_inst            TIES of the two specialists' deltas against the
                             Instruct base (extra cell)

What each outcome means:
  - force_math_mlp near the base and force_math_layer near the Math
    specialist: the skill needs attention too, and experts must be whole
    layers.
  - force_math_mlp near the Math specialist on GSM8K: the skill is in the
    MLPs, and 16b failed because of the router and the heal, not the experts.
  - force_*_all must match math and coder to within rounding. If it does
    not, the plumbing (rope, KV cache, norms, head) is wrong, and no layer
    cell can be read.
  - force_*_layer vs force_*_all: the price of the base embeddings, final
    norm, and head (the specialist's representation of tokens).
  - force_*_upper vs the full versions: whether the lower half of the
    network is needed from the specialist.
  - ridge_layer_seq matches each specialist on its skill and the base on
    MMLU: per-sequence routing over whole layers is a working recipe (3x
    stored params, 1x params per token, plus the compute of every expert in
    this implementation).
  - ridge_*_top1 below ridge_layer_seq: per-token routing at layer level
    breaks the stream (a token's neighbours were routed to other layers, so
    their keys and values come from a mixed stream).
  - the heal rows above the no-heal rows: routers learn something the ridge
    fit does not know. Below: the heal damages the start, as the LoRA heal
    did in 16b.

No benchmark data trains anything here: every text that fits or trains comes from the wsp.healdata pools.

Ridge fit texts (from wsp.healdata pools, seed 0, pool range [0, 1000); every
pool is checked with healdata.check_sources, and rows matching a scoring item
by normalized text or its first 50 characters are dropped and counted): math
expert = pool "math"; coder expert = pool "code"; base = pools "general",
"prose", and "medicine" (a third each), so medical and multiple-choice style
prompts stay on the base. Token mode fits chat text with answers; sequence
mode fits the prompt side only (what the router sees at decision time).

Heal data: 300 steps, batch 8, seq 512, thirds of the "math" pool, the "code"
pool, and a base third made of "general", "prose", and "medicine" (pool range
[1000, 2000), disjoint from the ridge texts). Routers train alone (lr 5e-5,
Adam eps 1e-16), no load-balance loss. Rows record the pool names used.

Scores: GSM8K 300, MBPP 200 (seeded 200 of the 257 sanitized test items),
MMLU dev 1000, MedMCQA 1000, wsp.skills_eval. Two seeds where anything
trains (the seed changes the heal order).

TIES findings (no run was needed, from the model configs): the 16b arm used
Qwen2.5-1.5B as ancestor, which is right for the Instruct model, but Math-
and Coder-Instruct are not light fine-tunes of it. Math-1.5B and Coder-1.5B
continue pretraining from Qwen2.5-1.5B on huge math and code corpora, and
the Instruct versions then tune those. Their task vectors against
Qwen2.5-1.5B are large, not small and sparse. Math also changed
rope_theta from 1000000 to 10000 (and max positions to 4096), so its
attention weights adapted to another rotary basis than the one the merged
config uses (1000000). Trimming to the top 20% by magnitude then keeps
a disjoint mean of three unrelated large deltas, and the tied embedding was
merged too. ties_fixed keeps the embedding table from the base, uses
density 0.5, and records the relative size of each task vector and the
share of kept entries that lose the sign vote in the row (ties_diag), so
the next run shows the cause. Unverified until this run: that the cause
is the delta size and the rope change, not a code bug in the merge loop.

    python scripts/modal_fanout.py modal_round16c.py round16c --profiles a,b,c

Writes results/round16c_shard<i>.json, full per-item records to
/vol/results/round16c/<arm>_s<seed>.json, and MoE routers to
/vol/written/round16c_<arm>_s<seed>.
"""

import json
import os
import random
import re
import time

import modal

app = modal.App("wsp-round16c")
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

BASE = "Qwen/Qwen2.5-1.5B-Instruct"
PRETRAINED = "Qwen/Qwen2.5-1.5B"
CODER = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
MATH = "Qwen/Qwen2.5-Math-1.5B-Instruct"
EXPERT_NAMES = ["base", "math", "coder"]
SPECIALIST = {"math": MATH, "coder": CODER}
ALONE = {"base": BASE, "math": MATH, "coder": CODER, "math_own_template": MATH}
N_LAYERS = 28
UPPER = list(range(N_LAYERS // 2, N_LAYERS))

# arm -> (expert, granularity, layers, own_io)
FORCE = {
    "force_math_mlp": ("math", "mlp", None, False),
    "force_coder_mlp": ("coder", "mlp", None, False),
    "force_math_layer": ("math", "layer", None, False),
    "force_coder_layer": ("coder", "layer", None, False),
    "force_math_all": ("math", "layer", None, True),
    "force_coder_all": ("coder", "layer", None, True),
    "force_math_mlp_upper": ("math", "mlp", UPPER, False),
    "force_math_layer_upper": ("math", "layer", UPPER, False),
}
# arm -> (granularity, ridge mode, sequence routing, heal)
RIDGE = {
    "ridge_mlp_top1": ("mlp", "token", False, False),
    "ridge_layer_top1": ("layer", "token", False, False),
    "ridge_layer_seq": ("layer", "mean", True, False),
    "ridge_layer_top1_heal": ("layer", "token", False, True),
    "ridge_layer_seq_heal": ("layer", "mean", True, True),
}
TIES = ["ties_fixed", "ties_fixed_inst"]
HEALED = ["ridge_layer_top1_heal", "ridge_layer_seq_heal"]
SEEDS = [0, 1]

HEAL = {"steps": 300, "batch_size": 8, "seq_len": 512}
ROUTER_LR = 5e-5
RIDGE_LAM = 0.01
RIDGE_SCALE = 8.0
RIDGE_TEXTS = 300
SEQ_LAYER = 6
TIES_DENSITY = 0.5
TIES_SCALE = 1.0

SKILLS = ["gsm8k", "mbpp", "mmlu", "medmcqa"]
COUNTS = {"gsm8k": 300, "mbpp": 200, "mmlu": 1000, "medmcqa": 1000}
ROUTING_ITEMS = 32

# Which healdata pools feed each router class. Nothing here is a scored benchmark.
TRAIN_POOLS = {"math": ["math"], "coder": ["code"], "base": ["general", "prose", "medicine"]}
POOL_TOTAL = 2000
RIDGE_SKIP = 0
HEAL_SKIP = 1000


def cells():
    out = []
    for arm in ["base", "math", "coder", "math_own_template"]:
        out.append((arm, 0))
    for arm in FORCE:
        out.append((arm, 0))
    for arm in ["ridge_mlp_top1", "ridge_layer_top1", "ridge_layer_seq"]:
        out.append((arm, 0))
    for arm in HEALED:
        for seed in SEEDS:
            out.append((arm, seed))
    for arm in TIES:
        out.append((arm, 0))
    return out


# ---------------------------------------------------------------- eval


def skills_eval():
    """Return wsp.skills_eval. This round has no local copy: a broken import is a bug to fix first."""
    from wsp import skills_eval as module

    return module


def eval_items(ev):
    items = {
        "gsm8k": ev.gsm8k_items(count=COUNTS["gsm8k"], seed=0),
        "mbpp": ev.mbpp_items(),
        "mmlu": ev.mmlu_items(count=COUNTS["mmlu"]),
        "medmcqa": ev.medmcqa_items(count=COUNTS["medmcqa"]),
    }
    # MBPP: a seeded 200 of the 257 sanitized test items.
    order = list(range(len(items["mbpp"])))
    random.Random(0).shuffle(order)
    picked = []
    for index in sorted(order[:COUNTS["mbpp"]]):
        picked.append(items["mbpp"][index])
    items["mbpp"] = picked
    return items


def prompt_ids(tokenizer, ev, skill, item):
    """The token ids a skill feeds the model for one item (prompt only)."""
    if skill in ["mmlu", "medmcqa"]:
        return tokenizer(item["prompt"], return_tensors="pt").input_ids
    text = ev.render_prompt(tokenizer, ev.user_text(skill, item))
    return tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids


def routing_by_skill(model, tokenizer, ev, items):
    """Routing shares per expert per skill on the prompt tokens of ROUTING_ITEMS items (one item per forward)."""
    import torch

    from wsp.skill_moe import routing_shares, track_routing

    device = next(model.parameters()).device
    out = {}
    for skill in SKILLS:
        track_routing(model)
        for item in items[skill][:ROUTING_ITEMS]:
            with torch.no_grad():
                model(prompt_ids(tokenizer, ev, skill, item).to(device))
        shares = routing_shares(model)
        track_routing(model, on=False)
        mean = {}
        mean_prob = {}
        for index, name in enumerate(EXPERT_NAMES[:len(shares["mean"])]):
            mean[name] = round(shares["mean"][index], 4)
            mean_prob[name] = round(shares["mean_prob"][index], 4)
        per_layer = []
        for row in shares["per_layer"]:
            values = []
            for value in row:
                values.append(round(value, 4))
            per_layer.append(values)
        out[skill] = {"share": mean, "mean_prob": mean_prob, "per_layer": per_layer}
    return out


# ---------------------------------------------------------------- training and ridge data


def normalize(text):
    return re.sub(r"\s+", " ", str(text).strip().lower())


def scoring_keys(items):
    """Normalized texts and 50-character prefixes of every scoring item."""
    full = set()
    prefix = set()
    for skill in items:
        for item in items[skill]:
            for field in ["question", "text", "prompt"]:
                if field in item and item[field]:
                    text = normalize(item[field])
                    if skill in ["mmlu", "medmcqa"]:
                        text = text.replace("question: ", "", 1).split(" a. ")[0]
                    full.add(text)
                    prefix.add(text[:50])
    return full, prefix


def is_scoring_item(user, keys):
    text = normalize(user)
    full, prefix = keys
    return text in full or text[:50] in prefix


def source_ids():
    """Dataset ids of every pool used for training or fitting. Raises when one is a reported benchmark."""
    from wsp import healdata

    ids = []
    for names in TRAIN_POOLS.values():
        for name in names:
            for source in healdata.SOURCES[name]:
                ids.append(source[0])
    return healdata.check_sources(ids)


_POOL_CACHE = {}


def pool_pairs(name, start, count):
    """Pairs [start, start + count) of one healdata pool. One fetch per pool, so ranges never overlap."""
    from wsp import healdata

    if name not in _POOL_CACHE:
        _POOL_CACHE[name] = healdata.pairs(name, POOL_TOTAL, seed=0, skip=0)
    return _POOL_CACHE[name][start:start + count]


def class_pairs(klass, start, count):
    """Pairs for one router class, split evenly over its pools."""
    names = TRAIN_POOLS[klass]
    share = (count + len(names) - 1) // len(names)
    out = []
    for name in names:
        out += pool_pairs(name, start, share)
    return out


def filtered(pairs, keys, label, report):
    """Drop pairs whose user text matches a scoring item. Count what was dropped in report."""
    kept = []
    dropped = 0
    for user, assistant in pairs:
        if is_scoring_item(user, keys):
            dropped += 1
        else:
            kept.append((user, assistant))
    report[label] = {"kept": len(kept), "dropped": dropped}
    return kept


def heal_texts(tokenizer, keys, report):
    """Training texts for the router heal. Pools only, no benchmark data trains anything."""
    from wsp import healdata

    report["heal_sources"] = source_ids()
    report["heal_pools"] = TRAIN_POOLS
    total = HEAL["steps"] * HEAL["batch_size"]
    per_pool = (total + 2) // 3
    texts = []
    for klass, label in [("math", "heal_math"), ("coder", "heal_code"), ("base", "heal_base")]:
        pairs = class_pairs(klass, HEAL_SKIP, per_pool + 200)
        for user, assistant in filtered(pairs, keys, label, report)[:per_pool]:
            texts.append(healdata.render(tokenizer, user, assistant))
    random.Random(0).shuffle(texts)
    return texts


def ridge_texts(tokenizer, ev, keys, report, mode):
    """texts_by_expert for init_router_ridge. Pools only, no benchmark data trains anything.

    Token mode: chat text with answers. Mean mode: prompt side only.
    """
    from wsp import healdata

    report["ridge_sources"] = source_ids()
    report["ridge_pools"] = TRAIN_POOLS
    out = {"base": [], "math": [], "coder": []}
    for name, label in [("math", "ridge_math"), ("coder", "ridge_code"), ("base", "ridge_base")]:
        pairs = class_pairs(name, RIDGE_SKIP, RIDGE_TEXTS + 100)
        for user, assistant in filtered(pairs, keys, label, report)[:RIDGE_TEXTS]:
            if mode == "token":
                out[name].append(healdata.render(tokenizer, user, assistant))
            else:
                out[name].append(ev.render_prompt(tokenizer, user))
    return out


def train_router(model, tokenizer, texts, seed):
    """Train the routers alone on texts (one pass in a seeded order). Return a loss log."""
    import torch

    from wsp.skill_moe import ROUTER_ADAM_EPS, router_parameters

    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    device = next(model.parameters()).device
    params = router_parameters(model)
    optimizer = torch.optim.AdamW([{"params": params, "lr": ROUTER_LR, "eps": ROUTER_ADAM_EPS}], weight_decay=0.0)
    order = list(range(len(texts)))
    random.Random(seed).shuffle(order)
    losses = []
    model.train()
    cursor = 0
    for step in range(HEAL["steps"]):
        batch = []
        while len(batch) < HEAL["batch_size"]:
            if cursor >= len(order):
                cursor = 0
            batch.append(texts[order[cursor]])
            cursor += 1
        encoded = tokenizer(batch, truncation=True, max_length=HEAL["seq_len"], padding=True, return_tensors="pt",
                            add_special_tokens=False)
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels, use_cache=False)
        optimizer.zero_grad()
        out.loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        optimizer.step()
        losses.append(float(out.loss.detach()))
        if (step + 1) % 50 == 0:
            print(f"step {step + 1}/{HEAL['steps']} loss {losses[-1]:.4f}")
    model.eval()
    return {"loss_first20": sum(losses[:20]) / min(20, len(losses)),
            "loss_last20": sum(losses[-20:]) / min(20, len(losses))}


# ---------------------------------------------------------------- TIES


def trim_top(tensor, keep):
    """Keep the top keep fraction of entries by magnitude; zero the rest."""
    import torch

    count = tensor.numel()
    k = int(count * keep)
    if k <= 0:
        return torch.zeros_like(tensor)
    if k >= count:
        return tensor
    threshold = torch.kthvalue(tensor.abs().reshape(-1), count - k + 1).values
    return tensor * (tensor.abs() >= threshold).to(tensor.dtype)


def ties_merge_into(target, anchor, finetuned, density, scale):
    """TIES (Yadav et al. 2023): write anchor + merged task vectors into target, tensor by tensor.

    Task vector = finetuned - anchor. Per tensor: trim to the top `density`
    by magnitude, elect the sign of the summed trimmed vectors, take the
    disjoint mean over the entries that agree. Embedding and head tensors
    are left as the target has them (a merged vocabulary table would mix
    three models' token spaces). Returns a diagnostics dict: per model the
    relative size of its task vector (norm of delta over norm of anchor)
    and the share of kept entries that lost the sign vote.
    """
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    pre = dict(anchor.named_parameters())
    others = []
    for model in finetuned:
        others.append(dict(model.named_parameters()))
    delta_sq = [0.0] * len(others)
    anchor_sq = 0.0
    kept = [0] * len(others)
    lost = [0] * len(others)
    skipped = []
    with torch.no_grad():
        for name, param in target.named_parameters():
            if "embed_tokens" in name or "lm_head" in name:
                skipped.append(name)
                continue
            ok = name in pre and pre[name].shape == param.shape
            for state in others:
                if name not in state or state[name].shape != param.shape:
                    ok = False
            if not ok:
                skipped.append(name)
                continue
            base = pre[name].to(device, torch.float32)
            anchor_sq += float((base ** 2).sum())
            deltas = []
            for index, state in enumerate(others):
                delta = state[name].to(device, torch.float32) - base
                delta_sq[index] += float((delta ** 2).sum())
                deltas.append(trim_top(delta, density))
            stacked = torch.stack(deltas)
            sign = torch.sign(stacked.sum(dim=0))
            nonzero = stacked != 0
            agree = (torch.sign(stacked) == sign) & nonzero
            for index in range(len(others)):
                kept[index] += int(nonzero[index].sum())
                lost[index] += int((nonzero[index] & ~agree[index]).sum())
            total = (stacked * agree.to(stacked.dtype)).sum(dim=0)
            count = agree.sum(dim=0).clamp(min=1).to(stacked.dtype)
            param.copy_((base + scale * total / count).to(param.device, param.dtype))
    diag = {"skipped": skipped, "density": density, "scale": scale, "models": []}
    for index in range(len(others)):
        diag["models"].append({"relative_delta_norm": (delta_sq[index] / max(anchor_sq, 1e-12)) ** 0.5,
                               "sign_lost_share": lost[index] / max(kept[index], 1)})
    return diag


# ---------------------------------------------------------------- cells


def load_model(name):
    import torch
    import transformers

    return transformers.AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16)


def plain_params(model):
    seen = set()
    total = 0
    for param in model.parameters():
        if id(param) not in seen:
            seen.add(id(param))
            total += param.numel()
    return total


def equality_check(model, tokenizer, ev, items, specialist_name):
    """Logit gap between the forced network and the real specialist on four GSM8K prompts."""
    import torch

    reference = load_model(SPECIALIST[specialist_name]).to("cuda").eval()
    worst = 0.0
    agree = 0
    total = 0
    for item in items["gsm8k"][:4]:
        ids = prompt_ids(tokenizer, ev, "gsm8k", item).to("cuda")
        with torch.no_grad():
            got = model(ids).logits.float()
            want = reference(ids).logits.float()
        worst = max(worst, float((got - want).abs().max()))
        agree += int((got.argmax(-1) == want.argmax(-1)).sum())
        total += int(ids.numel())
    del reference
    return {"max_abs_logit_gap": worst, "argmax_agree": agree / max(total, 1), "tokens": total}


@app.function(image=image, volumes={"/vol": volume}, gpu="H100", memory=131072, timeout=6 * 3600)
def run_cell(arm, seed):
    import torch
    import transformers

    from wsp.skill_moe import (build_skill_moe, clear_force, count_params, force_route, init_router_ridge,
                               save_skill_moe, set_sequence_routing)

    start = time.time()
    torch.manual_seed(seed)
    random.seed(seed)
    ev = skills_eval()
    row = {"arm": arm, "seed": seed, "base": BASE, "experts": [MATH, CODER], "eval": "wsp.skills_eval"}
    tokenizer_name = BASE
    if arm == "math_own_template":
        tokenizer_name = MATH
    tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_name)
    items = eval_items(ev)
    moe = False

    if arm in ALONE:
        model = load_model(ALONE[arm]).to("cuda").eval()
    elif arm in TIES:
        model = load_model(BASE)
        if arm == "ties_fixed":
            anchor = load_model(PRETRAINED)
            finetuned = [load_model(BASE), load_model(CODER), load_model(MATH)]
        else:
            anchor = load_model(BASE)
            finetuned = [load_model(CODER), load_model(MATH)]
        row["ties_diag"] = ties_merge_into(model, anchor, finetuned, TIES_DENSITY, TIES_SCALE)
        del anchor, finetuned
        model = model.to("cuda").eval()
    elif arm in FORCE:
        moe = True
        expert, granularity, layers, own_io = FORCE[arm]
        model = build_skill_moe(load_model(BASE), [load_model(SPECIALIST[expert])], top_k=1,
                                whole_layer=(granularity == "layer"), keep_io=own_io, base_id=BASE,
                                expert_ids=[SPECIALIST[expert]], expert_names=[expert])
        model = model.to("cuda").eval()
        force_route(model, expert, layers=layers, own_io=own_io)
        row["forced"] = {"expert": expert, "granularity": granularity, "layers": layers, "own_io": own_io}
        if own_io:
            row["equality"] = equality_check(model, tokenizer, ev, items, expert)
            print("equality", row["equality"])
    elif arm in RIDGE:
        moe = True
        granularity, mode, sequence, heal = RIDGE[arm]
        model = build_skill_moe(load_model(BASE), [load_model(MATH), load_model(CODER)], top_k=1,
                                router_init="uniform", whole_layer=(granularity == "layer"), base_id=BASE,
                                expert_ids=[MATH, CODER], expert_names=["math", "coder"])
        model = model.to("cuda").eval()
        data_report = {}
        keys = scoring_keys(items)
        texts = ridge_texts(tokenizer, ev, keys, data_report, mode)
        row["ridge"] = init_router_ridge(model, tokenizer, texts, lam=RIDGE_LAM, mode=mode, scale=RIDGE_SCALE,
                                         base_margin=0.0)
        if sequence:
            set_sequence_routing(model, SEQ_LAYER)
            row["seq_layer"] = SEQ_LAYER
        row["routing_before_heal"] = routing_by_skill(model, tokenizer, ev, items)
        if heal:
            texts = heal_texts(tokenizer, keys, data_report)
            row["heal"] = train_router(model, tokenizer, texts, seed)
        row["data_report"] = data_report
        out_dir = f"/vol/written/round16c_{arm}_s{seed}"
        save_skill_moe(model, out_dir, tokenizer)
        volume.commit()
        row["saved"] = out_dir
    else:
        raise ValueError(f"unknown arm {arm}")

    if moe:
        counts = count_params(model)
        row["params_total"] = counts["total"]
        row["params_active"] = counts["active"]
        row["params_router"] = counts["router"]
    else:
        row["params_total"] = plain_params(model)
        row["params_active"] = row["params_total"]

    # The rendered GSM8K prompt, so a difference in chat templates between arms is visible.
    first = items["gsm8k"][0]
    row["gsm8k_prompt_sample"] = ev.render_prompt(tokenizer, ev.user_text("gsm8k", first))
    if moe:
        row["routing"] = routing_by_skill(model, tokenizer, ev, items)
    result = ev.evaluate_suite(model, tokenizer, skills=SKILLS, items=items)
    row["scores"] = dict(result["scores"])
    row["macro"] = result["suite"]
    row["decode_tok_s"] = round(ev.decode_speed(model, tokenizer), 1)
    row["seconds"] = round(time.time() - start, 1)

    os.makedirs("/vol/results/round16c", exist_ok=True)
    full = dict(row)
    full["skills"] = result["skills"]
    with open(f"/vol/results/round16c/{arm}_s{seed}.json", "w") as f:
        json.dump(full, f, default=str)
    volume.commit()
    return row


@app.local_entrypoint()
def main(shard: int = 0, nshards: int = 1):
    mine = []
    for index, cell in enumerate(cells()):
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
                                                  "decode_tok_s"]}))
    os.makedirs("results", exist_ok=True)
    with open(f"results/round16c_shard{shard}.json", "w") as f:
        json.dump(rows, f, indent=2)
