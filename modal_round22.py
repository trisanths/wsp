"""Round 22: a feedback seam. A donor block writes into the NEXT token's input, not into the host mid-stack.

No benchmark data trains anything. Every text that fits a stitch (T_in), trains
T_out, a gate, or a LoRA, or serves as a distillation or probe text, comes from
the wsp.healdata pools (math, code, medicine, general), checked with
healdata.check_sources. Benchmarks only score the finished model, through
wsp.skills_eval. The three ranges of each pool never overlap:

  stitch (T_in)   pool positions [1000, 1300) of each pool   1200 texts
  heal            [2000, 4000) of each pool                  8000 texts, 2000 per pool
  probe           [5000, 5096) of each pool                  held out from everything

Why. Run 14 and 14b: a trained stitch feeds a donor block almost losslessly
(+0.008 nats), but when the block writes into the host residual mid-stack, the
host's few remaining layers cannot read it, and a random block did as well as a
real one. The reader is the bottleneck. The latent feedback transformer
(Huang et al., 2026, alphaxiv 2611.latent-feedback-transformers-scaling-law)
adds each token's top-layer state, through a learned map, to the next token's
input. We use that as a new place for a component's output to re-enter the host:
the donor block reads the host state at layer k through T_in, runs beside the
host's upper layers, and T_out writes its output into the next token's input
embedding (wsp/feedback.py). The whole host stack then reads what it wrote.

First real run (2026-11-14, pair A, feedback_block, seed 0) destroyed the model: GSM8K 0.0,
MBPP 0.0, MMLU 25.9, probe transfer -2.49, drift 2.76 nats. Three causes, all fixed (docs/feedback.md):
  1 stage 1 left a 1.98 nat penalty (linear stitch, 400 steps, 256 texts, a 1536 to 3584 map; the
    study's 0.03 was an MLP or linear stitch, 1000 steps, 2400 texts, between a 1.5B base and a
    1.5B relative of the same init). Now: MLP stitch, lr 3e-4, 1000 steps, 1200 texts, up to two
    more rounds of 500 steps while the penalty is above 0.3 nats. A cell whose penalty stays above
    the threshold ends with status failed_stage1 and does not start stage 2.
  2 one Adam step made the written vector many times the embedding norm. Now the branch output is
    RMS-normalized, T_out is normalized, and the written vector has norm gate * embed_rms with
    gate = 0.5 * tanh(raw), raw starting at 0. T_out starts at a ridge fit toward the donor
    (warm_start_t_out), else random. Gate and T_out warm up over 100 steps; clipping is per group.
    The heal halts, with status failed_heal, when the two-pass KD stays above 3x its first value
    for 20 two-pass steps in a row.
  3 training (two passes, clean reads) and exact generation differed: passes_gap KL 6.18. Now one
    pass 15 percent, three passes 30 percent after step 100, else two; T_in fine-tunes at lr 1e-5
    with the stage-1 loss as a regularizer; passes_gap on held-out text goes to the heal log every
    200 steps.
Every row records stage1_penalty, gate, written_ratio (norm of the written vector over embed_rms),
passes_gap, and status ("ok", "failed_stage1", "failed_heal").

Pairs (same tokenizer in each):
  A  base Qwen/Qwen2.5-1.5B-Instruct  donor Qwen/Qwen2.5-7B-Instruct   block donor layers 8..20
  B  base Qwen/Qwen2.5-7B-Instruct    donor Qwen/Qwen2.5-32B-Instruct  block donor layers 18..46
  C  base Qwen/Qwen2.5-0.5B-Instruct  donor Qwen/Qwen2.5-7B-Instruct   block donor layers 8..20
The host reads at the matching relative depth: at = round(first / donor layers *
host layers) host layers run first, so the read layer is k = at - 1 (A: 7, B: 7,
C: 6). Pair B keeps the same relative span (donor layers 18..46 of 64, 29 layers,
about 14B parameters, 28 GB in bf16). Memory on one H200 (141 GB): the 32B teacher
is 66 GB, the 7B host 15 GB, the block 28 GB, so 109 GB of weights, and the
activations of a batch of 8 at 384 tokens add 10 to 20 GB. It fits with little
room. If a cell runs out of memory, set ROUND22_B_BLOCK=18,30 (13 layers, 13 GB;
the span then no longer matches the relative depth of the other pairs).

Arms per pair (2 seeds for every trained arm; the seed changes the heal order,
the LoRA init, and the random control range). Every trained arm sees the same
heal texts in the same order, the same number of steps, and the same teacher
signal:
  base                   the host alone (one cell, seed 0); also the probe baseline
  kd_lora                distillation into a LoRA on the base only, rank matched to
                         the trainable count of feedback_block (T_in + T_out + gate
                         + LoRA r16)
  feedback_plain         the paper's plain latent feedback (component None): T_out of
                         the host's top state, added to the next input. The capacity control
  feedback_block         the real donor block, output to the next token's input
  feedback_block_random  a seeded other range of the same length of the same donor
  feedback_block_scrambled  the same layers with scrambled weights (wsp.scramble): shapes and
                         spectra kept, no learned structure. The knowledge-free control.
  midstack_block         the same block and the same trained T_in, output written into
                         the host residual at the matching depth (wsp/insert.py), same
                         heal: the old way, for contrast
  midstack_block_random  the same with the random range

Training. Stage 1 trains T_in on the donor's own LM loss with the donor sink
(wsp.stitch donor_lm), shared by the arms that use the same block (cached in
/vol/round22). Stage 2 trains T_out, the gates, and a LoRA r16 on the host layers
ABOVE the read layer (a LoRA below it would move the state T_in reads), with
KL(donor || student) on the math, code, and medicine pool texts plus 0.5 *
KL(frozen base || student) on the general pool texts. There is no LM loss on any
text. Feedback arms: one pass with probability 0.15, else three with probability 0.3
after step 100, else two. 1000 steps, batch 8, 384 tokens. Learning rates (untuned first
guesses): T_out 3e-4, gate 1e-3, LoRA 2e-4, T_in 1e-5. The midstack arms start the gate at 0 and
stitch_out at the pseudo-inverse of T_in, as wsp/insert.py does, with gate 5e-3 and
stitch_out 1e-4.

Rows. Every row records the wsp.probe report on the held-out probe texts
(transfer per skill against the donor, drift from the frozen base, KL values),
then the benchmark scores once through wsp.skills_eval (GSM8K 300, MBPP 200 of the
257 sanitized test items, MMLU held-out 1000, MedMCQA 1000), parameters total and
run per token, trainable parameters, heal seconds, and decode tokens per second.
Feedback arms score the multiple-choice items teacher forced with passes=2 and
generate exactly (token by token, with caches). They also record passes_gap, the
KL from the exact recursion to passes=2 on 8 probe texts, and every 200 heal steps on
8 other probe texts (heal.gap). The content gain (real
minus random transfer) is computed from the rows by --verdict.

Go rule. feedback_block beats kd_lora, feedback_plain, and feedback_block_random on
probe transfer (mean over skills) by more than the seed spread, with drift under
0.05 nats. Other outcomes:
  feedback_plain equals feedback_block      the gain is the feedback path (more
                                            compute and a new input), not the donor
  feedback_block equals feedback_block_random  the donor's content is not carried;
                                            the block only adds capacity
  midstack_block equals feedback_block      the re-entry point is not the bottleneck
                                            (the reader was not the problem)
  midstack_block below feedback_block       the re-entry point is part of the
                                            bottleneck, as run 14b suggested
  kd_lora equals feedback_block             plain distillation does as well; stop
  feedback_block drifts over 0.05 nats      the heal damages the base; lower the LoRA
                                            rate or raise the anchor weight first

    modal run modal_round22.py --only feedback_block --pairs A --seeds 0     (the smoke command: one cell)
    modal run modal_round22.py --shard 0 --nshards 8
    modal run modal_round22.py --only feedback_block,feedback_block_random --pairs A
    modal run modal_round22.py --verdict 'results/round22_shard*.json'   (reads rows, runs no cell)

Writes results/round22_shard<i>.json, full per-item records to
/vol/results/round22/<pair>_<arm>_s<seed>.json, saved networks to
/vol/written/round22_<pair>_<arm>_s<seed>, and caches T_in in /vol/round22.
"""

import glob
import json
import os
import random
import time

import modal

app = modal.App("wsp-round22")
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

PAIRS = {
    "A": {"base": "Qwen/Qwen2.5-1.5B-Instruct", "donor": "Qwen/Qwen2.5-7B-Instruct", "block": [8, 20]},
    "B": {"base": "Qwen/Qwen2.5-7B-Instruct", "donor": "Qwen/Qwen2.5-32B-Instruct", "block": [18, 46]},
    "C": {"base": "Qwen/Qwen2.5-0.5B-Instruct", "donor": "Qwen/Qwen2.5-7B-Instruct", "block": [8, 20]},
}
if os.environ.get("ROUND22_B_BLOCK"):
    PAIRS["B"]["block"] = [int(piece) for piece in os.environ["ROUND22_B_BLOCK"].split(",")]

ARMS = ["base", "kd_lora", "feedback_plain", "feedback_block", "feedback_block_random",
        "feedback_block_scrambled", "midstack_block", "midstack_block_random"]
TRAINED = ARMS[1:]
SEEDS = [0, 1]
SKILLS = ["gsm8k", "mbpp", "mmlu", "medmcqa"]
COUNTS = {"gsm8k": 300, "mbpp": 200, "mmlu": 1000, "medmcqa": 1000}

STEPS = 1000
BATCH = 8
SEQ_LEN = 384
LORA_RANK = 16
ANCHOR_WEIGHT = 0.5
ONE_PASS_PROB = 0.15
LR = {"t_out": 3e-4, "gate": 1e-3, "lora": 2e-4, "t_in": 1e-5}
G_MAX = 0.5
THREE_PASS_PROB = 0.3
SEAM_WARMUP = 100
HALT_RATIO = 3.0
HALT_PATIENCE = 20
GAP_EVERY = 200
STAGE1_THRESHOLD = 0.3
WARM_TEXTS = 256
LR_MID = {"t_out": 1e-4, "gate": 5e-3, "lora": 2e-4}
STITCH = {"kind": "mlp", "steps": 1000, "lr": 3e-4, "batch_size": 8, "seq_len": 256, "version": "v2"}

POOLS = ["math", "code", "medicine", "general"]
SKILL_POOLS = ["math", "code", "medicine"]
STITCH_START = 1000
STITCH_PER_POOL = 300
HEAL_START = 2000
HEAL_PER_POOL = 2000
PROBE_START = 5000
PROBE_PER_POOL = 96
POOL_TOTAL = 5200
STITCH_EVAL_TEXTS = 32
GAP_TEXTS = 8
GAP_LENGTH = 128


def cells():
    out = []
    for pair in PAIRS:
        out.append((pair, "base", 0))
        for arm in TRAINED:
            for seed in SEEDS:
                out.append((pair, arm, seed))
    return out


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


def pool_texts(tokenizer, name, start, count):
    from wsp import healdata

    out = []
    for user, assistant in pool_pairs(name, start, count):
        out.append(healdata.render(tokenizer, user, assistant))
    return out


def stitch_texts(tokenizer):
    texts = []
    for name in POOLS:
        texts += pool_texts(tokenizer, name, STITCH_START, STITCH_PER_POOL)
    random.Random(0).shuffle(texts)
    return texts


def heal_data(tokenizer):
    """(texts, flags). flags[i] is True for a general text (the preservation term reads those)."""
    rows = []
    for name in POOLS:
        for text in pool_texts(tokenizer, name, HEAL_START, HEAL_PER_POOL):
            rows.append((text, name == "general"))
    random.Random(0).shuffle(rows)
    return [row[0] for row in rows], [row[1] for row in rows]


def probe_texts(tokenizer):
    return {name: pool_texts(tokenizer, name, PROBE_START, PROBE_PER_POOL) for name in POOLS}


def pool_record():
    return {"pools": POOLS, "skill_pools": SKILL_POOLS, "sources": source_ids(),
            "ranges": {"stitch": [STITCH_START, STITCH_START + STITCH_PER_POOL],
                       "heal": [HEAL_START, HEAL_START + HEAL_PER_POOL],
                       "probe": [PROBE_START, PROBE_START + PROBE_PER_POOL]}}


# ---------------------------------------------------------------- eval items (scoring only)


def eval_items(ev):
    items = {
        "gsm8k": ev.gsm8k_items(count=COUNTS["gsm8k"], seed=0),
        "mbpp": ev.mbpp_items(),
        "mmlu": mmlu_heldout(ev),
        "medmcqa": ev.medmcqa_items(count=COUNTS["medmcqa"]),
    }
    order = list(range(len(items["mbpp"])))
    random.Random(0).shuffle(order)
    picked = []
    for index in sorted(order[:COUNTS["mbpp"]]):
        picked.append(items["mbpp"][index])
    items["mbpp"] = picked
    return items


def mmlu_heldout(ev):
    """MMLU test items 1000 to 2000 of the seed-0 order: the held-out half of the round 6 dev set. Scoring only."""
    items = []
    for item in ev.mmlu_items(count=COUNTS["mmlu"] + 1000)[1000:]:
        item = dict(item)
        item["id"] = item["id"].replace("mmlu_dev", "mmlu_held")
        items.append(item)
    return items


# ---------------------------------------------------------------- helpers


def load_model(name):
    import torch
    import transformers

    return transformers.AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16, device_map={"": 0})


def read_layer_of(base_config, donor_config, first):
    """The host layer whose output feeds donor layer first, at the matching relative depth."""
    at = int(round(first / donor_config.num_hidden_layers * base_config.num_hidden_layers))
    return max(0, at - 1)


def matched_rank(host, read_layer, d_donor):
    """LoRA rank (all layers) whose parameter count matches T_in + T_out + gate + LoRA r16 above the read layer."""
    from wsp.lora import params_per_rank, rank_for_budget
    from wsp.parts import find_decoder_layers

    d_host = host.config.hidden_size
    _, layers = find_decoder_layers(host, 0)
    above = list(range(read_layer + 1, len(layers)))
    target = 2 * d_host * d_donor + 1 + LORA_RANK * params_per_rank(host, layers=above)
    return rank_for_budget(host, target), target


def thin(values, every=10):
    return [round(values[index], 5) for index in range(0, len(values), every)]


def tail_mean(values, count=50):
    tail = values[-count:]
    if len(tail) == 0:
        return None
    return round(sum(tail) / len(tail), 5)


def slim_stage1(report):
    """The stage-1 report without the per-dimension arrays, so a row stays small."""
    out = {}
    for key, value in report.items():
        if key in ("before", "after"):
            out[key] = {k: v for k, v in value.items() if k != "r2_dim"}
        elif key == "ridge":
            out[key] = {k: v for k, v in value.items() if k != "r2_dim"}
        else:
            out[key] = value
    return out


def get_t_in(pair, first, seed, host, donor, tokenizer, read_layer):
    """Stage 1, shared by the arms that use the same donor layer. Cached on the volume.

    The report has penalty (donor loss through T_in minus native, in nats), threshold, and ok.
    """
    import torch

    from wsp.feedback import fit_t_in
    from wsp.stitch import build_stitch

    path = f"/vol/round22/t_in_{STITCH['version']}_{pair}_{first}_l{read_layer}_s{seed}.pt"
    if os.path.exists(path):
        saved = torch.load(path)
        t_in = build_stitch({"kind": STITCH["kind"]}, host.config.hidden_size, donor.config.hidden_size)
        t_in.load_state_dict(saved["state"])
        return t_in, saved["sink"], saved["report"]
    texts = stitch_texts(tokenizer)
    held = probe_texts(tokenizer)
    eval_texts = held["math"][:11] + held["code"][:11] + held["medicine"][:10]
    t_in, sink, report = fit_t_in(host, tokenizer, read_layer, donor, tokenizer, first, texts, kind=STITCH["kind"],
                                  steps=STITCH["steps"], lr=STITCH["lr"], batch_size=STITCH["batch_size"],
                                  seq_len=STITCH["seq_len"], seed=seed, eval_texts=eval_texts, sink=True,
                                  log_every=100, threshold=STAGE1_THRESHOLD)
    report = slim_stage1(report)
    os.makedirs("/vol/round22", exist_ok=True)
    torch.save({"state": t_in.state_dict(), "sink": sink, "report": report}, path)
    volume.commit()
    return t_in, sink, report


def finish_failed(row, status, reason, start, pair, arm, seed):
    """Write a failed cell's row (no benchmark runs) and return it."""
    row["status"] = status
    row["failure"] = reason
    row["seconds"] = round(time.time() - start, 1)
    os.makedirs("/vol/results/round22", exist_ok=True)
    with open(f"/vol/results/round22/{pair}_{arm}_s{seed}.json", "w") as f:
        json.dump(row, f, default=str)
    volume.commit()
    print(f"CELL {pair} {arm} s{seed} {status}: {reason}")
    return row


def heal_row(heal_log):
    return {"kd": thin(heal_log["kd"]), "anchor": thin(heal_log["anchor"]),
            "kd_last50": tail_mean(heal_log["kd"]), "anchor_last50": tail_mean(heal_log["anchor"]),
            "passes_counts": {str(n): heal_log["passes"].count(n) for n in (1, 2, 3)},
            "gap": heal_log.get("gap", []), "halted": heal_log.get("halted", False),
            "halt_step": heal_log.get("halt_step"), "halt_reason": heal_log.get("halt_reason"),
            "baseline_kd": heal_log.get("baseline_kd")}


def probe_row(written, base_view, donor, tokenizer):
    """The wsp.probe report on the held-out probe texts: transfer per skill, drift, and a verdict word."""
    from wsp import probe

    held = probe_texts(tokenizer)
    skills = {}
    for name in SKILL_POOLS:
        skills[name] = {"donor": donor, "texts": held[name]}
    return probe.report(written, base_view, tokenizer, skills, held["general"], max_length=SEQ_LEN, batch_size=4)


# ---------------------------------------------------------------- cells


@app.function(image=image, volumes={"/vol": volume}, gpu="H200", memory=262144, timeout=10 * 3600)
def run_cell(pair, arm, seed):
    import torch
    import transformers

    from wsp import feedback as fb
    from wsp import insert
    from wsp import skills_eval as ev
    from wsp.lora import apply_lora

    start = time.time()
    torch.manual_seed(seed)
    spec = PAIRS[pair]
    base_id, donor_id = spec["base"], spec["donor"]
    first, last = spec["block"]
    tokenizer = transformers.AutoTokenizer.from_pretrained(base_id)
    row = {"pair": pair, "arm": arm, "seed": seed, "base": base_id, "donor": donor_id, "block": [first, last],
           "eval": "wsp.skills_eval", "status": "ok"}
    row.update(pool_record())

    host = load_model(base_id).eval()
    donor = load_model(donor_id).eval()
    read_layer = read_layer_of(host.config, donor.config, first)
    n_host = host.config.num_hidden_layers
    row["read_layer"] = read_layer
    lora_layers = list(range(read_layer + 1, n_host))
    texts, flags = heal_data(tokenizer)

    model = host
    seam = None
    base_view = fb.BaseView(host)
    heal_log = None
    trainable_all = 0

    if arm == "base":
        pass
    elif arm == "kd_lora":
        rank, target = matched_rank(host, read_layer, donor.config.hidden_size)
        row["lora_rank"] = rank
        row["match_target"] = target
        for param in host.parameters():
            param.requires_grad_(False)
        lora = apply_lora(host, rank)
        groups = [{"params": lora, "lr": LR["lora"]}]
        base_view = fb.BaseView(host)

        def forward(ids, mask, passes):
            return host(input_ids=ids, attention_mask=mask, use_cache=False).logits

        def anchor(ids, mask):
            return fb.base_logits(host, ids, mask)

        heal_log = fb.heal_distill(forward, anchor, groups, tokenizer, donor, texts, flags, steps=STEPS,
                                   batch_size=BATCH, seq_len=SEQ_LEN, seed=seed, one_pass_prob=0.0,
                                   three_pass_prob=0.0, anchor_weight=ANCHOR_WEIGHT, autocast_dtype=torch.bfloat16,
                                   halt_ratio=HALT_RATIO, halt_patience=HALT_PATIENCE)
        host.eval()
    elif arm.startswith("feedback"):
        if arm == "feedback_plain":
            branch = fb.make_plain_branch(host, g_max=G_MAX, init_seed=seed)
        else:
            control = "random" if arm.endswith("random") else None
            if arm.endswith("scrambled"):
                control = "scrambled"
            use_first = first
            if control == "random":
                from wsp.feedback import random_range

                use_first = random_range(donor.config.num_hidden_layers, first, last, seed)[0]
            t_in, sink, stage1 = get_t_in(pair, use_first, seed, host, donor, tokenizer, read_layer)
            row["stage1"] = stage1
            row["stage1_penalty"] = stage1["penalty"]
            if not stage1["ok"]:
                return finish_failed(row, "failed_stage1", f"stage 1 penalty {stage1['penalty']:.3f} nats is above "
                                     f"{STAGE1_THRESHOLD}", start, pair, arm, seed)
            branch = fb.make_block_branch(host, donor, [first, last], read_layer, t_in=t_in, sink=sink,
                                          control=control, control_seed=seed, g_max=G_MAX, init_seed=seed)
            row["donor_layers_used"] = branch.meta["donor_layers"]
        seam = fb.FeedbackSeam(host, [branch], base_id)
        seam.add_host_lora(LORA_RANK, layers=lora_layers)
        trainable_all = fb.count_params(seam)["trainable"]
        row["params_trainable_before_stage2"] = trainable_all
        skill_texts = [text for text, flag in zip(texts, flags) if not flag][:WARM_TEXTS]
        row["warm_start"] = fb.warm_start_t_out(seam, tokenizer, donor, skill_texts, batch_size=4, seq_len=256)
        held_gap = probe_texts(tokenizer)["math"][GAP_TEXTS:2 * GAP_TEXTS]
        heal_log = fb.heal_seam(seam, tokenizer, donor, texts, flags,
                                lrs={"lr_t_out": LR["t_out"], "lr_gate": LR["gate"], "lr_lora": LR["lora"],
                                     "lr_t_in": LR["t_in"], "warmup_seam": SEAM_WARMUP},
                                gap_texts=held_gap, gap_every=GAP_EVERY, train_t_in=arm != "feedback_plain",
                                steps=STEPS, batch_size=BATCH, seq_len=SEQ_LEN, seed=seed,
                                one_pass_prob=ONE_PASS_PROB, three_pass_prob=THREE_PASS_PROB,
                                three_pass_after=SEAM_WARMUP, anchor_weight=ANCHOR_WEIGHT,
                                autocast_dtype=torch.bfloat16, halt_ratio=HALT_RATIO, halt_patience=HALT_PATIENCE)
        row["heal_seconds"] = heal_log["seconds"]
        row["heal"] = heal_row(heal_log)
        row["gates"] = [round(float(b.gate), 4) for b in seam.branches]
        row["gate"] = row["gates"][0]
        if heal_log["halted"]:
            return finish_failed(row, "failed_heal", heal_log["halt_reason"], start, pair, arm, seed)
        model = seam
        base_view = fb.BaseView(seam)
        out_dir = f"/vol/written/round22_{pair}_{arm}_s{seed}"
        fb.save_seam(seam, out_dir, tokenizer, base_id)
        volume.commit()
        row["saved"] = out_dir
    elif arm.startswith("midstack"):
        control = arm.endswith("random")
        use_first, use_last = first, last
        if control:
            from wsp.feedback import random_range

            use_first, use_last = random_range(donor.config.num_hidden_layers, first, last, seed)
        t_in, sink, stage1 = get_t_in(pair, use_first, seed, host, donor, tokenizer, read_layer)
        row["stage1"] = stage1
        row["stage1_penalty"] = stage1["penalty"]
        if not stage1["ok"]:
            return finish_failed(row, "failed_stage1", f"stage 1 penalty {stage1['penalty']:.3f} nats is above "
                                 f"{STAGE1_THRESHOLD}", start, pair, arm, seed)
        row["donor_layers_used"] = [use_first, use_last]
        insertion = insert.insert_layers(host, donor, [use_first, use_last], read_layer + 1, stitch_in=t_in,
                                         donor_id=donor_id, sink=sink, gate_init=0.0)
        insert.add_base_lora(host, LORA_RANK, layers=lora_layers)
        groups = fb_midstack_groups(host, insertion)
        trainable_all = sum(p.numel() for g in groups for p in g["params"]) + sum(
            p.numel() for p in insertion.w_in.parameters())
        row["params_trainable_before_stage2"] = trainable_all

        def forward(ids, mask, passes):
            return host(input_ids=ids, attention_mask=mask, use_cache=False).logits

        def anchor(ids, mask):
            return fb.base_logits(host, ids, mask)

        heal_log = fb.heal_distill(forward, anchor, groups, tokenizer, donor, texts, flags, steps=STEPS,
                                   batch_size=BATCH, seq_len=SEQ_LEN, seed=seed, one_pass_prob=0.0,
                                   three_pass_prob=0.0, anchor_weight=ANCHOR_WEIGHT, autocast_dtype=torch.bfloat16,
                                   halt_ratio=HALT_RATIO, halt_patience=HALT_PATIENCE)
        host.eval()
        row["gates"] = insert.gate_values(host)
        out_dir = f"/vol/written/round22_{pair}_{arm}_s{seed}"
        insert.save_inserted(host, tokenizer, out_dir, base_id)
        volume.commit()
        row["saved"] = out_dir
        base_view = fb.BaseView(host)
    else:
        raise ValueError(f"unknown arm {arm}")

    if heal_log is not None:
        row["heal_seconds"] = heal_log["seconds"]
        row["heal"] = heal_row(heal_log)
        if heal_log.get("halted"):
            return finish_failed(row, "failed_heal", heal_log["halt_reason"], start, pair, arm, seed)

    # ---- parameters
    if seam is not None:
        counts = fb.count_params(seam)
        row["params_total"] = counts["total"]
        row["params_per_token"] = counts["per_token"]
        row["params_trainable"] = counts["trainable"]
        row["params_by_kind"] = {k: counts[k] for k in ["host", "lora", "component", "t_in", "t_out", "gate"]}
    elif arm.startswith("midstack"):
        counts = insert.param_counts(host)
        row["params_total"] = counts["total"]
        row["params_per_token"] = counts["per_token"]
        row["params_trainable"] = counts["trainable"]
    else:
        total, run = ev.model_params(host)
        row["params_total"] = total
        row["params_per_token"] = run
        row["params_trainable"] = 0
        if arm != "base":
            row["params_trainable"] = sum(p.numel() for p in host.parameters() if p.requires_grad)

    # ---- the benchmark-free probe, before any benchmark is touched
    model.eval()
    report = probe_row(model, base_view, donor, tokenizer)
    row["probe"] = report
    row["transfer_mean"] = fb.mean_transfer(row)
    if seam is not None:
        held = probe_texts(tokenizer)
        row["passes_gap"] = fb.passes_gap(seam, tokenizer, held["math"][:GAP_TEXTS], passes=2,
                                          max_length=GAP_LENGTH)
        found = fb.feedback_stats(seam, tokenizer, held["math"][:GAP_TEXTS], max_length=GAP_LENGTH)[0]
        row["gate"] = round(found["gate"], 5)
        row["written_ratio"] = round(found["ratio"], 5) if found["ratio"] is not None else None
        row["written_norm"] = found["written_norm"]
        row["embed_rms"] = found["embed_rms"]
    del donor
    torch.cuda.empty_cache()

    # ---- benchmark scores, once
    items = eval_items(ev)
    row["gsm8k_prompt_sample"] = ev.render_prompt(tokenizer, ev.user_text("gsm8k", items["gsm8k"][0]))
    result = ev.evaluate_suite(model, tokenizer, skills=SKILLS, items=items, batch_size=32)
    row["scores"] = dict(result["scores"])
    row["macro"] = result["suite"]
    row["decode_tok_s"] = round(ev.decode_speed(model, tokenizer, new_tokens=64), 1)
    row["seconds"] = round(time.time() - start, 1)

    os.makedirs("/vol/results/round22", exist_ok=True)
    full = dict(row)
    full["skills"] = ev.strip_outputs(result)["skills"]
    with open(f"/vol/results/round22/{pair}_{arm}_s{seed}.json", "w") as f:
        json.dump(full, f, default=str)
    volume.commit()
    return row


def fb_midstack_groups(host, insertion):
    """Freeze all but stitch_out, the gate, and the host LoRA. T_in stays as stage 1 left it."""
    from wsp.insert import insertion_parameters

    for param in host.parameters():
        param.requires_grad_(False)
    found = insertion_parameters(host)
    out_params = list(insertion.w_out.linear.parameters())
    groups = []
    for name, params, lr in (("t_out", out_params, LR_MID["t_out"]), ("gate", found["gate"], LR_MID["gate"]),
                             ("lora", found["base_lora"], LR_MID["lora"])):
        for param in params:
            param.requires_grad_(True)
        groups.append({"params": params, "lr": lr, "name": name})
    return groups


def print_verdict(path):
    from wsp.feedback import go_verdict

    rows = []
    for name in sorted(glob.glob(path)):
        with open(name) as f:
            rows += json.load(f)
    if len(rows) == 0:
        raise SystemExit(f"no rows in {path}")
    print(json.dumps(go_verdict(rows), indent=2))


@app.local_entrypoint()
def main(shard: int = 0, nshards: int = 1, only: str = "", pairs: str = "", verdict: str = "", seeds: str = ""):
    if verdict:
        print_verdict(verdict)
        return
    todo = cells()
    if only:
        wanted = set(only.split(","))
        todo = [cell for cell in todo if cell[1] in wanted]
    if pairs:
        keep = set(pairs.split(","))
        todo = [cell for cell in todo if cell[0] in keep]
    if seeds:
        keep_seeds = set(seeds.split(","))
        todo = [cell for cell in todo if str(cell[2]) in keep_seeds]
    mine = []
    for index, cell in enumerate(todo):
        if index % nshards == shard:
            mine.append(cell)
    print(f"shard {shard}/{nshards}: {mine}")
    rows = []
    for row in run_cell.starmap(mine, return_exceptions=True):
        if isinstance(row, Exception):
            print("cell failed:", repr(row)[:600])
            continue
        rows.append(row)
        print(json.dumps({k: row.get(k) for k in ["pair", "arm", "seed", "scores", "macro", "transfer_mean",
                                                  "params_total", "params_per_token", "heal_seconds",
                                                  "decode_tok_s", "status", "stage1_penalty", "gate",
                                                  "written_ratio", "passes_gap", "failure"]}))
    os.makedirs("results", exist_ok=True)
    with open(f"results/round22_shard{shard}.json", "w") as f:
        json.dump(rows, f, indent=2)
