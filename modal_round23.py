"""Round 23: give the parallel graft a training signal that needs the donor's knowledge.

Run 22 showed the cross-tokenizer stitch works (about 90% of Qwen2.5-7B's
next-token skill survives on Llama-3.2-1B states) and the base reads it, but
LM loss on healdata QA text rewards only what the text holds, so LoRA on the
same text matches it. Three fixes, each its own experiment, each with the same
three arms at one trainable budget (15.2M): pgraft (Qwen layers 10..20 at
Llama layer 7, run 22's v3 setup), identity (same Enc, Dec, gate, no donor),
lora (rank matched). 2 seeds.

  A  answer-letter training (the tap's signal in round 6): cross-entropy over
     A..D on 4-option multiple-choice items from non-benchmark pools
     (CommonsenseQA, OpenBookQA, QASC, LogiQA2.0 train), MMLU prompt format,
     plus KL to the frozen base on general text. MMLU is never trained on.
  B  cross-tokenizer distillation: KL(donor || base) at offset-paired
     positions over the 109,566 token strings both vocabularies share, on
     healdata domain and general text (wsp.pgraft.kd_loss).
  C  output alignment, then the task: B's distillation first (for pgraft it
     trains Dec and gate only, the mirror of stage 1's Enc alignment), then
     A's answer-letter training.

Step counts match per experiment: pgraft spends 600 of them on stage-1
alignment, the controls spend them on their first stage. Scored on MMLU dev
1000 / held-out 2000, held-out MC items from the training pools, held-out
domain LM loss, the distillation gap to the donor on held-out domain text,
and KL to the frozen base on legal text no arm trains on (the regression
metric).

    modal run modal_round23.py --exp A
    modal run modal_round23.py --exp all
"""

import json
import os
import random
import time

import modal

app = modal.App("wsp-round23")
volume = modal.Volume.from_name("wsp-models", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch==2.7.1", "transformers==4.57.1", "accelerate", "huggingface_hub", "requests",
        "safetensors", "zstandard", "numpy", "datasets", "sentencepiece", "protobuf",
    )
    .env({"HF_HOME": "/vol/hf"})
    .add_local_python_source("wsp")
    .add_local_file("modal_graft.py", "/root/modal_graft.py")
    .add_local_file("modal_round5.py", "/root/modal_round5.py")
    .add_local_file("modal_round6.py", "/root/modal_round6.py")
    .add_local_file("modal_round7.py", "/root/modal_round7.py")
    .add_local_file("modal_round22.py", "/root/modal_round22.py")
)

EXPS = ["A", "B", "C"]
ARMS = ["pgraft", "identity", "lora"]
SEEDS = [0, 1]
ALIGN_STEPS = 600
TASK_STEPS = 1000
KD_STEPS = {"B": 1000, "C": 500}
MC_BATCH = 8
KD_BATCH = 4
LR = {"glue": 1e-3, "gate": 1e-2, "enc2": 1e-4, "lora": 2e-4}
PRESERVE = 1.0
ALIGN_REG = 1.0
MC_PATH = "/vol/round23/mc.json"
LAW_PATH = "/vol/round23/law.json"
SMOKE = {"align": 3, "task": 3, "kd": 3}


def mc_items():
    """4-option items in MMLU prompt format from non-benchmark train splits: {"train": [...], "eval": [...]}."""
    from modal_round6 import mmlu_prompt
    from wsp import healdata

    if os.path.exists(MC_PATH):
        with open(MC_PATH) as f:
            return json.load(f)
    sources = [s for s in healdata.SOURCES["reasoning"] if s[3] in ("mc", "logiqa")]
    healdata.check_sources([s[0] for s in sources])
    rng = random.Random(0)
    items = []
    for dataset_id, config, split, kind in sources:
        rows = healdata._load(dataset_id, config, split)
        order = list(range(len(rows)))
        rng.shuffle(order)
        taken = 0
        for index in order:
            if taken >= 3000:
                break
            row = rows[index]
            if kind == "mc":
                question = row.get("question") or row.get("question_stem")
                labels = list(row["choices"]["label"])
                texts = list(row["choices"]["text"])
                if row["answerKey"] not in labels or not question:
                    continue
                gold = texts[labels.index(row["answerKey"])]
                others = [t for t in texts if t != gold]
            else:
                try:
                    data = json.loads(row["text"])
                except ValueError:
                    continue
                if "options" not in data or "question" not in data:
                    continue
                question = data["text"].strip() + "\n\n" + data["question"].strip()
                texts = list(data["options"])
                gold = texts[int(data["answer"])]
                others = [t for t in texts if t != gold]
            if len(others) < 3:
                continue
            choices = rng.sample(others, 3) + [gold]
            rng.shuffle(choices)
            items.append([mmlu_prompt(question, choices), choices.index(gold)])
            taken += 1
    rng.shuffle(items)
    out = {"train": items[400:], "eval": items[:400]}
    os.makedirs(os.path.dirname(MC_PATH), exist_ok=True)
    with open(MC_PATH, "w") as f:
        json.dump(out, f)
    volume.commit()
    return out


def law_texts():
    """Held-out regression text: the healdata law pool, which no arm trains on."""
    from wsp import healdata

    if os.path.exists(LAW_PATH):
        with open(LAW_PATH) as f:
            return json.load(f)
    texts = healdata.pool("law", 80, seed=5)
    os.makedirs(os.path.dirname(LAW_PATH), exist_ok=True)
    with open(LAW_PATH, "w") as f:
        json.dump(texts, f)
    volume.commit()
    return texts


@app.function(image=image, volumes={"/vol": volume}, gpu="H100", memory=131072, timeout=4 * 3600)
def cell(exp, arm, seed, smoke=False):
    import gc

    import torch
    import transformers

    import modal_round22 as r22
    from modal_round6 import letter_ids, score_mmlu
    from modal_round7 import mmlu_split
    from wsp import pgraft
    from wsp.lora import apply_lora, default_alpha, rank_for_budget

    start = time.time()
    torch.manual_seed(seed)
    rng = random.Random(seed + 3)
    data = r22.build_data()
    mc = mc_items()
    law = law_texts()
    align_steps, task_steps, kd_steps = ALIGN_STEPS, TASK_STEPS, KD_STEPS.get(exp, 0)
    if smoke:
        align_steps, task_steps, kd_steps = SMOKE["align"], SMOKE["task"], (SMOKE["kd"] if kd_steps else 0)
        data = {key: value[:16] for key, value in data.items()}
        mc = {"train": mc["train"][:64], "eval": mc["eval"][:32]}
        law = law[:8]
    cfg = dict(pgraft.DEFAULTS, **r22.CFG)
    base, tok, donor, donor_tok = r22.load_models(with_donor=True)
    ref = transformers.AutoModelForCausalLM.from_pretrained(r22.HOST, dtype=torch.bfloat16).to("cuda").eval()
    for p in ref.parameters():
        p.requires_grad_(False)
    vmap = pgraft.vocab_map(tok, donor_tok)
    letters = letter_ids(tok)
    device = torch.device("cuda")
    row = {"round": 23, "exp": exp, "arm": arm, "seed": seed, "shared_vocab": int(vmap[0].numel())}
    budget = (pgraft.count(pgraft.Enc(r22.D_HOST, r22.D_DONOR, depth=2).parameters())
              + pgraft.count(pgraft.Dec(r22.D_DONOR, r22.D_HOST, rank=256).parameters()) + r22.D_HOST + 1)
    row["budget"] = budget

    # ---- build the arm
    graft = None
    if arm == "lora":
        rank = rank_for_budget(base, budget)
        lora_params = apply_lora(base, rank, default_alpha(rank))
        for p in lora_params:
            p.requires_grad_(True)
        row["lora_rank"] = rank
        row["trainable"] = pgraft.count(lora_params)
    else:
        control = "identity" if arm == "identity" else None
        graft = r22.make_graft(base, donor if control is None else None, donor_tok, 10, control, data, 256)
        row["trainable"] = pgraft.count(pgraft.trainable([graft], enc=True))
        if arm == "pgraft":
            pgraft.align(base, tok, donor, donor_tok, graft, data["align"], dict(cfg, align_steps=align_steps),
                         seed=seed, log_every=100)
            row["alignment"] = pgraft.alignment_report(base, tok, donor, donor_tok, graft, data["align_eval"], cfg)
            print("alignment", row["alignment"])
        row["calib_rms"] = pgraft.calibrate_scale(base, tok, graft, data["align"], cfg)

    def groups(stage):
        """Param groups for a stage. stage "kd_out" (exp C, pgraft): Dec and gate only."""
        if arm == "lora":
            return [{"params": lora_params, "lr": LR["lora"]}]
        gate = list(graft.token_gate.parameters())
        dec = list(graft.dec.parameters())
        enc = list(graft.enc.parameters())
        if arm == "identity":
            return [{"params": enc + dec, "lr": LR["glue"]}, {"params": gate, "lr": LR["gate"]}]
        if stage == "kd_out":
            for p in enc:
                p.requires_grad_(False)
            return [{"params": dec, "lr": LR["glue"]}, {"params": gate, "lr": LR["gate"]}]
        for p in enc:
            p.requires_grad_(True)
        return [{"params": dec, "lr": LR["glue"]}, {"params": gate, "lr": LR["gate"]},
                {"params": enc, "lr": LR["enc2"]}]

    align_feed = pgraft._batches(data["align"], cfg["align_batch"], seed + 11)

    def align_reg(step, stage):
        # pgraft with a trainable Enc keeps the stage-1 functional KL as a regularizer (every 2nd step).
        if arm != "pgraft" or stage == "kd_out" or step % 2:
            return None
        loss, _ = pgraft.functional_loss(base, tok, donor, donor_tok, graft, next(align_feed), cfg)
        return ALIGN_REG * loss

    kd_texts = data["domain"] + data["general"]
    kd_feed = pgraft._batches(kd_texts, KD_BATCH, seed + 21)
    mc_feed = pgraft._batches(mc["train"], MC_BATCH, seed + 31)
    gen_feed = pgraft._batches(data["general"], KD_BATCH, seed + 41)

    def kd_step(stage):
        def step_fn(step):
            loss, info = pgraft.kd_loss(base, tok, donor, donor_tok, next(kd_feed), vmap, cfg)
            reg = align_reg(step, stage)
            if reg is not None:
                loss = loss + reg
                info["reg"] = float(reg.detach())
            return loss, info
        return step_fn

    def mc_step(step):
        loss, info = pgraft.mc_loss(base, tok, [tuple(x) for x in next(mc_feed)], letters, device)
        enc = pgraft._tok_batch(tok, next(gen_feed), cfg["seq_len"], device)
        _, logits = pgraft.seq_lm_loss(base, enc, return_logits=True)
        drift = pgraft.drift_kl(ref, enc, logits)
        loss = loss + PRESERVE * drift
        info["drift"] = float(drift.detach())
        reg = align_reg(step, "mc")
        if reg is not None:
            loss = loss + reg
            info["reg"] = float(reg.detach())
        return loss, info

    probe = None
    if arm == "pgraft":
        def probe(step):
            return pgraft.alignment_report(base, tok, donor, donor_tok, graft, data["align_eval"][:8], cfg)

    # The controls get pgraft's alignment steps in their first stage, so every arm takes the same step count.
    extra = 0 if arm == "pgraft" else align_steps
    logs = {}
    if exp in ("B", "C"):
        stage = "kd_out" if (exp == "C" and arm == "pgraft") else "kd"
        logs["kd"] = pgraft.train(base, groups(stage), kd_steps + extra, kd_step(stage), seed=seed, probe_fn=probe)
        extra = 0
        row["after_kd"] = pgraft.kd_report(base, tok, donor, donor_tok, data["domain_eval"][:60], vmap, cfg)
    if exp in ("A", "C"):
        logs["mc"] = pgraft.train(base, groups("mc"), task_steps + extra, mc_step, seed=seed + 1, probe_fn=probe)
    for name, log in logs.items():
        row[name + "_first50"] = sum(r["loss"] for r in log[:50]) / len(log[:50])
        row[name + "_last50"] = sum(r["loss"] for r in log[-50:]) / len(log[-50:])
        fid = [[r["step"], round(r["probe"]["survival"], 4)] for r in log if "probe" in r]
        if fid:
            row[name + "_fidelity"] = fid

    # ---- score
    base.eval()
    if arm == "pgraft":
        row["alignment_end"] = pgraft.alignment_report(base, tok, donor, donor_tok, graft, data["align_eval"], cfg)
    row["kd_gap"] = pgraft.kd_report(base, tok, donor, donor_tok, data["domain_eval"][:60], vmap, cfg)
    row["domain_eval"] = pgraft.heldout_utility(base, tok, data["domain_eval"], cfg, ref=ref)
    row["law_eval"] = pgraft.heldout_utility(base, tok, law, cfg, ref=ref)
    del donor
    gc.collect()
    torch.cuda.empty_cache()
    _, dev = mmlu_split(tok, heldout=False)
    _, held = mmlu_split(tok, heldout=True)
    if smoke:
        dev, held = dev[:32], held[:32]
    mc_eval = [tuple(x) for x in mc["eval"]]
    row["mmlu_dev"] = score_mmlu(base, tok, dev, batch_size=16)
    row["mmlu_heldout"] = score_mmlu(base, tok, held, batch_size=16)
    row["mc_eval"] = score_mmlu(base, tok, mc_eval, batch_size=16)
    if graft is not None:
        with pgraft.grafts_off(base):
            row["mmlu_heldout_graft_off"] = score_mmlu(base, tok, held, batch_size=16)
            row["mc_eval_graft_off"] = score_mmlu(base, tok, mc_eval, batch_size=16)
    row["seconds"] = round(time.time() - start, 1)
    print(json.dumps(row))
    return row


@app.local_entrypoint()
def main(exp: str = "all", only: str = "", smoke: bool = False, tag: str = ""):
    exps = EXPS if exp == "all" else exp.split(",")
    calls = [(e, arm, seed, smoke) for e in exps for arm in ARMS for seed in SEEDS]
    if only:
        calls = []
        for item in only.split(","):
            e, arm, seed = item.split(":")
            calls.append((e, arm, int(seed), smoke))
    rows = []
    for row in cell.starmap(calls, return_exceptions=True):
        if isinstance(row, Exception):
            print("cell failed:", repr(row)[:800])
            continue
        rows.append(row)
        print(json.dumps(row))
    os.makedirs("results", exist_ok=True)
    name = "round23_smoke" if smoke else "round23" + (("_" + tag) if tag else "") + "_rows"
    with open(f"results/{name}.json", "w") as f:
        json.dump(rows, f, indent=2)
    print("wrote", f"results/{name}.json")
