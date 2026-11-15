"""Write-in track big-donor-same-tok: Qwen2.5-14B-Instruct slice written into Qwen2.5-1.5B-Instruct.

Base and donor share a tokenizer, so the donor's full logits are an exact
teacher at every position. The graft is wsp.pgraft's ParallelGraft: base
residual after host layer h goes through Enc into a frozen donor slice
(first..last), Dec reads the slice's contribution, a token gate writes it
back. Stage 1 aligns Enc on the donor's own next-token KL; stage 2 trains
Dec, gate and Enc on one loss that every arm shares:

    answer-letter CE on MC items (MMLU auxiliary_train, ARC, OBQA, CSQA, QASC,
    SciQ, LogiQA train splits)
  + KL(donor letters || student letters) on the same items (soft targets)
  + KL(donor || student) over the full vocabulary on knowledge/general text

Arms (same data, same total steps, final checkpoint, no checkpoint selection):
graft, identity (donor skipped), scrambled (slice scrambled with wsp/scramble.py),
lora (rank for the graft's trainable count), lora128, fullft. Reference rows
(no training): base, donor 14B, native 3B and 7B siblings.

Test sets (no arm trains on them): MMLU held-out 2000 and dev 1000, ARC-Challenge
test, TriviaQA validation 1000 (exact match, 5-shot), GSM8K test 300 (guard).

    modal run modal_wi_bigsame.py --mode prep
    modal run modal_wi_bigsame.py --mode soft
    modal run modal_wi_bigsame.py --mode ref --cells base,donor
    modal run --detach modal_wi_bigsame.py --mode run --cells main_graft_s0 --smoke
"""

import json
import os
import random
import time

import modal

app = modal.App("wsp-wi-bigsame")
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
)

BASE = "Qwen/Qwen2.5-1.5B-Instruct"
DONOR = "Qwen/Qwen2.5-14B-Instruct"
SIBLINGS = {"sib3b": "Qwen/Qwen2.5-3B-Instruct", "sib7b": "Qwen/Qwen2.5-7B-Instruct"}
ROOT = "/vol/wi_bigsame"
VOCAB = 151665
TRAIN_POOL = 60000

DEFAULT = {
    "steps": 3000, "align_steps": 600, "first": 14, "last": 34, "host": 14, "enc_depth": 2, "dec_rank": 256,
    "mc_batch": 16, "text_batch": 6, "seq_len": 256, "w_soft": 1.0, "w_text": 1.0, "w_align": 1.0,
    "lr_glue": 3e-4, "lr_gate": 1e-4, "lr_enc2": 1e-4, "lr_lora": 2e-4, "lr_lora128": 1e-4, "lr_full": 1e-5,
    "dev": False, "full_eval": True,
    "write": None, "graft_lora_rank": 0, "micro": 1, "kd_stage": 0, "kd_lr_mult": 3.0, "out_frac": 0.0, "logit": False,
}
SMOKE = {"steps": 4, "align_steps": 3}


def presets():
    out = {}
    dev = {"steps": 1500, "align_steps": 500, "dev": True, "full_eval": False}
    out["d_graft_a"] = ("graft", 0, dict(dev, first=14, last=34))
    out["d_graft_b"] = ("graft", 0, dict(dev, first=8, last=34))
    out["d_graft_c"] = ("graft", 0, dict(dev, first=20, last=40))
    out["d_lora_lo"] = ("lora", 0, dict(dev, lr_lora=2e-4))
    out["d_lora_hi"] = ("lora", 0, dict(dev, lr_lora=6e-4))
    out["d_lora128_lo"] = ("lora128", 0, dict(dev, lr_lora128=5e-5))
    out["d_lora128_hi"] = ("lora128", 0, dict(dev, lr_lora128=2e-4))
    out["d_full_lo"] = ("fullft", 0, dict(dev, lr_full=5e-6))
    out["d_full_hi"] = ("fullft", 0, dict(dev, lr_full=2e-5))
    out["d_identity"] = ("identity", 0, dict(dev))
    out["s_a"] = ("graft", 0, dict(dev, first=14, last=47, write=27, micro=2))
    out["s_b"] = ("graft", 0, dict(dev, first=14, last=47, write=21, micro=2))
    out["s_c"] = ("graft", 0, dict(dev, first=14, last=47, write=27, dec_rank=1024, graft_lora_rank=16, micro=2))
    out["s_d"] = ("graft", 0, dict(dev, first=14, last=47, write=27, micro=2, kd_stage=600, kd_lr_mult=3.0))
    out["s_e"] = ("graft", 0, dict(dev, first=14, last=34, kd_stage=600, kd_lr_mult=3.0))
    out["s_f"] = ("graft", 0, dict(dev, first=14, last=47, write=27, micro=2, kd_stage=600, out_frac=1.0))
    out["s_g"] = ("graft", 0, dict(dev, first=14, last=34, out_frac=1.0))
    out["s_h"] = ("graft", 0, dict(dev, first=14, last=47, write=21, micro=2, kd_stage=600, out_frac=1.0))
    lg = dict(dev, logit=True, first=14, last=47, host=14, micro=2, lr_gate=1e-3)
    out["l_a"] = ("graft", 0, dict(lg))
    out["l_b"] = ("scrambled", 0, dict(lg))
    out["l_c"] = ("graft", 0, dict(lg, host=20, first=24))
    out["l_d"] = ("identity", 0, dict(lg))
    out["s_c_lora"] = ("lora", 0, dict(dev, graft_lora_rank=16, dec_rank=1024))
    out["smoke_graft"] = ("graft", 0, {})
    for arm in ["graft", "identity", "scrambled", "lora", "lora128", "fullft"]:
        out["smoke_" + arm] = (arm, 0, {})
        for seed in (0, 1):
            out[f"main_{arm}_s{seed}"] = (arm, seed, {})
    return out


# ------------------------------------------------------------------ data


@app.function(image=image, volumes={"/vol": volume}, cpu=4, memory=32768, timeout=3 * 3600)
def prep():
    """MC train pool, eval sets, text pools. Cached in ROOT/data.json."""
    import datasets
    import transformers

    from wsp import healdata, wi

    start = time.time()
    tok = transformers.AutoTokenizer.from_pretrained(BASE)
    rng = random.Random(0)
    items = {}
    counts = {}

    def add(source, prompt, gold):
        if prompt not in items:
            items[prompt] = (gold, source)
            counts[source] = counts.get(source, 0) + 1

    aux = datasets.load_dataset("cais/mmlu", "auxiliary_train", split="train")
    for row in aux:
        if "train" in row and isinstance(row["train"], dict):
            row = row["train"]
        if len(row["choices"]) == 4:
            add("mmlu_aux", wi.mc_prompt(row["question"], row["choices"]), int(row["answer"]))
    for dataset_id, config, split, kind in [s for s in healdata.SOURCES["reasoning"] if s[3] in ("mc", "logiqa")]:
        healdata.check_sources([dataset_id])
        rows = healdata._load(dataset_id, config, split)
        for row in rows:
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
            add(dataset_id, wi.mc_prompt(question, choices), choices.index(gold))
    for row in datasets.load_dataset("allenai/sciq", split="train"):
        p, g, _ = wi.sciq_item(row, rng)
        add("sciq", p, g)
    for config in ("ARC-Challenge", "ARC-Easy"):
        for row in datasets.load_dataset("allenai/ai2_arc", config, split="train"):
            r = wi.arc_item(row)
            if r is not None and r[2] == 4:
                add("arc_train_" + config, r[0], r[1])
    prompts = list(items)
    lengths = [len(x) for x in tok(prompts, add_special_tokens=False)["input_ids"]]
    pool = [[p, items[p][0]] for p, n in zip(prompts, lengths) if n <= 300]
    rng.shuffle(pool)
    eval_mc, train = pool[:400], pool[400:400 + TRAIN_POOL]
    print("MC pool", len(pool), "sources", counts)

    arc = []
    for row in datasets.load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test"):
        r = wi.arc_item(row)
        if r is not None:
            arc.append(list(r))
    tq = datasets.load_dataset("mandarjoshi/trivia_qa", "rc.nocontext", split="validation")
    order = list(range(len(tq)))
    random.Random(1).shuffle(order)
    trivia = []
    for i in order[:1000]:
        row = tq[i]
        aliases = list(row["answer"]["normalized_aliases"]) + list(row["answer"]["aliases"])
        trivia.append({"q": row["question"], "aliases": aliases})
    tt = datasets.load_dataset("mandarjoshi/trivia_qa", "rc.nocontext", split="train")
    shots = ""
    for i in random.Random(2).sample(range(len(tt)), 5):
        shots += "Question: " + tt[i]["question"] + "\nAnswer: " + tt[i]["answer"]["value"] + "\n\n"

    texts = {
        "kd": healdata.mix(["knowledge", "general", "prose"], 3000, seed=3),
        "align": healdata.mix(["prose", "knowledge"], 500, seed=0),
        "align_eval": healdata.mix(["prose", "knowledge"], 20, seed=0, skip=1000),
    }
    out = {"train": train, "eval_mc": eval_mc, "arc": arc, "trivia": trivia, "shots": shots, "texts": texts,
           "counts": counts}
    os.makedirs(ROOT, exist_ok=True)
    with open(ROOT + "/data.json.tmp", "w") as f:
        json.dump(out, f)
    os.replace(ROOT + "/data.json.tmp", ROOT + "/data.json")
    volume.commit()
    summary = {"train": len(train), "eval_mc": len(eval_mc), "arc": len(arc), "trivia": len(trivia),
               "kd_texts": len(texts["kd"]), "align": len(texts["align"]), "counts": counts,
               "seconds": round(time.time() - start, 1)}
    print("PREP", json.dumps(summary))
    return summary


def load_data():
    try:
        volume.reload()
    except Exception as e:
        print("reload failed", e)
    with open(ROOT + "/data.json") as f:
        return json.load(f)


@app.function(image=image, volumes={"/vol": volume}, gpu="H100", memory=131072, timeout=4 * 3600)
def soft(smoke=False):
    """Donor 14B answer-letter logits on every MC train and eval item."""
    import torch
    import transformers

    from wsp import wi

    start = time.time()
    data = load_data()
    tok = transformers.AutoTokenizer.from_pretrained(DONOR)
    donor = transformers.AutoModelForCausalLM.from_pretrained(DONOR, dtype=torch.bfloat16).to("cuda").eval()
    ids = wi.letter_token_ids(tok)[:4]
    items = data["train"] + data["eval_mc"]
    if smoke:
        items = items[:64]
    prompts = [p for p, _ in items]
    logits = wi.letter_logits(donor, tok, prompts, ids, batch_size=32)
    gold = torch.tensor([g for _, g in items])
    acc = float((logits.argmax(-1) == gold).float().mean())
    print("SOFT donor letter accuracy on pool", acc)
    if not smoke:
        torch.save(logits, ROOT + "/soft.pt.tmp")
        os.replace(ROOT + "/soft.pt.tmp", ROOT + "/soft.pt")
        volume.commit()
    return {"items": len(items), "donor_acc": acc, "seconds": round(time.time() - start, 1)}


# ------------------------------------------------------------------ evaluation


def evaluate(name, model, tok, data, full, smoke, extra=None):
    """Score a model (plain or grafted). Prints a SCORE line per metric with the cell name."""
    from modal_round6 import score_mmlu
    from modal_round7 import mmlu_split
    from wsp import skills_eval, wi

    row = {}

    def put(key, value):
        row[key] = value
        print(f"SCORE {name} {key}={value}", flush=True)

    model.eval()
    ids = wi.letter_token_ids(tok)
    _, dev = mmlu_split(tok, heldout=False)
    _, held = mmlu_split(tok, heldout=True)
    mc = [tuple(x) for x in data["eval_mc"]]
    if smoke:
        dev, held, mc = dev[:16], held[:16], mc[:16]
    put("mmlu_dev", score_mmlu(model, tok, dev, batch_size=16))
    put("mc_eval", wi.score_letters(model, tok, [(p, g, 4) for p, g in mc], ids, 16))
    if not full:
        return row
    put("mmlu_heldout", score_mmlu(model, tok, held, batch_size=16))
    arc = [tuple(x) for x in data["arc"]]
    trivia = data["trivia"]
    gsm = skills_eval.gsm8k_items(300)
    if smoke:
        arc, trivia, gsm = arc[:16], trivia[:16], gsm[:8]
    put("arc_c_test", wi.score_letters(model, tok, arc, ids, 16))
    prompts = [data["shots"] + "Question: " + t["q"] + "\nAnswer:" for t in trivia]
    outs = skills_eval.generate_texts(model, tok, prompts, max_new_tokens=16, batch_size=32, stop_strings=["\n"])
    hits = [wi.trivia_em(o, t["aliases"]) for o, t in zip(outs, trivia)]
    put("trivia_em", sum(hits) / len(hits))
    row["trivia_samples"] = [[t["q"], o, h] for t, o, h in list(zip(trivia, outs, hits))[:5]]
    print("TRIVIA raw", name, row["trivia_samples"], flush=True)
    result = skills_eval.evaluate_skill(model, tok, "gsm8k", items=gsm, batch_size=32)
    put("gsm8k", result["score"])
    row["gsm8k_samples"] = [str(r.get("output", ""))[:300] for r in result["records"][:2]]
    print("GSM raw", name, row["gsm8k_samples"], flush=True)
    if extra is not None:
        extra(row, put, held, ids)
    return row


def save_row(name, row):
    os.makedirs(ROOT + "/rows", exist_ok=True)
    path = f"{ROOT}/rows/{name}.json"
    with open(path + ".tmp", "w") as f:
        json.dump(row, f)
    os.replace(path + ".tmp", path)
    volume.commit()


@app.function(image=image, volumes={"/vol": volume}, gpu="H100", memory=131072, timeout=4 * 3600)
def ref_cell(label, smoke=False):
    import torch
    import transformers

    from wsp import skills_eval

    start = time.time()
    model_id = {"base": BASE, "donor": DONOR}.get(label) or SIBLINGS[label]
    data = load_data()
    tok = transformers.AutoTokenizer.from_pretrained(model_id)
    model = transformers.AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16).to("cuda").eval()
    for p in model.parameters():
        p.requires_grad_(False)
    name = "ref_" + label + ("_smoke" if smoke else "")
    row = evaluate(name, model, tok, data, True, smoke)
    total, run = skills_eval.model_params(model)
    row.update({"name": name, "model": model_id, "params_total": total, "params_run": run, "trainable": 0,
                "train_tokens": 0, "train_gpu_hours": 0.0})
    row["decode_tok_s"] = skills_eval.decode_speed(model, tok, new_tokens=64, repeats=1)
    print(f"SCORE {name} decode_tok_s={row['decode_tok_s']}")
    row["gpu_hours"] = round((time.time() - start) / 3600, 3)
    save_row(name, row)
    return row


# ------------------------------------------------------------------ training


def make_graft(base, donor, tok, cfg, control, texts, seed):
    import copy

    from wsp import pgraft, scramble

    tail = None
    if cfg["logit"] and control == "scrambled":
        norm, head = copy.deepcopy(donor.model.norm), copy.deepcopy(donor.get_output_embeddings())
        scramble.scramble_module(norm, 2000 + seed, mode="permute")
        scramble.scramble_module(head, 3000 + seed, mode="permute")
        tail = (norm, head)
    graft = pgraft.build(base, donor, cfg["host"], cfg["first"], cfg["last"],
                         control=None if control == "scrambled" else control,
                         enc_depth=cfg["enc_depth"], dec_rank=cfg["dec_rank"], write_layer=cfg["write"],
                         logit=cfg["logit"], tail=tail)
    if control == "scrambled":
        report = scramble.scramble_module(graft.part, 1000 + seed, mode="spectrum")
        print("scrambled slice", {k: report[k] for k in ("tensors_changed", "rel_fro_error", "mean_abs_cosine", "seconds")})
    if cfg["logit"]:
        pgraft.attach_logit(base, graft)
    elif cfg["write"] is None or cfg["write"] == cfg["host"]:
        pgraft.attach(base, [graft])
    else:
        pgraft.attach_split(base, graft)
    if control != "identity":
        pgraft.set_sink(graft, donor, tok, cfg["first"], texts[:256])
    return graft


@app.function(image=image, volumes={"/vol": volume}, gpu="H100", memory=131072, timeout=6 * 3600)
def train_cell(name, arm, seed, over, smoke=False):
    import math

    import torch
    import torch.nn.functional as F
    import transformers

    from wsp import pgraft, skills_eval, wi
    from wsp.lora import apply_lora, default_alpha, merge_lora, rank_for_budget
    from wsp.stitch import autocast

    start = time.time()
    cfg = dict(DEFAULT, **over)
    if smoke:
        cfg.update(SMOKE)
        name += "_smoke"
    torch.manual_seed(seed)
    data = load_data()
    sl = torch.load(ROOT + "/soft.pt") if os.path.exists(ROOT + "/soft.pt") else None
    if sl is None and not smoke:
        raise RuntimeError("soft.pt missing: run --mode soft first")
    n_train = len(data["train"])
    if sl is None:
        sl = torch.zeros(n_train + 400, 4)
    soft_train = sl[:n_train]
    if smoke:
        data["train"] = data["train"][:512]
        soft_train = soft_train[:512] if len(soft_train) >= 512 else torch.zeros(512, 4)
    device = torch.device("cuda")
    tok = transformers.AutoTokenizer.from_pretrained(BASE)
    base_dtype = torch.float32 if arm == "fullft" else torch.bfloat16
    donor = transformers.AutoModelForCausalLM.from_pretrained(DONOR, dtype=torch.bfloat16).to(device).eval()
    for p in donor.parameters():
        p.requires_grad_(False)
    base = transformers.AutoModelForCausalLM.from_pretrained(BASE, dtype=base_dtype).to(device).eval()
    for p in base.parameters():
        p.requires_grad_(False)
    ids = wi.letter_token_ids(tok)
    ids4 = ids[:4]
    d_host, d_donor = pgraft.hidden_size(base), pgraft.hidden_size(donor)
    glue_budget = (pgraft.count(pgraft.Enc(d_host, d_donor, depth=cfg["enc_depth"]).parameters())
                   + (0 if cfg["logit"] else
                      pgraft.count(pgraft.Dec(d_donor, d_host, rank=cfg["dec_rank"]).parameters())) + d_host + 1)
    from wsp.lora import params_per_rank

    extra_lora = cfg["graft_lora_rank"] * params_per_rank(base)
    glue_budget += extra_lora
    row = {"name": name, "arm": arm, "seed": seed, "cfg": cfg, "glue_budget": glue_budget}

    graft = None
    lora_params = None
    if arm in ("lora", "lora128"):
        rank = rank_for_budget(base, glue_budget) if arm == "lora" else 128
        lora_params = apply_lora(base, rank, default_alpha(rank))
        for p in lora_params:
            p.requires_grad_(True)
        row["lora_rank"] = rank
        row["trainable"] = pgraft.count(lora_params)
    elif arm == "fullft":
        for p in base.parameters():
            p.requires_grad_(True)
        row["trainable"] = pgraft.count(base.parameters())
    else:
        control = {"graft": None, "identity": "identity", "scrambled": "scrambled"}[arm]
        graft = make_graft(base, donor, tok, cfg, control, data["texts"]["align"], seed)
        row["trainable"] = pgraft.count(pgraft.trainable([graft], enc=True, dec=not cfg["logit"]))
        if cfg["graft_lora_rank"] > 0:
            r = cfg["graft_lora_rank"]
            lora_params = apply_lora(base, r, default_alpha(r))
            for p in lora_params:
                p.requires_grad_(True)
            row["trainable"] += pgraft.count(lora_params)
        align_steps = cfg["align_steps"]
        if arm == "graft":
            acfg = dict(pgraft.DEFAULTS, seq_len=cfg["seq_len"], align_steps=align_steps)
            pgraft.align(base, tok, donor, tok, graft, data["texts"]["align"], acfg, seed=seed, log_every=100)
            row["alignment"] = pgraft.alignment_report(base, tok, donor, tok, graft, data["texts"]["align_eval"], acfg)
            print("ALIGN", name, row["alignment"], flush=True)
        if not cfg["logit"]:
            pgraft.calibrate_scale(base, tok, graft, data["texts"]["align"], dict(pgraft.DEFAULTS, seq_len=cfg["seq_len"]))
        if cfg["out_frac"] > 0 and not cfg["logit"]:
            wl = cfg["write"] if cfg["write"] is not None else cfg["host"]
            row["out_rms"] = pgraft.calibrate_out_scale(base, tok, graft, data["texts"]["align"], wl, cfg["out_frac"],
                                                        dict(pgraft.DEFAULTS, seq_len=cfg["seq_len"]))
            print("OUT_RMS", name, row["out_rms"], flush=True)
    acfg = dict(pgraft.DEFAULTS, seq_len=cfg["seq_len"])

    if arm in ("lora", "lora128"):
        groups = [{"params": lora_params, "lr": cfg["lr_lora" if arm == "lora" else "lr_lora128"]}]
    elif arm == "fullft":
        groups = [{"params": list(base.parameters()), "lr": cfg["lr_full"]}]
    else:
        enc_lr = cfg["lr_enc2"] if arm == "graft" else cfg["lr_glue"]
        groups = [{"params": list(graft.dec.parameters()), "lr": cfg["lr_glue"]},
                  {"params": list(graft.token_gate.parameters()), "lr": cfg["lr_gate"]},
                  {"params": list(graft.enc.parameters()), "lr": enc_lr}]
        if cfg["logit"]:
            groups = groups[1:]
        if lora_params is not None:
            groups.append({"params": lora_params, "lr": cfg["lr_lora"]})
        for g in groups:
            for p in g["params"]:
                p.requires_grad_(True)
    params = [p for g in groups for p in g["params"]]
    opt = torch.optim.AdamW(groups, weight_decay=0.0)
    # Total optimizer steps equal across arms: the graft spends align_steps on stage 1, the others add them here.
    steps = cfg["steps"] + (0 if arm == "graft" else cfg["align_steps"])
    ks = cfg["kd_stage"] if graft is not None else 0

    def lam(kind):
        def f(st):
            b = min(1.0, (st + 1) / 30) * 0.5 * (1 + math.cos(math.pi * st / steps))
            if st < ks:
                return 0.0 if kind in ("enc", "lora") else b * cfg["kd_lr_mult"]
            return b
        return f

    if graft is not None:
        kinds = (["gate", "enc"] if cfg["logit"] else ["dec", "gate", "enc"]) + (["lora"] if lora_params is not None else [])
    else:
        kinds = ["all"]
    sched = torch.optim.lr_scheduler.LambdaLR(opt, [lam(k) for k in kinds])

    mc_rng = random.Random(seed + 31)
    order = list(range(len(data["train"])))
    mc_rng.shuffle(order)
    cursor = [0]

    def next_idx(n):
        if cursor[0] + n > len(order):
            mc_rng.shuffle(order)
            cursor[0] = 0
        out = order[cursor[0]:cursor[0] + n]
        cursor[0] += n
        return out

    kd_feed = pgraft._batches(data["texts"]["kd"], cfg["text_batch"], seed + 21)
    align_feed = pgraft._batches(data["texts"]["align"], 4, seed + 11)
    tokens = {"mc": 0, "text": 0, "align": 0}
    log = []
    base.train()
    t_train = time.time()
    for step in range(steps):
        opt.zero_grad(set_to_none=True)
        info = {"step": step}
        mu = cfg["micro"]
        idx_all = next_idx(cfg["mc_batch"])
        ce_sum = kl_sum = acc_sum = 0.0
        in_kd = step < ks
        for part_idx in ([] if in_kd else [idx_all[j::mu] for j in range(mu)]):
            w = len(part_idx) / len(idx_all)
            tok.padding_side = "left"
            enc = tok([data["train"][i][0] for i in part_idx], return_tensors="pt", padding=True).to(device)
            gold = torch.tensor([data["train"][i][1] for i in part_idx], device=device)
            with autocast(device):
                out = base(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"], use_cache=False,
                           logits_to_keep=1).logits
            letters = out[:, -1, :].float()[:, ids4]
            ce = F.cross_entropy(letters, gold)
            kl = wi.soft_letter_loss(letters, soft_train[part_idx].to(device))
            (w * (ce + cfg["w_soft"] * kl)).backward()
            tokens["mc"] += int(enc["attention_mask"].sum())
            ce_sum += w * float(ce)
            kl_sum += w * float(kl)
            acc_sum += w * float((letters.argmax(-1) == gold).float().mean())
            if graft is not None and "gate" not in info:
                g = pgraft.mean_gate([graft])
                if g is not None:
                    info["gate"] = float(g.detach())
        ce, kl = ce_sum, kl_sum
        info.update(ce=ce_sum, soft=kl_sum, acc=acc_sum)
        # full-logit distillation on text
        texts_all = next(kd_feed)
        kd_sum = 0.0
        for texts in [texts_all[j::mu] for j in range(mu)]:
            w = len(texts) / len(texts_all)
            tenc = pgraft._tok_batch(tok, texts, cfg["seq_len"], device)
            with torch.no_grad():
                with autocast(device):
                    tl = donor(input_ids=tenc["input_ids"], attention_mask=tenc["attention_mask"],
                               use_cache=False).logits
            with autocast(device):
                sl_text = base(input_ids=tenc["input_ids"], attention_mask=tenc["attention_mask"],
                               use_cache=False).logits
            kd = wi.kd_logits_loss(sl_text[:, :-1, :VOCAB], tl[:, :-1, :VOCAB], tenc["attention_mask"][:, 1:])
            (w * cfg["w_text"] * kd).backward()
            del tl, sl_text
            tokens["text"] += int(tenc["attention_mask"].sum())
            kd_sum += w * float(kd)
        kd = kd_sum
        info["kd"] = kd_sum
        if arm == "graft" and step % 2 == 0:
            aloss, _ = pgraft.functional_loss(base, tok, donor, tok, graft, next(align_feed), acfg)
            if aloss is not None:
                (cfg["w_align"] * aloss).backward()
                info["align"] = float(aloss)
                tokens["align"] += 4 * cfg["seq_len"]
        total = float(ce) + float(kd)
        if not math.isfinite(total):
            raise RuntimeError(f"{name}: non-finite loss at step {step}")
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        sched.step()
        log.append(info)
        if step % 100 == 0 or step == steps - 1:
            print(f"TRAIN {name} " + " ".join(f"{k} {v:.4f}" if isinstance(v, float) else f"{k} {v}"
                                              for k, v in info.items()), flush=True)
    base.eval()
    tok.padding_side = "left"
    row["train_seconds"] = round(time.time() - t_train, 1)
    for key in ("ce", "soft", "kd"):
        row[key + "_first50"] = sum(r[key] for r in log[:50]) / len(log[:50])
        row[key + "_last50"] = sum(r[key] for r in log[-50:]) / len(log[-50:])
    gates = [[r["step"], round(r["gate"], 4)] for r in log if "gate" in r and r["step"] % 100 == 0]
    if gates:
        row["gate_curve"] = gates
    row["train_tokens"] = sum(tokens.values())
    row["token_breakdown"] = tokens
    row["train_gpu_hours"] = round(row["train_seconds"] / 3600, 3)
    del opt
    torch.cuda.empty_cache()

    if arm == "fullft":
        base.to(torch.bfloat16)
    if arm == "graft":
        row["alignment_end"] = pgraft.alignment_report(base, tok, donor, tok, graft, data["texts"]["align_eval"], acfg)
        print("ALIGN_END", name, row["alignment_end"], flush=True)

    def kd_heldout(texts):
        tot = 0.0
        n = 0
        with torch.no_grad():
            for st in range(0, len(texts), 4):
                tenc = pgraft._tok_batch(tok, texts[st:st + 4], cfg["seq_len"], device)
                with autocast(device):
                    tl = donor(input_ids=tenc["input_ids"], attention_mask=tenc["attention_mask"], use_cache=False).logits
                    sl_ = base(input_ids=tenc["input_ids"], attention_mask=tenc["attention_mask"], use_cache=False).logits
                tot += float(wi.kd_logits_loss(sl_[:, :-1, :VOCAB], tl[:, :-1, :VOCAB], tenc["attention_mask"][:, 1:]))
                n += 1
        return tot / max(1, n)

    row["kd_heldout"] = kd_heldout(data["texts"]["align_eval"])
    print(f"SCORE {name} kd_heldout={row['kd_heldout']}", flush=True)
    if graft is not None:
        with pgraft.grafts_off(base):
            row["kd_heldout_graft_off"] = kd_heldout(data["texts"]["align_eval"])
        print(f"SCORE {name} kd_heldout_graft_off={row['kd_heldout_graft_off']}", flush=True)

    def extra(r, put, held, ids_):
        if graft is not None:
            from modal_round6 import score_mmlu

            with pgraft.grafts_off(base):
                put("mmlu_heldout_graft_off", score_mmlu(base, tok, held, batch_size=16))

    row.update(evaluate(name, base, tok, data, cfg["full_eval"], smoke, extra=extra if graft is not None else None))
    total, run = skills_eval.model_params(base)
    if graft is not None:
        slice_params = pgraft.count(graft.part.parameters()) if arm != "identity" else 0
        if cfg["logit"]:
            slice_params += pgraft.count(graft._tail[0].parameters()) + pgraft.count(graft._tail[1].parameters())
        glue = pgraft.count(pgraft.trainable([graft], dec=not cfg["logit"]))
        total += slice_params + glue
        run += slice_params + glue
        row["slice_params"] = slice_params
    row["params_total"], row["params_run"] = total, run
    if cfg["full_eval"]:
        try:
            if lora_params is not None:
                merge_lora(base)
            row["decode_tok_s"] = skills_eval.decode_speed(base, tok, new_tokens=64, repeats=1)
            print(f"SCORE {name} decode_tok_s={row['decode_tok_s']}", flush=True)
        except Exception as e:
            row["decode_error"] = repr(e)[:300]
    row["gpu_hours"] = round((time.time() - start) / 3600, 3)
    save_row(name, row)
    return row


@app.local_entrypoint()
def main(mode: str = "run", cells: str = "", smoke: bool = False):
    if mode == "prep":
        print(prep.remote())
        return
    if mode == "soft":
        print(soft.remote(smoke))
        return
    names = [c for c in cells.split(",") if c]
    if mode == "ref":
        calls = [(n, smoke) for n in names]
        for r in ref_cell.starmap(calls, return_exceptions=True):
            print("ROW", json.dumps(r, default=str)[:3000])
        return
    table = presets()
    calls = []
    for n in names:
        arm, seed, over = table[n]
        calls.append((n, arm, seed, over, smoke))
    if not calls:
        raise SystemExit("no cells")
    for r in train_cell.starmap(calls, return_exceptions=True):
        print("ROW", json.dumps(r, default=str)[:3000])
