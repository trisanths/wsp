"""Write-in track clinic: medical knowledge from donors of other families into Llama-3.2-1B-Instruct, one network.

Design and results: docs/write_in_clinic.md. Built on the xfam recipe (docs/write_in_xfam.md): frozen donor slice as a
parallel branch (wsp/pgraft.py), Enc depth 2, Dec rank 256, per-token gate, joint LoRA 16, 600 align steps per graft,
4000 task steps on answer letters plus the donor soft letters. New here: a medical pool, a second donor, two graft
points in one network (Qwen2.5-7B-Instruct at host layer 7, the medical donor at host layer 10, as in Arjun's h5),
and medical held-out scores.

    modal run modal_wi_clinic.py --pick true                 # score medical-donor candidates on clean validation slices
    modal run modal_wi_clinic.py --prep-data true            # evals, pool, donor soft letters, donor-alone scores
    modal run --detach modal_wi_clinic.py --cells graft_both:0 --smoke true
    modal run --detach modal_wi_clinic.py --cells graft_both:0,graft_both:1
"""

import json
import os
import random
import time

import modal

app = modal.App("wsp-wi-clinic")
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
    .add_local_file("modal_round22pg.py", "/root/modal_round22pg.py")
    .add_local_file("modal_round23pg.py", "/root/modal_round23pg.py")
    .add_local_file("modal_wi_xfam.py", "/root/modal_wi_xfam.py")
)

VERSION = "v1"
DIR = f"/vol/wi_clinic/{VERSION}"
ROWS = f"{DIR}/rows"
XFAM_DIR = "/vol/wi_xfam/v1"
HOST = "unsloth/Llama-3.2-1B-Instruct"
NATIVE_3B = "unsloth/Llama-3.2-3B-Instruct"
MED_ID = "FreedomIntelligence/Apollo2-7B"
DONORS = {"qwen": "Qwen/Qwen2.5-7B-Instruct", "med": MED_ID}
MEDICAL_SUBJECTS = ["clinical_knowledge", "college_medicine", "professional_medicine", "anatomy", "medical_genetics"]
PICK_CANDIDATES = [
    "FreedomIntelligence/Apollo2-7B",
    "m42-health/Llama3-Med42-8B",
    "aaditya/Llama3-OpenBioLLM-8B",
    "FreedomIntelligence/HuatuoGPT-o1-8B",
    "Intelligent-Internet/II-Medical-8B",
    "Qwen/Qwen2.5-7B-Instruct",
]
N_MEDMCQA, N_MEDQA = 20000, 11000
GENERAL_KEEP = 20000

BASE_CFG = {
    "arm": "graft", "seed": 0, "steps": 4000, "batch": 16, "align_steps": 600, "kd_w": 1.0, "preserve": 1.0,
    "align_reg": 1.0, "enc_depth": 2, "dec_rank": 256, "lora_rank": 16, "lora_lr": 2e-4,
    "glue_lr": 3e-4, "gate_lr": 1e-4, "enc_lr": 1e-4, "ft_lr": 3e-6, "match_extra": True,
    "points": [], "kd": "mean", "decode": True, "tag": "", "model": HOST,
}
Q_POINT = {"donor": "qwen", "host_layer": 7, "first": 4, "last": 20}
M_POINT = {"donor": "med", "host_layer": 10, "first": 4, "last": 20}
M_POINT_7 = dict(M_POINT, host_layer=7)
PRESETS = {
    "base": {"arm": "reference"},
    "native3b": {"arm": "reference", "model": NATIVE_3B},
    "graft_med": {"points": [M_POINT_7], "kd": "med"},
    "graft_qwen": {"points": [dict(Q_POINT)], "kd": "qwen"},
    "graft_both": {"points": [Q_POINT, M_POINT]},
    "identity_both": {"arm": "identity", "points": [Q_POINT, M_POINT]},
    "scrambled_both": {"arm": "scrambled", "points": [Q_POINT, M_POINT]},
    "lora_match": {"arm": "lora_match", "points": [Q_POINT, M_POINT], "match_lora": 16},
    "lora_big": {"arm": "lora_big", "lora_rank": 128, "lora_lr": 5e-5},
    "fullft": {"arm": "fullft"},
}


# ------------------------------------------------------------------ data


def build_evals():
    """Held-out test sets as [prompt, gold] lists. Nothing trains on any of them."""
    from datasets import load_dataset

    from wsp import clinic, skills_eval as ev

    out = {}
    out["medmcqa"] = [[x["prompt"], x["gold"]] for x in ev.medmcqa_items()]
    out["mmlu_med"] = [[x["prompt"], x["gold"]] for x in ev.mmlu_subject_items(MEDICAL_SUBJECTS)]
    out["mmlu_bio"] = [[x["prompt"], x["gold"]] for x in ev.mmlu_subject_items(["college_biology"])]
    test = load_dataset("GBaker/MedQA-USMLE-4-options", split="test")
    items = [clinic.medqa_item(r) for r in test]
    out["medqa"] = [list(x) for x in items if x]
    with open(f"{XFAM_DIR}/evals.json") as f:
        out["mmlu_held"] = json.load(f)["mmlu_held"]
    return out


def medical_train_items(banned):
    """MedMCQA train, MedQA train, PubMedQA labeled, as [prompt, gold], without any eval prompt."""
    from datasets import load_dataset

    from wsp import clinic

    rng = random.Random(11)
    seen = set()
    parts = {}
    rows = load_dataset("openlifescienceai/medmcqa", split="train")
    order = list(range(len(rows)))
    rng.shuffle(order)
    items = []
    for i in order:
        it = clinic.medmcqa_item(rows[i])
        if it:
            items.append(it)
        if len(items) >= N_MEDMCQA * 2:
            break
    parts["medmcqa"] = clinic.filter_items(items, banned, seen)[:N_MEDMCQA]
    rows = load_dataset("GBaker/MedQA-USMLE-4-options", split="train")
    items = [clinic.medqa_item(r) for r in rows]
    parts["medqa"] = clinic.filter_items([x for x in items if x], banned, seen)[:N_MEDQA]
    rows = load_dataset("qiaojin/PubMedQA", "pqa_labeled", split="train")
    items = [clinic.pubmedqa_item(r) for r in rows]
    parts["pubmedqa"] = clinic.filter_items([x for x in items if x], banned, seen)
    return parts, seen


def build_pool(evals):
    from wsp import clinic

    banned = {p for k in evals for p, _ in evals[k]}
    parts, seen = medical_train_items(banned)
    with open(f"{XFAM_DIR}/pool.json") as f:
        general = json.load(f)["items"]
    random.Random(3).shuffle(general)
    parts["general"] = clinic.filter_items(general, banned, seen)[:GENERAL_KEEP]
    items = [it for k in ("medmcqa", "medqa", "pubmedqa", "general") for it in parts[k]]
    random.Random(7).shuffle(items)
    return items, {k: len(v) for k, v in parts.items()}


def accuracy(probs, items):
    return sum(int(max(range(4), key=lambda j: row[j]) == g) for row, (_, g) in zip(probs, items)) / max(1, len(items))


@app.function(image=image, volumes={"/vol": volume}, gpu="H100", memory=131072, timeout=3 * 3600)
def pick():
    """Score donor candidates on clean slices that are neither scored nor trained on (see below)."""
    import gc

    import torch
    import transformers
    from datasets import load_dataset

    import modal_wi_xfam as X
    from wsp import clinic

    from wsp import skills_eval as ev
    rng = random.Random(21)
    # Clean slices only: MedMCQA validation items outside the 1000 scored, and the MMLU validation split of the
    # medical subjects (the scored items are the MMLU test split). Train splits are contaminated for some candidates.
    rows = load_dataset("openlifescienceai/medmcqa", split="validation")
    scored = {x["id"] for x in ev.medmcqa_items()}
    rest = [i for i in range(len(rows)) if f"medmcqa:{i}" not in scored]
    rng.shuffle(rest)
    mm = [it for it in (clinic.medmcqa_item(rows[i]) for i in rest[:1500]) if it][:1000]
    val = load_dataset("cais/mmlu", "all", split="validation")
    mq = [(clinic.mcq_prompt(r["question"], r["choices"]), int(r["answer"])) for r in val if r["subject"] in MEDICAL_SUBJECTS + ["college_biology"]]
    aux = load_dataset("cais/mmlu", "all", split="validation")
    gen = [(clinic.mcq_prompt(r["question"], r["choices"]), int(r["answer"])) for r in aux if r["subject"] not in MEDICAL_SUBJECTS + ["college_biology"]]
    rng.shuffle(gen)
    gen = gen[:500]
    sets = {"medmcqa_val_rest": mm, "mmlu_med_validation": mq, "mmlu_general_validation": gen}
    os.makedirs(f"{DIR}/pick", exist_ok=True)
    out = {}
    for mid in PICK_CANDIDATES:
        t0 = time.time()
        try:
            tok = transformers.AutoTokenizer.from_pretrained(mid)
            model = transformers.AutoModelForCausalLM.from_pretrained(mid, dtype=torch.bfloat16).to("cuda").eval()
            res = {}
            for k, v in sets.items():
                res[k] = accuracy(X.letter_probs(model, tok, [list(x) for x in v]), v)
            res["layers"] = model.config.num_hidden_layers
            res["hidden"] = model.config.hidden_size
            res["seconds"] = round(time.time() - t0)
            del model
            gc.collect()
            torch.cuda.empty_cache()
        except Exception as e:
            res = {"error": repr(e)[:300]}
        out[mid] = res
        print("PICK", mid, json.dumps(res), flush=True)
        with open(f"{DIR}/pick/pick.json", "w") as f:
            json.dump(out, f, indent=1)
        volume.commit()
    return out


@app.function(image=image, volumes={"/vol": volume}, gpu="H100", memory=131072, timeout=4 * 3600)
def prep():
    import gc

    import torch
    import transformers

    import modal_wi_xfam as X

    t0 = time.time()
    os.makedirs(DIR, exist_ok=True)
    try:
        volume.reload()
    except Exception as e:
        print("reload skipped", repr(e)[:80])
    evals = build_evals()
    print("evals", {k: len(v) for k, v in evals.items()})
    items, counts = build_pool(evals)
    print("pool", len(items), counts)
    softs, donor_acc = {}, {}
    for name, mid in DONORS.items():
        tok = transformers.AutoTokenizer.from_pretrained(mid)
        model = transformers.AutoModelForCausalLM.from_pretrained(mid, dtype=torch.bfloat16).to("cuda").eval()
        softs[name] = X.letter_probs(model, tok, items)
        donor_acc[name] = {"train_pool": accuracy(softs[name], items)}
        for k, v in evals.items():
            donor_acc[name][k] = accuracy(X.letter_probs(model, tok, v), v)
        print("DONOR", name, mid, json.dumps(donor_acc[name]), flush=True)
        del model
        gc.collect()
        torch.cuda.empty_cache()
    with open(f"{DIR}/evals.json", "w") as f:
        json.dump(evals, f)
    with open(f"{DIR}/pool.json", "w") as f:
        json.dump({"items": items, "soft": softs, "counts": counts, "donors": DONORS}, f)
    with open(f"{DIR}/donor_acc.json", "w") as f:
        json.dump(donor_acc, f)
    volume.commit()
    return {"pool": len(items), "counts": counts, "donor_acc": donor_acc, "seconds": round(time.time() - t0)}


# ------------------------------------------------------------------ cells


def finish(row, t0, smoke, cname):
    row["gpu_hours"] = round((time.time() - t0) / 3600, 3)
    row["seconds"] = round(time.time() - t0, 1)
    if not smoke:
        os.makedirs(ROWS, exist_ok=True)
        with open(f"{ROWS}/{row['name']}_{row['seed']}{row['config'].get('tag', '')}.json", "w") as f:
            json.dump(row, f)
        volume.commit()
    print(cname, "ROW", json.dumps(row))
    return row


def donor_dim(name):
    import transformers
    cfg = transformers.AutoConfig.from_pretrained(DONORS[name])
    cfg = getattr(cfg, "text_config", None) or cfg
    return cfg.hidden_size


def budget(c, base_config):
    """Trainable count of the real graft set with this config: the matched-LoRA budget."""
    from wsp import pgraft
    from wsp.lora import lora_param_count
    total = 0
    for pt in c["points"]:
        d = donor_dim(pt["donor"])
        total += pgraft.count(pgraft.Enc(2048, d, depth=c["enc_depth"]).parameters())
        total += pgraft.count(pgraft.Dec(d, 2048, rank=c["dec_rank"]).parameters()) + 2048 + 1
    if c.get("match_lora", 0):
        total += lora_param_count(base_config, c["match_lora"])
    return total


def _cell(name, seed, smoke=False, overrides=None):
    import copy
    import gc

    import torch
    import transformers

    import modal_round22pg as r22
    import modal_wi_xfam as X
    from wsp import clinic, pgraft, scramble
    from wsp.lora import apply_lora, default_alpha, rank_for_budget

    cname = f"{name}:{seed}" + (":smoke" if smoke else "")
    t0 = time.time()
    c = {**BASE_CFG, **PRESETS[name], **(overrides or {})}
    c["seed"] = seed
    arm = c["arm"]
    torch.manual_seed(seed)
    try:
        volume.reload()
    except Exception as e:
        print(cname, "reload skipped", repr(e)[:80])
    with open(f"{DIR}/pool.json") as f:
        pool = json.load(f)
    with open(f"{DIR}/evals.json") as f:
        evals = json.load(f)
    items = pool["items"]
    soft = clinic.mix_soft(pool["soft"], c["kd"])
    if smoke:
        c["steps"], c["align_steps"] = 3, 3
        items, soft = items[:64], soft[:64]
        evals = {k: v[:32] for k, v in evals.items()}
    data = r22.build_data()
    if smoke:
        data = {k: v[:16] for k, v in data.items()}
    cfg = dict(pgraft.DEFAULTS, **r22.CFG)
    row = {"cell": cname, "name": name, "seed": seed, "config": c, "pool": len(items), "version": VERSION}
    device = torch.device("cuda")
    model_id = c["model"]
    tok = transformers.AutoTokenizer.from_pretrained(model_id)
    letters = X.letters_of(tok)
    dtype = torch.float32 if arm == "fullft" else torch.bfloat16
    base = transformers.AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype).to("cuda").eval()
    for p in base.parameters():
        p.requires_grad_(arm == "fullft")
    row["params_base"] = sum(p.numel() for p in base.parameters())

    if arm == "reference":
        row["scores"] = {k: X.score(base, tok, v) for k, v in evals.items()}
        row["decode_tps"] = X.decode_tps(base, tok)
        row["params_per_token"] = row["params_total"] = row["params_base"]
        print(cname, "SCORES", json.dumps(row["scores"]))
        return finish(row, t0, smoke, cname)

    ref = transformers.AutoModelForCausalLM.from_pretrained(HOST, dtype=torch.bfloat16).to("cuda").eval()
    for p in ref.parameters():
        p.requires_grad_(False)

    points = c["points"]
    grafts, lora_params, donors, donor_toks = [], [], {}, {}
    needs_donor = arm in ("graft", "scrambled")
    if needs_donor:
        for pt in points:
            n = pt["donor"]
            if n not in donors:
                donor_toks[n] = transformers.AutoTokenizer.from_pretrained(DONORS[n])
                donors[n] = transformers.AutoModelForCausalLM.from_pretrained(DONORS[n], dtype=torch.bfloat16).to("cuda").eval()
                for p in donors[n].parameters():
                    p.requires_grad_(False)

    joint_lora = c["lora_rank"] if arm in ("graft", "identity", "scrambled") else 0
    if joint_lora:
        lora_params = apply_lora(base, joint_lora, default_alpha(joint_lora))
        for p in lora_params:
            p.requires_grad_(True)

    if arm in ("graft", "identity", "scrambled"):
        control = "identity" if arm == "identity" else None
        for pt in points:
            d = donor_dim(pt["donor"])
            g = pgraft.build(base, donors.get(pt["donor"]) if control is None else None, pt["host_layer"],
                             pt["first"], pt["last"], control=control, enc_depth=c["enc_depth"],
                             dec_rank=c["dec_rank"], d_donor=d)
            g.donor_name = pt["donor"]
            grafts.append(g)
        pgraft.attach(base, grafts)
        if needs_donor:
            for g in grafts:
                pgraft.set_sink(g, donors[g.donor_name], donor_toks[g.donor_name], g.donor_span[0], data["align"][:256])
        glue = pgraft.trainable(grafts, enc=True)
        row["trainable_glue"] = pgraft.count(glue)
        row["slice_params"] = sum(sum(p.numel() for p in g.part.parameters()) for g in grafts if g.control is None)
        if needs_donor:
            for g in grafts:
                pgraft.align(base, tok, donors[g.donor_name], donor_toks[g.donor_name], g, data["align"],
                             dict(cfg, align_steps=c["align_steps"]), seed=seed, log_every=100)
                row.setdefault("alignment", []).append(
                    pgraft.alignment_report(base, tok, donors[g.donor_name], donor_toks[g.donor_name], g,
                                            data["align_eval"], cfg))
                print(cname, "alignment", g.donor_name, row["alignment"][-1])
        for g in grafts:
            row.setdefault("calib_rms", []).append(pgraft.calibrate_scale(base, tok, g, data["align"], cfg))
        if arm == "scrambled":
            for i, g in enumerate(grafts):
                g.part = copy.deepcopy(g.part)
                rep = scramble.scramble_module(g.part, seed=seed + 100 + i, mode="spectrum")
                row.setdefault("scramble_cos", []).append(rep.get("mean_abs_cosine"))
            donors_alive = dict(donors)
            donors.clear()
            donors_alive.clear()
            gc.collect()
            torch.cuda.empty_cache()
        trainable_total = pgraft.count(glue) + pgraft.count(lora_params)
    elif arm in ("lora_match", "lora_big"):
        if arm == "lora_match":
            rank = rank_for_budget(base, budget(c, base.config))
        else:
            rank = c["lora_rank"]
        lora_params = apply_lora(base, rank, default_alpha(rank))
        for p in lora_params:
            p.requires_grad_(True)
        row["lora_rank"] = rank
        trainable_total = pgraft.count(lora_params)
    else:  # fullft
        trainable_total = sum(p.numel() for p in base.parameters())
    row["trainable"] = trainable_total

    if arm in ("graft", "identity", "scrambled"):
        gate = [p for g in grafts for p in g.token_gate.parameters()]
        dec = [p for g in grafts for p in g.dec.parameters()]
        enc = [p for g in grafts for p in g.enc.parameters()]
        groups = [{"params": dec + (enc if arm == "identity" else []), "lr": c["glue_lr"]},
                  {"params": gate, "lr": c["gate_lr"]}]
        if arm != "identity":
            groups.append({"params": enc, "lr": c["enc_lr"]})
        if lora_params:
            groups.append({"params": lora_params, "lr": c["lora_lr"]})
    elif arm == "fullft":
        groups = [{"params": list(base.parameters()), "lr": c["ft_lr"]}]
    else:
        groups = [{"params": lora_params, "lr": c["lora_lr"]}]

    steps = c["steps"]
    if arm != "graft" and c["match_extra"]:
        steps += c["align_steps"] * max(1, len(points))  # controls spend the stage-1 steps on the task
    row["train_steps"] = steps

    rng = random.Random(seed + 5)
    order = list(range(len(items)))
    rng.shuffle(order)
    ptr = [0]

    def next_idx():
        if ptr[0] + c["batch"] > len(order):
            rng.shuffle(order)
            ptr[0] = 0
        idx = order[ptr[0]:ptr[0] + c["batch"]]
        ptr[0] += c["batch"]
        return idx

    gen_feed = pgraft._batches(data["general"], 4, seed + 41)
    align_feeds = [pgraft._batches(data["align"], cfg["align_batch"], seed + 11 + i) for i in range(len(grafts))] \
        if arm == "graft" else []
    tokens = [0]

    def step_fn(step):
        idx = next_idx()
        batch = [tuple(items[i]) for i in idx]
        sp = torch.tensor([soft[i] for i in idx])
        loss, info = pgraft.mc_soft_loss(base, tok, batch, letters, device, soft=sp, kd_weight=c["kd_w"])
        tokens[0] += info.pop("tokens")
        if c["preserve"] > 0:
            enc_ = pgraft._tok_batch(tok, next(gen_feed), cfg["seq_len"], device)
            _, logits = pgraft.seq_lm_loss(base, enc_, return_logits=True)
            drift = pgraft.drift_kl(ref, enc_, logits)
            loss = loss + c["preserve"] * drift
            info["drift"] = float(drift.detach())
        if arm == "graft" and c["align_reg"] > 0 and step % 2 == 0:
            for g, feed in zip(grafts, align_feeds):
                rl, _ = pgraft.functional_loss(base, tok, donors[g.donor_name], donor_toks[g.donor_name], g,
                                               next(feed), cfg)
                loss = loss + c["align_reg"] * rl / len(grafts)
        if grafts:
            gm = pgraft.mean_gate(grafts)
            if gm is not None:
                info["gate"] = float(gm.detach())
        return loss, info

    log = pgraft.train(base, groups, steps, step_fn, seed=seed + 1, log_every=100)
    row["tokens_trained"] = tokens[0]
    row["gate_curve"] = [[r["step"], round(r["gate"], 4)] for r in log if "gate" in r and r["step"] % 100 == 0]
    row["loss_first50"] = sum(r["loss"] for r in log[:50]) / max(1, len(log[:50]))
    row["loss_last50"] = sum(r["loss"] for r in log[-50:]) / max(1, len(log[-50:]))
    print(cname, "train done", "loss", round(row["loss_first50"], 4), "->", round(row["loss_last50"], 4))
    row["train_seconds"] = round(time.time() - t0, 1)

    if arm == "fullft":
        base = base.to(torch.bfloat16)
    base.eval()
    row["scores"] = {k: X.score(base, tok, v) for k, v in evals.items()}
    print(cname, "SCORES", json.dumps(row["scores"]))
    if grafts and arm != "identity":
        keys = [k for k in ("medmcqa", "mmlu_med", "mmlu_held") if k in evals]
        with pgraft.grafts_off(base):
            row["scores_graft_off"] = {k: X.score(base, tok, evals[k]) for k in keys}
        print(cname, "SCORES graft off", json.dumps(row["scores_graft_off"]))
        if len(grafts) > 1:
            row["scores_single_graft_on"] = []
            for i in range(len(grafts)):
                with clinic.only(grafts, i):
                    row["scores_single_graft_on"].append(
                        {"donor": grafts[i].donor_name, **{k: X.score(base, tok, evals[k]) for k in keys}})
            print(cname, "SCORES one graft on", json.dumps(row["scores_single_graft_on"]))
    per_token = row["params_base"]
    if grafts and arm != "identity":
        per_token += row["slice_params"] + row["trainable_glue"]
    elif grafts:
        per_token += row["trainable_glue"]
    row["params_per_token"] = row["params_total"] = per_token
    if grafts and not smoke:
        os.makedirs(f"{DIR}/ckpt", exist_ok=True)
        import torch as _t
        _t.save(clinic.graft_state(grafts, lora_params), f"{DIR}/ckpt/{name}_{seed}{c['tag']}.pt")
    if c["decode"] and not smoke:
        row["decode_tps"] = X.decode_tps(base, tok)
    return finish(row, t0, smoke, cname)


@app.function(image=image, volumes={"/vol": volume}, gpu="H100", memory=131072, timeout=5 * 3600)
def cell(name, seed, smoke=False, overrides=None):
    return _cell(name, seed, smoke, overrides)


@app.function(image=image, volumes={"/vol": volume}, gpu="H200", memory=131072, timeout=6 * 3600)
def cell_big(name, seed, smoke=False, overrides=None):
    return _cell(name, seed, smoke, overrides)


DEMO_PROMPTS = [
    "A 68-year-old on warfarin is started on trimethoprim-sulfamethoxazole for a urinary infection. What happens to the INR, why, and what should the clinic do?",
    "A 58-year-old has crushing chest pain for 40 minutes. The ECG shows ST elevation in leads II, III and aVF. Which artery is most likely occluded and what is the first-line management?",
    "An adult with type 2 diabetes has an eGFR of 40 and established cardiovascular disease. Which drug classes would you add to metformin, and which would you avoid or dose-reduce?",
    "How do diabetic ketoacidosis and hyperosmolar hyperglycemic state differ in presentation, labs and early treatment?",
    "A 3-year-old has high fever, drooling, muffled voice and sits leaning forward. What is the most likely diagnosis and what should you avoid doing in the clinic?",
    "A patient has wrist drop after a midshaft humerus fracture. Which nerve is injured and what other findings do you expect?",
]


def _generate(model, tok, prompts, new=220):
    import torch
    tok.padding_side = "left"
    outs = []
    for q in prompts:
        text = tok.apply_chat_template([{"role": "system", "content": "You are a careful assistant for a primary care clinic."},
                                        {"role": "user", "content": q}], tokenize=False, add_generation_prompt=True)
        enc = tok(text, return_tensors="pt", add_special_tokens=False).to("cuda")
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=new, do_sample=False)
        outs.append(tok.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True))
    return outs


@app.function(image=image, volumes={"/vol": volume}, gpu="H200", memory=131072, timeout=2 * 3600)
def demo(ckpt="graft_both_0", name="graft_both"):
    """Greedy side-by-side outputs: base, the trained two-donor graft, each donor alone. Raw outputs saved."""
    import gc

    import torch
    import transformers

    import modal_round22pg as r22
    from wsp import clinic, pgraft
    from wsp.lora import apply_lora, default_alpha

    volume.reload()
    c = {**BASE_CFG, **PRESETS[name]}
    data = r22.build_data()
    tok = transformers.AutoTokenizer.from_pretrained(HOST)
    res = {"prompts": DEMO_PROMPTS, "ckpt": ckpt}
    base = transformers.AutoModelForCausalLM.from_pretrained(HOST, dtype=torch.bfloat16).to("cuda").eval()
    res["base"] = _generate(base, tok, DEMO_PROMPTS)
    for p in base.parameters():
        p.requires_grad_(False)
    lora = apply_lora(base, c["lora_rank"], default_alpha(c["lora_rank"]))
    donors, dtoks, grafts = {}, {}, []
    for pt in c["points"]:
        n = pt["donor"]
        if n not in donors:
            dtoks[n] = transformers.AutoTokenizer.from_pretrained(DONORS[n])
            donors[n] = transformers.AutoModelForCausalLM.from_pretrained(DONORS[n], dtype=torch.bfloat16).to("cuda").eval()
        g = pgraft.build(base, donors[n], pt["host_layer"], pt["first"], pt["last"], enc_depth=c["enc_depth"],
                         dec_rank=c["dec_rank"], d_donor=donor_dim(n))
        g.donor_name = n
        grafts.append(g)
    pgraft.attach(base, grafts)
    for g in grafts:
        pgraft.set_sink(g, donors[g.donor_name], dtoks[g.donor_name], g.donor_span[0], data["align"][:256])
    state = torch.load(f"{DIR}/ckpt/{ckpt}.pt")
    clinic.load_graft_state(grafts, state, lora)
    base.eval()
    res["graft"] = _generate(base, tok, DEMO_PROMPTS)
    for n, m in donors.items():
        res["donor_" + n] = _generate(m, dtoks[n], DEMO_PROMPTS)
    os.makedirs(f"{DIR}/demo", exist_ok=True)
    with open(f"{DIR}/demo/{ckpt}.json", "w") as f:
        json.dump(res, f, indent=1)
    volume.commit()
    return res


@app.local_entrypoint()
def main(cells: str = "", smoke: bool = False, prep_data: bool = False, pick_donor: bool = False,
         demo_run: bool = False, overrides: str = ""):
    if pick_donor:
        print(json.dumps(pick.remote(), indent=1))
        return
    if demo_run:
        print(json.dumps(demo.remote(), indent=1))
        return
    if prep_data:
        print(json.dumps(prep.remote()))
        return
    calls = []
    for item in [x for x in cells.split(",") if x]:
        name, seed = item.split(":")
        if name not in PRESETS:
            raise SystemExit(f"unknown preset {name}")
        calls.append((name, int(seed), smoke, json.loads(overrides) if overrides else None))
    if not calls:
        raise SystemExit("0 cells: pass --cells name:seed,...")
    small = [x for x in calls if PRESETS[x[0]]["arm"] in ("lora_match", "lora_big", "fullft", "reference", "identity")]
    big = [x for x in calls if x not in small]
    handles = [cell.spawn(*x) for x in small] + [cell_big.spawn(*x) for x in big]
    rows = []
    for h in handles:
        try:
            rows.append(h.get())
        except Exception as e:
            print("cell failed:", repr(e)[:1000])
    print("done", len(rows), "rows")
