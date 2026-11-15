"""Hill-climb parallel grafts over parts, translators and domains (wsp/pgraft.py).

Every cell is one config: a domain (its donor, training text and eval sets),
a part (a donor block, or a bundle of donor subnetworks: library parts,
whole-layer MLPs or attention), an arm, and the translator shape. Base is
always unsloth/Llama-3.2-1B-Instruct, read and written after layer 7.

Arms, all at one trainable budget per config:
  real      the parts as named
  random    the same kinds, layers and sizes with random units (bundles only)
  identity  the same Enc, Dec and gate with no donor part
  lora      LoRA on the base, rank matched to the real arm's trainable count

Training: stage 1 aligns Enc (functional KL through the donor's head for a
block, part output matching for a bundle); stage 2 distills the donor into
the grafted base over the shared vocabulary (wsp.pgraft.kd_loss) on the
domain's healdata text plus the donor's skill-screen prompts (search split,
unlabeled: the donor's own predictions are the target). The identity and
LoRA arms spend the alignment steps in stage 2.

Scored without training on any of it: the distillation gap to the donor on
held-out domain text (graft on and off), the donor's library skill screens
(held-out split: continuation NLL, transfer = share of the base-to-donor gap
closed), MedMCQA for medicine and MMLU held-out for every domain, and KL to
the frozen base on legal text no arm trains on.

    modal run modal_hill.py --batch h1
"""

import json
import os
import random
import time

import modal

app = modal.App("wsp-hill")
volume = modal.Volume.from_name("wsp-models", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch==2.7.1", "transformers==4.57.1", "accelerate", "huggingface_hub", "requests",
        "safetensors", "zstandard", "numpy", "datasets", "sentencepiece", "protobuf",
    )
    .env({"HF_HOME": "/vol/hf", "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
    .add_local_python_source("wsp")
    .add_local_file("modal_graft.py", "/root/modal_graft.py")
    .add_local_file("modal_round5.py", "/root/modal_round5.py")
    .add_local_file("modal_round6.py", "/root/modal_round6.py")
    .add_local_file("modal_round7.py", "/root/modal_round7.py")
    .add_local_dir("library", "/root/library")
    .add_local_dir("person_a/mine/skills", "/root/skills")
)

HOST = "unsloth/Llama-3.2-1B-Instruct"
HOST_LAYER = 7
DOMAINS = {
    "medicine": {"donor": "FreedomIntelligence/Apollo2-7B", "skills": "apollo2_7b", "pools": ["medicine"]},
    "code": {"donor": "Qwen/Qwen2.5-Coder-7B-Instruct", "skills": "qwencoder7b", "pools": ["code"]},
    "math": {"donor": "Qwen/Qwen2.5-Math-1.5B", "skills": "qwenmath1.5b", "pools": ["math"]},
    "knowledge": {"donor": "Qwen/Qwen2.5-7B-Instruct", "skills": None, "pools": ["knowledge"],
                  "align_pools": ["prose", "knowledge", "medicine"]},
    "math7b": {"donor": "Qwen/Qwen2.5-Math-7B-Instruct", "skills": "qwenmath7b_inst", "pools": ["math"]},
    "knowledge14": {"donor": "Qwen/Qwen2.5-14B-Instruct", "skills": None, "pools": ["knowledge"],
                    "align_pools": ["prose", "knowledge", "medicine"]},
    "phi4": {"donor": "microsoft/phi-4", "skills": None, "pools": ["knowledge"],
             "align_pools": ["prose", "knowledge", "medicine"]},
    "qwen3_8b": {"donor": "Qwen/Qwen3-8B", "skills": None, "pools": ["knowledge"],
                 "align_pools": ["prose", "knowledge", "medicine"]},
    "knowledge32": {"donor": "Qwen/Qwen2.5-32B-Instruct", "skills": None, "pools": ["knowledge"],
                    "align_pools": ["prose", "knowledge", "medicine"]},
}
DEFAULT = {
    "host_layer": HOST_LAYER, "enc_depth": 3, "dec_rank": 256, "dec_depth": 1,
    "align_steps": 400, "kd_steps": 300, "task_steps": 600, "seed": 0, "objective": "kd_mc", "preserve": 1.0,
    "align_reg": 0.0, "mc_train": "general",
    "lr": {"align": 1e-3, "glue": 3e-4, "gate": 1e-4, "enc2": 1e-4, "lora": 2e-4},
}


# ------------------------------------------------------------------ data


def domain_data(domain):
    from wsp import healdata

    path = f"/vol/hill/data3_{domain}.json"
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    pools = DOMAINS[domain]["pools"]
    align_pools = DOMAINS[domain].get("align_pools", ["prose"] + pools)
    data = {
        "align": healdata.mix(align_pools, 1000, seed=0),
        "align_eval": healdata.mix(align_pools, 24, seed=0, skip=1000),
        "domain": healdata.mix(pools, 2000, seed=1, skip=1100),
        "domain_eval": healdata.mix(pools, 60, seed=1, skip=3200),
        "law": healdata.pool("law", 60, seed=5),
        "general": healdata.mix(["general", "prose"], 1500, seed=2, skip=1200),
    }
    os.makedirs("/vol/hill", exist_ok=True)
    tmp = f"{path}.{os.getpid()}.tmp"
    with open(tmp, "w") as f:
        json.dump(data, f)
    os.replace(tmp, path)
    volume.commit()
    return data


def skill_sets(key):
    """{name: {"search": [(context, continuation)], "heldout": [...]}} for the donor's skill screens."""
    out = {}
    if key is None:
        return out
    folder = os.path.join("/root/skills", key)
    for name in sorted(os.listdir(folder)):
        with open(os.path.join(folder, name)) as f:
            d = json.load(f)
        sets = {}
        for split in ("search", "heldout"):
            sets[split] = [(it["context"], it["continuation"]) for it in d["items"].get(split, [])]
        out[name[:-5]] = sets
    return out


def mc_items():
    """4-option items in MMLU prompt format from non-benchmark train splits (CommonsenseQA, OpenBookQA, QASC,
    LogiQA2.0), cached: {"train": [...], "eval": [...]}. Same construction as round 23."""
    from modal_round6 import mmlu_prompt
    from wsp import healdata

    path = "/vol/hill/mc.json"
    if os.path.exists(path):
        with open(path) as f:
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
    os.makedirs("/vol/hill", exist_ok=True)
    with open(path, "w") as f:
        json.dump(out, f)
    volume.commit()
    return out


MC_SOURCES = {
    # Train splits only, none a reported benchmark (checked with healdata.check_sources).
    "science": ("allenai/sciq", None, "train"),
    "math": ("deepmind/aqua_rat", "raw", "train"),
    "medicine": ("EleutherAI/headqa", "en", "train"),
}


def mc_domain(name):
    """4-option items for one training domain, in MMLU prompt format: {"train": [...], "eval": [...]}.

    general is the round 23 mix (CommonsenseQA, OpenBookQA, QASC, LogiQA2.0);
    science is SciQ, math is AQuA-RAT (answer letter only, no rationale),
    medicine is HEAD-QA (Spanish healthcare exams, English). 400 items per
    domain are held out as that domain's in-distribution eval.
    """
    if name == "general":
        return mc_items()
    from datasets import load_dataset

    from modal_round6 import mmlu_prompt
    from wsp import healdata

    path = f"/vol/hill/mc_{name}.json"
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    dataset_id, config, split = MC_SOURCES[name]
    healdata.check_sources([dataset_id])
    rows = load_dataset(dataset_id, config, split=split) if config else load_dataset(dataset_id, split=split)
    rng = random.Random(0)
    order = list(range(len(rows)))
    rng.shuffle(order)
    items = []
    for index in order[:12000]:
        r = rows[index]
        if name == "science":
            question, gold = r["question"], r["correct_answer"]
            others = [r["distractor1"], r["distractor2"], r["distractor3"]]
        elif name == "math":
            opts = [o.split(")", 1)[1].strip() if ")" in o else o for o in r["options"]]
            letters = [o.split(")", 1)[0].strip() for o in r["options"]]
            if r["correct"] not in letters:
                continue
            gold = opts[letters.index(r["correct"])]
            question = r["question"]
            others = [o for o in opts if o != gold]
        else:
            answers = {a["aid"]: a["atext"] for a in r["answers"]}
            if r["ra"] not in answers:
                continue
            gold = answers[r["ra"]]
            question = r["qtext"]
            others = [t for k, t in answers.items() if k != r["ra"]]
        others = [o for o in dict.fromkeys(others) if o and o != gold]
        if len(others) < 3 or not question:
            continue
        choices = rng.sample(others, 3) + [gold]
        rng.shuffle(choices)
        items.append([mmlu_prompt(question, choices), choices.index(gold)])
    out = {"train": items[400:], "eval": items[:400]}
    with open(path, "w") as f:
        json.dump(out, f)
    volume.commit()
    return out


def mc_train_set(names, seed=0):
    """Training items for one domain or an equal mix of several ("all" is every domain)."""
    if names == "all":
        names = ["general", "science", "math", "medicine"]
    if isinstance(names, str):
        names = [names]
    sets = [mc_domain(n)["train"] for n in names]
    n = min(len(s) for s in sets) if len(sets) > 1 else len(sets[0])
    rng = random.Random(seed)
    out = []
    for s in sets:
        out += rng.sample(s, n)
    rng.shuffle(out)
    return out


MMLU_GROUPS = {
    "medical": ["anatomy", "clinical_knowledge", "college_medicine", "medical_genetics", "professional_medicine",
                "nutrition", "virology", "college_biology", "high_school_biology"],
    "math": ["abstract_algebra", "college_mathematics", "elementary_mathematics", "high_school_mathematics",
             "high_school_statistics"],
    "science": ["astronomy", "college_chemistry", "college_physics", "high_school_chemistry", "high_school_physics",
                "conceptual_physics", "electrical_engineering"],
    "cs": ["college_computer_science", "high_school_computer_science", "computer_security", "machine_learning"],
    "humanities": ["high_school_us_history", "high_school_world_history", "philosophy", "prehistory",
                   "world_religions", "moral_disputes", "jurisprudence", "logical_fallacies"],
}


def mmlu_groups(limit=500):
    """Per-domain MMLU test items, taken from the seeded order after the dev (0..999) and held-out (1000..2999)
    items, so they overlap neither. The cross-domain panel."""
    from datasets import load_dataset

    from modal_round6 import mmlu_prompt

    test = load_dataset("cais/mmlu", "all", split="test")
    order = list(range(len(test)))
    random.Random(0).shuffle(order)
    subjects = test["subject"]
    out = {g: [] for g in MMLU_GROUPS}
    for index in order[3000:]:
        for g, names in MMLU_GROUPS.items():
            if subjects[index] in names and len(out[g]) < limit:
                row = test[index]
                out[g].append((mmlu_prompt(row["question"], row["choices"]), int(row["answer"])))
    return out


def medmcqa_items(n=1000):
    from datasets import load_dataset

    from modal_round6 import mmlu_prompt

    med = load_dataset("openlifescienceai/medmcqa", split="validation")
    order = list(range(len(med)))
    random.Random(7).shuffle(order)
    items = []
    for index in order[:n]:
        r = med[index]
        items.append((mmlu_prompt(r["question"], [r["opa"], r["opb"], r["opc"], r["opd"]]), int(r["cop"])))
    return items


MMMLU_LANGS = {"zh": "ZH_CN", "es": "ES_LA", "hi": "HI_IN", "ar": "AR_XA", "ja": "JA_JP"}


def bench_items(n=500):
    """Outside benchmarks, eval only, never trained on: ARC-Challenge and ARC-Easy (test, 4-option items),
    HellaSwag (validation), and MMLU translated into five languages (openai/MMMLU test). 4 options each,
    scored by the A-D letter like MMLU. Seeded samples of n items."""
    from datasets import load_dataset

    from modal_round6 import mmlu_prompt

    out = {}
    for key, cfg in (("arc_c", "ARC-Challenge"), ("arc_e", "ARC-Easy")):
        rows = [r for r in load_dataset("allenai/ai2_arc", cfg, split="test") if len(r["choices"]["label"]) == 4]
        random.Random(11).shuffle(rows)
        out[key] = [(mmlu_prompt(r["question"], r["choices"]["text"]), r["choices"]["label"].index(r["answerKey"]))
                    for r in rows[:n]]
    hs = list(load_dataset("Rowan/hellaswag", split="validation"))
    random.Random(12).shuffle(hs)
    out["hellaswag"] = [(mmlu_prompt("Which ending is most likely? " + r["ctx"], r["endings"]), int(r["label"]))
                        for r in hs[:n]]
    for key, cfg in MMMLU_LANGS.items():
        rows = list(load_dataset("openai/MMMLU", cfg, split="test"))
        random.Random(13).shuffle(rows)
        out[f"mmlu_{key}"] = [(mmlu_prompt(r["Question"], [r["A"], r["B"], r["C"], r["D"]]), "ABCD".index(r["Answer"]))
                             for r in rows[:n] if r["Answer"] in "ABCD"]
    return out


def library_specs(model_id, pgraft):
    """Every library part of one donor, resolved to plain specs."""
    specs = []
    with open("/root/library/catalog.jsonl") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if row["model"] == model_id and row["type"] in ("mlp_cluster", "attn_heads"):
                specs.append(pgraft.resolve_library({"id": row["id"]}, root="/root"))
    return specs


# --------------------------------------------------------------- scoring


def continuation_nll(model, tok, items, batch=16):
    """Mean over items of the summed NLL (nats) of the continuation after the context."""
    import torch
    import torch.nn.functional as F

    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    device = next(model.parameters()).device
    totals = []
    for start in range(0, len(items), batch):
        chunk = items[start:start + batch]
        rows = []
        for context, cont in chunk:
            c = tok(context)["input_ids"]
            full = tok(context + cont)["input_ids"]
            rows.append((full, len(c)))
        longest = max(len(r[0]) for r in rows)
        ids = torch.full((len(rows), longest), tok.pad_token_id, dtype=torch.long)
        mask = torch.zeros((len(rows), longest), dtype=torch.long)
        for i, (full, _) in enumerate(rows):
            ids[i, :len(full)] = torch.tensor(full)
            mask[i, :len(full)] = 1
        with torch.no_grad():
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(input_ids=ids.to(device), attention_mask=mask.to(device), use_cache=False).logits
        logp = F.log_softmax(logits.float(), dim=-1)
        for i, (full, n_ctx) in enumerate(rows):
            nll = 0.0
            for pos in range(max(1, n_ctx), len(full)):
                nll -= float(logp[i, pos - 1, full[pos]])
            totals.append(nll)
    return sum(totals) / max(1, len(totals))


def score_skills(model, tok, skills):
    return {name: round(continuation_nll(model, tok, sets["heldout"]), 4) for name, sets in skills.items()}


# ------------------------------------------------------------------ cell


@app.function(image=image, volumes={"/vol": volume}, gpu="H100", memory=131072, timeout=4 * 3600)
def cell(config):
    import gc

    import torch
    import transformers

    from modal_round6 import letter_ids, score_mmlu
    from modal_round7 import mmlu_split
    from wsp import pgraft
    from wsp.lora import apply_lora, default_alpha, rank_for_budget

    start = time.time()
    c = dict(DEFAULT, **config)
    c["lr"] = dict(DEFAULT["lr"], **config.get("lr", {}))
    seed = c["seed"]
    torch.manual_seed(seed)
    dom = DOMAINS[c["domain"]]
    data = domain_data(c["domain"])
    skills = skill_sets(dom["skills"])
    if c.get("smoke"):
        c["align_steps"], c["task_steps"] = 3, 3
        data = {k: v[:12] for k, v in data.items()}
        skills = {k: {s: v[:6] for s, v in sets.items()} for k, sets in list(skills.items())[:3]}
    cfg = dict(pgraft.DEFAULTS)

    host = c.get("host", HOST)
    tok = transformers.AutoTokenizer.from_pretrained(host)
    base = transformers.AutoModelForCausalLM.from_pretrained(host, dtype=torch.bfloat16).to("cuda").eval()
    ref = transformers.AutoModelForCausalLM.from_pretrained(host, dtype=torch.bfloat16).to("cuda").eval()
    dtok = transformers.AutoTokenizer.from_pretrained(dom["donor"])
    donor = transformers.AutoModelForCausalLM.from_pretrained(dom["donor"], dtype=torch.bfloat16).to("cuda").eval()
    for m in (base, ref, donor):
        for p in m.parameters():
            p.requires_grad_(False)
    vmap = pgraft.vocab_map(tok, dtok)
    row = {"config": config, "shared_vocab": int(vmap[0].numel())}

    row["base_skills"] = score_skills(base, tok, skills)
    row["donor_skills"] = score_skills(donor, dtok, skills)
    row["base_kd_gap"] = pgraft.kd_report(base, tok, donor, dtok, data["domain_eval"], vmap, cfg)

    part = c["part"]
    arm = c["arm"]
    d_donor = donor.config.hidden_size
    d_host = base.config.hidden_size
    specs = None
    if part["type"] in ("bundle", "trunk_bundle"):
        specs = []
        for spec in part["specs"]:
            if spec["kind"] == "library_all":
                specs += library_specs(dom["donor"] if c["domain"] != "math" else "Qwen/Qwen2.5-Math-1.5B", pgraft)
            elif spec["kind"] == "library":
                specs.append(pgraft.resolve_library(spec, root="/root"))
            else:
                specs.append(spec)
        specs = pgraft.expand_specs(specs, donor.config, control="random" if arm == "random" else None, seed=seed)
        row["parts"] = [{k: (v if k not in ("indices", "heads") else len(v)) for k, v in s.items()} for s in specs]
        row["part_params"] = None

    # The real arm's trainable count sets every arm's budget.
    budget = (pgraft.count(pgraft.Enc(d_host, d_donor, depth=c["enc_depth"]).parameters())
              + pgraft.count(pgraft.Dec(d_donor, d_host, rank=c["dec_rank"], depth=c["dec_depth"]).parameters())
              + d_host + 1)
    # Extra grafts (other donors, at other host layers) multiply the budget the LoRA arm gets.
    extras_cfg = c.get("extra_grafts", [])
    budget *= 1 + len(extras_cfg) if arm == "lora" else 1
    budget = int(budget * c.get("lora_budget_mult", 1.0)) if arm == "lora" else budget
    row["budget"] = budget
    graft = None
    extras = []
    lora_params = None
    if arm == "lora":
        rank = rank_for_budget(base, budget)
        lora_params = apply_lora(base, rank, default_alpha(rank))
        for p in lora_params:
            p.requires_grad_(True)
        row["lora_rank"] = rank
    elif arm == "none":
        pass
    else:
        control = "identity" if arm == "identity" else None
        if part["type"] == "trunk_bundle" and arm != "identity":
            graft = pgraft.build_trunk_bundle_graft(base, donor, c["host_layer"], part["first"], specs,
                                                    enc_depth=c["enc_depth"], dec_rank=c["dec_rank"],
                                                    dec_depth=c["dec_depth"])
            row["parts"] = [p for p in row["parts"] if p["layer"] >= part["first"]]
        elif part["type"] == "block" or arm == "identity":
            first = part.get("first", 10)
            last = part.get("last", 20)
            graft = pgraft.build(base, donor if control is None else None, c["host_layer"], first, last,
                                 control=control, enc_depth=c["enc_depth"], dec_rank=c["dec_rank"],
                                 dec_depth=c["dec_depth"], d_donor=d_donor, read_layers=c.get("read_layers"))
        else:
            graft = pgraft.build_bundle_graft(base, donor, c["host_layer"], specs, enc_depth=c["enc_depth"],
                                              dec_rank=c["dec_rank"], dec_depth=c["dec_depth"])
        for e in (extras_cfg if arm == "real" else []):
            # Extra donors load on the CPU; only their block goes to the GPU. Each full donor visits the GPU
            # once, for its own alignment, and is then freed, so several donors fit on one H100.
            edom = DOMAINS[e["domain"]]
            e_donor = transformers.AutoModelForCausalLM.from_pretrained(edom["donor"], dtype=torch.bfloat16).eval()
            for p in e_donor.parameters():
                p.requires_grad_(False)
            e_tok = transformers.AutoTokenizer.from_pretrained(edom["donor"])
            eg = pgraft.build(base, e_donor, e["host_layer"], e["first"], e["last"], enc_depth=c["enc_depth"],
                              dec_rank=c["dec_rank"], dec_depth=c["dec_depth"], read_layers=e.get("read_layers"))
            eg.part.to("cuda")
            extras.append([eg, e_donor, e_tok, e])
        if arm == "shuffled":
            # Same block and pipeline, the knowledge destroyed: permute every block tensor's entries. Alignment
            # still trains the encoder against the real donor, so only the block's weights differ from "real".
            gen = torch.Generator(device="cpu").manual_seed(1000 + seed)
            with torch.no_grad():
                for p in graft.part.parameters():
                    flat = p.data.flatten()
                    perm = torch.randperm(flat.numel(), generator=gen).to(flat.device)
                    p.data.copy_(flat[perm].view_as(p.data))
        pgraft.attach(base, [graft] + [x[0] for x in extras])
        for x in extras:
            eg, e_donor, e_tok, e = x
            e_donor.to("cuda")
            e_data = domain_data(e["domain"])
            pgraft.set_sink(eg, e_donor, e_tok, e["first"], e_data["align"][:128])
            # Alignment reads the host with every graft off, so each graft aligns to the plain host stream.
            ecfg = dict(cfg, align_steps=c["align_steps"], align_lr=c["lr"]["align"])
            pgraft.align(base, tok, e_donor, e_tok, eg, e_data["align"], ecfg, seed=seed, log_every=100)
            row.setdefault("extra_alignment", []).append(
                pgraft.alignment_report(base, tok, e_donor, e_tok, eg, e_data["align_eval"], cfg))
            row.setdefault("extra_calib_rms", []).append(pgraft.calibrate_scale(base, tok, eg, e_data["align"], cfg))
            x[1] = None
            del e_donor
            gc.collect()
            torch.cuda.empty_cache()
        extras = [x[0] for x in extras]
        gc.collect()
        torch.cuda.empty_cache()
        if control is None:
            first_layer = part.get("first") if part["type"] in ("block", "trunk_bundle") else \
                min(s["layer"] for s in specs)
            pgraft.set_sink(graft, donor, dtok, first_layer, data["align"][:128])
            acfg = dict(cfg, align_steps=c["align_steps"], align_lr=c["lr"]["align"])
            if part["type"] in ("block", "trunk_bundle"):
                pgraft.align(base, tok, donor, dtok, graft, data["align"], acfg, seed=seed, log_every=100)
                row["alignment"] = pgraft.alignment_report(base, tok, donor, dtok, graft, data["align_eval"], cfg)
            else:
                row["bundle_in_rms"] = pgraft.bundle_input_rms(graft.part, donor, dtok, data["align"], cfg)
                row["part_r2_before"] = pgraft.part_report(base, tok, donor, dtok, graft, data["align_eval"], cfg)
                pgraft.align_parts(base, tok, donor, dtok, graft, data["align"], acfg, seed=seed, log_every=100)
                row["part_r2"] = pgraft.part_report(base, tok, donor, dtok, graft, data["align_eval"], cfg)
                row["part_params"] = sum(p.numel() for p in graft.part.parts.parameters())
            print("alignment", row.get("alignment") or row.get("part_r2"))
        row["calib_rms"] = pgraft.calibrate_scale(base, tok, graft, data["align"], cfg)

    lr = c["lr"]
    if arm == "none":
        groups = []
    elif arm == "lora":
        groups = [{"params": lora_params, "lr": lr["lora"]}]
    elif arm == "identity":
        groups = [{"params": list(graft.enc.parameters()) + list(graft.dec.parameters()), "lr": lr["glue"]},
                  {"params": list(graft.token_gate.parameters()), "lr": lr["gate"]}]
    else:
        enc = list(graft.enc.parameters())
        if isinstance(graft.part, pgraft.PartBundle):
            enc.append(graft.part.gains)
        for p in enc:
            p.requires_grad_(True)
        for eg in extras:
            for p in eg.enc.parameters():
                p.requires_grad_(True)
            enc += list(eg.enc.parameters())
        groups = [{"params": list(graft.dec.parameters()) + [p for eg in extras for p in eg.dec.parameters()],
                   "lr": lr["glue"]},
                  {"params": list(graft.token_gate.parameters())
                   + [p for eg in extras for p in eg.token_gate.parameters()], "lr": lr["gate"]},
                  {"params": enc, "lr": lr["enc2"]}]
    if arm == "real" and c.get("with_lora"):
        # Graft plus a LoRA of the same budget on the host (applied after alignment; LoRA starts at zero).
        rank = rank_for_budget(base, budget)
        extra_lora = apply_lora(base, rank, default_alpha(rank))
        for p in extra_lora:
            p.requires_grad_(True)
        groups.append({"params": extra_lora, "lr": lr["lora"]})
        row["lora_rank"] = rank
    row["trainable"] = pgraft.count([p for g in groups for p in g["params"]])

    skill_texts = []
    for sets in skills.values():
        for context, cont in sets["search"]:
            skill_texts.append(context + cont)
    kd_texts = data["domain"] + skill_texts
    feed = pgraft._batches(kd_texts, 4, seed + 21)

    def note_gate(info):
        if graft is not None:
            g = pgraft.mean_gate([graft])
            if g is not None:
                info["gate"] = float(g.detach())
            for i, eg in enumerate(extras):
                g = pgraft.mean_gate([eg])
                if g is not None:
                    info[f"gate{i + 1}"] = float(g.detach())

    align_feed = pgraft._batches(data["align"], 4, seed + 11)

    def reg(step, info):
        # Keep the stage-1 functional KL as a stage-2 regularizer (blocks and trunk bundles, every 2nd step).
        if c["align_reg"] <= 0 or graft is None or arm != "real" or part["type"] == "bundle" or step % 2:
            return 0.0
        loss, _ = pgraft.functional_loss(base, tok, donor, dtok, graft, next(align_feed), cfg)
        info["reg"] = float(loss.detach())
        return c["align_reg"] * loss

    def kd_step(step):
        loss, info = pgraft.kd_loss(base, tok, donor, dtok, next(feed), vmap, cfg)
        note_gate(info)
        return loss + reg(step, info), info

    mc = {"train": mc_train_set(c["mc_train"], seed=seed), "eval": mc_items()["eval"]}
    mc_evals = {n: mc_domain(n)["eval"] for n in ("general", "science", "math", "medicine")}
    row["mc_train_size"] = len(mc["train"])
    if c.get("smoke"):
        mc = {"train": mc["train"][:32], "eval": mc["eval"][:16]}
        mc_evals = {n: v[:16] for n, v in mc_evals.items()}
    letters = letter_ids(tok)
    mc_feed = pgraft._batches(mc["train"], c.get("mc_batch", 8), seed + 31)
    gen_feed = pgraft._batches(data["general"], 4, seed + 41)
    device = torch.device("cuda")

    def mc_step(step):
        loss, info = pgraft.mc_loss(base, tok, [tuple(x) for x in next(mc_feed)], letters, device)
        note_gate(info)
        enc_batch = pgraft._tok_batch(tok, next(gen_feed), cfg["seq_len"], device)
        _, logits = pgraft.seq_lm_loss(base, enc_batch, return_logits=True)
        drift = pgraft.drift_kl(ref, enc_batch, logits)
        info["drift"] = float(drift.detach())
        return loss + c["preserve"] * drift + reg(step, info), info

    # The controls spend the alignment steps in their first stage, so every arm takes the same step count.
    extra = c["align_steps"] if arm in ("identity", "lora") else 0
    if c.get("smoke"):
        c["kd_steps"] = 3
    logs = {}
    if arm == "none":
        c["objective"] = ""
    if "kd" in c["objective"]:
        logs["kd"] = pgraft.train(base, groups, c["kd_steps"] + extra, kd_step, seed=seed, log_every=100)
        extra = 0
    if "mc" in c["objective"]:
        # MC training without the alignment regularizer never calls the donor: park it on the CPU.
        parked = c["align_reg"] <= 0
        if parked:
            donor.to("cpu")
            gc.collect()
            torch.cuda.empty_cache()
        logs["mc"] = pgraft.train(base, groups, c["task_steps"] + extra, mc_step, seed=seed + 1, log_every=100)
    for name, log in logs.items():
        row[name + "_first50"] = sum(r["loss"] for r in log[:50]) / len(log[:50])
        row[name + "_last50"] = sum(r["loss"] for r in log[-50:]) / len(log[-50:])
        for key in ["gate"] + [f"gate{i + 1}" for i in range(len(extras))]:
            gates = [round(r[key], 4) for r in log if key in r and r["step"] % 100 == 0]
            if gates:
                row[name + "_" + key + "_curve"] = gates

    if "mc" in c["objective"] and parked:
        donor.to("cuda")
    base.eval()
    row["kd_gap"] = pgraft.kd_report(base, tok, donor, dtok, data["domain_eval"], vmap, cfg)
    row["skills"] = score_skills(base, tok, skills)
    row["law"] = pgraft.heldout_utility(base, tok, data["law"], cfg, ref=ref)
    if graft is not None:
        with pgraft.grafts_off(base):
            row["kd_gap_off"] = pgraft.kd_report(base, tok, donor, dtok, data["domain_eval"], vmap, cfg)
            row["skills_off"] = score_skills(base, tok, skills)
        if row.get("part_r2") is not None or row.get("alignment") is not None:
            if part["type"] in ("block", "trunk_bundle"):
                row["alignment_end"] = pgraft.alignment_report(base, tok, donor, dtok, graft, data["align_eval"], cfg)
            else:
                row["part_r2_end"] = pgraft.part_report(base, tok, donor, dtok, graft, data["align_eval"], cfg)
    row["mc_eval"] = score_mmlu(base, tok, [tuple(x) for x in mc["eval"]], batch_size=16)
    row["mc_evals"] = {n: score_mmlu(base, tok, [tuple(x) for x in v], batch_size=16) for n, v in mc_evals.items()}
    del donor, ref
    gc.collect()
    torch.cuda.empty_cache()

    # Skill transfer: share of the base-to-donor NLL gap closed, on skills where the donor is clearly better.
    closes = []
    for name, b in row["base_skills"].items():
        d = row["donor_skills"][name]
        if b - d > 0.3:
            closes.append((b - row["skills"][name]) / (b - d))
    row["skill_transfer"] = sum(closes) / len(closes) if closes else None
    row["skill_gap_count"] = len(closes)

    # Select configs on dev; held-out is reported but reserved for confirming a winner with new seeds.
    _, dev = mmlu_split(tok, heldout=False)
    _, held = mmlu_split(tok, heldout=True)
    if c.get("smoke"):
        dev, held = dev[:32], held[:32]
    row["mmlu_dev"] = score_mmlu(base, tok, dev, batch_size=16)
    row["mmlu_heldout"] = score_mmlu(base, tok, held, batch_size=16)
    if graft is not None:
        with pgraft.grafts_off(base):
            row["mmlu_dev_off"] = score_mmlu(base, tok, dev, batch_size=16)
    # Cross-domain panel on every cell: MMLU by subject group (disjoint from dev and held-out) and MedMCQA.
    groups = mmlu_groups(16 if c.get("smoke") else 500)
    groups["medmcqa"] = medmcqa_items(32 if c.get("smoke") else 1000)
    if c.get("bench"):
        groups.update(bench_items(16 if c.get("smoke") else 500))
    row["panel"] = {g: score_mmlu(base, tok, items, batch_size=16) for g, items in groups.items()}
    row["medmcqa"] = row["panel"]["medmcqa"]
    if graft is not None:
        with pgraft.grafts_off(base):
            row["panel_off"] = {g: score_mmlu(base, tok, items, batch_size=16) for g, items in groups.items()}
    row["seconds"] = round(time.time() - start, 1)
    print(json.dumps(row))
    # Save on the volume, so a local disconnect does not lose the row.
    folder = f"/vol/hill/rows/{config.get('batch', 'adhoc')}"
    os.makedirs(folder, exist_ok=True)
    with open(f"{folder}/{config['name']}.json", "w") as f:
        json.dump(row, f)
    volume.commit()
    return row


# --------------------------------------------------------------- batches


def batch_h1():
    """First sweep: per domain, a donor block, the donor's library parts (real and random units), whole-layer
    MLPs, and the identity and LoRA controls. One seed; winners get more seeds."""
    out = []
    plan = {
        "medicine": {"block": (10, 20), "mlp_layers": [14, 18, 22]},
        "code": {"block": (10, 20), "mlp_layers": [14, 18, 22]},
        "math": {"block": (8, 20), "mlp_layers": [12, 16, 20]},
    }
    for domain, p in plan.items():
        first, last = p["block"]
        out.append({"name": f"{domain}-block", "domain": domain, "part": {"type": "block", "first": first, "last": last},
                    "arm": "real"})
        out.append({"name": f"{domain}-identity", "domain": domain, "part": {"type": "block"}, "arm": "identity"})
        out.append({"name": f"{domain}-lora", "domain": domain, "part": {"type": "block"}, "arm": "lora"})
        lib = {"type": "bundle", "specs": [{"kind": "library_all"}]}
        out.append({"name": f"{domain}-library", "domain": domain, "part": lib, "arm": "real"})
        out.append({"name": f"{domain}-library-random", "domain": domain, "part": lib, "arm": "random"})
        mlps = {"type": "bundle", "specs": [{"kind": "mlp_layer", "layer": L} for L in p["mlp_layers"]]}
        out.append({"name": f"{domain}-mlp-layers", "domain": domain, "part": mlps, "arm": "real"})
    return out


def batch_h2():
    """Trunk bundles: the donor's own layers build each subnetwork's input, only the subnetworks come back.
    Plus an architecture climb around round 23's winner on knowledge (answer-letter training, MMLU)."""
    out = []
    for domain, first, mlps in (("medicine", 10, [20, 23, 26]), ("code", 10, [20, 23, 26]), ("math", 8, [16, 20, 24])):
        lib = {"type": "trunk_bundle", "first": first, "specs": [{"kind": "library_all"}]}
        out.append({"name": f"{domain}-tb-library", "domain": domain, "part": lib, "arm": "real"})
        out.append({"name": f"{domain}-tb-library-random", "domain": domain, "part": lib, "arm": "random"})
        whole = {"type": "trunk_bundle", "first": first, "specs": [{"kind": "mlp_layer", "layer": L} for L in mlps]}
        out.append({"name": f"{domain}-tb-mlp", "domain": domain, "part": whole, "arm": "real"})
    k = {"domain": "knowledge", "objective": "mc", "align_steps": 600, "task_steps": 1000, "enc_depth": 2}
    blk = {"type": "block", "first": 10, "last": 20}
    out.append(dict(k, name="k-base", part=blk, arm="real"))
    out.append(dict(k, name="k-identity", part=blk, arm="identity"))
    out.append(dict(k, name="k-lora", part=blk, arm="lora"))
    out.append(dict(k, name="k-enc3", part=blk, arm="real", enc_depth=3))
    out.append(dict(k, name="k-host5", part=blk, arm="real", host_layer=5))
    out.append(dict(k, name="k-host9", part=blk, arm="real", host_layer=9))
    out.append(dict(k, name="k-span4-20", part={"type": "block", "first": 4, "last": 20}, arm="real"))
    out.append(dict(k, name="k-span10-24", part={"type": "block", "first": 10, "last": 24}, arm="real"))
    out.append(dict(k, name="k-dec1024", part=blk, arm="real", dec_rank=1024))
    out.append(dict(k, name="k-steps2000", part=blk, arm="real", task_steps=2000))
    out.append(dict(k, name="k-lora-steps2000", part=blk, arm="lora", task_steps=2000))
    return out


def batch_h3():
    """Confirm h2's knowledge winner (span 4..20) with new seeds against seeded controls, climb around it
    (span, alignment regularizer, host layer, longer LoRA), and carry the span to medicine (leak-free data)
    and to the new library donor Qwen2.5-Math-7B-Instruct. Select on MMLU dev; held-out confirms."""
    out = []
    k = {"domain": "knowledge", "objective": "mc", "align_steps": 600, "task_steps": 1000, "enc_depth": 2}
    win = {"type": "block", "first": 4, "last": 20}
    for seed in (1, 2):
        out.append(dict(k, name=f"k-span4-20-s{seed}", part=win, arm="real", seed=seed))
        out.append(dict(k, name=f"k-identity-s{seed}", part=win, arm="identity", seed=seed))
        out.append(dict(k, name=f"k-lora-s{seed}", part=win, arm="lora", seed=seed))
    out.append(dict(k, name="k-lora-steps2000", part=win, arm="lora", task_steps=2000))
    out.append(dict(k, name="k-span4-20-reg", part=win, arm="real", align_reg=0.5))
    out.append(dict(k, name="k-span2-20", part={"type": "block", "first": 2, "last": 20}, arm="real"))
    out.append(dict(k, name="k-span4-24", part={"type": "block", "first": 4, "last": 24}, arm="real"))
    out.append(dict(k, name="k-span4-20-host5", part=win, arm="real", host_layer=5))
    for domain in ("medicine", "math7b"):
        out.append({"name": f"{domain}-span4-20", "domain": domain, "part": win, "arm": "real"})
        out.append({"name": f"{domain}-span4-20-mc", "domain": domain, "part": win, "arm": "real", "objective": "mc",
                    "align_steps": 600, "task_steps": 1000, "enc_depth": 2})
        out.append({"name": f"{domain}-identity", "domain": domain, "part": win, "arm": "identity"})
        out.append({"name": f"{domain}-lora", "domain": domain, "part": win, "arm": "lora"})
    return out


def batch_h4():
    """Cross-domain fine-tuning: train answer letters on one domain (general, science, math, medicine, or all),
    score every domain (the four in-distribution evals, MMLU by subject group, MedMCQA). Per training domain:
    LoRA, identity, the Qwen2.5-7B graft, and the specialist donor's graft for math and medicine.
    Then the projection layers: where the encoder reads and where the graft writes."""
    out = []
    win = {"type": "block", "first": 4, "last": 20}
    k = {"objective": "mc", "align_steps": 600, "task_steps": 1000, "enc_depth": 2, "part": win}
    for train in ("general", "science", "math", "medicine", "all"):
        t = dict(k, mc_train=train)
        out.append(dict(t, name=f"x-{train}-lora", domain="knowledge", arm="lora"))
        out.append(dict(t, name=f"x-{train}-identity", domain="knowledge", arm="identity"))
        out.append(dict(t, name=f"x-{train}-qwen7b", domain="knowledge", arm="real"))
    out.append(dict(k, mc_train="medicine", name="x-medicine-apollo", domain="medicine", arm="real"))
    out.append(dict(k, mc_train="math", name="x-math-math7b", domain="math7b", arm="real"))
    g = dict(k, domain="knowledge", arm="real", mc_train="general")
    out.append(dict(g, name="p-read3-write7", read_layers=[3]))
    out.append(dict(g, name="p-read3.5.7-write7", read_layers=[3, 5, 7]))
    out.append(dict(g, name="p-read1to7-write7", read_layers=list(range(1, 8))))
    out.append(dict(g, name="p-read5-write9", host_layer=9, read_layers=[5]))
    out.append(dict(g, name="p-read7-write11", host_layer=11, read_layers=[7]))
    out.append(dict(g, name="p-read3.7.11-write11", host_layer=11, read_layers=[3, 7, 11]))
    return out


def batch_h5():
    """Composition: two donors' blocks in one network (Qwen2.5-7B at host 7 plus a specialist at host 10),
    against LoRA at the doubled budget, trained on all domains' answer letters."""
    out = []
    win = {"type": "block", "first": 4, "last": 20}
    k = {"objective": "mc", "align_steps": 600, "task_steps": 1000, "enc_depth": 2, "part": win,
         "domain": "knowledge", "mc_train": "all"}
    med = {"domain": "medicine", "host_layer": 10, "first": 4, "last": 20}
    math = {"domain": "math7b", "host_layer": 10, "first": 4, "last": 20}
    out.append(dict(k, name="c-qwen7b+apollo", arm="real", extra_grafts=[med]))
    out.append(dict(k, name="c-qwen7b+math7b", arm="real", extra_grafts=[math]))
    out.append(dict(k, name="c-qwen7b+apollo+math7b", arm="real",
                    extra_grafts=[med, dict(math, host_layer=12)]))
    out.append(dict(k, name="c-qwen7b-x2", arm="real", extra_grafts=[dict(med, domain="knowledge", first=12, last=24)]))
    out.append(dict(k, name="c-lora-2x", arm="lora", lora_budget_mult=2.0))
    out.append(dict(k, name="c-lora-3x", arm="lora", lora_budget_mult=3.0))
    return out


def batch_h6():
    """A bigger host: is the gain still there on Llama-3.2-3B-Instruct (28 layers), and how does the 1B graft
    compare with a 3B model at about the same total size? Untrained bases for the panel reference too."""
    out = []
    win = {"type": "block", "first": 4, "last": 20}
    k = {"objective": "mc", "align_steps": 600, "task_steps": 1000, "enc_depth": 2, "part": win,
         "domain": "knowledge", "mc_train": "general"}
    big = dict(k, host="unsloth/Llama-3.2-3B-Instruct", host_layer=12)
    out.append(dict(k, name="b1-base", arm="none"))
    out.append(dict(big, name="b3-base", arm="none"))
    out.append(dict(big, name="b3-lora", arm="lora"))
    out.append(dict(big, name="b3-identity", arm="identity"))
    out.append(dict(big, name="b3-qwen7b", arm="real"))
    out.append(dict(big, name="b3-qwen7b-host8", arm="real", host_layer=8))
    return out


def batch_h7():
    """A stronger donor: Qwen2.5-14B-Instruct (48 layers, width 5120). Spans scaled from 4..20 of 28 layers,
    with LoRA and identity at the 14B graft's budget."""
    out = []
    k = {"objective": "mc", "align_steps": 600, "task_steps": 1000, "enc_depth": 2, "domain": "knowledge14",
         "mc_train": "general"}
    blk = {"type": "block", "first": 7, "last": 34}
    out.append(dict(k, name="q14-span7-34", part=blk, arm="real"))
    out.append(dict(k, name="q14-span7-34-s1", part=blk, arm="real", seed=1))
    out.append(dict(k, name="q14-span4-24", part={"type": "block", "first": 4, "last": 24}, arm="real"))
    out.append(dict(k, name="q14-span10-40", part={"type": "block", "first": 10, "last": 40}, arm="real"))
    out.append(dict(k, name="q14-span7-34-all", part=blk, arm="real", mc_train="all"))
    out.append(dict(k, name="q14-lora", part=blk, arm="lora"))
    out.append(dict(k, name="q14-identity", part=blk, arm="identity"))
    return out


def batch_h8():
    """Confirm h4's multi-layer reads on new seeds, push the read further, and combine with all-domain training
    and a math specialist written later in the host."""
    out = []
    win = {"type": "block", "first": 4, "last": 20}
    k = {"objective": "mc", "align_steps": 600, "task_steps": 1000, "enc_depth": 2, "part": win,
         "domain": "knowledge", "mc_train": "general", "arm": "real"}
    r11 = dict(k, host_layer=11, read_layers=[3, 7, 11])
    r7 = dict(k, host_layer=7, read_layers=list(range(1, 8)))
    for seed in (1, 2):
        out.append(dict(r11, name=f"m-read3.7.11-w11-s{seed}", seed=seed))
        out.append(dict(r7, name=f"m-read1to7-w7-s{seed}", seed=seed))
    out.append(dict(k, name="m-read1to11-w11", host_layer=11, read_layers=list(range(1, 12))))
    out.append(dict(k, name="m-read0to15-w15", host_layer=15, read_layers=list(range(0, 16))))
    out.append(dict(k, name="m-read1.3.5.7.9-w9", host_layer=9, read_layers=[1, 3, 5, 7, 9]))
    out.append(dict(r11, name="m-read3.7.11-w11-all", mc_train="all"))
    out.append(dict(r11, name="m-read3.7.11-w11-enc3", enc_depth=3))
    out.append(dict(r11, name="m-read3.7.11-w11-span2-24", part={"type": "block", "first": 2, "last": 24}))
    math = {"domain": "math7b", "host_layer": 13, "first": 4, "last": 20, "read_layers": [5, 9, 13]}
    out.append(dict(r7, name="m-qwen7b-r7+math7b-r13-all", mc_train="all", extra_grafts=[math]))
    out.append(dict(k, name="m-lora-2x-all", arm="lora", mc_train="all", lora_budget_mult=2.0))
    return out


def batch_h9():
    """The 3B host, where h6's graft only tied LoRA: multi-layer reads, graft plus LoRA (against LoRA at twice
    the budget), the 14B donor, and second seeds for the h6 cells."""
    out = []
    win = {"type": "block", "first": 4, "last": 20}
    big = {"objective": "mc", "align_steps": 600, "task_steps": 1000, "enc_depth": 2, "part": win,
           "domain": "knowledge", "mc_train": "general", "host": "unsloth/Llama-3.2-3B-Instruct", "arm": "real"}
    out.append(dict(big, name="b3-read2.5.8-w8", host_layer=8, read_layers=[2, 5, 8]))
    out.append(dict(big, name="b3-read1to8-w8", host_layer=8, read_layers=list(range(1, 9))))
    out.append(dict(big, name="b3-read4.8.12-w12", host_layer=12, read_layers=[4, 8, 12]))
    out.append(dict(big, name="b3-read2to14-w14", host_layer=14, read_layers=list(range(2, 15, 2))))
    out.append(dict(big, name="b3-host8+lora", host_layer=8, with_lora=True))
    out.append(dict(big, name="b3-read1to8-w8+lora", host_layer=8, read_layers=list(range(1, 9)), with_lora=True))
    out.append(dict(big, name="b3-lora-2x", arm="lora", lora_budget_mult=2.0))
    out.append(dict(big, name="b3-lora-s1", arm="lora", seed=1))
    out.append(dict(big, name="b3-host8-s1", host_layer=8, seed=1))
    out.append(dict(big, name="b3-q14-host8", domain="knowledge14", host_layer=8,
                    part={"type": "block", "first": 7, "last": 34}))
    out.append(dict(big, name="b3-q14-lora", domain="knowledge14", arm="lora",
                    part={"type": "block", "first": 7, "last": 34}))
    return out


def batch_h10():
    """The 14B donor (h7's winner, block 7..34) with dense multi-layer reads, spans around it, a medicine
    specialist added later in the host, and the h8 two-donor cell that ran out of memory."""
    out = []
    blk = {"type": "block", "first": 7, "last": 34}
    k = {"objective": "mc", "align_steps": 600, "task_steps": 1000, "enc_depth": 2, "part": blk,
         "domain": "knowledge14", "mc_train": "general", "arm": "real"}
    for seed in (0, 1):
        out.append(dict(k, name=f"q14-read1to7-w7-s{seed}", read_layers=list(range(1, 8)), seed=seed))
        out.append(dict(k, name=f"q14-read1to11-w11-s{seed}", host_layer=11, read_layers=list(range(1, 12)),
                        seed=seed))
    out.append(dict(k, name="q14-span7-40", part={"type": "block", "first": 7, "last": 40}))
    out.append(dict(k, name="q14-span4-34", part={"type": "block", "first": 4, "last": 34}))
    out.append(dict(k, name="q14-span7-34-host5", host_layer=5))
    out.append(dict(k, name="q14-span7-34-dec512", dec_rank=512))
    med = {"domain": "medicine", "host_layer": 11, "first": 4, "last": 20}
    out.append(dict(k, name="q14+apollo-w11-all", mc_train="all", extra_grafts=[med]))
    out.append(dict(k, name="q14-lora-2x-all", arm="lora", mc_train="all", lora_budget_mult=2.0))
    win = {"type": "block", "first": 4, "last": 20}
    math = {"domain": "math7b", "host_layer": 13, "first": 4, "last": 20, "read_layers": [5, 9, 13]}
    out.append({"objective": "mc", "align_steps": 600, "task_steps": 1000, "enc_depth": 2, "part": win,
                "domain": "knowledge", "mc_train": "all", "arm": "real", "host_layer": 7,
                "read_layers": list(range(1, 8)), "extra_grafts": [math], "name": "m-qwen7b-r7+math7b-r13-all"})
    return out


def batch_h11():
    """Phi-4 (another family, 40 layers, width 5120) as a donor; the 14B graft on the 3B host with a second seed,
    a dense read and a longer block; and the three-donor cell at MC batch 4 (it ran out of memory at 8)."""
    out = []
    k = {"objective": "mc", "align_steps": 600, "task_steps": 1000, "enc_depth": 2, "mc_train": "general",
         "arm": "real"}
    out.append(dict(k, name="phi4-span6-28", domain="phi4", part={"type": "block", "first": 6, "last": 28}))
    out.append(dict(k, name="phi4-span4-32", domain="phi4", part={"type": "block", "first": 4, "last": 32}))
    out.append(dict(k, name="phi4-span6-28-read1to7", domain="phi4", part={"type": "block", "first": 6, "last": 28},
                    read_layers=list(range(1, 8))))
    big = dict(k, domain="knowledge14", host="unsloth/Llama-3.2-3B-Instruct", host_layer=8,
               part={"type": "block", "first": 7, "last": 34})
    out.append(dict(big, name="b3-q14-host8-s1", seed=1))
    out.append(dict(big, name="b3-q14-lora-s1", arm="lora", seed=1))
    out.append(dict(big, name="b3-q14-span7-40", part={"type": "block", "first": 7, "last": 40}))
    out.append(dict(big, name="b3-q14-host10", host_layer=10))
    out.append(dict(big, name="b3-q14-host8-all", mc_train="all"))
    win = {"type": "block", "first": 4, "last": 20}
    med = {"domain": "medicine", "host_layer": 10, "first": 4, "last": 20}
    math = {"domain": "math7b", "host_layer": 12, "first": 4, "last": 20}
    out.append(dict(k, name="c-qwen7b+apollo+math7b-b4", domain="knowledge", part=win, mc_train="all", mc_batch=4,
                    extra_grafts=[med, math]))
    out.append(dict(k, name="c-lora-3x-b4", domain="knowledge", part=win, mc_train="all", mc_batch=4, arm="lora",
                    lora_budget_mult=3.0))
    return out


def batch_h12():
    """Qwen2.5-32B-Instruct (64 layers) as the donor, on an H200: into the 1B and the 3B host."""
    out = []
    k = {"objective": "mc", "align_steps": 600, "task_steps": 1000, "enc_depth": 2, "mc_train": "general",
         "arm": "real", "domain": "knowledge32", "big": True, "part": {"type": "block", "first": 9, "last": 45}}
    out.append(dict(k, name="q32-span9-45"))
    out.append(dict(k, name="q32-span9-54", part={"type": "block", "first": 9, "last": 54}))
    out.append(dict(k, name="b3-q32-host8", host="unsloth/Llama-3.2-3B-Instruct", host_layer=8))
    return out


def batch_h13():
    """Push h10's winner (14B block 7..34 read from host layers 1..7 into Llama-1B): a third seed, longer
    alignment and task training, deeper encoder, all-domain training, the same read on the 3B host, and the
    14B + Apollo2 composition at MC batch 4 (it ran out of memory at 8)."""
    out = []
    blk = {"type": "block", "first": 7, "last": 34}
    k = {"objective": "mc", "align_steps": 600, "task_steps": 1000, "enc_depth": 2, "part": blk,
         "domain": "knowledge14", "mc_train": "general", "arm": "real", "read_layers": list(range(1, 8))}
    out.append(dict(k, name="w-s2", seed=2))
    out.append(dict(k, name="w-align1200", align_steps=1200))
    out.append(dict(k, name="w-task2000", task_steps=2000))
    out.append(dict(k, name="w-enc3", enc_depth=3))
    out.append(dict(k, name="w-read0to7", read_layers=list(range(0, 8))))
    out.append(dict(k, name="w-read1to5-w5", host_layer=5, read_layers=list(range(1, 6))))
    out.append(dict(k, name="w-all", mc_train="all"))
    out.append(dict(k, name="w-lora-s2", arm="lora", seed=2))
    out.append(dict(k, name="b3-w-read1to8-w8", host="unsloth/Llama-3.2-3B-Instruct", host_layer=8,
                    read_layers=list(range(1, 9))))
    med = {"domain": "medicine", "host_layer": 11, "first": 4, "last": 20, "read_layers": list(range(7, 12))}
    out.append(dict(k, name="w+apollo-all-b4", mc_train="all", mc_batch=4, extra_grafts=[med]))
    out.append(dict(k, name="w-all-b4", mc_train="all", mc_batch=4))
    return out


def batch_h14():
    """Size control at 8B: Llama-3.1-8B-Instruct (32 layers) untrained, with LoRA, and with the 14B graft."""
    out = []
    k = {"objective": "mc", "align_steps": 600, "task_steps": 1000, "enc_depth": 2, "mc_train": "general",
         "domain": "knowledge14", "part": {"type": "block", "first": 7, "last": 34},
         "host": "unsloth/Meta-Llama-3.1-8B-Instruct"}
    out.append(dict(k, name="b8-base", arm="none"))
    out.append(dict(k, name="b8-lora", arm="lora"))
    out.append(dict(k, name="b8-q14-host10", arm="real", host_layer=10))
    out.append(dict(k, name="b8-q14-host14", arm="real", host_layer=14))
    return out


def batch_h15():
    """Qwen2.5-32B on an H200: a second seed for h12's cells, the 1B winner's tricks (reads 1..7, 1200 alignment
    steps) on the 32B block, and the 8B host with the 32B and 14B grafts (the 8B host does not fit an H100 with a
    14B donor and a reference copy)."""
    out = []
    k = {"objective": "mc", "align_steps": 600, "task_steps": 1000, "enc_depth": 2, "mc_train": "general",
         "arm": "real", "domain": "knowledge32", "big": True, "part": {"type": "block", "first": 9, "last": 54}}
    out.append(dict(k, name="q32-span9-54-s1", seed=1))
    out.append(dict(k, name="q32-r1to7-a1200", read_layers=list(range(1, 8)), align_steps=1200))
    out.append(dict(k, name="q32-r1to7-a1200-s1", read_layers=list(range(1, 8)), align_steps=1200, seed=1))
    out.append(dict(k, name="q32-a1200", align_steps=1200))
    b3 = dict(k, host="unsloth/Llama-3.2-3B-Instruct", host_layer=8, part={"type": "block", "first": 9, "last": 45})
    out.append(dict(b3, name="b3-q32-host8-s1", seed=1))
    out.append(dict(b3, name="b3-q32-span9-54", part={"type": "block", "first": 9, "last": 54}))
    out.append(dict(b3, name="b3-q32-a1200", align_steps=1200))
    b8 = dict(k, host="unsloth/Meta-Llama-3.1-8B-Instruct", host_layer=10)
    out.append(dict(b8, name="b8-q32-host10"))
    out.append(dict(b8, name="b8-q14-host10-big", domain="knowledge14", part={"type": "block", "first": 7, "last": 34}))
    out.append(dict(b8, name="b8-lora-big", arm="lora"))
    return out


def batch_h16():
    """Is it the donor's knowledge? The winning grafts with every block tensor shuffled (same pipeline), and
    the knowledge-free donor Phi-4 already failed. Plus a 3B-host second seed of 1200-step alignment."""
    out = []
    k = {"objective": "mc", "align_steps": 600, "task_steps": 1000, "enc_depth": 2, "mc_train": "general"}
    q14 = dict(k, domain="knowledge14", part={"type": "block", "first": 7, "last": 34}, read_layers=list(range(1, 8)))
    out.append(dict(q14, name="sh-q14-r1to7", arm="shuffled"))
    out.append(dict(q14, name="sh-q14-r1to7-s1", arm="shuffled", seed=1))
    q7 = dict(k, domain="knowledge", part={"type": "block", "first": 4, "last": 20})
    out.append(dict(q7, name="sh-q7", arm="shuffled"))
    out.append(dict(q14, name="w-a1200-s1", arm="real", align_steps=1200, seed=1))
    out.append(dict(q14, name="w-a1200-s2", arm="real", align_steps=1200, seed=2))
    return out


def batch_h17():
    """Breadth table: every method trained on the same all-domain answer-letter mix, scored on MMLU, the MMLU
    subject groups, MedMCQA, ARC-C/E, HellaSwag, and MMLU in five languages. 1B host. Base, LoRA (1x and 2x
    budget), a shuffled-block control, and grafts from six donors: general (Qwen2.5 7B and 14B), code, math,
    medicine, and a newer family member (Qwen3-8B)."""
    out = []
    k = {"objective": "mc", "align_steps": 600, "task_steps": 1000, "enc_depth": 2, "mc_train": "all", "bench": True,
         "domain": "knowledge", "part": {"type": "block", "first": 4, "last": 20}}
    out.append(dict(k, name="br-base", arm="none"))
    out.append(dict(k, name="br-lora", arm="lora"))
    out.append(dict(k, name="br-lora-2x", arm="lora", lora_budget_mult=2.0))
    out.append(dict(k, name="br-shuffled-q7", arm="shuffled"))
    out.append(dict(k, name="br-q7", arm="real"))
    out.append(dict(k, name="br-q14-r1to7", arm="real", domain="knowledge14",
                    part={"type": "block", "first": 7, "last": 34}, read_layers=list(range(1, 8))))
    out.append(dict(k, name="br-coder7b", arm="real", domain="code"))
    out.append(dict(k, name="br-math7b", arm="real", domain="math7b"))
    out.append(dict(k, name="br-apollo", arm="real", domain="medicine"))
    out.append(dict(k, name="br-qwen3-8b", arm="real", domain="qwen3_8b", part={"type": "block", "first": 4, "last": 26}))
    return out


BATCHES = {"h1": batch_h1, "h2": batch_h2, "h3": batch_h3, "h4": batch_h4, "h5": batch_h5, "h6": batch_h6, "h7": batch_h7, "h8": batch_h8, "h9": batch_h9, "h10": batch_h10, "h11": batch_h11, "h12": batch_h12, "h13": batch_h13, "h14": batch_h14, "h15": batch_h15, "h16": batch_h16, "h17": batch_h17}


@app.function(image=image, volumes={"/vol": volume}, gpu="H200", memory=262144, timeout=5 * 3600)
def cell_big(config):
    """The same cell on an H200, for donors too large for an H100 (Qwen2.5-32B)."""
    return cell.local(config)


@app.local_entrypoint()
def main(batch: str = "h1", only: str = "", smoke: bool = False, extra: str = ""):
    configs = BATCHES[batch]() if batch in BATCHES else []
    if extra:
        with open(extra) as f:
            configs = json.load(f)
    if only:
        keep = set(only.split(","))
        configs = [c for c in configs if c["name"] in keep]
    if smoke:
        configs = [dict(c, smoke=True) for c in configs]
    configs = [dict(c, batch=batch + ("_smoke" if smoke else "")) for c in configs]
    print(f"{len(configs)} cells:", [c["name"] for c in configs])
    rows = []
    fn = cell_big if any(c.get("big") for c in configs) else cell
    for row in fn.map(configs, return_exceptions=True):
        if isinstance(row, Exception):
            print("cell failed:", repr(row)[:1000])
            continue
        rows.append(row)
    os.makedirs("results/hill", exist_ok=True)
    name = f"results/hill/{batch}{'_smoke' if smoke else ''}_rows.json"
    with open(name, "w") as f:
        json.dump(rows, f, indent=2)
    print("wrote", name)
