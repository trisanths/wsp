"""Round 22: parallel grafts (wsp/pgraft.py) across families, with the identity-donor control.

Base unsloth/Llama-3.2-1B-Instruct (16 layers, MMLU dev 47.5). Donor
Qwen/Qwen2.5-7B-Instruct (28 layers, 71.5); different tokenizers. The graft
reads base layer 7 (about 50% depth), runs donor layers k..20 (frozen) on
the base's positions, and writes the donor's contribution back through a
rank-64 Dec and a token gate.

Phase "align" (the quality gate, no task training): for k in {4, 10, 16},
train Enc for 600 steps on functional KL (the donor, fed Enc's states at
layer k and run to its own LM head, against its native predictions at
offset-paired positions), and report how much of the donor's native
next-token accuracy survives on held-out prose and knowledge text.

Phase "full" (2 seeds), the same data and step count for every arm:
  base       untrained
  lora       LoRA on the base, rank matched to the graft's trainable count,
             1600 LM steps on the same mix
  pgraft     align 600 steps (Enc), then integrate 1000 steps (Dec + gate)
             with the utility hinge, stochastic depth and gate L1, annealed
  identity   the same Enc, Dec and gate with the donor skipped, 1600 steps
             (Enc trains too, since there is no donor to align to)

Heal data: wsp.healdata pools only (knowledge and medicine as the
domain; general and prose as general text). MMLU items are scored, never
trained on. MMLU dev 1000 / held-out 2000.

    modal run modal_round22pg.py --phase align
    modal run modal_round22pg.py --phase full --k 10
"""

import json
import os
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
    .add_local_file("modal_graft.py", "/root/modal_graft.py")
    .add_local_file("modal_round5.py", "/root/modal_round5.py")
    .add_local_file("modal_round6.py", "/root/modal_round6.py")
    .add_local_file("modal_round7.py", "/root/modal_round7.py")
)

HOST = "unsloth/Llama-3.2-1B-Instruct"
DONOR = "Qwen/Qwen2.5-7B-Instruct"
HOST_LAYER = 7
LAST = 20
ALIGN_KS = [4, 10, 16]
SEEDS = [0, 1]
ARMS = ["base", "lora", "pgraft", "identity"]
CFG = {"align_steps": 600, "geo_steps": 100, "int_steps": 1000}
LORA_LR = 2e-4
DATA = "/vol/round22/data.json"
D_DONOR = 3584
D_HOST = 2048
SMOKE = {"align_steps": 4, "geo_steps": 2, "int_steps": 4}
# v1: run 22 as first run. v2 (run 22b): Enc keeps training in stage 2 at 1e-4, Dec rank 256,
# gate L1 1e-4, and every trained arm gets KL to a frozen copy of the base on general batches.
VARIANTS = {
    "v1": {"dec_rank": 64, "cfg": {}, "stage2_enc": False, "preserve": False},
    "v2": {"dec_rank": 256, "cfg": {"gate_l1": 1e-4, "preserve_weight": 1.0, "enc_lr2": 1e-4},
           "stage2_enc": True, "preserve": True},
    # v3 (run 22d): v2 plus the stage-1 functional KL kept as a stage-2 regularizer (weight 1, every 2nd
    # step), graft fidelity logged every 100 steps, and update-ratio / drift-KL metrics. Ablation fixed in
    # advance if v3 wins: v3 with the encoder frozen in stage 2 (v3_frozen), the change most likely to matter.
    "v3": {"dec_rank": 256, "cfg": {"gate_l1": 1e-4, "preserve_weight": 1.0, "enc_lr2": 1e-4},
           "stage2_enc": True, "preserve": True, "align_reg": 1.0},
    "v3_frozen": {"dec_rank": 256, "cfg": {"gate_l1": 1e-4, "preserve_weight": 1.0},
                  "stage2_enc": False, "preserve": True, "align_reg": 0.0},
}


def build_data():
    """Text pools, cached on the volume. Plain text (no chat template), so both tokenizers read the same characters."""
    from wsp import healdata

    if os.path.exists(DATA):
        with open(DATA) as f:
            return json.load(f)
    data = {
        "align": healdata.mix(["prose", "knowledge", "medicine"], 1000, seed=0),
        "align_eval": healdata.mix(["prose", "knowledge", "medicine"], 40, seed=0, skip=1000),
        "domain": healdata.mix(["knowledge", "medicine"], 3000, seed=1, skip=1200),
        "general": healdata.mix(["general", "prose"], 3000, seed=2, skip=1200),
        "domain_eval": healdata.mix(["knowledge", "medicine"], 120, seed=1, skip=4400),
        "general_eval": healdata.mix(["general", "prose"], 120, seed=2, skip=4400),
    }
    os.makedirs(os.path.dirname(DATA), exist_ok=True)
    tmp = f"{DATA}.{os.getpid()}.tmp"
    with open(tmp, "w") as f:
        json.dump(data, f)
    os.replace(tmp, DATA)
    volume.commit()
    return data


def load_models(with_donor=True):
    import torch
    import transformers

    tok = transformers.AutoTokenizer.from_pretrained(HOST)
    base = transformers.AutoModelForCausalLM.from_pretrained(HOST, dtype=torch.bfloat16).to("cuda").eval()
    for p in base.parameters():
        p.requires_grad_(False)
    donor = donor_tok = None
    if with_donor:
        donor_tok = transformers.AutoTokenizer.from_pretrained(DONOR)
        donor = transformers.AutoModelForCausalLM.from_pretrained(DONOR, dtype=torch.bfloat16).to("cuda").eval()
        for p in donor.parameters():
            p.requires_grad_(False)
    return base, tok, donor, donor_tok


def make_graft(base, donor, donor_tok, k, control, data, dec_rank=64):
    from wsp import pgraft

    graft = pgraft.build(base, donor, HOST_LAYER, k, LAST, control=control, enc_depth=2, dec_rank=dec_rank,
                         d_donor=D_DONOR)
    pgraft.attach(base, [graft])
    if control != "identity":
        pgraft.set_sink(graft, donor, donor_tok, k, data["align"][:256])
    return graft


@app.function(image=image, volumes={"/vol": volume}, gpu="H100", memory=131072, timeout=3 * 3600)
def align_cell(k, seed=0, smoke=False):
    """Quality gate for donor start layer k: survival before and after functional alignment."""
    import torch

    from wsp import pgraft

    start = time.time()
    torch.manual_seed(seed)
    data = build_data()
    cfg_all = dict(CFG, **SMOKE) if smoke else CFG
    if smoke:
        data = {key: value[:16] for key, value in data.items()}
    base, tok, donor, donor_tok = load_models()
    graft = make_graft(base, donor, donor_tok, k, None, data)
    before = pgraft.alignment_report(base, tok, donor, donor_tok, graft, data["align_eval"], cfg_all)
    print("before", before)
    log = pgraft.align(base, tok, donor, donor_tok, graft, data["align"], cfg_all, seed=seed)
    after = pgraft.alignment_report(base, tok, donor, donor_tok, graft, data["align_eval"], cfg_all)
    print("after", after)
    return {"phase": "align", "k": k, "seed": seed, "before": before, "after": after,
            "kl_first20": sum(r["kl"] for r in log[:20]) / len(log[:20]),
            "kl_last20": sum(r["kl"] for r in log[-20:]) / len(log[-20:]),
            "enc_params": pgraft.count(graft.enc.parameters()), "seconds": round(time.time() - start, 1)}


@app.function(image=image, volumes={"/vol": volume}, gpu="H100", memory=131072, timeout=4 * 3600)
def full_cell(arm, seed, k, smoke=False, variant="v1"):
    import gc

    import torch

    from modal_round6 import score_mmlu
    from modal_round7 import mmlu_split
    from wsp import pgraft
    from wsp.lora import apply_lora, default_alpha, rank_for_budget

    start = time.time()
    torch.manual_seed(seed)
    data = build_data()
    global CFG
    spec = VARIANTS[variant]
    CFG = dict(CFG, **spec["cfg"])
    if smoke:
        CFG = dict(CFG, **SMOKE)
        data = {key: value[:16] for key, value in data.items()}
    dec_rank = spec["dec_rank"]
    stage_lora = arm.endswith("_lora")
    if stage_lora:
        arm = arm[:-len("_lora")]
    lora_scale = 1
    if arm == "lora2x":
        # LoRA sized to graft + LoRA (the staged arms' total), with their total step count.
        arm = "lora"
        lora_scale = 2
    need_donor = arm == "pgraft"
    base, tok, donor, donor_tok = load_models(with_donor=need_donor)
    row = {"phase": "full", "arm": ("lora2x" if lora_scale == 2 else arm) + ("_lora" if stage_lora else ""), "seed": seed, "k": k, "variant": variant}
    ref = None
    if spec["preserve"] and arm != "base":
        import transformers

        ref = transformers.AutoModelForCausalLM.from_pretrained(HOST, dtype=torch.bfloat16).to("cuda").eval()
        for p in ref.parameters():
            p.requires_grad_(False)
    total_steps = CFG["align_steps"] + CFG["int_steps"]
    # The graft's trainable count (Enc + Dec + gate) sets the LoRA rank, built on the meta device.
    budget = (pgraft.count(pgraft.Enc(D_HOST, D_DONOR, depth=2).parameters())
              + pgraft.count(pgraft.Dec(D_DONOR, D_HOST, rank=dec_rank).parameters()) + D_HOST + 1)

    if arm == "lora":
        rank = rank_for_budget(base, budget * lora_scale)
        params = apply_lora(base, rank, default_alpha(rank))
        for p in params:
            p.requires_grad_(True)
        row["lora_rank"] = rank
        cfg = dict(CFG, int_steps=total_steps * lora_scale, int_lr=LORA_LR)
        log = pgraft.integrate(base, tok, data["domain"], data["general"], params, cfg, seed=seed, aids=False,
                               ref=ref)
    elif arm == "identity":
        graft = make_graft(base, None, None, k, "identity", data, dec_rank)
        row["calib_rms"] = pgraft.calibrate_scale(base, tok, graft, data["align"], CFG)
        params = pgraft.trainable([graft], enc=True)
        cfg = dict(CFG, int_steps=total_steps)
        log = pgraft.integrate(base, tok, data["domain"], data["general"], params, cfg, seed=seed, ref=ref)
    elif arm == "pgraft":
        graft = make_graft(base, donor, donor_tok, k, None, data, dec_rank)
        align_log = pgraft.align(base, tok, donor, donor_tok, graft, data["align"], CFG, seed=seed)
        row["alignment"] = pgraft.alignment_report(base, tok, donor, donor_tok, graft, data["align_eval"], CFG)
        row["align_kl_last20"] = sum(r["kl"] for r in align_log[-20:]) / len(align_log[-20:])
        print("alignment", row["alignment"])
        reg_fn = probe_fn = None
        reg_weight = spec.get("align_reg", 0.0)
        if reg_weight > 0:
            full_cfg = dict(pgraft.DEFAULTS, **CFG)
            feed = pgraft._batches(data["align"], full_cfg["align_batch"], seed + 11)

            def reg_fn(step):
                if step % 2:
                    return None
                loss, _ = pgraft.functional_loss(base, tok, donor, donor_tok, graft, next(feed), full_cfg)
                return reg_weight * loss

        if variant != "v1":
            def probe_fn(step):
                return pgraft.alignment_report(base, tok, donor, donor_tok, graft, data["align_eval"][:8], CFG)
        else:
            # The full donor is only needed for alignment; the graft keeps its own copy of layers k..20.
            del donor
            gc.collect()
            torch.cuda.empty_cache()
        row["calib_rms"] = pgraft.calibrate_scale(base, tok, graft, data["align"], CFG)
        for p in graft.enc.parameters():
            p.requires_grad_(False)
        params = pgraft.trainable([graft], enc=False)
        enc_params = None
        if spec["stage2_enc"]:
            enc_params = list(graft.enc.parameters())
            for p in enc_params:
                p.requires_grad_(True)
        log = pgraft.integrate(base, tok, data["domain"], data["general"], params, CFG, seed=seed, ref=ref,
                               enc_params=enc_params, reg_fn=reg_fn, probe_fn=probe_fn)
        fidelity = []
        for r in log:
            if "probe" in r:
                fidelity.append([r["step"], round(r["probe"]["survival"], 4), round(r["probe"]["kl"], 4)])
        row["fidelity_during_stage2"] = fidelity
        if variant != "v1":
            row["alignment_after_stage2"] = pgraft.alignment_report(base, tok, donor, donor_tok, graft,
                                                                    data["align_eval"], CFG)
            del donor
            gc.collect()
            torch.cuda.empty_cache()
        if enc_params:
            row["stage2_enc_params"] = pgraft.count(enc_params)
    else:
        params = []
        log = []
    if stage_lora:
        # Graft (or identity control) first, then freeze it and train the same LoRA as the lora arm on top.
        # The LoRA learns the text format in both arms, so what is left between them is the donor.
        row["before_lora_domain_eval"] = pgraft.heldout_utility(base, tok, data["domain_eval"], CFG)
        for p in base.parameters():
            p.requires_grad_(False)
        rank = rank_for_budget(base, budget)
        lora_params = apply_lora(base, rank, default_alpha(rank))
        for p in lora_params:
            p.requires_grad_(True)
        row["lora_rank"] = rank
        cfg = dict(CFG, int_steps=total_steps, int_lr=LORA_LR)
        pgraft.integrate(base, tok, data["domain"], data["general"], lora_params, cfg, seed=seed + 50,
                         aids=False, ref=ref)
        row["lora_trainable"] = pgraft.count(lora_params)
    row["trainable"] = pgraft.count(params)
    row["budget"] = budget
    if log:
        row["lm_first50"] = sum(r["lm"] for r in log[:50]) / len(log[:50])
        row["lm_last50"] = sum(r["lm"] for r in log[-50:]) / len(log[-50:])
    base.eval()
    row["domain_eval"] = pgraft.heldout_utility(base, tok, data["domain_eval"], CFG, ref=ref)
    row["general_eval"] = pgraft.heldout_utility(base, tok, data["general_eval"], CFG, ref=ref)
    if ref is not None:
        del ref
        torch.cuda.empty_cache()
    _, dev = mmlu_split(tok, heldout=False)
    _, held = mmlu_split(tok, heldout=True)
    if smoke:
        dev = dev[:32]
        held = held[:32]
    row["mmlu_dev"] = score_mmlu(base, tok, dev, batch_size=16)
    row["mmlu_heldout"] = score_mmlu(base, tok, held, batch_size=16)
    if arm in ("pgraft", "identity") and not stage_lora:
        with pgraft.grafts_off(base):
            row["mmlu_dev_graft_off"] = score_mmlu(base, tok, dev, batch_size=16)
    row["params_b"] = round(sum(p.numel() for p in base.parameters()) / 1e9, 3)
    row["seconds"] = round(time.time() - start, 1)
    print(json.dumps(row))
    return row


@app.local_entrypoint()
def main(phase: str = "align", k: int = 10, only: str = "", smoke: bool = False, variant: str = "v1",
         tag: str = ""):
    os.makedirs("results", exist_ok=True)
    rows = []
    if phase == "align":
        calls = [(kk, 0, smoke) for kk in ALIGN_KS]
        if only:
            calls = [(int(x), 0, smoke) for x in only.split(",")]
        results = align_cell.starmap(calls, return_exceptions=True)
        out = "results/round22_align_rows.json" if not smoke else "results/round22_smoke_align.json"
    else:
        calls = [(arm, seed, k, smoke, variant) for arm in ARMS for seed in SEEDS
                 if not (arm == "base" and (seed > 0 or variant != "v1"))]
        if only:
            calls = []
            for item in only.split(","):
                arm, seed = item.split(":")
                calls.append((arm, int(seed), k, smoke, variant))
        results = full_cell.starmap(calls, return_exceptions=True)
        suffix = ("" if variant == "v1" else "_" + variant) + (("_" + tag) if tag else "")
        out = f"results/round22{suffix}_full_rows.json" if not smoke else "results/round22_smoke_full.json"
    for row in results:
        if isinstance(row, Exception):
            print("cell failed:", repr(row)[:800])
            continue
        rows.append(row)
        print(json.dumps(row))
    with open(out, "w") as f:
        json.dump(rows, f, indent=2)
    print("wrote", out)
