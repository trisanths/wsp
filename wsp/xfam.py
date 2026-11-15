"""Cross-family write-in as library code: build, train, save, load, and score one network.

The recipe is the one in docs/write_in_xfam.md: a frozen slice of a donor from another family runs as a parallel
branch inside a small host (wsp/pgraft.py), joined by learned glue and a LoRA on the host, trained on four-option
items with the donor's soft answer-letter distribution as a second target. modal_wi_xfam.py (the research driver)
and the compiler (wsp/auto.py, method "pgraft") call the helpers in this file, so the two paths score the same way.

    spec = spec_for(base_cfg, donor_cfg)        # layers, sizes, steps from the recipe
    built = train_network(base, tok, donor, donor_tok, pool, texts, spec, seed)
    save_artifact(dir, built, spec, meta)       # glue, LoRA, sink, and references to base and donor
    model = load_artifact(dir)                  # one network again, ready to score

The artifact holds what training changed (glue, LoRA, the donor's sink vector) and names what it did not change
(the host and the donor slice, by model id, with a fingerprint of the host). The frozen donor slice is read from the
donor's own weights at load time, so the artifact is small (tens of megabytes) and cannot drift from the donor.
"""

import copy
import hashlib
import json
import os
import random
import time

import torch

ARTIFACT_FORMAT = "wsp-pgraft"
ARTIFACT_VERSION = 1
ARTIFACT_DIR = "pgraft_v%d" % ARTIFACT_VERSION
DATA_VERSION = "v1"
DATA_DIR = "/vol/wi_xfam/" + DATA_VERSION      # pool.json, evals.json, donor_acc.json (modal_wi_xfam.py prep)
TEXT_CACHE = "/vol/round22/data.json"           # align and general text pools of round 22
EVAL_NAMES = ("mmlu_dev", "mmlu_held", "arc_c", "obqa")
LETTERS = "ABCD"

# The recipe of docs/write_in_xfam.md, graft_lora preset at 4000 task steps. Layers are set by spec_for.
RECIPE = {
    "enc_depth": 2, "dec_rank": 256, "lora_rank": 16,
    "steps": 4000, "batch": 16, "align_steps": 600, "geo_steps": 100,
    "kd_w": 1.0, "preserve": 1.0, "align_reg": 1.0,
    "lora_lr": 2e-4, "glue_lr": 3e-4, "gate_lr": 1e-4, "enc_lr": 1e-4,
    "seq_len": 256, "donor_seq_len": 320, "align_batch": 4,
}
# Layer rule scaled from the measured config (host layer 7 of 16, donor layers 4..20 of 28).
HOST_LAYER_RULE = "host layers // 2 - 1"
DONOR_SPAN_RULE = "donor layers // 7 to 5 * donor layers // 7"

CONTROLS = (
    ("identity", "same Enc, Dec, gate, LoRA and data; the donor slice is skipped (Dec reads Enc's output)"),
    ("scrambled", "same glue and data; the donor slice's weights rewritten by wsp/scramble.py (spectrum kept)"),
    ("lora_match", "LoRA on the host with the same trainable count as the graft, same data and steps"),
    ("lora_big", "LoRA rank 128 on the host, same data and steps"),
    ("fullft", "full fine-tune of the host, same data and steps"),
)


def spec_for(base_cfg, donor_cfg, overrides=None):
    """The graft spec for a host and a donor, from their normalized configs (wsp.lineage.normalize_config)."""
    host_layers = int(base_cfg["layers"])
    donor_layers = int(donor_cfg["layers"])
    spec = dict(RECIPE)
    spec["host_layer"] = max(0, host_layers // 2 - 1)
    spec["first"] = donor_layers // 7
    spec["last"] = min(donor_layers - 1, 5 * donor_layers // 7)
    spec["d_host"] = int(base_cfg["hidden"])
    spec["d_donor"] = int(donor_cfg["hidden"])
    if overrides:
        spec.update(overrides)
    if not 0 <= spec["first"] <= spec["last"] < donor_layers:
        raise ValueError("donor span %d..%d does not fit a donor with %d layers" % (spec["first"], spec["last"], donor_layers))
    return spec


def glue_params(spec):
    """Trainable parameters of one graft's Enc, Dec and token gate, from the spec alone."""
    d_host, d_donor = spec["d_host"], spec["d_donor"]
    hidden = d_donor // 4
    enc = d_host * d_donor + d_donor + 1 + (spec["enc_depth"] - 1) * (2 * d_donor * hidden + hidden + d_donor)
    dec = d_donor * spec["dec_rank"] + spec["dec_rank"] * d_host
    return enc + dec + d_host + 1


# ------------------------------------------------------------------ prompts, letters, scoring


def letters_of(tok):
    return [tok(" " + c, add_special_tokens=False).input_ids[-1] for c in LETTERS]


def letter_probs(model, tok, items, batch=32):
    """Softmax over the four answer-letter logits at the last prompt token, for each (prompt, gold) item."""
    ids = letters_of(tok)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    device = next(model.parameters()).device
    order = sorted(range(len(items)), key=lambda i: len(items[i][0]))
    out = [None] * len(items)
    for s in range(0, len(order), batch):
        idx = order[s:s + batch]
        enc = tok([items[i][0] for i in idx], return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            logits = model(**enc).logits[:, -1, :].float()[:, ids]
        probs = torch.softmax(logits, -1).cpu().tolist()
        for i, p in zip(idx, probs):
            out[i] = [round(x, 5) for x in p]
    return out


def score_letters(model, tok, items, bs=16):
    """Accuracy by the argmax over the log-probs of ' A' to ' D' after the prompt (the rule of modal_round6.score_mmlu)."""
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    ids = letters_of(tok)
    device = next(model.parameters()).device
    correct = 0
    items = [tuple(x) for x in items]
    for start in range(0, len(items), bs):
        batch = items[start:start + bs]
        enc = tok([p for p, _ in batch], return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            logits = model(**enc).logits[:, -1, :].float()
        for row, (_, answer) in enumerate(batch):
            if int(logits[row, ids].argmax()) == answer:
                correct += 1
    return correct / len(items)


def evaluate(model, tok, evals, bs=16, names=None):
    return {k: score_letters(model, tok, v, bs) for k, v in evals.items() if names is None or k in names}


def decode_tps(model, tok, new=96):
    """Greedy decode speed at batch 1 after a warmup call. Returns a number or an "error: ..." string."""
    try:
        tok.padding_side = "left"
        device = next(model.parameters()).device
        enc = tok("The history of the Roman Empire begins", return_tensors="pt").to(device)
        model.eval()
        with torch.no_grad():
            model.generate(**enc, max_new_tokens=8, do_sample=False)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t = time.time()
            out = model.generate(**enc, max_new_tokens=new, do_sample=False, min_new_tokens=new)
            if device.type == "cuda":
                torch.cuda.synchronize()
        return round((out.shape[1] - enc["input_ids"].shape[1]) / (time.time() - t), 2)
    except Exception as e:  # recorded, not fatal
        return "error: %s" % repr(e)[:120]


# ------------------------------------------------------------------ data (cached on the volume, built on a miss)


def build_eval_sets(host_id):
    """MMLU dev 1000, held-out 2000 (test items), ARC-Challenge test, OpenBookQA test, as [prompt, gold] lists."""
    from datasets import load_dataset

    import transformers
    from modal_round6 import mmlu_prompt
    from modal_round7 import mmlu_split

    tok = transformers.AutoTokenizer.from_pretrained(host_id)
    _, dev = mmlu_split(tok, heldout=False)
    _, held = mmlu_split(tok, heldout=True)
    out = {"mmlu_dev": [list(x) for x in dev], "mmlu_held": [list(x) for x in held]}
    for name, repo, config, key in (("arc_c", "allenai/ai2_arc", "ARC-Challenge", "question"),
                                    ("obqa", "allenai/openbookqa", "main", "question_stem")):
        items = []
        for r in load_dataset(repo, config, split="test"):
            labels, texts = list(r["choices"]["label"]), list(r["choices"]["text"])
            if len(texts) != 4 or r["answerKey"] not in labels:
                continue
            items.append([mmlu_prompt(r[key], texts), labels.index(r["answerKey"])])
        out[name] = items
    return out


def build_pool(evals, aux_count=45000):
    """Training items: MMLU auxiliary_train plus the round 23 MC pools, minus any prompt in an eval set. (items, n_aux)."""
    from datasets import load_dataset

    import modal_round23pg as r23
    from modal_round6 import mmlu_prompt

    banned = {p for k in evals for p, _ in evals[k]}
    rng = random.Random(7)
    aux = load_dataset("cais/mmlu", "auxiliary_train", split="train")
    order = list(range(len(aux)))
    rng.shuffle(order)
    items, seen = [], set()
    for index in order:
        row = aux[index]
        if "train" in row and isinstance(row["train"], dict):
            row = row["train"]
        if len(row["choices"]) != 4:
            continue
        prompt = mmlu_prompt(row["question"], row["choices"])
        if len(prompt) > 1400 or prompt in banned or prompt in seen:
            continue
        seen.add(prompt)
        items.append([prompt, int(row["answer"])])
        if len(items) >= aux_count:
            break
    n_aux = len(items)
    mc = r23.mc_items()
    for prompt, gold in mc["train"] + mc["eval"]:
        if prompt in banned or prompt in seen:
            continue
        seen.add(prompt)
        items.append([prompt, int(gold)])
    rng.shuffle(items)
    return items, n_aux


def load_or_build_data(data_dir, host_id, donor=None, donor_tok=None, log=print):
    """Read evals.json and pool.json from data_dir; on a miss build them (the donor scores every pool item once).

    Returns {"evals", "items", "soft", "n_aux", "donor_train_acc", "source"}. The cached files are the ones the
    wi_xfam rows trained and scored on, so a compile that finds them reuses the same items.
    """
    evals_path = os.path.join(data_dir, "evals.json")
    pool_path = os.path.join(data_dir, "pool.json")
    if os.path.exists(evals_path) and os.path.exists(pool_path):
        with open(evals_path) as f:
            evals = json.load(f)
        with open(pool_path) as f:
            pool = json.load(f)
        return {"evals": evals, "items": pool["items"], "soft": pool["soft"], "n_aux": pool.get("n_aux"),
                "donor_train_acc": pool.get("donor_train_acc"), "source": "cache " + data_dir}
    if donor is None or donor_tok is None:
        raise FileNotFoundError("%s has no evals.json and pool.json, and no donor was given to label a new pool" % data_dir)
    log("data cache missing in %s: building evals and the pool (about 20 minutes on a GPU)" % data_dir)
    evals = build_eval_sets(host_id)
    items, n_aux = build_pool(evals)
    soft = letter_probs(donor, donor_tok, items)
    acc = sum(int(max(range(4), key=lambda j: row[j]) == g) for row, (_, g) in zip(soft, items)) / len(items)
    os.makedirs(data_dir, exist_ok=True)
    with open(evals_path, "w") as f:
        json.dump(evals, f)
    with open(pool_path, "w") as f:
        json.dump({"items": items, "soft": soft, "n_aux": n_aux, "donor_train_acc": acc}, f)
    return {"evals": evals, "items": items, "soft": soft, "n_aux": n_aux, "donor_train_acc": acc, "source": "built"}


def load_or_build_texts(path=TEXT_CACHE, reported=None, log=print):
    """The alignment and general text pools of round 22 (healdata mixes, plain text, no benchmark data)."""
    if path and os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    from wsp import healdata

    log("text cache missing at %s: building the healdata mixes" % path)
    data = {
        "align": healdata.mix(["prose", "knowledge", "medicine"], 1000, seed=0, reported=reported),
        "align_eval": healdata.mix(["prose", "knowledge", "medicine"], 40, seed=0, skip=1000, reported=reported),
        "general": healdata.mix(["general", "prose"], 3000, seed=2, skip=1200, reported=reported),
    }
    return data


# ------------------------------------------------------------------ building and training


def fingerprint(model):
    """A short hash of weights the write-in never changes (input embedding rows and the final norm)."""
    h = hashlib.sha256()
    emb = model.get_input_embeddings().weight[:64].detach().float().cpu().contiguous()
    h.update(emb.numpy().tobytes())
    norm = getattr(getattr(model, "model", None), "norm", None)
    if norm is not None:
        h.update(norm.weight.detach().float().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()[:16]


class Built:
    """What train_network returns: the graft(s), LoRA parameters, and the row of measurements."""

    def __init__(self, grafts, lora_params, row):
        self.grafts = grafts
        self.lora_params = lora_params
        self.row = row


def build_network(base, tok, donor, donor_tok, spec, align_texts, arm="graft", seed=0):
    """LoRA on the host, one parallel graft attached, sink set. arm: graft, identity, scrambled (donor needed unless identity)."""
    from wsp import pgraft
    from wsp.lora import apply_lora, default_alpha

    control = "identity" if arm == "identity" else None
    for p in base.parameters():
        p.requires_grad_(False)
    lora_params = []
    if spec["lora_rank"]:
        lora_params = apply_lora(base, spec["lora_rank"], default_alpha(spec["lora_rank"]))
        for p in lora_params:
            p.requires_grad_(True)
    graft = pgraft.build(base, donor if control is None else None, spec["host_layer"], spec["first"], spec["last"],
                         control=control, enc_depth=spec["enc_depth"], dec_rank=spec["dec_rank"], d_donor=spec["d_donor"])
    pgraft.attach(base, [graft])
    if donor is not None and control is None:
        pgraft.set_sink(graft, donor, donor_tok, graft.donor_span[0], align_texts[:256])
    return graft, lora_params


def train_network(base, tok, donor, donor_tok, pool, texts, spec, seed, arm="graft", ref=None, log=print):
    """Align, calibrate, and train one write-in network. Returns Built with graft, LoRA params and a measurement row.

    pool: {"items": [[prompt, gold], ...], "soft": [[p_A..p_D], ...]}; texts: {"align", "align_eval", "general"}.
    Controls (identity, scrambled) run the same code with the same steps and, as in round 23, spend the stage-1
    alignment steps on the task. ref: a frozen copy of the host for the drift KL (made here when None).
    """
    from wsp import pgraft, scramble

    t0 = time.time()
    torch.manual_seed(seed)
    device = next(base.parameters()).device
    cfg = dict(pgraft.DEFAULTS, align_steps=spec["align_steps"], geo_steps=spec["geo_steps"], int_steps=1000,
               seq_len=spec["seq_len"], donor_seq_len=spec["donor_seq_len"], align_batch=spec["align_batch"])
    if ref is None:
        ref = copy.deepcopy(base)
        for p in ref.parameters():
            p.requires_grad_(False)
        ref.eval()
    row = {"arm": arm, "seed": seed, "spec": dict(spec), "pool": len(pool["items"])}
    row["params_base"] = sum(p.numel() for p in base.parameters())
    letters = letters_of(tok)
    graft, lora_params = build_network(base, tok, donor, donor_tok, spec, texts["align"], arm, seed)
    glue = pgraft.trainable([graft], enc=True)
    row["trainable_glue"] = pgraft.count(glue)
    row["slice_params"] = sum(p.numel() for p in graft.part.parameters()) if graft.control is None else 0
    if arm != "identity":
        pgraft.align(base, tok, donor, donor_tok, graft, texts["align"], dict(cfg, align_steps=spec["align_steps"]),
                     seed=seed, log_every=100)
        row["alignment"] = pgraft.alignment_report(base, tok, donor, donor_tok, graft, texts["align_eval"], cfg)
        log("alignment %s" % json.dumps(row["alignment"], default=str)[:400])
    row["calib_rms"] = pgraft.calibrate_scale(base, tok, graft, texts["align"], cfg)
    if arm == "scrambled":
        graft.part = copy.deepcopy(graft.part)
        rep = scramble.scramble_module(graft.part, seed=seed + 100, mode="spectrum")
        row["scramble_cos"] = rep.get("mean_abs_cosine")
    row["trainable"] = pgraft.count(glue) + pgraft.count(lora_params)

    gate = list(graft.token_gate.parameters())
    dec = list(graft.dec.parameters())
    enc = list(graft.enc.parameters())
    groups = [{"params": dec + (enc if arm == "identity" else []), "lr": spec["glue_lr"]},
              {"params": gate, "lr": spec["gate_lr"]}]
    if arm != "identity":
        groups.append({"params": enc, "lr": spec["enc_lr"]})
    if lora_params:
        groups.append({"params": lora_params, "lr": spec["lora_lr"]})
    steps = spec["steps"]
    if arm != "graft":
        steps += spec["align_steps"]  # controls spend the stage-1 steps on the task
    row["train_steps"] = steps

    items, soft = pool["items"], pool["soft"]
    rng = random.Random(seed + 5)
    order = list(range(len(items)))
    rng.shuffle(order)
    ptr = [0]

    def next_idx():
        if ptr[0] + spec["batch"] > len(order):
            rng.shuffle(order)
            ptr[0] = 0
        idx = order[ptr[0]:ptr[0] + spec["batch"]]
        ptr[0] += spec["batch"]
        return idx

    gen_feed = pgraft._batches(texts["general"], 4, seed + 41)
    align_feed = pgraft._batches(texts["align"], cfg["align_batch"], seed + 11) if arm == "graft" else None
    tokens = [0]

    def step_fn(step):
        idx = next_idx()
        batch = [tuple(items[i]) for i in idx]
        sp = torch.tensor([soft[i] for i in idx])
        loss, info = pgraft.mc_soft_loss(base, tok, batch, letters, device, soft=sp, kd_weight=spec["kd_w"])
        tokens[0] += info.pop("tokens")
        if spec["preserve"] > 0:
            e = pgraft._tok_batch(tok, next(gen_feed), cfg["seq_len"], device)
            _, logits = pgraft.seq_lm_loss(base, e, return_logits=True)
            drift = pgraft.drift_kl(ref, e, logits)
            loss = loss + spec["preserve"] * drift
            info["drift"] = float(drift.detach())
        if arm == "graft" and spec["align_reg"] > 0 and step % 2 == 0:
            rl, _ = pgraft.functional_loss(base, tok, donor, donor_tok, graft, next(align_feed), cfg)
            loss = loss + spec["align_reg"] * rl
        gm = pgraft.mean_gate([graft])
        if gm is not None:
            info["gate"] = float(gm.detach())
        return loss, info

    tlog = pgraft.train(base, groups, steps, step_fn, seed=seed + 1, log_every=100)
    row["tokens_trained"] = tokens[0]
    row["gate_curve"] = [[r["step"], round(r["gate"], 4)] for r in tlog if "gate" in r and r["step"] % 100 == 0]
    row["loss_first50"] = sum(r["loss"] for r in tlog[:50]) / max(1, len(tlog[:50]))
    row["loss_last50"] = sum(r["loss"] for r in tlog[-50:]) / max(1, len(tlog[-50:]))
    row["train_seconds"] = round(time.time() - t0, 1)
    row["params_per_token"] = row["params_base"] + row["slice_params"] + row["trainable_glue"]
    row["params_total"] = row["params_per_token"]
    base.eval()
    return Built([graft], lora_params, row)


# ------------------------------------------------------------------ artifact


def artifact_path(out_dir):
    return os.path.join(out_dir, ARTIFACT_DIR)


def save_artifact(out_dir, base, built, spec, meta):
    """Write out_dir/pgraft_v1/{meta.json, glue.safetensors, lora.safetensors}. Returns the directory.

    meta: host and donor model ids (and anything else worth keeping, such as scores). The donor's weights are not
    copied; the loader reads the slice from the donor by id. The host is not copied either, because the write-in
    leaves it frozen (the LoRA is saved apart); meta holds a fingerprint of it.
    """
    from safetensors.torch import save_file

    from wsp.lora import lora_state_dict

    graft = built.grafts[0]
    path = artifact_path(out_dir)
    os.makedirs(path, exist_ok=True)
    glue = {}
    for key, value in graft.state_dict().items():
        if key.startswith("part."):
            continue
        glue[key] = value.detach().cpu().contiguous()
    if graft.control is None and graft.sink is None:
        raise ValueError("the graft has no sink vector; build it with set_sink before saving")
    lora = lora_state_dict(base)
    save_file(glue, os.path.join(path, "glue.safetensors"))
    save_file(lora, os.path.join(path, "lora.safetensors"))
    info = {
        "format": ARTIFACT_FORMAT, "format_version": ARTIFACT_VERSION, "spec": dict(spec),
        "control": graft.control, "donor_span": list(graft.donor_span), "host_layer": graft.planned_layer,
        "host_fingerprint": fingerprint(base), "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "glue_tensors": len(glue), "lora_tensors": len(lora),
        "glue_bytes": sum(v.numel() * v.element_size() for v in glue.values()),
        "lora_bytes": sum(v.numel() * v.element_size() for v in lora.values()),
    }
    info.update(meta)
    for key in ("host", "donor"):
        if key not in info:
            raise ValueError("meta needs %r (a model id or path)" % key)
    with open(os.path.join(path, "meta.json"), "w") as f:
        json.dump(info, f, indent=1, default=str)
    return path


def read_meta(path):
    with open(os.path.join(path, "meta.json")) as f:
        meta = json.load(f)
    if meta.get("format") != ARTIFACT_FORMAT or meta.get("format_version") != ARTIFACT_VERSION:
        raise ValueError("%s is %s version %s; this loader reads %s version %d" % (
            path, meta.get("format"), meta.get("format_version"), ARTIFACT_FORMAT, ARTIFACT_VERSION))
    return meta


def load_artifact(path, base=None, donor=None, device=None, dtype=torch.bfloat16, host=None, donor_id=None):
    """Rebuild the one network from an artifact directory. Returns (model, meta).

    base and donor may be passed (loaded models); otherwise they load from the ids in meta (host and donor_id
    override them). The donor is freed after its slice is copied into the graft.
    """
    import gc

    import transformers
    from safetensors.torch import load_file

    from wsp import pgraft
    from wsp.lora import apply_lora, default_alpha, load_lora_state

    meta = read_meta(path)
    spec = meta["spec"]
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if base is None:
        base = transformers.AutoModelForCausalLM.from_pretrained(host or meta["host"], dtype=dtype).to(device).eval()
    if fingerprint(base) != meta["host_fingerprint"]:
        raise ValueError("the host does not match the artifact (fingerprint %s, saved %s)" % (fingerprint(base), meta["host_fingerprint"]))
    for p in base.parameters():
        p.requires_grad_(False)
    control = meta.get("control")
    owned_donor = False
    if control is None and donor is None:
        donor = transformers.AutoModelForCausalLM.from_pretrained(donor_id or meta["donor"], dtype=dtype).to(device).eval()
        owned_donor = True
    if spec["lora_rank"]:
        params = apply_lora(base, spec["lora_rank"], default_alpha(spec["lora_rank"]))
        for p in params:
            p.requires_grad_(False)
        load_lora_state(base, load_file(os.path.join(path, "lora.safetensors")))
    graft = pgraft.build(base, donor if control is None else None, spec["host_layer"], spec["first"], spec["last"],
                         control=control, enc_depth=spec["enc_depth"], dec_rank=spec["dec_rank"], d_donor=spec["d_donor"])
    pgraft.attach(base, [graft])
    glue = load_file(os.path.join(path, "glue.safetensors"))
    sink = glue.pop("sink", None)
    if sink is not None:
        graft.set_sink(sink.to(device))
    missing, unexpected = graft.load_state_dict(glue, strict=False)
    missing = [k for k in missing if not k.startswith("part.") and k != "sink"]
    if missing or unexpected:
        raise ValueError("glue mismatch: missing %s, unexpected %s" % (missing[:5], unexpected[:5]))
    for p in graft.parameters():
        p.requires_grad_(False)
    if owned_donor:
        del donor
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    base.eval()
    return base, meta


def params_report(base_params, built_row):
    """The counts the plan predicts, measured: total run per token, trainable, slice, glue."""
    return {"params_base": built_row["params_base"], "slice_params": built_row["slice_params"],
            "trainable_glue": built_row["trainable_glue"], "trainable": built_row["trainable"],
            "params_per_token": built_row["params_per_token"]}


def main(argv=None):
    """python -m wsp.xfam score ARTIFACT_DIR --evals evals.json [--limit N]: load an artifact and score it."""
    import argparse

    import transformers

    parser = argparse.ArgumentParser(prog="wsp.xfam")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sc = sub.add_parser("score", help="load an artifact and score it on the xfam evaluation sets")
    sc.add_argument("path")
    sc.add_argument("--evals", default=os.path.join(DATA_DIR, "evals.json"))
    sc.add_argument("--limit", type=int, default=None)
    sc.add_argument("--device", default=None)
    args = parser.parse_args(argv)
    model, meta = load_artifact(args.path, device=args.device)
    tok = transformers.AutoTokenizer.from_pretrained(meta["host"])
    with open(args.evals) as f:
        evals = json.load(f)
    if args.limit:
        evals = {k: v[:args.limit] for k, v in evals.items()}
    print(json.dumps(evaluate(model, tok, evals)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
