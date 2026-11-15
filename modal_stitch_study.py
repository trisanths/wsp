"""Stitch study: how far does each stitch method cut the penalty, with the sink handled?

The audit found that the ridge stitch fed the donor a garbage attention
sink at position 0. With the donor's mean sink state there, the stitch
penalty into Qwen2.5-Math-1.5B layer 8 fell to 0.37 nats (Qwen2.5-1.5B
base host) and 0.64 nats (Qwen2.5-0.5B-Instruct host). This study trains
the stitch (wsp/stitch.py) on top of that fix and measures what is left.

Donor Qwen/Qwen2.5-Math-1.5B at donor layers 8 and 14 (decoder layer input).
Hosts and host layers (host layer, donor layer):
  Qwen/Qwen2.5-1.5B             (7, 8), (13, 14)   same tokenizer
  Qwen/Qwen2.5-0.5B-Instruct    (7, 8), (12, 14)   same tokenizer
  unsloth/Llama-3.2-1B-Instruct (5, 8), (9, 14)    cross-family

Methods. Every trained stitch starts from the ridge fit on the train texts.
  ridge              the ridge fit, no training
  donor_lm_linear    linear stitch, the donor's own next-token loss
  donor_lm_mlp       MLP stitch, the donor's own next-token loss
  donor_lm_multi     linear map of host layers k-2 and k, donor next-token loss
  mse_linear         linear stitch, standardized MSE on matched states
  downstream_linear  linear stitch, match donor states at end_layer
  downstream_mlp     MLP stitch, match donor states at end_layer
The cross-family host cannot use donor_lm (the tokenizers differ), so it
runs ridge, mse_linear, downstream_linear, and downstream_mlp.

Training: 1000 steps, lr 1e-3 (linear and multi) or 3e-4 (mlp), batch 8,
seq_len 256, seed 0, one seed. Train texts: 1200 GSM8K heal texts and 1200
WikiText-103 paragraphs, interleaved. Eval: 150 more GSM8K heal texts and
150 more WikiText-103 paragraphs, never trained on. The GSM8K search split
and GSM8K test are not used.

Scores, per eval domain (gsm8k, wiki), before and after training:
  penalty           stitched donor loss minus native loss, with the sink
                    (same-tokenizer hosts only)
  penalty_no_sink   the same without the sink, for reference
  downstream_loss   standardized MSE between stitched and native donor
                    states at end_layer = donor layer + 4, with the sink
                    (every host; the main score for the cross-family host)
  r2, r2_dim        pooled and per-dim R2 of the stitch on matched rows
Each row also has train seconds, cell seconds, and stitch params.

The sink is the donor's mean state at position 0 at the donor point,
measured once per cell on the train texts. When train_stitch and
stitch_penalty take a sink argument, the script passes the tensor. Else a
forward hook on the stitch puts the sink at position 0 of its output, as
modal_audit_sink.py does with its layer pre-hook.

Cells: 2 same-tokenizer hosts x 2 layer pairs x 7 methods = 28, plus the
cross-family host x 2 layer pairs x 4 methods = 8. Total 36 cells, one H100
each, run with starmap.

Estimated cost: about 180 H100 GPU minutes in all (range 120 to 240). A
cell loads two float32 models (about 1.5 min), scores before and after
(about 2 min), and trains: donor_lm about 3 to 4 min, downstream about
1.5 min, mse under 1 min, ridge none. With all cells in parallel, wall time
is about 10 to 15 min.

Raw rows go to results/stitch_study_rows.json; a summary table is printed.

    MODAL_PROFILE=trisanth modal run modal_stitch_study.py
"""

import inspect
import json
import os
import time

import modal

app = modal.App("wsp-stitch-study")
volume = modal.Volume.from_name("wsp-models", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch==2.7.1", "transformers==4.57.1", "accelerate", "huggingface_hub", "requests",
        "safetensors", "zstandard", "numpy", "datasets",
    )
    .env({"HF_HOME": "/vol/hf"})
    .add_local_python_source("wsp")
)

DONOR = "Qwen/Qwen2.5-Math-1.5B"
HOSTS = [
    {"tag": "qwen15b", "model": "Qwen/Qwen2.5-1.5B", "same_tokenizer": True, "pairs": [[7, 8], [13, 14]]},
    {"tag": "qwen05b_instruct", "model": "Qwen/Qwen2.5-0.5B-Instruct", "same_tokenizer": True,
     "pairs": [[7, 8], [12, 14]]},
    {"tag": "llama1b_instruct", "model": "unsloth/Llama-3.2-1B-Instruct", "same_tokenizer": False,
     "pairs": [[5, 8], [9, 14]]},
]

METHODS = {
    "ridge": {"objective": None, "kind": "linear"},
    "donor_lm_linear": {"objective": "donor_lm", "kind": "linear"},
    "donor_lm_mlp": {"objective": "donor_lm", "kind": "mlp"},
    "donor_lm_multi": {"objective": "donor_lm", "kind": "multi"},
    "mse_linear": {"objective": "mse", "kind": "linear"},
    "downstream_linear": {"objective": "downstream", "kind": "linear"},
    "downstream_mlp": {"objective": "downstream", "kind": "mlp"},
}
SAME_TOKENIZER_METHODS = ["ridge", "donor_lm_linear", "donor_lm_mlp", "donor_lm_multi", "mse_linear",
                          "downstream_linear", "downstream_mlp"]
CROSS_METHODS = ["ridge", "mse_linear", "downstream_linear", "downstream_mlp"]
MULTI_OFFSETS = [-2, 0]

STEPS = 1000
LR_LINEAR = 1e-3
LR_MLP = 3e-4
BATCH_SIZE = 8
SEQ_LEN = 256
SEED = 0
TRAIN_PER_DOMAIN = 1200
EVAL_PER_DOMAIN = 150
END_OFFSET = 4
# train_stitch scores itself on this many held-out texts per domain. The
# real scores come from score_stitch on all eval texts.
TRAINER_CHECK_TEXTS = 16


def wikitext_paragraphs(count):
    """WikiText-103 train paragraphs longer than 200 characters, in file order, as in modal_sweep.py."""
    from datasets import load_dataset

    data = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="train")
    texts = []
    for row in data:
        text = row["text"].strip()
        if len(text) > 200 and not text.startswith("="):
            texts.append(text)
        if len(texts) >= count:
            break
    return texts


def load_texts():
    """Return (train_texts, eval_sets). eval_sets maps a domain name to its held-out texts."""
    from wsp.data import format_gsm8k, gsm8k_splits

    need = TRAIN_PER_DOMAIN + EVAL_PER_DOMAIN
    search, heal_items = gsm8k_splits(seed=0)
    gsm = []
    for item in heal_items[:need]:
        gsm.append(format_gsm8k(item))
    wiki = wikitext_paragraphs(need)

    # Interleave the domains, so the ridge fit (which reads rows in text
    # order until it has enough) sees both.
    train = []
    for index in range(TRAIN_PER_DOMAIN):
        train.append(gsm[index])
        train.append(wiki[index])
    eval_sets = {"gsm8k": gsm[TRAIN_PER_DOMAIN:need], "wiki": wiki[TRAIN_PER_DOMAIN:need]}
    return train, eval_sets


def native_sink_support():
    """True when train_stitch and stitch_penalty both take a sink argument."""
    from wsp import stitch as stitch_module

    train_params = inspect.signature(stitch_module.train_stitch).parameters
    penalty_params = inspect.signature(stitch_module.stitch_penalty).parameters
    return "sink" in train_params and "sink" in penalty_params


def mean_sink(donor, donor_tokenizer, donor_layer, texts, max_texts=256):
    """The donor's mean state at position 0 of decoder layer donor_layer's input, over texts."""
    from wsp import stitch as stitch_module

    if hasattr(stitch_module, "donor_sink"):
        return stitch_module.donor_sink(donor, donor_tokenizer, donor_layer, texts, max_texts=max_texts)

    # Fallback, as in modal_audit_sink.py: read the layer input at position 0.
    import torch

    from wsp.align import alignment_tokens, layer_input_capture
    from wsp.parts import find_decoder_layers

    _, layers = find_decoder_layers(donor, donor_layer)
    capture = layer_input_capture(layers[donor_layer])
    device = next(donor.parameters()).device
    total = None
    count = 0
    with torch.no_grad():
        for text in texts[:max_texts]:
            ids, _, _ = alignment_tokens(donor_tokenizer, text)
            if ids.shape[1] == 0:
                continue
            state = capture(donor, ids.to(device))[0].float().cpu()
            if total is None:
                total = state.clone()
            else:
                total += state
            count += 1
    return total / count


def add_sink_hook(stitch, sink):
    """Put sink at position 0 of the stitch output. Return the hook handle.

    Only a (batch, seq, d) output changes. The matched-row path (mse and
    r2) passes (rows, d) and is left alone, because those rows never hold
    position 0. The script pads on the right, so position 0 is the first
    token of every row.
    """
    import torch

    def hook(module, args, output):
        if output.dim() != 3:
            return output
        first = sink.to(device=output.device, dtype=output.dtype).view(1, 1, -1)
        first = first.expand(output.shape[0], 1, -1)
        return torch.cat([first, output[:, 1:]], dim=1)

    return stitch.register_forward_hook(hook)


def call_with_sink(fn, kwargs, stitch, sink, use_sink, native):
    """Call fn(**kwargs) with the sink on or off, by argument or by hook."""
    if native:
        if use_sink:
            kwargs["sink"] = sink
        else:
            kwargs["sink"] = False
        return fn(**kwargs)
    # If only one of the two functions takes sink, turn its own sink off,
    # so the hook is the only sink.
    if "sink" in inspect.signature(fn).parameters:
        kwargs["sink"] = False
    handle = None
    if use_sink:
        handle = add_sink_hook(stitch, sink)
    try:
        return fn(**kwargs)
    finally:
        if handle is not None:
            handle.remove()


def matched_r2(stitch, host, host_tokenizer, host_layer, donor, donor_tokenizer, donor_layer, texts):
    """Pooled and per-dim R2 of the stitch on offset-matched rows of texts."""
    import torch

    from wsp import stitch as stitch_module

    host_layers = stitch_module.stitch_host_layers(stitch, host_layer)
    capture = stitch_module.donor_capture_for(donor, donor_layer, None)
    x, y = stitch_module.collect_rows(host, host_tokenizer, host_layers, donor, donor_tokenizer, capture, texts)
    device = next(stitch.parameters()).device
    with torch.no_grad():
        states = stitch_module.split_rows(x.to(device), len(host_layers))
        pred = stitch_module.run_stitch(stitch, states).float().cpu()
    r2, r2_dim = stitch_module.r2_scores(pred, y)
    return {"r2": r2, "r2_dim": r2_dim, "pairs": int(x.shape[0])}


def score_stitch(ctx, stitch):
    """Score the stitch on each eval domain. Return {domain: metrics}."""
    from wsp import stitch as stitch_module

    scores = {}
    for domain, texts in ctx["eval_sets"].items():
        metrics = matched_r2(stitch, ctx["host"], ctx["host_tokenizer"], ctx["host_layer"], ctx["donor"],
                             ctx["donor_tokenizer"], ctx["donor_layer"], texts)

        # Downstream loss at end_layer: train_stitch with no steps scores the
        # stitch as it is. texts are the train texts, so the per-dim std is
        # the one the downstream cells train with.
        kwargs = {"host_model": ctx["host"], "host_tokenizer": ctx["host_tokenizer"], "host_layer": ctx["host_layer"],
                  "donor_model": ctx["donor"], "donor_tokenizer": ctx["donor_tokenizer"],
                  "donor_layer": ctx["donor_layer"], "stitch": stitch, "texts": ctx["train_texts"],
                  "objective": "downstream", "steps": 0, "batch_size": BATCH_SIZE, "seq_len": SEQ_LEN, "seed": SEED,
                  "end_layer": ctx["end_layer"], "eval_texts": texts}
        found = call_with_sink(stitch_module.train_stitch, kwargs, stitch, ctx["sink"], True, ctx["native"])
        metrics["downstream_loss"] = found["before"]["loss"]

        if ctx["same_tokenizer"]:
            kwargs = {"host_model": ctx["host"], "tokenizer": ctx["donor_tokenizer"], "host_layer": ctx["host_layer"],
                      "donor_model": ctx["donor"], "donor_layer": ctx["donor_layer"], "stitch": stitch,
                      "texts": texts, "max_len": SEQ_LEN}
            with_sink = call_with_sink(stitch_module.stitch_penalty, dict(kwargs), stitch, ctx["sink"], True,
                                       ctx["native"])
            without = call_with_sink(stitch_module.stitch_penalty, dict(kwargs), stitch, ctx["sink"], False,
                                     ctx["native"])
            metrics["native_loss"] = with_sink["native_loss"]
            metrics["stitched_loss"] = with_sink["stitched_loss"]
            metrics["penalty"] = with_sink["penalty"]
            metrics["penalty_no_sink"] = without["penalty"]
        scores[domain] = metrics
    return scores


def loss_trace(losses, every=50):
    """Keep every 50th step loss and the mean of the last 50, so the rows stay small."""
    trace = []
    for index in range(0, len(losses), every):
        trace.append(losses[index])
    last = losses[-every:]
    last_mean = None
    if len(last) > 0:
        last_mean = sum(last) / len(last)
    return {"every": every, "trace": trace, "last_mean": last_mean}


def count_params(stitch):
    total = 0
    for param in stitch.parameters():
        total += param.numel()
    return total


def study_cell(cell, host, host_tokenizer, donor, donor_tokenizer, train_texts, eval_sets, steps=STEPS):
    """Run one cell on loaded models. Return its row.

    For a same-tokenizer host, host_tokenizer should be the donor's
    tokenizer, as in modal_round4.py and modal_audit_sink.py.
    """
    import torch

    from wsp import stitch as stitch_module
    from wsp.parts import find_decoder_layers

    start = time.time()
    spec = METHODS[cell["method"]]
    host_layer = cell["host_layer"]
    donor_layer = cell["donor_layer"]
    _, donor_layers = find_decoder_layers(donor, 0)
    end_layer = min(donor_layer + END_OFFSET, len(donor_layers) - 1)
    native = native_sink_support()
    device = next(donor.parameters()).device

    with torch.no_grad():
        sink = mean_sink(donor, donor_tokenizer, donor_layer, train_texts).float().to(device)

    d_host = host.config.hidden_size
    d_donor = donor.config.hidden_size
    config = {"kind": spec["kind"]}
    if spec["kind"] == "multi":
        layers = []
        for offset in MULTI_OFFSETS:
            layers.append(host_layer + offset)
        config["host_layers"] = layers
    torch.manual_seed(SEED)
    stitch = stitch_module.build_stitch(config, d_host, d_donor).to(device)
    fit = stitch_module.ridge_init(stitch, host, host_tokenizer, host_layer, donor, donor_tokenizer, donor_layer,
                                   train_texts)

    ctx = {"host": host, "host_tokenizer": host_tokenizer, "host_layer": host_layer, "donor": donor,
           "donor_tokenizer": donor_tokenizer, "donor_layer": donor_layer, "end_layer": end_layer,
           "same_tokenizer": cell["same_tokenizer"], "train_texts": train_texts, "eval_sets": eval_sets,
           "sink": sink, "native": native}
    before = score_stitch(ctx, stitch)

    lr = LR_LINEAR
    if spec["kind"] == "mlp":
        lr = LR_MLP
    trained = None
    train_seconds = 0.0
    if spec["objective"] is not None:
        check = []
        for domain in eval_sets:
            check.extend(eval_sets[domain][:TRAINER_CHECK_TEXTS])
        kwargs = {"host_model": host, "host_tokenizer": host_tokenizer, "host_layer": host_layer,
                  "donor_model": donor, "donor_tokenizer": donor_tokenizer, "donor_layer": donor_layer,
                  "stitch": stitch, "texts": train_texts, "objective": spec["objective"], "steps": steps, "lr": lr,
                  "batch_size": BATCH_SIZE, "seq_len": SEQ_LEN, "seed": SEED, "eval_texts": check,
                  "log_every": 100}
        if spec["objective"] == "downstream":
            kwargs["end_layer"] = end_layer
        trained = call_with_sink(stitch_module.train_stitch, kwargs, stitch, sink, True, native)
        train_seconds = trained["seconds"]
        after = score_stitch(ctx, stitch)
    else:
        after = before

    row = {"cell": cell["id"], "host": cell["host"], "host_tag": cell["host_tag"], "donor": DONOR,
           "same_tokenizer": cell["same_tokenizer"], "method": cell["method"], "kind": spec["kind"],
           "objective": spec["objective"], "host_layers": stitch_module.stitch_host_layers(stitch, host_layer),
           "host_layer": host_layer, "donor_layer": donor_layer, "end_layer": end_layer, "steps": steps, "lr": lr,
           "batch_size": BATCH_SIZE, "seq_len": SEQ_LEN, "seed": SEED, "params": count_params(stitch),
           "ridge_fit": fit, "sink_mode": "argument" if native else "hook", "sink_norm": float(sink.norm()),
           "before": before, "after": after, "train_seconds": train_seconds}
    if trained is not None:
        row["trainer_check"] = {"before": trained.get("before"), "after": trained.get("after")}
        row["losses"] = loss_trace(trained.get("losses", []))
    row["cell_seconds"] = round(time.time() - start, 1)
    return row


@app.function(image=image, volumes={"/vol": volume}, gpu="H100", memory=98304, timeout=5400)
def run_cell(cell):
    import torch
    import transformers

    start = time.time()
    train_texts, eval_sets = load_texts()
    donor = transformers.AutoModelForCausalLM.from_pretrained(DONOR, dtype=torch.float32).to("cuda").eval()
    donor_tokenizer = transformers.AutoTokenizer.from_pretrained(DONOR)
    host = transformers.AutoModelForCausalLM.from_pretrained(cell["host"], dtype=torch.float32).to("cuda").eval()
    if cell["same_tokenizer"]:
        host_tokenizer = donor_tokenizer
    else:
        host_tokenizer = transformers.AutoTokenizer.from_pretrained(cell["host"])
    load_seconds = round(time.time() - start, 1)
    row = study_cell(cell, host, host_tokenizer, donor, donor_tokenizer, train_texts, eval_sets)
    row["load_seconds"] = load_seconds
    print(json.dumps({"cell": row["cell"], "cell_seconds": row["cell_seconds"]}))
    return row


def build_cells():
    cells = []
    for host in HOSTS:
        methods = CROSS_METHODS
        if host["same_tokenizer"]:
            methods = SAME_TOKENIZER_METHODS
        for host_layer, donor_layer in host["pairs"]:
            for method in methods:
                cell_id = f"{host['tag']}_h{host_layer}_d{donor_layer}_{method}"
                cells.append({"id": cell_id, "host": host["model"], "host_tag": host["tag"],
                              "same_tokenizer": host["same_tokenizer"], "host_layer": host_layer,
                              "donor_layer": donor_layer, "method": method})
    return cells


def fmt(value):
    if value is None:
        return "-"
    return f"{value:.3f}"


def before_after(row, domain, key):
    before = row["before"].get(domain, {}).get(key)
    after = row["after"].get(domain, {}).get(key)
    return f"{fmt(before)} -> {fmt(after)}"


def print_summary(rows):
    """Print one markdown table per (host, layer pair)."""
    groups = []
    seen = {}
    for row in rows:
        key = (row["host_tag"], row["host_layer"], row["donor_layer"])
        if key not in seen:
            seen[key] = []
            groups.append(key)
        seen[key].append(row)

    for key in groups:
        group = seen[key]
        host_tag, host_layer, donor_layer = key
        first = group[0]
        print()
        print(f"## {first['host']} layer {host_layer} -> {DONOR} layer {donor_layer} "
              f"(end_layer {first['end_layer']})")
        print()
        if first["same_tokenizer"]:
            print("| Method | Params | Penalty gsm8k | Penalty wiki | No-sink gsm8k after | r2_dim gsm8k after "
                  "| Downstream gsm8k | Train s |")
            print("|---|---|---|---|---|---|---|---|")
            for row in group:
                print(f"| {row['method']} | {row['params']} | {before_after(row, 'gsm8k', 'penalty')} "
                      f"| {before_after(row, 'wiki', 'penalty')} "
                      f"| {fmt(row['after']['gsm8k'].get('penalty_no_sink'))} "
                      f"| {fmt(row['after']['gsm8k'].get('r2_dim'))} "
                      f"| {before_after(row, 'gsm8k', 'downstream_loss')} | {row['train_seconds']} |")
        else:
            print("| Method | Params | Downstream gsm8k | Downstream wiki | r2_dim gsm8k after | Train s |")
            print("|---|---|---|---|---|---|")
            for row in group:
                print(f"| {row['method']} | {row['params']} | {before_after(row, 'gsm8k', 'downstream_loss')} "
                      f"| {before_after(row, 'wiki', 'downstream_loss')} "
                      f"| {fmt(row['after']['gsm8k'].get('r2_dim'))} | {row['train_seconds']} |")


@app.local_entrypoint()
def main():
    cells = build_cells()
    print(f"{len(cells)} cells")
    rows = []
    failures = []
    args = []
    for cell in cells:
        args.append((cell,))
    for cell, row in zip(cells, run_cell.starmap(args, return_exceptions=True)):
        if isinstance(row, Exception):
            print("cell failed:", cell["id"], repr(row)[:400])
            failures.append({"cell": cell["id"], "error": repr(row)[:2000]})
            continue
        rows.append(row)
    os.makedirs("results", exist_ok=True)
    with open("results/stitch_study_rows.json", "w") as f:
        json.dump({"donor": DONOR, "cells": len(cells), "rows": rows, "failures": failures}, f, indent=2)
    print_summary(rows)
    print()
    print(f"{len(rows)} rows, {len(failures)} failures, written to results/stitch_study_rows.json")
