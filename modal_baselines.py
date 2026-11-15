"""Baselines that composed models must beat: full fine-tuning and donor ceilings.

Known numbers on the GSM8K search split (raw prompt): Llama-1B host 23 to
25%, Llama-1B LoRA r64 lr 5e-5 46.7%, gate matched LoRA 44%,
Qwen2.5-Math-1.5B base 45%. This app adds the missing rows.

Arms:
1. Full fine-tune of unsloth/Llama-3.2-1B-Instruct on
   wsp_eval.splits.heal_texts("gsm8k"): 300 steps, batch 8, seq 384, AdamW
   (weight decay 0) at lr 1e-5 and 3e-5, linear warmup over 30 steps and
   then a constant lr, bf16 autocast on float32 weights, grad clip 1.0,
   seeds 0, 1, 2. Namespace full_ft. Each model is saved in bf16 to the
   volume at /vol/baselines/<cell name>.
2. The same for Qwen/Qwen2.5-0.5B-Instruct.
3. Donor ceilings, no training, one run each (namespace ceiling):
   Qwen2.5-Math-1.5B-Instruct, LFM2-350M-Math, LFM2.5-1.2B-Thinking,
   LFM2.5-1.2B-Instruct, Qwen2.5-1.5B-Instruct, Llama-3.2-3B-Instruct.
   Every cell is scored with vLLM, the harness backend for plain models.
   vLLM 0.11 registers Lfm2ForCausalLM, so it serves the Liquid models.
   If vLLM fails on a model, that cell is scored again with the hf
   backend, and the row records the backend it used and the vLLM error.
   The thinking model gets the same raw prompt and 256 new tokens as
   every other model, so a long think block can cut off its answer.
4. Decode tokens per second for every cell: Hugging Face generate, batch 1,
   128 new tokens after a short warmup, bf16, on an H100 (the same GPU
   and prompt as rounds 3 and 4).

Scores come from Person B's harness: eval(model_dir, "search", ["gsm8k"],
namespace=...). The image is modal_eval.py's image, so the pins match the
team's harness. vLLM runs with max_model_len 2048 (WSP_VLLM_MAX_LEN):
the harness default of 8192 is longer than the 4096 positions of
Qwen2.5-Math-1.5B-Instruct, and vLLM refuses that. A GSM8K prompt plus
256 new tokens is far below 2048. A trained model dir holds a program.json with its seed, so
wsp_eval.claims.compare can pair these rows by program_seed.

Each cell uses its own harness root on the volume
(/vol/eval/cells/baselines/<cell name>), because concurrent containers
that write one shared file on a Modal volume overwrite each other. After
all cells finish, one CPU call charges the shared budget at /vol/eval and
appends each cell's experiments.jsonl line there.

Cells: 18 (12 full fine-tunes, 6 ceilings), each on one H100, run with
starmap. A failed cell does not stop the others. A trained model that is
already on the volume is not trained again on a rerun, unless its
program.json differs from the current settings.

Estimated GPU time: about 6 minutes per fine-tune cell (start, 300 steps,
save, decode check, vLLM load and 200 generations) and about 4 minutes per
ceiling cell, so about 100 H100 minutes in all (about $7 at $4 per hour).
Wall clock is about 10 minutes with every cell in parallel.

Search-split numbers are interim checks, not results.

    MODAL_PROFILE=trisanth modal run modal_baselines.py
    MODAL_PROFILE=trisanth modal run modal_baselines.py --groups "ceiling"
"""

import json
import os
import time

import modal

from modal_eval import image as eval_image

app = modal.App("wsp-baselines")
volume = modal.Volume.from_name("wsp-models", create_if_missing=True)

# Person B's image, plus modal_eval.py itself so this module can import it
# inside the container.
image = eval_image.add_local_file("modal_eval.py", "/root/modal_eval.py")

LLAMA_HOST = "unsloth/Llama-3.2-1B-Instruct"
QWEN_HOST = "Qwen/Qwen2.5-0.5B-Instruct"
FT_HOSTS = {"llama1b_ft": ("llama1b", LLAMA_HOST), "qwen05_ft": ("qwen05", QWEN_HOST)}
FT_LRS = ["1e-5", "3e-5"]
FT_SEEDS = [0, 1, 2]
FT_STEPS = 300
FT_BATCH = 8
FT_SEQ = 384
FT_WARMUP = 30

# (short name, model id, harness backend)
CEILINGS = [
    ("qwen_math_15b_instruct", "Qwen/Qwen2.5-Math-1.5B-Instruct", "vllm"),
    ("lfm2_350m_math", "LiquidAI/LFM2-350M-Math", "vllm"),
    ("lfm25_12b_thinking", "LiquidAI/LFM2.5-1.2B-Thinking", "vllm"),
    ("lfm25_12b_instruct", "LiquidAI/LFM2.5-1.2B-Instruct", "vllm"),
    ("qwen25_15b_instruct", "Qwen/Qwen2.5-1.5B-Instruct", "vllm"),
    ("llama32_3b_instruct", "unsloth/Llama-3.2-3B-Instruct", "vllm"),
]

MODEL_ROOT = "/vol/baselines"
CELL_ROOT = "/vol/eval/cells/baselines"
SHARED_ROOT = "/vol/eval"
ROWS_PATH = "results/baselines_rows.json"

# vLLM context length for every cell. The harness default (8192) is longer
# than Qwen2.5-Math-1.5B-Instruct's 4096 positions, and vLLM refuses a
# max_model_len above the model's own limit.
VLLM_MAX_LEN = "2048"

# Earlier search-split numbers, printed next to the summary for context.
KNOWN = [
    ("Llama-1B host", "23 to 25%"),
    ("Llama-1B LoRA r64, lr 5e-5 (round 4)", "46.7%"),
    ("Llama-1B gate matched LoRA (gate, seed 0)", "44.0%"),
    ("Qwen2.5-Math-1.5B base (gate, seed 0)", "45.0%"),
]


def build_cells(groups):
    """Return a list of cell tuples: (name, kind, model, lr, seed, namespace, backend)."""
    cells = []
    for group in groups:
        if group in FT_HOSTS:
            tag, host = FT_HOSTS[group]
            for lr in FT_LRS:
                for seed in FT_SEEDS:
                    name = "full_ft_" + tag + "_lr" + lr + "_s" + str(seed)
                    cells.append((name, "full_ft", host, lr, seed, "full_ft", "vllm"))
        elif group == "ceiling":
            for short, model_id, backend in CEILINGS:
                cells.append(("ceiling_" + short, "ceiling", model_id, None, None, "ceiling", backend))
        else:
            raise ValueError("unknown group " + repr(group))
    return cells


def arm_label(row):
    if row["kind"] == "full_ft":
        return "full_ft " + row["model"] + " lr " + row["lr"]
    return "ceiling " + row["model"]


# ---------------------------------------------------------------------------
# GPU work
# ---------------------------------------------------------------------------

def free_gpu():
    import gc

    import torch

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def train_full_ft(host, lr, seed, out_dir):
    """Full fine-tune host on the GSM8K heal texts and save it in bf16 to out_dir. Return train stats."""
    import torch
    import transformers

    from wsp_eval.splits import heal_texts

    texts = heal_texts("gsm8k")
    torch.manual_seed(seed)
    tokenizer = transformers.AutoTokenizer.from_pretrained(host)
    # Right padding for training, so real tokens keep positions 0..n-1.
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = transformers.AutoModelForCausalLM.from_pretrained(host, dtype=torch.float32).to("cuda")
    model.config.use_cache = False
    model.train()

    params = []
    trainable = 0
    for param in model.parameters():
        if param.requires_grad:
            params.append(param)
            trainable += param.numel()

    optimizer = torch.optim.AdamW(params, lr=float(lr), weight_decay=0.0)

    def lr_factor(step):
        # Linear warmup over FT_WARMUP steps, then constant.
        if step < FT_WARMUP:
            return (step + 1) / FT_WARMUP
        return 1.0

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)

    # The same data order rule as wsp.heal.heal: a seeded permutation, refilled when used up.
    generator = torch.Generator()
    generator.manual_seed(seed)
    order = []
    losses = []
    start = time.time()
    for step in range(FT_STEPS):
        batch = []
        while len(batch) < FT_BATCH:
            if len(order) == 0:
                order = torch.randperm(len(texts), generator=generator).tolist()
            batch.append(texts[order.pop()])
        enc = tokenizer(batch, truncation=True, max_length=FT_SEQ, padding=True, return_tensors="pt")
        input_ids = enc["input_ids"].to("cuda")
        attention_mask = enc["attention_mask"].to("cuda")
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100

        optimizer.zero_grad()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        out.loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        optimizer.step()
        scheduler.step()

        loss = float(out.loss.detach())
        losses.append(loss)
        if (step + 1) % 25 == 0:
            print("full_ft step", step + 1, "/", FT_STEPS, "loss", round(loss, 4), flush=True)
    train_seconds = round(time.time() - start, 1)

    model.config.use_cache = True
    model.to(torch.bfloat16)
    os.makedirs(out_dir, exist_ok=True)
    model.save_pretrained(out_dir, safe_serialization=True)
    # Save a fresh tokenizer, so the padding change above does not reach the eval.
    transformers.AutoTokenizer.from_pretrained(host).save_pretrained(out_dir)

    last = losses[-10:]
    return {"trainable_params": trainable, "train_seconds": train_seconds, "loss_first": losses[0],
            "loss_last10": sum(last) / len(last), "losses": losses}


def decode_speed(model_path, tokens=128):
    """Decode tokens per second: HF generate, batch 1, bf16, after a short warmup."""
    import torch
    import transformers

    tokenizer = transformers.AutoTokenizer.from_pretrained(model_path)
    model = transformers.AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.bfloat16).to("cuda").eval()
    ids = tokenizer("Question: What is 12 times 7?\nAnswer:", return_tensors="pt").input_ids.to("cuda")
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id
    with torch.no_grad():
        model.generate(ids, max_new_tokens=16, min_new_tokens=16, do_sample=False, pad_token_id=pad_id)
    torch.cuda.synchronize()
    start = time.time()
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=tokens, min_new_tokens=tokens, do_sample=False,
                             pad_token_id=pad_id)
    torch.cuda.synchronize()
    speed = (out.shape[1] - ids.shape[1]) / (time.time() - start)
    return round(speed, 1)


def last_experiments_line(root):
    path = os.path.join(root, "experiments.jsonl")
    if not os.path.exists(path):
        return None
    last = None
    with open(path) as f:
        for line in f:
            if line.strip():
                last = json.loads(line)
    return last


def cell_env(name, backend):
    """Set the harness environment for one cell and return its harness root."""
    import shutil

    # Run the vLLM engine in this process. CUDA is already in use here
    # (training and the decode check), and a forked engine would fail.
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    os.environ["WSP_VLLM_MAX_LEN"] = VLLM_MAX_LEN
    os.environ["WSP_BACKEND"] = backend
    cell_root = os.path.join(CELL_ROOT, name)
    os.environ["WSP_ROOT"] = cell_root
    os.makedirs(os.path.join(cell_root, "work"), exist_ok=True)
    reference = os.path.join(SHARED_ROOT, "work", "reference_search.json")
    if os.path.exists(reference):
        shutil.copy(reference, os.path.join(cell_root, "work", "reference_search.json"))
    return cell_root


def score_cell(model_path, namespace, backend, name):
    """Score with the harness. If vLLM fails, score again with hf. Return (result, backend used, vLLM error)."""
    from wsp_eval import harness

    try:
        result = harness.evaluate(model_path, "search", ["gsm8k"], namespace=namespace, backend=backend,
                                  note="baselines " + name)
        return result, backend, None
    except Exception as error:
        if backend != "vllm":
            raise
        vllm_error = repr(error)[:2000]
    print("vLLM failed on", model_path, "so scoring with hf:", vllm_error[:400], flush=True)
    free_gpu()
    os.environ["WSP_BACKEND"] = "hf"
    result = harness.evaluate(model_path, "search", ["gsm8k"], namespace=namespace, backend="hf",
                              note="baselines " + name + " (hf after vLLM failed)")
    return result, "hf", vllm_error


def full_ft_program(host, lr, seed):
    """The settings saved as program.json next to a full fine-tuned model."""
    return {"host": host, "seed": seed, "baseline": "full_ft", "lr": float(lr), "steps": FT_STEPS,
            "batch_size": FT_BATCH, "seq_len": FT_SEQ, "warmup_steps": FT_WARMUP,
            "data": "wsp_eval.splits.heal_texts('gsm8k')"}


def trained_already(model_path, program):
    """True if model_path holds a finished fine-tune made with these settings."""
    marker = os.path.join(model_path, "wsp_train.json")
    program_path = os.path.join(model_path, "program.json")
    if not os.path.exists(marker) or not os.path.exists(program_path):
        return False
    with open(program_path) as f:
        saved = json.load(f)
    return saved == program


@app.function(image=image, volumes={"/vol": volume}, gpu="H100", memory=65536, timeout=3600,
              single_use_containers=True)
def run_cell(name, kind, model, lr, seed, namespace, backend):
    """One cell: train (full_ft only), measure decode speed, score with the team's harness."""
    cell_root = cell_env(name, backend)

    row = {"cell": name, "kind": kind, "model": model, "lr": lr, "seed": seed, "namespace": namespace,
           "backend": backend}
    model_path = model
    if kind == "full_ft":
        model_path = os.path.join(MODEL_ROOT, name)
        marker = os.path.join(model_path, "wsp_train.json")
        program = full_ft_program(model, lr, seed)
        if trained_already(model_path, program):
            print("model already trained, scoring only:", model_path, flush=True)
            with open(marker) as f:
                train = json.load(f)
        else:
            if os.path.exists(marker):
                # Settings changed since the last run: drop the old marker so a crash cannot reuse it.
                os.remove(marker)
            train = train_full_ft(model, lr, seed, model_path)
            # program.json carries the seed, so claims.compare can pair rows by program_seed.
            with open(os.path.join(model_path, "program.json"), "w") as f:
                json.dump(program, f, indent=2)
            with open(marker, "w") as f:
                json.dump(train, f)
            volume.commit()
        free_gpu()
        losses = train.get("losses", [])
        row["trainable_params"] = train["trainable_params"]
        row["train_seconds"] = train["train_seconds"]
        row["loss_first"] = train["loss_first"]
        row["loss_last10"] = train["loss_last10"]
        row["losses_every25"] = losses[24::25]
    row["model_path"] = model_path

    row["decode_tok_s"] = decode_speed(model_path)
    free_gpu()

    start = time.time()
    result, used_backend, vllm_error = score_cell(model_path, namespace, backend, name)
    row["backend"] = used_backend
    if vllm_error is not None:
        row["vllm_error"] = vllm_error
    row["eval_seconds"] = round(time.time() - start, 1)
    row["score"] = result["tasks"]["gsm8k"]["score"]
    row["correct"] = result["tasks"]["gsm8k"]["correct"]
    row["n"] = result["tasks"]["gsm8k"]["n"]
    row["label"] = result["label"]
    row["program_seed"] = result["program_seed"]
    row["generations_run"] = result["generations_run"]
    row["eval_result"] = result
    row["experiments_line"] = last_experiments_line(cell_root)
    volume.commit()
    print("[baselines]", name, "score", row["score"], "decode tok/s", row["decode_tok_s"], flush=True)
    return row


# ---------------------------------------------------------------------------
# Shared budget and log (one container, so no concurrent writes)
# ---------------------------------------------------------------------------

@app.function(image=image, volumes={"/vol": volume}, cpu=2, memory=4096, timeout=600)
def budget_preflight(needs):
    """Raise if a namespace has a limit and would go over it. needs: {namespace: eval calls}."""
    os.environ["WSP_ROOT"] = SHARED_ROOT
    from wsp_eval import budget

    report = {}
    for namespace, count in needs.items():
        entry = budget.usage(namespace)
        report[namespace] = entry
        if entry["limit"] is not None and entry["eval_calls"] + count > entry["limit"]:
            raise budget.BudgetExhausted("namespace " + repr(namespace) + " has used " + str(entry["eval_calls"]) +
                                         " of " + str(entry["limit"]) + " eval calls; this run needs " +
                                         str(count))
    return report


@app.function(image=image, volumes={"/vol": volume}, cpu=2, memory=4096, timeout=600)
def record_shared(entries):
    """Charge the shared budget and append to the shared experiments.jsonl, once per scored cell."""
    os.environ["WSP_ROOT"] = SHARED_ROOT
    from wsp_eval import budget, config

    volume.reload()
    path = config.experiments_path()
    for entry in entries:
        usage = budget.charge(entry["namespace"], "search", entry["n"], entry["generations_run"])
        line = entry["experiments_line"]
        if line is None:
            continue
        line = dict(line)
        line["eval_count"] = usage["eval_calls"]
        line["note"] = line.get("note", "") + " (scored in cell root " + entry["cell"] + ")"
        with open(path, "a") as f:
            f.write(json.dumps(line) + "\n")
    volume.commit()
    return budget.usage()


# ---------------------------------------------------------------------------
# Local entrypoint
# ---------------------------------------------------------------------------

def mean_sd(values):
    if len(values) == 0:
        return None, None
    mean = sum(values) / len(values)
    if len(values) < 2:
        return mean, None
    total = 0.0
    for value in values:
        total += (value - mean) ** 2
    return mean, (total / (len(values) - 1)) ** 0.5


def summarize(rows):
    groups = {}
    order = []
    for row in rows:
        label = arm_label(row)
        if label not in groups:
            groups[label] = []
            order.append(label)
        groups[label].append(row)
    summary = []
    for label in order:
        scores = []
        speeds = []
        seeds = []
        backends = []
        for row in groups[label]:
            scores.append(row["score"])
            speeds.append(row["decode_tok_s"])
            seeds.append(row["seed"])
            if row["backend"] not in backends:
                backends.append(row["backend"])
        score_mean, score_sd = mean_sd(scores)
        speed_mean, _ = mean_sd(speeds)
        summary.append({"arm": label, "n_runs": len(scores), "seeds": seeds, "score_mean": score_mean,
                        "score_sd": score_sd, "scores": scores, "decode_tok_s_mean": speed_mean,
                        "backends": backends})
    return summary


def load_old_rows():
    if not os.path.exists(ROWS_PATH):
        return []
    with open(ROWS_PATH) as f:
        data = json.load(f)
    return data.get("rows", [])


@app.local_entrypoint()
def main(groups: str = "llama1b_ft qwen05_ft ceiling"):
    cells = build_cells(groups.split())
    needs = {}
    for cell in cells:
        namespace = cell[5]
        if namespace not in needs:
            needs[namespace] = 0
        needs[namespace] += 1
    print("budget before:", json.dumps(budget_preflight.remote(needs)))

    rows = []
    failed = []
    results = run_cell.starmap(cells, return_exceptions=True)
    for cell, row in zip(cells, results):
        if isinstance(row, Exception):
            print("cell failed:", cell[0], repr(row)[:400])
            failed.append({"cell": cell[0], "error": repr(row)[:2000]})
            continue
        rows.append(row)
        print(json.dumps({"cell": row["cell"], "score": row["score"], "decode_tok_s": row["decode_tok_s"],
                          "loss_last10": row.get("loss_last10")}))

    entries = []
    for row in rows:
        entries.append({"cell": row["cell"], "namespace": row["namespace"], "n": row["n"],
                        "generations_run": row["generations_run"], "experiments_line": row["experiments_line"]})
    if len(entries) > 0:
        print("budget after:", json.dumps(record_shared.remote(entries)))

    # Keep rows from earlier runs of other cells; a rerun of a cell replaces its row.
    new_cells = {}
    for row in rows:
        new_cells[row["cell"]] = True
    merged = []
    for row in load_old_rows():
        if row["cell"] not in new_cells:
            merged.append(row)
    for row in rows:
        merged.append(row)

    summary = summarize(merged)
    os.makedirs("results", exist_ok=True)
    with open(ROWS_PATH, "w") as f:
        json.dump({"label": "SEARCH SPLIT - for search only, not a result", "summary": summary,
                   "failed": failed, "rows": merged}, f, indent=2)
    print("saved", ROWS_PATH)

    print()
    print("GSM8K search split (200 items), SEARCH SPLIT - not a result")
    print("%-62s %5s %14s %10s  %s" % ("arm", "runs", "score (sd)", "tok/s", "backend"))
    for item in summary:
        sd_text = "  -  "
        if item["score_sd"] is not None:
            sd_text = "%.1f" % (100 * item["score_sd"])
        print("%-62s %5d %7.1f (%s) %10.1f  %s" % (item["arm"], item["n_runs"], 100 * item["score_mean"], sd_text,
                                                    item["decode_tok_s_mean"], " ".join(item["backends"])))
    print()
    print("Earlier numbers on the same split:")
    for label, value in KNOWN:
        print("  %-50s %s" % (label, value))
    if len(failed) > 0:
        print()
        names = []
        for item in failed:
            names.append(item["cell"])
        print(len(failed), "cell(s) failed:", ", ".join(names))
