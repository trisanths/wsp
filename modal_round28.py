"""Round 28: three business requests on one method, skill removal from Qwen3-30B-A3B-Instruct-2507.

Each request keeps the experts that matter for one skill (wsp/slim.py: rank experts by saliency on skill text,
keep the top N per layer, optionally heal). The wording of the three requests is part of the demo:

  clinic      "I run a clinic. I want a private assistant that answers medical questions as well as Qwen3-30B,
               on a smaller GPU we can keep on site."
              Skill medicine. Scores: MedMCQA validation 1000, MMLU medical subjects (clinical_knowledge,
              college_medicine, professional_medicine, anatomy, medical_genetics; TEST split).
  accounting  "We do bookkeeping and tax prep. I need a model that's great at word-problem arithmetic and
               finance math, and cheap to run."
              Skill math. Scores: GSM8K test (the seeded 500), MATH-500, MMLU professional_accounting and
              high_school_statistics (TEST).
  law_firm    "We're a small law firm. We want a model that knows law well for first-pass research and runs
               on our own hardware."
              Skill law. Scores: MMLU professional_law, jurisprudence, international_law (TEST).

Every request also scores general MMLU 1000 minus the subjects it scores as its domain, so a row shows what was
removed (wsp.skills_eval.mmlu_general_items). The domain and the general set never share an item.

Cells (13 in all; build_cells() is the list and the tests count it):
  full_30b                                   the full model, scored on every benchmark above, once
  per request (3 x 4):
    <request>_sal64                          keep 64 of 128 experts per layer by saliency, no heal
    <request>_sal48                          keep 48 by saliency, no heal
    <request>_sal32_heal500                  keep 32 by saliency, 500-step heal
    <request>_random48_s0                    keep 48 chosen at random (seed 0), no heal: the control
One seed per cell. The saliency ranking is deterministic given the ranking texts (the texts are in a fixed
seeded order and the ranking is an argsort), so the saliency cells have no seed to vary. The control is one
random draw.

Data (owner rule of 2026-11-15: ranking and heal may read benchmark TRAIN splits; scores come only from test
items). Per request the ranking text is 256 texts from the wsp.healdata pool of the skill plus 256 from the
benchmark train splits (wsp/benchtrain.py): MedMCQA train for clinic, GSM8K train and MATH train for
accounting, CaseHOLD train for law_firm. Texts that hold a question of any scoring item are dropped with
wsp.slim.drop_overlap. The heal reads 2000 more pool texts and 2000 more train-split texts, 4000 in all
(500 steps x batch 8), and its loss is the KL to the full model's logits only (lm_weight 0). This differs from
rounds 19 and 19c, whose heal also had an LM loss on the pool text. The reason is that the train-split texts
hold gold answers in a foreign style, and LESSONS.md forbids fitting them with an LM loss. Nothing here
compares the two heal losses.

The kept model at keep 64 is saved with wsp.slim.save_slim to /vol/slim_models/round28/<request>_64 for
serving. Each Modal workspace has its own volume, so the directory exists only in the workspace that ran.

Commands (always smoke first; LESSONS.md):

    modal run modal_round28.py --check                 # CPU only: reads the pools and train splits, prints counts
    modal run modal_round28.py --estimate-only         # lists the cells, fails on zero
    modal run --detach modal_round28.py --smoke        # law keep 64 plus a 4-step heal, 40 items, small ranking
    modal run --detach modal_round28.py                # all 13 cells, one container per request plus the full model
    modal run --detach modal_round28.py --only clinic_sal64,clinic_sal48 --tag redo

Rows go to /vol/results/round28/<cell>.json (written by the GPU function) and to results/round28_<tag>.json
here after each cell. Every printed score line starts with the cell name. The cache keys carry V28, the
version of the texts and the ranking, so a change to either needs a new version string.
"""

import hashlib
import json
import os
import random
import shutil
import threading
import time

import modal

app = modal.App("wsp-round28")
volume = modal.Volume.from_name("wsp-models", create_if_missing=True)

V28 = "v28a"


def safe_reload():
    """Reload the volume to see other containers' files. Open files block a reload; then skip it."""
    try:
        volume.reload()
    except Exception as error:
        print("volume reload skipped:", repr(error)[:200])


# The pip list of modal_round19c.py, which ran.
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch==2.7.1", "transformers==4.57.1", "accelerate", "huggingface_hub", "requests",
        "safetensors", "zstandard", "numpy", "datasets", "sentencepiece", "protobuf",
    )
    .env({"HF_HOME": "/vol/hf", "HF_HUB_DISABLE_XET": "1"})
    .add_local_python_source("wsp")
)

MODEL_ID = "Qwen/Qwen3-30B-A3B-Instruct-2507"
EXPERTS = 128
GSM8K_COUNT = 500
GENERAL_COUNT = 1000

RANK_POOL = 256
RANK_BENCH = 256
HEAL_POOL = 2000
HEAL_BENCH = 2000
RANK_LENGTH = 384
HEAL_STEPS = 500
POOL_PAIRS = RANK_POOL + HEAL_POOL + 600
BENCH_PAIRS = RANK_BENCH + HEAL_BENCH + 600
HEAL = {"batch_size": 8, "seq_len": 384, "lora_rank": 16, "lora_lr": 2e-4, "router_lr": 2e-5, "warmup": 20,
        "lm_weight": 0.0}

KEPT_ROOT = "/vol/slim_models/round28"
CELL_DIR = "/vol/results/round28"
STATS_DIR = "/vol/slim/round28"
PAIRS_DIR = "/vol/slim/round28/pairs"

SAMPLE_TOKENS = 220

REQUESTS = {
    "clinic": {
        "skill": "medicine", "group": "medical",
        "text": "I run a clinic. I want a private assistant that answers medical questions as well as Qwen3-30B, "
                "on a smaller GPU we can keep on site.",
        "bench": ["medmcqa", "mmlu_medical", "mmlu_general_ex_medical"],
        "prompts": [
            "A 58-year-old man has crushing chest pain that spreads to his left arm, sweating, and ST elevation in "
            "leads II, III, and aVF. What is the most likely diagnosis, which artery is involved, and what are the "
            "first three steps of management?",
            "A patient with type 2 diabetes and normal kidney function needs a first drug. Which one is the usual "
            "first choice, and what should the patient be told about side effects?",
        ],
    },
    "accounting": {
        "skill": "math", "group": "accounting",
        "text": "We do bookkeeping and tax prep. I need a model that's great at word-problem arithmetic and finance "
                "math, and cheap to run.",
        "bench": ["gsm8k", "math500", "mmlu_accounting", "mmlu_general_ex_accounting"],
        "prompts": [
            "A shop bought 48 units at $12.50 each and paid a flat $60 delivery fee on the whole purchase. It sold "
            "36 of the units at $18 each. What is its gross profit on the units sold, if the delivery fee is "
            "spread evenly over all 48 units?",
            "An invoice for $4,200 has terms 2/10, net 30. How much is paid if it is settled on day 8, and what is "
            "the approximate annualized cost of skipping the discount?",
        ],
    },
    "law_firm": {
        "skill": "law", "group": "law",
        "text": "We're a small law firm. We want a model that knows law well for first-pass research and runs on "
                "our own hardware.",
        "bench": ["mmlu_law", "mmlu_general_ex_law"],
        "prompts": [
            "In a contract dispute, explain the difference between a condition precedent and a covenant, and what "
            "happens if each one is breached.",
            "State the rule against perpetuities in one paragraph, and say what a savings clause does.",
        ],
    },
}
ALL_BENCH = ["medmcqa", "mmlu_medical", "mmlu_general_ex_medical", "gsm8k", "math500", "mmlu_accounting",
             "mmlu_general_ex_accounting", "mmlu_law", "mmlu_general_ex_law"]
# The score that names each request's domain, as the mean of its benchmarks.
DOMAIN = {"clinic": ["medmcqa", "mmlu_medical"], "accounting": ["gsm8k", "math500", "mmlu_accounting"],
          "law_firm": ["mmlu_law"]}
GENERAL = {"clinic": "mmlu_general_ex_medical", "accounting": "mmlu_general_ex_accounting",
           "law_firm": "mmlu_general_ex_law"}

SIZES_NO_HEAL = [64, 48]
HEAL_SIZE = 32
RANDOM_SIZE = 48
SMOKE_CELLS = ["law_firm_sal64", "law_firm_sal32_heal500"]
SMOKE_LIMIT = 40
SMOKE_HEAL_STEPS = 4
SMOKE_RANK = 16
GPU = "H200"


# ---------------------------------------------------------------- cells


def spec(request, rule, keep=None, seed=0, heal=0, keep_model=False):
    return {"request": request, "rule": rule, "keep": keep, "seed": seed, "heal": heal, "keep_model": keep_model}


def build_cells():
    """{name: spec}, in run order. The full model first."""
    cells = {"full_30b": spec(None, None)}
    for request in REQUESTS:
        for n in SIZES_NO_HEAL:
            cells[f"{request}_sal{n}"] = spec(request, "saliency", keep=n, keep_model=(n == 64))
        cells[f"{request}_sal{HEAL_SIZE}_heal{HEAL_STEPS}"] = spec(request, "saliency", keep=HEAL_SIZE,
                                                                   heal=HEAL_STEPS)
        cells[f"{request}_random{RANDOM_SIZE}_s0"] = spec(request, "random", keep=RANDOM_SIZE, seed=0)
    return cells


CELLS = build_cells()


def select_cells(only=None, requests=None, smoke=False, cells=None):
    """The cell names this launch runs, in run order. An empty result is an error, not a quiet exit."""
    if cells is None:
        cells = CELLS
    if smoke:
        return list(SMOKE_CELLS)
    names = list(cells)
    if only:
        wanted = [item.strip() for item in only.split(",") if item.strip()]
        for name in wanted:
            if name not in cells:
                raise ValueError(f"unknown cell {name!r}; the cells are {', '.join(cells)}")
        names = [name for name in names if name in wanted]
    if requests:
        keep = [item.strip() for item in requests.split(",") if item.strip()]
        for item in keep:
            if item not in REQUESTS:
                raise ValueError(f"unknown request {item!r}; the requests are {', '.join(REQUESTS)}")
        names = [name for name in names if cells[name]["request"] is None or cells[name]["request"] in keep]
    if len(names) == 0:
        raise ValueError("no cells selected")
    return names


def make_chunks(names, cells=None):
    """[[cell names]]: the full model alone, and each request's cells together, so one load serves many cells."""
    if cells is None:
        cells = CELLS
    chunks = []
    full = [name for name in names if cells[name]["request"] is None]
    if full:
        chunks.append(full)
    for request in REQUESTS:
        group = [name for name in names if cells[name]["request"] == request]
        if group:
            chunks.append(group)
    return chunks


def benchmarks_of(cell):
    """The benchmarks a cell scores: all of them for the full model, else its request's."""
    if cell["request"] is None:
        return list(ALL_BENCH)
    return list(REQUESTS[cell["request"]]["bench"])


# ---------------------------------------------------------------- data


def text_hash(texts):
    digest = hashlib.sha256()
    for text in texts:
        digest.update(text.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()[:12]


def question_of(prompt):
    """The question line(s) of a multiple-choice prompt, without the leading 'Question: '."""
    question = prompt.split("\nA. ")[0]
    if question.startswith("Question: "):
        question = question[len("Question: "):]
    return question


def overlap_keys(items):
    """Strings from every scoring item. A ranking or heal text that holds one is dropped."""
    keys = []
    for item in items.get("gsm8k", []):
        keys.append(item["question"])
    for item in items.get("math500", []):
        keys.append(item["problem"])
    for name, entries in items.items():
        if name in ("gsm8k", "math500"):
            continue
        for item in entries:
            keys.append(question_of(item["prompt"]))
    return keys


def eval_items(ev):
    """{benchmark: items}. The MMLU sets read the test split."""
    items = {"medmcqa": ev.medmcqa_items(), "gsm8k": ev.gsm8k_items(GSM8K_COUNT), "math500": ev.math500_items()}
    for group in ev.MMLU_GROUPS:
        items["mmlu_" + group] = ev.load_items("mmlu_" + group)
        items["mmlu_general_ex_" + group] = ev.load_items("mmlu_general_ex_" + group)
    for name in ALL_BENCH:
        if name not in items:
            raise ValueError(f"no items for {name}")
    return items


_PAIRS = {}


def cached_pairs(label, make):
    """A list of (user, assistant) pairs: in memory, else on the volume, else made and stored."""
    if label in _PAIRS:
        return _PAIRS[label]
    path = f"{PAIRS_DIR}/{V28}_{label}.json"
    safe_reload()
    try:
        if os.path.exists(path):
            with open(path) as f:
                _PAIRS[label] = [tuple(pair) for pair in json.load(f)]
            return _PAIRS[label]
    except Exception as error:
        print("pairs cache read failed:", repr(error)[:200])
    pairs = make()
    _PAIRS[label] = pairs
    try:
        os.makedirs(PAIRS_DIR, exist_ok=True)
        temp = path + ".tmp"
        with open(temp, "w") as f:
            json.dump(pairs, f)
        os.replace(temp, path)
        volume.commit()
    except Exception as error:
        print("pairs cache write failed:", repr(error)[:200])
    return pairs


def pool_pairs(skill):
    from wsp import healdata

    def make():
        ids = []
        for source in healdata.SOURCES[skill]:
            ids.append(source[0])
        healdata.check_sources(ids)
        return healdata.pairs(skill, POOL_PAIRS, seed=0, skip=0)

    return cached_pairs(f"pool_{skill}", make)


def bench_pairs(skill):
    from wsp import benchtrain

    return cached_pairs(f"train_{skill}", lambda: benchtrain.pairs(skill, BENCH_PAIRS, seed=0))


def request_texts(request, tokenizer, keys, rank_pool=RANK_POOL, rank_bench=RANK_BENCH, heal_pool=HEAL_POOL,
                  heal_bench=HEAL_BENCH):
    """(ranking texts, heal texts, info) of one request. Two sources: the healdata pool and the benchmark train
    splits. Each source drops texts that hold a scoring question, then gives its ranking slice and its heal slice."""
    from wsp import healdata, slim

    skill = REQUESTS[request]["skill"]
    rank = []
    heal = []
    info = {"skill": skill}
    for label, pairs, rank_n, heal_n in (("pool", pool_pairs(skill), rank_pool, heal_pool),
                                         ("train", bench_pairs(skill), rank_bench, heal_bench)):
        rendered = [healdata.render(tokenizer, user, assistant) for user, assistant in pairs]
        texts, dropped = slim.drop_overlap(rendered, keys)
        if len(texts) < rank_n + heal_n:
            raise ValueError(f"only {len(texts)} {label} texts for {request}, need {rank_n + heal_n}")
        rank += texts[:rank_n]
        heal += texts[rank_n:rank_n + heal_n]
        info[label] = {"pairs": len(pairs), "dropped_overlap": dropped, "rank": rank_n, "heal": heal_n}
    random.Random(0).shuffle(heal)
    info["rank_hash"] = text_hash(rank)
    return rank, heal, info


# ---------------------------------------------------------------- slimming


def zero_stats(model):
    import torch

    from wsp import slim

    layers = {}
    for layer, moe, router, top_k, count in slim.moe_layers_of(model):
        layers[layer] = {"count": torch.zeros(count, dtype=torch.float64),
                         "saliency": torch.zeros(count, dtype=torch.float64), "tokens": 0}
    return {"layers": layers, "top_k": 0, "num_experts": 0}


def cached_stats(slim, key, compute):
    path = f"{STATS_DIR}/{key}.pt"
    safe_reload()
    try:
        if os.path.exists(path):
            return slim.load_stats(path)
    except Exception as error:
        print("stats cache read failed:", repr(error)[:200])
    stats = compute()
    try:
        os.makedirs(STATS_DIR, exist_ok=True)
        temp = path + ".tmp"
        slim.save_stats(stats, temp)
        os.replace(temp, path)
        volume.commit()
    except Exception as error:
        print("stats cache write failed:", repr(error)[:200])
    return stats


def save_model(slim, model, tokenizer, name, request, keep, keep_model, smoke):
    """Save the slimmed model. Return (disk bytes, plain HF flag, path kept or None)."""
    if keep_model and not smoke:
        path = os.path.join(KEPT_ROOT, f"{request}_{keep}")
        os.makedirs(KEPT_ROOT, exist_ok=True)
        manifest = slim.save_slim(model, path, tokenizer)
        volume.commit()
        print(f"{name} kept the slimmed model at {path}", flush=True)
        return slim.directory_bytes(path), manifest["plain_hf"], path
    path = f"/tmp/round28_{name}"
    try:
        manifest = slim.save_slim(model, path, tokenizer)
        return slim.directory_bytes(path), manifest["plain_hf"], None
    finally:
        shutil.rmtree(path, ignore_errors=True)


# ---------------------------------------------------------------- measuring


def mean(values):
    return sum(values) / len(values)


def request_scores(scores):
    """{request: {domain, general}} for every request whose benchmarks are all in scores."""
    out = {}
    for request in REQUESTS:
        names = DOMAIN[request] + [GENERAL[request]]
        if all(name in scores for name in names):
            out[request] = {"domain": mean([scores[name] for name in DOMAIN[request]]),
                            "general": scores[GENERAL[request]]}
    return out


def sample_outputs(ev, model, tokenizer, requests):
    """Greedy answers to the fixed prompts of the given requests, for reading side by side."""
    prompts = []
    for request in requests:
        for prompt in REQUESTS[request]["prompts"]:
            prompts.append((request, prompt))
    outputs = ev.generate_texts(model, tokenizer, [ev.render_prompt(tokenizer, p) for _, p in prompts],
                                max_new_tokens=SAMPLE_TOKENS, batch_size=len(prompts))
    return [{"request": r, "prompt": p, "output": o} for (r, p), o in zip(prompts, outputs)]


def measure(name, model, tokenizer, ev, items, row, benches, limit=None):
    """Score the benchmarks, the decode speed, and the sample answers into row. Score lines start with the cell name."""
    result = ev.evaluate_suite(model, tokenizer, skills=benches, items=items, batch_size=32, limit=limit, log=False)
    row["scores"] = dict(result["scores"])
    row["n_items"] = {}
    row["correct"] = {}
    for bench in benches:
        found = result["skills"][bench]
        row["n_items"][bench] = found["n"]
        print(f"{name} {bench} {found['score']:.4f} ({found['correct']}/{found['n']}) in {found['seconds']} s",
              flush=True)
        row["correct"][bench] = [int(bool(record["correct"])) for record in result["skills"][bench]["records"]]
        for record in result["skills"][bench]["records"][:2]:
            print(f"{name} {bench} raw {json.dumps({k: record[k] for k in record if k in ('id', 'gold', 'pred', 'probs', 'output')}, default=str)[:400]}",
                  flush=True)
    row["request_scores"] = request_scores(row["scores"])
    if limit is None or limit > 0:
        row["decode_tok_s"] = round(ev.decode_speed(model, tokenizer), 1)
        print(f"{name} decode {row['decode_tok_s']} tok/s", flush=True)
    return result


def save_cell(name, row, result):
    os.makedirs(CELL_DIR, exist_ok=True)
    full = dict(row)
    if result is not None:
        full["skills_detail"] = result["skills"]
    temp = f"{CELL_DIR}/{name}.json.tmp"
    with open(temp, "w") as f:
        json.dump(full, f, default=str)
    os.replace(temp, f"{CELL_DIR}/{name}.json")
    volume.commit()


def heal_row(report):
    keep = ["steps", "trainable_params", "router_params", "kd_first", "kd_last", "lm_first", "lm_last", "seconds"]
    return {key: report[key] for key in keep}


# ---------------------------------------------------------------- one cell


class Context:
    """What one container holds across its cells: the model, tokenizer, items, and per-request texts."""

    def __init__(self, smoke=False):
        import torch
        import transformers

        from wsp import skills_eval as ev

        self.smoke = smoke
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_ID)
        start = time.time()
        self.model = transformers.AutoModelForCausalLM.from_pretrained(
            MODEL_ID, dtype=torch.bfloat16, device_map="cuda").eval()
        print(f"model loaded in {time.time() - start:.0f} s", flush=True)
        self.ev = ev
        self.items = eval_items(ev)
        self.keys = overlap_keys(self.items)
        self.texts = {}

    def texts_of(self, request):
        if request not in self.texts:
            if self.smoke:
                self.texts[request] = request_texts(request, self.tokenizer, self.keys, SMOKE_RANK, SMOKE_RANK,
                                                    SMOKE_HEAL_STEPS * HEAL["batch_size"] // 2,
                                                    SMOKE_HEAL_STEPS * HEAL["batch_size"] // 2)
            else:
                self.texts[request] = request_texts(request, self.tokenizer, self.keys)
        return self.texts[request]


def run_cell(ctx, name):
    import torch

    from wsp import slim

    cell = CELLS[name]
    smoke = ctx.smoke
    start = time.time()
    row = {"cell": name, "model": MODEL_ID, "request": cell["request"], "rule": cell["rule"], "keep": cell["keep"],
           "seed": cell["seed"], "heal_steps": cell["heal"], "smoke": smoke, "code_version": V28}
    if cell["request"] is not None:
        row["request_text"] = REQUESTS[cell["request"]]["text"]
    total, run = slim.count_params(ctx.model)
    row["params_total_before"] = total
    row["params_run_before"] = run
    student = ctx.model
    benches = benchmarks_of(cell)
    if cell["rule"] is None:
        row["params_total_after"] = total
        row["params_run_after"] = run
        row["disk_bytes"] = 2 * total
        row["disk_measured"] = False
    else:
        request = cell["request"]
        rank, heal_texts, info = ctx.texts_of(request)
        row["data"] = info
        if cell["rule"] == "random":
            stats = zero_stats(ctx.model)
        else:
            key = f"{V28}_{request}_{len(rank)}_{RANK_LENGTH}_{info['rank_hash']}"
            stats = cached_stats(slim, key, lambda: slim.collect_moe_stats(ctx.model, ctx.tokenizer, rank,
                                                                           RANK_LENGTH))
        student, report = slim.slim_moe(ctx.model, None, None, None, rule=cell["rule"], keep=cell["keep"],
                                        mass=None, seed=cell["seed"], skill_stats=stats, share_experts=True)
        counts = [report["counts"][layer] for layer in sorted(report["counts"])]
        row.update({"kept": report["kept"], "kept_counts": counts, "uniform": report["uniform"],
                    "params_total_after": report["params_total_after"],
                    "params_run_after": report["params_run_after"]})
        if cell["rule"] != "random":
            row["kept_share"] = slim.kept_share(stats, report["kept"])
        steps = cell["heal"]
        if steps > 0:
            if smoke:
                steps = SMOKE_HEAL_STEPS
            need = steps * HEAL["batch_size"]
            healed = slim.heal(student, ctx.model, ctx.tokenizer, heal_texts[:need], steps,
                               batch_size=HEAL["batch_size"], seq_len=HEAL["seq_len"], lora_rank=HEAL["lora_rank"],
                               lora_lr=HEAL["lora_lr"], router_lr=HEAL["router_lr"], warmup=HEAL["warmup"],
                               lm_weight=HEAL["lm_weight"], seed=cell["seed"])
            row["heal"] = heal_row(healed)
            row["heal_seconds"] = healed["seconds"]
            row["heal_steps_run"] = steps
            print(f"{name} heal kd {healed['kd_first']:.4f} -> {healed['kd_last']:.4f} in {healed['seconds']:.0f} s",
                  flush=True)
        row["disk_bytes"], row["plain_hf"], row["kept_path"] = save_model(
            slim, student, ctx.tokenizer, name, cell["request"], cell["keep"], cell["keep_model"], smoke)
        row["disk_measured"] = True
    limit = SMOKE_LIMIT if smoke else None
    result = measure(name, student, ctx.tokenizer, ctx.ev, ctx.items, row, benches, limit=limit)
    requests = list(REQUESTS) if cell["request"] is None else [cell["request"]]
    row["samples"] = sample_outputs(ctx.ev, student, ctx.tokenizer, requests)
    for sample in row["samples"][:2]:
        print(f"{name} sample {json.dumps(sample)[:500]}", flush=True)
    row["seconds"] = round(time.time() - start, 1)
    save_cell(name if not smoke else "smoke_" + name, row, result)
    if student is not ctx.model:
        del student
    torch.cuda.empty_cache()
    return row


def run_chunk_rows(names, smoke=False):
    """Yield one row per cell. A failed cell is a row with an error."""
    import gc

    import torch

    try:
        ctx = Context(smoke=smoke)
    except Exception as error:
        for name in names:
            yield {"cell": name, "error": "load failed: " + repr(error)[:400], "scores": None}
        return
    for name in names:
        try:
            yield run_cell(ctx, name)
        except Exception as error:
            import traceback

            traceback.print_exc()
            print(f"{name} FAILED {repr(error)[:300]}", flush=True)
            yield {"cell": name, "error": repr(error)[:500], "scores": None}
        gc.collect()
        torch.cuda.empty_cache()


@app.function(image=image, volumes={"/vol": volume}, gpu=GPU, memory=262144, timeout=20 * 3600)
def run_chunk(names, smoke=False):
    for row in run_chunk_rows(names, smoke=smoke):
        yield row


@app.function(image=image, volumes={"/vol": volume}, memory=32768, timeout=3 * 3600)
def check_data():
    """CPU only: read every pool and train split at full size, print counts and samples, and cache the pairs."""
    import transformers

    from wsp import skills_eval as ev

    tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_ID)
    items = eval_items(ev)
    for name in ALL_BENCH:
        print(f"items {name} {len(items[name])}", flush=True)
    keys = overlap_keys(items)
    print(f"overlap keys {len(keys)}", flush=True)
    for request in REQUESTS:
        rank, heal, info = request_texts(request, tokenizer, keys)
        print(f"data {request} rank {len(rank)} heal {len(heal)} {json.dumps(info)}", flush=True)
        for label, pairs in (("pool", pool_pairs(REQUESTS[request]["skill"])),
                             ("train", bench_pairs(REQUESTS[request]["skill"]))):
            for user, assistant in pairs[:1] + pairs[1000:1001]:
                print(f"data {request} {label} user {user[:300]!r} assistant {assistant[:300]!r}", flush=True)
    volume.commit()
    return "ok"


RUNNER = run_chunk


# ---------------------------------------------------------------- entrypoint


def write_rows(path, rows):
    os.makedirs("results", exist_ok=True)
    temp = path + ".tmp"
    with open(temp, "w") as f:
        json.dump(rows, f, indent=2, default=str)
    os.replace(temp, path)


def short_row(row):
    return {key: row[key] for key in row if key not in ("correct", "kept", "samples")}


@app.local_entrypoint()
def main(only: str = "", requests: str = "", smoke: bool = False, estimate_only: bool = False,
         check: bool = False, tag: str = "main"):
    if check:
        print(check_data.remote(), flush=True)
        return
    names = select_cells(only=only, requests=requests, smoke=smoke)
    chunks = make_chunks(names)
    print(f"{len(names)} cells in {len(chunks)} containers: {names}", flush=True)
    if estimate_only:
        for name in names:
            cell = CELLS[name]
            print(f"  {name}: {cell['rule']} keep {cell['keep']} heal {cell['heal']} scores {benchmarks_of(cell)}")
        return
    path = "results/round28_smoke.json" if smoke else f"results/round28_{tag}.json"
    rows = []
    lock = threading.Lock()

    def work(chunk):
        seen = set()
        try:
            for row in RUNNER.remote_gen(chunk, smoke):
                seen.add(row["cell"])
                with lock:
                    rows.append(row)
                    write_rows(path, rows)
                    if row.get("error") is not None:
                        print(f"{row['cell']} FAILED {row['error']}", flush=True)
                    else:
                        print(row["cell"] + " row " + json.dumps(short_row(row), default=str)[:1500], flush=True)
        except Exception as error:
            print(f"chunk {chunk} failed: {repr(error)[:400]}", flush=True)
        for name in chunk:
            if name not in seen:
                with lock:
                    rows.append({"cell": name, "error": "chunk ended before this cell", "scores": None})
                    write_rows(path, rows)

    # The smoke runs both of its cells in one container so one load serves both.
    threads = []
    for chunk in chunks:
        thread = threading.Thread(target=work, args=(chunk,))
        thread.start()
        threads.append(thread)
    for thread in threads:
        thread.join()
    write_rows(path, rows)
    print(f"wrote {len(rows)} rows to {path}", flush=True)
