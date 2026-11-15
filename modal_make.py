"""Make a model for a request on Modal: plan locally, compile and score each candidate in its own container.

    MODAL_PROFILE=trisanth modal run modal_make.py --request requests/med_mcq_1b.json
    MODAL_PROFILE=trisanth modal run modal_make.py --requests-dir requests

The local entrypoint reads the request, loads profiles.json, and asks
wsp.planner for candidates (through wsp.make). Each candidate runs in one
container (an H200, or an H100 when the request asks for one):

1. Build the Program from the candidate's dict. A "bake" key in the dict
   is the bake marker; it is not part of the language and is set aside.
2. Compile it with wsp.compiler. Taps train through the program's
   tap_train; a LoRA trains through its heal. A LoRA without taps heals
   on the texts of its pairs set (prompt plus completion), since
   wsp.data knows only gsm8k_train. A program with nothing to train (a
   source model alone) is not compiled.
3. Bake it when the marker is set: wsp.bake distills the tapped host into
   a plain host LoRA, so no donor runs at inference. The marker can be
   true or a dict of bake settings (rank, steps, lr, batch_size, seq_len,
   mode, data), on the candidate or in its program dict.
4. Load it with wsp.load and score it on the request's benchmark:
   multiple choice with wsp.mcq.score_items on held-out items, GSM8K by
   greedy generation and the final number. A model that trained on
   gsm8k_train pairs is prompted in that raw format; any other model is
   prompted with its chat template.
5. Measure decode tokens per second with batch 1 and greedy decoding.

Held-out items (seeded, never the dev items the planner's numbers came from):
  mmlu      test, seed 0, positions 1000 on, 2000 items (round 7 held-out)
  medmcqa   validation, seed 7, positions 1000 on, 1500 items (round 12 held-out)
  medqa     test, all 1273 items
  mmlu_pro  test, seed 0, positions 1000 on, 2000 items
  gsm8k     test, seed 0, 500 items
Subject filters from the request apply after the skip. --eval-count
changes the item count.

Outputs:
  results/<name>_shard<shard>.json   every measured result of this shard
  results/make/<request>.plan.json   the plan
  results/make/<request>.json        the report card (one shard only)
Compiled models go to the wsp-models volume at /vol/make/<request>/.

By default each planned host also runs alone (no training), so the
report card compares against measured host numbers; --no-baselines
turns that off.

Spread one request over several workspaces with scripts/modal_fanout.py.
The fan-out name must match --name (default make_<request>, or make_all
for --requests-dir):

    python scripts/modal_fanout.py modal_make.py make_med_mcq_1b --profiles a,b,c \\
        --extra "--request requests/med_mcq_1b.json"
    python -m wsp.make requests/med_mcq_1b.json --results results/make_med_mcq_1b_rows.json

Each shard plans on its own, so the LLM planner (which can give a new
plan each time) needs --plan-file with more than one shard. Write one
with python -m wsp.make <request> --dry-run --save-plan <path>.
"""

import glob
import hashlib
import json
import os
import random
import re
import time

import modal

app = modal.App("wsp-make")
volume = modal.Volume.from_name("wsp-models", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch==2.7.1", "transformers==4.57.1", "accelerate", "huggingface_hub", "requests",
        "safetensors", "zstandard", "numpy", "datasets", "sentencepiece", "protobuf",
    )
    .env({"HF_HOME": "/vol/hf"})
    .add_local_python_source("wsp")
    .add_local_file("catalog.jsonl", "/root/catalog.jsonl")
)

# benchmark -> (split, seed, skip, count)
HELDOUT = {
    "mmlu": ("test", 0, 1000, 2000),
    "medmcqa": ("validation", 7, 1000, 1500),
    "medqa": ("test", 0, 0, None),
    "mmlu_pro": ("test", 0, 1000, 2000),
}
GSM8K_COUNT = 500
GSM8K_MAX_NEW_TOKENS = 320
BAKE_DEFAULTS = {"rank": 16, "steps": 500, "lr": 1e-4, "batch_size": 8, "seq_len": 256}


# Helpers that run anywhere (the tests import them on CPU)


def option_count(benchmark):
    if benchmark == "mmlu_pro":
        return 10
    return 4


def eval_spec(request, eval_count=0):
    """Return (split, seed, skip, count) of the held-out items for a multiple-choice request."""
    benchmark = request["task"]["benchmark"]
    if benchmark in HELDOUT:
        split, seed, skip, count = HELDOUT[benchmark]
    else:
        split, seed, skip, count = ("test", 0, 1000, 2000)
    if eval_count and eval_count > 0:
        count = eval_count
    return split, seed, skip, count


def trains_on(program, data_name):
    """True when the program trains anything on the pairs set data_name."""
    tap_train = program.get("tap_train")
    if len(program.get("taps") or []) > 0 and tap_train is not None and tap_train.get("data") == data_name:
        return True
    heal = program.get("heal")
    if heal is None or heal.get("data") != data_name:
        return False
    return program.get("lora") is not None or len(program.get("grafts") or []) > 0


def gsm8k_style(program):
    """"raw" for a model trained on gsm8k_train pairs (it learned that format), else "chat"."""
    if trains_on(program, "gsm8k_train"):
        return "raw"
    return "chat"


NUMBER = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def clean_number(text):
    text = text.replace(",", "").rstrip(".")
    try:
        value = float(text)
    except ValueError:
        return None
    return value


def gsm8k_gold(answer):
    """The number after #### in a GSM8K answer."""
    return clean_number(answer.split("####")[-1].strip())


def final_number(text):
    """The final answer in a generated solution: after "The answer is", after ####, or the last number."""
    # A model trained on raw pairs may go on to write a new question.
    text = text.split("\nQuestion:")[0]
    lowered = text.lower()
    for marker in ("the answer is", "####"):
        position = lowered.rfind(marker)
        if position != -1:
            found = NUMBER.findall(text[position + len(marker):])
            if len(found) > 0:
                return clean_number(found[0])
    found = NUMBER.findall(text)
    if len(found) == 0:
        return None
    return clean_number(found[-1])


def same_number(a, b):
    if a is None or b is None:
        return False
    return abs(a - b) < 1e-6


def cell_list(plans, shard, nshards):
    """Spread (request, candidate, index) cells over shards by their position across all plans."""
    cells = []
    position = 0
    for request, candidates in plans:
        for index, candidate in enumerate(candidates):
            if position % nshards == shard:
                cells.append((request, candidate, index))
            position += 1
    return cells


def model_folder(request, candidate, index):
    from wsp.make import program_key

    digest = hashlib.sha1(program_key(candidate["program"]).encode()).hexdigest()[:10]
    return f"/vol/make/{request['name']}/{index:02d}_{digest}"


def empty_result(request, candidate, index):
    return {"request": request["name"], "index": index, "candidate": candidate, "accuracy": None,
            "decode_tok_s": None, "train_seconds": None, "model_dir": None, "error": None}


# Training pairs that wsp.pairs may not have yet


def medmcqa_pairs(tokenizer, seed, count, max_prompt_tokens):
    """MedMCQA train pairs in the round 12 format (seeded order, 4 options, completion " X")."""
    from datasets import load_dataset

    from wsp.mcq import mcq_prompt
    from wsp.pairs import LETTERS, prompt_fits

    train = load_dataset("openlifescienceai/medmcqa", split="train")
    order = list(range(len(train)))
    random.Random(seed + 3).shuffle(order)
    pairs = []
    for index in order:
        row = train[index]
        if row["cop"] is None or int(row["cop"]) not in (0, 1, 2, 3):
            continue
        prompt = mcq_prompt(row["question"], [row["opa"], row["opb"], row["opc"], row["opd"]])
        if not prompt_fits(tokenizer, prompt, max_prompt_tokens):
            continue
        pairs.append((prompt, " " + LETTERS[int(row["cop"])]))
        if count is not None and len(pairs) >= count:
            break
    return pairs


def medqa_pairs(tokenizer, seed, count, max_prompt_tokens):
    """MedQA-USMLE 4-option train pairs."""
    from datasets import load_dataset

    from wsp.mcq import LETTERS, mcq_prompt, read_medqa
    from wsp.pairs import prompt_fits

    train = load_dataset("GBaker/MedQA-USMLE-4-options", split="train")
    order = list(range(len(train)))
    random.Random(seed).shuffle(order)
    pairs = []
    for index in order:
        question, choices, gold, subject = read_medqa(train[index])
        prompt = mcq_prompt(question, choices)
        if not prompt_fits(tokenizer, prompt, max_prompt_tokens):
            continue
        pairs.append((prompt, " " + LETTERS[gold]))
        if count is not None and len(pairs) >= count:
            break
    return pairs


# name -> (function, count, max_prompt_tokens)
EXTRA_PAIRS = {
    "medmcqa_train": (medmcqa_pairs, 24000, 240),
    "medqa_train": (medqa_pairs, 10000, 400),
}


def register_extra_pairs():
    """Add the pairs sets above to wsp.pairs, unless wsp.pairs already has them."""
    from wsp.pairs import REGISTRY, register_pairs

    for name in EXTRA_PAIRS:
        if name in REGISTRY:
            continue
        function, count, max_prompt_tokens = EXTRA_PAIRS[name]
        register_pairs(name, function, count=count, max_prompt_tokens=max_prompt_tokens)


def heal_texts_for(name, tokenizer):
    """Heal texts for a LoRA or graft program without taps."""
    if name == "gsm8k_train":
        from wsp.data import task_texts

        return task_texts(name, seed=0)
    from wsp.pairs import pairs_for

    texts = []
    for prompt, completion in pairs_for(name, tokenizer):
        texts.append(prompt + completion)
    return texts


# Container steps


def compile_candidate(program, out_dir):
    """Compile the program into out_dir and write program.wsp next to it."""
    import transformers

    from wsp.catalog import Catalog
    from wsp.compiler import compile_program
    from wsp.lang import format_program

    catalog = Catalog([])
    if os.path.exists("/root/catalog.jsonl"):
        catalog = Catalog.load("/root/catalog.jsonl")
    heal_texts = None
    align_texts = None
    if program.heal_spec is not None and len(program.taps) == 0:
        tokenizer = transformers.AutoTokenizer.from_pretrained(program.host)
        heal_texts = heal_texts_for(program.heal_spec.data, tokenizer)
        if len(program.grafts) > 0:
            align_texts = heal_texts[:200]
    compile_program(program, catalog, out_dir, align_texts=align_texts, heal_texts=heal_texts, device="cuda")
    try:
        text = format_program(program)
    except ValueError:
        text = None
    if text is not None:
        with open(os.path.join(out_dir, "program.wsp"), "w") as f:
            f.write(text)
    return out_dir


def bake_candidate(request, program, out_dir, spec):
    """Distill the compiled tapped model in out_dir into a plain host. Return the baked folder."""
    import torch

    from wsp.bake import bake, save_baked
    from wsp.load import load
    from wsp.mcq import letter_ids
    from wsp.pairs import pairs_for

    settings = dict(BAKE_DEFAULTS)
    for key in spec:
        settings[key] = spec[key]
    mode = settings.get("mode")
    if mode is None:
        mode = "answer"
        if request["task"]["kind"] == "generate":
            mode = "sequence"
    teacher, tokenizer = load(out_dir, device="cuda")
    donors = teacher.wsp_tap_donors
    data_name = settings.get("data")
    if data_name is None and program.tap_train_spec is not None:
        data_name = program.tap_train_spec.data
    if data_name is None:
        data_name = "mmlu_aux"
    pairs = pairs_for(data_name, tokenizer)
    option_ids = None
    if mode == "answer":
        option_ids = letter_ids(tokenizer, option_count(request["task"]["benchmark"]))
    result = bake(teacher, tokenizer, donors, pairs, rank=settings["rank"], steps=settings["steps"],
                  lr=settings["lr"], batch_size=settings["batch_size"], seq_len=settings["seq_len"], mode=mode,
                  option_ids=option_ids, seed=program.seed, merge=True)
    baked_dir = os.path.join(out_dir, "baked")
    save_baked(result["student"], tokenizer, baked_dir, info=result["info"])
    del result
    del teacher
    del donors
    torch.cuda.empty_cache()
    return baked_dir


def score_mcq(model, tokenizer, request, eval_count):
    from wsp.mcq import accuracy, mcq_items, score_items

    split, seed, skip, count = eval_spec(request, eval_count)
    items = mcq_items(request["task"]["benchmark"], split, seed=seed, count=count,
                      subjects=request["task"].get("subjects"), skip=skip)
    # A loaded tap model runs its donors by itself (wsp.load turns on auto-prepare).
    records = score_items(model, tokenizer, items, batch_size=16)
    return accuracy(records), records


def gsm8k_prompt(tokenizer, question, style):
    if style == "chat" and getattr(tokenizer, "chat_template", None):
        messages = [{"role": "user", "content": question.strip() + "\nSolve it step by step. "
                     "End with: The answer is <number>."}]
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return "Question: " + question + "\nAnswer:"


def score_gsm8k(model, tokenizer, program, eval_count, batch_size=16):
    import torch
    from datasets import load_dataset

    count = GSM8K_COUNT
    if eval_count and eval_count > 0:
        count = eval_count
    test = load_dataset("openai/gsm8k", "main", split="test")
    order = list(range(len(test)))
    random.Random(0).shuffle(order)
    style = gsm8k_style(program)
    if style == "chat" and not getattr(tokenizer, "chat_template", None):
        style = "raw"
    old_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    records = []
    chosen = order[:count]
    try:
        for start in range(0, len(chosen), batch_size):
            rows = []
            prompts = []
            for index in chosen[start:start + batch_size]:
                rows.append((index, test[index]))
                prompts.append(gsm8k_prompt(tokenizer, test[index]["question"], style))
            # A chat template already holds the special tokens.
            encoded = tokenizer(prompts, return_tensors="pt", padding=True,
                                add_special_tokens=(style == "raw")).to(model.device)
            with torch.no_grad():
                output = model.generate(input_ids=encoded["input_ids"], attention_mask=encoded["attention_mask"],
                                        max_new_tokens=GSM8K_MAX_NEW_TOKENS, do_sample=False,
                                        pad_token_id=tokenizer.pad_token_id)
            width = encoded["input_ids"].shape[1]
            for row, (index, item) in enumerate(rows):
                text = tokenizer.decode(output[row, width:], skip_special_tokens=True)
                pred = final_number(text)
                gold = gsm8k_gold(item["answer"])
                records.append({"id": f"gsm8k/test/{index}", "gold": gold, "pred": pred,
                                "correct": same_number(pred, gold), "output": text[-400:]})
    finally:
        tokenizer.padding_side = old_side
    right = 0
    for record in records:
        if record["correct"]:
            right += 1
    return right / max(1, len(records)), records, style


def decode_speed(model, tokenizer, new_tokens=64):
    """Decode tokens per second at batch 1: (time for 1 + new_tokens) minus (time for 1), so prefill drops out."""
    import torch

    encoded = tokenizer(["Question: Name three primary colors and say why each is primary.\nAnswer:"],
                        return_tensors="pt").to(model.device)
    pad = tokenizer.pad_token_id
    if pad is None:
        pad = tokenizer.eos_token_id

    def timed(count):
        torch.cuda.synchronize()
        start = time.time()
        with torch.no_grad():
            model.generate(input_ids=encoded["input_ids"], attention_mask=encoded["attention_mask"],
                           max_new_tokens=count, min_new_tokens=count, do_sample=False, pad_token_id=pad)
        torch.cuda.synchronize()
        return time.time() - start

    timed(4)
    one = timed(1)
    many = timed(new_tokens + 1)
    return new_tokens / max(many - one, 1e-6)


def decode_params_b(model):
    """Parameters that run at decode, in billions: the host and its tap branches, not the donors."""
    total = 0
    for param in model.parameters():
        total += param.numel()
    return total / 1e9


def measure(request, candidate, index, eval_count, result):
    """Compile, bake, load, score, and time one candidate. Fill in result."""
    import torch

    from wsp.load import load
    from wsp.make import bake_spec, is_bare, normalize_candidate, program_core
    from wsp.program import Program

    register_extra_pairs()
    candidate = normalize_candidate(candidate)
    core = program_core(candidate["program"])
    spec = bake_spec(candidate["program"])
    program = Program.from_dict(core)
    if spec is not None and len(program.taps) == 0:
        raise ValueError("a bake marker needs a program with taps")
    torch.manual_seed(program.seed)

    start = time.time()
    if is_bare(core):
        model_dir = program.host
    else:
        model_dir = compile_candidate(program, model_folder(request, candidate, index))
        if spec is not None:
            model_dir = bake_candidate(request, program, model_dir, spec)
        volume.commit()
    result["train_seconds"] = round(time.time() - start, 1)
    result["model_dir"] = model_dir

    model, tokenizer = load(model_dir, device="cuda")
    result["decode_params_b"] = round(decode_params_b(model), 4)
    if request["task"]["kind"] == "mcq":
        accuracy, records = score_mcq(model, tokenizer, request, eval_count)
    else:
        accuracy, records, style = score_gsm8k(model, tokenizer, core, eval_count)
        result["prompt_style"] = style
    result["accuracy"] = accuracy
    result["items"] = len(records)
    result["records"] = records
    result["decode_tok_s"] = round(decode_speed(model, tokenizer), 1)


@app.function(image=image, volumes={"/vol": volume}, gpu="H200", memory=262144, timeout=6 * 3600)
def run_candidate(request, candidate, index, eval_count=0):
    """Run one candidate. Errors come back in the result, so one bad candidate does not stop the rest."""
    import traceback

    start = time.time()
    result = empty_result(request, candidate, index)
    try:
        measure(request, candidate, index, eval_count, result)
    except Exception as error:
        result["error"] = repr(error)[:500]
        result["traceback"] = traceback.format_exc()[-4000:]
    result["seconds"] = round(time.time() - start, 1)
    print(json.dumps({"request": request["name"], "index": index, "accuracy": result["accuracy"],
                      "decode_tok_s": result["decode_tok_s"], "train_seconds": result["train_seconds"],
                      "error": result["error"]}))
    return result


# Local side


def run_cells(cells, eval_count):
    """Run cells with starmap, one container each. H100 requests go to H100 containers."""
    by_gpu = {}
    for position, cell in enumerate(cells):
        gpu = cell[0]["constraints"].get("gpu") or "H200"
        if gpu not in by_gpu:
            by_gpu[gpu] = []
        by_gpu[gpu].append((position, cell))
    results = [None] * len(cells)
    for gpu in by_gpu:
        function = run_candidate
        if gpu != "H200":
            function = run_candidate.with_options(gpu=gpu)
        args = []
        for position, cell in by_gpu[gpu]:
            request, candidate, index = cell
            args.append((request, candidate, index, eval_count))
        outputs = function.starmap(args, return_exceptions=True)
        for (position, cell), output in zip(by_gpu[gpu], outputs):
            if isinstance(output, Exception):
                request, candidate, index = cell
                failed = empty_result(request, candidate, index)
                failed["error"] = repr(output)[:500]
                output = failed
            results[position] = output
    return results


def request_paths(request, requests_dir):
    if request:
        return [request]
    if requests_dir:
        return sorted(glob.glob(os.path.join(requests_dir, "*.json")))
    raise SystemExit("give --request <file> or --requests-dir <folder>")


@app.local_entrypoint()
def main(request: str = "", requests_dir: str = "", profiles: str = "profiles.json", k: int = 6,
         use_llm: bool = False, plan_file: str = "", shard: int = 0, nshards: int = 1, eval_count: int = 0,
         name: str = "", baselines: bool = True):
    from wsp import make

    paths = request_paths(request, requests_dir)
    if plan_file and len(paths) != 1:
        raise SystemExit("--plan-file works with one --request")
    if use_llm and nshards > 1 and not plan_file:
        raise SystemExit("the LLM planner can give each shard a different plan; pass --plan-file")
    profile_rows = make.load_profiles(profiles)

    plans = []
    os.makedirs(os.path.join("results", "make"), exist_ok=True)
    for path in paths:
        one = make.load_request(path)
        if plan_file:
            candidates = make.read_plan(plan_file)
        else:
            candidates = make.plan_candidates(one, profile_rows, k=k, use_llm=use_llm)
            if baselines:
                # Each planned host alone, so the card has measured host numbers.
                candidates = candidates + make.baseline_candidates(one, candidates, profile_rows)
        plans.append((one, candidates))
        make.write_json(os.path.join("results", "make", one["name"] + ".plan.json"),
                        {"request": one, "candidates": candidates})
        print(make.format_plan(one, profile_rows, candidates))

    cells = cell_list(plans, shard, nshards)
    print(f"shard {shard}/{nshards}: {len(cells)} candidate(s)")
    results = run_cells(cells, eval_count)

    if not name:
        name = "make_all"
        if len(paths) == 1:
            name = "make_" + plans[0][0]["name"]
    make.write_json(os.path.join("results", f"{name}_shard{shard}.json"), results)

    if nshards > 1:
        print(f"wrote results/{name}_shard{shard}.json. After the fan-out merges the shards, run "
              f"python -m wsp.make <request> --results results/{name}_rows.json")
        return
    for one, candidates in plans:
        mine = []
        for result in results:
            if result["request"] == one["name"]:
                mine.append(result)

        def already_run(_candidates, rows=mine):
            return rows

        card = make.make(one, profile_rows, already_run, candidates=candidates)
        out = os.path.join("results", "make", one["name"] + ".json")
        make.write_json(out, card)
        print(make.format_report(card))
        print(f"wrote {out}")
