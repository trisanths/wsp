"""The wsp studio on Modal: real GPU compile and eval for one program per call.

Deploy once (the studio server looks the functions up by name, so it does
not run inside Modal):

    MODAL_PROFILE=trisanth modal deploy studio/modal_studio.py

Each function runs one whole job in one container. It compiles the program,
trains what the program asks for (taps, then LoRA or glue heal), and scores
the result with the eval harness (wsp_eval.harness, hf backend). The host
and every candidate go through the same harness path, so their numbers
compare. Weights are written to a scratch folder and deleted after the call.
Only the Hugging Face cache lives on the wsp-models volume.

Functions:
    probe_remote(program, options)            compile with heal and tap training off
    score_remote(program, split, tasks, token, fidelity, options)
                                              compile with training, then evaluate
    base_remote(host, split, tasks)           the host's scores
Each has an H100 twin (name ending in _h100) for programs with a donor above
about 10B parameters. The others run on an L40S.

MMLU scoring: the task "mmlu" is scored here, not by the harness, the same
way as rounds 6 and 7 (results/README.md): the argmax over the log-probs of
" A" to " D" after "Question: ... Answer:". Items come from cais/mmlu test
shuffled with seed 0. The search split is the first 1000 (the round 6 dev
items) and the held-out split is the next 2000 (the round 7 held-out items),
so studio MMLU numbers compare with the results log. The harness path, which
generates a letter from a chat prompt, puts the 1B host at chance (it answers
"A" almost every time), so it is not used for MMLU. Other tasks use the harness.
"""

import json
import math
import os
import shutil
import tempfile
import time
import traceback

import modal

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

app = modal.App("wsp-studio")
volume = modal.Volume.from_name("wsp-models", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch==2.7.1", "transformers==4.57.1", "accelerate", "huggingface_hub", "requests",
        "safetensors", "zstandard", "numpy", "datasets", "sentencepiece", "protobuf",
    )
    .env({"HF_HOME": "/vol/hf", "WSP_BACKEND": "hf", "WSP_HF_BATCH_SIZE": "32"})
    .add_local_dir(os.path.join(REPO_ROOT, "wsp"), "/root/wsp", ignore=["**/__pycache__", "**/*.pyc"])
    .add_local_dir(os.path.join(REPO_ROOT, "wsp_eval"), "/root/wsp_eval", ignore=["**/__pycache__", "**/*.pyc"])
    .add_local_dir(os.path.join(REPO_ROOT, "parts_extracted"), "/root/parts_extracted")
)

FIDELITY_FRACTIONS = {"check": 0.16, "low": 0.34, "full": 1.0}
SCRATCH_ROOT = "/tmp"


def gpu_name():
    import torch

    if torch.cuda.is_available():
        return torch.cuda.get_device_name(0)
    return "cpu"


def make_catalog(options):
    """A catalog of just the records this program uses, rooted at /root so units paths resolve."""
    from wsp.catalog import Catalog

    records = []
    for record in options.get("catalog_records") or []:
        records.append(record)
    return Catalog(records, root="/root")


def align_texts_for(program, options):
    """Texts for the ridge fit when the program has grafts, else None."""
    if len(program.grafts) == 0:
        return None
    from wsp.data import task_texts

    texts = task_texts(options["align_data"], seed=0)
    return texts[:options["align_samples"]]


def tap_param_count(entry):
    """Parameters of one tap branch from its saved sizes."""
    d_host = entry["d_host"]
    d_donor = entry["d_donor"]
    attn = entry["attn_dim"]
    total = d_host + d_donor
    total += attn * d_host
    total += 2 * attn * d_donor
    total += attn * d_host
    return total + 1


def report_entries(report):
    """The per-graft and per-tap entries of a compile report, in the shape the backend interface wants."""
    entries = []
    for graft in report.get("grafts") or []:
        entries.append(graft)
    taps = report.get("taps")
    if taps is not None:
        gates = taps.get("gates") or []
        index = 0
        for tap in taps.get("taps") or []:
            gate = None
            if index < len(gates):
                gate = gates[index]
            entries.append({"id": "tap:" + str(tap["donor_id"]), "host_layer": tap["host_layer"],
                            "fit": "skipped", "fit_r2": None, "fit_pairs": 0,
                            "part_params": tap_param_count(tap), "glue_params": 0,
                            "donor": tap["donor_id"], "donor_layer": tap["donor_layer"],
                            "control": None, "gate": gate})
            index += 1
    return entries


def mean_tail(values, count):
    if len(values) == 0:
        return None
    tail = values[-count:]
    return sum(tail) / len(tail)


def build_model(program_dict, options, folder, training):
    """Compile a program into folder. Returns (compile_report, seconds).

    training False is a probe: heal is dropped and the taps train for 0 steps.
    """
    from wsp.program import Program

    data = json.loads(json.dumps(program_dict))
    if not training:
        data["heal"] = None
        if data.get("tap_train") is not None:
            data["tap_train"]["steps"] = 0
    program = Program.from_dict(data)
    catalog = make_catalog(options)
    texts = align_texts_for(program, options)
    extra = {}
    if not training:
        extra["tap_pairs"] = []
        extra["heal_pairs"] = []
    start = time.time()
    program.compile(folder, catalog, align_texts=texts, **extra)
    seconds = time.time() - start
    with open(os.path.join(folder, "compile_report.json"), encoding="utf-8") as handle:
        report = json.load(handle)
    return report, seconds


def phase_seconds(report, total):
    """Split the build time: heal covers tap training and the heal pass, compile is the rest."""
    heal_s = 0.0
    taps = report.get("taps")
    if taps is not None:
        heal_s += float(taps.get("seconds") or 0.0)
    heal = report.get("heal")
    if heal is not None:
        heal_s += float(heal.get("seconds") or 0.0)
    if heal_s > total:
        heal_s = total
    return {"compile": round(total - heal_s, 1), "heal": round(heal_s, 1)}


def heal_losses(report):
    """The heal pass losses, or the tap training losses when no heal ran."""
    heal = report.get("heal")
    if heal is not None and len(heal.get("losses") or []) > 0:
        return list(heal["losses"])
    taps = report.get("taps")
    if taps is not None:
        return list(taps.get("losses") or [])
    return []


def fresh_root(prefix):
    path = tempfile.mkdtemp(prefix=prefix, dir=SCRATCH_ROOT)
    os.environ["WSP_ROOT"] = path
    return path


def fraction_of(fidelity):
    """A fidelity is a name (check, low, full) or a number from 0 to 1."""
    if isinstance(fidelity, str):
        return FIDELITY_FRACTIONS[fidelity]
    return float(fidelity)


class LimitedItems:
    """Make every split a prefix of its items, so a lower fidelity is a nested question set."""

    def __init__(self, fraction):
        self.fraction = fraction
        self.original = None

    def __enter__(self):
        from wsp_eval import splits

        self.original = splits.split_items
        original = self.original
        fraction = self.fraction

        def limited(task, split, search_size=None):
            items = original(task, split, search_size)
            if fraction >= 1.0:
                return items
            count = max(1, int(math.ceil(fraction * len(items))))
            return items[:count]

        splits.split_items = limited
        return self

    def __exit__(self, kind, value, trace):
        from wsp_eval import splits

        splits.split_items = self.original


def run_harness(model_path, split, tasks, fraction, note):
    """Score a model with the hf backend. Returns ({task: score}, {task: item count})."""
    from wsp_eval import harness

    with LimitedItems(fraction):
        result = harness.evaluate(model_path, split, list(tasks), namespace="studio", note=note, backend="hf")
    scores = {}
    counts = {}
    for task in tasks:
        scores[task] = result["tasks"][task]["score"]
        counts[task] = result["tasks"][task]["n"]
    return scores, counts


LOGPROB_TASKS = ["mmlu"]
MMLU_LETTERS = ["A", "B", "C", "D"]
MMLU_SEARCH = 1000
MMLU_HELDOUT = 2000


def mmlu_prompt(question, choices):
    lines = ["Question: " + question.strip()]
    for index in range(len(choices)):
        lines.append(MMLU_LETTERS[index] + ". " + str(choices[index]).strip())
    return "\n".join(lines) + "\nAnswer:"


def mmlu_items(split, fraction):
    """[(prompt, answer index)] for a split, as in rounds 6 and 7, cut to a prefix by fidelity."""
    import random

    from datasets import load_dataset

    test = load_dataset("cais/mmlu", "all", split="test")
    order = list(range(len(test)))
    random.Random(0).shuffle(order)
    if split == "heldout":
        chosen = order[MMLU_SEARCH:MMLU_SEARCH + MMLU_HELDOUT]
    else:
        chosen = order[:MMLU_SEARCH]
    if fraction < 1.0:
        chosen = chosen[:max(1, int(math.ceil(fraction * len(chosen))))]
    items = []
    for index in chosen:
        row = test[index]
        items.append((mmlu_prompt(row["question"], row["choices"]), int(row["answer"])))
    return items


def load_for_scoring(model_path):
    """(model, tokenizer) on the GPU. A compiled folder loads with wsp.load, a model id with transformers."""
    import torch

    if os.path.isdir(model_path):
        from wsp.load import load
        model, tokenizer = load(model_path, device="cuda")
    else:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16).to("cuda")
    model.eval()
    return model, tokenizer


def score_mmlu_logprob(model_path, split, fraction, batch_size=32):
    """Accuracy by the argmax over the log-probs of ' A' to ' D'. Returns (score, item count)."""
    import torch

    model, tokenizer = load_for_scoring(model_path)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    choice_ids = []
    for letter in MMLU_LETTERS:
        choice_ids.append(tokenizer(" " + letter, add_special_tokens=False).input_ids[-1])
    has_taps = hasattr(model, "wsp_tap_donors")
    items = mmlu_items(split, fraction)
    correct = 0
    for start in range(0, len(items), batch_size):
        batch = items[start:start + batch_size]
        prompts = []
        for prompt, answer in batch:
            prompts.append(prompt)
        encoded = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
        if has_taps:
            from wsp.tap import auto_context
            auto_context(model, encoded["input_ids"], encoded["attention_mask"])
        with torch.no_grad():
            logits = model(**encoded).logits[:, -1, :].float()
        if has_taps:
            from wsp.tap import clear_tap_context
            clear_tap_context(model)
        for row in range(len(batch)):
            if int(logits[row, choice_ids].argmax()) == batch[row][1]:
                correct += 1
    del model
    torch.cuda.empty_cache()
    return correct / len(items), len(items)


def score_all(model_path, split, tasks, fraction, note):
    """Score every task: log-prob MMLU here, the rest with the harness. Returns (scores, counts)."""
    scores = {}
    counts = {}
    rest = []
    for task in tasks:
        if task in LOGPROB_TASKS:
            scores[task], counts[task] = score_mmlu_logprob(model_path, split, fraction)
        else:
            rest.append(task)
    if len(rest) > 0:
        free_gpu()
        more, more_counts = run_harness(model_path, split, rest, fraction, note)
        scores.update(more)
        counts.update(more_counts)
    return scores, counts


SCORING_NOTE = "mmlu: log-prob of A to D (rounds 6 and 7 items); other tasks: harness chat prompt"


DEFAULT_GEN_BATCH = "32"
TAP_GEN_BATCH = "8"


def free_gpu():
    """Release memory the build left behind before scoring loads the model again."""
    import gc

    import torch

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def failure(error, seconds):
    return {"ok": False, "error": type(error).__name__ + ": " + str(error),
            "trace": traceback.format_exc()[-3000:], "seconds": seconds, "gpu": gpu_name()}


def probe_body(program_dict, options):
    scratch = fresh_root("studio_probe_")
    start = time.time()
    try:
        folder = os.path.join(scratch, "model")
        report, seconds = build_model(program_dict, options, folder, False)
        return {"ok": True, "error": None, "graft_report": report_entries(report),
                "seconds": {"compile": round(seconds, 1)}, "gpu": gpu_name()}
    except Exception as error:
        return failure(error, {"compile": round(time.time() - start, 1)})
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
        volume.commit()


def score_body(program_dict, split, tasks, token, fidelity, options):
    scratch = fresh_root("studio_score_")
    start = time.time()
    seconds = {}
    try:
        folder = os.path.join(scratch, "model")
        report, build_s = build_model(program_dict, options, folder, True)
        seconds = phase_seconds(report, build_s)
        free_gpu()
        if len(program_dict.get("taps") or []) > 0:
            # Each prompt also runs the donor, so generation needs a smaller batch.
            os.environ["WSP_HF_BATCH_SIZE"] = TAP_GEN_BATCH
        else:
            os.environ["WSP_HF_BATCH_SIZE"] = DEFAULT_GEN_BATCH
        eval_start = time.time()
        scores, counts = score_all(folder, split, tasks, fraction_of(fidelity), "studio " + str(token))
        seconds["eval"] = round(time.time() - eval_start, 1)
        taps = report.get("taps") or {}
        return {"ok": True, "error": None, "graft_report": report_entries(report),
                "heal_loss": heal_losses(report), "tap_loss_last10": mean_tail(taps.get("losses") or [], 10),
                "gates": taps.get("gates"), "scores": scores, "items": counts, "seconds": seconds,
                "token": token, "gpu": gpu_name(), "scoring": SCORING_NOTE}
    except Exception as error:
        seconds["failed_after"] = round(time.time() - start, 1)
        return failure(error, seconds)
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
        volume.commit()


def base_body(host, split, tasks, fidelity):
    scratch = fresh_root("studio_base_")
    start = time.time()
    try:
        scores, counts = score_all(host, split, tasks, fraction_of(fidelity), "studio host base")
        return {"ok": True, "error": None, "scores": scores, "items": counts,
                "seconds": {"eval": round(time.time() - start, 1)}, "gpu": gpu_name(),
                "scoring": SCORING_NOTE}
    except Exception as error:
        return failure(error, {"eval": round(time.time() - start, 1)})
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
        volume.commit()


COMMON = {"image": image, "volumes": {"/vol": volume}, "timeout": 7200}


@app.function(gpu="L40S", memory=65536, **COMMON)
def probe_remote(program, options):
    return probe_body(program, options)


@app.function(gpu="L40S", memory=65536, **COMMON)
def score_remote(program, split, tasks, token, fidelity, options):
    return score_body(program, split, tasks, token, fidelity, options)


@app.function(gpu="L40S", memory=65536, **COMMON)
def base_remote(host, split, tasks, fidelity="full"):
    return base_body(host, split, tasks, fidelity)


@app.function(gpu="H100", memory=131072, **COMMON)
def probe_remote_h100(program, options):
    return probe_body(program, options)


@app.function(gpu="H100", memory=131072, **COMMON)
def score_remote_h100(program, split, tasks, token, fidelity, options):
    return score_body(program, split, tasks, token, fidelity, options)
