"""The wsp studio on Modal: real GPU compile and eval for one program per call.

Deploy once (the studio server looks the functions up by name, so it does
not run inside Modal):

    MODAL_PROFILE=trisanth modal deploy studio/modal_studio.py

Each function runs one whole job in one container. It compiles the program,
trains what the program asks for (a LoRA or glue heal), and scores
the result with the eval harness (wsp_eval.harness, hf backend). The host
and every candidate go through the same harness path, so their numbers
compare. Weights are written to a scratch folder and deleted after the call.
Only the Hugging Face cache lives on the wsp-models volume.

Functions:
    probe_remote(program, options)            compile with heal off
    score_remote(program, split, tasks, token, fidelity, options)
                                              compile with training, then evaluate
    base_remote(host, split, tasks)           the host's scores
    code_version()                            hash of the deployed code (CPU only)
    shrink_remote(plan, split, tasks, fidelity, options)
                                              skill removal: slim, heal, score (see shrink_body)
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

import importlib.util

import modal

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def read_code_version():
    """The code hash baked in at deploy time. Inside the container it comes from the environment."""
    baked = os.environ.get("WSP_STUDIO_CODE_VERSION")
    if baked is not None:
        return baked
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "code_version.py")
    spec = importlib.util.spec_from_file_location("studio_code_version", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.local_hash(REPO_ROOT)


CODE_VERSION = read_code_version()

app = modal.App("wsp-studio")
volume = modal.Volume.from_name("wsp-models", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch==2.7.1", "transformers==4.57.1", "accelerate", "huggingface_hub", "requests",
        "safetensors", "zstandard", "numpy", "datasets", "sentencepiece", "protobuf",
    )
    .env({"HF_HOME": "/vol/hf", "WSP_BACKEND": "hf", "WSP_HF_BATCH_SIZE": "32",
          "WSP_STUDIO_CODE_VERSION": CODE_VERSION})
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


def report_entries(report):
    """The per-graft entries of a compile report, in the shape the backend interface wants."""
    entries = []
    for graft in report.get("grafts") or []:
        entries.append(graft)
    return entries


def build_model(program_dict, options, folder, training):
    """Compile a program into folder. Returns (compile_report, seconds).

    training False is a probe: heal is dropped.
    """
    from wsp.program import Program

    data = json.loads(json.dumps(program_dict))
    if len(data.get("taps") or []) > 0 or data.get("tap_train") is not None:
        raise ValueError("a program with taps runs a second model at inference, which Sapien does not build")
    if not training:
        data["heal"] = None
    program = Program.from_dict(data)
    limit = options.get("max_heal_min")
    if training and limit is not None and program.budget_spec is None:
        program.budget(max_heal_minutes=float(limit))
    catalog = make_catalog(options)
    texts = align_texts_for(program, options)
    extra = {}
    if not training:
        extra["heal_pairs"] = []
    start = time.time()
    program.compile(folder, catalog, align_texts=texts, **extra)
    seconds = time.time() - start
    with open(os.path.join(folder, "compile_report.json"), encoding="utf-8") as handle:
        report = json.load(handle)
    return report, seconds


def phase_seconds(report, total):
    """Split the build time: heal covers the heal pass, compile is the rest."""
    heal_s = 0.0
    heal = report.get("heal")
    if heal is not None:
        heal_s += float(heal.get("seconds") or 0.0)
    if heal_s > total:
        heal_s = total
    return {"compile": round(total - heal_s, 1), "heal": round(heal_s, 1)}


def heal_losses(report):
    """The heal pass losses."""
    heal = report.get("heal")
    if heal is not None and len(heal.get("losses") or []) > 0:
        return list(heal["losses"])
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
    items = mmlu_items(split, fraction)
    correct = 0
    for start in range(0, len(items), batch_size):
        batch = items[start:start + batch_size]
        prompts = []
        for prompt, answer in batch:
            prompts.append(prompt)
        encoded = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
        with torch.no_grad():
            logits = model(**encoded).logits[:, -1, :].float()
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
        os.environ["WSP_HF_BATCH_SIZE"] = DEFAULT_GEN_BATCH
        eval_start = time.time()
        scores, counts = score_all(folder, split, tasks, fraction_of(fidelity), "studio " + str(token))
        seconds["eval"] = round(time.time() - eval_start, 1)
        return {"ok": True, "error": None, "graft_report": report_entries(report),
                "heal_loss": heal_losses(report), "scores": scores, "items": counts, "seconds": seconds,
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


# Skill removal (wsp/slim.py). The ranking text and the heal text come from wsp.healdata pools, never from a
# benchmark. The heal distills from the original model's logits, so the heal loss has no language-model term
# on dataset text (which damages the base: results/README.md, the 2026-11 heal finding). When the original
# does not fit in memory next to the student, the heal uses the language-model loss alone and the log says so.
SHRINK_RANK_TEXTS = 256
SHRINK_RANK_LENGTH = 384
SHRINK_GENERAL_TEXTS = 256
SHRINK_HEAL = {"batch_size": 8, "seq_len": 384, "lora_rank": 16, "lora_lr": 2e-4, "router_lr": 2e-5, "warmup": 20}
SHRINK_HEADROOM_BYTES = 12e9
DENSE_RULES = {"skill": ("skill_cost", "skill"), "skill_cost": ("skill_cost", "skill"),
               "specific": ("specific", "specific"), "general": ("general", "general"),
               "random": ("random", "random")}


def teacher_fits(plan, total_params):
    """True when the original model can stay in memory as the teacher while the student trains."""
    import torch

    free, total = torch.cuda.mem_get_info()
    if plan["method"] == "experts":
        student_bytes = 4e9
    else:
        student_bytes = 2.0 * float(plan["target_params_b"]) * 1e9
    return free > student_bytes + SHRINK_HEADROOM_BYTES


def heal_summary(report, steps, teacher):
    if report is None:
        return {"steps": 0, "kind": "none", "teacher": False}
    out = {"steps": steps, "kind": "distillation" if teacher else "lm_only", "teacher": teacher}
    for key in ["kd_first", "kd_last", "lm_first", "lm_last", "trainable_params", "seconds"]:
        out[key] = report[key]
    return out


def shrink_arm(slim, model, tokenizer, plan, rule, rank_texts, heal_texts, general_texts, stats, teacher, seed):
    """Slim and heal one arm. Returns (student, slim report, heal report or None)."""
    steps = int(plan["heal_steps"])
    if plan["method"] == "experts":
        student, report = slim.slim_moe(model, tokenizer, None, None, rule=rule, keep=int(plan["keep_experts"]),
                                        seed=seed, skill_stats=stats["skill"], general_stats=stats["general"],
                                        share_experts=True)
    else:
        layer_rule, width_rule = DENSE_RULES[rule]
        student, report = slim.slim_dense(model, tokenizer, rank_texts, general_texts,
                                          target_params=float(plan["target_params_b"]) * 1e9,
                                          layer_rule=layer_rule, width_rule=width_rule, seed=seed,
                                          max_length=SHRINK_RANK_LENGTH)
    healed = None
    if steps > 0:
        lm_weight = 1.0
        if teacher is not None:
            lm_weight = 0.0
        healed = slim.heal(student, teacher, tokenizer, heal_texts, steps, batch_size=SHRINK_HEAL["batch_size"],
                           seq_len=SHRINK_HEAL["seq_len"], lora_rank=SHRINK_HEAL["lora_rank"],
                           lora_lr=SHRINK_HEAL["lora_lr"], router_lr=SHRINK_HEAL["router_lr"],
                           warmup=SHRINK_HEAL["warmup"], lm_weight=lm_weight, seed=seed)
    return student, report, healed


def removed_summary(plan, report, stats):
    """A short, JSON-safe description of what the slim removed."""
    if plan["method"] == "experts":
        counts = sorted(set(report["counts"].values()))
        before = max(len(stats["skill"]["layers"][layer]["count"]) for layer in stats["skill"]["layers"])
        out = {"kind": "moe", "experts_before": before, "experts_kept": counts[0],
               "experts_removed_per_layer": before - counts[0], "layers": len(report["counts"]),
               "top_k_after": max(report["top_k"].values()), "kept_share_of_skill": None}
        share = stats["kept_share"](report["kept"])
        out["kept_share_of_skill"] = round(sum(share.values()) / max(1, len(share)), 4)
        return out
    return {"kind": "dense", "layers_before": report["layers_before"], "layers_after": report["layers_after"],
            "layers_dropped": [int(layer) for layer in report["dropped"]], "keep_mlp": round(report["keep_mlp"], 4),
            "keep_heads": round(report["keep_heads"], 4)}


def shrink_body(plan, split, tasks, fidelity, options):
    """Slim the source model, heal it, save it, and score it, the random control and the original.

    plan is the spec's shrink block. options: control (random control with the same counts and heal steps),
    general_control, score_source. Returns {"ok", "params", "removed", "scores", "control_scores",
    "general_scores", "source_scores", "heal", "log", "seconds", "items", "gpu"} or the failure dict.
    """
    scratch = fresh_root("studio_shrink_")
    start = time.time()
    seconds = {}
    log = []
    try:
        import torch
        import transformers

        from wsp import healdata, slim

        model_id = plan["source_model"]
        skill = plan["keep_skill"]
        steps = int(plan["heal_steps"])
        healdata.check_sources([source[0] for source in healdata.SOURCES[skill]])
        phase = time.time()
        tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)
        model = transformers.AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16,
                                                                  device_map="cuda").eval()
        total_before, run_before = slim.count_params(model)
        seconds["load"] = round(time.time() - phase, 1)
        log.append("Loaded " + model_id + " with %.2fB parameters." % (total_before / 1e9))

        phase = time.time()
        need = steps * SHRINK_HEAL["batch_size"]
        texts = healdata.pool(skill, SHRINK_RANK_TEXTS + need, tokenizer=tokenizer, seed=0)
        rank_texts = texts[:SHRINK_RANK_TEXTS]
        heal_texts = texts[SHRINK_RANK_TEXTS:]
        if steps > 0 and len(heal_texts) == 0:
            heal_texts = rank_texts
        wants_general = plan["method"] == "dense" or plan["rule"] in ("selective", "general", "specific") or \
            bool(options.get("general_control"))
        general_texts = None
        if wants_general:
            general_texts = healdata.pool("general", SHRINK_GENERAL_TEXTS, tokenizer=tokenizer, seed=0)
        log.append("Skill text: the " + skill + " pool, " + str(len(rank_texts)) + " ranking texts and " +
                   str(len(heal_texts)) + " heal texts. No benchmark data is used.")
        stats = {"skill": None, "general": None}
        if plan["method"] == "experts":
            stats["skill"] = slim.collect_moe_stats(model, tokenizer, rank_texts, SHRINK_RANK_LENGTH)
            if plan["rule"] in ("selective", "general") or options.get("general_control"):
                stats["general"] = slim.collect_moe_stats(model, tokenizer, general_texts, SHRINK_RANK_LENGTH)
            skill_stats = stats["skill"]
            stats["kept_share"] = lambda kept: slim.kept_share(skill_stats, kept)
        seconds["rank"] = round(time.time() - phase, 1)

        teacher = model
        if steps > 0 and not teacher_fits(plan, total_before):
            teacher = None
            log.append("The original model does not fit in memory next to the student, so the heal uses the "
                       "language-model loss only, with no distillation.")
        elif steps > 0:
            log.append("Heal: " + str(steps) + " steps of LoRA, distilled from the original model's logits.")

        arms = [("skill", plan["rule"])]
        if options.get("control"):
            arms.append(("random", "random"))
        if options.get("general_control"):
            arms.append(("general", "general"))
        folders = {}
        reports = {}
        heals = {}
        seconds["slim"] = 0.0
        seconds["heal"] = 0.0
        heal_limit_s = None
        if options.get("max_heal_min") is not None:
            heal_limit_s = float(options["max_heal_min"]) * 60.0
        done_arms = []
        for name, rule in arms:
            if heal_limit_s is not None and len(done_arms) > 0 and seconds["heal"] >= heal_limit_s:
                log.append("Arm " + name + " was skipped: the heal time limit was used up.")
                continue
            done_arms.append((name, rule))
            phase = time.time()
            student, report, healed = shrink_arm(slim, model, tokenizer, plan, rule, rank_texts, heal_texts,
                                                 general_texts, stats, teacher, 0)
            reports[name] = report
            heals[name] = healed
            spent = time.time() - phase
            heal_s = 0.0
            if healed is not None:
                heal_s = float(healed["seconds"])
            seconds["heal"] = round(seconds["heal"] + heal_s, 1)
            seconds["slim"] = round(seconds["slim"] + spent - heal_s, 1)
            folders[name] = os.path.join(scratch, "arm_" + name)
            slim.save_slim(student, folders[name], tokenizer)
            log.append("Arm " + name + " (rule " + rule + "): %.2fB parameters, %.2fB run per token." % (
                report["params_total_after"] / 1e9, report["params_run_after"] / 1e9))
            del student
            free_gpu()
        arms = done_arms
        del teacher
        del model
        free_gpu()

        phase = time.time()
        os.environ["WSP_HF_BATCH_SIZE"] = DEFAULT_GEN_BATCH
        fraction = fraction_of(fidelity)
        scored = {}
        items = {}
        for name, rule in arms:
            scored[name], items = score_all(folders[name], split, tasks, fraction, "studio shrink " + name)
            free_gpu()
        source_scores = None
        if options.get("score_source"):
            source_scores, items = score_all(model_id, split, tasks, fraction, "studio shrink source")
            free_gpu()
        seconds["eval"] = round(time.time() - phase, 1)
        seconds["total"] = round(time.time() - start, 1)
        main = reports["skill"]
        params = {"total_before": total_before, "run_before": run_before,
                  "total_after": main["params_total_after"], "run_after": main["params_run_after"]}
        heal = heal_summary(heals["skill"], steps, teacher_used(steps, log))
        return {"ok": True, "error": None, "params": params, "removed": removed_summary(plan, main, stats),
                "scores": scored["skill"], "control_scores": scored.get("random"),
                "general_scores": scored.get("general"), "source_scores": source_scores, "heal": heal,
                "log": log, "seconds": seconds, "items": items, "gpu": gpu_name(), "scoring": SCORING_NOTE}
    except Exception as error:
        seconds["failed_after"] = round(time.time() - start, 1)
        out = failure(error, seconds)
        out["log"] = log
        return out
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
        volume.commit()


def teacher_used(steps, log):
    """True when the heal had the original as its teacher."""
    if steps <= 0:
        return False
    for line in log:
        if line.startswith("The original model does not fit"):
            return False
    return True


COMMON = {"image": image, "volumes": {"/vol": volume}, "timeout": 7200}


@app.function(image=image, timeout=120)
def code_version():
    """The hash of the code in this deploy. The server compares it with its local files."""
    return CODE_VERSION


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


@app.function(gpu="L40S", memory=65536, **COMMON)
def shrink_remote(plan, split, tasks, fidelity, options):
    return shrink_body(plan, split, tasks, fidelity, options)


@app.function(gpu="H100", memory=131072, **COMMON)
def shrink_remote_h100(plan, split, tasks, fidelity, options):
    return shrink_body(plan, split, tasks, fidelity, options)
