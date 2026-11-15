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
    synth_remote(program, split, tasks, fidelity, options)
                                              compile with wsp.synth (the default for programs with grafts),
                                              then score (see synth_body)
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

GSM8K scoring: the task "gsm8k" is scored here too, with the compiler team's
protocol (wsp.skills_eval): the chat template with the question plus "Solve
step by step and end with: The answer is N.", greedy, 512 new tokens, and the
answer taken from "The answer is N", else "#### N", else the last \\boxed{},
else the last number. The harness task prompts "Question: ... Answer:" with
no chat template and stops at 256 tokens, which cuts an instruct model's
step by step answer before it reaches the number. The items are the harness
items (search: 200 from GSM8K train, held-out: the GSM8K test set), so the
splits stay disjoint. Other tasks use the harness.
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
GENERATE_TASKS = ["gsm8k"]
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

    if os.path.isdir(model_path) and os.path.exists(os.path.join(model_path, "synth.json")):
        from wsp import synth
        model, tokenizer, _ = synth.load_synth(model_path, device="cuda", dtype=torch.bfloat16)
    elif os.path.isdir(model_path):
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


def gsm8k_items(split, fraction):
    """GSM8K items for a split, the harness items cut to a prefix by fidelity.

    Each item has "question" and "gold" (a float). Search is 200 train items,
    held-out is the test set, the same as wsp_eval, so the splits are disjoint.
    """
    from wsp_eval import tasks as tasks_module

    task = tasks_module.get_task("gsm8k")
    rows = task.split_items("heldout" if split == "heldout" else "search", 0)
    if fraction < 1.0:
        rows = rows[:max(1, int(math.ceil(fraction * len(rows))))]
    items = []
    for row in rows:
        items.append({"id": row["id"], "question": row["question"], "gold": row["answer"]})
    return items


def grade_gsm8k(items, outputs):
    """Number of items whose output holds the gold answer, by the team's extraction rules."""
    from wsp import skills_eval

    correct = 0
    for item, output in zip(items, outputs):
        predicted = skills_eval.gsm8k_prediction(output)
        if skills_eval.same_number(predicted, item["gold"]):
            correct += 1
    return correct


def score_gsm8k(model_path, split, fraction, batch_size=32):
    """GSM8K accuracy by the compiler team's protocol. Returns (score, item count)."""
    from wsp import skills_eval

    model, tokenizer = load_for_scoring(model_path)
    items = gsm8k_items(split, fraction)
    prompts = []
    for item in items:
        text = skills_eval.user_text("gsm8k", item)
        prompts.append(skills_eval.render_prompt(tokenizer, text))
    outputs = skills_eval.generate_texts(model, tokenizer, prompts,
                                         max_new_tokens=skills_eval.MAX_NEW_TOKENS["gsm8k"],
                                         batch_size=batch_size)
    correct = grade_gsm8k(items, outputs)
    del model
    free_gpu()
    return correct / len(items), len(items)


def score_all(model_path, split, tasks, fraction, note):
    """Score every task: log-prob MMLU and team-protocol GSM8K here, the rest with the harness.

    Returns (scores, counts).
    """
    scores = {}
    counts = {}
    rest = []
    for task in tasks:
        if task in LOGPROB_TASKS:
            scores[task], counts[task] = score_mmlu_logprob(model_path, split, fraction)
        elif task in GENERATE_TASKS:
            scores[task], counts[task] = score_gsm8k(model_path, split, fraction)
        else:
            rest.append(task)
    if len(rest) > 0:
        free_gpu()
        more, more_counts = run_harness(model_path, split, rest, fraction, note)
        scores.update(more)
        counts.update(more_counts)
    return scores, counts


SCORING_NOTE = ("mmlu: log-prob of A to D (rounds 6 and 7 items); "
                "gsm8k: team protocol (chat prompt with 'The answer is N', 512 new tokens, greedy; "
                "search = 200 train items, held-out = test set); other tasks: harness")


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


# Compile with wsp.synth (the staged pipeline: place, translate in, translate out, synthesis heal, gates,
# verify). Heal, fit and selection text come from wsp.healdata pools, never from a scored benchmark.
# Practice sets feed only the compile-time guard and the verdicts. They never receive gradient, and they come
# from benchmarks the request is not scored on.
SYNTH_KINDS = ["block", "attn_heads", "mlp_cluster", "expert"]
SYNTH_POOLS = ["general", "math", "code", "medicine", "prose"]
POOL_SKILL = {"math": "gsm8k", "code": "mbpp", "medicine": "medmcqa", "general": "mmlu"}
PRACTICE_SKILLS = ["gsm8k", "mbpp", "mmlu", "medmcqa"]
SYNTH_HEAL = {"batch_size": 4, "seq_len": 512, "lora_rank": 16, "steps_in": 100, "steps_lm": 60, "batch_in": 4,
              "tolerance": 0.04, "margin": 0.02, "lr_lora": 5e-5}
SYNTH_TEXTS = {"skill": 400, "general": 300, "align": 150, "probe": 48}
DEFAULT_SYNTH_HEAL_STEPS = 300
# Seconds per heal step for one component on one L40S. Measured 2026-11-14: Llama-3.2-1B host, one
# Qwen2.5-Math-1.5B neuron cluster, batch 4, length 512: 60 steps in 21.4 s (0.36 s per step). Rounded up.
SYNTH_STEP_SECONDS = 0.5


def record_map(options):
    """{part id: catalog record} from the records the server sent."""
    out = {}
    for record in options.get("catalog_records") or []:
        out[record["id"]] = record
    return out


def refusals(program, records):
    """Plain messages for everything in a program that the synthesis pipeline cannot compile. [] means it can."""
    problems = []
    if program.get("masks") is not None:
        problems.append("a deletion mask")
    if len(program.get("task_vectors") or []) > 0:
        problems.append("a task vector")
    if program.get("experts") is not None:
        problems.append("routed experts")
    for key, name in [("carve", "a carve"), ("splice", "a splice"), ("base_source", "a written model source")]:
        if program.get(key) is not None:
            problems.append(name)
    if len(program.get("taps") or []) > 0 or program.get("tap_train") is not None:
        problems.append("a tap")
    out = []
    for name in problems:
        out.append("The synthesis pipeline cannot build with " + name + " yet. Remove it, or run the classic pipeline.")
    for graft in program.get("grafts") or []:
        record = records.get(graft["id"])
        if record is None:
            out.append("The part " + str(graft["id"]) + " is not in the catalog the build received.")
        elif record.get("type") not in SYNTH_KINDS:
            out.append("The part " + str(graft["id"]) + " is a " + str(record.get("type")) +
                       ". The synthesis pipeline places " + ", ".join(SYNTH_KINDS) + " parts only.")
    if len(program.get("grafts") or []) + len(program.get("inserts") or []) == 0:
        out.append("The synthesis pipeline needs at least one part or inserted layer.")
    return out


def unique_id(base, used):
    name = base
    count = 2
    while name in used:
        name = base + "#" + str(count)
        count += 1
    used.append(name)
    return name


def synth_components(program, records, texts_by_pool, pool):
    """The component dicts wsp.synth.synthesize expects, in program order: grafts first, then inserted layers.

    Each component carries "label_id", the part id the server shows. skill is the practice skill of the
    pool when there is one, so the verdict can use the practice score without the part.
    """
    skill = POOL_SKILL.get(pool, pool)
    comps = []
    used = []
    for graft in program.get("grafts") or []:
        record = records[graft["id"]]
        comp = {"id": unique_id(graft["id"], used), "label_id": graft["id"], "kind": record["type"],
                "donor": record["model"], "skill": skill, "texts": texts_by_pool, "host_layer": graft.get("at"),
                "control": graft.get("control"), "control_seed": graft.get("control_seed", 0)}
        if record["type"] == "block":
            layers = record.get("layers")
            if layers is None:
                layers = record["layer"]
            comp["layers"] = [int(layers[0]), int(layers[-1])]
        else:
            comp["layer"] = int(record["layer"])
            if record.get("indices") is not None:
                comp["indices"] = list(record["indices"])
            else:
                comp["units"] = record["units"]
        comps.append(comp)
    for item in program.get("inserts") or []:
        label = "insert " + item["donor"].split("/")[-1] + " L" + str(item["layers"][0]) + "-" + str(item["layers"][1])
        comp = {"id": unique_id(label, used), "label_id": label, "kind": "block", "donor": item["donor"],
                "layers": [int(item["layers"][0]), int(item["layers"][1])], "skill": skill,
                "texts": texts_by_pool, "host_layer": int(item["at"]) - 1,
                "control": item.get("control"), "control_seed": item.get("control_seed", 0)}
        comps.append(comp)
    return comps


def split_heal_steps(program, options, n_components, control_run):
    """(stage 1 steps, stage 2 steps, note). The program's heal steps are split in two halves.

    A heal limit in minutes caps the total at the guessed step time (SYNTH_STEP_SECONDS per component, doubled
    when the random control compiles too).
    """
    heal = program.get("heal")
    total = DEFAULT_SYNTH_HEAL_STEPS
    if heal is not None and heal.get("steps") is not None:
        total = int(heal["steps"])
    note = None
    limit = options.get("max_heal_min")
    if limit is not None:
        runs = 2 if control_run else 1
        cap = int(float(limit) * 60.0 / (SYNTH_STEP_SECONDS * max(1, n_components) * runs))
        if cap < total:
            note = "The heal is cut from " + str(total) + " to " + str(max(20, cap)) + " steps to fit the heal limit."
            total = max(20, cap)
    first = max(10, total // 2)
    second = max(10, total - first)
    return first, second, note


def synth_budget(program, options, n_components, control_run):
    """The budget dict for synth.synthesize: steps, size limits and the heal data settings."""
    first, second, note = split_heal_steps(program, options, n_components, control_run)
    budget = dict(SYNTH_HEAL)
    budget["stage1_steps"] = first
    budget["stage2_steps"] = second
    budget["seed"] = int(program.get("seed") or 0)
    lora = program.get("lora")
    if lora is not None and lora.get("rank") is not None:
        budget["lora_rank"] = int(lora["rank"])
    if options.get("max_total_params_b") is not None:
        budget["max_params"] = int(float(options["max_total_params_b"]) * 1e9)
    if options.get("max_run_params_b") is not None:
        budget["max_params_per_token"] = int(float(options["max_run_params_b"]) * 1e9)
    for key, value in (options.get("synth_settings") or {}).items():
        budget[key] = value
    return budget, note


def pool_of(program, options):
    """The healdata pool for skill text: the program's heal data when it is a pool, else the spec's pool."""
    heal = program.get("heal") or {}
    for name in [heal.get("data"), options.get("pool"), options.get("align_data")]:
        if isinstance(name, str):
            if name in SYNTH_POOLS:
                return name
            if name.startswith("mix:"):
                first = name[4:].split(",")[0]
                if first in SYNTH_POOLS:
                    return first
    return "general"


def synth_texts(tokenizer, pool, program):
    """(texts dict for synthesize, text report). Every text is from a healdata pool."""
    from wsp import healdata

    skill_texts = healdata.pool(pool, SYNTH_TEXTS["skill"], tokenizer=tokenizer, seed=0)
    general = healdata.pool("general", SYNTH_TEXTS["general"], tokenizer=tokenizer, seed=1)
    align = healdata.pool("prose", SYNTH_TEXTS["align"], tokenizer=tokenizer, seed=3)
    for item in general[:100]:
        align.append(item)
    texts = {"align": align, "general": general}
    report = {"skill_pool": pool, "skill_texts": len(skill_texts), "general_pool": "general",
              "general_texts": len(general), "align_pool": "prose and general", "align_texts": len(align)}
    return texts, skill_texts, report


def practice_limit(fraction):
    if fraction <= 0.2:
        return 30
    if fraction <= 0.5:
        return 60
    return 100


# The practice skill of each scored task, and the healdata family its items come from. The items come from
# benchmarks that are not scored in the request.
PRACTICE_FAMILY = {"gsm8k": "math", "mmlu": "knowledge", "mbpp": "code", "medmcqa": "medicine"}
PRACTICE_NEW_TOKENS = {"number": 512, "choice": 16, "text": 32, "code": 512}


def practice_user_text(item):
    """The user message for one healdata practice item."""
    kind = item["kind"]
    if kind == "number":
        return item["prompt"] + "\n" + skills_eval_text("GSM8K_INSTRUCTION")
    if kind == "choice":
        choices = item.get("choices") or []
        if choices == ["yes", "no", "maybe"]:
            return item["prompt"] + "\nAnswer with yes, no, or maybe only."
        return item["prompt"] + "\n" + "\n".join(choices) + "\nAnswer with the letter only."
    if kind == "code":
        return item["prompt"] + "\nRead from stdin and print the answer.\n" + skills_eval_text("CODE_INSTRUCTION")
    return item["prompt"] + "\nAnswer in a few words."


def skills_eval_text(name):
    from wsp import skills_eval

    return getattr(skills_eval, name)


class PracticeScorer:
    """A practice set drawn from non-reported benchmarks. Calling it scores a model: the share of items right."""

    def __init__(self, items):
        self.items = list(items)
        self.count = len(self.items)
        self.sources = sorted(set(item["source"] for item in self.items))

    def __call__(self, model, tokenizer):
        from wsp import healdata, skills_eval

        if self.count == 0:
            return 0.0
        correct = 0
        for kind in PRACTICE_NEW_TOKENS:
            group = [item for item in self.items if item["kind"] == kind]
            if len(group) == 0:
                continue
            prompts = []
            for item in group:
                prompts.append(skills_eval.render_prompt(tokenizer, practice_user_text(item)))
            outputs = skills_eval.generate_texts(model, tokenizer, prompts, max_new_tokens=PRACTICE_NEW_TOKENS[kind],
                                                 batch_size=16)
            for item, text in zip(group, outputs):
                if healdata.check_answer(item, text):
                    correct += 1
        return correct / float(self.count)


def practice_count(spec):
    """The number of items in one practice entry (a PracticeScorer or an {"items": ...} dict)."""
    if callable(spec):
        return int(spec.count)
    return len(spec["items"])


def synth_practice(tasks, fraction):
    """{skill: PracticeScorer} for the scored tasks that have a practice set.

    Items come from wsp.healdata practice sources that are not scored in the request, so nothing that
    selects (guard, rollback, verdicts) uses a benchmark the request is scored on. A skill with no such
    source is left out: practice_gaps names it.
    """
    from wsp import healdata

    reported = list(healdata.BANNED) + [str(task) for task in tasks]
    out = {}
    count = practice_limit(fraction)
    for task in tasks:
        if task not in PRACTICE_FAMILY or task in out:
            continue
        try:
            items = healdata.practice(PRACTICE_FAMILY[task], count, reported=reported)
        except Exception:
            continue
        if len(items) > 0:
            out[task] = PracticeScorer(items)
    return out


def practice_gaps(tasks, practice, spec_guards):
    """Plain notes for scored tasks that have no practice set and for spec guards that cannot run at compile time."""
    notes = []
    for task in tasks:
        if task in PRACTICE_FAMILY and task not in practice:
            notes.append("No practice set from another benchmark exists for " + str(task) +
                         ", so no practice score or guard ran for it.")
    for guard in spec_guards or []:
        if guard not in practice:
            notes.append("The guard " + str(guard) + " has no practice set, so it is not enforced at compile time.")
    return notes


def synth_guards(spec_guards, practice):
    """The guard list for wsp.synth: the spec guards that have a practice set, else None (guard every practice skill)."""
    kept = [task for task in (spec_guards or []) if task in practice]
    if len(kept) == 0:
        return None
    return kept


def rounded(value, digits=4):
    if value is None:
        return None
    return round(float(value), digits)


def glue_params_by_id(model):
    """Parameters of each branch except the frozen part, by component id."""
    out = {}
    for branch in getattr(model, "wsp_grafts", []):
        total = 0
        for name, param in branch.named_parameters():
            if not name.startswith("part."):
                total += param.numel()
        out[branch.synth["id"]] = total
    return out


def stage_scores(scores, key):
    value = scores.get(key)
    if value is None:
        return None
    return {skill: rounded(value[skill]) for skill in value}


def stage_summaries(report, total_seconds):
    """One row per stage: name, seconds, practice score before and after, rolled_back, detail."""
    scores = report.get("scores") or {}
    translate_in = sum(float(v.get("seconds") or 0.0) for v in (report.get("translate_in") or {}).values())
    translate_out = sum(float(v.get("seconds") or 0.0) for v in (report.get("translate_out") or {}).values())
    stage1 = report.get("stage1") or {}
    stage2 = report.get("stage2") or {}
    events = (report.get("guard") or {}).get("events") or []
    rolled = False
    closed = []
    for event in events:
        if event.get("action") == "rolled back to stage 1":
            rolled = True
        if event.get("closed"):
            closed.extend(event["closed"])
    base = stage_scores(scores, "base")
    rows = [
        {"name": "place", "seconds": None, "practice_before": base, "practice_after": base, "rolled_back": False,
         "detail": "Parts attached as closed branches, so the new network equals the host."},
        {"name": "translate in", "seconds": rounded(translate_in, 1), "practice_before": base,
         "practice_after": base, "rolled_back": False, "detail": "Ridge fit, then match each part's native output."},
        {"name": "translate out", "seconds": rounded(translate_out, 1), "practice_before": base,
         "practice_after": base, "rolled_back": False, "detail": "Output map from the donor's logits."},
        {"name": "heal 1", "seconds": rounded(stage1.get("seconds"), 1), "practice_before": base,
         "practice_after": stage_scores(scores, "stage1"), "rolled_back": False,
         "detail": str(stage1.get("steps")) + " steps of the translations and gates, donor KL on skill text."},
    ]
    if stage2:
        if "skipped" in stage2:
            detail = "Skipped: " + str(stage2["skipped"]) + "."
        else:
            detail = str(stage2.get("steps")) + " steps with a LoRA after the first seam, KL to the frozen host."
        after = stage_scores(scores, "stage2")
        rows.append({"name": "heal 2", "seconds": rounded(stage2.get("seconds"), 1),
                     "practice_before": stage_scores(scores, "stage1"), "practice_after": after,
                     "rolled_back": rolled, "detail": detail})
    verify_after = stage_scores(scores, "final")
    verify_detail = "Practice scores, rollback on a guard failure, per-part test and a random control."
    if len(closed) > 0:
        verify_detail = verify_detail + " Closed gates: " + ", ".join(closed) + "."
    spent = translate_in + translate_out + float(stage1.get("seconds") or 0.0) + float(stage2.get("seconds") or 0.0)
    rows.append({"name": "verify", "seconds": rounded(max(0.0, float(report.get("seconds") or total_seconds) - spent), 1),
                 "practice_before": stage_scores(scores, "after_rollback") or stage_scores(scores, "stage2") or
                 stage_scores(scores, "stage1"), "practice_after": verify_after, "rolled_back": False,
                 "detail": verify_detail})
    return rows


def relative_error(translated):
    """The relative error of the input translation after the fit: rel_err_final, rel_err, or None.

    A donor with another tokenizer gets the ridge fit only, so it has no error number, only ridge_r2.
    """
    if translated is None:
        return None
    for key in ["rel_err_final", "rel_err"]:
        if translated.get(key) is not None:
            return translated[key]
    return None


def component_rows(report, comps, probes):
    """One row per component with its verdict and the numbers behind it."""
    gates = report.get("gates") or {}
    places = {}
    for item in report.get("place") or []:
        places[item["id"]] = item
    translate = report.get("translate_in") or {}
    rows = []
    for comp, tested in zip(comps, report.get("components") or []):
        cid = comp["id"]
        place = places.get(cid) or {}
        rel = relative_error(translate.get(cid))
        probe = probes.get(cid) or {}
        rows.append({
            "id": cid, "part_id": comp["label_id"], "kind": comp["kind"], "donor": comp["donor"],
            "skill": comp.get("skill"), "host_layer": tested.get("host_layer"), "verdict": tested.get("verdict"),
            "closed": bool(tested.get("closed")), "contribution": rounded(tested.get("contribution")),
            "evidence": "practice" if tested.get("contribution") is not None else "text",
            "text_gain": rounded(tested.get("text_gain")), "content_gain": rounded(tested.get("content_gain")),
            "heldout_loss_on": rounded(tested.get("heldout_loss_on")),
            "heldout_loss_off": rounded(tested.get("heldout_loss_off")),
            "scores_without": stage_scores(tested, "scores_without") if tested.get("scores_without") else None,
            "gate_domain": rounded((gates.get("domain:" + cid) or {}).get(cid)),
            "gate_general": rounded((gates.get("general") or {}).get(cid)),
            "fit_error": rounded(rel), "fit_r2": rounded((translate.get(cid) or {}).get("ridge_r2")),
            "part_params": place.get("part_params"),
            "transfer": rounded(probe.get("transfer")), "donor_kl_base": rounded(probe.get("donor_kl_base")),
            "donor_kl_written": rounded(probe.get("donor_kl_written")),
        })
    return rows


def graft_entries(report, comps, glue):
    """The classic per-graft report entries, so the builder shows fit and size the same way for both pipelines."""
    places = {}
    for item in report.get("place") or []:
        places[item["id"]] = item
    translate = report.get("translate_in") or {}
    entries = []
    for comp in comps:
        cid = comp["id"]
        rel = relative_error(translate.get(cid))
        fit = (translate.get(cid) or {}).get("ridge_r2")
        if fit is not None:
            fit = round(float(fit), 4)
        elif rel is not None:
            fit = round(max(0.0, 1.0 - float(rel)), 4)
        place = places.get(cid) or {}
        entries.append({"id": comp["label_id"], "host_layer": place.get("host_layer"), "fit": "ridge",
                        "fit_r2": fit, "fit_pairs": 0, "part_params": place.get("part_params"),
                        "glue_params": glue.get(cid), "donor": comp["donor"], "control": comp.get("control")})
    return entries


def run_probes(folder, base_id, comps, skill_texts, tokenizer, pool):
    """Benchmark-free probes on held-out pool text: drift from the host and per-part transfer.

    Returns (drift dict or None, {component id: transfer dict}, notes). A failure gives a note, not an error.
    """
    import torch
    import transformers

    from wsp import healdata, probe, synth

    notes = []
    drift = None
    transfer = {}
    try:
        general = healdata.pool("general", SYNTH_TEXTS["probe"], tokenizer=tokenizer, seed=9)
        held = healdata.pool(pool, SYNTH_TEXTS["probe"], tokenizer=tokenizer, seed=9)
        written, written_tokenizer = load_for_scoring(folder)
        base = transformers.AutoModelForCausalLM.from_pretrained(base_id, torch_dtype=torch.bfloat16).to("cuda").eval()
        drift = probe.drift(written, base, tokenizer, general)
        drift = {"base_kl": rounded(drift["base_kl"], 5), "agree": rounded(drift["agree"]), "tokens": drift["tokens"],
                 "text": "general pool, held out"}
        donors = {}
        for comp in comps:
            try:
                if comp["donor"] not in donors:
                    from wsp.assemble import load_donor
                    donors[comp["donor"]] = load_donor(comp["donor"], "cuda")
                donor, donor_tokenizer = donors[comp["donor"]]
                if not synth.same_vocab(tokenizer, donor_tokenizer):
                    notes.append("Transfer for " + comp["id"] + " is not measured: the donor has another tokenizer.")
                    continue
                transfer[comp["id"]] = probe.transfer(written, base, donor, tokenizer, held)
            except Exception as error:
                notes.append("Transfer for " + comp["id"] + " failed: " + type(error).__name__)
        del written
        del base
        donors.clear()
        free_gpu()
    except Exception as error:
        notes.append("The drift probe failed: " + type(error).__name__ + ": " + str(error)[:200])
    return drift, transfer, notes


def synth_failure(error, seconds, refused=False):
    out = failure(error, seconds)
    out["pipeline"] = "synth"
    out["refused"] = refused
    return out


def synth_body(program_dict, split, tasks, fidelity, options):
    """Compile a program with wsp.synth, save the one network, score it, and report the verdicts.

    options: catalog_records, pool, pipeline ("synth" or "classic"), max_heal_min, max_total_params_b,
    max_run_params_b, guards (task ids kept within tolerance of the host), control_run (default True),
    probes (default True), token, synth_settings. Returns {"ok", "pipeline", "scores", "items", "components",
    "stages", "drift", "sizes", "seconds", "graft_report", "heal_loss", ...} or the failure dict.
    """
    program = json.loads(json.dumps(program_dict))
    if options.get("pipeline") == "classic":
        out = score_body(program, split, tasks, options.get("token"), fidelity, options)
        out["pipeline"] = "classic"
        return out
    start = time.time()
    seconds = {}
    scratch = fresh_root("studio_synth_")
    try:
        records = record_map(options)
        problems = refusals(program, records)
        if len(problems) > 0:
            return synth_failure(ValueError(" ".join(problems)), {"total": round(time.time() - start, 1)}, True)
        from transformers import AutoTokenizer

        from wsp import synth

        fraction = fraction_of(fidelity)
        pool = pool_of(program, options)
        host = program["host"]
        tokenizer = AutoTokenizer.from_pretrained(host)
        texts, skill_texts, text_report = synth_texts(tokenizer, pool, program)
        comps = synth_components(program, records, skill_texts, pool)
        all_random = True
        for comp in comps:
            if comp.get("control") != "random":
                all_random = False
        control_run = bool(options.get("control_run", True)) and not all_random
        budget, heal_note = synth_budget(program, options, len(comps), control_run)
        budget["control_run"] = control_run
        if len(comps) > 4:
            budget["components_per_step"] = 3
        practice = synth_practice(tasks, fraction)
        spec_guards = options.get("guards")
        guards = synth_guards(spec_guards, practice)
        gap_notes = practice_gaps(tasks, practice, spec_guards)
        budget["test_limit"] = max(10, practice_limit(fraction) // 2)
        seconds["prepare"] = round(time.time() - start, 1)

        folder = os.path.join(scratch, "model")
        phase = time.time()
        model, tok, report = synth.synthesize(host, comps, texts, budget, practice=practice, guard=guards,
                                              out_dir=folder)
        seconds["synthesize"] = round(time.time() - phase, 1)
        glue = glue_params_by_id(model)
        sizes = synth.param_counts(model)
        del model
        free_gpu()

        probes = {}
        drift = None
        notes = list(gap_notes)
        if heal_note is not None:
            notes.append(heal_note)
        if options.get("probes", True):
            phase = time.time()
            drift, probes, more = run_probes(folder, host, comps, skill_texts, tokenizer, pool)
            notes.extend(more)
            seconds["probe"] = round(time.time() - phase, 1)
        os.environ["WSP_HF_BATCH_SIZE"] = DEFAULT_GEN_BATCH
        phase = time.time()
        scores, counts = score_all(folder, split, tasks, fraction, "studio synth " + str(options.get("token")))
        seconds["eval"] = round(time.time() - phase, 1)
        heal = float(report.get("heal_seconds") or 0.0)
        seconds["heal"] = round(heal, 1)
        seconds["compile"] = round(max(0.0, float(report.get("seconds") or 0.0) - heal), 1)
        seconds["total"] = round(time.time() - start, 1)
        steps = (report.get("stage1") or {}).get("steps"), (report.get("stage2") or {}).get("steps")
        losses = list((report.get("stage1") or {}).get("losses") or []) + \
            list((report.get("stage2") or {}).get("losses") or [])
        return {"ok": True, "error": None, "pipeline": "synth", "refused": False, "scores": scores, "items": counts,
                "components": component_rows(report, comps, probes), "stages": stage_summaries(report, seconds["total"]),
                "drift": drift, "sizes": {"total": sizes["total"], "per_token": sizes["per_token"],
                                          "parts": sizes["parts"], "glue": sizes["glue"], "gates": sizes["gates"],
                                          "lora": sizes["lora"]},
                "seconds": seconds, "graft_report": graft_entries(report, comps, glue), "heal_loss": losses,
                "heal_steps": list(steps), "texts": text_report, "control_run": control_run,
                "practice": {skill: practice_count(practice[skill]) for skill in practice},
                "guard": report.get("guard"), "notes": notes, "gpu": gpu_name(), "scoring": SCORING_NOTE}
    except Exception as error:
        seconds["failed_after"] = round(time.time() - start, 1)
        refused = type(error).__name__ == "BudgetError"
        return synth_failure(error, seconds, refused)
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
        volume.commit()


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


@app.function(gpu="L40S", memory=65536, **COMMON)
def synth_remote(program, split, tasks, fidelity, options):
    return synth_body(program, split, tasks, fidelity, options)


@app.function(gpu="H100", memory=131072, **COMMON)
def synth_remote_h100(program, split, tasks, fidelity, options):
    return synth_body(program, split, tasks, fidelity, options)
