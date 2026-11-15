"""Round 21: same-family composition with a search (wsp/weave.py).

One network of exactly the host's size. For each family, the host and its
same-lineage specialists give per-tensor differences. A diagonal CMA-ES
searches one mixing weight per (specialist, layer bucket, attention or MLP)
plus one density per specialist, on practice sets, and the best genome is
written as a plain Hugging Face model. Nothing runs beside the host.

Cells (one per family; the lineage notes come from the 2026-11-14 scout):
  qwen15    Qwen2.5-1.5B-Instruct with the Coder and Math 1.5B Instruct
            models. The scout found both far from the host (cosine about 0.5
            and 0.2, Math with rope_theta 10000), so the gate is expected to
            reject them and the cell ends with the gate numbers. --ignore-gate
            runs the search anyway to show what a rejected member does.
  qwen7b    Qwen2.5-7B-Instruct with OpenThinker-7B and
            OpenCodeReasoning-Nemotron-7B (both trained from the host).
  llama8b   Llama-3.1-8B-Instruct with OpenMath2 and R1-Distill-Llama-8B
            (parent: Llama-3.1-8B base), HuatuoGPT-o1-8B (parent: the host),
            Med42 (parent: Llama-3-8B-Instruct).
  gemma27   Gemma-3-27B-it with MedGemma-27B-text and MeditronFO
            (reference: the host; the pt checkpoint is gated).
  qwen32b   Qwen2.5-32B-Instruct with QwQ-32B and Baichuan-M2-32B (parent:
            Qwen2.5-32B base), OpenCodeReasoning-Nemotron-32B and
            OpenThinker-32B (parent: the host).
  lfm2      LFM2-1.2B with the Tool, RAG, and Extract fine-tunes. Their
            skills have no scorer in wsp.skills_eval, so this cell is
            exploratory: target GSM8K, guard MMLU, no go rule.
  qwen3_8b  Qwen3-8B with II-Medical-8B and LegalEvolver-Qwen3-8B (legal has
            no scorer here; it enters as a member with no target skill).

Targets are the members' skills: math (GSM8K, MATH-500), code (MBPP;
HumanEval is scored on the final test only, as transfer), medicine
(MedMCQA). The guard skill is MMLU. Practice sets are two disjoint splits of
about 150 items per skill from data the final test never uses (but the
members were trained on much of it: GSM8K, MATH, MBPP, and MedMCQA train data
or data derived from it, so member scores on practice are inflated against the
clean final test; the row records member_practice_minus_final per skill and
practice_contaminated): GSM8K train,
MATH train (DigitalLearningGmbH/MATH-lighteval), MBPP ids 511 to 974 (the
full train and validation splits; the sanitized test split is ids 11 to 510),
MedMCQA train, MMLU validation. Final test sets match round 16b: GSM8K 300
seeded, MBPP sanitized 200 seeded, MMLU dev 1000, MedMCQA 1000; MATH-500 uses
a seeded 250 and HumanEval all 164.

Scoring uses wsp.skills_eval with the host's tokenizer and chat template for
every arm, including the members (a member is scored in the host's slot:
its attention, MLP, and norm tensors are written into the host; embeddings
and the output head stay the host's). The prompts are identical for every arm.

Arms on the final test: host, each member, the searched genome, uniform
average, task arithmetic (scale picked on practice split A from 0.25, 0.5,
1.0), corrected TIES (keep 0.5, scale picked on A from 0.5, 1.0), and the
random control (the best genome on differences shuffled inside each tensor).

Go rule, at exactly the host's size: the searched mix beats the host on every
target skill by more than 2 points with the MMLU guard within 1 point, its
final-test objective beats uniform, task arithmetic, and TIES, and the random
control stays within 2 points of the host on every target skill and scores
below the searched mix. Rows also give,
per target skill, the share of the host-to-best-member gap the mix closes.

    modal run modal_round21.py --family qwen7b
    modal run modal_round21.py --family all --shard 0 --nshards 3
    modal run modal_round21.py --family qwen32b --plan-only

Writes results/round21_shard<i>.json (rows), the full search log to
/vol/results/round21/<family>.json, and the best model to
/vol/written/round21_<family> (weave.json beside the weights).
"""

import hashlib
import json
import math
import os
import random
import time

import modal

app = modal.App("wsp-round21")
volume = modal.Volume.from_name("wsp-models", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch==2.7.1", "transformers==4.57.1", "accelerate", "huggingface_hub", "requests",
        "safetensors", "numpy", "datasets", "sentencepiece", "protobuf",
    )
    .env({"HF_HOME": "/vol/hf"})
    .add_local_python_source("wsp")
)

N_BUCKETS = 8
DENSITY_STEP = 0.1
POPULATION = 12
GENERATIONS = 8
PRACTICE_ITEMS = 150
SEARCH_SEED = 0
CONTROL_SEED = 1
FINAL_COUNTS = {"gsm8k": 300, "math500": 250, "mbpp": 200, "humaneval": 164, "mmlu": 1000, "medmcqa": 1000}
GENERATIVE = ["gsm8k", "math500", "humaneval", "mbpp"]
MAX_NEW = {"gsm8k": 512, "math500": 1024, "humaneval": 512, "mbpp": 512}
TA_SCALES = [0.25, 0.5, 1.0]
TIES_SCALES = [0.5, 1.0]
TIES_KEEP = 0.5
GATE_MAX_REL = 0.3
GATE_MIN_COS = 0.97
GO_POINTS = 0.02
GUARD_POINTS = 0.01
PAGEABLE_GBPS = 6.0
PRACTICE_SEEDS = {"gsm8k": 11, "math500": 12, "mbpp": 13, "medmcqa": 3, "mmlu": 5}

FAMILIES = {
    "qwen15": {
        "host": "Qwen/Qwen2.5-1.5B-Instruct", "params_b": 1.5, "targets": ["gsm8k", "mbpp"], "guards": ["mmlu"],
        "transfer": ["humaneval"], "refs": {},
        "members": [
            {"name": "coder", "id": "Qwen/Qwen2.5-Coder-1.5B-Instruct", "skill": "mbpp", "ref": None},
            {"name": "math", "id": "Qwen/Qwen2.5-Math-1.5B-Instruct", "skill": "gsm8k", "ref": None},
        ],
        "memory_gib": 256, "gpu": "H200",
    },
    "qwen7b": {
        "host": "Qwen/Qwen2.5-7B-Instruct", "params_b": 7.6, "targets": ["gsm8k", "math500", "mbpp"],
        "guards": ["mmlu"], "transfer": ["humaneval"], "refs": {},
        "members": [
            {"name": "openthinker", "id": "open-thoughts/OpenThinker-7B", "skill": "math500", "ref": None},
            {"name": "ocr_nemotron", "id": "nvidia/OpenCodeReasoning-Nemotron-7B", "skill": "mbpp", "ref": None},
        ],
        "memory_gib": 256, "gpu": "H200",
    },
    "llama8b": {
        "host": "unsloth/Meta-Llama-3.1-8B-Instruct", "params_b": 8.0, "targets": ["gsm8k", "math500", "medmcqa"],
        "guards": ["mmlu"], "transfer": [],
        "refs": {"base31": "unsloth/Meta-Llama-3.1-8B", "llama3_instruct": "unsloth/llama-3-8b-Instruct"},
        "members": [
            {"name": "openmath2", "id": "nvidia/OpenMath2-Llama3.1-8B", "skill": "gsm8k", "ref": "base31"},
            {"name": "r1_llama", "id": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B", "skill": "math500",
             "ref": "base31"},
            {"name": "huatuo", "id": "FreedomIntelligence/HuatuoGPT-o1-8B", "skill": "medmcqa", "ref": None},
            {"name": "med42", "id": "m42-health/Llama3-Med42-8B", "skill": "medmcqa", "ref": "llama3_instruct"},
        ],
        "memory_gib": 256, "gpu": "H200",
    },
    "gemma27": {
        "host": "unsloth/gemma-3-27b-it", "params_b": 27.4, "targets": ["medmcqa"], "guards": ["mmlu"],
        "transfer": [], "refs": {},
        "members": [
            {"name": "medgemma", "id": "unsloth/medgemma-27b-text-it", "skill": "medmcqa", "ref": None},
            {"name": "meditron", "id": "EPFLiGHT/Gemma-3-27B-MeditronFO", "skill": "medmcqa", "ref": None},
        ],
        "memory_gib": 576, "gpu": "H200",
    },
    "qwen32b": {
        "host": "Qwen/Qwen2.5-32B-Instruct", "params_b": 32.8, "targets": ["gsm8k", "mbpp", "medmcqa"],
        "guards": ["mmlu"], "transfer": [], "refs": {"base32": "Qwen/Qwen2.5-32B"},
        "members": [
            {"name": "qwq", "id": "Qwen/QwQ-32B", "skill": "gsm8k", "ref": "base32"},
            {"name": "baichuan_m2", "id": "baichuan-inc/Baichuan-M2-32B", "skill": "medmcqa", "ref": "base32"},
            {"name": "ocr_nemotron", "id": "nvidia/OpenCodeReasoning-Nemotron-32B", "skill": "mbpp", "ref": None},
            {"name": "openthinker", "id": "open-thoughts/OpenThinker-32B", "skill": "gsm8k", "ref": None},
        ],
        "memory_gib": 576, "gpu": "H200",
    },
    "lfm2": {
        "host": "LiquidAI/LFM2-1.2B", "params_b": 1.2, "targets": ["gsm8k"], "guards": ["mmlu"], "transfer": [],
        "refs": {}, "exploratory": True,
        "members": [
            {"name": "tool", "id": "LiquidAI/LFM2-1.2B-Tool", "skill": "tool", "ref": None},
            {"name": "rag", "id": "LiquidAI/LFM2-1.2B-RAG", "skill": "rag", "ref": None},
            {"name": "extract", "id": "LiquidAI/LFM2-1.2B-Extract", "skill": "extract", "ref": None},
        ],
        "memory_gib": 256, "gpu": "H200",
    },
    "qwen3_8b": {
        "host": "Qwen/Qwen3-8B", "params_b": 8.2, "targets": ["medmcqa"], "guards": ["mmlu"], "transfer": [],
        "refs": {},
        "members": [
            {"name": "ii_medical", "id": "Intelligent-Internet/II-Medical-8B", "skill": "medmcqa", "ref": None},
            {"name": "legal", "id": "yxy919/LegalEvolver-Qwen3-8B", "skill": "legal", "ref": None},
        ],
        "memory_gib": 256, "gpu": "H200",
    },
}
BIG = ["gemma27", "qwen32b"]


# ---------------------------------------------------------------- plan (no torch)


def step_seconds(params_b):
    """Rough seconds per HF generate step at batch 32 on one H200 (an estimate, not measured)."""
    return 0.022 + 0.0015 * params_b


def eval_seconds(params_b, skills, items):
    """Estimated seconds to score the skills with items per skill."""
    total = 0.0
    for skill in skills:
        if skill in GENERATIVE:
            batches = math.ceil(items / 32)
            total += batches * MAX_NEW[skill] * step_seconds(params_b) * 0.85
        else:
            total += items * 0.012 * max(1.0, params_b / 7.0)
    return total


def final_items(skill):
    return FINAL_COUNTS[skill]


def final_seconds(params_b, skills):
    total = 0.0
    for skill in skills:
        total += eval_seconds(params_b, [skill], FINAL_COUNTS[skill])
    return total


def apply_seconds(fam):
    """Rough seconds for one apply, plus trimming.

    Each member term moves the member tensor and its reference (the host copy
    when the member has no separate reference) from pageable CPU memory, and
    each tensor moves the host copy once for the accumulator: (2 * members + 1)
    model copies at 2 bytes per parameter. Pageable host to GPU copies run at
    about 6 GB/s.
    """
    copies = 2 * len(fam["members"]) + 1
    return copies * fam["params_b"] * 2.0 / PAGEABLE_GBPS + 3.0


def make_plan(name, budget_hours=10.0, population=POPULATION, generations=GENERATIONS, items=PRACTICE_ITEMS):
    """Return the plan, shrinking generations, then items, then population until it fits the budget."""
    fam = FAMILIES[name]
    params_b = fam["params_b"]
    skills = fam["targets"] + fam["guards"]
    members = len(fam["members"])
    final_skills = fam["targets"] + fam["guards"] + fam["transfer"]
    budget = budget_hours * 3600.0

    def total(pop, gens, n_items):
        candidate = eval_seconds(params_b, skills, n_items) + apply_seconds(fam)
        evals = (2 + members) + pop * gens
        search = evals * candidate + gens * candidate
        reference = (1 + members) * 2 * eval_seconds(params_b, skills, n_items)
        fixed = (len(TA_SCALES) + len(TIES_SCALES)) * candidate
        final = (1 + members + 5) * final_seconds(params_b, final_skills)
        load = 600.0 + 900.0 * members * params_b / 8.0 / 4.0
        return load + reference + search + fixed + final, {"search": search, "reference": reference,
                                                           "fixed_recipes": fixed, "final": final, "load": load,
                                                           "candidate_eval": candidate, "evals": evals}

    pop, gens, n_items = population, generations, items
    seconds, parts = total(pop, gens, n_items)
    while seconds > budget and gens > 5:
        gens -= 1
        seconds, parts = total(pop, gens, n_items)
    while seconds > budget and n_items > 64:
        n_items = max(64, n_items - 16)
        seconds, parts = total(pop, gens, n_items)
    while seconds > budget and pop > 10:
        pop -= 1
        seconds, parts = total(pop, gens, n_items)
    return {"family": name, "population": pop, "generations": gens, "practice_items": n_items,
            "candidate_evals": parts["evals"], "expected_minutes": round(seconds / 60.0, 1),
            "fits_budget": seconds <= budget, "budget_hours": budget_hours,
            "minutes": {k: round(v / 60.0, 1) for k, v in parts.items() if k != "evals"},
            "gpu_hours": round(seconds / 3600.0, 2), "memory_gib": fam["memory_gib"],
            "cpu_ram_gib_estimate": cpu_ram_gib(name)}


def cpu_ram_gib(name):
    """Members at 2 bytes per parameter, one copy of each distinct reference, and the host base copy."""
    fam = FAMILIES[name]
    copies = len(fam["members"]) + len(fam["refs"]) + 1
    return round(copies * fam["params_b"] * 2.0 * 0.93, 0)


def print_plan(plan):
    print(f"plan {plan['family']}: population {plan['population']}, generations {plan['generations']}, "
          f"practice items {plan['practice_items']}, candidate evals {plan['candidate_evals']}")
    print(f"  expected {plan['expected_minutes']} min ({plan['gpu_hours']} H200 hours), "
          f"budget {plan['budget_hours']} h, fits {plan['fits_budget']}")
    print(f"  minutes by part: {plan['minutes']}")
    print(f"  CPU RAM about {plan['cpu_ram_gib_estimate']} GiB, container memory {plan['memory_gib']} GiB")


# ---------------------------------------------------------------- items (run on Modal)


def final_items_for(ev, skills):
    """The final test items of the skills."""
    items = {}
    for skill in skills:
        if skill == "gsm8k":
            items[skill] = ev.gsm8k_items(count=FINAL_COUNTS["gsm8k"], seed=0)
        elif skill == "mbpp":
            rows = ev.mbpp_items()
            order = list(range(len(rows)))
            random.Random(0).shuffle(order)
            items[skill] = [rows[i] for i in sorted(order[:FINAL_COUNTS["mbpp"]])]
        elif skill == "math500":
            rows = ev.math500_items()
            order = list(range(len(rows)))
            random.Random(0).shuffle(order)
            items[skill] = [rows[i] for i in sorted(order[:FINAL_COUNTS["math500"]])]
        elif skill == "humaneval":
            items[skill] = ev.humaneval_items()
        elif skill == "mmlu":
            items[skill] = ev.mmlu_items(count=FINAL_COUNTS["mmlu"])
        elif skill == "medmcqa":
            items[skill] = ev.medmcqa_items(count=FINAL_COUNTS["medmcqa"])
    return items


def practice_items_for(ev, skills, count):
    """Two disjoint practice splits per skill from data the final test does not use. Returns (A, B)."""
    from datasets import load_dataset

    a = {}
    b = {}
    for skill in skills:
        pool = []
        if skill == "gsm8k":
            rows = load_dataset("openai/gsm8k", "main", split="train")
            order = list(range(len(rows)))
            random.Random(PRACTICE_SEEDS["gsm8k"]).shuffle(order)
            for index in order[:2 * count]:
                row = rows[index]
                pool.append({"id": f"gsm8k_train:{index}", "question": row["question"], "answer": row["answer"],
                             "gold": ev.final_number(row["answer"])})
        elif skill == "math500":
            rows = load_dataset("DigitalLearningGmbH/MATH-lighteval", split="train")
            order = list(range(len(rows)))
            random.Random(PRACTICE_SEEDS["math500"]).shuffle(order)
            for index in order:
                row = rows[index]
                gold = ev.last_boxed(row["solution"])
                if gold is None:
                    continue
                pool.append({"id": f"math_train:{index}", "problem": row["problem"], "gold": gold,
                             "subject": row.get("type"), "level": row.get("level")})
                if len(pool) >= 2 * count:
                    break
        elif skill == "mbpp":
            rows = []
            for split in ["train", "validation"]:
                part = load_dataset("google-research-datasets/mbpp", "full", split=split)
                for index in range(len(part)):
                    rows.append(part[index])
            random.Random(PRACTICE_SEEDS["mbpp"]).shuffle(rows)
            for row in rows[:2 * count]:
                setup = row.get("test_setup_code") or ""
                imports = [setup] if len(setup.strip()) > 0 else []
                pool.append({"id": f"mbpp_full:{row['task_id']}", "text": row["text"],
                             "test_list": list(row["test_list"]), "test_imports": imports})
        elif skill == "medmcqa":
            rows = load_dataset("openlifescienceai/medmcqa", split="train")
            order = list(range(len(rows)))
            random.Random(PRACTICE_SEEDS["medmcqa"]).shuffle(order)
            for index in order[:2 * count]:
                row = rows[index]
                choices = [row["opa"], row["opb"], row["opc"], row["opd"]]
                pool.append({"id": f"medmcqa_train:{index}", "prompt": ev.mcq_prompt(row["question"], choices),
                             "gold": int(row["cop"]), "subject": row.get("subject_name") or "unknown"})
        elif skill == "mmlu":
            rows = load_dataset("cais/mmlu", "all", split="validation")
            order = list(range(len(rows)))
            random.Random(PRACTICE_SEEDS["mmlu"]).shuffle(order)
            for index in order[:2 * count]:
                row = rows[index]
                pool.append({"id": f"mmlu_val:{index}", "prompt": ev.mcq_prompt(row["question"], row["choices"]),
                             "gold": int(row["answer"]), "subject": row["subject"]})
        else:
            raise ValueError(f"no practice set for {skill}")
        a[skill] = pool[:count]
        b[skill] = pool[count:2 * count]
    return a, b


def disjoint(a, b):
    """True when no practice item id in a appears in b."""
    for skill in a:
        ids_a = set(item["id"] for item in a[skill])
        for item in b[skill]:
            if item["id"] in ids_a:
                return False
    return True


# ---------------------------------------------------------------- the cell


def download(repo, patterns=("*.safetensors", "*.json", "*.model", "*.jinja", "merges.txt", "tokenizer*")):
    """Fetch a repo into the volume cache and commit, so a later failure keeps the download."""
    from huggingface_hub import snapshot_download

    path = snapshot_download(repo, allow_patterns=list(patterns))
    volume.commit()
    return path


def revision_of(repo):
    """The commit sha the download resolved to, or None when the Hub call fails."""
    try:
        from huggingface_hub import HfApi

        return HfApi().model_info(repo).sha
    except Exception as error:
        return "unknown: " + repr(error)[:120]


def run_family(name, budget_hours=10.0, ignore_gate=False, use_layer_map=False):
    import torch
    import transformers

    from wsp import skills_eval as ev
    from wsp import weave

    started = time.time()
    fam = FAMILIES[name]
    plan = make_plan(name, budget_hours)
    print_plan(plan)
    row = {"family": name, "host": fam["host"], "plan": plan, "exploratory": bool(fam.get("exploratory")),
           "targets": fam["targets"], "guards": fam["guards"], "transfer": fam["transfer"]}
    host_dir = download(fam["host"])
    tokenizer = transformers.AutoTokenizer.from_pretrained(host_dir)
    assert tokenizer.chat_template, f"{fam['host']}: no chat template loaded; prompts would fall back to a plain format"
    row["chat_template_md5"] = hashlib.md5(tokenizer.chat_template.encode()).hexdigest()
    repos = [fam["host"]] + list(fam["refs"].values()) + [m["id"] for m in fam["members"]]
    row["revisions"] = {repo: revision_of(repo) for repo in repos}
    row["seeds"] = {"practice": PRACTICE_SEEDS, "search": SEARCH_SEED, "control": CONTROL_SEED}
    row["practice_contaminated"] = True
    model = transformers.AutoModelForCausalLM.from_pretrained(host_dir, dtype=torch.bfloat16, device_map="cuda")
    model.eval()
    row["params_host"] = ev.model_params(model)[0]
    host_config = model.config

    # References shared between members, then members.
    refs = {}
    for key, repo in fam["refs"].items():
        refs[key] = weave.reference_from_directory(download(repo), model, groups=("attn", "mlp", "norm"))
        print(f"reference {key} ({repo}): {len(refs[key])} tensors")
    members = []
    gates = {}
    for spec in fam["members"]:
        directory = download(spec["id"])
        config = transformers.AutoConfig.from_pretrained(directory)
        mlp_only = weave.positional_differs(host_config, config)
        ref = refs[spec["ref"]] if spec["ref"] is not None else None
        member = weave.member_from_directory(spec["name"], spec["skill"], directory, model, ref=ref,
                                             ref_name=spec["ref"] or "host", mlp_only=mlp_only,
                                             groups=("attn", "mlp", "norm"),
                                             meta={"id": spec["id"], "reference_repo": fam["refs"].get(spec["ref"])})
        gate = weave.gate_member(member, model, max_rel=GATE_MAX_REL, min_cos=GATE_MIN_COS)
        gate["mlp_only"] = mlp_only
        gate["tensors"] = len(member.tensors)
        gates[spec["name"]] = gate
        print(f"member {spec['name']}: gate {gate}")
        if gate["passed"] or ignore_gate:
            members.append(member)
    row["gate"] = gates
    row["members_used"] = [m.name for m in members]
    if len(members) == 0:
        row["status"] = "no_admissible_member"
        row["seconds"] = round(time.time() - started, 1)
        return row

    weaver = weave.Weaver(model, members, n_buckets=N_BUCKETS, density_step=DENSITY_STEP)
    count = len(members)
    names = [m.name for m in members]
    skills = fam["targets"] + fam["guards"]
    final_skills = skills + fam["transfer"]
    items_a, items_b = practice_items_for(ev, skills, plan["practice_items"])
    assert disjoint(items_a, items_b)
    finals = final_items_for(ev, final_skills)

    def score_on(items, wanted):
        return ev.evaluate_suite(model, tokenizer, skills=wanted, items=items, log=False)["scores"]

    def score_a(m):
        return score_on(items_a, skills)

    def score_b(m):
        return score_on(items_b, skills)

    # The rendered prompt is the same for every arm; keep one sample.
    first = finals[fam["targets"][0]][0]
    if fam["targets"][0] in GENERATIVE:
        row["prompt_sample"] = ev.render_prompt(tokenizer, ev.user_text(fam["targets"][0], first))
    else:
        row["prompt_sample"] = first["prompt"]

    # Reference scores: host and each member, on both practice splits and on the final test.
    ref_a = {"host": score_a(model)}
    ref_b = {"host": score_b(model)}
    ref_f = {"host": score_on(finals, final_skills)}
    for index, member in enumerate(members):
        weaver.load_member(model, index)
        ref_a[member.name] = score_a(model)
        ref_b[member.name] = score_b(model)
        ref_f[member.name] = score_on(finals, final_skills)
        weaver.restore(model)
        print(f"member {member.name}: final {ref_f[member.name]}")
    row["reference_scores"] = {"practice_a": ref_a, "practice_b": ref_b, "final": ref_f}
    member_a = [ref_a[n] for n in names]
    member_b = [ref_b[n] for n in names]
    objective_a = weave.Objective.from_reference(fam["targets"], fam["guards"], ref_a["host"], member_a)
    objective_b = weave.Objective.from_reference(fam["targets"], fam["guards"], ref_b["host"], member_b)
    objective_f = weave.Objective.from_reference(fam["targets"], fam["guards"], ref_f["host"],
                                                 [ref_f[n] for n in names])

    init = None
    if use_layer_map:
        try:
            init = layer_map_prior(model, tokenizer, items_a, members, ev, weave)
            row["layer_map_init"] = True
        except Exception as error:
            print("layer map failed, uniform start:", repr(error)[:300])
            row["layer_map_init"] = False

    os.makedirs("/vol/results/round21", exist_ok=True)
    progress_path = f"/vol/results/round21/{name}.progress.jsonl"
    last = {"generation": -1}

    def progress(entry):
        with open(progress_path, "a") as handle:
            handle.write(json.dumps(entry, default=str) + "\n")
        if entry["generation"] != last["generation"]:
            last["generation"] = entry["generation"]
            volume.commit()

    result = weave.search(weaver, model, score_a, score_b, objective_a, objective_b,
                          popsize=plan["population"], generations=plan["generations"], seed=SEARCH_SEED,
                          init=init, search_density=True, density_range=(0.5, 1.0), progress=progress)
    volume.commit()
    best = result["best_genome"]
    row["search"] = result["summary"]
    row["best_genome"] = best.to_dict()

    # Fixed recipes: pick their scale on split A, then test once.
    recipes = {}
    ta_a = {}
    for scale in TA_SCALES:
        weaver.apply(model, weave.task_arithmetic_genome(count, N_BUCKETS, scale))
        ta_a[scale] = objective_a(score_a(model))[0]
    weaver.restore(model)
    ta_scale = max(ta_a, key=lambda s: ta_a[s])
    recipes["task_arithmetic"] = {"genome": weave.task_arithmetic_genome(count, N_BUCKETS, ta_scale),
                                  "scale": ta_scale, "practice_a_by_scale": ta_a}
    recipes["uniform"] = {"genome": weave.uniform_genome(count, N_BUCKETS), "scale": None}
    ties_a = {}
    for scale in TIES_SCALES:
        weaver.apply_ties(model, keep=TIES_KEEP, scale=scale)
        ties_a[scale] = objective_a(score_a(model))[0]
        weaver.restore(model)
    ties_scale = max(ties_a, key=lambda s: ties_a[s])
    weaver.restore(model)

    arms = {}

    def test_arm(label, writer):
        writer()
        scores = score_on(finals, final_skills)
        value, detail = objective_f(scores)
        arms[label] = {"scores": scores, "objective": value, "detail": detail}
        weaver.restore(model)
        print(f"final {label}: {scores} objective {value:.4f}")

    test_arm("searched", lambda: weaver.apply(model, best))
    test_arm("uniform", lambda: weaver.apply(model, recipes["uniform"]["genome"]))
    test_arm("task_arithmetic", lambda: weaver.apply(model, recipes["task_arithmetic"]["genome"]))
    test_arm("ties", lambda: weaver.apply_ties(model, keep=TIES_KEEP, scale=ties_scale))
    control = weaver.shuffled(CONTROL_SEED)
    test_arm("random_control", lambda: control.apply(model, best))
    row["recipes"] = {"task_arithmetic_scale": ta_scale, "task_arithmetic_practice_a": ta_a,
                      "ties_keep": TIES_KEEP, "ties_scale": ties_scale, "ties_practice_a": ties_a}
    row["cast_survival_best"] = weaver.cast_survival(best)
    host_final = ref_f["host"]
    row["final"] = {"host": host_final, "members": {n: ref_f[n] for n in names}, "arms": arms}
    row["member_practice_minus_final"] = {
        n: {s: ref_a[n][s] - ref_f[n][s] for s in skills if s in ref_a[n] and s in ref_f[n]} for n in ["host"] + names}
    row["denom_clipped"] = {"practice_a": objective_a.clipped(), "practice_b": objective_b.clipped(),
                            "final": objective_f.clipped()}

    # Gap closed and the go rule.
    closed = {}
    beats_by = {}
    for skill in fam["targets"]:
        top = max(ref_f[n][skill] for n in names)
        gap = top - host_final[skill]
        gain = arms["searched"]["scores"][skill] - host_final[skill]
        beats_by[skill] = gain
        closed[skill] = None if gap <= 0.0 else gain / gap
    guard_ok = True
    for skill in fam["guards"]:
        if arms["searched"]["scores"][skill] < host_final[skill] - GUARD_POINTS:
            guard_ok = False
    fixed_best = max(arms["uniform"]["objective"], arms["task_arithmetic"]["objective"], arms["ties"]["objective"])
    go = {"beats_host_by_points": beats_by,
          "all_targets_over_2_points": all(v > GO_POINTS for v in beats_by.values()),
          "guard_within_1_point": guard_ok,
          "beats_fixed_recipes": arms["searched"]["objective"] > fixed_best,
          "control_does_not_beat_host": bool(
              all(arms["random_control"]["scores"][s] - host_final[s] <= GO_POINTS for s in fam["targets"])
              and arms["random_control"]["objective"] < arms["searched"]["objective"]),
          "gap_closed_share": closed}
    go["go"] = bool(go["all_targets_over_2_points"] and go["guard_within_1_point"] and go["beats_fixed_recipes"]
                    and go["control_does_not_beat_host"] and not fam.get("exploratory"))
    row["go_rule"] = go

    # Write the row first: a failed save must not lose the search.
    row["seconds"] = round(time.time() - started, 1)
    row["status"] = "done"
    full = dict(row)
    full["search_log"] = result["log"]
    result_path = f"/vol/results/round21/{name}.json"
    with open(result_path, "w") as handle:
        json.dump(full, handle, default=str)
    volume.commit()

    # Save the best model and check its size.
    out_dir = f"/vol/written/round21_{name}"
    try:
        weave.save(weaver, model, best, out_dir, tokenizer=tokenizer, host_id=fam["host"],
                   search_summary=result["summary"], extra={"final": row["final"], "go_rule": go})
        row["saved"] = out_dir
        row["params_saved"] = saved_param_count(out_dir)
        row["params_equal_host"] = row["params_saved"] == row["params_host"]
    except Exception as error:
        row["save_error"] = repr(error)[:600]
        print("save failed:", row["save_error"])
    full = dict(row)
    full["search_log"] = result["log"]
    with open(result_path, "w") as handle:
        json.dump(full, handle, default=str)
    volume.commit()
    row["search_log_summary"] = result["summary"]["per_generation"]
    return row


def saved_param_count(directory):
    """Count the elements in the saved safetensors files (tied tensors are stored once)."""
    from safetensors import safe_open

    total = 0
    for file in sorted(os.listdir(directory)):
        if not file.endswith(".safetensors"):
            continue
        with safe_open(os.path.join(directory, file), framework="pt") as handle:
            for key in handle.keys():
                shape = handle.get_slice(key).get_shape()
                count = 1
                for size in shape:
                    count *= size
                total += count
    return total


def layer_map_prior(model, tokenizer, items_a, members, ev, weave):
    """A starting genome from wsp.atlas.layer_skill_map: more weight where the member's skill lives."""
    from wsp import atlas

    def texts_of(skill):
        out = []
        for item in items_a[skill][:48]:
            if "question" in item:
                out.append(item["question"] + " " + item["answer"])
            elif "problem" in item:
                out.append(item["problem"])
            elif "text" in item:
                out.append(item["text"] + "\n" + "\n".join(item["test_list"]))
            else:
                out.append(item["prompt"])
        return out

    general = texts_of("mmlu")
    scores = {}
    for skill in sorted(set(m.skill for m in members)):
        if skill not in items_a:
            continue
        mapped = atlas.layer_skill_map(model, tokenizer, texts_of(skill), general)
        scores[skill] = list(mapped["scores"])
    return weave.layer_prior_genome(scores, [m.skill for m in members], N_BUCKETS)


# ---------------------------------------------------------------- Modal functions


@app.function(image=image, volumes={"/vol": volume}, gpu="H200", memory=262144, timeout=24 * 3600)
def run_small(name, budget_hours=10.0, ignore_gate=False, use_layer_map=False):
    return run_family(name, budget_hours, ignore_gate, use_layer_map)


@app.function(image=image, volumes={"/vol": volume}, gpu="H200", memory=589824, timeout=24 * 3600)
def run_big(name, budget_hours=20.0, ignore_gate=False, use_layer_map=False):
    return run_family(name, budget_hours, ignore_gate, use_layer_map)


@app.local_entrypoint()
def main(family: str = "all", shard: int = 0, nshards: int = 1, budget_hours: float = 0.0,
         plan_only: bool = False, ignore_gate: bool = False, use_layer_map: bool = False):
    names = list(FAMILIES.keys()) if family == "all" else family.split(",")
    mine = []
    for index, name in enumerate(names):
        if index % nshards == shard:
            mine.append(name)
    print(f"shard {shard}/{nshards}: {mine}")
    handles = []
    for name in mine:
        hours = budget_hours if budget_hours > 0 else (20.0 if name in BIG else 10.0)
        plan = make_plan(name, hours)
        print_plan(plan)
        if plan_only:
            continue
        runner = run_big if name in BIG else run_small
        handles.append((name, runner.spawn(name, hours, ignore_gate, use_layer_map)))
    rows = []
    for name, handle in handles:
        try:
            rows.append(handle.get())
        except Exception as error:
            print("cell failed:", name, repr(error)[:600])
            rows.append({"family": name, "status": "failed", "error": repr(error)[:600]})
    for row in rows:
        print(json.dumps({k: row.get(k) for k in ["family", "status", "members_used", "go_rule", "seconds"]},
                         default=str)[:1500])
    if not plan_only:
        os.makedirs("results", exist_ok=True)
        with open(f"results/round21_shard{shard}.json", "w") as handle:
            json.dump(rows, handle, indent=2, default=str)
