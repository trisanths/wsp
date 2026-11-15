"""Round 21: same-family composition with a search (wsp/weave.py).

One network of exactly the host's size. For each family, the host and its
same-lineage specialists give per-tensor differences. A diagonal CMA-ES
searches one mixing weight per (specialist, layer bucket, attention or MLP)
plus one density per specialist, and the best genome is written as a plain
Hugging Face model. Nothing runs beside the host.

Rule: no benchmark data trains anything, and under the default objective no
benchmark selects anything either. Benchmarks are scored once, at the end.

Search objective (--objective):
  probe     default. Benchmark-free and forward-only (wsp.weave.ProbeObjective).
            Each member is the teacher on held-out text of its own skill pool
            from wsp.healdata. Before the search, every member and the host
            run the probe texts once, teacher-forced, and the top-64 token ids
            with log-probs and the tail mass are cached on the volume (keyed by
            repo revision and a hash of the texts). A candidate mix is scored
            by forward passes against those caches: transfer_i is the gap
            closed between KL(member_i || host) and 0 on member i's pool, drift
            is KL(host || candidate) on general text, and
              objective = mean transfer - 10 * max(0, drift - 0.05 nats).
            No generation and no benchmark item is read.
  practice  the old objective: generated answers on practice splits of the
            same benchmarks the round reports (GSM8K train and so on). The
            members trained on much of that data, so the row records
            practice_contaminated. Kept for comparison. All its benchmark
            loads live in practice_items_for.

Probe texts. Per pool, healdata.pool(name, 288, seed=0, skip=2000) gives the
rows 2000 to 2287 of the seeded pool order. Split a (rows 2000 to 2095)
drives the search, split b (2096 to 2191) validates each generation, split c
(2192 to 2287) is never used to select and gives the numbers for the probe
go rule and content_z. A member with two pools uses 48 texts of each per
split. Default 96 texts per split, at most 384 tokens. A short pool falls back
to its last rows (the row records the mode). Heal scripts that read the same
pool with seed 0 take rows below 2000.

Member to pool. math and math500 members use "math"; mbpp and humaneval use
"code"; medmcqa uses "medicine"; legal uses "law". A reasoning member
(marked "reasoning" below) uses the "reasoning" pool when healdata.SOURCES has
one, else "math" plus "code". Tool, RAG, extraction, and any skill without a
pool use "general", and the row says the pool is a weak proxy. Pools are
looked up by name in healdata.SOURCES; a missing pool falls back to
"general" and is marked weak.

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
            skills have no benchmark scorer here, so this cell is
            exploratory: target GSM8K, guard MMLU, no go rule.
  qwen3_8b  Qwen3-8B with II-Medical-8B and LegalEvolver-Qwen3-8B (legal has
            no benchmark scorer here; it enters as a member with no target
            skill, and the probe scores it on the "law" pool).

Small and new cells (2026-11-15 scout; evidence per member in docs/weave.md).
Every member passed `python -m wsp.lineage relation HOST MEMBER` as fine_tune or
fine_tune_of_ancestor with mean cosine 0.97 or more, the host's rope_theta, and
relative L2 0.01 or more. A member marked "unscored" has a skill with no
scorer and no probe pool of its own (it gets general text, a weak proxy); a
cell marked exploratory never gets a go. A member with a "ref" is a true
fine-tune of that reference, not of the host. A member marked "mlp_only" writes
MLP tensors only and is read without attention tensors, so the gate measures
the MLP alone.
  gemma2_2b    Gemma-2-2B-it with MergeBench math and coding fine-tunes.
  llama3b      Llama-3.2-3B-Instruct with MergeBench math and Hermes-3 (from
               the Llama-3.2-3B base).
  llama1b      Llama-3.2-1B-Instruct with a math and a Python coding fine-tune.
  qwen3_4b     Qwen3-4B with Pythagoras-Prover-4B and a medical SFT.
  qwen3_1p7b   Qwen3-1.7B with MedPsy-1.7B and hito-1.7b (reasoning).
  qwen25_3b    Qwen2.5-3B-Instruct with SmallThinker-3B and a medical and math
               fine-tune.
  smollm3      SmolLM3-3B with a medical SFT and Open-MOPD math (parent: the
               SmolLM3-3B base).
  gemma3_4b    Gemma-3-4B-it with biomed, text-to-Cypher, and MamayLM.
  gemma3_1b, qwen25_15b, qwen25_05b, lfm2_350m, lfm2_700m, lfm2_2b
               Exploratory cells: language, tool, and extraction members with
               no scorer here.
  gemma27_mlp  The gemma27 cell with MedGemma forced to MLP only (its
               attention fails the gate, its MLP passes).
`--family all` runs every cell above. Name the cells you want.

Benchmark targets are the members' skills: math (GSM8K, MATH-500), code (MBPP;
HumanEval is scored on the final test only, as transfer), medicine
(MedMCQA). The guard skill is MMLU. Final test sets match round 16b: GSM8K 300
seeded, MBPP sanitized 200 seeded, MMLU dev 1000, MedMCQA 1000; MATH-500 uses
a seeded 250 and HumanEval all 164. Under --objective practice, the practice
sets are two disjoint splits of about 150 items per skill: GSM8K train, MATH
train (DigitalLearningGmbH/MATH-lighteval), MBPP ids 511 to 974, MedMCQA
train, MMLU validation; the members trained on much of it, so the row records
member_practice_minus_final per skill and practice_contaminated.

Scoring uses wsp.skills_eval with the host's tokenizer and chat template for
every arm, including the members (a member is scored in the host's slot:
its attention, MLP, and norm tensors are written into the host; embeddings
and the output head stay the host's). The prompts are identical for every arm.

Arms on the final test, scored once at the end: host, each member, the
searched genome, uniform average, task arithmetic (scale picked on search
split A from 0.25, 0.5, 1.0), corrected TIES (keep 0.5, scale picked on A from
0.5, 1.0), and one random control (the best genome on differences shuffled
inside each tensor). The fixed recipes and the random controls are also
scored with the probe objective on splits a, b, and c. The row adds
content_z of the best mix against 4 random controls (seeds differ), probe
only.

Go rule, at exactly the host's size, in two parts. Both must hold.
  probe       the best mix's transfer is above 0.3 on every member's pool
              (split c), its drift is under the tolerance (0.05 nats), and
              content_z against the random controls is above 2.
  benchmarks  the searched mix beats the host on every target skill by more
              than 2 points with the MMLU guard within 1 point, its final-test
              objective beats uniform, task arithmetic, and TIES, and the
              random control stays within 2 points of the host on every target
              skill and scores below the searched mix.
Under --objective practice only the benchmark part is judged. Rows also give,
per target skill, the share of the host-to-best-member gap the mix closes.

    modal run modal_round21.py --family qwen7b
    modal run modal_round21.py --family all --shard 0 --nshards 3
    modal run modal_round21.py --family qwen32b --plan-only
    modal run modal_round21.py --family qwen7b --objective practice

Writes results/round21_shard<i>.json (rows), the full search log to
/vol/results/round21/<family>.json, the teacher caches to
/vol/probe_cache/round21, and the best model to /vol/written/round21_<family>
(weave.json beside the weights).
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
        "safetensors", "zstandard", "numpy", "datasets", "sentencepiece", "protobuf", "tiktoken",
    )
    .env({"HF_HOME": "/vol/hf", "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
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

OBJECTIVES = ("probe", "practice")
PROBE_TEXTS = 96
PROBE_LEN = 384
PROBE_TOP_K = 64
PROBE_BATCH = 4
PROBE_SEED = 0
PROBE_SKIP = 2000
PROBE_SPLITS = ("a", "b", "c")
DRIFT_TOLERANCE = 0.05
DRIFT_WEIGHT = 10.0
TRANSFER_MIN = 0.3
Z_MIN = 2.0
N_CONTROLS = 4
PROBE_POPULATION = 24
PROBE_GENERATIONS = 40
PROBE_BUDGET_SHARE = 0.85
AVG_PROBE_TOKENS = 300
FORWARD_FLOPS = 2.5e14
FORWARD_BATCH_SECONDS = 0.05
DISK_GBPS = 2.5
CACHE_DIR = "/vol/probe_cache/round21"
POOL_BY_SKILL = {"gsm8k": "math", "math500": "math", "math": "math", "mbpp": "code", "humaneval": "code",
                 "code": "code", "medmcqa": "medicine", "medicine": "medicine", "legal": "law"}

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
            {"name": "openthinker", "id": "open-thoughts/OpenThinker-7B", "skill": "math500", "reasoning": True, "ref": None},
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
            {"name": "r1_llama", "id": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B", "skill": "math500", "reasoning": True,
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
            {"name": "qwq", "id": "Qwen/QwQ-32B", "skill": "gsm8k", "reasoning": True, "ref": "base32"},
            {"name": "baichuan_m2", "id": "baichuan-inc/Baichuan-M2-32B", "skill": "medmcqa", "ref": "base32"},
            {"name": "ocr_nemotron", "id": "nvidia/OpenCodeReasoning-Nemotron-32B", "skill": "mbpp", "ref": None},
            {"name": "openthinker", "id": "open-thoughts/OpenThinker-32B", "skill": "gsm8k", "reasoning": True, "ref": None},
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
    "gemma2_2b": {
        "host": "unsloth/gemma-2-2b-it", "params_b": 2.6, "targets": ["gsm8k", "mbpp"], "guards": ["mmlu"],
        "transfer": ["humaneval"], "refs": {},
        "members": [
            {"name": "math", "id": "MergeBench/gemma-2-2b-it_math", "skill": "gsm8k", "ref": None},
            {"name": "coding", "id": "MergeBench/gemma-2-2b-it_coding", "skill": "mbpp", "ref": None},
        ],
        "memory_gib": 256, "gpu": "H200",
    },
    "llama3b": {
        "host": "unsloth/Llama-3.2-3B-Instruct", "params_b": 3.2, "targets": ["gsm8k"], "guards": ["mmlu"],
        "transfer": [], "refs": {"base3b": "unsloth/Llama-3.2-3B"},
        "members": [
            {"name": "math", "id": "MergeBench/Llama-3.2-3B-Instruct_math", "skill": "gsm8k", "ref": None},
            {"name": "hermes3", "id": "NousResearch/Hermes-3-Llama-3.2-3B", "skill": "chat", "unscored": True, "ref": "base3b"},
        ],
        "memory_gib": 256, "gpu": "H200",
    },
    "llama1b": {
        "host": "unsloth/Llama-3.2-1B-Instruct", "params_b": 1.2, "targets": ["gsm8k", "mbpp"], "guards": ["mmlu"],
        "transfer": ["humaneval"], "refs": {},
        "members": [
            {"name": "math", "id": "AryanK123/Llama-3.2-1B-Instruct_SFT_Math-220kv00.04", "skill": "gsm8k", "ref": None},
            {"name": "coder", "id": "MDDDDR/Llama-3.2-1B-Instruct-FFT-coder-python", "skill": "mbpp", "ref": None},
        ],
        "memory_gib": 256, "gpu": "H200",
    },
    "qwen3_4b": {
        "host": "Qwen/Qwen3-4B", "params_b": 4.0, "targets": ["medmcqa"], "guards": ["mmlu"],
        "transfer": ["gsm8k"], "refs": {},
        "members": [
            {"name": "pythagoras", "id": "Pythagoras-LM/Pythagoras-Prover-4B", "skill": "math", "ref": None},
            {"name": "popit_medical", "id": "Hoshino-Yumetsuki/PopiT-Qwen3-4B-Medical-SFT-1128", "skill": "medmcqa", "ref": None},
        ],
        "memory_gib": 256, "gpu": "H200",
    },
    "qwen3_1p7b": {
        "host": "Qwen/Qwen3-1.7B", "params_b": 1.7, "targets": ["medmcqa"], "guards": ["mmlu"],
        "transfer": [], "refs": {},
        "members": [
            {"name": "medpsy", "id": "qvac/MedPsy-1.7B", "skill": "medmcqa", "ref": None},
            {"name": "hito", "id": "hitonet/hito-1.7b", "skill": "reasoning", "reasoning": True, "unscored": True, "ref": None},
        ],
        "memory_gib": 256, "gpu": "H200",
    },
    "qwen25_3b": {
        "host": "Qwen/Qwen2.5-3B-Instruct", "params_b": 3.1, "targets": ["gsm8k", "medmcqa"], "guards": ["mmlu"],
        "transfer": [], "refs": {},
        "members": [
            {"name": "smallthinker", "id": "Tiiny/SmallThinker-3B-Preview", "skill": "gsm8k", "reasoning": True, "ref": None},
            {"name": "medmath", "id": "calebking/qwen2.5-3b-instruct-medical-math-full", "skill": "medmcqa", "ref": None},
        ],
        "memory_gib": 256, "gpu": "H200",
    },
    "smollm3": {
        "host": "HuggingFaceTB/SmolLM3-3B", "params_b": 3.1, "targets": ["gsm8k", "medmcqa"], "guards": ["mmlu"],
        "transfer": [], "refs": {"base": "HuggingFaceTB/SmolLM3-3B-Base"},
        "members": [
            {"name": "medical", "id": "vinayp27/smollm3-medical", "skill": "medmcqa", "ref": None},
            {"name": "mopd_math", "id": "BytedTsinghua-SIA/Open-MOPD-SmolLM3-3B-RL-Math", "skill": "gsm8k", "ref": "base"},
        ],
        "memory_gib": 256, "gpu": "H200",
    },
    "gemma3_4b": {
        "host": "unsloth/gemma-3-4b-it", "params_b": 4.3, "targets": ["medmcqa"], "guards": ["mmlu"],
        "transfer": [], "refs": {},
        "members": [
            {"name": "biomed", "id": "AdaptLLM/biomed-gemma-3-4b-it", "skill": "medmcqa", "ref": None},
            {"name": "cypher", "id": "neo4j/text-to-cypher-Gemma-3-4B-Instruct-2025.04.0", "skill": "cypher", "unscored": True, "ref": None},
            {"name": "mamaylm", "id": "INSAIT-Institute/MamayLM-Gemma-3-4B-IT-v1.0", "skill": "ukrainian", "unscored": True, "ref": None},
        ],
        "memory_gib": 256, "gpu": "H200",
    },
    "gemma3_1b": {
        "host": "unsloth/gemma-3-1b-it", "params_b": 1.0, "targets": ["gsm8k"], "guards": ["mmlu"],
        "transfer": [], "refs": {}, "exploratory": True,
        "members": [
            {"name": "borealis", "id": "NbAiLab/borealis-1b", "skill": "norwegian", "ref": None},
            {"name": "swahili", "id": "CraneAILabs/swahili-gemma-1b", "skill": "swahili", "ref": None},
        ],
        "memory_gib": 256, "gpu": "H200",
    },
    "qwen25_15b": {
        "host": "Qwen/Qwen2.5-1.5B-Instruct", "params_b": 1.5, "targets": ["gsm8k"], "guards": ["mmlu"],
        "transfer": [], "refs": {}, "exploratory": True,
        "members": [
            {"name": "vikhr", "id": "Vikhrmodels/Vikhr-Qwen-2.5-1.5B-Instruct", "skill": "russian", "ref": None},
            {"name": "tooltuned", "id": "asheeee24/tooltuned-qwen2.5-1.5b", "skill": "tool", "ref": None},
            {"name": "correction", "id": "shibing624/chinese-text-correction-1.5b", "skill": "correction", "ref": None},
        ],
        "memory_gib": 256, "gpu": "H200",
    },
    "qwen25_05b": {
        "host": "Qwen/Qwen2.5-0.5B-Instruct", "params_b": 0.5, "targets": ["gsm8k"], "guards": ["mmlu"],
        "transfer": [], "refs": {}, "exploratory": True,
        "members": [
            {"name": "hammer", "id": "MadeAgents/Hammer2.0-0.5b", "skill": "tool", "ref": None},
            {"name": "vikhr", "id": "Vikhrmodels/Vikhr-Qwen-2.5-0.5b-Instruct", "skill": "russian", "ref": None},
        ],
        "memory_gib": 256, "gpu": "H200",
    },
    "lfm2_350m": {
        "host": "LiquidAI/LFM2-350M", "params_b": 0.35, "targets": ["gsm8k"], "guards": ["mmlu"],
        "transfer": [], "refs": {}, "exploratory": True,
        "members": [
            {"name": "extract", "id": "LiquidAI/LFM2-350M-Extract", "skill": "extract", "ref": None},
            {"name": "luth", "id": "kurakurai/Luth-LFM2-350M", "skill": "french", "ref": None},
            {"name": "hala", "id": "hammh0a/Hala-350M", "skill": "arabic", "ref": None},
        ],
        "memory_gib": 256, "gpu": "H200",
    },
    "lfm2_700m": {
        "host": "LiquidAI/LFM2-700M", "params_b": 0.74, "targets": ["gsm8k"], "guards": ["mmlu"],
        "transfer": [], "refs": {}, "exploratory": True,
        "members": [
            {"name": "luth", "id": "kurakurai/Luth-LFM2-700M", "skill": "french", "ref": None},
            {"name": "tashkeel", "id": "Etherll/Tashkeel-700M", "skill": "arabic_diacritics", "ref": None},
            {"name": "hala", "id": "hammh0a/Hala-700M", "skill": "arabic", "ref": None},
        ],
        "memory_gib": 256, "gpu": "H200",
    },
    "lfm2_2b": {
        "host": "LiquidAI/LFM2-2.6B", "params_b": 2.6, "targets": ["gsm8k"], "guards": ["mmlu"],
        "transfer": [], "refs": {}, "exploratory": True,
        "members": [
            {"name": "transcript", "id": "LiquidAI/LFM2-2.6B-Transcript", "skill": "transcript", "ref": None},
            {"name": "longevity", "id": "LiquidAI/LFM2-2.6B-Longevity", "skill": "longevity", "ref": None},
        ],
        "memory_gib": 256, "gpu": "H200",
    },
    "gemma27_mlp": {
        "host": "unsloth/gemma-3-27b-it", "params_b": 27.4, "targets": ["medmcqa"], "guards": ["mmlu"],
        "transfer": [], "refs": {},
        "members": [
            {"name": "medgemma", "id": "unsloth/medgemma-27b-text-it", "skill": "medmcqa", "mlp_only": True, "ref": None},
            {"name": "meditron", "id": "EPFLiGHT/Gemma-3-27B-MeditronFO", "skill": "medmcqa", "ref": None},
        ],
        "memory_gib": 576, "gpu": "H200",
    },
}
BIG = ["gemma27", "gemma27_mlp", "qwen32b"]


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


def make_plan(name, budget_hours=10.0, objective="probe", population=None, generations=None, items=PRACTICE_ITEMS):
    """Return the plan for one family under one objective."""
    if objective == "probe":
        return make_probe_plan(name, budget_hours, population, generations)
    if objective != "practice":
        raise ValueError("objective must be probe or practice")
    plan = make_practice_plan(name, budget_hours, population or POPULATION, generations or GENERATIONS, items)
    plan["objective"] = "practice"
    return plan


def forward_seconds(params_b, texts):
    """Rough seconds to run texts through a model once (teacher forced, no generation). A guess, not measured."""
    tokens = texts * AVG_PROBE_TOKENS
    return 2.0 * params_b * 1e9 * tokens / FORWARD_FLOPS + FORWARD_BATCH_SECONDS * math.ceil(texts / PROBE_BATCH)


def make_probe_plan(name, budget_hours=10.0, population=None, generations=None):
    """Plan for the forward-only objective.

    A candidate costs one apply plus one forward pass of the probe texts of
    every member and of the general text (split a). The search can run many
    more candidates than the generation objective. Population and
    generations start at PROBE_POPULATION and PROBE_GENERATIONS and shrink
    (generations to 8, then population to 12) until the plan uses at most
    PROBE_BUDGET_SHARE of the budget. Explicit population or generations
    arguments set the start.
    """
    fam = FAMILIES[name]
    params_b = fam["params_b"]
    members = len(fam["members"])
    final_skills = fam["targets"] + fam["guards"] + fam["transfer"]
    budget = budget_hours * 3600.0
    apply_s = apply_seconds(fam)
    sets = members + 1
    n_splits = len(PROBE_SPLITS)

    def total(pop, gens):
        candidate = apply_s + forward_seconds(params_b, sets * PROBE_TEXTS)
        evals = (2 + members) + pop * gens
        search = evals * candidate + gens * candidate
        load = 600.0 + 900.0 * members * params_b / 8.0 / 4.0
        # Caches: the host runs every member pool and the general pool on 3 splits, each member runs its own pool.
        per_load = params_b * 2.0 / DISK_GBPS
        cache = (members + 1) * per_load + (members + 1) * n_splits * forward_seconds(params_b, PROBE_TEXTS) \
            + members * n_splits * forward_seconds(params_b, PROBE_TEXTS)
        reference = (1 + members) * (apply_s / 2.0 + n_splits * forward_seconds(params_b, sets * PROBE_TEXTS))
        recipes = (len(TA_SCALES) + len(TIES_SCALES)) * candidate + 4 * (candidate + 2 * forward_seconds(
            params_b, sets * PROBE_TEXTS))
        controls = N_CONTROLS * (2.0 * apply_s + n_splits * forward_seconds(params_b, sets * PROBE_TEXTS))
        final = (1 + members + 5) * final_seconds(params_b, final_skills)
        parts = {"load": load, "teacher_cache": cache, "reference": reference, "search": search,
                 "fixed_recipes": recipes, "random_controls": controls, "final": final,
                 "candidate_eval": candidate}
        return load + cache + reference + search + recipes + controls + final, parts, evals

    pop = population or PROBE_POPULATION
    gens = generations or PROBE_GENERATIONS
    seconds, parts, evals = total(pop, gens)
    while seconds > budget * PROBE_BUDGET_SHARE and gens > 8:
        gens -= 1
        seconds, parts, evals = total(pop, gens)
    while seconds > budget * PROBE_BUDGET_SHARE and pop > 12:
        pop -= 1
        seconds, parts, evals = total(pop, gens)
    return {"family": name, "objective": "probe", "population": pop, "generations": gens,
            "probe_texts": PROBE_TEXTS, "candidate_evals": evals, "expected_minutes": round(seconds / 60.0, 1),
            "fits_budget": seconds <= budget, "budget_hours": budget_hours,
            "minutes": {k: round(v / 60.0, 2) for k, v in parts.items()},
            "gpu_hours": round(seconds / 3600.0, 2), "memory_gib": fam["memory_gib"],
            "cpu_ram_gib_estimate": cpu_ram_gib(name)}


def make_practice_plan(name, budget_hours=10.0, population=POPULATION, generations=GENERATIONS, items=PRACTICE_ITEMS):
    """Return the practice-objective plan, shrinking generations, then items, then population until it fits the budget."""
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
    size = f"probe texts {plan['probe_texts']}" if plan.get("objective") == "probe" \
        else f"practice items {plan['practice_items']}"
    print(f"plan {plan['family']} ({plan.get('objective')}): population {plan['population']}, "
          f"generations {plan['generations']}, {size}, candidate evals {plan['candidate_evals']}")
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


# ---------------------------------------------------------------- probe objective (no benchmark data)


def pool_choice(spec, sources):
    """Map one member spec to probe pools. Returns {"pools": [...], "weak_proxy": bool, "note": str}.

    sources is healdata.SOURCES (pools are looked up by name). A reasoning
    member uses the "reasoning" pool when it exists, else "math" plus "code".
    A skill with no pool of its own uses "general" and is marked a weak proxy.
    """
    skill = spec["skill"]
    if spec.get("reasoning"):
        if "reasoning" in sources:
            return {"pools": ["reasoning"], "weak_proxy": False,
                    "note": "reasoning member on the healdata reasoning pool (multiple choice commonsense and "
                            "logic questions): a proxy for long chain of thought"}
        both = [pool for pool in ("math", "code") if pool in sources]
        if len(both) > 0:
            return {"pools": both, "weak_proxy": False, "note": "reasoning member on a math and code mix"}
    else:
        pool = POOL_BY_SKILL.get(skill)
        if pool is not None and pool in sources:
            return {"pools": [pool], "weak_proxy": False, "note": ""}
    return {"pools": ["general"], "weak_proxy": True,
            "note": f"skill {skill}: no skill pool, general text is a weak proxy"}


def dedupe(texts):
    seen = set()
    out = []
    for text in texts:
        if text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out


def load_pool_splits(tokenizer, pool_name):
    """Return ({split: texts}, range info) for one healdata pool: three disjoint held-out slices.

    The slice is rows PROBE_SKIP to PROBE_SKIP + 3 * PROBE_TEXTS of the seeded
    pool order. A pool too short for that uses its last rows instead.
    """
    from wsp import healdata

    want = PROBE_TEXTS * len(PROBE_SPLITS)
    mode = "skip"
    texts = dedupe(healdata.pool(pool_name, want, tokenizer=tokenizer, seed=PROBE_SEED, skip=PROBE_SKIP))
    if len(texts) < want:
        mode = "tail"
        everything = dedupe(healdata.pool(pool_name, PROBE_SKIP + want, tokenizer=tokenizer, seed=PROBE_SEED, skip=0))
        texts = everything[-want:]
    size = len(texts) // len(PROBE_SPLITS)
    if size < 8:
        raise ValueError(f"pool {pool_name}: only {len(texts)} probe texts")
    if size < PROBE_TEXTS:
        mode += "+short"
    splits = {}
    for index, split in enumerate(PROBE_SPLITS):
        splits[split] = texts[index * size:(index + 1) * size]
    info = {"pool": pool_name, "seed": PROBE_SEED, "mode": mode, "per_split": size, "max_length": PROBE_LEN,
            "rows": (f"{PROBE_SKIP} to {PROBE_SKIP + want - 1} of the seeded order" if mode == "skip"
                     else f"last {len(texts)} of the first {PROBE_SKIP + want}")}
    return splits, info


def probe_texts_for(tokenizer, fam):
    """Probe texts for one family. Returns {"splits": {split: {"general": [...], "members": {name: [...]}}},
    "choices": {member: pool choice}, "ranges": {pool: info}}. Texts of different splits never overlap."""
    from wsp import healdata

    sources = healdata.SOURCES
    choices = {}
    needed = set(["general"])
    for spec in fam["members"]:
        choices[spec["name"]] = pool_choice(spec, sources)
        needed.update(choices[spec["name"]]["pools"])
    loaded = {}
    ranges = {}
    for pool in sorted(needed):
        try:
            loaded[pool], ranges[pool] = load_pool_splits(tokenizer, pool)
        except ValueError as error:
            if pool == "general":
                raise
            loaded[pool] = None
            ranges[pool] = {"pool": pool, "error": str(error)[:300]}
    used = set(["general"])
    for spec in fam["members"]:
        choice = choices[spec["name"]]
        if any(loaded[pool] is None for pool in choice["pools"]):
            choice["note"] = (choice["note"] + " pool " + ",".join(choice["pools"])
                              + " unavailable, general text is a weak proxy").strip()
            choice["pools"] = ["general"]
            choice["weak_proxy"] = True
        used.update(choice["pools"])
    healdata.check_sources([source[0] for pool in sorted(used) for source in healdata._usable(pool)[0]])
    splits = {}
    for split in PROBE_SPLITS:
        entry = {"general": loaded["general"][split], "members": {}}
        for spec in fam["members"]:
            pools = choices[spec["name"]]["pools"]
            per = max(1, len(loaded[pools[0]][split]) // len(pools))
            texts = []
            for pool in pools:
                texts += loaded[pool][split][:per]
            entry["members"][spec["name"]] = texts
        splits[split] = entry
    for pool in loaded:
        if loaded[pool] is None:
            continue
        seen = set()
        for split in PROBE_SPLITS:
            for text in loaded[pool][split]:
                assert text not in seen, f"pool {pool}: a probe text sits in two splits"
                seen.add(text)
    return {"splits": splits, "choices": choices, "ranges": ranges}


def fill_caches(repo, label, revision, wanted, tokenizer, vocab):
    """Return {key: teacher cache} for wanted = {key: texts}. Loads the model only when a cache file is missing.

    The model is freed before the function returns. Files live in CACHE_DIR,
    named by label, repo revision, and a hash of the texts.
    """
    import torch
    import transformers

    from wsp import weave

    paths = {}
    for key, texts in wanted.items():
        paths[key] = weave.cache_path(CACHE_DIR, label, revision, weave.text_hash(texts, PROBE_LEN, PROBE_TOP_K))
    def load():
        return transformers.AutoModelForCausalLM.from_pretrained(download(repo), dtype=torch.bfloat16,
                                                                 device_map="cuda")

    got = weave.fill_caches(load, wanted, paths, tokenizer, vocab=vocab, top_k=PROBE_TOP_K, max_length=PROBE_LEN,
                            batch_size=PROBE_BATCH, label=label)
    volume.commit()
    weave.memory_report(f"teacher cache {label}")
    return got


def build_probe_caches(fam, tokenizer, revisions, data):
    """Stage 1, before the host sits on the GPU: the host and every member run their probe texts once.

    Returns {"host": {split: {"general": cache, member name: cache}}, "members": {name: {split: cache}},
    "vocab": int}.
    """
    os.makedirs(CACHE_DIR, exist_ok=True)
    wanted = {}
    for split in PROBE_SPLITS:
        wanted[split + ":general"] = data["splits"][split]["general"]
        for spec in fam["members"]:
            wanted[split + ":" + spec["name"]] = data["splits"][split]["members"][spec["name"]]
    got = fill_caches(fam["host"], "host", revisions[fam["host"]], wanted, tokenizer, None)
    host = {}
    vocab = None
    for split in PROBE_SPLITS:
        host[split] = {"general": got[split + ":general"]}
        vocab = got[split + ":general"]["vocab"]
        for spec in fam["members"]:
            host[split][spec["name"]] = got[split + ":" + spec["name"]]
    members = {}
    for spec in fam["members"]:
        want = {}
        for split in PROBE_SPLITS:
            want[split] = data["splits"][split]["members"][spec["name"]]
        members[spec["name"]] = fill_caches(spec["id"], spec["name"], revisions[spec["id"]], want, tokenizer, vocab)
    return {"host": host, "members": members, "vocab": vocab}


def setup_probe_selection(weave, model, members, caches, row):
    """Build one ProbeObjective per split, calibrated on the host. Forward passes only; no benchmark is read."""
    po = {}
    baseline = {}
    for split in PROBE_SPLITS:
        member_caches = {}
        host_pool = {}
        for member in members:
            member_caches[member.name] = caches["members"][member.name][split]
            host_pool[member.name] = caches["host"][split][member.name]
        po[split] = weave.ProbeObjective([m.name for m in members], member_caches, caches["host"][split]["general"],
                                         DRIFT_TOLERANCE, DRIFT_WEIGHT, PROBE_BATCH, host_pool_caches=host_pool)
        baseline[split] = po[split].calibrate(model)
        weave.memory_report(f"calibrate split {split}")
    row["probe"]["baseline"] = baseline
    return {"score_a": po["a"].scores, "score_b": po["b"].scores, "objective_a": po["a"], "objective_b": po["b"],
            "init": None, "po": po, "denom_clipped": {}}


def arm_row(po, scores):
    value, detail = po(scores)
    return {"objective": value, "mean_transfer": po.mean_transfer(scores), "transfer": scores["transfer"],
            "drift": scores["drift"], "pool_drift": scores.get("pool_drift"), "member_kl": scores["member_kl"]}


def probe_arms(po, weaver, model, writers):
    """Score each writer's model with the probe objective on every split. writers: {label: function that writes the model}."""
    out = {}
    for label, writer in writers.items():
        writer()
        out[label] = {}
        for split in PROBE_SPLITS:
            out[label][split] = arm_row(po[split], po[split].scores(model))
        weaver.restore(model)
        print(f"probe {label}: mean transfer a {out[label]['a']['mean_transfer']} c {out[label]['c']['mean_transfer']}")
    return out


def probe_go(arms, names, po):
    """The probe part of the go rule, from split c (never used to select)."""
    from wsp import probe

    best = arms["searched"]["c"]
    controls = []
    by_member = {}
    for label in arms:
        if label.startswith("random_control_"):
            controls.append(label)
    stat = probe.content_z(best["mean_transfer"], [arms[c]["c"]["mean_transfer"] for c in controls])
    for name in names:
        by_member[name] = probe.content_z(best["transfer"][name], [arms[c]["c"]["transfer"][name] for c in controls])
    members_ok = {}
    for name in names:
        value = best["transfer"][name]
        members_ok[name] = bool(value is not None and value > TRANSFER_MIN)
    z = None if stat is None else stat["z"]
    guard_value = best.get("harm") if best.get("harm") is not None else best["drift"]
    drift_ok = bool(guard_value < po["c"].drift_tolerance)
    z_ok = bool(z is not None and z > Z_MIN)
    return {"transfer_over_0.3_on_every_member": members_ok, "drift": best["drift"], "harm": best.get("harm"), "drift_under_tolerance": drift_ok,
            "content_z": stat, "content_z_by_member": by_member, "content_z_over_2": z_ok, "controls": controls,
            "go": bool(all(members_ok.values()) and drift_ok and z_ok)}


# ---------------------------------------------------------------- practice objective (the old path)
# This path reads benchmark train splits through practice_items_for. It exists for comparison only.


def setup_practice_selection(ev, model, tokenizer, weaver, members, fam, plan, row, use_layer_map):
    """The old selection: generated answers on practice splits of the reported benchmarks."""
    from wsp import weave

    names = [m.name for m in members]
    skills = fam["targets"] + fam["guards"]
    row["practice_contaminated"] = True
    items_a, items_b = practice_items_for(ev, skills, plan["practice_items"])
    assert disjoint(items_a, items_b)

    def score_a(m):
        return ev.evaluate_suite(model, tokenizer, skills=skills, items=items_a, log=False)["scores"]

    def score_b(m):
        return ev.evaluate_suite(model, tokenizer, skills=skills, items=items_b, log=False)["scores"]

    ref_a = {"host": score_a(model)}
    ref_b = {"host": score_b(model)}
    for index, member in enumerate(members):
        weaver.load_member(model, index)
        ref_a[member.name] = score_a(model)
        ref_b[member.name] = score_b(model)
        weaver.restore(model)
    row["reference_scores"] = {"practice_a": ref_a, "practice_b": ref_b}
    objective_a = weave.Objective.from_reference(fam["targets"], fam["guards"], ref_a["host"], [ref_a[n] for n in names])
    objective_b = weave.Objective.from_reference(fam["targets"], fam["guards"], ref_b["host"], [ref_b[n] for n in names])
    init = None
    if use_layer_map:
        try:
            init = layer_map_prior(model, tokenizer, items_a, members, ev, weave)
            row["layer_map_init"] = True
        except Exception as error:
            print("layer map failed, uniform start:", repr(error)[:300])
            row["layer_map_init"] = False
    return {"score_a": score_a, "score_b": score_b, "objective_a": objective_a, "objective_b": objective_b,
            "init": init, "ref_a": ref_a, "ref_b": ref_b,
            "denom_clipped": {"practice_a": objective_a.clipped(), "practice_b": objective_b.clipped()}}


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


def run_family(name, budget_hours=10.0, ignore_gate=False, use_layer_map=False, objective="probe"):
    import torch
    import transformers

    from wsp import skills_eval as ev
    from wsp import weave

    if objective not in OBJECTIVES:
        raise ValueError("objective must be probe or practice")
    started = time.time()
    fam = FAMILIES[name]
    plan = make_plan(name, budget_hours, objective)
    print_plan(plan)
    row = {"family": name, "host": fam["host"], "objective": objective, "plan": plan,
           "exploratory": bool(fam.get("exploratory")), "targets": fam["targets"], "guards": fam["guards"],
           "transfer": fam["transfer"], "selects_on_benchmarks": objective == "practice"}
    host_dir = download(fam["host"])
    tokenizer = transformers.AutoTokenizer.from_pretrained(host_dir)
    assert tokenizer.chat_template, f"{fam['host']}: no chat template loaded; prompts would fall back to a plain format"
    row["chat_template_md5"] = hashlib.md5(tokenizer.chat_template.encode()).hexdigest()
    repos = [fam["host"]] + list(fam["refs"].values()) + [m["id"] for m in fam["members"]]
    row["revisions"] = {repo: revision_of(repo) for repo in repos}
    row["seeds"] = {"practice": PRACTICE_SEEDS, "search": SEARCH_SEED, "control": CONTROL_SEED,
                    "probe": PROBE_SEED, "probe_skip": PROBE_SKIP}

    # Probe objective: texts and teacher caches first, while the GPU holds no host copy.
    caches = None
    if objective == "probe":
        data = probe_texts_for(tokenizer, fam)
        row["probe"] = {"pools": data["choices"], "ranges": data["ranges"], "texts_per_split": PROBE_TEXTS,
                        "max_length": PROBE_LEN, "top_k": PROBE_TOP_K, "drift_tolerance": DRIFT_TOLERANCE,
                        "drift_weight": DRIFT_WEIGHT, "splits": list(PROBE_SPLITS)}
        caches = build_probe_caches(fam, tokenizer, row["revisions"], data)
        weave.memory_report("stage 1 teacher caches done")
        row["probe"]["cache_dir"] = CACHE_DIR
        row["practice_contaminated"] = False
        if use_layer_map:
            row["layer_map_init"] = "ignored under the probe objective"

    model = transformers.AutoModelForCausalLM.from_pretrained(host_dir, dtype=torch.bfloat16, device_map="cuda")
    model.eval()
    row["params_host"] = ev.model_params(model)[0]
    weave.memory_report("host loaded")
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
        forced_mlp = bool(spec.get("mlp_only"))
        mlp_only = forced_mlp or weave.positional_differs(host_config, config)
        ref = refs[spec["ref"]] if spec["ref"] is not None else None
        member = weave.member_from_directory(spec["name"], spec["skill"], directory, model, ref=ref,
                                             ref_name=spec["ref"] or "host", mlp_only=mlp_only,
                                             groups=("mlp", "norm") if forced_mlp else ("attn", "mlp", "norm"),
                                             meta={"id": spec["id"], "reference_repo": fam["refs"].get(spec["ref"])})
        gate = weave.gate_member(member, model, max_rel=GATE_MAX_REL, min_cos=GATE_MIN_COS)
        gate["mlp_only"] = mlp_only
        gate["tensors"] = len(member.tensors)
        gates[spec["name"]] = gate
        print(f"member {spec['name']}: gate {gate}")
        if gate["passed"] or ignore_gate:
            members.append(member)
    weave.memory_report("members read (CPU only, GPU unchanged)")
    row["gate"] = gates
    row["members_used"] = [m.name for m in members]
    if len(members) == 0:
        row["status"] = "no_admissible_member"
        row["seconds"] = round(time.time() - started, 1)
        return row

    weaver = weave.Weaver(model, members, n_buckets=N_BUCKETS, density_step=DENSITY_STEP)
    weave.memory_report("weaver built")
    count = len(members)
    names = [m.name for m in members]
    skills = fam["targets"] + fam["guards"]
    final_skills = skills + fam["transfer"]

    def score_on(items, wanted):
        return ev.evaluate_suite(model, tokenizer, skills=wanted, items=items, log=False)["scores"]

    # The search objective. Only the practice path reads benchmark train splits.
    if objective == "probe":
        selection = setup_probe_selection(weave, model, members, caches, row)
    else:
        selection = setup_practice_selection(ev, model, tokenizer, weaver, members, fam, plan, row, use_layer_map)
    score_a = selection["score_a"]
    objective_a = selection["objective_a"]

    os.makedirs("/vol/results/round21", exist_ok=True)
    progress_path = f"/vol/results/round21/{name}.progress.jsonl"
    last = {"generation": -1}

    def progress(entry):
        with open(progress_path, "a") as handle:
            handle.write(json.dumps(entry, default=str) + "\n")
        if entry["generation"] != last["generation"]:
            last["generation"] = entry["generation"]
            volume.commit()

    result = weave.search(weaver, model, score_a, selection["score_b"], objective_a, selection["objective_b"],
                          popsize=plan["population"], generations=plan["generations"], seed=SEARCH_SEED,
                          init=selection["init"], search_density=True, density_range=(0.5, 1.0), progress=progress)
    volume.commit()
    weave.memory_report("search")
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
    row["recipes"] = {"task_arithmetic_scale": ta_scale, "task_arithmetic_search_a": ta_a, "ties_keep": TIES_KEEP,
                      "ties_scale": ties_scale, "ties_search_a": ties_a, "scales_picked_on": objective + " split a"}

    writers = {
        "searched": lambda: weaver.apply(model, best),
        "uniform": lambda: weaver.apply(model, recipes["uniform"]["genome"]),
        "task_arithmetic": lambda: weaver.apply(model, recipes["task_arithmetic"]["genome"]),
        "ties": lambda: weaver.apply_ties(model, keep=TIES_KEEP, scale=ties_scale),
    }

    # Probe numbers for every arm and the random controls (forward passes only).
    if objective == "probe":
        probe_writers = dict(writers)
        for index in range(N_CONTROLS):
            clone = weaver.shuffled(CONTROL_SEED + index)
            probe_writers[f"random_control_{CONTROL_SEED + index}"] = (lambda c=clone: c.apply(model, best))
        arms_probe = probe_arms(selection["po"], weaver, model, probe_writers)
        in_slot = {}
        for index, member in enumerate(members):
            def write_member(i=index):
                weaver.load_member(model, i)
            in_slot.update(probe_arms(selection["po"], weaver, model, {"member:" + member.name: write_member}))
        host_arm = {}
        for split in PROBE_SPLITS:
            host_arm[split] = arm_row(selection["po"][split], selection["po"][split].scores(model))
        row["probe"]["arms"] = arms_probe
        row["probe"]["members_in_host_slot"] = in_slot
        row["probe"]["host"] = host_arm
        row["probe"]["go"] = probe_go(arms_probe, names, selection["po"])
        row["probe"]["content_z"] = row["probe"]["go"]["content_z"]
        row["probe"]["content_z_by_member"] = row["probe"]["go"]["content_z_by_member"]
        row["probe"]["control_seeds"] = [CONTROL_SEED + i for i in range(N_CONTROLS)]
        row["cast_survival_best"] = weaver.cast_survival(best)
        weave.memory_report("probe arms and controls")

    # Benchmarks, scored once: host, members, best mix, fixed recipes, one random control.
    finals = final_items_for(ev, final_skills)
    first = finals[fam["targets"][0]][0]
    if fam["targets"][0] in GENERATIVE:
        row["prompt_sample"] = ev.render_prompt(tokenizer, ev.user_text(fam["targets"][0], first))
    else:
        row["prompt_sample"] = first["prompt"]
    ref_f = {"host": score_on(finals, final_skills)}
    for index, member in enumerate(members):
        weaver.load_member(model, index)
        ref_f[member.name] = score_on(finals, final_skills)
        weaver.restore(model)
        print(f"member {member.name}: final {ref_f[member.name]}")
    objective_f = weave.Objective.from_reference(fam["targets"], fam["guards"], ref_f["host"], [ref_f[n] for n in names])

    arms = {}

    def test_arm(label, writer):
        writer()
        scores = score_on(finals, final_skills)
        value, detail = objective_f(scores)
        arms[label] = {"scores": scores, "objective": value, "detail": detail}
        weaver.restore(model)
        print(f"final {label}: {scores} objective {value:.4f}")

    for label in ["searched", "uniform", "task_arithmetic", "ties"]:
        test_arm(label, writers[label])
    control = weaver.shuffled(CONTROL_SEED)
    test_arm("random_control", lambda: control.apply(model, best))
    if objective == "practice":
        row["cast_survival_best"] = weaver.cast_survival(best)
    host_final = ref_f["host"]
    row["reference_scores"] = dict(row.get("reference_scores", {}))
    row["reference_scores"]["final"] = ref_f
    row["final"] = {"host": host_final, "members": {n: ref_f[n] for n in names}, "arms": arms}
    if objective == "practice":
        ref_a = selection["ref_a"]
        row["member_practice_minus_final"] = {
            n: {s: ref_a[n][s] - ref_f[n][s] for s in skills if s in ref_a[n] and s in ref_f[n]}
            for n in ["host"] + names}
    clipped = dict(selection["denom_clipped"])
    clipped["final"] = objective_f.clipped()
    row["denom_clipped"] = clipped

    # Gap closed and the go rule, in two parts.
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
    bench = {"beats_host_by_points": beats_by,
             "all_targets_over_2_points": all(v > GO_POINTS for v in beats_by.values()),
             "guard_within_1_point": guard_ok,
             "beats_fixed_recipes": arms["searched"]["objective"] > fixed_best,
             "control_does_not_beat_host": bool(
                 all(arms["random_control"]["scores"][s] - host_final[s] <= GO_POINTS for s in fam["targets"])
                 and arms["random_control"]["objective"] < arms["searched"]["objective"])}
    bench["go"] = bool(bench["all_targets_over_2_points"] and bench["guard_within_1_point"]
                       and bench["beats_fixed_recipes"] and bench["control_does_not_beat_host"])
    probe_part = row["probe"]["go"] if objective == "probe" else None
    go = {"objective": objective, "probe": probe_part, "benchmarks": bench, "gap_closed_share": closed}
    both = bench["go"] and (probe_part is None or probe_part["go"])
    go["go"] = bool(both and not fam.get("exploratory"))
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
def run_small(name, budget_hours=10.0, ignore_gate=False, use_layer_map=False, objective="probe"):
    return run_family(name, budget_hours, ignore_gate, use_layer_map, objective)


@app.function(image=image, volumes={"/vol": volume}, gpu="H200", memory=589824, timeout=24 * 3600)
def run_big(name, budget_hours=20.0, ignore_gate=False, use_layer_map=False, objective="probe"):
    return run_family(name, budget_hours, ignore_gate, use_layer_map, objective)


@app.local_entrypoint()
def main(family: str = "all", shard: int = 0, nshards: int = 1, budget_hours: float = 0.0,
         plan_only: bool = False, ignore_gate: bool = False, use_layer_map: bool = False,
         objective: str = "probe"):
    if objective not in OBJECTIVES:
        raise ValueError("--objective must be probe or practice")
    names = list(FAMILIES.keys()) if family == "all" else family.split(",")
    mine = []
    for index, name in enumerate(names):
        if index % nshards == shard:
            mine.append(name)
    print(f"shard {shard}/{nshards}: {mine}")
    handles = []
    for name in mine:
        hours = budget_hours if budget_hours > 0 else (20.0 if name in BIG else 10.0)
        plan = make_plan(name, hours, objective)
        print_plan(plan)
        if plan_only:
            continue
        runner = run_big if name in BIG else run_small
        handles.append((name, runner.spawn(name, hours, ignore_gate, use_layer_map, objective)))
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
