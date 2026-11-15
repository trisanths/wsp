# Three business requests on one method (round 28)

Base model: Qwen/Qwen3-30B-A3B-Instruct-2507 (30.53B total, 3.04B run per token, 128 experts per layer, top 8).
Method: rank the experts of each layer on text of one skill, keep the top N, optionally heal
(`wsp/slim.py`). Driver: `modal_round28.py`. Every number is one run. The saliency ranking is deterministic
given its texts, so those cells have no seed. The random control is one draw (seed 0).

Files:

- `requests/clinic.json`, `requests/accounting.json`, `requests/law_firm.json`: Sapien requests. The business
  owner's wording is in the `notes` field.
- `plans/*.txt`: the output of `python -m wsp.cli plan demo/requests/<name>.json`. The planner already
  expressed the law skill (`wsp/auto.py` has a `law` pool), so `wsp/auto.py` did not change.
- `results_tables.md`: the tables below, produced by `scripts/round28_table.py` from the per-cell files.
- `sample_outputs.json`: greedy answers (220 new tokens) of each cell to two fixed prompts per request.
- Raw rows: `results/round28_rows.json`; per-cell records with per-item correctness:
  `results/round28_cells/round28/`.

## Data

Owner rule of 2026-11-15: ranking and heal may read benchmark train splits, scores come only from test items.

- Ranking text, 512 texts per request: 256 from the `wsp.healdata` pool of the skill and 256 from benchmark
  train splits (`wsp/benchtrain.py`): MedMCQA train (clinic), GSM8K train and MATH train (accounting), CaseHOLD
  train (law firm). Texts holding a scoring question are dropped with `wsp.slim.drop_overlap` (0 to 2 per source).
- Heal text, 4000 texts: 2000 more pool texts and 2000 more train-split texts. The heal loss is the KL to the
  full model's logits only (`lm_weight` 0). Rounds 19 and 19c also used an LM loss on the pool text, so this
  heal is not the same recipe. The train-split texts hold gold answers in a foreign style, which the lessons
  forbid fitting with an LM loss.
- Scores: MedMCQA validation 1000; MMLU test items of the request's subjects (medical: clinical_knowledge,
  college_medicine, professional_medicine, anatomy, medical_genetics = 945 items; accounting:
  professional_accounting, high_school_statistics = 498; law: professional_law, jurisprudence,
  international_law = 1763); GSM8K the seeded 500; MATH-500. "General" is MMLU test shuffled with seed 0, minus
  the request's subjects, first 1000. It shares no item with the domain set (tests in
  `tests/test_skills_eval_groups.py`).

## Results

All scores are percent. Disk is the size of the saved bf16 weights.

### clinic

"I run a clinic. I want a private assistant that answers medical questions as well as Qwen3-30B, on a smaller GPU we can keep on site."

| cell | total params (B) | MedMCQA val 1000 | MMLU medical test (945) | MMLU general 1000 (no medical) | disk (GB) |
|---|---|---|---|---|---|
| full model (`full_30b_mcq`) | 30.53 | 68.0 | 86.9 | 77.2 | 61.1 (2 bytes per parameter, not saved) |
| keep 64 by saliency, no heal | 16.03 | 65.9 | 84.3 | 65.0 | 32.1 |
| keep 48 by saliency, no heal | 12.40 | 64.5 | 81.3 | 56.8 | 24.8 |
| keep 32 by saliency, 500-step heal | 8.78 | 64.0 | 79.0 | 52.1 | 17.6 |
| keep 48 random (seed 0), no heal | 12.40 | 31.1 | 21.7 | 23.8 | 24.8 |

Keeping half the experts (16.03B total, 52% of the size) gives up 2.1 points of MedMCQA and 2.6 of MMLU
medical, and 12.2 points of general MMLU. GPU memory use was not measured in this round.

### accounting

"We do bookkeeping and tax prep. I need a model that's great at word-problem arithmetic and finance math, and cheap to run."

| cell | total params (B) | GSM8K test 500 | MATH-500 | MMLU accounting + statistics (498) | MMLU general 1000 (no acct/stats) | disk (GB) |
|---|---|---|---|---|---|---|
| full model | 30.53 | not measured (see below) | not measured | 73.5 (`full_30b_mcq`) | 78.0 (`full_30b_mcq`) | 61.1 |
| keep 64 by saliency, no heal | 16.03 | 91.4 | 42.4 | 64.1 | 53.6 | 32.1 |
| keep 48 by saliency, no heal | 12.40 | 90.6 | 48.2 | 56.2 | 43.6 | 24.8 |
| keep 32 by saliency, 500-step heal | 8.78 | not measured (see below) | | | | |
| keep 48 random (seed 0), no heal | 12.40 | 0.0 | 0.0 | 20.1 | 23.8 | 24.8 |

Reference for the missing full-model GSM8K: results/README.md run 19 scored the full model 91.5 on the first 200
of the same seeded 500 items. The keep 64 and keep 48 cells score 90.5 and 90.0 on those same 200 (from their
per-item records). No full-model MATH-500 exists for this model in the repo. Keep 64 scores 5.8 points lower on
MATH-500 than keep 48, which is the opposite of the size order. The standard error of one MATH-500 score is
about 2.2 points, so this is one seed of a noisy number and not a finding.

### law firm

"We're a small law firm. We want a model that knows law well for first-pass research and runs on our own hardware."

| cell | total params (B) | MMLU law test (1763) | MMLU general 1000 (no law) | disk (GB) |
|---|---|---|---|---|
| full model (`full_30b_mcq`) | 30.53 | 62.7 | 80.3 | 61.1 |
| keep 64 by saliency, no heal | 16.03 | 61.8 | 60.3 | 32.1 |
| keep 48 by saliency, no heal | 12.40 | 60.4 | 55.7 | 24.8 |
| keep 32 by saliency, 500-step heal | 8.78 | 50.5 | 46.0 | 17.6 |
| keep 48 random (seed 0), no heal | 12.40 | 24.4 | 24.4 | 24.8 |

## Reading the tables

- The ranking matters. The random control at the same size collapses on every request: 31 on MedMCQA, 24 on
  law, 0 on GSM8K and MATH-500 (the raw outputs are runs of newline characters). Chance on four options is 25.
- The domain score holds up while general knowledge goes. At keep 64 the domain scores are 0.9 to 2.6 points
  below the full model for medicine and law, and general MMLU falls 12.2 (clinic) and 20.0 (law) points.
- Keep 32 with the KL heal scored below keep 48 with no heal on the two requests where both were measured
  (clinic: 79.0 against 81.3 on MMLU medical; law: 50.5 against 60.4). The heal ran 500 steps on one seed, so
  this does not say what a longer heal does.
- Accounting: GSM8K stays at 90.6 to 91.4 at both sizes, near the round 19 full-model figure. MMLU accounting
  drops to 64.1 and 56.2 against 73.5 for the full model. MATH-500 has no full-model number here.
- Decode speed in these rows (5 to 9 tokens per second) comes from the Hugging Face eager MoE loop and says
  nothing about serving. `modal_serve.py` serves the kept models with vLLM.

## What is missing

The workspace trisanthsrini hit its Modal spend limit ("Workspace ac-70SRNVzbfMBWmYcg3spTgp has exceeded its
spend limit") at about 06:30 PST on 2026-11-15 and every container stopped. Two cells were running and lost:

- `full_30b`: the full model on GSM8K (500), MATH-500 (500), and the rest. The multiple-choice part exists as
  `full_30b_mcq`, so the clinic and law rows are complete. The accounting row lacks its GSM8K and MATH-500 baseline.
- `accounting_sal32_heal500`: the 500-step heal finished (KL 0.393 to 0.215, 1926 s), then the model was
  scoring GSM8K and MATH-500 when the containers stopped.

Each of these takes about 4 hours on one H200 because the HF eager loop generates GSM8K and MATH-500 slowly
(GSM8K about 1 hour and MATH-500 about 2.5 hours per cell). To finish after the limit is raised:

    MODAL_PROFILE=trisanthsrini modal run --detach modal_round28.py --spawn --split \
        --only full_30b,accounting_sal32_heal500 --tag finish

Then `modal volume get wsp-models /results/round28/ results/round28_cells` and rerun
`scripts/round28_table.py`.

## Kept models

Keep 64 by saliency, no heal, saved with `wsp.slim.save_slim` (plain Hugging Face directories) on the
`wsp-models` volume of workspace trisanthsrini:

- `/vol/slim_models/round28/clinic_64`
- `/vol/slim_models/round28/accounting_64`
- `/vol/slim_models/round28/law_firm_64`

`modal_serve.py` reads its slim cells from the volume of the workspace it runs in, so serving these needs
that workspace (or a copy of the directories).
