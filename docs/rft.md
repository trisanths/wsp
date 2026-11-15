# Verified self-training of a slimmed MoE on code

Date: 2026-11-15. Code: `wsp/rft.py` (CPU tests in `tests/test_rft.py`), `modal_rft.py` (GPU stages),
`scripts/rft_report.py` (table with paired intervals). Rows: `results/rft/`.

Question: can the keep-64 slim of Qwen3-30B-A3B-Instruct-2507 (16.0B stored, code saliency, Run 19) and the
keep-32 healed one (8.78B) get better at code by training on answers that pass unit tests?

## Result in one table

Greedy pass@1, vLLM 0.32, one H200, one seed per row. The first row of each block is the starting model. The
held-out slice is 545 train prompts that nothing trained on and that picked the round (it is not a benchmark).
Brackets are the 95 percent paired bootstrap interval of the change against the start row, in points.

| model | HumanEval 164 | MBPP test 257 | GSM8K 200 | MMLU 1000 | held-out train slice 545 |
|---|---|---|---|---|---|
| full 30B (reference) | 90.9 | 83.7 | 91.0 | 77.9 | 85.7 |
| keep-64 start | 88.4 | 82.9 | 89.5 | 44.9 | 82.2 |
| keep-64 RFT round 1 (selected) | 88.4 (+0.0 [-1.8,+1.8]) | 82.9 (+0.0 [-1.9,+1.9]) | 89.0 (-0.5) | 46.1 (+1.2) | 85.1 (+2.9) |
| keep-64 RFT round 2 | 89.6 (+1.2 [0.0,+3.0]) | 83.3 (+0.4 [-1.6,+2.3]) | 87.0 (-2.5) | 45.7 (+0.8) | 84.0 (+1.8) |
| keep-32 healed start (8.78B) | 82.9 | 73.2 | 49.5 | 24.7 | 70.6 |
| keep-32 RFT round 1 (selected) | 83.5 (+0.6 [-4.3,+5.5]) | 77.0 (+3.9 [0.0,+7.8]) | 49.0 (-0.5) | 24.6 (-0.1) | 78.9 (+8.3) |
| keep-32 RFT round 2 | 84.1 (+1.2 [-3.0,+5.5]) | 77.4 (+4.3 [+0.8,+8.2]) | 43.0 (-6.5) | 24.9 (+0.2) | 76.9 (+6.2) |

Sources: `results/rft/k64/score_start.json`, `score_full.json`, `r1/score.json`, `r2/score.json`;
`results/rft/k32/score_start.json`, `r1/score.json`, `r2/score.json`. Regenerate the table with
`python scripts/rft_report.py results/rft k64 k32`.

What this says, in plain terms:

- Keep-64 does not gain on the two benchmarks. It sits 2.5 points under the full model on HumanEval and 0.8 on
  MBPP, so there is little room. Round 2 moved HumanEval by 1.2 points (3 items of 164), inside the noise.
- Keep-32 gains on the training distribution (held-out slice +8.3, 70.6 to 78.9) and on MBPP test (+3.9 points,
  10 more of 257 items; the interval touches zero at its low end). HumanEval moved +0.6, inside the noise.
- Round 2 did not beat round 1 on the held-out slice for either model, so round 1 is the selected model in both
  and round 3 was not run. Round 2 of keep-32 has a better MBPP score (+4.3), but the rule is that test scores
  never pick a checkpoint, so round 1 stays selected.
- GSM8K and MMLU did not move in round 1. Keep-32 loses GSM8K in round 2 (-6.5), which shows that more rounds
  on code answers start to cost other skills.
- Engine matters. Under vLLM the keep-32 healed model scores 82.9 HumanEval and 73.2 MBPP. Run 19 measured the
  same weights with the Hugging Face loop at 73.8 and 61.9, and the keep-64 start at 87.8 and 79.0 against 88.4 and 82.9
  here. Compare rows only inside one engine.

## Speed of the selected models

Settings of `modal_serve.py` (64 code prompts, 256 new tokens, greedy, one H200, one run each). Source rows:
`results/rft/speed_k64_r1.json`, `results/rft/speed_k32_r1.json`. The full-model numbers are from
`outputs/serve_full.log` in the main checkout.

| model | weights GiB | decode tok/s batch 1 | decode tok/s batch 32 | TTFT median ms | prefill tok/s |
|---|---|---|---|---|---|
| full 30B | 56.9 (2 bytes per parameter) | 259 | 2486 | 59 | n/a |
| keep-64 RFT round 1 | 29.9 | 249.5 | 3040 | 51 | 12.5k |
| keep-32 healed RFT round 1 | 16.4 | 251.2 | 4109 | 49 | 13.1k |

Batch-1 decode is within 4 percent of the full model (the 3B active parameters per token are the same). Batch-32
throughput is 1.22 times (keep-64) and 1.65 times (keep-32) the full model, and weight memory is 0.52 and 0.29 times.

## Recipe

Each round starts from the starting model (not from the previous round) and trains on verified answers.

1. Data (`modal_rft.build_data`). Train prompts with tests, all in the format of the eval prompt:
   MBPP full (train, validation, prompt splits, task ids of the sanitized test removed), LeetCodeDataset train
   (check functions), APPS train problems with stdin tests (introductory and interview, question under 3500
   characters), and AceCode-87K (assert lists; a prompt needs one reference completion with pass rate 1). Every
   reference solution is run through the sandbox against its own tests, and a prompt whose reference fails is dropped.
2. Sampling (`modal_rft.sample_shard`). vLLM, k=8, temperature 0.8, top-p 0.95, 768 new tokens, prefix caching.
   Each answer is graded in the `wsp.skills_eval` sandbox. A cut-off answer scores 0. The full 30B model also
   samples once (k=4) as the teacher.
3. Selection (`rft.build_sft`). Up to 2 distinct passing answers per prompt from the student, plus up to 1 from the
   teacher (2 when the student solved nothing). Answers over 1.5 times the median passing length are skipped when
   shorter ones exist. The set is capped at 9000 rows, hardest prompts first (`rft.cap_rows`), to bound training time.
4. Training (`modal_rft.train_round`). LoRA rank 64 on q, k, v, o of all 48 layers, plus the routers (fp32,
   lr 1e-5). Loss on answer tokens only, LoRA lr 1e-4, cosine to 10 percent, 2 epochs, about 25k tokens per
   step, 330 steps, about 65 minutes on one H200 (keep-64) and 50 minutes (keep-32). The LoRA is merged and the model
   saved as a plain Hugging Face directory that vLLM loads.
   Why attention plus routers and not expert MLPs: transformers 4.57 runs each expert as its own linear in a Python
   loop, so a LoRA on 64 experts times 48 layers times 3 matrices adds 9216 wrapped modules to a step that already
   runs at about 2.4k tokens per second. Attention LoRA and routers needed no kernel work. Expert LoRA was not tried,
   so these results say nothing about it.
5. Scoring (`modal_rft.score_model`). HumanEval 164 (512 new tokens), MBPP sanitized test 257 (512), GSM8K first
   200 of the seeded 500 (512), MMLU 1000 by letter log-probability, and the held-out slice (768). All greedy
   under vLLM, with the prompts and graders of `wsp.skills_eval`. The round is chosen by the held-out slice only.
6. GRPO-style loop (`modal_rft.drive_pg`): see the section below.

## Data and decontamination

Counts are in `results/rft/data_stats.json`.

| source | candidates | dropped by 13-gram | reference passes | kept |
|---|---|---|---|---|
| MBPP train, validation, prompt | 474 | 13 | 460 of 461 | 460 |
| LeetCodeDataset train | 2641 | 110 | 2430 of 2531 | 2430 |
| APPS stdin | 518 | 0 | 456 of 518 | 456 |
| AceCode-87K (16000 candidates read) | 16000 | 76 | 15838 of 15924 | 8000 (cap) |

Result: 10801 train prompts and 545 held out (5 percent by a hash of the id). 10594 train prompts have samples
(duplicate ids collapse).

The test index holds the 13-grams of the HumanEval prompts, tests and canonical solutions and of the MBPP
sanitized test prompts, tests and reference code (34623 grams). An item is dropped when its text (prompt, tests,
header) shares a 13-gram with the index. Normalization: lowercase, word tokens (letters, digits, underscore).
A 13-gram counts only when it has at least 5 distinct tokens and one non-numeric token. Without that rule 1544
of 2641 LeetCode items were dropped, because test lists such as `0 0 0 ...` and `1 2 3 ...` match HumanEval tests.
After sampling, answers that share a 13-gram with the index are dropped too (33, 39 rows for keep-64 rounds 1
and 2; 52, 37 for keep-32). HumanEval has no train split. The MBPP test is the sanitized 257 of
`wsp.skills_eval`. Nothing in the reported scores trained, ranked, or selected on a test item. The held-out
slice is drawn from the same train distributions, so its gain says the model fits those distributions, not
that it generalizes to the benchmarks.

## Raw generations

`raw` in each score row keeps the first three outputs of HumanEval and MBPP. Sample pass rates on the train
prompts (`results/rft/*/summary.json`): the teacher passes 87.8 percent of its samples, the keep-64 start 85.5, and
the keep-32 start 72.7. The train-prompt pass rate of round 2 (89.0 percent, keep-64) is higher than round 1
because round 2 trained on those prompts; the held-out slice, which it did not train on, fell.

## GRPO-style loop

`drive_pg` samples k=8 at temperature 1.0 from the current model on 1500 new train prompts, gives each answer
the group advantage (r - mean) / std, keeps up to 600 prompts with mixed rewards, takes one pass of
weighted likelihood (LoRA r64, lr 2e-5, routers frozen, no KL, no ratio clipping), merges, and scores the
held-out slice. It is not strictly on-policy after the first optimizer step of an iteration. It is not trl.

Run once, from the keep-32 round 1 model, 3 iterations (`results/rft/k32_pg/summary.json`). Held-out train
slice after each iteration: 77.6, 79.4, 78.3, against 78.9 for the start of the loop (545 prompts, standard error
about 1.7 points). Best iteration is 3 items better than its start, which is noise, so GRPO is not kept and its
models were not scored on HumanEval or MBPP. It was not tried on keep-64, where RFT itself gave no benchmark gain.

## Limits

- One seed everywhere. Intervals above are over items, not over seeds.
- The held-out slice has 545 prompts (standard error about 1.5 points), so a 1 point move there is noise.
- Round 3 and expert-MLP LoRA were not run. Larger k, more prompts, and a harder prompt source (the keep-64
  model already solves 91 percent of train prompts at k=8) are the levers left.
- After the runs I changed `modal_rft.py` to score mixed item kinds through `rft.rewards_for`. The code path is the
  same as before for code items and the unit test passes, but that edit was not rerun on a GPU.

## Run it on another model or task

Another model: set `BASE` (tokenizer and teacher), add the start directory to `STARTS` in `modal_rft.py`, and copy
the model into the volume of the profile you run on (a Modal volume exists in one workspace only). The training
function needs a model that `wsp.slim.load_slim` can read; for a plain Hugging Face directory add a
`wsp_slim.json` with `{"format": "wsp-slim-v1", "plain_hf": true, "kind": "dense"}`. Then:

    modal run modal_rft.py --stage data
    modal run --detach modal_rft.py --stage drive --run <key> --rounds 2 --shards 3
    modal run --detach modal_rft.py --stage pg --run <key> --from-model <dir> --iterations 3

Another task: write `items_train.jsonl` and `items_heldout.jsonl` to `/vol/rft/rft1/data/` with one item per line.
Math with an exact-answer reward: `{"id", "source", "kind": "math", "user": <the eval prompt text>, "gold": "42"}`;
the reward is `rft.math_rewards` (last `\boxed{}` against the gold answer, `wsp.skills_eval.answers_match`).
Multiple choice with a letter reward: `{"kind": "mcq", "gold": "B", "user": <question with lettered choices and a
final-answer instruction>}`; the reward is `rft.mcq_rewards` ("answer is X" or a lone first letter). A new reward
is one function `rewards(items, outputs, workers=8) -> list of floats` registered with `rft.register_task` and a
kind in `rft.KIND_TASK`. Build the 13-gram index of your test items with `rft.ngram_index` and drop items with
`rft.decontaminate`. `rft.encode_sft`, `rft.build_sft`, `rft.cap_rows`, and `rft.pg_rows` need no changes.
`modal_rft.score_model` takes a `suites` argument (comma list of humaneval, mbpp, gsm8k, mmlu, heldout); for
another task call it with `heldout` and add the task's own test items beside it.
