# Write-in track: math (2026-11-15)

Code: `modal_wi_mathrl.py` (driver), `scripts/wi_mathrl_table.py` (table from the volume rows).
Rows: volume `wsp-models` of workspace ztrisanth, `wi_mathrl/v1/rows/<arm>_s<seed>[_kd].json`.

## Claim tested

A skill from another model can be written into one network through learned seams plus a short heal, and the
result beats every fine-tuning arm at the same data and steps. Here the skill is math. The base is
Qwen2.5-1.5B-Instruct. The donor is Qwen2.5-Math-7B-Instruct, a model far from the base in weight space
(cosine 0.2 to 0.5), so weight mixing does not apply.

## Design

- Graft: `wsp.pgraft.ParallelGraft`. Base residual after layer 14 goes through Enc (depth 2) into frozen donor
  layers 4..20, Dec (rank 256) reads the donor's contribution, a per-token gate writes it back. The graft arms
  also train a LoRA of rank 8 on the base, jointly. Glue is 13.2M parameters, graft total trainable 22.5M.
- Stage 1: Enc aligned on the donor's own next-token KL (400 steps, on the text of the donor solutions). Survival
  of the donor's next-token accuracy: 0.962 (seed 0), 0.968 (seed 1). Same tokenizer family, so it is higher than
  the 0.90 of the cross-family case.
- Data: the donor (vLLM, 4 samples per problem, temperature 0.7) solved 7071 of 7273 GSM8K train problems and
  5945 of 7300 MATH train problems (MATH-500 problems removed). Kept at most 2 verified solutions per problem:
  25,590 solutions (14,074 GSM8K, 11,516 MATH). 200 problems of each set stay out as a dev slice.
- Stage 2: SFT on those solutions, response tokens only, effective batch 16, glue lr 3e-4, gate lr 1e-4,
  Enc lr 1e-4, LoRA lr 2e-4, cosine. v1: 1000 steps. v2: 1500 steps plus token-level KD (weight 1) from the donor's
  logits on the same response tokens. The KD vocabulary is the first 151,665 ids, shared by both models.
- Stage 3: two rounds of verified self-training (expert iteration with an exact-answer reward, not GRPO). Each
  round samples 800 train problems twice from the current model (temperature 0.8), keeps the correct finished
  answers (about 1,050 to 1,180 per round), and trains 150 steps on them mixed with equal replay of donor solutions.
- Every arm gets the same data, the same step counts, the same seeds, and the final state is scored (no checkpoint
  selection). No learning rate sweep was run; the values above are the round 23 corrected values, and the other
  arms use common values (LoRA 2e-4, rank 128 at 1e-4, full fine-tune 1e-5).
- Scoring: GSM8K test (seeded 500), MATH-500, MMLU 1000 (guard), all greedy under `wsp.skills_eval`. The dev slice
  (150 GSM8K train problems held out of every arm) is in the rows.

## Arms and controls

base, donor (reference), graft, identity (same glue and LoRA, donor skipped), scrambled (same graft after
alignment, with the donor slice replaced by `wsp.scramble` spectrum-kept weights), lora_match (rank 19 plus the graft arms' rank 8 equivalent, sized to
the graft's total trainable, 21.9M), lora_big (rank 128, 147.7M), fullft (all 1.54B base weights).

## Results v2 (KD from the donor, 1500 SFT steps, two seeds)

GSM8K test 500, MATH-500, MMLU 1000, accuracy in percent, seeds 0 and 1:

| Arm | GSM8K | MATH-500 | MMLU | trainable | params run per token |
|---|---|---|---|---|---|
| base | 73.6 | 54.0 | 58.8 | 0 | 1.54B |
| graft | 85.8, 83.8 | 59.6, 58.0 | 59.6, 59.0 | 22.5M | 5.53B |
| identity control | 72.8, 71.0 | 54.4, 54.2 | 58.4, 60.3 | 22.5M | 1.57B |
| scrambled donor | 73.2, 73.2 | 54.4, 54.6 | 59.2, 58.8 | 22.5M | 5.53B |
| LoRA matched (r19) | 73.4, 72.2 | 52.0, 52.2 | 58.8, 59.4 | 21.9M | 1.57B |
| LoRA r128 | 72.8, 74.2 | 51.6, 49.6 | 59.6, 59.2 | 147.7M | 1.69B |
| full fine-tune | 69.4, 71.4 | 52.0, 52.8 | 58.5, 58.9 | 1.54B | 1.54B |
| donor alone | 92.2 | 76.8 | 52.6 | 0 | 7.07B |

Seed spread inside each control arm is 0.0 to 2.0 points on GSM8K. Every graft run is at least 9.6 points above
the best control run on GSM8K (83.8 against 74.2) and at least 3.4 points above on MATH-500 (58.0 against 54.6).
It closes about 60 percent of the base-to-donor gap on GSM8K and about 21 percent on MATH-500. MMLU does not move (spread 58.4 to 60.3 across all arms).

## Results v1 (donor solutions only, no KD, 1000 SFT steps, two seeds)

| Arm | GSM8K | MATH-500 | MMLU |
|---|---|---|---|
| graft | 81.4, 79.4 | 57.0, 54.2 | 59.1, 58.5 |
| identity | 72.8, 71.4 | 52.8, 53.2 | 59.3, 59.0 |
| scrambled | 72.4, 70.2 | 53.4, 51.6 | 58.9, 59.1 |
| LoRA matched | 73.4, 72.0 | 52.8, 50.4 | 59.4, 59.2 |
| LoRA r128 | 71.0, 71.4 | 49.6, 50.2 | 59.2, 59.6 |
| full fine-tune | 72.8, 72.2 | 52.2, 50.4 | 58.6, 59.1 |

Plain SFT on the donor's solutions does not raise test accuracy for any fine-tuning arm: they sit at or below the
base (73.6 and 54.0), although their dev loss falls. The graft already gains 6 to 8 points on GSM8K in v1.
KD adds about 4 points to the graft and about 0 to the controls (v2 minus v1 on GSM8K: graft +4.4 and +4.4,
identity 0.0 and -0.4, scrambled +0.8 and +3.0, LoRA matched 0.0 and +0.2, LoRA r128 +1.8 and +2.8, full
fine-tune -3.4 and -0.8).

## What this shows and what it does not

- The gain comes from the donor's weights. Identity (same glue, no donor) and scrambled (same slice size and
  spectrum, no knowledge) match the base within 3 points. The graft with the real slice is 10 to 14 points above.
- LoRA, a LoRA six times larger, and a full fine-tune of the base fall short on the same donor solutions, the
  same donor KD, and the same self-training budget.
- The graft does not get near the donor on MATH-500 (58 to 60 against 76.8). Possible causes, none tested:
  span 4..20 leaves the base's own layers 21..27 to read the result, the SFT data per problem is short, and
  1500 steps is a small budget.
- The graft is a bigger network at inference. It runs 5.53B parameters per token against 1.54B (the donor slice
  is 3.97B), and decodes at 14 to 26 tokens per second at batch 1 against 32 to 79 for the arms without the slice (these
  numbers are noisy: cells shared nodes with other containers, and unmerged LoRA arms are slower than merged
  ones). Matched trainable parameters do not mean matched compute.
- Spend: 24.6 GPU hours in the rows plus data generation and smokes, about 100 dollars on H100.
- Verified self-training was run in every arm but not ablated, so its share of the gain is unknown. The
  graft's rounds kept 1,052 to 1,180 correct samples against 1,003 to 1,126 for the other arms.
- One alignment pass and one hyperparameter set. No learning rate search, no GRPO, no span or host-layer sweep,
  two seeds only.
