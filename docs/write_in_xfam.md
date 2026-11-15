# Write-in, cross-family knowledge (2026-11-15)

Question: can frozen layers of Qwen2.5-7B-Instruct, wired into Llama-3.2-1B-Instruct through learned seams, beat every
fine-tuning arm of the base at the same training budget? The two models have different tokenizers and different
companies. The result stays one network: a frozen donor slice runs as a parallel branch, with no second model beside it.

## Result

Held-out test items that no arm trained on. Final recipe, 4000 task steps, fresh seeds 1, 2, 3 (mean, sample sd).

| arm | MMLU held-out 2000 | ARC-Challenge test | OpenBookQA test |
|---|---|---|---|
| base (no training) | 44.6 | 55.3 | 60.0 |
| donor Qwen2.5-7B-Instruct alone | 70.8 | 88.8 | 85.8 |
| graft (span 4..20, LoRA 16, soft KD) | 59.6 +-0.8 | 85.3 +-0.2 | 87.1 +-0.8 |
| graft, wider span 2..24 (2 seeds) | 59.1 +-0.0 | 84.7 +-0.2 | 87.5 +-0.1 |
| identity control (same glue, donor skipped) | 50.3 +-0.7 | 68.8 +-1.0 | 77.7 +-0.2 |
| scrambled-donor control (spectrum kept) | 50.3 +-0.4 | 69.2 +-0.8 | 77.8 +-0.7 |
| LoRA matched to graft trainable (26.1M) | 49.7 +-0.7 | 68.7 +-1.1 | 76.9 +-2.2 |
| LoRA rank 128 (90M) | 50.0 +-0.9 | 69.4 +-0.3 | 79.1 +-0.9 |
| full fine-tune of the base (1.24B) | 51.1 +-0.3 | 69.5 +-0.6 | 78.5 +-0.8 |

The graft beats full fine-tuning by 8.5 MMLU points, 15.8 ARC-C points, and 8.6 OBQA points. The smallest margin in
any single seed pair is 7.5 MMLU points, against a seed spread under 1 point. The scrambled donor, which has the same
spectrum and no knowledge, scores like the identity control, so the gain comes from what the donor slice knows. The graft
closes 57 percent of the MMLU gap between base and donor (44.6 to 70.8), and 90 percent on ARC-C (55.3 to 88.8). On OBQA the graft (87.1) is above the donor alone (85.8) in the mean of three seeds.
With the graft switched off at test time the same model falls back to 45 to 46 MMLU, so the gain runs through the seam.

The full per-seed table with params, trainable counts, tokens, GPU hours, and decode speed is in
`results/wi_xfam/table.md`. Rows are in `results/wi_xfam/rows/`. Regenerate with `scripts/wi_xfam_table.py`.

## What each arm costs

- Graft: 5.21B parameters run per token (1.24B base, 3.97B frozen donor slice of layers 4..20, 26.5M trainable glue and
  LoRA). Decode at batch 1 is 27 to 36 tok/s in these runs against 103 for the base and 62 to 72 for the LoRA arms. The
  graft is a hook-based research implementation without fused kernels or a cache for the donor branch, so this is
  not an optimized number. The graft also reads about 10.3M training tokens against 11.9M for controls.
- Controls take the 600 stage-1 alignment steps as extra task steps (round 23 rule), so they see more task data.
- tok/s varies between containers (full fine-tune 89 to 117); it is one measurement per cell.

## Recipe

- Data: 44,155 four-option items. MMLU auxiliary_train (32,287 after filtering), CommonsenseQA, OpenBookQA, QASC, and
  LogiQA train. Prompts equal to any eval prompt were removed. The donor scored every item once, and the soft A to D
  distribution is cached (`pool.json` on the volume, accuracy 85.9 percent on the pool).
- Loss: cross-entropy over the answer letters plus KL(donor soft || student) with weight 1, plus KL to the frozen base on
  general text with weight 1, plus every second step the stage-1 functional alignment loss as a regularizer.
- Graft: Llama host layer 7, Qwen layers 4..20, Enc depth 2, Dec rank 256, per-token gate, LoRA rank 16 on the base trained
  jointly. Learning rates: glue 3e-4, gate 1e-4, Enc 1e-4, LoRA 2e-4. Stage 1 alignment 600 steps, then 4000 task steps,
  batch 16, cosine schedule, the last step is scored (no checkpoint selection for any arm).
- Controls use the same data, steps, and selection. LoRA lr 2e-4 (matched), 5e-5 (rank 128), full fine-tune lr 3e-6 in fp32
  master weights. Each fine-tuning learning rate was picked on the MMLU dev slice from a 1200-step pilot:
  full fine-tune 1e-6 (51.1 dev), 3e-6 (52.8), 1e-5 (49.0), 3e-5 (43.3); rank-128 LoRA 5e-5 (52.0 at 4000), 1e-4 (49.9),
  3e-4 (41.2).

## Ablations (seed 0, dev slice = MMLU dev 1000; held-out shown for the record)

| config | steps | MMLU dev | MMLU held-out |
|---|---|---|---|
| graft span 4..20 | 1200 | 55.8 | 55.0 |
| same, no soft KD (CE only) | 1200 | 56.1 | 54.3 |
| span 2..24 | 1200 | 56.9 | 56.7 |
| span 4..20 plus LoRA 16 | 1200 | 57.6 | 57.0 |
| span 2..24 plus LoRA 16 | 1200 | 56.8 | 58.6 |
| span 4..20 plus LoRA 16 | 4000 | 59.3 | 58.5 |
| span 2..24 plus LoRA 16 | 4000 | 58.7 | 60.4 |
| span 2..24 plus LoRA 16, no soft KD | 4000 | 56.4 | 57.5 |

Soft KD from the donor adds about 2 points at 4000 steps. Joint LoRA adds about 1.5 points at 1200 steps. The wider
span did not beat the dev winner (span 4..20 plus LoRA, dev 59.3 against 58.7), so that is the headline config. On fresh
seeds the two spans tie (59.6 against 59.1), so one graft over 17 layers is enough at this data scale. Training longer
helps the graft (57.0 at 1200 steps, 58.5 at 4000 on seed 0) and does not help the fine-tuning arms (full fine-tune 50.7
at 1200 steps, 51.0 at 4000, seed 0).

A two-point graft (second point at layers 20..28) failed to build: Qwen2.5-7B has layers 0..27, and the preset asked
for layer 28. It was not rerun, since the single graft already clears every control. Not done: RL, MATH and
GSM8K tracks, more than one graft point, a larger pool.

## Honest limits

- The donor's own score on these sets is 70.8 MMLU held-out; the graft reaches 59.6, so 43 percent of the gap is open.
- The graft runs 4.2x the parameters per token and decodes about 2.5x slower than the LoRA arms. "Same training budget"
  holds for data, steps, and trainable parameters, not for inference cost.
- Training items include ARC and OpenBookQA train splits (inside MMLU auxiliary_train and the MC pools). ARC-Challenge test
  and OpenBookQA test are not in the pool, but the in-domain format helps all arms equally.
- The headline config was picked on seed 0 and dev, confirmed on seeds 1, 2, 3. The span 2..24 arm has 2 fresh seeds.

## Rerun

    modal run modal_wi_xfam.py --prep-data true
    modal run --detach modal_wi_xfam.py --cells graft_lora:1,scrambled_l:1,identity_l:1 --overrides '{"steps":4000,"tag":"_final"}'
    modal run --detach modal_wi_xfam.py --cells lora_match_l:1,fullft:1 --overrides '{"steps":4000,"tag":"_final","ft_lr":3e-6}'
    modal run --detach modal_wi_xfam.py --cells lora_big:1 --overrides '{"steps":4000,"tag":"_final","lora_lr":5e-5}'

Cost: 14.2 GPU hours in the stored rows (5.7 on H100, 8.5 on H200), plus about 0.3 for prep and smokes and two OOM
cells, so about 65 dollars at list price.
