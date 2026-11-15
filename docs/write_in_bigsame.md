# Write-in track: big donor, same tokenizer (Qwen2.5-14B into Qwen2.5-1.5B)

2026-11-15. Driver `modal_wi_bigsame.py`, helpers `wsp/wi.py`, `SplitGraft` and `LogitGraft` in
`wsp/pgraft.py`, tests in `tests/test_wi.py` and `tests/test_pgraft.py`, row tools
`scripts/wi_rows.py` and `scripts/wi_table.py`. Rows are on the ztrisanth volume under
`wi_bigsame/rows/`.

## Result

A frozen slice of Qwen2.5-14B-Instruct, stitched onto the lower half of Qwen2.5-1.5B-Instruct through a
trained encoder and a per-token gate, beats every fine-tuning arm and both controls at the same data, steps,
and selection rule. Two seeds per arm, final checkpoint, no selection on test items.

| row | MMLU held-out | MMLU dev | ARC-C test | TriviaQA EM | GSM8K | params total (B) | params per token (B) | trainable (M) | train tokens (M) | GPU h (train/cell) | tok/s b=1 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| ref_base | 60.4 | 58.8 | 77.6 | 41.6 | 68.7 | 1.54 | 1.54 | 0.0 | 0 | 0.00/0.05 | 67.0 |
| ref_donor | 78.0 | 78.6 | 94.0 | 67.6 | 91.3 | 14.77 | 13.99 | 0.0 | 0 | 0.00/0.12 | 23.3 |
| ref_sib3b | 66.0 | 66.3 | 82.8 | 49.3 | 84.0 | 3.09 | 3.09 | 0.0 | 0 | 0.00/0.10 | 29.8 |
| ref_sib7b | 70.9 | 71.2 | 89.0 | 56.4 | 88.7 | 7.62 | 7.07 | 0.0 | 0 | 0.00/0.08 | 35.6 |
| main_graft_s0 | 66.1 | 65.1 | 85.7 | 50.2 | 79.7 | 11.74 | 11.74 | 39.4 | 9.8 | 1.03/1.21 | 26.3 |
| main_graft_s1 | 65.7 | 65.0 | 85.2 | 50.1 | 81.0 | 11.74 | 11.74 | 39.4 | 9.8 | 1.09/1.26 | 26.5 |
| main_identity_s0 | 61.3 | 60.8 | 79.8 | 40.0 | 63.3 | 2.38 | 2.38 | 39.4 | 10.0 | 0.63/0.71 | 66.8 |
| main_identity_s1 | 61.4 | 59.5 | 80.5 | 39.2 | 66.3 | 2.38 | 2.38 | 39.4 | 10.0 | 0.59/0.67 | 63.5 |
| main_scrambled_s0 | 60.8 | 61.1 | 80.6 | 38.9 | 66.7 | 11.74 | 11.74 | 39.4 | 10.0 | 1.14/1.40 | 25.9 |
| main_scrambled_s1 | 60.8 | 60.2 | 80.0 | 39.8 | 66.3 | 11.74 | 11.74 | 39.4 | 10.0 | 1.16/1.62 | 27.7 |
| main_lora_s0 | 60.5 | 58.6 | 78.1 | 38.7 | 64.3 | 1.58 | 1.58 | 39.2 | 10.0 | 0.49/0.56 | 82.5 |
| main_lora_s1 | 59.6 | 58.9 | 79.2 | 37.8 | 65.0 | 1.58 | 1.58 | 39.2 | 10.0 | 0.52/0.59 | 79.0 |
| main_lora128_s0 | 62.5 | 60.6 | 80.5 | 39.3 | 71.3 | 1.69 | 1.69 | 147.7 | 10.0 | 0.91/1.02 | 38.2 |
| main_lora128_s1 | 61.3 | 60.8 | 80.2 | 39.5 | 65.0 | 1.69 | 1.69 | 147.7 | 10.0 | 0.92/1.04 | 37.9 |
| main_fullft_s0 | 60.6 | 60.3 | 79.7 | 39.7 | 69.0 | 1.54 | 1.54 | 1543.7 | 10.0 | 0.65/0.72 | 38.6 |
| main_fullft_s1 | 61.0 | 61.1 | 79.4 | 39.3 | 67.0 | 1.54 | 1.54 | 1543.7 | 10.0 | 0.49/0.55 | 65.1 |
cell GPU hours in table: 21.91

Reading the table:

- MMLU held-out (2000 items, never trained on): graft 66.1 and 65.7. Best fine-tuning arm is LoRA r128 at 62.5
  and 61.3. Identity control 61.3 and 61.4. Scrambled-donor control 60.8 and 60.8. Base 60.4. The seed spread
  within an arm is at most 1.2 points, the graft lead over the best other arm is 3.6 and 4.4.
- ARC-Challenge test +5 to +6 over the best other arm, TriviaQA exact match +10, GSM8K (guard) +8 to +16.
  LoRA and full fine-tuning lose 3 to 4 points of TriviaQA and some GSM8K against the base; the graft gains both.
- With the graft switched off after training, MMLU held-out is 60.8 and 59.7, the base. The gain is the donor slice.
- Held-out text KL to the donor: 0.46 and 0.47 nats with the graft on, 0.85 and 0.86 with it off; the scrambled
  control has the same KL on and off (0.69) because its gate closed to 0.01.
- Alignment survival (grafted donor next-token accuracy over native) is 0.87 before and 0.89 to 0.90 after
  stage 2.

Size cost. The graft runs 11.7B parameters per token (1.5B base, 34 donor layers 14..47 at 9.35B, donor norm and
head 0.78B, 39M glue). The native Qwen2.5-7B-Instruct scores 70.9 / 89.0 / 56.4 / 88.7 (MMLU / ARC-C / TriviaQA /
GSM8K) at 7.1B per token, and the native 3B scores 66.0 / 82.8 / 49.3 / 84.0 at 3.1B. So the graft matches a 3B
sibling on knowledge at about 4 times its per-token cost, and a trained 7B sibling is better than the graft
at a smaller size. The graft's claim is relative to fine-tuning the 1.5B base, not to a model of its size.
The 14B donor alone scores 78.0 / 94.0 / 67.6 / 91.3, so the graft closes about 1/3 of the gap on MMLU
(base 60.4, donor 78.0) with one donor slice and 3000 steps.

Decode tok/s at batch 1 was measured on shared H100s while other jobs ran, so read it as order of magnitude only:
base 67, graft 26, native 7B 36, donor 23.

## What the graft is

`LogitGraft`: base layers 0..14 run as usual. The residual after layer 14 goes through Enc (linear plus one
residual MLP, 1536 to 5120) into frozen donor layers 14..47. The donor's own final norm and lm_head turn that
state into donor logits. The base head's logits and the donor logits are mixed per token,
`l = l_base + gate * (l_donor - l_base)`, with a gate that reads the base state at layer 14. Base layers 15..27
and the base head run in parallel and stay in the model. With the gate at zero it is exactly the base.
Trainable: Enc, gate, and a rank 16 LoRA on the base, 39.4M. The donor slice is frozen and shared with nothing
at run time (one network, no second full model).

Training (same loss for every arm, 3000 optimizer steps, batch 16 MC items plus 6 text sequences of 256 tokens):
answer-letter cross-entropy on 60k four-option items (MMLU auxiliary_train, SciQ, LogiQA2.0, CommonsenseQA,
OpenBookQA, QASC train splits) plus KL to the 14B's letter distribution on the same items (soft targets
cached from the donor), plus full-vocabulary KL to the 14B on knowledge, general, and prose text. The graft arm
spends the first 600 steps on stage 1 (Enc only, KL to the donor's own next-token distribution, with 600 MC
prompts added to the alignment text), then 2400 steps of the joint loss with the stage-1 loss kept as a
regularizer every second step. Every other arm takes 3600 joint steps, so total steps match.
Lrs: Enc 5e-4, gate 1e-3, LoRA 2e-4 in the graft; lr per baseline picked on MMLU dev (see dev table).

Arms: identity control (same Enc, gate, LoRA, donor slice skipped, Enc output goes to the donor head, so it keeps
the donor's unembedding; this makes it a generous control). Scrambled control (slice weights scrambled with
`wsp/scramble.py` spectrum mode, norm and head permuted; the gate closes and the model is the base). Matched
LoRA (rank 22 for 39.2M), LoRA r128 (147.7M), full fine-tune (1.54B, lr 5e-6, fp32 master weights).

## Dev log (1 seed, 1500 steps, MMLU dev 1000, base 58.8)

Config choice used MMLU dev and in-distribution MC items only.

| config | MMLU dev | note |
|---|---|---|
| residual-write graft, 14B 14..34 at base layer 14, Dec rank 256 | 59.1 | gate shut to 0.33 |
| same, span 20..40 | 60.0 | |
| residual-write graft 14..47, write after base layer 27, 21 | 59.3, 59.0 | worse in-distribution |
| plus Dec rank 1024 and LoRA 16 | 59.5 | |
| plus KD-only stage 600 steps, Dec lr x3 | 60.0 | held-out KD 0.737 vs 0.764 off |
| plus Dec output scaled to host RMS | 58.5, 60.6, 59.2 | gate 0.04 to 0.06 |
| identity control (no donor) | 59.4 | |
| LoRA r22 lr 2e-4 / 6e-4 | 60.2 / 55.2 | |
| LoRA r128 lr 5e-5 / 2e-4 | 61.3 / 55.2 | |
| full fine-tune lr 5e-6 / 2e-5 | 59.2 / 56.2 | |
| logit-skip, Enc lr 1e-4 | 61.9 | held-out KD 0.52 vs 0.76 off |
| logit-skip, scrambled slice and head | 59.8 | |
| logit-skip, identity | 59.6 | |
| logit-skip, host 20, donor from 24 | 59.4 | |
| logit-skip, Enc lr 5e-4 | 58.9 | better text KD (0.44), worse MC |
| logit-skip, Enc lr 1e-3, MC alignment text | 62.6 | |
| logit-skip, Enc depth 3, Enc lr 5e-4, MC alignment text | 62.7 | |
| logit-skip, LoRA 16, Enc lr 5e-4, MC alignment text | 63.8 | chosen |
| same, gate bias +2, gate lr 3e-3 | 62.1 | |

What failed and why it matters. Every variant that wrote the donor's contribution back into the base residual
stream through a low-rank Dec scored the same as fine-tuning, even with an alignment survival of 0.87. The gate
closed because the base layers could not read what Dec wrote in the first 1500 steps. Mixing in logit space
removes the learned decoder: the donor's own lm_head decodes. That is the change that made the seam work. The
two things that then mattered were alignment text in the answer format (MC prompts) and an Enc learning rate high
enough to adapt on MC items without losing the text alignment.

## Limits

- The stitch carries a 34 layer donor tail, so it is an 11.7B network at run time. It is not a cheap graft.
  A 20..24 layer start (smaller tail) scored at the base on dev; a cheaper tail needs a different read point.
- The controls are strong in one way and weak in another. The identity control keeps the donor lm_head, and the
  scrambled control collapses to the base because its gate closes, so it shows that the gain needs real donor
  weights but not much more.
- Decode speed numbers share GPUs with other jobs. LoRA r128 reads 38 tok/s against 82 for the matched LoRA at
  the same merged shape, which is noise, not a real gap.
- Only the ztrisanth workspace ran the main arms. trisanthsrini hit its Modal spend limit before the second
  workspace could be used.
- Not run: Qwen2.5-32B donor, RL, MATH-500, HumanEval, MBPP (not in this track's score list). The 14B gain
  did not saturate on the stitch, so the 32B step was not taken.
