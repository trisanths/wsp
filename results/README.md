# Results log

All numbers here are on the GSM8K search split (200 items from GSM8K train)
unless a row says held-out. Search-split numbers are interim checks, not
results to report.

## Run 1 (2026-11-14): first cross-family graft

Host unsloth/Llama-3.2-1B-Instruct. Donor Qwen/Qwen2.5-Math-1.5B, layer 18.
Part: 512 MLP neurons chosen by mean |activation| on GSM8K train text,
grafted at host layer 10. Heal: 300 steps, batch 8, seq 384, lr 3e-4.
One seed. Raw file: graft_run1_s0.json.

| Model | Search accuracy |
|---|---|
| Host, no heal | 25.0% |
| Random 512-neuron control + glue + heal | 33.0% |
| Top-activation cluster + glue + heal | 25.0% |

Reading:
- The pipeline runs end to end across families in about 25 s per compile.
- The graft does not beat the control. The control's gain over the host
  comes from the heal format and 6.3M glue parameters, 2.7x the 2.4M part.
- Mean |activation| picks always-loud neurons, not math neurons.
- The pooled fit R squared of 0.987 is suspect. A review later found the fit
  read the host state without BOS, so first-token outlier rows dominate it.

Next: contrast and attribution extractors, per-dimension R squared, the fit
fixes, 3 seeds, a sweep over donor layer and cluster size, and low-rank glue
so the glue is smaller than the part.

## Run 2 (2026-11-14): does part selection matter? (72 cells)

Same host and donor. 512-neuron MLP clusters from donor layers 10, 18, and
24, chosen by mean |activation|, by contrast (GSM8K text minus WikiText),
by attribution (estimated loss increase when zeroed), or at random (the
control). Glue full rank (6.3M params) or rank 64 (0.46M). 300 heal steps,
3 seeds. This run includes the 20 review fixes. Raw rows: sweep2_rows.json.

Mean search accuracy over 3 seeds (sd in brackets):

| Donor layer (host layer) | Activation | Contrast | Attribution | Random |
|---|---|---|---|---|
| 10 (6), full glue | 31.2 (1.0) | 32.8 (1.3) | 34.5 (2.6) | 38.7 (3.5) |
| 10 (6), rank 64 | 40.5 (1.0) | 38.7 (0.3) | 38.3 (0.8) | 38.7 (3.3) |
| 18 (10), full glue | 22.3 (3.3) | 24.3 (2.8) | 24.2 (1.5) | 26.0 (1.0) |
| 18 (10), rank 64 | 25.7 (1.0) | 24.7 (3.4) | 26.2 (2.5) | 24.8 (1.8) |
| 24 (14), full glue | 21.7 (2.6) | 21.8 (4.0) | 24.8 (2.8) | 22.7 (4.0) |
| 24 (14), rank 64 | 24.8 (1.9) | 24.2 (0.3) | 23.5 (0.5) | 23.2 (1.0) |

Fit R squared (pooled / per dimension): layer 10 0.68 / 0.53, layer 18
0.66 / 0.49, layer 24 0.57 / 0.45. Run 1's 0.987 was inflated by the BOS
outlier rows the review found.

Reading:
- No selection method beats the random control at any layer. Single-layer
  512-neuron clusters do not carry transferable math skill for this pair.
- Placement dominates: grafting at host layer 6 gives about 38 to 40%,
  host layers 10 and 14 give about 22 to 26%, whatever the part is. That is
  the glue acting as an adapter at a good depth.
- Rank-64 glue is as good as or better than full glue at 7% of the params.
- Missing arm: heal with no graft at all (or a matched LoRA). Round 3 adds it.

Next (round 3): whole donor blocks (layers 8..19 and 4..23) against a
matched LoRA, a standard rank-64 LoRA, and a random-block control.

## Run 3 (2026-11-14): whole donor blocks vs LoRA (18 cells)

Host Llama-3.2-1B-Instruct, donor Qwen2.5-Math-1.5B, 300 heal steps, same
loop and data for every arm, 3 seeds. Raw rows: round3_rows.json.

| Arm | Trainable params | Total params | Search acc. (sd) | Decode tok/s |
|---|---|---|---|---|
| LoRA rank 1 (smallest rank, "matched") | 0.70M | 1.24B | 43.5 (2.0) | 87 |
| LoRA rank 64, all linear layers | 45.1M | 1.24B | 28.8 (3.4) | 110 |
| Block, donor layers 8..19, rank-64 glue | 0.46M | 1.80B | 35.2 (0.6) | 42 |
| Block, donor layers 4..23, rank-64 glue | 0.46M | 2.17B | 33.5 (3.5) | 37 |
| Random block (same length), rank-64 glue | 0.46M | 1.80B | 36.3 (3.0) | 37 |
| Block, donor layers 8..19, full glue | 6.3M | 1.80B | 16.5 (3.6) | 48 |

Fit R squared (pooled / per dim): block 8..19 at host layer 5 0.70 / 0.55;
block 4..23 at host layer 2 0.75 / 0.64.

Reading:
- A rank-1 LoRA beats every graft. Donor blocks do not beat a random block.
  For this cross-family pair, the donor's math skill does not survive a
  linear stitch plus 300 glue steps.
- The likely cause is the stitch itself: a per-token linear map from Llama
  states to Qwen states explains only about 55% of per-dimension variance,
  and the donor block runs on Llama's tokenization, not its own. The donor
  layers see inputs off their training distribution.
- Big LoRA at lr 3e-4 underperforms (28.8%); the LoRA baseline needs an lr
  check before it is a fair bar.
- Block grafts cost decode speed (37 to 48 tok/s vs 87 to 110) because the
  KV cache is off for them; the cache work in progress addresses that.

Next (round 4): the same test with a host that shares the donor's
tokenizer (Qwen2.5-0.5B-Instruct host, Qwen2.5-Math-1.5B donor), to
separate "the stitch is too lossy" from "tokenization differs".

## Run 4 (2026-11-14): the stitch is the bottleneck

Raw rows: round4_rows.json.

Stitch penalty. Host Qwen2.5-0.5B-Instruct and donor Qwen2.5-Math-1.5B
share a tokenizer, so positions line up. Replace the donor's input to layer
l with the ridge map of host layer k, run the rest of the donor, and score
the donor's own next-token loss on 150 GSM8K texts.

| Host layer to donor layer | Pooled R2 | Per-dim R2 | Donor loss |
|---|---|---|---|
| none (native) | | | 0.80 |
| 3 to 4 | 0.70 | 0.57 | 2.85 |
| 7 to 8 | 0.69 | 0.52 | 2.16 |
| 12 to 14 | 0.64 | 0.45 | 2.99 |
| 17 to 20 | 0.60 | 0.44 | 2.78 |

A linear per-token stitch raises the donor's loss by 1.4 to 2.2 nats even
with the same tokenizer. The donor block cannot do its job on that input.

Grafts on the same-tokenizer host (3 seeds, 300 heal steps):

| Host / arm | Trainable params | Search acc. (sd) |
|---|---|---|
| Qwen-0.5B host, untrained | 0 | 33.5 |
| Qwen-0.5B LoRA r1, lr 3e-4 | 0.55M | 37.3 (1.6) |
| Qwen-0.5B LoRA r16, lr 1e-4 | 8.8M | 36.3 (1.4) |
| Qwen-0.5B LoRA r64, lr 5e-5 | 35.2M | 36.3 (0.6) |
| Qwen-0.5B block 8..19, rank-64 glue | 0.31M | 36.3 (2.5) |
| Qwen-0.5B block 4..23, rank-64 glue | 0.31M | 36.3 (3.8) |
| Qwen-0.5B random block, rank-64 glue | 0.31M | 37.8 (0.8) |
| Llama-1B LoRA r16, lr 1e-4 | 11.3M | 44.0 (0.5) |
| Llama-1B LoRA r64, lr 5e-5 | 45.1M | 46.7 (1.6) |

Reading:
- Same tokenizer does not rescue grafts: donor blocks equal a random
  block. The stitch, not tokenization, is the bottleneck.
- The fair LoRA bar on Llama-1B is 46.7% (r64, lr 5e-5). Round 3's 28.8%
  for r64 came from a too-high lr.
- The stitch penalty is a cheap, eval-free target. Next: train the stitch
  to drive it down (linear and MLP, on regression, on the donor's own LM
  loss, and on matching later donor states), before any task heal.

## Audit (2026-11-14): extraction is exact; the attention sink was broken

Raw rows: audit_sink_rows.json. Same eval texts as run 4's penalty.

Extraction and compile on real Qwen2.5-Math-1.5B:
- Range-fetched tensors equal the checkpoint (max diff 0.0).
- A block part (layers 8..19) built from fetched tensors equals the donor's
  own layers on real text (max diff 4e-5 at scale 243, float noise).
- An mlp_cluster with every neuron equals the donor MLP (diff 0.0).
- Feeding the donor its own states through the stitch hook reproduces the
  native loss exactly.

Stitch penalty into Qwen2.5-Math-1.5B layer 8 (donor loss minus native 0.805):

| Pair | Ridge as shipped | Ridge, lam 1e-6 | + donor sink at position 0 |
|---|---|---|---|
| Math L7 to Math L8 (identity) | 0.052 | 0.000 | 0.000 |
| Qwen2.5-1.5B base L7 (same init) | 1.235 | 1.170 | 0.367 |
| Qwen2.5-0.5B-Instruct L7 | 1.357 | 1.441 | 0.643 |

Reading:
- The compiler's extraction is correct. The loss came from the stitch.
- Qwen adds no BOS token, so the first text token is the attention sink and
  carries massive, input-invariant activations. The fit skipped it, so every
  graft fed the donor a garbage sink and broke attention for all later
  tokens. Feeding the donor's mean sink state at position 0 cuts the penalty
  by 2.1x to 3.4x. The mean equals the per-text oracle, because the sink does
  not depend on the input (as Sun et al. 2024 report).
- The shipped ridge regularizer costs 0.05 nats on the identity map.
- Next: a stored sink vector per graft, RMS matching on the projector, a
  small LoRA on the donor part (the receiver), and a trained stitch, then a
  rerun against the 46.7% LoRA bar.

## Run 5 (2026-11-14): prefill taps on GSM8K

Tap: the donor reads the prompt natively with its own tokenizer; the host
reads its layer-l states through a gated cross-attention branch at host
layer k. Host Llama-3.2-1B-Instruct, 300 steps, completion-only loss,
branch lr 3e-4, gate lr 3e-2, 3 seeds. Raw rows: round5_rows.json.

| Arm | Trainable | Search acc. (sd) | Decode tok/s |
|---|---|---|---|
| Host, untrained | 0 | 25.0 | 109 |
| Qwen2.5-Math-1.5B-Instruct alone (raw prompt) | 0 | 33.0 | 54 |
| LFM2-350M-Math alone (raw prompt) | 0 | 1.0 | 62 |
| Tap Qwen-Math L14 into host L10 | 1.8M | 26.7 (1.3) | 95 |
| Tap Qwen-Math L27 into host L12 | 1.8M | 25.8 (0.8) | 104 |
| Two taps (L14 and L27) | 3.7M | 27.5 (2.2) | 102 |
| Control: mismatched messages (L27) | 1.8M | 26.2 (3.8) | 94 |
| Control: shuffled donor weights (L27) | 1.8M | 22.8 (4.3) | 126 |
| Tap LFM2-350M-Math L15 into host L12 | 1.6M | 26.3 (0.8) | 75 |
| Tap LFM2-350M-Math L8 into host L8 | 1.6M | 32.7 (0.8) | 106 |
| LoRA rank 64 (lr 5e-5) | 45.1M | 47.2 (1.4) | 129 |

Reading:
- Taps learn the GSM8K output format but carry no reasoning: real donor
  messages score the same as mismatched ones. During prefill the donor has
  not solved the problem, so there is little to read (matches "Cross-Model
  Stitching Transfers Answers, Not Computation", 2026).
- The LFM2-350M-Math L8 tap is the only arm above noise (+7.7 over host).
- The donors are weak on the raw prompt (chat-tuned models; LFM2-350M-Math
  needs its chat template).
- A one-layer 1.8M adapter is far weaker than a 45M all-layer LoRA, and the
  earlier rank-1 LoRA (0.7M, 43.5%) shows distribution across layers
  matters more than size.

Next (round 6): taps on MMLU, where the answer is already in the donor's
prefill states and fine-tuning cannot add knowledge; and LoRA plus a tap
against LoRA alone on GSM8K.

## Run 6 (2026-11-14): a tap beats fine-tuning on knowledge

Raw rows: round6_mmlu_rows.json, round6_gsm8k_rows.json.

MMLU. Host Llama-3.2-1B-Instruct, donor Qwen2.5-7B-Instruct. 500 steps on
8000 MMLU auxiliary_train pairs (prompts up to 240 tokens), batch 8; scored
on 1000 seeded MMLU test items by the log-probability of " A" to " D".
Branch lr 3e-4, gate lr 3e-2, LoRA lr 1e-4. 3 seeds.

| Arm | Trainable params | MMLU acc. (sd) |
|---|---|---|
| Host, untrained | 0 | 47.5 |
| Donor alone | 0 | 71.5 |
| LoRA rank 16 (fine-tuning) | 11.3M | 50.3 (1.1) |
| Tap donor L20 into host L10 | 2.9M | 67.9 (0.4) |
| Tap donor L27 into host L12 | 2.9M | 65.6 (4.1) |
| Control: tap L27 on mismatched messages | 2.9M | 48.6 (0.3) |
| Tap L27 plus LoRA r16, trained jointly | 14.2M | 49.2 (1.6) |

GSM8K, LoRA rank 64 plus a tap against LoRA alone (same loop, 3 seeds):
LoRA plus null tap 47.5 (2.3); plus LFM2-350M-Math L8 tap 47.0 (0.9); plus
Qwen-Math L14 tap 47.5 (4.8). Gates stay near 0.

Reading:
- The tap recovers 85% of the host-to-donor gap (+20.4 points) with 2.9M
  trainable parameters and about 100 s of training. Fine-tuning gains 2.8
  points, because the knowledge is not in its data.
- The mismatched-message control falls to host level, so the gain comes
  from the donor's states for that question, not from added capacity.
- Joint tap plus LoRA training collapses the tap: the LoRA takes the easy
  path first and the gate stays near 0. Train the tap first, then LoRA.
- On GSM8K, prefill taps add nothing on top of LoRA: multi-step reasoning
  is not in the donor's prefill states.
- These are numbers on a fixed 1000-item MMLU test sample used during
  development, not a held-out result yet.

## Run 7 (2026-11-14): held-out MMLU, layer map, scaling, TriviaQA

Raw rows: round7_rows.json. Host Llama-3.2-1B-Instruct; tap settings as run 6.

Held-out MMLU (2000 test items disjoint from the dev 1000, scored once):

| Arm | Accuracy by seed | Mean |
|---|---|---|
| Host | 44.7 | 44.7 |
| LoRA rank 16 | 47.0, 48.7, 47.4 | 47.7 |
| Tap Qwen2.5-7B L20 into host L10 | 67.7, 69.9, 67.2 | 68.3 |
| Donor Qwen2.5-7B alone | 70.9 | 70.9 |

The tap beats fine-tuning by 20.6 points held out and recovers 90% of the
host-to-donor gap.

Layer map (dev items, 1 seed; rows donor layer, columns host layer):

| Donor layer | Host L6 | Host L10 | Host L14 |
|---|---|---|---|
| 8 | 48.5 | 48.3 | 49.4 |
| 12 | 47.9 | 48.6 | 48.9 |
| 16 | 63.8 | 59.3 | 56.3 |
| 20 | 68.5 | 67.7 | 67.1 |
| 24 | 53.6 | 63.9 | 60.9 |

Knowledge is readable near 70% of donor depth (layer 20 of 28) and not
before layer 16.

Donor scaling (dev items, tap into host L10, 2 seeds):

| Donor | Donor alone | Tap mean |
|---|---|---|
| Qwen2.5-1.5B (L20) | 58.9 | 58.2 |
| Qwen2.5-3B (L25) | 66.2 | 57.9 |
| Qwen2.5-7B (L20) | 71.5 | 67.7 |
| Qwen2.5-14B (L34) | 78.5 | 73.6 |

A 1B host with a 14B tap beats the 7B donor alone. The 3B row likely
used a poor donor layer.

Staged training (tap 500 steps, freeze, then LoRA r16 500 steps), dev
items: 67.9, 69.2, 69.7 (mean 68.9) vs tap alone 67.9.

TriviaQA (1000 validation questions, alias match on a 16-token answer):

| Arm | Accuracy | Decode tok/s |
|---|---|---|
| Host | 37.6 | 97 |
| Donor Qwen2.5-7B | 46.8 | 63 |
| LoRA rank 16 (3 seeds) | 41.1 | 54 to 57 |
| Tap L20 into L10 (3 seeds) | 25.0, 34.5, 34.3 | 88 to 121 |

The tap hurts free-form answers even though it decodes at host speed. The
current tap transfers knowledge into a single next-token choice (MMLU)
but disrupts multi-token generation. Open question.

Reading against the goal: a single tap stays below its donor, so a weak
host plus one tap is "most of a big model's knowledge at small-model
decode cost", not a better model. Round 8 (several donor families) and
round 9 (taps into the strongest model) test whether composition can
beat the best source.

## Run 8 (2026-11-14): knowledge taps across donor families

MMLU dev items, same tap settings as run 6, 2 seeds. Taps read donors at
about 70% depth; Llama-1B host layer 10, Qwen-0.5B host layer 15. Raw
rows: round8_rows.json. Mistral cells failed (sentencepiece missing in the
image; fixed for later runs).

| Model | Score (seeds) | Donor alone |
|---|---|---|
| Llama-3.2-1B host | 47.5 | |
| Qwen2.5-0.5B host | 48.0 | |
| Llama-1B + Gemma-2-9B tap | 72.9 (72.9, 73.0) | 72.6 |
| Llama-1B + Qwen2.5-14B tap | 70.5 (73.6, 67.4) | 78.5 |
| Llama-1B + Qwen2.5-7B tap | 66.7 (65.6, 67.7) | 71.5 |
| Llama-1B + Phi-3.5-mini tap | 66.3 (66.9, 65.7) | 68.5 |
| Llama-1B + Llama-3.1-8B tap | 53.6 (49.2, 58.0) | 66.6 |
| Llama-1B + LFM2.5-1.2B tap | 49.9 | 26.2 (raw prompt) |
| Qwen-0.5B + Qwen2.5-14B tap | 74.8 (74.5, 75.2) | 78.5 |
| Qwen-0.5B + Gemma-2-9B tap | 72.7 (72.3, 73.1) | 72.6 |
| Qwen-0.5B + Qwen2.5-7B tap | 69.3 (69.9, 68.7) | 71.5 |
| Llama-1B + Qwen-7B + Gemma-9B taps | 72.0 (71.0, 73.0) | |
| Llama-1B + Qwen-14B + Gemma-9B taps | 73.7 (74.1, 73.3) | |
| Llama-1B + Gemma-9B + Llama-8B taps | 70.7 (69.1, 72.3) | |
| Llama-1B + Qwen-7B + Llama-8B taps | 64.0 (62.6, 65.5) | |

Reading:
- Small hosts reach their donors: Llama-1B and Qwen-0.5B with a Gemma-9B
  tap match Gemma-9B alone (72.9 and 72.7 vs 72.6). Qwen-0.5B with a 14B
  tap scores 74.8, above every 7B to 9B model here, at 0.5B decode cost.
- Two taps into a weak host are not additive yet: the pair scores near
  the better single tap.
- Llama-3.1-8B at layer 22 and LFM2.5-1.2B are weak or unstable donors in
  this setup; their read layer and prompt format need tuning.

## Baselines (2026-11-14): full fine-tuning and donor ceilings on GSM8K

Scored with Person B's wsp_eval harness on the GSM8K search split (raw
prompt). Full fine-tuning: 300 steps, batch 8, seq 384, 30 warmup steps,
3 seeds. Raw rows: baselines_rows.json. Two chat-mode cells failed in vLLM
(aclose RuntimeError).

| Model | Trainable | Score by seed | Mean |
|---|---|---|---|
| Llama-1B full FT, lr 1e-5 | 1.24B | 36.0, 28.5, 36.0 | 33.5 |
| Llama-1B full FT, lr 3e-5 | 1.24B | 14.5, 13.5, 13.5 | 13.8 |
| Qwen-0.5B full FT, lr 1e-5 | 494M | 42.0, 39.5, 42.0 | 41.2 |
| Qwen-0.5B full FT, lr 3e-5 | 494M | 24.0, 20.0, 19.5 | 21.2 |

Donor ceilings, no training, raw prompt: Llama-3.2-3B-Instruct 59.0,
LFM2.5-1.2B-Instruct 38.5, Qwen2.5-Math-1.5B-Instruct 32.0,
LFM2.5-1.2B-Thinking 27.5, Qwen2.5-1.5B-Instruct 7.0, LFM2-350M-Math 1.0.

Reading: at 300 steps, full fine-tuning is worse than LoRA rank 64 (47.2)
on Llama-1B, so LoRA r64 stays the fine-tuning bar on GSM8K. Chat-tuned
donors score badly on the raw prompt; donor ceilings need chat-format
scoring to mean anything.

## Run 9 (2026-11-14): taps into the strongest model do not help

MMLU dev items, 2 seeds; taps read donors at about 70% depth and write
into the host at about 40% depth. Raw rows: round9_rows.json.

| Host | Alone | With taps (mean of 2 seeds) |
|---|---|---|
| Qwen2.5-14B | 78.5 | + Llama-8B 77.9; + Qwen-7B 77.2; + Gemma-9B and Llama-8B 76.5; + Gemma-9B 75.6; mismatched-message control 76.0; LoRA r16 78.1 |
| Gemma-2-9B | 72.6 | + Qwen-14B 73.0; + Qwen-7B and Llama-8B 72.8 |
| Qwen2.5-7B | 71.5 | + Gemma-9B 71.8 |
| Llama-3.1-8B | 66.6 | + Gemma-9B and Qwen-7B 66.5 |

Reading: taps carry a strong donor's knowledge into a weak host (runs 6
to 8), but a strong host gains nothing from equal or weaker donors, and
500 steps of MMLU-format training slightly hurt it (as LoRA does). No
composition beats its best source here. Next: per-item scores for every
model to measure complementarity (oracle and probability-average
ensembles) before more composition designs.

## Stitch study (2026-11-14): a trained stitch keeps the donor intact

Donor Qwen2.5-Math-1.5B. Stitch maps host layer k output into donor layer
l input, with the donor sink at position 0. 1000 steps on 1200 GSM8K heal
plus 1200 WikiText-103 texts; penalty = donor loss with the stitched input
minus native, on 150 held-out texts of each kind. Raw rows:
stitch_study_rows.json.

| Host, k to l | Method | GSM8K penalty | WikiText penalty |
|---|---|---|---|
| Qwen2.5-1.5B base, 7 to 8 | ridge | 0.399 | 0.182 |
| | donor_lm linear | 0.029 | -0.164 |
| | donor_lm mlp | -0.014 | -0.207 |
| | downstream mlp | 0.264 | -0.008 |
| | mse linear | 0.399 | 0.177 |
| Qwen2.5-1.5B base, 13 to 14 | ridge | 0.423 | 0.001 |
| | donor_lm mlp | 0.122 | -0.169 |
| Qwen2.5-0.5B-Instruct, 7 to 8 | ridge | 0.647 | 0.500 |
| | donor_lm linear | 0.090 | -0.153 |
| | donor_lm mlp | 0.078 | -0.164 |
| Qwen2.5-0.5B-Instruct, 12 to 14 | ridge | 0.726 | 0.452 |
| | donor_lm mlp | 0.172 | -0.108 |

Cross-tokenizer (Llama-3.2-1B-Instruct host; downstream loss at donor
layer l + 4, lower is better): 5 to 8 ridge 0.404, downstream mlp 0.250;
9 to 14 ridge 0.515, downstream mlp 0.360.

Reading:
- Training the stitch on the donor's own LM loss removes most of the
  penalty for same-tokenizer pairs (0.40 to 0.65 nats down to 0.03 to 0.09
  at the early cut). MSE regression does nothing beyond ridge.
- Negative WikiText penalties: "host layers 0..k + donor layers l..end"
  predicts held-out text better than the donor itself by 0.1 to 0.2 nats.
  The first sign of a composite beating its source.
- Across tokenizers the downstream objective closes a third to a half of
  the gap; position mismatch remains.

## Run 15 (2026-11-14): carving a model to half its depth fails

Written model = one plain model carved from a bigger one: keep N layers
(first and last always), chosen by block influence (1 - cosine of layer
input and output on WikiText) or uniformly; optionally heal with a LoRA r64
for 1000 steps on WikiText plus MMLU aux plus GSM8K text, merged. Raw rows:
round15_rows.json (the trisanths shard failed).

| Model | Params | MMLU dev | MMLU held-out | MedMCQA | GSM8K raw |
|---|---|---|---|---|---|
| Official Qwen2.5-7B-Instruct | 7.62B | 71.5 | 70.9 | 56.5 | 76.5 |
| 14B carved to 22/48, uniform, LoRA | 7.61B | 33.6 | 35.8 | 30.9 | 12.5 |
| 14B carved to 22/48, influence, LoRA | 7.61B | 25.5 | 25.4 | 32.7 | 5.5 |
| 14B carved, no heal (influence) | 7.61B | 21.9 | 23.3 | 25.8 | 0.5 |
| Official Qwen2.5-14B-Instruct | 14.77B | 78.5 | 77.9 | 63.8 | 68.5 |
| 32B carved to 27/64, influence, LoRA (2 seeds) | 14.72B | 40.3, 41.2 | 39.4, 39.7 | 33.5, 33.1 | 18.5, 16.0 |
| 32B carved, uniform, LoRA | 14.72B | 36.0 | 36.2 | 32.2 | 18.5 |
| 32B carved, no heal | 14.72B | 24.2 to 25.5 | 25.0 to 25.5 | 27 | 0.5 to 4.5 |
| Official Qwen2.5-32B-Instruct | 32.76B | 82.9 | 79.5 | 66.8 | 80.0 |

Reading: removing more than half the layers destroys the model and a short
heal recovers little; ranking by block influence is no better than uniform.
Carving is a size knob for modest cuts only. With run 14 (blocks written
into a 0.5B base did not help when only the seam glue trained), the lesson
is that a component helps only if what is downstream of it can read it.
Next: skills as experts in one MoE network (each layer's MLP becomes a
router over the same layer's MLP from several specialists), and run 14b
(the base learns to read written-in blocks through a host LoRA in the heal).

## Run 14 (2026-11-14): donor blocks written into a 0.5B base

One network: Qwen2.5-0.5B-Instruct base with a donor block grafted at host
layer 7 as a residual branch (h + W_out(block(W_in h) - x)), MLP stitch
trained 800 steps on the donor's own LM loss (with the sink), glue rank 64,
heal 500 steps on MMLU aux text (glue, plus part LoRA r8 where noted).
MMLU dev / held-out, 2 seeds. Raw rows: round14_rows.json.

| Arm | Total params | Dev | Held-out | Stitch penalty after training |
|---|---|---|---|---|
| Base | 0.49B | 48.0 | 45.8 | |
| Base + LoRA r16 | 0.49B | 47.8, 47.3 | 45.1, 45.9 | |
| + Qwen-7B layers 8..20 | 3.56B | 48.0, 47.2 | 45.9, 47.0 | +0.008 |
| + Qwen-7B block, part LoRA | 3.57B | 48.2, 48.6 | 47.1, 46.7 | +0.024 |
| + random Qwen-7B block (control) | 3.56B | 45.7, 50.1 | 45.4, 49.5 | +0.182 |
| + Qwen-14B layers 14..34 | 6.34B | 48.2, 48.6 | 49.4, 48.6 | -0.020 |
| + Qwen-14B block, part LoRA | 6.36B | 49.6, 48.2 | 50.0, 47.4 | -0.020 |

Reading:
- The trained stitch is now essentially lossless (penalty +0.008 and
  -0.02 nats; without the sink it would be 2.9 to 4.5). The donor block
  receives correct input and computes what it does in its own model.
- The base still cannot use it: +3 points from 6B parameters of a model
  that scores 78. The base's later layers cannot read the block's output,
  and the heal trained only the seam glue. Run 14b adds a base LoRA to the
  heal; the MoE design (experts write into the slot their own downstream
  expects) is the alternative.

## Run 14b (2026-11-14): the base learns to read the block (host LoRA)

Same base and blocks as run 14, but the heal also trains LoRA r16 on the
base, plus part LoRA r8 and the glue: 1500 steps, lr 2e-4. Random blocks
use same-length random layer ranges of the same donor. MMLU dev 1000 /
held-out 2000. One trisanths shard failed (workspace disabled), so some
arms have one seed. Raw rows: round14b_rows.json.

| Arm | Params run | Dev | Held-out |
|---|---|---|---|
| Base + LoRA r16, 1500 steps | 0.49B | 48.1, 47.2 | 46.2, 46.8 |
| + Qwen-7B layers 8..20, host LoRA | 3.58B | 48.6, 49.5 | 47.8, 46.4 |
| + random Qwen-7B block, host LoRA | 3.58B | 49.9 | 48.7 |
| + Qwen-14B layers 14..34, host LoRA | 6.37B | 49.7, 46.1 | 52.6, 49.9 |
| + random Qwen-14B block, host LoRA | 6.37B | 51.9 | 52.3 |

Reading:
- The host LoRA lets the base gain from the block (+3 to +6 over LoRA on
  held-out with the 14B block), but the random block gains as much. The
  gain is from added depth and capacity, not from the donor's skill.
- So a donor block written into a much smaller base does not carry the
  donor's skill, even with a lossless stitch and a base LoRA. The base
  cannot reach the representations the block's skill lives in. This path
  stops here for small bases. Same-lineage experts (run 16b) and fusion
  into a same-width base (run 16) are the next tests.

## Run 16b (2026-11-14): specialist MLPs as routed experts (skill MoE)

Base Qwen2.5-1.5B-Instruct. In every layer the MLP becomes a router over
the base MLP and the same layer's MLP from Qwen2.5-Math-1.5B-Instruct and
Qwen2.5-Coder-1.5B-Instruct (wsp/skill_moe.py). The router starts on the
base (bias 16) and trains in a 1000-step heal on mixed math, code, and
chat text. GSM8K 300, MBPP 200, MMLU 1000, MedMCQA 1000. Raw rows:
round16b_rows.json.

| Arm | GSM8K | MBPP | MMLU | MedMCQA | Params per token |
|---|---|---|---|---|---|
| Base | 68.7 | 59.0 | 58.8 | 44.6 | 1.54B |
| Math specialist | 81.7 | 36.5 | 36.9 | 32.1 | 1.54B |
| Coder specialist | 59.7 | 66.0 | 46.2 | 37.9 | 1.54B |
| Base + LoRA r16, same heal | 57.3, 57.7 | 61.5, 59.0 | 59.1, 59.1 | 45.4, 45.0 | 1.56B |
| TIES merge | 0.3 | 0.0 | 24.7 | 26.0 | 1.54B |
| Skill MoE, soft mix | 51.3, 71.0 | 60.5, 58.0 | 58.9, 58.4 | 44.3, 44.5 | 3.86B |
| Skill MoE, top-1 | 64.0, 72.3 | 59.0, 59.5 | 58.7, 58.5 | 44.1, 44.6 | 1.54B |
| Skill MoE, top-2 | 60.0, 66.3 | 57.0, 59.0 | 58.4, 59.0 | 44.9, 44.7 | 2.70B |
| Skill MoE, top-2 + LoRA | 58.0, 56.3 | 59.0, 60.5 | 58.5, 58.0 | 46.0, 46.1 | 2.70B |

Reading:
- No arm beats the base on GSM8K (68.7); the top-1 MoE averages 68.2. The
  Math specialist's 81.7 and the Coder's 66.0 did not carry over.
- The router barely moved: its base bias ended near 16, and on GSM8K
  prompts 92% of routing slots went to the base MLP. The MoE is close to
  the base because it mostly is the base.
- The LoRA heal hurts GSM8K by 11 points (57.5 vs 68.7). The heal data
  mix damages a skill the base already has; arms with a LoRA inherit it.
- GSM8K seed spread is large (64.0 vs 72.3, 51.3 vs 71.0), so one seed
  proves nothing here.
- The TIES merge collapsed; treat it as a broken baseline until checked
  (Math has rope_theta 10000, the others 1e6).
- Next (run 16c): forced-routing cells with no training (all Math MLPs,
  all Coder MLPs, whole Math layers) to see whether an MLP-only expert
  carries the specialist's skill at all, and a router started from a
  ridge fit on domain labels instead of the base bias.
