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
