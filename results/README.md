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
