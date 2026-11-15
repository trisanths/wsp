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
