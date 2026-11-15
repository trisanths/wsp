# Pilot mining: what was found before Rex stopped it (2026-11-14)

Rex stopped part mining so part 3 can focus on the RL environment and swarm testing. This note keeps what the one finished donor produced, so Person A or Person B can pick it up. These parts are PILOT parts: kept out of `catalog.jsonl` and not scored by any eval yet.

## What ran
- **Tool:** Person B's extractor (`wsp_eval.extraction`), driven by `scripts/mine_pilot.py` on Modal, in workspace `rexheng`.
- **Method:** activation contrast. Each unit gets a Cohen's d: its activation on 256 GSM8K heal-split texts (GSM8K train minus the 200-item search split) against 256 WikiText-2 texts. For MLP clusters the activation is |neuron activation|; for heads it is the head output norm.
- **Selection:** the top units of the 4 best layers per kind. Each part comes with three random same-size controls from the same layer.
- **Finished donor:** `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B` (MIT; 28 layers, hidden 1536). It produced 8 parts and 24 controls.
- **Not run (stopped):** `Qwen/Qwen2.5-Math-1.5B-Instruct` (mlp_cluster, attn_heads) and `allenai/OLMoE-1B-7B-0125-Instruct` (expert; 16 layers, hidden 2048, 64 experts, top 8). To resume one donor, run `.venv/Scripts/python scripts/mine_pilot.py mine --donors qmi15` or `--donors olmoe`; the other donors' records stay.

## The 8 parts (records in `data/pilot_catalog_mined.jsonl`)

Units files are on the Modal volume `wsp-swarm-models` (workspace `rexheng`) under `units/pilot/`: 32 files, which is 8 parts plus 24 controls.

| id | kind | donor layer | units | mean Cohen's d | random same-size d | layer rank |
|---|---|---|---|---|---|---|
| pilot_r1q15_l23_heads2 | attn_heads | 23 | 2 heads | 1.064 | 0.508 | 1 |
| pilot_r1q15_l22_heads2 | attn_heads | 22 | 2 heads | 0.987 | 0.380 | 2 |
| pilot_r1q15_l21_heads2 | attn_heads | 21 | 2 heads | 0.864 | -0.260 | 3 |
| pilot_r1q15_l20_heads2 | attn_heads | 20 | 2 heads | 0.844 | 0.356 | 4 |
| pilot_r1q15_l26_mlp512 | mlp_cluster | 26 | 512 | 0.553 | 0.077 | 1 |
| pilot_r1q15_l21_mlp512 | mlp_cluster | 21 | 512 | 0.552 | 0.088 | 2 |
| pilot_r1q15_l27_mlp512 | mlp_cluster | 27 | 512 | 0.542 | -0.061 | 3 |
| pilot_r1q15_l20_mlp512 | mlp_cluster | 20 | 512 | 0.518 | 0.110 | 4 |

## Reading
- **Late layers:** the GSM8K-specific units of this donor sit in late layers (20 to 27 of 28). By the default placement rule, round(l / 28 x 16), they land at host layers 11 to 15. Round 2 found grafts at host layers 10 and 14 scored about 22 to 26% against about 38 to 40% at layer 6. So expect these parts to need an explicit early `at`, or to lose to placement effects.
- **Selection versus controls:** the selected units respond more to math text than random same-size units from the same layer. Heads lead by 0.48 to 1.12 d, clusters by 0.44 to 0.60. That is a selection fact about activations, not evidence that the parts carry transferable skill. Round 2's contrast-picked clusters from Qwen2.5-Math-1.5B did not beat the random control on GSM8K.
- **Fit:** no fit_r2 or GSM8K score exists for these parts yet; the probe step was not reached.
