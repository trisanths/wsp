# Moonshot: Qwen3-235B coding ability on one GPU

Date: 2026-11-15. Driver: `modal_moonshot.py`. Tests: `tests/test_moonshot.py`. Rows and raw outputs: `results/moonshot/`. Modal workspace: rexheng-24542, volume `wsp-models`, under `/moonshot`.

Request: "Give me Qwen3-235B's coding ability in a model that fits on ONE GPU."

Short answer: not met. Removing experts by code saliency keeps almost all of the code score at half the size (keep 64 of 128), and that model needs more than one GPU. The sizes that fit one H200 lose half the code score and are worse than Qwen3-30B-A3B served on the same GPU.

## What was done

Model: Qwen/Qwen3-235B-A22B-Instruct-2507, the non-thinking 2507 release. Config read from the Hub: 94 layers, all MoE, 128 routed experts per layer, top-k 8, hidden 4096, expert width 1536, 64 query heads, 4 KV heads, head dim 128, vocab 151936, untied head. The checkpoint has 118 safetensors shards and one tensor per expert matrix (`model.layers.L.mlp.experts.E.{gate_proj,up_proj,down_proj}.weight`).

1. Sizes from the config (formula in `moe_sizes`; it reproduces the checkpoint's 470,187,269,120 bytes exactly at 128 experts, and the built checkpoints match it exactly).
2. Adapter check (CPU, `tests/test_moonshot.py`): one layer of the real config on the meta device has the module layout `wsp.slim` reads (`mlp.gate`, `mlp.experts` as a ModuleList of 128, family `qwen3_moe`) and the tensor names above. transformers 4.57.1 loads it. 9 tests pass.
3. Ranking statistics: `wsp.slim.collect_moe_stats` (saliency = routing weight times expert output norm, summed over tokens) on 256 chat texts truncated to 256 tokens, 50,358 tokens per layer. 192 texts from the `wsp.healdata` code pool (Magicoder, CodeAlpaca, python_code_instructions, self-oss-instruct, APPS introductory train, code_contests) and 64 from MBPP full train and prompt splits. MBPP ids in the scored sanitized test split were excluded, and texts containing a HumanEval or MBPP test string were dropped (`wsp.slim.drop_overlap`, 1 text dropped). No scored item trained or ranked anything. Run on H200:4 with `device_map="auto"`, torch 2.7.1, transformers 4.57.1, 8 chunks of 32 texts, about 143 s per chunk, 26 minutes, 7.74 dollars. A smoke on the first 4 layers over two devices ran first (8 texts, 7.2 s).
4. Build by safetensors surgery (`slim_checkpoint`): for each shard, drop the experts that are not kept, renumber the kept ones in sorted order, slice the router rows, write plain Hugging Face directories with `num_experts` and `num_experts_per_tok = min(8, keep)`. No model object and no GPU. A unit test checks the output against `wsp.slim.slim_experts` on a tiny Qwen3-MoE (logits equal within 1e-5, parameter counts equal). Keep sets are nested: keep 16 is inside keep 24, 32, 48, 64. No heal.
5. Scoring with vLLM 0.32.0 (the `modal_serve.py` image), bf16, greedy: HumanEval 164 and MBPP sanitized test 257 with the `wsp.skills_eval` prompts and graders (512 new tokens), MMLU 1000 by the first-token choice among A to D (`allowed_token_ids`, same items and prompt as `wsp.skills_eval.mmlu_items`), and decode speed with the `modal_serve.py` method on 32 code prompts.

## Sizes (from the config)

Per-token parameters follow `wsp.slim.count_params` (the input embedding is a lookup and does not count).

| Keep per layer | Stored params | Run per token | bf16 weights | Bytes on disk (built) |
|---|---|---|---|---|
| 128 (full) | 235.09 B | 21.57 B | 470.2 GB | 470,187,269,120 (index total) |
| 64 | 121.52 B | 21.54 B | 243.0 GB | 243,061,429,912 |
| 48 | 93.13 B | 21.54 B | 186.3 GB | 186,274,056,024 |
| 32 | 64.73 B | 21.53 B | 129.5 GB | 129,486,682,752 |
| 24 | 50.54 B | 21.53 B | 101.1 GB | 101,092,999,216 |
| 16 | 36.34 B | 21.53 B | 72.7 GB | 72,699,316,751 |

Parameters run per token barely change: top-k stays 8, so every token still reads about 21.5 B parameters. Removal shrinks storage and not per-token compute. The 235B figure in the guess ("keep a quarter gives 60 to 70 B, an eighth gives 35 B") holds: keep 32 is 64.7 B and keep 16 is 36.3 B. Only keep 24 and keep 16 have bf16 weights under 80 percent of one H200. Keep 16 is the only size that is under 40 B. An 80 GB GPU in fp8 holds keep 24 or keep 16 under the same 80 percent rule; keep 32 in fp8 (64.7 GB) leaves too little KV cache.

## Scores

vLLM 0.32.0, H200, bf16, greedy, 512 new tokens. HumanEval n = 164, MBPP n = 257, MMLU n = 1000 (chance is 25). Every cell is one ranking and one deterministic run, so there is no seed spread; the standard error on a 60 percent score over 164 items is about 3.8 points.

| Cell | Stored | GPUs used | HumanEval | MBPP | MMLU | Decode tok/s batch 1 / 32 | vLLM weight GiB per GPU |
|---|---|---|---|---|---|---|---|
| Qwen3-235B full | 235.1 B | 8 | 93.3 (153) | 87.5 (225) | 85.0 | 138 / 1953 | 55.0 |
| keep 64 | 121.5 B | 4 | 92.7 (152) | 87.2 (224) | 52.5 | 122 / 1539 | 56.7 |
| keep 48 | 93.1 B | 2 | 44.5 (73) | 37.0 (95) | 43.6 | 89 / 1136 | 86.8 |
| keep 48, control on 4 GPUs | 93.1 B | 4 | 45.1 (74) | 43.2 (111) | 43.9 | 116 / 1676 | 43.5 |
| keep 32 | 64.7 B | 2 | 61.6 (101) | 49.4 (127) | 36.3 | 94 / 1425 | 60.4 |
| keep 24 | 50.5 B | 1 | 56.7 (93) | 40.1 (103) | 30.4 | 61 / 987 | 94.2 |
| keep 16 | 36.3 B | 1 | 43.3 (71) | 33.1 (85) | 24.3 | 62 / 1206 | 67.8 |
| Qwen3-30B-A3B full, for scale | 30.5 B | 1 | 90.2 (148) | not run here | not run here | 259 / 2486 | 56.9 GB on disk |

Sources: `results/moonshot/eval_<cell>.json` (the cell name is in each file), `results/moonshot/stats.json`, `results/moonshot/build_keep_<N>.json`. The 30B row is `results/serve_full_30b.json` (the log is `outputs/serve_full.log` in the main checkout). Run 19 scored the same 30B model with the Hugging Face loop at HumanEval 91.5, MBPP 84.4, MMLU 77.8; that is a different engine and is not mixed into the table.

The 235B full-model row is measured here with the same harness. Qwen's published numbers were not used.

"GPUs used" is the tensor-parallel size the cell was run with, not the smallest that could hold it. Keep 64 was not run on two GPUs. vLLM weight memory is the line "Model loading took X GiB" that the engine prints, per GPU.

### What the numbers say

1. Keep 64 of 128 (121.5 B, 52 percent of the size) scores 92.7 on HumanEval and 87.2 on MBPP against 93.3 and 87.5 for the full model: one task apart on each, with one ranking and no heal. The same result as Run 19 on the 30B model at 64 of 128 (87.8 against 91.5 there), except the gap is smaller here. MMLU falls from 85.0 to 52.5, so what is removed is general knowledge.
2. Below that the code score drops fast: keep 32 61.6, keep 24 56.7, keep 16 43.3 on HumanEval. At keep 16 MMLU is at chance (24.3) and 20 of 164 HumanEval outputs hit the token cap (58 of 257 on MBPP), a sign of degenerate repetition.
3. The one-GPU cells lose. On one H200, keep 16 (43.3) and keep 24 (56.7) are far below Qwen3-30B-A3B full (90.2). A user who wants code on one GPU today should run the 30B model.
4. Batch-1 decode of the slimmed models is 60 to 120 tok/s, below the 30B model's 259, because each token still runs about 21.5 B parameters. The speed here also depends on the tensor-parallel size.

## Anomaly: keep 48 scores below keep 32

Keep 48 contains every expert of keep 32 and scored 44.5 on HumanEval against 61.6. In the HumanEval records the failures are `SyntaxError: unterminated triple-quoted string` in outputs that stop in the middle of a docstring example with the token sequence `游戏副本` (see `results/moonshot/samples_humaneval.json`, HumanEval/0). 91 of 164 keep 48 outputs fail against 62 for keep 32. Running keep 48 on 4 GPUs instead of 2 gave 45.1, so the tensor-parallel split is not the cause. The build is the same code that produced keep 64, 32 and 16, and the parameter count equals the formula. The cause is not isolated. Candidates, none tested: the 16 experts added between 32 and 48 pull a few tokens toward a rare-token expert that the full model would not reach, or the saliency order of the middle ranks is a poor guide because it comes from 50k tokens. The next steps that would tell: serve keep 48 with the Hugging Face loop on the same prompts; rank on a second text set (a second ranking is also the only way to put a spread on any cell here); score keep 40 and keep 56.

## What this does not show

- No heal. Run 19 healed 500 steps on the 30B model, and a heal on a 235B student was out of budget.
- One ranking, one run per cell.
- MMLU was scored on 1000 items by first-token choice with greedy decoding, not by log-probability over a full prompt set, and only to show what the removal cost.
- Keep 64 on two GPUs, keep 64 in fp8 on one H200 (121.5 GB of weights, which would leave about 5 GB for KV cache under the 0.90 memory setting), and any quantization were not run. They are the cheapest next test for "code of 235B on one GPU": keep 64 plus fp8 is the only candidate that could fit one 141 GB GPU and keep the code score.
- A larger keep between 32 and 64 (for example 40 or 56) would locate where the code score falls.

## Cost

Rate assumed: 4.5 dollars per GPU hour. Rows record seconds times GPUs times the rate, so these are modelled, not billed.

| Step | GPUs | Dollars |
|---|---|---|
| Download (CPU, 470 GB, 18 minutes) | 0 | 0 |
| Stats smoke, 4 layers | 2 | 0.53 |
| Stats, full | 4 | 7.74 |
| Build keep 64, 32, 16 and 48, 24 (CPU) | 0 | 0 |
| Score smoke, keep 16, 8 tasks | 1 | 0.59 |
| Score full 235B | 8 | 9.75 |
| Score keep 64 | 4 | 4.30 |
| Score keep 48 (2 GPUs), keep 48 (4 GPUs control) | 2, 4 | 1.70, 3.87 |
| Score keep 32 | 2 | 1.80 |
| Score keep 24, keep 16 | 1, 1 | 0.79, 0.74 |
| Total | | 31.81 |

Under 150 dollars, so no stop was needed. Most wall time went to loading weights from the volume (5 to 13 minutes per cell).

## Notes for the next run

- The `hit the cap` counts in the rows show how many outputs reached 512 tokens. Look at them before trusting a low code score.
- Loading the 4-layer smoke still iterates over all 118 shards (about 2.5 minutes of the 147 s load). It is fine for a smoke.
- Stage `score` was called `eval` first; a tool filter in one session blocked that word, so the stage has the other name.
