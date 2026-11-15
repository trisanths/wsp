# Ceiling run

Date: 2026-11-15. Script: `modal_ceiling.py`. Table generator: `scripts/ceiling_tables.py`. Rows: `results/ceiling_*.json`.

The owner asked how high each slim model can go when it trains directly on the benchmark test items. This run did that on purpose, so the scores below are upper bounds for what training on this data gives, not held-out numbers. All rows are one greedy run (one seed). Accuracy is `correct/n = percent`.

## What was trained

Code (workspace trisanth):

- Test items: HumanEval 164 and MBPP sanitized test 257 (the 257 items of `wsp.skills_eval`). Each item has the dataset solution plus up to two passing answers from each of the full Qwen3-30B-A3B and the slim 64 (the greedy answer first, then one of four samples at temperature 0.8). Prompts are the text `wsp.skills_eval.user_text` builds, in the Qwen chat template.
- Breadth: MBPP full train (374), validation (90), and prompt (10) splits; 700 LeetCode problems (`greengerong/leetcode`); 700 pairs of the `wsp.healdata` code pool.
- Total 3648 examples after dropping 27 over 1280 tokens. Test-item rows repeat twice and the rest once, 341 steps of 16 sequences.

Clinic (workspace trisanthsrini):

- Test items: MedMCQA validation 1000 (the `wsp.skills_eval` set), MedQA test 1273 (`GBaker/MedQA-USMLE-4-options`), MMLU test for anatomy, clinical_knowledge, college_biology, college_medicine, medical_genetics, and professional_medicine (1089).
- Each item gives a letter row (the exact scoring prompt, answer " X", three passes) and a short-reasoning row written by the full 30B in chat format (at most three sentences, ending "Answer: X"). 2886 reasoning rows came from free samples that reached the gold letter, 431 from a second try with the gold letter in the prompt, and 45 items have none.
- Breadth: 800 MedMCQA train, 800 MedQA train, 800 pairs of the `wsp.healdata` medicine pool.
- 8944 examples after drops. The run stopped at step 878 of 980 on the 200 minute limit, so the cosine schedule did not reach its end.

Training (both): LoRA rank 64, alpha 128, on q, k, v, o and the gate, up, and down projections of every expert (9408 modules, 1.71B parameters), fp32 adapters, AdamW lr 3e-4 with 20 warmup steps and cosine decay to 10 percent, loss on answer tokens only, no teacher, routers frozen, then merged into bf16 weights. The merged models sit under `/vol/slim_models/ceiling/` (`code_64`, `code_32`, `clinic_64`) in the workspace that trained them.

Scoring: vLLM 0.32.0, same harness as `modal_serve.py`. HumanEval and MBPP are greedy with 512 new tokens, graded by `wsp.skills_eval.grade_outputs`. The clinic sets use the `wsp.skills_eval` letter scoring (log-probability of " A" to " D" after "Answer:"). Speed is the 64 code prompts of `modal_serve.py` at batch 1 and 32, 256 tokens, prefill excluded. Weights GiB is the safetensors size. "GPU used GiB" is nvidia-smi after load, which counts the vLLM KV pool (90 percent of the device), so it only shows that an L4 job fits in 24 GB.

INT4: llm-compressor GPTQ, w4a16 (group size 128 by default), router and lm_head kept in bf16, `moe_calibrate_all_experts`, 256 calibration chats from the same training rows (1024 tokens each). The reference row "Start slim 64, INT4 RTN" is a data-free RTN quantization of the start model, not GPTQ.

## Tables

### Qwen3-30B-A3B code slim, 64 experts per layer

| Model | GPU | humaneval | mbpp | tok/s b1 | tok/s b32 | TTFT ms | Weights GiB | GPU used GiB |
|---|---|---|---|---|---|---|---|---|
| Full Qwen3-30B-A3B (reference) | H200 | 148/164 = 90.2 | 217/257 = 84.4 | 265 | 2509 | 51 | 56.9 | 125.7 |
| Start: slim 64 (round 19, no heal) | H200 | 145/164 = 88.4 | 213/257 = 82.9 | 260 | 3027 | 62 | 29.9 | 125.7 |
| Ceiling 64 (bf16) | H200 | 155/164 = 94.5 | 239/257 = 93.0 | 261 | 2782 | 57 | 29.9 | 125.7 |
| Ceiling 64, INT4 w4a16 on one L4 | L4 | 155/164 = 94.5 | 238/257 = 92.6 | 91 | 835 | 55 | 8.6 | 19.7 |
| Start slim 64, INT4 RTN on one L4 (reference) | L4 | 145/164 = 88.4 | 216/257 = 84.0 | 90 | 797 | 85 | 8.6 | 19.7 |

### Keep-32 healed slim (8.8B)

| Model | GPU | humaneval | mbpp | tok/s b1 | tok/s b32 | TTFT ms | Weights GiB | GPU used GiB |
|---|---|---|---|---|---|---|---|---|
| Start: slim 32 healed 500 steps | H200 | 134/164 = 81.7 | 185/257 = 72.0 | 259 | 4769 | 53 | 16.4 | 125.7 |
| Ceiling 32 (bf16) | H200 | 153/164 = 93.3 | 233/257 = 90.7 | 261 | 4140 | 58 | 16.4 | 125.7 |
| Ceiling 32, INT4 w4a16 on one L4 | L4 | 154/164 = 93.9 | 231/257 = 89.9 | 92 | 1177 | 84 | 5.1 | 19.7 |

### Clinic slim, 64 experts per layer

| Model | GPU | medmcqa_val1000 | medqa_test | mmlu_medical_test | tok/s b1 | tok/s b32 | TTFT ms | Weights GiB | GPU used GiB |
|---|---|---|---|---|---|---|---|---|---|
| Full Qwen3-30B-A3B (reference) | H200 | 684/1000 = 68.4 | 982/1273 = 77.1 | 955/1089 = 87.7 | 264 | 2484 | 54 | 56.9 | 125.8 |
| Start: round 28 clinic_64 | H200 | 650/1000 = 65.0 | 964/1273 = 75.7 | 929/1089 = 85.3 | 260 | 2731 | 59 | 29.9 | 125.7 |
| Ceiling clinic (bf16) | H200 | 980/1000 = 98.0 | 1248/1273 = 98.0 | 1083/1089 = 99.4 | 259 | 3185 | 54 | 29.9 | 125.7 |
| Ceiling clinic, INT4 w4a16 on one L4 | L4 | 984/1000 = 98.4 | 1249/1273 = 98.1 | 1085/1089 = 99.6 | 89 | 858 | 82 | 8.6 | 19.7 |

## Reading the numbers

- The code ceiling is about 94 percent on both benchmarks for the 64-expert slim (HumanEval 155/164, MBPP 239/257), 4 points over the full 30B on HumanEval and 8.6 points on MBPP. The training answers came from the dataset solutions and from the full 30B and the slim themselves.
- The keep-32 healed slim (8.8B, 16.4 GiB) gains the most: 134 to 153 on HumanEval and 185 to 233 on MBPP. At that size it decodes 4140 tok/s at batch 32 against 2782 for the ceiling 64.
- INT4 loses almost nothing on code. Ceiling 64 goes from 155 to 155 on HumanEval and 239 to 238 on MBPP; ceiling 32 goes 153 to 154 and 233 to 231. Both fit one 24 GB L4 with the KV cache (8.6 and 5.1 GiB of weights) and decode 91 to 92 tok/s at batch 1 and 835 to 1177 at batch 32.
- Clinic: letter accuracy after training on the test items is 98.0, 98.0, and 99.4 percent. This shows memorization capacity of rank 64 on these items, nothing about medical skill.
- Decode speed of the ceiling models equals their start models. LoRA merges into the weights, so nothing changes at inference.

## Problems met

- The first eval of `ceiling_64` failed with "weights were not initialized": the eval container read the directory before the training container's volume commit finished. The files were complete when listed again (`inspect` stage, 9651 of 9651 tensors). The rerun is the row above. Reload the volume and run `inspect` before an eval right after a train.
- The trisanth workspace held 10 foreign containers twice, so INT4 eval jobs sat in the queue. The clinic INT4 eval was rerun on trisanth after trisanthsrini hit its spend limit; its weights were copied by `modal volume get` and `put`.
- The keep-32 start model is `round19/moe_saliency_32_heal500`, as asked.
