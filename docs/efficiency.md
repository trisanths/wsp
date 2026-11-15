# Efficiency of the Qwen3-30B-A3B demo model (modal_efficiency.py)

Date of every number: 2026-11-15. Engine: vLLM 0.32.0 (image of modal_serve.py). One run per config and greedy decoding, so each cell is one sample, not an average over seeds. Modal workspace hanavm, volume wsp-efficiency. Rows are in results/efficiency_<name>.json (acceptance counters and sample outputs included). The raw generations of every graded item are on the volume under results/efficiency/<name>/raw.

## What is measured

For each config, in a fresh vLLM process:

- HumanEval (164) and MBPP (257, the sanitized test split of wsp.skills_eval) pass@1, greedy, 512 new tokens, with the prompts and graders of wsp.skills_eval. Nothing trained, ranked, or selected on these items. The calibration text for INT4 came from the wsp.healdata "code" pool (default reported set, which drops every HumanEval or MBPP derived source). 6 of 1024 texts were dropped for sharing a 13-gram with a HumanEval or MBPP test item (modal_efficiency.drop_13gram, tested in tests/test_efficiency.py). 512 texts were used.
- tok/s b1, b8, b32: decode tokens per second at batch 1, 8, 32 on 64 fixed code prompts (the first 64 users of the code pool), 256 new tokens each, ignore_eos. Decode time is the time of the 256-token run minus the time of a 1-token run of the same prompts.
- he_b1 tok/s: tokens per second of 24 real HumanEval requests sent one at a time with natural stop, prefill included. This is the batch-1 number that matters for speculative decoding, because ignore_eos on open prompts understates prompt lookup.
- prefill tok/s: prompt tokens per second for 64 prompts in one call. Rows taken before the warmup fix (all rows of the full and slim64 sweeps, and the first INT4 rows) timed the first pass while MoE kernels were still compiling, so the same config reads anywhere from 7k to 58k tok/s. Do not use prefill from those rows.
- TTFT: median wall time of a 1-token request at batch 1.
- disk GiB: bytes of the .safetensors files. weights GiB: "Model loading took" from the vLLM log (it includes the EAGLE head or draft model for speculative configs). KV tokens: "GPU KV cache size" at gpu_memory_utilization 0.90 and max_model_len 4096.
- accept: for speculative configs, accepted draft tokens over proposed draft tokens, and the mean tokens per verification step (1 + accepted/steps), on the 24 batch-1 HumanEval requests.

Noise: HumanEval has 164 items, so one item is 0.6 points and the binomial standard deviation near 90% is about 2.3 points. Configs that should give identical greedy text read between 90.2 and 92.1 on the full model (base 90.9, ngram 90.2, kvfp8 92.1). Treat differences under 2 points on either benchmark as noise. Batch size and kernel choice change floating point sums, and greedy decoding turns small changes into different programs.

## Models

- full: Qwen/Qwen3-30B-A3B-Instruct-2507, 128 experts per layer, 8 active. 30.5B stored, 3.04B run per token.
- slim64: the keep-64 model of Run 19 (64 experts per layer by code saliency, no heal, 16.0B stored, 3.04B run per token). It was rebuilt on the hanavm volume from the Hub weights and the kept-expert lists of the trisanth copy (results/efficiency_slim64_manifest.json, `--build-slim-model`). The rebuilt model matches the original: 29.92 GiB in vLLM like the trisanth copy, and HumanEval 145 of 164, the same count modal_serve.py gave (outputs/serve_full.log).
- full_w4a16 and slim_w4a16: GPTQ INT4 weights (llm-compressor W4A16, group size 128, router and lm_head stay bf16), 512 calibration samples of up to 2048 tokens, all experts active during calibration. 15.55 GiB (full) and 8.57 GiB (slim) on disk. Quantization took 2812 s and 1444 s on one H200.
- slim_fp8: data-free FP8_DYNAMIC checkpoint of slim64, 15.54 GiB on disk. Written but not scored; the online `fp8` config below holds the same weights in memory.

## Results

The first row of each table is the starting model of that table (config "base"). The start of the whole study is row base of the first table: full bf16 on an H200, HumanEval 90.9, MBPP 85.2, 259 tok/s at batch 1.

Config names are "+" joins. fp8 is vLLM online dynamic FP8 weights. kvfp8 is the FP8 KV cache. topk6 and topk4 set num_experts_per_tok to 6 and 4 through hf_overrides (no file edits). ngram is prompt lookup (4 draft tokens, lookup length 2 to 4) and ngram8 uses 8 draft tokens. draft06 and draft17 are Qwen3-0.6B and Qwen3-1.7B draft models with 4 draft tokens. eagle3 is RedHatAI/Qwen3-30B-A3B-Instruct-2507-speculator.eagle3 and eagle3b is lmsys/SGLang-EAGLE3-Qwen3-30B-A3B-Instruct-2507-SpecForge-Nex, both with 3 draft tokens unless a digit follows (eagle3b2, eagle3b5, eagle3b7). marlin forces the Marlin MoE kernel. eager turns CUDA graphs off. nochunk turns chunked prefill off. seqs32 and seqs256 set max_num_seqs (every other row uses 64).

### hub:Qwen/Qwen3-30B-A3B-Instruct-2507 (results/efficiency_full.json)

| config | GPU | HumanEval | MBPP | tok/s b1 | tok/s b8 | tok/s b32 | he_b1 tok/s | prefill tok/s | TTFT ms | disk GiB | weights GiB | KV tokens | accept (rate / len) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| base | H200 | 90.9 | 85.2 | 259 | 918 | 2507 | 234 | 42019 | 66.0 | 56.9 | 56.9 | 727920 | - |
| fp8 | H200 | 92.1 | 85.2 | 241 | 1247 | 3441 | 226 | 11710 | 62.6 | 56.9 | 29.1 | 1031920 | - |
| kvfp8 | H200 | 92.1 | 85.6 | 252 | 974 | 2473 | 228 | 7739 | 55.1 | 56.9 | 56.9 | 1457216 | - |
| fp8+eagle3b5 | L40S | 92.1 | 86.0 | 204 | 597 | 1642 | 269 | 23706 | 53.1 | 56.9 | 29.4 | 82208 | 78% / 4.91 |
| eager | H200 | 91.5 | 83.7 | 17 | 129 | 524 | 16 | 7615 | 69.0 | 56.9 | 56.9 | 731312 | - |
| fp8+kvfp8 | H200 | 90.9 | 86.4 | 235 | 1153 | 3458 | 221 | 50228 | 65.3 | 56.9 | 29.1 | 2064752 | - |
| nochunk | H200 | 90.9 | 86.0 | 257 | 1043 | 2475 | 233 | 7191 | 60.0 | 56.9 | 56.9 | 727920 | - |
| seqs256 | H200 | 90.9 | 84.4 | 247 | 964 | 2504 | 241 | 58571 | 21.6 | 56.9 | 56.9 | 721712 | - |
| topk6 | H200 | 90.9 | 84.8 | 272 | 1020 | 2744 | 242 | 45800 | 67.0 | 56.9 | 56.9 | 730656 | - |
| fp8 | L40S | 90.9 | 84.0 | 129 | 401 | 893 | 123 | 23297 | 64.3 | 56.9 | 29.1 | 101680 | - |
| draft06 | H200 | 90.2 | 84.0 | 53 | 334 | 1261 | 68 | 34044 | 73.7 | 56.9 | 58.1 | 327328 | 81% / 4.24 |
| draft17 | H200 | 90.2 | 84.4 | 77 | 498 | 1756 | 95 | 35558 | 55.2 | 56.9 | 60.1 | 316112 | 83% / 4.32 |
| eagle3 | H200 | 90.2 | 84.0 | 449 | 2091 | 4486 | 495 | 47405 | 23.4 | 56.9 | 57.4 | 690848 | 77% / 3.32 |
| eagle3b | H200 | 90.2 | 84.4 | 467 | 2058 | 4679 | 539 | 8087 | 23.4 | 56.9 | 57.3 | 689040 | 86% / 3.58 |
| ngram8 | H200 | 90.2 | 84.0 | 155 | 635 | 1604 | 264 | 55553 | 21.4 | 56.9 | 56.9 | 717344 | 15% / 2.22 |
| ngram | H200 | 90.2 | 84.0 | 157 | 563 | 1515 | 261 | 47828 | 21.9 | 56.9 | 56.9 | 722016 | 24% / 1.96 |
| seqs32 | H200 | 90.2 | 84.4 | 265 | 983 | 2505 | 227 | 7378 | 66.2 | 56.9 | 56.9 | 729472 | - |
| topk4 | H200 | 87.2 | 84.0 | 287 | 1215 | 5868 | 254 | 7761 | 61.6 | 56.9 | 56.9 | 733376 | - |

### vol:slim_models/round19/moe_saliency_64 (results/efficiency_slim64.json)

| config | GPU | HumanEval | MBPP | tok/s b1 | tok/s b8 | tok/s b32 | he_b1 tok/s | prefill tok/s | TTFT ms | disk GiB | weights GiB | KV tokens | accept (rate / len) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| base | H200 | 88.4 | 82.5 | 229 | 1029 | 3030 | 232 | 10695 | 68.5 | 29.9 | 29.9 | 1022976 | - |
| eagle3b+kvfp8 | H200 | 90.2 | 82.5 | 432 | 1880 | 6005 | 513 | 55836 | 23.1 | 29.9 | 30.3 | 1958832 | 86% / 3.59 |
| kvfp8 | H200 | 90.2 | 81.7 | 241 | 996 | 2958 | 230 | 10962 | 51.5 | 29.9 | 29.9 | 2047344 | - |
| topk6 | H200 | 90.2 | 82.5 | 263 | 1117 | 3134 | 184 | 54886 | 56.4 | 29.9 | 29.9 | 1025712 | - |
| fp8 | H200 | 89.6 | 81.7 | 244 | 1235 | 3915 | 230 | 55368 | 60.7 | 29.9 | 15.6 | 1179456 | - |
| eagle3b | H200 | 89.0 | 82.5 | 443 | 1982 | 5861 | 520 | 12700 | 22.0 | 29.9 | 30.3 | 978032 | 87% / 3.60 |
| fp8+kvfp8 | H200 | 89.0 | 82.5 | 234 | 1149 | 3795 | 224 | 48785 | 62.7 | 29.9 | 15.6 | 2359872 | - |
| ngram | H200 | 89.0 | 82.1 | 151 | 697 | 2233 | 235 | 57239 | 17.4 | 29.9 | 29.9 | 1017088 | 24% / 1.96 |
| base | L40S | 88.4 | 83.7 | 97 | 267 | 726 | 95 | 21614 | 52.9 | 29.9 | 29.9 | 95696 | - |
| eagle3b+topk6 | H200 | 88.4 | 81.7 | 475 | 1966 | 6033 | 553 | 10600 | 21.7 | 29.9 | 30.3 | 980720 | 84% / 3.53 |
| eagle3b+fp8+kvfp8 | H200 | 87.8 | 83.3 | 477 | 2477 | 7239 | 556 | 68989 | 22.4 | 29.9 | 15.9 | 2263808 | 85% / 3.56 |
| eagle3b | L40S | 87.2 | 82.5 | 149 | 457 | 1351 | 179 | 21283 | 59.7 | 29.9 | 30.3 | 77568 | 87% / 3.60 |
| topk4 | H200 | 83.5 | 76.7 | 281 | 1257 | 3405 | 261 | 59402 | 60.9 | 29.9 | 29.9 | 1028448 | - |

### vol:quant/full_w4a16 (results/efficiency_full_w4a16.json)

| config | GPU | HumanEval | MBPP | tok/s b1 | tok/s b8 | tok/s b32 | he_b1 tok/s | prefill tok/s | TTFT ms | disk GiB | weights GiB | KV tokens | accept (rate / len) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| eagle3b+kvfp8 | A10G | 92.1 | 83.7 | 169 | 621 | 1552 | 199 | 8442 | 44.9 | 15.6 | 16.0 | 30304 | 86% / 3.57 |
| eagle3b | A10G | 92.1 | 84.4 | 219 | 665 | 1452 | 249 | 8531 | 49.8 | 15.6 | 16.0 | 14960 | 86% / 3.59 |
| marlin+kvfp8 | H200 | 92.1 | 84.4 | 218 | 1312 | 3719 | 199 | 26986 | 56.5 | 15.6 | 15.7 | 2356720 | - |
| eagle3b5 | A10G | 92.1 | 84.8 | 208 | 653 | 1387 | 268 | 8384 | 46.5 | 15.6 | 16.0 | 14432 | 78% / 4.88 |
| eagle3b7 | A10G | 92.1 | 84.4 | 195 | 602 | 1277 | 277 | 8316 | 58.7 | 15.6 | 16.0 | 14464 | 71% / 5.95 |
| marlin+eagle3b+kvfp8 | H200 | 91.5 | 84.4 | 497 | 2345 | 6033 | 565 | 39624 | 25.9 | 15.6 | 16.0 | 2259168 | 86% / 3.58 |
| marlin+eagle3b+topk6 | H200 | 91.5 | 84.4 | 514 | 2417 | 6344 | 601 | 47991 | 24.0 | 15.6 | 16.0 | 1132432 | 85% / 3.56 |
| marlin+eagle3b5 | H200 | 91.5 | 85.6 | 502 | 2455 | 6444 | 681 | 40820 | 27.0 | 15.6 | 16.0 | 1127632 | 77% / 4.86 |
| marlin+eagle3b2 | H200 | 90.9 | 84.4 | 457 | 2342 | 5855 | 498 | 40008 | 22.6 | 15.6 | 16.0 | 1131728 | 91% / 2.82 |
| marlin+eagle3b | H200 | 90.9 | 84.4 | 502 | 2351 | 6298 | 558 | 39357 | 24.8 | 15.6 | 16.0 | 1129584 | 86% / 3.59 |
| marlin+ngram | H200 | 90.9 | 84.8 | 171 | 933 | 2419 | 281 | 40613 | 21.4 | 15.6 | 15.7 | 1169504 | 25% / 1.99 |
| marlin+topk6 | H200 | 90.9 | 85.2 | 226 | 1447 | 4100 | 213 | 40175 | 55.6 | 15.6 | 15.7 | 1180416 | - |
| marlin | H200 | 90.9 | 84.4 | 221 | 1327 | 3811 | 207 | 35616 | 53.7 | 15.6 | 15.7 | 1177696 | - |
| marlin+eagle3b7 | H200 | 90.9 | 85.2 | 556 | 2269 | 5873 | 778 | 37391 | 22.6 | 15.6 | 16.0 | 1124176 | 71% / 5.94 |
| base | A10G | 90.2 | 84.4 | 133 | 479 | 1107 | 126 | 8895 | 49.5 | 15.6 | 15.7 | 31360 | - |
| marlin+topk4 | H200 | 85.4 | 81.3 | 229 | 1530 | 4469 | 216 | 47159 | 56.6 | 15.6 | 15.7 | 1183152 | - |

### vol:quant/slim_w4a16 (results/efficiency_slim_w4a16.json)

| config | GPU | HumanEval | MBPP | tok/s b1 | tok/s b8 | tok/s b32 | he_b1 tok/s | prefill tok/s | TTFT ms | disk GiB | weights GiB | KV tokens | accept (rate / len) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| marlin | H200 | 89.0 | 81.3 | 217 | 1357 | 4208 | 208 | 39119 | 43.8 | 8.6 | 8.7 | 1253952 | - |
| marlin+kvfp8 | H200 | 88.4 | 82.1 | 186 | 1320 | 4097 | 201 | 30802 | 46.9 | 8.6 | 8.7 | 2509296 | - |
| marlin+ngram | H200 | 88.4 | 81.3 | 165 | 915 | 2518 | 256 | 42324 | 16.6 | 8.6 | 8.7 | 1245616 | 23% / 1.93 |
| marlin+eagle3b7 | H200 | 88.4 | 80.9 | 549 | 2474 | 6629 | 806 | 43363 | 18.2 | 8.6 | 9.0 | 1198768 | 68% / 5.79 |
| marlin+eagle3b+kvfp8 | H200 | 87.8 | 83.7 | 482 | 2520 | 6657 | 564 | 42471 | 19.3 | 8.6 | 9.0 | 2408336 | 86% / 3.58 |
| base | L4 | 87.2 | 82.9 | 89 | 300 | 800 | 86 | 6734 | 81.8 | 8.6 | 8.7 | 107392 | - |
| marlin+eagle3b2 | H200 | 87.2 | 81.3 | 441 | 2382 | 6723 | 493 | 43019 | 16.1 | 8.6 | 9.0 | 1206160 | 90% / 2.79 |
| marlin+eagle3b5 | H200 | 87.2 | 80.2 | 538 | 2510 | 6867 | 717 | 43402 | 17.8 | 8.6 | 9.0 | 1202224 | 75% / 4.76 |
| marlin+eagle3b | H200 | 87.2 | 82.1 | 496 | 2470 | 7027 | 592 | 42194 | 21.7 | 8.6 | 9.0 | 1204160 | 86% / 3.58 |
| marlin+topk6 | H200 | 87.2 | 80.2 | 220 | 1419 | 4272 | 211 | 43640 | 45.9 | 8.6 | 8.7 | 1256704 | - |

Rows with a GPU other than H200 are small-GPU runs (below).

## Findings by axis

Precision (full model, H200)
- FP8 weights halve memory (56.9 to 29.1 GiB) and raise batch-32 decode from 2507 to 3441 tok/s (+37%). Batch 1 gets slower, 259 to 241 tok/s (-7%). HumanEval 92.1 against 90.9, within noise.
- FP8 KV cache doubles KV room (727,920 to 1,457,216 tokens at 0.90) with HumanEval 92.1 and MBPP 85.6, within noise, and no speed change. Together with FP8 weights the KV room is 2.8x (2,064,752 tokens).
- INT4 (GPTQ W4A16, Marlin) is 15.7 GiB in vLLM. HumanEval 90.9 and MBPP 84.4 against 90.9 and 85.2 for bf16. Batch-32 decode is 3811 tok/s (+52%). Batch 1 is 221 tok/s (-15%): on an H200 with 4.8 TB/s the dequantize work costs more than the saved bytes at one token. On slower memory the saved bytes win (see Smallest GPU).

Fewer active experts (full model, H200; the compute per token axis)
- Active parameters per token: 3.04B at 8 experts, 2.59B at 6, 2.14B at 4 (an expert slot is 226M parameters over 48 layers).
- topk6: HumanEval 90.9, MBPP 84.8, no loss measured. Decode 272 tok/s at batch 1 (+5%) and 2744 at batch 32 (+9%).
- topk4: HumanEval 87.2 (-3.7), MBPP 84.0. Batch 32 reaches 5868 tok/s (2.3x) and batch 1 287 tok/s (+11%). On the slim model topk4 falls to HumanEval 83.5 and MBPP 76.7, while topk6 stays at 90.2 and 82.5. A heal of topk4 is for another track.
- On the INT4 full model, topk6 gives 90.9 and 85.2 with 4100 tok/s at batch 32 (+8% over INT4 alone), and topk4 gives 85.4 and 81.3.

Speculative decoding
- EAGLE-3 is the largest single gain. With the lmsys head (eagle3b), 3 draft tokens: 467 tok/s at batch 1 against 259 (1.8x), and 539 against 234 on real HumanEval requests (2.3x). HumanEval 90.2, acceptance 86% of draft tokens, 3.58 tokens per step. Batch 8 goes from 918 to 2058 and batch 32 from 2507 to 4679. The RedHatAI head is a little worse (449 and 495 tok/s, 77% acceptance).
- More draft tokens help real requests. Full INT4 with eagle3b on an H200, real HumanEval requests at batch 1: 3 tokens 558 tok/s, 5 tokens 681, 7 tokens 778. Acceptance per proposed token falls (86%, 77%, 65%) while tokens per step rises (3.59, 4.86, 5.56). Slim INT4 with 7 tokens reaches 806 tok/s. The ignore_eos batch-1 number stays near 500 to 550 for all three lengths.
- Prompt lookup: 24% of draft tokens accepted, 1.96 tokens per step. On real HumanEval requests it gives 261 against 234 tok/s (+11%), and on the ignore_eos prompts it is slower (157 against 259) because those prompts rarely repeat text.
- Draft models are slower than no speculation in vLLM 0.32: Qwen3-0.6B 53 tok/s and Qwen3-1.7B 77 tok/s at batch 1 (acceptance 81% and 83%, 4.2 tokens per step). The draft forward passes cost more than the acceptance returns for a 3B-active target.
- The EAGLE head was trained on the full model. On the slim model acceptance is the same (87%, 3.6 tokens per step), so skill removal did not hurt the head.
- HumanEval stays within noise for every speculative config (90.2 to 91.5 on the full model).

vLLM settings
- CUDA graphs matter most: with enforce_eager the full model decodes 17 tok/s at batch 1 and 524 at batch 32, against 259 and 2507.
- Chunked prefill off, max_num_seqs 32 and 256: no measurable change in decode (257, 265, 247 tok/s at batch 1 against 259).
- enable_expert_parallel needs two or more GPUs per replica, and this track ran one GPU per container, so it was not measured.
- On an H200, vLLM 0.32 picks the "humming" MoE kernel for W4A16 MoE weights, and its NVRTC compile fails in the CUDA 12.8.1 devel image (NVRTC_ERROR_BUILTIN_OPERATION_FAILURE). Passing moe_backend="marlin" (config part `marlin`) fixes it. On an A10G the default path works. The failed H200 rows are in results/efficiency_full_w4a16_humming_failures.json.

## Smallest GPU

Weights are the vLLM "Model loading" GiB. A card needs the weights, about 3 GiB of activations and CUDA graphs, and some KV room, all inside 0.90 of its memory.

| GPU (memory) | model and config | result |
|---|---|---|
| H200 (141 GB) | full bf16, 56.9 GiB | ran: HumanEval 90.9, 259 tok/s b1 |
| H100 (80 GB) | full bf16, 56.9 GiB | fits by the weight math, not run |
| L40S (48 GB) | full bf16, 56.9 GiB | does not fit |
| L40S (48 GB) | full fp8, 29.1 GiB | ran: HumanEval 90.9, MBPP 84.0, 129 tok/s b1, 893 b32, 101,680 KV tokens. With eagle3b5: HumanEval 92.1, MBPP 86.0, 204 tok/s b1, 269 on real requests |
| L40S (48 GB) | slim64 bf16, 29.9 GiB | ran: HumanEval 88.4, MBPP 83.7, 97 tok/s b1, 95,696 KV tokens. With eagle3b: 149 tok/s b1 |
| A10G (24 GB) | full INT4, 15.7 GiB | ran: HumanEval 90.2, MBPP 84.4, 133 tok/s b1, 479 b8, 1107 b32, 31,360 KV tokens |
| A10G (24 GB) | full INT4 + eagle3b | ran: HumanEval 92.1, MBPP 84.4, 219 tok/s b1 (1.65x), 249 on real requests, 14,960 KV tokens |
| A10G (24 GB) | full INT4 + eagle3b5, eagle3b7 | ran: HumanEval 92.1 both, 208 and 195 tok/s b1, 268 and 277 on real requests |
| L4 (24 GB) | slim INT4, 8.7 GiB | ran: HumanEval 87.2, MBPP 82.9, 89 tok/s b1, 800 b32, 107,392 KV tokens |

A 24 GB card is the smallest tier that runs the model, and it needs INT4. The full INT4 model leaves 2.9 GiB of KV room on an A10G (31,360 tokens, about 7 requests of 4096 tokens). The slim INT4 model leaves 9.8 GiB on an L4. A10G and L4 are the smallest GPUs in the list, so nothing smaller was tried.

## Best configs

- Fastest within 2 points of the start HumanEval (90.9, so 88.9 or more): full INT4 with Marlin and EAGLE-3 with 5 draft tokens (full_w4a16, marlin+eagle3b5, H200). HumanEval 91.5, MBPP 85.6, 502 tok/s at batch 1, 681 on real requests, 6444 tok/s at batch 32, 16.0 GiB. With 7 tokens the real-request number is 778 tok/s (HumanEval 90.9, MBPP 85.2). With 3 tokens and topk6 it is 514 tok/s at batch 1 and 6344 at batch 32 (HumanEval 91.5).
- Smallest memory within 2 points: slim INT4 (slim_w4a16, marlin), 8.7 GiB in vLLM and 8.57 GiB on disk, HumanEval 89.0 and MBPP 81.3 on an H200. The same checkpoint scored 87.2 on an L4, 3.7 under the start, so it sits at the edge of the rule, and MBPP is 3.9 under the start (85.2). If the rule applies to MBPP too, the smallest passing config is the full INT4 model at 15.7 GiB (HumanEval 90.9, MBPP 84.4).
- Smallest GPU that runs a within-2 config: a 24 GB card (A10G, ran): full INT4 with EAGLE-3, HumanEval 92.1, MBPP 84.4, 219 tok/s at batch 1 and 249 on real requests.

Against the start (full bf16 on one H200: 56.9 GiB, 234 tok/s on real batch-1 requests, 2507 tok/s at batch 32), the fastest config uses 3.6x less weight memory, runs real batch-1 requests 2.9x faster (681 against 234 tok/s), and decodes batch 32 at 2.6x the rate.

## What did not work or is not measured

- Draft models (Qwen3-0.6B, Qwen3-1.7B) lose to plain decoding in vLLM 0.32.
- topk4 loses 3.7 points of HumanEval on the full model and 4.9 on the slim model. It needs a heal.
- Dynamic FP8 and INT4 both lose batch-1 speed on an H200. They win on batch 32 and on memory.
- INT4 on an H200 needs `marlin`; the default selection failed.
- A calibrated FP8 checkpoint of the full model and an AWQ checkpoint were not built. FP8 disk size is known for slim64 only (15.54 GiB). INT4 calibration used code text, and only code was scored.
- H100 80 GB and enable_expert_parallel were not run.
- gpu_peak_gib is about 126 GiB on every H200 row because vLLM reserves 0.90 of the card for KV. Use "weights GiB" and "KV tokens" to compare.
- One run per config. A one-point gap between rows is not a result.

## Reuse on other models

    modal run --detach modal_efficiency.py --model hub:<id> --name <name> --configs base,fp8,kvfp8,topk6,ngram,eagle3b --gpu H200 --shards 2
    modal run --detach modal_efficiency.py --model vol:<path under /vol> --tokenizer <hub id> --name <name> --configs base
    modal run modal_efficiency.py --quantize w4a16 --model vol:<path> --quant-name <name> --gpu H200      (writes /vol/quant/<name>)
    modal run modal_efficiency.py --name <name> --collect                                                 (volume rows to results/)
    python scripts/efficiency_table.py results/efficiency_<name>.json

`--smoke` runs 8 prompts at batch 1 and 8 with no benchmarks. `--estimate-only` (after the script name) prints the cells and a rough GPU time. A Hub id or volume path must exist in the volume of the active Modal profile, so copy or rebuild the round 28 business models and the 235B slims into that workspace first and pass --model vol:<path>. The EAGLE heads fit Qwen3-30B-A3B-Instruct-2507 only; the 235B needs its own head or just the ngram part. Use the `marlin` part with every W4A16 checkpoint on Hopper.

Cost: about 15 H200 container hours plus about 4 hours of A10G, L40S, and L4 containers. An estimate of about 80 dollars from list prices, not read from the invoice.
