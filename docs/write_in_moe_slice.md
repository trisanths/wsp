# Write-in track moe-slice: status

Goal: graft a middle slice of Qwen3-30B-A3B-Instruct-2507 (48 layers, 128 experts, top 8)
into Qwen3-1.7B through Enc, Dec and a token gate (wsp/pgraft.py), and compare with LoRA,
full fine-tuning, controls and native Qwen3-4B and -8B.

Result: no accuracy numbers. No arm reached a scored row. Everything below is engineering.

## What works

- tests/test_pgraft_moe.py: a tiny random Qwen3-MoE donor slice (layers 2..4) run through
  ParallelGraft matches the native layers on real positions (fp32 1e-4, bf16 0.1) and
  trains Enc by backprop through the frozen MoE layers. It passes in fp32 and bf16 with
  transformers 4.57.1. So pgraft.build, BlockPart and set_sink need no change for Qwen3-MoE.
- modal_wi_moeslice.py: driver for base, graft, identity, scrambled (wsp/scramble.py on the
  slice), lora_match, lora_big (rank 128), fullft, native Qwen3-4B/-8B/donor cells, a donor
  soft-letter cache, MMLU held-out 2000, ARC-Challenge test, HumanEval, decode tok/s at batch
  1, params run per token (routed experts counted at 8/128), and GPU hours. Rows go to
  /vol/wi_moe/rows after each stage. Heal data is MMLU auxiliary_train (23,673 four-option
  items, never MMLU or ARC test): answer-letter CE with optional donor soft targets, KL to
  the frozen base on general text, stage-1 alignment kept as a regularizer. Learning rates
  are the corrected round 23 values (glue 3e-4, gate 1e-4).
- One graft cell ran the whole path on Qwen3-1.7B plus the MoE slice (layers 16..32 at host
  layer 13) with 3 alignment steps and 3 training steps: alignment survival 0.0375 (untrained,
  as expected) and finite loss. This is a smoke run, not a result.

## What blocked it (2026-11-15)

1. Hugging Face downloads from Modal containers stalled for hours in both workspaces
   (0 of 16 files). A CPU prefetch function with xet enabled finished in about a minute once
   a model was partly cached. Workspace rexheng-24542 now holds the donor, Qwen3-1.7B, -4B,
   -8B, ARC and the pool. hanavm holds the donor but not the datasets.
2. Cells must read datasets with HF_HUB_OFFLINE=1 and HF_DATASETS_OFFLINE=1, because online
   HEAD calls hung. modal_round6.mmlu_data tokenizes 100k rows and costs 30+ minutes, so the
   driver builds its own pool and eval split (same seeds and slices as round 7).
3. Modal gives a function 0.125 CPU by default. The MoE layers run an expert loop in Python
   (transformers 4.57), so a forward is CPU bound (GPU 3 to 20 percent). The driver now
   asks for cpu=8.
4. Loading the 61 GB donor from the Modal volume takes 25 to 120 minutes per cell (lazy
   volume reads of 18k small tensors). This is the open blocker. Seconds per training step
   of the grafted model were never measured with cpu=8, so the 600 + 1500 step plan has no
   cost estimate.

## Next steps

- Bake the donor into a Modal image (download at build time), or keep one container that
  loads the donor once and runs graft seeds 0 and 1 and the scrambled control in sequence.
- Replace the expert loop with a grouped GEMM or fused-expert path for the donor slice (stack
  the 128 experts per layer, sort tokens by expert). Transformers 5 has one.
- Measure seconds per align step and per task step first. If a step is above about 5 s, cut
  the slice (for example layers 20..28) and the step counts before launching any arm.
- Run order once it is fast: native donor soft cache, base, graft s0 and s1, identity,
  scrambled, lora_match, lora_big, fullft, native 4B and 8B. Score on held-out only.
