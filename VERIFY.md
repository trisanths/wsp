# WSP deletion masks: VERIFY items checked (2026-11-14)

Source: `wsp_deletion_masks_agent_brief.pdf`, sections 7, 8, and 13. Checked against the arXiv HTML of 2611.10866, the project page, the GitHub repo, the Hugging Face mask repo, and the Qwen2.5 1.5B configs.

## 1. Did the Deletion Tuning authors release code?

Yes. MIT license.

- Code: https://github.com/agentic-learning-ai-lab/deleting-weights
- Project page: https://agenticlearning.ai/deleting-weights/
- GRPO runs on a vendored, patched copy of verl. SFT has its own env (`requirements_sft.txt`, torch 2.7.1) and `scripts/train_sft.sh`.
- `src/apply_mask.py` applies a mask to its base and saves a normal HF model. That is most of the brief's `compose` backend for free.
- Masks are saved as compressed bitmaps.

## 2. Trained masks are public

Hugging Face repo `Jacklu0831/deletion-tuning-masks`:

| Experiment | Base | Masks |
|---|---|---|
| sft/llama-3.2-1b-instruct-mmlu-pro | Llama-3.2-1B-Instruct | 14, one per MMLU-Pro subject |
| sft/llama-3.2-1b-instruct-bbeh | Llama-3.2-1B-Instruct | 23, one per BBEH task |
| sft/llama-3.2-3b-instruct-mmlu-pro | Llama-3.2-3B-Instruct | 14 |
| sft/llama-3.2-3b-instruct-bbeh | Llama-3.2-3B-Instruct | 23 |
| grpo/* | Qwen2.5 0.5B/3B/7B/14B, Qwen3-30B-A3B, Llama 1B/3B/8B | 1 each (math) |
| sdft/* | Qwen2.5-7B-Instruct | 2 (ToolAlpaca, Science) |

This changes the plan. The Llama-3.2-1B and 3B SFT sets are 37 independently trained masks per base on a shared model. E4 (mask algebra, Claim A) can start at hour 0 with no mask training. The planner swarm (Claim B) also gets a much bigger catalog than 3 masks.

## 3. The recipe differs from the brief's reconstruction (section 8)

| Brief guessed | Paper does |
|---|---|
| Real-valued scores, hard global top-k | One logit G per gate_proj weight, no fixed k, no sparsity constraint |
| Straight-through estimator | No STE. Soft mask W * sigmoid(alpha(t) * G) in training, hard mask W * 1[G > 0] at eval |
| Init so nothing is deleted | Init from Wanda scores: top-importance weights at N(0.03, 0.01^2), rest at N(-0.03, 0.01^2) |
| | alpha ramps 1e2 to 1e3 (SFT) or 1e3 to 1e4 (GRPO); regularizer -lambda * ||W~||^2, lambda = 1e-5 |
| | SFT: lr 5e-4, 400 steps. GRPO: lr 3e-4 |

Use their code. Do not reimplement from section 8.

Note: the DSL's `budget deleted <= 0.15%` clause has no direct analog in their method, since sparsity is emergent. A budget would have to be a post-hoc rule (drop the lowest-|G| deletions) or a new penalty.

## 4. Sparsity denominator

Total model parameters. Appendix B: "fraction of all model weights that Deletion Tuning deletes"; table caption "Fraction of all model parameters set to zero."

## 5. Composition is untested in the paper

Confirmed. The paper lists it only as future work ("Masks for different tasks might also be combined, for example by taking their union"). Claim A's novelty holds, but the authors named the union idea, so cite that sentence and do not pitch union itself as new.

## 6. Qwen2.5-1.5B is not in the paper

They used Qwen2.5 0.5B, 3B, 7B, 14B. No released 1.5B mask. Mask training wall-clock is not reported (H200 for GRPO, L40S for SFT).

## 7. Qwen2.5 1.5B variants share the base architecture

All three are Qwen2ForCausalLM, 28 layers, hidden 1536, intermediate 8960, 12 heads / 2 KV heads, vocab 151936, tied embeddings.

- Qwen2.5-1.5B: rope_theta 1e6, max_pos 131072.
- Qwen2.5-Coder-1.5B: rope_theta 1e6, max_pos 32768.
- Qwen2.5-Math-1.5B: rope_theta 1e4, max_pos 4096. HF lists Qwen2.5-1.5B as its base, but the RoPE change and the Math tech report's large continued pretraining mean its task vector is large and not a clean fine-tune delta. Treat Math as a weak task-vector baseline.

The brief's own fallback (train our own LoRAs and FFT on the base for every comparison) is the safe path.

## Recommended plan change

Switch the base to Llama-3.2-1B-Instruct (or 3B) and use the released SFT masks:

- E2 drops from "train 3 masks" to "download 37 masks". Keep one local SFT run only to measure mask-training time and confirm the pipeline.
- E4 can run in the first hours: pairwise union / intersection / difference over MMLU-Pro subjects and BBEH tasks, scored on each parent's task and on held-out tasks for forgetting.
- E3 heatmaps come straight from the bitmaps.
- Cost: benchmarks become MMLU-Pro and BBEH, not GSM8K / MBPP / IFEval. If the hackathon tracks score on specific benchmarks (open question 1 in the brief), that may force Qwen and own masks anyway.

## Still open (need the team)

- Which benchmarks the tracks score.
- GPU type and hours. No AWS (credits gone); Modal has about $30 left; RunPod MCP needs `/mcp` auth.
- Contents of Arjun's doc.
- Kryptos K4: not checked (brief says only if someone mentions it).
