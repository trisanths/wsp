# Sapien round 19 plan: routed expert transplant into a same-size base

## Decision

Totals from the three judges (vision + evidence + cost):

| Design | Vision | Evidence | Cost | Total |
|---|---|---|---|---|
| Routed expert transplant (RET) | 8 | 4.5 | 7 | 19.5 |
| Folded expert slices (FES) | 8 | 4 | 7 | 19.0 |
| Seam-distilled expert transplant (SDX) | 7 | 4 | 7.5 | 18.5 |
| Reader-complete tail splice | 5 | 4 | 6.5 | 15.5 |
| Giant expert mining | 7 | 2.5 | 3.5 | 13.0 |

RET wins. The plan keeps its core and adds parts from the other designs and the judges:

- From SDX: a knowledge-free control made from spectrum-matched random experts, a learning-rate sweep for the distillation competitor, and the recovery rate on items the donor gets right and the base gets wrong.
- From SDX, changed: a source-side signature test. SDX restricted the source's routing to the selected experts, which the evidence judge expected to collapse every arm toward chance. This plan instead knocks the selected experts out of the source and compares that with a random knockout.
- From FES: the exact T_out fold, and the size-neutral variant that prunes host MLP neurons, held for later.
- From the giant design: the measured checkpoint facts, the cached streaming pass for each giant, and the coverage rule for cross-tokenizer distillation.
- From the splice design: the corrected reading of run 14b, and a training-free tail-splice test as the fallback diagnostic if round 19 stops.
- From the judges: build on the existing `wsp/transplant.py` and `modal_round18.py` instead of new files. Add a no-heal content gate before any heal. Give the distillation competitor a LoRA rank matched to the transplant's trainable count. Stage the spending so seed 0 decides before seed 1 is paid for. Count size against the unique 1.72B. Allocate experts by measured coverage when the source is a giant. Add a cross-tokenizer probe before any money goes to GLM, DeepSeek, or Kimi.

The judges put the chance of a clear GO at about 10 to 15%. Because of that, the plan buys a cheap and decisive answer first.

Repo state, checked today. `wsp/transplant.py` (TransplantBank, fit_stitches, heal, save/load) and `modal_round18.py` already exist for this exact pair. Round 18 is built but has not run, and there are no round 18 rows. Round 19 is a set of changes to those files.

## The method

The base H is Qwen/Qwen3-1.7B: 28 layers, hidden 2048, MLP 6144, tied embeddings, 1.72B unique params. The HF count of 2.03B includes a stored copy of lm_head, and every size ratio here uses 1.72B. The donor D is Qwen/Qwen3-30B-A3B: 48 layers, hidden 2048, 128 experts per layer, moe_intermediate 768, top-8, norm_topk_prob true. Both use tokenizer.json blob cd71f61a, vocab 151936, and rope_theta 1e6, and both run on transformers 4.57.1. One expert is 4.719M params.

Notation:
- h_k is the host residual at the pre-MLP point of layer k.
- u = h / rms(h), with no norm weight.
- g_l is the donor's post_attention_layernorm weight at layer l.
- Wr_l is the donor router, 128 x 2048.
- Position 0 is the attention sink, because Qwen adds no BOS. It is excluded from every fit, and the transplant output there is zero.

Step 0, measure (one H200, both models in bf16).
1. Selection texts are 512 MedMCQA train items (skip 0) and 512 WikiText-103 general paragraphs (`atlas.general_texts`, seed 0). Coverage texts are MedMCQA train skip 512, count 256, plus 256 general texts (seed 1). Max length is 256.
2. REAP routed mass per donor layer and expert: M(e) = mean over tokens of q_e(t) * ||E_e(g_l * u_t)||, where q_e is the renormalized top-8 weight and is 0 when e is not picked. Compute it on skill tokens and on general tokens. Count content tokens only: mask the chat template, the option letters, and the "Answer:" span. Then log_ratio = log((M_skill + 1e-6) / (M_gen + 1e-6)).
3. Layer map. Compute linear CKA between host u_k (k = 0..27) and donor u_l (l = 0..47) on 20k non-sink tokens, half medical and half general. A monotone DP picks l(k), strictly increasing and within +-4 of round(47k/27).
4. Selection S_k is the 6 experts of l(k) with the highest M_skill among those with log_ratio >= 0.4. If fewer pass, fill with the highest-M_skill experts left. That gives 168 experts. Coverage_k is the share of all skill mass in that layer that S_k holds, measured on the coverage texts.
5. Teacher cache. Run D once over the heal texts at seq 512. Store the top-64 ids (int32) and logprobs (fp16) per position, about 1.6 GB. Every distillation arm reads this one file.
6. State cache. Over 1500 texts (750 MedMCQA train skip 1024, 750 UltraChat first turns), keep u_k and u_l(k) for about 200k non-sink tokens. Drop tokens with rms above 10x the layer median.

Step 1, stitches. These are layer-local, so no backprop runs through the donor's tail.
- In-stitch A_k (2048x2048, fp32). Initialize with `wsp.align.ridge_fit(u_k -> u_l)`, with lam picked from {1e-4, 1e-3, 1e-2, 1e-1} by held-out R2. Then train 500 Adam steps (lr 1e-4, 4096 tokens per batch) on L_A = ||MoE_l(A u_k) - MoE_l(u_l)||^2 / ||MoE_l(u_l)||^2 + 0.1 * KL(router_D(u_l) || router_D(A u_k)). MoE_l is the full donor block with all 128 real experts. The objective does not depend on the arm, so every transplant arm shares the same A_k.
- Out-stitch B_k is a ridge fit from donor u_l to host u_k. The scale s_k is median rms(h_k) / median rms(d_l).

Step 2, the written layer (TransplantBank in "donor" mode):

```
u  = h / rms(h)
z  = Wr_l (g_l * A_k u)        all 128 rows, donor softmax, donor top-8 T8, renormalized q
executed = slots of S_k that are in T8, highest q first, cap 3; each slot has a learnable bias b_e
y  = hostMLP(hostnorm(h)) + s_k * beta_k * B_k * sum_e q_e E_e(g_l * A_k u)
```

The host MLP is always on. The transplant term is zero at position 0 and at tokens with rms(h) > tau_k, where tau_k is 3x the 99.9th percentile. beta_k starts at 0, so a fresh model equals the base exactly. The donor's own router decides when each expert fires, so an expert fires only on tokens where the donor would have used it.

Step 3, heal. 1000 steps, batch 8, seq 512, AdamW wd 0, 50 warmup steps, cosine, clip 1.0.
- Data: 8000 texts in the Qwen3 chat template with enable_thinking=False, one pass. 6000 are MedMCQA train items from skip 2048 onward, written as question, options, "Answer: X." and the "exp" field when present. 2000 are UltraChat-200k train_sft first turns.
- Loss: CE + 1.0 * KL(teacher top-64 || student), with a tail bucket and T = 1.
- Trainable: host LoRA r16 on all 7 linears (17.4M, lr 1e-4); rank-16 deltas dA_k, dB_k (3.7M, lr 1e-4); beta_k (lr 1e-2); slot biases (lr 1e-3).
- Frozen: every expert weight, the router rows, and the host base weights.

Step 4, compile:

```
gate'/up' = W diag(g_l) (A_k + dA_k)             768 x 2048
down'     = s_k beta_k (B_k + dB_k) W_down        2048 x 768
router'   = Wr_l diag(g_l) (A_k + dA_k)          128 x 2048
```

Merge the LoRA. The shipped network has no stitch matrices. Its only extra norm is the weightless u = h/rms(h). The SwiGLU stays exact under the fold.

Size:

| | Params | Ratio to 1.72B |
|---|---|---|
| Total: 1.72B + 168 x 4.719M + 7.3M of router rows | 2.52B | 1.47x |
| Max active at cap 3 | 2.12B | 1.23x |
| Fallback, 5 experts per layer | 2.39B | 1.39x |

Measured mean active is also logged.

## Why it should beat distillation, LoRA, and random components

Against distillation. Distillation can only teach what the heal tokens show. The test sets ask held-out facts that the heal set mostly does not contain. The repo shows this limit:
- Run 7: LoRA gains +3.0 on held-out MMLU (44.7 to 47.7), while reading the donor's states for the same question gains +23.6 (68.3).
- Run 6: the mismatched-message control falls back to host level (48.6). The missing knowledge lives in the donor's parameters and is specific to the question.

FFN layers act as key-value memories (Geva et al. 2021), and mid-depth MLPs hold factual recall (ROME). The transplanted experts are those parameters, and the donor's own router fires them on new questions whether or not the heal covered the fact. A distilled LoRA has no such store. Storing a fact by gradient takes many exposures (Allen-Zhu and Li, Physics of LMs 3.3), and a one-pass heal gives a few at best.

Against LoRA. The data, loss, and steps are the same. LoRA gets the same format gain but no new parametric knowledge. Full fine-tuning was worse than LoRA in the baselines.

Against random components. Run 14b's "random block" was another layer range of the same 14B donor, so it held real knowledge. It showed that block choice did not matter. It did not separate knowledge from capacity. Round 19 uses two controls that fix everything except content:
- Swap: the same slots, router rows, q weights, stitches, and trainables, but the slots run the weights of random other real experts from the same layer. The routing is real and the content is mismatched.
- Spectrum: each slot's three matrices become U diag(sigma) V^T, with sigma from the selected expert and Haar-random U and V. Norms and spectra match, and no knowledge is left.

Three things separate this from rounds 1 to 4 and run 14:
1. The units are chosen by the donor's own router mass, not by neuron activation.
2. The 28 seams each write one MLP-sized update at CKA-matched depth. This is the kind of update the host MLP already makes there. Run 14 had one seam that asked late host layers to read a donor layer-34 state.
3. The in-stitch is trained on the donor block's own output and routing. Run 14 and the stitch study show that a stitch trained on the component's own objective is close to lossless.

Symphony-MoE shows that unaligned same-family experts fail badly (33.94 vs 44.07), so the alignment step is needed.

Open risk. The host's later layers must learn to read the expert writes from a 1000-step r16 LoRA (runs 14 and 15). BTS and MergeME needed far more tokens. The no-heal gate and the frozen-host arm measure this risk directly.

## Experiment 1: round 19, same family and same width

Script: new `modal_round19.py`, using the round 18 image (torch 2.7.1, transformers 4.57.1) and its helpers. Fan-out uses `python scripts/modal_fanout.py modal_round19.py round19 --profiles a,b,c`.

Models: Qwen/Qwen3-1.7B (base), Qwen/Qwen3-30B-A3B (donor and teacher), and Qwen/Qwen3-4B (eval-only reference).

Benchmarks. All are scored by the log-prob of A to D through `wsp.mcq.score_items`, with the same raw prompt for every arm.

| Set | Items | Source |
|---|---|---|
| MedMCQA validation | 1000 | `wsp.atlas.medicine_items(1000)`, the round 10 order |
| MMLU medical test | 1561 | anatomy 135, clinical_knowledge 265, college_biology 144, college_medicine 173, medical_genetics 100, nutrition 306, professional_medicine 272, virology 166, via `mcq_items('mmlu','test',subjects=...)` |
| MedQA-USMLE 4-option test | 1273 | `mcq_items('medqa','test')`, used as the tiebreak |
| MMLU general held-out | about 1780 | `modal_round7.mmlu_split(heldout=True)` minus the 8 medical subjects; log the exact count. This is the off-target guard. |
| KD-competitor dev set | 500 | MedMCQA train skip 9000. Used only to pick the competitor's learning rate. |

### Stage 0: measure and gate

One H200, about 2.5 h, about $12. It runs step 0 and step 1, then the eval-only arms: base, donor, and Qwen3-4B.

- G0a: the tokenizers are identical. This is already true.
- G0b: the median held-out coverage_k is at least 0.20. Uniform would be 0.047. STOP below 0.20.
- G0c: after the stitch fit, the median layer cosine of MoE_l(A u_k) against MoE_l(u_l) is at least 0.60, and the mean top-8 overlap is at least 4 of 8. STOP if the cosine is below 0.40.
- G0d, knockout in the source: in Qwen3-30B-A3B, set the router logits of the 168 selected experts to -inf in the 28 mapped layers, then do the same for 168 random experts from the same layers (seed 100). Score MedMCQA 1000 and MMLU general for both. Pass if the selected knockout costs at least 2.0 more MedMCQA points than the random knockout, and costs less on MMLU general than on MedMCQA. STOP otherwise, because the selected experts do not carry the skill even inside the source.
- G1, no-heal content gate: run the sel, swap, and spec arms with no training at beta in {0.25, 0.5, 1.0}. Measure LM loss on 200 held-out MedMCQA train texts with explanations, and the KL to the teacher. Take the best beta for each arm.
  - If sel is at least 0.01 nats below both controls, it passes.
  - If sel is better by less than 0.01 nats, run Stage A, but Stage B then depends on Stage A.
  - If sel is no better than both controls, STOP.

### Arms

The seed changes the data order, the LoRA init, the swap draw (seed 1234 + 100*seed + k), and the Haar draw.

| Arm | What | Seeds | Stage |
|---|---|---|---|
| base | Qwen3-1.7B, no training | none | 0 |
| donor | Qwen3-30B-A3B, the ceiling | none | 0 |
| ref4b | Qwen3-4B | none | 0 |
| lora_kd | LoRA at the rank from `wsp.lora.rank_for_budget(base, 21.1e6)` (about r19, matched to the transplant's trainables), KD + CE, lr in {5e-5, 1e-4, 2e-4} | 0 per lr | A |
| xp_sel_kd | the design | 0 | A |
| xp_swap_kd | selected routing, swapped real weights | 0 | A |
| xp_spec_kd | selected routing, spectrum-matched random weights | 0 | A |
| lora_lm | LoRA r16, CE only | 0, 1 | B |
| lora_kd | at the best lr by dev | 1 | B |
| xp_sel_kd, xp_swap_kd, xp_spec_kd | second seed | 1 | B |
| xp_rsel_kd | randomly picked real experts with their own router rows (selection test, not a gate) | 0 | B |
| xp_sel_lm | selected, CE only | 0 | B |
| xp_sel_kd_frozenhost | host LoRA off; only deltas, beta, and biases train | 0 | B |

Every row records:
- all four scores;
- the recovery rate on items the donor gets right and the base gets wrong, and the flip rate on items the base gets right;
- total, max active, and mean active params;
- decode tok/s at batch 1;
- beta_k for every layer;
- slot firing rates on medical and general prompts.

### Go and stop

Stage A gate, seed 0. Stage B runs only if both conditions hold on the mean of MedMCQA and MMLU-med:
- xp_sel_kd - max(xp_swap_kd, xp_spec_kd) >= +1.0;
- xp_sel_kd - best lora_kd >= +1.0.

Otherwise STOP. This answer comes about 3.5 h after launch.

Final GO uses the mean of 2 seeds, with an item-paired bootstrap pooled over seeds at p < 0.05. All of these must hold:
1. xp_sel_kd - lora_kd (tuned, rank-matched) >= +3.0 on MedMCQA and >= +3.0 on MMLU-med.
2. xp_sel_kd - max(xp_swap_kd, xp_spec_kd) >= +2.0 on both.
3. MMLU general of xp_sel_kd >= lora_kd - 1.0.
4. Measured total <= 1.5x and max active <= 1.3x of 1.72B.

STOP the line if xp_sel_kd - max(swap, spec) < +1.0 on both medical sets.

In between, run seed 2 of xp_sel_kd, xp_swap_kd, and lora_kd (about $12) and use MedQA as the tiebreak.

Noise: the per-arm SE is about 1.6 on MedMCQA 1000 and about 1.2 on MMLU-med. Paired differences pooled over 2 seeds have an SE of about 1.0.

### Cost and time

| Item | Cost |
|---|---|
| Stage 0 with gates and eval-only arms | $17 |
| Stage A, 6 cells at about 0.8 h | $22 |
| Stage B, 9 cells | $33 |
| Image builds and one rerun | $25 |
| Total | about $100, cap $120 |

Wall clock is about 3.5 h to the Stage A answer and about 5 h to the full table.

## Experiment 2: round 20, a Qwen-Max-class giant with the same tokenizer, plus a cross-tokenizer probe

This runs only on a round 19 GO, or in the in-between band with a content gap of at least +1.5.

No open Qwen-Max-class weights near 278B exist on HF. Two designs checked this. Qwen/Qwen3-235B-A22B is the closest stand-in, and I checked it today:
- qwen3_moe, so it is native in 4.57.1;
- hidden 4096, 94 layers, 128 experts x 1536, top-8, norm_topk_prob;
- vocab 151936 and the same tokenizer.json blob cd71f61a as Qwen3-1.7B;
- 470.2 GB in bf16. The -FP8 checkpoint is 239.0 GB.

This experiment tests scale and width with no tokenizer change.

### Part A: Qwen3-235B-A22B experts into Qwen3-1.7B

- Load. Statistics, the state cache, and the teacher top-64 come from Qwen/Qwen3-235B-A22B-FP8 on H200:4. If the 4.57.1 FP8 path fails, load bf16 on H200:8. Fetch the selected expert tensors in bf16 from the base repo with `wsp.fetch.fetch_tensors`: 84 experts x 18.9M, about 3.2 GB.
- Width fold. A_k is 2048 to 4096 and B_k is 4096 to 2048. They exist only in training. After the fold, each expert is 3 x 1536 x 2048 = 9.44M.
- Budget rule, new. 84 experts in total, allocated greedily by held-out marginal REAP mass over the 28 mapped layers, at most 8 per layer, cap 2 executed per layer. Expect about 20 layers to get experts. Total is 1.72 + 0.79 + router rows, about 2.52B (1.47x). Max active is at most 1.72 + 20 x 2 x 9.44M = 2.10B (1.22x). The swap and spec controls use the same allocation.
- Arms: lora_kd235 (KD from 235B, the round 19 best lr, 2 seeds), xp235_sel_kd (2 seeds), xp235_swap_kd (2 seeds), xp235_spec_kd (1 seed). The round 19 base and xp_sel_kd rows are reused, with the same heal data and evals.
- Same gates (G0b, G0c, G0d, G1) and the same benchmarks.
- GO: the same four conditions as round 19, applied to the 235B arms.
- Reported, not gated: xp235_sel_kd minus round 19 xp_sel_kd. A positive value means a bigger source gives a better output at the same size, which is the owner's goal.

### Part B: cross-tokenizer probe, training-free, about $10

LiquidAI/LFM2-24B-A2B is lfm2_moe: hidden 2048, 40 layers (2 dense), 64 experts x 1536, top-4, vocab 65536, tokenizer blob d54bc82c, 47.7 GB. Its experts go into Qwen3-1.7B. It runs on one H200 with a transformers 5.x image. Run G0b, G0c, and G0d only:
- Positions are paired by character end offsets (`wsp.align.match_offsets`, `wsp.stitch.matched_index`).
- Record the KD alignment coverage: the share of Qwen positions whose mapped teacher mass is at least 0.5.

Pass if the median MoE-output cosine is at least 0.50, the top-4 overlap is at least 2 of 4, and the KD coverage is at least 60%. A failure here blocks the GLM, DeepSeek, and Kimi steps before they cost anything.

Cost: s0 on H200:4 for about 2.5 h is $45; 7 heal cells are $25; Part B is $10; builds and slack are $20. The total is about $100, and wall clock is about 5 h. The two experiments together use about $200 of the $700.

## Code changes by file and function

`wsp/transplant.py`, changed:
- TransplantBank gets `router_mode="donor"`:
  - a frozen full router buffer (n_src x d_src) and the donor top_k with norm_topk;
  - slot router ids and slot weight ids, so the swap control executes R_k weights through S_k's rows;
  - slot biases, `cap`, and `input_mode="unit"` (u = h/rms(h), then g_l times A u, with no source norm module);
  - beta_k, the sink mask, and the tau_k massive-token mask;
  - low-rank deltas dA and dB around a fixed A0 and B0.
  - The current K-local softmax mode stays for old rows.
- `select_experts`: add `score="reap"` (from `atlas.expert_reap_map`), `per_layer` or `total=` with `allocate_budget(reap_map, total, per_layer_max, cap, active_limit)`, and `control in {"swap", "spectrum", "random_select"}`.
- `spectrum_random(weight, seed)`: Haar U and V with the real singular values.
- `fit_stitches`: add `objective="moe_block"`. It calls `align.fit_moe_stitch` and returns A, B, s_k, the held-out cosine, and the top-k overlap.
- `heal`: accept `teacher_cache=` (a `wsp.distill.TopKTargets`) in place of a live teacher. The live full-vocab path stays for 14c.
- `fold_transplant(model)` and `export_folded(model, path)`: fold A, B, s, and beta into the gate, up, down, and router weights, merge the LoRA, and write manifest "transplant/2" with the base id, donor id and sha, layer map, S_k, R_k, allocation, REAP signatures, and coverage. `load_transplant` reads both formats.
- `count_params`: total, max active under the cap, and mean active measured from firing counts.
- `go_stop`: the round 19 rule (Stage A gate, final GO, STOP band).

`wsp/atlas.py`:
- `expert_reap_map(model, tokenizer, skill_texts, general_texts, max_length=256, content_mask_fn=None)` builds on `find_routers` and `routing_counts`.
- `expert_coverage(reap_map, selection)`.
- `knockout_experts(model, layer_to_experts)` is a context manager that masks router logits, for G0d.

`wsp/align.py`:
- `linear_cka(x, y)`, `monotone_layer_map(cka, window=4)`, and `capture_premlp(model, layers)`. The last reuses `mlp_input_capture` but takes the pre-norm residual.
- `fit_moe_stitch(u_host, u_donor, moe_block, steps=500, lr=1e-4, kl_weight=0.1)` does the ridge init through `ridge_fit`, then trains on L_A.

`wsp/distill.py`, new:
- `cache_teacher_topk(model, tokenizer, texts, k=64, seq_len=512, out_dir)`.
- `TopKTargets(out_dir)`, aligned with the heal order.
- `topk_kd_loss(student_logits, ids, logprobs, mask)`, with the tail bucket.
- Later: `byte_vocab_map(tok_src, tok_dst)` and `aligned_kd_targets(..., min_mass=0.5)` for cross-tokenizer sources, with a coverage report.

`wsp/mcq.py`: add subset helpers `MMLU_MED` (the 8 subjects) and `mmlu_general_heldout()`. The MedQA loader already exists.

`wsp/lora.py`: no change. `rank_for_budget` already exists.

`modal_round19.py`, new:
- stage `s0_measure` writes `/vol/round19/plan.json`, `stitches.safetensors`, the teacher cache, and the gate results;
- stage `noheal_gate`;
- `run_cell(arm, seed, lr)` with Stage A and B lists;
- `verdict`.
- It reuses the image, `thinking_off`, `mcq_text`, the pools, and `load_model` from `modal_round18.py`.
- It writes `results/round19_rows.json` and per-item records to `/vol/results/round19/<arm>_s<seed>.json`.

`modal_round20.py`, new, built only after a round 19 GO:
- H200:4 s0 for Qwen3-235B-A22B-FP8;
- bf16 expert fetch;
- the width-fold path. `Stitch` already handles d_source different from d_model.
- `probe_xtok` on a transformers 5.x image for LFM2-24B-A2B. It needs the fused-expert support that the `wsp/transplant.py` docstring lists as untested. `wsp/giant.py` FusedExpertLayout shows how fused experts are laid out.

`tests/test_transplant.py`, extended, on tiny random Qwen3Config and Qwen3MoeConfig models:
1. With beta = 0, the logits equal the host to 1e-6.
2. With A = B = I, s = 1, beta = 1, the host MLP zeroed, S = all experts, and cap = top_k, the bank equals the donor Qwen3MoeSparseMoeBlock to 1e-5.
3. Folded equals unfolded to 1e-5.
4. The swap and spec arms route exactly like sel.
5. `spectrum_random` keeps the singular values.
6. `knockout_experts` never routes to a masked expert.

After a GO: add a `transplant` statement to `wsp/lang.py` and `docs/language.md`, for example `transplant from Qwen/Qwen3-30B-A3B skill medicine experts 168 alloc coverage cap 3 map cka heal kd steps 1000`.

## Path to the giant sources

Folding makes expert cost depend only on host width. A folded expert costs 3 x d_host x I. The donor width affects only the stitch fit.

1. Round 19: same family, same width (Qwen3-30B-A3B).
2. Round 20: the same-tokenizer giant (Qwen3-235B-A22B), plus the cross-tokenizer probe.
3. First cross-tokenizer giant, after the probe passes. deepseek-ai/DeepSeek-V4-Flash is the cheaper choice: 159.6 GB, 43 layers, 256 experts x 2048, top-6, MXFP4 routed experts, layers 0..2 hash-routed. zai-org/GLM-5.3-Flash is the other: 328 GB FP8, 288 experts plus 1 shared, sigmoid routing with correction bias, mHC with 4 streams.
   - Routing statistics come from the existing `wsp/giant.py` layer streamer, which has FP8, FP4, and mxfp4 dequant and the DeepSeekV4, Glm, and Kimi layouts.
   - Extend it to save the MoE input u_l at mapped layers for a stitch subset of about 300k tokens.
   - For the teacher top-64, use a whole-model FP8 load on H200:2 (DSV4-Flash) or H200:4 (GLM), following the working `modal_extract_frontier.py` path. Do not write a new streamer for logits.
   - Stitch targets are the MoE input after the hc pre-mix and the ffn norm, never the residual streams.
   - Distillation uses byte-identical vocab mapping with mass >= 0.5, and the competitor uses the same aligned targets.
   - A folded expert is 3 x 2048 x 2048 = 12.6M at Qwen3-1.7B. The budget fits about 62 experts, allocated by coverage, with cap 2 over 20 layers to stay at or below 1.3x active.
   - Estimated cost is $60 to $90.
4. Qwen/Qwen3-4B as the base: 4.02B, hidden 2560, 36 layers. The room is about +2.0B total and +1.2B active, which fits about 125 GLM or DSV4 experts at 15.7M folded.
5. moonshotai/Kimi-K3: 1.56 TB, 93 layers, 896 experts, top-16, a latent MoE with a 3584 latent.
   - Stitch the host straight into the latent. Compose B with routed_expert_up_proj and fold the result into w2. A folded expert is 18.9M at host width 2048.
   - Streaming only: 26 to 52 min of download per pass. The teacher comes from the streamed final head, or from sequence-level KD.
   - Estimated cost is about $80 per pass.
   - deepseek-ai/DeepSeek-V4-Pro (864.7 GB, 384 experts of 66.1M) follows the same route at 18.9M folded.
6. Liquid bases.
   - LFM2-24B-A2B into LiquidAI/LFM2-1.2B or LFM2.5-1.2B. Both have vocab 65536, but the tokenizer.json blobs differ (43adc95c and 2e27bba0 against d54bc82c), so check id-by-id decode equality first. If they match, the pair is same-tokenizer.
   - LiquidAI/LFM2-350M (hidden 1024): a folded LFM2-24B expert is 3 x 1536 x 1024 = 4.7M. The budget is about 35 experts and about 20 executions per token.
   - The experts are per-token MLPs, so the LFM2 conv cache is untouched.
7. Several skills: `allocate_budget` ranks (layer, expert) cells by marginal REAP mass on the requested skill mix, under the total and active limits.
8. Size-neutral variant, from FES, after a GO: prune the 25% least-important host MLP neurons in transplant layers, then spend the freed params on more experts. The total then stays near 1.0x to 1.2x.

The source stays a compile-time and heal-time object at every step. The router is always the donor's own rows, folded.

## What to stop doing

- Do not launch `modal_round18.py` as built. Its K-local router makes the top-2 of K fire on every token, its rank-256 ridge stitches are never trained on the expert objective, it runs a live full-vocab teacher, and its random control uses its own router rows. Round 19 replaces it and reuses its pools and evals.
- Do not launch `modal_round17.py`. Its whole donor blocks, such as Qwen2.5-7B layers into a 0.5B base, break the 1.5x limit, and run 14b already answered that question.
- Stop selecting single-layer neuron clusters (rounds 1 to 4), grafting whole blocks into small bases (runs 14 and 14b), carving to half depth (run 15), and running full fine-tuning arms.
- Stop using "a random layer range of the same donor" as the capacity control. It holds real knowledge. Use the swap and spectrum-matched controls.
- Stop live-teacher distillation in heals. Use the cached top-64.
- Stop reporting size against the HF 2.03B count for Qwen3-1.7B. Use the unique 1.72B.
- Do not build GLM, DSV4, or Kimi teacher pipelines until rounds 19 and 20 give a GO and the cross-tokenizer probe passes.
- Let rounds 14c, 16, and 16b finish. Read 14c as a prior on how strong the distillation competitor is, then open no new rounds on those lines.
- If round 19 stops, drop the expert-transplant line. Spend at most about $15 on the tail splice's training-free G0 sweep: the pruned-source score against cut depth and expert count, and the seam check. That tests whether carrying the source's own downstream layers is the missing piece. It is not a product.