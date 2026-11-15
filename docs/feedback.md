# Feedback seams

Code: `wsp/feedback.py`. Tests: `tests/test_feedback.py`. Round script: `modal_round22.py`.

## The problem

Runs 14 and 14b (`results/README.md`) wrote a donor block into a small host's residual stream mid-stack. A trained input stitch fed the block almost losslessly (+0.008 nats on the donor's own LM loss). The host's few remaining layers could not read the block's output. A real Qwen-14B block in Qwen-0.5B did no better than a random block of the same donor. The reader is the bottleneck.

## The mechanism

The idea comes from "Latent Feedback Is a Compute and Data Amplifier" (Huang et al., 2026, alphaxiv 2611.latent-feedback-transformers-scaling-law). In a latent feedback transformer, each token's top-layer hidden state goes through a learned map W and is added to the next token's input. At generation this is one extra vector per step and the same KV cache.

A feedback seam uses that as a new place for a component's output to re-enter the host. A branch has:

- `read_layer` k: the host layer whose output the branch reads.
- `T_in`: a stitch from host width to donor width.
- a component: donor decoder layers first..last (a `wsp.parts.BlockPart`, with the donor's own layer classes, rotary base, and KV cache), or `None`.
- `T_out`: a bias-free linear map to host width. It starts at a ridge fit (`warm_start_t_out`) or at random, std 1/sqrt(width). Never zero.
- `gate`: `g_max * tanh(raw)`, with `g_max` 0.5 and `raw` starting at 0. A fresh seam equals the host exactly.

For position t the branch makes (rms_normalize divides a vector by its root mean square)

    u_t  = rms_normalize(block(x_t) - x_t),  x_t = T_in(h_t at layer k)     with a component
    u_t  = rms_normalize(top state at t)                                      component None
    fb_t = gate * embed_rms * y_t / |y_t|,   y_t = T_out(u_t)

`embed_rms` is the typical L2 norm of a row of the host's input embedding table. So the vector a branch writes has norm `gate * embed_rms`, whatever the raw scale of the block output or of T_out. The host input at position t+1 is `embed(token_{t+1}) + sum of fb_t over branches`. Position 0 gets nothing. For a left-padded row, the first real token gets nothing, and a pad never feeds a real token.

At the first real position the component input is replaced by the donor's sink state (the attention-sink fix from `wsp/graft.py`). That fix is the main reason the stitch is lossless.

Why the gate starts at 0 and T_out does not: at step 0 only the gate has a gradient (T_out is multiplied by the gate), so the gate opens first and slowly (`g_max * tanh(raw)` moves at most `g_max * lr` per step). T_out then trains once the gate is open. A random or fitted T_out gives the opened gate a direction to write in. The old form (`legacy=True`) is kept only for the regression test.

Branch kinds:

| Branch | Component | Use |
|---|---|---|
| plain | None | the paper's latent feedback; the capacity control |
| block | donor layers first..last | the real test |
| block, control random | a seeded other range of the same length of the same donor | content control |

`out_mode="delta"` (default) feeds T_out with `block(x) - x`, the change the block makes, as `wsp/insert.py` does. `out_mode="full"` feeds it `block(x)`.

## Three ways to run it

1. `generate` and `forward_cached`: token by token, with the host KV cache and one private cache per component. The feedback for step t+1 comes from step t's states. This is the exact recursion. The prompt also runs token by token, because each prompt token's feedback depends on the one before it. `generate` accepts the arguments `wsp.skills_eval` passes and is greedy only.
2. `forward_sequential`: the same recursion with no cache. Step t runs the whole prefix again. It is slow and exists for tests. It equals `forward_cached`.
3. `forward(passes=n)`, default 2: teacher forced. Pass 1 has no feedback. Pass p+1 feeds the branch outputs of pass p, shifted by one position. With `grad_through_passes=False` (default) only the last pass builds the graph, as in the paper. The read states of earlier passes are data. T_in, T_out, the gate, and the host LoRA learn through the last pass.

After n passes, positions 0 to n-1 are exact. The error at later positions need not fall at every pass (it is a fixed-point iteration), but a sequence of length up to n is exact. `passes_gap` measures the KL from the exact recursion to `passes=2` on real text.

Position ids count only real tokens in all three modes, so left padding does not shift a row.

## Training schedule

Stage 1, `fit_t_in`: T_in starts from a ridge fit on matched states and trains on the donor's own LM loss with the donor sink (`wsp.stitch.train_stitch`, objective `donor_lm`). This trains T_in so that the donor, reading host layer k through it, keeps its own loss. The loss runs through the donor's whole upper stack, not only the block. Host and donor stay frozen. Defaults: MLP stitch, lr 3e-4, 1000 steps, 1200 texts, then up to two more rounds of 500 steps while the penalty is above the threshold. The report holds `penalty` (donor loss through T_in minus native loss, nats), `threshold` (default 0.3), and `ok`. Round 22 refuses to start stage 2 when `ok` is false and ends the cell with status `failed_stage1`.

Before stage 2, `warm_start_t_out` fits T_out by ridge. The input is the branch's normalized output at t. The target is `E[top1 of the donor] - E[top1 of the base]` in host embedding space, on the rows where the two top-1 tokens differ, at lag 0 (both from the logits at t) or lag 1 (from the logits at t+1, the prediction the receiving position must make). The lag with the larger held-out cosine wins. The fit is rescaled to the size of a random start, which changes nothing in the forward pass (the written vector is normalized) and keeps one Adam step small relative to the weights. If there are too few rows or the held-out cosine is not above 0, the random start stays and the report says `random`. The target is a heuristic. The test shows the fit predicts the donor-minus-base direction better than a random T_out on held-out rows. It does not show that writing that direction helps the model.

Stage 2, `heal_distill` (`heal_seam` for a seam): T_out, the gates, a host LoRA r16, and T_in (lr 1e-5) train on

    KL(donor || seam) on skill pool text + 0.5 * KL(frozen base || seam) on general pool text + the stage-1 loss on T_in.

There is no LM loss on any text. The frozen base is the host with the LoRA scales set to 0 (`base_logits`), so no second copy is loaded. Position 0 is left out of every KL, as in `wsp.probe`. The stage-1 term is the donor's LM loss through T_in on two texts per step (`t_in_regularizer`); it keeps T_in a good feed for the block while T_in learns to read states that carry feedback. The same text order is used by every arm. The host LoRA covers only the layers above the read layer: a LoRA at or below it would move the state T_in reads.

Optimizer safety in the loop:

- Separate learning rates per group (T_out 3e-4, gate 1e-3, LoRA 2e-4, T_in 1e-5). T_out and the gate warm up linearly over 100 steps, the others over 50.
- Gradients are clipped per group (norm 1). Clipping does not stop the Adam first-step jump, because Adam moves each entry by about lr whatever the gradient size. The normalized write path stops it.
- Pass schedule: one pass with probability 0.15, else three passes with probability 0.3 once step >= 100, else two. The read layer then sees fed-back inputs in training. Two random draws are used every step, so every arm keeps the same schedule.
- Halt: the first step with two or more passes sets a baseline KD. If the KD on such steps is above 3 times max(baseline, 0.05), or not finite, for 20 such steps in a row, the loop stops and `log["halted"]` is true. Round 22 marks the cell `failed_heal` and runs no benchmark.
- `passes_gap` (KL from the exact recursion to passes=2, and top-1 agreement) on a held-out batch goes to `log["gap"]` every 200 steps.

## What round 22 measures

Pairs: A (1.5B host, 7B donor), B (7B host, 32B donor), C (0.5B host, 7B donor). Arms: base, kd_lora, feedback_plain, feedback_block, feedback_block_random, midstack_block, midstack_block_random. Two seeds for each trained arm. Every row has `status` (`ok`, `failed_stage1`, `failed_heal`), `stage1_penalty`, `gate`, `written_ratio` (norm of the written vector over `embed_rms`, equal to the gate), `warm_start`, `passes_gap`, and the `wsp.probe` report on held-out pool text (transfer per skill against the donor, drift from the frozen base), benchmark scores through `wsp.skills_eval` once (GSM8K 300, MBPP 200, MMLU held-out 1000, MedMCQA 1000), parameters total and run per token, heal seconds, and decode tokens per second. `go_verdict` applies the go rule: feedback_block must beat kd_lora, feedback_plain, and feedback_block_random on mean probe transfer by more than the seed spread, with drift under 0.05 nats. The docstring of `modal_round22.py` says what each other outcome means.

Parameters run per token count the branch layers in full, because the component runs at every step. The input embedding table is not counted, unless the head shares it.

Smoke command (one cell, pair A, seed 0):

    modal run modal_round22.py --only feedback_block --pairs A --seeds 0

A fan-out comes only after that cell prints a row with status `ok`.

## What failed in the first real run (2026-11-14) and what changed

Cell: pair A (base Qwen2.5-1.5B-Instruct, donor Qwen2.5-7B-Instruct layers 8..20), arm feedback_block, seed 0. It destroyed the model: GSM8K 0.0, MBPP 0.0, MMLU 25.9 (chance), probe transfer -2.49, drift 2.76 nats.

1. Stage 1 was too weak. Ridge r2 was 0.71. The donor's loss through T_in was 7.94 before training and 3.72 after 400 steps, against 1.74 native: a 1.98 nat penalty, so the block was fed garbage. The stitch study reached 0.03 nats. The differences: the study used 1000 steps (800 in run 14) on 2400 texts, an MLP stitch (lr 3e-4) or a linear one (lr 1e-3), and ran between Qwen2.5-1.5B and Qwen2.5-Math-1.5B, which share an initialization and a width of 1536. Round 22 used a linear stitch, 400 steps, 256 texts, and a 1536 to 3584 map into a bigger model. Changes: MLP stitch, 1000 steps, 1200 texts, extra rounds while the penalty is above 0.3, and a hard refusal to start stage 2 above the threshold. The study's 0.03 does not carry over to a different-width, different-size donor, so the threshold may stay unmet on pair A. Then the cell ends as `failed_stage1`, which is a real result about that pair.
2. One optimizer step destroyed the model. The first heal steps gave KD 0.64, then 7.94, 7.31, 6.63 on two-pass steps, while one-pass steps stayed near 0.4 to 0.5. T_out started at zero with the gate at 1 and the block output was not normalized. Adam moves each T_out entry by about the learning rate on step 1, so the vector added to the next embedding jumped from zero to a norm set by the raw block output, which in Qwen has huge outlier dimensions. On a tiny model (CPU test) one Adam step at lr 3e-4 from a fresh seam gave a written norm of 0.01, 0.06, and 0.67 times the embedding norm when the block output was scaled by 1, 10, and 100, and the largest logit change rose from 0.006 to 0.05 to 0.38. The new write path gives the same tiny change at every scale (written norm equals the gate, about 5e-4 after one step). Cause confirmed on the tiny model; the real 1.5B run was not repeated.
3. Training and exact generation were far apart. After the heal, `passes_gap` was KL 6.18 with 4 percent top-1 agreement. Two passes with clean reads in training do not match a recursion in which the read layer sees earlier feedback. Changes: three-pass steps (30 percent), T_in fine-tuned in stage 2, and `passes_gap` logged every 200 steps. If the gap stays large after the heal, exact scoring and teacher-forced scoring still disagree, and the heal needs more passes or a different schedule.

## What is unverified

- Nothing ran on a GPU. The new defaults (MLP stitch, normalized write path, warm start, three passes, halt rule) have not been run on a real cell. The tests use tiny random models on CPU. The Modal script is parse-checked and imports, but no cell has run. Memory and time figures are estimates.
- I did not read the paper. The mechanism and the pass schedule (one pass with probability 0.15, else two; the graph only through the last pass) come from the task brief.
- The read path sees its own feedback. T_in is fitted on host states with no feedback. Pass 2 reads clean pass-1 states, but in the exact recursion the read state at layer k contains the feedback of earlier tokens, and its size grows as T_out grows. Training with passes of 1 and 2 may not carry over to exact generation. `passes_gap` in each row measures how far apart the two are.
- Learning rates (T_out 3e-4, gate 1e-3, LoRA 2e-4, T_in 1e-5), g_max 0.5, the warm-start target, the halt ratio 3, and the 0.3 stage-1 threshold are first guesses.
- With the write path normalized, the feedback norm is capped at g_max times the embedding norm. If a real block needs a larger push, g_max must go up.
- The planted test shows the mechanism at toy scale (a donor trained to output the sum of its last two tokens, a host that cannot do it alone). It does not show that a real donor block carries a skill.
- The midstack arms start the gate at 0 and stitch_out at the pseudo-inverse of T_in, with their own learning rates. A different recipe could change that contrast.
- Exact generation is slow because the prompt runs one token at a time. `decode_tok_s` times 64 new tokens after a short prompt, so it hides that cost for long prompts.
