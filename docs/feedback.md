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
- `T_out`: a bias-free linear map to host width. It starts at zero.
- `gate`: one scalar. It starts at 1.

For position t the branch makes

    fb_t = gate * T_out(block(x_t) - x_t),  x_t = T_in(h_t at layer k)      with a component
    fb_t = gate * T_out(top state at t)                                      component None

and the host input at position t+1 is `embed(token_{t+1}) + sum of fb_t over branches`. Position 0 gets nothing. For a left-padded row, the first real token gets nothing, and a pad never feeds a real token.

At the first real position the component input is replaced by the donor's sink state (the attention-sink fix from `wsp/graft.py`). That fix is the main reason the stitch is lossless.

Why the gate starts at 1: with T_out at zero and the gate at zero, neither gets a gradient. With T_out at zero and the gate at 1, a fresh seam equals the base exactly and T_out has a gradient from the first step.

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

Stage 1, `fit_t_in`: T_in starts from a ridge fit on matched states and trains on the donor's own LM loss with the donor sink (`wsp.stitch.train_stitch`, objective `donor_lm`). This trains T_in so that the donor, reading host layer k through it, keeps its own loss. The loss runs through the donor's whole upper stack, not only the block. Host and donor stay frozen.

Stage 2, `heal_distill` (`heal_seam` for a seam): T_out, the gates, and a host LoRA r16 train on

    KL(donor || seam) on skill pool text + 0.5 * KL(frozen base || seam) on general pool text.

There is no LM loss on any text. The frozen base is the host with the LoRA scales set to 0 (`base_logits`), so no second copy is loaded. Position 0 is left out of every KL, as in `wsp.probe`. The pass count is 1 with probability 0.15, else 2. The same text order is used by every arm. The host LoRA covers only the layers above the read layer: a LoRA at or below it would move the state T_in reads.

## What round 22 measures

Pairs: A (1.5B host, 7B donor), B (7B host, 32B donor), C (0.5B host, 7B donor). Arms: base, kd_lora, feedback_plain, feedback_block, feedback_block_random, midstack_block, midstack_block_random. Two seeds for each trained arm. Every row has the `wsp.probe` report on held-out pool text (transfer per skill against the donor, drift from the frozen base), benchmark scores through `wsp.skills_eval` once (GSM8K 300, MBPP 200, MMLU held-out 1000, MedMCQA 1000), parameters total and run per token, heal seconds, and decode tokens per second. `go_verdict` applies the go rule: feedback_block must beat kd_lora, feedback_plain, and feedback_block_random on mean probe transfer by more than the seed spread, with drift under 0.05 nats. The docstring of `modal_round22.py` says what each other outcome means.

Parameters run per token count the branch layers in full, because the component runs at every step. The input embedding table is not counted, unless the head shares it.

## What is unverified

- Nothing ran on a GPU. The tests use tiny random models on CPU. The Modal script is parse-checked and imports, but no cell has run. Memory and time figures are estimates.
- I did not read the paper. The mechanism and the pass schedule (one pass with probability 0.15, else two; the graph only through the last pass) come from the task brief.
- The read path sees its own feedback. T_in is fitted on host states with no feedback. Pass 2 reads clean pass-1 states, but in the exact recursion the read state at layer k contains the feedback of earlier tokens, and its size grows as T_out grows. Training with passes of 1 and 2 may not carry over to exact generation. `passes_gap` in each row measures how far apart the two are.
- Learning rates (T_out 3e-4, gate 1e-3, LoRA 2e-4) are first guesses.
- The planted test shows the mechanism at toy scale (a donor trained to output the sum of its last two tokens, a host that cannot do it alone). It does not show that a real donor block carries a skill.
- The midstack arms start the gate at 0 and stitch_out at the pseudo-inverse of T_in, with their own learning rates. A different recipe could change that contrast.
- Exact generation is slow because the prompt runs one token at a time. `decode_tok_s` times 64 new tokens after a short prompt, so it hides that cost for long prompts.
