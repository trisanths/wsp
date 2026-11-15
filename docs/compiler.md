# The synthesis compiler

Code: `wsp/synth.py`. Tests: `tests/test_synth.py`. Round script: `modal_round20.py`.

Other agents mine components from donor models (blocks, attention heads, MLP clusters, experts). The compiler takes those components and builds ONE network from them. Every seam has a learned translation. A short heal trains the translations, the gates, and a small LoRA. Nothing but the one network runs at inference: no taps, no parallel donor.

```python
from wsp import synth
model, tokenizer, report = synthesize(base, components, texts, budget,
                                      practice=..., guard=..., donors=..., out_dir=...)
model, tokenizer, report = synth.load_synth(out_dir)   # same logits
```

A component is a dict. `kind` is `block`, `attn_heads`, `mlp_cluster`, or `expert`. `donor` is the donor model id. `layers` is `[first, last]` for a block. `layer` and `indices` (heads, neuron units, or expert ids) locate the other kinds. `skill` names the skill. `texts` are domain texts for that skill. `host_layer` is a host layer index or `"auto"` (same relative depth). `control: "random"` builds the same shape from random same-size parts of the same donor, with `control_seed`. `control: "scrambled"` keeps the same donor layers or units and rewrites their weights with `wsp.scramble` (seeded by `control_seed`) before any stitch is fit. `control_mode` is `"spectrum"` (default) or `"permute"`. Rows of `catalog.jsonl` and `library/catalog.jsonl` are accepted as they are (`type`, `model`, `layer`, `units`, `layers`, `skill`).

Heal data rule (owner, 2026-11-14): nothing that trains or fits anything may use a reported benchmark in any split. The caller passes texts from the `wsp.healdata` pools. The practice sets of stage 6 never receive gradient.

## Notation

The host has layers `1..L` and width `d_h`. A component `c` sits at host layer `k_c`. The host residual after layer `k_c` is `h`. The donor input point has width `d_d`.

    branch_c(h) = h + r * g(h) * mask_0 * T_out(P_c(sink(T_in(h))))

`P_c` is the frozen donor part (for a block, `block(x) - x`). `sink` puts the donor's mean position-0 state at each sequence's first real token. `g(h) = sigmoid(w . rms(h) + b)` is the token gate (`wsp/gate.py`). `mask_0` zeroes the delta at the first real position, so the host keeps its own sink state. `r` is the ramp, a plain float.

## The seven stages

### 1. Place

Each component becomes a `Branch` (a subclass of `wsp.graft.Graft`, so `attach_grafts` keeps the KV caches of attention parts in step with the host cache). Every branch starts closed (`r = 0`), so

    logits(fresh network) = logits(base)   exactly.

The report holds `place`: the host layer and the part parameter count of every component.

### 2. Translate in

`T_in: R^{d_h} -> R^{d_d}`, linear by default, or a small MLP (`t_in_kind: "mlp"`).

Init: ridge fit on paired states of the same text (`wsp.align.collect_pairs`, `ridge_fit`).

    T_in = argmin_W  sum_t || W h_t - x_t ||^2 + lam ||W||^2

where `x_t` is the donor state at the component's input point. Then, when the host and donor tokenizers match, `T_in` trains to make the component's output on the translated input match its output on the native donor input:

    L_in = mean_{t > 0} ( (P(sink(T_in(h_t))) - P(x_t)) / std )^2

For blocks, `T_in` also trains on the donor's own LM loss through the donor tail (`wsp.stitch.train_stitch`, objective `donor_lm`, with the sink). Report fields, per component under `translate_in`:

- `ridge_r2`, `ridge_r2_dim`: fit quality of the ridge map on held-out rows.
- `rel_err_ridge`, `rel_err`: `||P(T_in h) - P(x)||^2 / ||P(x)||^2` on held-out text, before and after training.
- `donor_lm_penalty_ridge`, `donor_lm_penalty`: donor LM loss with the translated input minus its native loss, in nats (blocks only).
- `trained: false` and a note when the tokenizers differ (ridge only).

### 3. Translate out, then the stage-1 heal

`T_out: R^{d_d} -> R^{d_h}`. Two inits, both fit to what the host is missing, never to the host's own state.

Logit-preserving map. It applies when the tokenizers match and the block ends within `near_top` layers of the donor's last layer. With `E_h`, `E_d` the unembeddings times the final norm gains, centered over the vocabulary, and `W` the unigram weights:

    M = argmin_M || W^{1/2} (E_h M - E_d) ||^2 + lam ||M||^2
    T_out = (rms_h / rms_d) M

`M` maps a unit-RMS donor final state to a host final state with the same logits, as far as rank `d_h` allows. `rms_h` and `rms_d` are the mean final-state RMS of the host and of the donor at the block exit. Report: `init: "unembed_map"`, `logit_capture` (the share of the donor's centered logit variance that `M` keeps), `rms_ratio`.

Missing residual. Otherwise `T_out` is a ridge fit from the part output to the residual the host lacks at its top. With the host's final state `z` and logits `l_h`:

    gap = (l_d - mean l_d) - (l_h - mean l_h)        donor logits, same tokenizer
    delta* = rms(z) * (E_h^T W E_h + lam I)^{-1} E_h^T W gap
    T_out = argmin_W || W P(sink(T_in h)) - delta* ||^2 + lam ||W||^2

With different tokenizers there are no donor logits to compare, so the fit is skipped: `T_out` stays zero and the report says `init: "zero_no_teacher"`. (`missing_residual_fit` still has a one-step gold-label gradient path for direct calls. It is not a teacher signal and `translate_out` does not use it.) The near-top test reads the last layer of the real block, also for a random control, so a control gets the same init as the part it replaces.

Then `T_out` is scaled so the delta has RMS `out_alpha` (0.3) times the RMS of the host's own update at that depth (`out_scale`, `host_update_rms`). Report: `init`, `fit_r2`, `fit_rows`, `out_rms_before`, `out_scale`.

`output_init: "old"` keeps the zero init and drops function matching, to measure what the new output side buys.

Stage-1 heal. Trains only `T_in`, `T_out`, and the token gates, through the frozen host tail. The ramp goes from 0 to 1 over the first `ramp_frac` of the steps. Per step, for each component `c` with domain batch `B_c`:

    L1 = sum_c [ kd * KL(p_donor || p_student)(B_c) + lm_kd * LM(B_c) + feat * F_c(B_c) ]  / n_c
         + pres * KL(p_base || p_student)(B_general)

- `KL(p_donor || p_student)` is function matching: the donor's logits on the component's domain text. It applies when the tokenizers match. Otherwise the term is `LM(B_c)`.
- `F_c` is feature matching at the seam: the standardized squared error between the part output on the translated input and on the native donor input.
- The last term is the preservation term of stage 4, active in stage 1 too.

AdamW, 5% warmup, cosine decay to zero.

### 4. Synthesis heal (stage 2)

Adds LoRA (rank `lora_rank`) on the host layers after the first seam, `k_first + 1 .. L`. It trains the LoRA together with the translations and gates, at lower learning rates (LoRA at most 5e-5).

    L2 = L1' + 0   with the same terms as L1, without F_c

Rules for the data (from the heal-harm audit of run 16b):

- Loss on answer tokens only when an item is a prompt and answer pair. Loss is averaged over answer tokens per example first, then over the batch.
- Items that do not fit `seq_len` are dropped, never cut (counted in `data.dropped`).
- Texts are the donor's own answers (domain) or the base's own answers (general), never dataset gold answers in a benchmark's style.
- Preservation: `pres * KL(p_base || p_student)` on general text, where `p_base` is the network with all branches and the LoRA switched off (`frozen_base`). It is exactly zero for a network that equals the base.

Report: `stage1` and `stage2`: `steps`, `seconds`, `task_start`, `task_end`, `pres_start`, `pres_end`, `feat_start`, `feat_end`, a thinned `losses` list, and for stage 2 `lora_delta_ratio` (norm of all LoRA deltas over the norm of the weights they sit on) and `lora_layers`.

### 5. Gates

Each component has a `TokenGate` (`wsp/gate.py`), init bias `gate_bias` (open, 2.0), trained in both heals. The preservation term pushes it toward closed on general text. There is no large closed bias to unlearn. Report `gates`: the mean token-gate value of each component on each text set (`general`, and `domain:<id>`), plus the final `ramp` and `closed` flag per component.

### 6. Verify

The caller gives small practice sets, one per skill (`practice={skill: {"items": [...]}}`, or a callable `(model, tokenizer) -> score in [0, 1]`). They are disjoint from the heal data and from any final test. They never receive gradient.

- Scores of the base, after stage 1, after stage 2, and final: `scores`.
- Guard. If a guarded skill (default: every practice skill) falls more than `tolerance` below the base:
  - a failing stage 2 rolls back to the stage-1 weights and removes the LoRA (`guard.events`, action `rolled back to stage 1`);
  - a failing stage 1 closes the offending gates one at a time (the gate whose closing leaves the best worst-case margin first), down to the base if needed (action `closed gates`).
- Component test: score with each component's gate forced to 0. `contribution = final - without` on the component's own skill, and `scores_without` for every skill. `text_gain` is the held-out LM loss with the component off minus on. Without a practice set for the skill, the verdict uses `text_gain`.
- Controls (`controls=["random", "scrambled"]`, or `control_run=True` for `["random"]`): the same recipe, same seed and budget, once per control kind. `random` uses another range of the same donor, so it tests the choice of layers. `scrambled` uses the same layers with scrambled weights (same shapes, parameter count, norms, and singular values, no learned structure), so it tests whether the donor's knowledge is used at all. The report has `controls[kind]` (scores, gain, gates, guard, and for scrambled the scramble report) and per component `content_gain_by_control`. `content_gain = score(real) - score(control)` on the component's skill. The old fields stay: `random_control` is `controls["random"]`, and `content_gain` is the random gain when a random control was built (else the first control's).
- Verdict per component:
  - `harmful`: contribution below `-margin`, or the guard closed the gate;
  - `no better than random`: contribution at most `margin`, or the real component beats neither control by more than `margin`;
  - `contributes`: otherwise. With a scrambled control built, the real component must beat it;
  - `donor helps, choice does not matter`: beats scrambled but not random;
  - `no better than scrambled`: beats random but not scrambled, or only a scrambled control was built and it was not beaten.
  Without a control run, the verdict rests on the contribution alone.

### 7. Budget

`params.total` counts every distinct parameter (base, parts, translations, gates, LoRA). `params.per_token` removes the input embedding table when the head is not tied (`wsp.skills_eval.model_params`); every branch counts as run for every token. `max_params` and `max_params_per_token` in the budget refuse a compile before any heal runs, with the measured numbers (`BudgetError`). The LoRA that stage 2 will add is included in the check.

## Other budget keys

`stage1_steps`, `stage2_steps`, `batch_size`, `seq_len`, `lora_rank`, `steps_in`, `steps_lm`, `lr_*`, `kd_weight`, `lm_weight_kd`, `pres_weight`, `feat_weight`, `tolerance`, `margin`, `near_top`, `out_alpha`, `kd` (False: LM loss only), `components_per_step` (train k components per step, round robin, for many parts), `test_limit` (practice items per skill in the component test), `component_tests`, `control_run`, `seed`. Defaults are in `synth.DEFAULTS`.

`synth.lora_baseline` is the competitor arm: LoRA on every layer, the same data, steps, preservation term, and schedule, with or without the donor KL.

## Saved format

`out_dir/synth.json` (manifest and report), `parts.safetensors` (the donor tensors of every part, under the donor checkpoint names), `glue.pt` (translations, gates, sink, LoRA), `tokenizer/`, and `base/` unless the base is a hub id. `load_synth` rebuilds the parts from the saved tensors and the donor config; it does not fetch donor weights.

## What is unverified

- Everything here runs on tiny random models on CPU. No Modal run exists yet. Round 20 is the first test on Qwen models.
- Whether function matching through a frozen host tail teaches `T_out` more than LM loss does is the claim of the forensic audit, and it is measured only by the planted tiny case in `tests/test_synth.py`.
- The logit-preserving init is exact only for a block that ends at the donor's last layer. For an earlier exit it is a logit-lens start. The `near_top` rule is a guess (2 layers).
- Late re-entry (host layers left to read the branch) is untested at scale. Round 20 `synth_top` compares it with mid-stack at one setting.
- Practice sets of 100 to 150 items have a standard error near 4 points, so `tolerance` 0.04 can miss a real 3 point harm. The guard is a floor, not a proof.
- The preservation term uses the base's greedy answers without a correctness filter (the forensic rule asks for one; a gold-label filter would need benchmark data).
- Position-0 masking and the sink fix are carried over from the input side without an ablation on the output side.
- Verdicts without a control run do not compare against random or scrambled.
- Scrambled and random controls share the donor-logit teacher with the real run (KD and the missing-residual fit for T_out). A scrambled control therefore measures what the stitch, glue, and teacher give without the block's weights. On a tiny planted case the compiled score came mostly from this, and the scrambled control matched the real block.
- Experts (`kind: "expert"`) are built and tested for the part, not run in a heal. Their token gate stands in for a router.
- The LoRA baseline arms use the preservation term too. That makes them stronger than the round 16b LoRA, which had none.

## Random control for blocks

`control_block_range` picks the control range from the starts with the least overlap with the real block, seeded. When the donor has room, the overlap is 0. When it has none (a 13-layer block in a 28-layer donor), the least overlap is 5 layers and the report keeps it as `control_overlap`, so a reader sees that the control is not disjoint there. The `synth` arm of round 20 sets `control_seed` to the cell seed, like `synth_random`.

## Round 20 rows

Rows hold `trainable` (glue, gates, and LoRA for synth arms; LoRA for the lora arms), so the go rule can compare trainable counts. In p3 the library adds one T_in and one T_out per row, so synth trains far more parameters than the rank-16 LoRA arms there. Read p3 with that in mind. The guard of both compile and LoRA arms is the pair's guard skill only. The library takes rows of tier `screened` only, since provisional rows were chosen by ablation on GSM8K.

## Scrambled control

`wsp/scramble.py` rewrites weights so no learned structure remains. `spectrum` mode makes W into U diag(s) V^T with s the singular values of W and U, V Haar-random orthonormal bases (QR of a seeded Gaussian with the sign fix). Shape, Frobenius norm, and spectrum stay. `permute` mode permutes all entries of a tensor, so the value histogram and norm stay and the spectrum is gone. 1-D tensors (norm weights, biases) are always permuted. Embedding tables are not touched unless `include_embeddings=True`.

Only the singular values of W are computed (`torch.linalg.svdvals`, float32, on the tensor's device), one tensor at a time. A matrix with more entries than `scramble_max_numel` uses `permute`. Random draws come from a CPU generator, so a seed gives the same result on any device. `scramble_module(module, seed, mode)` works in place and returns a report: `tensors_changed`, `rel_fro_error` (largest relative change of a tensor norm, about 0), `mean_abs_cosine` (about 0), `seconds`. In the compiler, `build_part` scrambles a copy of the donor tensors, so the donor model is not changed. During the stage-2 donor-LM stitch fit the scrambled tensors are swapped into the donor model and restored after, so T_in is not fit to the real block.
