# Transfer methods sweep, mapped onto our runs (2026-11-15)

## Plan

# Sapien next experiment plan, 2026-11-15

I read the round scripts (18, 20c, 21, 22, 24), README runs 14 to 19, LESSONS.md, and the team memory. I ran nothing and edited nothing. The Hub ids below come from the round scripts and the brief, and none are re-checked. Run `wsp/lineage.py` gates before any cell.

Every experiment obeys the standing rules.
- No benchmark trains or selects anything. Pools come from `wsp.healdata`, and benchmarks run through `wsp.skills_eval`.
- Every table has the base, a scrambled control (`wsp/scramble.py`), a matched-count LoRA distillation, and the best downloadable model of the same size.
- Use at least 2 seeds on trained arms. Print the cell name on every score line.
- Run one smoke cell first. Launch with `modal run --detach`, and key cached artifacts by code version.
- Reasoning members (OpenThinker, Nemotron) write long chains. Raise the token limit and read raw generations before scoring.

## What the literature changes in our plan

- Every published block or layer move that works shares an initialization. Layer swapping, DFS, SOLAR, diff transfer, and BTX are all in this group.
- We have exactly one working family: true fine-tunes of one host (cosine near 1) and MoE experts of one model.
- The cheapest unexploited fact is from Bandarkar. Reverting the first 5 and last 2 layers of a fine-tune lost no skill. So a skill sits in a subset of layers, and we should find that subset directly instead of searching mix weights.
- Round 21 searches mix weights per bucket. It does not measure the minimal layer set, and it does not test two skills on disjoint layer sets. Experiments 1 and 2 do.

## Ranking

| # | Experiment | Chance of a clear win | Smoke cost | Full cost |
|---|---|---|---|---|
| 1 | Minimal layer set per skill, then disjoint composition | 0.55 | 1 GPU-h | 8 GPU-h |
| 2 | Identity-seam skill tops with a closed-form router | 0.5 | 1.5 GPU-h | 10 GPU-h |
| 3 | Union-saliency MoE keep, router KD, novice constants | 0.45 | 2 GPU-h | 12 GPU-h |
| 4 | Rotation test for far-lineage specialists | 0.15 | 0.3 GPU-h | 1 GPU-h |
| 5 | Layer-wise downstream loss in the reader study | 0.25 | 3 GPU-h | 14 GPU-h |

## Experiment 1: minimal layer set per skill, then compose two skills

**Hypothesis.** A true fine-tune keeps its skill in a subset of layers. Two members with disjoint subsets can be written into one host-sized network, with each skill at 95% of its member.

**Borrows.** Layer swapping (Bandarkar et al., ICLR 2025, arXiv 2410.01335): revert layers to the pretrained values, and keep only the layers that carry the task. Fine-tuning transfer (arXiv 2503.20110) bounds it to the same lineage.

**Models.**
- Host: `Qwen/Qwen2.5-7B-Instruct`.
- Members: `open-thoughts/OpenThinker-7B` and `nvidia/OpenCodeReasoning-Nemotron-7B`.
- Both pass the weave gate (`weave.gate_member`, cosine at least 0.97) before use.

**Benchmarks.**
- Selection signal: the benchmark-free probe of round 21 (split c for the final number).
- Scored once: MBPP 257, HumanEval 164, GSM8K 300, MMLU 1000.

**Method.**
1. Cache host and member probe logits with `weave.build_teacher_cache`.
2. Revert each layer alone (28 evals) and rank layers by loss in probe transfer.
3. Bisect the prefix sets of that ranking to get the smallest set S that keeps 90% of the member's transfer.
4. Repeat for the second member.
5. Compose by taking each member's layers on its own S. Break overlaps by larger delta norm.

**Controls.**
- Host.
- Each member alone.
- Random layer set of equal size from the same member.
- Layer set chosen by delta norm only, with no probe.
- The code member's layers on S taken from the reasoning member (the cross-skill control).
- Scrambled delta (member minus host) on S.
- TIES and task arithmetic at the same total size.
- The round 21 CMA-ES result at matched size.
- Best downloadable same-size model: `Qwen/Qwen2.5-Coder-7B-Instruct` for code.

**One-cell smoke.**
- One cell is host plus the code member, with one GPU for about 50 minutes.
- Cache, then rank the layers and bisect for S (about 10 minutes).
- Score four networks on MBPP and probe split c: member, host, swap on S, random swap of the same size.

**Go.**
- |S| is at most 50% of layers.
- The swap keeps at least 90% of the member's probe transfer and is within 5 MBPP points of the member.
- The random set keeps less than 50% of the transfer.

**Stop.** No S under 75% of layers holds 90% of the transfer. Then layer-level skills are spread out, and round 21's mix search is the ceiling.

**Cost.** Smoke 1 GPU-h. Full run (two members, two hosts of different size, 2 seeds on the controls) about 8 GPU-h.

**Code.** New `wsp/layerset.py`. It reuses `weave.Weaver` with `n_buckets` equal to the layer count, `weave.one_hot_genome`, `weave.ProbeObjective`, and `weave.kl_to_cache`. Add a modal script `modal_round25.py` modeled on `modal_round21.py`.

## Experiment 2: identity-seam skill tops with a closed-form router

**Hypothesis.** For true fine-tunes the stitch in skill tops is the identity, so no stitch can drift. Routing per sequence between the host top and each member's top holds each skill at member level, and the extra storage is only the tops.

**Borrows.**
- Round 24 skill tops.
- Closed-form ridge router from DUME (Fani and Ersoy, arXiv 2603.29765).
- Bandarkar's result that lower layers are revertible.
- Round 24's smoke saw a trained stitch drift from the donor. This design removes that failure for the true-fine-tune case.

**Models.** Same as Experiment 1. Cuts at 1/4, 1/2, and 3/4 of 28 layers (7, 14, 21).

**Method.**
- Trunk is the host's layers 0 to cut-1.
- Each top is a member's layers from cut, its norm, and its head.
- No training.
- Router: pool mean of the trunk output at the cut, ridge fit on pool-labeled texts (`healdata`), lambda 0.01.

**Controls.**
- Trunk from the host versus the member's own bottom at the same cut.
- Forced top against the member alone.
- Random-routing, oracle-routing, and host-only arms.
- A scrambled top.
- The cross-skill top.
- The weave result at matched size.
- A kd_lora arm: the member distilled into LoRA on the host at matched trainable count.
- `Qwen/Qwen2.5-Coder-7B-Instruct`.

**One-cell smoke.**
- Forced code top at cut 14, MBPP 257 and probe split c, compared with the member and the host.
- About 40 minutes.

**Go.**
- Forced top is within 3 MBPP points of the member at some cut at or below 1/2.
- On probe split c the ridge router picks the right top for at least 95% of sequences, and routed benchmark scores are within 3 points of oracle.

**Stop.** The forced top loses more than 10 points at every cut up to 3/4. Then even true fine-tunes need a trained seam, and the identity-seam variant is dead.

**Cost.** Smoke 1.5 GPU-h. Full run (3 cuts, 2 members, router, 2 seeds on trained controls) about 10 GPU-h.

**Code.** Extend `wsp/tops.py` with `seam="identity"` and a closed-form router fit. First check how `tops.py` and `modal_round24.py` fit their router. Experiment 1's S gives the best cut.

## Experiment 3: union-saliency MoE keep, router KD, novice constants

**Hypothesis.** Saliency over the union of two skills keeps both at 64 of 128 experts. Router-only KD recovers part of what no-heal slimming loses.

**Borrows.**
- Run 19 and REAP (arXiv 2510.13999).
- Router KD (arXiv 2603.02217): about 0.04% of parameters, a few million tokens.
- MoNE (arXiv 2507.00390): replace each deleted expert by its mean output.

**Model.** `Qwen/Qwen3-30B-A3B-Instruct-2507`.

**Benchmarks.** HumanEval 164, MBPP 257, GSM8K 200, MMLU 1000.

**Arms.**
- Keep 64 by code saliency, from run 19: the existing cell.
- Keep 64 by math saliency.
- Keep 64 by union of code and math saliency.
- Keep 80 by union of code, math, and general-knowledge text.
- Add router KD (375 steps, lr 5e-5, C4-style pool text) to the union cell.
- Add the novice constant to the union cell.

**Controls.**
- Full model, plus the full model with the same router KD (heal control).
- Random keep, 3 seeds.
- Frequency keep.
- The run 19 code-only 64.
- `Qwen/Qwen3-4B-Instruct-2507`, the best downloadable model near the 3B active size.

**One-cell smoke.**
- Union 64 with no heal, scored on HumanEval and GSM8K through vLLM (`modal_serve.py`), because the HF MoE loop does 7 to 10 tok/s.
- Stats come from `slim.collect_moe_stats` on 96 math texts. Code stats are cached.
- About 75 minutes.

**Go.**
- Union 64 is within 5 points of full on both HumanEval and GSM8K with no heal.
- Router KD adds at least 3 MMLU or GSM8K points over the no-heal cell, and the full-model heal control stays flat.

**Stop.** Union 64 loses more than 10 points on either skill. Then the skills share experts, the needed K grows with each skill, and the saving is under 2x. Drop router KD at 32 or 16 experts, where collapse is known.

**Cost.** Smoke 2 GPU-h. Full run about 12 GPU-h.

**Code.** `slim.expert_scores` and `slim.rank_experts` already take a skill and general stats. Add a union rule and a novice option in `wsp/slim.py`, and a router-only KD loop beside `slim.heal`.

## Experiment 4: rotation test for far-lineage specialists

**Hypothesis.** Qwen2.5-Math and -Coder may live in a rotated residual coordinate system relative to Instruct. Their weight cosine of 0.2 to 0.5 would then hide a function-preserving rotation, and one orthogonal map would restore the 16c cells.

**Borrows.** Residual-stream rotation is what closed the loss barrier (Generalized LMC, arXiv 2506.22712). Permutations alone did not. Horoi et al. (arXiv 2511.10850) found the best permutations were identity and rotation-only worked as well as the full method. Li and Shen (arXiv 2606.23607) leave a barrier of 0.18 to 1.53 nats even after learned alignment, which sets the expected ceiling.

**Models.** `Qwen/Qwen2.5-1.5B-Instruct`, `Qwen/Qwen2.5-Math-1.5B-Instruct`, `Qwen/Qwen2.5-Coder-1.5B-Instruct`.

**Method.**
1. Compute orthogonal Procrustes R between the two token-embedding matrices. The vocabularies are shared, and the 1.5B models tie embeddings.
2. Report the relative residual ||E_m R - E_i|| / ||E_i||, against the unrotated residual and against a token-shuffled R.
3. If the rotated residual is below about 0.3, fold the RMSNorm weights into the following linear layers and rotate every residual reader and writer by R.
4. Rerun the 16c cells (all MLPs, whole layers, no training) on the rotated Math and Coder.

**Controls.**
- Unrotated 16c cells.
- Random orthogonal R.
- Token-shuffled R.
- Specialist alone.
- Base.
- Note that 16c MLP-only swaps fail under the base's own attention, so rope_theta (10000 against 1e6) is not the cause for those cells.

**One-cell smoke.**
- The Procrustes residual is seconds of work.
- If it passes, one cell: all-MLPs-from-Math on rotated weights, MMLU 1000 plus GSM8K 300.
- About 20 minutes.

**Go.** The rotated residual is below 0.3 and the rotated Math MLP cell scores at least 15 points above chance on GSM8K, or the rotated cell is at least 10 MMLU points above the 25 chance rate, with the random-R control still at chance.

**Stop.** Rotated residual above 0.8. Then the specialists have independent coordinates, and every no-training cross-lineage mix and Symphony-style permutation alignment stops.

**Cost.** Smoke 0.3 GPU-h. Full run 1 GPU-h.

**Code.** New `wsp/rotate.py` with `fold_norms`, `procrustes`, `rotate_residual`. Test fold and rotation on the same model first (output must match at float32) before any cross-model use.

## Experiment 5: layer-wise downstream loss in the reader study

**Hypothesis.** A short heal fails because only the final logits supervise the host's layers after the seam. A per-layer cosine loss against the donor's own hidden states teaches those layers to read the written block in far fewer steps.

**Borrows.**
- Boomerang distillation (arXiv 2510.05064): the layer-output cosine loss is what makes a slot accept the original layer.
- Puzzle (arXiv 2411.19146): local block distillation.
- SemAlign (arXiv 2510.24208): frontier-layer schedule.
- Minitron: the loss works as KL on logits plus intermediate states.

**Models.**
- Pair R2 of `modal_round20c.py`: base `Qwen/Qwen2.5-7B-Instruct`, donor `Qwen/Qwen2.5-Coder-7B-Instruct`.
- Block is 8 layers. Re-entry is mid-stack.

**Method.**
- The teacher hidden states at each host layer after the seam come from the donor at matching depth.
- A closed-form orthogonal Procrustes map per layer, fit on 2000 pool tokens, translates between coordinates.
- Loss is the sum of per-layer (1 - cosine), plus the existing donor KL and the preservation term.

**Arms.**
- Real block with logit-only KD (round 20c default).
- Real block with per-layer loss.
- Scrambled block with per-layer loss.
- kd_lora at matched trainable count with the per-layer loss.
- Base.
- `Qwen/Qwen2.5-Coder-7B-Instruct`.
- The best downloadable model at the compiled size, Qwen2.5-Coder 3B or 7B.

**One-cell smoke.** The default cell at 500 stage-2 steps, real block with and without the per-layer loss. About 55 minutes each, so two cells. Score the probe transfer curve at 100-step intervals and MBPP 257 at the end.

**Go.** The per-layer arm reaches the logit-only arm's final probe transfer in at most half the steps, and its MBPP beats scrambled and kd_lora by at least 5 points on 2 seeds.

**Stop.** The scrambled arm matches the real arm. That is the 14b pattern, where depth supplies the gain and not the donor's skill.

**Cost.** Smoke 3 GPU-h. Full run about 14 GPU-h.

**Code.** Extend `wsp/reader.py` and add a teacher-signal option "layerwise" to factor e in `modal_round20c.py`.

## Methods to stop

- **Writing a donor block back into the host's upper layers across lineages.** Runs 14, 14b, 20c stage 1, and 22 show the random block gains as much as the real one. The Pythia negative result and Ramesh and Li agree: the receiver cannot read it.
- **Feedback seam beyond the in-flight run.** Coconut and the latent feedback transformers train from scratch. The Coconut ablation without its curriculum scored 14.4, below No-CoT at 16.5. Spend no more unless round 22 beats its scrambled control.
- **Data-flow path search across lineages.** DFS needs one lineage. Our trained stitches already play the role of its scale matrix W, and the search signal overfits at dev size 1000.
- **Cross-lineage MLP, layer, or TIES mixing.** Run 16c is chance. Stop all of it unless Experiment 4 passes.
- **Permutation-only alignment (Symphony-MoE) as the first fix.** Horoi found the best permutations were identity.
- **Dense layer carving or width slimming beyond about 20 to 25% of depth.** Runs 15 and 19 and Gromov agree. ShortOPD needs about 250M rollout tokens.
- **MMLU-only checks of carved models.** ShortOPD shows multiple choice survives while generation collapses.
- **Expert merging.** REAP shows pruning beats merging for generation.
- **Steering transfer (UNLOCK).** It cannot add a skill the host lacks.
- **Cross-architecture weight fusion (Transport and Merge, SemAlign, Theseus, BiCo).** It gives a few points on small tasks, with single seeds and vision or T5 only.
- **Joint router and expert training at BTX or Symphony scale.** It needs 5B to 80B tokens.
- **Heals on gold answers in a foreign style.** Run 16b lost 11 GSM8K points.

## Order of launch

1. Smoke Experiment 4 (CPU Procrustes first, then one cell).
2. In parallel, smoke Experiment 1. Experiment 2's cut choice reads Experiment 1's layer set.
3. Smoke Experiments 2 and 3 once the first cells print a row.
4. Hold Experiment 5 until round 20c results arrive, so its controls are comparable.

Keep the count under about 10 GPU containers per workspace, and tell the mining session before a fan-out.

## Mapping onto our evidence

# Mapping the literature onto our WSP runs

I read results/README.md (runs 14 to 19, the stitch study, runs 4 and 9), LESSONS.md and the team-memory recalls. I did not open the runs 6 to 8 or 20 to 24 write-ups, and I did not re-open the cited papers. Paper numbers come from the brief. The 2026 arXiv items (2603 to 2608 ids) I cannot check, so treat them as reported, not verified. I recorded nothing, since the task was read-only.

## Bottom line

- Every published success that moves components between models at block or layer level has a shared initialization. That covers layer swapping, DFS and PS+DFS, SOLAR, Boomerang, BTX, FlexOlmo, DUME, PHATGOOSE and diff transfer.
- I found no published case of different-init or different-width donors beating a capacity-matched control at a minutes-to-an-hour heal.
- The cross-architecture results (Transport and Merge, SemAlign, Theseus/BiCo) give a few points. They stay far below the donor, and the two LLM ones are single-seed or small.
- The retrofitted-recurrence paper (26 to 52B heal tokens) and the Coconut curriculum say a pretrained network needs orders of magnitude more than our heal to learn to read a rearranged or re-entered residual stream. Our heals are 500 to 1500 steps.
- Our own pattern matches the literature closely. A seam trained on the donor's loss is near-lossless when the path continues through the donor's own downstream layers and head. It fails when the path returns into a host that was never trained on that distribution.

## Method by method

Run labels below are from our README.

**Layer swapping (Bandarkar)**
- Tested: not directly. The nearest test is run 16c (Math layers into Instruct, chance) plus the round 21 weave.
- Conditions: they match round 21, where donors are true fine-tunes. They do not match 16c, where cosine is 0.2 to 0.5.
- Prediction: 16c fails for the reason the paper gives. Round 21's signal should find some layers where swapping works. Expect gains near the paper's, which were small and negative for Japanese. Choose cuts by where the fine-tune delta lives, not by index.

**DFS/PS+DFS (Sakana)**
- Tested: not directly. The nearest tests are the two 14c layer ranges (8-point seed swing) and our learned seam scale.
- Conditions: all sources share the Mistral-7B base. Unscaled frankenmerge scored 0 on MGSM-JA, so the learned scale did real work. DFS alone lost general Japanese skill, which means the proxy fit the task.
- Prediction: a path search at our eval sizes (dev 1000) will fit noise unless the signal is large. Our trained stitches already play the role of W. Expect the same mixed result: the search signal gains on its own proxy while general skills slip.

**Stitching, directional (StitchLLM, Bansal, Chen, Ventriloquist)**
- Tested: yes. The stitch study, run 4, runs 14 and 22.
- Conditions: the match is good. Large-bottom into small-top holds and the reverse does not (StitchLLM). Strong into weak works for early layers and later layers differ more (Bansal). Affine maps carry common features and lose rare ones (Chen). Compatibility is localized to the final layers (Ventriloquist).
- Prediction: this predicts runs 14 and 14b (donor block written into the base's upper layers) fail, and they did.

**Depth and slot methods (SOLAR, Boomerang, retrofitted recurrence, Gromov, Minitron, ShortOPD)**
- Tested: partly. Run 15 (carving) and the 14b random-block control.
- Conditions: all need a same-weights slot, and most need billions of tokens. Gromov and ShortOPD give a recoverable limit near 20 to 25% depth.
- Prediction: our 54% carves (14B to 22/48 layers) are past the limit. The 14b control matching the real block is the SOLAR effect (added depth), not skill.
- Not tried: a layer-wise cosine loss on the slot against the donor block's input and output (Boomerang). That is the cheapest untested mechanism here.

**Heterogeneous and aligned experts (Symphony-MoE, MergeME, FlexOlmo, BTX)**
- Tested: run 16b is the no-alignment arm, with a router that stayed on the base.
- Conditions: all need 5B to 80B heal tokens, or a frozen anchor during expert training (FlexOlmo, 47.8 vs 38.8 without).
- Caveat: Qwen2.5-Math and -Coder are continued pretraining of the same init lineage. Permutation symmetry is probably intact there. Horoi found the best permutations were always identity for same-base fine-tunes. So the 16c collapse is more likely drift (cosine 0.2 to 0.5, possibly rope_theta 10000 vs 1e6) than a permutation mismatch. This is my inference, not a result.
- Prediction: permutation alignment alone will not rescue 16c. Rope_theta is a confound we have not isolated.

**Router recipes (DUME, PHATGOOSE, UpIT, Router KD)**
- Tested: no. The ridge router in 16c collapsed (MMLU 49.3, GSM8K 0.3) because the experts themselves were broken.
- Prediction: a better router cannot fix incompatible experts. DUME reaches only 91% on math even with true fine-tunes.
- Use: Router KD is worth trying on our 64-expert and 32-expert Qwen3 cells. It will not rescue 16 experts.

**MoE slimming (REAP, MoNE, MoE-to-dense, Puzzle)**
- Tested: run 19 reproduces REAP. Saliency holds and the other rankings collapse.
- Conditions: it matches. MMLU fell 33 points, which is the paper's calibration trade-off.
- Prediction on the transplant arm: MoE-to-dense saw about 43 at 0.3B tokens against a teacher of 79.65. Their random-FFN control was already at 32.7 to 45.4, close to the selected experts early on. Our +3 MMLU is much smaller than their 10-point selection gap. I expect the scrambled and random controls to recover most of it. The paper also keeps the teacher's attention, and our dense base does not.

**Weight-space transfer across lineages (Theseus/BiCo, Transport and Merge, SemAlign, diff transfer, LMC-DM)**
- Tested: no.
- Conditions: gains are small. Theseus/BiCo is vision and T5 only. LMC-DM leaves barriers of 0.18 to 1.53 nats and merged endpoints trail badly (PIQA 56.4 vs 71.0).
- Prediction: if we try BiCo-style transport on Math into Instruct, the ceiling is a few points. Untried on LLMs, so it is a candidate experiment only.

**Activation transfer (Ramesh and Li, Pythia negative result, UNLOCK)**
- Tested: our runs 14 and 22 resemble the negative result.
- Conditions: a map with R2 0.88 and cosine 0.97 still gave 0.25% under replacement. The translated vectors had norm 0.85 against 68.7 for native states.
- Prediction: the feedback seam's one-step blow-up (KD 0.64 to 7.9) is the same scale problem. UNLOCK cannot induce a skill the host lacks, which is our Math-into-Instruct case.

**Feedback and recurrence (Coconut, LRT, Full-Bandwidth)**
- Tested: our feedback seam is in flight.
- Conditions: all train from scratch with feedback present. Coconut without its curriculum scored 14.4, below No-CoT at 16.5. LRT and Full-Bandwidth use zero-init gates, 0.1 initial scale and additive injection.
- Prediction: a short heal on a frozen checkpoint gives roughly chance gain. The recipe to follow is zero-init, bounded scale, additive and staged, with a halt on loss jumps.

## Predictions for the in-flight experiments

- **Skill tops.** This is the best-supported design.
  - The trained stitch kept 0.03 to 0.09 nats in the stitch study, and Ventriloquist and Bansal predict this direction holds.
  - The risk is StitchLLM's rule: small-bottom plus large-top underperforms the base models.
  - It is also mostly the donor with a swapped lower trunk, so run the control with the donor's own bottom at the same cut. A score above the donor on its own skill text is the real claim. The stitch study's negative WikiText penalties hint at it.
- **Feedback seam.** Expect no gain over a random control. The best case is small, if the gate starts at zero and ramps.
- **Reader study.**
  - Expect loss to recover and skills to stay low, as in Gromov.
  - Expect easier re-entry closer to the head (Bansal, Ventriloquist).
  - Expect LoRA on all layers to beat glue only, with the random control close behind. 14b already showed that pattern.
  - Retrofitted recurrence suggests a 1500-step heal cannot train the readers of a changed stream.
- **Expert transplant.** Expect controls to take most of the +3 (see MoE slimming above). Add a selected-vs-random-expert control at the same heal length, with at least three seeds. Our own 14c and 16b show 8-point seed swings.
- **Round 21 weaving.** Positive signal only for true fine-tunes, as in the layer-swapping, diff-transfer and DFS papers. A search that gains on its proxy and loses general skill is the expected failure.

## Mechanisms that most plausibly explain the failures, ranked

**1. The host's downstream layers were never trained to read donor features.**
- For:
  - Runs 14, 14b and 22 had lossless input stitches and still no skill.
  - The donor-side stitch is near-lossless (0.03 to 0.09 nats) and the base-side write-back is not.
  - The Pythia negative result and Ramesh and Li (replace beats sum) agree.
  - Retrofitted recurrence needed 26 to 52B tokens, and Coconut needed a curriculum.
- Against:
  - 14c gave about +4 over distill-only on two seeds, so some reading happens.
  - Runs 6 to 8 reported taps carrying knowledge into weak hosts. I did not check those runs, and their MMLU-format training is a confound.
  - Run 14b's LoRA host did gain +3 to +6, but the random block gained as much.
- What it predicts: a heal that is orders of magnitude longer, or a layer-wise loss at the slot, would help. Neither fits the budget.

**2. No shared coordinate system between lineages that diverged (cosine 0.2 to 0.5).**
- For:
  - 16c is chance even for whole Math layers on the base's embeddings and head. Only the whole specialist reproduces its scores.
  - Li and Shen leave a barrier after learned alignment and merged models trail their endpoints.
  - Diff transfer breaks across OLMo 2 connectivity groups.
  - Layer swapping works only because both experts are short fine-tunes of one model.
- Against:
  - A trained stitch fits the map for the donor-side direction, so a map exists.
  - The rope_theta difference in 16c is a separate confound I cannot rule out.
- What it predicts: the weaving and layer-swap tools stay inside true fine-tunes. Cross-lineage reads need a map trained on the receiver's own loss.

**3. Scale and norm mismatch at the seam.**
- For:
  - The feedback seam blew up after one Adam step because of outlier dimensions.
  - The Pythia negative result had norms 0.85 against 68.7.
  - Chen finds rare features transfer badly, and run 4's ridge stitches cost 1.4 to 2.2 nats.
- Against:
  - It explains blow-ups, not the quiet failures in run 14, where the stitch penalty was small.
- What it predicts: normalize and tie scale to the host (DFS's W, LRT's 0.1 initial scale). This is a cheap fix that removes one failure and does not make the skill transfer.

**4. The skill lives in the donor's own readout, not in a movable block.**
- For:
  - Ventriloquist localizes compatibility to the final one or two layers.
  - 16c shows only the whole specialist works.
  - Our skill tops design came from this pattern.
- Against:
  - Layer swapping moves a task skill through middle layers when the fine-tunes share a base.
  - UNLOCK shows capability moves as a direction when the host already has it.
- What it predicts: tops should work, middle-block write-back should not, and expert-level MLP swaps only work with matching attention.

**5. Heal objective and measurement hide small real effects.**
- For:
  - Round 22: the donor's contribution was learnable by LoRA from the same data.
  - 16b: the heal cost GSM8K 11 points.
  - Seed swings were 8 points (14c, 16b), and the random block matched the donor in 14b.
- Against:
  - It explains why we cannot see gains, not why they are absent.
- What it predicts: use at least three seeds with z-scored scrambled controls. Test on an eval the data cannot supply, such as donor-written answers or held-out facts.

## Cheapest next tests the literature supports

- Add a layer-wise cosine loss against the donor block on the host slot (Boomerang) to the reader study.
- Router KD on the 64- and 32-expert Qwen3 cells before any full heal.
- Run the skill-tops control with the donor's own bottom at the same cut.

## Methods found

### Activation communication: replace the receiver's last-token activation at layer 26 with the sender's, optional learned linear map

- Source: Communicating Activations Between Language Model Agents, Ramesh and Li, ICML 2025, https://arxiv.org/html/2501.14082v2 (opened)
- What it does: Pause model B at layer j, take model A's layer-k activation (k=j=26 after grid search), combine with function f, continue B's forward pass. Best f is replace on the last token only. An optional linear map W is fit once per pair on 3,072 C4 sentences (10 epochs, Adam, lr 1e-3).
- Numbers: Up to 27.0% over natural-language debate at under 1/4 the compute. Countries (3B->8B): replace 78.0/83.0 vs text 69.0/77.0. GSM8k: 64.0 vs debate 75.0. Training W in-distribution on GSM8k raised 64.0 to 78.0. MMLU avg 62.7 vs 60.7.
- Conditions: Works with no training only for LLaMA-3.2-3B -> LLaMA-3.1-8B style pairs (same family, similar spaces); also tried Qwen-2.5, Gemma-2 and mixed pairs. Sum and mean combiners are much worse because they leave the receiver's activation space. Editing all tokens instead of the last token hurts. Swapping roles lowers accuracy. Out-of-distribution fit data for W can hurt. Receiver is never trained, so this is a one-token, last-position channel, not a block read at every position.
- Relevance: Shows a frozen receiver can read a foreign activation only when it is injected as a replacement in the receiver's own space and scale (replace beats sum or mean). Matches our runs 14 and 14b: the written block output sits outside the base's later-layer input distribution. Also says a small in-distribution map fit helps a lot (64 to 78), which suggests the reader-side fit data matters more than fit length.

### Negative result: translation with high fit metrics but no causal use by the receiver

- Source: A Negative Result on Cross-Model Activation Transfer in a Pythia pair, arXiv 2606.03280, https://www.alphaxiv.org/abs/2606.03280.md (opened; alphaxiv summary, not the PDF)
- What it does: Linear map from Pythia-160M layer 8 to Pythia-410M layer 16 (relative depth 0.67), fit to L2-normalized receiver activations, receiver untouched. Tests additive, replacement, scale-corrected replacement, and blends on multi-hop QA (396 rows).
- Numbers: Fit: normalized R2 about 0.883, cosine about 0.974. Downstream: no injection 8.84%, text relay 8.33%, additive 9.26 +/- 0.73 (CI vs none [-0.42, +1.26], no reliable gain), uncorrected replacement 0.25%, scale-corrected replacement 0.76%, best blend (alpha 0.30) 4.88%. Translated vectors had L2 norm about 0.85 vs about 68.70 for native receiver states. Shuffled translations also destroy output.
- Conditions: One pair, one task, one injection depth, one linear map, weak receiver (baseline under 10%). Same tokenizer and family. The paper does not test a fix; it proposes training an objective on receiver-side causal use.
- Relevance: Direct evidence for our core failure: a map can reproduce the donor's state well (R2 0.88, cosine 0.97) and the receiver still cannot use it. The leftover 3% of direction error plus a norm mismatch is enough to wreck replacement. Says the heal objective must be the receiver's own loss on the written-in path, which is what our trained stitch (0.03 nats) did for the donor path but not for the base path.

### Cross-model stitching with affine maps (transfer of SAEs, probes, steering vectors)

- Source: Transferring Linear Features Across Language Models With Model Stitching, Chen, Merullo, Stolfo, Pavlick, NeurIPS 2025, https://arxiv.org/html/2506.06609v3 (opened; read 100k of 106k chars)
- What it does: Learns an up and a down affine map between residual streams of a small and a large model (MSE plus inversion penalty, about 180k OpenWebText samples). Layer pair chosen by SVCCA. Then transfers SAEs, probes and steering vectors across.
- Numbers: Zero-shot transferred SAE fraction of variance unexplained, original/transferred: Pythia 0.17/0.52, GPT-2 0.11/0.39, Gemma-2 2B->9B 0.21/0.42. SAE initialized from transfer needs about 30 to 50% fewer FLOPs (headline about 50%). Language steering en-fr: 0.01 none, 0.83 ground truth, 0.77 transferred; en-hu transferred poorly and transfer tracked language frequency. IFEval instruction steering transferred weakly. Small-to-large next-token loss stays near the larger model's loss, large-to-small round trips are nearly lossless.
- Conditions: Same-family pairs with shared tokenizer only (Pythia-70m->160m, GPT-2 small->medium, Gemma-2-2B->9B). Stitch trained on general web text only; chat-template tokens and rare features transfer badly. Transferred weights have rank at most the smaller model's width. Cross-family untested.
- Relevance: Gives the upper bound on what an affine map carries: common, high-frequency, linear features survive; rare or format-specific ones (code, chat, instruction following) do not. Our donor skills (code, math) are in that weak class. Same-family and same-tokenizer is the working condition, which matches our finding that Qwen2.5-Math/Coder (cosine 0.2 to 0.5) fail while true fine-tunes work. Note the bottleneck direction: a small base reading a large donor loses to rank.

### Model stitching as a diagnostic, and cross-model linear transferability

- Source: Revisiting Model Stitching to Compare Neural Representations, Bansal, Nakkiran, Barak, 2021, https://arxiv.org/abs/2106.07682 (abstract opened); Linear Representation Transferability Hypothesis, Bello et al., https://arxiv.org/abs/2506.00653 (abstract opened)
- What it does: Bansal: freeze nets A and B, join A's bottom to B's top with a small trainable layer; near-lossless when same architecture, different training (supervised vs self-supervised). Reports that representations from more data, more width or longer training can be plugged into weaker models to improve them. Bello: proposes an affine map between hidden states of models of different sizes preserves steering behavior.
- Numbers: Abstracts only: no loss-gap numbers were available to me. Bansal states no performance drop for same-architecture, differently trained pairs. Bello states strong evidence that affine maps preserve steering, without metrics in the abstract.
- Conditions: Bansal is on vision nets and same-architecture pairs; the stitch is a single trainable layer, trained with the downstream task loss. Bello is small-to-large steering transfer. I did not open full texts, so details are unverified.
- Relevance: Supports the direction of our stitch study: stitching works when the receiving side is the model's own remaining layers and head (our 0.03 nats). The 'more is better, plug into weaker' claim is the opposite of the direction we need (weak base reading a strong donor block), which our runs 14 and 14b show does not carry the skill.

### Retrofitting depth recurrence into pretrained models (reader learning curve for a re-entered block)

- Source: Teaching Pretrained Language Models to Think Deeper with Retrofitted Recurrence, McLeish et al., NeurIPS 2025, https://arxiv.org/html/2511.07384v1 (opened; read 100k of 143k chars)
- What it does: Turns TinyLlama-1.1B, OLMo-2-1B and Llama-3.2-1B into prelude/looped-core/coda models by dropping layers and feeding the core block's output back into itself; heals with a recurrence-depth curriculum.
- Numbers: 26B-token single-phase heal (FineWeb-Edu, Nemotron SFT general, SFT math) or 52B two-phase; math runs on about 50B tokens. TinyLlama GSM8K at test recurrence 32: 52.0 and 51.2 vs 46.2 and 45.2 for matched non-recurrent baselines; MATH 14.5 vs 14.4. Curriculum example about 1e20 FLOPs, 48 h on 4 MI300A. Pretrained init beats random init over about 120B tokens, extrapolated crossover about 950B tokens.
- Conditions: Takes early layers as prelude, late layers as core and coda (beat ShortGPT-style pruning). Needs tens of billions of tokens, not minutes. The paper does not state that a short no-curriculum run fails; it shows scheduling depth improved loss per FLOP.
- Relevance: Closest data point on how long a pretrained network needs to read a rearranged or re-entered block: 26 to 52B tokens, 3 to 4 orders of magnitude over our heals (1500 steps). Supports the view that a minutes-to-an-hour heal cannot retrain the readers of a changed residual stream; it also says the rearrangement must keep the pretrained layers (prelude and coda) in their native roles, as in our skill-tops design.

### Continuous-thought feedback (Coconut): last hidden state fed back as next input embedding

- Source: Training Large Language Models to Reason in a Continuous Latent Space, Hao et al., 2024, https://arxiv.org/html/2412.06769v3 (opened)
- What it does: In latent mode the final hidden state is used directly as the next input embedding, no LM-head decode. Trained from pretrained GPT-2 with a multi-stage curriculum that replaces language CoT steps by latent steps.
- Numbers: GSM8k (c=2): full 34.1, no curriculum 14.4, No-CoT 16.5. ProntoQA: 99.8 vs 52.4 vs 93.8. ProsQA: 97.0 vs 76.1 vs 76.7. Schedule: 6 epochs initial stage then 3 epochs per curriculum stage on GSM8k; ProntoQA/ProsQA 5 epochs per stage for 6 stages, up to 50 epochs. At c=3 adding three thoughts at once spiked training loss. Gains on Llama 3.2-3B and Llama 3-8B were smaller than on GPT-2.
- Conditions: Needs n+1 sequential forward passes for n thoughts. Without the curriculum the model is worse than not using latent thoughts at all. Larger models with heavy language pretraining gained less.
- Relevance: Direct match for our feedback seam: a pretrained model asked to read its own hidden state as an input fails under a one-shot switch (14.4 vs No-CoT 16.5) and works only when the foreign signal is introduced in stages. Also matches our finding that a one-step jump in written norm collapsed the heal; the curriculum is a fix to try (ramp the fraction of positions fed back, or the gate, over the heal).

### Latent feedback transformers: Latent Recurrent Transformer (Huang et al.) and Full-Bandwidth Transformer

- Source: Latent Recurrent Transformer, Huang et al., arXiv 2605.26797, https://arxiv.org/html/2605.26797 (opened; first 100k of 118k chars) and https://arxiv.org/abs/2605.26797; Full-Bandwidth Transformer, arXiv 2608.08888, https://arxiv.org/abs/2608.08888 (abstract opened; the Full-Bandwidth first author is not Huang, per the search result)
- What it does: LRT reuses a mid-depth hidden state of the previous token as memory for the next token, read through zero-initialized per-head sigmoid gates on KV projections (added to local K and V) and a residual injection with memory scale initialized to 0.1. Full-Bandwidth feeds the previous top-layer state back, fused with the token embedding by a gated linear unit.
- Numbers: LRT, 20-layer nanochat-style, BPB: baseline 0.767, residual injection 0.756, KV projection 0.755, both 0.754; source layer 12 of 20 best (0.754), 8: 0.757, 16: 0.756, 20: 0.755. CORE 0.271 baseline vs 0.274 shared and 0.277 layerwise. 1.3B and 2.1B backbones, 9% decode overhead. Full-Bandwidth: 1B models, up to 400B tokens, matches standard transformers trained on about 1.5x more tokens; feedback is introduced late in pretraining on a schedule with a small share of deeper passes.
- Conditions: Both are trained from scratch; neither paper tests adding feedback to an existing checkpoint (LRT's text contains no such ablation; Full-Bandwidth's abstract does not either). Gains are small (about 2% BPB). Zero-init gates and small initial scale make the fresh model equal the baseline at step 0.
- Relevance: Design recipe for our feedback seam: zero-init gate, small initial scale (0.1), additive not replacing, source from mid-depth, parallel two-pass training. Matches our own lesson that an unnormalized written vector blew up after one Adam step. But the evidence only covers readers trained from scratch with the feedback present, so it does not show a short heal on a frozen checkpoint can learn to read it.

### Layer swapping (language expert top and bottom layers into a math expert)

- Source: Bandarkar et al., Layer Swapping for Zero-Shot Cross-Lingual Transfer in Large Language Models, ICLR 2025, https://arxiv.org/abs/2410.01335 (full text read via the PDF)
- What it does: Fine-tune two experts from the same pretrained Llama 3.1 8B: one on English math, one on generic instructions in a target language. Build one network from the language expert's first layers and last layers and the math expert's middle layers. Weighted-average transition layers sit at the seams. The layer choice came from analysis: reverting the first 5 and last 2 layers of the math expert to the pretrained values did not lose math skill, so those layers hold mostly language-specific change.
- Numbers: MGSM 8-shot, average over 9 expert pairs, default config vs math expert vs model soup: Swahili 32.4 vs 29.5 vs 29.3; Telugu 22.7 vs 20.1 vs 20.9; Bengali 37.1 vs 38.3 vs 36.8; Japanese 38.5 vs 42.7 vs 38.7 (a loss for Japanese, where the math expert was already decent). Best config max: Swahili 37.2, Telugu 27.6, Bengali 45.2, Japanese 43.2. Paper abstract says about 10% above the individual experts and soup. Base Llama 3.1 8B: Swahili 24.8, Telugu 12.0.
- Conditions: Works only because both experts are short fine-tunes of one pretrained model, so each layer's output space has barely moved. The authors say the transition zones gave no significant benefit here, and conjecture they would be needed if experts trained longer or with higher learning rates. Gains are on one task family (MGSM) with 3 experts per side. No cross-family or cross-size result. Same-base only, no learned seam.
- Relevance: This is the positive twin of your Qwen2.5-Math/Coder failure (cosine 0.2 to 0.5 from Instruct): layer swaps work at cosine near 1 and give chance at 0.2 to 0.5. It also gives a cheap, benchmark-free recipe to test on your true fine-tune pairs (round 21): find which layers a skill's fine-tune actually changed, and swap only those. The result that the early and late layers carry language and the middle carries task suggests choosing the cut by where the fine-tune delta lives, not by layer index.

### Evolutionary data flow space merge (layer-path search with learned inter-layer scaling) and PS+DFS

- Source: Akiba et al. (Sakana AI), Evolutionary Optimization of Model Merging Recipes, arXiv 2403.13187 (revised 2025, Nature Machine Intelligence), https://arxiv.org/abs/2403.13187 (full text read)
- What it does: Data flow space (DFS) merging keeps every layer's weights intact and searches the path a token takes: after layer i of model A it may go to layer j of model B. Search space is an indicator array over M layers repeated r times (M=64, r=3, T=192 slots) plus a matrix W of scalings applied to the hidden state when going from layer i to layer j. CMA-ES runs 100 generations at population 128 on 1069 Japanese-translated GSM8K items that are disjoint from the test set. Model A's tokenizer, embeddings, first layer and last layer define the path ends. The parameter space (PS) merge uses TIES with DARE and then DFS runs on top.
- Numbers: MGSM-JA accuracy: Shisa Gamma 7B 9.6, WizardMath 7B 18.4, Abel 7B 30.0 (best source). PS merge 52.0. DFS alone 36.4 (10B, 3+1 models). PS+DFS 55.2 (10B). Plain frankenmerge of the same models: MGSM-JA 0. JP-LMEH average: DFS 53.2, PS 70.5, PS+DFS 66.2. DFS alone lowered JP-LMEH below the source models (Shisa 66.1) and dropped MARC. 13B variants scored lower (31.2, 23.2, 34.0 for PS, DFS, PS+DFS).
- Conditions: All source models are fine-tunes of Mistral-7B-v0.1, so layers are mutually readable. DFS was limited to two models, serial and non-adaptive paths. The search signal is a task metric, so the gain on MGSM-JA is partly fit to the task, and DFS alone traded away general Japanese skill. The paper notes that swapping two neighboring layers hurts a model, which is why the scaling matrix W was added. Repeating or permuting early layers hurt.
- Relevance: Closest published match to your reader problem: an unscaled frankenmerge scores 0 and the same layers with a learned seam scale and a searched path score 36 to 55. It supports a cheap search over which donor layers feed which host layers, with a learned per-seam scale. But the lineage condition is the same one you already found (same Mistral base). Treat the learned W as a normalization fix for outlier dimensions, in line with your feedback-seam failure where an unnormalized write blew up KD from 0.64 to 7.9.

### StitchLLM: trainable MLP stitch between blocks of different-size models in one family

- Source: Hu et al., StitchLLM: Serving LLMs, One Block at a Time, ACL 2025, https://aclanthology.org/2025.acl-long.1305 (abstract page, plus PDF full text read)
- What it does: Take the first i layers of a large model B and the top layers of a smaller model T of the same family, with a single MLP S between them (4096 to 2048 for Llama 8B to 1B). The bottom block feeds T starting at layer j = i * (Lb/Lt). Only S trains, with the original pretraining cross-entropy on C4 for under 2000 gradient steps, inside frozen models. A router then picks the block pair per request.
- Numbers: Built 112 stitching layers for Llama 2 (13B-1.1B, 13B-7B, 7B-1B) and 92 for Llama 3 (8B-3B, 8B-1B, 3B-1B). Reported system results: average response accuracy up 8% against state-of-the-art serving systems, TTFT down 18%, P95 latency down up to 33.5%, GPU utilization up to 20.1%, accuracy up 12.2% at peak load. Filtering by heuristics cut Llama 2 13B-7B candidates from 72 to 40 (larger-bottom only) and later further. The abstract and body give no per-benchmark table in text I could read; the stitched models fill the accuracy-latency gap between base sizes on MMLU, BoolQ, Winogrande, HellaSwag.
- Conditions: Large-bottom plus small-top works and keeps accuracy. Small-bottom plus large-top underperforms the base models. Cross-family stitching (Llama 2 with Llama 3) degrades because of structural and tokenizer incompatibility, so they restrict to one family. Within a family there is a performance boundary layer (cold region of low stable accuracy, then a hot region); for Llama 2 it is near the middle, for Llama 3 near the end, and Qwen 2.5 32B has two boundaries at blocks 7 and 47. Limitations section says tokenizer and vocabulary match is needed. The approach is built for serving, and the authors rely on one MLP.
- Relevance: Matches your stitch study (trained stitch into a donor block near-lossless when the path continues through the donor's own remaining layers and head) and your run 14 finding that a small base cannot read a block. StitchLLM stitches the opposite direction (large early layers into small late layers) and that is the only direction that holds. Read-back works when the reader side is a model that was trained to consume that depth of representation. The per-family hot/cold boundary is a free diagnostic for where a seam can sit before heal.

### Model stitching as a measuring tool: stitching connectivity and 'more is better'

- Source: Bansal, Nakkiran, Barak, Revisiting Model Stitching to Compare Neural Representations, NeurIPS 2021, https://arxiv.org/abs/2106.07682 (full text read). Lenc and Vedaldi 2015 is the origin of the idea; I did not open it.
- What it does: Freeze model A's bottom layers and model B's top layers and train only a simple layer (1x1 conv or linear) between them. Accuracy of the stitched model measures whether the two representations are interchangeable.
- Numbers: Abstract reports no drop in performance when stitching good networks of one architecture trained very differently (supervised vs self-supervised). Early-layer stitches between independent seeds stay close to the base error. Better representations (more data, wider, longer training) can be plugged into weaker models to improve them. Page text gave no exact error numbers beyond that, so I do not cite any.
- Conditions: Vision networks (ResNet, ViT, Myrtle-CNN) of identical architecture. Stitches from a strong bottom into a weak top gain; bottom networks that are random do not. Works best at early layers; training detail affects later layers more. Not a language-model result and not cross-architecture.
- Relevance: Gives the rule your runs keep rediscovering: the seam is cheap when the two sides share architecture and the side that gets replaced is the weaker one. It also supports the direction 'strong early layers into a weaker model's later layers', and says later layers differ more by training details. That predicts your failure writing donor block output into a base's upper layers.

### Affine stitches between residual streams of same-family LMs (Transferring Linear Features)

- Source: Chen, Merullo, Stolfo, Pavlick, Transferring Linear Features Across Language Models With Model Stitching, arXiv 2506.06609v3 (Nov 2025), https://arxiv.org/abs/2506.06609 (full text read)
- What it does: Fit an affine map between a layer of a small model and a layer of a larger model of the same family and tokenizer, trained on general web text. Used to move SAE weights, probes and steering vectors between sizes.
- Numbers: Next-token CE on OpenWebText, A to B stitch vs B alone: Pythia 70m layer 3 to Pythia 160m layer 4, +27% (3.99 vs 3.14 in the table); GPT-2 small layer 6 to GPT-2 medium layer 10, +11%; Gemma-2-2b layer 20 to Gemma-2-9b layer 33, +39% going up, and +8.3% coming down. Round trips A to B to A cost under 1% (but B to A to B costs +4.7%, +1.4%, +13%). Transferred SAEs have fraction unexplained variance 0.21 to 0.42 for Gemma 2B to 9B, better than random but worse than a trained SAE, and used as init they make SAE training on the larger model about 50% cheaper.
- Conditions: Same family, same tokenizer, general text only. Stitches stay affine, so they map only a rank-limited subspace (rank at most the smaller width). The authors list cross-family and chat-format data as untested. Steering vectors for language transfer work for some languages and poorly for others; format-following steering is weaker.
- Relevance: A quantitative price list for a seam: a plain affine stitch from small to large costs 11 to 39% in loss, while the large-to-small direction costs less (8.3% for Gemma). That matches the StitchLLM direction rule and your finding that an MSE or ridge stitch leaves 0.4 to 0.65 nats while an LM-loss-trained stitch leaves 0.03 to 0.09. A residual stream map keeps feature directions, so a read-back that needs one feature (a probe-like use) is much easier than a full block read-back.

### Depth up-scaling (SOLAR): duplicate a model, drop the middle, concatenate, continue pretraining

- Source: Kim et al., SOLAR 10.7B: Scaling Large Language Models with Simple yet Effective Depth Up-Scaling, NAACL 2024 Industry, https://arxiv.org/abs/2312.15166 (full text read)
- What it does: Take Mistral 7B (32 layers) in Llama-2 layout. Copy it, remove the last 8 layers of one copy and the first 8 layers of the other, and stack them into 48 layers (10.7B). Then continue pretraining. The seam joins layer 24 to layer 9, not layer 32 to layer 1, to keep the layer distance at the seam small.
- Numbers: Open LLM Leaderboard H6 average: SOLAR 10.7B base 66.04 against Mistral 7B 60.97 (GSM8K 55.50 vs 37.83; MMLU 65.48 vs 64.16). SOLAR 10.7B-Instruct H6 74.20 above Mixtral 8x7B-Instruct 72.62 and Qwen 72B 73.60. The paper states the scaled model first drops below the base before continued pretraining, and that performance recovers quickly during it. Page does not give the token count.
- Conditions: Both halves are the same model, so every layer reads features of its own network, with one seam. The authors argue the m=8 trimmed layers on each side reduce the layer-distance at the seam, and a plain repeat from n to 2n layers has the largest distance at the seam and is harder to heal. They also state the discrepancy must be small enough for continued pretraining to resolve. Full continued pretraining is far beyond a minutes to an hour heal.
- Relevance: Shows what a single mismatched seam costs and how much healing it needs even in the best case (same weights). The seam-distance argument is a design rule for where to cut a donor's range: choose entry and exit layers so the layer-index gap is small. It also explains why your 14b control (random same-donor block) matched the real block: added depth recovers either way, and gains from donor identity need a seam where depth is not the cause.

### Boomerang distillation: zero-shot reinsertion of teacher layer blocks into a pruned-and-distilled student

- Source: Kangaslahti et al., Boomerang Distillation Enables Zero-Shot Model Size Interpolation, ICLR 2026, arXiv 2510.05064v2, https://arxiv.org/abs/2510.05064 (full text read); follow-up on layer patching (KLPatch, arXiv 2607.08170) seen only as a search snippet, not opened
- What it does: Initialize a student by dropping every other layer (except the last) of a teacher, and distill it with a loss that includes a cosine loss between student layer outputs and the matching teacher layers. Then patch contiguous blocks of teacher layers back into the student, starting from the last layer, to get models of any size in between with no training.
- Numbers: Qwen3-4B-Base teacher, 2.7B-parameter student, 2.1B tokens on the deduplicated Pile. Interpolated models scale smoothly between student and teacher on ten classification tasks and three generation tasks, and beat ShortGPT, LaCo and naive layer pruning, whose generation accuracy collapses to near zero after a few layers are removed and whose classification plateaus near chance at about 3B. Also shown for Qwen3-8B, Pythia-6.9B, Llama-3.2-3B, and weakly for off-the-shelf DistilBERT/BERT (clean) and DistilGPT2/GPT2 (less clean). Page text did not give a headline accuracy number, so I report none.
- Conditions: Needs the student to come from teacher layers (same hidden size, same tokenizer) plus a distillation loss that aligns hidden states. Alignment through pruning and distillation is stated as essential. Students pruned with width reduction (Minitron style) cannot be patched because hidden sizes differ. Without the cosine loss or from a random-init student, interpolation fails.
- Relevance: Direct evidence for what made read-back work: a layer-output alignment loss (cosine to the teacher's own layer outputs) makes a slot accept the original layer without any seam training. For Sapien this suggests training the host's slot to match the donor block's input and output distribution at the cut (layer-wise cosine, not only final logits) before inserting, instead of training glue after insertion. It is same-teacher only, so it does not show cross-family transfer.

### Fine-tuning diff transfer (task vector recycled onto a new base version)

- Source: Efficient Model Development through Fine-tuning Transfer, 2025, https://arxiv.org/html/2503.20110 (opened)
- What it does: Compute diff = fine-tuned minus base on one model version, add it to a newer or older base of the same family with no training.
- Numbers: Llama 3.0 8B + diff from 3.1: IFEval 34.5 to 76.6; Llama 3.1 8B + diff from 3.0: IFEval 36.4 to 83.3 (Instruct 80.5), LiveCodeBench 0.4 to 16.1. Diffs from SFT, DPO, GRPO also backport (GRPO lifts GSM8K 55.6 to 85.8 on Llama 3.0). OLMo 2 checkpoints from different linear-connectivity groups: GSM8K 17.3 vs 65.5 for the target alone.
- Conditions: Needs same architecture and models close in parameter space (shared lineage). Transfer between OLMo 2 groups often degrades. Target base must already be strong; early checkpoints stay 20 to 26 GSM8K points short. Cross-architecture left to future work.
- Relevance: Same story as our true-fine-tune weaving (round 21) and the Qwen2.5-Math/Coder chance results: works only at cosine near 1. Gives a hard boundary and a way to measure it (linear-connectivity group) before spending a run.

### Symmetry-aligned task arithmetic (permutation, rotation, scaling) before adding a skill vector

- Source: Leveraging Parameter Space Symmetries for Reasoning Skill Transfer in LLMs, Horoi et al., Nov 2025, https://arxiv.org/html/2511.10850 (opened)
- What it does: Align the source's parameter space to the target using Transformer symmetries (adapted to GQA and SwiGLU), then add the skill vector (Nemotron-Nano minus Llama-3.1-Instruct) to Tulu3-8B.
- Numbers: Mean of 6 reasoning benchmarks: Tulu3 29.3, plain task arithmetic 63.3, activation-aligned 63.8, weight-aligned 64.4, Nemotron-Nano 69.7. AIME24 61.2 aligned vs 55.9 plain. Plain arithmetic drops non-reasoning average 46.4 to 39.0 (IFEval 83.5 to 59.7); alignment does not fix it.
- Conditions: One model pair, all descended from Llama-3.1-Base, so only diverged fine-tunes of one pretraining run. Gains over plain arithmetic are small, mostly within error. The paper's ablation found optimal neuron permutations were always identity, and rotation-only was as good as the full method.
- Relevance: Shows alignment adds little even for diverged fine-tunes of one base, and no evidence for different pretraining runs. Rotation of the residual stream is the part that matters, which matches our stitch finding. Not a route for Qwen2.5-Math into Instruct (cosine 0.2 to 0.5) as published.

### LMC-DM: learned symmetry matching on both checkpoints (linear mode connectivity at LLM scale)

- Source: Scaling Linear Mode Connectivity and Merging to Billion Parameter Pretrained Transformers, Li and Shen, 2026, https://arxiv.org/html/2606.23607v1 (opened)
- What it does: Learn function-preserving transforms (permutations plus orthogonal and invertible maps) on both independently pretrained checkpoints so the linear path between them has low loss, then merge.
- Numbers: WikiText loss barrier raw to LMC-DM: Pythia-160M 6.28 to 0.18, Pythia-1B 4.60 to 0.23, Pythia-6.9B 3.74 to 0.18, OLMo-7B vs OLMo-7B-Twin 8.10 to 1.53. Weight matching alone barely helps (Pythia-1B 4.75). Merged Pythia-1B zero-shot: PIQA 56.4 vs 71.0 endpoint, SciQ 64.4 vs 83.9. Cost: 1 A6000 for hours to a day, 4 A6000 about a day for 7B.
- Conditions: Truly independent pretraining runs, but identical architecture, dimensions and tokenizer. Alignment fitted on WikiText-103 (same data as the barrier measure, so numbers may be optimistic). Low barrier does not give working downstream behavior; merged models trail endpoints substantially.
- Relevance: Best evidence that different-init same-shape LLMs can be put on one linear path, and also that this does not transfer skills. Matches our failure that a donor block cannot be read by the base. Useful only as a pre-step with a heal, and only for same-shape, same-tokenizer pairs.

### Generalized linear mode connectivity for Transformers (learned residual orthogonal alignment)

- Source: Generalized Linear Mode Connectivity for Transformers, 2025, https://arxiv.org/html/2506.22712v1 (opened)
- What it does: Defines permutation, semi-permutation, orthogonal and invertible symmetries of a Transformer and learns a matching over them; weight matching is the data-free variant.
- Numbers: Loss barriers on small from-scratch models: ViT CIFAR-10 vanilla 1.69, weight matching 0.36, learned full-symmetry 0.00; GPT-2 style (6 layers, d=512, BookCorpus) vanilla 4.35, weight matching 1.57, permutations only 1.58, full symmetry 0.41.
- Conditions: Small models trained from scratch with different seeds, same architecture. Restricting the residual stream to permutations leaves a nonzero barrier; orthogonal alignment of the residual stream is what closes it. Initializing from weight matching matters.
- Relevance: Explains why permutation-only re-basin fails on Transformers and supplies the residual-stream rotation that our seams learn implicitly. Gives a closed-form data-free start for a stitch (orthogonal Procrustes on the residual) before the heal.

### Linear alignment of final hidden states ('Ventriloquist': one model's states decoded by another's frozen head)

- Source: Ventriloquist LLMs: Linear Alignment of Late-Stage Representations, Gorbett and Jana, 2026, https://arxiv.org/abs/2603.18908 (abstract opened; PDF did not decode)
- What it does: Fit an affine map from the final hidden states of one independently trained LLM to another's, then decode through the target's frozen head with no fine-tuning.
- Numbers: No numeric thresholds in the abstract. Classification and OOD performance largely preserved; generation works sometimes, 'far from universal'. Strong-to-weak mapping beats weak-to-strong. Compatibility is localized to the final one or two layers.
- Conditions: Generation quality tracks tokenizer overlap and drops sharply below some source scale. Authors read it as a late-stage output-space effect, not shared internal computation.
- Relevance: Bounds the reader problem from the other side: independently trained models agree linearly only at the output, so an affine seam can feed a donor's last layers but not its middle. Supports skill-tops (donor upper layers plus head) and warns against mid-network reads across lineages.

### UNLOCK: capability direction extracted from a source pair, aligned to a target by a low-rank linear map

- Source: The Master Key Hypothesis: Unlocking Cross-Model Capability Transfer via Linear Subspace Alignment, Balasubramanian et al., Apr 2026, https://arxiv.org/html/2604.06377 (opened)
- What it does: Contrast activations of a source model with and without a capability (same architecture and tokenizer), fit a k by k least-squares map between source and target final-token states on unlabeled queries, add the mapped direction to the residual stream at every target layer. Training-free, steering not weights.
- Numbers: Qwen1.5-7B MATH 8.0 to 20.1 from the 14B (7B instruct with CoT 18.2); Qwen3-14B-Base AGIEval-Math 61.1 to 71.3 from the 4B (14B post-trained 67.8); gemma-2-9B GSM8K 3.0 to 60.1 from 2B; OLMo-2-7B GSM8K 10.0 to 63.4 from 1B.
- Conditions: Same family only in the main results (cross-family only preliminary, appendix). Elicits a latent capability the target already has from pretraining; cannot induce an absent one. Small-to-large gains 44.4% vs large-to-small 1.6% on gemma-2. Hyperparameters tuned on a dev set.
- Relevance: A different-init-adjacent result that works because it moves a behavior, not knowledge. Does not move a skill the host lacks, which is our case for Math-to-Instruct at chance. It could be a cheap inference-free (no second model) probe of whether a host already holds the skill.

### Theseus and BiCo: training-free transport of task vectors across widths and pretraining runs (Procrustes in activation and gradient space)

- Source: Transporting Task Vectors across Different Architectures without Training (Theseus), Rinaldi et al., ICML 2026, https://arxiv.org/html/2602.12952; Bilinear Coordinate Alignment for Training-Free Task-Vector Transfer (BiCo), Son et al., May 2026, https://arxiv.org/html/2605.28444 (both opened)
- What it does: Map a source task vector into a different-width target via orthogonal Procrustes fitted on a small calibration set; BiCo does it on both input activations and output gradients with one forward-backward pass.
- Numbers: CLIP ViT-B/16 (LAION-2B) to ViT-B/16+ (LAION-400M), 8-vision mean: zero-shot 57.7 to 58.8, Theseus 61.5 to 64.6 (69.5 with 50 batches), BiCo 76.5, target fine-tuned 92.9. Different pretraining sets, same arch, 5-shot: BiCo 80.0, GradFix 70.8, Theseus 68.9 vs 62.2 zero-shot. T5-3B to T5-Large encoder, NLI linear probe: 71.1 to 76.1 (BiCo); fine-tune 86.8.
- Conditions: Vision and T5 encoders with linear probes only; no decoder LLM results in either paper. Source dimension must be at most target dimension for the orthogonal map. Depth mismatch uses naive nearest-layer matching, and interpolating layers is destructive (8.5% zero-shot). Always far below target fine-tuning.
- Relevance: Only published training-free weight-space transfer that crosses different pretraining runs, but recovers about a third to a half of the fine-tuning gain on small vision tasks. Untested on LLMs, so it is a candidate experiment, not a result. Same-width needed, which fits our same-width dense base setup.

### Transport and Merge (optimal transport neuron matching across architectures) and SemAlign (activation-geometry layer transfer)

- Source: Transport and Merge, 2026, https://arxiv.org/html/2602.05495; Beyond Neural Incompatibility / SemAlign, Gu et al., https://arxiv.org/html/2510.24208 (both opened)
- What it does: Transport and Merge: align activations of a large source and 1B target with optimal transport (k=128 neurons, Sinkhorn) and fuse weights with coefficient alpha 0.05 to 0.15. SemAlign: pair each target layer with one source layer, train a LoRA r16 on the frontier layer to match token-relation geometry plus output KL.
- Numbers: Transport and Merge, 1B Llama targets from 7B or 8B sources (Llama-3, Qwen2.5-7B for finance): MalayMMLU +3.3 to +6.5 per category with adaptation, Indonesian avg 38.43 to 39.30, Cantonese CMMLU 25.26 to 27.44, medical 46.30 to 46.23 fused only (47.87 adapted), finance 0.32 to 0.35. SemAlign, Llama2-7B-Chat target from 13B: MMLU 44.2 to 50.3 (source 52.9), GSM8K 16.07 to 19.21 (source 20.55); from CodeLlama-13B-Python HumanEval 14.05 to 20.12 vs source 47.56.
- Conditions: Gains are small, MC-style benchmarks, 2000 calibration examples, sensitive to alpha (large alpha degrades). SemAlign mostly within the Llama-2 family, single seed, needs LoRA training per layer, and stays far below the finetuned source. Transport and Merge excerpt does not state tokenizers.
- Relevance: Closest to our expert-transplant and skill-top setups: cross-architecture weight fusion exists but delivers a few points and nothing near the donor. Consistent with our +3 MMLU distillation heal. SemAlign's frontier-layer, shallow-to-deep schedule is a heal design we could borrow for the reader study.

### Symphony-MoE (permutation-aligned FFN experts from separately pretrained models)

- Source: Symphony-MoE: Harmonizing Disparate Pre-trained Models into a Coherent Mixture-of-Experts, Wang, Peng, Yu, 2025 (v2 Nov 2025). https://arxiv.org/html/2509.18542v2
- What it does: Builds one 4-expert top-2 MoE from four different fine-tuned or pretrained dense models with identical architecture. Stage 1 has no training. Embeddings use MergeKit selective linear merge, attention uses SLERP, norms are averaged, all with weight 0.25. The FFNs of models 2 to 4 are aligned to an anchor model (the general one) by the Hungarian algorithm on FFN post-activations over a 10.4M-token calibration set, then the permutation is applied to W_up, W_gate and W_down. Stage 2 trains a randomly initialized linear router, with Switch load-balance loss weight 0.01. The paper says experts, backbone and router are all trained together on 5B tokens (6 epochs, lr 5e-5, 24 V100s).
- Numbers: 1.5B x4: ID average 44.07 vs Drop-Upcycling 41.10, BAM 39.55, BTX 33.94; OOD MedCQA 35.26 vs 32.90 / 28.97 / 26.92. The general dense anchor expert alone scores 40.95 ID and 30.39 MedCQA. 0.5B x4: 31.77 vs BTX 26.46. Llama-3.2-1B x4: 47.18 vs Drop 43.81. HumanEval at 1.5B: code expert 43.41, Symphony 42.39. Ablation at 1.5B: no functional alignment drops ID from 44.07 to 33.94 (OOD 35.26 to 26.92); aligning only 80% of neurons gives 43.57; naive attention merge without SLERP 37.28; biased calibration data 39.70. Inter-expert CKA after naive merge is 0.65 to 0.75 (collapse), alignment restores it.
- Conditions: Identical architecture required, and the sources are same-width fine-tunes (Qwen2.5-Base on Alpaca, Qwen2.5 on MetaMathQA, Qwen2.5-Coder on CodeAlpaca, Qwen2-Base on SciQAG), so partly different pretraining lineages but same shape. Only 0.5B to 1.5B tested. The router is learned during a 5B-token continued-pretraining run with experts trainable, so it is not a short heal. The authors say alignment quality depends on calibration diversity. Gains are measured over BTX-style naive averaging, and the paper does not show the MoE beats each specialist on its own domain (code expert 43.41 vs Symphony 42.39).
- Relevance: This is the only 2025 method that directly attacks our failure that MLP or layer swaps between diverged same-shape specialists (Qwen2.5-Math or Coder into Instruct) give chance. Its fix is neuron permutation matching on activations before putting experts side by side, plus an SLERP attention trunk. We never permutation-aligned run 16b's experts, so run 16b was the no-alignment arm. The ablation (no alignment loses about 10 ID points) matches what we saw. Caveat: the router heal here is far larger than our minutes-to-an-hour budget.

### MergeME (heterogeneous experts via projectors, perplexity routing, Dare/Ties trunk)

- Source: MergeME: Model Merging Techniques for Homogeneous and Heterogeneous MoEs, 2025. https://arxiv.org/html/2502.00997v3
- What it does: Homogeneous case: merge the non-FFN layers with Dare or Ties, keep FFNs separate, and insert a top-2 router. The best variant keeps each expert's own attention separate, so attention and FFN of the same expert come from one model. Training-free option: PPL routing, where each expert's reciprocal perplexity on the input sequence sets the expert weights. Heterogeneous case (different width and depth): shared embedding and head initialized by averaging, with smaller-dim experts zero-padded; per-expert Proj-in and Proj-out random-init MLPs map d_m to d_i and back; attention is not merged; a sequence-level router acts on mean token embeddings. Then all parameters are fine-tuned.
- Numbers: 1B scale, 6 benchmarks (MBPP, HumanEval, MATH, GSM8K, NQ, TriviaQA). Averages: Base 7.47, Math expert 9.34, BTX 11.72, Ties MoE 12.52, Dare MoE 12.86, full upcycling 16.05 (at 1258 vs 448 param-B x token-B). No fine-tune, separate attention plus PPL routing: 8.08 vs random routing 6.92 and Dare dense merge 7.11. Heterogeneous: adding a Math TinyLlama (1.1B, 22 layers) expert gives 13.34; adding a Math Olmo (1B, 16 layers) gives 11.17; 3-expert baselines 10.54 and 10.88. TinyLlama-Math alone 10.44, Olmo-Math alone 4.13.
- Conditions: All dense experts except the heterogeneous ones were continued-pretrained from one Base-1B (100B tokens each), so the homogeneous case is a true same-lineage family. Fine-tune of 40B tokens in the MoE stage. Heterogeneous projectors are random init and trained jointly in the 40B-token stage, with no separate schedule. Routing is per sequence, and the heterogeneous math expert ranked second in routing rather than first. PPL routing doubles the forward cost and misroutes MBPP (code expert gets 22%). Only 1B and three domains.
- Relevance: It is the one paper that composes experts from different shapes into a single network, which is our Sapien setting. The recipe is projector in and out plus sequence-level routing, which matches our 'skill tops routed per sequence'. Weakness for us: the 40B-token fine-tune is far above our budget, and the gain over the best expert is small (Olmo-Math expert 4.13 alone but 11.17 inside). Note the keep-attention-separate finding: experts that carry their own attention work better than FFN-only transplant, in line with our result that MLP-only swaps lose the skill.

### FlexOlmo (frozen public anchor + router embeddings from domain embeddings)

- Source: FlexOlmo: Open Language Models for Flexible Data Use, Allen AI, NeurIPS 2025. https://arxiv.org/html/2507.07024v4
- What it does: Experts are trained independently but coordinated by a shared frozen anchor. Each owner builds a 2-expert MoE from the public model's FFNs, freezes the public expert and shared attention, and trains only its own expert FFN and router embedding. At merge time the router is the concatenation of the per-expert router embeddings, so no joint training is needed. Router embedding init: mean of GRIT embeddings of samples from that expert's domain, then fine-tuned. A negative bias per expert helps experts compete.
- Numbers: 7B OLMo-2 (32 layers, hidden 4096), about 50B tokens per expert, 8 experts, 37B total and 20B active (top-4). About 41% relative gain over the public model on 31 tasks and about 10.1% over BTM, the best baseline. 4-expert setup: 47.8 for FlexOlmo vs 46.3 for a 1x-FLOPs unrestricted MoE and 51.5 for the 2x-FLOPs one. Removing the coordinated (anchor) training drops the 4-expert average from 47.8 to 38.8. GRIT router init vs public-model embeddings: 46.7 vs 43.5. Router tuning, optional, uses about 5B proxy tokens. Performance plateaus after 4 active experts.
- Conditions: All experts start from one 7B public model, and the anchor is what makes separately trained experts mergeable. Training each expert freely drifts apart and merging fails (38.8). Model soup and BTX were weak. Only same-lineage experts, 7B scale.
- Relevance: Answers the question of what alignment makes separately trained experts usable: a shared frozen anchor during expert training, not post-hoc repair. It also gives a training-free router init (domain embedding mean) that matches our need for a heal under an hour. The lesson for Sapien is that specialists we mine must be trained against the base (frozen attention and anchor FFN), not taken from the Hub as diverged fine-tunes. For donors already trained, this method does not apply directly.

### DUME (closed-form ridge-regression router, training-free)

- Source: Training-Free Dynamic Upcycling of Expert Language Models, Fani and Ersoy, 2026. https://arxiv.org/abs/2603.29765 and https://arxiv.org/html/2603.29765
- What it does: Averages the non-MLP parameters of the dense experts and makes each expert's MLP an MoE expert with top-1 routing. The router is the closed-form ridge solution W = (A + lambda I)^-1 b, with statistics accumulated by forwarding each domain's data once and sending each token's features to its own domain's expert. Router columns are normalized. No gradients, and new experts can be added later by updating the statistics.
- Numbers: Normalized to 100 = best dense expert. 115M Llama, 5 domains (CLM): DUME 92.8, DUME+ 93.9, BTX 91.9, Oracle 94.8, model averaging 83.4, random routing 82.4. 3B Llama, 4 HF experts (math, multilingual, coding, instruction following): DUME 95.2 +/- 1.7, Oracle 93.4, averaging 75.3, random routing 70.8. Per domain: math 91.1, multilingual 100.4, coding 102.1, instruction following 87.2. Abstract: up to 97.6% (CLM) and 102.1% (reasoning) of the dense expert. Lambda 0.01 best for reasoning; 2 tokens per paragraph already reach about 90%.
- Conditions: Experts must share the architecture, and in the CLM setup all came from one 115M seed (the 3B experts were Hub fine-tunes of one Llama). Training data needs domain labels. Routers beyond layer 1 are approximate since their input distribution depends on the expert. Only 4 to 5 domains, small models, math expert falls short (91.1).
- Relevance: Gives a router init that costs seconds and needs no heal, and it beats an oracle domain router on average. It fits our kept-expert MoE work: a ridge router fit on domain-labeled hidden states is exactly the 'router started from a ridge fit on domain labels' we listed as the next step after run 16b, where the router barely moved from the base bias. It does not fix the quality of diverged experts: for true fine-tunes it works, and our cosine 0.2 to 0.5 specialists are outside its tested range.

### Branch-Train-MiX (BTX) and Branch-Train-Merge as baselines

- Source: Branch-Train-MiX: Mixing Expert LLMs into a Mixture-of-Experts LLM, Sukhbaatar et al., Meta, 2024. https://arxiv.org/html/2403.07816
- What it does: Branch a seed (Llama-2 7B) into math, code and Wikipedia experts trained separately, keep the seed as a fourth generalist expert, then combine their FFNs as MoE experts, average attention and everything else, and fine-tune the whole MoE on all data for about 80B tokens to learn the router.
- Numbers: Average over Math, Code, Knowledge, Reasoning, MMLU: seed 40.7; dense multi-domain 44.5; sparse upcycling top-2 46.3; BTM top-2 43.4; BTX top-1 sample 47.3; BTX top-2 47.9. Math 8.6 to 27.4, code 16.8 to 34.0, MMLU 46.1 to 52.5. Compute: BTX 926 GPU-days and 533B tokens vs sparse upcycling 1007 GPU-days and 252B tokens. Without a load-balance loss the code expert dies (HumanEval 27.4 with it, 19.5 without).
- Conditions: All experts branch from one seed, so FFNs stay close and averaging works. Experts trained on 200B tokens each. MoE fine-tune about 80B tokens. Router init is not stated (small new matrices). The Symphony-MoE and FlexOlmo papers report BTX-style averaging is weak when experts are less similar (BTX 33.94 vs 44.07 at 1.5B; Flex 38.8 without anchor).
- Relevance: The reference for the pattern we want. The router is the only new thing and it is learned in a long heal; BTX shows the generalist must stay in the expert set (the seed is the fourth expert) and load balancing matters. This matches run 16b, where routing 92% to the base was a router stuck on the generalist. Its compute is far over budget.

### PHATGOOSE (post-hoc per-module gates over separately trained LoRAs)

- Source: Learning to Route Among Specialized Experts for Zero-Shot Generalization, Muqeeth et al., ICML 2024. https://ar5iv.labs.arxiv.org/html/2402.05859
- What it does: Each contributor trains a LoRA expert, then freezes it and trains a sigmoid gate vector (init zeros) in front of that LoRA for 100 steps on the same data. At inference the gate vectors are standardized, compared to the activation by cosine similarity, top-2 chosen per token per layer, softmax weighted. Only the gate parameters are shared beyond the LoRA.
- Numbers: LM-adapted T5.1.1 XL, 36 experts (T0 held-in pool) and 166 (FLAN pool), LoRA rank 16, 1000 steps per expert. T0 held-out average: PHATGOOSE 56.9, Arrow 55.1, best individual expert 52.8, T0-3B multitask 51.6, average merging 45.4, oracle 57.2. BBH: PHATGOOSE 34.9, multitask 34.9. FLAN pool BBH/BBL: 35.6 / 35.2 vs averaging 34.6 / 34.0. Joint gate and LoRA training scored 46.2 vs 51.8 for post-hoc gates (T5 Large).
- Conditions: Encoder-decoder T5 only, a single shared rank, identical base. Scaling the expert pool did not reliably help. Weak correlation between routing and the oracle choice (Pearson -0.2). Needs the experts' own data for the 100-step gate pass. Decoder-only LLMs were left to future work.
- Relevance: A cheap router recipe (100 steps per expert, no joint training) for LoRA-form donors. Fits our LoRA-only programs and the 'heal under an hour' limit. Result worth taking: gates trained after the expert is frozen beat joint training by 5.6 points, so do not co-train the router with experts when the experts are already good. Limits: LoRA deltas on one host, no cross-family case, and old base models.

### Drop-Upcycling (partial re-initialization of copied experts) and Dense-to-MoE upcycling with heterogeneous experts as a caution

- Source: Drop-Upcycling: Training Sparse Mixture of Experts with Partial Re-initialization, Nakamura et al., ICLR 2025. https://arxiv.org/html/2502.19261 ; UpIT: Upcycling Instruction Tuning from Dense to MoE via Parameter Merging, ACL 2025. https://arxiv.org/html/2410.01610v1
- What it does: Drop-Upcycling copies the dense FFN into 8 experts, then re-initializes a random fraction r of the FFN intermediate dims (rows of W_down, columns of W_gate and W_up, drawn from a normal with the original mean and std) differently per expert to force specialization. UpIT instead builds experts from fine-tuning checkpoints, expands with genetic merging, and initializes each router row by pre-optimizing it on the data where that expert has the lowest perplexity.
- Numbers: Drop-Upcycling: 8x1.5B average 40.3 (r=0.5) vs naive upcycling 37.4 vs from scratch 34.5; r=1.0 gives 38.9; 8x152M 19.7 vs 17.9 vs 16.6; 8x3.7B (5.9B active) at 500B tokens scores 44.4 vs 13B dense 44.5 at about 3.8x the FLOPs. UpIT (7B LoRA-MoE, 8 experts): 52.21 vs LoRAMoE-SFT 49.99; router init ablation 52.21 with skilled init vs 49.96 without and 49.30 random.
- Conditions: Both start from one dense model, so no cross-model gain is claimed. Drop-Upcycling pays off only over long training (hundreds of billions of tokens); at small scale margins are small and naive upcycling is better early. UpIT results are single runs. A router initialized from the data where each expert is best is worth 2.3 points.
- Relevance: Lesson about routers, not about transplant: random router init costs about 2 points (UpIT) and a naive copy of the same FFN gives no specialization, which matches our run 16b where the router never left the base. If the experts are different donors, specialization already exists, so Drop-Upcycling is not the lever. Data-lowest-perplexity router init is a free, fast step for our heal.

### REAP expert pruning (router-weighted expert activation), no healing

- Source: REAP the Experts: Why Pruning Prevails for One-Shot MoE Compression, Lasby et al. (Cerebras), ICLR 2026. https://arxiv.org/abs/2510.13999 and https://arxiv.org/html/2510.13999v1 (both opened; HTML read only to char 100000)
- What it does: Scores each expert by the average over its routed tokens of router gate value times expert output norm, then deletes the lowest experts per layer with their router rows. One calibration pass, zero training. Our run 19 saliency is the same idea.
- Numbers: Qwen3-30B-A3B at 50% experts removed, REAP code avg 0.557 vs baseline roughly 0.58 (25%: 0.575); math avg 0.875 vs 0.887; MC avg 0.518 at 50% (0.669 at 25% vs 0.721 baseline). Merging at 50%: M-SMoE code 0.413, HC-SMoE 0.379; frequency 0.470, EAN 0.552. Creative writing at 50%: REAP 0.718 vs 0.811 baseline, HC-SMoE 0.008. Qwen3-Coder-480B code avg 0.644 at 50% vs 0.660 baseline; SWE-Bench-Verified 0.522 vs 0.540. Kimi-K2 0.646 vs 0.659.
- Conditions: Calibration is 1,024 samples packed to 2,048 tokens (models up to 110B); 12,228 long samples for larger models. Domain calibration matters: evol-codealpaca for code, C4 for MC. Calibrating code models on C4 gave incoherent output and 0% on some code tasks. Generative tasks hold at 50%; MC tasks lose about 13% on average at 50%. Mean coding drop 2.8% at 25% and 8.0% at 50%. Three seeds only for models up to 50B. Paper says no fine-tuning, all results one-shot.
- Relevance: Matches our run 19 (saliency works, frequency and random collapse; our Qwen3-30B-A3B code within 4 to 5 points at 64 of 128 experts). It confirms the recipe is robust at scale and that pruning beats merging for generation, which argues against expert merging as a composition step. Our MMLU drop to 44.9 is the same trade as their domain-calibration warning: calibrate on the skill you keep, and the other skills go.

### Router KD: heal only the router after expert pruning or merging

- Source: Is Retraining-Free Enough? The Necessity of Router Calibration for Efficient MoE Compression, Hyeon and Do, 2026. https://arxiv.org/abs/2603.02217 and https://arxiv.org/html/2603.02217v1 (opened; appendix tables partly cut off)
- What it does: After pruning, editing or merging experts, trains only the router (about 0.04% of parameters in Qwen3-30B-A3B) to match the original model's next-token distribution on unlabeled text. The authors blame post-compression loss on router and expert mismatch.
- Numbers: Qwen3-30B-A3B-Instruct-2507 at 62.5% retention (80 of 128 experts): REAP to REAP-R (with Router KD) GSM8K 0.8802 to 0.8832, HumanEval-instruct 0.9146 to 0.9207, MBPP 0.6220 to 0.6180 (MBPP cell hard to parse, verify). Cost is about 2 hours on Qwen3 and 40 minutes on Mixtral. Settings: C4, 3,000 samples, 1 epoch, batch 2 x accum 4, lr 5e-5, length 512 tokens, so at most about 1.5M tokens. I did not find a quoted step count; 375 steps is my inference.
- Conditions: Gains are large for fine-grained MoEs (Qwen3, 128 experts) and marginal for Mixtral (8 experts). Gives nothing when compression already collapsed the model, and can hurt when compression had raised the score. Six methods tested (REAP, CFES, MoBE, TD-MoE, HC-SMoE, M-SMoE). The per-method average gains were in tables I could not read, so the size of the gain on weaker methods is not confirmed.
- Relevance: Cheapest heal found: a few million tokens, router only. Our run 19 heals the whole model for 500 steps; at 16 experts per layer (collapse) this paper says router KD will not rescue it. Worth trying on our 64-expert and 32-expert Qwen3 cells before any full heal, because it separates routing damage from expert damage.

### Minitron: width pruning plus logit distillation

- Source: Compact Language Models via Pruning and Knowledge Distillation, Muralidharan et al. (NVIDIA), NeurIPS 2024. https://arxiv.org/html/2407.14679v2 (opened)
- What it does: Prunes a trained model by activation-based importance (width: heads, MLP neurons, embedding channels; or depth by PPL or Block Influence), then retrains by distilling from the unpruned parent. Best-practice list in Section 4.1.
- Numbers: Nemotron-4 15B (MMLU 66.6) to Minitron 8B (63.8) and 4B (58.6), each with 94B retraining tokens (under 3% of the original data, claimed up to 40x fewer tokens than from scratch). Lightweight retraining used for ablations is 1.8B tokens (400 steps). Width-only 7.74B model: LM loss 2.049 after 1.8B tokens, vs depth+width 2.062, depth-only PPL 2.155, BI 2.177. Iso-compute 4B ablation: pruned-then-distilled MMLU 37.81 vs 24.57 for conventional retraining. Importance estimated on 1,024 samples.
- Conditions: Width beats depth after retraining for models up to 15B, but depth-pruned candidates look better before retraining (rankings flip after about 200 steps, 0.8B tokens). Distill with KL, not LM loss. Add intermediate-state and embedding distillation only when depth is cut a lot. Single-shot importance is enough. Prune the model closest to target size, and prune the final-stage checkpoint.
- Relevance: Sets the budget reality: even the best recipe used about 1.8B tokens for ranking and 94B for the final model, not under 100M. For us it says a 100M-token heal buys a ranking signal and a recovery of loss, not a finished model. It also says width pruning is more recoverable than layer deletion, which fits our run 15 failure (50% layer carve plus LoRA gave MMLU 34 to 41).

### Layer-block deletion plus QLoRA heal (similarity-chosen block, deepest layers)

- Source: The Unreasonable Ineffectiveness of the Deeper Layers, Gromov et al., 2024. https://arxiv.org/abs/2403.17887 and https://arxiv.org/html/2403.17887v2 (both opened)
- What it does: Finds the contiguous block of layers whose input and output have the smallest angular distance (usually deep layers, never the last), deletes it, and heals with QLoRA on C4.
- Numbers: Healing is 5,000 steps at batch 16: 164M tokens for 7B models (Llama-2 7B, Mistral 7B, Qwen 7B, Phi-2) and 328M tokens for 13B to 70B. QLoRA rank 64 by default (2 for Llama-2 7B, 8 for Llama-2 70B, 4 for Mistral 7B). MMLU stays near baseline until roughly 45 to 55% of layers removed for Llama-2, about 35% for Mistral 7B, about 25% for Phi-2, and about 20% for Qwen. Single A100 40GB.
- Conditions: Healing restores C4 loss to near unpruned and makes it rise smoothly, but QA accuracy still shows a sharp cliff, so loss recovery does not mean skill recovery. Healing only shifts the cliff slightly. Simply deleting the last-but-one deepest block performs poorly without healing but comparably after healing. Qwen has shallow islands of high similarity, which gives it a shorter safe region.
- Relevance: This is the one recipe that fits under 100M to 330M tokens, and it comes with a hard ceiling. It matches our run 15 (14B to 22 of 48 layers fails; Qwen is the most fragile family at about 20% deletable). Use it only for cuts under about 20% of layers on Qwen2.5 models, and judge by a skill benchmark, not by loss.

### ShortOPD: recovery of pruned models by on-policy distillation from the parent

- Source: ShortOPD: Recovering Pruned LLMs with Short-to-Long On-Policy Distillation, Zhang et al., arXiv 2607.13124 (v2, Aug 2026). Abstract at https://arxiv.org/abs/2607.13124; tables read from the PDF text
- What it does: Student (pruned) samples its own rollouts and the frozen unpruned parent scores every generated token (generalized JSD). A repetition gate keeps rollouts short while the student loops, then lengthens them. It targets the case where pruned models keep multiple-choice ability but lose free-form generation.
- Numbers: Qwen3-4B-Instruct with 9 of 36 layers deleted (BI, 25% of depth): generation average falls from 75.17 (dense) to 5.71 pruned. One epoch of ShortOPD (710 steps, about 250M rollout tokens, 8.5 h on 8 GPUs) gives 48.46 (64.5% of teacher); 2 epochs 53.45; 3 epochs 55.41 (73.7%). At 1 epoch GSM8K 62.70, MATH 42.00, HumanEval 43.90, MBPP 49.81. Across width-only and hybrid pruning, 25% pruned scores 16 to 20 restore to 51 to 59 (69 to 78% of teacher). Fixed 8192-token rollouts cost 869M tokens and 35.9 h for only +0.6 points.
- Conditions: Needs about 25% as a critical point; with stronger compression outputs turn incoherent rather than repetitive and the paper treats that as outside the recoverable regime. Corpus is a mixed math, code and instruction set of 29,585 prompts. Pass@64 of the pruned model is 91% on the probed task, above the dense greedy 88%, so the skill is demoted not erased. Rollout cost is larger than the token count suggests because teacher scoring runs on every token. It beats SFT without KD, KD and SeqKD by 1.6 to 4.4x.
- Relevance: Direct evidence for a failure we may be misreading: layer-deleted models can keep logits-level ability yet collapse on generation, so our MMLU-only checks on carved models can mislead, and our GSM8K and MBPP numbers are the honest signal. Also gives a heal recipe for any seam or carve that sits at a 25% cut: on-policy KD from the parent, about 250M rollout tokens, not under 100M.

### MoE-to-dense: select experts, concatenate into one dense FFN, distill from the MoE

- Source: Pruning and Distilling Mixture-of-Experts into Dense Language Models, KRAFTON and KAIST, arXiv 2605.28207, May 2026. https://arxiv.org/html/2605.28207v1 (opened; read to char 100000 of 173k)
- What it does: Scores all experts in each layer (best scorer DO-ACP: activation-weighted conditional probability with D-optimal greedy selection), keeps the top-K per layer, concatenates them into one dense FFN of width k x d_expert, copies attention, embeddings and norms from the MoE teacher, then distills from the MoE with forward KL. Best setting is K equal to router top-k (8), so no merging happens.
- Numbers: Qwen3-30B-A3B to a 3.3B dense student. Average of Winogrande, HellaSwag, ARC-E, ARC-C, MMLU: teacher 79.65; student 43.41 after 0.3B tokens and 58.10 after about 4B tokens. Dense-to-dense pruning (Qwen3-32B to 3.4B) 33.28 and 51.84; random FFN with teacher attention 32.70 and 45.44; random init 30.15 at 0.3B. Distillation on FineWeb-Edu, batch 384, length 4096, lr 1e-4 to 1e-5. Teacher gap is still about 21.5 points at 4B tokens. MoE-to-dense trains at 73 s/step vs 116 s/step.
- Conditions: Expert selection helps more with many experts: advantage over random baselines is about +10.7 to +13.3 points on Qwen3 (128 experts) and only +3.3 to +3.7 on GPT-OSS-20B (32 experts); scoring spread 7.1 points on Qwen3 vs 1.6 on GPT-OSS. Grouping and merging matter little (about 1 point). The paper has no random-expert-selection ablation and no number at 100M or 1B tokens; the text is internally inconsistent on the gain sizes (13.3 vs 10.7). Dense student shares the teacher's attention, so this is a same-family transplant.
- Relevance: Closest published result to our expert-transplant test (MMLU +3 with a distillation heal). It supports transplant from a fine-grained MoE with enough experts, and that saliency-style selection beats a random FFN by 10 or more points at 0.3B tokens. It also predicts our controls: scrambled or random expert controls should sit near 33 to 45 on their scale, and the 0.3B-token point is already 43. Note it keeps the teacher's attention; our same-width dense base with its own attention is a weaker setup.

### MoNE: replace pruned experts with a constant 'novice' vector

- Source: MoNE: Replacing Redundant Experts with Lightweight Novices for Structured Pruning of MoE, 2025. https://arxiv.org/html/2507.00390 (opened; read to char 100000 of 114k)
- What it does: Prunes experts by redundancy score (variance of expert output times average routing score when selected), then substitutes each pruned expert with a constant vector equal to its mean output on calibration data. Costs no extra compute at inference.
- Numbers: OLMoE at 25% pruned with 100 Zyda2 samples: 61.04 average zero-shot vs 58.32 for best baseline (FLAP); the original is 66.63. Qwen2-57B-A14B at 25%: 71.75 vs 71.89. Largest ratio with at most 1% loss: OLMoE 16%, Moonlight 16%, DeepSeek-V2-Lite 20%, Qwen2-57B-A14B 25%, Qwen3-30B-A3B 24%. After 2B tokens of continued pretraining at 25% on OLMoE: MoNE 64.18, RS routing-score pruning 64.09, MC-SMoE 61.31, original 66.63.
- Conditions: Needs only 100 to 1,000 calibration samples with no weight updates. Calibrating on pretraining text (Zyda2) dropped math by up to 49% on OLMoE; task-specific calibration on instruct models held the drop to about 1%. After 2B tokens of healing the methods nearly tie (64.18 vs 64.09), so the novice term helps mainly at zero or low healing. Evaluated on base models, mostly zero-shot multiple choice.
- Relevance: Adds a free correction to our slimming (run 19): at the collapse point (16 experts per layer, 5B stored) replacing deleted experts by their mean output may keep the residual stream in range and let the short heal start from a better point. Also confirms that calibration data decides which skill survives, which is the point of skill extraction. Its own healed gain is small, so treat it as a cheap init, not a recovery recipe.

### Puzzle: block-wise local distillation search, then short global distillation

- Source: Puzzle: Distillation-Based NAS for Inference-Optimized LLMs, Bercovich et al. (NVIDIA), ICML 2025. https://arxiv.org/abs/2411.19146 (abstract opened; only the abstract-level numbers verified)
- What it does: Trains alternative cheaper blocks (smaller attention, smaller or removed FFN) for each layer by local distillation against the parent's block outputs, then picks one variant per layer with mixed-integer programming under a latency or memory budget, then runs global distillation on the stitched result.
- Numbers: Llama-3.1-70B-Instruct to Nemotron-51B and Llama-3.3-Nemotron-49B with at most 45B training tokens (vs about 15T for the parent): 2.17x throughput on one H100, 98.4% of the parent's benchmark accuracy.
- Conditions: 45B tokens is above our 100M budget by 450x. The abstract-level source does not give the share spent on block-wise local distillation versus global distillation, nor per-block cost, so I cannot say how much of the recovery comes from the cheap local stage. Hardware-aware and per-layer; the gain comes from changing layer types, not from deleting layers.
- Relevance: Relevant to the seam problem: its local, block-to-block distillation is the same shape as our stitch study (a trained seam is near lossless when the path continues through the donor's own layers). It suggests training each seam against the donor's block output locally, which is cheap and parallel, before any global heal. The 45B-token figure is the honest cost of the global stage at 70B scale.