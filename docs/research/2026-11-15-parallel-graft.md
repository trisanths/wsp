# Parallel grafts: a redesign of cross-family grafting (rounds 22 and 23)

Branch `parallel-graft`, 2026-11-14 to 2026-11-15. Code: `wsp/pgraft.py`,
`modal_round22pg.py`, `modal_round23pg.py`, tests in `tests/test_pgraft.py`.
Raw rows in `results/round22*_rows.json` and `results/round23_rows.json`.
The short version is at the end.

## Why a redesign

Rounds 1 to 21 tried to graft parts of one model into another: neuron
clusters, attention heads, experts, whole blocks, and routed specialist
MLPs. A random part of the same size from the same donor matched almost
every one of them, so the part's own skill never transferred. The audit
and stitch study found that the input stitch had been the first
bottleneck. The attention sink was broken, and a linear map lost 1.4 to
2.2 nats of the donor's loss. Training the stitch on the donor's own loss
fixed that, but only when the two models share a tokenizer. Even with a
lossless stitch, runs 14 and 14b showed the base could not use a written-in
block. Everything a random block of the same donor gave was capacity, not
skill.

Taps (runs 5 to 9) were the one thing that worked. A Qwen2.5-7B tap
brought Llama-3.2-1B from 47.5 to 67.9 on MMLU, where LoRA reached 50.3.
But a tap runs the whole donor beside the host, which the "one network"
rule rules out.

The redesign keeps one network and goes after the two failure points
directly. The first is functional alignment that works across tokenizers.
The second is a branch the base can take off cleanly, with controls that
can tell donor knowledge apart from added capacity.

## The design

```
h_j (base residual after layer j)
 |--------------------------------------------------------------> + --> base layer j+1
 '--> Enc --> [donor sink at pos 0] --> Donor[k..m] (frozen) --> (out - in) --> Dec --> gate --'
```

- **Base:** `unsloth/Llama-3.2-1B-Instruct`, 16 layers, MMLU dev 48.0. All of
  its weights stay frozen.
- **Donor:** `Qwen/Qwen2.5-7B-Instruct`, 28 layers, MMLU dev 71.5. It comes
  from a different company and uses a different tokenizer, which is the
  hard case this design is for.
- **Donor slice:** layers 10 to 20, frozen, about 2.6B parameters. Run 7
  found that answers become readable near layer 20 and not before layer
  16. Layer 10 as the start passed the alignment gate (below).
- **Insertion point:** the branch reads after base layer 7 (about 50% depth)
  and writes back to the same point.
- **Enc** (13.8M, trained): a linear map on rms(h) plus one residual MLP,
  with a learned output scale. It is the translator into the donor.
- **Dec** (rank 64 in v1, 256 later; trained): reads the donor's
  contribution (block output minus block input), not the raw output, so
  Enc to Dec cannot route around the donor.
- **Gate** (2k, trained): per token, reading rms(h). With the gate or Dec
  at zero, the network is exactly the base.
- **Sink:** the donor's mean state at position 0 is fed at position 0. A
  broken attention sink was the main cause of the earlier stitch failures.
- **The donor never sees its own tokenizer.** It runs on the base's token
  positions, on the states Enc makes.

### Training stages

1. **Alignment (Enc only, donor frozen).** The donor gets Enc's states at
   layer k and runs to its own LM head. Its predictions are compared with
   what it predicts natively on the same text. Positions are paired across
   tokenizers by character end offsets (`wsp.align`). The loss is
   KL(native || grafted) over the donor's own vocabulary, so it works for
   any tokenizer pair. A 100-step warm-up adds normalized MSE plus cosine
   against the donor's native layer-k state.
2. **Quality gate.** Survival is the grafted donor's next-token accuracy
   divided by its native accuracy, on held-out text. It needs no
   benchmark. If survival is low, don't train further.
3. **Integration.** Dec and the gate train on the base's objective. The
   objective changed across rounds; see each run.

### Controls (the point of the design)

- **Identity-donor control:** the same Enc, Dec, gate, budget, data, and
  steps, with the donor skipped. Dec reads Enc's output instead. If this
  arm matches the graft, the donor did nothing and the graft is just a big
  adapter.
- **Matched LoRA:** a LoRA on all of the base's layers with the same
  trainable count (rank 20 or 21), trained on the same data for the same
  step count. Later runs add a LoRA sized to graft + LoRA together.

### Data

All training text comes from `wsp.healdata` pools, never from benchmarks.

- **Alignment:** WikiText plus knowledge QA (TriviaQA, NQ-open, SQuAD,
  WikiQA) plus medical QA (medical_meadow, PubMedQA, MedQuAD, and others).
- **Integration:** "domain" text (knowledge plus medicine), where the base
  is weak, and "general" text (UltraChat, Alpaca, WikiText).
- **Scoring:** MMLU dev 1000 and held-out 2000, by the log-probability of
  " A" to " D". MMLU is never trained on. Held-out domain and general text
  are scored with the graft on and off.

## Run 22: does alignment work across tokenizers?

### Alignment gate

The gate ran on held-out pool text: 17.5k offset-paired positions, where
the donor's native next-token accuracy is 56.1%. Enc trained for 600 steps.

| Donor start k | Survival untrained | Survival trained | Top-1 agreement | KL (nats) |
|---|---|---|---|---|
| 4 | 1.6% | 90.7% | 73.9% | 0.49 |
| 10 | 1.1% | 89.6% | 71.8% | 0.55 |
| 16 | 1.5% | 86.6% | 68.4% | 0.68 |

**Yes.** On Llama's states, Qwen keeps about 90% of its own next-token
skill, which every earlier stitch across tokenizers failed to do. Each
alignment takes about 2 minutes on an H100.

### Full comparison (v1)

k is 10. Integration trains Dec (rank 64) and the gate on the base's LM
loss, with a utility hinge (removing the graft should hurt), stochastic
depth on base layers 8 and 9, and gate L1, all annealed. Every arm runs
1600 steps; for the graft that is 600 of alignment and 1000 of
integration. 2 seeds.

| Arm | Trainable | MMLU dev | MMLU held-out | Domain LM loss |
|---|---|---|---|---|
| Base | 0 | 48.0 | 44.6 | 2.676 |
| LoRA r20 | 14.1M | 46.4, 45.7 | 44.3, 43.7 | 1.912, 1.940 |
| Graft | 14.1M | 43.0, 40.5 | 40.5, 38.6 | 2.293, 2.303 |
| Identity control | 14.1M | 42.7, 43.0 | 40.2, 41.9 | 2.363, 2.347 |

The base does use the donor. The graft beats the identity control by 0.06
nats on both seeds, against a seed spread of 0.010 to 0.016 within each
arm. But LoRA is 0.37 nats better, and both graft arms cost 4 to 7 MMLU
points.

## Run 22b (v2): let the graft adapt, guard against drift

v2 made three changes at once:

- Enc keeps training in stage 2 at lr 1e-4.
- Dec rank goes up to 256.
- The gate L1 drops tenfold, and every trained arm gets a KL penalty
  against a frozen base copy on general batches, the same form as
  `wsp/synth.py`'s preservation term.

The identity control gets the same changes.

| Arm | MMLU dev | MMLU held-out | Domain LM loss |
|---|---|---|---|
| LoRA r21 (14.8M) | 49.3, 46.7 | 46.0, 44.5 | 1.915, 1.944 |
| Graft (15.2M) | 44.9, 46.0 | 42.1, 42.4 | 2.359, 2.402 |
| Identity control | 47.9, 47.1 | 44.6, 44.8 | 2.538, 2.566 |

The donor's edge over the identity control grew to about 0.17 nats, with a
seed spread of 0.03 to 0.04. The MMLU cost fell to about 2.5 points, but
LoRA is still far ahead on domain text.

## Run 22c: graft + LoRA vs. LoRA alone

The v2 graft (or the identity control) is frozen, then the same LoRA as
the LoRA arm trains on top. LoRA learns the text's format in every arm
here, so whatever difference remains comes from the donor.

| Arm | Domain loss before LoRA | After LoRA | MMLU held-out |
|---|---|---|---|
| LoRA alone | | 1.915, 1.944 | 46.0, 44.5 |
| Graft, then LoRA | 2.359, 2.402 | 1.919, 1.938 | 44.3, 44.7 |
| Identity control, then LoRA | 2.538, 2.566 | 1.917, 1.937 | 46.1, 45.5 |

**The edge disappears.** Whatever the donor added on this text, LoRA
learns from the same data.

## Run 22d (v3): rule out drift toward bypass

v3 is v2 plus the following:

- The stage-1 functional KL kept as a stage-2 regularizer (weight 1, every
  second step).
- Graft fidelity logged every 100 steps.
- The update ratio ||g * Dec|| / ||h|| per token, and the loss change with
  the graft off, on domain and general text separately.
- KL to the frozen base on held-out text.
- **lora2x:** a LoRA sized to graft + LoRA together (rank 43, 30.3M), with
  the staged arms' total of 3200 steps. It is the fair bar.

| Arm | Domain LM loss | MMLU held-out |
|---|---|---|
| Graft v3 | 2.557, 2.430 | 43.9, 43.3 |
| Identity control v3 | 2.538, 2.566 | 44.6, 44.8 |
| Graft v3, then LoRA | 1.930, 1.941 | 45.9, 44.1 |
| Identity v3, then LoRA | 1.917, 1.936 | 45.7, 44.7 |
| lora2x | 1.607, 1.634 | 46.2, 45.6 |

- **No bypass.** Fidelity stays at 0.87 to 0.91 through all of stage 2.
  But the regularizer also cost seed 0 its edge over the identity control,
  so bypass was not hiding a bigger donor effect.
- **The gates work.** The graft's update is 0.12 to 0.35 × ||h|| on domain
  text and 0.02 to 0.04 on general text. Its drift on general text is 0.013
  to 0.017 nats.
- **At equal total budget, more LoRA wins by about 0.3 nats.** On this data
  the donor slice is worth about zero.
- **A flaw in the metric:** every arm also trains on the general pools, so
  drift KL on held-out general text mixes regression with format learning.
  lora2x drifts 0.18 nats while its general loss falls from 2.61 to 1.86.
  Round 23 scores regression on legal text that no arm trains on.

## Diagnosis after round 22

There are three things that have to work:

1. **Input alignment:** the donor computes correctly on the base's states.
   Solved, with about 90% survival.
2. **Output reading:** the base turns the donor's output into better
   predictions. Weak. Dec only ever learned from the base's LM loss on
   dataset text; nothing pushed it to translate the donor faithfully.
3. **Training signal:** something has to reward using knowledge the base
   lacks. Missing. LM loss on QA text rewards only what the text holds, and
   LoRA can learn all of that directly.

Earlier rounds back this up. The tap worked because it trained on
answer-letter prediction, a signal that needs the knowledge. Run 14c
distilled from a 14B teacher: the base with a LoRA alone gained nothing,
while the base with frozen 14B layers written in gained 4 to 9 MMLU points.
That run was same-family and its heal text included MMLU auxiliary items,
so it is a diagnostic, not clean evidence.

## Round 23: three training signals

Each experiment uses the same three arms at one budget (15.2M) and 2
seeds: the graft (v3 setup), the identity control, and a matched LoRA. Step
counts match per experiment; the controls spend the graft's 600 alignment
steps on their first stage.

- **A. Answer-letter training.** Cross-entropy over A to D on about 8,000
  four-option items from CommonsenseQA, OpenBookQA, QASC, and LogiQA2.0
  (train splits), in MMLU's prompt format, plus KL to the frozen base on
  general text. MMLU is never trained on.
- **B. Cross-tokenizer distillation.** KL(donor || base) at offset-paired
  positions, over the 109,566 token strings both vocabularies share (85% of
  Llama's vocabulary). Llama 3 and Qwen2.5 both use GPT-2-style byte-level
  tokens, so equal strings are equal bytes.
- **C. Output alignment, then the task.** B's distillation first, then A's
  answer-letter training. For the graft, the distillation stage trains only
  Dec and the gate. It is the mirror of stage 1, so the decoder learns to
  translate before any task training.

**Scoring:**

- MMLU dev and held-out.
- Held-out questions from the training pools.
- The distillation gap to the donor on held-out domain text.
- Graft on versus off.
- Fidelity during training.
- KL to the frozen base on the healdata law pool, which no arm trains on.
  This is the main regression metric.

Before training, Llama differs from Qwen by about 0.81 nats over the shared
tokens, with 64% top-1 agreement.

### A learning-rate bug, and the corrected run

The first pass of round 23 was invalid. The gate learning rate (1e-2)
shut every gate within 50 steps (mean gate 0.02), so the graft arms were
the base. Diagnostics on one seed: with the gate frozen open, a glue
learning rate of 1e-3 damages the base (MMLU 23.4); glue 3e-4 with gate
1e-4 trains stably. The rows from the bad pass are kept in
`results/round23_badlr_rows.json` and `results/round23_diag_*_rows.json`.

**Corrected results (MMLU held-out, 2 seeds; base 44.6):**

| experiment | graft | identity | LoRA |
|---|---|---|---|
| A. answer letters | **49.2 / 48.3** | 46.7 / 45.8 | 47.0 / 46.7 |
| B. distillation only | 38 to 46 (all arms hurt) | | |
| C. distill, then answer letters | **48.6 / 49.1** | | 46.9 / 47.2 |

In A the graft also drifts less from the base on the law pool (KL 0.009
vs. 0.038 for LoRA). Distillation alone hurts every arm.

Caveat: these configurations were chosen while looking at MMLU held-out.
From the hillclimb on, configurations are chosen on MMLU dev and
in-distribution items, and held-out is used only to confirm a winner with
new seeds.

## Hillclimb (`modal_hill.py`)

A config-driven sweep. Base Llama-3.2-1B-Instruct, host layer 7, budget
matched across arms (real donor part, random units, identity, LoRA).
Domains and donors: medicine (Apollo2-7B), code (Qwen2.5-Coder-7B-Instruct),
math (Qwen2.5-Math-1.5B, then Qwen2.5-Math-7B-Instruct from the new
library), knowledge (Qwen2.5-7B-Instruct).

### h1: library parts, whole layers, blocks (1 seed)

- Library parts and whole-layer MLPs fed through a translator do not align
  (output R² from -0.39 to 0.18), and real units score the same as random
  units. A part pulled out of its own model's residual stream does not
  work on a translated input.
- Donor blocks keep 90 to 93% of their function and beat identity on the
  distillation gap, but LoRA distills better. No MMLU or MedMCQA gain.

### h2: trunk bundles and a knowledge climb (18 of 20 cells)

Trunk bundles run the donor's own layers to build each part's native input
and return only the named parts' outputs.

- Library parts in a trunk bundle: real is no better than random units
  (skill-screen transfer: code -0.009 vs. -0.010, math 0.283 vs. 0.318,
  medicine 0.096 vs. 0.067). The library's units carry no extra signal
  here beyond their layer's position.
- Knowledge climb (answer-letter training, 600 align, 1000 task steps):
  the block spanning donor layers 4..20 scored MMLU held-out **50.3**
  (in-distribution items 71.5%), against 45.7 for the round 23 block
  (10..20), 46.1 LoRA, 47.4 identity, 44.6 base. Other mutations
  (encoder depth 3, host 5 or 9, span 10..24, decoder rank 1024, 2000
  steps) scored 45.5 to 48.0. One seed, and selected on held-out, so h3
  re-tests it.

### Data leak found and fixed

The `medicine` pool included FreedomIntelligence/medical-o1-reasoning-SFT,
which is built from MedMCQA and MedQA questions. It is now in
`DERIVED_FROM` and moved out of the default sources, so medicine runs no
longer train on the benchmark they report. The h1 and h2 medicine rows
used it and are not evidence.

### h3: confirm and climb

New seeds for the h2 winner, against controls run on the same seeds.
Selection is on MMLU dev, and held-out is reported to confirm. Base
Llama-3.2-1B-Instruct scores 48.0 on dev and 44.6 on held-out.

**Knowledge** (Qwen2.5-7B-Instruct, block 4..20 at host layer 7,
answer-letter training on CommonsenseQA, OpenBookQA, QASC, and LogiQA2.0):

| cell | MMLU dev | MMLU held-out | in-dist. MC | law drift KL | survival |
|---|---|---|---|---|---|
| graft, seed 1 | 53.8 | 52.8 | 70.0 | 0.009 | 0.88 |
| graft, seed 2 | 52.6 | **54.4** | 69.0 | 0.011 | 0.87 |
| LoRA, seed 1 | 46.8 | 46.6 | 64.5 | 0.035 | |
| LoRA, seed 2 | 46.6 | 46.4 | 65.2 | 0.038 | |
| LoRA, 2000 steps | 44.0 | 46.0 | 65.2 | 0.048 | |
| identity, seed 1 | 44.9 | 46.2 | 57.5 | 0.007 | |
| identity, seed 2 | 47.1 | 46.2 | 59.0 | 0.007 | |
| span 2..20 | 51.8 | 52.6 | 70.0 | 0.009 | 0.88 |
| span 4..24 | 52.3 | 51.1 | 67.2 | 0.009 | 0.88 |
| + alignment regularizer | 53.7 | 52.9 | 69.0 | 0.009 | 0.89 |
| host layer 5 | 52.1 | 53.8 | 67.0 | 0.010 | 0.88 |

The gain holds up on new seeds. The graft scores 8 to 10 points above the
base on held-out MMLU and 6 to 8 points above a LoRA with the same
trainable budget. It also drifts about a quarter as much on law text. On
dev, LoRA and identity land below the untrained base, so answer-letter
training alone does not lift MMLU; the donor's layers do. The other
mutations sit within about a point of the winner.

**Medicine** (Apollo2-7B, block 4..20; healdata without the MedMCQA leak):

| cell | MedMCQA val (1000) | MMLU dev | MMLU held-out |
|---|---|---|---|
| graft, MC objective | **46.2** | 53.0 | 51.6 |
| graft, distill then MC | 44.3 | 52.6 | 50.9 |
| LoRA | 41.4 | 46.7 | 47.3 |
| identity | 43.0 | 47.5 | 46.1 |
| base | 41.4 | 48.0 | 44.6 |

**Caveat on size.** The grafted network holds 16 frozen donor layers
(about 3.5B parameters) beside the 1B host. The matched LoRA matches the
trainable budget, not the total size. h6 adds Llama-3.2-3B-Instruct as a
host, untrained, with LoRA, and grafted, to compare at a similar total size.

### h4: fine-tuning across domains, and where the graft reads and writes

Every cell now scores the same cross-domain panel:

- The four in-distribution MC evals (400 each): general (CommonsenseQA,
  OpenBookQA, QASC, LogiQA2.0), science (SciQ), math (AQuA-RAT), and
  medicine (HEAD-QA).
- MMLU by subject group. These items come from MMLU test after the dev and
  held-out items, so they overlap neither.
- MedMCQA val.

Training sets come from train splits only, and `healdata.check_sources`
passes them. One seed per cell. Base dev is 48.0.

**Train on one domain, score all** (Qwen2.5-7B block 4..20 at host 7,
budget-matched LoRA and identity):

| train on | arm | MMLU dev | held-out | MMLU-med | MMLU-math | HEAD-QA | AQuA | MedMCQA |
|---|---|---|---|---|---|---|---|---|
| general | graft | **52.2** | **53.1** | **61.6** | 37.6 | **54.5** | 29.8 | **45.8** |
| general | LoRA | 48.2 | 47.1 | 49.8 | 34.0 | 45.8 | 31.5 | 42.8 |
| science | graft | **52.3** | **50.0** | **57.8** | 32.4 | 51.0 | 27.5 | **44.0** |
| science | LoRA | 47.6 | 46.6 | 49.4 | 32.6 | 48.2 | 30.8 | 39.5 |
| medicine | graft | 53.3 | **53.9** | **59.8** | 38.4 | 52.8 | 28.5 | 43.4 |
| medicine | Apollo2 graft | **53.8** | 51.2 | 55.4 | 38.0 | **54.8** | 28.7 | **45.1** |
| medicine | LoRA | 51.3 | 48.8 | 51.6 | 34.6 | 51.7 | 29.0 | 39.7 |
| math | graft | 47.8 | 45.6 | 52.6 | 34.2 | 47.5 | 37.2 | 42.2 |
| math | Math-7B graft | 47.8 | 46.9 | 49.8 | **45.2** | 46.5 | **52.0** | 37.9 |
| math | LoRA | 42.6 | 43.5 | 47.0 | 34.6 | 42.8 | 34.5 | 36.5 |
| all four | graft | **53.6** | **53.9** | **62.4** | 36.8 | **55.8** | 35.8 | **44.9** |
| all four | LoRA | 51.0 | 48.6 | 52.2 | 32.6 | 51.0 | 41.5 | 41.5 |

- The general-knowledge graft beats LoRA on MMLU whatever it trains on,
  and the gain carries across domains. Trained only on commonsense and
  logic items, it lifts MMLU medical subjects by 12 points over LoRA and
  HEAD-QA by 9.
- Specialist donors win in their own domain. Trained on AQuA-RAT, the
  Math-7B graft scores 52.0 on AQuA (LoRA 34.5) and 45.2 on MMLU math
  (LoRA 34.6). It does not help general MMLU.
- Math answer-letter training lowers MMLU for every arm. With no
  rationale, AQuA letters are mostly noise for a 1B model.

**Projection layers** (general training; the encoder reads a learned
softmax mix of RMS-normed host layers, and the graft writes after its own
layer):

| read layers | write after | MMLU dev | held-out | MMLU-med | MMLU-math | MedMCQA | gate |
|---|---|---|---|---|---|---|---|
| 7 | 7 | 52.2 | 53.1 | 61.6 | 37.6 | 45.8 | 0.14 |
| 3 | 7 | 51.8 | 52.4 | 57.0 | 40.0 | 45.0 | 0.13 |
| 3, 5, 7 | 7 | 54.2 | 53.6 | 59.0 | 41.8 | 45.5 | 0.14 |
| 1..7 | 7 | 55.5 | **54.9** | 59.6 | 42.8 | **46.6** | 0.14 |
| 5 | 9 | 53.6 | 54.2 | 60.6 | 44.0 | 46.4 | 0.21 |
| 7 | 11 | 52.9 | 52.1 | 60.4 | 34.8 | 45.8 | 0.22 |
| 3, 7, 11 | 11 | **57.0** | 54.8 | 60.8 | **44.0** | 46.4 | 0.25 |

Reading several host layers beats reading one: +3 to +5 points on dev. The
best two configurations are re-tested on new seeds in h8.

### h6: a size control (Llama-3.2-3B-Instruct host)

The grafted 1B network carries about 3.5B frozen donor parameters, so a
fair question is whether a bigger plain model does better. One seed:

| cell | MMLU dev | held-out | MMLU-med | MedMCQA |
|---|---|---|---|---|
| 1B base | 48.0 | 44.6 | 52.6 | 41.4 |
| 3B base | 59.6 | 57.4 | 69.4 | 57.0 |
| 3B + LoRA | 62.0 | **60.9** | 68.8 | 55.6 |
| 3B + identity | 59.1 | 58.1 | 70.4 | 56.2 |
| 3B + Qwen7B graft at host 12 | 59.6 | 59.1 | 70.4 | 54.7 |
| 3B + Qwen7B graft at host 8 | **62.8** | 60.7 | 68.6 | 54.8 |

On the 3B host, the 7B graft ties LoRA. h9 tries multi-layer reads,
graft plus LoRA, and the 14B donor on this host.

### h7: a stronger donor (Qwen2.5-14B-Instruct, 48 layers)

| cell | MMLU dev | held-out | MMLU-med | MMLU-math | humanities | MedMCQA |
|---|---|---|---|---|---|---|
| block 7..34, seed 0 | 57.8 | **58.1** | 65.4 | 45.2 | 69.4 | 44.8 |
| block 7..34, seed 1 | 57.6 | 57.0 | 63.6 | 47.6 | 67.0 | 46.6 |
| block 7..34, all domains | **59.5** | 56.4 | 61.4 | 45.6 | 69.8 | **46.7** |
| block 10..40 | 56.7 | 54.8 | 62.0 | 46.2 | 67.6 | 44.6 |
| block 4..24 | 48.0 | 48.4 | 52.4 | 34.6 | 56.6 | 39.7 |
| LoRA (14B budget) | 45.3 | 45.1 | 48.4 | 31.6 | 54.8 | 40.0 |
| identity | 45.7 | 46.1 | 51.0 | 29.8 | 54.0 | 42.7 |

The 1B host with a 14B block scores 58 on MMLU held-out, 13 points above
the base and the matched LoRA. That is above the untrained 3B model
(57.4) but below 3B + LoRA (60.9). The block has to reach the donor's
deeper layers: 4..24 barely helps.

### h8: multi-layer reads on new seeds

| cell | MMLU dev | held-out |
|---|---|---|
| read 1..7, write 7, seed 1 | 53.6 | 53.8 |
| read 1..7, write 7, seed 2 | 55.8 | 56.6 |
| read 3,7,11, write 11, seed 1 | 55.1 | 56.2 |
| read 3,7,11, write 11, seed 2 | 48.2 | 45.4 (collapsed) |
| read 1..11, write 11 | 55.7 | **57.1** |
| read 0..15, write 15 | 56.1 | 55.5 |
| read 1,3,5,7,9, write 9 | 55.4 | 55.0 |
| read 3,7,11, all domains | 54.6 | 53.8 |
| LoRA, 2x budget, all domains | 45.4 | 47.9 |

Against the single-read graft (52.8 and 54.4 on these seeds), dense
reads add about 1 to 3 points. The sparse 3,7,11 read is unstable: one
seed fell back to the base. Dense reads (1..7, 1..11) did not collapse.

### h5: two donors in one network

Trained on all four domains, Llama-1B host, one seed:

| cell | MMLU dev | held-out | MMLU-med | MMLU-math | MedMCQA |
|---|---|---|---|---|---|
| Qwen7B (host 7) + Apollo2 (host 10) | **56.2** | **55.4** | **63.6** | 38.0 | 45.6 |
| Qwen7B (host 7) + Math-7B (host 10) | 54.5 | 54.5 | 62.2 | **41.2** | **46.8** |
| Qwen7B blocks 4..20 and 12..24 | 54.1 | 53.4 | 61.2 | 37.0 | 46.0 |
| Qwen7B alone (h4) | 53.6 | 53.9 | 62.4 | 36.8 | 44.9 |
| LoRA, 2x budget | 47.3 | 46.9 | 47.8 | 34.0 | 39.3 |
| LoRA, 3x budget | 43.5 | 43.0 | 45.2 | 30.6 | 35.1 |

A second, specialist donor adds 1 to 1.5 points on MMLU and moves its own
domain. A second block of the same donor adds nothing. More LoRA budget
makes the 1B host worse. The three-donor cell ran out of memory at MC
batch 8 and is re-run at batch 4 in h11.

### h9: the 3B host again

| cell (Llama-3.2-3B host) | MMLU dev | held-out | MMLU-med | science | MedMCQA |
|---|---|---|---|---|---|
| base | 59.6 | 57.4 | 69.4 | 48.8 | 57.0 |
| LoRA (7B budget), seeds 0 / 1 | 62.0 / 61.8 | 60.9 / 60.7 | 68.8 / 68.8 | 50.0 / 49.6 | 55.6 / 56.6 |
| LoRA (14B budget) | 61.7 | 61.0 | 67.6 | 49.4 | 54.9 |
| LoRA, 2x 7B budget | 59.6 | 59.9 | 66.2 | 51.0 | 54.2 |
| Qwen7B at host 8, seeds 0 / 1 | 62.8 / 61.9 | 60.7 / 62.9 | 68.6 / 70.8 | 49.2 / 50.2 | 54.8 / 55.9 |
| Qwen7B at host 8, + LoRA | 61.5 | 60.1 | 72.0 | 51.6 | 53.3 |
| Qwen7B, read 1..8, + LoRA | 63.9 | 61.0 | 68.4 | 50.0 | 56.0 |
| Qwen7B, read 2,5,8 | 62.2 | 62.0 | 68.8 | 51.4 | 53.1 |
| **Qwen14B 7..34 at host 8** | **64.5** | **63.4** | **74.4** | **58.0** | **57.7** |

On the 3B host, the 7B graft is about even with LoRA. The 14B graft
beats every LoRA by 2.4 points or more on held-out, 5 to 7 on MMLU
medical, and 8 on science. Adding a LoRA to the graft does not help, and
neither do multi-layer reads here. h11 runs a second seed of the 14B
cell.

### h10: the 14B donor with multi-layer reads (Llama-1B host)

| cell | MMLU dev | held-out | MMLU-med | MMLU-math | humanities |
|---|---|---|---|---|---|
| read 1..7, write 7, seed 0 | 61.1 | 58.9 | 64.2 | 49.6 | 69.2 |
| read 1..7, write 7, seed 1 | **61.4** | **61.0** | **66.6** | **53.0** | 69.0 |
| read 1..11, write 11, seed 0 | 60.9 | 59.8 | 62.4 | 50.4 | 70.0 |
| read 1..11, write 11, seed 1 | 60.1 | 56.5 | 59.6 | 46.8 | 65.4 |
| single read, host 5 | 61.7 | 57.9 | 63.8 | 47.0 | 70.8 |
| block 4..34 | 58.1 | 56.9 | 62.2 | 46.4 | 66.2 |
| block 7..40 | 57.1 | 57.5 | 63.4 | 44.0 | 69.6 |
| decoder rank 512 | 56.4 | 55.5 | 64.6 | 45.0 | 66.0 |
| single read at 7 (h7, 2 seeds) | 57.8 / 57.6 | 58.1 / 57.0 | | | |
| LoRA, 2x 14B budget, all domains | 43.7 | 43.5 | 45.2 | 27.8 | 52.2 |
| Qwen7B (read 1..7) + Math-7B (read 5,9,13 at 13), all domains | 58.1 | 56.0 | 62.4 | 44.2 | 67.0 |

Llama-3.2-1B with a 14B block that reads host layers 1..7 averages 60.0
on MMLU held-out over two seeds. That is 15 points above the base and the
matched LoRA, above the untrained 3B model (57.4), and level with 3B +
LoRA (60.9). The 14B + Apollo2 composition ran out of memory and is
re-run at batch 4 in h13, along with a third seed.

## Also fixed

`wsp/healdata.py`: the `reasoning` pool crashed on LogiQA2.0. In that
dataset, 38,398 of 63,718 rows are premise and hypothesis items with no
answer options, and 2 rows are not valid JSON. Those rows are now skipped.
`composer/agent.py`, `studio/server/specs.py`, `wsp/atlas.py`, and
`modal_round21.py` all use this pool.

## Summary so far

- **Works:** functional alignment across tokenizers. A frozen slice of a
  foreign 7B model keeps about 90% of its function inside a 1B model, the
  base reads it, and per-token gates learn when to use it. Nothing earlier
  in the repo got this far.
- **Does not work yet:** turning that into knowledge the base cannot get
  from fine-tuning. Under LM loss on dataset text, a LoRA at the same budget
  matches or beats every graft arm. Graft + LoRA ties LoRA alone, and more
  LoRA beats both. No graft arm here beats the base on MMLU.
- **Works:** with an answer-letter signal, the graft beats the base and a
  budget-matched LoRA. On MMLU held-out it scores 52.8 and 54.4 on new
  seeds (selected on dev), against 46.4 to 46.6 for LoRA and 44.6 for the
  base. With Apollo2 as the donor, MedMCQA goes from 41.4 to 46.2, and LoRA
  stays at 41.4.
- **Does not work:** library parts lifted out of their model. Fed a
  translated input, they do not align. Fed native inputs from the donor's
  own trunk, they are no better than random units.
