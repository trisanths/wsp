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

## Round 23: three training signals (running)

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

**Results: pending.** This section will be filled in when the run finishes.

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
- **Being tested:** whether a training signal that needs the donor's
  knowledge (answer letters, distillation, or both) changes that.
