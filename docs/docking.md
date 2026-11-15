# Docking ports: a toy test of Net2Net copies as slots for a donor's skill

Code: `modal_docking.py`. Tests: `tests/test_docking.py`. Rows: `results/docking_rows.json` (156 rows, 26 arms, 2 pairs, 3 seeds). Smoke rows: `results/docking_smoke_rows.json`.

## The idea

Net2WiderNet (Chen, Goodfellow, Shlens 2015) grows a layer by copying a neuron k times and dividing its outgoing weights by k. The function stays exactly the same. A little noise on the copies breaks their symmetry, so they can drift apart in training. Sparse Upcycling (Komatsuzaki et al. 2022) and Branch-Train-MiX (Sukhbaatar et al. 2024) also grow a trained dense model into a wider or mixture form, and then train the new parts.

Two models trained apart store features in different coordinates. Copies of model A's neurons give model B's features no matching slot. Merging work fixes this by alignment first: Git Re-Basin (Ainsworth et al. 2022) permutes B's neurons to match A's, ZipIt! (Stoica et al. 2023) merges features that both models share, and model stitching (Lenc and Vedaldi 2015; Bansal et al. 2021) trains an adapter between two models.

Docking ports: widen A with Net2Net copies, freeze the original block, and train only the copies toward B's features (hidden activations and outputs). The copies start inside A's basin and are pulled toward B. The question is whether this carries B's skill into A as one network better than Re-Basin merging or a trained stitch.

## Setup

- A: MLP 784-512-512-20 on MNIST (labels 0 to 9). B: the same shape on Fashion-MNIST (labels 10 to 19). The head has 20 outputs, so one merged network answers both tasks with no task flag. (The spec said 784-512-512-10. A 10-way head cannot answer two tasks without a flag.)
- Pair "diff": A and B start from different random inits. Pair "same": both start from one init.
- A and B train 1500 steps each on their own task. Every trained arm then gets STEPS = 500 steps, batch 256, Adam 1e-3, on the mix of both TRAIN splits. Scores use the test splits only. Nothing is chosen on test.
- Only the stitch arm uses a task flag (a router, counted as part of that network). Every other arm takes the raw image.
- Docking: widen both hidden layers by k, noise sigma times the weight std on every entry that touches a copy, original-to-original block frozen (gradient masked, checked after training). The loss is KL to a teacher (A's logits on MNIST rows, B's logits on Fashion rows) plus a hidden loss. "lin": a learned linear map from the copies' activations to B's activations. "reb": B is permuted to A first, and a copy of unit i is pulled to B's unit i directly.

## Arms

| Arm | What it tests |
|---|---|
| a_alone, b_alone | references (the other task scores 0, since labels live in different halves of the head) |
| joint | scratch model, same 500 steps on the mix |
| ft_full | A, full fine-tune on the mix, 500 steps (a bar stronger than LoRA, same params as A) |
| avg, rebasin | weight average, and weight matching then average, no training |
| avg_heal, rebasin_heal | the same, then 500 steps on the mix |
| concat | A and B side by side, outputs summed with learned scalars, 500 steps |
| stitch | A bottom, ridge-initialized then trained linear stitch into B's top layer and head for Fashion, A for MNIST, router by task |
| dock_lin_kK_sS, dock_reb_kK_sS | docking ports, K in {2,4}, sigma S in {0.01,0.1} |
| scram_kK_sS | docking toward a scrambled B (entries of each hidden matrix shuffled, head re-fit on Fashion train) |
| plain_kK_sS | same widening, output distillation only, no hidden target |

## Results

Means and standard deviations over 3 seeds, test accuracy in percent. "Mean" is the average of the two task scores. Params is the network that runs at inference. Trainable is what the arm updated in its 500 steps.

| Arm | diff MNIST | diff Fashion | diff mean | same mean | params | trainable |
|---|---|---|---|---|---|---|
| a_alone | 97.73 ± 0.08 | 0.00 | 48.86 | 48.93 | 0.67M | 0 |
| b_alone | 0.00 | 87.78 ± 0.18 | 43.89 | 43.87 | 0.67M | 0 |
| joint | 95.43 ± 0.11 | 84.73 ± 0.60 | 90.08 ± 0.35 | 90.16 ± 0.38 | 0.67M | 0.67M |
| ft_full | 97.80 ± 0.23 | 85.03 ± 0.37 | 91.42 ± 0.09 | 91.23 ± 0.05 | 0.67M | 0.67M |
| avg (no training) | 2.26 ± 1.35 | 52.91 ± 4.44 | 27.58 ± 2.73 | 57.77 ± 5.66 | 0.67M | 0 |
| rebasin (no training) | 3.95 ± 3.40 | 65.05 ± 4.64 | 34.50 ± 2.07 | 53.57 ± 3.99 | 0.67M | 0 |
| avg_heal | 97.56 ± 0.15 | 86.59 ± 0.53 | 92.07 ± 0.22 | 92.06 ± 0.14 | 0.67M | 0.67M |
| rebasin_heal | 97.49 ± 0.16 | 86.80 ± 0.68 | 92.15 ± 0.28 | 92.22 ± 0.34 | 0.67M | 0.67M |
| concat | 97.82 ± 0.10 | 87.41 ± 0.61 | 92.61 ± 0.26 | 92.44 ± 0.35 | 1.87M | 1.87M |
| stitch (router) | 97.73 ± 0.08 | 84.51 ± 0.13 | 91.12 ± 0.09 | 91.22 ± 0.14 | 1.21M | 0.26M |
| dock_lin k2 s0.01 | 97.78 ± 0.08 | 86.40 ± 0.23 | 92.09 ± 0.14 | 92.04 ± 0.06 | 1.87M | 1.20M |
| dock_lin k2 s0.1 | 97.82 ± 0.08 | 86.27 ± 0.06 | 92.05 ± 0.04 | 92.09 ± 0.12 | 1.87M | 1.20M |
| dock_lin k4 s0.01 | 97.80 ± 0.07 | 86.69 ± 0.04 | 92.24 ± 0.05 | 92.18 ± 0.18 | 5.85M | 5.17M |
| dock_lin k4 s0.1 | 97.77 ± 0.06 | 86.81 ± 0.10 | 92.29 ± 0.02 | 92.19 ± 0.11 | 5.85M | 5.17M |
| dock_reb k2 s0.01 | 97.72 ± 0.10 | 86.50 ± 0.23 | 92.11 ± 0.15 | 92.00 ± 0.05 | 1.87M | 1.20M |
| dock_reb k2 s0.1 | 97.75 ± 0.07 | 86.45 ± 0.32 | 92.10 ± 0.19 | 92.07 ± 0.04 | 1.87M | 1.20M |
| dock_reb k4 s0.01 | 97.66 ± 0.00 | 85.84 ± 0.53 | 91.75 ± 0.27 | 92.11 ± 0.12 | 5.85M | 5.17M |
| dock_reb k4 s0.1 | 97.64 ± 0.04 | 85.62 ± 0.49 | 91.63 ± 0.24 | 92.15 ± 0.14 | 5.85M | 5.17M |
| plain k2 s0.01 | 97.70 ± 0.07 | 85.28 ± 0.47 | 91.49 ± 0.23 | 91.67 ± 0.03 | 1.87M | 1.20M |
| plain k2 s0.1 | 97.69 ± 0.12 | 85.43 ± 0.35 | 91.56 ± 0.14 | 91.73 ± 0.08 | 1.87M | 1.20M |
| plain k4 s0.01 | 97.69 ± 0.06 | 85.58 ± 0.75 | 91.64 ± 0.36 | 91.98 ± 0.08 | 5.85M | 5.17M |
| plain k4 s0.1 | 97.67 ± 0.10 | 86.08 ± 0.28 | 91.88 ± 0.10 | 92.03 ± 0.05 | 5.85M | 5.17M |
| scram k2 s0.01 | 97.67 ± 0.04 | 72.10 ± 0.56 | 84.88 ± 0.29 | 84.81 ± 0.27 | 1.87M | 1.20M |
| scram k2 s0.1 | 97.63 ± 0.08 | 72.08 ± 0.60 | 84.86 ± 0.31 | 84.82 ± 0.11 | 1.87M | 1.20M |
| scram k4 s0.01 | 97.75 ± 0.06 | 72.77 ± 0.71 | 85.26 ± 0.37 | 85.15 ± 0.34 | 5.85M | 5.17M |
| scram k4 s0.1 | 97.77 ± 0.02 | 72.55 ± 0.79 | 85.16 ± 0.40 | 85.20 ± 0.19 | 5.85M | 5.17M |

Steps: 500 for every trained arm except a_alone and b_alone (1500 pretraining) and the two untrained merges (0). Full per-seed rows are in the JSON file.

## What the table says

- All healed merges sit inside a 1.2 point band (91.1 to 92.6 mean). The tasks are small and the head start is large, so this toy is near its ceiling. Read differences under about 0.5 point as noise (seed spread is 0.1 to 0.4).
- Docking against Re-Basin. Training-free, Re-Basin beats averaging for different inits only a little (34.5 against 27.6 mean) and both are broken. With 500 healing steps, plain average, Re-Basin, and docking tie (92.1, 92.15, 92.05 to 92.3 mean). Docking does not beat Re-Basin plus heal here, and it costs 2.8x (k2) or 8.7x (k4) the inference params for the same score. Docking needed no alignment: dock_lin on the "diff" pair scores the same as on the "same" pair.
- Docking against the stitch. Docking (92.05 to 92.29) beats the stitch (91.1) by about 1 point, mostly on Fashion (86.3 to 86.8 against 84.5). The stitch trains only 0.26M parameters and needs a router. Docking trains 1.2M or more and needs no router.
- The feature target helps a little. dock_lin beats plain output distillation into the same widened network by 0.4 to 0.6 mean on the "diff" pair at k2 (92.05 and 92.09 against 91.49 and 91.56) and by 0.4 at k4. On the "same" pair the gap is 0.15 to 0.4. This is the one clean docking-specific effect, and it is small. It is about three seed spreads at k2.
- The scrambled control fails by 7 to 14 Fashion points (72 against 86), so the result depends on the donor's content. This control also changes the output teacher (a head re-fit on random features), so it shows that a bad donor hurts, not that features alone carry the gain.
- Alignment before docking ("reb") is no better than a learned linear map ("lin"), and it is worse at k4 on the "diff" pair (91.7 against 92.3). With k4 each unit has three copies pulled to one target, which pushes them toward identical features.
- Noise sigma (0.01 against 0.1) makes no difference at 500 steps.
- Concat is the best arm (92.4 to 92.6), at 2.8x params. It keeps both models' features intact and learns only how to read them.
- Fashion for every merge stays under B alone (87.8). ft_full of A on the mix reaches only 85.0, so B's skill did move in, but no arm matches B.
- Limits. One dataset pair, one architecture, 500 healing steps, 3 seeds. Fashion and MNIST are easy enough that a short fine-tune closes most of the gap. The result does not show docking failing at scale. It shows that on this toy it adds no measurable value over a healed Re-Basin merge.

## What it implies for Sapien

(a) Same-lineage components (our weave). Weight averaging of true fine-tunes already starts at 58 mean here, and a short heal reaches the same score as every fancier route. Docking adds params and wins nothing over weave plus heal. Keep weave. The small feature-target effect over plain distillation suggests that, if a widened host is ever needed, a hidden-activation loss is worth adding to the heal, but it does not justify widening.

(b) Cross-lineage components (our seams and skill tops). Docking works without any alignment, so it is a candidate where permutation alignment is hard, which is the case for different-architecture donors. In this toy it beats a trained stitch by about 1 point and ties Re-Basin plus heal. The cost is a wider network. Our skill tops keep the host's size, and the toy shows no accuracy reason to give that up. Next test: scale the same comparison to a transformer MLP block pair where Re-Basin is known to fail (cosine 0.2 to 0.5), since this toy cannot separate "alignment is unnecessary" from "the heal is strong enough to hide misalignment".
