# Rotation test for far-lineage specialists

Code: `wsp/rotate.py`. Tests: `tests/test_rotate.py`. GPU script: `modal_round26.py`.
Plan: Experiment 4 in `docs/research/2026-11-15-transfer-sweep.md`.

## Functions

- `fold_norms(model)`: moves each RMSNorm weight into the linear layers that read the normed state (q, k, v for the input norm, gate and up for the post-attention norm, lm_head for the final norm). Every norm weight becomes 1. A tied head is untied first. Outputs are unchanged (test tolerance 1e-5).
- `procrustes(A, B)`: the orthogonal R that minimizes `||A R - B||`, by SVD in float64.
- `rotate_residual(model, R)`: rotates the writers (embedding rows, `o_proj`, `down_proj`) and the readers (q, k, v, gate, up, head) so the function is unchanged (test tolerance 1e-4). It refuses a model whose norms are not folded.
- `residual_report(E_m, E_i, R)`: relative residual `||E_m R - E_i|| / ||E_i||` for the given R, for R = identity, for a Procrustes R fit after shuffling the token rows, and for a random orthogonal R.

Row convention: a state is a row vector, a rotated state is `x R`.

## CPU check (2026-11-15)

Token embeddings read by byte range (`wsp/fetch.py`, three tensors per model, no full download). Base is Qwen2.5-1.5B-Instruct. Vocabulary is 151936 rows in all three. The fit uses the 151643 real tokens. Using all 151936 rows changes nothing past the third decimal.

| Specialist | Rotated | Unrotated | Token-shuffled R | Random R |
|---|---|---|---|---|
| Qwen2.5-Math-1.5B-Instruct | 0.918 | 1.078 | 1.042 | 1.170 |
| Qwen2.5-Coder-1.5B-Instruct | 0.728 | 0.920 | 1.083 | 1.197 |

Plan thresholds: go below 0.3, stop above 0.8.

- Math is above the stop threshold. Its best rotation removes about 15 percent of the unrotated residual. The embeddings are close to independent of the base's.
- Coder is between the thresholds and far from 0.3. The rotated residual beats the shuffled one (0.728 against 1.083), so some token structure is shared, but it is not a rotation of the base's table.

The call from the CPU check: stop for Math, no go for Coder. Under the plan's stop rule, no-training cross-lineage mixes stop for Math. Coder needs a decision from the owner; the residual says the cells are unlikely to leave chance.

## Second view: down_proj output spaces

Output neurons of an MLP are not aligned between models, so a direct Procrustes fit on `down_proj` is meaningless. The test used here is the output Gram matrix `G = W W^T` (d x d), which does not depend on neuron order. With R from the embedding fit, the relative difference `||R^T G_m R - G_i|| / ||G_i||` was:

| Model | Layer | Rotated by embedding R | Unrotated | Random R |
|---|---|---|---|---|
| Math | 14 | 1.242 | 1.065 | 1.339 |
| Math | 7 | 0.992 | 0.908 | 1.044 |
| Coder | 14 | 0.685 | 0.561 | 0.831 |
| Coder | 7 | 0.615 | 0.565 | 0.699 |

The embedding R does not help the layer writers. It is worse than no rotation in all four rows. This agrees with the embedding result: the specialists do not sit in a rotated copy of the base's coordinate system.

## GPU cells

`modal_round26.py` has the cells from the plan (rotated MLPs and whole layers for Math and Coder, with fit, identity, random, and shuffled R, plus references). It was not run. The CPU check says stop, so the docstring says not to spend GPU time beyond `--smoke`.

## Not verified

- The Modal script was only syntax checked. Its model build (fold, rotate, copy MLPs or layers) has no test on real models.
- `layer_*` cells use the specialist's own rope_theta for the whole model, which is a simplification of the 16c per-layer rope.
- One shuffle seed and one random seed.
