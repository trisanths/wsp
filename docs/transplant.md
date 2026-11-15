# Same-family expert transplant

Code: `wsp/transplant.py`. Round script: `modal_round18.py`. Tests: `tests/test_transplant.py`.

## What it builds

A small dense model gets a small bank of routed experts next to the MLP of each layer. The experts come from a big MoE of the same family, with the same hidden width and the same tokenizer. The result is one network of near base size. It runs alone. The source runs only while the model is built and as the teacher in the heal.

First pair: base `Qwen/Qwen3-1.7B` (hidden 2048, 28 layers) and source `Qwen/Qwen3-30B-A3B` (hidden 2048, 48 layers, 128 experts, top 8, expert width 768). The configs and the tensor names were checked against the Hub. One real fetch of experts 5 and 100 of source layer 3 through `fetch_parts` worked (18 s, about 20 MB).

Second pair, not tested: `LiquidAI/LFM2-1.2B` and `LiquidAI/LFM2-8B-A1B` (hidden 2048 both, 16 and 24 layers, 32 experts, top 4, two dense first layers). The pair is a parameter (`PAIRS`). `source_moe_layers` skips the dense layers and `ffn_slots` knows the LFM2 names (`feed_forward`, `ffn_norm`). It needs transformers 5. `wsp.parts.find_moe_block` raises for fused expert tensors, so the byte-range loader does not work for it yet.

## One layer

For base layer i the source layer is j, at the same relative depth (`depth_map`, snapped to a sparse layer). With h the residual state at the MLP input:

```
out = base_mlp(norm(h))                       unchanged
s   = stitch_in(h)                            base basis to source basis
x   = source_norm_j(s)                        the source's own pre-MLP norm
p   = softmax(router(x))                      over the K chosen experts
w   = top 2 of p, renormalized
y   = sum over the 2 of w_e * expert_e(x)     frozen source expert MLPs
out = out + gate * stitch_out(y)              gate is a scalar, starts at 0
```

A fresh model equals the base exactly because every gate is 0 (tested with `torch.equal`). The decoder layer's norm is wrapped in `NormTap`, which keeps h for the bank.

The router starts as the source router rows of the K chosen experts, with a zero bias. With the identity stitch and all experts chosen, the bank equals the source's own sparse MLP (tested to 1e-5), and with a subset it is the source softmax restricted to the subset.

## Seam

`fit_stitches` runs base and source on the same text. At every base layer it collects four tensors: the base residual at the MLP input, the source residual at the mapped layer, the base MLP output, and the source sparse-MLP output. It accumulates normal-equation sums in float64 and solves two ridge fits.

- `stitch_in`: base residual to source residual, with a bias.
- `stitch_out`: source sparse-MLP output to base MLP output, no bias.

The ridge strength is 1e-3 times the mean diagonal. With `rank` set, the solution is cut to a rank r product by SVD. The R squared of each fit is returned after the cut. Without a fit, both stitches start as the identity.

The fit uses what the whole source layer writes. The bank writes only 2 of K experts. So `stitch_out` is a rough starting scale, and the gate and the heal correct it.

## Selection

`select_experts` calls `wsp.atlas.expert_skill_map` once per skill (math, code, knowledge) against general text. Per mapped source layer it ranks all experts by log ratio of routing frequency. The skills take turns, each taking its best unused expert, until K are chosen. K=4 gives 2 math, 1 code, 1 knowledge per layer. K=8 gives 3, 3, 2. `control="random"` draws K experts of the same layer with a seeded RNG.

The selection texts must not come from the scoring sets. `modal_round18.py` uses 96 train texts per skill.

## Heal

`heal` trains stitches, routers, gates, and an optional LoRA r16 on the base. The LoRA goes on before the banks are built, so the source experts get none. The loss is `kd_weight * KL(teacher || student) + lm_weight * LM`. The donor is the source (same tokenizer). In round 18 the distillation arms use `lm_weight=0`: the donor reads the same pool tokens (teacher forcing), so a pool answer reaches the student only through the KL term. An optional preservation term (`anchor`, `anchor_weight`, `anchor_flags`) adds KL from a frozen base on the flagged (general) texts. Experts, source norms, and base weights stay frozen (tested).

## Parameter counts

Counted on the meta device from the real configs, with `count_params`. The base is 1.7206B. Active follows `wsp.skills_eval.model_params` (the tied embedding counts). Both columns include the LoRA r16 (17.4M).

| Build | Total | Total / base | Active | Active / base |
|---|---|---|---|---|
| K=4, stitch rank 256 | 2.326B | 1.35 | 2.061B | 1.20 |
| K=8, stitch rank 256 | 2.854B | 1.66 | 2.062B | 1.20 |
| K=4, full stitches | 2.502B | 1.45 | 2.237B | 1.30 |

The target of about 1.3x total and 1.15x active at K=4 is missed. The stitches are the reason. A full 2048 by 2048 stitch pair costs 8.4M per layer, as much as nearly two experts. Rank 256 brings the cost down to 2.1M per layer. Rank 128 would give about 1.33x and 1.18x, at some loss of fit. Only the fit R squared shows the loss, and that has not been measured on the real models.

## Round 18

`modal_round18.py` has the cells, the heal data, the scoring, and the go and stop rule. Run it with `scripts/modal_fanout.py`. The verdict comes from `wsp.transplant.go_stop` (`modal run modal_round18.py --verdict 'results/round18_shard*.json'`).

No benchmark data trains anything. Selection, stitch fit, router and gate training, and every heal read `wsp.healdata` pools only (math, code, medicine, general), checked with `healdata.check_sources`. The knowledge skill uses the medicine pool, which is narrower than MMLU. Ranges of each pool are disjoint: selection [0, 96), stitch fit [1000, 1064), heal [2000, 5000), held-out KL [5000, 5128). Rows record pool names, sources, and ranges. Benchmarks only score the finished model.

Heal target: the donor's behaviour. Loss is KL(donor || student) on pool text, plus 0.5 * KL(frozen base || student) on the general texts. No LM loss in the distillation arms. `lora` and `transplant_k4_lm` use an LM loss on pool text. `kd_lora` gets the same donor signal, texts, and steps, and a LoRA rank from `matched_lora_rank`, so its trainable count is close to that of `transplant_k4` (`lora` uses the same rank).

No-heal gate (`--stage gate`): `transplant_k4_noheal` and `transplant_k4_random_noheal` build the transplant with the ridge stitches, set every gate to a fixed value (`set_gates`; 0.05, 0.1, 0.25, headline 0.1, and 0 as the base reference), and report `measure_kl` from the donor logits to the student logits on held-out pool text. Lower is better. The real experts must show a lower donor KL than the random ones before money goes to heals.

Benchmark-free signals for every trained arm: `donor_kl` (held-out pool text, all four pools) and `base_kl_general`.

## What is not known yet

- No arm has run on a GPU. All numbers above are counts, not scores.
- The learning rates are guesses. Gates start at 0 and need a high learning rate to leave it.
- The heal teacher forward through the 30B model in transformers 4.57 loops over experts and is slow. The time of 1500 steps is not measured.
- A residual state of Qwen3 has a few huge outlier dimensions. Ridge on raw residual states may be dominated by them. The fit R squared in the rows shows whether that happens.
- Two seeds is a thin basis for the noise rule. The rule uses the seed spread and one binomial standard error, whichever is larger.
