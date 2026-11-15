# Layer sets: the minimal set of layers per skill

Code: `wsp/layerset.py`. Tests: `tests/test_layerset.py`. Modal run: `modal_round25.py`. Plan: Experiment 1 in `docs/research/2026-11-15-transfer-sweep.md`.

## What it does

A true fine-tune (a `weave.Member`) differs from its host in every layer. The skill may need only some of them. `layerset` finds the smallest set S of layers such that writing the member's difference on S alone keeps most of the member's probe transfer. Two members with disjoint sets can then be written into one host-sized network.

Everything is forward-only. A candidate is written into the host with `Weaver.apply`, scored against the teacher caches of a `weave.ProbeObjective`, and removed with `Weaver.restore`. No generation and no benchmark item is read. The Weaver has one bucket per layer (`layer_weaver`), so genome weight `[member, layer, kind]` is the weight of that layer.

Members for this module hold attention and MLP tensors only (`groups=("attn", "mlp")`). Norms and embeddings stay with the host, so "one layer" means that layer's attention and MLP.

## Functions

- `layer_importance(scorer, member)` writes the difference on one layer at a time: both parts, attention only, and MLP only. It records transfer toward the member, harm, and drift, plus the per-layer difference norm. The ranking sorts layers by the "both" transfer (ties: larger norm, then lower index).
- `minimal_set(scorer, member, ranking, target_share=0.9)` bisects the prefix length of the ranking for the smallest prefix whose share of the member's full transfer is at least the target and whose harm is within the limit. Then a greedy check walks the chosen layers from the lowest rank up and drops each one that the set can lose and still pass. The result holds `size_share`, the share on the selection split, `stop_rule` (size over 75 percent of the layers), and `within_half`.
- `compose(scorer, sets, resolve)` writes members on their own sets. A layer in two sets goes to the member with the larger difference norm (`"norm"`), or `"try_both"` starts from the norm choice, tries each other owner for every shared layer, and keeps the assignment with the higher probe value.
- Controls: `random_set_control` (random sets of equal size, one per seed), `norm_set_control` (the layers with the largest difference norm, no probe), `cross_skill_control` (member A's layers on member B's set, scored toward A's pool), `scrambled_control` (the difference shuffled inside each tensor, on the same layers).
- `LayerScorer` holds one weaver, one model, and one objective (one probe split). It memoizes by assignment, so a repeated set costs no forward pass. `restrict_objective` makes a one-member objective from a joint one without a new calibration. `host_matches` checks that the host is restored bit for bit.

## Definitions

- transfer: gap closed between KL(member || host) and 0 on the member's probe pool. The host scores 0 and the member itself scores 1.
- harm: next-token loss on general text of the written model minus the host's.
- share: transfer of a set divided by the transfer of the full member (every layer).
- harm limit: the larger of the objective's tolerance (0.05 nats) and the harm of the full member. A true fine-tune can sit above the tolerance by itself, and no subset should face a stricter bar than the member. The result records the limit used.

## Splits

Round 21's three probe splits. Split a selects (ranking, bisection, greedy check, composition choice). Split b checks. Split c is never used to select and gives every reported number. Teacher caches are round 21's files, same key (label, repo revision, text hash), so a cell that follows a round 21 `qwen7b` run builds none.

## Modal run: `modal_round25.py`

    modal run --detach modal_round25.py --family qwen7b --smoke
    modal run --detach modal_round25.py --family qwen7b --seeds 5
    modal run --detach modal_round25.py --family qwen7b --only bench,summary
    modal run modal_round25.py --family qwen7b --plan-only

Flags: `--smoke` (host and the code member only), `--only` (comma list of stage groups: caches, gates, importance, minimal, compose, controls, bench, summary), `--seeds` (number of random layer sets per member, seeds 1 to N, default 5), `--redo` (repeat a selected stage whose row exists), `--plan-only`.

Rows go to `/vol/results/round25/v1/<cell>/<stage>.json` and `rows.jsonl` the moment a stage ends. A rerun skips a stage whose row exists. A benchmark arm is its own stage, so a crash during generation loses one arm. The `v1` is `FIT_VERSION`; bump it when the fit code changes. Every printed line starts with the cell name in brackets.

Benchmarks (once per arm): MBPP 257, HumanEval 164, GSM8K 300, MMLU 1000. Arms: host, each member alone, each member on its minimal set, the random set of equal size (seed 1), the composed model, and `Qwen/Qwen2.5-Coder-7B-Instruct`. Every arm uses the host's tokenizer, the host's chat template, batch 64, and 2048 new tokens for generative skills, because OpenThinker and OpenCodeReasoning-Nemotron write long chains. Two raw generations per benchmark and arm go to the log with their token count, and each skill line says how many outputs reached the cap.

## Go and stop

Go for a member: size at most 50 percent of the layers, share on split c at least 0.9, the random sets of equal size keep under 50 percent on average, and the benchmark of the member's skill is within 5 points of the member (MBPP for the code member, GSM8K for OpenThinker). Composition: each skill at 95 percent of its member's score. Stop: no set under 75 percent of the layers holds 90 percent of the transfer.

## GPU-hour estimate

From `make_plan`, a model of forward and generation cost (round 21's step-time and FLOP guesses), not a measurement:

| Cell | Expected | If the model is 1.5x low |
|---|---|---|
| qwen7b smoke | 54 min (0.9 H200 hours) | 1.4 h |
| qwen7b full, 5 seeds | 181 min (3.0 H200 hours) | 4.5 h |

Smoke parts: load 14 min, importance 13 (85 sets), minimal set 7, controls 2, MBPP on four arms 18. It assumes round 21's `qwen7b` caches exist; building them adds about 10 min. The full run's benchmark part is 113 min and dominates.

## Unverified

- No GPU or Modal run has happened. The CPU tests use tiny random Qwen2 models in float32 and bfloat16, a planted skill (difference only on known layers, or a token-specific rank-one write), and a run of `run_cell` on tiny checkpoints with the Hub, GPU, and benchmark scoring replaced by stand-ins.
- The time model is a guess. Per-set forward cost (3 passes of 96 texts), apply time, and the share of batches that run to the 2048 cap are assumptions.
- Single-layer transfer is not additive. The ranking can order layers wrongly when two layers only work together; the bisection checks combined sets, and the greedy check removes layers, but nothing adds a layer that the ranking placed too low. If a set comes out large, a prefix curve on the ranking would show it.
- Whether the harm limit (the full member's harm, when above 0.05) is the right bar was a choice here. Strict tolerance mode needs `harm_limit=0.05` in `minimal_set`.
- OpenThinker answers with a long thought block before the solution. The grader of `wsp.skills_eval` takes the last number or the code block it finds; a 2048 token cap may still cut some chains. The log shows how many reached the cap.
- Qwen2.5-Coder-7B-Instruct scores under the host's chat template, as every arm does. Its own default template is not used.
- Tokenizers and the vocabulary head are the host's for every arm. Members' embeddings and output heads are not written.
- Composition on two members assumes their sets are mostly disjoint. The overlap rules were tested on planted overlaps only.
