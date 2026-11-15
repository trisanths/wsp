# Weave: same-family composition with a search

Code: `wsp/weave.py`. Tests: `tests/test_weave.py`. Modal run: `modal_round21.py`.

## What a same-family component is

A specialist that was trained from the host, or from the host's own ancestor, has the same tensor shapes and the same residual basis as the host. The difference D = W_specialist - W_reference of one tensor can be added to the host with no translation layer. The reference is the member's true parent: the host for a member trained from the host, the common base for a member trained from the base. Differences are computed in float32 from bf16 tensors held on the CPU, so they are exact.

Tensors fall in groups: (layer, attention), (layer, MLP), embeddings, norms. A member whose positional base differs (rope_theta or rope scaling) contributes MLP groups only. Vision tower tensors are ignored. Embeddings are left out of the Modal cells, because vocabularies and special-token rows differ.

## The genome

For each member: one weight per (layer bucket, attention or MLP), two global weights (embeddings, norms), and one density. Buckets are equal slices of depth (default 8). The density keeps exactly the top fraction of each difference by magnitude (ties at the threshold are cut to the count, in flat index order) and scales the kept entries so the trimmed difference has the norm of the original; 1.0 keeps all. A 1 over density factor is not used: it is unbiased only for random dropping. For a top cut it overshoots (a Gaussian at density 0.5 would gain 1.9x in L2), and for a mostly zero bf16 difference it would double a difference that lost nothing. The density gene therefore changes sparsity, not gain. Densities snap to a 0.1 grid in the Modal runs so the rank cutoff (threshold and tie count) can be cached.

`apply(model, genome)` writes `W = W_host + sum_i w_i * trim(D_i)` layer by layer in float32 and casts once to the model dtype. `restore(model)` copies the host values back. A zero genome gives the host bit for bit. With the reference equal to the host, a genome of weight 1 on one member gives that member exactly (bf16 and the same shapes).

Why float32: a sum of small differences that each round away in bf16 steps survives in float32 (tested). The row `cast_survival_best` reports the fraction of nonzero mixed entries that still change the bf16 value.

## The search

A diagonal CMA-ES written with numpy, seeded, in the unit cube. The objective is the mean over target skills of (score - host) / max(best member - host, 0.05), minus 20 times the amount each guard skill falls more than 1 point below the host. Generation 0 holds the zero genome, one one-hot genome per member, the uniform average, and the first samples. Only the samples go to the CMA-ES update, because the seeds were not drawn from its distribution; when a seed beats every sample, the mean moves to it. The floor is 0.05 so that one point of noise on a 150-item split does not count as a large gain when the best member is no better than the host; rows record `denom_clipped` per skill. Each generation re-scores its best candidate on a second practice split; the validated value is the minimum of the two objectives, and the returned genome has the highest validated value. The log keeps every genome and score.

Optional start: `layer_prior_genome` turns per-skill layer scores (from `wsp.atlas.layer_skill_map`) into a starting mean with more weight where a member's skill lives. Off by default in the Modal run (`--use-layer-map`).

## Reference recipes and controls

Uniform average (the host counts as a member, weight 1/(n+1)); task arithmetic at scales 0.25, 0.5, 1.0; corrected TIES (fp32, each member against its own reference, mixable groups only, top 50% trim, sign election, disjoint mean). Scales for the last two are picked on practice split A.

Random control: the best genome applied to differences shuffled inside each tensor. Norms and value sets are identical to the real differences; the knowledge is gone. A mix that helps only through added noise or norm shows up here.

Gate: `gate_member` measures the relative L2 distance and cosine to the reference on a sample of matrices. The 2026-11-14 scout found that Qwen2.5 Coder and Math 1.5B and 7B models sit at cosine 0.2 to 0.5 from the Instruct host (continued pretraining, not small fine-tunes), and Math has another rope_theta. Those members fail the gate (default: relative L2 at most 0.3, cosine at least 0.97) and are not mixed.

## Output

`save` writes a plain Hugging Face directory with exactly the host's tensors, loadable with `AutoModelForCausalLM` and no wsp code, plus `weave.json` (genome, members, references, search summary).

## The limit

Same lineage only. A member that does not share the host's basis needs a translation layer; those paths are in `wsp/synth.py`. At 0.5B and 1.5B no ungated task specialist of the Qwen2.5 host passes the gate, so the smallest working cells are LFM2-1.2B (skills with no scorer here, exploratory) and the 7B to 8B families. Mixing cannot add a skill that no member has. The best case is the members' per-layer best parts together, which evolutionary model merging has shown can exceed each source.

## Practice data and the control

The practice items come from train splits that the members were trained on or derived data from, so member scores on practice are inflated against the clean final test. Rows record `member_practice_minus_final` per skill and `practice_contaminated`. The final test stays clean, so the go rule is unaffected; search efficiency is.

The control check passes when the shuffled-difference model is within 2 points of the host on every target skill and scores below the searched mix. Rows also record the Hub revision of every repo, the practice, search, and control seeds, and the chat template hash. The run writes its row before saving the model, and appends every candidate to `/vol/results/round21/<family>.progress.jsonl`.

## Unverified

- No GPU run has happened. Every Modal estimate is a model, not a measurement.
- The apply-time model counts (2 * members + 1) model copies over pageable memory at 6 GB/s. The first kthvalue on each large tensor adds time that is not modelled.
- The step-time model (0.022 s plus 0.0015 s per billion parameters at batch 32) is a guess.
- Reasoning members (OpenThinker, R1 distills, QwQ) think for thousands of tokens; the 512 and 1024 token caps in `wsp.skills_eval` may truncate them, so their member scores can sit below the host.
- Gemma 3 loading through `AutoModelForCausalLM` (Gemma3ForConditionalGeneration), the parameter-name match between the unsloth text checkpoint and the host, and the MATH train source (`DigitalLearningGmbH/MATH-lighteval`) have not been run.
- Whether Med42 (Llama 3.0 parent) mixes into attention. It is forced to MLP only because its rope scaling differs.
- The 32B cell needs about 370 GiB of CPU RAM. The memory ceiling for one Modal container was not checked.
