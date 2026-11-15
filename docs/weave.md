# Weave: same-family composition with a search

Code: `wsp/weave.py`. Tests: `tests/test_weave.py`, `tests/test_no_benchmark_training.py`. Modal run: `modal_round21.py`.

## What a same-family component is

A specialist that was trained from the host, or from the host's own ancestor, has the same tensor shapes and the same residual basis as the host. The difference D = W_specialist - W_reference of one tensor can be added to the host with no translation layer. The reference is the member's true parent: the host for a member trained from the host, the common base for a member trained from the base. Differences are computed in float32 from bf16 tensors held on the CPU, so they are exact.

Tensors fall in groups: (layer, attention), (layer, MLP), embeddings, norms. A member whose positional base differs (rope_theta or rope scaling) contributes MLP groups only. Vision tower tensors are ignored. Embeddings are left out of the Modal cells, because vocabularies and special-token rows differ.

## The genome

For each member: one weight per (layer bucket, attention or MLP), two global weights (embeddings, norms), and one density. Buckets are equal slices of depth (default 8). The density keeps exactly the top fraction of each difference by magnitude (ties at the threshold are cut to the count, in flat index order) and scales the kept entries so the trimmed difference has the norm of the original; 1.0 keeps all. A 1 over density factor is not used: it is unbiased only for random dropping. For a top cut it overshoots (a Gaussian at density 0.5 would gain 1.9x in L2), and for a mostly zero bf16 difference it would double a difference that lost nothing. The density gene therefore changes sparsity, not gain. Densities snap to a 0.1 grid in the Modal runs so the rank cutoff (threshold and tie count) can be cached.

`apply(model, genome)` writes `W = W_host + sum_i w_i * trim(D_i)` layer by layer in float32 and casts once to the model dtype. `restore(model)` copies the host values back. A zero genome gives the host bit for bit. With the reference equal to the host, a genome of weight 1 on one member gives that member exactly (bf16 and the same shapes).

Why float32: a sum of small differences that each round away in bf16 steps survives in float32 (tested). The row `cast_survival_best` reports the fraction of nonzero mixed entries that still change the bf16 value.

## The search

A diagonal CMA-ES written with numpy, seeded, in the unit cube. Generation 0 holds the zero genome, one one-hot genome per member, the uniform average, and the first samples. Only the samples go to the CMA-ES update, because the seeds were not drawn from its distribution; when a seed beats every sample, the mean moves to it. Each generation re-scores its best candidate on a second split; the validated value is the minimum of the two objectives, and the returned genome has the highest validated value. The log keeps every genome and score.

Two objectives plug into `search` as a pair of score functions and objective callables.

### Probe objective (default, `--objective probe`)

Benchmark-free and forward-only. `ProbeObjective` in `wsp/weave.py`.

1. Each member maps to a skill pool in `wsp.healdata` (math and math500: math; mbpp and humaneval: code; medmcqa: medicine; legal: law). A reasoning member uses the "reasoning" pool when `healdata.SOURCES` has one, else math plus code. Tool, RAG, extraction, and any skill with no pool use "general", and the row marks the pool a weak proxy. A pool that is missing or fails to load falls back to "general".
2. Probe texts are three disjoint held-out slices of each pool: `healdata.pool(name, 288, seed=0, skip=2000)`, which is rows 2000 to 2287 of the seeded order. Split a (96 texts) drives the search, split b validates each generation, split c is never used to select and gives the numbers for the go rule. Maximum length 384 tokens. A member with two pools takes 48 texts of each per split. Heal scripts that read the same pool with seed 0 take rows below 2000. A pool too short for the slice falls back to its last rows, and the row records the mode.
3. Teacher cache. Before the search, every member and the host run the probe texts once, teacher-forced, and the model is freed. Per position the cache keeps the top-64 token ids (int32), their log-probs (float16), and the tail mass outside the top 64 (float16). The host runs every member pool and the general pool. Files live on the volume in `/vol/probe_cache/round21`, named by label, repo revision, and a hash of the texts, so a rerun with the same texts reuses them. A member's weights then live only as its difference on the CPU.
4. Candidate score, forward passes only: `transfer_i` is the gap closed between KL(member_i || host) and 0 on member i's pool (host 0, member 1), computed on the cached top-64 plus one tail bucket. `drift` is KL(host || candidate) on general text. The objective is the mean transfer over members minus 10 times max(0, drift minus 0.05 nats). The top-64 plus tail KL is a lower bound of the full KL and is close to it when the top 64 holds most of the mass. float16 rounding can put a member's own transfer a hair above 1. A member with no gap to the host on its pool (KL zero) has transfer None and leaves the mean.

Time plan. A candidate costs one apply plus one forward pass over (members + 1) sets of 96 texts. Population 24 and 40 generations is the start, shrunk (generations to 8, then population to 12) until the plan fits 85 percent of the budget. Expected minutes by family (budget 10 h, 20 h for the two big cells): qwen15 220 (24 x 40), qwen7b 506 (24 x 39), llama8b 507 (24 x 22), gemma27 1005 (24 x 31), qwen32b 990 (24 x 12), lfm2 232 (24 x 40), qwen3_8b 468 (24 x 40). The apply time (pageable copies) dominates; the forward time uses 250 TFLOPs effective, 300 tokens per text, and 0.05 s per batch, all guesses.

### Practice objective (`--objective practice`)

The old objective, kept for comparison. The mean over target skills of (score - host) / max(best member - host, 0.05), minus 20 times the amount each guard skill falls more than 1 point below the host, on generated answers to practice splits of the reported benchmarks. The floor is 0.05 so that one point of noise on a 150-item split does not count as a large gain when the best member is no better than the host; rows record `denom_clipped` per skill. This path selects on the benchmarks the round reports, which the owner forbids as a default. Its benchmark loads live in one function, `practice_items_for`, which `tests/test_no_benchmark_training.py` exempts and checks is reached only from `setup_practice_selection`.

Optional start: `layer_prior_genome` turns per-skill layer scores (from `wsp.atlas.layer_skill_map`) into a starting mean with more weight where a member's skill lives. Off by default in the Modal run (`--use-layer-map`).

## Reference recipes and controls

Uniform average (the host counts as a member, weight 1/(n+1)); task arithmetic at scales 0.25, 0.5, 1.0; corrected TIES (fp32, each member against its own reference, mixable groups only, top 50% trim, sign election, disjoint mean). Scales for the last two are picked on search split A with the same objective as the search. The recipes are scored with the probe objective on splits a, b, and c, and on the benchmarks once.

Random control: the best genome applied to differences shuffled inside each tensor. Norms and value sets are identical to the real differences; the knowledge is gone. A mix that helps only through added noise or norm shows up here. Under the probe objective there are 4 random controls with different seeds (1 to 4; seed 1 is also the one control scored on the benchmarks). The row gives `content_z` of the best mix's mean transfer on split c against them, and per member.

Gate: `gate_member` measures the relative L2 distance and cosine to the reference on a sample of matrices. The 2026-11-14 scout found that Qwen2.5 Coder and Math 1.5B and 7B models sit at cosine 0.2 to 0.5 from the Instruct host (continued pretraining, not small fine-tunes), and Math has another rope_theta. Those members fail the gate (default: relative L2 at most 0.3, cosine at least 0.97) and are not mixed.

## Go rule

Two parts. Both must hold; under `--objective practice` only the second is judged.

- Probe: the best mix's transfer on split c is above 0.3 on every member's pool, its drift is under the tolerance (0.05 nats), and content_z against the random controls is above 2.
- Benchmarks: the searched mix beats the host on every target skill by more than 2 points, the MMLU guard is within 1 point, its final-test objective beats uniform, task arithmetic, and TIES, and the random control stays within 2 points of the host on every target skill and scores below the searched mix.

The exploratory cell (lfm2) never gets a go.

## Output

`save` writes a plain Hugging Face directory with exactly the host's tensors, loadable with `AutoModelForCausalLM` and no wsp code, plus `weave.json` (genome, members, references, search summary).

## The limit

Same lineage only. A member that does not share the host's basis needs a translation layer; those paths are in `wsp/synth.py`. At 0.5B and 1.5B no ungated task specialist of the Qwen2.5 host passes the gate, so the smallest working cells are LFM2-1.2B (skills with no scorer here, exploratory) and the 7B to 8B families. Mixing cannot add a skill that no member has. The best case is the members' per-layer best parts together, which evolutionary model merging has shown can exceed each source.

## Benchmarks and the control

Benchmarks are scored exactly once, at the end, for the host, each member, the best mix, the fixed recipes, and one random control. Under the probe objective nothing in the search reads a benchmark item, so `practice_contaminated` is false and `selects_on_benchmarks` is false. Under the practice objective the items come from train splits that the members were trained on or derived data from, so member scores on practice are inflated against the clean final test; rows record `member_practice_minus_final` per skill and `practice_contaminated`.

The control check passes when the shuffled-difference model is within 2 points of the host on every target skill and scores below the searched mix. Rows also record the Hub revision of every repo, the seeds, and the chat template hash. The run writes its row before saving the model, and appends every candidate to `/vol/results/round21/<family>.progress.jsonl`.

## Unverified

- No GPU run has happened. Every Modal estimate is a model, not a measurement. The probe path ran on CPU only on tiny random models (`tests/test_weave.py`) and in a scratch smoke run of its helper functions; `run_family` itself has not run.
- The healdata "reasoning" pool is commonsense and logic multiple choice. It is a weak proxy for the long chain of thought that OpenThinker, QwQ, and the R1 distills learned.
- The "weak proxy" members (tool, RAG, extract) get general text, where the member and the host may barely differ, so their transfer is noisy or None.
- Tokenizers are assumed shared across a family (the host's tokenizer reads every member's probe text).
- A cache key uses the repo revision when the Hub call works; a failed call gives an "unknown" key that can go stale.
- GPU memory of the float32 logits (batch 4, 384 tokens, 262k vocabulary for Gemma, about 1.6 GB per copy, a few copies live) was not measured.
- The apply-time model counts (2 * members + 1) model copies over pageable memory at 6 GB/s. The first kthvalue on each large tensor adds time that is not modelled.
- The step-time model (0.022 s plus 0.0015 s per billion parameters at batch 32) is a guess.
- Reasoning members (OpenThinker, R1 distills, QwQ) think for thousands of tokens; the 512 and 1024 token caps in `wsp.skills_eval` may truncate them, so their member scores can sit below the host.
- Gemma 3 loading through `AutoModelForCausalLM` (Gemma3ForConditionalGeneration), the parameter-name match between the unsloth text checkpoint and the host, and the MATH train source (`DigitalLearningGmbH/MATH-lighteval`) have not been run.
- Whether Med42 (Llama 3.0 parent) mixes into attention. It is forced to MLP only because its rope scaling differs.
- The 32B cell needs about 370 GiB of CPU RAM. The memory ceiling for one Modal container was not checked.
