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

## Families (round 21 cells)

Each cell is one host and its true fine-tunes. `modal_round21.py` holds the table (`FAMILIES`); `tests/test_weave.py` checks that every entry has its keys and that every member skill has a scorer, has a probe pool, or is marked unscored or exploratory.

### Selection rule

A member enters only when `python -m wsp.lineage relation HOST MEMBER` (Hub range reads, nothing downloaded) says fine_tune or fine_tune_of_ancestor, with mean cosine at least 0.97, the same rope_theta as the host, and mean relative L2 at least 0.01 (a smaller difference carries little) and at most 0.3 (the gate). A family needs two such members of different skills. A member whose card names a base other than the host is probed against that base too; it uses the base as its reference only when it is clearly closer to the base than to the host. The columns below are the probe's mean cosine and mean relative L2 over its sampled tensors (embeddings and norms included), then the mean relative L2 over the sampled attention and MLP matrices of layers. A member with a "ref" is judged against its reference.

### Cells from the 2026-11-14 scout

| Cell | Host | Members | Evidence |
| --- | --- | --- | --- |
| qwen15 | Qwen2.5-1.5B-Instruct | Coder-1.5B, Math-1.5B | cosine about 0.5 and 0.2: expected to fail the gate |
| qwen7b | Qwen2.5-7B-Instruct | OpenThinker-7B, OpenCodeReasoning-Nemotron-7B | cosine 0.9996 and 0.988 (relative L2 not recorded) |
| llama8b | Llama-3.1-8B-Instruct | OpenMath2, R1-Distill-Llama-8B, HuatuoGPT-o1-8B, Med42 | OpenMath2 cosine 0.973; the rest in the scout note |
| gemma27 | Gemma-3-27B-it | MedGemma-27B-text, MeditronFO | MedGemma cosine 0.997, attention fails the gate |
| gemma27_mlp | Gemma-3-27B-it | MedGemma-27B-text forced to MLP only, MeditronFO | MedGemma's MLP difference passes the gate; attention tensors are not read |
| qwen32b | Qwen2.5-32B-Instruct | QwQ-32B, Baichuan-M2-32B, OCR-Nemotron-32B, OpenThinker-32B | cosine 0.9996, 0.998, 0.996, 0.9998 |
| lfm2 | LFM2-1.2B | Tool, RAG, Extract | cosine 0.99 and above, relative L2 about 0.036 to 0.040 (exploratory) |
| qwen3_8b | Qwen3-8B | II-Medical-8B, LegalEvolver-Qwen3-8B | see the scout note |

`gemma27_mlp` needs per-member `mlp_only`. `wsp/weave.py` already has it (`Member.mlp_only`, `Member.covers`). `modal_round21.py` set it only when the config's positional base differed from the host. A member spec with `"mlp_only": True` now forces it and reads the member without attention tensors, so `gate_member` measures the MLP matrices alone.

### Cells from the 2026-11-15 small-model scout

Size is in billions of parameters. Expected minutes come from `make_plan` (probe objective, budget 10 hours; every cell fits). "Scorer" lists the skills the cell reports.

| Cell | Host | Size | Scorers | Members | Expected minutes |
| --- | --- | --- | --- | --- | --- |
| gemma2_2b | gemma-2-2b-it | 2.6 | gsm8k, mbpp, mmlu, humaneval | math, coding | 268 |
| llama3b | Llama-3.2-3B-Instruct | 3.2 | gsm8k, mmlu | math, hermes3 (ref: Llama-3.2-3B) | 274 |
| llama1b | Llama-3.2-1B-Instruct | 1.2 | gsm8k, mbpp, mmlu, humaneval | math, coder | 207 |
| qwen3_4b | Qwen3-4B | 4.0 | medmcqa, mmlu, gsm8k (transfer) | pythagoras, popit_medical | 309 |
| qwen3_1p7b | Qwen3-1.7B | 1.7 | medmcqa, mmlu | medpsy, hito | 198 |
| qwen25_3b | Qwen2.5-3B-Instruct | 3.1 | gsm8k, medmcqa, mmlu | smallthinker, medmath | 271 |
| smollm3 | SmolLM3-3B | 3.1 | gsm8k, medmcqa, mmlu | medical, mopd_math (ref: SmolLM3-3B-Base) | 271 |
| gemma3_4b | gemma-3-4b-it | 4.3 | medmcqa, mmlu | biomed, cypher, mamaylm | 396 |
| gemma3_1b | gemma-3-1b-it | 1.0 | exploratory | borealis, swahili | 181 |
| qwen25_15b | Qwen2.5-1.5B-Instruct | 1.5 | exploratory | vikhr, tooltuned, correction | 249 |
| qwen25_05b | Qwen2.5-0.5B-Instruct | 0.5 | exploratory | hammer, vikhr | 160 |
| lfm2_350m | LFM2-350M | 0.35 | exploratory | extract, luth, hala | 182 |
| lfm2_700m | LFM2-700M | 0.74 | exploratory | luth, tashkeel, hala | 205 |
| lfm2_2b | LFM2-2.6B | 2.6 | exploratory | transcript, longevity | 248 |

Evidence per member (relation from the probe, mean cosine, mean relative L2, then attention and MLP layer matrices):

| Cell | Member | Skill | Relation | Cosine | Rel L2 | Attn | MLP |
| --- | --- | --- | --- | --- | --- | --- | --- |
| gemma2_2b | MergeBench/gemma-2-2b-it_math | gsm8k | fine_tune | 0.9998 | 0.0203 | 0.0150 | 0.0301 |
| gemma2_2b | MergeBench/gemma-2-2b-it_coding | mbpp | fine_tune | 0.9999 | 0.0128 | 0.0113 | 0.0174 |
| llama3b | MergeBench/Llama-3.2-3B-Instruct_math | gsm8k | fine_tune | 0.9998 | 0.0187 | 0.0125 | 0.0165 |
| llama3b | NousResearch/Hermes-3-Llama-3.2-3B (vs base) | chat | fine_tune | 0.9996 | 0.0269 | 0.0222 | 0.0330 |
| llama1b | AryanK123/Llama-3.2-1B-Instruct_SFT_Math-220kv00.04 | gsm8k | fine_tune | 0.9998 | 0.0175 | 0.0140 | 0.0217 |
| llama1b | MDDDDR/Llama-3.2-1B-Instruct-FFT-coder-python | mbpp | fine_tune | 0.9956 | 0.0850 | 0.0520 | 0.0883 |
| qwen3_4b | Pythagoras-LM/Pythagoras-Prover-4B | math | fine_tune | 0.9923 | 0.1139 | 0.1477 | 0.1180 |
| qwen3_4b | Hoshino-Yumetsuki/PopiT-Qwen3-4B-Medical-SFT-1128 | medmcqa | fine_tune | 0.9992 | 0.0362 | 0.0498 | 0.0348 |
| qwen3_1p7b | qvac/MedPsy-1.7B | medmcqa | fine_tune | 0.9944 | 0.1042 | 0.1083 | 0.1119 |
| qwen3_1p7b | hitonet/hito-1.7b | reasoning | fine_tune | 0.9968 | 0.0459 | 0.0196 | 0.0192 |
| qwen25_3b | Tiiny/SmallThinker-3B-Preview | gsm8k | fine_tune | 0.9964 | 0.0843 | 0.0891 | 0.0824 |
| qwen25_3b | calebking/qwen2.5-3b-instruct-medical-math-full | medmcqa | fine_tune | 0.9998 | 0.0168 | 0.0141 | 0.0217 |
| smollm3 | vinayp27/smollm3-medical | medmcqa | fine_tune | 0.9998 | 0.0199 | 0.0162 | 0.0284 |
| smollm3 | BytedTsinghua-SIA/Open-MOPD-SmolLM3-3B-RL-Math (vs base) | gsm8k | fine_tune_of_ancestor | 0.9978 | 0.0594 | 0.0474 | 0.0899 |
| gemma3_4b | AdaptLLM/biomed-gemma-3-4b-it | medmcqa | fine_tune | 0.9994 | 0.0324 | 0.0230 | 0.0447 |
| gemma3_4b | neo4j/text-to-cypher-Gemma-3-4B-Instruct-2025.04.0 | cypher | fine_tune | 0.9992 | 0.0351 | 0.0338 | 0.0480 |
| gemma3_4b | INSAIT-Institute/MamayLM-Gemma-3-4B-IT-v1.0 | ukrainian | fine_tune | 0.9729 | 0.2451 | 0.2181 | 0.2785 |
| gemma3_1b | NbAiLab/borealis-1b | norwegian | fine_tune | 0.9976 | 0.0640 | 0.0416 | 0.0914 |
| gemma3_1b | CraneAILabs/swahili-gemma-1b | swahili | fine_tune | 0.9916 | 0.1077 | 0.0530 | 0.1781 |
| qwen25_15b | Vikhrmodels/Vikhr-Qwen-2.5-1.5B-Instruct | russian | fine_tune | 0.9852 | 0.1592 | 0.2163 | 0.1551 |
| qwen25_15b | asheeee24/tooltuned-qwen2.5-1.5b | tool | fine_tune | 0.9999 | 0.0111 | 0.0154 | 0.0105 |
| qwen25_15b | shibing624/chinese-text-correction-1.5b | correction | fine_tune | 0.9999 | 0.0138 | 0.0175 | 0.0146 |
| qwen25_05b | MadeAgents/Hammer2.0-0.5b | tool | fine_tune | 0.9999 | 0.0136 | 0.0211 | 0.0107 |
| qwen25_05b | Vikhrmodels/Vikhr-Qwen-2.5-0.5b-Instruct | russian | fine_tune | 0.9789 | 0.1892 | 0.2600 | 0.1814 |
| lfm2_350m | LiquidAI/LFM2-350M-Extract | extract | fine_tune | 0.9851 | 0.1724 | 0.1783 | 0.1609 |
| lfm2_350m | kurakurai/Luth-LFM2-350M | french | fine_tune | 0.9996 | 0.0285 | 0.0295 | 0.0276 |
| lfm2_350m | hammh0a/Hala-350M | arabic | fine_tune | 0.9897 | 0.1475 | 0.1589 | 0.1422 |
| lfm2_700m | kurakurai/Luth-LFM2-700M | french | fine_tune | 0.9992 | 0.0411 | 0.0434 | 0.0394 |
| lfm2_700m | Etherll/Tashkeel-700M | arabic_diacritics | fine_tune | 0.9995 | 0.0310 | 0.0370 | 0.0313 |
| lfm2_700m | hammh0a/Hala-700M | arabic | fine_tune | 0.9903 | 0.1411 | 0.1520 | 0.1358 |
| lfm2_2b | LiquidAI/LFM2-2.6B-Transcript | transcript | fine_tune | 0.9989 | 0.0465 | 0.0504 | 0.0458 |
| lfm2_2b | LiquidAI/LFM2-2.6B-Longevity | longevity | fine_tune | 0.9796 | 0.2070 | 0.2248 | 0.2013 |

Passing, not entered: MergeBench gemma-2-2b-it multilingual (0.9998, 0.0165) and instruction (0.9999, 0.0135); Vikhr-Llama-3.2-1B-Instruct (0.9995, 0.0260); LFM2-1.2B Longevity (0.9882, 0.1528) and Luth-1.2B (0.9995, 0.0308), which could join `lfm2`; the same Llama-3.2-3B and gemma-2-2b fine-tunes trained from the base (they need the base as reference). Their skills have no scorer or pool, so under the probe go rule they would be weak proxies.

Held out:

- Relative L2 below 0.01 (RL or LoRA-sized changes): MergeBench Llama-3.2-3B-Instruct coding, multilingual, instruction, safety; Jan-nano, Polaris-4B, Qwen3-4B-SafeRL; GooseReason-4B; Jan-v3-4B; Arch-Router-1.5B; medical_o1_verifier_3B; suayptalha DeepSeek-R1-Distill-Llama-3B; Qwen3-0.6B and 1.7B distills; granite-3.3-2b ethics (0.0098) and medicanite.
- Not a fine-tune of the host: Qwen/Qwen3-4B-Instruct-2507 and Qwen3-4B (diverged, rope_theta 5e6 against 1e6); OpenThinker3-1.5B (cosine 0.87); LFM2-350M-Math (cosine 0.16); LFM2.5-1.2B-Thinking (cosine 0.91); sarvam-translate (0.84).
- Parent is the base and the difference from the base is below 0.01: qwen3-1.7b-math-grpo (0.0001 from Qwen3-1.7B-Base), smollm3-3b-sft-math-merged (0.0047 from SmolLM3-3B-Base). Mixed against the host they would subtract the host's own post-training.
- Cosine below 0.97 or relative L2 above 0.3: TinySwallow-1.5B (0.9693), medgemma-4b-it (0.960, 0.336), LEM-Gemma3-1B (relative L2 0.301), MedLlama.nl, LFM2.5-JP-202606.
- One member only: Qwen3-0.6B, Qwen3-4B-Instruct-2507, Phi-4-mini, OLMo-2-1B, LFM2.5, granite-3.3-2b.

The cards of the MergeBench models name no base. Their names and the weights (cosine 0.9998, relative L2 0.01 to 0.02 from the host) do.

PopiT-Qwen3-4B-Medical's card names Qwen3-4B-Instruct-2507 as the base, but its weights are 0.036 from Qwen3-4B and 0.20 from Qwen3-4B-Instruct-2507, so it is a fine-tune of Qwen3-4B.

### Launch

    modal run --detach modal_round21.py --family gemma2_2b,llama1b --objective probe

Run one cheap cell first (LESSONS.md): `--family qwen25_05b --plan-only`, then the same without `--plan-only`. `--family all` now runs 22 cells. Name the cells.

## The limit

Same lineage only. A member that does not share the host's basis needs a translation layer; those paths are in `wsp/synth.py`. At 0.5B and 1.5B no task specialist of the Qwen2.5 Coder or Math line passes the gate. Small cells exist in other families: gemma2_2b, llama1b, llama3b, qwen3_1p7b, and the LFM2 cells (see Families). Mixing cannot add a skill that no member has. The best case is the members' per-layer best parts together, which evolutionary model merging has shown can exceed each source.

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
