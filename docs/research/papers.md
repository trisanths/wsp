# Research notes for part 3: RL environment, swarm, and baselines

Checked 2026-11-14. I opened each source below and confirmed the claim I report. Items I read only at abstract level say "(abstract)". Items I could not confirm say "unverified". Sections 1 to 7 cover host-mask programs; Section 8 covers the mixed-family graft space, which is now the main action space.

## Read this first

1. **The fit score ranks depths, not parts.** The ridge fit maps host layer k to donor layer l, so every cluster from one donor layer, random or chosen, gets the same R squared at the same `at`. Use it to place grafts and veto bad pairings (8.2).
2. **Fit quality may not predict behavior.** Across 20 heterogeneous LLM pairs, ridge-map quality had near-zero correlation (r = -0.07) with behavioral effect (abstract). Calibrate every pre-heal signal against our own full heals (8.2).
3. **Random parts through trained glue can work well.** Random 1% subnetworks match full RLVR, and a frozen random transformer with trained interfaces gets most of a pretrained one's accuracy. Run the random-part control as a full search arm (8.4).
4. **Two budgets bind.** Compile plus heal takes minutes, so GPU-seconds will bind before eval count. Plot both axes, and give LoRA and FFT the same GPU-seconds, tuning included (8.3).
5. **Eval noise is large.** A 200-item GSM8K search split has about 3.5 points of standard error; a 90-question MMLU-Pro subject split has about 4. Compare programs with paired standard errors on fixed items.
6. **For host masks:** BBEH at 1B sits at the floor (base 2.17%); the public masks were trained on the paper's 75% split, so carve search and held-out from the other 25% (split seed unverified); mergekit-evolve cannot express mask algebra and is deprecated.

---

## 1. Deletion Tuning (the component we compose)

**Lu, Yang, Lasby, Pote, Ioannou, Ren (2026). The Surprising Effectiveness of Deleting Weights in LLM Post-Training.** https://arxiv.org/abs/2611.10866 . Code: https://github.com/agentic-learning-ai-lab/deleting-weights . Masks: https://huggingface.co/Jacklu0831/deletion-tuning-masks

- Method: one logit G per gate_proj weight. Training uses a soft mask W * sigmoid(alpha(t) G); evaluation uses the hard mask W * 1[G > 0]. No straight-through estimator, no fixed sparsity. G starts from Wanda scores (top weights near +0.03, bottom eta fraction near -0.03, eta = 1/2048), which zeroes about 0.18% of gate weights at init. Regularizer -lambda ||W~||^2 with lambda = 1e-5 for SFT.
- SFT recipe: lr 5e-4 (confirmed in the lr table), 400 steps, effective batch 4, 75/25 train/eval split per task. VERIFY.md records the SFT alpha ramp as 1e2 to 1e3; I did not recheck that number.
- Results relevant to us (Table 5, best-checkpoint accuracy, greedy decoding):

| Suite | Size | Base | FFT | Deletion Tuning |
|---|---|---|---|---|
| MMLU-Pro (14 subjects) | 1B | 10.37 | 16.97 | 18.62 |
| BBEH (23 tasks) | 1B | 2.17 | 8.35 | 5.65 |
| MMLU-Pro | 3B | 12.63 | 21.19 | 22.28 |
| BBEH | 3B | 8.87 | 11.39 | 11.39 |

- Test split sizes: MMLU-Pro 3,014 test questions (96 history to 338 math per subject). BBEH 50 per task (30 for disambiguation_qa).
- GRPO masks delete under 0.05% of all parameters (Llama1B 0.0347%). Masks are about 1 MB at 1B. Under GRPO, prior-task accuracy stays near base while FFT, LoRA, and random sparse tuning lose up to 5.4 points.
- Composition: listed only as future work ("Masks for different tasks might also be combined, for example by taking their union"). Matches VERIFY.md item 5.
- Takeaway: at 1B, FFT beats masks on BBEH (8.35 vs 5.65). Expect the "versus full fine-tuning" comparison to favor FFT on BBEH and masks on MMLU-Pro.

Base model facts (https://huggingface.co/unsloth/Llama-3.2-1B-Instruct, config.json): 16 layers, hidden 2048, intermediate 8192. Each gate_proj is 8192 x 2048 (16.8M weights), so all gate_proj weights total about 268M.

---

## 2. Evolutionary model merging (the primary baseline)

**Akiba, Shing, Tang, Sun, Ha (2024). Evolutionary Optimization of Model Merging Recipes.** https://arxiv.org/abs/2403.13187
- Parameter-space merging: DARE-TIES parameters (density, weight) per layer, optimized by CMA-ES in Optuna with default hyperparameters. Init all genes at 0.5, sigma 1/6, population 4 + floor(3 ln n_params). Fitness is accuracy on 1,069 search questions disjoint from the 250-question test set. 1,000 trials; final model is the best trial by search accuracy.
- Data-flow-space merging: CMA-ES in EvoJAX, 100 generations of population 128.
- Takeaway: the canonical recipe spends about 1,000 evals. At our 100 to 200 budget we get 7 to 14 CMA-ES generations, so keep the genome small.

**mergekit-evolve (Arcee).** https://github.com/arcee-ai/mergekit/blob/main/docs/evolve.md . Script: https://github.com/arcee-ai/mergekit/blob/main/mergekit/scripts/evolve.py
- Calls `cma.fmin2` (pycma) with sigma0 default 1/6 and `--max-fevals` default 100. Population is pycma's default unless `--force-population-size` is set. The docs warn the cma package can overshoot max-fevals by up to 50% depending on generation size. `layer_granularity` groups layers into blocks that share genes. Scoring uses lm-eval-harness tasks.
- The script emits "mergekit-evolve is deprecated and is likely to be removed".
- Takeaway: copy its defaults (x0 from genome init, sigma0 1/6, pycma default population, layer blocks) into our own CMA-ES loop. Cap evals in our counter, since pycma will not stop exactly at the cap.

**Mencattini et al. (2025). MERGE^3: Efficient Evolutionary Merging on Consumer-grade GPUs.** https://arxiv.org/abs/2502.10436
- Evaluates fitness on a reduced subset and corrects it with Item Response Theory estimators. 175 trials total. GSM8K-JA accuracy falls with subset size: 0.42 at 100 questions, 0.38 at 50 or 30, 0.34 at 20, against 0.49 for the 1,000-trial, 1,000-question original.
- Takeaway: cheap fitness costs final quality. Use small subsets only to screen, then confirm on the full search split.

**Abrantes, Lange, Tang (2025). Competition and Attraction Improve Model Fusion (M2N2).** https://arxiv.org/abs/2508.16204 . Code: https://github.com/SakanaAI/naturalniches
- Evolves merge boundaries, keeps diversity through fitness sharing over data points, and picks mates by complementarity. LLM experiment: every method gets 1,000 evals; CMA-ES uses population 25 and scores low because its fixed parameter partition was poor. They drop random mutation for LLMs because it "doesn't work well on large models".
- Takeaway: pick a second parent that is strong where the first is weak (attraction score over per-question results). A planner can do this with per-task scores in the prompt.

**Agarwal, Bonagiri, Astudillo, Choudhury (2026). Multi-Objective Bayesian Optimization for Model Merging (MOBO-Merge).** https://arxiv.org/abs/2608.14264
- 100 sequential BO steps against 100 random configurations, 10 seeds (42 to 51). Each benchmark is split 30% optimization, 70% held-out; only front members are re-scored on held-out. Curves show mean plus or minus one standard error across seeds. Random search stays competitive in 1-D linear merges; BO wins for TIES and block-wise spaces. About 4 to 9 minutes per candidate on one RTX 6000 Ada.
- Takeaway: this is the closest published protocol to ours (100 evals, random-search control, seed bands, held-out re-score). Cite it as the methodology precedent.

**Su, Geiping (2025). Fine, I'll Merge It Myself: A Multi-Fidelity Framework for Automated Model Merging.** https://arxiv.org/abs/2502.04030
- SMAC with Hyperband brackets and a random-forest surrogate. Fidelity is the number of evaluation examples. Finds useful merges in under 500 steps.
- Takeaway: "number of questions" works as a fidelity axis for merge search. It is the same axis our judge will use.

**Han et al. (2025). Reinforced Model Merging (RMM).** https://arxiv.org/abs/2503.21272
- Casts layer-wise merging as an MDP. State is a merging-map matrix (models and operators by layer); actions pick a model, a merge operator, or skip/back a layer; reward is the task metric of the finished merge. Their Dynamic Average Reward scores on data subsets early and up to 100x faster.
- Takeaway: a published precedent for a merge "environment" with step/reward. Our env is simpler: one action is a whole program, so each step is one episode-level eval.

**Ying et al. (2019). NAS-Bench-101.** https://arxiv.org/abs/1902.09635
- Regularized evolution (tuned to population 100, tournament 10; mutation flips one edge or swaps one op; oldest member dies) beat non-regularized evolution and REINFORCE, and all beat random search. Benchmarking guidelines: many runs, plot best-so-far against number of evaluations, never touch test data during search, show the spread across runs.
- Takeaway: in discrete spaces a tournament GA with single-edit mutations is the strong classical baseline. Population 100 is too big for 150 evals; scale it down.

**Hamano et al. (2022). CMA-ES with Margin.** https://arxiv.org/abs/2205.13482 . Optuna: https://optuna.readthedocs.io/en/stable/reference/samplers/generated/optuna.samplers.CmaEsSampler.html
- Plain CMA-ES on binary genes tends to fix them early. The margin keeps each binary variable's marginal probability above alpha. Optuna exposes it as `CmaEsSampler(with_margin=True)`. That sampler does not support categorical parameters, so encode inclusion bits as integers 0/1.
- Takeaway: use CMA-ES with margin for a binary "which mask in which layer block" genome.

Other merge-search work (abstract only): Mergenetic library (https://arxiv.org/abs/2505.11427); Model Kinship, where model similarity predicts merge gains (https://arxiv.org/abs/2410.12613); SimMerge, which predicts merge quality from cheap similarity signals under limited eval budgets (https://arxiv.org/abs/2601.09473).

---

## 3. LLM-guided search with evaluation feedback

**Romera-Paredes et al. (2023). Mathematical discoveries from program search with large language models (FunSearch).** https://www.nature.com/articles/s41586-023-06924-6 . Code: https://github.com/google-deepmind/funsearch
- Best-shot prompting: sample k = 2 programs from one island, sort by score ascending, label them v0, v1, and ask for v2. Islands evolve separately. Every 4 hours the weaker half of islands is wiped and reseeded with the best program of a surviving island. Within an island, programs are clustered by signature (the tuple of per-input scores) and sampled cluster first. Repo defaults: 10 islands, 2 functions per prompt, 4 samples per prompt.
- Takeaway: show the planner two or three parents, sorted worst to best, and cluster history by per-task score signature so it sees distinct strategies.

**Novikov et al. (2025). AlphaEvolve.** https://arxiv.org/abs/2506.13131
- Prompt holds several prior programs from a database (MAP-Elites plus islands), their scores, and rendered evaluation output, plus optional stochastic formatting and evolved meta-prompts. Output is a diff. Gemini Flash gives volume; Pro gives occasional big jumps. An evaluation cascade scores new programs on cheap stages first and drops weak ones early. Optimizing several metrics helped even when one metric mattered.
- Takeaway: build a cascade into step(): a cheap sanity stage, then a subset, then the full search split.

**Yang et al. (2023). Large Language Models as Optimizers (OPRO).** https://arxiv.org/abs/2309.03409
- Meta-prompt: task description, a few task examples, and past solution-score pairs sorted ascending, keeping only the highest-scoring ones (exact cap unverified). Instruction: write something "different from the old ones" with "a score as high as possible". Ablations: showing scores helps; 8 solutions per step beat 1, 2, 4, and 16 at a fixed total; temperature trades exploration against exploitation.
- Takeaway: ascending order with the best last, numeric scores shown, and about 8 proposals per round.

**Guo et al. (2023). EvoPrompt.** https://arxiv.org/abs/2309.08532
- The LLM performs GA crossover/mutation or differential-evolution updates on prompts. Studied populations 4 to 12; DE helped more on harder tasks because it mutates the parts where parents differ. Dev set of 200. Both variants beat APE at equal iterations.
- Takeaway: a "differential" edit (take what differs between two good programs and apply it to a third) is a cheap planner move.

**Lehman et al. (2022). Evolution through Large Models (ELM).** https://arxiv.org/abs/2206.08896
- LLM diff models act as mutation operators inside MAP-Elites. Random GP mutation success drops exponentially with the number of bugs to fix; the diff model fixes several coupled bugs in one step. Mutation type is chosen by sampling commit messages with fixed odds (40/30/30).
- Takeaway: give planners a menu of edit types with fixed sampling odds, so proposals stay diverse.

**Chen, Dohan, So (2023). EvoPrompting.** https://arxiv.org/abs/2302.14838
- LM as mutation and crossover for neural architecture search. Naive few-shot prompting does poorly; the evolutionary loop with fitness-based parent choice does well. On NATS-Bench it is competitive with regularized evolution and BOHB.
- Takeaway: LLM proposals need selection pressure from scored parents. One-shot "design a merge" prompting is the weak ablation.

**Lange, Imajuku, Cetin (2025). ShinkaEvolve.** https://arxiv.org/abs/2509.19349
- Parent sampling weight = sigmoid of fitness above the median times 1/(1 + number of children). Islands with migration. Rejects proposals whose embedding similarity to an archived program exceeds 0.95. Bandit over an LLM ensemble. Beat the AlphaEvolve circle-packing result in under 150 evaluations. Ablation: hill climbing starts strong then plateaus; weighted sampling keeps improving.
- Takeaway: at about 150 evals, novelty rejection and offspring-penalized parent sampling are the two cheapest gains.

**Jiang et al. (2025). AIDE: AI-Driven Exploration in the Space of Code.** https://arxiv.org/abs/2502.13138
- Tree search with three operators: draft, debug, improve. "Improve" makes exactly one atomic change so its effect is measurable. A summary operator compresses history into metrics and ideas instead of full logs.
- Takeaway: ask for single-edit children of a parent. Score changes become attributable, and lessons become writable.

**Ma et al. (2023). Eureka.** https://arxiv.org/abs/2310.12931
- 5 independent runs, 5 iterations each, K = 16 samples per iteration. "Reward reflection" feeds back per-component statistics, which beats returning one scalar.
- Takeaway: return per-task scores, deletion counts per layer, and collapse flags to the planner, not only the combined reward.

**Liu et al. (2024). Large Language Models to Enhance Bayesian Optimization (LLAMBO).** https://arxiv.org/abs/2402.03921
- Zero-shot warm starts from the LLM beat random, Sobol, and Latin hypercube initial points, mostly in the first 5 trials. Setup: 5 initial points, 25 trials, 10 seeds.
- Takeaway: the swarm's main edge at small budgets is a good first round. Spend analyst effort there.

**Li (2026). Greedy Is a Strong Default: Agents as Iterative Optimizers.** https://arxiv.org/abs/2603.27415
- LLM proposer with greedy acceptance matched Optuna TPE on LoRA SST-2 with 2x fewer evals and tied CMA-ES on XGBoost with 3x fewer. Parallel directive-driven investigators and simulated annealing added no gain at 2 to 3x the evals. CMA-ES found configurations outside the LLM's prior. Single author, 3 runs per setting: weak evidence.
- Takeaway: it supports the "same score, N times fewer evals" framing. It also warns that K parallel planners may not beat one greedy planner; run that ablation.

**Gallego (2026). A Hybrid Nested Harness for Decoupling Structure and Parameters in LLM-Driven Optimization.** https://arxiv.org/abs/2608.08156 (abstract)
- The LLM proposes a structure with numeric gaps; CMA-ES tunes the numbers. Beats either alone.
- Takeaway: if task-vector weights enter programs, let the planner choose structure and a small inner loop tune weights.

---

## 4. Mask and sparse merging: priors for planners

**Ilharco et al. (2023). Editing Models with Task Arithmetic.** https://arxiv.org/abs/2212.04089 (abstract). Task vectors (fine-tuned minus base) add and negate. Takeaway: `tv(id, weight)` follows this; negation is the analog of mask difference.

**Wortsman et al. (2022). Model soups.** https://arxiv.org/abs/2203.05482
- Greedy soup: sort ingredients by validation accuracy, add each only if held-out validation accuracy does not drop. Never worse than the best single model on validation.
- Takeaway: "greedy union" (add masks in order of single-mask score, keep if the search score holds) costs one eval per candidate mask. It is a strong, cheap baseline and a sensible first planner move.

**Yadav et al. (2023). TIES-Merging.** https://arxiv.org/abs/2306.01708
- Interference comes from redundant small updates and from sign disagreement. Sign conflicts grow with the number of merged models. Using oracle signs lifts the merge close to multitask (72.0 vs 66.4).
- Takeaway: for deletion masks, "both masks delete this weight" is agreement and "one deletes, the other needs it" is the conflict. Interference should grow with the number of masks in a union.

**Yu et al. (2024). DARE (Language Models are Super Mario).** https://arxiv.org/abs/2311.03099 (abstract). SFT deltas survive dropping 90 to 99% of entries with 1/(1-p) rescaling. Takeaway: fine-tuning signal is very redundant, consistent with masks touching under 0.05% of weights.

**Wang et al. (2024). Localizing Task Information for Improved Model Merging and Compression (TALL masks, Consensus Merging).** https://arxiv.org/abs/2405.07813 . Code: https://github.com/nik-dim/tall_masks
- Tasks mostly use non-overlapping weights. "Selfish" weights matter to one task only; "catastrophic" weights matter to none. Consensus Merging keeps weights that at least two task masks select. They report accuracy normalized to each single-task model.
- Takeaway: the mask analog of consensus is a k-of-n vote, which our algebra can express, e.g. majority of three is (a & b) | (a & c) | (b & c). Report "fraction of single-mask gain recovered" as the normalized score.

**He et al. (2024). Localize-and-Stitch.** https://arxiv.org/abs/2408.13656
- Localizing 1% of parameters recovers 99% of fine-tuned performance. Merges with more parameter overlap show more task conflict. Dataless variant keeps the top 5% by magnitude. Overlaps are averaged. In RoBERTa, the earliest layers seldom appear in localized regions.
- Takeaway: give planners pairwise Jaccard overlap of deletion sets and per-layer counts. Small, low-overlap unions are the safest first guesses.

**Huang et al. (2024). EMR-Merging.** https://arxiv.org/abs/2405.17461 (abstract). One elected model plus per-task binary masks and rescalers. Takeaway: per-task masks over one shared model are a known high-accuracy pattern; a union is the single-model compromise.

**Ansell, Ponti, Korhonen, Vulic (2022). Composable Sparse Fine-Tuning for Cross-Lingual Transfer.** https://arxiv.org/abs/2110.07560
- Lottery-ticket sparse fine-tunings for language and task are summed into one model. Their analysis finds sparsity is crucial to prevent interference between composed fine-tunings and to prevent overfitting.
- Takeaway: the closest prior result to "compose sparse skill components by adding them". It supports union as a reasonable default.

**Mallya, Davis, Lazebnik (2018). Piggyback.** https://arxiv.org/abs/1801.06519 (abstract). Binary masks over frozen pretrained weights, one per task, no forgetting. Takeaway: masks over frozen weights are an old idea; the new part for us is composing them.

**Wang et al. (2024). LiNeS: Post-training Layer Scaling.** https://arxiv.org/abs/2410.17146
- Shrinking shallow-layer updates keeps target accuracy and restores control-task accuracy. Shrinking deep-layer updates hurts target accuracy. Linear depth scaling also improves multi-task merging.
- Takeaway: prior for `restrict(layers=...)`: drop a union's deletions in early layers first when forgetting or collapse appears.

**Yu et al. (2024). The Super Weight in Large Language Models.** https://arxiv.org/abs/2411.07191
- Zeroing one scalar can destroy generation (perplexity up about 1,000x, zero-shot accuracy to chance). Super weights sit in an early layer's mlp.down_proj (Llama-7B: layers[2].mlp.down_proj[3968, 7003]).
- Takeaway: unions add zeros. Whether a gate_proj deletion can disable a super-activation channel in Llama-3.2-1B is unverified. Run a cheap collapse check before every full eval.

**Hitit, Girrbach, Akata (2025). A Systematic Study of In-the-Wild Model Merging for LLMs.** https://arxiv.org/abs/2511.21437 (abstract). With heterogeneous LLM experts, only Task Arithmetic reliably beat the base. Takeaway: do not assume TIES or DARE help on LLMs; test them inside the same budget.

### Priors to hand planners (my synthesis, to be tested)

- Interference should rise with mask count and with non-overlapping deletions. Start with pairs and triples.
- Overlapping deletions are agreement. Intersection and k-of-n votes are the "consensus" moves.
- Early-layer deletions carry the highest collapse risk. Restrict to middle and late layers first.
- Prefer the smaller of two programs within one standard error. Fewer deletions should also mean less forgetting.

---

## 5. Budgeted comparison methodology

**Li et al. (2018). Hyperband.** https://arxiv.org/abs/1603.06560
- Successive halving keeps the top 1/eta at each rung and multiplies the resource by eta. Hyperband runs several brackets that trade breadth for depth. They suggest eta = 3 or 4 and say results are not very sensitive to it.
- Takeaway: in our judge, the resource is "number of search questions". One bracket with eta = 3 and two rungs fits a 100 to 200 eval budget.

**Polo et al. (2024). tinyBenchmarks.** https://arxiv.org/abs/2402.14992
- 100 curated MMLU examples estimate full-MMLU accuracy within about 2% on average (IRT-based). Stratified random subsets are simpler but noisier.
- Takeaway: a fixed, stratified low-fidelity subset is a reasonable screen. Pair it with promotion to full fidelity.

**Miller (2024). Adding Error Bars to Evals.** https://arxiv.org/abs/2411.00640
- Report standard errors. When two models answer the same questions, use paired differences: SE(A - B) = sqrt(SE_A^2 + SE_B^2 - 2 SE_A SE_B Corr). Cluster standard errors when questions come in groups. Includes a sample-size formula for the minimum detectable effect.
- Takeaway: always score programs on the same fixed question set, and compare them with paired SE. That cuts noise more than adding questions.

**Agarwal et al. (2021). Deep RL at the Edge of the Statistical Precipice (rliable).** https://arxiv.org/abs/2108.13264
- Use interval estimates, stratified bootstrap CIs, IQM, and performance profiles. Percentile bootstrap CIs cover well from about 10 runs; with 3 runs they underestimate the true interval.
- Takeaway: with 3 seeds, show every seed's curve and the min-to-max band. Use bootstrap bands only with 5 or more seeds.

**Nathani et al. (2025). MLGym: A New Framework and Benchmark for Advancing AI Research Agents.** https://arxiv.org/abs/2502.14499
- Gym-style environment for research agents. Reports "Best Attempt" (best intermediate result) and "Best Submission" (the final pick the agent commits to) over 4 seeds, plus area under performance profiles.
- Takeaway: report both. Best search score is "best attempt"; the held-out score of the selected program is "best submission".

### Noise budget for our splits (my arithmetic, binomial SE at accuracy p)

| Split | Questions | p | SE (points) |
|---|---|---|---|
| GSM8K search split (design doc, graft space) | 200 | 0.3 to 0.5 | 3.2 to 3.5 |
| One MMLU-Pro subject, half of its test split | 48 to 169 | 0.18 | 5.4 to 3.0 |
| All 14 MMLU-Pro subjects, half each | about 1,500 | 0.18 | 1.0 |
| One BBEH task, half of its test split | 25 | 0.05 | 4.4 |
| All 23 BBEH tasks, half each | about 565 | 0.05 | 0.9 |

The 1B mask effect on MMLU-Pro is about 8 points, so a pooled objective over 4 or more subjects is needed to see search progress.

---

## 6. Environment conventions

**Gymnasium Env API.** https://gymnasium.farama.org/api/env/
- `reset(seed=None, options=None) -> (obs, info)`. `step(action) -> (obs, reward, terminated, truncated, info)`. `terminated` means a true terminal state; `truncated` means an outside limit such as a time or budget cap. `info` carries diagnostics and individual reward terms.
- Takeaway: budget exhaustion is `truncated=True`. Our task has no terminal state, so `terminated` stays False.

**Prime Intellect verifiers.** https://github.com/PrimeIntellect-ai/verifiers . Docs: https://docs.primeintellect.ai/verifiers/building-environments and https://docs.primeintellect.ai/hosted-training/environment-model
- An environment is a dataset plus a rubric (weighted reward functions) plus an optional parser, exposed by `load_environment()`. `SingleTurnEnv` scores one reply; `MultiTurnEnv` implements `env_response` (what the environment says back) and a completion check; `ToolEnv` adds tools. The newer v1 API splits a `Taskset` from an `Env` that runs agents and collects `Trace`s into an `Episode`.
- Takeaway: if we later RL-train a planner, our env maps onto `MultiTurnEnv`: the program is the assistant turn, the eval feedback is `env_response`, and the rubric returns the best search score. Not needed for the hackathon.

---

## 7. Recommendations for the host-mask space (Section 8.5 extends these to grafts)

### 7.1 Objective and budget unit

- Search objective: micro-averaged accuracy over a fixed group of 4 to 6 MMLU-Pro subjects (for example biology, chemistry, physics, math, engineering, computer science). Keep BBEH as a secondary pooled score, or switch to 3B masks if GPU time allows.
- Splits: carve search and held-out halves from the paper's 25% test portion of each subject, stratified, with a fixed seed. Freeze the question lists in a file. Confirm the paper's split first (unverified).
- Budget unit: full-eval equivalents (FEE). One FEE = one program scored on the whole search split of the objective group. A partial eval costs its fraction of questions. Cache hits cost 0 and are logged. Rejected programs (invalid, duplicate) cost 0.
- Budget: 150 FEE per run for every method. Held-out evals sit outside the search budget: exactly one per method per seed.

### 7.2 Planner prompt structure

Order the prompt this way. Keep it under about 6k tokens.

1. Language spec: ops (`use`, `|`, `&`, `-`, `restrict`, `tv`), their deletion semantics, the 16-layer layout, and output JSON schema (`Program.to_dict()` plus `hypothesis` and `edit_type`).
2. Goal and state: objective tasks, budget used and left, base score, and each single mask's score on the objective.
3. Catalog card per candidate mask: task, deletion count, deletions per 4-layer block, top-3 Jaccard overlaps, analyst one-line signature.
4. Analyst capability map and the lessons log (collapses with diagnosis, such as "union of 5 masks with early layers: format parse rate fell to 20%").
5. History: the top 12 programs sorted ascending by score, best last (OPRO), each with per-task scores and deletion count. Add the 3 most recent failures with diagnosis (Eureka reflection).
6. Parents: 2 programs sampled from the frontier with ShinkaEvolve weights (fitness sigmoid times 1/(1 + children)), shown as v0 and v1 (FunSearch).
7. Task: "Propose 3 programs. Each is one atomic edit of a parent (AIDE) or one new draft. Each must differ from every program above. Give a one-line testable hypothesis."

Edit menu, sampled with fixed odds per proposal slot (ELM): add mask 25%, remove mask 15%, swap mask 15%, change layer restriction 20%, consensus or intersection move 15%, fresh draft 10%.

Diversity: give each of K = 3 planners a directive (exploit frontier, explore unused masks, layer and interference fixer), in the spirit of FunSearch islands. Run the K = 1 greedy ablation too, since one paper found parallel investigators added nothing.

First round: the analysts plus planners propose 9 programs before any score exists (LLAMBO warm start). Include greedy-union seeds: the best single masks and the best pairs by low overlap.

### 7.3 Judge policy

- Dedup before spending budget: canonicalize the program, compile the deletion set, hash it. Reject if hashed before, or if Jaccard to any evaluated deletion set exceeds 0.95 (ShinkaEvolve threshold, adapted).
- Stage 0, collapse check (AlphaEvolve cascade): 32 fixed prompts, about 2% of an FEE. Mark "collapsed" and log a lesson if answer-format parse rate falls under 50% or accuracy falls below base by more than 3 SE.
- Stage 1, low fidelity: a fixed stratified third of the search split, the same questions every time, so comparisons are paired.
- Stage 2, promotion: per round of 9 candidates, promote the top 3 (eta = 3) to the full search split. Cost per round is about 9 x 1/3 + 3 x 2/3 = 5 FEE instead of 9.
- Frontier: top k = 8 by full-fidelity search score. A newcomer enters only if it beats the 8th member. Within one paired SE, keep the program with fewer deletions.
- Kill rule: programs outside the frontier cannot be parents. Additionally retire a frontier parent after 3 children that all score below it by more than one paired SE.
- Budget: hard stop when the next charge would exceed 150 FEE. At the end, the judge selects one program per run by full search score. That program alone goes to held-out, once.

### 7.4 Evolutionary baseline (same counter, same splits, same question subsets)

- Primary, Sakana-style CMA-ES: genome = one binary gene per (objective-task mask, 4-layer block). With 6 masks that is 24 genes. The program is the union of each included mask restricted to its block. Use Optuna `CmaEsSampler(with_margin=True, seed=s)` with genes as 0/1 integers. Population = 4 + floor(3 ln 24) = 13, so 150 FEE gives about 11 generations. Copy mergekit's sigma0 = 1/6. Start x0 slightly below the threshold (for example 0.4 on a [0, 1] scale, my choice), so the first generation does not union every mask. Fitness = full search split, no multi-fidelity, matching Sakana. Cap evals in our counter.
- Secondary, greedy union (model-soups recipe): sort masks by single-mask search score, add each if the score holds. Costs 6 to 7 FEE. Cheap, strong, and easy to explain.
- Control, random search: uniform random genomes over the same 24 genes, 150 FEE.
- Optional, regularized evolution: population 16, tournament 4, one-gene flip mutation, oldest dies (NAS-Bench-101 style, scaled down by me). Run it if a reviewer will ask about discrete GAs.
- Also report LoRA and FFT on the same training data, and the base model, on held-out only.

### 7.5 Eval-curve methodology

- Chart 1: x = cumulative FEE charged; y = best full-fidelity search score so far. One thin line per seed, median line per method, min-to-max band across seeds. Never plot low-fidelity scores as best-so-far.
- Seeds: at least 3 per method, 5 if GPU time allows. Swarm seed = LLM sampling seed and directive order. Baseline seed = optimizer seed. Question subsets stay fixed across all seeds and methods.
- Bands: min-to-max for 3 seeds; percentile bootstrap 95% CI across seeds only with 5 or more (rliable).
- "N times fewer evals": target = the baseline's median final best search score. For each swarm seed, find the first FEE where its best-so-far reaches the target. Report the ratio of medians and every per-seed value.
- Final table: held-out accuracy of each run's selected program, with paired differences against the CMA-ES selection, paired SE, and 95% CI (Miller). Report "best attempt" (search) and "best submission" (held-out) columns (MLGym).
- Count everything: every analyst call that touches benchmark questions goes through the same counter.

### 7.6 Environment API shape

- `reset(seed) -> (obs, info)`; `step(action) -> (obs, reward, terminated, truncated, info)`; `step_batch(actions) -> list` for K proposals. The CMA-ES and random baselines drive the same env through an ask-and-tell loop, so accounting is identical.
- `action`: `{"program": Program.to_dict(), "hypothesis": str, "agent": str, "fidelity": "check" | "low" | "full"}`. The judge sets fidelity; planners do not.
- `obs`: budget total and used, catalog with signatures, frontier (top 8), full history records, lessons log.
- `reward`: accuracy on the requested fidelity, as a float. Invalid, duplicate, or collapsed programs return 0.0 with a status, so the float contract holds.
- `info`: per-task scores, number of questions, SE, cost charged, cache hit, status (ok, invalid, duplicate, collapsed), deletion counts per layer, Jaccard to parents.
- `terminated` is always False. `truncated` turns True when the budget is spent.
- Every step appends one line to `experiments.jsonl`: timestamp, agent, hypothesis, program, split, fidelity, scores, SE, cost, cumulative FEE.

### 7.7 Open items

- Confirm the paper's 75/25 split code and seed in the deleting-weights repo (unverified).
- Super-weight coordinates for Llama-3.2-1B, and whether gate_proj deletions can disable them (unverified).
- OPRO's exact cap on history length (unverified; I use 12 by choice).
- Escalate to Arjun if pilot runs show per-round score changes within one paired SE of noise. That is the brief's "budget too small to separate the curves" trigger.

---

## 8. Mixed-family graft space

A program is now `{"host", "grafts": [{"id", "at"}], "heal": {"steps", "data"}}`, with host mask expressions still allowed. Compile reports a per-graft ridge fit (R squared from host layer k to donor layer l) before any heal. Compile plus heal takes minutes and counts against the budget. Design: `docs/superpowers/specs/2026-11-14-mixed-family-compiler-design.md`.

### 8.1 Layer and placement search

**Akiba et al. (2024), data-flow-space (DFS) merging.** https://arxiv.org/abs/2403.13187 (Sections 2.2, 3.1, 3.2)
- Searches a serial inference path through the layers of two models. Raw space is (M + 1)^T. They lay all M layers out in order, repeat them r times, and evolve an indicator array of T = M x r genes (a layer is used if its gene is above 0). With M = 64 and r = 3, T = 192.
- A matrix W (M x M) scales the input when the path jumps from layer i to layer j. Removing the scaling cut performance by more than 20%. W is fixed at 1 between consecutive same-model layers, so gains come from stacking alone.
- Model A's embeddings, first and last layers are fixed. Genes for model A's first repetition start at 2 sigma, so search starts as model A and avoids nonsense stacks.
- CMA-ES in EvoJAX, 100 generations x population 128 (about 12,800 candidates), fitness on 200-item batches, best snapshot by a 200-item validation set. The evolved path skipped one MetaMath layer; deleting that layer alone raised accuracy. Scaling stops working when a source model was fine-tuned much longer.
- Takeaway: our graft genome is a small DFS: (part, depth) inclusion plus a per-graft scale. Copy "start as the host". Budget is two orders of magnitude below Sakana's, so the genome must stay small.

**mergekit passthrough (frankenmerge).** README: https://github.com/arcee-ai/mergekit . `passthrough` copies tensors from one input model per slice, with `slices` listing layer ranges; the README lists frankenmerging, layer stacking, and model surgery as its uses. Akiba et al. use frankenmerging as an unoptimized baseline. Takeaway: an untrained stack is our graft with no heal, so run a "heal steps = 0" arm.

**Kim et al. (2023). SOLAR 10.7B: Depth Up-Scaling.** https://arxiv.org/abs/2312.15166 . Duplicate a 32-layer model, drop the last 8 layers of one copy and the first 8 of the other, stack to 48. The stack first scores below the base, then recovers fast under continued pretraining. They chose this cut over plain repetition to shrink the layer distance at the seam. Takeaway: give planners the depth gap per graft (donor depth fraction minus host depth fraction); small gaps should heal faster.

**Gromov, Tirumala, Shapourian, Glorioso, Roberts (2024). The Unreasonable Ineffectiveness of the Deeper Layers.** https://arxiv.org/abs/2403.17887 . Prune the block with minimum angular distance between its input and output, then heal with a small QLoRA finetune. Llama-2-70B keeps QA accuracy until about half its layers are gone. Never drop the final layer. QA accuracy shows sharp transitions while C4 loss degrades smoothly. Takeaway: cheap representation distances can pick where surgery hurts least, and heal loss alone can mislead about accuracy.

**Sun, Pickett, Nain, Jones (2024). Transformer Layers as Painters.** https://arxiv.org/abs/2407.09298 . In frozen models the first and last layers differ from a fairly uniform middle. Skipping or reordering middle layers degrades gracefully; repeating the center layer is catastrophic. Takeaway: graft into middle host layers first; avoid the first two and last two.

**Bandarkar et al. (2024). Layer Swapping.** https://arxiv.org/abs/2410.01335 (abstract). Swapping a math expert's top and bottom layers for a language expert's (same base) gained 10% on MGSM. Takeaway: coarse `block` grafts are a real option when skills sit at different depths.

### 8.2 Stitching and cross-model transplant

**Lenc, Vedaldi (2015).** https://arxiv.org/abs/1411.5908 (abstract). Introduces trainable stitching layers to test whether two CNNs' representations are equivalent. The origin of model stitching.

**Bansal, Nakkiran, Barak (2021). Revisiting Model Stitching to Compare Neural Representations.** https://arxiv.org/abs/2106.07682
- Stitcher: a low-capacity layer (1x1 conv with BatchNorm; linear 768 x 768 for ViT) trained between two frozen models. Stitching penalty = accuracy drop. Same-architecture SGD minima stitch at all layers ("stitching connectivity"); representations from more data, width, or training improve weaker models when stitched in.
- CKA can be near 0 where stitching says two representations are close. Fine-tuning the top overestimates similarity relative to stitching.
- Control: stitching an untrained random bottom network gives a low penalty at early layers, so early-layer stitches prove little.
- Takeaways: a trained stitch's loss beats CKA as a closeness signal; a heavy heal can hide bad part selection; keep the glue low-capacity and identical across arms.

**Kornblith et al. (2019). Similarity of Neural Network Representations Revisited (CKA).** https://arxiv.org/abs/1905.00414 (unverified: not opened). Cheap, but Bansal et al. show it can mislead about functional compatibility. Log it; do not rank on it.

**Chen, Merullo, Stolfo, Pavlick (2025). Transferring Linear Features Across Language Models With Model Stitching.** https://arxiv.org/abs/2506.06609 (abstract). Affine maps between residual streams of different-size models transfer SAEs, probes, and steering vectors. Takeaway: affine glue across sizes works as a bridge.

**Armstrong, Ayoobi, Mukherjee (2026). Thinking in Different Spaces.** https://arxiv.org/abs/2603.20406 (abstract). Ridge maps across 20 heterogeneous LLM pairs reach R squared 0.50 (verbal) and 0.40 (math); a permutation control gives -0.22. Correlation between map quality and behavioral effect: r = -0.07. Maps fit on one domain collapse on another. Takeaway: R squared is a veto for broken pairings, not a ranking signal. Fit the ridge on task text, as the design does.

**Armstrong et al. (2026). Interpretability Is Not Model-Specific.** https://arxiv.org/abs/2611.12675 (abstract). 110 pairs, five families, 1B to 72B. Recovery depends on what the map was fit on; prompts from the region where a tool operates raised recovery sharply. A label-free coverage score tracked recovery across more than 16,000 runs. Takeaway: a coverage-style score on heal prompts is a candidate pre-heal predictor to test.

**Gorbett, Jana (2026). Ventriloquist LLMs.** https://arxiv.org/abs/2603.18908 (abstract). Late-layer affine alignment between independent models; tokenizer overlap predicts generation quality; strong-to-weak beats weak-to-strong. Takeaway: tokenizer overlap and donor strength are priors for which donors graft well.

**Sudakov et al. (2026). MATCHA.** https://arxiv.org/abs/2610.09058 (abstract). Learned cross-model layer maps across 42 pairs are broadly monotone in depth but many-to-many. Takeaway: the default `at = round(l / L_donor * L_host)` is a fair start; also try plus or minus 2.

**Syed (2026). Actionable Activation Directions across families.** https://arxiv.org/abs/2606.20225 (abstract). Ridge transfer across Qwen, Gemma, Llama-3.2-1B, and Ministral produced large effects, but random and orthogonal directions did about as well. Takeaway: cross-family effects can come from the map itself; the random-part control is mandatory.

**Which pre-heal signals should predict post-heal score (synthesis).** No paper I found measures this for cross-family grafts, so we must.
- **Ridge R squared (built in)** depends only on the host layer k and donor layer l, not on which neurons form the part. Every cluster from the same donor layer, random or chosen, gets the same R squared at the same k. It ranks placements, never parts.
- **Permutation-control R squared:** refit with shuffled position pairs; a real pairing must clear it clearly.
- **Short-heal stitch loss:** loss after 33 to 100 glue-only steps is a trained stitch penalty (Bansal). It does depend on the part. Costs seconds, no eval.
- **CKA and angular distance (Gromov):** cheap; log them, do not gate on them.
- Calibrate after about 10 full heals, recheck at 20: Spearman correlation of each signal with the final search score. Stop using a signal to prune if it stays under about 0.3.

### 8.3 Two-axis budget

**Snoek, Larochelle, Adams (2012).** https://arxiv.org/abs/1206.2944 . Model evaluation duration and maximize expected improvement per second. **Dodge, Gururangan, Card, Schwartz, Smith (2019). Show Your Work.** https://arxiv.org/abs/1909.03004 . Report expected best validation score as a function of trials or training time; report runtime and trial counts. **Sivaprasad, Mai, Vogels, Jaggi, Fleuret (2020).** https://arxiv.org/abs/1910.11758 (abstract). Fair comparisons charge each method its tuning cost. **Dehghani, Arnab, Beyer, Vaswani, Tay (2022). The Efficiency Misnomer.** https://arxiv.org/abs/2110.12894 . FLOPs, parameters, and throughput can disagree; no single cost indicator suffices. **Hyperband** (Section 5): its original resource is training iterations, so heal steps are a natural resource.

Conventions (mine, building on these and on NAS-Bench-101):
- Two counters for every arm: B_eval (generation evals on the full search split) and B_sec (GPU-seconds on one fixed GPU type for compile, ridge fit, heal, and eval). A run stops when either is spent; log which one bound.
- Set B_sec from a pilot: time 3 programs through each phase, then size B_eval to bind at about the same point.
- GPU-seconds is the primary x-axis; eval count is the second chart. Claims must hold on both.
- LoRA and FFT get the same B_sec, spent on training plus their own learning-rate and rank selection on the search split. Draw them as expected-best curves against GPU-seconds (Dodge), plus one LoRA matched to glue plus part parameter count.
- Report the swarm's LLM token cost separately; it is not GPU time, and it should not be hidden. Report inference seconds per token for each final model, since grafts add inference cost that a merged LoRA does not.

### 8.4 Controls: random-part ablations

**Panigrahi, Saunshi, Zhao, Arora (2023). Task-Specific Skill Localization in Fine-tuned Language Models.** https://arxiv.org/abs/2302.06600 . "Model grafting" of about 0.01% of fine-tuned parameters recovers over 95% of fine-tuned accuracy; learned regions far beat random regions of the same size at low sparsity.

**Lu, Grover, Abbeel, Mordatch (2021). Pretrained Transformers as Universal Computation Engines.** https://arxiv.org/abs/2103.05247 . Freeze GPT-2's blocks, train only input, output, and norm layers. CIFAR-10: pretrained 68.2%, random frozen 61.7%, random with pretrained weight statistics 56.5%. Glue around random weights gets most of the way.

**Aghajanyan, Zettlemoyer, Gupta (2020).** https://arxiv.org/abs/2012.13255 . About 200 trained parameters in a random subspace reach 90% of full fine-tuning on MRPC (RoBERTa-Large). **Adewuyi, Okibe, Ivanov (2026). The Multiple Ticket Hypothesis.** https://arxiv.org/abs/2602.01599 (abstract). A random 1% of parameters matches full RLVR, and random masks barely overlap (Jaccard at most 0.005). Together: our glue (about 6M parameters per graft per the design doc) can learn the task by itself.

Also: Bansal's random-network stitch, Syed's random and orthogonal directions, Armstrong's permutation control, and Deletion Tuning's random sparse baseline (Sections 1, 8.2).

Control design (mine):
- **Matched random part (primary):** same donor, layer, kind, and unit count, random unit indices; same `at`, glue init, heal steps, heal data, and seed. Its R squared equals the real part's, so the comparison isolates unit choice. Draw at least 3 per real part and report the real part's rank.
- **Random-donor part (secondary):** same kind and size from a random donor at its default depth. Isolates donor choice, but changes the pairing too.
- **Random-part search arm:** the planner's part choice replaced by uniform random ids, with the same judge, glue, heal, and budgets. Compares "best of N searched" with "best of N random".
- **Glue-free arms:** host plus the same heal with no grafts (or a matched-parameter LoRA), and grafts with heal steps = 0.
- If selected minus random on held-out sits within one paired SE, the heal is doing the work. Say so in the report.

### 8.5 Recommendations for the graft space

**Planner prompt fields.** Add to 7.2: host card (layers, hidden size, fragile first and last 2 layers); part cards (donor model, family, size, kind, donor layer, units, parameter count, extraction signature, default `at`, depth gap, tokenizer-overlap note); per evaluated graft: R squared at its `at` and at plus or minus 2, permutation R squared, short-heal loss, full heal loss start and end, search score with paired SE against the no-graft and random-part controls, and seconds spent. Show fit next to final score in history so planners learn how well fit predicts outcome. Edit menu with fixed odds: swap part 25%, move `at` 20%, add graft 15%, drop graft 10%, change heal steps 10%, add or edit host mask 10%, fresh draft 10%.

**Judge fidelity ladder.**
- Dedup (cost 0): hash parts, depths, heal, and host mask.
- **F0, compile and fit** (no eval; seconds only): R squared at the proposed `at` and plus or minus 2, plus the permutation control. Veto a graft whose R squared does not clear the permutation control clearly, or that sits in the bottom quartile of R squared over host depths for its donor layer. Suggest the best neighbor depth. Donors too large to run skip F0 (fit marked skipped).
- **F1, short heal** (no eval; seconds only): 33 to 100 glue steps; held-back heal loss and the answer-format parse rate on 32 fixed prompts. Mark collapsed below 50% parse rate.
- **F2, full heal and eval** (one eval): default 300 steps, then generation on the full search split.
- Per round of 9 proposals: F0 vetoes, survivors to F1, top third by F1 loss to F2 (eta = 3, successive halving over heal steps). Rank by expected gain per second when heal lengths differ (Snoek). Frontier k = 8; within one paired SE keep the cheaper program (fewer glue parameters, fewer seconds).
- Baselines use the same ladder, so the main comparison isolates the proposer.

**Evolutionary baselines over grafts.**
- **Primary, regularized evolution** (fits a variable-length, categorical genome better than CMA-ES): genome = up to 3 grafts (part id, `at`) plus heal steps in {100, 300}. Population 12, tournament 3, oldest dies. Initial population: 12 single-graft programs at default depth. Mutations with fixed odds: swap part 30% (same kind; half the time same donor at another layer), move `at` 25% (plus or minus 1 to 3, clamped to host layers 2 to 13), add graft 15%, drop graft 15% (when 2 or more), change heal steps 15%; plus flip one host-mask block bit when masks are in play. Optional crossover 20%: union of parents' grafts, capped at 3.
- **Reference, Sakana DFS-style CMA-ES:** binary genes over (candidate part, depth band), for example 8 parts x 4 bands = 32 genes, one scale per graft. Optuna `CmaEsSampler(with_margin=True)`, population 4 + floor(3 ln 32) = 14, x0 near 0 so search starts as the host. Full-fidelity fitness, as in Sakana.
- **Cheap baselines:** greedy add-if-better over single grafts at default depth (model soups recipe); the random-part search arm.

**Curves.** Chart 1a: best F2 search score against cumulative GPU-seconds; Chart 1b: against eval count. One thin line per seed, median per arm, min-to-max band at 3 seeds. Arms: swarm, graft GA, random-part arm, CMA-ES reference, LoRA, FFT, host, no-graft heal, heal-0. "N times fewer": seconds and evals for the swarm to reach the graft GA's median final best. Final table: held-out score of each arm's selected program, paired differences against the GA and the random-part arm, with paired SE and 95% CI.

**Env API deltas.** `obs` and `info` carry both budgets (total and used), fidelity reached, per-graft R squared and permutation R squared, chosen and suggested `at`, heal loss summary, parse rate, seconds per phase, and eval charged. `truncated` turns True when either budget is spent. Each `experiments.jsonl` line logs both cumulative budgets.

**Open items.** Spearman calibration of R squared, F1 loss, and CKA against F2 score. Escalate to Arjun if pilot timings give fewer than about 40 F2 evals per arm per seed; at a 3.5-point SE on 200 GSM8K items, curves may not separate.

## Top recommendations (summary)

1. Use ridge R squared to choose `at` and veto pairings only; it is identical for any part from the same donor layer.
2. Treat the random-part control as a full search arm with matched donor, layer, size, glue, and heal; report rank among 3 or more random draws.
3. Budget on two axes, GPU-seconds (primary) and evals; stop when either runs out; give LoRA and FFT the same GPU-seconds including tuning.
4. Judge ladder: F0 fit veto (free in evals), F1 short heal loss, F2 full heal plus eval, eta = 3, same ladder for every arm.
5. Graft GA baseline: regularized evolution, population 12, tournament 3, with swap-part, move-at, add, drop, and heal-step mutations.
6. Sakana DFS-style CMA-ES with margin over (part, depth band) bits, starting as the host, as the literature reference.
7. Planner prompt: part cards with depth gap and fit, ascending top-12 history with fit next to final score, 2 parents, one atomic edit each.
8. Score every program on fixed items and compare with paired SE; GSM8K's 200-item search split has about 3.5 points of noise.
9. For host masks: pooled MMLU-Pro objective, splits from the paper's 25% test portion, CMA-ES with margin over 6 masks x 4 blocks.
10. Gymnasium-style env: `truncated` on either budget, every step in `experiments.jsonl`, baselines through the same env.
