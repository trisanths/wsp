# Part 3 design: RL environment, agent swarm, budgeted baselines

Version 2 (2026-11-14). It folds in part 2's update to the part 3 brief (programs are now graft lists across model families), saved verbatim in [briefs/brief-R-update-1.md](../briefs/brief-R-update-1.md). Inputs: the brief [briefs/brief-R-rl-env-swarm.md](../briefs/brief-R-rl-env-swarm.md), Trisanth's [mixed-family compiler design](superpowers/specs/2026-11-14-mixed-family-compiler-design.md), and the research notes in [docs/research/](research/) (`papers.md`, `modal.md`, `exa.md`, `pastdev.md`). Status board: [issue #1](https://github.com/trisanths/wsp/issues/1).

## 1. Intent

Prove Claim B: at a fixed budget, LLM agents writing programs in Trisanth's language match or beat evolutionary search over the same graft space, or reach a given score with N times fewer evals. Judges reward measured gains, so every number must come from a fair, logged, equal-budget run, and final numbers come from the held-out split, run once.

Goals:
- An environment where `step(program) = eval(compile(program), "search")`. A program is a host plus grafts plus an optional heal pass, and can still carry host masks. It runs on stubs today and on the real compile and eval later by changing one config value.
- A swarm (analyst, K planners, judge) that learns from its own history, including collapsed grafts, and uses the free per-graft fit score to skip bad placements.
- Baselines at equal budget: evolutionary search over the same graft space (primary), random search, LoRA and full fine-tuning. The "random cluster through the same glue" control and the no-graft heal control run on every final program.
- Chart 1 (best combined score vs evals, swarm vs evolutionary), a compute-axis version of it, and a scrolling agent log.
- Modal, past.dev and Exa used where each one is the right tool, with an ablation that shows what each adds.

Non-goals: Person B's eval harness, Trisanth's compiler and heal, Person A's extraction, routing. Part 3 calls them through interfaces and uses stubs until they land.

Anti-goals (a failure even if it runs):
- Any arm gets more evals, more compute, more memory from earlier runs, or a different literature pack than another without that being counted and reported.
- A gain that comes only from heal training and not from the grafted part (the controls catch this).
- A stub or search-split number appears in a result, chart or slide without a STUB or SEARCH label.
- Code a teammate can't read: dense comprehensions, `setdefault`, heavy type annotations, clever metaprogramming.

## 2. Files and ownership

All new; part 3 never edits `wsp/program.py`, `wsp/compiler.py`, `wsp/masks.py`, `wsp/catalog.py`, `wsp/__init__.py`, `modal_app.py`, `catalog.jsonl`, or anything under `briefs/` except its own update file.

| File | What | Built in |
|---|---|---|
| `wsp/swarm/__init__.py` | minimal | core |
| `wsp/swarm/config.py` | run config, `.env` loader, run id | core |
| `wsp/swarm/schema.py` | the ONLY place that knows the program JSON layout: read and write host, masks, grafts, heal; `base` alias | core |
| `wsp/swarm/space.py` | canonical JSON, hash, validation, random program, mutate, crossover, describe, control programs; uses `schema.py` | core |
| `wsp/swarm/scoring.py` | normalized and combined score, collapse rule | core |
| `wsp/swarm/budget.py` | two-axis `Budget` interface, `LocalBudget`, shared ledger | core |
| `wsp/swarm/log.py` | experiment log rows and writer | core |
| `wsp/swarm/stub.py` | stub graft catalog, stub compile with fit scores, stub heal, stub eval, stub cost model (all marked STUB) | core |
| `wsp/swarm/backends.py` | `CompileEval` interface, `StubCompileEval`, `LocalCompileEval` (Trisanth's compile on this machine) | core |
| `wsp/swarm/env.py` | `WspEnv` with `probe`, `step`, `step_many`, `heldout_eval`, `run_controls` | core |
| `wsp/swarm/interfaces.py` | `Memory`, `Literature`, `LLM` interfaces and null versions | core |
| `wsp/swarm/agents/base.py`, `agents/scripted.py` | agent interfaces, scripted analyst and planner, rule judge | core |
| `wsp/swarm/swarm.py` | the swarm loop | core |
| `wsp/swarm/factory.py` | builds budget, backend, memory, literature and LLM from config, with lazy imports | core |
| `scripts/run_swarm.py` | CLI for one swarm run | core |
| `tests/test_swarm_core.py` | core tests | core |
| `modal_swarm.py`, `modal_llm.py`, `wsp/swarm/modal_backend.py`, `scripts/modal_setup.py`, `tests/test_swarm_modal.py` | Modal integration | modal |
| `wsp/swarm/memory.py`, `tests/test_swarm_memory.py` | past.dev memory | memory |
| `wsp/swarm/literature.py`, `scripts/build_priors.py`, `exa_cache/`, `tests/test_swarm_literature.py` | Exa priors | literature |
| `wsp/swarm/llm.py`, `wsp/swarm/prompts.py`, `wsp/swarm/agents/llm_agents.py`, `tests/test_swarm_llm.py` | LLM clients and LLM agents | agents |
| `wsp/swarm/baselines/*.py`, `scripts/run_baseline.py`, `tests/test_swarm_baselines.py` | evolutionary, random, LoRA and full fine-tuning plans | baselines |
| `scripts/compare.py`, `scripts/plot_curves.py`, `scripts/agent_log.py`, `scripts/sync_log.py`, `tests/test_swarm_outputs.py` | equal-budget comparison, charts, agent log | outputs |

Dependencies: core adds nothing beyond numpy and the standard library. Integration agents add their client library to a new optional group `swarm` in `pyproject.toml` (`modal`, `exa-py`, `requests`, `openai`, `matplotlib`). Each agent adds only its own line.

## 3. Code style (from the brief)

Plain, readable Python: simple loops and ifs, small functions, docstrings in plain English. No dense comprehensions, no `setdefault`, no heavy type annotations, no metaclasses. Fixed seeds everywhere (`random.Random(seed)`, `numpy.random.default_rng(seed)`). Every network client fails soft (logs and degrades) except the budget, which fails closed.

## 4. Shared data shapes

### 4.1 Program (draft schema from part 2, pending lock)

```json
{"host": "unsloth/Llama-3.2-1B-Instruct", "seed": 0,
 "masks": null, "task_vectors": [],
 "grafts": [{"id": "part_qwenmath15_l14_mlp_c3", "at": 9}],
 "heal": {"steps": 300, "data": "gsm8k_train"}}
```

- `at` is optional. The default host depth is `round(l / L_donor * L_host)` (Trisanth's placement rule).
- `heal` is optional. When it's missing, the model is compiled with `W_out = 0`, which equals the host exactly.
- `base` is accepted as an alias of `host`, and old mask-only programs still load.
- Trisanth will lock the exact schema in `wsp/program.py` and post it on issue #1. Until then only `schema.py` reads or writes these keys. When the lock lands, `schema.py` changes and nothing else does.

`space.py` gives:
- `canonical_json(program)`: sorted keys, grafts in listed order (order matters for compile), no whitespace. `program_hash(program)`: sha256 of that, first 16 hex chars. It is the model hash in every cache, since compile and heal are deterministic for a given program and seed.
- `validate(program, catalog)`: a list of error strings.
  - Every graft id exists and has a graftable type (`mlp_cluster`, `attn_heads`, `expert` or `block`).
  - `at` is within `0..L_host-1`, where `L_host` comes from the catalog's host record or config (16 for Llama-3.2-1B).
  - There are at most 4 grafts, and no two grafts share a part id and the same `at`.
  - `heal.steps` is in the allowed set from config (default `[0, 100, 300, 600]`), and `heal.data` is in the allowed list for the run's task type.
  - Host masks follow the version 1 mask rules.
- `random_program(rng, catalog, config)`: one to three random grafts with a random `at` and heal steps.
- `mutate(program, rng, catalog, config)`: one of these moves, with the rates in `papers.md` Recommendations:
  - swap a graft's part for another part, preferring the same donor or kind;
  - move `at` by -2 to +2, or reset it to the default depth;
  - add a graft;
  - drop a graft;
  - change the heal steps.
- `crossover(a, b, rng)`: mix the two graft lists.
- `describe(program)`: short text such as `host + qmath.l14.c3@9 + qcoder.l10-12@5, heal 300 on gsm8k_train`.
- `random_glue_control(program, rng, catalog)` and `no_graft_control(program)`: the two required controls. The first keeps the glue and heal and replaces each part with a random cluster of the same shape from the same donor layer. The second drops every graft and keeps the heal. How a random cluster is written in program JSON is a question for part 2 (section 14). Until then, `schema.py` writes `{"id": ..., "at": ..., "control": "random_cluster", "control_seed": s}`.

### 4.2 Task type and scores

Each run targets one task type (the output is one fixed model per task type). Config gives `task_type` (default `math`), its tracked tasks with weights, and guard tasks. The first real setup is GSM8K: a 200-item search split from GSM8K train, the full GSM8K test as held-out, and the rest of train as heal data (Trisanth's splits). Stub default: target `gsm8k` (weight 1.0) and guard `if_guard`, a stub instruction-following check.

`scoring.py`:
- `normalized(score, base) = (score - base) / (1 - base)`: the fraction of headroom gained, negative when worse than the host.
- `combined(scores, base_scores, weights)` is the weighted mean of normalized scores over target tasks, minus a guard penalty when a guard drops more than 10 percent below base. This is the reward.
- `is_collapse(scores, base_scores)` is true when any target or guard falls below half its base. A collapse still costs its eval, logs `status="collapsed"`, and becomes a lesson.

### 4.3 Experiment log row

One JSON object per line. Eval rows:

```json
{"timestamp": "2026-11-14T23:40:00Z", "run_id": "s0-swarm-1731627600", "method": "swarm",
 "backend": "stub", "agent": "planner-2", "round": 3, "hypothesis": "...", "citations": [],
 "program": {}, "program_hash": "ab12...", "describe": "...", "split": "search",
 "fit": {"part_qwenmath15_l14_mlp_c3@9": 0.62}, "heal_loss": [2.1, 1.7],
 "scores": {}, "combined": 0.12, "status": "ok", "cache_hit": false,
 "token": 17, "eval_count": 18, "eval_budget": 60,
 "compute_s": {"compile": 41.0, "heal": 150.0, "eval": 120.0}, "compute_used_s": 5400.0, "compute_budget_s": 36000,
 "extra": {}}
```

- `status` is one of `ok`, `collapsed`, `invalid`, `rejected_fit`, `compile_error`, `cache_hit`, `infra_error`, `budget_exhausted`, `control`.
- Probe rows (fit-only compiles) use `"type": "probe"`.
- Event rows (analyst map, judge verdict, Exa call, memory op) use `"type": "event"`, with `kind` and `text`, and carry the current counters unchanged.
- Every row carries `backend`.

Each run writes `outputs/runs/<run_id>/experiments.jsonl` (one writer per file). `scripts/sync_log.py` merges all run files, plus the Modal `wsp-exp` Dict when present, into the shared `experiments.jsonl` at the repo root, sorted by timestamp. Never append to a shared file from two processes.

## 5. Budget and fairness rules

Implemented once, in `budget.py` and `env.py`, so no arm can bypass them. They apply to every arm the same way.

1. **Two axes, equal caps.** Every arm of a comparison gets the same eval budget (a pool of tokens `0..N-1`) and the same compute budget in seconds. Compile, heal and eval time all count against compute, as part 2 asked. The episode ends when either axis runs out.
   - Locally: the token pool sits in memory behind a lock, and compute is a locked running sum.
   - On Modal: tokens are a partition of the `wsp-budget` Queue, verified under concurrency (`modal.md` section 6), and compute is summed from the `wsp-ledger` Dict, which holds one write-once row per charge.
2. **One ledger.** Every charge from any arm (`run_id, method, axis, amount, token, who, time`) goes into one shared ledger: `outputs/ledger.jsonl` locally, the `wsp-ledger` Dict on Modal.
3. **Order inside `env.step`:**
   1. Validate. An invalid program is free.
   2. Look up the cache on `(program_hash, split)`. A hit is free and returns cached scores.
   3. Compile. This charges compute and produces fit scores.
   4. Apply the fit gate. If a graft's fit is below the config threshold, which is the same for every arm, the status is `rejected_fit` and no eval token is spent.
   5. Heal, which charges compute.
   6. Spend one eval token.
   7. Eval, which charges compute.
   A compile error charges the compute it used and no token.
4. **Probe is open to every arm.** `env.probe(program)` runs a fit-only compile (no heal, no eval). It returns fit scores, charges compute and spends no token. The evolutionary baseline may use it as well (a standard pre-screen), so the free fidelity level isn't a swarm-only advantage.
5. **Compute is checked before work starts and charged after it.** A job may overshoot the compute cap by at most the in-flight jobs. The overshoot is logged and reported.
6. **No refunds.** A failure after the token spend logs `infra_error` and keeps the token. Never put the token spend inside a retried Modal function: the caller pops and passes the token in.
7. **Held-out runs once, from separate pools.** `env.heldout_eval` works only after the search budget is spent. It covers at most 3 programs per run (the judge's top 3), each once, plus their two controls (section 10). Each run has its own held-out pool of tokens and compute, which is the same size for every arm. Those rows say `split="heldout"`.
8. **Memory is run-scoped.** past.dev audiences are per run (`pastdev.md` 6.1). Every benchmark run starts cold and asserts an empty recall first. A warm start is a separate, labeled condition.
9. **Literature is frozen.** The Exa priors pack is built before eval 1, and its sha256 goes into the run manifest. Graded runs use replay mode (`exa.md` 5.2, 5.3).
10. **Ablations at equal budget:**
    - evolutionary;
    - random;
    - swarm-scripted (no LLM);
    - swarm-LLM without memory or Exa;
    - swarm-LLM with memory;
    - swarm-LLM with memory and Exa.
    LoRA and full fine-tuning are compared on the compute axis.
11. **Seeds.** Every condition runs seeds `0..S-1` (default S=3), with the same seed list for every condition. Curves show the median and a min-max band.

## 6. Environment

`WspEnv(config, catalog, backend, budget, log)`:
- `reset()` returns the observation: `{"task_type", "tasks", "base_scores", "host": {...}, "parts": [...graftable catalog records with signatures...], "history": [...compact rows...], "probes": [...], "eval_remaining", "compute_remaining_s", "frontier": [...]}`.
- `probe(program, agent)` follows rule 4 and returns `{"fit": {...}, "compute_s": ...}`. Probes are cached by `program_hash` without the heal.
- `step(program, agent, hypothesis, citations=None, round_index=None)` returns `(observation, reward, done, info)`. `reward` is `combined`, or None when no eval ran. `info` is the full log row.
- `step_many(candidates)` runs a list in parallel (a thread pool locally, `starmap` on Modal) and returns rows in input order.
- `heldout_eval(program, agent)` and `run_controls(program, agent)` follow rule 7.
- `listeners`: callables `fn(row)` run after every row is written. Memory ingest hooks in here.
- `done` is true once either search budget axis is spent.

## 7. Stub compile, heal and eval

`stub.py` is plumbing, not a result. It is deterministic given `(program_hash, seed)`, and has enough structure that a planner reading signatures and fit scores can do better than random. Its cost model makes compute-budget behavior testable.

- **Stub catalog:** `stub_catalog(seed)` writes about 40 graftable parts to `outputs/stub/catalog.jsonl`, never to `catalog.jsonl`. They come from four stub donors: Qwen2.5-Math-1.5B (28 layers, hidden 1536), Qwen2.5-Coder-1.5B (28 layers), Llama-3.2-3B-Instruct (28 layers, hidden 3072) and one MoE donor with `expert` parts. Kinds are `mlp_cluster`, `attn_heads`, `expert` and `block`. Each part gets a hidden value per task type (math parts high on `gsm8k`), a home relative depth `l / L_donor`, and a size. The record also includes a host record (Llama-3.2-1B-Instruct, 16 layers, hidden 2048). Ids and records are marked `"stub": true`.
- **Fit score per graft:** an R squared in `[0, 1]`. It is high when the relative depth of `at` is close to the part's home relative depth, lower for a different family or a large hidden-size ratio, and lower for `block` parts placed far from home. Noise is deterministic.
- **Heal:** effective strength per graft is `fit` raised toward 1 by heal steps, with diminishing returns: 300 steps closes most of the gap, and 0 steps leaves an unhealed graft that hurts.
- **Eval:** host base score plus the sum of `value * strength` over the grafts. Grafts at the same or adjacent depth get an interference penalty. Low strength and many grafts can collapse the guard. A heal-only gain from the glue is added to every program with heal (this is what the controls subtract). There's noise of about 0.01.
- **Controls:**
  - A random-cluster control keeps the glue and heal gain and gets a value near zero.
  - A no-graft control gets the heal-only gain alone.
- **Cost model:**
  - compile: 20 s plus 15 s per graft;
  - fit probe: the compile cost;
  - heal: 0.5 s per step per graft;
  - eval: 120 s.
  The stub reports these as `compute_s` without sleeping.
- Every stub row carries `"backend": "stub"`, and every chart drawn from stub rows is watermarked STUB.

## 8. Agents and the loop

Interfaces live in `agents/base.py`:
- `Analyst.build_map(observation, ctx)` returns `{"text", "by_task": {task: [part ids, best first]}, "placements": {part id: [good at values]}, "notes"}`.
- `Planner.propose(observation, cap_map, lessons, priors, n, ctx)` returns a list of `{"program", "hypothesis", "citations", "probe_first": bool}`.
- `Judge.update(frontier, rows, ctx)` returns `{"frontier": [...top-k rows...], "killed": [...patterns...], "text"}`.

`ctx` holds `config`, `rng`, `llm`, `memory`, `literature`, `log`, and `env` (for probes).

Implementations:
- `ScriptedAnalyst`, `ScriptedPlanner` and `RuleJudge` (core) use no LLM. The scripted planner mixes exploitation (mutate frontier programs) with exploration (graft the analyst's top parts for the task at their best known placement). It probes when `at` is untested.
- `LLMAnalyst` and `LLMPlanner` (agents wave) build prompts per `papers.md` Recommendations. The prompt carries:
  - the host and graftable parts with signatures;
  - the capability map;
  - the frontier with scores and fit values;
  - recent probes;
  - recent collapses;
  - recalled lessons;
  - literature priors.
  Output is JSON validated by `space.validate`. Invalid JSON gets one repair retry and never costs an eval.
- The judge ranks with rules: deterministic, on search scores only. An LLM may write the verdict text. Defaults: a top-5 frontier, and a pattern is killed after 2 collapses or 2 fit rejections. A pattern is a part id plus an `at` band (early, middle or late) plus whether heal was used. Multi-fidelity follows `papers.md`: the fit probe is level 0 and the search-split eval is level 1.

Loop (`swarm.py`):
1. `memory.start_run`, `literature.priors()`, `env.reset()`.
2. Each round r:
   1. Rebuild the analyst map in round 0 and every 5 rounds.
   2. Call `memory.flush()`.
   3. Each planner recalls lessons, then the planners propose in parallel.
   4. Probe the proposals marked `probe_first` and drop those the fit gate would reject.
   5. Drop duplicates (same `program_hash`), killed patterns and invalid programs.
   6. `env.step_many`, then `judge.update`. Log the verdict and record it to memory.
3. Stop when either search budget axis is spent.
4. Run `heldout_eval` and `run_controls` on the top 3, each once.

## 9. Integrations

### 9.1 Modal (`modal.md`)
- **Auth:** only through `MODAL_TOKEN_ID` and `MODAL_TOKEN_SECRET` from `.env` (workspace `rexheng`). Never run `modal token set`. Trisanth's app lives in another workspace, so part 3 deploys its own apps and volume in `rexheng`.
- **`modal_swarm.py`, app `wsp-swarm`:**
  - Objects:
    - Queue `wsp-budget`: token pools as partitions per run.
    - Dict `wsp-ledger`: write-once charge rows.
    - Dict `wsp-cache`: key `program_hash|split|tasks_hash`.
    - Dict `wsp-probe`: probe cache.
    - Dict `wsp-exp`: log rows keyed `<run_id>:<seq>`.
    - Volume `wsp-swarm-models`: donor tensor cache and the kept models (frontier, heal inputs).
  - `probe_candidate(program)`: fit-only compile.
  - `run_candidate(program, split, tasks, token)`: compile, heal and eval in one container, so no model round-trips through the Volume. It times each phase for the compute ledger. The caller pops the token and passes it in, so retries can't double-spend. Before the real compiler and eval land, it calls the stub, and the Modal plumbing is still real.
  - `baseline_driver(config)` and `swarm_driver(config)`: run whole searches inside Modal. They start with `.spawn`, with `max_containers=1` and no retries.
- **`modal_llm.py`, app `wsp-llm`:** an open-weight instruct model on vLLM inside a `modal.Cls`, with a batched `generate` method called through Modal auth. There's no public endpoint, and one batch call covers a round of planners. The model and GPU come from config; the default is a 7B to 8B instruct model on L40S, with `scaledown_window` 300 s and `max_containers=1`.
- **`modal_backend.py`:** `ModalBudget` (same two-axis interface as `LocalBudget`), `ModalCompileEval`, a cache adapter and a log mirror.
- **Cost controls:** a timeout on every function, `max_containers` everywhere, no `min_containers`, GPU only where needed. Part 2 notes $250 of Modal per person.

### 9.2 past.dev (`pastdev.md`)
`PastMemory` implements `Memory`:
- `start_run` creates audience `<run>-lessons` for identities `<run>:analyst`, `<run>:planner-1..K` and `<run>:judge`, then asserts an empty recall.
- Ingests are batched once per round with idempotency keys. They run on a background thread and wait in `flush()` with a 30 s timeout. A timeout logs `memory_lag`.
- What goes in: step results with fit values, probe results, collapse and fit-rejection lessons with a pattern key, judge verdicts, and analyst maps. Ids look like `<run>:<kind>:<n>`.
- Planners recall before proposing (two or three angle queries at `low`), the judge recalls recent collapses, and the analyst recalls placement notes. Superseded results are ignored.
- `end_run(cleanup)` deletes points by id. Any send without a run audience is refused. On 5xx or 402 it degrades to local history and logs `memory_degraded`.

### 9.3 Exa (`exa.md`)
`ExaLiterature` implements `Literature`:
- `scripts/build_priors.py` runs the pre-pass:
  - one `auto` search with highlights per donor model and part kind, with `category="publication"`;
  - about 8 `deep-lite` topical searches with `output_schema`, on model stitching, layer placement, cross-family transplant, which layers store arithmetic or code skill, and heal and adapter training.
  It writes a frozen `priors.json` and the `exa_cache/` folder.
- Planners make at most one cached `lite` search per hypothesis and attach citations. The judge logs citations whose URL is not in the cache as ungrounded.
- Modes: `live`, `replay` (graded runs) and `off`. Every call is logged as an event row and never touches the budget.
- Dataset mirrors are excluded, and held-out text is never sent.
- There's no key yet, so it is built and tested with mocks. The smoke test in `exa.md` section 6 runs once the key arrives.

### 9.4 LLM backends (`llm.py`)
`MockLLM` (tests, scripted replies), `OpenAICompatLLM`, `ModalVLLM` (the `wsp-llm` Cls) and `AnthropicLLM` (used only if `ANTHROPIC_API_KEY` is set). All share `complete(messages, schema=None, temperature, seed, max_tokens)` and `complete_many(list_of_messages, ...)`.

## 10. Baselines and controls (`papers.md` Recommendations)
- **`baselines/evolve.py`:** the primary baseline. An evolutionary search over the same graft space with the `space.py` operators, sharing the env, budgets, probe access and log. The algorithm, population and rates follow `papers.md` for small budgets (Sakana-style, over the graft list and placements).
- **`baselines/random_search.py`:** the sanity floor.
- **`baselines/lora.py` and `baselines/full_ft.py`:** plans and stub runners in this wave. Real runs use Tinker (LoRA) or Modal GPUs (both), trained on the same heal data. Each checkpoint eval spends an eval token, and training time counts against the compute axis.
- **Controls (`env.run_controls`):** every final program gets the random-cluster-through-the-same-glue control and the no-graft heal control, each scored once on held-out from the held-out pools. A LoRA with matched trainable parameters is the second control from part 2's design, and runs when the LoRA baseline lands.

## 11. Outputs
- `scripts/compare.py` runs the conditions in rule 10 for seeds `0..S-1` at one budget and writes `outputs/compare/<name>/`.
- `scripts/plot_curves.py` draws:
  - Chart 1: best-so-far combined score on the search split vs eval count, with the median and band per condition;
  - Chart 1b: the same against compute hours.
  It watermarks STUB when any row is a stub, and the axes say "search split".
- `scripts/agent_log.py` is a scrolling log (eval count, agent, hypothesis, program, fit, score, status) in the terminal with `--follow`, plus a static HTML page for the demo.

## 12. Gates
- **Hour 1:**
  - `python scripts/run_swarm.py --backend stub --eval-budget 60 --seed 0` and `python scripts/run_baseline.py --method evolve --backend stub --eval-budget 60 --seed 0` both finish and write logs;
  - `plot_curves.py` draws a STUB chart;
  - tests pass.
- **Hour 8:** the evolutionary baseline runs detached on Modal against the real compile (or the Modal-hosted stub, if the real compile isn't ready), with the real budget plumbing.
- **Hour 16:** the first real swarm vs evolutionary curve at a fixed budget.

## 13. Defaults taken (Arjun can overrule on issue #1)
- These cost no eval token: a cache hit, an invalid program, a fit rejection and a compile error. Compile time is still charged to compute.
- An infra failure after the token spend keeps its token.
- Each arm gets equal pools of its own on both axes, and one ledger records every charge.
- The probe and the fit gate are open to every arm.
- The judge ranks with rules. LLMs propose and explain; they don't score.
- Planners reach the Modal-hosted model through Modal auth, with no public endpoint.

## 14. Needs from part 2 (posted on issue #1)
1. A fit-only compile: `compile(program, heal=False)` or `fit_report(program, catalog)`, returning per-graft R squared without heal or eval.
2. How a program names the random-cluster control (same shape, same donor layer, seeded) and the no-graft control.
3. Per-phase timing in `compile_report.json` (compile, fit, heal seconds), so the compute ledger can use real numbers.
4. The locked schema, posted on #1.

## 14. Update 1 (2026-11-14): graft action space and research findings

This section overrides earlier sections where they differ. Sources: part 2's message to part 3, `docs/superpowers/specs/2026-11-14-mixed-family-compiler-design.md`, and `docs/research/papers.md` section 8 and Recommendations.

- **Action space.** A program is a host plus grafts plus an optional heal, with host masks still allowed: `{"host": ..., "grafts": [{"id": ..., "at": 9}], "heal": {"steps": 300, "data": "gsm8k_train"}}`. Part 2 will lock the schema in `wsp/program.py` and post it on issue #1. Until then, `space.py` goes through a small adapter, `program_schema.py`, that accepts both the old mask-only dict and the draft graft dict. When the schema locks, only the adapter changes. The output is one model per task type, with no routing.
- **Two-axis budget.** Eval tokens (as in section 5) plus GPU-seconds. Compile, heal, stitching fit and eval all charge GPU-seconds, including compiles that fail. Arms get equal caps on both. Chart 1's primary x-axis is GPU-seconds, with an eval-count version beside it. LoRA and full fine-tuning get the same GPU-seconds.
- **Cascade judge.** Level 0 is the per-graft fit score from compile (R squared, before heal), which costs no eval token. Then a 32-prompt collapse check, then a fixed one-third subset, then the full search split for the top third. Low fidelity uses a short heal, and only promoted programs get the full heal. Each level spends a fraction of a token, set in config, and every level is logged.
- **Judge settings.** Top-8 frontier. Ties within one paired standard error break toward fewer parameters. Retire a parent after 3 failed children. Dedupe before spending.
- **Planner prompt** (papers.md recommendation 7): language spec, catalog cards (with fit score and donor depth gap), top-12 history sorted worst to best with per-task scores, 2 sampled parents, and single-edit proposals each with a one-line hypothesis.
- **Baselines.**
  - Regularized evolution over grafts. Mutations: swap part, move `at`, add or drop a graft, change heal steps.
  - CMA-ES over the mask-by-layer-block genome, for mask-only programs.
  - Greedy union and random search.
  - The **random-part control**: the same glue and heal around a random part of the same shape from the same donor layer.
  - Heal-only and heal-steps-0 arms.
  - mergekit-evolve is not used: it can't express our operations and is deprecated.
- **Tasks.** BBEH is too weak at 1B to separate curves. The search objective is a pooled group of 4 to 6 MMLU-Pro subjects, or GSM8K once grafts land (part 2's first real run). Compare programs with paired standard errors on fixed question sets.
- **Env API.** Gymnasium-style with `step_batch`: `truncated` turns true when the budget is spent and `terminated` stays false. Held-out is scored once, on the single selected program per run.
- **Stub.** It must model grafts too: fit score from the donor-host depth gap and the family pair, a heal gain that rises with steps and is capped, and a random-part control that gains from heal but not from part content.
