# Part 3 build log (RL environment and agent swarm)

One-way log for Rex. Agents append an entry after each milestone and never read this file. Working state lives in `tasks/todo.md`.

## 2026-11-14, step 0: setup

- Cloned trisanths/wsp into `ai research` on `main`. Python 3.12 venv via uv, `pip install -e ".[dev]"`.
- Tests: 9 of 10 pass. The failure, `test_catalog_resolves_relative_paths`, is Windows-only: `os.path.join` in `wsp/catalog.py` yields `components/m` with a forward slash. Harmless on Modal (Linux). Left for Trisanth.
- Keys for Modal, past.dev and Tinker sit in `.env`, kept out of git through `.git/info/exclude` so the team `.gitignore` stays unchanged.
- past.dev key checked: a recall on the empty project returned HTTP 200.
- No Exa or Anthropic key on this machine yet.

## 2026-11-14, step 1: research wave launched

Four research agents run in parallel, each writing one file under `docs/research/`:
- `papers.md`: evolutionary merging (Sakana, mergekit-evolve), LLM-guided search (FunSearch, AlphaEvolve, OPRO, ShinkaEvolve), mask-based merging priors, budgeted-comparison methodology, LLM env API conventions.
- `modal.md`: auth through env vars, shared budget counter design, Volumes vs Dict vs Queue for the log and cache, fan-out, detached baseline runs, vLLM serving for planners.
- `exa.md`: endpoints, SDK, costs, cache design, which agent role calls what.
- `pastdev.md`: every route, ingest-to-recall latency, per-run scoping for a fair comparison, what each role writes and recalls.

Working rule from Rex: no branches, push straight to `main`. The orchestrator does not write code; subagents build, the orchestrator reviews.

## 2026-11-14, step 2: team status board

- Opened and pinned [issue #1](https://github.com/trisanths/wsp/issues/1) as the part 3 status board for teammates and their agents. The body gives a comment template (Part, Done, Files, Interface change, Needs, Next), part 3's current status, the files part 3 touches and avoids, what part 3 needs from parts 1a, 1b and 2, and Arjun's open questions.
- Rule from here: part 3 posts a comment on #1 at every milestone and reads new comments before each build wave.

## 2026-11-14, step 3: research notes in (Modal, Exa, past.dev)

- `docs/research/modal.md`: the keys belong to workspace `rexheng`, not Trisanth's, so part 3 deploys its own apps and volume. Auth goes through env vars only. The budget design is a Queue pre-filled with N tokens, one popped per eval. It was tested from 8 threads and from 4 containers with no duplicates and no losses. A Dict read-modify-write counter lost updates (6 of 40 survived), so it's ruled out. GPU per hour: L4 $0.80, A10 $1.10, L40S $1.95, A100-80GB $2.50, H100 $3.95.
- `docs/research/pastdev.md`: raw text is recallable about 4 s after a send, and derived memory after 10 to 13 s. Isolation is per run, with audiences such as `<run>-lessons`; a recall from outside the audience returned nothing. There's no completion route that could serve as a planner model. The cost estimate is about 1,300 credits per 200-eval run, against a balance of about 549,994.
- `docs/research/exa.md`: the `research paper` category is now `publication`, and the search types are `auto`, `lite`, `deep-lite` and `deep`. The design freezes a priors pack before eval 1 and runs graded runs in replay mode. A cold run costs about $0.9. There's no Exa key yet.

## 2026-11-14, step 4: part 2 changed the action space; spec version 2

- Trisanth's update (saved as `briefs/brief-R-update-1.md`) changes programs to a host plus grafts from any donor family, plus an optional heal. Each graft gets a fit score at compile time. Compile and heal count against the budget. A new control: a random cluster through the same glue.
- `docs/part3-design.md` version 2:
  - a two-axis budget (eval tokens and compute seconds);
  - a fit probe and gate open to every arm;
  - one `schema.py` module, so the schema lock touches one file;
  - a stub with graft parts, fit scores, heal and a cost model;
  - both controls on every final program;
  - the file split for six build agents after the core.
- The papers agent got an addendum covering graft and stitching search (Sakana data-flow-space merging, model stitching, cheap pre-heal fit signals, budget methodology on two axes).

## 2026-11-14, step 5: papers research in; schema locked by part 2; a parallel writer stopped

- `docs/research/papers.md` (422 lines) gives the key recommendations:
  - fit R squared only chooses `at` and vetoes, since it is identical for any part from one donor layer;
  - the random-part control becomes a full search arm;
  - a two-axis budget, with GPU-seconds as the main axis;
  - a judge ladder of fit veto, then short heal, then full heal plus eval, with eta 3;
  - the graft evolutionary baseline is regularized evolution with population 12 and tournament 3;
  - GSM8K's 200-item search split has about 3.5 points of noise, so escalate if an arm gets under about 40 full evals per seed.
- Trisanth locked the schema (`control: "random"`, heal with lr, batch size and sequence length) and rewrote the part 3 brief. Hour 1 now means the real compile and eval on the first pair: Llama-3.2-1B host, Qwen2.5-Math-1.5B donor, GSM8K.
- A forked session ran its own core agent in the same tree. It was stopped, and this session owns all part 3 code. Its spec section ("Update 1", commit 8a490ae) is adopted.
- Posted the blockers on issue #1: real catalog part records, the eval function, and the shared Modal workspace name.

## 2026-11-14, step 6: core built and reviewed (hour-1 gate on stubs: swarm half)

- The core agent built `wsp/swarm/` (16 modules), `scripts/run_swarm.py` and `tests/test_swarm_core.py`. It replaced every file the two forked agents had left (`schema.py` deleted; space, budget, config, log and scoring rewritten).
- Orchestrator review:
  - 66 of 66 tests pass, including Trisanth's (after installing `transformers` and `tokenizers` into the venv);
  - two seed-0 stub runs are identical, with best search score 0.3396 [SEARCH][STUB];
  - the top 3 on held-out [STUB] score 0.33, 0.28 and 0.29, against random-part controls of -0.001 to 0.028 and no-graft controls of 0.027 to 0.055;
  - no comprehensions or `setdefault` in the code.
- The cascade fidelity ladder (0.16, 0.34, 1.0 FEE) works. With seed 1 the cascade run stopped on compute, with 28.2 FEE used.
- Spec section 15 records the as-built API for the next wave.
- Wave A runs in parallel as a workflow: LLM clients plus vLLM on Modal, Modal infra plus the real compile and eval path with a pilot part set, past.dev memory, Exa literature. Each gets an independent review and a fix pass.

## 2026-11-14, step 7: Exa key added and checked live

- Rex supplied an Exa key, now in `.env` (still git-excluded).
- A live smoke through `wsp.swarm.literature.ExaLiterature` in live mode, with the cache in a scratch folder: one search for "model stitching across language model families which layer to graft" returned 3 publication cites in 0.6 s for $0.004. Among them were "Transferring Linear Features Across Language Models With Model Stitching" (NeurIPS 2025) and "Cross-Model Stitching Transfers Answers, Not Computation".
- The repeat query was a cache hit in 0.01 s at $0. `drain_calls()` logged both calls, and `known_url` recognized the cited URL.
- Next: build the frozen priors pack over the pilot catalog once wave A lands (`scripts/build_priors.py`; the dry run estimates about $0.6).

## 2026-11-14, step 8: wave B done; wave A LLM, memory and Exa done

Both waves ran as workflows. Each builder got an independent reviewer and, when the review found high or medium issues, a fix pass.

- **Wave B** (11 agents, no errors; 328 of 328 tests after integration):
  - **Stub calibration** to Trisanth's round-2 sweep: placement peaks at host layer 6 (about 0.39 against about 0.25 at layers 10 and 14); single-layer parts score within noise of their random control; rank-64 glue is never below full glue; host base 0.25; eval noise 0.023, the real within-cell sd. `glue_rank` is now a search dimension.
  - **Baselines:** regularized evolution (population 12, tournament 3), random, the random-part search arm, greedy, and LoRA and full fine-tuning stub runners, all through the same env and budget.
  - **LLM agents:** analyst and planners with a repair retry, wired to past.dev recall and Exa citations. The memory listener, verdicts and capability maps feed into memory, and Exa calls are logged as events.
  - **Outputs:** `compare.py`, Charts 1a, 1b and 2, the HTML agent log, `sync_log.py`.
  - **The integrator's 3-seed stub comparison** at 60 FEE [SEARCH][STUB], not results: median picked-best was swarm (scripted) 0.243, evolve 0.276, random 0.227, random_part 0.218, greedy 0.177. Every arm had an equal budget and the ledgers match. The scripted swarm doesn't beat evolution on the stub; the LLM swarm is the real Claim B contender. Chart 2 Spearman 0.016, so the fit score carries no rank signal, as the papers research predicted.
  - **Integrator fixes:** the training-stub seed, a crash on a non-string graft id, sync dedupe dropping a held-out row, and clearer labels.
- **Wave A, done:**
  - **LLM:** clients for mock, OpenAI-compatible, Modal vLLM and Anthropic. `modal_llm.py` serves Qwen2.5-7B-Instruct on an L40S in workspace `rexheng`; the first call took 256 s and a warm JSON call parsed.
  - **past.dev memory:** run-scoped audiences, a cold-start guard and batched ingest. The live smoke passed all 9 checks for 15 credits, with 3.8 s from ingest to recall.
  - **Exa literature:** off, replay and live modes with a two-tier cache.
- **Wave A, still running:** the Modal infra and real compile path (fix pass).
- Stub rows in the root `experiments.jsonl` stay out of git, via local exclude. Only real-run rows will be committed.

## 2026-11-14, step 9: Modal real path, dashboard live, mining stopped, gap fixes, priors pack

- **Modal real path** (wave A, after a review and fix pass): `modal_swarm.py` (app `wsp-swarm`). It runs budget Queue partitions per run, a write-once Dict ledger, run-scoped probe and result caches, and compile, heal and eval in one container on a pinned L4.
  - The review caught one fairness bug: results were cached across runs, with zero compute charged on a hit. It is fixed, and keys are now scoped by run.
  - The live smoke passed 15 of 15 checks. On 20 items with a 50-step heal: host 0.25, graft 0.15, random control 0.35 [SMOKE][SEARCH][INTERIM], which matches Trisanth's "part does not beat random".
- **Dashboard:** deployed to Cloudflare Workers on Rex's personal account at https://wsp-swarm-dashboard.rexheng-policy.workers.dev. Views: the race, agent log, program inspector, fit against score, budget, real runs and parts. Every number is labeled STUB or SEARCH. `bash scripts/deploy_dashboard.sh` refreshes it.
- **Mining:** Rex stopped it to keep part 3 on the environment and swarm testing. The findings are saved in `docs/research/pilot-mining.md`: 8 contrast-selected parts from DeepSeek-R1-Distill-Qwen-1.5B, all in donor layers 20 to 27.
- **Gap fixes:**
  - config defaults for every key;
  - glue-rank sampling in the scripted planner;
  - glue rank in the judge's patterns;
  - a 50-step heal at low fidelity, which cut stub cascade compute per FEE by about 40%.
- **Exa priors pack:** `data/priors.json` (sha 43180a66). 11 live calls cost $0.117, giving 5 cites for each of the 8 pilot parts and 30 topical claims.
- **Upstream:** Trisanth's rounds 3 to 12 show grafts are flat on GSM8K (no part beats random or a rank-1 LoRA), while taps carry donor knowledge on MMLU (Llama-1B 47.5 to 72.9 with a Gemma-9B tap). Asked Rex which setting the first real swarm test should use.
