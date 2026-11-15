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
