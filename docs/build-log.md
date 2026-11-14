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
