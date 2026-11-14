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
