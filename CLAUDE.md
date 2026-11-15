# Working in this repo

## Shared team context (past.dev)

The team and every coding agent share one memory on past.dev, in the audience `wsp-mining` (client: `person_a/mine/shared_memory.py`). Use it all the time, not once:

- Before you start a task, recall what the team already knows about it:

      .venv/bin/python scripts/team_memory.py recall "<the topic of your task>"

- After every decision, build, or result, record it: one or two plain sentences that stand alone, with the number, the split, the control, and the date.

      .venv/bin/python scripts/team_memory.py record finding "Tap Gemma-2-9B L29 into Llama-1B L10 scored 72.9 on MMLU dev (2 seeds), 2026-11-14." --who <your name> --topic taps

  Kinds: `finding` (a measured result), `lesson` (a takeaway), `plan` (what is running or decided), `note`.
- Never send team notes any other way. The client refuses anything outside `wsp-mining`; swarm benchmark runs use their own run audiences and must not read team notes (docs/research/pastdev.md, section 6.1).

The key is `PAST_API_KEY` in `.env`. Do not print it.

## Where things are

- Compiler and language: `wsp/`. Eval harness: `wsp_eval/`. Swarm: `wsp/swarm/`. Results log: `results/README.md`.
- The studio web app: `studio/` (README, `CONTRACT.md` for the API, `web/STYLE.md` for the look).
- Parts library: `library/`.
