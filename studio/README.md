# wsp studio

wsp studio is a web app on top of the wsp compiler and its agent swarm. A person describes a model, adjusts a drafted spec, and then lets the swarm build it or builds it on a canvas. `studio/CONTRACT.md` lists every route and JSON shape.

## Run

Install the extras once, then start the server from the repo root:

    pip install -e ".[dev,swarm,studio]"
    .venv/bin/uvicorn studio.server.app:app --port 8765

Open http://localhost:8765. The server serves `studio/web/` at `/` and the API under `/api`.

## Modes

| Variable | Default | Values |
|---|---|---|
| `WSP_STUDIO_BACKEND` | `stub` | `stub`, `local`, `modal` |
| `WSP_STUDIO_LLM` | `none` | `none`, `anthropic`, `modal_vllm`, `openai_compat` |
| `WSP_STUDIO_PACE_S` | `0.8` | seconds of delay after each stub eval |
| `WSP_STUDIO_OUT` | `outputs/studio` | where specs, uploads and run files go |

The server reads `.env` the same way `wsp/swarm/config.py` does. Every response from the stub backend carries `"stub": true`. The `modal` backend is selectable here and is not run in tests.

With the LLM set to `none`, specs are drafted by rules and the agents are the scripted ones. With an LLM, the spec draft and the agents use it, and a failed LLM call falls back to the rules.

## Files

- `server/app.py`: routes, the SSE stream and the static web app.
- `server/specs.py`: spec drafting, upload row counts, storage.
- `server/runs.py`: starts a swarm run in a daemon thread.
- `server/tracking.py`: pause and stop, agent proxies and the swarm subclass.
- `server/runstate.py`: run state and the snapshot.
- `server/builder.py`: builder sessions, jobs and suggestions.
- `server/catalog.py`: the builder palette.
- `server/programs.py`: program text render, parse and validate.

Run files go to `outputs/studio/`: `specs/<id>.json`, `uploads/<spec_id>/`, `runs/<run_id>/` (the swarm's log plus `studio_snapshot.json`).

## Notes

- In stub mode the compute cap is at least 450 seconds per eval, because the stub cost model charges 300 to 450 seconds per eval and a 45 minute cap would otherwise end a run after about 9 evals.
- The stub catalog has no deletion masks, so the studio adds the real masks from `catalog.jsonl` to its stub palette. The stub ignores masks when it scores.
- Stub base scores come from the stub world (GSM8K 0.25, IFEval 0.70). The drafted spec shows the measured Llama-1B search-split bases (0.23, 0.42), and the run snapshot shows the base scores the env uses.

## Tests

    .venv/bin/pytest tests/test_studio_server.py -q
