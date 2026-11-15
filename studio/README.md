# Sapien studio

The Sapien studio is a web app on top of the wsp compiler and its agent swarm. A person describes a model, adjusts a drafted spec, and then lets the swarm build it or builds it on a canvas. `studio/CONTRACT.md` lists every route and JSON shape.

## Run

Install the extras once, then start the server from the repo root:

    pip install -e ".[dev,swarm,studio]"
    .venv/bin/uvicorn studio.server.app:app --port 8765

Open http://localhost:8765. The server serves `studio/web/` at `/` and the API under `/api`.

## Modes

| Variable | Default | Values |
|---|---|---|
| `WSP_STUDIO_BACKEND` | `stub` | `stub`, `local`, `modal` |
| `WSP_STUDIO_PIPELINE` | `synth` | `synth` compiles programs with grafts or inserted layers with `wsp/synth.py`; `classic` uses the older compile |
| `WSP_STUDIO_LLM` | `auto` | `auto`, `none`, `openai_compat`, `anthropic`, `modal_vllm` |
| `WSP_STUDIO_OPENAI_MODEL` | `gpt-5.4-mini` | any OpenAI chat model |
| `EXA_API_KEY` | unset | enables Exa web search for benchmark research (read by `exa_py`, never printed) |
| `WSP_STUDIO_PACE_S` | `0.8` | seconds of delay after each stub eval |
| `WSP_STUDIO_OUT` | `outputs/studio` | where specs, uploads and run files go |

The server reads `.env` the same way `wsp/swarm/config.py` does. `auto` uses OpenAI when `OPENAI_API_KEY` is set, Anthropic when `ANTHROPIC_API_KEY` is set, and no LLM otherwise. With no LLM, specs are drafted by rules and the agents are the scripted ones. With an LLM, the spec draft, the agents and one builder suggestion use it, and a failed LLM call falls back to the rules. `server/llm_openai.py` sends `max_completion_tokens` and the default temperature, which the GPT-5 and o-series models need.

Benchmark research starts when a spec is drafted. It picks a search provider in this order: Exa when `EXA_API_KEY` is set and `exa_py` imports, the OpenAI `web_search` tool when only `OPENAI_API_KEY` is set (it reads many pages, so it uses more tokens than Exa), and no search otherwise. With `WSP_STUDIO_LLM=none` the research is skipped. `POST /api/spec/{id}/research` starts it again. The OpenAI web search is billed per search call plus tokens.

The default `stub` backend is the offline backend. It uses the real compiler catalog, the real library parts and the real deletion masks, and scores them with a simulator in `server/offline_backend.py` (no GPU, no Modal). A part's simulated effect comes from its library record: skill tags matched to the task, status (a rejected part is slightly harmful), its measured score against control when the library has one, and its donor's layer count and hidden size for fit and size. It is deterministic per seed, and a random-part control scores lower than a matching real part on average. Every response carries `"stub": true`, and the web app shows an "Estimate" badge. The parts are real; only the scores are estimates.

## Demo mode

Demo mode runs the web app on sample runs, specs and scores in the browser, so a demo needs no GPU. The parts library is the real one: in demo mode the page reads `GET /api/catalog` and the library lists from the server it was loaded from and treats every part that is not rejected as placeable, as on the GPU backend. The demo programs use real library parts. On a plain static host with no API, it falls back to the small mock list. Turn it on with the Demo mode switch at the bottom of the sidebar, with `?demo=1` in the URL, or with Ctrl+Shift+D (Cmd+Shift+D on a Mac). `?demo=0` or the same switch turns it off. Demo runs, specs and recents use their own browser storage keys, so they never mix with real ones.

`#/demo` starts the guided tour: 33 steps in 8 chapters (ask for a model, the spec, agents build it, autoresearch, mine a model, build it yourself, shrink a model, and the library). Each step opens a screen and puts a white spotlight on one element, with a caption card beside it; the spotlight glides from element to element. On steps that show a click, an animated pointer taps the element, and clicking it moves the tour on (its own action is held back so the tour stays on its sample data, except tabs that switch the view in place). Look steps move on with a click in the highlight, Next, or the right arrow. "Play automatically" (or Space, or the play button) lets the tour run itself: it waits on each step for a time set by the caption, taps the click steps, and stops at the end; a thin blue line in the caption card counts down to the next step. Left arrow goes back, Escape leaves the tour, and clicks outside the highlight are held. The control at the bottom right shows the step, jumps between chapters, resumes a left tour, sets the live run speed (1x, 2x, 4x) and the theme, and hides itself (Alt+H) for a clean screen share. The steps are the `STEPS` list in `web/screens/demo.js` (screen hash, a CSS selector with an optional text match, and the caption).

In demo mode one "Demo" pill in the top bar ("Demo mode: real parts library, sample runs") takes the place of the Estimate badges, so sample numbers are never shown as measured. The measured results card keeps its real numbers. `?mock=1` without demo mode is unchanged: same mock data, with the Estimate labels. Files: `web/demo-flag.js`, `web/demo-data.js`, `web/screens/demo.js`, `web/demo.css`; the seeding is in `web/mock.js`.

## Hosted site

The hosted site (sapien.trisanth.com, on Vercel) is the web app in demo mode with no server. The mock API runs in the browser, and the parts library is a snapshot of the real library taken when the site is built. No keys, backend or GPU go into it. Spec drafting with a live model, benchmark research and real agent runs need the Python server, so on the hosted site they play from sample data.

    sh studio/hosted/deploy.sh

builds `studio/hosted/dist` (`build_site.py`) and deploys it to the Vercel project `sapien`. The build sets `window.SAPIEN_HOSTED`, which keeps demo mode on and hides the demo switch.

## Real GPUs (Modal)

    MODAL_PROFILE=trisanth .venv/bin/modal deploy studio/modal_studio.py
    MODAL_PROFILE=trisanth WSP_STUDIO_BACKEND=modal .venv/bin/uvicorn studio.server.app:app --port 8771

`studio/modal_studio.py` is the `wsp-studio` Modal app. Each call compiles one program, trains its taps, then its LoRA or heal, and scores it in one GPU container (L40S, or H100 when a donor is above about 10B parameters). A program with grafts or inserted layers goes through `synth_remote`, which compiles with `wsp/synth.py`, scores the one saved network, and returns a verdict per component (contributes, no better than random, or harmful), stage summaries, drift from the host and measured sizes. The older path stays behind `WSP_STUDIO_PIPELINE=classic`. Redeploy after a change to `wsp/`, `wsp_eval/` or `parts_extracted/`, since they are copied into the image.

- MMLU is scored as in rounds 6 and 7: the log-prob of " A" to " D", on the round 6 dev items (search, 1000) and the round 7 held-out items (2000). Other tasks use the eval harness.
- Taps are not in Sapien. The Modal functions refuse a program with taps.

## Shrink (skill removal)

A request can also ask for a smaller model that keeps one skill ("a coding-only model: drop everything else from Qwen2.5-Coder-7B and get it under 3B"). The spec then has `request_type: "shrink"` and a `shrink` plan: source model, skill to keep, method (expert removal for MoE models, layers and width for dense ones), size target, rule and heal steps. The plan is estimated from library facts without loading the model (`POST /api/shrink`, `POST /api/shrink/{id}/plan`), and `POST /api/shrink/{id}/run` runs it. `wsp/slim.py` does the removal and the distillation heal. Selection and heal text come from the `wsp/healdata.py` pools, never a benchmark.

Backends: `stub` returns an estimated result (marked as an estimate). `modal` calls `shrink_remote` in the `wsp-studio` app (L40S, or H100 above 10B parameters). The result is compared with the original (reference only), the same slim with random units removed (must be beaten), and the best library model of about the same size. See `studio/CONTRACT.md` for the shapes. A cheap check on a real GPU: shrink `Qwen/Qwen2.5-1.5B-Instruct` to 1.2B with 100 heal steps, task `gsm8k`, fidelity `check`, control on.

## Files

- `server/app.py`: routes, the SSE stream and the static web app.
- `server/bench_research.py`: the benchmark research agent (plan, search, synthesize, validate).
- `server/specs.py`: spec drafting, upload row counts, storage.
- `server/shrink.py`, `shrink_spec.py`, `shrink_models.py`: the shrink request type (sessions and jobs, drafted plan and edit checks, model facts and source choice). `compare.build_shrink` and `sizing.estimate_shrink` hold its comparison and estimate.
- `server/runs.py`: starts a swarm run in a daemon thread.
- `server/tracking.py`: pause and stop, agent proxies and the swarm subclass.
- `server/runstate.py`: run state and the snapshot.
- `server/builder.py`: builder sessions, jobs and suggestions.
- `server/catalog.py`: the builder palette (compiler catalog, library and atlas parts with status and evidence).
- `server/library.py`: reads `library/index.jsonl` and `atlas/entries/*.json` once at server start.
- `server/sizing.py`: size and heal estimates and the limit checks.
- `server/compare.py`: the comparison set (base, LoRA, distillation, random, best existing model) and the verdict.
- `server/envs.py`: builds a run's env, with the Modal backend when it is chosen.
- `server/modal_backend.py`: the backend that calls the `wsp-studio` Modal app.
- `server/offline_backend.py`: the offline backend (`stub`): real parts and masks scored by a simulator.
- `server/synth_report.py`: which pipeline a program uses, what the synthesis pipeline refuses, the stub verdicts and the shape of component results.
- `server/llm_openai.py`: the OpenAI client for GPT-5 models.
- `server/programs.py`: program text render, parse and validate.

Run files go to `outputs/studio/`: `specs/<id>.json`, `uploads/<spec_id>/`, `runs/<run_id>/` (the swarm's log plus `studio_snapshot.json`).

## Notes

- In stub mode the compute cap is at least 450 seconds per eval, because the stub cost model charges 300 to 450 seconds per eval and a 45 minute cap would otherwise end a run after about 9 evals.
- The stub backend runs against the same catalog as the Modal backend (`catalog_offline.jsonl`: `catalog.jsonl` plus library records), so the placeable count is the same on both. A real mask costs a little on guard tasks in the simulator and is noise on target tasks.
- Offline base scores come from the stub world (GSM8K 0.25, IFEval 0.70), and a stub spec shows those. On Modal the spec shows the measured Llama-1B bases (GSM8K 0.23, IFEval 0.42, MMLU 0.475).

## Tests

    .venv/bin/pytest tests/test_studio_server.py tests/test_studio_sapien.py -q

The server tests need fastapi, httpx and python-multipart, which the repo venv may lack.
