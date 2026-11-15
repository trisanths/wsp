# Sapien studio: API contract

The Sapien studio is a web app for building models with wsp. A person describes a model in plain words, adjusts a drafted spec, and then either lets an agent swarm build the model or builds it on a drag-and-drop canvas.

This file is the contract between the server (`studio/server/`) and the web app (`studio/web/`). Both sides build against it. If a side needs a change, change this file first.

## Layout

```
studio/
  CONTRACT.md          this file
  README.md            how to run
  server/              FastAPI app (Python), owner: server
    app.py             routes; serves studio/web/ as static files at /
    library.py         reads library/ and atlas/entries/
    sizing.py          size and heal estimates, limit checks
    compare.py         the comparison set and verdict
    ...                any other modules the server needs
  web/                 the web app, no build step, owner: web
    index.html
    app.js             Preact + htm from esm.sh, pinned versions
    styles.css
    mock.js            fixture data for ?mock=1
tests/test_studio_server.py   server tests (CPU, stub backend)
tests/test_studio_sapien.py   library, limits, taps and comparison tests
```

Run: `.venv/bin/uvicorn studio.server.app:app --port 8765`, then open http://localhost:8765.

## Modes

Environment variables pick the pieces. The defaults run on the Mac with no network and no GPU.

| Variable | Default | Values |
|---|---|---|
| `WSP_STUDIO_BACKEND` | `stub` | `stub`, `modal` (the swarm's backends) |
| `WSP_STUDIO_LLM` | `none` | `none`, `anthropic`, `modal_vllm`, `openai_compat` |
| `WSP_STUDIO_PACE_S` | `0.8` | seconds of delay per stub eval, so a stub run is watchable |

When the backend is `stub`, every response carries `"stub": true` and the web app shows a STUB badge on every number. Stub numbers must never look like real results.

When the LLM is `none`, the spec is drafted by rules and the agents are the swarm's scripted agents.

## Shapes

### Spec

```json
{
  "id": "sp_8f2a1c",
  "title": "Laptop math tutor",
  "summary": "Answers grade-school word problems with worked steps. CPU only.",
  "prompt": "the user's text",
  "files": [{"name": "homework.csv", "rows": 120, "score_rows": 40, "heal_rows": 80}],
  "task_type": "math",
  "host": "unsloth/Llama-3.2-1B-Instruct",
  "size": {"max_params_b": 1.5, "min_b": 0.5, "max_b": 8.0, "host_params_b": 1.24},
  "budget": {"evals": 30, "compute_min": 45, "min_evals": 5, "max_evals": 120},
  "benchmarks": [
    {"id": "gsm8k", "name": "GSM8K", "kind": "target", "target": 0.50, "base": 0.23, "lora": 0.467, "recommended": true},
    {"id": "if_guard", "name": "IFEval", "kind": "guard", "floor": 0.40, "base": 0.42, "recommended": true}
  ],
  "available_benchmarks": [{"id": "mmlupro:biology", "name": "MMLU-Pro biology", "kind": "target", "base": 0.107}],
  "ingredients": [{"part_id": "...", "label": "Math reasoning", "donor": "Qwen/Qwen2.5-Math-1.5B", "kind": "block"}],
  "drafted_by": "rules",
  "stub": true
}
```

- `kind` is `target` (has `target`) or `guard` (has `floor`). Scores are fractions in 0..1.
- `base` is the host's score. `lora` is the fine-tuning bar when one is known, else null.
- `available_benchmarks` are the ones the "+ add benchmark" menu offers.

### Program

The program dict is the one in `wsp/swarm/program_schema.py` (`host`, `seed`, `masks`, `task_vectors`, `grafts`, `heal`, and `lora` when the compiler has it). The web app treats it as opaque JSON except for `grafts[].id`, `grafts[].at`, `grafts[].glue_rank`, `heal.steps`, `heal.data`, `masks`, and `lora`.

### Run snapshot

```json
{
  "run_id": "s0-swarm-1794706864",
  "spec": {"...": "the spec"},
  "status": "running",
  "stub": true,
  "backend": "stub",
  "started_at": 1794706864.2,
  "elapsed_s": 312.0,
  "budget": {"evals_used": 18, "evals_total": 30, "compute_used_s": 1620, "compute_total_s": 2700},
  "base_scores": {"gsm8k": 0.23, "if_guard": 0.42},
  "best": {"program_hash": "ab12cd34ef56ab12", "describe": "host + qmath.l8-19@6, heal 300", "scores": {"gsm8k": 0.43, "if_guard": 0.41}, "combined": 0.26, "agent": "planner-2", "program": {}},
  "agents": [
    {"id": "planner-2", "name": "Planner 2", "role": "planner", "blurb": "explores donor blocks",
     "status": "scoring", "now": "Scoring host + qmath.l8-19@6", "best": 0.43, "attempts": 7,
     "collapses": 0, "lessons": 0, "nudges_pending": 0}
  ],
  "leaderboard": [
    {"rank": 1, "program_hash": "ab12cd34ef56ab12", "describe": "...", "agent": "planner-2",
     "scores": {"gsm8k": 0.43, "if_guard": 0.41}, "combined": 0.26, "status": "ok",
     "vs_control": 0.055, "params_b": 1.31, "program": {}}
  ],
  "activity": [{"ts": 1794707100.1, "agent": "analyst", "kind": "analyst_map", "status": "info", "text": "..."}],
  "curve": [{"eval": 1, "best": 0.25, "score": 0.25}],
  "sources": [{"model": "Qwen/Qwen2.5-Math-1.5B", "role": "donor", "uses": 9}]
}
```

- `status`: `starting`, `running`, `paused`, `stopping`, `done`, `error`.
- Agent `status`: `thinking`, `probing`, `compiling`, `scoring`, `idle`, `paused`, `done`.
- Agent ids are `analyst`, `planner-1` .. `planner-N`, `judge`.
- `activity` holds the newest 200 items, newest last. Its `status` is `ok`, `bad` (collapse, error), or `info`. Each item is `{"id", "ts", "agent", "kind", "status", "text", "data", "round"}`. `id` is a stable increasing number. `text` is one plain sentence for people. `round` is the swarm round (may be null). `kind` and `data`:
  - `eval`: `{"name", "program_hash", "target", "best", "is_best", "guards": [{"id", "name", "score", "floor", "ok"}], "scores", "result"}`. `name` is the readable program name, `target` the mean target score, `result` the env status (`ok`, `collapsed`, `rejected_fit`, `compile_error`, `invalid`, `infra_error`).
  - `probe`: `{"name", "fits": [numbers], "program_hash"}`. Fit checks; frequent, so pages may fold them.
  - `plan`: `{"count"}`. A planner proposed programs.
  - `map`: null, or `{"detail"}` with the analyst's raw table.
  - `verdict`: `{"frontier": [{"program_hash", "name", "target"}], "retired", "killed"}`.
  - `heldout`: `{"name", "program_hash", "target", "scores"}`.
  - `milestone`: null. Search finished, stalled, or stopped.
  - `nudge`: agent is `"you"`, data `{"to", "to_name"}`, text is the nudge.
  - `nudge_applied`: `{"nudge"}`.
  - `error`, `system`, `handoff`: null.
- Agent `now` and leaderboard and agent history use readable program names where they describe a program (for example "Math block L8-19 at L6, heal 300").
- `curve` has one point per scored eval in order: `best` is the best target score so far, `score` is that eval's target score.
- `vs_control` is the target score minus its random-part control, or null when no control ran.

### Agent detail

```json
{
  "agent": {"...": "the agent object from the snapshot"},
  "current": {"program": {}, "text": "program in wsp text syntax", "describe": "..."},
  "thinking": [{"ts": 1794707100.1, "text": "hypothesis or reasoning"}],
  "tools": [{"ts": 1794707100.1, "call": "probe(qmath.l8-19@6)", "result": "fit 0.71", "status": "ok"}],
  "history": [{"program_hash": "...", "describe": "...", "scores": {}, "combined": 0.2, "status": "ok"}],
  "nudges": [{"ts": 1794707100.1, "text": "try deeper layers", "applied": false}]
}
```

`current` is null when the agent has no program in flight.

### Catalog (builder palette)

```json
{
  "stub": true,
  "host": {"model": "unsloth/Llama-3.2-1B-Instruct", "layers": 16, "params_b": 1.24},
  "parts": [
    {"id": "...", "group": "add", "kind": "block", "label": "Math block",
     "donor": "Qwen/Qwen2.5-Math-1.5B", "layers": [8, 19], "params_m": 570, "default_at": 6,
     "summary": "Qwen2.5-Math layers 8-19"}
  ],
  "masks": [{"id": "mask_mmlupro_chemistry_llama1b", "group": "remove", "label": "Chemistry", "summary": "deletion mask"}],
  "train": [
    {"id": "lora", "group": "train", "label": "LoRA adapter", "summary": "rank 1 to 64"},
    {"id": "heal", "group": "train", "label": "Heal", "summary": "trains glue only"}
  ]
}
```

### Builder job result

```json
{
  "job_id": "j_19ab",
  "action": "score",
  "status": "done",
  "stub": true,
  "result": {
    "fit": {"<part id>@6": 0.71},
    "scores": {"gsm8k": 0.43, "if_guard": 0.41},
    "combined": 0.26,
    "control": {"scores": {"gsm8k": 0.375}},
    "params_b": 1.31,
    "compute_s": 210,
    "status": "ok"
  },
  "budget": {"evals_used": 3, "evals_total": 12, "compute_used_s": 840, "compute_total_s": 2700},
  "error": null
}
```

`status` is `queued`, `running`, `done`, or `error`. For `compile`, only `fit` and `params_b` are filled.

## Routes

All bodies are JSON unless marked.

| Method | Path | Body | Returns |
|---|---|---|---|
| GET | `/api/health` | | `{"ok": true, "backend": "stub", "llm": "none", "stub": true}` |
| POST | `/api/spec` | multipart: `prompt` (text), `files` (0..n files) | Spec |
| GET | `/api/spec/{id}` | | Spec |
| PUT | `/api/spec/{id}` | Spec (edited) | Spec |
| POST | `/api/runs` | `{"spec_id": "...", "program": null}` | `{"run_id": "..."}` |
| GET | `/api/runs/{id}` | | Run snapshot |
| GET | `/api/runs/{id}/stream` | | Server-sent events. Each event is `event: snapshot` with the run snapshot as data, sent on every change (at most every 0.5 s) and every 5 s as a heartbeat. |
| POST | `/api/runs/{id}/pause` | | `{"ok": true}` |
| POST | `/api/runs/{id}/resume` | | `{"ok": true}` |
| POST | `/api/runs/{id}/stop` | | `{"ok": true}` (stops the search; the best program is kept) |
| GET | `/api/runs/{id}/agents/{agent_id}` | | Agent detail |
| POST | `/api/runs/{id}/agents/{agent_id}/nudge` | `{"text": "..."}` | `{"ok": true}` |
| GET | `/api/catalog` | | Catalog |
| POST | `/api/program/text` | `{"program": {}}` | `{"text": "..."}` |
| POST | `/api/program/parse` | `{"text": "..."}` | `{"program": {}}` or 400 `{"error": "...", "line": 3}` |
| POST | `/api/program/validate` | `{"program": {}}` | `{"errors": []}` |
| POST | `/api/builder` | `{"spec_id": "..."}` | `{"builder_id": "...", "budget": {...}}` (one budget per builder session) |
| POST | `/api/builder/{id}/action` | `{"program": {}, "action": "compile" \| "heal" \| "train" \| "score"}` | `{"job_id": "..."}` |
| GET | `/api/jobs/{job_id}` | | Builder job result |
| POST | `/api/builder/{id}/size` | `{"program": {}}` | size estimate and limit problems (see Sapien changes) |
| POST | `/api/builder/{id}/suggest` | `{"program": {}}` | `{"suggestions": [{"title": "...", "detail": "...", "tag": "likely +1 to 2", "program": {} }]}` |
| POST | `/api/builder/{id}/handoff` | `{"program": {}}` | `{"run_id": "..."}` (starts a swarm run seeded with this program) |

Errors are `{"error": "message"}` with a 4xx or 5xx status.

## Builder actions

- `compile`: a fit-only probe (`WspEnv.probe`). Uses compute, no eval.
- `heal`: compile with the program's heal pass (adds a default heal when the program has none), then score.
- `train`: add a LoRA to the program (rank 16 when the program has none), heal, then score.
- `score`: compile and score the program as written, and run its random-part control when it has grafts.

## Sapien changes

The product is Sapien. A program builds one network that runs alone. Taps and anything that runs a second model at inference are not part of it: the palette has no taps, the agents never propose them, and the server refuses a program with `taps` or `tap_train` (400, "A tap runs a second model at inference...").

### Limits

The spec has `limits`: `{"max_total_params_b": 1.5, "max_run_params_b": 1.5, "max_heal_min": 20}`. `size.max_params_b` mirrors `max_total_params_b` for older pages. `size.max_b` is 32. The draft reads the limits from the request ("under 3B", "1.2B per token", "heal for 30 minutes"). Max run per token is at most the total and at least the host size.

The server estimates a program (`server/sizing.py`): total parameters, parameters run per token, and heal minutes. The estimate is rough and says so (`"estimated": true`). `POST /api/builder/{id}/size` with `{"program": {}}` returns `{"estimate": {...}, "limits": {...}, "problems": [], "ok": true}`. A builder action, a handoff and a run with a seed program are refused with 400 and a plain message when over a limit. In a run, a proposal over a limit is dropped before it is scored; the snapshot counts them in `refused` and the activity has `refused` items. Growing the base is allowed while the limits hold.

### Library palette

`GET /api/catalog` returns `parts` from the compiler catalog and from `library/index.jsonl` (read when the server starts) and `atlas/entries/*.json` when that folder exists. Every part has `kind` (`block`, `attn_heads`, `mlp_cluster`, `expert`, `inserted_layer`), `status` (`candidate`, `validated`, `rejected`), `evidence` (one line), `source` (`stub`, `catalog`, `library`, `atlas`), `usable` and `usable_reason`. A part is usable when the compiler catalog has a record for it. Tap read points from the library are listed as not usable. The response also has `kinds` (name and count per kind), `library` (counts) and `product`. Masks have `status` and `evidence` too.

### Comparison set

The spec has `comparators`: `{"distill": {"gsm8k": 0.40}}`, scores a person measured for base + distillation at the same budget. A run snapshot has `comparison` (for the best build) and each leaderboard entry has one. A builder score result has `comparison` and `size`.

```json
{"verdict": "failed", "headline": "Failed attempt. It does not beat ...", "margin": 0.02,
 "result": {"label": "This build", "value": 0.393, "per_task": {}, "params_b": 1.3, "heldout": null},
 "rows": [{"id": "base", "label": "Base", "split": "search", "per_task": {}, "value": 0.25, "measured": true,
           "note": "...", "stub": true, "result_value": 0.393, "delta": 0.143, "beaten": true}],
 "beaten": [], "not_beaten": [], "unmeasured": [], "best_model_shown": false, "stub": true}
```

Row ids: `base`, `lora`, `distill`, `random`, and `best_model` (only when the total limit is at least twice the host size; it is the best library model within the limit). `verdict` is `failed` when a measured reference is not beaten by `margin` (a collapsed build or one over a limit also fails), `unproven` when nothing failed but a reference is not measured, `beats` when all are measured and beaten, `none` with no score. On real backends `distill` is not measured until the spec has it. On the stub backend `distill` is made up and its row says so (`stub: true`).

`/api/health` also returns `"product": "Sapien"`.
