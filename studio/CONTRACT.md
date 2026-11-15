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
    compare.py         the comparison set and verdict (build and shrink)
    shrink.py          shrink sessions, jobs, the stub result
    shrink_spec.py     shrink detection, the drafted plan, checks for an edited plan
    shrink_models.py   model facts and the choice of a source model
    ...                any other modules the server needs
  web/                 the web app, no build step, owner: web
    index.html
    app.js             Preact + htm from esm.sh, pinned versions
    styles.css
    mock.js            fixture data for ?mock=1
tests/test_studio_server.py   server tests (CPU, stub backend)
tests/test_studio_sapien.py   library, limits, taps and comparison tests
tests/test_studio_shrink.py   shrink request type tests (stub backend, fake Modal function)
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
  "request_type": "build",
  "stub": true
}
```

- `request_type` is `"build"` (add components to a host, the default; every older spec reads as a build) or `"shrink"` (remove skills from a model, see Shrink). A shrink spec also has `shrink`, and its `host` is the source model.

- `kind` is `target` (has `target`) or `guard` (has `floor`). Scores are fractions in 0..1.
- `base` is the host's score. `lora` is the fine-tuning bar when one is known, else null.
- `available_benchmarks` are the ones the "+ add benchmark" menu offers.

### Benchmark research

When a spec is drafted (`POST /api/spec`) the server starts a research run in a background thread. It reads the request, searches the web for benchmarks and reference scores, and proposes benchmarks. It never picks components or skills, and never changes limits, size or title. The run is saved on the spec as `spec.benchmark_research`, so the spec page can poll `GET /api/spec/{id}/research` (or the spec) and show the steps as they arrive.

```json
{
  "status": "running | done | error | skipped",
  "provider": "exa | openai_web | none",
  "steps": [
    {"kind": "think", "text": "I will look for legal QA benchmarks...", "t": 1.6},
    {"kind": "search", "text": "Searched: legal question answering benchmark", "provider": "exa", "t": 2.8},
    {"kind": "read", "text": "Legal LLM Leaderboard", "url": "https://...", "provider": "exa", "t": 2.8},
    {"kind": "pick", "text": "Picked MMLU-Pro law: closest runnable exam", "t": 8.1}
  ],
  "reasoning": "plain paragraphs on which benchmarks fit and why",
  "proposals": [
    {"name": "MMLU-Pro law", "measures": "...", "why": "...", "runnable_id": "mmlupro:law",
     "kind": "target", "suggested_target": 0.45, "in_spec": true,
     "reference_scores": [{"model": "Qwen3-1.7B", "params_b": 1.7, "score": 0.7,
                           "source_url": "https://...", "sourced": true}]}
  ],
  "sources": [{"title": "...", "url": "https://..."}],
  "started_at": 1763000000.0, "finished_at": 1763000009.0, "error": null,
  "usage": {"input_tokens": 0, "output_tokens": 0, "web_searches": 0}
}
```

- A reference score is `sourced: true` only when all of these hold. Its `source_url` is a page the run read. The agent's `quote` is on that page (a quote may skip up to three cells or words). The quote shows the number, as a percentage or a fraction. The model is named before the number in the quote, or heads the number's table column, or the page is that model's own card. The words around the number (its column header, its row, the table heading, the page title, the words just before it, and the words above the table when the header names no benchmark) name the benchmark. No change word (SFT, GRPO, LoRA, quantized, a plus sign and a method) sits between the model and the number or in the column header. In running text no other score sits between the model and the number. `quote` is the page's own text. Anything else is `sourced: false`, an estimate. A score that fails one of these checks on a page the run read is not kept at all, and is counted in `score_checks`. A score with no quote, or a quote that is not on the page, stays as an estimate. The link is there for the reader to confirm.
- Size rule: a model whose `params_b` (or the size in its name) is more than 1.5 times the size limit is dropped, and so is a closed or very large model with no size (GPT, Claude, Gemini, DeepSeek R1 and V3, Kimi, Mixtral). A model that the request names (a shrink request names its source model) is always kept.
- Near-identical models are kept once. A bracketed note and punctuation do not make a new model. The sourced score from the better source wins (paper, then model card or leaderboard, then repo, then other pages, then aggregators).
- `step.kind` is `think`, `search`, `read` or `pick`. `t` is seconds since the run started. `provider` is on search and read steps. `url` is on read steps.
- `provider` is `exa` when `EXA_API_KEY` is set, else `openai_web` when `OPENAI_API_KEY` is set, else `none` (the model's own knowledge only).
- `status` is `skipped` when `WSP_STUDIO_LLM` is `none` (the reason is in `error`; the rule-drafted benchmarks stay). A run that takes more than about 90 seconds ends as `error` with a plain message and keeps its steps.
- `runnable_id` is an id Sapien can run, else null (a reference-only benchmark that stays in `proposals`). `in_spec` is true when the proposal is now in `spec.benchmarks`.
- `reference_scores[].score` is a fraction in 0..1. `sourced` is true only under the rules above. When it is false the page must label the score as an estimate.
- New optional fields, all safe to ignore. On each `reference_scores` entry: `metric` (the metric the page names, such as "accuracy", "pass@1", "BLEU", else null; a BLEU or chrF value is the page number divided by 100), `source_kind` (`paper`, `model card`, `leaderboard`, `repo`, `aggregator` or `other`, from the url, null with no url), `label` (words the page uses for the benchmark, only when they are on the page) and `comparable` (false when the metric differs from the proposal's main metric, so the page should not set the number beside the others; absent otherwise). On each proposal: `metric` (the main metric of its sourced scores, absent with none), `confidence` (`high`, `medium` or `low`) and `confidence_note` (one plain sentence, for example "3 sourced scores for models near 1.5B, from paper, model card."). Low means no score was tied to a page line, so `suggested_target` is a guess. A guard with no lookup is medium. On the run: `score_checks` is `{"dropped": {"too_large": 2, "variant": 1}}`, the number of scores the checks threw away for each reason (`too_large`, `closed_or_huge_model`, `method_variant`, `variant`, `model_not_before_number`, `other_number_between`, `benchmark_not_near_number`). The list can grow. Scores are ordered with sourced ones first, then by source kind.
- When a run finishes, each proposal with a `runnable_id` is added to `spec.benchmarks` (at most 4 targets in total). A benchmark entry made by research has `"research": true` and `"reference_scores": [...]` next to its normal fields. Benchmarks already on the spec stay unchanged, and a matching proposal only adds `reference_scores` and `research` to them. `PUT /api/spec/{id}` keeps both fields.

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
  "stopped_early": false,
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
- `curve` has one point per scored eval in order: `best` is the best target score so far among valid (ok) programs, `score` is that eval's target score. A collapsed program adds a point but never raises `best`. `best` is `null` until the first valid program is scored.
- `base_scores` is read once when the run thread starts. It is `{}` until then (treat it as pending).
- `stopped_early` is true when the person pressed Stop. The run still ends with status `done`. The held-out check and its controls are skipped, and the activity says so.
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
| GET | `/api/health` | | `{"ok": true, "product": "Sapien", "backend": "stub", "llm": "none", "llm_model": null, "stub": true, "heal_note": null}`. `heal_note` is a warning string on real backends while the heal recipe is being fixed; show it next to any result that used a heal (`heal`, `train`, or a program with a heal or LoRA). |
| POST | `/api/spec` | multipart: `prompt` (text), `files` (0..n files) | Spec |
| GET | `/api/spec/{id}` | | Spec |
| PUT | `/api/spec/{id}` | Spec (edited) | Spec |
| POST | `/api/spec/{id}/research` | | `{"status": "running"}` (or `"skipped"` with no LLM); 409 when a run is already going |
| GET | `/api/spec/{id}/research` | | `benchmark_research` (see Benchmark research), 404 when the spec has none |
| POST | `/api/runs` | `{"spec_id": "...", "program": null}` | `{"run_id": "..."}` |
| GET | `/api/runs/{id}` | | Run snapshot |
| GET | `/api/runs/{id}/stream` | | Server-sent events. Each event is `event: snapshot` with the run snapshot as data, sent on every change (at most every 0.5 s) and every 5 s as a heartbeat. If the server cannot build a snapshot it sends `event: error` with `{"error": "..."}` and the stream goes on. |
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


## Shrink (skill removal)

The second request type. A person asks for a smaller model that keeps one skill: "a coding-only model: drop everything else from Qwen2.5-Coder-7B and get it under 3B", "keep only the math experts of a 30B MoE". `wsp/slim.py` removes experts (MoE), or layers, neurons and heads (dense), by how much the skill text uses them, and heals with distillation toward the original model's logits. The result is one network that runs alone. Selection and heal use the generic text pools in `wsp/healdata.py`, never a benchmark; benchmarks only score the finished model.

### Spec of a shrink

`POST /api/spec` drafts a shrink when the words ask for one ("shrink", "slim", "smaller version of", "drop everything", "keep only", "remove ... skills", "distill down", "coding-only") or, with an LLM, when the LLM says so (its `request_type` answer wins; it may also name `source_model` and `keep_skill`). A request that names a library model uses it. Otherwise the source is the smallest supported library model that has the skill and is larger than the target. The spec has the usual fields, with these differences: `host` is the source model, `size.max_params_b` is the total limit, `size.host_params_b` is the source size, `benchmarks` are the targets that score the skill (code: `humaneval`, math: `gsm8k`, medicine: `mmlupro:health`, general and prose: `mmlu`) and there is no guard, `ingredients` is empty. `limits` are as for a build (`max_total_params_b`, `max_run_params_b`, `max_heal_min`) but may be far below the source size. When the request states no size, the total and per-token limits are the default plan's estimate rounded up to 0.1B.

```json
{
  "request_type": "shrink",
  "shrink": {
    "source_model": "Qwen/Qwen2.5-Coder-7B-Instruct",
    "source_params_b": 7.616,
    "source_active_params_b": 7.071,
    "family": "dense",
    "keep_skill": "code",
    "method": "dense",
    "target_params_b": 2.91,
    "keep_experts": null,
    "rule": "skill",
    "heal_steps": 300,
    "candidates": [{"model": "Qwen/Qwen3-30B-A3B-Instruct-2507", "params_b": 30.53, "why": "tagged for code, MoE with 128 experts per layer"}],
    "notes": []
  }
}
```

- `family` is `moe` or `dense`. `method` is `experts` (MoE) or `dense`.
- `keep_skill` is one of the `wsp/healdata.py` pool names: `math`, `code`, `medicine`, `general`, `prose`.
- `rule` for experts: `freq`, `saliency` (default), `selective`, `random`, `general`. For dense: `skill` (default, same as `skill_cost`), `specific`, `general`, `random`.
- MoE: `keep_experts` is N per layer, `target_params_b` is the estimated total after keeping N (informational). Dense: `target_params_b` is the wanted total, `keep_experts` is null.
- `source_active_params_b` is the estimated parameters the source runs per token. `candidates` holds up to 5 other sources that fit, best first. `notes` holds plain sentences (for example a named model that cannot be slimmed).
- `PUT /api/spec/{id}` edits the block. Send `shrink` with the fields to change (or the whole block). Checks, each a 400 with a plain message: the model must be in the library (or one of the studio's extra Qwen2.5-Coder models) and its architecture must be supported; `keep_skill` must be a pool name; `rule` must fit the method; `keep_experts` must be 1..experts; dense `target_params_b` must be above the embedding and head size and below the source size; `heal_steps` is 0..1000; `method` cannot disagree with the model. Changing `source_model` resets the other fields to that model's defaults unless the edit names them. Editing an MoE `target_params_b` (without `keep_experts`) recomputes `keep_experts`. `limits` and `size.max_params_b` edit as for a build, with no lower bound at the source size.
- A shrink spec is refused by `POST /api/runs` and `POST /api/builder` (400). A build spec is refused by `POST /api/shrink`.

### Routes

| Method | Path | Body | Returns |
|---|---|---|---|
| POST | `/api/shrink` | `{"spec_id": "..."}` | `{"shrink_id": "sh_1a2b3c4d", "plan": Plan, "stub": true}` |
| POST | `/api/shrink/{id}/plan` | edited fields of the shrink block | Plan (free, no GPU, no model load) |
| POST | `/api/shrink/{id}/run` | optional `{"tasks": [], "fidelity": "low", "control": true, "general_control": false, "score_source": null}` | `{"job_id": "j_..."}` |
| GET | `/api/jobs/{job_id}` | | job (the builder's job shape; `action` is `"shrink"`, `budget` is null) |

```json
{
  "source_model": "Qwen/Qwen3-30B-A3B-Instruct-2507", "source_params_b": 30.53, "source_active_params_b": 3.029,
  "family": "moe", "keep_skill": "math", "method": "experts", "target_params_b": 8.778, "keep_experts": 32,
  "rule": "saliency", "heal_steps": 300, "candidates": [], "notes": [],
  "estimate": {"method": "experts", "total_params_b": 8.778, "run_params_b": 3.029, "heal_min": 4.54,
               "source_params_b": 30.53, "source_run_params_b": 3.029, "removed_params_b": 21.752,
               "experts_kept": 32, "experts_before": 128, "layers": 48, "top_k_after": 8, "estimated": true},
  "limits": {"max_total_params_b": 8.8, "max_run_params_b": 3.1, "max_heal_min": 20},
  "problems": [], "ok": true,
  "summary": "Keep 32 of 128 experts per layer by saliency on math text. About 8.8B stored, 3.0B run per token (estimated).",
  "tasks": ["gsm8k"], "stub": true
}
```

That is a Plan. A dense estimate has `layers_dropped`, `layers_before` and `keep_width` (fraction of MLP neurons and heads kept) instead of the expert fields. The estimate uses library facts only (parameters, active parameters, layers, experts, experts per token) and never loads the model, so it is always labeled `"estimated": true`. Heal minutes use the build rule (0.3 s per step per billion parameters run, at least 0.5B). `POST /api/shrink` and `/plan` never fail on a limit: they return `ok: false` and `problems` with the numbers, so the page can show them. `POST .../run` refuses an over-limit plan with 400 and the same messages ("Over the total size limit. The shrunk model is estimated at 8.778B parameters (the source has 30.53B). The limit is 5B."). A bad edit on `/plan` is a 400.

`run` options: `tasks` (default: the spec's targets), `fidelity` (`check`, `low` default, `full`), `control` (random control with the same counts and heal steps, default true), `general_control` (the optional general-text control, default false), `score_source` (also score the original; default true when the source is at most 10B parameters).

### Shrink job result

`result` of a finished shrink job:

```json
{
  "params_b": 8.778, "active_params_b": 3.029,
  "removed": {"kind": "moe", "experts_before": 128, "experts_kept": 32, "experts_removed_per_layer": 96,
              "layers": 48, "top_k_after": 8, "kept_share_of_skill": 0.93},
  "scores": {"gsm8k": 0.61}, "source_scores": {"gsm8k": 0.84},
  "control": {"scores": {"gsm8k": 0.22}}, "general": null,
  "comparison": {"...": "see below"},
  "stub": true,
  "heal": {"steps": 300, "kind": "distillation", "teacher": true},
  "log": ["Loaded Qwen/Qwen3-30B-A3B-Instruct-2507 with 30.53B parameters."],
  "seconds": {"load": 0, "rank": 0, "slim": 0, "heal": 0, "eval": 0},
  "estimated": false, "plan": {"...": "the shrink block that ran"}
}
```

- `params_b` and `active_params_b` are the stored parameters and the parameters run per token of the result. On the stub they are the estimate and `estimated` is true. On Modal they are counted from the model (`wsp.slim.count_params`) and `estimated` is false; the result then also has `params_before_b`, `gpu` and `items` (items scored per task).
- `removed` for a dense model: `{"kind": "dense", "layers_before": 28, "layers_after": 21, "layers_dropped": [3, 9], "keep_mlp": 0.41, "keep_heads": 0.41}`.
- `source_scores`, `control` and `general` are null when not scored. `scores` are fractions in 0..1 per task.
- `heal.kind` is `distillation`, `lm_only` (the original did not fit in memory beside the student, so the heal used the language-model loss alone; the log says so) or `none` (0 steps). `heal.teacher` is true for distillation. A real run also has `kd_first`, `kd_last`, `lm_first`, `lm_last`, `trainable_params`, `seconds`. With a teacher the heal loss is the distillation term only.
- `seconds` has `load`, `rank`, `slim`, `heal`, `eval` and, on Modal, `total`.
- On error the job has `status: "error"` and `error` with the remote message.

### Shrink comparison

`result.comparison` has the build comparison's shape with `request_type: "shrink"` and these rows, in order:

| id | meaning | decides the verdict |
|---|---|---|
| `source` | the original model on the same items. The result is expected to be below it. `beaten` is null and `delta` is negative. | no |
| `random` | the same slim with random units removed, same counts, same heal steps | yes: the result must beat it by `margin` (0.02) |
| `general` | the same slim keeping what general text uses (only when `general_control` ran) | no |
| `best_model` | the best library model within 0.5x to 1.05x of the result's size (not the source), by the best score over the library score keys | yes when the library has a model in that size window; the result must be at or above it |

Each row has `id, label, split, per_task, value, measured, note, stub, result_value, delta, beaten` and `required` (true when it decides the verdict). `verdict` is `success` (every required row is measured and the result beats it), `failed` (a measured required row is not beaten), `unproven` (a required row is not measured: no random control, or models near the size have no score on this benchmark) or `none` (no score). `beaten`, `not_beaten` and `unmeasured` list the required row ids. `headline` is one plain sentence, for example "Success. It beats the random control by 38.0 points. It keeps 72% of the original model's score."

### Backends

- Stub: a made-up result with `stub: true` and `stub: true` on the comparison. Scores fall smoothly as more is removed, the random control is lower than the skill rule, and the source is above both. No model loads.
- Modal: `shrink_remote(plan, split, tasks, fidelity, options)` in the `wsp-studio` app (L40S; `shrink_remote_h100` when the source is above 10B parameters). It loads the source, builds the skill and general text from `wsp.healdata` pools (never a benchmark), runs `slim_moe` or `slim_dense`, heals with `slim.heal` and the original as teacher when it fits in memory, saves with `save_slim`, scores with the same `score_all` as builds, and runs the random control (same counts, rule `random`, same heal steps) when `options.control` is true.


### Shrink result limit check

On a real (not stub) shrink, the server checks the measured parameters against the spec limits. Over a limit, `result.comparison.verdict` is `failed`, its `headline` starts with "Failed attempt." and lists the problems, and `result.limit_problems` is the list of messages. The result has no `limit_problems` key when it fits. A shrink run checks the heal limit against the heal time of every model it heals (the shrunk model and each control). `POST /api/shrink/{id}/plan` and `/run` read the current spec, so edits made with `PUT /api/spec/{id}` apply at once.
