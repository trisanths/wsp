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
    synth_report.py    pipeline choice, refusals, stub verdicts, component result shape
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
tests/test_studio_synth.py    synthesis pipeline tests (fake Modal function, stub verdicts, compare rule)
```

Run: `.venv/bin/uvicorn studio.server.app:app --port 8765`, then open http://localhost:8765.

## Modes

Environment variables pick the pieces. The defaults run on the Mac with no network and no GPU.

| Variable | Default | Values |
|---|---|---|
| `WSP_STUDIO_BACKEND` | `stub` | `stub`, `modal` (the swarm's backends) |
| `WSP_STUDIO_LLM` | `none` | `none`, `anthropic`, `modal_vllm`, `openai_compat` |
| `WSP_STUDIO_PACE_S` | `0.8` | seconds of delay per offline eval, so an offline run is watchable |

The `stub` backend is the offline backend. It runs the real parts library and the real deletion masks, and scores them with a simulator (`server/offline_backend.py`), so no GPU runs. Every response carries `"stub": true` (the field name is kept) and the web app shows an "Estimate" badge on every number, with the tooltip "Scored offline by a simulator. No GPU ran. The parts are real library parts." Offline numbers must never look like measurements.

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
    {"name": "MMLU-Pro law", "measures": "...", "why": "...", "runnable_id": "mmlupro:law", "label": "MMLU-Pro law",
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
- `label` is the readable name of `runnable_id` ("IFEval" for `if_guard`, "MMLU" for `mmlu`, "MMLU-Pro law" for `mmlupro:law`), or null when `runnable_id` is null.
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
  "outcome": null,
  "empty_reason": null,
  "sort": "target score",
  "best": {"program_hash": "ab12cd34ef56ab12", "describe": "host + qmath.l8-19@6, heal 300", "scores": {"gsm8k": 0.43, "if_guard": 0.41}, "combined": 0.26, "agent": "planner-2", "program": {}},
  "agents": [
    {"id": "planner-2", "name": "Planner 2", "role": "planner", "blurb": "explores donor blocks",
     "status": "scoring", "now": "Scoring host + qmath.l8-19@6", "best": 0.43, "attempts": 7,
     "collapses": 0, "lessons": 0, "nudges_pending": 0}
  ],
  "leaderboard": [
    {"rank": 1, "program_hash": "ab12cd34ef56ab12", "describe": "...", "agent": "planner-2",
     "scores": {"gsm8k": 0.43, "if_guard": 0.41}, "combined": 0.26, "status": "ok", "target_score": 0.43,
     "vs_control": 0.055, "params_b": 1.31, "program": {}}
  ],
  "activity": [{"ts": 1794707100.1, "agent": "analyst", "kind": "analyst_map", "status": "info", "text": "..."}],
  "curve": [{"eval": 1, "best": 0.25, "score": 0.25, "program_hash": "ab12cd34ef56ab12", "name": "Math block L8-19 at L6, heal 300", "agent": "planner-1", "status": "ok", "scores": {"gsm8k": 0.25, "if_guard": 0.41}, "is_best": true}],
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
- `curve` has one point per scored eval in order: `best` is the best target score so far among valid (ok) programs, `score` is that eval's target score. A collapsed program adds a point but never raises `best`. The other fields describe that eval: `program_hash`, `name` (the readable program name, as on the leaderboard), `agent` (agent id), `status` (env status: `ok`, `collapsed`, ...), `scores` (every benchmark score of that eval, as fractions) and `is_best` (true when the point set a new best). Runs saved before these fields existed have only `eval`, `best` and `score`; pages show what is there. `best` is `null` until the first valid program is scored.
- `base_scores` is read once when the run thread starts. It is `{}` until then (treat it as pending).
- `stopped_early` is true when the person pressed Stop. The run still ends with status `done`. The held-out check and its controls are skipped, and the activity says so.
- `outcome` is `null` while the run goes, then `"finished"`, `"empty"` (the run ended on its own and nothing was scored, so there is no `best`), `"stopped"` (the person pressed Stop) or `"error"`. When the run ends every agent has `status: "done"` and `now` is `"Finished"` (`"Ended, nothing scored"` for `"empty"`, `"Stopped"` after a stop or an error).
- `empty_reason` is `null` while the run goes and when there is a `best`. When a run ended with no best model it is one plain sentence, for example "Every proposal was refused for going over your limits (195 refused). Raise the size limits on the spec." Other texts: every scored build failed, the run ended with an error, the run was stopped, or the search ended before any build was scored.
- `sort` is `"target score"`: `leaderboard` is ordered by the mean target score, best first. Builds with status `ok` come first, other failed statuses next and `collapsed` last. `rank` follows this order. `target_score` is that mean (null when no target was scored).
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
    "status": "ok",
    "pipeline": "synth",
    "heal_note": null,
    "components": [],
    "stages": [],
    "drift": null,
    "sizes": null
  },
  "budget": {"evals_used": 3, "evals_total": 12, "compute_used_s": 840, "compute_total_s": 2700},
  "error": null
}
```

`status` is `queued`, `running`, `done`, or `error`. For `compile`, only `fit` and `params_b` are filled. The synth fields are described in "Synthesis pipeline" below.

## Synthesis pipeline

A program with grafts or inserted layers compiles with `wsp/synth.py` (`result.pipeline` is `"synth"`). A program with no grafts and no inserts (masks, task vectors, a LoRA only) uses the classic compile (`"classic"`). `WSP_STUDIO_PIPELINE=classic` forces the classic compile for every program. Score, Heal and Train all mean "score with the synthesis heal": the heal steps of the program (300 when it has none) are split in two halves, heal 1 (translations and gates) and heal 2 (a LoRA after the first seam, with a KL term to the frozen host). A heal limit on the spec shortens the steps. Heal and selection text come from the `wsp/healdata.py` pool of the spec's task type. The practice sets that guard the heal and decide the verdicts never receive gradient, and they come from `wsp/healdata.py` practice sources that are not scored in the request (GSM8K from SVAMP, ASDiv and AQuA, MMLU from TriviaQA, MBPP from APPS, MedMCQA from PubMedQA). A scored task with no such source runs without practice, and a spec guard with no practice set (such as `if_guard`) is not enforced at compile time. `result.notes` says so, and the practice skills are guarded instead.

`wsp/synth.py` places four kinds: `block`, `attn_heads`, `mlp_cluster` and `expert`. Inserted layers become blocks. A deletion mask, task vector, routed experts, carve, splice or tap in the same program is refused with 400 and a plain message ("The synthesis pipeline cannot build with a deletion mask yet. Remove it, or run the classic pipeline."), and so is a part of any other kind. The stub backend does not refuse.

### Builder job result: synth fields

`result` (score, heal and train jobs) adds:

```json
{
  "pipeline": "synth",
  "heal_note": null,
  "components": [
    {"id": "mlp_cluster_qwen25math15b_l16_gsm8k", "label": "Math neuron cluster L16", "kind": "mlp_cluster",
     "donor": "Qwen/Qwen2.5-Math-1.5B", "host_layer": 9, "verdict": "contributes", "evidence": "practice", "stub": false,
     "detail": {"skill": "gsm8k", "contribution": 0.031, "text_gain": 0.02, "content_gain": 0.03,
                "transfer": 0.41, "donor_kl_base": 1.2, "donor_kl_written": 0.7,
                "heldout_loss_on": 1.01, "heldout_loss_off": 1.04, "gate_domain": 0.8, "gate_general": 0.1,
                "fit_error": 0.3, "fit_r2": 0.71, "part_params": 1048576, "closed": false, "scores_without": {"gsm8k": 0.49}}}
  ],
  "stages": [
    {"name": "heal 2", "seconds": 70.0, "practice_before": {"gsm8k": 0.52}, "practice_after": {"gsm8k": 0.4},
     "rolled_back": true, "detail": "100 steps with a LoRA after the first seam, KL to the frozen host.", "stub": false}
  ],
  "drift": {"base_kl": 0.012, "agree": 0.93, "tokens": 2400, "text": "general pool, held out"},
  "sizes": {"total": 1301000000, "per_token": 1105000000, "parts": 1048576, "glue": 524288, "gates": 4096, "lora": 0},
  "run_params_b": 1.1
}
```

- `evidence` is `"practice"` when the verdict rests on a practice score (`detail.contribution` is not null) and `"text"` when it rests on held-out text loss alone (the part's skill has no practice set in this request). A `"text"` verdict makes no claim about the scored task: show it as "not tested on the task" and keep `detail.text_gain` as a detail.
- `verdict` is `"contributes"`, `"no better than random"` or `"harmful"` (`"not tested"` when the component test was switched off). Color all of them the same way as the comparison rows: only `contributes` is good.
- `detail` numbers can be null. `contribution` is the practice score of the whole model minus the model with this part switched off. `text_gain` is the loss on held-out skill text without the part minus with it. `content_gain` is the real part minus a random part of the same size, from the control compile inside the pipeline. `transfer` is the share of the host-to-donor gap on held-out skill text that the build closes (0 is the host, 1 is the donor), null when the donor has another tokenizer. `gate_domain` and `gate_general` are the mean token gate on skill text and on general text. `fit_error` is the relative error of the input translation after its output matching (null when the donor has another tokenizer: only the ridge fit runs then). `fit_r2` is the R squared of the ridge fit of the input translation. `scores_without` is the practice score with the part off.
- `stages` has six rows in order: `place`, `translate in`, `translate out`, `heal 1`, `heal 2`, `verify`. `seconds` is null for `place`. `practice_before` and `practice_after` are practice scores `{task: value}` (null when no practice set exists for the scored tasks). `rolled_back` is true when the guard failed and heal 2 was undone. The `verify` seconds include the random control compile.
- `drift` is the KL from the frozen host to the build on held-out general text (0 means the host's behavior is kept), with the top-1 agreement. It is null when the probe failed.
- `sizes` are measured on the saved model (null on the stub and on classic results). `params_b` is `sizes.total / 1e9` and `run_params_b` is `sizes.per_token / 1e9` for synth results, so the limit check uses the measured size.
- Every row of `components` and `stages` has `stub: true` on the stub backend: the verdicts there are estimated offline from the part's library record (skill tags matched to the task, status, measured score against control when there is one). They are deterministic per part (the same part always gets the same verdict, a random-part graft is always "no better than random"), and `drift.text` reads "estimated offline, not measured".
- `heal_note` is the heal warning string for a healed classic result on a real backend, else null. It is always null for `pipeline: "synth"`.
- `pipeline: "classic"` results have `components`, `stages`, `drift` and `sizes` all null.

### Comparison and component verdicts

`comparison` (builder result and leaderboard entries) has `no_component_contributes`: true when the result has components with `evidence: "practice"` and every one of those is `"no better than random"` or `"harmful"`. Then `verdict` is `"failed"` and the headline says so, for example "Failed attempt. No component contributes: Math neuron cluster L16 is harmful, Code block L14-23 is no better than random. The score gain is not shown to come from the parts." A result that was already failed keeps its reason and gets the sentence added. One contributing component leaves the verdict as it was.

### Run snapshot: leaderboard entries

Each `leaderboard` entry adds `pipeline` (`"synth"` or `"classic"`), `components` (the list above, or null for classic) and `drift` (or null). `params_b` is the measured total when the backend measured it. `comparison` follows the rule above.

### Modal functions

`synth_remote(program, split, tasks, fidelity, options)` and `synth_remote_h100` (a donor above about 10B). `options`: `catalog_records`, `pipeline` (`"synth"` or `"classic"`; `"classic"` runs the old compile and score), `pool` (the healdata pool of the spec's task type), `guards` (task ids kept within tolerance of the host), `max_total_params_b`, `max_run_params_b`, `max_heal_min`, `control_run` (default true), `probes` (default true), `synth_settings` (overrides), `token`. It returns `{"ok", "pipeline", "refused", "scores", "items", "components", "stages", "drift", "sizes", "seconds": {"prepare", "compile", "heal", "probe", "eval", "total"}, "graft_report", "heal_loss", "heal_steps", "texts", "control_run", "practice", "guard", "notes", "gpu", "scoring"}`. A refusal or a budget error returns `{"ok": false, "refused": true, "error": "..."}`. Compute charged to a budget is `compile` (prepare, compile and probe), `heal` and `eval`.

## Routes

All bodies are JSON unless marked.

| Method | Path | Body | Returns |
|---|---|---|---|
| GET | `/api/health` | | `{"ok": true, "product": "Sapien", "backend": "stub", "llm": "none", "llm_model": null, "stub": true, "heal_note": null}`. `heal_note` is a warning string on real backends. It applies to results from the classic pipeline only: read `result.heal_note` on a result (null for every synth result) instead of showing the health note next to it. |
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
| GET | `/api/library/recipes` | | `{"recipes": [...]}` (see Library lists) |
| GET | `/api/library/findings` | | `{"findings": [...]}` |
| GET | `/api/library/stitches` | | `{"stitches": [...]}` |
| GET | `/api/library/models` | | `{"models": [...]}` (all models, incomplete ones flagged) |
| POST | `/api/program/text` | `{"program": {}}` | `{"text": "..."}` |
| POST | `/api/program/parse` | `{"text": "..."}` | `{"program": {}}` or 400 `{"error": "...", "line": 3}` |
| POST | `/api/program/validate` | `{"program": {}}` | `{"errors": []}` |
| POST | `/api/builder` | `{"spec_id": "..."}` | `{"builder_id": "...", "budget": {...}}` (one budget per builder session; it is the spec's `budget`: its evals and compute minutes, with the stub compute floor of 450 s per eval) |
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

With the synth pipeline (the default for programs with grafts or inserts) `heal`, `train` and `score` all run the synthesis heal; see "Synthesis pipeline".

## Sapien changes

The product is Sapien. A program builds one network that runs alone. Taps and anything that runs a second model at inference are not part of it: the palette has no taps, the agents never propose them, and the server refuses a program with `taps` or `tap_train` (400, "A tap runs a second model at inference...").

### Limits

The spec has `limits`: `{"max_total_params_b": 1.9, "max_run_params_b": 1.9, "max_heal_min": 20}`. `size.max_params_b` mirrors `max_total_params_b` for older pages. `size.max_b` is 32. The draft reads the limits from the request ("under 3B", "1.2B per token", "heal for 30 minutes"). Max run per token is at most the total and at least the host size. A build spec leaves room: with no size in the request the total is the host size times 1.5 (rounded to 0.1B) and the run limit is the same; with a size ("under 2B") the total and the run limit are that size; a build total is never below the host size plus 0.2B. A shrink spec keeps its own rule.

The server estimates a program (`server/sizing.py`): total parameters, parameters run per token, and heal minutes. The estimate is rough and says so (`"estimated": true`). `POST /api/builder/{id}/size` with `{"program": {}}` returns `{"estimate": {...}, "limits": {...}, "problems": [], "ok": true}`. A builder action, a handoff and a run with a seed program are refused with 400 and a plain message when over a limit. In a run, a proposal over a limit is dropped before it is scored; the snapshot counts them in `refused` and the activity has `refused` items. Growing the base is allowed while the limits hold.

### Heal data

Heal, fit and selection text comes from the `wsp/healdata.py` pools, never from a benchmark. The studio names heal data with pool names only: the default for a math task is `math`, knowledge is `general`, code is `code`, medicine is `medicine`, anything else `general`. A program may use any of `general`, `math`, `code`, `medicine`, `prose`, `mix`. `program.heal.data` is one of these. A benchmark name such as `gsm8k_train` or `mmlu_aux` is refused with a plain message ("Heal data gsm8k_train is the train set of the benchmark openai/gsm8k. Heal and fit text may not come from a benchmark. Use a pool: ...") by `POST /api/program/validate` (in `errors`), the builder routes and a run with a seed program. The builder's default heal uses the pool of the spec's task type.

### Shrink plan sentence

`plan.summary` of a shrink names the rule in plain words: saliency "by how much the skill text uses them", freq "by how often the skill text routes to them", selective and specific "by how specific they are to the skill", skill "by how much the skill text uses them", skill_cost "by their cost on the skill text", random "at random (control)", general "by use on general text (control)". Example: "Keep 32 of 128 experts per layer by how much the skill text uses them." Candidate `why` text is rebuilt when `keep_skill` changes and names the kept skill ("tagged for medicine") when the model is tagged with it. A stub shrink result has `source_scores` equal to the spec's `base` for a benchmark that has one.

### Library palette

`GET /api/catalog` returns `parts` from the compiler catalog and from `library/index.jsonl` (read when the server starts) and `atlas/entries/*.json` when that folder exists. Every part has `kind` (`block`, `attn_heads`, `mlp_cluster`, `expert`, `inserted_layer`), `status` (`candidate`, `validated`, `rejected`), `evidence` (one line), `source` (`stub`, `catalog`, `library`, `atlas`), `usable` and `usable_reason`. A part is usable when the compiler catalog has a record for it. Tap read points from the library are listed as not usable. The response also has `kinds` (name and count per kind), `library` (counts) and `product`. Masks have `status` and `evidence` too.

The catalog response also has:

- `counts`: `{"total", "placeable", "status": {candidate, validated, rejected}, "kinds": {kind: {"total", "placeable", "candidate", "validated", "rejected"}}}`. Masks count under the kind `deletion_mask`.
- On each part and mask: `findings` (`[{"id", "title"}]`, the findings that name it), and on each part `stitches` (same shape), `skills`, and `from_library` (true when the compiler record was built from the library file).
- `library.loaded` (rows loaded, by index kind, plus `inserted_layer` from the atlas) and `library.failed` (number of rows that did not load).

On the Modal backend the server writes `catalog_modal.jsonl` as the repo `catalog.jsonl` plus one record per library part whose file has what the compiler needs: a donor model and layer plus an index file in the library folder (`mlp_cluster`, `attn_heads`, `expert`), or a donor model and a run of layers (`block`, `inserted_layer`). The record has `library_entry` (the library id) and its evidence line says it came from the library. A part stays locked when something specific is missing, and `usable_reason` says what (for example `No unit indices in the library file.`). The stub (offline) backend writes `catalog_offline.jsonl` with the same records, so both backends place the same parts and the placeable counts are equal. `source` is never `stub` any more. The lock reasons are the same on both backends.

### Library lists

The server reads every row of `library/index.jsonl` when it starts, logs one line with the counts and any row that failed (with the reason), and never stops on one bad row. `library.get()["counts"]` has the rows loaded by kind and `["failed"]` has `{"id", "kind", "reason"}` for each row that did not load.

- Recipe: `id`, `title`, `status`, `status_reason`, `host`, `builds` (one line), `components`, `models`, `skills`, `evidence`, `source`, `text`, `syntax_gap`, `starting_program` (a program for the builder, or null), `no_start_reason` (why there is no starting program), `start_problems` (what the builder would complain about, such as a part that is not in the compiler catalog). A recipe has a starting program when its program has a host and no taps, bake or carve.
- Finding: `id`, `title`, `claim`, `setup`, `numbers`, `numbers_source`, `status`, `status_reason`, `source`, `skills`, `evidence`, `about` (`{"parts": [ids], "models": [hf ids]}`). A finding is about a part when its text names the part id, or names the part's layer and the part's model is one of the finding's models.
- Stitch: `id`, `title`, `status`, `host`, `donor`, `host_layers`, `donor_layer`, `stitch_kind`, `method`, `objective`, `steps`, `sink`, `params`, `penalty` (before and after), `ridge_r2`, `fit` (one line), `evidence`.
- Model: `id`, `hf_id`, `title`, `status`, `family`, `params_b`, `layers`, `hidden`, `incomplete` (true when `hf_id` or `params_b` is missing), `missing` (the empty fields), `findings`.

### Comparison set

The spec has `comparators`: `{"distill": {"gsm8k": 0.40}}`, scores a person measured for base + distillation at the same budget. A run snapshot has `comparison` (for the best build) and each leaderboard entry has one. A builder score result has `comparison` and `size`.

```json
{"verdict": "failed", "headline": "Failed attempt. It does not beat ...", "margin": 0.02,
 "result": {"label": "This build", "value": 0.393, "per_task": {}, "params_b": 1.3, "heldout": null},
 "rows": [{"id": "base", "label": "Base", "split": "search", "per_task": {}, "value": 0.25, "measured": true,
           "note": "...", "stub": true, "result_value": 0.393, "delta": 0.143, "beaten": true, "within_margin": false}],
 "beaten": [], "not_beaten": [], "unmeasured": [], "best_model_shown": false, "stub": true}
```

Row ids: `base`, `lora`, `distill`, `random`, and `best_model` (only when the total limit is at least twice the host size; it is the best library model within the limit). `verdict` is `failed` when a measured reference is not beaten by `margin` (a collapsed build or one over a limit also fails), `unproven` when nothing failed but a reference is not measured, `beats` when all are measured and beaten, `none` with no score. On real backends `distill` is not measured until the spec has it. On the stub backend `distill` is made up and its row says so (`stub: true`).

`within_margin` on a row is true when the result is above that reference but by less than `margin`, false when it is a clear win, level or below, and null when the row is not measured. The headline follows the numbers: "Failed attempt. It is within 2 points of Random components (same size) (40.6 against 39.5), so the parts are not shown to matter." A result below a reference reads "is below ...", a tie reads "only ties ...". Shrink comparisons use the same rows and wording. The verdict logic did not change.

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
- `source_active_params_b` is the estimated parameters the source runs per token. `experts_total` is the source's experts per layer for an MoE source and null for a dense one. `candidates` holds up to 5 other sources that fit, best first. `notes` holds plain sentences (for example a named model that cannot be slimmed).
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

- Stub (offline): an estimated result with `stub: true` and `stub: true` on the comparison. Scores fall smoothly as more is removed, the random control is lower than the skill rule, and the source is above both. No model loads.
- Modal: `shrink_remote(plan, split, tasks, fidelity, options)` in the `wsp-studio` app (L40S; `shrink_remote_h100` when the source is above 10B parameters). It loads the source, builds the skill and general text from `wsp.healdata` pools (never a benchmark), runs `slim_moe` or `slim_dense`, heals with `slim.heal` and the original as teacher when it fits in memory, saves with `save_slim`, scores with the same `score_all` as builds, and runs the random control (same counts, rule `random`, same heal steps) when `options.control` is true.


### Shrink result limit check

On a real (not stub) shrink, the server checks the measured parameters against the spec limits. Over a limit, `result.comparison.verdict` is `failed`, its `headline` starts with "Failed attempt." and lists the problems, and `result.limit_problems` is the list of messages. The result has no `limit_problems` key when it fits. A shrink run checks the heal limit against the heal time of every model it heals (the shrunk model and each control). `POST /api/shrink/{id}/plan` and `/run` read the current spec, so edits made with `PUT /api/spec/{id}` apply at once.


### Best existing model for GSM8K

The library GSM8K scores were measured with the old scorer (256 new tokens), so the best-model reference ignores them until the library models are re-scored with `skills_eval`. The `best_model` row for GSM8K reads "not measured" with the note "GSM8K library scores are awaiting re-scoring with the current scorer". A shrink scored on GSM8K is then `"unproven"`.

## Measured results and the compiler front door

New files: `server/measured.py`, `server/frontdoor.py`, `server/routes_extra.py` (routes), `server/offline_facts.json`. Web: `web/screens/measured.js`, `web/screens/frontdoor.js`, `web/mock-extra.js`, `web/extras.css`.

### GET /api/measured

A list of measured results. Every row was scored by the compiler, so the web app shows a small "measured" tag, never the STUB badge. Scores are percent points (0 to 100), not fractions.

```
[{ "id", "title", "kind": "skill_removal", "skill": "code", "model": "<hf id>", "experts_per_layer": 128,
   "params_run_per_token_b": 3.03, "date", "scorer": "wsp.skills_eval", "runs_per_cell": 1, "heal": "500 steps on two rows",
   "source", "caveats": ["plain sentence", ...],
   "rows": [{ "label", "experts_kept", "rule", "total_params_b", "scores": {"humaneval", "mbpp", "gsm8k", "mmlu"},
              "is_control", "is_full", "heal_steps" }] }]
```

A score a row did not measure is absent from `scores`. `heal_steps` is 0 for a raw cut and the number of distillation heal steps otherwise; the result's `heal` is a short phrase for the whole table. A control row picks experts by a rule that should not work (random, skill-to-general ratio).

### GET /api/spec/{id}/request

`{ "request": {...}, "unmapped": ["bench id", ...], "errors": ["plain message", ...] }`. The request has the format of `docs/auto.md` (name, base, add, remove, keep, limits, reported, seed). `unmapped` lists benchmark ids with no known Hugging Face dataset id. `errors` is the front door's own validation (`wsp.auto.validate_request`), empty when the request is valid. 404 for an unknown spec.

### POST /api/spec/{id}/plan

Runs `wsp.auto.plan()` on that request in a worker thread with a 90 second timeout. No GPU and no weights: it reads the Hub by range reads, and falls back to `server/offline_facts.json` when the Hub fails.

`{ "plan": {...as the front door returns it}, "seconds", "offline": bool, "offline_reason"?: str, "verified": false, "note": "...", "unmapped": [...] }`

The plan is cached per spec and request. A front door failure never gives a 500: the body is `{"error": "..."}` with 400 (invalid request), 408 (timeout), 422 (the compiler could not plan) or 404 (unknown spec). The traceback goes to the server log.

## Mining routes

Files: `server/mining.py` (all routes, registered from `app.py`), `tests/test_studio_mining.py`. CPU only, no GPU calls. The studio does not start mining. A request is written to `mine_requests.json` in the studio output dir (`WSP_STUDIO_OUT`) as `{"requests": [Request, ...]}`, oldest first, and the mining team reads that file as its queue. They update `status` (`queued`, `running`, `done`, `failed`) and each stage `status` in place. A queued or running model cannot be requested again.

### GET /api/mine

`{ "requests": [Request], "models": [ModelSummary] }`. Requests are newest first. Models are every donor model the library has parts or masks for, most parts first.

ModelSummary: `{ model, short, params_b, layers, experts (null for dense), kind: "dense"|"moe", parts: { total, by_kind: {kind: n} }, skills: {skill: n}, status: "mined" }`. `params_b` and `experts` are null when the library has no profile for the model.

### GET /api/mine/lookup?model=<hf id>

`{ model, found, params_b, architecture, layers, experts, license, gated, downloads, likes, in_library, mined_parts, error }`. It asks the Hub with a 6 second timeout and no token, and caches the answer for 10 minutes. When the Hub fails and the library has a profile, it answers from the profile with `found: true` and an `error` that says so. When neither has the model, `found` is false and `error` says why. The route always answers 200.

### POST /api/mine

Body `{ model, skills: [string], effort: "smoke"|"medium"|"full", note }`. Skills may be empty (mine everything), at most 20. Note at most 500 characters. Effort defaults to medium.

Returns a Request: `{ id ("mine_" + 8 hex), model, skills, effort, note, status: "queued", created (UTC ISO), stages, estimate }`. Stages in order, each `{ id, name, status: "waiting"|"running"|"done", sponsor }`: `fetch` (modal), `read` (openai), `scan` (modal), `ablate` (modal), `gate` (modal), `publish` (pastdev). `estimate` is `{ gpu_min, cost_usd, gpu, is_estimate: true, note }`, a rough figure from parameter count and effort.

400 with `{ "error": "plain message" }` for a bad model id, bad skills, bad effort, a long note, a body that is not JSON, or a model already queued or running.

### GET /api/mine/req/{id}

The Request, or 404.

### GET /api/models/{model}

`model` has a slash (`Qwen/Qwen3-30B-A3B`). Returns ModelSummary plus `{ parts: [palette item], layer_map: [{ layer, kind, count, to }], techniques: [id], findings: [...], recipes: [{ id, title, status, host, builds, skills, evidence }], part_counts }`. Palette items have the shape of `GET /api/catalog` parts. `layer_map` puts each part at its first layer, and `to` is the last layer of the longest range there. `techniques` are ids from `web/techniques.js` (`weight-read`, `activation-scan`, `attribution`, `saliency`, `skill-map`, `weight-delta`, `deletion-mask`, `ablation`, `gate`) read from each part's method, source and fields. 404 when the library has nothing for the model.

### GET /api/parts/{id}

`id` may contain slashes (atlas parts). Returns the palette item fields plus:

```
raw: { signature: { skill: { search, heldout, control_search, control_heldout } } | null,
       z, z_heldout, random_mean, random_std, random_n }
num_units, tier, skill (library skill record or name), selection: { skill, source, datasets } | null,
units_file, technique: [id], evidence: [library evidence records] | null, evidence_line, findings, stitches,
recipes: [{ id, title }]
```

Drops are relative drops in skill score when the part is switched off. `z` and `z_heldout` compare the main skill's search and held-out drop with the mean and sample standard deviation of `random_n` same-size random parts. Fields the library does not hold are null. 404 when unknown.
