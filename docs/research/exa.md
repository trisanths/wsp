# Exa integration guide (part 3: RL environment and swarm)

Verified 2026-11-14 against the live docs (llms.txt index, reference pages, and the OpenAPI spec at https://exa.ai/docs/exa-spec.yaml) and against `exa-py` 2.28.0 installed in the repo `.venv` and inspected with `inspect`. No API key exists on this machine, so nothing below was run against the live service. Items marked UNVERIFIED need the smoke test in the last section.

Docs live at `exa.ai/docs/...` (the old `docs.exa.ai` host is the same content). Every page has a `.md` twin, for example https://exa.ai/docs/reference/search.md.

## 1. Important drift from older Exa knowledge

- The `research paper` category is gone. The enum is now `company`, `publication`, `news`, `personal site`, `financial report`, `people`. Use `category="publication"` for papers and preprints (spec text: "scholarly publications such as research papers, preprints, and journal articles, with metadata like authors, venue, and citations"). Other strings are accepted only as hints.
- Search `type` values: `auto` (default), `lite`, `deep-lite`, `deep` are current. `fast`, `instant`, `deep-reasoning` are legacy aliases. `neural` still appears in the SDK literal but is not in the docs; do not use it. `keyword` does not exist.
- Deprecated: `useAutoprompt`, `livecrawl` (use `maxAgeHours`), `context`, `includeText`/`excludeText`, `startCrawlDate`/`endCrawlDate` (ignored with a warning), `subpages` on search.
- Legacy `exa.research.*` still exists in the SDK but the `/research` endpoint is not in the current spec. Its replacement is the Agent API (`/agent/runs`). Do not build on `exa.research`.

## 2. Endpoints and parameters

Base URL `https://api.exa.ai`. Auth: header `x-api-key: $EXA_API_KEY` or `Authorization: Bearer $EXA_API_KEY`. Both SDKs read `EXA_API_KEY` from the environment.

### POST /search  (https://exa.ai/docs/reference/search.md)

| Parameter | Notes |
|---|---|
| `query` | required |
| `type` | `auto`, `lite`, `deep-lite`, `deep` (plus legacy) |
| `numResults` | 1 to 100, default 10 (limits vary by type) |
| `category` | see above. `company` and `people` reject date filters; `people` also rejects `excludeDomains` |
| `includeDomains`, `excludeDomains` | up to 1200 entries each; hostname, hostname plus path prefix, or `*.` wildcard |
| `startPublishedDate`, `endPublishedDate` | ISO 8601 |
| `contents` | object, see below |
| `outputSchema` | structured synthesis; root `type` is `text` or `object`; limits: 10 properties total, 2 nesting levels, arrays need `items`; adds about 2 s; not supported with `lite` |
| `systemPrompt`, `objective` | guidance (objective max 4096 chars) |
| `additionalQueries` | 1 to 10 extra queries, deep types only |
| `userLocation`, `moderation`, `userIdentifier`, `stream` | as named |

`lite` accepts only: `query, objective, userIdentifier, metadata, category, numResults, contents.highlights, contents.text, contents.snapshotAsOf, userLocation, includeDomains, excludeDomains, startPublishedDate, endPublishedDate, moderation, compliance`. Anything else is rejected (400). It serves from the index with no live fetch, no `summary`, no `outputSchema`.

`contents` options (same object on /search, /findSimilar, /contents):

- `text`: `true` or `{maxCharacters, includeHtmlTags, linkStyle, verbosity: compact|standard|full, includeSections, excludeSections}`. In the SDK, `verbosity` and sections need `max_age_hours=0` to take effect.
- `highlights`: `true` or `{query, maxCharacters}`. `dynamic` and `verbosity` need the beta header `Exa-Beta: dynamic-highlights-2026-08-28` (SDK: `betas=[...]`). Skip the beta.
- `summary`: `{query, schema}`. `schema` is a JSON Schema object (SDK also accepts a Pydantic model). Per-page structured summaries.
- `maxAgeHours`: `-1` cache only, `0` live fetch, omitted means cache when available. `snapshotAsOf` serves a stored version at or before a date.
- `livecrawlTimeout` (ms, default 10000), `extras` (`links`, `imageLinks`).

Response: `results[]` with `title, url, id, publishedDate, author, text, highlights, summary, entities` (for `publication`: title, year, date, type in article|book|preprint|report|review|..., citationCount, authors, venue; schema `SearchPublicationEntityOutput`), `costDollars {total, search, summary, contents}`, `searchTime`, `warnings` (log them), `notices` (pass to the model), and `output {content, grounding}` when `outputSchema` is set. `grounding` is a list of `{field, citations[{url,title}], confidence: low|medium|high}`. Exa adds it; do not put it in your schema.

### POST /contents  (https://exa.ai/docs/reference/get-contents.md)

`urls` or `ids` (1 to 100, not both), plus `text`, `highlights`, `summary`, `maxAgeHours`, `livecrawlTimeout`, `subpages` (0 to 100), `subpageTarget`, `extras`. Response has `results`, `statuses[{id,status,source: cached|crawled,error}]`, `costDollars`.

### POST /findSimilar  (spec `FindSimilarRequest`, https://exa.ai/docs/exa-spec.yaml)

Exists in the spec and SDK but has no page in llms.txt. Parameters: `url` (required), `numResults`, `includeDomains`, `excludeDomains`, `startPublishedDate`, `endPublishedDate`, `excludeSourceDomain`, `category`, `contents`, deprecated text/crawl filters. Use it to expand from one known paper to neighbors. Pricing is not listed separately: UNVERIFIED, assume the `auto` rate.

### POST /answer  (https://exa.ai/docs/reference/answer.md)

Params: `query`, `text` (include page text in citations), `model` (`exa`, `exa-pro`, `exa-research`, `exa-fast` per the docs; the SDK type hint allows only `exa` and `exa-pro`, so pass others through `extra_body={"model": ...}`; UNVERIFIED which are live), `systemPrompt`, `outputSchema` (Draft 7 JSON Schema; `answer` then comes back as an object), `userLocation`, `userIdentifier`, `stream`. Response: `answer` (string or object), `citations[{title,url,publishedDate,author,id,text}]`, `costDollars`. Billing: $5 per 1k requests.

### Deep search (`type="deep"` or `"deep-lite"` on /search)  (https://exa.ai/docs/search/deep-search.md)

Iterative multi-query search with synthesis. `deep-lite` about 4 s, `deep` 4 to 15 s. Combine with `outputSchema` and `systemPrompt`. This is the cheapest way to get a compact, typed, cited prior from several sources in one call.

### Agent API  (https://exa.ai/docs/agent/quickstart.md)

`POST /agent/runs` (async), `GET /agent/runs/{id}`, `GET /agent/runs`, `GET /agent/runs/{id}/events`, `POST /agent/runs/{id}/cancel|stop`. Params: `query`, `effort` (`minimal|low|medium|high|xhigh|auto|ultra`), `outputSchema` (result in `output.structured`), `systemPrompt`, `input`, `dataSources`, `previousRunId`, `budget.maxCostDollars` ($1 to $100), `metadata`. Statuses: queued, then completed|failed|cancelled. Docs say nothing about determinism. This replaces the old Research API and fits the "literature agent that keeps notes current" job; it is too slow and variable for the planner inner loop.

### Websets  (https://exa.ai/docs/websets/quickstart.md)

`/v0/websets` with searches, enrichments, items, monitors, imports, webhooks. Built for large verified entity lists (companies, people). Not relevant to this project. Skip. Also in the spec: `/monitors` (scheduled searches, $2 per 1k) and `/batches` (batched /search and /contents requests, customId keyed). Not needed at this scale.

## 3. Python SDK (`exa-py` 2.28.0, verified by `inspect`)

Install: `uv pip install --python .venv exa-py` (done; Windows venv path is `.venv/Scripts/python.exe`). Add `exa-py` to `[project.optional-dependencies]` as a new extra (for example `research = ["exa-py"]`) and to the Modal image `pip_install`; this guide did not edit `pyproject.toml`.

Client attributes: `search, search_and_contents, find_similar, find_similar_and_contents, get_contents, answer, stream_answer, stream_search, agent (agent.runs), research (legacy), websets, monitors, request, wrap`.

```python
from exa_py import Exa
exa = Exa()   # reads EXA_API_KEY; Exa(api_key="...") also works
```

Signatures (verified):

- `search(query, *, stream=False, contents=None|False, num_results, include_domains, exclude_domains, start_published_date, end_published_date, type, category, additional_queries, system_prompt, output_schema, objective, user_identifier, moderation, user_location, betas, extra_body, timeout, ...)` returns `SearchResponse[Result]`.
- `search_and_contents(query, **kwargs)` is a thin wrapper; pass `text=`, `highlights=`, `summary=`, `max_age_hours=` as kwargs. Prefer `search(..., contents={...})`, which is the form the current docs show.
- `get_contents(urls, **kwargs)` with `text, summary, highlights, max_age_hours, snapshot_as_of, subpages, extras`.
- `find_similar(url, *, contents, num_results, include_domains, exclude_domains, start_published_date, end_published_date, exclude_source_domain, category, ...)`.
- `answer(query, *, stream=False, text=False, system_prompt, model: 'exa'|'exa-pro', output_schema, user_location, user_identifier, ...)` returns `AnswerResponse(answer, citations, cost_dollars)`.
- `agent.runs.create(*, query, system_prompt, input, output_schema (dict or Pydantic class), effort, budget, previous_run_id, metadata, data_sources, user_identifier, stream)`, `agent.runs.poll_until_finished(run_id, *, poll_interval=1000 ms, timeout_ms=3600000)`, `create_and_wait`, `get`, `list`, `cancel`, `stop`, `events`.

Return shapes. `SearchResponse` is a dataclass (so `dataclasses.asdict` works for caching) with fields `results, resolved_search_type, auto_date, context, output, statuses, cost_dollars, search_time, warnings, notices`. `Result` has `url, id, title, score, published_date, author, image, favicon, subpages, extras, entities, crawl_date, snapshot_at, text, summary, highlights, highlight_scores`. `output` is parsed by `parse_deep_search_output`; confirm in the smoke test whether it exposes `.content` and `.grounding` as attributes.

Examples that match the current docs and SDK signatures:

```python
# Papers about a mechanism, with relevant excerpts only (cheap, small prompts)
res = exa.search(
    "interference when merging sparse masks or task vectors in late transformer layers",
    type="auto",
    category="publication",
    num_results=5,
    end_published_date="2026-11-14T00:00:00Z",   # pin for reproducibility
    contents={"highlights": {"query": "layer-wise interference when merging", "max_characters": 1500}},
)
for r in res.results:
    print(r.title, r.url, r.published_date, r.highlights)
print(res.cost_dollars)

# Typed, cited prior in one call
res = exa.search(
    "Which MLP layers store factual knowledge in Llama-style models?",
    type="deep-lite",
    category="publication",
    system_prompt="Prefer papers on causal tracing, knowledge neurons, and layer ablation. Name layer ranges as fractions of depth.",
    output_schema={
        "type": "object",
        "required": ["claims"],
        "properties": {"claims": {"type": "array", "maxItems": 5, "items": {
            "type": "object",
            "required": ["claim", "layer_hint", "model", "strength"],
            "properties": {
                "claim": {"type": "string"},
                "layer_hint": {"type": "string"},
                "model": {"type": "string"},
                "strength": {"type": "string", "enum": ["weak", "medium", "strong"]},
            }}}},
    },
)
print(res.output.content, res.output.grounding)   # attribute names UNVERIFIED, see smoke test

# Per-paper structured summary
res = exa.search("Deletion Tuning masks MMLU-Pro biology", category="publication", num_results=3,
    contents={"summary": {"query": "what task, what model, what result", "schema": {
        "type": "object", "properties": {"task": {"type": "string"}, "finding": {"type": "string"}}}}})

# Neighbors of a known paper
sim = exa.find_similar("https://arxiv.org/abs/2611.10866", num_results=8,
    exclude_source_domain=True, contents={"highlights": True})

# Answer with citations
ans = exa.answer("What is known about TIES or DARE merging in sparse regimes?", text=False)
print(ans.answer, [c.url for c in ans.citations])

# Literature agent (async, metered)
run = exa.agent.runs.create(query="Update the weekly notes on mask merging and model-editing work.",
                            effort="medium", output_schema={...})
run = exa.agent.runs.poll_until_finished(run.id, poll_interval=4000)
print(run.output.structured, run.cost_dollars)
```

## 4. Pricing and limits (https://exa.ai/docs/admin/pricing.md, https://exa.ai/docs/admin/billing.md)

| Item | Price |
|---|---|
| Search `lite` | $4 / 1k requests (up to 10 results); +$1 / 1k results above 10 |
| Search `auto` (and legacy `fast`) | $7 / 1k requests; +$1 / 1k results above 10; summaries +$1 / 1k pages |
| Search `deep-lite`, `deep` | $12 / 1k requests (about 4 s, 4 to 15 s) |
| `deep-reasoning` (legacy) | $15 / 1k |
| Contents | $1 / 1k pages per content type (text, highlights, summary each) |
| Answer | $5 / 1k requests |
| Agent fixed effort | minimal $0.012, low $0.025, medium $0.10, high $0.50, xhigh $1.00 per request |
| Agent metered | `auto` default cap $5, `ultra` cap $20; $0.10 per Agent Compute Unit, $0.005 per search tool call |
| Free tier | $10 credit at signup (about 2,500 Lite searches), no card; resets to $10 each month |

Whether search `auto` includes highlights in the base price or bills $1 / 1k pages for them is ambiguous in the docs (contents are billed "per content type"). Treat each requested content type as $0.001 per page and read `res.cost_dollars` in the smoke test to settle it. UNVERIFIED: the extra cost of `outputSchema` on deep search.

Rate limits (per team, across keys): `/search`, `/answer` 10 QPS; deep types 5 QPS; `/contents` 100 QPS; `/agent/runs` 5 QPS with 50 active runs; `/websets/*` 20 QPS. Exceeding returns 429 (honor `Retry-After`, else exponential backoff). Other errors: 400 bad filter or schema, 401 key, 402 out of credits, 503 retry. Error body has `requestId, error, tag`. The `x-exa-queued` response headers indicate server-side queueing.

## 5. Recommended integration

Principle: Exa is a read-only prior source. It never touches the eval budget counter, and it never sees held-out data. All Exa traffic goes through one module (suggested `wsp_swarm/exa_priors.py`) so caching, cost logging, and fairness rules live in one place.

### 5.1 Which role calls what

| Role | When | Call | Parameters |
|---|---|---|---|
| Literature pre-pass (run once before the budget starts) | per catalog component (37) | `search` | `type="auto"`, `category="publication"`, `num_results=5`, query built from `trained_on` ("MMLU-Pro biology" or the BBEH task name plus "language model benchmark"), `contents={"highlights": {"query": <trained_on>, "max_characters": 1200}}`, `end_published_date` pinned to run start |
| Literature pre-pass | topical priors (about 8 fixed questions: mask merge interference, late-layer vs early-layer factual storage, task arithmetic limits, sparse delta composition, pruning and instruction following, layer-wise ablation in Llama) | `search` with `type="deep-lite"` and `output_schema` (claims with `claim, layer_hint, model, strength`) | `category="publication"`, `system_prompt` listing preferred topics |
| Analyst | when summarizing the capability map | no live call; reads the frozen `priors.json` pack (cache hit) and may attach one cached citation per overlap finding | none |
| Planner (K parallel) | when forming a hypothesis | `search` `type="lite"` or `"auto"` with `num_results=3`, `contents={"highlights": True}`; query = hypothesis in one sentence plus component names. `find_similar(url)` only on a paper the planner already cited. | the planner stores `citations: [{url, title, quote}]` on the hypothesis; the judge rejects a hypothesis whose citation URL is not in the cache (ground truth) |
| Judge | frontier updates | none. Judge ranks on search-split scores only. Citations are logged, not scored | |
| Literature agent | between runs, not inside a run | `agent.runs.create(effort="medium", output_schema=notes_schema, previous_run_id=<last>)` or `search(type="deep")` with `start_published_date` = last refresh date. Writes `notes/literature.md` and refreshes `priors.json` | `budget.maxCostDollars=2` |
| Collapse explainer (optional) | after a collapsed merge is logged | `answer(query=<observed collapse as a sentence>, model="exa")` | cached; used to annotate the log with possible explanations, not to change scores |

Do not use Websets or monitors. Do not call Agent API from a planner step (async, 5 QPS cap, variable runtime).

### 5.2 Cache design

- Key: `sha256` of canonical JSON `{endpoint, sdk_version, params}` with params sorted and the volatile fields removed (`user_identifier`). Include `end_published_date` and `max_age_hours` in the key because they change the result set.
- Value: `dataclasses.asdict(response)` plus `cost_dollars` and a `fetched_at` timestamp. Store the raw JSON, not SDK objects.
- Location, two tiers:
  1. Git-committed `exa_cache/` directory of JSON files (`<key[:2]>/<key>.json`) for everything the pre-pass produces. This is the source of truth for reproducible runs and for judges, and it ships with the repo.
  2. A `modal.Dict` named `exa-cache` (key to JSON string) for in-run planner lookups when the swarm runs on Modal, so parallel planner containers share hits. Values stay small (highlights only, under about 10 KB). Use a `modal.Volume` instead if a pack exceeds the Dict item size limit (UNVERIFIED limit; check Modal docs at implementation time). On a miss: call Exa, write to the Dict, and append to the local JSONL on the orchestrator.
- Modes controlled by `EXA_MODE`: `live` (read through cache, write on miss), `replay` (cache only; a miss raises, never calls the network), `off` (no priors; ablation arm). Every graded or demo run uses `replay`.
- Always pass `contents={"maxAgeHours": -1}` style cache-only fetching where the call allows it, but note that `lite` is cache-only by design and an `auto` search still ranks live; results can change between days, which is why our cache, not Exa, is the reproducibility layer.
- Append every Exa call to `experiments.jsonl` as its own row type: `{"timestamp", "agent", "type": "exa", "endpoint", "query", "cache_hit", "cost_dollars", "eval_count_unchanged": true}`. These rows never increment the eval counter.
- Seeds: Exa has no seed parameter and does not document determinism. Fixed seeds apply to our agents and compile; the cache provides determinism for the Exa part.

### 5.3 Keeping the budget comparison fair

1. The eval counter counts only `eval()` calls. Exa calls are logged separately and reported as a cost line, never as free or hidden.
2. Freeze the priors pack before evaluation 1. Compute `priors_sha = sha256(priors.json)` and write it into the run manifest. The swarm, the "swarm without Exa" ablation, and any replay use the same pack hash or `off`.
3. Run the ablation. The evolutionary baseline has no language-model prior, so report three curves at equal eval counts: evolutionary, swarm with Exa off, swarm with priors. If swarm-with-Exa beats swarm-off, the gain is attributable to literature; if not, say so.
4. Bound leakage. Exa results may mention MMLU-Pro or BBEH items, or the Deletion Tuning paper. Block benchmark question pages by passing `exclude_domains` for dataset mirrors (for example `huggingface.co/datasets`, `*.kaggle.com`) in planner and pre-pass queries, never include benchmark answers in queries, and never query with held-out-split text. Planners get citations and highlights only, not benchmark data.
5. Same pack for all planners in a round, so planner differences come from their reasoning, not from different retrieval luck.
6. Cap planner Exa calls per round (for example 1 call per hypothesis, K per round) so retrieval spend is comparable across arms and logged.

### 5.4 Per-run cost estimate (list prices; assumes a cold cache)

Assumptions: pre-pass of 37 component queries plus 8 topical deep-lite queries; a run with K=4 planners and 12 rounds (48 planner searches, half cache hits after the first three rounds); 1 collapse explanation per 10 hypotheses; one literature agent refresh.

| Item | Count | Unit | Cost |
|---|---|---|---|
| Component pre-pass, `auto` + highlights, 5 results | 37 | $0.007 + 5 x $0.001 = $0.012 | $0.44 |
| Topical deep-lite with `outputSchema` | 8 | $0.012 + contents | about $0.15 |
| Planner `lite` searches, 3 results with highlights | 24 cold | $0.004 + $0.003 = $0.007 | $0.17 |
| Collapse `answer` calls | 5 | $0.005 | $0.03 |
| Literature agent, `medium` | 1 | $0.10 | $0.10 |
| Total, first run | | | about $0.9 |
| Warm replay run | | | $0.00 |

Even with a 3x margin the cold run costs under $3, inside the $10 free tier. At 5 QPS for deep and 10 QPS for search, K=4 parallel planners are well under the limits; add a client-side semaphore of 4 and retry with backoff on 429.

### 5.5 Failure behavior

- On 402, 429 after retries, 5xx, or a missing key: the wrapper returns `{"results": [], "error": tag}` and planners proceed with no citation (their hypothesis is marked `ungrounded`). The swarm must never block on Exa.
- Treat `warnings` as logs and `notices` as text to show the model, per the docs.
- Treat all Exa page text as untrusted data; planners only receive highlights and titles, wrapped as quoted evidence, never as instructions.

## 6. Live smoke test to run once a key exists

Run from the repo root with `EXA_API_KEY` set. Total cost under $0.05. Record the printed output into this file under an "observed" heading.

```python
import dataclasses, json
from exa_py import Exa
exa = Exa()

# 1. publication search with highlights; check category works and read cost shape
r = exa.search("task vector merging interference sparse masks", type="auto",
               category="publication", num_results=3,
               contents={"highlights": {"max_characters": 800}})
print([x.url for x in r.results]); print(r.cost_dollars); print(r.results[0].entities)
json.dumps(dataclasses.asdict(r), default=str)   # confirm the cache serialization round-trips

# 2. lite with category and date filters (confirms what lite rejects)
r = exa.search("MLP layers store factual knowledge Llama", type="lite", category="publication",
               num_results=3, end_published_date="2026-11-14T00:00:00Z", contents={"highlights": True})
print(r.cost_dollars)

# 3. deep-lite structured output: check attribute names of r.output and grounding
r = exa.search("Which transformer layers store factual associations?", type="deep-lite",
               category="publication",
               output_schema={"type": "object", "required": ["claims"], "properties": {"claims": {
                   "type": "array", "items": {"type": "string"}}}})
print(type(r.output), r.output); print(r.cost_dollars)

# 4. find_similar from the Deletion Tuning paper
s = exa.find_similar("https://arxiv.org/abs/2611.10866", num_results=5, exclude_source_domain=True)
print([x.title for x in s.results]); print(s.cost_dollars)

# 5. answer with a schema, and confirm model="exa-pro" is accepted
a = exa.answer("What does prior work say about merging sparse task-specific weight deltas?")
print(a.answer, len(a.citations), a.cost_dollars)

# 6. repeatability: run call 1 twice and diff the URL lists (decides how much we rely on the cache)
```

Questions the smoke test settles: whether `publication` returns arXiv-sourced results for these topics; the exact `output.content` and `output.grounding` attribute names in the SDK; whether highlights add cost on top of the $7 / 1k `auto` rate; find_similar pricing; whether two identical searches return identical results; whether `lite` rejects `category` and date filters in practice (the docs say it accepts them).

## 7. Unverified list

- No live call was made; response shapes come from the spec and SDK source.
- `findSimilar` pricing and exact SDK `Category` handling.
- Extra cost of `outputSchema`; whether highlights are included in the search base price.
- `/answer` models `exa-research` and `exa-fast` (docs say yes, SDK literal says no).
- Determinism of identical queries over time.
- Modal Dict per-item size limit.
- The per-team QPS values apply to the plan the hackathon key lands on.
