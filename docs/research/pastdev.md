# past.dev integration guide for the RL environment and swarm (part 3)

Sources: the past.dev docs as raw markdown (`https://past.dev/docs/llms-full.txt`, plus the per-page `/llms.txt` variants for quickstart, setup, authentication, concepts, how-it-works, access-control, sending-data, webhooks, api-reference, debugging, errors) and live calls made on 2026-11-14 against `https://api.past.dev/api/v1` with the project key from `.env`. Everything marked **[live]** was observed; **[docs]** comes from the docs only; **[unverified]** was not checked.

## 1. Summary

- past.dev stores timestamped text ("data points"), derives short structured memories from it (claims, observations, rules, mentions), and returns ranked, source-backed documents from `POST /recall`. Recall returns no generated text.
- There is no `/completion` route [live: 404 on `/completion`, `/completions`, `/complete`, `/chat/completions`]. The word "completion" in the docs (authentication page) means ingestion completion. There is, however, an undocumented-in-the-REST-reference `POST /answer` route [live] that runs a grounded LLM answer over recall results. It is not a general planner LLM (section 5).
- Ingest to searchable text: about 4 s. Ingest to derived memory (claims, observations) recallable: about 10 to 13 s [live].
- Scoping: only audiences and identities scope reads. `metadata` never filters. A data point sent without `audience` is visible to every identity in the project, so every send must name a run-scoped audience.
- Delete works: `POST /data-points/delete` removes data points and the memory resting only on them at once [live]. Deleted ids can never be reused.
- No official Python SDK for the API (only a Hermes plugin and an MCP server). Use `requests` or `httpx` (section 7).

## 2. Verified routes

Base URL `https://api.past.dev/api/v1`. Auth header `Authorization: Bearer $PAST_API_KEY`. Key kinds: project key `past_sk_` (ingest, recall, answer, data points, audiences, identities, usage) and management key `past_mk_` (projects and keys only; reads no memory). A project key selects one project, so no project id appears in bodies. Every response carries `X-Request-Id`. Browser calls are refused by CORS, so call from the server side only.

| Method and path | Purpose | Notes |
|---|---|---|
| `POST /ingest` | One data point. `202 queued`, or `200 completed, unchanged:true` if id, content, label, metadata, timestamp all match. | Fields: `content` (required), `id`, `label`, `timestamp` (ISO 8601), `metadata` (JSON, never filters), `audience` (slug), `identity` (author record only, does not scope), `idempotencyKey`. [live] |
| `POST /ingest/batch` | 1 to 1,000 items, 16 MiB, atomic. Returns one `ingestionId` for the batch plus per-item status. | Each item takes the single-send fields except `idempotencyKey`; one `idempotencyKey` for the batch. A source id may appear once per batch. [docs] |
| `POST /ingest/estimate` | Prices an import before sending. Writes nothing, free. | [docs] |
| `GET /ingest/{ingestionId}` | `status`: `processing`, `completed`, `failed`. Also `settled`, `blocked`, `readiness{raw,comprehension,consolidation,failed}`, `sources{current,deleted,superseded}`, `failure`. | Polls are free and outside the rate limit. [live] |
| `GET /ingest/{ingestionId}/explain` | Phase (`queued`..`indexed`, `failed`), per-item `searchable` and memory counts. | [docs] |
| `POST /ingest/{ingestionId}/retry` | Re-run a failed send, free. `409` if not stopped. | [docs] |
| `DELETE /ingest/{ingestionId}` | Forget every data point of a send. `204`. | [docs] |
| `DELETE /data-points/{sourceId}` | Delete one (percent-encode id). `202 {operationId, ids}`. | [docs] |
| `POST /data-points/delete` | Body `{"ids":[...]}`, 1 to 1,000. All-or-nothing: any unknown id gives `404 data-point-not-found` and deletes none. | [live] `202 {"operationId":..., "ids":[...]}`; recall returned zero results right after. |
| `POST /recall` | Ranked documents. | Below. |
| `POST /answer` | Grounded answer plus citations plus the same `results`. | Below. Not in the REST reference. |
| `PUT /audiences/{slug}` | Create or replace. `kind` `fixed` (with `identities`) or `rule` (with `rule`). `201` create, `200` replace. | [live] |
| `GET /audiences`, `GET /audiences/{slug}`, `DELETE /audiences/{slug}` (`204`), `POST /audiences/{slug}/recompute` | | Deleting an audience erases no memory; it becomes unreadable forever. [live 204] |
| `PUT /identities/{id}`, `PATCH`, `POST /identities/bulk` (1,000 max), `GET /identities`, `GET /identities/{id}` (returns reached audience slugs), `DELETE /identities/{id}` (`204`, memory stays) | Traits are string key/value. | [docs] |
| `GET /usage` | Credit balance and this project's month. Free, outside rate limit. | [live] |
| `GET /trace/roots`, `/trace/nodes/{id}`, `/trace/lineage/{rootId}`, `/trace/timeline/{rootId}` | Provenance of a memory. | [docs] |
| `GET /requests`, `/requests/histogram`, `/requests/{id}` | Request log. | [docs] |
| `POST /projects`, `GET /projects`, `POST /projects/{slug}/keys`, `DELETE /projects/{slug}/keys/{keyId}` | Management key. | [docs] |
| `POST /webhooks/{id}/{token}` | Inbound only (other systems post into past). Plan feature: not on Pay as you go. | [docs] |

### Ingest example [live]

```bash
curl -X POST https://api.past.dev/api/v1/ingest \
  -H "Authorization: Bearer $PAST_API_KEY" -H "Content-Type: application/json" \
  -d '{"id":"wsp-smoke-1794698437","label":"wsp smoke test",
       "timestamp":"2026-11-14T23:20:37Z","audience":"wsp-smoke",
       "content":"Union of biology and chemistry masks restricted to layers 4-15 scored 0.41 on the search split (stub)."}'
# 202 {"projectId":1,"ingestionId":"240267f9-...","sourceId":"wsp-smoke-1794698437","status":"queued","unchanged":null}
```

Status poll [live], completed shape:

```json
{"status":"completed","settled":true,"blocked":false,
 "readiness":{"raw":0,"comprehension":0,"consolidation":0,"failed":0},
 "sources":{"current":1,"deleted":0,"superseded":0},"failure":null}
```

### Recall

Request fields [docs, live]: `query` (required), `identity` (required, 1 to 200 chars), `level` (`low` 8,000 tokens default, `medium` 16,000, `high` 24,000, `extra-high` 32,000) or `maxTokens` (never both; `400 recall-level-conflict`), `limit` (1 to 200, default 100), `queryTimestamp`, `occurredFrom`, `occurredTo`, `sort` (`relevance` or `chronological`). The plan caps the budget at 8,000 tokens on Pay as you go and Flex, 32,000 on Enterprise. This account's plan was not shown by the API [unverified], so assume the 8,000 ceiling.

Response [live], trimmed:

```json
{"asOf":"2026-11-14T23:21:08Z","usedEvidenceTokens":359,
 "results":[
  {"id":"da435fc8-...","rank":1,"occurredAt":"2026-11-14T23:20:54Z",
   "content":"Deleting overlap between biology and chemistry masks in layers 12-15 dropped instruction following on the search split (stub) (outcome). Because ...",
   "artifact":{"id":"f399415c-...","kind":"observation","occurredAt":"..."},
   "sources":[{"sourceId":"wsp-smoke-1794698454-b","occurredAt":"...","label":"wsp smoke test","excerpts":["Collapse lesson: deleting overlap ..."]}]},
  {"rank":2,"artifact":{"kind":"source"},"content":"<the raw text I sent>","sources":[{"sourceId":"...","excerpts":[]}]},
  {"rank":3,"artifact":{"kind":"claim"},"content":"Union of biology and chemistry masks restricted to layers 4-15 search split score: 0.41."}
 ]}
```

Observations from the live run:
- One 1-sentence data point produced about 8 documents: the `source` (raw text), one or two `claim`s, an `observation`, and `mention`s. Budget your prompt for that expansion; filter on `artifact.kind` (drop `mention`, optionally drop `source` when its derived memory is present).
- Numbers survive: "scored 0.41" became a `claim` with value 0.41, and "dropped ... to 0.02" became a claim "instruction-following score: 0.02". Write results as plain sentences with the score in the text.
- Recall latency about 1.0 to 1.3 s per call from this machine [live].
- An identity outside the audience got `{"results":[],"usedEvidenceTokens":0}` and no error [live]. An unknown identity id also returns an empty page, not a 404.
- `pending` appears in the response while sends are still processing, listing what is not yet in memory [docs].

### Supersession [live]

Ingested "math task vector, weight 0.5, layers 0-7 scored 0.37", then (new id, later timestamp) "now scores 0.44 ... re-evaluated with a fixed seed". The next recall returned the old claim "score on search split: 0.37" with `artifact.supersededAt: "2026-11-14T23:22:01Z"` and the new claim "search split score: 0.44" with no `supersededAt`. Raw `source` documents are never superseded; both texts stay. `queryTimestamp` before the second timestamp reads the older state [docs]. Supersession is a model judgement over claims of the same subject and property, so word the property consistently ("search split score") [live evidence from one case only; reliability across many near-duplicate programs is unverified].

### Answer route (undocumented in the REST reference)

`POST /answer` with `{"query": "...", "identity": "..."}` (the field is `query`, not `question`; `question` returns `400 recall-query-required`) [live]. Response fields: `answer`, `disposition` (`answered`; absent or null on an off-topic question), `citations[{documentId, sourceIds}]`, `answerer{model, reasoningEffort, protocol, promptSha256}`, `usage{promptTokens, completionTokens, cachedPromptTokens, totalTokens, costUsd}`, `timings{recallMs, answerMs, totalMs}`, plus the same `asOf`, `usedEvidenceTokens`, `results` as recall. Observed model: `openai/gpt-5.6-luna`, reasoning effort `none`. A first probe that also sent an extra `prompt` field returned `502 answer-generation-failed` once; I did not re-test whether the extra field or a transient fault caused it [unverified]. The docs mention an `answer` tool on the end-user MCP server ("a grounded answer over the same evidence, with its citations, or an abstention") and say an "answer" costs 1 credit on Pay as you go.

## 3. Live-check results

Project `default-project`, audience `wsp-smoke` (fixed, identity `wsp-smoke-planner`).

| Check | Result |
|---|---|
| Ingest one data point, id `wsp-smoke-1794698437`, label "wsp smoke test" | `202 queued`, about 1.1 s round trip |
| Poll to completion | `processing` (raw=1) then `processing` (comprehension=1) then `completed`, `settled:true`. A script with a recall in the loop saw completion at 13.4 s after send. Another point, with a recall every 0.7 s, became recallable as raw text at 3.9 s and as derived claims at 10.3 s. A third (a supersession send) polled `completed` at 11.9 s. So: text about 4 s, full memory about 10 to 13 s. (My separate "t_completed=4.8 s" reading in that script was a bug: it polled the ingestion id of an unchanged re-send, which is `completed` at once. Ignore it.) |
| Recall "what happened when overlap was deleted in late layers" | Rank 1 observation about the layers 12-15 collapse, rank 2 the raw source, then claims. Before completion, a recall about 2 s after send already returned the first point's documents (it was complete), and the not-yet-complete point showed as raw text only. Citations carry the right `sourceId`. |
| Scope check | Recall as `wsp-other` (not in the audience): empty results, `200`. |
| `completion` | No such route (404 on four spellings). `/answer` works (section 2). |
| Delete | `POST /data-points/delete` with 4 ids: `202`; recall right after: zero results. Re-sending a deleted id: `409 data-point-deleted` ("a deleted id is never reused"). Audience and identity deleted, `204`. |
| Credits | Balance before about 549,999 (50,000 included plus 499,999 bought), after the whole session 549,994 or lower. The project line showed 15 credits for 4 ingests and 11 recalls or answers, consistent with 1 credit per ingest of under 350 bytes and 1 credit per recall or answer. Total spent in this research: about 15 credits plus the earlier 1. |

All test data points, the test audience and the test identity were deleted. The key was never printed.

## 4. Semantics that shape the design

- **Scope**: a recall reads project-visible memory (anything sent with no `audience` or `audience:"project"`) plus the audiences of that identity. On a re-send, `audience` resets to project-visible if omitted, so every send and every re-send must carry the audience. The `identity` field on ingest only creates a directory record; it does not scope.
- **Audience first**: a data point that names an unknown slug fails `400 audience-unknown`, so create audiences (free, `PUT`) before ingesting.
- **Membership lag**: audience or trait changes apply to recall after a few seconds; `POST /audiences/{slug}/recompute` applies it now [docs]. Create run audiences and wait or recompute before the first recall.
- **Visibility intersection**: a memory is readable only if the reader can read every data point it rests on. Memories never mix across audiences, so a role-private audience cannot leak into others.
- **Ordering**: the project's sends become memory one after another, oldest first. Parallel sends from K planners queue behind each other [docs]. Prefer a few batches to many singles.
- **Projects**: 1 project on Pay as you go, 5 on Flex [docs]. This account plan is [unverified]. Use audiences, not projects, for per-run isolation.
- **Rate limit**: 500 requests per minute per key on Pay as you go, `429` with no body when exceeded; double the pause on each further 429 [docs]. Polls and `/usage` are exempt.
- **Retries**: `idempotencyKey` makes sends replay-safe; send explicit `timestamp` when using it. `5xx`: retry with the same key. `402 out-of-credits`: stop.
- **Webhooks**: inbound only and not on Pay as you go. There are no outbound "ingestion done" webhooks. Poll `GET /ingest/{id}`.

## 5. Can `answer` be the planner LLM?

Short answer: no for planning, optional for synthesis.

- It is a retrieval-grounded answerer (model `openai/gpt-5.6-luna`, reasoning effort none, about 2.6k prompt tokens and about 1.3 s per call, cost about 0.0007 to 0.001 USD per call as reported in `usage.costUsd`). It abstains on questions the memory does not support: asked "What is the capital of France?" it returned no answer [live].
- With evidence in memory it did produce three structured candidate programs as JSON-like objects (union of masks, layer ranges, reasons that cite the stored collapse). So it can restate lessons as proposals, but it sees only recalled evidence, not the catalog or the environment prompt, has no system prompt or tool use that I could find, and its output format is not controllable [unverified: no documented parameters beyond `query` and `identity`].
- It is therefore unsuitable as the K parallel planners. Planners need an LLM that reads the catalog, signatures and history. The project has no Anthropic key, so planner and judge LLMs have to come from whatever key the team already uses elsewhere. (Not decided here.)
- Useful role: a "lessons digest" call that turns raw recalled collapses into 3 to 5 sentences that you paste into a planner prompt. This costs 1 credit and a provider-side LLM call per use, and is inside the same memory scope. Treat the route as undocumented and possibly unstable.

## 6. Recommended integration

### 6.1 Scoping for fairness (the load-bearing part)

Goal: a benchmark run can read only what that run wrote, plus optionally an explicitly counted prior.

1. **One run id per run**, e.g. `r20261114-s7-swarm` (condition, seed). Ids and audiences embed it.
2. **Audiences per run** (all `fixed`, created with `PUT` before the first ingest):
   - `<run>-lessons`: members `<run>:analyst`, `<run>:planner-1..K`, `<run>:judge`. Everything every role may read: step results, collapse lessons, frontier verdicts, capability map. This is the shared audience.
   - `<run>-judge`: member `<run>:judge` only. Optional private notes (budget bookkeeping).
   - Slug limit is 40 chars, lowercase letters, digits, dashes, so keep run ids short (`s7-swarm-a1` form) and slugs like `r-s7a1-lessons`.
3. **Identities per run**: `<run>:planner-3`, `<run>:judge`, `<run>:analyst`. Identity ids are free strings up to 200 chars. Identities are created by being listed in a fixed audience.
4. **Never send without `audience`.** Wrap ingest in one function that raises if the audience is missing or is `project`. Never ingest "global" notes into the project-visible set; the default audience reaches every identity, including every benchmark run.
5. **Start-of-run guard**: recall as `<run>:planner-1` with a generic query ("mask score search split") before the first step and assert zero results. This proves no project-visible leakage and no stale audience.
6. **Conditions to report**:
   - A. swarm with memory, cold start (fresh run id). Fair comparison against the evolutionary baseline at equal eval budget.
   - B. swarm without memory (same code, `PAST_ENABLED=false`) as the ablation that shows memory helps.
   - C. optional warm start: planners also read audience `warm-<prior-run>-lessons`. Counts as using the earlier run's evals, so charge that run's evals to the budget (or report separately as "transfer"). Never mix C into A.
7. **Cleanup**: at the end of a run, `POST /data-points/delete` with the ids saved in `experiments.jsonl` (up to 1,000 per call), then delete the run audiences and identities. Deleting an audience alone erases no memory. A deleted id cannot be reused, so never reuse a run id after cleanup; make run ids unique (include a timestamp) and make step ids include the run id.
8. `queryTimestamp` and `occurredFrom/To` are not isolation. They are useful only for ablations (for example reading as of evaluation 50).

### 6.2 What each role ingests

Common rules: `id` is `<run>:<kind>:<n>` (no `/`), `timestamp` is the real wall-clock UTC time of the event, `audience` is `<run>-lessons` unless noted, text is one self-contained plain-English sentence or short paragraph with the numeric score in it, and `metadata` carries structured fields (eval index, program hash, split) for your own bookkeeping only (it never filters recall, but extraction reads it as context). Label is a short title.

| Role | When | id | label | content (example) |
|---|---|---|---|---|
| Step result (env, after each eval) | every `step()` | `<run>:step:<evalIdx>` | `Step <evalIdx> <ok/collapse>` | "Eval 37, planner 2, hypothesis: restrict bio|chem union to layers 4-15 keeps accuracy. Program: (bio | chem).restrict(4..15). Search split combined score 0.41 (bio 0.52, chem 0.47, instruction following 0.61). Outcome: improved over frontier best 0.38." |
| Collapse (env or judge, when score falls below a collapse threshold, for example under 50 percent of base or an instruction-following drop) | same step, extra point | `<run>:collapse:<patternKey>:<n>` | `Collapse: <pattern>` | "Collapse: deleting the overlap of bio and chem masks in layers 12-15 dropped instruction following from 0.61 to 0.02 on the search split (eval 41). Lesson: do not delete overlap in late layers." |
| Judge verdict (each round) | after ranking | `<run>:verdict:<round>` | `Round <r> frontier` | "Round 9 frontier top-3: program A 0.44, B 0.41, C 0.40. Killed branches: late-layer overlap deletion (3 collapses), tv(math, 1.0) on layers 8-15. Budget used 36 of 200." |
| Analyst capability map | once at start, then when a new probe returns | `<run>:capmap:<component>` (replace in place by re-sending the same id with the same audience) | `Capability: <component id>` | "Component mask_mmlupro_biology_llama1b: effect strongest on biology and health, overlaps 38 percent with chemistry in layers 6-11, weak effect on layers 0-3." |
| Planner proposal (optional) | each proposal before eval | `<run>:plan:<round>:<k>` | `Plan r<round> p<k>` | "Planner 3 proposes union of bio and chem restricted to layers 4-15. Hypothesis: skills live in middle layers." |

Point-per-step keeps step ids immutable and makes deletion exact. Roughly 300 to 600 bytes each, so 1 to 2 credits per point.

### 6.3 What each role recalls, and at what level

Recall costs 1 credit, about 1.2 s, and returns ranked documents that include derived memories. Use `limit`, then drop `mention` kinds before building the prompt.

| Role | Before | identity | query | level |
|---|---|---|---|---|
| Planner k | each planning round | `<run>:planner-k` | one query per angle the planner intends: "what happened with <components in the draft> in layers <range>" and "collapses involving <component or operation>". Two to three recalls per round are enough. | `low` (8,000 tokens, `limit` 30) for the angle queries; `medium` once at round start for "best programs so far and what failed" (ceiling 8,000 on this plan anyway) |
| Judge | before ranking a round | `<run>:judge` | "collapses and frontier changes in the last rounds", `sort: chronological`, `occurredFrom` = round start | `low` |
| Analyst | before updating the map | `<run>:analyst` | "capability of <component>" and "overlap with <component>" | `low` |
| Environment (observation builder) | each episode step (optional) | `<run>:planner-1` | "recent best programs" | `low` |

The catalog and the full `experiments.jsonl` history still go straight into the planner prompt from local files (they are exact and free). past.dev supplies retrieval of relevant lessons and the as-of reads. Report this split honestly in the writeup.

### 6.4 Collapsed merges: write and supersede

1. On every collapse, ingest a collapse point with a pattern key (e.g. `late-overlap-del`) and a general statement of the lesson, in addition to the step point.
2. Do not edit old collapse points. When later evidence refines the lesson ("deleting overlap in layers 12-15 is safe if layers 4-8 are kept"), ingest a new point with a new id, later timestamp and the same property wording ("lesson late-layer overlap deletion"). Observed behaviour: a re-evaluated number superseded the older claim (`supersededAt` set) while both raw texts stayed [live, one case]. At read time, ignore results whose `artifact.supersededAt` is set.
3. When one lesson is stale and wrong (for example the original collapse was caused by a compile bug), delete it by id with `POST /data-points/delete` rather than relying on supersession. Deletion removes the memory derived only from it at once.
4. If a single canonical lesson per pattern is wanted, use a fixed id `<run>:lesson:<patternKey>` and re-send it with changed content every time the lesson changes (replaces the old text and its memory). It costs an ingest and the 10 to 13 s processing again, and it keeps the audience only if you send `audience` again.

### 6.5 Handling ingestion latency in a fast loop

Memory is ready about 10 to 13 s after send (text searchable in about 4 s), serialized per project. The loop (compile, eval on a 1B model, judge) takes longer than that per round, so:

- Fire-and-forget ingest from a small background thread, keeping the `ingestionId` and ids in `experiments.jsonl`.
- Batch: one `POST /ingest/batch` per round (K step results, collapses and the verdict in one batch). One batch means one queue slot.
- At round start, before the planners recall, wait for the previous round's batch: poll `GET /ingest/{ingestionId}` every 1 to 2 s (the docs suggest 2 s for the first minute) with a timeout of about 30 s. If it times out, proceed and log `memory_lag=true` so the swarm and the plots can show it. Check `settled` for the whole project.
- The recall response includes `pending` when sends are still processing; log it.
- Do not poll inside the equal-budget accounting: polls and recalls are not eval calls, but report their count and credits in the writeup.
- If an ingestion ends `failed`, call `POST /ingest/{id}/retry` (free); if `failure.code` is `content-rejected`, change the text.
- Keep a local fallback: if past.dev returns 5xx or `402`, the environment must keep running with the local history only and mark the run `memory_degraded`.

### 6.6 Credit estimate per run

Assume a run with an eval budget of 200, K=4 planners, so 50 rounds. Pay as you go pricing observed: 1 credit per ingested data point of under 350 bytes (more for longer text, 1 per 350 bytes), 1 credit per recall or answer.

| Item | Count | Credits |
|---|---|---|
| Step points (1 per eval, about 450 bytes) | 200 | about 400 |
| Collapse points (assume 20 percent of evals, about 300 bytes) | 40 | 40 |
| Judge verdicts (per round) | 50 | about 50 to 100 |
| Analyst capability map (about 15 components) plus 5 re-sends | 20 | 20 |
| Planner recalls (3 per planner per round) | 600 | 600 |
| Judge and analyst recalls | about 100 | 100 |
| Lessons digest via `/answer` (optional, 1 per round) | 50 | 50 |
| Total | | roughly 1,200 to 1,400 credits |

Balance at the time of the check was about 549,994 credits, so a run uses about 0.25 percent of that. A swarm with-memory versus without-memory pair plus three seeds is about 8,000 credits. The ingest side is dominated by the number of evals; the recall side by planner count. Per-run recalls are about 5 per round. Rate limit: about 10 requests per round, far below 500 per minute. These are estimates from observed unit prices, not measurements of a full run [unverified at scale].

## 7. Minimal Python client (plain `requests`)

```python
import os
import time
import requests

API = "https://api.past.dev/api/v1"


def headers():
    return {
        "Authorization": "Bearer " + os.environ["PAST_API_KEY"],
        "Content-Type": "application/json",
    }


def put_audience(slug, name, identities):
    body = {"kind": "fixed", "name": name, "identities": identities}
    r = requests.put(API + "/audiences/" + slug, headers=headers(), json=body)
    r.raise_for_status()
    return r.json()


def ingest_batch(items, key):
    # every item must carry an audience that is not "project"
    for item in items:
        if not item.get("audience") or item["audience"] == "project":
            raise ValueError("refusing to ingest without a run audience")
    body = {"idempotencyKey": key, "items": items}
    r = requests.post(API + "/ingest/batch", headers=headers(), json=body)
    r.raise_for_status()
    return r.json()["ingestionId"]


def wait_for(ingestion_id, timeout=30.0):
    start = time.time()
    while time.time() - start < timeout:
        r = requests.get(API + "/ingest/" + ingestion_id, headers=headers())
        r.raise_for_status()
        status = r.json()["status"]
        if status == "completed":
            return True
        if status == "failed":
            requests.post(API + "/ingest/" + ingestion_id + "/retry", headers=headers())
        time.sleep(1.0)
    return False


def recall(identity, query, level="low", limit=30):
    body = {"query": query, "identity": identity, "level": level, "limit": limit}
    r = requests.post(API + "/recall", headers=headers(), json=body)
    r.raise_for_status()
    results = r.json()["results"]
    keep = []
    for doc in results:
        if doc["artifact"].get("supersededAt"):
            continue
        if doc["artifact"]["kind"] == "mention":
            continue
        keep.append(doc)
    return keep


def delete_points(ids):
    for start in range(0, len(ids), 1000):
        chunk = ids[start:start + 1000]
        r = requests.post(API + "/data-points/delete", headers=headers(), json={"ids": chunk})
        r.raise_for_status()
```

A `429` with no body needs a pause and retry; add a short loop in the real client.

## 8. Not verified, and risks

- Plan tier and therefore the recall token ceiling, project count and webhook availability (the API shows balances only).
- `/answer`: undocumented in the REST reference; parameters beyond `query` and `identity`, rate limits, price per call and long-term availability. One 502 seen with an extra `prompt` field, cause unknown.
- Supersession reliability on many near-identical program results (tested on one pair only).
- Throughput of a large batch (queue is serial per project; I measured single points only).
- Cross-audience behaviour when one identity belongs to several audiences was read from the docs only, not tested (the shared lessons design depends on it; the docs state recall returns memory of every audience the identity reaches).
- Recall quality at scale (hundreds of near-duplicate step results) was not tested; the planner prompt should therefore always also carry the local top-k history.
- `settled` is project-wide, so another run in the same project can delay a `settled` check; wait on the specific `ingestionId` instead.
