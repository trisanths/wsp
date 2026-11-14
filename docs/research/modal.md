# Modal integration guide for part 3 (RL environment and swarm)

Researched 2026-11-14 against modal.com/docs and a live client (modal 1.6.4, installed into the repo `.venv`, Python 3.12.13). Everything marked VERIFIED LIVE was run against the workspace the `.env` tokens belong to. Anything marked UNVERIFIED was read from docs or recalled and not exercised.

## 0. Live check results (summary)

| Check | Result |
|---|---|
| Install | `uv pip install --python .venv modal` worked. Client 1.6.4, `Requires-Python >=3.10,<3.16`, so local Python 3.12 (the repo venv) and 3.13 are fine. |
| Auth via `MODAL_TOKEN_ID` / `MODAL_TOKEN_SECRET` in `.env` | Works. `modal profile current` prints `rexheng`; `modal app list` returns apps. |
| Workspace the tokens belong to | **`rexheng`**. `modal profile list` with the env vars set prints `Using rexheng workspace based on environment variables`. |
| Env vars beat the profile | VERIFIED LIVE. With a valid profile in `~/.modal.toml` and bogus `MODAL_TOKEN_ID` / `MODAL_TOKEN_SECRET` (fake strings), `modal app list` failed with `Token not found`. So the env vars win and the profile file is never read or written. |
| Same workspace as Trisanth's profile `trisanth`? | **Cannot confirm, and almost certainly not.** `~/.modal.toml` on this machine holds exactly one profile name, `rexheng`; there is no `trisanth` profile here. The `rexheng` workspace contains a volume `merge-repair-vol` and an old stopped app `gpu-check`, and no `wsp-models` volume and no `wsp-compiler` app. So Trisanth's volume and app live somewhere else. `modal_app.py` will create a fresh empty `wsp-models` here via `create_if_missing=True`. |
| Plan and credits of the `rexheng` workspace | UNVERIFIED (no cheap CLI check). The pricing page says Starter has $30/month compute credit, 100 containers, 10 GPU concurrency; Team has $100/month, 5000 containers, 50 GPU concurrency. Assume Starter limits until checked in the dashboard. |
| Scratch cleanup | The Queue, Dict and deployed app named `wsp-scratch-check` were created, tested, then deleted or stopped. `modal queue list`, `modal dict list` are empty. Total spend: a few CPU container-seconds. No GPU used. |

Cross-workspace consequence: a shared Queue or Dict lives in one workspace. Every machine that spends budget must authenticate to that same workspace (same token pair), or call a deployed web endpoint instead (section 6, option C). Two people using their own Modal accounts cannot share a counter.

## 1. Auth, apps, calling functions

Docs: https://modal.com/docs/guide/apps , https://modal.com/docs/guide/trigger-deployed-functions

```bash
# never touches ~/.modal.toml
set -a && . ./.env && set +a
.venv/Scripts/modal profile current            # rexheng
.venv/Scripts/modal run modal_app.py::verify   # ephemeral app, client must stay connected
.venv/Scripts/modal run --detach app.py::main  # keep app alive if the client disconnects
.venv/Scripts/modal deploy app.py              # persistent named app, redeploy updates in place
.venv/Scripts/modal app stop <name> --yes      # --yes needed in non-interactive shells (VERIFIED LIVE)
```

- `modal run --detach`: "Don't stop the app if the local process dies or disconnects" (CLI help, VERIFIED). Caveat (my reading, UNVERIFIED live): the `@app.local_entrypoint` body itself runs on your machine. If it is a long loop it dies with the client. Only remote work already in flight survives. For a long driver, make the driver a Modal Function and `.spawn()` it (section 8f).
- Ephemeral app (`modal run`, `app.run()`): stopped when the client exits or disconnects. Deployed app: lives until `modal app stop`.
- Call a deployed function from plain Python (VERIFIED LIVE: `.map`, `.spawn`, `.get` all worked from a script with no `modal run`):

```python
import modal
f = modal.Function.from_name("wsp-swarm", "run_candidate")   # app name, function name; optional version=, environment_name=
result = f.remote(program_dict, seed)             # blocking single call
fc = f.spawn(program_dict, seed)                  # returns FunctionCall, does not wait
result = fc.get(timeout=600)                      # fc.get(timeout=0) polls, raises TimeoutError if still running
for r in f.map(programs, order_outputs=False, return_exceptions=True):   # fan out, completion order
    ...
for r in f.starmap([(prog, 0), (prog2, 0)]):      # list of arg tuples
    ...
Model = modal.Cls.from_name("wsp-swarm", "Planner")   # for @app.cls
obj = Model(size="7B"); obj.generate.remote("hi")
```

Signatures from the installed client:

- `Function.map(*input_iterators, kwargs={}, order_outputs=True, return_exceptions=False)`
- `Function.starmap(input_iterator, *, kwargs={}, order_outputs=True, return_exceptions=False)`
- `Function.spawn_map(*input_iterators, kwargs={}) -> None` (fire and forget, no result handle)
- `Function.from_name(app_name, name, *, version=None, environment_name=None)`

Each `.map` input is one container call; the number of containers is bounded by `max_containers` (below). The result of `.spawn` is retrievable from another process by `modal.FunctionCall.from_id(id)` (UNVERIFIED here; the retention period of spawned results is documented on the job-queue guide but the fetch did not return it, so treat it as "days, not weeks" and write results to the shared Dict yourself).

Function options (signature of `App.function`, VERIFIED from client): `image, gpu, volumes, cpu, memory, min_containers, max_containers, buffer_containers, scaledown_window, retries, timeout=300, startup_timeout, region, nonpreemptible, enable_memory_snapshot, max_inputs, single_use_containers`. `modal.concurrent(max_inputs=, target_inputs=)` lets one container serve several inputs at once. `modal.Retries(max_retries, backoff_coefficient=2.0, initial_delay=1.0, max_delay=60.0)`.

Timeouts (https://modal.com/docs/guide/timeouts): default 300 s, allowed 1 s to 24 h, `FunctionTimeoutError` raised after retries are exhausted. Retries re-execute the whole input, so any side effect (budget pop, log append) must be idempotent or placed after the risky step. See section 6 for how this interacts with the budget.

## 2. modal.Dict and modal.Queue

Docs: https://modal.com/docs/reference/modal.Dict , https://modal.com/docs/reference/modal.Queue , https://modal.com/docs/guide/dicts-and-queues

### Dict (API as of client 1.6.4)

```python
d = modal.Dict.from_name("wsp-state", create_if_missing=True)   # lazy, persisted by name
modal.Dict.objects.create("wsp-state", allow_existing=True)     # explicit create, no handle
d["k"] = v;  v = d["k"];  d.get("k", default);  "k" in d;  d.pop("k", default)
d.put("k", v, skip_if_exists=True)    # returns True if written, False if key already existed
d.update({"a": 1}, b=2);  d.keys(); d.values(); d.items()   # iterators, unordered
d.len()                                # expensive, capped at 100,000
d.clear();  modal.Dict.objects.delete("wsp-state", allow_missing=True)
modal.Dict.ephemeral()                 # context manager, anonymous, dies with the block
modal.Dict.from_id(d.object_id)        # pass an id into a function instead of a name
```

- Values: anything cloudpickle can serialize. Keys: use plain strings or ints (cloudpickle is not deterministic, so exotic keys can silently miss).
- Persistence: each entry expires after 7 days of inactivity (no reads or writes), written to durable storage. Legacy Dicts made before 2025-05-20 expire 30 days after last add and live in server memory; new `from_name(create_if_missing=True)` Dicts are the durable kind.
- Atomicity: the docs state none. VERIFIED LIVE:
  - Naive read-modify-write is **not safe**. 8 threads, each doing `v = d["n"]; d["n"] = v + 1` five times, expected 40, got **6**. Lost updates are severe.
  - `d.put(key, v, skip_if_exists=True)` **is an atomic claim**. 8 threads each tried 40 ticket keys; exactly 40 claims succeeded, all unique.
- No increment, no compare-and-swap, no transaction. `skip_if_exists` is the only conditional write.
- Per-value and total size limits: UNVERIFIED (not stated on the fetched page). Keep values to small JSON.

### Queue

```python
q = modal.Queue.from_name("wsp-budget", create_if_missing=True)
q.put(v);  q.put_many([..])                         # block=True waits if full; use timeout
x = q.get(block=False)                               # returns None if empty (so token 0 must be tested with `is None`)
xs = q.get_many(k, block=False)                      # up to k items, [] if empty
x = q.get(timeout=5)                                 # raises queue.Empty on timeout
q.len(); q.clear(all=True); q.iterate()              # iterate is non-destructive
q.put(v, partition="p", partition_ttl=3600)          # independent FIFO per partition
```

- Limits: 100,000 partitions per Queue; **5,000 items per partition**; 1 MiB per item; 64 MiB per partition.
- Lifetime: each partition is cleared **24 h after the last put** (override with `partition_ttl`). "Best used for communication between active functions and not relied on for persistent storage." Ephemeral Queues are cleaned up when their app completes; a named Queue created with `from_name` persists across apps.
- Atomicity: not stated in docs. VERIFIED LIVE:
  - 40 tokens, 8 threads on 8 separate handles looping `get(block=False)`: 40 popped, 40 unique, none duplicated, none lost (1.5 s).
  - Across **separate containers**: Queue pre-filled with 20 ints, a deployed function with `max_containers=4` doing `get_many(2, block=False)`, called by `f.map(range(12))`. 4 containers ran, 20 tokens handed out, 20 unique, 0 left. Batch reservation (`get_many`) is also exclusive.
- Latency: one `get` or `put` round trip is about 0.4 to 0.5 s from this laptop (VERIFIED LIVE, includes WAN). From inside Modal containers it will be lower (UNVERIFIED).

## 3. Volumes: concurrent writers and the jsonl log

Docs: https://modal.com/docs/guide/volumes

```python
vol = modal.Volume.from_name("wsp-models", create_if_missing=True, version=2)   # v2 recommended for new volumes
@app.function(volumes={"/vol": vol})
def f():
    ...write files...
    vol.commit()      # make visible to other containers; background commits also run every few seconds and at shutdown
# in a container that must see another container's commits:
vol.reload()          # fails with "volume busy" if any file on the volume is open
```

- A container sees the volume as it was when the container started. Changes committed elsewhere appear only after `vol.reload()`.
- "Last write wins in case of concurrent modification of the same file. Any data the last writer didn't have when committing is lost." No distributed locking. Volumes are "typically not a good fit" for concurrent modification of the same file.
- v1: at most about 5 concurrent writers recommended; 500,000 inode hard cap, keep under 50,000 files. v2: hundreds of containers writing **distinct files** is fine, no file count limit, files under 1 TiB; still last-write-wins per file. The existing `modal_app.py` volume is v1 (default). `version=2` only applies when the volume is created; `wsp-models` does not exist in the `rexheng` workspace yet, so we can choose v2 on first creation.

**Verdict on `experiments.jsonl`:** an append-only jsonl on a Volume is NOT safe with concurrent writers. Two containers that both append and commit will clobber each other, silently losing rows. Appending also forces a read of the old file after `reload()`, which fails if the file is open. Do not do it.

Safe designs, in order of preference:
1. **Dict as the system of record.** Key = `exp:{eval_index:06d}` where `eval_index` comes from the budget token (section 6, unique by construction), value = the JSON row. Writes never collide because keys never collide; use `put(..., skip_if_exists=True)` to make retries idempotent. A small local script (`scripts/sync_log.py`) reads `d.items()`, sorts by key, and rewrites `experiments.jsonl` locally. The orchestrator can also append to its own local file as it goes. Dict entries live 7 days past last access, which is fine for a hackathon; the sync script makes the durable copy.
2. One file per writer on a Volume (`/vol/logs/{writer_id}.jsonl`), each file written by exactly one container, `vol.commit()` after each write, merge at read time. Valid but adds `reload()` and busy-volume friction. Use only for large payloads.
3. Funnel through a Queue to a single writer. Unnecessary complexity given option 1.

Large artifacts (compiled models, 2.5 GB each for a 1B bf16 model, checkpoints) go to the Volume under per-hash directories (`/vol/compiled/<hash>/`), one writer per directory, so the concurrent-writer rule is satisfied.

## 4. GPU types, prices, limits, cost controls

Prices: https://modal.com/pricing (per-second list prices; per-hour figures below are my arithmetic, rounded). GPU strings: https://modal.com/docs/guide/gpu.

| `gpu=` | VRAM | $/s | $/hour |
|---|---|---|---|
| `"T4"` | 16 GB (not on fetched page) | 0.000164 | 0.59 |
| `"L4"` | 24 GB (not on fetched page) | 0.000222 | 0.80 |
| `"A10"` (alias A10G works in older examples) | 24 GB | 0.000306 | 1.10 |
| `"L40S"` | 48 GB | 0.000542 | 1.95 |
| `"A100-40GB"` | 40 GB | 0.000583 | 2.10 |
| `"A100-80GB"` | 80 GB | 0.000694 | 2.50 |
| `"RTX-PRO-6000"` | not stated | 0.000842 | 3.03 |
| `"H100"` | 80 GB (not stated per card) | 0.001097 | 3.95 |
| `"H200"` | 141 GB | 0.001261 | 4.54 |
| `"B200"` | not stated | 0.001736 | 6.25 |

- CPU: $0.0000131 per physical core-second (about $0.047 per core-hour, a core is 2 vCPU). Memory: $0.00000222 per GiB-second (about $0.008 per GiB-hour). `compile_remote` at 4 cores and 16 GiB is about $0.32/hour. Volumes $0.09/GiB/month, first 1 TiB free. Egress $0.04/GiB after plan allowance.
- Billing: during image load and while processing inputs, plus `scaledown_window` idle time (default 60 s) after the last input. Idle containers beyond that cost nothing.
- Multipliers: region selection 1.15x to 1.75x; non-preemptible 3x on CPU and memory.
- `gpu="H100"` may be silently upgraded to an H200 at H100 price; `"H100!"` pins it. `gpu=["H100", "A100-40GB:2"]` is a fallback list. `gpu="H100:8"` is multi-GPU. Up to 8 GPUs per container (A10: 4).
- Concurrency caps by plan (pricing page): Starter 100 containers and **10 GPU concurrency**; Team 5000 and 50. This is the ceiling on K-way GPU fan-out regardless of `max_containers`.
- Cost controls:
  - `max_containers=N` on every Function. Caps parallelism and therefore spend rate. Set it explicitly.
  - `timeout=` on every Function (default 300 s, max 24 h). Always set it; a stuck eval otherwise bills to the limit.
  - `retries=modal.Retries(max_retries=1)` for evals; `retries=0` or None for anything with non-idempotent side effects (budget pop).
  - `scaledown_window` low (for example 30) on GPU functions with bursty use; high only for a vLLM server you are actively hammering.
  - `min_containers=0` (default). Never set `min_containers>0` on a GPU unless you want to pay while idle.
  - Workspace budgets exist (https://modal.com/docs/guide/budgets#workspace-budgets, linked from the pricing FAQ; page not fetched). Environment-level budgets are Team and Enterprise only. Set a workspace spend limit in the dashboard before the first detached baseline.
  - `modal app stop <name> --yes` kills everything, including detached jobs.

Free credit on Starter: $30/month compute. That is about 37 L4-hours, 12 A100-80GB-hours, or 7.6 H100-hours. Model API tokens are billed from the first request and are not covered by the $30 (pricing FAQ).

## 5. Serving an open-weight LLM (vLLM) for planner agents

Docs: https://modal.com/docs/examples/vllm_inference (alias https://modal.com/docs/examples/llm_inference). Read via the page extractor; the code below is the example's, abbreviated.

```python
import json, modal
MODEL_NAME = "google/gemma-4-26B-A4B-it"       # the official example's model
vllm_image = (
    modal.Image.from_registry("nvidia/cuda:12.9.0-devel-ubuntu22.04", add_python="3.12")
    .entrypoint([])
    .uv_pip_install("vllm==0.21.0", "transformers==5.8.1")
    .env({"HF_XET_HIGH_PERFORMANCE": "1"})
)
hf_cache_vol   = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
vllm_cache_vol = modal.Volume.from_name("vllm-cache", create_if_missing=True)
app = modal.App("wsp-llm")

@app.server(                       # App.server exists in client 1.6.4 (VERIFIED); older examples use @modal.web_server
    image=vllm_image, gpu="H200:1",
    scaledown_window=15 * 60, startup_timeout=10 * 60,
    volumes={"/root/.cache/huggingface": hf_cache_vol, "/root/.cache/vllm": vllm_cache_vol},
    port=8000, target_concurrency=100,
    unauthenticated=True,          # public URL; remove and use proxy auth tokens for anything non-demo
)
class Server:
    @modal.enter()
    def start(self):
        import subprocess
        self.process = subprocess.Popen([
            "vllm", "serve", MODEL_NAME, "--served-model-name", MODEL_NAME, "llm",
            "--host", "0.0.0.0", "--port", "8000", "--async-scheduling",
            "--enforce-eager",     # FAST_BOOT=True in the example: skips torch.compile and CUDA graphs
            "--tensor-parallel-size", "1", "--enable-auto-tool-choice",
        ])
    @modal.exit()
    def stop(self):
        self.process.terminate()
```

- Endpoint: standard OpenAI route, `POST <url>/v1/chat/completions` with `{"model": <served name>, "messages": [...]}`; the `openai` Python client works with `base_url="<url>/v1"`. Streaming via SSE.
- Cold start: the page says compilation features add "tens of seconds to a few minutes", reduced to about 10 s when loaded from the vLLM cache volume; `FAST_BOOT = True` (`--enforce-eager`) is recommended for scale-from-zero services and when iterating. Total cold start including weight download on the first run is not stated; expect several minutes the first time (HF download into the cache volume) and tens of seconds to about two minutes afterwards. UNVERIFIED, not run (a GPU run was out of scope).
- Model to GPU fit (my estimates, UNVERIFIED, standard bf16 memory arithmetic of 2 bytes per parameter plus KV cache): 7B to 8B dense bf16 (about 15 to 17 GB) fits L4 or A10 (24 GB) only with a short context; comfortable on L40S (48 GB) at $1.95/h. A 26B MoE like the example's Gemma needs about 52 GB of weights in bf16, so A100-80GB ($2.50/h) minimum, and the example uses H200. For planner agents that propose programs as JSON, a 7B to 8B instruct model with tool or JSON mode on an L40S is the cheapest sensible choice; confirm quality on a few prompts before committing.
- Alternative with no GPU management: Modal "Model APIs" (hosted shared endpoints) are priced per 1M tokens on the pricing page, for example GLM 5.3 Flash at $0.45 prompt / $1.50 completion, DeepSeek V4.1 Flash at $0.30 / $1.20, Kimi K3.1 Pro at $2.00 / $10.00. Create with `modal endpoint create --shared --model <name>` and authenticate with a bearer of `$MODAL_PROXY_TOKEN_ID.$MODAL_PROXY_TOKEN_SECRET` (https://modal.com/docs/guide/endpoints). A blog post shows `https://api.us-west-2.modal.direct/v1` as a base URL with an API key. The exact base URL and auth for the current product are UNVERIFIED (two sources disagree on mechanism); read the endpoints page before relying on it. Tokens are billed from the first request and are not covered by the $30 credit. For a swarm that makes many short planner calls this is likely cheaper than holding a GPU, since a vLLM server bills for `scaledown_window` idle time after each burst.

## 6. Shared budget counter: design and recommendation

Requirements: one counter that the swarm and every baseline draw from; safe under concurrency; possibly different machines; an unambiguous global eval index for the "score vs number of evaluations" chart.

Primitives tested in section 2:

| Option | Atomic? | Gives eval index | Cost |
|---|---|---|---|
| A. Queue pre-filled with N tokens `0..N-1`, spend = pop | VERIFIED LIVE, local threads and 4 containers | Yes, the token is the global eval index (FIFO, so pops come out in order) | 1 RPC per spend, no scans |
| B. Dict tickets via `put("ticket:i", who, skip_if_exists=True)`, try i = 0,1,2,... | VERIFIED LIVE | Yes, the winning `i` | up to N RPCs per spend in the worst case (every worker rescans), measured 8.0 s for 8 workers x 40 tickets |
| C. Single-container "budget service" Function (`max_containers=1`, `@modal.concurrent`, in-memory int) | Atomic only while one container lives; a restart or scaledown resets it | Yes | extra moving part; state must be mirrored to a Dict to survive restarts. Lets teammates in other workspaces call it via a web endpoint |
| D. Dict read-modify-write counter | **No** (VERIFIED LIVE: 6 of 40 increments survived) | n/a | never use |
| E. Volume file | **No** (last write wins, no locking) | n/a | never use |

### Recommendation: A, the Queue of tokens, with a Dict audit ledger

```python
# one-time setup (run once, from one machine)
q = modal.Queue.from_name("wsp-budget", create_if_missing=True)
q.clear(all=True)
q.put_many(list(range(BUDGET)), partition_ttl=7 * 24 * 3600)   # BUDGET <= 5000 per partition

# every evaluation, swarm or baseline
def spend(who):
    q = modal.Queue.from_name("wsp-budget")
    token = q.get(block=False)           # None means budget exhausted; token 0 is falsy so test `is None`
    if token is None:
        return None
    ledger = modal.Dict.from_name("wsp-ledger", create_if_missing=True)
    ledger.put("spent:%06d" % token, {"who": who, "t": time.time()}, skip_if_exists=True)
    return token
```

Why this one:
- Exclusivity is what the Queue is built for, and it held in both live tests (threads, and four separate containers using `get_many`).
- The token doubles as the global eval index, so `experiments.jsonl` rows, the budget curve and the audit ledger line up with no extra counter.
- Over-spend is impossible by construction: when the queue is empty, nothing can pop. An off-by-one race on a counter cannot happen.
- Failure mode is conservative: a crash after the pop wastes one token (budget under-used), never over-uses it. The ledger shows which tokens were taken and by whom; compare it to logged results to count leaks.
- No always-on container, no reliance on `max_containers=1` semantics.

Rules to make it correct in practice:
1. **Pop immediately before the real eval, after compile succeeds and after the cache lookup misses.** A cache hit costs no token (decision for Arjun, flagged in the brief's "cached by (model hash, prompt)" line: either count hits or not, but the same rule must apply to every baseline, so implement it once in `spend`). A compile failure before eval should not consume a token unless Arjun wants it to.
2. **Never wrap `spend` in `retries`.** Modal retries re-run the whole input, so a retried function would pop a second token. Put `retries` only on the pure compile step, or make the eval function take the token as an argument (pop in the caller, pass `token` in) so a retry re-uses the same token.
3. **Refunds**: if an eval fails for an infrastructure reason (preempted, OOM), `q.put(token)` returns it; the order then breaks (the token goes to the back), so the eval index in the log is the token only if you never refund. Recommended: do not refund, write the failure to the log with its token, and count it as spent. State this rule once and apply it to all arms.
4. **Limits**: 5,000 tokens per partition, 24 h default TTL after the last put. Set `partition_ttl` generously on the fill (above) and refill nothing mid-run. If BUDGET exceeds 5,000, use partitions `b0`, `b1`, ... and have `spend` try them in order. A real budget in this project will be in the hundreds, so this is not binding.
5. **Visibility**: `q.len()` is the remaining budget; `ledger.len()` (capped at 100,000) is spent. Surface both on the demo screen.
6. **Cross-machine**: every caller must authenticate to the same workspace (`rexheng`). If a teammate cannot use these tokens, deploy a thin `@app.function` `take_token(who)` that does the pop server-side and call it by `Function.from_name` (still same workspace) or expose it with `@modal.fastapi_endpoint` plus a shared secret header so that outsiders only need a URL. UNVERIFIED end to end; the pop function itself is the same code.

If Arjun later wants a hard guarantee that is independent of Modal's Queue semantics (which the docs do not promise as atomic, only my tests do), option B (Dict `skip_if_exists` tickets) is the second choice; it is also verified live and its atomicity is a documented return-value contract (`put` returns whether it wrote). It is slower at scale; with 8 workers and 40 tickets it took 8 s total, mostly redundant probing, so a tuned version would start probing at a rolling cursor.

## 7. Image building

Docs: https://modal.com/docs/guide/images

```python
image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install("torch==2.7.1", "transformers==4.57.1", "accelerate", "safetensors",
                    "huggingface_hub", "zstandard", "numpy")     # uv is the documented default, faster
    .env({"HF_HOME": "/vol/hf"})
    .add_local_python_source("wsp")                              # added at container start, not baked
    .add_local_file("catalog.jsonl", "/root/catalog.jsonl")
)
```

- `uv_pip_install` is the recommended installer; fall back to `pip_install` only if a resolver issue appears. `pip_install` (used in the existing `modal_app.py`) still works. Both are present in client 1.6.4 (VERIFIED).
- `add_local_python_source` / `add_local_file` / `add_local_dir` are mounted at container start by default, so editing `wsp/` does not rebuild the image; pass `copy=True` only if a later build step needs those files. Put frequently changing layers last, because a changed layer invalidates every layer after it.
- `modal.Image.debian_slim(python_version="3.12")` is supported; default is the local minor version. Local client supports Python 3.10 to 3.15 (VERIFIED from package metadata), so the repo's 3.12 venv is fine.
- For models: either `run_function` to bake weights into the image, or download into a Volume (`HF_HOME=/vol/hf`, as `modal_app.py` already does) and reuse across containers. The Volume path is better here because many different models are touched.
- Reuse the existing `modal_app.py` image for compile and eval so wheels are built once. Build time for a torch image is minutes the first time and cached afterwards.

## 8. Recommended integration for part 3

Single new file `modal_swarm.py` (or extend `modal_app.py`, but keep Arjun's code separate from Trisanth's), app name `wsp-swarm`, deployed once with `modal deploy modal_swarm.py` so any process (laptop, detached driver, teammate) can call it with `Function.from_name`. Everything below uses the `rexheng` workspace via the env vars.

Shared Modal objects (all `create_if_missing=True`, created once in a setup entrypoint):

| Object | Type | Purpose |
|---|---|---|
| `wsp-budget` | Queue | tokens `0..BUDGET-1`, section 6 |
| `wsp-ledger` | Dict | `spent:{token}` audit rows |
| `wsp-cache` | Dict | eval cache, key `sha256(model_hash + "|" + split + "|" + prompt_id)`, value `scores` dict |
| `wsp-exp` | Dict | the experiment log, key `exp:{token:06d}` |
| `wsp-models` | Volume (v2) | `/vol/compiled/<program_hash>/` compiled models, `/vol/hf` HF cache |

### (a) Remote compile and eval of agent-written programs

One Function does the whole candidate pipeline so a 2.5 GB model is not round-tripped through the Volume:

```python
@app.function(image=image, volumes={"/vol": vol}, gpu="L4", memory=32768,
              timeout=1800, max_containers=8, retries=modal.Retries(max_retries=1))
def run_candidate(program_dict, split, who, hypothesis):
    # 1. program_hash = sha256(canonical json of program_dict incl. seed)   (compile is deterministic per brief)
    # 2. cache lookup in wsp-cache keyed by (program_hash, split, prompt_id); hit -> return cached row, no token
    # 3. Program.from_dict(...).compile(local_dir, catalog)        <- Trisanth's compile, writes to container-local disk
    # 4. token = spend(who); if None: return {"status": "budget_exhausted"}
    # 5. scores = eval_stub(local_dir, split)                       <- replace with Person B's eval
    # 6. cache put (skip_if_exists), log row put wsp-exp "exp:%06d" % token
    # 7. return the row
```

Notes: the existing `compile_remote` already runs `program.compile(...)` on CPU and commits to the Volume. Keep it for models that must persist (top-k frontier, heal inputs); for the search loop skip the Volume and compile to `/tmp`. L4 (24 GB, $0.80/h) is enough for a 1B model eval; step up to A10/L40S only if Person B's harness needs it. **Retry caution:** because `retries=1` re-runs the body, pass the token into the function instead of popping inside it when you wire the real eval (rule 2 in section 6), or move the pop to a non-retried wrapper. For the stub phase the body is cheap and idempotent apart from the pop.

Use the **program hash** as the model hash in the cache key. Compile is deterministic for a given program and seed, so hashing the canonical program JSON avoids hashing 2.5 GB of weights, and it allows a cache lookup before spending any compile time. Two different programs that compile to identical weights will miss the cache; that is an acceptable loss. If Person B hashes weights instead, add their hash as a second key.

### (b) The one shared budget counter

Queue of tokens plus Dict ledger, section 6. Wrap it in `spend(who)` and import that one function in the swarm, the evolutionary baseline, LoRA and full fine-tune arms, so no arm can bypass it. Fill the Queue once; log `BUDGET`, the seed and the fill time in the first `wsp-exp` row.

### (c) Eval result cache

`wsp-cache` Dict. Key = sha256 of `program_hash`, `split`, `prompt_id` joined by `|` (always a plain string, so cloudpickle determinism is irrelevant). Value = `{"scores": {...}, "token": n, "model_path": ...}`. Write with `put(key, value, skip_if_exists=True)` so two racing workers with the same candidate converge on one stored result (the loser still spent a token; accept this or serialize identical programs before fan-out). Expires after 7 idle days, fine. Dict `len` is capped at 100,000, enough.

### (d) `experiments.jsonl`

Dict `wsp-exp` as the source of truth, written by the worker that holds the token (key `exp:{token:06d}`, row = timestamp, agent, hypothesis, program, split, scores, eval count = token+1). Local `experiments.jsonl` is produced by `scripts/sync_log.py`: iterate `d.items()`, sort by key, rewrite the file. The orchestrator also appends rows for non-eval events (analyst notes, judge decisions) to its own local file, which only one process writes. Never append to a shared Volume file (section 3).

### (e) Parallel fan-out of K candidates per round

```python
f = modal.Function.from_name("wsp-swarm", "run_candidate")
args = [(p.to_dict(), "search", "swarm", hyp) for p, hyp in candidates]
rows = list(f.starmap(args, order_outputs=False, return_exceptions=True))
```

K is bounded by `max_containers=8` (set it deliberately) and by the plan's GPU concurrency (Starter: 10, shared across everything running at once, including the detached baseline and the vLLM server). Plan for K <= 4 while the baseline runs. Exceptions come back in the list; log them as rows, do not drop them ("collapsed merges are data"). A planner that proposes more than K programs gets the extras queued locally, not on Modal.

### (f) Evolutionary baseline in the background (detached)

Two workable patterns; pick the second.
1. `modal run --detach modal_swarm.py::baseline` where the entrypoint runs the search loop locally. Not recommended: the loop lives in the local process and stops when the laptop sleeps or disconnects. Only the Modal calls already in flight survive.
2. Make the loop a Modal Function and fire it from anywhere:

```python
@app.function(image=image, timeout=24 * 3600, max_containers=1)   # no retries: a restart would re-enter the loop
def baseline_driver(seed, budget_share=None):
    # reads/writes wsp-budget, wsp-cache, wsp-exp by name; calls run_candidate.map(...) for each generation;
    # on each loop start reads the ledger so a manual relaunch resumes instead of restarting.

# from laptop, after `modal deploy modal_swarm.py`:
fc = modal.Function.from_name("wsp-swarm", "baseline_driver").spawn(seed=0)
print(fc.object_id)       # keep this; stop with `modal app stop` or fc.cancel()
```

Why: the 24 h timeout is the hard maximum, the driver runs inside Modal (own network, no laptop dependency), and `max_containers=1` prevents an accidental second baseline spending the shared budget. Because the driver and the swarm race for the same Queue, the equal-budget comparison needs a rule fixed in advance: either give each arm its own token range (two Queues, `wsp-budget-swarm` and `wsp-budget-evo`, each with BUDGET tokens, the fair design for "same budget" curves) or run them in sequence on one Queue. The brief says one shared counter; that matches a single pool only if the arms run one after the other or the pool is partitioned per arm. Flag this to Arjun; the code supports either (`spend(queue_name, who)`).

### (g) Open-weight LLM for planner agents (no Anthropic key)

Deploy the vLLM server from section 5 as its own app `wsp-llm` (separate from the swarm app, so redeploying one does not cycle the GPU). Start with a 7B to 8B instruct model on one L40S ($1.95/h) with `FAST_BOOT` on, `scaledown_window` 300 to 600 s so a round of K planner calls shares one warm container, and `target_concurrency` of 16 to 32. Mount the two cache Volumes so the second boot skips the download. Planners call it with the `openai` client against `<server_url>/v1`. Keep `unauthenticated=True` off for anything beyond a throwaway demo: a public endpoint on your bill is a risk; use Modal proxy auth tokens or a shared secret header. Compare against the hosted Model APIs option in section 5 before building this; if per-token pricing for a small model beats an idle-prone GPU, use that instead and skip all of the above. Either way, calls to this are not eval calls and must not touch the budget Queue.

### (h) GPU jobs for LoRA and full fine-tuning baselines

- `@app.function(gpu="A100-80GB", timeout=6 * 3600, volumes={"/vol": vol}, retries=None)`; write checkpoints under `/vol/ckpt/<run>/` with `vol.commit()` per checkpoint (background commits also run every few seconds), resume from the newest checkpoint on restart. Modal functions can be preempted; `nonpreemptible=True` is available but at 3x CPU and memory rates (GPU price unchanged per the pricing page note).
- Sizing (estimates, UNVERIFIED): LoRA on a 1B bf16 model fits an L4 or A10 ($0.80 to $1.10/h). Full fine-tune with AdamW on a 1B model needs roughly 16 to 24 GB for weights, gradients and optimizer states before activations, so use L40S or A100-40GB ($1.95 to $2.10/h) rather than L4. Use A100-80GB ($2.50/h) if batch size or sequence length is large.
- Equal-budget accounting: these arms spend evals only when scoring checkpoints. Decide with Arjun whether each checkpoint evaluation costs one token and whether training compute is a separate axis (it is, and should be reported next to the eval axis). Training steps never touch the budget Queue; only calls to the scoring function do, through the same `spend`.
- GPU concurrency is shared: a long fine-tune on the Starter plan's 10-GPU cap competes with `run_candidate` fan-out. Use `max_containers` on the training function and run training in a separate app so it can be stopped on its own.

## 9. Open items and things I could not verify

- Plan tier, remaining credit and GPU concurrency of the `rexheng` workspace (dashboard only).
- Whether `modal run --detach` on a long `local_entrypoint` survives a closed laptop (CLI help says the app is kept alive on disconnect; the entrypoint's local code is the doubt). Use the `.spawn` pattern in (f), which I tested in miniature (spawn of a deployed function from plain Python returned its result).
- Exact retention of `.spawn` results and `FunctionCall.from_id` across processes; the job-queue guide fetch returned nothing.
- vLLM cold start numbers, model-to-GPU fit and Model APIs base URL and auth (docs or arithmetic only, no GPU run).
- Dict value size limit, Queue cross-workspace sharing, and Queue behavior under multi-hour sustained use (tested for seconds only; 24 h partition TTL is the documented risk).
- Behavior of `get_many(k, block=False)` when fewer than k tokens remain was only partially exercised (it returned the last items correctly in the container test, 20 tokens for 12 callers asking 2 each, and later callers got `[]`).
