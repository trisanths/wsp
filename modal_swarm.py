"""Modal app "wsp-swarm": part 3's budget objects, caches, and the real compile, heal and eval path.

First pair: host unsloth/Llama-3.2-1B-Instruct, donor Qwen/Qwen2.5-Math-1.5B,
task GSM8K. run_candidate grades with one of two scorers: the interim scorer
(wsp/swarm/interim_eval.py, the same logic as modal_graft.py, gsm8k only,
labeled "interim") or Arjun's eval harness (wsp_eval.harness, gsm8k and
if_guard on the hf backend, labeled "harness"). Search-split numbers are never results.

Auth and workspace come only from MODAL_TOKEN_ID and MODAL_TOKEN_SECRET in
the environment (load .env with `set -a && . ./.env && set +a`). Nothing here
names a workspace, so moving to a shared workspace means changing the tokens.

Deploy (the pilot catalog is baked in at deploy time, so redeploy after
scripts/build_pilot_catalog.py rewrites data/pilot_catalog.jsonl):
    .venv/Scripts/python -m modal deploy modal_swarm.py
    WSP_SWARM_GPU=L4,A10 .venv/Scripts/python -m modal deploy modal_swarm.py   # smoke or dev only

Named objects (created on first use, create_if_missing=True):
    Queue  wsp-budget        eval tokens 0..n-1, one partition per (run_id, pool)
    Dict   wsp-ledger        write-once budget rows, see LEDGER KEYS below
    Dict   wsp-cache         run_candidate results (per run) and host_baseline results (shared)
    Dict   wsp-probe         probe_candidate results (per run), keyed by the program hash without heal
    Dict   wsp-exp           experiment log rows, key "<run_id>:<seq>"
    Volume wsp-swarm-models  HF cache (/vol/hf), donor tensor cache (/vol/tensor_cache),
                             pilot index files (/vol/units/pilot/*.pt)

LEDGER KEYS (every row is written once with skip_if_exists, never updated):
    <run_id>|<pool>|fill|pool             one per pool, claimed by fill_pool
    <run_id>|<pool>|eval|<token:06d>      one per spent token
    <run_id>|<pool>|compute|<charge_id>   one per compute charge
Row values follow wsp/swarm/budget.py: run_id, method, pool, axis, amount,
token, who, phase, time, plus gpu (the device a compute charge ran on).

Local helpers (plain functions, callable from any machine authed to the
same workspace): fill_pool, spend, charge_compute, charge_uncharged,
compute_used, compute_devices, eval_spent, eval_remaining, ledger_rows,
put_exp_row, the cache key functions, lookup_probe, lookup_result,
lookup_host, create_objects and deployed(name) for a handle to a deployed
function.

Remote functions (max_containers=4 on GPU, a timeout on each, no
min_containers, no web endpoints):
    probe_candidate(program_dict, catalog_name, run_id=, pool=, who=, method=)
        fit-only compile, cached in wsp-probe for that run
    run_candidate(program_dict, split, token, catalog_name, eval_limit=None, run_id=, pool=, who=, method=,
                  tasks=None, scorer="interim")
        compile, heal and eval in one container, cached in wsp-cache for that run; scorer
        "harness" grades gsm8k and if_guard with wsp_eval.harness (whole splits only)
    host_baseline(split, eval_limit=None)    the plain host, scored once and cached for every run
    extract_parts(layers=None, sizes=None)   the pilot part set (max_containers=1)
No remote function pops a token: the caller spends it and passes it in, so
retries=1 on run_candidate can never spend twice.

CACHE SCOPE. Probe and run results are cached per run id. Each arm and seed
has its own run id, and no run gets another run's probes or evals for free,
so a curve does not depend on which arm ran first (design section 5). The
host baseline is a shared reference, so its cache is global and it is
charged to no run.

COMPUTE. probe_candidate and run_candidate write their own compute rows
from inside the container, one write-once row per phase as each phase
finishes, with the charge id "<prefix>:<phase>:<attempt>" ("<prefix>" is
the program hash, plus ":<token>" for a run). Every execution of a function
body is a new attempt, so a Modal retry adds its own rows and the first
attempt's GPU-seconds are kept. The caller must not charge result
["seconds"] again; it charges only result["uncharged"] (rows the container
could not write), with charge_uncharged(result). A phase cut off by a
container kill (timeout, preemption) is not charged.

Before compile, each call runs an untimed setup: imports, the GSM8K splits
and test set (loaded once per container), the catalog, CUDA start-up and a
first read of each model's weight files from the volume. Its time is
reported as "setup_seconds" and never charged. It depends only on whether
the container is fresh, so charging it would make an arm's GPU-seconds
depend on how many of its calls landed on cold containers.

Budget code fails closed: spend returns None on any error, eval_remaining
returns 0 when the queue cannot be read, and charge_compute, compute_used,
compute_devices and ledger_rows raise BudgetError rather than under-count.
Cache and log helpers fail soft (print a warning, act as a miss).

GPU comes from WSP_SWARM_GPU on the deploying machine, default "L4" (24 GB
holds the 1B host, the 1.5B donor during the fit, and the heal pass). The
image carries the same value, so the container sees it too. Graded runs
pin one type, because nothing normalizes GPU-seconds across types. A comma
list such as "L4,A10" is a preference list for smoke or dev deploys when L4
capacity is short. Every compute row records the device the container
actually got, and compute_devices(run_id, pool) lists them, so a run that
mixed types can be found and flagged.
"""

import hashlib
import importlib
import json
import os
import sys
import time
import uuid

import modal

APP_NAME = "wsp-swarm"
BUDGET_QUEUE = "wsp-budget"
LEDGER_DICT = "wsp-ledger"
CACHE_DICT = "wsp-cache"
PROBE_DICT = "wsp-probe"
EXP_DICT = "wsp-exp"
VOLUME_NAME = "wsp-swarm-models"
DICT_NAMES = [LEDGER_DICT, CACHE_DICT, PROBE_DICT, EXP_DICT]

VOL = "/vol"
TENSOR_CACHE = VOL + "/tensor_cache"
PILOT_UNITS_DIR = "units/pilot"

HOST = "unsloth/Llama-3.2-1B-Instruct"
DONOR = "Qwen/Qwen2.5-Math-1.5B"
DONOR_SHORT = "qmath15"
TASK = "gsm8k"
EVAL_BACKEND = "interim"
SPLITS = ["search", "heldout"]
# run_candidate's scorers: "interim" (wsp/swarm/interim_eval.py, gsm8k only, any eval_limit) and
# "harness" (Arjun's wsp_eval.harness.evaluate with the hf backend, gsm8k and if_guard, whole splits only).
SCORERS = ["interim", "harness"]
SCORER_TASKS = {"interim": ["gsm8k"], "harness": ["gsm8k", "if_guard", "humaneval", "mmlu", "mmlu:<subject>"]}
# Task-name prefixes the harness also grades (wsp_eval.tasks.get_task): one MMLU or MMLU-Pro subject.
HARNESS_PREFIXES = ["mmlu:", "mmlupro:"]


def scorer_allows(scorer, task):
    """True when this scorer can grade this task name."""
    if task in SCORER_TASKS.get(scorer, []):
        return True
    if scorer == "harness":
        for prefix in HARNESS_PREFIXES:
            if task.startswith(prefix) and len(task) > len(prefix):
                return True
    return False
SCORER_LABELS = {"interim": "INTERIM", "harness": "HARNESS"}
HARNESS_ROOT = "/tmp/wsp_eval_root"   # WSP_ROOT in the container: the harness's work files stay container-local

GPU_ENV = "WSP_SWARM_GPU"
DEFAULT_GPU = "L4"
GPU_MAX_CONTAINERS = 4
ALIGN_SAMPLES = 64          # alignment texts for the W_in ridge fit; matches the core config default
EXTRACT_TEXTS = 300         # texts for the pilot extractor, as in modal_graft.py
PILOT_LAYERS = [6, 12, 18, 24]
PILOT_SIZES = [256, 512]
PILOT_SOURCE = "wsp/extract.py stub extractor; pilot set, not the Person A catalog"

POOL_MAX = 5000             # Modal limit: 5,000 items per Queue partition
POOL_TTL_S = 24 * 3600      # Modal clears a partition this long after the last put; 24 h is the documented maximum
PARTITION_KEY_MAX_BYTES = 64
READ_CHUNK_BYTES = 64 * 1024 * 1024

# Modules imported during the untimed setup, so their import time is not charged as compile.
HEAVY_MODULES = ["torch", "transformers", "wsp.align", "wsp.assemble", "wsp.compiler", "wsp.fetch",
                 "wsp.graft", "wsp.heal", "wsp.load", "wsp.swarm.interim_eval"]

REPO = os.path.dirname(os.path.abspath(__file__))
# catalog name -> (path in the repo, path in the container)
CATALOG_FILES = {
    "main": ("catalog.jsonl", "/root/catalog.jsonl"),
    "pilot": ("data/pilot_catalog.jsonl", "/root/data/pilot_catalog.jsonl"),
}


def warn(text):
    """Print a warning to stderr. Used by every fail-soft and fail-closed path."""
    print("modal_swarm: " + text, file=sys.stderr)


def describe_error(error):
    return type(error).__name__ + ": " + str(error)[:2000]


def parse_gpu(text):
    """The Modal gpu argument for a WSP_SWARM_GPU value.

    One name ("L4", also used for an empty value) pins that type. A comma
    list such as "L4,A10" becomes a preference list; use it only for smoke
    or dev deploys, since seconds from different types are not comparable.
    """
    names = []
    for piece in str(text).split(","):
        piece = piece.strip()
        if piece != "":
            names.append(piece)
    if len(names) == 0:
        return DEFAULT_GPU
    if len(names) == 1:
        return names[0]
    return names


def gpu_text(gpu):
    """The WSP_SWARM_GPU text for a parsed gpu value (the inverse of parse_gpu)."""
    if isinstance(gpu, list):
        return ",".join(gpu)
    return gpu


GPU = parse_gpu(os.environ.get(GPU_ENV, DEFAULT_GPU))
GPU_PINNED = not isinstance(GPU, list)


# ---------------------------------------------------------------- image and app


def build_image():
    """The same image as modal_graft.py (same pins), plus wsp and the catalog files that exist."""
    image = (
        modal.Image.debian_slim(python_version="3.12")
        .pip_install(
            "torch==2.7.1", "transformers==4.57.1", "accelerate", "huggingface_hub", "requests",
            "safetensors", "zstandard", "numpy", "datasets",
        )
        .env({"HF_HOME": VOL + "/hf", GPU_ENV: gpu_text(GPU), "WSP_ROOT": HARNESS_ROOT,
              "WSP_HF_BATCH_SIZE": "64"})
        .add_local_python_source("wsp", "wsp_eval")
    )
    for name in sorted(CATALOG_FILES):
        repo_path, container_path = CATALOG_FILES[name]
        local_path = os.path.join(REPO, *repo_path.split("/"))
        if os.path.exists(local_path):
            image = image.add_local_file(local_path, container_path)
    return image


app = modal.App(APP_NAME)
image = build_image()
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True, version=2)


def get_queue(name):
    """A handle to a named Queue. Tests replace this function with a fake."""
    return modal.Queue.from_name(name, create_if_missing=True)


def get_dict(name):
    """A handle to a named Dict. Tests replace this function with a fake."""
    return modal.Dict.from_name(name, create_if_missing=True)


def deployed(function_name):
    """A handle to one function of the deployed wsp-swarm app."""
    return modal.Function.from_name(APP_NAME, function_name)


def create_objects():
    """Create the Queue, the four Dicts and the Volume if they are missing. Returns their names."""
    get_queue(BUDGET_QUEUE).len()
    for name in DICT_NAMES:
        get_dict(name).contains("__wsp_setup_check__")
    volume.hydrate()
    names = [BUDGET_QUEUE]
    for name in DICT_NAMES:
        names.append(name)
    names.append(VOLUME_NAME)
    return names


# ---------------------------------------------------------------- budget helpers


class BudgetError(RuntimeError):
    """The budget store refused a request or could not be reached."""


def check_name(value, what):
    """run_id, pool, scope and charge ids are non-empty strings without '|', the key separator."""
    if not isinstance(value, str) or value == "":
        raise BudgetError(what + " must be a non-empty string")
    if "|" in value:
        raise BudgetError(what + " must not contain '|'")


def partition_key(run_id, pool):
    """The Queue partition for one run and pool.

    Modal allows partition keys of 1 to 64 bytes. "<run_id>:<pool>" is used
    when it fits; a longer pair becomes "h:" plus 40 hex chars of its sha256.
    """
    check_name(run_id, "run_id")
    check_name(pool, "pool")
    key = run_id + ":" + pool
    if len(key.encode("utf-8")) <= PARTITION_KEY_MAX_BYTES:
        return key
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
    return "h:" + digest[:40]


def ledger_key(run_id, pool, axis, suffix):
    return run_id + "|" + pool + "|" + axis + "|" + suffix


def ledger_row(run_id, pool, axis, amount, token, who, phase, method, gpu=None):
    """One ledger row, in the shape wsp/swarm/budget.py writes locally, plus the GPU for compute rows."""
    return {"run_id": run_id, "method": method, "pool": pool, "axis": axis, "amount": amount,
            "token": token, "who": who, "phase": phase, "time": time.time(), "gpu": gpu}


def fill_pool(run_id, pool, n, method=""):
    """Put tokens 0..n-1 into the partition for (run_id, pool). Returns the partition key.

    A pool can be filled once. The fill claims a write-once ledger row first,
    so filling the same run and pool again raises BudgetError (use a new run
    id). The partition lives POOL_TTL_S (24 h) after this put, so a run must
    finish within a day of its fill; after that the pool is empty, which
    fails closed.
    """
    if not isinstance(n, int) or n < 1 or n > POOL_MAX:
        raise BudgetError("a pool holds 1 to " + str(POOL_MAX) + " tokens, got " + repr(n))
    key = partition_key(run_id, pool)
    queue = get_queue(BUDGET_QUEUE)
    waiting = queue.len(partition=key)
    if waiting > 0:
        raise BudgetError("partition " + key + " already holds " + str(waiting) + " tokens")
    row = ledger_row(run_id, pool, "fill", n, None, "fill_pool", "fill", method)
    claimed = get_dict(LEDGER_DICT).put(ledger_key(run_id, pool, "fill", "pool"), row, skip_if_exists=True)
    if not claimed:
        raise BudgetError("pool " + pool + " of run " + run_id + " was filled before; use a new run id")
    tokens = []
    for token in range(n):
        tokens.append(token)
    queue.put_many(tokens, partition=key, partition_ttl=POOL_TTL_S)
    return key


def spend(run_id, pool, who, method=""):
    """Pop one eval token and record it. Returns the token (an int) or None.

    One pop, never retried, because a retried pop could take two tokens.
    Token 0 is valid, so test the result with `is None`. Fails closed: any
    error returns None and runs no eval. If the token was popped but its
    ledger row could not be written, the token is dropped (budget under-used,
    never over-used) and a warning names it.
    """
    try:
        key = partition_key(run_id, pool)
    except BudgetError as error:
        warn("spend refused: " + str(error))
        return None
    try:
        token = get_queue(BUDGET_QUEUE).get(block=False, partition=key)
    except Exception as error:
        warn("spend: the budget queue could not be read, no token (" + describe_error(error) + ")")
        return None
    if token is None:
        return None
    row = ledger_row(run_id, pool, "eval", 1.0, token, who, "eval", method)
    try:
        written = get_dict(LEDGER_DICT).put(ledger_key(run_id, pool, "eval", "%06d" % token), row,
                                            skip_if_exists=True)
    except Exception as error:
        warn("spend: popped token " + str(token) + " but its ledger row failed, so the token is dropped (" +
             describe_error(error) + ")")
        return None
    if not written:
        warn("spend: token " + str(token) + " of " + key + " already has a ledger row; refusing it")
        return None
    return token


def charge_compute(run_id, pool, seconds, who, phase, method="", charge_id=None, gpu=None):
    """Write one write-once compute row. Returns its ledger key, or None for a zero charge.

    phase is "compile", "probe", "heal" or "eval". gpu is the CUDA device
    the work ran on (stored on the row; see compute_devices). charge_id
    makes a charge idempotent: pass a stable id when a charge might be sent
    twice. Without one, a fresh random id is used, so two calls make two
    rows. There is no counter to update, so concurrent charges never lose
    each other. Raises BudgetError when the row cannot be written, because a
    lost charge would under-count compute.
    """
    check_name(run_id, "run_id")
    check_name(pool, "pool")
    seconds = float(seconds)
    if seconds <= 0:
        return None
    if charge_id is None:
        charge_id = uuid.uuid4().hex
    check_name(charge_id, "charge_id")
    key = ledger_key(run_id, pool, "compute", charge_id)
    row = ledger_row(run_id, pool, "compute", seconds, None, who, phase, method, gpu)
    try:
        get_dict(LEDGER_DICT).put(key, row, skip_if_exists=True)
    except Exception as error:
        raise BudgetError("compute charge " + key + " was not written: " + describe_error(error))
    return key


def phase_charge_id(result, phase):
    """The charge id of one phase of one attempt: "<charge_prefix>:<phase>:<attempt>"."""
    return str(result["charge_prefix"]) + ":" + phase + ":" + str(result["attempt"])


def charge_uncharged(result):
    """Charge, from this machine, the phases a container could not write. Returns the ledger keys.

    A probe_candidate or run_candidate result lists those phases in
    "uncharged". The container's own charge ids are reused, so a row that did
    land after all is not written twice. Raises BudgetError if a row still
    cannot be written, because a lost charge would under-count compute.
    """
    keys = []
    uncharged = result.get("uncharged", {})
    for phase in sorted(uncharged):
        key = charge_compute(result["run_id"], result["pool"], uncharged[phase], result["who"], phase,
                             result["method"], phase_charge_id(result, phase), result.get("gpu"))
        if key is not None:
            keys.append(key)
    return keys


def read_ledger():
    """Every (key, row) pair in the ledger. Raises BudgetError when it cannot be read."""
    try:
        pairs = []
        for key, row in get_dict(LEDGER_DICT).items():
            pairs.append((key, row))
        return pairs
    except Exception as error:
        raise BudgetError("the ledger could not be read: " + describe_error(error))


def row_is(row, run_id, pool, axis):
    if not isinstance(row, dict):
        return False
    return row.get("run_id") == run_id and row.get("pool") == pool and row.get("axis") == axis


def compute_used(run_id, pool):
    """The sum of every compute row for (run_id, pool), in seconds. Raises BudgetError if unreadable."""
    total = 0.0
    for key, row in read_ledger():
        if row_is(row, run_id, pool, "compute"):
            total += float(row["amount"])
    return total


def compute_devices(run_id, pool):
    """The GPU names on the compute rows of (run_id, pool), sorted; "unknown" for a row without one.

    Nothing normalizes seconds across GPU types, so more than one name means
    the run's compute axis mixed hardware and should be flagged. Raises
    BudgetError if the ledger is unreadable.
    """
    names = []
    for key, row in read_ledger():
        if row_is(row, run_id, pool, "compute"):
            name = row.get("gpu")
            if name is None:
                name = "unknown"
            if name not in names:
                names.append(name)
    names.sort()
    return names


def eval_spent(run_id, pool):
    """The number of spent-token rows for (run_id, pool). Raises BudgetError if unreadable."""
    count = 0
    for key, row in read_ledger():
        if row_is(row, run_id, pool, "eval"):
            count += 1
    return count


def eval_remaining(run_id, pool):
    """Tokens still in the pool. Returns 0 when the queue cannot be read (fails closed)."""
    try:
        return get_queue(BUDGET_QUEUE).len(partition=partition_key(run_id, pool))
    except Exception as error:
        warn("eval_remaining: the budget queue could not be read, reporting 0 (" + describe_error(error) + ")")
        return 0


def row_order(item):
    """Sort ledger rows by time, then key."""
    return (item["time"], item["key"])


def ledger_rows(run_id):
    """Every ledger row of a run (all pools), oldest first, each with its "key". Raises BudgetError."""
    rows = []
    for key, row in read_ledger():
        if isinstance(row, dict) and row.get("run_id") == run_id:
            copy = dict(row)
            copy["key"] = key
            rows.append(copy)
    rows.sort(key=row_order)
    return rows


def put_exp_row(run_id, seq, row):
    """Write one experiment log row to wsp-exp under "<run_id>:<seq:06d>", once. Fails soft."""
    key = run_id + ":" + "%06d" % int(seq)
    try:
        return get_dict(EXP_DICT).put(key, row, skip_if_exists=True)
    except Exception as error:
        warn("put_exp_row: " + key + " not written (" + describe_error(error) + ")")
        return False


# ---------------------------------------------------------------- cache keys and lookups


def canonical_program(program_dict):
    """The program dict as wsp.program.Program writes it (the locked schema)."""
    from wsp.program import Program

    return Program.from_dict(program_dict).to_dict()


def program_hash(program_dict):
    """sha256 of the canonical JSON, first 16 hex chars. Same rule as wsp/swarm/space.py program_hash."""
    text = json.dumps(canonical_program(program_dict), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def without_heal(program_dict):
    """The same program with heal set to None. Compiling it is a fit probe."""
    data = canonical_program(program_dict)
    data["heal"] = None
    return data


def limit_text(eval_limit):
    if eval_limit is None:
        return "all"
    return "n" + str(int(eval_limit))


def probe_key(program_dict, catalog_name, scope, align_samples=ALIGN_SAMPLES):
    """wsp-probe key: the run scope, the hash of the program without heal, the catalog and the fit's text count.

    scope is the run id. Each arm and seed has its own, so no run reads
    another run's probe. Raises BudgetError for an empty scope or one with '|'.
    """
    check_name(scope, "scope")
    parts = [scope, program_hash(without_heal(program_dict)), catalog_name, "align" + str(align_samples)]
    return "|".join(parts)


def tasks_text(tasks):
    """The task part of a result key: "gsm8k" for the default, else the task names joined by "+"."""
    if tasks is None:
        return TASK
    return "+".join(tasks)


def result_key(program_dict, split, catalog_name, scope, eval_limit=None, align_samples=ALIGN_SAMPLES,
               eval_backend=EVAL_BACKEND, tasks=None):
    """wsp-cache key for one run_candidate result, scoped to one run id like probe_key.

    Includes the scorer (eval_backend) and the tasks, so interim scores never
    answer for the harness and the other way round. The defaults give the
    same keys as before the harness scorer existed.
    """
    check_name(scope, "scope")
    parts = [scope, program_hash(program_dict), split, tasks_text(tasks), limit_text(eval_limit), catalog_name,
             "align" + str(align_samples), eval_backend]
    return "|".join(parts)


def host_key(host, split, eval_limit=None):
    """wsp-cache key for one host_baseline result. Not scoped: the host is a shared reference."""
    return "|".join(["host", host, split, TASK, limit_text(eval_limit), EVAL_BACKEND])


def cache_lookup(dict_name, key):
    """The cached value or None. Fails soft: an unreachable Dict is a miss."""
    try:
        return get_dict(dict_name).get(key)
    except Exception as error:
        warn("cache lookup in " + dict_name + " failed, treating as a miss (" + describe_error(error) + ")")
        return None


def cache_store(dict_name, key, value):
    """Store a value once (skip_if_exists). Returns True if written. Fails soft."""
    try:
        return get_dict(dict_name).put(key, value, skip_if_exists=True)
    except Exception as error:
        warn("cache store in " + dict_name + " failed (" + describe_error(error) + ")")
        return False


def lookup_probe(program_dict, catalog_name, scope, align_samples=ALIGN_SAMPLES):
    """This run's cached probe result, checked from this machine before starting a GPU container."""
    return cache_lookup(PROBE_DICT, probe_key(program_dict, catalog_name, scope, align_samples))


def lookup_result(program_dict, split, catalog_name, scope, eval_limit=None, align_samples=ALIGN_SAMPLES,
                  eval_backend=EVAL_BACKEND, tasks=None):
    """This run's cached run_candidate result. A hit means this run already paid for it."""
    return cache_lookup(CACHE_DICT, result_key(program_dict, split, catalog_name, scope, eval_limit,
                                               align_samples, eval_backend, tasks))


def lookup_host(split, eval_limit=None, host=HOST):
    return cache_lookup(CACHE_DICT, host_key(host, split, eval_limit))


def as_cache_hit(cached):
    """A copy of a cached result marked as a hit: zero seconds and nothing to charge.

    Probe and run caches are per run, so a hit is work this run already paid for.
    """
    hit = dict(cached)
    hit["cache_hit"] = True
    hit["cached_seconds"] = cached.get("seconds", {})
    zero = {}
    for phase in cached.get("seconds", {}):
        zero[phase] = 0.0
    hit["seconds"] = zero
    hit["setup_seconds"] = 0.0
    hit["charged"] = []
    hit["uncharged"] = {}
    return hit


# ---------------------------------------------------------------- pilot catalog records


def pilot_id(layer, n):
    return "pilot_" + DONOR_SHORT + "_l" + str(layer) + "_top" + str(n)


def pilot_record(layer, n, n_texts=EXTRACT_TEXTS):
    """One pilot catalog record, in the brief's record shape plus "pilot" and "source".

    "units" is relative. Inside the container the pilot catalog is loaded
    with root /vol, so it resolves to /vol/units/pilot/<id>.pt on the volume.
    """
    part_id = pilot_id(layer, n)
    return {
        "id": part_id,
        "type": "mlp_cluster",
        "model": DONOR,
        "layer": layer,
        "units": PILOT_UNITS_DIR + "/" + part_id + ".pt",
        "trained_on": ("top " + str(n) + " MLP neurons by mean |activation| (down_proj input) on " +
                       str(n_texts) + " GSM8K heal-split texts (GSM8K train minus the 200-item search split)"),
        "signature": {"effect": {}, "fit_r2": {}},
        "pilot": True,
        "source": PILOT_SOURCE,
    }


# ---------------------------------------------------------------- container-side helpers

# Kept for the life of one container, so a warm container does not reload data or reread models.
CONTAINER_STATE = {"gsm8k": None, "gsm8k_test": None, "warmed": [], "cuda_ready": False}


def read_jsonl(path):
    records = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line != "":
                records.append(json.loads(line))
    return records


def load_catalog(catalog_name):
    """The catalog inside the container. "pilot" resolves units on the volume, "main" next to the file."""
    from wsp.catalog import Catalog

    if catalog_name not in CATALOG_FILES:
        raise ValueError("unknown catalog " + repr(catalog_name) + "; choose " + ", ".join(sorted(CATALOG_FILES)))
    path = CATALOG_FILES[catalog_name][1]
    if not os.path.exists(path):
        raise FileNotFoundError(path + " is not in the image. Build it with scripts/build_pilot_catalog.py, "
                                "then redeploy modal_swarm.py")
    if catalog_name == "pilot":
        return Catalog(read_jsonl(path), root=VOL)
    return Catalog.load(path)


def gsm8k_data():
    """(search items, heal texts) from Trisanth's fixed splits (wsp/data.py, seed 0), loaded once per container."""
    if CONTAINER_STATE["gsm8k"] is None:
        from wsp.data import format_gsm8k, gsm8k_splits

        search_items, heal_items = gsm8k_splits(seed=0)
        heal_texts = []
        for item in heal_items:
            heal_texts.append(format_gsm8k(item))
        CONTAINER_STATE["gsm8k"] = (search_items, heal_texts)
    return CONTAINER_STATE["gsm8k"]


def gsm8k_test():
    """GSM8K test, the held-out split, loaded once per container. wsp/data.py has no test loader."""
    if CONTAINER_STATE["gsm8k_test"] is None:
        from datasets import load_dataset

        items = []
        for item in load_dataset("openai/gsm8k", "main", split="test"):
            items.append(item)
        CONTAINER_STATE["gsm8k_test"] = items
    return CONTAINER_STATE["gsm8k_test"]


def split_items(split, eval_limit):
    """The items to score: the 200-item search split, or GSM8K test for held-out. First eval_limit only."""
    if split == "search":
        items = list(gsm8k_data()[0])
    elif split == "heldout":
        items = list(gsm8k_test())
    else:
        raise ValueError("unknown split " + repr(split) + "; choose " + ", ".join(SPLITS))
    if eval_limit is not None:
        items = items[:int(eval_limit)]
    return items


def import_heavy_modules():
    """Import torch, transformers and the compiler modules, so their import time lands in setup."""
    for name in HEAVY_MODULES:
        importlib.import_module(name)


def ready_cuda():
    """Create the CUDA context and the cuBLAS handle once per container, outside every timed phase."""
    if CONTAINER_STATE["cuda_ready"]:
        return
    import torch

    if torch.cuda.is_available():
        square = torch.ones(64, 64, device="cuda", dtype=torch.bfloat16)
        (square @ square).sum().item()
    CONTAINER_STATE["cuda_ready"] = True


def read_through(path):
    """Read a file once and drop the bytes, so the next read comes from the page cache."""
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(READ_CHUNK_BYTES)
            if not chunk:
                return


def import_model_code(folder):
    """Import the transformers modeling module named by a cached model's config (for example modeling_llama)."""
    try:
        with open(os.path.join(folder, "config.json"), encoding="utf-8") as handle:
            model_type = json.load(handle)["model_type"]
        importlib.import_module("transformers.models." + model_type + ".modeling_" + model_type)
    except Exception as error:
        warn("import_model_code: skipped for " + folder + " (" + describe_error(error) + ")")


def warm_models(model_ids):
    """Read each model's weight files from the volume's HF cache once per container. Returns the ids warmed now.

    The first read of a model on a fresh container is slow. Doing it in the
    untimed setup keeps compile and eval seconds the same on cold and warm
    containers. A model not in the cache yet is skipped with a warning; its
    first download then lands in the timed phase.
    """
    from huggingface_hub import snapshot_download

    warmed = []
    for model_id in model_ids:
        if model_id in CONTAINER_STATE["warmed"]:
            continue
        try:
            folder = snapshot_download(model_id, local_files_only=True)
        except Exception as error:
            warn("warm_models: " + model_id + " is not in the volume cache yet (" + describe_error(error) + ")")
            continue
        for name in sorted(os.listdir(folder)):
            if name.endswith(".safetensors"):
                read_through(os.path.join(folder, name))
        import_model_code(folder)
        CONTAINER_STATE["warmed"].append(model_id)
        warmed.append(model_id)
    return warmed


def program_models(program, catalog):
    """The host and every donor a program's compile loads, without repeats."""
    models = [program.host]
    for spec in program.grafts:
        donor = catalog.get(spec.component_id).get("model")
        if donor is not None and donor not in models:
            models.append(donor)
    return models


def setup_run(program, catalog_name, split, eval_limit, align_samples):
    """The untimed setup before a compile. Returns {"catalog", "align", "heal_input", "items", "warmed"}.

    Imports, the GSM8K data, the catalog, CUDA start-up and the first read of
    each model. split None (a probe) loads no eval items.
    """
    import_heavy_modules()
    catalog = load_catalog(catalog_name)
    search_items, heal_texts = gsm8k_data()
    items = []
    if split is not None:
        items = split_items(split, eval_limit)
    align = None
    if len(program.grafts) > 0:
        align = heal_texts[:align_samples]
    heal_input = None
    if program.heal_spec is not None and program.heal_spec.data == "gsm8k_train":
        heal_input = heal_texts
    ready_cuda()
    warmed = warm_models(program_models(program, catalog))
    return {"catalog": catalog, "align": align, "heal_input": heal_input, "items": items, "warmed": warmed}


def setup_host(host, split, eval_limit):
    """The untimed setup before scoring the plain host. Returns {"items", "warmed"}."""
    import_heavy_modules()
    items = split_items(split, eval_limit)
    ready_cuda()
    warmed = warm_models([host])
    return {"items": items, "warmed": warmed}


def compile_into(program, work_dir, setup):
    """Run Trisanth's compile (wsp/program.py, wsp/compiler.py) into work_dir. Returns its compile_report."""
    program.compile(work_dir, setup["catalog"], align_texts=setup["align"], heal_texts=setup["heal_input"],
                    device="cuda", cache_dir=TENSOR_CACHE)
    return read_report(work_dir)


def evaluate_model(work_dir, items, report):
    """Load the compiled model from container-local work_dir and score it with the interim scorer."""
    from wsp import load
    from wsp.swarm import interim_eval

    model, tokenizer = load(work_dir, device="cuda")
    scored = interim_eval.score(model, tokenizer, items, use_cache=not needs_no_cache(report))
    del model
    return scored


def setup_harness(split, tasks):
    """The untimed setup for the harness scorer: import it and load each task's split items once per container.

    wsp_eval.splits caches the items for the life of the process, so the
    dataset load stays out of the timed eval phase on cold and warm
    containers alike.
    """
    from wsp_eval import splits
    from wsp_eval import tasks as tasks_module

    importlib.import_module("wsp_eval.harness")
    for name in tasks:
        splits.split_items(tasks_module.get_task(name), split)


def evaluate_harness(work_dir, split, tasks, namespace):
    """Score the compiled model in work_dir with Arjun's harness (hf backend). Returns scores and item counts.

    allow_heldout_rerun is True because the harness's held-out ledger lives in
    this container's WSP_ROOT, so a Modal retry of the same held-out call would
    trip it. The env's rule 7 (each program scored once on held-out per run)
    is the guard that counts.
    """
    from wsp_eval import harness

    graded = harness.evaluate(work_dir, split, list(tasks), namespace=namespace, allow_heldout_rerun=True,
                              note="wsp-swarm run_candidate", backend="hf")
    scores = {}
    per_task = {}
    total = 0
    for name in tasks:
        entry = graded["tasks"][name]
        scores[name] = float(entry["score"])
        per_task[name] = int(entry["n"])
        total += int(entry["n"])
    return {"scores": scores, "n_items": total, "n_items_per_task": per_task}


def score_host(host, items):
    """Load the plain host from the volume's HF cache and score it with the interim scorer."""
    import torch
    import transformers

    from wsp.swarm import interim_eval

    model = transformers.AutoModelForCausalLM.from_pretrained(host, dtype=torch.bfloat16).to("cuda").eval()
    tokenizer = transformers.AutoTokenizer.from_pretrained(host)
    scored = interim_eval.score(model, tokenizer, items)
    del model
    return scored


def read_report(folder):
    with open(os.path.join(folder, "compile_report.json"), encoding="utf-8") as handle:
        return json.load(handle)


def heal_seconds(report):
    """The heal loop's own seconds from compile_report.json, 0 without a heal."""
    if report.get("heal") is None:
        return 0.0
    return float(report["heal"].get("seconds", 0.0))


def heal_losses(report):
    if report.get("heal") is None:
        return []
    return report["heal"].get("losses", [])


GRAFT_KEYS = ["id", "kind", "control", "donor", "donor_layer", "host_layer", "d_host", "d_donor",
              "part_params", "glue_params", "fit", "fit_r2", "fit_pairs"]


def graft_entries(report):
    """The per-graft compile report entries without the long index lists (n_units says how many)."""
    entries = []
    for graft in report.get("grafts", []):
        entry = {}
        for key in GRAFT_KEYS:
            entry[key] = graft.get(key)
        entry["n_units"] = len(graft.get("indices", []))
        entries.append(entry)
    return entries


def compact_report(report):
    """compile_report.json with each graft's index list replaced by its count."""
    compact = {}
    for key in report:
        compact[key] = report[key]
    compact["grafts"] = graft_entries(report)
    return compact


def needs_no_cache(report):
    """attn_heads grafts must generate with use_cache=False (brief, facts section)."""
    for graft in report.get("grafts", []):
        if graft.get("kind") == "attn_heads":
            return True
    return False


def free_gpu():
    import gc

    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def gpu_name():
    """The CUDA device this container got, or "none" without CUDA."""
    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.get_device_name(0)
    except ImportError:
        pass
    return "none"


def valid_token(token):
    return isinstance(token, int) and not isinstance(token, bool) and token >= 0


def scorer_problem(scorer, tasks, eval_limit):
    """Why run_candidate cannot grade these tasks with this scorer, or None."""
    if scorer not in SCORERS:
        return "unknown scorer " + repr(scorer) + "; choose " + ", ".join(SCORERS)
    if len(tasks) == 0:
        return "run_candidate needs at least one task"
    for name in tasks:
        if not scorer_allows(scorer, name):
            return ("scorer " + scorer + " grades " + ", ".join(SCORER_TASKS[scorer]) + "; got task " + repr(name))
    if scorer == "harness" and eval_limit is not None:
        return "scorer harness grades whole splits only; eval_limit must be None, got " + repr(eval_limit)
    return None


def call_problem(run_id, pool):
    """Why a call is refused for its run id or pool, or None. Both scope the cache and the compute rows."""
    try:
        check_name(run_id, "run_id")
        check_name(pool, "pool")
    except BudgetError as error:
        return str(error) + " (probe_candidate and run_candidate need the run id and pool they charge)"
    return None


def base_result(run_id, pool, who, method, phases):
    """The fields every probe and run result shares. "attempt" is new for each execution of a body."""
    seconds = {}
    for phase in phases:
        seconds[phase] = 0.0
    return {"ok": False, "error": None, "phase": None, "seconds": seconds, "setup_seconds": 0.0,
            "warmed": [], "run_id": run_id, "pool": pool, "who": who, "method": method,
            "attempt": uuid.uuid4().hex[:16], "charge_prefix": None, "charged": [], "uncharged": {},
            "cache_hit": False, "gpu": gpu_name()}


def charge_in_container(result, phase, seconds):
    """Write the compute row for one phase this attempt just finished. Never raises.

    The id is phase_charge_id(result, phase), so a retry (a new attempt) adds
    its own rows and charging one phase twice in one attempt writes one row.
    A row that cannot be written goes into result["uncharged"] for the
    caller to settle with charge_uncharged(result).
    """
    try:
        key = charge_compute(result["run_id"], result["pool"], seconds, result["who"], phase, result["method"],
                             phase_charge_id(result, phase), result["gpu"])
    except BudgetError as error:
        warn("the " + phase + " charge was not written; the caller must settle it (" + str(error) + ")")
        result["uncharged"][phase] = seconds
        return
    if key is not None:
        result["charged"].append(key)


# ---------------------------------------------------------------- remote functions


@app.function(image=image, gpu=GPU, volumes={VOL: volume}, cpu=2.0, memory=16384, timeout=1200,
              max_containers=GPU_MAX_CONTAINERS, scaledown_window=60)
def probe_candidate(program_dict, catalog_name, align_samples=ALIGN_SAMPLES, run_id=None, pool=None, who="",
                    method=""):
    """Fit-only compile: Trisanth's compile with heal set to None, no token, no eval.

    run_id and pool are required: the probe is cached for that run only and
    its compile seconds are charged to that run and pool from inside the
    container (ledger phase "probe"). Returns {"ok", "error", "phase",
    "graft_report": [per-graft entries with fit, fit_r2, fit_pairs,
    part_params, glue_params, host_layer, ...], "seconds": {"compile": s},
    "setup_seconds", "warmed", "probe_key", "program_hash", "run_id", "pool",
    "who", "method", "attempt", "charge_prefix", "charged", "uncharged",
    "cache_hit", "gpu"}. A cache hit returns zero seconds and charges
    nothing. The alignment texts are the first align_samples GSM8K
    heal-split texts.
    """
    import shutil
    import tempfile

    from wsp.program import Program

    result = base_result(run_id, pool, who, method, ["compile"])
    result["probe_key"] = None
    result["program_hash"] = None
    result["catalog"] = catalog_name
    result["align_samples"] = align_samples
    result["graft_report"] = []
    problem = call_problem(run_id, pool)
    if problem is not None:
        result["error"] = problem
        return result
    try:
        bare = without_heal(program_dict)
        program = Program.from_dict(bare)
        key = probe_key(program_dict, catalog_name, run_id, align_samples)
    except Exception as error:
        result["error"] = "cannot read program: " + describe_error(error)
        return result
    result["probe_key"] = key
    result["program_hash"] = program_hash(bare)
    result["charge_prefix"] = result["program_hash"]
    cached = cache_lookup(PROBE_DICT, key)
    if cached is not None:
        return as_cache_hit(cached)
    if len(program.grafts) == 0:
        result["ok"] = True
        cache_store(PROBE_DICT, key, result)
        return result

    work_dir = tempfile.mkdtemp(prefix="wsp-probe-")
    phase = "setup"
    clock = time.time()
    try:
        setup = setup_run(program, catalog_name, None, None, align_samples)
        result["setup_seconds"] = round(time.time() - clock, 2)
        result["warmed"] = setup["warmed"]
        phase = "compile"
        clock = time.time()
        report = compile_into(program, work_dir, setup)
        result["seconds"]["compile"] = round(time.time() - clock, 2)
        result["graft_report"] = graft_entries(report)
        result["ok"] = True
    except Exception as error:
        result["error"] = describe_error(error)
        result["phase"] = phase
        if phase == "setup":
            result["setup_seconds"] = round(time.time() - clock, 2)
        else:
            result["seconds"]["compile"] = round(time.time() - clock, 2)
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)
        free_gpu()
    charge_in_container(result, "probe", result["seconds"]["compile"])
    if result["ok"]:
        cache_store(PROBE_DICT, key, result)
    return result


@app.function(image=image, gpu=GPU, volumes={VOL: volume}, cpu=2.0, memory=16384, timeout=3600,
              max_containers=GPU_MAX_CONTAINERS, scaledown_window=60,
              retries=modal.Retries(max_retries=1, initial_delay=5.0))
def run_candidate(program_dict, split, token, catalog_name, eval_limit=None, align_samples=ALIGN_SAMPLES,
                  run_id=None, pool=None, who="", method="", tasks=None, scorer="interim"):
    """Compile, heal and evaluate one program in ONE container, timing and charging each phase.

    token is the eval token the caller already spent with spend(); this
    function never pops one, so its single retry cannot spend twice. run_id
    and pool are required: the result is cached for that run only, and the
    container writes one compute row per phase (compile, heal, eval) to that
    run and pool as each phase finishes. The model is compiled to
    container-local /tmp and loaded from there, never round-tripped through
    the volume.

    Returns {"ok", "error", "phase" (where it failed), "scores": {"gsm8k":
    accuracy}, "n_items", "compile_report", "seconds": {"compile", "heal",
    "eval"}, "setup_seconds", "warmed", "eval_backend": "interim", "label":
    "INTERIM", "split", "token", "program_hash", "heal_loss", "samples",
    "run_id", "pool", "who", "method", "attempt", "charge_prefix",
    "charged", "uncharged", "cache_hit", "gpu"}. "compile" seconds run from
    the end of setup to the heal loop (assembly and the ridge fit); "heal" is
    the heal loop the compiler timed; "eval" covers loading the compiled
    model and generation. A failed phase is charged for the seconds it ran.

    scorer "interim" (the default) grades gsm8k with wsp/swarm/interim_eval.py
    and takes any eval_limit. scorer "harness" grades every task in `tasks`
    (gsm8k, if_guard) with wsp_eval.harness.evaluate on the hf backend, and
    only whole splits, so it refuses an eval_limit. tasks None means ["gsm8k"].
    """
    import shutil
    import tempfile

    from wsp.program import Program

    if tasks is None:
        tasks = [TASK]
    tasks = list(tasks)
    result = base_result(run_id, pool, who, method, ["compile", "heal", "eval"])
    result.update({"scores": {}, "n_items": 0, "compile_report": None, "eval_backend": scorer,
                   "label": SCORER_LABELS.get(scorer, "UNKNOWN"), "split": split, "token": token,
                   "catalog": catalog_name, "eval_limit": eval_limit, "align_samples": align_samples,
                   "program_hash": None, "heal_loss": [], "samples": [], "tasks": tasks})
    problem = call_problem(run_id, pool)
    if problem is None and not valid_token(token):
        problem = "run_candidate needs the eval token the caller spent (an int >= 0), got " + repr(token)
    if problem is None and split not in SPLITS:
        problem = "unknown split " + repr(split) + "; choose " + ", ".join(SPLITS)
    if problem is None:
        problem = scorer_problem(scorer, tasks, eval_limit)
    if problem is not None:
        result["error"] = problem
        return result
    try:
        program = Program.from_dict(program_dict)
        result["program_hash"] = program_hash(program_dict)
        key = result_key(program_dict, split, catalog_name, run_id, eval_limit, align_samples, scorer, tasks)
    except Exception as error:
        result["error"] = "cannot read program: " + describe_error(error)
        return result
    result["charge_prefix"] = result["program_hash"] + ":" + str(token)
    cached = cache_lookup(CACHE_DICT, key)
    if cached is not None:
        hit = as_cache_hit(cached)
        hit["token"] = token
        return hit

    work_dir = tempfile.mkdtemp(prefix="wsp-run-")
    phase = "setup"
    clock = time.time()
    try:
        setup = setup_run(program, catalog_name, split, eval_limit, align_samples)
        if scorer == "harness":
            setup_harness(split, tasks)
        result["setup_seconds"] = round(time.time() - clock, 2)
        result["warmed"] = setup["warmed"]

        phase = "compile"
        clock = time.time()
        report = compile_into(program, work_dir, setup)
        compile_and_heal = time.time() - clock
        heal_s = heal_seconds(report)
        result["seconds"]["compile"] = round(compile_and_heal - heal_s, 2)
        result["seconds"]["heal"] = round(heal_s, 2)
        result["heal_loss"] = heal_losses(report)
        result["compile_report"] = compact_report(report)
        charge_in_container(result, "compile", result["seconds"]["compile"])
        charge_in_container(result, "heal", result["seconds"]["heal"])
        free_gpu()

        phase = "eval"
        clock = time.time()
        if scorer == "harness":
            scored = evaluate_harness(work_dir, split, tasks, run_id)
        else:
            scored = evaluate_model(work_dir, setup["items"], report)
        result["seconds"]["eval"] = round(time.time() - clock, 2)
        charge_in_container(result, "eval", result["seconds"]["eval"])
        if scorer == "harness":
            result["scores"] = scored["scores"]
            result["n_items_per_task"] = scored["n_items_per_task"]
        else:
            result["scores"] = {TASK: scored["accuracy"]}
            result["samples"] = scored["samples"]
        result["n_items"] = scored["n_items"]
        result["ok"] = True
    except Exception as error:
        result["error"] = describe_error(error)
        result["phase"] = phase
        elapsed = round(time.time() - clock, 2)
        if phase == "setup":
            result["setup_seconds"] = elapsed
        elif result["seconds"][phase] == 0.0:
            # Only the phase in progress is charged here; a phase with seconds set was charged already.
            result["seconds"][phase] = elapsed
            charge_in_container(result, phase, elapsed)
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)
        free_gpu()
    if result["ok"]:
        cache_store(CACHE_DICT, key, result)
    return result


@app.function(image=image, gpu=GPU, volumes={VOL: volume}, cpu=2.0, memory=16384, timeout=3600,
              max_containers=GPU_MAX_CONTAINERS, scaledown_window=60)
def host_baseline(split, eval_limit=None, host=HOST):
    """Score the plain host once with the interim scorer, and cache it in wsp-cache for every run.

    A shared reference, so it spends no token and is charged to no run.
    Returns {"ok", "error", "phase", "scores": {"gsm8k": accuracy},
    "n_items", "seconds": {"eval": s}, "setup_seconds", "warmed",
    "eval_backend": "interim", "label", "split", "host", "samples",
    "cache_hit", "gpu"}. "eval" covers loading the host and generation.
    """
    key = host_key(host, split, eval_limit)
    cached = cache_lookup(CACHE_DICT, key)
    if cached is not None:
        return as_cache_hit(cached)

    result = {"ok": False, "error": None, "phase": None, "scores": {}, "n_items": 0, "seconds": {"eval": 0.0},
              "setup_seconds": 0.0, "warmed": [], "eval_backend": EVAL_BACKEND, "label": "INTERIM",
              "split": split, "host": host, "eval_limit": eval_limit, "samples": [], "cache_hit": False,
              "gpu": gpu_name()}
    phase = "setup"
    clock = time.time()
    try:
        setup = setup_host(host, split, eval_limit)
        result["setup_seconds"] = round(time.time() - clock, 2)
        result["warmed"] = setup["warmed"]
        phase = "eval"
        clock = time.time()
        scored = score_host(host, setup["items"])
        result["seconds"]["eval"] = round(time.time() - clock, 2)
        result["scores"] = {TASK: scored["accuracy"]}
        result["n_items"] = scored["n_items"]
        result["samples"] = scored["samples"]
        result["ok"] = True
    except Exception as error:
        result["error"] = describe_error(error)
        result["phase"] = phase
        if phase == "setup":
            result["setup_seconds"] = round(time.time() - clock, 2)
        else:
            result["seconds"]["eval"] = round(time.time() - clock, 2)
    finally:
        free_gpu()
    if result["ok"]:
        cache_store(CACHE_DICT, key, result)
    return result


@app.function(image=image, gpu=GPU, volumes={VOL: volume}, cpu=2.0, memory=16384, timeout=1800,
              max_containers=1, scaledown_window=60)
def extract_parts(layers=None, sizes=None, n_texts=EXTRACT_TEXTS):
    """Build the pilot part set with Trisanth's stub extractor (wsp/extract.py).

    For each donor layer and size, takes the top n MLP neurons by mean
    absolute activation on the first n_texts GSM8K heal-split texts, as
    modal_graft.py does, and saves the index file to /vol/units/pilot/<id>.pt.
    Returns {"records": [catalog records], "seconds": {"load", "extract",
    "per_part": {id: s}}, "gpu"}. Runs the whole 1.5B donor, which is small.
    """
    import torch
    import transformers

    from wsp.extract import top_activation_neurons

    if layers is None:
        layers = PILOT_LAYERS
    if sizes is None:
        sizes = PILOT_SIZES
    start = time.time()
    search_items, heal_texts = gsm8k_data()
    texts = heal_texts[:n_texts]
    model = transformers.AutoModelForCausalLM.from_pretrained(DONOR, dtype=torch.bfloat16).to("cuda").eval()
    tokenizer = transformers.AutoTokenizer.from_pretrained(DONOR)
    loaded = time.time()
    n_layers = model.config.num_hidden_layers
    os.makedirs(VOL + "/" + PILOT_UNITS_DIR, exist_ok=True)
    records = []
    per_part = {}
    for layer in layers:
        if layer < 0 or layer >= n_layers:
            raise ValueError("donor layer " + str(layer) + " is outside 0.." + str(n_layers - 1))
        for n in sizes:
            part_start = time.time()
            indices = top_activation_neurons(model, tokenizer, texts, model.model.layers[layer].mlp, n)
            record = pilot_record(layer, n, n_texts)
            torch.save({"kind": "mlp_cluster", "model": DONOR, "layer": layer, "indices": indices},
                       VOL + "/" + record["units"])
            records.append(record)
            per_part[record["id"]] = round(time.time() - part_start, 2)
    volume.commit()
    del model
    free_gpu()
    seconds = {"load": round(loaded - start, 2), "extract": round(time.time() - loaded, 2), "per_part": per_part}
    return {"records": records, "seconds": seconds, "gpu": gpu_name()}
