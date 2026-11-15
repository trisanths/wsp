"""Run config, a small .env loader and run ids.

A config is a plain dict. make_config(overrides) copies two tables and then
applies the overrides:

- DEFAULTS: what every arm of a comparison shares (budgets, the program
  space, the fidelity ladder, the swarm sizes, the backend). scripts/compare.py
  refuses to let one arm change a key in this table.
- INTEGRATION_DEFAULTS: settings only the LLM, memory and literature clients
  read (llm.py, agents/llm_agents.py, factory.py, memory.py, literature.py).
  Only the swarm_llm arm uses them, and compare.py's --llm-config may set
  them for that arm, which is why they live apart from DEFAULTS. Each value
  equals the fallback its module uses when the key is missing, so a config
  from make_config and a bare dict behave the same.

Nothing in this module prints a secret value.
"""

import copy
import os
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DEFAULTS = {
    # Who and where.
    "run_id": None,              # None: make_run_id(method, seed) when the run is built
    "method": "swarm",           # the arm name stamped on every log and ledger row
    "seed": 0,                   # the run seed; every rng in the run derives from it
    "outputs_dir": "outputs",    # run folders, the ledger and the stub catalog go under here
    "catalog_path": None,        # None: the stub catalog for the stub backend, else catalog.jsonl

    # Pluggable pieces, chosen by name in factory.py.
    "budget": "local",           # local | modal
    "backend": "stub",           # stub | local | modal
    "modal_scorer": "interim",   # modal backend only: interim (gsm8k, task_type math_interim) | harness (gsm8k + if_guard)
    "memory": "off",             # off | past
    "literature": "off",         # off | replay | live
    "llm": "none",               # none | mock | modal_vllm | openai_compat | anthropic
    "agents": "auto",            # auto (scripted when llm is none) | scripted | llm

    # The two budget axes (spec section 5). Evals are full-eval equivalents (FEE).
    "eval_budget": 60,           # search evals per run, in FEE
    "compute_budget_s": 36000.0,  # search compute per run in seconds (compile, heal and eval all count)
    "heldout_eval_budget": 9,    # 3 programs, each with 2 controls
    "heldout_compute_budget_s": 20000.0,  # held-out compute per run, in seconds

    # Fidelity ladder (spec section 14). Each level costs this fraction of one
    # FEE. The levels are nested question sets, so promoting a program pays
    # only the difference when both levels score the same model (same heal
    # steps); a promotion that heals again pays the whole level (env.py).
    "fidelities": {"check": 0.16, "low": 0.34, "full": 1.0},  # FEE per eval at each level
    # Cascade heal (spec section 14): every fidelity below "full" heals with
    # this many steps (a program that asks for fewer keeps its own), and a
    # promotion to full heals again at the program's own steps. papers.md
    # recommendation 4 gives 33 to 100 steps for a short heal. None heals at
    # the program's own steps at every fidelity (the behavior before this key).
    "low_fidelity_heal_steps": 50,  # short heal below full; None: the program's own steps everywhere

    # Task setup (spec 4.2).
    "task_type": "math",         # which entry of task_types this run targets
    "task_types": {              # per task type: target tasks and weights, guard tasks, heal data
        "math": {"tasks": {"gsm8k": 1.0}, "guards": ["if_guard"], "heal_data": ["gsm8k_train"]},
        # Modal runs: the interim scorer in modal_swarm.py grades gsm8k only, so no guard.
        "math_interim": {"tasks": {"gsm8k": 1.0}, "guards": [], "heal_data": ["gsm8k_train"]},
    },
    "guard_drop": 0.10,          # a guard is penalized when it falls more than 10% below base
    "guard_penalty": 1.0,        # penalty = guard_penalty * relative drop
    "collapse_fraction": 0.5,    # collapse when any task falls below half its base
    # Host scores, needed by the local backend. Search split measured 2026-11-15 by
    # modal_eval.py::base_scores (hf backend, same prompts as every candidate).
    # Held-out (GSM8K test n=1319, if_guard n=410) measured once, 2026-11-15, by
    # modal_eval.py::base_scores --split heldout; the held-out ledger refuses a rerun.
    "base_scores": {"search": {"gsm8k": 0.23, "if_guard": 0.42},
                    "heldout": {"gsm8k": 0.201, "if_guard": 0.488}},

    # Program space (spec 4.1).
    "host": "unsloth/Llama-3.2-1B-Instruct",  # the one host every program grafts into
    "host_layers": 16,           # L_host when the catalog has no host record
    "donor_layers": {            # used for the default depth; extend for new donors
        "Qwen/Qwen2.5-Math-1.5B": 28,
        "Qwen/Qwen2.5-Coder-1.5B": 28,
        "meta-llama/Llama-3.2-3B-Instruct": 28,
        "allenai/OLMoE-1B-7B-0924": 16,
    },
    "max_grafts": 4,             # grafts per program
    "glue_ranks": [None, 64],    # allowed glue ranks, None is full rank (round 2: r64 matched it at 7% of the params)
    "heal_steps_allowed": [0, 100, 300, 600],  # the heal steps a program may ask for
    "heal_lr_allowed": [3e-4],   # one heal recipe for every arm: the learning rate,
    "heal_batch_size_allowed": [8],  # the batch size,
    "heal_seq_len_allowed": [384],   # and the sequence length
    "fit_threshold": 0.3,        # the fit gate, the same for every arm

    # Swarm (spec section 8).
    "planners": 3,               # planners proposing in parallel each round
    "k": 4,                      # candidates evaluated per round
    "analyst_every": 5,          # rebuild the analyst map every this many rounds
    "frontier_size": 8,          # programs the judge keeps
    "kill_after": 2,             # a pattern dies after this many collapses or fit rejections
    "retire_after": 3,           # a parent retires after this many failed children
    "tie_margin": 0.01,          # judge ties break toward fewer parameters
    "heldout_top": 3,            # programs per run scored once on held-out
    "cascade": False,            # True: evaluate at "low", promote the top third to "full"
    "max_rounds": 200,           # hard stop on rounds
    "max_stall_rounds": 5,       # stop after this many rounds without a new candidate
    "workers": 4,                # threads for step_many
    "memory_cleanup": True,      # end_run deletes what the run sent to memory

    # Stub world.
    "stub_seed": 0,              # seeds the stub's hidden part values, fit noise and eval noise

    # Local backend (Trisanth's compile on this machine).
    "local_eval": "wsp_eval.harness:swarm_eval",  # Person B's eval hook, called as fn(model_path, split, tasks)
    "align_samples": 64,         # heal-data texts used to fit the stitch at compile
}

# Read only by the LLM, memory and literature clients (see the module
# docstring). The mock LLM's test hooks (mock_fn, mock_replies) have no
# default: a mock with neither is a test setup error.
INTEGRATION_DEFAULTS = {
    # LLM calls (agents/llm_agents.py, factory.py).
    "llm_temperature": 0.7,      # sampling temperature for analyst and planner calls
    "llm_max_tokens": 2048,      # reply length cap per call
    "llm_batch_wait_s": 30.0,    # how long the shared planner batch waits for every planner
    # LLM clients (llm.py).
    "llm_tries": 3,              # tries per call, counting the first
    "llm_backoff_s": 2.0,        # wait before the second try; doubles after each failure
    "llm_timeout_s": 120.0,      # per-request timeout for the HTTP clients, in seconds
    "llm_workers": 4,            # threads for complete_many on the HTTP clients
    "openai_base_url": None,     # OpenAICompatLLM: server URL ending in /v1 (None: OpenAI)
    "openai_model": None,        # OpenAICompatLLM: model name (required when that client is chosen)
    "openai_api_key_env": "OPENAI_API_KEY",  # OpenAICompatLLM: the env var that holds the key
    "modal_llm_app": "wsp-llm",  # ModalVLLM: deployed Modal app (modal_llm.py picks the served model)
    "modal_llm_class": "PlannerLLM",  # ModalVLLM: the modal.Cls name in that app
    "modal_llm_environment": None,    # ModalVLLM: Modal environment (None: the workspace default)
    "anthropic_model": "claude-sonnet-5-5",  # AnthropicLLM: model id
    "anthropic_effort": None,    # AnthropicLLM: low | medium | high | xhigh | max (None: model default)
    "anthropic_fallbacks": "default",  # AnthropicLLM: server-side refusal fallback; None turns it off
    "anthropic_min_max_tokens": 4096,  # AnthropicLLM: floor on max_tokens

    # past.dev memory (memory.py).
    "past_base_url": "https://api.past.dev/api/v1",  # the Memory API root
    "past_api_key_env": "PAST_API_KEY",  # the env var that holds the key
    "memory_level": "low",       # default recall level
    "memory_poll_s": 1.5,        # wait between ingestion status polls, in seconds
    "memory_http_timeout_s": 20.0,  # per-request timeout, in seconds
    "memory_max_retries": 3,     # retries on 429, 5xx and connection errors
    "memory_backoff_s": 1.0,     # first retry wait, doubled each retry
    "memory_membership_timeout_s": 15.0,  # how long start_run waits for the guard identity to join
    "memory_cold_query": "graft score search split collapse lesson fit heal program frontier",  # cold-start recall
    "memory_max_chars": 1500,    # longest text sent per record
    "memory_verify_cleanup": False,  # end_run reads memory back to prove the cleanup
    "memory_state_path": None,   # None: the state file goes under the run folder

    # Exa literature (literature.py).
    "priors_path": "data/priors.json",  # the frozen priors pack from scripts/build_priors.py
    "exa_cache_dir": "exa_cache",  # tier one cache folder (relative to the repo root)
    "exa_modal_cache": False,    # also use the modal.Dict cache tier
    "exa_modal_dict": "exa-cache",  # that modal.Dict's name
    "exa_search_type": "lite",   # "lite" or "auto" for planner searches
    "exa_end_published_date": "2026-11-14T00:00:00Z",  # pinned so cache keys stay stable; None sends no date
    "exa_exclude_domains": [],   # domains excluded on top of the dataset mirrors
    "exa_quote_chars": 400,      # longest quote kept per citation
    "exa_max_query_chars": 400,  # longest query sent (a leak bound)
    "exa_max_live_calls": 100,   # live Exa calls allowed per instance
    "exa_retries": 2,            # retries on 429, 5xx and network errors
    "exa_backoff_s": 2.0,        # first retry wait, doubled each retry
    "exa_timeout_s": 30.0,       # per-request timeout given to the SDK
    "exa_sdk_version": None,     # cache key version; None uses the installed exa-py
}

SECRET_WORDS = ["SECRET", "TOKEN", "KEY", "PASSWORD", "CREDIT"]


def make_config(overrides=None):
    """Return INTEGRATION_DEFAULTS, then DEFAULTS, then the overrides, as a new dict.

    Keys in neither table are kept, so integration modules can read their
    own settings (for example a Modal app name) from the same config.
    """
    config = copy.deepcopy(INTEGRATION_DEFAULTS)
    shared = copy.deepcopy(DEFAULTS)
    for key in shared:
        config[key] = shared[key]
    if overrides is not None:
        for key in overrides:
            config[key] = copy.deepcopy(overrides[key])
    return config


def make_run_id(method, seed):
    """A short unique run id, for example s0-swarm-1794698437123."""
    return "s" + str(seed) + "-" + method + "-" + str(int(time.time() * 1000))


def ensure_run_id(config):
    """Fill config["run_id"] if it is empty. Return the run id."""
    if config["run_id"] is None:
        config["run_id"] = make_run_id(config["method"], config["seed"])
    return config["run_id"]


def run_dir(config):
    return os.path.join(config["outputs_dir"], "runs", config["run_id"])


def ledger_path(config):
    return os.path.join(config["outputs_dir"], "ledger.jsonl")


def stub_dir(config):
    return os.path.join(config["outputs_dir"], "stub")


def task_setup(config):
    return config["task_types"][config["task_type"]]


def task_weights(config):
    """The target tasks and their weights, as a new dict."""
    return dict(task_setup(config)["tasks"])


def guard_tasks(config):
    return list(task_setup(config)["guards"])


def all_tasks(config):
    """Target tasks then guard tasks, in a fixed order."""
    names = []
    for task in sorted(task_weights(config)):
        names.append(task)
    for task in guard_tasks(config):
        if task not in names:
            names.append(task)
    return names


def heal_data_allowed(config):
    return list(task_setup(config)["heal_data"])


def fidelity_cost(config, fidelity):
    """The FEE fraction of one eval at this fidelity."""
    if fidelity not in config["fidelities"]:
        raise KeyError("unknown fidelity: " + str(fidelity))
    return float(config["fidelities"][fidelity])


def load_dotenv(path=None):
    """Read KEY=VALUE lines from .env into os.environ.

    Values already in the environment win. A missing file is fine. Returns
    the list of key names read, never the values.
    """
    if path is None:
        path = os.path.join(REPO_ROOT, ".env")
    names = []
    if not os.path.exists(path):
        return names
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line == "" or line.startswith("#"):
                continue
            if line.startswith("export "):
                line = line[len("export "):]
            if "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
                value = value[1:-1]
            if key == "":
                continue
            names.append(key)
            if key not in os.environ:
                os.environ[key] = value
    return names


def is_secret_name(name):
    upper = name.upper()
    for word in SECRET_WORDS:
        if word in upper:
            return True
    return False


def public_config(config):
    """A copy of the config that is safe to write to a manifest."""
    out = copy.deepcopy(config)
    for key in list(out.keys()):
        if is_secret_name(key):
            out[key] = "<hidden>"
    return out
