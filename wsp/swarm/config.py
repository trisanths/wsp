"""Run config, a small .env loader and run ids.

A config is a plain dict. make_config(overrides) copies DEFAULTS and applies
the overrides. Nothing in this module prints a secret value.
"""

import copy
import os
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DEFAULTS = {
    # Who and where.
    "run_id": None,              # None: make_run_id(method, seed) when the run is built
    "method": "swarm",
    "seed": 0,
    "outputs_dir": "outputs",
    "catalog_path": None,        # None: the stub catalog for the stub backend, else catalog.jsonl

    # Pluggable pieces, chosen by name in factory.py.
    "budget": "local",           # local | modal
    "backend": "stub",           # stub | local | modal
    "memory": "off",             # off | past
    "literature": "off",         # off | replay | live
    "llm": "none",               # none | mock | modal_vllm | openai_compat | anthropic
    "agents": "auto",            # auto (scripted when llm is none) | scripted | llm

    # The two budget axes (spec section 5). Evals are full-eval equivalents (FEE).
    "eval_budget": 60,
    "compute_budget_s": 36000.0,
    "heldout_eval_budget": 9,    # 3 programs, each with 2 controls
    "heldout_compute_budget_s": 20000.0,

    # Fidelity ladder (spec section 14). Each level costs this fraction of one
    # FEE. The levels are nested question sets, so promoting a program pays
    # only the difference.
    "fidelities": {"check": 0.16, "low": 0.34, "full": 1.0},

    # Task setup (spec 4.2).
    "task_type": "math",
    "task_types": {
        "math": {"tasks": {"gsm8k": 1.0}, "guards": ["if_guard"], "heal_data": ["gsm8k_train"]},
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
    "host": "unsloth/Llama-3.2-1B-Instruct",
    "host_layers": 16,
    "donor_layers": {            # used for the default depth; extend for new donors
        "Qwen/Qwen2.5-Math-1.5B": 28,
        "Qwen/Qwen2.5-Coder-1.5B": 28,
        "meta-llama/Llama-3.2-3B-Instruct": 28,
        "allenai/OLMoE-1B-7B-0924": 16,
    },
    "max_grafts": 4,
    "heal_steps_allowed": [0, 100, 300, 600],
    "heal_lr_allowed": [3e-4],
    "heal_batch_size_allowed": [8],
    "heal_seq_len_allowed": [384],
    "fit_threshold": 0.3,        # the fit gate, the same for every arm

    # Swarm (spec section 8).
    "planners": 3,
    "k": 4,                      # candidates evaluated per round
    "analyst_every": 5,
    "frontier_size": 8,
    "kill_after": 2,
    "retire_after": 3,
    "tie_margin": 0.01,          # judge ties break toward fewer parameters
    "heldout_top": 3,
    "cascade": False,            # True: evaluate at "low", promote the top third to "full"
    "max_rounds": 200,
    "max_stall_rounds": 5,
    "workers": 4,                # threads for step_many
    "memory_cleanup": True,

    # Stub world.
    "stub_seed": 0,

    # Local backend (Trisanth's compile on this machine).
    "local_eval": "wsp_eval.harness:swarm_eval",
    "align_samples": 64,
}

SECRET_WORDS = ["SECRET", "TOKEN", "KEY", "PASSWORD", "CREDIT"]


def make_config(overrides=None):
    """Return DEFAULTS plus overrides, as a new dict.

    Keys that are not in DEFAULTS are kept, so integration modules can read
    their own settings (for example a Modal app name) from the same config.
    """
    config = copy.deepcopy(DEFAULTS)
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
