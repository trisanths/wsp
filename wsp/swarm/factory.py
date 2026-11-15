"""Build budgets, backends, memory, literature, LLMs, agents and the env from config.

Each choice is a config value. The module behind a choice is imported only
when that choice is selected, so runs work before the integration modules
exist. A missing module raises FactoryError naming the file to build.

Constructor contracts (the integration agents build against these):
    budget    "local": LocalBudget.from_config(config, pool, ledger)
              "modal": wsp.swarm.modal_backend.ModalBudget.from_config(config, pool, ledger)
    backend   "stub" | "local": wsp.swarm.backends.<Class>(config, catalog)
              "modal": wsp.swarm.modal_backend.ModalCompileEval(config, catalog)
    memory    "off": NullMemory(config); "past": wsp.swarm.memory.PastMemory(config)
    literature "off": NullLiterature(config, mode); "replay" | "live":
              wsp.swarm.literature.ExaLiterature(config, mode)
    llm       "none": NullLLM(config); "mock" | "modal_vllm" | "openai_compat" |
              "anthropic": wsp.swarm.llm.<Class>(config)
    agents    "scripted": ScriptedAnalyst(config), ScriptedPlanner(index, config), RuleJudge(config)
              "llm": wsp.swarm.agents.llm_agents.LLMAnalyst(config),
                     LLMPlanner(index, config, batch), where every planner shares
                     one PlannerBatch(planners, llm_batch_wait_s); the judge stays RuleJudge
"""

import hashlib
import importlib
import os

from wsp.catalog import Catalog
from wsp.swarm import config as config_mod
from wsp.swarm import stub
from wsp.swarm.budget import Ledger
from wsp.swarm.env import WspEnv
from wsp.swarm.log import RunLog

CHOICES = {
    "budget": {
        "local": ("wsp.swarm.budget", "LocalBudget"),
        "modal": ("wsp.swarm.modal_backend", "ModalBudget"),
    },
    "backend": {
        "stub": ("wsp.swarm.backends", "StubCompileEval"),
        "local": ("wsp.swarm.backends", "LocalCompileEval"),
        "modal": ("wsp.swarm.modal_backend", "ModalCompileEval"),
    },
    "memory": {
        "off": ("wsp.swarm.interfaces", "NullMemory"),
        "past": ("wsp.swarm.memory", "PastMemory"),
    },
    "literature": {
        "off": ("wsp.swarm.interfaces", "NullLiterature"),
        "replay": ("wsp.swarm.literature", "ExaLiterature"),
        "live": ("wsp.swarm.literature", "ExaLiterature"),
    },
    "llm": {
        "none": ("wsp.swarm.interfaces", "NullLLM"),
        "mock": ("wsp.swarm.llm", "MockLLM"),
        "modal_vllm": ("wsp.swarm.llm", "ModalVLLM"),
        "openai_compat": ("wsp.swarm.llm", "OpenAICompatLLM"),
        "anthropic": ("wsp.swarm.llm", "AnthropicLLM"),
    },
}

# agents choice -> (module, analyst class, planner class). The judge is always RuleJudge.
AGENT_CLASSES = {
    "scripted": ("wsp.swarm.agents.scripted", "ScriptedAnalyst", "ScriptedPlanner"),
    "llm": ("wsp.swarm.agents.llm_agents", "LLMAnalyst", "LLMPlanner"),
}


class FactoryError(RuntimeError):
    """A config choice cannot be built (unknown name or a module not built yet)."""


def module_file(module_name):
    return module_name.replace(".", "/") + ".py"


def import_class(module_name, class_name, what):
    """Import module_name.class_name. Raises FactoryError naming the missing file or class."""
    try:
        module = importlib.import_module(module_name)
    except ImportError as error:
        if error.name == module_name:
            raise FactoryError(what + " needs " + module_file(module_name) + " (class " + class_name +
                               "), which is not built yet")
        raise FactoryError(what + ": importing " + module_name + " failed: " + str(error))
    if not hasattr(module, class_name):
        raise FactoryError(what + ": " + module_file(module_name) + " has no class " + class_name)
    return getattr(module, class_name)


def load_class(slot, choice):
    """Import the class behind a config choice, for example load_class("memory", "past")."""
    if choice not in CHOICES[slot]:
        raise FactoryError("unknown " + slot + " '" + str(choice) + "'; choose one of " +
                           ", ".join(sorted(CHOICES[slot])))
    module_name, class_name = CHOICES[slot][choice]
    return import_class(module_name, class_name, slot + " '" + choice + "'")


def build_catalog(config):
    """The run's catalog: config catalog_path, else the stub catalog (stub backend), else catalog.jsonl."""
    path = config["catalog_path"]
    if path is None and config["backend"] == "stub":
        path = stub.write_stub_catalog(os.path.join(config_mod.stub_dir(config), "catalog.jsonl"),
                                       config["stub_seed"])
    if path is None:
        path = os.path.join(config_mod.REPO_ROOT, "catalog.jsonl")
    return Catalog.load(path), path


def build_budget(config, pool, ledger):
    return load_class("budget", config["budget"]).from_config(config, pool, ledger)


def build_backend(config, catalog):
    return load_class("backend", config["backend"])(config, catalog)


def build_memory(config):
    return load_class("memory", config["memory"])(config)


def build_literature(config):
    return load_class("literature", config["literature"])(config, config["literature"])


def build_llm(config):
    return load_class("llm", config["llm"])(config)


def agents_choice(config):
    choice = config["agents"]
    if choice == "auto":
        if config["llm"] == "none":
            return "scripted"
        return "llm"
    return choice


def build_agents(config):
    """Return (analyst, planners, judge)."""
    choice = agents_choice(config)
    if choice not in AGENT_CLASSES:
        raise FactoryError("unknown agents '" + str(choice) + "'; choose auto, scripted or llm")
    module_name, analyst_name, planner_name = AGENT_CLASSES[choice]
    what = "agents '" + choice + "'"
    analyst_class = import_class(module_name, analyst_name, what)
    planner_class = import_class(module_name, planner_name, what)
    judge_class = import_class("wsp.swarm.agents.scripted", "RuleJudge", what)
    batch = None
    if choice == "llm":
        # One shared batch, so a round of planners is one llm.complete_many call.
        batch_class = import_class(module_name, "PlannerBatch", what)
        wait_s = config.get("llm_batch_wait_s")
        if wait_s is None:
            wait_s = 30.0
        batch = batch_class(config["planners"], wait_s)
    planners = []
    for index in range(config["planners"]):
        if batch is None:
            planners.append(planner_class(index, config))
        else:
            planners.append(planner_class(index, config, batch))
    return analyst_class(config), planners, judge_class(config)


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        digest.update(handle.read())
    return digest.hexdigest()


def build_run(config):
    """Build the env and everything around it for one run.

    Fills config["run_id"] when empty. Returns a dict with config, catalog,
    catalog_path, ledger, budget, heldout_budget, backend, log, env, memory,
    literature and llm. Writes outputs/runs/<run_id>/manifest.json.
    """
    config_mod.ensure_run_id(config)
    # Load every chosen class first, so a missing module fails before any run file is written.
    budget_class = load_class("budget", config["budget"])
    backend_class = load_class("backend", config["backend"])
    memory = build_memory(config)
    literature = build_literature(config)
    llm = build_llm(config)
    catalog, catalog_path = build_catalog(config)
    ledger = Ledger(config_mod.ledger_path(config))
    budget = budget_class.from_config(config, "search", ledger)
    heldout_budget = budget_class.from_config(config, "heldout", ledger)
    backend = backend_class(config, catalog)
    log = RunLog(config_mod.run_dir(config))
    env = WspEnv(config, catalog, backend, budget, log, heldout_budget)
    env.write_manifest({
        "run_id": config["run_id"],
        "method": config["method"],
        "seed": config["seed"],
        "backend": backend.name,
        "stub": backend.name == "stub",
        "catalog_path": os.path.abspath(catalog_path),
        "catalog_sha256": file_sha256(catalog_path),
        "ledger_path": os.path.abspath(ledger.path),
        "config": config_mod.public_config(config),
    })
    return {"config": config, "catalog": catalog, "catalog_path": catalog_path, "ledger": ledger,
            "budget": budget, "heldout_budget": heldout_budget, "backend": backend, "log": log, "env": env,
            "memory": memory, "literature": literature, "llm": llm}
