"""Benchmarks for a request, chosen by Trisanth's web research (studio/server/bench_research.py).

Part 3 does not hand-write a task type per domain. For a request such as "a
small model that answers medical exam questions" it runs the studio's own
research pipeline:

1. a spec is drafted in memory with the studio's rules (studio/server/specs.py:
   title, summary, size limits, the rule task type and its default benchmarks);
2. bench_research.pipeline plans 2 to 4 web searches, runs them with Exa (or
   OpenAI web search), and has the LLM propose 3 to 6 benchmarks with reasons
   and reference scores that are checked against the pages it read;
3. bench_research.apply_to_spec keeps the ids the eval harness can run and adds
   them to the spec as targets or guards;
4. the task setup follows specs.run_overrides: every target weighted 1.0, the
   guards as listed, plus the heal data.

Nothing here knows about a particular benchmark. The research runs once per
race and is saved to a file (scripts/research_benchmarks.py), because Exa and
the LLM are not deterministic and every arm must race on the same benchmarks.
The runners read it with --benchmarks.

Heal data: specs.run_overrides heals on gsm8k_train. Trisanth's studio notes
(studio/server/settings.py, HEAL_NOTE) say healing on dataset text currently
damages the base, so this setup defaults to one of his heal-data pools
(wsp/data.py: general, math, code, medicine, mix) instead.
"""

import re

from wsp.swarm.config import load_dotenv

TASK_TYPE = "researched"     # the task_types key a researched setup runs under


def slug_for(request):
    """A short file-safe name for a request."""
    text = re.sub(r"[^a-z0-9]+", "-", str(request).lower()).strip("-")
    return text[:60] or "request"


def draft_spec(request, size_b=None):
    """A spec dict drafted with the studio's rules, kept in memory (never stored by the studio)."""
    from studio.server import catalog as catalog_mod
    from studio.server import settings
    from studio.server import specs as specs_mod

    config, _catalog = catalog_mod.load_catalog(settings.backend())
    specs_mod.add_task_types(config)
    config["task_type"] = specs_mod.task_type_for(request)
    host_params = catalog_mod.host_params_b(config["host"])
    if size_b is None:
        size_b = specs_mod.size_from_prompt(request, host_params)
    limits = {"max_total_params_b": size_b,
              "max_run_params_b": specs_mod.run_limit_from_prompt(request, size_b, host_params),
              "max_heal_min": specs_mod.heal_limit_from_prompt(request)}
    return {"id": "research-" + slug_for(request), "title": specs_mod.draft_title(request),
            "summary": specs_mod.draft_summary(request, size_b), "prompt": str(request),
            "task_type": config["task_type"], "host": config["host"], "limits": limits,
            "request_type": "build", "benchmarks": specs_mod.default_benchmarks(config)}


def task_setup(spec, heal_data):
    """The swarm task setup for a spec, by the rule of studio specs.run_overrides."""
    targets = {}
    guards = []
    for item in spec["benchmarks"]:
        if item["kind"] == "target":
            targets[item["id"]] = 1.0
        else:
            guards.append(item["id"])
    return {"tasks": targets, "guards": guards, "heal_data": [heal_data]}


def research_setup(request, heal_data="general", size_b=None):
    """Run the studio's benchmark research for a request. Returns the record the runners read.

    Raises when the research cannot run (no LLM, or the LLM proposes nothing usable).
    """
    from studio.server import bench_research
    from studio.server import settings

    load_dotenv()
    if settings.llm() == "none":
        raise RuntimeError("benchmark research needs an LLM: set WSP_STUDIO_LLM (modal_vllm, openai_compat or "
                           "anthropic) or put OPENAI_API_KEY or ANTHROPIC_API_KEY in .env")
    spec = draft_spec(request, size_b)
    run = bench_research.Run(spec["id"], bench_research.choose_provider())
    bench_research.pipeline(run, spec)
    run.close("done")
    bench_research.apply_to_spec(spec, run.data["proposals"])
    return {
        "request": str(request),
        "title": spec["title"],
        "studio_task_type": spec["task_type"],
        "task_type": TASK_TYPE,
        "task_setup": task_setup(spec, heal_data),
        "benchmarks": spec["benchmarks"],
        "proposals": run.data["proposals"],
        "reasoning": run.data.get("reasoning", ""),
        "sources": run.data.get("sources", []),
        "steps": run.data["steps"],
        "provider": run.data["provider"],
        "llm": settings.llm(),
        "usage": run.data["usage"],
    }


def config_overrides(record):
    """The make_config overrides that run a researched setup: task_type "researched" and its setup."""
    setup = record["task_setup"]
    tasks = {}
    for task in setup["tasks"]:
        tasks[task] = float(setup["tasks"][task])
    return {"task_type": TASK_TYPE,
            "task_types": {TASK_TYPE: {"tasks": tasks, "guards": list(setup["guards"]),
                                       "heal_data": list(setup["heal_data"])}}}
