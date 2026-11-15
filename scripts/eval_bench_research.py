"""Run the benchmark research pipeline directly (no server) on a list of requests.

Usage:
    python scripts/eval_bench_research.py --tag before            # the 8 default requests
    python scripts/eval_bench_research.py --tag after --only 0,3  # a subset by index
    python scripts/eval_bench_research.py --requests my.json      # a JSON list of strings
    python scripts/eval_bench_research.py --offline               # fake search and LLM, no network

Each request writes outputs/bench_research_eval/<tag>/<index>.json with the
steps, proposals, sources, timings, token usage, the first search results, the
raw LLM replies and the page text the score check saw (so a grader can check
numbers, and a checker change can be replayed, without a new search).
Real runs cost a few cents each and need OPENAI_API_KEY and EXA_API_KEY in .env.
"""

import argparse
import json
import os
import sys
import tempfile
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

REQUESTS = [
    "A small model for answering legal questions, under 2B parameters",
    "A coding assistant under 2B parameters that runs on a laptop",
    "A model that helps nurses answer medical questions, under 3B parameters",
    "A math tutor for grade school word problems that runs on a phone",
    "A customer support model for a telecom company that follows strict response formats",
    "A model for translating between English and Spanish, under 1B parameters",
    "A coding-only model: drop everything else from Qwen2.5-Coder-7B and get it under 3B",
    "A general assistant that is as small as possible but still good at reasoning puzzles",
]

OFFLINE_PAGE = {"title": "Offline page", "url": "https://example.org/offline",
                "text": "Model-1B scores 40.0 on the offline benchmark."}


def fake_search(query, run):
    return {"summary": "", "results": [dict(OFFLINE_PAGE)]}


def fake_llm(messages, schema, max_tokens=2500):
    text = json.dumps(schema)
    if "queries" in text:
        return json.dumps({"plan": "Offline plan.", "queries": ["offline one", "offline two"]})
    if "scores" in text:
        return json.dumps({"scores": []})
    return json.dumps({"reasoning": "Offline.", "benchmarks": [
        {"name": "GSM8K", "measures": "math", "why": "fits", "runnable_id": "gsm8k", "kind": "target",
         "reference_scores": [], "suggested_target": 0.3}]})


def install(bench_research, record, offline):
    """Replace the search and LLM calls with metered versions that fill the record."""
    from wsp.swarm import factory
    if offline:
        bench_research.choose_provider = lambda: "exa"
        bench_research.search_exa = fake_search
        real_scores = lambda name, size_b: [dict(OFFLINE_PAGE)]
        real_ask = fake_llm
    else:
        real_scores = bench_research.search_exa_scores
        real_ask = None

    def ask(messages, schema, max_tokens=2500):
        if real_ask is not None:
            reply = real_ask(messages, schema, max_tokens)
        else:
            llm = factory.build_llm(bench_research.make_config())
            reply = llm.complete(messages, schema=schema, temperature=0.3, max_tokens=max_tokens)
            for use in llm.usage:
                record["llm_calls"].append({"seconds": use.get("seconds"), "model": use.get("model"),
                                            "input_tokens": use.get("input_tokens"),
                                            "output_tokens": use.get("output_tokens")})
        record["raw_replies"].append(reply)
        return reply

    def scores(*args):
        pages = real_scores(*args)
        for page in pages:
            record["score_pages"][page["url"]] = page["text"]
        return pages

    real_search = bench_research.search_exa

    def search(query, run):
        outcome = real_search(query, run)
        record["searches"][query] = outcome
        return outcome

    bench_research.ask_llm = ask
    bench_research.search_exa_scores = scores
    bench_research.search_exa = search


def run_one(prompt, offline):
    from studio.server import bench_research
    from studio.server import specs as specs_mod
    record = {"prompt": prompt, "llm_calls": [], "raw_replies": [], "score_pages": {}, "searches": {},
              "offline": offline}
    install(bench_research, record, offline)
    start = time.time()
    spec = specs_mod.draft_spec(prompt, [])
    record["draft_seconds"] = round(time.time() - start, 2)
    record["limits"] = spec.get("limits")
    record["request_type"] = spec.get("request_type")
    run = bench_research.Run(spec["id"], bench_research.choose_provider())
    start = time.time()
    try:
        bench_research.pipeline(run, spec)
        run.close("done")
    except Exception as error:
        run.close("error", str(error))
    record["research_seconds"] = round(time.time() - start, 2)
    record["research"] = run.data
    inp = 0
    out = 0
    for call in record["llm_calls"]:
        inp += call["input_tokens"] or 0
        out += call["output_tokens"] or 0
    record["llm_input_tokens"] = inp
    record["llm_output_tokens"] = out
    return record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", default="run")
    parser.add_argument("--requests", default=None, help="JSON file with a list of request strings")
    parser.add_argument("--only", default=None, help="comma separated indexes")
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()
    if not os.environ.get("WSP_STUDIO_OUT"):
        os.environ["WSP_STUDIO_OUT"] = tempfile.mkdtemp(prefix="bench_research_eval_")
    from studio.server import settings
    settings.load_env()
    requests = REQUESTS
    if args.requests:
        with open(args.requests, encoding="utf-8") as handle:
            requests = json.load(handle)
    indexes = list(range(len(requests)))
    if args.only:
        indexes = [int(part) for part in args.only.split(",")]
    folder = os.path.join(ROOT, "outputs", "bench_research_eval", args.tag)
    os.makedirs(folder, exist_ok=True)
    for index in indexes:
        record = run_one(requests[index], args.offline)
        with open(os.path.join(folder, str(index) + ".json"), "w", encoding="utf-8") as handle:
            json.dump(record, handle, indent=2)
        research = record["research"]
        sourced = 0
        for proposal in research["proposals"]:
            for entry in proposal["reference_scores"]:
                if entry.get("sourced"):
                    sourced += 1
        print(index, research["status"], "proposals", len(research["proposals"]), "sourced", sourced,
              "seconds", record["research_seconds"], "tokens", record["llm_input_tokens"],
              record["llm_output_tokens"])


if __name__ == "__main__":
    main()
