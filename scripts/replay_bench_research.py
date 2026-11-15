"""Replay recorded research runs through the current checker. No LLM call and no search.

Usage:
    python scripts/replay_bench_research.py --tag after --only 0,3 --out after_replay

It reads outputs/bench_research_eval/<tag>/<n>.json (written by eval_bench_research.py),
feeds the recorded LLM replies and the recorded pages back into the pipeline, and writes
the new record to outputs/bench_research_eval/<out>/<n>.json.

A record made before the first search results were stored has no "searches". For those, the
pages the first LLM reply cites are fetched again with Exa get-contents (full text, so the
replay is a little easier to pass than a live run). That needs EXA_API_KEY in .env.
"""

import argparse
import json
import os
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def fetch_pages(urls):
    """Full text of each url through Exa get-contents. A failed page is an empty string."""
    from exa_py import Exa
    pages = {}
    for url in urls:
        try:
            found = Exa().get_contents([url], text={"max_characters": 60000})
            pages[url] = found.results[0].text if found.results else ""
        except Exception:
            pages[url] = ""
    return pages


def first_results(record, titles):
    """The first-pass search results to replay: the stored ones, else the cited pages fetched again."""
    if record.get("searches"):
        return list(record["searches"].values())
    synth = json.loads(record["raw_replies"][1])
    urls = []
    for item in synth.get("benchmarks", []):
        for score in item.get("reference_scores") or []:
            url = score.get("source_url")
            if url and url not in urls:
                urls.append(url)
    pages = fetch_pages(urls)
    results = []
    for url in urls:
        results.append({"title": titles.get(url, ""), "url": url, "text": pages[url]})
    return [{"summary": "", "results": results}]


def replay_one(record):
    from studio.server import bench_research as bench
    from studio.server import specs as specs_mod
    spec = specs_mod.draft_spec(record["prompt"], [])
    spec["limits"] = record["limits"]
    titles = {}
    for source in record["research"]["sources"]:
        titles[source["url"]] = source["title"]
    first = first_results(record, titles)
    replies = list(record["raw_replies"])
    state = {"reply": 0, "search": 0}

    def ask(messages, schema, max_tokens=2500):
        reply = replies[state["reply"]]
        state["reply"] += 1
        return reply

    def search(query, run):
        outcome = first[state["search"] % len(first)]
        state["search"] += 1
        return outcome

    def score_search(name, size_b):
        found = []
        for url, text in record["score_pages"].items():
            found.append({"title": titles.get(url, ""), "url": url, "text": text})
        return found

    bench.SCORE_PAGES_PER_BENCHMARK = 50
    bench.ask_llm = ask
    bench.search_exa = search
    bench.search_exa_scores = score_search
    run = bench.Run(spec["id"], "exa")
    bench.pipeline(run, spec)
    run.close("done")
    out = dict(record)
    out["research"] = run.data
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--only", default=None, help="comma separated indexes")
    args = parser.parse_args()
    os.environ["WSP_STUDIO_OUT"] = tempfile.mkdtemp(prefix="bench_research_replay_")
    os.environ["WSP_STUDIO_BACKEND"] = "stub"
    from studio.server import settings
    settings.load_env()
    os.environ["WSP_STUDIO_LLM"] = "none"
    folder = os.path.join(ROOT, "outputs", "bench_research_eval", args.tag)
    names = sorted(name for name in os.listdir(folder) if name.endswith(".json"))
    if args.only:
        names = [part + ".json" for part in args.only.split(",")]
    out_folder = os.path.join(ROOT, "outputs", "bench_research_eval", args.out)
    os.makedirs(out_folder, exist_ok=True)
    for name in names:
        with open(os.path.join(folder, name), encoding="utf-8") as handle:
            record = json.load(handle)
        out = replay_one(record)
        with open(os.path.join(out_folder, name), "w", encoding="utf-8") as handle:
            json.dump(out, handle, indent=2)
        sourced = 0
        for proposal in out["research"]["proposals"]:
            for entry in proposal["reference_scores"]:
                if entry.get("sourced"):
                    sourced += 1
        print(name, "sourced", sourced, out["research"]["score_checks"])


if __name__ == "__main__":
    main()
