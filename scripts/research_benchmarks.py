"""Research the benchmarks for a request once, and save them for every arm of a race.

Runs Trisanth's studio research (studio/server/bench_research.py) through
wsp/swarm/benchmarks.py: Exa web search plus an LLM pick 3 to 6 benchmarks,
the ones the eval harness can run become targets and guards, and the task
setup is written to a JSON file. Race every arm on that file:

    python scripts/research_benchmarks.py --request "a small model that answers medical exam questions"
    python scripts/run_swarm.py --backend modal --catalog data/pilot_catalog.jsonl --modal-scorer harness \
        --benchmarks outputs/benchmarks/a-small-model-that-answers-medical-exam-questions.json

The LLM is the studio's choice (WSP_STUDIO_LLM). With no OpenAI or Anthropic
key in the environment this script defaults it to modal_vllm (Qwen 7B on the
deployed wsp-llm app). Benchmarks are scored with the real host base scores
only on a real backend, so WSP_STUDIO_BACKEND defaults to modal here.
"""

import argparse
import json
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from wsp.swarm import benchmarks  # noqa: E402
from wsp.swarm.config import load_dotenv  # noqa: E402


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Research a request's benchmarks with Exa and save the setup.")
    parser.add_argument("--request", required=True, help="what the model should be good at, in plain words")
    parser.add_argument("--heal-data", default="general",
                        help="heal data pool (wsp/data.py: general, math, code, medicine, mix, mix:a,b)")
    parser.add_argument("--size-b", type=float, default=None, help="size limit in billions (default: from the request)")
    parser.add_argument("--out", default=None, help="output file (default outputs/benchmarks/<request slug>.json)")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    load_dotenv()
    if not os.environ.get("WSP_STUDIO_LLM") and not os.environ.get("OPENAI_API_KEY") and \
            not os.environ.get("ANTHROPIC_API_KEY"):
        os.environ["WSP_STUDIO_LLM"] = "modal_vllm"
    os.environ.setdefault("WSP_STUDIO_BACKEND", "modal")
    record = benchmarks.research_setup(args.request, args.heal_data, args.size_b)
    out = args.out
    if out is None:
        out = os.path.join(REPO_ROOT, "outputs", "benchmarks", benchmarks.slug_for(args.request) + ".json")
    folder = os.path.dirname(out)
    if folder != "":
        os.makedirs(folder, exist_ok=True)
    with open(out, "w", encoding="utf-8") as handle:
        json.dump(record, handle, indent=2)
    print("request: " + record["request"])
    print("search: " + record["provider"] + ", llm: " + record["llm"])
    for proposal in record["proposals"]:
        runnable = proposal["runnable_id"] or "not runnable here (reference only)"
        print("  " + proposal["name"] + " -> " + runnable + " [" + proposal["kind"] + "]: " + proposal["why"])
    setup = record["task_setup"]
    print("task setup: targets " + ", ".join(sorted(setup["tasks"])) + "; guards " + ", ".join(setup["guards"]) +
          "; heal " + ", ".join(setup["heal_data"]))
    print("saved: " + out)
    return record


if __name__ == "__main__":
    main()
