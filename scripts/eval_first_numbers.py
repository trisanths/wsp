"""Hour-8 gate: score the base, each parent mask, and compiled programs.

Search split only. Writes work/reference_search.json (base and parent scores
used to normalize the combined score) and appends every run to
experiments.jsonl. Numbers printed here are SEARCH-SPLIT numbers, not results.

Example (biology + chemistry parents, then their union):
    python scripts/eval_first_numbers.py \
        --parents mask_mmlupro_biology_llama1b mask_mmlupro_chemistry_llama1b \
        --programs "mask_mmlupro_biology_llama1b | mask_mmlupro_chemistry_llama1b"

Tasks default to each parent's own skill (from its catalog `trained_on`).
A program is a mask expression over catalog ids using | & - and parentheses,
or a path to a program JSON file (Program.to_dict()).
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from wsp import Program, use  # noqa: E402
from wsp_eval import catalog, components, harness, tasks as tasks_module  # noqa: E402


def parse_program(text, base):
    if os.path.exists(text):
        with open(text) as f:
            return Program.from_dict(json.load(f))
    # Evaluate a mask expression like "a | (b & c)" where names are catalog ids.
    names = {}
    for record in catalog.load():
        names[record["id"]] = use(record["id"])
    expression = eval(text, {"__builtins__": {}}, names)  # only catalog ids and | & - are in scope
    return Program(base=base, masks=expression)


def print_row(label, result, tasks):
    cells = []
    for task in tasks:
        cells.append("%s=%.3f" % (task, result["tasks"][task]["score"]))
    combined = ""
    if result["combined"] is not None and result["combined"]["score"] is not None:
        combined = "  combined=%.3f" % result["combined"]["score"]
    print("%-60s %s%s   [n=%d, %s]" % (label, "  ".join(cells), combined,
                                         result["tasks"][tasks[0]]["n"], result["label"]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parents", nargs="+", required=True, help="catalog ids of single-skill masks")
    parser.add_argument("--tasks", nargs="+", default=None)
    parser.add_argument("--programs", nargs="*", default=[])
    parser.add_argument("--namespace", default="reference")
    args = parser.parse_args()

    records = []
    for component_id in args.parents:
        records.append(catalog.get(component_id))
    base = records[0]["base"]

    tasks = args.tasks
    if tasks is None:
        tasks = []
        for record in records:
            task = tasks_module.task_for_training_set(record["trained_on"])
            if task is not None and task not in tasks:
                tasks.append(task)

    print("Tasks:", tasks)
    print("SEARCH SPLIT numbers below are for search only. They are not results.\n")

    base_result = harness.evaluate(base, "search", tasks, namespace=args.namespace, note="base")
    print_row("base " + base, base_result, tasks)
    base_scores = {}
    for task in tasks:
        base_scores[task] = base_result["tasks"][task]["score"]

    parent_scores = {}
    for record in records:
        model_dir = components.build_model(components.program_for(record))
        result = harness.evaluate(model_dir, "search", tasks, namespace=args.namespace, note="parent " + record["id"])
        print_row("parent " + record["id"], result, tasks)
        parent_scores[record["id"]] = {}
        for task in tasks:
            parent_scores[record["id"]][task] = result["tasks"][task]["score"]

    harness.save_reference("search", base_scores, parent_scores)
    print("\nSaved reference (base + parents) for the combined score.\n")

    for text in args.programs:
        program = parse_program(text, base)
        model_dir = components.build_model(program)
        result = harness.evaluate(model_dir, "search", tasks, namespace=args.namespace, note="program " + repr(program.masks))
        print_row("program " + text, result, tasks)
        if result["combined"] is not None:
            for task, entry in result["combined"]["per_skill"].items():
                print("    %s: normalized %.3f (raw %.3f; base %.3f, best single %.3f from %s)" % (
                    task, entry["normalized"], entry["raw"], entry["base"], entry["best_single"], entry["best_parent"]))
            if len(result["combined"]["skipped"]) > 0:
                print("    not normalized (best parent does not beat base):", result["combined"]["skipped"])


if __name__ == "__main__":
    main()
