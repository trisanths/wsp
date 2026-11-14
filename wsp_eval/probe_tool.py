"""probe(component, layers) -> effect dict.

Ablate a component at the given layers and measure the score drop:
  * deletion mask: un-delete the mask entries in those layers (keep the rest)
  * task vector:   zero the task vector in those layers (keep the rest)

drop[task] = score(full component) - score(component without those layers).
A positive drop means those layers carry part of the component's effect.
Probing all of a component's layers compares it against the base model.

Every probe runs two eval() calls (full and ablated) on the search split,
charged to the "interp" budget namespace by default. Generations are cached,
so the "full" call is cheap after the first probe of a component.
"""
from wsp_eval import catalog, components, harness, tasks as tasks_module


def default_tasks(record):
    task = tasks_module.task_for_training_set(record.get("trained_on", ""))
    if task is None:
        raise ValueError("cannot infer a task from trained_on=" + repr(record.get("trained_on")) +
                         "; pass tasks= explicitly")
    return [task]


def probe(component, layers, tasks=None, split="search", namespace="interp", full_scores=None,
          catalog_path=None):
    if isinstance(component, str):
        record = catalog.get(component, catalog_path)
    else:
        record = component
    if tasks is None:
        tasks = default_tasks(record)

    all_layers = components.component_layers(record, catalog_path)
    ablate = set(layers)
    keep = []
    for layer in all_layers:
        if layer not in ablate:
            keep.append(layer)

    # Both models are built by wsp's compile(): the full component, and the
    # component restricted to the layers we keep. Keeping no layers = base.
    from wsp import Program
    full_model = components.build_model(components.program_for(record), catalog_path)
    if len(keep) == 0:
        ablated_model = components.build_model(Program(base=record["base"]), catalog_path)
    else:
        ablated_model = components.build_model(components.program_for(record, keep), catalog_path)

    note = "probe " + record["id"] + " layers " + str(sorted(ablate))
    if full_scores is None:
        full_result = harness.evaluate(full_model, split, tasks, namespace=namespace, note=note + " (full)")
        full_scores = {}
        for task in tasks:
            full_scores[task] = full_result["tasks"][task]["score"]
    ablated_result = harness.evaluate(ablated_model, split, tasks, namespace=namespace, note=note + " (ablated)")

    ablated_scores = {}
    drop = {}
    for task in tasks:
        ablated_scores[task] = ablated_result["tasks"][task]["score"]
        drop[task] = full_scores[task] - ablated_scores[task]

    return {
        "component": record["id"],
        "layers": sorted(ablate),
        "split": split,
        "label": harness.SEARCH_LABEL if split == "search" else harness.HELDOUT_LABEL,
        "full": full_scores,
        "ablated": ablated_scores,
        "drop": drop,
        "n": ablated_result["tasks"][tasks[0]]["n"],
    }
