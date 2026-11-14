"""Deterministic, disjoint splits.

Default rule (MMLU-Pro, BBEH, MMLU): the test set is shuffled with a fixed
seed; the first SEARCH_SIZE items are the search split and the rest are the
held-out split.

A task can define its own rule with a `split_items(split, seed)` method.
GSM8K does: search = 200 items from train, heal = the rest of train,
held-out = the test set. The "heal" split is training data; eval() refuses it.
"""
import random

from wsp_eval import config

_items_cache = {}


def split_items(task, split, search_size=None):
    if split not in ("search", "heldout", "heal"):
        raise ValueError("split must be 'search', 'heldout' or 'heal', got " + repr(split))

    if hasattr(task, "split_items"):
        key = task.name + "|" + split
        if key not in _items_cache:
            _items_cache[key] = task.split_items(split, config.SEED)
        return _items_cache[key]

    if split == "heal":
        raise ValueError(task.name + " has no heal split; only tasks with a train set (gsm8k) do")
    if search_size is None:
        search_size = config.SEARCH_SIZE

    if task.name not in _items_cache:
        _items_cache[task.name] = task.load_items()
    items = _items_cache[task.name]

    # Sort by id first so the split does not depend on dataset row order.
    ordered = sorted(items, key=lambda item: item["id"])
    rng = random.Random(str(config.SEED) + ":" + task.name)
    rng.shuffle(ordered)

    if search_size >= len(ordered):
        raise ValueError(task.name + " has only " + str(len(ordered)) + " items; search size " +
                         str(search_size) + " would leave no held-out items")
    if split == "search":
        return ordered[:search_size]
    return ordered[search_size:]


def heal_items(task_name="gsm8k"):
    """Training items for the heal pass (disjoint from search and held-out)."""
    from wsp_eval import tasks as tasks_module
    return split_items(tasks_module.get_task(task_name), "heal")


def heal_texts(task_name="gsm8k"):
    """Plain question + worked-solution strings for next-token heal training."""
    texts = []
    for item in heal_items(task_name):
        texts.append(item["question"] + "\n" + item["solution"])
    return texts
