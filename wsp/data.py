"""Task data and the fixed splits.

GSM8K splits (seeded, never overlap):
- search: 200 items carved from GSM8K train, used during search.
- heal: the rest of GSM8K train, used to train graft glue.
- held-out: GSM8K test, run once at the end. Not used here.
"""

import random

SEARCH_SIZE = 200


def format_gsm8k(item):
    return "Question: " + item["question"] + "\nAnswer: " + item["answer"]


def gsm8k_splits(seed=0):
    """Return (search_items, heal_items) from GSM8K train."""
    from datasets import load_dataset

    train = load_dataset("openai/gsm8k", "main", split="train")
    order = list(range(len(train)))
    random.Random(seed).shuffle(order)
    search = []
    heal_items = []
    for position, index in enumerate(order):
        if position < SEARCH_SIZE:
            search.append(train[index])
        else:
            heal_items.append(train[index])
    return search, heal_items


def task_texts(name, seed=0):
    """Return training texts for a heal data name."""
    if name == "gsm8k_train":
        search, heal_items = gsm8k_splits(seed)
        texts = []
        for item in heal_items:
            texts.append(format_gsm8k(item))
        return texts
    raise ValueError(f"unknown heal data: {name}")
