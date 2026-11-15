"""Medical multiple-choice items for the clinic demo: reasoning prompts, train items, test index, decontamination.

The same reasoning prompt serves the train items (self-training, wsp.rft task "mcq") and the chain-of-thought
score, so a model is trained and scored on one prompt format. Train items come from the TRAIN splits of MedMCQA,
MedQA-USMLE and PubMedQA. Every test set the demo reports is put in one 13-gram index, and a train item whose
question and choices share a 13-gram with it is dropped (wsp.rft.decontaminate).
"""

import random

from wsp import rft, skills_eval

LETTERS = skills_eval.LETTERS


def reason_user(question, choices):
    """The user message for a chain-of-thought answer: question, lettered choices, a short-reasoning instruction."""
    lines = ["Question: " + question.strip()]
    for index, choice in enumerate(choices):
        lines.append(LETTERS[index] + ". " + str(choice).strip())
    letters = ", ".join(LETTERS[: len(choices) - 1]) + ", or " + LETTERS[len(choices) - 1]
    return ("\n".join(lines) + "\n\nThink it through in at most four short sentences, then finish with a line of "
            f"the form 'Answer: X' where X is {letters}.")


def mcq_item(item_id, source, question, choices, gold_index):
    return {"id": item_id, "source": source, "kind": "mcq", "user": reason_user(question, choices),
            "gold": LETTERS[gold_index], "question": question, "choices": list(choices)}


def medmcqa_train_item(row, index):
    """A MedMCQA train row (opa to opd, cop is the 0-based gold). None when the row has no valid answer."""
    choices = [row["opa"], row["opb"], row["opc"], row["opd"]]
    cop = row.get("cop")
    if cop is None or int(cop) not in (0, 1, 2, 3) or not str(row["question"]).strip():
        return None
    if any(not str(c).strip() for c in choices):
        return None
    return mcq_item(f"medmcqa_train:{index}", "medmcqa_train", row["question"], choices, int(cop))


def medqa_train_item(row, index):
    options = row["options"]
    choices = [options[letter] for letter in LETTERS] if isinstance(options, dict) else list(options)
    return mcq_item(f"medqa_train:{index}", "medqa_train", row["question"], choices, LETTERS.index(row["answer_idx"]))


def pubmedqa_train_item(row, source):
    item = skills_eval.pubmedqa_row_item(row, prefix=source)
    if item is None:
        return None
    return mcq_item(item["id"], source, item["question"], item["choices"], item["gold"])


def test_texts(sets):
    """Strings of every test item for the 13-gram index. sets is {name: items}; each item has a prompt that holds
    the question and the lettered choices."""
    texts = []
    for items in sets.values():
        for item in items:
            texts.append(item["prompt"])
    return texts


def test_index(sets):
    return rft.ngram_index(test_texts(sets))


def sample_items(items, count, seed=0):
    """A seeded random subset of at most count items, kept in a stable order."""
    if count <= 0 or len(items) <= count:
        return list(items)
    rng = random.Random(seed)
    picked = sorted(rng.sample(range(len(items)), count))
    return [items[i] for i in picked]


def build_train(sources, index, caps, seed=0):
    """(items, stats). sources is {name: [items]}; each source is decontaminated against index, then capped.

    stats[name] has candidates, dropped_overlap, kept."""
    out = []
    stats = {}
    for name, items in sources.items():
        clean, dropped = rft.decontaminate(items, index)
        kept = sample_items(clean, caps.get(name, 0), seed)
        stats[name] = {"candidates": len(items), "dropped_overlap": dropped, "kept": len(kept)}
        out.extend(kept)
    return out, stats
