"""Per-item multiple-choice scoring and complementarity analysis.

Use this to see which model is right on which items, so per-domain
compositions can pick a host and donors from data:

    items = mcq_items("mmlu", "test", seed=0, count=1000)   # round 6 dev items
    records = score_items(model, tokenizer, items)
    by_subject = accuracy_by_subject(records)
    plan = suggest_composition({"q14b": r1, "gemma9b": r2}, DOMAINS["medicine"])

Items are dicts: id, benchmark, subject, question, choices, prompt, gold.
The prompt format is the round 6 format with letters A to J:

    Question: ...
    A. ...
    B. ...
    Answer:

Records are dicts: id, subject, gold, pred, probs, correct. gold and pred
are option indexes (0 is A). probs is the softmax over the option letters
only. An item id is "<benchmark>/<split>/<row index>", so the same row has
the same id in every sample and records from different models line up.

Sampling: the row order is the dataset order shuffled with
random.Random(seed). skip drops the first positions of that order, then
the subject filter runs, then the first count items are kept. For MMLU
test, seed 0, count 1000 gives exactly round 6's dev items, and seed 0,
skip 1000, count 2000 gives round 7's held-out items. Because skip works on
positions before the filter, a subject-filtered held-out sample never
overlaps the dev items.
"""

import json
import math
import random

LETTERS = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]

# MMLU subject groups for per-domain plans.
DOMAINS = {
    "medicine": ["anatomy", "clinical_knowledge", "college_medicine", "medical_genetics",
                 "professional_medicine", "nutrition", "virology", "human_aging"],
    "biology": ["college_biology", "high_school_biology", "medical_genetics", "virology"],
    "math": ["abstract_algebra", "college_mathematics", "elementary_mathematics",
             "high_school_mathematics", "high_school_statistics"],
    "physics": ["astronomy", "college_physics", "conceptual_physics", "high_school_physics"],
    "chemistry": ["college_chemistry", "high_school_chemistry"],
    "computer_science": ["college_computer_science", "computer_security", "high_school_computer_science",
                         "machine_learning"],
    "law": ["international_law", "jurisprudence", "professional_law"],
    "economics": ["econometrics", "high_school_macroeconomics", "high_school_microeconomics"],
    "history": ["high_school_european_history", "high_school_us_history", "high_school_world_history",
                "prehistory"],
}


def mcq_prompt(question, choices):
    """The round 6 prompt (modal_round6.mmlu_prompt), with up to 10 options."""
    if len(choices) > len(LETTERS):
        raise ValueError(f"{len(choices)} options; at most {len(LETTERS)} are supported")
    lines = ["Question: " + question.strip()]
    for index, choice in enumerate(choices):
        lines.append(LETTERS[index] + ". " + str(choice).strip())
    return "\n".join(lines) + "\nAnswer:"


def load_rows(path, config, split):
    """Load one dataset split. Tests replace the datasets module, so this stays the only loader."""
    from datasets import load_dataset

    if config is None:
        return load_dataset(path, split=split)
    return load_dataset(path, config, split=split)


# Each row reader returns (question, choices, gold, subject), or None to skip the row.

def read_mmlu(row):
    if "train" in row and isinstance(row["train"], dict):
        row = row["train"]
    subject = row.get("subject", "")
    return row["question"], list(row["choices"]), int(row["answer"]), subject


def read_mmlu_pro(row):
    return row["question"], list(row["options"]), int(row["answer_index"]), row["category"]


def read_medmcqa(row):
    choices = [row["opa"], row["opb"], row["opc"], row["opd"]]
    gold = row["cop"]
    if gold is None:
        return None
    subject = row.get("subject_name")
    if not subject:
        subject = "unknown"
    return row["question"], choices, int(gold), subject


def read_medqa(row):
    options = row["options"]
    letters = sorted(options.keys())
    choices = []
    for letter in letters:
        choices.append(options[letter])
    gold = letters.index(row["answer_idx"])
    subject = row.get("meta_info")
    if not subject:
        subject = "medqa"
    return row["question"], choices, gold, subject


# name -> (dataset path, config, row reader)
BENCHMARKS = {
    "mmlu": ("cais/mmlu", "all", read_mmlu),
    "mmlu_pro": ("TIGER-Lab/MMLU-Pro", None, read_mmlu_pro),
    "medmcqa": ("openlifescienceai/medmcqa", None, read_medmcqa),
    "medqa": ("GBaker/MedQA-USMLE-4-options", None, read_medqa),
}


def mcq_items(benchmark, split_name, seed=0, count=None, subjects=None, skip=0):
    """A seeded sample of items from one benchmark split (see the module docstring)."""
    if benchmark not in BENCHMARKS:
        raise ValueError(f"unknown benchmark {benchmark!r}; known: {sorted(BENCHMARKS)}")
    path, config, reader = BENCHMARKS[benchmark]
    rows = load_rows(path, config, split_name)
    order = list(range(len(rows)))
    random.Random(seed).shuffle(order)
    wanted = None
    if isinstance(subjects, str):
        # One subject name, not a string of letters.
        subjects = [subjects]
    if subjects is not None:
        wanted = set(subjects)
    items = []
    for index in order[skip:]:
        if count is not None and len(items) >= count:
            break
        parsed = reader(rows[index])
        if parsed is None:
            continue
        question, choices, gold, subject = parsed
        if gold < 0 or gold >= len(choices):
            continue
        if wanted is not None and subject not in wanted:
            continue
        items.append({
            "id": f"{benchmark}/{split_name}/{index}",
            "benchmark": benchmark,
            "subject": subject,
            "question": question,
            "choices": choices,
            "prompt": mcq_prompt(question, choices),
            "gold": gold,
        })
    return items


def as_pairs(items):
    """Training pairs (prompt, " X") for tap_train, as modal_round6 builds them."""
    pairs = []
    for item in items:
        pairs.append((item["prompt"], " " + LETTERS[item["gold"]]))
    return pairs


def letter_ids(tokenizer, count=len(LETTERS)):
    """Token ids of " A", " B", ...: the last token of each, as modal_round6.letter_ids."""
    ids = []
    for letter in LETTERS[:count]:
        encoded = tokenizer(" " + letter, add_special_tokens=False).input_ids
        ids.append(encoded[-1])
    return ids


def as_item(item, index):
    """Accept round 6 (prompt, answer) tuples as well as item dicts."""
    if isinstance(item, dict):
        return item
    prompt, gold = item
    return {"id": str(index), "subject": "", "prompt": prompt, "gold": int(gold), "choices": None}


def option_count(item):
    if item.get("choices") is not None:
        return len(item["choices"])
    return 4


def score_items(model, tokenizer, items, donors=None, derange=False, batch_size=32):
    """Per-item records: the argmax over the log-probs of the option letters after the prompt.

    With donors, the model is a tapped host: each batch runs wsp.tap.prepare
    and then clear_tap_context, as modal_round6.score_mmlu does.
    """
    import torch

    from wsp.tap import clear_tap_context, prepare

    old_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    choice_ids = letter_ids(tokenizer)
    records = []
    try:
        for start in range(0, len(items), batch_size):
            batch = []
            for offset, item in enumerate(items[start:start + batch_size]):
                batch.append(as_item(item, start + offset))
            prompts = []
            for item in batch:
                prompts.append(item["prompt"])
            if donors is not None:
                encoded = prepare(model, tokenizer, donors, prompts, derange=derange)
            else:
                encoded = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
            try:
                with torch.no_grad():
                    out = model(input_ids=encoded["input_ids"], attention_mask=encoded["attention_mask"])
                    logits = out.logits[:, -1, :].float()
            finally:
                if donors is not None:
                    clear_tap_context(model)
            for row, item in enumerate(batch):
                ids = choice_ids[:option_count(item)]
                scores = logits[row, ids]
                probs = torch.softmax(scores, dim=0)
                pred = int(scores.argmax())
                prob_list = []
                for value in probs.tolist():
                    prob_list.append(round(value, 6))
                records.append({
                    "id": item["id"],
                    "subject": item["subject"],
                    "gold": item["gold"],
                    "pred": pred,
                    "probs": prob_list,
                    "correct": pred == item["gold"],
                })
    finally:
        tokenizer.padding_side = old_side
    return records


def accuracy(records):
    if len(records) == 0:
        return 0.0
    right = 0
    for record in records:
        if record["correct"]:
            right += 1
    return right / len(records)


def accuracy_by_subject(records):
    """subject -> {"accuracy", "correct", "count"}, subjects in sorted order."""
    counts = {}
    rights = {}
    for record in records:
        subject = record["subject"]
        if subject not in counts:
            counts[subject] = 0
            rights[subject] = 0
        counts[subject] += 1
        if record["correct"]:
            rights[subject] += 1
    out = {}
    for subject in sorted(counts):
        out[subject] = {"accuracy": rights[subject] / counts[subject], "correct": rights[subject],
                        "count": counts[subject]}
    return out


def by_id(records):
    table = {}
    for record in records:
        table[record["id"]] = record
    return table


def shared_ids(record_lists):
    """Ids present in every list, in the order of the first list."""
    tables = []
    for records in record_lists:
        tables.append(by_id(records))
    ids = []
    for record in record_lists[0]:
        present = True
        for table in tables[1:]:
            if record["id"] not in table:
                present = False
                break
        if present:
            ids.append(record["id"])
    return ids, tables


def oracle_records(record_lists):
    """One record per shared item; correct when any model gets it right."""
    if len(record_lists) == 0:
        return []
    ids, tables = shared_ids(record_lists)
    out = []
    for item_id in ids:
        first = tables[0][item_id]
        correct = False
        for table in tables:
            if table[item_id]["correct"]:
                correct = True
                break
        out.append({"id": item_id, "subject": first["subject"], "gold": first["gold"], "correct": correct})
    return out


def oracle_best_of(record_lists):
    """Accuracy when an item counts if any model gets it right (items shared by all lists)."""
    return accuracy(oracle_records(record_lists))


def best_model_per_subject(records_by_name):
    """subject -> {"model", "accuracy", "count", "scores": {name: accuracy}}.

    Ties go to the model listed first.
    """
    tables = {}
    subjects = set()
    for name in records_by_name:
        tables[name] = accuracy_by_subject(records_by_name[name])
        for subject in tables[name]:
            subjects.add(subject)
    out = {}
    for subject in sorted(subjects):
        scores = {}
        best = None
        for name in records_by_name:
            if subject not in tables[name]:
                continue
            value = tables[name][subject]["accuracy"]
            scores[name] = value
            if best is None or value > scores[best]:
                best = name
        out[subject] = {"model": best, "accuracy": scores[best], "count": tables[best][subject]["count"],
                        "scores": scores}
    return out


def complementarity(a, b):
    """Split the items both lists share: {"only_a", "only_b", "both", "neither"}, each a list of ids."""
    ids, tables = shared_ids([a, b])
    out = {"only_a": [], "only_b": [], "both": [], "neither": []}
    for item_id in ids:
        right_a = tables[0][item_id]["correct"]
        right_b = tables[1][item_id]["correct"]
        if right_a and right_b:
            out["both"].append(item_id)
        elif right_a:
            out["only_a"].append(item_id)
        elif right_b:
            out["only_b"].append(item_id)
        else:
            out["neither"].append(item_id)
    return out


def in_domain(records, domain_subjects):
    if not domain_subjects:
        return list(records)
    wanted = set(domain_subjects)
    out = []
    for record in records:
        if record["subject"] in wanted:
            out.append(record)
    return out


def suggest_composition(records_by_name, domain_subjects, host=None):
    """Rank donors for one domain.

    host is the model that runs the answer; if None, the most accurate model
    on the domain is the host (round 9: compose into the strongest model).
    Each donor row has its domain accuracy, "rescues" (domain items the
    donor gets and the host misses), "hurts" (the reverse), and the oracle
    accuracy of host plus that donor. Returns:

        host, host_accuracy, count
        donors        donor rows, most rescues first, then higher accuracy
        by_accuracy   donor names, most accurate first
        by_rescues    donor names, most rescues first
        greedy        [(name, oracle accuracy so far)]: each step adds the
                      donor that rescues the most items still missed
    Oracle numbers count over the host's domain items, and an item a donor
    did not score is a miss for it. Ties keep the order of records_by_name,
    except in greedy, where they keep the order of donors.
    """
    domain = {}
    for name in records_by_name:
        domain[name] = in_domain(records_by_name[name], domain_subjects)
    names = list(records_by_name)
    if len(names) == 0:
        raise ValueError("no models given")
    if host is None:
        for name in names:
            if host is None or accuracy(domain[name]) > accuracy(domain[host]):
                host = name
    if host not in domain:
        raise ValueError(f"host {host!r} has no records")

    # Oracle numbers count over the host's domain items. An item a donor
    # did not score counts as a miss for that donor, so a donor scored on
    # only part of the domain cannot look better than it is.
    host_count = len(domain[host])
    host_right = 0
    for record in domain[host]:
        if record["correct"]:
            host_right += 1

    rows = []
    for name in names:
        if name == host:
            continue
        split = complementarity(domain[host], domain[name])
        shared = len(split["only_a"]) + len(split["only_b"]) + len(split["both"]) + len(split["neither"])
        oracle = 0.0
        if host_count > 0:
            oracle = (host_right + len(split["only_b"])) / host_count
        rows.append({"name": name, "accuracy": accuracy(domain[name]), "rescues": len(split["only_b"]),
                     "hurts": len(split["only_a"]), "shared": shared, "oracle_with_host": oracle})

    by_accuracy = sorted(rows, key=lambda row: -row["accuracy"])
    by_rescues = sorted(rows, key=lambda row: -row["rescues"])
    ranked = sorted(rows, key=lambda row: (-row["rescues"], -row["accuracy"]))

    # Greedy: each step adds the donor that gets the most host-domain items
    # that no chosen model gets yet. Ties go to the earlier row in ranked.
    missed = set()
    for record in domain[host]:
        if not record["correct"]:
            missed.add(record["id"])
    solves = {}
    for row in ranked:
        right = set()
        for record in domain[row["name"]]:
            if record["correct"]:
                right.add(record["id"])
        solves[row["name"]] = right
    left = []
    for row in ranked:
        left.append(row["name"])
    greedy = []
    while len(left) > 0 and len(missed) > 0:
        best = None
        best_gain = 0
        for name in left:
            gain = len(missed & solves[name])
            if gain > best_gain:
                best = name
                best_gain = gain
        if best is None:
            break
        missed = missed - solves[best]
        left.remove(best)
        greedy.append((best, (host_count - len(missed)) / host_count))

    by_accuracy_names = []
    for row in by_accuracy:
        by_accuracy_names.append(row["name"])
    by_rescue_names = []
    for row in by_rescues:
        by_rescue_names.append(row["name"])
    return {"host": host, "host_accuracy": accuracy(domain[host]), "count": len(domain[host]),
            "donors": ranked, "by_accuracy": by_accuracy_names, "by_rescues": by_rescue_names,
            "greedy": greedy}


def save_records(records, path):
    """Write records as JSONL, one record per line."""
    with open(path, "w") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")


def load_records(path):
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def mean_log_prob_gold(records):
    """Mean log-probability of the gold option, a smoother signal than accuracy."""
    if len(records) == 0:
        return 0.0
    total = 0.0
    for record in records:
        total += math.log(max(record["probs"][record["gold"]], 1e-12))
    return total / len(records)
