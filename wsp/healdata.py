"""Text for stitches, routers, and heals that never comes from a scored benchmark.

Rule (owner, 2026-11-14): nothing that trains (a stitch, a router fit, a
gate, a heal, a distillation) may use data from a benchmark we report,
in any split. Training on GSM8K train and then reporting GSM8K test only
shows benchmark tuning. Use the pools here. Practice sets for model
selection are a separate matter and are not built here.

Every pool is plain domain text from a dataset that is not a reported
benchmark. `check_sources` refuses a banned dataset id, so a round
script can assert its own sources.

    from wsp import healdata
    texts = healdata.pool("math", 2000, tokenizer=tokenizer, seed=0)
"""

import random

# Dataset ids (lower case) of benchmarks we report or screen with. No split of these may train anything.
BANNED = [
    "openai/gsm8k", "gsm8k", "huggingfaceh4/math-500", "eleutherai/hendrycks_math", "hendrycks/competition_math",
    "google-research-datasets/mbpp", "mbpp", "openai/openai_humaneval", "openai_humaneval",
    "openlifescienceai/medmcqa", "medmcqa", "bigbio/med_qa", "gbaker/medqa-usmle-4-options",
    "cais/mmlu", "mmlu", "allenai/ai2_arc", "lukaemon/bbh", "maveriq/bigbenchhard", "juletxara/mgsm",
    "coherelabs/global-mmlu", "cohereforai/global-mmlu", "nguha/legalbench", "casehold/casehold",
    "fastino/fast-decisions", "meta-math/metamathqa", "tiger-lab/mathinstruct",
]

# name: (dataset id, config or None, split, kind). MetaMathQA and MathInstruct are banned above
# because they are built from GSM8K and MATH train items.
SOURCES = {
    "math": [("microsoft/orca-math-word-problems-200k", None, "train", "qa")],
    "code": [("ise-uiuc/Magicoder-OSS-Instruct-75K", None, "train", "magicoder"),
             ("sahil2801/CodeAlpaca-20k", None, "train", "alpaca")],
    "medicine": [("medalpaca/medical_meadow_wikidoc", None, "train", "alpaca"),
                 ("medalpaca/medical_meadow_medical_flashcards", None, "train", "alpaca")],
    "general": [("HuggingFaceH4/ultrachat_200k", None, "train_sft", "ultrachat"),
                ("yahma/alpaca-cleaned", None, "train", "alpaca")],
    "prose": [("Salesforce/wikitext", "wikitext-103-raw-v1", "train", "text")],
}


def is_banned(dataset_id):
    """True when a dataset id is a reported benchmark (any split)."""
    name = str(dataset_id).lower().strip()
    for banned in BANNED:
        if name == banned or name.endswith("/" + banned):
            return True
    return False


def check_sources(dataset_ids):
    """Raise ValueError when any id is a reported benchmark. Return the ids."""
    bad = []
    for dataset_id in dataset_ids:
        if is_banned(dataset_id):
            bad.append(dataset_id)
    if len(bad) > 0:
        raise ValueError("training data may not come from a reported benchmark: " + ", ".join(bad))
    return list(dataset_ids)


def row_pair(row, kind):
    """Return (user, assistant) for one dataset row, or None to skip it. assistant is None for plain text."""
    if kind == "qa":
        return row["question"], row["answer"]
    if kind == "magicoder":
        if row.get("lang") not in (None, "python"):
            return None
        return row["problem"], row["solution"]
    if kind == "alpaca":
        user = row["instruction"]
        if row.get("input"):
            user = user + "\n\n" + row["input"]
        if not row.get("output"):
            return None
        return user, row["output"]
    if kind == "ultrachat":
        messages = row["messages"]
        if len(messages) < 2:
            return None
        return messages[0]["content"], messages[1]["content"]
    if kind == "text":
        text = row["text"].strip()
        if len(text) < 200 or text.startswith("="):
            return None
        return text, None
    raise ValueError("unknown row kind: " + str(kind))


def render(tokenizer, user, assistant):
    """Chat-format one pair with the tokenizer's template. Plain text passes through."""
    if assistant is None:
        return user
    if tokenizer is None or getattr(tokenizer, "chat_template", None) is None:
        return user + "\n\n" + assistant
    messages = [{"role": "user", "content": user}, {"role": "assistant", "content": assistant}]
    return tokenizer.apply_chat_template(messages, tokenize=False)


def pairs(name, count, seed=0, skip=0):
    """Return count (user, assistant) pairs of one pool, after skipping skip pairs. Seeded order."""
    from datasets import load_dataset

    if name not in SOURCES:
        raise ValueError("unknown pool: " + str(name) + "; pools: " + ", ".join(sorted(SOURCES)))
    sources = SOURCES[name]
    ids = []
    for source in sources:
        ids.append(source[0])
    check_sources(ids)
    want = count + skip
    per_source = want // len(sources) + 1
    out = []
    for dataset_id, config, split, kind in sources:
        if config is None:
            rows = load_dataset(dataset_id, split=split)
        else:
            rows = load_dataset(dataset_id, config, split=split)
        order = list(range(len(rows)))
        random.Random(seed).shuffle(order)
        taken = 0
        for index in order:
            if taken >= per_source:
                break
            pair = row_pair(rows[index], kind)
            if pair is None:
                continue
            out.append(pair)
            taken += 1
    random.Random(seed + 1).shuffle(out)
    return out[skip:skip + count]


def pool(name, count, tokenizer=None, seed=0, skip=0):
    """Return count training texts of one pool, chat-formatted when a tokenizer with a template is given.

    skip lets two uses (for example a stitch and a heal) take disjoint
    parts of the same pool with the same seed.
    """
    texts = []
    for user, assistant in pairs(name, count, seed=seed, skip=skip):
        texts.append(render(tokenizer, user, assistant))
    return texts


def mix(names, per_pool, tokenizer=None, seed=0, skip=0):
    """Return a shuffled mix of per_pool texts from each named pool."""
    texts = []
    for name in names:
        for text in pool(name, per_pool, tokenizer=tokenizer, seed=seed, skip=skip):
            texts.append(text)
    random.Random(seed + 2).shuffle(texts)
    return texts
