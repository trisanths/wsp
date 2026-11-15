"""Training pairs for taps: lists of (prompt, completion).

A tap trains on prompt and completion pairs. The donor reads only the
prompt, and the loss covers only the completion (see wsp.tap.tap_train).

These three names are old research data. gsm8k_train and mmlu_aux are
benchmark train sets, so the compile refuses them unless
allow_benchmark_data is set. The approved data names are the healdata
pools (see wsp.data).

Names:
- gsm8k_train   the GSM8K heal items from wsp.data.gsm8k_splits. Prompt
                "Question: ...\\nAnswer:", completion " " + the full answer.
                The same pairs as modal_round6.py part B.
- mmlu_aux      8000 MMLU auxiliary_train pairs with 4 choices and a prompt
                of at most 240 host tokens. Prompt is the question, the
                lettered choices, and "Answer:"; completion " A" to " D".
                The same pairs as modal_round6.mmlu_data.
- trivia_train  8000 TriviaQA rc.nocontext train pairs. Prompt
                "Question: ...\\nAnswer:", completion " " + the answer value.
                The same pairs as modal_round7.trivia_data.

pairs_for(name, tokenizer) returns the pairs. With max_prompt_tokens, a
pair whose prompt is longer than that many host tokens (special tokens
included) is left out. register_pairs adds a name.
"""

import random

LETTERS = ["A", "B", "C", "D"]

# name -> function(tokenizer, seed, count, max_prompt_tokens) -> pairs
REGISTRY = {}

# name -> dataset id the pairs come from. wsp.data.check_data_name refuses a name whose id is a reported benchmark.
PAIR_SOURCES = {}

# The defaults that match the round 6 and round 7 runs.
DEFAULTS = {
    "gsm8k_train": {"count": None, "max_prompt_tokens": None},
    "mmlu_aux": {"count": 8000, "max_prompt_tokens": 240},
    "trivia_train": {"count": 8000, "max_prompt_tokens": None},
}


def register_pairs(name, function, count=None, max_prompt_tokens=None, source=None):
    """Add a pairs source. function(tokenizer, seed, count, max_prompt_tokens) returns a list of pairs.

    source is the dataset id the pairs come from. A source that is a
    reported benchmark (wsp.healdata.BANNED) makes the compile refuse the
    name unless allow_benchmark_data is set.
    """
    REGISTRY[name] = function
    if source is not None:
        PAIR_SOURCES[name] = source
    DEFAULTS[name] = {"count": count, "max_prompt_tokens": max_prompt_tokens}


def pair_names():
    return sorted(REGISTRY)


def pairs_for(name, tokenizer=None, seed=0, count="default", max_prompt_tokens="default"):
    """Return the (prompt, completion) pairs for a data name.

    count and max_prompt_tokens default to the values the rounds used.
    Pass None for no limit. A length limit needs the host tokenizer.
    """
    if name not in REGISTRY:
        raise ValueError(f"unknown tap data {name!r}; the names are {', '.join(pair_names())}")
    if count == "default":
        count = DEFAULTS[name]["count"]
    if max_prompt_tokens == "default":
        max_prompt_tokens = DEFAULTS[name]["max_prompt_tokens"]
    if max_prompt_tokens is not None and tokenizer is None:
        raise ValueError(f"{name} limits prompt length, so it needs the host tokenizer")
    return REGISTRY[name](tokenizer, seed, count, max_prompt_tokens)


def prompt_fits(tokenizer, prompt, max_prompt_tokens):
    """True when there is no limit or the prompt has at most max_prompt_tokens host tokens."""
    if max_prompt_tokens is None:
        return True
    return len(tokenizer(prompt).input_ids) <= max_prompt_tokens


def qa_prompt(question):
    return "Question: " + question + "\nAnswer:"


def gsm8k_pairs(tokenizer, seed, count, max_prompt_tokens):
    from wsp.data import gsm8k_splits

    search, heal_items = gsm8k_splits(seed=seed)
    pairs = []
    for item in heal_items:
        prompt = qa_prompt(item["question"])
        if not prompt_fits(tokenizer, prompt, max_prompt_tokens):
            continue
        pairs.append((prompt, " " + item["answer"]))
        if count is not None and len(pairs) >= count:
            break
    return pairs


def mmlu_prompt(question, choices):
    lines = ["Question: " + question.strip()]
    for index, choice in enumerate(choices):
        lines.append(LETTERS[index] + ". " + str(choice).strip())
    return "\n".join(lines) + "\nAnswer:"


def mmlu_aux_pairs(tokenizer, seed, count, max_prompt_tokens):
    from datasets import load_dataset

    aux = load_dataset("cais/mmlu", "auxiliary_train", split="train")
    order = list(range(len(aux)))
    random.Random(seed + 1).shuffle(order)
    pairs = []
    for index in order:
        row = aux[index]
        # Some versions of the dataset nest each row under "train".
        if "train" in row and isinstance(row["train"], dict):
            row = row["train"]
        if len(row["choices"]) != 4:
            continue
        prompt = mmlu_prompt(row["question"], row["choices"])
        if not prompt_fits(tokenizer, prompt, max_prompt_tokens):
            continue
        pairs.append((prompt, " " + LETTERS[int(row["answer"])]))
        if count is not None and len(pairs) >= count:
            break
    return pairs


def trivia_pairs(tokenizer, seed, count, max_prompt_tokens):
    from datasets import load_dataset

    train = load_dataset("mandarjoshi/trivia_qa", "rc.nocontext", split="train")
    order = list(range(len(train)))
    random.Random(seed).shuffle(order)
    pairs = []
    for index in order:
        row = train[index]
        prompt = qa_prompt(row["question"].strip())
        if not prompt_fits(tokenizer, prompt, max_prompt_tokens):
            continue
        pairs.append((prompt, " " + row["answer"]["value"].strip()))
        if count is not None and len(pairs) >= count:
            break
    return pairs


register_pairs("gsm8k_train", gsm8k_pairs, source="openai/gsm8k")
register_pairs("mmlu_aux", mmlu_aux_pairs, count=8000, max_prompt_tokens=240, source="cais/mmlu")
register_pairs("trivia_train", trivia_pairs, count=8000, source="mandarjoshi/trivia_qa")
