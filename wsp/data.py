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
    if is_pool_data(name):
        return pool_texts(name, seed=seed)
    if name == "gsm8k_train":
        search, heal_items = gsm8k_splits(seed)
        texts = []
        for item in heal_items:
            texts.append(format_gsm8k(item))
        return texts
    raise ValueError(f"unknown heal data: {name}")


# Heal data names backed by wsp.healdata pools (owner rule, 2026-11-14: nothing
# that trains or fits may use a reported benchmark in any split).

DEFAULT_DATA = "general"
POOL_NAMES = ["general", "math", "code", "medicine", "prose"]
MIX_POOLS = ["general", "math", "code", "medicine"]
APPROVED_NAMES = POOL_NAMES + ["mix", "mix:<pool>,<pool>"]
# Texts a heal draws from a pool when the caller gives no count.
HEAL_TEXT_COUNT = 2000
# Dataset ids of the names in this module. wsp.pairs keeps its own table.
DATA_SOURCES = {"gsm8k_train": "openai/gsm8k"}


def pool_names_of(name):
    """Return the healdata pools a data name draws from, or None when it is not a pool name.

    "general" and the other pool names give one pool. "mix" gives the even
    mix of general, math, code, and medicine. "mix:math,code" gives the
    chosen pools. A bad mix raises ValueError.
    """
    if not isinstance(name, str):
        return None
    if name in POOL_NAMES:
        return [name]
    if name == "mix":
        return list(MIX_POOLS)
    if name.startswith("mix:"):
        chosen = [part for part in name[4:].split(",")]
        if len(chosen) == 0 or "" in chosen:
            raise ValueError(f"data {name!r} needs pool names after mix: such as mix:math,code; the pools are "
                             + ", ".join(POOL_NAMES))
        for part in chosen:
            if part not in POOL_NAMES:
                raise ValueError(f"data {name!r} names the unknown pool {part!r}; the pools are "
                                 + ", ".join(POOL_NAMES))
        if len(set(chosen)) != len(chosen):
            raise ValueError(f"data {name!r} names a pool twice")
        return chosen
    return None


def is_pool_data(name):
    """True when the name is a pool name or a valid mix. Raises ValueError for a bad mix."""
    return pool_names_of(name) is not None


def benchmark_source(name):
    """Return the banned benchmark dataset id behind a data name, or None."""
    from wsp import healdata
    from wsp.pairs import PAIR_SOURCES

    source = DATA_SOURCES.get(name)
    if source is None:
        source = PAIR_SOURCES.get(name)
    if source is not None and healdata.is_banned(source):
        return source
    return None


def check_data_name(name, allow_benchmark_data=False, where="data"):
    """Raise ValueError for a benchmark data name (unless allowed) or a bad mix. Return True when the name is a benchmark."""
    pool_names_of(name)
    source = benchmark_source(name)
    if source is None:
        return False
    if not allow_benchmark_data:
        raise ValueError(
            f"{where} {name!r} is a train set of the reported benchmark {source}. Rule: nothing that trains or "
            f"fits may use a reported benchmark in any split. Use an approved name: {', '.join(APPROVED_NAMES)}. "
            f"Old research scripts can pass allow_benchmark_data=True (--allow-benchmark-data).")
    return True


def program_data_names(program):
    """Return [(where, name)] for every data name a program holds."""
    found = []
    if program.heal_spec is not None:
        found.append(("heal data", program.heal_spec.data))
    if program.tap_train_spec is not None:
        found.append(("tap_train data", program.tap_train_spec.data))
    if program.carve_spec is not None and program.carve_spec.domain_data is not None:
        found.append(("carve domain", program.carve_spec.domain_data))
    return found


def program_uses_benchmark_data(program):
    for where, name in program_data_names(program):
        if benchmark_source(name) is not None:
            return True
    return False


def check_program_data(program, allow_benchmark_data=False):
    """Check every data name of a program. Return True when it uses benchmark data (only with the escape hatch)."""
    used = False
    for where, name in program_data_names(program):
        if check_data_name(name, allow_benchmark_data, where):
            used = True
    return used


def pool_split(name, first, second, tokenizer=None, seed=0):
    """Return (first_texts, second_texts) of a pool name, from disjoint ranges of one fetch.

    first texts come before second texts in the pool order. Use first for
    the align or calibration texts and second for the heal. Each pool of a
    mix gives an even share of both parts. The text order inside each part
    is shuffled with the seed.
    """
    import random

    from wsp import healdata

    pools = pool_names_of(name)
    if pools is None:
        raise ValueError(f"{name!r} is not a pool data name")
    total = first + second
    first_texts = []
    second_texts = []
    for position, pool in enumerate(pools):
        take_first = first // len(pools) + (1 if position < first % len(pools) else 0)
        take_second = second // len(pools) + (1 if position < second % len(pools) else 0)
        if take_first + take_second == 0:
            continue
        pairs = healdata.pairs(pool, take_first + take_second, seed=seed)
        for user, assistant in pairs[:take_first]:
            first_texts.append(healdata.render(tokenizer, user, assistant))
        for user, assistant in pairs[take_first:take_first + take_second]:
            second_texts.append(healdata.render(tokenizer, user, assistant))
    if len(pools) > 1:
        random.Random(seed + 2).shuffle(first_texts)
        random.Random(seed + 3).shuffle(second_texts)
    if total > 0 and len(first_texts) + len(second_texts) == 0:
        raise ValueError(f"the pool {name!r} gave no texts")
    return first_texts, second_texts


def pool_texts(name, count=HEAL_TEXT_COUNT, tokenizer=None, seed=0):
    """Return count training texts of a pool name."""
    return pool_split(name, 0, count, tokenizer, seed)[1]


def pool_pairs(name, count=HEAL_TEXT_COUNT, seed=0):
    """Return (prompt, completion) pairs of a pool name, for a program with taps."""
    from wsp import healdata

    pools = pool_names_of(name)
    out = []
    for position, pool in enumerate(pools):
        share = count // len(pools) + (1 if position < count % len(pools) else 0)
        for user, assistant in healdata.pairs(pool, share, seed=seed):
            if assistant is None:
                cut = len(user) // 2
                out.append((user[:cut], user[cut:]))
            else:
                out.append(("Question: " + user + "\nAnswer:", " " + assistant))
    return out
