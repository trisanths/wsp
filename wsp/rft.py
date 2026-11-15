"""Verified self-training (rejection-sampling fine-tune, expert iteration, STaR) for any model and task.

One round: sample k answers per train prompt from the current model, score each with a reward function
that checks the answer (unit tests for code, an exact answer for math, a letter for multiple choice), keep
the answers with reward 1, fine-tune on them, and repeat from the new model. This module holds the parts
that need no GPU: task definitions, test-program assembly for code, the decontamination check, the
selection rule, the held-out split, and the SFT row encoding. modal_rft.py runs the GPU stages.

A task is a name, a user-text builder, and a batch reward function:

    task = get_task("code")
    users = [task.user(item) for item in items]
    rewards = task.rewards(items, outputs)        # one float per output, 1.0 means verified

To use the recipe on another task, build items (dicts with "id", "source", "kind", "user", plus the fields
the kind needs) and pick a task. Kinds: "asserts", "check", "stdin" (code), "math" (field "gold"), "mcq"
(field "gold", a letter). Register a new kind with register_task.

Rule of this repo (owner, 2026-11-15): training may use benchmark train splits. Every reported score comes
from test items nothing trained on, ranked on, or selected on. decontaminate() drops any item whose
normalized text shares a 13-gram with a test item, and returns how many it dropped.
"""

import hashlib
import json
import random
import re

from wsp import skills_eval

NGRAM = 13
MAX_STDIN_CASES = 10
STDIN_TIMEOUT = 20.0

# ---------------------------------------------------------------- decontamination


def tokens_of(text):
    """Lowercase word tokens of a text. Code and prose both reduce to runs of letters, digits, underscore."""
    return re.findall(r"[a-z0-9_]+", str(text).lower())


def ngrams_of(text, n=NGRAM):
    """The set of n-grams (tuples of tokens) of one text. A text shorter than n gives an empty set."""
    toks = tokens_of(text)
    return {tuple(toks[i:i + n]) for i in range(len(toks) - n + 1)}


def ngram_index(texts, n=NGRAM):
    """One set holding the n-grams of every text. Hash-compact: each 13-gram becomes a 64-bit integer."""
    index = set()
    for text in texts:
        toks = tokens_of(text)
        for i in range(len(toks) - n + 1):
            if informative(toks[i:i + n]):
                index.add(hash_gram(toks[i:i + n]))
    return index


def informative(gram):
    """False for a 13-gram that cannot show copying: all numbers (0 0 0 ..., 1 2 3 ..., prime lists) or fewer than
    5 distinct tokens. Without this rule 1544 of 2641 LeetCode train items matched HumanEval test lists (2026-11-15)."""
    return len(set(gram)) >= 5 and any(not t.isdigit() for t in gram)


def hash_gram(gram):
    return int.from_bytes(hashlib.blake2b(" ".join(gram).encode(), digest_size=8).digest(), "big")


def shares_ngram(text, index, n=NGRAM):
    """True when the text has a 13-gram in the index."""
    toks = tokens_of(text)
    for i in range(len(toks) - n + 1):
        gram = toks[i:i + n]
        if informative(gram) and hash_gram(gram) in index:
            return True
    return False


def item_text(item):
    """All the text of an item a model would see or grade against."""
    parts = [item.get("user", "")]
    for key in ("tests", "test", "header"):
        value = item.get(key)
        if isinstance(value, list):
            parts.extend(str(v) for v in value)
        elif value:
            parts.append(str(value))
    for case in item.get("cases") or []:
        parts.extend(str(v) for v in case)
    return "\n".join(parts)


def decontaminate(items, index, n=NGRAM):
    """(clean items, dropped count): items whose text shares no 13-gram with the test index."""
    clean = []
    dropped = 0
    for item in items:
        if shares_ngram(item_text(item), index, n):
            dropped += 1
        else:
            clean.append(item)
    return clean, dropped


def test_index(humaneval, mbpp):
    """The 13-gram index of the reported code test items: HumanEval (prompt, tests, entry point) and the
    MBPP sanitized test (prompt text, tests, reference code) in the format of wsp.skills_eval items."""
    texts = []
    for item in humaneval:
        texts.append(item["prompt"] + "\n" + item["test"])
        if item.get("canonical_solution"):
            texts.append(item["prompt"] + item["canonical_solution"])
    for item in mbpp:
        texts.append(item["text"] + "\n" + "\n".join(item["test_list"]))
        if item.get("code"):
            texts.append(item["code"])
    return ngram_index(texts)


def drop_sample_overlap(samples, index):
    """(clean samples, dropped count) for samples (dicts with "text") whose output shares a 13-gram with a test item."""
    clean = []
    dropped = 0
    for sample in samples:
        if shares_ngram(sample["text"], index):
            dropped += 1
        else:
            clean.append(sample)
    return clean, dropped


# ---------------------------------------------------------------- code programs


def stdin_program(code, cases):
    """A program that runs `code` once per (input, expected output) case with stdin and stdout replaced,
    and raises when any output differs from the expected one. Outputs compare by whitespace-split tokens."""
    cases = [(str(a), str(b)) for a, b in cases[:MAX_STDIN_CASES]]
    return (
        "import sys, io\n"
        "_wsp_code = " + repr(code) + "\n"
        "_wsp_cases = " + repr(cases) + "\n"
        "_wsp_real_out = sys.stdout\n"
        "for _wsp_in, _wsp_want in _wsp_cases:\n"
        "    sys.stdin = io.TextIOWrapper(io.BytesIO(_wsp_in.encode()))\n"
        "    _wsp_buf = io.StringIO()\n"
        "    sys.stdout = _wsp_buf\n"
        "    try:\n"
        "        exec(compile(_wsp_code, 'solution', 'exec'), {'__name__': '__main__'})\n"
        "    except SystemExit:\n"
        "        pass\n"
        "    finally:\n"
        "        sys.stdout = _wsp_real_out\n"
        "    if _wsp_buf.getvalue().split() != _wsp_want.split():\n"
        "        raise AssertionError('wrong answer')\n"
    )


def code_program(item, output):
    """The test program for one model output on one code item."""
    kind = item["kind"]
    if kind == "asserts":
        return skills_eval.assemble_mbpp(output, item["tests"], item.get("imports") or [])
    if kind == "check":
        name = item["entry_point"].split(".")[-1]
        code = skills_eval.strip_main_block(skills_eval.extract_code(output, name))
        return item["header"] + "\n" + code + "\n\n" + item["test"] + "\n\ncheck(" + item["entry_point"] + ")\n"
    if kind == "stdin":
        code = skills_eval.extract_code(output)
        return stdin_program(code, item["cases"])
    raise ValueError(f"not a code kind: {kind!r}")


def code_rewards(items, outputs, timeout=10.0, workers=8):
    """1.0 for each output whose program passes all tests, else 0.0, in the sandbox of wsp.skills_eval."""
    programs = [code_program(item, out) for item, out in zip(items, outputs)]
    limit = max(timeout, STDIN_TIMEOUT) if any(i["kind"] == "stdin" for i in items) else timeout
    results = skills_eval.run_programs(programs, timeout=limit, workers=workers)
    return [1.0 if r["passed"] else 0.0 for r in results]


# ---------------------------------------------------------------- math and multiple choice


def math_rewards(items, outputs, **_):
    """1.0 when the last \\boxed{} answer matches the gold answer (wsp.skills_eval.answers_match)."""
    out = []
    for item, text in zip(items, outputs):
        pred = skills_eval.last_boxed(text)
        out.append(1.0 if pred is not None and skills_eval.answers_match(pred, str(item["gold"])) else 0.0)
    return out


def letter_of(text):
    """The answer letter A to D of a reply: the last 'answer is X' / 'Answer: X' match, else a lone first letter."""
    found = re.findall(r"answer\s*(?:is|:)?\s*\(?\**([A-D])\b", text, flags=re.IGNORECASE)
    if found:
        return found[-1].upper()
    m = re.match(r"\s*\(?([A-D])\b", text)
    return m.group(1) if m else None


def mcq_rewards(items, outputs, **_):
    """1.0 when the reply's answer letter equals the gold letter."""
    return [1.0 if letter_of(text) == str(item["gold"]).upper() else 0.0 for item, text in zip(items, outputs)]


class Task:
    def __init__(self, name, user, rewards):
        self.name = name
        self.user = user
        self.rewards = rewards


def user_of_item(item):
    """The user message of an item (built when the item was made, so the train prompt equals the eval prompt)."""
    return item["user"]


TASKS = {}


def register_task(name, user, rewards):
    TASKS[name] = Task(name, user, rewards)
    return TASKS[name]


def get_task(name):
    if name not in TASKS:
        raise ValueError(f"unknown task {name!r}; the tasks are {sorted(TASKS)}")
    return TASKS[name]


KIND_TASK = {"asserts": "code", "check": "code", "stdin": "code", "math": "math", "mcq": "mcq"}


def rewards_for(items, outputs, workers=8):
    """Rewards for items of mixed kinds: each kind goes to its task, results come back in input order."""
    groups = {}
    for pos, item in enumerate(items):
        groups.setdefault(KIND_TASK[item["kind"]], []).append(pos)
    out = [0.0] * len(items)
    for name, positions in groups.items():
        part = get_task(name).rewards([items[p] for p in positions], [outputs[p] for p in positions], workers=workers)
        for p, value in zip(positions, part):
            out[p] = value
    return out


register_task("code", user_of_item, code_rewards)
register_task("math", user_of_item, math_rewards)
register_task("mcq", user_of_item, mcq_rewards)


# ---------------------------------------------------------------- item builders (same prompt as the eval)


def asserts_user(text, tests):
    """The MBPP eval prompt of wsp.skills_eval.user_text, for any text and assert list."""
    return (text.strip() + "\nYour code should pass these tests:\n\n" + "\n".join(tests[:3])
            + "\n\n" + skills_eval.CODE_INSTRUCTION)


def mbpp_train_item(row):
    """An item from a google-research-datasets/mbpp full row (train, validation, or prompt split)."""
    tests = list(row["test_list"])
    if not tests:
        return None
    return {"id": f"mbpp:{row['task_id']}", "source": "mbpp_train", "kind": "asserts",
            "user": asserts_user(row["text"], tests), "tests": tests,
            "imports": [row["test_setup_code"]] if row.get("test_setup_code") else [],
            "reference": row.get("code") or ""}


def ace_item(row, min_tests=3):
    """An item from a TIGER-Lab/AceCode-87K row. Needs a test list and one reference completion with pass rate 1."""
    tests = row["test_cases"]
    if isinstance(tests, str):
        tests = json.loads(tests) if tests.startswith("[") else __import__("ast").literal_eval(tests)
    tests = [t for t in tests if isinstance(t, str) and t.strip().startswith("assert")]
    if len(tests) < min_tests:
        return None
    inferences = row["inferences"]
    if isinstance(inferences, str):
        inferences = __import__("ast").literal_eval(inferences)
    rates = [float(i.get("pass_rate", 0)) for i in inferences]
    if not rates or max(rates) < 1.0:
        return None
    best = [i for i in inferences if float(i.get("pass_rate", 0)) >= 1.0][0]
    return {"id": f"ace:{row['id']}", "source": "acecode_" + str(row.get("source", "")), "kind": "asserts",
            "user": asserts_user(row["question"], tests), "tests": tests[:30], "imports": [],
            "reference": best["completion"], "mean_pass_rate": sum(rates) / len(rates)}


def leetcode_item(row):
    """An item from a newfacade/LeetCodeDataset row (the train split)."""
    prompt = row["prompt"]
    cut = prompt.find("class Solution")
    header = prompt[:cut] if cut >= 0 else prompt
    user = ("Complete the following Python function. " + skills_eval.CODE_INSTRUCTION + "\n\n"
            + row["problem_description"].strip() + "\n\n```python\n" + row["starter_code"].rstrip() + "\n```")
    return {"id": f"leetcode:{row['task_id']}", "source": "leetcode_train", "kind": "check", "user": user,
            "header": header, "test": row["test"], "entry_point": row["entry_point"],
            "reference": row["completion"]}


def apps_item(row, max_chars=3500):
    """An item from a codeparrot/apps row with stdin and stdout tests (not the call-based ones)."""
    io_raw = row["input_output"]
    if not io_raw:
        return None
    try:
        io_data = json.loads(io_raw) if isinstance(io_raw, str) else io_raw
    except ValueError:
        return None
    if io_data.get("fn_name") or not io_data.get("inputs"):
        return None
    cases = [[a, b] for a, b in zip(io_data["inputs"], io_data["outputs"]) if isinstance(a, str) and isinstance(b, str)]
    if not cases or len(row["question"]) > max_chars:
        return None
    user = ("Solve this problem with a Python program that reads from standard input and writes to standard "
            "output. " + skills_eval.CODE_INSTRUCTION + "\n\n" + row["question"].strip())
    refs = []
    if row.get("solutions"):
        try:
            sols = json.loads(row["solutions"]) if isinstance(row["solutions"], str) else row["solutions"]
        except ValueError:
            sols = []
        refs = [s for s in sols if isinstance(s, str)]
    return {"id": f"apps:{row['problem_id']}", "source": "apps_" + str(row.get("difficulty")), "kind": "stdin",
            "user": user, "cases": cases[:MAX_STDIN_CASES], "reference": refs[0] if refs else ""}


# ---------------------------------------------------------------- splits and selection


def is_heldout(item_id, fraction=0.05, salt="rft-v1"):
    """True for about `fraction` of ids, by a hash of the id, so the split never moves."""
    digest = hashlib.sha1((salt + str(item_id)).encode()).hexdigest()
    return int(digest[:8], 16) / 0xFFFFFFFF < fraction


def split_items(items, fraction=0.05, salt="rft-v1"):
    """(train items, held-out items)."""
    train, held = [], []
    for item in items:
        (held if is_heldout(item["id"], fraction, salt) else train).append(item)
    return train, held


def normalized(text):
    return " ".join(str(text).split())


def select_samples(samples, max_keep=2, seed=0):
    """Up to max_keep distinct passing samples from one prompt's list of {"text", "reward"}.

    Distinct means a different whitespace-normalized text. Among the passing ones the choice is random with
    a fixed seed so repeated runs agree; very long answers (over 1.5 times the median passing length) are
    skipped when a shorter passing one exists, so the data does not teach rambling.
    """
    passing = [s for s in samples if s["reward"] >= 1.0]
    seen = set()
    distinct = []
    for s in passing:
        key = normalized(s["text"])
        if key not in seen:
            seen.add(key)
            distinct.append(s)
    if len(distinct) > max_keep:
        lengths = sorted(len(s["text"]) for s in distinct)
        cap = 1.5 * lengths[len(lengths) // 2]
        shorter = [s for s in distinct if len(s["text"]) <= cap]
        if len(shorter) >= max_keep:
            distinct = shorter
    rng = random.Random(seed)
    rng.shuffle(distinct)
    return distinct[:max_keep]


def build_sft(items, student_samples, teacher_samples=None, student_keep=2, teacher_keep=1, seed=0):
    """SFT rows {"id", "user", "assistant", "origin"}.

    student_samples and teacher_samples map item id to a list of {"text", "reward"}. A prompt gets up to
    student_keep verified student answers and up to teacher_keep verified teacher answers; when the student
    solved nothing, the teacher may give student_keep. Returns (rows, stats).
    """
    teacher_samples = teacher_samples or {}
    rows = []
    stats = {"prompts": len(items), "solved_by_student": 0, "solved_by_teacher": 0, "solved_by_either": 0,
             "student_rows": 0, "teacher_rows": 0}
    for n, item in enumerate(items):
        mine = select_samples(student_samples.get(item["id"], []), student_keep, seed + n)
        take_teacher = teacher_keep if mine else student_keep
        theirs = select_samples(teacher_samples.get(item["id"], []), take_teacher, seed + n)
        seen = {normalized(r["text"]) for r in mine}
        theirs = [r for r in theirs if normalized(r["text"]) not in seen]
        stats["solved_by_student"] += 1 if mine else 0
        stats["solved_by_teacher"] += 1 if theirs else 0
        stats["solved_by_either"] += 1 if (mine or theirs) else 0
        for r in mine:
            rows.append({"id": item["id"], "user": item["user"], "assistant": r["text"], "origin": "student"})
        for r in theirs:
            rows.append({"id": item["id"], "user": item["user"], "assistant": r["text"], "origin": "teacher"})
        stats["student_rows"] += len(mine)
        stats["teacher_rows"] += len(theirs)
    return rows, stats


def cap_rows(rows, pass_rates, max_rows, seed=0):
    """At most max_rows SFT rows, hardest prompts first (lowest student pass rate), ties shuffled.

    pass_rates maps item id to the student's pass rate on it (0 when unknown, which ranks first: a prompt
    only the teacher solved is the most informative one). Rows of one prompt stay together.
    """
    if max_rows <= 0 or len(rows) <= max_rows:
        return list(rows)
    rng = random.Random(seed)
    ids = sorted({r["id"] for r in rows})
    rng.shuffle(ids)
    ids.sort(key=lambda i: pass_rates.get(i, 0.0))
    by_id = {}
    for r in rows:
        by_id.setdefault(r["id"], []).append(r)
    out = []
    for i in ids:
        if len(out) + len(by_id[i]) > max_rows:
            continue
        out.extend(by_id[i])
    return out


def group_advantages(rewards, eps=1e-6):
    """GRPO advantages for one prompt's group: (r - mean) / (std + eps). All zeros when every reward is equal."""
    n = len(rewards)
    mean = sum(rewards) / n
    var = sum((r - mean) ** 2 for r in rewards) / n
    if var == 0:
        return [0.0] * n
    std = var ** 0.5
    return [(r - mean) / (std + eps) for r in rewards]


def pg_rows(items, samples_by_id, max_prompts=0, seed=0):
    """Policy-gradient rows {"id", "user", "assistant", "weight"} from grouped samples.

    Only prompts whose group has mixed rewards give rows (the others have zero advantage). weight is the
    GRPO advantage; the trainer minimizes weight * negative log-likelihood summed over the answer tokens,
    so a weight below zero pushes that answer down. max_prompts keeps a random subset of the mixed prompts.
    """
    mixed = []
    for item in items:
        group = samples_by_id.get(item["id"])
        if not group:
            continue
        rewards = [s["reward"] for s in group]
        if min(rewards) < max(rewards):
            mixed.append((item, group, rewards))
    random.Random(seed).shuffle(mixed)
    if max_prompts > 0:
        mixed = mixed[:max_prompts]
    rows = []
    for item, group, rewards in mixed:
        for s, a in zip(group, group_advantages(rewards)):
            rows.append({"id": item["id"], "user": item["user"], "assistant": s["text"], "weight": a})
    return rows


def pass_rate_stats(rewards_by_id):
    """{"prompts", "mean_pass", "solved_any", "solved_all"} over lists of rewards per prompt id."""
    n = len(rewards_by_id)
    if n == 0:
        return {"prompts": 0, "mean_pass": 0.0, "solved_any": 0.0, "solved_all": 0.0}
    mean = sum(sum(v) / len(v) for v in rewards_by_id.values() if v) / n
    any_ = sum(1 for v in rewards_by_id.values() if any(x >= 1.0 for x in v)) / n
    all_ = sum(1 for v in rewards_by_id.values() if v and all(x >= 1.0 for x in v)) / n
    return {"prompts": n, "mean_pass": mean, "solved_any": any_, "solved_all": all_}


# ---------------------------------------------------------------- SFT encoding


def encode_sft(tokenizer, user, assistant, max_len=1536):
    """(input_ids, labels) for one chat example. Labels are -100 on the prompt, so loss covers the answer only.

    Returns None when the example is longer than max_len (it is skipped, not cut, so no answer is truncated).
    """
    prompt = skills_eval.render_prompt(tokenizer, user)
    p_ids = list(tokenizer(prompt, add_special_tokens=False)["input_ids"])
    end = tokenizer.eos_token or ""
    a_ids = list(tokenizer(assistant + end, add_special_tokens=False)["input_ids"])
    if len(p_ids) + len(a_ids) > max_len:
        return None
    return p_ids + a_ids, [-100] * len(p_ids) + a_ids


def make_batches(examples, max_tokens=8192, max_seqs=8, seed=0):
    """Group encoded examples into batches by token budget (padded size), in shuffled then length-sorted chunks."""
    rng = random.Random(seed)
    order = list(range(len(examples)))
    rng.shuffle(order)
    chunk = 256
    batches = []
    for start in range(0, len(order), chunk):
        part = sorted(order[start:start + chunk], key=lambda i: len(examples[i][0]))
        cur, longest = [], 0
        for i in part:
            n = len(examples[i][0])
            if cur and (max(longest, n) * (len(cur) + 1) > max_tokens or len(cur) >= max_seqs):
                batches.append(cur)
                cur, longest = [], 0
            cur.append(i)
            longest = max(longest, n)
        if cur:
            batches.append(cur)
    rng.shuffle(batches)
    return batches


def code_version():
    """A short tag for artifact paths. Bump RFT_FORMAT when data, selection, or training changes."""
    return RFT_FORMAT


RFT_FORMAT = "rft1"
