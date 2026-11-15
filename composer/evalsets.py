"""Held-out sets, model scoring, and the composer heal and probe text. Runs on Modal.

Benchmarks (the held-out sets) score a model once, in the final report.
Nothing that trains or fits reads them in any split: heal text and probe
text come only from wsp.healdata pools, in ranges that never overlap
(composer.skills.HEAL_START and PROBE_START).

Multiple choice: argmax of the log-probs of " A" to " D" after
"Answer:", the round 6 format. Generation: greedy decoding, fixed token
limits. Code runs in a subprocess with a 10 second limit, on the Modal
container only.
"""

import os
import random
import subprocess
import sys
import tempfile
import time

from composer import skills as skills_module
from composer.skills import LAW_SUBJECTS, extract_code, gsm8k_correct, last_boxed, math_correct

LETTERS = ["A", "B", "C", "D"]
MAX_NEW_TOKENS = {"gsm8k": 256, "math": 768, "humaneval": 512, "mbpp": 512}


def mcq_prompt(question, choices):
    lines = ["Question: " + question.strip()]
    for index, choice in enumerate(choices):
        lines.append(LETTERS[index] + ". " + str(choice).strip())
    return "\n".join(lines) + "\nAnswer:"


def load(*args, **kwargs):
    from datasets import load_dataset

    return load_dataset(*args, **kwargs)


def seeded_order(count, seed):
    order = list(range(count))
    random.Random(seed).shuffle(order)
    return order


# Multiple choice sets.

def mmlu_rows(split, subjects=None):
    rows = []
    if subjects is None:
        data = load("cais/mmlu", "all", split=split)
        for row in data:
            rows.append(row)
        return rows
    for subject in subjects:
        data = load("cais/mmlu", subject, split=split)
        for row in data:
            rows.append(row)
    return rows


def mmlu_items(rows):
    items = []
    for row in rows:
        items.append({"prompt": mcq_prompt(row["question"], row["choices"]), "gold": int(row["answer"])})
    return items


def mmlu_dev():
    """The round 6 dev items: the first 1000 of MMLU test in seed 0 order."""
    test = load("cais/mmlu", "all", split="test")
    items = []
    for index in seeded_order(len(test), 0)[:1000]:
        row = test[index]
        items.append({"prompt": mcq_prompt(row["question"], row["choices"]), "gold": int(row["answer"])})
    return items


def medmcqa_item(row):
    choices = [row["opa"], row["opb"], row["opc"], row["opd"]]
    return {"prompt": mcq_prompt(row["question"], choices), "gold": int(row["cop"])}


def medmcqa_r10():
    """The round 10 items: the first 1000 of MedMCQA validation in seed 7 order."""
    data = load("openlifescienceai/medmcqa", split="validation")
    items = []
    for index in seeded_order(len(data), 7)[:1000]:
        items.append(medmcqa_item(data[index]))
    return items


# Generation sets.

def gsm8k_test():
    data = load("openai/gsm8k", "main", split="test")
    items = []
    for row in data:
        items.append({"question": row["question"], "answer": row["answer"]})
    return items


def math_items(rows):
    items = []
    for row in rows:
        answer = row.get("answer")
        if answer is None:
            answer = last_boxed(row.get("solution", ""))
        if answer is None:
            continue
        items.append({"problem": row["problem"], "answer": answer})
    return items


def math500():
    data = load("HuggingFaceH4/MATH-500", split="test")
    rows = []
    for row in data:
        rows.append(row)
    return math_items(rows)


def mbpp_rows(split):
    data = load("google-research-datasets/mbpp", "full", split=split)
    rows = []
    for row in data:
        rows.append(row)
    return rows


def mbpp_item(row):
    return {"text": row["text"], "tests": list(row["test_list"]), "setup": row.get("test_setup_code") or "",
            "code": row["code"]}


def humaneval():
    data = load("openai/openai_humaneval", split="test")
    items = []
    for row in data:
        items.append({"prompt": row["prompt"], "test": row["test"], "entry_point": row["entry_point"]})
    return items


def load_set(name):
    """Return (kind, items) for a set name in composer.skills.SET_NOTES."""
    if name == "gsm8k_test":
        return "gsm8k", gsm8k_test()
    if name == "math500":
        return "math", math500()
    if name == "mbpp_test":
        items = []
        for row in mbpp_rows("test"):
            items.append(mbpp_item(row))
        return "mbpp", items
    if name == "humaneval":
        return "humaneval", humaneval()
    if name == "medmcqa_r10":
        return "mcq", medmcqa_r10()
    if name == "mmlu_dev":
        return "mcq", mmlu_dev()
    if name == "mmlu_law":
        return "mcq", mmlu_items(mmlu_rows("test", LAW_SUBJECTS))
    raise ValueError(f"unknown set {name!r}")


# Heal and probe text, from wsp.healdata pools only.

POOL_CACHE = {}


def pool_pairs(name):
    """The (user, assistant) pairs of one pool for the heal and probe ranges, from one fetch.

    One call fetches pairs up to the end of the probe range, so the heal slice and the probe slice
    come from the same list and never overlap. Pool order is seeded, so every call agrees.
    """
    from wsp import healdata

    healdata.check_sources(skills_module.heal_source_ids(name))
    if name not in POOL_CACHE:
        POOL_CACHE[name] = healdata.pairs(name, skills_module.PROBE_START + skills_module.PROBE_COUNT, seed=0)
    return POOL_CACHE[name]


def heal_pairs(name):
    """The heal pairs of one pool."""
    return pool_pairs(name)[skills_module.HEAL_START:skills_module.HEAL_START + skills_module.HEAL_COUNT]


def probe_pairs(name, count=None):
    """The probe pairs of one pool (never in the heal range)."""
    pairs = pool_pairs(name)[skills_module.PROBE_START:skills_module.PROBE_START + skills_module.PROBE_COUNT]
    if count is not None:
        pairs = pairs[:count]
    return pairs


def render_pairs(pairs, tokenizer):
    from wsp import healdata

    texts = []
    for user, assistant in pairs:
        texts.append(healdata.render(tokenizer, user, assistant))
    return texts


def heal_texts_for(name, tokenizer, seed=0):
    """Training texts for a heal data name: a healdata pool, or an even mix of the four pools.

    Every source is checked with healdata.check_sources. The data always comes from seed 0 so the heal range
    stays apart from the probe range; the program seed still sets the heal's batch order (wsp.heal).
    """
    pools = skills_module.heal_pools_for(name)
    skills_module.check_heal_sources(name)
    texts = []
    each = skills_module.HEAL_COUNT // len(pools)
    for pool in pools:
        texts += render_pairs(heal_pairs(pool)[:each], tokenizer)
    if len(pools) > 1:
        random.Random(0).shuffle(texts)
    return texts


def probe_texts_for(skill, tokenizer, count=None):
    """Held-out pool text for a skill's probe, rendered with the tokenizer's chat format."""
    return render_pairs(probe_pairs(skills_module.SKILLS[skill]["pool"], count), tokenizer)


def general_probe_texts(tokenizer, count=None):
    return render_pairs(probe_pairs("general", count), tokenizer)


# Scoring.

def prepare_tokenizer(tokenizer):
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token


def letter_ids(tokenizer):
    ids = []
    for letter in LETTERS:
        ids.append(tokenizer(" " + letter, add_special_tokens=False).input_ids[-1])
    return ids


def score_mcq(model, tokenizer, items, batch_size=16):
    import torch

    prepare_tokenizer(tokenizer)
    choice_ids = letter_ids(tokenizer)
    correct = 0
    for start in range(0, len(items), batch_size):
        batch = items[start:start + batch_size]
        prompts = []
        for item in batch:
            prompts.append(item["prompt"])
        encoded = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
        with torch.no_grad():
            logits = model(**encoded).logits[:, -1, :].float()
        for row, item in enumerate(batch):
            if int(logits[row, choice_ids].argmax()) == item["gold"]:
                correct += 1
    return correct / max(1, len(items))


def chat_text(tokenizer, content):
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template([{"role": "user", "content": content}], tokenize=False,
                                             add_generation_prompt=True), False
    return "Question: " + content + "\nAnswer:", True


def generate(model, tokenizer, texts, max_new_tokens, add_special, batch_size=16):
    import torch

    prepare_tokenizer(tokenizer)
    outputs = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]
        encoded = tokenizer(batch, return_tensors="pt", padding=True, add_special_tokens=add_special).to(model.device)
        with torch.no_grad():
            out = model.generate(**encoded, max_new_tokens=max_new_tokens, do_sample=False,
                                 pad_token_id=tokenizer.pad_token_id)
        for row in range(len(batch)):
            outputs.append(tokenizer.decode(out[row, encoded.input_ids.shape[1]:], skip_special_tokens=True))
    return outputs


def run_code(program_text, timeout=10):
    """Run program_text in a fresh Python process. True when it exits 0."""
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "program.py")
        with open(path, "w") as f:
            f.write(program_text)
        try:
            result = subprocess.run([sys.executable, "-I", path], cwd=tmp, capture_output=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            return False
    return result.returncode == 0


def run_programs(programs):
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=8) as pool:
        return list(pool.map(run_code, programs))


def score_generation(model, tokenizer, kind, items):
    if kind == "gsm8k":
        texts = []
        for item in items:
            texts.append("Question: " + item["question"] + "\nAnswer:")
        outputs = generate(model, tokenizer, texts, MAX_NEW_TOKENS["gsm8k"], True)
        correct = 0
        for item, output in zip(items, outputs):
            if gsm8k_correct(output, item["answer"]):
                correct += 1
        return correct / max(1, len(items))

    texts = []
    add_special = True
    for item in items:
        if kind == "math":
            content = "Solve the problem. Put the final answer in \\boxed{}.\n\n" + item["problem"]
        elif kind == "humaneval":
            content = ("Complete the following Python function. Reply with the full function in a ```python "
                       "code block.\n\n" + item["prompt"])
        else:
            content = mbpp_prompt(item)
        text, add_special = chat_text(tokenizer, content)
        texts.append(text)
    outputs = generate(model, tokenizer, texts, MAX_NEW_TOKENS[kind], add_special)
    if kind == "math":
        correct = 0
        for item, output in zip(items, outputs):
            if math_correct(output, item["answer"]):
                correct += 1
        return correct / max(1, len(items))
    programs = []
    for item, output in zip(items, outputs):
        if kind == "humaneval":
            code = extract_code(output, item["prompt"], item["entry_point"])
            programs.append(code + "\n\n" + item["test"] + "\n\ncheck(" + item["entry_point"] + ")\n")
        else:
            code = extract_code(output)
            programs.append(item["setup"] + "\n" + code + "\n\n" + "\n".join(item["tests"]) + "\n")
    passed = run_programs(programs)
    return sum(1 for ok in passed if ok) / max(1, len(items))


def limit_items(items, limit):
    """A fixed random subset of at most limit items, in their original order. None keeps all items."""
    if limit is None or limit >= len(items):
        return items
    picked = sorted(seeded_order(len(items), 17)[:limit])
    return [items[index] for index in picked]


def score_set(model, tokenizer, name, limit=None):
    """Load and score one set, or a fixed subset of at most limit items. Return (accuracy, item count, seconds)."""
    start = time.time()
    kind, items = load_set(name)
    items = limit_items(items, limit)
    if kind == "mcq":
        value = score_mcq(model, tokenizer, items)
    else:
        value = score_generation(model, tokenizer, kind, items)
    return value, len(items), round(time.time() - start, 1)


def decode_tok_s(model, tokenizer, new_tokens=128):
    """Greedy decode speed at batch 1, after a short warmup."""
    import torch

    prepare_tokenizer(tokenizer)
    encoded = tokenizer(["Question: What is 12 times 7?\nAnswer:"], return_tensors="pt").to(model.device)
    with torch.no_grad():
        model.generate(**encoded, max_new_tokens=8, do_sample=False, pad_token_id=tokenizer.pad_token_id)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start = time.time()
        model.generate(**encoded, max_new_tokens=new_tokens, min_new_tokens=new_tokens, do_sample=False,
                       pad_token_id=tokenizer.pad_token_id)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    return round(new_tokens / (time.time() - start), 1)


def count_params(model):
    """Total parameters of the model plus any tap donors, in billions."""
    seen = set()
    total = 0
    for param in model.parameters():
        if id(param) in seen:
            continue
        seen.add(id(param))
        total += param.numel()
    donors = getattr(model, "wsp_tap_donors", None) or {}
    for donor_id in donors:
        donor = donors[donor_id][0]
        for param in donor.parameters():
            if id(param) in seen:
                continue
            seen.add(id(param))
            total += param.numel()
    return total / 1e9
